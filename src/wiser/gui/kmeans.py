import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
from astropy import units as u
from sklearn.cluster import KMeans as SklearnKMeans
from PySide2.QtCore import QObject, Qt, Signal, Slot
from PySide2.QtGui import QIntValidator, QDoubleValidator
from PySide2.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.kmeans_dialog_ui import Ui_KMeansDialog
from wiser.gui.run_history import RunHistoryManagerBase
from wiser.gui.spectra_table import SpectraTableController
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataMeta,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    ExternalRasterHandle,
    NoChunkingScheme,
    PriorityClass,
)
from wiser.utils.task_stage_utils import (
    get_good_band_runs,
    split_dataset_tile_by_good_band_runs,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    BasePlanMeta,
    DatasetPlanMeta,
    ResourceModel,
    SemanticTask,
    SequentialStage,
    WriteSpec,
)
from wiser.utils.worker_runtime import get_process_storage_client

from wiser.raster.spectrum import NumPyArraySpectrum, Spectrum
from wiser.raster.utils import (
    convert_spectrum_wavelengths,
    finite_unmasked_row_mask,
    get_band_values,
)

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet

_KMEANS_DATA_IGNORE = -1


class KMeansInitMethod(Enum):
    KMEANS_PLUS_PLUS = "k-means++"
    RANDOM = "random"
    MANUAL = "manual"


class KMeansAlgorithm(Enum):
    LLOYD = "lloyd"
    ELKAN = "elkan"


@dataclass(frozen=True)
class KMeansParameters:
    """K-means configuration keyed by dataset and all algorithm options.

    ``dataset_id`` identifies which WISER dataset these parameters apply to and
    is used (together with every other field) as a dict key when storing results
    in application state.

    ``_manual_spectra`` is excluded from the auto-generated ``__eq__``/``__hash__``
    (``compare=False``) because numpy arrays are not directly hashable.  Custom
    ``__eq__`` and ``__hash__`` implementations below include them via
    element-wise comparison and ``ndarray.tobytes()``.
    """

    dataset_id: int
    k: int
    init_method: KMeansInitMethod
    num_inits: Optional[int]
    max_iter: Optional[int]
    tol: Optional[float]
    seed: Optional[int]
    algorithm: KMeansAlgorithm
    _manual_spectra: Optional[Sequence[np.ndarray]] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KMeansParameters):
            return NotImplemented
        if (
            self.dataset_id,
            self.k,
            self.init_method,
            self.num_inits,
            self.max_iter,
            self.tol,
            self.seed,
            self.algorithm,
        ) != (
            other.dataset_id,
            other.k,
            other.init_method,
            other.num_inits,
            other.max_iter,
            other.tol,
            other.seed,
            other.algorithm,
        ):
            return False
        if self._manual_spectra is None and other._manual_spectra is None:
            return True
        if self._manual_spectra is None or other._manual_spectra is None:
            return False
        if len(self._manual_spectra) != len(other._manual_spectra):
            return False
        return all(np.array_equal(a, b) for a, b in zip(self._manual_spectra, other._manual_spectra))

    def __hash__(self) -> int:
        spectra_key: Optional[tuple] = None
        if self._manual_spectra is not None:
            spectra_key = tuple(
                (np.asarray(arr).tobytes(), np.asarray(arr).shape, str(np.asarray(arr).dtype))
                for arr in self._manual_spectra
            )
        return hash(
            (
                self.dataset_id,
                self.k,
                self.init_method,
                self.num_inits,
                self.max_iter,
                self.tol,
                self.seed,
                self.algorithm,
                spectra_key,
            )
        )

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_dataset_id(self) -> int:
        return self.dataset_id

    def get_k(self) -> int:
        return self.k

    def get_init_method(self) -> KMeansInitMethod:
        return self.init_method

    def get_num_inits(self) -> Optional[int]:
        return self.num_inits

    def get_max_iter(self) -> Optional[int]:
        return self.max_iter

    def get_tol(self) -> Optional[float]:
        return self.tol

    def get_seed(self) -> Optional[int]:
        return self.seed

    def get_algorithm(self) -> KMeansAlgorithm:
        return self.algorithm

    def get_manual_spectra(self) -> Optional[Sequence[np.ndarray]]:
        """Return initial spectra when :attr:`init_method` is ``MANUAL``, else ``None``."""
        return self._manual_spectra


class KMeansCentroids:
    """Convenience wrapper around a (k, b) float32 centroid array."""

    def __init__(self, centroids: np.ndarray) -> None:
        if centroids.ndim != 2:
            raise ValueError(f"Expected 2D centroid array of shape (k, b), got {centroids.shape}")
        self._centroids = np.asarray(centroids, dtype=np.float32)

    def get_centroid(self, index: int) -> np.ndarray:
        """Return the 1-D spectrum for cluster *index* (shape ``(b,)``)."""
        return self._centroids[index]

    def num_centroids(self) -> int:
        """Return k, the number of cluster centroids."""
        return self._centroids.shape[0]


# ---------------------------------------------------------------------------
# Run history (application state) — record + manager
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KMeansRunRecord:
    """Immutable record of one completed K-Means run.

    Mirrors the run records used by PCA, MNF, and linear unmixing so that all
    four past-run histories share the :class:`RunHistoryManagerBase`
    infrastructure.  Unlike the legacy ``KMeansParameters``-keyed dict, every
    completed run is its own record: re-running with identical settings (or an
    unseeded run that draws a fresh random state) produces a second,
    independently viewable and deletable entry rather than silently
    overwriting the first.

    The record snapshots its own small, self-contained payload — the
    ``centroids`` array and the resolved ``effective_seed`` — and only
    *references* the heavy input cube by ``input_dataset_id``.  If that dataset
    is later closed, the run moves into the past-runs viewer's "closed" section
    instead of being lost.

    ``params`` carries the full run configuration, including any manual-init
    seed spectra (already captured as plain ``numpy`` arrays on
    ``KMeansParameters._manual_spectra``), so the record stays dependency-free
    on the live spectrum/dataset layer.

    ``effective_seed`` is the integer random state actually handed to
    scikit-learn.  When the user leaves the seed blank it is drawn explicitly
    before the run (rather than letting sklearn pull from numpy's global state)
    so that even unseeded runs are exactly reproducible.
    """

    run_id: int
    timestamp: datetime.datetime
    input_dataset_id: int
    # Snapshotted name so the table can still show something meaningful after
    # the live input dataset is closed.
    input_dataset_name_snapshot: str
    params: KMeansParameters
    centroids: KMeansCentroids
    effective_seed: Optional[int]


class KMeansRunHistoryManager(RunHistoryManagerBase[KMeansRunRecord]):
    """Owns the in-memory list of completed K-Means runs.

    Lives on :class:`~wiser.gui.app_state.ApplicationState` so the history
    persists across closing and reopening the K-Means dialog.  K-Means tracks
    only input-dataset liveness (its centroids are snapshotted into the
    record), so the shared base needs no overrides.
    """


# region KMeans TaskStage


def _write_kmeans_labels_meta(
    input_ref: DataRef,
    full_input_region: DataRegion,
    labels_write: "WriteSpec",
) -> None:
    _ = full_input_region
    client = get_process_storage_client()
    input_meta = client.get_meta(input_ref)
    output_meta = client.get_meta(labels_write.ref)
    labels_meta = replace(
        output_meta,
        crs_wkt=input_meta.crs_wkt,
        geotransform=input_meta.geotransform,
    )
    client.write_meta(labels_write.ref, labels_meta)


def _run_kmeans(
    input_ref: DataRef,
    input_region: DataRegion,
    labels_write: "WriteSpec",
    centroids_ref: DataRef,
    params: "KMeansParameters",
) -> None:
    client = get_process_storage_client()
    _ = input_region

    # Read the full dataset as float32 (y, x, b_total)
    image_data, region_meta = client.read_data(input_ref, filter_data=False)
    image_array = np.asarray(np.ma.getdata(image_data), dtype=np.float32)

    if image_array.ndim != 3:
        raise ValueError(f"Expected dataset shape [y][x][b], got {image_array.shape}")

    y, x, b_total = image_array.shape

    # Remove bad bands: extract only good-band runs and concatenate into (y, x, b_good)
    if region_meta.bad_bands is None:
        good_band_runs = [(0, b_total)]
    else:
        good_band_runs = get_good_band_runs(np.asarray(region_meta.bad_bands))

    if len(good_band_runs) == 0:
        raise ValueError("KMeans requires at least one valid band; all bands are flagged as bad.")

    good_chunks = split_dataset_tile_by_good_band_runs(image_array, good_band_runs)
    image_good = np.concatenate(good_chunks, axis=2)  # (y, x, b_good)
    b_good = image_good.shape[2]

    # Flatten to (n_pixels, b_good) while preserving the flat index -> (y, x) mapping
    flat = image_good.reshape(y * x, b_good)

    # A pixel is unusable if it holds any non-finite value (NaN/Inf) in a good
    # band, or it hits the nodata sentinel. Both kinds of row are dropped before
    # clustering and get the data-ignore label in the output -- the same approach
    # PCA/MNF/decorrelation use to stay NaN-resistant, instead of erroring out.
    # finite_unmasked_row_mask is the shared row-validity helper: it drops the
    # NaN/Inf rows (which also covers the nodata-is-NaN case) and AND-s out the
    # finite-sentinel nodata rows passed via its optional mask.
    nodata = region_meta.nodata
    nodata_mask = None
    if nodata is not None and not np.isnan(nodata):
        nodata_mask = flat == nodata
    valid_indices = np.where(finite_unmasked_row_mask(flat, nodata_mask))[0]

    flat_valid = flat[valid_indices]  # (n_valid, b_good)

    # Defensive sanity check: after dropping non-finite/nodata rows, nothing
    # non-finite should remain. If it does, the cleaning above missed something.
    if not np.all(np.isfinite(flat_valid)):
        raise ValueError(
            "KMeans input still contains NaN or infinite values after removing "
            "nodata and NaN/Inf pixels. Check your dataset for corrupt values."
        )

    k = params.get_k()
    if flat_valid.shape[0] < k:
        raise ValueError(
            f"KMeans needs at least k={k} valid (finite, non-nodata) pixels, but only "
            f"{flat_valid.shape[0]} remain after removing nodata and NaN/Inf pixels."
        )

    # Build the init argument
    init_method = params.get_init_method()
    if init_method == KMeansInitMethod.MANUAL:
        manual_spectra = params.get_manual_spectra()
        if manual_spectra is None or len(manual_spectra) == 0:
            raise ValueError("KMeansInitMethod.MANUAL requires manual_spectra to be provided.")
        # Each manual spectrum is assumed to be full-band; strip to good bands
        init_arg = np.stack(
            [
                np.concatenate([np.asarray(s, dtype=np.float32)[start:end] for start, end in good_band_runs])
                for s in manual_spectra
            ],
            axis=0,
        )  # (k, b_good)
        n_init = 1
    else:
        init_arg = init_method.value  # "k-means++" or "random"
        n_init = params.get_num_inits() if params.get_num_inits() is not None else 10

    kmeans = SklearnKMeans(
        n_clusters=params.get_k(),
        init=init_arg,
        n_init=n_init,
        max_iter=params.get_max_iter() if params.get_max_iter() is not None else 300,
        tol=params.get_tol() if params.get_tol() is not None else 1e-4,
        random_state=params.get_seed(),
        algorithm=params.get_algorithm().value,
    )

    labels_valid = kmeans.fit_predict(flat_valid)  # (n_valid,) int64

    # Scatter labels back to (y*x,), filling nodata pixels with -1
    labels_flat = np.full((y * x,), fill_value=_KMEANS_DATA_IGNORE, dtype=np.float32)
    labels_flat[valid_indices] = labels_valid.astype(np.float32)
    labels_image = labels_flat.reshape(y, x, 1)  # (y, x, 1)

    # Write labels
    assert labels_write.region is not None, "labels WriteSpec must have a non-None region"
    labels_write.region.validate_array_shape(labels_image)
    client.write_spec(labels_write, labels_image)

    # Expand centroids from (k, b_good) back to (k, b_total) by scattering good bands
    # back to their original positions; bad-band columns remain zero.
    centroids_compact = kmeans.cluster_centers_.astype(np.float32)  # (k, b_good)
    centroids_full = np.full((params.get_k(), b_total), fill_value=np.nan, dtype=np.float32)
    band_offset = 0
    for start, end in good_band_runs:
        run_length = end - start
        centroids_full[:, start:end] = centroids_compact[:, band_offset : band_offset + run_length]
        band_offset += run_length

    client.write_data(centroids_ref, centroids_full)


@dataclass
class KMeansStage(SequentialStage):
    """
    Full-image k-means clustering stage.

    Runs on the whole dataset at once (NoChunkingScheme) and allocates two outputs:
      - a (y, x, 1) int32 dataset holding the per-pixel cluster label
      - a (k, b) float32 array holding the k cluster centroids (bad-band columns are zero)
    """

    _labels_ref_name: str = "kmeans_labels"
    _centroids_ref_name: str = "kmeans_centroids"
    _params: Optional[KMeansParameters] = None
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = NoChunkingScheme

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [
            DataBinding(self._labels_ref_name, kind="dataset"),
            DataBinding(self._centroids_ref_name, kind="array"),
        ]
        # Centroids has no spatial region; expose it as a DataRef via broadcast_input
        self.broadcast_input |= {
            "centroids_ref": DataBinding(self._centroids_ref_name),
        }

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("KMeansStage requires a DatasetRegionRef input region")
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=1,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for KMeansStage"
        assert self._params is not None, "KMeansStage requires _params to be set"

        y = input_meta.height
        x = input_meta.width
        b = input_meta.bands
        k = self._params.get_k()

        labels_size_est = y * x * 1 * np.dtype(np.float32).itemsize
        centroids_size_est = k * b * np.dtype(np.float32).itemsize

        return [
            AllocationRequest(
                name=self._labels_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=labels_size_est,
                shape=(y, x, 1),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._labels_ref_name),
            ),
            AllocationRequest(
                name=self._centroids_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=centroids_size_est,
                shape=(k, b),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._centroids_ref_name),
            ),
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        labels_write = output_writes[self._labels_ref_name]
        centroids_ref: DataRef = broadcast_inputs["centroids_ref"]
        return partial(_run_kmeans, input_ref, input_region, labels_write, centroids_ref, self._params)

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        labels_write = output_writes[self._labels_ref_name]
        return partial(_write_kmeans_labels_meta, input_ref, full_input_region, labels_write)


def get_kmeans_stage(
    dataset_ref: DataRef,
    params: KMeansParameters,
    labels_ref_name: str = "kmeans_labels",
    centroids_ref_name: str = "kmeans_centroids",
) -> KMeansStage:
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    input_plan_meta = DatasetPlanMeta(
        shape=dataset_meta.shape,
        dtype=np.dtype(dataset_meta.elem_type),
    )
    return KMeansStage(
        _labels_ref_name=labels_ref_name,
        _centroids_ref_name=centroids_ref_name,
        _params=params,
        default_executor="process",
        input_plan_meta=input_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=NoChunkingScheme,
    )


def get_kmeans_pipeline(
    dataset_ref: DataRef,
    params: KMeansParameters,
    labels_ref_name: str = "kmeans_labels",
    centroids_ref_name: str = "kmeans_centroids",
) -> AlgorithmPipeline:
    return AlgorithmPipeline([get_kmeans_stage(dataset_ref, params, labels_ref_name, centroids_ref_name)])


# endregion


class KMeansSemanticTask(QObject, SemanticTask):
    """Semantic task that runs K-means clustering and loads the label image into WISER."""

    result_ready = Signal(object, object, object)

    def __init__(
        self,
        app_state: "ApplicationState",
        source_dataset: "RasterDataSet",
        input_ref: DataRef,
        params: KMeansParameters,
        labels_ref_name: str = "kmeans_labels",
        centroids_ref_name: str = "kmeans_centroids",
    ):
        QObject.__init__(self)
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=get_kmeans_pipeline(input_ref, params, labels_ref_name, centroids_ref_name),
            task_title="K-Means Clustering",
            task_variables={
                "K": params.get_k(),
                "Dataset": source_dataset.get_name(),
                "Init Method": params.get_init_method().value,
            },
        )
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._params = params
        self._labels_ref_name = labels_ref_name
        self._centroids_ref_name = centroids_ref_name
        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        labels_ref = bindings.get(self._labels_ref_name)
        if labels_ref is None:
            raise KeyError(f"Missing K-Means labels output binding: {self._labels_ref_name}")
        centroids_ref = bindings.get(self._centroids_ref_name)
        if centroids_ref is None:
            raise KeyError(f"Missing K-Means centroids output binding: {self._centroids_ref_name}")

        storage_client = get_process_storage_client()

        labels_meta = storage_client.get_meta(labels_ref)
        height, width, bands = labels_meta.shape
        labels_region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=bands)
        labels_data, _ = storage_client.read_region(labels_ref, labels_region, filter_data=False)

        centroids_data, _ = storage_client.read_data(centroids_ref)

        self.result_ready.emit(np.asarray(labels_data), labels_meta, np.asarray(centroids_data))

    @Slot(object, object, object)
    def _load_result_into_wiser(
        self, labels_data: object, labels_meta: object, centroids_data: object
    ) -> None:
        centroids_obj = KMeansCentroids(np.asarray(centroids_data))
        self._app_state.add_kmeans_centroids(self._params, centroids_obj)

        labels_array = np.asarray(labels_data)  # (y, x, 1)
        labels_by_band = labels_array.transpose(2, 0, 1)  # (1, y, x)

        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        labels_dataset = loader.dataset_from_numpy_array(labels_by_band, cache)

        source_name = self._source_dataset.get_name() or "Dataset"
        timestamp = datetime.datetime.now().isoformat()
        k = self._params.get_k()
        labels_dataset.set_name(self._app_state.unique_dataset_name(f"K-Means Labels (k={k}): {source_name}"))
        labels_dataset.set_description(f"K-Means cluster label image (k={k}): {source_name} ({timestamp})")
        labels_dataset.set_data_ignore_value(_KMEANS_DATA_IGNORE)

        if self._source_dataset.get_spatial_metadata().get_spatial_ref():
            labels_dataset.copy_spatial_metadata(self._source_dataset.get_spatial_metadata())

        self._app_state.add_dataset(labels_dataset, view_dataset=False)


class KMeansCentroidsDialog(QDialog):
    """Lists all stored K-Means centroid results.

    Each entry is shown as a button whose label summarises the
    ``KMeansParameters`` used to produce it (using the dataset *name* rather
    than the raw ID).  Clicking a button opens a :class:`SpectrumPlotGeneric`
    that displays every centroid spectrum named ``"{i}-centroid"``.
    """

    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        self.setWindowTitle("K-Means Centroids")
        self.resize(620, 400)
        self._scroll: Optional[QScrollArea] = None
        self._build_ui()
        app_state.kmeans_centroids_changed.connect(self._rebuild_content)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll)
        self._rebuild_content()

    def _rebuild_content(self) -> None:
        """Rebuild the scrollable list of centroid buttons from current app state."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)

        all_centroids = self._app_state.get_all_kmeans_centroids()
        if not all_centroids:
            layout.addWidget(QLabel("No K-Means centroids have been stored yet."))
        else:
            for params, centroids in all_centroids.items():
                btn_text = self._format_params(params)
                btn = QPushButton(btn_text)
                btn.setToolTip(btn_text)
                btn.clicked.connect(lambda checked=False, p=params, c=centroids: self._show_centroids(p, c))
                layout.addWidget(btn)

        self._scroll.setWidget(container)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_params(self, params: KMeansParameters) -> str:
        """Build a human-readable summary of *params*, using the dataset name."""
        try:
            ds = self._app_state.get_dataset(params.get_dataset_id())
            ds_name = ds.get_name() or f"id={params.get_dataset_id()}"
        except KeyError:
            ds_name = f"id={params.get_dataset_id()}"

        parts = [
            f"Dataset: {ds_name}",
            f"k={params.get_k()}",
            f"init={params.get_init_method().value}",
        ]
        if params.get_num_inits() is not None:
            parts.append(f"n_init={params.get_num_inits()}")
        if params.get_max_iter() is not None:
            parts.append(f"max_iter={params.get_max_iter()}")
        if params.get_tol() is not None:
            parts.append(f"tol={params.get_tol()}")
        if params.get_seed() is not None:
            parts.append(f"seed={params.get_seed()}")
        parts.append(f"algo={params.get_algorithm().value}")
        if params.get_manual_spectra() is not None:
            parts.append("init_spectra=manual")
        return " | ".join(parts)

    def _show_centroids(self, params: KMeansParameters, centroids: KMeansCentroids) -> None:
        """Open a SpectrumPlotGeneric showing every centroid in *centroids*."""
        spectra = [
            NumPyArraySpectrum(
                centroids.get_centroid(i),
                name=f"{i}-centroid",
                source_name=self._format_params(params),
            )
            for i in range(centroids.num_centroids())
        ]
        title = f"K-Means Centroids — {self._format_params(params)}"
        self._app_state.show_spectra_in_plot(spectra, plot_title=title, parent=self)


class KMeansDialog(QDialog):
    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._selected_dataset_id: Optional[int] = None
        self._centroids_dialog: Optional[KMeansCentroidsDialog] = None

        self._ui = Ui_KMeansDialog()
        self._ui.setupUi(self)

        self._ui.wdgt_advanced_options.setVisible(False)
        self._ui.btn_advanced_options.setText("Advanced Options \u25b6")
        self._ui.btn_advanced_options.clicked.connect(self._toggle_advanced_options)
        self._ui.btn_view_centroids.clicked.connect(self._on_view_centroids)

        self._init_cbox_init_method()
        self._init_cbox_algo()
        self._init_validators()
        self._ui.cbox_init_method.currentIndexChanged.connect(self._on_init_method_changed)

        self._spectra_table = SpectraTableController(
            self._ui.tbl_wdgt_init_spectra,
            app_state,
            self,
            name_column_label=self.tr("Spectrum"),
            remove_tooltip=self.tr("Remove init spectrum"),
            add_filter=self._filter_spectra_by_band_count,
        )
        self._ui.btn_add_collected_spec.clicked.connect(self._spectra_table.on_add_collected_clicked)
        self._ui.btn_import_spec.clicked.connect(self._spectra_table.on_import_clicked)

        app_state.dataset_added.connect(self._on_datasets_changed)
        app_state.dataset_removed.connect(self._on_datasets_changed)

    def _init_cbox_init_method(self) -> None:
        cbox = self._ui.cbox_init_method
        cbox.clear()
        for method in KMeansInitMethod:
            cbox.addItem(method.value, method)
        # Manual-init widgets start hidden; shown only when init method is manual.
        self._set_manual_init_widgets_visible(False)

    def _init_cbox_algo(self) -> None:
        cbox = self._ui.cbox_algo
        cbox.clear()
        for algo in KMeansAlgorithm:
            cbox.addItem(algo.value, algo)

    def _init_validators(self) -> None:
        # Positive integers only (minimum 1)
        self._ui.ledit_k_clusters.setValidator(QIntValidator(1, 2_147_483_647, self))
        self._ui.ledit_num_inits.setValidator(QIntValidator(1, 2_147_483_647, self))
        self._ui.ledit_max_iter.setValidator(QIntValidator(1, 2_147_483_647, self))

        # Any integer (positive or negative)
        self._ui.ledit_seed.setValidator(QIntValidator(-2_147_483_648, 2_147_483_647, self))

        # Positive float
        pos_float_validator = QDoubleValidator(0.0, 1.0e308, 10, self)
        pos_float_validator.setNotation(QDoubleValidator.ScientificNotation)
        self._ui.ledit_tol.setValidator(pos_float_validator)

    def _set_manual_init_widgets_visible(self, visible: bool) -> None:
        """Show or hide the manual-init spectra widgets as a group.

        Qt layouts and spacer items have no ``setVisible``; to keep the "Add
        Collected Spectrum"/"Import Spectrum" row from leaving an empty gap when
        the init method isn't manual, we hide the two buttons and collapse the
        expanding spacer between them, then invalidate the row's layout so it
        recomputes its geometry.
        """
        self._ui.tbl_wdgt_init_spectra.setVisible(visible)
        self._ui.btn_add_collected_spec.setVisible(visible)
        self._ui.btn_import_spec.setVisible(visible)
        if visible:
            self._ui.hspacer_add_spec.changeSize(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        else:
            self._ui.hspacer_add_spec.changeSize(0, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._ui.hlayout_add_spec.invalidate()

    def _filter_spectra_by_band_count(self, specs: List[Spectrum]) -> List[Spectrum]:
        """Drop spectra whose band count doesn't match the selected dataset.

        Used as the SpectraTableController's ``add_filter``, so it runs over the
        whole batch before any rows are created.  When no dataset is selected we
        can't validate, so every spectrum is accepted — the run-time check in
        :meth:`perform_kmeans` is the backstop.  Mismatched spectra are skipped
        and reported together in a single warning.
        """
        dataset = self.get_selected_dataset()
        if dataset is None:
            return specs

        expected = dataset.num_bands()
        matching: List[Spectrum] = []
        mismatched: List[Spectrum] = []
        for spec in specs:
            (matching if spec.num_bands() == expected else mismatched).append(spec)

        if mismatched:
            details = "\n".join(
                self.tr("• {0}: {1} bands").format(spec.get_name() or self.tr("<unnamed>"), spec.num_bands())
                for spec in mismatched
            )
            QMessageBox.warning(
                self,
                self.tr("Band Count Mismatch"),
                self.tr(
                    "These spectra were skipped because their band count does not "
                    "match the input dataset ({0} bands):\n\n{1}"
                ).format(expected, details),
            )
        return matching

    def _on_init_method_changed(self, index: int) -> None:
        method = self._ui.cbox_init_method.itemData(index)
        is_manual = method is KMeansInitMethod.MANUAL

        # Manual-init spectra entry (table + add/import buttons + spacer) only
        # applies when the user supplies the initial centroids by hand.
        self._set_manual_init_widgets_visible(is_manual)

        # Disable num_inits and seed when manual (centroid positions are fixed)
        self._ui.ledit_num_inits.setEnabled(not is_manual)
        self._ui.lbl_num_inits.setEnabled(not is_manual)
        self._ui.ledit_seed.setEnabled(not is_manual)
        self._ui.lbl_seed.setEnabled(not is_manual)

    def _toggle_advanced_options(self) -> None:
        visible = not self._ui.wdgt_advanced_options.isVisible()
        self._ui.wdgt_advanced_options.setVisible(visible)
        arrow = "\u25bc" if visible else "\u25b6"
        self._ui.btn_advanced_options.setText(f"Advanced Options {arrow}")

    def get_k_clusters(self) -> Optional[int]:
        text = self._ui.ledit_k_clusters.text().strip()
        return int(text) if text else None

    def get_init_method(self) -> KMeansInitMethod:
        return self._ui.cbox_init_method.currentData()

    def get_num_inits(self) -> Optional[int]:
        text = self._ui.ledit_num_inits.text().strip()
        return int(text) if text else None

    def get_max_iter(self) -> Optional[int]:
        text = self._ui.ledit_max_iter.text().strip()
        return int(text) if text else None

    def get_tol(self) -> Optional[float]:
        text = self._ui.ledit_tol.text().strip()
        return float(text) if text else None

    def get_seed(self) -> Optional[int]:
        text = self._ui.ledit_seed.text().strip()
        return int(text) if text else None

    def get_algorithm(self) -> KMeansAlgorithm:
        return self._ui.cbox_algo.currentData()

    def _on_view_centroids(self) -> None:
        if self._centroids_dialog is None or not self._centroids_dialog.isVisible():
            self._centroids_dialog = KMeansCentroidsDialog(self._app_state, parent=self)
        self._centroids_dialog.show()
        self._centroids_dialog.raise_()
        self._centroids_dialog.activateWindow()

    def _on_datasets_changed(self, *args) -> None:
        """Refresh the input-dataset combo while preserving the current selection."""
        current_id = self._ui.cbox_input_dataset.currentData()
        preserve_id = current_id if (current_id is not None and int(current_id) >= 0) else None
        self.show_kmeans(dataset_id=preserve_id)

    def show_kmeans(self, dataset_id: Optional[int] = None) -> None:
        cbox = self._ui.cbox_input_dataset
        cbox.clear()
        cbox.addItem(self.tr("(no data)"), -1)
        for dataset in self._app_state.get_datasets():
            cbox.addItem(dataset.get_name(), dataset.get_id())

        if dataset_id is not None:
            index = cbox.findData(dataset_id)
            cbox.setCurrentIndex(index if index >= 0 else 0)
        else:
            cbox.setCurrentIndex(0)

    def showEvent(self, event):
        self.show_kmeans(dataset_id=self._selected_dataset_id)
        super().showEvent(event)

    def select_dataset(self, dataset_id: Optional[int]) -> None:
        self._selected_dataset_id = dataset_id
        self.show_kmeans(dataset_id=dataset_id)

    def get_selected_dataset(self):
        dataset_id = self._ui.cbox_input_dataset.currentData()
        if dataset_id is None or int(dataset_id) < 0:
            return None
        return self._app_state.get_dataset(int(dataset_id))

    def _collect_manual_init_spectra(self, dataset, k: int) -> List[np.ndarray]:
        """Return the staged manual-init centroids as full-band arrays.

        Re-validates the table against this run (the add-time filter only checks
        band counts when a dataset was selected, and the selection or table may
        have changed since): the number of spectra must equal ``k`` because
        scikit-learn requires ``init.shape[0] == n_clusters``, and every spectrum
        must match the dataset's band count because ``_run_kmeans`` treats each
        as a full-band centroid.  When the dataset carries wavelengths, each
        spectrum must also sit on the dataset's exact wavelength grid once its
        own wavelengths are cast into the dataset's units (see
        :meth:`_validate_spectra_on_dataset_grid`); when it doesn't, the user is
        warned and bands are matched positionally.

        Raises:
            ValueError: If the count differs from ``k``, a band count mismatches,
                or a spectrum is off the dataset's wavelength grid.  Surfaced to
                the user as a warning by :meth:`accept`.
            KeyError: If a staged collected spectrum/library has since been
                removed from app state (propagated from ``collect_spectra``).
        """
        specs = self._spectra_table.collect_spectra()
        if len(specs) != k:
            raise ValueError(
                self.tr(
                    "Manual initialization requires exactly k={0} spectra, but {1} are staged. "
                    "Add or remove spectra so the count matches k."
                ).format(k, len(specs))
            )

        expected_bands = dataset.num_bands()
        mismatched = [s for s in specs if s.num_bands() != expected_bands]
        if mismatched:
            details = "\n".join(
                self.tr("• {0}: {1} bands").format(s.get_name() or self.tr("<unnamed>"), s.num_bands())
                for s in mismatched
            )
            raise ValueError(
                self.tr(
                    "These manual spectra don't match the input dataset's band count " "({0} bands):\n\n{1}"
                ).format(expected_bands, details)
            )

        # Wavelength-unit handling: when the dataset exposes a wavelength unit,
        # each manual spectrum must already sit on the dataset's grid once its
        # own wavelengths are cast into that unit (no interpolation — we never
        # silently alter centroid values). Without a usable unit we can't align
        # by wavelength, so warn and fall back to matching bands by position.
        # (has_wavelengths() only checks that the band-info key is present;
        # get_band_unit() can still be None, so gate on the unit itself.)
        target_unit = dataset.get_band_unit() if dataset.has_wavelengths() else None
        if target_unit is not None:
            self._validate_spectra_on_dataset_grid(dataset, specs, target_unit)
        else:
            QMessageBox.warning(
                self,
                self.tr("K-Means"),
                self.tr(
                    "The input dataset has no wavelength units, so K-Means will match the "
                    "manual spectra to bands by position without casting units."
                ),
            )

        return [np.asarray(s.get_spectrum(), dtype=np.float32) for s in specs]

    def _validate_spectra_on_dataset_grid(self, dataset, specs: List[Spectrum], target_unit: u.Unit) -> None:
        """Require each manual spectrum to lie on the dataset's exact wavelength grid.

        Each spectrum's wavelengths are converted into ``target_unit`` (the
        dataset's wavelength unit) and compared element-wise against the
        dataset's grid (band counts are already known to match).  Spectra with
        no wavelengths, units that can't convert, or a differing grid are
        collected and reported together in one ``ValueError`` (surfaced as a
        warning by :meth:`accept`).  Nothing is resampled — centroid values are
        used exactly as entered.
        """
        target_wvls = np.asarray(get_band_values(dataset.get_wavelengths(), target_unit), dtype=np.float64)

        problems = []  # list[tuple[Spectrum, str]]
        for s in specs:
            try:
                spec_wvls = convert_spectrum_wavelengths(s, target_unit)
            except ValueError:
                problems.append((s, self.tr("has no wavelengths to verify against the dataset's grid")))
                continue
            except u.UnitConversionError:
                problems.append(
                    (
                        s,
                        self.tr("wavelength units are not convertible to the dataset's units ({0})").format(
                            target_unit
                        ),
                    )
                )
                continue
            if not np.allclose(spec_wvls, target_wvls, rtol=0.0, atol=1e-9):
                problems.append((s, self.tr("wavelengths do not match the dataset's grid")))

        if problems:
            details = "\n".join(
                self.tr("• {0}: {1}").format(s.get_name() or self.tr("<unnamed>"), reason)
                for s, reason in problems
            )
            raise ValueError(
                self.tr(
                    "These manual spectra are not on the input dataset's wavelength grid "
                    "(in {0}). Re-collect or re-import them so they match the dataset:\n\n{1}"
                ).format(target_unit, details)
            )

    def perform_kmeans(self):
        selected_dataset = self.get_selected_dataset()
        if selected_dataset is None:
            raise ValueError("No dataset selected for K-Means clustering")

        k = self.get_k_clusters()
        if k is None:
            raise ValueError("K (number of clusters) must be specified")

        init_method = self.get_init_method()
        # For manual init, gather the staged spectra as full-band centroids and
        # validate them against this run. Other init methods don't use them.
        manual_spectra = (
            self._collect_manual_init_spectra(selected_dataset, k)
            if init_method == KMeansInitMethod.MANUAL
            else None
        )

        params = KMeansParameters(
            dataset_id=selected_dataset.get_id(),
            k=k,
            init_method=init_method,
            num_inits=self.get_num_inits(),
            max_iter=self.get_max_iter(),
            tol=self.get_tol(),
            seed=self.get_seed(),
            algorithm=self.get_algorithm(),
            _manual_spectra=manual_spectra,
        )

        dataset_ref = self._app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=selected_dataset)
        )

        kmeans_task = KMeansSemanticTask(
            app_state=self._app_state,
            source_dataset=selected_dataset,
            input_ref=dataset_ref,
            params=params,
        )

        task_plan = self._app_services.task_planner.plan_semantic_task(kmeans_task)
        future = self._app_services.task_manager.register_and_submit_task_plan(
            self._app_services.scheduler, task_plan
        )
        return future

    def accept(self):
        try:
            self.perform_kmeans()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, self.tr("K-Means"), str(exc))
            return
        QMessageBox.information(self, self.tr("K-Means"), self.tr("K-Means is running in the background."))

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from PySide2.QtGui import QIntValidator, QDoubleValidator
from PySide2.QtWidgets import QDialog

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.kmeans_dialog_ui import Ui_KMeansDialog
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    NoChunkingScheme,
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
    SequentialStage,
    WriteSpec,
)
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    pass


class KMeansInitMethod(Enum):
    KMEANS_PLUS_PLUS = "k-means++"
    RANDOM = "random"
    MANUAL = "manual"


class KMeansAlgorithm(Enum):
    LLOYD = "lloyd"
    ELKAN = "elkan"


@dataclass(frozen=True)
class KMeansParameters:
    """K-means configuration (mirrors the k-means dialog, plus optional manual-initialization spectra)."""

    k: int
    init_method: KMeansInitMethod
    num_inits: Optional[int]
    max_iter: Optional[int]
    tol: Optional[float]
    seed: Optional[int]
    algorithm: KMeansAlgorithm
    _manual_spectra: Optional[Sequence[np.ndarray]] = field(default=None, repr=False, compare=False)

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
        """
        Return initial spectra when :attr:`init_method` is
        :attr:`KMeansInitMethod.MANUAL`, else usually ``None``.
        """
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

    # Remove rows that contain the data ignore value
    nodata = region_meta.nodata
    if nodata is not None:
        if np.isnan(nodata):
            nodata_row_mask = np.any(np.isnan(flat), axis=1)
        else:
            nodata_row_mask = np.any(flat == nodata, axis=1)
        valid_indices = np.where(~nodata_row_mask)[0]
    else:
        valid_indices = np.arange(y * x)

    flat_valid = flat[valid_indices]  # (n_valid, b_good)

    # Error out on any remaining NaN or infinite values
    if not np.all(np.isfinite(flat_valid)):
        raise ValueError(
            "KMeans input contains NaN or infinite values in valid (non-nodata) pixels. "
            "Check your dataset for corrupt values."
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
    labels_flat = np.full((y * x,), fill_value=-1, dtype=np.int32)
    labels_flat[valid_indices] = labels_valid.astype(np.int32)
    labels_image = labels_flat.reshape(y, x, 1)  # (y, x, 1)

    # Write labels
    assert labels_write.region is not None, "labels WriteSpec must have a non-None region"
    labels_write.region.validate_array_shape(labels_image)
    client.write_spec(labels_write, labels_image)

    # Expand centroids from (k, b_good) back to (k, b_total) by scattering good bands
    # back to their original positions; bad-band columns remain zero.
    centroids_compact = kmeans.cluster_centers_.astype(np.float32)  # (k, b_good)
    centroids_full = np.zeros((params.get_k(), b_total), dtype=np.float32)
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
        _ = input_region
        return None

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

        labels_size_est = y * x * 1 * np.dtype(np.int32).itemsize
        centroids_size_est = k * b * np.dtype(np.float32).itemsize

        return [
            AllocationRequest(
                name=self._labels_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=labels_size_est,
                shape=(y, x, 1),
                dtype=np.dtype(np.int32),
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

        self._ui = Ui_KMeansDialog()
        self._ui.setupUi(self)

        self._ui.wdgt_advanced_options.setVisible(False)
        self._ui.btn_advanced_options.setText("Advanced Options \u25b6")
        self._ui.btn_advanced_options.clicked.connect(self._toggle_advanced_options)

        self._init_cbox_init_method()
        self._init_cbox_algo()
        self._init_validators()
        self._ui.cbox_init_method.currentIndexChanged.connect(self._on_init_method_changed)

    def _init_cbox_init_method(self) -> None:
        cbox = self._ui.cbox_init_method
        cbox.clear()
        for method in KMeansInitMethod:
            cbox.addItem(method.value, method)
        self._ui.tbl_wdgt_init_spectra.setVisible(False)

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

    def _on_init_method_changed(self, index: int) -> None:
        method = self._ui.cbox_init_method.itemData(index)
        is_manual = method is KMeansInitMethod.MANUAL

        self._ui.tbl_wdgt_init_spectra.setVisible(is_manual)

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

    def show_kmeans(self, dataset_id: Optional[int] = None) -> None:
        pass

    def showEvent(self, event):
        self.show_kmeans(dataset_id=self._selected_dataset_id)
        super().showEvent(event)

    def select_dataset(self, dataset_id: Optional[int]) -> None:
        self._selected_dataset_id = dataset_id
        self.show_kmeans(dataset_id=dataset_id)

    def accept(self):
        super().accept()

import inspect
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from sklearn.decomposition import IncrementalPCA, PCA
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from astropy import units as u

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset import RasterDataSet
from wiser.raster.spectrum import Spectrum
from wiser.utils.primitives import (
    DEFAULT_FLOAT_TYPE,
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    ExternalParams,
    NoChunkingScheme,
    SpectraBatchRef,
    SpectraListPlanMeta,
    SpatialTileScheme,
    SpectralBatchDatasetScheme,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    BasePlanMeta,
    DatasetPlanMeta,
    MapStage,
    ResourceModel,
    SequentialStage,
    WriteSpec,
)
from wiser.utils.worker_runtime import get_process_storage_client
from wiser.raster.utils import compute_PCA_on_image
from wiser.utils.numba_wrapper import convert_to_float32_if_needed

PCA_MEMORY_CUTOFF_BYTES = 4 * 1024**3
# Eigenvalues below max(λ) * this factor are treated as numerical null space and
# zeroed before persisting (avoids junk ~1e-15 tails after legitimate small EVs).
_EIGENDECOMP_RELATIVE_EIGENVALUE_CUTOFF = 1e-14
TotalLike = Union[int, DataRef]
NumComponentsLike = Union[int, DataRef]


class SpectralImageComputeFunction(Protocol):
    def __call__(
        self,
        target_image_arr: np.ndarray,
        target_wavelengths: np.ndarray,
        target_bad_bands: np.ndarray,
        min_wvl: np.float64,
        max_wvl: np.float64,
        reference_spectra: np.ndarray,
        reference_spectra_wvls: np.ndarray,
        reference_spectra_bad_bands: np.ndarray,
        reference_spectra_indices: np.ndarray,
        thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        ...


# region Task Stage utilities


def _run_save_external_dataset(
    input_ref: DataRef,
    save_params: Sequence[Any],
) -> None:
    """
    Reconstruct an external dataset in a worker process and save it to disk.

    Args:
        input_ref: External dataset ref previously registered with the storage service.
        save_params: Three-item sequence containing `(path, format, config)`.
    """
    if len(save_params) != 3:
        raise ValueError(f"save_params must contain [path, format, config], got {save_params!r}")

    path, format, config = save_params
    if not isinstance(path, str):
        raise TypeError(f"save path must be a string, got {type(path)}")
    if not isinstance(format, str):
        raise TypeError(f"save format must be a string, got {type(format)}")
    if not isinstance(config, dict):
        raise TypeError(f"save config must be a dict, got {type(config)}")

    client = get_process_storage_client()
    dataset = client.reconstruct_external_object(input_ref)
    if not isinstance(dataset, RasterDataSet):
        raise TypeError(f"Expected reconstruct_external_object to return RasterDataSet, got {type(dataset)}")

    loader = RasterDataLoader()
    loader.save_dataset_as(dataset, path, format, config)


def _prepare_general_spectral_image_inputs(
    input_ref: DataRef,
    reference_refs: Sequence[DataRef],
    min_wvl: u.Quantity,
    max_wvl: u.Quantity,
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    """
    Reconstruct a target dataset and reference spectra, then pack their values
    into NumPy arrays suitable for SAM/SFF image-mode kernels.
    """
    client = get_process_storage_client()
    target = client.reconstruct_external_object(input_ref)
    if not isinstance(target, RasterDataSet):
        raise TypeError(f"Expected RasterDataSet input, got {type(target)}")

    references: List[Spectrum] = []
    for reference_ref in reference_refs:
        reference = client.reconstruct_external_object(reference_ref)
        if not isinstance(reference, Spectrum):
            raise TypeError(f"Expected Spectrum reference, got {type(reference)}")
        references.append(reference)

    target_unit = target.get_band_unit()
    if target_unit is None:
        raise ValueError("Target dataset must have wavelength units for spectral image computation")

    target_image_cube = target.get_image_data()
    if isinstance(target_image_cube, np.ma.MaskedArray):
        target_image_arr = np.asarray(target_image_cube.data, dtype=np.float64)
    else:
        target_image_arr = np.asarray(target_image_cube, dtype=np.float64)
    if not target_image_arr.flags.c_contiguous:
        target_image_arr = np.ascontiguousarray(target_image_arr)

    target_wavelengths = np.array(
        [band_info["wavelength"].to(target_unit).value for band_info in target.get_band_info()],
        dtype=np.float64,
    )
    target_bad_bands_raw = target.get_bad_bands()
    if target_bad_bands_raw is None:
        target_bad_bands = np.ones((target.num_bands(),), dtype=np.bool_)
    else:
        target_bad_bands = np.asarray(target_bad_bands_raw, dtype=np.bool_)

    new_min_wvl = np.float64(min_wvl.to(target_unit).value)
    new_max_wvl = np.float64(max_wvl.to(target_unit).value)

    length_all_references = 0
    ref_offsets = [0]
    for reference in references:
        length_of_ref = reference.get_shape()[0]
        length_all_references += length_of_ref
        ref_offsets.append(ref_offsets[-1] + length_of_ref)

    new_refs_arr = np.full((length_all_references,), fill_value=np.nan, dtype=np.float64)
    new_refs_wvl = np.full((length_all_references,), fill_value=np.nan, dtype=np.float64)
    new_refs_bad_bands = np.ones((length_all_references,), dtype=np.bool_)

    for index, reference in enumerate(references):
        ref_unit = reference.get_wavelength_units()
        if ref_unit is None:
            raise ValueError(f"Reference spectrum '{reference.get_name()}' is missing wavelength units")

        start = ref_offsets[index]
        end = ref_offsets[index + 1]
        new_refs_arr[start:end] = np.asarray(reference.get_spectrum(), dtype=np.float64)
        new_refs_wvl[start:end] = np.asarray(
            [wavelength.to(target_unit).value for wavelength in reference.get_wavelengths()],
            dtype=np.float64,
        )
        reference_bad_bands = reference.get_bad_bands()
        if reference_bad_bands is None:
            new_refs_bad_bands[start:end] = True
        else:
            new_refs_bad_bands[start:end] = np.asarray(reference_bad_bands, dtype=np.bool_)

    threshold_arr = np.asarray(thresholds, dtype=np.float64)
    ref_offsets_arr = np.asarray(ref_offsets, dtype=np.uint32)
    if threshold_arr.shape[0] != len(references):
        raise ValueError(
            f"Number of thresholds ({threshold_arr.shape[0]})"
            f" must match number of references ({len(references)})"
        )

    return {
        "target_image_arr": target_image_arr,
        "target_wavelengths": target_wavelengths,
        "target_bad_bands": target_bad_bands,
        "min_wvl": new_min_wvl,
        "max_wvl": new_max_wvl,
        "reference_spectra_arr": new_refs_arr,
        "reference_spectra_wvls": new_refs_wvl,
        "reference_spectra_bad_bands": new_refs_bad_bands,
        "reference_spectra_indices": ref_offsets_arr,
        "thresholds": threshold_arr,
    }


def general_spectral_image_compute(
    input_ref: DataRef,
    output_refs: Sequence[DataRef],
    reference_refs: Sequence[DataRef],
    min_wvl: u.Quantity,
    max_wvl: u.Quantity,
    thresholds: Sequence[float],
    python_mode: bool,
    python_func: SpectralImageComputeFunction,
    numba_func: SpectralImageComputeFunction,
) -> None:
    """
    Shared worker entrypoint for image-mode spectral algorithms such as SAM and SFF.
    """
    packed_inputs = _prepare_general_spectral_image_inputs(
        input_ref=input_ref,
        reference_refs=reference_refs,
        min_wvl=min_wvl,
        max_wvl=max_wvl,
        thresholds=thresholds,
    )

    compute_func = python_func if python_mode else numba_func
    outputs = compute_func(
        packed_inputs["target_image_arr"],
        packed_inputs["target_wavelengths"],
        packed_inputs["target_bad_bands"],
        packed_inputs["min_wvl"],
        packed_inputs["max_wvl"],
        packed_inputs["reference_spectra_arr"],
        packed_inputs["reference_spectra_wvls"],
        packed_inputs["reference_spectra_bad_bands"],
        packed_inputs["reference_spectra_indices"],
        packed_inputs["thresholds"],
    )

    if len(outputs) != len(output_refs):
        raise ValueError(f"Expected {len(output_refs)} outputs, got {len(outputs)}")

    client = get_process_storage_client()
    for output_ref, output_value in zip(output_refs, outputs):
        output_arr = np.asarray(output_value)
        if output_arr.ndim != 3:
            raise ValueError(f"Expected 3D spectral image output, got shape {output_arr.shape}")
        client.write_data(output_ref, output_arr.transpose(1, 2, 0))


@dataclass
class SaveExternalDatasetStage(SequentialStage):
    _save_params: Sequence[Any] = ()
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=0,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = NoChunkingScheme

    def __post_init__(self):
        self.broadcast_input |= {
            "save_params": list(self._save_params),
        }

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        return input_region

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = (input_meta, chosen_scheme)
        return []

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = (input_region, output_writes)
        return partial(_run_save_external_dataset, input_ref, broadcast_inputs["save_params"])


@dataclass
class GeneralSpectralImageComputeStage(SequentialStage):
    _reference_refs: Sequence[DataRef] = ()
    _min_wvl: Optional[u.Quantity] = None
    _max_wvl: Optional[u.Quantity] = None
    _thresholds: Sequence[float] = ()
    _python_mode: bool = False
    _python_func: Optional[SpectralImageComputeFunction] = None
    _numba_func: Optional[SpectralImageComputeFunction] = None
    _output_specs: Sequence[Tuple[str, np.dtype]] = ()
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
        self.output_bindings = self.output_bindings + [DataBinding(name) for name, _ in self._output_specs]
        self.broadcast_input |= {
            "reference_refs": list(self._reference_refs),
            "thresholds": list(self._thresholds),
            "python_mode": bool(self._python_mode),
            "min_wvl": self._min_wvl,
            "max_wvl": self._max_wvl,
        }

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("General spectral image compute stage requires DatasetRegionRef input")
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=len(self._reference_refs),
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        if not isinstance(input_meta, DatasetPlanMeta):
            raise TypeError("General spectral image compute stage requires DatasetPlanMeta input")

        requests: List[AllocationRequest] = []
        for output_name, output_dtype in self._output_specs:
            requests.append(
                AllocationRequest(
                    name=output_name,
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=input_meta.height
                    * input_meta.width
                    * len(self._reference_refs)
                    * np.dtype(output_dtype).itemsize,
                    shape=(input_meta.height, input_meta.width, len(self._reference_refs)),
                    dtype=np.dtype(output_dtype),
                    delete_policy=self.get_output_delete_policy(output_name),
                )
            )
        return requests

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = input_region
        if self._python_func is None or self._numba_func is None:
            raise ValueError("General spectral image compute stage requires both python and numba functions")

        output_refs = [output_writes[output_name].ref for output_name, _ in self._output_specs]
        min_wvl = broadcast_inputs["min_wvl"]
        max_wvl = broadcast_inputs["max_wvl"]
        if min_wvl is None or max_wvl is None:
            raise ValueError("General spectral image compute stage requires min_wvl and max_wvl")

        return partial(
            general_spectral_image_compute,
            input_ref,
            output_refs,
            broadcast_inputs["reference_refs"],
            min_wvl,
            max_wvl,
            broadcast_inputs["thresholds"],
            broadcast_inputs["python_mode"],
            self._python_func,
            self._numba_func,
        )


def get_save_external_dataset_pipeline(
    dataset_ref: DataRef,
    path: str,
    format: str,
    config: Dict[str, Any],
) -> AlgorithmPipeline:
    """
    Build a no-chunking pipeline that reconstructs and saves an external dataset.

    Args:
        dataset_ref: External dataset ref to reconstruct inside the worker.
        path: Output path passed to `RasterDataLoader.save_dataset_as`.
        format: Output format passed to `RasterDataLoader.save_dataset_as`.
        config: Save configuration dictionary passed to `RasterDataLoader.save_dataset_as`.
    """
    if dataset_ref.shape is None:
        raise ValueError("dataset_ref.shape must be populated for save pipeline planning")
    if dataset_ref.dtype is None:
        raise ValueError("dataset_ref.dtype must be populated for save pipeline planning")

    dataset_plan_meta = DatasetPlanMeta(shape=dataset_ref.shape, dtype=np.dtype(dataset_ref.dtype))
    return AlgorithmPipeline(
        [
            SaveExternalDatasetStage(
                _save_params=[path, format, config],
                default_executor="process",
                input_plan_meta=dataset_plan_meta,
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=0,
                    scratch_bytes_per_scalar_in=0,
                ),
                chunking_scheme_type=NoChunkingScheme,
            )
        ]
    )


def get_general_spectral_image_pipeline(
    dataset_ref: DataRef,
    reference_refs: Sequence[DataRef],
    min_wvl: u.Quantity,
    max_wvl: u.Quantity,
    thresholds: Sequence[float],
    python_mode: bool,
    output_specs: Sequence[Tuple[str, np.dtype]],
    python_func: SpectralImageComputeFunction,
    numba_func: SpectralImageComputeFunction,
) -> AlgorithmPipeline:
    """
    Build a no-chunking pipeline for image-mode spectral algorithms that consume
    one target dataset and a list of reference spectra.
    """
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    dataset_plan_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))
    return AlgorithmPipeline(
        [
            GeneralSpectralImageComputeStage(
                _reference_refs=list(reference_refs),
                _min_wvl=min_wvl,
                _max_wvl=max_wvl,
                _thresholds=list(thresholds),
                _python_mode=python_mode,
                _python_func=python_func,
                _numba_func=numba_func,
                _output_specs=list(output_specs),
                default_executor="process",
                input_plan_meta=dataset_plan_meta,
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
                chunking_scheme_type=NoChunkingScheme,
            )
        ]
    )


def _run_compute_pca(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    pca_json_ref: DataRef,
    num_components: int,
) -> None:
    if not isinstance(input_region, DatasetRegionRef):
        raise TypeError("PCA stage requires DatasetRegionRef input_region")

    client = get_process_storage_client()
    image_data, image_meta = client.read_region(input_ref, input_region)
    image_cube = np.ma.array(image_data, copy=False).transpose(2, 0, 1)

    bad_bands = None
    if image_meta.bad_bands is not None:
        bad_bands = np.asarray(image_meta.bad_bands).astype(int).tolist()

    reduced_image, pca = compute_PCA_on_image(
        image_arr=image_cube,
        num_components=num_components,
        bad_bands=bad_bands,
        data_ignore=image_meta.nodata,
    )

    output_array = np.asarray(np.ma.getdata(reduced_image), dtype=np.float64)
    assert output_write.region is not None, "output_write.region cannot be None for PCA dataset output"
    output_write.region.validate_array_shape(output_array)
    client.write_spec(output_write, output_array)
    client.write_json_value(pca_json_ref, {"pca": pca})


def _write_pca_output_meta(
    input_ref: DataRef,
    full_input_region: DataRegion,
    output_write: "WriteSpec",
) -> None:
    if not isinstance(full_input_region, DatasetRegionRef):
        raise TypeError("PCA metadata write requires DatasetRegionRef full_input_region")

    client = get_process_storage_client()
    input_region_meta = client.get_region_meta(input_ref, full_input_region)
    output_meta = client.get_meta(output_write.ref)
    nodata = input_region_meta.nodata if input_region_meta.nodata is not None else np.nan
    pca_meta = replace(
        output_meta,
        elem_type=np.dtype(np.float64),
        nodata=nodata,
        bad_bands=None,
        wavelengths=None,
        wavelength_units=None,
        crs_wkt=input_region_meta.crs_wkt,
        geotransform=input_region_meta.geotransform,
    )
    client.write_meta(output_write.ref, pca_meta)


@dataclass
class ComputePcaStage(MapStage):
    _num_components: int = 1
    _output_ref_name: str = "pca_image"
    _pca_json_ref_name: str = "pca_model"
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
            DataBinding(self._output_ref_name),
            DataBinding(self._pca_json_ref_name, kind="json", residency="ram_cacheable"),
        ]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(input_region, DatasetRegionRef), "PCA stage requires DatasetRegionRef input"
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=self._num_components,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(input_meta, DatasetPlanMeta), "PCA stage input_meta must be DatasetPlanMeta"
        if self._num_components <= 0:
            raise ValueError(f"num_components must be positive, got {self._num_components}")

        dataset_request = AllocationRequest(
            name=self._output_ref_name,
            kind="dataset",
            residency="ram_cacheable",
            size_est=input_meta.height
            * input_meta.width
            * self._num_components
            * np.dtype(np.float64).itemsize,
            shape=(input_meta.height, input_meta.width, self._num_components),
            dtype=np.dtype(np.float64),
            delete_policy=self.get_output_delete_policy(self._output_ref_name),
        )
        json_request = AllocationRequest(
            name=self._pca_json_ref_name,
            kind="json",
            residency="ram_cacheable",
            size_est=4096,
            delete_policy=self.get_output_delete_policy(self._pca_json_ref_name),
        )
        return [dataset_request, json_request]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        pca_json_ref = output_writes[self._pca_json_ref_name].ref
        return partial(
            _run_compute_pca, input_ref, input_region, output_write, pca_json_ref, self._num_components
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_write_pca_output_meta, input_ref, full_input_region, output_write)


def get_pca_pipeline(
    dataset_ref: DataRef,
    num_components: int,
    output_ref_name: str,
    pca_json_ref_name: str,
) -> AlgorithmPipeline:
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    dataset_plan_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))
    if dataset_meta.bad_bands is not None:
        valid_bands = int(np.count_nonzero(np.asarray(dataset_meta.bad_bands) != 0))
    else:
        valid_bands = dataset_plan_meta.bands
    if num_components > valid_bands:
        raise ValueError(
            f"num_components must be <= valid input bands, got num_components={num_components}, "
            f"valid_bands={valid_bands}"
        )

    return AlgorithmPipeline(
        [
            ComputePcaStage(
                _num_components=num_components,
                _output_ref_name=output_ref_name,
                _pca_json_ref_name=pca_json_ref_name,
                default_executor="process",
                input_plan_meta=dataset_plan_meta,
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
                chunking_scheme_type=NoChunkingScheme,
            )
        ]
    )


def _prepare_continuum_removal_inputs(
    input_ref: DataRef,
    subset_image_ref: DataRef,
    x_axis_ref: DataRef,
    bad_bands_ref: DataRef,
    min_cols: int,
    min_rows: int,
    max_cols: int,
    max_rows: int,
    min_band: int,
    max_band: int,
) -> None:
    client = get_process_storage_client()
    subset_region = DatasetRegionRef(
        y0=min_rows,
        y1=max_rows,
        x0=min_cols,
        x1=max_cols,
        b0=min_band,
        b1=max_band,
    )
    image_data, region_meta = client.read_region(input_ref, subset_region, filter_data=False)
    image_data = np.ma.array(image_data, copy=False)
    if region_meta.nodata is not None:
        image_data = np.ma.masked_values(image_data, region_meta.nodata)

    (image_data,) = convert_to_float32_if_needed(image_data)
    image_data = np.asarray(image_data)
    if not image_data.flags.c_contiguous:
        image_data = np.ascontiguousarray(image_data)
    if np.ma.isMaskedArray(image_data):
        mask = np.ma.getmaskarray(image_data)
        image_data = np.asarray(np.ma.getdata(image_data), dtype=np.float64)
        image_data[mask] = np.nan
    else:
        image_data = np.asarray(image_data, dtype=np.float64)

    full_meta = client.get_meta(input_ref)
    if full_meta.wavelengths is not None:
        x_axis = np.asarray(full_meta.wavelengths[min_band:max_band], dtype=np.float64)
    else:
        x_axis = np.arange(min_band, max_band, dtype=np.float64)

    if full_meta.bad_bands is not None:
        bad_bands_arr = np.logical_not(np.asarray(full_meta.bad_bands, dtype=np.bool_))
    else:
        bad_bands_arr = np.zeros((full_meta.shape[2],), dtype=np.bool_)
    bad_bands_arr = np.asarray(bad_bands_arr[min_band:max_band], dtype=np.bool_)

    client.write_data(subset_image_ref, image_data)
    client.write_data(x_axis_ref, x_axis)
    client.write_data(bad_bands_ref, bad_bands_arr)


def _run_continuum_removal_tile(
    subset_image_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    x_axis_ref: DataRef,
    bad_bands_ref: DataRef,
) -> None:
    if not isinstance(input_region, DatasetRegionRef):
        raise TypeError("Continuum removal tile stage requires DatasetRegionRef input_region")

    from wiser.gui.permanent_plugins.continuum_removal_plugin import continuum_removal_image_numba

    client = get_process_storage_client()
    image_tile, _ = client.read_region(subset_image_ref, input_region, filter_data=False)
    x_axis, _ = client.read_data(x_axis_ref, filter_data=False)
    bad_bands_arr, _ = client.read_data(bad_bands_ref, filter_data=False)

    image_tile_array = np.asarray(np.ma.getdata(np.ma.array(image_tile, copy=False)), dtype=np.float64)
    if not image_tile_array.flags.c_contiguous:
        image_tile_array = np.ascontiguousarray(image_tile_array)

    rows = image_tile_array.shape[0]
    cols = image_tile_array.shape[1]
    bands = image_tile_array.shape[2]
    reduced_by_band = continuum_removal_image_numba(
        image_tile_array,
        np.asarray(bad_bands_arr, dtype=np.bool_),
        np.asarray(np.ma.getdata(x_axis), dtype=np.float64),
        rows,
        cols,
        bands,
    )
    reduced_tile = np.asarray(reduced_by_band, dtype=np.float64).transpose(1, 2, 0)
    assert output_write.region is not None, "Continuum removal output_write.region cannot be None"
    output_write.region.validate_array_shape(reduced_tile)
    client.write_spec(output_write, reduced_tile)


def _subset_geotransform(
    geotransform: Optional[tuple[float, ...]],
    min_cols: int,
    min_rows: int,
) -> Optional[tuple[float, ...]]:
    if geotransform is None:
        return None
    gt0, gt1, gt2, gt3, gt4, gt5 = geotransform
    return (
        float(gt0 + min_cols * gt1 + min_rows * gt2),
        float(gt1),
        float(gt2),
        float(gt3 + min_cols * gt4 + min_rows * gt5),
        float(gt4),
        float(gt5),
    )


def _write_continuum_removal_output_meta(
    input_ref: DataRef,
    output_write: "WriteSpec",
    min_cols: int,
    min_rows: int,
    max_cols: int,
    max_rows: int,
    min_band: int,
    max_band: int,
) -> None:
    _ = (max_cols, max_rows)
    client = get_process_storage_client()
    subset_region = DatasetRegionRef(
        y0=min_rows,
        y1=max_rows,
        x0=min_cols,
        x1=max_cols,
        b0=min_band,
        b1=max_band,
    )
    input_region_meta = client.get_region_meta(input_ref, subset_region)
    output_meta = client.get_meta(output_write.ref)
    continuum_meta = replace(
        output_meta,
        elem_type=np.dtype(np.float64),
        wavelengths=input_region_meta.wavelengths,
        wavelength_units=input_region_meta.wavelength_units,
        nodata=input_region_meta.nodata,
        bad_bands=input_region_meta.bad_bands,
        crs_wkt=input_region_meta.crs_wkt,
        geotransform=_subset_geotransform(input_region_meta.geotransform, min_cols, min_rows),
    )
    client.write_meta(output_write.ref, continuum_meta)


@dataclass
class ContinuumRemovalImageStage(MapStage):
    _output_ref_name: str = "continuum_removed_image"
    _prepared_subset_ref_name: str = "_continuum_subset_image"
    _x_axis_ref_name: str = "_continuum_x_axis"
    _bad_bands_ref_name: str = "_continuum_bad_bands"
    _min_cols: int = 0
    _min_rows: int = 0
    _max_cols: int = 0
    _max_rows: int = 0
    _min_band: int = 0
    _max_band: int = 0
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]
        self.broadcast_input |= {
            "subset_image_ref": DataBinding(self._prepared_subset_ref_name),
            "x_axis_ref": DataBinding(self._x_axis_ref_name),
            "bad_bands_ref": DataBinding(self._bad_bands_ref_name),
        }

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, DatasetRegionRef
        ), "Continuum removal stage requires DatasetRegionRef input"
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=input_region.b0,
            b1=input_region.b1,
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
        ), "Continuum removal stage input_meta must be DatasetPlanMeta"
        bands = input_meta.bands
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=input_meta.height * input_meta.width * bands * np.dtype(np.float64).itemsize,
                shape=input_meta.shape,
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            ),
            AllocationRequest(
                name=self._prepared_subset_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=input_meta.height * input_meta.width * bands * np.dtype(np.float64).itemsize,
                shape=input_meta.shape,
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._prepared_subset_ref_name),
            ),
            AllocationRequest(
                name=self._x_axis_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=bands * np.dtype(np.float64).itemsize,
                shape=(bands,),
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._x_axis_ref_name),
            ),
            AllocationRequest(
                name=self._bad_bands_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=bands * np.dtype(np.bool_).itemsize,
                shape=(bands,),
                dtype=np.dtype(np.bool_),
                delete_policy=self.get_output_delete_policy(self._bad_bands_ref_name),
            ),
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = input_ref
        output_write = output_writes[self._output_ref_name]
        subset_image_ref: DataRef = broadcast_inputs["subset_image_ref"]
        x_axis_ref: DataRef = broadcast_inputs["x_axis_ref"]
        bad_bands_ref: DataRef = broadcast_inputs["bad_bands_ref"]
        return partial(
            _run_continuum_removal_tile,
            subset_image_ref,
            input_region,
            output_write,
            x_axis_ref,
            bad_bands_ref,
        )

    def pre_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = (full_input_region, output_writes)
        subset_image_ref: DataRef = broadcast_inputs["subset_image_ref"]
        x_axis_ref: DataRef = broadcast_inputs["x_axis_ref"]
        bad_bands_ref: DataRef = broadcast_inputs["bad_bands_ref"]
        return partial(
            _prepare_continuum_removal_inputs,
            input_ref,
            subset_image_ref,
            x_axis_ref,
            bad_bands_ref,
            self._min_cols,
            self._min_rows,
            self._max_cols,
            self._max_rows,
            self._min_band,
            self._max_band,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = (full_input_region, broadcast_inputs)
        output_write = output_writes[self._output_ref_name]
        return partial(
            _write_continuum_removal_output_meta,
            input_ref,
            output_write,
            self._min_cols,
            self._min_rows,
            self._max_cols,
            self._max_rows,
            self._min_band,
            self._max_band,
        )


def get_continuum_removal_image_pipeline(
    dataset_ref: DataRef,
    min_cols: int,
    min_rows: int,
    max_cols: int,
    max_rows: int,
    min_band: int,
    max_band: int,
    output_ref_name: str,
) -> AlgorithmPipeline:
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    total_rows, total_cols, total_bands = dataset_meta.shape
    if not (0 <= min_cols <= max_cols <= total_cols):
        raise ValueError(f"Invalid column subset: ({min_cols}, {max_cols}) for total_cols={total_cols}")
    if not (0 <= min_rows <= max_rows <= total_rows):
        raise ValueError(f"Invalid row subset: ({min_rows}, {max_rows}) for total_rows={total_rows}")
    if not (0 <= min_band <= max_band <= total_bands):
        raise ValueError(f"Invalid band subset: ({min_band}, {max_band}) for total_bands={total_bands}")

    subset_shape = (max_rows - min_rows, max_cols - min_cols, max_band - min_band)
    dataset_plan_meta = DatasetPlanMeta(shape=subset_shape, dtype=np.dtype(np.float64))
    return AlgorithmPipeline(
        [
            ContinuumRemovalImageStage(
                _output_ref_name=output_ref_name,
                _min_cols=min_cols,
                _min_rows=min_rows,
                _max_cols=max_cols,
                _max_rows=max_rows,
                _min_band=min_band,
                _max_band=max_band,
                default_executor="process",
                input_plan_meta=dataset_plan_meta,
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
                chunking_scheme_type=SpatialTileScheme,
            )
        ]
    )


def get_good_band_runs(bad_bands: np.ndarray) -> list[tuple[int, int]]:
    bad_bands_array = np.asarray(bad_bands)
    if bad_bands_array.ndim != 1:
        raise ValueError(f"Expected 1D bad_bands array, got shape={bad_bands_array.shape}")

    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, is_good in enumerate(bad_bands_array != 0):
        if is_good:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, bad_bands_array.shape[0]))
    return runs


def split_dataset_tile_by_good_band_runs(
    tile_yxb: Union[np.ndarray, np.ma.MaskedArray],
    good_band_runs: list[tuple[int, int]],
) -> list[np.ndarray]:
    tile_array = np.asarray(np.ma.getdata(np.ma.array(tile_yxb, copy=False)))
    if tile_array.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [y][x][b], got {tile_array.shape}")
    return [tile_array[:, :, start:end] for start, end in good_band_runs]


def recombine_dataset_tile_from_good_band_runs(
    original_shape: tuple[int, int, int],
    good_band_runs: list[tuple[int, int]],
    filtered_chunks: list[np.ndarray],
    base_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(good_band_runs) != len(filtered_chunks):
        raise ValueError(
            f"Chunk count mismatch while recombining Savitzky-Golay output: "
            f"runs={len(good_band_runs)}, chunks={len(filtered_chunks)}"
        )

    if base_array is None:
        combined = np.zeros(original_shape, dtype=np.float64)
    else:
        combined = np.array(base_array, copy=True)
        if combined.shape != original_shape:
            raise ValueError(
                f"base_array shape must match original_shape: "
                f"base_array.shape={combined.shape}, original_shape={original_shape}"
            )

    for (start, end), chunk in zip(good_band_runs, filtered_chunks):
        expected_shape = (original_shape[0], original_shape[1], end - start)
        if chunk.shape != expected_shape:
            raise ValueError(
                f"Filtered chunk shape mismatch while recombining Savitzky-Golay output: "
                f"chunk.shape={chunk.shape}, expected={expected_shape}"
            )
        combined[:, :, start:end] = chunk
    return combined


def validate_no_unmasked_nonfinite_values(tile: Union[np.ndarray, np.ma.MaskedArray]) -> None:
    tile_array = np.ma.array(tile, copy=False)
    raw = np.asarray(np.ma.getdata(tile_array), dtype=np.float64)
    mask = np.ma.getmaskarray(tile_array)
    nonfinite_mask = ~np.isfinite(raw)
    if np.any(nonfinite_mask & ~mask):
        raise ValueError("Savitzky-Golay filter input contains unmasked NaN or Inf values")


def _resolve_good_band_runs_from_region_meta(region_meta, band_count: int) -> list[tuple[int, int]]:
    if region_meta.bad_bands is None:
        return [(0, band_count)]

    bad_bands_array = np.asarray(region_meta.bad_bands)
    if bad_bands_array.shape != (band_count,):
        raise ValueError(
            f"Bad bands shape must match dataset band count: "
            f"bad_bands shape={bad_bands_array.shape}, bands={band_count}"
        )
    return get_good_band_runs(bad_bands_array)


def _validate_savgol_parameters_against_runs(
    *,
    good_band_runs: list[tuple[int, int]],
    window_length: int,
    polyorder: int,
) -> None:
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd for mode='interp', got {window_length}")
    if polyorder < 0:
        raise ValueError(f"polyorder must be non-negative, got {polyorder}")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length must be greater than polyorder, got window_length={window_length}, "
            f"polyorder={polyorder}"
        )
    if len(good_band_runs) == 0:
        raise ValueError("Savitzky-Golay filter requires at least one contiguous good-band run")

    min_run_length = min(end - start for start, end in good_band_runs)
    if window_length > min_run_length:
        raise ValueError(
            f"window_length must be <= every contiguous good-band run length, got "
            f"window_length={window_length}, shortest_good_run={min_run_length}"
        )


def _run_savgol_filter_dataset_tile(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    window_length: int,
    polyorder: int,
) -> None:
    if not isinstance(input_region, DatasetRegionRef):
        raise TypeError("Savitzky-Golay dataset stage requires DatasetRegionRef input_region")

    client = get_process_storage_client()
    input_tile, input_region_meta = client.read_region(input_ref, input_region, filter_data=False)
    input_tile_raw = np.asarray(input_tile, dtype=np.float64)

    if input_tile_raw.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [y][x][b], got {input_tile_raw.shape}")

    good_band_runs = _resolve_good_band_runs_from_region_meta(input_region_meta, input_tile_raw.shape[2])
    _validate_savgol_parameters_against_runs(
        good_band_runs=good_band_runs,
        window_length=window_length,
        polyorder=polyorder,
    )

    exclusion_mask = np.zeros_like(input_tile_raw, dtype=np.bool_)
    if input_region_meta.nodata is not None:
        if np.isnan(input_region_meta.nodata):
            exclusion_mask |= np.isnan(input_tile_raw)
        else:
            exclusion_mask |= input_tile_raw == input_region_meta.nodata
    if input_region_meta.bad_bands is not None:
        bad_band_mask = (np.asarray(input_region_meta.bad_bands) == 0).reshape(1, 1, input_tile_raw.shape[2])
        exclusion_mask |= np.broadcast_to(bad_band_mask, input_tile_raw.shape)
    validate_no_unmasked_nonfinite_values(np.ma.array(input_tile_raw, mask=exclusion_mask, copy=False))

    chunks = split_dataset_tile_by_good_band_runs(input_tile_raw, good_band_runs)
    filtered_chunks = []
    for chunk in chunks:
        filtered_chunk = savgol_filter(
            chunk,
            window_length=window_length,
            polyorder=polyorder,
            deriv=0,
            axis=2,
            mode="interp",
        )
        filtered_chunks.append(np.asarray(filtered_chunk, dtype=np.float64))

    output_tile = recombine_dataset_tile_from_good_band_runs(
        original_shape=input_tile_raw.shape,
        good_band_runs=good_band_runs,
        filtered_chunks=filtered_chunks,
        base_array=input_tile_raw,
    )

    if input_region_meta.nodata is not None:
        nodata_mask = exclusion_mask.copy()
        if input_region_meta.bad_bands is not None:
            bad_band_mask = (np.asarray(input_region_meta.bad_bands) == 0).reshape(
                1, 1, input_tile_raw.shape[2]
            )
            nodata_mask &= ~np.broadcast_to(bad_band_mask, input_tile_raw.shape)
        if np.any(nodata_mask):
            output_tile[nodata_mask] = input_region_meta.nodata

    assert output_write.region is not None, "output_write.region cannot be None for Savitzky-Golay output"
    output_write.region.validate_array_shape(output_tile)
    client.write_spec(output_write, output_tile.astype(np.float64, copy=False))


def _write_savgol_output_meta(
    input_ref: DataRef,
    full_input_region: DataRegion,
    output_write: "WriteSpec",
) -> None:
    if not isinstance(full_input_region, DatasetRegionRef):
        raise TypeError("Savitzky-Golay metadata write requires DatasetRegionRef full_input_region")

    client = get_process_storage_client()
    input_region_meta = client.get_region_meta(input_ref, full_input_region)
    output_meta = client.get_meta(output_write.ref)
    savgol_meta = replace(
        output_meta,
        elem_type=np.dtype(np.float64),
        nodata=input_region_meta.nodata,
        wavelengths=input_region_meta.wavelengths,
        wavelength_units=input_region_meta.wavelength_units,
        bad_bands=input_region_meta.bad_bands,
        crs_wkt=input_region_meta.crs_wkt,
        geotransform=input_region_meta.geotransform,
    )
    client.write_meta(output_write.ref, savgol_meta)


@dataclass
class SavGolayFilterStage(MapStage):
    _window_length: int = 3
    _polyorder: int = 1
    _output_ref_name: str = "savgol_filtered_dataset"
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("Savitzky-Golay stage expects DatasetRegionRef input")
        return input_region

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "Savitzky-Golay stage input_meta must be DatasetPlanMeta"
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=input_meta.height
                * input_meta.width
                * input_meta.bands
                * np.dtype(np.float64).itemsize,
                shape=(input_meta.height, input_meta.width, input_meta.bands),
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(
            _run_savgol_filter_dataset_tile,
            input_ref,
            input_region,
            output_write,
            self._window_length,
            self._polyorder,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_write_savgol_output_meta, input_ref, full_input_region, output_write)


def get_savgol_filter_pipeline(
    dataset_ref: DataRef,
    window_length: int,
    polyorder: int,
    output_ref_name: str,
) -> AlgorithmPipeline:
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    band_count = int(dataset_meta.shape[2])
    if dataset_meta.bad_bands is None:
        good_band_runs = [(0, band_count)]
    else:
        bad_bands_array = np.asarray(dataset_meta.bad_bands)
        if bad_bands_array.shape != (band_count,):
            raise ValueError(
                f"Dataset bad bands shape must match input bands: "
                f"bad_bands shape={bad_bands_array.shape}, bands={band_count}"
            )
        good_band_runs = get_good_band_runs(bad_bands_array)

    _validate_savgol_parameters_against_runs(
        good_band_runs=good_band_runs,
        window_length=window_length,
        polyorder=polyorder,
    )

    input_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))
    return AlgorithmPipeline(
        [
            SavGolayFilterStage(
                _window_length=window_length,
                _polyorder=polyorder,
                _output_ref_name=output_ref_name,
                default_executor="process",
                input_plan_meta=input_meta,
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
                chunking_scheme_type=SpatialTileScheme,
            )
        ]
    )


# region Smoothing filter MapStages (spatial / spectral ndimage)


NDIMAGE_SMOOTHING_FILTER_REGISTRY: Dict[str, Callable[..., Any]] = {
    "uniform_filter": ndimage.uniform_filter,
    "median_filter": ndimage.median_filter,
    "gaussian_filter": ndimage.gaussian_filter,
}

_GAUSSIAN_FILTER_DEFAULT_TRUNCATE = 4.0


def _resolve_smoothing_ndimage_callable(
    *,
    ndimage_filter_fn: Optional[Callable[..., Any]],
    filter_registry_key: Optional[str],
) -> Callable[..., Any]:
    if ndimage_filter_fn is not None:
        return ndimage_filter_fn
    if filter_registry_key is None:
        raise ValueError("Provide _ndimage_filter_fn or _filter_registry_key (e.g. 'median_filter').")
    fn = NDIMAGE_SMOOTHING_FILTER_REGISTRY.get(filter_registry_key)
    if fn is None:
        raise ValueError(
            f"Unknown _filter_registry_key={filter_registry_key!r}. "
            f"Known keys: {sorted(NDIMAGE_SMOOTHING_FILTER_REGISTRY.keys())}"
        )
    return fn


def _normalize_smoothing_filter_kwargs(
    fn: Callable[..., Any],
    raw_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop None-valued keys; apply gaussian_filter defaults (e.g. truncate)."""
    kwargs = {k: v for k, v in dict(raw_kwargs or {}).items() if v is not None}
    if fn is ndimage.gaussian_filter or getattr(fn, "__name__", "") == "gaussian_filter":
        # SciPy default truncate is 4.0; we set explicitly so callers can rely on it.
        kwargs.setdefault("truncate", _GAUSSIAN_FILTER_DEFAULT_TRUNCATE)
    return kwargs


def _finalize_smoothing_filter_kwargs_spatial(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Force filtering on dataset axes (0, 1) = (y, x). ``size`` / ``sigma`` / ``radius`` must be int or float
    (same along y and x) or a length-2 pair of int/float for (axis 0, axis 1).
    """
    out = dict(kwargs)
    if "axes" in out and tuple(out["axes"]) != (0, 1):
        raise ValueError(
            "SmoothingFilterSpatial fixes axes=(0, 1) for (y, x); "
            "remove 'axes' from _filter_kwargs or use (0, 1)."
        )
    out["axes"] = (0, 1)

    name = getattr(fn, "__name__", "")
    if fn is ndimage.gaussian_filter or name == "gaussian_filter":
        sig = out["sigma"]
        if isinstance(sig, (int, float)):
            out["sigma"] = (sig, sig)
        elif isinstance(sig, (tuple, list)) and len(sig) == 2:
            if not isinstance(sig[0], (int, float)) or not isinstance(sig[1], (int, float)):
                raise TypeError(f"Spatial gaussian_filter 'sigma' pair must be int or float, got {sig!r}")
            out["sigma"] = (sig[0], sig[1])
        else:
            raise TypeError(
                "Spatial gaussian_filter 'sigma' must be int or float "
                f"or a pair of int/float, got {type(sig).__name__}"
            )
        if "radius" in out:
            rad = out["radius"]
            if isinstance(rad, (int, float)):
                out["radius"] = (rad, rad)
            elif isinstance(rad, (tuple, list)) and len(rad) == 2:
                if not isinstance(rad[0], (int, float)) or not isinstance(rad[1], (int, float)):
                    raise TypeError(
                        f"Spatial gaussian_filter 'radius' pair must be int or float, got {rad!r}"
                    )
                out["radius"] = (rad[0], rad[1])
            else:
                raise TypeError(
                    "Spatial gaussian_filter 'radius' must be int or float "
                    f"or a pair of int/float, got {type(rad).__name__}"
                )
    elif fn in (ndimage.uniform_filter, ndimage.median_filter) or name in ("uniform_filter", "median_filter"):
        sz = out["size"]
        if isinstance(sz, (int, float)):
            out["size"] = (sz, sz)
        elif isinstance(sz, (tuple, list)) and len(sz) == 2:
            if not isinstance(sz[0], (int, float)) or not isinstance(sz[1], (int, float)):
                raise TypeError(f"Spatial {name} 'size' pair must be int or float, got {sz!r}")
            out["size"] = (sz[0], sz[1])
        else:
            raise TypeError(
                f"Spatial {name} 'size' must be int or float or a pair of int/float, got {type(sz).__name__}"
            )
    return out


def _finalize_smoothing_filter_kwargs_spectral(
    fn: Callable[..., Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Force filtering on axis (2,) (spectral / band axis). Coerce size / sigma / radius to one scalar each.
    """
    out = dict(kwargs)
    if "axes" in out and tuple(out["axes"]) != (2,):
        raise ValueError(
            "SmoothingFilterSpectral fixes axes=(2,) (spectral axis); "
            "remove 'axes' from _filter_kwargs or use (2,)."
        )
    out["axes"] = (2,)

    name = getattr(fn, "__name__", "")
    if fn is ndimage.gaussian_filter or name == "gaussian_filter":
        if not isinstance(out["sigma"], (int, float)):
            raise TypeError(
                f"Spectral gaussian_filter 'sigma' must be int or float, got {type(out['sigma']).__name__}"
            )
        if "radius" in out:
            if not isinstance(out["radius"], (int, float)):
                raise TypeError(
                    "Spectral gaussian_filter 'radius' must be int or float, "
                    f"got {type(out['radius']).__name__}"
                )
    elif fn in (ndimage.uniform_filter, ndimage.median_filter) or name in ("uniform_filter", "median_filter"):
        if not isinstance(out["size"], (int, float)):
            raise TypeError(f"Spectral {name} 'size' must be int or float, got {type(out['size']).__name__}")
    return out


def _validate_smoothing_ndimage_parameter_combinations(
    fn: Callable[..., Any], kwargs: Dict[str, Any]
) -> None:
    """
    Semantic rules beyond signature checks (no array dry-run).
    """
    name = getattr(fn, "__name__", "")
    if fn is ndimage.gaussian_filter or name == "gaussian_filter":
        if "sigma" not in kwargs:
            raise ValueError("gaussian_filter requires keyword 'sigma' (must not be None).")
        if kwargs["sigma"] is None:
            raise ValueError("gaussian_filter keyword 'sigma' must not be None.")
        # radius is optional; truncate is set in _normalize_smoothing_filter_kwargs
    if fn in (ndimage.uniform_filter, ndimage.median_filter) or name in ("uniform_filter", "median_filter"):
        if "size" not in kwargs:
            raise ValueError(f"{name} requires keyword 'size' (must not be None).")


def _validate_ndimage_kwargs_match_signature(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> None:
    """
    Ensure every keyword is accepted by ``fn`` using inspect.signature and no value is None.
    We run _normalize_smoothing_filter_kwargs before calling this function to get rid of None values.
    """
    for k, v in kwargs.items():
        if v is None:
            raise ValueError(f"Filter keyword {k!r} must not be None after normalization (got None).")

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Cannot inspect signature of {fn!r}") from e

    params = list(sig.parameters.values())
    if not params:
        raise TypeError(f"{fn!r} has no parameters")

    var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    if var_kw:
        return

    # First parameter is the input array (positional); remaining names are keyword/positional-or-keyword.
    accepted: set[str] = set()
    for p in params[1:]:
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(p.name)

    for k in kwargs:
        if k not in accepted:
            raise TypeError(f"{getattr(fn, '__name__', fn)!r}() got an unexpected keyword argument {k!r}")


def build_smoothing_exclusion_mask(
    tile_yxb: np.ndarray,
    nodata: Optional[float],
    bad_bands: Optional[Any],
) -> np.ndarray:
    """
    Return a boolean mask (same shape as ``tile_yxb``) that is True wherever a
    sample should be treated as invalid before smoothing.

    Invalid means either:
    - the sample equals the dataset's nodata value (or is NaN when nodata is NaN), or
    - the sample belongs to a bad band (bad_bands[b] == 0).

    Split out from the tile runner so the test suite can build the same mask
    when computing the reference (expected) array.
    """
    mask = np.zeros(tile_yxb.shape, dtype=np.bool_)
    if nodata is not None:
        if np.isnan(nodata):
            mask |= np.isnan(tile_yxb)
        else:
            mask |= tile_yxb == nodata
    if bad_bands is not None:
        bad_band_mask = (np.asarray(bad_bands) == 0).reshape(1, 1, tile_yxb.shape[2])
        mask |= np.broadcast_to(bad_band_mask, tile_yxb.shape)
    return mask


def _smoothing_dataset_tile_exclusion_mask(tile_yxb: np.ndarray, region_meta: Any) -> np.ndarray:
    return build_smoothing_exclusion_mask(tile_yxb, region_meta.nodata, region_meta.bad_bands)


def _nan_aware_linear_ndimage_filtered_output(
    work: np.ndarray,
    ndimage_filter_fn: Callable[..., Any],
    filter_kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    NaN-aware linear smoothing: run ``ndimage_filter_fn`` on zero-filled values and on a
    finite mask, then divide (renormalize). Valid for ``uniform_filter`` and
    ``gaussian_filter``; not for ``median_filter``.
    """
    valid_mask = np.isfinite(work).astype(np.float64, copy=False)
    values = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)

    numerator = np.asarray(ndimage_filter_fn(values, **filter_kwargs), dtype=np.float64, order="C")
    denominator = np.asarray(ndimage_filter_fn(valid_mask, **filter_kwargs), dtype=np.float64, order="C")

    output_tile = np.full_like(numerator, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=output_tile, where=denominator > 0.0)
    return output_tile


def _write_smoothing_ndimage_output_meta(
    input_ref: DataRef,
    full_input_region: DataRegion,
    output_write: "WriteSpec",
) -> None:
    if not isinstance(full_input_region, DatasetRegionRef):
        raise TypeError("Smoothing filter metadata write requires DatasetRegionRef full_input_region")

    client = get_process_storage_client()
    input_region_meta = client.get_region_meta(input_ref, full_input_region)
    output_meta = client.get_meta(output_write.ref)
    out_meta = replace(
        output_meta,
        elem_type=np.dtype(np.float64),
        nodata=input_region_meta.nodata,
        wavelengths=input_region_meta.wavelengths,
        wavelength_units=input_region_meta.wavelength_units,
        bad_bands=input_region_meta.bad_bands,
        crs_wkt=input_region_meta.crs_wkt,
        geotransform=input_region_meta.geotransform,
    )
    client.write_meta(output_write.ref, out_meta)


def _run_smoothing_filter_ndimage_tile(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    ndimage_filter_fn: Callable[..., Any],
    filter_kwargs: Dict[str, Any],
    filter_kind: Optional[str] = None,
) -> None:
    """
    Read tile, mask invalid samples to NaN, coerce non-finite values to NaN, run ndimage filter,
    then copy original values back at excluded positions (nodata/bad-band cells from input).
    Extra NaNs from the kernel near edges are left as-is (see feature spec).
    """
    if not isinstance(input_region, DatasetRegionRef):
        raise TypeError("Smoothing filter stages require DatasetRegionRef input_region")

    client = get_process_storage_client()
    input_tile, region_meta = client.read_region(input_ref, input_region, filter_data=False)

    if input_tile.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [y][x][b], got {input_tile.shape}")

    exclusion = _smoothing_dataset_tile_exclusion_mask(input_tile, region_meta)

    work = np.asarray(input_tile, dtype=np.float64, order="C")
    work[exclusion] = np.nan
    work[~np.isfinite(work)] = np.nan

    if filter_kind in ("uniform_filter", "gaussian_filter"):
        # NaN-aware linear smoother: zero non-finite samples, filter mask and values with the
        # same kernel, then renormalize (valid for mean / Gaussian weights).
        output_tile = _nan_aware_linear_ndimage_filtered_output(work, ndimage_filter_fn, filter_kwargs)
    else:
        filtered = ndimage_filter_fn(work, **filter_kwargs)
        output_tile = np.asarray(filtered, dtype=np.float64, order="C")

    # Restore original stored values at mask locations so nodata/bad-band
    # semantics stay aligned with metadata.
    output_tile[exclusion] = input_tile[exclusion]

    assert output_write.region is not None, "output_write.region cannot be None for smoothing filter output"
    output_write.region.validate_array_shape(output_tile)
    client.write_spec(output_write, output_tile.astype(np.float64, copy=False))


@dataclass
class SmoothingFilterSpatial(MapStage):
    """
    Apply a scipy.ndimage filter over spatial axes (y, x).

    ``axes=(0, 1)`` is set automatically. ``size`` (uniform/median) or ``sigma`` / optional ``radius``
    (gaussian) must be one scalar (symmetric kernel) or a pair of two scalars for the two spatial axes.
    Do not pass ``axes`` in ``_filter_kwargs`` unless it is ``(0, 1)``.

    Uses ``SpectralBatchDatasetScheme`` so each work unit receives the full spatial extent
    (H, W) for a subset of bands. Because spatial filtering only needs neighbors along y and x,
    bands are independent and can be processed in slices without boundary effects.
    """

    _ndimage_filter_fn: Optional[Callable[..., Any]] = None
    _filter_registry_key: Optional[str] = None
    _filter_kwargs: Optional[Dict[str, Any]] = None
    _output_ref_name: str = "spatial_smoothed_dataset"
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpectralBatchDatasetScheme

    _resolved_ndimage_filter_fn: Callable[..., Any] = field(init=False, repr=False)
    _resolved_filter_kwargs: Dict[str, Any] = field(init=False, repr=False)
    _resolved_filter_kind: Optional[str] = field(init=False, repr=False)

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

        fn = _resolve_smoothing_ndimage_callable(
            ndimage_filter_fn=self._ndimage_filter_fn,
            filter_registry_key=self._filter_registry_key,
        )
        kwargs = _normalize_smoothing_filter_kwargs(fn, self._filter_kwargs)
        _validate_smoothing_ndimage_parameter_combinations(fn, kwargs)
        kwargs = _finalize_smoothing_filter_kwargs_spatial(fn, kwargs)
        _validate_ndimage_kwargs_match_signature(fn, kwargs)

        object.__setattr__(self, "_resolved_ndimage_filter_fn", fn)
        object.__setattr__(self, "_resolved_filter_kwargs", kwargs)
        object.__setattr__(
            self, "_resolved_filter_kind", self._filter_registry_key or getattr(fn, "__name__", None)
        )

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("SmoothingFilterSpatial expects DatasetRegionRef input")
        return input_region

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "SmoothingFilterSpatial input_meta must be DatasetPlanMeta"
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=input_meta.height
                * input_meta.width
                * input_meta.bands
                * np.dtype(np.float64).itemsize,
                shape=(input_meta.height, input_meta.width, input_meta.bands),
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(
            _run_smoothing_filter_ndimage_tile,
            input_ref,
            input_region,
            output_write,
            self._resolved_ndimage_filter_fn,
            self._resolved_filter_kwargs,
            self._resolved_filter_kind,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_write_smoothing_ndimage_output_meta, input_ref, full_input_region, output_write)


@dataclass
class SmoothingFilterSpectral(MapStage):
    """
    Apply a scipy.ndimage filter along the spectral (band) axis.

    ``axes=(2,)`` is set automatically. ``size`` (uniform/median) or ``sigma`` / optional ``radius``
    (gaussian) must be a single scalar (kernel along bands only). Do not pass ``axes`` in
    ``_filter_kwargs`` unless it is ``(2,)``.

    Uses ``SpatialTileScheme`` so each work unit receives a spatial tile with the full band depth
    (b0=0, b1=B). This is required for spectral filtering: the kernel along axis 2 needs access to
    every band of each pixel's spectrum to produce a correct result.
    """

    _ndimage_filter_fn: Optional[Callable[..., Any]] = None
    _filter_registry_key: Optional[str] = None
    _filter_kwargs: Optional[Dict[str, Any]] = None
    _output_ref_name: str = "spectral_smoothed_dataset"
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    _resolved_ndimage_filter_fn: Callable[..., Any] = field(init=False, repr=False)
    _resolved_filter_kwargs: Dict[str, Any] = field(init=False, repr=False)
    _resolved_filter_kind: Optional[str] = field(init=False, repr=False)

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

        fn = _resolve_smoothing_ndimage_callable(
            ndimage_filter_fn=self._ndimage_filter_fn,
            filter_registry_key=self._filter_registry_key,
        )
        kwargs = _normalize_smoothing_filter_kwargs(fn, self._filter_kwargs)
        _validate_smoothing_ndimage_parameter_combinations(fn, kwargs)
        kwargs = _finalize_smoothing_filter_kwargs_spectral(fn, kwargs)
        _validate_ndimage_kwargs_match_signature(fn, kwargs)

        object.__setattr__(self, "_resolved_ndimage_filter_fn", fn)
        object.__setattr__(self, "_resolved_filter_kwargs", kwargs)
        object.__setattr__(
            self, "_resolved_filter_kind", self._filter_registry_key or getattr(fn, "__name__", None)
        )

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("SmoothingFilterSpectral expects DatasetRegionRef input")
        return input_region

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "SmoothingFilterSpectral input_meta must be DatasetPlanMeta"
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=input_meta.height
                * input_meta.width
                * input_meta.bands
                * np.dtype(np.float64).itemsize,
                shape=(input_meta.height, input_meta.width, input_meta.bands),
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(
            _run_smoothing_filter_ndimage_tile,
            input_ref,
            input_region,
            output_write,
            self._resolved_ndimage_filter_fn,
            self._resolved_filter_kwargs,
            self._resolved_filter_kind,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_write_smoothing_ndimage_output_meta, input_ref, full_input_region, output_write)


# endregion Smoothing filter MapStages


def _running_covariance(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    mean_ref: DataRef,
    total: TotalLike,
    num_features: int = -1,
) -> None:
    client = get_process_storage_client()
    if isinstance(total, DataRef):
        total = _resolve_total_payload(total)
    output_ref = output_write.ref
    running_cov, _ = client.read_data(output_ref)
    noise, _ = client.read_region(input_ref, input_region)
    mean_arr, _ = client.read_data(mean_ref)
    input_region_meta = client.get_region_meta(input_ref, input_region)
    # We do the below because masked arrays have trouble with matrix multiplications
    if np.ma.isMaskedArray(noise):
        # Essentiall removes all the nodata and bad bands affects
        noise_raw = np.ma.getdata(noise.filled(0))
    else:
        noise_raw = np.asarray(noise)
    noise_raw = np.asarray(noise_raw)
    invalid_pixels = np.any(~np.isfinite(noise_raw), axis=2)
    noise_raw[invalid_pixels, :] = 0
    if np.ma.isMaskedArray(mean_arr):
        mean_arr_raw = np.ma.getdata(mean_arr)
    else:
        mean_arr_raw = np.asarray(mean_arr)
    assert noise_raw.ndim == 3, "noise_raw should have 3 dimensions"
    assert mean_arr_raw.ndim == 1, "mean_arr_raw should have 1 dimension"
    band_count = noise_raw.shape[2]
    good_band_mask = np.ones((band_count,), dtype=bool)
    if input_region_meta.bad_bands is not None:
        bad_bands_array = np.asarray(input_region_meta.bad_bands)
        if bad_bands_array.shape != (band_count,):
            raise ValueError(
                f"Bad bands shape must match dataset band count: "
                f"bad_bands shape={bad_bands_array.shape}, bands={band_count}"
            )
        good_band_mask = bad_bands_array != 0

    noise_raw = noise_raw[:, :, good_band_mask]
    if mean_arr_raw.shape[0] == band_count:
        mean_arr_raw = mean_arr_raw[good_band_mask]
    elif mean_arr_raw.shape[0] != noise_raw.shape[2]:
        raise ValueError(
            f"Filtered covariance mean width does not match filtered band count: "
            f"mean_width={mean_arr_raw.shape[0]}, filtered_bands={noise_raw.shape[2]}"
        )
    if num_features != -1 and noise_raw.shape[2] != num_features:
        raise ValueError(
            f"Filtered covariance feature count does not match requested num_features: "
            f"filtered_features={noise_raw.shape[2]}, requested={num_features}"
        )
    mean_arr_raw = mean_arr_raw[np.newaxis, np.newaxis, :]
    mean_centered_noise = noise_raw - mean_arr_raw
    flattened_noise = mean_centered_noise.reshape(-1, mean_centered_noise.shape[2])
    sum_outer_product = flattened_noise.T @ flattened_noise
    partial_cov_matrix = sum_outer_product / (total - 1)
    partial_cov_matrix = partial_cov_matrix[:, :, np.newaxis]
    running_cov += partial_cov_matrix
    client.write_data(output_ref, running_cov)


@dataclass
class CalcCovMatrixStage(SequentialStage):
    """
    Calculates the covariance matrix of a dataset. This assumes
    the data has not been mean subtracted. The input_ref data
    is assumed to be of shape [y][x][b] where [y][x] are the
    pixel axis and we want the noise of [b].
    """

    # You must override this
    _total_spectra: int = 0
    # You must define this
    _output_ref_name: str = "cov_running"
    _num_features: int = -1
    _internal_total_ref_name: str = "_calc_cov_total"
    # You must either override this or put it in broadcast_input
    _mean_ref: DataRef = None
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self):
        if "mean" not in self.broadcast_input:
            self.broadcast_input |= {"mean": self._mean_ref}
        if "internal_total_ref" not in self.broadcast_input:
            self.broadcast_input |= {"internal_total_ref": DataBinding(self._internal_total_ref_name)}
        if "total" not in self.broadcast_input:
            if self._total_spectra > 0:
                self.broadcast_input |= {"total": self._total_spectra}
            else:
                self.broadcast_input |= {"total": DataBinding(self._internal_total_ref_name)}
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        """
        The input region will be something like [k][m][b] where k < y and m < x.
        We want to write to a covarianec matrix of [b][b], so out output region
        should be [b][b]
        """
        assert isinstance(
            input_region, DatasetRegionRef
        ), "Input region for calculate shift difference noise must be DatasetRegionRef"

        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        """
        This stage will just allocate data for the covariance matrix. We
        will be writing to this array.
        """
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for CalculateCovarianceMatrix"
        feature_count = self._num_features if self._num_features != -1 else input_meta.bands

        if feature_count <= 0:
            raise ValueError(f"num_features must be positive when provided, got {self._num_features}")

        size_est = feature_count * feature_count * input_meta.dtype.itemsize
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=size_est,
                shape=(feature_count, feature_count, 1),
                dtype=input_meta.dtype,
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            ),
            AllocationRequest(
                name=self._internal_total_ref_name,
                kind="json",
                residency="ram_cacheable",
                size_est=64,
                delete_policy=self.get_output_delete_policy(self._internal_total_ref_name),
            ),
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        total = broadcast_inputs["internal_total_ref"]
        mean: DataRef = broadcast_inputs["mean"]
        return partial(
            _running_covariance,
            input_ref,
            input_region,
            output_write,
            mean,
            total,
            self._num_features,
        )

    def pre_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = output_writes
        total_ref: DataRef = broadcast_inputs["internal_total_ref"]
        provided_total = broadcast_inputs.get("total")
        if isinstance(provided_total, DataRef) and provided_total.ref_id == total_ref.ref_id:
            provided_total = None
        return partial(
            _copy_or_compute_valid_dataset_total,
            input_ref,
            full_input_region,
            total_ref,
            provided_total,
        )


def get_noise_covariance_pipeline(noise_ref: DataRef, output_ref_name: str) -> AlgorithmPipeline:
    mean_output_ref_name = "mean_stage"
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(noise_ref)
    plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=data_meta.elem_type)
    noise_mean_stage = get_spectral_mean_stage(noise_ref, mean_output_ref_name)
    noise_cov_stage = CalcCovMatrixStage(
        _total_spectra=data_meta.shape[2],
        _output_ref_name=output_ref_name,
        default_executor="process",
        input_plan_meta=plan_meta,
        broadcast_input={"mean": DataBinding(mean_output_ref_name)},
    )

    return AlgorithmPipeline([noise_mean_stage, noise_cov_stage])


def _running_mean(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    total: TotalLike,
) -> None:
    client = get_process_storage_client()
    output_ref = output_write.ref
    running_mean, _ = client.read_data(output_ref)
    if isinstance(total, DataRef):
        total = _resolve_total_payload(total)
    if total <= 0:
        raise ValueError(f"Spectral mean requires a positive total, got {total}")

    data, data_meta = client.read_region(input_ref, input_region)
    flattened = _flatten_valid_dataset_rows(data, data_meta)
    flattened = np.asarray(flattened, dtype=np.float64)
    if flattened.size == 0:
        return

    spectra_sum: np.ndarray = flattened.sum(axis=0, dtype=np.float64) / total
    running_mean += spectra_sum
    client.write_data(output_ref, running_mean)


def _good_band_mask_for_region_meta(region_meta, band_count: int) -> np.ndarray:
    good_band_mask = np.ones((band_count,), dtype=np.bool_)
    if region_meta.bad_bands is None:
        return good_band_mask

    bad_bands_array = np.asarray(region_meta.bad_bands)
    if bad_bands_array.shape != (band_count,):
        raise ValueError(
            f"Bad bands shape must match dataset band count: "
            f"bad_bands shape={bad_bands_array.shape}, bands={band_count}"
        )
    return bad_bands_array != 0


def _flatten_valid_dataset_rows(
    data: Union[np.ndarray, np.ma.MaskedArray],
    data_meta,
) -> np.ndarray:
    data_array = np.ma.array(data, copy=False)
    data_raw = np.asarray(np.ma.getdata(data_array), dtype=np.float64)
    data_mask = np.ma.getmaskarray(data_array)

    if data_raw.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [y][x][b], got {data_raw.shape}")

    good_band_mask = _good_band_mask_for_region_meta(data_meta, data_raw.shape[2])
    filtered_data = data_raw[:, :, good_band_mask]
    filtered_mask = data_mask[:, :, good_band_mask]
    flattened = filtered_data.reshape(-1, filtered_data.shape[2])
    # Drop any pixel whose surviving bands still contain masked, NaN, or Inf values.
    invalid_rows = np.any(filtered_mask.reshape(-1, filtered_mask.shape[2]), axis=1)
    invalid_rows |= np.any(~np.isfinite(flattened), axis=1)
    return flattened[~invalid_rows]


def count_valid_dataset_pixels(dataset_ref: DataRef) -> int:
    client = get_process_storage_client()
    data, data_meta = client.read_data(dataset_ref)
    return int(_flatten_valid_dataset_rows(data, data_meta).shape[0])


def _resolve_total_payload(total_like: TotalLike) -> int:
    client = get_process_storage_client()
    if isinstance(total_like, DataRef):
        total_payload = client.read_json_value(total_like)
        if not isinstance(total_payload, dict) or "total" not in total_payload:
            raise ValueError("Expected JSON total payload with key 'total'")
        return int(total_payload["total"])
    return int(total_like)


def _write_valid_dataset_total(
    input_ref: DataRef,
    full_input_region: DataRegion,
    total_ref: DataRef,
) -> None:
    if not isinstance(full_input_region, DatasetRegionRef):
        raise TypeError("Valid dataset total pre-task requires a DatasetRegionRef full_input_region")

    client = get_process_storage_client()
    region_meta = client.get_region_meta(input_ref, full_input_region)
    dataset_plan_meta = DatasetPlanMeta(
        shape=(
            full_input_region.y1 - full_input_region.y0,
            full_input_region.x1 - full_input_region.x0,
            full_input_region.b1 - full_input_region.b0,
        ),
        dtype=np.dtype(region_meta.elem_type),
    )

    total = 0
    for tile_region in SpatialTileScheme(tile_h=32, tile_w=32).iter_chunks(dataset_plan_meta):
        tile_region = DatasetRegionRef(
            y0=full_input_region.y0 + tile_region.y0,
            y1=full_input_region.y0 + tile_region.y1,
            x0=full_input_region.x0 + tile_region.x0,
            x1=full_input_region.x0 + tile_region.x1,
            b0=full_input_region.b0 + tile_region.b0,
            b1=full_input_region.b0 + tile_region.b1,
        )
        data_tile, data_tile_meta = client.read_region(input_ref, tile_region)
        total += _flatten_valid_dataset_rows(data_tile, data_tile_meta).shape[0]

    client.write_json_value(total_ref, {"total": int(total)})


def _copy_or_compute_valid_dataset_total(
    input_ref: DataRef,
    full_input_region: DataRegion,
    total_ref: DataRef,
    provided_total: Optional[TotalLike] = None,
) -> None:
    if provided_total is None:
        _write_valid_dataset_total(input_ref, full_input_region, total_ref)
        return

    resolved_total = _resolve_total_payload(provided_total)
    if resolved_total <= 0:
        _write_valid_dataset_total(input_ref, full_input_region, total_ref)
        return
    client = get_process_storage_client()
    client.write_json_value(total_ref, {"total": int(resolved_total)})


def _write_spectral_mean_meta(
    input_ref: DataRef,
    full_input_region: DataRegion,
    output_write: "WriteSpec",
) -> None:
    if not isinstance(full_input_region, DatasetRegionRef):
        raise TypeError("Spectral mean metadata write requires a DatasetRegionRef full_input_region")

    client = get_process_storage_client()
    input_region_meta = client.get_region_meta(input_ref, full_input_region)
    output_meta = client.get_meta(output_write.ref)
    output_bands = output_meta.shape[0]
    good_band_mask = _good_band_mask_for_region_meta(
        input_region_meta, full_input_region.b1 - full_input_region.b0
    )
    wavelengths = input_region_meta.wavelengths
    if wavelengths is not None and len(wavelengths) == len(good_band_mask):
        wavelengths = np.asarray(wavelengths)[good_band_mask]
    bad_bands = None
    if output_bands == len(good_band_mask):
        bad_bands = input_region_meta.bad_bands
    mean_meta = replace(
        output_meta,
        wavelengths=wavelengths,
        wavelength_units=input_region_meta.wavelength_units,
        bad_bands=bad_bands,
    )
    client.write_meta(output_write.ref, mean_meta)


@dataclass
class SpectralMeanStage(SequentialStage):
    """
    Computes a spectral mean after a pre-stage pass counts valid spectra rows.
    """

    # You should override this
    _output_ref_name: str = "spectral_mean_1"
    _internal_total_ref_name: str = "_internal_total"
    _dataset_ref: Optional[DataRef] = None

    def __post_init__(self):
        if "internal_total_ref" not in self.broadcast_input:
            self.broadcast_input |= {"internal_total_ref": DataBinding(self._internal_total_ref_name)}
        if "total" not in self.broadcast_input:
            self.broadcast_input |= {"total": DataBinding(self._internal_total_ref_name)}
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        """
        We just accumulate in one input ref, so we don't need the a data region slice
        """
        assert isinstance(
            input_region, DatasetRegionRef
        ), "Input region for calculate shift difference noise must be DatasetRegionRef"

        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        """
        This stage will just allocate data for the mean spectrum. We
        will be writing to this array.
        """
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for SpectralMeanStage"

        np_type = np.float64
        feature_count = input_meta.bands
        if self._dataset_ref is not None:
            meta = get_process_storage_client().get_meta(self._dataset_ref)
            if meta.bad_bands is not None:
                feature_count = int(np.count_nonzero(np.asarray(meta.bad_bands) != 0))

        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="spectrum",
                residency="ram_cacheable",
                size_est=feature_count * np.dtype(np_type).itemsize,
                shape=(feature_count,),
                dtype=np.dtype(np_type),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            ),
            AllocationRequest(
                name=self._internal_total_ref_name,
                kind="json",
                residency="ram_cacheable",
                size_est=64,
                delete_policy=self.get_output_delete_policy(self._internal_total_ref_name),
            ),
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        total = broadcast_inputs["internal_total_ref"]
        return partial(_running_mean, input_ref, input_region, output_write, total)

    def pre_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = output_writes
        total_ref: DataRef = broadcast_inputs["internal_total_ref"]
        provided_total = broadcast_inputs.get("total")
        if isinstance(provided_total, DataRef) and provided_total.ref_id == total_ref.ref_id:
            provided_total = None
        return partial(
            _copy_or_compute_valid_dataset_total,
            input_ref,
            full_input_region,
            total_ref,
            provided_total,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_write_spectral_mean_meta, input_ref, full_input_region, output_write)


def get_spectral_mean_stage(
    dataset_ref: DataRef,
    output_ref_name: str,
    input_binding: Optional[DataBinding] = None,
) -> SpectralMeanStage:
    """Build a :class:`SpectralMeanStage` for ``dataset_ref``.

    Args:
        dataset_ref: Dataset whose meta drives allocation sizes and whose
            pixels are read at execution time (unless ``input_binding`` points
            elsewhere).
        output_ref_name: Name for the mean-spectrum allocation.
        input_binding: Optional binding for which :class:`DataRef` to read
            tiles from.  Defaults to ``DataBinding("__task_input__")``.
    """
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=data_meta.elem_type)
    stage_kwargs: Dict[str, Any] = dict(
        _output_ref_name=output_ref_name,
        default_executor="process",
        input_plan_meta=plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=SpatialTileScheme,
        _dataset_ref=dataset_ref,
        output_bindings=[DataBinding(output_ref_name)],
    )
    if input_binding is not None:
        stage_kwargs["input_binding"] = input_binding
    return SpectralMeanStage(**stage_kwargs)


@dataclass(frozen=True)
class EigenVectorsAndValues:
    """
    Lightweight descriptor that references storage-backed eigen outputs.

    This object intentionally stores only reference IDs and shape metadata so it
    can be serialized to JSON cheaply and passed through task outputs.

    Eigen vectors should be in decreasing order of eigen value from left to right
    Each row should have an eigen vector

    Eigen vectors are in an [N][N] array and eigen vectors are stored in the rows.
    """

    eigen_vectors_ref: DataRef
    eigen_values_ref: DataRef
    num_vectors: int
    vector_dimension: int
    covariance_ref: Optional[DataRef] = None
    mean_ref: Optional[DataRef] = None
    good_band_mask_ref: Optional[DataRef] = None

    def count(self) -> int:
        return self.num_vectors

    def get_eigen_vector(self, i: int) -> np.ndarray:
        if i < 0 or i >= self.num_vectors:
            raise IndexError(f"eigen vector index out of range: {i}")
        client = get_process_storage_client()
        vector_batch_region = SpectraBatchRef(i0=i, i1=i + 1, length=self.vector_dimension)
        vector_batch, _ = client.read_region(self.eigen_vectors_ref, vector_batch_region)
        vector_array = np.asarray(np.ma.getdata(vector_batch))
        return vector_array[0]

    def get_eigen_value(self, i: int) -> float:
        if i < 0 or i >= self.num_vectors:
            raise IndexError(f"eigen value index out of range: {i}")
        client = get_process_storage_client()
        values, _ = client.read_data(self.eigen_values_ref)
        values_array = np.asarray(np.ma.getdata(values))
        return float(values_array[i])


def _write_eigendecomposition(
    input_ref: DataRef,
    input_region: DataRegion,
    output_info_ref: DataRef,
    output_vectors_ref: DataRef,
    output_values_ref: DataRef,
) -> None:
    client = get_process_storage_client()
    matrix, _ = client.read_region(input_ref, input_region)
    matrix_array = np.asarray(np.ma.getdata(matrix))
    if matrix_array.ndim == 3:
        assert matrix_array.shape[-1] == 1
        assert matrix_array.shape[0] == matrix_array.shape[1]
        matrix_array = np.squeeze(matrix_array, axis=2)
    if matrix_array.ndim != 2:
        raise ValueError(f"Expected 2D square matrix, got shape={matrix_array.shape}")
    if matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError(f"Expected square matrix, got shape={matrix_array.shape}")
    # np.linalg.eig returns eigenvectors as columns. We transpose to [N][d] rows.
    eigen_values, eigen_vectors = np.linalg.eig(matrix_array)
    eigen_values = np.real_if_close(eigen_values)
    eigen_vectors = np.real_if_close(eigen_vectors)
    sort_desc = np.argsort(eigen_values)[::-1]
    eigen_values = np.asarray(eigen_values[sort_desc], dtype=np.float64)
    # We transpose because np.linalg.eig gives us eigen vectors in the columns, but
    # we want them in the rows
    eigen_vectors = np.asarray(eigen_vectors[:, sort_desc].T, dtype=np.float64)

    max_ev = float(np.max(eigen_values))
    if max_ev > 0.0:
        rel_thresh = max_ev * _EIGENDECOMP_RELATIVE_EIGENVALUE_CUTOFF
        eigen_values = np.where(eigen_values < rel_thresh, 0.0, eigen_values)

    client.write_data(output_vectors_ref, eigen_vectors)
    client.write_data(output_values_ref, eigen_values)

    descriptor = EigenVectorsAndValues(
        eigen_vectors_ref=output_vectors_ref,
        eigen_values_ref=output_values_ref,
        num_vectors=eigen_vectors.shape[0],
        vector_dimension=eigen_vectors.shape[1],
    )
    client.write_json_value(output_info_ref, {"eigen": descriptor})


@dataclass
class EigenDecompositionStage(SequentialStage):
    """
    Compute eigendecomposition for a square [N][N] matrix and persist:
      - eigen vectors in an array [N][N] (eigen vectors are in the ),
      - eigen values in an array [N],
      - a lightweight JSON descriptor that references both arrays.
    """

    _output_ref_name: str = "eigenvectors_and_values"
    _vectors_ref_name: str = "eigen_vectors"
    _values_ref_name: str = "eigen_values"
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
        # Declare all three allocations so planners and delete-policy overrides see them
        # (vectors/values were always allocated in generate_allocation_requests but were
        # previously omitted here, so KEEP could not apply to those refs).
        self.output_bindings = self.output_bindings + [
            DataBinding(self._vectors_ref_name),
            DataBinding(self._values_ref_name),
            DataBinding(self._output_ref_name, kind="json"),
        ]
        self.broadcast_input |= {
            "eigen_vectors_ref": DataBinding(self._vectors_ref_name),
            "eigen_values_ref": DataBinding(self._values_ref_name),
        }

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        assert isinstance(
            input_meta, SpectraListPlanMeta
        ), "input_meta must be of type SpectraListPlanMeta for EigenDecompositionStage"
        if input_meta.num_spectra != input_meta.spectrum_length:
            raise ValueError(
                f"EigenDecompositionStage expects a square matrix, got shape="
                f"({input_meta.num_spectra}, {input_meta.spectrum_length})"
            )

        n = input_meta.num_spectra
        vectors_dtype = np.float64
        values_dtype = np.float64
        vectors_size_est = n * n * np.dtype(vectors_dtype).itemsize
        values_size_est = n * np.dtype(values_dtype).itemsize

        return [
            AllocationRequest(
                name=self._vectors_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=vectors_size_est,
                shape=(n, n),
                dtype=vectors_dtype,
                delete_policy=self.get_output_delete_policy(self._vectors_ref_name),
            ),
            AllocationRequest(
                name=self._values_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=values_size_est,
                shape=(n,),
                dtype=values_dtype,
                delete_policy=self.get_output_delete_policy(self._values_ref_name),
            ),
            AllocationRequest(
                name=self._output_ref_name,
                kind="json",
                residency="ram_cacheable",
                size_est=1024,
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            ),
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        output_vectors_ref: DataRef = broadcast_inputs["eigen_vectors_ref"]
        output_values_ref: DataRef = broadcast_inputs["eigen_values_ref"]
        return partial(
            _write_eigendecomposition,
            input_ref,
            input_region,
            output_write.ref,
            output_vectors_ref,
            output_values_ref,
        )


def get_eigendecomposition_stage(
    matrix_ref: DataRef,
    output_ref_name: str,
) -> EigenDecompositionStage:
    storage_client = get_process_storage_client()
    matrix_meta = storage_client.get_meta(matrix_ref)
    if len(matrix_meta.shape) != 2:
        raise ValueError(f"Expected 2D square matrix input for eigendecomposition, got {matrix_meta.shape}")
    if matrix_meta.shape[0] != matrix_meta.shape[1]:
        raise ValueError(f"Expected square matrix input, got {matrix_meta.shape}")

    n = int(matrix_meta.shape[0])
    input_meta = SpectraListPlanMeta(
        num_spectra=n,
        spectrum_length=n,
        dtype=matrix_meta.elem_type,
    )
    return EigenDecompositionStage(
        _output_ref_name=output_ref_name,
        _vectors_ref_name=f"{output_ref_name}_vectors",
        _values_ref_name=f"{output_ref_name}_values",
        default_executor="process",
        input_plan_meta=input_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=NoChunkingScheme,
    )


def get_eigendecomposition_pipeline(matrix_ref: DataRef, output_ref_name: str) -> AlgorithmPipeline:
    return AlgorithmPipeline([get_eigendecomposition_stage(matrix_ref, output_ref_name)])


def _build_diagonal_matrix(
    input_ref: DataRef,
    input_region: DataRegion,
    output_ref: DataRef,
) -> None:
    _ = input_region
    client = get_process_storage_client()
    values, _ = client.read_data(input_ref)
    values_array = np.asarray(np.ma.getdata(values), dtype=np.float64).ravel()
    diag_matrix = np.diag(values_array).astype(np.float64)
    client.write_data(output_ref, diag_matrix)


@dataclass
class DiagonalMatrixFromValuesStage(SequentialStage):
    """Build an (N, N) diagonal matrix from a 1-D vector of N values.

    Reads the primary input as a flat array of length N and writes
    ``np.diag(values)`` to the output allocation.  Intended for converting
    a vector of eigenvalues into the diagonal covariance matrix Λ.
    """

    _output_ref_name: str = "diagonal_matrix"
    _n: int = 0
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
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        n = self._n
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=n * n * np.dtype(np.float64).itemsize,
                shape=(n, n),
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_build_diagonal_matrix, input_ref, input_region, output_write.ref)


def _write_whitening_matrix(
    input_ref: DataRef,
    input_region: DataRegion,
    output_ref: DataRef,
) -> None:
    _ = input_region
    client = get_process_storage_client()
    envelope_payload = client.read_json_value(input_ref)
    if not isinstance(envelope_payload, dict) or "eigen" not in envelope_payload:
        raise ValueError("Expected JSON payload with key 'eigen' for whitening matrix stage input")

    descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
    if not isinstance(descriptor, EigenVectorsAndValues):
        raise TypeError("Expected payload['eigen'] to be an EigenVectorsAndValues instance")

    eigen_vectors, _ = client.read_data(descriptor.eigen_vectors_ref)
    eigen_values, _ = client.read_data(descriptor.eigen_values_ref)
    eigen_vectors_array = np.asarray(np.ma.getdata(eigen_vectors), dtype=np.float64)
    eigen_values_array = np.asarray(np.ma.getdata(eigen_values), dtype=np.float64)

    if eigen_vectors_array.ndim != 2:
        raise ValueError(f"Expected eigen vectors with 2D shape [n][d], got {eigen_vectors_array.shape}")
    if eigen_values_array.ndim != 1:
        raise ValueError(f"Expected eigen values with 1D shape [n], got {eigen_values_array.shape}")
    if eigen_vectors_array.shape[0] != eigen_values_array.shape[0]:
        raise ValueError(
            f"Eigen vector/value count mismatch: n_vectors={eigen_vectors_array.shape[0]}, "
            f"n_values={eigen_values_array.shape[0]}"
        )

    if np.any(eigen_values_array < 0):
        raise ValueError("Whitening matrix cannot be computed: one or more eigen values are negative")

    # Zero eigen values correspond to non-invertible directions, so keep their
    # whitening scale at 0 instead of dividing by 0.
    inverse_sqrt_values = np.zeros_like(eigen_values_array, dtype=np.float64)
    nonzero_mask = eigen_values_array > 0
    inverse_sqrt_values[nonzero_mask] = 1.0 / np.sqrt(eigen_values_array[nonzero_mask])
    inverse_sqrt_eigen_values = np.diag(inverse_sqrt_values)
    whitening_matrix = eigen_vectors_array.T @ inverse_sqrt_eigen_values @ eigen_vectors_array
    client.write_data(output_ref, whitening_matrix.astype(np.float64, copy=False))


@dataclass
class WhiteningMatrixStage(SequentialStage):
    """
    Build a whitening matrix from an EigenVectorsAndValues descriptor.

    Expects the stage input to be a JSON ref with payload:
      {"eigen": EigenVectorsAndValues(...)}
    """

    _output_ref_name: str = "whitening_matrix"
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
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, SpectraBatchRef
        ), "Input region for WhiteningMatrixStage must be SpectraBatchRef"
        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        assert isinstance(
            input_meta, SpectraListPlanMeta
        ), "input_meta must be of type SpectraListPlanMeta for WhiteningMatrixStage"

        dtype = np.float64
        size_est = input_meta.num_spectra * input_meta.spectrum_length * np.dtype(dtype).itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="array",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(input_meta.num_spectra, input_meta.spectrum_length),
            dtype=dtype,
            delete_policy=self.get_output_delete_policy(self._output_ref_name),
        )
        return [alloc_request]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(
            _write_whitening_matrix,
            input_ref,
            input_region,
            output_write.ref,
        )


def get_whitening_matrix_stage(
    eigen_descriptor_ref: DataRef,
    output_ref_name: str,
) -> WhiteningMatrixStage:
    storage_client = get_process_storage_client()
    envelope_payload = storage_client.read_json_value(eigen_descriptor_ref)
    if not isinstance(envelope_payload, dict) or "eigen" not in envelope_payload:
        raise ValueError("Expected JSON payload with key 'eigen' for whitening matrix stage input")
    descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
    if not isinstance(descriptor, EigenVectorsAndValues):
        raise TypeError("Expected payload['eigen'] to be an EigenVectorsAndValues instance")

    input_meta = SpectraListPlanMeta(
        num_spectra=descriptor.num_vectors,
        spectrum_length=descriptor.vector_dimension,
        dtype=np.dtype(np.float64),
    )
    return WhiteningMatrixStage(
        _output_ref_name=output_ref_name,
        default_executor="process",
        input_plan_meta=input_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=NoChunkingScheme,
    )


def get_whitening_matrix_pipeline(
    eigen_descriptor_ref: DataRef,
    output_ref_name: str,
) -> AlgorithmPipeline:
    return AlgorithmPipeline([get_whitening_matrix_stage(eigen_descriptor_ref, output_ref_name)])


def _multiply_matrix_refs(
    output_ref: DataRef,
    matrix_refs: Sequence[DataRef],
) -> None:
    client = get_process_storage_client()
    if len(matrix_refs) == 0:
        raise ValueError("MatrixMultiplicationStage requires at least one matrix ref")

    product: Optional[np.ndarray] = None
    for i, matrix_ref in enumerate(matrix_refs):
        matrix, _ = client.read_data(matrix_ref)
        matrix_array = np.asarray(np.ma.getdata(matrix), dtype=np.float64)
        if matrix_array.ndim == 3:
            if matrix_array.shape[-1] != 1:
                raise ValueError(
                    f"Expected matrix ref at index {i} to have shape [m][n] or [m][n][1], "
                    f"got {matrix_array.shape}"
                )
            matrix_array = np.squeeze(matrix_array, axis=2)
        if matrix_array.ndim != 2:
            raise ValueError(f"Expected matrix ref at index {i} to be 2D, got {matrix_array.shape}")
        if product is None:
            product = matrix_array
            continue

        if product.shape[1] != matrix_array.shape[0]:
            raise ValueError(
                f"Matrix chain shape mismatch at index {i}: "
                f"left shape={product.shape}, right shape={matrix_array.shape}"
            )
        product = product @ matrix_array
    assert product is not None, "product must not be None after validating matrix_refs"
    client.write_data(output_ref, product.astype(np.float64, copy=False))


@dataclass
class MatrixMultiplicationStage(SequentialStage):
    """
    Multiply a list of matrices in order: [A, B, C] -> A @ B @ C.
    """

    _output_ref_name: str = "matrix_product"
    _matrix_refs: Optional[Sequence[DataRef]] = None
    _matrix_input_names: Sequence[str] = ()
    _output_shape: Optional[tuple[int, int]] = None
    _output_dtype: np.dtype = np.dtype(np.float64)
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
        if len(self._matrix_input_names) == 0:
            if self._matrix_refs is None or len(self._matrix_refs) == 0:
                raise ValueError("MatrixMultiplicationStage requires matrix refs or matrix input names")
            generated_names = tuple(f"matrix_ref_{i}" for i in range(len(self._matrix_refs)))
            self._matrix_input_names = generated_names
            for name, matrix_ref in zip(generated_names, self._matrix_refs):
                if name not in self.broadcast_input:
                    self.broadcast_input |= {name: matrix_ref}
        else:
            for name in self._matrix_input_names:
                if name not in self.broadcast_input:
                    raise ValueError(
                        f"MatrixMultiplicationStage missing broadcast input for matrix name '{name}'"
                    )

        if self._output_shape is None:
            raise ValueError("MatrixMultiplicationStage requires an explicit output shape")

        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, SpectraBatchRef
        ), "Input region for MatrixMultiplicationStage must be SpectraBatchRef"
        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        assert isinstance(
            input_meta, SpectraListPlanMeta
        ), "input_meta must be of type SpectraListPlanMeta for MatrixMultiplicationStage"
        _ = chosen_scheme
        output_rows, output_cols = self._output_shape
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=output_rows * output_cols * np.dtype(self._output_dtype).itemsize,
                shape=(output_rows, output_cols),
                dtype=np.dtype(self._output_dtype),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = input_ref
        _ = input_region
        output_write = output_writes[self._output_ref_name]
        matrix_refs = [broadcast_inputs[name] for name in self._matrix_input_names]
        return partial(
            _multiply_matrix_refs,
            output_write.ref,
            matrix_refs,
        )


def get_matrix_multiplication_stage(
    matrix_refs: Sequence[DataRef],
    output_ref_name: str,
) -> MatrixMultiplicationStage:
    storage_client = get_process_storage_client()
    if len(matrix_refs) == 0:
        raise ValueError("matrix_refs must contain at least one matrix ref")

    matrix_shapes: list[tuple[int, int]] = []
    output_dtype = None
    for i, matrix_ref in enumerate(matrix_refs):
        matrix_meta = storage_client.get_meta(matrix_ref)
        if output_dtype is None:
            output_dtype = matrix_meta.elem_type
        shape = matrix_meta.shape
        if len(shape) == 3 and shape[-1] == 1:
            shape = shape[:2]
        if len(shape) != 2:
            raise ValueError(
                f"Expected matrix ref at index {i} to have shape [m][n], got {matrix_meta.shape}"
            )
        matrix_shapes.append((int(shape[0]), int(shape[1])))

    for i in range(1, len(matrix_shapes)):
        left_shape = matrix_shapes[i - 1]
        right_shape = matrix_shapes[i]
        if left_shape[1] != right_shape[0]:
            raise ValueError(
                f"Matrix chain shape mismatch between indices {i - 1} and {i}: "
                f"{left_shape} cannot be multiplied with {right_shape}"
            )

    input_rows, input_cols = matrix_shapes[0]
    output_shape = (matrix_shapes[0][0], matrix_shapes[-1][1])
    input_meta = SpectraListPlanMeta(
        num_spectra=input_rows,
        spectrum_length=input_cols,
        dtype=output_dtype,
    )

    return MatrixMultiplicationStage(
        _output_ref_name=output_ref_name,
        _matrix_refs=matrix_refs,
        _output_shape=output_shape,
        _output_dtype=np.dtype(np.float64),
        default_executor="process",
        input_plan_meta=input_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=NoChunkingScheme,
    )


def get_matrix_multiplication_pipeline(
    matrix_refs: Sequence[DataRef],
    output_ref_name: str,
) -> AlgorithmPipeline:
    return AlgorithmPipeline([get_matrix_multiplication_stage(matrix_refs, output_ref_name)])


def _apply_matrix_to_dataset(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    left_multiply_matrices: Sequence[DataRef],
    right_multiply_matrices: Sequence[DataRef],
) -> None:
    client = get_process_storage_client()
    data_tile, _ = client.read_region(input_ref, input_region)
    data_tile_array = np.asarray(np.ma.getdata(data_tile))

    if data_tile_array.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [m][n][b], got {data_tile_array.shape}")

    flattened = data_tile_array.reshape(-1, data_tile_array.shape[2]).astype(np.float64, copy=False)
    current_bands = flattened.shape[1]

    # Left matrices act on each spectrum as a column vector. With flattened rows, that is:
    #   x_col -> L @ x_col  ===  x_row -> x_row @ L.T
    for i, matrix_ref in enumerate(left_multiply_matrices):
        matrix, _ = client.read_data(matrix_ref)
        matrix_array = np.asarray(np.ma.getdata(matrix), dtype=np.float64)
        if matrix_array.ndim == 3:
            if matrix_array.shape[-1] != 1:
                raise ValueError(
                    f"Expected left matrix at index {i} to have shape [out][in] or [out][in][1], "
                    f"got {matrix_array.shape}"
                )
            matrix_array = np.squeeze(matrix_array, axis=2)
        if matrix_array.ndim != 2:
            raise ValueError(f"Expected left matrix at index {i} to be 2D, got {matrix_array.shape}")
        if matrix_array.shape[1] != current_bands:
            raise ValueError(
                f"Left matrix dimension mismatch at index {i}: "
                f"current_bands={current_bands}, matrix shape={matrix_array.shape}"
            )
        flattened = flattened @ matrix_array.T
        current_bands = matrix_array.shape[0]

    # Right matrices act directly on the flattened row representation:
    #   x_row -> x_row @ R
    for i, matrix_ref in enumerate(right_multiply_matrices):
        matrix, _ = client.read_data(matrix_ref)
        matrix_array = np.asarray(np.ma.getdata(matrix), dtype=np.float64)
        if matrix_array.ndim == 3:
            if matrix_array.shape[-1] != 1:
                raise ValueError(
                    f"Expected right matrix at index {i} to have shape [in][out] or [in][out][1], "
                    f"got {matrix_array.shape}"
                )
            matrix_array = np.squeeze(matrix_array, axis=2)
        if matrix_array.ndim != 2:
            raise ValueError(f"Expected right matrix at index {i} to be 2D, got {matrix_array.shape}")
        if matrix_array.shape[0] != current_bands:
            raise ValueError(
                f"Right matrix dimension mismatch at index {i}: "
                f"current_bands={current_bands}, matrix shape={matrix_array.shape}"
            )
        flattened = flattened @ matrix_array
        current_bands = matrix_array.shape[1]

    transformed_tile = flattened.reshape(data_tile_array.shape[0], data_tile_array.shape[1], current_bands)
    client.write_spec(output_write, transformed_tile.astype(data_tile_array.dtype, copy=False))


@dataclass
class ApplyMatrixToDatasetStage(MapStage):
    """
    Apply matrix chains to a [y][x][b] dataset tile-by-tile.

    The tile is flattened to [num_pixels, bands].
    - left_multiply_matrices are applied to each spectrum as column-vector transforms,
      which is implemented as flattened @ left_matrix.T
    - right_multiply_matrices are applied directly to the flattened row representation,
      which is implemented as flattened @ right_matrix
    """

    _output_ref_name: str = "matrix_applied_dataset"
    _left_multiply_matrices: Sequence[DataRef] = ()
    _right_multiply_matrices: Sequence[DataRef] = ()
    _left_multiply_matrix_names: Sequence[str] = ()
    _right_multiply_matrix_names: Sequence[str] = ()
    _output_bands: Optional[int] = None
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self):
        if len(self._left_multiply_matrix_names) == 0:
            self._left_multiply_matrix_names = tuple(
                f"left_matrix_ref_{i}" for i in range(len(self._left_multiply_matrices))
            )
            for name, matrix_ref in zip(self._left_multiply_matrix_names, self._left_multiply_matrices):
                if name not in self.broadcast_input:
                    self.broadcast_input |= {name: matrix_ref}
        else:
            if len(self._left_multiply_matrices) > 0 and len(self._left_multiply_matrix_names) != len(
                self._left_multiply_matrices
            ):
                raise ValueError("left matrix names must match left matrix refs count")
            for name in self._left_multiply_matrix_names:
                if name not in self.broadcast_input:
                    raise ValueError(f"Missing broadcast input for left matrix '{name}'")

        if len(self._right_multiply_matrix_names) == 0:
            self._right_multiply_matrix_names = tuple(
                f"right_matrix_ref_{i}" for i in range(len(self._right_multiply_matrices))
            )
            for name, matrix_ref in zip(self._right_multiply_matrix_names, self._right_multiply_matrices):
                if name not in self.broadcast_input:
                    self.broadcast_input |= {name: matrix_ref}
        else:
            if len(self._right_multiply_matrices) > 0 and len(self._right_multiply_matrix_names) != len(
                self._right_multiply_matrices
            ):
                raise ValueError("right matrix names must match right matrix refs count")
            for name in self._right_multiply_matrix_names:
                if name not in self.broadcast_input:
                    raise ValueError(f"Missing broadcast input for right matrix '{name}'")

        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, DatasetRegionRef
        ), "Input region for ApplyMatrixToDatasetStage must be DatasetRegionRef"
        if self._output_bands is None:
            return input_region
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=self._output_bands,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for ApplyMatrixToDatasetStage"
        out_bands = self._output_bands if self._output_bands is not None else input_meta.bands
        size_est = input_meta.height * input_meta.width * out_bands * input_meta.dtype.itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="dataset",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(input_meta.height, input_meta.width, out_bands),
            dtype=input_meta.dtype,
            delete_policy=self.get_output_delete_policy(self._output_ref_name),
        )
        return [alloc_request]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        left_multiply_matrices = [broadcast_inputs[name] for name in self._left_multiply_matrix_names]
        right_multiply_matrices = [broadcast_inputs[name] for name in self._right_multiply_matrix_names]
        return partial(
            _apply_matrix_to_dataset,
            input_ref,
            input_region,
            output_write,
            left_multiply_matrices,
            right_multiply_matrices,
        )


def get_apply_matrix_to_dataset_stage(
    dataset_ref: DataRef,
    matrix_ref: DataRef,
    output_ref_name: str,
) -> ApplyMatrixToDatasetStage:
    return get_apply_matrices_to_dataset_stage(
        dataset_ref=dataset_ref,
        left_multiply_matrices=(matrix_ref,),
        right_multiply_matrices=(),
        output_ref_name=output_ref_name,
    )


def get_apply_matrices_to_dataset_stage(
    dataset_ref: DataRef,
    left_multiply_matrices: Sequence[DataRef],
    right_multiply_matrices: Sequence[DataRef],
    output_ref_name: str,
) -> ApplyMatrixToDatasetStage:
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)

    if len(data_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {data_meta.shape}")

    current_bands = int(data_meta.shape[2])
    for i, matrix_ref in enumerate(left_multiply_matrices):
        matrix_meta = storage_client.get_meta(matrix_ref)
        shape = (
            matrix_meta.shape[:2]
            if len(matrix_meta.shape) == 3 and matrix_meta.shape[-1] == 1
            else matrix_meta.shape
        )
        if len(shape) != 2:
            raise ValueError(f"Expected left matrix shape [out][in], got {matrix_meta.shape}")
        if int(shape[1]) != current_bands:
            raise ValueError(
                f"Left matrix band mismatch at index {i}: current_bands={current_bands}, matrix shape={shape}"
            )
        current_bands = int(shape[0])

    for i, matrix_ref in enumerate(right_multiply_matrices):
        matrix_meta = storage_client.get_meta(matrix_ref)
        shape = (
            matrix_meta.shape[:2]
            if len(matrix_meta.shape) == 3 and matrix_meta.shape[-1] == 1
            else matrix_meta.shape
        )
        if len(shape) != 2:
            raise ValueError(f"Expected right matrix shape [in][out], got {matrix_meta.shape}")
        if int(shape[0]) != current_bands:
            raise ValueError(
                f"Right matrix band mismatch at index {i}: current_bands={current_bands}, "
                f"matrix shape={shape}"
            )
        current_bands = int(shape[1])

    input_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=np.dtype(data_meta.elem_type))
    return ApplyMatrixToDatasetStage(
        _output_ref_name=output_ref_name,
        _left_multiply_matrices=tuple(left_multiply_matrices),
        _right_multiply_matrices=tuple(right_multiply_matrices),
        _output_bands=current_bands,
        default_executor="process",
        input_plan_meta=input_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=SpatialTileScheme,
    )


def get_apply_matrix_to_dataset_pipeline(
    dataset_ref: DataRef,
    matrix_ref: DataRef,
    output_ref_name: str,
) -> AlgorithmPipeline:
    return AlgorithmPipeline([get_apply_matrix_to_dataset_stage(dataset_ref, matrix_ref, output_ref_name)])


def _prepare_dataset_rows_for_pca(
    dataset_block: Union[np.ndarray, np.ma.MaskedArray],
    bad_bands: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.ma.array(dataset_block, copy=False)
    raw = np.asarray(np.ma.getdata(arr), dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(f"Expected dataset block shape [y][x][b], got {raw.shape}")

    band_count = raw.shape[2]
    good_band_mask = np.ones((band_count,), dtype=bool)
    if bad_bands is not None:
        bad_bands_array = np.asarray(bad_bands)
        if bad_bands_array.shape != (band_count,):
            raise ValueError(
                f"Bad bands shape must match dataset band count: "
                f"bad_bands shape={bad_bands_array.shape}, bands={band_count}"
            )
        good_band_mask = bad_bands_array != 0

    flattened = raw.reshape(-1, band_count)[:, good_band_mask]
    if flattened.shape[1] == 0:
        raise ValueError("PCA cannot run because all bands are marked bad")

    flattened_mask = np.ma.getmaskarray(arr).reshape(-1, band_count)[:, good_band_mask]
    valid_rows = np.all(~flattened_mask, axis=1) & np.all(np.isfinite(flattened), axis=1)
    cleaned_rows = flattened[valid_rows, :]
    return cleaned_rows, good_band_mask


def _validate_prepared_pca_feature_count(
    *,
    flattened: np.ndarray,
    num_features: int,
) -> None:
    if flattened.ndim != 2:
        raise ValueError(f"Expected flattened PCA rows to be 2D, got {flattened.shape}")
    if flattened.shape[1] != num_features:
        raise ValueError(
            f"Prepared PCA feature count does not match requested num_features: "
            f"prepared_features={flattened.shape[1]}, requested={num_features}"
        )


def _expand_pca_outputs_to_full_bands(
    *,
    eigen_vectors_good: np.ndarray,
    covariance_good: np.ndarray,
    mean_good: np.ndarray,
    good_band_mask: np.ndarray,
    full_band_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigen_vectors = np.zeros((eigen_vectors_good.shape[0], full_band_count), dtype=np.float64)
    eigen_vectors[:, good_band_mask] = eigen_vectors_good

    covariance = np.zeros((full_band_count, full_band_count), dtype=np.float64)
    covariance[np.ix_(good_band_mask, good_band_mask)] = covariance_good

    mean = np.zeros((full_band_count,), dtype=np.float64)
    mean[good_band_mask] = mean_good
    return eigen_vectors, covariance, mean


def _pad_pca_eigen_outputs(
    *,
    eigen_vectors: np.ndarray,
    eigen_values: np.ndarray,
    num_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    if eigen_vectors.shape[0] != eigen_values.shape[0]:
        raise ValueError(
            f"Eigen vector/value count mismatch during PCA padding: "
            f"n_vectors={eigen_vectors.shape[0]}, n_values={eigen_values.shape[0]}"
        )
    if eigen_vectors.shape[0] > num_components:
        raise ValueError(
            f"Cannot pad PCA outputs when actual components exceed requested components: "
            f"actual={eigen_vectors.shape[0]}, requested={num_components}"
        )
    if eigen_vectors.shape[0] == num_components:
        return eigen_vectors, eigen_values

    padded_vectors = np.zeros((num_components, eigen_vectors.shape[1]), dtype=np.float64)
    padded_values = np.zeros((num_components,), dtype=np.float64)
    padded_vectors[: eigen_vectors.shape[0], :] = eigen_vectors
    padded_values[: eigen_values.shape[0]] = eigen_values
    return padded_vectors, padded_values


def _zero_small_pca_eigen_components(
    *,
    eigen_vectors: np.ndarray,
    eigen_values: np.ndarray,
    relative_cutoff: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    if eigen_vectors.shape[0] != eigen_values.shape[0]:
        raise ValueError(
            f"Eigen vector/value count mismatch during PCA cutoff: "
            f"n_vectors={eigen_vectors.shape[0]}, n_values={eigen_values.shape[0]}"
        )
    if eigen_values.size == 0:
        return eigen_vectors, eigen_values

    largest_eigen_value = float(np.max(eigen_values))
    if largest_eigen_value <= 0.0:
        zero_mask = np.ones_like(eigen_values, dtype=bool)
    else:
        zero_mask = eigen_values <= (largest_eigen_value * relative_cutoff)

    if not np.any(zero_mask):
        return eigen_vectors, eigen_values

    filtered_vectors = np.array(eigen_vectors, copy=True, dtype=np.float64)
    filtered_values = np.array(eigen_values, copy=True, dtype=np.float64)
    filtered_vectors[zero_mask, :] = 0.0
    filtered_values[zero_mask] = 0.0
    return filtered_vectors, filtered_values


def _resolve_num_components_payload(num_components_like: NumComponentsLike) -> int:
    client = get_process_storage_client()
    if isinstance(num_components_like, DataRef):
        payload = client.read_json_value(num_components_like)
        if not isinstance(payload, dict) or "num_components" not in payload:
            raise ValueError("Expected JSON num_components payload with key 'num_components'")
        return int(payload["num_components"])
    return int(num_components_like)


def _write_resolved_pca_num_components(
    input_ref: DataRef,
    full_input_region: DataRegion,
    resolved_num_components_ref: DataRef,
    requested_num_components: Optional[int],
    num_features: int,
) -> None:
    _ = full_input_region
    valid_pixels = count_valid_dataset_pixels(input_ref)
    max_components = min(num_features, max(0, valid_pixels - 1))
    if max_components <= 0:
        raise ValueError(
            f"PCA requires at least 2 valid samples and 1 valid feature; got "
            f"valid_pixels={valid_pixels}, num_features={num_features}"
        )

    if requested_num_components is None:
        resolved_num_components = max_components
    else:
        resolved_num_components = int(requested_num_components)
        if resolved_num_components <= 0 or resolved_num_components > max_components:
            raise ValueError(
                f"num_components must be in [1, {max_components}], got {resolved_num_components}"
            )

    client = get_process_storage_client()
    client.write_json_value(
        resolved_num_components_ref,
        {"num_components": int(resolved_num_components)},
    )


def _fit_dataset_pca_adaptive(
    input_ref: DataRef,
    input_region: DataRegion,
    output_info_ref: DataRef,
    output_vectors_ref: DataRef,
    output_values_ref: DataRef,
    output_covariance_ref: DataRef,
    output_mean_ref: DataRef,
    output_good_band_mask_ref: DataRef,
    num_components: NumComponentsLike,
    allocated_num_components: int,
    num_features: int,
    dataset_plan_meta: DatasetPlanMeta,
    test_full_pca: bool = True,
    data_variance_factor: int = 1,
) -> None:
    _ = input_region
    client = get_process_storage_client()
    num_components = _resolve_num_components_payload(num_components)
    bands = dataset_plan_meta.bands
    dataset_meta = client.get_meta(input_ref)
    bad_bands = dataset_meta.bad_bands
    dataset_size_bytes = (
        dataset_plan_meta.height
        * dataset_plan_meta.width
        * dataset_plan_meta.bands
        * dataset_plan_meta.dtype.itemsize
    )
    full_region = DatasetRegionRef(0, dataset_plan_meta.height, 0, dataset_plan_meta.width, 0, bands)

    if dataset_size_bytes <= PCA_MEMORY_CUTOFF_BYTES and test_full_pca:
        dataset, _ = client.read_region(input_ref, full_region)
        dataset_array = np.asarray(np.ma.getdata(dataset), dtype=np.float64)
        if dataset_array.ndim != 3:
            raise ValueError(f"Expected dataset shape [m][n][b], got {dataset_array.shape}")
        if dataset_array.shape[2] != bands:
            raise ValueError(
                f"Band mismatch in dataset for PCA: dataset_bands={dataset_array.shape[2]}, expected={bands}"
            )

        # Drops bad-band columns and nodata rows; `good_band_mask` keeps the
        # mapping back to original band positions for later expansion.
        flattened, good_band_mask = _prepare_dataset_rows_for_pca(dataset, bad_bands)
        if num_features != -1:
            _validate_prepared_pca_feature_count(flattened=flattened, num_features=num_features)
        total_rows = flattened.shape[0]
        if total_rows <= 1:
            raise ValueError("PCA requires at least 2 samples to derive eigen values")

        # sklearn caps at min(samples, features); we may end up with fewer
        # components than requested. The `_pad_pca_eigen_outputs` call below
        # squares this back up to the allocated row count.
        actual_components = min(num_components, flattened.shape[0], flattened.shape[1])
        pca = PCA(n_components=actual_components)
        pca.fit(flattened)
        eigen_values = np.asarray(pca.explained_variance_, dtype=np.float64)
        if num_features != -1:
            # Caller pre-stripped bad bands, so PCA's own outputs are already
            # in the canonical feature layout the rest of the pipeline expects.
            eigen_vectors = np.asarray(pca.components_, dtype=np.float64)
            covariance = np.asarray(pca.get_covariance(), dtype=np.float64)
            mean = np.asarray(pca.mean_, dtype=np.float64)
        else:
            # Caller passed full-band data; PCA fit only good-band columns, so
            # scatter the results back into full-band space (zeros at bad-band
            # indices)
            eigen_vectors, covariance, mean = _expand_pca_outputs_to_full_bands(
                eigen_vectors_good=np.asarray(pca.components_, dtype=np.float64),
                covariance_good=np.asarray(pca.get_covariance(), dtype=np.float64),
                mean_good=np.asarray(pca.mean_, dtype=np.float64),
                good_band_mask=good_band_mask,
                full_band_count=bands,
            )
        # Pad trailing rows with zeros so the array shape matches what the AllocationRequest reserved
        eigen_vectors, eigen_values = _pad_pca_eigen_outputs(
            eigen_vectors=eigen_vectors,
            eigen_values=eigen_values,
            num_components=allocated_num_components,
        )
    else:
        # This stage manages its own spatial reads instead of relying on the task system's
        # chunking policy, so the tile iteration stays next to the IPCA buffering logic.
        # The tile size is only chosen to guarantee at least num_components samples per
        # nominal tile, which is enough to form the first legal partial_fit batch.
        target_pixels = max(1, num_components)
        tile_h = max(1, min(dataset_plan_meta.height, int(np.sqrt(target_pixels))))
        tile_w = max(1, min(dataset_plan_meta.width, int(np.ceil(target_pixels / tile_h))))
        tile_scheme = SpatialTileScheme(tile_h=tile_h, tile_w=tile_w)

        total_rows = 0
        good_band_mask: Optional[np.ndarray] = None
        for tile_region in tile_scheme.iter_chunks(dataset_plan_meta):
            tile, _ = client.read_region(input_ref, tile_region)
            tile_array = np.asarray(np.ma.getdata(tile), dtype=np.float64)
            if tile_array.ndim != 3:
                raise ValueError(f"Expected dataset tile shape [m][n][b], got {tile_array.shape}")
            if tile_array.shape[2] != bands:
                raise ValueError(
                    f"Band mismatch in tile for IncrementalPCA: "
                    f"tile_bands={tile_array.shape[2]}, expected={bands}"
                )

            flattened, tile_good_band_mask = _prepare_dataset_rows_for_pca(tile, bad_bands)
            if num_features != -1:
                _validate_prepared_pca_feature_count(flattened=flattened, num_features=num_features)
            if good_band_mask is None:
                good_band_mask = tile_good_band_mask
            elif not np.array_equal(good_band_mask, tile_good_band_mask):
                raise ValueError("Bad-band mask changed between PCA tiles")

            total_rows += flattened.shape[0]

        if total_rows <= 1:
            raise ValueError("IncrementalPCA requires at least 2 samples to derive eigen values")
        if good_band_mask is None:
            raise ValueError("IncrementalPCA did not find any usable rows after filtering")

        actual_components = min(num_components, total_rows, int(np.count_nonzero(good_band_mask)))
        ipca = IncrementalPCA(n_components=actual_components)

        batch_buffer: list[np.ndarray] = []
        buffered_rows = 0
        first_fit_done = False
        for tile_region in tile_scheme.iter_chunks(dataset_plan_meta):
            tile, _ = client.read_region(input_ref, tile_region)
            flattened, tile_good_band_mask = _prepare_dataset_rows_for_pca(tile, bad_bands)
            if num_features != -1:
                _validate_prepared_pca_feature_count(flattened=flattened, num_features=num_features)
            if not np.array_equal(good_band_mask, tile_good_band_mask):
                raise ValueError("Bad-band mask changed between PCA tiles")
            if flattened.shape[0] == 0:
                continue

            batch_buffer.append(flattened)
            buffered_rows += flattened.shape[0]

            while buffered_rows >= actual_components:
                merged = np.concatenate(batch_buffer, axis=0)
                fit_batch = merged[:actual_components, :]
                remainder = merged[actual_components:, :]
                ipca.partial_fit(fit_batch)
                first_fit_done = True
                batch_buffer = [remainder] if remainder.size > 0 else []
                buffered_rows = remainder.shape[0] if remainder.size > 0 else 0

        if buffered_rows > 0:
            if not first_fit_done:
                raise ValueError(
                    f"Not enough samples to fit IncrementalPCA: "
                    f"samples={total_rows}, num_components={actual_components}"
                )
            merged = np.concatenate(batch_buffer, axis=0)
            ipca.partial_fit(merged)

        if not first_fit_done:
            raise ValueError(
                f"Not enough samples to fit IncrementalPCA: "
                f"samples={total_rows}, num_components={actual_components}"
            )

        singular_values = np.asarray(ipca.singular_values_, dtype=np.float64)
        eigen_values = (singular_values**2) / (total_rows - 1)
        if num_features != -1:
            eigen_vectors = np.asarray(ipca.components_, dtype=np.float64)
            covariance = np.asarray(ipca.get_covariance(), dtype=np.float64)
            mean = np.asarray(ipca.mean_, dtype=np.float64)
        else:
            eigen_vectors, covariance, mean = _expand_pca_outputs_to_full_bands(
                eigen_vectors_good=np.asarray(ipca.components_, dtype=np.float64),
                covariance_good=np.asarray(ipca.get_covariance(), dtype=np.float64),
                mean_good=np.asarray(ipca.mean_, dtype=np.float64),
                good_band_mask=good_band_mask,
                full_band_count=bands,
            )
        eigen_vectors, eigen_values = _pad_pca_eigen_outputs(
            eigen_vectors=eigen_vectors,
            eigen_values=eigen_values,
            num_components=allocated_num_components,
        )

    sort_desc = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sort_desc]
    eigen_vectors = eigen_vectors[sort_desc]
    eigen_vectors, eigen_values = _zero_small_pca_eigen_components(
        eigen_vectors=eigen_vectors,
        eigen_values=eigen_values,
    )

    client.write_data(output_vectors_ref, eigen_vectors)
    client.write_data(output_values_ref, eigen_values)  #  / data_variance_factor)
    client.write_data(output_covariance_ref, covariance)  # / data_variance_factor)
    client.write_data(output_mean_ref, mean)
    client.write_data(output_good_band_mask_ref, np.asarray(good_band_mask, dtype=np.bool_))
    descriptor = EigenVectorsAndValues(
        eigen_vectors_ref=output_vectors_ref,
        eigen_values_ref=output_values_ref,
        num_vectors=eigen_vectors.shape[0],
        vector_dimension=eigen_vectors.shape[1],
        covariance_ref=output_covariance_ref,
        mean_ref=output_mean_ref,
        good_band_mask_ref=output_good_band_mask_ref,
    )
    client.write_json_value(output_info_ref, {"eigen": descriptor})


@dataclass
class AdaptivePcaFitStage(SequentialStage):
    """
    Fit PCA over a dataset. If the dataset is 4GB or less we do a full PCA, if it is more than we
    iterate over spatial tiles using IncrementalPCA.

    The stage outputs an EigenVectorsAndValues JSON descriptor that references:
      - eigen vectors array [k][b]
      - eigen values array [k]
    where k = num_components.
    """

    _num_components: Optional[int] = 1
    _data_variance_factor: int = 1
    _output_ref_name: str = "ipca_eigenvectors_and_values"
    _vectors_ref_name: str = "ipca_eigen_vectors"
    _values_ref_name: str = "ipca_eigen_values"
    _covariance_ref_name: str = "ipca_covariance"
    _mean_ref_name: str = "ipca_mean"
    _good_band_mask_ref_name: str = "ipca_good_band_mask"
    _resolved_num_components_ref_name: str = "ipca_resolved_num_components"
    _dataset_plan_meta: Optional[DatasetPlanMeta] = None
    _num_features: int = -1
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = NoChunkingScheme
    test_full_pca: bool = True

    def __post_init__(self):
        # Include covariance/mean so planners and KEEP delete-policy overrides
        # see them (they are allocated in generate_allocation_requests).
        self.output_bindings = self.output_bindings + [
            DataBinding(self._vectors_ref_name),
            DataBinding(self._values_ref_name),
            DataBinding(self._covariance_ref_name),
            DataBinding(self._mean_ref_name),
            DataBinding(self._output_ref_name, kind="json"),
            DataBinding(self._good_band_mask_ref_name, kind="array"),
        ]
        self.broadcast_input |= {
            "ipca_vectors_ref": DataBinding(self._vectors_ref_name),
            "ipca_values_ref": DataBinding(self._values_ref_name),
            "ipca_covariance_ref": DataBinding(self._covariance_ref_name),
            "ipca_mean_ref": DataBinding(self._mean_ref_name),
            "ipca_good_band_mask_ref": DataBinding(self._good_band_mask_ref_name),
            "resolved_num_components_ref": DataBinding(self._resolved_num_components_ref_name),
            "dataset_plan_meta": self._dataset_plan_meta,
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
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for AdaptivePcaFitStage"
        feature_count = self._num_features if self._num_features != -1 else input_meta.bands
        if feature_count <= 0:
            raise ValueError(f"num_features must be positive when provided, got {self._num_features}")
        allocated_components = self._num_components if self._num_components is not None else feature_count
        if allocated_components <= 0:
            raise ValueError(f"num_components must be positive when provided, got {self._num_components}")
        if allocated_components > feature_count:
            raise ValueError(
                f"num_components must be <= available features, got num_components={allocated_components}, "
                f"features={feature_count}"
            )

        vectors_dtype = np.float64
        values_dtype = np.float64
        covariance_dtype = np.float64
        mean_dtype = np.float64
        mask_dtype = np.bool_
        vectors_size_est = allocated_components * feature_count * np.dtype(vectors_dtype).itemsize
        values_size_est = allocated_components * np.dtype(values_dtype).itemsize
        covariance_size_est = feature_count * feature_count * np.dtype(covariance_dtype).itemsize
        mean_size_est = feature_count * np.dtype(mean_dtype).itemsize
        good_band_mask_size_est = input_meta.bands * np.dtype(mask_dtype).itemsize

        return [
            AllocationRequest(
                name=self._vectors_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=vectors_size_est,
                shape=(allocated_components, feature_count),
                dtype=vectors_dtype,
                delete_policy=self.get_output_delete_policy(self._vectors_ref_name),
            ),
            AllocationRequest(
                name=self._values_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=values_size_est,
                shape=(allocated_components,),
                dtype=values_dtype,
                delete_policy=self.get_output_delete_policy(self._values_ref_name),
            ),
            AllocationRequest(
                name=self._covariance_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=covariance_size_est,
                shape=(feature_count, feature_count),
                dtype=covariance_dtype,
                delete_policy=self.get_output_delete_policy(self._covariance_ref_name),
            ),
            AllocationRequest(
                name=self._mean_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=mean_size_est,
                shape=(feature_count,),
                dtype=mean_dtype,
                delete_policy=self.get_output_delete_policy(self._mean_ref_name),
            ),
            AllocationRequest(
                name=self._good_band_mask_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=good_band_mask_size_est,
                shape=(input_meta.bands,),
                dtype=mask_dtype,
                delete_policy=self.get_output_delete_policy(self._good_band_mask_ref_name),
            ),
            AllocationRequest(
                name=self._resolved_num_components_ref_name,
                kind="json",
                residency="ram_cacheable",
                size_est=64,
                delete_policy=self.get_output_delete_policy(self._resolved_num_components_ref_name),
            ),
            AllocationRequest(
                name=self._output_ref_name,
                kind="json",
                residency="ram_cacheable",
                size_est=1024,
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            ),
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        output_vectors_ref: DataRef = broadcast_inputs["ipca_vectors_ref"]
        output_values_ref: DataRef = broadcast_inputs["ipca_values_ref"]
        output_covariance_ref: DataRef = broadcast_inputs["ipca_covariance_ref"]
        output_mean_ref: DataRef = broadcast_inputs["ipca_mean_ref"]
        output_good_band_mask_ref: DataRef = broadcast_inputs["ipca_good_band_mask_ref"]
        resolved_num_components_ref: DataRef = broadcast_inputs["resolved_num_components_ref"]
        dataset_plan_meta: DatasetPlanMeta = broadcast_inputs["dataset_plan_meta"]
        allocated_num_components = (
            self._num_components
            if self._num_components is not None
            else (self._num_features if self._num_features != -1 else dataset_plan_meta.bands)
        )
        return partial(
            _fit_dataset_pca_adaptive,
            input_ref,
            input_region,
            output_write.ref,
            output_vectors_ref,
            output_values_ref,
            output_covariance_ref,
            output_mean_ref,
            output_good_band_mask_ref,
            resolved_num_components_ref,
            int(allocated_num_components),
            self._num_features,
            dataset_plan_meta,
            self.test_full_pca,
            self._data_variance_factor,
        )

    def pre_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = output_writes
        resolved_num_components_ref: DataRef = broadcast_inputs["resolved_num_components_ref"]
        feature_count = self._num_features
        if feature_count == -1:
            dataset_plan_meta: DatasetPlanMeta = broadcast_inputs["dataset_plan_meta"]
            feature_count = dataset_plan_meta.bands
        return partial(
            _write_resolved_pca_num_components,
            input_ref,
            full_input_region,
            resolved_num_components_ref,
            self._num_components,
            int(feature_count),
        )


def get_adaptive_pca_partial_fit_stage(
    dataset_ref: DataRef,
    num_components: Optional[int],
    output_ref_name: str,
    num_features: int = -1,
) -> AdaptivePcaFitStage:
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    dataset_plan_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))
    num_samples = dataset_plan_meta.height * dataset_plan_meta.width
    max_components = min(dataset_plan_meta.bands, max(0, num_samples - 1))
    if num_components is not None and num_components > max_components:
        raise ValueError(
            f"num_components={num_components} exceeds max supported={max_components} "
            f"for shape={dataset_plan_meta.shape}"
        )

    return AdaptivePcaFitStage(
        _num_components=num_components,
        _output_ref_name=output_ref_name,
        _vectors_ref_name=f"{output_ref_name}_vectors",
        _values_ref_name=f"{output_ref_name}_values",
        _covariance_ref_name=f"{output_ref_name}_covariance",
        _mean_ref_name=f"{output_ref_name}_mean",
        _good_band_mask_ref_name=f"{output_ref_name}_good_band_mask",
        _resolved_num_components_ref_name=f"{output_ref_name}_resolved_num_components",
        _dataset_plan_meta=dataset_plan_meta,
        _num_features=num_features,
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=NoChunkingScheme,
    )


def get_adaptive_pca_partial_fit_pipeline(
    dataset_ref: DataRef,
    num_components: Optional[int],
    output_ref_name: str,
    num_features: int = -1,
) -> AlgorithmPipeline:
    return AlgorithmPipeline(
        [get_adaptive_pca_partial_fit_stage(dataset_ref, num_components, output_ref_name, num_features)]
    )


def _project_dataset_onto_eigenvectors(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    eigen_descriptor_ref: DataRef,
    spectral_mean_ref: Optional[DataRef],
    eigenvector_multiply_matrices: Sequence[DataRef],
    num_components: int,
) -> None:
    client = get_process_storage_client()
    data_tile, data_tile_meta = client.read_region(input_ref, input_region)
    envelope_payload = client.read_json_value(eigen_descriptor_ref)
    if not isinstance(envelope_payload, dict) or "eigen" not in envelope_payload:
        raise ValueError("Expected JSON payload with key 'eigen' for projection stage input")

    descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
    if not isinstance(descriptor, EigenVectorsAndValues):
        raise TypeError("Expected payload['eigen'] to be an EigenVectorsAndValues instance")

    eigen_vectors, _ = client.read_data(descriptor.eigen_vectors_ref)
    data_tile_array = np.ma.array(data_tile, copy=False)
    data_tile_raw = np.asarray(np.ma.getdata(data_tile_array), dtype=np.float64)
    data_tile_mask = np.ma.getmaskarray(data_tile_array)
    eigen_vectors_array = np.asarray(np.ma.getdata(eigen_vectors), dtype=np.float64)

    if data_tile_raw.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [m][n][b], got {data_tile_raw.shape}")
    if eigen_vectors_array.ndim != 2:
        raise ValueError(f"Expected eigen vectors shape [b][b], got {eigen_vectors_array.shape}")
    if num_components <= 0:
        raise ValueError(f"num_components must be positive, got {num_components}")

    bands = data_tile_raw.shape[2]
    good_band_mask = np.ones((bands,), dtype=np.bool_)
    if data_tile_meta.bad_bands is not None:
        bad_bands_array = np.asarray(data_tile_meta.bad_bands)
        if bad_bands_array.shape != (bands,):
            raise ValueError(
                f"Bad bands shape must match dataset tile bands: "
                f"bad_bands shape={bad_bands_array.shape}, tile_bands={bands}"
            )
        good_band_mask = bad_bands_array != 0
        if not np.any(good_band_mask):
            raise ValueError("Projection cannot run because all input bands are marked bad")

    filtered_data_tile = data_tile_raw[:, :, good_band_mask]
    filtered_data_tile_mask = data_tile_mask[:, :, good_band_mask]
    filtered_band_count = filtered_data_tile.shape[2]
    if eigen_vectors_array.shape[1] != filtered_band_count:
        raise ValueError(
            f"Band mismatch between filtered dataset tile and eigen vectors: "
            f"filtered_bands={filtered_band_count}, eigen_vector_dimension={eigen_vectors_array.shape[1]}"
        )
    if descriptor.vector_dimension != filtered_band_count:
        raise ValueError(
            f"Descriptor width must match filtered dataset tile bands: "
            f"descriptor_width={descriptor.vector_dimension}, filtered_bands={filtered_band_count}"
        )
    if num_components > eigen_vectors_array.shape[0]:
        raise ValueError(
            f"num_components exceeds available eigen vectors: "
            f"num_components={num_components}, available={eigen_vectors_array.shape[0]}"
        )

    projection_matrix = np.asarray(eigen_vectors_array[:num_components, :], dtype=np.float64)
    for i, matrix_ref in enumerate(eigenvector_multiply_matrices):
        matrix, _ = client.read_data(matrix_ref)
        matrix_array = np.asarray(np.ma.getdata(matrix), dtype=np.float64)
        if matrix_array.ndim == 3:
            if matrix_array.shape[-1] != 1:
                raise ValueError(
                    f"Expected projection matrix at index {i} to have shape [in][out] or [in][out][1], "
                    f"got {matrix_array.shape}"
                )
            matrix_array = np.squeeze(matrix_array, axis=2)
        if matrix_array.ndim != 2:
            raise ValueError(f"Expected projection matrix at index {i} to be 2D, got {matrix_array.shape}")
        if projection_matrix.shape[1] != matrix_array.shape[0]:
            raise ValueError(
                f"Projection matrix chain mismatch at index {i}: "
                f"eigen_matrix shape={projection_matrix.shape}, next matrix shape={matrix_array.shape}"
            )
        projection_matrix = projection_matrix @ matrix_array

    flattened = filtered_data_tile.reshape(-1, filtered_data_tile.shape[2])
    invalid_pixels = np.any(filtered_data_tile_mask, axis=2) | np.any(
        ~np.isfinite(filtered_data_tile), axis=2
    )
    if spectral_mean_ref is not None:
        spectral_mean, _ = client.read_data(spectral_mean_ref)
        spectral_mean_array = np.asarray(np.ma.getdata(spectral_mean), dtype=np.float64)
        if spectral_mean_array.ndim != 1:
            raise ValueError(f"Expected spectral mean shape [b], got {spectral_mean_array.shape}")
        if spectral_mean_array.shape[0] == bands:
            spectral_mean_array = spectral_mean_array[good_band_mask]
        elif spectral_mean_array.shape[0] != filtered_band_count:
            raise ValueError(
                f"Band mismatch between filtered dataset tile and spectral mean: "
                f"filtered_bands={filtered_band_count}, "
                f"spectral_mean_bands={spectral_mean_array.shape[0]}"
            )
        flattened = flattened - spectral_mean_array[np.newaxis, :]
    if flattened.shape[1] != projection_matrix.shape[1]:
        raise ValueError(
            f"Band mismatch between centered data and projection matrix: "
            f"data_bands={flattened.shape[1]}, projection_width={projection_matrix.shape[1]}"
        )
    projected_flattened = flattened @ projection_matrix.T
    projected_tile = projected_flattened.reshape(
        data_tile_raw.shape[0], data_tile_raw.shape[1], num_components
    )
    if data_tile_meta.nodata is not None and np.any(invalid_pixels):
        projected_tile[invalid_pixels, :] = data_tile_meta.nodata
    client.write_spec(output_write, projected_tile.astype(data_tile_raw.dtype, copy=False))


def _write_projected_dataset_meta(
    input_ref: DataRef,
    full_input_region: DataRegion,
    output_write: "WriteSpec",
) -> None:
    client = get_process_storage_client()
    input_region_meta = client.get_region_meta(input_ref, full_input_region)
    output_meta = client.get_meta(output_write.ref)
    projected_meta = replace(
        output_meta,
        elem_type=input_region_meta.elem_type,
        nodata=input_region_meta.nodata,
        bad_bands=None,
        wavelengths=None,
        wavelength_units=None,
        crs_wkt=input_region_meta.crs_wkt,
        geotransform=input_region_meta.geotransform,
    )
    client.write_meta(output_write.ref, projected_meta)


@dataclass
class ProjectOntoEigenVectorsStage(MapStage):
    """
    Project a [y][x][b] dataset onto the first k eigen vectors to produce [y][x][k].
    """

    _num_components: int = 1
    _output_ref_name: str = "projected_dataset"
    _eigen_descriptor_ref: Optional[DataRef] = None
    _spectral_mean_ref: Optional[DataRef] = None
    _eigenvector_multiply_matrices: Sequence[DataRef] = ()
    _eigenvector_multiply_matrix_names: Sequence[str] = ()
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self):
        if "eigen_descriptor_ref" not in self.broadcast_input:
            self.broadcast_input |= {"eigen_descriptor_ref": self._eigen_descriptor_ref}
        if "spectral_mean_ref" not in self.broadcast_input:
            self.broadcast_input |= {"spectral_mean_ref": self._spectral_mean_ref}
        if len(self._eigenvector_multiply_matrix_names) == 0:
            self._eigenvector_multiply_matrix_names = tuple(
                f"eigenvector_matrix_ref_{i}" for i in range(len(self._eigenvector_multiply_matrices))
            )
            for name, matrix_ref in zip(
                self._eigenvector_multiply_matrix_names,
                self._eigenvector_multiply_matrices,
            ):
                if name not in self.broadcast_input:
                    self.broadcast_input |= {name: matrix_ref}
        else:
            if len(self._eigenvector_multiply_matrices) > 0 and len(
                self._eigenvector_multiply_matrix_names
            ) != len(self._eigenvector_multiply_matrices):
                raise ValueError("projection matrix names must match projection matrix refs count")
            for name in self._eigenvector_multiply_matrix_names:
                if name not in self.broadcast_input:
                    raise ValueError(f"Missing broadcast input for projection matrix '{name}'")
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, DatasetRegionRef
        ), "Input region for ProjectOntoEigenVectorsStage must be DatasetRegionRef"
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=self._num_components,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for ProjectOntoEigenVectorsStage"
        if self._num_components <= 0:
            raise ValueError(f"num_components must be positive, got {self._num_components}")
        if self._num_components > input_meta.bands:
            raise ValueError(
                f"num_components must be <= input bands, got num_components={self._num_components}, "
                f"bands={input_meta.bands}"
            )

        size_est = input_meta.height * input_meta.width * self._num_components * input_meta.dtype.itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="dataset",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(input_meta.height, input_meta.width, self._num_components),
            dtype=input_meta.dtype,
            delete_policy=self.get_output_delete_policy(self._output_ref_name),
        )
        return [alloc_request]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        eigen_descriptor_ref: DataRef = broadcast_inputs["eigen_descriptor_ref"]
        spectral_mean_ref: Optional[DataRef] = broadcast_inputs["spectral_mean_ref"]
        eigenvector_multiply_matrices = [
            broadcast_inputs[name] for name in self._eigenvector_multiply_matrix_names
        ]
        return partial(
            _project_dataset_onto_eigenvectors,
            input_ref,
            input_region,
            output_write,
            eigen_descriptor_ref,
            spectral_mean_ref,
            eigenvector_multiply_matrices,
            self._num_components,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_write_projected_dataset_meta, input_ref, full_input_region, output_write)


def get_project_onto_eigenvectors_stage(
    dataset_ref: DataRef,
    eigen_descriptor_ref: DataRef,
    num_components: int,
    output_ref_name: str,
    eigenvector_multiply_matrices: Sequence[DataRef] = (),
) -> ProjectOntoEigenVectorsStage:
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")
    if num_components <= 0:
        raise ValueError(f"num_components must be positive, got {num_components}")
    if num_components > dataset_meta.shape[2]:
        raise ValueError(
            f"num_components must be <= input bands, got num_components={num_components}, "
            f"bands={dataset_meta.shape[2]}"
        )

    envelope_payload = storage_client.read_json_value(eigen_descriptor_ref)
    if not isinstance(envelope_payload, dict) or "eigen" not in envelope_payload:
        raise ValueError("Expected JSON payload with key 'eigen' for projection stage input")
    descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
    if not isinstance(descriptor, EigenVectorsAndValues):
        raise TypeError("Expected payload['eigen'] to be an EigenVectorsAndValues instance")
    if num_components > descriptor.num_vectors:
        raise ValueError(
            f"num_components exceeds available eigen vectors: "
            f"num_components={num_components}, available={descriptor.num_vectors}"
        )
    expected_input_width = dataset_meta.shape[2]
    if dataset_meta.bad_bands is not None:
        bad_bands_array = np.asarray(dataset_meta.bad_bands)
        if bad_bands_array.shape != (dataset_meta.shape[2],):
            raise ValueError(
                f"Dataset bad bands shape must match input bands: "
                f"bad_bands shape={bad_bands_array.shape}, bands={dataset_meta.shape[2]}"
            )
        expected_input_width = int(np.count_nonzero(bad_bands_array != 0))
    if descriptor.vector_dimension != expected_input_width:
        raise ValueError(
            f"Eigen vector dimension must match filtered input bands: "
            f"vector_dimension={descriptor.vector_dimension}, expected={expected_input_width}"
        )

    input_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))
    return ProjectOntoEigenVectorsStage(
        _num_components=num_components,
        _output_ref_name=output_ref_name,
        _eigen_descriptor_ref=eigen_descriptor_ref,
        _eigenvector_multiply_matrices=tuple(eigenvector_multiply_matrices),
        default_executor="process",
        input_plan_meta=input_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=SpatialTileScheme,
    )


def get_project_onto_eigenvectors_pipeline(
    dataset_ref: DataRef,
    eigen_descriptor_ref: DataRef,
    num_components: int,
    output_ref_name: str,
    eigenvector_multiply_matrices: Sequence[DataRef] = (),
) -> AlgorithmPipeline:
    return AlgorithmPipeline(
        [
            get_project_onto_eigenvectors_stage(
                dataset_ref,
                eigen_descriptor_ref,
                num_components,
                output_ref_name,
                eigenvector_multiply_matrices,
            )
        ]
    )


# ---------------------------------------------------------------------------
# Positive Semi-Definite Matrix Inverse (SVD pseudoinverse)
# ---------------------------------------------------------------------------


def _compute_psd_matrix_inverse(
    input_ref: DataRef,
    input_region: DataRegion,
    output_ref: DataRef,
    rcond: Optional[float],
) -> None:
    """Worker: compute the Moore-Penrose pseudoinverse of a PSD matrix via SVD.

    Singular values smaller than ``rcond * sigma_max`` are treated as zero,
    making this a numerically stable pseudoinverse for rank-deficient or
    near-singular matrices.

    Args:
        input_ref: DataRef for the input square PSD matrix of shape ``(N, N)``.
        input_region: Unused; present for framework compatibility.
        output_ref: DataRef to write the pseudoinverse result into.
        rcond: Relative threshold for singular-value truncation.  Singular
            values ``s[i] < rcond * s[0]`` are treated as zero.  When
            ``None``, defaults to ``eps * N`` where ``eps`` is the machine
            epsilon for float64 and ``N`` is the matrix side length.

    Raises:
        ValueError: If the input is not a 2-D square matrix.
    """
    _ = input_region
    client = get_process_storage_client()

    raw, _ = client.read_data(input_ref)
    matrix = np.asarray(np.ma.getdata(raw), dtype=np.float64)

    if matrix.ndim == 3 and matrix.shape[2] == 1:
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D square matrix, got shape {matrix.shape}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix for pseudoinverse, got shape {matrix.shape}")

    n = matrix.shape[0]
    effective_rcond = rcond if rcond is not None else np.finfo(np.float64).eps * n

    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

    # Truncate singular values below the threshold
    threshold = effective_rcond * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)

    # Pseudoinverse: V @ diag(s_inv) @ U^T
    pseudoinverse = (Vh.T * s_inv) @ U.T
    client.write_data(output_ref, pseudoinverse.astype(np.float64, copy=False))


@dataclass
class PosSemiDefMatrixInverse(SequentialStage):
    """Task stage that computes the pseudoinverse of a positive semi-definite matrix.

    Uses a truncated Singular Value Decomposition (SVD) to compute the
    Moore-Penrose pseudoinverse, which is numerically stable for rank-deficient
    or near-singular matrices.

    Given a PSD matrix ``M`` with SVD ``M = U @ diag(s) @ Vᴴ``, the
    pseudoinverse is:

    .. code-block:: text

        M⁺ = V @ diag(s_inv) @ Uᵀ

    where ``s_inv[i] = 1 / s[i]`` when ``s[i] > rcond * s[0]``, and
    ``s_inv[i] = 0`` otherwise.  Singular values below the threshold
    correspond to numerically zero directions and are left uninverted.

    The stage reads the full matrix in one shot (``NoChunkingScheme``) and
    writes a float64 result of the same shape.

    Attributes:
        _output_ref_name: Allocation name for the pseudoinverse output array.
            Defaults to ``"psd_matrix_inverse"``.
        _rcond: Relative singular-value threshold.  Singular values smaller
            than ``rcond * sigma_max`` are treated as zero.  When ``None``
            (default), the threshold is set to ``eps * N`` at runtime, where
            ``eps`` is the float64 machine epsilon and ``N`` is the matrix
            side length.

    Note:
        The input must be registered as an array ``DataRef`` with shape
        ``(N, N)``.  The stage validates squareness in the factory function
        ``get_pos_semi_def_matrix_inverse_stage`` before construction.  Use
        that factory rather than instantiating this class directly.
    """

    _output_ref_name: str = "psd_matrix_inverse"
    _rcond: Optional[float] = None
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=8,
            bytes_per_scalar_out=0,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = NoChunkingScheme

    def __post_init__(self) -> None:
        self.output_bindings = list(self.output_bindings) + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, SpectraBatchRef
        ), "Input region for PosSemiDefMatrixInverse must be SpectraBatchRef"
        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        """Allocate a single ``(N, N)`` float64 output array.

        Args:
            input_meta: Must be a ``SpectraListPlanMeta`` describing the
                ``(N, N)`` input matrix.
            chosen_scheme: Unused; present for interface compatibility.

        Returns:
            A one-element list containing the ``AllocationRequest`` for the
            pseudoinverse output.

        Raises:
            AssertionError: If ``input_meta`` is not a ``SpectraListPlanMeta``.
        """
        _ = chosen_scheme
        assert isinstance(
            input_meta, SpectraListPlanMeta
        ), "PosSemiDefMatrixInverse requires SpectraListPlanMeta input_meta"
        n = input_meta.num_spectra
        size_est = n * n * np.dtype(np.float64).itemsize
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=size_est,
                shape=(n, n),
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        """Return the worker callable for this stage.

        Args:
            input_ref: DataRef for the input PSD matrix.
            input_region: Region descriptor (unused for ``NoChunkingScheme``).
            output_writes: Mapping from output name to ``WriteSpec``.
            broadcast_inputs: Unused; present for interface compatibility.

        Returns:
            A ``partial`` wrapping ``_compute_psd_matrix_inverse``.
        """
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(
            _compute_psd_matrix_inverse,
            input_ref,
            input_region,
            output_write.ref,
            self._rcond,
        )


def get_pos_semi_def_matrix_inverse_stage(
    matrix_ref: DataRef,
    output_ref_name: str = "psd_matrix_inverse",
    rcond: Optional[float] = None,
) -> PosSemiDefMatrixInverse:
    """Build a :class:`PosSemiDefMatrixInverse` stage from a matrix ``DataRef``.

    Validates that the input is a 2-D square array before constructing the
    stage, so any shape errors surface at planning time rather than at runtime.

    Args:
        matrix_ref: DataRef for the input PSD matrix of shape ``(N, N)``.
        output_ref_name: Allocation name for the pseudoinverse output.
            Defaults to ``"psd_matrix_inverse"``.
        rcond: Relative singular-value truncation threshold.  Singular values
            smaller than ``rcond * sigma_max`` are treated as zero.  Pass
            ``None`` to use the automatic default ``eps * N``.

    Returns:
        A fully configured :class:`PosSemiDefMatrixInverse` stage.

    Raises:
        ValueError: If the input ``DataRef`` does not describe a 2-D square
            array.
    """
    storage_client = get_process_storage_client()
    matrix_meta = storage_client.get_meta(matrix_ref)

    shape = matrix_meta.shape
    if len(shape) == 3 and shape[2] == 1:
        shape = shape[:2]
    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D square matrix for pseudoinverse, got shape {matrix_meta.shape}")
    if shape[0] != shape[1]:
        raise ValueError(f"Expected a square matrix for pseudoinverse, got shape {matrix_meta.shape}")

    n = int(shape[0])
    input_meta = SpectraListPlanMeta(
        num_spectra=n,
        spectrum_length=n,
        dtype=np.dtype(matrix_meta.elem_type),
    )
    return PosSemiDefMatrixInverse(
        _output_ref_name=output_ref_name,
        _rcond=rcond,
        default_executor="process",
        input_plan_meta=input_meta,
        chunking_scheme_type=NoChunkingScheme,
    )


def get_pos_semi_def_matrix_inverse_pipeline(
    matrix_ref: DataRef,
    output_ref_name: str = "psd_matrix_inverse",
    rcond: Optional[float] = None,
) -> AlgorithmPipeline:
    """Return a single-stage pipeline that pseudoinverts a PSD matrix.

    Args:
        matrix_ref: DataRef for the input PSD matrix of shape ``(N, N)``.
        output_ref_name: Allocation name for the pseudoinverse output.
            Defaults to ``"psd_matrix_inverse"``.
        rcond: Relative singular-value truncation threshold passed through to
            :func:`get_pos_semi_def_matrix_inverse_stage`.  ``None`` uses the
            automatic default.

    Returns:
        An :class:`AlgorithmPipeline` containing a single
        :class:`PosSemiDefMatrixInverse` stage.
    """
    return AlgorithmPipeline([get_pos_semi_def_matrix_inverse_stage(matrix_ref, output_ref_name, rcond)])

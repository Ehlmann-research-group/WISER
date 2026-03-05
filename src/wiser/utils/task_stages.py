from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional
import numpy as np
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.utils.primitives import (
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


# region Task Stage utilities


def _running_covariance(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    mean_ref: DataRef,
    total: int,
) -> None:
    client = get_process_storage_client()
    output_ref = output_write.ref
    running_cov, _ = client.read_data(output_ref)
    noise, _ = client.read_region(input_ref, input_region)
    mean_arr, _ = client.read_data(mean_ref)
    assert noise.ndim == 3, "noise should have 3 dimensions"
    assert mean_arr.ndim == 1, "mean_arr should have 1 dimension"
    mean_arr = mean_arr[np.newaxis, np.newaxis, :]
    mean_centered_noise = noise - mean_arr
    flattened_noise = mean_centered_noise.reshape(-1, mean_centered_noise.shape[2])
    sum_outer_product = flattened_noise.T @ flattened_noise
    partial_cov_matrix = sum_outer_product / (total - 1)
    partial_cov_matrix = partial_cov_matrix[:, :, np.newaxis]
    running_cov += partial_cov_matrix
    client.write_data(output_ref, running_cov)


@dataclass
class CalcCovMatrixStage(SequentialStage):
    """
    Calculates the covariance matrix of a noise matrix. This
    assumes the data has been mean subtracted. The noise matrix
    is assumed to be of shape [y][x][b] where [y][x] are the
    pixel axis and we want the noise of [b].
    """

    # You must override this
    _total_spectra: int = 0
    # You must define this
    _output_ref_name: str = "cov_running"
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
    chunking_scheme_type = SpatialTileScheme

    def __post_init__(self):
        if "mean" not in self.broadcast_input:
            self.broadcast_input |= {"mean": self._mean_ref}
        self.broadcast_input |= {"total": self._total_spectra}
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

        size_est = input_meta.bands * input_meta.bands * input_meta.dtype.itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="array",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(input_meta.bands, input_meta.bands, 1),
            dtype=input_meta.dtype,
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
        total = broadcast_inputs["total"]
        mean: DataRef = broadcast_inputs["mean"]
        return partial(
            _running_covariance,
            input_ref,
            input_region,
            output_write,
            mean,
            total,
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


def _running_mean(input_ref: DataRef, input_region: DataRegion, output_write: "WriteSpec", total) -> None:
    client = get_process_storage_client()
    output_ref = output_write.ref
    running_mean, _ = client.read_data(output_ref)
    data, _ = client.read_region(input_ref, input_region)
    spectra_sum: np.ndarray = data.sum(axis=(0, 1)) / total
    running_mean += spectra_sum
    client.write_data(output_ref, running_mean)


@dataclass
class SpectralMeanStage(SequentialStage):
    """
    Expects the variable 'total' to be in broadcast inputs with its type
    """

    # You should override this
    _output_ref_name: str = "spectral_mean_1"

    def __post_init__(self):
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

        np_type = np.float32

        size_est = input_meta.bands * np.dtype(np_type).itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="spectrum",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(input_meta.bands,),
            dtype=np.dtype(np_type),
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
        total = broadcast_inputs["total"]
        return partial(_running_mean, input_ref, input_region, output_write, total)


def get_spectral_mean_stage(dataset_ref: DataRef, output_ref_name: str) -> SpectralMeanStage:
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=data_meta.elem_type)
    stage = SpectralMeanStage(
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
        broadcast_input={"total": plan_meta.height * plan_meta.width},
        output_bindings=[DataBinding(output_ref_name)],
    )
    return stage


@dataclass(frozen=True)
class EigenVectorsAndValues:
    """
    Lightweight descriptor that references storage-backed eigen outputs.

    This object intentionally stores only reference IDs and shape metadata so it
    can be serialized to JSON cheaply and passed through task outputs.
    """

    eigen_vectors_ref: DataRef
    eigen_values_ref: DataRef
    num_vectors: int
    vector_dimension: int

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
    eigen_values = np.asarray(eigen_values[sort_desc], dtype=np.float32)
    eigen_vectors = np.asarray(eigen_vectors[:, sort_desc].T, dtype=np.float32)

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
class EigendecompositionStage(SequentialStage):
    """
    Compute eigendecomposition for a square [N][N] matrix and persist:
      - eigen vectors in an array [N][N],
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
    chunking_scheme_type = NoChunkingScheme

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name, kind="json")]
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
        ), "input_meta must be of type SpectraListPlanMeta for EigendecompositionStage"
        if input_meta.num_spectra != input_meta.spectrum_length:
            raise ValueError(
                f"EigendecompositionStage expects a square matrix, got shape="
                f"({input_meta.num_spectra}, {input_meta.spectrum_length})"
            )

        n = input_meta.num_spectra
        vectors_dtype = np.float32
        values_dtype = np.float32
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
            ),
            AllocationRequest(
                name=self._values_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=values_size_est,
                shape=(n,),
                dtype=values_dtype,
            ),
            AllocationRequest(
                name=self._output_ref_name,
                kind="json",
                residency="ram_cacheable",
                size_est=1024,
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
) -> EigendecompositionStage:
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
    return EigendecompositionStage(
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
    eigen_vectors_array = np.asarray(np.ma.getdata(eigen_vectors), dtype=np.float32)
    eigen_values_array = np.asarray(np.ma.getdata(eigen_values), dtype=np.float32)

    if eigen_vectors_array.ndim != 2:
        raise ValueError(f"Expected eigen vectors with 2D shape [n][d], got {eigen_vectors_array.shape}")
    if eigen_values_array.ndim != 1:
        raise ValueError(f"Expected eigen values with 1D shape [n], got {eigen_values_array.shape}")
    if eigen_vectors_array.shape[0] != eigen_values_array.shape[0]:
        raise ValueError(
            f"Eigen vector/value count mismatch: n_vectors={eigen_vectors_array.shape[0]}, "
            f"n_values={eigen_values_array.shape[0]}"
        )

    inverse_sqrt_eigen_values = np.zeros_like(eigen_values_array, dtype=np.float32)
    assert (eigen_values_array > 0).all(), "All eigen values of a covariance matrix should be positive"
    inverse_sqrt_eigen_values = 1.0 / np.sqrt(eigen_values_array)
    whitening_matrix = inverse_sqrt_eigen_values[:, np.newaxis] * eigen_vectors_array
    client.write_data(output_ref, whitening_matrix.astype(np.float32, copy=False))


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
    chunking_scheme_type = NoChunkingScheme

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

        dtype = np.float32
        size_est = input_meta.num_spectra * input_meta.spectrum_length * np.dtype(dtype).itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="array",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(input_meta.num_spectra, input_meta.spectrum_length),
            dtype=dtype,
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
        dtype=np.dtype(np.float32),
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

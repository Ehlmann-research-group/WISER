from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
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
    MapStage,
    ResourceModel,
    SequentialStage,
    WriteSpec,
)
from wiser.utils.worker_runtime import get_process_storage_client

PCA_MEMORY_CUTOFF_BYTES = 4 * 1024**3

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
    if np.ma.isMaskedArray(noise):
        noise = np.ma.getdata(noise)
    if np.ma.isMaskedArray(mean_arr):
        mean_arr = np.ma.getdata(mean_arr)
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
    Calculates the covariance matrix of a dataset. This assumes
    the data has not been mean subtracted. The input_ref data
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
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

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

    Eigen vectors should be in decreasing order of eigen value from left to right
    Each row should have an eigen vector
    """

    eigen_vectors_ref: DataRef
    eigen_values_ref: DataRef
    num_vectors: int
    vector_dimension: int
    covariance_ref: Optional[DataRef] = None
    mean_ref: Optional[DataRef] = None

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
    # We transpose because np.linalg.eig gives us eigen vectors in the columns, but
    # we want them in the rows
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
    chunking_scheme_type: type[ChunkingScheme] = NoChunkingScheme

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

    if np.any(eigen_values_array < 0):
        raise ValueError("Whitening matrix cannot be computed: one or more eigen values are negative")

    inverse_sqrt_eigen_values = np.diag(1.0 / np.sqrt(eigen_values_array))
    whitening_matrix = eigen_vectors_array.T @ inverse_sqrt_eigen_values @ eigen_vectors_array
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
        matrix_array = np.asarray(np.ma.getdata(matrix), dtype=np.float32)
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
    client.write_data(output_ref, product.astype(np.float32, copy=False))


@dataclass
class MatrixMultiplicationStage(SequentialStage):
    """
    Multiply a list of matrices in order: [A, B, C] -> A @ B @ C.
    """

    _output_ref_name: str = "matrix_product"
    _matrix_refs: Optional[Sequence[DataRef]] = None
    _matrix_input_names: Sequence[str] = ()
    _output_shape: Optional[tuple[int, int]] = None
    _output_dtype: np.dtype = np.dtype(np.float32)
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
        _output_dtype=np.dtype(np.float32),
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
    matrix_ref: DataRef,
) -> None:
    client = get_process_storage_client()
    # Expected shape: (y, x, b)
    data_tile, _ = client.read_region(input_ref, input_region)
    # Expected shape: (k, b)
    matrix, _ = client.read_data(matrix_ref)

    data_tile_array = np.asarray(np.ma.getdata(data_tile))
    matrix_array = np.asarray(np.ma.getdata(matrix))

    if data_tile_array.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [m][n][b], got {data_tile_array.shape}")
    if matrix_array.ndim != 2:
        raise ValueError(f"Expected matrix shape [k][b], got {matrix_array.shape}")

    bands = data_tile_array.shape[2]
    if matrix_array.shape[1] != bands:
        raise ValueError(
            f"Band mismatch between dataset tile and matrix: "
            f"tile_bands={bands}, matrix_width={matrix_array.shape[1]}"
        )

    flattened = data_tile_array.reshape(-1, bands)
    transformed_flattened = flattened @ matrix_array.T
    transformed_tile = transformed_flattened.reshape(
        data_tile_array.shape[0], data_tile_array.shape[1], matrix_array.shape[0]
    )
    client.write_spec(output_write, transformed_tile.astype(data_tile_array.dtype, copy=False))


@dataclass
class ApplyMatrixToDatasetStage(MapStage):
    """
    Apply a [b][b] matrix to each spectrum in a [y][x][b] dataset.
    """

    _output_ref_name: str = "matrix_applied_dataset"
    _matrix_ref: Optional[DataRef] = None
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
        if "matrix_ref" not in self.broadcast_input:
            self.broadcast_input |= {"matrix_ref": self._matrix_ref}
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
        matrix_ref: DataRef = broadcast_inputs["matrix_ref"]
        return partial(
            _apply_matrix_to_dataset,
            input_ref,
            input_region,
            output_write,
            matrix_ref,
        )


def get_apply_matrix_to_dataset_stage(
    dataset_ref: DataRef,
    matrix_ref: DataRef,
    output_ref_name: str,
) -> ApplyMatrixToDatasetStage:
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    matrix_meta = storage_client.get_meta(matrix_ref)

    if len(data_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {data_meta.shape}")
    if len(matrix_meta.shape) != 2:
        raise ValueError(f"Expected matrix shape [k][b], got {matrix_meta.shape}")
    if data_meta.shape[2] != matrix_meta.shape[1]:
        raise ValueError(
            f"Band mismatch: dataset bands={data_meta.shape[2]}, " f"matrix width={matrix_meta.shape[1]}"
        )

    input_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=np.dtype(data_meta.elem_type))
    return ApplyMatrixToDatasetStage(
        _output_ref_name=output_ref_name,
        _matrix_ref=matrix_ref,
        _output_bands=matrix_meta.shape[0],
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


def _fit_dataset_pca_adaptive(
    input_ref: DataRef,
    input_region: DataRegion,
    output_info_ref: DataRef,
    output_vectors_ref: DataRef,
    output_values_ref: DataRef,
    output_covariance_ref: DataRef,
    output_mean_ref: DataRef,
    num_components: int,
    dataset_plan_meta: DatasetPlanMeta,
    test_full_pca: bool = True,
    data_variance_factor: int = 1,
) -> None:
    _ = input_region
    client = get_process_storage_client()
    bands = dataset_plan_meta.bands
    dataset_size_bytes = (
        dataset_plan_meta.height
        * dataset_plan_meta.width
        * dataset_plan_meta.bands
        * dataset_plan_meta.dtype.itemsize
    )
    full_region = DatasetRegionRef(0, dataset_plan_meta.height, 0, dataset_plan_meta.width, 0, bands)

    if dataset_size_bytes <= PCA_MEMORY_CUTOFF_BYTES and test_full_pca:
        pca = PCA(n_components=num_components)
        dataset, _ = client.read_region(input_ref, full_region)
        dataset_array = np.asarray(np.ma.getdata(dataset), dtype=np.float32)
        if dataset_array.ndim != 3:
            raise ValueError(f"Expected dataset shape [m][n][b], got {dataset_array.shape}")
        if dataset_array.shape[2] != bands:
            raise ValueError(
                f"Band mismatch in dataset for PCA: dataset_bands={dataset_array.shape[2]}, expected={bands}"
            )

        flattened = dataset_array.reshape(-1, bands)
        total_rows = flattened.shape[0]
        if total_rows <= 1:
            raise ValueError("PCA requires at least 2 samples to derive eigen values")

        pca.fit(flattened)
        eigen_values = np.asarray(pca.explained_variance_, dtype=np.float32)
        eigen_vectors = np.asarray(pca.components_, dtype=np.float32)
        covariance = np.asarray(pca.get_covariance(), dtype=np.float32)
        mean = np.asarray(pca.mean_, dtype=np.float32)
    else:
        ipca = IncrementalPCA(n_components=num_components)

        # This stage manages its own spatial reads instead of relying on the task system's
        # chunking policy, so the tile iteration stays next to the IPCA buffering logic.
        # The tile size is only chosen to guarantee at least num_components samples per
        # nominal tile, which is enough to form the first legal partial_fit batch.
        target_pixels = max(1, num_components)
        tile_h = max(1, min(dataset_plan_meta.height, int(np.sqrt(target_pixels))))
        tile_w = max(1, min(dataset_plan_meta.width, int(np.ceil(target_pixels / tile_h))))
        tile_scheme = SpatialTileScheme(tile_h=tile_h, tile_w=tile_w)

        batch_buffer: list[np.ndarray] = []
        buffered_rows = 0
        first_fit_done = False
        total_rows = 0
        for tile_region in tile_scheme.iter_chunks(dataset_plan_meta):
            tile, _ = client.read_region(input_ref, tile_region)
            tile_array = np.asarray(np.ma.getdata(tile), dtype=np.float32)
            if tile_array.ndim != 3:
                raise ValueError(f"Expected dataset tile shape [m][n][b], got {tile_array.shape}")
            if tile_array.shape[2] != bands:
                raise ValueError(
                    f"Band mismatch in tile for IncrementalPCA: "
                    f"tile_bands={tile_array.shape[2]}, expected={bands}"
                )

            flattened = tile_array.reshape(-1, bands)
            total_rows += flattened.shape[0]

            batch_buffer.append(flattened)
            buffered_rows += flattened.shape[0]

            while buffered_rows >= num_components:
                merged = np.concatenate(batch_buffer, axis=0)
                fit_batch = merged[:num_components, :]
                remainder = merged[num_components:, :]
                ipca.partial_fit(fit_batch)
                first_fit_done = True
                batch_buffer = [remainder] if remainder.size > 0 else []
                buffered_rows = remainder.shape[0] if remainder.size > 0 else 0

        if buffered_rows > 0:
            if not first_fit_done:
                raise ValueError(
                    f"Not enough samples to fit IncrementalPCA: "
                    f"samples={total_rows}, num_components={num_components}"
                )
            merged = np.concatenate(batch_buffer, axis=0)
            ipca.partial_fit(merged)

        if not first_fit_done:
            raise ValueError(
                f"Not enough samples to fit IncrementalPCA: "
                f"samples={total_rows}, num_components={num_components}"
            )

        if total_rows <= 1:
            raise ValueError("IncrementalPCA requires at least 2 samples to derive eigen values")

        singular_values = np.asarray(ipca.singular_values_, dtype=np.float32)
        eigen_values = (singular_values**2) / (total_rows - 1)
        eigen_vectors = np.asarray(ipca.components_, dtype=np.float32)
        covariance = np.asarray(ipca.get_covariance(), dtype=np.float32)
        mean = np.asarray(ipca.mean_, dtype=np.float32)

    sort_desc = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sort_desc]
    eigen_vectors = eigen_vectors[sort_desc]

    client.write_data(output_vectors_ref, eigen_vectors)
    client.write_data(output_values_ref, eigen_values / data_variance_factor)
    client.write_data(output_covariance_ref, covariance / data_variance_factor)
    client.write_data(output_mean_ref, mean)
    descriptor = EigenVectorsAndValues(
        eigen_vectors_ref=output_vectors_ref,
        eigen_values_ref=output_values_ref,
        num_vectors=eigen_vectors.shape[0],
        vector_dimension=eigen_vectors.shape[1],
        covariance_ref=output_covariance_ref,
        mean_ref=output_mean_ref,
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

    _num_components: int = 1
    _data_variance_factor: int = 1
    _output_ref_name: str = "ipca_eigenvectors_and_values"
    _vectors_ref_name: str = "ipca_eigen_vectors"
    _values_ref_name: str = "ipca_eigen_values"
    _covariance_ref_name: str = "ipca_covariance"
    _mean_ref_name: str = "ipca_mean"
    _dataset_plan_meta: Optional[DatasetPlanMeta] = None
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
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name, kind="json")]
        self.broadcast_input |= {
            "ipca_vectors_ref": DataBinding(self._vectors_ref_name),
            "ipca_values_ref": DataBinding(self._values_ref_name),
            "ipca_covariance_ref": DataBinding(self._covariance_ref_name),
            "ipca_mean_ref": DataBinding(self._mean_ref_name),
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
        if self._num_components <= 0:
            raise ValueError(f"num_components must be positive, got {self._num_components}")
        if self._num_components > input_meta.bands:
            raise ValueError(
                f"num_components must be <= input bands, got num_components={self._num_components}, "
                f"bands={input_meta.bands}"
            )

        vectors_dtype = np.float32
        values_dtype = np.float32
        covariance_dtype = np.float32
        mean_dtype = np.float32
        vectors_size_est = self._num_components * input_meta.bands * np.dtype(vectors_dtype).itemsize
        values_size_est = self._num_components * np.dtype(values_dtype).itemsize
        covariance_size_est = input_meta.bands * input_meta.bands * np.dtype(covariance_dtype).itemsize
        mean_size_est = input_meta.bands * np.dtype(mean_dtype).itemsize

        return [
            AllocationRequest(
                name=self._vectors_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=vectors_size_est,
                shape=(self._num_components, input_meta.bands),
                dtype=vectors_dtype,
            ),
            AllocationRequest(
                name=self._values_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=values_size_est,
                shape=(self._num_components,),
                dtype=values_dtype,
            ),
            AllocationRequest(
                name=self._covariance_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=covariance_size_est,
                shape=(input_meta.bands, input_meta.bands),
                dtype=covariance_dtype,
            ),
            AllocationRequest(
                name=self._mean_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=mean_size_est,
                shape=(input_meta.bands,),
                dtype=mean_dtype,
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
        output_vectors_ref: DataRef = broadcast_inputs["ipca_vectors_ref"]
        output_values_ref: DataRef = broadcast_inputs["ipca_values_ref"]
        output_covariance_ref: DataRef = broadcast_inputs["ipca_covariance_ref"]
        output_mean_ref: DataRef = broadcast_inputs["ipca_mean_ref"]
        dataset_plan_meta: DatasetPlanMeta = broadcast_inputs["dataset_plan_meta"]
        return partial(
            _fit_dataset_pca_adaptive,
            input_ref,
            input_region,
            output_write.ref,
            output_vectors_ref,
            output_values_ref,
            output_covariance_ref,
            output_mean_ref,
            self._num_components,
            dataset_plan_meta,
            self.test_full_pca,
            self._data_variance_factor,
        )


def get_adaptive_pca_partial_fit_stage(
    dataset_ref: DataRef,
    num_components: int,
    output_ref_name: str,
) -> AdaptivePcaFitStage:
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    dataset_plan_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))
    num_samples = dataset_plan_meta.height * dataset_plan_meta.width
    max_components = min(dataset_plan_meta.bands, max(0, num_samples - 1))
    if num_components > max_components:
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
        _dataset_plan_meta=dataset_plan_meta,
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
    num_components: int,
    output_ref_name: str,
) -> AlgorithmPipeline:
    return AlgorithmPipeline(
        [get_adaptive_pca_partial_fit_stage(dataset_ref, num_components, output_ref_name)]
    )


def _project_dataset_onto_eigenvectors(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    eigen_descriptor_ref: DataRef,
    spectral_mean_ref: Optional[DataRef],
    num_components: int,
) -> None:
    client = get_process_storage_client()
    data_tile, _ = client.read_region(input_ref, input_region)
    envelope_payload = client.read_json_value(eigen_descriptor_ref)
    if not isinstance(envelope_payload, dict) or "eigen" not in envelope_payload:
        raise ValueError("Expected JSON payload with key 'eigen' for projection stage input")

    descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
    if not isinstance(descriptor, EigenVectorsAndValues):
        raise TypeError("Expected payload['eigen'] to be an EigenVectorsAndValues instance")

    eigen_vectors, _ = client.read_data(descriptor.eigen_vectors_ref)
    data_tile_array = np.asarray(np.ma.getdata(data_tile))
    eigen_vectors_array = np.asarray(np.ma.getdata(eigen_vectors), dtype=np.float32)

    if data_tile_array.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [m][n][b], got {data_tile_array.shape}")
    if eigen_vectors_array.ndim != 2:
        raise ValueError(f"Expected eigen vectors shape [b][b], got {eigen_vectors_array.shape}")
    if num_components <= 0:
        raise ValueError(f"num_components must be positive, got {num_components}")

    bands = data_tile_array.shape[2]
    if eigen_vectors_array.shape[1] != bands:
        raise ValueError(
            f"Band mismatch between dataset tile and eigen vectors: "
            f"tile_bands={bands}, eigen_vector_dimension={eigen_vectors_array.shape[1]}"
        )
    if num_components > eigen_vectors_array.shape[0]:
        raise ValueError(
            f"num_components exceeds available eigen vectors: "
            f"num_components={num_components}, available={eigen_vectors_array.shape[0]}"
        )

    top_components = eigen_vectors_array[:num_components, :]
    flattened = data_tile_array.reshape(-1, bands)
    if spectral_mean_ref is not None:
        spectral_mean, _ = client.read_data(spectral_mean_ref)
        spectral_mean_array = np.asarray(np.ma.getdata(spectral_mean), dtype=np.float32)
        if spectral_mean_array.ndim != 1:
            raise ValueError(f"Expected spectral mean shape [b], got {spectral_mean_array.shape}")
        if spectral_mean_array.shape[0] != bands:
            raise ValueError(
                f"Band mismatch between dataset tile and spectral mean: "
                f"tile_bands={bands}, spectral_mean_bands={spectral_mean_array.shape[0]}"
            )
        flattened = flattened - spectral_mean_array[np.newaxis, :]
    projected_flattened = flattened @ top_components.T
    projected_tile = projected_flattened.reshape(
        data_tile_array.shape[0], data_tile_array.shape[1], num_components
    )
    client.write_spec(output_write, projected_tile.astype(data_tile_array.dtype, copy=False))


@dataclass
class ProjectOntoEigenVectorsStage(MapStage):
    """
    Project a [y][x][b] dataset onto the first k eigen vectors to produce [y][x][k].
    """

    _num_components: int = 1
    _output_ref_name: str = "projected_dataset"
    _eigen_descriptor_ref: Optional[DataRef] = None
    _spectral_mean_ref: Optional[DataRef] = None
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
        return partial(
            _project_dataset_onto_eigenvectors,
            input_ref,
            input_region,
            output_write,
            eigen_descriptor_ref,
            spectral_mean_ref,
            self._num_components,
        )


def get_project_onto_eigenvectors_stage(
    dataset_ref: DataRef,
    eigen_descriptor_ref: DataRef,
    num_components: int,
    output_ref_name: str,
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
    if descriptor.vector_dimension != dataset_meta.shape[2]:
        raise ValueError(
            f"Eigen vector dimension must match input bands: "
            f"vector_dimension={descriptor.vector_dimension}, bands={dataset_meta.shape[2]}"
        )

    input_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))
    return ProjectOntoEigenVectorsStage(
        _num_components=num_components,
        _output_ref_name=output_ref_name,
        _eigen_descriptor_ref=eigen_descriptor_ref,
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
) -> AlgorithmPipeline:
    return AlgorithmPipeline(
        [
            get_project_onto_eigenvectors_stage(
                dataset_ref,
                eigen_descriptor_ref,
                num_components,
                output_ref_name,
            )
        ]
    )

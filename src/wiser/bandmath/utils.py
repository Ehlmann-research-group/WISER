from typing import Any, List, Tuple, Union, TYPE_CHECKING, Optional, Dict, Awaitable

import re
import datetime
import os

import numpy as np

import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

import lark
import psutil
from lark import Tree

from osgeo import gdal

from enum import Enum
from dataclasses import dataclass

from wiser.raster.spectrum import NumPyArraySpectrum

from wiser.raster.serializable import SerializedForm

from wiser.gui.parallel_task import ParallelTaskProcess

from .types import VariableType
from wiser.raster.dataset import (
    RasterDataBand,
    RasterDataSet,
    SaveState,
    RasterDataDynamicBand,
)
from wiser.raster.spectrum import Spectrum
from .builtins.constants import (
    RATIO_OF_MEM_TO_USE,
    DEFAULT_IGNORE_VALUE,
    LHS_KEY,
    RHS_KEY,
)

from PySide2.QtWidgets import QMessageBox, QWidget

import logging

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.loader import RasterDataLoader
    from .types import BandMathExprInfo, BandMathValue

BandMathResultInfo = Tuple[VariableType, SerializedForm, str, "BandMathExprInfo"]


logger = logging.getLogger(__name__)

Number = Union[int, float]
Scalar = Union[int, float, bool]

TEMP_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_output")


class MathOperations(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    COMPARE = "compare"
    POWER = "power"
    TRIG_FUNCTION = "trig_function"
    DOT_PRODUCT = "dot_product"
    GENERAL = "general"


def bandmath_progress_callback(msg: str):
    print(f"Bandmath progress:\n{msg}")


def bandmath_error_callback(task: ParallelTaskProcess):
    print(f"Task error:\n{task.get_error()}")


def load_image_from_bandmath_result(
    result_type: Union[VariableType, RasterDataSet],
    result: SerializedForm,
    result_name: str,
    expression: Optional[str],
    expr_info: "BandMathExprInfo",
    loader: "RasterDataLoader" = None,
    app_state: "ApplicationState" = None,
) -> RasterDataSet:
    # Compute a timestamp to put in the description
    timestamp = datetime.datetime.now().isoformat()
    # Below result type occurs if the dataset was so big it need to be saved to disk
    if result_type == VariableType.IMAGE_CUBE_DATASET:
        metadata = result.get_metadata()
        if "save_state" not in metadata:
            metadata["save_state"] = SaveState.IN_DISK_NOT_SAVED
        ds = RasterDataSet.deserialize_into_class(result)
        if expression:
            ds.set_description(f"Computed image-cube:  {expression} ({timestamp})")
        if expr_info.spatial_metadata_source:
            ds.copy_spatial_metadata(expr_info.spatial_metadata_source)
        if expr_info.spectral_metadata_source:
            ds.copy_spectral_metadata(expr_info.spectral_metadata_source)
        if app_state:
            ds.set_name(app_state.unique_dataset_name(metadata.get("name", "Computed")))
            app_state.add_dataset(ds, view_dataset=False)
        return ds
    elif result_type == VariableType.IMAGE_CUBE:
        # In the case where the user just enters a variable (ex: "a"), then
        # we will get a serializedForm object back. We then make result be
        # the array of the value.
        if isinstance(result, SerializedForm):
            ds: RasterDataSet = result.get_serializable_class().deserialize_into_class(result)
        elif isinstance(result, np.ndarray):
            if not loader:
                raise AttributeError("Must pass loader into function if result_type is IMAGE_CUBE")
            cache = app_state.get_cache() if app_state else None
            ds = loader.dataset_from_numpy_array(result, cache)
        else:
            raise TypeError("The image result is neither a np.ndarray or SerializedForm")
        name = result_name if result_name else "Computed"
        ds.set_description(f"Computed image-cube:  {expression} ({timestamp})")

        if expr_info.spatial_metadata_source:
            ds.copy_spatial_metadata(expr_info.spatial_metadata_source)
        if expr_info.spectral_metadata_source:
            ds.copy_spectral_metadata(expr_info.spectral_metadata_source)
        if app_state:
            ds.set_name(app_state.unique_dataset_name(name))
            app_state.add_dataset(ds, view_dataset=False)

        return ds
    return None


def save_image_from_bandmath_result(
    result_type: Union[VariableType],
    result: SerializedForm,
    result_name: str,
    expression: Optional[str],
    expr_info: "BandMathExprInfo",
    output_folder: str,
    loader: "RasterDataLoader" = None,
    app_state: "ApplicationState" = None,
) -> RasterDataSet:
    # Compute a timestamp to put in the description
    timestamp = datetime.datetime.now().isoformat()
    save_path = os.path.join(output_folder, result_name)
    # Below result type occurs if the dataset was so big it need to be saved to disk
    if result_type == VariableType.IMAGE_CUBE_DATASET:
        metadata = result.get_metadata()
        if "save_state" not in metadata:
            metadata["save_state"] = SaveState.IN_DISK_NOT_SAVED
        ds = RasterDataSet.deserialize_into_class(result)
        if expression:
            ds.set_description(f"Computed image-cube:  {expression} ({timestamp})")
        if expr_info.spatial_metadata_source:
            ds.copy_spatial_metadata(expr_info.spatial_metadata_source)
        if expr_info.spectral_metadata_source:
            ds.copy_spectral_metadata(expr_info.spectral_metadata_source)
        if loader:
            loader.save_dataset_as(ds, save_path, format="ENVI")

        return ds
    elif result_type == VariableType.IMAGE_CUBE:
        if not loader:
            raise AttributeError("Must pass loader into function if result_type is IMAGE_CUBE")
        cache = app_state.get_cache() if app_state else None
        name = result_name if result_name else "Computed"
        ds = loader.dataset_from_numpy_array(result, cache)
        ds.set_description(f"Computed image-cube:  {expression} ({timestamp})")
        ds.set_name(name)
        if expr_info.spatial_metadata_source:
            ds.copy_spatial_metadata(expr_info.spatial_metadata_source)
        if expr_info.spectral_metadata_source:
            ds.copy_spectral_metadata(expr_info.spectral_metadata_source)
        if loader:
            loader.save_dataset_as(ds, save_path, format="ENVI", config=None)

        return ds
    return None


def load_band_from_bandmath_result(
    result: SerializedForm,
    result_name: str,
    expression: str,
    expr_info: "BandMathExprInfo",
    parent: QWidget = None,
    loader: "RasterDataLoader" = None,
    app_state: "ApplicationState" = None,
) -> RasterDataSet:
    if not loader:
        raise AttributeError("Must pass loader into function")
    # Compute a timestamp to put in the description
    timestamp = datetime.datetime.now().isoformat()
    if isinstance(result, SerializedForm):
        generic_raster_band_class = result.get_serializable_class()
        raster_band: RasterDataBand = generic_raster_band_class.deserialize_into_class(
            result,
        )
        assert isinstance(raster_band, (RasterDataDynamicBand, RasterDataBand))
        arr = raster_band.get_data()
        if arr.ndim == 2:
            arr = arr[np.newaxis, :]
            cache = app_state.get_cache() if app_state else None
            new_dataset = loader.dataset_from_numpy_array(arr, cache)
    elif isinstance(result, np.ndarray):
        # Convert the image band into a 1-band image cube
        result = result[np.newaxis, :]
        cache = app_state.get_cache() if app_state else None
        new_dataset = loader.dataset_from_numpy_array(result, cache)
    else:
        raise RuntimeError(
            f"Expected result to be Serialized Form or np.ndarray but instead got: {type(result)}"
        )

    if not result_name:
        result_name = parent.tr("Computed") if parent else "Computed"

    if app_state:
        new_dataset.set_name(app_state.unique_dataset_name(result_name))
    new_dataset.set_description(f"Computed image-band:  {expression} ({timestamp})")

    if expr_info.spatial_metadata_source:
        new_dataset.copy_spatial_metadata(expr_info.spatial_metadata_source)

    if app_state:
        app_state.add_dataset(new_dataset, view_dataset=False)

    return new_dataset


def save_band_from_bandmath_result(
    result: SerializedForm,
    result_name: str,
    expression: str,
    expr_info: "BandMathExprInfo",
    output_folder: str,
    parent: QWidget = None,
    loader: "RasterDataLoader" = None,
    app_state: "ApplicationState" = None,
) -> RasterDataSet:
    if not loader:
        raise AttributeError("Must pass loader into function")
    if not app_state:
        raise AttributeError("Must pass app_state into function")
    save_path = os.path.join(output_folder, result_name)
    # Compute a timestamp to put in the description
    timestamp = datetime.datetime.now().isoformat()
    if isinstance(result, SerializedForm):
        generic_raster_band_class = result.get_serializable_class()
        raster_band: RasterDataBand = generic_raster_band_class.deserialize_into_class(
            result,
        )
        assert isinstance(raster_band, (RasterDataDynamicBand, RasterDataBand))
        arr = raster_band.get_data()
        if arr.ndim == 2:
            arr = arr[np.newaxis, :]
            cache = app_state.get_cache() if app_state else None
            new_dataset = loader.dataset_from_numpy_array(result, cache)
    elif isinstance(result, (np.ndarray, np.ma.MaskedArray)):
        # Convert the image band into a 1-band image cube
        result = result[np.newaxis, :]
        cache = app_state.get_cache() if app_state else None
        new_dataset = loader.dataset_from_numpy_array(result, cache)
    else:
        raise RuntimeError(
            f"Expected result to be Serialized Form or np.ndarray but instead got: {type(result)}"
        )

    if not result_name:
        result_name = parent.tr("Computed") if parent else "Computed"

    if app_state:
        new_dataset.set_name(app_state.unique_dataset_name(result_name))
        new_dataset.set_id(app_state.take_next_id())

    new_dataset.set_description(f"Computed image-band:  {expression} ({timestamp})")

    if expr_info.spatial_metadata_source:
        new_dataset.copy_spatial_metadata(expr_info.spatial_metadata_source)

    loader.save_dataset_as(new_dataset, save_path, format="ENVI", config=None)

    return new_dataset


def load_spectrum_from_bandmath_result(
    result: Union[SerializedForm, np.ndarray],
    result_name: str,
    expression: str,
    expr_info: "BandMathExprInfo",
    timestamp: str,
    parent: QWidget = None,
    app_state: "ApplicationState" = None,
) -> Spectrum:
    if not result_name:
        result_name = parent.tr("Computed:  {expression} ({timestamp})")
        result_name = result_name.format(expression=expression, timestamp=timestamp)
    if isinstance(result, SerializedForm):
        new_spectrum = result.get_serializable_class().deserialize_into_class(result)
    elif isinstance(result, np.ndarray):
        new_spectrum = NumPyArraySpectrum(result, name=result_name)
    else:
        raise TypeError("The spectrum result is neither a np.ndarray or SerializedForm")

    if expr_info.spectral_metadata_source:
        new_spectrum.copy_spectral_metadata(expr_info.spectral_metadata_source)

    if app_state is not None:
        app_state.set_active_spectrum(new_spectrum)

    return new_spectrum


def bandmath_success_callback(
    parent: QWidget,
    app_state: "ApplicationState",
    results: List[BandMathResultInfo],
    expression: str,
    batch_enabled: bool,
    load_into_wiser: bool,
    output_folder: str = "",
):
    # If the process gets cancelled, the results will be None. So we do nothing.
    if not results:
        return
    try:
        # This call back is called both when batch is enabled or disabled, so we must make
        # sure we load results into WISER if batch is disabled.
        if load_into_wiser or not batch_enabled:
            for result_type, result, result_name, expr_info in results:
                logger.debug("Result of band-math evaluation is type " + f"{result_type}, value:\n{result}")

                # Compute a timestamp to put in the description
                timestamp = datetime.datetime.now().isoformat()

                loader = app_state.get_loader()
                if result_type == VariableType.IMAGE_CUBE_DATASET or result_type == VariableType.IMAGE_CUBE:
                    load_image_from_bandmath_result(
                        result_type=result_type,
                        result=result,
                        result_name=result_name,
                        expression=expression,
                        expr_info=expr_info,
                        loader=loader,
                        app_state=app_state,
                    )

                elif result_type == VariableType.IMAGE_BAND:
                    load_band_from_bandmath_result(
                        result=result,
                        result_name=result_name,
                        expression=expression,
                        expr_info=expr_info,
                        parent=parent,
                        loader=loader,
                        app_state=app_state,
                    )

                elif result_type == VariableType.SPECTRUM:
                    load_spectrum_from_bandmath_result(
                        result=result,
                        result_name=result_name,
                        expression=expression,
                        expr_info=expr_info,
                        timestamp=timestamp,
                        parent=parent,
                        app_state=app_state,
                    )
        if output_folder:
            for result_type, result, result_name, expr_info in results:
                logger.debug("Result of band-math evaluation is type " + f"{result_type}, value:\n{result}")

                # Compute a timestamp to put in the description
                timestamp = datetime.datetime.now().isoformat()

                loader = app_state.get_loader()
                if result_type == VariableType.IMAGE_CUBE_DATASET or result_type == VariableType.IMAGE_CUBE:
                    save_image_from_bandmath_result(
                        result_type=result_type,
                        result=result,
                        result_name=result_name,
                        expression=expression,
                        expr_info=expr_info,
                        loader=loader,
                        app_state=app_state,
                        output_folder=output_folder,
                    )

                elif result_type == VariableType.IMAGE_BAND:
                    save_band_from_bandmath_result(
                        result=result,
                        result_name=result_name,
                        expression=expression,
                        expr_info=expr_info,
                        loader=loader,
                        app_state=app_state,
                        output_folder=output_folder,
                    )

    except Exception as e:
        logger.exception("Couldn't evaluate band-math expression")
        QMessageBox.critical(
            parent,
            parent.tr("Bandmath Evaluation Error"),
            parent.tr("Couldn't evaluate band-math expression")
            + f"\n{expression}\n"
            + parent.tr("Reason:")
            + f"\n{e}",
        )
        return


def get_result_dtype(
    dtype1: np.dtype,
    dtype2: np.dtype,
    operation: MathOperations = MathOperations.GENERAL,
) -> np.dtype:
    """
    Determines the resulting NumPy dtype after performing a specified mathematical operation
    between two input dtypes.

    Parameters:
    - dtype1 (np.dtype): The dtype of the first operand.
    - dtype2 (np.dtype): The dtype of the second operand.
    - operation (MathOperations): The mathematical operation to perform.

    Returns:
    - np.dtype: The resulting dtype after the operation.

    Raises:
    - ValueError: If an unsupported operation is provided.
    """
    if dtype1 is None and dtype2 is not None:
        return dtype2

    elif dtype1 is not None and dtype2 is None:
        return dtype1

    elif operation in {
        MathOperations.ADD,
        MathOperations.SUBTRACT,
        MathOperations.MULTIPLY,
        MathOperations.DIVIDE,
        MathOperations.POWER,
        MathOperations.DOT_PRODUCT,
        MathOperations.GENERAL,
    }:
        # For arithmetic operations, use NumPy's type promotion rules
        return np.result_type(dtype1, dtype2)

    elif operation == MathOperations.COMPARE:
        # Comparison operations yield boolean results
        return np.bool_

    elif operation == MathOperations.TRIG_FUNCTION:
        # Trigonometric functions typically return floating-point types
        if np.issubdtype(dtype1, np.floating):
            return dtype1  # Retain the floating type if already floating
        else:
            # Promote to a higher precision floating type if input is integer
            return np.float32

    else:
        raise ValueError(f"Unsupported operation: {operation}")


def get_valid_ignore_value(dataset: gdal.Dataset, default_ignore_value: float):
    """
    Determines an appropriate data ignore value for a GDAL dataset based on its data type.

    Parameters:
    - dataset: GDAL dataset object
    - default_ignore_value: The default data ignore value to use if it fits into the dataset's data type

    Returns:
    - A data ignore value that fits into the dataset's data type
    """
    # Get the data type of the first band
    band = dataset.GetRasterBand(1)
    gdal_dtype = band.DataType

    # Map GDAL data types to NumPy data types
    gdal_dtype_to_numpy_dtype = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }

    # Check if the GDAL data type is supported
    if gdal_dtype not in gdal_dtype_to_numpy_dtype:
        raise ValueError(f"Unsupported GDAL data type: {gdal.GetDataTypeName(gdal_dtype)}")

    numpy_dtype = gdal_dtype_to_numpy_dtype[gdal_dtype]

    # Get the min and max values for the data type
    if np.issubdtype(numpy_dtype, np.integer):
        type_info = np.iinfo(numpy_dtype)
    elif np.issubdtype(numpy_dtype, np.floating):
        type_info = np.finfo(numpy_dtype)
    else:
        raise ValueError(f"Unsupported NumPy data type: {numpy_dtype}")

    min_value = type_info.min
    max_value = type_info.max

    # Check if the default ignore value fits into the data type
    if min_value <= default_ignore_value <= max_value:
        # Ensure the type of default_ignore_value matches the data type
        if (np.issubdtype(numpy_dtype, np.integer) and isinstance(default_ignore_value, int)) or (
            np.issubdtype(numpy_dtype, np.floating) and isinstance(default_ignore_value, (int, float))
        ):
            return default_ignore_value

    # Return the minimum value for the data type if the default ignore value doesn't fit
    return min_value


def pop_future(
    q: queue.Queue,
    enqueue_fn,  # callable that enqueues (future, meta...) into q
):
    """Pop and return the next future from a queue (index 0), enqueuing one first if the queue is empty."""
    if q.empty():
        enqueue_fn()
    return q.get()[0]


def maybe_prefetch_next(should_read_next: bool, prefetch_fn):
    """Invoke a prefetch function only if the next chunk should be read."""
    if should_read_next:
        prefetch_fn()


@dataclass(frozen=True)
class ReadPlan:
    shape: tuple
    value: Optional[np.ndarray] = None
    future: Optional[Awaitable[np.ndarray]] = None

    def __post_init__(self):
        if self.value is None and self.future is None:
            raise ValueError("Both self.value and self.future are None! One must exist!")


def band_subset_shape(lhs: "BandMathValue", band_list: List[int]) -> tuple:
    """Return the expected array shape for a band-subset without reading the data."""
    lhs_value_shape = list(lhs.get_shape())
    lhs_value_shape[0] = len(band_list)
    lhs_value_shape = tuple(lhs_value_shape)
    return lhs_value_shape


def plan_lhs_read(
    lhs: "BandMathValue",
    index_list_current: List[int],
    index_list_next: List[int],
    read_task_queue: Dict[str, queue.Queue],
    read_thread_pool: ThreadPoolExecutor,
    event_loop: asyncio.AbstractEventLoop,
) -> Tuple[ReadPlan, bool]:
    """Plan how to obtain LHS data for the current band chunk (and optionally prefetch the next).

    This function creates a :class:`ReadPlan` describing how the left-hand-side (LHS) operand
    should be produced for a band-math evaluation over a subset of bands.

    - If ``lhs.value`` is already a NumPy array, the requested band subset is computed
      immediately and returned as a value-based plan (no asynchronous I/O).
    - Otherwise, the function ensures a read task exists for ``index_list_current``, pops the
      resulting awaitable from the LHS task queue, and returns a future-based plan. If
      ``should_continue_reading_bands(index_list_next, lhs)`` is True, it also schedules a
      prefetch read for the next band chunk.

    Args:
        lhs: The left-hand-side band-math operand.
        index_list_current: Band indices for the current chunk to be evaluated.
        index_list_next: Band indices for the next chunk to prefetch (if enabled).
        read_task_queue: Mapping of queue keys (e.g., ``LHS_KEY``) to per-operand task queues.
            The LHS queue is expected to store items whose first element is an awaitable/future
            yielding a NumPy array for a requested band subset.
        read_thread_pool: Thread pool used to execute blocking dataset reads.
        event_loop: Event loop used when scheduling dataset reads and creating futures.

    Returns:
        A tuple ``(lhs_plan, should_read_next)`` where:
        - ``lhs_plan`` is a :class:`ReadPlan` with ``shape`` set to the expected shape of the
          current band subset and either ``value`` (immediate) or ``future`` (I/O-backed) set.
        - ``should_read_next`` indicates whether a prefetch for ``index_list_next`` was enqueued.

    Raises:
        KeyError: If ``LHS_KEY`` is not present in ``read_task_queue``.
        Exception: Any exception related to queue operations or task scheduling may be raised
            immediately; exceptions from dataset reads will surface when the returned future is
            awaited.
    """
    shape = band_subset_shape(lhs, index_list_current)

    # If lhs is a numpy array, it is already fully loaded into memory, so we don't
    # do any async io
    if isinstance(lhs.value, np.ndarray):
        return ReadPlan(value=lhs.as_numpy_array_by_bands(index_list_current), shape=shape), False

    def enqueue_lhs_future():
        read_lhs_future_onto_queue(
            lhs,
            index_list_current,
            event_loop,
            read_thread_pool,
            read_task_queue[LHS_KEY],
        )

    # Pop the future from the queue. If queue is empty, enqueue future
    lhs_future = pop_future(read_task_queue[LHS_KEY], enqueue_lhs_future)

    should_read_next = should_continue_reading_bands(index_list_next, lhs)

    # If we should, read the next bands into memory and put the future onto the queue
    if should_read_next:
        read_lhs_future_onto_queue(
            lhs,
            index_list_next,
            event_loop,
            read_thread_pool,
            read_task_queue[LHS_KEY],
        )

    return ReadPlan(future=lhs_future, shape=shape), should_read_next


def plan_rhs_read(
    rhs: "BandMathValue",
    lhs: "BandMathValue",
    lhs_read_plan: ReadPlan,
    should_read_next: bool,
    index_list_current: List[int],
    index_list_next: List[int],
    read_task_queue: Dict[str, queue.Queue],
    read_thread_pool: ThreadPoolExecutor,
    event_loop: asyncio.AbstractEventLoop,
) -> ReadPlan:
    """Plan how to obtain RHS data for the current band chunk (and optionally prefetch the next).

    This function creates a :class:`ReadPlan` describing how the RHS operand should be produced
    for a band-math operation given the LHS read plan.

    - If the RHS does not require asynchronous I/O (e.g., it is not an image-cube/dataset operand
      or the LHS is already in-memory), the RHS value is constructed immediately via
      :func:`make_image_cube_compatible_by_bands` and returned as a value-based plan.
    - Otherwise, this function ensures a RHS read task exists for the current band chunk,
      pops the resulting future from the RHS task queue, and returns a future-based plan.
      If ``should_read_next`` is True, it also enqueues a prefetch read for the next band chunk.

    Args:
        rhs: The right-hand-side band-math operand.
        lhs: The left-hand-side band-math operand (used to compute the next-chunk shape).
        lhs_read_plan: The read plan produced for the LHS operand for the current chunk.
            Its ``shape`` is used as the target shape for making RHS compatible.
        should_read_next: Whether to prefetch the RHS for ``index_list_next``.
        index_list_current: Band indices for the current chunk.
        index_list_next: Band indices for the next chunk to prefetch (if enabled).
        read_task_queue: Mapping of queue keys (e.g., ``RHS_KEY``) to per-operand task queues.
            The RHS queue is expected to contain items whose first element is an awaitable/future
            yielding a NumPy array.
        read_thread_pool: Thread pool used for scheduling dataset reads.
        event_loop: Event loop used to create/attach asynchronous read futures.

    Returns:
        A :class:`ReadPlan` for the RHS operand. The returned plan will have:
        - ``shape`` set to ``lhs_read_plan.shape``.
        - either ``value`` set (if computed immediately), or ``future`` set (if I/O is required).

    Raises:
        KeyError: If ``RHS_KEY`` is not present in ``read_task_queue``.
        Exception: Propagates any exception raised by the underlying queue/future creation
            utilities when the returned future is later awaited.
    """
    # If the lhs is an array, then it is already fully loaded into memory so we just do a regular
    # subset. Or if the rhs is not an image cube we also do a regular subset
    if isinstance(lhs.value, np.ndarray) or (
        rhs.type not in (VariableType.IMAGE_CUBE, VariableType.IMAGE_CUBE_DATASET)
    ):
        return ReadPlan(
            value=make_image_cube_compatible_by_bands(rhs, lhs_read_plan.shape, index_list_current),
            shape=lhs_read_plan.shape,
        )

    def enqueue_rhs_future():
        read_rhs_future_onto_queue(
            rhs,
            lhs_read_plan.shape,
            index_list_current,
            event_loop,
            read_thread_pool,
            read_task_queue[RHS_KEY],
        )

    # Pop the rhs future off the queue. If there is no future on the queue, put one there
    rhs_future = pop_future(read_task_queue[RHS_KEY], enqueue_rhs_future)

    # If we should, read another future onto the queue
    if should_read_next:
        next_lhs_shape = band_subset_shape(lhs, index_list_next)
        read_rhs_future_onto_queue(
            rhs,
            next_lhs_shape,
            index_list_next,
            event_loop,
            read_thread_pool,
            read_task_queue[RHS_KEY],
        )

    return ReadPlan(future=rhs_future, shape=lhs_read_plan.shape)


async def resolve_read_plan(read_plan: ReadPlan):
    if read_plan.value is not None:
        return read_plan.value
    return await read_plan.future


async def get_lhs_rhs_values_async(
    lhs: "BandMathValue",
    rhs: "BandMathValue",
    index_list_current: List[int],
    index_list_next: List[int],
    read_task_queue: Dict[str, queue.Queue],
    read_thread_pool: ThreadPoolExecutor,
    event_loop: asyncio.AbstractEventLoop,
):
    lhs_read_plan, read_next = plan_lhs_read(
        lhs=lhs,
        index_list_current=index_list_current,
        index_list_next=index_list_next,
        read_task_queue=read_task_queue,
        read_thread_pool=read_thread_pool,
        event_loop=event_loop,
    )

    rhs_read_plan = plan_rhs_read(
        rhs=rhs,
        lhs=lhs,
        lhs_read_plan=lhs_read_plan,
        should_read_next=read_next,
        index_list_current=index_list_current,
        index_list_next=index_list_next,
        read_task_queue=read_task_queue,
        read_thread_pool=read_thread_pool,
        event_loop=event_loop,
    )

    return await resolve_read_plan(lhs_read_plan), await resolve_read_plan(rhs_read_plan)


async def get_lhs_value_async(
    lhs: "BandMathValue",
    index_list_current: List[int],
    index_list_next: List[int],
    read_task_queue: queue.Queue,
    read_thread_pool: ThreadPoolExecutor,
    event_loop: asyncio.AbstractEventLoop,
):
    """Return the LHS value for the current band chunk, prefetching the next chunk if needed."""
    lhs_plan, _ = plan_lhs_read(
        lhs=lhs,
        index_list_current=index_list_current,
        index_list_next=index_list_next,
        read_task_queue=read_task_queue,
        read_thread_pool=read_thread_pool,
        event_loop=event_loop,
    )
    return await resolve_read_plan(lhs_plan)


def get_lhs_rhs_values(lhs: "BandMathValue", rhs: "BandMathValue", index_list: List[int]):
    rhs_value = None
    same_datasets = False
    lhs_value = lhs.as_numpy_array_by_bands(index_list)

    # Get the rhs value from the queue. If there isn't one on the queue we put one on the queue and wait

    if (
        isinstance(lhs.value, RasterDataSet)
        and isinstance(rhs.value, RasterDataSet)
        and lhs.value == rhs.value
    ):
        same_datasets = True
    if same_datasets:
        rhs_value = lhs_value
    else:
        rhs_value = make_image_cube_compatible_by_bands(rhs, lhs_value.shape, index_list)

    return lhs_value, rhs_value


def read_lhs_future_onto_queue(
    lhs: "BandMathValue",
    index_list: List[int],
    event_loop,
    read_thread_pool,
    read_task_queue,
):
    future = event_loop.run_in_executor(read_thread_pool, lhs.as_numpy_array_by_bands, index_list)
    read_task_queue.put((future, (min(index_list), max(index_list))))


def read_rhs_future_onto_queue(
    rhs: "BandMathValue",
    lhs_value_shape: Tuple[int],
    index_list: List[int],
    event_loop,
    read_thread_pool,
    read_task_queue,
):
    future = event_loop.run_in_executor(
        read_thread_pool,
        make_image_cube_compatible_by_bands,
        rhs,
        lhs_value_shape,
        index_list,
    )
    read_task_queue.put((future, (min(index_list), max(index_list))))


def should_continue_reading_bands(band_index_list_sorted: List[int], lhs: "BandMathValue"):
    """Determine whether additional band reads are required for an image cube.

    This function checks whether it is valid and necessary to continue reading
    spectral bands from the left-hand-side (LHS) operand during band-math
    evaluation. It assumes the LHS represents an `ImageCube`-type value and that
    the requested band indices are already sorted in ascending order.

    Reading should not continue if the LHS is an intermediate result (i.e.,
    already materialized in memory) or if no band indices are provided.

    Args:
        band_index_list_sorted: Sorted list of requested band indices
        lhs: A ``BandMathValue`` representing the left-hand-side operand.
            Must correspond to an image cube.

    Returns:
        True if band reading should continue; False otherwise.

    Raises:
        AssertionError: If the requested band range exceeds the total number
            of bands in the image cube.

    Notes:
        This function assumes the evaluator has already validated that the
        requested band indices are within bounds of the dataset.
    """
    total_num_bands, _, _ = lhs.get_shape()
    if lhs.is_intermediate:
        return False
    if band_index_list_sorted == [] or band_index_list_sorted is None:
        return False
    max_curr_band = band_index_list_sorted[-1]
    min_curr_band = band_index_list_sorted[0]
    assert (max_curr_band - min_curr_band) < total_num_bands
    return True


def max_bytes_to_chunk(dataset_bytes: int) -> Tuple[float, bool]:
    """
    Determines the maximum number of bytes to use for chunking.

    This function returns a value representing the maximum number of bytes
    that should be used for chunking a dataset. If chunking is not needed,
    the returned value will indicate so.

    Parameters
    ----------
    dataset_bytes : int
        The total number of bytes in the dataset.

    Returns
    -------
    Tuple[float, bool]
        A tuple containing:
        - The number of bytes to use for chunking.
        - A boolean indicating whether chunking should be applied.
    """
    available_mem = psutil.virtual_memory().available
    if dataset_bytes > available_mem:
        return (available_mem * RATIO_OF_MEM_TO_USE, True)
    else:
        return (dataset_bytes, False)


def write_raster_to_dataset(
    out_dataset_gdal,
    band_index_list: List[int],
    result: np.ma.MaskedArray,
    gdal_elem_type: int,
    default_ignore_value: Number = DEFAULT_IGNORE_VALUE,
):
    if isinstance(result, np.ma.MaskedArray):
        result = np.ma.filled(result, default_ignore_value)

    gdal_band_list_current = [band + 1 for band in band_index_list]

    out_dataset_gdal.WriteRaster(
        0,
        0,
        out_dataset_gdal.RasterXSize,
        out_dataset_gdal.RasterYSize,
        result.tobytes(),
        buf_xsize=out_dataset_gdal.RasterXSize,
        buf_ysize=out_dataset_gdal.RasterYSize,
        buf_type=gdal_elem_type,
        band_list=gdal_band_list_current,
    )
    out_dataset_gdal.FlushCache()


def np_dtype_to_gdal(np_dtype):
    """Converts a NumPy dtype to the corresponding GDAL GDT type."""

    # Create a mapping between NumPy dtypes and GDAL GDT types
    dtype_mapping = {
        np.dtype("int8"): gdal.GDT_Byte,
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("int16"): gdal.GDT_Int16,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("int32"): gdal.GDT_Int32,
        np.dtype("uint32"): gdal.GDT_UInt32,
        np.dtype("float16"): gdal.GDT_Float32,
        np.dtype("float32"): gdal.GDT_Float32,
        np.dtype("float64"): gdal.GDT_Float64,
        np.dtype("complex64"): gdal.GDT_CFloat32,
        np.dtype("complex128"): gdal.GDT_CFloat64,
    }

    # Handle cases where the dtype is not in the mapping
    if np_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")

    return dtype_mapping[np_dtype]


def remove_trailing_number(filepath):
    # Regular expression pattern to match " space followed by digits" at the end of the path
    pattern = r"(.*)\s\d+$"

    # Use re.match to see if the pattern matches the filepath
    match = re.match(pattern, filepath)

    # If a match is found, return the group without the trailing space and number
    if match:
        return match.group(1)

    # Otherwise, return the original filepath
    return filepath


def get_unused_file_path_in_folder(folder_to_search: str, result_name: str):
    result_path = os.path.join(folder_to_search, result_name)
    count = 2
    while os.path.exists(result_path):
        result_path = remove_trailing_number(result_path)
        result_path += f" {count}"
        count += 1
    return result_path


def print_tree_with_meta(tree: lark.ParseTree, indent=0):
    indent_str = "  " * indent
    if isinstance(tree, Tree):
        # Print the node type and its meta information if present
        meta_info = ""
        if hasattr(tree, "meta") and tree.meta is not None:
            meta_info = f"(unique_id: {getattr(tree.meta, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree.data} {meta_info}")
        # Recursively print children nodes
        for child in tree.children:
            print_tree_with_meta(child, indent + 1)
    else:
        # If it's a terminal node (e.g., a token), print its value and its meta if available
        meta_info = ""
        if hasattr(tree, "unique_id") or hasattr(tree, "LEFT"):
            meta_info = f"(unique_id: {getattr(tree, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree} {meta_info} (Terminal)")


def get_dimensions(type: VariableType, shape: Tuple) -> str:
    """
    This helper function takes a band-math value-type with a specified shape,
    and returns a human-readable string version of the value's shape.

    If the variable-type isn't an image cube, image band, or spectrum, this
    fucntion returns the empty string ``''``.
    """
    if type == VariableType.IMAGE_CUBE:
        return f"{shape[2]}x{shape[1]}, {shape[0]} bands"

    elif type == VariableType.IMAGE_BAND:
        return f"{shape[1]}x{shape[0]}"

    elif type == VariableType.SPECTRUM:
        return f"{shape[0]} bands"

    return ""


def raise_shape_mismatch(arg1_type, arg1_shape, arg2_type, arg2_shape):
    """
    Given two argument types and shapes, this helper function raises a
    ``ValueError`` with an error-message that reports the shape-mismatch.
    """
    s1 = f"{arg1_type.name} ({get_dimensions(arg1_type, arg1_shape)})"
    s2 = f"{arg2_type.name} ({get_dimensions(arg2_type, arg2_shape)})"
    raise ValueError(f"Incompatible operand shapes: {s1}, {s2}")


def prepare_array(arr):
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0.0)
    arr = np.nan_to_num(arr)
    return arr


def is_scalar(value: "BandMathValue") -> bool:
    """
    Returns ``True`` if the band-math value is a number or Boolean, ``False``
    otherwise.
    """
    return value.type in [VariableType.NUMBER, VariableType.BOOLEAN]


def is_number(value: "BandMathValue") -> bool:
    """
    Returns ``True`` if the band-math value is a number, ``False`` otherwise.
    """
    return value.type == VariableType.NUMBER


def reorder_args(lhs_type: VariableType, rhs_type: VariableType, lhs: Any, rhs: Any) -> Tuple[Any, Any]:
    """
    This function reorders the input arguments such that:
    *   If only one argument is an image cube, it will be on the LHS.
    *   Otherwise, if neither argument is an image cube, and only one argument
        is an image band, it will be on the LHS.
    *   Otherwise, if neither argument is an image band, and only one argument
        is a spectrum, it will be on the LHS.
    *   If none of the above hold, then the LHS and RHS are not reordered.

    This reordering of arguments makes it easier to implement many band-math
    operations.
    """
    # Since logical AND is commutative, arrange the arguments to make the
    # calculation logic easier.
    if lhs_type == VariableType.IMAGE_CUBE_BATCH or rhs_type == VariableType.IMAGE_CUBE_BATCH:
        if lhs_type != VariableType.IMAGE_CUBE_BATCH:
            (rhs, lhs) = (lhs, rhs)

    elif lhs_type == VariableType.IMAGE_CUBE or rhs_type == VariableType.IMAGE_CUBE:
        # If there is only one image cube, make sure it is on the LHS.
        if lhs_type != VariableType.IMAGE_CUBE:
            (rhs, lhs) = (lhs, rhs)

    elif lhs_type == VariableType.IMAGE_BAND_BATCH or rhs_type == VariableType.IMAGE_BAND_BATCH:
        if lhs_type != VariableType.IMAGE_BAND_BATCH:
            (rhs, lhs) = (lhs, rhs)

    elif lhs_type == VariableType.IMAGE_BAND or rhs_type == VariableType.IMAGE_BAND:
        # No image cubes.
        # If there is only one image band, make sure it is on the LHS.
        if lhs_type != VariableType.IMAGE_BAND:
            (rhs, lhs) = (lhs, rhs)

    elif lhs_type == VariableType.SPECTRUM or rhs_type == VariableType.SPECTRUM:
        # No image bands.
        # If there is only one spectrum, make sure it is on the LHS.
        if lhs_type != VariableType.SPECTRUM:
            (rhs, lhs) = (lhs, rhs)

    return (lhs, rhs)


def check_image_cube_compatible(arg: "BandMathExprInfo", cube_shape: Tuple[int, int, int]) -> None:
    """
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-cube with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-cube to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_CUBE``
    *   ``IMAGE_BAND``
    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-cube shape.
    """
    assert len(cube_shape) == 3

    # Only certain types can be compatible with operations involving image cubes
    if arg.result_type not in [
        VariableType.IMAGE_CUBE,
        VariableType.IMAGE_CUBE_BATCH,
        VariableType.IMAGE_BAND,
        VariableType.IMAGE_BAND_BATCH,
        VariableType.SPECTRUM,
        VariableType.NUMBER,
        VariableType.BOOLEAN,
    ]:
        raise ValueError(f"Cannot perform operation between IMAGE_CUBE and {arg.result_type}")

    # Dimensions:  [band][y][x]
    if arg.result_type == VariableType.IMAGE_CUBE_BATCH or arg.result_type == VariableType.IMAGE_BAND_BATCH:
        return
    elif arg.result_type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        if arg.shape != cube_shape:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.result_type, arg.shape)

    elif arg.result_type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        if arg.shape != cube_shape[1:]:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.result_type, arg.shape)

    elif arg.result_type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        if arg.shape != cube_shape[:1]:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.result_type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.result_type in [VariableType.NUMBER, VariableType.BOOLEAN]


def check_image_band_compatible(arg: "BandMathExprInfo", band_shape: Tuple[int, int]) -> None:
    """
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-band with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-band to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_BAND``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-band shape.
    """
    assert len(band_shape) == 2

    if arg.result_type not in [
        VariableType.IMAGE_BAND,
        VariableType.NUMBER,
        VariableType.BOOLEAN,
    ]:
        raise ValueError(f"Cannot perform operation between IMAGE_BAND and {arg.result_type}")

    if arg.result_type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        if arg.shape != band_shape:
            raise_shape_mismatch(VariableType.IMAGE_BAND, band_shape, arg.result_type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.result_type in [VariableType.NUMBER, VariableType.BOOLEAN]


def check_spectrum_compatible(arg: "BandMathExprInfo", spectrum_shape: Tuple[int]) -> None:
    """
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on a spectrum with the specified shape.
    This generally means that the return value can be broadcast against a
    spectrum to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified spectrum shape.
    """
    assert len(spectrum_shape) == 1

    if arg.result_type not in [
        VariableType.SPECTRUM,
        VariableType.NUMBER,
        VariableType.BOOLEAN,
    ]:
        raise ValueError(f"Cannot perform operation between SPECTRUM and {arg.result_type}")

    if arg.result_type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        if arg.shape != spectrum_shape:
            raise_shape_mismatch(VariableType.SPECTRUM, spectrum_shape, arg.result_type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.result_type in [VariableType.NUMBER, VariableType.BOOLEAN]


def make_image_cube_compatible(
    arg: "BandMathValue", cube_shape: Tuple[int, int, int]
) -> Union[np.ndarray, Scalar]:
    """
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-cube with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-cube to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_CUBE``
    *   ``IMAGE_BAND``
    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-cube shape.
    """
    if arg.type not in [
        VariableType.IMAGE_CUBE,
        VariableType.IMAGE_BAND,
        VariableType.SPECTRUM,
        VariableType.NUMBER,
        VariableType.BOOLEAN,
    ]:
        raise TypeError(f"Can't make a {arg.type} value compatible with " + "an image-cube")

    result: Union[np.ndarray, Scalar] = None

    # Dimensions:  [band][y][x]

    if arg.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        result = arg.as_numpy_array()
        assert result.ndim == 3

        if result.shape != cube_shape:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.type, arg.shape)

    elif arg.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        result = arg.as_numpy_array()
        assert result.ndim == 2

        if result.shape != cube_shape[1:]:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.type, result.shape)

    elif arg.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array()
        assert result.ndim == 1

        if result.shape != (cube_shape[0],):
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.type, arg.shape)

        # To ensure the spectrum is broadcast across the image's pixels,
        # reshape to effectively create a 1x1 image.
        # New dimensions:  [band][y=1][x=1]
        result = result[:, np.newaxis, np.newaxis]

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def make_image_cube_compatible_by_bands(
    arg: "BandMathValue", cube_shape: Tuple[int, int, int], band_list: List[int]
) -> Union[np.ndarray, Scalar]:
    """
    Given a band-math value and a list of bands, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-cube with the specified
    shape and bands. This generally means that the return value can be broadcast against
    an image-cube to achieve the "expected" behavior. This function grabs the data
    by bands. Doing so is faster because the data is stored in bsq format. It is
    important that band_list is sorted from least to greatest and is continuous.
    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:
    *   ``IMAGE_CUBE``
    *   ``IMAGE_BAND``
    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``
    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-cube shape.
    """
    if arg.type not in [
        VariableType.IMAGE_CUBE,
        VariableType.IMAGE_BAND,
        VariableType.SPECTRUM,
        VariableType.NUMBER,
        VariableType.BOOLEAN,
    ]:
        raise TypeError(f"Can't make a {arg.type} value compatible with " + "an image-cube")

    result: Union[np.ndarray, Scalar] = None

    # Dimensions:  [band][y][x]
    if arg.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        result = arg.as_numpy_array_by_bands(band_list)
        assert result.ndim == 3 or (result.ndim == 2 and len(band_list) == 1)

        if not are_shapes_broadcastable(result.shape, cube_shape):
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.type, arg.get_shape())

    elif arg.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        result = arg.as_numpy_array_by_bands(band_list)
        assert result.ndim == 2

        if (result.shape != cube_shape[1:]) and (result.shape != cube_shape and len(band_list) == 1):
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.type, arg.get_shape())

    elif arg.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array_by_bands(band_list)
        max_dim = max(result.shape)
        result = np.squeeze(result).reshape(max_dim)
        assert result.ndim == 1

        if (result.shape != (cube_shape[0],)) and len(band_list) != 1:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape, arg.type, arg.get_shape())

        # To ensure the spectrum is broadcast across the image's pixels,
        # reshape to effectively create a 1x1 image.
        # New dimensions:  [band][y=1][x=1]
        result = result[:, np.newaxis, np.newaxis]

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def are_shapes_equivalent(shape1, shape2):
    # Remove leading 1s from both shapes
    trimmed_shape1 = tuple(dim for dim in shape1 if dim != 1)
    trimmed_shape2 = tuple(dim for dim in shape2 if dim != 1)

    # Compare the resulting shapes
    return trimmed_shape1 == trimmed_shape2


def are_shapes_broadcastable(shape1, shape2):
    """
    Check if two shapes are broadcastable according to NumPy's broadcasting rules.
    Broadcasting means that two shapes are compatible if:
      - The dimensions are the same, or
      - One of the dimensions is 1, or
      - The smaller shape can be extended with leading 1's to match the larger shape.
    """
    # Reverse the shapes to align from the last dimension
    rev_shape1 = shape1[::-1]
    rev_shape2 = shape2[::-1]

    # Compare dimensions from the last one backward (trailing dimensions)
    for dim1, dim2 in zip(rev_shape1, rev_shape2):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False

    return True


def make_image_band_compatible(
    arg: "BandMathValue", band_shape: Tuple[int, int]
) -> Union[np.ndarray, Scalar]:
    """
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-band with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-band to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_BAND``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-band shape.
    """
    if arg.type not in [
        VariableType.IMAGE_BAND,
        VariableType.NUMBER,
        VariableType.BOOLEAN,
    ]:
        raise TypeError(f"Can't make a {arg.type} value compatible with " + "an image-band")

    result: Union[np.ndarray, Scalar] = None

    if arg.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        result = arg.as_numpy_array()
        assert result.ndim == 2

        if result.shape != band_shape:
            raise_shape_mismatch(VariableType.IMAGE_BAND, band_shape, arg.type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def make_spectrum_compatible(arg: "BandMathValue", spectrum_shape: Tuple[int]) -> Union[np.ndarray, Scalar]:
    """
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on a spectrum with the specified shape.
    This generally means that the return value can be broadcast against a
    spectrum to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified spectrum shape.
    """
    if arg.type not in [
        VariableType.SPECTRUM,
        VariableType.NUMBER,
        VariableType.BOOLEAN,
    ]:
        raise TypeError(f"Can't make a {arg.type} value compatible with " + "a spectrum")

    result: Union[np.ndarray, Scalar] = None

    if arg.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array()
        assert result.ndim == 1

        if result.shape != spectrum_shape:
            raise_shape_mismatch(VariableType.SPECTRUM, spectrum_shape, arg.type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def check_metadata_compatible():
    pass

"""
Tests for the band-math peak-memory estimate and the refusal raised before an
evaluation whose estimate exceeds the usable share of available memory.

Available memory is patched; no test allocates a large array.  The large
operands are zero-stride views from ``np.broadcast_to``, whose shape and dtype
are all the estimator reads.
"""

import os
import queue
from types import SimpleNamespace
from unittest import mock

import lark
import numpy as np
import pytest

import tests.context  # noqa: F401

from osgeo import gdal
from PySide6.QtWidgets import QDialog

from test_utils.test_model import WiserTestModel
from wiser.bandmath import VariableType, BandMathMemoryError
from wiser.bandmath import evaluator
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.bandmath.functions import get_builtin_functions
from wiser.gui.bandmath_dialog import BandMathDialog, BandmathBatchJob
from wiser.raster.dataset import RasterDataSet


def patch_available(available_bytes):
    return mock.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=available_bytes))


def make_job(expr, variables, use_synchronous_method=True, number_of_intermediates=1):
    tree = lark.Lark.open(
        "bandmath.lark", rel_to=evaluator.__file__, start="expression", propagate_positions=True
    ).parse(expr)
    evaluator.UniqueIDAssigner().visit(tree)
    return evaluator.SingleBandMathJob(
        bandmath_expr=expr,
        expr_info=get_bandmath_expr_info(expr, variables, {}),
        result_name="result",
        cache=None,
        lower_variables=variables,
        lower_functions=get_builtin_functions(),
        number_of_intermediates=number_of_intermediates,
        tree=tree,
        use_synchronous_method=use_synchronous_method,
        temp_folder_path=evaluator.TEMP_FOLDER_PATH,
    )


def cube(dtype=np.float32, shape=(2, 3, 4)):
    return (VariableType.IMAGE_CUBE, np.zeros(shape, dtype=dtype))


def big_cube():
    """A 100 x 1000 x 1000 float32 cube (400 MB nominal) backed by one element."""
    return (VariableType.IMAGE_CUBE, np.broadcast_to(np.float32(0), (100, 1000, 1000)))


# Each 2 x 3 x 4 float32 operand: 24 elements, 5 bytes each (4-byte value plus
# a 1-byte mask), so one array is 120 bytes; the two inputs hold 192 bytes.


class TestEstimate:
    def test_binary_operator_whole_array(self):
        job = make_job("a + b", {"a": cube(), "b": cube()})
        with patch_available(10**12):
            # Two reads (240), one read transient (120) and the result (120),
            # plus the 192 resident input bytes.
            assert evaluator.estimate_bandmath_peak_bytes(job) == 480 + 192

    def test_integer_operand_is_estimated_at_float64(self):
        job = make_job("a * 2", {"a": cube(np.int16)})
        with patch_available(10**12):
            # 9 bytes per element: one read, its transient and the result,
            # plus the 48 resident input bytes.
            assert evaluator.estimate_bandmath_peak_bytes(job) == 3 * 24 * 9 + 48

    def test_sibling_result_is_held_while_the_next_sibling_evaluates(self):
        job = make_job("(a + b) * (a - b)", {"a": cube(), "b": cube()})
        with patch_available(10**12):
            # The held 120-byte sum plus the 480-byte peak of the difference.
            assert evaluator.estimate_bandmath_peak_bytes(job) == 600 + 192

    def test_function_counts_its_argument_and_a_full_size_result(self):
        job = make_job("sin(a)", {"a": cube()})
        with patch_available(10**12):
            assert evaluator.estimate_bandmath_peak_bytes(job) == 360 + 96

    def test_chunked_path_floor_is_one_band_per_operand(self):
        job = make_job("a + b", {"a": cube(), "b": cube()}, use_synchronous_method=False)
        with patch_available(50):
            # One 3 x 4 band is 60 bytes: the window's two reads, transient and
            # result (240), the two next-window prefetches (120) and one pending
            # write (60), plus the 192 resident input bytes.
            assert evaluator.estimate_bandmath_peak_bytes(job) == 240 + 120 + 60 + 192

    def test_chunked_path_siblings_evaluate_concurrently(self):
        job = make_job("(a + b) * (a - b)", {"a": cube(), "b": cube()}, use_synchronous_method=False)
        with patch_available(10**12):
            # Every band fits one window, so the chunk is the whole cube: both
            # 480-byte sibling peaks at once, four prefetches and a pending write.
            assert evaluator.estimate_bandmath_peak_bytes(job) == 960 + 480 + 120 + 192


class TestCheck:
    def test_raises_when_estimate_exceeds_eighty_percent_of_available(self):
        job = make_job("a + 1", {"a": big_cube()})
        with patch_available(int(2.0e9)):
            estimate = evaluator.estimate_bandmath_peak_bytes(job)
            with pytest.raises(BandMathMemoryError) as excinfo:
                evaluator.check_bandmath_memory(job)
        message = str(excinfo.value)
        assert f"{estimate / 1e9:.1f} GB" in message
        assert "2.0 GB" in message
        assert "spatial subset" in message

    def test_passes_just_under_the_threshold(self):
        job = make_job("a + 1", {"a": big_cube()})
        with patch_available(10**12):
            estimate = evaluator.estimate_bandmath_peak_bytes(job)
        # Just under the threshold the whole-array path runs; just over it the
        # band-chunked fallback is chosen, whose estimate also exceeds it.
        available = int(np.ceil(estimate / 0.8)) + 8
        with patch_available(available):
            assert evaluator.estimate_bandmath_peak_bytes(job) == estimate
            evaluator.check_bandmath_memory(job)
        with patch_available(available - 16):
            with pytest.raises(BandMathMemoryError):
                evaluator.check_bandmath_memory(job)


def test_start_bandmath_evaluation_refuses_before_starting_a_process():
    variables = {"a": big_cube()}
    expr_info = get_bandmath_expr_info("a + 1", variables, {})
    with patch_available(int(2.0e9)), mock.patch.object(evaluator, "ProcessManager") as process_manager:
        with pytest.raises(BandMathMemoryError) as excinfo:
            evaluator.start_bandmath_evaluation(
                bandmath_expr="a + 1",
                expr_info=expr_info,
                result_name="result",
                cache=None,
                variables=variables,
                functions={},
            )
    process_manager.assert_not_called()
    assert "2.0 GB" in str(excinfo.value)


@pytest.fixture
def test_model():
    model = WiserTestModel()
    yield model
    model.close_app()


def test_bandmath_dialog_shows_refusal_and_creates_no_dataset(test_model):
    dataset = test_model.load_dataset(np.zeros((2, 3, 4), dtype=np.float32))
    variables = {"a": (VariableType.IMAGE_CUBE, dataset)}
    dialog = mock.MagicMock()
    dialog.exec.return_value = QDialog.Accepted
    dialog.get_expression.return_value = "a + 1"
    dialog.get_expression_info.return_value = get_bandmath_expr_info("a + 1", variables, {})
    dialog.get_variable_bindings.return_value = variables
    dialog.get_result_name.return_value = "result"
    dialog.is_batch_processing_enabled.return_value = False
    dialog.load_results_into_wiser.return_value = True
    test_model.main_window._bandmath_dialog = dialog
    num_datasets = len(test_model.app_state.get_datasets())

    with patch_available(100), mock.patch("wiser.gui.app.QMessageBox.critical") as critical:
        test_model.main_window.show_bandmath_dialog()

    critical.assert_called_once()
    message = critical.call_args[0][2]
    assert "0.0 GB" in message and "spatial subset" in message
    assert test_model.app_state.get_running_processes() == {}
    assert len(test_model.app_state.get_datasets()) == num_datasets


def write_envi(folder, name, shape):
    path = os.path.join(folder, name)
    ds = gdal.GetDriverByName("ENVI").Create(path, shape[2], shape[1], shape[0], gdal.GDT_Float32)
    for band in range(shape[0]):
        ds.GetRasterBand(band + 1).WriteArray(np.ones(shape[1:], dtype=np.float32))
    ds = None
    return path


def test_batch_reports_oversized_file_and_runs_the_others(test_model, tmp_path):
    folder = str(tmp_path)
    write_envi(folder, "small", (1, 10, 10))
    write_envi(folder, "large", (1, 1000, 1000))
    variables = {"a": (VariableType.IMAGE_CUBE_BATCH, folder)}
    job = BandmathBatchJob(
        job_id=0,
        expression="a + 1",
        expr_info=get_bandmath_expr_info("a + 1", variables, {}),
        variables=variables,
        input_folder=folder,
        output_folder="",
        load_into_wiser=True,
        result_suffix="_result",
    )
    dialog = BandMathDialog(test_model.app_state, parent=test_model.main_window)
    dialog._add_batch_job_to_table(job)

    # The subprocess does not see patched memory, so the job it would run is
    # evaluated in this process instead.
    with (
        patch_available(100_000),
        mock.patch.object(evaluator, "ProcessManager") as process_manager,
        mock.patch("wiser.gui.bandmath_dialog.QMessageBox.critical") as critical,
    ):
        dialog._run_batch_job(job)
        critical.assert_not_called()
        bandmath_job_data = process_manager.call_args[0][1]["bandmath_job_data"]
        child_conn = mock.MagicMock()
        return_queue = queue.Queue()
        evaluator.eval_bandmath_expressions(bandmath_job_data, child_conn, return_queue)

    messages = [c[0][0] for c in child_conn.send.call_args_list]
    for message in messages:
        dialog.on_bandmath_job_status(job, message)
    errors = {name: text for name, text, _ in job.get_errors()}
    assert list(errors) == ["large_result"]
    assert "spatial subset" in errors["large_result"]
    assert job.get_btn_view_errors().isEnabled()

    results = return_queue.get_nowait()
    completed = [r for r in results if r[0] is not None]
    assert [r[2] for r in completed] == ["small_result"]
    np.testing.assert_array_equal(np.asarray(completed[0][1]), np.full((1, 10, 10), 2, dtype=np.float32))


def mock_dataset(shape):
    dataset = mock.MagicMock(spec=RasterDataSet)
    dataset.get_shape.return_value = shape
    dataset.get_elem_type.return_value = np.dtype(np.float32)
    return (VariableType.IMAGE_CUBE, dataset)


class TestPathSelection:
    def test_whole_array_estimate_over_threshold_selects_chunked_path(self):
        # Two 1 GB float32 cubes: the whole-array estimate (5 GB) exceeds 80% of 6 GB
        # although the result times its intermediates (3 GB) fits.
        shape = (250, 1000, 1000)
        job = make_job(
            "a / b", {"a": mock_dataset(shape), "b": mock_dataset(shape)}, number_of_intermediates=3
        )
        with patch_available(int(6e9)):
            assert evaluator._estimate_peak_bytes(job, False, 0) == 4 * 250_000_000 * 5
            chunked, max_chunking_bytes = evaluator.select_bandmath_chunking(job)
            assert chunked
            assert max_chunking_bytes == 6e9 * 0.25
            evaluator.check_bandmath_memory(job)

    def test_refused_when_one_band_window_exceeds_threshold(self):
        shape = (2, 20000, 20000)
        job = make_job(
            "a / b", {"a": mock_dataset(shape), "b": mock_dataset(shape)}, number_of_intermediates=3
        )
        with patch_available(int(6e9)):
            assert evaluator.select_bandmath_chunking(job)[0]
            # One 20000 x 20000 band at 5 bytes per element is 2 GB.
            assert evaluator.estimate_bandmath_peak_bytes(job) == 7 * 2_000_000_000
            with pytest.raises(BandMathMemoryError):
                evaluator.check_bandmath_memory(job)

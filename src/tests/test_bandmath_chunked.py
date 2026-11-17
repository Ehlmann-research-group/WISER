"""Tests for the chunked (on-disk) band-math path, evaluated in-process."""

import gc
import shutil
import tempfile

import lark
import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401

from wiser.bandmath import VariableType, evaluator
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.bandmath.functions import get_builtin_functions
from wiser.raster.data_cache import DataCache
from wiser.raster.loader import RasterDataLoader


@pytest.fixture
def temp_folder(monkeypatch):
    folder = tempfile.mkdtemp()
    monkeypatch.setattr(evaluator, "TEMP_FOLDER_PATH", folder)
    yield folder
    gc.collect()
    shutil.rmtree(folder, ignore_errors=True)


def _job(expr, dataset, cache):
    variables = {"d": (VariableType.IMAGE_CUBE, dataset)}
    tree = lark.Lark.open(
        "bandmath.lark", rel_to=evaluator.__file__, start="expression", propagate_positions=True
    ).parse(expr)
    evaluator.UniqueIDAssigner().visit(tree)
    return evaluator.SingleBandMathJob(
        bandmath_expr=expr,
        expr_info=get_bandmath_expr_info(expr, variables, {}),
        result_name="chunked",
        cache=cache,
        lower_variables=variables,
        lower_functions=get_builtin_functions(),
        number_of_intermediates=1,
        tree=tree,
        use_synchronous_method=False,
        temp_folder_path=evaluator.TEMP_FOLDER_PATH,
    )


def _evaluate_one_band_per_chunk(job):
    _, out_dataset, _, _ = evaluator.eval_singular_bandmath_expr_async(job, max_chunking_bytes=1)
    path = out_dataset.get_filepaths()[0]
    # Release the writer's GDAL handle so the data reaches disk.
    del out_dataset
    gc.collect()
    written = gdal.Open(path)
    values = written.ReadAsArray()
    written = None
    return values


CUBE = np.linspace(0.1, 1.0, 4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("d > 0.5", (CUBE > 0.5).astype(np.byte)),
        ("d + 1", CUBE + 1),
        ("d - 1", CUBE - 1),
        ("d * 2", CUBE * 2),
        ("d / 2", CUBE / 2),
        ("d ** 2", CUBE**2),
    ],
)
def test_numpy_backed_cube_one_band_per_chunk(temp_folder, expr, expected):
    cache = DataCache()
    dataset = RasterDataLoader().dataset_from_numpy_array(CUBE.copy(), cache)
    dataset.set_id(1)
    values = _evaluate_one_band_per_chunk(_job(expr, dataset, cache))
    np.testing.assert_allclose(values, expected, rtol=1e-6)


def test_evaluator_construction_error_propagates(temp_folder, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("evaluator construction failed")

    monkeypatch.setattr(evaluator, "BandMathEvaluatorAsync", fail)
    cache = DataCache()
    dataset = RasterDataLoader().dataset_from_numpy_array(CUBE.copy(), cache)
    dataset.set_id(1)
    with pytest.raises(RuntimeError, match="evaluator construction failed"):
        evaluator.eval_singular_bandmath_expr_async(_job("d + 1", dataset, cache), max_chunking_bytes=1)

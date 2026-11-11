"""A chunked band-math result with no-data pixels survives a project round trip.

A large band-math result is written band window by band window to a temporary file
(issue #798 labels its masked pixels with an ignore value no valid pixel holds), and a
project save copies that temporary file into the bundle (issue #853).  Reopening the
project after the temporary folder is gone must give back the same ignore value and
mask.
"""

import gc
import os
import shutil

import lark
import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401

from tests.test_project_orchestrate import _FakeAppState
from wiser.bandmath import VariableType, evaluator
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.bandmath.functions import get_builtin_functions
from wiser.bandmath.utils import load_image_from_bandmath_result
from wiser.project.orchestrate import load_project, save_project
from wiser.raster.data_cache import DataCache

NODATA = -9999.0
SOURCE = np.array(
    [
        [[NODATA, 0.9], [0.01, 0.9]],
        [[0.01, NODATA], [0.9, 0.9]],
        [[0.9, 0.01], [0.01, NODATA]],
    ],
    dtype=np.float32,
)


def _source_dataset(app, folder):
    path = os.path.join(folder, "source.tif")
    gdal_ds = gdal.GetDriverByName("GTiff").Create(
        path, SOURCE.shape[2], SOURCE.shape[1], SOURCE.shape[0], gdal.GDT_Float32
    )
    for i in range(SOURCE.shape[0]):
        gdal_ds.GetRasterBand(i + 1).WriteArray(SOURCE[i])
        gdal_ds.GetRasterBand(i + 1).SetNoDataValue(NODATA)
    gdal_ds = None
    ds = app.get_loader().load_from_file(path, interactive=False)[0]
    assert ds.get_data_ignore_value() == NODATA
    return ds


def _evaluate_chunked(expr, dataset):
    variables = {"d": (VariableType.IMAGE_CUBE, dataset)}
    tree = lark.Lark.open(
        "bandmath.lark", rel_to=evaluator.__file__, start="expression", propagate_positions=True
    ).parse(expr)
    evaluator.UniqueIDAssigner().visit(tree)
    expr_info = get_bandmath_expr_info(expr, variables, {})
    job = evaluator.SingleBandMathJob(
        bandmath_expr=expr,
        expr_info=expr_info,
        result_name="chunked",
        cache=DataCache(),
        lower_variables=variables,
        lower_functions=get_builtin_functions(),
        number_of_intermediates=1,
        tree=tree,
        use_synchronous_method=False,
        temp_folder_path=evaluator.TEMP_FOLDER_PATH,
    )
    result_type, out_dataset, result_name, expr_info = evaluator.eval_singular_bandmath_expr_async(
        job, max_chunking_bytes=1
    )
    serialized = out_dataset.get_serialized_form()
    # Release the writer's GDAL handle so the ENVI header and data reach disk.
    del out_dataset
    gc.collect()
    return load_image_from_bandmath_result(result_type, serialized, result_name, expr, expr_info)


@pytest.mark.parametrize(
    "expr, dtype, ignore_value",
    [
        ("d > 0.05", np.int16, NODATA),
        ("d * 2", np.float32, NODATA),
    ],
)
def test_chunked_masked_result_reopens_with_ignore_value_and_mask(
    tmp_path, monkeypatch, expr, dtype, ignore_value
):
    temp_dir = tmp_path / "temp_output"
    temp_dir.mkdir()
    monkeypatch.setattr(evaluator, "TEMP_FOLDER_PATH", str(temp_dir))
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    app = _FakeAppState()
    source = _source_dataset(app, str(source_dir))
    app.add_dataset(source)
    result = _evaluate_chunked(expr, source)
    assert result.get_filepaths()[0].startswith(str(temp_dir))
    assert result.get_elem_type() == np.dtype(dtype)
    assert result.get_data_ignore_value() == ignore_value
    expected_mask = SOURCE == NODATA
    np.testing.assert_array_equal(np.ma.getmaskarray(result.get_image_data()), expected_mask)
    app.add_dataset(result)
    result_id = result.get_id()

    save_project(app, tmp_path / "session.wiserproj")
    del result
    app = None
    gc.collect()
    shutil.rmtree(temp_dir)

    restored_app = _FakeAppState()
    report = load_project(tmp_path / "session.wiserproj", restored_app, extract_dir=tmp_path / "unpacked")
    assert report["datasets"] == []
    restored = restored_app.get_dataset(result_id)
    assert not restored.get_filepaths()[0].startswith(str(temp_dir))
    assert restored.get_data_ignore_value() == ignore_value
    np.testing.assert_array_equal(np.ma.getmaskarray(restored.get_image_data()), expected_mask)


def test_in_memory_comparison_mask_survives_project_round_trip(tmp_path):
    # A comparison result such as "d > 0.05" is int8, which ENVI cannot store.
    data = np.array([[[1, 0], [0, -128]]], dtype=np.int8)
    app = _FakeAppState()
    mask = app.get_loader().dataset_from_numpy_array(data)
    mask.set_data_ignore_value(-128)
    app.add_dataset(mask)
    mask_id = mask.get_id()

    save_project(app, tmp_path / "session.wiserproj")

    restored_app = _FakeAppState()
    report = load_project(tmp_path / "session.wiserproj", restored_app, extract_dir=tmp_path / "unpacked")
    assert report["datasets"] == []
    restored = restored_app.get_dataset(mask_id)
    assert restored.get_elem_type() == np.dtype(np.int16)
    assert restored.get_data_ignore_value() == -128
    restored_data = restored.get_image_data()
    np.testing.assert_array_equal(np.ma.getdata(restored_data), data)
    np.testing.assert_array_equal(np.ma.getmaskarray(restored_data), data == -128)

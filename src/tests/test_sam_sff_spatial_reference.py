"""SAM and SFF image-cube results carry the target's spatial reference.

The results are computed on the full target grid, so each result dataset holds
the target's geotransform and CRS, and Save As writes them to the ENVI header.
"""

import os

import numpy as np
import pytest
from astropy import units as u
from osgeo import gdal, osr

import tests.context  # noqa: F401

from test_utils.test_model import WiserTestModel
from wiser.gui.generic_spectral_tool import SpectralComputationInputs
from wiser.gui.spectral_angle_mapper_tool import SAMTool
from wiser.gui.spectral_feature_fitting_tool import SFFTool
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import NumPyArraySpectrum

pytestmark = [pytest.mark.functional]

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "test_utils", "test_datasets", "caltech_4_100_150_nm")


@pytest.fixture
def test_model():
    model = WiserTestModel()
    yield model
    model.close_app()


@pytest.mark.parametrize(("tool_cls", "num_results"), [(SAMTool, 2), (SFFTool, 3)])
def test_image_cube_results_keep_target_spatial_reference(test_model, tmp_path, tool_cls, num_results):
    target = test_model.load_dataset(FIXTURE)
    expected_gt = target.get_geo_transform()
    expected_wkt = target.get_wkt_spatial_reference()
    assert expected_wkt
    assert expected_gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    reference = NumPyArraySpectrum(
        np.array([0.30, 0.20, 0.25, 0.40], dtype=np.float32),
        name="ref",
        wavelengths=target.get_wavelengths(),
    )
    inputs = SpectralComputationInputs(
        target=target,
        mode="Image Cube",
        refs=[reference],
        thresholds=[np.float32(10.0)],
        global_thr=None,
        min_wvl=0 * u.nm,
        max_wvl=3000 * u.nm,
        lib_name_by_spec_id=None,
    )
    future = tool_cls(app_state=test_model.app_state).find_matches(spectral_inputs=inputs, test_mode=True)
    future.result(timeout=180)
    test_model.app.processEvents()
    results = test_model.app_state.get_datasets()[-num_results:]
    assert len(results) == num_results
    assert all(r is not target for r in results)

    expected_srs = osr.SpatialReference(wkt=expected_wkt)
    for i, result in enumerate(results):
        assert result.get_geo_transform() == expected_gt, result.get_name()
        assert result.get_wkt_spatial_reference() == expected_wkt, result.get_name()

        path = str(tmp_path / f"result_{i}.img")
        RasterDataLoader().save_dataset_as(result, path, format="ENVI", config=None)
        written = gdal.Open(path)
        saved_gt, saved_wkt = written.GetGeoTransform(), written.GetProjection()
        written = None
        assert saved_gt == pytest.approx(expected_gt), result.get_name()
        assert saved_wkt, result.get_name()
        assert osr.SpatialReference(wkt=saved_wkt).IsSame(expected_srs), result.get_name()

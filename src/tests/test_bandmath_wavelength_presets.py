"""Spectral-index presets and wavelength-bound band selection for band math.

The same preset must pick the physically equivalent bands on sensors whose band
numbering differs: EMIT (285 bands, ~7.4 nm spacing), AVIRIS-NG (425 bands,
~5 nm) and AVIRIS-classic (224 bands, ~10 nm, four spectrometers whose ranges
overlap).  Expected band indices below are the nearest band to each preset's
reference wavelength on each grid.
"""

import os

import numpy as np
import pytest
from astropy import units as u

import tests.context  # noqa: F401

from wiser import bandmath
from wiser.bandmath import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.bandmath.presets import (
    DEFAULT_WAVELENGTH_TOLERANCE_NM,
    SPECTRAL_INDEX_PRESETS,
    get_spectral_index_preset,
    resolve_preset_bands,
)
from wiser.raster.data_cache import DataCache
from wiser.raster.dataset import RasterDataDynamicBand
from wiser.raster.loader import RasterDataLoader
from wiser.raster.utils import build_band_info_from_wavelengths

TEST_DATASETS = os.path.join(os.path.dirname(__file__), "..", "test_utils", "test_datasets")
EMIT_PATH = os.path.join(TEST_DATASETS, "EMIT_L2A_RFL_001_20260821T180123_2623312_016_subset_aid0000.nc")
AVIRIS_NG_PATH = os.path.join(TEST_DATASETS, "jpl_425_7_7.hdr")
NO_WAVELENGTHS_PATH = os.path.join(TEST_DATASETS, "envi.hdr")
FOUR_BAND_PATH = os.path.join(TEST_DATASETS, "caltech_4_100_150_nm.hdr")

# Band centers (nm) of the Cuprite AVIRIS-classic fixture
# (cuprite_224_90_140_nm.hdr, granule f230918t01p00r11).
CUPRITE_WAVELENGTHS_NM = [
    378.9298,
    388.5940,
    398.2625,
    407.9355,
    417.6129,
    427.2946,
    436.9808,
    446.6713,
    456.3662,
    466.0655,
    475.7692,
    485.4773,
    495.1898,
    504.9066,
    514.6279,
    524.3536,
    534.0836,
    543.8180,
    553.5568,
    563.3000,
    573.0477,
    582.7996,
    592.5560,
    602.3168,
    612.0819,
    621.8515,
    631.6254,
    641.4037,
    651.1865,
    660.9736,
    670.7651,
    680.5610,
    654.7923,
    664.5994,
    674.4012,
    684.1979,
    693.9894,
    703.7756,
    713.5566,
    723.3325,
    733.1031,
    742.8685,
    752.6287,
    762.3837,
    772.1335,
    781.8781,
    791.6174,
    801.3516,
    811.0805,
    820.8043,
    830.5228,
    840.2361,
    849.9442,
    859.6471,
    869.3448,
    879.0372,
    888.7245,
    898.4066,
    908.0834,
    917.7551,
    927.4214,
    937.0827,
    946.7387,
    956.3895,
    966.0351,
    975.6755,
    985.3106,
    994.9406,
    1004.565,
    1014.185,
    1023.799,
    1033.408,
    1043.012,
    1052.611,
    1062.204,
    1071.793,
    1081.376,
    1090.954,
    1100.526,
    1110.094,
    1119.656,
    1129.213,
    1138.765,
    1148.311,
    1157.853,
    1167.389,
    1176.920,
    1186.446,
    1195.966,
    1205.482,
    1214.992,
    1224.497,
    1233.996,
    1243.491,
    1252.980,
    1262.464,
    1252.773,
    1262.746,
    1272.718,
    1282.691,
    1292.662,
    1302.634,
    1312.606,
    1322.577,
    1332.548,
    1342.519,
    1352.490,
    1362.460,
    1372.430,
    1382.400,
    1392.369,
    1402.339,
    1412.308,
    1422.277,
    1432.245,
    1442.214,
    1452.182,
    1462.150,
    1472.118,
    1482.085,
    1492.052,
    1502.019,
    1511.986,
    1521.952,
    1531.918,
    1541.885,
    1551.850,
    1561.816,
    1571.781,
    1581.746,
    1591.711,
    1601.675,
    1611.640,
    1621.604,
    1631.568,
    1641.531,
    1651.494,
    1661.458,
    1671.420,
    1681.383,
    1691.345,
    1701.307,
    1711.269,
    1721.231,
    1731.192,
    1741.153,
    1751.114,
    1761.075,
    1771.036,
    1780.996,
    1790.956,
    1800.915,
    1810.875,
    1820.834,
    1830.793,
    1840.752,
    1850.710,
    1860.669,
    1870.627,
    1871.784,
    1868.964,
    1879.025,
    1889.085,
    1899.141,
    1909.196,
    1919.248,
    1929.298,
    1939.346,
    1949.391,
    1959.435,
    1969.475,
    1979.514,
    1989.550,
    1999.584,
    2009.615,
    2019.645,
    2029.672,
    2039.696,
    2049.719,
    2059.739,
    2069.756,
    2079.772,
    2089.785,
    2099.796,
    2109.804,
    2119.810,
    2129.814,
    2139.816,
    2149.815,
    2159.812,
    2169.807,
    2179.799,
    2189.789,
    2199.777,
    2209.762,
    2219.745,
    2229.726,
    2239.705,
    2249.681,
    2259.655,
    2269.626,
    2279.595,
    2289.562,
    2299.527,
    2309.490,
    2319.449,
    2329.407,
    2339.363,
    2349.316,
    2359.267,
    2369.215,
    2379.161,
    2389.105,
    2399.047,
    2408.986,
    2418.923,
    2428.858,
    2438.790,
    2448.720,
    2458.648,
    2468.573,
    2478.496,
    2488.417,
    2498.336,
]

loader = RasterDataLoader()


def _make_dataset(wavelengths, bands=None):
    num_bands = len(wavelengths) if wavelengths is not None else bands
    dataset = loader.dataset_from_numpy_array(np.zeros((num_bands, 2, 3), dtype=np.float32))
    dataset.set_name("synthetic")
    if wavelengths is not None:
        dataset.set_band_list(build_band_info_from_wavelengths(wavelengths))
    return dataset


def _load(path):
    return loader.load_from_file(path=path, interactive=False)[0]


def _cuprite():
    return _make_dataset([w * u.nm for w in CUPRITE_WAVELENGTHS_NM])


EXPECTED_BANDS = {
    "emit": {
        "NDVI": {"nir": 64, "red": 38},
        "SAVI": {"nir": 64, "red": 38},
        "NDWI": {"r860": 64, "r1240": 115},
        "MSI": {"r1600": 164, "r820": 59},
    },
    "aviris_ng": {
        "NDVI": {"nir": 97, "red": 57},
        "SAVI": {"nir": 97, "red": 57},
        "NDWI": {"r860": 97, "r1240": 172},
        "MSI": {"r1600": 244, "r820": 89},
    },
    "cuprite": {
        "NDVI": {"nir": 53, "red": 29},
        "SAVI": {"nir": 53, "red": 29},
        "NDWI": {"r860": 53, "r1240": 93},
        "MSI": {"r1600": 131, "r820": 49},
    },
}

SENSOR_LOADERS = {
    "emit": lambda: _load(EMIT_PATH),
    "aviris_ng": lambda: _load(AVIRIS_NG_PATH),
    "cuprite": _cuprite,
}


@pytest.mark.parametrize("sensor", sorted(EXPECTED_BANDS))
def test_presets_resolve_same_physical_bands_across_sensors(sensor):
    dataset = SENSOR_LOADERS[sensor]()
    for preset_name, expected in EXPECTED_BANDS[sensor].items():
        preset = get_spectral_index_preset(preset_name)
        resolved = resolve_preset_bands(preset, dataset, DEFAULT_WAVELENGTH_TOLERANCE_NM)
        assert resolved == expected, preset_name
        for var, band_index in resolved.items():
            actual = dataset.get_wavelengths()[band_index].to_value(u.nm)
            reference = preset.band_wavelengths[var].to_value(u.nm)
            assert abs(actual - reference) <= DEFAULT_WAVELENGTH_TOLERANCE_NM


def test_preset_catalog_documents_wavelengths_and_sources():
    assert [p.name for p in SPECTRAL_INDEX_PRESETS] == ["NDVI", "SAVI", "NDWI", "MSI"]
    for preset in SPECTRAL_INDEX_PRESETS:
        assert bandmath.get_bandmath_variables(preset.expression) == set(preset.band_wavelengths)
        assert preset.reference
        for wavelength in preset.band_wavelengths.values():
            assert wavelength.unit == u.nm


def test_preset_reference_wavelengths():
    def nm(name):
        return {k: v.to_value(u.nm) for k, v in get_spectral_index_preset(name).band_wavelengths.items()}

    assert nm("NDVI") == {"nir": 860.0, "red": 660.0}
    assert nm("SAVI") == {"nir": 860.0, "red": 660.0}
    assert nm("NDWI") == {"r860": 860.0, "r1240": 1240.0}
    assert nm("MSI") == {"r1600": 1600.0, "r820": 820.0}


def test_unknown_preset_raises():
    with pytest.raises(KeyError, match="EVI"):
        get_spectral_index_preset("EVI")


def test_preset_on_dataset_without_wavelengths_raises():
    dataset = _load(NO_WAVELENGTHS_PATH)
    assert not dataset.has_wavelengths()
    with pytest.raises(ValueError, match="has no band wavelengths"):
        resolve_preset_bands(get_spectral_index_preset("NDVI"), dataset, DEFAULT_WAVELENGTH_TOLERANCE_NM)


def test_preset_wavelength_beyond_sensor_range_raises():
    # A VNIR-only grid (400-1000 nm) has no band near the 1240 nm NDWI band.
    dataset = _make_dataset([w * u.nm for w in np.arange(400.0, 1000.0, 5.0)])
    with pytest.raises(ValueError, match=r"No band .* within 10(\.0)? nm of 1240(\.0)? nm"):
        resolve_preset_bands(get_spectral_index_preset("NDWI"), dataset, DEFAULT_WAVELENGTH_TOLERANCE_NM)


def test_wavelength_in_gap_between_bands_raises_not_nearest():
    # caltech_4_100_150_nm has bands at ~472, 532, 662 and 853 nm.  750 nm lies
    # ~90 nm from the nearest band, so no band is chosen.
    dataset = _load(FOUR_BAND_PATH)
    band = RasterDataDynamicBand(dataset, wavelength_value=750.0, wavelength_units=u.nm, epsilon=10.0)
    with pytest.raises(ValueError, match="No band"):
        band.resolve_band_index()


def test_dynamic_band_without_wavelengths_raises_value_error():
    dataset = _load(NO_WAVELENGTHS_PATH)
    with pytest.raises(ValueError, match="has no band wavelengths"):
        RasterDataDynamicBand(dataset, wavelength_value=860.0, wavelength_units=u.nm, epsilon=10.0)


def test_dynamic_band_resolves_in_micrometers():
    dataset = _load(EMIT_PATH)
    band = RasterDataDynamicBand(dataset, wavelength_value=0.86, wavelength_units=u.um, epsilon=0.01)
    assert band.resolve_band_index() == 64


def test_dynamic_band_get_data_raises_when_unresolved():
    dataset = _make_dataset([w * u.nm for w in np.arange(400.0, 1000.0, 5.0)])
    band = RasterDataDynamicBand(dataset, wavelength_value=2200.0, wavelength_units=u.nm, epsilon=10.0)
    with pytest.raises(ValueError, match="No band"):
        band.get_data()


@pytest.mark.parametrize(
    "description, wavelength, expected",
    [
        ("860.47", 860.47 * u.nm, "Band 0: 860.47 nm"),
        ("", 860.47 * u.nm, "Band 0: 860.47 nm"),
        ("0.86", 0.86 * u.um, "Band 0: 0.86 um"),
        ("860.47 nm", 860.47 * u.nm, "Band 0: 860.47 nm"),
        ("Red edge", 705.0 * u.nm, "Band 0: Red edge"),
        ("860.47", None, "Band 0: 860.47"),
    ],
)
def test_band_label_shows_wavelength_units(description, wavelength, expected):
    dataset = _make_dataset(None, bands=1)
    info = {"index": 0, "description": description}
    if wavelength is not None:
        info["wavelength"] = wavelength
    dataset.set_band_list([info])
    assert dataset.get_band_label(0) == expected


@pytest.mark.multiprocessing
@pytest.mark.parametrize(
    "preset_name, formula",
    [
        ("NDVI", lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"])),
        ("SAVI", lambda b: 1.5 * (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.5)),
        ("NDWI", lambda b: (b["r860"] - b["r1240"]) / (b["r860"] + b["r1240"])),
        ("MSI", lambda b: b["r1600"] / b["r820"]),
    ],
)
def test_preset_expression_matches_published_formula(preset_name, formula):
    preset = get_spectral_index_preset(preset_name)
    rng = np.random.default_rng(827)
    arrays = {var: rng.uniform(0.05, 0.6, size=(4, 5)).astype(np.float32) for var in preset.band_wavelengths}
    variables = {var: (VariableType.IMAGE_BAND, arr) for var, arr in arrays.items()}
    expr_info = get_bandmath_expr_info(preset.expression, variables, {})

    process_manager = bandmath.start_bandmath_evaluation(
        succeeded_callback=lambda _: None,
        status_callback=lambda _: None,
        error_callback=lambda _: None,
        bandmath_expr=preset.expression,
        expr_info=expr_info,
        result_name="preset",
        cache=DataCache(),
        variables=variables,
        functions={},
    )
    process_manager.get_task().wait()
    [(result_type, result, _name, _info)] = process_manager.get_task().get_result()

    assert result_type == VariableType.IMAGE_BAND
    np.testing.assert_allclose(np.asarray(result), formula(arrays), rtol=1e-5)

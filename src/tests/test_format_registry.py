"""Tests for raster format dispatch -- the registry, identification, and overrides.

Historically ``load_from_file`` attempted every registered format against every
file and kept whichever happened to succeed last, so which implementation opened
a file depended on dict declaration order and on how many other formats also
accepted it.  These tests pin the replacement:

* candidate *ordering* is declared, not accidental,
* a format that positively identifies a file wins immediately,
* exactly one file handle is opened per load,
* an explicit ``format=`` is obeyed and fails loudly rather than falling back.

They also lock two file-naming regressions the old sidecar resolution had:  an
ENVI cube whose data file is ``.dat``, and a GeoTIFF spelled ``.tiff``.
"""

import os

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

from osgeo import gdal

from wiser.raster.dataset_impl import (
    Confidence,
    ENVI_GDALRasterDataImpl,
    GDALRasterDataImpl,
    GTiff_GDALRasterDataImpl,
    NetCDF_GDALRasterDataImpl,
    PDS3_GDALRasterDataImpl,
    SaveState,
)
from wiser.raster.format_registry import (
    RASTER_FORMATS,
    candidates_for,
    format_for_impl,
    format_names,
    get_format,
)
from wiser.raster.loader import RasterDataLoader

pytestmark = [pytest.mark.functional]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_raster(path, driver):
    """Create a small single-band raster at ``path`` using ``driver``."""
    gdal.UseExceptions()
    ds = gdal.GetDriverByName(driver).Create(str(path), 8, 8, 1, gdal.GDT_Int16)
    ds.GetRasterBand(1).WriteArray(np.arange(64, dtype=np.int16).reshape(8, 8))
    ds = None
    return str(path)


@pytest.fixture
def envi_dat(tmp_path):
    """An ENVI cube whose data file is ``.dat`` (not the assumed ``.img``)."""
    _write_raster(tmp_path / "scene.dat", "ENVI")
    return tmp_path


@pytest.fixture
def tiff_with_stray_hdr(tmp_path):
    """A GeoTIFF spelled ``.tiff`` with an unrelated same-stem ENVI header.

    GDAL's ENVI driver identifies any file that has a same-stem ``.hdr`` beside
    it, so without an extension restriction ENVI would claim this GeoTIFF.
    """
    _write_raster(tmp_path / "pic.tiff", "GTiff")
    _write_raster(tmp_path / "other.dat", "ENVI")
    os.replace(tmp_path / "other.hdr", tmp_path / "pic.hdr")
    return tmp_path


# ---------------------------------------------------------------------------
# Registry shape and candidate ordering
# ---------------------------------------------------------------------------


def test_every_format_has_a_unique_name():
    names = format_names()
    assert len(names) == len(set(names)), "format names are the override tokens; they must be unique"


def test_lookup_is_case_insensitive():
    assert get_format("envi") is get_format("ENVI")
    assert get_format("no-such-format") is None


def test_exactly_one_fallback_and_it_sorts_last():
    fallbacks = [s for s in RASTER_FORMATS if s.is_fallback]
    assert len(fallbacks) == 1

    for path in ("a.hdr", "a.tif", "a.nc", "a.unknown"):
        assert candidates_for(path)[-1].is_fallback, f"fallback must be last for {path}"


def test_extension_promotes_but_never_excludes():
    """The extension orders the search; it does not restrict it."""
    order = candidates_for("scene.nc")

    assert order[0].name == "NetCDF", "the format claiming .nc should be tried first"
    # Every non-fallback format is still a candidate -- a file with a misleading
    # or unusual extension must still be loadable.
    assert {s.name for s in order} == {s.name for s in RASTER_FORMATS}


def test_ordering_is_by_priority_not_declaration_order():
    """Two formats claim .img; the higher-priority one is tried first."""
    order = [s.name for s in candidates_for("scene.img")]
    assert order.index("PDS3") < order.index("ENVI")

    within_claiming = [s.priority for s in candidates_for("scene.img") if s.claims_extension("scene.img")]
    assert within_claiming == sorted(within_claiming, reverse=True)


def test_files_without_an_extension_are_still_dispatched():
    order = candidates_for("scene")
    assert order[0].name == "ENVI", "ENVI data files commonly have no extension"
    assert order[-1].is_fallback


# ---------------------------------------------------------------------------
# identify()
# ---------------------------------------------------------------------------


def test_identify_recognizes_its_own_format(envi_dat):
    assert ENVI_GDALRasterDataImpl.identify(str(envi_dat / "scene.dat")) == Confidence.YES


def test_identify_resolves_a_sidecar_before_deciding(envi_dat):
    """Selecting the .hdr must identify as ENVI, not fail on the header itself."""
    assert ENVI_GDALRasterDataImpl.identify(str(envi_dat / "scene.hdr")) == Confidence.YES


def test_identify_rejects_other_formats(envi_dat):
    scene = str(envi_dat / "scene.dat")
    for impl in (GTiff_GDALRasterDataImpl, PDS3_GDALRasterDataImpl, NetCDF_GDALRasterDataImpl):
        assert impl.identify(scene) == Confidence.NO, f"{impl.__name__} should not claim an ENVI file"


def test_catch_all_never_claims_certainty(envi_dat):
    """The GDAL fallback must defer to every named format, never outrank one."""
    assert GDALRasterDataImpl.identify(str(envi_dat / "scene.dat")) == Confidence.MAYBE


def test_identify_does_not_raise_on_a_nonexistent_file(tmp_path):
    missing = str(tmp_path / "nope.img")
    for impl in (ENVI_GDALRasterDataImpl, GTiff_GDALRasterDataImpl, GDALRasterDataImpl):
        assert impl.identify(missing) in (Confidence.NO, Confidence.MAYBE)


def test_identify_does_not_open_a_handle(envi_dat, monkeypatch):
    opens = []
    monkeypatch.setattr(gdal, "OpenEx", lambda *a, **k: opens.append(a) or None)

    ENVI_GDALRasterDataImpl.identify(str(envi_dat / "scene.dat"))
    assert opens == [], "identify() must not open the dataset"


def test_data_extensions_exclude_a_same_stem_impostor(tiff_with_stray_hdr):
    """A stray .hdr beside a .tiff must not make it an ENVI dataset."""
    pic = str(tiff_with_stray_hdr / "pic.tiff")

    assert ENVI_GDALRasterDataImpl.identify(pic) == Confidence.NO
    assert GTiff_GDALRasterDataImpl.identify(pic) == Confidence.YES


# ---------------------------------------------------------------------------
# Sidecar resolution -- the two regressions
# ---------------------------------------------------------------------------


def test_envi_header_resolves_to_a_dat_data_file(envi_dat):
    """Regression:  only "" and .img were tried, so .dat cubes failed to open."""
    resolved = ENVI_GDALRasterDataImpl.get_load_filename(str(envi_dat / "scene.hdr"))
    assert os.path.basename(resolved) == "scene.dat"


def test_envi_header_with_no_data_file_reports_what_it_tried(tmp_path):
    (tmp_path / "orphan.hdr").write_text("ENVI\n")

    with pytest.raises(ValueError, match="orphan"):
        ENVI_GDALRasterDataImpl.get_load_filename(str(tmp_path / "orphan.hdr"))


def test_world_file_resolves_to_a_tiff_spelled_in_full(tmp_path):
    """Regression:  .tfw resolved only to .tif, never to .tiff."""
    _write_raster(tmp_path / "pic.tiff", "GTiff")
    (tmp_path / "pic.tfw").write_text("1.0\n0.0\n0.0\n-1.0\n0.0\n0.0\n")

    resolved = GTiff_GDALRasterDataImpl.get_load_filename(str(tmp_path / "pic.tfw"))
    assert os.path.basename(resolved) == "pic.tiff"


# ---------------------------------------------------------------------------
# load_from_file
# ---------------------------------------------------------------------------


def test_load_picks_the_identifying_format(envi_dat):
    (ds,) = RasterDataLoader().load_from_file(str(envi_dat / "scene.hdr"), interactive=False)
    assert isinstance(ds.get_impl(), ENVI_GDALRasterDataImpl)
    assert ds.get_name() == "scene.dat"


def test_load_opens_exactly_one_handle(envi_dat, monkeypatch):
    """The old loop opened one dataset per candidate driver and discarded all but one."""
    opens = []
    real_open = gdal.OpenEx
    monkeypatch.setattr(gdal, "OpenEx", lambda *a, **k: (opens.append(a[0]), real_open(*a, **k))[1])

    RasterDataLoader().load_from_file(str(envi_dat / "scene.hdr"), interactive=False)
    assert len(opens) == 1, f"expected a single open, got {opens}"


def test_load_prefers_the_certain_format_over_a_merely_possible_one(tiff_with_stray_hdr):
    (ds,) = RasterDataLoader().load_from_file(str(tiff_with_stray_hdr / "pic.tiff"), interactive=False)
    assert isinstance(ds.get_impl(), GTiff_GDALRasterDataImpl)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        RasterDataLoader().load_from_file(str(tmp_path / "absent.img"))


def test_unreadable_file_raises_rather_than_returning_nothing(tmp_path):
    junk = tmp_path / "junk.xyz"
    junk.write_bytes(b"not a raster at all")

    with pytest.raises(ValueError, match="unsupported format"):
        RasterDataLoader().load_from_file(str(junk), interactive=False)


def test_gtiff_raises_when_gdal_returns_no_dataset(tmp_path, monkeypatch):
    """OpenEx can return None without raising; try_load_file must not wrap None."""
    tif = _write_raster(tmp_path / "pic.tif", "GTiff")
    monkeypatch.setattr(gdal, "OpenEx", lambda *a, **k: None)

    with pytest.raises(ValueError, match="Unable to open"):
        GTiff_GDALRasterDataImpl.try_load_file(tif)


def test_envi_raises_when_gdal_returns_no_dataset(envi_dat, monkeypatch):
    monkeypatch.setattr(gdal, "OpenEx", lambda *a, **k: None)

    with pytest.raises(ValueError, match="Unable to open"):
        ENVI_GDALRasterDataImpl.try_load_file(str(envi_dat / "scene.hdr"))


# ---------------------------------------------------------------------------
# Explicit override
# ---------------------------------------------------------------------------


def test_override_selects_the_named_format(envi_dat):
    (ds,) = RasterDataLoader().load_from_file(str(envi_dat / "scene.hdr"), interactive=False, format="ENVI")
    assert isinstance(ds.get_impl(), ENVI_GDALRasterDataImpl)


def test_override_fails_loudly_instead_of_guessing_again(tiff_with_stray_hdr):
    """A wrong override must surface the mistake, not silently detect something else."""
    with pytest.raises(ValueError):
        RasterDataLoader().load_from_file(
            str(tiff_with_stray_hdr / "pic.tiff"), interactive=False, format="NetCDF"
        )


def test_unknown_override_name_lists_the_valid_ones(envi_dat):
    with pytest.raises(ValueError, match="Unknown raster format") as excinfo:
        RasterDataLoader().load_from_file(str(envi_dat / "scene.dat"), format="Nonsense")

    for name in format_names():
        assert name in str(excinfo.value)


# ---------------------------------------------------------------------------
# Mapping an opened dataset back to its format
# ---------------------------------------------------------------------------


def test_format_for_impl_round_trips(envi_dat):
    (ds,) = RasterDataLoader().load_from_file(str(envi_dat / "scene.hdr"), interactive=False)
    assert format_for_impl(ds.get_impl()) == "ENVI"


def test_format_for_impl_matches_on_exact_type_not_subclass():
    """A subclass must not be reported as its registered base class."""

    class _Subclass(ENVI_GDALRasterDataImpl):
        def __init__(self):
            # No dataset is needed to check a type, but the inherited __del__
            # inspects both of these, so give it something benign to find.
            self.gdal_dataset = None
            self._save_state = SaveState.UNKNOWN

    assert format_for_impl(_Subclass()) is None

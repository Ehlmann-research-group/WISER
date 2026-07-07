"""Spectrum persistence (issue #620).

Persists the active spectrum and the collected spectra, using the dependency
resolver (#617) to pick, per spectrum, between a faithful reference and a
self-contained snapshot.  Spectra are the resolver's first consumer, since they
are the first state with a cut-able edge: a dataset-backed spectrum holds a
*live* dataset, so referencing an unsaved dataset would dangle -- freeze the
computed values instead.

A spectrum serializes to one of three ``kind``s:

* ``numpy`` -- self-contained values + wavelengths + metadata.  Used for a
  ``NumPyArraySpectrum`` (always faithful) and for any dataset-backed spectrum
  whose dataset is not being saved (frozen: data preserved, liveness lost).
* ``raster-backed`` -- a ``SpectrumAtPoint`` referenced by dataset id + point +
  area; rebuilt live against the restored dataset.
* ``roi-average`` -- an ``ROIAverageSpectrum`` referenced by dataset id + roi id.

Value/wavelength arrays are 1-D and small, so they are inlined in the manifest
rather than written to array sidecars.  Spectrum ids are not preserved: the
app_state ``_all_spectra`` index is ``[DERIVED]`` and nothing references a
spectrum by id, so the manifest list order is the only identity that matters.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from astropy import units as u

from wiser.raster.spectrum import (
    NumPyArraySpectrum,
    RasterDataSetSpectrum,
    ROIAverageSpectrum,
    Spectrum,
    SpectrumAtPoint,
    SpectrumAverageMode,
)

from ..resolver import Dependency, DependencyResolver, SavePolicy, resolver_for_all_datasets

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

KIND_NUMPY = "numpy"
KIND_RASTER_BACKED = "raster-backed"
KIND_ROI_AVERAGE = "roi-average"


def spectrum_dependencies(spectrum: Spectrum) -> List[Dependency]:
    """The datasets/ROIs a spectrum depends on (empty for a self-contained one).

    Dataset-backed spectra hold live objects, so the id is dereferenced here; a
    dataset/ROI that has lost its app_state id yields a ``None`` id, which the
    resolver treats as a cut edge (forcing a snapshot).
    """
    deps: List[Dependency] = []
    if isinstance(spectrum, RasterDataSetSpectrum):
        dataset = spectrum.get_dataset()
        deps.append(Dependency("dataset", dataset.get_id() if dataset is not None else None))
    if isinstance(spectrum, ROIAverageSpectrum):
        roi = spectrum.get_roi()
        deps.append(Dependency("roi", roi.get_id() if roi is not None else None))
    return deps


def spectrum_to_pyrep(spectrum: Spectrum, resolver: DependencyResolver) -> Dict[str, Any]:
    """Serialize one spectrum, faithful if its dependencies are saved else frozen."""
    decision = resolver.classify(spectrum_dependencies(spectrum), snapshotable=True)
    if decision.policy is SavePolicy.FAITHFUL and isinstance(spectrum, RasterDataSetSpectrum):
        return _reference_pyrep(spectrum)
    # NumPyArraySpectrum (self-contained) and every SNAPSHOT freeze to a numpy entry.
    return _snapshot_pyrep(spectrum)


def save_spectra(
    app_state: "ApplicationState",
    manifest: Dict[str, Any],
    resolver: Optional[DependencyResolver] = None,
) -> None:
    """Write the collected spectra (ordered) and the active spectrum into the manifest.

    Without an explicit ``resolver`` (until the save dialog #626 lets the user
    deselect dataset roots), every dataset is treated as saved.
    """
    if resolver is None:
        resolver = resolver_for_all_datasets(app_state)

    active = app_state.get_active_spectrum()
    manifest["spectra"] = {
        "collected": [spectrum_to_pyrep(s, resolver) for s in app_state.get_collected_spectra()],
        "active": spectrum_to_pyrep(active, resolver) if active is not None else None,
    }


def load_spectra(manifest: Dict[str, Any], app_state: "ApplicationState") -> List[Dict[str, Any]]:
    """Reconstruct spectra from the manifest into ``app_state``.

    Runs after datasets (#618) and ROIs (#619) so faithful references resolve.
    Returns the manifest entries that could not be restored (a faithful
    reference whose dataset/ROI is absent, or an unknown ``kind``) so the caller
    can warn without aborting the load.
    """
    section = manifest.get("spectra", {})
    dropped: List[Dict[str, Any]] = []

    for entry in section.get("collected", []):
        spectrum = spectrum_from_pyrep(entry, app_state)
        if spectrum is None:
            dropped.append(entry)
            continue
        app_state.collect_spectrum(spectrum)

    active_entry = section.get("active")
    if active_entry is not None:
        spectrum = spectrum_from_pyrep(active_entry, app_state)
        if spectrum is None:
            dropped.append(active_entry)
        else:
            app_state.set_active_spectrum(spectrum)

    return dropped


def spectrum_from_pyrep(entry: Dict[str, Any], app_state: "ApplicationState") -> Optional[Spectrum]:
    """Reconstruct one spectrum, or ``None`` if it cannot be restored.

    ``None`` covers a missing faithful dependency, an unknown ``kind``, and a
    malformed entry (a missing required field or an unparseable value): a single
    bad entry is dropped and reported by :func:`load_spectra`, never allowed to
    abort opening the project.
    """
    kind = entry.get("kind")
    try:
        if kind == KIND_NUMPY:
            return _numpy_from_pyrep(entry)
        if kind == KIND_RASTER_BACKED:
            return _raster_backed_from_pyrep(entry, app_state)
        if kind == KIND_ROI_AVERAGE:
            return _roi_average_from_pyrep(entry, app_state)
    except (KeyError, TypeError, ValueError):
        return None
    return None


# -- serialize ------------------------------------------------------------------


def _reference_pyrep(spectrum: RasterDataSetSpectrum) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "name": spectrum.get_name(),
        "color": spectrum.get_color(),
        "avg_mode": spectrum.get_avg_mode().name,
        "dataset_id": spectrum.get_dataset().get_id(),
    }
    if isinstance(spectrum, ROIAverageSpectrum):
        entry["kind"] = KIND_ROI_AVERAGE
        entry["roi_id"] = spectrum.get_roi().get_id()
    else:  # SpectrumAtPoint
        entry["kind"] = KIND_RASTER_BACKED
        entry["point"] = list(spectrum.get_point())
        entry["area"] = list(spectrum.get_area())
    return entry


def _snapshot_pyrep(spectrum: Spectrum) -> Dict[str, Any]:
    values = np.asarray(spectrum.get_spectrum())
    entry: Dict[str, Any] = {
        "kind": KIND_NUMPY,
        "name": spectrum.get_name(),
        "color": spectrum.get_color(),
        "source_name": spectrum.get_source_name(),
        "editable": spectrum.is_editable(),
        "discardable": spectrum.is_discardable(),
        "values": values.tolist(),
        "bad_bands": np.asarray(spectrum.get_bad_bands()).astype(bool).tolist(),
    }
    if spectrum.has_wavelengths():
        entry["wavelengths"] = [float(w.value) for w in spectrum.get_wavelengths()]
        units = spectrum.get_wavelength_units()
        entry["wavelength_units"] = str(units) if units is not None else None
    return entry


# -- deserialize ----------------------------------------------------------------


def _numpy_from_pyrep(entry: Dict[str, Any]) -> NumPyArraySpectrum:
    arr = np.asarray(entry["values"], dtype=float)
    spectrum = NumPyArraySpectrum(
        arr,
        name=entry.get("name"),
        source_name=entry.get("source_name"),
        wavelengths=_wavelengths_from_pyrep(entry),
        editable=entry.get("editable", True),
        discardable=entry.get("discardable", True),
    )
    bad_bands = entry.get("bad_bands")
    if bad_bands is not None:
        spectrum.set_bad_bands(np.asarray(bad_bands, dtype=bool))
    _apply_color(spectrum, entry)
    return spectrum


def _raster_backed_from_pyrep(entry: Dict[str, Any], app_state: "ApplicationState") -> Optional[Spectrum]:
    dataset = _lookup_dataset(app_state, entry.get("dataset_id"))
    if dataset is None:
        return None
    point, area = entry.get("point"), entry.get("area")
    if point is None or area is None:
        # point/area identify which pixel(s) the spectrum reads; defaulting a
        # missing one would restore a silently wrong spectrum, so drop instead.
        return None
    spectrum = SpectrumAtPoint(dataset, tuple(point), tuple(area), _avg_mode(entry))
    _apply_identity(spectrum, entry)
    return spectrum


def _roi_average_from_pyrep(entry: Dict[str, Any], app_state: "ApplicationState") -> Optional[Spectrum]:
    dataset = _lookup_dataset(app_state, entry.get("dataset_id"))
    roi_id = entry.get("roi_id")
    roi = app_state.get_roi(id=roi_id) if roi_id is not None else None
    if dataset is None or roi is None:
        return None
    spectrum = ROIAverageSpectrum(dataset, roi, _avg_mode(entry))
    _apply_identity(spectrum, entry)
    return spectrum


def _wavelengths_from_pyrep(entry: Dict[str, Any]) -> Optional[List[Any]]:
    wavelengths = entry.get("wavelengths")
    if wavelengths is None:
        return None
    unit_str = entry.get("wavelength_units")
    if unit_str:
        unit = u.Unit(unit_str)
        return [w * unit for w in wavelengths]
    return list(wavelengths)


def _lookup_dataset(app_state: "ApplicationState", ds_id: Any):
    if ds_id is None or not app_state.has_dataset(ds_id):
        return None
    return app_state.get_dataset(ds_id)


def _avg_mode(entry: Dict[str, Any]) -> SpectrumAverageMode:
    name = entry.get("avg_mode")
    if name in SpectrumAverageMode.__members__:
        return SpectrumAverageMode[name]
    return SpectrumAverageMode.MEAN


def _apply_identity(spectrum: Spectrum, entry: Dict[str, Any]) -> None:
    name = entry.get("name")
    if name is not None:
        spectrum.set_name(name)
    _apply_color(spectrum, entry)


def _apply_color(spectrum: Spectrum, entry: Dict[str, Any]) -> None:
    color = entry.get("color")
    if color is not None:
        spectrum.set_color(color)

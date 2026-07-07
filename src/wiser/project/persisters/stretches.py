"""Contrast-stretch persistence (issue #622).

The committed per-band contrast stretches (``ApplicationState._stretches``, keyed
by ``(dataset_id, band_index)``) are ``[SOURCE]`` state lost on close; without
them a restored session falls back to a default stretch on every band and looks
visibly wrong.  Each stretch is a leaf on the dataset cascade -- a stretch cannot
outlive its dataset (``ApplicationState.remove_dataset`` deletes a dataset's
stretches with it), so a stretch whose dataset is not saved is dropped rather
than snapshotted, and on load a stretch whose dataset was not restored is dropped
rather than left as an orphan.

Stretches form a small closed polymorphic hierarchy with no serialization of
their own.  Like the spectra persister, the dispatch lives here rather than on
the classes: the ``...UsingNumba`` variants are numba jitclasses that cannot host
plain methods, and only their parameters are needed.  ``stretch_to_pyrep`` reads
the plain fields off whichever variant is held (jitclass fields are readable from
Python); ``stretch_from_pyrep`` rebuilds the pure-Python variant, which is
behaviorally identical -- the JIT variant is a performance choice re-derived when
the stretch builder next recomputes it.

A ``StretchComposite`` (a conditioner wrapping a base stretch) serializes its two
halves recursively.  ``StretchHistEqualize`` is reconstructed through its normal
constructor: it retains only the normalized ``_cdf`` and ``_histo_edges``, but
because the constructor renormalizes the CDF by its last element, feeding back
``diff(cdf) * diff(edges)`` as the histogram bins reproduces the identical CDF.
``StretchDecorrelation`` carries no per-instance state (it recomputes its joint
transform from the data at apply time), so it needs only its type tag.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from wiser.raster.stretch import (
    StretchBase,
    StretchBaseUsingNumba,
    StretchComposite,
    StretchDecorrelation,
    StretchHistEqualize,
    StretchHistEqualizeUsingNumba,
    StretchLinear,
    StretchLinearUsingNumba,
    StretchLog2,
    StretchLog2UsingNumba,
    StretchSquareRoot,
    StretchSquareRootUsingNumba,
)

from ..resolver import Dependency, DependencyResolver, resolver_for_all_datasets

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

TAG_LINEAR = "linear"
TAG_EQUALIZE = "equalize"
TAG_DECORRELATION = "decorrelation"
TAG_SQRT = "sqrt"
TAG_LOG2 = "log2"
TAG_COMPOSITE = "composite"
TAG_NONE = "none"

# Each stretch type is held as either its pure-Python class or a numba jitclass
# variant; match both when serializing.  When numba is unavailable the wrapper
# aliases the variant back to the plain class, so the tuple simply repeats.
_LINEAR = (StretchLinear, StretchLinearUsingNumba)
_EQUALIZE = (StretchHistEqualize, StretchHistEqualizeUsingNumba)
_SQRT = (StretchSquareRoot, StretchSquareRootUsingNumba)
_LOG2 = (StretchLog2, StretchLog2UsingNumba)
_BASE = (StretchBase, StretchBaseUsingNumba)


def save_stretches(
    app_state: "ApplicationState",
    manifest: Dict[str, Any],
    resolver: Optional[DependencyResolver] = None,
) -> None:
    """Write committed per-band stretches into ``manifest['stretches']``.

    A stretch whose dataset is not being saved is dropped: it is a cascade leaf
    with nothing to snapshot once its dataset is gone.  Without an explicit
    ``resolver`` every dataset is treated as saved.
    """
    if resolver is None:
        resolver = resolver_for_all_datasets(app_state)

    entries: List[Dict[str, Any]] = []
    for (ds_id, band_index), stretch in app_state.get_all_stretches().items():
        if stretch is None or not resolver.is_saved(Dependency("dataset", ds_id)):
            continue
        pyrep = stretch_to_pyrep(stretch)
        if pyrep is None:
            continue
        entries.append({"dataset_id": ds_id, "band_index": band_index, "stretch": pyrep})
    manifest["stretches"] = entries


def load_stretches(manifest: Dict[str, Any], app_state: "ApplicationState") -> List[Dict[str, Any]]:
    """Reconstruct committed stretches from the manifest into ``app_state``.

    Runs after datasets (#618).  A stretch whose dataset was not restored is
    dropped to preserve the invariant that a stretch never outlives its dataset;
    an unreconstructable stretch is dropped too.  Returns the dropped entries so
    the caller can warn without aborting the load.
    """
    dropped: List[Dict[str, Any]] = []
    for entry in manifest.get("stretches", []):
        ds_id = entry.get("dataset_id")
        band_index = entry.get("band_index")
        if ds_id is None or band_index is None or not app_state.has_dataset(ds_id):
            dropped.append(entry)
            continue
        stretch = stretch_from_pyrep(entry.get("stretch", {}))
        if stretch is None:
            dropped.append(entry)
            continue
        app_state.set_stretches(ds_id, (band_index,), [stretch])
    return dropped


def stretch_to_pyrep(stretch: Any) -> Optional[Dict[str, Any]]:
    """Serialize one stretch to a pyrep dict, or ``None`` for an unknown type."""
    if isinstance(stretch, StretchComposite):
        first = stretch_to_pyrep(stretch.first())
        second = stretch_to_pyrep(stretch.second())
        if first is None or second is None:
            return None
        return {"type": TAG_COMPOSITE, "first": first, "second": second}
    if isinstance(stretch, _LINEAR):
        return {"type": TAG_LINEAR, "lower": float(stretch._lower), "upper": float(stretch._upper)}
    if isinstance(stretch, _EQUALIZE):
        return {
            "type": TAG_EQUALIZE,
            "cdf": np.asarray(stretch._cdf, dtype=float).tolist(),
            "histo_edges": np.asarray(stretch._histo_edges, dtype=float).tolist(),
        }
    if isinstance(stretch, StretchDecorrelation):
        return {"type": TAG_DECORRELATION}
    if isinstance(stretch, _SQRT):
        return {"type": TAG_SQRT}
    if isinstance(stretch, _LOG2):
        return {"type": TAG_LOG2}
    # Checked last: every per-band stretch subclasses StretchBase.
    if isinstance(stretch, _BASE):
        return {"type": TAG_NONE}
    return None


def stretch_from_pyrep(data: Dict[str, Any]) -> Optional[Any]:
    """Reconstruct one stretch, or ``None`` if the entry is malformed/unknown.

    Always builds the pure-Python variant; a malformed entry (a degenerate linear
    bound, a missing field) is dropped rather than raised so one bad stretch
    cannot abort opening the project.
    """
    if not isinstance(data, dict):
        # A null or non-dict entry (e.g. ``{"stretch": null}`` from a hand-edited
        # manifest) is dropped rather than dereferenced.
        return None
    tag = data.get("type")
    try:
        if tag == TAG_COMPOSITE:
            first = stretch_from_pyrep(data.get("first", {}))
            second = stretch_from_pyrep(data.get("second", {}))
            if first is None or second is None:
                return None
            return StretchComposite(first, second)
        if tag == TAG_LINEAR:
            return StretchLinear(data["lower"], data["upper"])
        if tag == TAG_EQUALIZE:
            return _equalize_from_pyrep(data)
        if tag == TAG_DECORRELATION:
            return StretchDecorrelation()
        if tag == TAG_SQRT:
            return StretchSquareRoot()
        if tag == TAG_LOG2:
            return StretchLog2()
        if tag == TAG_NONE:
            return StretchBase()
    except (KeyError, TypeError, ValueError, IndexError):
        # IndexError guards an empty equalize CDF/edges (``cdf[-1]`` in the
        # stretch constructor); the rest guard malformed scalar fields.
        return None
    return None


def _equalize_from_pyrep(data: Dict[str, Any]) -> StretchHistEqualize:
    """Rebuild a histogram-equalize stretch through its normal constructor.

    The stretch keeps only the normalized ``_cdf`` and ``_histo_edges``; since the
    constructor renormalizes the CDF by its last element, ``diff(cdf) *
    diff(edges)`` are histogram bins that reproduce the same CDF.
    """
    cdf = np.asarray(data["cdf"], dtype=float)
    edges = np.asarray(data["histo_edges"], dtype=float)
    bins = np.diff(cdf, prepend=0.0) * np.diff(edges)
    return StretchHistEqualize(bins, edges)

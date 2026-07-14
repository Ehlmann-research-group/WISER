"""Band-math saved-expression persistence (issue #625).

The band-math saved-expression list is ``[SOURCE]`` state.  It historically lived
only inside the ``BandMathDialog``'s saved-expressions combo-box; it is now backed
by ``ApplicationState`` (``get_bandmath_expressions`` / ``set_bandmath_expressions``),
so the persister reads and writes a plain list of expression strings like every
other item and the list survives the dialog closing.  On load the store is
repopulated; the dialog seeds its combo-box from it the next time it opens (or the
load orchestrator #627 refreshes an already-open dialog).  Independent of datasets.
"""

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


def save_bandmath(app_state: "ApplicationState", manifest: Dict[str, Any]) -> None:
    """Write the saved band-math expressions into ``manifest['bandmath_expressions']``."""
    manifest["bandmath_expressions"] = list(app_state.get_bandmath_expressions())


def load_bandmath(manifest: Dict[str, Any], app_state: "ApplicationState") -> List[Any]:
    """Restore the saved band-math expressions into ``app_state``.

    An absent section leaves the store untouched.  Non-string entries in a
    malformed list (or a non-list section) are dropped and returned so the caller
    can warn without aborting the load.
    """
    if "bandmath_expressions" not in manifest:
        return []
    raw = manifest["bandmath_expressions"]
    if not isinstance(raw, list):
        return [raw]
    expressions = [entry for entry in raw if isinstance(entry, str)]
    dropped = [entry for entry in raw if not isinstance(entry, str)]
    app_state.set_bandmath_expressions(expressions)
    return dropped

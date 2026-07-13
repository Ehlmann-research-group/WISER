"""Shared pyrep serialization convention for project files.

A *pyrep* is a plain, JSON-serializable ``dict`` describing one persistable
object.  Every persistable type exposes ``to_pyrep() -> dict`` and a
``from_pyrep(data)`` reconstructor; each dict carries a ``"type"`` tag so the
loader can dispatch to the right reconstructor.  Large numpy arrays are never
inlined -- they are written to the bundle's ``arrays/`` area and referenced by
key (see :class:`~wiser.project.bundle.ProjectBundle`).

``from_pyrep`` implementations should parse leniently: ignore unknown keys and
supply defaults for missing ones, so that adding a new *optional* field is a
non-breaking change that needs no migration (see :mod:`wiser.project.migrate`).
"""

from typing import Any, Callable, Dict

PYREP_TYPE_KEY = "type"
ARRAY_REF_KEY = "$array"


class UnknownPyrepType(Exception):
    """Raised when a pyrep dict carries a ``type`` tag with no reconstructor."""


_FROM_PYREP: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register_pyrep(type_tag: str, from_pyrep_fn: Callable[[Dict[str, Any]], Any]) -> None:
    """Register the reconstructor for a pyrep ``type`` tag."""
    _FROM_PYREP[type_tag] = from_pyrep_fn


def from_pyrep(data: Dict[str, Any]) -> Any:
    """Reconstruct the object described by a pyrep dict.

    Dispatches on the dict's ``"type"`` tag.  Raises :class:`UnknownPyrepType`
    for an unregistered tag so the loader can warn and continue rather than
    crash.
    """
    tag = data.get(PYREP_TYPE_KEY)
    reconstruct = _FROM_PYREP.get(tag)
    if reconstruct is None:
        raise UnknownPyrepType(f"No pyrep reconstructor registered for type {tag!r}")
    return reconstruct(data)


def array_ref(key: str) -> Dict[str, str]:
    """Build a manifest-embeddable reference to an array sidecar."""
    return {ARRAY_REF_KEY: key}


def is_array_ref(obj: Any) -> bool:
    """Return whether ``obj`` is an array-sidecar reference."""
    return isinstance(obj, dict) and ARRAY_REF_KEY in obj


def array_ref_key(obj: Dict[str, str]) -> str:
    """Return the relative sidecar key from an array-sidecar reference."""
    return obj[ARRAY_REF_KEY]

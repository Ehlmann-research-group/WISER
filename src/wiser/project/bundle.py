"""On-disk project bundle: ``manifest.json`` plus raster and array sidecars.

A bundle is a directory.  The single-file ``.wiserproj`` form is just that
directory zipped; :func:`zip_bundle` / :func:`unzip_bundle` convert between the
two.  Bulk data never goes in the manifest: large arrays are written to
``arrays/`` and referenced by key, and raster pixels to ``datasets/``.
"""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .migrate import CURRENT_FORMAT_VERSION, ProjectFormatError, migrate_up
from .pyrep import array_ref, array_ref_key, is_array_ref

PathLike = Union[str, Path]


def _resolve_within(root: PathLike, rel: str) -> Path:
    """Resolve ``rel`` under ``root``, refusing paths that escape the bundle.

    Returns the absolute resolved path when ``rel`` stays inside ``root``;
    raises :class:`~wiser.project.migrate.ProjectFormatError` otherwise. Keeps
    manifest- or archive-supplied paths from reaching outside the bundle
    directory (``../`` segments, absolute paths).
    """
    root = Path(root).resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ProjectFormatError(f"Path escapes the bundle root: {rel!r}") from None
    return target


class ProjectBundle:
    """A WISER project bundle rooted at a directory."""

    MANIFEST_NAME = "manifest.json"
    DATASETS_DIR = "datasets"
    ARRAYS_DIR = "arrays"
    EXTENSION = ".wiserproj"

    def __init__(self, root: PathLike):
        self._root = Path(root)

    @classmethod
    def create(cls, root: PathLike) -> "ProjectBundle":
        """Create (or reuse) an empty bundle directory at ``root``."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @classmethod
    def open(cls, root: PathLike) -> "ProjectBundle":
        """Open an existing bundle directory containing a manifest."""
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a project bundle directory: {root}")
        if not (root / cls.MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"No {cls.MANIFEST_NAME} in bundle: {root}")
        return cls(root)

    @property
    def root(self) -> Path:
        return self._root

    # -- manifest -----------------------------------------------------------

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        """Write ``manifest`` to ``manifest.json``, stamping the current
        ``format_version`` if the caller did not set one."""
        manifest = dict(manifest)
        manifest.setdefault("format_version", CURRENT_FORMAT_VERSION)
        path = self._root / self.MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def read_manifest(self) -> Dict[str, Any]:
        """Read ``manifest.json`` and migrate it up to the current schema.

        Raises :class:`~wiser.project.migrate.ProjectTooNewError` if the bundle
        was written by a newer WISER than this one understands.
        """
        path = self._root / self.MANIFEST_NAME
        manifest = json.loads(path.read_text())
        return migrate_up(manifest)

    @property
    def format_version(self) -> int:
        path = self._root / self.MANIFEST_NAME
        manifest = json.loads(path.read_text())
        return int(manifest.get("format_version", 1))

    # -- array sidecars -----------------------------------------------------

    def add_array(self, key: str, array: np.ndarray) -> Dict[str, str]:
        """Write ``array`` to ``arrays/<key>.npy`` and return a
        manifest-embeddable reference (``{"$array": "arrays/<key>.npy"}``)."""
        rel = f"{self.ARRAYS_DIR}/{key}.npy"
        dest = self._root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, array)
        return array_ref(rel)

    def fetch_array(self, ref: Union[Dict[str, str], str]) -> np.ndarray:
        """Load an array written with :meth:`add_array`, given either its
        reference dict or its relative key.

        The key is confined to the bundle directory: a reference that resolves
        outside the bundle is rejected rather than read, since the manifest may
        come from an untrusted source."""
        rel = array_ref_key(ref) if is_array_ref(ref) else ref
        return np.load(_resolve_within(self._root, rel))

    # -- raster sidecars ----------------------------------------------------

    def raster_sidecar_path(self, filename: str) -> Path:
        """Return the on-disk path under ``datasets/`` for a raster sidecar.

        The bundle owns the path; writing the raster bytes is the dataset
        persister's job (issue #618)."""
        dest = self._root / self.DATASETS_DIR
        dest.mkdir(parents=True, exist_ok=True)
        return dest / filename


def zip_bundle(bundle: ProjectBundle, zip_path: PathLike) -> Path:
    """Package a bundle directory into a single ``.wiserproj`` zip file."""
    zip_path = Path(zip_path)
    root = bundle.root
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root))
    return zip_path


def unzip_bundle(zip_path: PathLike, dest_dir: PathLike) -> ProjectBundle:
    """Extract a ``.wiserproj`` zip into ``dest_dir`` and open it as a bundle.

    Guards against zip-slip: a member whose path escapes ``dest_dir`` (via
    ``../`` or an absolute path) is refused before anything is extracted, since
    a project file may come from an untrusted source."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            _resolve_within(dest_dir, name)
        zf.extractall(dest_dir)
    return ProjectBundle.open(dest_dir)

"""
Tests for src/devtools/patch_cv2_config_for_bundle.py.

These tests run entirely in-process using a temporary directory. They are
skipped inside a frozen bundle because patch_cv2_config_for_bundle.py is a
build-time devtool and is not shipped inside the app.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

if getattr(sys, "frozen", False):
    pytest.skip(
        "patch_cv2_config_for_bundle is a build-time devtool; skipped in frozen bundle",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.unit,
]


def _load_patcher():
    """Import the devtools script by file path so it works without a package install."""
    script = Path(__file__).resolve().parent.parent / "devtools" / "patch_cv2_config_for_bundle.py"
    spec = importlib.util.spec_from_file_location("patch_cv2_config_for_bundle", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = _load_patcher()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOCK_CONFIG_39 = """\
import os
PYTHON_EXTENSIONS_PATHS = [
    os.path.join('/opt/micromamba/envs/wiser-prod/lib/python3.9/site-packages/cv2', 'python-3.9')
] + PYTHON_EXTENSIONS_PATHS
"""


def _make_cv2_dir(tmp: Path) -> Path:
    cv2_dir = tmp / "cv2"
    cv2_dir.mkdir()
    return cv2_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPatchCv2Config(unittest.TestCase):
    def test_versioned_config_is_overwritten(self):
        """config-3.9.py should be replaced with our frozen-aware template."""
        with tempfile.TemporaryDirectory() as tmp_str:
            cv2_dir = _make_cv2_dir(Path(tmp_str))
            config = cv2_dir / "config-3.9.py"
            config.write_text(_STOCK_CONFIG_39, encoding="utf-8")

            patcher.patch_cv2_directory(cv2_dir)

            result = config.read_text(encoding="utf-8")
            self.assertIn("sys._MEIPASS", result)
            self.assertIn("python-3.9", result)
            self.assertNotIn("micromamba", result)
            self.assertNotIn("/opt/", result)

    def test_frozen_branch_uses_meipass(self):
        """The generated file's frozen branch must reference sys._MEIPASS."""
        with tempfile.TemporaryDirectory() as tmp_str:
            cv2_dir = _make_cv2_dir(Path(tmp_str))
            (cv2_dir / "config-3.9.py").write_text(_STOCK_CONFIG_39, encoding="utf-8")

            patcher.patch_cv2_directory(cv2_dir)

            result = (cv2_dir / "config-3.9.py").read_text(encoding="utf-8")
            # The frozen branch must build the path from sys._MEIPASS.
            self.assertIn('os.path.join(sys._MEIPASS, "cv2", "python-3.9")', result)

    def test_non_frozen_branch_uses_file_relative_path(self):
        """The generated file's non-frozen branch must not hard-code any absolute path."""
        with tempfile.TemporaryDirectory() as tmp_str:
            cv2_dir = _make_cv2_dir(Path(tmp_str))
            (cv2_dir / "config-3.9.py").write_text(_STOCK_CONFIG_39, encoding="utf-8")

            patcher.patch_cv2_directory(cv2_dir)

            result = (cv2_dir / "config-3.9.py").read_text(encoding="utf-8")
            self.assertIn("__file__", result)

    def test_multiple_python_versions(self):
        """Patcher handles several versioned config files at once."""
        with tempfile.TemporaryDirectory() as tmp_str:
            cv2_dir = _make_cv2_dir(Path(tmp_str))
            for version in ("3.9", "3.10", "3.11"):
                (cv2_dir / f"config-{version}.py").write_text(_STOCK_CONFIG_39, encoding="utf-8")

            patcher.patch_cv2_directory(cv2_dir)

            for version in ("3.9", "3.10", "3.11"):
                result = (cv2_dir / f"config-{version}.py").read_text(encoding="utf-8")
                self.assertIn(f"python-{version}", result)
                self.assertNotIn("micromamba", result)

    def test_unversioned_config_is_not_touched(self):
        """config.py (no version digits) must not be modified."""
        with tempfile.TemporaryDirectory() as tmp_str:
            cv2_dir = _make_cv2_dir(Path(tmp_str))
            (cv2_dir / "config-3.9.py").write_text(_STOCK_CONFIG_39, encoding="utf-8")
            unversioned = cv2_dir / "config.py"
            original = "# original\nBINARIES_PATHS = []\n"
            unversioned.write_text(original, encoding="utf-8")

            patcher.patch_cv2_directory(cv2_dir)

            self.assertEqual(unversioned.read_text(encoding="utf-8"), original)

    def test_exits_nonzero_when_directory_missing(self):
        """Script must call sys.exit(1) when the cv2 directory does not exist."""
        with self.assertRaises(SystemExit) as ctx:
            patcher.patch_cv2_directory(Path("/this/path/does/not/exist/cv2"))
        self.assertEqual(ctx.exception.code, 1)

    def test_exits_nonzero_when_no_versioned_configs_found(self):
        """Script must call sys.exit(1) when the directory contains no config-M.m.py."""
        with tempfile.TemporaryDirectory() as tmp_str:
            cv2_dir = _make_cv2_dir(Path(tmp_str))
            (cv2_dir / "config.py").write_text("# only unversioned\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as ctx:
                patcher.patch_cv2_directory(cv2_dir)
            self.assertEqual(ctx.exception.code, 1)

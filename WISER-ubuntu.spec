# -*- mode: python ; coding: utf-8 -*-
"""
Ubuntu/Linux PyInstaller spec for WISER.

This script assumes you use conda/micromamba for environment management
and that CONDA_PREFIX is set when running pyinstaller.
"""
import glob
import sys
import os
import subprocess

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

# Make spec portable: derive project root from SPECPATH. :contentReference[oaicite:2]{index=2}
spec_dir = os.path.abspath(SPECPATH)
project_root = spec_dir  # assumes the spec file sits at repo root; adjust if needed

# If your repo layout is different (e.g., spec in docker/), you can do:
# project_root = os.path.abspath(os.path.join(spec_dir, ".."))

devtools_path = os.path.join(project_root, "src", "devtools")
sys.path.insert(0, os.path.abspath(devtools_path))

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT


block_cipher = None

# --- Data files to ship alongside the app (onedir bundle) ---
existing_datas = [
    ("./LICENSE", "."),
    ("./src/wiser/bandmath/bandmath.lark", "wiser/bandmath"),
    ("./src/wiser/data", "wiser/data"),
    ("./src/test_utils/test_datasets", "test_utils/test_datasets"),
    ("./src/test_utils/test_spectra", "test_utils/test_spectra"),
    ("./src/example_plugins", "example_plugins"),
    ("./src/tests", "tests"),
    ("./icons/wiser.iconset/icon_256x256.png", "icons/wiser.iconset")
]

existing_hidden_imports = [
    "PySide2.QtXml",
]

# --- GDAL plugins (Linux uses .so) ---
conda_env_prefix = os.environ.get("CONDA_PREFIX")
if not conda_env_prefix:
    raise RuntimeError(
        "CONDA_PREFIX is not set. Run pyinstaller from inside the conda env "
        "(e.g. `micromamba run -n wiser pyinstaller WISER-ubuntu.spec`)."
    )

existing_binaries = [
    (f"{conda_env_prefix}/lib/gdalplugins/gdal_HDF4.so", "gdalplugins"),
    (f"{conda_env_prefix}/lib/gdalplugins/gdal_HDF5.so", "gdalplugins"),
    (f"{conda_env_prefix}/lib/gdalplugins/gdal_netCDF.so", "gdalplugins"),
    (f"{conda_env_prefix}/lib/gdalplugins/gdal_JP2OpenJPEG.so", "gdalplugins"),
]

sqlite_libs = glob.glob(os.path.join(conda_env_prefix, "lib", "libsqlite3.so*"))
existing_binaries += [(p, ".") for p in sqlite_libs]


def _print_tuple_entries(title, entries):
    print(f"\n=== {title} ===")
    if not entries:
        print("(none)")
        return
    for entry in entries:
        print(entry)


cv2_build_env_script = os.path.join(project_root, "scripts", "debug_cv2_build_env.py")
print("\n=== CV2 BUILD ENV DIAGNOSTIC SCRIPT ===")
if os.path.isfile(cv2_build_env_script):
    print(f"Running: {sys.executable} {cv2_build_env_script}")
    cv2_build_env_result = subprocess.run(
        [sys.executable, cv2_build_env_script],
        cwd=project_root,
        check=False,
    )
    print(f"Diagnostic script exit code: {cv2_build_env_result.returncode}")
else:
    print(f"Diagnostic script not found: {cv2_build_env_script}")

# FIRST PASS: build Analysis to discover top-level modules
temp_a = Analysis(
    ["src/wiser/__main__.py"],
    pathex=[project_root],
    binaries=existing_binaries,
    datas=existing_datas,
    hiddenimports=existing_hidden_imports,
    hookspath=[],
    runtime_hooks=[
        "pyinstaller_hooks/set_wiser_env_prod.py",
        "pyinstaller_hooks/pyi_rth_cv2.py",
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# BUILD UP NEW hiddenimports by collecting submodules for every top-level package
top_modules = {entry[0].split(".", 1)[0] for entry in temp_a.pure}

IGNORED_TOP_PACKAGES = {
    "PySide2",
}

for pkg in sorted(top_modules):
    if any(pkg == ign or pkg.startswith(ign + ".") for ign in IGNORED_TOP_PACKAGES):
        continue
    existing_hidden_imports.extend(collect_submodules(pkg))

# Remove duplicates while preserving order
_seen = set()
_hidden = []
for m in existing_hidden_imports:
    if m not in _seen:
        _seen.add(m)
        _hidden.append(m)
existing_hidden_imports = _hidden

# OpenCV dynamic libs fix: collect cv2 shared objects (.so only on Linux)
cv2_binaries = collect_dynamic_libs(
    "cv2",
    search_patterns=["cv2*.so", "cv2*.dylib", "python-*/cv2*.so", "python-*/cv2*.dylib"],
)
cv2_data_files_diagnostic = collect_data_files("cv2")
existing_binaries += cv2_binaries

cv2_existing_binary_entries = [
    entry for entry in existing_binaries if isinstance(entry, tuple) and "cv2" in entry[0]
]

_print_tuple_entries("PYINSTALLER CV2 BINARIES", cv2_binaries)
_print_tuple_entries("PYINSTALLER CV2 DATA FILES", cv2_data_files_diagnostic)
_print_tuple_entries("CV2 BINARIES CURRENTLY INCLUDED IN existing_binaries", cv2_existing_binary_entries)

# SECOND PASS: rebuild Analysis with full hiddenimports
a = Analysis(
    ["src/wiser/__main__.py"],
    pathex=[project_root],
    binaries=existing_binaries,
    datas=existing_datas,
    hiddenimports=existing_hidden_imports,
    hookspath=[],
    runtime_hooks=[
        "pyinstaller_hooks/set_wiser_env_prod.py",
        "pyinstaller_hooks/pyi_rth_cv2.py",
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WISER_Bin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed/no console (use True if you want terminal logs)
    icon=['icons\\wiser.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="WISER",
)

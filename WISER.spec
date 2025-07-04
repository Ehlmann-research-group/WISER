# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_all, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

from pathlib import Path
import json, subprocess, shlex, os, sys
import importlib.metadata as imeta
from packaging.version import Version, InvalidVersion

datas = [('src\\wiser\\bandmath\\bandmath.lark', 'wiser\\bandmath')]
binaries = [('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\plugins\\platforms', 'platforms'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\plugins\\iconengines', 'iconengines'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_FITS.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_netCDF.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_HDF4.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_HDF5.dll', 'gdalplugins')]
hiddenimports = ['PySide2.QtSvg', 'PySide2.QtXml']
tmp_ret = collect_all('osgeo')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a1 = Analysis(
    ['src\\wiser\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)

print(f"------------------------d0sfr9804235n0h894ujb538053b3bas----------------------------")

def safe_version(dist):
    """Return normalized version or '0' if unavailable."""
    try:
        return imeta.version(dist)
    except imeta.PackageNotFoundError:
        return "0"

# 1) PyInstaller-bundled distributions  ──────────────────────────────────
dist_map = imeta.packages_distributions()
all_modules = {
    entry[0] for entry in a1.pure
} | {
    entry[0] for entry in a1.binaries
}
print(f"a1.binaries: {a1.binaries}")


top_modules = { tup[0].split('.', 1)[0] for tup in all_modules }          # tuples may be 3-tuple
frozen_pkgs = {
    d: safe_version(d)
    for mod in top_modules
    for d in dist_map.get(mod, [])
}

print(f"frozen_pkgs: {frozen_pkgs}")

# 2) Conda packages explicitly requested by the user  ───────────────────
conda_pkgs = {}
try:
    # '--from-history' gives just user-requested specs
    out = subprocess.check_output(
        ["conda", "env", "export", "--name", "wiser-source-2", "--from-history", "--json"],
        text=True, stderr=subprocess.DEVNULL
    )
    hist = json.loads(out)
    for spec in hist.get("dependencies", []):
        if isinstance(spec, str) and "=" in spec:
            name, ver, *_ = spec.split("=", 2)
            conda_pkgs[name] = ver
        else:
            name = spec.split("=")[0]
            print(f"name: {name}")
            conda_pkgs[name] = "N/A"
except Exception:
    print(f"!@! CONDA NOT FOUND CAUSE EXCEPTION")
    pass                                  # conda not on PATH, ignore

print(f"!@! conda_pkgs: {conda_pkgs}")

# 3) Pip packages the user installed explicitly  ────────────────────────
pip_pkgs = {}
try:
    out = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--not-required", "--format=json"],
        text=True, stderr=subprocess.DEVNULL
    )
    for row in json.loads(out):
        pip_pkgs[row["name"]] = row["version"]
except Exception:
    print(f"!#! PIP NOT FOUND CAUSE EXCEPTION")
    pass                                  # pip not available, ignore
print(f"!#! pip_pkgs: {pip_pkgs}")

# # 4) Merge them (PyInstaller > conda > pip)  ─────────────────────────────
# merged = {}
# for src in (pip_pkgs, conda_pkgs, frozen_pkgs):     # later dicts override earlier
#     merged.update(src)

# find the set of names present in all three sources
common_pkgs_conda = set(conda_pkgs) & set(frozen_pkgs)
common_pkgs_pip = set(pip_pkgs) & set(frozen_pkgs)

print(f"!$! common_pkgs_conda: {common_pkgs_conda}")
print()
print(f"!$! common_pkgs_pip: {common_pkgs_pip}")

# build merged dict just for those names
# pick the version from frozen_pkgs as the “canonical” one,
# but you could choose pip_pkgs or conda_pkgs if you prefer
merged_conda = {pkg: frozen_pkgs[pkg] for pkg in common_pkgs_conda}
merged_pip = {pkg: frozen_pkgs[pkg] for pkg in common_pkgs_pip}

print(f"!$! merged_conda: {merged_conda}")
print()
print(f"!$! merged_pip: {merged_pip}")

merged = {}
for src in (merged_pip, merged_conda):
    merged.update(src)

print(f"!$! merged: {merged}")

# 5) Write out requirements file  ───────────────────────────────────────
out_file = Path("wiser_frozen_requirements.txt")
with out_file.open("w", encoding="utf-8") as fh:
    for pkg in sorted(merged, key=str.lower):
        fh.write(f"{pkg}=={merged[pkg]}\n")

print(f"[spec] wrote {len(merged)} packages to {out_file}")

print(f"------------------------d0sfr9804235n0h894ujb538053b3bas----------------------------")

print(f"------------------------108j310u4nu10831904141vn41n4891y----------------------------")

print(f"a1.pure: {a1.pure}")
# 3) BUILD UP NEW hiddenimports by collecting submodules for every top-level package
top_modules = { entry[0].split('.', 1)[0] for entry in a1.pure }


IGNORED_TOP_PACKAGES = {
    "PySide2",
}

for pkg in sorted(top_modules):
    # if pkg is in the ignore list, or is a submodule of something in it, skip
    if any(pkg == ign or pkg.startswith(ign + ".")
           for ign in IGNORED_TOP_PACKAGES):
        continue
    hiddenimports.extend(collect_submodules(pkg))

print(f"top_modules: {top_modules}")
# for pkg in sorted(top_modules):
#     try:
#         # collect_submodules will only grab real submodules
#         hiddenimports.extend(collect_submodules(pkg))
#     except Exception:
#         # skip anything that isn't really a package
#         pass

# remove duplicates while preserving order
_seen = set()
_hidden = []
for m in hiddenimports:
    if m not in _seen:
        _seen.add(m)
        _hidden.append(m)
print(f"hidden: {_hidden}")
hiddenimports = _hidden

print(f"------------------------108j310u4nu10831904141vn41n4891y----------------------------")
# 4) SECOND PASS: rebuild Analysis with the full hiddenimports list
a = Analysis(
    ['src\\wiser\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)

# 5) the rest of your spec unchanged
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WISER',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons\\wiser.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WISER',
)
print(f"Done!")
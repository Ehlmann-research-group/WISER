# -*- mode: python ; coding: utf-8 -*-
'''
This script assumes you use conda for your environment management.
'''
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(SPECPATH), 'WISER', 'src', 'devtools')))

from PyInstaller.utils.hooks import collect_all, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

import subprocess

from write_analysis_deps import write_deps_from_analysis

conda_env_prefix = os.environ.get("CONDA_PREFIX")

if not conda_env_prefix:
    raise RuntimeError("Must be in a conda environment to run WISER's pyinstaller built script!")

# Whatever conda environment you use should have the below .dll's in Library\lib\gdalplugins
datas = [('src\\wiser\\bandmath\\bandmath.lark', 'wiser\\bandmath'), ('src\\wiser\\data', 'wiser\\data')]
binaries = [(f'{conda_env_prefix}\\Library\\plugins\\platforms', 'platforms'),
(f'{conda_env_prefix}\\Library\\plugins\\iconengines', 'iconengines'),
(f'{conda_env_prefix}\\Library\\lib\\gdalplugins\\gdal_FITS.dll', 'gdalplugins'),
(f'{conda_env_prefix}\\Library\\lib\\gdalplugins\\gdal_netCDF.dll', 'gdalplugins'),
(f'{conda_env_prefix}\\Library\\lib\\gdalplugins\\gdal_HDF4.dll', 'gdalplugins'),
(f'{conda_env_prefix}\\Library\\lib\\gdalplugins\\gdal_HDF5.dll', 'gdalplugins')]
hiddenimports = ['PySide2.QtSvg', 'PySide2.QtXml']
tmp_ret = collect_all('osgeo')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


temp_a = Analysis(
    ['src\\wiser\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_hooks/set_wiser_env_prod.py'],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)

# BUILD UP NEW hiddenimports by collecting submodules for every top-level package
top_modules = { entry[0].split('.', 1)[0] for entry in temp_a.pure }


IGNORED_TOP_PACKAGES = {
    "PySide2",
}

for pkg in sorted(top_modules):
    # if pkg is in the ignore list, or is a submodule of something in it, skip
    if any(pkg == ign or pkg.startswith(ign + ".")
           for ign in IGNORED_TOP_PACKAGES):
        continue
    hiddenimports.extend(collect_submodules(pkg))

# Remove duplicates while preserving order
_seen = set()
_hidden = []
for m in hiddenimports:
    if m not in _seen:
        _seen.add(m)
        _hidden.append(m)
hiddenimports = _hidden


# SECOND PASS: rebuild Analysis with the full hiddenimports list
a = Analysis(
    ['src\\wiser\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_hooks/set_wiser_env_prod.py'],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)

# Write dependencies resolved by PyInstaller
write_deps_from_analysis(a, out_path="build/pyinstaller_dependencies.txt")

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

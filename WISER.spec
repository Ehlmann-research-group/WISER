# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_all, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

import subprocess

# Create the dependency list file
result = subprocess.run(
    [sys.executable, "src/devtools/write_dependencies.py"],
    capture_output=True,
    text=True,
)

datas = [('src\\wiser\\bandmath\\bandmath.lark', 'wiser\\bandmath')]
binaries = [('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\plugins\\platforms', 'platforms'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\plugins\\iconengines', 'iconengines'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_FITS.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_netCDF.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_HDF4.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_HDF5.dll', 'gdalplugins')]
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
    runtime_hooks=[],
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
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)

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

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('src\\wiser\\bandmath\\bandmath.lark', 'wiser\\bandmath')]
binaries = [('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\plugins\\platforms', 'platforms'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\plugins\\iconengines', 'iconengines'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_FITS.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_netCDF.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_HDF4.dll', 'gdalplugins'), ('C:\\Users\\jgarc\\anaconda3\\envs\\wiser-source\\Library\\lib\\gdalplugins\\gdal_HDF5.dll', 'gdalplugins')]
hiddenimports = ['PySide2.QtSvg', 'PySide2.QtXml']
tmp_ret = collect_all('osgeo')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


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
pyz = PYZ(a.pure)

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

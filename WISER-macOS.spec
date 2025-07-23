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

block_cipher = None

existing_datas = [
                 ('./LICENSE', '.'),
                 ('./src/wiser/bandmath/bandmath.lark', 'wiser/bandmath'),
             ]

existing_hidden_imports = [
                 'PySide2.QtXml',
             ]

existing_binaries = [
        ('/opt/homebrew/Caskroom/miniconda/base/envs/wiser-source/lib/gdalplugins/drivers.ini', 'gdalplugins'),
        ('/opt/homebrew/Caskroom/miniconda/base/envs/wiser-source/lib/gdalplugins/gdal_FITS.dylib', 'gdalplugins'),
        ('/opt/homebrew/Caskroom/miniconda/base/envs/wiser-source/lib/gdalplugins/gdal_HDF4.dylib', 'gdalplugins'),
        ('/opt/homebrew/Caskroom/miniconda/base/envs/wiser-source/lib/gdalplugins/gdal_HDF5.dylib', 'gdalplugins'),
        ('/opt/homebrew/Caskroom/miniconda/base/envs/wiser-source/lib/gdalplugins/gdal_netCDF.dylib', 'gdalplugins'),
    ]  

temp_a = Analysis(['src/wiser/__main__.py'],
             pathex=['/Users/joshuagk/Documents/WISER'],
             binaries=existing_binaries,
             datas=existing_datas,
             hiddenimports=existing_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

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
    existing_hidden_imports.extend(collect_submodules(pkg))

# Remove duplicates while preserving order
_seen = set()
_hidden = []
for m in existing_hidden_imports:
    if m not in _seen:
        _seen.add(m)
        _hidden.append(m)
existing_hidden_imports = _hidden


# SECOND PASS: rebuild Analysis with the full existing_hidden_imports list
a = Analysis(['src/wiser/__main__.py'],
             pathex=['/Users/joshuagk/Documents/WISER'],
             binaries=existing_binaries,
             datas=existing_datas,
             hiddenimports=existing_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='WISER_Bin',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='WISER')
app = BUNDLE(coll,
             name='WISER.app',
             icon='icons/wiser.icns',
             bundle_identifier='edu.caltech.gps.WISER',
             info_plist={
                 'NSPrincipalClass': 'NSApplication',
                 'NSHighResolutionCapable': True,
                 'NSAppleScriptEnabled': False
             }
)

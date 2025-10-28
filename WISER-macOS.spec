# -*- mode: python ; coding: utf-8 -*-
'''
This script assumes you use conda for your environment management.
'''
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(SPECPATH), 'WISER', 'src', 'devtools')))

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

import subprocess

from write_analysis_deps import write_deps_from_analysis

block_cipher = None

existing_datas = [
                 ('./LICENSE', '.'),
                 ('./src/wiser/bandmath/bandmath.lark', 'wiser/bandmath'),
                 ('./src/wiser/data', 'wiser/data'),
                 ('./src/test_utils/test_datasets', 'test_utils/test_datasets'),
                 ('./src/test_utils/test_spectra', 'test_utils/test_spectra'),
                 ('./src/example_plugins', 'example_plugins'),
                 ('./src/tests', 'tests'),
             ]

existing_hidden_imports = [
                 'PySide2.QtXml',
             ]

conda_env_prefix = os.environ.get("CONDA_PREFIX")
existing_binaries = [
        (f'{conda_env_prefix}/lib/gdalplugins/gdal_HDF4.dylib', 'gdalplugins'),
        (f'{conda_env_prefix}/lib/gdalplugins/gdal_HDF5.dylib', 'gdalplugins'),
        (f'{conda_env_prefix}/lib/gdalplugins/gdal_netCDF.dylib', 'gdalplugins'),
        (f'{conda_env_prefix}/lib/gdalplugins/gdal_JP2OpenJPEG.dylib', 'gdalplugins'),
]

temp_a = Analysis(['src/wiser/__main__.py'],
             pathex=['/Users/joshuagk/Documents/WISER'],
             binaries=existing_binaries,
             datas=existing_datas,
             hiddenimports=existing_hidden_imports,
             hookspath=[],
             runtime_hooks=['pyinstaller_hooks/set_wiser_env_prod.py'],
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

# There is an issue with pyinstaller and opencv. We are using the fix here:
# https://github.com/orgs/pyinstaller/discussions/7493#discussioncomment-5315487
cv2_binaries = collect_dynamic_libs(
    "cv2",
    search_patterns=["cv2*.so", "cv2*.dylib", "python-*/cv2*.so", "python-*/cv2*.dylib"]
)

print(f"!@#$% cv2_binaries:\n{cv2_binaries}")

existing_binaries += cv2_binaries

# SECOND PASS: rebuild Analysis with the full existing_hidden_imports list
a = Analysis(['src/wiser/__main__.py'],
             pathex=['/Users/joshuagk/Documents/WISER'],
             binaries=existing_binaries,
             datas=existing_datas,
             hiddenimports=existing_hidden_imports,
             hookspath=[],
             runtime_hooks=['pyinstaller_hooks/set_wiser_env_prod.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

# Write dependencies resolved by PyInstaller
# write_deps_from_analysis(a, out_path="build/pyinstaller_dependencies.txt")

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
          console=False)
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

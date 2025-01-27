# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

block_cipher = None

existing_datas = [
                 ('./LICENSE', '.'),
                 ('./src/wiser/bandmath/bandmath.lark', 'wiser/bandmath'),
             ]

existing_hidden_imports = [
                 'PySide2.QtXml',
             ]

a = Analysis(['src/wiser/__main__.py'],
             pathex=['/Users/joshuagk/Documents/WISER'],
             binaries=[],
             datas=existing_datas,
             hiddenimports=existing_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
print(f"ANALYSISSSSS DONE====")
# Remove all instances of 'libtiff.6.dylib' except the one you specifically want.
# a.binaries = [
#     (name, path, typecode) if ('libtiff.6.dylib' not in name and 'PIL' not in name)
#     else print(f"(name, path, typecode): {name}, {path}, {typecode}")
#     for (name, path, typecode) in a.binaries
# ]
# a.binaries = [
#     (name, path, typecode) 
#     for (name, path, typecode) in a.binaries
#     if 'libtiff.6.dylib' not in name and 'PIL' not in name
# ]

# a.binaries = a.binaries-TOC([
#     ('libtiff.6.dylib', 'PIL/.dylib', None)
# ])

# Now add back the specific libtiff.6.dylib you want to include.
# Make sure you provide the correct absolute path to this file.
# a.binaries.append(
#     ('libtiff.6.dylib', '/opt/homebrew/Caskroom/miniconda/base/envs/wiser-source/lib/libtiff.6.dylib', 'BINARY')
# )
# a.binaries.append(
#     ('Correct', '/opt/homebrew/Caskroom/miniconda/base/envs/wiser-source/lib/libtiff.6.dylib', 'BINARY')
# )

# print(f"a.scripts: {a.scripts}")
# print(f"a.pure: {a.pure}")
# print(f"a.pathex: {a.pathex}")
# print(f"a.binaries: {a.binaries}")
# print(f"a.datas: {a.datas}")

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

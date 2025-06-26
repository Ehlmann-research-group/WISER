# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

existing_datas = [
                 ('./LICENSE', '.'),
                 ('./src/wiser/bandmath/bandmath.lark', 'wiser/bandmath'),
             ]

existing_hidden_imports = [
                 'PySide2.QtXml',
             ]

existing_binaries = [
        ('/opt/homebrew/Caskroom/miniconda/base/envs/intel-wiser/lib/gdalplugins/gdal_HDF4.dylib', 'gdalplugins'),
        ('/opt/homebrew/Caskroom/miniconda/base/envs/intel-wiser/lib/gdalplugins/gdal_HDF5.dylib', 'gdalplugins'),
        ('/opt/homebrew/Caskroom/miniconda/base/envs/intel-wiser/lib/gdalplugins/gdal_netCDF.dylib', 'gdalplugins'),
    ]  

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
          target_arch='x86_64',
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

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['src/wiser/__main__.py'],
             pathex=['/Users/donnie/Projects/WISER'],
             binaries=[],
             datas=[
                 ('./LICENSE', '.'),
                 ('./src/wiser/bandmath/bandmath.lark', 'wiser/bandmath'),
             ],
             hiddenimports=[
                 'PySide2.QtXml',
             ],
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


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

# # -*- mode: python ; coding: utf-8 -*-


# from pathlib import Path
# import json, subprocess, shlex, os, sys
# import importlib.metadata as imeta
# from packaging.version import Version, InvalidVersion
# # 2) Conda packages explicitly requested by the user  ───────────────────
# conda_pkgs = {}

# env_prefix = sys.exec_prefix
# print(f"env_prefix: {env_prefix}")
# conda_exe = os.path.join(env_prefix, "..\..\Scripts\conda.exe")
# print(f"conda_exe: {conda_exe}")
# cmd = f'conda run -n wiser-source-2 conda env export --name wiser-source-2 --from-history --json'
# try:
#     # '--from-history' gives just user-requested specs
#     out = subprocess.check_output(
#         [conda_exe, "env", "export", "--name", "wiser-source-2", "--from-history", "--json"],
#         text=True, stderr=subprocess.DEVNULL
#     )
# #     out = subprocess.check_output(
# #     cmd,
# #     shell=True,            # pass the string into your shell
# #     text=True,             # get str instead of bytes
# #     stderr=subprocess.DEVNULL
# # )
#     hist = json.loads(out)
#     for spec in hist.get("dependencies", []):
#         if isinstance(spec, str) and "=" in spec:
#             name, ver, *_ = spec.split("=", 2)
#             conda_pkgs[name] = ver
#         else:
#             name = spec.split("=")[0]
#             print(f"name: {name}")
#             conda_pkgs[name] = "N/A"
# except Exception:
#     print(f"!@! CONDA NOT FOUND CAUSE EXCEPTION")
#     pass                                  # conda not on PATH, ignore
# # print(f"conda_pkgs: {conda_pkgs}")
# pip_pkgs = {}
# try:
#     out = subprocess.check_output(
#         [sys.executable, "-m", "pip", "list", "--not-required", "--format=json"],
#         text=True, stderr=subprocess.DEVNULL
#     )
#     for row in json.loads(out):
#         pip_pkgs[row["name"]] = row["version"]
# except Exception:
#     print(f"!#! PIP NOT FOUND CAUSE EXCEPTION")
#     pass                                  # pip not available, ignore
# print(f"!#! pip_pkgs: {pip_pkgs}")

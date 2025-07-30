'''
Script to bootstrap WISER. This runs in pyinstaller .

Dev environment:

'''

from pathlib import Path
import os, subprocess, sys
import textwrap
import os, stat, shutil, time
from importlib._bootstrap_external import _NamespacePath


APP_HOME = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(__file__).parent

src_path = os.path.join(APP_HOME, "src")  # Path to our source code
wiser_path = os.path.join(src_path, "wiser")  # Path to wiser
zip_archive = os.path.join(APP_HOME, "src_bundle")  # Path to folder with src_bundle.pyz
requirements_path = os.path.join(str(APP_HOME), "requirements.txt")
env_dir = APP_HOME / "runtime_env"
python_path = os.path.join(
                str(env_dir),
                "Scripts" if os.name == "nt" else "bin",
                "python.exe" if os.name == "nt" else "python"
            )
uv_path = APP_HOME / "uv"
uv_exec_path = uv_path / ("uv.exe" if os.name == "nt" else "uv")
print(f"APP_HOME: {APP_HOME}")
print(f"wiser_path: {wiser_path}")
print(f"src_path: {src_path}")
print(f"type(src_path): {type(src_path)}")
print(f"zip_archive: {zip_archive}")
print(f"type(APP_HOME): {type(APP_HOME)}")
print(f"requirements_path: {requirements_path}")
print(f"uv env_dir: {env_dir}")
print(f"python_path: {python_path}")
print(f"uv uv_path: {uv_path}")
print(f"uv_exec_path: {uv_exec_path}")


def ensure_env():
    env_dir = APP_HOME / "runtime_env"
    
    # if env_dir.exists():
    #     clear_env(env_dir)

    subprocess.run([uv_exec_path, "venv", str(env_dir), "--python", "3"], check=True)

    try:
        res = subprocess.run(
            [
                uv_exec_path, "pip", "install",
                "--python",
                python_path,
                "-r",
                requirements_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    # Always decode and print full log
    print(f"stdout:\n{res.stdout}\n----------------------")
    print(f"stderr:\n{res.stderr}\n----------------------")
    if res.returncode != 0:
        sys.exit(res.returncode)

    return env_dir


def relaunch_in_venv(env_dir: Path):
    # Sanitize PyInstaller-specific library tweaks for the child:
    env = {**os.environ, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH_ORIG", ""),
           "PYTHONPATH": zip_archive, "PYTHONHOME": ""}

    proc = subprocess.Popen(
        [str(python_path), "src_bundle/src_bundle.pyz", *sys.argv[1:]],
        env=env,
        cwd=str(APP_HOME), 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Echo in real time:
    for line in proc.stdout:
        print(line, end="", flush=True)
    sys.exit(proc.wait())

def main():
    if os.environ.get("_WISER_ALREADY_REEXEC") != "1":
        env = ensure_env()
        print(f"env dir: {env}")
        os.environ["_WISER_ALREADY_REEXEC"] = "1"
        print(f"relaunch start")
        relaunch_in_venv(env)
    else:
        from src.wiser.app import main as start_app  # heavy imports live here
        print(f"else statement start")
        start_app()

if __name__ == "__main__":
    main()

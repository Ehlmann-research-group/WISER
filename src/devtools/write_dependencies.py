import context
from wiser.version import VERSION

import json, subprocess, os, sys
import shutil
import platform as _platform

def export_current_conda_env(from_history: bool = False) -> dict:
    """
    Export the Conda environment that this Python process is running in.
    Returns the parsed JSON export.
    """
    # 1. Figure out where Conda lives and which env we’re in
    prefix = os.environ.get("CONDA_PREFIX", sys.exec_prefix)
    conda_cmd = shutil.which("conda") or os.path.join(prefix, "Scripts", "conda.exe")

    cmd = [
        conda_cmd,
        "env", "export",
        "--prefix", prefix,
        "--json",
    ]
    if from_history:
        cmd.insert(-1, "--from-history")
    
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

if __name__ == "__main__":
    version_name  = f"{VERSION}"
    platform_name = f"{_platform.system().lower()}" # If needed, add the cpu architecture here
    filename = f"wiser_dependencies_{version_name}_{platform_name}.json"

    # base directory of this script
    BASE = os.path.dirname(__file__)

    # ensure generated/ exists
    out_dir = os.path.join(BASE, "generated")
    os.makedirs(out_dir, exist_ok=True)

    filepath = os.path.join(
        out_dir,
        filename
    )

    print(f"filepath: {filepath}")

    full_env = export_current_conda_env()
    print(json.dumps(full_env, indent=2))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(full_env, f, indent=2)

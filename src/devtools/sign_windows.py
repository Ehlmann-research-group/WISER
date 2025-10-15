import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

RUN_URL_RE = re.compile(
    r"""
    ^https://github\.com/
    (?P<owner>[^/]+)/(?P<repo>[^/]+)         # org/repo
    /actions/runs/(?P<run_id>\d+)            # run id
    (?:/artifacts/\d+)?/?$                   # optional artifact suffix
    """,
    re.VERBOSE,
)


def die(msg: str, code: int = 1):  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download GH Actions artifact and run NSIS.")
    p.add_argument("--link", required=True, help="GitHub Actions run URL")
    p.add_argument("--root", default=".", help="Project root (default: current dir)")
    p.add_argument("--dist-name", default="dist", help='Dist folder name under root (default: "dist")')
    p.add_argument("--artifact-name", default="wiser-Windows", help="Artifact name in the run")
    p.add_argument("--nsis", default=os.environ.get("NSIS", "makensis"), help="Path to makensis.exe")
    p.add_argument("--nsi-script", default=r"install-win\win-install.nsi", help="NSIS script path")
    p.add_argument(
        "--app-version", dest="app_version", default=os.environ.get("APP_VERSION"), help="WISER version"
    )
    p.add_argument(
        "--sha1",
        dest="sha1_thumbprint",
        default=os.environ.get("SHA1_THUMBPRINT"),
        help="Windows signing SHA1 thumbprint",
    )
    return p.parse_args()


def ensure_gh_available():
    from shutil import which

    if which("gh") is None:
        die("GitHub CLI (gh) not found in PATH.")


def parse_run_url(url: str):
    m = RUN_URL_RE.match(url.strip())
    if not m:
        die("Link must look like https://github.com/<org>/<repo>/actions/runs/<RUN_ID>[/artifacts/<ID>]")
    return m.group("owner"), m.group("repo"), m.group("run_id")


def prepare_dist(root: Path, dist_name: str) -> Path:
    dist_dir = root / dist_name
    if dist_dir.exists():
        # Make sure it's empty
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    return dist_dir


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print("❌ Command failed:", " ".join(cmd))
        print("Exit code:", e.returncode)
        if e.stdout:
            print("---- STDOUT ----")
            print(e.stdout)
        if e.stderr:
            print("---- STDERR ----")
            print(e.stderr)
        raise


def main():
    args = parse_args()

    # Resolve paths
    root = Path(args.root).resolve()
    dist_dir = prepare_dist(root, args.dist_name)

    # Checks
    ensure_gh_available()
    owner, repo, run_id = parse_run_url(args.link)

    print(f"Owner/Repo: {owner}/{repo}")
    print(f"Run ID: {run_id}")
    print(f"Artifact name: {args.artifact_name}")
    print(f"Dist dir: {dist_dir}")

    # gh run download: prefer explicit repo for robustness
    # Equivalent gh command:
    #   gh run download <RUN_ID> -R <owner>/<repo> -n <artifact-name> --dir <dist_dir>
    run(
        [
            "gh",
            "run",
            "download",
            run_id,
            "-R",
            f"{owner}/{repo}",
            "-n",
            args.artifact_name,
            "--dir",
            str(dist_dir),
        ]
    )

    # Run NSIS installer (optional flags passed if provided)
    nsis = args.nsis
    nsi_script = args.nsi_script

    nsis_cmd = [nsis, "/NOCD"]
    if args.app_version:
        nsis_cmd.append(f"/DWISER_VERSION={args.app_version}")
    if args.sha1_thumbprint:
        nsis_cmd.append(f"/DSHA1_THUMBPRINT={args.sha1_thumbprint}")
    nsis_cmd.append(str(nsi_script))

    run(nsis_cmd)

    print("Done. Artifact downloaded and installer executed.")


if __name__ == "__main__":
    main()

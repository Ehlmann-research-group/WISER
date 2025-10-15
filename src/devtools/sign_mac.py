import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

RUN_URL_RE = re.compile(
    r"""
    ^https://github\.com/
    (?P<owner>[^/]+)/(?P<repo>[^/]+)
    /actions/runs/(?P<run_id>\d+)
    (?:/artifacts/\d+)?/?$
    """,
    re.VERBOSE,
)


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print("❌ Command failed:", " ".join(str(c) for c in cmd), file=sys.stderr)
        print("Exit code:", e.returncode, file=sys.stderr)
        raise


def which(name: str) -> Optional[str]:
    from shutil import which as _which

    return _which(name)


def ensure_tool(name: str, hint: str = ""):
    if which(name) is None:
        die(f"Required tool '{name}' not found in PATH. {hint}")


def parse_run_url(url: str):
    m = RUN_URL_RE.match(url.strip())
    if not m:
        die("Link must look like https://github.com/<org>/<repo>/actions/runs/<RUN_ID>[/artifacts/<ID>]")
    return m.group("owner"), m.group("repo"), m.group("run_id")


def prepare_dist(root: Path, dist_name: str) -> Path:
    dist_dir = root / dist_name
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    return dist_dir


def find_app_bundle(dist_dir: Path, app_name: str) -> Path:
    # Prefer the expected name; otherwise fall back to first *.app under dist
    expected = dist_dir / f"{app_name}.app"
    if expected.exists():
        return expected
    candidates = list(dist_dir.glob("*.app"))
    if not candidates:
        die(f"No .app bundle found in {dist_dir}. Expected '{app_name}.app'.")
    if len(candidates) > 1:
        print("WARNING: multiple .app bundles found; using:", candidates[0], file=sys.stderr)
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description="Download GH Actions artifact and package WISER for macOS.")
    p.add_argument("--link", required=True, help="GitHub Actions run URL")
    p.add_argument("--root", default=".", help="Project root (default: current dir)")
    p.add_argument("--dist-name", default="dist", help='Dist folder name under root (default: "dist")')
    p.add_argument("--artifact-name", default="wiser-mac-arm", help="Artifact name in the run to download")
    p.add_argument(
        "--app-name",
        default=os.environ.get("APP_NAME", "WISER"),
        help='App bundle base name (default: "WISER")',
    )
    p.add_argument(
        "--app-version", default=os.environ.get("APP_VERSION"), help="Version string for DMG filename"
    )
    p.add_argument("--sign-script", default="../../install-mac/sign_wiser.sh", help="Codesign script to run (bash)")
    p.add_argument("--notarize", action="store_true", help="Submit DMG to Apple notarization with notarytool")
    p.add_argument("--apple-id", default=os.environ.get("AD_USERNAME"), help="Apple ID (or set AD_USERNAME)")
    p.add_argument(
        "--team-id", default=os.environ.get("AD_TEAM_ID"), help="Apple Team ID (or set AD_TEAM_ID)"
    )
    p.add_argument(
        "--app-password",
        default=os.environ.get("AD_PASSWORD"),
        help="App-specific password (or set AD_PASSWORD)",
    )
    p.add_argument("--notary-wait", action="store_true", help="Pass --wait to notarytool submit")
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.root).resolve()
    dist_dir = prepare_dist(root, args.dist_name)

    # Tools
    ensure_tool("gh", "Install: https://cli.github.com/")
    ensure_tool("hdiutil")
    ensure_tool("xcrun")  # for notarytool

    owner, repo, run_id = parse_run_url(args.link)
    print(f"Repo: {owner}/{repo}  Run ID: {run_id}")
    print(f"Artifact: {args.artifact_name}")
    print(f"Dist dir: {dist_dir}")

    # Download artifact into dist/
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

    # Locate app bundle
    app_bundle = find_app_bundle(dist_dir, args.app_name)
    print(f"App bundle: {app_bundle}")

    # Codesign via your script
    sign_script = (root / args.sign_script).resolve()
    if not sign_script.exists():
        die(f"Sign script not found: {sign_script}")
    # Ensure executable bit (safe no-op if already set)
    try:
        mode = sign_script.stat().st_mode
        sign_script.chmod(mode | 0o111)
    except Exception:
        pass

    # Many sign scripts expect to be run from repo root
    run(["bash", str(sign_script), str(app_bundle)], cwd=str(root))

    # Create DMG (tmp then convert to compressed UDZO)
    tmp_dmg = dist_dir / "tmp.dmg"
    final_name = f"{args.app_name}-{args.app_version}.dmg" if args.app_version else f"{args.app_name}.dmg"
    final_dmg = dist_dir / final_name

    # hdiutil create
    run(
        [
            "hdiutil",
            "create",
            str(tmp_dmg),
            "-ov",
            "-volname",
            args.app_name,
            "-fs",
            "HFS+",
            "-srcfolder",
            str(app_bundle),
        ]
    )

    # hdiutil convert → UDZO
    run(["hdiutil", "convert", str(tmp_dmg), "-format", "UDZO", "-o", str(final_dmg)])

    # remove tmp dmg
    try:
        tmp_dmg.unlink(missing_ok=True)  # py3.8+: ok to keep try/except
    except TypeError:
        if tmp_dmg.exists():
            tmp_dmg.unlink()

    print(f"DMG created: {final_dmg}")

    # Notarization (optional)
    if args.notarize:
        if not (args.apple_id and args.team_id and args.app_password):
            die(
                "Notarization requested but --apple-id/--team-id/"
                "--app-password not provided (or env AD_USERNAME/"
                "AD_TEAM_ID/AD_PASSWORD)."
            )
        notary_cmd = [
            "xcrun",
            "notarytool",
            "submit",
            str(final_dmg),
            "--apple-id",
            args.apple_id,
            "--team-id",
            args.team_id,
            "--password",
            args.app_password,
        ]
        if args.notary_wait:
            notary_cmd.append("--wait")
        run(notary_cmd)

    print("Done. Artifact downloaded, app signed, DMG built" + (" and notarized." if args.notarize else "."))


if __name__ == "__main__":
    main()

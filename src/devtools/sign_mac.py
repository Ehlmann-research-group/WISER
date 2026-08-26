"""Code-sign, notarize, staple, and package the WISER macOS application.

The app bundle is either downloaded from a GitHub Actions build run (``--link``) or taken
from disk (``--app-path``, used by CI, where the build already produced it). Everything
after that is the same code path for both callers.

The sequence follows Apple's recommended order for Developer ID distribution:

    sign the .app -> notarize it -> staple the ticket to the .app -> build the DMG from the
    stapled app -> sign the DMG -> notarize it -> staple the ticket to the DMG -> verify

Stapling the .app before the DMG is built is what lets a user who drags WISER into
/Applications launch it with no network round trip to Apple.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

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


def run(cmd, redact: Sequence[str] = (), capture: bool = False, **kwargs):
    shown = []
    for part in cmd:
        part = str(part)
        for secret in redact:
            if secret:
                part = part.replace(secret, "***")
        shown.append(part)
    print("+", " ".join(shown))
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            **kwargs,
        )
    except subprocess.CalledProcessError as e:
        print("Command failed:", " ".join(shown), file=sys.stderr)
        print("Exit code:", e.returncode, file=sys.stderr)
        raise
    return proc.stdout if capture else None


def ensure_tool(name: str, hint: str = ""):
    if shutil.which(name) is None:
        die(f"Required tool '{name}' not found in PATH. {hint}")


def parse_run_url(url: str):
    m = RUN_URL_RE.match(url.strip())
    if not m:
        die("Link must look like https://github.com/<org>/<repo>/actions/runs/<RUN_ID>[/artifacts/<ID>]")
    return m.group("owner"), m.group("repo"), m.group("run_id")


def find_app_bundle(dist_dir: Path, app_name: str) -> Path:
    expected = dist_dir / f"{app_name}.app"
    if expected.exists():
        return expected
    candidates = list(dist_dir.glob("*.app"))
    if not candidates:
        die(f"No .app bundle found in {dist_dir}. Expected '{app_name}.app'.")
    if len(candidates) > 1:
        die(f"Multiple .app bundles found in {dist_dir}: {[str(c) for c in candidates]}")
    return candidates[0]


def acquire_from_run(link: str, root: Path, dist_name: str, artifact_name: str, app_name: str) -> Path:
    """Download the build artifact for a run and unpack the app bundle from it."""
    ensure_tool("gh", "Install: https://cli.github.com/")
    ensure_tool("tar")
    ensure_tool("shasum")

    dist_dir = root / dist_name
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    owner, repo, run_id = parse_run_url(link)
    print(f"Repo: {owner}/{repo}  Run ID: {run_id}  Artifact: {artifact_name}")

    run(
        [
            "gh",
            "run",
            "download",
            run_id,
            "-R",
            f"{owner}/{repo}",
            "-n",
            artifact_name,
            "--dir",
            str(dist_dir),
        ]
    )

    # The checksum file records the path as dist/WISER.app.tgz, so verify from the root.
    run(["shasum", "-a", "256", "-c", str(dist_dir / "WISER.app.tgz.sha256")], cwd=str(root))
    # tar rather than unzip: it preserves the symlinks and modes inside the bundle.
    run(["tar", "-C", str(dist_dir), "-xzf", str(dist_dir / "WISER.app.tgz")])

    return find_app_bundle(dist_dir, app_name)


def notary_flags(args) -> tuple[list[str], list[str]]:
    """Return (notarytool credential flags, secret strings to redact from logs)."""
    if args.notary_key:
        key = Path(args.notary_key).resolve()
        if not key.exists():
            die(f"App Store Connect API key not found: {key}")
        if not args.notary_key_id:
            die("--notary-key requires --notary-key-id (or NOTARY_KEY_ID).")
        flags = ["--key", str(key), "--key-id", args.notary_key_id]
        # An individual key inherits the caller's role and carries no issuer.
        if args.notary_issuer:
            flags += ["--issuer", args.notary_issuer]
        return flags, [args.notary_key_id, args.notary_issuer or ""]

    if args.apple_id and args.team_id and args.app_password:
        return (
            ["--apple-id", args.apple_id, "--team-id", args.team_id, "--password", args.app_password],
            [args.app_password],
        )

    die(
        "Notarization requires either an App Store Connect API key "
        "(--notary-key/--notary-key-id, env NOTARY_KEY_PATH/NOTARY_KEY_ID) or an Apple ID "
        "(--apple-id/--team-id/--app-password, env AD_USERNAME/AD_TEAM_ID/AD_PASSWORD)."
    )


def notarize(target: Path, flags: list[str], secrets: list[str]):
    """Submit to Apple and wait. Raises unless the submission is Accepted."""
    out = run(
        ["xcrun", "notarytool", "submit", str(target), *flags, "--wait", "--output-format", "json"],
        redact=secrets,
        capture=True,
    )
    try:
        result = json.loads(out)
    except json.JSONDecodeError:
        die(f"notarytool returned output that is not JSON:\n{out}")

    status = result.get("status")
    submission_id = result.get("id")
    print(f"Notarization {submission_id}: {status}")
    if status != "Accepted":
        if submission_id:
            # Printed directly so the credential flags never reach the log.
            subprocess.run(["xcrun", "notarytool", "log", submission_id, *flags], text=True)
        die(f"Notarization of {target.name} was not accepted (status: {status}).")


def staple(target: Path):
    run(["xcrun", "stapler", "staple", str(target)])
    run(["xcrun", "stapler", "validate", str(target)])


def build_dmg(app: Path, dmg: Path, volname: str):
    tmp_dmg = dmg.with_name("tmp.dmg")
    run(
        ["hdiutil", "create", str(tmp_dmg), "-ov", "-volname", volname, "-fs", "HFS+", "-srcfolder", str(app)]
    )
    if dmg.exists():
        dmg.unlink()
    run(["hdiutil", "convert", str(tmp_dmg), "-format", "UDZO", "-o", str(dmg)])
    tmp_dmg.unlink(missing_ok=True)


def assess_gatekeeper(app: Path, dmg: Path, require: bool):
    """Run the end-user Gatekeeper assessment, which needs assessments to be enabled.

    On a machine with `spctl --status` disabled the assessment reports "accepted" for
    anything, so a pass there means nothing. CI passes require=True to make that a hard
    failure rather than a false green.
    """
    status = subprocess.run(["spctl", "--status"], text=True, capture_output=True).stdout
    if "assessments enabled" not in status:
        msg = (
            "Gatekeeper assessments are disabled on this machine, so the end-user "
            "Gatekeeper check cannot be performed (spctl would report 'accepted' for "
            "anything). Re-enable with: sudo spctl --master-enable"
        )
        if require:
            die(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
        print("The signature and stapled tickets were still verified directly.", file=sys.stderr)
        return

    run(["spctl", "--assess", "--type", "exec", "--verbose=4", str(app)])
    run(
        [
            "spctl",
            "--assess",
            "--type",
            "open",
            "--context",
            "context:primary-signature",
            "--verbose=4",
            str(dmg),
        ]
    )


def parse_args():
    p = argparse.ArgumentParser(description="Code-sign, notarize, staple, and package WISER for macOS.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--link", help="GitHub Actions run URL to download the build artifact from")
    src.add_argument("--app-path", help="Path to an already-built .app bundle to sign")

    p.add_argument("--root", default=".", help="Project root (default: current dir)")
    p.add_argument("--dist-name", default="dist", help='Dist folder name under root (default: "dist")')
    p.add_argument("--artifact-name", default="wiser-macOS-X64", help="Artifact name in the run to download")
    p.add_argument(
        "--app-name",
        default=os.environ.get("APP_NAME", "WISER"),
        help='App bundle base name (default: "WISER")',
    )
    p.add_argument(
        "--app-version", default=os.environ.get("APP_VERSION"), help="Version string for the DMG filename"
    )
    p.add_argument(
        "--arch",
        default=None,
        choices=["x64", "arm64"],
        help="Target arch for the DMG filename (default: inferred from --artifact-name)",
    )
    p.add_argument(
        "--identity",
        default=os.environ.get("AD_CODESIGN_KEY_NAME"),
        help="Code-signing identity (or set AD_CODESIGN_KEY_NAME)",
    )
    p.add_argument(
        "--sign-script",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "install-mac", "sign_wiser.sh"),
        help="Codesign script to run (bash)",
    )

    p.add_argument("--notarize", action="store_true", help="Notarize and staple with notarytool")
    p.add_argument(
        "--notary-key",
        default=os.environ.get("NOTARY_KEY_PATH"),
        help="App Store Connect API key .p8 (or set NOTARY_KEY_PATH)",
    )
    p.add_argument(
        "--notary-key-id",
        default=os.environ.get("NOTARY_KEY_ID"),
        help="App Store Connect Key ID (or set NOTARY_KEY_ID)",
    )
    p.add_argument(
        "--notary-issuer",
        default=os.environ.get("NOTARY_ISSUER_ID"),
        help="App Store Connect Issuer ID; omit for an individual key (or set NOTARY_ISSUER_ID)",
    )
    p.add_argument("--apple-id", default=os.environ.get("AD_USERNAME"), help="Apple ID (or set AD_USERNAME)")
    p.add_argument(
        "--team-id", default=os.environ.get("AD_TEAM_ID"), help="Apple Team ID (or set AD_TEAM_ID)"
    )
    p.add_argument(
        "--app-password",
        default=os.environ.get("AD_PASSWORD"),
        help="App-specific password (or set AD_PASSWORD)",
    )

    p.add_argument(
        "--require-gatekeeper",
        action="store_true",
        help="Fail if Gatekeeper assessments are disabled instead of warning",
    )
    p.add_argument(
        "--release-tag", default=None, help="If set, upload the finished DMG to this GitHub release tag"
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/repo for --release-tag (default: inferred from --link)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.release_tag and not args.app_version:
        die("--release-tag requires --app-version so the uploaded DMG carries the version.")
    if not args.identity:
        die("No code-signing identity. Pass --identity or set AD_CODESIGN_KEY_NAME.")

    ensure_tool("hdiutil")
    ensure_tool("xcrun")
    ensure_tool("codesign")
    if args.notarize:
        ensure_tool("ditto")
        ensure_tool("spctl")

    root = Path(args.root).resolve()

    if args.link:
        app = acquire_from_run(args.link, root, args.dist_name, args.artifact_name, args.app_name)
        owner, repo, _ = parse_run_url(args.link)
        repo_slug = args.repo or f"{owner}/{repo}"
    else:
        app = Path(args.app_path).resolve()
        if not app.is_dir():
            die(f"No app bundle at {app}")
        repo_slug = args.repo
    print(f"App bundle: {app}")

    flags, secrets = notary_flags(args) if args.notarize else ([], [])

    sign_script = Path(args.sign_script).resolve()
    if not sign_script.exists():
        die(f"Sign script not found: {sign_script}")
    env = {**os.environ, "AD_CODESIGN_KEY_NAME": args.identity}
    run(["bash", str(sign_script), str(app)], cwd=str(root), env=env)

    if args.notarize:
        # notarytool cannot take a bare bundle; ditto preserves the symlinks codesign needs.
        app_zip = app.with_suffix(".app.zip")
        app_zip.unlink(missing_ok=True)
        run(["ditto", "-c", "-k", "--keepParent", str(app), str(app_zip)])
        notarize(app_zip, flags, secrets)
        app_zip.unlink(missing_ok=True)
        staple(app)

    arch = args.arch or ("arm64" if "ARM64" in args.artifact_name.upper() else "x64")
    stem = f"{args.app_name}-{args.app_version}" if args.app_version else args.app_name
    dmg = app.parent / f"{stem}-macos-{arch}.dmg"
    build_dmg(app, dmg, args.app_name)

    run(["codesign", "--force", "--timestamp", "--sign", args.identity, str(dmg)])

    if args.notarize:
        notarize(dmg, flags, secrets)
        staple(dmg)

    print("-- Verifying the finished artifacts...")
    run(["codesign", "--verify", "--strict", "--deep", "--verbose=2", str(app)])
    run(["codesign", "--verify", "--verbose=2", str(dmg)])
    if args.notarize:
        run(["xcrun", "stapler", "validate", str(app)])
        run(["xcrun", "stapler", "validate", str(dmg)])
        assess_gatekeeper(app, dmg, args.require_gatekeeper)

    print(f"DMG ready: {dmg}")

    if args.release_tag:
        if not repo_slug:
            die("--release-tag needs --repo (or GITHUB_REPOSITORY) when signing from --app-path.")
        ensure_tool("gh", "Install: https://cli.github.com/")
        run(["gh", "release", "upload", args.release_tag, str(dmg), "-R", repo_slug, "--clobber"])
        print(f"Uploaded {dmg.name} to release {args.release_tag}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mirror GitHub release assets into the LASP Nexus raw repository.

WISER's installers are published as GitHub release assets. This copies each one into a
Nexus raw repository so the binaries survive loss of the GitHub repository, together with
a manifest and a `sha256sum -c`-compatible checksum file that make the copy restorable on
its own.

Layout written per release, relative to the Nexus repository root:

    releases/<tag>/<asset name>
    releases/<tag>/source/<repo>-<tag>-source.tar.gz
    releases/<tag>/source/<repo>-<tag>-source.zip
    releases/<tag>/manifest.json
    releases/<tag>/SHA256SUMS
    releases/<tag>/release-notes.md

Only the `/repository/` path of the Nexus host is reachable from outside the LASP network,
so this speaks plain PUT/GET/HEAD to the raw repository and never uses the Nexus REST API.

Credentials come from the environment: NEXUS_USERNAME and NEXUS_PASSWORD (a Nexus user
token secret works in place of a password), plus an optional GITHUB_TOKEN.

Usage:
    python src/devtools/backup_releases_to_nexus.py --tag v3.0b0
    python src/devtools/backup_releases_to_nexus.py --all --dry-run
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "Ehlmann-research-group/WISER"

# The public reverse proxy in front of the LASP Nexus. The internal name
# (artifacts.pdmz.lasp.colorado.edu) is not in public DNS and resolves to RFC1918 space, so
# it is unreachable from GitHub-hosted runners; only this path is published.
DEFAULT_NEXUS_URL = "https://lasp.colorado.edu/repository/wiser-raw"

CHUNK = 1 << 20
RETRIES = 3
TIMEOUT = 600
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# Nexus reports a raw asset's SHA-1 in its ETag as {SHA1{<hex>}}.
ETAG_SHA1_RE = re.compile(r"\{SHA1\{([0-9a-f]{40})\}\}")


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def log(msg: str = ""):
    print(msg, flush=True)


class StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drops Authorization when a redirect crosses to a different host.

    GitHub answers an asset or source-archive download with a redirect to signed
    object-storage URLs, which reject a request that also carries an Authorization header.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            for store in (new.headers, new.unredirected_hdrs):
                for key in [k for k in store if k.lower() == "authorization"]:
                    del store[key]
        return new


_OPENER = urllib.request.build_opener(StripAuthOnRedirect)


def _attempt(req: urllib.request.Request, timeout: int):
    return _OPENER.open(req, timeout=timeout)


def _retry_sleep(attempt: int):
    time.sleep(min(2**attempt, 30))


def request_json(url: str, headers: Dict[str, str]) -> object:
    for attempt in range(1, RETRIES + 1):
        try:
            with _attempt(urllib.request.Request(url, headers=headers), TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == RETRIES:
                raise
        except OSError:
            if attempt == RETRIES:
                raise
        _retry_sleep(attempt)
    raise RuntimeError("unreachable")


def head(url: str, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Returns the response headers, or None when the object does not exist."""
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="HEAD")
            with _attempt(req, TIMEOUT) as resp:
                return dict(resp.headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in RETRY_STATUS or attempt == RETRIES:
                raise
        except OSError:
            if attempt == RETRIES:
                raise
        _retry_sleep(attempt)
    raise RuntimeError("unreachable")


def download(url: str, headers: Dict[str, str], dest: Path) -> Tuple[str, str, int]:
    """Streams *url* to *dest*, returning its (sha256, sha1, size).

    Hashing while streaming keeps peak memory flat and avoids a second pass over an asset
    that can approach half a gigabyte.
    """
    for attempt in range(1, RETRIES + 1):
        sha256, sha1, size = hashlib.sha256(), hashlib.sha1(), 0
        try:
            with _attempt(urllib.request.Request(url, headers=headers), TIMEOUT) as resp:
                with dest.open("wb") as out:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        sha256.update(block)
                        sha1.update(block)
                        size += len(block)
            return sha256.hexdigest(), sha1.hexdigest(), size
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == RETRIES:
                raise
        except OSError:
            if attempt == RETRIES:
                raise
        _retry_sleep(attempt)
    raise RuntimeError("unreachable")


def put(url: str, headers: Dict[str, str], source: Path, content_type: str) -> int:
    """Uploads *source* to *url*, reopening the file per attempt so a retry restarts clean."""
    for attempt in range(1, RETRIES + 1):
        size = source.stat().st_size
        hdrs = dict(headers)
        hdrs["Content-Type"] = content_type
        hdrs["Content-Length"] = str(size)
        try:
            with source.open("rb") as body:
                req = urllib.request.Request(url, data=body, headers=hdrs, method="PUT")
                with _attempt(req, TIMEOUT) as resp:
                    return resp.status
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                die("Nexus rejected the credentials (401). Check NEXUS_USERNAME / NEXUS_PASSWORD.")
            if exc.code == 403:
                die(f"Nexus refused the upload (403). The account needs add/edit on the raw repo: {url}")
            if exc.code == 413:
                die(f"Nexus or the reverse proxy rejected the body as too large (413): {size} bytes.")
            if exc.code == 400:
                die(
                    f"Nexus rejected the upload (400) for {url}. A repository set to 'Disable "
                    "redeploy' cannot replace an object that already exists."
                )
            if exc.code not in RETRY_STATUS or attempt == RETRIES:
                raise
        except OSError:
            if attempt == RETRIES:
                raise
        _retry_sleep(attempt)
    raise RuntimeError("unreachable")


def put_bytes(url: str, headers: Dict[str, str], data: bytes, content_type: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "payload"
        path.write_bytes(data)
        return put(url, headers, path, content_type)


def nexus_url(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/")] + [quote(p) for p in parts])


def github_headers(token: Optional[str], accept: str = "application/vnd.github+json") -> Dict[str, str]:
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def basic_auth(user: str, password: str) -> Dict[str, str]:
    encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def list_releases(repo: str, token: Optional[str]) -> List[dict]:
    releases, page = [], 1
    while True:
        url = f"{GITHUB_API}/repos/{repo}/releases?per_page=100&page={page}"
        batch = request_json(url, github_headers(token))
        if not isinstance(batch, list) or not batch:
            break
        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return releases


def resolve_commit(repo: str, tag: str, token: Optional[str]) -> Optional[str]:
    """The commit a tag points at, or None when the tag has no ref yet (draft release)."""
    try:
        commit = request_json(f"{GITHUB_API}/repos/{repo}/commits/{tag}", github_headers(token))
        return commit.get("sha") if isinstance(commit, dict) else None
    except (urllib.error.HTTPError, OSError):
        return None


def digest_sha256(asset: dict) -> Optional[str]:
    """The SHA-256 GitHub records for an asset, when it has one.

    Assets uploaded before GitHub began recording digests report nothing here, in which
    case the hash is computed from the bytes as they are streamed.
    """
    value = asset.get("digest") or ""
    return value.split("sha256:", 1)[1] if value.startswith("sha256:") else None


def verify_upload(url: str, headers: Dict[str, str], size: int, sha1: str, name: str) -> List[str]:
    """Confirms what Nexus stored matches what was sent. Returns a list of problems."""
    stored = head(url, headers)
    if stored is None:
        return [f"{name}: not present in Nexus after upload"]
    problems = []
    remote_size = stored.get("Content-Length")
    if remote_size is not None and int(remote_size) != size:
        problems.append(f"{name}: Nexus reports {remote_size} bytes, expected {size}")
    match = ETAG_SHA1_RE.search(stored.get("ETag", ""))
    if match and match.group(1) != sha1:
        problems.append(f"{name}: Nexus SHA-1 {match.group(1)} does not match uploaded {sha1}")
    return problems


class Backup:
    def __init__(self, args, gh_token: Optional[str], nexus_auth: Dict[str, str]):
        self.repo = args.repo
        self.base = args.nexus_url.rstrip("/")
        self.prefix = args.prefix.strip("/")
        self.dry_run = args.dry_run
        self.force = args.force
        self.include_source = args.source
        self.gh_token = gh_token
        self.nexus_auth = nexus_auth
        self.rows: List[Tuple[str, str, str]] = []
        self.problems: List[str] = []
        self.uploaded_bytes = 0
        self.warned_unreadable = False

    def path_for(self, tag: str, *parts: str) -> str:
        return nexus_url(self.base, self.prefix, tag, *parts)

    def record(self, status: str, name: str, detail: str = ""):
        self.rows.append((status, name, detail))
        log(f"  {status:<9} {name}{f'  ({detail})' if detail else ''}")

    def existing_size(self, url: str) -> Optional[int]:
        try:
            stored = head(url, self.nexus_auth)
        except urllib.error.HTTPError as exc:
            # A dry run needs no credentials, but then Nexus may refuse the read; report the
            # objects as absent rather than failing, so the size estimate still comes out.
            if exc.code in (401, 403) and self.dry_run:
                if not self.warned_unreadable:
                    log("  note: Nexus read refused without credentials; presence checks skipped")
                    self.warned_unreadable = True
                return None
            raise
        if stored is None:
            return None
        length = stored.get("Content-Length")
        return int(length) if length is not None else None

    def transfer(
        self,
        source_url: str,
        gh_headers: Dict[str, str],
        dest_url: str,
        name: str,
        expected_size: Optional[int],
        expected_sha256: Optional[str],
        content_type: str,
        workdir: Path,
    ) -> Optional[dict]:
        """Copies one object from GitHub to Nexus. Returns its recorded metadata."""
        if not self.force:
            present = self.existing_size(dest_url)
            # A size match is a cheap staleness check; SHA256SUMS carries the strong proof.
            if present is not None and (expected_size is None or present == expected_size):
                self.record("present", name, f"{present} bytes")
                return {"name": name, "size": present, "sha256": expected_sha256, "skipped": True}

        if self.dry_run:
            size_text = f"{expected_size} bytes" if expected_size is not None else "size unknown"
            self.record("would-add", name, size_text)
            return {"name": name, "size": expected_size, "sha256": expected_sha256, "skipped": True}

        local = workdir / "payload"
        try:
            sha256, sha1, size = download(source_url, gh_headers, local)
            if expected_sha256 and sha256 != expected_sha256:
                self.problems.append(f"{name}: download hash {sha256} != GitHub digest {expected_sha256}")
                self.record("FAILED", name, "hash mismatch on download")
                return None
            put(dest_url, self.nexus_auth, local, content_type)
            found = verify_upload(dest_url, self.nexus_auth, size, sha1, name)
            if found:
                self.problems.extend(found)
                self.record("FAILED", name, "verification failed")
                return None
            self.uploaded_bytes += size
            self.record("uploaded", name, f"{size} bytes")
            return {"name": name, "size": size, "sha256": sha256, "skipped": False}
        finally:
            local.unlink(missing_ok=True)

    def run_release(self, release: dict, workdir: Path):
        tag = release["tag_name"]
        log(f"\n{tag}  ({len(release.get('assets') or [])} asset(s))")
        gh_asset_headers = github_headers(self.gh_token, "application/octet-stream")
        recorded: List[dict] = []

        for asset in release.get("assets") or []:
            if asset.get("state") != "uploaded":
                self.record("skipped", asset["name"], f"state={asset.get('state')}")
                continue
            entry = self.transfer(
                source_url=asset["url"],
                gh_headers=gh_asset_headers,
                dest_url=self.path_for(tag, asset["name"]),
                name=asset["name"],
                expected_size=asset.get("size"),
                expected_sha256=digest_sha256(asset),
                content_type=asset.get("content_type") or "application/octet-stream",
                workdir=workdir,
            )
            if entry:
                entry.update(
                    content_type=asset.get("content_type"),
                    download_count=asset.get("download_count"),
                    created_at=asset.get("created_at"),
                    nexus_path=f"{self.prefix}/{tag}/{asset['name']}",
                )
                recorded.append(entry)

        source_entries = self.run_source_archives(release, tag, workdir) if self.include_source else []
        self.write_metadata(release, tag, recorded, source_entries)

    def run_source_archives(self, release: dict, tag: str, workdir: Path) -> List[dict]:
        """Copies GitHub's generated source archives, whose sizes are unknown until fetched."""
        short = self.repo.split("/")[-1]
        entries = []
        for key, suffix, content_type in (
            ("tarball_url", "tar.gz", "application/gzip"),
            ("zipball_url", "zip", "application/zip"),
        ):
            url = release.get(key)
            if not url:
                continue
            name = f"{short}-{tag}-source.{suffix}"
            entry = self.transfer(
                source_url=url,
                gh_headers=github_headers(self.gh_token),
                dest_url=self.path_for(tag, "source", name),
                name=f"source/{name}",
                expected_size=None,
                expected_sha256=None,
                content_type=content_type,
                workdir=workdir,
            )
            if entry:
                entry["nexus_path"] = f"{self.prefix}/{tag}/source/{name}"
                entries.append(entry)
        return entries

    def write_metadata(self, release: dict, tag: str, assets: List[dict], sources: List[dict]):
        """Writes the manifest, checksums and release notes; always refreshed, never skipped."""
        manifest = {
            "schema": 1,
            "repository": self.repo,
            "tag": tag,
            "release_id": release.get("id"),
            "release_name": release.get("name"),
            "commit_sha": resolve_commit(self.repo, tag, self.gh_token),
            "target_commitish": release.get("target_commitish"),
            "prerelease": release.get("prerelease"),
            "created_at": release.get("created_at"),
            "published_at": release.get("published_at"),
            "html_url": release.get("html_url"),
            "backed_up_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "nexus_base_url": self.base,
            "assets": [{k: v for k, v in a.items() if k != "skipped"} for a in assets],
            "source_archives": [{k: v for k, v in s.items() if k != "skipped"} for s in sources],
        }
        lines = [f"{a['sha256']}  {a['name']}" for a in assets + sources if a.get("sha256")]
        unknown = [a["name"] for a in assets + sources if not a.get("sha256")]
        if unknown:
            log(f"  note: no SHA-256 recorded for {len(unknown)} object(s); re-run with --force to compute")

        if self.dry_run:
            self.record("would-add", "manifest.json / SHA256SUMS / release-notes.md")
            return

        put_bytes(
            self.path_for(tag, "manifest.json"),
            self.nexus_auth,
            json.dumps(manifest, indent=2).encode(),
            "application/json",
        )
        put_bytes(
            self.path_for(tag, "SHA256SUMS"),
            self.nexus_auth,
            ("\n".join(lines) + "\n").encode(),
            "text/plain",
        )
        put_bytes(
            self.path_for(tag, "release-notes.md"),
            self.nexus_auth,
            (release.get("body") or "").encode(),
            "text/markdown",
        )
        self.record("uploaded", "manifest.json / SHA256SUMS / release-notes.md")

    def summarize(self) -> str:
        counts: Dict[str, int] = {}
        for status, _, _ in self.rows:
            counts[status] = counts.get(status, 0) + 1
        parts = [f"{count} {status}" for status, count in sorted(counts.items())]
        gib = self.uploaded_bytes / (1 << 30)
        return f"{', '.join(parts)} — {gib:.2f} GiB transferred"


def write_step_summary(backup: Backup, tags: List[str]):
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    lines = [
        f"## Release backup to Nexus — {', '.join(tags) or 'nothing'}",
        "",
        f"`{backup.base}/{backup.prefix}/`",
        "",
        backup.summarize(),
        "",
        "| status | object |",
        "|---|---|",
    ]
    lines += [f"| {status} | `{name}` |" for status, name, _ in backup.rows]
    if backup.problems:
        lines += ["", "### Problems", ""] + [f"- {p}" for p in backup.problems]
    with open(target, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Back up GitHub release assets to a Nexus raw repository.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--tag", help="Back up a single release tag")
    scope.add_argument("--all", action="store_true", help="Back up every non-draft release")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO,
        help="owner/repo to read releases from",
    )
    parser.add_argument(
        "--nexus-url",
        default=os.environ.get("NEXUS_URL") or DEFAULT_NEXUS_URL,
        help="Nexus raw repository URL",
    )
    parser.add_argument("--prefix", default="releases", help="Path prefix inside the repository")
    parser.add_argument(
        "--no-source", dest="source", action="store_false", help="Skip GitHub's generated source archives"
    )
    parser.add_argument("--force", action="store_true", help="Re-upload objects already in Nexus")
    parser.add_argument("--dry-run", action="store_true", help="Report what would transfer; upload nothing")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Succeed when a release has no assets (a release published before its installers are attached)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    gh_token = os.environ.get("GITHUB_TOKEN") or None

    nexus_auth: Dict[str, str] = {}
    if not args.dry_run:
        user = os.environ.get("NEXUS_USERNAME")
        password = os.environ.get("NEXUS_PASSWORD")
        if not user or not password:
            die("NEXUS_USERNAME and NEXUS_PASSWORD must be set (or pass --dry-run).")
        nexus_auth = basic_auth(user, password)

    if args.all:
        releases = [r for r in list_releases(args.repo, gh_token) if not r.get("draft")]
        if not releases:
            die(f"No published releases found in {args.repo}.")
    else:
        releases = [
            request_json(f"{GITHUB_API}/repos/{args.repo}/releases/tags/{args.tag}", github_headers(gh_token))
        ]

    backup = Backup(args, gh_token, nexus_auth)
    log(f"{args.repo} -> {backup.base}/{backup.prefix}/" + ("  [dry run]" if args.dry_run else ""))

    empty = []
    with tempfile.TemporaryDirectory(prefix="wiser-nexus-") as tmp:
        for release in sorted(releases, key=lambda r: r.get("published_at") or ""):
            if not (release.get("assets") or []):
                empty.append(release["tag_name"])
            backup.run_release(release, Path(tmp))

    tags = [r["tag_name"] for r in releases]
    log(f"\n{backup.summarize()}")
    write_step_summary(backup, tags)

    if backup.problems:
        log("\nProblems:")
        for problem in backup.problems:
            log(f"  - {problem}")
        return 1
    if empty:
        joined = ", ".join(empty)
        # An empty release is expected while sweeping history (v1.3b1 never had assets
        # attached), but for a named tag it usually means the installers are not up yet.
        if args.tag and not args.allow_empty:
            die(
                f"No assets on {joined}. WISER publishes a release before attaching installers, "
                "so re-run once they are uploaded (or pass --allow-empty)."
            )
        log(f"\nNote: no assets attached to {joined}; only metadata was stored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

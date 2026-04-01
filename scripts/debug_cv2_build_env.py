#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import cv2


def print_header(title: str) -> None:
    line = "=" * len(title)
    print()
    print(line)
    print(title)
    print(line)


def print_file_contents(path: Path) -> None:
    print(f"--- {path} ---")
    try:
        print(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        print(path.read_text(errors="replace"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"<failed to read: {exc}>")


def main() -> None:
    cv2_file = Path(cv2.__file__).resolve()
    cv2_dir = cv2_file.parent
    versioned_config = cv2_dir / f"config-{sys.version_info.major}.{sys.version_info.minor}.py"

    print_header("BUILD ENV CV2 INFO")
    print(f"sys.executable: {sys.executable}")
    print(f"cv2.__file__: {cv2.__file__}")
    print(f"cv2_dir: {cv2_dir}")

    print_header("CONFIG FILES")
    config_candidates = {cv2_dir / "config.py", versioned_config}
    config_candidates.update(path for path in cv2_dir.rglob("config*.py") if path.is_file())
    if config_candidates:
        for path in sorted(config_candidates):
            print_file_contents(path)
    else:
        print("No config*.py files found under cv2_dir.")

    print_header("CANDIDATE CV2 SHARED LIBRARIES")
    shared_libs = sorted(
        path
        for path in cv2_dir.rglob("*")
        if path.is_file() and path.name.startswith("cv2") and path.suffix == ".so"
    )
    if shared_libs:
        for path in shared_libs:
            print(path)
    else:
        print("No cv2*.so files found under cv2_dir.")

    print_header("RELEVANT CV2 PACKAGE FILES")
    relevant_files = sorted(
        path
        for path in cv2_dir.rglob("*")
        if path.is_file()
        and (path.name.startswith("config") or path.suffix == ".so" or "python-" in str(path.parent))
    )
    if relevant_files:
        for path in relevant_files:
            print(path)
    else:
        print("No relevant files matched the diagnostic filters.")


if __name__ == "__main__":
    main()

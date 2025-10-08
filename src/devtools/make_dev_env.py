#!/usr/bin/env python3
import sys
import re
import io
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "This script needs PyYAML. In CI we install it with `pip install pyyaml`.",
        file=sys.stderr,
    )
    sys.exit(1)


def canonical_conda_name(s: str) -> str:
    """Normalize a conda dep string to just the base package name:
    - drop comments, list bullets, channel prefix, build selectors
    - strip version/comparison parts
    """
    s = s.strip()
    if s.startswith("- "):
        s = s[2:].strip()
    s = s.split("#", 1)[0].strip()
    # remove build selector like [build=openblas]
    s = re.sub(r"\[.*?\]", "", s)
    # drop channel prefix 'conda-forge::'
    if "::" in s:
        s = s.split("::", 1)[1]
    # drop version/operator part
    s = re.split(r"[<>=! ]", s, maxsplit=1)[0]
    return s.lower()


def canonical_pip_name(s: str) -> str:
    s = s.strip()
    if s.startswith("- "):
        s = s[2:].strip()
    s = s.split("#", 1)[0].strip()
    return re.split(r"[=<>!~ ]", s, maxsplit=1)[0].lower()


def parse_additions(txt: str):
    conda_deps, pip_deps = [], []
    mode = None
    for raw in io.StringIO(txt):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("dependencies"):
            mode = "conda"
            continue
        if lower.startswith("pip"):
            mode = "pip"
            continue
        # normalize possible "- " prefix
        line = line[2:].strip() if line.startswith("- ") else line
        if mode == "conda":
            conda_deps.append(line)
        elif mode == "pip":
            pip_deps.append(line)
    return conda_deps, pip_deps


def split_deps_keep_order(deps_list):
    """Return (conda_items:list[str|dict_except_pip], pip_block:list|None, had_pip_string:bool)"""
    conda_items = []
    pip_block = None
    had_pip_string = False

    for d in deps_list or []:
        if isinstance(d, dict) and "pip" in d:
            pip_block = d["pip"]
        elif isinstance(d, str) and d.strip().lower() == "pip":
            had_pip_string = True
        else:
            conda_items.append(d)
    return conda_items, pip_block, had_pip_string


def main(prod_yml, additions_txt, out_yml):
    prod = yaml.safe_load(Path(prod_yml).read_text(encoding="utf-8")) or {}
    deps = prod.get("dependencies") or []

    # Separate conda items and existing pip block/string, preserving order
    conda_items, pip_block, had_pip_string = split_deps_keep_order(deps)

    add_conda, add_pip = parse_additions(Path(additions_txt).read_text(encoding="utf-8"))

    # Merge conda deps (skip duplicates by canonical name)
    existing_names = {canonical_conda_name(d) for d in conda_items if isinstance(d, str)}
    for dep in add_conda:
        if canonical_conda_name(dep) not in existing_names:
            conda_items.append(dep)
            existing_names.add(canonical_conda_name(dep))

    # Merge pip deps if any provided; create pip block if needed
    if add_pip:
        if pip_block is None:
            pip_block = []
        existing_pip_names = {canonical_pip_name(p) for p in pip_block}
        for p in add_pip:
            if canonical_pip_name(p) not in existing_pip_names:
                pip_block.append(p)
                existing_pip_names.add(canonical_pip_name(p))

    # Rebuild dependencies: conda first, pip last
    new_deps = list(conda_items)
    if pip_block is not None:
        # Only add the 'pip' package string if we CREATED the pip block and didn't have it before
        if not had_pip_string and add_pip:
            new_deps.append("pip")
        new_deps.append({"pip": pip_block})

    prod["dependencies"] = new_deps

    # Optional: give the dev env a different name if you want
    # prod["name"] = "wiser-dev"

    Path(out_yml).write_text(yaml.safe_dump(prod, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_yml}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: make_dev_env.py <wiser-prod.yml> <wiser-dev-additions.txt> <wiser-dev.yml>",
            file=sys.stderr,
        )
        sys.exit(2)
    main(*sys.argv[1:])

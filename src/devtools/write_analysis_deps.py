# --- collect resolved distributions from PyInstaller's Analysis ---
def write_deps_from_analysis(analysis_obj, out_path="build/pyinstaller_dependencies.txt",
                             exclude_modules=("wiser",), exclude_dists=()):
    """
    Collect top-level Python modules from the Analysis (pure + C extensions),
    map them to installed distributions via importlib.metadata, and write
    'dist==version' lines to a file.
    """
    from pathlib import Path
    import importlib.metadata as im

    # Map importable top-level module -> distribution(s)
    pkg_to_dists = im.packages_distributions()

    # 1) Modules found by PyInstaller (pure Python)
    module_tops = set()
    for dest, _src, typ in analysis_obj.pure:
        # 'dest' is like 'numpy.core' or 'wiser.__main__'
        module_tops.add(dest.split('.', 1)[0])

    # 2) C-extensions (.pyd/.so) listed in binaries as EXTENSION
    for dest, _src, typ in analysis_obj.binaries:
        if typ == 'EXTENSION':
            # dest can be 'numpy/core/_multiarray_umath.pyd' or similar
            normalized = dest.replace("\\", "/")
            top = normalized.split('/', 1)[0].split('.', 1)[0]
            module_tops.add(top)

    # Filter out your own project or anything you don't want
    module_tops = {
        m for m in module_tops
        if not any(m == ex or m.startswith(ex + ".") for ex in exclude_modules)
    }

    # Map modules -> distributions (some modules map to multiple dists, e.g. namespaces)
    dists = set()
    for mod in module_tops:
        for dist in pkg_to_dists.get(mod, []):
            if dist not in exclude_dists:
                dists.add(dist)

    # Resolve versions and write
    rows = []
    for dist in sorted(dists, key=str.lower):
        try:
            ver = im.version(dist)
        except Exception:
            # Skip things without a distribution record
            continue
        rows.append(f"{dist}=={ver}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[deps] wrote {len(rows)} packages to {out}")

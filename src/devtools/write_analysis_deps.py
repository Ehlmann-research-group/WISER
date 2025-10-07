
# --- robust deps writer: works even without importlib.metadata.packages_distributions() ---
def write_deps_from_analysis(analysis_obj, out_path="build/pyinstaller_dependencies.txt",
                             exclude_modules=("wiser",), exclude_dists=()):
    """
    Collect the top-level modules that PyInstaller actually analyzed (pure + C-extensions),
    map them to installed distributions and write 'dist==version' lines.
    """
    from pathlib import Path
    try:
        import importlib.metadata as im
    except Exception:
        import importlib_metadata as im  # backport if present

    def build_packages_distributions_fallback():
        """Return {top_level_pkg: [dist_name, ...]} using top_level.txt or file scanning."""
        mapping = {}
        for dist in im.distributions():
            dist_name = dist.metadata.get('Name', '')
            tops = set()

            # Primary: top_level.txt
            try:
                tl = dist.read_text('top_level.txt')
            except Exception:
                tl = None
            if tl:
                for line in tl.splitlines():
                    s = line.strip()
                    if s and s != '__pycache__':
                        tops.add(s)

            # Fallback: inspect files to infer top-level package names
            if not tops:
                files = dist.files or []
                for f in files:
                    parts = getattr(f, 'parts', None) or str(f).split('/')
                    if not parts:
                        continue
                    top = parts[0]
                    if top.endswith('.py'):
                        top = top[:-3]
                    if top and top != '__pycache__':
                        tops.add(top)

            for pkg in tops:
                mapping.setdefault(pkg, []).append(dist_name)
        return mapping

    # Prefer stdlib API if available, else fallback
    try:
        pkg_to_dists = im.packages_distributions()
    except Exception:
        pkg_to_dists = build_packages_distributions_fallback()

    # Collect top-level module names from what PyInstaller actually included
    module_tops = set()

    # Pure Python modules
    for dest, _src, _typ in analysis_obj.pure:
        module_tops.add(dest.split('.', 1)[0])

    # C-extensions (.pyd/.so) detected as EXTENSION; be lenient on type and filename
    for dest, _src, typ in analysis_obj.binaries:
        norm = dest.replace("\\", "/")
        if typ == 'EXTENSION' or norm.endswith(('.pyd', '.so')):
            top = norm.split('/', 1)[0]
            if top.endswith(('.pyd', '.so')):
                top = top.rsplit('.', 1)[0]
            if top:
                module_tops.add(top)

    # Exclude your own package(s) etc.
    module_tops = {
        m for m in module_tops
        if not any(m == ex or m.startswith(ex + ".") for ex in exclude_modules)
    }

    # Map modules -> distributions, then resolve versions
    dists = set()
    for mod in module_tops:
        for dist in pkg_to_dists.get(mod, []):
            if dist and dist not in exclude_dists:
                dists.add(dist)

    lines = []
    for dist in sorted(dists, key=str.lower):
        try:
            ver = im.version(dist)
        except Exception:
            continue
        lines.append(f"{dist}=={ver}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[deps] wrote {len(lines)} packages to {out}")


import os
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"<failed to read: {exc}>"


def _find_cv2_python_folder(meipass: str | None) -> str | None:
    if not (meipass and getattr(sys, "frozen", False)):
        return None

    cv2_root = os.path.join(meipass, "cv2")
    if not os.path.isdir(cv2_root):
        return None

    for name in os.listdir(cv2_root):
        if name.startswith("python-"):
            candidate = os.path.join(cv2_root, name)
            if os.path.isdir(candidate):
                return candidate
    return None


def _write_runtime_debug(root: Path, cv2_python_folder: str | None) -> None:
    debug_path = root / "cv2_runtime_debug.txt"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    lines = []

    def add_header(title: str) -> None:
        line = "=" * len(title)
        lines.extend(["", line, title, line])

    meipass = getattr(sys, "_MEIPASS", None)
    add_header("ROOT INFO")
    lines.append(f"sys.executable: {sys.executable}")
    lines.append(f"sys._MEIPASS: {meipass}")
    lines.append(f"resolved_root: {root}")
    lines.append(f"cv2_python_folder: {cv2_python_folder}")

    add_header("SYS.PATH")
    for index, entry in enumerate(sys.path):
        lines.append(f"{index}: {entry}")

    bundled_init_files = sorted(root.rglob("cv2/__init__.py"))
    add_header("BUNDLED CV2 PACKAGE FILES")
    if bundled_init_files:
        for path in bundled_init_files:
            lines.append(str(path))
    else:
        lines.append("No bundled cv2/__init__.py files found.")

    bundled_config_files = sorted(path for path in root.rglob("config*.py") if "cv2" in path.parts)
    add_header("BUNDLED CV2 CONFIG FILES")
    if bundled_config_files:
        for path in bundled_config_files:
            lines.append(str(path))
            lines.append(_read_text(path))
    else:
        lines.append("No bundled cv2 config*.py files found.")

    bundled_shared_libs = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.startswith("cv2") and path.suffix == ".so"
    )
    add_header("CANDIDATE CV2 SHARED LIBRARIES")
    if bundled_shared_libs:
        for path in bundled_shared_libs:
            lines.append(str(path))
        likely_parent = bundled_shared_libs[0].parent
        lines.append(f"likely_shared_lib_parent: {likely_parent}")
        lines.append(f"likely_shared_lib_parent_on_sys_path: {str(likely_parent) in sys.path}")
    else:
        lines.append("No bundled cv2*.so files found.")

    try:
        debug_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


meipass = getattr(sys, "_MEIPASS", None)
root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
cv2_python_folder = _find_cv2_python_folder(meipass)
_write_runtime_debug(root, cv2_python_folder)

if cv2_python_folder:
    try:
        meipass_index = sys.path.index(meipass)
    except ValueError:
        meipass_index = None

    if cv2_python_folder not in sys.path:
        if meipass_index is not None:
            sys.path.insert(meipass_index, cv2_python_folder)
        else:
            sys.path.insert(0, cv2_python_folder)

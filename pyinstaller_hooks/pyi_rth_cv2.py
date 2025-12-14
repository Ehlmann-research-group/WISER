import os
import sys

cv2_python_folder = None

# Get sys._MEIPASS (only exists in PyInstaller)
meipass = getattr(sys, "_MEIPASS", None)

if meipass and getattr(sys, "frozen", False):
    # Check if <_MEIPASS>/cv2 exists
    cv2_root = os.path.join(meipass, "cv2")
    if os.path.isdir(cv2_root):
        # Look inside for a subfolder that starts with "python-"
        for name in os.listdir(cv2_root):
            if name.startswith("python-"):
                candidate = os.path.join(cv2_root, name)
                if os.path.isdir(candidate):
                    cv2_python_folder = candidate
                    break

# If we found it, put it on sys.path right before _MEIPASS
if cv2_python_folder:
    try:
        meipass_index = sys.path.index(meipass)
    except ValueError:
        meipass_index = None

    if cv2_python_folder not in sys.path:
        if meipass_index is not None:
            # Insert *right before* sys._MEIPASS
            sys.path.insert(meipass_index, cv2_python_folder)
        else:
            # Fallback: put it at the front
            sys.path.insert(0, cv2_python_folder)

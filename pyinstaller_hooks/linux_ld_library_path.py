import os
import sys

# In onedir mode, sys._MEIPASS points to the app's top-level directory
# (where your collected .so files typically live).
base = getattr(sys, "_MEIPASS", None)
if base:
    old = os.environ.get("LD_LIBRARY_PATH", "")
    # Prepend bundled dir so its libs are found first
    os.environ["LD_LIBRARY_PATH"] = base + (":" + old if old else "")

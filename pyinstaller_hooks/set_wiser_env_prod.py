# PyInstaller runtime hook: default WISER_ENV to 'prod' for frozen apps
import os

# Only set if not already provided by the environment
os.environ.setdefault("WISER_ENV", "prod")



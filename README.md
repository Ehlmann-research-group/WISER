# WISER: The Workbench for Imaging Spectroscopy Exploration and Research

WISER is an open-source, extensible tool for visualizing and analyzing spectral
imaging data. It is written in Python and provides a cross-platform GUI built
on Qt 5 with PySide2. GDAL is used for loading and saving spectral data, and
NumPy for internal data representation.

WISER is supported on macOS (ARM and Intel), Windows 10/11, and Linux.

## Documentation

- **[WISER Documentation](https://ehlmann-research-group.github.io/WISER/)** —
  User manual, developer guide, and plugin API reference
- **[Plugin Repository](https://github.com/Ehlmann-research-group/WISER-Plugin-Repository)** —
  Community-contributed plugins

## Quick Start (Development)

```bash
# Clone the repo and set up the dev environment
cd etc
make install-dev-env        # macOS/Linux
# On Mac, specify architecture: make install-dev-env ENV=arm (or ENV=intel)

# Activate and run
conda activate wiser-dev
cd ../src
python -m wiser
```

For full setup instructions, see the
[Developer Environment Setup](doc/sphinx-general-wiser-docs/source/developer-content/environment-setup.md)
guide.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make wiser package importable for sphinx.ext.autodoc (used by extending-wiser pages)
sys.path.insert(0, os.path.abspath("../../../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "general-wiser-docs"
copyright = "2026, Joshua Garcia-Kimble"
author = "Joshua Garcia-Kimble"
release = "2.1b1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/Ehlmann-research-group/WISER",
    "use_repository_button": True,
    "max_navbar_depth": 10,
}

html_logo = "_static/icon_128x128.png"

html_title = "WISER Docs"

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "enum_tools.autoenum",
]

# autodoc2_packages = [
#     "../../../src/wiser",
# ]

autodoc2_output_dir = "api"
autodoct_render_plugin = "myst"

autodoc2_hidden_objects = {"inherited"}
autodoc2_class_docstring = "merged"

# sphinx.ext.autodoc options (used by extending-wiser plugin API pages)
autodoc_default_options = {
    "member-order": "bysource",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_css_files = ["style.css"]

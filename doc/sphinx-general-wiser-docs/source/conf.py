# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make wiser package importable for sphinx.ext.autodoc (used by extending-wiser pages)
sys.path.insert(0, os.path.abspath("../../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WISER"
copyright = "2019-2026, California Institute of Technology"
author = "Ehlmann Research Group"
release = "2.1b1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/Ehlmann-research-group/WISER",
    "repository_branch": "main",
    "path_to_docs": "doc/sphinx-general-wiser-docs/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "max_navbar_depth": 3,
}

html_logo = "_static/icon_128x128.png"
html_favicon = "_static/icon_128x128.png"

html_title = "WISER Documentation"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "enum_tools.autoenum",
]

myst_enable_extensions = [
    "colon_fence",  # ::: fences for Sphinx directives in Markdown
    "deflist",  # definition lists
    "tasklist",  # renders - [ ] checkboxes
    "attrs_inline",  # inline attribute syntax
]

# sphinx.ext.autodoc options (used by extending-wiser plugin API pages)
autodoc_default_options = {
    "member-order": "bysource",
}

autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_css_files = ["style.css"]

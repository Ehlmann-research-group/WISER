# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make wiser package importable for sphinx.ext.viewcode source links.
# autodoc2 uses AST-based analysis and does NOT import wiser at build time,
# so heavy runtime deps (PySide2/6, GDAL) are not required in the docs environment.
sys.path.insert(0, os.path.abspath("../../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WISER"
copyright = "2019-2026, California Institute of Technology"
author = "Ehlmann Research Group"
release = "3.0b0"

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
    "autodoc2",  # AST-based API docs — replaces sphinx.ext.autodoc
    "sphinx.ext.napoleon",  # Google/NumPy docstring support
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "colon_fence",  # ::: fences for Sphinx directives in Markdown
    "deflist",  # definition lists
    "tasklist",  # renders - [ ] checkboxes
    "attrs_inline",  # inline attribute syntax
]

# Generate anchors for headings h1-h3 so intra-page links like [text](#some-heading)
# resolve. Without this every such link is an unresolved xref.
myst_heading_anchors = 3

# autodoc2: index the wiser package via AST (no imports at build time)
autodoc2_packages = [
    {
        "path": "../../../src/wiser",
        "module": "wiser",
    }
]
# Write generated auto-API pages to apidocs/ but exclude them from the build —
# we use explicit .. autodoc2-object:: directives in hand-crafted .rst pages instead.
autodoc2_output_dir = "apidocs"
autodoc2_render_plugin = "rst"

autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = ["apidocs"]  # suppress orphan warnings for auto-generated API pages


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_css_files = ["style.css"]

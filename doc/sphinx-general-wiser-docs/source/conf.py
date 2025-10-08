# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "general-wiser-docs"
copyright = "2025, Joshua Garcia-Kimble"
author = "Joshua Garcia-Kimble"
release = "1.4b1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/Ehlmann-research-group/WISER",
    "use_repository_button": True,
}

html_logo = "_static/icon_128x128.png"

html_title = "WISER Docs"

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.autosectionlabel",
    "enum_tools.autoenum",
]

# autodoc2_packages = [
#     "../../../src/wiser",
# ]

autodoc2_output_dir = "api"
autodoct_render_plugin = "myst"

autodoc2_hidden_objects = {"inherited"}
autodoc2_class_docstring = "merged"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

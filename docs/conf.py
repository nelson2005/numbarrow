# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


sys.path.insert(0, os.path.abspath('..'))


project = "numbarrow"
copyright = "2025-2026, Mikhail Goykhman"
author = "Mikhail Goykhman"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# The trailing slash matters: sphinx-sitemap concatenates, so without it
# every emitted URL reads ".../numbarrowen/index.html".
html_baseurl = "https://goykhman.github.io/numbarrow/"
# The default scheme is "{lang}{version}{link}", which injects an "en/" segment
# for a directory the build never emits: the HTML is written flat and docs.yml
# publishes it verbatim, so every sitemap URL 404s. Sphinx's own canonical link
# is built from html_baseurl alone and is correct, so without this the two
# artifacts of one build disagree. sphinx-sitemap logs an info line rather than
# a warning, so `sphinx-build -W` does not catch it.
sitemap_url_scheme = "{link}"

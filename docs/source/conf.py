# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.append("../../")
sys.path.append(os.path.abspath('../minionpy'))
import minionpy

project = 'Minion'
copyright = '2025, Khoirul Faiq Muzakka'
author = 'Khoirul Faiq Muzakka'
release = '0.2.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'nbsphinx',  
    "breathe"
]
exclude_patterns = ['_build', '**.ipynb_checkpoints']


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
#html_theme = "alabaster"
html_static_path = ['_static']

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

breathe_projects = {
    "Minion": "../xml"
}
breathe_default_project = "Minion"



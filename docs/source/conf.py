# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import warnings
import importlib.util
sys.path.append("../../")
sys.path.append(os.path.abspath('../minionpy'))
import minionpy
try:
    from sphinx.deprecation import RemovedInSphinx80Warning
except ImportError:
    RemovedInSphinx80Warning = None

project = 'Minion'
copyright = '2025, Khoirul Faiq Muzakka'
author = 'Khoirul Faiq Muzakka'
release = '1.9.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]
if importlib.util.find_spec("nbsphinx") is not None:
    extensions.append("nbsphinx")
nbsphinx_execute = "never"
breathe_enabled = importlib.util.find_spec("breathe") is not None and os.path.exists(os.path.abspath("../xml/index.xml"))
if breathe_enabled:
    extensions.append("breathe")
templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
#html_theme = "alabaster"
html_static_path = ['_static'] if os.path.isdir(os.path.join(os.path.dirname(__file__), '_static')) else []

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

# Silence upstream Breathe/Sphinx compatibility deprecation noise during docs build.
if RemovedInSphinx80Warning is not None:
    warnings.filterwarnings("ignore", category=RemovedInSphinx80Warning, module=r"breathe\.project")

if not breathe_enabled:
    from docutils import nodes
    from docutils.parsers.rst import Directive

    class DoxygenIndexFallback(Directive):
        has_content = False
        option_spec = {"project": str}

        def run(self):
            paragraph = nodes.paragraph(
                text="Doxygen XML was not found; generate docs/xml to include the C++ API index."
            )
            return [paragraph]

    def setup(app):
        app.add_directive("doxygenindex", DoxygenIndexFallback)

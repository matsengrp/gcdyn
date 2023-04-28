# Configuration file for the Sphinx documentation builder.
#
# Taken from https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = "gcdyn"
author = "Erick Matsen"
copyright = '2022, Erick Matsen'

# No version in docs, doesn't play nice with versioneer
# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autodoc",
    # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    # Link to other project's documentation (see mapping below)
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.viewcode",
    # support NumPy and Google style docstrings
    "sphinx.ext.napoleon",
    # Automatically document param types (less noise in class signature)
    # NOTE: this disables autodoc_type_aliases used below (i.e. numpy.typing.ArrayLike are not properly condensed).
    "sphinx_autodoc_typehints",
    # track to do list items
    "sphinx.ext.todo",
    "sphinxarg.ext",
    # Copy button for code blocks
    'sphinx_copybutton',
    # render command line output
    # "sphinxcontrib.programoutput",
    # jupyter notebooks
    "myst_nb",
    # docstring tests
    "sphinx.ext.doctest",
]


# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ['dollarmath']
nb_execution_mode = "off"  # NOTE: this is off because we are committing executed notebooks for testing
nb_execution_allow_errors = False
nb_merge_streams = True

# show todos in output
todo_include_todos = True

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'ete3': ('http://etetoolkit.org/docs/latest/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'equinox': ('https://docs.kidger.site/equinox/', None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False
)  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no class summary, inherit base class summary

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    # 'special-members': '__init__',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/matsengrp/gcdyn',
    'use_repository_button': True,     # add a "link to repository" button
    "show_navbar_depth": 2,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# https://stackoverflow.com/questions/67473396/shorten-display-format-of-python-type-annotations-in-sphinx
# NOTE: the sphinx_autodoc_typehints extentension above disables this, so aliases are not properly condensed.
autodoc_type_aliases = {
    'ArrayLike': 'ArrayLike'
}

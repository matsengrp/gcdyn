# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
AUTODOC_STUBS = ${SOURCEDIR}/_autosummary

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

# Create a .py (percent format with multiline comments) from an .ipynb in the first place:
# jupytext --update-metadata '{"jupytext": {"cell_markers": "\"\"\""}}' --to py:percent <name-of-notebook>.ipynb

# Convert .py to .ipynb (don't seem to need to --execute? perhaps sphinx-build does it...), then build html:
%: Makefile
	rm -rf ${BUILDDIR}
	rm -rf ${AUTODOC_STUBS}
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

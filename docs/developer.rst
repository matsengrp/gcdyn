Developer tools
===============

Environment setup::

  export PYTHONNOUSERSITE=1  # isolate this environment from any existing system packages
  export CHANNELS="-c https://prefix.dev/conda-forge -c https://prefix.dev/bioconda"  # avoid using anaconda.org
  micromamba create -n gcdyn python=3.9 $CHANNELS
  micromamba activate gcdyn

Developer install::

  git clone git@github.com:matsengrp/gcdyn  #  -b alt-encode
  micromamba install $CHANNELS make
  make install

.. warning::

  Pip installation of ETE's PyQt5 dependency has been found to fail (with an error like `this <https://stackoverflow.com/questions/70961915/error-while-installing-pytq5-with-pip-preparing-metadata-pyproject-toml-did-n)>`_) on ARM Mac.
  You will need to install PyQt5 (e.g. with `Conda <https://anaconda.org/anaconda/pyqt>`_), and then try ``make install-no-pyqt`` instead of the command above.

Run tests::

  make test

Test notebooks::

  make notebooks

Format code::

  make format

Lint::

  make lint

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  make docs

Docs are automatically deployed to github pages via a workflow on push to the main branch.

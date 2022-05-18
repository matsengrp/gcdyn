Open source code repository
===========================

All code is freely available at `<https://github.com/matsengrp/gcdyn>`_

Developer tools
===============

Developer install::

  make install

.. warning::

  pip installation of the PyQt5 dependency has been found to fail (with an error like `this <https://stackoverflow.com/questions/70961915/error-while-installing-pytq5-with-pip-preparing-metadata-pyproject-toml-did-n)>`_) on ARM Mac.
  You will need to install PyQt5 (e.g. with `Conda <https://anaconda.org/anaconda/pyqt>`_), and then retry the command above.

Run tests::

  make test

Format code::

  make format

Lint::

  make lint

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  make docs

Docs are automatically deployed to github pages via a workflow on push to the main branch.

Todo list
=========

.. todolist::

Developer tools
===============

Developer install::

  make install

.. warning::

  Pip installation of ETE's PyQt5 dependency has been found to fail (with an error like `this <https://stackoverflow.com/questions/70961915/error-while-installing-pytq5-with-pip-preparing-metadata-pyproject-toml-did-n)>`_) on ARM Mac.
  You will need to install PyQt5 (e.g. with `Conda <https://anaconda.org/anaconda/pyqt>`_), and then try ``make install-no-pyqt`` instead of the command above.

Run tests::

  make test

Format code::

  make format

Lint::

  make lint

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  make docs

Docs are automatically deployed to github pages via a workflow on push to the main branch.

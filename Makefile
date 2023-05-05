default:

install:
	pip install -r requirements.txt
	pip install -e .[PyQt5]

install-no-pyqt:
	pip install -r requirements.txt
	pip install -e .

install-torchdms:
	pip install git+https://github.com/matsengrp/torchdms

# note: we use seperate pytest commands for notebooks because they are slow
#       so we don't want to run them on GitHub Actions macos runners. Otherwise
#       we would put everything into a pytest.ini file.
test:
	pytest --doctest-modules

notebooks:
	pytest --nbval notebooks/bdms-replay.ipynb notebooks/bdms-inhomogeneous.ipynb notebooks/message-passing.ipynb

format:
	black gcdyn experiments tests phylax

lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 . --count --max-complexity=30 --max-line-length=127 --statistics

docs:
	make -C docs html

.PHONY: install test notebooks format lint docs

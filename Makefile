default:

install:
	pip install -r requirements.txt
	pip install -e .[PyQt5]

install-no-pyqt:
	pip install -r requirements.txt
	pip install -e .

install-torchdms:
	pip install git+https://github.com/matsengrp/torchdms

test:
	pytest
	pytest --nbval notebooks/bdms_replay.ipynb notebooks/bdms_inhomogeneous.ipynb

format:
	black gcdyn tests
	docformatter --in-place gcdyn/*.py

lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 . --count --max-complexity=30 --max-line-length=127 --statistics

docs:
	make -C docs html

.PHONY: install test format lint docs

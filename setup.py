from setuptools import setup

import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gcdyn",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Erick Matsen",
    author_email="ematsen@gmail.com",
    description="inference of affinity-fitness response functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matsengrp/gcdyn",
    packages=["gcdyn", "experiments"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.9.*",
    install_requires=[
        "ete3",
        "matplotlib",
        "pandas",
        "jaxlib",
        "jax[cpu]",
        "jaxopt",
        "biopython",
        "diffrax",
        "equinox",
        "tensorflow",
        "tqdm",
    ],
    extras_require={
        "PyQt5": [
            "PyQt5",
        ],
    },
)

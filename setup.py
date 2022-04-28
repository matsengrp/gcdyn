import setuptools
import versioneer


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gcdyn",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Erick Matsen",
    author_email="ematsen@gmail.com",
    description="inference affinity-fitness response functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matsengrp/gcdyn",
    packages=['gcdyn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "ete3",
        "biopython",
        "matplotlib",
        "pandas",
        "scipy",
        "seaborn",
        "jaxlib",
        "jax",
    ],
)

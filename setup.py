#!/usr/bin/env python3

import os
import io
from setuptools import find_packages, setup

"""
git tag {VERSION}
git push --tags
python setup.py sdist
twine upload dist/*
"""

VERSION = "0.3.0"

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(
    name="repsim",
    version=VERSION,
    author="Richard Lange",
    description=("Representational similarity and geometry"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/wrongu/repsim/tarball/" + VERSION,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[],
    install_requires=[
        "torch"
    ]
)

from setuptools import setup, find_packages
from memmpy import __version__

with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="memmpy",
    version=__version__,
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=requirements,
    description="Memory mapped of datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pwolle/memmpy",
    author="Paul Wollenhaupt",
    author_email="paul.wollenhaupt@gmail.com",
)

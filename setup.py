import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="transformertopic",
    version="1.4",
    description="Topic modeling using sentence_transformer",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nareto/transformertopic",
    author="Renato Budinich",
    author_email="rennabh@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy",
                      "seaborn",
                      "sentence-transformers",
                      "hdbscan",
                      "pytextrank",
                      "umap-learn",
                      "pacmap",
                      "loguru"],
)

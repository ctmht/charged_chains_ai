from setuptools import setup, find_packages

setup(
    name = "charged_chains_ai",
    version = "0.0.1",
    package_dir = {"": "src"},
    packages = find_packages(where = "src"),
)
# To use: pip install -e .
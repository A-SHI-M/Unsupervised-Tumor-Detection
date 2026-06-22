from setuptools import setup, find_packages

setup(
    name="tumor_detection",
    version="0.0.1",
    author="A-SHI-M",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

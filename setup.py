from setuptools import find_packages, setup

setup(
    name="mallows",
    version="0.0.1-dev",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
    test_suite="tests",
    extras_require={},
)

"""Setup script for package."""
import re
from setuptools import setup, find_packages

VERSION = re.search(r'^VERSION\s*=\s*"(.*)"', open("komma_dev/version.py").read(), re.M).group(1)

with open("README.md", "rb") as f:
    LONG_DESCRIPTION = f.read().decode("utf-8")

setup(
    name="komma_dev",
    version=VERSION,
    description="Package for automatic comma placements for Danish texts.",
    long_description=LONG_DESCRIPTION,
    author="Soren Lind Kristiansen",
    author_email="sorenlind@mac.com",
    keywords="natural language processing danish",
    platforms=["Any"],
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    zip_safe=False,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=['jupyter', 'sklearn', 'tensorflow', 'keras', 'regex', 'pytest'],
    dependency_links=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.6',
    ])

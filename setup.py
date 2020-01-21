import setuptools
from distutils.core import Extension
import numpy

ext_modules = [
    Extension(
        "fastcountvectorizer._ext",
        sources=["fastcountvectorizer/_ext.cpp"],
        depends=["fastcountvectorizer/_ext.h"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastcountvectorizer",
    version="0.1.0",
    author="Santiago M. Mola",
    author_email="santi@mola.io",
    description="A faster CountVectorizer alternative.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smola/fastcountvectorizer",
    packages=["fastcountvectorizer", "fastcountvectorizer.tests"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=["scikit-learn", "numpy"],
    tests_require=["pytest"],
    ext_modules=ext_modules,
)

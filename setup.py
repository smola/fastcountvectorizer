import setuptools
from distutils.core import Extension
import numpy
from glob import glob

ext_modules = [
    Extension(
        "fastcountvectorizer._ext",
        sources=glob("fastcountvectorizer/*.cpp"),
        depends=glob("fastcountvectorizer/*.h")
        + glob("fastcountvectorizer/thirdparty/**/*.*", recursive=True),
        include_dirs=[numpy.get_include(), "fastcountvectorizer/thirdparty"],
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

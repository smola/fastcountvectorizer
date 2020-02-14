import platform
from glob import glob

import setuptools
from setuptools import Extension

extra_compile_args = []
if platform.system() in ["Linux", "Darwin"]:
    extra_compile_args += ["-Wall", "-Wextra"]
if platform.system() == "Darwin":
    extra_compile_args += ["-std=c++11", "-stdlib=libc++"]


class get_pybind11_include:
    # hack to evaluate pybind11 includes lazily, once they are installed
    def __str__(self):
        import pybind11

        return pybind11.get_include()


ext_modules = [
    Extension(
        "fastcountvectorizer._ext",
        sources=glob("fastcountvectorizer/*.cpp"),
        depends=glob("fastcountvectorizer/*.h")
        + glob("fastcountvectorizer/thirdparty/**/*.*", recursive=True),
        include_dirs=[get_pybind11_include(), "fastcountvectorizer/thirdparty"],
        language="c++",
        extra_compile_args=extra_compile_args,
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
    install_requires=["scikit-learn", "numpy", "pybind11"],
    setup_requires=["pybind11"],
    tests_require=["pytest"],
    ext_modules=ext_modules,
)

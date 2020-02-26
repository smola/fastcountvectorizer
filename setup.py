import sys
from glob import glob

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext


class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "fastcountvectorizer._ext",
        sources=glob("fastcountvectorizer/*.cpp"),
        depends=glob("fastcountvectorizer/*.h")
        + glob("fastcountvectorizer/thirdparty/**/*.*", recursive=True),
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            "fastcountvectorizer/thirdparty",
        ],
        language="c++",
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    # https://github.com/pybind/pybind11/issues/1604
    if sys.platform == "darwin":
        flags = ["-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        # macOS >= 10.12, because of https://github.com/pybind/pybind11/issues/1604
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.12"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append("-Wall")
            opts.append("-Wextra")
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
            # https://github.com/pybind/pybind11/issues/1604
            if has_flag(self.compiler, "-fsized-deallocation"):
                opts.append("-fsized-deallocation")
        elif ct == "msvc":
            pass
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastcountvectorizer",
    author="Santiago M. Mola",
    author_email="santi@mola.io",
    description="A faster CountVectorizer alternative.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smola/fastcountvectorizer",
    packages=["fastcountvectorizer", "fastcountvectorizer.tests"],
    keywords="sklearn scikit-learn nlp ngrams",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.5",
    install_requires=["scikit-learn", "numpy", "pybind11>=2.4"],
    setup_requires=["setuptools_scm", "pybind11>=2.4"],
    tests_require=["pytest"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    use_scm_version={"write_to": "fastcountvectorizer/version.py"},
)

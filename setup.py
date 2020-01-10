import setuptools

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
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["scikit-learn",],
    tests_require=["pytest"],
)

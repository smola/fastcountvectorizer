# FastCountVectorizer ![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/smola/fastcountvectorizer/fastcountvectorizer-ci/master) [![Documentation Status](https://readthedocs.org/projects/fastcountvectorizer/badge/?version=latest)](https://fastcountvectorizer.readthedocs.io/en/latest/?badge=latest)

FastCountVectorizer is a faster alternative to [scikit-learn](https://scikit-learn.org/)'s [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

## Installation

```
pip install fastcountvectorizer
```

## Documentation

See [full documentation](https://fastcountvectorizer.readthedocs.io/en/latest/).

## License

Copyright (c) 2020 Santiago M. Mola

FastCountVectorizer is released under the [MIT License](LICENSE).

The following files are included from or derived from third party projects:

* [`fastcountvectorizer.py`](fastcountvectorizer/fastcountvectorizer.py) is derived from scikit-learn's [`scikit-learn/sklearn/feature_extraction/text.py`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py), licensed under a 3-clause BSD license. The original list of authors and license text can be found in the file header.
* [`_csr.h`](fastcountvectorizer/_csr.h) is derived from scipy's [`scipy/sparse/sparsetools/csr.h`](https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h), licensed under a 3-clause BSD license. The original list of authors and license text can be found in the file header.
* `fastcountvectorizer/thirdparty/tsl` includes the [`tsl::sparse_map`](https://github.com/Tessil/sparse-map) project, released under the MIT License.
* `fastcountvectorizer/thirdparty` includes the [`xxHash`](https://github.com/Cyan4973/xxHash) project, released under a BSD-2 Clause license.


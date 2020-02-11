# FastCountVectorizer ![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/smola/fastcountvectorizer/fastcountvectorizer-ci/master)

FastCountVectorizer is a faster alternative to [scikit-learn](https://scikit-learn.org/)'s [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

## Installation

TBD

## Documentation

TBD

## Deviations from scikit-learn implementation

FastCountVectorizer behaves mostly as a subset of CountVectorizer. However, it doesn't do whitespace normalization. This is arguably a better default behavior, but [fixing it in scikit-learn would break backwards compatibility](https://github.com/scikit-learn/scikit-learn/issues/7475).

## License

Copyright (c) 2020 Santiago M. Mola

FastCountVectorizer is released under the [MIT License](LICENSE).

The following files are included from or derived from third party projects:

* [`fastcountvectorizer.py`](fastcountvectorizer/fastcountvectorizer.py) is derived from scikit-learn's [`scikit-learn/sklearn/feature_extraction/text.py`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py), licensed under a 3-clause BSD license. The original list of authors and license text can be found in the file header.
* `fastcountvectorizer/thirdparty/tsl` includes the [`tsl::sparse_map`](https://github.com/Tessil/sparse-map) project, released under the MIT License.


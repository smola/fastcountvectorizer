# -*- coding: utf-8 -*-
# Authors: Santiago M. Mola <santi@mola.io>
#
# This file is derived from scikit-learn/sklearn/feature_extraction/text.py
# whose authors are:
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck
#          Robert Layton <robertlayton@gmail.com>
#          Jochen Wersdörfer <jochen@wersdoerfer.de>
#          Roman Sinayev <roman.sinayev@gmail.com>
#
# Its original license is reproduced here:
#
# New BSD License
#
# Copyright (c) 2007–2019 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

import math
import numbers
from operator import itemgetter
import re

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.utils import _IS_32BIT

from ._ext import _CharNgramCounter


class FastCountVectorizer(BaseEstimator):
    """Convert a collection of text documents to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    input : string {'content'}
        Indicates the input type. Currently, only 'content' (default value) is
        supported. The input is expected to be a sequence of items of type
        string.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.

    analyzer : string, {'char', 'char_wb'}
        Analyzer mode. If set to 'char' (default value, only option) character
        ngrams will be used.

        .. warning:: FastCountVectorizer does not apply any kind of
           preprocessing to inputs. Note that this is different from
           scikit-learn's CountVectorizer performs, which applies whitespace
           normalization.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    binary : bool, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform(). Defaults
        to np.float64.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
    """

    def __init__(
        self,
        input="content",
        ngram_range=(1, 1),
        analyzer="char",
        min_df=1,
        max_df=1.0,
        binary=False,
        dtype=np.int64,
    ):
        self.input = input
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.dtype = dtype
        self.vocabulary_ = None

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : list
            A list of strings.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : list
            A list of strings.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # Parameters are only validated on fit (not on __init__)
        # to keep compatibility with CountVectorizer.
        self._validate_params()
        self._validate_raw_documents(raw_documents)
        vocab, X = self._count_vocab(raw_documents, fixed_vocab=False)
        if self.binary:
            X.data.fill(1)
        self.vocabulary_ = vocab
        return X

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : list
            A list of strings.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        self._validate_raw_documents(raw_documents)
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name.

        Returns
        -------
        feature_names : list
            A list of feature names.
        """
        return [
            self._to_string(t)
            for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))
        ]

    def _validate_params(self):
        if self.input != "content":
            raise ValueError('only input="content" is currently supported')
        if self.analyzer not in ("char", "char_wb"):
            raise ValueError("unsupported analyzer: %s" % self.analyzer)
        min_n, max_m = self.ngram_range
        if min_n > max_m:
            raise ValueError(
                "Invalid value for ngram_range=%s "
                "lower boundary larger than the upper boundary." % str(self.ngram_range)
            )
        if min_n <= 0:
            raise ValueError(
                "Invalid value for ngram_range=%s "
                "minimum ngram size must be equal or greater than 1."
            )

    def _validate_raw_documents(self, raw_documents):
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, " "string object received."
            )

    def _count_vocab(self, docs, fixed_vocab=False):
        if fixed_vocab:
            vocab = self.vocabulary_
        else:
            vocab = None

        min_ngram, max_ngram = self.ngram_range

        n_doc = 0
        counter = _CharNgramCounter(
            self.analyzer, min_ngram, max_ngram, fixed_vocab=vocab
        )
        for doc in docs:
            doc = self._preprocess(doc)
            counter.process(doc)
            n_doc += 1

        if not fixed_vocab:
            counter.expand_counts()

            min_df, max_df = self._frequency_limits(n_doc)
            if min_df > 1 or max_df < n_doc:
                self.stop_words_ = counter.limit_features(min_df, max_df)
            else:
                self.stop_words_ = set()
                counter.sort_features()

            vocab = counter.get_vocab()

        values, indices, indptr = counter.get_result()
        del counter

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(
                    (
                        "sparse CSR array has {} non-zero "
                        "elements and requires 64 bit indexing, "
                        "which is unsupported with 32 bit Python."
                    ).format(indptr[-1])
                )

        counts = sp.csr_matrix(
            (values, indices, indptr),
            shape=(len(indptr) - 1, len(vocab)),
            dtype=self.dtype,
        )

        counts.sort_indices()

        return vocab, counts.astype(self.dtype, copy=False)

    def _to_string(self, s):
        if isinstance(s, bytes):
            return s.decode("latin-1")
        return s

    def _frequency_limits(self, n_doc):
        min_df = self.min_df
        max_df = self.max_df

        max_df = max_df if isinstance(max_df, numbers.Integral) else max_df * n_doc
        min_df = min_df if isinstance(min_df, numbers.Integral) else min_df * n_doc
        if max_df < min_df:
            raise ValueError("max_df corresponds to < documents than min_df")

        return int(math.ceil(min_df)), int(max_df)

    _white_spaces_wb = re.compile(r"(^\s*|\s\s+|\s*$)")

    def _preprocess(self, doc):
        if self.analyzer == "char_wb":
            doc = self._white_spaces_wb.sub(" ", doc)

        return doc

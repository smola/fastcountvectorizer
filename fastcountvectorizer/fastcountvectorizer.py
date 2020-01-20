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

from array import array
from collections import defaultdict
import numbers
import numpy as np
import itertools
from operator import itemgetter
import scipy.sparse as sp
from sklearn.utils import _IS_32BIT
from sklearn.base import BaseEstimator
from ._ext import _count_ngrams


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

    analyzer : string, {'char'}
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

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform(). Defaults
        to np.float64.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    """

    def __init__(
        self,
        input="content",
        ngram_range=(1, 1),
        analyzer="char",
        min_df=1,
        max_df=1.0,
        dtype=np.int64,
    ):
        self.input = input
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_df = max_df
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
        vocab, X = self._count_vocab(raw_documents)
        X, vocab = self._limit_features(X, vocab)
        X = self._sort_features(X, vocab)
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
        _, X = self._count_fixed_vocab(raw_documents, self.vocabulary_)
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
        if self.analyzer != "char":
            raise ValueError('only analyzer="char" is currently supported')
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

    def _count_vocab(self, docs):
        vocab = defaultdict()
        vocab.default_factory = vocab.__len__

        min_ngram, max_ngram = self.ngram_range
        vocab, counts = self._count_vocab_from_docs(
            n=max_ngram, docs=docs, vocab=vocab,
        )

        if min_ngram == max_ngram:
            return dict(vocab), counts

        counts = _expand_counts(vocab, counts, min_ngram, max_ngram)
        prefix_counts = _prefix_counts(
            self._analyze_fixed, vocab, docs, min_ngram, max_ngram
        )
        counts.resize(prefix_counts.shape)
        counts += prefix_counts

        return dict(vocab), counts.astype(self.dtype)

    def _count_fixed_vocab(self, raw_documents, vocab):
        values = array("i")
        j_indices = []
        indptr = [0]

        for doc in raw_documents:
            counters = defaultdict(int)
            for term in self._analyze(doc):
                idx = vocab.get(term, -1)
                if idx >= 0:
                    counters[idx] += 1
            j_indices.extend(counters.keys())
            values.extend(counters.values())
            indptr.append(len(j_indices))

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(
                    (
                        "sparse CSR array has {} non-zero "
                        "elements and requires 64 bit indexing, "
                        "which is unsupported with 32 bit Python."
                    ).format(indptr[-1])
                )
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32

        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocab)),
            dtype=self.dtype,
        )

        X.sort_indices()

        return vocab, X

    def _to_string(self, s):
        if isinstance(s, bytes):
            return s.decode("latin-1")
        return s

    def _count_vocab_from_docs(self, n, docs, vocab):
        values, indices, indptr = _count_ngrams(n, docs, vocab)

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

        return vocab, counts

    def _sort_features(self, X, vocabulary):
        """Sort features by name
        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode="clip")
        return X

    def _limit_features(self, X, vocabulary):
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        This does not prune samples with zero features.
        """
        min_df = self.min_df
        max_df = self.max_df
        needs_limit = (
            (isinstance(min_df, numbers.Integral) and min_df > 1) or min_df > 0.0
        ) or ((isinstance(min_df, numbers.Integral) and max_df > 0) or max_df < 1.0)

        if not needs_limit:
            return X, vocabulary

        n_doc = X.shape[0]
        high = max_df if isinstance(max_df, numbers.Integral) else max_df * n_doc
        low = min_df if isinstance(min_df, numbers.Integral) else min_df * n_doc
        if high < low:
            raise ValueError("max_df corresponds to < documents than min_df")

        # Calculate a mask based on document frequencies
        dfs = self._document_frequency(X)
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        vocabulary = {t: new_indices[i] for t, i in vocabulary.items() if mask[i]}
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError(
                "After pruning, no terms remain. Try a lower"
                " min_df or a higher max_df."
            )
        return X[:, kept_indices], vocabulary

    def _document_frequency(self, X):
        """Count the number of non-zero values for each feature in sparse X."""
        if sp.isspmatrix_csr(X):
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            return np.diff(X.indptr)

    def _analyze(self, doc):
        doc_len = len(doc)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(0, doc_len - n + 1):
                yield doc[i : i + n]

    def _analyze_fixed(self, doc, n):
        if n == 1 and not isinstance(doc, bytes):
            return iter(doc)
        return (doc[i : i + n] for i in range(0, len(doc) - n + 1))


def _expand_counts(vocab, counts, min_ngram, max_ngram):
    initial_len = len(vocab)

    i_ = array("i")
    j_ = array("i")
    for term, idx in list(vocab.items()):
        j_.append(idx)
        i_.append(idx)
        for n in range(min_ngram, max_ngram):
            new_term = term[-n:]
            new_idx = vocab[new_term]
            j_.append(new_idx)
            i_.append(idx)

    n_iter = max_ngram - min_ngram + 1
    j_ = np.frombuffer(j_, dtype=np.intc)
    i_ = np.frombuffer(i_, dtype=np.intc)
    data = np.ones(initial_len * n_iter)

    modifier = sp.coo_matrix(
        (data, (i_, j_)), shape=(counts.shape[1], len(vocab)), dtype=np.int64,
    )

    return counts * modifier


def _prefix_counts(analyzer, vocab, docs, min_ngram, max_ngram):
    i_ = array("i")
    j_ = array("i")
    for row, doc in enumerate(docs):
        for n in range(min_ngram, max_ngram):
            for term in itertools.islice(analyzer(doc, n), max_ngram - n):
                idx = vocab[term]
                i_.append(row)
                j_.append(idx)

    i_ = np.frombuffer(i_, dtype=np.intc)
    j_ = np.frombuffer(j_, dtype=np.intc)
    values = np.ones(len(i_), dtype=np.intc)
    return sp.coo_matrix(
        (values, (i_, j_)), shape=(len(docs), len(vocab)), dtype=np.int64,
    ).tocsr()

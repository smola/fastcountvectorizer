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
import re
import warnings
from collections.abc import Mapping
from operator import itemgetter

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import (
    strip_accents_ascii,
    strip_accents_unicode,
)
from sklearn.utils import _IS_32BIT

from ._ext import _CharNgramCounter
from ._stopwords import ENGLISH_STOP_WORDS


class FastCountVectorizer(BaseEstimator):
    """Convert a collection of text documents to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    .. note:: This class is has some differences compared to
       `scikit-learn's CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_.
       These are noted below. You can also check the alternative compatibility API at :py:class:`CountVectorizer`.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}, default='content'
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be a sequence of items that
        can be of type string or byte.

    encoding : string, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.

    analyzer : string, {'char', 'word'}
        Whether the feature should be made of word n-gram or character n-grams.
        Defaults to word.

        .. warning:: FastCountVectorizer does not apply any kind of
           preprocessing to inputs. Note that this is different from
           CountVectorizer, which applies white space normalization for the char
           analyzer.

    stop_words : list, default=None
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

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

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

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
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        ngram_range=(1, 1),
        analyzer="word",
        stop_words=None,
        min_df=1,
        max_df=1.0,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.dtype = dtype
        self.vocabulary = vocabulary

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
        self._validate_vocabulary()
        self._validate_raw_documents(raw_documents)
        vocab, X = self._count_vocab(raw_documents, fixed_vocab=self.fixed_vocabulary_)
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
        self._check_vocabulary()
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
        self._check_vocabulary()
        return [
            self._to_string(t)
            for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))
        ]

    def _compat_mode(self):
        return False

    def _validate_params(self):
        self._warn_for_unused_params()
        if self.input not in ("content", "file", "filename"):
            raise ValueError("unsupported input=%s" % self.input)
        if self.analyzer not in ("char", "word"):
            raise ValueError("unsupported analyzer=%s" % self.analyzer)
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
        self._normalize_params()

    def _warn_for_unused_params(self):
        if self.analyzer != "word":
            if self.stop_words is not None:
                warnings.warn(
                    "The parameter 'stop_words' will not be used"
                    " since 'analyzer' != 'word'"
                )

    def _normalize_params(self):
        if self.stop_words is not None and not isinstance(self.stop_words, frozenset):
            if self._compat_mode():
                if self.stop_words == "english":
                    self.stop_words = ENGLISH_STOP_WORDS
                elif isinstance(self.stop_words, str):
                    raise ValueError("not a built-in stop list: %s" % self.stop_words)
                else:
                    self.stop_words = frozenset(self.stop_words)
            else:
                if isinstance(self.stop_words, str):
                    raise ValueError("stop_words cannot be a string")
                else:
                    self.stop_words = frozenset(self.stop_words)

    def _validate_raw_documents(self, raw_documents):
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, " "string object received."
            )

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(vocabulary.values())
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in range(len(vocabulary)):
                    if i not in indices:
                        msg = "Vocabulary of size %d doesn't contain index " "%d." % (
                            len(vocabulary),
                            i,
                        )
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fitted)"""
        if not hasattr(self, "vocabulary_"):
            self._validate_vocabulary()
            if not self.fixed_vocabulary_:
                raise NotFittedError("Vocabulary not fitted or provided")

        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

    def _count_vocab(self, docs, fixed_vocab=False):
        if fixed_vocab:
            vocab = self.vocabulary_
        else:
            vocab = None

        min_ngram, max_ngram = self.ngram_range

        n_doc = 0
        counter = _CharNgramCounter(
            self.analyzer,
            min_ngram,
            max_ngram,
            fixed_vocab=vocab,
            stop_words=self.stop_words,
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

    _white_spaces = re.compile(r"\s\s+")

    def _preprocess(self, doc):
        if self.input == "content":
            pass
        elif self.input == "filename":
            with open(doc, "r", encoding=self.encoding, errors=self.decode_error) as fh:
                doc = fh.read()
        elif self.input == "file":
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if self.strip_accents is not None:
            if self.strip_accents == "unicode":
                doc = strip_accents_unicode(doc)
            elif self.strip_accents == "ascii":
                doc = strip_accents_ascii(doc)
            else:
                raise ValueError(
                    'Invalid value for "strip_accents": %s' % self.strip_accents
                )

        if self.analyzer == "char" and self._compat_mode():
            doc = self._white_spaces.sub(" ", doc)

        return doc

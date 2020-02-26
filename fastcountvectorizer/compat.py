import numpy as np

from .fastcountvectorizer import FastCountVectorizer


class CountVectorizer(FastCountVectorizer):
    """Convert a collection of text documents to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    .. note:: This class is provided to maximize compatibility with
       `scikit-learn's CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_.
       You can also check our new :py:class:`FastCountVectorizer`.

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

        .. note:: The character analyzer (``analyzer="char"``) applies
           whitespace normalization to the input. This behavior is present for
           compatibility with scikit-learn's CountVectorizer (see
           `scikit-learn#7475 <https://github.com/scikit-learn/scikit-learn/issues/7475>`_).
           In contrast, :py:class:`FastCountVectorizer` does not do any whitespace
           normalization.

    stop_words : string {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (`read more <https://scikit-learn.org/stable/modules/feature_extraction.html#using-stop-words>`_).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``

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
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            ngram_range=ngram_range,
            analyzer=analyzer,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

    def get_stop_words(self):
        """Build or fetch the effective stop words list.
        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        self._normalize_params()
        return self.stop_words

    def _compat_mode(self):
        return True

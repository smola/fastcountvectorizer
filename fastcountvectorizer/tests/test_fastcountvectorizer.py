import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import lil_matrix

from fastcountvectorizer import FastCountVectorizer


def check_cv(cv, input, output, vocab):
    X = cv.fit_transform(input)
    assert vocab == cv.get_feature_names()
    assert_array_almost_equal(X.todense(), output.todense())
    assert_array_almost_equal(cv.transform(input).todense(), output.todense())


def test_fastcountvectorizer_validate_params():
    FastCountVectorizer().fit(["foo"])

    FastCountVectorizer(input="content")
    with pytest.raises(ValueError):
        FastCountVectorizer(input="file").fit(["foo"])
    with pytest.raises(ValueError):
        FastCountVectorizer(input="filename").fit(["foo"])
    with pytest.raises(ValueError):
        FastCountVectorizer(input="unsupported").fit(["foo"])

    FastCountVectorizer(analyzer="char").fit(["foo"])
    FastCountVectorizer(analyzer="word").fit(["foo"])
    with pytest.raises(ValueError):
        FastCountVectorizer(analyzer="char_wb").fit(["foo"])
    with pytest.raises(ValueError):
        FastCountVectorizer(input="unsupported").fit(["foo"])


def test_fastcountvectorizer_word_return_dtype():
    input = ["abc"]
    cv = FastCountVectorizer()
    result = cv.fit_transform(input)
    assert result.dtype == np.int64

    cv = FastCountVectorizer(dtype=np.int64)
    result = cv.fit_transform(input)
    assert result.dtype == np.int64

    cv = FastCountVectorizer(dtype=np.int32)
    result = cv.fit_transform(input)
    assert result.dtype == np.int32

    cv = FastCountVectorizer(dtype=np.float64)
    result = cv.fit_transform(input)
    assert result.dtype == np.float64


def test_fastcountvectorizer_char_ngram1():
    cv = FastCountVectorizer(analyzer="char", ngram_range=(1, 1))
    check_cv(
        cv, input=["abc"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["a", "b", "c"]
    )
    check_cv(
        cv, input=["cba"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["a", "b", "c"]
    )
    check_cv(
        cv,
        input=["cba", "ade"],
        output=lil_matrix([[1, 1, 1, 0, 0], [1, 0, 0, 1, 1]]).tocsr(),
        vocab=["a", "b", "c", "d", "e"],
    )


def test_fastcountvectorizer_word_ngram1():
    cv = FastCountVectorizer(analyzer="word", ngram_range=(1, 1))
    check_cv(
        cv,
        input=["aaa bbb ccc"],
        output=lil_matrix([[1, 1, 1]]).tocsr(),
        vocab=["aaa", "bbb", "ccc"],
    )
    check_cv(
        cv,
        input=["bbb aaa ccc"],
        output=lil_matrix([[1, 1, 1]]).tocsr(),
        vocab=["aaa", "bbb", "ccc"],
    )
    check_cv(
        cv,
        input=["ccc bbb aaa", "aaa  ddd\teee"],
        output=lil_matrix([[1, 1, 1, 0, 0], [1, 0, 0, 1, 1]]).tocsr(),
        vocab=["aaa", "bbb", "ccc", "ddd", "eee"],
    )


def test_fastcountvectorizer_char_ngram1_unicode():
    cv = FastCountVectorizer(analyzer="char", ngram_range=(1, 1))
    check_cv(
        cv, input=["ǟƂƇ"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["Ƃ", "Ƈ", "ǟ"]
    )
    check_cv(
        cv, input=["ƇƂǟ"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["Ƃ", "Ƈ", "ǟ"]
    )


def test_fastcountvectorizer_char_ngram1_2():
    cv = FastCountVectorizer(analyzer="char", ngram_range=(1, 2))
    check_cv(
        cv,
        input=["abc"],
        output=lil_matrix([[1, 1, 1, 1, 1]]).tocsr(),
        vocab=["a", "ab", "b", "bc", "c"],
    )
    check_cv(
        cv,
        input=["cba"],
        output=lil_matrix([[1, 1, 1, 1, 1]]).tocsr(),
        vocab=["a", "b", "ba", "c", "cb"],
    )
    check_cv(
        cv,
        input=["cba", "ade"],
        output=lil_matrix(
            [[1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1]]
        ).tocsr(),
        vocab=["a", "ad", "b", "ba", "c", "cb", "d", "de", "e"],
    )


def test_fastcountvectorizer_char_ngram1_3():
    cv = FastCountVectorizer(analyzer="char", ngram_range=(1, 3))
    check_cv(
        cv,
        input=["abcef"],
        vocab=["a", "ab", "abc", "b", "bc", "bce", "c", "ce", "cef", "e", "ef", "f"],
        output=lil_matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).tocsr(),
    )


def test_fastcountvectorizer_word_ngram1_3():
    cv = FastCountVectorizer(analyzer="word", ngram_range=(1, 3))
    check_cv(
        cv,
        input=["aaa bbb ccc eee fff"],
        vocab=[
            "aaa",
            "aaa bbb",
            "aaa bbb ccc",
            "bbb",
            "bbb ccc",
            "bbb ccc eee",
            "ccc",
            "ccc eee",
            "ccc eee fff",
            "eee",
            "eee fff",
            "fff",
        ],
        output=lil_matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).tocsr(),
    )

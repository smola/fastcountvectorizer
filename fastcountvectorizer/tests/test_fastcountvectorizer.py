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
    FastCountVectorizer(analyzer="char_wb").fit(["foo"])
    with pytest.raises(ValueError):
        FastCountVectorizer(analyzer="word").fit(["foo"])
    with pytest.raises(ValueError):
        FastCountVectorizer(input="unsupported").fit(["foo"])


def test_fastcountvectorizer_return_dtype():
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


def test_fastcountvectorizer_ngram1():
    cv = FastCountVectorizer(ngram_range=(1, 1))
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


def test_fastcountvectorizer_ngram1_char_wb():
    cv = FastCountVectorizer(analyzer="char_wb", ngram_range=(1, 1))
    check_cv(
        cv,
        input=["abc"],
        output=lil_matrix([[2, 1, 1, 1]]).tocsr(),
        vocab=[" ", "a", "b", "c"],
    )
    check_cv(
        cv,
        input=["cba"],
        output=lil_matrix([[2, 1, 1, 1]]).tocsr(),
        vocab=[" ", "a", "b", "c"],
    )
    check_cv(
        cv,
        input=["cba", "ade"],
        output=lil_matrix([[2, 1, 1, 1, 0, 0], [2, 1, 0, 0, 1, 1]]).tocsr(),
        vocab=[" ", "a", "b", "c", "d", "e"],
    )


def test_fastcountvectorizer_ngram1_unicode():
    cv = FastCountVectorizer(ngram_range=(1, 1))
    check_cv(
        cv, input=["ǟƂƇ"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["Ƃ", "Ƈ", "ǟ"]
    )
    check_cv(
        cv, input=["ƇƂǟ"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["Ƃ", "Ƈ", "ǟ"]
    )


def test_fastcountvectorizer_ngram1_2():
    cv = FastCountVectorizer(ngram_range=(1, 2))
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


def test_fastcountvectorizer_ngram1_2_char_wb():
    cv = FastCountVectorizer(analyzer="char_wb", ngram_range=(1, 2))
    check_cv(
        cv,
        input=["a bb"],
        output=lil_matrix([[4, 1, 1, 1, 2, 1, 1]]).tocsr(),
        vocab=[" ", " a", " b", "a", "b", "b ", "bb"],
    )
    check_cv(
        cv,
        input=["abc"],
        output=lil_matrix([[2, 1, 1, 1, 1, 1, 1, 1]]).tocsr(),
        vocab=[" ", " a", "a", "ab", "b", "bc", "c", "c "],
    )
    check_cv(
        cv,
        input=["cba"],
        output=lil_matrix([[2, 1, 1, 1, 1, 1, 1, 1]]).tocsr(),
        vocab=[" ", " c", "a", "a ", "b", "ba", "c", "cb"],
    )
    check_cv(
        cv,
        input=["cba", "ade"],
        output=lil_matrix(
            [
                [2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [2, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            ]
        ).tocsr(),
        vocab=[
            " ",
            " a",
            " c",
            "a",
            "a ",
            "ad",
            "b",
            "ba",
            "c",
            "cb",
            "d",
            "de",
            "e",
            "e ",
        ],
    )


def test_fastcountvectorizer_ngram1_3():
    cv = FastCountVectorizer(ngram_range=(1, 3))
    check_cv(
        cv,
        input=["abcef"],
        vocab=["a", "ab", "abc", "b", "bc", "bce", "c", "ce", "cef", "e", "ef", "f"],
        output=lil_matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).tocsr(),
    )

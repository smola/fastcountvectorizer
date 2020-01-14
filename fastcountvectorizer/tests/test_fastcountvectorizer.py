from scipy.sparse import lil_matrix
from numpy.testing import assert_array_almost_equal

from fastcountvectorizer import FastCountVectorizer


def check_cv(cv, input, output, vocab):
    X = cv.fit_transform(input)
    assert vocab == cv.get_feature_names()
    assert_array_almost_equal(X.todense(), output.todense())
    assert_array_almost_equal(cv.transform(input).todense(), output.todense())


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


def test_fastcountvectorizer_ngram1_3():
    cv = FastCountVectorizer(ngram_range=(1, 3))
    check_cv(
        cv,
        input=["abcef"],
        vocab=["a", "ab", "abc", "b", "bc", "bce", "c", "ce", "cef", "e", "ef", "f"],
        output=lil_matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).tocsr(),
    )

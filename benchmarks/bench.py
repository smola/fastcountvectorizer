from sklearn.datasets import fetch_20newsgroups

from fastcountvectorizer import FastCountVectorizer

newsgroups = fetch_20newsgroups(subset="all")
DOCS = newsgroups.data[:2000]


class Suite:

    param_names = ["analyzer", "ngram_range"]

    params = (
        ("char", "word"),
        (
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 4),
        ),
    )

    def setup(self, analyzer, ngram_range):
        cv = FastCountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        self.fitted_cv = cv.fit(DOCS)

    def time_fit(self, analyzer, ngram_range):
        cv = FastCountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        cv.fit(DOCS)

    def time_transform(self, analyzer, ngram_range):
        self.fitted_cv.transform(DOCS)

    def peakmem_fit(self, analyzer, ngram_range):
        cv = FastCountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        cv.fit(DOCS)

    def peakmem_transform(self, analyzer, ngram_range):
        self.fitted_cv.transform(DOCS)

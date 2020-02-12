import gc
from timeit import Timer

from memory_profiler import memory_usage
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from fastcountvectorizer import FastCountVectorizer


def run_countvectorizer_fit(ngram_range):
    cv = CountVectorizer(analyzer="char", ngram_range=ngram_range)
    cv.fit(docs)


def run_fastcountvectorizer_fit(ngram_range):
    cv = FastCountVectorizer(ngram_range=ngram_range)
    cv.fit(docs)


docs = None


def main():
    global docs
    print("Fetching dataset...")
    newsgroups = fetch_20newsgroups(subset="all")
    docs = newsgroups.data
    for max_ngram in range(1, 5):
        print("Running CountVectorizer(ngram_range=(1, %d))..." % max_ngram)
        t = Timer(
            "run_countvectorizer_fit(ngram_range=(1, %d))" % max_ngram,
            setup="from __main__ import docs, run_countvectorizer_fit",
        ).timeit(3)
        t /= 3
        print("Time: %fs" % t)
        gc.collect()
        mem_usage = memory_usage(
            lambda: run_countvectorizer_fit(ngram_range=(1, max_ngram))
        )
        print("Max memory: %fmb" % max(mem_usage))
        print("Running FastCountVectorizer(ngram_range=(1, %d))..." % max_ngram)
        t = Timer(
            "run_fastcountvectorizer_fit(ngram_range=(1, %d))" % max_ngram,
            setup="from __main__ import docs, run_fastcountvectorizer_fit",
        ).timeit(3)
        t /= 3
        print("Time: %fs" % t)
        gc.collect()
        mem_usage = memory_usage(
            lambda: run_fastcountvectorizer_fit(ngram_range=(1, max_ngram))
        )
        print("Max memory: %fmb" % max(mem_usage))


if __name__ == "__main__":
    main()

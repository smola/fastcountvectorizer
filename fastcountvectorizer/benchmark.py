from fastcountvectorizer import FastCountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from memory_profiler import memory_usage
from timeit import Timer
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run(docs, Vectorizer, ngram_range):
    cv = Vectorizer(analyzer="char", ngram_range=ngram_range)
    cv.fit(docs)

docs = None

def run_benchmark(n_docs, Vectorizer, ngram_range):
    print("Running %s(ngram_range=%s)..." % (Vectorizer.__name__, ngram_range))
    docs_subset = docs[:n_docs]
    gc.collect()
    t = Timer(
        "run(docs, Vectorizer, ngram_range)",
        globals={
            'run': run,
            'docs': docs_subset,
            'Vectorizer': Vectorizer,
            'ngram_range': ngram_range,
        }
    ).repeat(repeat=3, number=1)
    t = min(t)
    print("Time: %fs" % t)

    gc.collect()
    mem_usage = memory_usage(lambda: run(docs_subset, Vectorizer, ngram_range))
    mem_usage = max(mem_usage)
    print("Max memory: %fmb" % mem_usage)

    return {
        'n_docs': n_docs,
        'impl': Vectorizer.__name__,
        'min_n': ngram_range[0],
        'max_n': ngram_range[1],
        'time': t,
        'mem': mem_usage,
        }

def run_benchmarks():
    global docs
    print("Fetching dataset...")
    newsgroups = fetch_20newsgroups(subset="all")
    docs = newsgroups.data
    results = []
    for n_docs in (10, 100, 1000, 10000):
        for Vectorizer in (FastCountVectorizer, CountVectorizer):
            for max_ngram in range(1, 5):
                results.append(run_benchmark(n_docs, Vectorizer, (1, max_ngram)))
                if max_ngram != 1:
                    results.append(run_benchmark(n_docs, Vectorizer, (max_ngram, max_ngram)))
    return results

def main():
    result = run_benchmarks()
    df = pd.DataFrame(result)
    df['ngram_range'] = df[['min_n', 'max_n']].apply(lambda r: '(%d, %d)' % (r[0], r[1]), axis=1)
    #%matplotlib gtk
    #df.query('min_n == 1 and max_n == 4').pivot(index='n_docs', columns='impl', values='time').plot.bar(stacked=False, logy=True)

if __name__ == "__main__":
    main()

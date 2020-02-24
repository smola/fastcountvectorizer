
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "_analyzers.h"
#include "_collections.h"
#include "_counters.h"
#include "_csr.h"
#include "_sputils.h"
#include "_strings.h"

namespace py = pybind11;

CharNgramCounter::CharNgramCounter(const std::string& a,
                                   const unsigned int min_n,
                                   const unsigned int max_n,
                                   py::object fixed_vocab,
                                   py::object stop_words)
    : min_n(min_n),
      max_n(max_n),
      fixed_vocab(fixed_vocab),
      analyzer(ngram_analyzer::make(a, stop_words)) {
  result_array_len = 0;
  if (need_expand_counts()) {
    prefixes = new std::vector<string_with_kind>();
  } else {
    prefixes = nullptr;
  }
  values = new std::vector<std::int64_t>();
  indices = new index_vector();
  indptr = new index_vector();
  indptr->push_back(0);
}

CharNgramCounter::~CharNgramCounter() {
  delete analyzer;
  delete prefixes;
  delete values;
  delete indices;
  delete indptr;
}

void CharNgramCounter::process(const py::str& obj) {
  if (have_fixed_focab()) {
    const py::dict fixed_vocab_dict =
        py::reinterpret_borrow<py::dict>(fixed_vocab);
    indices->set_max_value(fixed_vocab_dict.size());
    indptr->set_max_value(fixed_vocab_dict.size());

    for (unsigned int n = min_n; n <= max_n; n++) {
      ngram_analysis_counts* counts = analyzer->analyze(n, obj);

      const py::dict fixed_vocab_dict =
          py::reinterpret_borrow<py::dict>(fixed_vocab);

      while (counts->next()) {
        const py::str key = counts->pykey();
        if (!fixed_vocab_dict.contains(key)) {
          continue;
        }

        const std::size_t term_idx = py::cast<py::int_>(fixed_vocab_dict[key]);
        result_array_len++;
        indices->set_max_value(result_array_len);
        indices->push_back(term_idx);
        values->push_back(counts->count());
      }

      delete counts;
    }

    indptr->set_max_value(result_array_len);
    indptr->push_back(result_array_len);

  } else {
    const unsigned int n = max_n;

    if (need_expand_counts()) {
      prefixes->push_back(analyzer->prefix(n - 1, obj));
    }

    ngram_analysis_counts* counts = analyzer->analyze(n, obj);

    result_array_len += counts->size();
    values->reserve(counts->size());
    indices->set_max_value({vocab.size(), result_array_len});
    indices->reserve(counts->size());
    indptr->set_max_value({vocab.size(), result_array_len});
    indptr->push_back(result_array_len);

    while (counts->next()) {
      const std::size_t term_idx = vocab[counts->key()];
      indices->push_back(term_idx);
      values->push_back(counts->count());
    }

    delete counts;
  }
}

bool CharNgramCounter::have_fixed_focab() const {
  return !fixed_vocab.is_none();
}

bool CharNgramCounter::need_expand_counts() const {
  return !have_fixed_focab() && min_n < max_n;
}

bool vocab_idx_less_than(const std::pair<string_with_kind, size_t>& a,
                         const std::pair<string_with_kind, size_t>& b) {
  return a.second < b.second;
}

void count_expansion_csr_matrix(ngram_analyzer* analyzer, vocab_map& vocab,
                                std::vector<std::intptr_t>& conv_indices,
                                const unsigned int min_n,
                                const unsigned int max_n) {
  // copy vocab (and sort) for iteration concurrent with modification
  std::vector<std::pair<string_with_kind, size_t>> vocab_copy =
      vocab.to_vector();

  // sort is required omit storing old term index when computing the conversion
  // matrix (see below).
  std::sort(vocab_copy.begin(), vocab_copy.end(), vocab_idx_less_than);

  // compute conversion matrix (_, conv_indices, _) in CSR format.
  // actual values are omitted, since they are always 1.
  // indptr is also omitted since it is always in increments of max_n-min_n
  conv_indices.resize(vocab.size() * (size_t)(max_n - min_n));
  size_t i_indices = 0;
  for (auto it = vocab_copy.begin(); it != vocab_copy.end(); it++) {
    string_with_kind new_term = it->first;
    for (unsigned int n = max_n - 1; n >= min_n; n--) {
      new_term = analyzer->suffix(new_term);
      const size_t term_idx = vocab[new_term];
      assert(term_idx >= vocab_copy.size());
      conv_indices[i_indices++] = (std::intptr_t)term_idx;
    }
  }
}

template <class I>
void prefixes_add_csr_matrix(ngram_analyzer* analyzer, vocab_map& vocab,
                             const std::vector<string_with_kind>& prefixes,
                             std::vector<I>& prefixes_indptr,
                             std::vector<I>& prefixes_indices,
                             const unsigned int min_n,
                             const unsigned int max_n) {
  prefixes_indices.reserve(prefixes.size() * (max_n - min_n));
  prefixes_indptr.resize(prefixes.size() + 1);
  prefixes_indptr[0] = 0;
  for (unsigned int i = 0; i < prefixes.size(); i++) {
    std::vector<string_with_kind> prefix_ngrams =
        analyzer->prefix_ngrams(prefixes[i], min_n, max_n - 1);
    for (auto it = prefix_ngrams.cbegin(); it != prefix_ngrams.cend(); it++) {
      prefixes_indices.push_back((I)vocab[*it]);
    }
    prefixes_indptr[i + 1] = (I)prefixes_indices.size();
  }
}

void CharNgramCounter::expand_counts() {
  if (!need_expand_counts()) {
    return;
  }

  if (vocab.size() > std::numeric_limits<std::intptr_t>::max()) {
    throw std::overflow_error("too many vocabulary terms");
  }

  // compute conversion matrix (_, conv_indices, _) in CSR format.
  // actual values are omitted, since they are always 1.
  // indptr is also omitted since it is always in increments of max_n-min_n
  std::vector<std::intptr_t> conv_indices;
  count_expansion_csr_matrix(analyzer, vocab, conv_indices, min_n, max_n);

  // compute CSR matrix for prefixes, data is always 1
  std::vector<std::intptr_t> prefixes_indices;
  std::vector<std::intptr_t> prefixes_indptr;
  prefixes_add_csr_matrix(analyzer, vocab, *prefixes, prefixes_indptr,
                          prefixes_indices, min_n, max_n);
  delete prefixes;
  prefixes = nullptr;

  // final matrix shape
  const std::size_t n_row = indptr->size() - 1;
  const std::size_t n_col = vocab.size();
  const auto nnz_per_B_row = (std::size_t)(max_n - min_n);

  auto new_indptr = new index_vector();
  auto new_indices = new index_vector();
  auto new_values = new std::vector<std::int64_t>();

  csr_matmat_add_Bx1_diagprefix_fixed_nnz(
      n_row, n_col, *indptr, *indices, *values, conv_indices, nnz_per_B_row,
      prefixes_indptr, prefixes_indices, *new_indptr, *new_indices,
      *new_values);

  std::swap(indptr, new_indptr);
  delete new_indptr;
  std::swap(indices, new_indices);
  delete new_indices;
  std::swap(values, new_values);
  delete new_values;
}

std::vector<std::size_t> CharNgramCounter::document_frequencies() const {
  std::vector<std::size_t> docfreq(vocab.size(), 0);
  for (std::size_t i = 0; i < indptr->size() - 1; i++) {
    for (auto j = (std::size_t)(*indptr)[i]; j < (size_t)(*indptr)[i + 1];
         j++) {
      const auto idx = (std::size_t)(*indices)[j];
      docfreq[idx]++;
    }
  }
  return docfreq;
}

template <class K, class V>
struct _kv_less_k {
  bool operator()(const std::pair<K, V>& lhs,
                  const std::pair<K, V>& rhs) const {
    return lhs.first < rhs.first;
  }
};

py::set CharNgramCounter::limit_features(const std::size_t min_df,
                                         const std::size_t max_df) {
  py::set stop_words;
  std::vector<std::int64_t> new_vocab_indices(vocab.size(), -1);
  std::vector<std::size_t> docfreq = document_frequencies();
  std::vector<std::pair<string_with_kind, std::size_t>> vocab_copy =
      vocab.to_vector();

  std::sort(vocab_copy.begin(), vocab_copy.end(),
            _kv_less_k<string_with_kind, size_t>());

  std::size_t new_index = 0;
  for (auto it = vocab_copy.begin(); it != vocab_copy.end(); it++) {
    const std::size_t old_idx = it->second;
    const std::size_t f = docfreq[old_idx];
    if (f >= min_df && f <= max_df) {
      vocab.set_index(it->first, new_index);
      new_vocab_indices[old_idx] = (std::int64_t)new_index;
      new_index++;
    } else {
      py::str pystr = static_cast<py::str>(it->first);
      stop_words.add(pystr);
      vocab.erase(it->first);
    }
  }

  auto new_indices = new index_vector();
  auto new_indptr = new index_vector();
  auto new_values = new std::vector<std::int64_t>();
  transform_indices(vocab.size(), new_vocab_indices, *indptr, *indices, *values,
                    *new_indptr, *new_indices, *new_values);
  std::swap(indices, new_indices);
  delete new_indices;
  std::swap(indptr, new_indptr);
  delete new_indptr;
  std::swap(values, new_values);
  delete new_values;

  return stop_words;
}

void CharNgramCounter::sort_features() {
  std::vector<std::int64_t> new_vocab_indices(vocab.size(), -1);
  std::vector<std::pair<string_with_kind, size_t>> vocab_copy =
      vocab.to_vector();

  std::sort(vocab_copy.begin(), vocab_copy.end(),
            _kv_less_k<string_with_kind, size_t>());

  size_t new_index = 0;
  for (auto it = vocab_copy.begin(); it != vocab_copy.end(); it++) {
    const size_t old_idx = it->second;
    vocab.set_index(it->first, new_index);
    new_vocab_indices[old_idx] = (std::int64_t)new_index;
    new_index++;
  }

  transform_indices(vocab.size(), new_vocab_indices, *indptr, *indices, *values,
                    *indptr, *indices, *values);
}

py::array CharNgramCounter::get_values() {
  assert(values != nullptr);
  py::array v = py::array_t<std::int64_t>(values->size(), values->data());
  delete values;
  values = nullptr;
  return v;
}

py::array CharNgramCounter::get_indices() {
  assert(indices != nullptr);
  py::array v = indices->to_numpy();
  delete indices;
  indices = nullptr;
  return v;
}

py::array CharNgramCounter::get_indptr() {
  assert(indptr != nullptr);
  py::array v = indptr->to_numpy();
  delete indptr;
  indptr = nullptr;
  return v;
}

py::tuple CharNgramCounter::get_result() {
  return py::make_tuple(get_values(), get_indices(), get_indptr());
}

py::dict CharNgramCounter::get_vocab() { return vocab.flush_to_pydict(); }

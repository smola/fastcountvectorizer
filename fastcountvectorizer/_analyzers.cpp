
#include <pybind11/pybind11.h>

#include <deque>
#include <stdexcept>
#include <vector>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "_analyzers.h"

namespace py = pybind11;

ngram_analyzer* ngram_analyzer::make(const std::string& type) {
  if (type == "char") {
    return new char_ngram_analyzer();
  } else {
    throw std::invalid_argument("invalid analyzer type");
  }
}

string_with_kind char_ngram_prefix_handler::prefix(const unsigned int n,
                                                   const py::str& doc) const {
  const char* data = (char*)PyUnicode_1BYTE_DATA(doc.ptr());
  const auto kind = (uint8_t)PyUnicode_KIND(doc.ptr());
  const auto len = PyUnicode_GET_LENGTH(doc.ptr());
  const unsigned int prefix_len = std::min((unsigned int)len, n);
  return string_with_kind::compact(data, prefix_len * kind, kind);
}

string_with_kind char_ngram_prefix_handler::suffix(
    const string_with_kind& s) const {
  return s.suffix();
}

std::vector<string_with_kind> char_ngram_prefix_handler::prefix_ngrams(
    const string_with_kind& s, const unsigned int min_n,
    const unsigned int max_n) const {
  std::vector<string_with_kind> ngrams;
  const std::uint8_t kind = s.kind();
  const unsigned int prefix_len = (unsigned int)s.size() / kind;
  if (prefix_len < min_n) {
    return ngrams;
  }
  const unsigned int new_max_n = std::min(prefix_len, max_n);
  for (unsigned int n = min_n; n <= new_max_n; n++) {
    const unsigned int end = new_max_n - n + 1;
    for (unsigned int start = 0; start < end; start++) {
      string_with_kind ngram =
          string_with_kind::compact(s.data() + start * kind, n * kind, kind);
      ngrams.push_back(ngram);
    }
  }
  return ngrams;
}

char_ngram_analysis_counts::char_ngram_analysis_counts(const unsigned int n,
                                                       const py::str& doc,
                                                       const uint8_t kind)
    : base_ngram_analysis_counts<counter_map>(counter_map(n * kind)),
      n(n),
      kind(kind) {
  const char* data = (char*)PyUnicode_1BYTE_DATA(doc.ptr());
  const auto len = PyUnicode_GET_LENGTH(doc.ptr());
  const auto byte_len = (std::size_t)len * kind;

  for (std::size_t i = 0; i <= byte_len - n * kind; i += kind) {
    const char* data_ptr = data + i;
    counters.increment_key(data_ptr);
  }
}

py::str char_ngram_analysis_counts::pykey() const {
  return py::reinterpret_steal<py::str>(
      PyUnicode_FromKindAndData(kind, it.key(), n));
}

string_with_kind char_ngram_analysis_counts::key() const {
  return string_with_kind::compact(it.key(), (std::size_t)n * kind, kind);
}

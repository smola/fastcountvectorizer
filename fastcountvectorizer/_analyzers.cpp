
#include <pybind11/pybind11.h>

#include <stdexcept>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "_analyzers.h"

namespace py = pybind11;

ngram_analyzer* ngram_analyzer::make(const std::string& type) {
  if (type == "char") {
    return new base_ngram_analyzer<char_ngram_analysis_counts,
                                   char_ngram_prefix_handler>();
  } else if (type == "char_wb") {
    return new base_ngram_analyzer<char_wb_ngram_analysis_counts,
                                   char_wb_ngram_prefix_handler>();
  } else {
    throw std::invalid_argument("invalid analyzer type");
  }
}

string_with_kind char_ngram_prefix_handler::prefix(unsigned int n,
                                                   const py::str& doc) {
  const char* data = (char*)PyUnicode_1BYTE_DATA(doc.ptr());
  const auto kind = (uint8_t)PyUnicode_KIND(doc.ptr());
  const auto len = PyUnicode_GET_LENGTH(doc.ptr());
  const unsigned int prefix_len = std::min((unsigned int)len, n);
  return string_with_kind::compact(data, prefix_len * kind, kind);
}

string_with_kind char_ngram_prefix_handler::suffix(const string_with_kind& s) {
  return s.suffix();
}

string_with_kind char_ngram_prefix_handler::ngram(const string_with_kind& s,
                                                  unsigned int n,
                                                  std::size_t pos) {
  return string_with_kind::compact(s.data() + (pos * s.kind()), (n * s.kind()),
                                   s.kind());
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

char_wb_ngram_analysis_counts::char_wb_ngram_analysis_counts(
    const unsigned int n, const py::str& doc, const uint8_t kind)
    : base_ngram_analysis_counts<string_with_kind_counter_map>(
          string_with_kind_counter_map()) {
  const char* data = (char*)PyUnicode_1BYTE_DATA(doc.ptr());
  const auto len = (std::size_t)PyUnicode_GET_LENGTH(doc.ptr());
  const auto byte_len = (std::size_t)len * kind;

  // first and last characters are guaranteed to be spaces
  auto ws_lower = string_find(data, byte_len, kind, ' ', 0);
  assert(ws_lower == 0);
  auto ws_upper = string_find(data, byte_len, kind, ' ', 1);
  assert(ws_upper != std::string::npos);

  while (ws_upper != std::string::npos) {
    auto word_len = ws_upper - ws_lower;
    if (word_len <= n) {
      counters.increment_key(string_with_kind::compact(data + (ws_lower * kind),
                                                       word_len * kind, kind));
    } else {
      for (std::size_t i = ws_lower; i <= ws_upper - n + 1; i++) {
        counters.increment_key(
            string_with_kind::compact(data + (i * kind), n * kind, kind));
      }
    }
    ws_lower = ws_upper;
    ws_upper = string_find(data, byte_len, kind, ' ', ws_upper + 1);
  }
}

py::str char_wb_ngram_analysis_counts::pykey() const {
  return it.key().toPyObject();
}

string_with_kind char_wb_ngram_analysis_counts::key() const { return it.key(); }

string_with_kind char_wb_ngram_prefix_handler::prefix(unsigned int n,
                                                      const py::str& doc) {
  const char* data = (char*)PyUnicode_1BYTE_DATA(doc.ptr());
  const auto kind = (uint8_t)PyUnicode_KIND(doc.ptr());
  const auto len = (std::size_t)PyUnicode_GET_LENGTH(doc.ptr());

  const auto next_whitespace = string_find(data, len * kind, kind, ' ', 1);
  assert(next_whitespace != std::string::npos);

  const auto prefix_len = std::min(next_whitespace, (std::size_t)n);
  return string_with_kind::compact(data, prefix_len * kind, kind);
}

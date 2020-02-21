
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
  } else if (type == "word") {
    return new word_ngram_analyzer();
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

py::object make_token_pattern() {
  py::module re = py::module::import("re");
  return re.attr("compile")(R"((?u)\b\w\w+\b)");
}

word_ngram_analyzer::word_ngram_analyzer()
    : re_token_pattern(make_token_pattern()) {}

word_ngram_prefix_handler::word_ngram_prefix_handler()
    : re_token_pattern(make_token_pattern()) {}

word_ngram_analysis_counts::word_ngram_analysis_counts(
    unsigned int n, const py::str& doc, uint8_t Py_UNUSED(kind),
    const py::object& token_pattern)
    : base_ngram_analysis_counts<string_with_kind_counter_map>(
          string_with_kind_counter_map()) {
  std::deque<string_with_kind> token_queue;
  py::object re_finditer = token_pattern.attr("finditer");
  for (auto it = py::cast<py::iterator>(re_finditer(doc));
       it != py::iterator::sentinel(); ++it) {
    py::str token = it->attr("group")().cast<py::str>();
    token_queue.push_back(static_cast<string_with_kind>(token));
    if (token_queue.size() > n) {
      token_queue.pop_front();
    }
    if (token_queue.size() == n) {
      string_with_kind key = string_with_kind::join(
          token_queue.cbegin(), token_queue.cend(), token_queue.size());
      counters.increment_key(key);
    }
  }
}

py::str word_ngram_analysis_counts::pykey() const {
  return static_cast<py::str>(it.key());
}

string_with_kind word_ngram_analysis_counts::key() const { return it.key(); }

string_with_kind word_ngram_prefix_handler::prefix(unsigned int n,
                                                   const py::str& doc) const {
  if (n == 0 || PyUnicode_GET_LENGTH(doc.ptr()) == 0) {
    return string_with_kind("", 0, 1);
  }
  std::vector<string_with_kind> token_queue;
  token_queue.reserve(n);
  py::object re_finditer = re_token_pattern.attr("finditer");
  for (py::iterator it = re_finditer(doc);
       it != py::iterator::sentinel() && token_queue.size() < n; ++it) {
    py::str token = it->attr("group")();
    token_queue.push_back(static_cast<string_with_kind>(token));
  }
  return string_with_kind::join(token_queue.cbegin(), token_queue.cend(),
                                token_queue.size());
}

string_with_kind word_ngram_prefix_handler::suffix(
    const string_with_kind& s) const {
  if (s.empty()) {
    return string_with_kind("", 0, 1);
  }

  const auto cp_ws_lower = string_find(s, ' ', 0);
  assert(cp_ws_lower != 0);

  if (cp_ws_lower == std::string::npos) {
    return string_with_kind("", 0, 1);
  }

  const auto byte_offset = (cp_ws_lower + 1) * s.kind();
  const auto byte_len = s.size() - byte_offset;
  return string_with_kind::compact(s.data() + byte_offset, byte_len, s.kind());
}

std::vector<string_with_kind> word_ngram_prefix_handler::prefix_ngrams(
    const string_with_kind& s, const unsigned int min_n,
    const unsigned int max_n) const {
  std::vector<string_with_kind> ngrams;
  if (s.empty()) {
    return ngrams;
  }

  std::vector<string_with_kind> tokens;
  tokens.reserve(max_n);

  auto cp_ws_lower = string_find(s, ' ', 0);
  assert(cp_ws_lower != 0);

  if (cp_ws_lower == std::string::npos) {
    tokens.emplace_back(s);
  } else {
    tokens.emplace_back(
        string_with_kind::compact(s.data(), cp_ws_lower * s.kind(), s.kind()));
    while (cp_ws_lower != std::string::npos) {
      const auto cp_ws_upper = string_find(s, ' ', cp_ws_lower + 1);
      const auto cp_until = std::min(cp_ws_upper, s.size() / s.kind() - 1);
      const auto cp_len = cp_until - cp_ws_lower;
      const auto byte_len = cp_len * s.kind();
      const auto byte_offset = (cp_ws_lower + 1) * s.kind();
      tokens.emplace_back(string_with_kind::compact(s.data() + byte_offset,
                                                    byte_len, s.kind()));
      cp_ws_lower = cp_ws_upper;
    }
  }

  std::size_t start = 0;
  for (auto it = tokens.cbegin(); it != tokens.cend(); it++, start++) {
    const auto max_ngram_size =
        std::min((unsigned int)(tokens.size() - start), max_n);
    for (auto n = min_n; n <= max_ngram_size; n++) {
      ngrams.push_back(string_with_kind::join(it, tokens.cend(), n));
    }
  }

  return ngrams;
}

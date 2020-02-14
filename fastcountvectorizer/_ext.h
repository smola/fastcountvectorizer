
#ifndef FCV_EXT_H
#define FCV_EXT_H

#include <cstdint>
#include <vector>

#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>

#include "_sputils.h"
#include "_strings.h"
#include "tsl/sparse_map.h"

namespace py = pybind11;

class vocab_map {
 private:
  tsl::sparse_map<string_with_kind, size_t> _m;

 public:
  size_t operator[](const string_with_kind& k);
  py::dict flush_to_pydict();
  std::vector<std::pair<string_with_kind, size_t>> to_vector() const;
  size_t size() const { return _m.size(); }
  void erase(const string_with_kind& k) { _m.erase(k); }
  void set_index(const string_with_kind& k, const size_t v) { _m[k] = v; }
};

class counter_map : public tsl::sparse_map<const char*, std::int64_t,
                                           fixed_length_string_hash,
                                           fixed_length_string_equal_to> {
 public:
  explicit counter_map(const size_t str_length)
      : tsl::sparse_map<const char*, std::int64_t, fixed_length_string_hash,
                        fixed_length_string_equal_to>(
            0, fixed_length_string_hash(str_length),
            fixed_length_string_equal_to(str_length)) {}
  void increment_key(const char* k);
};

class CharNgramCounter {
 private:
  vocab_map vocab;
  unsigned int min_n;
  unsigned int max_n;

  std::size_t result_array_len;
  std::vector<string_with_kind>* prefixes;
  std::vector<std::int64_t>* values;
  index_vector* indices;
  index_vector* indptr;

  void prepare_vocab();
  bool need_expand_counts() const;
  std::vector<size_t> document_frequencies() const;

 public:
  CharNgramCounter(unsigned int min_n, unsigned int max_n);
  ~CharNgramCounter();

  void process(const py::str& obj);
  void expand_counts();
  py::set limit_features(std::size_t min_df, std::size_t max_df);
  py::array get_values();
  py::array get_indices();
  py::array get_indptr();
  py::tuple get_result();
  py::dict get_vocab();
};

#endif  // FCV_EXT_H
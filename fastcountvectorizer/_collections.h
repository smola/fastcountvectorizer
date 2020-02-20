
#ifndef FCV_COLLECTIONS_H
#define FCV_COLLECTIONS_H

#include <pybind11/pybind11.h>

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

#endif  // FCV_COLLECTIONS_H
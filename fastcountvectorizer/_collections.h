
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

template <class Key, class Hash = std::hash<Key>,
          class EqualTo = std::equal_to<Key>>
class base_counter_map
    : public tsl::sparse_map<Key, std::int64_t, Hash, EqualTo> {
 public:
  typedef tsl::sparse_map<Key, std::int64_t, Hash, EqualTo> base_map;

  base_counter_map() = default;
  base_counter_map(Hash hash, EqualTo equal) : base_map(0, hash, equal) {}

  using base_map::end;
  using base_map::find;
  using base_map::insert;

  void increment_key(const Key& k) {
    auto it = find(k);
    if (it == end()) {
      insert({k, 1});
    } else {
      it.value()++;
    }
  }
};

class counter_map
    : public base_counter_map<const char*, fixed_length_string_hash,
                              fixed_length_string_equal_to> {
 public:
  explicit counter_map(const size_t str_length)
      : base_counter_map<const char*, fixed_length_string_hash,
                         fixed_length_string_equal_to>(
            fixed_length_string_hash(str_length),
            fixed_length_string_equal_to(str_length)) {}
};

class string_with_kind_counter_map : public base_counter_map<string_with_kind> {

};

#endif  // FCV_COLLECTIONS_H
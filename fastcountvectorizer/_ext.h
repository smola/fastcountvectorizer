
#ifndef FCV_EXT_H
#define FCV_EXT_H

#include <cstdint>
#include <vector>

#include "Python.h"
#include "_sputils.h"
#include "_strings.h"
#include "tsl/sparse_map.h"

class vocab_map {
 private:
  tsl::sparse_map<string_with_kind, size_t> _m;

 public:
  size_t operator[](const string_with_kind& k);
  int flush_to(PyObject* dest_dict);
  size_t size() const { return _m.size(); }
};

class counter_map
    : public tsl::sparse_map<const char*, npy_int64, fixed_length_string_hash,
                             fixed_length_string_equal_to> {
 public:
  explicit counter_map(const size_t str_length)
      : tsl::sparse_map<const char*, npy_int64, fixed_length_string_hash,
                        fixed_length_string_equal_to>(
            0, fixed_length_string_hash(str_length),
            fixed_length_string_equal_to(str_length)) {}
  void increment_key(const char* k);
};

class CharNgramCounter {
 private:
  vocab_map vocab;
  const unsigned int n;

  size_t result_array_len;
  std::vector<npy_int64>* values;
  index_vector* indices;
  index_vector* indptr;

  void prepare_vocab();

 public:
  CharNgramCounter(const unsigned int n);
  ~CharNgramCounter();

  void process_one(PyUnicodeObject* obj);
  PyObject* get_values();
  PyObject* get_indices();
  PyObject* get_indptr();
  int copy_vocab(PyObject* dest_vocab);
};

#endif  // FCV_EXT_H
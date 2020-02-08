
#ifndef FCV_EXT_H
#define FCV_EXT_H

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "Python.h"

class string_with_kind : public std::string {
 private:
  uint8_t _kind;

 public:
  string_with_kind(const char* str, const size_t size, const uint8_t kind)
      : std::string(str, size), _kind(kind) {}
  uint8_t kind() const { return _kind; }

  string_with_kind compact() const;
  bool operator==(const string_with_kind& other) const;
  bool operator!=(const string_with_kind& other) const;
};

namespace std {
template <>
struct hash<string_with_kind> {
  size_t operator()(const string_with_kind& k) const {
    return hash<string>()(k);
  }
};
}  // namespace std

PyObject* to_PyObject(const string_with_kind& str);

class vocab_map {
 private:
  std::unordered_map<string_with_kind, size_t> _m;

 public:
  size_t operator[](const string_with_kind& k);
  int flush_to(PyObject* dest_dict);
  size_t size() const { return _m.size(); }
};

typedef std::unordered_map<string_with_kind, size_t> counter_map;

class CharNgramCounter {
 private:
  vocab_map vocab;
  const unsigned int n;

  size_t result_array_len;
  std::vector<size_t>* values;
  std::vector<size_t>* indices;
  std::vector<size_t>* indptr;

  void prepare_vocab();
  static PyObject* _vector_to_numpy(const std::vector<size_t>* v);

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
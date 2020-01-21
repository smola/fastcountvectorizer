
#ifndef FCV_EXT_H
#define FCV_EXT_H

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "Python.h"

struct FullLightStringHash; /* Forward */
struct LightStringHash;     /* Forward */
struct LightStringEqual;    /* Forward */

class FullLightString {
 private:
  char* _data;
  size_t _byte_len;
  size_t _hash;
  unsigned char _kind;
  bool _owned;

  char* copy_data() const;

 public:
  FullLightString(char* data, const size_t byte_len, const unsigned char kind,
                  const size_t hash)
      : _data(data),
        _byte_len(byte_len),
        _hash(hash),
        _kind(kind),
        _owned(false) {}

  FullLightString() : FullLightString(NULL, 0, PyUnicode_1BYTE_KIND, 0) {}

  void own();
  void free();
  PyObject* toPyObject() const;

  bool operator==(const FullLightString& other) const;

  friend FullLightStringHash;
};

struct FullLightStringHash {
  std::size_t operator()(const FullLightString& k) const noexcept;
};

class LightString {
 private:
  char* _data;
  size_t _hash;

 public:
  LightString(char* data, const size_t hash) : _data(data), _hash(hash) {}

  LightString() : LightString(NULL, 0) {}

  FullLightString to_full(const size_t byte_len,
                          const unsigned char kind) const;

  friend LightStringHash;
  friend LightStringEqual;
};

struct LightStringHash {
  size_t operator()(const LightString& k) const noexcept;
};

struct LightStringEqual {
 private:
  size_t _len;

 public:
  LightStringEqual() : _len(0) {}

  LightStringEqual(size_t len) : _len(len) {}

  bool operator()(const LightString& lhs, const LightString& rhs) const;
};

typedef std::unordered_map<FullLightString, int, FullLightStringHash> vocab_map;

typedef std::unordered_map<LightString, int, LightStringHash, LightStringEqual>
    counter_map;

class CharNgramCounter {
 private:
  vocab_map vocab;
  const int n;

  size_t result_array_len;
  std::vector<size_t>* values;
  std::vector<size_t>* indices;
  std::vector<size_t>* indptr;

  void prepare_vocab();
  static PyObject* _vector_to_numpy(const std::vector<size_t>* v);

 public:
  CharNgramCounter(const int n);
  ~CharNgramCounter();

  void process_one(PyUnicodeObject* obj);
  PyObject* get_values();
  PyObject* get_indices();
  PyObject* get_indptr();
  int copy_vocab(PyObject* dest_vocab);
};

#endif  // FCV_EXT_H
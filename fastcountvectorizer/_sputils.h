
#ifndef FCV_SPUTILS_H
#define FCV_SPUTILS_H

#include <cstdint>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL fcv_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarraytypes.h>

// vector_to_numpy converts a std::vector to a numpy.array.
template <typename I>
PyObject* vector_to_numpy(const std::vector<I>& v);

// index_vector is an integer vector that uses initially int32_t and switches
// dynamically to int32_t when needed. This is useful for arrays to be used as
// scipy.sparse indexes, which will use int32 when possible and will trigger an
// index copy if int64 is used when the maximum number fits in an int32.
//
// Limits come from numpy, not cstdint.
class index_vector {
 private:
  std::vector<npy_int32>* v32;
  std::vector<npy_int64>* v64;
  bool use_64;

  explicit index_vector(bool use_64);

 public:
  index_vector();
  ~index_vector();
  void set_max_value(size_t val);
  void set_max_value(const std::vector<size_t>& vals);
  void reserve(size_t n);
  void push_back(size_t n);
  size_t size() const;
  PyObject* to_numpy() const;
  bool is_64() const { return use_64; }
  std::vector<npy_int32>& data32() { return *v32; }
  const std::vector<npy_int32>& data32() const { return *v32; }
  std::vector<npy_int64>& data64() { return *v64; }
  const std::vector<npy_int64>& data64() const { return *v64; }
  npy_int64 operator[](size_t i) const;
};

#endif  // FCV_SPUTILS_H
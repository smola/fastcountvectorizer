
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL fcv_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include "_sputils.h"

template <typename I>
PyObject* _vector_to_numpy(const std::vector<I>& v, const int typenum) {
  PyObject* a;
  const size_t size = v.size();
  npy_intp shape[1];
  shape[0] = (npy_intp)size;
  a = PyArray_SimpleNew(1, shape, typenum);
  memcpy(PyArray_DATA((PyArrayObject*)a), v.data(), size * sizeof(I));
  return a;
}

template <>
PyObject* vector_to_numpy<npy_int32>(const std::vector<npy_int32>& v) {
  return _vector_to_numpy<npy_int32>(v, NPY_INT32);
}

template <>
PyObject* vector_to_numpy<npy_int64>(const std::vector<npy_int64>& v) {
  return _vector_to_numpy<npy_int64>(v, NPY_INT64);
}

bool needs_npy_int64(const size_t val) { return val > NPY_MAX_INT32; }

index_vector::index_vector(const bool use_64) : use_64(use_64) {
  v32 = new std::vector<npy_int32>();
  v64 = nullptr;
}

index_vector::index_vector() : index_vector(false) {}

index_vector::~index_vector() {
  delete v32;
  delete v64;
}

void index_vector::set_max_value(size_t val) {
  if (!use_64 && needs_npy_int64(val)) {
    use_64 = true;
    v64 = new std::vector<npy_int64>(v32->size());
    for (size_t i = 0; i < v32->size(); i++) {
      (*v64)[i] = (*v32)[i];
    }
    delete v32;
    v32 = nullptr;
  }
}

void index_vector::set_max_value(const std::vector<size_t>& vals) {
  for (auto it = vals.begin(); it != vals.end(); it++) {
    set_max_value(*it);
  }
}

void index_vector::reserve(const size_t n) {
  if (use_64) {
    v64->reserve(n);
  } else {
    v32->reserve(n);
  }
}

void index_vector::push_back(const size_t n) {
  if (use_64) {
    if (n > NPY_MAX_INT64) {
      throw std::overflow_error("too many elements");
    }
    v64->push_back((npy_int64)n);
  } else {
    assert(n <= NPY_MAX_INT32);
    v32->push_back((npy_int32)n);
  }
}

size_t index_vector::size() const {
  if (use_64) {
    return v64->size();
  } else {
    return v32->size();
  }
}

PyObject* index_vector::to_numpy() const {
  if (use_64) {
    return vector_to_numpy(*v64);
  } else {
    return vector_to_numpy(*v32);
  }
}

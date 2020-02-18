
#include <limits>
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL fcv_ARRAY_API
#include <numpy/arrayobject.h>

#include "_sputils.h"

template <typename I>
PyObject* _vector_to_numpy(const std::vector<I>& v, const int typenum) {
  PyObject* a;
  const std::size_t size = v.size();
  std::intptr_t shape[1];
  shape[0] = (std::intptr_t)size;
  a = PyArray_SimpleNew(1, shape, typenum);
  memcpy(PyArray_DATA((PyArrayObject*)a), v.data(), size * sizeof(I));
  return a;
}

template <>
PyObject* vector_to_numpy<std::int32_t>(const std::vector<std::int32_t>& v) {
  return _vector_to_numpy<std::int32_t>(v, NPY_INT32);
}

template <>
PyObject* vector_to_numpy<std::int64_t>(const std::vector<std::int64_t>& v) {
  return _vector_to_numpy<std::int64_t>(v, NPY_INT64);
}

template <class T>
bool needs_int64(const T val) {
  return val > std::numeric_limits<std::int32_t>::max();
}

index_vector::index_vector(const bool use_64) : use_64(use_64) {
  v32 = new std::vector<std::int32_t>();
  v64 = nullptr;
}

index_vector::index_vector() : index_vector(false) {}

index_vector::~index_vector() {
  delete v32;
  delete v64;
}

void index_vector::set_max_value(size_t val) {
#if SIZEOF_SIZE_T == 8
  if (!use_64 && needs_int64(val)) {
    if (val > std::numeric_limits<std::int64_t>::max()) {
      throw std::overflow_error(
          "too many values: 64 bits indexing not supported on 32 bits "
          "architectures");
    }
    use_64 = true;
    v64 = new std::vector<std::int64_t>(v32->size());
    for (std::size_t i = 0; i < v32->size(); i++) {
      (*v64)[i] = (*v32)[i];
    }
    delete v32;
    v32 = nullptr;
  }
#else
  if (needs_int64(val)) {
    throw std::overflow_error(
        "too many values: 64 bits indexing not supported on 32 bits "
        "architectures");
  }
#endif
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

void index_vector::push_back(const std::size_t n) {
  if (use_64) {
    if (n > std::numeric_limits<std::int64_t>::max()) {
      throw std::overflow_error("too many elements");
    }
    v64->push_back((std::int64_t)n);
  } else {
    assert(n <= std::numeric_limits<std::int32_t>::max());
    v32->push_back((std::int32_t)n);
  }
}

std::size_t index_vector::size() const {
  if (use_64) {
    return v64->size();
  } else {
    return v32->size();
  }
}

std::int64_t index_vector::operator[](const std::size_t i) const {
  if (use_64) {
    return (*v64)[i];
  } else {
    return (std::int64_t)(*v32)[i];
  }
}

PyObject* index_vector::to_numpy() const {
  if (use_64) {
    return vector_to_numpy(*v64);
  } else {
    return vector_to_numpy(*v32);
  }
}

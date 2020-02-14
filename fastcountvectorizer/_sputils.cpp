
#include "_sputils.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

template <class T>
inline bool needs_int64(const T val) {
  return val > std::numeric_limits<std::int32_t>::max();
}

index_vector::index_vector(const bool use_64) : use_64(use_64) {
  if (use_64) {
    v32 = nullptr;
    v64 = new std::vector<std::int64_t>();
  } else {
    v32 = new std::vector<std::int32_t>();
    v64 = nullptr;
  }
}

index_vector::index_vector() : index_vector(false) {}

index_vector::~index_vector() {
  delete v32;
  delete v64;
}

void index_vector::set_max_value(size_t val) {
  if (sizeof(std::size_t) == 8) {
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
  } else {
    if (needs_int64(val)) {
      throw std::overflow_error(
          "too many values: 64 bits indexing not supported on 32 bits "
          "architectures");
    }
  }
}

void index_vector::set_max_value(const std::vector<std::size_t>& vals) {
  for (auto it = vals.begin(); it != vals.end(); it++) {
    set_max_value(*it);
  }
}

void index_vector::reserve(const std::size_t n) {
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

py::array index_vector::to_numpy() const {
  if (use_64) {
    return py::array_t<std::int64_t>(v64->size(), v64->data());
  } else {
    return py::array_t<std::int32_t>(v32->size(), v32->data());
  }
}


#include "_collections.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "_strings.h"

namespace py = pybind11;

size_t vocab_map::operator[](const string_with_kind& k) {
  auto it = _m.find(k);
  size_t idx;
  if (it == _m.end()) {
    idx = _m.size();
    _m[k] = idx;
  } else {
    idx = it->second;
  }
  return idx;
}

py::dict vocab_map::flush_to_pydict() {
  py::dict dest_vocab = py::dict();
  auto it = _m.begin();
  int error = 0;
  while (it != _m.end()) {
    if (error == 0) {
      py::str key = it->first.toPyObject();
      py::int_ value(it->second);
      if (PyDict_SetItem(dest_vocab.ptr(), key.ptr(), value.ptr()) != 0) {
        error = -1;
      }
    };
    it = _m.erase(it);
  }
  return dest_vocab;
}

std::vector<std::pair<string_with_kind, size_t>> vocab_map::to_vector() const {
  std::vector<std::pair<string_with_kind, size_t>> result;
  result.reserve(_m.size());
  for (auto it = _m.begin(); it != _m.end(); it++) {
    result.emplace_back(*it);
  }
  return result;
}

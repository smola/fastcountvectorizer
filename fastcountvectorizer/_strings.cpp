
#include "_strings.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

string_with_kind string_with_kind::compact(const char* str, const size_t size,
                                           const uint8_t kind) {
  if (kind == 1) {
    return string_with_kind(str, size, kind);
  }

  py::str obj = py::reinterpret_steal<py::str>(
      PyUnicode_FromKindAndData(kind, str, (Py_ssize_t)size / kind));
  const auto new_kind = (uint8_t)PyUnicode_KIND(obj.ptr());
  const auto new_byte_len = (size_t)PyUnicode_GET_LENGTH(obj.ptr()) * new_kind;
  string_with_kind result((char*)PyUnicode_1BYTE_DATA(obj.ptr()), new_byte_len,
                          new_kind);
  return result;
}

bool string_with_kind::operator==(const string_with_kind& other) const {
  if (size() != other.size()) {
    return false;
  }
  if (_kind != other._kind) {
    return false;
  }
  return memcmp(data(), other.data(), size()) == 0;
}

bool string_with_kind::operator!=(const string_with_kind& other) const {
  return !operator==(other);
}

py::str string_with_kind::toPyObject() const {
  return py::reinterpret_steal<py::str>(
      PyUnicode_FromKindAndData(kind(), data(), (Py_ssize_t)size() / kind()));
}

string_with_kind string_with_kind::suffix() const {
  return string_with_kind::compact(data() + kind(), size() - kind(), kind());
}
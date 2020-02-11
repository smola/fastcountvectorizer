
#include "_strings.h"

string_with_kind string_with_kind::compact(const char* str, const size_t size,
                                           const uint8_t kind) {
  if (kind == 1) {
    return string_with_kind(str, size, kind);
  }

  PyObject* obj = PyUnicode_FromKindAndData(kind, str, (Py_ssize_t)size / kind);
  const auto new_kind = (uint8_t)PyUnicode_KIND(obj);
  const auto new_byte_len = (size_t)PyUnicode_GET_LENGTH(obj) * new_kind;
  string_with_kind result((char*)PyUnicode_1BYTE_DATA(obj), new_byte_len,
                          new_kind);
  Py_DECREF(obj);
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

PyObject* string_with_kind::toPyObject() const {
  return PyUnicode_FromKindAndData(kind(), data(), (Py_ssize_t)size() / kind());
}

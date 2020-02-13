
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

template <class L, class R>
bool _string_with_kind_less(const string_with_kind& lhs,
                            const string_with_kind& rhs) {
  const size_t lhs_cp_len = lhs.size() / sizeof(L);
  const size_t rhs_cp_len = rhs.size() / sizeof(R);
  const size_t min_cp_len = std::min(lhs_cp_len, rhs_cp_len);
  const L* lhs_data = (L*)lhs.data();
  const R* rhs_data = (R*)rhs.data();

  for (size_t i = 0; i < min_cp_len; i++) {
    const L l = lhs_data[i];
    const R r = rhs_data[i];
    if (l < r) {
      return true;
    }
    if (r > l) {
      return false;
    }
  }

  return lhs_cp_len < rhs_cp_len;
}

template <>
bool _string_with_kind_less<uint8_t, uint8_t>(const string_with_kind& lhs,
                                              const string_with_kind& rhs) {
  const size_t lhs_cp_len = lhs.size();
  const size_t rhs_cp_len = rhs.size();
  const size_t min_cp_len = std::min(lhs_cp_len, rhs_cp_len);

  const int cmp = memcmp(lhs.data(), rhs.data(), min_cp_len);
  if (cmp < 0) {
    return true;
  }

  if (cmp > 0) {
    return false;
  }

  return lhs_cp_len < rhs_cp_len;
}

#if defined(HAVE_WMEMCMP)

#if SIZEOF_WCHAR_T == 2
#define _FCV__WCHAR_T uint16_t
#elif SIZEOF_WCHAR_T == 4
#define _FCV__WCHAR_T uint32_t
#endif

#ifdef _FCV__WCHAR_T
template <>
bool _string_with_kind_less<_FCV__WCHAR_T, _FCV__WCHAR_T>(
    const string_with_kind& lhs, const string_with_kind& rhs) {
  const size_t lhs_cp_len = lhs.size() / sizeof(_FCV__WCHAR_T);
  const size_t rhs_cp_len = rhs.size() / sizeof(_FCV__WCHAR_T);
  const size_t min_cp_len = std::min(lhs_cp_len, rhs_cp_len);

  const int cmp =
      wmemcmp((wchar_t*)lhs.data(), (wchar_t*)rhs.data(), min_cp_len);
  if (cmp < 0) {
    return true;
  }

  if (cmp > 0) {
    return false;
  }

  return lhs_cp_len < rhs_cp_len;
}

#undef _FCV__WCHAR_T
#endif
#endif

bool string_with_kind::operator<(const string_with_kind& other) const {
  if (other.empty()) {
    return false;
  }

  if (empty()) {
    return true;
  }

  const auto lk = _kind;
  const auto rk = other._kind;

  if (lk == 1) {
    if (rk == 1) {
      return _string_with_kind_less<uint8_t, uint8_t>(*this, other);
    } else if (rk == 2) {
      return _string_with_kind_less<uint8_t, uint16_t>(*this, other);
    } else {
      return _string_with_kind_less<uint8_t, uint32_t>(*this, other);
    }
  } else if (lk == 2) {
    if (rk == 1) {
      return _string_with_kind_less<uint16_t, uint8_t>(*this, other);
    } else if (rk == 2) {
      return _string_with_kind_less<uint16_t, uint16_t>(*this, other);
    } else {
      return _string_with_kind_less<uint16_t, uint32_t>(*this, other);
    }
  } else {
    if (rk == 1) {
      return _string_with_kind_less<uint32_t, uint8_t>(*this, other);
    } else if (rk == 2) {
      return _string_with_kind_less<uint32_t, uint16_t>(*this, other);
    } else {
      return _string_with_kind_less<uint32_t, uint32_t>(*this, other);
    }
  }
}

py::str string_with_kind::toPyObject() const {
  return py::reinterpret_steal<py::str>(
      PyUnicode_FromKindAndData(kind(), data(), (Py_ssize_t)size() / kind()));
}

string_with_kind string_with_kind::suffix() const {
  return string_with_kind::compact(data() + kind(), size() - kind(), kind());
}
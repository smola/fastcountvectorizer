
#include "_strings.h"

#include <pybind11/pybind11.h>

#include <deque>
#include <iterator>
#include <vector>

namespace py = pybind11;

string_with_kind::string_with_kind(py::str s)
    : string_with_kind(static_cast<char*>(PyUnicode_DATA(s.ptr())),
                       static_cast<size_t>(PyUnicode_GET_LENGTH(s.ptr()) *
                                           PyUnicode_KIND(s.ptr())),
                       static_cast<uint8_t>(PyUnicode_KIND(s.ptr()))) {}

string_with_kind string_with_kind::compact(const char* str, const size_t size,
                                           const uint8_t kind) {
  if (kind == 1) {
    return string_with_kind(str, size, kind);
  }

  py::str obj = py::reinterpret_steal<py::str>(
      PyUnicode_FromKindAndData(kind, str, static_cast<ssize_t>(size / kind)));
  return static_cast<string_with_kind>(obj);
}

template <class D, class S>
inline void _unicode_copy(void* dst, const void* src, std::size_t cp_len) {
  if (sizeof(D) == sizeof(S)) {
    memcpy(dst, src, cp_len * sizeof(S));
  } else {
    D* dst_typed = (D*)dst;
    S* src_typed = (S*)src;
    for (std::size_t i = 0; i < cp_len; i++) {
      dst_typed[i] = (D)src_typed[i];
    }
  }
}

inline void _unicode_copy(void* dst, const std::uint8_t dst_kind,
                          const void* src, const std::uint8_t src_kind,
                          std::size_t cp_len) {
  assert(dst_kind >= src_kind);
  if (cp_len == 0) {
    return;
  }

  if (dst_kind == 1) {
    _unicode_copy<uint8_t, uint8_t>(dst, src, cp_len);
  } else if (dst_kind == 2) {
    if (src_kind == 1) {
      _unicode_copy<uint16_t, uint8_t>(dst, src, cp_len);
    } else {
      _unicode_copy<uint16_t, uint16_t>(dst, src, cp_len);
    }
  } else {
    if (src_kind == 1) {
      _unicode_copy<uint32_t, uint8_t>(dst, src, cp_len);
    } else if (src_kind == 2) {
      _unicode_copy<uint32_t, uint16_t>(dst, src, cp_len);
    } else {
      _unicode_copy<uint32_t, uint32_t>(dst, src, cp_len);
    }
  }
}

inline void _unicode_set(void* dst, const std::uint8_t kind, std::size_t pos,
                         char c) {
  if (kind == 1) {
    ((uint8_t*)dst)[pos] = (uint8_t)c;
  } else if (kind == 2) {
    ((uint16_t*)dst)[pos] = (uint16_t)c;
  } else {
    ((uint32_t*)dst)[pos] = (uint32_t)c;
  }
}

template <class It>
string_with_kind string_with_kind::join(const It begin, const It end,
                                        const std::size_t size) {
  static_assert(std::is_same<typename std::iterator_traits<It>::value_type,
                             string_with_kind>::value,
                "iterator type must be string_with_kind");
  static_assert(
      std::is_convertible<typename std::iterator_traits<It>::iterator_category,
                          std::forward_iterator_tag>::value,
      "iterator must be forward");

  if (size == 0) {
    return string_with_kind("", 0, 1);
  }

  // calculate total size of the result
  std::size_t cp_len = 0;
  std::uint8_t kind = 1;
  std::size_t i = 0;
  for (It it = It(begin); it != end && i < size; it++, i++) {
    string_with_kind token = *it;
    cp_len += token.size() / token.kind();
    kind = std::max(kind, token.kind());
  }

  // add space for token separators
  cp_len += std::max((std::size_t)0, size - 1);

  // actual concat
  const std::size_t byte_len = cp_len * kind;
  char* buffer = new char[byte_len];
  std::size_t cp_pos = 0;
  i = 0;
  for (It it = It(begin); it != end && i < size; it++, i++) {
    const string_with_kind token = *it;
    const auto token_len = token.size() / token.kind();
    _unicode_copy(buffer + (cp_pos * kind), kind, token.data(), token.kind(),
                  token_len);
    cp_pos += token_len;
    if (cp_pos < cp_len) {
      _unicode_set(buffer, kind, cp_pos, ' ');
    }
    cp_pos++;
  }

  string_with_kind result = string_with_kind(buffer, byte_len, kind);
  delete[] buffer;
  return result;
}

template string_with_kind
    string_with_kind::join<std::deque<string_with_kind>::const_iterator>(
        std::deque<string_with_kind>::const_iterator,
        std::deque<string_with_kind>::const_iterator, std::size_t);

template string_with_kind
    string_with_kind::join<std::vector<string_with_kind>::const_iterator>(
        std::vector<string_with_kind>::const_iterator,
        std::vector<string_with_kind>::const_iterator, std::size_t);

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

string_with_kind::operator py::str() const {
  return py::reinterpret_steal<py::str>(PyUnicode_FromKindAndData(
      _kind, data(), static_cast<ssize_t>(size() / _kind)));
}

string_with_kind string_with_kind::suffix() const {
  if (empty()) {
    return *this;
  }

  return string_with_kind::compact(data() + kind(), size() - kind(), kind());
}

template <class T>
std::size_t _string_with_kind_find(const void* s, const std::size_t size,
                                   const T c, const std::size_t pos) {
  auto data = (T*)s;
  const auto cp_len = size / sizeof(T);
  for (std::size_t i = pos; i < cp_len; i++) {
    if (data[i] == c) {
      return i;
    }
  }
  return std::string::npos;
}

std::size_t string_find(const void* s, const std::size_t size,
                        const uint8_t kind, const char c,
                        const std::size_t pos) {
  if (kind == 1) {
    return _string_with_kind_find<uint8_t>(s, size, (uint8_t)c, pos);
  } else if (kind == 2) {
    return _string_with_kind_find<uint16_t>(s, size, (uint16_t)c, pos);
  } else {
    return _string_with_kind_find<uint32_t>(s, size, (uint32_t)c, pos);
  }
}

std::size_t string_find(const string_with_kind& s, char c, std::size_t pos) {
  return string_find(s.data(), s.size(), s.kind(), c, pos);
}

std::size_t string_find(const py::str& s, char c, std::size_t pos) {
  const auto data = PyUnicode_1BYTE_DATA(s.ptr());
  const auto len = (std::size_t)PyUnicode_GET_LENGTH(s.ptr());
  const auto kind = (uint8_t)PyUnicode_KIND(s.ptr());
  return string_find(data, len * kind, kind, c, pos);
}

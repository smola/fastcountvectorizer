
#ifndef FCV_STRINGS_H
#define FCV_STRINGS_H

#include <pybind11/pybind11.h>

#include <string>

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace py = pybind11;

class string_with_kind {
 private:
  std::string base;
  uint8_t _kind;

 public:
  string_with_kind(const char* str, const size_t size, const uint8_t kind)
      : base(str, size), _kind(kind) {}
  explicit string_with_kind(py::str s);
  uint8_t kind() const { return _kind; }

  static string_with_kind compact(const char* str, size_t size, uint8_t kind);

  friend std::hash<string_with_kind>;
  const char* data() const { return base.data(); }
  std::size_t size() const { return base.size(); }
  bool empty() const { return base.empty(); }
  bool operator==(const string_with_kind& other) const;
  bool operator!=(const string_with_kind& other) const;
  bool operator<(const string_with_kind& other) const;
  explicit operator py::str() const;
  string_with_kind suffix() const;
};

namespace std {
template <>
struct hash<string_with_kind> {
  size_t operator()(const string_with_kind& k) const {
    return hash<string>()(k.base);
  }
};
}  // namespace std

class fixed_length_string_hash {
 private:
  size_t length;

 public:
  explicit fixed_length_string_hash(const size_t length) noexcept
      : length(length) {}
  size_t operator()(const char* const& str) const {
#if SIZEOF_SIZE_T == 8
    return XXH64(str, length, 0);
#else
    return XXH32(str, length, 0);
#endif
  }
};

struct fixed_length_string_equal_to {
 private:
  size_t length;

 public:
  explicit fixed_length_string_equal_to(const size_t length) noexcept
      : length(length) {}
  bool operator()(const char* const& lhs, const char* const& rhs) const {
    return memcmp(lhs, rhs, length) == 0;
  }
};

#endif  // FCV_STRINGS_H
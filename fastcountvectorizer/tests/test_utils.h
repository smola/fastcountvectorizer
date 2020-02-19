
#ifndef FCV_PYUTILS_H
#define FCV_PYUTILS_H

#include "_strings.h"

// make_pystring creates a string_with_kind with the correct endianness for each
// kind.
template <class T>
string_with_kind make_string_with_kind(const std::vector<T> data) {
  return string_with_kind(reinterpret_cast<const char*>(data.data()),
                          data.size() * sizeof(T), sizeof(T));
}

#endif  // FCV_PYUTILS_H


#ifndef FCV_PYUTILS_H
#define FCV_PYUTILS_H

#include <sstream>

#include "_strings.h"
#include "catch.hpp"

// make_pystring creates a string_with_kind with the correct endianness for each
// kind.
template <class T>
string_with_kind make_string_with_kind(const std::vector<T> data) {
  return string_with_kind(reinterpret_cast<const char*>(data.data()),
                          data.size() * sizeof(T), sizeof(T));
}

namespace Catch {
template <>
struct StringMaker<string_with_kind> {
  static std::string convert(string_with_kind const& value) {
    std::ostringstream os;
    os << "string_with_kind(data=";
    os.write(value.data(), static_cast<std::streamsize>(value.size()));
    os << ", size=" << value.size()
       << ", kind=" << static_cast<int>(value.kind()) << ")";
    return os.str();
  }
};
}  // namespace Catch

#endif  // FCV_PYUTILS_H

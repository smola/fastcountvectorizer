#include <sstream>
#include <vector>

#include "Python.h"
#include "_ext.h"
#include "buzhash.h"
#include "thirdparty/catch.hpp"

static bool py_initialized = false;

void initialize_python() {
  if (!py_initialized) {
    py_initialized = true;
  }
  Py_Initialize();
}

namespace Catch {
template <>
struct StringMaker<FullLightString> {
  static std::string convert(FullLightString const& s) {
    std::ostringstream out;
    out << "FullLightString(byte_len=" << s.byte_len()
        << ", kind=" << ((int)s.kind())
        << ", data=" << std::string(s.data(), s.byte_len()) << ")";
    return out.str();
  }
};
}  // namespace Catch

TEST_CASE("LightStringEqual") {
  std::string s1 = "abcde";
  std::string s2 = "abxxx";
  LightString ls1(const_cast<char*>(s1.data()), 42);
  LightString ls2(const_cast<char*>(s2.data()), 42);

  REQUIRE(LightStringEqual(0)(ls1, ls2));
  REQUIRE(LightStringEqual(1)(ls1, ls2));
  REQUIRE(LightStringEqual(2)(ls1, ls2));
  REQUIRE(!LightStringEqual(3)(ls1, ls2));
  REQUIRE(!LightStringEqual(4)(ls1, ls2));
  REQUIRE(!LightStringEqual(5)(ls1, ls2));
}

TEST_CASE("LightStringHash") {
  LightString ls1(nullptr, 42);
  LightString ls2(nullptr, 42);
  LightString ls3(nullptr, 1);

  REQUIRE(LightStringHash()(ls1) == LightStringHash()(ls1));
  REQUIRE(LightStringHash()(ls1) == LightStringHash()(ls2));
  REQUIRE(LightStringHash()(ls1) != LightStringHash()(ls3));
}

#define BUZHASH(s, l) buzhash::Buzhash<size_t>::hash_once(s.data(), l)

TEST_CASE("LightString::to_full") {
  initialize_python();

  std::string s1("abcde", 5);
  LightString ls1(const_cast<char*>(s1.data()), BUZHASH(s1, 5));

#ifdef WORDS_BIG_ENDIAN
  std::vector<char> s2({0, 'a', 0, 'b', 0, 'c', 0, 'd', 0, 'e'});
#else
  std::vector<char> s2({'a', 0, 'b', 0, 'c', 0, 'd', 0, 'e', 0});
#endif
  LightString ls2(const_cast<char*>(s2.data()), BUZHASH(s2, 10));

#ifdef WORDS_BIG_ENDIAN
  std::vector<char> s3(
      {0, 0, 0, 'a', 0, 0, 0, 'b', 0, 0, 0, 'c', 0, 0, 0, 'd', 0, 0, 0, 'e'});
#else
  std::vector<char> s3({
      'a', 0, 0, 0, 'b', 0, 0, 0, 'c', 0, 0, 0, 'd', 0, 0, 0, 'e', 0, 0, 0,
  });
#endif
  LightString ls3(const_cast<char*>(s3.data()), BUZHASH(s3, 20));

  REQUIRE(ls1.to_full(2, 1) ==
          FullLightString(const_cast<char*>(s1.data()), 2, 1, BUZHASH(s1, 2)));
  REQUIRE(ls2.to_full(4, 2) ==
          FullLightString(const_cast<char*>(s1.data()), 2, 1, BUZHASH(s1, 2)));
  REQUIRE(ls3.to_full(8, 4) ==
          FullLightString(const_cast<char*>(s1.data()), 2, 1, BUZHASH(s1, 2)));
}
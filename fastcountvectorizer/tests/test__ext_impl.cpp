#include "_ext.h"
#include "thirdparty/catch.hpp"

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

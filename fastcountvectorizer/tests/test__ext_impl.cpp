#include "Python.h"
#include "_ext.h"
#include "thirdparty/catch.hpp"

static bool py_initialized = false;

void initialize_python() {
  if (!py_initialized) {
    py_initialized = true;
  }
  Py_Initialize();
}

TEST_CASE("string_with_kind") {
  string_with_kind s = string_with_kind("", 0, 1);
  REQUIRE(s.size() == 0);
  REQUIRE(s.empty());
  REQUIRE(s == s);
  REQUIRE(s != string_with_kind("", 0, 2));
  REQUIRE(s == string_with_kind("", 0, 2).compact());
  REQUIRE(s != string_with_kind("", 0, 4));
  REQUIRE(s == string_with_kind("", 0, 4).compact());
  REQUIRE(s != string_with_kind("a", 1, 1));

  s = string_with_kind("a", 1, 1);
  REQUIRE(s.size() == 1);
  REQUIRE(!s.empty());
  REQUIRE(s == s);
  REQUIRE(s != string_with_kind("", 0, 2));
  REQUIRE(s != string_with_kind("", 0, 2).compact());
  REQUIRE(s != string_with_kind("a\x00", 2, 2));
  REQUIRE(s == string_with_kind("a\x00", 2, 2).compact());
  REQUIRE(s != string_with_kind("a\x00\x00\x00", 4, 4));
  REQUIRE(s == string_with_kind("a\x00\x00\x00", 4, 4).compact());
}

#include "Python.h"
#include "_strings.h"
#include "catch.hpp"
#include "test_utils.h"

void test_string_with_kind_basic(const char* STR, size_t SIZE, uint8_t KIND) {
  auto str = string_with_kind(STR, SIZE, KIND);
  REQUIRE(str == str);
  REQUIRE(str != string_with_kind("xyz", 3, 1));
  REQUIRE(str != string_with_kind(STR, SIZE, (uint8_t)(KIND + 1)));
  REQUIRE(str.size() == SIZE);
  REQUIRE(str.empty() == (SIZE == 0));
  REQUIRE(str.kind() == KIND);
}

TEST_CASE("string_with_kind basic") {
  test_string_with_kind_basic("", 0, 1);
  test_string_with_kind_basic("", 0, 2);
  test_string_with_kind_basic("", 0, 4);
  test_string_with_kind_basic("abc", 3, 2);
  test_string_with_kind_basic("abc", 3, 4);
  test_string_with_kind_basic("\0a\0b\0c", 6, 2);
  test_string_with_kind_basic("a\0b\0c\0", 6, 2);
}

TEST_CASE("string_with_kind compact") {
  REQUIRE(string_with_kind::compact("", 0, 1) == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind::compact("", 0, 2) == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind::compact("abc", 3, 1) ==
          string_with_kind::compact("abc", 3, 1));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint16_t>({519}).data(), 2, 2) ==
          make_string_with_kind<uint16_t>({519}));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint16_t>({1, 2, 3}).data(), 6, 2) ==
          make_string_with_kind<uint8_t>({1, 2, 3}));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint16_t>({301, 302, 303}).data(), 6, 2) ==
          make_string_with_kind<uint16_t>({301, 302, 303}));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint32_t>({1, 2, 3}).data(), 12, 4) ==
          make_string_with_kind<uint8_t>({1, 2, 3}));
}

TEST_CASE("string_with_kind to PyObject") {
  py::str obj;

  obj = string_with_kind("", 0, 1).toPyObject();
  REQUIRE(obj.ptr() != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj.ptr()) == 0);
  REQUIRE(PyUnicode_KIND(obj.ptr()) == 1);

  obj = string_with_kind("abc", 3, 1).toPyObject();
  REQUIRE(obj.ptr() != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj.ptr()) == 3);
  REQUIRE(PyUnicode_KIND(obj.ptr()) == 1);

  obj = make_string_with_kind<uint16_t>({1, 2, 3}).toPyObject();
  REQUIRE(obj.ptr() != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj.ptr()) == 3);
  REQUIRE(PyUnicode_KIND(obj.ptr()) == 1);
}

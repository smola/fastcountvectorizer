#include "Python.h"
#include "_strings.h"
#include "catch.hpp"
#include "pyutils.h"

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
  initialize_python();

  test_string_with_kind_basic("", 0, 1);
  test_string_with_kind_basic("", 0, 2);
  test_string_with_kind_basic("", 0, 4);
  test_string_with_kind_basic("abc", 3, 2);
  test_string_with_kind_basic("abc", 3, 4);
  test_string_with_kind_basic("\0a\0b\0c", 6, 2);
  test_string_with_kind_basic("a\0b\0c\0", 6, 2);
}

TEST_CASE("string_with_kind compact") {
  initialize_python();

  REQUIRE(string_with_kind::compact("", 0, 1) == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind::compact("", 0, 2) == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind::compact("abc", 3, 1) ==
          string_with_kind::compact("abc", 3, 1));
#ifdef WORDS_BIG_ENDIAN
  REQUIRE(string_with_kind::compact("\x01\x8C", 2, 2) ==
          string_with_kind("\x01\x8C", 2, 2));
  REQUIRE(string_with_kind::compact("\0a\0b\0c", 6, 2) ==
          string_with_kind("abc", 3, 1));
  REQUIRE(string_with_kind::compact("\1a\1b\1c", 6, 2) ==
          string_with_kind("\1a\1b\1c", 6, 2));
  REQUIRE(string_with_kind::compact("\0\0\0a\0\0\0b\0\0\0c", 12, 2) ==
          string_with_kind("abc", 3, 1));
#else
  REQUIRE(string_with_kind::compact("\x8C\x01", 2, 2) ==
          string_with_kind("\x8C\x01", 2, 2));
  REQUIRE(string_with_kind::compact("a\0b\0c\0", 6, 2) ==
          string_with_kind("abc", 3, 1));
  REQUIRE(string_with_kind::compact("a\1b\1c\1", 6, 2) ==
          string_with_kind("a\1b\1c\1", 6, 2));
  REQUIRE(string_with_kind::compact("a\0\0\0b\0\0\0c\0\0\0", 12, 4) ==
          string_with_kind("abc", 3, 1));
#endif
}

TEST_CASE("string_with_kind to PyObject") {
  initialize_python();

  PyObject* obj;

  obj = string_with_kind("", 0, 1).toPyObject();
  REQUIRE(obj != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj) == 0);
  REQUIRE(PyUnicode_KIND(obj) == 1);
  Py_DECREF(obj);

  obj = string_with_kind("abc", 3, 1).toPyObject();
  REQUIRE(obj != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj) == 3);
  REQUIRE(PyUnicode_KIND(obj) == 1);
  Py_DECREF(obj);

#ifdef WORDS_BIG_ENDIAN
  obj = string_with_kind("\0a\0b\0c", 6, 2).toPyObject();
#else
  obj = string_with_kind("a\0b\0c\0", 6, 2).toPyObject();
#endif
  REQUIRE(obj != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj) == 3);
  REQUIRE(PyUnicode_KIND(obj) == 1);
  Py_DECREF(obj);
}

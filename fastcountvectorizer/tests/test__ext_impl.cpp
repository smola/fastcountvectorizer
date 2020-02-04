#include "Python.h"
#include "_ext.h"
#include "pyutils.h"
#include "thirdparty/catch.hpp"

void test_string_with_kind_basic(const char* STR, int SIZE, int KIND) {
  auto str = string_with_kind(STR, SIZE, KIND);
  REQUIRE(str == str);
  REQUIRE(str != string_with_kind("xyz", 3, 1));
  REQUIRE(str != string_with_kind(STR, SIZE, KIND + 1));
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

  REQUIRE(string_with_kind("", 0, 1).compact() == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind("", 0, 2).compact() == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind("abc", 3, 1).compact() ==
          string_with_kind("abc", 3, 1));
#ifdef WORDS_BIG_ENDIAN
  REQUIRE(string_with_kind("\x01\x8C", 2, 2).compact() ==
          string_with_kind("\x01\x8C", 2, 2));
  REQUIRE(string_with_kind("\0a\0b\0c", 6, 2).compact() ==
          string_with_kind("abc", 3, 1));
  REQUIRE(string_with_kind("\1a\1b\1c", 6, 2).compact() ==
          string_with_kind("\1a\1b\1c", 6, 2));
  REQUIRE(string_with_kind("\0\0\0a\0\0\0b\0\0\0c", 12, 2).compact() ==
          string_with_kind("abc", 3, 1));
#else
  REQUIRE(string_with_kind("\x8C\x01", 2, 2).compact() ==
          string_with_kind("\x8C\x01", 2, 2));
  REQUIRE(string_with_kind("a\0b\0c\0", 6, 2).compact() ==
          string_with_kind("abc", 3, 1));
  REQUIRE(string_with_kind("a\1b\1c\1", 6, 2).compact() ==
          string_with_kind("a\1b\1c\1", 6, 2));
  REQUIRE(string_with_kind("a\0\0\0b\0\0\0c\0\0\0", 12, 4).compact() ==
          string_with_kind("abc", 3, 1));
#endif
}

TEST_CASE("string_with_kind to PyObject") {
  initialize_python();

  PyObject* obj;

  obj = to_PyObject(string_with_kind("", 0, 1));
  REQUIRE(obj != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj) == 0);
  REQUIRE(PyUnicode_KIND(obj) == 1);
  Py_DECREF(obj);

  obj = to_PyObject(string_with_kind("abc", 3, 1));
  REQUIRE(obj != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj) == 3);
  REQUIRE(PyUnicode_KIND(obj) == 1);
  Py_DECREF(obj);

#ifdef WORDS_BIG_ENDIAN
  obj = to_PyObject(string_with_kind("\0a\0b\0c", 6, 2));
#else
  obj = to_PyObject(string_with_kind("a\0b\0c\0", 6, 2));
#endif
  REQUIRE(obj != nullptr);
  REQUIRE(PyUnicode_GET_LENGTH(obj) == 3);
  REQUIRE(PyUnicode_KIND(obj) == 1);
  Py_DECREF(obj);
}

TEST_CASE("CharNgramCounter(1)") {
  initialize_python();

  CharNgramCounter counter(1);
  PyObject* obj = PyUnicode_FromString("abcde");
  REQUIRE(obj != NULL);
  PyObject* vocab = PyDict_New();
  REQUIRE(vocab != NULL);
  counter.process_one((PyUnicodeObject*)obj);
  REQUIRE(counter.copy_vocab(vocab) == 0);
  REQUIRE(PyDict_Size(vocab) == 5);
  PyObject* indptr = counter.get_indptr();
  REQUIRE(indptr != nullptr);
  PyObject* indices = counter.get_indices();
  REQUIRE(indices != nullptr);
  PyObject* values = counter.get_values();
  REQUIRE(values != nullptr);
  Py_XDECREF(values);
  Py_XDECREF(indices);
  Py_XDECREF(indptr);
  Py_XDECREF(obj);
  Py_XDECREF(vocab);
}

TEST_CASE("CharNgramCounter(1) UCS2 simple") {
  initialize_python();

  CharNgramCounter counter(1);
#ifdef WORDS_BIG_ENDIAN
  PyObject* obj = PyUnicode_FromKindAndData(2, "\0a\0b\0c\0d\0e", 5);
#else
  PyObject* obj = PyUnicode_FromKindAndData(2, "a\0b\0c\0d\0e\0", 5);
#endif
  REQUIRE(obj != NULL);
  PyObject* vocab = PyDict_New();
  REQUIRE(vocab != NULL);
  counter.process_one((PyUnicodeObject*)obj);
  REQUIRE(counter.copy_vocab(vocab) == 0);
  REQUIRE(PyDict_Size(vocab) == 5);
  PyObject* indptr = counter.get_indptr();
  REQUIRE(indptr != nullptr);
  PyObject* indices = counter.get_indices();
  REQUIRE(indices != nullptr);
  PyObject* values = counter.get_values();
  REQUIRE(values != nullptr);
  Py_XDECREF(values);
  Py_XDECREF(indices);
  Py_XDECREF(indptr);
  Py_XDECREF(obj);
  Py_XDECREF(vocab);
}

TEST_CASE("CharNgramCounter(1) UCS2") {
  initialize_python();

  CharNgramCounter counter(1);
#ifdef WORDS_BIG_ENDIAN
  PyObject* obj = PyUnicode_FromKindAndData(2, "\1a\1b\1c\1d\1e", 5);
#else
  PyObject* obj = PyUnicode_FromKindAndData(2, "a\1b\1c\1d\1e\1", 5);
#endif
  REQUIRE(obj != NULL);
  PyObject* vocab = PyDict_New();
  REQUIRE(vocab != NULL);
  counter.process_one((PyUnicodeObject*)obj);
  REQUIRE(counter.copy_vocab(vocab) == 0);
  REQUIRE(PyDict_Size(vocab) == 5);
  PyObject* indptr = counter.get_indptr();
  REQUIRE(indptr != nullptr);
  PyObject* indices = counter.get_indices();
  REQUIRE(indices != nullptr);
  PyObject* values = counter.get_values();
  REQUIRE(values != nullptr);
  Py_XDECREF(values);
  Py_XDECREF(indices);
  Py_XDECREF(indptr);
  Py_XDECREF(obj);
  Py_XDECREF(vocab);
}

TEST_CASE("CharNgramCounter(1) UCS4 simple") {
  initialize_python();

  CharNgramCounter counter(1);
#ifdef WORDS_BIG_ENDIAN
  PyObject* obj =
      PyUnicode_FromKindAndData(4, "\0\0\0a\0\0\0b\0\0\0c\0\0\0d\0\0\0e", 5);
#else
  PyObject* obj =
      PyUnicode_FromKindAndData(4, "a\0\0\0b\0\0\0c\0\0\0d\0\0\0e\0\0\0", 5);
#endif
  REQUIRE(obj != NULL);
  PyObject* vocab = PyDict_New();
  REQUIRE(vocab != NULL);
  counter.process_one((PyUnicodeObject*)obj);
  REQUIRE(counter.copy_vocab(vocab) == 0);
  REQUIRE(PyDict_Size(vocab) == 5);
  PyObject* indptr = counter.get_indptr();
  REQUIRE(indptr != nullptr);
  PyObject* indices = counter.get_indices();
  REQUIRE(indices != nullptr);
  PyObject* values = counter.get_values();
  REQUIRE(values != nullptr);
  Py_XDECREF(values);
  Py_XDECREF(indices);
  Py_XDECREF(indptr);
  Py_XDECREF(obj);
  Py_XDECREF(vocab);
}
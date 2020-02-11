#include "Python.h"
#include "_ext.h"
#include "catch.hpp"
#include "pyutils.h"

TEST_CASE("vocab_map") {
  initialize_python();

  vocab_map v;

  REQUIRE(v.size() == 0);

  REQUIRE(v[string_with_kind("a", 1, 1)] == 0);
  REQUIRE(v[string_with_kind("a", 1, 1)] == 0);
  REQUIRE(v[string_with_kind("b", 1, 1)] == 1);
  REQUIRE(v[string_with_kind("a", 1, 1)] == 0);
  REQUIRE(v[string_with_kind("b", 1, 1)] == 1);

  REQUIRE(v.size() == 2);

  PyObject* dict = PyDict_New();
  REQUIRE(dict != nullptr);
  REQUIRE(v.flush_to(dict) == 0);

  REQUIRE(v.size() == 0);
  REQUIRE(PyDict_Size(dict) == 2);

  PyObject *key, *value;
  Py_ssize_t pos = 0;

  while (PyDict_Next(dict, &pos, &key, &value)) {
    if (PyUnicode_1BYTE_DATA(key)[0] == 'a') {
      REQUIRE(PyLong_AsLong(value) == 0);
    } else if (PyUnicode_1BYTE_DATA(key)[0] == 'b') {
      REQUIRE(PyLong_AsLong(value) == 1);
    } else {
      FAIL("unexpected key");
    }
  }

  Py_DECREF(dict);

  // No effect, but no crash
  REQUIRE(v.flush_to(dict) == 0);
}

TEST_CASE("vocab_map error") {
  initialize_python();

  vocab_map v;
  REQUIRE(v[string_with_kind("a", 1, 1)] == 0);
  PyObject* bad_dict = PyList_New(0);
  REQUIRE(v.flush_to(bad_dict) == -1);
  REQUIRE(PyErr_Occurred() != nullptr);
  PyErr_Clear();
  Py_DECREF(bad_dict);
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
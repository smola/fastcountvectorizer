#include "Python.h"
#include "_sputils.h"
#include "catch.hpp"
#include "pyutils.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL fcv_ARRAY_API
#include "numpy/arrayobject.h"

TEST_CASE("index_vector") {
  initialize_python();

  index_vector v;
  PyObject* obj;
  v.push_back(1);
  v.set_max_value(5);
  v.push_back(2);
  REQUIRE(!v.is_64());

  obj = v.to_numpy();
  REQUIRE(obj != nullptr);
  REQUIRE(PyArray_CheckExact(obj));
  REQUIRE(PyArray_TYPE((PyArrayObject*)obj) == NPY_INT32);
  Py_XDECREF(obj);

  v.set_max_value(NPY_MAX_INT32);
  REQUIRE(!v.is_64());
  obj = v.to_numpy();
  REQUIRE(obj != nullptr);
  REQUIRE(PyArray_CheckExact(obj));
  REQUIRE(PyArray_TYPE((PyArrayObject*)obj) == NPY_INT32);
  Py_XDECREF(obj);

#if NPY_BITSOF_INTP == 64
  v.set_max_value(((size_t)NPY_MAX_INT32) + 1);
  REQUIRE(v.is_64());
  obj = v.to_numpy();
  REQUIRE(obj != nullptr);
  REQUIRE(PyArray_CheckExact(obj));
  REQUIRE(PyArray_TYPE((PyArrayObject*)obj) == NPY_INT64);
  Py_XDECREF(obj);

  v.set_max_value(NPY_MAX_INT64);
  REQUIRE(v.is_64());
  obj = v.to_numpy();
  REQUIRE(obj != nullptr);
  REQUIRE(PyArray_CheckExact(obj));
  REQUIRE(PyArray_TYPE((PyArrayObject*)obj) == NPY_INT64);
  Py_XDECREF(obj);

  v.set_max_value(1);
  REQUIRE(v.is_64());
#endif
}

TEST_CASE("index_vector overflow") {
  initialize_python();
  index_vector v;

#if NPY_BITSOF_INTP == 32
  REQUIRE_THROWS(v.set_max_value(((size_t)NPY_MAX_INT32) + 1));
#else
  REQUIRE_THROWS(v.set_max_value(((size_t)NPY_MAX_INT64) + 1));
#endif
}
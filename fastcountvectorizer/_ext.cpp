// vim: ts=2:sw=2:sts=2:et
// Authors: Santiago M. Mola <santi@mola.io>
// License: MIT License

#include <vector>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/numpyconfig.h>

#include "_ext.h"
#include "_strings.h"

size_t vocab_map::operator[](const string_with_kind& k) {
  auto it = _m.find(k);
  size_t idx;
  if (it == _m.end()) {
    idx = _m.size();
    _m[k] = idx;
  } else {
    idx = it->second;
  }
  return idx;
}

int vocab_map::flush_to(PyObject* dest_vocab) {
  PyObject* key;
  PyObject* value;
  auto it = _m.begin();
  int error = 0;
  while (it != _m.end()) {
    if (error == 0) {
      key = it->first.toPyObject();
      value = PyLong_FromSize_t(it->second);
      if (PyDict_SetItem(dest_vocab, key, value) != 0) {
        error = -1;
      }
      Py_DECREF(key);
      Py_DECREF(value);
    };
    it = _m.erase(it);
  }
  return error;
}

void counter_map::increment_key(const char* k) {
  auto it = find(k);
  if (it == end()) {
    insert({k, 1});
  } else {
    it.value()++;
  }
}

void CharNgramCounter::prepare_vocab() {}

PyObject* CharNgramCounter::_vector_to_numpy(const std::vector<size_t>* v) {
  PyObject* a;
  const size_t size = v->size();
  npy_intp shape[1];
  shape[0] = (npy_intp)size;
  a = PyArray_SimpleNew(1, shape, NPY_UINT64);
  memcpy(PyArray_DATA((PyArrayObject*)a), v->data(), size * sizeof(size_t));
  return a;
}

CharNgramCounter::CharNgramCounter(const unsigned int n) : n(n) {
  prepare_vocab();
  result_array_len = 0;
  values = new std::vector<size_t>();
  indices = new std::vector<size_t>();
  indptr = new std::vector<size_t>();
  indptr->push_back(0);
}

CharNgramCounter::~CharNgramCounter() {
  delete values;
  delete indices;
  delete indptr;
}

void CharNgramCounter::process_one(PyUnicodeObject* obj) {
  const char* data = (char*)PyUnicode_1BYTE_DATA(obj);
  const auto len = (size_t)PyUnicode_GET_LENGTH(obj);
  const auto kind = (uint8_t)PyUnicode_KIND(obj);
  const size_t byte_len = len * kind;

  counter_map counters(kind * n);
  counter_map::iterator cit;

  for (size_t i = 0; i <= byte_len - n * kind; i += kind) {
    const char* data_ptr = data + i;
    counters.increment_key(data_ptr);
  }

  result_array_len += counters.size();
  values->reserve(counters.size());
  indices->reserve(counters.size());
  indptr->push_back(result_array_len);

  for (cit = counters.begin(); cit != counters.end(); cit++) {
    const size_t term_idx =
        vocab[string_with_kind::compact(cit->first, n * kind, kind)];
    indices->push_back(term_idx);
    values->push_back(cit->second);
  }
}

PyObject* CharNgramCounter::get_values() {
  PyObject* v = _vector_to_numpy(values);
  delete values;
  values = NULL;
  return v;
}

PyObject* CharNgramCounter::get_indices() {
  PyObject* v = _vector_to_numpy(indices);
  delete indices;
  indices = NULL;
  return v;
}

PyObject* CharNgramCounter::get_indptr() {
  PyObject* v = _vector_to_numpy(indptr);
  delete indptr;
  indptr = NULL;
  return v;
}

int CharNgramCounter::copy_vocab(PyObject* dest_vocab) {
  return vocab.flush_to(dest_vocab);
}

typedef struct {
  PyObject_HEAD PyObject* vocab;
  CharNgramCounter* counter;
} CharNgramCounterObject;

static int CharNgramCounter_init(CharNgramCounterObject* self, PyObject* args,
                                 PyObject* kwds) {
  PyObject* vocab;
  Py_ssize_t n;
  static const char* kwlist[] = {"n", "vocab", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nO", const_cast<char**>(kwlist),
                                   &n, &vocab)) {
    return -1;
  }

  if (!PyDict_Check(vocab)) {
    PyErr_SetString(PyExc_TypeError, "vocab must be a dict");
    return -1;
  }

  if (n <= 0) {
    PyErr_SetString(PyExc_ValueError, "n must be greater than 0");
    return -1;
  }

  Py_INCREF(vocab);
  self->vocab = vocab;
  self->counter = new CharNgramCounter((unsigned int)n);
  return 0;
}

static void CharNgramCounter_dealloc(CharNgramCounterObject* self) {
  Py_XDECREF(self->vocab);
  delete self->counter;
  self->counter = NULL;
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* CharNgramCounter_process(CharNgramCounterObject* self,
                                          PyObject* args, PyObject* kwds) {
  PyObject* doc;
  static const char* kwlist[] = {"doc", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist),
                                   &doc)) {
    return NULL;
  }

  if (!PyUnicode_Check(doc)) {
    PyErr_SetString(PyExc_TypeError, "all documents must be strings");
    return NULL;
  }

  self->counter->process_one((PyUnicodeObject*)doc);

  Py_RETURN_NONE;
}

static PyObject* CharNgramCounter_get_result(CharNgramCounterObject* self,
                                             PyObject* Py_UNUSED(ignored)) {
  PyObject* values = self->counter->get_values();
  PyObject* indices = self->counter->get_indices();
  PyObject* indptr = self->counter->get_indptr();
  PyObject* result = PyTuple_Pack(3, values, indices, indptr);
  if (result == NULL) {
    Py_DECREF(values);
    Py_DECREF(indices);
    Py_DECREF(indptr);
    return NULL;
  }

  if (self->counter->copy_vocab(self->vocab) != 0) {
    Py_DECREF(values);
    Py_DECREF(indices);
    Py_DECREF(indptr);
    return NULL;
  }

  Py_DECREF(values);
  Py_DECREF(indices);
  Py_DECREF(indptr);
  return result;
}

static PyMethodDef CharNgramCounter_methods[] = {
    {"process", (PyCFunction)CharNgramCounter_process,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_result", (PyCFunction)CharNgramCounter_get_result, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static PyTypeObject CharNgramCounterType = {PyVarObject_HEAD_INIT(NULL, 0)};

static PyModuleDef _extmodule = {PyModuleDef_HEAD_INIT};
#pragma GCC diagnostic pop

static void CharNgramCounterType_init_struct() {
  CharNgramCounterType.tp_name = "fastcountvectorizer._ext._CharNgramCounter";
  CharNgramCounterType.tp_basicsize = sizeof(CharNgramCounterObject);
  CharNgramCounterType.tp_itemsize = 0;
  CharNgramCounterType.tp_dealloc = (destructor)CharNgramCounter_dealloc;
  CharNgramCounterType.tp_flags = Py_TPFLAGS_DEFAULT;
  CharNgramCounterType.tp_doc = NULL;
  CharNgramCounterType.tp_new = PyType_GenericNew;
  CharNgramCounterType.tp_init = (initproc)CharNgramCounter_init;
  CharNgramCounterType.tp_methods = CharNgramCounter_methods;
}

static void PyModuleDef_init_struct() {
  _extmodule.m_name = "fastcountvectorizer._ext";
  _extmodule.m_size = -1;
}

PyMODINIT_FUNC PyInit__ext(void) {
  PyModuleDef_init_struct();
  CharNgramCounterType_init_struct();
  if (PyType_Ready(&CharNgramCounterType) < 0) return NULL;

  PyObject* m = PyModule_Create(&_extmodule);
  if (m == NULL) return NULL;

  import_array();

  Py_INCREF(&CharNgramCounterType);
  if (PyModule_AddObject(m, "_CharNgramCounter",
                         (PyObject*)&CharNgramCounterType) < 0) {
    Py_DECREF(&CharNgramCounterType);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}

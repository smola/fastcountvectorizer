// vim: ts=2:sw=2:sts=2:et
// Authors: Santiago M. Mola <santi@mola.io>
// License: MIT License

#include <unordered_map>
#include <vector>

#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/numpyconfig.h>

#include "buzhash.h"

class FullLightString {
 public:
  char* data;
  size_t len;
  size_t _hash;
  unsigned char kind;
  bool _owned;

  FullLightString(char* data, size_t len, unsigned char kind, size_t hash)
      : data(data), len(len), _hash(hash), kind(kind), _owned(false) {}

  FullLightString() : FullLightString(NULL, 0, PyUnicode_1BYTE_KIND, 0) {}

  void own() {
    if (!_owned) {
      const char* old_data = data;
      data = new char[len];
      memcpy(data, old_data, len);
      _owned = true;
    }
  }

  void free() {
    if (_owned) {
      delete data;
    }
  }

  PyObject* toPyObject() const {
    return PyUnicode_FromKindAndData(kind, data, len / kind);
  }
};

struct FullLightStringHash {
  std::size_t operator()(const FullLightString& k) const noexcept {
    return k._hash;
  }
};

struct FullLightStringEqual {
  bool operator()(const FullLightString& lhs,
                  const FullLightString& rhs) const {
    if (lhs.len != rhs.len) {
      return false;
    }
    return memcmp(lhs.data, rhs.data, lhs.len) == 0;
  }
};

class LightString {
 public:
  char* data;
  size_t _hash;

  LightString(char* data, size_t hash) : data(data), _hash(hash) {}

  LightString() : LightString(NULL, 0) {}

  FullLightString to_full(const size_t len, const unsigned char kind) const {
    PyObject* obj;
    FullLightString str;
    if (kind == PyUnicode_1BYTE_KIND) {
      return FullLightString(data, len, kind, _hash);
    }
    obj = PyUnicode_FromKindAndData(kind, data, len / kind);
    if (PyUnicode_KIND(obj) == kind) {
      Py_DECREF(obj);
      return FullLightString(data, len, kind, _hash);
    }
    str = FullLightString((char*)PyUnicode_1BYTE_DATA(obj),
                          PyUnicode_GET_LENGTH(obj) * PyUnicode_KIND(obj),
                          (unsigned char)PyUnicode_KIND(obj),
                          buzhash::Buzhash<std::size_t>::hash_once(
                              (char*)PyUnicode_1BYTE_DATA(obj),
                              PyUnicode_GET_LENGTH(obj) * PyUnicode_KIND(obj)));
    str.own();
    Py_DECREF(obj);
    return str;
  }
};

struct LightStringHash {
  std::size_t operator()(const LightString& k) const noexcept {
    return k._hash;
  }
};

class LightStringEqual {
 public:
  size_t len;

  LightStringEqual() : len(0) {}

  LightStringEqual(size_t len) : len(len) {}

  bool operator()(const LightString& lhs, const LightString& rhs) const {
    return memcmp(lhs.data, rhs.data, len) == 0;
  }
};

typedef std::unordered_map<FullLightString, int, FullLightStringHash,
                           FullLightStringEqual>
    vocab_map;
typedef std::unordered_map<LightString, int, LightStringHash, LightStringEqual>
    counter_map;

class CharNgramCounter {
 private:
  vocab_map vocab;
  const int n;

  size_t result_array_len;
  std::vector<size_t>* values;
  std::vector<size_t>* indices;
  std::vector<size_t>* indptr;

  void prepare_vocab() {}

 public:
  CharNgramCounter(const int n) : n(n) {
    prepare_vocab();
    result_array_len = 0;
    values = new std::vector<size_t>();
    indices = new std::vector<size_t>();
    indptr = new std::vector<size_t>();
    indptr->push_back(0);
  }

  ~CharNgramCounter() {
    if (values != NULL) {
      delete values;
    }
    if (indices != NULL) {
      delete indices;
    }
    if (indptr != NULL) {
      delete indptr;
    }
  }

  void process_one(PyUnicodeObject* obj) {
    const char* data = (char*)PyUnicode_1BYTE_DATA(obj);
    const size_t len = PyUnicode_GET_LENGTH(obj);
    const auto kind = (unsigned char)PyUnicode_KIND(obj);
    const size_t byte_len = len * kind;

    char* data_ptr = (char*)data;
    LightString str;
    size_t cur_byte_idx = 0;

    counter_map counters(10, LightStringHash(), LightStringEqual(n * kind));
    counter_map::iterator cit;
    vocab_map::iterator vit;

    while (cur_byte_idx <= byte_len - n * kind) {
      // read ngram
      str.data = data_ptr;
      str._hash = buzhash::Buzhash<size_t>::hash_once(data_ptr, n * kind);
      cur_byte_idx += kind;
      data_ptr += kind;

      // increment counters
      cit = counters.find(str);
      if (cit == counters.end()) {
        counters[str] = 1;
      } else {
        cit->second++;
      }
    }

    result_array_len += counters.size();
    values->reserve(counters.size());
    indices->reserve(counters.size());
    indptr->push_back(result_array_len);

    for (cit = counters.begin(); cit != counters.end(); cit++) {
      FullLightString full_str = cit->first.to_full(n * kind, kind);
      vit = vocab.find(full_str);
      if (vit == vocab.end()) {
        const size_t term_idx = vocab.size();
        full_str.own();
        vocab[full_str] = term_idx;
        indices->push_back(term_idx);
      } else {
        full_str.free();
        const size_t term_idx = vit->second;
        indices->push_back(term_idx);
      }
      values->push_back(cit->second);
    }
  }

  PyObject* _vector_to_numpy(const std::vector<size_t>* v) {
    PyObject* a;
    const size_t size = v->size();
    npy_intp shape[1];
    shape[0] = (npy_intp)size;
    a = PyArray_SimpleNew(1, shape, NPY_UINT64);
    memcpy(PyArray_DATA((PyArrayObject*)a), v->data(), size * sizeof(size_t));
    return a;
  }

  PyObject* get_values() {
    PyObject* v = _vector_to_numpy(values);
    delete values;
    values = NULL;
    return v;
  }

  PyObject* get_indices() {
    PyObject* v = _vector_to_numpy(indices);
    delete indices;
    indices = NULL;
    return v;
  }

  PyObject* get_indptr() {
    PyObject* v = _vector_to_numpy(indptr);
    delete indptr;
    indptr = NULL;
    return v;
  }

  int copy_vocab(PyObject* dest_vocab) {
    PyObject* key;
    PyObject* value;
    vocab_map::iterator it = vocab.begin();
    char* data;
    int error = 0;
    while (it != vocab.end()) {
      if (error == 0) {
        key = it->first.toPyObject();
        value = PyLong_FromSize_t(it->second);
        if (PyDict_SetItem(dest_vocab, key, value) != 0) {
          error = -1;
        }
        Py_DECREF(key);
        Py_DECREF(value);
      }
      data = it->first.data;
      it = vocab.erase(it);
      free(data);
    }
    return 0;
  }
};

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
                                   &n, &vocab))
    return -1;

  if (!PyDict_Check(vocab)) {
    PyErr_SetString(PyExc_TypeError, "vocab must be a dict");
    return -1;
  }

  Py_INCREF(vocab);
  self->vocab = vocab;
  self->counter = new CharNgramCounter(n);
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
    {NULL} /* Sentinel */
};

static PyTypeObject CharNgramCounterType = {PyVarObject_HEAD_INIT(NULL, 0)};

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

static PyModuleDef _extmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "fastcountvectorizer._ext",
    .m_doc = NULL,
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__ext(void) {
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

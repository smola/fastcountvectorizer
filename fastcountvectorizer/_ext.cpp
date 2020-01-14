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

  int process_all(PyObject* obj) {
    const size_t n_docs = PySequence_Length(obj);
    PyObject* err;
    PyObject* it;
    PyObject* el;

    indptr->reserve(n_docs);

    it = PySeqIter_New(obj);
    if (it == NULL) {
      return -1;
    }

    while ((el = PyIter_Next(it)) != NULL) {
      if (!PyUnicode_Check(el)) {
        Py_DECREF(el);
        Py_DECREF(it);
        PyErr_SetString(PyExc_TypeError, "all documents must be strings");
        return -1;
      }

      process_one((PyUnicodeObject*)el);

      Py_DECREF(el);
    }
    Py_DECREF(it);
    if ((err = PyErr_Occurred()) != NULL) {
      Py_DECREF(err);
      return -1;
    }

    return 0;
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

// void count_ngrams(list docs, object vocab_dict_)
static PyObject* count_ngrams(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* vocab;
  PyObject* docs;
  Py_ssize_t n;
  static const char* kwlist[] = {"n", "docs", "vocab", NULL};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "nOO", const_cast<char**>(kwlist), &n, &docs, &vocab)) {
    return NULL;
  }

  if (!PySequence_Check(docs)) {
    PyErr_SetString(PyExc_TypeError, "second argument must be a sequence");
    return NULL;
  }

  if (!PyDict_Check(vocab)) {
    PyErr_SetString(PyExc_TypeError, "third argument must be a dict");
    return NULL;
  }

  auto counter = CharNgramCounter(n);
  if (counter.process_all(docs) != 0) {
    return NULL;
  }

  PyObject* values = counter.get_values();
  PyObject* indices = counter.get_indices();
  PyObject* indptr = counter.get_indptr();
  PyObject* result = PyTuple_Pack(3, values, indices, indptr);
  if (result == NULL) {
    Py_DECREF(values);
    Py_DECREF(indices);
    Py_DECREF(indptr);
    return NULL;
  }

  if (counter.copy_vocab(vocab) != 0) {
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

static PyMethodDef methods[] = {{"_count_ngrams", (PyCFunction)count_ngrams,
                                 METH_VARARGS | METH_KEYWORDS, NULL},
                                {NULL, NULL, 0, NULL}};

static PyModuleDef _extmodule = {
    PyModuleDef_HEAD_INIT, .m_name = "fastcountvectorizer._ext",
    .m_doc = NULL,         .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__ext(void) {
  PyObject* m = PyModule_Create(&_extmodule);
  if (m == NULL) return NULL;

  import_array();
  return m;
}

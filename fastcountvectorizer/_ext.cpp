// vim: ts=2:sw=2:sts=2:et
// Authors: Santiago M. Mola <santi@mola.io>
// License: MIT License

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL fcv_ARRAY_API
#include <numpy/arrayobject.h>

#include "_csr.h"
#include "_ext.h"
#include "_sputils.h"
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

std::vector<std::pair<string_with_kind, size_t>> vocab_map::to_vector() const {
  std::vector<std::pair<string_with_kind, size_t>> result;
  result.reserve(_m.size());
  for (auto it = _m.begin(); it != _m.end(); it++) {
    result.emplace_back(*it);
  }
  return result;
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

CharNgramCounter::CharNgramCounter(const unsigned int min_n,
                                   const unsigned int max_n)
    : min_n(min_n), max_n(max_n) {
  prepare_vocab();
  result_array_len = 0;
  if (need_expand_counts()) {
    prefixes = new std::vector<string_with_kind>();
  } else {
    prefixes = nullptr;
  }
  values = new std::vector<std::int64_t>();
  indices = new index_vector();
  indptr = new index_vector();
  indptr->push_back(0);
}

CharNgramCounter::~CharNgramCounter() {
  delete prefixes;
  delete values;
  delete indices;
  delete indptr;
}

void CharNgramCounter::process_one(PyUnicodeObject* obj) {
  const unsigned int n = max_n;
  const char* data = (char*)PyUnicode_1BYTE_DATA(obj);
  const auto len = PyUnicode_GET_LENGTH(obj);
  const auto kind = (uint8_t)PyUnicode_KIND(obj);
  const size_t byte_len = (size_t)len * kind;

  counter_map counters(kind * n);
  counter_map::iterator cit;

  if (need_expand_counts()) {
    const unsigned int prefix_len = (len <= max_n) ? (unsigned int)len : max_n;
    prefixes->push_back(string_with_kind(data, prefix_len * kind, kind));
  }

  for (size_t i = 0; i <= byte_len - n * kind; i += kind) {
    const char* data_ptr = data + i;
    counters.increment_key(data_ptr);
  }

  result_array_len += counters.size();
  values->reserve(counters.size());
  indices->set_max_value({vocab.size(), result_array_len});
  indices->reserve(counters.size());
  indptr->set_max_value({vocab.size(), result_array_len});
  indptr->push_back(result_array_len);

  for (cit = counters.begin(); cit != counters.end(); cit++) {
    const size_t term_idx =
        vocab[string_with_kind::compact(cit->first, n * kind, kind)];
    indices->push_back(term_idx);
    values->push_back(cit->second);
  }
}

bool CharNgramCounter::need_expand_counts() const { return min_n < max_n; }

bool vocab_idx_less_than(const std::pair<string_with_kind, size_t>& a,
                         const std::pair<string_with_kind, size_t>& b) {
  return a.second < b.second;
}

void count_expansion_csr_matrix(vocab_map& vocab,
                                std::vector<std::intptr_t>& conv_indices,
                                const unsigned int min_n,
                                const unsigned int max_n) {
  // copy vocab (and sort) for iteration concurrent with modification
  std::vector<std::pair<string_with_kind, size_t>> vocab_copy =
      vocab.to_vector();

  // sort is required omit storing old term index when computing the conversion
  // matrix (see below).
  std::sort(vocab_copy.begin(), vocab_copy.end(), vocab_idx_less_than);

  // compute conversion matrix (_, conv_indices, _) in CSR format.
  // actual values are omitted, since they are always 1.
  // indptr is also omitted since it is always in increments of max_n-min_n
  conv_indices.resize(vocab.size() * (size_t)(max_n - min_n));
  size_t i_indices = 0;
  for (auto it = vocab_copy.begin(); it != vocab_copy.end(); it++) {
    string_with_kind new_term = it->first;
    for (unsigned int n = max_n - 1; n >= min_n; n--) {
      new_term = new_term.suffix();
      const size_t term_idx = vocab[new_term];
      assert(term_idx >= vocab_copy.size());
      conv_indices[i_indices++] = (std::intptr_t)term_idx;
    }
  }
}

void prefixes_add_csr_matrix(vocab_map& vocab,
                             const std::vector<string_with_kind>& prefixes,
                             std::vector<std::intptr_t>& prefixes_indptr,
                             std::vector<std::intptr_t>& prefixes_indices,
                             const unsigned int min_n,
                             const unsigned int max_n) {
  prefixes_indices.reserve(prefixes.size() * (max_n - min_n));
  prefixes_indptr.resize(prefixes.size() + 1);
  prefixes_indptr[0] = 0;
  for (unsigned int i = 0; i < prefixes.size(); i++) {
    const uint8_t kind = prefixes[i].kind();
    const unsigned int prefix_len = (unsigned int)prefixes[i].size() / kind;
    const unsigned int new_max_n =
        (prefix_len <= max_n - 1) ? prefix_len : max_n - 1;
    for (unsigned int n = new_max_n; n >= min_n; n--) {
      for (unsigned int start = 0; start < max_n - n; start++) {
        string_with_kind new_term = string_with_kind::compact(
            prefixes[i].data() + (start * kind), (n * kind), kind);
        prefixes_indices.push_back((std::intptr_t)vocab[new_term]);
      }
    }
    prefixes_indptr[i + 1] = (std::intptr_t)prefixes_indices.size();
  }
}

void CharNgramCounter::expand_counts() {
  if (!need_expand_counts()) {
    return;
  }

  if (vocab.size() > std::numeric_limits<std::intptr_t>::max()) {
    throw std::overflow_error("too many vocabulary terms");
  }

  // compute conversion matrix (_, conv_indices, _) in CSR format.
  // actual values are omitted, since they are always 1.
  // indptr is also omitted since it is always in increments of max_n-min_n
  std::vector<std::intptr_t> conv_indices;
  count_expansion_csr_matrix(vocab, conv_indices, min_n, max_n);

  // compute CSR matrix for prefixes, data is always 1
  std::vector<std::intptr_t> prefixes_indices;
  std::vector<std::intptr_t> prefixes_indptr;
  prefixes_add_csr_matrix(vocab, *prefixes, prefixes_indptr, prefixes_indices,
                          min_n, max_n);
  delete prefixes;
  prefixes = nullptr;

  // final matrix shape
  const std::size_t n_row = indptr->size() - 1;
  const std::size_t n_col = vocab.size();
  const auto nnz_per_B_row = (std::size_t)(max_n - min_n);

  auto new_indptr = new index_vector();
  auto new_indices = new index_vector();
  auto new_values = new std::vector<std::int64_t>();

  csr_matmat_add_Bx1_diagprefix_fixed_nnz(
      n_row, n_col, *indptr, *indices, *values, conv_indices, nnz_per_B_row,
      prefixes_indptr, prefixes_indices, *new_indptr, *new_indices,
      *new_values);

  std::swap(indptr, new_indptr);
  delete new_indptr;
  std::swap(indices, new_indices);
  delete new_indices;
  std::swap(values, new_values);
  delete new_values;
}

std::vector<std::size_t> CharNgramCounter::document_frequencies() const {
  std::vector<std::size_t> docfreq(vocab.size(), 0);
  for (std::size_t i = 0; i < indptr->size() - 1; i++) {
    for (auto j = (std::size_t)(*indptr)[i]; j < (size_t)(*indptr)[i + 1];
         j++) {
      const auto idx = (std::size_t)(*indices)[j];
      docfreq[idx]++;
    }
  }
  return docfreq;
}

PyObject* CharNgramCounter::limit_features(const size_t min_df,
                                           const size_t max_df) {
  PyObject* stop_words = PySet_New(nullptr);
  std::vector<std::int64_t> new_vocab_indices(vocab.size(), -1);
  std::vector<size_t> docfreq = document_frequencies();
  std::vector<std::pair<string_with_kind, size_t>> vocab_copy =
      vocab.to_vector();
  size_t new_index = 0;
  for (auto it = vocab_copy.begin(); it != vocab_copy.end(); it++) {
    const std::size_t old_idx = it->second;
    const std::size_t f = docfreq[old_idx];
    if (f >= min_df && f <= max_df) {
      vocab.set_index(it->first, new_index);
      new_vocab_indices[old_idx] = (std::int64_t)new_index;
      new_index++;
    } else {
      PyObject* pystr = it->first.toPyObject();
      PySet_Add(stop_words, pystr);
      Py_DECREF(pystr);
      vocab.erase(it->first);
    }
  }

  auto new_indices = new index_vector();
  auto new_indptr = new index_vector();
  auto new_values = new std::vector<std::int64_t>();
  transform_indices(vocab.size(), new_vocab_indices, *indptr, *indices, *values,
                    *new_indptr, *new_indices, *new_values);
  std::swap(indices, new_indices);
  delete new_indices;
  std::swap(indptr, new_indptr);
  delete new_indptr;
  std::swap(values, new_values);
  delete new_values;

  return stop_words;
}

PyObject* CharNgramCounter::get_values() {
  PyObject* v = vector_to_numpy(*values);
  delete values;
  values = NULL;
  return v;
}

PyObject* CharNgramCounter::get_indices() {
  PyObject* v = indices->to_numpy();
  delete indices;
  indices = NULL;
  return v;
}

PyObject* CharNgramCounter::get_indptr() {
  PyObject* v = indptr->to_numpy();
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
  Py_ssize_t min_n, max_n;
  static const char* kwlist[] = {"min_n", "max_n", "vocab", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nnO",
                                   const_cast<char**>(kwlist), &min_n, &max_n,
                                   &vocab)) {
    return -1;
  }

  if (!PyDict_Check(vocab)) {
    PyErr_SetString(PyExc_TypeError, "vocab must be a dict");
    return -1;
  }

  if (min_n <= 0) {
    PyErr_SetString(PyExc_ValueError, "min_n must be greater than 0");
    return -1;
  }

  if (max_n < min_n) {
    PyErr_SetString(PyExc_ValueError,
                    "max_n must be equal or greather than min_n");
    return -1;
  }

  Py_INCREF(vocab);
  self->vocab = vocab;
  self->counter =
      new CharNgramCounter((unsigned int)min_n, (unsigned int)max_n);
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

static PyObject* CharNgramCounter_postprocess(CharNgramCounterObject* self,
                                              PyObject* Py_UNUSED(ignored)) {
  try {
    self->counter->expand_counts();
  } catch (std::exception& e) {
    PyErr_SetString(PyExc_SystemError, e.what());
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject* CharNgramCounter_limit_features(CharNgramCounterObject* self,
                                                 PyObject* args,
                                                 PyObject* kwds) {
  Py_ssize_t min_df, max_df;
  static const char* kwlist[] = {"min_df", "max_df", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nn", const_cast<char**>(kwlist),
                                   &min_df, &max_df)) {
    return nullptr;
  }

  if (min_df < 0) {
    PyErr_SetString(PyExc_ValueError, "min_df must be equal or greater than 0");
    return nullptr;
  }

  if (max_df < min_df) {
    PyErr_SetString(PyExc_ValueError,
                    "max_df must be equal or greater than min_df");
    return nullptr;
  }

  return self->counter->limit_features((size_t)min_df, (size_t)max_df);
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
    {"postprocess", (PyCFunction)CharNgramCounter_postprocess, METH_NOARGS,
     NULL},
    {"limit_features", (PyCFunction)CharNgramCounter_limit_features,
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

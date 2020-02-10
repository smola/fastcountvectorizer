
#include "Python.h"

/* defined in _ext.cpp */
PyMODINIT_FUNC PyInit__ext(void);

static bool py_initialized = false;

void initialize_python() {
  if (py_initialized) {
    return;
  }
  py_initialized = true;
  Py_Initialize();
  auto m = PyInit__ext();
  if (PyErr_Occurred()) {
    PyErr_Print();
    assert(false);
  }
  assert(m != nullptr);
}

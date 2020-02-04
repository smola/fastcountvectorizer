
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
  PyInit__ext();
}

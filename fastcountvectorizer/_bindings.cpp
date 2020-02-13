
#include <pybind11/pybind11.h>

#include "_ext.h"

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
  m.doc() = "fastcountvectorizer internal extension. DO NOT USE DIRECTLY.";
  py::class_<CharNgramCounter>(m, "_CharNgramCounter")
      .def(py::init<const unsigned int, const unsigned int>(), py::arg("min_n"),
           py::arg("max_n"))
      .def("process", &CharNgramCounter::process, py::arg("doc"))
      .def("expand_counts", &CharNgramCounter::expand_counts)
      .def("limit_features", &CharNgramCounter::limit_features,
           py::arg("min_df"), py::arg("max_df"))
      .def("sort_features", &CharNgramCounter::sort_features)
      .def("get_vocab", &CharNgramCounter::get_vocab,
           py::return_value_policy::move)
      .def("get_result", &CharNgramCounter::get_result,
           py::return_value_policy::move);
}
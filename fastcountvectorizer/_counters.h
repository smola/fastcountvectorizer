#ifndef FCV_COUNTERS_H
#define FCV_COUNTERS_H

#include <pybind11/pybind11.h>

#include "_collections.h"
#include "_sputils.h"

namespace py = pybind11;

class CharNgramCounter {
 private:
  vocab_map vocab;
  unsigned int min_n;
  unsigned int max_n;

  std::size_t result_array_len;
  std::vector<string_with_kind>* prefixes;
  std::vector<std::int64_t>* values;
  index_vector* indices;
  index_vector* indptr;

  void prepare_vocab();
  bool need_expand_counts() const;
  std::vector<size_t> document_frequencies() const;

 public:
  CharNgramCounter(unsigned int min_n, unsigned int max_n);
  ~CharNgramCounter();

  void process(const py::str& obj);
  void expand_counts();
  py::set limit_features(std::size_t min_df, std::size_t max_df);
  void sort_features();
  py::array get_values();
  py::array get_indices();
  py::array get_indptr();
  py::tuple get_result();
  py::dict get_vocab();
};

#endif  // FCV_COUNTERS_H
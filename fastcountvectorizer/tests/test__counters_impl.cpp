
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "_counters.h"
#include "catch.hpp"
#include "test_utils.h"

TEST_CASE("CharNgramCounter(char, 1, 1)") {
  CharNgramCounter counter("char", 1, 1, py::none(), py::none(), true);
  counter.process(py::str("abcde"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 5);
  py::array indptr = counter.get_indptr();
  REQUIRE(indptr.ptr() != nullptr);
  py::array indices = counter.get_indices();
  REQUIRE(indices.ptr() != nullptr);
  py::array values = counter.get_values();
  REQUIRE(values.ptr() != nullptr);
}

TEST_CASE("CharNgramCounter(char, 1, 1) twice") {
  CharNgramCounter counter =
      CharNgramCounter("char", 1, 1, py::none(), py::none(), true);
  counter.process(py::str("abcde"));
  counter.process(py::str("abcde"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 5);
}

TEST_CASE("CharNgramCounter(char, 1, 1) UCS2") {
  CharNgramCounter counter("char", 1, 1, py::none(), py::none(), true);
  counter.process(static_cast<py::str>(
      make_string_with_kind<uint16_t>({7001, 7002, 7003, 7004, 7005})));
  REQUIRE(counter.get_vocab().size() == 5);
  py::array indptr = counter.get_indptr();
  REQUIRE(indptr.ptr() != nullptr);
  py::array indices = counter.get_indices();
  REQUIRE(indices.ptr() != nullptr);
  py::array values = counter.get_values();
  REQUIRE(values.ptr() != nullptr);
}

TEST_CASE("CharNgramCounter(char, 1, 1) UCS4") {
  CharNgramCounter counter("char", 1, 1, py::none(), py::none(), true);
  counter.process(static_cast<py::str>(
      make_string_with_kind<uint32_t>({70001, 70002, 70003, 70004, 70005})));
  REQUIRE(counter.get_vocab().size() == 5);
  py::array indptr = counter.get_indptr();
  REQUIRE(indptr.ptr() != nullptr);
  py::array indices = counter.get_indices();
  REQUIRE(indices.ptr() != nullptr);
  py::array values = counter.get_values();
  REQUIRE(values.ptr() != nullptr);
}

TEST_CASE("CharNgramCounter(char, 1, 2)") {
  CharNgramCounter counter("char", 1, 2, py::none(), py::none(), true);
  counter.process(py::str("abcde"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 9); /* 5 + 4 */
}

TEST_CASE("CharNgramCounter(char, 1, 3)") {
  CharNgramCounter counter("char", 1, 3, py::none(), py::none(), true);
  counter.process(py::str("abcde"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 12); /* 5 + 4 + 3 */
}

TEST_CASE("CharNgramCounter(word, 1, 3)") {
  CharNgramCounter counter("word", 1, 3, py::none(), py::none(), true);
  counter.process(py::str("aaa bbb ccc ddd eee"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 12); /* 5 + 4 + 3 */
}

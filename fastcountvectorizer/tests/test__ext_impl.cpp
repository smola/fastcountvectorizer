
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "_ext.h"
#include "catch.hpp"
#include "test_utils.h"

TEST_CASE("vocab_map") {
  vocab_map v;

  REQUIRE(v.size() == 0);

  REQUIRE(v[string_with_kind("a", 1, 1)] == 0);
  REQUIRE(v[string_with_kind("a", 1, 1)] == 0);
  REQUIRE(v[string_with_kind("b", 1, 1)] == 1);
  REQUIRE(v[string_with_kind("a", 1, 1)] == 0);
  REQUIRE(v[string_with_kind("b", 1, 1)] == 1);

  REQUIRE(v.size() == 2);

  py::dict dict = v.flush_to_pydict();

  REQUIRE(v.size() == 0);
  REQUIRE(dict.size() == 2);

  for (auto it = dict.begin(); it != dict.end(); it++) {
    std::string key = py::reinterpret_steal<py::str>(it->first);
    int val = py::reinterpret_steal<py::int_>(it->second);

    if (key == std::string("a")) {
      REQUIRE(val == 0);
    } else if (key == std::string("b")) {
      REQUIRE(val == 1);
    } else {
      FAIL("unexpected key");
    }
  }

  // No effect, but no crash
  REQUIRE(v.flush_to_pydict().empty());
}

TEST_CASE("CharNgramCounter(1, 1)") {
  CharNgramCounter counter(1, 1);
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

TEST_CASE("CharNgramCounter(1, 1) twice") {
  CharNgramCounter counter = CharNgramCounter(1, 1);
  counter.process(py::str("abcde"));
  counter.process(py::str("abcde"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 5);
}

TEST_CASE("CharNgramCounter(1, 1) UCS2") {
  CharNgramCounter counter(1, 1);
  counter.process(
      make_string_with_kind<uint16_t>({7001, 7002, 7003, 7004, 7005})
          .toPyObject());
  REQUIRE(counter.get_vocab().size() == 5);
  py::array indptr = counter.get_indptr();
  REQUIRE(indptr.ptr() != nullptr);
  py::array indices = counter.get_indices();
  REQUIRE(indices.ptr() != nullptr);
  py::array values = counter.get_values();
  REQUIRE(values.ptr() != nullptr);
}

TEST_CASE("CharNgramCounter(1, 1) UCS4") {
  CharNgramCounter counter(1, 1);
  counter.process(
      make_string_with_kind<uint32_t>({70001, 70002, 70003, 70004, 70005})
          .toPyObject());
  REQUIRE(counter.get_vocab().size() == 5);
  py::array indptr = counter.get_indptr();
  REQUIRE(indptr.ptr() != nullptr);
  py::array indices = counter.get_indices();
  REQUIRE(indices.ptr() != nullptr);
  py::array values = counter.get_values();
  REQUIRE(values.ptr() != nullptr);
}

TEST_CASE("CharNgramCounter(1, 2)") {
  CharNgramCounter counter(1, 2);
  counter.process(py::str("abcde"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 9); /* 5 + 4 */
}

TEST_CASE("CharNgramCounter(1, 3)") {
  CharNgramCounter counter(1, 3);
  counter.process(py::str("abcde"));
  counter.expand_counts();
  REQUIRE(counter.get_vocab().size() == 12); /* 5 + 4 + 3 */
}

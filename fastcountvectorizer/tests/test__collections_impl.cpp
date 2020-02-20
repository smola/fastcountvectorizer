
#include <pybind11/pybind11.h>

#include "_collections.h"
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

#include "_sputils.h"
#include "catch.hpp"
#include "test_utils.h"

TEST_CASE("index_vector") {
  index_vector v;
  py::array obj;
  v.push_back(1);
  v.set_max_value(5);
  v.push_back(2);
  REQUIRE(!v.is_64());
  REQUIRE(v.data32() == std::vector<int32_t>({1, 2}));

  obj = v.to_numpy();
  REQUIRE(obj.ptr() != nullptr);
  REQUIRE(obj.c_style);

  v.set_max_value(std::numeric_limits<int32_t>::max());
  REQUIRE(!v.is_64());
  REQUIRE(v.data32() == std::vector<int32_t>({1, 2}));

  obj = v.to_numpy();
  REQUIRE(obj.ptr() != nullptr);
  REQUIRE(obj.c_style);

  if (sizeof(intptr_t) == 8) {
    v.set_max_value(((size_t)std::numeric_limits<int32_t>::max()) + 1);
    REQUIRE(v.is_64());
    REQUIRE(v.data64() == std::vector<int64_t>({1, 2}));

    obj = v.to_numpy();
    REQUIRE(obj.ptr() != nullptr);
    REQUIRE(obj.c_style);

    v.set_max_value(std::numeric_limits<int64_t>::max());
    REQUIRE(v.is_64());

    obj = v.to_numpy();
    REQUIRE(obj.ptr() != nullptr);
    REQUIRE(obj.c_style);

    v.set_max_value(1);
    REQUIRE(v.is_64());
  }
}

TEST_CASE("index_vector overflow") {
  index_vector v;

  if (sizeof(intptr_t) == 4) {
    REQUIRE_THROWS(
        v.set_max_value(((size_t)std::numeric_limits<int32_t>::max()) + 1));
  } else if (sizeof(intptr_t) == 8) {
    REQUIRE_THROWS(
        v.set_max_value(((size_t)std::numeric_limits<int64_t>::max()) + 1));
  }
}
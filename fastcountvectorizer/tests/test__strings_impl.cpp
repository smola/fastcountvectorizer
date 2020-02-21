
#include <deque>

#include "Python.h"
#include "_strings.h"
#include "catch.hpp"
#include "test_utils.h"

void test_string_with_kind_basic(const char* STR, size_t SIZE, uint8_t KIND) {
  auto str = string_with_kind(STR, SIZE, KIND);
  REQUIRE(str == str);
  REQUIRE(str != string_with_kind("xyz", 3, 1));
  REQUIRE(str != string_with_kind(STR, SIZE, (uint8_t)(KIND + 1)));
  REQUIRE(str.size() == SIZE);
  REQUIRE(str.empty() == (SIZE == 0));
  REQUIRE(str.kind() == KIND);
}

TEST_CASE("string_with_kind basic") {
  test_string_with_kind_basic("", 0, 1);
  test_string_with_kind_basic("", 0, 2);
  test_string_with_kind_basic("", 0, 4);
  test_string_with_kind_basic("abc", 3, 2);
  test_string_with_kind_basic("abc", 3, 4);
  test_string_with_kind_basic("\0a\0b\0c", 6, 2);
  test_string_with_kind_basic("a\0b\0c\0", 6, 2);
}

TEST_CASE("string_with_kind compact") {
  REQUIRE(string_with_kind::compact("", 0, 1) == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind::compact("", 0, 2) == string_with_kind("", 0, 1));
  REQUIRE(string_with_kind::compact("abc", 3, 1) ==
          string_with_kind::compact("abc", 3, 1));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint16_t>({519}).data(), 2, 2) ==
          make_string_with_kind<uint16_t>({519}));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint16_t>({1, 2, 3}).data(), 6, 2) ==
          make_string_with_kind<uint8_t>({1, 2, 3}));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint16_t>({301, 302, 303}).data(), 6, 2) ==
          make_string_with_kind<uint16_t>({301, 302, 303}));
  REQUIRE(string_with_kind::compact(
              make_string_with_kind<uint32_t>({1, 2, 3}).data(), 12, 4) ==
          make_string_with_kind<uint8_t>({1, 2, 3}));
}

TEST_CASE("string_with_kind less") {
  for (int lk = 1; lk <= 4; lk *= 2) {
    for (int rk = 1; rk <= 4; rk *= 2) {
      REQUIRE_FALSE(string_with_kind("", 0, (uint8_t)lk) <
                    string_with_kind("", 0, (uint8_t)rk));
    }
  }

  std::vector<std::pair<string_with_kind, string_with_kind>> equal_pairs = {
      {make_string_with_kind<uint8_t>({'a'}),
       make_string_with_kind<uint8_t>({'a'})},
      {make_string_with_kind<uint8_t>({'a'}),
       make_string_with_kind<uint16_t>({'a'})},
      {make_string_with_kind<uint8_t>({'a'}),
       make_string_with_kind<uint32_t>({'a'})},
      {make_string_with_kind<uint16_t>({'a'}),
       make_string_with_kind<uint32_t>({'a'})},
      {make_string_with_kind<uint32_t>({'a'}),
       make_string_with_kind<uint32_t>({'a'})},
  };

  for (const auto& it : equal_pairs) {
    REQUIRE_FALSE(it.first < it.second);
    REQUIRE_FALSE(it.second < it.first);
  }

  std::vector<std::pair<string_with_kind, string_with_kind>> less_pairs = {
      {make_string_with_kind<uint8_t>({}),
       make_string_with_kind<uint8_t>({'a'})},
      {make_string_with_kind<uint8_t>({}),
       make_string_with_kind<uint16_t>({'a'})},
      {make_string_with_kind<uint16_t>({}),
       make_string_with_kind<uint16_t>({'a'})},
      {make_string_with_kind<uint32_t>({}),
       make_string_with_kind<uint16_t>({'a'})},
      {make_string_with_kind<uint8_t>({'a'}),
       make_string_with_kind<uint8_t>({'b'})},
      {make_string_with_kind<uint8_t>({'a'}),
       make_string_with_kind<uint8_t>({'a', 'a'})},
      {make_string_with_kind<uint8_t>({'a'}),
       make_string_with_kind<uint16_t>({1020})},
      {make_string_with_kind<uint8_t>({'a'}),
       make_string_with_kind<uint32_t>({70020})},
      {make_string_with_kind<uint16_t>({1020}),
       make_string_with_kind<uint16_t>({1021})},
      {make_string_with_kind<uint16_t>({1020}),
       make_string_with_kind<uint16_t>({1020, 1020})},
      {make_string_with_kind<uint16_t>({1020}),
       make_string_with_kind<uint32_t>({1020, 70020})},
  };

  for (const auto& it : less_pairs) {
    REQUIRE(it.first < it.second);
    REQUIRE_FALSE(it.second < it.first);
  }
}

TEST_CASE("py::str to string_with_kind") {
  REQUIRE(static_cast<string_with_kind>(py::str("")) ==
          string_with_kind("", 0, 1));
  REQUIRE(static_cast<string_with_kind>(py::str("abc")) ==
          string_with_kind("abc", 3, 1));
}

TEST_CASE("string_with_kind to py::str") {
  REQUIRE(static_cast<py::str>(string_with_kind("", 0, 1)).equal(py::str("")));
  REQUIRE(static_cast<py::str>(string_with_kind("abc", 3, 1))
              .equal(py::str("abc")));

  string_with_kind str = make_string_with_kind<uint16_t>({0x30A1});
  py::str pystr = static_cast<py::str>(str);
  REQUIRE(static_cast<string_with_kind>(pystr) == str);
}

TEST_CASE("string_with_kind::join") {
  std::deque<string_with_kind> lst;
  lst.emplace_back(static_cast<string_with_kind>(py::str("")));
  REQUIRE(string_with_kind::join(lst.cbegin(), lst.cend(), lst.size()) ==
          make_string_with_kind<uint8_t>({}));

  lst.emplace_back(static_cast<string_with_kind>(py::str("")));
  REQUIRE(string_with_kind::join(lst.cbegin(), lst.cend(), lst.size()) ==
          make_string_with_kind<uint8_t>({' '}));

  lst.emplace_back(static_cast<string_with_kind>(py::str("abc")));
  REQUIRE(string_with_kind::join(lst.cbegin(), lst.cend(), lst.size()) ==
          make_string_with_kind<uint8_t>({' ', ' ', 'a', 'b', 'c'}));

  lst.emplace_back(make_string_with_kind<uint16_t>({0x9d50}));
  REQUIRE(
      string_with_kind::join(lst.cbegin(), lst.cend(), lst.size()) ==
      make_string_with_kind<uint16_t>({' ', ' ', 'a', 'b', 'c', ' ', 0x9d50}));

  lst.emplace_back(make_string_with_kind<uint32_t>({0x2F804}));
  REQUIRE(string_with_kind::join(lst.cbegin(), lst.cend(), lst.size()) ==
          make_string_with_kind<uint32_t>(
              {' ', ' ', 'a', 'b', 'c', ' ', 0x9d50, ' ', 0x2F804}));
}

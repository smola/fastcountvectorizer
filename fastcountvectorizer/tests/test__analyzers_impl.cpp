#include "_analyzers.h"
#include "catch.hpp"
#include "test_utils.h"

TEST_CASE("char analyzer prefix") {
  ngram_analyzer* analyzer = ngram_analyzer::make("char");

  REQUIRE(analyzer->prefix(0, py::str("")) == string_with_kind("", 0, 1));
  REQUIRE(analyzer->prefix(1, py::str("")) == string_with_kind("", 0, 1));
  REQUIRE(analyzer->prefix(2, py::str("")) == string_with_kind("", 0, 1));

  REQUIRE(analyzer->prefix(0, py::str("a")) == string_with_kind("", 0, 1));
  REQUIRE(analyzer->prefix(1, py::str("a")) == string_with_kind("a", 1, 1));
  REQUIRE(analyzer->prefix(2, py::str("a")) == string_with_kind("a", 1, 1));

  REQUIRE(analyzer->prefix(0, py::str("ab")) == string_with_kind("", 0, 1));
  REQUIRE(analyzer->prefix(1, py::str("ab")) == string_with_kind("a", 1, 1));
  REQUIRE(analyzer->prefix(2, py::str("ab")) == string_with_kind("ab", 2, 1));
  REQUIRE(analyzer->prefix(3, py::str("ab")) == string_with_kind("ab", 2, 1));

  delete analyzer;
}

TEST_CASE("char analyzer suffix") {
  ngram_analyzer* analyzer = ngram_analyzer::make("char");

  REQUIRE(analyzer->suffix(string_with_kind("", 0, 1)) ==
          string_with_kind("", 0, 1));
  REQUIRE(analyzer->suffix(string_with_kind("a", 1, 1)) ==
          string_with_kind("", 0, 1));
  REQUIRE(analyzer->suffix(string_with_kind("ab", 2, 1)) ==
          string_with_kind("b", 1, 1));

  delete analyzer;
}

TEST_CASE("char analyzer prefix_ngrams") {
  ngram_analyzer* analyzer = ngram_analyzer::make("char");

  REQUIRE(analyzer->prefix_ngrams(string_with_kind("", 0, 1), 1, 1) ==
          std::vector<string_with_kind>({}));
  REQUIRE(analyzer->prefix_ngrams(string_with_kind("", 0, 1), 1, 2) ==
          std::vector<string_with_kind>({}));

  REQUIRE(analyzer->prefix_ngrams(string_with_kind("a", 1, 1), 1, 1) ==
          std::vector<string_with_kind>({string_with_kind("a", 1, 1)}));
  REQUIRE(analyzer->prefix_ngrams(string_with_kind("a", 1, 1), 1, 2) ==
          std::vector<string_with_kind>({string_with_kind("a", 1, 1)}));
  REQUIRE(analyzer->prefix_ngrams(string_with_kind("a", 1, 1), 2, 2) ==
          std::vector<string_with_kind>({}));

  REQUIRE(analyzer->prefix_ngrams(string_with_kind("ab", 2, 1), 1, 2) ==
          std::vector<string_with_kind>({string_with_kind("a", 1, 1),
                                         string_with_kind("b", 1, 1),
                                         string_with_kind("ab", 2, 1)}));

  delete analyzer;
}
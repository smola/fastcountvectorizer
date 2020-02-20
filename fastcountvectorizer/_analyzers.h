
#ifndef FCV_ANALYZERS_H
#define FCV_ANALYZERS_H

#include <pybind11/pybind11.h>

#include <iterator>

#include "_collections.h"

class ngram_analysis_counts {
 public:
  virtual ~ngram_analysis_counts(){};
  virtual bool next() = 0;
  virtual py::str pykey() const = 0;
  virtual string_with_kind key() const = 0;
  virtual std::int64_t count() const = 0;
  virtual std::size_t size() const = 0;
};

class ngram_analyzer {
 public:
  virtual ~ngram_analyzer(){};
  virtual ngram_analysis_counts* analyze(unsigned int n,
                                         const py::str& doc) const = 0;
  virtual string_with_kind prefix(unsigned int n, const py::str& doc) const = 0;
  virtual string_with_kind suffix(const string_with_kind& s) const = 0;
  virtual string_with_kind ngram(const string_with_kind& s, unsigned int n,
                                 std::size_t pos) const = 0;

  static ngram_analyzer* make(const std::string& type);
};

template <class Counters>
class base_ngram_analysis_counts : public ngram_analysis_counts {
 protected:
  Counters counters;
  bool started;
  typename Counters::const_iterator it;

  explicit base_ngram_analysis_counts(Counters counters)
      : counters(std::move(counters)), started(false) {}

 public:
  ~base_ngram_analysis_counts() override = default;

  bool next() override {
    if (!started) {
      started = true;
      it = counters.cbegin();
      return it != counters.cend();
    }

    it++;
    return it != counters.cend();
  }

  std::int64_t count() const override { return it.value(); }

  std::size_t size() const override { return counters.size(); };
};

template <class Analysis, class PrefixHandler>
class base_ngram_analyzer : public ngram_analyzer {
 public:
  base_ngram_analyzer() = default;
  ~base_ngram_analyzer() override = default;

  ngram_analysis_counts* analyze(unsigned int n,
                                 const py::str& doc) const override {
    return new Analysis(n, doc);
  }

  string_with_kind prefix(unsigned int n, const py::str& doc) const override {
    return PrefixHandler::prefix(n, doc);
  }

  string_with_kind suffix(const string_with_kind& s) const override {
    return PrefixHandler::suffix(s);
  }
  string_with_kind ngram(const string_with_kind& s, unsigned int n,
                         std::size_t pos) const override {
    return PrefixHandler::ngram(s, n, pos);
  }
};

class char_ngram_prefix_handler {
 public:
  static string_with_kind prefix(unsigned int n, const py::str& doc);
  static string_with_kind suffix(const string_with_kind& s);
  static string_with_kind ngram(const string_with_kind& s, unsigned int n,
                                std::size_t pos);
};

class char_wb_ngram_prefix_handler : public char_ngram_prefix_handler {
 public:
  static string_with_kind prefix(unsigned int n, const py::str& doc);
};

class char_ngram_analysis_counts
    : public base_ngram_analysis_counts<counter_map> {
 private:
  unsigned int n;
  uint8_t kind;

  char_ngram_analysis_counts(unsigned int n, const py::str& doc, uint8_t kind);

 public:
  char_ngram_analysis_counts(unsigned int n, const py::str& doc)
      : char_ngram_analysis_counts(n, doc, (uint8_t)PyUnicode_KIND(doc.ptr())) {
  }
  ~char_ngram_analysis_counts() override = default;

  py::str pykey() const override;
  string_with_kind key() const override;
};

class char_wb_ngram_analysis_counts
    : public base_ngram_analysis_counts<string_with_kind_counter_map> {
 private:
  char_wb_ngram_analysis_counts(unsigned int n, const py::str& doc,
                                uint8_t kind);

 public:
  char_wb_ngram_analysis_counts(unsigned int n, const py::str& doc)
      : char_wb_ngram_analysis_counts(n, doc,
                                      (uint8_t)PyUnicode_KIND(doc.ptr())) {}
  ~char_wb_ngram_analysis_counts() override = default;

  py::str pykey() const override;
  string_with_kind key() const override;
};

#endif  // FCV_ANALYZERS_H

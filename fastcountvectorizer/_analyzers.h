
#ifndef FCV_ANALYZERS_H
#define FCV_ANALYZERS_H

#include <pybind11/pybind11.h>

#include <vector>

#include "_collections.h"

class ngram_analysis_counts {
 public:
  virtual ~ngram_analysis_counts() = default;
  virtual bool next() = 0;
  virtual py::str pykey() const = 0;
  virtual string_with_kind key() const = 0;
  virtual std::int64_t count() const = 0;
  virtual std::size_t size() const = 0;
};

class ngram_analyzer {
 public:
  virtual ~ngram_analyzer() = default;
  virtual ngram_analysis_counts* analyze(unsigned int n,
                                         const py::str& doc) const = 0;
  virtual string_with_kind prefix(unsigned int n, const py::str& doc) const = 0;
  virtual string_with_kind suffix(const string_with_kind& s) const = 0;
  virtual std::vector<string_with_kind> prefix_ngrams(
      const string_with_kind& s, unsigned int min_n,
      unsigned int max_n) const = 0;

  static ngram_analyzer* make(const std::string& type, py::object stop_words);
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

template <class PrefixHandler>
class base_ngram_analyzer : public ngram_analyzer {
 protected:
  PrefixHandler prefix_handler;

 public:
  explicit base_ngram_analyzer(PrefixHandler prefix_handler = PrefixHandler())
      : prefix_handler(prefix_handler) {}
  ~base_ngram_analyzer() override = default;

  string_with_kind prefix(const unsigned int n,
                          const py::str& doc) const override {
    return prefix_handler.prefix(n, doc);
  }

  string_with_kind suffix(const string_with_kind& s) const override {
    return prefix_handler.suffix(s);
  }

  std::vector<string_with_kind> prefix_ngrams(
      const string_with_kind& s, const unsigned int min_n,
      const unsigned int max_n) const override {
    return prefix_handler.prefix_ngrams(s, min_n, max_n);
  }
};

class char_ngram_prefix_handler {
 public:
  virtual string_with_kind prefix(unsigned int n, const py::str& doc) const;
  string_with_kind suffix(const string_with_kind& s) const;
  std::vector<string_with_kind> prefix_ngrams(const string_with_kind& s,
                                              unsigned int min_n,
                                              unsigned int max_n) const;
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

class char_ngram_analyzer
    : public base_ngram_analyzer<char_ngram_prefix_handler> {
  ngram_analysis_counts* analyze(unsigned int n,
                                 const py::str& doc) const override {
    return new char_ngram_analysis_counts(n, doc);
  }
};

class word_ngram_analysis_counts
    : public base_ngram_analysis_counts<string_with_kind_counter_map> {
 private:
  word_ngram_analysis_counts(unsigned int n, const py::str& doc, uint8_t kind,
                             const py::object& token_pattern,
                             const py::object& stop_words);

 public:
  word_ngram_analysis_counts(unsigned int n, const py::str& doc,
                             const py::object& token_pattern,
                             const py::object& stop_words)
      : word_ngram_analysis_counts(n, doc, (uint8_t)PyUnicode_KIND(doc.ptr()),
                                   token_pattern, stop_words) {}
  ~word_ngram_analysis_counts() override = default;

  py::str pykey() const override;
  string_with_kind key() const override;
};

class word_ngram_prefix_handler {
 private:
  py::object re_token_pattern;
  py::object stop_words;

 public:
  word_ngram_prefix_handler(py::object stop_words);
  string_with_kind prefix(unsigned int n, const py::str& doc) const;
  string_with_kind suffix(const string_with_kind& s) const;
  std::vector<string_with_kind> prefix_ngrams(const string_with_kind& s,
                                              unsigned int min_n,
                                              unsigned int max_n) const;
};

class word_ngram_analyzer
    : public base_ngram_analyzer<word_ngram_prefix_handler> {
 private:
  py::object re_token_pattern;
  py::object stop_words;

 public:
  explicit word_ngram_analyzer(py::object stop_words);
  ngram_analysis_counts* analyze(unsigned int n,
                                 const py::str& doc) const override {
    return new word_ngram_analysis_counts(n, doc, re_token_pattern, stop_words);
  }
};

#endif  // FCV_ANALYZERS_H

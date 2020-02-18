
#ifndef FCV_SPUTILS_H
#define FCV_SPUTILS_H

#include <cstdint>
#include <vector>

// vector_to_numpy converts a std::vector to a numpy.array.
template <typename I>
PyObject* vector_to_numpy(const std::vector<I>& v);

// index_vector is an integer vector that uses initially int32_t and switches
// dynamically to int32_t when needed. This is useful for arrays to be used as
// scipy.sparse indexes, which will use int32 when possible and will trigger an
// index copy if int64 is used when the maximum number fits in an int32.
//
// Limits come from numpy, not cstdint.
class index_vector {
 private:
  std::vector<std::int32_t>* v32;
  std::vector<std::int64_t>* v64;
  bool use_64;

  explicit index_vector(bool use_64);

 public:
  index_vector();
  ~index_vector();
  void set_max_value(std::size_t val);
  void set_max_value(const std::vector<std::size_t>& vals);
  void reserve(std::size_t n);
  void push_back(std::size_t n);
  std::size_t size() const;
  PyObject* to_numpy() const;
  bool is_64() const { return use_64; }
  std::vector<std::int32_t>& data32() { return *v32; }
  const std::vector<std::int32_t>& data32() const { return *v32; }
  std::vector<std::int64_t>& data64() { return *v64; }
  const std::vector<std::int64_t>& data64() const { return *v64; }
  std::int64_t operator[](std::size_t i) const;
};

#endif  // FCV_SPUTILS_H
//
// This file is derived from scipy's scipy/sparse/sparsetools/csr.h version
// 546e458 (Mar 11, 2019). The original file is released under a BSD-3 clause
// license.
//
// The original license text is:
//
// Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef FCV_CSR_H
#define FCV_CSR_H

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "_sputils.h"

template <class T>
constexpr void assert_valid_index_type() {
  static_assert(std::is_integral<T>(), "index must be integral");
  static_assert(sizeof(T) <= sizeof(size_t),
                "index type cannot be larger than size_t");
  static_assert(
      std::numeric_limits<T>::max() < std::numeric_limits<size_t>::max(),
      "index maximum value must be lower than size_t maximum value");
}

//
// D = A * B + C
//
// Arguments:
//   n_row, n_col: D rows and columns
//   Ap: A indptr
//   Aj: A column (j) indices
//   Bj: B indices
//   nnz_per_row: nnz per B row
//   Cp: C indptr
//   Cj: C column (j) indices
//
// - All B values are assumed to be 1.
// - B left-side is a diagonal matrix of n_row size.
// - All B rows have a fixed number of nnz (nnz_per_row).
// - C has all non-zero elements set to 1.
//
template <class TA, class TB, class TC>
size_t csr_matmat_add_pass1_diagprefix_fixed_nnz(
    const size_t n_row, const size_t n_col, const std::vector<TA>& Ap,
    const std::vector<TA>& Aj, const std::vector<TB>& Bj,
    const size_t nnz_per_row, const std::vector<TC>& Cp,
    const std::vector<TC>& Cj) {
  assert_valid_index_type<TA>();
  assert_valid_index_type<TB>();
  assert_valid_index_type<TC>();
  assert(n_row <= n_col);

  // method that uses O(n) temp storage
  std::vector<std::size_t> mask(n_col - n_row, (std::size_t)-1);

  std::size_t nnz = 0;
  for (size_t i = 0; i < n_row; i++) {
    size_t row_nnz = 0;

    std::size_t jj_start = (size_t)Ap[i];
    std::size_t jj_end = (size_t)Ap[i + 1];

    // Because B has the diagonal prefix, D left side contains A.
    row_nnz += jj_end - jj_start;

    // A*B
    for (std::size_t jj = jj_start; jj < jj_end; jj++) {
      std::size_t j = (size_t)Aj[jj];

      for (std::size_t kk = j * nnz_per_row; kk < (j + 1) * nnz_per_row; kk++) {
        std::size_t k = (std::size_t)Bj[kk] - n_row;
        if (mask[k] != i) {
          mask[k] = i;
          row_nnz++;
        }
      }
    }

    // +C
    jj_start = (std::size_t)Cp[i];
    jj_end = (std::size_t)Cp[i + 1];
    for (std::size_t jj = jj_start; jj < jj_end; jj++) {
      std::size_t j = (std::size_t)Cj[jj] - n_row;
      assert(j < mask.size());  // DEBUG
      if (mask[j] != i) {
        mask[j] = i;
        row_nnz++;
      }
    }

    size_t next_nnz = nnz + row_nnz;

    if (row_nnz > std::numeric_limits<intptr_t>::max() - nnz) {
      /*
       * Index overflowed. Note that row_nnz <= n_col and cannot overflow
       */
      throw std::overflow_error("nnz of the result is too large");
    }

    nnz = next_nnz;
  }

  return nnz;
}

// D = A*B+C pass 2
// See pass 1 for assumptions on inputs.
template <class TA, class TAD, class TB, class TC, class TD, class TDD>
void csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
    const size_t nnz, const size_t n_row, const size_t n_col,
    const std::vector<TA>& Ap, const std::vector<TA>& Aj,
    const std::vector<TAD>& Ax, const std::vector<TB>& Bj,
    const size_t nnz_per_row, const std::vector<TC>& Cp,
    const std::vector<TC>& Cj, std::vector<TD>& Dp, std::vector<TD>& Dj,
    std::vector<TDD>& Dx) {
  assert_valid_index_type<TA>();
  assert_valid_index_type<TB>();
  assert_valid_index_type<TC>();
  assert(n_row <= n_col);

  Dj.resize(nnz);
  Dx.resize(nnz);
  Dp.resize(n_row + 1);
  Dp[0] = 0;
  size_t ptr = 0;

  std::vector<TD> next(n_col - n_row, -1);
  std::vector<TDD> sums(n_col - n_row, 0);

  for (size_t i = 0; i < n_row; i++) {
    TD head = -2;
    size_t length = 0;

    size_t jj_start = (size_t)Ap[i];
    size_t jj_end = (size_t)Ap[i + 1];
    for (size_t jj = jj_start; jj < jj_end; jj++) {
      size_t j = (size_t)Aj[jj];
      TAD v = Ax[jj];

      // diagonal prefix
      Dj[ptr] = (TD)j;
      Dx[ptr] = v;
      ptr++;

      // A*B
      for (size_t kk = j * nnz_per_row; kk < (j + 1) * nnz_per_row; kk++) {
        size_t k = (size_t)Bj[kk] - n_row;

        sums[k] += v;

        if (next[k] == -1) {
          next[k] = head;
          head = (TD)k;
          length++;
        }
      }
    }

    // +C
    jj_start = (size_t)Cp[i];
    jj_end = (size_t)Cp[i + 1];
    for (size_t jj = jj_start; jj < jj_end; jj++) {
      size_t j = (size_t)Cj[jj] - n_row;
      sums[j]++;
      if (next[j] == -1) {
        next[j] = head;
        head = (TD)j;
        length++;
      }
    }

    for (size_t jj = 0; jj < length; jj++) {
      Dj[ptr] = head + (TD)n_row;
      Dx[ptr] = (TDD)sums[(size_t)head];
      ptr++;

      TD temp = head;
      head = next[(size_t)head];

      next[(size_t)temp] = -1;  // clear arrays
      sums[(size_t)temp] = 0;
    }

    Dp[i + 1] = (TD)ptr;
  }
}

template <class TB, class TC>
std::size_t csr_matmat_add_pass1_diagprefix_fixed_nnz(
    const std::size_t n_row, const std::size_t n_col, const index_vector& Ap,
    const index_vector& Aj, const std::vector<TB>& Bj,
    const std::size_t nnz_per_row, const std::vector<TC>& Cp,
    const std::vector<TC>& Cj) {
#if SIZEOF_SIZE_T == 8
  if (Aj.is_64()) {
    return csr_matmat_add_pass1_diagprefix_fixed_nnz(
        n_row, n_col, Ap.data64(), Aj.data64(), Bj, nnz_per_row, Cp, Cj);
  }
#endif
  return csr_matmat_add_pass1_diagprefix_fixed_nnz(
      n_row, n_col, Ap.data32(), Aj.data32(), Bj, nnz_per_row, Cp, Cj);
}

template <class TAD, class TB, class TC, class TDD>
void csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
    const std::size_t nnz, const std::size_t n_row, const std::size_t n_col,
    const index_vector& Ap, const index_vector& Aj, const std::vector<TAD>& Ax,
    const std::vector<TB>& Bj, const size_t nnz_per_row,
    const std::vector<TC>& Cp, const std::vector<TC>& Cj, index_vector& Dp,
    index_vector& Dj, std::vector<TDD>& Dx) {
#if SIZEOF_SIZE_T == 8
  const bool A64 = Aj.is_64();
  const bool D64 = Dj.is_64();
  if (A64 && D64) {
    csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
        nnz, n_row, n_col, Ap.data64(), Aj.data64(), Ax, Bj, nnz_per_row, Cp,
        Cj, Dp.data64(), Dj.data64(), Dx);
  } else if (A64 && !D64) {
    csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
        nnz, n_row, n_col, Ap.data64(), Aj.data64(), Ax, Bj, nnz_per_row, Cp,
        Cj, Dp.data32(), Dj.data32(), Dx);
  } else if (!A64 && D64) {
    csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
        nnz, n_row, n_col, Ap.data32(), Aj.data32(), Ax, Bj, nnz_per_row, Cp,
        Cj, Dp.data64(), Dj.data64(), Dx);
  } else {
    csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
        nnz, n_row, n_col, Ap.data32(), Aj.data32(), Ax, Bj, nnz_per_row, Cp,
        Cj, Dp.data32(), Dj.data32(), Dx);
  }
#else
  csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
      nnz, n_row, n_col, Ap.data32(), Aj.data32(), Ax, Bj, nnz_per_row, Cp, Cj,
      Dp.data32(), Dj.data32(), Dx);
#endif
}

template <class TAD, class TB, class TC, class TDD>
void csr_matmat_add_Bx1_diagprefix_fixed_nnz(
    const std::size_t n_row, const std::size_t n_col, const index_vector& Ap,
    const index_vector& Aj, const std::vector<TAD>& Ax,
    const std::vector<TB>& Bj, const std::size_t nnz_per_row,
    const std::vector<TC>& Cp, const std::vector<TC>& Cj, index_vector& Dp,
    index_vector& Dj, std::vector<TDD>& Dx) {
  auto nnz = csr_matmat_add_pass1_diagprefix_fixed_nnz(n_row, n_col, Ap, Aj, Bj,
                                                       nnz_per_row, Cp, Cj);

  Dp.set_max_value({n_row, n_col, nnz});
  Dj.set_max_value({n_row, n_col, nnz});

  csr_matmat_add_pass2_Bx1_diagprefix_fixed_nnz(
      nnz, n_row, n_col, Ap, Aj, Ax, Bj, nnz_per_row, Cp, Cj, Dp, Dj, Dx);
}

template <class T>
inline std::size_t transform_indices_pass1(
    const std::vector<std::int64_t>& transformation, const std::vector<T>& Aj) {
  std::size_t nnz = 0;
  for (std::size_t i = 0; i < Aj.size(); i++) {
    if (transformation[(std::size_t)Aj[i]] >= 0) {
      nnz++;
    }
  }
  return nnz;
}

inline std::size_t transform_indices_pass1(
    const std::vector<std::int64_t>& transformation, const index_vector& Aj) {
#if SIZEOF_SIZE_T == 8
  if (Aj.is_64()) {
    return transform_indices_pass1(transformation, Aj.data64());
  }
#endif
  return transform_indices_pass1(transformation, Aj.data32());
}

template <class I, class T, class D>
inline void transform_indices_pass2(
    const std::size_t maxnnz, const std::vector<std::int64_t>& transformation,
    const std::vector<I>& Ap, const std::vector<I>& Aj,
    const std::vector<D>& Ax, std::vector<T>& Bp, std::vector<T>& Bj,
    std::vector<D>& Bx) {
  std::size_t nnz = 0;
  Bj.resize(maxnnz);
  Bx.resize(maxnnz);
  Bp.resize(Ap.size());
  Bp[0] = 0;
  for (std::size_t i = 0; i < Ap.size() - 1; i++) {
    const auto ii_start = (std::size_t)Ap[i];
    const auto ii_end = (std::size_t)Ap[i + 1];
    for (std::size_t ii = ii_start; ii < ii_end; ii++) {
      const auto j = (std::size_t)Aj[ii];
      const std::int64_t new_j = transformation[j];
      if (new_j < 0) {
        continue;
      }
      Bj[nnz] = (T)new_j;
      Bx[nnz] = Ax[ii];
      nnz++;
    }
    Bp[i + 1] = (T)nnz;
  }
}

template <class D>
inline void transform_indices_pass2(
    const std::size_t maxnnz, const std::vector<std::int64_t>& transformation,
    const index_vector& Ap, const index_vector& Aj, const std::vector<D>& Ax,
    index_vector& Bp, index_vector& Bj, std::vector<D>& Bx) {
#if SIZEOF_SIZE_T == 8
  const bool A64 = Aj.is_64();
  const bool B64 = Bj.is_64();
  if (A64 && B64) {
    transform_indices_pass2(maxnnz, transformation, Ap.data64(), Aj.data64(),
                            Ax, Bp.data64(), Bj.data64(), Bx);
  } else if (A64 && !B64) {
    transform_indices_pass2(maxnnz, transformation, Ap.data64(), Aj.data64(),
                            Ax, Bp.data32(), Bj.data32(), Bx);
  } else if (!A64 && B64) {
    transform_indices_pass2(maxnnz, transformation, Ap.data32(), Aj.data32(),
                            Ax, Bp.data64(), Bj.data64(), Bx);
  } else {
    transform_indices_pass2(maxnnz, transformation, Ap.data32(), Aj.data32(),
                            Ax, Bp.data32(), Bj.data32(), Bx);
  }
#else
  transform_indices_pass2(maxnnz, transformation, Ap.data32(), Aj.data32(), Ax,
                          Bp.data32(), Bj.data32(), Bx);
#endif
}

template <class I, class T>
void transform_indices(const std::size_t maxidx,
                       const std::vector<std::int64_t>& transformation,
                       const index_vector& Ap, const index_vector& Aj,
                       const std::vector<I>& Ax, index_vector& Bp,
                       index_vector& Bj, std::vector<T>& Bx) {
  const std::size_t nnz = transform_indices_pass1(transformation, Aj);
  Bp.set_max_value({maxidx, nnz, Ap.size()});
  Bj.set_max_value({maxidx, nnz, Ap.size()});
  transform_indices_pass2(nnz, transformation, Ap, Aj, Ax, Bp, Bj, Bx);
}

#endif  // FCV_CSR_H
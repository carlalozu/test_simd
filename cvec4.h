// This file is part of the portable parton-level event generator Pepper.
// Copyright (C) 2023-2025 The Pepper Collaboration
// Pepper is licensed under version 3 of the GPL, see COPYING for details.
// Please respect the MCnet academic usage guidelines, see GUIDELINES.

#ifndef PEPPER_CVEC4_H
#define PEPPER_CVEC4_H

#include "math.h"
#include "vec4.h"
#include <Kokkos_SIMD.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
using simd_double = Kokkos::Experimental::simd<double>;
using simd_mask_double = Kokkos::Experimental::simd_mask<double>;

using CSIMD = std::complex<simd_double>;

template <typename C> struct CVec4_T : std::array<C, 4> {

  // Make base class constructors available
  using std::array<C, 4>::array;

  template <typename T>
  KOKKOS_INLINE_FUNCTION CVec4_T(const Vec4_T<T>& v)
      : std::array<C, 4> {v[0], v[1], v[2], v[3]} {};

  template <typename T>
  KOKKOS_INLINE_FUNCTION CVec4_T(const Vec4_T<T>& vr, const Vec4_T<T>& vi)
      : std::array<C, 4> {C {vr[0], vi[0]}, C {vr[1], vi[1]}, C {vr[2], vi[2]},
                          C {vr[3], vi[3]}} {};

  KOKKOS_INLINE_FUNCTION CVec4_T(const C& i0, const C& i1, const C& i2,
                                 const C& i3)
      : std::array<C, 4> {i0, i1, i2, i3}
  {
  }

  template <typename T> KOKKOS_INLINE_FUNCTION Vec4_T<T> real() const
  {
    return {(*this)[0].real(), (*this)[1].real(), (*this)[2].real(),
            (*this)[3].real()};
  }

  template <typename T> KOKKOS_INLINE_FUNCTION Vec4_T<T> imag() const
  {
    return {(*this)[0].imag(), (*this)[1].imag(), (*this)[2].imag(),
            (*this)[3].imag()};
  }

  KOKKOS_INLINE_FUNCTION C abs() const { return Complex::sqrt(abs2()); };
  KOKKOS_INLINE_FUNCTION C abs2() const { return (*this) * (*this); };
  KOKKOS_INLINE_FUNCTION C p_plus() const { return (*this)[0] + (*this)[3]; }
  KOKKOS_INLINE_FUNCTION C p_minus() const { return (*this)[0] - (*this)[3]; }
  KOKKOS_INLINE_FUNCTION CVec4_T& operator+=(const CVec4_T& rhs)
  {
    (*this)[0] += rhs[0];
    (*this)[1] += rhs[1];
    (*this)[2] += rhs[2];
    (*this)[3] += rhs[3];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION CVec4_T& operator-=(const CVec4_T& rhs)
  {
    (*this)[0] -= rhs[0];
    (*this)[1] -= rhs[1];
    (*this)[2] -= rhs[2];
    (*this)[3] -= rhs[3];
    return *this;
  }

  template <typename T> KOKKOS_INLINE_FUNCTION CVec4_T& operator*=(T rhs)
  {
    (*this)[0] *= rhs;
    (*this)[1] *= rhs;
    (*this)[2] *= rhs;
    (*this)[3] *= rhs;
    return *this;
  }

  template <typename T> KOKKOS_INLINE_FUNCTION CVec4_T& operator/=(T rhs)
  {
    (*this)[0] /= rhs;
    (*this)[1] /= rhs;
    (*this)[2] /= rhs;
    (*this)[3] /= rhs;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION void conjugate()
  {
    (*this)[0] = Complex::conj((*this)[0]);
    (*this)[1] = Complex::conj((*this)[1]);
    (*this)[2] = Complex::conj((*this)[2]);
    (*this)[3] = Complex::conj((*this)[3]);
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION friend CVec4_T operator*(CVec4_T lhs, T rhs)
  {
    return lhs *= rhs;
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION friend CVec4_T operator*(T lhs, CVec4_T rhs)
  {
    return rhs *= lhs;
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION friend CVec4_T operator/(CVec4_T lhs, T rhs)
  {
    return lhs /= rhs;
  }
  KOKKOS_INLINE_FUNCTION friend CVec4_T operator+(CVec4_T lhs,
                                                  const CVec4_T& rhs)
  {
    return lhs += rhs;
  }
  KOKKOS_INLINE_FUNCTION friend CVec4_T operator-(CVec4_T lhs,
                                                  const CVec4_T& rhs)
  {
    return lhs -= rhs;
  }
  KOKKOS_INLINE_FUNCTION friend C operator*(const CVec4_T& lhs,
                                            const CVec4_T& rhs)
  {
    return lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] -
           lhs[3] * rhs[3];
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION friend C operator*(const CVec4_T& lhs,
                                            const Vec4_T<T>& rhs)
  {
    return lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] -
           lhs[3] * rhs[3];
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION friend C operator*(const Vec4_T<T>& lhs,
                                            const CVec4_T& rhs)
  {
    return lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] -
           lhs[3] * rhs[3];
  }

  friend std::ostream& operator<<(std::ostream& o, const CVec4_T& v)
  {
    return o << "(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3]
             << ")";
  }

  KOKKOS_INLINE_FUNCTION void print() const
  {
    printf("( {%10.3g,%10.3g}, {%10.3g,%10.3g}, {%10.3g,%10.3g}, "
           "{%10.3g,%10.3g} )",
           (*this)[0].real(), (*this)[0].imag(), (*this)[1].real(),
           (*this)[1].imag(), (*this)[2].real(), (*this)[2].imag(),
           (*this)[3].real(), (*this)[3].imag());
  }

  KOKKOS_INLINE_FUNCTION void print_simd() const
  {

    std::cout << "{";
    for (std::size_t k = 0; k < 4; ++k) {
      std::cout << "Re: (";
      for (std::size_t i = 0; i < (*this)[k].real().size(); ++i) {
        std::cout << std::setw(5) << std::setprecision(3)
                  << (*this)[k].real()[i];
      }
      std::cout << ")";
      std::cout << "Im: (";
      for (std::size_t i = 0; i < (*this)[k].imag().size(); ++i) {
        std::cout << std::setw(5) << std::setprecision(3)
                  << (*this)[k].imag()[i];
      }
      std::cout << ")";
    }
    std::cout << "}" << std::endl;
  }
};

template <typename C, typename T>
KOKKOS_INLINE_FUNCTION CVec4_T<C> operator*(const C& scalar,
                                            const Vec4_T<T>& vec)
{
  return {scalar * vec[0], scalar * vec[1], scalar * vec[2], scalar * vec[3]};
};

KOKKOS_INLINE_FUNCTION CSIMD operator*(const float& scalar,
                                       const CSIMD& complex_scalar)
{
  return {scalar * complex_scalar.real(), scalar * complex_scalar.imag()};
};

// Specialization for complex vectors
using CVec4 = CVec4_T<Complex::complex<double>>;
using CVec4SIMD = CVec4_T<CSIMD>;

#endif

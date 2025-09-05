// This file is part of the portable parton-level event generator Pepper.
// Copyright (C) 2023-2025 The Pepper Collaboration
// Pepper is licensed under version 3 of the GPL, see COPYING for details.
// Please respect the MCnet academic usage guidelines, see GUIDELINES.

#ifndef PEPPER_CVEC4_H
#define PEPPER_CVEC4_H

#include "math.h"
#include "vec4_copy.h"
#include <Kokkos_SIMD.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
using simd_double = Kokkos::Experimental::simd<double>;
using simd_mask_double = Kokkos::Experimental::simd_mask<double>;
using CSIMD = std::complex<simd_double>;

/**
 * CVec4_T
 *
 * Use CVec4 without deriving from std::array
 * This seemed to give a speedup but was deemed not necessary
 * when unrolling the kernels completely.
 */
template <typename C>
struct CVec4_T
{

  C c0, c1, c2, c3;

  KOKKOS_INLINE_FUNCTION CVec4_T() : c0(0), c1(0), c2(0), c3(0) {};
  KOKKOS_INLINE_FUNCTION CVec4_T(const CVec4_T &other)
      : c0(other.c0), c1(other.c1), c2(other.c2), c3(other.c3) {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION CVec4_T(const Vec4_T<T> &v)
      : c0(v.e), c1(v.px), c2(v.py), c3(v.pz) {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION CVec4_T(const Vec4_T<T> &vr, const Vec4_T<T> &vi)
      : c0(C{vr.e, vi.e}),
        c1(C{vr.px, vi.px}),
        c2(C{vr.py, vi.py}),
        c3(C{vr.pz, vi.pz}) {}

  KOKKOS_INLINE_FUNCTION CVec4_T(const C &i0, const C &i1, const C &i2, const C &i3)
      : c0(i0), c1(i1), c2(i2), c3(i3) {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION Vec4_T<T> real() const
  {
    return {c0.real(), c1.real(), c2.real(), c3.real()};
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION Vec4_T<T> imag() const
  {
    return {c0.imag(), c1.imag(), c2.imag(), c3.imag()};
  }

  KOKKOS_INLINE_FUNCTION C abs() const { return Complex::sqrt(abs2()); };
  KOKKOS_INLINE_FUNCTION C abs2() const { return (*this) * (*this); };
  KOKKOS_INLINE_FUNCTION C p_plus() const { return c0 + c3; }
  KOKKOS_INLINE_FUNCTION C p_minus() const { return c0 - c3; }
  KOKKOS_INLINE_FUNCTION CVec4_T &operator+=(const CVec4_T &rhs)
  {
    c0 += rhs.c0;
    c1 += rhs.c1;
    c2 += rhs.c2;
    c3 += rhs.c3;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION CVec4_T &operator-=(const CVec4_T &rhs)
  {
    c0 -= rhs.c0;
    c1 -= rhs.c1;
    c2 -= rhs.c2;
    c3 -= rhs.c3;
    return *this;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION CVec4_T &operator*=(T rhs)
  {
    c0 *= rhs;
    c1 *= rhs;
    c2 *= rhs;
    c3 *= rhs;
    return *this;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION CVec4_T &operator/=(T rhs)
  {
    c0 /= rhs;
    c1 /= rhs;
    c2 /= rhs;
    c3 /= rhs;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION void conjugate()
  {
    c0 = Complex::conj(c0);
    c1 = Complex::conj(c1);
    c2 = Complex::conj(c2);
    c3 = Complex::conj(c3);
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
                                                  const CVec4_T &rhs)
  {
    return lhs += rhs;
  }
  KOKKOS_INLINE_FUNCTION friend CVec4_T operator-(CVec4_T lhs,
                                                  const CVec4_T &rhs)
  {
    return lhs -= rhs;
  }
  KOKKOS_INLINE_FUNCTION friend C operator*(const CVec4_T &lhs,
                                            const CVec4_T &rhs)
  {
    return lhs.c0 * rhs.c0 - lhs.c1 * rhs.c1 - lhs.c2 * rhs.c2 -
           lhs.c3 * rhs.c3;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION friend C operator*(const CVec4_T &lhs,
                                            const Vec4_T<T> &rhs)
  {
    return lhs.c0 * rhs.e - lhs.c1 * rhs.px - lhs.c2 * rhs.py -
           lhs.c3 * rhs.pz;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION friend C operator*(const Vec4_T<T> &lhs,
                                            const CVec4_T &rhs)
  {
    return lhs.e * rhs.c0 - lhs.px * rhs.c1 - lhs.py * rhs.c2 -
           lhs.pz * rhs.c3;
  }

  friend std::ostream &operator<<(std::ostream &o, const CVec4_T &v)
  {
    // return o << "(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << ")";
    return o << "( {" << v.c0.real() << "," << v.c0.imag() << "}, {"
             << v.c1.real() << "," << v.c1.imag() << "}, {" << v.c2.real()
             << "," << v.c2.imag() << "}, {" << v.c3.real() << ","
             << v.c3.imag() << "} )";
  }

  KOKKOS_INLINE_FUNCTION void print() const
  {
    printf("( {%10.3g,%10.3g}, {%10.3g,%10.3g}, {%10.3g,%10.3g}, "
           "{%10.3g,%10.3g} )",
           c0.real(), c0.imag(), c1.real(),
           c1.imag(), c2.real(), c2.imag(),
           c3.real(), c3.imag());
  }

  KOKKOS_INLINE_FUNCTION void print_simd() const
  {

    std::cout << "{";
    for (std::size_t k = 0; k < 4; ++k)
    {
      std::cout << "Re: (";
      for (std::size_t i = 0; i < (*this)[k].real().size(); ++i)
      {
        std::cout << std::setw(5) << std::setprecision(3)
                  << (*this)[k].real()[i];
      }
      std::cout << ")";
      std::cout << "Im: (";
      for (std::size_t i = 0; i < (*this)[k].imag().size(); ++i)
      {
        std::cout << std::setw(5) << std::setprecision(3)
                  << (*this)[k].imag()[i];
      }
      std::cout << ")";
    }
    std::cout << "}" << std::endl;
  }
};

template <typename C, typename T>
KOKKOS_INLINE_FUNCTION CVec4_T<C> operator*(const C &scalar,
                                            const Vec4_T<T> &vec)
{
  return {scalar * vec.e, scalar * vec.px, scalar * vec.py, scalar * vec.pz};
};

KOKKOS_INLINE_FUNCTION CSIMD operator*(const float &scalar,
                                       const CSIMD &complex_scalar)
{
  return {scalar * complex_scalar.real(), scalar * complex_scalar.imag()};
};

// Specialization for complex vectors
using CVec4 = CVec4_T<Complex::complex<double>>;
using CVec4SIMD = CVec4_T<CSIMD>;

#endif

// This file is part of the portable parton-level event generator Pepper.
// Copyright (C) 2023-2025 The Pepper Collaboration
// Pepper is licensed under version 3 of the GPL, see COPYING for details.
// Please respect the MCnet academic usage guidelines, see GUIDELINES.

#ifndef PEPPER_VEC4_H
#define PEPPER_VEC4_H

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>

namespace Complex = Kokkos;

/**
 * Vec4_T
 *
 * Use Vec4 without deriving from std::array
 * This seemed to give a speedup but was deemed not necessary
 * when unrolling the kernels completely.
 */
template <typename T>
struct Vec4_T
{

  T e, px, py, pz;

  KOKKOS_INLINE_FUNCTION Vec4_T() : e(0), px(0), py(0), pz(0) {};
  KOKKOS_INLINE_FUNCTION Vec4_T(T _e, T _px, T _py, T _pz)
  {
    e = _e;
    px = _px;
    py = _py;
    pz = _pz;
  };
  // NOTE: The following signed_abs member function is not physical, but can be
  // used to print the invariant mass of four vectors. Negative values then
  // mean that the invariant mass squared is negative (which is usually due to
  // numerics).
  KOKKOS_FUNCTION
  T signed_abs() const
  {
    const T _abs2{abs2()};
    return (_abs2 < 0.0 ? -1.0 : 1.0) * Kokkos::sqrt(Kokkos::abs(_abs2));
  };
  KOKKOS_INLINE_FUNCTION
  T abs() const { return Kokkos::sqrt(abs2()); };
  KOKKOS_INLINE_FUNCTION
  T abs2() const { return (*this) * (*this); };
  KOKKOS_INLINE_FUNCTION
  T p_plus() const { return e + pz; }
  KOKKOS_INLINE_FUNCTION
  T p_minus() const { return e - pz; }
  KOKKOS_INLINE_FUNCTION
  T p_perp() const { return Kokkos::sqrt(p_perp2()); };
  KOKKOS_INLINE_FUNCTION
  T p_perp2() const
  {
    return px * px + py * py;
  }
  KOKKOS_INLINE_FUNCTION
  T m_perp() const { return Kokkos::sqrt(m_perp2()); };
  KOKKOS_INLINE_FUNCTION
  T m_perp2() const { return p_plus() * p_minus(); }
  KOKKOS_INLINE_FUNCTION
  T y() const { return 0.5 * Kokkos::log(p_plus() / p_minus()); }
  KOKKOS_INLINE_FUNCTION
  T cosPhi() const
  {
    if (p_perp() == 0)
      return 0.;
    return Kokkos::max(Kokkos::min(px / p_perp(), 1.0), -1.0);
  }
  KOKKOS_INLINE_FUNCTION
  T phi() const
  {
    if (py > 0.)
      return Kokkos::acos(cosPhi());
    else
      return -Kokkos::acos(cosPhi());
  }
  KOKKOS_INLINE_FUNCTION
  T vec3_abs() const { return Kokkos::sqrt(vec3_abs2()); }
  KOKKOS_INLINE_FUNCTION
  T vec3_abs2() const
  {
    return px * px + py * py + pz * pz;
  }

  KOKKOS_INLINE_FUNCTION
  Vec4_T operator-() const
  {
    return {-e, -px, -py, -pz};
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T &operator+=(const Vec4_T &rhs)
  {
    e += rhs.e;
    px += rhs.px;
    py += rhs.py;
    pz += rhs.pz;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T &operator-=(const Vec4_T &rhs)
  {
    e -= rhs.e;
    px -= rhs.px;
    py -= rhs.py;
    pz -= rhs.pz;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T &operator*=(T rhs)
  {
    e *= rhs;
    px *= rhs;
    py *= rhs;
    pz *= rhs;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T &operator+=(T rhs)
  {
    e += rhs;
    px += rhs;
    py += rhs;
    pz += rhs;
    return *this;
  }
  // scalar ops both orders
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator*(Vec4_T lhs, T rhs)
  {
    return lhs *= rhs;
  }
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator*(T lhs, Vec4_T rhs)
  {
    return rhs *= lhs;
  }
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator+(Vec4_T lhs, T rhs)
  {
    return lhs += rhs;
  }
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator+(T lhs, Vec4_T rhs)
  {
    return rhs += lhs;
  }

  // binary vector ops
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator+(Vec4_T lhs, const Vec4_T &rhs)
  {
    return lhs += rhs;
  }
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator-(Vec4_T lhs, const Vec4_T &rhs)
  {
    return lhs -= rhs;
  }
  KOKKOS_INLINE_FUNCTION
  friend T operator*(const Vec4_T &lhs, const Vec4_T &rhs)
  {
    return lhs.e * rhs.e - lhs.px * rhs.px - lhs.py * rhs.py -
           lhs.pz * rhs.pz;
  };

  friend std::ostream &operator<<(std::ostream &o, const Vec4_T &v)
  {
    return o << "(" << v.e << ", " << v.px << ", " << v.py << ", " << v.pz
             << ")";
  }
};

// Specialization
using Vec4 = Vec4_T<double>;
using Vec4SIMD = Vec4_T<Kokkos::Experimental::simd<double>>;

#endif

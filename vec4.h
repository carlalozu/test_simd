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

template <typename T> struct Vec4_T : std::array<T, 4> {

  KOKKOS_INLINE_FUNCTION Vec4_T() : std::array<T, 4> {} {};
  KOKKOS_INLINE_FUNCTION Vec4_T(T e, T px, T py, T pz)
  {
    (*this)[0] = e;
    (*this)[1] = px;
    (*this)[2] = py;
    (*this)[3] = pz;
  };
  // NOTE: The following signed_abs member function is not physical, but can be
  // used to print the invariant mass of four vectors. Negative values then
  // mean that the invariant mass squared is negative (which is usually due to
  // numerics).
  KOKKOS_FUNCTION
  T signed_abs() const
  {
    const T _abs2 {abs2()};
    return (_abs2 < 0.0 ? -1.0 : 1.0) * Kokkos::sqrt(Kokkos::abs(_abs2));
  };
  KOKKOS_INLINE_FUNCTION
  T abs() const { return Kokkos::sqrt(abs2()); };
  KOKKOS_INLINE_FUNCTION
  T abs2() const { return (*this) * (*this); };
  KOKKOS_INLINE_FUNCTION
  T p_plus() const { return (*this)[0] + (*this)[3]; }
  KOKKOS_INLINE_FUNCTION
  T p_minus() const { return (*this)[0] - (*this)[3]; }
  KOKKOS_INLINE_FUNCTION
  T p_perp() const { return Kokkos::sqrt(p_perp2()); };
  KOKKOS_INLINE_FUNCTION
  T p_perp2() const
  {
    return (*this)[1] * (*this)[1] + (*this)[2] * (*this)[2];
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
    return Kokkos::max(Kokkos::min((*this)[1] / p_perp(), 1.0), -1.0);
  }
  KOKKOS_INLINE_FUNCTION
  T phi() const
  {
    if ((*this)[2] > 0.)
      return Kokkos::acos(cosPhi());
    else
      return -Kokkos::acos(cosPhi());
  }
  KOKKOS_INLINE_FUNCTION
  T vec3_abs() const { return Kokkos::sqrt(vec3_abs2()); }
  KOKKOS_INLINE_FUNCTION
  T vec3_abs2() const
  {
    return (*this)[1] * (*this)[1] + (*this)[2] * (*this)[2] +
           (*this)[3] * (*this)[3];
  }

  KOKKOS_INLINE_FUNCTION
  Vec4_T operator-() const
  {
    return {-(*this)[0], -(*this)[1], -(*this)[2], -(*this)[3]};
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T& operator+=(const Vec4_T& rhs)
  {
    (*this)[0] += rhs[0];
    (*this)[1] += rhs[1];
    (*this)[2] += rhs[2];
    (*this)[3] += rhs[3];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T& operator-=(const Vec4_T& rhs)
  {
    (*this)[0] -= rhs[0];
    (*this)[1] -= rhs[1];
    (*this)[2] -= rhs[2];
    (*this)[3] -= rhs[3];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T& operator*=(T rhs)
  {
    (*this)[0] *= rhs;
    (*this)[1] *= rhs;
    (*this)[2] *= rhs;
    (*this)[3] *= rhs;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  Vec4_T& operator+=(T rhs)
  {
    (*this)[0] += rhs;
    (*this)[1] += rhs;
    (*this)[2] += rhs;
    (*this)[3] += rhs;
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
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator+(Vec4_T lhs, const Vec4_T& rhs)
  {
    return lhs += rhs;
  }
  KOKKOS_INLINE_FUNCTION friend Vec4_T operator-(Vec4_T lhs, const Vec4_T& rhs)
  {
    return lhs -= rhs;
  }
  KOKKOS_INLINE_FUNCTION
  friend T operator*(const Vec4_T& lhs, const Vec4_T& rhs)
  {
    return lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] -
           lhs[3] * rhs[3];
  };

  friend std::ostream& operator<<(std::ostream& o, const Vec4_T& v)
  {
    return o << "(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3]
             << ")";
  }
};

// Specialization
using Vec4 = Vec4_T<double>;
using Vec4SIMD = Vec4_T<Kokkos::Experimental::simd<double>>;

#endif

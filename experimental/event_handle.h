
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <cmath>
#include "cvec4.h"
#include "vec4.h"

using C = Complex::complex<double>;
using tag_type = Kokkos::Experimental::vector_aligned_tag;

/**
 * Event handle dummy class
 *
 * This class replicates the event handle in the original code
 * but using the Vec4 structures that do not use std::array
 * as base class.
 */
struct Evt
{

    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    int _n;

    // Allocate views for the arrays
    Kokkos::View<double *, MemorySpace> _c0r;
    Kokkos::View<double *, MemorySpace> _c1r;
    Kokkos::View<double *, MemorySpace> _c2r;
    Kokkos::View<double *, MemorySpace> _c3r;
    Kokkos::View<double *, MemorySpace> _c0i;
    Kokkos::View<double *, MemorySpace> _c1i;
    Kokkos::View<double *, MemorySpace> _c2i;
    Kokkos::View<double *, MemorySpace> _c3i;
    Kokkos::View<double *, MemorySpace> e;
    Kokkos::View<double *, MemorySpace> px;
    Kokkos::View<double *, MemorySpace> py;
    Kokkos::View<double *, MemorySpace> pz;

    // Constructor
    Evt(int n)
        : _n(n), _c0r(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c0r"), n), _c1r(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c1r"), n), _c2r(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c2r"), n), _c3r(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c3r"), n), _c0i(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c0i"), n), _c1i(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c1i"), n), _c2i(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c2i"), n), _c3i(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c3i"), n), e(Kokkos::view_alloc(Kokkos::WithoutInitializing, "e"), n), px(Kokkos::view_alloc(Kokkos::WithoutInitializing, "px"), n), py(Kokkos::view_alloc(Kokkos::WithoutInitializing, "py"), n), pz(Kokkos::view_alloc(Kokkos::WithoutInitializing, "pz"), n)
    {
    }

    KOKKOS_INLINE_FUNCTION C internal_c0(const int &c)
    {
        return C{_c0r(c), _c0i(c)};
    }
    KOKKOS_INLINE_FUNCTION void internal_c0(const int &c, const C &val)
    {
        _c0r(c) = val.real();
        _c0i(c) = val.imag();
    }

    KOKKOS_INLINE_FUNCTION C internal_c1(const int &c)
    {
        return C{_c1r(c), _c1i(c)};
    }
    KOKKOS_INLINE_FUNCTION void internal_c1(const int &c, const C &val)
    {
        _c1r(c) = val.real();
        _c1i(c) = val.imag();
    }

    KOKKOS_INLINE_FUNCTION C internal_c2(const int &c)
    {
        return C{_c2r(c), _c2i(c)};
    }
    KOKKOS_INLINE_FUNCTION void internal_c2(const int &c, const C &val)
    {
        _c2r(c) = val.real();
        _c2i(c) = val.imag();
    }

    KOKKOS_INLINE_FUNCTION C internal_c3(const int &c)
    {
        return C{_c3r(c), _c3i(c)};
    }
    KOKKOS_INLINE_FUNCTION void internal_c3(const int &c, const C &val)
    {
        _c3r(c) = val.real();
        _c3i(c) = val.imag();
    }

    KOKKOS_INLINE_FUNCTION CVec4 internal_c(const int &i)
    {
        CVec4 ret;
        ret.c0 = internal_c0(i);
        ret.c1 = internal_c1(i);
        ret.c2 = internal_c2(i);
        ret.c3 = internal_c3(i);
        return ret;
    }

    template <typename tag_type>
    KOKKOS_INLINE_FUNCTION CVec4SIMD internal_c(const int &i, tag_type tag)
    {
        simd_double c0r(&_c0r(i), tag);
        simd_double c1r(&_c1r(i), tag);
        simd_double c2r(&_c2r(i), tag);
        simd_double c3r(&_c3r(i), tag);

        simd_double c0i(&_c0i(i), tag);
        simd_double c1i(&_c1i(i), tag);
        simd_double c2i(&_c2i(i), tag);
        simd_double c3i(&_c3i(i), tag);

        Vec4SIMD cr = {c0r, c1r, c2r, c3r};
        Vec4SIMD ci = {c0i, c1i, c2i, c3i};

        return {cr, ci};
    }

    KOKKOS_INLINE_FUNCTION Vec4 internal_p(const int &i)
    {
        return {e(i), px(i), py(i), pz(i)};
    }

    template <typename tag_type>
    KOKKOS_INLINE_FUNCTION Vec4SIMD internal_p(const int &i, tag_type tag)
    {
        simd_double s_e(&e(i), tag);
        simd_double s_px(&px(i), tag);
        simd_double s_py(&py(i), tag);
        simd_double s_pz(&pz(i), tag);

        return {s_e, s_px, s_py, s_pz};
    }

    template <typename tag_type>
    KOKKOS_INLINE_FUNCTION void set_internal_c(const int &i, const CVec4SIMD &c, tag_type tag)
    {
        c.c0.real().copy_to(&_c0r(i), tag);
        c.c0.imag().copy_to(&_c0i(i), tag);

        c.c1.real().copy_to(&_c1r(i), tag);
        c.c1.imag().copy_to(&_c1i(i), tag);

        c.c2.real().copy_to(&_c2r(i), tag);
        c.c2.imag().copy_to(&_c2i(i), tag);

        c.c3.real().copy_to(&_c3r(i), tag);
        c.c3.imag().copy_to(&_c3i(i), tag);
    }

    KOKKOS_INLINE_FUNCTION void set_internal_c(const int &i, const CVec4 &c)
    {
        internal_c0(i, c.c0);
        internal_c1(i, c.c1);
        internal_c2(i, c.c2);
        internal_c3(i, c.c3);
    }

    KOKKOS_INLINE_FUNCTION void reset_arrays()
    {
        Kokkos::parallel_for("init_arrays", _n, KOKKOS_LAMBDA(int i) {
            _c0r(i) = static_cast<double>(i + 1);
            _c1r(i) = static_cast<double>(i + 2);
            _c2r(i) = static_cast<double>(i + 3);
            _c3r(i) = static_cast<double>(i + 4);
            _c0i(i) = static_cast<double>(i + 5);
            _c1i(i) = static_cast<double>(i + 6);
            _c2i(i) = static_cast<double>(i + 7);
            _c3i(i) = static_cast<double>(i + 8);
            e(i)    = static_cast<double>(i + 9);
            px(i)   = static_cast<double>(i + 10);
            py(i)   = static_cast<double>(i + 11);
            pz(i)   = static_cast<double>(i + 12); });
    }
};

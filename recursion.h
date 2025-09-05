
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <cmath>
// #include "cvec4_copy.h"
// #include "vec4_copy.h"
// #include "event_handle_copy.h"
#include "cvec4.h"
#include "vec4.h"
#include "event_handle.h"
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;
using C = Complex::complex<double>;
using tag_type = Kokkos::Experimental::vector_aligned_tag;


KOKKOS_INLINE_FUNCTION void evaluate_ggg_vertex_kernel_v(Evt evt, int i)
{
  tag_type tag {};

  CVec4SIMD ja {evt.internal_c(i, tag)};
  CVec4SIMD jb {evt.internal_c(i+8, tag)};
  Vec4SIMD pa {evt.internal_p(i, tag)};
  Vec4SIMD pb {evt.internal_p(i+8, tag)};

  Vec4SIMD pab {pa + pb};
  CVec4SIMD jc =
      (ja * (pb + pab)) * jb + (ja * jb) * (pa - pb) - (jb * (pa + pab)) * ja;
  jc *= -1.0;
  
  CVec4SIMD jd {evt.internal_c(i+16, tag)};
  evt.set_internal_c(i, (jd + jc), tag);
};

KOKKOS_INLINE_FUNCTION void evaluate_ggg_vertex_kernel(Evt evt, int i)
{
  CVec4 ja {evt.internal_c(i)};
  CVec4 jb {evt.internal_c(i+8)};
  Vec4 pa {evt.internal_p(i)};
  Vec4 pb {evt.internal_p(i+8)};

  Vec4 pab {pa + pb};
  CVec4 jc =
  (ja * (pb + pab)) * jb + (ja * jb) * (pa - pb) - (jb * (pa + pab)) * ja;
  jc *= -1.0;

  CVec4 jd {evt.internal_c(i+16)};
  evt.set_internal_c(i, (jd + jc));
};
KOKKOS_INLINE_FUNCTION void evaluate_ggg_vertex_kernel_unrolled_single_value(const Evt& evt, int i)
{
    // CVec4 ja {evt.internal_c(i)};
    double ja0_real = evt._c0r(i);
    double ja1_real = evt._c1r(i);
    double ja2_real = evt._c2r(i);
    double ja3_real = evt._c3r(i);
    
    double ja0_imag = evt._c0i(i);
    double ja1_imag = evt._c1i(i);
    double ja2_imag = evt._c2i(i);
    double ja3_imag = evt._c3i(i);
    
    // CVec4 jb {evt.internal_c(i+8)};
    double jb0_real = evt._c0r(i+8);
    double jb1_real = evt._c1r(i+8);
    double jb2_real = evt._c2r(i+8);
    double jb3_real = evt._c3r(i+8);
    
    double jb0_imag = evt._c0i(i+8);
    double jb1_imag = evt._c1i(i+8);
    double jb2_imag = evt._c2i(i+8);
    double jb3_imag = evt._c3i(i+8);

    // CVec4SIMD jd {evt.internal_c(i+16, tag)};
    double jd0_real = evt._c0r(i+16);
    double jd1_real = evt._c1r(i+16);
    double jd2_real = evt._c2r(i+16);
    double jd3_real = evt._c3r(i+16);
    
    double jd0_imag = evt._c0i(i+16);
    double jd1_imag = evt._c1i(i+16);
    double jd2_imag = evt._c2i(i+16);
    double jd3_imag = evt._c3i(i+16);

    // Vec4 pa {evt.internal_p(i)};
    double pa0 = evt.e(i);
    double pa1 = evt.px(i);
    double pa2 = evt.py(i);
    double pa3 = evt.pz(i);
    
    // Vec4 pb {evt.internal_p(i+8)};
    double pb0 = evt.e(i+8);
    double pb1 = evt.px(i+8);
    double pb2 = evt.py(i+8);
    double pb3 = evt.pz(i+8);

    // add (ja * (pb + pab)) * jb
    double pr1 = ja0_real * (pa0 + pb0 + pb0) - ja1_real * (pa1 + pb1 + pb1) -
            ja2_real * (pa2 + pb2 + pb2) - ja3_real * (pa3 + pb3 + pb3);
    double pi1 = ja0_imag * (pa0 + pb0 + pb0) - ja1_imag * (pa1 + pb1 + pb1) -
            ja2_imag * (pa2 + pb2 + pb2) - ja3_imag * (pa3 + pb3 + pb3);

    // subtract (jb * (pa + pab)) * ja
    double pr2 = jb0_real * (pb0 + pa0 + pa0) - jb1_real * (pb1 + pa1 + pa1) -
            jb2_real * (pb2 + pa2 + pa2) - jb3_real * (pb3 + pa3 + pa3);
    double pi2 = jb0_imag * (pb0 + pa0 + pa0) - jb1_imag * (pb1 + pa1 + pa1) -
            jb2_imag * (pb2 + pa2 + pa2) - jb3_imag * (pb3 + pa3 + pa3);

    // add (ja * jb) * (pa - pb)
    double pr3 = ja0_real * jb0_real - ja0_imag * jb0_imag -
            (ja1_real * jb1_real - ja1_imag * jb1_imag) -
            (ja2_real * jb2_real - ja2_imag * jb2_imag) -
            (ja3_real * jb3_real - ja3_imag * jb3_imag);
    double pi3 = ja0_real * jb0_imag + ja0_imag * jb0_real -
            (ja1_real * jb1_imag + ja1_imag * jb1_real) -
            (ja2_real * jb2_imag + ja2_imag * jb2_real) -
            (ja3_real * jb3_imag + ja3_imag * jb3_real);


    jd0_real -= pr1 * jb0_real - pi1 * jb0_imag;
    jd0_real += pr2 * ja0_real - pi2 * ja0_imag;
    jd0_real -= pr3 * (pa0 - pb0);

    jd0_imag -= pi1 * jb0_real + pr1 * jb0_imag;
    jd0_imag += pi2 * ja0_real + pr2 * ja0_imag;
    jd0_imag -= pi3 * (pa0 - pb0);

    jd1_real -= pr1 * jb1_real - pi1 * jb1_imag;
    jd1_real += pr2 * ja1_real - pi2 * ja1_imag;
    jd1_real -= pr3 * (pa1 - pb1);

    jd1_imag -= pi1 * jb1_real + pr1 * jb1_imag;
    jd1_imag += pi2 * ja1_real + pr2 * ja1_imag;
    jd1_imag -= pi3 * (pa1 - pb1);

    jd2_real -= pr1 * jb2_real - pi1 * jb2_imag;
    jd2_real += pr2 * ja2_real - pi2 * ja2_imag;
    jd2_real -= pr3 * (pa2 - pb2);

    jd2_imag -= pi1 * jb2_real + pr1 * jb2_imag;
    jd2_imag += pi2 * ja2_real + pr2 * ja2_imag;
    jd2_imag -= pi3 * (pa2 - pb2);

    jd3_real -= pr1 * jb3_real - pi1 * jb3_imag;
    jd3_real += pr2 * ja3_real - pi2 * ja3_imag;
    jd3_real -= pr3 * (pa3 - pb3);

    jd3_imag -= pi1 * jb3_real + pr1 * jb3_imag;
    jd3_imag += pi2 * ja3_real + pr2 * ja3_imag;
    jd3_imag -= pi3 * (pa3 - pb3);

    // evt.set_internal_c(i, jd);
    evt._c0r(i) = jd0_real;
    evt._c1r(i) = jd1_real;
    evt._c2r(i) = jd2_real;
    evt._c3r(i) = jd3_real;
    
    evt._c0i(i) = jd0_imag;
    evt._c1i(i) = jd1_imag;
    evt._c2i(i) = jd2_imag;
    evt._c3i(i) = jd3_imag;
};


KOKKOS_INLINE_FUNCTION void evaluate_ggg_vertex_kernel_unrolled(const Evt& evt, int i)
{
  tag_type tag {};
    // CVec4 ja {evt.internal_c(i)};
    simd_double ja0_real(&evt._c0r(i), tag);
    simd_double ja1_real(&evt._c1r(i), tag);
    simd_double ja2_real(&evt._c2r(i), tag);
    simd_double ja3_real(&evt._c3r(i), tag);
    
    simd_double ja0_imag(&evt._c0i(i), tag);
    simd_double ja1_imag(&evt._c1i(i), tag);
    simd_double ja2_imag(&evt._c2i(i), tag);
    simd_double ja3_imag(&evt._c3i(i), tag);
    
    // CVec4 jb {evt.internal_c(i+8)};
    simd_double jb0_real(&evt._c0r(i+8), tag);
    simd_double jb1_real(&evt._c1r(i+8), tag);
    simd_double jb2_real(&evt._c2r(i+8), tag);
    simd_double jb3_real(&evt._c3r(i+8), tag);
    
    simd_double jb0_imag(&evt._c0i(i+8), tag);
    simd_double jb1_imag(&evt._c1i(i+8), tag);
    simd_double jb2_imag(&evt._c2i(i+8), tag);
    simd_double jb3_imag(&evt._c3i(i+8), tag);

    // CVec4SIMD jd {evt.internal_c(i+16, tag)};
    simd_double jd0_real(&evt._c0r(i+16), tag);
    simd_double jd1_real(&evt._c1r(i+16), tag);
    simd_double jd2_real(&evt._c2r(i+16), tag);
    simd_double jd3_real(&evt._c3r(i+16), tag);
    
    simd_double jd0_imag(&evt._c0i(i+16), tag);
    simd_double jd1_imag(&evt._c1i(i+16), tag);
    simd_double jd2_imag(&evt._c2i(i+16), tag);
    simd_double jd3_imag(&evt._c3i(i+16), tag);

    // Vec4 pa {evt.internal_p(i)};
    simd_double pa0(&evt.e(i), tag);
    simd_double pa1(&evt.px(i), tag);
    simd_double pa2(&evt.py(i), tag);
    simd_double pa3(&evt.pz(i), tag);
    
    // Vec4 pb {evt.internal_p(i+8)};
    simd_double pb0(&evt.e(i+8), tag);
    simd_double pb1(&evt.px(i+8), tag);
    simd_double pb2(&evt.py(i+8), tag);
    simd_double pb3(&evt.pz(i+8), tag);

    // add (ja * (pb + pab)) * jb
    simd_double pr1 = ja0_real * (pa0 + pb0 + pb0) - ja1_real * (pa1 + pb1 + pb1) -
            ja2_real * (pa2 + pb2 + pb2) - ja3_real * (pa3 + pb3 + pb3);
    simd_double pi1 = ja0_imag * (pa0 + pb0 + pb0) - ja1_imag * (pa1 + pb1 + pb1) -
            ja2_imag * (pa2 + pb2 + pb2) - ja3_imag * (pa3 + pb3 + pb3);

    // subtract (jb * (pa + pab)) * ja
    simd_double pr2 = jb0_real * (pb0 + pa0 + pa0) - jb1_real * (pb1 + pa1 + pa1) -
            jb2_real * (pb2 + pa2 + pa2) - jb3_real * (pb3 + pa3 + pa3);
    simd_double pi2 = jb0_imag * (pb0 + pa0 + pa0) - jb1_imag * (pb1 + pa1 + pa1) -
            jb2_imag * (pb2 + pa2 + pa2) - jb3_imag * (pb3 + pa3 + pa3);

    // add (ja * jb) * (pa - pb)
    simd_double pr3 = ja0_real * jb0_real - ja0_imag * jb0_imag -
            (ja1_real * jb1_real - ja1_imag * jb1_imag) -
            (ja2_real * jb2_real - ja2_imag * jb2_imag) -
            (ja3_real * jb3_real - ja3_imag * jb3_imag);
    simd_double pi3 = ja0_real * jb0_imag + ja0_imag * jb0_real -
            (ja1_real * jb1_imag + ja1_imag * jb1_real) -
            (ja2_real * jb2_imag + ja2_imag * jb2_real) -
            (ja3_real * jb3_imag + ja3_imag * jb3_real);


    jd0_real -= pr1 * jb0_real - pi1 * jb0_imag;
    jd0_real += pr2 * ja0_real - pi2 * ja0_imag;
    jd0_real -= pr3 * (pa0 - pb0);

    jd0_imag -= pi1 * jb0_real + pr1 * jb0_imag;
    jd0_imag += pi2 * ja0_real + pr2 * ja0_imag;
    jd0_imag -= pi3 * (pa0 - pb0);

    jd1_real -= pr1 * jb1_real - pi1 * jb1_imag;
    jd1_real += pr2 * ja1_real - pi2 * ja1_imag;
    jd1_real -= pr3 * (pa1 - pb1);

    jd1_imag -= pi1 * jb1_real + -pr1 * jb1_imag;
    jd1_imag += pi2 * ja1_real + -pr2 * ja1_imag;
    jd1_imag -= pi3 * (pa1 - pb1);

    jd2_real -= pr1 * jb2_real - pi1 * jb2_imag;
    jd2_real += pr2 * ja2_real - pi2 * ja2_imag;
    jd2_real -= pr3 * (pa2 - pb2);

    jd2_imag -= pi1 * jb2_real + pr1 * jb2_imag;
    jd2_imag += pi2 * ja2_real + pr2 * ja2_imag;
    jd2_imag -= pi3 * (pa2 - pb2);

    jd3_real -= pr1 * jb3_real - pi1 * jb3_imag;
    jd3_real += pr2 * ja3_real - pi2 * ja3_imag;
    jd3_real -= pr3 * (pa3 - pb3);

    jd3_imag -= pi1 * jb3_real + pr1 * jb3_imag;
    jd3_imag += pi2 * ja3_real + pr2 * ja3_imag;
    jd3_imag -= pi3 * (pa3 - pb3);

    // evt.set_internal_c(i, jd, tag);
    jd0_real.copy_to(&evt._c0r(i), tag);
    jd1_real.copy_to(&evt._c1r(i), tag);
    jd2_real.copy_to(&evt._c2r(i), tag);
    jd3_real.copy_to(&evt._c3r(i), tag);
    
    jd0_imag.copy_to(&evt._c0i(i), tag);
    jd1_imag.copy_to(&evt._c1i(i), tag);
    jd2_imag.copy_to(&evt._c2i(i), tag);
    jd3_imag.copy_to(&evt._c3i(i), tag);
};


KOKKOS_INLINE_FUNCTION void evaluate_ggg_vertex_kernel_highway(const Evt& evt, int i)
{
    const hn::ScalableTag<double> d;
    using V = hn::Vec<decltype(d)>;
    // CVec4 ja {evt.internal_c(i)};
    const V ja0_real = hn::Load(d, &evt._c0r(i));
    const V ja1_real = hn::Load(d, &evt._c1r(i));
    const V ja2_real = hn::Load(d, &evt._c2r(i));
    const V ja3_real = hn::Load(d, &evt._c3r(i));
    
    const V ja0_imag = hn::Load(d, &evt._c0i(i));
    const V ja1_imag = hn::Load(d, &evt._c1i(i));
    const V ja2_imag = hn::Load(d, &evt._c2i(i));
    const V ja3_imag = hn::Load(d, &evt._c3i(i));
    
    // CVec4 jb {evt.internal_c(i+8)};
    const V jb0_real = hn::Load(d, &evt._c0r(i+8));
    const V jb1_real = hn::Load(d, &evt._c1r(i+8));
    const V jb2_real = hn::Load(d, &evt._c2r(i+8));
    const V jb3_real = hn::Load(d, &evt._c3r(i+8));
    
    const V jb0_imag = hn::Load(d, &evt._c0i(i+8));
    const V jb1_imag = hn::Load(d, &evt._c1i(i+8));
    const V jb2_imag = hn::Load(d, &evt._c2i(i+8));
    const V jb3_imag = hn::Load(d, &evt._c3i(i+8));

    // CVec4SIMD jd {evt.internal_c(i+16, tag)};
    V jd0_real = hn::Load(d, &evt._c0r(i+16));
    V jd1_real = hn::Load(d, &evt._c1r(i+16));
    V jd2_real = hn::Load(d, &evt._c2r(i+16));
    V jd3_real = hn::Load(d, &evt._c3r(i+16));
    
    V jd0_imag = hn::Load(d, &evt._c0i(i+16));
    V jd1_imag = hn::Load(d, &evt._c1i(i+16));
    V jd2_imag = hn::Load(d, &evt._c2i(i+16));
    V jd3_imag = hn::Load(d, &evt._c3i(i+16));

    // Vec4 pa {evt.internal_p(i)};
    const V pa0 = hn::Load(d, &evt.e(i));
    const V pa1 = hn::Load(d, &evt.px(i));
    const V pa2 = hn::Load(d, &evt.py(i));
    const V pa3 = hn::Load(d, &evt.pz(i));
    
    // Vec4 pb {evt.internal_p(i+8)};
    const V pb0 = hn::Load(d, &evt.e(i+8));
    const V pb1 = hn::Load(d, &evt.px(i+8));
    const V pb2 = hn::Load(d, &evt.py(i+8));
    const V pb3 = hn::Load(d, &evt.pz(i+8));

    // add (ja * (pb + pab)) * jb
    const V pr1 = ja0_real * (pa0 + pb0 + pb0) - ja1_real * (pa1 + pb1 + pb1) -
            ja2_real * (pa2 + pb2 + pb2) - ja3_real * (pa3 + pb3 + pb3);
    const V pi1 = ja0_imag * (pa0 + pb0 + pb0) - ja1_imag * (pa1 + pb1 + pb1) -
            ja2_imag * (pa2 + pb2 + pb2) - ja3_imag * (pa3 + pb3 + pb3);

    // subtract (jb * (pa + pab)) * ja
    const V pr2 = jb0_real * (pb0 + pa0 + pa0) - jb1_real * (pb1 + pa1 + pa1) -
            jb2_real * (pb2 + pa2 + pa2) - jb3_real * (pb3 + pa3 + pa3);
    const V pi2 = jb0_imag * (pb0 + pa0 + pa0) - jb1_imag * (pb1 + pa1 + pa1) -
            jb2_imag * (pb2 + pa2 + pa2) - jb3_imag * (pb3 + pa3 + pa3);

    // add (ja * jb) * (pa - pb)
    const V pr3 = ja0_real * jb0_real - ja0_imag * jb0_imag -
            (ja1_real * jb1_real - ja1_imag * jb1_imag) -
            (ja2_real * jb2_real - ja2_imag * jb2_imag) -
            (ja3_real * jb3_real - ja3_imag * jb3_imag);
    const V pi3 = ja0_real * jb0_imag + ja0_imag * jb0_real -
            (ja1_real * jb1_imag + ja1_imag * jb1_real) -
            (ja2_real * jb2_imag + ja2_imag * jb2_real) -
            (ja3_real * jb3_imag + ja3_imag * jb3_real);


    jd0_real -= pr1 * jb0_real - pi1 * jb0_imag;
    jd0_real += pr2 * ja0_real - pi2 * ja0_imag;
    jd0_real -= pr3 * (pa0 - pb0);

    jd0_imag -= pi1 * jb0_real + pr1 * jb0_imag;
    jd0_imag += pi2 * ja0_real + pr2 * ja0_imag;
    jd0_imag -= pi3 * (pa0 - pb0);

    jd1_real -= pr1 * jb1_real - pi1 * jb1_imag;
    jd1_real += pr2 * ja1_real - pi2 * ja1_imag;
    jd1_real -= pr3 * (pa1 - pb1);

    jd1_imag -= pi1 * jb1_real + pr1 * jb1_imag;
    jd1_imag += pi2 * ja1_real + pr2 * ja1_imag;
    jd1_imag -= pi3 * (pa1 - pb1);

    jd2_real -= pr1 * jb2_real - pi1 * jb2_imag;
    jd2_real += pr2 * ja2_real - pi2 * ja2_imag;
    jd2_real -= pr3 * (pa2 - pb2);

    jd2_imag -= pi1 * jb2_real + pr1 * jb2_imag;
    jd2_imag += pi2 * ja2_real + pr2 * ja2_imag;
    jd2_imag -= pi3 * (pa2 - pb2);

    jd3_real -= pr1 * jb3_real - pi1 * jb3_imag;
    jd3_real += pr2 * ja3_real - pi2 * ja3_imag;
    jd3_real -= pr3 * (pa3 - pb3);

    jd3_imag -= pi1 * jb3_real + pr1 * jb3_imag;
    jd3_imag += pi2 * ja3_real + pr2 * ja3_imag;
    jd3_imag -= pi3 * (pa3 - pb3);

    // evt.set_internal_c(i, jd, tag);
    hn::Store(jd0_real, d, &evt._c0r(i));
    hn::Store(jd1_real, d, &evt._c1r(i));
    hn::Store(jd2_real, d, &evt._c2r(i));
    hn::Store(jd3_real, d, &evt._c3r(i));
    
    hn::Store(jd0_imag, d, &evt._c0i(i));
    hn::Store(jd1_imag, d, &evt._c1i(i));
    hn::Store(jd2_imag, d, &evt._c2i(i));
    hn::Store(jd3_imag, d, &evt._c3i(i));
};
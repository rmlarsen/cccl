//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_TRIG_H
#define _CUDA___FP_FPMP_MATH_IMPL_TRIG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_trig.h - fpmp2 trigonometric functions (sin, cos, tan, asin/acos/atan, atan2, sincos, *pi)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
#include <cuda/std/__bit/countl.h> // countl_zero for the Payne-Hanek normalization
#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/numbers>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * Internal kernels for sin, cos and sincos (fp32mp2)
 * ====================================================================
 * Algorithm:
 *   1. Argument reduction: x = n*(pi/2) + r, |r| <= pi/4
 *      - Tiny (|x| < pi/4): no reduction
 *      - Fast (|x| < 2^20): Cody-Waite with exact error tracking
 *        via two_mult_fma + two_sum (4-piece pi/2, within 8.5e-32)
 *      - Large (|x| >= 2^20): Payne-Hanek using integer 2/pi table, combining
 *        x_hi and x_lo fractions in 64-bit fixed-point before converting to
 *        fp32mp2 (pure fp32 arithmetic, no fp64). Delivers ~46 bits in the
 *        reduced argument r; final precision can be lower for tan near
 *        singularities, where small input quantization is amplified by tan'.
 *   2. Evaluate sin(r) and cos(r) via Taylor polynomials in fp32mp2
 *      sin: 8 terms (x through x^15), cos: 9 terms (1 through x^16)
 *   3. Map to correct quadrant using n mod 4
 *      sincos computes both kernels; sin/cos call sincos internally
 * ====================================================================
 */

/*
 * Payne-Hanek stage 1: compute |a| * (2/pi) via integer arithmetic,
 * extract 2-bit quadrant and 62-bit unsigned fraction in [0, 1).
 * Does NOT apply the >0.5 adjustment -- caller handles that.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_ph_frac(_FpType __a_hi, unsigned* __q_out, uint32_t* __frac_hi, uint32_t* __frac_lo) noexcept
{
  constexpr unsigned int __i2opi[] = {
    0x3c439041U,
    0xdb629599U,
    0xf534ddc0U,
    0xfc2757d1U,
    0x4e441529U,
    0xa2f9836eU,
  };

  uint32_t __ia = ::cuda::std::bit_cast<uint32_t>(__a_hi);
  uint32_t __result[7];
  uint32_t __hi;
  uint32_t __lo;
  int __iq;

  int __e = (int) ((__ia >> 23U) & 0xFFU) - 128;
  __ia    = (__ia << 8U) | 0x80000000U;
  __hi    = 0;

  for (__iq = 0; __iq < 6; __iq++)
  {
    uint64_t __p   = (uint64_t) __i2opi[__iq] * __ia + __hi;
    __result[__iq] = (uint32_t) __p;
    __hi           = (uint32_t) (__p >> 32);
  }
  __result[__iq] = __hi;

  /* Extract the window containing quadrant + fraction bits.
   * For e >= 0 (|a| >= 2): standard extraction with left shift.
   * For e < 0 (|a| < 2): extraction from idx=4 with right shift
   *   to handle small inputs without extending the table.
   */
  uint32_t __lo2;
  if (__e >= 0)
  {
    uint32_t __ue  = (uint32_t) __e;
    uint32_t __idx = 4U - (__ue >> 5U);
    __ue           = __ue & 31U;
    __hi           = __result[__idx + 2];
    __lo           = __result[__idx + 1];
    __lo2          = (__idx > 0) ? __result[__idx] : 0U;
    if (__ue != 0U)
    {
      uint32_t __q = 32U - __ue;
      __hi         = (__hi << __ue) + (__lo >> __q);
      __lo         = (__lo << __ue) + (__lo2 >> __q);
    }
  }
  else
  {
    int __r = -__e;
    __hi    = __result[6];
    __lo    = __result[5];
    __lo2   = __result[4];
    if (__r < 32)
    {
      uint32_t __q  = (uint32_t) (32 - __r);
      uint32_t __ur = (uint32_t) __r;
      __hi          = __hi >> __ur;
      __lo          = (__lo >> __ur) | (__result[6] << __q);
    }
    else
    {
      __hi = 0;
      __lo = __result[6];
    }
  }

  *__q_out   = __hi >> 30U;
  *__frac_hi = (__hi << 2U) + (__lo >> 30U);
  *__frac_lo = (__lo << 2U);
}

/*
 * Payne-Hanek stage 2: convert a 64-bit unsigned fraction in [0, 0.5)
 * (after the >0.5 adjustment) to an fp32mp2 angle by multiplying
 * by pi/2 using 64 bits of pi/4.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_frac_to_angle(uint32_t __hi, uint32_t __lo, uint32_t __s, _FpType* __r_hi, _FpType* __r_lo) noexcept
{
  /* Normalize: shift so MSB of hi is 1.
   * countl_zero is well defined for 0 (it yields 32), unlike __builtin_clz, and lowers
   * to the clz instruction on device.
   */
  uint32_t __lz = (uint32_t) ::cuda::std::countl_zero(__hi);

  if (__lz >= 32U)
  {
    __lz += (__lo == 0U) ? 0U : (uint32_t) ::cuda::std::countl_zero(__lo);
    __hi             = __lo;
    __lo             = 0U;
    uint32_t __shift = __lz - 32U;
    if (__shift != 0U)
    {
      __hi <<= __shift;
    }
  }
  else if (__lz != 0U)
  {
    __hi = (__hi << __lz) | (__lo >> (32U - __lz));
    __lo = __lo << __lz;
  }

  /* Multiply by pi/2 using 64 bits of pi/4.
   * pi/4 = 0x0.C90FDAA2_2168C234...
   * The *2 (pi/4 -> pi/2) is in biased_exp = 127 - lz.
   */
  constexpr uint32_t __pio4_hi32 = 0xC90FDAA2U;
  constexpr uint32_t __pio4_lo32 = 0x2168C234U;

  uint64_t __p_hh = (uint64_t) __hi * __pio4_hi32;
  uint64_t __p_hl = (uint64_t) __hi * __pio4_lo32;
  uint64_t __p_lh = (uint64_t) __lo * __pio4_hi32;

  uint64_t __combined = __p_hh + (__p_hl >> 32) + (__p_lh >> 32);
  uint32_t __rhi      = (uint32_t) (__combined >> 32);
  uint32_t __rlo      = (uint32_t) __combined;

  if ((int32_t) __rhi > 0)
  {
    __rhi = (__rhi << 1) | (__rlo >> 31);
    __rlo = __rlo << 1;
    __lz++;
  }

  /* Convert to fp32mp2 */
  uint32_t __biased_exp = 127U - __lz;
  uint32_t __f1_bits    = __s | (__biased_exp << 23) | ((__rhi >> 8) & 0x7FFFFFU);

  uint32_t __rem       = (__rhi << 24) | (__rlo >> 8);
  uint32_t __rem_extra = __rlo << 24;

  if (__rem == 0U)
  {
    *__r_hi = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo = _FpType(0);
    return;
  }

  uint32_t __rlz = (uint32_t) ::cuda::std::countl_zero(__rem);

  uint32_t __rem_norm = (__rlz > 0U) ? ((__rem << __rlz) | (__rem_extra >> (32U - __rlz))) : __rem;

  int __biased_exp2 = (int) __biased_exp - 24 - (int) __rlz;
  if (__biased_exp2 < 1)
  {
    *__r_hi = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo = _FpType(0);
  }
  else
  {
    uint32_t __f2_bits = __s | ((uint32_t) __biased_exp2 << 23) | ((__rem_norm >> 8) & 0x7FFFFFU);
    *__r_hi            = ::cuda::std::bit_cast<_FpType>(__f1_bits);
    *__r_lo            = ::cuda::std::bit_cast<_FpType>(__f2_bits);
  }
}

/*
 * Trigonometric argument reduction for fp32mp2.
 * Returns quadrant (mod 4) and reduced argument r  in  [-pi/4, pi/4].
 *
 * Three paths:
 *   Tiny:  |x| < pi/4 -> no reduction
 *   Fast:  |x| < 2^20 -> Cody-Waite with exact error tracking
 *   Large: |x| >= 2^20 -> Payne-Hanek (integer 2/pi table)
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_trig_reduction(
  _FpType __x_hi, _FpType __x_lo, int* __quadrant, _FpType* __r_hi, _FpType* __r_lo) noexcept
{
  using __afloat = fp32mp2_high;

  _FpType __abs_hi    = (__x_hi < _FpType(0)) ? -__x_hi : __x_hi;
  uint32_t __abs_bits = ::cuda::std::bit_cast<uint32_t>(__abs_hi);

  /* No reduction for |x| < pi/4 */
  if (__abs_bits < 0x3F490FDBU)
  {
    *__quadrant = 0;
    *__r_hi     = __x_hi;
    *__r_lo     = __x_lo;
    return;
  }

  /* Inf / NaN -> return NaN, quadrant 0 */
  if (__abs_bits >= 0x7F800000U)
  {
    *__quadrant = 0;
    *__r_hi     = __x_hi - __x_hi;
    *__r_lo     = _FpType(0);
    return;
  }

  if (__abs_bits < 0x49800000U)
  {
    /* -- Fast path: Cody-Waite for |x_hi| < 2^20 --
     *
     * pi/2 split into 4 float pieces, whose sum misses pi/2 by 8.5e-32.
     * C1 has 2 trailing zero mantissa bits, making n*C1 exact
     * for |n| < 2^12 via two_mult_fma, and accurate for larger n.
     *
     * Error tracking: two_mult_fma gives exact n*C1 = ph + pl,
     * two_sum gives exact x_hi - ph = s + e.
     * Remaining corrections accumulated in fp32mp2_high
     * to preserve ~46 bits when s is near zero (catastrophic
     * cancellation near multiples of pi).
     */
    constexpr _FpType __c1 = _FpType(1.5707962512969971e+000);
    constexpr _FpType __c2 = _FpType(7.5497894158615964e-008);
    constexpr _FpType __c3 = _FpType(5.3903029534742384e-015);
    constexpr _FpType __c4 = _FpType(-0x1.cceba4p-74);

    int __n       = ::cuda::__fpmp_fp2int_rn(__x_hi * _FpType(0x1.45f306p-1f));
    _FpType __n_f = ::cuda::__fpmp_int2fp_rn<_FpType>(__n);

    /* Exact product n*C1 = ph + pl */
    _FpType __pl;
    _FpType __ph = ::cuda::__fpmp_two_mult_fma(__n_f, __c1, &__pl);

    /* Exact subtraction x_hi - ph = s + e */
    _FpType __e;
    _FpType __s = ::cuda::__fpmp_two_sum(__x_hi, -__ph, &__e);

    /* Build result as fp32mp2_high from exact (s, e),
     * then accumulate corrections with full precision.
     */
    __afloat __result(__s, __e);
    /* Subtracting pl and adding x_lo as single limbs gives the same values as adding them as
     * pairs with a zero low word, in fewer operations, and makes sin, cos, sincos and tan 4-10%
     * faster on sm_100 and sm_120. On sm_90, nvcc schedules the single-limb form worse: sin,
     * cos and sincos ran 1-4% slower than with the pairs (tan 4% faster), so sm_90 keeps them.
     */
    NV_IF_ELSE_TARGET(NV_IS_EXACTLY_SM_90,
                      (__result = __result + __afloat(-__pl); __result = __result + __afloat(__x_lo);),
                      (__result = __result - __pl; __result = __result + __x_lo;))

    /* Exact product n*C2 = nC2_hi + nC2_lo via two_mult_fma */
    _FpType __n_c2_lo;
    _FpType __n_c2_hi = ::cuda::__fpmp_two_mult_fma(__n_f, __c2, &__n_c2_lo);
    __result          = __result - __afloat(__n_c2_hi, __n_c2_lo);

    /* nC3_hi + nC3_lo = n*(C3 + C4) to within 2^-47 * n*C3: n*C3 exactly, and n*C4 added
     * to its low word by one fma. Near a multiple of pi/2, r is far smaller than n*C3, so
     * neither rounding n*C3 (error up to 2^-24 * n*C3) nor cutting pi/2 at C3
     * (error n * 9.5e-23) keeps the error small relative to r.
     */
    _FpType __n_c3_lo;
    _FpType __n_c3_hi = ::cuda::__fpmp_two_mult_fma(__n_f, __c3, &__n_c3_lo);
    __n_c3_lo         = ::cuda::__fpmp_fma_rn(__n_f, __c4, __n_c3_lo);
    __result          = __result - __afloat(__n_c3_hi, __n_c3_lo);

    *__quadrant = __n;
    *__r_hi     = __result.hi();
    *__r_lo     = __result.lo();
  }
  else
  {
    /* -- Slow path: |x_hi| >= 2^20 -- */

    /* Payne-Hanek: combine x_hi and x_lo 2/pi fractions in
     * 64-bit fixed-point BEFORE the pi/2 multiply to avoid
     * precision loss from floating-point cancellation.
     */
    uint32_t __fhi;
    uint32_t __flo;
    unsigned __q_hi;
    ::cuda::__internal_fpmp2_ph_frac(__x_hi, &__q_hi, &__fhi, &__flo);

    uint32_t __x_hi_sign = ::cuda::std::bit_cast<uint32_t>(__x_hi) & 0x80000000U;
    int __q              = (int) __q_hi;

    /* Add x_lo contribution in fixed-point.
     * |x_lo| <= |x_hi|*2^-24 can still span many quadrants,
     * and even small |x_lo| can dominate the fraction when
     * the result angle is near zero.  Handle ALL non-zero x_lo.
     */
    if (__x_lo != _FpType(0))
    {
      _FpType __abs_lo       = (__x_lo < _FpType(0)) ? -__x_lo : __x_lo;
      uint32_t __abs_lo_bits = ::cuda::std::bit_cast<uint32_t>(__abs_lo);
      bool __same_sign       = (__x_lo > _FpType(0)) == (__x_hi > _FpType(0));

      uint32_t __fhi2 = 0;
      uint32_t __flo2 = 0;
      unsigned __q_lo = 0;

      if (__abs_lo_bits >= 0x00800000U)
      {
        ::cuda::__internal_fpmp2_ph_frac(__abs_lo, &__q_lo, &__fhi2, &__flo2);
      }

      uint64_t __f1 = ((uint64_t) __fhi << 32) | __flo;
      uint64_t __f2 = ((uint64_t) __fhi2 << 32) | __flo2;

      if (__same_sign)
      {
        __q += (int) __q_lo;
        uint64_t __sum = __f1 + __f2;
        if (__sum < __f1)
        {
          __q++;
        }
        __fhi = (uint32_t) (__sum >> 32);
        __flo = (uint32_t) __sum;
      }
      else
      {
        __q -= (int) __q_lo;
        if (__f1 >= __f2)
        {
          __f1 -= __f2;
        }
        else
        {
          __f1 = 0ULL - (__f2 - __f1);
          __q--;
        }
        __fhi = (uint32_t) (__f1 >> 32);
        __flo = (uint32_t) __f1;
      }
    }

    uint32_t __top_bit = __fhi >> 31U;
    __q += __top_bit;
    if (__x_hi_sign != 0U)
    {
      __q = 0U - (unsigned) __q;
    }

    if (__top_bit != 0U)
    {
      __fhi = ~__fhi;
      __flo = ~__flo;
      __x_hi_sign ^= 0x80000000U;
    }

    if (__fhi == 0U && __flo == 0U)
    {
      *__quadrant = (int) __q;
      *__r_hi     = _FpType(0);
      *__r_lo     = _FpType(0);
      return;
    }

    *__quadrant = (int) __q;
    ::cuda::__internal_fpmp2_frac_to_angle(__fhi, __flo, __x_hi_sign, __r_hi, __r_lo);
  }
}

/*
 * Sin kernel: evaluate sin(x) for |x| <= pi/4 using fp32mp2 Taylor series.
 * sin(x) = x + x^3*Q(x^2), Q(u) = Sum Taylor coefficients from -1/3! to -1/15!.
 * Upper terms (s7..s4) in single precision, lower terms (s3..s1) in fp32mp2.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin_kernel(_FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr __ffloat __s1(-1.6666666666666666e-01);
  constexpr __ffloat __s2(8.3333333333333333e-03);
  constexpr __ffloat __s3(-1.9841269841269841e-04);
  constexpr float __s4(2.7557319223985893e-06f);
  constexpr float __s5(-2.5052108385441719e-08f);
  constexpr float __s6(1.6059043836821615e-10f);
  constexpr float __s7(-7.6471637318198165e-13f);

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __x2 = __x * __x;
  float __x2f   = __x2.hi();

  float __qf = __s7;
  __qf       = ::cuda::__fpmp_fma_rn(__qf, __x2f, __s6);
  __qf       = ::cuda::__fpmp_fma_rn(__qf, __x2f, __s5);
  __qf       = ::cuda::__fpmp_fma_rn(__qf, __x2f, __s4);

  __ffloat __q = __qf * __x2 + __s3;
  __q          = __q * __x2 + __s2;
  __q          = __q * __x2 + __s1;

  __ffloat __result = renormalize(__q * __x2 * __x + __x);
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
}

/*
 * Cos kernel: evaluate cos(x) for |x| <= pi/4 using fp32mp2 Taylor series.
 * cos(x) = 1 + x^2*Q(x^2), Q(u) = Sum Taylor coefficients from -1/2! to 1/16!.
 * Upper terms (c8..c4) in single precision, lower terms (c3..c1) in fp32mp2.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos_kernel(_FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  constexpr float __c1(-0.5f);
  constexpr __ffloat __c2(4.1666666666666667e-02);
  constexpr __ffloat __c3(-1.3888888888888889e-03);
  constexpr float __c4(2.4801587301587302e-05f);
  constexpr float __c5(-2.7557319223985893e-07f);
  constexpr float __c6(2.0876756987868099e-09f);
  constexpr float __c7(-1.1470745597729725e-11f);
  constexpr float __c8(4.7794773323873853e-14f);

  __ffloat __x(__x_hi, __x_lo);
  __ffloat __x2 = __x * __x;
  float __x2f   = __x2.hi();

  float __qf = __c8;
  __qf       = ::cuda::__fpmp_fma_rn(__qf, __x2f, __c7);
  __qf       = ::cuda::__fpmp_fma_rn(__qf, __x2f, __c6);
  __qf       = ::cuda::__fpmp_fma_rn(__qf, __x2f, __c5);
  __qf       = ::cuda::__fpmp_fma_rn(__qf, __x2f, __c4);

  __ffloat __q = __qf * __x2 + __c3;
  __q          = __q * __x2 + __c2;
  __q          = __q * __x2 + __c1;

  __ffloat __result = renormalize(__q * __x2 + 1.0f);
  *__res_hi         = __result.hi();
  *__res_lo         = __result.lo();
}

/*
 * ====================================================================
 * Internal kernels for asin, acos, atan and atan2 (fp32mp2)
 * ====================================================================
 *
 * All four functions find the angle of a point (M, m), 0 <= m <= M, from one table:
 *
 *   atan(m / M) = atan(c) + atan(t),   c = i/16,   t = (m - c M) / (M + c m).
 *
 * i is the integer part of 16 m/M + 1/4 (from a single-precision quotient), which leaves
 * -1/64 <= t <= 3/64 and makes the leading words of m and c M cancel exactly (Sterbenz:
 * m / (c M) is within [3/4, 7/4] for i >= 1). The asymmetric interval keeps the first table
 * entry from cancelling half of its own magnitude against t.
 *
 *   atan(x), atan2(y, x):  (M, m) = (|x|, |y|) sorted, one pair division for t.
 *   asin(x), acos(x):      (M, m) = (|x|, sqrt((1 - |x|)(1 + |x|))) sorted, a point of the
 *                          unit circle. There t is replaced by the sine of the same angle,
 *                          (m - c M) cos(atan(c)), a product by a table entry instead of a
 *                          division, and the kernel is asin instead of atan. For i = 0 that
 *                          is m itself, so asin(x) for small x does not depend on the root.
 *
 * The octant is folded into the table: its four rows hold atan(c), pi/2 - atan(c),
 * pi/2 + atan(c) and pi - atan(c), and the odd rows take -t. The rows at c = 0 and c = 1
 * are also the exact results for zero and infinite arguments.
 *
 * Both kernels are t (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8), minimax on |t| <= 0.04689
 * (Sollya fpminimax, k1 and k2 as 48-bit pairs, k3 and k4 as single floats; relative error
 * below 2^-56.3 for atan and 2^-58.4 for asin).
 *
 * NaN, infinite and out-of-domain arguments are answered on entry: nothing below relies on
 * them propagating through the arithmetic.
 * ====================================================================
 */

struct alignas(8) __fpmp2_atan_entry
{
  float __hi;
  float __lo;
};

// Rounded from 400-bit values (Sollya). Hidden, like the functions that read it, so that
// shared libraries built against different versions of this table do not share one copy.
_CCCL_VISIBILITY_HIDDEN _CCCL_GLOBAL_CONSTANT __fpmp2_atan_entry __fpmp2_atan_table[5][17] = {
  // atan(i/16)
  {
    {0.0f, 0.0f},
    {0x1.ff55bcp-5f, -0x1.1a6042p-30f},
    {0x1.fd5baap-4f, -0x1.54f424p-30f},
    {0x1.7b97b4p-3f, 0x1.79cb6p-28f},
    {0x1.f5b76p-3f, -0x1.b4dfc8p-29f},
    {0x1.362774p-2f, -0x1.1f0286p-27f},
    {0x1.6f6194p-2f, 0x1.e4defp-30f},
    {0x1.a64eecp-2f, 0x1.e611fep-29f},
    {0x1.dac67p-2f, 0x1.586ed4p-28f},
    {0x1.0657eap-1f, -0x1.6499e6p-26f},
    {0x1.1e00bap-1f, 0x1.7bdfd6p-26f},
    {0x1.345f02p-1f, -0x1.98e422p-28f},
    {0x1.4978fap-1f, 0x1.934f7p-28f},
    {0x1.5d5898p-1f, 0x1.c5a6c6p-27f},
    {0x1.700a7cp-1f, 0x1.5e118cp-27f},
    {0x1.819d0cp-1f, -0x1.1d4eb6p-26f},
    {0x1.921fb6p-1f, -0x1.777a5cp-26f},
  },
  // pi/2 - atan(i/16)
  {
    {0x1.921fb6p0f, -0x1.777a5cp-25f},
    {0x1.822508p0f, -0x1.2ea75ap-25f},
    {0x1.7249fap0f, 0x1.532d44p-25f},
    {0x1.62acbep0f, 0x1.594c38p-25f},
    {0x1.5368cap0f, -0x1.5c2c6p-25f},
    {0x1.4495d8p0f, 0x1.a08c8ap-26f},
    {0x1.36475p0f, 0x1.e57aaep-27f},
    {0x1.288bfap0f, 0x1.a8920cp-27f},
    {0x1.1b6e1ap0f, -0x1.a28838p-25f},
    {0x1.0ef3cp0f, 0x1.3ad296p-25f},
    {0x1.031f58p0f, -0x1.ab5242p-28f},
    {0x1.efe068p-1f, 0x1.77444ep-26f},
    {0x1.dac67p-1f, 0x1.586ed4p-27f},
    {0x1.c6e6d2p-1f, 0x1.71bf18p-29f},
    {0x1.b434eep-1f, 0x1.8809fep-28f},
    {0x1.a2a26p-1f, -0x1.d1a604p-26f},
    {0x1.921fb6p-1f, -0x1.777a5cp-26f},
  },
  // pi/2 + atan(i/16)
  {
    {0x1.921fb6p0f, -0x1.777a5cp-25f},
    {0x1.a21a64p0f, -0x1.c04d6p-25f},
    {0x1.b1f57p0f, -0x1.0887f8p-27f},
    {0x1.c192acp0f, -0x1.2103c4p-27f},
    {0x1.d0d6a2p0f, -0x1.92c85ap-25f},
    {0x1.dfa992p0f, 0x1.031406p-27f},
    {0x1.edf81ap0f, 0x1.2f5936p-26f},
    {0x1.fbb37p0f, 0x1.4dcd86p-26f},
    {0x1.0468a8p1f, 0x1.59c9bep-24f},
    {0x1.0aa5d4p1f, 0x1.eb1c58p-24f},
    {0x1.10900ap1f, -0x1.5cc538p-24f},
    {0x1.16279cp1f, -0x1.d54b7p-24f},
    {0x1.1b6e1ap1f, -0x1.a28838p-24f},
    {0x1.2066p1f, 0x1.7cf7aap-24f},
    {0x1.25127ap1f, -0x1.1ff5fap-25f},
    {0x1.29771ep1f, -0x1.0310dcp-24f},
    {0x1.2d97c8p1f, -0x1.99bc5cp-28f},
  },
  // pi - atan(i/16)
  {
    {0x1.921fb6p1f, -0x1.777a5cp-24f},
    {0x1.8a225ep1f, 0x1.59de48p-25f},
    {0x1.8234d8p1f, -0x1.2268c6p-28f},
    {0x1.7a663ap1f, -0x1.e2e26p-29f},
    {0x1.72c44p1f, -0x1.69d35ep-24f},
    {0x1.6b5ac6p1f, 0x1.ac65f4p-24f},
    {0x1.643382p1f, 0x1.80f228p-24f},
    {0x1.5d55d8p1f, -0x1.0d55dap-25f},
    {0x1.56c6e8p1f, -0x1.8d014ap-24f},
    {0x1.5089bap1f, 0x1.e1ac1cp-24f},
    {0x1.4a9f86p1f, 0x1.298daep-24f},
    {0x1.4507f4p1f, 0x1.a213e6p-24f},
    {0x1.3fc176p1f, 0x1.6f50acp-24f},
    {0x1.3ac99p1f, -0x1.b02f36p-24f},
    {0x1.361d16p1f, 0x1.730dc6p-26f},
    {0x1.31b872p1f, 0x1.9fb2a2p-25f},
    {0x1.2d97c8p1f, -0x1.99bc5cp-28f},
  },
  // cos(atan(i/16))
  {
    {1.0f, 0.0f},
    {0x1.ff00cp-1f, -0x1.3ee8fcp-26f},
    {0x1.fc0bd8p-1f, 0x1.141e3cp-26f},
    {0x1.f73b06p-1f, -0x1.3e058ep-30f},
    {0x1.f0b684p-1f, 0x1.1a55e4p-26f},
    {0x1.e8b19ap-1f, -0x1.ed38d4p-27f},
    {0x1.df668ap-1f, -0x1.20672cp-27f},
    {0x1.d51292p-1f, 0x1.fd94eap-26f},
    {0x1.c9f25cp-1f, 0x1.6ffb76p-27f},
    {0x1.be3f26p-1f, 0x1.7e31cp-29f},
    {0x1.b22cd6p-1f, -0x1.24f05p-26f},
    {0x1.a5e8d2p-1f, 0x1.6be074p-26f},
    {0x1.99999ap-1f, -0x1.99999ap-27f},
    {0x1.8d5edp-1f, 0x1.c3bcap-27f},
    {0x1.8151bcp-1f, -0x1.e40478p-27f},
    {0x1.7585e6p-1f, 0x1.03fcdep-26f},
    {0x1.6a09e6p-1f, 0x1.9fcef4p-27f},
  },
};

/*
 * atan(m / M) for pairs 0 <= m <= M (to within rounding), M normal, folded into `__row` of
 * the table; `__rho` is m / M in single precision. With `__unit`, (M, m) is a point of the
 * unit circle and the reduced argument is a sine (see above).
 *
 * In the kernel, k3 and k4 run in single precision; the step at k2 adds its product to the
 * low word of k2, which is exact enough because the product is below 2^-9 of k2. The
 * remaining steps are pair steps whose sums are Fast2Sums: each adds a product to a term
 * at least 2^8 times larger.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan_kernel(
  const bool __unit,
  const _FpType __m_hi,
  const _FpType __m_lo,
  const _FpType __M_hi,
  const _FpType __M_lo,
  const _FpType __rho,
  const int __row,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const float __k1_hi = __unit ? 0x1.555556p-3f : -0x1.555556p-2f;
  const float __k1_lo = __unit ? -0x1.5555fcp-28f : 0x1.5556a8p-27f;
  const float __k2_hi = __unit ? 0x1.333334p-4f : 0x1.99999ap-3f;
  const float __k2_lo = __unit ? -0x1.834838p-29f : -0x1.f455e8p-29f;
  const float __k3    = __unit ? 0x1.6db664p-5f : -0x1.2491dp-3f;
  const float __k4    = __unit ? 0x1.f3bf64p-6f : 0x1.c51d9ap-4f;

  float __e;

  const int __i   = static_cast<int>(::cuda::__fpmp_fma_rn(16.0f, __rho, 0.25f));
  const float __c = ::cuda::__fpmp_mul_rn(static_cast<float>(__i), 0.0625f);

  // n = m - c M: the high words cancel exactly, and the difference is renormalized because
  // its low word can exceed what is left of the high one
  float __cM_lo;
  const float __cM_hi = ::cuda::__fpmp_two_mult_fma(__c, __M_hi, &__cM_lo);
  __cM_lo             = ::cuda::__fpmp_fma_rn(__c, __M_lo, __cM_lo);
  float __n_lo;
  const float __n_hi =
    ::cuda::__fpmp_two_sum(::cuda::__fpmp_sub_rn(__m_hi, __cM_hi), ::cuda::__fpmp_sub_rn(__m_lo, __cM_lo), &__n_lo);

  float __t_hi;
  float __t_lo;
  if (__unit)
  {
    // t = n cos(atan(c))
    const __fpmp2_atan_entry __cos = __fpmp2_atan_table[4][__i];
    __t_hi                         = ::cuda::__fpmp_two_mult_fma(__n_hi, __cos.__hi, &__t_lo);
    __t_lo                         = ::cuda::__fpmp_fma_rn(__n_hi, __cos.__lo, __t_lo);
    __t_lo                         = ::cuda::__fpmp_fma_rn(__n_lo, __cos.__hi, __t_lo);
  }
  else
  {
    // t = n / (M + c m)
    float __cm_lo;
    const float __cm_hi = ::cuda::__fpmp_two_mult_fma(__c, __m_hi, &__cm_lo);
    __cm_lo             = ::cuda::__fpmp_fma_rn(__c, __m_lo, __cm_lo);
    const float __d_hi  = ::cuda::__fpmp_fast_two_sum(__M_hi, __cm_hi, &__e);
    const float __d_lo  = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__M_lo, __cm_lo), __e);
    ::cuda::__fpmp2_div(__n_hi, __n_lo, __d_hi, __d_lo, &__t_hi, &__t_lo);
  }
  if ((__row & 1) != 0)
  {
    __t_hi = -__t_hi;
    __t_lo = -__t_lo;
  }

  // u = t^2
  float __u_lo;
  const float __u_hi = ::cuda::__fpmp_two_mult_fma(__t_hi, __t_hi, &__u_lo);
  __u_lo             = ::cuda::__fpmp_fma_rn(::cuda::__fpmp_add_rn(__t_hi, __t_hi), __t_lo, __u_lo);

  // q2 = k2 + u (k3 + u k4), as k2_hi plus a low word
  const float __w     = ::cuda::__fpmp_fma_rn(__k4, __u_hi, __k3);
  const float __q2_lo = ::cuda::__fpmp_fma_rn(__w, __u_hi, __k2_lo);

  // q1 = k1 + u q2
  float __p_lo;
  const float __p_hi  = ::cuda::__fpmp_two_mult_fma(__k2_hi, __u_hi, &__p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__q2_lo, __u_hi, __p_lo);
  __p_lo              = ::cuda::__fpmp_fma_rn(__k2_hi, __u_lo, __p_lo);
  const float __q1_hi = ::cuda::__fpmp_fast_two_sum(__k1_hi, __p_hi, &__e);
  const float __q1_lo = ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__k1_lo, __p_lo), __e);

  // z = u q1
  const float __z_hi = ::cuda::__fpmp_two_mult_fma(__q1_hi, __u_hi, &__p_lo);
  __p_lo             = ::cuda::__fpmp_fma_rn(__q1_hi, __u_lo, __p_lo);
  const float __z_lo = ::cuda::__fpmp_fma_rn(__q1_lo, __u_hi, __p_lo);

  // y = t z
  const float __y_hi = ::cuda::__fpmp_two_mult_fma(__t_hi, __z_hi, &__p_lo);
  __p_lo             = ::cuda::__fpmp_fma_rn(__t_hi, __z_lo, __p_lo);
  const float __y_lo = ::cuda::__fpmp_fma_rn(__t_lo, __z_hi, __p_lo);

  // table entry + t + y: the entry is zero or exceeds |t|, and |y| < 2^-9 |t|
  const __fpmp2_atan_entry __s = __fpmp2_atan_table[__row][__i];
  float __e2;
  const float __h1 = ::cuda::__fpmp_fast_two_sum(__s.__hi, __t_hi, &__e);
  const float __h2 = ::cuda::__fpmp_fast_two_sum(__h1, __y_hi, &__e2);
  const float __l  = ::cuda::__fpmp_add_rn(
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__e, __e2), __s.__lo), ::cuda::__fpmp_add_rn(__t_lo, __y_lo));
  *__res_hi = ::cuda::__fpmp_fast_two_sum(__h2, __l, __res_lo);
}

/*
 * The pair s = sqrt((1 - a)(1 + a)) for a pair 0 <= a <= 1, and min(a, s) / max(a, s) in
 * single precision (`__rho`).
 *
 * 1 - a and 1 + a are exact up to the rounding of their low words, so the product keeps its
 * relative accuracy as a approaches 1. The root is y ~ sqrt(v) and h ~ 1 / (2 sqrt(v)) in
 * single precision, then the residual v - y^2 (exact in its high word) times h. The host takes
 * y from sqrtf; the device refines its approximate reciprocal square root by one coupled
 * Newton step on y and h, and since that instruction flushes subnormals to zero, a product
 * below 2^-64 is scaled first.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __internal_fpmp2_asin_cofunction(
  const _FpType __a_hi, const _FpType __a_lo, _FpType* __s_hi, _FpType* __s_lo, _FpType* __rho) noexcept
{
  float __e;
  float __om_lo;
  const float __h     = ::cuda::__fpmp_fast_two_sum(1.0f, -__a_hi, &__e);
  const float __om_hi = ::cuda::__fpmp_fast_two_sum(__h, ::cuda::__fpmp_sub_rn(__e, __a_lo), &__om_lo);
  const float __op_hi = ::cuda::__fpmp_fast_two_sum(1.0f, __a_hi, &__e);
  const float __op_lo = ::cuda::__fpmp_add_rn(__e, __a_lo);

  float __v_lo;
  float __v_hi = ::cuda::__fpmp_two_mult_fma(__om_hi, __op_hi, &__v_lo);
  __v_lo       = ::cuda::__fpmp_fma_rn(__om_hi, __op_lo, __v_lo);
  __v_lo       = ::cuda::__fpmp_fma_rn(__om_lo, __op_hi, __v_lo);

  const bool __tiny  = __v_hi < 0x1p-64f;
  const float __up   = __tiny ? 0x1p64f : 1.0f;
  const float __down = __tiny ? 0x1p-32f : 1.0f;
  __v_hi             = ::cuda::__fpmp_mul_rn(__v_hi, __up);
  __v_lo             = ::cuda::__fpmp_mul_rn(__v_lo, __up);

  // v = 0 only for a = 1
  float __y;
  float __hh;
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      const float __r0 = (__v_hi > 0.0f) ? ::cuda::__fpmp_rsqrt_rn(__v_hi) : 0.0f;
      const float __y0 = ::cuda::__fpmp_mul_rn(__v_hi, __r0);
      const float __h0 = ::cuda::__fpmp_mul_rn(0.5f, __r0);
      const float __d  = ::cuda::__fpmp_fma_rn(-__y0, __h0, 0.5f);
      __y              = ::cuda::__fpmp_fma_rn(__y0, __d, __y0);
      __hh             = ::cuda::__fpmp_fma_rn(__h0, __d, __h0);
    }),
    ({
      __y  = ::sqrtf(__v_hi);
      __hh = (__v_hi > 0.0f) ? 0.5f / __y : 0.0f;
    }))
  float __resid = ::cuda::__fpmp_fma_rn(-__y, __y, __v_hi);
  __resid       = ::cuda::__fpmp_add_rn(__resid, __v_lo);

  *__s_hi = ::cuda::__fpmp_mul_rn(__y, __down);
  *__s_lo = ::cuda::__fpmp_mul_rn(::cuda::__fpmp_mul_rn_nocontract(__resid, __hh), __down);

  // a > s: s / a, the only case the scaling above can apply to; else a / s, with 1 / s = 2 h
  *__rho = (__a_hi > *__s_hi) ? ::cuda::__fpmp_mul_rn(*__s_hi, ::cuda::__fpmp_rcp_rn(__a_hi))
                              : ::cuda::__fpmp_mul_rn(__a_hi, ::cuda::__fpmp_add_rn(__hh, __hh));
}

/*
 * ====================================================================
 * sincos(x, &s, &c) - sine and cosine of the same argument
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine and cosine sincos(x, &s, &c) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * sincos for fp32mp2: compute sin(x) and cos(x) simultaneously.
 * Shared argument reduction, separate sin/cos kernels on [-pi/4, pi/4],
 * quadrant-based swap and sign adjustment.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincos(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
  ::cuda::__internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);

  int __q = __quadrant & 3;
  if (__q < 0)
  {
    __q += 4;
  }

  if (__q & 1)
  {
    float __t;
    __t    = __s_hi;
    __s_hi = __c_hi;
    __c_hi = __t;
    __t    = __s_lo;
    __s_lo = __c_lo;
    __c_lo = __t;
  }
  if (__q == 1 || __q == 2)
  {
    __c_hi = -__c_hi;
    __c_lo = -__c_lo;
  }
  if (__q == 2 || __q == 3)
  {
    __s_hi = -__s_hi;
    __s_lo = -__s_lo;
  }

  *__sin_hi = __s_hi;
  *__sin_lo = __s_lo;
  *__cos_hi = __c_hi;
  *__cos_lo = __c_lo;
}

/*
 * --------------------------------------------------------------------
 * Sine and cosine sincos(x, &s, &c) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincos(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(sin, _CCCL_FPMP_SINQ, __x_hi, __x_lo, __sin_hi, __sin_lo);
  _CCCL_FPMP_CALL_FP64MP2_MATH(cos, _CCCL_FPMP_COSQ, __x_hi, __x_lo, __cos_hi, __cos_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(sincos)

/*
 * ====================================================================
 * sin(x) - sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Sine sin(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * sin for fp32mp2: calls sincos and returns only the sine.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __c_hi, __c_lo;
  ::cuda::__fpmp2_sincos(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
}

/*
 * --------------------------------------------------------------------
 * Sine sin(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(sin, _CCCL_FPMP_SINQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(sin)

/*
 * ====================================================================
 * cos(x) - cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cosine cos(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * cos for fp32mp2: calls sincos and returns only the cosine.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __s_hi, __s_lo;
  ::cuda::__fpmp2_sincos(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Cosine cos(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(cos, _CCCL_FPMP_COSQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(cos)

/*
 * ====================================================================
 * tan(x) - tangent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Tangent tan(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * Algorithm (no FP64 dependency on the hot path):
 *   1. Reduce x to r in [-pi/4, pi/4] via the shared
 *      __internal_fpmp2_trig_reduction; this also returns the quadrant
 *      index q (modulo 4).
 *   2. Evaluate sin(r) and cos(r) on the reduced interval via the
 *      shared __internal_fpmp2_sin_kernel / __internal_fpmp2_cos_kernel.
 *   3. tan has period pi, so only the LSB of q matters:
 *        q even  ->  tan(x) =  sin(r) / cos(r)
 *        q odd   ->  tan(x) = -cos(r) / sin(r)        (= -cot(r))
 *      The full quadrant-mod-4 sign dance used by sincos is unnecessary
 *      here because tan(x + pi) = tan(x) absorbs the q == 2,3 sign
 *      flips that sincos performs on its sin/cos outputs.
 *
 * One shared reduction + one + sin kernel + one cos kernel + one fp32mp2 division.  Reusing the
 * already-tuned sin/cos kernels inherits their ~46-bit accuracy
 * envelope without having to fit a separate tan polynomial.
 *
 * Singularities at x === pi/2 (mod pi) produce +-inf through the q-odd
 * branch when sin(r) underflows to zero (matches the IEEE
 * convention; signed-infinity direction follows the rounded reduced
 * argument).  Inf / NaN inputs propagate to NaN through the reduction.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  int __quadrant;
  float __r_hi, __r_lo;
  ::cuda::__internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
  ::cuda::__internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);

  using __mp2_t = fpmp2<float>;
  __mp2_t __s(__s_hi, __s_lo);
  __mp2_t __c(__c_hi, __c_lo);

  __mp2_t __result = (__quadrant & 1) ? __mp2_t(-__c / __s) : __mp2_t(__s / __c);

  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

/*
 * --------------------------------------------------------------------
 * Tangent tan(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_tan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH(tan, _CCCL_FPMP_TANQ, __x_hi, __x_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(tan)

/*
 * ====================================================================
 * atan(x) - arc tangent
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc tangent atan(x) (fp32mp2)
 * --------------------------------------------------------------------
 * atan(|x|) is the angle of (1, |x|), or of (|x|, 1) taken from pi/2 for |x| > 1, through
 * the shared kernel. Beyond 2^60, and for an infinite x, the result is +-pi/2 to the last
 * bit; a NaN gives NaN in both limbs. The sign is applied last, so atan(-0) = -0.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  const uint32_t __x_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const bool __neg        = (__x_bits & 0x80000000u) != 0u;
  const float __a_hi      = ::cuda::std::bit_cast<float>(__x_bits & 0x7FFFFFFFu);
  const float __a_lo      = __neg ? -__x_lo : __x_lo;

  float __r_hi;
  float __r_lo;
  if (!(__a_hi <= 0x1p60f))
  {
    if (__a_hi != __a_hi)
    {
      *__res_hi = __x_hi;
      *__res_lo = __x_hi;
      return;
    }
    __r_hi = __fpmp2_atan_table[1][0].__hi;
    __r_lo = __fpmp2_atan_table[1][0].__lo;
  }
  else
  {
    // |x| > 1: the angle is taken from pi/2
    const bool __swap = __a_hi > 1.0f;
    const float __rho = __swap ? ::cuda::__fpmp_rcp_rn(__a_hi) : __a_hi;
    ::cuda::__internal_fpmp2_atan_kernel(
      false,
      __swap ? 1.0f : __a_hi,
      __swap ? 0.0f : __a_lo,
      __swap ? __a_hi : 1.0f,
      __swap ? __a_lo : 0.0f,
      __rho,
      __swap ? 1 : 0,
      &__r_hi,
      &__r_lo);
  }

  *__res_hi = __neg ? -__r_hi : __r_hi;
  *__res_lo = __neg ? -__r_lo : __r_lo;
}

/*
 * A NaN result of the fp64mp2 inverse trigonometric functions goes into both limbs, as in
 * their fp32mp2 counterparts.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_nan_both_limbs(double* __res_hi, double* __res_lo) noexcept
{
  if (*__res_hi != *__res_hi)
  {
    *__res_lo = *__res_hi;
  }
}

/*
 * --------------------------------------------------------------------
 * Arc tangent atan(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_atan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  if (__x_hi == 0.0) // atan(+-0) = +-0: the conversions do not keep the sign of zero
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  _CCCL_FPMP_CALL_FP64MP2_MATH(atan, _CCCL_FPMP_ATANQ, __x_hi, __x_lo, __res_hi, __res_lo);
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(atan)

/*
 * ====================================================================
 * atan2(y, x) - arc tangent of y/x, quadrant aware
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc tangent atan2(y, x) (fp32mp2)
 * --------------------------------------------------------------------
 * The angle of (max(|x|, |y|), min(|x|, |y|)) through the shared kernel, in the table row
 * of its octant, with the sign of y applied last. Special values follow C99 F.10.1.4:
 *
 *   atan2(+-0, x)   = +-0 for x > 0 or +0, +-pi for x < 0 or -0
 *   atan2(y, +-0)   = +-pi/2 for y != 0
 *   atan2(+-inf, x) = +-pi/2 for finite x, +-pi/4 for +inf, +-3pi/4 for -inf
 *   atan2(y, +inf)  = +-0, atan2(y, -inf) = +-pi for finite y
 *   NaN in either argument gives NaN in both limbs
 *
 * These are the table entries at c = 0 and c = 1 of the same row. Signs are those of the
 * high limbs.
 *
 * The operands are scaled by a common power of two that brings the larger one into
 * [1, 2^64): its reciprocal then stays normal (the device's approximate reciprocal flushes
 * subnormals to zero), the divisor cannot overflow, and the quotient keeps its low word
 * whenever the result has room for one. Subnormal operands are ordinary inputs.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan2(
  const float __y_hi,
  const float __y_lo,
  const float __x_hi,
  const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  const uint32_t __x_bits  = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __y_bits  = ::cuda::std::bit_cast<uint32_t>(__y_hi);
  const uint32_t __ax_bits = __x_bits & 0x7FFFFFFFu;
  const uint32_t __ay_bits = __y_bits & 0x7FFFFFFFu;
  const bool __x_neg       = (__x_bits & 0x80000000u) != 0u;
  const bool __y_neg       = (__y_bits & 0x80000000u) != 0u;

  // |y| > |x|: the angle is taken from the y axis
  const bool __swap = __ay_bits > __ax_bits;
  const int __row   = (__x_neg ? 3 : 0) ^ (__swap ? 1 : 0);

  float __r_hi;
  float __r_lo;
  if (__ax_bits >= 0x7F800000u || __ay_bits >= 0x7F800000u || (__ax_bits | __ay_bits) == 0u)
  {
    if (__ax_bits > 0x7F800000u || __ay_bits > 0x7F800000u)
    {
      *__res_hi = __x_hi + __y_hi;
      *__res_lo = *__res_hi;
      return;
    }
    // Both infinite: the diagonal. Otherwise an axis: one operand infinite, or both zero.
    const __fpmp2_atan_entry __s = __fpmp2_atan_table[__row][(__ax_bits == __ay_bits && __ax_bits != 0u) ? 16 : 0];
    __r_hi                       = __s.__hi;
    __r_lo                       = __s.__lo;
  }
  else
  {
    const float __ax_hi = ::cuda::std::bit_cast<float>(__ax_bits);
    const float __ay_hi = ::cuda::std::bit_cast<float>(__ay_bits);
    const float __ax_lo = __x_neg ? -__x_lo : __x_lo;
    const float __ay_lo = __y_neg ? -__y_lo : __y_lo;
    float __m_hi        = __swap ? __ax_hi : __ay_hi;
    float __m_lo        = __swap ? __ax_lo : __ay_lo;
    float __M_hi        = __swap ? __ay_hi : __ax_hi;
    float __M_lo        = __swap ? __ay_lo : __ax_lo;

    const float __scale = (__M_hi < 1.0f) ? 0x1p64f : ((__M_hi >= 0x1p64f) ? 0x1p-64f : 1.0f);
    __m_hi              = ::cuda::__fpmp_mul_rn(__m_hi, __scale);
    __m_lo              = ::cuda::__fpmp_mul_rn(__m_lo, __scale);
    __M_hi              = ::cuda::__fpmp_mul_rn(__M_hi, __scale);
    __M_lo              = ::cuda::__fpmp_mul_rn(__M_lo, __scale);
    if (__M_hi < 1.0f) // both operands below 2^-64
    {
      __m_hi = ::cuda::__fpmp_mul_rn(__m_hi, 0x1p86f);
      __m_lo = ::cuda::__fpmp_mul_rn(__m_lo, 0x1p86f);
      __M_hi = ::cuda::__fpmp_mul_rn(__M_hi, 0x1p86f);
      __M_lo = ::cuda::__fpmp_mul_rn(__M_lo, 0x1p86f);
    }

    const float __rho = ::cuda::__fpmp_mul_rn(__m_hi, ::cuda::__fpmp_rcp_rn(__M_hi));
    ::cuda::__internal_fpmp2_atan_kernel(false, __m_hi, __m_lo, __M_hi, __M_lo, __rho, __row, &__r_hi, &__r_lo);
  }

  *__res_hi = __y_neg ? -__r_hi : __r_hi;
  *__res_lo = __y_neg ? -__r_lo : __r_lo;
}

/*
 * --------------------------------------------------------------------
 * Arc tangent atan2(y, x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * Note: On CUDA device, _CCCL_FPMP_ATAN2Q widens through double atan2 (no fp128 intrinsic); _CCCL_FPMP_CBRTQ is
 * reconstructed from __nv_fp128_pow.
 *
 * The conversions do not keep the sign of a zero y, so that case is answered here:
 * atan2(+-0, x) = +-0 for x > 0 or +0, +-pi for x < 0 or -0.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_atan2(
  const double __y_hi,
  const double __y_lo,
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  if (__y_hi == 0.0 && __x_hi == __x_hi)
  {
    const bool __x_neg = ::cuda::std::signbit(__x_hi);
    const bool __y_neg = ::cuda::std::signbit(__y_hi);
    const double __hi  = __x_neg ? 0x1.921fb54442d18p+1 : 0.0; // pi = __hi + __lo
    const double __lo  = __x_neg ? 0x1.1a62633145c07p-53 : 0.0;
    *__res_hi          = __y_neg ? -__hi : __hi;
    *__res_lo          = __y_neg ? -__lo : __lo;
    return;
  }
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  __fpmp_fp128 __res =
    _CCCL_FPMP_ATAN2Q(::cuda::__fpmp2_to_quad(__y_hi, __y_lo), ::cuda::__fpmp2_to_quad(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_quad(__res, __res_hi, __res_lo);
#  else
  double __res =
    ::cuda::std::atan2(::cuda::__fpmp2_to_double(__y_hi, __y_lo), ::cuda::__fpmp2_to_double(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_double(__res, __res_hi, __res_lo);
#  endif
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
} // __internal_fpmp2_atan2

_CCCL_FPMP_MATH_DISPATCH_2A_YX(atan2)

/*
 * ====================================================================
 * asin(x) - arc sine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc sine asin(x) (fp32mp2)
 * --------------------------------------------------------------------
 * asin(|x|) is the angle of the unit-circle point (s, |x|), s = sqrt((1 - |x|)(1 + |x|)),
 * through the shared kernel; the sign is applied last, so asin(-0) = -0. |x| > 1 (decided
 * on both limbs) and NaN give NaN in both limbs.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  const uint32_t __x_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const bool __neg        = (__x_bits & 0x80000000u) != 0u;
  const float __a_hi      = ::cuda::std::bit_cast<float>(__x_bits & 0x7FFFFFFFu);
  const float __a_lo      = __neg ? -__x_lo : __x_lo;

  if (!(__a_hi <= 1.0f) || (__a_hi == 1.0f && __a_lo > 0.0f))
  {
    *__res_hi = (__a_hi != __a_hi) ? __x_hi : ::cuda::std::__fp_nan<float>();
    *__res_lo = *__res_hi;
    return;
  }

  float __s_hi;
  float __s_lo;
  float __rho;
  ::cuda::__internal_fpmp2_asin_cofunction(__a_hi, __a_lo, &__s_hi, &__s_lo, &__rho);

  // |x| > s: the angle is taken from pi/2
  const bool __swap = __a_hi > __s_hi;
  float __r_hi;
  float __r_lo;
  ::cuda::__internal_fpmp2_atan_kernel(
    true,
    __swap ? __s_hi : __a_hi,
    __swap ? __s_lo : __a_lo,
    __swap ? __a_hi : __s_hi,
    __swap ? __a_lo : __s_lo,
    __rho,
    __swap ? 1 : 0,
    &__r_hi,
    &__r_lo);

  *__res_hi = __neg ? -__r_hi : __r_hi;
  *__res_lo = __neg ? -__r_lo : __r_lo;
}

/*
 * --------------------------------------------------------------------
 * asin and acos of an fp64mp2 x with 1/2 <= |x| <= 1
 * --------------------------------------------------------------------
 * There the functions depend on 1 - |x|: acos(|x|) = h = 2 asin(sqrt((1 - |x|) / 2)), which is
 * small near 1, and asin(|x|) = pi/2 - h, acos(-|x|) = pi - h. The sum of the limbs in
 * binary128 (or binary64) drops a low limb beyond its precision, and with it 1 - |x|, so
 * 1 - |x| is formed from the limbs instead: the high limbs subtract exactly. `__form` selects
 * h (0), pi - h (1) or pi/2 - h (2). |x| > 1 gives NaN through the square root.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_acos_from_one(
  const double __a_hi, const double __a_lo, const int __form, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  // pi to 113 bits, as the sum of three doubles
  const __fpmp_fp128 __pi =
    static_cast<__fpmp_fp128>(0x1.921fb54442d18p+1) + static_cast<__fpmp_fp128>(0x1.1a62633145c07p-53)
    + static_cast<__fpmp_fp128>(-0x1.f1976b7ed8fbcp-109);
  const __fpmp_fp128 __two = static_cast<__fpmp_fp128>(2);
  const __fpmp_fp128 __d   = static_cast<__fpmp_fp128>(1.0 - __a_hi) - static_cast<__fpmp_fp128>(__a_lo);
  const __fpmp_fp128 __h   = __two * _CCCL_FPMP_ASINQ(_CCCL_FPMP_SQRTQ(__d / __two));
  ::cuda::__fpmp2_from_quad((__form == 0) ? __h : ((__form == 1) ? __pi - __h : __pi / __two - __h), __res_hi, __res_lo);
#  else
  constexpr double __pi = 0x1.921fb54442d18p+1;
  const double __d      = (1.0 - __a_hi) - __a_lo;
  // sqrt(1/2) sqrt(d) rather than sqrt(d / 2), which underflows for the smallest subnormal d
  const double __h = 2.0 * ::asin(0x1.6a09e667f3bcdp-1 * ::sqrt(__d));
  ::cuda::__fpmp2_from_double((__form == 0) ? __h : ((__form == 1) ? __pi - __h : 0.5 * __pi - __h), __res_hi, __res_lo);
#  endif
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Arc sine asin(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * asin(+-0) = +-0 is answered here (the conversions lose the sign of zero), and |x| >= 1/2
 * goes through 1 - |x|, which also decides the domain on both limbs.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_asin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  if (__x_hi == 0.0)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  const bool __neg    = __x_hi < 0.0;
  const double __a_hi = __neg ? -__x_hi : __x_hi;
  if (__a_hi >= 0.5)
  {
    ::cuda::__internal_fpmp2_acos_from_one(__a_hi, __neg ? -__x_lo : __x_lo, 2, __res_hi, __res_lo);
    if (__neg)
    {
      *__res_hi = -*__res_hi;
      *__res_lo = -*__res_lo;
    }
    return;
  }
  _CCCL_FPMP_CALL_FP64MP2_MATH(asin, _CCCL_FPMP_ASINQ, __x_hi, __x_lo, __res_hi, __res_lo);
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(asin)

/*
 * ====================================================================
 * acos(x) - arc cosine
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Arc cosine acos(x) (fp32mp2)
 * --------------------------------------------------------------------
 * acos(x) is the angle of the unit-circle point (x, s), s = sqrt((1 - |x|)(1 + |x|)),
 * through the shared kernel, in the table rows of the right or the left half plane. The
 * product form of s keeps the result accurate for x near 1, where it is small, and
 * acos(1) = +0. |x| > 1 (decided on both limbs) and NaN give NaN in both limbs.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  const uint32_t __x_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const bool __neg        = (__x_bits & 0x80000000u) != 0u;
  const float __a_hi      = ::cuda::std::bit_cast<float>(__x_bits & 0x7FFFFFFFu);
  const float __a_lo      = __neg ? -__x_lo : __x_lo;

  if (!(__a_hi <= 1.0f) || (__a_hi == 1.0f && __a_lo > 0.0f))
  {
    *__res_hi = (__a_hi != __a_hi) ? __x_hi : ::cuda::std::__fp_nan<float>();
    *__res_lo = *__res_hi;
    return;
  }

  float __s_hi;
  float __s_lo;
  float __rho;
  ::cuda::__internal_fpmp2_asin_cofunction(__a_hi, __a_lo, &__s_hi, &__s_lo, &__rho);

  // s > |x|: the angle is taken from pi/2
  const bool __swap = __s_hi > __a_hi;
  ::cuda::__internal_fpmp2_atan_kernel(
    true,
    __swap ? __a_hi : __s_hi,
    __swap ? __a_lo : __s_lo,
    __swap ? __s_hi : __a_hi,
    __swap ? __s_lo : __a_lo,
    __rho,
    (__neg ? 3 : 0) ^ (__swap ? 1 : 0),
    __res_hi,
    __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Arc cosine acos(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 * |x| >= 1/2 goes through 1 - |x|, which keeps acos accurate where it is small (x near 1) and
 * decides the domain on both limbs.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_acos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  const bool __neg    = __x_hi < 0.0;
  const double __a_hi = __neg ? -__x_hi : __x_hi;
  if (__a_hi >= 0.5)
  {
    ::cuda::__internal_fpmp2_acos_from_one(__a_hi, __neg ? -__x_lo : __x_lo, __neg ? 1 : 0, __res_hi, __res_lo);
    return;
  }
  _CCCL_FPMP_CALL_FP64MP2_MATH(acos, _CCCL_FPMP_ACOSQ, __x_hi, __x_lo, __res_hi, __res_lo);
  ::cuda::__internal_fpmp2_nan_both_limbs(__res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_1A(acos)

/*
 * ====================================================================
 * sinpi(x), cospi(x), sincospi(x) - sin(pi*x) and cos(pi*x)
 * ====================================================================
 * fp32mp2 reduces x = n + f, |f| <= 1/2, by splitting the limb pair.
 * f is folded into [-1/4, 1/4], scaled by pi, and evaluated with the
 * sincos Taylor kernels. No radian argument reduction is involved.
 * fp64mp2 remains a double-precision fallback.
 */

_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept;

/*
 * --------------------------------------------------------------------
 * Sine of pi*x sinpi(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinpi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sincospi(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
}

/*
 * --------------------------------------------------------------------
 * Sine of pi*x sinpi(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_sinpi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::sinpi(__xd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(::cuda::std::sin(__xd * ::cuda::std::__numbers<double>::__pi()), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(sinpi)

/*
 * ====================================================================
 * cospi(x) - cos(pi * x)
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cosine of pi*x cospi(x) (fp32mp2)
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cospi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  float __s_hi, __s_lo;
  ::cuda::__internal_fpmp2_sincospi(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
}

/*
 * --------------------------------------------------------------------
 * Cosine of pi*x cospi(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cospi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (::cuda::__fpmp2_from_double(::cospi(__xd), __res_hi, __res_lo);),
    (::cuda::__fpmp2_from_double(::cuda::std::cos(__xd * ::cuda::std::__numbers<double>::__pi()), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(cospi)

/*
 * ====================================================================
 * sincospi(x, &s, &c) - sin(pi * x) and cos(pi * x)
 * ====================================================================
 */

/*
 * Split a float into a nearby integer and an exact residual in [-1/2, 1/2].
 * The integer is returned mod 2; only its parity changes sin(pi*x) and cos(pi*x).
 */
[[nodiscard]] _CCCL_FPMP_CORE_API int __internal_fpmp2_pi_limb(float __x, float* __f) noexcept
{
  // |x| >= 2^23: ulp >= 1, so a finite float is an integer and the fraction is 0.
  // |x| >= 2^24: ulp >= 2, so it is an even integer and its parity is 0.
  constexpr float __integral_threshold      = 0x1p23f;
  constexpr float __even_integral_threshold = 0x1p24f;

  const float __abs = ::cuda::std::fabs(__x);
  if (__abs >= __even_integral_threshold)
  {
    *__f = 0.0f;
    return 0;
  }
  if (__abs >= __integral_threshold)
  {
    const int __n = ::cuda::__fpmp_fp2int_rn(__x);
    *__f          = 0.0f;
    return __n & 1;
  }

  int __n          = ::cuda::__fpmp_fp2int_rn(__x);
  const float __nf = ::cuda::__fpmp_int2fp_rn<float>(__n);
  float __frac     = __x - __nf;
  if (__frac > 0.5f)
  {
    ++__n;
    __frac -= 1.0f;
  }
  else if (__frac < -0.5f)
  {
    --__n;
    __frac += 1.0f;
  }
  *__f = __frac;
  return __n & 1;
}

/*
 * --------------------------------------------------------------------
 * Sine and cosine of pi*x sincospi(x, &s, &c) (fp32mp2)
 * --------------------------------------------------------------------
 * x = n + f, |f| <= 1/2. Fold f into [-1/4, 1/4] so pi*f is inside the
 * Taylor kernels, then apply the cofunction identity and the sign (-1)^n.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  using __afloat = fp32mp2_high;

  const float __abs_hi = ::cuda::std::fabs(__x_hi);
  // Inf and NaN share an all-ones exponent. isfinite(-0) is true, so this does not
  // treat a signed zero as non-finite the way a raw sign-bit check would.
  if (!::cuda::std::isfinite(__x_hi) || !::cuda::std::isfinite(__x_lo))
  {
    *__sin_hi = ::cuda::std::numeric_limits<float>::quiet_NaN();
    *__sin_lo = 0.0f;
    *__cos_hi = ::cuda::std::numeric_limits<float>::quiet_NaN();
    *__cos_lo = 0.0f;
    return;
  }

  int __odd;
  float __f_hi;
  float __f_lo;
  // Same thresholds as __internal_fpmp2_pi_limb: below 2^23 the high limb still has a fraction.
  constexpr float __integral_threshold      = 0x1p23f;
  constexpr float __even_integral_threshold = 0x1p24f;
  if (__abs_hi < __integral_threshold)
  {
    int __n           = ::cuda::__fpmp_fp2int_rn(__x_hi);
    const float __n_f = ::cuda::__fpmp_int2fp_rn<float>(__n);
    const float __e   = __x_hi - __n_f;
    float __fl        = 0.0f;
    float __fh        = ::cuda::__fpmp_two_sum(__e, __x_lo, &__fl);
    if (__fh > 0.5f || (__fh == 0.5f && __fl > 0.0f))
    {
      ++__n;
      float __err = 0.0f;
      __fh        = ::cuda::__fpmp_two_sum(__fh, -1.0f, &__err);
      __fl += __err;
    }
    else if (__fh < -0.5f || (__fh == -0.5f && __fl < 0.0f))
    {
      --__n;
      float __err = 0.0f;
      __fh        = ::cuda::__fpmp_two_sum(__fh, 1.0f, &__err);
      __fl += __err;
    }
    __f_hi = ::cuda::__fpmp_two_sum(__fh, __fl, &__f_lo);
    __odd  = __n & 1;
  }
  else
  {
    const int __odd_hi = (__abs_hi < __even_integral_threshold) ? (::cuda::__fpmp_fp2int_rn(__x_hi) & 1) : 0;
    float __lo_f       = 0.0f;
    const int __odd_lo = ::cuda::__internal_fpmp2_pi_limb(__x_lo, &__lo_f);
    __odd              = __odd_hi ^ __odd_lo;
    __f_hi             = __lo_f;
    __f_lo             = 0.0f;
  }

  // Captured before the fold. CUDA's contract is sinpi(-0) == -0. two_sum of that
  // fraction yields +0, and the sine kernel keeps the sign it is given, so the
  // input sign has to be put back when the fraction is exactly zero. The odd
  // negation stays on the cosine and is not applied to that sine zero.
  const bool __exact_integer = (__f_hi == 0.0f) && (__f_lo == 0.0f);

  /* Fold into [-1/4, 1/4]. q == 1 uses sin=cos, cos=sin; q == 3 uses sin=-cos, cos=sin. */
  int __q = 0;
  if (__f_hi > 0.25f || (__f_hi == 0.25f && __f_lo > 0.0f))
  {
    __q         = 1;
    float __err = 0.0f;
    __f_hi      = ::cuda::__fpmp_two_sum(0.5f, -__f_hi, &__err);
    __f_lo      = __err - __f_lo;
  }
  else if (__f_hi < -0.25f || (__f_hi == -0.25f && __f_lo < 0.0f))
  {
    __q         = 3;
    float __err = 0.0f;
    __f_hi      = ::cuda::__fpmp_two_sum(__f_hi, 0.5f, &__err);
    __f_lo += __err;
  }
  {
    const float __renorm = ::cuda::__fpmp_two_sum(__f_hi, __f_lo, &__f_lo);
    __f_hi               = __renorm;
  }

  /* The first three Cody-Waite pieces of pi/2, doubled to pi (within 1.9e-22; doubling a
   * float is exact). pi * f involves no cancellation, so their truncation error stays
   * relative to the angle and the fourth piece is not needed here. */
  constexpr float __pi1 = 2.0f * 1.5707962512969971e+00f;
  constexpr float __pi2 = 2.0f * 7.5497894158615964e-08f;
  constexpr float __pi3 = 2.0f * 5.3903029534742384e-15f;

  float __p_lo       = 0.0f;
  const float __p_hi = ::cuda::__fpmp_two_mult_fma(__f_hi, __pi1, &__p_lo);
  __afloat __angle(__p_hi, __p_lo);
  float __q_lo       = 0.0f;
  const float __q_hi = ::cuda::__fpmp_two_mult_fma(__f_lo, __pi1, &__q_lo);
  __angle            = __angle + __afloat(__q_hi, __q_lo);
  __angle            = __angle + ::cuda::__fpmp_mul_rn_nocontract(__f_hi, __pi2);
  __angle            = __angle + ::cuda::__fpmp_mul_rn_nocontract(__f_hi, __pi3);

  float __s_hi, __s_lo, __c_hi, __c_lo;
  ::cuda::__internal_fpmp2_sin_kernel(__angle.hi(), __angle.lo(), &__s_hi, &__s_lo);
  ::cuda::__internal_fpmp2_cos_kernel(__angle.hi(), __angle.lo(), &__c_hi, &__c_lo);

  if (__q == 1)
  {
    const float __t_hi = __s_hi;
    const float __t_lo = __s_lo;
    __s_hi             = __c_hi;
    __s_lo             = __c_lo;
    __c_hi             = __t_hi;
    __c_lo             = __t_lo;
  }
  else if (__q == 3)
  {
    const float __t_hi = __s_hi;
    const float __t_lo = __s_lo;
    __s_hi             = -__c_hi;
    __s_lo             = -__c_lo;
    __c_hi             = __t_hi;
    __c_lo             = __t_lo;
  }
  if (__odd != 0)
  {
    if (!__exact_integer)
    {
      __s_hi = -__s_hi;
      __s_lo = -__s_lo;
    }
    __c_hi = -__c_hi;
    __c_lo = -__c_lo;
  }
  if (__exact_integer)
  {
    __s_hi = ::cuda::std::copysign(__s_hi, __x_hi);
    __s_lo = ::cuda::std::copysign(__s_lo, __x_hi);
  }

  *__sin_hi = __s_hi;
  *__sin_lo = __s_lo;
  *__cos_hi = __c_hi;
  *__cos_lo = __c_lo;
}

/*
 * --------------------------------------------------------------------
 * Sine and cosine of pi*x sincospi(x, &s, &c) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_sincospi(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      double __sd;
      double __cd;
      ::sincospi(__xd, &__sd, &__cd);
      ::cuda::__fpmp2_from_double(__sd, __sin_hi, __sin_lo);
      ::cuda::__fpmp2_from_double(__cd, __cos_hi, __cos_lo);
    }),
    ({
      double __xpi = __xd * ::cuda::std::__numbers<double>::__pi();
      ::cuda::__fpmp2_from_double(::cuda::std::sin(__xpi), __sin_hi, __sin_lo);
      ::cuda::__fpmp2_from_double(::cuda::std::cos(__xpi), __cos_hi, __cos_lo);
    }))
}

_CCCL_FPMP_MATH_DISPATCH_1A_2OUT(sincospi)

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_TRIG_H

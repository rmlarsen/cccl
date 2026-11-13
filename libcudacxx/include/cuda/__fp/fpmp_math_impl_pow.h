//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_POW_H
#define _CUDA___FP_FPMP_MATH_IMPL_POW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_pow.h - fpmp2 power / root functions (pow, cbrt, rcbrt, hypot, norm/rnorm)
    ==================================================================================================
*/

#include <cuda/__fp/fpmp_math_impl.h>
// Sibling families whose kernels this family calls (exp/log are used by pow).
#include <cuda/__fp/fpmp_math_impl_exp.h>
#include <cuda/std/__floating_point/constants.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * ====================================================================
 * pow(x, y) - x raised to the power y
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Power function pow(x, y) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 * pow(a, b) = exp(b ln|a|), negated for a negative a and an odd integer b.
 *
 * exp turns an absolute error in its argument into the same relative error in its result, and
 * |b ln|a|| reaches 104 before the result underflows, so b ln|a| rounded to the format's own
 * precision would cost up to log2(104) ~= 7 of the 46 bits. ln|a| is therefore formed to about
 * 2^-58 relative, as three floats, and b ln|a| to about 2^-49 absolute. Against binary128, normal
 * results measure within 3u^2 (u = 2^-24), mean 0.3u^2, at every magnitude of b ln|a|.
 *
 * ln|a|: |a| = 2^k z with z in [0.701, 1.402). The bucket holding z_hi, one of 128, selects 1/c, with
 *   at most 9 significant bits, so that r0 = z_hi/c - 1 = fma(z_hi, 1/c, -1) is exact, and
 *   ln(c) = Lc0 + Lc1 + Lc2. The bucket around 1 has c = 1, which keeps ln|a| relatively accurate
 *   as a approaches 1. r = r0 + z_lo/c is exact as three floats, and
 *       ln|a| = k ln2 + ln(c) + r + delta,   delta = ln(1 + r) - r = -r^2/2 + r^3 P(r),
 *   where |r| < 2^-7.5 leaves delta needing only double-float accuracy. Lc0 and the leading part of
 *   ln2 are multiples of 2^-16, so that k ln2_0 + Lc0 is exact.
 * b ln|a|: three exact products, the multiple k_e of ln2/128 nearest them, and the reduced
 *   argument phi = b ln|a| - k_e ln2/128, |phi| < 2^-8.5, as a float pair.
 * exp: e^phi - 1 by a degree-5 polynomial, times 2^(j/128), j = k_e mod 128, from the table, times
 *   2^n, n = k_e div 128, applied in two steps so that a subnormal result is rounded once.
 *
 * Special values follow C99 Annex F (F.10.4.4) on the value hi + lo: b is an integer when both
 * limbs are, and odd when one limb is an odd integer, which for |b| >= 2^24 is a property of b_lo.
 * Overflow and underflow are decided from b_hi * ln|a| before the exact products are formed, so
 * no intermediate is infinite or NaN.
 * --------------------------------------------------------------------
 */

/*
 * ln|x| for finite x = x_hi + x_lo > 0 as __l0 + __l1 + __l2, |__l1| < 2^-22 |__l0|, within
 * about 2^-58 of ln|x| relatively.
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_pow_log(float __x_hi, float __x_lo, float* __l0, float* __l1, float* __l2) noexcept
{
  constexpr float __ln2_0    = 0x1.62e4p-1f; // ln2 = __ln2_0 + __ln2_1 + __ln2_2
  constexpr float __ln2_1    = 0x1.7f7d1cp-20f;
  constexpr float __ln2_2    = 0x1.ef357ap-45f;
  constexpr float __third_hi = 0x1.555556p-2f; // 1/3 = __third_hi + __third_lo
  constexpr float __third_lo = -0x1.555556p-27f;

  int __k_adj = 0;
  if (__x_hi < 0x1p-126f) // a subnormal x_hi, whose lo is zero, is scaled into the normal range
  {
    __x_hi  = ::cuda::__fpmp_mul_rn(__x_hi, 0x1p24f);
    __x_lo  = ::cuda::__fpmp_mul_rn(__x_lo, 0x1p24f);
    __k_adj = -24;
  }
  // x_hi = z_hi 2^__ke, and r = r0 + rho_hi + rho_lo = z/c - 1, exactly
  __fpmp2_log_table_entry __t;
  float __z_hi, __r0, __rho_hi, __rho_lo;
  const int __ke = ::cuda::__internal_fpmp2_log_bucket(__x_hi, &__z_hi, &__t);
  ::cuda::__internal_fpmp2_log_residual(__z_hi, __x_lo, __ke, __t, &__r0, &__rho_hi, &__rho_lo);

  // k ln2 + ln(c) = h0 + h1 + h2, h0 exact
  const float __kf = static_cast<float>(__ke + __k_adj);
  const float __h0 = ::cuda::__fpmp_fma_rn(__kf, __ln2_0, __t.__logc0);
  float __kl1_lo, __h1_lo;
  const float __kl1_hi = ::cuda::__fpmp_two_mult_fma(__kf, __ln2_1, &__kl1_lo);
  const float __h1     = ::cuda::__fpmp_two_sum(__kl1_hi, __t.__logc1, &__h1_lo);
  const float __h2 =
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_fma_rn(__kf, __ln2_2, __t.__logc2), ::cuda::__fpmp_add_rn(__kl1_lo, __h1_lo));

  // r as a pair for the nonlinear terms. r0 + rho_hi is a valid fast two-sum unless |r0| < |rho_hi|,
  // where those terms are below 2^-40 of ln|x| and need no more accuracy.
  float __r_lo;
  const float __r_hi = ::cuda::__fpmp_fast_two_sum(__r0, __rho_hi, &__r_lo);
  __r_lo             = ::cuda::__fpmp_add_rn(__r_lo, __rho_lo);

  // delta = -r^2/2 + r^3 P(r), P(r) = 1/3 - r/4 + r^2/5 - r^3/6 + r^4/7 - r^5/8, as a pair
  float __sq_lo;
  const float __sq_hi = ::cuda::__fpmp_two_mult_fma(__r_hi, __r_hi, &__sq_lo);
  __sq_lo             = ::cuda::__fpmp_fma_rn(2.0f * __r_hi, __r_lo, __sq_lo);
  float __cu_lo;
  const float __cu_hi = ::cuda::__fpmp_two_mult_fma(__sq_hi, __r_hi, &__cu_lo);
  __cu_lo             = ::cuda::__fpmp_fma_rn(__sq_lo, __r_hi, ::cuda::__fpmp_fma_rn(__sq_hi, __r_lo, __cu_lo));
  float __p_tail      = ::cuda::__fpmp_fma_rn(__r_hi, -0x1p-3f, 0x1.24924ap-3f);
  __p_tail            = ::cuda::__fpmp_fma_rn(__r_hi, __p_tail, -0x1.555556p-3f);
  __p_tail            = ::cuda::__fpmp_fma_rn(__r_hi, __p_tail, 0x1.99999ap-3f);
  __p_tail            = ::cuda::__fpmp_mul_rn_nocontract(__sq_hi, __p_tail);
  float __p_lo;
  const float __p_hi = ::cuda::__fpmp_fast_two_sum(__third_hi, -0.25f * __r_hi, &__p_lo);
  __p_lo =
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__p_lo, __third_lo), ::cuda::__fpmp_fma_rn(-0.25f, __r_lo, __p_tail));
  float __q_lo;
  const float __q_hi = ::cuda::__fpmp_two_mult_fma(__cu_hi, __p_hi, &__q_lo);
  __q_lo             = ::cuda::__fpmp_fma_rn(__cu_lo, __p_hi, ::cuda::__fpmp_fma_rn(__cu_hi, __p_lo, __q_lo));
  float __d_lo;
  const float __d_hi = ::cuda::__fpmp_fast_two_sum(-0.5f * __sq_hi, __q_hi, &__d_lo);
  __d_lo             = ::cuda::__fpmp_add_rn(__d_lo, ::cuda::__fpmp_fma_rn(-0.5f, __sq_lo, __q_lo));

  // ln|x| = h0 + h1 + r0 + rho_hi + d_hi, summed exactly (each fast two-sum adds a term no larger
  // than the running sum, or into a zero sum), plus the remaining terms below 2^-22 of it.
  float __e0, __e1, __e2, __e3;
  float __sum = ::cuda::__fpmp_fast_two_sum(__h0, __h1, &__e0);
  __sum       = ::cuda::__fpmp_fast_two_sum(__sum, __r0, &__e1);
  __sum       = ::cuda::__fpmp_fast_two_sum(__sum, __rho_hi, &__e2);
  __sum       = ::cuda::__fpmp_fast_two_sum(__sum, __d_hi, &__e3);
  float __g1, __g2, __g3;
  float __mid = ::cuda::__fpmp_two_sum(__e0, __e1, &__g1);
  __mid       = ::cuda::__fpmp_two_sum(__mid, __e2, &__g2);
  __mid       = ::cuda::__fpmp_two_sum(__mid, __e3, &__g3);
  *__l0       = __sum;
  *__l1       = __mid;
  *__l2       = ::cuda::__fpmp_add_rn(
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__g1, __g2), ::cuda::__fpmp_add_rn(__g3, __h2)),
    ::cuda::__fpmp_add_rn(__rho_lo, __d_lo));
} // __internal_fpmp2_pow_log

/*
 * e^(b L) = (__y_hi + __y_lo) 2^__n for b = b_hi + b_lo and L = l0 + l1 + l2 from
 * __internal_fpmp2_pow_log, given |b_hi l0| <= 105.
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_pow_exp(
  const float __b_hi,
  const float __b_lo,
  const float __l0,
  const float __l1,
  const float __l2,
  float* __y_hi,
  float* __y_lo,
  int* __n) noexcept
{
  // f = b L = p0 + (p1 + q0 + r0) + (q1 + r1 + s)
  float __p1, __q1, __r1;
  const float __p0 = ::cuda::__fpmp_two_mult_fma(__b_hi, __l0, &__p1);
  const float __q0 = ::cuda::__fpmp_two_mult_fma(__b_hi, __l1, &__q1);
  const float __r0 = ::cuda::__fpmp_two_mult_fma(__b_lo, __l0, &__r1);
  const float __s  = ::cuda::__fpmp_fma_rn(__b_lo, __l1, ::cuda::__fpmp_mul_rn_nocontract(__b_hi, __l2));

  // k = round(p0 / (ln2/128)); phi = f - k ln2/128 as a pair, with ln2/128 = c0 + c1 + c2.
  // k c0 and p0 - k c0 are exact.
  float __kf;
  const int __k   = ::cuda::__internal_fpmp2_round_product(__p0, __fpmp2_inv_ln2_128, &__kf);
  const float __d = ::cuda::__fpmp_fma_rn(-__kf, __fpmp2_ln2_128_0, __p0);
  float __kc1_lo;
  const float __kc1_hi = ::cuda::__fpmp_two_mult_fma(__kf, __fpmp2_ln2_128_1, &__kc1_lo);
  float __v1, __v2, __v3, __v4;
  const float __u  = ::cuda::__fpmp_two_sum(__d, -__kc1_hi, &__v1);
  float __w        = ::cuda::__fpmp_two_sum(__p1, __q0, &__v2);
  __w              = ::cuda::__fpmp_two_sum(__w, __r0, &__v3);
  const float __ph = ::cuda::__fpmp_two_sum(__u, __w, &__v4);
  const float __pl = ::cuda::__fpmp_add_rn(
    ::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__v1, __v2), ::cuda::__fpmp_add_rn(__v3, __v4)),
    ::cuda::__fpmp_sub_rn(::cuda::__fpmp_add_rn(::cuda::__fpmp_add_rn(__q1, __r1), __s),
                          ::cuda::__fpmp_fma_rn(__kf, __fpmp2_ln2_128_2, __kc1_lo)));
  float __phi_lo;
  const float __phi_hi = ::cuda::__fpmp_fast_two_sum(__ph, __pl, &__phi_lo);

  ::cuda::__internal_fpmp2_exp_tail(__phi_hi, __phi_lo, __k, __y_hi, __y_lo, __n);
} // __internal_fpmp2_pow_exp

// Whether the integer __x is odd; every float of magnitude 2^24 or more is even. An integer
// b = b_hi + b_lo is odd when exactly one limb is.
_CCCL_FPMP_CORE_API bool __internal_fpmp2_pow_limb_is_odd(const float __x) noexcept
{
  return ::cuda::__fpmp_internal_fabs(__x) < 0x1p24f && (::cuda::__fpmp_fp2int_rz(__x) & 1) != 0;
}

_CCCL_FPMP_CORE_API void __internal_fpmp2_pow(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  const float __inf      = ::cuda::std::__fp_inf<float>();
  const float __abs_b_hi = ::cuda::__fpmp_internal_fabs(__b_hi);
  // x = |a|, and the sign of the result
  float __x_hi = __a_hi;
  float __x_lo = __a_lo;
  float __sign = 1.0f;

  // Everything but a finite a > 0 with a finite b != 0: NaN, zero and infinite operands, b = 0,
  // and a negative a, which continues with |a| when b is an integer.
  if (!(__a_hi > 0.0f && __a_hi < __inf && __abs_b_hi < __inf && __b_hi != 0.0f))
  {
    // pow(a, +-0) = 1 and pow(+1, b) = 1, even for a NaN operand.
    if (__b_hi == 0.0f || (__a_hi == 1.0f && __a_lo == 0.0f))
    {
      *__res_hi = 1.0f;
      *__res_lo = 0.0f;
      return;
    }
    if (__a_hi != __a_hi || __b_hi != __b_hi)
    {
      *__res_hi = __a_hi + __b_hi;
      *__res_lo = *__res_hi;
      return;
    }
    const bool __a_neg = (::cuda::std::bit_cast<uint32_t>(__a_hi) >> 31) != 0;
    __x_hi             = __a_neg ? -__a_hi : __a_hi;
    __x_lo             = __a_neg ? -__a_lo : __a_lo;
    bool __b_is_int    = false;
    bool __b_is_odd    = false;
    if (__abs_b_hi < __inf)
    {
      __b_is_int = ::cuda::__fpmp_internal_trunc<float>(__b_hi) == __b_hi
                && ::cuda::__fpmp_internal_trunc<float>(__b_lo) == __b_lo;
      __b_is_odd = __b_is_int
                && ::cuda::__internal_fpmp2_pow_limb_is_odd(__b_hi) != ::cuda::__internal_fpmp2_pow_limb_is_odd(__b_lo);
    }
    __sign = (__a_neg && __b_is_odd) ? -1.0f : 1.0f;
    // pow(+-0, b): +-0 for b > 0 and +-inf for b < 0, signed for an odd integer b
    if (__x_hi == 0.0f)
    {
      *__res_hi = __b_hi < 0.0f ? __sign * __inf : __sign * 0.0f;
      *__res_lo = 0.0f;
      return;
    }
    // pow(+-inf, b): +-inf for b > 0 and +-0 for b < 0, signed for an odd integer b
    if (__x_hi == __inf)
    {
      *__res_hi = __b_hi < 0.0f ? __sign * 0.0f : __sign * __inf;
      *__res_lo = 0.0f;
      return;
    }
    // pow(a, +-inf) for a finite a != 0: 1 for |a| = 1, else +inf or +0 by |a| > 1 and the sign of b
    if (__abs_b_hi == __inf)
    {
      const bool __abs_gt_one = __x_hi > 1.0f || (__x_hi == 1.0f && __x_lo > 0.0f);
      const bool __abs_is_one = __x_hi == 1.0f && __x_lo == 0.0f;
      if (__abs_is_one)
      {
        *__res_hi = 1.0f;
      }
      else
      {
        *__res_hi = (__b_hi > 0.0f) == __abs_gt_one ? __inf : 0.0f;
      }
      *__res_lo = 0.0f;
      return;
    }
    // a < 0 finite, b finite: NaN unless b is an integer
    if (!__b_is_int)
    {
      *__res_hi = ::cuda::std::__fp_nan<float>();
      *__res_lo = *__res_hi;
      return;
    }
  }

  float __l0, __l1, __l2;
  ::cuda::__internal_fpmp2_pow_log(__x_hi, __x_lo, &__l0, &__l1, &__l2);
  // b ln|a| above 89.5 overflows and below -105 rounds to zero; deciding it here keeps the
  // products below from overflowing.
  const float __f = ::cuda::__fpmp_mul_rn(__b_hi, __l0);
  if (!(__f <= 89.5f))
  {
    *__res_hi = __sign * __inf;
    *__res_lo = 0.0f;
    return;
  }
  if (__f < -105.0f)
  {
    *__res_hi = __sign * 0.0f;
    *__res_lo = 0.0f;
    return;
  }
  float __y_hi, __y_lo, __r_hi, __r_lo;
  int __n;
  ::cuda::__internal_fpmp2_pow_exp(__b_hi, __b_lo, __l0, __l1, __l2, &__y_hi, &__y_lo, &__n);
  ::cuda::__internal_fpmp2_exp_scale(__y_hi, __y_lo, __n, &__r_hi, &__r_lo);
  *__res_hi = __sign * __r_hi;
  *__res_lo = __sign * __r_lo;
} // __internal_fpmp2_pow

/*
 * --------------------------------------------------------------------
 * Power function pow(x, y) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void __internal_fpmp2_pow(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  _CCCL_FPMP_CALL_FP64MP2_MATH_2A(pow, _CCCL_FPMP_POWQ, __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}

_CCCL_FPMP_MATH_DISPATCH_2A(pow)

/*
 * ====================================================================
 * Single-precision seeds for cbrt(x) and rcbrt(x)
 * ====================================================================
 *
 * r^(1/3) and r^(-1/3) for the reduced argument r in [1/4, 2), to a
 * relative error of about 2^-23; the fp32mp2 Halley step that follows
 * needs no more.
 *
 * The device evaluates ex2(+-lg2(r) / 3) with the SFU lg2/ex2 pair. A
 * host whose fmaf is a hardware instruction refines a minimax polynomial
 * instead, which is faster there than exp2f and log2f. Every multiply-add
 * is an explicit fma, so that result depends neither on the compiler's
 * contraction nor on the C library. Over every float in [1/4, 2) these
 * seeds err by at most 2^-23.98 (cbrt) and 2^-23.29 (rcbrt).
 *
 * Any other host keeps exp2f and log2f: with fmaf a library call, the
 * polynomial seeds make cbrt 9% and rcbrt 14% slower on x86-64.
 *
 * Polynomials: Sollya 8.0,
 *   fpminimax(x^(1/3),  4, [|single...|], [1/4; 2], relative, floating)
 *   fpminimax(x^(-1/3), 3, [|single...|], [1/4; 2], relative, floating)
 * with relative errors of at most 1.87e-3 and 1.69e-2.
 */

// FP_FAST_FMAF is the <cmath> statement that fmaf is hardware; clang does
// not make it on x86, where __FMA__ says the same.
#  if defined(FP_FAST_FMAF) || defined(__FMA__)
#    define _CCCL_FPMP_HOST_SEED_POLY 1
#  else
#    define _CCCL_FPMP_HOST_SEED_POLY 0
#  endif

#  if _CCCL_FPMP_HOST_SEED_POLY

/* Host seed for cbrt: one Halley step s - s (s^3 - r) / (2 s^3 + r) on
 * the quartic, with s^3 - r = fma(s2_hi, s, -r) + s2_lo s for the exact
 * square s2_hi + s2_lo = s^2. A relative error e becomes 2/3 e^3:
 * 4.4e-9, below the rounding of the step. */
_CCCL_FPMP_CORE_API float __internal_fpmp2_cbrt_seed_host(const float __r) noexcept
{
  float __s = ::cuda::__fpmp_fma_rn(-0x1.1d6df4p-4f, __r, 0x1.8bab52p-2f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, -0x1.b0554p-1f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, 0x1.21c40ap+0f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, 0x1.94a7acp-2f);

  float __s2_lo;
  const float __s2_hi = ::cuda::__fpmp_two_mult_fma(__s, __s, &__s2_lo);
  const float __numer = ::cuda::__fpmp_fma_rn(__s2_lo, __s, ::cuda::__fpmp_fma_rn(__s2_hi, __s, -__r));
  const float __denom = ::cuda::__fpmp_fma_rn(__s2_hi + __s2_hi, __s, __r);
  return ::cuda::__fpmp_fma_rn(-__s, __numer / __denom, __s);
}

/* Host seed for rcbrt: two division-free steps s (1 + u/3 + 2u^2/9),
 * u = 1 - r s^3, on the cubic. A relative error e becomes 14/3 e^3:
 * 2.2e-5 after the first step and 5e-14 after the second. */
_CCCL_FPMP_CORE_API float __internal_fpmp2_rcbrt_seed_host(const float __r) noexcept
{
  float __s = ::cuda::__fpmp_fma_rn(-0x1.092b32p-2f, __r, 0x1.2a688p+0f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, -0x1.e31ca6p+0f);
  __s       = ::cuda::__fpmp_fma_rn(__s, __r, 0x1.f6b196p+0f);

  for (int __i = 0; __i < 2; ++__i)
  {
    const float __u  = ::cuda::__fpmp_fma_rn(-__r, (__s * __s) * __s, 1.0f);
    const float __hf = ::cuda::__fpmp_fma_rn(0x1.c71c72p-3f, __u, 0x1.555556p-2f);
    __s              = ::cuda::__fpmp_fma_rn(__u * __s, __hf, __s);
  }
  return __s;
}

#  endif // _CCCL_FPMP_HOST_SEED_POLY

_CCCL_FPMP_CORE_API float __internal_fpmp2_cbrt_seed(const float __r) noexcept
{
#  if _CCCL_FPMP_HOST_SEED_POLY
  NV_IF_TARGET(NV_IS_HOST, (return ::cuda::__internal_fpmp2_cbrt_seed_host(__r);))
#  endif // _CCCL_FPMP_HOST_SEED_POLY
  return ::cuda::__fpmp_fast_exp2(0x1.555556p-2f * ::cuda::__fpmp_fast_log2(__r));
}

_CCCL_FPMP_CORE_API float __internal_fpmp2_rcbrt_seed(const float __r) noexcept
{
#  if _CCCL_FPMP_HOST_SEED_POLY
  NV_IF_TARGET(NV_IS_HOST, (return ::cuda::__internal_fpmp2_rcbrt_seed_host(__r);))
#  endif // _CCCL_FPMP_HOST_SEED_POLY
  return ::cuda::__fpmp_fast_exp2(-0x1.555556p-2f * ::cuda::__fpmp_fast_log2(__r));
}

/*
 * ====================================================================
 * cbrt(x) - cube root
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Cube root cbrt(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   1. Special cases: pass +-0, +-Inf, NaN through unchanged
 *      (cbrt(x) == x for these inputs).
 *   2. Operate on |x|; cbrt is odd, sign is restored at the end.
 *   3. Pre-scale denormal inputs by 2^24 (= 2^(3*8)) so the exponent
 *      extraction sees a normal float; the +8 bias in the result
 *      exponent is undone in the final scaling step.
 *   4. Argument reduction:  ax = r * 2^(3*nexpo)  with
 *      nexpo = round((expo - 126) / 3).  The reduced r lies in
 *      [1/4, 2): ax = m * 2^(expo - 126) with m in [1/2, 1), and
 *      (expo - 126) - 3 nexpo is -1, 0 or 1.
 *   5. Initial single-precision approximation of r_hi^(1/3), ~23 bits
 *      (__internal_fpmp2_cbrt_seed), which lets one Halley step recover
 *      full fp32mp2 precision.
 *   6. One Halley iteration in fp32mp2 arithmetic
 *      (cubic convergence -> ~70 theoretical bits, capped by the
 *       ~46-bit fp32mp2 precision):
 *         t_new = t + t * (r - t^3) / (2 t^3 + r)
 *      The numerator (r - t^3) cancels catastrophically (t^3 ~= r),
 *      so it is evaluated with the accurate fp32mp2 fma:
 *         numer = fma<accurate>(-t^2, t, r)
 *      The denominator is well-conditioned (~3 r); a single-precision
 *      reciprocal of denom.hi() is enough -- the resulting correction
 *      contributes only at the 2^-46 level after multiplication by t.
 *   7. Multiply the result back by 2^(nexpo - denorm_div3) using
 *      a power-of-two scale factor (exact, no rounding).
 *   8. Restore the sign.
 *
 * No fp64 operations.  Negative inputs are supported (cbrt(-x) = -cbrt(x)).
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cbrt(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  // 1/3 in single precision (round-to-nearest). (expo - 126) / 3 is at
  // least 1/6 away from a rounding tie, so the product rounds to the same
  // integer as the exact quotient.
  constexpr float __third_f = 0x1.555556p-2f;

  const uint32_t __xbits   = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __absbits = __xbits & 0x7FFFFFFFu;
  const uint32_t __signbit = __xbits & 0x80000000u;

  if (__absbits == 0u || __absbits >= 0x7F800000u)
  {
    *__res_hi = __x_hi;
    *__res_lo = (__absbits >= 0x7F800000u) ? 0.0f : __x_lo;
    return;
  }

  /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
  float __ax_hi = ::cuda::std::bit_cast<float>(__absbits);
  float __ax_lo = (__signbit != 0u) ? -__x_lo : __x_lo;

  /* Denormal pre-scaling: multiply by 2^24 (chosen so the offset is
   * divisible by 3 -> denorm_div3 = 8 unscales the result later). */
  int __denorm_div3         = 0;
  uint32_t __scaled_absbits = __absbits;
  if ((__absbits >> 23) == 0u)
  {
    constexpr float __scale_up = 0x1.0p24f;
    __ax_hi *= __scale_up;
    __ax_lo *= __scale_up;
    __denorm_div3    = 8;
    __scaled_absbits = ::cuda::std::bit_cast<uint32_t>(__ax_hi);
  }

  /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
  const int __expo  = static_cast<int>(__scaled_absbits >> 23);
  const int __nexpo = ::cuda::__fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

  /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
   * (The mantissa is untouched; only the biased exponent shifts.)
   * Use multiplication by 2^23 instead of left-shift to avoid UB
   * when (3 * nexpo) is negative. */
  constexpr int __exp_shift = 1 << 23;
  const int __delta_exp     = 3 * __nexpo;
  const int __new_bits      = static_cast<int>(__scaled_absbits) - __delta_exp * __exp_shift;
  const float __r_hi        = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(__new_bits));

  /* r_lo: scale by the same power of two via float multiply.  Split
   * 2^(-3*nexpo) into two normal-range factors: for x near max float
   * |3*nexpo| can reach ~129, which would give an invalid biased
   * exponent of -2 if applied as a single bit-cast.  Splitting keeps
   * each factor's biased exponent in the normal range (about
   * [62, 190]); the product stays exact for all valid inputs. */
  const int __half_pow  = -__delta_exp / 2;
  const int __rest_pow  = -__delta_exp - __half_pow;
  const float __scale_a = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __half_pow) * __exp_shift));
  const float __scale_b = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __rest_pow) * __exp_shift));
  const float __r_lo    = (__ax_lo * __scale_a) * __scale_b;

  /* Initial cbrt approximation (~23 bits). */
  const float __s = ::cuda::__internal_fpmp2_cbrt_seed(__r_hi);

  /* Halley refinement in fp32mp2:  t_new = t + t * (r - t^3) / (2 t^3 + r).
   *
   * The catastrophic cancellation in (r - t^3) is handled by an
   * accurate fma:  fma<accurate>(-t^2, t, r) computes r - t^2 * t
   * with a single rounding error followed by an exact correction,
   * preserving the small difference that drives the iteration.
   */
  const __ffloat __r(__r_hi, __r_lo);
  const __ffloat __t(__s);

  // t^2 = s^2 exactly
  float __t2_lo;
  const float __t2_hi = ::cuda::__fpmp_two_mult_fma(__s, __s, &__t2_lo);
  const __ffloat __t2(__t2_hi, __t2_lo);
  // numer = r - t^3 (computed as fma(-t^2, t, r) with accurate ff fma)
  const __ffloat __numer = fma<fpmp2_accuracy::high>(-__t2, __t, __r);
  // denom = 2 t^3 + r ~= 3 r, well-conditioned so fast add suffices
  const __ffloat __t3    = __t2 * __s;
  const __ffloat __denom = __t3 * 2.0f + __r;

  /* Single-precision reciprocal of denom.hi() is enough: the
   * correction u_corr ~ 2^-23 contributes t * u_corr ~ 2^-46 to
   * t_new -- exactly fp32mp2 precision. */
  const float __inv_denom = ::cuda::__fpmp_rcp_rn(__denom.hi());
  const __ffloat __u_corr = __numer * __inv_denom;
  const __ffloat __t_new  = __s + __s * __u_corr;

  /* Scale back by 2^(nexpo - denorm_div3) via an exact power-of-two
   * float multiply.  back_shift stays in a range that keeps the
   * scale factor a normal float for all valid float inputs
   * (biased exponent is always in [77, 170]). */
  const int __back_shift   = __nexpo - __denorm_div3;
  const float __scale_back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __exp_shift));
  float __t_hi_back        = __t_new.hi() * __scale_back;
  float __t_lo_back        = __t_new.lo() * __scale_back;

  /* Restore sign (cbrt is an odd function). */
  if (__signbit != 0u)
  {
    __t_hi_back = -__t_hi_back;
    __t_lo_back = -__t_lo_back;
  }

  *__res_hi = __t_hi_back;
  *__res_lo = __t_lo_back;
} // __internal_fpmp2_cbrt

/*
 * --------------------------------------------------------------------
 * Cube root cbrt(x) (fp64mp2) - binary128 wrapper
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_cbrt(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
  __fpmp_fp128 __res = _CCCL_FPMP_CBRTQ(::cuda::__fpmp2_to_quad(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_quad(__res, __res_hi, __res_lo);
#  else
  double __res = ::cuda::std::cbrt(::cuda::__fpmp2_to_double(__x_hi, __x_lo));
  ::cuda::__fpmp2_from_double(__res, __res_hi, __res_lo);
#  endif
}

_CCCL_FPMP_MATH_DISPATCH_1A(cbrt)

/*
 * ====================================================================
 * rcbrt(x) - reciprocal cube root
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Reciprocal cube root rcbrt(x) = 1/cbrt(x) (fp32mp2) - dedicated implementation
 * --------------------------------------------------------------------
 *   1. Special cases:
 *        cbrt(+-0)   = +-Inf  (result inherits the sign of x)
 *        cbrt(+-Inf) = +-0
 *        cbrt(NaN)  = NaN
 *   2. Operate on |x|; rcbrt is odd, sign is restored at the end.
 *   3. Pre-scale denormal inputs by 2^24 (= 2^(3*8)) so the exponent
 *      extraction sees a normal float; the +8 offset ends up as a
 *      +8 shift on the back-scale (denorm_div3 = 8).
 *   4. Argument reduction:  ax = r * 2^(3*nexpo) with
 *      nexpo = round((expo - 126) / 3).  The reduced r lies in
 *      [1/4, 2), as for cbrt.
 *   5. Initial single-precision approximation of r_hi^(-1/3), ~23 bits
 *      (__internal_fpmp2_rcbrt_seed), which lets one Halley step recover
 *      full fp32mp2 precision.
 *   6. One Halley iteration in fp32mp2 (cubic convergence ->
 *      ~70 theoretical bits, capped by ~46-bit fp32mp2 precision):
 *
 *          u     = 1 - r * t^3              (catastrophic cancellation)
 *          t_new = t * (1 + u/3 + (2/9) u^2)
 *
 *      The residual u is the dominant source of error and is computed
 *      with the accurate fp32mp2 fma:
 *         u = fma<accurate>(-r, t^3, 1)
 *      preserving the small difference that drives the iteration.
 *      The Halley quadratic (1/3 + (2/9) u) is well conditioned (no
 *      cancellation), so a single fast fma is sufficient there.
 *   7. Multiply the result back by 2^(-nexpo + denorm_div3) using a
 *      power-of-two scale factor (exact, no rounding).
 *   8. Restore the sign.
 *
 * No fp64 operations.  No final reciprocal: the algorithm targets 1/cbrt
 * directly, which is why it is faster than cbrt(x) followed by a divide.
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_rcbrt(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  using __ffloat = fp32mp2_low;

  // 1/3 and 2/9 in single precision (round-to-nearest). nexpo rounds as
  // in cbrt, and the Halley step multiplies their rounding by u, which is
  // of the order of the seed's error.
  constexpr float __third_f      = 0x1.555556p-2f; // ~= 1/3
  constexpr float __two_ninths_f = 0x1.c71c72p-3f; // ~= 2/9

  const uint32_t __xbits   = ::cuda::std::bit_cast<uint32_t>(__x_hi);
  const uint32_t __absbits = __xbits & 0x7FFFFFFFu;
  const uint32_t __signbit = __xbits & 0x80000000u;

  /* Special inputs:
   *   +-0   -> +-Inf
   *   +-Inf -> +-0
   *   NaN  -> NaN  (propagated via x_hi)
   */
  if (__absbits == 0u || __absbits >= 0x7F800000u)
  {
    if (__absbits == 0u)
    {
      *__res_hi = ::cuda::std::bit_cast<float>(__signbit | 0x7F800000u);
    }
    else if (__absbits == 0x7F800000u)
    {
      *__res_hi = ::cuda::std::bit_cast<float>(__signbit);
    }
    else
    {
      *__res_hi = __x_hi; // NaN
    }
    *__res_lo = 0.0f;
    return;
  }

  /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
  float __ax_hi = ::cuda::std::bit_cast<float>(__absbits);
  float __ax_lo = (__signbit != 0u) ? -__x_lo : __x_lo;

  /* Denormal pre-scaling: multiply by 2^24. */
  int __denorm_div3         = 0;
  uint32_t __scaled_absbits = __absbits;
  if ((__absbits >> 23) == 0u)
  {
    constexpr float __scale_up = 0x1.0p24f;
    __ax_hi *= __scale_up;
    __ax_lo *= __scale_up;
    __denorm_div3    = 8;
    __scaled_absbits = ::cuda::std::bit_cast<uint32_t>(__ax_hi);
  }

  /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
  const int __expo  = static_cast<int>(__scaled_absbits >> 23);
  const int __nexpo = ::cuda::__fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

  /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
   * Use multiplication by 2^23 instead of left-shift to avoid UB
   * when (3 * nexpo) is negative. */
  constexpr int __exp_shift = 1 << 23;
  const int __delta_exp     = 3 * __nexpo;
  const int __new_bits      = static_cast<int>(__scaled_absbits) - __delta_exp * __exp_shift;
  const float __r_hi        = ::cuda::std::bit_cast<float>(static_cast<uint32_t>(__new_bits));

  /* r_lo: scale by 2^(-3*nexpo) via float multiply.  Split into two
   * normal-range factors to keep each biased exponent in roughly
   * [62, 190] for all valid float inputs. */
  const int __half_pow  = -__delta_exp / 2;
  const int __rest_pow  = -__delta_exp - __half_pow;
  const float __scale_a = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __half_pow) * __exp_shift));
  const float __scale_b = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __rest_pow) * __exp_shift));
  const float __r_lo    = (__ax_lo * __scale_a) * __scale_b;

  /* Initial 1/cbrt approximation (~23 bits). */
  const float __s = ::cuda::__internal_fpmp2_rcbrt_seed(__r_hi);

  /* Halley refinement in fp32mp2:  t_new = t * (1 + u/3 + (2/9) u^2)
   * with u = 1 - r * t^3.
   *
   * The cancellation in (1 - r * t^3) is the only sensitive step;
   * everything else (t^2, t^3, the Halley quadratic, the final
   * combination) is well conditioned in fast fp32mp2 arithmetic.
   */
  const __ffloat __r(__r_hi, __r_lo);

  // t^2 = s^2 exactly
  float __t2_lo;
  const float __t2_hi = ::cuda::__fpmp_two_mult_fma(__s, __s, &__t2_lo);
  const __ffloat __t2(__t2_hi, __t2_lo);
  const __ffloat __t3 = __t2 * __s; // t^3

  // u = 1 - r*t^3 (accurate fma to preserve catastrophic cancellation)
  const __ffloat __u = fma<fpmp2_accuracy::high>(-__r, __t3, 1.0f);

  // Halley quadratic factor:  hf = 1/3 + (2/9) u   (no cancellation)
  const __ffloat __hf = fma<fpmp2_accuracy::def>(__two_ninths_f, __u, __third_f);

  // delta = u * t * hf,  then  t_new = t + delta
  const __ffloat __ut    = __u * __s;
  const __ffloat __t_new = __s + __hf * __ut;

  /* Scale back by 2^(-nexpo + denorm_div3) via an exact power-of-two
   * float multiply.  back_shift stays in [-43, +49] for all valid
   * float inputs, so the biased exponent is always in [84, 176]. */
  const int __back_shift   = -__nexpo + __denorm_div3;
  const float __scale_back = ::cuda::std::bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __exp_shift));
  float __t_hi_back        = __t_new.hi() * __scale_back;
  float __t_lo_back        = __t_new.lo() * __scale_back;

  /* Restore sign (rcbrt is an odd function). */
  if (__signbit != 0u)
  {
    __t_hi_back = -__t_hi_back;
    __t_lo_back = -__t_lo_back;
  }

  *__res_hi = __t_hi_back;
  *__res_lo = __t_lo_back;
} // __internal_fpmp2_rcbrt

/*
 * --------------------------------------------------------------------
 * Reciprocal cube root rcbrt(x) = 1/cbrt(x) (fp64mp2) - double fallback
 * --------------------------------------------------------------------
 */
_CCCL_FPMP_CORE_API void
__internal_fpmp2_rcbrt(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  double __xd = ::cuda::__fpmp2_to_double(__x_hi, __x_lo);
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::cuda::__fpmp2_from_double(::rcbrt(__xd), __res_hi, __res_lo);),
                    (::cuda::__fpmp2_from_double(1.0 / ::cuda::std::cbrt(__xd), __res_hi, __res_lo);))
}

_CCCL_FPMP_MATH_DISPATCH_1A(rcbrt)

/*
 * ====================================================================
 * hypot, rhypot, norm3d, norm4d, rnorm3d, rnorm4d - Euclidean norms and their reciprocals
 * ====================================================================
 */

/*
 * --------------------------------------------------------------------
 * Euclidean norm sqrt(x_1^2 + ... + x_n^2) and its reciprocal (fp32mp2 and fp64mp2)
 * --------------------------------------------------------------------
 * The operands are scaled by a power of two chosen from the largest high limb, so that
 * no square overflows and the square of the largest one, with its rounding error, is a
 * pair of normal numbers; the scaling is exact, except for operands too small next to
 * the largest one to matter. The squares are formed and summed as pairs and the sum goes
 * through the square root kernel, or the reciprocal square root kernel.
 *
 * Against MPFR the results are within 4.6 u^2 of the norm and 5.5 u^2 of its reciprocal
 * (u = 2^-24, 2^-53; 0.3 and 0.4 u^2 on average), over the whole range, subnormal
 * operands and results included.
 *
 * Special cases, answered on entry (C Annex F for hypot, extended to n operands and to
 * the reciprocal): an infinite operand gives +inf (+0 for the reciprocal), also next to
 * a NaN; otherwise a NaN operand gives NaN; all operands zero give +0 (+inf). A norm
 * that overflows gives +inf, a reciprocal that overflows, +inf.
 */
template <typename _FpType, int _Np, bool _Reciprocal>
_CCCL_FPMP_CORE_API void __internal_fpmp2_norm(
  const _FpType (&__x_hi)[_Np], const _FpType (&__x_lo)[_Np], _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  constexpr _FpType __max = (::cuda::std::numeric_limits<_FpType>::max)();
  constexpr _FpType __inf = ::cuda::std::numeric_limits<_FpType>::infinity();
  constexpr _FpType __nan = ::cuda::std::numeric_limits<_FpType>::quiet_NaN();

  // The largest magnitude; a NaN never enters it.
  _FpType __m = _FpType(0);
  for (int __k = 0; __k < _Np; ++__k)
  {
    const _FpType __a = ::cuda::__fpmp_internal_fabs(__x_hi[__k]);
    __m               = (__a > __m) ? __a : __m;
  }
  if (!(__m > _FpType(0) && __m <= __max))
  {
    // __m is zero or infinite.
    bool __has_nan = false;
    for (int __k = 0; __k < _Np; ++__k)
    {
      __has_nan = __has_nan || ::cuda::__fpmp_internal_isnan(__x_hi[__k]);
    }
    if (__m > __max)
    {
      *__res_hi = _Reciprocal ? _FpType(0) : __inf;
      *__res_lo = _FpType(0);
    }
    else if (__has_nan)
    {
      *__res_hi = *__res_lo = __nan;
    }
    else
    {
      *__res_hi = _Reciprocal ? __inf : _FpType(0);
      *__res_lo = _FpType(0);
    }
    return;
  }

  // The scaled largest operand m lies in [2^-40, 2^60) (fp32mp2; [2^-49, 2^60) from a
  // subnormal) or [2^-474, 2^500) (fp64mp2), where m^2 and the error of that product are
  // normal numbers and four squares cannot overflow.
  constexpr _FpType __big       = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p60f) : _FpType(0x1.0p500);
  constexpr _FpType __small     = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-40f) : _FpType(0x1.0p-450);
  constexpr _FpType __big_scl   = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-80f) : _FpType(0x1.0p-600);
  constexpr _FpType __big_inv   = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p80f) : _FpType(0x1.0p600);
  constexpr _FpType __small_scl = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p100f) : _FpType(0x1.0p600);
  constexpr _FpType __small_inv = __fpmp2_is_fp32_v<_FpType> ? _FpType(0x1.0p-100f) : _FpType(0x1.0p-600);
  const _FpType __scl           = (__m >= __big) ? __big_scl : (__m < __small) ? __small_scl : _FpType(1);
  const _FpType __inv           = (__m >= __big) ? __big_inv : (__m < __small) ? __small_inv : _FpType(1);

  // s + c = sum of the squares, c holding the rounding errors and the low-limb terms.
  _FpType __s = _FpType(0);
  _FpType __c = _FpType(0);
  for (int __k = 0; __k < _Np; ++__k)
  {
    const _FpType __h = ::cuda::__fpmp_mul_rn(__x_hi[__k], __scl);
    const _FpType __l = ::cuda::__fpmp_mul_rn(__x_lo[__k], __scl);
    const _FpType __p = ::cuda::__fpmp_mul_rn(__h, __h);
    _FpType __e       = ::cuda::__fpmp_fma_rn(__h, __h, -__p);
    __e               = ::cuda::__fpmp_fma_rn(::cuda::__fpmp_add_rn(__h, __h), __l, __e);
    __e               = ::cuda::__fpmp_fma_rn(__l, __l, __e);
    if (__k == 0)
    {
      __s = __p;
      __c = __e;
    }
    else
    {
      _FpType __t;
      __s = ::cuda::__fpmp_two_sum(__s, __p, &__t);
      __c = ::cuda::__fpmp_add_rn(__c, ::cuda::__fpmp_add_rn(__t, __e));
    }
  }
  _FpType __a_lo;
  const _FpType __a_hi = ::cuda::__fpmp_fast_two_sum(__s, __c, &__a_lo);

  _FpType __r_hi, __r_lo;
  if constexpr (_Reciprocal)
  {
    ::cuda::__fpmp2_rsqrt(__a_hi, __a_lo, &__r_hi, &__r_lo);
    __r_hi = ::cuda::__fpmp_mul_rn(__r_hi, __scl);
    __r_lo = ::cuda::__fpmp_mul_rn(__r_lo, __scl);
  }
  else
  {
    ::cuda::__fpmp2_sqrt(__a_hi, __a_lo, &__r_hi, &__r_lo);
    __r_hi = ::cuda::__fpmp_mul_rn(__r_hi, __inv);
    __r_lo = ::cuda::__fpmp_mul_rn(__r_lo, __inv);
  }
  // Overflow, or a NaN operand next to finite ones.
  if (!(__r_hi <= __max))
  {
    __r_lo = ::cuda::__fpmp_internal_isnan(__r_hi) ? __nan : _FpType(0);
  }
  *__res_hi = __r_hi;
  *__res_lo = __r_lo;
}

#  define _CCCL_FPMP_NORM_2A(name, recip)                                               \
    template <typename _FpType>                                                         \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                   \
      const _FpType __x_hi,                                                             \
      const _FpType __x_lo,                                                             \
      const _FpType __y_hi,                                                             \
      const _FpType __y_lo,                                                             \
      _FpType* __res_hi,                                                                \
      _FpType* __res_lo) noexcept                                                       \
    {                                                                                   \
      const _FpType __hi[2] = {__x_hi, __y_hi};                                         \
      const _FpType __lo[2] = {__x_lo, __y_lo};                                         \
      ::cuda::__internal_fpmp2_norm<_FpType, 2, recip>(__hi, __lo, __res_hi, __res_lo); \
    }                                                                                   \
    _CCCL_FPMP_MATH_DISPATCH_2A(name)

#  define _CCCL_FPMP_NORM_3A(name, recip)                                               \
    template <typename _FpType>                                                         \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                   \
      const _FpType __a_hi,                                                             \
      const _FpType __a_lo,                                                             \
      const _FpType __b_hi,                                                             \
      const _FpType __b_lo,                                                             \
      const _FpType __c_hi,                                                             \
      const _FpType __c_lo,                                                             \
      _FpType* __res_hi,                                                                \
      _FpType* __res_lo) noexcept                                                       \
    {                                                                                   \
      const _FpType __hi[3] = {__a_hi, __b_hi, __c_hi};                                 \
      const _FpType __lo[3] = {__a_lo, __b_lo, __c_lo};                                 \
      ::cuda::__internal_fpmp2_norm<_FpType, 3, recip>(__hi, __lo, __res_hi, __res_lo); \
    }                                                                                   \
    _CCCL_FPMP_MATH_DISPATCH_3A(name)

#  define _CCCL_FPMP_NORM_4A(name, recip)                                               \
    template <typename _FpType>                                                         \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                   \
      const _FpType __a_hi,                                                             \
      const _FpType __a_lo,                                                             \
      const _FpType __b_hi,                                                             \
      const _FpType __b_lo,                                                             \
      const _FpType __c_hi,                                                             \
      const _FpType __c_lo,                                                             \
      const _FpType __d_hi,                                                             \
      const _FpType __d_lo,                                                             \
      _FpType* __res_hi,                                                                \
      _FpType* __res_lo) noexcept                                                       \
    {                                                                                   \
      const _FpType __hi[4] = {__a_hi, __b_hi, __c_hi, __d_hi};                         \
      const _FpType __lo[4] = {__a_lo, __b_lo, __c_lo, __d_lo};                         \
      ::cuda::__internal_fpmp2_norm<_FpType, 4, recip>(__hi, __lo, __res_hi, __res_lo); \
    }                                                                                   \
    _CCCL_FPMP_MATH_DISPATCH_4A(name)

_CCCL_FPMP_NORM_2A(hypot, false)
_CCCL_FPMP_NORM_2A(rhypot, true)
_CCCL_FPMP_NORM_3A(norm3d, false)
_CCCL_FPMP_NORM_3A(rnorm3d, true)
_CCCL_FPMP_NORM_4A(norm4d, false)
_CCCL_FPMP_NORM_4A(rnorm4d, true)

#  undef _CCCL_FPMP_NORM_2A
#  undef _CCCL_FPMP_NORM_3A
#  undef _CCCL_FPMP_NORM_4A

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_POW_H

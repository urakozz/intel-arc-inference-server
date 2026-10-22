// nb_common.h - spec 25c: the Bolt family's shared scalar pieces (src/kernels/bolt/nb_*.cl include it; ocloc gets -I the
// family's directory). tests/kernels/bolt_ref.h has every function below line for line - EDIT THE TWO TOGETHER.
//
//   bf16f / rne_bf16 / rf   bf16 <-> fp32 (round-to-nearest-even; NaN is not expected and not handled)
//   nb_exp                  Sleef's expf u10 (torch's Vectorized<float>::exp; src/common/exp_torch.h, the loader's A)
//   nb_log1p                fdlibm's s_log1pf in plain fp32 operations
//   nb_sigmoid              torch's vectorised fp32 sigmoid: 1 / (exp(0 - x) + 1)
//   nb_silu_conv / nb_silu_gate / nb_softplus   the conv's v / (1 + exp(-v)), the gated norm's z (1 / (1 + exp(-z))),
//                           mamba_ssm's softplus (threshold 20, no clamp)
// Every operation is an IEEE fp32 operation (the files that include this set FP_CONTRACT OFF; fma is spelled out), so the
// device and the host twin give the same bits.
#ifndef NB_COMMON_H
#define NB_COMMON_H

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }

// Sleef_expf8_u10 (sleefsimdsp.c xexpf, AVX2 + FMA): mg_attn_eager.cl's exp_torch, the same lines.
inline float nb_exp(float d) {
  const int q = convert_int_rte(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
  const float qf = (float)q;
  float s = fma(qf, -0.693145751953125f, d);
  s = fma(qf, -1.428606765330187045e-06f, s);
  float u = 0.000198527617612853646278381f;
  u = fma(u, s, 0.00139304355252534151077271f);
  u = fma(u, s, 0.00833336077630519866943359f);
  u = fma(u, s, 0.0416664853692054748535156f);
  u = fma(u, s, 0.166666671633720397949219f);
  u = fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + fma(ss, u, s);
  if (d < -104.0f) return 0.0f;
  if (100.0f < d) return INFINITY;
  u = u * as_float((uint)((q >> 1) + 127) << 23);
  return u * as_float((uint)((q - (q >> 1)) + 127) << 23);
}

// fdlibm's s_log1pf.c (FreeBSD), x > -1 finite (the softplus feeds it exp(v) >= 0).
inline float nb_log1p(float x) {
  const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f;
  const float Lp1 = 6.6666668653e-01f, Lp2 = 4.0000000596e-01f, Lp3 = 2.8571429849e-01f, Lp4 = 2.2222198546e-01f,
              Lp5 = 1.8183572590e-01f, Lp6 = 1.5313838422e-01f, Lp7 = 1.4798198640e-01f;
  const int hx = as_int(x);
  const int ax = hx & 0x7fffffff;
  int k = 1, hu = 0;
  float f = 0.0f, c = 0.0f;
  if (hx < 0x3ed413d0) {
    if (ax >= 0x3f800000) return x == -1.0f ? -INFINITY : NAN;
    if (ax < 0x38000000) {
      if (ax < 0x33800000) return x;
      return x - (x * x) * 0.5f;
    }
    if (hx > 0 || hx <= (int)0xbe95f619) {
      k = 0;
      f = x;
      hu = 1;
    }
  }
  if (hx >= 0x7f800000) return x + x;
  if (k != 0) {
    float u;
    if (hx < 0x5a000000) {
      u = 1.0f + x;
      hu = as_int(u);
      k = (hu >> 23) - 127;
      c = k > 0 ? 1.0f - (u - x) : x - (u - 1.0f);
      c = c / u;
    } else {
      u = x;
      hu = as_int(u);
      k = (hu >> 23) - 127;
      c = 0.0f;
    }
    hu &= 0x007fffff;
    if (hu < 0x3504f4) {
      u = as_float(hu | 0x3f800000);
    } else {
      k += 1;
      u = as_float(hu | 0x3f000000);
      hu = (0x00800000 - hu) >> 2;
    }
    f = u - 1.0f;
  }
  const float hfsq = (0.5f * f) * f;
  const float fk = (float)k;
  if (hu == 0) {
    if (f == 0.0f) {
      if (k == 0) return 0.0f;
      c = c + fk * ln2_lo;
      return fk * ln2_hi + c;
    }
    const float R = hfsq * (1.0f - 0.6666666666666666f * f);
    if (k == 0) return f - R;
    return fk * ln2_hi - ((R - (fk * ln2_lo + c)) - f);
  }
  const float s = f / (2.0f + f);
  const float z = s * s;
  const float R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
  if (k == 0) return f - (hfsq - s * (hfsq + R));
  return fk * ln2_hi - ((hfsq - (s * (hfsq + R) + (fk * ln2_lo + c))) - f);
}

inline float nb_sigmoid(float x) { return 1.0f / (nb_exp(0.0f - x) + 1.0f); }
inline float nb_silu_conv(float v) { return v / (1.0f + nb_exp(-v)); }
inline float nb_silu_gate(float z) { return z * (1.0f / (1.0f + nb_exp(-z))); }
inline float nb_softplus(float v) { return v <= 20.0f ? nb_log1p(nb_exp(v)) : v; }

#endif

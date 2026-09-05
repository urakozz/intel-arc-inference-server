// pf_gated_head.cl -- `Qwen3_5RMSNormGated` over one GDN v-head at runtime `M`
// and S = 1. Grid (48 v-heads, M), work-group 128, so lane `i` owns channel `i`
// and the variance tree is exactly the 128 lanes.
//
// `src/kernels/prep.cl:355-385` transcribed with two edits and nothing else:
// `#ifndef M / #define M 1` is gone, and `uint m_count` is appended to the
// argument list and used in the `z` load's stride. All five RNE points
// (`o_b`, `z_b`, `n_b`, `t_b`, the final product), the 128-lane variance tree,
// `1.0f / sqrt(var + 1e-6f)` and the `silu` factor LAST are the numerics
// contract and are untouched -- `tests/prefill/pf_attn_test.cc` holds the M = 1
// output bit-identical to `prep_gated_head_M1`, which is only available because
// this is the same text.
//
// **The S = 1 correction does not reach this kernel**, and that is worth
// stating rather than discovering: decode's `GATED_S` is ALREADY 1 (qkv||z runs
// unsplit, model::Qwen35's table), so its widened cost at M = C is what it is.
// What this file buys over `prep_gated_head_M2048` is only the runtime `M`.
//
//   o_b = rne_bf16(gdn_o[m][h][i])                    (recurrence output -> bf16)
//   z_b = rne_bf16(qkvz_partials[m][Z_OFF + h.128 + i])
//   var = mean_i(f32(o_b)^2)                          (128 -> 64 -> ... -> 1)
//   n_b = rne_bf16(f32(o_b) . (1 / sqrt(var + 1e-6)))
//   t_b = rne_bf16(f32(gated_w[i]) . f32(n_b))        (plain w, NO +1)
//   x_out[m][h.128+i] = rne_bf16(f32(t_b) . silu_f32(f32(z_b)))

#define GATED_S 1
#if GATED_S != 1
#error "pf_gated_head: the prefill path folds ONE slice (plan 6b ruling R1)"
#endif
#define GATED_HEADS 48
#define HEAD_DIM 128
#define QKVZ_N 16384
#define Z_OFF 10240
#define GATED_OUT_N (GATED_HEADS * HEAD_DIM)
#define WG_GATED 128

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

__attribute__((reqd_work_group_size(WG_GATED, 1, 1)))
__kernel void pf_gated_head(__global const float* restrict qkvz_partials,
                            __global const float* restrict gdn_o,
                            __global const ushort* restrict gated_w,
                            __global ushort* restrict x_out, uint m_count) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float red[HEAD_DIM];

  const ushort o_b = rne_bf16(gdn_o[((size_t)m * GATED_HEADS + h) * HEAD_DIM + i]);
  float za = 0.f;
  for (uint s = 0; s < GATED_S; ++s)
    za += qkvz_partials[((size_t)s * m_count + m) * QKVZ_N + Z_OFF + h * HEAD_DIM + i];
  const ushort z_b = rne_bf16(za);

  const float o_f = bf16f(o_b);
  red[i] = o_f * o_f;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_GATED / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] += red[i + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float var = red[0] / (float)HEAD_DIM;
  const float rstd = 1.0f / sqrt(var + 1e-6f);

  const ushort n_b = rne_bf16(o_f * rstd);
  const ushort t_b = rne_bf16(bf16f(gated_w[i]) * bf16f(n_b));
  x_out[(size_t)m * GATED_OUT_N + h * HEAD_DIM + i] =
      rne_bf16(bf16f(t_b) * silu_f32(bf16f(z_b)));
}

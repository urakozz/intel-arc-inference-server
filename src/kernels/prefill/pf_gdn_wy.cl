// pf_gdn_wy.cl - `gdn_chunk` part B: the intra-chunk WY representation.
// `A`, the unit-lower-triangular solve `T = (I - A)^-1`, the `W`/`U` recompute,
// and `A2`. Four entry points, one file.
//
// The algorithm is transcribed once in `tests/prefill/gdn_chunk_ref.h` and
// **that file and this one must be edited together**. FLA's stages, for the
// record: `chunk_scaled_dot_kkt.py:82-112` (A), `solve_tril.py:64-100` (T),
// `wy_fast.py:69-115` (vb/kb, w/u), `chunk_o.py:90-125` (A2).
//
// Every kernel's grid is **(48 heads, ceil(C/64))**; `c_count` is the call's
// total position count, so the tail chunk's live length is
// `L = min(CT, c_count - chunk*CT)` and every loop is bounded by it. Entries of
// `A`, `T` and `A2` outside `[0,L) x [0,L)` are written **exactly 0.0f**,
// including `T`'s dead diagonal - the convention `gdn_wy_test` asserts.
//
// ---------------------------------------------------------------------------
// THE MASK PAIR - the one thing to check by reading the two kernels together
// ---------------------------------------------------------------------------
//   pf_gdn_A :  A[i][j] = beta[i] * (k_i . k_j) * exp(gc[i]-gc[j])   for  i > j
//   pf_gdn_A2: A2[i][j] =           (q_i . k_j) * exp(gc[i]-gc[j])   for  j <= i
//
// **A2's mask INCLUDES the diagonal (`<=`, chunk_o.py:123) and A's does NOT
// (`>`, chunk_scaled_dot_kkt.py:108).** Getting the pair backwards is an
// off-by-one in the recurrence that a 64-position test catches and a
// 1-position test does not. They differ in exactly three things - the left
// operand, the `beta[i]` factor, and that mask - and they are written adjacent
// below so a reader checks them against each other on one screen.
//
// ---------------------------------------------------------------------------
// Rounding and reduction discipline
// ---------------------------------------------------------------------------
//   * every 128-term dot uses gdn_step.cl:52-65's band tree: band `b`
//     accumulates 8 terms in ASCENDING kk with explicit `fma`, then the 16
//     band partials collapse with stride = 8, 4, 2, 1. The tree's ORDER is what
//     the host reference reproduces; that it runs inside one work-item here and
//     across 16 subgroups in `pf_gdn_scan` does not change the order.
//   * `q` is read as `f32(word) * Q_SCALE` - the scale is applied at READ, in
//     fp32, after the l2norm's bf16 round (P6). `k` is read unscaled (P7);
//     the `* 1.0f` in the shared dot is exact and is there so both callers use
//     one function.
//   * **Q1-Q4, the four bf16 roundings the chunked form ADDS** and for which
//     decode has no twin, all live in `pf_gdn_wu`:
//       Q1 vb[j][x] = rne(v[j][x] * beta[j])                  wy_fast.py:88
//       Q2 kb[j][k] = rne(k[j][k] * beta[j] * exp(gc[j]))     wy_fast.py:110
//       Q3 u[i][x]  = rne(SUM_{j<=i} T[i][j] * f32(vb[j][x])) wy_fast.py:89
//       Q4 w[i][k]  = rne(SUM_{j<=i} T[i][j] * f32(kb[j][k])) wy_fast.py:112
//     They are kept because they are the reference vLLM runs, and they are the
//     dominant term of the numerics band
//     (docs/prefill-l1-preregistration-2026-09-05.md §2.2). Each expression is
//     rounded ONCE, at its end, as FLA's `.to(dtype)` does.
//   * `T` is **fp32**, deviating from FLA's `solve_tril(..., output_dtype =
//     k.dtype)` (chunk.py:47-49) - plan 6b ruling R8, deliberate and recorded.

#define HEADS 48
#define DIM 128
#define CONV_ROWS 10240
#define Q_OFF 0
#define K_OFF 2048
#define V_OFF 4096
#define CT 64             /* the FLA intra-chunk size (PrefillScratch::kGdnChunk) */
#define BANDS 16
#define BAND_K 8
#define Q_SCALE 0.08838834764831845f   /* 1/sqrt(128), applied to q in fp32 */

#define WG_TRI 256        /* pf_gdn_A / pf_gdn_A2 / pf_gdn_wu */
#define WG_SOLVE 64       /* one lane per column j */

/* pf_gdn_wu's tiling after ruling A27 - see the header block above (4). */
#define SG 16             /* SIMD16: 16 subgroups of 16 lanes */
#define WU_COLS 64        /* output columns per work-group: 128 / 2 (grid.z) */
#define TS_LD 68          /* TsT's row stride: 64 padded to a multiple of 4 */
#define PB 4              /* positions per block; each subgroup owns two blocks */
#define VPW 2             /* output columns per lane */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// gdn_step.cl:52-65's band tree, over two SLM-staged bf16 rows. `ascale` is
// 1.0f for `k` (exact) and Q_SCALE for `q`.
inline float band_dot(__local const ushort* restrict a, __local const ushort* restrict b,
                      float ascale) {
  float p[BANDS];
  for (uint band = 0; band < BANDS; ++band) {
    float acc = 0.0f;
    for (uint j = 0; j < BAND_K; ++j) {          // ascending within the band
      const uint k = band * BAND_K + j;
      acc = fma(bf16f(a[k]) * ascale, bf16f(b[k]), acc);
    }
    p[band] = acc;
  }
  for (uint stride = BANDS / 2; stride > 0; stride >>= 1) {
    for (uint band = 0; band < stride; ++band) p[band] += p[band + stride];
  }
  return p[0];
}

// Stage L rows of one 128-wide slice of `xb` into SLM, row-major [L][DIM].
inline void stage_rows(__global const ushort* restrict xb, uint base_m, uint L, uint off,
                       __local ushort* restrict dst) {
  for (uint p = get_local_id(0); p < L * DIM; p += WG_TRI) {
    const uint i = p / DIM, d = p % DIM;
    dst[p] = xb[(size_t)(base_m + i) * CONV_ROWS + off + d];
  }
}

// ---------------------------------------------------------------------------
// (1) A[i][j] = beta[i] * (k_i . k_j) * exp(gc[i] - gc[j])   for i > j, else 0.
//     Grid (48, nchunks), WG 256. Work-item `lid` walks pairs p = lid,
//     lid+256, ... over CT*CT.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_TRI, 1, 1)))
__kernel void pf_gdn_A(__global const ushort* restrict xb,
                       __global const float* restrict g_cum,
                       __global const float* restrict beta,
                       __global float* restrict A, uint c_count) {
  const uint h = get_group_id(0), chunk = get_group_id(1);
  const uint kh = h / 3;                          // repeat_interleave(., 3)
  const uint base_m = chunk * CT;
  const uint L = min((uint)CT, c_count - base_m);

  __local ushort ks[CT * DIM];
  __local float gcs[CT], bts[CT];
  stage_rows(xb, base_m, L, K_OFF + kh * DIM, ks);
  if (get_local_id(0) < L) {
    const uint i = get_local_id(0);
    gcs[i] = g_cum[(size_t)(base_m + i) * HEADS + h];
    bts[i] = beta[(size_t)(base_m + i) * HEADS + h];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  __global float* restrict At = A + ((size_t)(chunk * HEADS + h) * CT) * CT;
  for (uint p = get_local_id(0); p < CT * CT; p += WG_TRI) {
    const uint i = p / CT, j = p % CT;
    float val = 0.0f;
    if (i < L && j < L && i > j)                  // NOTE: strict - see the header
      val = bts[i] * band_dot(ks + i * DIM, ks + j * DIM, 1.0f) * exp(gcs[i] - gcs[j]);
    At[p] = val;
  }
}

// ---------------------------------------------------------------------------
// (2) A2[i][j] = (q_i . k_j) * exp(gc[i] - gc[j])   for j <= i, else 0.
//     Grid (48, nchunks), WG 256. Identical to (1) except: the left operand is
//     q scaled by Q_SCALE, there is no beta[i] factor, and the mask INCLUDES
//     the diagonal.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_TRI, 1, 1)))
__kernel void pf_gdn_A2(__global const ushort* restrict xb,
                        __global const float* restrict g_cum,
                        __global float* restrict A2, uint c_count) {
  const uint h = get_group_id(0), chunk = get_group_id(1);
  const uint kh = h / 3;
  const uint base_m = chunk * CT;
  const uint L = min((uint)CT, c_count - base_m);

  __local ushort ks[CT * DIM], qs[CT * DIM];
  __local float gcs[CT];
  stage_rows(xb, base_m, L, K_OFF + kh * DIM, ks);
  stage_rows(xb, base_m, L, Q_OFF + kh * DIM, qs);
  if (get_local_id(0) < L)
    gcs[get_local_id(0)] = g_cum[(size_t)(base_m + get_local_id(0)) * HEADS + h];
  barrier(CLK_LOCAL_MEM_FENCE);

  __global float* restrict At = A2 + ((size_t)(chunk * HEADS + h) * CT) * CT;
  for (uint p = get_local_id(0); p < CT * CT; p += WG_TRI) {
    const uint i = p / CT, j = p % CT;
    float val = 0.0f;
    if (i < L && j < L && j <= i)                 // NOTE: the diagonal IS included
      val = band_dot(qs + i * DIM, ks + j * DIM, Q_SCALE) * exp(gcs[i] - gcs[j]);
    At[p] = val;
  }
}

// ---------------------------------------------------------------------------
// (3) **T = (I + A)^-1**, IN PLACE over A. Grid (48, nchunks), WG 64 = one lane
//     per column j.
//
//     **THE SIGN.** FLA negates `A` on the way in - `solve_tril.py:82` is
//     `b_A = -tl.where(m_A, b_A, 0)` - so with `A` stored positive by
//     `pf_gdn_A` (as `chunk_scaled_dot_kkt.py` stores it), the object this
//     kernel produces is `(I + A)^-1`. Plan 6b's algebra section writes
//     `T = (I - A)^-1`, which is a **transcription error in the plan**: at
//     L = 2 the recurrence needs the coefficient on `Delta_0` in `vn_1` to be
//     `-beta_1 exp(g_1) (k_1 . k_0)`, i.e. exactly `-A[1][0]`, and
//     `(I - A)^-1` supplies `+A[1][0]`. Derived by hand and then confirmed
//     against the FLA line above; the end-to-end band in
//     `tests/prefill/gdn_chunk_test.cc` is what surfaced it, because an
//     `(I - A) T = I` identity check passes either way.
//
//     The negation is applied at staging, where FLA applies it, and the
//     substitution below is then the plain one, `i` ascending:
//         T[i][j] = As[i][j] + SUM_{j<l<i} As[i][l] * T[l][j],   As = -A
//     `As[i][*]` is read at step i, BEFORE row i is overwritten; `T[l][*]` for
//     l < i is already written. `l` ascending, explicit fma.
//
//     **The two barriers are the whole correctness argument for the in-place
//     solve.** The first separates "every lane has READ row i's As values" from
//     "row i becomes T"; the second separates that write from step i+1's reads.
//     Rows >= L stay zero - the kernel must NOT write the identity diagonal
//     past L - and lanes j > i write nothing.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_SOLVE, 1, 1)))
__kernel void pf_gdn_solve(__global float* restrict A, uint c_count) {
  const uint h = get_group_id(0), chunk = get_group_id(1);
  const uint base_m = chunk * CT;
  const uint L = min((uint)CT, c_count - base_m);
  const uint j = get_local_id(0);

  __local float As[CT * CT];
  __global float* restrict At = A + ((size_t)(chunk * HEADS + h) * CT) * CT;
  for (uint p = j; p < CT * CT; p += WG_SOLVE) {
    const uint pi = p / CT, pj = p % CT;
    As[p] = pi > pj ? -At[p] : 0.0f;             // solve_tril.py:82
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint i = 0; i < L; ++i) {
    float acc = 0.0f;
    if (j < i) {
      acc = As[i * CT + j];
      for (uint l = j + 1; l < i; ++l) acc = fma(As[i * CT + l], As[l * CT + j], acc);
    } else if (j == i) {
      acc = 1.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // every lane has READ row i's A values
    if (j <= i) As[i * CT + j] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);   // row i is T before step i+1 reads it
  }

  for (uint p = j; p < CT * CT; p += WG_SOLVE) At[p] = As[p];
}

// ---------------------------------------------------------------------------
// (4) vb/kb (Q1/Q2), then u = T*vb and w = T*kb (Q3/Q4), stored bf16.
//
//     REWRITTEN under ruling A27 (2026-09-05): an output tile, fp32 operands in
//     SLM, and a MIRROR-PAIRED position block.
//     ------------------------------------------------------------------------
//     As delivered by L1-core this was `for (p = lid; p < L*DIM; p += 256)` -
//     **one output `(i, x)` per work-item per step, with no tile**. Each output
//     streamed its whole `j <= i` chain out of SLM: per `j`, `Ts[i*64+j]` plus
//     `vbs[j*128+x]` and `kbs[j*128+x]` (bf16, each needing the `shl`/`mov`
//     widen) for **2 FMAs**, with `Ts[i][j]` re-loaded for every `x` and
//     `vb[j][x]` for every `i`. It measured **2.0595 ms per GDN layer per chunk
//     = 0.79 TFLOP/s** on 1.636 GFLOP (docs/prefill-gdn-scan-2026-09-05.md
//     §5.1), ~18% FMA density.
//
//     Four changes, and nothing else:
//
//     * **Grid (48 heads, nchunks, 2).** `get_group_id(2)` picks which 64 of the
//       128 output columns this work-group owns. That halves the operand
//       staging, which is what buys the fp32 staging below inside the same SLM
//       budget class, and it doubles the launch to 3072 work-groups.
//     * **`vbs`/`kbs` are staged as fp32**, holding `f32(rne(...))` - the SAME
//       rounded bf16 value, widened once at staging instead of once per use, so
//       **the widen leaves the inner loop entirely**. Q1 and Q2 are unmoved:
//       what changed is the container, not the rounding, and a bf16 value is
//       exact in fp32 by construction.
//     * **`T` is staged TRANSPOSED**, `TsT[j][i] = T[i][j]`, at row stride
//       TS_LD = 68 so that four consecutive `i` at one `j` are one 16 B aligned
//       vector read. 68 rather than 65 (conflict-free writes, unaligned reads)
//       or 64 (aligned reads, 16-way conflicted writes): it leaves a 4-way bank
//       conflict on the 16 staging writes each work-item does, ~3% of compute.
//     * **An 8 x 2 output tile per work-item, whose 8 positions are a MIRROR
//       PAIR.** Subgroup `sgid` takes `cs = sgid & 1` (which 32 of the 64
//       columns) and `b = sgid >> 1` (which pair of position blocks); lane `l`
//       owns the two CONSECUTIVE columns `32*cs + 2l` and `+1`, so a subgroup
//       reads 128 contiguous bytes of `vbs` and of `kbs` per `j`.
//
//       **Why mirrored, and not 8 consecutive positions.** The sum is
//       triangular. A tile of `i = 56..63` would carry 484 of the work-group's
//       2080 `(i,j)` pairs against a mean of 260 - 1.86x - and a work-group
//       finishes when its slowest subgroup does. Pairing the low block `4b..4b+3`
//       with its mirror `60-4b..63-4b` gives EVERY subgroup exactly
//       `(16b+10) + (16(15-b)+10) = 260` pairs.
//
//     The loop runs in two phases so it is exactly as long as it must be: phase
//     A over `j = 0 .. min(4b+3, L-1)` with both blocks, phase B over the rest
//     of the high block's range with the high block alone. **Ascending `j`, one
//     accumulator per output, contiguous across the two phases - so the
//     association is the one `gdn_chunk_ref::wu` documents, unchanged, and the
//     bar is bit equality against `pf_gdn_wu_legacy` below.**
//
//     No mask is needed on the triangle: `pf_gdn_solve` writes exactly `0.0f`
//     for `j > i`, so the 12 of 260 pairs (4.6%) a block's shared `j` range
//     over-runs cost `fma(0.0f, vb, acc)`, which is a bitwise no-op - `0*x` is
//     `+/-0` for finite `x`, `acc` is never `-0.0f` (it starts at `+0.0f` and an
//     exactly-cancelling sum returns `+0.0f` under round-to-nearest), and
//     `acc + 0.0f` is `acc`. The `j` range is still clamped to `L-1`, so no
//     work-item reads a `vbs`/`kbs` row the staging loop did not write.
//
//     SLM: TsT 64x68 fp32 (17,408 B) + vbs and kbs 64x64 fp32 (16,384 B each)
//     + bts/egs (512 B) = **50,688 B**, against the previous 49,152 B.
//       w, u  bf16 [C][48][128], indexed ((m * 48 + h) * 128 + x)
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_TRI, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_wu(__global const ushort* restrict xb,
                        __global const float* restrict T,
                        __global const float* restrict g_cum,
                        __global const float* restrict beta,
                        __global ushort* restrict w,
                        __global ushort* restrict u, uint c_count) {
  const uint h = get_group_id(0), chunk = get_group_id(1), cxh = get_group_id(2);
  const uint kh = h / 3;
  const uint base_m = chunk * CT;
  const uint L = min((uint)CT, c_count - base_m);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG, lane = lid % SG;
  const uint cs = sgid & 1u;                 // which 32 of this group's 64 columns
  const uint b = sgid >> 1;                  // which mirror pair of position blocks
  const uint c0 = cs * (SG * VPW) + lane * VPW;   // first column inside the 64-wide half
  const uint dcol = cxh * WU_COLS;                // the half's base in the 128-wide head dim

  __local float TsT[CT * TS_LD];
  __local float vbs[CT * WU_COLS], kbs[CT * WU_COLS];
  __local float bts[CT], egs[CT];

  __global const float* restrict Tt = T + ((size_t)(chunk * HEADS + h) * CT) * CT;
  // TsT[j][i] = T[i][j]. The global read is coalesced; the SLM write strides by
  // TS_LD, which is the 4-way conflict the header prices.
  for (uint p = lid; p < CT * CT; p += WG_TRI) TsT[(p % CT) * TS_LD + (p / CT)] = Tt[p];
  if (lid < L) {
    const size_t idx = (size_t)(base_m + lid) * HEADS + h;
    bts[lid] = beta[idx];
    egs[lid] = exp(g_cum[idx]);                   // one exp per (j, head)
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint p = lid; p < L * WU_COLS; p += WG_TRI) {
    const uint j = p / WU_COLS, d = p % WU_COLS;
    const size_t row = (size_t)(base_m + j) * CONV_ROWS;
    const float v = bf16f(xb[row + V_OFF + h * DIM + dcol + d]);
    const float kv = bf16f(xb[row + K_OFF + kh * DIM + dcol + d]);
    vbs[p] = bf16f(rne_bf16(v * bts[j]));                                       // Q1
    kbs[p] = bf16f(rne_bf16(kv * bts[j] * egs[j]));                             // Q2
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint il = b * PB;                    // the low block's first position
  const uint ih = CT - PB - b * PB;          // 60 - 4b, its mirror
  // A block that starts past L has no live output; its `j` range is empty and
  // its accumulators are never stored.
  const int jl = il < L ? (int)min(il + PB - 1, L - 1) : -1;
  const int jh = ih < L ? (int)min(ih + PB - 1, L - 1) : -1;

  float au[2 * PB][VPW], aw[2 * PB][VPW];
#pragma unroll
  for (uint p = 0; p < 2 * PB; ++p) {
    au[p][0] = 0.0f; au[p][1] = 0.0f;
    aw[p][0] = 0.0f; aw[p][1] = 0.0f;
  }

  // --- phase A: both blocks, j ascending -----------------------------------
  for (int j = 0; j <= jl; ++j) {
    const float2 vv = vload2(0, vbs + (uint)j * WU_COLS + c0);
    const float2 kk = vload2(0, kbs + (uint)j * WU_COLS + c0);
    const float4 tl = vload4(0, TsT + (uint)j * TS_LD + il);
    const float4 th = vload4(0, TsT + (uint)j * TS_LD + ih);
    const float tv[2 * PB] = {tl.s0, tl.s1, tl.s2, tl.s3, th.s0, th.s1, th.s2, th.s3};
#pragma unroll
    for (uint p = 0; p < 2 * PB; ++p) {
      au[p][0] = fma(tv[p], vv.s0, au[p][0]);
      au[p][1] = fma(tv[p], vv.s1, au[p][1]);
      aw[p][0] = fma(tv[p], kk.s0, aw[p][0]);
      aw[p][1] = fma(tv[p], kk.s1, aw[p][1]);
    }
  }

  // --- phase B: the high block alone, j continuing to ascend ----------------
  for (int j = jl + 1; j <= jh; ++j) {
    const float2 vv = vload2(0, vbs + (uint)j * WU_COLS + c0);
    const float2 kk = vload2(0, kbs + (uint)j * WU_COLS + c0);
    const float4 th = vload4(0, TsT + (uint)j * TS_LD + ih);
    const float tv[PB] = {th.s0, th.s1, th.s2, th.s3};
#pragma unroll
    for (uint t = 0; t < PB; ++t) {
      const uint p = PB + t;
      au[p][0] = fma(tv[t], vv.s0, au[p][0]);
      au[p][1] = fma(tv[t], vv.s1, au[p][1]);
      aw[p][0] = fma(tv[t], kk.s0, aw[p][0]);
      aw[p][1] = fma(tv[t], kk.s1, aw[p][1]);
    }
  }

#pragma unroll
  for (uint p = 0; p < 2 * PB; ++p) {
    const uint i = p < PB ? il + p : ih + (p - PB);
    if (i < L) {
      const size_t out = ((size_t)(base_m + i) * HEADS + h) * DIM + dcol + c0;
      u[out + 0] = rne_bf16(au[p][0]);                                          // Q3
      u[out + 1] = rne_bf16(au[p][1]);
      w[out + 0] = rne_bf16(aw[p][0]);                                          // Q4
      w[out + 1] = rne_bf16(aw[p][1]);
    }
  }
}

// ---------------------------------------------------------------------------
// (4b) `pf_gdn_wu` EXACTLY as it stood before ruling A27's rewrite, kept as the
//      bitwise reference for `tests/prefill/gdn_wy_test.cc` and launched by
//      nothing else - `src/runtime/prefill/gdn.cc` binds only the tiled kernel
//      above. Its grid is (48, nchunks, 1): one work-item owns all 128 columns.
//
//      It is here for the same reason `pf_gdn_conv_legacy` is: the rewrite
//      re-associates nothing, so the pre-registered bar is bit equality, and
//      device-vs-device is the only comparison that can carry it for `w` (whose
//      Q2 factor `exp(gc[j])` is 3 ulp on the device and correctly rounded on
//      the host, which is why the standing host bar for `w` is <= 2 bf16 ulp).
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_TRI, 1, 1)))
__kernel void pf_gdn_wu_legacy(__global const ushort* restrict xb,
                               __global const float* restrict T,
                               __global const float* restrict g_cum,
                               __global const float* restrict beta,
                               __global ushort* restrict w,
                               __global ushort* restrict u, uint c_count) {
  const uint h = get_group_id(0), chunk = get_group_id(1);
  const uint kh = h / 3;
  const uint base_m = chunk * CT;
  const uint L = min((uint)CT, c_count - base_m);
  const uint lid = get_local_id(0);

  __local float Ts[CT * CT];
  __local ushort vbs[CT * DIM], kbs[CT * DIM];
  __local float bts[CT], egs[CT];

  __global const float* restrict Tt = T + ((size_t)(chunk * HEADS + h) * CT) * CT;
  for (uint p = lid; p < CT * CT; p += WG_TRI) Ts[p] = Tt[p];
  if (lid < L) {
    const size_t idx = (size_t)(base_m + lid) * HEADS + h;
    bts[lid] = beta[idx];
    egs[lid] = exp(g_cum[idx]);                   // one exp per (j, head)
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint p = lid; p < L * DIM; p += WG_TRI) {
    const uint j = p / DIM, d = p % DIM;
    const size_t row = (size_t)(base_m + j) * CONV_ROWS;
    const float v = bf16f(xb[row + V_OFF + h * DIM + d]);
    const float kv = bf16f(xb[row + K_OFF + kh * DIM + d]);
    vbs[p] = rne_bf16(v * bts[j]);                                              // Q1
    kbs[p] = rne_bf16(kv * bts[j] * egs[j]);                                    // Q2
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint p = lid; p < L * DIM; p += WG_TRI) {
    const uint i = p / DIM, x = p % DIM;
    float au = 0.0f, aw = 0.0f;
    for (uint j = 0; j <= i; ++j) {               // j ascending, explicit fma
      const float t = Ts[i * CT + j];
      au = fma(t, bf16f(vbs[j * DIM + x]), au);
      aw = fma(t, bf16f(kbs[j * DIM + x]), aw);
    }
    const size_t out = ((size_t)(base_m + i) * HEADS + h) * DIM + x;
    u[out] = rne_bf16(au);                                                      // Q3
    w[out] = rne_bf16(aw);                                                      // Q4
  }
}

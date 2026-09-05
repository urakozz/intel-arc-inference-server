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

/* pf_gdn_A2's / pf_gdn_A's quadrant tiling after ruling A28 - see (2) below. */
#define AQ 32             /* quadrant edge: CT / 2, so grid.z is 2 x 2 */
#define ASG 16            /* SIMD16 - the deliberate width, see (2) */
#define TI 2              /* output positions per work-item */
#define TJ 2              /* output columns per work-item */

/* pf_gdn_wu's tiling after ruling A27 - see the header block above (4). */
#define SG 16             /* SIMD16: 16 subgroups of 16 lanes - MEASURED, see (4) */
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

// ---------------------------------------------------------------------------
// `band_dot`'s tree, for a TI x TJ = 2 x 2 output TILE, over fp32 rows already
// staged in SLM (ruling A28). `a0`/`a1` are the tile's two LEFT rows (already
// carrying `ascale`, which is why there is no scale parameter here), `b0`/`b1`
// its two RIGHT rows; the result is
//     .s0 = a0.b0   .s1 = a0.b1   .s2 = a1.b0   .s3 = a1.b1
// and each component is bit-for-bit `band_dot`'s value for that pair.
// ---------------------------------------------------------------------------
inline float4 band4(__local const float* restrict a0, __local const float* restrict a1,
                    __local const float* restrict b0, __local const float* restrict b1,
                    uint kb) {
  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (uint t = 0; t < BAND_K; ++t) {          // ascending within the band
    const float x0 = a0[kb + t], x1 = a1[kb + t];
    const float y0 = b0[kb + t], y1 = b1[kb + t];
    acc.s0 = fma(x0, y0, acc.s0);
    acc.s1 = fma(x0, y1, acc.s1);
    acc.s2 = fma(x1, y0, acc.s2);
    acc.s3 = fma(x1, y1, acc.s3);
  }
  return acc;
}

// The 16 band partials collapsed in EXACTLY `band_dot`'s order. Its loop
//     for (stride = 8, 4, 2, 1) for (b < stride) p[b] += p[b + stride]
// expands to the balanced tree
//     ( ((P0+P8)+(P4+P12)) + ((P2+P10)+(P6+P14)) )
//   + ( ((P1+P9)+(P5+P13)) + ((P3+P11)+(P7+P15)) )
// -- the bands in bit-reversed order, paired. Written as the statements below
// rather than as a `p[16]` array, the live set is FOUR float4 values instead of
// sixteen, which is what lets the tile fit at SIMD16 with room to spare; the
// expression is the same one, term for term and parenthesis for parenthesis,
// and that is the whole reason the rewrite is bitwise.
inline float4 tile_dot(__local const float* restrict a0, __local const float* restrict a1,
                       __local const float* restrict b0, __local const float* restrict b1) {
#define BD(b) band4(a0, a1, b0, b1, (b) * BAND_K)
  float4 s0 = BD(0) + BD(8);                   // q[0]
  s0 = s0 + (BD(4) + BD(12));                  // r[0] = q[0] + q[4]
  float4 t2 = BD(2) + BD(10);                  // q[2]
  t2 = t2 + (BD(6) + BD(14));                  // r[2] = q[2] + q[6]
  s0 = s0 + t2;                                // s[0] = r[0] + r[2]
  float4 s1 = BD(1) + BD(9);                   // q[1]
  s1 = s1 + (BD(5) + BD(13));                  // r[1] = q[1] + q[5]
  float4 t3 = BD(3) + BD(11);                  // q[3]
  t3 = t3 + (BD(7) + BD(15));                  // r[3] = q[3] + q[7]
  s1 = s1 + t3;                                // s[1] = r[1] + r[3]
  return s0 + s1;
#undef BD
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
//     Identical to (1) except: the left operand is q scaled by Q_SCALE, there
//     is no beta[i] factor, and the mask INCLUDES the diagonal.
//
//     REWRITTEN under ruling A28 (2026-09-05): a QUADRANT grid, fp32 operands
//     in SLM, and a 2 x 2 output tile.
//     ------------------------------------------------------------------------
//     As delivered this was `for (p = lid; p < CT*CT; p += 256)` -- one output
//     per work-item per step, with no tile, exactly `pf_gdn_wu`'s old shape --
//     and it measured **1.0313 ms per GDN layer per chunk = 0.79 TFLOP/s** on
//     0.818 GFLOP (docs/prefill-gdn-wu-conv-2026-09-05.md 6.1). The Xe2
//     assembly named TWO terms, not one
//     (docs/prefill-gdn-a2a-simd32-2026-09-05.md 3.1):
//       * FMA density **10.5%** -- 47% of the 610-instruction hot loop is the
//         bf16 widen, and with one output per work-item no widened word is
//         reused;
//       * occupancy **37.5%** -- 33,024 B of SLM is 3 work-groups per 128 KB
//         Xe-core, and at SIMD32 that is 24 of its 64 thread slots. `pf_gdn_A`,
//         the same code with one staged operand, fits 4 groups x 16 threads and
//         issues at **0.502** instructions/XVE/clock against this kernel's
//         **0.257**. That ratio is the whole 1.68x between them.
//
//     Four changes, and nothing else:
//
//     * **Grid (48 heads, nchunks, 4).** `z` names a quadrant: `iz = z >> 1`
//       owns `i in [32*iz, 32*iz+32)` and `jz = z & 1` owns the same range of
//       `j`, so a work-group holds 1024 of the 4096 outputs and stages only the
//       32 q rows and 32 k rows it needs. The quadrant (iz = 0, jz = 1) is
//       **entirely above the diagonal** -- every entry has j >= 32 > i -- so it
//       writes its 1024 exact `0.0f`s and returns without staging. The other
//       three compute 3 x 1024 = 3072 dots, which is what the old kernel's
//       predication already paid (12 of 16 iterations x 4096); the split makes
//       it explicit instead of burning it in dead lanes.
//     * **`qs`/`ks` staged as fp32**, holding `f32(q_word) * Q_SCALE` and
//       `f32(k_word)` -- the same fp32 values the old kernel formed inside the
//       `fma`'s first argument (IGC emits the `mul` and the `mad` separately,
//       so the product was rounded to fp32 there too), widened once at staging
//       instead of once per use. **SLM: 2 x 32 x 128 x 4 = exactly 32,768 B**,
//       which is 4 work-groups per Xe-core. `gcs` therefore does NOT live in
//       SLM -- 256 B more would drop that to 3 -- and is read from global.
//     * **`intel_reqd_sub_group_size(16)`, chosen and not inherited.** At
//       32,768 B an Xe-core holds 4 work-groups at either width, so the width
//       decides residency: 4 x 16 = **64 threads at SIMD16**, the whole budget,
//       against 4 x 8 = 32 at SIMD32. The measured price of that trade is in
//       the two rows above, and item 1 of the same task measured the other side
//       of it on `pf_gdn_wu`.
//     * **A 2 x 2 output tile.** `bi = lid >> 4`, `bj = lid & 15`; the work-item
//       owns i in {32iz+2bi, +1} and j in {32jz+2bj, +1}. `bi` is uniform
//       inside a SIMD16 thread, so the `qs` reads broadcast and the thread's 16
//       lanes store 32 contiguous fp32 of one A2 row -- a coalesced 128 B write.
//       Four `fma` per four operand values takes the density to ~67%.
//
//     **No rounding point moves and no sum is re-associated**: `tile_dot` above
//     is `band_dot`'s expression, term for term. The bar is therefore bit
//     equality against `pf_gdn_A2_legacy` below.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_TRI, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ASG)))
__kernel void pf_gdn_A2(__global const ushort* restrict xb,
                        __global const float* restrict g_cum,
                        __global float* restrict A2, uint c_count) {
  const uint h = get_group_id(0), chunk = get_group_id(1), z = get_group_id(2);
  const uint iz = z >> 1, jz = z & 1u;
  const uint kh = h / 3;
  const uint base_m = chunk * CT;
  const uint L = min((uint)CT, c_count - base_m);
  const uint lid = get_local_id(0);
  const uint bi = lid / ASG, bj = lid % ASG;    // a 16 x 16 grid of 2 x 2 tiles
  const uint i0 = iz * AQ + bi * TI, j0 = jz * AQ + bj * TJ;

  __global float* restrict At = A2 + ((size_t)(chunk * HEADS + h) * CT) * CT;
  if (iz == 0 && jz == 1) {                     // wholly above the diagonal
#pragma unroll
    for (uint a = 0; a < TI; ++a)
      vstore2((float2)(0.0f, 0.0f), 0, At + (size_t)(i0 + a) * CT + j0);
    return;
  }

  __local float qs[AQ * DIM], ks[AQ * DIM];
  for (uint p = lid; p < AQ * DIM; p += WG_TRI) {
    const uint r = p / DIM, d = p % DIM;
    const uint gi = iz * AQ + r, gj = jz * AQ + r;
    qs[p] = gi < L
                ? bf16f(xb[(size_t)(base_m + gi) * CONV_ROWS + Q_OFF + kh * DIM + d]) * Q_SCALE
                : 0.0f;
    ks[p] = gj < L ? bf16f(xb[(size_t)(base_m + gj) * CONV_ROWS + K_OFF + kh * DIM + d]) : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint qr = bi * TI * DIM, kr = bj * TJ * DIM;
  const float4 dot = tile_dot(qs + qr, qs + qr + DIM, ks + kr, ks + kr + DIM);
  const float dv[4] = {dot.s0, dot.s1, dot.s2, dot.s3};

  // `gcs` is read here rather than staged: 256 B of SLM would cost the fourth
  // resident work-group. Rows past L read 0.0f and are masked away below.
  const float gi[TI] = {
      i0 < L ? g_cum[(size_t)(base_m + i0) * HEADS + h] : 0.0f,
      i0 + 1 < L ? g_cum[(size_t)(base_m + i0 + 1) * HEADS + h] : 0.0f};
  const float gj[TJ] = {
      j0 < L ? g_cum[(size_t)(base_m + j0) * HEADS + h] : 0.0f,
      j0 + 1 < L ? g_cum[(size_t)(base_m + j0 + 1) * HEADS + h] : 0.0f};

#pragma unroll
  for (uint a = 0; a < TI; ++a) {
    const uint i = i0 + a;
    const float v0 =
        (i < L && j0 < L && j0 <= i) ? dv[a * TJ] * exp(gi[a] - gj[0]) : 0.0f;
    const float v1 =
        (i < L && j0 + 1 < L && j0 + 1 <= i) ? dv[a * TJ + 1] * exp(gi[a] - gj[1]) : 0.0f;
    vstore2((float2)(v0, v1), 0, At + (size_t)i * CT + j0);
  }
}

// ---------------------------------------------------------------------------
// (2b) `pf_gdn_A2` EXACTLY as it stood before ruling A28's rewrite, kept as the
//      bitwise reference for `tests/prefill/gdn_wy_test.cc` and launched by
//      nothing else -- `src/runtime/prefill/gdn.cc` binds only the tiled kernel
//      above. Its grid is (48, nchunks, 1).
//
//      It is here for the same reason `pf_gdn_wu_legacy` is: the rewrite
//      re-associates nothing, so the pre-registered bar is bit equality, and
//      device-vs-device is the only comparison that can carry it -- the host
//      reference's `exp` is correctly rounded where the device's is 3 ulp,
//      which is why case 4's host bar is <= 4 fp32 ulp and not equality.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_TRI, 1, 1)))
__kernel void pf_gdn_A2_legacy(__global const ushort* restrict xb,
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
//
//     **SIMD16 IS MEASURED, NOT INHERITED - do not "fix" this attribute.**
//     A28 4.2 derived that dropping to SIMD32 (which is what the legacy entry
//     point below compiles as) would cost 0.465 ms/layer/chunk against the
//     tiled SIMD16 kernel's 0.9298, and named register pressure as its risk.
//     Ruling A28 option (b) built it and MEASURED **1.1505 ms - 1.24x SLOWER**
//     (docs/prefill-gdn-a2a-simd32-2026-09-05.md 2). It was reverted by the
//     pre-registered decision rule. Both halves of the named risk fired:
//       * no spill (`-abortOnSpill 4` would have failed the build), but the
//         32 fp32 accumulators cost 64 of the 128 GRF at SIMD32 and IGC paid
//         for it in register moves -- phase A's body went 45 instructions /
//         32 mad / 71% FMA density at SIMD16 to **84 / 32 / 38%**, 34 of the
//         39 added instructions being plain `mov`;
//       * SLM is 50,688 B, so an Xe-core holds 2 work-groups either way: 32
//         threads at SIMD16 but only **16 at SIMD32**, and the measured issue
//         rate fell 0.121 -> 0.071 instructions/XVE/clock, which is more than
//         the 1.38x of issued instructions the widening saved.
//     The mapping itself IS width-agnostic (`b = (lid/16)>>1` is thread-uniform
//     at SIMD32 and the two halves' `cs` cover the 64 columns exactly once);
//     `w`/`u` came out bit-identical to the legacy kernel at all three widths.
//     The width is not the free lever it looked like; the kernel's remaining
//     lever is the un-unrolled runtime-bounded `j` loop (A28 4.2 term 2).
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

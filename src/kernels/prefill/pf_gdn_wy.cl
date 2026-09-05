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
// (3) T = (I - A)^-1, IN PLACE over A. Grid (48, nchunks), WG 64 = one lane per
//     column j. Forward substitution, `i` ascending:
//         T[i][j] = A[i][j] + SUM_{j<l<i} A[i][l] * T[l][j]
//     `A[i][*]` is read at step i, BEFORE row i is overwritten; `T[l][*]` for
//     l < i is already written. `l` ascending, explicit fma.
//
//     **The two barriers are the whole correctness argument for the in-place
//     solve.** The first separates "every lane has READ row i's A values" from
//     "row i becomes T"; the second separates that write from step i+1's reads.
//     Rows >= L are left as pf_gdn_A wrote them (zero) - the kernel must NOT
//     write the identity diagonal past L - and lanes j > i write nothing.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_SOLVE, 1, 1)))
__kernel void pf_gdn_solve(__global float* restrict A, uint c_count) {
  const uint h = get_group_id(0), chunk = get_group_id(1);
  const uint base_m = chunk * CT;
  const uint L = min((uint)CT, c_count - base_m);
  const uint j = get_local_id(0);

  __local float As[CT * CT];
  __global float* restrict At = A + ((size_t)(chunk * HEADS + h) * CT) * CT;
  for (uint p = j; p < CT * CT; p += WG_SOLVE) As[p] = At[p];
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
//     Grid (48, nchunks), WG 256. SLM: Ts[64][64] fp32 (16 KB) + vbs and kbs
//     [64][128] bf16 (16 KB each) = 48 KB.
//       w, u  bf16 [C][48][128], indexed ((m * 48 + h) * 128 + x)
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_TRI, 1, 1)))
__kernel void pf_gdn_wu(__global const ushort* restrict xb,
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

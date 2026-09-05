// pf_gdn_scan.cl - `gdn_chunk` part C: the chunk-to-chunk sequential state
// scan. One kernel, and the only one in the GDN family that touches
// `gdn_state`.
//
// ---------------------------------------------------------------------------
// The tile mapping - REWRITTEN under ruling A25 (2026-09-05)
// ---------------------------------------------------------------------------
// The grid is unchanged: (48 v-heads, 4 state-column chunks) = 192 work-groups
// of 256 = 16 subgroups of 16 lanes. Work-group `(h, c)` still owns state
// columns `[32c, 32c+32)` of head `h`, and work-item `(sgid, lane)` still owns
// **16 fp32 of state in registers** - k-rows `8*sgid .. 8*sgid+7`, columns
// `32c+lane` and `32c+lane+16`. The state stays k-major, so the 128 v of a row
// are contiguous and a subgroup's 16 lanes read one 64 B line, exactly as
// decode does.
//
// **What changed is the work distribution inside stages 1 and 2, and nothing
// else.** As delivered by L1-core those two stages were decode's `gdn_step`
// mapping (`gdn_step.cl:26-65`) widened one position at a time: 64 sequential
// 256-lane band-tree reductions each, five barriers per tree, with only
// `sgid == 0` - 16 of 256 work-items - running the epilogue. That is 640
// barriers per 64-position sub-chunk, 20,480 per work-group per layer, and it
// measured **15.071 ms per GDN layer per chunk = 0.64 TFLOP/s on ~9.66
// GFLOP** - 35.1% of the whole `--pp 4096` walk
// (docs/prefill-pp-attribution-2026-09-05.md §4).
//
// Stages 1 and 2 are two 64x128x32 matmuls per sub-chunk, so they get a matmul
// shape instead:
//
//   * the chunk-START `S` tile is staged **once per sub-chunk** into
//     `Ss[128][32]` fp32 (16 KB SLM), written straight out of the work-items'
//     own state registers - never re-read from global. It is constant through
//     stages 1-2, which both read the chunk-start state.
//   * every work-item gets an **output tile of 8**: 64 positions x 32 columns
//     = 2048 outputs / 256 lanes. Subgroup `sgid` owns positions
//     `i in [4*sgid, 4*sgid+4)`; lane `l` owns columns `l` and `l+16` - the
//     same two columns it already owns in registers. So a subgroup's 16 lanes
//     read `Ss[k][0..31]`, two full 64 B lines, per `k`, and `w[i][k]` /
//     `q[i][k]` are subgroup-uniform.
//   * each of those 8 outputs is a **private ascending-k fp32 `fma`
//     accumulation over all 128 k**. No `red[]`, no tree, no idle epilogue.
//   * `A2` is read from global at a subgroup-uniform address rather than
//     staged: under this mapping each of the 64 `A2` rows is read by exactly
//     one subgroup, so staging it was pure overhead. SLM per work-group
//     therefore FALLS, 26.5 KB -> 24.5 KB.
//   * **barriers per sub-chunk: 640 -> 3.**
//
// Stage 3 is untouched: it was already per-work-item with no barriers, and it
// keeps `S` in registers. `Ss` is a read-only copy of the chunk-start tile, so
// nothing is written back to it.
//
// ---------------------------------------------------------------------------
// The algebra, and the ONE reassociation relative to decode
// ---------------------------------------------------------------------------
// Per 64-chunk `t`, with `S` the state as it stands at the chunk's FIRST
// position (transcribed in tests/prefill/gdn_chunk_ref.h, which must be edited
// with this file):
//
//   vn[i][x] = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x]
//   o[i][x]  = ( SUM_k q[i][k] * S[k][x] ) * exp(gc[i]) + SUM_{j<=i} A2[i][j]*vn[j][x]
//   S[k][x] <- S[k][x] * exp(gl) + SUM_i k[i][k] * ( vn[i][x] * exp(gl - gc[i]) )
//
// **`o` reads the CHUNK-START `S`, not the updated one.** Computing `vn` and
// `o` before touching `S` is what removes FLA's `[C/64][48][128][128]` fp32 `h`
// buffer (100,663,296 B at C = 2048, derived) entirely - we never materialise a
// per-chunk state snapshot because the scan is fused.
//
// **The reassociation.** Decode interleaves the decay with the rank-1 update
// per position: `S *= exp(g_i)` then `S += kf_i (x) delta_i`, 64 times
// (gdn_step.cl:318-349). The chunk applies `exp(gl)` ONCE and then adds
// `SUM_i kf_i (x) (vn_i * exp(gl - gc_i))`. Algebraically identical; in fp32 it
// replaces 64 sequential multiplies by one, and `exp(gl - gc_i)` by a product
// of `exp(g_j)`. This is a named source of the state band that
// `tests/prefill/gdn_chunk_test.cc` records, and it is a far smaller term than
// Q1-Q4's four bf16 roundings in `pf_gdn_wu`
// (docs/prefill-l1-preregistration-2026-09-05.md §2.2).
//
// **The SECOND source, new with A25 and pre-registered before it was built**
// (docs/prefill-gdn-scan-2026-09-05.md §1.3): both 128-term contractions above
// now run as ONE ascending-k `fma` chain per output instead of
// `gdn_step.cl:52-65`'s 16-band tree. Algebraically identical, differently
// rounded, and it is why A22's 3.506e-02 is not this kernel's bar. The token
// gate is the arbiter; the band is re-measured and recorded.
//
// **Nothing here rounds** (gdn_step.cl:317-356, P9) and `gdn_o` is written fp32
// (gdn_step.cl:368-370, P10) - `pf_gated_head`, `gdn_chunk`'s tenth launch,
// does that rounding. P6 (the q-scale in fp32, after the bf16 read) and P7
// (bf16 `k`) are unchanged and sit where they always did.
//
// SLM: Ss[128][32] (16 KB) + vn[64][32] (8 KB) + gcv[64] / expg[64] (512 B)
//      = 24.5 KB.

#define HEADS 48
#define DIM 128
#define CONV_ROWS 10240
#define Q_OFF 0
#define K_OFF 2048
#define CT 64             /* the FLA intra-chunk size (PrefillScratch::kGdnChunk) */
#define SG 16             /* SIMD16: 16 subgroups of 16 lanes */
#define BAND_K 8          /* k-rows per subgroup band: 128 / 16 */
#define CHUNK_V 32        /* state columns per work-group: 128 / 4 */
#define VPW 2             /* v columns per work-item: 32 / 16 */
#define PPW 4             /* positions per work-item in stages 1-2: 64 / 16 */
#define WG_SCAN 256
#define Q_SCALE 0.08838834764831845f   /* 1/sqrt(128), applied to q in fp32 */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((reqd_work_group_size(WG_SCAN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_scan(__global const ushort* restrict xb,
                          __global const ushort* restrict w,
                          __global const ushort* restrict u,
                          __global const float* restrict A2,
                          __global const float* restrict g_cum,
                          __global float* restrict state,
                          __global float* restrict gdn_o, uint c_count) {
  const uint h = get_group_id(0);          // v-head
  const uint c = get_group_id(1);          // state-column chunk
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;              // owns k-rows [8*sgid, 8*sgid+8)
  const uint lane = lid % SG;              // owns columns 32c+lane and 32c+lane+16
  const uint kh = h / 3;                   // repeat_interleave(., 3)
  const uint col0 = c * CHUNK_V + lane, col1 = col0 + SG;
  const uint i0 = sgid * PPW;              // this subgroup's four positions

  __local float Ss[DIM][CHUNK_V];          // the chunk-START state tile
  __local float vn[CT][CHUNK_V];
  __local float gcv[CT], expg[CT];

  // The work-item's 8x2 state tile into registers (gdn_step.cl:273-278).
  const size_t sbase = (size_t)h * DIM * DIM + (size_t)(sgid * BAND_K) * DIM + col0;
  float S[BAND_K][VPW];
  for (uint kk = 0; kk < BAND_K; ++kk) {
    S[kk][0] = state[sbase + (size_t)kk * DIM];
    S[kk][1] = state[sbase + (size_t)kk * DIM + SG];
  }

  const uint nch = (c_count + CT - 1) / CT;
  for (uint t = 0; t < nch; ++t) {
    const uint base_m = t * CT;
    const uint L = min((uint)CT, c_count - base_m);
    const uint ilast = L - 1;

    // --- stage the gate, and the chunk-START S tile into SLM ------------------
    if (lid < L) {
      const float g = g_cum[(size_t)(base_m + lid) * HEADS + h];
      gcv[lid] = g;
      expg[lid] = exp(g);
    }
#pragma unroll
    for (uint kk = 0; kk < BAND_K; ++kk) {
      Ss[sgid * BAND_K + kk][lane] = S[kk][0];
      Ss[sgid * BAND_K + kk][lane + SG] = S[kk][1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float gl = gcv[L - 1];

    // --- 1. vn[i][x] = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x] -----------
    // Four positions x two columns per work-item; ascending k, one fma chain
    // per output. The `min(., ilast)` clamp keeps the w reads of a short final
    // sub-chunk in bounds; those lanes' results are simply not stored.
    {
      __global const ushort* restrict wp[PPW];
      float a0[PPW], a1[PPW];
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        wp[p] = w + ((size_t)(base_m + min(i0 + p, ilast)) * HEADS + h) * DIM;
        a0[p] = 0.0f;
        a1[p] = 0.0f;
      }
      for (uint k = 0; k < DIM; ++k) {          // ascending k over all 128
        const float s0 = Ss[k][lane], s1 = Ss[k][lane + SG];
#pragma unroll
        for (uint p = 0; p < PPW; ++p) {
          const float wv = bf16f(wp[p][k]);
          a0[p] = fma(s0, wv, a0[p]);
          a1[p] = fma(s1, wv, a1[p]);
        }
      }
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        const uint i = i0 + p;
        if (i < L) {
          const size_t ub = ((size_t)(base_m + i) * HEADS + h) * DIM;
          vn[i][lane] = bf16f(u[ub + col0]) - a0[p];
          vn[i][lane + SG] = bf16f(u[ub + col1]) - a1[p];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // vn is now live for every position

    // --- 2. o[i][x], from the CHUNK-START S ----------------------------------
    {
      __global const float* restrict At = A2 + ((size_t)(t * HEADS + h) * CT) * CT;
      __global const ushort* restrict qp[PPW];
      float a0[PPW], a1[PPW];
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        qp[p] = xb + (size_t)(base_m + min(i0 + p, ilast)) * CONV_ROWS + Q_OFF + kh * DIM;
        a0[p] = 0.0f;
        a1[p] = 0.0f;
      }
      for (uint k = 0; k < DIM; ++k) {          // ascending k over all 128
        const float s0 = Ss[k][lane], s1 = Ss[k][lane + SG];
#pragma unroll
        for (uint p = 0; p < PPW; ++p) {
          const float qv = bf16f(qp[p][k]) * Q_SCALE;   // P6
          a0[p] = fma(qv, s0, a0[p]);
          a1[p] = fma(qv, s1, a1[p]);
        }
      }
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        const uint i = i0 + p;
        if (i < L) {
          float o0 = a0[p] * expg[i], o1 = a1[p] * expg[i];
          __global const float* restrict Ai = At + (size_t)i * CT;
          for (uint j = 0; j <= i; ++j) {              // j ascending, explicit fma
            const float a = Ai[j];
            o0 = fma(a, vn[j][lane], o0);
            o1 = fma(a, vn[j][lane + SG], o1);
          }
          __global float* restrict op =
              gdn_o + ((size_t)(base_m + i) * HEADS + h) * DIM + c * CHUNK_V;
          op[lane] = o0;                               // P10: fp32
          op[lane + SG] = o1;
        }
      }
    }

    // --- 3. the state update. THIS is the reassociation named in the header --
    // Unchanged from L1-core: already per-work-item, no barriers, S in
    // registers. `k[i][kk]` is subgroup-uniform and `vn[i][.]` is two SLM
    // reads per position for 16 fma.
    const float dl = exp(gl);
    for (uint kk = 0; kk < BAND_K; ++kk) {
      S[kk][0] *= dl;
      S[kk][1] *= dl;
    }
    for (uint i = 0; i < L; ++i) {                    // i ascending
      const float sc = exp(gl - gcv[i]);
      const float d0 = vn[i][lane] * sc, d1 = vn[i][lane + SG] * sc;
      const size_t kbase = (size_t)(base_m + i) * CONV_ROWS + K_OFF + kh * DIM;
      for (uint kk = 0; kk < BAND_K; ++kk) {          // kk ascending
        const float kfv = bf16f(xb[kbase + sgid * BAND_K + kk]);            // P7
        S[kk][0] = fma(kfv, d0, S[kk][0]);
        S[kk][1] = fma(kfv, d1, S[kk][1]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // the next chunk reuses Ss, gcv, expg, vn
  }

  // The tile goes back once, after every chunk of this call (gdn_step.cl:378-381).
  for (uint kk = 0; kk < BAND_K; ++kk) {
    state[sbase + (size_t)kk * DIM] = S[kk][0];
    state[sbase + (size_t)kk * DIM + SG] = S[kk][1];
  }
}

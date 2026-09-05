// pf_gdn_scan.cl - `gdn_chunk` part C: the chunk-to-chunk sequential state
// scan. One kernel, and the only one in the GDN family that touches
// `gdn_state`.
//
// **THE TILE MAPPING IS `gdn_step.cl`'s, unchanged** (gdn_step.cl:26-49). Grid
// (48 heads, 4 state-column chunks), work-group 256 = 16 subgroups of 16 lanes.
// Work-group `(h, c)` owns state columns `[32c, 32c+32)` of head `h`; work-item
// `(sgid, lane)` owns k-rows `8*sgid .. 8*sgid+7` and columns `32c+lane` and
// `32c+lane+16` - **16 fp32 of state in registers**. Both reduction trees are
// gdn_step.cl:52-65's: band `sgid` accumulates 8 terms in ascending `kk` with
// `fma`, then the 16 band partials collapse with stride = 8, 4, 2, 1 and a
// barrier after every step.
//
// Choosing the same tile and the same trees is **deliberate**: it is the
// closest the chunked form can sit to the recurrent one, and spec 2 §6.3's band
// is measured against exactly that choice. The state stays k-major so the 128 v
// of a row are contiguous and a subgroup's 16 lanes read one 64 B line, exactly
// as decode does.
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
// of `exp(g_j)`. This is the named source of the state band that
// `tests/prefill/gdn_chunk_test.cc` records, and it is a far smaller term than
// Q1-Q4's four bf16 roundings in `pf_gdn_wu`
// (docs/prefill-l1-preregistration-2026-09-05.md §2.2).
//
// **Nothing here rounds** (gdn_step.cl:317-356, P9) and `gdn_o` is written fp32
// (gdn_step.cl:368-370, P10) - `pf_gated_head`, `gdn_chunk`'s tenth launch,
// does that rounding.
//
// SLM: red[16][32] (2 KB) + vn[64][32] (8 KB) + A2s[64][64] (16 KB) +
//      gcv[64] / expg[64] (512 B) = 26.5 KB.

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

  __local float red[SG][CHUNK_V];
  __local float vn[CT][CHUNK_V];
  __local float A2s[CT * CT];
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

    // --- stage the chunk's triangular block and its gate ---------------------
    __global const float* restrict At = A2 + ((size_t)(t * HEADS + h) * CT) * CT;
    for (uint p = lid; p < CT * CT; p += WG_SCAN) A2s[p] = At[p];
    if (lid < L) {
      const float g = g_cum[(size_t)(base_m + lid) * HEADS + h];
      gcv[lid] = g;
      expg[lid] = exp(g);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float gl = gcv[L - 1];

    // --- 1. vn[i][x] = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x] -----------
    for (uint i = 0; i < L; ++i) {
      const size_t wbase = ((size_t)(base_m + i) * HEADS + h) * DIM;
      float p0 = 0.0f, p1 = 0.0f;
      for (uint kk = 0; kk < BAND_K; ++kk) {          // ascending kk within the band
        const float wv = bf16f(w[wbase + sgid * BAND_K + kk]);
        p0 = fma(S[kk][0], wv, p0);
        p1 = fma(S[kk][1], wv, p1);
      }
      red[sgid][lane] = p0;
      red[sgid][lane + SG] = p1;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint stride = SG / 2; stride > 0; stride >>= 1) {
        if (sgid < stride) {
          red[sgid][lane] += red[sgid + stride][lane];
          red[sgid][lane + SG] += red[sgid + stride][lane + SG];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (sgid == 0) {
        vn[i][lane] = bf16f(u[wbase + col0]) - red[0][lane];
        vn[i][lane + SG] = bf16f(u[wbase + col1]) - red[0][lane + SG];
      }
      barrier(CLK_LOCAL_MEM_FENCE);   // the next i reuses `red`; vn[i] is now live
    }

    // --- 2. o[i][x], from the CHUNK-START S ----------------------------------
    for (uint i = 0; i < L; ++i) {
      const size_t qbase = (size_t)(base_m + i) * CONV_ROWS + Q_OFF + kh * DIM;
      float r0 = 0.0f, r1 = 0.0f;
      for (uint kk = 0; kk < BAND_K; ++kk) {
        const float qv = bf16f(xb[qbase + sgid * BAND_K + kk]) * Q_SCALE;   // P6
        r0 = fma(qv, S[kk][0], r0);
        r1 = fma(qv, S[kk][1], r1);
      }
      red[sgid][lane] = r0;
      red[sgid][lane + SG] = r1;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint stride = SG / 2; stride > 0; stride >>= 1) {
        if (sgid < stride) {
          red[sgid][lane] += red[sgid + stride][lane];
          red[sgid][lane + SG] += red[sgid + stride][lane + SG];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (sgid == 0) {
        float a0 = red[0][lane] * expg[i], a1 = red[0][lane + SG] * expg[i];
        for (uint j = 0; j <= i; ++j) {               // j ascending, explicit fma
          const float a = A2s[i * CT + j];
          a0 = fma(a, vn[j][lane], a0);
          a1 = fma(a, vn[j][lane + SG], a1);
        }
        __global float* restrict op =
            gdn_o + ((size_t)(base_m + i) * HEADS + h) * DIM + c * CHUNK_V;
        op[lane] = a0;                                // P10: fp32
        op[lane + SG] = a1;
      }
      barrier(CLK_LOCAL_MEM_FENCE);   // the next i reuses `red`
    }

    // --- 3. the state update. THIS is the reassociation named in the header --
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
    barrier(CLK_LOCAL_MEM_FENCE);   // the next chunk reuses A2s, gcv, expg, vn
  }

  // The tile goes back once, after every chunk of this call (gdn_step.cl:378-381).
  for (uint kk = 0; kk < BAND_K; ++kk) {
    state[sbase + (size_t)kk * DIM] = S[kk][0];
    state[sbase + (size_t)kk * DIM + SG] = S[kk][1];
  }
}

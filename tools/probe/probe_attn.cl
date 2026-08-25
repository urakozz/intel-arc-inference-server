// probe_attn.cl - ONE kernel, `probe_attn`, which is `attn_decode` with one
// term removed at a time. It exists to answer the question spec 1.5's
// re-assessment memo ranks above every design (§5.2): **what does
// `attn_decode`'s 224.046 µs/launch actually buy?** Four pre-registered cost
// models have died on that kernel (docs/15 §2, §L5) and this file is the
// project's answer to "stop fitting, start subtracting".
//
// **It is a copy, not an include.** `src/kernels/attn.cl` is production and
// stays byte-for-byte untouched; this file duplicates `attn_decode`'s body so
// that a probe can gut it. `PB_MODE_BASE` (every switch at its default) is the
// faithful replica and its FIRST job is to reproduce the in-situ 224.046 µs.
// If it does not, no ablation below means anything and the report says so.
//
// Every ablation deliberately breaks the arithmetic. Nothing here is ever
// compared against `attn_ref.h`, nothing here is ever bound by the runtime, and
// the outputs are read only to stop the compiler deleting the work.
//
// ---------------------------------------------------------------------------
// The switches (all default to the base kernel's behaviour)
// ---------------------------------------------------------------------------
//   PB_BLOCK      KV positions per work-group      (64 = what ships)
//   PB_MAXLEN     grid extent                      (16384 = the loader default)
//   PB_GQA        q-head passes per work-group     (6 = GQA 6:1; 1 isolates the reread)
//   PB_STAGE      0 whole kernel | 1 return at the early-out | 2 return after q staging
//   PB_NO_K       1: the K global load becomes arithmetic (same fma count)
//   PB_NO_V       1: the V global load becomes arithmetic (same fma count)
//   PB_HOT_K      1: every K load reads block 0's rows - same MESSAGE count, L1-resident
//   PB_HOT_V      1: same for V
//   PB_KT / PB_VT 1: index K / V as a TRANSPOSED cache - [KV_HEADS][MAXLEN][HD]
//                   instead of [MAXLEN][KV_HEADS][HD]. Same buffer, same message
//                   count, same bytes; consecutive positions of one kv-head go
//                   from 2048 B apart to 512 B apart, i.e. contiguous.
//   PB_NO_PART    1: the `attn_part` stores are predicated off (nothing else changes)
//   PB_VEC        1 scalar ushort (base) | 2 ushort2 | 4 ushort4 loads - message width
//   PB_EXP        0 `exp` (base) | 1 `native_exp` | 2 a multiply (no transcendental at all)
//   PB_NO_TREE    1: drop the 4-step SLM reduction AND its 4 barriers
//   PB_NO_BARRIER 1: drop the wave's 6 barriers, keep every instruction
//   PB_SG_TREE    1: the tree's first 3 barriers become sub_group_barrier - the SAME
//                   arithmetic in the SAME order, only the sync scope narrows
//   PB_NO_SOFTMAX 1: drop the whole online update (fmax, exp, ssum, the rescales)
//   PB_DEP_BREAK  1: waves stop being loop-carried - (mx, sm) are per-wave, acc is a sum
//   PB_PREFETCH   1: the next wave's K row is loaded before this wave's barriers
//
// The one thing NO switch may do is let the compiler delete work: every
// ablation that removes a load replaces it with arithmetic over `p`/`lid` so
// the loop is not invariant, and every early return still writes `attn_part`.

#ifndef PB_BLOCK
#define PB_BLOCK 64
#endif
#ifndef PB_MAXLEN
#define PB_MAXLEN 16384
#endif
#ifndef PB_GQA
#define PB_GQA 6
#endif
#ifndef PB_STAGE
#define PB_STAGE 0
#endif
#ifndef PB_NO_K
#define PB_NO_K 0
#endif
#ifndef PB_NO_V
#define PB_NO_V 0
#endif
#ifndef PB_HOT_K
#define PB_HOT_K 0
#endif
#ifndef PB_HOT_V
#define PB_HOT_V 0
#endif
#ifndef PB_KT
#define PB_KT 0
#endif
#ifndef PB_VT
#define PB_VT 0
#endif
#ifndef PB_NO_PART
#define PB_NO_PART 0
#endif
#ifndef PB_VEC
#define PB_VEC 1
#endif
#ifndef PB_EXP
#define PB_EXP 0
#endif
#ifndef PB_NO_TREE
#define PB_NO_TREE 0
#endif
#ifndef PB_NO_BARRIER
#define PB_NO_BARRIER 0
#endif
#ifndef PB_SG_TREE
#define PB_SG_TREE 0
#endif
#ifndef PB_NO_SOFTMAX
#define PB_NO_SOFTMAX 0
#endif
#ifndef PB_DEP_BREAK
#define PB_DEP_BREAK 0
#endif
#ifndef PB_PREFETCH
#define PB_PREFETCH 0
#endif

// The model's dimensions - attn.cl's, verbatim, because a probe that measures a
// different shape measures nothing.
#define KV_HEADS 4
#define HD 256
#define SCALE 0.0625f
#define WAVE_P 16                 /* positions per wave = subgroups per work-group */
#define WAVES (PB_BLOCK / WAVE_P)
#define SG 16                     /* SIMD16: 16 subgroups of 16 lanes */
#define PER_LANE 16               /* elements of the 256-dim dot per lane */
#define NBLOCKS (PB_MAXLEN / PB_BLOCK)
#define PART 258                  /* {mx, sm, acc[256]} */
#define WG_DEC 256

#if PB_BLOCK % WAVE_P != 0
#error "probe_attn: PB_BLOCK must be a multiple of 16"
#endif
#if PB_VEC != 1 && PB_VEC != 2 && PB_VEC != 4
#error "probe_attn: PB_VEC must be 1, 2 or 4"
#endif
// PB_PREFETCH rewrites the K load itself, so it cannot be combined with the
// three switches that also rewrite it. A build that asks for both would
// silently get one of them, which is exactly the kind of quiet disagreement
// this project puts in binary names to avoid.
#if PB_PREFETCH && (PB_NO_K || PB_HOT_K || PB_VEC != 1)
#error "probe_attn: PB_PREFETCH is exclusive with PB_NO_K, PB_HOT_K and PB_VEC"
#endif
#if PB_NO_K && (PB_VEC != 1 || PB_HOT_K)
#error "probe_attn: PB_NO_K removes the load PB_VEC and PB_HOT_K describe"
#endif
#if PB_NO_V && (PB_HOT_V || PB_VT)
#error "probe_attn: PB_NO_V removes the load PB_HOT_V and PB_VT describe"
#endif
#if PB_SG_TREE && (PB_NO_TREE || PB_NO_BARRIER)
#error "probe_attn: PB_SG_TREE narrows barriers the other two remove"
#endif
#if PB_NO_K && PB_KT
#error "probe_attn: PB_NO_K removes the load PB_KT describes"
#endif
#if PB_PREFETCH && PB_KT
#error "probe_attn: PB_PREFETCH writes its own K address"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// The barrier, switchable. `PB_NO_BARRIER` produces a wrong answer on purpose:
// the question it prices is what six work-group barriers per wave cost, and the
// only way to price them is to run the identical instruction stream without
// them.
#if PB_NO_BARRIER
#define PB_BARRIER() ((void)0)
#else
#define PB_BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#endif

// The transcendental, switchable. `exp` is what ships (never `native_exp` -
// docs/12's rounding discipline forbids it in a production kernel because the
// host reference cannot follow it); PB_EXP 1 and 2 exist to price it, not to
// propose it.
#if PB_EXP == 1
#define PB_EXPF(x) native_exp(x)
#elif PB_EXP == 2
// A stand-in with the same shape (one input, one output, no branch) and none of
// the cost. Not exp, not close to exp, and not meant to be.
#define PB_EXPF(x) ((x) * 0.5f + 1.0f)
#else
#define PB_EXPF(x) exp(x)
#endif

// ---------------------------------------------------------------------------
// probe_attn - grid (KV_HEADS, NBLOCKS), work-group 256, SIMD16.
// The argument list is attn_decode's exactly, so the host can bind the same
// buffers to any variant.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_DEC, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void probe_attn(__global const uint* restrict ctrl,
                         __global const float* restrict attn_q,
                         __global const ushort* restrict kv_k,
                         __global const ushort* restrict kv_v,
                         __global float* restrict attn_part) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;
  __local float qs[HD];
  __local float dot_red[WG_DEC];

  const uint pos = ctrl[0];
  const uint n_act = 1;

  const uint bstart = blk * PB_BLOCK;
  if (bstart >= pos + n_act) return;

#if PB_STAGE == 1
  // The grid floor: every live work-group returns here too, so what is left is
  // 1024 work-groups' dispatch, the early-out test and one store.
  attn_part[(((size_t)j * NBLOCKS + blk)) * PART + lid] = (float)lid;
  return;
#else

  for (uint qhl = 0; qhl < PB_GQA; ++qhl) {
    const uint qh = j * 6u + qhl;
    {
      const uint m = 0;
      PB_BARRIER();
      qs[lid] = attn_q[((size_t)m * 24u + qh) * HD + lid];
      PB_BARRIER();

#if PB_STAGE == 2
      // The staging floor: the q-head loop, two barriers per pass and one
      // 1 KB global read per pass. `qs` is read by the store below, so nothing
      // here is dead.
      attn_part[(((size_t)qh * NBLOCKS + blk)) * PART + 2 + lid] = qs[lid];
#else
      const uint bound = pos + m;
      float mx = -INFINITY, sm = 0.0f, acc = 0.0f;

#if PB_PREFETCH
      // Wave 0's K row, loaded before the loop; inside the loop the NEXT wave's
      // row is loaded before this wave's barriers, so the load latency has the
      // whole tree and the whole online update to hide under.
      float kcur[PER_LANE];
      {
        const uint p0 = bstart + sgid;
        for (uint t = 0; t < PER_LANE; ++t) {
          const uint d = lane + SG * t;
          kcur[t] = p0 <= bound ? bf16f(kv_k[((size_t)p0 * KV_HEADS + j) * HD + d]) : 0.0f;
        }
      }
#endif

      for (uint w = 0; w < WAVES; ++w) {
        const uint p = bstart + w * WAVE_P + sgid;
        float a = 0.0f;

#if PB_PREFETCH
        float knext[PER_LANE];
        {
          const uint pn = bstart + (w + 1u) * WAVE_P + sgid;
          const uint pl = (w + 1u < WAVES && pn <= bound) ? pn : bstart + sgid;
          for (uint t = 0; t < PER_LANE; ++t) {
            const uint d = lane + SG * t;
            knext[t] = bf16f(kv_k[((size_t)pl * KV_HEADS + j) * HD + d]);
          }
        }
        if (p <= bound)
          for (uint t = 0; t < PER_LANE; ++t) a = fma(qs[lane + SG * t], kcur[t], a);
#else
        if (p <= bound) {
          // The K row this subgroup's position reads. PB_HOT_K collapses every
          // position onto block 0, which keeps the message COUNT and the
          // instruction stream identical and makes every one of them an L1 hit.
#if PB_HOT_K
          const uint kp = bstart + sgid;
#else
          const uint kp = p;
#endif
          // PB_KT: the transposed cache for K - see PB_VT below. Subgroups of one
          // wave then read rows 512 B apart instead of 2048 B, so a wave's 8 KB
          // is one contiguous run instead of a 32 KB strided span.
#if PB_KT
          __global const ushort* restrict krow = kv_k + ((size_t)j * PB_MAXLEN + kp) * HD;
#else
          __global const ushort* restrict krow = kv_k + ((size_t)kp * KV_HEADS + j) * HD;
#endif
#if PB_NO_K
          // No global traffic, same 16 fma and same 16 SLM reads of `qs`. The
          // second operand varies with p and d so nothing hoists.
          for (uint t = 0; t < PER_LANE; ++t) {
            const uint d = lane + SG * t;
            a = fma(qs[d], (float)((p + d) & 7u), a);
          }
          (void)krow;
#elif PB_VEC == 2
          __global const ushort2* restrict kv2 = (__global const ushort2*)krow;
          for (uint t = 0; t < PER_LANE / 2; ++t) {
            const ushort2 kk = kv2[lane + SG * t];
            a = fma(qs[2u * (lane + SG * t)], bf16f(kk.s0), a);
            a = fma(qs[2u * (lane + SG * t) + 1u], bf16f(kk.s1), a);
          }
#elif PB_VEC == 4
          __global const ushort4* restrict kv4 = (__global const ushort4*)krow;
          for (uint t = 0; t < PER_LANE / 4; ++t) {
            const ushort4 kk = kv4[lane + SG * t];
            const uint d0 = 4u * (lane + SG * t);
            a = fma(qs[d0], bf16f(kk.s0), a);
            a = fma(qs[d0 + 1u], bf16f(kk.s1), a);
            a = fma(qs[d0 + 2u], bf16f(kk.s2), a);
            a = fma(qs[d0 + 3u], bf16f(kk.s3), a);
          }
#else
          for (uint t = 0; t < PER_LANE; ++t) {
            const uint d = lane + SG * t;                // lanes read 16 consecutive dims
            a = fma(qs[d], bf16f(krow[d]), a);
          }
#endif
        }
#endif  // PB_PREFETCH

        PB_BARRIER();
        dot_red[lid] = a;
        PB_BARRIER();
#if !PB_NO_TREE
        for (uint stride = SG / 2; stride > 0; stride >>= 1) {
          if (lane < stride) dot_red[sgid * SG + lane] += dot_red[sgid * SG + lane + stride];
          // Every read and every write in this loop is inside ONE subgroup, so
          // three of the four steps need only a subgroup fence. The LAST one
          // must stay work-group wide: `dot_red[s*SG]` is then read by all 256
          // work-items across all 16 subgroups. docs/12 lists this as "not
          // used, not measured - worth trying with a probe in hand"; PB_SG_TREE
          // is the probe. The arithmetic and its order do not move at all.
#if PB_SG_TREE
          if (stride > 1) {
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
          } else {
            PB_BARRIER();
          }
#else
          PB_BARRIER();
#endif
        }
#endif

#if PB_NO_SOFTMAX
        // The wave without its online update: the tree's output still feeds the
        // V accumulation (so the SLM reads and the serial fma chain stay), and
        // fmax, exp, the ssum chain and both rescales are gone.
        float tsum = 0.0f;
        for (uint s = 0; s < WAVE_P; ++s) {
          const uint ps = bstart + w * WAVE_P + s;
#if PB_NO_V
          const float vf = (float)((ps + lid) & 7u);
#else
#if PB_HOT_V
          const uint vp = bstart + s;
#else
          const uint vp = ps;
#endif
          // PB_VT: the SAME 16 messages of 32 B over the SAME 8 KB, addressed as
          // if the cache were `[KV_HEADS][MAXLEN][HD]`. Consecutive positions of
          // one kv-head then sit 512 B apart instead of 2048 B. It is the only
          // difference, so what the row buys is the stride and nothing else.
          // (The VALUE read is wrong - this is a probe, and `j*MAXLEN + p` is a
          // legal index into the same allocation, which is the point.)
#if PB_VT
          const size_t vidx = ((size_t)j * PB_MAXLEN + vp) * HD + lid;
#else
          const size_t vidx = ((size_t)vp * KV_HEADS + j) * HD + lid;
#endif
          const float vf = ps <= bound ? bf16f(kv_v[vidx]) : 0.0f;
#endif
          tsum = fma(dot_red[s * SG], vf, tsum);
        }
        sm += 1.0f;
        acc += tsum;
        mx = fmax(mx, dot_red[sgid * SG]);
#else
        float sc[WAVE_P];
        for (uint s = 0; s < WAVE_P; ++s) {
          const uint ps = bstart + w * WAVE_P + s;
          sc[s] = ps <= bound ? dot_red[s * SG] * SCALE : -INFINITY;
        }
#if PB_DEP_BREAK
        // The loop-carried chain, cut: (mx, sm) do not cross the wave boundary
        // and `acc` accumulates instead of being rescaled. Every load, every
        // exp and every fma below is the same count as the base kernel's; what
        // is gone is wave w+1's dependence on wave w's result.
        float nmx = -INFINITY;
#else
        float nmx = mx;
#endif
        for (uint s = 0; s < WAVE_P; ++s) nmx = fmax(nmx, sc[s]);
        float resc;
        if (nmx > -INFINITY) {
          resc = PB_EXPF(mx - nmx);
          float ssum = 0.0f;
          for (uint s = 0; s < WAVE_P; ++s) {
            sc[s] = PB_EXPF(sc[s] - nmx);
            ssum += sc[s];
          }
#if PB_DEP_BREAK
          sm += ssum;
#else
          sm = fma(sm, resc, ssum);
          mx = nmx;
#endif
        } else {
          resc = 1.0f;
          for (uint s = 0; s < WAVE_P; ++s) sc[s] = 0.0f;
        }

        float tsum = 0.0f;
        for (uint s = 0; s < WAVE_P; ++s) {
          const uint ps = bstart + w * WAVE_P + s;
#if PB_NO_V
          const float vf = (float)((ps + lid) & 7u);
#else
#if PB_HOT_V
          const uint vp = bstart + s;
#else
          const uint vp = ps;
#endif
          // PB_VT: the SAME 16 messages of 32 B over the SAME 8 KB, addressed as
          // if the cache were `[KV_HEADS][MAXLEN][HD]`. Consecutive positions of
          // one kv-head then sit 512 B apart instead of 2048 B. It is the only
          // difference, so what the row buys is the stride and nothing else.
          // (The VALUE read is wrong - this is a probe, and `j*MAXLEN + p` is a
          // legal index into the same allocation, which is the point.)
#if PB_VT
          const size_t vidx = ((size_t)j * PB_MAXLEN + vp) * HD + lid;
#else
          const size_t vidx = ((size_t)vp * KV_HEADS + j) * HD + lid;
#endif
          const float vf = ps <= bound ? bf16f(kv_v[vidx]) : 0.0f;
#endif
          tsum = fma(sc[s], vf, tsum);
        }
#if PB_DEP_BREAK
        acc += tsum;
#else
        acc = fma(acc, resc, tsum);
#endif
#endif  // PB_NO_SOFTMAX

#if PB_PREFETCH
        for (uint t = 0; t < PER_LANE; ++t) kcur[t] = knext[t];
#endif
      }

      __global float* restrict out = attn_part + (((size_t)qh * NBLOCKS + blk)) * PART;
#if PB_NO_PART
      // The stores, predicated off by a condition no compiler can fold: every
      // instruction above still runs and still feeds `acc`, and the 1.6 MB of
      // `attn_part` traffic is gone. What the difference prices is the write.
      if (acc == 1.0e30f) {
        out[2 + lid] = acc;
        if (lid == 0) {
          out[0] = mx;
          out[1] = sm;
        }
      }
#else
      out[2 + lid] = acc;
      if (lid == 0) {
        out[0] = mx;
        out[1] = sm;
      }
#endif
#endif  // PB_STAGE == 2
    }
  }
#endif  // PB_STAGE == 1
}

# Spec 6a - attention baselines and the flash-attention tile probe, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure today's attention costs (spec 6 P0) and find the tiling of a fused bf16 flash-attention kernel that is correct to cosine 0.99999 and fastest on real shapes (spec 6 P1), so plan 6b can write the production kernel from a measured winner.

**Architecture:** A probe, `tools/probe/probe_flash_attn.{cl,cc}`, built like `probe_w8a8`. The kernel is one parameterised OpenCL source compiled per arm. It reads Q from a `pf_q`-shaped buffer and K/V straight from a KV-cache-shaped buffer, keeps O in registers with an fp32 online softmax, and writes `pf_o`'s layout. The host checks every arm against an fp64 CPU reference and times it against the production composed path (`runtime::prefill::attn_chunk` on L0) as interleaved pairs.

**Tech Stack:** OpenCL C (ocloc AOT, bmg-g31, 256 GRF), Level Zero, C++20, CMake; the box via `tools/box.sh`.

**Spec:** `docs/superpowers/specs/2026-09-25-spec6-flash-attention-128k-design.md` (§5 P0, §6 P1).

## Global Constraints

- Math: S = q k^T in fp32 (bf16 DPAS `intel_sub_group_bf16_bf16_matrix_mad_k16`), times `ATTN_SCALE` = 1/16 on the fp32 score; causal mask, key position <= `pos` + query row; fp32 online softmax (running max m, running sum l, O rescaled by `exp(m_old - m_new)`); P = `exp(s - m)` rounded to bf16 as the PV operand; O / l at the end. `exp`, never `native_exp`.
- Layouts (from `src/runtime/prefill/attn.cc`): Q `pf_q` bf16 `[C][24][256]`, q-head h = 6 j + l for kv head j; KV cache bf16 `[pos][4][256]`; output **`pf_o` fp32 `[24][rows][256]` with per-head stride `rows * 256`, `rows = pad256(C)`**, the buffer `pf_attn_gate` reads. This corrects spec 6 §3.1, which named `pf_attn`; 6b amends the spec.
- Correctness bar per arm: per-(row, head) cosine **>= 0.99999** against an fp64 CPU reference, on sampled rows that include row 0, the last row, and rows on a KV-tile diagonal.
- Timing: interleaved control/candidate pairs after a warm-up, median paired ratio with its range (docs/probe-w4a8-2026-09-23.md §14.4). Device 0, idle box (no render-node holder, no container).
- Probe-only: no production file changes.

## Review Focus

- **Rows past C in the last tile.** `rows = pad256(C)`, and a 300-row chunk has 212 padded rows. The kernel must write finite values there or not at all, and never read past the Q buffer. Pinned in Task 3 by running C = 300 and checking rows 0..299 against the reference and 300..511 for finiteness.
- **Depth not a multiple of the KV tile.** The last KV tile is partial, `depth = pos + C` arbitrary. The mask must cover columns >= depth, not only the causal bound. Pinned in Task 3 with pos = 777, C = 300 (depth 1077).
- **The first row of a chunk at depth.** Row 0 at `pos` attends to pos + 1 keys, across many tiles, and the tiles it skips must not change m. Pinned by always sampling row 0 at pos 16384.
- **Large score magnitudes.** A running max that starts at -INF must not produce NaN from `-INF - -INF`. Pinned in Task 3 by one case whose Q rows are scaled by 30 (scores in the hundreds).
- **Heads split across work-groups** (HPW < 6). Each work-group must address its own head slice of Q and O. Pinned in Task 4: every arm passes the same correctness check.

---

### Task 1: P0, the baselines

**Files:**
- Create: `tools/probe/attn_baseline.sh`
- Create: `docs/probe-flash-attn-2026-09-25.md` (the record, first section)

- [x] **Step 1: The script**

`tools/probe/attn_baseline.sh`, run from the Mac. It uses `tools/bench_decode.sh` as BENCHMARKS.md does, with `B70_PREFILL_PROFILE=1` for the phase rows, which is diagnostic only: that run adds waits, so its total is not a bench row.

```bash
#!/usr/bin/env bash
# Spec 6 P0: today's attention costs. Diagnostic rows (profiled runs add waits);
# the unprofiled rows are the bench numbers.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ZE_AFFINITY_MASK=0
echo "## pp4096, profiled (attention phases: attn_qk, attn_sm, attn_pv)"
B70_PREFILL_PROFILE=1 tools/bench_decode.sh --pp 4096 --runs 1 --no-build 2>&1 | \
  grep -E "attn_(qk|sm|pv)|pp \|" || true
for pp in 4096 8192 16384; do
  echo "## pp$pp, unprofiled, median of 3"
  tools/bench_decode.sh --pp "$pp" --runs 3 --no-build 2>&1 | grep -E " pp \|"
done
for d in 4096 16000; do
  echo "## decode at depth $d, max_len 16384, median of 3"
  tools/bench_decode.sh --depth "$d" --runs 3 --no-build 2>&1 | grep -E "tg|\| *[0-9]"
done
```

If `bench_decode.sh` names its flags differently (read its usage block first), adjust the script to the real ones and note it in the record. The profiled phase names must match `src/runtime/prefill/profile.cc`.

- [x] **Step 2: Run it on an idle box and record**

Run: `tools/probe/attn_baseline.sh | tee /tmp/p0.txt` (after the DRM-holder check: no process may hold `/dev/dri/renderD*`).

In `docs/probe-flash-attn-2026-09-25.md`, write section "1. P0, the baselines" with a table:
- the attention phase ms (qk + sm + pv) of the profiled pp4096 run;
- pp4096, pp8192 and pp16384 t/s;
- the per-chunk increment (derived: the pp16384 total minus the pp8192 total, over 4 chunks);
- decode t/s at depths 4096 and 16000.

Label every number measured or derived, as the repo's records do.

- [x] **Step 3: Commit**

```bash
git add tools/probe/attn_baseline.sh docs/probe-flash-attn-2026-09-25.md
git commit -m "probe: attention baselines before flash attention (spec 6 P0)"
```

---

### Task 2: the probe harness, the CPU reference, and the composed path as control

**Files:**
- Create: `tools/probe/probe_flash_attn.cc`
- Modify: `tools/probe/CMakeLists.txt` (append a stanza modelled on `probe_w8a8`'s, linking `b70_prefill_host b70_runtime b70_loader b70_l0 b70_model` and `b70_link_prefill`)

**Interfaces:**
- Produces: `probe_flash_attn <pos> <C> [--arms all|<name>] [--qscale S]`. It builds random Q (bf16, N(0, 1) times qscale) and a K/V cache of depth `pos + C` (bf16 N(0, 1)) in the production layouts. It runs the composed path through `runtime::prefill::attn_chunk` (backend L0) into a `PrefillScratch` with `max_len = pad256(pos + C)` rounded up to a multiple of 256. It computes the fp64 reference on sampled (row, head) pairs, prints each path's worst cosine and max abs error, and, from Task 4, the paired timings.

- [x] **Step 1: The reference and the composed control, failing on nothing yet**

In `probe_flash_attn.cc`:
- `ref_row(head h, row m)`: fp64. s_n = dot(q[m][h], k[n][h/6]) / 16 for n <= pos + m; softmax; o = sum_n p_n v[n][h/6]; returns 256 doubles.
- Sampled rows: {0, 1, 7, 8, 63, 64, C/2, C-2, C-1} plus 16 rows drawn from `std::mt19937(pos + C)`, clipped to [0, C), for every one of the 24 heads.
- The composed run: fill `pf_q` rows 0..C-1 (and rows C..pad256(C)-1 with 0) and the cache, call `attn_chunk(cx, kc, s, pos, C, pf_q, kv_k, kv_v, PrefillBackend::L0)`, `cx.wait()`, download `pf_o`.
- Cosine and max abs error per sampled pair against the reference, reported as `composed: worst cos X at (h, m), max abs Y`.

- [x] **Step 2: Run the control**

Run: `tools/box.sh run 'ZE_AFFINITY_MASK=0 ./build/tools/probe/probe_flash_attn 16384 2048'` and `... 777 300` and `... 0 2048 --qscale 30`.
Expected: the composed path prints a worst cosine. Record the three values in the probe doc, section "2. The composed path against fp64". **They calibrate the bar.** If the composed path itself is below 0.99999 anywhere, write down its value, and the flash bar becomes "no worse than the composed path's worst". Record this change in the doc before Task 3.

- [x] **Step 3: Commit**

```bash
git add tools/probe/probe_flash_attn.cc tools/probe/CMakeLists.txt docs/probe-flash-attn-2026-09-25.md
git commit -m "probe: flash-attention harness, fp64 reference, composed path as control (spec 6 P1)"
```

---

### Task 3: the flash kernel, one arm, correct

**Files:**
- Create: `tools/probe/probe_flash_attn.cl`
- Modify: `tools/probe/probe_flash_attn.cc`, `tools/probe/CMakeLists.txt` (the `add_ocloc_kernel` rows)

**Interfaces:**
- Produces: kernel `pfa(const ushort* Q, const ushort* Kc, const ushort* Vc, float* O, uint pos, uint C, uint rows)`, grid (ceil(C / RPW), 4, 6 / HPW), WG `16 * HPW * RPW / 8`. Defines: `KT` (KV positions per tile, 32 or 64), `RPW` (query rows per work-group per head, a multiple of 8), `HPW` (q-heads per work-group: 6, 3 or 1), `QREG` (1 = Q kept in registers for the whole loop, 0 = re-read per KV tile). Variant name `pfa_KT<KT>_R<RPW>_H<HPW>_Q<QREG>`.

- [x] **Step 1: The failing case**

In the harness, add the `pfa` run for the first arm `pfa_KT32_R16_H6_Q1`: same inputs, output into a fresh `[24][pad256(C)][256]` fp32 buffer, the same sampled comparison, printed as `pfa_KT32_R16_H6_Q1: worst cos ...`, and a finiteness scan of rows [C, pad256(C)). The harness exits non-zero if the worst cosine is under the Task 2 bar.

Run it before the kernel exists. Expected: FAIL at load, the variant binary missing.

- [x] **Step 2: The kernel**

`tools/probe/probe_flash_attn.cl`. Each sub-group owns **8 query rows of one head**. Its O accumulator is 16 `float8` (8 rows x 256 dims, 128 GRF). The QK^T accumulator for one KV tile is `KT / 16` `float8`s, laid out with row in the component and key position in the lane. That is exactly the bf16 DPAS A-operand layout (pf_gemm.cl: `af[a]` component r = row, lane = k), so P needs no data movement to become the PV operand. The K and V loads are pf_gemm.cl's two verified idioms:
- **K^T, from the `TRANSB` branch:** `intel_sub_group_2d_block_read_transpose_32b_16r8x1c` over the cache viewed as `[pos][512 dwords]`, giving lane = key position and 8 dwords = 16 dims.
- **V, from the `!TRANSB` branch:** `intel_sub_group_2d_block_read_transform_16b_32r16x2c` over the cache `[pos][1024 bf16]`, giving VNNI pairs of positions, lane = dim.

```c
// probe_flash_attn - spec 6 P1: fused bf16 flash attention, one arm per -D set.
// Contract and layouts: docs/superpowers/plans/2026-09-25-spec6a-flash-attn-baseline-and-probe.md.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#define SG 16
#define HD 256
#define NDA (HD / 16)                 /* 16 dim-atoms of O */
#define NKA (KT / 16)                 /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)           /* sub-groups per work-group */
#define ATTN_SCALE (1.0f / 16.0f)
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  return (ushort)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
}

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pfa(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                  // kv head
  const uint h = j * 6u + get_group_id(2) * HPW + s / (RPW / 8u);  // q head
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));   // first of 8 rows
  if (r0 >= C) return;                                             // no barriers below
  const uint depth = pos + C;
  const int q_w = 24 * HD * 2, q_h = (int)C, q_p = 24 * HD * 2;
  const int kt_w = 4 * HD * 2, kt_h = (int)depth, kt_p = 4 * HD * 2;   // K as dwords: 512 wide
  const int v_w = 4 * HD * 2, v_h = (int)depth, v_p = 4 * HD * 2;

  short8 qa[NDA];                      // Q, 8 rows x 256 dims, as 16 A operands
#define LOAD_Q()                                                                      \
  for (uint kk = 0; kk < NDA; kk += 2) {                                              \
    ushort t[16];                                                                     \
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,       \
                                              (int2)((int)(h * HD + 16u * kk), (int)r0), t); \
    qa[kk] = as_short8(vload8(0, t));                                                 \
    qa[kk + 1] = as_short8(vload8(0, t + 8));                                         \
  }
#if QREG
  LOAD_Q();
#endif
  float8 o[NDA];
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
  const uint last = min(depth, pos + r0 + 8u);          // keys this SG's rows can see
  for (uint t0 = 0; t0 < last; t0 += KT) {
#if !QREG
    LOAD_Q();
#endif
    float8 sacc[NKA];
    for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
    for (uint kk = 0; kk < NDA; ++kk)
      for (uint b = 0; b < NKA; ++b) {
        uint kb[8];
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kc, kt_w, kt_h, kt_p,
            (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);
      }
    // scale, mask, running max per row (component r = row r0 + r; lane = key t0 + 16 b + l)
    float8 tmax = (float8)(-INFINITY);
    for (uint b = 0; b < NKA; ++b) {
      const uint key = t0 + 16u * b + l;
#define MASK(r) sacc[b].s##r = (key <= pos + r0 + r && key < depth) ? sacc[b].s##r * ATTN_SCALE : -INFINITY;
      MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
      tmax = fmax(tmax, sacc[b]);
    }
    float8 mnew;
#define RED(r) mnew.s##r = fmax(m.s##r, sub_group_reduce_max(tmax.s##r));
    RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
    const float8 corr = exp(m - mnew);          // m = -INF on the first tile: exp(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - mnew);     // masked: exp(-INF) = 0
      psum += p;
#define CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));
      CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
    }
    float8 rsum;
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
    SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
    lsum = lsum * corr + rsum;
    m = mnew;
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    // O += P V: A = pa[b] (8 rows x 16 keys), B = V (16 keys x 16 dims, VNNI)
    for (uint b = 0; b < NKA; b += 2)
      for (uint dd = 0; dd < NDA; dd += 2) {
        uint vb[32];   // 32 keys x 32 dims: [c*16 + jj], c = dim half, jj = key pair
        intel_sub_group_2d_block_read_transform_16b_32r16x2c(
            (__global void*)Vc, v_w, v_h, v_p, (int2)((int)(j * HD + 16u * dd), (int)(t0 + 16u * b)), vb);
        for (uint c = 0; c < 2u; ++c)
          for (uint ks = 0; ks < 2u; ++ks)
            o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);
      }
  }
  // O / l, into pf_o's layout [24][rows][256] fp32
  const float8 inv = 1.0f / lsum;
  __global float* Oh = O + (size_t)h * rows * HD;
  for (uint dd = 0; dd < NDA; ++dd) {
    float8 w = o[dd] * inv;
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&w);
  }
}
```

Three details that are **not negotiable**, because the correctness bar depends on them:
- `exp`, not `native_exp`.
- The mask covers both `key <= pos + row` and `key < depth`.
- Tiles are walked from key 0 up to this sub-group's causal limit, `last`.

If the compiler refuses an intrinsic spelling, use the one pf_gemm.cl uses for the same operation and note it. With `NKA = KT / 16` odd, the PV loop's `b += 2` must be guarded, so the arms use KT 32 or 64 only.

`tools/probe/CMakeLists.txt`:

```cmake
set(PFA_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_flash_attn.cl)
foreach(arm "32 16 6 1" "32 32 6 1" "64 16 6 1" "32 16 3 1" "32 16 1 1" "32 16 6 0" "64 16 6 0" "32 32 3 1")
  separate_arguments(f UNIX_COMMAND "${arm}")
  list(GET f 0 KT)
  list(GET f 1 R)
  list(GET f 2 H)
  list(GET f 3 QR)
  add_ocloc_kernel(pfa_KT${KT}_R${R}_H${H}_Q${QR} SOURCE ${PFA_CL}
                   DEFINES KT=${KT} RPW=${R} HPW=${H} QREG=${QR} OPTIONS -cl-intel-256-GRF-per-thread)
endforeach()
```

(`R32 H6` would be 24 sub-groups, WG 384, which is allowed. No arm exceeds 32 sub-groups.)

- [x] **Step 3: Run it to pass**

Run the four Review Focus cases on `pfa_KT32_R16_H6_Q1`: `16384 2048`, `777 300`, `0 2048 --qscale 30`, `0 64`.
Expected: the worst cosine is at or above the bar in all four, rows [C, pad256(C)) are finite, and the build log shows the arm's spill line (record it: a spill is a timing fact, not a correctness failure).

- [x] **Step 4: Commit**

```bash
git add tools/probe/probe_flash_attn.cl tools/probe/probe_flash_attn.cc tools/probe/CMakeLists.txt
git commit -m "probe: pfa, a fused bf16 flash-attention kernel, correct against fp64 (spec 6 P1)"
```

---

### Task 4: the sweep, and the winner

**Files:**
- Modify: `tools/probe/probe_flash_attn.cc` (`--arms all`, the timing), `docs/probe-flash-attn-2026-09-25.md`

- [x] **Step 1: All arms correct**

`--arms all` runs every built arm through the Task 3 check. Any arm under the bar is reported and excluded from timing.

- [x] **Step 2: Timing**

For each surviving arm, in interleaved rounds (control = composed `attn_chunk` on L0; candidate = the arm), 11 rounds after a 20-iteration warm-up, L0 event timestamps summed over the arm's launches, median paired ratio and range. Shapes:
- **pos 2048, C 2048:** the second chunk of pp4096, where F1 lives.
- **pos 16384, C 2048.**
- **pos 30720, C 2048:** depth 32768, the deepest the composed control fits comfortably.

Also report each arm's achieved TFLOP/s on the 30720 shape, from its causal FLOP count 4 x 24 x 256 x sum_m(pos + m + 1) (derived), against the bf16 DPAS peak 183.45.

- [x] **Step 3: Record and pick**

In the probe doc, section "3. The sweep": every arm's spill line, worst cosine, the three paired ratios, and TFLOP/s at depth. The **winner** is the fastest arm at pos 30720 among those at or above the bar, provided it is not slower than the composed control at pos 2048. If no arm beats the control at pos 2048, say so: that is spec 6's F1 in doubt, and 6b starts from that finding.

State the winner in one line at the top of the doc:

`Winner: pfa_KT<..>_R<..>_H<..>_Q<..>, <ratio>x the composed path at depth 32k, <TFLOP/s> TFLOP/s.`

If **no** arm passes the correctness bar, the spec's stopping rule applies: record the worst cosines and stop.

- [x] **Step 4: Commit**

```bash
git add tools/probe/probe_flash_attn.cc docs/probe-flash-attn-2026-09-25.md
git commit -m "probe: flash-attention tile sweep, winner <arm> (spec 6 P1)"
```

(Replace `<arm>` with the winner's name.)

---

**Done when:** the probe doc records P0, the composed path's calibration, the sweep and one winner, or a stopping-rule finding. Plan 6b (the production kernel, integration, 128k and the long-context gates) is written from that record.

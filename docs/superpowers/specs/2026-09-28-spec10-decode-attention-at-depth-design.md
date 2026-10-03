# Spec 10 - decode attention at depth

**Status:** design, 2026-09-28, for operator review.

**Order:** after spec 8 (MTP) and spec 9 (`lm_head`) in the operator's queue.
It must keep the `M` loop that spec 8's verify step uses (M = 1..4).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

Long agentic sessions decode at depth. Spec 6 F3 missed (spec 6 §8):

| depth | decode t/s | bandwidth-derived t/s | share |
|---:|---:|---:|---:|
| 32768 | 20.60 | 26.17 | 78.7 % |
| 65536 | 15.46 | 23.34 | 66.2 % |
| 130816 | 10.19 | 19.20 | 53.1 % |

**Where the time goes (derived).** One token at depth d reads the weights
(15.540 GB) plus the KV of 16 FA layers: 16 x d x 4 kv-heads x 256 x 2 B x 2
(K, V) = 64 KiB per position, **8.57 GB at 130816**. A step at 4k depth takes
33.9 ms (plan 8a, M = 1). At 130816, 1 / 10.19 t/s = 98.1 ms, so attention
costs about **64 ms for 8.57 GB: ~134 GB/s, 22 % of the 608 GB/s device**,
where the weight GEMVs run at ~590 GB/s. Decode attention is the one part of
the step that does not stream at device bandwidth, and at 128k it is two
thirds of the step.

**Today's kernels** (phase 0 §9.5, `src/runtime/buffers.h`):
- `attn_decode`: one work-group per (kv head, block of `kAttnBlock` = 64
  positions), 6 q-heads x M queries each, online softmax in fp32, K/V streamed
  through SLM in 16-row slabs; writes `(max, sum, acc[256])` per (q head, block,
  m) to `attn_part [24][max_len / 64][M][258]` fp32.
- `attn_reduce`: one work-group per (q head, m), combines the
  `ceil(seq_len / 64)` partials, applies the gate.

`kAttnBlock` 64 was tuned at 16k (`buffers.h` notes: -1.127 / -1.092 / -0.070
ms per halving from 256, with `attn_reduce` on a steep ramp, 80 -> 450 us). At
130816 there are 2044 blocks per head group, so the partial table is written
and read in full every layer: 24 x 2044 x 258 x 4 B = **50.6 MB per layer,
0.81 GB per token written and again read** (derived, M = 1), about 19 % on
top of the KV bytes, and `attn_reduce`'s serial walk over 2044 partials per
head.

## 2. The decision (proposed)

**A new decode-attention kernel pair, our own L0/OpenCL C, tuned for depth**,
beside the current one and selectable (`B70_DECODE_ATTN=v1|v2`) until its
gates pass; the current kernels stay as the reference. Levers, to be measured
one at a time in P1:

1. **Many positions per work-group** (a loop over 256 to 2048 positions inside
   the work-group, online softmax carried across): the partial table shrinks
   4-32x and `attn_reduce` walks 4-32x fewer entries. The grid stays large
   enough for occupancy only if the positions per work-group scale with depth:
   `positions_per_wg = max(64, ceil(seq_len / target_wgs))`, computed on the
   device from `pos` in `attn_prep` (the captured list is fixed; the kernel
   reads its stride from `Control`).
2. **Wide, contiguous loads.** K and V are `[pos][4][256]` bf16 per layer: one
   kv head's 512 B per position is strided by 2 KiB. Load 2D blocks (the
   `intel_sub_group_2d_block_read` family `pf_flash_attn` uses) or
   `vload16` rows straight into registers, skip the SLM staging for M <= 4
   (each K/V row feeds only 6 x M dot products), and prefetch the next block.
3. **Fused reduce.** The last work-group of each (kv head) to finish (a
   device-side counter, ordered so the result is deterministic) combines the
   partials, removing `attn_reduce`'s launch; or keep the launch and make it a
   parallel tree over the (now few) partials.
4. **exp2 with the scale folded** (spec 6c's lever, operator-approved there),
   if the kernel turns out instruction-bound after 1-3.

## 3. Design constraints

- **Bitwise determinism** (replay determinism test): no atomics whose order
  changes a sum; the fused reduce, if chosen, combines in a fixed order.
- **M = 1..4** (spec 8's verify at M = K + 1): every variant compiles at those
  M, and M2 of spec 8 (verify rows against M = 1) passes with the new kernels.
- **max_len variants** L16384, L32768 and L131072 as today; the shallow-depth
  step (4k at max_len 131072, spec 6 F4) must not regress.
- The KV layout does not change (spec 7's snapshots and blocks depend on it).

## 4. Correctness gates

- **A1, kernel.** Against the current kernels on real captured q and KV at
  depths 4k, 32k, 128k, M = 1..4: per (q head, m) output cosine >= 0.99999, max
  abs error recorded.
- **A2.** Golden gates, determinism and replay bitwise, `flash_long_test`,
  passkey 3/3 at 5 / 50 / 95 % of 120k on `l0-int8`, spec 7's snapshot tests,
  spec 8's M2 (when merged).
- **A3.** Greedy decode of 256 tokens at 32k from the committed long prompt:
  identical to v1 except at near-ties under the golden rule.

## 5. Speed bars

Idle box, device 0, interleaved pairs against v1, median of 3.

- **P0 (first).** Per-kernel time of `attn_prep`, `attn_decode`, `attn_reduce`
  at depths 4k, 32k, 64k, 128k (profile capture), effective GB/s of
  `attn_decode` against the KV bytes, and the partial-table traffic.
- **A-F3.** Spec 6 F3 met: decode >= 90 % of the bandwidth-derived rate, i.e.
  **>= 23.6 / 21.0 / 17.3 t/s at 32768 / 65536 / 130816** (derived from the
  table in §1).
- **A-F4.** 4k depth within 1 % of v1 at max_len 131072 and 16384.
- **Recorded:** M = 2..4 verify-step times at depth (spec 8's cost model).

## 6. Stages

- **10a, probe:** P0; a probe kernel in `tools/probe/` sweeping lever 1
  (positions per work-group) and lever 2 (load path), with A1 per arm, at 32k
  and 128k. Winner and the reduce choice (lever 3).
- **10b, build:** the v2 kernels, the `Control` stride, the selector, A1-A3,
  A-F3, A-F4, the record; v2 becomes the default when every gate passes.

## 7. Out of scope

- An int8 (or fp8-emulated) KV cache: it halves the KV bytes at depth but
  needs its own accuracy study; a separate spec after this one, measured
  against v2.
- Prefill attention (spec 6's flash kernel and its research levers 3-5).
- Changing the KV layout.

## 8. Amendment - 2026-09-28, 10b results (plan `2026-09-28-spec10b-decode-attn-build.md`)

**Operator rulings (2026-09-28).**
1. **exp2 is approved** for decode attention (the E1 of 10a's winner, ~10 % over plain
   `exp`), subject to the plan's gates. It failed the golden gate and does not ship (below).
2. **The stride must not depend on `n_active`** (10a's open point), otherwise spec 8's M2
   (verify rows bitwise equal to M = 1 decode) breaks; M2 must pass with v2 at every K.

**How ruling 2 was met - a per-row stride.** A launch-wide stride that is a function of
`pos` alone is still not enough: the verify at `pos` and the plain step at `pos + m` see
different `pos`, and any stride that moves with depth moves between them at some `pos`.
So v2's stride is per ROW, a function of that row's own key count `L = pos + m + 1`:
`ppw(L) = max(64, roundup64(ceil(L / 32)))`. Row m of any launch then walks exactly the
blocks, waves and orders of the M = 1 step at `pos + m`; positions past a row's bound
(the longer walk of the verify's other rows) score `-INF`, weight exactly 0, and are
neutral. `ppw` moves every 2048 keys, so the at most 4 rows of a launch have at most two
strides; a work-group runs the two row groups one after the other (only for 3 positions
in 2048). **Measured: M2a logits and GDN slots bitwise at k = 1, 2, 3 on both prompt
lengths (n = 1998 crosses the 2048 step), M2b 0 differing ids, RF3-RF5 pass**
(`mtp_verify_test` with v2); `attn_v2_test` checks the same property directly (every M =
2..4 row bitwise the M = 1 row) at 27 synthetic positions and 4 real depths.

**Other departures from the plan (rulings, 10b).**
- No `Control` word: `attn_decode_v2` and `attn_reduce_v2` derive the stride from
  `Control::pos` themselves (the probe's `pad[0]` is now spec 8's `gdn_live`). `attn_prep`
  and `attn.cl` are untouched; nothing is cached, so spec 7's restores need nothing.
- **Grid (4, 32) and partials `[24][32][M][258]`**, not `MAXLEN / 64` blocks: no row ever
  needs more than 32 work-groups, so v2 has no MAXLEN and one binary per M
  (`attn_v2_M<M>_T32`) serves every max_len; the V 2D block read takes its surface height
  from the last written position. `attn_part` keeps v1's allocation (v1 stays selectable;
  v2's partials fit inside it at every max_len >= 2048).
- The selector is `B70_DECODE_ATTN=v1|v2`, read at each capture (decode, verify and draft
  lists alike); the CLI's engine line names it.

**exp2 failed the golden gate, so v2 ships with `exp` (ruling 1 applied "subject to the
gates").** With exp2 (10a's `..._E1`) the golden gate fails on `cjk` row 9, a determined
row: engine 105874 at 13.99905 against 95895 at 13.96513, the golden's 95895 13.9375 /
105874 13.875 (one golden bf16 step apart); v1 and v2-with-`exp` both pass the same gate
in the same session. v2 therefore ships as `P0_L2_D1_Q1` (attn.cl's `exp` and 1/16
scale); the exp2 form stays buildable (`-DB70_ATTN_V2_EXP2=ON`) for a later look. The
cost: 10a measured exp2 at +10 % of kernel time at M = 1; in the step it is 1.3 / 3.1 /
4.5 % at 32768 / 65536 / 130816 (28.21 / 25.27 / 21.04 t/s with exp2, against the table
below, same method, earlier session). A-F3 is met either way.

**Gates (v2 as shipped, `exp`).**

| gate | bar | result |
|---|---|---|
| A1 | cosine >= 0.99999 per (q head, m) vs v1 on real q/KV, 4k/32k/128k, M = 1..4; repeatable | 4096 / 32768 / 65536 / 130816: worst 0.999999998 / 0.999999988 / 0.999999491 / 0.999999761, max abs <= 0.0156 (1 bf16 ulp); bitwise repeatable; every M = 2..4 row bitwise = the M = 1 step; 27 synthetic positions (depth 0, block edges, the 2048 stride step, the cache end) likewise; **pass** |
| A2 | golden gates, determinism, replay, flash_long_test, passkey 3/3, snapshots, spec 8 M2 | full suite with v2 the default: 104 passed, 0 failed (`mtp_head_test` skips as on main); passkey `l0-int8` 3/3 at 5 / 50 / 95 % of 119939 ids; M2 bitwise at k = 1..3; **pass** |
| A3 | 256 greedy ids at 32k identical to v1 except at near-ties | identical for 133 ids, then one near-tie: at decode step 132 (pos 32900) v1's own state gives 28258 20.12565 vs 5639 20.04437 (margin 0.081, under one bf16 ulp at 20, 0.125) and v2 on the same state 5639 20.02276 vs 28258 20.01396 (margin 0.009); `probe_decode_attn fork`. **Pass as a near-tie** (the exp2 build happened to match all 256) |
| A-F3 | >= 23.6 / 21.0 / 17.3 t/s at 32768 / 65536 / 130816 | **27.85 / 24.51 / 20.13** (v1 20.65 / 15.42 / 10.17); **pass** |
| A-F4 | 4k within 1 % of v1 at max_len 131072 and 16384 | 31.41 vs 29.66 t/s (131072), 31.47 vs 29.67 (16384): v2 6 % faster; **pass** |
| recorded | M = 2..4 verify-step times | max_len 16384 at depth 4096 / 12288: v2 1.09-1.11x / 1.23-1.28x faster than v1 (BENCHMARKS) |

**v2 is the default** (`kDefaultDecodeAttn`); v1 stays selectable and bitwise unchanged.
Numbers and conditions: docs/BENCHMARKS.md "Decode attention at depth (spec 10)".

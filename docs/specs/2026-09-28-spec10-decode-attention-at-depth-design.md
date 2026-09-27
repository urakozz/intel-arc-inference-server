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

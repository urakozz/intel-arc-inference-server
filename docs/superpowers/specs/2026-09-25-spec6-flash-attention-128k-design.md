# Spec 6 - fused flash attention in prefill, and 128k context

**Status:** design, 2026-09-25; implemented 2026-09-26 (plans 6a, 6b). Results and
the operator's rulings are in §8.

**Order, set by the operator:** spec 5 (int8 prefill linears) is the
foundation, done 2026-09-24. This spec is the first feature on top of it.
Prefix caching and MTP come after.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

The operator's workload is long agentic coding sessions (memory:
target-workload-agentic-coding). Two things stop the engine running them past
16k tokens:

1. **Prefill attention materialises its scores.** The composed path (ruling
   A14: QK^T GEMM, then `pf_softmax_causal`, then the PV GEMM) writes S (fp32)
   and P (bf16) as `[6][kC][max_len]` buffers, `pf_s` and `pf_p`. At
   max_len 131072 that is 6 x 2048 x 131072 x 6 B, **about 9.7 GB** (derived),
   which does not fit beside the model in 32 GB.
2. **Decode attention is built for one max_len.** `attn_decode` / `attn_reduce`
   bake `MAXLEN` as a stride. Only L16384 exists for M = 1, and the captured
   list's grid is `max_len / 64` blocks whatever the depth.

The target is **128k context** (operator, 2026-09-25): KV cache bf16,
16 FA layers x 4 kv heads x 256 x 2 (K, V) x 2 B = 64 KiB per position, so
**8.4 GB** at 131072 (derived). With ~18 GB of loaded model (the loader's
18.087 GB report) that leaves ~5 GB for prefill scratch, decode state and slack
once `pf_s` / `pf_p` are gone.

## 2. The decision

**A fused flash-attention kernel of our own, on the Level Zero list, bf16 DPAS
with fp32 accumulation and fp32 online softmax** (operator: approach 1, "fused
one of course"). Rejected in the design discussion:

- **The composed GEMMs tiled over KV positions:** bounded memory, but S and P
  still go through memory every tile, so speed stays where it is.
- **sycl-tla's `XeFMHAFwdKernel`:** it is already in the build tree, but it
  brings back the SYCL-L0 boundary (host waits) that spec 2.1 removed, and its
  build log shows ~500-register spills.

Attention math stays bf16. int8 QK^T or PV is out of scope, because attention
scores are the most precision-sensitive tensor in the model and spec 5's
evidence covers the linears only.

## 3. Design

### 3.1 `pf_flash_attn`

- **Inputs:** `pf_q` bf16 `[C][24][256]` (RoPE'd, from `pf_attn_prep_q16`); the
  layer's KV cache bf16 `[pos][4][256]`, into which the chunk's own K and V are
  already written (as today). **Output:** `pf_attn` bf16 `[C][24][256]`, before
  `pf_attn_gate`. Everything before and after the kernel is unchanged.
- **Math, per query row and head:** stream KV tiles; S = q k^T in fp32, then
  x `ATTN_SCALE` (1/16 = 1/sqrt(256), `pf_attn.cl`'s constant, applied to the
  fp32 scores as `pf_softmax_causal` does); causal mask
  (position <= base + row); running max m, running sum l; O is rescaled by
  exp(m_old - m_new) and accumulates P V with P = exp(s - m) rounded to bf16
  as the DPAS operand. At the end O / l, then bf16. exp is `exp`, not
  `native_exp`, matching `pf_softmax_causal`. **One rounding point differs by
  construction:** the composed path rounds the *normalised* P (already divided
  by the row sum) to bf16. Flash rounds the *unnormalised* exp(s - m) and
  divides by l at the end. Together with the summation order, this is why K1
  is a cosine bar and not bit-exactness.
- **Tiling:** one work-group per (kv head, query tile). The six q-heads of
  the kv group are one work-group, so each K and V tile is read from global
  memory once and staged in SLM for all six. KV tiles entirely past the query
  tile's last row are skipped; only the diagonal tile is masked.
- **The tile shape is not chosen here** (P1). Head dim 256 makes the O
  accumulator large: 6 heads x R rows x 256 x 4 B is 393 KB for R = 64
  (derived), against 512 KB of GRF per 32-sub-group work-group at 256 GRF.

### 3.2 Integration

- `attn_chunk` calls `pf_flash_attn` in place of QK^T, softmax and PV on the
  `l0` and `l0-int8` backends. The composed path stays selectable
  (`B70_PREFILL_ATTN=composed`) as the correctness reference, the role
  sycl-tla plays for the GEMM.
- `pf_s` and `pf_p` are allocated **only when the composed path is selected**.
  On the default path prefill scratch no longer scales with max_len.
- The launch arithmetic in `step.cc` and `attn.h` (`attn_chunk_launches`)
  follows the new kernel's count, and `prefill_smoke_test` pins it.

### 3.3 128k

- `attn_decode` and `attn_reduce` variants at L131072 (M = 1), beside L16384.
- `--max-len` accepted up to 131072 by both CLIs. `loader::load`'s RoPE table,
  the KV cache, `attn_part` (`[24][max_len/64][M][258]` fp32, 202 MB at 128k,
  derived) and any other max_len-sized buffer are checked to fit, with a load
  report line that sums them.
- **Decode grid at shallow depth.** The captured list launches `max_len / 64`
  = 2048 blocks per head group every token. Blocks past `pos` exit at once, but
  their dispatch is paid, and `attn_reduce` walks the partial table. If F4
  (§5) fails, the grid becomes **indirect**: `attn_prep` writes this step's
  group count (`ceil((pos + 1) / 64)`) into device memory, and the captured
  list launches `attn_decode` / `attn_reduce` with
  `zeCommandListAppendLaunchKernelIndirect`. The list is still captured once.

## 4. Correctness gates

- **K1, kernel.** `pf_flash_attn` against the composed path on the same Q, K
  and V (real captures from one FA layer, and random), at depths 2k, 16k and
  32k, chunks of 2048 and 300 rows. Bar: per-(row, head) output cosine
  >= 0.99999 and the max abs error recorded. Not bit-exact: online softmax
  sums in a different order.
- **K2, golden gates.** Every existing registration passes unchanged: prose,
  code and cjk on `l0` and `l0-int8`, the chunk-1000 and chunk-16 pairs, and
  consistency, determinism and replay.
- **K3, long context.** There is no CPU oracle at 128k, so:
  - flash against composed end to end at 32k context (the composed path still
    fits there): logits cosine >= 0.9999 and the same greedy tokens for 64
    steps;
  - at 128k: finite logits, determinism across two runs, replay bitwise;
  - **passkey retrieval**: a 5-digit number stated once near the start of
    ~120k tokens of filler text must be reproduced when asked for at the end,
    at three placements (5 %, 50 %, 95 % depth). This is the behavioural check
    that long attention is right.
- **K4.** `prefill_smoke_test` launch counts and `prefill_replay_test` pass on
  the new kernel.

## 5. Speed bars

Idle box, device 0, interleaved control and candidate, median of 3.

- **P0, baseline (first, before any kernel).** The attention phase of a
  4096-id prefill today (the 148.2 ms of `prefill-parity-2026-09-20.md`
  predates causal QK^T, parity S3), prefill of one chunk at 8k and 16k depth,
  and decode at depths 4k and 16k at max_len 16384.
- **F1.** The attention phase of a 4096-id prefill **<= 80 ms**. vLLM's flash
  kernel measured 63.1 ms (`prefill-parity-2026-09-20.md`). pp4096 must not
  fall below spec 5's 2104.50 t/s record.
- **F2.** One 2048-id chunk prefilled at 32k, 64k and 128k depth: recorded,
  and the attention part must reach **>= 60 % of the bf16 DPAS peak** (183.45
  TFLOP/s) on the full-tile KV range (derived from its FLOP count).
- **F3.** Decode at depths 4k, 32k, 64k and 128k with max_len 131072:
  **>= 90 % of the bandwidth-derived rate**: the effective bandwidth decode
  measures today divided by the bytes one token reads (weights plus KV up to
  that depth). About 18 t/s at 128k (derived).
- **F4.** Decode at 4k depth with max_len 131072 within **2 %** of the same
  depth at max_len 16384. If it fails, §3.3's indirect grid is required.

## 6. Stages

- **P0 (measure):** the baselines of §5.
- **P1 (probe):** a tile-shape sweep for `pf_flash_attn`: query rows
  32/64/128, KV tile 64/128, heads per work-group 6 against 2 or 3 (halving
  the O accumulator, reading K/V twice or three times), at 256 GRF. Measured on
  real shapes at 16k depth, with a K1-style correctness check per arm. Its
  winner is the kernel's tiling.
- **P2 (kernel):** `pf_flash_attn` in `src/kernels/prefill/`, K1.
- **P3 (integration):** `attn_chunk`, the composed path as a selectable
  reference, lazy `pf_s` / `pf_p`, launch arithmetic, K2 and K4, F1.
- **P4 (128k):** the L131072 variants, the max_len plumbing and load report,
  F3 and F4, and the indirect grid if F4 fails.
- **P5 (long-context gates):** K3, F2, the BENCHMARKS rows at depth, and
  this spec's amendment with results.

**Stopping rule.** If K1 fails at every P1 tile shape, stop and record: the
design assumption that fp32 online softmax matches the composed path within
0.99999 is then wrong, and that needs the operator's call before more kernel
work.

## 7. Out of scope

- int8 KV cache (it would halve decode's KV bytes at depth, but raises an
  accuracy question of its own); int8 attention math.
- Prefix caching and session continuation, MTP: next, on top of this.
- The ~0.7 s short-chunk floor of the linear path (probe-w4a8 §15.6).
- max_len beyond 131072.

## 8. Amendment - 2026-09-26, results

Plans 6a (probe) and 6b (integration, 128k) are done on branch `spec6b-flash-128k`.
Every number is **measured** unless marked **derived**; the rows are in
`docs/BENCHMARKS.md` ("Flash attention and 128k (spec 6)") and
`docs/probe-flash-attn-2026-09-25.md`.

### 8.1 Corrections to the design

- **§3.1's output is `pf_o`, not `pf_attn`.** `pf_flash_attn` writes fp32
  `[24][pad256(C)][256]`, per-head stride `rows * 256`, exactly where the composed path
  wrote it, so `pf_attn_gate` and everything after it are unchanged. `pf_attn` (bf16,
  pre-gate) was never its output.
- **The kernel is plan 6a's arm `pfa_KT64_R16_H6_Q0`, promoted with its text unchanged**
  (`src/kernels/prefill/pf_flash_attn.cl`): 16 query rows and all six q-heads of a kv
  group per work-group, 64-key KV tiles, Q re-read per tile, 256 GRF, no SLM staging.
  One launch per FA layer, grid `(ceil(C / 16), 4, 1)`.
- **The decode variants are L131072 and L32768**, beside L16384 and L4096. L32768 exists
  for K3a's 32k engines: max_len 32768 + 256 has no binary, and 131072 has no room for the
  composed path's score scratch at that depth. A max_len with no binary fails at capture
  naming the variant (`attn_decode_M1_L65536_B64`), pinned by `cli_reject_maxlen_uncompiled`.

### 8.2 Operator rulings

- **Ruling A, 2026-09-25:** integrate this kernel now, optimise later. F1 and F2 are
  recorded as measurements, not gates; the speed gate is pp4096 no regression (flash
  >= 0.99x the paired composed run).
- **K3a, 2026-09-26, redefined.** The original bar (flash against composed at 32k: logits
  cosine >= 0.9999 and the same 64 greedy tokens) measured 0.999662 on `l0-int8`, with the
  tokens splitting at step 10, and about 0.99988 on `l0`. The cause is §3.1's one
  rounding point, the bf16 rounding of the unnormalised exp(s - m) against the normalised
  P, amplified by the model: the same near-tie class as spec 5 §8. It is not depth: on
  `l0` the gap is 0.99988 at 2048 ids (one chunk, pos 0) and at 32704. The new K3a: flash
  is **no further from the CPU oracle than composed**, per backend (mean last-row logits
  cosine against the oracle, flash >= composed - 1e-6), on `l0` and `l0-int8` at the 32k
  prompt, plus the 128k behavioural checks. The old figures are measurements now.

### 8.3 Gates

| gate | bar | result |
|---|---|---|
| K1 kernel vs fp64 | cos >= 0.99999, pad rows finite | worst 0.999998112 / 0.999998227 / 0.999999009 / 0.999998031 on (16384, 2048), (777, 300), (0, 2048, qscale 30), (0, 64); pad rows finite. Against the composed path in `attn_chunk_test`: 0.999994971 and 0.999994768 on every row and head |
| K2 golden gates | unchanged | all pass in flash mode; in composed mode all pass and the gate output is line-for-line identical to the pre-spec-6 build (`tools/probe/composed_vs_main.sh`) |
| K3a | flash >= composed - 1e-6 vs the CPU oracle, per backend | **pass**, on the golden set's 96 decision rows (`flash_vs_oracle_test`): `l0` 0.999960061 vs 0.999959889, `l0-int8` 0.999934564 vs 0.999931742 (mean logit cosine, flash vs composed). At 32704 ids (`flash_long_test`) every logit is finite in both modes on both backends. **Not at the 32k capture itself:** the CPU oracle there (`tools/oracle/last_logits.py`, chunked through one cache, checked against one forward at 3000 ids: cos 0.999925) measured 3799 s to reach 4096 ids and 11359 s to reach 8192, so 32704 is a day-class run; the script keeps what it reaches and `flash_long_test <snap> <ids> <file>` applies the bar at depth when a file exists |
| K3a (old bar, now a measurement) | | flash vs composed at 32704 ids: 0.999661844 on `l0-int8`, first differing greedy step 10 of 64; ~0.99988 on `l0` |
| K3b 128k | determinism, replay bitwise, finite | two fresh engines on 131000 ids bitwise equal; replay x2 bitwise equal to immediate; logits finite |
| passkey | 3/3 at 5 / 50 / 95% depth | 3/3 on `l0-int8`, 3/3 on `l0` (119939 ids; every generation " 71432.") |
| K4 | launch counts, replay | `prefill_smoke_test` pins flash at 8449 launches per chunk at every C (8705 on `l0-int8`), and a composed session after a flash one bitwise equal to fresh runs of each; `prefill_replay_test` passes |

### 8.4 Speed

| bar | target | result |
|---|---|---|
| pp4096 no regression (the gate under ruling A) | flash >= 0.99x paired composed | 1.0095 and 1.0102 (2125.12 / 2105.02, 2125.30 / 2103.76 t/s); **pass** |
| F1 | attention <= 80 ms per 4096 ids | 116.3 ms (`attn_flash`), composed 132.2 ms; **missed, recorded (ruling A)** |
| F2 | >= 60% of bf16 peak at depth | 28.3 TFLOP/s, 15.4% of 183.45 at pp65536, where attention is 51.3% of GPU time (derived); **missed, recorded (ruling A)** |
| F3 | decode >= 90% of the bandwidth-derived rate at depth | 78.7% at 32768, 66.2% at 65536, 53.1% at 130816 (20.60 / 15.46 / 10.19 t/s against 26.17 / 23.34 / 19.20 derived); **missed, recorded**. Met by spec 10's decode attention v2 (2026-09-28): 27.85 / 24.51 / 20.13 t/s, spec 10 §8 |
| F4 | decode at 4k, max_len 131072 within 2% of 16384 | 29.28 against 29.27 t/s, 1.0003; **pass, so §3.3's indirect grid was not built** |

Memory at 128k (the load line both CLIs print): model 18.116 GB, KV 8.590 GB, decode
state 0.590 GB, prefill scratch 0.700 GB, int8 0.085 GB, **28.081 GB of 32.530 GB**.
Prefill at depth: 1499.37 t/s at 32768 ids, 1121.74 at 65536, 746.57 at 130816.

### 8.5 What is next

Attention is half the GPU time of a 64k prefill and decode falls to 53% of its
bandwidth rate at 128k, so both attention kernels are the follow-up: a split-d or
SLM-staged `pf_flash_attn` (F1, F2), and decode attention at depth (F3).

## 9. Amendment - 2026-09-27, spec 6c (plan `2026-09-27-spec6c-flash-exp2-rows8.md`)

**Shipped: 8 query rows per work-group** (`RPW=8`, 6 sub-groups, grid `(ceil(C / 8), 4,
1)`), research lever 2, **and `exp2`** (lever 1, `EXP2=1`), in place of §3.1's "`exp`,
never `native_exp`".

**Operator ruling, 2026-09-27: `exp2` on, with its K3a miss on `l0` accepted.** Plan 6c
first shipped `EXP2=0` because `exp2` misses K3a on `l0` by 7.6e-6 (below). The operator
enabled it: `l0-int8`, the default backend, passes K3a; `l0` with `exp2` (0.999951) is
still closer to the oracle than `l0-int8` has ever been (0.99993-0.99994). K3a's
tolerance is therefore 1e-5 on `l0` and stays 1e-6 on `l0-int8`
(`tools/probe/flash_vs_oracle.sh`). The `exp2` rows below are what ships.

**The `exp2` finding.** K1 (fp64, random data) is bit-identical with `exp2`, as the
research measured. The engine gate is not: with `exp2(s * ATTN_SCALE * log2e - m)`,
`flash_vs_oracle_test` measured `l0` flash 0.999951245 against composed 0.999959889
(bar composed - 1e-6: **FAIL** by 7.6e-6), `l0-int8` 0.999938695 vs 0.999931742 (pass).
Folding log2e after the subtraction instead (`exp2((s - m) * log2e)`) is worse: the `l0`
golden gate itself fails (95 of 96 decision rows determined) and `l0-int8` falls to
0.999904279 (FAIL). The same near-tie class as §8.2's K3a ruling: one ulp-level move in P's
bf16 rounding point, amplified by the model. It needs a ruling (accept the K3a miss on
`l0`, or a numerics change that keeps `exp`'s rounding) before E ships.

**Gates (R, as committed):** K1 passes on all 7 cases, including the new (0, 1), (4096, 1)
(the 4097-id tail chunk, 7 pad rows) and (0, 9): worst cosine 0.999998031 or better, pad
rows finite; K2 all golden-labelled tests pass in flash mode (8/8); K3a passes and is
bit-identical to main (`l0` 0.999960061 >= 0.999959889, `l0-int8` 0.999934564 >=
0.999931742); K4 `prefill_smoke_test` (launch counts unchanged) and `prefill_replay_test`
pass; the full suite is 97/97.

**Speed** (docs/BENCHMARKS.md "Spec 6c", diagnostic grade, load 4.5-19.9): pp4096 1.0037x
main (2136.68 vs 2128.81 t/s), `attn_flash` 101 ms against 117 (**F1 80 ms: missed**),
pp65536 1.074x (1206.97), pp130816 1.097x (821.37), decode at 4k unchanged (29.24 vs 29.25).
With `exp2` (shipped since the ruling above): pp4096 1.015x, `attn_flash` 81 ms (F1 met within 1 ms),
pp65536 1.158x, pp130816 1.222x.

## 10. Amendment - 2026-10-05, `--max-len auto` (branch `max-len-auto`)

§3.3 sized the context by hand: 131072 was checked to fit and §8.4 measured it (28.081 of
32.530 GB). Since spec 10 neither decode attention (v2) nor the MTP lists (spec 8 §12) bake a
max_len, so the length is a property of the card's memory alone, and `b70-serve` now picks the
largest one that fits. **`--max-len auto` is `b70-serve`'s default; `b70-decode` keeps 16384**
(bench rows stay comparable) and accepts `auto`.

**One source of truth for buffer sizes.** Every allocation that a model and a max_len decide
is a device-free function in `runtime/buffer_sizes.h` (archive `b70_plan`, no Level Zero):
`PersistentDims::sizes`, `DecodeScratchDims::sizes`, `PrefillScratchDims::sizes` (the eager
members and the four lazy ones), `MtpDims::sizes`, `mtp_prefill_hidden_bytes`,
`int8_scratch_sizes` and `int8_scale_bytes`. The buffer classes inherit their constants and
`sizes()` from those bases and their constructors allocate exactly what `sizes()` returns, so
a buffer change moves the plan with it. Allocations are byte-identical to before
(`buffers_test`'s table, reproduced on the host by `memory_plan_test`).

**The planner** (`runtime/memory_plan.h`): `plan(desc, max_len, mtp, model_bytes, path)`
returns `memory_line()`'s five components - model (the loaded weights, `report.total() -
report.rope_bytes`, plus the RoPE table at max_len, 256 B per position), KV, decode state
(with MTP: the head's buffers and prefill hidden rows), prefill scratch (eager, plus the lazy
buffers the backend builds: the slab on `l0` and on `l0-int8` with MTP, the dequant scratch on
`sycl-tla`, pf_s / pf_p on the composed attention path - 73,728 B per position on Qwen3.8,
the only prefill scratch that scales with max_len), int8 (Int8State and every int4 linear's
column scales). It reproduces §8.4's line exactly where the formula covers it: KV
8,589,934,592, decode state 590,450,624, prefill scratch 699,514,776, int8 84,680,704 B, and
the model 18,116,331,520 B as docs/13's measured load total with its table re-sized to 131072.
`max_len_that_fits(desc, mtp, model_bytes, device_bytes, reserve_bytes, cap, path)` bisects
the largest multiple of 256 at most `cap` whose plan + reserve fits; 0 if not even 4096 does.
`Engine::memory_use()` returns the allocated five, `memory_line()` formats them with the
planner's formatter.

**The bound is the trained context.** `loader::trained_context` reads config.json's
`max_position_embeddings` (`text_config`, else top level: 262144 for Qwen3.8, Agnes and
Ornith). `load()` refuses a max_len above it; it is auto's cap. Spec 14 §3.3's fixed 65536
ceiling for Agnes (`ModelDesc::max_len_ceiling`) is removed: that was this plan, done once by
hand for the bf16 head.

**Loading order.** The plan needs the loaded bytes and `load()` needs a max_len, so auto loads
at 4096, plans, and `loader::set_max_len` rebuilds the RoPE table at the chosen length before
the Engine exists (nothing reads the table's length from the allocation; every kernel indexes
it by position). An explicit `--max-len N` loads at N as before and is held to the same plan:
if it does not fit, the CLI refuses before the Engine with the plan's breakdown and the
largest length that would. With `B70_DECODE_ATTN=v1` auto takes the largest compiled v1
length at or below the fit. `--mem-reserve-gb G` (default 1.5) is what the plan leaves free
for what it does not count: the driver's own allocations, kernel modules, the captured
command lists (decode, MTP draft / verify, prefill), the int8 sign tables, transient colmax
buffers, allocator slack. **1.5 GB is an estimate; the box confirms or tunes it.**

**Derived lengths** (default reserve, the 32.530 GB card, the int8 head; `memory_plan_test`
prints them): Qwen3.8 192256 (31.025 GB planned), with MTP 162816; Qwen3.8 with the bf16 head
173824; Agnes 134144, with MTP 109824; Qwen3.8 on the composed attention path 92672. Beyond
131072 is unmeasured on the card: no kernel indexes the KV past a per-layer slice (at 262144 a
layer's K is 2^28 bf16), but passkey and decode at depth are the gates.

**Gates (box, `docs/superpowers/plans/box-validation-queue.md`):** P1 `memory_plan_box_test`
(Qwen3.8, with `mtp`, Agnes): every planned component and every buffer equals the allocation at
16384 and at auto, and the 131072 line again equals §8.4's; P2 `b70-serve` at auto starts and
serves (`golden_server_test` now runs it at auto); P3 passkey 3/3 at 95% of the auto length;
P4 decode t/s at a depth near the auto length; P5 the reserve: the device's free memory after
a long session at auto, the 1.5 GB confirmed or tuned.

**Amendment (2026-10-05): `attn_part` is sized for the decode-attention pair.** v1 strides its
partials by every 64-position block of the cache ([24][max_len / 64][8][258] fp32: 405.8 MB at
131072, 0.595 GB at 192k); v2, the default since spec 10, never has more than 32 per row
([24][32][8][258]: 6.34 MB at any max_len). `DecodeScratchDims::sizes(max_len, desc, attn)`
now sizes it by the pair `B70_DECODE_ATTN` selects (`DecodeAttn` and `decode_attn()` moved
from capture.h to buffer_sizes.h for that), so the plan and the allocation follow the same
switch; capture refuses to bind v1 over buffers sized for v2. The §8.4 line is reproduced
under `B70_DECODE_ATTN=v1`; with v2 the same engine plans 0.399 GB less decode state. Auto
lengths at the default reserve (derived): Qwen3.8 int8 head 201,216 (169,984 with MTP), bf16
head 181,760; Agnes 139,520 (114,176 with MTP). `tools/probe/probe_decode_attn.cc` captures
both pairs over one engine, so it builds that engine under v1.

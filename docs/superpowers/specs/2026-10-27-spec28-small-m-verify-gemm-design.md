# Spec 28 - a small-M decode / verify GEMM: M = 1..16 rows, one dequant per weight tile, DPAS

**Status (2026-10-27): DESIGN - nothing built.** First item of the Qwen speed track (operator, 2026-10-27: (1) this
spec, (2) DFlash on the card + lookup inside DFlash - spec 19 §13, (3) the tensor-parallel probe on Qwen3.8 - spec 17
§R2, (4) prefill chunk 4096 - plan 5d). Builds follow one plan at a time; the operator is asked before each.

**Plans:** `docs/superpowers/plans/2026-10-27-spec28a-small-m-probe.md` (P0, the go / no-go line) and
`docs/superpowers/plans/2026-10-27-spec28b-small-m-build.md` (the production kernel and its consumers). **Box rows:**
**68** (28a) and **69** (28b) - the next free rows after 67 on 2026-10-27 (rows 63-65 were spec 26's, dropped
2026-10-27; 66 is spec 13a's, 67 the prefill bandwidth profile; row 68, proposed for spec 17a on 2026-10-23, is
reassigned here and 17a moves to 71). Rows 68-75 are allocated together for the speed track: 68 / 69 spec 28, 70 plan
19f, 71-74 plans 17a-17d, 75 plan 5d. The plans describe their stages; nothing is added to
`tools/box_validate/stages.sh` until a plan is built.

**Grades.** Every number is **measured** (on this box, by this repo, with the record cited), **derived** (arithmetic
from measured numbers or descriptors, shown), **estimated** (a judgement) or **published by others** (someone else's
figure on someone else's hardware, cited; never a bar by itself).

## 1. Why

Speculative decoding and batching both turn one decode step into M rows. Their gain is set by what an M-row step
costs against a 1-row step, and ours costs far more than it needs to.

| fact | value | grade | source |
|---|---|---|---|
| plain decode, Qwen3.8, one card, int8 head | 31.21 t/s (~32 ms a step) | measured | docs/BENCHMARKS.md (spec 9 rows) |
| verify step at M = 2 / 3 / 4 | 1.17 / 1.52 / 1.74 plain steps | measured | docs/BENCHMARKS.md "P0, acceptance and step costs"; docs/probe-mtp-2026-09-27.md |
| verify step at M = 5..8 | ~2.62 at M = 8 (M = 3 -> 4's slope, +0.22 a row) | derived (extrapolated, never measured: row 29's `r29.cost` is pending) | docs/probe-dflash-2026-10-08.md §3 |
| DFlash over `--mtp auto` on A4 tool calls | +54 % (7.21 / (2.62 + 0.2) against 3.82 / 2.30) | derived (on the extrapolated verify) | docs/probe-dflash-2026-10-08.md §3 |
| where the M = 4 verify step's extra time goes (profiled lists; shares, not absolute times) | int4 GEMV family 23.9 -> 40.3 ms (1.69x), **16.4 of the 24.6 ms added**; `attn_decode` (v1 then) 2.5 -> 9.2 ms; `gdn_step` 0.75 -> 1.67; the bf16 head 4.38 -> 4.50 (it already reads its weights once for all rows) | measured 2026-09-27 | docs/probe-mtp-2026-09-27.md (the per-family table) |
| our decode GEMV | per-row: `acc[m]` / `gacc[m]`, and `dot8` re-unpacks every int4 nibble once **per row** - the weight load is shared, the dequant and FMA work is M times | code read | `src/kernels/gemv.cl` (the `for (m < M)` loop inside the `j` loop, `dot8`); docs/probe-batching-2026-10-23.md §1.1 |
| vllm-sly-radiance (a public vLLM ROCm image, one AMD R9700, ~640 GB/s - about our bandwidth): step time | T(M) = 29.9 + 0.393 x M ms (their fit): M = 8 ~1.09, M = 16 ~1.19 of their M = 1 step | published by others (fit), ratios derived from it | vllm-sly-radiance's technical notes |
| why theirs is flat | their decode GEMM is a WMMA tile whose fragment is shared by M <= 16 rows, with split-K | published by others | the same |
| their plain step | ~30 ms, about ours | published by others | the same |

**The gap is the kernel, not the hardware:** their plain step equals ours; their eighth row costs ~1.5 % of a step
(derived from the fit), ours ~22 % (derived from the measured M = 3 -> 4 slope). Every consumer of M rows pays it:

| consumer | today | what a flat verify buys (derived; the 1.4 / 1.2 are this spec's targets, not measurements) |
|---|---|---|
| MTP (spec 8), `--mtp auto` K = 3 on tool calls | 3.82 ids / 2.30 steps = 1.66 | at verify(4) = 1.2: 3.82 / (1.2 + 3 x 0.185) = 2.18, **+31 %** |
| DFlash (spec 19), K = 7 on A4 tool calls | 7.21 / (2.62 + 0.2) = 2.56 (itself on an extrapolation) | at verify(8) = 1.4: 7.21 / 1.6 = 4.51, **+76 %** |
| batching (spec 13 §9, the 8-row budget) | B rows cost like a B-row verify (docs/probe-batching-2026-10-23.md §1.1: the GEMVs are row-agnostic, so the same per-row cost) | B = 4 slots at ~1.2 steps instead of ~1.74 |
| DFlash K up to 15 (spec 19 §13) | not possible: the cost of 16 rows would exceed the acceptance | possible only after this spec (M = 16) |

The verify step also has parts that are not GEMVs - the attention's per-row KV walk (the 2026-09-27 table is v1's;
spec 10's v2, the default since, shares K / V waves across rows), `gdn_step` writing M x 151 MB of GDN state at M rows (~2 ms at M = 8, derived, spec 19 §11), the head at
M rows, the argmax. **P0 (plan 28a) attributes the measured verify step into GEMV and non-GEMV time** before the
kernel is built, so the go rule is set against the whole step, not the GEMV alone.

## 2. The kernel

**`gemm_sm` (small-M GEMM), M = 1..16 rows, weight-stationary, DPAS.** One new source, `src/kernels/gemm_sm.cl`,
beside `gemv.cl`; it reads exactly the weights `gemv` reads (int4 g64 symmetric, both layouts, the same buffers - no
repack, no second copy) and writes exactly the partials `gemv` writes, so every consumer stays as it is.

1. **A sub-group owns 16 output columns** (one DPAS N tile) and walks its K range (split-K slice s of S), one 64-row
   group at a time, as `gemv` does.
2. **Dequant once per tile.** Per group: load the 16 columns' 64 nibbles (layout 0: eight u32 rows of `[K/8][N]`;
   layout 1: the 512 B tile plus 32 B of scales - each lane already holds its column's 8 consecutive k in one u32) and
   turn each nibble into the bf16 integer `q - 8` (exact: |q - 8| <= 8). Each lane's u32 becomes four bf16 pairs along
   K, which is the DPAS B operand's per-lane form; **no shuffle, and the work is the same at M = 1 and M = 16**.
3. **DPAS over the rows.** The M activation rows (bf16, `[M][K]`, contiguous) are the A operand, in row blocks of 8
   (`dpas` repeat count 8; rows past M are zero). Four k16 DPAS cover the group into a per-group fp32 accumulator;
   M = 9..16 issue a second row block against the **same** B registers.
4. **Scale per group, as `gemv` does:** `acc[m] += gacc[m] * scale[n]` in fp32 after each group - the checkpoint's g64
   semantics kept, no scale folded into bf16 weights (that would round `q x scale` to 8 mantissa bits).
5. **Split-K and the output contract:** grid dimension 1 is the slice; slice s writes `out[(s M + m) N + n]`, the
   layout `prep_res_fold` / `prep_silu_mul` / the argmax already read. **S per shape is P0's choice**; the default is
   the shape's current S (`kQwen38Tuning`), because a different S changes the consumer's binary (`SILU_S` is baked
   into `prep.cl`; `Capture::check_sizes` asserts it). A different S is taken only if P0 measures it >= 5 % faster at
   M = 8 for that shape, with the consumer variant it needs.

### 2.1 Prior art: oneDNN's Xe2 int4 decode GEMM

oneDNN (public source; its gemmstone `kernel.db` entries for s4 x f16 on BMG, i.e. our silicon) already ships a small-n
int4 GEMM. What it does for **n <= 16 token rows** (read from the source, published by others; nothing of it measured
here):

- **The weights are the DPAS A operand, the tokens the B operand (n columns).** A B tile is 16 columns wide on Xe2, so
  n = 1..16 tokens cost the **same DPAS count**; it runs DPAS even at n = 1.
- **A work-group of row-subgroups x K-slices, split-K inside the work-group, an SLM reduction** of the slices at the
  end: n <= 8 - 2 row-subgroups x 8 K-slices, no B prefetch; n = 9..16 - 8 x 4, with a 128-k prefetch.
- **K interleaved in chunks** = lcm(load size, prefetch stride, group_k / wgK), so neighbouring slices read adjacent
  cache lines and each group's scales load once per chunk.
- **int4 dequantised in registers inside the k-loop, before DPAS** (no SLM for weights), by the **magic-exponent
  trick**: s4 xor 0x8888 -> u4; OR the nibble into the mantissa of the 16-bit magic 0x4300 (= 128 + x, exact); one add of
  -(128 + 8); multiply by the group scale.
- **Weights streamed with 2D block loads and a deep 2D block prefetch (80-160 k ahead)**, a split barrier every 128 k,
  and **256-GRF mode** when the register footprint needs it.
- **Pitfall (reported with it):** GRF-mode flags not applied froze an Xe2 card. Every 256-GRF binary of ours sets the
  option on its ocloc line (`-cl-intel-256-GRF-per-thread`, as `pf_moe_gemm` does) and the probe checks the built
  binary's GRF mode before a timed arm runs.

**What we take, and how P0 decides:** §2's sketch above puts the weights on the B side (a sub-group owns 16 output
columns; rows in blocks of 8 on A, so M = 9..16 needs a second DPAS row block). oneDNN's orientation - **weights on A,
tokens on B** - covers M <= 16 in one DPAS column block, so its DPAS count is flat to 16, and its output is the
transposed tile (`C[N][M]`, written to our `[(s M + m) N + n]` partials by the store). **P0 times both orientations**
(plan 28a's arms `wB` and `wA`), and for the winner the levers above one at a time: the in-work-group split-K with the
SLM reduction against our grid split-K (which keeps today's consumers), the magic-exponent dequant against our shift
dequant (`GEMV_DEQ_SHIFT`), 2D block loads and prefetch depth, and the GRF mode. The production kernel takes the
orientation and levers P0 measures fastest at M = 8 summed over a token. Property R must hold in the chosen form: with
an **in-work-group** split-K the SLM reduction order is fixed by the work-group shape, so R holds only if that shape is
the same for every M >= 2 (oneDNN switches shape at n = 9 - we may not, or rows 9..16 would differ from rows 1..8: P0
measures the cost of one shape for all M).

**The DPAS type (P0 decides):** **bf16** x dequantised-bf16 (above), or **i8 x i4** (activations quantised to int8 per
row, weights int4 straight into the mixed-precision DPAS). The decode step is bandwidth-bound, so the type changes the
ALU work, not the bytes; the i8 arm also needs a per-row activation quantisation and - for accuracy - the h8 rotation
(docs/17-int8-prefill.md: unrotated int8 activations cost 2.8-5.1 % on real layers, docs/probe-w4a8-2026-09-23.md
§14.2), which decode has never paid. **i8 x i4 is adopted only if it is >= 10 % faster than bf16 summed over a token's
linears at M = 8 AND passes §3's bars as bf16 does**; otherwise bf16 (the expected outcome, estimated).

**The head and the bf16 rows.** The int8 lm_head (`gemv_i8w`, per-column scale, 5120 x 248320, 1.27 GB) gets the
same form: int8 -> bf16 is exact (|q| <= 127), DPAS bf16, the column scale at the end - `gemm_sm_i8w`. The bf16 `a||b`
(5120 x 128) and the MTP head's bf16 linears are recorded by P0 and stay `gemv_bf16` unless P0 shows they matter.

**M = 1 keeps today's path, bitwise.** Every list built at M = 1 (plain decode, `ingest`, `generate`, the MTP draft
list, the prefill tail's head) binds the existing `gemv` / `gemv_i8w` binaries, unchanged; every list built at
**M >= 2** binds `gemm_sm`. One rule in the capture (`runtime::gemv_kind(M)`), not a user flag. A test-only capture
option forces either kernel at any M (for §3's comparisons); there is no environment variable and no CLI switch (the
operator's 2026-10-27 ruling against more options).

**Variants:** one binary per (shape, M) as today's `gemv_M<M>_K_N` names, for M = 2..8 (the lists that exist: MTP
M <= 4, DFlash M <= 8, 13a's rows) and M = 16 (P0, batching, DFlash K <= 15); M = 9..15 are added with the plan that
first binds them (spec 19's K <= 15, spec 13b's row budget). Qwen3.8's six int4 shapes and the int8 head first; Agnes's
gate||up' / down' (PROVISIONAL tuning, spec 14 §6) in 28b too, since it shares the capture.

**Row independence (property R, by construction, held by test):** a DPAS output row depends only on its own A row; the
per-group order (4 k16 DPAS, then the scale), the row block of 8 and S are the same for every M >= 2. So **row m's
output is bitwise identical for every M in 2..16 and whatever the other rows hold.** This keeps what spec 13a's batching
check and spec 19's AdaptiveK need: a verify row does not change when K changes, and a slot's row does not change with
its neighbours - **at M >= 2**. Against M = 1 (`gemv`) rows differ by accumulation order (§3).

## 3. Numerics and gates

DPAS sums the 16 products of a k16 step inside the systolic array and the group's four steps into one fp32
accumulator; `gemv` sums eight products per `dot8` in program order. **The two are not bitwise equal**, as chunked
prefill is not bitwise one-row decode. The gates:

- **N1, M = 1 unchanged:** every existing binary checksum-identical (G0), `tools/kernel_cmdlines` +N / -0 / ~0 (only
  `gemm_sm*` lines added), the full suite green.
- **N2, the kernel against a reference:** `gemm_sm` against an fp64 host reference of int4 g64 x bf16 on real weights
  (the checkpoint's layer 63) with real and random activation rows: max |error| <= 2^-16 x sum_k |w_k x_k| per
  output (the fp32 accumulation bound with margin), cosine >= 0.9999999 per row; and against `gemv` at the same M the
  same bound. Recorded with the max / median per shape.
- **N3, property R:** for M = 2..8 and 16, every row bitwise equal to the same row at every other M >= 2, and with the
  other rows replaced by random rows; 10^3 random trials per shape.
- **N4, MTP's M2 (spec 8 §4), now at M = 2..8:** each verify row's logits against M = 1 decode of the same tokens,
  **cosine >= 0.99999**; each GDN state slot against the M = 1 state, **cosine >= 0.99999**, max abs error recorded;
  after a commit with j < k accepted, 64 further greedy tokens equal a run that never saw the rejected drafts. These
  were always M2's bars (it never required bitwise); spec 8b measured bitwise because the GEMV is per-row.
- **N5, spec 19's D3 restated:** D3 asked M = 5..8 verify rows bitwise M = 1. Under this spec it becomes N4's bars at
  M = 5..8 plus N3 (bitwise across M >= 2). Spec 19 §13 records the change.
- **N6, greedy is lossless under the golden tie rule (spec 8 M3, spec 19 D2):** greedy with `--mtp K`, `--spec dflash`
  and `--spec lookup` against `--spec off`, 256 tokens on the golden prompts and A4: identical except where the golden
  tie rule (docs/14-golden-gate.md) classes the divergence as a near-tie; **zero non-tie divergences**; the tie count
  recorded beside today's.
- **N7, sampling (spec 8 M4, spec 19 D4):** unchanged - host-side; the seeded end-to-end run reproducible bitwise.
- **N8, determinism:** replay bitwise with `gemm_sm` bound (spec 8 M5); the prefix cache's C2 with MTP on.

## 4. The consumers, and how each switches

| consumer | its M-row lists | how it switches | what else changes |
|---|---|---|---|
| **MTP** (spec 8, `--mtp K` / `--mtp auto`) | `runtime::build_verify` at M = K + 1 (2..4) | the capture's `gemv_kind(M)`: no engine flag | the cost table: `MtpCost`'s verify row re-measured at M = 1..8 (28b), replacing the measured-then-extrapolated defaults; `--mtp auto` then picks K on the new costs. More than 3 MTP drafts (`MtpDims::kSlots` = 4) is a follow-on decision, not here |
| **DFlash** (spec 19c, `--spec dflash`) | `build_verify_taps` at M = K + 1 (2..8); the drafter's own int8 linears at its block of 8 rows | the same rule; the drafter's int8 GEMVs at M = 8 bind `gemm_sm_i8w` | `MtpCost::dflash`'s verify row measured (no more +0.22 a row); D3 as N5; K <= 15 only after 28b (spec 19 §13) |
| **prompt lookup** (spec 19e, `--spec lookup`) | the MTP verify lists | as MTP | - |
| **batching** (spec 13b, `--batch N`) | the batched decode list at B rows (B x (K + 1) with MTP in every slot) | the same rule at M = B | spec 13a's check ("each slot's rows equal its run alone") holds bitwise **against the slot run alone through `gemm_sm`** (N3) and under the tie rule against M = 1; the 8-row budget (spec 13 §9) can grow to 16 rows - 13b's decision, on 28b's measured step |
| **pipeline parallel** (spec 16, `--pp 2` verify lists) | each stage's verify lists | the same rule per stage | none |
| **tensor parallel** (spec 17 §R2) | each card's verify lists at per-card shapes | the same rule at the per-card shapes | the crossings carry M rows |
| prefill | C >= 256 rows, the slab GEMMs | not touched | - |

## 5. Speed bars

Idle box (both cards free of DRM holders), device 0, interleaved pairs after a warm-up (clock state moves sequential
arms by up to 35 %, docs/probe-w4a8-2026-09-23.md §14.4), median of 5 for kernels, of 3 for whole steps; depth 4096 and 32768; int8 head,
bf16 KV (int8 KV recorded).

- **S1 (P0, 28a):** the go rule, §6.
- **S2 (28b), the verify step on the real list:** `probe_mtp_steps` / `probe_verify_m` at M = 1..8 and 16:
  **verify(8) <= 1.4 plain steps** at depth 4096 (target set from P0's derived value, never above 1.4), verify(2) and
  verify(4) recorded against 1.17 / 1.74; **M = 1 within noise of today** (its binaries are G0's).
- **S3 (28b), end to end:** `tools/probe/mtp_d1.sh greedy 4096` - the tool-call geomean at `--mtp auto` against
  today's 1.633x (BENCHMARKS "D1, greedy decode"): **>= +10 %** relative; golden prose no worse than today's best K.
  DFlash's D5 (spec 19) is re-run with the measured cost table by plan 19f, not here.

## 6. P0's go / no-go rule (plan 28a)

P0 times Qwen3.8's GEMV shapes - `qkv||z` 5120 x 16384 (L1 S1, 48 layers), `out_proj` 6144 x 5120 (L0 S4, 48),
`q||k||v` 5120 x 14336 (L0 S2, 16), `o_proj` 6144 x 5120 (L0 S4, 16), `gate||up` 5120 x 34816 (L0 S8, 64), `down`
17408 x 5120 (L0 S4, 64), the int8 head 5120 x 248320, `a||b` recorded - at **M = 1 / 2 / 4 / 8 / 16**, the current
kernel against stand-in `gemm_sm` arms (bf16 and i8 x i4), interleaved pairs on the box; and profiles the real verify
list at M = 1..8 for the non-GEMV share.

- **GO** (plan 28b may start, the operator asked) when ALL hold:
  (a) **the GEMM sum:** a token's linears (each shape x its layer count, plus the head) at **M = 8 cost <= 1.25x** the
  same sum at M = 1 through today's `gemv` (measured, stand-in kernel);
  (b) **the whole step:** the derived **verify(8) = measured non-GEMV time at M = 8 + (a)'s GEMM sum <= ~1.4 plain
  steps** (derived from measured parts);
  (c) **numerics:** the stand-in passes N2 and N3 on the box.
- **The operator's call:** (b) between 1.4 and 1.6 plain steps.
- **NO-GO** - recorded, spec 28 parked - when (c) fails, or (b) is above 1.6, or (a) above 1.5. The attribution then
  names the next lever (the attention's per-row walk, the GDN slot writes - spec 19 §11's replay-state lever - or the
  head).
- **Recorded, not gating:** the orientation (weights on B or on A, §2.1) and each oneDNN lever's effect; M = 2 / 4 / 16
  per shape; S per shape at M = 8; the i8 x i4 arm and the choice of type;
  `gemm_sm` at M = 1 against `gemv` (if it is as fast, §8 question 1); depth 32768's non-GEMV share.

## 7. Stages

- **28a, P0 (row 68):** the probe, its stand-in kernels and host reference (Mac), the shape table and the step
  attribution (box), the verdict in §9.
- **28b, build (row 69):** `gemm_sm` / `gemm_sm_i8w`, the capture rule, every M >= 2 list on it, N1-N8, S2-S3, the
  measured cost table, BENCHMARKS rows; after GO and the operator's go.

## 8. Out of scope, and open decisions

**Out of scope:** M > 16; the MoE expert GEMVs (`moe_gate_up` / `moe_down`, Ornith; Bolt's / K2's / Kolibri's own
engines and kernels) - a follow-on per engine once 28b's numbers exist; the prefill's short-prompt floor (separate,
docs/17-int8-prefill.md); the MTP head's bf16 M-row GEMVs unless P0 shows they matter.

**Open decisions (operator):**
1. **M = 1 through `gemm_sm` too?** If P0 measures it as fast as `gemv`, one kernel for every M would make every row
   bitwise across all M (N3 down to M = 1) - at the price of new decode numerics for plain decode and a new golden
   baseline (G0 is no longer bitwise). Proposed: **no** - keep M = 1 bitwise, as the operator asked. **Decided (operator, 2026-10-27): no.**
2. **bf16 or i8 x i4:** by §2's rule (expected bf16).
3. **Agnes in 28b or a follow-on:** proposed **in 28b** - **decided (operator, 2026-10-27): in 28b** (same capture, its PROVISIONAL S copied as for `gemv`).

## 9. P0 verdict

(Empty until plan 28a runs: the shape table, the step attribution, the type and S chosen, the GO / NO-GO line.)

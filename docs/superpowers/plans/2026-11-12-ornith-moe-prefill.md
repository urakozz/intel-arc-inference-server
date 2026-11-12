# Ornith 1.5 MoE prefill rework - about 2x prefill (design targets Ornith and Flash-Next)

**Status (2026-11-12): Tasks 1-3 DONE (probe GO; the sort and the fused int4 GEMM bound, bitwise; Ornith 4k l0
3,377 -> 6,552 t/s), Task 5 in progress.** Evidence: `docs/probes/ornith-prefill-profile-2026-11-12.md` (card 0, main
238f0582). Earlier evidence this plan builds on: `docs/probes/probe-fused-dequant-2026-09-22.md` (fused dequant on a
DENSE shape: bitwise, but rejected at 3.3x the cost of the pass it removed), `docs/probes/probe-w4a8-2026-09-23.md`
(W4A8 rejected: 2.79 % error), open-work 3b's `pf_gemm_q4.cl` (int4 read in registers, bitwise the slab walk). Box
row **83** (new, Task 1 adds it).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh card 0` (speed) or `card <N>` (correctness), long jobs detached (`nohup setsid`, logs
> under `~/b70/logs/ornith-moe-pf/`), checked at most every 30 min with one blocking wait.

**Goal:** Ornith 1.5 prefill on one card from 3,362 / 3,259 t/s (4k / 16k, `l0`, the default) and 3,937 / 3,676 t/s
(`l0-int8`) to about 2x at 4k. The MoE FFN is 72 % (l0) / 66 % (l0-int8) of prefill time at 4k; its three sinks are
the per-chunk expert-weight pass (29-33 %), the grouped GEMMs at ~18 % of peak (17-23 %) and the one-work-group sort
(~10 %).

**Design targets:** Ornith (256 experts + 1 shared, top-8, hidden 2048, width 512) and Qwen3.8-Flash-Next (512 + 1
shared, top-10, hidden 2560, width 640, 4 hyper-connection streams - the HC streams sit outside the MoE block, whose
input is one normed [C][2560] row set and whose output is `out`, folded by `q4_hc_combine_norm`). Every new kernel is
parametric in (experts, top_k, hidden, width) and built at both shapes. **K2, Kolibri and Bolt are out of scope:** they
keep binding today's binaries, bitwise unchanged (G0); see "Later" at the end.

## Expected gains (all DERIVED from the profile's kernel times, not measured)

Kernel ms of the instrumented 4k l0 run (1188 ms in all): weights 391, grouped GEMMs 277, sort 123, route 36, gather +
combine 28, everything outside the MoE 333. The t/s below scale today's plain numbers by the kernel-time ratio.

| task | what goes | 4k l0 ms saved | 4k l0 t/s after (cumulative) |
|---|---|---:|---:|
| 2 sort: multi-work-group histogram + scan | 123 -> ~10 (3 launches of ~40 us a layer and chunk) | ~113 | ~3,720 |
| 3 + 4 fused int4 grouped GEMM over multi-tile work items | weights 391 + GEMM 277 = 668 -> ~260 (DPAS at ~40 % of peak = ~130 ms, plus ~130 ms of in-register dequant at the 2026-09-22 measured rate of 5.2e11 conversions/s for ~270 items of 128 rows a layer and chunk) | ~410 | ~6,000 |
| 5 l0-int8's MoE on the fused bf16 path | l0-int8 only: its 331 + 200 ms -> ~260 | (l0-int8 ~270) | l0-int8 ~6,300 |
| 6 chunk cap 4096 for MoE models | fewer partial expert items (~-25 % dequant), half the per-chunk launches | ~30-50 | ~6,300-6,600 |

At 16k (l0, 4955 ms): sort -450, weights + GEMM 2472 -> ~1040, chunk ~-160: ~2900 ms, **~5,500 t/s (~1.7x)** - the
16k rows keep attention (590 ms) and GDN (623 ms), which this plan does not touch. Reaching ~7.7k at 4k needs a further
~100-150 ms out of GDN, the dense linears or the norms: named in the closing record, not planned here.

**Order and why:** measure the fused designs first (Task 1): the 2026-09-22 probe rejected fused dequant on a dense
shape, and the only reason it should win here is that the separate pass converts all 257 blocks for ~64 rows each -
the probe must show the amortisation per work item. The sort (Task 2) is independent, bitwise and safe, and removes
the KC = 2048 SLM limit that Task 6 needs. Then the GEMM (Tasks 3-4), then the numerics change (Task 5), then the chunk
(Task 6), Flash-Next (Task 7) and the record (Task 8).

**Stop rules:**
- Task 1: if no fused candidate meets the break-even below at C = 2048 on Ornith, Tasks 3-4 do not start; the plan
  continues with Task 2, Task 6 (the separate pass paid once per 4096 tokens) and asks the operator whether to probe a
  chunk cap of 8192 (a PrefillScratchDims change that reaches the dense models' scratch).
- Any bitwise gate that fails stops the task until the first differing value is explained (a 1-ulp difference is a
  finding, as in `pf_moe_test`).

## Dependencies

- Main at or after 8b82ed3d. Branch `ornith-moe-prefill`, own worktree.
- Box data: the Ornith int4 checkpoint (`ornith`), its f32 references (`oracle_ornith_f32`, `oracle_ornith_f32_pfi8`),
  the long32k f32 reference prefill_split_ornith uses; Flash-Next: the synthetic checkpoints (`oracle_q4exp_synth`,
  `oracle_q4exp_synth_golden`) and Intel's checkpoint at `--layers 18` (`q4exp_intel`, `oracle_q4exp_layers`). Card 1
  holds the operator's Flash-Next quantisation over the weekend: until it ends, every job here takes card 0 only.

## Global constraints

- **Additive kernels.** New sources only: `src/kernels/prefill/pf_moe_sort_mw.cl` (Task 2) and
  `src/kernels/prefill/pf_moe_q4gemm.cl` (Tasks 3-4). `pf_moe.cl`, `pf_moe_gemm.cl`, `q4_pf_moe.cl` and every existing
  command line stay byte for byte until Task 8 (`tools/kernel_cmdlines`: + only; G0's `g0.sha` shows every pre-existing
  binary identical). K2's, Kolibri's and Bolt's host code is not touched.
- **A/B switch:** `B70_PF_MOE=old` (env, test and probe use only; zero-config allows env vars for experiments) puts a
  model back on today's walk. No user-facing flag. Task 8 deletes the losing path and the variable.
- **Numerics per task:** Tasks 2, 3, 4 and 7 are **bitwise** today's output (the sorted layout is a function of the
  route rows; every output row keeps today's B values - the slab statement for statement or the magic-exponent form,
  which gives the same fp32 value `q - 8` - and today's per-row DPAS chain: k ascending, k16 halves, from zero, no
  split-K). Tasks 5 and 6 are **judged** by the f32 judge with the 1-bf16-ulp near-tie rule (memory: moe-gates-f32-
  judge): the engine's distance to f32 no larger than the bf16 golden's, both counts printed; l0-int8 is judged
  against the plain f32 reference (ruling 2026-10-29).
- **Timing:** control and candidate as interleaved pairs after a warm-up, median of rounds, card 0 only (card 1 is
  ~3.4 % slower on prefill); every number labelled "quiet box" or "busy box". Probe ms are L0 kernel timestamps.
- **Launch counts:** `moe_chunk_launches` stays the one home of the per-chunk count; the tests that assert 2061 /
  2021 (`ornith_prefill_{l0,l0-int8}_test`) and Flash-Next's `prefill_chunk_launches` take the new value from it.

---

### Task 1: the probe - fused grouped GEMM candidates and the sort, on real routes (Mac, then box)

**Files:**
- Create `tools/probe/probe_moe_prefill.cc` + `tools/probe/probe_moe_prefill.cl` (probe-only kernels), build stanza in
  `tools/probe/CMakeLists.txt`.
- Modify `tests/golden/ornith_prefill_test.cc`: a `--dump-routes DIR` mode that writes the route rows of all 40 layers
  after a real prefill (the scratch keeps them per layer for R2) - 4096 ids of the long prompt, chunks of 2048, and one
  run at `--prefill-chunk 1024`; and the same for Flash-Next through `qwen4exp_prefill_test` at `--layers 18` on
  Intel's checkpoint.
- Modify `tools/box_validate/stages.sh`: row 83 and stages `r83.routes` (the dumps), `r83.probe` (opt-in, timed).

**What the probe runs** (one layer's shapes, random int4 layout-1 weights, the dumped route rows of every layer, the
real sort's tile table; C = 2048 and 4096 from the 4096-id dump):

| arm | what |
|---|---|
| control | today's production binaries: `pf_moe_dequant_gu` x 2 batches + `pf_moe_gemm` SILU, `pf_moe_dequant_dn` + `pf_moe_gemm` down |
| F1 | fused, today's 32-row tiles, each sub-group converts its own 64 columns in registers (`pf_gemm_q4.cl`'s layout-1 read) |
| F2-MI | fused over WORK ITEMS of up to MI = 64 / 128 / 256 rows of one expert (whole 32-row tiles; a sub-group whose 32 rows are past the item's end skips its DPAS), the work-group's 32k x 256n B slab converted ONCE into a double-buffered SLM image in the VNNI-transform register layout, 4 n x (MI / 32) m sub-groups of 32 m x 64 n, 256 GRF |
| F3-MI | as F2 with the magic-exponent conversion (`0x4300 | q` as bf16 = 128 + q, minus 136: the same fp32 `q - 8`, then x scale and rne as the slab) |
| F4 | the best of F2 / F3 with persistent work-groups (grid = 2 x the Xe cores' slots, items taken from an atomic counter reset by the sort; each item is computed by one work-group, so the output does not depend on who takes it) |
| sort | today's `pf_moe_sort` vs a first histogram + scan + scatter (Task 2's design) |

- Every fused arm: `memcmp` of h and y against the control over all rows (raw 32-bit words). A fast arm that is not
  bitwise is a failed arm.
- Flash-Next: the same arms at its shapes (gate||up 2560 x 1280, down 640 x 2560, 512 + 1 blocks, top-10) on the
  `--layers 18` dumps.

- [x] **Step 1 (Mac):** the probe, the dump mode, row 83; `tools/box_validate.sh --dry-run --only r83` prints the
  commands; host tests green. Commit `probe_moe_prefill: fused grouped GEMM arms and the sort on real routes (Ornith
  MoE prefill Task 1)`.
- [x] **Step 2 (box, card 0):** `r83.routes`, then `r83.probe`. **Done 2026-11-12: GO** - the chosen arm (F1, 64-row
  sub-groups, magic-exponent conversion, MI 64) 2.74 ms a layer-chunk at C 2048 (break-even 6.7, target 4.2; this
  harness's control 10.66), 4.35 at C 4096 (control 12.20); sort 1.50 -> 0.048 ms; Flash-Next 27.1 -> 5.65 ms,
  sort 2.94 -> 0.078; all arms bitwise (W4A8 integer-oracle exact). Record:
  `docs/probes/probe-moe-prefill-2026-11-12.md` (numerics of the W4A8 forms included). Record `docs/probes/probe-moe-prefill-2026-11-xx.md`:
  per arm ms a layer and chunk at C = 2048 / 4096, TFLOP/s, items, bitwise verdict, register spill from the shader
  dump (`-cl-intel-256-GRF-per-thread`).
  - **Gate (pre-registered, written into the record before the run):** today's weights + GEMM per layer and chunk is
    8.35 ms at C = 2048 (668 ms / 80, the profile). **Break-even:** the best fused arm <= 6.7 ms (0.8x). **Target:**
    <= 4.2 ms (0.5x). Bitwise on every arm that counts. The sort arm <= 0.3 ms a layer and chunk (today 1.5).
  - The record names the chosen MI, conversion form and scheduling for Ornith and for Flash-Next (they may differ:
    Flash-Next has ~40 rows an expert at C = 2048, Ornith ~64).
  - Below break-even: the stop rule.

### Task 2: the sort as a multi-work-group histogram + scan (Mac, then box) - BITWISE

**Files:** create `src/kernels/prefill/pf_moe_sort_mw.cl`; `src/kernels/prefill/CMakeLists.txt` (Ornith:
`pf_moe_sort_mw_E256_T8`), `src/kernels/CMakeLists.txt` (Flash-Next: `pf_moe_sort_mw_E512_T10`, built now, bound in
Task 7); `src/kernels/prefill/pf_kernels.h` (variant name); `src/runtime/prefill/moe.cc` (bind under
`B70_PF_MOE != old`); `tests/kernels/pf_moe_ref.h` / `pf_moe_ref_test.cc` (the blocked walk == the serial walk);
`tests/kernels/pf_moe_test.cc` (new arm).

Three launches, no atomics, no SLM staging of the whole chunk (so no KC):
1. `pf_moe_hist` - grid (ceil(C / TB)), TB = 64 tokens a work-group: per-block counts `cnt[blk][e]` (ids as ushort,
   any number of experts up to 1024 lanes x experts-per-lane);
2. `pf_moe_scan` - one work-group: per expert the exclusive prefix over blocks, the experts' padded tile offsets, the
   header, the tile table padded to tmax with NONE, the shared expert's tiles; and (for Task 3) the work-item table
   and the persistent counter reset;
3. `pf_moe_scatter` - grid (ceil(C / TB)): each block's pairs walked in ascending (token, slot) order inside the block
   (an SLM per-expert rank), row = expert base + block prefix + rank; padding rows NONE.
Rows inside an expert stay in ascending (token, slot) order, so header, tiles, `row_tok` and `pair_row` are byte for
byte today's.

- [x] **Step 1 (Mac):** host twin first (`pf_moe_ref_test`: the blocked walk == `pf_moe_ref::sort` on random routes,
  the adversaries - all tokens to 8 / 10 experts, one-row experts, C = 1, 37, 2048, 4096, 16384), then the kernel and
  its binding. `mac_check` green. Commit.
- [x] **Step 2 (box, card 1 - correctness):** **Done 2026-11-12**: pf_moe_test's mw arm exact (device routes, the
  crafted ones, C 1 / 37 / 2048, random 4096 / 16384, its work items the host's); the whole Ornith prefill of
  long32k[0, 4096) old sort vs mw sort: every layer's routes and the logits BITWISE (the G0 compare's content,
  run as an A/B of `B70_PF_MOE`); ornith_prefill_{l0,l0-int8}_test PASS; pf_moe_ref_test, memory_plan_test PASS.
  The scan also writes the work items (MI 64) Task 3's GEMM walks. Sort a layer-chunk 0.048 ms (probe) = ~4 ms
  at 4k. `pf_moe_test` (sort arm: every buffer exact against `pf_moe_sort` at C = 37 / 2048 and
  against the host twin at 4096 / 16384), `ornith_prefill_l0_test` (new launch count), `r83.probe`'s sort arm in situ.
  - **Gates:** sort buffers bitwise the old kernel's; Ornith's prefill logits and routes bitwise main's (G0 mechanism:
    `G0_BITWISE_RE` extended for this run with `prefill_gate_ornith_l0_test`, `prefill_gate_ornith_l0_c16_test`,
    `ornith_prefill_l0_test`, `--baseline main`); `B70_PREFILL_PROFILE=1` 4k l0: the sort row <= 15 ms (today 123).
  - Commit the numbers.

### Task 3: the fused int4 grouped GEMM over work items (Mac, then box) - BITWISE

**Files:** create `src/kernels/prefill/pf_moe_q4gemm.cl` (Task 1's winning F-arm, made production: gate||up with the
SiLU epilogue and down with the rne epilogue, verbatim from `pf_moe_gemm.cl`; B read as int4 layout-1 blocks at the
layer's block base + id - one address line, as `q4_pf_moe.cl`'s `Q4_EXPERT_*`, so Flash-Next's tiering hook can
replace it); CMake stanzas at Ornith's shapes (K2048 N1024 SILU, K512 N2048) and Flash-Next's (K2560 N1280 SILU, K640
N2560); `pf_kernels.h`; `src/runtime/prefill/moe.cc` (l0: gather, one gate||up launch, one down launch - no weight
batch, no dequant); `src/runtime/buffer_sizes.cc` (`moe_prefill_layout`: the weight batch region only while a backend
still needs it - l0 alone frees ~0.54 GB); `memory_plan` / `memory_plan_test`; `tests/kernels/pf_moe_test.cc` (fused
== today's grouped, bitwise, both batches' experts, the shared block, C = 37 / 2048, the reversed chunk).

- [x] **Step 1 (Mac):** the kernel, the binding (l0 only; l0-int8 keeps today's gate||up and takes the fused down,
  which is bf16 on both backends today), the scratch sizes, the tests. `mac_check` green. Commit.
- [x] **Step 2 (box, card 0):** **Done 2026-11-12**: `pf_moe_q4gemm` (the probe's F1.a8b2m1n8.c2: 64-row
  sub-groups, MI 64 work items from the sort's scan, the magic-exponent conversion, n-blocks on grid dimension 0).
  pf_moe_test 4b: h and y bitwise today's grouped GEMM at C 2048 and 37; the whole Ornith prefill of
  long32k[0, 4096) on l0 AND l0-int8 (its down is fused): every layer's routes and the logits BITWISE today's walk
  (`B70_PF_MOE` A/B, card 1). Card 0, 3 interleaved rounds, pp4096 tg32: **l0 3,377 -> 6,552 t/s (1.94x)**, l0-int8
  3,910 -> 5,267 (down only). `B70_PREFILL_PROFILE=1` 4k l0: moe_weights 0, moe_gemm 194.5 ms (gate <= 534, target
  334: both met), moe_sort 3.7 ms. (The weight batch region stays until Task 5 removes its last user.)
  `pf_moe_test`, `ornith_prefill_{l0,l0-int8}_test`, `prefill_gate_ornith_*`,
  `prefill_split_ornith_*`, `memory_plan_test`.
  - **Gates:**
    - fused h / y bitwise today's grouped GEMM on the card (every row);
    - Ornith l0 and l0-int8 prefill output bitwise main's (the extended G0 compare, as Task 2);
    - `B70_PREFILL_PROFILE=1` 4k l0: weights + GEMM <= 0.8 x 668 ms (break-even), recorded against the 334 ms target;
    - `--max-len auto` for Ornith does not shrink (printed: it should grow by the freed batch).
  - Commit the numbers.

### Task 4: scheduling and item size tuned in situ (box, then Mac) - BITWISE

Only what Task 1 left open: MI per shape, persistent vs plain grid, prefetch depth (the oneDNN pattern: 2D block
prefetch of the int4 stream 2-4 k-steps ahead, shared across the work-group). Each variant is a define of
`pf_moe_q4gemm.cl`, compared in situ, not a new kernel.

- [ ] **Step 1 (box, card 0):** interleaved pairs of the Task 3 binary against each variant on the 4k and 16k l0
  bench (`b70-decode --bench --prefill-length N --tg 32`, `B70_PREFILL_PROFILE` off for the t/s, on for the phase
  row). Record in the probe doc.
- [ ] **Step 2 (Mac, then box):** the winner becomes the built variant (the others deleted); `pf_moe_test` and the
  extended G0 compare again.
  - **Gates:** bitwise as Task 3; 4k l0 t/s >= Task 3's + 3 % or the variant is not taken (a variant that wins < 3 %
    is noise on this box); the MoE GEMM row's TFLOP/s printed against the profile's 33.5.

### Task 5: l0-int8's MoE on the fused path (Mac, then box) - JUDGED - needs operator decision D1

Today l0-int8 requantises the whole gate||up array to rotated int8 every chunk (~200 ms of its 331 at 4k) and runs an
int8 grouped GEMM. After Task 3 the fused bf16 gate||up removes that pass and is closer to f32 (the operator asked on
2026-10-31 for l0-int8 to be "fast and accurate", and ruled l0 the MoE default for routing accuracy). A fused W4A8
gate||up is not offered: unrotated int8 activations were rejected on 2026-09-23 (2.79 % error), and rotating the
weights inside the GEMM costs more than the pass.

**Files:** `src/runtime/prefill/moe.cc` (l0-int8: the dense linears stay int8, the MoE runs the l0 walk; `moe_prepare_
int8` and the requant scratch go), `buffer_sizes.cc` (the weight batch region goes for Ornith entirely), tests' launch
counts, `tools/box_validate/stages.sh` r13 descriptions.

- [x] **Step 1 (Mac):** implement behind D1; `mac_check` green. Commit. **Done 2026-11-12**: `pf_moe_w4a8.cl`
  (pf_quant_g64 per token and 64-k group, pf_moe_gather_i8g, the s4 DPAS GEMM over the work items, SiLU epilogue)
  and the bf16 experts (`pf_moe_q4gemm`) for l0-int8, both as test switches (`B70_PF_MOE=w4a8` / `=i8bf16`);
  pf_moe_test 4c: quantiser and gather exact, h exact against the host's integer sums.
- [x] **Step 2 (box, card 0):** **Done 2026-11-12 - neither candidate passes the f32 judge's token gate; l0-int8 keeps
  the rotated int8 gate||up (its down is fused since Task 3).** R3 (determined rows exact, one chunk / chunks of 16):
  today 90/92 (one accepted) / 92/92 PASS; W4A8 89/92 / 91/92 FAIL; bf16 experts 90/92 (row 18 not accepted) /
  90/92 FAIL. R2 differing sets (f32 judge, bf16 golden 1140): today 1314, W4A8 1264, bf16 experts 1326.
  pp4096 / pp16384 (card 0, 3 rounds): today's l0-int8 3,913; W4A8 7,593; bf16 experts 6,804 / 5,879 (l0 6,538 /
  5,676). The operator's further evidence (2026-11-12: the misses' margins, A4, the f32-judged serving rows, all
  three forms on card 1) decides whether W4A8 becomes the default. `prefill_gate_ornith_l0-int8_test` (+ `_c16`),
  `prefill_split_ornith_l0-int8_test`,
  `ornith_prefill_l0-int8_test`, interleaved 4k / 16k l0-int8 bench.
  - **Gates (judged):** by the plain f32 reference, the engine's differing MoE sets and token misses no more than
    today's l0-int8 (printed beside the bf16 golden's); 4k l0-int8 t/s >= today's l0-int8 x 1.5 (derived: ~1.6x).
  - If D1 is "no": l0-int8 keeps the int8 gate||up; it gets the fused down only (Task 3), and this task closes with
    that number.

### Task 6: the chunk cap 4096 for MoE models (Mac, then box) - JUDGED - may need operator decision D2

After Task 2 there is no KC, and after Task 3 there is no per-chunk weight pass: what a bigger chunk still buys is
fewer partial expert items (~128 of ~270 items a chunk are an expert's tail), half the per-chunk launches, and fuller
tiles for Flash-Next (~40 rows an expert at 2048). The cap goes to the existing `kCMax` = 4096 (plan 5d); 8192+ would
change `PrefillScratchDims` for every model and is not in this plan.

**Files:** `src/runtime/buffer_sizes.cc` (`check_prefill_chunk_cap`: the MoE refusal goes for models on the new walk;
`moe_prefill_layout` sized by the session's cap), `memory_plan.cc` (Ornith's default cap), `k2_kernels.h` /
`qwen4exp_sizes.h` untouched for K2; Flash-Next's `kPfC` follows in Task 7; `memory_plan_test`; `cli_reject_*` tests
that expect the 2048 refusal for Ornith.

- [ ] **Step 1 (Mac):** the cap, the scratch, the tests. Commit.
- [ ] **Step 2 (box, card 0):** `--prefill-chunk 4096` vs 2048 interleaved at 4k and 16k (l0, l0-int8);
  `prefill_gate_ornith_*`, `prefill_split_ornith_*` at cap 4096.
  - **Gates:** judged as Task 5 (the chunk boundaries move, so GDN state and attention rounding move); scratch +
    `--max-len auto` printed (the freed weight batch should pay for the larger rows).
  - **Default:** if 4096 gains >= 5 % at 4k and at 16k, it becomes Ornith's default cap (the operator's +5 % bar,
    open-work item 5); below that, D2.

### Task 7: Flash-Next on the new walk (Mac, then box) - BITWISE (sort, GEMM), JUDGED (its chunk cap)

**Files:** `src/runtime/qwen4exp/qwen4exp_prefill.cc` (bind `pf_moe_sort_mw_E512_T10` and the fused GEMMs; the
shared expert: ours is an int4 layout-1 block - fused; Intel's bf16 form keeps its `_shb` copy into a bf16 B and a
`SHB` define of `pf_moe_q4gemm.cl` reads that one block as bf16 - a uniform per-work-group branch), `qwen4exp_sizes.h`
(`kPfC`, `kPfBatchBytes` - the batch region goes for our form), `tests/kernels/qwen4exp_pf_kernels_test.cc` section 6,
`qwen4exp_plan_test`, `cli_reject_qwen4exp_prefill_chunk`.

The tiering of plan 21v2a reads cold experts from pinned host memory: a fused GEMM must never read them over PCIe per
work item. The address line stays the one hook; when 21v2a lands, a cold expert used in the chunk is copied as int4
(4x fewer bytes than today's bf16 pass writes) into a VRAM staging slot before the GEMM. This task only keeps the hook
in one place; 21v2a's tasks own the copy.

- [ ] **Step 1 (Mac):** the binding behind `B70_PF_MOE`, the kernel test's fused arm (fused == today's grouped,
  bitwise, both shared forms, at the real shapes on synthetic weights; the sort at C = 2048 / 4096 on the 512-expert
  adversaries). Commit.
- [ ] **Step 2 (box, card 0):** `r33.k1`, `r33.prefill` (both synthetic forms), `r33.golden`, `r33.golden_real`
  (`--layers 18`, Intel's checkpoint), `r33.speed`'s pp4096 at `--layers 18` interleaved against main.
  - **Gates:** at cap 2048 every r33 output bitwise main's (the extended G0 compare with the r33 tests); pp4096 at
    `--layers 18` >= main's x 1.3 (derived from the same sinks; Flash-Next's sort and pass were never profiled - the
    phase table is printed and recorded); cap 4096 judged by r33.golden's tie-aware gate and r33.golden_real.

### Task 8: the record, the clean-up and the final gates (box idle, then Mac)

- [ ] **Step 1 (box, card 0, idle):** interleaved pairs, main's build (control) against the branch head (candidate),
  median of 3, `b70-decode --bench --tg 32`: Ornith pp4096 and pp16384 on `l0` and `l0-int8`.
  - **Gate (final):** against today's 3,362 / 3,259 (l0) and 3,937 / 3,676 (l0-int8) - 4k l0 >= 1.7x (6,000 t/s),
    16k l0 >= 1.5x; l0-int8 no slower than l0 at either length. The derived target (~2x at 4k) is reported beside the
    measured one; a miss below the gate keeps the item open with the measured phase table.
- [ ] **Step 2 (box):** **Ornith decode unchanged:** no decode binary moved (`g0.sha`), `ornith_decode_test` PASS
  (replay bitwise, R2), plain decode at depth 4096 within 1 % of 142.8 t/s (interleaved pairs). **Other models:** G0
  (row 0) PASS against `--baseline main` - Qwen3.8 bitwise and every K2 / Kolibri / Bolt binary identical (their
  host code is untouched); the Ornith prefill gates (r13.gates, r13.split, r13.prefill) and Flash-Next's r33 rows PASS.
- [ ] **Step 3 (Mac):** delete the losing path and `B70_PF_MOE`: Ornith's `pf_moe_sort`, `pf_moe_dequant_*` and its
  three `pf_moe_gemm_K2048_N1024*` / `_K512_N2048` binaries (Ornith-only shapes); Flash-Next's `q4_pf_sort` and
  `q4_pf_dequant_*` if Task 7 passed. `pf_moe_gemm.cl` and `pf_moe.cl`'s gather / combine stay (K2 / Kolibri bind the
  GEMM source; the combine is unchanged). Then: BENCHMARKS "Ornith 1.5 MoE (spec 15)" prefill rows, the profile doc's
  "after" table (`B70_PREFILL_PROFILE=1` re-run), spec 15 "as built", open-work -> Closed recently. Commit; offer the
  ff-merge to main.

**Which tasks need the box:**

| task | Mac | box | numerics |
|---|---|---|---|
| 1 probe | step 1 | step 2 (card 0) | bitwise per arm |
| 2 sort | step 1 | step 2 | bitwise |
| 3 fused GEMM | step 1 | step 2 | bitwise |
| 4 tuning | step 2 | steps 1-2 | bitwise |
| 5 l0-int8 MoE | step 1 | step 2 | judged (D1) |
| 6 chunk 4096 | step 1 | step 2 | judged (D2 if < 5 %) |
| 7 Flash-Next | step 1 | step 2 | bitwise at 2048, judged at 4096 |
| 8 record | step 3 | steps 1-2 (idle) | - |

Test and benchmark runs (Task 1 step 2, the box steps' measurements) go to a Sonnet agent; implementation to Opus.

## Operator decisions

- **D1 (before Task 5):** may `l0-int8` run Ornith's (and any MoE model's) experts through the bf16 fused path? Its
  output changes (judged by the plain f32 reference); expected faster (~+270 ms saved at 4k, derived) and closer to
  f32. Recommendation: yes - l0-int8 then means int8 dense linears, bf16 experts.
- **D2 (after Task 6, only if 4096 gains < 5 %):** Ornith's default chunk cap 2048 or 4096 - folded into open-work
  item 5's choice of 4096-row chunks.
- **Stop rule of Task 1:** if no fused arm beats break-even, whether to probe a chunk cap of 8192 instead.

### Rulings (operator, 2026-11-12)

- **D1: yes, then revised the same day.** First ruling: l0-int8's experts run on the fused bf16 path, judged against
  the f32 reference. Revised with the new target (Ornith l0-int8 prefill >= 11,000 t/s at 4k on one card, ~370 ms for
  4096 tokens): **l0-int8's fused expert GEMM is W4A8** - int4 unpacked to int8 in registers (exact, no requant pass),
  int8 DPAS accumulated in int32 per 64-wide k group, then the g64 weight scale x the activation scale applied in fp32
  per group. The bf16 fused GEMM stays the l0 path and is l0-int8's fallback only if W4A8 fails the f32 judge. Task 1's
  probe gets a W4A8 arm under the same break-even rule. Implementer's note, sent to the operator: unrotated int4 weights
  cannot take Hadamard-rotated activations (x W = (H x)(H W), and H W is not int4), so the arm runs unrotated
  activations, per token and per token and 64-k group, and its error on real Ornith activations decides whether it
  reaches the engine (the 2026-09-23 study measured 2.8 % / 5.1 % for the per-token form on Qwen3.8).
  Agreed the same day, with one numerics-only arm (nothing built in the engine): the Hadamard folded offline into
  the expert weights (H^T W over gate||up's 2048 inputs and down's 512), re-quantised to int4 g64 by RTN, then W4A8
  with rotated int8 activations - it prices a rotated re-quantised checkpoint, which the operator decides on if it
  passes and the unrotated forms do not. Down may stay W4A16 (bf16) on its own if only gate||up passes.
- **D2: chunk cap 4096 for MoE models** if the f32 judge passes, even if the gain is under 5 %.
- **Scope:** Ornith and Flash-Next only. K2, Kolibri and Bolt stay on the old kernels, bitwise unchanged under G0.
- **Stretch:** the ~6.3k of this plan is the floor; the record names what is left, in ms a phase, toward 370 ms.

## Later (not in this plan)

K2 (100 experts top-8 plus MoVA, `k2_pf_moe.cl`), Kolibri (384 top-6, ushort sort, bf16 shared slot, `kol_pf_moe.cl`)
and Bolt (`nb_pf_moe_gemm.cl`'s relu2 epilogue, real rows only) each need their shapes built from `pf_moe_sort_mw.cl`
/ `pf_moe_q4gemm.cl` (Bolt also the relu2 epilogue), their bind in their prefill walk, and their own gates (bitwise
their grouped GEMM, then their prefill rows) - when those models are back in focus.

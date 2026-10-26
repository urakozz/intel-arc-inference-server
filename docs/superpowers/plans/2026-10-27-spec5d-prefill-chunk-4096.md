# Spec 5d - prefill chunk 4096: kC widened, the scratch through the planner, the chunk-bound sites, gates unchanged

**Status (2026-10-27): PLANNED - fourth in the Qwen speed track (after spec 28, plan 19f and spec 17a); starts when the
operator says go.** Nothing built. Box queue row **75** (spec 28's header lists the track's rows 68-75). Under spec 5
(the int8 prefill, `docs/superpowers/specs/2026-09-24-spec5-int8-prefill-linears-design.md` §10 points here).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the main engine (Qwen3.8 first; Agnes on the same code) prefills in chunks of up to **4096** positions on
`l0-int8` (and `l0`), with the per-chunk scratch sized from the session's chunk width through the memory planner, every
chunk-bound site moved from the constant 2048 to that width, the gates unchanged, and the speed measured against 2048.

## Why, and what to expect

| fact | value | grade | source |
|---|---|---|---|
| pp4096 today, chunk 2048 (two chunks) | **2125.12 t/s** (1927.4 ms) | measured | docs/BENCHMARKS.md "Flash attention and 128k" |
| "2048 is the widest this build runs"; the width sweep 512 / 1024 / 2048 is monotone (a fixed cost per chunk) | 932 / 1222 / 1406 t/s (bf16 era) | measured 2026-09-09 | docs/BENCHMARKS.md "Chunk width" |
| the h8 path rebuilds a rotated int8 slab of every linear **per chunk**; on gate‖up that requant is **1.32 of 4.44 ms (~30 %)** of the linear's time at 2048 rows | ~30 % | measured | docs/17-int8-prefill.md "What the speed is made of"; docs/probe-w4a8-2026-09-23.md §15.5 |
| a wider chunk amortises the requant: about 1.8x on gate‖up at 4096 | 1.8x | derived (probe-w4a8 §14.7, "what could still move it") | docs/probe-w4a8-2026-09-23.md |
| the per-chunk h8 linears, all 64 layers | ~602.5 ms of a 2048 chunk | derived (sums of measured kernels) | docs/probe-w4a8-2026-09-23.md §15.5 |
| ruling A13 (2048, not 4096) | taken on the **bf16** GEMM shapes of the time: four of six best at M = 2048, every shape but out / o_proj slower at 4096; the requant's doubling share weighed as nearly cancelling | measured then (bf16 / sycl-tla era); the int8 GEMM at 4096 rows never measured | `src/runtime/buffers.h` (the kC comment) |
| the lever's source | ranked second among the transferable levers in the 2026-10-27 review of vllm-sly-radiance (public), which runs 4096-token chunks | published by others | - |

**Expected: +8-15 % at pp4096 (estimated)** - the requant's share of linear time halves (~30 % -> ~15 %) and the
linears are ~60 % of the step at 4096 (1205 of 1927 ms, derived), so ~9 %; the other per-chunk fixed costs (the slab
walk's per-chunk dequant, launches, the floor of docs/17-int8-prefill.md) halve too, the upper end. A13's evidence is the
risk: if the int8 GEMM is slower per row at 4096, the gain shrinks or reverses - **Task 1 measures that before anything
is built**. Prompts shorter than 2048 are unchanged by construction (`C = min(chunk, remaining)`).

## The memory, through the planner

`PrefillScratch` is 699,514,776 B at kC 2048 (`tests/runtime/buffers_test.cc`'s table); every row but the last-position
logits / argmax and the conv seed scales with kC, so at 4096 it is ~1.398 GB: **+~0.70 GB** for Qwen3.8 (derived; the
brief's +~0.6 GB estimate is a little low). Beside it, scaling with kC: `Int8State::xq` int8 [kC][17408] +35.7 MB; MTP's
prefill hidden [kC + 1][5120] bf16 +21 MB; DFlash's tap staging [kC][5][5120] bf16 +105 MB (with a drafter). **~0.76 GB
a card with MTP (derived)**, ~0.86 GB with DFlash.

That memory comes out of the KV cache: **~12k positions at bf16 KV (64 KiB a position), ~24k at int8 KV** under
`--max-len auto` (derived). The operator's goal is long context, so the chunk is a **planner choice, not a constant**:
the scratch is sized by the session's `chunk_cap` (2048 or 4096), and the planner takes 4096 only by the rule the
operator rules on (open question 1 below). Every byte is recomputed, not doubled, in `buffers_test` (A13's rule).

## The chunk-bound sites (the list Task 2 works through; each gets the session's chunk_cap or a refusal by name)

- `src/runtime/buffer_sizes.h` (`PrefillScratchDims::kC` -> `kCMax = 4096` + the session's cap; `prefill_taps_bytes`,
  `mtp_prefill_hidden_bytes`, the MoE prefill layout's tile bound), `src/runtime/buffers.h` / `.cc` (`PrefillScratch`),
  the memory planner's prefill term (`runtime::memory_plan`, `--max-len auto`).
- `src/runtime/prefill/`: `engine_prefill.cc` (the chunk check, the default chunk), `int8.cc` / `int8.h` and
  `linear_l0.cc` / `.h` (pad256 <= cap), `attn.cc`, `gdn.cc` / `gdn.h` (kC / kGdnChunk = 64 blocks at 4096 - the grid
  and its occupancy note), `moe.cc`, `step.cc` / `step.h`, `pipeline_prefill.cc` (`ids` [kDepth][kC], the C list
  {1, 256, kC}), `src/runtime/pipeline_engine.h`.
- `src/runtime/engine.h` (the documented default; `mtp_pf_hid_`, `df_pf_taps_`), the CLIs (`b70_decode.cc`
  `--prefill-chunk` bound and help, `pipeline_decode.h`).
- Kernels: `pf_embed.cl` (comment only); **`pf_moe` bakes `KC = 2048` into its SLM sort staging** (Ornith): Ornith stays
  capped at 2048 (refused by name above it) unless a KC 4096 variant fits SLM - not in this plan.
- **The prefix cache's block hook:** `Engine::kBlock` = 2048 and `prefill_chunk_rows(..., hooked, kBlock)` cut every
  hooked chunk at the next 2048 block end, so **in `b70-serve` with the prefix cache on, chunk 4096 changes nothing**
  unless the hook changes (open question 2).
- Tests that pin 2048: `buffers_test`, `prefill_consistency_test`, `prefill_smoke_test`, `prefill_determinism_test`,
  `prefill_backend_equivalence_test`, `prefill_gate_test`, `gemm_batched_test`, `gdn_chunk_test`,
  `ornith_prefill_test`.
- Other engines (Bolt, Kolibri, K2, qwen4exp) have their own `kPfC` = 2048: out of scope.

## Gates (unchanged) and the speed bar

- **G0 at cap 2048:** with the cap at 2048 every binary and every token is today's (the default until Task 5).
- **The golden gates at chunk 4096** on `l0-int8` and `l0`: `prefill_gate_test` (the golden tie rule,
  docs/14-golden-gate.md), `prefill_consistency_test` (chunked against one-row ingest under the tie rule),
  `prefill_determinism_test` (bitwise run to run), `prefill_backend_equivalence_test`; A4 tool calls on `l0-int8` **>=
  25 / 36** (spec 5 §9's count); MTP's prefill head KV and DFlash's projection per position (bitwise across chunkings at
  multiples of 64, `dflash_engine_test` PF); `--pp 2` prefill (`pp_prefill` tests) at 4096.
- **Speed:** pp4096 / pp16384 / pp32768 / pp65536 at chunk 4096 against 2048, interleaved pairs after a warm-up, median
  of 3, idle box, device 0; pp512 / pp1024 within noise. **Go (4096 becomes the planner's choice by question 1's rule):
  pp4096 >= +5 % and no depth slower by more than 1 %**; expected +8-15 % (estimated). Below +5 %: 4096 stays opt-in
  (`--prefill-chunk 4096` with the cap) and the record says why.

## Open questions for the operator

1. **Context against speed:** (i) context first - `--max-len auto` plans at chunk 2048 and 4096 is used only when the
   memory is there anyway (an explicit `--max-len` below the auto value, or a model / KV form that already reaches
   262144); (ii) speed first - 4096 always, `--max-len auto` ~12k (bf16 KV) / ~24k (int8 KV) shorter. **Proposed: (i).**
2. **The prefix cache's granularity under 4096 chunks:** (A) keep `kBlock` 2048 for hashing and the KV ranges, take the
   state snapshot only at chunk ends that are block ends (every 4096 under a 4096 chunk); a hit at an odd block resumes
   from the previous state point and re-prefills <= 2048 positions (~1 s at worst, derived) - request-end snapshots,
   which agentic turns resume from, are unaffected; (B) keep the hooked path at 2048 (the gain only for cold, unhooked
   prefill - not the server); (C) `kBlock` 4096 (coarser reuse everywhere). **Proposed: (A)**, after Task 2 reads how
   `PrefixCache::reserve` / the lookup treat a block end without a state.

---

### Task 1: P0 - is the int8 linear path faster per row at 4096? (box, `r75.p0`)

- [ ] **Step 1:** `probe_w8a8` with `B70_W8A8_M=4096` (a 4096-row capture from `tools/probe/mk_capture.py`) against two
  2048-row calls, every shape of probe-w4a8 §15.5, h8 and the bf16 control, interleaved pairs, median of 5, idle. **Stop
  rule:** if the summed linear time a 4096-token prompt pays is not >= 5 % lower at M = 4096, stop and record (A13
  stands for the int8 path too). Record in `docs/probe-prefill-chunk-4096-<date>.md`. Commit
  `git commit -S -m "probe: the int8 prefill linears at 4096 rows against 2 x 2048 (plan 5d)"`.

### Task 2: the chunk cap through the code (Mac)

- [ ] **Step 1: failing tests** - `buffers_test` (the 4096 table recomputed row by row, the 2048 table unchanged),
  the planner's prefill term at both caps, `--prefill-chunk` bounds and refusals (Ornith above 2048 by name).
- [ ] **Step 2: implement** through the site list above; the cap 2048 default; `tools/mac_check.sh --base main
  --quick` green, cmdlines **~0** (no kernel line moves: the prefill kernels are shape-free in C, `pf_moe` keeps KC 2048).
  Commit `git commit -S -m "prefill: the chunk cap as a session value (2048 or 4096), the scratch through the planner (plan 5d)"`.

### Task 3: the prefix cache's hook at 4096 (Mac, by the operator's answer to question 2)

- [ ] **Step 1:** implement the ruled option with host tests (`prefix_cache` tests: a hit at an even and an odd block,
  a request-end restore) and the mock serving tests; commit
  `git commit -S -m "prefix: block-end snapshots under 4096-row chunks (plan 5d, option <A|B|C>)"`.

### Task 4: gates and speed on the card (box, `r75.*`)

- [ ] **Step 1:** the gates above at chunk 4096 (both backends), G0 at cap 2048.
- [ ] **Step 2:** the speed table; apply the bar. Commit `git commit -S -m "docs: prefill chunk 4096 - gates and speed (plan 5d)"`.

### Task 5: the default and the record

- [ ] **Step 1:** on the bar, the planner's rule of question 1 becomes the default; `src/runtime/buffers.h`'s A13 comment
  amended (the int8 path's measurement); BENCHMARKS "Chunk width" gains the 4096 row; spec 5 §10's result line;
  docs/19-running-models.md's `--max-len auto` row if it moved. Commit `git commit -S -m "docs: plan 5d as built"`.

### Task 6: the box queue row

- [ ] **Step 1:** row **75** and its block in `tools/box_validate/stages.sh` (`test_box_validate.py` passes):

```text
row 75 "plan 5d - prefill chunk 4096: the int8 linears at 4096 rows (P0), the chunk cap through the planner, the gates at 4096 (golden tie rule, consistency, determinism, A4 >= 25 / 36, MTP / DFlash prefill, --pp 2), speed against 2048"
stage r75.k0    75 default cpu  -     g0.sha,g0.bitwise,g0.suite "G0 at cap 2048: every binary and token as main"
stage r75.host  75 default cpu  -     -                     "buffers_test, the planner's prefill term, the CLI refusals, the prefix-cache hook tests"
stage r75.p0    75 default gpu  qwen  r75.k0                "probe_w8a8 at M 4096 against 2 x 2048, every shape, h8 and bf16; idle, interleaved, median of 5"
stage r75.gates 75 default card qwen  r75.k0                "prefill_gate / consistency / determinism / backend_equivalence at chunk 4096 (l0-int8, l0), A4 on l0-int8, MTP and DFlash prefill, pp_prefill"
stage r75.speed 75 optin   gpu  qwen  r75.gates             "pp4096 / 16384 / 32768 / 65536 (and 512 / 1024) at chunk 4096 vs 2048; idle, interleaved, median of 3"
rownote 75 "r75.p0 is the stop rule: if the int8 linears are not >= 5 % cheaper per 4096 tokens at 4096 rows, the rest of the plan stops. Hand back: the P0 table, the gate lines, the speed table."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: box queue row 75 - prefill chunk 4096 (plan 5d)"`.

**Gate for the plan:** host tests green and `tools/mac_check.sh --base main` PASS (cmdlines ~0); on the box P0 passes its
stop rule, every gate above passes at chunk 4096, and the speed table is recorded against its bar.

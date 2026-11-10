# Spec 17c - tensor parallel: Qwen3.8 prefill across two cards (the Bolt-first text is kept below)

## Review 2026-11-10 (against main 16fe2a14; nothing here is built - 17c starts when the owner has reviewed 17b)

Every task below the status lines is still Bolt's (`bolt_prefill*`, `nb_pf_*`, rows 61 / 68 / 70). Stale, and what
replaces it for Qwen3.8:

1. **The engine and the baselines.** Qwen3.8's prefill is `runtime::Engine::prefill` (src/runtime/prefill/*, the l0
   backend by default since sycl-tla was dropped; l0-int8 opt-in) and, on two cards, `PipelineEngine::prefill` (spec
   16c's chunk pipeline). The baselines are one card's prefill and `--parallel pp`'s, measured by `b70-decode --bench
   --prefill-length N` (rows 23 / 75), not `r61.speed` / `r68.prefill`. Chunk 4096 (plan 5d) and the 2048 / 4096 cap
   rule (`prefill_chunk_cap_that_fits`) apply per card.
2. **Bitwise.** The prefill GEMMs (pf_gemm*, the q4 / int8 tiles) accumulate a tile's whole K in one register chain.
   Splitting K over two cards changes that order, so TP prefill is **judged** against one card's prefill (golden tie
   rule / the f32 judge), not bitwise, whatever crossing form 17b ships. The column-parallel GEMMs and the per-head
   prefill kernels stay bitwise per column / head if they bind the same tiles. The **hybrid** (prefill as `--parallel pp`,
   then scatter the state into the TP layout) is bitwise one card's prefill, because pp is. It needs either both
   layouts resident or a reshuffle after prefill (memory; spec 17 §4.5).
3. **The options to measure** (Task 1): (W) the wavefront - chunk i + 1's layer l overlapping chunk i's all-reduce at
   layer l (external engine, spec 17 §R2.8); (T) plain TP prefill with the chunk's crossings on a copy engine; (H) the
   hybrid. The crossing at 2048 rows is 40 MiB fp32 (folded) or 80 MiB (slices) a call. The probe measured only M <= 8,
   so the first box step is that rate (P2P push, landing in the reader's VRAM, probe §3.3) overlapped and not.
4. **int8 KV prefill (3e).** `attn_chunk_kv8` widens the cache through `kv8_raw`, and buffer_sizes.cc:258 allocates it
   only at 24 / 4 heads. Per-card 12 / 2 needs that scratch and `pf_flash_attn_kv8_raw_variant(12, 2)`, plus the
   l0-int8 path's per-card column scales.
5. **The snapshot hook** (spec 7 §3.2) under TP fires when both cards have finished the block, and `save_state` /
   `save_kv` gather into the one-card layout (17b builds the gather). `--prefix-split-last` (the last id by one decode
   replay) works unchanged on 17b's decode.
6. **CLI.** No new flag: `--parallel tp --prefill` / `--bench --prefill-length` lift 17b's refusals.
7. **Box row 80** (79 is 17b's); every two-card stage kind `gpu`.

Operator decisions for 17c (in REPORT-17-review.md): the option (W / T / H) after the crossing-rate step; whether a
judged TP prefill is acceptable or the hybrid's bitwise property is required.

## As built 2026-11-10 (branch spec17c-tp; box row 80 run by hand under `gpu_lock.sh box`)

- [x] **Task 1 - the crossing rate (D8's first step).** `tools/probe/probe_tp_pf.cc`, record
  `docs/probes/probe-tp-prefill-2026-11-10.md`: a 2048-row fp32 crossing is 4.35 ms with both directions at once
  (~9.6 GB/s a direction - copy engine, compute-list copy and kernel push alike; pulls 1.8 GB/s); a cross-device event
  round trip 6.3 us; the copy engines' crossings and the compute lists' GEMMs do not slow each other.
- [x] **Task 2 - the option: the wavefront (W), judged** (D8 below). Code:
  - `runtime::prefill::step_tp_unit` / `step_tp_head` (step.cc / step.h): a TP card's chunk as 128 half-layer units;
    walk_layers' mixer and MLP bodies factored out (`mixer_body`, `mlp_body`; one card's launches and order unchanged),
    linear_l0 takes an optional output buffer (the row-parallel linears write each chunk's own half).
  - `src/kernels/prefill/pf_tp.cl`: `pf_tp_res_fold` (pf_res_fold with its slice replaced by card 0's half + card 1's).
  - The per-card prefill binaries (src/kernels/prefill/CMakeLists.txt "Spec 17c", `B70_TP_PREFILL_KERNELS`): five slab
    dequants, the GDN chain at 8 / 24 heads, the FA prep / gate / flash at 12 / 2, the int8 KV's three at 12 / 2.
  - `kv8.cl`: the flash path's head literals by name (`KV8F_*`; the default binaries byte-identical to main's, checked
    on the box) - the per-card flash read its scales and rows at 4 kv heads before, garbage ids under `--kv-cache int8`.
  - `buffer_sizes.cc`: the int8 widening scratch also at 12 / 2 heads (Review point 4).
  - `runtime::TpEngine::prefill` / `prepare_prefill` / `set_prefill_chunk_cap` (runtime/prefill/tp_prefill.cc, in
    b70_prefill_host as Engine::prefill is; `TpEngine::Card` moved to runtime/tp_engine_card.h): chunks of at most the
    cap (2048) in pairs (`runtime::tp_prefill_chunks`: a pair splits on the 64-grid, a 2048-id prompt is 1024 + 1024);
    each card appends the pair's units alternating chunk A / B, after each unit signals an event, its COPY-ENGINE list
    copies the card's half into the peer's landing slot (two a chunk, by unit parity) and signals; the next unit of that
    chunk waits for both cards' copies. Events: one pool (`l0::SyncEventPool`, 1024 events), reset between pairs; host
    waits bounded, on a miss every event is signalled, the engine throws by name and stays failed until reset(). The
    head: the last row's fold, the final norm, the int8 vocabulary half, tp.cl's argmax, the host merge (17b's
    `merge_heads`). No crossing for the last layer's down except the head's row.
  - The plan (`tp_card_plan` / `tp_max_len_that_fits` take a PrefillPath; `tp_prefill_extra_bytes`: 0.27 GB a card at
    cap 2048 beside the one-card scratch at the card's shapes) and the CLI: `--parallel tp --prefill` and `--bench
    --prefill-length N` run TP prefill (the pp row as one card's); `--prefill-backend l0-int8` refused by name (a card's
    down has K 8704, not whole 1024-k rotation blocks).
- [x] **Task 3 - box row 80** (tools/box_validate/stages.sh): r80.host, r80.prefill, r80.kv8, r80.decode, r80.cli, opt-in
  r80.speed; every two-card stage kind `gpu`. Run by hand in order (results below).

**Tests (box, 2026-11-10).**
- Host: tp_desc_test (+ the chunk pairs and the prefill plan), pipeline_args_test, cli_reject_tp_* pass.
- tp_prefill_test (bf16 KV): PASS. T1 bitwise between the cards on every case; T2 against one card's prefill: prose,
  code, long (2820), long32k[:6000] identical 64 ids; the two-call session a near-tie at id 10 (0.07 ulp). Replay
  bitwise.
- tp_prefill_kv8_test (int8 KV): prose, code, the two-call session identical; long32k[:6000] a near-tie (0.32 ulp);
  long diverged at id 62 (4.79 ulp, rank 1) - sent to its f32 judge (`f32_refs.sh forced-qwen`, key
  `tp_prefill_kv8_long_fa12c9167b2eaa28`, 4 min of CPU): see the row-80 result line below.
- Chunk invariance: KV bitwise and the first id equal at chunk 1024 vs 2048; the GDN state at the prompt end is not
  - and one card walking the same two chunk lists differs in the same ~2.49 MB, so it is a property of the GDN prefill
  (the end state depends on the last chunk's rows; one card's uniform cuts never vary them), not of TP. Open item.
- tp_decode_test / tp_decode_kv8_test / tp_fail_test (17b's decode): PASS.
- Smoke: `b70-decode --parallel tp --prefill --ids prose --n 24` = `--parallel 1 --prefill`'s ids.

**Speed** (`interleave.sh`, median of 3 after a warm-up, order reversed every other round, box shared with 3l between
our locked runs; spreads <= 0.3 %), `b70-decode --bench --prefill-length N --tg 32 --max-len 32768`:

| prompt ids | one card pp t/s | tp pp t/s | x one card | tg after it, one / tp |
|---|---|---|---|---|
| 2048 | 1746.32 | 2479.27 | **1.42** | 33.79 / 54.74 (1.62x) |
| 16384 | 1524.28 | 2672.68 | **1.75** | 31.64 / 50.24 (1.59x) |

Where the time goes (tp_prefill_test `time`, the timing seam TpOptions::prefill_timing_no_crossing): at 2048 ids 814 ms
with crossings, 714 ms without (compute alone; one card's half would be ~586 ms); at 16384 6159 vs 5756 ms. The
crossings hide all but 100 / 400 ms; the rest is a card's compute being ~22 % (2048, two 1024-row chunks) / ~7 % (16384)
above half of one card's.

**Decisions (built at the recommendation; the options in REPORT-17c.md):** D8 the wavefront with fp32 crossings on the
copy engines, judged (not the hybrid, not bf16 crossings, not in-order T); l0-int8 prefill refused under TP; the
snapshot hook under TP (spec 7 §3.2) left to 17d with the prefix cache; chunk cap 2048 (4096 by --prefill-chunk).

**Open:** the GDN end-state's dependence on the last chunk (one card's too); a card's compute overhead at 1024-row
chunks (levers: one 2048 chunk when the prompt is a single chunk and its crossings in two halves of rows; the
send fused into the down / out_proj epilogue); 17d (MTP, prefix cache and the hook, the server); a RECORD-grade re-run
of r80.speed on an idle box before a README line.

## The plan as written before the review (Bolt-first text kept for Bolt's later TP plan)


**Conditions (operator, 2026-10-23; for Bolt - superseded for Qwen3.8 by the 2026-10-27 status below):** this plan starts only after **plan 17b is merged and its row passed on the box**,
which itself needed spec 17 §9's GO (P0 after `r22.devices` / `r22.p1`; Bolt's rows 59-62 passed; the crossing cost
<= ~1 ms a token), **and after row 61's `r61.speed` has measured Bolt's two-card (`--pp 2`) prefill** - the choice
below is made against that number (spec 17 §4.5) - and the operator says go.

**Status (2026-10-27): PARKED -> after 17b (rewritten for Qwen3.8) has passed its row.** Retargeted to Qwen3.8 first
(spec 17 §R2): prefill under `--tp 2` for the main engine, the external two-B70 engine's chunked wavefront (one chunk's
all-reduce overlapped with the next chunk's compute) measured first, beside the hybrid (prefill as `--pp 2`, decode as
TP) and plain TP prefill; the baselines are one card and row 23's two-card `--pp 2` prefill (`r23.s2_*`) (spec 17
§R2.8, §4.5). Box queue row **73**. The 2026-10-23 Bolt-first text below is kept for Bolt's later TP plan.

**Status (2026-10-23, superseded 2026-10-27): PARKED - begins on the conditions above.** Rewritten 2026-10-23 for Bolt first (spec 17 §R);
the 2026-10-05 version (Qwen3.8, the PP-hybrid against spec 16c) is in git history.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** prefill under `b70-decode <bolt> --tp 2 --prefill` by the option Task 1 picks: **(T)** TP prefill - the
chunk's crossings (32 MiB fp32 a crossing at C = 2048, derived) on a copy engine overlapped with the next projection,
fp32 or bf16 by the accuracy gate; or **(H)** the hybrid - prefill as Bolt's `--pp 2` walk, decode as TP (each card
holds both layouts, or the layers are reshuffled after prefill; the TP planner decides by memory). T2 on prefill; D2.

**Architecture:** spec 17 §4.5. Bolt's `--pp 2` prefill is sequential (spec 25 §6.2), so TP prefill competes with one
card's compute rate; the decision uses `r61.speed` (measured) and P0's `r68.prefill` (the measured crossing rate,
overlapped and not).

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§4.5, §5 T2, §6 D2). Precedents: spec
25d (`BoltEngine::prefill`, the `nb_pf_*` walk, the 64-grid chunk rule), spec 16c (per-chunk resources doubled, the
back-pressure rule, bounded host waits).

## Dependencies and branch points

- Main with 17b merged; rows 61 (incl. `r61.speed`) and 69 passed. Branch **`spec17c-tp-prefill`** from main.

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` never committed;
  `oracle-out*` symlinked. No `rm -rf`.
- `--tp 1` unchanged (G0). The chunk rule unchanged (a multiple of 64 on the absolute 64-grid).
- **Every two-card stage is `gpu`**; idle box, interleaved pairs, median of 3.

## Review Focus

1. **After prefill each card holds exactly its decode-layout state** (its Mamba heads' SSM state and conv-ring channels,
   its kv head's KV), whichever option ran; T1 holds for the first decoded token.
2. **T2 on prefill:** against `--pp 2 --prefill` under spec 25's prefill bars (h taps, SSM state, logits cosine; routes
   with near-ties); chunks of 512 / 1024 / 2048 and splits at multiples of 64 bitwise each other under `--tp 2`.
3. **(T) the overlap is real:** per-chunk crossing time hidden behind compute, shown per block from timestamps.
4. **(H) memory:** both layouts (or the reshuffle's peak) fit beside KV at the target max_len; the plan shows it.
5. **The snapshot hook** (25e's) under `--tp 2` fires when both cards have finished the block.

---

### Task 1: the choice (Mac + the box's numbers)

- [ ] From `r61.speed` (Bolt `--pp 2` prefill, measured) and `r68.prefill` (the crossing rate), the derived (T) and (H)
  rates at 886 / 8192 / 32000 / 65536; the choice and a proposed D2 bar written into spec 17 §4.5. Commit
  `git commit -S -m "spec 17: Bolt's TP prefill option, from rows 61 and 68 (spec 17c)"`.

### Task 2: the chosen path (Mac)

**Files:** `src/runtime/bolt/bolt_prefill*.{h,cc}`, `src/kernels/bolt/nb_pf_*.cl` (per-card variants only),
`src/runtime/bolt/bolt_sizes.{h,cc}`, new `tests/prefill/bolt_tp_prefill_plan_test.cc` (host) and
`tests/prefill/bolt_tp_prefill_test.cc` (cards).

- [ ] Failing tests (Review Focus 1, 2, 4, 5), implement, PASS on the Mac's host half. Commit
  `git commit -S -m "prefill: Bolt prefill under --tp 2 (<option>) (spec 17c)"`.

### Task 3: the box queue row (proposed 70, or the next free row)

```text
row 70 "spec 17c - Bolt prefill under tensor parallel (plan 17c)"
stage r70.k0     70 default cpu -                     g0.sha,g0.bitwise,g0.suite "--tp 1 unchanged"
stage r70.host   70 default cpu -                     -          "bolt_tp_prefill_plan_test"
stage r70.synth  70 default gpu bolt_synth            r70.k0     "bolt_tp_prefill_test on the N 12 synthetics: Review Focus 1, 2, 5"
stage r70.golden 70 default gpu bolt_g128,oracle_bolt r70.synth  "SergiioB's g128: T2 on prefill against --pp 2"
stage r70.speed  70 optin   gpu bolt_g128             r70.golden "D2: --tp 2 vs --pp 2 prefill at 886 / 8192 / 32000 / 65536, idle, interleaved, median of 3; per-card busy time"
```
- [ ] Commit `git commit -S -m "docs: box queue row 70 - Bolt TP prefill (spec 17c)"`.

**Gate for the plan:** T2 on prefill green against `--pp 2`, the chunk / split bitwise checks, D2 measured, `--tp 1`
unchanged.

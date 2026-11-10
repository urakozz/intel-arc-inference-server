# Spec 17b - tensor parallel: Qwen3.8 decode across two cards (the Bolt-first text is kept below)

## Review 2026-11-10 (against main 16fe2a14; the plan is built from this section, not from the Bolt text below)

The 2026-10-27 status line retargeted this plan to Qwen3.8 but the tasks were never rewritten: every task below the
status lines is Bolt's (`BoltDesc`, `bolt_loader`, `nb_*` kernels, `--pp 2` as the reference, rows 69 / 72). What is
stale against today's code, and what replaces it:

1. **The engine is the main one, not Bolt's.** Qwen3.8 runs `runtime::Engine` / `runtime::capture` /
   `loader::load` on `model::ModelDesc`. The descriptor already carries per-model head counts, and the kernels already
   take them as defines with name suffixes (`_GK<k>V<v>`, `_Q<q>KV<kv>`, `_I<intermediate>`; spec 15c built Ornith that
   way). So a per-card TP descriptor (8 / 24 GDN heads, 12 / 2 FA heads, intermediate 8704, the head's 124160 rows)
   gives per-card shapes, buffer sizes and binary names through the existing code. TP is a new capture mode beside
   `Stage` (the crossing after out_proj / o_proj / down, its wait fused into the next residual fold), a new
   `runtime::TpEngine` beside `PipelineEngine` (one Level Zero context over both cards, one list per card), and a
   loader path that slices the checkpoint per card through the column map `load_linear` already builds
   (column-parallel: a sub-range of columns; row-parallel: the qweight / scales pointers offset by the card's K rows).
2. **The reference is one card, not `--pp 2`.** Qwen3.8 fits one card, and `--parallel pp` is bitwise one card anyway
   (spec 16 P1). T2 compares with `runtime::Engine` on card 0.
3. **Bitwise to one card is possible, which spec 17 §5 says it is not.** Per output, at M = 1:
   - column-parallel linears (qkv‖z, a‖b, q‖k‖v, gate‖up, the lm_head rows): bitwise per column. Same K, same S, the
     same per-column reduction in gemv.cl / gemv_bf16.cl / gemv_i8w.cl, and the per-card weights are byte subsets of
     the one-card repack.
   - per-head kernels (gdn_step and its conv ring, prep_gated_head, attn_prep, decode attention v2 or its kv8 twin,
     the reduce) and prep_silu_mul (per element): bitwise per head or element.
   - row-parallel linears plus the residual fold (out_proj, o_proj, down into `prep_res_fold`): one card sums the
     split-K slices in order, `((p0 + p1) + p2) + p3` (prep.cl). **Fold-then-send** (the probe's form: each card folds
     its own slices and sends one fp32 row, and the reader adds card 0's row then card 1's) gives
     `(p0 + p1) + (p2 + p3)`. That is not one card's order, so this form **needs the f32 judge**. Both cards still agree
     bitwise with each other (T1). **Slice exchange** (each card runs the row-parallel GEMV at the one-card S / 2 over
     the one-card slice boundaries, sends its raw slices, and the fold sums all four in one-card order) is **bitwise
     one card**. It costs about 2x the crossing bytes and half the work-groups on those three GEMVs. Derived: about
     +0.4 ms a token for the crossing and +0.6 to 1.0 ms for the GEMVs, so roughly 52 instead of 56 t/s. Both forms are
     one kernel pair with defines. The plan builds both and measures both (operator decision D1).
   - the RMS norms (prep_norm_finish), the embedding and argmax: bitwise, because the residual is bitwise on both cards
     (and to one card in the slice form) and the argmax tie rule (lower id) is bracket-free.
   - MTP verify at M >= 2 (17d): gemm_sm rows are matched on exact (K, N, S, layout). Per-card rows with the same knobs
     would keep the slice form bitwise one card's verify too.
4. **The crossing is route (P) only, and it lands in the reader's own VRAM** (probe §3.3: a card never sees a flag
   change in memory it does not own). The flag is a plain store plus a system-scope release fence, not the atomic
   store `pp_send` uses (1.6 us a call slower). Each wait is bounded. Once one wait has failed, every later wait in the
   list skips its spin, so a lost card costs one bound, not 128. The host reads the state words after the fences and
   throws by name, and the engine stays failed until `reset()`.
5. **The head exchange is a host merge** (operator decision D2). Each card's argmax over its vocabulary half writes its
   global id and value into its Control (`out_value`, carved from Control's pad). After both fences the host picks the
   larger value (ties to the lower id, which is card 0's) and writes the id into both Controls, as `--parallel pp`
   already mirrors Control every token. That is no device crossing and no extra wait.
6. **The CLI is zero-config.** `--tp N` / `--tensor-parallel-size` are void. TP is the existing `--parallel` flag's
   `tp` value (`parse_parallel` refuses it today by name). `--pipeline-split` / `--pipeline-handoff` belong to pp and
   are refused with `tp`. replica is serve-only and refused in b70-decode anyway. Two visible B70s and peer access 0 -> 1
   and 1 -> 0 are required, refused by name before the load. `--parallel auto` is unchanged in 17b (decision D3).
7. **Other parallel modes.** replica (13b) and TP both take both cards, and the single `--parallel` value makes them
   exclusive. Slots (13c) are one-card only (b70_serve.cc chooses them after the pp / replica paths return). TP plus
   slots is out of scope for spec 17 (§8). The M-row crossing that would carry them exists (decision D4).
8. **int8 KV.** The default is still bf16 (`kDefaultKvCache`), with `--kv-cache int8` as the option. TP builds both.
   `kv8.cl` hard-codes 24 q / 4 kv heads (lines 92-99). 17b adds an `FA_Q_HEADS` / `FA_KV_HEADS` override like attn.cl's,
   which leaves the default token stream unchanged. 3f / 3h's plain-decode reader changes live in that file, so the
   per-card binaries inherit them. 3e's prefill widening (`kv8_raw`, allocated only at 24 / 4 heads) is 17c's.
   `attn_verify_dpas.cl` `#error`s on any other heads, and 3l (branch verify-attn-3l) is rewriting it now. The per-card
   DPAS verify is 17d's, after 3l merges.
9. **17b is plain decode at M = 1.** MTP, DFlash, lookup, AdaptiveK, the prefix cache, `--prefix-split-last` and serving
   are 17d's. Prefill is 17c's, so under `--parallel tp` the prompt is ingested by decode replays and `--prefill` /
   `--prefill-length` are refused by name (decision D5). `--bench --depth N` ingests by replays on one card too, so the
   arms stay like for like.
10. **Snapshots.** `save_state` / `save_kv` in the one-card host layout (the gather) move into 17b, test-only in use,
    because T1 / T2 compare state and KV with one card through them. The restore (scatter) and the prefix cache stay
    17d's.
11. **The box.** Rows 69 / 72 are taken. The next free row is **79** (17c 80, 17d 81 when built). Every TP stage is
    kind `gpu` (`gpu_lock.sh box`); kind `card` forbids two-card work. `remote.sh` runs no prerequisite closure: a stage
    runs only when its AFTER stages are PASS in the state, so run the row's stages in order (`--only r79` plus `--with`
    for the opt-in speed stage). Builds go through `tools/box.sh build` from this worktree (tree `~/b70/trees/spec17b`).
12. **Speed bar.** D1-dense plain is `--parallel tp` against `--parallel 1` (>= 1.35x, the operator's 2026-10-27 bar),
    interleaved (tools/box_validate/interleave.sh), median of 3, at depth 4096 and a long context. The `--spec auto`
    bar (>= 1.25x, the spelling that replaced `--mtp auto`) is 17d's.
13. **Later levers, recorded but not in 17b:** the send fused into the row-parallel GEMV's epilogue (saber-engine,
    spec 17 §10; 17b ships the send as its own launch, +128 a token, and the wait fused into the fold) and the qkv‖z
    per-card cell re-tune (S2 or layout 0, probe §6, about 0.8 ms a token).

### Tasks (Qwen3.8, 2026-11-10) - these replace Tasks 1-4 below for this plan

- **Task 1 - the TP view, the slices, the plan (host tests).** `model::tp_card_desc(one, crossing)` (per-card widths,
  table, small tables; doc_w 0, so the W check is void per card and the test sums the two cards instead);
  `loader::load(..., TpSlice{rank, size})` (column / row slicing of every linear, the GDN small tensors per channel and
  head, a‖b's rows, the head's rows); `runtime::tp_plan` (per-card bytes, `--max-len auto` from the tighter card).
  Tests: `tp_desc_test` (shapes against spec 17 §R2.5, both crossings), `tp_slice_test` (a synthetic GPTQ checkpoint
  sliced per card: every card column / row is the one-card repack's, both cards together are the whole), `tp_plan_test`.
- **Task 2 - the crossing and the per-card lists.** `src/kernels/tp.cl` (`tp_send`, `tp_res_fold`, the vocabulary-half
  argmax pair), `src/runtime/tp_protocol.h` (the host mirror, promoted from tools/probe with slices and the
  skip-after-failure rule) and `tp_protocol_test` on host threads, the capture's `Tp` mode, `runtime::TpEngine`, the CMake
  block of per-card binaries (bf16 and int8 KV), and kv8.cl's and prep.cl's additive head / width overrides. Box tests:
  `tp_kernels_test` (the fold and send against `prep_res_fold` bitwise, both forms), `tp_decode_test` (T1 bitwise
  between the cards; T2 bitwise one card in the slice form, judged in the folded form; replay determinism),
  `tp_fail_test` (T4: a dropped list, a stale stamp, reset() recovers).
- **Task 3 - the CLI.** `b70-decode --parallel tp` (cli/tp_decode.h), its refusals before any device, `cli_reject_tp_*`.
- **Task 4 - box row 79**, below.

```text
row 79 "spec 17b - Qwen3.8 decode under tensor parallel (--parallel tp, plan 17b)"
stage r79.host   79 default cpu -                g0.sha    "tp_desc_test, tp_slice_test, tp_plan_test, tp_protocol_test, pipeline_args_test"
stage r79.k1     79 default gpu -                r79.host  "tp_kernels_test on both cards: the TP fold = prep_res_fold bitwise (slices), the fixed-order sum (folded), stale / dropped caught"
stage r79.decode 79 default gpu qwen             r79.k1    "tp_decode_test (bf16 KV, int8 head): T1 bitwise between the cards, T2 vs one card (bitwise slices, judged folded), replay"
stage r79.kv8    79 default gpu qwen             r79.decode "tp_decode_kv8_test: the same over --kv-cache int8"
stage r79.fail   79 default gpu qwen             r79.k1    "tp_fail_test: T4 - a dropped list, a crossing timeout, reset() recovers"
stage r79.cli    79 default gpu qwen             r79.decode "b70-decode --parallel tp --ids: ids equal --parallel 1's (slices); refusals"
stage r79.speed  79 optin   gpu qwen             r79.decode "D1-dense plain: --parallel tp (both crossings) vs --parallel 1, depth 4096 and 16384, interleaved, median of 3"
```

### Operator decisions (17b; the built default is the recommendation, each one a constant or one line)

D1 crossing form, D2 head exchange, D3 `--parallel auto`, D4 TP plus slots, D5 the prompt path under 17b,
D6 row numbers - the options and recommendations are in `.claude/worktrees/spec17b-tp/REPORT-17-review.md` (untracked)
and the 17b record.


## As built 2026-11-10 (branch spec17b-tp; box row 79 run by hand under `gpu_lock.sh box`)

- **Code.**
  - `model::tp_card_desc` (model_desc.*) is the per-card descriptor: 8 / 24 GDN heads, 12 / 2 FA heads, MLP 8704, the
    head's 124160 rows.
  - `loader::load_tp` (loader.*) slices each card straight from the checkpoint: about 9.7 GB a card, the two cards'
    int4 bytes summing to one card's.
  - `src/kernels/tp.cl`: `tp_send`, `tp_res_fold` (the wait fused into the fold) and the vocabulary-half argmax. The
    CMake block "Spec 17b" builds the per-card binaries, plus additive overrides in kv8.cl (FA heads) and prep.cl
    (INTERMEDIATE 8704). Main's kv8.cl and the edited one give byte-identical binaries (checked on the box).
  - `runtime::build_tp`, the capture's Tp mode: 902 launches a card.
  - `runtime::TpEngine` (tp_engine.*): the host merge of the head (D2), bounded waits, the gather into the one-card
    snapshot layout (tp_gather.h).
  - The plan (tp_plan.*), `b70-decode --parallel tp` (cli/tp_decode.h), and b70-serve's refusal until 17d.
- **Tests.**
  - Host: tp_desc_test, tp_slice_test, tp_protocol_test and pipeline_args_test pass.
  - Box: tp_kernels_test passes (300 replays: slices bitwise prep_res_fold, folded equal to the host's sum).
  - tp_decode_test, bf16 and int8 KV:
    - slices: T1 bitwise and T2 **BITWISE one card** (64 ids, logits, residual taps, 166.7 MB of state, the KV), and
      the same again after reset;
    - folded: T1 bitwise; T2 identical ids against one card (judged by the golden tie rule, not needed); logits not
      bitwise (another summation order, as expected); the run again after reset is bitwise the first.
  - tp_fail_test: each card dropped once throws in ~65 ms, naming the timeout, and reset() recovers.
  - `b70-decode --parallel tp --ids prose --n 64` gives the same ids as `--parallel 1`.
- **Speed** (docs/BENCHMARKS.md "Tensor parallel (spec 17)"), tg 256, interleaved, median of 3:

  | depth | one card | slices | folded |
  |---|---|---|---|
  | 4096 | 33.49 t/s | 49.99 t/s (1.49x) | 53.95 t/s (1.61x) |
  | 16384 | 31.65 t/s | 47.24 t/s (1.49x) | 50.80 t/s (1.61x) |

  The D1-dense bar (>= 1.35x) holds for both crossings.
- **D1 applied by its rule.** Slices are 7.3 % slower than folded, more than the ~5 % allowed, so
  `runtime::kTpDefaultCrossing` is Folded. Slices stays built and tested; switching back is one word.
- **Open (17c / 17d and the record):**
  - prefill under TP;
  - MTP / DFlash / lookup, the restore and the server;
  - the send fused into the GEMV epilogue;
  - the qkv‖z per-card cell;
  - a RECORD-grade (idle box) re-run of r79.speed before any README line.

## The plan as written before the review (Bolt-first text kept for Bolt's later TP plan; its Tasks 1-4 are NOT this plan's)


**Conditions (operator, 2026-10-23; for Bolt - superseded for Qwen3.8 by the 2026-10-27 status below):** this plan starts only when **spec 17 §9 records GO** - P0 (plan 17a) ran after
`r22.devices` / `r22.p1`, **Bolt's rows 59-62 passed**, and P0 measured one Bolt token's crossings at **<= ~1 ms**
(spec 17 §3) - and the operator says go (one implementing plan at a time).

**Status (2026-10-27): PARKED -> next after spec 28 (28a / 28b), spec 19's plan 19f and spec 17a's GO.** Retargeted
to Qwen3.8 first (spec 17 §R2): this plan is rewritten for the main engine - `b70-decode <qwen3.8> --tp 2` (Agnes, once next, dropped 2026-11-07):
spec 17 §R2.5's split re-fused per card at load, the all-reduce on the route 17a chooses, the vocabulary-split head, the
verify lists on spec 28's `gemm_sm` at per-card shapes - when 17a records GO and the operator says go. Box queue row
**72** (the speed track's rows are 68-75, spec 28's header). The 2026-10-23 Bolt-first text below is kept for Bolt's later
TP plan (Bolt has no drafter since 2026-10-27).

**Status (2026-10-23, superseded 2026-10-27): PARKED - begins on the conditions above.** Rewritten 2026-10-23 for Bolt first (spec 17 §R);
the 2026-10-05 version (Qwen3.8 in the main engine) is in git history. The dense models are not in this plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `b70-decode <bolt> --tp 2`: each card holds its slice of every block (spec 17 §4.2) and runs its own captured
list at per-card shapes; the crossings use P0's chosen mechanism and direction in a fixed fold order so both cards
hold a bitwise-identical residual; the MoE by P0's scheme (spec 17 §4.3); the head split by vocabulary with the argmax
exchange; T1, T2, T4; D1-Bolt, D3.

**Architecture:** spec 17 §4.1-§4.6, §4.9, §4.10. `model::BoltDesc` gains a `tp_rank` / `tp_size` view deriving every
per-card shape; `loader::load_bolt` gains a TP placement (both devices hold every block, each its re-fused slice -
`in_proj`'s five column ranges, `q‖k‖v`, `fc1‖shared_up` with the replicated fc1 part - and the replicated router,
`fc1` / `fc2`, embedding, `norm_f`); `runtime::bolt::tp_plan` beside `runtime::bolt::plan` (plan == allocation);
`BoltEngine` with one list per card, the crossing landing buffers (page-aligned, their own allocations), the fold
kernel's TP form (`nb_res_norm` remote sources or the fold-then-send epilogue, per P0), the routed part of the MoE
crossing.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§4.1-§4.6, §4.9, §4.10, §5 T1 / T2 / T4,
§6 D1-Bolt / D3, §9 the verdict). Precedents: spec 16b (one context over both cards, `pp_send` / `pp_recv`, bounded
fences), spec 25b / 25c (the loader, `BoltEngine`, the `nb_*` family, B3's bars).

## Dependencies and branch points

- Main with 17a's verdict (GO) merged. Branch **`spec17b-tp-decode`** from main.
- Box: rows 59-62 PASSed; `bolt_g128` and `oracle_bolt` present; the N 12 synthetics (`bolt_synth`).

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` copied if missing, never
  committed; `oracle-out*` symlinked. No `rm -rf`.
- **`--tp 1` is today's Bolt engine bit for bit:** every existing kernel binary unchanged (`tools/kernel_cmdlines`:
  additions only; G0's sha). Per-card shape variants are new binaries.
- `--tp 2` with `--pp 2`, `--replica`, `--pipeline-split` or `--pipeline-handoff` refused by name before any device.
- **Every two-card stage is `gpu`** (`tools/gpu_lock.sh box`); host tests `cpu`. Timed arms: idle box, interleaved pairs
  after a warm-up, median of 3; device 1's gap recorded.
- Shapes only from `BoltDesc`; no Bolt constant in a kernel or a planner.

## Review Focus

1. **Slicing on boundaries:** a host test slices synthetic packed tensors and checks each card's re-fused slice
   dequantises to exactly the rows / columns spec 17 §4.2 names (heads 0-63 / 64-127; B / C groups 0-3 / 4-7; kv head
   0 / 1; shared-expert halves; every K slice a whole number of g64 groups; EP's expert halves or intra-expert's I / 2).
2. **T1:** after every block the two cards' residuals and MoE route sets byte-identical (a debug tap compares them),
   golden prompts + 64 tokens; a route mismatch fails.
3. **T2:** against `--pp 2` (bitwise one card's order) under the tie-aware gate and B3's bars (routes with near-ties,
   the SSM state / h cosines), short / 4k / agentic.
4. **The argmax exchange:** the same id on both cards, ties to the lower id.
5. **T4:** a missing card, no peer access, a crossing timeout, a stale stamp: clear errors within the bound; the
   engine marked failed until `reset()`.

---

### Task 1: the TP view, the slices and the plan (Mac)

**Files:** `src/model/bolt.{h,cc}` (the TP view), `src/loader/bolt_loader.{h,cc}`, `src/loader/bolt_repack.{h,cc}`,
`src/loader/bolt_layout.h`, `src/runtime/bolt/bolt_sizes.{h,cc}`, new `tests/loader/bolt_tp_slice_test.cc`,
`tests/runtime/bolt_tp_plan_test.cc`.

- [ ] **Step 1: failing tests** (Review Focus 1; the plan's bytes per card - spec 17 §4.6's ~22.5 GB derived - and
  `--max-len auto` from the tighter card).
- [ ] **Step 2: implement, PASS.** Commit `git commit -S -m "loader: Bolt sliced per card at load, the TP plan (spec 17b)"`.

### Task 2: the crossing and the per-card lists (Mac)

**Files:** `src/kernels/bolt/nb_prep.cl` (the fold's TP form), `src/kernels/bolt/nb_moe.cl` (the routed part of the
crossing), `src/kernels/CMakeLists.txt` (per-card variants), `src/runtime/bolt/bolt_capture.{h,cc}`,
`src/runtime/bolt/bolt_engine.{h,cc}`, `src/runtime/bolt/bolt_buffers.{h,cc}`, new `tests/runtime/bolt_tp_protocol_test.cc`
(two host threads), `tests/kernels/bolt_tp_kernels_test.cc` (the twin: the TP fold = the one-card fold's sum in the
fixed order).

- [ ] **Step 1: failing tests**, then implement; `tools/box_check.sh --base main` (the new variants on the
  Mac's OpenCL, INDICATIVE). Commit `git commit -S -m "runtime: Bolt decode across two cards - the crossing, the per-card lists, the head exchange (spec 17b)"`.

### Task 3: the CLI and the refusals (Mac)

- [ ] `b70-decode <bolt> --tp 1|2` (`--tensor-parallel-size`), the exclusivity refusals, `cli_reject_tp_*` tests.
  Commit `git commit -S -m "cli: b70-decode <bolt> --tp 2 and its refusals (spec 17b)"`.

### Task 4: the box queue row (proposed 69, or the next free row)

- [ ] Row and stages (`test_box_validate.py` passes):

```text
row 69 "spec 17b - Bolt decode under tensor parallel (plan 17b)"
stage r69.k0      69 default cpu  -                    g0.sha,g0.bitwise,g0.suite "D3: --tp 1 is today's engine - every existing binary identical"
stage r69.host    69 default cpu  -                    -          "bolt_tp_slice_test, bolt_tp_plan_test, bolt_tp_protocol_test, cli_reject_tp_*"
stage r69.k1      69 default gpu  -                    r69.k0     "bolt_tp_kernels_test on both cards: the TP fold = the fixed-order sum, the routed crossing both forms"
stage r69.synth   69 default gpu  bolt_synth           r69.k1     "bolt_tp_decode_test on the N 12 synthetics: plan == allocation per card, T1 per block, T2 vs --pp 2"
stage r69.p4      69 default gpu  bolt_synth           r69.synth  "T4: missing card, no peer access, a dropped crossing, a stale stamp - bounded, named"
stage r69.golden  69 default gpu  bolt_g128,oracle_bolt r69.synth "SergiioB's g128 under --tp 2: T1, T2 (B3 short / 4k / agentic) against --pp 2"
stage r69.speed   69 optin   gpu  bolt_g128            r69.golden "D1-Bolt: --tp 2 vs --pp 2 at depth 886 / 8192 / 32000 / 65536, idle, interleaved, median of 3; bar >= 1.4x at 8192 (proposed)"
rownote 69 "After spec 17 §9's GO. Every two-card stage is gpu."
```
- [ ] Commit `git commit -S -m "docs: box queue row 69 - Bolt TP decode (spec 17b)"`.

**Gate for the plan:** host tests green; on the box T1 bitwise, T2 green against `--pp 2`, T4 bounded, D3 (G0), D1-Bolt
measured and recorded in docs/BENCHMARKS.md "Tensor parallel (spec 17)".

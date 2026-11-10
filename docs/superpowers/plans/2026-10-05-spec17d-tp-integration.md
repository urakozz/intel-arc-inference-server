# Spec 17d - tensor parallel: Qwen3.8 speculation, snapshots, sampling, the server, the record (the Bolt-first text is kept below)

**Status (2026-11-10): BUILT on branch spec17d-tp** (not merged) - see "As built".

## As built 2026-11-10 (branch spec17d-tp; box row 81 run by hand under `gpu_lock.sh box`)

Review items 2, 4-9 built; item 3 (DFlash under TP) is not - refused by name (decision D12 below).

- [x] **MTP under TP (item 2).** The MTP head loads WHOLE on each card (`loader::load_tp(..., mtp)`: one card's shapes,
  0.85 GB a card) - both cards hold the same residual, so both run the same head on the same input, no crossing, and
  keep the same head KV (MtpBuffers' TP form: the GDN slots at the card's shapes, the head's KV layer at its own 24 / 4
  heads). Code:
  - `runtime::build_tp_verify` / `build_tp_draft` (capture.cc's `TpVerify` / `TpDraft` modes): the TP walk at M rows
    (17b's crossings at M rows, the per-row GDN slots at the card's heads, tp.cl's argmax per row) plus spec 8's tail
    at the head's shapes (`TpLink::head`, checked to fit the card's decode scratch at M <= 4); the draft list's lm_head
    is the card's int8 vocabulary half and tp.cl's argmax, the host merges the two winners (D2's rule) before the next
    draft list. 912 launches a verify list, 23 a draft list.
  - `TpEngine`: draft / verify / commit / mtp_step1 (ingest and generate run verify(0) + commit with the head, as
    Engine), spec 28c's deep lists (DPAS by depth), max_verify_k, set_draft_input, the T4 drop on any list.
  - Per-card binaries (src/kernels/CMakeLists.txt "Spec 17d", the folded crossing): gdn_step_slots_M1..4_GK8V24,
    prep_gated_head / prep_silu_mul (I8704) / attn_prep / attn_v2 (+ _KV8) at M = 2..4, attn_verify_dpas at 12 / 2
    (the kernel takes FA_Q_HEADS / FA_KV_HEADS with GQA 6; its default binaries byte-identical to main's, checked by
    compiling both sources on the box), tp_cross / tp_argmax at M = 2..4, and six gemm_sm rows at the per-card shapes
    (kSmallMTuning, kSmVerify4; gate||up at grid S 8). verify_attn.h's table applies to 12 / 2 (verify_attn_heads).
- [x] **AdaptiveK (item 4).** `MtpCost::tp_int8_head()` measured by `tp_mtp_test cost` (medians of 5 interleaved
  rounds): verify 1.00 / 1.09 / 1.13 / 1.20, drafts 0.16 / 0.32 / 0.48 of the K = 0 iteration (18.93 ms at depth 4096);
  at 32768 1.00 / 1.01 / 1.07 / 1.13 and 0.15 a draft. b70-serve and b70-decode use it under TP.
- [x] **Sampling (item 5, D9).** `TpEngine::read_logits_into` / `read_draft_logits_into` gather both cards' halves of
  each row into the full row (2 x 0.5 MB a row); the sampler is one card's (PipelineEngineAdapterT). tp_mtp_test holds
  every verify row's gathered argmax to the merged id and a draft picked from the gathered row to the merged draft.
- [x] **The server (item 6).** `b70-serve --parallel tp` (serve_tp): spec 16d's `PipelineEngineAdapterT<TpEngine>` -
  `--spec off / auto / mtp / lookup`, host sampling, the prefix cache. Refused by name: DFlash, a draft vocabulary, the
  l0-int8 prefill, other models (cli::check_tp, check_tp_model).
- [x] **Snapshots / prefix cache (item 7).** `TpEngine::load_state` / `load_kv` scatter the one-card host layouts
  (tp_gather.h: `scatter_state`, `scatter_kv`, the head's KV layer spliced in after the FA layers - Engine's MTP
  layout), save_state takes the live GDN slot and the head's hidden row. Prefill's block hook under TP: a pair never
  straddles the hook's stride (`tp_hook_stride`: 2 x the chunk cap rounded to blocks, 4096 at cap 2048 - plan 5d's
  rule (A)), the hook runs after such a pair and after the prompt. With the head, TP prefill crosses the last unit's
  every row and each card runs the head's KV per chunk (`prefill::step_tp_final` + `step_tp_mtp_kv`, in passes of
  cap / 2 rows the card's scratch holds, a [10240][1024] head slab).
- [x] **D3 (item 8).** `--parallel auto` in b70-serve picks tp for Qwen3.8 on two B70s (peer access: the hardware gate)
  with `--max-len auto` and nothing TP refuses (cli::tp_auto); an asked --max-len keeps one card / pp (its spare memory
  may be sequence slots). b70-decode's auto is unchanged (decision D13).
- [x] **Box row 81** (tools/box_validate/stages.sh): r81.host, r81.kernels, r81.mtp, r81.kv8, r81.prior, r81.cli,
  r81.serve, opt-in r81.cost and r81.speed (tools/box_validate/depth_prompts.py).

**Tests (box, 2026-11-10).** Host: tp_slice_test (+ scatter, splice), verify_attn_test (12 / 2), gemm_sm_names_test,
pipeline_args_test, tp_desc_test, cli_reject_tp_* pass. gemm_sm_test on the six per-card rows: PASS (N2 vs fp64, vs
gemv, property R). tp_mtp_test, bf16 KV: M1 (plain ids with the head) identical to plain TP on prose / code / long /
long32k[:6000]; M2 at K = 1..3 identical on prose, code, long32k[:6000] (acceptance at K = 3 0.39 / 0.85 / 0.65);
on `long` every K diverges at id 62 (198 -> 271, 23 ulp, rank 1) - the f32 judge of that exact context (17c's D10 file,
the same 64 ids) REJECTS 271 (decision D11). int8 KV: the mirror image - plain TP takes 271 there (17c's D10 row), MTP
takes 198 (the f32 top; its judge being made). T1 the head bitwise between the cards; D9 gathers exact; S1 the hook at
4096 and 6000, restore bitwise (state, KV, 64 ids); S2 one card's snapshot into TP and TP's into one card: identical or
a near-tie (0.22 / 0.86 ulp); T4 a dropped verify step throws in ~0.8 s, reset() recovers. 17b / 17c's tests unchanged
(tp_kernels / tp_decode / tp_fail / tp_prefill PASS). The teacher-forced verify-rows probe (`rows` / `rows1`): TP's
verify rows drift from its plain rows exactly as one card's do (max |dlogit| up to ~9 vs ~13 over 30 steps) - the
id-62 flip is the verify path's rounding landing on a sensitive row, not a TP fault. b70-serve (golden + tool-call
sets, 39 requests, prefix cache on): `--parallel tp --spec off / auto` and one card's `--spec auto` against one card's
`--spec off` - 0 FAIL by the f32 judge, the rest identical or awaiting references (SERVE_JUDGE_TBD).

**Speed** (the bar: single-user serve decode with `--spec auto`, TP faster than one card at depth 4096 and 32768).
`b70-serve --spec auto --prefix-cache-gb 0`, 256 ids, decode-only t/s, a warm-up request a server, 3 rounds in rotated
order, medians (docs/BENCHMARKS.md "Speculative decoding under tensor parallel"):

| depth (prompt ids) | one card `--spec auto` t/s | tp `--spec auto` t/s | x one card | acceptance one / tp |
|---|---|---|---|---|
| 4096 (4063) | 62.12 | 90.39 | **1.455** | (both: K 3 throughout) |
| 32768 (31821) | 62.36 | 91.68 | **1.470** | 0.707 / 0.695 |

Rounds: one card 62.21 / 62.12 / 62.01 and 62.36 / 62.21 / 62.36, tp 90.39 / 90.39 / 90.51 and 91.68 / 91.66 / 91.78 (spread
under 0.3 %); load average 2.6 - 9 (the box otherwise idle; a first run beside a CPU reference job, prompts of 5199 /
40046 ids, gave 1.451x / 1.458x).

At `--ids long` through b70-decode (2820-id prompt, 64 ids, K 3): one card 86.9 t/s, tp 110.3 t/s (1.27x; plain 33.8 /
54.0). The bar holds at both depths: **1.46x / 1.47x**.

## Review 2026-11-10 (against main 16fe2a14; nothing here is built - 17d starts after 17c)

Every task below the status lines is Bolt's (`BoltEngine` snapshots, LiLiCorr, `b70-serve <bolt> --tp 2`, row 71).
Stale, and what replaces it for Qwen3.8:

1. **Task 2 (LiLiCorr under TP) is out of scope**: LiLiCorr was Bolt's drafter (spec 26, dropped 2026-10-27).
   Qwen3.8's proposers under TP are the MTP head, the DFlash drafter and prompt lookup (`--spec` values today: auto,
   off, mtp, lookup, dflash, dflash-lookup; `--mtp auto` is now `--spec auto`, b70-serve's default).
2. **MTP verify on the per-card shapes.** gemm_sm rows are matched on exact (K, N, S, layout) (`kSmallMTuning`), so each
   per-card shape needs its own row and binaries at M 2..8. With 17b's slice form and the one-card knobs, verify stays
   bitwise one card's verify. The DPAS verify attention (`attn_verify_dpas.cl`) `#error`s on any heads but 24 / 4, and
   3l (branch verify-attn-3l) is rewriting it now: the 12 / 2 build comes after 3l merges. The crossing at M rows uses
   17b's kernels at M (counters per work-group and row, kept equal across lists of different M). The head: TP'd like a
   layer (2 crossings a draft step) or replicated (0.85 GB a card, no crossing), to be measured. The draft vocabulary
   (`--draft-vocab`) splits at the vocabulary boundary.
3. **DFlash.** Replicated on both cards (int8, ~2.05 GB): both hold the identical residual, so the taps are local. Its
   verify lists are build_verify_taps at per-card shapes. The 8 GDN slots per card are half-size.
4. **AdaptiveK** (src/server/adaptive_k.*) picks K from an `MtpCost` table of verify / draft costs in plain steps. TP
   needs its own table (`tp`) from measured per-card verify(M) / draft costs, since the M-row crossing moves the
   ratios.
5. **Sampling.** The server reads full logits rows (`read_logits_into`). Under TP either gather both halves (2 x 0.5 MB
   a row over PCIe) or merge per-card top-k plus partial log-sum-exps (spec 17 §4.4). Proposed: the gather, which is
   exact and simple.
6. **The server.** `b70-serve --parallel tp` (no new flag) behind the one-card adapter path (`cli::EngineAdapter`-like,
   server::EngineIface), not `BatchEngineIface` (one sequence). replica and slots stay exclusive with tp.
7. **Snapshots / prefix cache.** The one-card host layout (17b's gather plus 17d's scatter). With 17b's slice form an
   entry is bitwise interchangeable between `--parallel 1`, pp, replica and tp.
8. **Bars.** D1-dense `--parallel tp --spec auto` vs `--parallel 1 --spec auto` >= 1.25x on the tool-call set; D4
   `--spec dflash`. `--parallel auto` choosing tp is decided here (D3).
9. **Box row 81**; the benchy stage `gpu-self`.

## The plan as written before the review (Bolt-first text kept for Bolt's later TP plan; Task 2 void)


**Conditions (operator, 2026-10-23; for Bolt - superseded for Qwen3.8 by the 2026-10-27 status below):** this plan starts only after **plans 17b and 17c are merged and their rows passed**
(which needed spec 17 §9's GO: P0 after `r22.devices` / `r22.p1`; Bolt's rows 59-62 passed; the crossing cost <= ~1 ms
a token), and - for Task 2 - after **spec 26c (LiLiCorr in the engine) has passed its row** on `--pp 2`; and the
operator says go.

**Status (2026-10-27): PARKED -> after 17b and 17c (rewritten for Qwen3.8) have passed their rows.** Retargeted to
Qwen3.8 first (spec 17 §R2): snapshots in the one-card host layout, MTP and DFlash under `--tp 2` (the MTP head TP'd or
replicated by 17b's measurement; the DFlash drafter replicated - spec 17 §R2.8), sampling over the split vocabulary,
`b70-serve --tp 2`, T3, D1-dense / D4 against one card's `--mtp auto` and `--spec dflash`, the record. Box queue row
**74**. The 2026-10-23 Bolt-first text below is kept for Bolt's later TP plan; its LiLiCorr task (Task 2) is void -
spec 26 was dropped on 2026-10-27.

**Status (2026-10-23, superseded 2026-10-27): PARKED - begins on the conditions above.** Rewritten 2026-10-23 for Bolt first (spec 17 §R);
the 2026-10-05 version (Qwen3.8: spec 7 per card, MTP, `{card 0 part, card 1 part}` entries) is in git history.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** everything else on Bolt under `--tp 2`: snapshots in the **one-card host layout** (spec 17 §4.7), spec 26's
LiLiCorr drafter under TP (spec 17 §4.8), sampling over the split vocabulary (§4.4), `b70-serve <bolt> --tp 2`; T3;
D4; the record. Kolibri is a later plan (17e, not written) on these pieces, after spec 20b's checkpoint and its own P0
line (spec 17 §3).

**Architecture:** spec 17 §4.4, §4.7-§4.9. `BoltEngine::{save,load}_{state,kv}` gather each card's heads (SSM state,
conv-ring channels, its kv head) into `runtime::bolt::{state_runs, kv_runs}`'s one host layout and scatter on restore;
the drafter's placement by spec 17 §4.8's decision (replicated / TP'd / one card) from 17a's numbers and 26a's
`r63.cost`; the taps read locally (both cards hold the residual); verify at M = 2..8 with M-row crossings; the
per-card top-k + partial log-sum-exp merge on the host; `cli::bolt` with `--tp`.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest, uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§4.4, §4.7-§4.9, §5 T3, §6 D4, §7).
Precedents: spec 25e (snapshots, `BoltEngineAdapterT`, `b70-serve <bolt>`), spec 26c (the tap verify, the slots, the
commit), spec 16d (snapshots byte-identical across modes).

## Dependencies and branch points

- Main with 17b and 17c merged (rows passed); 26c merged and passed for Task 2. Branch **`spec17d-tp-integration`**.

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` never committed;
  `oracle-out*` symlinked. No `rm -rf`.
- `--tp 1` unchanged (G0). `--tp 2` exclusive with `--pp 2` and `--replica` (spec 13 §10), refused by name.
- **Every two-card stage is `gpu`**; `r71.benchy` is `gpu-self` like `r62.benchy`.
- A RECORD-grade row (idle box, both cards) before any README headline changes.

## Review Focus

1. **One host layout:** an entry saved under `--tp 2` is byte-identical to the same position's entry under `--pp 2`;
   restores in every direction (`--tp 2` <-> `--pp 2`, every split) give the cold run's tokens - nothing refused.
2. **Sampling is exact:** the merged per-card top-k with the two partial log-sum-exps equals filtering the full logits
   (host test on synthetic split logits, every filter combination); seeded runs reproducible; LiLiCorr's full-vocabulary
   top-8 and log-probs equal under the merge.
3. **LiLiCorr under TP:** verify rows bitwise M = 1 per card (spec 26 D3) at M = 2..8; the taps read locally equal
   `--pp 2`'s; greedy speculative output identical to `--tp 2` plain decode; acceptance within noise of `--pp 2`'s.
4. **The server:** `b70-serve <bolt> --tp 2` with the prefix cache and the drafter on; `golden_server_test --chat`
   against `b70-decode <bolt> --tp 2`.

---

### Task 1: snapshots and sampling (Mac)

- [ ] Failing tests (`bolt_tp_snapshot_test` host, Review Focus 1-2), implement, PASS. Commit
  `git commit -S -m "runtime: Bolt snapshots under --tp 2 in the one-card host layout, the split-vocabulary sampling merge (spec 17d)"`.

### Task 2: LiLiCorr under TP (Mac) - VOID: Bolt's drafter, out of scope for Qwen3.8 (review 2026-11-10)

- [ ] The drafter placement per spec 17 §4.8's recorded decision; the M-row verify across the crossings; Review
  Focus 3's tests. Commit `git commit -S -m "runtime: LiLiCorr under --tp 2 (spec 17d)"`.

### Task 3: the server (Mac)

- [ ] `b70-serve <bolt> --tp 2`, the adapters, `bolt_serve_args_test` with the TP refusals. Commit
  `git commit -S -m "server: b70-serve <bolt> --tp 2 (spec 17d)"`.

### Task 4: the box queue row (proposed 71, or the next free row) and the record

```text
row 71 "spec 17d - Bolt under tensor parallel: snapshots, LiLiCorr, the server, the record (plan 17d)"
stage r71.k0      71 default cpu      -                     g0.sha,g0.bitwise,g0.suite "--tp 1 unchanged"
stage r71.host    71 default cpu      -                     -          "bolt_tp_snapshot_test, the sampling merge, bolt_serve_args_test"
stage r71.snap    71 default gpu      bolt_synth            r71.k0     "bolt_snapshot_gpu_test under --tp 2: block ends bitwise cold; --tp 2 <-> --pp 2 restores"
stage r71.spec    71 default gpu      bolt_g128,lilicorr    r71.snap   "LiLiCorr under --tp 2: D3 per card at M 2..8, greedy identity, acceptance vs --pp 2"
stage r71.chat    71 default gpu      bolt_g128             r71.snap   "golden_server_test --chat on b70-serve <bolt> --tp 2 == b70-decode <bolt> --tp 2"
stage r71.speed   71 optin   gpu      bolt_g128,lilicorr    r71.spec   "D4: --tp 2 + LiLiCorr vs --pp 2 + LiLiCorr, A4 + golden, idle, rotated rounds"
stage r71.benchy  71 optin   gpu-self bolt_g128,uvx         r71.chat   "llama-benchy at 886 / 8192 / 32000 / 65536: --tp 2 vs --pp 2, SergiioB's published TP 2 beside"
```
- [ ] docs/BENCHMARKS.md "Tensor parallel (spec 17)" complete; spec 17's as-built section; README only with a
  RECORD-grade row; docs/10-the-box.md (TP's lock: the whole box). Commit
  `git commit -S -m "spec 17: tensor parallel on Bolt - the record"`.

**Gate for the plan:** T3 green, Review Focus 1-4 on the box, D4 and the benchy rows recorded, `--tp 1` unchanged.

# Spec 22m - the coding mask: `--expert-mask coding`, a lossy opt-in mode that drops cold experts

**Status (2026-10-10): planned; begins when decision 9 sets the coding mask's kept fraction** (from plan 22a's pruning
curve, P0.8, against P0.7's cache speed) - whatever the tier's go / no-go. It needs plan 22b's expert table (22b's Task
1, Task 3's table and Task 4's binding with O0; built first on 22b's branch when decision 9 comes before the go), not
the cache or the mirror. Box queue row 40 (renumbers at build time if taken).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** spec 22 §4b, the operator's ruling of 2026-10-09: **a coding-only mode, opt-in, never the default.** A
per-model mask of kept (layer, expert) pairs, calibrated by REAP on coding / agentic traces only (the A4 set and the
opencode recordings), shipped beside the checkpoint as a named coding mask with its provenance, selected explicitly by
`--expert-mask coding` on `b70-decode` / `b70-serve`. The route kernel sets a dropped expert's logit to -inf before the
fp32 softmax, takes the top 10 of the kept experts and renormalises over them (the router's own rule over the kept
set); the loader never loads a dropped expert. At decision 9's fraction (50 % = 32.1 GB at int4 g64) the whole model
fits two B70s with no host tier. Every gate records the KL against the full model (P0.8's curve is the bar); the
mode's own bar is A4 tool-call accuracy on code.

**Architecture:** spec 22 §4b, §7 decision 9. No new kernel family.
- **The mask file** is plan 22a's format (`spec22-mask-1`: `keep` u8 [48][512], `mtp_keep` u8 [512], metadata: model,
  checkpoint revision, criterion, kept fraction, the calibration traces with their sha256). The named coding mask is
  `<snapshot>-expert-mask-coding.safetensors` (or `$B70_Q4_MASK`), written by `offload_curve.py reap --kept <decision 9>
  --sources A4,opencode[,agentic] --ship <snapshot>` - expert ids are the original's, so one mask serves Intel's
  checkpoint and ours (the provenance names the checkpoint whose traces made it).
- **The route:** `q4_route` gains `-DROUTE_MASK=1` (`_MASK`, composing with 22c's `_CNT` as `_MASK_CNT` when that exists):
  one more argument, the layer's 16 u32 of keep bits; `lg = rne(logits)`, then -inf for a dropped expert, the softmax
  over 512 in fp32, the top 10 by rank with **dropped experts ranked below every kept one** (so a kept expert whose
  probability underflows to 0 still wins a slot over a dropped one), ties to the lower id, renormalised in the
  reference's order. The prefill's route is the same binary on grid (1, C); the head's router uses `mtp_keep` (all ones
  by default: the head stays whole - this plan's proposal, see Dependencies).
- **The loader:** a layer's `gate_up` / `down` allocations hold only its kept experts, compacted in id order; 22b's table
  rows map kept ids to compacted blocks and dropped ids to 22b's poison block; the dropped tensors are skipped (counted
  and printed as `skipped by the coding mask`, so the unconsumed-tensor rule still holds); the planner counts kept
  bytes, so at 50 % the whole model plans as fitting two cards without the tier.
- **The mode is visible:** the CLIs print one startup line `expert mask: coding (LOSSY), kept <f> - <provenance>`; the
  server's request log carries the mask tag; `--expert-mask` absent means the full model, always.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest; Python 3 / torch in `agnes-ref-img` with 21a's 5.19.0
site (the masked reference sets).

**Spec:** `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (§3 P0.8, §4b, §7 decision 9); plan
22a (the mask file, `qwen4exp_ref.py --expert-mask`, `maskeval`, `offload_curve.py reap / prune-report`, the record's
pruning curve); plan 22b (the table, `ExpertMap`'s `None`, the poison block). Spec 21 §14 (`q4_route`: two experts a
lane, the 16-slot route row), §15 (the prefill's route on grid (1, C)), §16 (A4's set and reference).

## Dependencies and branch points

- **Decision 9** (the kept fraction) ruled from 22a's record; the shipped mask is `reap`'s at that fraction, calibrated
  as decision 9 says (with or without the A4 scenarios: 22a records both arms - the held-out arm is the honest A4
  number, the in-sample arm the shipped one if the operator picks it).
- **Plan 22b's table pieces merged** (Task 1's `_TBL` binaries, the table upload with identity rows, the engine's
  binding with O0). If plan 22b's cache has merged too, the mask composes with it in plan 22d Task 6, not here.
- **Row 30 / 34's CPU data** for the masked reference sets: Intel's checkpoint, 21a's golden prompts, the A4 set.
- **Open point (proposed here):** the MTP head's experts stay unmasked (`mtp_keep` all ones; 1.34 GB resident) - the
  mask changes the main model's output, the head only drafts; acceptance with the mask is measured (Task 5). A masked
  head is one flag away (`mtp_keep` from `reap --mtp`), recorded if the operator asks for it.

## Global Constraints

- Branch `spec22m-coding-mask` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never
  printed; `oracle-out*` symlinked.
- **Commit after each step that changes a file** (signed `git commit -S`; never bypass). No `rm -rf`. No merge, no push.
- **Never the default:** no path selects a mask without `--expert-mask` (or `$B70_Q4_MASK` for tests only, refused by
  the server unless `--expert-mask` is given too); `--expert-mask none` is the full model. A test pins it.
- **F0:** existing binaries unchanged (`q4_route`'s define off preprocesses as main's); new binaries `_MASK` only.
- Every number measured, or marked derived / estimated / proposed. Box: `flock ~/b70-gpu.lock` (both cards), detached,
  polled; interleaved pairs, median of 3, `uptime`, idle grade; CPU oracle runs one at a time, `free -g` >= 70 GB.
- Mac checks `tools/mac_check.sh --base main --kernels`; Python tests in `agnes-ref-img` with the 5.19.0 site.

## Review Focus

1. **The engine's masked route is the reference's masked route.** `q4ref::route_masked` (the host twin) equals
   `qwen4exp_ref.py`'s masked hook on the fixture's rows bitwise (ids and weights), including a planted tie at the 10th
   kept expert and a row where a kept expert's probability underflows to 0 beside dropped experts with higher logits;
   the kernel equals the twin bitwise. 22a's hook gains the same "dropped below every kept" rank rule if it lacks it
   (one line, its identity tests rerun).
2. **An all-kept mask is the full model, bitwise** (route, ids, logits, decode and prefill); `--expert-mask none` and no
   flag are byte-identical runs.
3. **A dropped expert is never read.** Its table entries are the poison block (a read would NaN the logits); every test
   run also checks every route row it reads back for a dropped id.
4. **Loss is recorded beside every gate.** The masked engine's KL to the full model's golden logits and its argmax
   agreement print next to the reference's own masked numbers from 22a (`maskeval`): the engine must reproduce the
   masked reference (F3), and the loss is the mask's, not the engine's.
5. **Provenance travels with the mask.** The loader refuses a mask whose `model` differs, whose shape is wrong, which
   keeps fewer than 10 experts in a layer or whose format is not `spec22-mask-1`, each by name; the startup line prints
   criterion, fraction and calibration sources.

---

### Task 1: the mask file and its loader (host)

**Files:**
- Create: `src/loader/qwen4exp_mask.{h,cc}`, `tests/loader/qwen4exp_mask_test.cc`, `tools/oracle/qwen4exp_mask_fixture.py`
  -> `tests/loader/testdata/q4_mask_{tiny,48}.safetensors` (seeded, committed)
- Modify: `src/loader/CMakeLists.txt`, `tests/CMakeLists.txt`, `tools/oracle/offload_curve.py` (`reap --ship <snapshot>`:
  writes `<snapshot>-expert-mask-coding.safetensors`; `--mtp` scores the head from `mtp_p` / `mtp_onorm`)

**Interfaces:**

```cpp
namespace loader {
struct Q4ExpertMask {
  uint32_t layers = 0, experts = 0;
  std::vector<uint32_t> keep_bits;     // [layers][16] u32: bit e of word e / 32 - what the route kernel binds
  std::vector<uint32_t> mtp_bits;      // [16]
  std::vector<uint32_t> kept;          // per layer
  double fraction = 0; std::string criterion, provenance;
  bool kept_at(uint32_t layer, uint32_t e) const;
};
// "coding" -> <snapshot>-expert-mask-coding.safetensors (or $B70_Q4_MASK); "none" -> empty; else a file path.
std::string q4_mask_path(const std::string& snapshot, const std::string& flag);
Q4ExpertMask load_q4_mask(const std::string& file, const model::Qwen4ExpDesc& d);   // Review Focus 5's refusals
}
```

- [ ] **Step 1: failing test** `qwen4exp_mask_test`: the fixtures load (bits = the u8 `keep`), each refusal by name, `none`
  -> no mask, `coding` resolves beside the snapshot. FAIL.
- [ ] **Step 2: implement**; PASS; `test_offload_curve.py` gains `--ship` and `--mtp`. Commit `git commit -S -m "loader:
  the coding mask file - keep bits per layer, provenance, refusals by name (spec 22m)"`.

### Task 2: the masked route kernel

**Files:**
- Modify: `src/kernels/qwen4exp/q4_moe.cl` (`ROUTE_MASK`), `src/kernels/qwen4exp_kernels.h` (`route_variant(M, bool count
  = false, bool mask = false)`), `src/kernels/CMakeLists.txt` (a block `# ==== Spec 22m: the coding mask ==== (begin) /
  (end)`: `q4_route_M<1..4>_E512_T10_N528_L256_MASK` and, when 22c's block exists, `_MASK_CNT`), `tests/kernels/qwen4exp_ref.h`
  (`q4ref::route_masked`), `tools/oracle/qwen4exp_fixture.py` + `tests/kernels/qwen4exp_fixture.h` (the masked route
  cases from 22a's hook), `tests/kernels/qwen4exp_ref_test.cc`, `tests/kernels/qwen4exp_kernels_test.cc`,
  `tests/kernels/qwen4exp_variant_names_test.cc`, `tools/mac/clrun/qwen4exp_run.cc`
- Test: `qwen4exp_ref_test`, the names test (host); `qwen4exp_kernels_test` (card); `qwen4exp_run` (Mac GPU)

- [ ] **Step 1: failing tests** - Review Focus 1 on the host (fixture from 22a's hook, regenerated with the new cases)
  and the names. FAIL.
- [ ] **Step 2: implement**; host PASS; the preprocessed-source identity of every existing `q4_route` variant recorded;
  the card case (kernel = twin bitwise at M = 1..4 and on grid (1, 2048); an all-ones mask = the plain binary bitwise);
  `qwen4exp_run` exact. Mac gate. Commit `git commit -S -m "kernels: q4_route _MASK - dropped experts at -inf before the
  softmax, ranked below every kept one, renormalised over the kept top-10 (spec 22m)"`.

### Task 3: the loader skips dropped experts; the engine and the CLIs

**Files:**
- Modify: `src/loader/qwen4exp_loader.{h,cc}` (`load_qwen4exp` takes the mask: compacted kept blocks, the table rows from
  the mask, the skipped count), `src/loader/qwen4exp_experts.{h,cc}` (22b's map from a mask: `None` homes),
  `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (`plan` over kept bytes), `src/runtime/qwen4exp/qwen4exp_capture.cc` and
  `qwen4exp_prefill.cc` (bind `_MASK` and the layer's keep bits when a mask is loaded), `src/runtime/qwen4exp/qwen4exp_engine.{h,cc}`
  (`mask()` accessor, the startup line), `src/cli/qwen4exp_decode.h`, `src/cli/qwen4exp_serve.h`, `src/cli/b70_serve.cc`
  (`--expert-mask coding|none|FILE`; the server refuses `$B70_Q4_MASK` without the flag), `tests/runtime/qwen4exp_plan_test.cc`,
  `tests/CMakeLists.txt`
- Create: `tests/runtime/qwen4exp_mask_test.cc` (card)
- Test: `qwen4exp_plan_test` (host: the whole model at 50 % kept fits two cards at 32768 without the tier; at 87.5 % it
  does not), `qwen4exp_mask_test` (card), `cli_reject_qwen4exp_mask_*` (binary)

- [ ] **Step 1: failing tests.** `qwen4exp_mask_test <synth ckpt>`: Review Focus 2 (all-kept bitwise, `none` = no flag);
  a seeded 50 % mask on the synthetics: kept blocks read back = the repack's, dropped entries = the poison block, route
  rows never hold a dropped id over 2200 prefill + 64 decode rows, the planned bytes = the allocated; `--pp 2` bitwise
  `--pp 1`. The refusals: `cli_reject_qwen4exp_mask_model` (another model's mask), `_shape`, `_env_only` (server). FAIL.
- [ ] **Step 2: implement**; PASS on the Mac's host half; Level Zero syntax. Commit `git commit -S -m "runtime: --expert-mask
  coding - the loader skips dropped experts, the table maps kept ids to compacted blocks, the route masks (spec 22m)"`.

### Task 4: the masked reference sets (CPU)

**Files:**
- Modify: `tools/box_validate/qwen4exp_oracle.sh` (mode `masked`: `qwen4exp_ref.py run --expert-mask <the coding mask>` on
  `q4exp_short`, `4k`, `agentic` (and `32k` opt-in) -> `<data>/oracle-out-q4exp-mask/`; and on the synthetics with their
  seeded masks -> `<data>/oracle-out-q4exp-synth/{ours,intel}/mask/`), `tools/box_validate/data.sh` (`have
  oracle_q4exp_mask`, `have q4exp_mask_coding`)
- [ ] **Step 1:** the mode (`DRY_RUN=1` prints it; times ESTIMATED from row 30's walls). Commit `git commit -S -m
  "box_validate: qwen4exp_oracle.sh masked - golden sets under the coding mask (spec 22m)"`.

### Task 5: the gates on the card - F3 masked, A4, speed (box)

**Files:**
- Modify: `tools/toolcall/engine_generate.sh` (`DECODE_ARGS`, if plan 22d has not added it), `tests/golden/qwen4exp_golden_test.cc` (a `mask FILE` argument: the engine masked against a masked golden set,
  plus the KL / argmax against the full set printed beside 22a's `maskeval` numbers), `tests/CMakeLists.txt`
  (`qwen4exp_golden_mask_synth_{ours,intel}_test`, `qwen4exp_golden_mask_intel_test` - the whole model at decision 9's
  fraction on two cards, no tier)

- [ ] **Step 1: the tests** (SKIP 77 without their data). Commit `git commit -S -m "tests: F3 under the coding mask
  against the masked reference, loss printed against the full model (spec 22m)"`.
- [ ] **Step 2 (box):** F3 masked on the synthetics and on the whole model (tokens tie-aware, routing incl. no dropped id,
  gate S); KL / argmax to the full model next to the reference's (Review Focus 4); A4 - `engine_generate.sh` with
  `DECODE_ARGS='--pp 2 --expert-mask coding'` (the extra b70-decode flags plan 22d Task 4 adds; added here if 22m lands
  first) on the set, `score.py` against the full model's
  reference **and** against the masked reference (`prune-a4`'s outputs): the coding mode's bar is decision 9's, PROPOSED as
  "at least the masked reference's tool-call accuracy minus one scenario"; MTP acceptance with the mask (head whole);
  speed - `b70-decode --bench --depth 4096 / 32768`, `--prefill-length 4096 / 32768`, `serve_benchy.sh` rows, the whole
  model at the kept fraction on two cards, against the all-resident truncated rows per layer and spec 21 §3's derived roofline (spec 22 §4b's "about the
  all-resident speed"). Commit
  `git commit -S -m "spec 22m: the coding mask on the whole model - F3 masked, A4 on code, the loss and the speed"`.

### Task 6: docs, the queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` ("22m as built"; decision 9 as ruled),
  `docs/superpowers/plans/box-validation-queue.md` (row 40), `tools/box_validate/stages.sh` (a `row 40` block),
  `docs/19-running-models.md` (the coding mode: lossy, opt-in, what it measured), `docs/BENCHMARKS.md` (its rows, marked
  LOSSY), this plan's status line

- [ ] **Step 1:** stages `r40.k0` (G0: every existing `q4_route` binary unchanged), `r40.host`, `r40.k1` (`kbins` of `_MASK`
  + the kernel case), `r40.mask` (`qwen4exp_mask_test`), opt-in `r40.oracle` (Task 4, CPU, hours), `r40.golden`
  (synthetics; the whole model when its masked set exists), opt-in `r40.a4`, `r40.speed`, `r40.benchy`;
  `test_box_validate.py` passes; `--dry-run --only r40`. Commit `git commit -S -m "box: queue row 40 - spec 22m, the
  coding mask"`; after the box run, "22m as built", commit `git commit -S -m "docs: spec 22 (22m as built)"`.

**Gate for the plan:** Mac - `qwen4exp_mask_test` (host half), `qwen4exp_ref_test`'s masked route cases, the names test,
`qwen4exp_plan_test` (50 % fits two cards, 87.5 % does not), `qwen4exp_run` exact, the existing `q4_route` variants'
preprocessed identity, cmdlines additions only, the never-default test. Box - G0; all-kept bitwise; F3 masked against the
masked reference; no dropped expert ever routed; the loss, A4 on code against decision 9's bar, and the speed recorded.

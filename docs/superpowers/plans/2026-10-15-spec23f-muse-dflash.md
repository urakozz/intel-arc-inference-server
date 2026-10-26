# Spec 23f - Muse-Glimmer's DFlash drafter: P0 (acceptance, verify cost, the stopping rule), then the engine

> DROPPED (operator, 2026-10-27): support removed from the engine in ef3f1d7; kept for the record.
>
> Reusable later (revive from git: `git show 4c491f8:<path>` - 4c491f8, this branch's parent on main, is the last
> commit that has all of it):
> - the DFlash v1 drafter in its Muse form: `src/kernels/muse/mg_dflash.cl`, `src/runtime/dflash/muse_drafter.h`, `src/runtime/dflash/muse_drafter.cc`, the Muse form of `src/model/dflash.h` / `src/model/dflash.cc`, `src/runtime/dflash/dflash_capture.cc` and `src/kernels/dflash_kernels.h`; int4 g64 drafter linears (`DraftFormat::Int4G64`, `src/loader/dflash_repack.cc`)
> - tests: `tests/runtime/muse_dflash_test.cc`, `tests/loader/muse_assistant_test.cc`, `tests/kernels/muse_dflash_names_test.cc`, the Muse form of `tests/kernels/dflash_ref.h` and `tools/oracle/dflash_ref.py`
> - dense attention: `src/kernels/muse/mg_attn.cl` (flash decode, sliding window over a ring or full NoPE, output gate), `src/kernels/muse/mg_attn_eager.cl`, prefill attention `src/kernels/muse/mg_pf_attn.cl` (DPAS flash + eager twin) - all three still live, byte-identical, in `src/kernels/bolt/` (Bolt binds them); their twins `tests/kernels/muse_ref.h`, `tests/kernels/muse_pf_ref.h` (the full-layer half lives on as `tests/kernels/mg_attn_ref.h`)

**Status (2026-10-16): built on the Mac, both parts, on ONE branch `spec23f-muse-dflash` (the operator's 2026-10-15
ruling: Part 2 built regardless of P0; P0 picks the defaults) - nothing of it has run on a card. Box queue row 47
(`tools/box_validate.sh --only r47`). As built: spec 23 §17 "23f as built"; the rulings: the plan's SDD ledger.**
Part 1's box halves (Task 2 Step 2, Task 3 Step 2, Task 4 Step 2, Task 5 on the box's numbers) are r47.taps / r47.accept
/ r47.cost / r47.verdict; Part 2's D1-D5 on the card are r47.d1 / r47.d3 / r47.d2 / r47.serve / r47.benchy.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent per part; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit after each step** (signed, `git commit -S`; a step that does not build or pass yet goes in as `WIP: ...`) - never leave a step's files uncommitted. When you stop for any reason, leave `HANDOFF-spec23f.md` (untracked, worktree root): the last step done, what is half-done, the next command.

**Goal:** decide with measurements whether `meta-models/Muse-Glimmer-30B-assistant` pays on our card (teacher-forced acceptance on the CPU, the verify cost at M = 1..16 on the box, the projection against plain decode AND prompt lookup), and only on a "go" build it into the engine: the drafter on the card, the draft / verify / commit lists, `b70-serve --spec dflash`, greedy losslessly and sampled exactly (spec 19's D0-D5).

**Architecture:** spec 23 §8; spec 19 §2 / §5 (the DFlash design) at Muse's shapes. P0 reuses spec 19a's tooling (`dflash_ref.py`, `dflash_accept.py`, `probe_draft_cost`, `lookup_accept.py`) with Muse's form added; the engine reuses 23c's decode list at M = K + 1 rows as the verify, spec 8's acceptance (`spec_accept`), spec 19's commit-side context projection and ring.

**Tech Stack:** Python 3 + torch (CPU, the oracle image with spec 21a's 5.19.0 site); C++17, OpenCL C, Level Zero (the probe and the engine).

**Spec:** `docs/superpowers/specs/2026-10-15-spec23-muse-glimmer-design.md` (§8, §4.2 MG6, §11 decisions 9-13). Spec 19 (`2026-10-05-spec19-draft-model-speculative-decoding-design.md` §2, §5, §6 D0-D5), its probe record `docs/probe-dflash-2026-10-08.md`, plans 19a (P0 tooling and its status) and 19b / 19c (the drafter and the loop), plan 19e and `docs/probe-prompt-lookup-2026-10-05.md` (lookup's offline replay).

## Dependencies and branch points

- **Part 1:** 23a merged (`muse_ref.py` with `--taps`, the A4 set, the golden sets on the box). The GPU probe needs only `gemv.cl` / `gemv_i8w.cl` at Muse's shapes (no engine).
- **Part 2:** P0's "go" recorded in spec 23 §8; 23c, 23d, 23e merged.
- **What 19a found** (it is this plan's prior): on Qwen3.8's DFlash 2 drafter, int8 and 4-bit drafters equal bf16's acceptance to 0.01 (E_7 7.09-7.10); agentic tool-call output accepts almost whole blocks (E_7 7.21 on A4), code 6.15, prose saturates ~2.2; the verify cost beyond M = 4 was extrapolated, not measured (row 29 measures it). Muse's block is 16: E_K for K up to 15 and verify up to M = 16 are new ground.

## Global Constraints

- Branches `spec23f-muse-dflash-p0` (Part 1) and `spec23f-muse-dflash-engine` (Part 2) from main; signed commits; no merge, no push. Box rules: lock, detached, polled, **interleaved pairs after a warm-up, median of 3** for every timing (sequential arms mis-state ratios under clock drift).
- The drafter reference is **transformers 5.19.0** (`MuseGlimmerAssistantModel` + `DFlashTokenCandidateGenerator`): window `|q - k| <= 2048` (MU:142-152; decision 13), 15 drafts per block (G:1658-1660; decision 10), raw embedding without the target's weight-less norm (G:1563-1579), raw head rows (G:1658).
- **Our formats only** for the drafter: bf16 as shipped, int8 per-channel RTN or int4 g64 RTN quantised by us (spec 19 decision 2's form; loader/rtn.h's `rtn_int4_g64`); AutoRound g64 only if RTN loses (decision 9). No third-party drafter format is loaded by the engine; `incoai/Muse-Glimmer-30B-DFlash2` is a P0 measurement arm only (decision 12).
- D0: without `--spec dflash` every kernel binary sha256-identical to main's and the suites green.
- No `rm -rf`. No third-party engine named beyond transformers and vLLM.

## Review Focus

1. **The tap offset:** taps are the OUTPUT of target layers 1, 13, 25, 37, 49 (OC:112-131 with a list), concatenated in that order before `fc` (33280 = 5 x 6656, A:298-299). A test feeds transformers' own `output_hidden_states=[...]` into both the port and transformers' candidate generator on the tiny pair and compares drafts bitwise.
2. **The block:** row 0 = the anchor at position p (the target's newest sample, not yet through the target), rows 1-15 = mask 201818 at p+1..p+15; drafts = argmax of the target head on rows 1-15; row r sees context `[p + r - 2048, p - 1]` and all 16 rows (bidirectional). A test at p = 2047, 2048, 2049, 3000 crosses the window's edge for row 0 and row 15.
3. **The context is only committed positions** (after a verify: the anchor + the accepted drafts, G:1606-1618); a rejected draft's taps never reach the ring; the block's own K / V are never kept (G:1620-1621).
4. **The drafter's norm rounds twice** (`w x bf16(x̂)`, A:50-55), unlike the target's centered norm; its attention has no gate and no 3.87; its RoPE uses its own 5e5 table; its embedding rows skip the target's weight-less norm.
5. **Verify rows at M = 1..16 are bitwise M = 1** (D3): logits, KV rows, ring rows, taps - otherwise greedy is not lossless (D2).

---

## Part 1 - P0

### Task 1: the drafter reference in Muse's form (Mac)

**Files:**
- Modify: `tools/oracle/dflash_ref.py` (a `muse` form selected by `model_type: muse_glimmer_assistant`), `tools/oracle/test_dflash_ref.py`, `tools/oracle/README.md` ("DFlash reference": the Muse form)

**Interfaces:**
- Produces: `config_from_dict` reads `MuseGlimmerAssistantConfig` (`block_size`, `mask_token_id`, `target_layer_ids` at the top level, AC:81-89) into `DFlashConfig(version=1, form="muse", window=2048, window_inclusive=True, embed_norm=False, head_cap=False, norm_two_roundings=True)`; `read_drafter_tensors` maps the assistant's names (`encoder.fc`, `encoder.output_norm_enc`, `layers.N.*`, `norm`); `draft_block(anchor_id, p, K, ctx)` returns K <= 15 drafts (rows 1..K of the 16-row block - the block keeps 16 rows whatever K; rows past K are computed and discarded, as transformers does).
- [x] **Step 1: failing test** `test_dflash_ref.py::test_muse_equals_transformers` on 23a's tiny pair (`muse_make_tiny.py --variant l8` target + `--variant assist`): a 60-id prompt through the target with `output_hidden_states=[1, 5]`, then transformers' `DFlashTokenCandidateGenerator.get_candidates` and the port's `draft_block` from the same taps: candidate logits (fp32 mode) and ids bitwise at anchors 20..59; at a window of 16 (the `win16` assistant variant) across its edge (Review Focus 2); a run where 3 drafts are accepted then the next block - the port's context equals transformers' cache after its `crop` (Review Focus 3); Review Focus 4's three differences each flip a draft when removed.
- [x] **Step 2: implement, PASS** (`agnes-ref-img` + the 5.19.0 site). Commit `oracle: dflash_ref.py - Muse-Glimmer's assistant form, bitwise against transformers 5.19.0 (spec 23f)`.

### Task 2: tap dumps from the engine-format reference (box CPU)

**Files:**
- Create: `tools/oracle/muse_taps.py`, `tools/oracle/test_muse_taps.py`
- Modify: `tools/box_validate/muse_oracle.sh` (mode `taps`), `tools/box_validate/data.sh` (`have oracle_muse_taps`)

**Interfaces:**
- Produces: `muse_taps.py <gptq-snapshot> --source NAME:IDS:N_PROMPT [...] --out <dir>` - one teacher-forced forward per source over [prompt + the reference's own greedy continuation] (23a's golden / A4 outputs), writing `dump_taps.py`'s format (`save_dump`: `ids`, `greedy`, `top_ids`, `top_logits` (the CAPPED row, top 64), `lse`, `tap.L{1,13,25,37,49}`, metadata `layers`, `tap_from`) so `dflash_accept.py` reads it unchanged; `tap_from` = max(0, n_prompt - 2048) (the drafter's window: older taps are never read).
- Sources: the 36 A4 scenarios with 23a's int4 reference continuations (512 ids), `short_code`, `short_prose` (32-id golden continuations; extended to 256 ids by `muse_ref.py run --gen 256` when time allows), and the agentic session with a 512-id continuation.
- [x] **Step 1: failing test** on the tiny pair: the dump's taps equal `muse_ref.Recorder`'s and transformers' `hidden_states[i]` bitwise; `load_dump` round-trips. Implement, PASS. Commit `oracle: muse_taps.py - Muse-Glimmer tap dumps in dump_taps' format (spec 23f)`.
- [ ] **Step 2 (box CPU):** `muse_oracle.sh <data> taps` (resumable, `DRY_RUN=1` first for the ESTIMATED hours and peak RSS - 19a's dumps peaked ~4x their estimate: start with `MUSE_REF_MIN_GB` 64). **(Box: r47.taps.)**

### Task 3: teacher-forced acceptance and the lookup baseline (box CPU; the tiny on the Mac)

**Files:**
- Modify: `tools/oracle/dflash_accept.py` (`--kmax 15`; arms `bf16`, `int8` (per-channel RTN, simulated), `int4g64` (RTN with `loader/rtn.h`'s rule restated, simulated), `dflash2` (incoai's checkpoint through `dflash_ref`'s v2 path, decision 12); `--q raw|capped` for the sampled acceptance estimate), `tools/oracle/test_dflash_accept.py`, `tools/spec/lookup_accept.py` (a `--dumps` source: the same continuations), `tools/oracle/dflash_p0.sh` (a `muse` mode)

- [x] **Step 1: failing tests** on the tiny pair: the batched acceptance equals per-anchor `draft_block` (as 19a's); `--kmax 15` with a 16-row block; the int4 g64 simulation equals `rtn_int4_g64`'s dequant on a fixture; `--q capped` applies the target's multiplier and softcap to the drafter's rows before the softmax. Implement, PASS. Commit `oracle: dflash_accept for Muse-Glimmer - K to 15, the int4 g64 arm, raw / capped q (spec 23f)`.
- [ ] **Step 2 (box CPU):** per corpus (A4, code, prose, the agentic session) and arm: E_K for K = 1..15 (greedy), the per-depth acceptance, and the sampled estimate at T 1.0 / top-k 64 / top-p 0.95 for both q forms (spec 8 §11's expected-acceptance formula); lookup's E_K on the same continuations (`lookup_accept.py --dumps`, n = 3, K <= 15). Record in `docs/probe-muse-dflash-<date>.md`. Commit `probe: Muse-Glimmer DFlash acceptance on our prompts (spec 23f)`. **(Box: r47.accept (the Mac's half: 18d615f, docs/probe-muse-dflash-2026-10-16.md).)**

### Task 4: the verify cost at M = 1..16 and one block's draft cost (box GPU)

**Files:**
- Modify: `tools/probe/probe_draft_cost.cc` (`--shapes muse`: the target's four int4 GEMVs per layer x 52 layers on random weights of the real shapes (13.4 GB allocated) + the int8 head at M rows + argmax, at M = 1..16; and one drafter block at M = 16 - fc 33280 -> 6656 over the new context rows, 5 layers' q / k / v / o / gate||up / down, the head over 15 rows - in bf16, int8 and int4 g64), `tools/probe/draft_cost_shapes.h` (Muse's shapes, read from the two configs), `src/kernels/CMakeLists.txt` (`B70_MUSE_VERIFY_M16`, additive: `gemv.cl` / `gemv_i8w.cl` / `argmax.cl`-copy variants at M = 9..16 for Muse's shapes - an `argmax.cl` M > 8 needs `src/kernels/muse/mg_argmax.cl`, a copy, since `argmax.cl:43` caps M at 8), `tools/box_validate/muse_oracle.sh` or the row's stage (the probe command)
- Interfaces: `probe_draft_cost --shapes muse [--mmax 16] [--rounds 3] [--depth 4096]` prints per M: the GEMV sum (ms), the head (ms), the estimated attention term at the depth (KV bytes / measured bandwidth), the total as a ratio to M = 1 (interleaved pairs M = 1 / M), and the drafter block (ms) per format; for M = 9..16 also the two-pass form (two M <= 8 launches) so the better is known.
- [x] **Step 1:** implement; Mac gates (L0 / OpenCL syntax, `kernel_cmdlines` additions only, the new variants compiled on the Mac's OpenCL where they fit, `B70_JOBS=2 tools/mac_check.sh --base main --quick --kernels` exit 0). Commit `probe: Muse-Glimmer verify cost at M = 1..16 and one drafter block (spec 23f)`.
- [ ] **Step 2 (box GPU):** run it at depths 4096 and 32768 (both cards idle; the GPU lock). **(Box: r47.cost.)**

### Task 5: the projection and the verdict

**Files:**
- Create: `tools/spec/muse_p0_project.py` (+ `test_muse_p0_project.py`)
- Modify: `docs/probe-muse-dflash-<date>.md`, spec 23 (§8 "P0 verdict"), this plan's status

- [x] **Step 1:** per corpus and arm: DFlash tokens per plain step at each K = E_K / (verify(K + 1) + draft), the best K; lookup's at its best K (its draft is free; the same verify costs); plain = 1.0. **(Ruled 2026-10-15: the engine half, Tasks 6-9, is built regardless - this rule is recorded, not applied; the step still records the table and the defaults.) Stopping rule (spec 23 §8.3): "go" only if DFlash beats plain decode by >= 10 % AND prompt lookup by >= 10 % on the coding (golden code) and agentic (A4 + the agentic session) sets.** Record the table, the chosen arm (decision 9's rule: int4 g64 if within 0.05 of bf16's E_K at the chosen K), K_max and the q form (decision 11). Commit `spec 23: P0 verdict for the Muse-Glimmer drafter`.
- [ ] **Step 2 (stop):** on "no go", Part 2 does not start; the record says which term lost (acceptance, verify cost beyond M = 8, or the draft cost) and what would change it (the M-row GEMV lever, spec 8 A7). **(Box: not applied: the operator's ruling built Part 2; r47.verdict records the box's verdict.)**

**Gate for Part 1:** the verdict recorded with its numbers; Part 2 starts only on "go".

---

## Part 2 - the engine (after "go")

### Task 6: the drafter on the card

**Files:**
- Create: `src/model/muse_assistant.{h,cc}` (`MuseAssistantDesc`, held to its config), `src/loader/muse_assistant_loader.{h,cc}`, `tests/loader/muse_assistant_test.cc`
- Modify: `src/loader/CMakeLists.txt`, `src/model/CMakeLists.txt`, `src/runtime/muse/muse_sizes.{h,cc}` (the drafter's bytes, ring 2048 x 20,480 B, taps 16 x 66,560 B; the plan's `MuseReserve::drafter`)

**Interfaces:**

```cpp
namespace model { struct MuseAssistantDesc { uint32_t layers = 5, hidden = 6656, q_heads = 32, kv_heads = 8,
  head_dim = 128, inter = 19968, block = 16, window = 2048, mask_id = 201818; float eps = 1e-5f, theta = 5e5f;
  std::vector<uint32_t> taps = {1, 13, 25, 37, 49}; };
  MuseAssistantDesc check_muse_assistant_config(const common::json::Value&, const MuseGlimmerDesc& target); }
namespace loader { enum class DraftFormat { Bf16, Int8, Int4G64 };
  struct MuseDrafterWeights;   // fc, norm_enc, 5 layers (q||k||v, o, gate||up, down, norms), final norm
  MuseDrafterWeights load_muse_assistant(l0::Context& ctx, const std::string& path, const model::MuseAssistantDesc& d,
                                         DraftFormat f); }
```
Refused by name: a vocab or hidden unlike the target's, a tap layer >= the target's layers, a block other than 16, a window other than 2048, a `mask_embedding` file (none ships; refuse rather than guess its meaning).
- [x] **Step 1: failing test** (host: names, shapes, the RTN at load against `rtn_int4_g64` / spec 9's int8 rule; refusals; the drafter's bytes in the plan; `--max-len auto` still 131072 on one card). Implement, PASS. Commit `loader: the Muse-Glimmer drafter on the card in bf16 / int8 / int4 g64 (spec 23f)`.

### Task 7: the drafter's kernels and the draft list (D1)

**Files:**
- Create: `src/kernels/muse/mg_dflash.cl` (`mg_df_norm` - the drafter's two-rounding norm; `mg_df_embed` - raw rows of [anchor, mask x 15]; `mg_df_ctx_prep` - the context projection's per-layer k_norm + RoPE into ring slot `p % 2048`; `mg_df_q_prep`), `tests/kernels/muse_df_ref.h`, `tests/runtime/muse_dflash_test.cc`
- Modify: `src/kernels/muse/mg_attn.cl` (a `-DDF_BLOCK=16` mode: GQA 4 (32 q / 8 kv), bidirectional, the ring window `|q - k| <= 2048` plus the 16 block rows), `src/kernels/muse_kernels.h`, `src/kernels/CMakeLists.txt`, `src/runtime/muse/muse_capture.{h,cc}` (`build_draft`), `src/runtime/muse/muse_engine.{h,cc}`

- [x] **Step 1: failing tests:** the twin `muse_df_ref.h` bitwise against a fixture from Task 1's reference (norm, embed, context prep, eager attention over the ring + block at the window's edge); `muse_dflash_test` (label `gpu`): **D1** - the engine's drafts from its own dumped taps against `dflash_ref.py`'s Muse form fed the same taps: ids equal at T 0 (tie-aware at the top-2), logits cosine >= 0.9999 (bf16) / 0.999 (int8, int4).
- [x] **Step 2: implement** the draft list (captured once; the anchor and K from Control; the head on rows 1..15 through `gemv_i8w` at M 15 and `mg_argmax` M 15). Commit `runtime: the Muse-Glimmer draft list (spec 23f, D1)`.

### Task 8: verify at M <= 16, the commit-side projection, the loop (D2, D3, D4)

**Files:**
- Modify: `src/runtime/muse/muse_capture.cc` (`build_verify(M)`: 23c's decode list at M = 2..16 with the taps of every row written to `tap_rows[M][5][6656]`; `build_commit`: the projection of rows 0..j - count read from Control - fc -> norm_enc -> per layer k / v -> k_norm -> RoPE -> ring), `src/kernels/CMakeLists.txt` (the decode list's variants at M = 2..16: `mg_prep`, `mg_attn` (M <= 16), the GEMVs, the head, argmax), `src/runtime/muse/muse_prefill_engine.cc` (the prefill writes the taps of the last 2048 prompt positions and fills the drafter's ring), `src/cli/muse_serve.h` (`--spec dflash --draft-model PATH --draft-format bf16|int8|int4`, `--spec lookup`, `--spec-max K` 1..15, `--spec-cost`), `src/server/spec_accept.{h,cc}` (q from the drafter's rows: raw or capped per decision 11), `tests/runtime/muse_verify_test.cc`, `tests/server/muse_spec_test.cc`

- [x] **Step 1: failing tests:** **D3** (`muse_verify_test`, label `gpu`) - verify rows at M = 2..16 bitwise the M = 1 steps (logits, full KV, ring rows, taps) on the synthetic and on 2100 ids (crossing the window); the commit's projected ring rows bitwise Task 7's context prep of the same taps (Review Focus 3); **D2** - `--spec dflash` greedy equals `--spec off` token for token on the golden prompts and A4, whatever K the policy picks; **D4** (`muse_spec_test`, host) - 10^6 seeded samples of the sampled acceptance match the capped target distribution within spec 8 M4's bars for both q forms; a prefix-cache restore with the drafter's fc rows (spec 23 §8.4: 13,312 B a position, the last 2048) continues exactly as a cold run's drafts; a restore without them is correct (lower acceptance only).
- [x] **Step 2: implement, PASS.** Commit `runtime: Muse-Glimmer DFlash - verify at M <= 16, the commit-side projection, the loop (spec 23f, D2-D4)`.

### Task 9: speed (D5), docs, the queue row

**Files:**
- Modify: `docs/BENCHMARKS.md`, `docs/19-running-models.md` (the `--spec dflash` flags for Muse), `README.md` (CLI table), spec 23 ("23f as built"), this plan's status, `docs/superpowers/plans/box-validation-queue.md` (row 47), `tools/box_validate/stages.sh` (row 47 block)

- [ ] **Step 1: D5 (box):** `b70-serve <muse>` with llama-benchy at the operator's flags and on the agentic session: `--spec off`, `--spec lookup`, `--spec dflash` (auto K). **`dflash` becomes Muse's default only if it beats both by >= 10 % on the agentic set and loses nowhere by more than 3 %** (spec 19 D5's form); otherwise it ships as an option. **(Box: r47.benchy.)**
- [x] **Step 2: queue row 47. Stage list (sketch):**
  - `r47.host` default cpu: `test_dflash_ref.py` (Muse form), `test_muse_taps.py`, `test_dflash_accept.py`, `test_muse_p0_project.py`;
  - `r47.taps` optin cpu `muse_gptq,oracle_muse`: Task 2 Step 2 (hours);
  - `r47.accept` optin cpu `oracle_muse_taps`: Task 3 Step 2 (all arms; incoai's drafter `rownote`d as an optional download);
  - `r47.cost` default gpu, no data needed (random weights of the real shapes): Task 4 Step 2 at 4096 / 32768;
  - `r47.verdict` manual: Task 5;
  - after "go": `r47.k0` (D0), `r47.k1` (the new binaries, the twin), `r47.d1`, `r47.d3`, `r47.d2`, `r47.serve`, `r47.benchy` (D5).
  `rownote`s: Part 2's stages SKIP "no go" until spec 23 §8 records "go"; the drafter download (5.1 GB).
- [x] **Step 3: commit** `docs: spec 23 (23f as built), box queue row 47 - the Muse-Glimmer drafter`.

**Gate for the plan:** Part 1 - the drafter reference bitwise transformers on the tiny pair; the acceptance tables, the verify-cost table and the verdict recorded. Part 2 (after "go") - D0-D4 green on the box, D5 recorded and the default chosen by its rule.

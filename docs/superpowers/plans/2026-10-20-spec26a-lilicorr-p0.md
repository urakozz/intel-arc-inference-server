# Spec 26a - LiLiCorr P0: the reference, teacher-forced acceptance on our Bolt, the verify and block costs, the projection (defaults, not a gate)

> DROPPED (operator, 2026-10-27): support removed from the engine in 99adbb0; kept for the record.
>
> Reusable later (revive from git: `git show c57e08e:<path>` - c57e08e, this branch's parent on main, is the last
> commit that has all of it):
> - sink attention: `src/kernels/dflash/ll_attn.cl` (the learned sink logit added in the softmax reduce); the top-k with the logsumexp `src/kernels/dflash/ll_topk.cl`; the bf16 GEMV with an fp32 output scale `src/kernels/dflash/ll_gemv_bf16_g.cl`; host names in `src/kernels/dflash_kernels.h`
> - the NVFP4 decoder: `src/loader/nvfp4.h`, `src/loader/nvfp4.cc`, `tests/loader/nvfp4_test.cc`; NVFP4 into our formats `src/loader/lilicorr_repack.h`, `src/loader/lilicorr_repack.cc`, `tests/loader/lilicorr_repack_test.cc`
> - twins and tests: `tests/kernels/lilicorr_ref.h`, `tests/kernels/lilicorr_ref_test.cc`, `tests/kernels/lilicorr_kernels_test.cc`; the oracle's LiLiCorr form (sinks) in `tools/oracle/dflash_ref.py`

**Status (2026-10-23): built on the Mac, branch `spec26a-lilicorr-p0` - the reference, the tooling, the probe, the
projection and queue row 63 (spec 26 §12 "26a as built"); nothing on a card; the real drafter's weights read on the Mac
(decoded exactly, 1,112,803,882 parameters, the formats' bytes and weight error); the box still needs them (`hf download
nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr --revision 449d24140160b8e51871bfc0f0a7338abf640aa1`).
The box's halves are r63.vllmref (Task 1 Step 3's run), r63.accept + r63.union (Task 3), r63.cost (Task 4 Step 2), r63.verdict
(Task 5 on the box's numbers), r63.vllm_xpu (Task 6); spec 26 §11 holds the PROVISIONAL defaults until then.** Box queue
row **63**. P0 sets the defaults (K, the drafter's format, the q form) and the expectation; **it gates nothing** - the engine
half (26b, 26c) is built whatever it shows (the operator's 2026-10-15 ruling for DFlash).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** a CPU reference of LiLiCorr in `tools/oracle/dflash_ref.py` (a third form beside vLLM's DFlash / DFlash 2 and
Muse's), its NVFP4 weights decoded exactly; the drafter's teacher-forced acceptance E_K (K = 1..7) on **our** target's taps
(SergiioB's g128 dequantised - what we serve; LiLiCorr was trained against NVIDIA's 2-bit 24GB release, Decision 9) per
corpus and per drafter format; the verify cost at M = 1..8 at Bolt's shapes and the drafter block's cost; the expert union of
M consecutive rows (what a verify's MoE reads); the projection against plain decode and prompt lookup; the defaults recorded in
spec 26 ("P0 verdict").

**Architecture:** spec 26 §1-§4, §10 decisions 1, 2, 4, 5, 6, 9, 10, 12. Spec 19a's / 23f's P0 tooling extended: `dflash_ref.py`
(the drafter), 25a's tap dumps (`oracle-out-bolt-taps`, dump_taps' format: `tap.L{1,18,35,51,68,85}`, `greedy`, the capped
top-64 logits, `lse`), `dflash_accept.py` (teacher-forced E_K from one block per anchor), `tools/spec/lookup_accept.py --dumps`,
`probe_draft_cost` (the GPU half), `tools/spec/muse_p0_project.py`'s structure for the projection.

**Tech Stack:** Python 3 + torch (CPU, the oracle image with spec 21a's 5.19.0 site); C++17 / Level Zero for the probe.

**Spec:** `docs/superpowers/specs/2026-10-20-spec26-lilicorr-design.md` (§1-§4, §7 D1's bars, §8, §10). Precedents: plans 19a
and 23f Part 1 task for task; `docs/probe-dflash-2026-10-08.md`, `docs/probe-muse-dflash-2026-10-16.md` (what the tables look
like). Spec 25a provides the target side.

## Dependencies and branch points

- **Plan 25a merged** for the tiny Bolt (`bolt_make_tiny.py`) and `bolt_ref.py`'s `--taps`; **row 57's `r57.taps`** on the box
  for the real tap dumps (Tasks 3-5 wait on them) and its golden route records (Task 3's union counts).
- **The drafter** (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` @ `449d2414`, 962 MB): downloaded on the Mac and the
  box (`hf download`; small enough for the Mac). The Mac's HF cache holds its config / README / quant config already.
- **The GPU probe** needs only `gemv.cl` / `gemv_i8w.cl` / `gemv_bf16.cl` / Muse's `mg_attn` and 25c's `nb_*` binaries if they
  exist (else stand-ins as 25c's P0) - no engine.

## Global Constraints

- Branch `spec26a-lilicorr-p0` from main; signed commits; no merge, no push. Box rules: lock, detached, polled, **interleaved
  pairs after a warm-up, median of 3** for every timing.
- **The reference is vLLM's module, ported** (Decision 1): `V/models/lilicorr.py`, `V/models/qwen3_dflash{,2}.py`,
  `V/spec_decode/{lilicorr,dflash2,dflash}/speculator.py` at `3901d56800`, semantics only - nothing copied; the window
  `|q - k| <= 1023`; 7 drafts from an 8-row block; the lattice is the walk's only score; q = softmax(lattice / T).
- **Our formats only** for the drafter: exact (W' bf16 + g), int8 per row (19b's rule on W'), int4 g64 RTN (23f's, from g · W'
  in fp64). The NVFP4 file is read by our decoder, never served as NVFP4.
- No `rm -rf`. No third-party engine named beyond vLLM (and SGLang as the head's credited origin).

## Review Focus

1. **The anchor row is the newest committed position's normed context row** (`hidden_norm(fc(taps))` at p - 1 when the anchor
   token sits at p), not the anchor token's embedding and not the pre-norm fc output; `anchor_valid` true.
2. **The lattice and the walk:** `start` broadcast over the previous index for slot 0, `pair[s]` for slots 1..6; the walk starts
   at previous index 0; scores x 8.0 in fp32; the relative slot bias indexed `slot(i) - slot(j) + 7` (the trained offset, also
   for K < 7); the candidates' log-probs (a full-vocabulary logsumexp) enter only as the five features.
3. **NVFP4 decodes exactly:** every (e2m1 code, e4m3 scale) pair's product is exactly a bf16 (and an f16); the two NaN scale
   codes are refused; `g` is applied to the GEMV output, never folded into W' (spec 26 §3).
4. **The sinks:** a per-head logit joining the softmax denominator, not scaled by `head_dim^-0.5`, no value; dropping it changes
   the output.
5. **The taps are the residual after Bolt's blocks 1, 18, 35, 51, 68, 85** in that order, concatenated before `fc` (24576 = 6 x
   4096).

---

### Task 1: the LiLiCorr form of the reference (Mac)

**Files:**
- Modify: `tools/oracle/dflash_ref.py` (a `lilicorr` form selected by `architectures: [LiLiCorrDraftModel]`; the NVFP4 decode in
  `read_drafter_tensors`; the LiLiCorr head; the anchor in the session; the walk over the lattice), `tools/oracle/test_dflash_ref.py`,
  `tools/oracle/README.md`
- Create: `tools/oracle/lilicorr_make_tiny.py` (a random tiny LiLiCorr: 2 layers, hidden 256, 4 q / 1 kv x 64, FF 512, window 16,
  block 8, conv 2 / 16, sinks, the head at hidden 64 / 2 layers / 4 heads / candidate_topk 8, NVFP4-packed linears with a
  non-trivial global scale - for the tiny Bolt target of 25a), `tools/oracle/lilicorr_vllm_check.py` (Decision 1's cross-check)

**Interfaces:**
- Produces: `DFlashConfig(form="lilicorr", version=2, window 1024, window_inclusive False, sinks True, cand_topk 8, lil_* ...)`;
  `nvfp4_decode(weight_u8, scale_e4m3, g) -> (W' bf16-exact fp32 tensor, g)`; `Drafter.draft_block(anchor_id, p, K, ctx,
  temperature, seed)` returns K <= 7 drafts, the 7 x 8 candidates, their log-probs, the lattice [7][8][8] and the realized
  scores; `ContextKV` carries the anchor row (`fc` plane's last committed row, normed).
- [x] **Step 1: the failing tests** (`test_dflash_ref.py::test_lilicorr_*`, on the tiny pair): Review Focus 1-5, each as a
  mutation that must change the drafts or the lattice; the NVFP4 enumeration (Review Focus 3); window 16's edge (rows 0 and 7 at
  p = 15, 16, 17); K = 1..7 from one block equal to the first K of the K = 7 walk at T 0; the sampled walk reproducible from its
  seed (vLLM's murmur3 keying, already in the module).
- [x] **Step 2: implement, PASS.** Commit `git commit -S -m "oracle: dflash_ref.py - the LiLiCorr form (sinks, window 1024, the lattice head, the anchor, NVFP4 decoded exactly) (spec 26a)"`.
- [x] **Step 3: Decision 1's cross-check** (the script; the run is r63.vllmref's - vLLM does not import in the Mac's image) (`lilicorr_vllm_check.py`): vLLM's `LiLiCorrHead` instantiated on CPU in a one-rank
  process group with the tiny head's weights unquantised (fp32), fed the same inputs as the port's head: the lattice bitwise in
  fp32 (else the max |diff| and the reason recorded). Runs in a container with a vLLM CPU install on the Mac, or in the box's
  vLLM image (CPU tensors) - `DRY_RUN=1` prints which. Commit `git commit -S -m "oracle: LiLiCorr head cross-checked against vLLM's own module on CPU (spec 26a)"`.

### Task 2: the tiny pair end to end and the dump reader (Mac)

**Files:**
- Modify: `tools/oracle/dflash_accept.py` (`--model bolt-lilicorr`: K 1..7 from one 8-row block per anchor; arms `exact`,
  `int8`, `int4g64`; `--q lattice|lattice+lp` for the sampled estimate), `tools/oracle/test_dflash_accept.py`,
  `tools/oracle/dflash_p0.sh` (a `lilicorr` mode)

- [x] **Step 1: the failing tests** on 25a's tiny Bolt + Task 1's tiny LiLiCorr (taps from `bolt_ref.py run --taps`): the batched
  acceptance equals per-anchor `draft_block`; the int8 arm's row scales absorb g (spec 26 §3: q independent of g; the stored
  scale `fp32(g · s')`); the int4 arm equals `rtn_int4_g64` on `g · W'`; `--q lattice+lp` (the alternative, recorded only) differs.
- [x] **Step 2: implement, PASS. Commit** `git commit -S -m "oracle: dflash_accept for Bolt + LiLiCorr - K 1..7, exact / int8 / int4 g64 arms, the q forms (spec 26a)"`.

### Task 3: teacher-forced acceptance, lookup, the expert union (box CPU)

- [ ] **Step 1 (box CPU, `r63.accept` + `r63.union`; the tooling built - `expert_union.py`, `dflash_p0.sh MODEL=lilicorr`):** per corpus (the 36 A4 scenarios' 512-id continuations, `short_code`, `short_prose`,
  the agentic session) and arm: E_K for K = 1..7 (greedy), the per-depth acceptance, the sampled estimate at T 1.0 / top-p 0.95
  for both q forms (spec 8 §11's expected-acceptance formula); prompt lookup's E_K on the same continuations (`lookup_accept.py
  --dumps`, n = 3, K <= 7). **The expert union:** from the golden / A4 route records (25a: ids per row and MoE layer), for every
  run of M = 2..8 consecutive rows the number of distinct experts per layer - mean, p50, p90 per M and per E bucket (240 .. 496),
  and the routed bytes a verify would read (Σ_layer u x block(I_l)). Record in `docs/probe-lilicorr-<date>.md`. Commit
  `git commit -S -m "probe: LiLiCorr acceptance on our Bolt, lookup's, the expert union of M rows (spec 26a)"`.

### Task 4: the verify and block costs (box GPU)

**Files:**
- Modify: `tools/probe/probe_draft_cost.cc` (`--shapes bolt-lilicorr`), `tools/probe/draft_cost_shapes.h`, `src/kernels/CMakeLists.txt`
  (additive variants at M = 2..8 for Bolt's GEMV shapes if 25c has not built them)

**Interfaces:** `probe_draft_cost --shapes bolt-lilicorr [--mmax 8] [--rounds 3] [--depth 4096|65536]` prints per M: the verify by
layer type at real shapes on **one card**, scaled by the counts (40 Mamba with the state read once and M slot writes of 4 MiB -
and the replay alternative's 2 x 4 MiB, Decision 4; 8 attention at the depth; 40 MoE with u experts from Task 3's p50 and p90 -
the union bytes as a GEMV-only stand-in, and the plain M-row form's 9 M blocks, Decision 10; the head at M rows), the ratio to
M = 1 (interleaved pairs), and one drafter block at M = 8 in each format (fc over 8 new rows, the 5 layers, the head at 7 rows,
the top-8 + logsumexp stand-in, the LiLiCorr head's 56-row projections as small GEMMs).
- [x] **Step 1:** implement; Mac gates (`B70_JOBS=2 tools/mac_check.sh --base main --quick --kernels`). Commit
  `git commit -S -m "probe: Bolt's verify cost at M = 1..8 by layer type (slots vs replay, plain vs union MoE) and one LiLiCorr block (spec 26a)"`.
- [ ] **Step 2 (box, `r63.cost`):** run at 4096 and 65536.

### Task 5: the projection and the defaults

**Files:**
- Create: `tools/spec/lilicorr_p0_project.py` (+ `test_lilicorr_p0_project.py`)
- Modify: `docs/probe-lilicorr-<date>.md`, spec 26 (§"P0 verdict"), this plan's status

- [x] **Step 1 (the tool; the DERIVED defaults in spec 26 §11 - the box's numbers are r63.verdict's):** per corpus and arm: tokens per plain step = E_K / (verify(K + 1) + block) at each K, the best K; lookup's at
  its best K; plain = 1.0; the published SPEED-Bench figures beside ours (on the 24GB target - Decision 9's gap made visible).
  **Record, do not gate** (the operator's ruling): the drafter format by Decision 2's rule, K_max, the q form, whether the replay
  lever (Decision 4) and the union MoE (Decision 10) clear their thresholds, the cost row 26c's `AdaptiveK` starts from. Commit
  `git commit -S -m "spec 26: LiLiCorr P0 verdict - defaults and expectation"`.

### Task 6 (optional, Decision 12): vLLM XPU with LiLiCorr on the box

- [ ] **Step 1 (r63.vllm_xpu, by hand):** in the box's vLLM XPU image, SergiioB's checkpoint at TP 2 with `--speculative-config
  '{"method":"dflash","model":<LiLiCorr>,"num_speculative_tokens":7,"rejection_sample_method":"standard"}'` - if it starts: its
  mean acceptance length on the A4 prompts and the agentic transcript (greedy and T 1.0), recorded beside Task 3's; if it does not
  (the card says SM120 only), the error recorded and the step closed.

### Task 7: docs, the box queue row

- [x] **Step 1:** the queue row 63 (`python3 tools/box_validate/test_box_validate.py` passes):

```text
row 63 "spec 26a - LiLiCorr P0: the reference (dflash_ref.py's LiLiCorr form, NVFP4 decoded exactly, cross-checked against vLLM's head), acceptance on our Bolt, the expert union, the verify / block costs, the projection (plan 26a)"
stage r63.host     63 default cpu  oracle_image                 -          "test_dflash_ref.py (LiLiCorr), test_dflash_accept.py, test_lilicorr_p0_project.py"
stage r63.vllmref  63 optin   cpu  vllm_image                   -          "lilicorr_vllm_check.py in the vLLM image: the head bitwise in fp32"
stage r63.accept   63 optin   cpu  oracle_bolt_taps,lilicorr    -          "E_K per corpus and arm, lookup, the expert union (hours)"
stage r63.cost     63 default gpu  -                            -          "probe_draft_cost --shapes bolt-lilicorr at 4096 / 65536"
stage r63.verdict  63 manual   -   -                            r63.accept "lilicorr_p0_project.py: the defaults into spec 26 (not a gate)"
stage r63.vllm_xpu 63 manual   gpu bolt_g128,lilicorr           -          "optional: vLLM XPU serving Bolt with LiLiCorr - external acceptance figure"
rownote 63 "Download the drafter (962 MB): hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr --revision 449d24140160b8e51871bfc0f0a7338abf640aa1"
```
- [x] **Step 2: commit** `git commit -S -m "docs: box queue row 63 - the LiLiCorr P0"`.

**Gate for the plan:** the reference's tests green on the tiny pair (Review Focus 1-5 each able to fail), the vLLM cross-check
recorded; on the box - the acceptance tables, the union counts, the cost table and the projection recorded with the defaults
they imply. Nothing here blocks 26b / 26c.

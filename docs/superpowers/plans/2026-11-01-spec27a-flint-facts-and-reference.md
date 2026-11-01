# Spec 27a - Flint: the facts, the CPU reference, the tiny checkpoints, golden sets, the A4 set, the taps

**Status (2026-11-01): plan written, nothing built.** Box queue row: none yet (assigned when the operator asks).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the reference every later spec 27 stage is graded against: a facts sheet pinning every rounding point of spec 27 section 1 to `T:line` of the pinned `modeling_flint.py`; `tools/oracle/flint_ref.py` (restated ops, each bitwise the pinned module, the eager experts loop forced); tiny random Flint variants incl. an int4 g64 one in the checkpoint's key layout; golden sets (f32, bf16 on the dequantised quant); the A4 set rendered by Flint's template; the tap dumps; the header facts the layout needs (dtypes of `gate.weight`, `softmax_offset`, norms; qzeros).

**Architecture:** spec 27 sections 1, 5, 9 (F1). Precedents, followed task for task: plan 25a (`2026-10-20-spec25a-bolt-facts-and-reference.md`), `tools/oracle/kolibri_ref.py` / `kolibri_fixture.py`, `tools/oracle/stream.py`, `tools/oracle/dequant.py`, `tools/oracle/bolt_make_tiny.py`, `tools/box_validate/kolibri_oracle.sh`, `tools/toolcall/{make_set,score}.py`, `tools/toolcall/a4_ref.sh`.

**Tech Stack:** Python 3, torch CPU, **transformers pinned at `0ea108daf169dd66a1800e48a56d2251d4ff76bd`** (the merge of PR #49824; or the release containing `flint` when it ships - one pin, refused otherwise by name), safetensors, the `agnes-ref-img` container on the Mac (8 GB, 4 CPUs), the box CPU for real weights.

**Spec:** `docs/superpowers/specs/2026-11-01-spec27-flint-design.md` (sections 1, 5, 9 F1, 12 risks).

## Dependencies and branch points
- None on the engine. No C++, no kernel. 27b, 27c, 27q depend on it.
- **Weights.** Mac: only small files and range reads of shard headers / a few tensors (few MB). The HF cache does not hold Flint on the Mac: Task 1 fetches `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, the index of both repos, and `modeling_flint.py` / `configuration_flint.py` at the pinned commit into a cache dir outside the tree. Box: the operator's downloads (`RecursiveSI/flint`, `urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ`); the operator downloads, never this plan.

## Global Constraints
- Branch `spec27a-flint-reference` from main; signed commits; no merge, no push. `tools/box.env` copied from the main checkout if missing, never printed or committed.
- **Nothing of the model is vendored:** configs are built in code from the fetched `config.json`; tests needing the real files SKIP (exit 77) without them. The pinned `modeling_flint.py` is imported at run time, never copied into the tree (its sha256 is checked: record it in the facts sheet; the 2026-11-01 read was `869d318cec68654a...`).
- **The reference's attention is eager and its experts loop is eager** (`use_experts_implementation` forced to the eager loop; asserted in a test by comparing against the grouped path on one tiny input and recording the difference).
- One oracle container at a time on the Mac; the CPU model lock `/tmp/b70-cpu-model.lock` around every real-weight run; no `rm -rf`; every number in docs is measured or marked derived / estimated.

## Review Focus
1. **The norm.** `w * (x * rsqrt(mean + eps))` in fp32, ONE rounding to bf16; rounding `x_hat` first (Kolibri's order) must fail a test. Same for q / k head norms, which run **before** RoPE.
2. **RoPE only on sliding layers** (35); NoPE on the other 14 (a RoPE added to a full layer changes the output); cos / sin fp32 -> bf16; the rotation's three roundings.
3. **Sinks.** One fp32 scalar per head in the softmax denominator of **every** layer; the sink column dropped after the softmax; probabilities rounded to bf16 before PV. Tests: sink = -inf equals no sink; a dominant sink collapses the output to ~0.
4. **The router chain.** fp32 logits from the widened bf16 hidden and the widened bf16 matrix; fp32 sigmoid; top-6 on the **scores** (planted ties and a saturated-sigmoid tie: record which id `torch.topk` on the pinned CPU build takes - torch does not specify it - and pin that order; selecting on the logit instead must fail the saturated-tie test); `w = s / sum` fp32 without epsilon, the sum's order over `topk`'s descending output pinned (25a's precedent for Bolt); a bf16-rounded logit must flip a planted route.
5. **The combine.** Per expert `rne(rne(down) * w)` with w fp32 unrounded, sequential bf16 adds in ascending expert id (rank order must differ on some row), then `+ shared` (ungated, I 1024), then the residual.
6. **Layer table and untied head.** Layers 0-1 dense and full-NoPE; 48 full + MoE; layers 2-4 sliding; `lm_head` is not the embedding.

---

### Task 1: fetch, pin, and the remote code in the pinned release on a tiny config (Mac)
**Files:** Create `tools/oracle/flint_env.sh` (fetch + pin + site, the `qwen4exp_env.sh` pattern), `tools/oracle/flint_tiny_check.py`.
- [ ] **Step 1:** `flint_env.sh <dir>` downloads the files listed above at the pinned commit (GitHub raw for the code, HF raw for the small files), prints their sha256s, builds the pinned site. `flint_tiny_check.py`: a random tiny `FlintConfig` (8 layers keeping the real pattern's kinds: dense layers 0-1 are full-NoPE as in the real config, then `layer_types` [full, full, sliding, sliding, full, sliding, sliding, full], `mlp_only_layers [0,1]`, hidden 256, 8 q / 2 kv x 32, E 32, top-6, expert I 64, shared I 128, dense I 192, window 17), fp32 and bf16, 24-id prompt then `generate(max_new_tokens=8, do_sample=False)` against 32 ids as one forward: argmax of the last 8 rows equal; logits max |diff| printed.
- [ ] **Step 2: run on the Mac** (the 24a container command). Expected `tiny check: OK`; if the pinned code cannot build, stop and report.
- [ ] **Step 3: commit** `oracle: Flint's reference code pinned and run end to end on a tiny config (spec 27a)`.

### Task 2: the tiny variants and `flint_make_tiny.py` (Mac)
**Files:** Create `tools/oracle/flint_make_tiny.py`, `tools/oracle/test_flint_make_tiny.py`.
- Variants (`--variant NAME`, checkpoint tensor names as shipped: per-expert unfused `mlp.experts.E.{gate,up,down}_proj`, `mlp.gate.weight`, `shared_expert.*`, `self_attn.softmax_offset`, `q_norm` / `k_norm`): `bf16`, `fp32`, `ties` (two identical router rows; two experts whose logits saturate fp32 sigmoid at 1.0), `sinks-large`, `int4-g64` (every quantised linear RTN int4 g64 sym in the `auto_round:auto_gptq` bytes, `src/loader/rtn.h`'s rule, signed scales as AutoRound writes them, `extra_config` gate rule in the config).
- [ ] **Step 1: the failing tests:** each variant loads through the pinned class; int4 dequant equals `(q - 8) * scale` exactly; the quantised key set equals the real index's key *shape* (normalised by replacing integers); the layer table equals `layer_types`.
- [ ] **Step 2: implement; PASS. Commit** `oracle: Flint tiny variants incl. the AutoRound int4 g64 key layout (spec 27a)`.

### Task 3: `flint_ref.py` and F1 (Mac)
**Files:** Create `tools/oracle/flint_ref.py`, `tools/oracle/test_flint_ref.py`; modify `tools/oracle/README.md` (a section and the Files rows).
**Interfaces:**
```python
PINNED_TRANSFORMERS_COMMIT = "0ea108daf169dd66a1800e48a56d2251d4ff76bd"
def config(snapshot, layers=None) -> dict          # layers=N truncates; layer table from layer_types
def expected_names(cfg, form) -> list[str]         # "bf16" | "ar-g64" (the operator's quant)
def build_streamed(snapshot, cfg, weights="int4")  # int4 dequantised on demand (engine-format reference) | "bf16"
class Recorder:   # h after each block; route ids / w / scores of ranks 0..7 and the 6th-7th gap per MoE layer;
                  # attention outputs; taps; logits
def rms_norm(x, w, eps); def route(x, gate_w, k=6); def combine(d_rows, w, ids)
def attention(q, k, v, sink, window, scale)        # eager, fp32 scores, sink column, bf16 P
def rope(q, k, cos, sin)                           # the three-rounding chain
def main()                                         # run | ppl | facts | hfcheck | taps | gateq | a4
```
- [ ] **Step 1: the failing tests** (each against the unmodified pinned code, un-streamed; SKIP without the fetched files): one test per Review Focus point, each able to fail by the named mutation (round `x_hat` first; RoPE on a NoPE layer; sink dropped / scaled by head_dim^-0.5 / added to PV; bf16-rounded logit; rank-order combine; epsilon added to the renormalisation; shared gated; tied head); `test_chunked_equals_whole` (chunks of 1, 2, 3, 64, 127, 128, 129, 255 through the cache against one forward, bitwise where the op is row-wise, within a measured bound for attention across the window edge 513); `test_eager_experts_forced`; `test_int4_streamed_equals_dense` (the streamed int4 reference equals the model on the dequantised twin bitwise); `test_tap_convention`.
- [ ] **Step 2: run, FAIL; implement until PASS.** `hfcheck <snapshot> --layers N` runs the pinned code against the port on real weights (Task 6).
- [ ] **Step 3: README section; commit** `oracle: flint_ref.py - the pinned Flint code layer-streamed, F1 on the tiny variants (spec 27a)`.

### Task 4: the facts sheet (Mac: fetched files, range reads)
**Files:** Create `docs/probe-flint-2026-11-<dd>.md`, `tools/oracle/flint_facts.py` (prints the tables; `--headers` range-reads every shard header of both repos into a cache outside the tree; `--sample` reads named tensors by range); modify `docs/README.md` (one line).
- [ ] **Step 1: the rounding chain**, one row per point with `T:line` citations (norm, q / k norm, RoPE, scores, sink, P rounding, PV, o, router, combine, shared add, residuals, head).
- [ ] **Step 2: the checkpoint:** every shard header of the urakozz quant and of the original, asserted against what the spec 27 review read by range on 2026-11-01 (all of `gate.weight`, `softmax_offset`, norms, head, embedding bf16; scales f16; experts unfused in both; `softmax_offset` renamed to `sinks` by `conversion_mapping.py`, a pure rename); **every** `qzeros` word (the review sampled 20 tensors, all 0x77777777); scale range; the sink values per layer (sampled -9.0 .. 10.25); the tensor count formula (109,612 = 47 x 256 x 9 + 1,324) asserted. These fix 27b's layout.
- [ ] **Step 3: tokens and template:** render the template on a fixed set (system + tools + user, assistant with reasoning and two tool calls, tool result, dict vs string arguments) with the pinned transformers jinja and record the byte strings as fixtures `tests/tokenizer/flint_template_*.txt` (the K2 `dump_k2.py` pattern); the stop ids; BOS never.
- [ ] **Step 4: commit** `docs: Flint's rounding chain, checkpoint headers and template fixtures (spec 27a)`.

### Task 5: prompts, the A4 set, the golden prompts, `flint_oracle.sh` (Mac)
**Files:** Create `tools/oracle/flint_prompts.py`, `tools/box_validate/flint_oracle.sh`; modify `tools/toolcall/make_set.py` only by adding a `--model flint` template mapping if the existing `--from` cannot take Flint's template (a new branch, existing behaviour untouched).
- [ ] **Step 1:** the golden prompts (short, 2k, 16k, 64k agentic-transcript ids from `make_long_prompt.sh`), the A4 set rendered by Flint's template and tokenizer (`--from`), the tool-call scorer's Flint format (`<tool_call>` + `<arg_key>` / `<arg_value>`; a failing test first, `score.py` adds a format, existing formats unchanged).
- [ ] **Step 2:** `flint_oracle.sh` modes `facts`, `golden` (f32, bf16-on-dequant), `ppl`, `taps`, `a4`, `gateq`, each detached-capable (`tools/probe/detach.sh`), the lock, the RAM floor. **Step 3: commit** `oracle: Flint prompts, the A4 set and the box script (spec 27a)`.

### Task 6: the box stage (CPU only; the operator's box session)
- [ ] **Step 1:** `facts --headers` on the box copies; **Step 2:** `hfcheck` at `--layers 4`, `--layers 12`, then the whole model (layer-streamed, detached): the pinned code against the port on real weights, bitwise outside the documented bf16 differences; **Step 3:** the golden sets (f32 on the dequantised quant; bf16 on the dequantised quant) for the prompts, with the route-gap distribution recorded (the near-tie tolerance for 27c / 27d); **Step 4:** taps; **Step 5:** the A4 reference (hours); **Step 6:** record in the facts sheet; commit `docs: Flint reference results on the box CPU (spec 27a)`.

## Gates
- **F1** (spec 27 section 9) on the tiny variants, fp32 and bf16, prompt and cached decode. The real-weight `hfcheck` bitwise outside recorded differences. The facts sheet answers the section 1 open items. The f32 / bf16 golden distance (the MoE judge baseline) is recorded as found.

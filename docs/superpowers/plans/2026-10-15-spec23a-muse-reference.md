# Spec 23a - Muse-Glimmer-30B: the facts, the CPU reference, golden sets, the A4 set, gate Q

> DROPPED (operator, 2026-10-27): support removed from the engine in 98c13ef; kept for the record.
>
> Reusable later (revive from git: `git show c57e08e:<path>` - c57e08e, this branch's parent on main, is the last
> commit that has all of it):
> - dense attention: `src/kernels/muse/mg_attn.cl` (flash decode, sliding window over a ring or full NoPE, output gate), `src/kernels/muse/mg_attn_eager.cl`, prefill attention `src/kernels/muse/mg_pf_attn.cl` (DPAS flash + eager twin) - all three still live, byte-identical, in `src/kernels/bolt/` (Bolt binds them); their twins `tests/kernels/muse_ref.h`, `tests/kernels/muse_pf_ref.h` (the full-layer half lives on as `tests/kernels/mg_attn_ref.h`)
> - the between-GEMV kernels `src/kernels/muse/mg_prep.cl` (weight-less / centered norms, sandwich post-add, q / k prep, SiLU x up), `src/kernels/muse/mg_pf_prep.cl`, `src/kernels/muse/mg_argmax.cl`; host names `src/kernels/muse_kernels.h`; the GEMV cells at its dense shapes in `src/kernels/CMakeLists.txt`'s Spec 23c / 23d blocks
> - the dense descriptor / layout / loader: `src/model/muse_glimmer.h`, `src/model/muse_glimmer.cc`, `src/loader/muse_layout.h`, `src/loader/muse_loader.h`, `src/loader/muse_loader.cc`, `src/loader/muse_repack.h`, `src/loader/muse_repack.cc`; the engine `src/runtime/muse/` (buffers, capture, prefill, sizes, RoPE)
> - tests: `tests/kernels/muse_ref_test.cc`, `tests/kernels/muse_pf_ref_test.cc`, `tests/kernels/muse_kernels_test.cc`, `tests/kernels/muse_pf_kernels_test.cc`, `tests/loader/muse_layout_test.cc`, `tests/loader/muse_repack_test.cc`, `tests/loader/muse_load_checkpoint_test.cc`, `tests/loader/muse_synth_host_test.cc`, `tests/model/muse_glimmer_test.cc` (+ `tests/model/muse_glimmer/`), the oracle `tools/oracle/muse_ref.py`; `tests/CMakeLists.txt`'s Spec 23b-23e blocks

**Status (2026-10-15): built on the Mac (branch `spec23a-muse-reference`) - Tasks 1-5 and 7's code done and their gates green; MG1 bitwise on the tiny variants AND on real weights (`hfcheck --layers 4`, the int4 checkpoint); golden sets made on the Mac for the two short prompts (spec 23 §12); the long golden sets, both perplexities on the bf16 form, the -GPTQ repo's g_idx scan and gate Q are box queue row 42's (box CPU only). Spec 23 §12 "23a as built".**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit after each step** (signed, `git commit -S`; a step that does not build or pass yet goes in as `WIP: ...`) - never leave a step's files uncommitted. When you stop for any reason, leave `HANDOFF-spec23a.md` (untracked, worktree root): the last step done, what is half-done, the next command.

**Goal:** the reference every later 23 stage is graded against: a facts sheet with every rounding point of spec 23 §1.1 pinned from transformers 5.19.0 source (file:line); `tools/oracle/muse_ref.py`, transformers' own `MuseGlimmerForConditionalGeneration` text path layer-streamed, **bitwise** equal to the un-streamed transformers forward on the tiny variants (MG1); golden sets (short and long prompts, the GPTQ checkpoint dequantised) with raw and capped logits and the drafter taps; the A4 set rendered by Muse's template; gate Q (the GPTQ checkpoint against the bf16 original, spec 23 §3.3).

**Architecture:** spec 23 §1, §3.3, §4.1. Qwen4Exp's arrangement (`tools/oracle/qwen4exp_ref.py`, spec 21 §12): the transformers model class built on `meta`, `stream.py`'s `attach` materialising one decoder layer at a time, the int4 linears dequantised on demand (`(q - 8) x scale`), `attn_implementation="eager"`, a recorder by forward hooks, `layer_major` for batches. Muse needs no new mechanism: dense SwiGLU, no experts, no PLE; the vision tower is never materialised.

**Tech Stack:** Python 3, torch 2.14.1 CPU, **transformers 5.19.0** through spec 21a's site (`tools/oracle/qwen4exp_env.sh` - the same pins; `agnes-ref-img` keeps 5.15.0), safetensors; the Mac's `agnes-ref-img` container (8 GB / 4 CPUs: the operator's cap) for tests and the tiny model; the box CPU (`tools/oracle/run_in_container.sh`) for real weights.

**Spec:** `docs/superpowers/specs/2026-10-15-spec23-muse-glimmer-design.md` (§1, §1.1, §1.2, §3, §4, §9, §11 decisions 1, 5, 14). Precedents: plan 21a (`2026-10-09-spec21a-qwen4exp-facts-and-reference.md`), `tools/oracle/kolibri_ref.py` (window, NoPE, `run` / `ppl` / `facts`), `tools/box_validate/kolibri_oracle.sh` / `qwen4exp_oracle.sh` (resumable runs, `MemAvailable` floor, exit 77), `tools/toolcall/{make_set,score,oracle_generate}.py`, `a4_ref.sh`.

## Dependencies and branch points

- **None on the engine.** No C++, no kernel. 23b (`expected_names`, the synthetic checkpoints), 23c (the kernel fixture imports this reference's ops), 23e (the A4 set, the template renders) and 23f (the taps) depend on it.
- **Weights.** The tiny model `optimum-intel-internal-testing/tiny-random-muse-glimmer` at `cb70af19` (27.1 MB) is fetched into the HF cache by Task 1 (never vendored). The real checkpoints are box-side: `Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ` (22.8 GB; the engine-format reference) and `meta-models/Muse-Glimmer-30B` (59.6 GB; gate Q's bf16 reference, decision 14). On the Mac only their small files are read (both are in the HF cache: config, index, generation config, tokenizer files, template, README). Every real-weight step SKIPs (exit 77, "missing data") until the operator downloads them on the box (`df -h ~` first).
- **Licence:** Apache-2.0 (both cards): small files (configs, the template) may be vendored into `tests/`.

## Global Constraints

- Branch `spec23a-muse-reference` from main; signed commits; no merge, no push. Box tree automatic (`tools/box.sh dir`); `tools/box.env` copied from the main checkout if missing, never read into a log, never committed.
- **One reference version:** transformers 5.19.0. `muse_ref.py` refuses any other version, naming the installed one.
- **Never change `agnes-ref-img`**; 5.19.0 lives in the `--target` site (`oracle-out-q4exp/site` on the Mac, `$DATA/q4exp-site` on the box - reuse it, do not make a second).
- One oracle container at a time on the Mac; box CPU runs detached (`tools/probe/detach.sh`), polled, `MemAvailable` checked; no GPU, no GPU lock.
- Golden prompts must reach past 2048 positions (below it every sliding window covers the whole prompt: the window and the ring are untested).
- Every number in docs is measured, or marked derived / estimated / published. Facts cite transformers 5.19.0 as spec 23 abbreviates (`M:`, `C:`, `A:`, `G:`, `MU:`, `OC:`, `RP:`, `T:`).
- No `rm -rf`. No third-party engine is named beyond what spec 23 cites (transformers, vLLM).
- The rendering uses the ORIGINAL's `chat_template.jinja` (sha256 `cfc67e5f...`; spec 23 decision 1), never the GPTQ repo's older one.

## Review Focus

1. **Bitwise, not close.** The streamed port equals un-streamed transformers bit for bit in bf16 eager and fp32, prompt and cached decode, at lengths that cross the sliding window (the tiny variant with window 16 makes that cheap; the real-window case runs at 2100 positions once). A tolerance anywhere hides the rounding points 23c must copy.
2. **The two weight-less norms use `pow`, the centered ones `rsqrt`** (M:110-113 vs M:128-129); the final norm's weight is plain `w` with one cast (M:116-119); the drafter's norm (A:50-55) rounds twice. A test per form proves the port calls the module, not a restatement, or that a restatement is bitwise the module.
3. **The softcap's four bf16 roundings** (M:1178-1181): the recorder keeps the raw `lm_head` row and the capped row; a test plants two raw logits that round to the same capped value and checks the recorded `cap_tie` flag and torch's argmax (the lower id).
4. **Untied head.** The downloaded tiny is tied (no `lm_head` tensor); every forward test also runs on the untied variant, where a port that reads `embed_tokens` for the head fails.
5. **The taps are the OUTPUT of layers 1, 13, 25, 37, 49** (OC:112-131 with a list; spec 23 §8.1): the recorder's taps equal `hidden_states[i]` of transformers' own `output_hidden_states=[1, 13, ...]` call on an 8-layer tiny with `target_layer_ids` [1, 5] - bitwise.

---

### Task 1: the tiny model end to end in 5.19.0 (Mac)

**Files:**
- Create: `tools/oracle/muse_tiny_check.py`
- Test: its exit code

**Interfaces:**
- Consumes: `tools/oracle/qwen4exp_env.sh <dir>` (spec 21a; idempotent).
- Produces: nothing imported; the facts it prints go into Task 2's sheet.

- [x] **Step 1: fetch the tiny model** (`uvx --from huggingface_hub hf download optimum-intel-internal-testing/tiny-random-muse-glimmer`, 27 MB) and write `muse_tiny_check.py <snapshot>`: `AutoConfig` (prints `layer_types`, `layer_rope_theta`, `tie_word_embeddings`), `MuseGlimmerForConditionalGeneration.from_pretrained(..., dtype=bf16, attn_implementation="eager")` and fp32; a 24-id prompt: `generate(max_new_tokens=8, do_sample=False)` with the cache, then the same 32 ids as one uncached forward - the last 8 rows' logits equal the cached steps' (bitwise in fp32; in bf16 print max |diff|); a 40-id forward with `output_hidden_states=[1, 3]` returns exactly two non-None entries, layer 1's and layer 3's outputs (OC:115-131). Exit 0 only if all hold.
- [x] **Step 2: run it on the Mac:**

```sh
docker run --rm --memory 8g --memory-swap 8g --cpus 4 -e OMP_NUM_THREADS=3 -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/tmp/hf -v "$PWD":/ws -w /ws agnes-ref-img:latest sh -c \
  'tools/oracle/qwen4exp_env.sh oracle-out-q4exp/site && PYTHONPATH=/ws/oracle-out-q4exp/site \
   python3 tools/oracle/muse_tiny_check.py $(ls -d /hf/hub/models--optimum-intel-internal-testing--tiny-random-muse-glimmer/snapshots/*/)'
```
Expected: `tiny check: OK` plus the printed config lines. If 5.19.0 refuses the tiny config, stop and report (never switch the reference version).
- [x] **Step 3: commit** `oracle: the Muse-Glimmer tiny model end to end in transformers 5.19.0 (spec 23a)`.

### Task 2: the facts sheet (Mac: source and small files only)

**Files:**
- Create: `docs/probe-muse-glimmer-2026-10-15.md` (date = the day written), `tools/oracle/muse_facts.py`
- Modify: `docs/README.md` (one line)

**Interfaces:**
- Produces: `muse_facts.py <bf16-snapshot> <gptq-snapshot> [--scan-quant]` (small files; shard headers by HTTP range; with `--scan-quant` every `qzeros` and `g_idx` tensor read by range - ~210 MB in all, so the full scan is the box's, Task 6; on the Mac a sample) printing every table the sheet quotes.

- [x] **Step 1: the rounding chain**, one row per point with M:line - spec 23 §1.1 items 1-4 exactly, each with the module that computes it (`MuseGlimmerRMSNorm` with / without scale, `MuseGlimmerTextCenteredRMSNorm`, the qk scale's operand order, `apply_rotary_pos_emb` with bf16 cos / sin, `eager_attention_forward`, the gate, the sandwich, the logits chain). Add the masks: sliding `kv > q - 2048` (MU:93-102) built per layer type (M:499-502); NoPE by `layer_rope_theta[i] == 0` (M:512); the single rotary table from `rope_parameters` (M:186-191).
- [x] **Step 2: the checkpoint facts:** both repos' tensor groups, dtypes, shapes and bytes (spec 23 §2 / §3 tables, re-derived by the script, not copied); the GPTQ `quantization_config` and why `loader::QuantConfig::parse` accepts it (`src/loader/quant.h:21-33`); the sampled `qzeros` / `g_idx` values; the 809 vision names by prefix; `tie_word_embeddings` false and both `lm_head` / `embed_tokens` shipped; the per-layer bytes and the read-per-token table of spec 23 §2.
- [x] **Step 3: tokens and template:** `tokenizer.json` sha256 equal across both repos; the added-token table (200000-202047, the eight named ones of spec 23 §1); the post-processor adds BOS while the template writes `bos_token` (one BOS when rendered and tokenised as `apply_chat_template(tokenize=True)` does - printed ids of a one-message render); both templates' sha256 and their diff (spec 23 §1.2: the GPTQ one lacks the system-branch normalisation); `response_template` printed; `generation_config.json` (EOS {200001, 200008}, T 1.0, top-k 64, top-p 0.95).
- [x] **Step 4: the drafter's facts** for 23f: the assistant's config, its header table (5,111,970,304 B), its RMSNorm's two roundings (A:50-55), the window `|q - k| <= 2048` (MU:142-152) and the 15 drafts (G:1633-1660), the raw head rows; the GPTQ repo and the original sharing `tokenizer.json`.
- [x] **Step 5: commit** `probe: Muse-Glimmer-30B facts - rounding chain, checkpoints, tokens, the drafter (spec 23a)`.

### Task 3: `muse_ref.py` and MG1 on the tiny variants

**Files:**
- Create: `tools/oracle/muse_ref.py`, `tools/oracle/test_muse_ref.py`, `tools/oracle/muse_make_tiny.py`
- Modify: `tools/oracle/README.md` (a section "The Muse-Glimmer reference", the Files table)

**Interfaces:**
- Consumes: `tools/oracle/stream.py` (`attach`, `dequant_t` with `group_size` 64, `checkpoint_reader`), `qwen4exp_ref.py`'s patterns (copied where model-specific; `stream.py` is imported, not edited).
- Produces (imported by Tasks 5-7, 23b's `make_synth.py` / `check.py`, 23c's fixture generator, 23f's tap driver):

```python
# tools/oracle/muse_ref.py
PINNED_TRANSFORMERS = "5.19.0"
TAP_LAYERS = (1, 13, 25, 37, 49)            # the assistant's target_layer_ids (AC:89)
def text_config(snapshot: str, layers: int | None = None) -> "MuseGlimmerTextConfig"
    # config.json's text_config; layers=N truncates num_hidden_layers, layer_types[:N], layer_rope_theta[:N]
def expected_names(tc, form: str) -> list[str]   # form "gptq" (qweight/scales/qzeros/g_idx) | "bf16"
def vision_names(index: dict) -> list[str]       # every name the text path skips (809 on the real index)
def softcap(z: "Tensor", mult: float, cap: float) -> "Tensor"   # M:1178-1181's four bf16 roundings
def build_streamed(snapshot: str, tc, form: str) -> tuple["MuseGlimmerForConditionalGeneration", "Lazy"]
class Recorder:     # per layer l and row: H.L<l> (residual after the layer), attn.L<l>, mlp.L<l>;
                    # taps [rows][5][6656]; logits.raw / logits.cap (bf16 widened to f32); cap_tie (u8)
def layer_major(model, lazy)          # batched forward over several sequences, one layer at a time
def main()  # subcommands: run | ppl | hfcheck | facts
```

`run <snapshot> --prompt <ids> --out <p>.golden.safetensors --gen 32 [--layers N] [--taps] [--act-tail 256] [--logits-tail 1024]`: `H.L<l>` bf16 `[tail][6656]`, `logits.raw` / `logits.cap` f32 `[tail + gen][202048]`, `cap_tie` u8 `[T + gen]` (1 where the capped top-2 are equal in bf16 while the raw top-2 differ), `tokens`, `nll`, with `--taps` `taps` bf16 `[T + gen][5][6656]`; greedy continuation by argmax over the **capped** row (what `generate(do_sample=False)` does) with the raw argmax recorded beside it.

- [x] **Step 1: the variants** - `muse_make_tiny.py <tiny-snapshot> <out> --variant untied|l8|gqa16|win16|int4|assist`: `untied` writes an independent random `lm_head.weight` and `tie_word_embeddings: false`; `l8` 8 layers (`[s,s,s,f] x 2`); `gqa16` 16 q / 1 kv; `win16` `sliding_window` 16; `int4` the linears RTN-packed g64 in the GPTQ names (`qzeros` 0x77777777, `g_idx` i // 64, `quantization_config` copied from the real GPTQ config); `assist` a 2-layer `MuseGlimmerAssistantModel` (hidden 64, 4 q / 2 kv x 16, block 16, `target_layer_ids` [1, 5], `mask_token_id` 201818) for 23f. All bf16, seeded, written with the original's template and tokenizer files.
- [x] **Step 2: the failing tests** (`test_muse_ref.py`; each against transformers 5.19.0 un-streamed; SKIP with a message when the tiny is absent):
  - `test_streamed_equals_hf_bf16` / `_fp32`: on `untied`, `l8`, `gqa16` and `win16`: prompts of 40 and (on `win16`) 100 ids - crossing the window several times - then 8 cached decode steps: logits (raw and capped) and every `H.L*` bitwise (Review Focus 1); on `l8` once at 2100 ids with the real window 2048;
  - `test_norm_forms`: the embed norm and the final norm equal `MuseGlimmerRMSNorm` with / without scale bitwise and differ from an `rsqrt` restatement on a crafted vector where `pow` and `rsqrt` round apart (if no such vector exists in 10^6 random draws, the test records "indistinguishable" and passes - Review Focus 2);
  - `test_qk_scale_after_norm`: q = `bf16(bf16(norm(q)) * 3.87)` - a port scaling before the norm fails;
  - `test_nope_switch`: the full layer gets no rotary (a position shift of the whole prompt leaves its attention output equal; a sliding layer's moves);
  - `test_gate`: a zeroed `gate_proj` halves the attention output exactly (sigmoid(0) = 0.5);
  - `test_softcap_ties`: Review Focus 3;
  - `test_untied`: Review Focus 4 - the forward on `untied` differs from the same weights with the head tied;
  - `test_taps`: Review Focus 5 on `l8`;
  - `test_int4`: on `int4`, the dequant equals `(q - 8) * scale` and the forward equals transformers on the dequantised weights bitwise;
  - `test_layers_truncation`: `text_config(..., layers=4)` on `l8` equals transformers with `num_hidden_layers=4` bitwise;
  - `test_expected_names`: `expected_names(tc, "gptq")` equals the real GPTQ index's text names (1875) and `vision_names` the other 809; `"bf16"` the original's text names.
- [x] **Step 3: run, expect FAIL** (`ModuleNotFoundError: muse_ref`):

```sh
docker run --rm --memory 8g --memory-swap 8g --cpus 4 -e OMP_NUM_THREADS=3 -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/ws/oracle-out-q4exp/site -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/tmp/hf \
  -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/oracle/test_muse_ref.py
```
- [x] **Step 4: implement until every test passes;** `hfcheck <snapshot> --layers N` runs transformers' own model on real weights (layer-streamed, eager) against the port - for the box (Task 6). **Commit** `oracle: muse_ref.py - transformers 5.19.0's Muse-Glimmer layer-streamed, raw and capped logits, the drafter taps; MG1 bitwise on the tiny variants (spec 23a)`.
- [x] **Step 5: README section** (the site rule, subcommands, RAM / time per prompt ESTIMATED until Task 6 measures, what is bitwise against what). **Commit** `oracle: README - the Muse-Glimmer reference (spec 23a)`.

### Task 4: the golden prompts (Mac)

**Files:**
- Create: `tools/oracle/muse_prompts.py`, `tests/golden/prompts/muse_agentic.json`, `tests/golden/prompts/muse_{short_prose,short_code,4k,8k,32k,agentic}.ids`

- [x] **Step 1:** `muse_prompts.py <bf16-snapshot>` writes: `short_prose` / `short_code` = `prose.txt` / `code.txt` tokenised by Muse's tokenizer (no special tokens; < 2048: every window covers the prompt); `4k` / `8k` / `32k` = the first 4096 / 8192 / 32768 ids of the long text (`long32k.ids` is Qwen3.8's ids: decode it with Qwen3.8's tokenizer, `tok::default_tokenizer_json()`'s file, then encode with Muse's; print the char count used); `agentic` = `muse_agentic.json` - an original multi-turn coding session written for this repo (system prompt, a user task about this repository, assistant turns with `reasoning_content` and ATEM tool calls given as objects, tool results holding `src/runtime/kolibri/kolibri_sizes.h` and `src/loader/kolibri1_layout.h`), rendered with the original's template (`apply_chat_template`, `reasoning_strength="high"`, `current_date="2026-10-15"`, `add_generation_prompt=True`) to ~12k ids. Refuse a set where `4k` / `8k` / `32k` / `agentic` are not all > 2048; print each length and the BOS count of `agentic` (must be 1).
- [x] **Step 2: commit** `tests: Muse-Glimmer golden prompts - short, 4k / 8k / 32k and an agentic session past the window (spec 23a)`.

### Task 5: the A4 set in Muse's format (Mac)

**Files:**
- Create: `tests/golden/toolcall-muse/` (the 36 scenarios: `<name>.json` messages + `<name>.ids`, `manifest.json`)
- Modify: `tools/toolcall/make_set.py` (`--model muse`), `tools/toolcall/score.py` (`--format atem`), `tools/toolcall/test_make_set.py`, `tools/toolcall/test_score.py`

**Interfaces:**
- Produces: `make_set.py --from tests/golden/toolcall --model muse --tokenizer <bf16-snapshot> <out>`: the 36 conversations of Qwen3.8's set re-rendered through the original's template - tool-call arguments passed as objects (T:28-30 raises on strings), `reasoning_strength="low"`, `current_date="2026-10-15"`, tools from `tools/toolcall/tools.json`; `score.py --format atem`: the first call of a generation = the first complete `<atem:invoke>` in a non-`self`, non-`user` channel (spec 23 §6.5's selection rule), its name with `ns.ns` collapsed to a registered `ns`, arguments JSON-decoded per value else the raw string; a parse failure = a tool channel opened with no complete invoke.

- [x] **Step 1: failing tests:** `test_make_set.py::test_muse_render` (a scenario renders byte-identical to `apply_chat_template` on its messages; one BOS; no `<|patch|>`); `test_score.py::test_atem` (calls inside `to=self` ignored; a doubled name collapsed; `{"n": "7"}` from `<atem:parameter name="n">7</atem:parameter>` is `7`; an unterminated invoke is a parse failure; a turn whose first message has no `<|start|>`).
- [x] **Step 2: implement, PASS; write the set** (`tests/golden/toolcall-muse/`, committed - the tokenizer is Apache-2.0, the conversations are the repo's own). Print the id range (shortest / longest).
- [x] **Step 3: commit** `toolcall: the A4 set in Muse-Glimmer's ATEM format, score.py --format atem (spec 23a)`.

### Task 6: the box runner, perplexity, golden sets (box CPU)

**Files:**
- Create: `tools/box_validate/muse_oracle.sh`
- Modify: `tools/box_validate/data.sh` (`have muse_gptq`, `have muse_bf16`, `have oracle_muse`, `have oracle_muse_a4`), `tools/oracle/README.md`

**Interfaces:**
- Produces: `muse_oracle.sh <data tree> tiny|facts|ppl|hfcheck|golden|golden-bf16|a4|gateq|taps` (kolibri_oracle.sh's shape: resumable per prompt into `oracle-out-muse*.partial/`, moved when complete; `MemAvailable` floor `MUSE_REF_MIN_GB` (default 24 for tiny, 40 for the GPTQ form, 72 for bf16); exit 77 when short or when a snapshot is absent). `golden` -> `oracle-out-muse/<p>.{ids,golden.safetensors,log}` for every Task 4 prompt (GPTQ form dequantised, `--gen 32 --taps`); `golden-bf16` -> `oracle-out-muse-bf16/` (short and agentic only). Inputs: `SNAP_MUSE` (default the HF cache's `models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ`), `SNAP_MUSE_BF16` (`models--meta-models--Muse-Glimmer-30B`).

- [x] **Step 1: write the script and the `have` lines;** `DRY_RUN=1` prints every command, the RAM floor and an ESTIMATED time per prompt with no container (estimate: 25.2 B linear parameters dequantised per layer pass; the GPTQ form's dequant cost per forward).
- [ ] **Step 2 (box CPU): MG1's real-weight half** *(partly run on the Mac 2026-10-15: the qzeros scan of the -AutoRound repo CLEAN, `hfcheck --layers 4` on the int4 form BITWISE, the int4 perplexity of the golden prompts in their logs; left for the box: the -GPTQ repo's g_idx scan, `ppl` on both forms, `hfcheck` on bf16)* - `facts --scan-quant` (every `qzeros` 0x77777777, every `g_idx` = i // 64, counted), `ppl` on `prose.txt`, `code.txt` and the agentic transcript for both forms (finite, below a sanity ceiling of 30 on prose - ESTIMATED -, the numbers recorded), `hfcheck --layers 4` on both forms (bitwise).
- [ ] **Step 3 (box CPU): the golden sets** *(partly run on the Mac 2026-10-15 - spec 23 §12 lists which; the rest at ~53 ms a layer-token on the Mac is the box's)* - `golden` for every prompt, `golden-bf16` for `short_prose` and `agentic`; each log prints the `cap_tie` count and the raw top-1 / top-2 gap distribution (min, p01, median). Record times and RSS. **Commit** `oracle: muse_oracle.sh - resumable golden runs, perplexity, the full quant scan (spec 23a)` (outputs are git-ignored data).

### Task 7: gate Q (box CPU)

**Files:**
- Create: `tools/oracle/muse_gateq.py`, `tools/oracle/test_muse_gateq.py`
- Modify: `tools/toolcall/oracle_generate.py` (`--model muse`), `tools/toolcall/a4_ref.sh` (`muse`: `set` copies `tests/golden/toolcall-muse`; `ref` runs both forms), `tools/box_validate/muse_oracle.sh` (`a4`, `gateq`)

**Interfaces:**
- Produces: `oracle_generate.py --model muse [--form gptq|bf16] <snapshot> <set dir> <out dir>` (`muse_ref.py` streamed, greedy on the capped row, 512 new ids - K2's precedent for a reasoning-first model at `reasoning_strength` low; `--batch` / `--resident` as for the other models); `muse_gateq.py kl <bf16 run> <gptq snapshot> --set <dir> --out <json>`: teacher-forced on the bf16 run's continuations, per row KL(p_bf16 || p_gptq) over the capped distribution at T 1.0 and top-1 agreement, per scenario and pooled (mean, p50, p99); `muse_gateq.py verdict <a4 dir> <kl json>`: spec 23 §3.3's bars (decision 5) -> PASS / GAP with the numbers.

- [x] **Step 1: failing tests** (`test_muse_gateq.py`, on the tiny `int4` vs `untied` pair): KL of identical models is exactly 0 and top-1 1.0; the verdict reads a planted A4 table (parse failures, agreement) and a KL json and prints PASS / GAP by the bars.
- [x] **Step 2: implement, PASS.** Commit `oracle: gate Q - the GPTQ checkpoint against the bf16 original, A4 and teacher-forced KL (spec 23a)`.
- [ ] **Step 3 (box CPU):** (why this gate decides: the checkpoint was calibrated on AutoRound's default `NeelNanda/pile-10k`, 512 x 2048 tokens of web text with no chat template, channels or ATEM calls - spec 23 §3.1, auto-round `da52d6da`'s `examples/Muse-Glimmer/Muse-Glimmer_W4A16.ipynb`.) `muse_oracle.sh <data> a4` (both forms, 36 scenarios, resumable, hours - DRY_RUN's estimate first), then `gateq`; `score.py --format atem <out> bf16 gptq`; the verdict. **If GAP:** record it in spec 23 §3.3 and stop - stage 23q (our quantisation) is the operator's call; 23b-23e continue on the synthetic checkpoints meanwhile. *(Superseded 2026-10-15, operator: the quant is kept - gate Q is a recorded measurement, `verdict` prints WITHIN / OUTSIDE and exits 0. Not run on the Mac: ~90 h a form ESTIMATED; queue row 42's `r42.a4` / `r42.gateq`.)*

### Task 8: docs, the queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-spec23-muse-glimmer-design.md` (a section "23a as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 42), `tools/box_validate/stages.sh` (its `row 42` block)

- [x] **Step 1:** "23a as built": the version that loaded the tiny, departures from this plan, the measured perplexities, the `cap_tie` counts, gate Q's verdict with its numbers.
- [x] **Step 2: the queue row 42** - merged = the reference and its tooling, nothing on the card. **Stage list (sketch; this plan's implementer writes them):**
  - `r42.host` default cpu: `test_muse_ref.py`, `test_muse_gateq.py`, the toolcall tests, in the oracle image with the 5.19.0 site;
  - `r42.facts` default cpu `muse_gptq`: `muse_oracle.sh facts` (the full quant scan);
  - `r42.ppl` / `r42.hfcheck` optin cpu `muse_gptq` (and `muse_bf16` for the bf16 half);
  - `r42.golden` optin cpu `muse_gptq,oracle_image`: every prompt with taps (hours);
  - `r42.golden_bf16` optin cpu `muse_bf16`;
  - `r42.a4` optin cpu `muse_gptq,muse_bf16`: both A4 references (hours);
  - `r42.gateq` optin cpu `oracle_muse_a4`: KL / top-1 and the verdict.
  `rownote`s: the downloads (22.8 GB + 59.6 GB; `df -h` first), the RAM floors, decision 5's bars are PROPOSED. `python3 tools/box_validate/test_box_validate.py` passes; `tools/box_validate.sh --dry-run --only r42` prints them.
- [x] **Step 3: commit** `docs: spec 23 (23a as built), box queue row 42 - the Muse-Glimmer reference`.

**Gate for the plan:** Mac - Task 1's tiny check OK; every test of `test_muse_ref.py` / `test_muse_gateq.py` / the toolcall tests green; the facts sheet complete (every §1.1 point cited); the prompts and the A4 set committed. Box CPU (when the weights exist) - the full quant scan clean, perplexities sane, `hfcheck` bitwise, the golden sets with taps written, gate Q's verdict recorded.

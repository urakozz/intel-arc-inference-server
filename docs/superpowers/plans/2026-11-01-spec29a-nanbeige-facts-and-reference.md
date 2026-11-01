# Spec 29a - Nanbeige4.3-33B: the facts, the f32 judge and the bf16 twin, tiny models, golden and A4 sets, tokenizer / template parity

**Status (2026-11-01): PLANNED.** Nothing built. No box queue row (operator rule: rows only when asked); the box stages are in prose at the end.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; `WIP:` if needed); write a handoff file when you stop.

**Goal:** the reference every later spec 29 stage is graded against: the facts sheet with every rounding point of HF's Llama pinned (RMSNorm two-rounding form, bf16 RoPE chain, eager softmax dtype), `tools/oracle/dense_ref.py` (the **f32 judge** of the dequantised int4 g64 weights, layer-streamed, and the **bf16 twin** = the same weights in plain bf16 `LlamaForCausalLM`, eager attention), tiny random Llama and tiny int4 g64 builders, the golden / A4 / agentic prompt sets, tokenizer and template parity (spec 29 D6), the RoPE-table precision measurement, and the amendment of `docs/14-golden-gate.md` with the near-miss rule (spec 29 §10).

**Architecture:** spec 29 §1, §10, §11 (R6, R9, decision 7 and 9). Precedents: plan 25a (Bolt) task for task, `tools/oracle/stream.py` (layer streaming), `dequant.py` (`(q - 8) x scale`), `ornith_ref.py --dtype f32 --force` (f32 teacher-forced judge), `kolibri_ref.py`.

**Tech Stack:** Python 3 (the Mac's system python has neither torch nor transformers: use the oracle container `tools/oracle/run_in_container.sh` / `agnes-ref-img` and the pinned transformers site as plan 25a does), safetensors, the repo's Rust tokenizer test harness.

**Spec:** `docs/superpowers/specs/2026-11-01-spec29-nanbeige-design.md`.

## Dependencies and branch points

- None on the engine. Branch `spec29a-nanbeige-reference` from main; signed commits; no merge, no push. Box tree automatic; `tools/box.env` is the operator's, never printed or committed.
- **Weights:** the Mac has none. The box has `Nanbeige/Nanbeige4.3-33B` (bf16, 66 GB) and `urakozz/Nanbeige4.3-33B-W4A16-g64-AutoRound-GPTQ` (4 shards). The operator downloads; this plan only reads. Small files (config, tokenizer, template) are fetched with `curl` / range reads on the Mac.
- No public tiny Llama of this shape is in the HF cache (checked 2026-11-01): the tiny models are built in code.

## Global Constraints

- One pinned transformers version (the one plan 25a pins; recorded in the facts sheet); refused by name otherwise. Reference attention is **eager** (diagnostic path, the judge); sdpa is not used.
- Nothing of the model is vendored: configs are built at run time from the cached `config.json`; tests that need the real config SKIP (exit 77) without the cache. Only the tokenizer fixtures the repo already commits for other families may be committed (precedent: `tests/tokenizer/kolibri`, `bolt`) - **ask the operator before committing Nanbeige's 30 MB-class `tokenizer.json`** (open item; a hash + a regeneration script is the default).
- CPU model runs: container, `--memory` capped, one at a time, the lock `/tmp/b70-cpu-model.lock`; box RAM checked first (`MemAvailable` >= the script's floor); detached and resumable.
- Every number in docs is measured, derived, published or estimated, marked.

## Review Focus

1. **The two-rounding norm.** `weight * hidden.to(bf16)` after an fp32 normalisation: the facts sheet cites the transformers lines; the judge in fp32 has no rounding, the twin has HF's. A test shows the engine's single-rounding form differs from the twin on random rows (so 29c's `dn_` norm define matters and is selected by measurement).
2. **The RoPE chain.** fp32 `inv_freq`, fp32 `pos x inv_freq`, cos / sin cast to the activation dtype, `q x cos + rotate_half(q) x sin` in bf16 ops. The table-precision measurement (Task 5) compares fp32-product and fp64 angles at positions 1k..355k and prints the max angle error per pair and the logits effect on a tiny model.
3. **Eager softmax dtype.** HF eager computes softmax in fp32 then casts to the query dtype before P x V (a rounding of the probabilities); the judge (fp32 everywhere) and the twin (bf16) differ here; the engine's flash kernels keep fp32 until the end. Recorded.
4. **The judge is of the engine's weights**, not of the original bf16 weights: the int4 g64 dequant `(q - 8) x scale` (signed scales allowed - AutoRound's form), scales fp16, qzeros ignored for sym (checked all 0x88888888-style).
5. **Padding ids.** Ids 166107..166911 exist in the head; the judge's argmax masks them exactly as the engine will (`VOCAB_USED` 166107), and a test plants a padding row with a huge logit.

---

### Task 1: the facts sheet (Mac: source, small files, range reads)

**Files:** Create `docs/probe-nanbeige-2026-11-<dd>.md`, `tools/oracle/dense_facts.py` (prints the tables; `--headers` range-reads the shard headers of both repos at pinned revisions into a cache outside the tree).

- [ ] **Step 1:** the rounding chain, one row per point with transformers `file:line` citations: RMSNorm, RoPE (`apply_rotary_pos_emb`, `LlamaRotaryEmbedding`), eager attention (`eager_attention_forward`: scale, softmax dtype), MLP, the residual adds (bf16), the final norm, the head.
- [ ] **Step 2:** the checkpoint facts from the shard headers: tensor names and dtypes (471 in the original; the quantised repo's `qweight / qzeros / scales` per linear), shapes, `quantization_config.json`, sym check on a sampled `qzeros`, the derived byte tables of spec 29 §2 re-derived from headers.
- [ ] **Step 3:** tokens and template: `added_tokens`, the 8,003-char template's minja feature list, rendered examples (no system, system, tools, a tool call with string / number / bool / null / list / dict arguments, tool results grouped, `preserve_thinking` true / false / undefined, `enable_thinking` false), the generation-prompt endings.
- [ ] **Step 4: commit** `docs: Nanbeige4.3-33B facts - rounding chain, checkpoint headers, tokens, template (spec 29a)`.

### Task 2: tiny models (no card)

**Files:** Create `tools/oracle/dense_make_tiny.py`, `tools/oracle/test_dense_make_tiny.py`.

- [ ] **Step 1: failing tests**, then **Step 2: implement.** `dense_make_tiny.py <out> --variant NAME [--seed 0]`, `save_pretrained`-shaped, real tokenizer / template copied from the cache at run time: `tiny` (4 layers, hidden 256, 4 q / 2 kv heads x 128 - GQA 2 stands in; **`tiny-gqa7`**: 14 q / 2 kv x 128 and **`tiny-gqa7-8kv`**: 56 q / 8 kv with hidden 7168 and 2 layers for the real GQA), `int4-g64` (every linear RTN int4 g64 sym in the `auto_round:auto_gptq` bytes, signed scales allowed, the real names), `padded-vocab` (head rows beyond `VOCAB_USED`). Tests: each variant loads in the pinned transformers; the dequant of `int4-g64` equals `(q - 8) x scale` exactly.
- [ ] **Step 3: commit** `oracle: tiny Llama variants and int4 g64 exports in the checkpoint's bytes (spec 29a)`.

### Task 3: `dense_ref.py` - the judge and the twin

**Files:** Create `tools/oracle/dense_ref.py`, `tools/oracle/test_dense_ref.py`; modify `tools/oracle/README.md`.

- [ ] **Step 1: failing tests** (each against the unmodified `LlamaForCausalLM` on the tiny variants): the judge in fp32 equals HF in fp32 within fp32 bounds (prompt and cached decode, chunks = one forward); the twin equals HF bf16 bit for bit; `build_streamed(weights="int4")` equals the port on the dequantised twin bitwise; the padding mask; the norm / RoPE / softmax restatements each tested bitwise against the module they restate.
- [ ] **Step 2: implement** (`run <snapshot> --prompt <ids> --out <p>.golden.safetensors --gen 32 [--layers N] [--weights int4|bf16] [--dtype f32|bf16] [--force] [--taps 1,25,49]` - the golden layout of plan 24a: `h.L<l>` for the last rows, logits rows; the tap dump is spec 29f's input). Layer-streamed (`stream.attach`), resumable per prompt.
- [ ] **Step 3: commit** `oracle: dense_ref.py - the f32 judge and bf16 twin of Nanbeige's int4 g64 weights (spec 29a)`.

### Task 4: the golden prompts, the A4 set, the agentic session (Mac)

**Files:** Create `tools/oracle/dense_prompts.py`, `tests/golden/prompts/dense_{short,4k,8k,agentic}.ids`, `tests/golden/toolcall-dense/` (the 36 scenarios), modify `tools/toolcall/{make_set,oracle_generate,a4_ref.sh,score}.py` (`--model dense`), and `tests/golden/prompts/dense_agentic_long.json` (a >= 150k-token coding session written for this repo: system prompt, a user task about this repository, tool calls and results of realistic size - no third-party text; built by a generator script, ids from the template).

- [ ] **Step 1: failing tests:** a scenario re-rendered by the engine-side template equals `apply_chat_template` byte for byte; `score.py`'s Qwen XML scoring reads Nanbeige's outputs (reasoning, `</think>`, content, `<tool_call>` blocks).
- [ ] **Step 2: implement; PASS; commit** `tests: Nanbeige golden prompts, A4 set and a long agentic session (spec 29a)`.

### Task 5: RoPE table precision (Mac, minutes)

**Files:** Create `tools/oracle/dense_rope_check.py`, `docs/probe-nanbeige-rope-2026-11-<dd>.md`.

- [ ] **Step 1:** at positions 1k, 8k, 32k, 128k, 240k, 330k, 355k: the fp32-product angle against the fp64 angle per pair (max, median), the bf16-rounded cos / sin of each form and how often they differ, the effect on a tiny GQA-7 model's logits cosine over a 4k window with positions offset to those depths. **Step 2:** the verdict (which form the engine's table uses - spec 29 decision 9) in the doc. **Step 3: commit** `probe: RoPE angle precision at 300k+ positions, theta 1.2e8 (spec 29a)`.

### Task 6: tokenizer and template parity (D6)

**Files:** Create `tools/tokenizer/dump_dense.py`, `tests/tokenizer/dense/template_cases.json`, `tests/tokenizer/dense_tokenizer_test.cc`, `tools/tokenizer/dense_strings.py` (>= 10,000 strings: code in several languages, CJK, tool XML, runs of spaces / tabs / newlines, byte-fallback bytes, special-token text, `<think>` forms).

- [ ] **Step 1:** `dump_dense.py` encodes every string with the slow `LlamaTokenizer(use_fast=False)` and with `tokenizer.json` (HF fast) and diffs them; any difference is listed. **Step 2:** the engine-side test loads `tokenizer.json` through `tok::Tokenizer` (the Rust `tokenizers` crate) and compares id for id with the slow tokenizer's ids. **Step 3:** the 8,003-char template through `chat::Template` (minja) against transformers for >= 40 conversations (`preserve_thinking` / `enable_thinking` / tools / grouped tool results / reasoning_content / `tool_call_format`), byte for byte; `ChatFormat` is not touched in this plan (29e), the test calls the renderer directly. Any mismatch is a finding and is fixed in the tokenizer / template handling in this plan (R6), not worked around.
- [ ] **Step 4: commit** `tokenizer: Nanbeige - tokenizer.json against the slow LlamaTokenizer, the template byte for byte (spec 29a)`.

### Task 7: the box CPU runner

**Files:** Create `tools/box_validate/dense_oracle.sh`; modify `tools/box_validate/data.sh` (`have` lines only; **no `stages.sh` rows**).

- [ ] **Step 1:** `dense_oracle.sh <data tree> tests|ppl|golden|golden-twin|a4|a4-twin|taps` (bolt_oracle.sh's shape: resumable per prompt into `*.partial/`, `MemAvailable` floor, exit 77 when short, `DRY_RUN=1`). **Step 2 (box CPU, detached):** `ppl` on prose / code / agentic (finite; recorded), the golden sets in the judge and in the twin, the near-miss statistics of the twin against the judge (the baseline spec 29 §10 compares the engine with), then the A4 references (days) and the tap dumps (for 29f). **Step 3: commit.**

### Task 8: docs

- [ ] **Step 1:** amend `docs/14-golden-gate.md` with spec 29 §10's near-miss rule (8 ulp, twin, 2x cosine distance) beside the one-ulp rule it keeps for Ornith / K2; `docs/README.md` one line; spec 29 gets a "29a as built" section (as-built facts, departures, the rope verdict, the tokenizer result, the twin-against-judge baseline). **Step 2: commit** `docs: spec 29 (29a as built), the near-miss rule in the golden-gate doc`.

**Gate for the plan:** Mac - every Python test green in the container; the facts sheet complete (each rounding point cited, header tables, template features); D6 (tokenizer ids and template bytes) **clean or every difference fixed**; the rope doc has its verdict. Box CPU - `ppl` finite, golden / A4 / tap outputs for the judge and the twin exist, the twin-against-judge baseline table is in the as-built section. **Handoff:** the judge's golden files and the twin's miss counts are the numbers 29c / 29d's D3 reads.

**Box stages (prose):** CPU only, no GPU lock. Order: `tests` (minutes), RoPE check (minutes), `ppl` (hours), `golden` and `golden-twin` (hours each, resumable), `a4` and `a4-twin` and `taps` (days, resumable, detached; RAM floor ~48 GB for the streamed judge, checked first).

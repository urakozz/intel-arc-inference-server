# Spec 14a - Agnes: the CPU reference, the fold proof, the golden sets

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a CPU reference for Agnes 3.0 Flash in `tools/oracle/`, checked against the checkpoint's own `modeling_agnes.py`; the proof on real data that the folded MLP equals the two-branch MLP (G1, CPU); Agnes golden sets and tool-call reference outputs dumped.

**Architecture:** `tools/oracle/dump.py` builds transformers' `Qwen3_5ForCausalLM` from a dequantised state dict (`tools/oracle/dequant.py`). For Agnes: a `name_map` (`delta_attn.` → `linear_attn.`, `global_attn.` → `self_attn.`) and a subclass whose MLP adds `parallel_ffn` (SwiGLU, intermediate 2048), as vLLM PR #57003's `AgnesMLP` does. `layer_types` from `global_attention_interval` = 4.

**Tech Stack:** Python 3 + torch + transformers (CPU, the oracle container `tools/oracle/run_in_container.sh`).

**Spec:** `docs/specs/2026-10-02-spec14-agnes-3-flash-design.md` (§1, §2 the fold, §3.4, §4 G1-G3, §6 14a).

## Global Constraints

- Branch `spec14a-agnes-oracle` from main; box tree `~/b70-inference-server-spec14a` (automatic, `tools/box.sh dir`). Copy `tools/box.env` from the main checkout if missing; never commit it or its contents.
- The checkpoint must be in the box's HF cache: `uvx --from huggingface_hub hf download urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ` (22.9 GB) and, for G3's reference, `Agnes-AI/Agnes-3.0-Flash` (bf16; check its size with `hf download --dry-run` first and the box's free disk with `df -h ~`; if it does not fit, say so and use the dequantised W4A16 as G3's reference, labelled).
- CPU oracle runs: `free -g` >= 70 GB available first (Agnes in bf16 is ~65 GB: check it fits with the container; if not, run layer by layer as `tools/oracle/last_logits.py`'s chunked path does), one at a time, detached (`tools/probe/detach.sh`), polled with a background until-loop. No GPU.
- Qwen3.8's oracle path is unchanged (its golden sets must reproduce bit for bit: rerun one Qwen3.8 golden prompt dump and compare).
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **`modeling_agnes.py` is the ground truth**, our subclass the tool: compare logits on a 64-id prompt (cosine >= 0.99999, same argmax at every position) before dumping anything.
2. **The fold on packed int4:** the CPU proof uses the GPTQ tensors exactly as the engine's loader will see them (stack `qweight`/`scales`/`qzeros` rows for gate and up; concatenate along the packed input dim for down at row 17408 / 8 = 2176 of `qweight`, group 272 of `scales`), dequantises the folded tensors, and compares the folded MLP output with the two-branch output on real layer inputs captured from the reference.
3. **`g_idx`** is present in the checkpoint: with `desc_act: false` it must be the identity grouping; assert it, because the fold relies on it.
4. **The MTP head** (`mtp.*`, 15 tensors) loads unchanged; record its presence for 14c.
5. The chat template (`chat_template.jinja`) is diffed against Qwen3.8's: tool-call format and `<think>` handling, for 14d.

---

### Task 1: the Agnes reference

**Files:** `tools/oracle/agnes.py` (name map, `AgnesForCausalLM` subclass of `Qwen3_5ForCausalLM` adding `parallel_ffn` per layer), `tools/oracle/dump.py` (detect `model_type == "agnes"` in `config.json` and use it), `tools/oracle/test_agnes.py`.

- [ ] **Step 1:** `test_agnes.py`: config translation (72 layers, FA at `l % 4 == 3`, parallel FFN 2048), name map on a list of real checkpoint keys (from `model.safetensors.index.json`), and the two-branch MLP on random weights equal to a hand-written reference.
- [ ] **Step 2:** implement; run `test_agnes.py` in the container.
- [ ] **Step 3:** Review Focus 1: our reference vs `modeling_agnes.py` (`trust_remote_code=True`) on a 64-id prompt, logits per position. Record in `docs/probe-agnes-2026-10-02.md`.
- [ ] **Step 4: Commit** `oracle: an Agnes 3.0 Flash CPU reference (spec 14a)`.

### Task 2: the fold proof (G1, CPU)

**Files:** `tools/oracle/agnes_fold.py`, `tools/oracle/test_agnes_fold.py`.

- [ ] **Step 1:** test: on a synthetic GPTQ layer pair (random int4, g64), the packed fold dequantises to exactly `[W_gate ; W_gate_p]`, `[W_up ; W_up_p]`, `[W_down | W_down_p]`.
- [ ] **Step 2:** on the real checkpoint, for layers 0, 3, 35, 71: capture each MLP's input from a reference forward on a golden prompt; folded MLP vs two-branch MLP output, cosine per row >= 0.999999 and max abs; Review Focus 2, 3.
- [ ] **Step 3:** doc section; **Commit** `oracle: the parallel-FFN fold, proved on the checkpoint (spec 14 G1)`.

### Task 3: golden sets and tool-call references

- [ ] **Step 1:** dump the golden prompts (prose, code, cjk; `tools/oracle/golden.sh`) for Agnes into `oracle-out-agnes/` on the box (not committed; note the path in `tools/oracle/README.md`).
- [ ] **Step 2:** the A4 tool-call set's reference outputs from Agnes (bf16 if it fits, else dequantised, labelled) with `tools/toolcall/oracle_generate.*`, into `~/agnes-toolcall-ref/` on the box.
- [ ] **Step 3:** Review Focus 4 and 5 in the doc; Qwen3.8 golden reproduction check (one prompt, bitwise); `docs/README.md` line. **Commit** `oracle: Agnes golden sets and tool-call references (spec 14a)`.

**Gate for the plan:** the reference matches `modeling_agnes.py`, G1 holds on the checkpoint, golden sets and A4 references dumped, Qwen3.8's oracle unchanged. Hand back with the doc's tables.

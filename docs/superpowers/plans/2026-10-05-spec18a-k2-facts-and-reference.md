# Spec 18a - K2-Horizon: facts, the CPU reference, golden sets

**Status (2026-10-05, branch `spec18a-k2-reference`):** the facts the small files decide and the
reference are written and tested; nothing has run on real weights, the box or a card.
- **Done:** `tools/oracle/k2_ref.py` + `test_k2_ref.py` (Task 3 Step 1, under these names rather
  than `k2.py` / `test_k2.py`: a plain-torch port of `modeling_k2_horizon.py` - vendored in
  `tools/oracle/third_party/k2_horizon/` - layer at a time with routing dumps, **bit-identical to
  the vendored HF model in bf16** on tiny random weights, prompt and cached decode; 18 tests, all
  pass in `agnes-ref-img`). The facts from config, both indexes, the shard headers, range-fetched
  small tensors and the modeling code: `docs/probe-k2-2026-10-05.md` (Task 1 (a); (c)'s BOS
  decision: K2 ids start with 0, `tokenize.py encode --bos`). Review Focus 1 (every §3 line, file:line)
  is in that doc; the re-read found no semantic change, only rounding detail, and no HF-vs-vLLM
  semantic disagreement.
- **Pending:** Task 1 (b) and Task 2 (box / card); Task 3 Step 2 and Task 4 (the real-weight run:
  golden sets, gap distribution, `hfcheck`, the template render) - commands in
  `tools/oracle/README.md` "The K2-Horizon reference". Operator's call: the int4 checkpoint for the
  gate's golden sets (what the engine runs; ~5-6 min per prompt, fits the Mac's 28 GB container),
  the bf16 one on the box for quantisation damage (~5 min per prompt warm, 75 GB) - both estimated.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the facts K2's build depends on, re-checked against today's engine; a layer-streamed CPU reference of `modeling_k2_horizon.py` with per-layer MoE and MoVA routing dumps; golden sets on K2's tokenisation; the chat-template check.

**Architecture:** the old stage-0 and oracle plans carried forward — **read them first for task detail**: `git show d307d0e:docs/superpowers/plans/2026-09-14-plan8a-spec4-stage0-facts.md` and `...plan8d-spec4-oracle.md`. What changes: the reference is **layer-streamed** with spec 14's `tools/oracle/stream.py` (the old plan materialised the whole dequantised model, ~87 GB estimated), it records MoVA and MoE expert ids and weights per layer and position, and the template check is added.

**Tech Stack:** Python 3, torch 2.14.1 CPU, transformers 5.15.0 (the oracle image / the Mac's `agnes-ref` image), C++17 + Level Zero for the device facts.

**Spec:** `docs/superpowers/specs/2026-10-05-spec18-k2-horizon-design.md` (§1, §3, §4 decision 4, §5.4, §6 K2, §8 18a).

## Global Constraints

- Branch `spec18a-k2-reference` from main; commit there; no merge, no push. Box tree automatic (`tools/box.sh dir`); `tools/box.env` copied from the main checkout if missing, never committed.
- **Where the CPU work runs** is the operator's call (spec §8): the box's oracle container, or Docker on the Mac if the 22.2 GB checkpoint fits there (container capped at 28 GB, layer-streamed). Device facts (Task 1 part b, Task 2) need the box.
- CPU runs: `free -g` >= 70 GB available on the box (or the 28 GB cap on the Mac), one at a time, detached, polled with a background until-loop. GPU work under `flock ~/b70-gpu.lock`.
- No `rm -rf`. Nothing about third-party engines in the repo.

## Review Focus

1. **Every line of spec §3 re-read against `modeling_k2_horizon.py`** (file:line cited): the grouped plain RMSNorm, the selection-only bias, × 2.5 after normalising, ascending-id order, the softplus threshold, RoPE dim 128 with `rotate_half`, GQA 4.
2. **The routing dumps** use the reference's own top-k, so ties are broken exactly as the engine must break them (spec 4 risk 2): record the 8th/9th (MoE) and 4th/5th (MoVA) selection-score gaps to size the routing diagnostic's tolerance.
3. **The stream** must reproduce the full model's logits: on a 32-id prompt, the streamed forward against the non-streamed one (both CPU), bitwise or within fp32 summation noise.
4. **BOS:** whether K2's tokenizer prepends id 0 by default, decided and recorded (spec 4 §4); prompt counts in the 24-64 window.
5. **The template:** K2's `chat_template.jinja` (51 KB, a configurable `tool_call_format`) rendered with HF's `apply_chat_template` on plain, thinking, tools, a tool call in history and a tool response; the emitted tool-call format named for 18d.

---

### Task 1: facts (old plan 8a Tasks 1-3, updated)

- [ ] **(a) quant checks** on the current checkpoint (int4 g64 sym, sequential `g_idx`, which tensors are bf16), **(b) device allocations** on the box (per-layer expert buffers 196.6 / 98.3 / 83.9 MB, KV per layer at 32k and 64k), **(c) K2 prompt ids** and BOS (Review Focus 4). `docs/probe-k2-2026-10-05.md`. **Commit** `probe: K2-Horizon facts (spec 18a)`.

### Task 2: the launch floor

- [ ] Reuse spec 15 P0's per-kernel floor if measured; else old plan 8a Task 2's launch probe at K2's small GEMV shape (2560 → 1536), replayed. **Commit** `probe: per-launch floor at K2 shapes (spec 18a)`.

### Task 3: the reference

**Files:** `tools/oracle/k2.py` (load glue for `modeling_k2_horizon.py` with `trust_remote_code`, routing hooks), `tools/oracle/test_k2.py`. **As built:** `tools/oracle/k2_ref.py` (a port of the modeling code, not load glue: the routing is recorded by the port's own router, and `hfcheck` runs the vendored modeling file layer-streamed against it), `tools/oracle/test_k2_ref.py`.

- [x] **Step 1:** tests on a tiny random K2 config (old plan 8d Task 1's approach): the streamed forward equals the full forward; routing dumps sorted by ascending id. (2026-10-05: plus every §3 trap and bitwise equality with the vendored HF model.)
- [ ] **Step 2:** Review Focus 1 and 3 on the real checkpoint; record in the doc. **Commit** `oracle: a layer-streamed K2-Horizon reference with routing dumps (spec 18a)`.

### Task 4: golden sets and the template

- [ ] **Step 1:** prose / code / cjk on K2's ids + 32 greedy tokens, with per-layer MoE and MoVA ids and weights, into `oracle-out-k2/` (gitignored); Review Focus 2's gap distribution; timing per prompt.
- [ ] **Step 2:** Review Focus 5; the A4 tool-call set re-rendered with K2's template (`tools/toolcall/make_set.py --from`, spec 14's method) into `tests/golden/toolcall-k2/`.
- [ ] **Step 3:** `docs/README.md` line. **Commit** `oracle: K2 golden sets, routing dumps, template check (spec 18a)`.

**Gate for the plan:** every fact recorded, the reference matches the non-streamed forward, golden sets and routing dumps exist, the template's tool-call format is named. Hand back with the doc's tables.

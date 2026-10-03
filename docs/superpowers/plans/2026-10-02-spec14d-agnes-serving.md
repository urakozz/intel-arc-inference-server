# Spec 14d - Agnes in the server: chat template, tool calls, A4, the vLLM row

> **Superseded 2026-10-03** by `2026-10-03-spec14-write-phase.md` and `2026-10-03-spec14-validation-checklist.md` (spec 14 §6). Kept for task detail.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ` serves chat with the checkpoint's template, reasoning and tool calls parsed (spec 7 §3.5 or an addition), A4 measured (G3), and one llama-benchy row against vLLM running the operator's PR #57003; the record.

**Architecture:** the server's template (`src/tokenizer/chat_template*`) renders the checkpoint's `chat_template.jinja`; the tool-call parser (`src/server/toolcall.{h,cc}`) is reused if Agnes emits the Qwen XML format, otherwise extended with Agnes's format behind the model descriptor. vLLM for the comparison: the box's vLLM XPU image with the PR branch's files overlaid, as the PR's test plan did.

**Tech Stack:** C++20, nlohmann::json, Jinja (the engine's template renderer), Python (A4 tooling), uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-02-spec14-agnes-3-flash-design.md` (§3.5, §4 G3, §5 vLLM row, §6 14d). Needs 14c merged, and 14a's A4 reference outputs (`~/agnes-toolcall-ref/` on the box).

## Global Constraints

- Branch `spec14d-agnes-serving` from main; box tree `~/b70-inference-server-spec14d`. Copy `tools/box.env` if missing; never commit it.
- Every GPU command (b70-serve, vLLM) under `flock ~/b70-gpu.lock`, detached, polled; the vLLM server and b70-serve never hold the card at the same time.
- Qwen3.8's serving tests (`golden_server_test`, `toolcall_test`, `prefix_server_test`, `mtp_server_test`) unchanged.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **The template diff** (14a recorded it): `<think>` handling, `preserve_thinking`, the tool-call block and the `<tool_response>` turn; the renderer handles every construct the template uses (a render test against the HF `apply_chat_template` output on 5 message lists, including tools and a tool response).
2. **Tool-call format:** take one A4 generation from Agnes and check it parses; if Agnes's format differs from Qwen XML, the parser addition gets the same streaming tests as spec 7a (byte-at-a-time equals whole-text).
3. **The vLLM comparison is like for like:** same checkpoint files, same max_len (65536 or less on both), same llama-benchy flags (the operator's: `--pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation`), b70-serve with `--lm-head bf16 --mtp 0` for the byte-matched row and the serving defaults for a second row.
4. **Prefix caching on Agnes** through the server: `llama-benchy --enable-prefix-caching` at depth 4k / 16k / 32k, cache on vs off (as spec 7's BENCHMARKS rows).
5. A4's reference is labelled (bf16 Agnes, or dequantised W4A16 if bf16 did not fit in 14a).

---

### Task 1: template and tool calls

- [ ] Render test (Review Focus 1); parser check (Review Focus 2) and, if needed, the addition with its tests; `golden_server_test`-style test for Agnes (a short greedy chat through the server equal to `b70-decode` on the same ids). **Commit** `server: Agnes chat template and tool calls (spec 14d)`.

### Task 2: A4 (G3)

- [ ] Run the 36-prompt tool-call set through `tools/toolcall/engine_generate.sh` on Agnes (`l0-int8`, serving defaults) and score with `tools/toolcall/score.py` against 14a's references; report the match count next to Qwen3.8's 25/36 (no bar). **Commit** `toolcall: A4 on Agnes (spec 14 G3)`.

### Task 3: the vLLM row and prefix caching

- [ ] **Step 1:** vLLM with the PR (overlay the PR branch's eight files onto the box's vLLM XPU image, serve `--language-model-only --max-model-len 65536`), llama-benchy with the operator's flags; b70-serve on the same flags (two rows: byte-matched and serving defaults).
- [ ] **Step 2:** Review Focus 4's prefix-caching rows.
- [ ] **Step 3: the record:** `docs/BENCHMARKS.md` "Agnes 3.0 Flash (spec 14)" (14c's rows + these); spec 14 amendment §8 with every gate and number; `README.md` (the scope section names both models; an Agnes row under the headline table); `docs/03-models.md` and `docs/13-loader.md` (the second model and the fold).
- [ ] **Step 4: Commit** `spec 14: Agnes 3.0 Flash served - A4, vLLM row, prefix caching (the record)`.

**Gate for the plan:** the template and parser tests, A4 reported, the vLLM and prefix-caching rows recorded, Qwen3.8's serving tests unchanged, full suite green.

# Spec 18d - K2-Horizon in the server

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve` serves K2: its chat template and tool-call format, reasoning, KV-only prefix-cache snapshots (spec 7), int8 `lm_head` by default; K4 (A4, passkey); the comparison rows against vLLM; the record.

**Architecture:** spec 18 §5.4. The server dispatches on `model_type`; K2's template is rendered by the engine's Jinja renderer (checked against HF) and its tool calls parsed by the existing parser or an addition for K2's format (18a named it). Spec 7's store and block hook apply with snapshots that carry no GDN state.

**Tech Stack:** C++17, nlohmann::json, the engine's Jinja renderer, Level Zero, Python (A4), uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-05-spec18-k2-horizon-design.md` (§5.4, §6 K3, K4, §7). Needs 18b and 18c merged; 18a's template check and `tests/golden/toolcall-k2/`.

## Global Constraints

- Branch `spec18d-k2-serving` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- K0 holds; the Qwen3.8 serving tests unchanged. Lock, detached, polled.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **The template renders identically** to HF's `apply_chat_template` on 18a's five message lists, including the configured `tool_call_format`; anything the renderer cannot express is a renderer change with its own test, not a K2 special case in the server.
2. **Tool calls:** if K2's format is not Qwen XML, the parser addition has the spec 7a streaming tests (byte-at-a-time equal to whole-text), and the 36 A4 outputs parse.
3. **KV-only snapshots:** `snapshot_test` and `prefix_gpu_*` registered for K2; a K2 entry is refused by a Qwen3.8 engine and vice versa (the store keys carry the model).
4. **The comparison rows:** b70-serve (one card, spec 18 decision 2's context) with llama-benchy at the operator's flags, beside the recorded vLLM baseline (PP = 2, fp8 KV, 390k, 44.43 t/s decode, spec 18 §7); the configurations differ and every row says how.
5. **EOS handling:** K2 has two EOS ids [1, 250019]; both stop generation.

---

### Task 1: template, tool calls, reasoning

- [ ] Review Focus 1, 2, 5; a short greedy chat through the server equal to `b70-decode` on the same ids. **Commit** `server: K2-Horizon template, tool calls and reasoning (spec 18d)`.

### Task 2: prefix cache and K4

- [ ] Review Focus 3; A4 on K2 against the bf16 model's reference outputs from 18a (no bar; reported); passkey at the chosen context's 5 / 50 / 95 %. **Commit** `server: K2 prefix cache and the A4 / passkey gates (spec 18d, K3-K4)`.

### Task 3: the record

- [ ] Review Focus 4; `docs/BENCHMARKS.md` "K2-Horizon (spec 18)" completed; spec 18 amendment §10 with every gate and number; README (the scope names the MoE models); `docs/03-models.md`, `docs/13-loader.md`. **Commit** `spec 18: K2-Horizon served - the record`.

**Gate for the plan:** K3 and K4 green, the template and parser tests green, the comparison rows recorded, K0 green.

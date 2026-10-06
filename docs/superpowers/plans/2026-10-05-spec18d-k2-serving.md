# Spec 18d - K2-Horizon in the server

**Status (2026-10-05, branch `spec18d-k2-serving-host`, rebased on 18c):** Task 1's host half
written on the Mac (the box unavailable, 18c not yet validated on the card): Review Focus 1 (K2's
template byte-identical to `apply_chat_template` on 18 message lists, every `tool_call_format`; the
renderer changes as patches to minja with their own test), 2 (K2's `xml` / `xml_typed` / `json`
parser with spec 7a's streaming tests, the HF renders' round trip, a fuzz), 5 (EOS [1, 250019]
through the model-dispatched `server::ChatFormat`, a mock-engine server test); K2's tokenizer equal
to HF's. `b70-serve` still refuses K2. As built: spec 18 §12. Left: the K2 engine adapter (once 18c
passes on the card) and lifting the refusal, Task 1's greedy chat through the server, Task 2
(Review Focus 3, K4), Task 3.

**Status (2026-10-06, branch `serving-18d-15e`): the engine side written blind on the Mac** (as
built: spec 18 §14; box work: box-validation-queue row 25). `b70-serve` serves K2 through
`cli::k2::K2EngineAdapterT<K2Engine>` (MTP, `--spec lookup` and non-l0 prefill refused by name;
`--pp` refused for any model: one card); Review Focus 3 - KV-only snapshots (`state_bytes` 0, the
48 layers' K / V rows and int8 scales by `kv_snapshot_runs`, the block hook on `K2Engine::prefill`,
zero-byte block-end snapshots in the store, the store keyed by K2's root) - host-tested by
`k2_serve_test`; Task 1's greedy chat = `golden_server_test --chat` (box) with its wiring in
`k2_serve_test`; K4's tooling (score.py's K2 reader, `make_set.py --from ... --kwargs`,
`oracle_generate.py`'s k2_ref branch, `a4_ref.sh`, passkey stages). Left, all box: row 25's stages
(Review Focus 3 on the card, the chat, K4's runs and the 22 GB checkpoint's reference, Review Focus
4's llama-benchy rows), then Task 3 (the record; its BENCHMARKS skeleton is in place).

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

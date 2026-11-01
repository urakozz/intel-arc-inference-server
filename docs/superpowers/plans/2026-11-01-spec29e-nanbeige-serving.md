# Spec 29e - Nanbeige4.3-33B served: tokenizer, the ChatML template with `<think>`, Qwen-XML tool calls, sampling, 1- and 2-stage `b70-serve`, session continuation and the KV-only prefix cache

**Status (2026-11-01): PLANNED - after 29c and 29d.** Nothing built. No box queue row.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed; `WIP:` if needed); handoff file when you stop.

**Goal:** `b70-serve <Nanbeige>` (one card for short contexts, `--pp 2` for the long ones): the checkpoint's own template rendered byte for byte, `<think>` reasoning split, Qwen3-Coder XML tool calls parsed into OpenAI `tool_calls`, the vendor's sampling defaults, **session continuation** (the next turn reuses the previous turn's live KV) and the **prefix cache** (KV blocks in pinned host RAM, zero state), graded by D6-D8 and the serving speed row S4.

**Architecture:** spec 29 §3 (tokenizer / template / parsers rows), §9 items 1-2, §10 D6-D8. Precedents: plan 25e (Bolt served: tokenizer + template fixtures, the output stream, the request path, the adapter, snapshots), spec 18d (K2: zero-state snapshots, `state_bytes() == 0`), spec 7 (`PrefixSession`, block 2048), `src/server/chat_format.{h,cc}` (the `Kind` enum, `template_kwargs()`), `src/server/toolcall.{h,cc}` (the Qwen XML parser, unchanged).

**Spec:** `docs/superpowers/specs/2026-11-01-spec29-nanbeige-design.md`.

## Dependencies and branch points

- 29c and 29d merged; 29a's D6 results (tokenizer parity, template bytes). Branch `spec29e-nanbeige-serving`.
- **Box:** both cards; the GPU lock; `llama-benchy`; the A4 set and the agentic session from 29a; the judge's A4 references.

## Global Constraints

- Signed commits; no merge, no push; no `rm -rf`.
- **No new CLI flag and no environment variable.** `preserve_thinking` is **not** a server option: the template's own default (undefined = reasoning kept) applies; clients may send `chat_template_kwargs` (`preserve_thinking`, `enable_thinking`, `tool_call_format`) and they reach the template - that is the only reason for the new `ChatFormat` kind.
- Shared-source edits limited to additions: `ChatFormat::Kind::Dense` + `for_model_type("llama")`, the `b70_serve.cc` dispatch, CMake. The Qwen / K2 / Kolibri / Bolt behaviour stays byte-identical (their tests are the gate).
- The prefix cache is **host RAM only, never on the per-token path**; a restore is a per-request copy.

## Review Focus

1. **The template is the checkpoint's.** The Hub's `chat_template.jinja` rendered by minja equals transformers byte for byte for every conversation shape (29a's D6 fixtures run again through the *server's* path, including `reasoning_content`, grouped tool results, `tool_call_format` json / xml, `preserve_thinking`, `enable_thinking` false).
2. **Reasoning and tool calls.** The prompt ends `<think>\n` so the output starts inside reasoning (the existing Qwen rule: `thinking = prompt ends in "<think>\n"`); `</think>` (166104) ends it; `<tool_call>` (166105) / `</tool_call>` (166106) are single tokens that the stream parser must handle when they arrive in one step or split across the speculative accept boundary (29f) - the parser is fed text, never ids, so a test feeds every split of a golden output.
3. **A tool call inside reasoning is reasoning**, not a call (Bolt's / Qwen's rule; a test).
4. **Stop tokens.** `<|im_end|>` 166101 ends a turn; the padding ids 166107+ can never be sampled (the sampler mask is `VOCAB_USED`); `<|endoftext|>` 166102 recorded against the vendor's `generation_config` and chosen in a test.
5. **Session continuation is exact:** the next request whose rendered ids extend the previous turn's ids (prompt + output, **as the template re-renders them**) reuses the live KV up to the longest common prefix, and the result equals a cold prefill of the whole prompt within the near-miss rule; a request that diverges earlier (an edited history) truncates the KV at the divergence, never reads past it.
6. **The prefix cache's blocks are KV only** (`state_bytes() == 0`, K2's zero-state marker) and survive `--pp 2` (per-stage block shadows as spec 16d: both stages' KV for a block).

---

### Task 1: tokenizer and template fixtures through the server path (Mac)

**Files:** Create `src/server/chat_format.cc` additions, `tests/server/dense_template_test.cc`, `tests/tokenizer/dense/` (fixtures from 29a), modify `src/server/chat_format.{h,cc}` (`Kind::Dense`, `for_model_type`, `template_kwargs() == true`, `string_content` per the template's `visible_text`).

- [ ] **Step 1: failing tests** (the 40+ conversations of 29a through `chat::Template` under `Kind::Dense`; the kwargs forwarding; the Qwen / K2 / Kolibri / Bolt kinds unchanged). **Step 2: implement. Step 3: commit** `server: Dense chat format - Nanbeige's template with kwargs forwarded (spec 29e)`.

### Task 2: the output stream (Mac)

- [ ] **Step 1: failing tests** (`tests/server/dense_stream_test.cc`): golden outputs (29a's A4 references: reasoning, `</think>`, content, one and several `<tool_call>` blocks with string / number / bool / null / list / dict arguments typed from the schema) split at every byte boundary and every token boundary; a call inside reasoning; an unterminated call at EOS (finish reason, error shape as Qwen's). **Step 2: implement** (the existing Qwen parser and reasoning rule; any Nanbeige difference found in 29a's rendered examples is a parser-level fix with a test). **Step 3: commit.**

### Task 3: the request path and sampling (Mac)

- [ ] **Step 1: failing tests:** defaults when the client sets none: temperature 1.0, top-p 0.95 (the vendor's agentic setting; the reasoning-chat setting 0.6 is a client choice), `max_tokens` default per the engine's rule, the stop set, the padding mask, `logprobs` shape unchanged; `reasoning_content` in assistant messages round-trips (history -> template -> ids). **Step 2: implement. Step 3: commit.**

### Task 4: the engine adapters, session continuation and the snapshots

**Files:** Create `src/cli/dense_serve.h` (one-stage and two-stage `EngineIface` adapters; no MTP; the DFlash / lookup hooks stay absent until 29f), `src/runtime/dense/dense_snapshot.{h,cc}`, tests `tests/runtime/dense_snapshot_test.cc`, `tests/server/dense_server_test.cc`.

- [ ] **Step 1: failing tests** (host fakes: the HTTP server over a fake engine through `PrefixSession`): continuation (longest-common-prefix truncation, no read past `pos`), cache save / restore of KV blocks of 2048 with the zero-state marker, restore equals a cold prefill's KV bitwise (one stage and two stages, bf16 and int8), eviction under the host-RAM budget (`prefix_cache_size.h`: min(32 GiB, MemTotal / 2, MemAvailable - 8 GiB)), refusal of a snapshot across stage counts **not needed** (the layout is the one-stage host layout byte for byte, as spec 16b made it). **Step 2: implement. Step 3: commit.**
- [ ] **Step 4: `b70-serve` dispatch** (`model_type: llama` -> dense; `--max-len auto`, `--pp 2`, `--kv-cache`, `--prefix-cache-gb` as existing; `--batch > 1`, `--mtp`, `--spec` (until 29f), `--prefill-backend l0-int8` refused by name with the reason). **Commit.**

### Task 5: on the cards (box)

- [ ] **Step 1: golden over HTTP** (`golden_server_test` with `B70_SERVE_ARGS`): the served golden equals the `b70-decode` golden. **Step 2: D7:** the A4 set (36 scenarios) through `b70-serve` on one card (bf16 KV, short) and `--pp 2` (int8 KV): `score.py` against the judge's references and the twin's; zero parser errors; call-level correctness not below the twin's. **Step 3: llama-benchy** (`--pp 1024 --tg 128 --depth 0 4096 32768 --exact-tg --latency-mode generation`, one card and `--pp 2`): the S4 rows. **Step 4: commit** the records.

### Task 6: D8 - the long agentic session (box)

- [ ] **Step 1:** replay 29a's >= 150k-token session turn by turn through `b70-serve --pp 2 --kv-cache int8` with the client's rendered histories (a replay script `tools/probe/replay_dense.sh`, the repo's `replay_ab.sh` pattern): per turn the number of prefilled ids (must equal the new ids only after the first), the time, the first-token latency, the decode rate, every tool call parsed. **Step 2:** the same session with the prefix cache forced to restore (server restarted between turns, cache warm in host RAM): restore time against the derived ~2 s per 26 GB, the turn's first token. **Step 3:** with `preserve_thinking: false` sent by the client once, the log shows the cold re-prefill it causes (recorded; the server does not guard against it). **Step 4: commit** `docs/probe-nanbeige-session-2026-11-<dd>.md`.

### Task 7: docs

- [ ] **Step 1:** `docs/19-running-models.md` (Nanbeige: served, flags, the measured rows), README flags table untouched (no new flags), spec 29 "29e as built". **Step 2: commit.**

**Gate for the plan:** Mac - template bytes, stream splits, request / sampling, snapshot / continuation host tests, every other family's server tests unchanged, `mac_check.sh` green. Box - served golden equals decode golden; D7 clean; D8: every tool call parses, no turn after the first prefills more than its new ids (plus a tolerance of one block), restore bitwise; llama-benchy rows recorded; the server survives the 330k session's last turns.

**Box stages (prose):** (1) the served golden and the A4 set on one card, then two cards; (2) llama-benchy; (3) the long replay (an hour or more: the first turn's cold prefill alone is the 14-26 minutes of spec 29 §9 when it starts at 150k+; start the session shorter and grow it); (4) the cache-restore variant. GPU lock throughout; one job at a time.

# Spec 27e - Flint serving: the template, the GLM-style tool-call parser, reasoning, snapshots and the prefix cache, `b70-serve flint`, A4 on the card

**Status (2026-11-01): plan written, nothing built.** Needs 27c, 27d, 27f (the long-context KV form decides the snapshot sizes).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `b70-serve <flint>` serving the OpenAI-compatible API with correct tool calls: Flint's template rendered exactly, the `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` parser (streaming = whole-text), reasoning split on `<think>`, stops 150001 / 150003 / 150005 / 150018, snapshots and the prefix cache (session continuation: long sessions never re-prefill), A4 and an agentic transcript on the card.

**Architecture:** spec 27 sections 1 (template), 8, 9 (F4, F5). Template: plan 25e (`2026-10-20-spec25e-bolt-serving.md`), Kolibri's `src/cli/kolibri_serve.h`, `src/runtime/kolibri/kolibri_engine.h` (save / load state and KV), `src/server/{chat_format,toolcall_k2,toolcall_kolibri,prefix_cache,deps}.*`, `tests/server/toolcall_k2_test.cc` (the 3000-case fuzz), `tools/tokenizer/dump_k2.py`, `tools/toolcall/*`.

## Global Constraints
- Branch `spec27e-flint-serving`; signed commits; no merge, no push; GPU jobs under `~/b70-gpu.lock`.
- F0 as in 27c: the Qwen, K2, Kolibri and Bolt chat formats and parsers behave bitwise as before (their tests are unchanged and green); the Flint parser is **selected by `ChatFormat` kind, never by sniffing `<tool_call>`** (Qwen and Kolibri share the opener).
- Lookup, MTP and DFlash are refused by name (`verify_k() == 0`) until 27g.
- One reference for tool-call semantics: the pinned template's own rendering (27a's `flint_template_*.txt` fixtures).

---

### Task 1: the tokenizer and the template fixtures (Mac)
**Files:** create `tests/tokenizer/flint_tokenizer_test.cc` (+ the 27a fixtures), modify `src/server/chat_format.{h,cc}` (add `Kind::Flint`, `for_model_type("flint")`, `name()`), `src/tokenizer/chat_template.cc` only if minja lacks a construct (a SHA-256-keyed patch for Flint's template, the Ornith precedent).
- [ ] **Step 1: failing tests:** token ids of the role / tool / think tokens (150000-150018 as read in spec 27 section 1) encode as single ids; the template rendered by minja equals the pinned-transformers rendering byte for byte on the fixtures (system + tools, reasoning + two calls, tool result, dict vs string arguments, `<think>` stripped from user text, role tokens escaped as `&lt;|user|&gt;`, empty content); BOS is never added; the generation prompt is `<|assistant|>` with no `<think>`. Constructs minja must handle: `replace`, `tojson(ensure_ascii=False)`, `namespace`, `.items()`, `is defined`, macros. **Step 2: implement; PASS; commit.**

### Task 2: the output stream and the parser (Mac)
**Files:** create `src/server/toolcall_flint.{h,cc}` (or a tag-table parameter of `toolcall_k2.cc` if the existing tests stay bitwise green - the plan picks the second only if `toolcall_k2_test` needs no edit), `tests/server/toolcall_flint_test.cc`; modify `chat_format.cc` `make_output_parser` (one new branch); `tools/toolcall/score.py` (a Flint format; existing formats untouched, test first).
- [ ] **Step 1: failing tests:** whole-text and split-stream parity (a 3000-case fuzz as K2's, splitting at every byte); several calls in one turn; no wrapper; schema-string arguments verbatim, other values trimmed JSON with verbatim fallback (K2's `convert`), the round trip against the template's own rendering (render a history with calls, parse the rendering back: equal); an unschematised string that is valid JSON (`"123"`): the behaviour is pinned and recorded (spec 27 section 8); partial tags at chunk boundaries; `</think>` splitting reasoning from content; a stop on 150005 right after `</tool_call>` ends the turn with `finish_reason tool_calls`; a stop on 150003 / 150001 / 150018 without a call is a plain `stop`. **Step 2: implement; PASS; commit.**

### Task 3: the request path, sampling, stops (Mac)
- [ ] **Step 1:** the stop ids read from `generation_config.json` (the quant repo carries it with the four EOS ids, read in the spec 27 review; the server refuses to start without stop ids, by name); sampling defaults from the same file when present (T 1.0, top-p 0.95, top-k 20 are in the original's file only - D7 decides whether the quant folder gets them; absent, the request's values apply - a test for each case); `reasoning_content` accepted in request history and rendered by the template; tool `arguments` parsed from the OpenAI JSON string to an object before rendering. Failing tests with the mock engine; **commit.**

### Task 4: the engine adapter, snapshots, `b70-serve` (host half on the Mac; card gates)
**Files:** create `src/cli/flint_serve.h`, `src/runtime/flint/flint_snapshot.*`, tests `tests/runtime/flint_snapshot_test.cc` (host), `flint_snapshot_gpu_test.cc` (card); modify `src/cli/b70_serve.cc` (dispatch beside Kolibri; refusals by name).
- [ ] **Step 1: the snapshot contract** (spec 27 section 8): state = the 35 rings' last 512 rows (36.7 MB), blocks = the full layers' KV per 2048 positions (int8 29.8 MB / bf16 58.7 MB), `kv_form` in the store root, `kFlintStoreKey`; chunks end at block ends. Host tests: layout arithmetic, save / load byte round trip on the mock. **Step 2: the card test:** restore at block ends equals the cold run bitwise (KV, rings) and the next 64 greedy tokens equal (F4). **Step 3: `b70-serve flint`** wiring: `deps.h` contract (`state_bytes`, `kv_bytes`, `kv_form`, `save/load_state`, `save/load_kv`, `set_block_hook`, `block()` = 2048), `verify_k() == 0`. **Commit each.**

### Task 5: A4, the agentic transcript, llama-benchy (box)
- [ ] **Step 1:** A4 on the card (27a's set rendered with Flint's template; the BF16 CPU reference from 27a / 27q): pass count and per-scenario differences, recorded, no bar (F5; the watch note's A4 comparison with LFM2.5 is moot, LFM2.5 was dropped 2026-10-27). **Step 2:** the agentic transcript: a recorded multi-turn coding session (tool calls, results, reasoning replay) replayed through the server at 100k+ context with session continuation: time to first token on the last turn (prefix cache hit: no re-prefill beyond the new tokens), tool calls parsed, no stray `<arg_key>` text in content. **Step 3:** llama-benchy decode / prefill numbers at depth through the server. **Step 4:** `--pp 2` serving smoke (Kolibri's pp serve path).

### Task 6: docs; D7
- [ ] The quant repo already carries `generation_config.json` (stop ids, no sampling defaults), `chat_template.jinja` (byte-identical to the original's) and `tokenizer.json` (the original's plus an explicit no-BOS post-processor): assert this on the box copy. If the operator chose D7 (a), give the exact `hf download RecursiveSI/flint generation_config.json --local-dir ...` command (the operator downloads, never this plan). Docs, "27e as built", a queue row only if asked. **Commit.**

## Gates
- F0 (other chat formats and parsers bitwise unchanged), F4 (restore bitwise; streaming = whole-text), F5 (A4 recorded, agentic transcript with correct tool calls and prefix-cache hits), the template fixtures byte-equal to the pinned rendering.

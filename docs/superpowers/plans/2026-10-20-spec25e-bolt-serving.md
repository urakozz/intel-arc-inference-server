# Spec 25e - Nemotron 3.5 Bolt served: tokenizer, template, Qwen3-Coder XML tool calls, the ultra_v3 reasoning rules, sampling defaults, hybrid prefix-cache snapshots

**Status (2026-10-20): planned; nothing built.** Box queue row **62**.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `b70-serve <bolt>` on two cards: the checkpoint's `chat_template.jinja` rendered byte-identical to transformers
5.19.0, the tokenizer id-identical, `ChatFormat::Kind::Bolt` (the template kwargs, OpenAI's `reasoning_effort` mapped -
Decision 13), the output stream with the `ultra_v3` reasoning rules and Qwen3-Coder XML tool calls through the existing
`server/toolcall.h` parser, NVIDIA's sampling defaults and EOS {2, 11} (Decision 12), prefix-cache snapshots at aligned block
ends carrying the SSM state, the conv columns and the KV blocks; A4 on the card against 25a's references; the agentic
transcript's prefix-cache hit rate; llama-benchy beside vLLM.

**Architecture:** spec 25 §6.3, §7, §8, §9 B4 / B5, §12 decisions 12, 13, 14. Spec 23e's arrangement (fixtures from a
`dump_*.py`, the template byte-identity corpus, the parser fuzz, the adapter, the snapshot runs) and 24e's (the hybrid
snapshot with the recurrent rings written back at their slots). The Qwen XML parser is reused as is; Bolt adds only reasoning
rules and kwargs.

**Tech Stack:** C++17 (server, minja, Rust tokenizers via the existing bridge), Python 3 + transformers 5.19.0 (the fixtures).

**Spec:** `docs/superpowers/specs/2026-10-20-spec25-nemotron-bolt-design.md` (§6.3, §7, §9, §12). Precedents: plans 23e and
24e; `server/{chat_format,toolcall,openai,server,prefix_cache}.*`, `src/cli/muse_serve.h` (the adapter, the startup line, the
refusals), `chat::Template` (minja), `tools/tokenizer/dump_muse.py`, `tools/box_validate/serve_{run,client}.{sh,py}`,
`tools/probe/{vllm_benchy,serve_benchy}.sh` (row 41's matrix). The reasoning-parser plugin both repos ship
(`ultra_v3_reasoning_parser.py`, sha256 `14e03de5...`: DeepSeek-R1's parser plus three rules) and NVIDIA's README (the vLLM
flags: `--tool-call-parser qwen3_coder`, `--reasoning-parser ultra_v3`, `--mamba-cache-mode align`).

## Dependencies and branch points

- **Plans 25c and 25d merged** (decode, prefill, the block hook). 25a's A4 references and agentic transcript.
- **The vendoring rule (Decision 20, ruled 2026-10-20: vendoring allowed; generated fixtures remain the default):** template / tokenizer fixtures are **generated outputs** (renders, ids, digests), not
  copies of the model's files; the template itself is read from the snapshot at run time. If the operator rules that the
  files may be vendored, `tests/tokenizer/bolt/` takes them as 23e did.
- **Not here:** `--spec dflash` (spec 26c), batching.

## Global Constraints

- Branch `spec25e-bolt-serving` from main; no kernel added (cmdlines +0 / -0 / ~0); every existing server / template /
  tokenizer test unchanged (B0).
- Template rendering never edits the model's template text; a minja patch, if needed, lands in `third_party/minja` with a
  `minja_ext_test` case and the `third_party/VERSIONS` sha (23e's precedent).
- No `rm -rf`. Signed commits; no merge, no push.

## Review Focus

1. **The system block is always there.** With no system message the template still renders `<|im_start|>system\n<|im_end|>\n`
   (its `system_message` is set to "" and `is defined` holds); a render without it would shift every prompt by four tokens
   and miss every prefix-cache entry.
2. **Reasoning follows the prompt's end, per request.** Prompt ends `<think>\n` (thinking on): the output opens inside
   reasoning until `</think>`; prompt ends `<think></think>` (thinking off): everything is content, a stray `</think>`
   included; `force_nonempty_content` turns reasoning that never closed into content. Any split of the output streams to the
   same deltas.
3. **Tool calls are the Qwen XML parser's,** schema-first typing; a call inside reasoning is reasoning; a call cut by
   `max_tokens` or a stop string is dropped and stripped (spec 23 decision 16's rule) with `finish_reason` `length`.
4. **Snapshots only at aligned block ends** restore bitwise the cold run: KV blocks, the 40 SSM states, the conv ring's three
   raw columns before the position (written back at their slots), on two cards and across a split change.
5. **`truncate_history_thinking`'s miss is measured, not hidden:** the agentic transcript's hit rate per request, and the
   first token where each miss diverges.

---

### Task 1: the tokenizer and the template fixtures (Mac)

**Files:**
- Create: `tools/tokenizer/dump_bolt.py` (`tokenizer` | `templates` | `parse`), `tests/tokenizer/bolt_tokenizer.json`,
  `tests/tokenizer/bolt_templates.json`, `tests/tokenizer/bolt_tokenizer_test.cc`, `tests/tokenizer/template_bolt_test.cc`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1:** `dump_bolt.py` (transformers 5.19.0 from 21a's site, small files only, requests key-sorted): the tokenizer
  fixture (131072 ids' round trips digested, the named tokens 0-18 and 990 both ways, texts with every tag, `corpus.txt` in
  digests); the template cases (no system / system; tools with nested parameters, enums, extra keys (`render_extra_keys`),
  `required`; tool calls with string / number / bool / null / list / dict arguments; consecutive tool results grouped into one
  user turn; reasoning before and after the last user turn with `truncate_history_thinking` true and false; `enable_thinking`
  false; `medium_effort`; list-of-parts content; the generation prompt) - each as the render's bytes and its ids.
- [ ] **Step 2: the failing tests,** then pass them: `template_bolt_test` (byte for byte on every case through
  `chat::Template`, Review Focus 1), `bolt_tokenizer_test` (id for id; no BOS added anywhere). **Commit** `git commit -S -m "tests: Bolt's template byte-identical and tokenizer id-identical to transformers 5.19.0 (spec 25e)"`.

### Task 2: the output stream (Mac)

**Files:**
- Create: `tests/server/bolt_parse_cases.json` (from `dump_bolt.py parse`: vLLM-style expected splits written by hand from
  `ultra_v3`'s rules and the qwen3_coder format, each case commented with the rule it tests), `tests/server/bolt_stream_test.cc`
- Modify: `src/server/chat_format.{h,cc}` (`Kind::Bolt`: `template_kwargs()` true, `string_content()` false, the reasoning
  mode from the prompt's end), the output stream's reasoning rules (`force_nonempty_content`; thinking-off makes a stray
  `</think>` content)

- [ ] **Step 1: the failing tests:** Review Focus 2 and 3 on fixed outputs and a 2000-output fuzz (whole == byte stream ==
  random 1-7 byte pieces); the A4 reference outputs of 25a (the model's own calls) parse with 0 failures where 25a's scorer
  found none.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "server: ChatFormat Bolt - ultra_v3's reasoning rules over the Qwen XML calls, streaming split-invariant (spec 25e)"`.

### Task 3: the request path and sampling (Mac)

**Files:**
- Modify: `src/server/openai.cc` / `chat_format.cc` (the kwargs: `enable_thinking` (default true), `medium_effort`,
  `truncate_history_thinking`, `force_nonempty_content`; `reasoning_effort` mapped per Decision 13, an explicit kwarg wins;
  unknown kwargs passed through as 23e does), `src/cli/bolt_serve.h` (generation config: NVIDIA's T 1.0, top-p 0.95, EOS
  [2, 11] as defaults - read from a `--generation-config FILE` or the snapshot's file when it carries sampling keys, else the
  published values with the startup line saying so), `tests/server/bolt_request_test.cc`

- [ ] **Step 1: the failing tests:** each kwarg reaches the render (byte-checked against Task 1's fixtures); the mapping
  table; image / video / audio parts refused (23e's list); EOS on either id; SergiioB's generation config (EOS 11, no sampling)
  plus the published defaults gives Decision 12's set; a seeded sampled request reproducible.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "server: Bolt's template kwargs, reasoning_effort mapping, NVIDIA's sampling defaults and EOS {2, 11} (spec 25e)"`.

### Task 4: the engine adapter, the snapshots, `b70-serve` (host half on the Mac; card gates)

**Files:**
- Create: `src/cli/bolt_serve.h` (`BoltEngineAdapterT`, store key "NEMOBOLT", the startup line: the template's sha, the
  sampling defaults, the split, the snapshot size), `tests/runtime/bolt_snapshot_test.cc` (host, a mock engine with the real
  layouts), `tests/runtime/bolt_snapshot_gpu_test.cc` (card)
- Modify: `src/runtime/bolt/bolt_engine.{h,cc}` (`state_bytes` / `save_state` / `load_state` / `save_kv` / `load_kv`, the
  block hook at aligned block ends only), `src/runtime/bolt/bolt_sizes.{h,cc}` (`state_runs`, `kv_runs`: one host layout by
  layer, independent of the split), `src/cli/b70_serve.cc` (`serve_bolt`), CMake files

- [ ] **Step 1: the failing tests:** host - the state is 167,772,160 B (fp32; 83,886,080 B with the bf16 arm) + 2,457,600 B
  of conv columns, KV 8,192 B a position; restores at 2048 / 4096 / 6144 through `PrefixSession` equal the cold run on the
  mock; a two-card snapshot loads on another split and the reverse; over HTTP (mock) the greedy chat == the decode run, the
  repeat restored (`cached_tokens` reported). Card - Review Focus 4 on the synthetic and SergiioB's checkpoint.
- [ ] **Step 2: implement** `b70-serve <bolt>`: two cards by default, `--pp 2` / `--pipeline-split` / `--pipeline-handoff`,
  `--max-len auto` (up to 262144, Decision 14), the int8 head default, the prefix cache with the hybrid snapshot; refused before
  the device by name: `--mtp`, `--spec dflash|lookup` (spec 26c), `--kv-cache int8`, `--prefill-backend l0-int8|sycl-tla`,
  `--max-len` above 262144, `--device` with `--pp 2`. PASS. **Commit** `git commit -S -m "cli: b70-serve <bolt> on two cards - hybrid snapshots at aligned block ends (SSM state, conv columns, KV) (spec 25e)"`.

### Task 5: A4 on the card, the agentic transcript, llama-benchy (box)

- [ ] **Step 1 (box):** A4 through `b70-serve` (greedy, 2048 new ids) scored against 25a's g128 and BF16 references
  (agreement, parse failures) - B5's engine row; the agentic transcript replayed request by request with the prefix cache on:
  per request the restored position, the hit rate, Review Focus 5's divergence points; llama-benchy at depth 886 / 8192 / 32000
  / 65536 beside the operator's vLLM XPU run of the same checkpoint (row 41's matrix, SergiioB's flags) and SergiioB's
  published table. Record in BENCHMARKS ("Nemotron 3.5 Bolt (spec 25)").

### Task 6: docs, the box queue row

**Files:**
- Modify: spec 25 ("25e as built"), this plan's status, `docs/BENCHMARKS.md`, `docs/19-running-models.md` (Bolt's flags),
  `README.md` (CLI table), `box-validation-queue.md` (row 62), `tools/box_validate/stages.sh` (`row 62`)

- [ ] **Step 1:** the row:

```text
row 62 "spec 25e - Nemotron 3.5 Bolt served: the template / tokenizer identities, ultra_v3 reasoning over Qwen XML calls, NVIDIA's sampling defaults, hybrid snapshots at aligned block ends, A4 and llama-benchy (plan 25e)"
stage r62.host      62 default cpu  -                       -           "template_bolt_test, bolt_tokenizer_test, bolt_stream_test, bolt_request_test, bolt_snapshot_test"
stage r62.snapshot  62 default gpu  bolt_g128               -           "bolt_snapshot_gpu_test: restores bitwise the cold run, two cards, split change"
stage r62.chat      62 default gpu  bolt_g128,oracle_bolt   -           "golden_server_test --chat on b70-serve <bolt> == b70-decode"
stage r62.a4        62 optin   gpu  bolt_g128,bolt_a4_int4  r62.chat    "A4 through b70-serve vs the g128 reference, and vs the BF16 one when the BF16 window has made it (B5's engine row)"
stage r62.agentic   62 optin   gpu  bolt_g128               r62.chat    "the agentic transcript with the prefix cache: hit rate, divergence points"
stage r62.benchy    62 optin   gpu  bolt_g128               r62.chat    "llama-benchy at 886 / 8192 / 32000 / 65536 beside vLLM XPU (row 41's matrix)"
```
- [ ] **Step 2: commit** `git commit -S -m "docs: spec 25 (25e as built), box queue row 62 - Bolt served"`.

**Gate for the plan:** Mac - the template byte for byte, the tokenizer id for id, every parser / request / snapshot host test,
every existing server and template test unchanged. Box - snapshots restore bitwise, the server equals `b70-decode`, A4 and the
transcript rows recorded, llama-benchy beside vLLM recorded.

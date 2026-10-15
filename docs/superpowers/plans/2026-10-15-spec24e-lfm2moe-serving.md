# Spec 24e - LFM2.5-24B-A2B served: tokenizer, template, the pythonic tool-call parser, `<think>`, sampling defaults and the repetition penalty, prefix-cache snapshots with the conv rings

**Status (2026-10-15): planned; nothing built.** Box queue row **53** (rows 42-47 are spec 23's). Built blind on the
Mac after 24d; the host half (tokenizer, template, parser, sampler, snapshot layout) is fully testable on the Mac.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `b70-serve <LFM2.5-24B-A2B>` for long agentic coding sessions: the tokenizer held to transformers 5.19.0's id for id; `chat_template.jinja` rendered byte for byte as Jinja / `apply_chat_template` does; the output read as reasoning (the prompt ends in `<think>`) then content and pythonic tool calls through a safe literal parser with spec 24 §4.6's edge cases; the sampling defaults of `generation_config.json` with HF's repetition penalty on the host sampler (decisions 1, 2); prefix-cache snapshots carrying the 30 conv layers' `Bx` columns beside the KV; A4 on the card against 24a's bf16 reference; the agentic transcript replayed for the cache's hit rate.

**Architecture:** spec 24 §4.6, §8 24e. Kolibri's serving pattern (spec 20e): `cli::lfm2moe::Lfm2MoeEngineAdapterT<Lfm2MoeEngine>` (`src/cli/lfm2moe_serve.h`, `server::EngineIface`) over 24c / 24d's engine (one card or two - the engine holds the placement); `server::ChatFormat::Kind::Lfm2` (`src/server/chat_format.{h,cc}`: the template's kwargs - `preserve_thinking` -, content lists passed through, the parser below); `server::Lfm2OutputStream` + `server::parse_pythonic_calls` (`src/server/toolcall_lfm2.{h,cc}`, pure host code); `server::Sampling::repetition_penalty` and `server::apply_repetition_penalty` (`src/server/deps.h`, `src/server/spec_accept.{h,cc}` beside `filter_probs`); the snapshot hooks of `EngineIface` (`src/server/deps.h:62-77`).

**Tech Stack:** C++17, nlohmann::json, minja (`third_party/minja`, b70-patched), the Rust `b70_tok` (tokenizers 0.22.2), cpp-httplib; Python 3 with transformers 5.19.0 (24a's site) for the fixtures.

**Spec:** `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` (§1 tokens / template / sampling, §2.8, §4.6 incl. the parser's edge-case list, §7 F4 / F6's A4, §8 24e, §10 decisions 1, 2, 7, 8). Precedents: plan 20e (`2026-10-06-spec20e-kolibri-serving.md`), plan 18d, `src/server/toolcall_kolibri.{h,cc}` + `tests/server/toolcall_kolibri_test.cc`, `src/server/chat_format.{h,cc}`, `src/cli/kolibri_serve.h`, `tests/server/kolibri_server_test.cc`, `tests/runtime/kolibri_snapshot{,_gpu}_test.cc`, `tools/tokenizer/dump_kolibri.py` + `tests/tokenizer/k2_template_*`, `tests/server/sampling_test.cc`, `src/server/openai.cc:79-143` (argument strings parsed to objects; `generation_sampling`). Ideas (Apache-2.0, read only, no code copied): OpenVINO Model Server `src/llm/io_processing/lfm2/lfm2_tool_parser.cpp`, `src/test/llm/output_parsers/lfm2{,5}_output_parser_test.cpp` at `5d8bdfb1`, commits `503ff866`, `5fe145e5`; OpenVINO GenAI `d54d3768`.

## Dependencies and branch points

- **Plans 24c and 24d merged** (decode, prefill, the block hook from the prefill walk). **24a's** template fixtures (Task 4 Step 3) and A4 reference (`oracle-out-lfm2moe-a4`). **24q's** RTN / AutoRound exports on the box.
- **Decision 1 (repetition penalty):** built as proposed - host-side, sampled requests only; greedy (temperature 0) stays the device argmax and ignores it (the gates are greedy); `B70_LFM_PENALTY_GREEDY=1` routes a greedy request with a penalty ≠ 1 through the host (logits read back) for the operator's A/B, off by default.
- **Decision 2 (defaults):** `generation_config.json`'s - sampled, T 0.1, top-k 50, top-p 1.0 (absent → HF's default), penalty 1.05 - as `b70-serve`'s defaults for this model; a request's fields override.
- **Decision 7 (parser leniency):** built as proposed (spec 24 §4.6's list); every lenient rule counted per request in `--log-requests` (`parser.rules`) so the operator sees what fired.
- **Decision 8 (`enable_thinking: false`):** accepted and ignored, logged once per request (`enable_thinking ignored: the template always reasons`); the response's reasoning field still carries the reasoning.
- **Not here:** `--spec lookup` (needs M-row verify lists for this family: a follow-up), `--mtp` (no head), `--kv-cache int8` (decision 11), `--draft-vocab`.

## Global Constraints

- Branch `spec24e-lfm2moe-serving` from main (after 24d); box tree automatic; `tools/box.env` copied if missing, never committed, never printed.
- **F0 for the other models:** Qwen3.8 / Agnes / Ornith / K2 / Kolibri / `qwen4_exp` serving byte-identical - their template fixtures, tool-call tests and server tests unchanged and green; `generation_sampling`'s change (absent fields → HF's `GenerationConfig` defaults) leaves Kolibri's resolved sampling identical (a test asserts it).
- **No `eval`, no Python, no regex backtracking without a bound** in the parser: a hand-written recursive-descent literal parser with a depth limit (64), a size limit (1 MiB a block) and the quote-recovery backtracking bounded by the block length.
- The tokenizer file is read from the snapshot (or `B70_LFM_TOKENIZER_JSON`); no LFM file is committed beyond the fixtures this plan generates (rendered texts and ids of this repo's own conversations; decision 9 covers the config).
- Every number measured, or marked derived / estimated / proposed. Box: GPU commands under `flock ~/b70-gpu.lock`, detached, polled.
- Mac checks: `tools/mac_check.sh --base main`. No `rm -rf`. Signed commits; no merge, no push.

## Review Focus

1. **A rendered turn parses back to its message, and a parsed turn re-renders to the generated bytes.** For every fixture call (each literal type, quotes, escapes per context, nested values, several calls, content before / after): `parse(render(message)) == message` and `render(parse(text)) == text` for text the template could have produced. The escapes are the template's inverse per context (spec 24 §4.6): a top-level string decodes `\\`, its own quote's escape, `\n`, `\r`; a string inside a list / dict decodes JSON's set; every other backslash is kept (`"C:\Users\test\file.txt"` → `C:\Users\test\file.txt`).
2. **Model output the template would never write still yields the call the model meant** (spec 24 §4.6's list, decision 7): `'it's a test'` and `'That's Johns' car.'` (quote recovery), code with unescaped `"__main__"` inside a `"..."` argument, delimiters inside strings, `"path"="x"` and `'read_file'(...)` (quoted names - a key never contains a quote), `,` without a space and newlines between calls, a trailing comma, `fn()`, `fn(name="")`, a bare call without brackets, a lone positional dict; and the invalid ones (a positional scalar, `k=` without a value, `inf`, an unterminated string) make the block content, verbatim with its tags.
3. **Generation that stops inside a block.** `max_tokens` after `fn(a=1)` but before `]` / `<|tool_call_end|>`: the call is emitted once (whole text and streamed alike - the duplicate-emission regression Model Server's `503ff866` test pins); `max_tokens` inside `fn(a="unterminat`: no call, and the block's text is not content; EOS inside a block likewise; a stray `<|tool_call_end|>` is dropped; EOS 124900 never appears in content.
4. **Reasoning is keyed on the prompt's `<think>` and kept verbatim.** The output opens inside reasoning; `</think>` closes it; no newline or space is stripped from either side (the template writes `"<think>" + thinking + "</think>"`, `chat_template.jinja:90-92`); a `<|tool_call_start|>` inside reasoning is reasoning text; generation ending inside reasoning gives reasoning only (`finish_reason` length).
5. **The snapshot is the KV plus the right two `Bx` columns.** `save_state` at position p stores, per conv layer, the ring slots of positions p-1 and p-2 (in that order, 245,760 B for 30 layers - 0 bytes are meaningful below position 2, written as zeros); `load_state(p)` writes them back to slots `(p-1) % 16`, `(p-2) % 16`; a restore at a block end (2048, 4096), at a prompt end (2049, 5000, 300) and at positions 0 / 1 / 2 continues bitwise as the cold run (KV, conv rings, logits).

---

### Task 1: the tokenizer and the template fixtures (Mac)

**Files:**
- Create: `tools/tokenizer/dump_lfm2moe.py`, `tests/tokenizer/lfm2moe_template_cases.json`, `tests/tokenizer/lfm2moe_template_*.txt` (generated), `tests/tokenizer/lfm2moe_tokenizer_test.cc`, `tests/tokenizer/lfm2moe_template_test.cc`
- Modify: `tests/CMakeLists.txt` (a block `# ==== Spec 24e: LFM2.5-24B-A2B serving (label lfm2moe) ==== (begin)`), `src/tokenizer/chat_template.cc` only if a minja gap is found (each fix a `b70 patch (spec 24e)` in minja with a `minja_ext_test` case against Jinja2, recorded in `third_party/VERSIONS`)

- [ ] **Step 1: the fixtures** - `dump_lfm2moe.py <LFM2.5 snapshot>` (24a's site): the cases of 24a Task 4 Step 3 plus: system / no system; tools / no tools (schemas key-sorted, the server's known limitation); a call with a string holding `'`, `\`, a newline, a tab, unicode; int, float, bool, null, list, nested dict arguments; two calls in one turn; content then calls; a tool result; reasoning on the turn after the last user (kept) and before it (dropped), `preserve_thinking` true; content as a list of text parts (concatenated without a separator - this template's `parse_content`, `chat_template.jinja:14-34`); `CONTINUE_FINAL_MESSAGE_TAG`; the generation prompt. Each case: the messages / tools JSON and `apply_chat_template`'s text, plus `tokenizer(text)` ids and `tokenizer.decode` of the generation's ids; a corpus (`tests/tokenizer/corpus.txt` + code + CJK + the agentic transcript) as ids.
- [ ] **Step 2: the failing tests:** `lfm2moe_template_test` - `chat::Template(<snapshot>).render(...)` equals every case byte for byte; `lfm2moe_tokenizer_test` - `tok::Tokenizer` encodes the corpus and every case id for id, decodes back, `vocab_size()` 125017, `token_to_id("<|tool_call_start|>")` 124905; both SKIP 77 when `B70_LFM_TOKENIZER_JSON` / the snapshot is absent. FAIL (no LFM cases).
- [ ] **Step 3: make them pass** (a minja patch only if a case needs one). **Commit** `git commit -S -m "tokenizer: LFM2.5-24B-A2B - tokenizer ids and the chat template byte for byte against transformers 5.19.0 (spec 24e)"`.

### Task 2: the pythonic tool-call parser and the output stream (Mac)

**Files:**
- Create: `src/server/toolcall_lfm2.h`, `src/server/toolcall_lfm2.cc`, `tests/server/toolcall_lfm2_test.cc`
- Modify: `src/server/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
namespace server {
struct PythonicRules { uint32_t quote_recovered = 0, quoted_name = 0, bare_call = 0, json_literal = 0,
                       positional_dict = 0, dropped_unfinished = 0, stray_end = 0; };
// One block's text between <|tool_call_start|> and <|tool_call_end|> (exclusive). Returns the calls, or nullopt when the
// block holds no valid call (the caller keeps the block as content). `complete` false: generation ended inside the block -
// the calls whose ')' closed are returned, the unfinished tail dropped (Review Focus 3).
std::optional<std::vector<ToolCall>> parse_pythonic_calls(std::string_view block, bool complete, uint64_t& seed,
                                                          PythonicRules& rules);
class Lfm2OutputStream : public OutputParser {   // the prompt ends in "<think>": the output opens inside reasoning
 public:
  explicit Lfm2OutputStream(nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;
  std::vector<Delta> finish() override;
  const PythonicRules& rules() const;
};
ParsedOutput parse_output_lfm2(const std::string& text, const nlohmann::json& tools);
}
```

- [ ] **Step 1: the failing tests** `toolcall_lfm2_test.cc` - one test per line of Review Focus 1-4 and per item of spec 24 §4.6's list, each also streamed at every split point of its text (the concatenated deltas equal the whole-text parse, call ids aside - spec 7a's rule): the round trip over Task 1's fixtures (Review Focus 1); the model-output idioms (Review Focus 2, with Model Server's test inputs as the idea list - `lfm2_output_parser_test.cpp:194-322`, `:357-427`, `:693-889` - restated in our words and our expected JSON); the stop-inside-a-block cases (Review Focus 3, incl. "`)` seen, then EOS" emitting exactly one call); reasoning (Review Focus 4); limits (depth 65, a 1 MiB + 1 block, 10^5 quotes - the backtracking stays linear-bounded and finishes in < 50 ms); `PythonicRules` counted. Run `ctest --preset mac-host -R '^toolcall_lfm2_test$'`: FAIL.
- [ ] **Step 2: implement** (a recursive-descent parser over the block's bytes: `call_list := '['? call (sep call)* sep? ']'?`; `call := name '(' (arg (sep arg)* sep?)? ')'`; `arg := name '=' value | dict`; `value := string | number | keyword | list | tuple | dict`; strings with the per-context escape sets; the quote-recovery rule with one bounded backtrack per candidate quote; JSON out through nlohmann with keys in the call's order). PASS.
- [ ] **Step 3: commit** `git commit -S -m "server: the LFM2.5 pythonic tool-call parser - a safe literal parser with the template's escapes, quote recovery, quoted names, stop-inside-a-block handling; reasoning keyed on the prompt's <think> (spec 24e)"`.

### Task 3: the chat format, sampling defaults and the repetition penalty (Mac)

**Files:**
- Modify: `src/server/chat_format.{h,cc}` (`Kind::Lfm2`: `for_model_type("lfm2_moe")`, `template_kwargs()` true with `preserve_thinking` (bool) validated, `string_content()` false), `src/server/deps.h` (`Sampling::repetition_penalty = 1.0f`), `src/server/openai.{h,cc}` (the request's `repetition_penalty`, (0, 2], a 400 naming it otherwise; `generation_sampling`: `repetition_penalty`, and absent fields take HF's `GenerationConfig` defaults - temperature 1.0, top-k 50, top-p 1.0, penalty 1.0; `eos_token_id` read as an int or a list, `stop_strings` as a string or a list where present), `src/server/spec_accept.{h,cc}` (`apply_repetition_penalty`), `src/server/server.cc` (the session's id set handed to the sampler), `tests/server/sampling_test.cc`, `tests/server/openai_test.cc`, `tests/server/toolcall_lfm2_test.cc` (Task 2's: the chat-format cases, as `toolcall_kolibri_test.cc` holds Kolibri's)

**Interfaces:**
- Produces: `void apply_repetition_penalty(float* logits, uint32_t vocab_used, const std::vector<uint32_t>& distinct_ids, float penalty)` - HF's rule (`logits_process.py:418-419`): `l < 0 ? l · p : l / p` for every distinct id of the prompt and the output, before `filter_probs` (temperature, top-k, top-p); `class SeenIds { void add(uint32_t); const std::vector<uint32_t>& distinct() const; }` (a 128000-bit set plus the list, per request).
- [ ] **Step 1: the failing tests:** `sampling_test` - the penalty against a hand-computed row (positive and negative logits, an id seen twice counted once, ids ≥ `vocab_used` ignored), penalty 1.0 bitwise no-op, the order (penalty then temperature then top-k then top-p) against an HF-ordered reference computed in the test; `openai_test` - LFM's `generation_config.json` resolves to {sampled, T 0.1, top-k 50, top-p 1.0, penalty 1.05}, Kolibri's resolves exactly as before, `eos_token_id` int and list, `stop_strings` string and list, a request `repetition_penalty` 0 / 3 refused naming it; `toolcall_lfm2_test` - `lfm2_moe` → `Kind::Lfm2`, `preserve_thinking` passed, `enable_thinking: false` accepted (decision 8). FAIL.
- [ ] **Step 2: implement; PASS; every existing server test unchanged and green. Commit** `git commit -S -m "server: LFM2's chat format, generation_config defaults with HF's absent-field rule, the repetition penalty on the host sampler (spec 24e, decisions 1, 2, 8)"`.

### Task 4: the engine adapter, the snapshots, `b70-serve` (Mac host half; card gates)

**Files:**
- Create: `src/cli/lfm2moe_serve.h`, `tests/runtime/lfm2moe_snapshot_test.cc` (host), `tests/runtime/lfm2moe_snapshot_gpu_test.cc` (card), `tests/server/lfm2moe_server_test.cc`
- Modify: `src/runtime/lfm2moe/lfm2moe_engine.{h,cc}` (`state_bytes`, `save_state`, `load_state`, `kv_bytes`, `save_kv`, `load_kv`, the block hook from the prefill walk), `src/cli/b70_serve.cc` (dispatch on `lfm2_moe`; refusals: `--mtp`, `--spec`, `--kv-cache int8`, `--layers`, `--draft-vocab`, each naming why), `src/cli/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `template <class Engine> struct Lfm2MoeEngineAdapterT : server::EngineIface` (Kolibri's adapter's shape, `src/cli/kolibri_serve.h:45-...`): `prefill` (24d's chunked walk), `step` (greedy: the device argmax; sampled: the logits row read back into pinned host memory, `apply_repetition_penalty` over the request's `SeenIds`, `filter_probs` / `draw`, the id written to Control - `serve_adapters.h:335-341`'s path), `block()` 2048, `state_bytes()` 245,760, `save_state` / `load_state` (Review Focus 5), `kv_bytes(n)` 20,480 x n, `save_kv` / `load_kv` (the 10 layers' rows), `kv_form()` 0, `set_block_hook`, `ingest`; `vocab_used` 125017. `b70-serve <lfm2moe>`: `--max-len auto` (131072), `--lm-head int8` (default), `--prefix-cache-gb auto`, `--prefix-split-last`, `--pp 1|2`, the sampling defaults (Task 3), `--log-requests` with `parser.rules`.
- [ ] **Step 1: the failing host test** `lfm2moe_snapshot_test`: the state layout (two slots per conv layer, the order, zeros below position 2), the slot arithmetic of `load_state` at p = 0, 1, 2, 15, 16, 17, 2048; the KV block layout; a mock engine round trip.
- [ ] **Step 2: implement the adapter and the snapshot calls;** host PASS; `b70-serve` dispatch and refusals (`cli_reject_lfm2moe_serve_*`).
- [ ] **Step 3: the card gates** (label `checkpoint;lfm2moe`, SKIP 77 without data): `lfm2moe_snapshot_gpu_test <ckpt>` (Review Focus 5: restores at 0 / 1 / 2 / 300 / 2048 / 2049 / 4096 / 5000 continue bitwise the cold run, on one card and two, a snapshot moving between them); `lfm2moe_server_test <ckpt>` over HTTP (greedy chat = `b70-decode --prefill`'s ids; a second request restored at 2048 reports `cached_tokens`; a seeded sampled request reproducible; streaming deltas = the whole response; a tool-call response with `tool_calls` and the reasoning field; `max_tokens` inside a call: Review Focus 3 end to end; no id ≥ 125017 ever emitted).
- [ ] **Step 4: Mac gate; commit** `git commit -S -m "serve: LFM2.5-24B-A2B behind b70-serve - the adapter, host sampling with the penalty, prefix-cache snapshots with the conv rings (spec 24e)"`.

### Task 5: A4 on the card and the agentic transcript (box)

**Files:**
- Modify: `tools/toolcall/engine_generate.sh` (an `lfm2moe` mode: the 36 scenarios of `tests/golden/toolcall-lfm2moe` through `b70-serve`, greedy, 2048 new ids), `tools/box_validate/serve_client.py` only if a field is missing, `tools/oracle/README.md` (the A4 rows for this model)

- [ ] **Step 1 (box):** A4 on the RTN export, then ours: `engine_generate.sh lfm2moe` → `score.py --format pythonic` against 24a's bf16 reference: agreement, parse failures (the engine's and the reference's), `PythonicRules` totals - decision 5's bar (PROPOSED ≥ 33 / 36, 0 parse failures) and decision 7's evidence.
- [ ] **Step 2 (box):** the agentic transcript (24a's `lfm2moe_agentic.json`, then the recorded opencode session when `OPENCODE_LOG` is set) replayed turn by turn through `b70-serve` with the prefix cache: per turn the prompt length, `cached_tokens`, TTFT and decode t/s; the hit rate, and the turns where the template's content-then-calls order diverged from the generated text (spec 24 §4.6's known miss) counted.
- [ ] **Step 3: commit** `git commit -S -m "spec 24e: LFM2.5-24B-A2B A4 on the card and the agentic transcript's cache hit rate"`.

### Task 6: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` ("24e as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 53), `tools/box_validate/stages.sh` (`row 53`), `docs/19-running-models.md` (the LFM2 section: serving commands, the sampling defaults, the penalty, `preserve_thinking`, "always thinks"), `README.md` (the `b70-serve` flag table's model list, `B70_LFM_TOKENIZER_JSON`, `B70_LFM_PENALTY_GREEDY`)

- [ ] **Step 1: the queue row 53** (`test_box_validate.py` passes; `--dry-run --only r53` prints it):

```text
row 53 "spec 24e - LFM2.5-24B-A2B served: template, tokenizer, the pythonic parser, reasoning, sampling defaults and the repetition penalty, prefix-cache snapshots with the conv rings (plan 24e)"
stage r53.host     53 default cpu -                              -        "host: lfm2moe_template_test, lfm2moe_tokenizer_test, toolcall_lfm2_test, sampling_test, openai_test, lfm2moe_snapshot_test"
stage r53.reject   53 default cpu -                              -        "cli_reject_lfm2moe_serve_* (mtp, spec, kv8, layers, draft-vocab)"
stage r53.snap     53 default gpu lfm2moe_rtn                    -        "lfm2moe_snapshot_gpu_test: restores bitwise the cold run (one card; two when present)"
stage r53.server   53 default gpu lfm2moe_rtn                    r53.snap "lfm2moe_server_test over HTTP: greedy = b70-decode, cached_tokens, seeded sampling, streaming, tool calls, max_tokens inside a call"
stage r53.a4       53 optin   gpu lfm2moe_rtn,lfm2moe_a4_ref     r53.server "A4 through b70-serve vs 24a's bf16 reference (score.py --format pythonic); ours when present"
stage r53.agentic  53 optin   gpu lfm2moe_rtn                    r53.server "the agentic transcript (and OPENCODE_LOG) replayed: cached_tokens, TTFT, t/s per turn, the hit rate"
rownote 53 "Decision 5's A4 bar (>= 33 / 36, 0 parse failures) and decision 7's rule counts are PROPOSED evidence; the operator rules after r53.a4."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: spec 24 (24e as built), box queue row 53 - LFM2.5-24B-A2B served"`.

**Gate for the plan:** Mac - the template byte for byte, the tokenizer id for id, every parser / sampler / snapshot host test, every existing server and template test unchanged. Box - snapshots restore bitwise, the server tests green over HTTP, A4 and the transcript rows recorded against decision 5's proposed bar.

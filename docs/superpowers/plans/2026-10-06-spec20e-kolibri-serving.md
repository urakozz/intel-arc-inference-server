# Spec 20e - Kolibri-1 served: template, reasoning, hermes JSON tool calls, prefix cache, KL4, the record

**Status (2026-10-06): Tasks 1-4 built blind on branch `spec20e-kolibri-serving`; Task 5's queue row and stages written (row 28), its speed rows and the record are box-only (r28.benchy, after 20b)** - spec 20 §13 has what was built and where the build departs from this plan (the undefined ids are 127998 / 127999, not 127923 / 127924; the template is `tokenizer_config.json`'s string - no `.jinja`, `chat::Template` reads it; list content joined before the render instead of a template case; the host test is `kolibri_snapshot_test`, the card one `kolibri_snapshot_gpu_test`; a request-end restore is bitwise the uncached session, not one cold prefill; `kolibri_prefill_engine.cc` needed no change; Kolibri's first sampled id is drawn from the prefill's row).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve <Kolibri-1>` serves OpenAI chat on two cards for long agentic sessions: Kolibri's 128k tokenizer and ChatML template (byte-identical to `apply_chat_template`), reasoning that the model opens itself, hermes-style JSON tool calls in `<tool_call>`, the model's sampling defaults, spec 7's prefix cache with the sliding rings in the snapshot, `--max-len auto` over both cards; KL4 (a German A4-style tool-call set, passkey at 262144); the speed rows and the record.

**Architecture:** spec 20 §4 (tokenizer and chat), §6 20e; spec 18 §12 is the precedent for a non-Qwen chat format (`server::ChatFormat`, a per-model `OutputParser`, vendored template files, `chat_template_kwargs`). The server's request path is unchanged; this plan adds `ChatFormat::Kind::Kolibri`, `server::KolibriOutputStream` (the Qwen parser's tags, K2's JSON call body - factored out, not copied), model-scoped sampling defaults from `generation_config.json`, `KolibriEngine`'s snapshot calls (sliding rings as the "state", the full layers' KV as the blocks, both cards packed in one host layout) and `cli::KolibriEngineAdapter` behind `server::EngineIface`.

**Tech Stack:** C++17, nlohmann::json, minja (the engine's Jinja renderer), the Rust `tokenizers` crate, Level Zero; Python 3 in `agnes-ref-img` (template / tokenizer dumps, A4 tooling); uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-05-spec20-kolibri-1-design.md` (§1 tokens and template, §4, §5 KL4 and speed, §6 20e, §7 decision 3). Facts: `docs/probe-kolibri-2026-10-05.md` ("Chat template, tokens, tool calls"). Plans 20c and 20d and their as-built sections. Precedents: `src/server/chat_format.{h,cc}`, `src/server/toolcall{,_k2}.{h,cc}`, `tests/tokenizer/k2/`, `tools/tokenizer/dump_k2.py`, `tests/server/k2_server_test.cc`, `tests/server/toolcall_k2_test.cc`, `src/cli/serve_adapters.h` (`EngineAdapter`), `src/server/prefix_cache.h`, `tools/toolcall/`, `tools/probe/k2_passkey.sh`, spec 18 §12.

## Dependencies and branch points

- **Plans 20c and 20d merged** (Tasks 3-6). Tasks 1-2 are host-only and need neither: they can run first, on the Mac.
- **Spec 16b is merged** (`0792415..dd2fc87`); 20c Task 6 and 20d Task 3 built Kolibri's two cards on its pieces (`l0` device view, `SyncEvent`, `runtime::pp_balance`, `PipelineLink` / `StageLink`, `pp_handoff.cl`, the Control mirror and `set_token`). This plan adds nothing to that mechanism. **Snapshots follow 16b's rule** (spec 16 §8): the host layout under `--pp 2` is the `--pp 1` layout byte for byte (device 0's layers then device 1's IS layer order), so an entry moves between one card and two unchanged. Spec 16d (the server under PP for `runtime::Engine`) is **not** needed. **The switches are 16b's as renamed on 2026-10-06** - `--pp N` / `--pipeline-parallel-size N`, `--pipeline-split auto|N`, `--pipeline-handoff copy|peer` - parsed in `b70-serve` for `kolibri1` with `cli/pipeline_args.h`'s parsers, Kolibri's default `--pp 2` and its `--pp 1` fit refusal exactly as `b70-decode` (20c Task 5); for every other model `b70-serve`'s pipeline behaviour is whatever 16d makes it, untouched here.
- **Spec 20b** for everything on real weights: the int4 checkpoint (`urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ`) and, for KL4's reference outputs, the bf16 source wherever 20b runs (decision 1). Real-weight gates SKIP 77 until then; the synthetic checkpoints of 20c carry the engine-side gates.
- **Decision 2** reaches this plan only through the engine (both arms served; the memory plan per arm). **Decision 3:** the served context is capped at the trained 262144 (`--max-len` above it refused naming decision 3); KL4's passkey runs at that ceiling. **Decision 4:** shared experts bf16, as 20c.
- **20a's correction to spec 20 §4:** the tool calls are hermes JSON (`Hermes2ProToolParser` in the plugin). The repo has no hermes parser: the Qwen parser reads Qwen XML only, K2's reads a JSON body inside K2's tags. Task 2 reuses both halves rather than adding a third format reader.

## Global Constraints

- Branch `spec20e-kolibri-serving` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked.
- **K0 for the server path:** `template_test`, `template_agnes_test`, `template_ornith_test`, `template_k2_test`, `toolcall_test`, `toolcall_k2_test`, `k2_server_test`, `ornith_server_test`, `protocol_test`, `golden_server_test`, `prefix_server_test`, `mtp_server_test`, `lookup_server_test` unchanged; Qwen's sampling defaults unchanged; no kernel binary changed (`tools/kernel_cmdlines` +0 / -0 / ~0).
- The vendored template files are the checkpoint's own, unmodified (Apache-2.0): `tools/oracle/third_party/kolibri1/{kolibri1_chat_template.jinja, tokenizer_config.json, generation_config.json}`; `tokenizer.json` (not vendored) comes from the CMake cache path `B70_KOLIBRI_TOKENIZER_JSON` or the environment variable of that name; absent: disabled on the Mac, SKIP 77 elsewhere.
- Renderer gaps are fixed in `third_party/minja` as "b70 patch (spec 20e)" with a `minja_ext_test` case and a `third_party/VERSIONS` line (spec 18 §12's rule) - never a Kolibri special case in the server.
- One weight format; no third-party quant path. Every number measured, or marked derived / estimated / proposed.
- Box: `flock ~/b70-gpu.lock` once for both cards, detached, polled; `uptime` with timings; the operator's llama-benchy flags.
- Mac checks `tools/mac_check.sh --base main` (`--kernels` not needed: no kernel changes). No `rm -rf`. Signed commits on the branch; no merge, no push.

## Review Focus

1. **Reasoning opened by the model.** With thinking on, Kolibri's prompt ends in `<|im_start|>assistant\n` and the output begins `<think>\n...</think>\n\n`; with thinking off the prompt ends in `<think>\n\n</think>\n\n` and the output is the answer. The Qwen parser keys reasoning on the prompt's ending, so it would put Kolibri's reasoning into `content`. Task 2 tests both, and an output whose first bytes are whitespace before `<think>`.
2. **A tool call that is not valid JSON, or is cut off.** `<tool_call>\n{"name": "f", "arguments": {"a": 1}\n</tool_call>` (a missing brace), a call cut by `max_tokens`, `"arguments"` as a JSON string holding an object, two calls separated by `\n`: valid calls are emitted at their `</tool_call>`, a broken one becomes content verbatim with its tags, a cut one becomes content; streaming any split equals the whole text (spec 7a's rule).
3. **Sampling never emits an id the tokenizer does not have.** The head has 128000 rows; the tokenizer defines 127998 ids (127923 and 127924 unassigned). The adapter masks undefined ids before host sampling; `finish_reason` is `stop` on either EOS id (127906 `<|im_end|>`, 127901 `<|endoftext|>`).
4. **A prefix-cache restore across a sliding window.** A conversation restored at a block end (2048, 4096) or at a request end (any position, e.g. 2049 or 5000) continues bitwise as a cold run would: the snapshot carries the last 512 positions of each of the 40 sliding layers, and positions before 0 (a restore at p < 512) are zeros the window never reads.
5. **German text through the whole path.** Umlauts, ß, long compounds and `„...“` quotes round-trip: tokenizer ids equal HF's, the detokenised stream equals the decoded ids, tool-call arguments with German strings keep their bytes (no `\u` re-escaping changes in content); covered in Tasks 1-2 and KL4's German set.

---

### Task 1: tokenizer and template (host)

**Files:**
- Create: `tests/tokenizer/kolibri/{chat_template.jinja,tokenizer_config.json,generation_config.json}` (copies of the vendored files), `tools/tokenizer/dump_kolibri.py`, `tests/tokenizer/kolibri_template_cases.json`, `tests/tokenizer/kolibri_template_<case>.txt` (one per case), `tests/tokenizer/kolibri_tokenizer.json` (the tokenizer fixture), `tests/tokenizer/kolibri_tokenizer_test.cc`
- Modify: `tests/CMakeLists.txt` (a block `# ==== Spec 20e: Kolibri-1 served (label kolibri) ==== (begin)` at the end), `third_party/minja` + `third_party/VERSIONS` + `tests/tokenizer/minja_ext_cases.json` only if a render differs
- Test: `template_kolibri_test`, `kolibri_tokenizer_test`, `minja_ext_test`

**Interfaces:**
- Consumes: `template_test` (the executable that `template_k2_test` runs with a cases file and a prefix), `chat::Template::render(..., kwargs)`, `tok::Tokenizer`.
- Produces: the fixtures Task 2 parses back (`kolibri_template_tool_call_history.txt`, `..._parallel_calls.txt`).

- [ ] **Step 1: the dumps.** `tools/tokenizer/dump_kolibri.py <snapshot-or-repo> <out dir>` (in `agnes-ref-img`, transformers 5.x, `tokenizer.json` fetched as a small file): renders `apply_chat_template(..., tokenize=False)` for these message lists, each object key-sorted (the server holds requests in `nlohmann::json`): `plain` (default effort high), `effort_{none,minimal,low,medium,xhigh,max}`, `thinking_off` (`enable_thinking=False`), `system_tools` (system text + two tools), `tools_no_system`, `tool_call_history` (one call, one result, the answer), `parallel_calls` (two calls, two tool messages -> one user turn), `reasoning_history` (assistant turns with `reasoning_content` before and after the last user query), `list_content`, `german` (umlauts, ß, „Anführungszeichen“, a 40-letter compound); writes the cases file (messages, tools, kwargs, BOS none, EOS `<|im_end|>`) and one `.txt` per case. Tokenizer fixture: `len(tok)` 127998, the specials 127900-127906 and 127913-127922, the plain added tags 127907-127912 both ways, 24 texts (German prose, code, digits, emoji, the tags inline, empty) - ids without special tokens (no BOS is ever added), decode with and without specials, three chat renders' ids (`tokenize=True`) equal to the server's encode of the rendered text.
- [ ] **Step 2: register, run, expect FAIL** where the renderer differs:

```cmake
if(B70_TOKENIZER_ENABLED)
  add_test(NAME template_kolibri_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/kolibri kolibri_template_cases.json kolibri)
  set(B70_KOLIBRI_TOKENIZER_JSON "$ENV{B70_KOLIBRI_TOKENIZER_JSON}"
    CACHE FILEPATH "Kolibri-1's tokenizer.json for kolibri_tokenizer_test (-D or the environment variable)")
  add_executable(kolibri_tokenizer_test tokenizer/kolibri_tokenizer_test.cc)
  target_include_directories(kolibri_tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kolibri_tokenizer_test PRIVATE b70_tokenizer)
  add_test(NAME kolibri_tokenizer_test COMMAND kolibri_tokenizer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/kolibri ${B70_KOLIBRI_TOKENIZER_JSON})
  set_tests_properties(template_kolibri_test kolibri_tokenizer_test PROPERTIES LABELS kolibri SKIP_RETURN_CODE 77)
endif()
```
(`template_k2_test`'s and `k2_tokenizer_test`'s registrations, argument for argument. The path is `Aleph-Alpha/Kolibri-1-BF16`'s `tokenizer.json` at commit `8c8b3489` (a small file; 20b's export copies it unchanged); once 20b has published the int4 repo, the default may name its snapshot in the HF cache as K2's does - an edit with the repo's commit in hand.) `ctest --preset mac-host -R '^(template_kolibri|kolibri_tokenizer|minja_ext)_test$'`.
- [ ] **Step 3: fix renderer gaps** as minja patches (each with a `minja_ext_cases.json` case rendered by `tools/tokenizer/dump_minja_ext.py`), until every case is byte-identical; the Qwen / Agnes / Ornith / K2 template tests stay byte-identical.
- [ ] **Step 4: run, expect PASS; commit** `git commit -S -m "tokenizer: Kolibri-1's template renders as apply_chat_template, its 128k tokenizer equals HF's (spec 20e)"`.

### Task 2: reasoning and hermes JSON tool calls (host)

**Files:**
- Create: `src/server/toolcall_kolibri.{h,cc}`, `tests/server/toolcall_kolibri_test.cc`, `tests/server/kolibri_server_test.cc`
- Modify: `src/server/toolcall.{h,cc}` (`parse_json_call`, moved out of `toolcall_k2.cc`'s anonymous namespace), `src/server/toolcall_k2.cc` (calls it), `src/server/chat_format.{h,cc}` (`Kind::Kolibri`), `src/server/openai.{h,cc}` (`parse_request` with sampling defaults), `src/server/server.{h,cc}` (`Options::sampling_defaults`), `src/server/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `toolcall_kolibri_test`, `kolibri_server_test`; K0: `toolcall_k2_test`, `toolcall_test`, `k2_server_test`, `protocol_test`

**Interfaces:**
- Consumes: `server::OutputParser`, `Delta`, `ToolCall`, `schema_string_type`, `new_call_seed`, `make_call_id` (`src/server/toolcall.h`).
- Produces:

```cpp
namespace server {
// toolcall.h - K2's JSON call body rule, shared (spec 18d's behaviour, unchanged for K2):
// {"name": <string>, "arguments": <object | string holding an object | absent>}; schema-string arguments
// written as another JSON value become its JSON text. False when it does not parse.
bool parse_json_call(const std::string& body, const nlohmann::json& tools, ToolCall& call);
// toolcall_kolibri.h
class KolibriOutputStream : public OutputParser {
 public:
  explicit KolibriOutputStream(nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;
  std::vector<Delta> finish() override;
};
// chat_format.h
struct ChatFormat { enum class Kind { Qwen, K2Horizon, Kolibri }; ...
  static ChatFormat kolibri() { return ChatFormat{Kind::Kolibri}; } };   // for_model_type("kolibri1")
// template_kwargs() is true for Kolibri (reasoning_effort, enable_thinking reach the template);
// check_template_kwargs: reasoning_effort in {none, minimal, low, medium, high, xhigh, max}, enable_thinking a bool.
// openai.h
Request parse_request(const std::string& body, bool chat, const Sampling& defaults);   // the 2-argument form = Sampling{}
// server.h
struct Options { ... std::optional<Sampling> sampling_defaults; };   // b70-serve sets it for kolibri1 only
}
```

`KolibriOutputStream`'s reading (whole text and streamed alike): optional leading whitespace then `<think>` opens reasoning (verbatim, up to `</think>`, the newline after the open tag dropped, trailing whitespace before the close tag dropped - the Qwen rule); otherwise the output is body. In the body: text outside calls is content (verbatim; only-whitespace content dropped, as K2's); `<tool_call>` ... `</tool_call>`: the trimmed body through `parse_json_call`, emitted at its close tag; a body that does not parse is content with its tags; cut by the end of generation: content. Holds back only what may still be a tag.

- [ ] **Step 1: the failing tests.** `toolcall_kolibri_test`: every case of Review Focus 1 and 2; the round trip - the assistant turns of Task 1's `tool_call_history` and `parallel_calls` renders parse back to the messages' `tool_calls` (name, arguments) and `reasoning_content`; spec 7a's split rule (byte at a time, random 1-7 byte pieces) on every case; a fuzz of 2000 generated outputs (valid, broken, cut) whose streamed deltas equal the whole parse; German argument strings byte-identical (Review Focus 5). `kolibri_server_test` (`tests/server/mock.h`'s engine and tokenizer): `chat_template_kwargs` `reasoning_effort` reaches the template, a bad value is a 400 naming it; reasoning / content / calls in the body and in streamed frames alike, `finish_reason: tool_calls`; both EOS ids stop; a request without `temperature` / `top_p` / `top_k` samples with 1.0 / 0.97 / 128 (Options::sampling_defaults), while a Qwen-format server keeps 1.0 / 0.95 / 20.
- [ ] **Step 2: register, run, expect FAIL:**

```cmake
add_executable(toolcall_kolibri_test server/toolcall_kolibri_test.cc)
target_include_directories(toolcall_kolibri_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(toolcall_kolibri_test PRIVATE b70_server)
add_test(NAME toolcall_kolibri_test COMMAND toolcall_kolibri_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
add_executable(kolibri_server_test server/kolibri_server_test.cc)
target_include_directories(kolibri_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri_server_test PRIVATE b70_server)
add_test(NAME kolibri_server_test COMMAND kolibri_server_test)
set_tests_properties(toolcall_kolibri_test kolibri_server_test PROPERTIES LABELS kolibri)
```
`ctest --preset mac-host -R '^(toolcall_kolibri|kolibri_server|toolcall_k2|toolcall|k2_server|protocol)_test$'`
- [ ] **Step 3: implement** (move `parse_json` -> `parse_json_call` first and run `toolcall_k2_test` green before writing the new parser).
- [ ] **Step 4: run, expect PASS** (K0 tests included); `tools/mac_check.sh --base main` green; **commit** `git commit -S -m "server: Kolibri-1 reasoning and hermes JSON tool calls, its sampling defaults, dispatched on model_type (spec 20e)"`.

### Task 3: the engine behind the server, the prefix cache (card)

**Files:**
- Create: `src/cli/kolibri_serve.h` (`cli::KolibriEngineAdapter`), `tests/runtime/kolibri_snapshot_test.cc`
- Modify: `src/runtime/kolibri/kolibri_engine.{h,cc}` and `kolibri_prefill_engine.cc` (snapshot calls), `src/cli/b70_serve.cc` (dispatch; 20c's refusal lifted), `tests/CMakeLists.txt` (20c's `cli_reject_kolibri_serve` removed)
- Test: `kolibri_snapshot_test` (card, synthetic and real), `golden_server_test` against Kolibri by hand (Step 5)

**Interfaces:**
- Consumes: 20c / 20d's `KolibriEngine` (`prefill`, `ingest`, `generate`, `read_logits`, `set_block_hook`), `server::EngineIface`, `cli::sample` (`src/cli/serve_adapters.h`), `server::PrefixCache`.
- Produces:

```cpp
// runtime::kolibri::KolibriEngine additions
static constexpr uint32_t kBlock = 2048;                        // = kPfC: block hooks fire at chunk ends
size_t state_bytes() const;            // sliding rings' last 512 positions, every sliding layer, both devices:
                                       // 40 x 512 x (K + V) x 512 x 2 B = 41,943,040 B (derived)
size_t kv_bytes(uint32_t n) const;     // n x 20,480 B: the 10 full layers' K and V, both devices
void save_state(void* host);           // device 0's sliding layers then device 1's, each K then V, positions pos-512 .. pos-1 ascending
void load_state(const void* host, uint32_t pos);   // into ring slots p & 4095; positions < 0 written as zeros; pos set on every device
void save_kv(uint32_t begin, uint32_t end, void* host);    // full layers in layer order (both devices), K then V, rows [begin, end)
void load_kv(uint32_t begin, uint32_t end, const void* host);
// set_token(uint32_t id) is 20c Task 6's (PipelineEngine::set_token's rule: both Control blocks) - used, not added
// cli::KolibriEngineAdapter : server::EngineIface - reset, prefill, step (generate(1); non-greedy: read_logits() from the
// last device, ids the tokenizer does not define set to -inf, cli::sample, set_token), max_len, pos, block,
// state_bytes, kv_bytes, save/load_state, save/load_kv, set_block_hook, ingest. mtp_k() = 0, verify_k() = 0.
```

- [ ] **Step 1: the failing card test** `kolibri_snapshot_test <ckpt> <ids>` (synthetic 5-layer under `--pp 1` and `--pp 2 --pipeline-split 3` - the engine options; real after 20b under `--pp 2`): a 5000-id prompt prefilled cold, logits / KV / rings and 32 greedy ids recorded; then (a) reset, `load_kv(0, 4096)` + `load_state(at 4096)` from the hook's saves + `prefill(tail)`: bitwise the cold run; (b) reset, restore at a request end 2049 (`save_state` + `save_kv(2048, 2049)` taken there) and continue: bitwise; (c) restore at 300 (< 512: zero-filled ring rows never read): bitwise; (d) 16b's rule: a state and KV saved under `--pp 2` restore under `--pp 1` (synthetic) and continue bitwise, and the reverse; `state_bytes` / `kv_bytes` are equal under both (Review Focus 4). Register (`LABELS "checkpoint;kolibri"`, SKIP 77 without data); it is disabled on the Mac host build.
- [ ] **Step 2: implement** the snapshot calls (layer order across both devices); the adapter; `b70-serve` dispatch on `model_type` `kolibri1`: `--pp` (default 2) / `--pipeline-split` / `--pipeline-handoff` and `--max-len N|auto` as `b70-decode` (`require_two_devices`, the peer check before the load, `runtime::kolibri::pp_split_and_len`), `--lm-head int8` default (spec 9, as every served model), `ChatFormat::kolibri()`, `Options::sampling_defaults` from `generation_config.json` (temperature 1.0, top_p 0.97, top_k 128), EOS from `generation_config.json` ({127906, 127901}, already read for every model), the prefix cache on by the existing flags; refused by name: `--mtp`, `--spec lookup` (no verify lists for Kolibri), `--kv-cache int8`.
- [ ] **Step 3: run on the box** (queue row stages; for development `flock ~/b70-gpu.lock ctest --test-dir build -R '^kolibri_snapshot' --output-on-failure`).
- [ ] **Step 4: the server end to end on the synthetic checkpoint:** `golden_server_test` pointed at the synth by hand (as 15e did for Ornith: it does not SKIP without a checkpoint, so it is not registered for Kolibri) - a greedy chat through the server equals `b70-decode --prefill` on the same rendered ids.
- [ ] **Step 5: commit** `git commit -S -m "server: b70-serve runs Kolibri-1 on two cards - its engine adapter, prefix-cache snapshots with the sliding rings (spec 20e)"`.

### Task 4: KL4 - the German tool-call set and passkey at 262144

**Files:**
- Modify: `tools/toolcall/make_set.py` (`--lang de`), `tools/toolcall/oracle_generate.py` (`--model kolibri`), `tools/toolcall/score.py` (`--format hermes`), `tools/toolcall/test_score.py`
- Create: `tests/golden/toolcall-kolibri-de/` (manifest, `.json`, `.ids` - made by make_set.py from the int4 snapshot's tokenizer, committed), `tools/probe/kolibri_passkey.sh`
- Test: `tools/toolcall/test_score.py` (host), the KL4 runs (box / wherever 20b ran)

**Interfaces:**
- Consumes: `kolibri_ref.KolibriRef.generate` (20a; `--device cuda|xpu` where available), `tools/toolcall/engine_generate.sh` (any model through `b70-decode --prefill`), `tools/probe/passkey.py`, Task 2's parser rules.
- Produces: `<out>/<name>.{bf16,l0,l0-i8head}.{ids,txt}` and the score table.

- [ ] **Step 1: failing tests** in `test_score.py`: `parse_first_call` with `fmt="hermes"` on `<tool_call>\n{"name": "grep", "arguments": {"pattern": "Schlüssel"}}\n</tool_call>` -> `("call", "grep", {"pattern": "Schlüssel"}, 1)`; a broken JSON body -> `"incomplete"`; no call -> `"no call"`; the Qwen XML cases unchanged with the default format. Run `python3 tools/toolcall/test_score.py`: FAIL.
- [ ] **Step 2: implement** `score.py --format hermes` (JSON body read with `json.loads`, arguments compared as JSON values), `make_set.py --lang de` (the six scenario templates' system prompt and user turns in German - written into a `DE` table in make_set.py - over the same six repository files, rendered by the snapshot's own template with `enable_thinking=False`, ids 800-3000 as before), `oracle_generate.py --model kolibri <snapshot> <set> <out>` (greedy 192 new ids through `KolibriRef.generate`, resumable as today). `test_score.py` PASSES.
- [ ] **Step 3: the set** (Mac or box CPU, needs only the tokenizer): `tools/oracle/run_in_container.sh "python3 tools/toolcall/make_set.py --lang de \$SNAP tests/golden/toolcall-kolibri-de"` with `ORACLE_MODEL` the int4 snapshot; commit the set.
- [ ] **Step 4: KL4 runs.** The bf16 reference (the 156 GB source, wherever 20b runs, `--device` there): `oracle_generate.py --model kolibri`; the engine: `LM_HEAD=bf16` and `int8` `tools/toolcall/engine_generate.sh <int4 snap> tests/golden/toolcall-kolibri-de <out> l0` on two cards; `score.py --format hermes <out> bf16 l0 l0-i8head`. **Gate:** Task 2's parser reads every `<tool_call>` the bf16 reference wrote (0 parse failures - a hard bar); the engine-vs-bf16 first-call agreement is reported, no bar until measured (as K2 and Ornith).
- [ ] **Step 5: passkey.** `tools/probe/kolibri_passkey.sh` (`k2_passkey.sh`'s shape: Kolibri's tokenizer, no BOS, `MAX_LEN=262144` on two cards, placements 0.05 / 0.5 / 0.95, 8 greedy ids, PASS if the key is in the text): **gate 3 / 3** at 262144 with the int8 head. Decode speed at that depth is recorded too (the full layers read 5.4 GB of KV per token there, derived - more than the weights' ~2.4 GB).
- [ ] **Step 6: commit** `git commit -S -m "toolcall: Kolibri-1 KL4 - the German A4-style set, hermes scoring, passkey at 262144 (spec 20e)"`.

### Task 5: the queue row, the speed rows, the record

**Files:**
- Modify: `docs/superpowers/plans/box-validation-queue.md`, `tools/box_validate/stages.sh`, `tools/probe/serve_benchy.sh` (`ZE_AFFINITY_MASK` from the environment, default 0, so a two-card model can be served), `docs/BENCHMARKS.md` ("Kolibri-1 (spec 20)" completed), the spec's "20e as built" section, `README.md` (scope: the German / English MoE on two cards; a Kolibri row), `docs/03-models.md`, `docs/13-loader.md`, `docs/19-running-models.md`

- [ ] **Step 1: the queue row** (next free number): no kernel changed (G0's sha is the check); the server path K0 list; `rN.snapshot` (`kolibri_snapshot_test`), `rN.serve` (b70-serve on the synth: one greedy chat = `b70-decode`), `rN.a4` (opt-in, after 20b), `rN.passkey` (opt-in), `rN.benchy` (opt-in); `rownote`s for what needs 20b's checkpoint and the bf16 reference's machine. `python3 tools/box_validate/test_box_validate.py` passes.
- [ ] **Step 2: speed rows** (box, two cards, real checkpoint): `MODEL=<int4 snap> MAX_LEN=262144 ZE_AFFINITY_MASK=0,1 tools/probe/serve_benchy.sh --pp 4096 --tg 256 --depth 1 --no-cache --exact-tg --latency-mode generation` (the operator's flags; this `--pp 4096` is llama-benchy's prompt length, b70-serve runs Kolibri's default `--pp 2` - `SERVE_ARGS='--pipeline-handoff peer'` for the second arm); prefix-cache rows at 4k / 16k / 32k depth (cache on); decode at 4k / 32k / 128k depth; every row states the cards, the attention arm (decision 2), the head form and the KV form. A vLLM row (Aleph Alpha's plugin) only if it serves Kolibri-1 on these two B70s - otherwise the table says none was run.
- [ ] **Step 3: the record** - BENCHMARKS, the as-built section (every gate and number of 20c-20e), README, docs/03, docs/13, docs/19.
- [ ] **Step 4: commit** `git commit -S -m "spec 20: Kolibri-1 served - KL4, the speed rows, the record"`.

**Gate for the plan:** Mac - `template_kolibri_test`, `kolibri_tokenizer_test`, `toolcall_kolibri_test`, `kolibri_server_test` green, the K0 server tests unchanged. Box - `kolibri_snapshot_test` bitwise (synthetic, one and two cards; real after 20b); one greedy chat through b70-serve equal to b70-decode; KL4 (0 parse failures on the bf16 reference outputs, agreement reported; passkey 3 / 3 at 262144); speed rows recorded; the record committed.

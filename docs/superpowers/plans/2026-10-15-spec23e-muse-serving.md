# Spec 23e - Muse-Glimmer-30B served: tokenizer, template, the channel / ATEM parser, reasoning strength, sampling with the softcap, prefix-cache snapshots

**Status (2026-10-16): built on the Mac (branch `spec23e-muse-serving`): Tasks 1-4 and 7 done, Task 5 Step 1 done; Task 5 Step 2 and Task 6's runs (A4, passkey, llama-benchy) are the box's - queue row 46. What was built and where it departs from this plan: spec 23 §16 "23e as built" (the state is 81,748,992 B, not 81,747,968; both Vishva007 repos ship the older template; the A4 arm runs b70-decode on the set's ids; Step 2 is a stage).**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit after each step** (signed, `git commit -S`; a step that does not build or pass yet goes in as `WIP: ...`) - never leave a step's files uncommitted. When you stop for any reason, leave `HANDOFF-spec23e.md` (untracked, worktree root): the last step done, what is half-done, the next command.

**Goal:** `b70-serve <muse>`: the OpenAI chat API over `MuseEngine` (one card by default, `--pp 2` optional) with the original's chat template rendered byte-identical to transformers 5.19.0, the tokenizer equal to transformers', a channel-aware output parser (reasoning in `to=self`, content in `to=user`, ATEM tool calls in tool channels) equal to transformers' `parse_response` on complete turns, `reasoning_strength`, the sampling defaults with the host-side softcap, the prefix cache with the sliding rings in the snapshot; MG4 restores, MG5 (A4 on the engine, passkey at 131072).

**Architecture:** spec 23 §1.2, §5.4, §6. Kolibri's serving (spec 20 §13) is the template: `cli::muse::MuseEngineAdapterT` behind `server::Server`, `ChatFormat::Kind::Muse`, `server::MuseOutputStream`, snapshots as `state_runs` / `kv_runs` with one host layout for every placement. Two references: transformers 5.19.0 (`apply_chat_template`, the tokenizer, `parse_response` with the checkpoint's `response_template`) for every byte; vLLM's Muse parsers (`V/tool_parsers/muse_glimmer_tool_parser.py`, `V/reasoning/muse_glimmer_reasoning_parser.py`, Apache-2.0) for damaged turns and streaming hold-back.

**Tech Stack:** C++17 (server, minja, the Rust `tokenizers` crate via `src/tokenizer/rs`), Python 3 + transformers 5.19.0 (the fixtures), Level Zero (snapshots on the card).

**Spec:** `docs/superpowers/specs/2026-10-15-spec23-muse-glimmer-design.md` (§1.2, §5.4, §6, §4.2 MG4-MG5, §11 decisions 1-4, 6, 15, 16). Precedents: plan 20e (`2026-10-06-spec20e-kolibri-serving.md`: `template_kolibri_test`, `kolibri_tokenizer_test`, `toolcall_kolibri_test`, `kolibri_snapshot_test`, `KolibriEngineAdapterT`), plan 21e (`dump_qwen4exp.py`, the `\p{M}` split), spec 18d (minja patches, each tested against Jinja2), `src/server/{chat_format,toolcall,spec_accept,openai}.h`.

## Dependencies and branch points

- **23c and 23d merged** (`MuseEngine` decode and prefill, the block hook); 23a's A4 set and `score.py --format atem`.
- The `\p{M}` split class (spec 21e) is in the Rust tokenizer path already - this plan only proves Muse's file through it.

## Global Constraints

- Branch `spec23e-muse-serving` from main; signed commits; no merge, no push; box rules as 23c.
- **No kernel** added or changed (`kernel_cmdlines` +0 / -0 / ~0).
- The template is the ORIGINAL's `chat_template.jinja` (sha256 `cfc67e5f...`; decision 1): `b70-serve` reads `<snapshot>/chat_template.jinja` only when its sha256 is that one, else refuses naming `--chat-template FILE`; `--chat-template FILE` overrides (the operator's way to use the GPTQ repo's older file deliberately).
- Every fixture is made by transformers 5.19.0 (spec 21a's site), committed, and the scripts that make them are committed beside them; tests pass `current_date="2026-10-15"` so no render depends on the clock.
- Vendored small files under `tests/tokenizer/muse/`: `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja` (Apache-2.0); `tokenizer.json` (28 MB) is NOT vendored - tests read it from the HF cache (`B70_MUSE_TOKENIZER_JSON`), SKIP 77 without it.
- No `rm -rf`. No third-party engine named beyond the Apache-2.0 references spec 23 cites (transformers, vLLM, OpenVINO Model Server / openvino.genai - ideas only, no code copied).

## Review Focus

1. **One BOS.** The template writes `bos_token` and the tokenizer's post-processor would add another; the server tokenises the rendered prompt without special tokens. A test asserts exactly one 200000 at position 0 of every render's ids, equal to `apply_chat_template(tokenize=True)`'s.
2. **The first output message has no `<|start|>`** (the prompt ends with `<|start|>assistant`): the parser's first header is ` to=<recipient><|message|>`; a stream that splits ` to=self` across pieces must not leak ` to` or `self` into content or reasoning (hold-back of a growing ` to=...` tail, `V/tool_parsers/...:_OPEN_TAIL_TO_RE`).
3. **`<|eom|>` is not EOS** but `<|eot|>` and `<|end_of_text|>` are; a reasoning block then a tool call is one generation; the last tool call ends with `<|eot|>` (T:187-191) and stops it.
4. **The softcap is applied exactly once, on the host, only for sampling:** greedy ids equal the device argmax of the raw row; a sampled request's probabilities are `softmax(20 tanh(z x 0.196.../20) / T)` before top-k / top-p; the MTP / lookup acceptance paths (`spec_accept`) see the same capped p.
5. **A tool call's string arguments in a request** are parsed into an object before rendering (the template raises on strings, T:28-30); a string that is not a JSON object is a 400 naming the call - never a template exception surfacing as a 500.
6. **A generation cut mid-call** (`max_tokens`, a stop string) never errors and never leaks markup: the unterminated `<atem:...>` block is removed from content, no partial call is emitted, `finish_reason` is `length` (spec 23 decision 16; OVMS `503ff8662` recovers the call instead, vLLM drops it - ideas only, `src/llm/io_processing/onyx/onyx_tool_parser.cpp:211-260` at `5d8bdfb19`).
7. **Argument values typed by the tool's schema first** (decision 15): a string-typed `"007"` / `"true"` / `" x "` reaches the client unchanged; an integer-typed `"007"` is `7`; the call's name comes from `<atem:invoke name>`, not the `to=` recipient (the model may write the bare name there - `onyx_tool_parser.hpp:47-51` - or `ns.ns` - vLLM).

---

### Task 1: tokenizer and template

**Files:**
- Create: `tools/tokenizer/dump_muse.py`, `tests/tokenizer/muse/{tokenizer_config.json,generation_config.json,chat_template.jinja}`, `tests/tokenizer/muse_tokenizer.json` (fixture), `tests/tokenizer/muse_templates.json` (fixture), `tests/tokenizer/muse_tokenizer_test.cc`, `tests/tokenizer/template_muse_test.cc`
- Modify: `tests/tokenizer/CMakeLists.txt`; `third_party/minja/*` only if a render differs (each change marked `b70 patch (spec 23e)`, tested in `tests/tokenizer/minja_ext_test.cc` against Jinja2, `third_party/VERSIONS`' sha256 updated)

**Interfaces:**
- Produces: `dump_muse.py tokenizer|templates|parse <bf16-snapshot> <out.json>` (transformers 5.19.0): `tokenizer` - ids of 24 texts (prose, code, digits, emoji, CJK, combining marks, every special tag inline, `<atem:...>` markup, empty) and corpus.txt's 10240 lines (digests), `templates` - the renders and ids of the message lists below, `parse` - Task 2's cases through `tokenizer.parse_response(text, prefix=prompt, tools=...)`.
- The message lists (`muse_templates.json`, >= 20): no system message (the default system text with `current_date`), a system message with and without "Reasoning strength", each of the four "Reasoning effort" casings, `reasoning_strength` low / medium / high / xhigh, tools with dotted names (`fs.read_file`) and bare names (`read_file`), `tool_namespace_descriptions`, call history with one and with parallel calls (each ATEM value type: true, false, null, an int, a float, a string with a newline and quotes, a list, an object), tool results by `name` and by `tool_call_id`, `reasoning_content` before a call, a final `to=user` answer, `recipient` / `end_turn` overrides, two consecutive assistant messages (the `<|eom|>` end token), `add_generation_prompt` on and off.

- [x] **Step 1: the fixtures** (on the Mac, `agnes-ref-img` with the 5.19.0 site). Commit `tokenizer: Muse-Glimmer fixtures from transformers 5.19.0 (spec 23e)`.
- [x] **Step 2: failing tests:** `muse_tokenizer_test` (the Rust tokenizer on the HF cache's `tokenizer.json` equals the fixture text by text, the corpus digests, the renders' ids; Review Focus 1; the eight named added tokens' ids); `template_muse_test` (every render byte-identical; `raise_exception` on string arguments surfaces as an error the server maps to a 400; `strftime_now` provided when `current_date` is absent - checked against a fixed clock hook).
- [x] **Step 3: implement until PASS** (minja patches only if needed). Commit `tokenizer: Muse-Glimmer's template byte-identical to transformers 5.19.0, its tokenizer equal (spec 23e)`.

### Task 2: the output parser

**Files:**
- Create: `src/server/toolcall_muse.h`, `src/server/toolcall_muse.cc`, `tests/server/toolcall_muse_test.cc`, `tests/server/muse_parse_cases.json` (fixture from `dump_muse.py parse`)
- Modify: `src/server/CMakeLists.txt`, `src/server/toolcall.{h,cc}` (only to share `convert_value` / `schema_string_type` / `make_call_id` if not already shared)

**Interfaces:**
- Consumes: `server::OutputParser`, `Delta`, `ToolCall`, `new_call_seed`, `make_call_id`, `schema_string_type` (`server/toolcall.h`).
- Produces:

```cpp
namespace server {
// Spec 23e (spec 23 §6.5): Muse-Glimmer's channels. Segments the output into messages
// ([<|start|>assistant] [ to=<rcpt>]<|message|> body until <|eom|> / <|eot|> / the next header);
// to=self -> reasoning (blocks joined with "\n", decision 2); to=user or no recipient -> content;
// any other recipient -> a tool channel whose complete <atem:invoke> blocks are the calls.
class MuseOutputStream : public OutputParser {
 public:
  explicit MuseOutputStream(nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;
  std::vector<Delta> finish() override;
};
struct MuseParsed { std::string reasoning, content; std::vector<ToolCall> tool_calls; };
MuseParsed parse_muse_output(const std::string& text, const nlohmann::json& tools);
std::string muse_normalize_name(const std::string& emitted, const nlohmann::json& tools);  // ns.ns -> ns
}
```

- [x] **Step 1: the failing test** `toolcall_muse_test`: (a) every complete case of `muse_parse_cases.json` equals `parse_response` except the documented differences - reasoning (transformers keeps the LAST block; our join is compared by its last block and by the whole joined text - decision 2) and value typing (decision 15: the fixture marks the cases where `parse_response`'s JSON-first typing differs, and the test asserts our schema-first value there), the assistant turns of Task 1's renders parse back to their messages (with arguments cast by the tools' schemas as `parse_response(tools=...)` casts them); (b) damaged turns from vLLM's parsers' documented cases: a `to=self` body with no `<|eom|>` followed by a bare `to=<tool><|message|>` header (the call survives), an `<atem:invoke>` echoed inside reasoning or inside `to=user` (not a call), a truncated invoke (content only, no call), no framing at all (the whole text is content), a doubled name `read_file.read_file` with a registered `read_file` (collapsed), `weather.get` against registered `calendar.get` (passed through unchanged), `to=read_file` with `<atem:invoke name="read_file">` and `to=fs` with `fs.read_file` (the invoke's name wins); a generation cut inside `<atem:parameter>`, inside `<atem:invoke name="`, and after `</atem:invoke>` but before `</atem:function_calls>` (Review Focus 6: the complete invoke is a call, an incomplete one is neither a call nor content); (c) Review Focus 2, 3 and 7, and a piece ending in `<|` held back until resolved (the structural-prefix hold-back, OVMS `gptoss/content_parser.cpp:110` at `db2f05cff` - idea only); (d) a 2000-output fuzz (random channel sequences, values, markers inside values) where every parse of the whole equals the stream of random 1-7 byte pieces.
- [x] **Step 2: implement, PASS.** Commit `server: Muse-Glimmer's channel and ATEM parser - parse_response on complete turns, vLLM's selection on damaged ones (spec 23e)`.

### Task 3: the request path (`ChatFormat::Muse`), reasoning strength, sampling with the softcap

**Files:**
- Create: `tests/server/muse_server_test.cc`
- Modify: `src/cli/b70_serve.cc` (read `stop_strings` beside `eos_ids`: a string or a list of strings, refused naming the key otherwise), `src/server/chat_format.{h,cc}` (`Kind::Muse`, `for_model_type("muse_glimmer")`, `template_kwargs()`, `check_template_kwargs`, `make_output_parser`), `src/server/openai.{h,cc}` (`reasoning_effort` mapped to `reasoning_strength` for Muse; image / video parts refused; string `arguments` parsed into objects), `src/server/deps.h` (`Sampling` gains `float logit_mult = 1.f, softcap = 0.f`), `src/server/spec_accept.{h,cc}` (`filter_probs` applies the cap when `softcap > 0`), `src/server/server.{h,cc}` (the defaults from `generation_config.json` via `generation_sampling`)

**Interfaces:**
- `check_template_kwargs(Muse, kwargs)`: `reasoning_strength` one of low / medium / high / xhigh (else `std::invalid_argument` naming it -> 400), `current_date` / `knowledge_cutoff` strings, `tool_namespace_descriptions` an object of strings; anything else passes to the template unchanged (as Kolibri's).
- `Sampling{temperature 1.0, top_k 64, top_p 0.95, logit_mult 0.19611613513818404, softcap 20}` when a request names none (decision 4); a request's own fields win; `logit_mult` / `softcap` are model properties, never request fields.

- [x] **Step 1: failing tests** `muse_server_test` (a mock engine, Kolibri's `kolibri_server_test` arrangement): kwargs reach the template (a bad `reasoning_strength` is a 400 naming it; `reasoning_effort: "low"` renders `Reasoning strength: low.`); an image part is a 400 naming the vision encoder; string arguments in history parsed (Review Focus 5); both EOS ids stop and `<|eom|>` does not (Review Focus 3); streamed frames carry reasoning / content / calls; the defaults apply when a request names none; Review Focus 4 (greedy = the raw argmax; sampled p = the capped softmax: a seeded 10^5-draw chi-square against the capped distribution, p >= 0.01); the spec 8 / 19e acceptance tests unchanged for other models (`softcap` 0); `generation_config.json`'s `stop_strings` read as a string or a list (HF allows both - openvino.genai `d54d3768`'s fix; Muse's file has none, so the test uses a modified copy) and merged with a request's `stop`; a restored request reports `usage.prompt_tokens_details.cached_tokens` (already in the server, `server/openai.cc:64-70` - asserted for Muse).
- [x] **Step 2: implement, PASS.** Commit `server: ChatFormat::Muse - reasoning strength, the vision refusal, the softcap on the host, generation_config's defaults (spec 23e)`.

### Task 4: snapshots and the prefix cache

**Files:**
- Create: `tests/runtime/muse_snapshot_test.cc` (host, simulated devices), `tests/runtime/muse_snapshot_gpu_test.cc`
- Modify: `src/runtime/muse/muse_sizes.{h,cc}` (`SnapRun`, `state_runs`, `kv_runs`, `state_snapshot_bytes`, `kv_snapshot_bytes`), `src/runtime/muse/muse_engine.{h,cc}` (`state_bytes`, `kv_bytes`, `save_state`, `load_state`, `save_kv`, `load_kv`)

**Interfaces:**
- State at pos p (spec 23 §5.4): the 39 sliding rings' rows `[p - 2047, p)` in layer order, `[K | V][layer][positions][2 x 128]`, positions below 0 as zero runs - **81,747,968 B** (derived); blocks: the 13 full layers' KV, **13,312 B a position**. One host layout for every placement (16b's rule); `load_state` resets a session whose hand-off failed first.
- [x] **Step 1: failing tests:** `muse_snapshot_test` - the layouts on the real shapes (run counts, a wrap, below 0), inside the allocations, one layout at splits 13 and 26; save / load on host buffers; restores through `PrefixSession` at block ends 2048 / 4096 and request ends 2049 / 5000 / 300 continuing as a cold run (block ends bitwise; request ends bitwise the session that never left - Kolibri's rule for splits off a multiple of 64); one card <-> two; the server over HTTP (greedy = the decode run; a second request restored at 2048). `muse_snapshot_gpu_test` (label `gpu`): the same on the 4-layer synthetic.
- [x] **Step 2: implement, PASS on the host.** Commit `runtime: Muse-Glimmer snapshots - the sliding rings in the state, the full layers' KV in blocks (spec 23e)`.

### Task 5: `b70-serve <muse>`

**Files:**
- Create: `src/cli/muse_serve.h` (`cli::muse::MuseEngineAdapterT`, `check_serve`, `settle`), `tests/cli/muse_adapter_compile.cc`
- Modify: `src/cli/b70_serve.cc` (dispatch on `model_type muse_glimmer`; 23c's refusal removed), `tests/CMakeLists.txt` (`cli_reject_serve_muse_*`)

- [x] **Step 1:** the adapter (Kolibri's `KolibriEngineAdapterT` shape: host sampling at the tokenizer's id count; the first id after a prefill sampled from the head's row when the request samples; the store keyed `MUSE`); `--pp 1` default, `--pp 2` with `--pipeline-split` / `--pipeline-handoff`; `--max-len auto|N` with the prefill scratch; `--lm-head int8` default; `--chat-template FILE`; the startup line (Muse-Glimmer-30B, chat format muse, the template's sha256 prefix, devices, prefill l0, lm_head, kv bf16, sampling defaults sampled T 1.00 top-k 64 top-p 0.95 softcap 20, the rings' last 2047 positions in the state). Refused before the device, by name: `--mtp` / `--spec mtp` (no MTP head), `--spec dflash` / `--spec lookup` (spec 23f: the verify lists), `--kv-cache int8` (decision 8), `--prefill-backend l0-int8` / `sycl-tla`, `--max-len` > 131072, `--device` with `--pp 2`, a `chat_template.jinja` that is not the original's without `--chat-template`. Tests: `cli_reject_serve_muse_{mtp,spec,kv8,int8pf,maxlen,template}` over `tests/model/muse_glimmer` (config + the GPTQ repo's older template for the last). Commit `cli: b70-serve <muse> - one card by default, the refusals (spec 23e)`.
- [ ] **Step 2 (box; queue row 46 r46.serve / r46.chat / r46.real - pending):** `golden_server_test --chat` on the synthetic: the server's greedy chat = `b70-decode --prefill --lm-head int8` on its `prompt_token_ids`; a seeded sampled request twice identical; the prefix-cache repeat identical to a cache-off server.

### Task 6: MG5 on the engine - A4, passkey, the speed rows

**Files:**
- Create: `tools/probe/muse_passkey.sh`
- Modify: `tools/toolcall/engine_generate.sh` (a `muse` arm: `b70-serve <muse>` with the set's `reasoning_strength`), `docs/BENCHMARKS.md` (PENDING rows)

- [ ] **Step 1 (box; the scripts and the BENCHMARKS rows are written; the runs are r46.a4 / r46.passkey / r46.benchy - pending):** A4 on the engine (box, one card, int8 and bf16 heads, `l0`): `score.py --format atem <out> bf16 engine` against 23a's bf16 reference - the bar: no more parse failures than the bf16 reference's; agreement recorded (for comparison, the int4 CPU reference's from gate Q). Passkey 3 / 3 at 5 / 50 / 95 % of 131072 (`muse_passkey.sh`: 130000 ids, `--max-len 131072`, int8 head; the decode speed at that depth recorded - ~1.74 GB of KV read a token there, derived). llama-benchy rows (`serve_benchy.sh`): pp4096 tg256 at the operator's flags, prefix caching at 4k / 16k / 32k, decode at 4k / 32k / 128k. Commit `bench: Muse-Glimmer A4 on the engine, passkey at 131072, llama-benchy rows (spec 23e)`.

### Task 7: docs, the queue row

**Files:**
- Modify: `docs/19-running-models.md` (a "Muse-Glimmer-30B" section: the checkpoint, the flags, reasoning strength, the sampling defaults, what is refused), `README.md` (the CLI tables: `b70-decode` / `b70-serve` rows for Muse, `--chat-template`), `docs/11-tokenizer-and-chat-template.md` (the template's sha256, the one-BOS rule, the parser), `docs/03-models.md` (the model), spec 23 ("23e as built"), this plan's status, `docs/superpowers/plans/box-validation-queue.md` (row 46), `tools/box_validate/stages.sh` (row 46 block)

- [x] **Step 1: queue row 46. Stage list (sketch):**
  - `r46.k0` default cpu `g0.*`: no kernel changed, the server path's host tests (`template_*_test`, `toolcall_*_test`, `*_server_test`, `prefix_server_test`, `lookup_server_test`) unchanged;
  - `r46.host` default cpu: `muse_tokenizer_test` (HF cache's `tokenizer.json`), `template_muse_test`, `toolcall_muse_test`, `muse_server_test`, `muse_snapshot_test`;
  - `r46.reject` default cpu: `cli_reject_serve_muse_*`;
  - `r46.snapshot` default gpu `oracle_muse_synth`: `muse_snapshot_gpu_test` (one card, two, a snapshot crossing between them);
  - `r46.serve` / `r46.chat` default gpu `oracle_muse_synth`: Task 5 Step 2;
  - `r46.real` default gpu `muse_gptq`: the startup line on the real checkpoint (`--max-len auto` -> 131072), a seeded request twice, `golden_server_test --chat`;
  - `r46.a4` optin gpu `muse_gptq,oracle_muse_a4`: Task 6's A4;
  - `r46.passkey` optin gpu `muse_gptq`; `r46.benchy` optin gpu-self `muse_gptq,uvx`.
- [x] **Step 2: commit** `docs: spec 23 (23e as built), README / docs 19 / 11 / 03, box queue row 46 - Muse-Glimmer served`.

**Gate for the plan:** Mac - the template byte-identical on every list, the tokenizer equal, the parser = `parse_response` on complete turns and split-invariant, the server tests, the snapshot layouts and host restores; `tools/mac_check.sh --base main --quick` exit 0 (`kernel_cmdlines` +0 / -0 / ~0). Box - the restores bitwise on the card, the server's greedy chat = `b70-decode`, MG5 (A4 parse failures not above the reference's, passkey 3 / 3 at 131072), the speed rows recorded.

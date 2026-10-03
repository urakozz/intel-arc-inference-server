# Spec 7a - tool calls in the response, the request log, and the P0 probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make `b70_serve` drivable by opencode (OpenAI `tool_calls` and `reasoning_content` from the Qwen XML output), log requests so a real opencode session can be recorded, and measure the numbers prefix caching depends on (P0).

**Architecture:** a pure host parser (`src/server/toolcall.{h,cc}`) turns the generated text into reasoning, content and tool calls, in one piece or incrementally; `server.cc` and `openai.cc` put its output in the chat response and stream frames. `b70_serve --log-requests DIR` writes each request and its token ids. Two probe binaries measure host-device copies and the tail-prefill cost at depth.

**Tech Stack:** C++20, nlohmann::json, Level Zero (`l0::Mem`, `l0::CmdList`), CMake/ctest, Python 3 for the log analysis.

**Spec:** `docs/superpowers/specs/2026-09-27-spec7-prefix-caching-design.md` (§3.5, §4 C6, §5 P0, §6 7a, the stopping rule).

## Global Constraints

- `tools/box.env` is untracked: never commit it or write its contents into a committed file.
- No `rm -rf` (a hook blocks it); delete files one by one.
- Box work: `tools/box.sh sync|build|test|run`, build with `-j44`, device 0 (`ZE_AFFINITY_MASK=0`). Anything over a few minutes runs detached: `tools/box.sh run 'tools/probe/detach.sh ~/LOG cmd ...'`, then poll the log for `ALLDONE`.
- Timing: warm-up first, median of 3; A/B comparisons as interleaved pairs (memory: probe-timing-clock-state).
- Every number in a doc is **measured** unless marked **derived** or **estimated**.
- `/v1/completions` output is unchanged. Every existing test passes unchanged.
- Commit on branch `spec7-prefix-cache`; do not merge, do not push.

## Review Focus

1. A tool call whose `<parameter>` value itself contains `<` or `</parameter` inside a code string (e.g. an `edit` of XML or C++ templates): the parser must take the value up to the **last** `</parameter>` before the next `<parameter=` or `</function>` at line start, not the first `<`. Test: `edit` with `oldString` = `"a </b> <c>\n"`.
2. Text that mentions `<tool_call>` inside reasoning (the model thinking aloud about the format): reasoning is never parsed for tool calls. Test in Task 1.
3. Generation cut by `max_tokens` mid-call: an incomplete call without `</function>` and without a complete `<function=NAME>` tag is returned as content text, not dropped. Test in Task 1.
4. A streamed piece that splits a tag (`<tool_` then `call>`): the streaming parser holds back a possible tag prefix and emits nothing wrong. Test in Task 1 by streaming every golden output one byte at a time and comparing with the one-piece parse.
5. A request with `tools` absent: tool-call text is still parsed (opencode may omit tools on side requests) and parameters are typed by guessing JSON, falling back to string. Test in Task 1.

---

### Task 1: the Qwen XML parser

**Files:**
- Create: `src/server/toolcall.h`, `src/server/toolcall.cc`
- Modify: `src/server/CMakeLists.txt` (add `toolcall.cc` to `b70_server`)
- Test: `tests/server/toolcall_test.cc`, registered in `tests/CMakeLists.txt` beside `openai_test` (same pattern: `add_executable`, include dirs `src` and `tests`, link `b70_server`, `add_test`).

**Interfaces (produces):**

```cpp
namespace server {
struct ToolCall {
  std::string id;         // "call_" + 24 hex chars, from a per-output counter-seeded RNG
  std::string name;
  nlohmann::json arguments;  // object
};
struct ParsedOutput {
  std::string reasoning;  // empty when thinking is off or none was produced
  std::string content;    // text outside reasoning and outside tool calls, trimmed of the
                          // whitespace that separated it from the calls
  std::vector<ToolCall> tool_calls;
};
// thinking = the prompt ended in "<think>\n" (enable_thinking on): the text up to the first
// "</think>" is reasoning. tools = the request's "tools" array or null.
ParsedOutput parse_output(const std::string& text, bool thinking, const nlohmann::json& tools);

struct Delta {
  enum Kind { Reasoning, Content, Call } kind;
  std::string text;  // Reasoning / Content
  ToolCall call;     // Call (complete)
  uint32_t index = 0;  // Call: its position among this output's calls
};
class OutputStream {
 public:
  OutputStream(bool thinking, nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece);  // a detokenised piece
  std::vector<Delta> finish();                         // end of generation
};
}
```

Parser rules (spec §3.5, `qwen3_xml` semantics):
- Reasoning: if `thinking`, everything before the first `</think>` is reasoning (strip one leading `\n` and trailing whitespace); the `</think>` and following `\n\n` are dropped. No `</think>` at all: the whole text is reasoning.
- After reasoning: `<tool_call>` ... `</tool_call>` blocks are calls; text between/around them is content.
- Inside a call: `<function=NAME>` then parameters `<parameter=KEY>\nVALUE\n</parameter>`. VALUE ends at the last `</parameter>` that is followed (after whitespace) by `<parameter=`, `</function>` or end of the call. Strip exactly one leading and one trailing `\n`.
- Type conversion from the tool's schema (`tools[i].function.parameters.properties[KEY].type`): `string` as is; `integer`/`number`/`boolean`/`object`/`array` and unknown type: `json::parse(VALUE)` if it parses, else the string. No schema for the key or no tools: try JSON, else string; but a VALUE that parses as a JSON string literal only if quoted.
- Unterminated at the end: a call with a complete `<function=NAME>` is emitted with the parameters that are complete; otherwise its text is content.
- Streaming: `push` emits reasoning/content deltas as soon as text is certainly not part of a tag (hold back at most the longest tag prefix, `</think>`, `<tool_call>`), and one `Call` delta per `</function>`. The concatenation of deltas must equal `parse_output` of the whole text.

- [ ] **Step 1: Write the failing tests** in `tests/server/toolcall_test.cc` (the repo's test style: `main` with `check(...)` from `tests/check.h`, see `tests/server/openai_test.cc`):
  - every file in `tests/golden/toolcall/*.json` (each has the generated text; read one to see its field names) parses to calls whose names and arguments equal what `tools/toolcall/score.py` extracts; generate the expected values once with a small Python script into `tests/server/toolcall_expected.json` and commit it;
  - the five Review Focus cases above;
  - streaming: each golden text pushed one byte at a time and in random 1-7 byte pieces (fixed seed) gives deltas whose concatenation equals the one-piece parse;
  - `thinking=false`: no reasoning split.
- [ ] **Step 2: Build and run on the Mac or the box, expect FAIL** (`tools/box.sh sync && tools/box.sh build && tools/probe/ctest_env.sh -- -R toolcall_test` via `tools/box.sh run`).
- [ ] **Step 3: Implement `toolcall.cc`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `server: Qwen XML tool calls and reasoning, parsed in one piece or streamed (spec 7 §3.5)`.

### Task 2: the chat response carries them

**Files:**
- Modify: `src/server/openai.h`, `src/server/openai.cc`, `src/server/server.cc`
- Test: `tests/server/openai_test.cc`, `tests/server/protocol_test.cc` (the `MockEngine` in `tests/server/mock.h` scripts the generated ids; extend it to script a text containing reasoning and a tool call)

**Interfaces:**
- Consumes: `parse_output`, `OutputStream`, `Delta`, `ToolCall` (Task 1).
- Produces: `completion_body(...)` gains a `const ParsedOutput*` (null for `/v1/completions`); new `stream_frame_reasoning(r,id,created,text)` and `stream_frame_tool_call(r,id,created,const ToolCall&,uint32_t index)`; `Usage` gains `uint32_t cached_tokens = 0` (written as `usage.prompt_tokens_details.cached_tokens`; stays 0 until plan 7c).

Rules: chat, non-streaming: `message = {role, content (null if empty and tool_calls non-empty), reasoning_content (omitted if empty), tool_calls (omitted if none: [{id, type:"function", function:{name, arguments: arguments.dump()}}])}`; `finish_reason` becomes `"tool_calls"` when there are calls and generation ended by EOS. Streaming: route `streamer->push(id)` text through `OutputStream`; `Reasoning` → `delta.reasoning_content`, `Content` → `delta.content`, `Call` → `delta.tool_calls = [{index, id, type, function:{name, arguments}}]`. Stop strings apply to content only (keep `server.cc`'s hold logic on the content path).

- [ ] **Step 1: Failing tests:** non-streaming and streaming bodies for a scripted output `"plan it</think>\n\nI'll read it.\n<tool_call>\n<function=read>\n<parameter=filePath>\n/w/x.cc\n</parameter>\n</function>\n</tool_call>"` with `enable_thinking` on; the same with thinking off; a `/v1/completions` request with the same text is byte-identical to today's body.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the server tests and `golden_server_test`, expect PASS.**
- [ ] **Step 5: Commit** `server: chat responses carry reasoning_content and tool_calls (spec 7 §3.5)`.

### Task 3: the request log

**Files:**
- Modify: `src/cli/b70_serve.cc` (flag `--log-requests DIR`), `src/server/server.h`/`server.cc` (an optional log hook in `Options`)
- Create: `tools/prefix/analyze_log.py`, `tools/prefix/README.md`

For each request, write `DIR/NNNNNN.json`: `{"t_start", "t_first_token", "t_end" (steady-clock seconds), "endpoint", "request" (the body as received), "prompt_ids", "out_ids", "response_text"}`. `NNNNNN` counts from 1 per server start.

`analyze_log.py DIR` prints, per request: prompt length, the longest common prefix with the **previous** request and with **any** earlier request (prompt ids + out ids), and whether that prefix ends exactly at an earlier prompt's end; then totals: the fraction of prompt tokens a perfect prefix cache could reuse. Unit-test it on a synthetic log (`tools/prefix/test_analyze_log.py`, run with `python3 -m pytest` or plain asserts like `tools/toolcall/test_score.py`).

- [ ] **Step 1:** failing test for `analyze_log.py` on a 3-request synthetic log; **Step 2:** implement; **Step 3:** add the flag and the hook; a `protocol_test` case that runs a request with a temp dir and reads the file back.
- [ ] **Step 4:** full server tests pass.
- [ ] **Step 5: Commit** `serve: --log-requests, and the prefix analysis of a log (spec 7 P0)`.

### Task 4: record an opencode session (operator step)

The agent cannot drive opencode. **Stop and hand back** with these instructions for the operator, then continue with Task 5 while waiting; Task 6 needs the log.

```
# on the box
build/src/cli/b70-serve <snapshot> --max-len 131072 --log-requests ~/oc-log-1 --port 8000
# on the Mac: ssh -L 8000:localhost:8000 <box>; opencode with an OpenAI-compatible provider
# at http://localhost:8000/v1, model b70; a real coding task of 20+ turns in this repo,
# long enough to pass 30k tokens of history.
```

If opencode rejects the responses, that is a Task 2 defect: fix it with a test that reproduces the body opencode choked on.

### Task 5: P0, host-device copies and the tail floor

**Files:**
- Create: `tools/probe/probe_host_copy.cc`, `tools/probe/probe_tail.cc`, registered as executables (not tests) in `tools/probe/CMakeLists.txt` the way `probe_flash_attn` is.
- Create: `docs/probe-prefix-cache-2026-09-27.md`

`probe_host_copy`: allocate `l0::Mem(ctx, MemKind::Host, n)` for n = 4, 8, 16, 24, 32, 48 GB (stop at the first failure, report the largest that worked); copy bandwidth device→host and host→device on an immediate `l0::CmdList` for 128 MiB (one KV block) and 166.7 MB (one state snapshot), median of 5 after 2 warm-ups; the strided form a KV range really takes: 16 layers × one contiguous `[n][4][256]` bf16 slice each for K and V, n = 2048 and n = 60000.

`probe_tail`: load the model at max_len 131072, `prefill` N ids of `tests/golden/prompts/long32k.ids` (repeated to length) then time `prefill` of T more ids, N ∈ {0, 30000, 60000}, T ∈ {16, 256, 1024, 2048}; median of 3 with a fresh `reset()` + base prefill each time (only the tail is timed).

- [ ] **Step 1:** write both; build; run detached on the idle box (box idle protocol, memory: box-idle-protocol).
- [ ] **Step 2:** write the doc: tables, then derived S1 and S2 estimates (restore bytes / measured bandwidth + tail time).
- [ ] **Step 3: Stopping rule (spec §6):** host-device bandwidth under 4 GB/s either way, or the largest pinned allocation under 8 GB → stop, commit the doc, hand back.
- [ ] **Step 4: Commit** `probe: host-device copies and the tail prefill at depth (spec 7 P0)`.

### Task 6: the hit rate of the recorded session

- [ ] **Step 1:** copy the operator's log into `tests/golden/opencode/session1/` (the request files only; if over 20 MB total, keep the first 40 requests) and run `tools/prefix/analyze_log.py` on it.
- [ ] **Step 2:** add a section to `docs/probe-prefix-cache-2026-09-27.md`: requests, how many are side requests (short, prefix < 1k shared with the main thread), the reusable fraction, and where the divergences fall (at an earlier prompt end, inside generated text, elsewhere). If thinking is dropped from history or kept, say which, from the ids.
- [ ] **Step 3: Commit** `probe: the prefix reuse an opencode session allows (spec 7 P0)`.

**Gate for the plan:** full suite green on the box (`tools/probe/ctest_env.sh --`), C6 passing, the P0 doc committed, stopping rule evaluated. Hand back with the P0 table and the hit-rate numbers.

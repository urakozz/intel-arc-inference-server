# Spec 3 / T4 - `b70-serve`, the golden-through-server gate, llama-benchy, host sampling, records Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire T1-T3 to the real engine as `b70-serve`, prove the server produces `b70-decode`'s exact ids and satisfies llama-benchy end to end, measure the HTTP cost against the CLI, add host sampling last (only if it measures under 2% of tg), record the gate, tag `spec3-done`.

**Architecture:** `src/cli/b70_serve.cc` = argument parsing + three adapters (`tok::Tokenizer`→`TokIface`, `chat::Template`→`TemplateIface`, `runtime::Engine`→`EngineIface`) + `server::Server`. It links exactly what `b70-decode` links plus `b70_tokenizer` and `b70_server`, prefill component included. The gate test spawns the real binary.

**Tech Stack:** C++17, `posix_spawn`, `httplib::Client`, `tools/bench_decode.sh`, `uvx llama-benchy` (present on the box at `~/.local/bin/uvx`, measured 2026-09-06).

**Spec:** §3.4 (amended), §3.5, §3.6 T4, §4, bars 3-6.

## Global Constraints

- Branch `spec1.7-codex-exp`; no push; commits end `Claude-Session: `.
- The Mac never compiles; `tools/box.sh` only; scp not heredocs. `ZE_AFFINITY_MASK` is passed through by `tools/bench_decode.sh`; series rows are device 0 unmasked; the two B70s differ by 3.4% on prefill (memory: box-idle-protocol).
- **Processes:** you may start `b70-serve` on the box and you must stop it yourself when done (`kill <its pid>` - the pid you recorded at start; never a pid you did not start). Never docker, never the operator's server, never delete anything.
- Record grade requires zero DRM holders (plan 6e Task 3 Step 1's command) and `docker ps -q | wc -l` = 0; otherwise iterate grade, labelled.
- Nothing in `src/runtime` or `src/kernels` changes (bar 6, tightened by the §3.4 amendment). Decode invariants re-proven after the last commit: 774 kernels / 19 modules, replay determinism, golden 94/94 + 93/93, decode bench row within the 0.09% drift rule of 32.22.
- Requires plans 7a, 7b, 7c landed.

---

## File structure

| path | responsibility |
|---|---|
| `src/cli/serve_adapters.h` | the three adapters (header-only, used by the binary and by nothing else) |
| `src/cli/b70_serve.cc` | the binary |
| `src/cli/CMakeLists.txt` | `b70-serve` target (mirrors `b70-decode`) |
| `tests/server/golden_server_test.cc` | bar 3 |
| `tools/serve_bench.sh` | starts the server, runs llama-benchy, stops the server, prints the rows |
| `docs/BENCHMARKS.md`, `README.md`, `docs/04-architecture.md`, `docs/11-...` | records |
| `docs/superpowers/specs/2026-09-09-spec3-gate-memo.md` | the memo (either path) |

---

### Task 1: Adapters and `b70-serve`

**Files:** Create `src/cli/serve_adapters.h`, `src/cli/b70_serve.cc`; modify `src/cli/CMakeLists.txt`.

**Interfaces:** Consumes `tok::Tokenizer`/`tok::Streamer` (7a/7b), `chat::Template` (7b), `server::{Deps,Options,Server}` (7c), `runtime::Engine`, `loader::load`, and the loader's snapshot resolver in `src/loader/snapshot.h` (read it: `load()`'s comment names "resolve_snapshot rules" - call the same function to get the directory that holds `tokenizer.json`, `chat_template.jinja`, `tokenizer_config.json`, `generation_config.json`).

- [ ] **Step 1: Adapters**

```cpp
// src/cli/serve_adapters.h
#pragma once
#include <set>
#include "server/deps.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/streamer.h"
#include "tokenizer/tokenizer.h"
#include "runtime/engine.h"
struct TokAdapter : server::TokIface {
  tok::Tokenizer t;
  explicit TokAdapter(const std::string& json) : t(json) {}
  std::vector<uint32_t> encode(std::string_view s) override { return t.encode(s, false); }
  std::string decode(const std::vector<uint32_t>& ids) override { return t.decode(ids, false); }
  struct S : server::StreamerIface { tok::Streamer st; explicit S(const tok::Tokenizer& t) : st(t) {}
    std::string push(uint32_t id) override { return st.push(id); } std::string flush() override { return st.flush(); } };
  std::unique_ptr<server::StreamerIface> streamer() override { return std::make_unique<S>(t); }
  uint32_t vocab_used() override { return t.vocab_size(); }   // 248077
};
struct TemplateAdapter : server::TemplateIface {
  chat::Template tmpl;
  explicit TemplateAdapter(const std::string& dir) : tmpl(dir) {}
  std::string render(const nlohmann::json& m, const nlohmann::json& tools, bool think) override { return tmpl.render(m, tools, think); }
};
struct EngineAdapter : server::EngineIface {
  runtime::Engine& eng;
  uint32_t vocab_used;
  explicit EngineAdapter(runtime::Engine& e, uint32_t vocab_used) : eng(e), vocab_used(vocab_used) {}
  void reset() override { eng.reset(); }
  void prefill(const std::vector<uint32_t>& ids) override {
#if B70_HAVE_PREFILL
    eng.prefill(ids);
#else
    eng.ingest(ids);
#endif
  }
  uint32_t step(const server::Sampling& s) override {
    const uint32_t id = eng.generate(1)[0];     // the pending id; one replay has now produced the next logits
    if (!s.greedy) sample_into_control(s);      // Task 5; a no-op stub until then
    return id;
  }
  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return eng.pos(); }
  void sample_into_control(const server::Sampling&) {}   // Task 5 replaces this
};
```

- [ ] **Step 2: The binary** - `b70-serve <snapshot-or-repo> [--host 0.0.0.0] [--port 8000] [--max-len 16384] [--device N] [--served-name NAME] [--queue 4]`. Parse exactly as `b70_decode.cc` does (`parse_u32`, `value(i, flag)`, rejections before the device); resolve the snapshot directory; load: `l0::Context ctx(device); loader::LoadedModel model = loader::load(ctx, path, max_len); runtime::Engine eng(ctx, std::move(model), max_len);` (the same `StdoutToStderr` trick around the load so stdout stays clean); read `generation_config.json` → `eos_token_id` (array or int) into `Options::eos_ids`; construct the three adapters and `server::Server`; print one line to stderr: `b70-serve: <name> on http://<host>:<port>, max_len <N>, eos [..], prefill <on|off>`; `listen()`. `SIGTERM`/`SIGINT` → `stop()` and exit 0.
- [ ] **Step 3: CMake** - copy `b70-decode`'s block: `add_executable(b70-serve b70_serve.cc)`, same includes, `target_link_libraries(b70-serve PRIVATE b70_runtime b70_loader b70_model b70_l0 b70_tokenizer b70_server)`, the same `B70_HAVE_PREFILL` / `b70_link_prefill` guard, `b70_target_kernel_dir`, the same `add_dependencies` on all kernels. Guard the whole target with `if(B70_TOKENIZER_ENABLED)`.
- [ ] **Step 4: Smoke on the box** - `tools/box.sh build` then, on the box (via `tools/box.sh run`), start it in the background with the gate checkpoint (urakozz 84575a1) on port 8123 and `ZE_AFFINITY_MASK=1` (`setsid nohup ./build/src/cli/b70-serve /home/user/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d --port 8123 > /tmp/b70-serve.log 2>&1 & echo $!` - record the pid), wait for `curl -s localhost:8123/v1/models`, then `curl -s localhost:8123/v1/chat/completions -d '{"model":"x","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'`. Expected: a JSON with a sentence naming Paris. Then a streaming request with `-N` and watch frames arrive one per ~31 ms. Stop it: `kill <pid>`. Paste the two responses into the commit message body.
- [ ] **Step 5: Commit** - `git commit -m "feat(cli): b70-serve - the OpenAI server over runtime::Engine + prefill"`.

---

### Task 2: Golden through the server (bar 3)

**Files:** Create `tests/server/golden_server_test.cc`; register in the T1 block of `tests/CMakeLists.txt` with `add_test(NAME golden_server_test COMMAND golden_server_test $<TARGET_FILE:b70-serve> $<TARGET_FILE:b70-decode> ${CMAKE_SOURCE_DIR}/tests/golden/prompts /home/user/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d)`.

- [ ] **Step 1: The test** - for each of `prose`, `code`, `cjk`:
  1. Read `<p>.txt` (strip trailing `\n`s exactly as `tools/oracle/tokenize.py` does) and `<p>.ids`.
  2. Spawn `b70-serve <snap> --port 0`? - no: pick a free port by binding a socket and closing it; spawn `b70-serve <snap> --port <P> --device` from `ZE_AFFINITY_MASK`-free env (the test inherits the env the suite runs with) via `posix_spawn`; poll `GET /v1/models` every 500 ms up to 120 s.
  3. `POST /v1/completions {"prompt": <txt>, "max_tokens": 32, "return_token_ids": true}` → `prompt_token_ids == <p>.ids` (**bar 1 meets bar 3**: our encode of the committed text reproduces the committed ids); `choices[0].token_ids` has 32 entries unless an EOS stopped it (then fewer, and the test records which).
  4. Spawn the CLI reference and capture stdout (one id per line). **CORRECTION
     (2026-09-09), the premise below was false and the implementer's clean block
     found it:** an earlier revision said "spawn `b70-decode <snap> --ids <p>.ids
     --n 32` … equality must be exact - same engine, same device, same replays,
     there is no tie question between an engine and itself." That is wrong.
     `b70-serve` prefills (`Engine::prefill`); `--ids` **ingests** (one replay
     per id). Those are independently rounded paths - that is the whole subject
     of ruling **A26**, which measured the divergence, found every one of it on
     a row where decode's own top-2 logits sit inside one bf16 ulp, and settled
     the grading: tie-aware plus teacher-forced. Measured here: `prose` and
     `code` 32/32 exact, `cjk` exact for 22 ids then divergent - the same shape
     A26 recorded for `cjk`.
     **The fix is to compare like with like, so bar 3 tests the layer it is
     for.** Give `b70-decode` a `--prefill` flag on `--ids` mode (route through
     `Engine::prefill` instead of `ingest`; `src/cli` is not a protected
     directory, `src/runtime` is untouched, and the flag is one branch beside
     the existing `have_pp` one), and have this test use it. Then the two sides
     run the same engine path and **exactness is a legitimate bar**: bar 3 is
     about the tokenizer, template and HTTP layer, and the prefill-vs-ingest
     question is already gated at 18/18 by `prefill_consistency_test`.
     Do **not** import tie-aware semantics here instead - this test is
     out-of-process and cannot see logits, so it could not apply them honestly.
     (If the server stopped early on EOS, compare the prefix and print it.)
  5. `SIGTERM` the server child, `waitpid`.
  Spawn the server ONCE for all three prompts (one 13 s load), the decode reference three times (three loads, ~40 s, estimated from the recorded 13 s load).
- [ ] **Step 2: Run** - `tools/box.sh test golden_server_test`. Expected: `3 prompts: prompt ids identical, 32/32 generated ids identical` ×3. A mismatch is a finding: print both id lists and stop.
- [ ] **Step 3: Commit** - `git commit -m "test(server): golden_server_test - b70-serve reproduces b70-decode's ids on the three golden prompts"`.

---

### Task 3: llama-benchy end to end (bar 4) and the HTTP-vs-CLI control (bar 5)

**Files:** Create `tools/serve_bench.sh`; results go to `docs/BENCHMARKS.md` in Task 6.

- [ ] **Step 1: `tools/serve_bench.sh`** - on the box: starts `b70-serve <MODEL> --port 8000 --served-name b70` (env `MODEL`, default the urakozz 84575a1 snapshot; `ZE_AFFINITY_MASK` honoured if set) with `setsid nohup`, records the pid, waits for `/v1/models`, runs

```bash
~/.local/bin/uvx llama-benchy --base-url http://127.0.0.1:8000/v1 --model b70 \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

(the standing command from `docs/BENCHMARKS.md:112-116` with the served name), captures its table, then `kill $PID` and waits. Prints the pp/tg rows and the server's stderr tail (the coherence test's answer).

- [ ] **Step 2: Idle proof** (plan 6e Task 3 Step 1's command) - record; grade accordingly.
- [ ] **Step 3: Run it three times** (each invocation is itself 3 runs, as the vLLM rows were taken): record `pp4096` and `tg256` per invocation and the median.
- [ ] **Step 4: The CLI control row, same session, same device** - `tools/bench_decode.sh --runs 3 --depth 4096 --tg 256` (unmasked, device 0) → tg256 median. **Bar 5:** `tg_http / tg_cli ≥ 0.98`. Also `tools/bench_decode.sh --pp 4096 --tg 256 --runs 3` → the device-side pp row, beside the HTTP pp row: the difference IS spec 3's cost of tokenizer + template + HTTP + first frame (state it in t/s and in ms per 4096 tokens: `4096/pp_http − 4096/pp_device`).
- [ ] **Step 5: Coherence** - llama-benchy exits non-zero if its coherence test fails; paste the server log's answer to "What is the capital of France?".
- [ ] **Step 6: Commit the script** - `git commit -m "bench: tools/serve_bench.sh - llama-benchy against b70-serve"`; numbers go to Task 6.

---

### Task 4: Open WebUI / opencode smoke (usability, not a bar)

- [ ] **Step 1:** With the server up, one `curl` streaming chat with a 3-turn conversation (the template test's messages) and `enable_thinking` on and off via `chat_template_kwargs`; confirm the `<think>` block appears only when on. Record the two first lines. (Open WebUI itself is the operator's to point at port 8000; say so in the report.)

---

### Task 5: Host sampling (§3.5) - last, and gated on its own measurement

**Files:** Modify `src/cli/serve_adapters.h` (`sample_into_control`), create `tests/server/sampling_test.cc` (host-only: the sampler function over a synthetic logits row).

- [ ] **Step 1: Pre-register** (commit before measuring): cost ≤ 2% of the 31.0 ms step = **≤ 0.62 ms/token** for `temperature 1.0, top_k 20, top_p 0.95`; derived estimate 0.2-0.4 ms (993 KB readback over PCIe at ~10 GB/s ≈ 0.1 ms + a 248k partial sort ≈ 0.1-0.3 ms).
- [ ] **Step 2: The sampler** - `uint32_t sample(const float* logits, uint32_t vocab_used, const server::Sampling& s, std::mt19937_64& rng)`: mask ids ≥ vocab_used; divide by temperature; top-k via `std::partial_sort` on indices (k = min(top_k, vocab_used)); softmax over the k; top-p cumulative cut (keep at least one); `std::discrete_distribution`. Unit test: on a synthetic row with one dominant logit, `temperature 1e-3` returns the argmax 1000/1000 times; with `top_k 1` always the argmax; with a flat row and `top_p 0.5`, all samples come from the first half of the sorted set; seeded → reproducible.
- [ ] **Step 3: `sample_into_control`** - after `generate(1)` returns, the replay's logits are in `eng.buffers().logits` (fp32 `[1][248320]`, `buffers.h:170`); read them back with `l0::CmdList::immediate(ctx)` + `imm.copy(host.data(), eng.buffers().logits.ptr(), 248320 * 4)` (the pattern `tests/golden/golden_gate_test.cc:161,370` uses); sample; write `eng.buffers().control.as<runtime::Control>()->cur_token[0] = id` - exactly the ingest protocol `Control` supports (spec §3.5) - so the next `generate(1)` embeds the sampled id and returns it. Note `generate(1)` returned the *previous* pending id; the sampled id replaces the argmax the device just wrote for the *next* step. The `imm` list and the host buffer are created once in the adapter, not per token.
- [ ] **Step 4: Measure** - `tools/serve_bench.sh` cannot pass `temperature`; instead a 256-token streaming request via `curl` with and without `"temperature": 1.0`, timed server-side: add a stderr line per request `gen: N tokens, X ms, Y ms/token` in `b70-serve` (one line, after the request). Compare ms/token greedy vs sampled, 3 requests each, median. **Ship if ≤ 0.62 ms slower; otherwise leave the code behind a `--sampling off` default and record the number** - greedy is never slowed either way (the readback runs only when `!greedy`).
- [ ] **Step 5: Commit** - `git commit -m "feat(serve): host temperature/top-k/top-p sampling, <X> ms/token measured (bar: 0.62)"`.

---

### Task 6: Gate, records, tag

**Files:** Modify `docs/BENCHMARKS.md` (new section "The spec-3 gate rows - `<SHA>`, <date>": HTTP pp/tg rows ×3 invocations + median, the CLI control rows, the HTTP delta in ms/4096 and in % of tg, box conditions, grade), `README.md` (v1 definition-of-done rows: server, `/v1/completions`, tokenizer parity - with the numbers), `docs/04-architecture.md` (Server status), `docs/11` (status: shipped), `docs/superpowers/specs/2026-09-04-spec3-tokenizer-http-design.md` (status line); create the memo `docs/superpowers/specs/2026-09-09-spec3-gate-memo.md` (template: `2026-09-05-spec2-gate-memo.md`).

- [ ] **Step 1: Score the six bars** - 1: `parity_test` + `streamer_test`; 2: `template_test`; 3: `golden_server_test`; 4: llama-benchy exit 0 with pp/tg rows; 5: `tg_http/tg_cli` vs 0.98; 6: full suite, `-Werror`, `git diff --stat spec2-done..HEAD -- src/runtime src/kernels` empty, decode invariants re-proven (774/19, replay determinism, golden 94/94 + 93/93, decode bench within 0.09% of 32.22).
- [ ] **Step 2: Records** - every cell labelled measured / derived; grade stated; the pp row labelled "HTTP-inclusive, C = 2048, urakozz 84575a1 (bf16 lm_head)" beside spec 2's device-side row and vLLM's 1973 with the three labels.
- [ ] **Step 3: Memo** - verdict table; the HTTP cost; what each stream cost in wall time; falsified predictions (e.g. the sampling estimate); owed items (Vishva byte-matched HTTP row if not taken; prefix caching - docs/04 follow-on); ruling request for what comes next.
- [ ] **Step 4: Commit and tag** - all bars met: `git commit -m "docs(spec3): record the gate - tg256 <X> t/s over HTTP (<Y>% of CLI), pp4096 <Z> t/s HTTP-inclusive"` and `git tag -a spec3-done -m "spec 3 gate: ..."`. Any bar short: commit the memo without the tag and STOP for the operator's ruling.

---

## Self-review

**Spec coverage:** T4's owned items (wiring, binary, golden-through-server, llama-benchy, HTTP-vs-CLI, host sampling) → Tasks 1-5; §4 gate + records + tag → Task 6; bars 3-6 → Tasks 2, 3, 6; §3.4 amended loop (`generate(1)`, `prefill` when available) → Task 1's `EngineAdapter`; `eos_token_id` both ids → Task 1 Step 2; `usage.prompt_tokens` = tokenizer count of the rendered prompt → 7c's core, verified here by bar 3's `prompt_token_ids` check. **Placeholders:** `<SHA>`, `<X>`, `<Y>`, `<Z>`, `<P>` are execution outputs. **Types:** adapters implement `deps.h` verbatim; `Sampling` flows `parse_request → step → sample_into_control` unchanged.

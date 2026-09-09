# Spec 3 / T3 - HTTP server (`src/server/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) with unbuffered per-token SSE, stop handling, a bounded FIFO, and OpenAI-shaped errors - built and tested against MOCK tokenizer/template/engine, with no GPU.

**Architecture:** `src/server/` is a leaf over three abstract interfaces (`TokIface`, `TemplateIface`, `EngineIface`) declared in `deps.h`; the real adapters over `tok::Tokenizer`, `chat::Template` and `runtime::Engine` are plan 7d's. One `Server` class on `cpp-httplib`; request parsing/validation in `openai.cc`; one generation core shared by the stream and non-stream paths; the streaming path writes each token's frame from inside httplib's chunked content provider immediately after the engine step returns.

**Tech Stack:** C++17, `cpp-httplib` (vendored single header, pinned), `nlohmann/json` (plan 7b's vendored copy), `std::thread`/`std::condition_variable`, `tests/check.h`.

**Spec:** `docs/superpowers/specs/2026-09-04-spec3-tokenizer-http-design.md` §3.3, §3.4 (amended 2026-09-06), §3.5 (parsing only), §7 risk 4.

## Global Constraints

- Branch `spec1.7-codex-exp`; no push; commits end `Claude-Session: `.
- The Mac never compiles; `tools/box.sh` only; never docker/kill/delete/restart on the box (a server the TEST starts in-process is not a box process - the protocol test binds a loopback port and stops its own server).
- `src/server/` depends on `third_party/` and on nothing in `src/` except `deps.h`'s own declarations (no tokenizer, no runtime, no l0) - the GPU is never linked.
- `-Wall -Wextra -Werror`; httplib is `-isystem`. `third_party/VERSIONS` gains the httplib line.
- Wire shapes are OpenAI's; extensions honoured: `min_tokens`, `ignore_eos`, `return_token_ids` (vLLM's shape: `prompt_token_ids` top-level, `choices[i].token_ids`), `stream_options.include_usage`. Unknown fields ignored.
- Requires plan 7b Task 1 (json vendored). If 7b has not landed, do 7b Task 1 here first (idempotent).

---

## File structure

| path | responsibility |
|---|---|
| `third_party/httplib/httplib.h` | vendored, pinned |
| `src/server/deps.h` | the three interfaces + `Sampling` |
| `src/server/openai.h`, `openai.cc` | JSON request → `Request` (validated), response/frame builders, error bodies |
| `src/server/server.h`, `server.cc` | `Server`: routes, FIFO, generation core, SSE |
| `src/server/CMakeLists.txt` | `b70_server` static lib |
| `tests/server/mock.h` | `MockTok`, `MockTemplate`, `MockEngine` |
| `tests/server/protocol_test.cc` | the protocol test (in-process server + `httplib::Client`) |

---

### Task 1: Vendor cpp-httplib

**Files:** Create `third_party/httplib/httplib.h`; modify `third_party/VERSIONS`, `third_party/CMakeLists.txt`.

- [ ] **Step 1:** `gh api repos/yhirose/cpp-httplib/releases/latest --jq .tag_name` → `<tag>`; `curl -sSL -o third_party/httplib/httplib.h https://raw.githubusercontent.com/yhirose/cpp-httplib/<tag>/httplib.h`; `shasum -a 256`; append the VERSIONS line.
- [ ] **Step 2:** `third_party/CMakeLists.txt`: `add_library(b70_third_party_httplib INTERFACE)`, `target_include_directories(... SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/httplib)`, `target_link_libraries(b70_third_party_httplib INTERFACE pthread)`. No OpenSSL (`CPPHTTPLIB_OPENSSL_SUPPORT` undefined - TLS is out of scope, spec §6).
- [ ] **Step 3:** Commit - `git commit -m "build(third_party): vendor cpp-httplib <tag>"`.

---

### Task 2: `deps.h` - the interfaces

**Files:** Create `src/server/deps.h`, `src/server/CMakeLists.txt` (empty lib for now: `add_library(b70_server STATIC openai.cc server.cc)` is added in Task 3; here just the interface target), modify `CMakeLists.txt` (`add_subdirectory(src/server)` after `src/tokenizer`).

**Interfaces (binding for plan 7d's adapters and this plan's mocks):**

```cpp
// src/server/deps.h
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>
namespace server {
struct StreamerIface {          // plan 7b's tok::Streamer behind an interface
  virtual ~StreamerIface() = default;
  virtual std::string push(uint32_t id) = 0;
  virtual std::string flush() = 0;
};
struct TokIface {
  virtual ~TokIface() = default;
  virtual std::vector<uint32_t> encode(std::string_view text) = 0;      // add_special = false
  virtual std::string decode(const std::vector<uint32_t>& ids) = 0;     // skip_special = false
  virtual std::unique_ptr<StreamerIface> streamer() = 0;
  virtual uint32_t vocab_used() = 0;                                    // 248077: ids >= this are padding
};
struct TemplateIface {
  virtual ~TemplateIface() = default;
  virtual std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                             bool enable_thinking) = 0;                 // throws std::runtime_error on a bad conversation
};
struct Sampling {                // §3.5: absent temperature → greedy (documented deviation from OpenAI's default 1.0)
  bool greedy = true;
  float temperature = 1.0f;
  uint32_t top_k = 20;           // generation_config.json defaults (docs/11)
  float top_p = 0.95f;
  uint64_t seed = 0;
  bool has_seed = false;
};
struct EngineIface {
  virtual ~EngineIface() = default;
  virtual void reset() = 0;
  virtual void prefill(const std::vector<uint32_t>& ids) = 0;   // prompt in; the first generated id becomes pending
  virtual uint32_t step(const Sampling& s) = 0;                 // returns the pending id, runs one replay (7d: generate(1)[0], sampling applied)
  virtual uint32_t max_len() = 0;
  virtual uint32_t pos() = 0;
};
struct Deps { TokIface& tok; TemplateIface& tmpl; EngineIface& engine; };
}  // namespace server
```

- [ ] **Step 1:** Write it; `add_library(b70_server INTERFACE)` temporarily with `target_include_directories(b70_server INTERFACE ${CMAKE_SOURCE_DIR}/src)` and `target_link_libraries(b70_server INTERFACE b70_third_party_json b70_third_party_httplib)`.
- [ ] **Step 2:** `tools/box.sh build` (header only compiles when included - Task 3's test includes it). Commit - `git commit -m "feat(server): deps.h - the tokenizer/template/engine interfaces the server is written against"`.

---

### Task 3: Request parsing and response shapes (`openai.*`) with their unit test

**Files:** Create `src/server/openai.h`, `openai.cc`, `tests/server/openai_test.cc`; `add_library(b70_server STATIC openai.cc)` replaces the INTERFACE lib.

**Interfaces:**

```cpp
// src/server/openai.h
namespace server {
struct Request {
  bool chat = false;                     // /v1/chat/completions vs /v1/completions
  nlohmann::json messages;               // chat
  nlohmann::json tools;                  // chat, may be null
  bool enable_thinking = true;           // chat_template_kwargs.enable_thinking, default true (the template's own default)
  std::string prompt;                    // completions
  std::optional<uint32_t> max_tokens;    // absent → max_len - prompt_len (spec: "up to context")
  uint32_t min_tokens = 0;
  bool ignore_eos = false;
  bool stream = false;
  bool include_usage = false;            // stream_options.include_usage
  std::vector<std::string> stop;         // 0..4 strings
  bool return_token_ids = false;
  Sampling sampling;
  std::string model;                     // echoed back
};
struct BadRequest : std::runtime_error { using std::runtime_error::runtime_error; };
// Throws BadRequest with an OpenAI-style message for: non-object body, missing
// messages/prompt, n != 1, empty messages, stop with > 4 entries, max_tokens 0,
// temperature < 0, top_p outside (0,1], top_k < 0.
Request parse_request(const std::string& body, bool chat);
// OpenAI error body: {"error":{"message":..,"type":..,"param":null,"code":null}}
std::string error_body(const std::string& message, const std::string& type);
struct Usage { uint32_t prompt_tokens = 0, completion_tokens = 0; };
// Non-stream bodies.
std::string completion_body(const Request& r, const std::string& id, uint64_t created,
                            const std::string& text, const std::string& finish_reason, Usage u,
                            const std::vector<uint32_t>* prompt_ids, const std::vector<uint32_t>* out_ids);
// Stream frames (each already prefixed "data: " and suffixed "\n\n").
std::string stream_frame_role(const Request& r, const std::string& id, uint64_t created);        // chat only: first frame carries {"role":"assistant"}
std::string stream_frame_text(const Request& r, const std::string& id, uint64_t created, const std::string& text);
std::string stream_frame_finish(const Request& r, const std::string& id, uint64_t created, const std::string& finish_reason);
std::string stream_frame_usage(const Request& r, const std::string& id, uint64_t created, Usage u);
inline const char* kDone = "data: [DONE]\n\n";
std::string models_body(const std::string& served_name, uint64_t created);
}
```

- [ ] **Step 1: Failing unit test** - `tests/server/openai_test.cc` (no server, no threads): parse a chat body with `messages`, `max_tokens 5`, `stop ["\n"]`, `stream true`, `stream_options {include_usage:true}`, `temperature 0.7`, `top_k 40`, unknown fields `cache_prompt`, `return_token_ids true`; check every field; parse `n: 2` → `BadRequest`; parse `"{"` → `BadRequest`; `temperature` absent → `sampling.greedy == true`; `temperature 0` → greedy; `temperature 0.7` → not greedy, top_k 40, top_p 0.95 default; `completion_body` parses back as JSON with `object == "chat.completion"`, `choices[0].message.content`, `usage`, and - with ids passed - `prompt_token_ids` and `choices[0].token_ids`; `stream_frame_text` starts with `data: ` and ends with `\n\n` and parses; `error_body` shape. Register: `add_executable(openai_test server/openai_test.cc)`, links `b70_server`, `add_test`.
- [ ] **Step 2: Run → fails to compile.**
- [ ] **Step 3: Implement `openai.cc`** - straightforward `nlohmann::json` handling; `parse_request` accepts both chat `messages` (array of objects with `role` string and `content` string - content arrays (multimodal) → BadRequest "content must be a string") and completions `prompt` (string only; arrays → BadRequest); `chat_template_kwargs.enable_thinking` bool if present; `sampling.greedy = !temperature.has_value() || temperature == 0`. Frames: chat → `object: "chat.completion.chunk"`, `choices[0].delta`; completions → `object: "text_completion"`, `choices[0].text`. `finish_reason` null on text frames, `"stop"`/`"length"` on the finish frame; the usage frame has `choices: []` and `usage`.
- [ ] **Step 4: Run → passes. Commit** - `git commit -m "feat(server): OpenAI request parsing and response/frame builders + openai_test"`.

---

### Task 4: `Server` - routes, FIFO, generation core, SSE

**Files:** Create `src/server/server.h`, `server.cc`; `target_sources(b70_server PRIVATE server.cc)`.

**Interfaces:**

```cpp
// src/server/server.h
namespace server {
struct Options {
  std::string host = "0.0.0.0";
  int port = 8000;                  // 0 = any free port (tests); see bound_port()
  std::string served_model = "b70";
  size_t queue_depth = 4;           // waiting requests beyond this get 503
  bool tcp_nodelay = true;
};
class Server {
 public:
  Server(Deps deps, Options opts);
  ~Server();
  bool listen();                    // blocking; false if bind failed
  bool start();                     // non-blocking: listens on a thread, returns once bound (tests)
  void stop();
  int bound_port() const;
 private:
  // The one generation core. `emit` is called with each safe-to-send text piece
  // (possibly "" for a held token) - the stream path writes a frame per call, the
  // non-stream path concatenates. Returns finish_reason and fills usage/ids.
  struct Outcome { std::string finish_reason; Usage usage; std::vector<uint32_t> prompt_ids, out_ids; std::string text; };
  Outcome generate(const Request& r, const std::function<void(const std::string&)>& emit);
  // FIFO: a ticket per request; `waiting_` bounded by queue_depth → 503.
  bool acquire(uint64_t& ticket); void release();
  ...
};
}
```

- [ ] **Step 1: Failing test** - Task 5's `protocol_test.cc` (write it now, all cases; it will not compile).
- [ ] **Step 2: Implement.** The load-bearing parts, in code:

```cpp
// generation core (server.cc)
Server::Outcome Server::generate(const Request& r, const std::function<void(const std::string&)>& emit) {
  Outcome o;
  const std::string prompt = r.chat ? deps_.tmpl.render(r.messages, r.tools, r.enable_thinking) : r.prompt;
  o.prompt_ids = deps_.tok.encode(prompt);
  const uint32_t max_len = deps_.engine.max_len();
  if (o.prompt_ids.empty()) throw BadRequest("prompt encodes to zero tokens");
  if (o.prompt_ids.size() + 1 > max_len) throw BadRequest("prompt is " + std::to_string(o.prompt_ids.size()) + " tokens; max_model_len is " + std::to_string(max_len));
  const uint32_t budget = uint32_t(max_len - o.prompt_ids.size());
  const uint32_t max_tokens = std::min(r.max_tokens.value_or(budget), budget);
  o.usage.prompt_tokens = uint32_t(o.prompt_ids.size());
  deps_.engine.reset();
  deps_.engine.prefill(o.prompt_ids);
  std::unique_ptr<StreamerIface> st = deps_.tok.streamer();
  // Stop strings: hold back up to (longest stop − 1) bytes so a stop split across
  // two tokens is never emitted.
  size_t hold = 0; for (const auto& s : r.stop) hold = std::max(hold, s.size() > 0 ? s.size() - 1 : 0);
  std::string pending;   // decoded text not yet emitted (because of `hold`)
  auto flush_pending = [&](bool final) {
    const size_t keep = final ? 0 : std::min(hold, pending.size());
    if (pending.size() > keep) { const std::string out = pending.substr(0, pending.size() - keep); o.text += out; emit(out); pending.erase(0, out.size()); }
  };
  o.finish_reason = "length";
  for (uint32_t i = 0; i < max_tokens; ++i) {
    const uint32_t id = deps_.engine.step(r.sampling);
    const bool is_eos = eos_.count(id) != 0;
    if (is_eos && !r.ignore_eos && o.out_ids.size() + 0 >= r.min_tokens) { o.finish_reason = "stop"; break; }
    o.out_ids.push_back(id);
    if (!is_eos) pending += st->push(id);            // EOS never becomes text, even under ignore_eos
    // stop strings, matched on the decoded text so far
    bool stopped = false;
    for (const auto& s : r.stop) {
      const size_t at = pending.find(s);
      if (at != std::string::npos) { pending.erase(at); stopped = true; break; }
    }
    if (stopped) { o.finish_reason = "stop"; flush_pending(true); break; }
    flush_pending(false);
  }
  if (o.finish_reason == "length") { pending += st->flush(); flush_pending(true); }
  o.usage.completion_tokens = uint32_t(o.out_ids.size());
  return o;
}
```

(`min_tokens`: while `out_ids.size() < min_tokens`, an EOS is treated as ignored: not a stop, counted, not emitted. The condition above expresses exactly that.) `eos_` is a `std::set<uint32_t>` in `Options` (add `std::vector<uint32_t> eos_ids` to `Options`; 7d fills it from `generation_config.json` - `[248046, 248044]`; the mock test sets `{248046}`).

Routes:

```cpp
svr_.Post("/v1/chat/completions", [this](const httplib::Request& q, httplib::Response& s) { handle(q, s, true); });
svr_.Post("/v1/completions",      [this](const httplib::Request& q, httplib::Response& s) { handle(q, s, false); });
svr_.Get ("/v1/models", [this](const httplib::Request&, httplib::Response& s) { s.set_content(models_body(opts_.served_model, created_), "application/json"); });
```

`handle`: parse (BadRequest → 400 with `error_body(msg, "invalid_request_error")`); `acquire` (false → 503 `error_body("server busy: N requests queued", "server_overloaded")`); non-stream → `generate` with an `emit` that appends, then `completion_body`; stream →

```cpp
s.set_header("Cache-Control", "no-cache");
s.set_header("X-Accel-Buffering", "no");
s.set_chunked_content_provider("text/event-stream",
  [this, r, ticket](size_t /*offset*/, httplib::DataSink& sink) mutable {
    const std::string id = next_id(r.chat); const uint64_t created = now();
    auto write = [&](const std::string& f) { return sink.write(f.data(), f.size()); };
    if (r.chat) write(stream_frame_role(r, id, created));
    Outcome o;
    try {
      o = generate(r, [&](const std::string& piece) { if (!piece.empty()) write(stream_frame_text(r, id, created, piece)); });
    } catch (const BadRequest& e) { write(std::string("data: ") + error_body(e.what(), "invalid_request_error") + "\n\n"); write(kDone); sink.done(); release(); return true; }
    write(stream_frame_finish(r, id, created, o.finish_reason));
    if (r.include_usage) write(stream_frame_usage(r, id, created, o.usage));
    write(kDone);
    sink.done();
    release();
    return true;
  });
```

`sink.write` hands the chunk to the socket immediately (httplib writes each chunk as it is produced; with `svr_.set_tcp_nodelay(true)` there is no Nagle delay). The provider runs on httplib's worker thread; the engine is serialised by the ticket, so a second request's provider blocks in `acquire` *before* it is admitted (503 beyond `queue_depth`). `acquire`/`release`: `std::mutex`, `std::condition_variable`, `next_ticket_`, `serving_`, `waiting_` - a waiting request increments `waiting_` under the lock and is refused if `waiting_ >= queue_depth`.

- [ ] **Step 3:** `start()`: `svr_.bind_to_port(host, port)` (or `bind_to_any_port` when `port == 0`, storing the result), then `thread_ = std::thread([this]{ svr_.listen_after_bind(); })`; `stop()`: `svr_.stop(); thread_.join()`. `listen()` = bind + `listen_after_bind()` on the caller's thread.
- [ ] **Step 4:** Build; run Task 5; commit - `git commit -m "feat(server): Server - routes, bounded FIFO, generation core, unbuffered SSE"`.

---

### Task 5: The protocol test (mock engine, no GPU)

**Files:** Create `tests/server/mock.h`, `tests/server/protocol_test.cc`; register (`add_executable(protocol_test server/protocol_test.cc)`, links `b70_server`, `add_test`).

- [ ] **Step 1: Mocks**

```cpp
// tests/server/mock.h
struct MockTok : server::TokIface {
  // word-level "tokenizer": ids are 1000 + index of first appearance; "<|im_end|>" is 248046.
  std::vector<std::string> vocab; std::map<std::string, uint32_t> index;
  uint32_t id_of(const std::string& w) { if (w == "<|im_end|>") return 248046; auto it = index.find(w); if (it != index.end()) return it->second; const uint32_t id = 1000 + uint32_t(vocab.size()); vocab.push_back(w); index[w] = id; return id; }
  std::vector<uint32_t> encode(std::string_view text) override { std::vector<uint32_t> out; std::string w; for (char c : text) { if (c == ' ' || c == '\n') { if (!w.empty()) out.push_back(id_of(w)); w.clear(); if (c == '\n') out.push_back(id_of("\n")); } else w += c; } if (!w.empty()) out.push_back(id_of(w)); return out; }
  std::string piece(uint32_t id) { if (id == 248046) return "<|im_end|>"; const std::string& w = vocab.at(id - 1000); return w == "\n" ? "\n" : w + " "; }
  std::string decode(const std::vector<uint32_t>& ids) override { std::string s; for (uint32_t i : ids) s += piece(i); return s; }
  struct S : server::StreamerIface { MockTok& t; explicit S(MockTok& t) : t(t) {} std::string push(uint32_t id) override { return t.piece(id); } std::string flush() override { return ""; } };
  std::unique_ptr<server::StreamerIface> streamer() override { return std::make_unique<S>(*this); }
  uint32_t vocab_used() override { return 248077; }
};
struct MockTemplate : server::TemplateIface {
  std::string render(const nlohmann::json& m, const nlohmann::json&, bool think) override {
    std::string s; for (const auto& x : m) s += x.at("role").get<std::string>() + ": " + x.at("content").get<std::string>() + "\n";
    return s + (think ? "assistant(think): " : "assistant: ");
  }
};
struct MockEngine : server::EngineIface {
  std::vector<uint32_t> script; size_t at = 0; int step_ms = 0;
  std::vector<uint32_t> last_prompt; std::vector<std::chrono::steady_clock::time_point> step_times;
  std::atomic<int> active{0}; std::atomic<int> max_active{0};
  void reset() override { at = 0; }
  void prefill(const std::vector<uint32_t>& ids) override { last_prompt = ids; }
  uint32_t step(const server::Sampling&) override {
    const int a = ++active; max_active = std::max(max_active.load(), a);
    if (step_ms) std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    step_times.push_back(std::chrono::steady_clock::now());
    const uint32_t id = at < script.size() ? script[at++] : 248046;
    --active; return id;
  }
  uint32_t max_len() override { return 1024; }
  uint32_t pos() override { return uint32_t(last_prompt.size() + at); }
};
```

- [ ] **Step 2: Cases** (each a function; `main` runs all; `httplib::Client c("127.0.0.1", port)`):
  1. `GET /v1/models` → 200, `data[0].id == "mock-model"`.
  2. Non-stream chat, script = ids of "Paris is the capital <|im_end|>": 200, `choices[0].message.content == "Paris is the capital "`, `finish_reason == "stop"`, `usage.prompt_tokens == MockTok.encode(MockTemplate.render(...)).size()`, `completion_tokens == 4`; the engine's `last_prompt` equals that encode.
  3. `max_tokens: 2` → content "Paris is ", `finish_reason == "length"`.
  4. Stream chat with `stream_options.include_usage`: collect the body via `c.Post(..., content_receiver)`; split on `\n\n`; frame 0 has `delta.role == "assistant"`; frames 1..4 have `delta.content` = the four pieces; then a frame with `finish_reason == "stop"` and empty delta; then a usage frame with `choices == []`; then `data: [DONE]`.
  5. **Unbuffered first frame** (spec §7 risk 4): `step_ms = 50`, script of 6 ids: record the arrival time of each chunk in the receiver; assert first content frame arrived < 120 ms after the request started (prefill is instant in the mock) and the last frame ≥ 250 ms after the first. Also assert, from `step_times`, that frame *k*'s arrival precedes step *k+1*'s completion - i.e. no frame waited for the next token.
  6. Stop strings: script "alpha beta STOP gamma", `stop: ["STOP"]` → content "alpha beta ", `finish_reason == "stop"`, no "STOP" anywhere in the stream; and a split-stop case: script pieces "ST", "OP" cannot occur in the word mock - instead test `stop: ["beta STOP"]` spanning two tokens → content "alpha ".
  7. `min_tokens: 3` with script "x <|im_end|> y z <|im_end|>" → 3 completion tokens, content "x y z " (EOS at position 2 ignored, not emitted); `ignore_eos: true, max_tokens: 4` → 4 tokens, content "x y z " (the second EOS counted, not emitted), `finish_reason == "length"`.
  8. `/v1/completions` with `prompt: "hello world"`, `return_token_ids: true` → no template (engine `last_prompt == encode("hello world")`), `prompt_token_ids` equals it, `choices[0].token_ids` equals the script prefix before EOS, `object == "text_completion"`.
  9. Errors: body `"{"` → 400 with `error.type == "invalid_request_error"`; `n: 2` → 400; a prompt of 1030 words (> max_len 1024) → 400 mentioning `max_model_len`; content array → 400.
  10. FIFO + 503: `step_ms = 30`, script of 20 ids, `queue_depth = 2`; fire 6 concurrent requests from 6 threads; expect ≥ 1 and ≤ 4 responses 503, the rest 200 with full content; `MockEngine.max_active == 1` (serialised).
  11. Unknown fields (`cache_prompt`, `return_token_ids` on a chat request, `logprobs: null`) → 200.
- [ ] **Step 3: Run** - `tools/box.sh test "openai_test|protocol_test"`. Expected: both OK; print the measured first-frame latency and the last-frame gap from case 5.
- [ ] **Step 4: Full suite** - `tools/box.sh test`: everything previously green plus these two; decode invariants unchanged (nothing under `src/runtime` or `src/kernels` was touched - `git diff --stat` proves it).
- [ ] **Step 5: Commit** - `git commit -m "test(server): protocol_test - 11 cases on a mock engine incl. unbuffered first frame and 503 FIFO"`.

---

### Task 6: Docs

- [ ] docs/04 "Server" gains a "Status (spec 3 T3)" paragraph: what exists, the three interfaces, the FIFO/503 rule, the deviation "absent temperature = greedy", the extension fields. Commit.

---

## Self-review

**Spec coverage:** endpoints, fields, response shapes, single stream + FIFO + 503, stop/EOS/min_tokens/max_tokens, errors → Tasks 3-5; unbuffered first token → Task 4 (chunk-per-token, `tcp_nodelay`) and Task 5 case 5; §3.5 parsing → Task 3 (`Sampling`), execution → plan 7d; vendoring + VERSIONS → Task 1; mock-tested without T1/T2/GPU → Task 5. **Placeholders:** `<tag>` is pin-time output. **Types:** `Sampling`, `EngineIface::step(const Sampling&)`, `Request`, `Outcome` used consistently across Tasks 2-5; plan 7d's adapters implement exactly `deps.h`.

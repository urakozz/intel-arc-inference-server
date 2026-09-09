# Spec 3 / T2 - Chat template (minja) and streaming detokeniser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `chat::Template` renders the checkpoint's `chat_template.jinja` byte-for-byte like `transformers.apply_chat_template`, and `tok::Streamer` turns a stream of ids into UTF-8-complete text with no U+FFFD for valid input.

**Architecture:** Two vendored single headers (`google/minja`, `nlohmann/json`) under `third_party/` with a `VERSIONS` file; `chat::Template` in `src/tokenizer/` (still a leaf: it reads three files from the snapshot and depends only on the two headers); `tok::Streamer` over plan 7a's `tok::Tokenizer` using the hold-and-flush algorithm docs/11 quotes from `openvino.genai`. Parity vectors are dumped once in the box venv (`transformers 5.14.1`).

**Tech Stack:** C++17, minja (header-only Jinja subset), nlohmann/json, `tests/check.h`; Python (`tools/` only) for the dumps.

**Spec:** `docs/superpowers/specs/2026-09-04-spec3-tokenizer-http-design.md` §3.2, bar 1 (streaming half), bar 2, §5, §7 risk 1.

## Global Constraints

- Branch `spec1.7-codex-exp`; no push; commits end `Claude-Session: `.
- The Mac never compiles; `tools/box.sh` only (JOBS 44); scp not heredocs. Dumps run in `~/auto-round/.venv/bin/python` on the box (`transformers 5.14.1`, `jinja2 3.1.6`, measured 2026-09-06), never in docker.
- Never docker, never kill a process you did not start, never delete on the box; stop on disk errors.
- `src/tokenizer/` stays a leaf. Python in `tools/` only. `-Wall -Wextra -Werror` - vendored headers are included with `-isystem` (`target_include_directories(... SYSTEM ...)`) so their warnings are not ours.
- Pinned versions in `third_party/VERSIONS`: one line per header, `name  url  tag-or-commit  sha256`. The values are execution outputs (the latest release tag at pin time), recorded in that file and in the commit message.
- Snapshot files: `/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64/{chat_template.jinja,tokenizer_config.json,generation_config.json}`; `B70_SNAPSHOT_DIR` env overrides in tests (default that path).
- Requires plan 7a landed (Task 4's `tok::Tokenizer`, Task 5's corpus) for Tasks 4-5 only; Tasks 1-3 build without it.

---

## File structure

| path | responsibility |
|---|---|
| `third_party/VERSIONS`, `third_party/minja/{minja.hpp,chat-template.hpp}`, `third_party/json/json.hpp` | vendored, pinned, `-isystem` |
| `src/tokenizer/chat_template.h`, `chat_template.cc` | `chat::Template` |
| `src/tokenizer/streamer.h`, `streamer.cc` | `tok::Streamer` |
| `tools/tokenizer/dump_template.py` | reference renders |
| `tests/tokenizer/template_messages.json`, `template_think_on.txt`, `template_think_off.txt`, `template_tools.txt` | committed parity vectors |
| `tests/tokenizer/template_test.cc`, `streamer_test.cc` | the tests |

---

### Task 1: Vendor minja and nlohmann/json

**Files:** Create `third_party/VERSIONS`, `third_party/minja/minja.hpp`, `third_party/minja/chat-template.hpp`, `third_party/json/json.hpp`, `third_party/CMakeLists.txt`; modify `CMakeLists.txt` (`add_subdirectory(third_party)` before `src/common`).

**Interfaces:** Produces INTERFACE targets `b70_third_party_json` and `b70_third_party_minja` (the latter links the former).

- [ ] **Step 1: Pin** - on the Mac (it has network): `gh api repos/nlohmann/json/releases/latest --jq .tag_name` and `gh api repos/google/minja/commits/main --jq .sha`. Record both.
- [ ] **Step 2: Fetch** - `curl -sSL -o third_party/json/json.hpp https://github.com/nlohmann/json/releases/download/<tag>/json.hpp` (the single-header release asset); `curl -sSL -o third_party/minja/minja.hpp https://raw.githubusercontent.com/google/minja/<sha>/include/minja/minja.hpp` and the same for `chat-template.hpp`. `shasum -a 256` each; write `VERSIONS`:

```
# name        source                                   pin                 sha256
json          https://github.com/nlohmann/json        <tag>               <sha256>
minja         https://github.com/google/minja         <commit sha>        <sha256 of minja.hpp> <sha256 of chat-template.hpp>
```

- [ ] **Step 3: `third_party/CMakeLists.txt`**

```cmake
# Vendored single headers (third_party/VERSIONS is the pin record). SYSTEM so
# -Werror applies to our code, not theirs.
add_library(b70_third_party_json INTERFACE)
target_include_directories(b70_third_party_json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/json)
add_library(b70_third_party_minja INTERFACE)
target_include_directories(b70_third_party_minja SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/minja)
target_link_libraries(b70_third_party_minja INTERFACE b70_third_party_json)
```

minja includes `<nlohmann/json.hpp>`; put json.hpp at `third_party/json/nlohmann/json.hpp` so that include resolves (adjust the path above accordingly and say so in VERSIONS).

- [ ] **Step 4: Compile check** - `tools/box.sh build`. Expected: unchanged build (nothing links the targets yet).
- [ ] **Step 5: Commit** - `git add third_party CMakeLists.txt && git commit -m "build(third_party): vendor nlohmann/json <tag> and google/minja <sha> (pinned in VERSIONS)"`.

---

### Task 2: `chat::Template`

**Files:** Create `src/tokenizer/chat_template.h`, `src/tokenizer/chat_template.cc`; modify `src/tokenizer/CMakeLists.txt`.

**Interfaces:**

```cpp
namespace chat {
class Template {
 public:
  // Reads <snapshot>/chat_template.jinja, tokenizer_config.json (bos/eos strings;
  // bos is null for this checkpoint) - throws std::runtime_error on any missing file.
  explicit Template(const std::string& snapshot_dir);
  // messages: the OpenAI `messages` array; tools: the `tools` array or null;
  // enable_thinking: the template's toggle; add_generation_prompt is always true
  // (this is the serving path). Returns the prompt TEXT - encode it with
  // tok::Tokenizer::encode(text, /*add_special=*/false).
  std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                     bool enable_thinking) const;
  const std::string& eos_token() const;   // "<|im_end|>"
};
}
```

- [ ] **Step 1: Failing test first** - write Task 3's `template_test.cc` skeleton with a single case that constructs `chat::Template` and calls `render` on one user message, checking the result starts with `<|im_start|>` and ends with `<|im_start|>assistant\n` (with or without a `<think>` block - accept either; the parity vectors settle it in Task 3). Register it (`add_executable(template_test …)`, links `b70_tokenizer`, `add_test`). `tools/box.sh test template_test` → expected: compile error, no such class.
- [ ] **Step 2: Implement**

```cpp
// src/tokenizer/chat_template.cc
#include "tokenizer/chat_template.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <minja/chat-template.hpp>
namespace chat {
namespace {
std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("chat::Template: cannot read " + p);
  std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
std::string token_string(const nlohmann::json& cfg, const char* key) {
  if (!cfg.contains(key) || cfg[key].is_null()) return "";
  const auto& v = cfg[key];
  return v.is_string() ? v.get<std::string>() : v.value("content", "");   // AddedToken form
}
}  // namespace
struct Template::Impl {
  std::unique_ptr<minja::chat_template> tmpl;
  std::string bos, eos;
};
Template::Template(const std::string& dir) : impl_(new Impl) {
  const nlohmann::json cfg = nlohmann::json::parse(slurp(dir + "/tokenizer_config.json"));
  impl_->bos = token_string(cfg, "bos_token");
  impl_->eos = token_string(cfg, "eos_token");
  impl_->tmpl = std::make_unique<minja::chat_template>(slurp(dir + "/chat_template.jinja"), impl_->bos, impl_->eos);
}
Template::~Template() = default;
std::string Template::render(const nlohmann::json& messages, const nlohmann::json& tools, bool enable_thinking) const {
  minja::chat_template_inputs in;
  in.messages = messages;
  in.tools = tools.is_null() ? nlohmann::json() : tools;
  in.add_generation_prompt = true;
  in.extra_context = nlohmann::json{{"enable_thinking", enable_thinking}};
  minja::chat_template_options opts;   // defaults: no polyfills forced
  return impl_->tmpl->apply(in, opts);
}
const std::string& Template::eos_token() const { return impl_->eos; }
}  // namespace chat
```

`minja::chat_template`, `chat_template_inputs` (`messages`, `tools`, `add_generation_prompt`, `extra_context`, `now`) and `chat_template_options` are the names in `chat-template.hpp` at every 2025-2026 commit; **open the vendored header and match its signatures exactly** (a field renamed upstream is a one-line fix here, recorded in the commit). The header declares `Template` with `std::unique_ptr<Impl> impl_` and `~Template()`.

- [ ] **Step 3: CMake** - `target_sources(b70_tokenizer PRIVATE chat_template.cc)`, `target_link_libraries(b70_tokenizer PUBLIC b70_third_party_minja)`.
- [ ] **Step 4: Run** - `tools/box.sh test template_test`. Expected: the skeleton case passes.
- [ ] **Step 5: Commit** - `git add src/tokenizer tests/tokenizer/template_test.cc tests/CMakeLists.txt && git commit -m "feat(tokenizer): chat::Template over minja (bos/eos from tokenizer_config)"`.

---

### Task 3: Template parity vectors and the parity test (bar 2)

**Files:** Create `tools/tokenizer/dump_template.py`; generated+committed `tests/tokenizer/template_messages.json`, `template_tools.json`, `template_think_on.txt`, `template_think_off.txt`, `template_tools.txt`; complete `tests/tokenizer/template_test.cc`.

- [ ] **Step 1: The conversation** - `template_messages.json`:

```json
[
  {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
  {"role": "user", "content": "What is the capital of France? 请用中文回答。"},
  {"role": "assistant", "content": "法国的首都是巴黎。"},
  {"role": "user", "content": "And Germany's? Reply with just the city - no punctuation."}
]
```

`template_tools.json`: one function tool (`get_weather(location: string)`) in OpenAI `tools` shape.

- [ ] **Step 2: `dump_template.py`** (box venv):

```python
#!/usr/bin/env python3
"""Reference renders of chat_template.jinja for tests/tokenizer/template_test.
Run ON THE BOX: ~/auto-round/.venv/bin/python tools/tokenizer/dump_template.py <snapshot_dir>"""
import json, sys, transformers
from transformers import AutoTokenizer
snap = sys.argv[1]
tok = AutoTokenizer.from_pretrained(snap)
msgs = json.load(open("tests/tokenizer/template_messages.json"))
tools = json.load(open("tests/tokenizer/template_tools.json"))
def dump(name, **kw):
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    open(f"tests/tokenizer/{name}", "w", encoding="utf-8", newline="").write(s)
    print(f"{name}: {len(s)} bytes")
dump("template_think_on.txt", enable_thinking=True)
dump("template_think_off.txt", enable_thinking=False)
dump("template_tools.txt", tools=tools, enable_thinking=False)
print("transformers", transformers.__version__)
```

- [ ] **Step 3: Dump** - `tools/box.sh sync && tools/box.sh run '~/auto-round/.venv/bin/python tools/tokenizer/dump_template.py /home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64'` then `tools/box.sh pull tests/tokenizer/template_think_on.txt` (and the other two). Record the byte counts and the transformers version in the commit message.
- [ ] **Step 4: The test**

```cpp
// tests/tokenizer/template_test.cc - spec 3 §2 bar 2: byte-for-byte vs transformers.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include "check.h"
#include "tokenizer/chat_template.h"
#include <nlohmann/json.hpp>
static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); CHECK(f.good()); std::stringstream s; s << f.rdbuf(); return s.str(); }
static std::string snapshot() { const char* e = std::getenv("B70_SNAPSHOT_DIR"); return e ? e : "/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64"; }
static void diff_report(const char* name, const std::string& got, const std::string& want) {
  size_t i = 0; while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  std::fprintf(stderr, "%s: differs at byte %zu of %zu/%zu\n  got : %s\n  want: %s\n", name, i, got.size(), want.size(),
               got.substr(i, 80).c_str(), want.substr(i, 80).c_str());
}
int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  chat::Template t(snapshot());
  CHECK_EQ(t.eos_token(), std::string("<|im_end|>"));
  const auto msgs = nlohmann::json::parse(slurp(dir + "/template_messages.json"));
  const auto tools = nlohmann::json::parse(slurp(dir + "/template_tools.json"));
  struct Case { const char* file; nlohmann::json tools; bool think; } cases[] = {
    {"template_think_on.txt", nullptr, true}, {"template_think_off.txt", nullptr, false}, {"template_tools.txt", tools, false}};
  int bad = 0;
  for (const Case& c : cases) {
    const std::string want = slurp(dir + "/" + c.file), got = t.render(msgs, c.tools, c.think);
    if (got != want) { diff_report(c.file, got, want); ++bad; } else std::printf("%s: %zu bytes, identical\n", c.file, got.size());
  }
  CHECK_EQ(bad, 0);
  std::printf("template_test OK\n");
  return 0;
}
```

Register with `add_test(NAME template_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)`.

- [ ] **Step 5: Run** - `tools/box.sh test template_test`. Expected: three `identical` lines. **If a case differs:** the diff report names the byte. Two known classes: (a) transformers' own normalisation (it strips a trailing newline from `chat_template.jinja` files in some versions - compare the raw file's last bytes with what the reference emits and, if that is the whole difference, apply the same strip in `Template::Template` with a comment naming the transformers version); (b) a Jinja construct minja lacks - the spec's recorded fallback: write `src/tokenizer/chat_template_fallback.jinja` (the same template with the construct rewritten), load it when the snapshot's template sha256 matches the checkpoint's, and record the construct in docs/11. The tools case may ALSO be short for reason (b) alone; if the two think cases are identical and only tools differs, that is bar 2 met and the tools fallback is a named follow-on.
- [ ] **Step 6: Commit** - `git add tools/tokenizer/dump_template.py tests/tokenizer && git commit -m "test(tokenizer): template parity vectors (transformers 5.14.1) + template_test"`.

---

### Task 4: `tok::Streamer`

**Files:** Create `src/tokenizer/streamer.h`, `streamer.cc`; modify `src/tokenizer/CMakeLists.txt`.

**Interfaces:**

```cpp
namespace tok {
// Incremental detokeniser (docs/11 "Streaming detokenisation", the openvino.genai
// algorithm): push(id) returns the text that became safe to emit (often "").
class Streamer {
 public:
  explicit Streamer(const Tokenizer& t);
  std::string push(uint32_t id);
  // Emits whatever is held (called at end of generation; a held U+FFFD-terminated
  // run is emitted as-is - the input was genuinely truncated mid-character).
  std::string flush();
 private:
  const Tokenizer& tok_;
  std::vector<uint32_t> pending_;
  size_t printed_ = 0;
};
}
```

- [ ] **Step 1: Failing test** - Task 5's `streamer_test.cc` (below); register; run → compile error.
- [ ] **Step 2: Implement**

```cpp
// src/tokenizer/streamer.cc
#include "tokenizer/streamer.h"
namespace tok {
Streamer::Streamer(const Tokenizer& t) : tok_(t) {}
std::string Streamer::push(uint32_t id) {
  pending_.push_back(id);
  const std::string text = tok_.decode(pending_, /*skip_special=*/false);
  // An incomplete multi-byte sequence decodes to U+FFFD at the end: hold it.
  if (text.size() >= 3 && text.compare(text.size() - 3, 3, "\xEF\xBF\xBD") == 0) return "";
  std::string out;
  if (text.size() > printed_) out = text.substr(printed_);
  printed_ = text.size();
  // A newline bounds the held run: after it, nothing earlier can change.
  if (text.find('\n') != std::string::npos) { pending_.clear(); printed_ = 0; }
  return out;
}
std::string Streamer::flush() {
  const std::string text = pending_.empty() ? std::string() : tok_.decode(pending_, false);
  std::string out = text.size() > printed_ ? text.substr(printed_) : std::string();
  pending_.clear(); printed_ = 0;
  return out;
}
}  // namespace tok
```

Note the newline rule: openvino resets on newline so the run stays bounded; because `printed_` is reset with `pending_`, the suffix logic stays consistent. (If the `\n` is the last byte of `text`, everything up to it has already been emitted in `out`.)

- [ ] **Step 3: CMake** - `target_sources(b70_tokenizer PRIVATE streamer.cc)`.
- [ ] **Step 4-5:** run Task 5's test, commit - `git commit -m "feat(tokenizer): tok::Streamer - hold-and-flush incremental detokeniser"`.

---

### Task 5: Streamer parity test (bar 1, streaming half)

**Files:** Create `tests/tokenizer/streamer_test.cc`; register (`add_test(NAME streamer_test COMMAND streamer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)`).

- [ ] **Step 1: The test**

```cpp
// tests/tokenizer/streamer_test.cc - bar 1: one id at a time == one-shot decode, no U+FFFD.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "check.h"
#include "tokenizer/streamer.h"
#include "tokenizer/tokenizer.h"
static std::vector<std::string> lines(const std::string& p) { /* as parity_test.cc */ }
int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  tok::Tokenizer t(tok::default_tokenizer_json());
  const auto ref = lines(dir + "/corpus.ids");
  size_t cases = 0, bad_concat = 0, bad_fffd = 0, held_max = 0;
  for (const std::string& l : ref) {
    std::vector<uint32_t> ids; std::istringstream is(l); uint32_t v; while (is >> v) ids.push_back(v);
    if (ids.empty()) continue;
    ++cases;
    const std::string once = t.decode(ids);
    tok::Streamer s(t);
    std::string cat; size_t held = 0, cur_held = 0;
    for (uint32_t id : ids) { const std::string o = s.push(id); if (o.empty()) { ++cur_held; held = std::max(held, cur_held); } else cur_held = 0; cat += o; }
    cat += s.flush();
    held_max = std::max(held_max, held);
    if (cat != once) { if (bad_concat < 5) std::fprintf(stderr, "concat mismatch: %s\n", once.c_str()); ++bad_concat; }
    if (cat.find("\xEF\xBF\xBD") != std::string::npos && once.find("\xEF\xBF\xBD") == std::string::npos) ++bad_fffd;
  }
  std::printf("streamer_test: %zu cases, %zu concat mismatches, %zu spurious U+FFFD, longest hold %zu ids\n",
              cases, bad_concat, bad_fffd, held_max);
  CHECK_EQ(bad_concat, size_t(0));
  CHECK_EQ(bad_fffd, size_t(0));
  // A CJK/emoji byte-split MUST occur somewhere in 10k lines - proves the hold path ran.
  CHECK(held_max >= 1);
  std::printf("streamer_test OK\n");
  return 0;
}
```

- [ ] **Step 2: Run** - `tools/box.sh test streamer_test`. Expected: 0 / 0, longest hold ≥ 1 (measured; record it). Then the full suite: `tools/box.sh test` - 60 + 7a's 2 + this plan's 2 pass, 1 designed skip.
- [ ] **Step 3: Commit** - `git commit -am "test(tokenizer): streamer_test - incremental == one-shot on the parity corpus"`.

---

### Task 6: Docs

- [ ] docs/11: "Status (spec 3 T2)" - minja pin, the three parity cases and their byte counts, any fallback recorded, the streamer's measured longest hold. docs/04's component table gets `third_party/` if it lists directories. Commit.

---

## Self-review

**Spec coverage:** §3.2 minja + fallback → Tasks 1-3; streamer algorithm + `push(id)` → Task 4; bar 2 (3-turn, `enable_thinking` both ways) → Task 3; bar 1 streaming half → Task 5; pinned `VERSIONS` → Task 1. Tools rendering is covered as a third parity case with the recorded-fallback path. **Placeholders:** `<tag>`, `<sha>`, `<sha256>` are pin-time outputs recorded in VERSIONS. **Types:** `chat::Template::render(const json&, const json&, bool)` and `tok::Streamer::push/flush` are used with the same signatures in Tasks 2-5 and consumed unchanged by plan 7d.

# Spec 3 / T1 - Tokenizer (HF `tokenizers` behind a C FFI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/tokenizer/` - a leaf library whose `encode`/`decode` are byte-identical to Python `tokenizers` 0.22.2 on the checkpoint's `tokenizer.json`, proven by a committed parity corpus.

**Architecture:** One Rust crate (`src/tokenizer/rs/`) wrapping the `tokenizers` crate behind six `extern "C"` functions, built by Cargo from CMake into a static archive; one C++ wrapper class `tok::Tokenizer`; one parity test driven by a corpus generated deterministically on the Mac and tokenised once by Python `tokenizers` in the box's existing venv.

**Tech Stack:** Rust (rustup, user-local on the box; crate `tokenizers` pinned to the venv's `0.22.2`), CMake 4.2 `add_custom_command`, C++17 `-Wall -Wextra -Werror`, `tests/check.h`.

**Spec:** `docs/superpowers/specs/2026-09-04-spec3-tokenizer-http-design.md` §3.1, §2 bar 1, §5 (with the 2026-09-06 amendment to §3.4).

## Global Constraints

- Branch `spec1.7-codex-exp`; no push; commits end `Claude-Session: `.
- The Mac never compiles. Build/test ONLY via `tools/box.sh` (JOBS 44). Box `user@box`; scp files, never heredocs over ssh. `~/auto-round/.venv/bin/python` on the box has `tokenizers 0.22.2`, `transformers 5.14.1` (measured 2026-09-06) - the parity dumps run THERE, never in docker.
- Never docker, never kill a process you did not start, never delete anything on the box, never restart the operator's server; stop on any disk error (84 GB free).
- `src/tokenizer/` depends on nothing else in `src/` (spec §5, docs/04 component rule). Python only under `tools/`.
- `-Wall -Wextra -Werror`; decode's 774 kernels / 19 modules and every existing test untouched - this stream adds files and one `add_subdirectory`.
- Tokenizer file used by every test: `/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64/tokenizer.json` (RTN snapshot; `tokenizer_config.json` beside it). Tests take the path from `B70_TOKENIZER_JSON` (env) and default to that path.
- Every number labelled measured / derived / estimated.

---

## File structure

| path | responsibility |
|---|---|
| `src/tokenizer/rs/Cargo.toml`, `Cargo.lock`, `src/lib.rs` | the crate: `tokenizers` + the C ABI, nothing else |
| `src/tokenizer/b70_tok.h` | the C ABI header (shared by Rust's exports and the C++ wrapper) |
| `src/tokenizer/tokenizer.h`, `tokenizer.cc` | `tok::Tokenizer` - RAII over the handle, `std::string`/`std::vector` in and out |
| `cmake/tokenizer.cmake` | finds cargo, runs it, defines `b70_tok_rs` (imported static) and `b70_tokenizer` (C++ static lib) |
| `tools/tokenizer/make_corpus.py` | deterministic corpus generator (no third-party imports) |
| `tools/tokenizer/dump_parity.py` | Python `tokenizers` → `corpus.ids` (+ its own roundtrip check) |
| `tests/tokenizer/corpus.txt`, `corpus.ids` | the committed parity vectors |
| `tests/tokenizer/tokenizer_smoke_test.cc`, `parity_test.cc` | the tests |
| `docs/10-the-box.md`, `docs/09-build-notes.md` (or wherever build notes live - `grep -l "ccache" docs/*.md`), `docs/11-tokenizer-and-chat-template.md` | one paragraph each |

---

### Task 1: Rust toolchain on the box, user-local

**Files:** Modify `docs/10-the-box.md` (append one paragraph).

**Interfaces:** Produces `~/.cargo/bin/cargo` on the box (1.9x stable, `--profile minimal`).

- [ ] **Step 1: Prove the box has no Rust and can reach the installer** - `tools/box.sh run 'ls ~/.cargo/bin 2>&1; curl -sI --max-time 5 https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init | head -1'`. Expected (measured 2026-09-06): `No such file or directory` and `HTTP/2 200`.
- [ ] **Step 2: Install** - `tools/box.sh run 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path --default-toolchain stable'`. No sudo, nothing outside `$HOME`.
- [ ] **Step 3: Verify** - `tools/box.sh run '~/.cargo/bin/cargo --version; ~/.cargo/bin/rustc --version; ~/.cargo/bin/rustc --print sysroot'`. Record all three lines.
- [ ] **Step 4: Document** - append to `docs/10-the-box.md`:

```markdown
## Rust (spec 3, T1 - 2026-09-06)
User-local `rustup` (`--profile minimal`, `--no-modify-path`): `~/.cargo/bin/{cargo,rustc}` = <the three lines from Step 3>. Nothing on PATH was changed; `cmake/tokenizer.cmake` looks in `~/.cargo/bin` first. Installed for the `tokenizers` crate (docs/11); it builds `oniguruma` from source through `cc`, so it needs `/usr/bin/cc` (gcc 15.2, present).
```

- [ ] **Step 5: Commit** - `git add docs/10-the-box.md && git commit -m "docs(box): user-local Rust toolchain for the tokenizer crate"`.

---

### Task 2: The crate and its C ABI

**Files:**
- Create: `src/tokenizer/rs/Cargo.toml`, `src/tokenizer/rs/src/lib.rs`, `src/tokenizer/b70_tok.h`
- Create (generated, committed): `src/tokenizer/rs/Cargo.lock`

**Interfaces:**
- Produces the C ABI below, consumed by Task 4's wrapper. Every string/array crossing the boundary is allocated by Rust and freed by `b70_tok_free_buf`; every function returns 0 on success or a negative code, with `b70_tok_last_error` holding a message.

- [ ] **Step 1: `Cargo.toml`** - pin the crate to the venv's Python package version so parity is by construction (both are the same Rust code):

```toml
[package]
name = "b70_tok"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
crate-type = ["staticlib"]

[dependencies]
# = the box venv's `tokenizers` Python package version (0.22.2, measured
# 2026-09-06). Default features keep `onig` - the regex engine the Python
# wheels use - so the pre-tokenizer runs the same code path as the dumps.
tokenizers = "=0.22.2"

[profile.release]
lto = "thin"
codegen-units = 1
panic = "abort"
```

- [ ] **Step 2: `b70_tok.h`**

```c
// src/tokenizer/b70_tok.h - the C ABI of the Rust crate. Allocation rule: every
// out-pointer is Rust-allocated and released ONLY by b70_tok_free_buf. Every
// function returns 0 on success, <0 on error; b70_tok_last_error() describes
// the most recent error on this thread.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct b70_tok b70_tok;
int  b70_tok_new(const char* tokenizer_json_path, b70_tok** out);
void b70_tok_free(b70_tok* t);
// vocab INCLUDING added tokens (248,077 for the checkpoint - docs/11).
uint32_t b70_tok_vocab_size(const b70_tok* t);
// text: UTF-8, `len` bytes (no NUL needed). add_special=0 is what the golden
// prompts and the chat path use (the template already wrote <|im_start|>...).
int  b70_tok_encode(const b70_tok* t, const char* text, size_t len, int add_special,
                    uint32_t** out_ids, size_t* out_n);
int  b70_tok_decode(const b70_tok* t, const uint32_t* ids, size_t n, int skip_special,
                    char** out_utf8, size_t* out_len);
// The piece for one id (e.g. "Ġthe" in byte-level form - for debugging only).
int  b70_tok_id_to_token(const b70_tok* t, uint32_t id, char** out, size_t* out_len);
// -1 if unknown.
int64_t b70_tok_token_to_id(const b70_tok* t, const char* token, size_t len);
void b70_tok_free_buf(void* p, size_t len_or_n, int is_ids);
const char* b70_tok_last_error(void);
#ifdef __cplusplus
}
#endif
```

- [ ] **Step 3: `lib.rs`**

```rust
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::slice;
use tokenizers::Tokenizer;

thread_local! { static LAST: RefCell<CString> = RefCell::new(CString::default()); }
fn set_err(e: impl std::fmt::Display) -> i32 {
    LAST.with(|l| *l.borrow_mut() = CString::new(e.to_string()).unwrap_or_default());
    -1
}

pub struct b70_tok(Tokenizer);

#[no_mangle]
pub extern "C" fn b70_tok_new(path: *const c_char, out: *mut *mut b70_tok) -> i32 {
    if path.is_null() || out.is_null() { return set_err("null argument"); }
    let p = unsafe { CStr::from_ptr(path) }.to_string_lossy().into_owned();
    match Tokenizer::from_file(&p) {
        Ok(t) => { unsafe { *out = Box::into_raw(Box::new(b70_tok(t))); } 0 }
        Err(e) => set_err(format!("{p}: {e}")),
    }
}
#[no_mangle]
pub extern "C" fn b70_tok_free(t: *mut b70_tok) { if !t.is_null() { unsafe { drop(Box::from_raw(t)); } } }
#[no_mangle]
pub extern "C" fn b70_tok_vocab_size(t: *const b70_tok) -> u32 { unsafe { &*t }.0.get_vocab_size(true) as u32 }

#[no_mangle]
pub extern "C" fn b70_tok_encode(t: *const b70_tok, text: *const c_char, len: usize, add_special: i32,
                                 out_ids: *mut *mut u32, out_n: *mut usize) -> i32 {
    let bytes = unsafe { slice::from_raw_parts(text as *const u8, len) };
    let s = match std::str::from_utf8(bytes) { Ok(s) => s, Err(e) => return set_err(e) };
    match unsafe { &*t }.0.encode(s, add_special != 0) {
        Ok(enc) => {
            let mut v: Vec<u32> = enc.get_ids().to_vec();
            v.shrink_to_fit();
            unsafe { *out_n = v.len(); *out_ids = v.as_mut_ptr(); }
            std::mem::forget(v);
            0
        }
        Err(e) => set_err(e),
    }
}
#[no_mangle]
pub extern "C" fn b70_tok_decode(t: *const b70_tok, ids: *const u32, n: usize, skip_special: i32,
                                 out: *mut *mut c_char, out_len: *mut usize) -> i32 {
    let ids = unsafe { slice::from_raw_parts(ids, n) };
    match unsafe { &*t }.0.decode(ids, skip_special != 0) {
        Ok(s) => { let mut b = s.into_bytes(); b.shrink_to_fit();
                   unsafe { *out_len = b.len(); *out = b.as_mut_ptr() as *mut c_char; }
                   std::mem::forget(b); 0 }
        Err(e) => set_err(e),
    }
}
#[no_mangle]
pub extern "C" fn b70_tok_id_to_token(t: *const b70_tok, id: u32, out: *mut *mut c_char, out_len: *mut usize) -> i32 {
    match unsafe { &*t }.0.id_to_token(id) {
        Some(s) => { let mut b = s.into_bytes(); b.shrink_to_fit();
                     unsafe { *out_len = b.len(); *out = b.as_mut_ptr() as *mut c_char; }
                     std::mem::forget(b); 0 }
        None => set_err(format!("id {id} has no token")),
    }
}
#[no_mangle]
pub extern "C" fn b70_tok_token_to_id(t: *const b70_tok, tok: *const c_char, len: usize) -> i64 {
    let bytes = unsafe { slice::from_raw_parts(tok as *const u8, len) };
    match std::str::from_utf8(bytes).ok().and_then(|s| unsafe { &*t }.0.token_to_id(s)) {
        Some(id) => id as i64, None => -1,
    }
}
#[no_mangle]
pub extern "C" fn b70_tok_free_buf(p: *mut std::ffi::c_void, len: usize, is_ids: i32) {
    if p.is_null() { return; }
    unsafe {
        if is_ids != 0 { drop(Vec::from_raw_parts(p as *mut u32, len, len)); }
        else { drop(Vec::from_raw_parts(p as *mut u8, len, len)); }
    }
}
#[no_mangle]
pub extern "C" fn b70_tok_last_error() -> *const c_char { LAST.with(|l| l.borrow().as_ptr()) }
```

- [ ] **Step 4: Build once by hand on the box** - `tools/box.sh sync && tools/box.sh run '~/.cargo/bin/cargo build --release --manifest-path src/tokenizer/rs/Cargo.toml --target-dir build/tokenizer-rs 2>&1 | tail -5; ls -la build/tokenizer-rs/release/libb70_tok.a'`. Expected: `Finished release`, an archive of ~10-20 MB (estimated; record the measured size). First build downloads crates (network verified 2026-09-06: `index.crates.io` 200); record the wall time.
- [ ] **Step 5: Pull and commit the lock file** - `tools/box.sh pull src/tokenizer/rs/Cargo.lock`. Commit: `git add src/tokenizer/rs src/tokenizer/b70_tok.h && git commit -m "feat(tokenizer): the tokenizers crate behind a six-function C ABI"`.

---

### Task 3: CMake integration

**Files:**
- Create: `cmake/tokenizer.cmake`
- Modify: `CMakeLists.txt` (one `include(cmake/tokenizer.cmake)` after line 12's `find_package(PkgConfig)`; one `add_subdirectory(src/tokenizer)` after `add_subdirectory(src/common)`)
- Create: `src/tokenizer/CMakeLists.txt`

**Interfaces:** Produces CMake targets `b70_tok_rs` (imported STATIC) and `b70_tokenizer` (STATIC, C++), the cache variable `B70_TOKENIZER` (AUTO/ON/OFF, mirrors `cmake/prefill.cmake`'s pattern) and `B70_TOKENIZER_ENABLED`.

- [ ] **Step 1: `cmake/tokenizer.cmake`**

```cmake
# The tokenizer component (spec 3 §3.1): one Rust crate built by cargo into a
# static archive. Same optionality shape as cmake/prefill.cmake - AUTO turns
# it on when cargo is usable, OFF leaves the decode build byte-identical.
set(B70_TOKENIZER "AUTO" CACHE STRING "Rust tokenizer component: AUTO, ON or OFF")
set_property(CACHE B70_TOKENIZER PROPERTY STRINGS AUTO ON OFF)
set(B70_TOKENIZER_ENABLED OFF)
if(NOT B70_TOKENIZER STREQUAL "OFF")
  find_program(B70_CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" "$ENV{CARGO_HOME}/bin")
  if(B70_CARGO)
    execute_process(COMMAND "${B70_CARGO}" --version OUTPUT_VARIABLE _cargo_v
                    RESULT_VARIABLE _cargo_rc OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(_cargo_rc 1)
    set(_cargo_v "cargo not found (looked in ~/.cargo/bin and PATH)")
  endif()
  if(_cargo_rc EQUAL 0)
    set(B70_TOKENIZER_ENABLED ON)
    message(STATUS "b70: tokenizer component ON  -- ${_cargo_v}")
  elseif(B70_TOKENIZER STREQUAL "ON")
    message(FATAL_ERROR "B70_TOKENIZER=ON but cargo is unusable: ${_cargo_v}")
  else()
    message(STATUS "b70: tokenizer component OFF -- ${_cargo_v}")
  endif()
endif()

if(B70_TOKENIZER_ENABLED)
  set(_rs_dir "${CMAKE_SOURCE_DIR}/src/tokenizer/rs")
  set(_rs_target_dir "${CMAKE_BINARY_DIR}/tokenizer-rs")
  set(B70_TOK_RS_LIB "${_rs_target_dir}/release/libb70_tok.a")
  file(GLOB_RECURSE _rs_sources "${_rs_dir}/src/*.rs")
  add_custom_command(
    OUTPUT "${B70_TOK_RS_LIB}"
    COMMAND "${B70_CARGO}" build --release --quiet
            --manifest-path "${_rs_dir}/Cargo.toml" --target-dir "${_rs_target_dir}"
    DEPENDS ${_rs_sources} "${_rs_dir}/Cargo.toml" "${_rs_dir}/Cargo.lock"
    COMMENT "cargo build --release (tokenizers crate)"
    VERBATIM)
  add_custom_target(b70_tok_rs_build DEPENDS "${B70_TOK_RS_LIB}")
  add_library(b70_tok_rs STATIC IMPORTED GLOBAL)
  set_target_properties(b70_tok_rs PROPERTIES IMPORTED_LOCATION "${B70_TOK_RS_LIB}")
  add_dependencies(b70_tok_rs b70_tok_rs_build)
  # What a Rust staticlib needs from the C runtime on Linux (rustc --print
  # native-static-libs says the same; verified in Task 3 Step 3).
  target_link_libraries(b70_tok_rs INTERFACE pthread dl m)
endif()
```

- [ ] **Step 2: `src/tokenizer/CMakeLists.txt`**

```cmake
if(NOT B70_TOKENIZER_ENABLED)
  return()
endif()
add_library(b70_tokenizer STATIC tokenizer.cc)
target_include_directories(b70_tokenizer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_tokenizer PUBLIC b70_tok_rs)
```

- [ ] **Step 3: Confirm the native-static-libs line** - `tools/box.sh run '~/.cargo/bin/rustc --print native-static-libs --crate-type staticlib - </dev/null 2>&1 | grep native-static-libs'`. Expected a line naming `-lgcc_s -lutil -lrt -lpthread -lm -ldl -lc` (the set varies by toolchain; add any that the link in Task 4 Step 5 reports missing, and record the line).
- [ ] **Step 4: Configure on the box** - `tools/box.sh build 2>&1 | grep -E "b70: (tokenizer|prefill)"`. Expected: `b70: tokenizer component ON -- cargo 1.xx`. Also `tools/box.sh run 'cmake -S . -B build-notok -DB70_TOKENIZER=OFF > /dev/null && echo configured-off'` proves OFF configures (do not build it; delete nothing - the directory is 1 KB of cache files).
- [ ] **Step 5: Commit** - `git add cmake/tokenizer.cmake CMakeLists.txt src/tokenizer/CMakeLists.txt && git commit -m "build(tokenizer): cargo step under B70_TOKENIZER (AUTO/ON/OFF)"`.

---

### Task 4: `tok::Tokenizer` wrapper and the smoke test

**Files:**
- Create: `src/tokenizer/tokenizer.h`, `src/tokenizer/tokenizer.cc`
- Create: `tests/tokenizer/tokenizer_smoke_test.cc`
- Modify: `tests/CMakeLists.txt` (append a guarded block)

**Interfaces:**
- Produces:

```cpp
namespace tok {
class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json_path);   // throws std::runtime_error
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete; Tokenizer& operator=(const Tokenizer&) = delete;
  uint32_t vocab_size() const;                                   // 248077 for the checkpoint
  std::vector<uint32_t> encode(std::string_view text, bool add_special = false) const;
  std::string decode(const std::vector<uint32_t>& ids, bool skip_special = false) const;
  std::string id_to_token(uint32_t id) const;
  std::optional<uint32_t> token_to_id(std::string_view token) const;
};
// The path every test defaults to; B70_TOKENIZER_JSON overrides.
std::string default_tokenizer_json();
}
```

- [ ] **Step 1: Write the failing smoke test**

```cpp
// tests/tokenizer/tokenizer_smoke_test.cc
#include <cstdio>
#include <string>
#include "check.h"
#include "tokenizer/tokenizer.h"
int main() {
  tok::Tokenizer t(tok::default_tokenizer_json());
  CHECK_EQ(t.vocab_size(), 248077u);                       // docs/11: 248044 + 33 added
  const std::vector<uint32_t> ids = t.encode("Hello, world!");
  CHECK(!ids.empty());
  CHECK_EQ(t.decode(ids), std::string("Hello, world!"));
  // Special tokens survive encode(add_special=false) → decode(skip_special=false)
  // as literal text, and map to the ids docs/11 records.
  CHECK_EQ(t.token_to_id("<|im_end|>").value_or(0), 248046u);
  CHECK_EQ(t.token_to_id("<|endoftext|>").value_or(0), 248044u);
  const std::vector<uint32_t> s = t.encode("<|im_start|>user\nhi<|im_end|>\n");
  CHECK_EQ(s.front(), 248045u);                            // <|im_start|>
  CHECK_EQ(t.decode(s), std::string("<|im_start|>user\nhi<|im_end|>\n"));
  CHECK(t.token_to_id("definitely-not-a-token-xyz") == std::nullopt);
  std::printf("tokenizer_smoke_test OK: vocab %u, %zu ids for the greeting\n", t.vocab_size(), ids.size());
  return 0;
}
```

(`<|im_start|>` = 248045 is docs/11's ordering `<|endoftext|>`=248044, `<|im_start|>`=248045, `<|im_end|>`=248046; if `token_to_id` disagrees, the test's constant is wrong, not the crate - fix the constant and record the measured id.)

- [ ] **Step 2: Register it** - append to `tests/CMakeLists.txt`:

```cmake
# --- spec 3 T1: tokenizer (host-only; needs the checkpoint's tokenizer.json) ---
if(B70_TOKENIZER_ENABLED)
  add_executable(tokenizer_smoke_test tokenizer/tokenizer_smoke_test.cc)
  target_include_directories(tokenizer_smoke_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(tokenizer_smoke_test PRIVATE b70_tokenizer)
  add_test(NAME tokenizer_smoke_test COMMAND tokenizer_smoke_test)
endif()
```

- [ ] **Step 3: Run it to see it fail** - `tools/box.sh test tokenizer_smoke` → expected: link error (no `tokenizer.cc` yet).
- [ ] **Step 4: Implement the wrapper**

```cpp
// src/tokenizer/tokenizer.h
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
struct b70_tok;
namespace tok {
class Tokenizer { /* the interface block above, verbatim */
 private:
  b70_tok* h_ = nullptr;
};
std::string default_tokenizer_json();
}  // namespace tok
```

```cpp
// src/tokenizer/tokenizer.cc
#include "tokenizer/tokenizer.h"
#include <cstdlib>
#include <stdexcept>
#include "tokenizer/b70_tok.h"
namespace tok {
namespace {
[[noreturn]] void fail(const char* what) {
  throw std::runtime_error(std::string("tok::Tokenizer::") + what + ": " + b70_tok_last_error());
}
}  // namespace
Tokenizer::Tokenizer(const std::string& path) {
  if (b70_tok_new(path.c_str(), &h_) != 0) fail("Tokenizer");
}
Tokenizer::~Tokenizer() { b70_tok_free(h_); }
uint32_t Tokenizer::vocab_size() const { return b70_tok_vocab_size(h_); }
std::vector<uint32_t> Tokenizer::encode(std::string_view text, bool add_special) const {
  uint32_t* ids = nullptr; size_t n = 0;
  if (b70_tok_encode(h_, text.data(), text.size(), add_special ? 1 : 0, &ids, &n) != 0) fail("encode");
  std::vector<uint32_t> out(ids, ids + n);
  b70_tok_free_buf(ids, n, 1);
  return out;
}
std::string Tokenizer::decode(const std::vector<uint32_t>& ids, bool skip_special) const {
  char* s = nullptr; size_t n = 0;
  if (b70_tok_decode(h_, ids.data(), ids.size(), skip_special ? 1 : 0, &s, &n) != 0) fail("decode");
  std::string out(s, n);
  b70_tok_free_buf(s, n, 0);
  return out;
}
std::string Tokenizer::id_to_token(uint32_t id) const {
  char* s = nullptr; size_t n = 0;
  if (b70_tok_id_to_token(h_, id, &s, &n) != 0) fail("id_to_token");
  std::string out(s, n);
  b70_tok_free_buf(s, n, 0);
  return out;
}
std::optional<uint32_t> Tokenizer::token_to_id(std::string_view t) const {
  const int64_t id = b70_tok_token_to_id(h_, t.data(), t.size());
  if (id < 0) return std::nullopt;
  return uint32_t(id);
}
std::string default_tokenizer_json() {
  if (const char* e = std::getenv("B70_TOKENIZER_JSON")) return e;
  return "/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64/tokenizer.json";
}
}  // namespace tok
```

- [ ] **Step 5: Run to pass** - `tools/box.sh test tokenizer_smoke`. Expected: `tokenizer_smoke_test OK`. If the link reports undefined `__…` symbols, add the library from Task 3 Step 3's line to `target_link_libraries(b70_tok_rs INTERFACE …)` - one commit, named.
- [ ] **Step 6: Decode invariants unchanged** - `tools/box.sh test "replay_determinism|kernel_table"` (with the snapshot argv the tests need - see tests/CMakeLists.txt). Expected: 774 kernels / 19 modules as before.
- [ ] **Step 7: Commit** - `git add src/tokenizer tests/tokenizer/tokenizer_smoke_test.cc tests/CMakeLists.txt && git commit -m "feat(tokenizer): tok::Tokenizer wrapper + smoke test (vocab 248077, special ids)"`.

---

### Task 5: The parity corpus and its dump

**Files:**
- Create: `tools/tokenizer/make_corpus.py`, `tools/tokenizer/dump_parity.py`
- Create (generated, committed): `tests/tokenizer/corpus.txt`, `tests/tokenizer/corpus.ids`

**Interfaces:** Produces `corpus.txt` (one line per case, `\n`-terminated; the case is the line WITHOUT its terminator, exactly as `tokenize.py` strips trailing newlines) and `corpus.ids` (same line count; whitespace-separated ids; an empty line for a case that encodes to zero ids). Consumed by Task 6 and by plan 7b's streamer test.

- [ ] **Step 1: `make_corpus.py`** - deterministic (`random.Random(20260906)`), stdlib only, ≥ 10,240 lines across these families, each family ≥ 1,000 lines: (a) English prose sentences from a fixed word list; (b) code lines (Python/C++/JSON fragments, tabs, 4- and 8-space indents, trailing spaces); (c) CJK (Chinese, Japanese kana+kanji, Korean hangul) sentences from fixed character pools; (d) emoji incl. ZWJ sequences, skin tones, flags; (e) whitespace runs (`" " * k`, `"\t" * k`, mixed, `k ∈ 1..40`, and `\r\n`); (f) contractions and case (`it's`, `IT'S`, `They'VE`); (g) digits (`2026`, `3.14159`, `1,000,000`, unicode digits `１２３`); (h) EVERY added token from `tokenizer.json` (`added_tokens[].content`), each alone, each embedded mid-sentence, each adjacent to another; (i) mixed lines combining two families. Interior `\n` never appears (a case is one line); `\r` may. Write the file with `newline=""` so `\r` survives. Print the family counts.
- [ ] **Step 2: `dump_parity.py`** - runs in the box venv:

```python
#!/usr/bin/env python3
"""tests/tokenizer/corpus.txt -> corpus.ids with Python `tokenizers` (the reference).
Run ON THE BOX: ~/auto-round/.venv/bin/python tools/tokenizer/dump_parity.py <tokenizer.json>
Also asserts decode(encode(x)) == x for every case and prints the version used."""
import sys, tokenizers
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
src = open("tests/tokenizer/corpus.txt", encoding="utf-8", newline="").read().split("\n")
if src and src[-1] == "": src.pop()
out, bad = [], 0
for line in src:
    ids = tok.encode(line, add_special_tokens=False).ids
    if tok.decode(ids, skip_special_tokens=False) != line: bad += 1
    out.append(" ".join(map(str, ids)))
open("tests/tokenizer/corpus.ids", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(f"tokenizers {tokenizers.__version__}: {len(src)} cases, {sum(len(o.split()) for o in out)} ids, {bad} roundtrip failures")
sys.exit(1 if bad else 0)
```

- [ ] **Step 3: Generate on the Mac, dump on the box** - `python3 tools/tokenizer/make_corpus.py && tools/box.sh sync && tools/box.sh run '~/auto-round/.venv/bin/python tools/tokenizer/dump_parity.py /home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64/tokenizer.json' && tools/box.sh pull tests/tokenizer/corpus.ids`. Expected: `tokenizers 0.22.2: 10240 cases, N ids, 0 roundtrip failures`. If roundtrip failures are non-zero, list the cases: they are corpus lines the REFERENCE cannot round-trip (e.g. lone surrogates), and they leave the corpus, recorded - parity is against the reference, not beyond it.
- [ ] **Step 4: Sizes** - record `wc -c tests/tokenizer/corpus.*` (estimated: 0.8 MB + 3 MB; the repo already carries a 4 MB safetensors fixture, so this is in range).
- [ ] **Step 5: Commit** - `git add tools/tokenizer tests/tokenizer/corpus.txt tests/tokenizer/corpus.ids && git commit -m "test(tokenizer): 10k-line parity corpus + reference ids (tokenizers 0.22.2)"`.

---

### Task 6: The parity test (spec bar 1, encode/decode halves)

**Files:**
- Create: `tests/tokenizer/parity_test.cc`
- Modify: `tests/CMakeLists.txt` (inside the T1 block)

**Interfaces:** Consumes Task 4's `tok::Tokenizer`, Task 5's files (paths relative to the source tree: pass `${CMAKE_SOURCE_DIR}/tests/tokenizer` as argv[1] in `add_test`).

- [ ] **Step 1: Write the test**

```cpp
// tests/tokenizer/parity_test.cc - spec 3 §2 bar 1, encode + decode halves.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "check.h"
#include "tokenizer/tokenizer.h"
static std::vector<std::string> lines(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  CHECK(f.good());
  std::stringstream ss; ss << f.rdbuf();
  std::string all = ss.str();
  std::vector<std::string> out; size_t b = 0;
  for (size_t i = 0; i < all.size(); ++i) if (all[i] == '\n') { out.emplace_back(all, b, i - b); b = i + 1; }
  if (b < all.size()) out.emplace_back(all, b);
  return out;
}
int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  tok::Tokenizer t(tok::default_tokenizer_json());
  const auto txt = lines(dir + "/corpus.txt"), ref = lines(dir + "/corpus.ids");
  CHECK_EQ(txt.size(), ref.size());
  CHECK(txt.size() >= 10000);
  size_t enc_bad = 0, dec_bad = 0, total_ids = 0;
  for (size_t i = 0; i < txt.size(); ++i) {
    std::vector<uint32_t> want; std::istringstream is(ref[i]); uint32_t v;
    while (is >> v) want.push_back(v);
    const std::vector<uint32_t> got = t.encode(txt[i]);
    total_ids += got.size();
    if (got != want) { if (enc_bad < 10) std::fprintf(stderr, "encode mismatch line %zu: %s\n", i + 1, txt[i].c_str()); ++enc_bad; }
    if (t.decode(got) != txt[i]) { if (dec_bad < 10) std::fprintf(stderr, "decode mismatch line %zu\n", i + 1); ++dec_bad; }
  }
  std::printf("parity_test: %zu cases, %zu ids, %zu encode mismatches, %zu decode mismatches\n",
              txt.size(), total_ids, enc_bad, dec_bad);
  CHECK_EQ(enc_bad, size_t(0));
  CHECK_EQ(dec_bad, size_t(0));
  std::printf("parity_test OK\n");
  return 0;
}
```

- [ ] **Step 2: Register** - inside the T1 block: `add_executable(parity_test tokenizer/parity_test.cc)`, same includes/links as the smoke test, `add_test(NAME parity_test COMMAND parity_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)`.
- [ ] **Step 3: Run** - `tools/box.sh test parity_test`. Expected: `0 encode mismatches, 0 decode mismatches` - by construction (same crate, same version). A non-zero count is a finding (a crate-version drift, or a corpus line the venv encoded differently) - report with the first ten lines, do not edit the corpus to pass.
- [ ] **Step 4: Full suite** - `tools/box.sh test`. Expected: 60 + 2 pass, 1 designed skip (long gate, unless its oracle has landed).
- [ ] **Step 5: Commit** - `git add tests/tokenizer/parity_test.cc tests/CMakeLists.txt && git commit -m "test(tokenizer): parity_test - 10k cases byte-identical to Python tokenizers"`.

---

### Task 7: Docs

**Files:** Modify `docs/11-tokenizer-and-chat-template.md` (a "Status (spec 3 T1, 2026-09-06)" paragraph under "Options for the BPE encoder": which option shipped, crate version, archive size and first-build time measured, the `pretokenize_regex` note below), `docs/09-*build*` notes (the cargo step, its cache dir `build/tokenizer-rs`, `B70_TOKENIZER`), `docs/04-architecture.md` component table row for `src/tokenizer/` if the table lists directories.

- [ ] **Step 1:** The regex note - `tokenizer_config.json`'s `pretokenize_regex` (read 2026-09-06) contains `\p{M}` in two places where docs/11's quoted pattern (read 2026-08-22 from `tokenizer.json`) does not; state both verbatim, that the crate reads `tokenizer.json` and the parity corpus is the arbiter, and that any future hand-written BPE must match `tokenizer.json`'s pattern, not the config's.
- [ ] **Step 2:** Commit - `git commit -am "docs(tokenizer): T1 status, build notes, the pretokenize_regex discrepancy"`.

---

## Self-review

**Spec coverage:** §3.1 (crate, C interface, static `.a`, CMake/Cargo, leaf) → Tasks 2-4; bar 1 encode/decode/special tokens/≥10k lines → Tasks 5-6; the streaming-detokeniser half of bar 1 → plan 7b (it needs the streamer, which is T2's); §5 Rust user-local → Task 1; `third_party/VERSIONS` is not touched here (no vendored header in T1; the crate pin lives in `Cargo.toml`/`Cargo.lock` and Task 7 says so). **Placeholders:** none - the two "record the measured X" items are execution outputs. **Types:** `tok::Tokenizer` signatures are identical in Task 4's interface block, its implementation, and Task 6's use.

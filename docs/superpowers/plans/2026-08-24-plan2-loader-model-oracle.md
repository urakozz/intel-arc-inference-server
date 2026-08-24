# Plan 2 of 3 - Loader, model description, oracle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything between the checkpoint files and the first kernel launch:
a dependency-free loader that takes the HF snapshot to the canonical on-device
layout with every assumption asserted, the qwen3_5 model description as data,
and the Python oracle that produces the golden tensors plan 3's tests compare
against - spec 1 Sections 6, 7 and 10.

**Architecture:** The loader is pure host C++17 (no new link dependencies -
`ze_loader` via `b70_l0` stays the only one; JSON is parsed by a ~250-line
in-tree parser). Weights are mmapped read-only, classified by tensor-name
suffix, repacked per-linear into the canonical tiled layouts measured in plan
1, fused at load (qkv‖z, q‖k‖v, gate‖up interleaved, a‖b padded), and uploaded
once through the synchronous immediate list. The model description
(`src/model/qwen35.*`) is a table - layer kinds, shapes, weight→buffer
bindings, per-linear `(layout, S)` - with no L0 or kernel code. The oracle
runs `transformers` on CPU inside the reference container and writes
safetensors golden files plus a dequant fixture that pins the meaning of the
nibbles for both sides.

**Tech Stack:** C++17, `g++`, CMake (existing skeleton), mmap/POSIX (Linux
only - a decided project constraint), `ocloc` (no new kernels in this plan),
Python 3 + torch + transformers 5.15 **inside the container only**.

**Spec:** `docs/superpowers/specs/2026-08-22-phase0-decode-core-design.md`
(Sections 1 "Platform and model acquisition", 2, 6, 7, 10, 11) and the
measured facts of plan 1 (`docs/probe-gemv-2026-08-24.md`, `docs/12-kernels.md`,
`docs/03-models.md`). Plan 3 (runtime + remaining kernels + CLI + golden
tests) follows this plan.

## Global Constraints

- C++17, `-Wall -Wextra -Werror`, `g++`; host links `ze_loader` only (via
  `b70_l0`). **No new third-party code** - JSON is parsed by `src/common/json.h`
  written in Task 1 (vendoring a JSON library was considered and rejected:
  goal 3 "own the dependencies", and the inputs are machine-written JSON from
  `safetensors`/HF, not adversarial).
- Linux only; the loader never downloads. Input is an absolute snapshot
  directory or an HF repo id resolved against `~/.cache/huggingface/hub` /
  `$HF_HOME/hub` via `refs/main` (spec §1). A missing snapshot is an error
  naming the path.
- Checkpoint facts to assert at load, failing by tensor name (spec §6.2, all
  verified on the real checkpoint in plan 0/1): `sym == true`,
  `desc_act == false`, `group_size == 64`; every `qzeros` word `== 0x77777777`
  (GPTQ v1: zero−1); `g_idx[k] == k/64`; shapes match `docs/03-models.md`;
  the 98 `dynamic` exclusions match exactly the tensors found in bf16.
  Dequant meaning: `w = (q − 8) · scale`, `q` = unsigned nibble, nibble `i`
  of `qweight[k/8][N]` word is `k = (k/8)*8 + i`.
- Canonical layouts are plan 1's (docs/12-kernels.md): int4 layout 1 tiles of
  136 u32 per `(n_tile 16, k_group 64)`; bf16 tiles `[n_tile][k_octet][8k][16n]`.
  **Both** int4 repack paths are kept; the per-linear layout comes from the
  model description's table, which defaults to layout 1 everywhere
  (controller ruling 2026-08-24 resolving spec §6.3's "loser is deleted"
  line: deleting layout 0 would forfeit a measured +3.5% best-per-shape
  option; flipping the table is a phase-1 tuning knob measured end-to-end,
  not exercised in this plan).
- Norm weights are stored as `1 + w` (bf16) except `linear_attn.norm` (plain
  `w`); `A_log` is stored as `−exp(A_log)` fp32; `dt_bias` fp32;
  `conv1d.weight` fp32 `[10240][4]`; RoPE table `cos/sin[max_model_len][32]`
  fp32 pairs with `inv_freq_i = 1e7^(−2i/64)` (doc 03 "Layer math").
- Loaded resident weight bytes must be asserted against `W` (doc 03:
  15.519 GB ± the padding this plan adds, itemised - the assert prints the
  breakdown, it does not hide it).
- Builds and tests run on the box via `tools/box.sh` (`JOBS=44`). Tests that
  need the real checkpoint use the snapshot path
  `~/.cache/huggingface/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/2a9077667e28aa53e61d91bdee5d7962e8674668/`
  on the box and are registered in ctest with the label `checkpoint` so
  `ctest -L checkpoint` can be excluded where the model is absent.
- The oracle (Tasks 7-8) runs in `vllm-xpu-env-next-p314-t214-vxkp0` on the
  box, CPU only, no `fla`, `HF_HUB_OFFLINE=1`. Oracle outputs land under
  `~/b70-oracle/` on the box and `tests/golden/` fixtures below 1 MB are
  committed; large golden files are never committed (`.gitignore` already
  excludes `*.safetensors`).
- Every task lands with its explanation: loader mechanism goes in a new
  `docs/13-loader.md` (Task 6), oracle usage in `tools/oracle/README.md`
  (Task 8). No component merges without its doc.
- Commit after every task; messages `feat:`/`test:`/`docs:`/`fix:`.

---

### Task 1: `src/common/json.h` - the in-tree JSON parser

**Files:**
- Create: `src/common/json.h` (header-only)
- Create: `tests/common/json_test.cc`
- Modify: `tests/CMakeLists.txt`

**Interfaces (namespace `common::json`):**
- `struct Value` - tagged union over `Null`, `bool`, `double`, `std::string`,
  `Array = std::vector<Value>`, `Object = std::map<std::string, Value>`.
  Accessors: `is_object()/is_array()/is_string()/is_number()/is_bool()/is_null()`;
  `const Object& obj() const`, `const Array& arr() const`,
  `const std::string& str() const`, `double num() const`, `bool boolean() const`
  (each throws `std::runtime_error` on kind mismatch, naming the expected and
  actual kind); convenience `const Value* find(const std::string& key) const`
  (object lookup, nullptr if absent) and
  `const Value& at(const std::string& key) const` (throws naming the key).
- `Value parse(std::string_view text)` - throws `std::runtime_error` with
  byte offset and a 20-char excerpt on malformed input.
- Supported: objects, arrays, strings with escapes `\" \\ \/ \b \f \n \r \t \uXXXX`
  (UTF-16 surrogate pairs → UTF-8), numbers via `strtod` (ints and floats -
  tensor offsets up to 2^53 are exact in double, and safetensors offsets for
  this checkpoint are < 2^35), `true/false/null`, arbitrary nesting,
  whitespace per RFC. NOT supported (documented in the header): comments,
  trailing commas, duplicate-key detection (last wins), NaN/Infinity.

- [ ] **Step 1: Write the failing test `tests/common/json_test.cc`**

```cpp
#include <cstdint>
#include "check.h"
#include "common/json.h"

using common::json::Value;
using common::json::parse;

int main() {
  // Object, array, nesting, numbers, bools, null.
  Value v = parse(R"({"a": 1, "b": [true, false, null, 2.5], "c": {"d": "x"}})");
  CHECK(v.is_object());
  CHECK_EQ(v.at("a").num(), 1.0);
  CHECK_EQ(v.at("b").arr().size(), size_t(4));
  CHECK(v.at("b").arr()[0].boolean());
  CHECK(v.at("b").arr()[2].is_null());
  CHECK_NEAR(v.at("b").arr()[3].num(), 2.5, 0.0);
  CHECK_EQ(v.at("c").at("d").str(), std::string("x"));
  CHECK(v.find("missing") == nullptr);

  // String escapes incl. \uXXXX and a surrogate pair (U+1F600 GRINNING FACE).
  Value s = parse(R"({"s": "a\"b\\c\ndé😀"})");
  CHECK_EQ(s.at("s").str(), std::string("a\"b\\c\nd\xC3\xA9\xF0\x9F\x98\x80"));

  // Large exact integer (safetensors offsets): 2^40 + 5 is exact in double.
  Value n = parse(R"({"off": 1099511627781})");
  CHECK_EQ(uint64_t(n.at("off").num()), uint64_t(1099511627781ULL));

  // The shapes that actually occur: a safetensors header entry.
  Value h = parse(R"({"model.layers.0.mlp.up_proj.qweight":
      {"dtype": "I32", "shape": [640, 17408], "data_offsets": [0, 44564480]}})");
  const Value& e = h.at("model.layers.0.mlp.up_proj.qweight");
  CHECK_EQ(e.at("dtype").str(), std::string("I32"));
  CHECK_EQ(uint64_t(e.at("shape").arr()[1].num()), uint64_t(17408));

  // Errors: position is reported; wrong-kind access throws.
  bool threw = false;
  try { parse("{\"a\": [1, }"); } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { (void)v.at("a").str(); } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  std::puts("json_test OK");
  return 0;
}
```

`tests/CMakeLists.txt` add (mirror `int4_test` - no L0, no kernels):
```cmake
add_executable(json_test common/json_test.cc)
target_include_directories(json_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME json_test COMMAND json_test)
```

- [ ] **Step 2: Run to verify it fails** - `tools/box.sh test json_test` →
  missing `common/json.h`.

- [ ] **Step 3: Implement `src/common/json.h`**

```cpp
#pragma once
// Minimal recursive-descent JSON parser for machine-written inputs
// (safetensors headers, HF config/index files). Deliberately in-tree: the
// project owns its dependencies (README goal 3), and the grammar subset
// below covers everything those producers emit.
//   Supported: objects, arrays, strings (escapes incl. \uXXXX with surrogate
//   pairs -> UTF-8), numbers via strtod (offsets < 2^53 are exact), true/
//   false/null. Not supported: comments, trailing commas, NaN/Infinity;
//   duplicate keys resolve to the last occurrence.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace common::json {

class Value {
 public:
  enum class Kind { Null, Bool, Number, String, Array, Object };
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value>;

  Value() : kind_(Kind::Null) {}
  static Value make_bool(bool b) { Value v; v.kind_ = Kind::Bool; v.bool_ = b; return v; }
  static Value make_num(double d) { Value v; v.kind_ = Kind::Number; v.num_ = d; return v; }
  static Value make_str(std::string s) { Value v; v.kind_ = Kind::String; v.str_ = std::move(s); return v; }
  static Value make_arr(Array a) { Value v; v.kind_ = Kind::Array; v.arr_ = std::move(a); return v; }
  static Value make_obj(Object o) { Value v; v.kind_ = Kind::Object; v.obj_ = std::move(o); return v; }

  bool is_null() const { return kind_ == Kind::Null; }
  bool is_bool() const { return kind_ == Kind::Bool; }
  bool is_number() const { return kind_ == Kind::Number; }
  bool is_string() const { return kind_ == Kind::String; }
  bool is_array() const { return kind_ == Kind::Array; }
  bool is_object() const { return kind_ == Kind::Object; }

  bool boolean() const { require(Kind::Bool); return bool_; }
  double num() const { require(Kind::Number); return num_; }
  const std::string& str() const { require(Kind::String); return str_; }
  const Array& arr() const { require(Kind::Array); return arr_; }
  const Object& obj() const { require(Kind::Object); return obj_; }

  const Value* find(const std::string& key) const {
    require(Kind::Object);
    auto it = obj_.find(key);
    return it == obj_.end() ? nullptr : &it->second;
  }
  const Value& at(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw std::runtime_error("json: missing key '" + key + "'");
    return *v;
  }

 private:
  static const char* kind_name(Kind k) {
    switch (k) {
      case Kind::Null: return "null";
      case Kind::Bool: return "bool";
      case Kind::Number: return "number";
      case Kind::String: return "string";
      case Kind::Array: return "array";
      case Kind::Object: return "object";
    }
    return "?";
  }
  void require(Kind k) const {
    if (kind_ != k)
      throw std::runtime_error(std::string("json: expected ") + kind_name(k) +
                               ", value is " + kind_name(kind_));
  }
  Kind kind_;
  bool bool_ = false;
  double num_ = 0;
  std::string str_;
  Array arr_;
  Object obj_;
};

namespace detail {

class Parser {
 public:
  explicit Parser(std::string_view t) : t_(t) {}
  Value parse_document() {
    Value v = parse_value();
    skip_ws();
    if (p_ != t_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    size_t end = p_ + 20 < t_.size() ? p_ + 20 : t_.size();
    throw std::runtime_error("json: " + std::string(what) + " at byte " +
                             std::to_string(p_) + " near '" +
                             std::string(t_.substr(p_, end - p_)) + "'");
  }
  void skip_ws() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) ++p_;
  }
  char peek() {
    if (p_ >= t_.size()) fail("unexpected end of input");
    return t_[p_];
  }
  char take() { char c = peek(); ++p_; return c; }
  void expect(char c) {
    if (take() != c) { --p_; fail("unexpected character"); }
  }
  bool consume_lit(std::string_view lit) {
    if (t_.substr(p_, lit.size()) == lit) { p_ += lit.size(); return true; }
    return false;
  }

  Value parse_value() {
    skip_ws();
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return Value::make_str(parse_string());
    if (consume_lit("true")) return Value::make_bool(true);
    if (consume_lit("false")) return Value::make_bool(false);
    if (consume_lit("null")) return Value();
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    fail("expected a value");
  }

  Value parse_object() {
    expect('{');
    Value::Object o;
    skip_ws();
    if (peek() == '}') { ++p_; return Value::make_obj(std::move(o)); }
    for (;;) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      o[std::move(key)] = parse_value();
      skip_ws();
      char c = take();
      if (c == '}') break;
      if (c != ',') { --p_; fail("expected ',' or '}'"); }
    }
    return Value::make_obj(std::move(o));
  }

  Value parse_array() {
    expect('[');
    Value::Array a;
    skip_ws();
    if (peek() == ']') { ++p_; return Value::make_arr(std::move(a)); }
    for (;;) {
      a.push_back(parse_value());
      skip_ws();
      char c = take();
      if (c == ']') break;
      if (c != ',') { --p_; fail("expected ',' or ']'"); }
    }
    return Value::make_arr(std::move(a));
  }

  Value parse_number() {
    size_t start = p_;
    if (peek() == '-') ++p_;
    while (p_ < t_.size() && ((t_[p_] >= '0' && t_[p_] <= '9') || t_[p_] == '.' ||
                              t_[p_] == 'e' || t_[p_] == 'E' || t_[p_] == '+' || t_[p_] == '-'))
      ++p_;
    std::string num(t_.substr(start, p_ - start));
    char* end = nullptr;
    double d = std::strtod(num.c_str(), &end);
    if (end != num.c_str() + num.size()) { p_ = start; fail("malformed number"); }
    return Value::make_num(d);
  }

  void append_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
      s += char(cp);
    } else if (cp < 0x800) {
      s += char(0xC0 | (cp >> 6));
      s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      s += char(0xE0 | (cp >> 12));
      s += char(0x80 | ((cp >> 6) & 0x3F));
      s += char(0x80 | (cp & 0x3F));
    } else {
      s += char(0xF0 | (cp >> 18));
      s += char(0x80 | ((cp >> 12) & 0x3F));
      s += char(0x80 | ((cp >> 6) & 0x3F));
      s += char(0x80 | (cp & 0x3F));
    }
  }
  uint32_t parse_hex4() {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = take();
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else { --p_; fail("bad \\u escape"); }
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string s;
    for (;;) {
      char c = take();
      if (c == '"') break;
      if (c != '\\') { s += c; continue; }
      char e = take();
      switch (e) {
        case '"': s += '"'; break;
        case '\\': s += '\\'; break;
        case '/': s += '/'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'n': s += '\n'; break;
        case 'r': s += '\r'; break;
        case 't': s += '\t'; break;
        case 'u': {
          uint32_t cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate
            expect('\\');
            expect('u');
            uint32_t lo = parse_hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("unpaired surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(s, cp);
          break;
        }
        default: --p_; fail("bad escape");
      }
    }
    return s;
  }

  std::string_view t_;
  size_t p_ = 0;
};

}  // namespace detail

inline Value parse(std::string_view text) { return detail::Parser(text).parse_document(); }

}  // namespace common::json
```

- [ ] **Step 4: Run** - `tools/box.sh test json_test` → `json_test OK`.

- [ ] **Step 5: Commit**

```bash
git add src/common/json.h tests/common/json_test.cc tests/CMakeLists.txt
git commit -m "feat(common): minimal in-tree JSON parser for safetensors/HF metadata"
```

---

### Task 2: Snapshot resolution + safetensors mmap + index manifest

**Files:**
- Create: `src/loader/snapshot.h`, `src/loader/snapshot.cc`
- Create: `src/loader/safetensors.h`, `src/loader/safetensors.cc`
- Create: `src/loader/CMakeLists.txt`; Modify: root `CMakeLists.txt` (add
  `add_subdirectory(src/loader)` after `src/common`)
- Create: `tests/loader/safetensors_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces (namespace `loader`):**
- `std::string resolve_snapshot(const std::string& arg)` - if `arg` is a
  directory containing `config.json`, returns it (normalised, trailing `/`);
  else treats it as `<org>/<name>`: base = `$HF_HOME` if set else
  `$HOME/.cache/huggingface`, dir = `base + "/hub/models--<org>--<name>"`,
  revision = first line of `<dir>/refs/main`, returns
  `<dir>/snapshots/<revision>/`. Every failure throws with the exact path it
  looked at. No network, ever.
- `struct TensorInfo { std::string dtype; std::vector<uint64_t> shape; uint64_t begin, end; uint32_t file; }`
  (`begin`/`end` are offsets into the file's data section; `file` indexes
  `SafetensorsSet::files`).
- `class MappedFile { explicit MappedFile(const std::string& path); const uint8_t* data() const; size_t size() const; ~unmaps; move-only }`
  - `open(O_RDONLY)` + `mmap(PROT_READ, MAP_PRIVATE)`, `madvise(MADV_SEQUENTIAL)`.
- `class SafetensorsSet` -
  `SafetensorsSet(const std::string& snapshot_dir)`: reads
  `model.safetensors.index.json` (must exist - single-file checkpoints are
  out of scope and error clearly), maps each file named in `weight_map`
  exactly once, parses each header (8-byte LE length + JSON), and builds the
  manifest **from the index**: for every index entry, the tensor must exist
  in its named file (else throw naming both); duplicates across files are
  resolved by the index (dedup-by-name, doc 03). Members:
  `const std::map<std::string, TensorInfo>& tensors() const`;
  `const uint8_t* data(const TensorInfo&) const` (pointer into the mmap at
  `header_end + begin`); `size_t bytes(const TensorInfo&) const`;
  `const std::vector<std::string>& file_names() const`.
  Also: `static std::vector<std::pair<std::string, TensorInfo>> parse_header(const uint8_t* p, size_t n)`
  exposed for the unit test.
- Dtype strings passed through as safetensors spells them: `"BF16"`, `"F16"`,
  `"F32"`, `"I32"` (the loader rejects anything else later, by name).

- [ ] **Step 1: Write the failing test `tests/loader/safetensors_test.cc`**

The test builds a tiny two-file safetensors set **itself** in a temp dir
(no checkpoint needed; runs anywhere):

```cpp
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "check.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

// Writes one safetensors file: 8-byte LE header length, header JSON, data.
static void write_st(const std::string& path, const std::string& header,
                     const std::vector<uint8_t>& data) {
  std::ofstream f(path, std::ios::binary);
  uint64_t n = header.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(header.data(), std::streamsize(header.size()));
  f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

int main() {
  const std::string dir = "/tmp/b70_st_test";
  std::remove((dir + "/a.safetensors").c_str());
  std::remove((dir + "/b.safetensors").c_str());
  std::remove((dir + "/model.safetensors.index.json").c_str());
  std::remove((dir + "/config.json").c_str());
  (void)system(("mkdir -p " + dir).c_str());

  // t0: 4 bf16 values in a.safetensors; t1: 8 bytes I32 in b.safetensors;
  // "dup" appears in BOTH files with different bytes - the index points at b.
  std::vector<uint8_t> da = {1, 0, 2, 0, 3, 0, 4, 0, /*dup in a:*/ 0xAA, 0xAA};
  std::vector<uint8_t> db = {5, 0, 0, 0, 6, 0, 0, 0, /*dup in b:*/ 0xBB, 0xBB};
  write_st(dir + "/a.safetensors",
           R"({"t0":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
           R"("dup":{"dtype":"BF16","shape":[1],"data_offsets":[8,10]}})", da);
  write_st(dir + "/b.safetensors",
           R"({"t1":{"dtype":"I32","shape":[2],"data_offsets":[0,8]},)"
           R"("dup":{"dtype":"BF16","shape":[1],"data_offsets":[8,10]}})", db);
  std::ofstream(dir + "/model.safetensors.index.json")
      << R"({"metadata":{},"weight_map":{"t0":"a.safetensors","t1":"b.safetensors","dup":"b.safetensors"}})";
  std::ofstream(dir + "/config.json") << "{}";

  // resolve_snapshot: a directory with config.json resolves to itself.
  std::string snap = loader::resolve_snapshot(dir);
  CHECK(snap.back() == '/');

  loader::SafetensorsSet set(snap);
  CHECK_EQ(set.tensors().size(), size_t(3));
  const auto& t0 = set.tensors().at("t0");
  CHECK_EQ(t0.dtype, std::string("BF16"));
  CHECK_EQ(t0.shape[0], uint64_t(2));
  CHECK_EQ(set.bytes(t0), size_t(8));
  CHECK_EQ(set.data(t0)[0], uint8_t(1));
  const auto& t1 = set.tensors().at("t1");
  CHECK_EQ(reinterpret_cast<const uint32_t*>(set.data(t1))[1], uint32_t(6));
  // Dedup by index: "dup" must come from b.safetensors.
  const auto& dup = set.tensors().at("dup");
  CHECK_EQ(set.data(dup)[0], uint8_t(0xBB));

  // Missing tensor in named file -> throw naming both.
  std::ofstream(dir + "/model.safetensors.index.json")
      << R"({"weight_map":{"ghost":"a.safetensors"}})";
  bool threw = false;
  try { loader::SafetensorsSet bad(snap); } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("ghost") != std::string::npos;
  }
  CHECK(threw);

  // Repo-id resolution failure names the path it tried.
  threw = false;
  try { loader::resolve_snapshot("no-such-org/no-such-model"); }
  catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("models--no-such-org--no-such-model") != std::string::npos;
  }
  CHECK(threw);
  std::puts("safetensors_test OK");
  return 0;
}
```

`tests/CMakeLists.txt` add:
```cmake
add_executable(safetensors_test loader/safetensors_test.cc)
target_include_directories(safetensors_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(safetensors_test PRIVATE b70_loader)
add_test(NAME safetensors_test COMMAND safetensors_test)
```

- [ ] **Step 2: Run to verify it fails** - `tools/box.sh test safetensors_test`.

- [ ] **Step 3: Implement**

`src/loader/snapshot.h`:
```cpp
#pragma once
#include <string>

namespace loader {
// Resolves a model argument to a snapshot directory (with trailing '/').
// Accepts an absolute/relative directory containing config.json, or an HF
// repo id "<org>/<name>" resolved against the local cache
// ($HF_HOME or ~/.cache/huggingface, + /hub/models--<org>--<name>, revision
// from refs/main). Never downloads; throws naming the exact path on failure.
std::string resolve_snapshot(const std::string& arg);
}  // namespace loader
```

`src/loader/snapshot.cc`:
```cpp
#include "loader/snapshot.h"
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace loader {
namespace {
bool is_dir(const std::string& p) {
  struct stat st{};
  return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
bool is_file(const std::string& p) {
  struct stat st{};
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
}  // namespace

std::string resolve_snapshot(const std::string& arg) {
  if (is_dir(arg)) {
    std::string dir = arg;
    if (dir.back() != '/') dir += '/';
    if (!is_file(dir + "config.json"))
      throw std::runtime_error("snapshot directory has no config.json: " + dir);
    return dir;
  }
  auto slash = arg.find('/');
  if (slash == std::string::npos || arg.find('/', slash + 1) != std::string::npos)
    throw std::runtime_error("not a directory and not an <org>/<name> repo id: " + arg);
  const char* hf_home = std::getenv("HF_HOME");
  std::string base = hf_home ? std::string(hf_home)
                             : std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
                                   "/.cache/huggingface";
  std::string model_dir =
      base + "/hub/models--" + arg.substr(0, slash) + "--" + arg.substr(slash + 1);
  if (!is_dir(model_dir))
    throw std::runtime_error("model not in local HF cache (run `hf download " + arg +
                             "`): " + model_dir);
  std::ifstream ref(model_dir + "/refs/main");
  std::string rev;
  if (!ref || !std::getline(ref, rev) || rev.empty())
    throw std::runtime_error("cannot read revision: " + model_dir + "/refs/main");
  std::string snap = model_dir + "/snapshots/" + rev + "/";
  if (!is_file(snap + "config.json"))
    throw std::runtime_error("snapshot incomplete (no config.json): " + snap);
  return snap;
}
}  // namespace loader
```

`src/loader/safetensors.h`:
```cpp
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace loader {

struct TensorInfo {
  std::string dtype;             // "BF16" | "F16" | "F32" | "I32" (as spelled by safetensors)
  std::vector<uint64_t> shape;
  uint64_t begin = 0, end = 0;   // offsets into the file's data section
  uint32_t file = 0;             // index into file_names()
};

// Read-only mmap of one file.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(MappedFile&& o) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

// The checkpoint's sharded tensor set, manifest = the index (dedup by name).
class SafetensorsSet {
 public:
  explicit SafetensorsSet(const std::string& snapshot_dir);
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  const uint8_t* data(const TensorInfo& t) const;
  size_t bytes(const TensorInfo& t) const { return t.end - t.begin; }
  const std::vector<std::string>& file_names() const { return names_; }

  // Parses one safetensors header (8-byte LE length + JSON). Exposed for tests.
  static std::vector<std::pair<std::string, TensorInfo>> parse_header(const uint8_t* p, size_t n);

 private:
  std::vector<MappedFile> files_;
  std::vector<size_t> data_start_;   // per file: 8 + header_len
  std::vector<std::string> names_;
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace loader
```

`src/loader/safetensors.cc`:
```cpp
#include "loader/safetensors.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "common/json.h"

namespace loader {

MappedFile::MappedFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open: " + path);
  struct stat st{};
  if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat failed: " + path); }
  size_ = size_t(st.st_size);
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  ::madvise(p, size_, MADV_SEQUENTIAL);
  data_ = static_cast<const uint8_t*>(p);
}
MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
}
MappedFile::MappedFile(MappedFile&& o) noexcept : data_(o.data_), size_(o.size_) {
  o.data_ = nullptr;
  o.size_ = 0;
}

std::vector<std::pair<std::string, TensorInfo>> SafetensorsSet::parse_header(const uint8_t* p,
                                                                             size_t n) {
  if (n < 8) throw std::runtime_error("safetensors: file too small");
  uint64_t hlen = 0;
  std::memcpy(&hlen, p, 8);
  if (8 + hlen > n) throw std::runtime_error("safetensors: header length exceeds file");
  common::json::Value h =
      common::json::parse(std::string_view(reinterpret_cast<const char*>(p) + 8, hlen));
  std::vector<std::pair<std::string, TensorInfo>> out;
  for (const auto& [name, e] : h.obj()) {
    if (name == "__metadata__") continue;
    TensorInfo t;
    t.dtype = e.at("dtype").str();
    for (const auto& d : e.at("shape").arr()) t.shape.push_back(uint64_t(d.num()));
    const auto& off = e.at("data_offsets").arr();
    t.begin = uint64_t(off[0].num());
    t.end = uint64_t(off[1].num());
    out.emplace_back(name, std::move(t));
  }
  return out;
}

SafetensorsSet::SafetensorsSet(const std::string& snapshot_dir) {
  std::ifstream ixf(snapshot_dir + "model.safetensors.index.json");
  if (!ixf)
    throw std::runtime_error("no model.safetensors.index.json in " + snapshot_dir +
                             " (single-file checkpoints are out of scope)");
  std::stringstream ss;
  ss << ixf.rdbuf();
  common::json::Value ix = common::json::parse(ss.str());
  const auto& wm = ix.at("weight_map").obj();

  std::map<std::string, uint32_t> file_id;
  std::map<std::string, std::map<std::string, TensorInfo>> per_file;  // file -> header map
  for (const auto& [tensor, filev] : wm) {
    const std::string& fname = filev.str();
    if (file_id.find(fname) == file_id.end()) {
      file_id[fname] = uint32_t(files_.size());
      files_.emplace_back(snapshot_dir + fname);
      names_.push_back(fname);
      auto entries = parse_header(files_.back().data(), files_.back().size());
      uint64_t hlen = 0;
      std::memcpy(&hlen, files_.back().data(), 8);
      data_start_.push_back(size_t(8 + hlen));
      auto& m = per_file[fname];
      for (auto& [n, t] : entries) m.emplace(n, std::move(t));
    }
    auto& header = per_file[fname];
    auto it = header.find(tensor);
    if (it == header.end())
      throw std::runtime_error("index names tensor '" + tensor + "' in " + fname +
                               " but the file's header has no such tensor");
    TensorInfo t = it->second;
    t.file = file_id[fname];
    tensors_.emplace(tensor, std::move(t));   // index is the manifest: one entry per name
  }
}

const uint8_t* SafetensorsSet::data(const TensorInfo& t) const {
  return files_[t.file].data() + data_start_[t.file] + t.begin;
}

}  // namespace loader
```

`src/loader/CMakeLists.txt`:
```cmake
add_library(b70_loader STATIC snapshot.cc safetensors.cc)
target_include_directories(b70_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_loader PUBLIC b70_l0)
```
Root `CMakeLists.txt`: `add_subdirectory(src/loader)` after `src/common`.

- [ ] **Step 4: Run** - `tools/box.sh test safetensors_test` → OK; full suite still green.

- [ ] **Step 5: Also verify against the real checkpoint on the box** (no new
  test file - a one-off command; the durable checkpoint test comes in Task 6):

Run: `tools/box.sh run "./build/tests/safetensors_test"` (already run by ctest) and
`tools/box.sh run "ls ~/.cache/huggingface/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/"`
Expected: the snapshot revision `2a9077667e28aa53e61d91bdee5d7962e8674668` exists (Task 6's test hardcodes the repo id, not the revision).

- [ ] **Step 6: Commit**

```bash
git add src/loader tests/loader/safetensors_test.cc tests/CMakeLists.txt CMakeLists.txt
git commit -m "feat(loader): snapshot resolution (HF cache, no download) + sharded safetensors mmap with index manifest"
```

---

### Task 3: Quant classification, invariant asserts, shared repack functions

**Files:**
- Create: `src/loader/quant.h`, `src/loader/quant.cc`
- Create: `src/common/repack.h` (free functions), Modify: `src/common/int4.h`
  and `src/common/bf16.h` (their `tiled()` / `from_rowmajor` become thin
  wrappers over the free functions - single source of truth for the tile math)
- Modify: `src/loader/CMakeLists.txt` (add `quant.cc`)
- Create: `tests/loader/quant_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces (namespace `loader`):**
- `struct QuantConfig { uint32_t bits, group_size; bool sym, desc_act; size_t dynamic_rule_count; }`
  - `static QuantConfig parse(const common::json::Value& config_json)` reads
  `quantization_config` accepting both metadata spellings (doc 02): keys
  `bits/group_size/sym/desc_act` (`quant_method` may be `"gptq"` or
  `"auto-round"` - not trusted, only recorded); `dynamic` is an object whose
  keys must ALL start with `"-:"` (throw on any `"+:"` - that is the broken-
  checkpoint case from BENCHMARKS.md); `dynamic_rule_count` = its size.
  Throws unless `bits == 4 && group_size == 64 && sym && !desc_act`.
- `enum class WKind { Int4, Bf16 };`
- `struct LinearSrc { WKind kind; uint32_t K, N; const uint32_t* qweight; const uint16_t* scales; const uint16_t* weight; std::string name; }`
  - `static LinearSrc classify(const SafetensorsSet&, const std::string& prefix)`:
  int4 if `<prefix>.qweight` exists (then `.scales` must too; `K = shape[0]*8`
  of qweight, `N = shape[1]`; dtypes I32/F16 asserted), bf16 if
  `<prefix>.weight` exists (HF row-major `[N][K]`: `N = shape[0], K = shape[1]`,
  dtype BF16), else throws naming the prefix. Suffix presence decides - the
  label never does (doc 02).
- `void assert_quant_invariants(const SafetensorsSet&, const QuantConfig&)` -
  scans EVERY `.qzeros` tensor word-by-word for `0x77777777` and every
  `.g_idx` for the identity `g_idx[k] == k/64`; on failure throws with tensor
  name, index, and the offending value. (~200 MB of reads; runs once at load.)
- **Namespace `common` (in `repack.h`):**
  - `void repack_int4_layout1(const uint32_t* qweight, const uint16_t* scales, uint32_t K, uint32_t N, uint32_t* out)` - exactly `Int4Gptq::tiled()`'s math (tile `(n_tile*G + g) * 136`; 128 u32 nibbles `[j*16+l]`; scales two-per-u32, even lane low half).
  - `struct ColSource { const uint32_t* qweight; const uint16_t* scales; uint32_t n; };`
    `void repack_int4_layout1_cols(uint32_t K, uint32_t N_total, const std::vector<ColSource>& cols, uint32_t* out)` - same tiles, but column `n` of the output reads `cols[n]` (used for fusion; `cols.size() == N_total`; all parts must share `K`).
  - `std::vector<ColSource> concat_cols(std::initializer_list<std::pair<const LinearParts*, ...>>)` - replaced by two small builders (see below) to stay concrete:
    `std::vector<ColSource> cols_concat(const std::vector<std::pair<const uint32_t*, std::pair<const uint16_t*, uint32_t>>>& parts)` is awkward - instead:
    `struct Part { const uint32_t* qweight; const uint16_t* scales; uint32_t N; };`
    `std::vector<ColSource> cols_concat(const std::vector<Part>& parts)` - parts laid end to end;
    `std::vector<ColSource> cols_interleave16(const Part& a, const Part& b)` - `a[0:16] b[0:16] a[16:32] …` (requires `a.N == b.N`, `a.N % 16 == 0`); used for gate‖up so the lane that holds `gate[n]` finds `up[n]` at a fixed +16-column offset (spec §6.4).
  - `void repack_bf16_tiled(const uint16_t* w_rowmajor /*[N][K]*/, uint32_t K, uint32_t N, uint16_t* out)` - exactly `Bf16Tiled::from_rowmajor`'s math.
  - `Int4Gptq::tiled()` and `Bf16Tiled::from_rowmajor` now call these (behaviour identical; the existing gemv/gemv_bf16 tests are the regression net).

- [ ] **Step 1: Write the failing test `tests/loader/quant_test.cc`**

```cpp
#include <cstdio>
#include <vector>
#include "check.h"
#include "common/int4.h"
#include "common/repack.h"
#include "loader/quant.h"
#include "common/json.h"

int main() {
  using common::json::parse;
  // QuantConfig: accepts the checkpoint's spelling, rejects "+:" rules.
  auto ok = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
      "desc_act":false,"quant_method":"gptq","dynamic":{"-:.*mtp.*":{}}}})");
  loader::QuantConfig qc = loader::QuantConfig::parse(ok);
  CHECK_EQ(qc.group_size, uint32_t(64));
  CHECK_EQ(qc.dynamic_rule_count, size_t(1));
  bool threw = false;
  try {
    auto bad = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "desc_act":false,"dynamic":{"+:.*mtp.*":{}}}})");
    loader::QuantConfig::parse(bad);
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try {
    auto g128 = parse(R"({"quantization_config":{"bits":4,"group_size":128,"sym":true,"desc_act":false}})");
    loader::QuantConfig::parse(g128);
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);

  // repack_int4_layout1 == Int4Gptq::tiled() (after the refactor this is
  // one implementation; the check pins the wrapper wiring).
  common::Int4Gptq w = common::Int4Gptq::random(128, 64, 7);
  std::vector<uint32_t> a = w.tiled();
  std::vector<uint32_t> b(a.size());
  common::repack_int4_layout1(w.qweight.data(), w.scales.data(), w.K, w.N, b.data());
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK_EQ(a[i], b[i]);

  // cols_concat: two parts side by side == one repack of manually concatenated weights.
  common::Int4Gptq p1 = common::Int4Gptq::random(128, 32, 8);
  common::Int4Gptq p2 = common::Int4Gptq::random(128, 32, 9);
  common::Int4Gptq cat; cat.K = 128; cat.N = 64;
  cat.qweight.resize(size_t(cat.K / 8) * cat.N);
  cat.scales.resize(size_t(cat.K / 64) * cat.N);
  for (uint32_t r = 0; r < cat.K / 8; ++r)
    for (uint32_t n = 0; n < 64; ++n)
      cat.qweight[size_t(r) * 64 + n] =
          n < 32 ? p1.qweight[size_t(r) * 32 + n] : p2.qweight[size_t(r) * 32 + (n - 32)];
  for (uint32_t g = 0; g < cat.K / 64; ++g)
    for (uint32_t n = 0; n < 64; ++n)
      cat.scales[size_t(g) * 64 + n] =
          n < 32 ? p1.scales[size_t(g) * 32 + n] : p2.scales[size_t(g) * 32 + (n - 32)];
  std::vector<uint32_t> want = cat.tiled();
  std::vector<uint32_t> got(want.size());
  std::vector<common::ColSource> cols = common::cols_concat(
      {{p1.qweight.data(), p1.scales.data(), 32}, {p2.qweight.data(), p2.scales.data(), 32}});
  common::repack_int4_layout1_cols(128, 64, cols, got.data());
  for (size_t i = 0; i < want.size(); ++i) CHECK_EQ(want[i], got[i]);

  // cols_interleave16: block n_tile 0 is a[0:16], tile 1 is b[0:16], tile 2 a[16:32]...
  std::vector<common::ColSource> il = common::cols_interleave16(
      {p1.qweight.data(), p1.scales.data(), 32}, {p2.qweight.data(), p2.scales.data(), 32});
  CHECK_EQ(il.size(), size_t(64));
  CHECK_EQ(il[0].qweight, p1.qweight.data());  CHECK_EQ(il[0].n, uint32_t(0));
  CHECK_EQ(il[16].qweight, p2.qweight.data()); CHECK_EQ(il[16].n, uint32_t(0));
  CHECK_EQ(il[32].qweight, p1.qweight.data()); CHECK_EQ(il[32].n, uint32_t(16));
  CHECK_EQ(il[63].qweight, p2.qweight.data()); CHECK_EQ(il[63].n, uint32_t(31));
  std::puts("quant_test OK");
  return 0;
}
```

`tests/CMakeLists.txt`: mirror `safetensors_test` (`quant_test`, links `b70_loader`).

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `src/common/repack.h`**

```cpp
#pragma once
// The canonical on-device layouts as free functions - the single source of
// truth shared by the test helpers (Int4Gptq::tiled, Bf16Tiled) and the
// loader, which repacks per-linear straight from the mmapped checkpoint.
// Layout definitions: docs/12-kernels.md.
#include <cstdint>
#include <vector>

namespace common {

// int4 layout 1: per (n_tile of 16, k_group of 64) one 136-u32 tile:
// 128 u32 nibbles [k_octet j][lane l] = qweight word (g*8+j, n_tile*16+l),
// then 16 f16 scales packed two per u32 (even lane in the low half).
inline void repack_int4_layout1(const uint32_t* qweight, const uint16_t* scales, uint32_t K,
                                uint32_t N, uint32_t* out) {
  const uint32_t G = K / 64, NT = N / 16;
  for (uint32_t nt = 0; nt < NT; ++nt)
    for (uint32_t g = 0; g < G; ++g) {
      uint32_t* tile = out + (size_t(nt) * G + g) * 136;
      for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t l = 0; l < 16; ++l)
          tile[j * 16 + l] = qweight[size_t(g * 8 + j) * N + nt * 16 + l];
      for (uint32_t l = 0; l < 16; l += 2)
        tile[128 + l / 2] = uint32_t(scales[size_t(g) * N + nt * 16 + l]) |
                            (uint32_t(scales[size_t(g) * N + nt * 16 + l + 1]) << 16);
    }
}

// A fused linear's output column n sourced from one part's column.
struct ColSource {
  const uint32_t* qweight;   // the part's [K/8][N_part] words
  const uint16_t* scales;    // the part's [K/64][N_part] f16
  uint32_t n;                // column within the part
  uint32_t n_part = 0;      // the part's N (stride for row indexing)
};

struct Part {
  const uint32_t* qweight;
  const uint16_t* scales;
  uint32_t N;
};

inline std::vector<ColSource> cols_concat(const std::vector<Part>& parts) {
  std::vector<ColSource> cols;
  for (const Part& p : parts)
    for (uint32_t n = 0; n < p.N; ++n) cols.push_back({p.qweight, p.scales, n, p.N});
  return cols;
}

// gate||up interleaved in 16-column blocks: a[0:16] b[0:16] a[16:32] b[16:32]...
inline std::vector<ColSource> cols_interleave16(const Part& a, const Part& b) {
  std::vector<ColSource> cols;
  for (uint32_t base = 0; base < a.N; base += 16) {
    for (uint32_t l = 0; l < 16; ++l) cols.push_back({a.qweight, a.scales, base + l, a.N});
    for (uint32_t l = 0; l < 16; ++l) cols.push_back({b.qweight, b.scales, base + l, b.N});
  }
  return cols;
}

inline void repack_int4_layout1_cols(uint32_t K, uint32_t N_total,
                                     const std::vector<ColSource>& cols, uint32_t* out) {
  const uint32_t G = K / 64, NT = N_total / 16;
  for (uint32_t nt = 0; nt < NT; ++nt)
    for (uint32_t g = 0; g < G; ++g) {
      uint32_t* tile = out + (size_t(nt) * G + g) * 136;
      for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t l = 0; l < 16; ++l) {
          const ColSource& c = cols[nt * 16 + l];
          tile[j * 16 + l] = c.qweight[size_t(g * 8 + j) * c.n_part + c.n];
        }
      for (uint32_t l = 0; l < 16; l += 2) {
        const ColSource& c0 = cols[nt * 16 + l];
        const ColSource& c1 = cols[nt * 16 + l + 1];
        tile[128 + l / 2] = uint32_t(c0.scales[size_t(g) * c0.n_part + c0.n]) |
                            (uint32_t(c1.scales[size_t(g) * c1.n_part + c1.n]) << 16);
      }
    }
}

// bf16 tiles [n_tile][k_octet][8 k][16 n] from HF row-major [N][K].
inline void repack_bf16_tiled(const uint16_t* w, uint32_t K, uint32_t N, uint16_t* out) {
  const uint32_t K8 = K / 8;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t k = 0; k < K; ++k)
      out[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16] = w[size_t(n) * K + k];
}

}  // namespace common
```

Refactors (behaviour-preserving; existing gemv/gemv_bf16 tests are the net):
`Int4Gptq::tiled()` body becomes `std::vector<uint32_t> t(size_t(N/16) * (K/64) * kTileU32); repack_int4_layout1(qweight.data(), scales.data(), K, N, t.data()); return t;` (include `common/repack.h`); `Bf16Tiled::from_rowmajor` likewise calls `repack_bf16_tiled`.

- [ ] **Step 4: Implement `src/loader/quant.h` / `quant.cc`**

`quant.h`:
```cpp
#pragma once
#include <cstdint>
#include <string>
#include "common/json.h"
#include "loader/safetensors.h"

namespace loader {

struct QuantConfig {
  uint32_t bits = 0, group_size = 0;
  bool sym = false, desc_act = true;
  size_t dynamic_rule_count = 0;
  // Throws unless bits==4, group_size==64, sym, !desc_act; throws on any
  // "+:" dynamic rule (the known-broken checkpoint pattern, BENCHMARKS.md).
  static QuantConfig parse(const common::json::Value& config_json);
};

enum class WKind { Int4, Bf16 };

struct LinearSrc {
  WKind kind;
  uint32_t K = 0, N = 0;
  const uint32_t* qweight = nullptr;  // Int4: [K/8][N]
  const uint16_t* scales = nullptr;   // Int4: [K/64][N] f16
  const uint16_t* weight = nullptr;   // Bf16: [N][K] row-major
  std::string name;
  // Suffix presence decides the kind; labels are never trusted (doc 02).
  static LinearSrc classify(const SafetensorsSet& set, const std::string& prefix);
};

// Scans every .qzeros word (must be 0x77777777 - GPTQ v1 stores zero-1, i.e.
// symmetric zero point 8) and every .g_idx (identity under g64). Throws with
// tensor name, element index and value on the first violation.
void assert_quant_invariants(const SafetensorsSet& set);

}  // namespace loader
```

`quant.cc`:
```cpp
#include "loader/quant.h"
#include <stdexcept>

namespace loader {

QuantConfig QuantConfig::parse(const common::json::Value& config_json) {
  const common::json::Value* qcv = config_json.find("quantization_config");
  if (!qcv) throw std::runtime_error("config.json has no quantization_config");
  QuantConfig q;
  q.bits = uint32_t(qcv->at("bits").num());
  q.group_size = uint32_t(qcv->at("group_size").num());
  q.sym = qcv->at("sym").boolean();
  q.desc_act = qcv->at("desc_act").boolean();
  if (const common::json::Value* dyn = qcv->find("dynamic")) {
    for (const auto& [rule, unused] : dyn->obj()) {
      (void)unused;
      if (rule.rfind("-:", 0) != 0)
        throw std::runtime_error(
            "quantization_config.dynamic has a non-exclusion rule '" + rule +
            "' - this is the broken-MTP-config pattern (BENCHMARKS.md); fix the checkpoint");
      ++q.dynamic_rule_count;
    }
  }
  if (q.bits != 4 || q.group_size != 64 || !q.sym || q.desc_act)
    throw std::runtime_error("unsupported quantization: need int4 g64 sym desc_act=false, got bits=" +
                             std::to_string(q.bits) + " g=" + std::to_string(q.group_size) +
                             " sym=" + (q.sym ? "true" : "false") +
                             " desc_act=" + (q.desc_act ? "true" : "false"));
  return q;
}

LinearSrc LinearSrc::classify(const SafetensorsSet& set, const std::string& prefix) {
  const auto& ts = set.tensors();
  auto qw = ts.find(prefix + ".qweight");
  if (qw != ts.end()) {
    auto sc = ts.find(prefix + ".scales");
    if (sc == ts.end()) throw std::runtime_error(prefix + ": qweight without scales");
    if (qw->second.dtype != "I32") throw std::runtime_error(prefix + ".qweight dtype " + qw->second.dtype);
    if (sc->second.dtype != "F16") throw std::runtime_error(prefix + ".scales dtype " + sc->second.dtype);
    LinearSrc s;
    s.kind = WKind::Int4;
    s.K = uint32_t(qw->second.shape[0]) * 8;
    s.N = uint32_t(qw->second.shape[1]);
    if (sc->second.shape[0] != s.K / 64 || sc->second.shape[1] != s.N)
      throw std::runtime_error(prefix + ".scales shape mismatch");
    s.qweight = reinterpret_cast<const uint32_t*>(set.data(qw->second));
    s.scales = reinterpret_cast<const uint16_t*>(set.data(sc->second));
    s.name = prefix;
    return s;
  }
  auto w = ts.find(prefix + ".weight");
  if (w != ts.end()) {
    if (w->second.dtype != "BF16") throw std::runtime_error(prefix + ".weight dtype " + w->second.dtype);
    LinearSrc s;
    s.kind = WKind::Bf16;
    s.N = uint32_t(w->second.shape[0]);
    s.K = uint32_t(w->second.shape[1]);
    s.weight = reinterpret_cast<const uint16_t*>(set.data(w->second));
    s.name = prefix;
    return s;
  }
  throw std::runtime_error("no qweight or weight for linear '" + prefix + "'");
}

void assert_quant_invariants(const SafetensorsSet& set) {
  for (const auto& [name, t] : set.tensors()) {
    if (name.size() > 7 && name.compare(name.size() - 7, 7, ".qzeros") == 0) {
      const uint32_t* p = reinterpret_cast<const uint32_t*>(set.data(t));
      size_t n = set.bytes(t) / 4;
      for (size_t i = 0; i < n; ++i)
        if (p[i] != 0x77777777u)
          throw std::runtime_error(name + "[" + std::to_string(i) + "] = " +
                                   std::to_string(p[i]) + ", expected 0x77777777 (sym zero-point 8)");
    } else if (name.size() > 6 && name.compare(name.size() - 6, 6, ".g_idx") == 0) {
      const int32_t* p = reinterpret_cast<const int32_t*>(set.data(t));
      size_t n = set.bytes(t) / 4;
      for (size_t i = 0; i < n; ++i)
        if (p[i] != int32_t(i / 64))
          throw std::runtime_error(name + "[" + std::to_string(i) + "] = " +
                                   std::to_string(p[i]) + ", expected identity k/64");
    }
  }
}

}  // namespace loader
```

`src/loader/CMakeLists.txt`: add `quant.cc`.

- [ ] **Step 5: Run** - `tools/box.sh test quant_test` OK, then the FULL suite
  (the `tiled()`/`from_rowmajor` refactor must leave `gemv_test`,
  `gemv_bf16_test` green).

- [ ] **Step 6: Commit**

```bash
git add src/common/repack.h src/common/int4.h src/common/bf16.h src/loader tests/loader/quant_test.cc tests/CMakeLists.txt
git commit -m "feat(loader): quant config asserts + classification; repack as shared free functions with fusion column maps"
```

---

### Task 4: Oracle dequant fixture - the nibble contract across languages

**Files:**
- Create: `tools/oracle/dequant.py`
- Create: `tests/loader/dequant_fixture_test.cc`; Modify: `tests/CMakeLists.txt`
- Create (generated, committed): `tests/golden/dequant_fixture.safetensors` (~60 KB)
- Modify: `.gitignore` (allow committed fixtures: add under the model-weights
  section `!tests/golden/*.safetensors  # small committed test fixtures only`)
- Modify: `tools/box.sh` (new subcommand `pull <path>`: `rsync -az "$BOX:$REMOTE_DIR/<path>" "<path>"` - used to bring generated fixtures back for committing)

**Interfaces:**
- `tools/oracle/dequant.py`:
  - `dequant_gptq(qweight_i32, scales_f16, group_size=64) -> torch.bfloat16 [K, N]`
    - unpack nibbles (`k = row*8 + i`, low nibble first), `q - 8`, multiply in
    **fp32**, cast **once** to bf16. This exact rounding path is what makes the
    C++ side bit-comparable (both round the same fp32 product once, RNE).
  - CLI: `python3 tools/oracle/dequant.py --write-fixture <out.safetensors>`
    writes tensors `qweight` I32 `[16, 32]` (K=128, N=32), `scales` F16
    `[2, 32]`, `dequant` BF16 `[128, 32]`, generated from a fixed torch seed
    (`torch.manual_seed(1234)`).
- `tests/loader/dequant_fixture_test.cc` - maps the fixture with
  `loader::MappedFile` + `SafetensorsSet::parse_header`, builds
  `common::Int4Gptq` from `qweight`/`scales`, and checks
  `common::f32_to_bf16(w.at(k, n)) == dequant[k][n]` **bit-exact** for every
  element. A single mismatch prints `(k, n)`, both bit patterns, and the
  nibble/scale involved.

- [ ] **Step 1: Write `tools/oracle/dequant.py`**

```python
#!/usr/bin/env python3
"""The reference meaning of the int4 GPTQ bits, shared with the C++ loader.

dequant_gptq computes w = (q - 8) * scale with the product in fp32 and ONE
cast to bf16 - the same single-rounding path as common::f32_to_bf16 on the
C++ side, which is what makes the fixture bit-exact comparable.

Conventions (docs/02, docs/03): qweight int32 [K/8, N], nibble i of word
(r, n) is k = r*8 + i (low nibble first); scales f16 [K/64, N]; symmetric,
zero point 8 (the checkpoint's v1 qzeros store 7 == zero-1; dropped).

Usage: dequant.py --write-fixture tests/golden/dequant_fixture.safetensors
"""
import argparse

import torch
from safetensors.torch import save_file


def dequant_gptq(qweight: torch.Tensor, scales: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    assert qweight.dtype == torch.int32 and scales.dtype == torch.float16
    rows, n = qweight.shape
    k = rows * 8
    shifts = torch.arange(8, dtype=torch.int32) * 4                     # [8]
    nibbles = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF      # [K/8, 8, N]
    q = nibbles.reshape(k, n)                                           # k = r*8 + i
    w32 = (q.to(torch.float32) - 8.0) * scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    return w32.to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-fixture", required=True)
    args = ap.parse_args()
    torch.manual_seed(1234)
    K, N = 128, 32
    qweight = torch.randint(-(2**31), 2**31 - 1, (K // 8, N), dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(K // 64, N, dtype=torch.float32) * 0.06 + 0.02).to(torch.float16)
    fx = {"qweight": qweight, "scales": scales, "dequant": dequant_gptq(qweight, scales)}
    save_file(fx, args.write_fixture)
    print(f"wrote {args.write_fixture}: qweight {tuple(qweight.shape)}, "
          f"scales {tuple(scales.shape)}, dequant {tuple(fx['dequant'].shape)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `pull` to `tools/box.sh`** (new case-arm, mirroring `run`):

```bash
  pull)  shift; rsync -az "$BOX:$REMOTE_DIR/$1" "$1" ;;
```
(usage line updated to `sync|build|test [regex]|run <cmd...>|pull <path>`.)

- [ ] **Step 3: Generate the fixture in the container on the box**

```bash
tools/box.sh sync
tools/box.sh run "mkdir -p tests/golden && docker run --rm --entrypoint python3 \
  -v ~/b70-inference-server:/ws -w /ws \
  vllm-xpu-env-next-p314-t214-vxkp0:latest \
  tools/oracle/dequant.py --write-fixture tests/golden/dequant_fixture.safetensors"
tools/box.sh pull tests/golden/dequant_fixture.safetensors
```
Expected: the `wrote …` line; the local file exists, ~60 KB.

- [ ] **Step 4: Write the failing test `tests/loader/dequant_fixture_test.cc`**

```cpp
// The dequant contract across languages: the oracle's bf16 dequant of a
// random packed tensor must equal the C++ side bit-for-bit. If this fails,
// the loader and the golden tensors disagree about what the nibbles mean and
// NOTHING downstream can be trusted.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "loader/safetensors.h"

int main(int argc, char** argv) {
  std::string path = argc > 1 ? argv[1] : "tests/golden/dequant_fixture.safetensors";
  loader::MappedFile f(path);
  auto entries = loader::SafetensorsSet::parse_header(f.data(), f.size());
  uint64_t hlen = 0;
  std::memcpy(&hlen, f.data(), 8);
  const uint8_t* base = f.data() + 8 + hlen;

  const loader::TensorInfo *qw = nullptr, *sc = nullptr, *dq = nullptr;
  for (auto& [name, t] : entries) {
    if (name == "qweight") qw = &t;
    else if (name == "scales") sc = &t;
    else if (name == "dequant") dq = &t;
  }
  CHECK(qw && sc && dq);
  common::Int4Gptq w;
  w.K = uint32_t(qw->shape[0]) * 8;
  w.N = uint32_t(qw->shape[1]);
  const uint32_t* qp = reinterpret_cast<const uint32_t*>(base + qw->begin);
  const uint16_t* sp = reinterpret_cast<const uint16_t*>(base + sc->begin);
  w.qweight.assign(qp, qp + size_t(w.K / 8) * w.N);
  w.scales.assign(sp, sp + size_t(w.K / 64) * w.N);
  const uint16_t* golden = reinterpret_cast<const uint16_t*>(base + dq->begin);

  size_t checked = 0;
  for (uint32_t k = 0; k < w.K; ++k)
    for (uint32_t n = 0; n < w.N; ++n, ++checked) {
      uint16_t ours = common::f32_to_bf16(w.at(k, n));
      uint16_t theirs = golden[size_t(k) * w.N + n];
      if (ours != theirs) {
        std::fprintf(stderr, "dequant mismatch at k=%u n=%u: ours=0x%04X oracle=0x%04X\n",
                     k, n, ours, theirs);
        return 1;
      }
    }
  CHECK_EQ(checked, size_t(w.K) * w.N);
  std::printf("dequant_fixture_test OK (%zu elements bit-exact)\n", checked);
  return 0;
}
```

`tests/CMakeLists.txt` add (the fixture path is relative to the repo root -
pass it explicitly so ctest's working directory doesn't matter):
```cmake
add_executable(dequant_fixture_test loader/dequant_fixture_test.cc)
target_include_directories(dequant_fixture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dequant_fixture_test PRIVATE b70_loader)
add_test(NAME dequant_fixture_test COMMAND dequant_fixture_test ${CMAKE_SOURCE_DIR}/tests/golden/dequant_fixture.safetensors)
```

- [ ] **Step 5: `.gitignore`** - in the "Model weights" section add, directly
  under `*.safetensors`:
```
!tests/golden/*.safetensors   # small committed test fixtures only
```

- [ ] **Step 6: Run** - `tools/box.sh test dequant_fixture_test` → bit-exact OK;
  full suite green.

- [ ] **Step 7: Commit**

```bash
git add tools/oracle/dequant.py tools/box.sh tests/loader/dequant_fixture_test.cc tests/CMakeLists.txt .gitignore
git add -f tests/golden/dequant_fixture.safetensors
git commit -m "feat(oracle): dequant reference + committed fixture; C++/Python nibble contract bit-exact"
```

---

### Task 5: `src/model/qwen35.*` - the model as data

**Files:**
- Create: `src/model/qwen35.h`, `src/model/qwen35.cc`, `src/model/CMakeLists.txt`
- Modify: root `CMakeLists.txt` (add `add_subdirectory(src/model)` after `src/loader`)
- Create: `tests/model/qwen35_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces (namespace `model`):** - this is the table plan 3's capture code
walks; it knows nothing of L0 or kernels.

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace model {

// One GEMV's compiled configuration. layout/S are measured values
// (docs/probe-gemv-2026-08-24.md); the table below is the single place they
// live. Flipping a row to layout 0 is the phase-1 tuning knob recorded in
// spec §6.3 - mechanism supports it, default is all layout 1.
struct GemvShape { uint32_t K, N, S, layout; };

enum class LinearId { QkvZ, AB, OutProj, GateUp, Down, Qkv, OProj, LmHead };

// How a fused device weight is assembled from checkpoint tensors.
enum class Fuse { Single, Concat, Interleave16 };
enum class WeightKind { Int4, Bf16 };

struct FusedLinear {
  LinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // checkpoint prefixes, in order; layer-relative
  uint32_t pad_n = 0;               // zero-pad N to shape.N (a||b: 96 -> 128)
};

enum class LayerKind { GDN, FA };

struct LayerDesc {
  uint32_t index;
  LayerKind kind;
  std::vector<FusedLinear> linears;       // in per-token execution order
  std::vector<std::string> small_tensors; // norms, conv1d, A_log, dt_bias (layer-relative)
};

struct Qwen35 {
  static constexpr uint32_t kLayers = 64, kHidden = 5120, kIntermediate = 17408;
  static constexpr uint32_t kVocab = 248320, kVocabUsed = 248077;
  static constexpr uint32_t kGdnKHeads = 16, kGdnVHeads = 48, kGdnHeadDim = 128;
  static constexpr uint32_t kFaQHeads = 24, kFaKvHeads = 4, kFaHeadDim = 256;
  static constexpr uint32_t kRotaryDim = 64;
  static constexpr double kRopeTheta = 1e7;

  static bool is_fa(uint32_t layer) { return layer % 4 == 3; }
  static const GemvShape& shape(LinearId id);         // the per-shape table
  static std::vector<LayerDesc> layers();             // all 64, fully populated
  // Checkpoint-name helpers ("model.language_model." prefix already stripped
  // by the loader's name mapping): e.g. layer_prefix(5) == "layers.5."
  static std::string layer_prefix(uint32_t layer);
};

}  // namespace model
```

`qwen35.cc` populates, with these exact values:

| LinearId | K | N | S | layout | kind | fuse | parts (layer-relative) |
|---|---|---|---|---|---|---|---|
| QkvZ | 5120 | 16384 | 1 | 1 | Int4 | Concat | `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z` |
| AB | 5120 | 128 | 1 | bf16-tiled | Bf16 | Concat, `pad_n` from 96 | `linear_attn.in_proj_a`, `linear_attn.in_proj_b` |
| OutProj | 6144 | 5120 | 16 | 1 | Int4 | Single | `linear_attn.out_proj` |
| GateUp | 5120 | 34816 | 4 | 1 | Int4 | Interleave16 | `mlp.gate_proj`, `mlp.up_proj` |
| Down | 17408 | 5120 | 16 | 1 | Int4 | Single | `mlp.down_proj` |
| Qkv | 5120 | 14336 | 1 | 1 | Int4 | Concat | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj` |
| OProj | 6144 | 5120 | 16 | 1 | Int4 | Single | `self_attn.o_proj` |
| LmHead | 5120 | 248320 | 1 | bf16-tiled | Bf16 | Single | *(top-level)* `lm_head` |

GDN layer linears: QkvZ, AB, OutProj, GateUp, Down; small tensors
`input_layernorm.weight`, `post_attention_layernorm.weight`,
`linear_attn.conv1d.weight`, `linear_attn.A_log`, `linear_attn.dt_bias`,
`linear_attn.norm.weight`. FA layer: Qkv, OProj, GateUp, Down; small tensors
the two layernorms plus `self_attn.q_norm.weight`, `self_attn.k_norm.weight`.

- [ ] **Step 1: Failing test `tests/model/qwen35_test.cc`**

```cpp
#include <cstdio>
#include "check.h"
#include "model/qwen35.h"

int main() {
  using model::LayerKind;
  using model::LinearId;
  using model::Qwen35;
  auto layers = Qwen35::layers();
  CHECK_EQ(layers.size(), size_t(64));
  size_t gdn = 0, fa = 0;
  for (const auto& l : layers) (l.kind == LayerKind::GDN ? gdn : fa)++;
  CHECK_EQ(gdn, size_t(48));
  CHECK_EQ(fa, size_t(16));
  CHECK(layers[3].kind == LayerKind::FA);
  CHECK(layers[62].kind == LayerKind::GDN);

  // Every GEMV shape is kernel-legal: N%64==0, K%64==0, S divides K/64.
  for (LinearId id : {LinearId::QkvZ, LinearId::AB, LinearId::OutProj, LinearId::GateUp,
                      LinearId::Down, LinearId::Qkv, LinearId::OProj, LinearId::LmHead}) {
    const auto& s = Qwen35::shape(id);
    CHECK_EQ(s.N % 64, uint32_t(0));
    CHECK_EQ(s.K % 64, uint32_t(0));
    CHECK_EQ((s.K / 64) % s.S, uint32_t(0));
  }
  // The measured table (docs/probe-gemv-2026-08-24.md): spot-check the two
  // that differ from naive expectations.
  CHECK_EQ(Qwen35::shape(LinearId::OutProj).K, uint32_t(6144));
  CHECK_EQ(Qwen35::shape(LinearId::OutProj).S, uint32_t(16));
  CHECK_EQ(Qwen35::shape(LinearId::GateUp).S, uint32_t(4));

  // Fusion wiring for layer 0 (GDN) and 3 (FA).
  CHECK_EQ(layers[0].linears.size(), size_t(5));
  CHECK(layers[0].linears[0].id == LinearId::QkvZ);
  CHECK_EQ(layers[0].linears[0].parts.size(), size_t(2));
  CHECK_EQ(layers[0].linears[1].pad_n, uint32_t(96));   // a||b padded 96 -> 128
  CHECK_EQ(layers[3].linears.size(), size_t(4));
  CHECK(layers[3].linears[0].id == LinearId::Qkv);
  CHECK_EQ(Qwen35::layer_prefix(5), std::string("layers.5."));
  std::puts("qwen35_test OK");
  return 0;
}
```

CMake: mirror `json_test` (link `b70_model`; `src/model/CMakeLists.txt` is
`add_library(b70_model STATIC qwen35.cc)` + `target_include_directories(... PUBLIC ${CMAKE_SOURCE_DIR}/src)`).

- [ ] **Step 2: Run to verify it fails; implement `qwen35.cc` from the tables above; run to green.**
- [ ] **Step 3: Full suite; commit** - `feat(model): qwen3_5 description as data - layers, fusion table, measured (layout, S) per shape`

---

### Task 6: The loader end-to-end - checkpoint → canonical device buffers

**Files:**
- Create: `src/loader/loader.h`, `src/loader/loader.cc`; Modify: `src/loader/CMakeLists.txt` (add `loader.cc`; link `b70_model`)
- Create: `tests/loader/load_checkpoint_test.cc`; Modify: `tests/CMakeLists.txt`
- Create: `docs/13-loader.md`; Modify: `README.md` (docs table row for 13)

**Interfaces (namespace `loader`):**

```cpp
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/quant.h"
#include "model/qwen35.h"

namespace loader {

struct DeviceWeight {
  l0::Mem mem;                 // the canonical bytes on device
  model::GemvShape shape;      // K, N, S, layout as the kernel variant needs
  model::WeightKind kind;
};

struct SmallTensors {          // per layer: everything that is not a GEMV weight
  l0::Mem norms;               // input_ln(1+w) ‖ post_ln(1+w) bf16 [2][5120]
  l0::Mem gdn;                 // GDN only: conv fp32 [10240][4] ‖ negA fp32 [48]
                               //   ‖ dt_bias fp32 [48] ‖ gated-norm w bf16 [128]
                               //   ‖ q_norm/k_norm (FA: (1+w) bf16 [2][256]) - one
                               //   packed block per layer; offsets in loader.cc
};

struct LoadReport {            // printed by load(); asserted by the test
  size_t int4_bytes = 0, scale_bytes = 0, bf16_linear_bytes = 0;
  size_t embed_bytes = 0, lm_head_bytes = 0, small_bytes = 0, pad_bytes = 0;
  size_t total() const;
  double seconds = 0;
};

struct LoadedModel {
  std::map<std::pair<uint32_t, model::LinearId>, DeviceWeight> linears;  // layer 65535 = top level (lm_head)
  std::vector<SmallTensors> layer_small;   // [64]
  l0::Mem embed;                            // bf16 [248320][5120] row-major (gathered)
  l0::Mem rope;                             // fp32 [max_len][2][32] cos/sin pairs
  LoadReport report;
};

// Loads the qwen3_5 checkpoint at `snapshot_or_repo` (resolve_snapshot rules)
// into ctx's device. Skips model.visual.* and (v1) mtp.*. Strips the
// "model.language_model." prefix so model::Qwen35's layer-relative names bind.
// Asserts quant invariants and the doc-03 shape table; throws by name on any
// mismatch. max_len sizes the RoPE table only (default 16384).
LoadedModel load(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len = 16384);

}  // namespace loader
```

**Implementation requirements (loader.cc):**

1. Name mapping: build a view of `SafetensorsSet::tensors()` with
   `model.language_model.` stripped; `model.visual.*` skipped; `mtp.*` skipped
   (counted, reported); `lm_head.weight` kept top-level. The 9B-era dedup rule
   is already handled by the index manifest (Task 2).
2. For every layer × `FusedLinear`: classify each part with
   `LinearSrc::classify` (which asserts dtypes and scale shapes), verify part
   `K`s agree and the summed/padded `N` equals `shape.N`, then repack into a
   host staging `std::vector` - int4 via `repack_int4_layout1_cols`
   (`cols_concat` / `cols_interleave16` / single-part concat), bf16 via a
   row-major concat staged buffer then `repack_bf16_tiled` (a‖b: two `[48][5120]`
   parts stacked to `[96][5120]`, zero rows to `[128][5120]`, `pad_n` bytes
   counted in `report.pad_bytes`). Upload with one `imm.copy` per linear.
   Staging buffer is reused (max linear = gate‖up ≈ 95 MB).
3. `lm_head`: classify at top level, `repack_bf16_tiled` (2.54 GB staged -
   allocate once), upload. `embed_tokens`: uploaded as-is, row-major.
4. Small tensors per layer, packed per the `SmallTensors` layout: layernorms
   stored as `1 + w` (fp32 add, one RNE cast back to bf16 - same rounding
   discipline as everywhere); `A_log` → `−exp(A_log)` fp32; `dt_bias` fp32;
   `conv1d.weight` BF16 `[10240][1][4]` → fp32 `[10240][4]`; gated norm plain;
   FA `q_norm/k_norm` as `1 + w`. Offsets are compile-time constants in
   loader.cc with a static_assert on the block size.
5. RoPE table: `rope[p][0][i] = cos(p · θ^(−2i/64))`, `[1][i] = sin(…)`,
   `i < 32`, fp32, `p < max_len` (doc 03 "Layer math": text-mode mRoPE ==
   plain RoPE).
6. `assert_quant_invariants` runs before any repack. The doc-03 shape table
   is enforced by construction (`shape.N`/`K` checks above) - a checkpoint
   with different shapes fails by linear name.
7. `LoadReport`: every byte uploaded is attributed to exactly one bucket;
   `load` prints the table and wall time. The **W cross-check**: expected
   read-per-token = `int4 + scale + bf16_linear(only a‖b actually read - but
   report both) + lm_head + small` vs doc 03's 15.519 GB. The load does NOT
   hard-fail on small drift; it must fail if `|total − expected| > 2%` where
   expected = 15.519 GB + pad_bytes + (small-tensor fp32 widening, itemised).
8. No MTP in v1: `mtp.*` tensors counted and reported as skipped.

- [ ] **Step 1: Failing test `tests/loader/load_checkpoint_test.cc`** (ctest label `checkpoint`; runs on the box only)

```cpp
// End-to-end load of the real checkpoint, with a device round-trip proof:
// one tile of layer 0's qkv||z read back from the device must equal the CPU
// repack of the mmapped source - the canonical bytes on device mean exactly
// what the kernels were tested against.
#include <cstdio>
#include <vector>
#include "check.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/qwen35.h"

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ";
  l0::Context ctx(0);
  loader::LoadedModel m = loader::load(ctx, arg);

  // Counts: 64 layers x linears + lm_head.
  CHECK_EQ(m.layer_small.size(), size_t(64));
  CHECK_EQ(m.linears.size(), size_t(48 * 5 + 16 * 4 + 1));

  // Resident total within 2% of W + declared padding/widening.
  const double gb = 1e9;
  std::printf("resident: %.3f GB (int4 %.3f, scales %.3f, lm_head %.3f)\n",
              m.report.total() / gb, m.report.int4_bytes / gb, m.report.scale_bytes / gb,
              m.report.lm_head_bytes / gb);
  CHECK(m.report.int4_bytes > 12.0e9 && m.report.int4_bytes < 12.4e9);
  CHECK(m.report.lm_head_bytes > 2.5e9 && m.report.lm_head_bytes < 2.6e9);

  // Device round-trip: tile 0 of layer 0 QkvZ vs CPU repack of the source.
  std::string snap = loader::resolve_snapshot(arg);
  loader::SafetensorsSet set(snap);
  auto strip = [](const std::string& s) { return "model.language_model." + s; };
  loader::LinearSrc qkv = loader::LinearSrc::classify(set, strip("layers.0.linear_attn.in_proj_qkv"));
  loader::LinearSrc z = loader::LinearSrc::classify(set, strip("layers.0.linear_attn.in_proj_z"));
  std::vector<common::ColSource> cols = common::cols_concat(
      {{qkv.qweight, qkv.scales, qkv.N}, {z.qweight, z.scales, z.N}});
  const uint32_t K = 5120, NT_check = 4;  // first 4 n-tiles
  std::vector<uint32_t> want(size_t(NT_check) * (K / 64) * 136);
  // repack only the first NT_check tiles: build a truncated col list
  std::vector<common::ColSource> cols4(cols.begin(), cols.begin() + NT_check * 16);
  common::repack_int4_layout1_cols(K, NT_check * 16, cols4, want.data());

  const loader::DeviceWeight& dw = m.linears.at({0u, model::LinearId::QkvZ});
  std::vector<uint32_t> got(want.size());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(got.data(), dw.mem.ptr(), got.size() * 4);
  size_t diff = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (want[i] != got[i] && ++diff == 1)
      std::fprintf(stderr, "first mismatch at u32 %zu: want %08X got %08X\n", i, want[i], got[i]);
  CHECK_EQ(diff, size_t(0));

  std::printf("load_checkpoint_test OK (%.1f s load)\n", m.report.seconds);
  return 0;
}
```

Note for the implementer: the device buffer's first `NT_check·(K/64)·136` u32
are exactly n-tiles 0..3 because layout-1 tiles are n_tile-outer - this is
the property the check rides on. If `load` orders differently, the test is
wrong, not the loader; flag it.

`tests/CMakeLists.txt`:
```cmake
add_executable(load_checkpoint_test loader/load_checkpoint_test.cc)
target_include_directories(load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(load_checkpoint_test PRIVATE b70_loader b70_model)
add_test(NAME load_checkpoint_test COMMAND load_checkpoint_test)
set_tests_properties(load_checkpoint_test PROPERTIES LABELS checkpoint TIMEOUT 600)
```

- [ ] **Step 2: Run to verify it fails; implement `loader.cc` per the numbered requirements; iterate on the box.**
  Expected load time: repack ~13 GB single-threaded + 19 GB upload - target
  under 120 s; report the measured figure. If over 5 minutes, report before
  optimising (a simple per-linear `std::async` pool over repack is the known
  lever; do not build it unless needed).

- [ ] **Step 3: `docs/13-loader.md`** - the component's explanation (project
  rule: mechanism in docs). Must cover: snapshot resolution rules (and why no
  download); the index-as-manifest dedup; suffix classification (labels never
  trusted) and every assert with its checkpoint-measured justification
  (0x77777777, g_idx identity, 98 exclusions); the fusion table (why qkv‖z
  concat, why gate‖up interleaves in 16s - the lane offset the kernel wants);
  the `(1+w)` norm baking (doc 03's Gemma-style RMSNorm) and `−exp(A_log)`;
  the per-linear `(layout, S)` table and the layer-0-flip tuning knob (ruling
  2026-08-24); the memory/report table with the measured resident bytes and
  load seconds from Step 2; what is deliberately skipped (visual, mtp) and
  what the 9B taught (dedup, doc 03). Date every measured number 2026-08-24.

- [ ] **Step 4: Full suite on the box including `-L checkpoint`; commit** -
  `feat(loader): end-to-end checkpoint load to canonical device buffers + docs/13`

---

### Task 7: Oracle - `tokenize.py`, `dump.py`, golden prompts

**Files:**
- Create: `tools/oracle/tokenize.py`, `tools/oracle/dump.py`
- Create: `tests/golden/prompts/prose.txt`, `tests/golden/prompts/code.txt`, `tests/golden/prompts/cjk.txt`
- Create: `tools/oracle/README.md` (skeleton; numbers filled by Task 8)

**Interfaces:**
- `tokenize.py <snapshot> encode <text-file>` → prompt token ids, whitespace-separated, on stdout (uses `AutoTokenizer.from_pretrained(snapshot)`; plain `encode`, no chat template - golden prompts are raw text). `decode <ids...>` for the reverse. Runs in the container.
- `dump.py <snapshot> --prompt <ids-file> --out <out.safetensors> [--gen 32] [--max-prompt 64]`:
  1. `cfg = AutoConfig.from_pretrained(snapshot); tc = cfg.get_text_config()`;
     `model = Qwen3_5ForCausalLM(tc)` on CPU, `torch.bfloat16`, eval mode, no `fla` installed (the pure-torch GDN path IS the reference - doc 03).
  2. State dict from the checkpoint via `safetensors` + `dequant.dequant_gptq`:
     for every int4 group (`X.qweight/.scales/.qzeros/.g_idx`) emit `X.weight = dequant_gptq(qweight, scales)`; bf16 tensors pass through; names mapped `model.language_model.` → `model.`; `model.visual.*` and `mtp.*` skipped; `lm_head.weight` kept. `model.load_state_dict(sd, strict=True)` - a missing/unexpected key is a hard error printed in full.
  3. Forward the prompt (≤ `--max-prompt` ids; error if longer) with `use_cache=True`, hooks capturing per layer `i`: `resid.L{i}` (decoder-layer output, all positions, bf16), `mixer.L{i}` (the `linear_attn`/`self_attn` submodule output), `mlp.L{i}`. After the prompt: `gdn_state.L{i}` fp32 and `conv_state.L{i}` from the cache for GDN layers.
  4. Greedy-generate `--gen` tokens one at a time through the same cache; record `logits` (fp32) at every prompt position and every generated step, and `tokens` (the generated ids, int32).
  5. Save everything in ONE safetensors file with exactly those names; print a manifest (name, dtype, shape) and the total size. Self-checks that must abort the run: strict-load success; `tokens` length == `--gen`; every GDN layer contributed a `gdn_state`.
- Prompts: `prose.txt` - 3-4 sentences of plain English; `code.txt` - a ~15-line Python function; `cjk.txt` - a Chinese paragraph plus two emoji (exercises multi-byte tokens and `lm_head`'s tail). Each must tokenize to between 24 and 64 ids (Task 8 verifies and trims).

- [ ] **Step 1: Write the three prompt files and both scripts** per the interfaces above. `dump.py`'s hook capture, cache access (`transformers` 5.15: the returned `past_key_values` holds per-layer `recurrent_states` / `conv_states` for GDN layers - read them defensively and fail with the attribute layout printed if the API differs), and the strict-load mapping are the substance; keep the script single-file and stdlib+torch+transformers+safetensors only.
- [ ] **Step 2: Smoke on the box, shortest prompt** -
```bash
tools/box.sh sync
tools/box.sh run "docker run --rm --entrypoint bash -v ~/b70-inference-server:/ws -w /ws \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  vllm-xpu-env-next-p314-t214-vxkp0:latest -c '
    SNAP=\$(ls -d /root/.cache/huggingface/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/*/ | head -1)
    mkdir -p /ws/oracle-out
    python3 tools/oracle/tokenize.py \"\$SNAP\" encode tests/golden/prompts/prose.txt > /ws/oracle-out/prose.ids
    python3 tools/oracle/dump.py \"\$SNAP\" --prompt /ws/oracle-out/prose.ids --out /ws/oracle-out/prose.golden.safetensors --gen 8'"
```
Expected: the manifest prints with `resid.L0..L63`, `gdn_state` for the 48 GDN layers, `logits`, `tokens` of length 8; wall time and peak RSS reported in your report. **This smoke uses `--gen 8` to bound runtime; Task 8 does the real `--gen 32` runs.** If the dequantised model emits garbage (e.g. `tokens` all one id), STOP and report - do not proceed to Task 8.
- [ ] **Step 3: Commit** - `feat(oracle): tokenizer + golden-dump scripts and prompts` (`oracle-out/` stays on the box; nothing large committed).

### Task 8: Oracle production run + records

**Files:**
- Modify: `tools/oracle/README.md` (fill: exact commands, runtimes, RSS, output manifest sizes, where outputs live on the box)
- Modify: `docs/superpowers/specs/2026-08-22-phase0-decode-core-design.md` §10 (append one line: delivered 2026-08-__, outputs at `~/b70-inference-server/oracle-out/` on the box, per-prompt file names)
- Modify: `docs/BENCHMARKS.md` (one line under Notes: the oracle's greedy tokens for the three prompts exist and are the plan-3 golden gate)

- [ ] **Step 1: Verify prompt token counts** (24-64 ids each, trim the text files if needed, re-commit them), then run all three prompts with `--gen 32` in one container invocation (model loads once; loop the prompts inside a small shell loop or extend `dump.py` to accept repeated `--prompt/--out` pairs - implementer's choice, say which).
- [ ] **Step 2: Sanity-check the outputs** - for each prompt: `tokens` length 32; greedy tokens are not degenerate (not all identical); `resid.L63` finite (no NaN - check with a 5-line python snippet in the same container run and paste the output).
- [ ] **Step 3: A cross-check against vLLM is DEFERRED to plan 3** (the golden test will compare our engine's tokens to these; if both disagree with vLLM later, the dequant convention is the suspect - this is recorded in the README's "trust chain" paragraph, which you must write).
- [ ] **Step 4: Fill the three docs; run the full suite once; commit** - `docs(oracle): golden runs recorded; spec §10 delivered`.

---

## Plan self-review (done at authoring, 2026-08-24)

**Spec coverage.** Spec §1 platform/acquisition → Task 2 (`resolve_snapshot`,
no-download errors); §6.1 naming/manifest → Tasks 2, 6; §6.2 asserts → Tasks
3, 6 (qzeros/g_idx full scan; shapes by construction; `"+:"` dynamic rejected);
§6.3 layouts → Task 3 (shared repack) + Task 5 (per-linear table, default all
layout 1, ruling recorded); §6.4 fusion + derived tensors → Tasks 3 (column
maps), 5 (table), 6 (baking: `1+w`, `−exp(A_log)`, conv fp32, RoPE); §6.5
memory/W assert → Task 6 (LoadReport, ±2% with itemised padding); §6.6 loader
test → Task 4 (dequant fixture, bit-exact) + Task 6 (device round-trip); §7
model description → Task 5; §10 oracle → Tasks 4, 7, 8. NOT in this plan (plan
3): runtime/control block/capture (§8), kernels (§9 remainder: prep, gdn_step,
attention, argmax, embed_gather), CLI/bench (§12), golden comparison tests
(§11's golden_test), replay determinism.

**Placeholder scan.** One deliberate blank: Task 8's spec-§10 line dates
itself `2026-08-__` at run time - the executor fills the real date; no TBD/TODO
otherwise (checked below by grep).

**Type consistency.** `common::ColSource/Part/cols_concat/cols_interleave16/`
`repack_int4_layout1[_cols]/repack_bf16_tiled` are declared once (Task 3) and
consumed in Tasks 3 (test), 6 (loader), with the same signatures;
`loader::resolve_snapshot/MappedFile/SafetensorsSet/TensorInfo` (Task 2) used
in Tasks 3, 4, 6 unchanged; `loader::QuantConfig/LinearSrc/WKind` (Task 3)
used in Task 6; `model::GemvShape/LinearId/FusedLinear/LayerDesc/Qwen35`
(Task 5) consumed by Task 6's `LoadedModel` keys and by the checkpoint test.
`ColSource.n_part` is set by both builders and used by `_cols` - note the
Task 3 test builds `ColSource` aggregates with 3 initialisers in two places
(`{p1.qweight, p1.scales, 32}`) where the 4th member `n_part` defaults to 0 -
**that is a bug in the test as first drafted**: `cols_concat` sets `n_part`
internally, so the two literal-`Part` calls are fine, but the `CHECK_EQ(il[0].qweight, ...)`
block only reads members that are set. The literals passed to `cols_concat`/
`cols_interleave16` are `Part{qweight, scales, N}` aggregates (3 members) -
consistent. No dangling reference: all `ColSource` vectors are consumed before
their source `Int4Gptq` objects go out of scope.

**Known risks for the executor, stated:** (1) `dump.py`'s cache-introspection
(`recurrent_states`/`conv_states` attribute names) is the most likely
transformers-API friction point - the task says fail loudly with the layout
printed, not guess. (2) The checkpoint test's tile-prefix assumption is named
in the task text. (3) Oracle RAM: dequantising to a full bf16 state dict peaks
~60-80 GB - the box has 121 GB; the container must not set a memory limit.
(4) Load-time target (<120 s) is a target, not a gate - measure, report,
escalate only past 5 minutes.

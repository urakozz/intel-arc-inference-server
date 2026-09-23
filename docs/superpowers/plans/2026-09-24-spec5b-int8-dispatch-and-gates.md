# Spec 5b - the int8 prefill backend in the engine, and its gates, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--pp-backend l0-int8` run every int4 prefill linear on plan 5a's h8 path, and decide acceptance with the spec's gates on the real engine.

**Architecture:** A third `PrefillBackend` value, `L0Int8`, counts as L0 everywhere except `pf_linear` and the gate‖up call, which route to `linear_i8` / `linear_i8_silu`. `PrefillEngine` owns the `Int8State`. `Engine::prefill` builds every linear's column scales before the first chunk, so recordings and replay stay one-time-work-free. The gates reuse `prefill_gate_test` (its argv[6] is the backend) and `tools/bench_decode.sh`.

**Tech Stack:** C++20, Level Zero, CMake/ctest, the box.

**Spec:** `docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md` (stages T3, T4). **Requires plan 5a merged.** Gate A4 needs plan 5c's tool-call set.

## Global Constraints

- The CLI spelling is `--pp-backend l0-int8`. It **replaces the spec's `--pp-math` flag**: the backend switch already reaches both CLIs, the engine and every test. The spec is amended in Task 4.
- The default backend stays `l0` until A2, A3, A4, B1 and B2 all pass (Task 4 flips it only then).
- `L0Int8` is L0 for attention, GDN, norms, the head, replay and launch accounting of everything but the linears.
- Launch arithmetic: the int8 path adds exactly one launch (the quantiser) per int4 linear. That is +4 per layer and **+256 per chunk** against L0.
- Idle-box protocol for every timed row (DRM-holder check, device 0, median of 3). Kernel ratios, if any are quoted, come from interleaved pairs.

## Review Focus

- **Every `== PrefillBackend::L0` site.** There are five in production (`step.cc:131`, `attn.h:59/68/112`, `attn.cc:59`, `engine_prefill.cc:89`). Missing one silently runs L0Int8's attention on another path. Pinned in Task 1 by a grep-based test step that requires zero remaining bare comparisons outside `prefill_backend.h`.
- **Scales built mid-recording.** If a weight's scales were first computed inside a recorded chunk, the recording would contain a host round-trip. Pinned in Task 2: `B70_PREFILL_REPLAY=1` on `l0-int8` must run and match the non-replay output bitwise.
- **Multi-chunk prompts.** The quantiser scratch is reused per linear, so chunk 2 must not see chunk 1's scales. Pinned by Task 3's `long` (2820-id, two-chunk) gate registration.
- **Chunk widths that are not multiples of 256.** Pinned by Task 3's registration at chunk 1000.
- **The first prefill's latency.** The scale pass for 400 linears runs once. Measured and recorded in Task 4, so a user's first request is not a surprise.

---

### Task 1: `PrefillBackend::L0Int8`

**Files:**
- Modify: `src/runtime/prefill_backend.h`, `src/runtime/prefill/step.cc`, `src/runtime/prefill/attn.h`, `src/runtime/prefill/attn.cc`, `src/runtime/prefill/engine_prefill.cc`, `src/cli/b70_serve.cc`, `src/cli/b70_decode.cc` (usage text only)
- Test: `tests/prefill/prefill_smoke_test.cc`

**Interfaces:**
- Produces: `PrefillBackend::L0Int8`, `inline bool is_l0(PrefillBackend b)` (true for L0 and L0Int8), `parse_prefill_backend("l0-int8")`, `prefill_backend_name(L0Int8) == "l0-int8"`, and `step_chunk_launches(L0Int8, C) == step_chunk_launches(L0, C) + 256`.

- [ ] **Step 1: Write the failing test**

In `prefill_smoke_test.cc`'s host-only section, beside the existing `step_chunk_launches(L0, ...)` checks:

```cpp
  runtime::PrefillBackend b{};
  CHECK(runtime::parse_prefill_backend("l0-int8", b) && b == runtime::PrefillBackend::L0Int8);
  CHECK(std::string(runtime::prefill_backend_name(runtime::PrefillBackend::L0Int8)) == "l0-int8");
  CHECK(runtime::is_l0(runtime::PrefillBackend::L0Int8) && runtime::is_l0(runtime::PrefillBackend::L0));
  CHECK(!runtime::is_l0(runtime::PrefillBackend::SyclTla));
  for (uint32_t C : {kShort, 1000u, 2048u})
    CHECK_EQ(runtime::prefill::step_chunk_launches(runtime::PrefillBackend::L0Int8, C),
             runtime::prefill::step_chunk_launches(runtime::PrefillBackend::L0, C) + 256);
```

- [ ] **Step 2: Run it to see it fail**

Run: `tools/box.sh test prefill_smoke_test`
Expected: build error, `L0Int8` is not a member of `PrefillBackend`.

- [ ] **Step 3: Implement**

`prefill_backend.h`:

```cpp
enum class PrefillBackend {
  SyclTla,   // spec 2's path: dequant -> host wait -> sycl-tla GEMM -> host wait
  L0,        // spec 2.1: every GEMM on the Level Zero list, no SYCL, no host waits
  L0Int8,    // spec 5: L0, with every int4 linear on the h8 int8 path
};

// L0Int8 is L0 for everything but the int4 linears.
inline bool is_l0(PrefillBackend b) { return b == PrefillBackend::L0 || b == PrefillBackend::L0Int8; }

inline const char* prefill_backend_name(PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? "sycl-tla" : b == PrefillBackend::L0 ? "l0" : "l0-int8";
}

inline bool parse_prefill_backend(const std::string& s, PrefillBackend& out) {
  if (s == "sycl-tla") { out = PrefillBackend::SyclTla; return true; }
  if (s == "l0") { out = PrefillBackend::L0; return true; }
  if (s == "l0-int8") { out = PrefillBackend::L0Int8; return true; }
  return false;
}
```

Replace each `backend == PrefillBackend::L0` / `b == PrefillBackend::L0` in `step.cc:131`, `attn.h:59,68,112`, `attn.cc:59` and `engine_prefill.cc:89` (`backend != PrefillBackend::L0` becomes `!is_l0(backend)`) with `is_l0(...)`. In `step.cc`, `step_chunk_launches`:

```cpp
size_t step_chunk_launches(PrefillBackend b, uint32_t C) {
  if (b == PrefillBackend::SyclTla) return 1 + 48 * kGdnLayerLaunches + 16 * kFaLayerLaunches;
  const size_t base = 1 + 48 * l0_gdn_layer_launches() + 16 * l0_fa_layer_launches(C);
  if (b == PrefillBackend::L0) return base;
  // spec 5: +1 launch (the activation quantiser) per int4 linear, 4 per layer.
  // The int8 gate||up is ALWAYS the fused SiLU form, while `base` follows
  // silu_fused(): an unfused session's base counts one pf_silu_mul per layer
  // the int8 walk never issues.
  return base + 64 * 4 - (silu_fused() ? 0 : 64);
}
```

Usage strings in both CLIs: `[--pp-backend sycl-tla|l0|l0-int8]`.

Then pin the Review Focus line:

Run: `grep -rn "== PrefillBackend::L0\b\|!= PrefillBackend::L0\b" src | grep -v prefill_backend.h`
Expected: no output.

- [ ] **Step 4: Run the tests to see them pass**

Run: `tools/box.sh test prefill_smoke_test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/prefill_backend.h src/runtime/prefill/step.cc src/runtime/prefill/attn.h \
        src/runtime/prefill/attn.cc src/runtime/prefill/engine_prefill.cc src/cli/b70_serve.cc \
        src/cli/b70_decode.cc tests/prefill/prefill_smoke_test.cc
git commit -m "runtime: PrefillBackend::L0Int8, L0 everywhere but the int4 linears (spec 5 T3)"
```

---

### Task 2: route the linears, own the state, build the scales up front

**Files:**
- Modify: `src/runtime/prefill/step.h`, `src/runtime/prefill/step.cc`, `src/runtime/prefill/engine_prefill.cc`
- Test: `tests/prefill/prefill_int8_test.cc`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: plan 5a's `Int8State`, `linear_i8`, `linear_i8_silu`; Task 1's `L0Int8`.
- Produces: `step_chunk(..., PrefillBackend backend, Int8State* q)`, where `q` must be non-null iff `backend == L0Int8`.

- [ ] **Step 1: Write the failing test**

`tests/prefill/prefill_int8_test.cc` (label `checkpoint;prefill`, SKIP 77 without the snapshot, the harness of `prefill_replay_test.cc`). Load the checkpoint once. For the `prose` prompt and for 2048 ids of `long`:

1. prefill on `l0`: record the first generated id and the last-row logits;
2. `reset()`, prefill on `l0-int8`: record the same; require finite logits, the same first id on `prose`, and a last-row logits cosine ≥ 0.99 against `l0` (a smoke bar; the real bars are Task 3's);
3. `reset()`, prefill on `l0-int8` with `set_prefill_replay(true)` twice: require the second run's logits **bitwise equal** to the non-replay int8 run's (Review Focus: no one-time work in a recording);
4. require `eng.prefill_launches()` delta for one 2048-id int8 chunk `== step_chunk_launches(L0Int8, 2048) + 5`.

Register it like `prefill_replay_test` in `tests/CMakeLists.txt`, with dependencies on every `kernel_pf_quant_had_*`, `kernel_pf_requant_rot_L*` and `kernel_pf_gemm_i8*`.

- [ ] **Step 2: Run it to see it fail**

Run: `tools/box.sh test prefill_int8_test`
Expected: FAIL. `l0-int8` still runs the bf16 linears, so the launch-count requirement fails (the int8 count is 256 higher than what runs).

- [ ] **Step 3: Implement**

`engine_prefill.cc`, `struct PrefillEngine`: add `std::unique_ptr<prefill::Int8State> int8;`, declared before `chunks` so recordings die first. In `Engine::prefill`, after `backend` is known and before the chunk loop:

```cpp
  prefill::Int8State* q = nullptr;
  if (backend == PrefillBackend::L0Int8) {
    if (!pfx_->int8) pfx_->int8 = std::make_unique<prefill::Int8State>(ctx_);
    q = pfx_->int8.get();
    // Every int4 linear's rotated column scales, now, outside any chunk and any
    // recording: the first int8 prefill pays this once, and no recorded list
    // ever holds the host finish of pf_colmax_rot.
    for (const auto& [key, w] : model_.linears)
      if (key.second != model::LinearId::LmHead && w.kind == model::WeightKind::Int4)
        q->scales(pfx_->cx, pfx_->kc, w);
  }
```

Pass `q` to `step_chunk` (and through `encode`). In `step.cc`, `pf_linear` gains `Int8State* q`:

```cpp
void pf_linear(Context& cx, KernelCache& kc, PrefillScratch& s, const DeviceWeight& w,
               const uint16_t* x, uint32_t M, PrefillBackend backend, Int8State* q) {
  if (backend == PrefillBackend::SyclTla) { linear_sycl(cx, kc, s, w, x, M); return; }
  if (backend == PrefillBackend::L0Int8) { linear_i8(cx, kc, s, *q, w, x, M); return; }
  linear_l0(cx, kc, s, w, x, M);
}
```

The gate‖up block becomes `fuse = l0 && (backend == PrefillBackend::L0Int8 || silu_fused())`, and inside `if (fuse)` calls `linear_i8_silu(...)` when `backend == L0Int8`, else `linear_l0_silu(...)` as now. `step_chunk` requires `(q != nullptr) == (backend == L0Int8)`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `tools/box.sh test "prefill_int8_test|prefill_smoke_test|prefill_replay_test|prefill_gate_l0_test"`
Expected: all PASS. The int8 test prints its cosine and first ids, and the L0 tests are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/prefill/step.h src/runtime/prefill/step.cc src/runtime/prefill/engine_prefill.cc \
        tests/prefill/prefill_int8_test.cc tests/CMakeLists.txt
git commit -m "runtime: l0-int8 routes every int4 prefill linear to the h8 path (spec 5 T3)"
```

---

### Task 3: gates A2 and A3 on the engine

**Files:**
- Modify: `tests/CMakeLists.txt` (registrations)
- Create: `tools/rotate/gate_compare.py`

**Interfaces:**
- Consumes: `prefill_gate_test` (argv[4] prompts, [5] chunk, [6] backend), which prints per greedy step a row under the header containing `logit-cos` and a final verdict.

- [ ] **Step 1: The registrations**

Beside `prefill_gate_l0_test` (tests/CMakeLists.txt:562), and inside the same `if(EXISTS ...)` guards as the `long` and RTN registrations:

```cmake
add_test(NAME prefill_gate_int8_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0-int8)
set_tests_properties(prefill_gate_int8_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
add_test(NAME prefill_gate_int8_c1000_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 1000 l0-int8)
set_tests_properties(prefill_gate_int8_c1000_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
```

The `long` prompt, with the same arguments as `prefill_gate_long_test` plus `l0-int8`: `prefill_gate_long_int8_test`.

- [ ] **Step 2: `gate_compare.py`**

It reads two `prefill_gate_test` logs (the L0 run and the int8 run). Per prompt, it collects every per-step row's `logit-cos` value (the column named in the header line `greedy N: step engine golden det? logit-cos engine-argmax`) and the verdict. It prints, per prompt, count / mean / min of logit-cos for both runs, and applies:

- **A3:** the int8 verdict is PASS whenever the L0 verdict is PASS, with the same determined-row count.
- **A2:** per prompt, `mean_cos(int8) >= mean_cos(l0) - 0.002` and `min_cos(int8) >= min_cos(l0) - 0.01`.

It exits non-zero if either fails, and prints which prompt failed it. It is written against the real log format: run `prefill_gate_l0_test` once, read its output, and make the parser match what is printed, failing loudly on a header it cannot find rather than guessing.

(A2 as written in the spec, prompt-logit relative L2 against the oracle, needs an all-positions logits dump the engine does not have. The gate's per-step logits after the int8 prefill carry the same information, because every decode step reads the KV and GDN state the prefill wrote. The spec is amended in Task 4.)

- [ ] **Step 3: Run the gates**

Run: `tools/box.sh test "prefill_gate_l0_test|prefill_gate_int8_test|prefill_gate_int8_c1000_test|prefill_gate_long_test|prefill_gate_long_int8_test"`, keeping each log (`ctest --output-on-failure -V` into a file per test). Then `python3 tools/rotate/gate_compare.py <l0 log> <int8 log>` for the default, c1000 and long pairs.
Expected: all five ctest entries PASS, and `gate_compare.py` prints A2 PASS and A3 PASS for all three pairs.

If A2 or A3 fails while plan 5a's kernel tests pass, **stop** (the spec's stopping rule). Record the numbers in the spec's amendment section and do not tune thresholds.

- [ ] **Step 4: Commit**

```bash
git add tests/CMakeLists.txt tools/rotate/gate_compare.py
git commit -m "tests: the l0-int8 golden gates and the A2/A3 comparison (spec 5 T4)"
```

---

### Task 4: speed B1 and B2, first-prefill cost, A4, then the default

**Files:**
- Modify: `docs/BENCHMARKS.md`, `docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md` (an amendment section), and only if every gate passes, `src/runtime/prefill/backend_sycl.cc` and `backend_sycl_absent.cc` (`default_prefill_backend`)

- [ ] **Step 1: Measure**

On an idle box, device 0:

```bash
ZE_AFFINITY_MASK=0 tools/bench_decode.sh --pp 4096 --pp-backend l0 --runs 3
ZE_AFFINITY_MASK=0 tools/bench_decode.sh --pp 4096 --pp-backend l0-int8 --runs 3
ZE_AFFINITY_MASK=0 tools/bench_decode.sh --runs 3
```

Alternate the first two twice (l0, int8, l0, int8), so the paired rows see the same clock state. Record: pp t/s for each, their ratio (**B1: int8 / l0 ≥ 1.20**), the decode tg t/s (**B2: within the noise of the 29.45 record**). Also record the **first** int8 prefill's extra wall time, the one-time scale pass: time `Engine::prefill` on a fresh engine against the second call.

- [ ] **Step 2: A4**

Run plan 5c's `tools/toolcall/score.py` on the `l0` and `l0-int8` greedy outputs. **A4:** int8 matches bf16 on at least as many prompts as `l0` does.

- [ ] **Step 3: Record**

`docs/BENCHMARKS.md`: a prefill series row `l0-int8, spec 5 h8 path` with pp t/s, spread, grade RECORD if idle, and the ratio to the paired `l0` row. The spec gets "## 7. Amendment - <date>, results": the `--pp-backend l0-int8` spelling, the A2 redefinition (from Task 3), and the A2/A3/A4/B1/B2 numbers and verdicts.

- [ ] **Step 4: The default, only if all five gates passed**

`default_prefill_backend()` returns `PrefillBackend::L0Int8` in both `backend_sycl.cc` and `backend_sycl_absent.cc`. Run the full suite (`tools/box.sh test`). Registrations that pin L0 still pass, since they name `l0` explicitly.

- [ ] **Step 5: Commit**

```bash
git add docs/BENCHMARKS.md docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md \
        src/runtime/prefill/backend_sycl.cc src/runtime/prefill/backend_sycl_absent.cc
git commit -m "prefill: l0-int8 results (spec 5 T4)"
```

**Done when:** the gates have a recorded verdict. Either h8 is the default with a BENCHMARKS row, or the spec's amendment records which bar failed and h8 stays opt-in.

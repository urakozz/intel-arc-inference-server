# Spec 6b - flash attention in production, and 128k context, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prefill attention runs plan 6a's winning fused kernel by default, with no max_len-sized score scratch, and the engine runs, and is gated, at 131072 tokens of context.

**Architecture:** `pfa_KT64_R16_H6_Q0` is promoted unchanged to `src/kernels/prefill/pf_flash_attn.cl`. `attn_chunk` launches it once per FA layer on the L0 backends, writing `pf_o` exactly where the composed path did, so `pf_attn_gate` and everything after is untouched. The composed path stays behind `B70_PREFILL_ATTN=composed` as the reference, and its `pf_s` / `pf_p` become lazy. Decode gets L131072 attention variants. The long-context gates are a 32k flash-against-composed end-to-end test, 128k determinism and replay, and a passkey-retrieval script.

**Tech Stack:** OpenCL C (ocloc AOT, 256 GRF), Level Zero, C++20, CMake/ctest, Python 3 in the oracle container for tokenisation; the box.

**Spec:** `docs/superpowers/specs/2026-09-25-spec6-flash-attention-128k-design.md`. The measured basis is `docs/probe-flash-attn-2026-09-25.md`.

## Global Constraints

- **Operator ruling 2026-09-25 ("A"):** integrate this kernel now and optimise later. Spec 6's speed bars F1 (attention <= 80 ms per 4096 ids) and F2 (>= 60 % of bf16 peak at depth) are **re-baselined, not gated**: the kernel measured 1.004x the composed path at depth 2k, 0.899x at 16k and 1.219x at 32k, 28.0 TFLOP/s. This plan records F1 and F2 as measurements and gates on **no regression**: pp4096 within 1 % of the paired composed run. A split-d / SLM-staged kernel is the follow-up.
- The kernel is the probe's, **text unchanged**, built with `KT=64 RPW=16 HPW=6 QREG=0` and `-cl-intel-256-GRF-per-thread`. Its correctness (cosine >= 0.99999 against fp64, finite pad rows) was measured in 6a and is re-pinned by Task 1's test.
- Output contract: `pf_o` fp32 `[24][pad256(C)][256]`, per-head stride `rows * 256`, `rows = attn_rows(C, backend)`. **This corrects spec 6 §3.1**, which named `pf_attn`; Task 5 amends the spec.
- `B70_PREFILL_ATTN`: unset or `flash` means flash, `composed` means the composed path. It applies on `l0` and `l0-int8`. `sycl-tla` always runs composed.
- The composed path must stay bitwise what it is today when selected: its golden and consistency registrations are re-run with `B70_PREFILL_ATTN=composed`.
- 128k is max_len **131072**, a multiple of 256 and of 64. The KV cache is bf16.
- Timing protocol: idle box (no render-node holder, no container), device 0, interleaved control/candidate, median of 3 or median paired ratio.

## Review Focus

- **A composed-selected session after a flash session.** The lazy `pf_s` / `pf_p` must be allocated on first composed use, not assumed. Pinned in Task 2: one engine prefills with flash, then with `composed`, and both outputs match their single-mode runs bitwise.
- **Chunks not a multiple of 16 rows.** The kernel's grid is `ceil(C / 16)` row blocks, and a partial block writes pad rows. Pinned by the existing chunk-1000 gate and Task 2's C = 300 case.
- **A max_len with no compiled decode variant.** `--max-len 65536` has no binary and must fail at capture with the variant's name, not segfault. Pinned in Task 3.
- **Memory at 128k.** Loaded model 18.1 GB + KV 8.4 GB + `attn_part` ~405 MB (`[24][2048][8][258]` fp32) + prefill scratch + int8 state must fit in 32 GB, and the engine must print the sum before the first prefill. Pinned in Task 3 by the load-report line and a 128k load on the box.
- **Passkey at the extreme.** The needle at 95 % depth is the one closest to the chunk boundary math and the deepest KV reads. It is one of Task 5's three placements.

---

### Task 1: `pf_flash_attn`, the production kernel and its test (P2)

**Files:**
- Create: `src/kernels/prefill/pf_flash_attn.cl` (copied from `tools/probe/probe_flash_attn.cl`, entry `pfa` renamed `pf_flash_attn`, header comment rewritten to point at spec 6, plan 6b and the probe record; kernel text otherwise unchanged)
- Modify: `src/kernels/prefill/CMakeLists.txt`, `src/kernels/prefill/pf_kernels.h`
- Test: `tests/prefill/pf_flash_attn_test.cc`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: kernel `pf_flash_attn(const ushort* Q, const ushort* Kc, const ushort* Vc, float* O, uint pos, uint C, uint rows)`, WG 192 (12 sub-groups), grid `(ceil(C / 16), 4, 1)`; `kernels::pf_flash_attn_variant()` returning `"pf_flash_attn"`.

- [x] **Step 1: The failing test**

`tests/prefill/pf_flash_attn_test.cc`, ported from `tools/probe/probe_flash_attn.cc`'s fp64 reference and sampled-row check (same sampling, same cosine and max-abs report), for the cases `(16384, 2048)`, `(777, 300)`, `(0, 2048, qscale 30)` and `(0, 64)`. Bar: worst cosine >= 0.99999 against fp64; rows [C, pad256(C)) finite. Registered under `B70_PREFILL_ENABLED` with `add_dependencies(... kernel_pf_flash_attn)`, like `pf_int8_test`.

Run: `tools/box.sh test pf_flash_attn_test`
Expected: FAIL at build (`kernel_pf_flash_attn` missing).

- [x] **Step 2: The kernel and its build row**

Copy the file and rename the entry. In `CMakeLists.txt`:

```cmake
# Spec 6 (plan 6b): the fused flash attention, plan 6a's winner pfa_KT64_R16_H6_Q0.
add_ocloc_kernel(pf_flash_attn SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                 DEFINES KT=64 RPW=16 HPW=6 QREG=0 OPTIONS -cl-intel-256-GRF-per-thread)
```

`pf_kernels.h`: `inline std::string pf_flash_attn_variant() { return "pf_flash_attn"; }`.

- [x] **Step 3: Run it to pass**

Run: `tools/box.sh test pf_flash_attn_test`
Expected: four cases printed, all worst cos >= 0.99999, pad rows finite, PASS.

- [x] **Step 4: Commit**

```bash
git add src/kernels/prefill/pf_flash_attn.cl src/kernels/prefill/CMakeLists.txt src/kernels/prefill/pf_kernels.h \
        tests/prefill/pf_flash_attn_test.cc tests/CMakeLists.txt
git commit -m "kernels: pf_flash_attn, plan 6a's winning fused attention, in production (spec 6 P2)"
```

---

### Task 2: `attn_chunk` on flash, the composed path lazy and selectable (P3)

**Files:**
- Modify: `src/runtime/prefill/attn.h`, `src/runtime/prefill/attn.cc`, `src/runtime/buffers.h`, `src/runtime/buffers.cc`, `src/runtime/prefill/step.cc` (launch arithmetic comment), `src/runtime/prefill/profile.h` / `profile.cc` (a `kAttnFlash` phase)
- Test: `tests/prefill/attn_chunk_test.cc`, `tests/runtime/buffers_test.cc`, `tests/prefill/prefill_smoke_test.cc`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `enum class AttnMode { Flash, Composed }` and `AttnMode attn_mode()` in `attn.h` (reads `B70_PREFILL_ATTN` once; anything but `composed` is Flash); `attn_chunk_launches(C, b)` returns **1** on L0 backends in Flash mode, and the current formula otherwise; `PrefillScratch::pf_s_buffer()` / `pf_p_buffer()` (lazy, like `slab_buffer()`), with `pf_s` / `pf_p` removed as members and `lazy_bytes()` counting them.

- [x] **Step 1: The failing tests**

- `attn_chunk_test.cc`: a new case runs `attn_chunk` at `(pos 16384, C 2048)` and `(777, 300)` in both modes on the same inputs (set `B70_PREFILL_ATTN` with `setenv` before the first call, or add a test-only override `set_attn_mode_for_test(AttnMode)`), and requires the flash `pf_o` to have per-(row, head) cosine >= 0.99999 against the composed `pf_o` on rows [0, C).
- `buffers_test.cc`: `PrefillScratch` at max_len 131072 allocates **no** `pf_s` / `pf_p` (its `bytes()` excludes them and `lazy_bytes()` is 0 until `pf_s_buffer()` is called).
- `prefill_smoke_test.cc`: on L0 and L0Int8, `step_chunk_launches(b, C)` drops by `16 * (attn_chunk_launches_composed(C) - 1)` in Flash mode. Pin the exact numbers for C = 64, 1000 and 2048.
- The Review Focus line: one engine, `prose`, prefill in flash mode, then `reset()` and prefill in composed mode. Each output must be bitwise equal to a fresh engine's run in that mode.

Run: `tools/box.sh test "attn_chunk_test|buffers_test|prefill_smoke_test"`
Expected: FAIL (no `AttnMode`, `pf_s` still eager).

- [x] **Step 2: Implement**

In `attn.cc`, at the top of `attn_chunk` after the argument checks:

```cpp
  if (l0 && attn_mode() == AttnMode::Flash) {
    // Spec 6: one launch for all four kv groups. pf_o's layout is the composed
    // path's ([24][rows][256], stride rows * 256), so pf_attn_gate is unchanged.
    require(s.pf_o.size() >= size_t(kQHeads) * rows * kHeadDim * sizeof(float), "pf_o is undersized");
    cx.launch(kc(kernels::pf_flash_attn_variant(), "pf_flash_attn"), (C + 15u) / 16u, kKvHeads, 1,
              {PtrArg(q), PtrArg(kv_k), PtrArg(kv_v), PtrArg(s.pf_o.ptr()), arg_val(pos), arg_val(C),
               arg_val(rows)});
    profile_wait(cx, Phase::kAttnFlash);
    return;
  }
```

The composed body that follows reads `s.pf_s_buffer()` / `s.pf_p_buffer()` instead of the members. The WG size is fixed by the kernel's `reqd_work_group_size`, so check that `cx.launch` sets it from the kernel (as it does for `pf_gemm`); if not, set it explicitly. In `buffers.cc`, move `pf_s` / `pf_p` to `std::unique_ptr<l0::Mem>` built on first access, with the sizes they have today; `pf_s_bytes()` returns 0 until then. Update `attn_chunk_launches` and the `step.cc` launch-arithmetic comment.

- [x] **Step 3: Run the tests and the gates**

Run: `tools/box.sh test "pf_flash_attn_test|attn_chunk_test|buffers_test|prefill_smoke_test|prefill_gate|prefill_consistency|prefill_determinism|prefill_replay|prefill_int8_test"`
Expected: all PASS in the default (flash) mode.

Then the composed reference, by hand:

```bash
tools/box.sh run 'cd build && B70_PREFILL_ATTN=composed ctest -R "prefill_gate_l0_test|prefill_gate_int8_test|prefill_consistency_l0_test"'
```

Expected: PASS, the composed path unchanged.

- [x] **Step 4: pp4096, no regression (the re-baselined F1)**

Interleave `B70_PREFILL_ATTN=composed` and flash on `tools/bench_decode.sh --pp 4096 --runs 3` (composed, flash, composed, flash; `bench_decode.sh` must forward `B70_PREFILL_ATTN` to the box, so add it to its env pass-through as `ZE_AFFINITY_MASK` is). Bar: flash median >= 0.99 x the paired composed median. Also run one profiled flash pp4096 and record the `attn_flash` phase ms (F1 as a measurement).

- [x] **Step 5: Commit**

```bash
git add src/runtime/prefill/attn.h src/runtime/prefill/attn.cc src/runtime/buffers.h src/runtime/buffers.cc \
        src/runtime/prefill/step.cc src/runtime/prefill/profile.h src/runtime/prefill/profile.cc \
        tests/prefill/attn_chunk_test.cc tests/runtime/buffers_test.cc tests/prefill/prefill_smoke_test.cc \
        tests/CMakeLists.txt tools/bench_decode.sh
git commit -m "prefill: attention runs pf_flash_attn; the composed path is lazy and selectable (spec 6 P3)"
```

---

### Task 3: 128k (P4)

**Files:**
- Modify: `src/kernels/CMakeLists.txt` (`add_attn_decode(1 131072)`, `add_attn_reduce(1 131072)`), `tools/bench_decode.sh` (forward `--max-len`), `src/cli/b70_decode.cc` and `src/cli/b70_serve.cc` (usage text; the memory line), `src/runtime/engine.h` / `engine.cc` or the CLIs (the memory report)
- Test: `tests/CMakeLists.txt` (a CLI rejection registration), `tests/runtime/buffers_test.cc`

**Interfaces:**
- Produces: `--max-len 131072` works in both CLIs. The engine prints one line after load: `memory: model <GB>, kv <GB>, decode state <GB>, prefill scratch <GB>, int8 <GB>, total <GB> of <device GB>`, the sum of the loader report, `PersistentBuffers`, `DecodeScratch`, `PrefillScratch::bytes() + lazy_bytes()` and `Int8State::bytes()`.

- [x] **Step 1: The failing checks**

- A CLI registration: `b70-decode <snap> --bench --depth 64 --max-len 65536` must exit non-zero with stderr containing `attn_decode_M1_L65536` (the missing variant named), not a crash.
- `buffers_test.cc`: `PersistentBuffers` and `DecodeBuffers` at max_len 131072 report KV bytes = 16 x 131072 x 4 x 256 x 2 x 2 and `attn_part` = 24 x 2048 x 8 x 258 x 4.

Expected: the registration FAILS today by crashing or with a different message (check which, and pin the message the fix produces).

- [x] **Step 2: Implement**

Add the two variant rows. Make capture throw with the variant name when the module is missing. The `KernelCache` pattern already does this for prefill; do the same where `capture.cc` loads `attn_decode_variant`. Add the memory line. Forward `--max-len` in `bench_decode.sh` (`--max-len N` passes through to `b70-decode`).

- [x] **Step 3: 128k on the box**

Run: `tools/box.sh run 'ZE_AFFINITY_MASK=0 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --max-len 131072'`
Expected: loads, prints the memory line with total < 32 GB, and prints a decode row.

- [x] **Step 4: F3 and F4, measured**

On an idle box:
- **F4:** `tools/bench_decode.sh --depth 4096 --runs 3` at max_len 16384, then at `--max-len 131072`, interleaved twice. Bar: the 131072 median within 2 % of the 16384 median.
- **F3:** `--depth` 32768, 65536 and 131000 at `--max-len 131072`, median of 3 each. Record t/s beside the bandwidth-derived rate: the decode bandwidth measured at depth 4096 (bytes per token over time per token), divided by the bytes one token reads at that depth (the weights, plus 64 KiB per cached position). Bar: >= 90 %.

- [x] (not needed: F4 measured 1.0003x) **Step 5: If F4 fails, the indirect grid**

Only if F4 missed its bar. `attn_prep` writes `ze_group_count_t {ceil((pos + 1) / 64), 24, 1}` for `attn_decode`, and the reduce count likewise, into a device buffer each step. `capture.cc` records `zeCommandListAppendLaunchKernelIndirect` for both kernels against that buffer. `attn_reduce` must then walk only the blocks `attn_decode` wrote. Re-run F4, and the decode golden gates (`golden_gate_test` and its registrations) must stay green. If F4 passed, skip this step and record that it was not needed.

- [x] **Step 6: Commit**

```bash
git add src/kernels/CMakeLists.txt tools/bench_decode.sh src/cli/b70_decode.cc src/cli/b70_serve.cc \
        src/runtime tests/CMakeLists.txt tests/runtime/buffers_test.cc
git commit -m "runtime: 131072-token context - decode variants, the memory report, F3/F4 (spec 6 P4)"
```

---

### Task 4: flash against composed at 32k, end to end (K3a)

**Files:**
- Create: `tests/prefill/flash_long_test.cc`, `tools/probe/mk_long_ids.py`
- Modify: `tests/CMakeLists.txt`

- [x] **Step 1: The long prompt**

`tools/probe/mk_long_ids.py <snapshot> <out.ids> <n>`: the concatenated text of `docs/*.md` and `src/**/*.cc` (sorted, repeated until long enough), tokenised with the snapshot's `tokenizer.json`, cut to n ids. Run it in the oracle container for n = 32768, and write the result to `tests/golden/prompts/long32k.ids`. That is about 200 KB; commit it, so the test needs no container.

- [x] **Step 2: The test** -- the original bar failed (logits cos 0.999661844, split at step 10); K3a redefined by the operator 2026-09-26, see spec 6 §8

`flash_long_test` (label `checkpoint;prefill`, SKIP 77 without the snapshot), max_len 32768 + 256: prefill `long32k.ids` minus its last 64 ids, in flash mode, then generate 64 greedy tokens and record them and the last-row logits. Repeat with `B70_PREFILL_ATTN=composed` in a fresh engine. Bars: the same 64 greedy tokens, and prefill last-row logits cosine >= 0.9999.

Run: `tools/box.sh test flash_long_test`
Expected: PASS, printing the cosine and the first divergence (none).

- [x] **Step 3: Commit**

```bash
git add tests/prefill/flash_long_test.cc tools/probe/mk_long_ids.py tests/golden/prompts/long32k.ids tests/CMakeLists.txt
git commit -m "tests: flash against composed end to end at 32k context (spec 6 K3)"
```

---

### Task 5: 128k gates, depth rows, the record (K3b, F2, P5)

**Files:**
- Create: `tools/probe/passkey.py`, `tools/probe/passkey.sh`
- Modify: `tests/prefill/flash_long_test.cc` (a `--128k` mode), `docs/BENCHMARKS.md`, `docs/superpowers/specs/2026-09-25-spec6-flash-attention-128k-design.md`, `docs/17-int8-prefill.md` (a pointer), `docs/README.md` (the probe record's line)

- [x] **Step 1: 128k determinism and replay**

`flash_long_test --128k` (not registered by default; run by hand): a 131000-id prompt (`mk_long_ids.py` at n = 131000, written to the box's scratch, not committed), prefilled twice in fresh engines. Bar: last-row logits bitwise equal and finite. Then once with `set_prefill_replay(true)` twice. Bar: bitwise equal to the immediate run.

- [x] **Step 2: Passkey retrieval**

`passkey.py <snapshot> <placement 0..1> <out.ids>`: filler text ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." repeated), with `The pass key is 71432. Remember it. 71432 is the pass key.` inserted at the placement fraction. The prompt ends with `What is the pass key? The pass key is`, tokenised to about 120000 ids, with no chat template (base continuation). `passkey.sh` runs placements 0.05, 0.5 and 0.95 through `b70-decode --ids <f> --n 8 --prefill --max-len 131072`, decodes the 8 ids, and passes if `71432` appears in them. Also run it with `--pp-backend l0` as a control.

Run: `tools/box.sh run 'tools/probe/passkey.sh'`
Expected: 3 of 3 on `l0-int8` and on `l0`.

- [x] **Step 3: Depth rows (F2 as a measurement)**

`tools/bench_decode.sh --pp N --max-len 131072 --runs 3` for N = 32768, 65536 and 131000, on an idle box. Record t/s, and the attention share from one profiled run at N = 65536.

- [x] **Step 4: The record**

- `docs/BENCHMARKS.md`: a section "Flash attention and 128k (spec 6)" with the pp4096 pair, the F1 phase ms, the depth rows, and the decode rows F3/F4.
- The spec: "## 8. Amendment - <date>, results": the pf_o correction, the operator's re-baselining of F1/F2 (ruling A, 2026-09-25), and every gate's number (K1 to K4, F1 to F4, passkey).
- `docs/README.md`: a line for `probe-flash-attn-2026-09-25.md` under the measurement records.

- [x] **Step 5: Commit**

```bash
git add tools/probe/passkey.py tools/probe/passkey.sh tests/prefill/flash_long_test.cc docs/BENCHMARKS.md \
        docs/superpowers/specs/2026-09-25-spec6-flash-attention-128k-design.md docs/README.md
git commit -m "spec 6: 128k gates (determinism, replay, passkey), depth rows, results"
```

---

**Done when:** flash is the default prefill attention with every gate green, `--max-len 131072` runs with the memory line under 32 GB, passkey is 3/3, and the spec's amendment records F1 to F4 as measured.

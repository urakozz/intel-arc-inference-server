# Spec 12b - int8 KV cache in the engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** store K and V as int8 with the scheme plan 12a chose, behind `--kv-cache bf16|int8`; quantise in the writers, dequantise in the readers; spec 7 and spec 8 interplay; gates Q2-Q5 and the speed bars.

**Architecture:**
- **Writers:** `attn_prep` (decode, `src/kernels/attn.cl`) and `pf_attn_prep` (prefill, `src/kernels/prefill/pf_attn_prep.cl`) quantise their K/V rows (and apply the rotation if chosen) and write the scales.
- **Readers:** decode attention v2 (`src/kernels/attn_v2.cl`, spec 10) and `pf_flash_attn` dequantise while loading; v1 decode and the composed prefill path stay bf16-KV only (references).
- **Buffers:** `src/runtime/buffers.{h,cc}` allocate int8 KV plus scales when selected; the MTP head's KV (spec 8) follows.
- **Snapshots:** `Engine::save_kv`/`load_kv` and `kv_bytes()` (spec 7) copy the int8 rows and the scales.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero, ctest.

**Spec:** `docs/superpowers/specs/2026-09-28-spec12-int8-kv-cache-design.md` (§3, §4 Q2-Q5, §5). Needs plan 12a's verdict (`docs/probe-int8-kv-2026-09-28.md`).

## Global Constraints

- Branch `spec12b-int8-kv` from main; box tree `~/b70-inference-server-spec12b`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. Interleaved pairs, median of 3, `uptime` recorded.
- `--kv-cache bf16` (the default until the gates pass) is bitwise today's engine. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Every KV writer and reader** is covered: decode attn_prep, prefill pf_attn_prep, the MTP head's KV fill in prefill and in verify (spec 8), decode v2, pf_flash_attn, save_kv/load_kv. A grep for `kv_k`/`kv_v` bindings in `capture.cc` and `step.cc` lists them; each gets a test.
2. **Mixing forms is refused:** a snapshot saved at bf16 cannot be loaded into an int8 engine (spec 7's cache keys gain the KV form).
3. **Verify rows bitwise equal to M = 1 (spec 8 M2)** still hold with int8 KV (quantisation per row is row-local).
4. **Scales at the positions MTP rejects:** stale int8 rows past `pos` are rewritten before read, as with bf16.
5. **The composed prefill path** with int8 KV is rejected with a clear error, not silently wrong.

---

### Task 1: buffers, writers, flag

- [ ] Buffers and the memory report line; `attn_prep` / `pf_attn_prep` quantise (+ rotation if chosen); `--kv-cache` in both CLIs; unit tests of the writers against a host reference (`tests/kernels/*` style). **Commit** `runtime: int8 KV cache buffers and writers (spec 12)`.

### Task 2: readers

- [ ] decode v2 and `pf_flash_attn` read int8 + scales; `attn_v2_test` and `pf_flash_attn_test` gain int8-KV cases (cosine against their bf16-KV outputs >= the tolerance 12a proposed). **Commit** `kernels: attention reads the int8 KV cache (spec 12)`.

### Task 3: spec 7 and spec 8 interplay

- [ ] save_kv/load_kv/kv_bytes; the cache key's KV form; the MTP head's KV; `snapshot_test`, `prefix_gpu_*`, `mtp_verify_test` with int8 KV (Review Focus 2-4). **Commit** `runtime: snapshots and MTP with the int8 KV cache (spec 12)`.

### Task 4: gates, speed, record

- [ ] **Step 1:** Q2 golden gates (`l0`, `l0-int8`) with int8 KV; Q3 `flash_long_test` oracle-closeness at 32k with 12a's tolerance and passkey 3/3 at 120k; Q4 A4 >= 25/36 and 256 greedy ids diverging only at near-ties; Q5 determinism/replay; full suite.
- [ ] **Step 2:** interleaved pairs bf16-KV vs int8-KV: decode at 32k / 64k / 128k (bars >= 1.1x / 1.15x / 1.2x), prefill at depth (<= 3 % regression), the memory line at 131072.
- [ ] **Step 3:** make int8 the `b70-serve` default only if every gate passes; BENCHMARKS section "int8 KV cache (spec 12)"; spec 12 amendment §8; README depth line.
- [ ] **Step 4: Commit** `spec 12: int8 KV cache gates and speed (the record)`.

**Gate for the plan:** Q2-Q5 green, speed measured (misses recorded), full suite green.

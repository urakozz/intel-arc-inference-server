# Spec 10b - decode attention v2 in production

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** put plan 10a's winning decode-attention design into the captured decode list as v2, selectable against v1 (`B70_DECODE_ATTN=v1|v2`), pass A1-A3, meet A-F3/A-F4, and make v2 the default.

**Architecture:** new kernels in `src/kernels/` beside `attn.cl`'s `attn_decode`/`attn_reduce` (v1 unchanged), compiled for M = 1..4 and max_len 16384 / 32768 / 131072; the positions-per-work-group stride written by `attn_prep` into `Control` each step so the fixed captured list serves every depth; the capture (`src/runtime/capture.cc`) binds v1 or v2 at capture time; `attn_part` resized to v2's partial count.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero, CMake/ctest.

**Spec:** `docs/specs/2026-09-28-spec10-decode-attention-at-depth-design.md` (§3 constraints, §4 A1-A3, §5 A-F3, A-F4, §6 10b). Needs plan 10a's verdict (`docs/probe-decode-attn-2026-09-28.md`).

## Global Constraints

- Branch `spec10b-decode-attn-v2` from main; box tree `~/b70-inference-server-spec10b` automatically. Copy `tools/box.env` from the main checkout if missing; never commit it. Symlink the `oracle-out*` dirs into the box tree.
- Other agents share the GPU: every GPU command under `flock ~/b70-gpu.lock`, detached, background until-loop polling for `ALLDONE rc=`. Record `uptime` with every timing table.
- v1 stays selectable and bitwise unchanged. The KV layout does not change. Replay determinism holds with v2 (no order-dependent atomics).
- Interleaved pairs v1/v2 after a warm-up, median of 3. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Depth 0 to 1 and tiny `pos`:** the stride formula at `pos + 1 < 64` (one partial, reduce trivial) and exactly at block boundaries.
2. **`pos` reaching `max_len - M`** (spec 8's verify near the end of the KV): no read past the RoPE table or KV.
3. **spec 7 snapshots** restore `pos` below the current one: the next step's stride is recomputed from the restored `pos`, never cached.
4. **The partial buffer** sized for the worst case of v2's formula at every max_len variant; the load report line updated.
5. **MTP (spec 8) M = 2..4 lists**, if on main: they capture v2 too, and spec 8's M2 gate passes.

---

### Task 1: the kernels

**Files:** `src/kernels/attn_v2.cl` (the 10a winner), `src/kernels/CMakeLists.txt` (variants M = 1..4 x L16384 / L32768 / L131072), `src/kernels/kernels.h` (names), `src/kernels/attn.cl` (only `attn_prep`: write the stride word, behind a define so v1's binary is unchanged). Test: `tests/kernels/attn_v2_test.cc` - A1 against v1 on real captured q/KV (reuse 10a's capture) at depths 4k / 32k / 128k, M = 1..4: cosine >= 0.99999, bitwise repeatable; Review Focus 1, 2.

- [ ] Failing tests → implement → pass → **Commit** `kernels: decode attention v2 at depth (spec 10 §2)`.

### Task 2: the capture and the selector

**Files:** `src/runtime/capture.cc`, `src/runtime/buffers.{h,cc}` (`attn_part` size, the `Control` word), `src/runtime/control.h`. Tests: `tests/runtime/buffers_test.cc` sizes; replay determinism with v2; Review Focus 3 (spec 7's `snapshot_test` with v2), 4, 5.

- [ ] Failing tests → implement → pass → **Commit** `runtime: capture decode attention v1 or v2 (spec 10 §3)`.

### Task 3: gates, speed, default

- [ ] **Step 1: A2, A3** with v2: golden gates, determinism, replay, `flash_long_test`, passkey 3/3 (`tools/probe/k3b_passkey.sh`) at 120k, `snapshot_test`; 256 greedy tokens at 32k identical to v1 except at near-ties.
- [ ] **Step 2: A-F3, A-F4:** interleaved v1/v2 decode at 4k / 32k / 64k / 128k, max_len 131072 (and 4k at max_len 16384); bars >= 23.6 / 21.0 / 17.3 t/s at depth, 4k within 1 %; M = 2..4 verify-step times at depth if spec 8's lists exist.
- [ ] **Step 3:** make v2 the default only if every gate passes; full suite green.
- [ ] **Step 4: the record:** `docs/BENCHMARKS.md` section "Decode attention at depth (spec 10)"; spec 10 amendment §8; spec 6 §8's F3 row gets a pointer; `README.md` depth line updated.
- [ ] **Step 5: Commit** `spec 10: decode attention v2 - gates, depth rows, default (the record)`.

**Gate for the plan:** A1-A3 passing, A-F3/A-F4 measured (a miss is recorded and handed back, not tuned around), full suite green.

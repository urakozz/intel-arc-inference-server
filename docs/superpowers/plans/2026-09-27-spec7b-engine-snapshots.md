# Spec 7b - engine snapshots, restore and block write-through

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** give `runtime::Engine` the calls prefix caching needs: save and restore the GDN state at a position, copy a KV position range to and from host memory, and call back at every completed 2048-position block during prefill; prove restore is bitwise (C1) and write-through is cheap (S3).

**Architecture:** all copies are device↔host copies on the prefill immediate list (`pfx_->cx`), outside the captured decode list, which reads `pos` and the persistent buffers that restore writes in place. The block hook runs on the host between chunks, where `prefill()` already waits (`pfx_->cx.wait()` before each chunk). When a hook is set, the first chunk of a prefill starting at `base` ends at the next multiple of 2048.

**Tech Stack:** C++20, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-09-27-spec7-prefix-caching-design.md` (§3.1, §3.2, §3.4 first bullet, §4 C1, §5 S3). Plan 7a's P0 doc (`docs/probe-prefix-cache-2026-09-27.md`) has the measured copy rates.

## Global Constraints

- `tools/box.env` is untracked: never commit it or its contents. No `rm -rf`.
- Box: `tools/box.sh sync|build|test|run`, `-j44`, device 0; long runs detached (`tools/probe/detach.sh`).
- Timing: interleaved pairs after warm-up, median of 3.
- `b70_decode`, the benchmarks and every existing test are unchanged in behaviour; with no hook set, `prefill()` chunks exactly as today.
- Commit on `spec7-prefix-cache`; no merge, no push.

## Review Focus

1. Restore to a position **below** the current `pos` while stale KV sits above it: decode and prefill must never read KV at positions ≥ the restored `pos` before writing them. C1's "overwrite with a second prompt first" case covers it; keep it.
2. `conv_ring` is a ring indexed by position modulo `kConvRing` (16): the snapshot must be the whole ring, and restore at a position with a different `pos % 16` than the current one must still be exact. C1 restores to 4395 over a state at position 3000, a different position modulo 16.
3. The prefill replay cache (`pfx_->chunks`, keyed by pos and rows) with the aligned first chunk: new (pos, rows) pairs appear; the FIFO bound of 8 must still hold, and replay must stay bitwise (`prefill_replay_test` passes unchanged; add a replay case with a hook set).
4. A hook that throws: `prefill()` must leave `pos` consistent with the chunks actually written (the exception propagates after the chunk's state is complete).
5. `save_kv` / `load_kv` at range ends that are not multiples of 2048, and an empty range (`begin == end`): exact, no copy for empty.

---

### Task 1: state and KV copies

**Files:**
- Modify: `src/runtime/engine.h`, `src/runtime/engine.cc` (or `src/runtime/prefill/engine_prefill.cc` if the prefill immediate list is only reachable there; the copies may use a dedicated immediate `l0::CmdList` created once in the constructor instead, as `EngineAdapter` does)
- Test: `tests/runtime/snapshot_test.cc` (checkpoint label, registered like `replay_determinism_test`: `LABELS checkpoint`, `TIMEOUT 900`, depends on `${B70_DECODE_LIST_KERNELS}` and links the prefill host library the way the prefill tests do)

**Interfaces (produces):**

```cpp
// runtime::Engine
static constexpr uint32_t kBlock = 2048;
size_t state_bytes() const;                 // gdn_state + conv_ring bytes (166.72 MB)
size_t kv_bytes(uint32_t n_pos) const;      // n_pos * 16 * 4 * 256 * 2 B * 2 (K and V)
// Host pointers are l0::MemKind::Host allocations (device-visible); blocking.
void save_state(void* host) const;          // gdn_state then conv_ring
void load_state(const void* host, uint32_t pos);  // writes both, then control.pos = pos,
                                                  // control.n_active = 0
void save_kv(uint32_t begin, uint32_t end, void* host) const;  // layout: K [16][end-begin][4][256]
                                                               // then V, same
void load_kv(uint32_t begin, uint32_t end, const void* host);
```

- [ ] **Step 1: Failing test** `snapshot_test` case A (C1). `cur_token` is never restored (spec §3.3 step 4), so every run prefills at least one id after the restore point. Load the model at max_len 16384. Straight run: `reset()`; `prefill` 4395 ids of `long32k.ids`; `save_state` + `save_kv(0,4395)` into host `l0::Mem`s; `prefill` id 4395 alone; generate 64 greedy tokens → list X. Restore run: `reset()`; `prefill` 3000 different ids (`tests/golden/prompts/` code prompt, repeated) so state and KV hold other data at other positions; `load_state(snap, 4395)` + `load_kv(0,4395)`; `prefill` id 4395 alone; generate 64 → list Y. Check X == Y, and that the device KV `[0, 4395)` read back after the restore equals the saved copy byte for byte. Case B: the same with the restore point at 4096 (a block end). Case C: `save_kv(4096, 4395)` then `load_kv(4096, 4395)` round-trips exactly; an empty range copies nothing.
- [ ] **Step 2: Run on the box, expect FAIL** (does not compile).
- [ ] **Step 3: Implement.** One `appendMemoryCopy` per contiguous run: KV is `[16 layers][max_len][4][256]`, so a range is 16 contiguous copies each for K and V.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `runtime: Engine::save_state/load_state/save_kv/load_kv (spec 7 §3.4, C1)`.

### Task 2: the block hook and aligned chunks

**Files:**
- Modify: `src/runtime/engine.h`, `src/runtime/prefill/engine_prefill.cc`
- Test: `tests/runtime/snapshot_test.cc` (cases D-F), `tests/prefill/prefill_replay_test.cc` (one case with a hook)

**Interfaces (produces):**

```cpp
// Called on the host after every prefill chunk whose end position is a multiple of kBlock,
// with the device idle and the state exactly at `end_pos`; also after the LAST chunk of every
// prefill (the prompt end), whatever its end. `is_block_end` tells which.
using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
void set_block_hook(BlockHook hook);  // empty function = none (today's chunking)
```

With a hook set, chunk k of a prefill starting at `base` has rows `min(chunk, kBlock - (pos % kBlock), remaining)`, so every chunk ends at a block end or at the prompt end. Without a hook, chunking is unchanged.

- [ ] **Step 1: Failing tests:** D: hook set, prefill 5000 ids from 0 → hook calls at (2048,true), (4096,true), (5000,false), in order. E: hook set, `prefill` from pos 300 of 4000 ids → chunks 1748, 2048, 204 rows; calls at (2048,true), (4096,true), (4300,false). F: in the hook, `save_state` + `save_kv(prev_end, end)` each time; afterwards restore the 4096 snapshot plus KV `[0,4096)` into a fresh session and prefill ids `[4096, 5000)` → the last-row argmax and 64 greedy tokens equal the straight-through run (bitwise: the chunk boundaries match those of the hooked run, so the state is the same).
- [ ] **Step 2: Expect FAIL. Step 3: Implement. Step 4: PASS, plus `prefill_replay_test` and the golden gates unchanged.**
- [ ] **Step 5: Commit** `prefill: block hook and block-aligned chunks when set (spec 7 §3.2)`.

### Task 3: S3, the write-through cost

**Files:** Create `tools/probe/probe_writethrough.cc` (executable, not a test).

Interleaved pairs on the idle box, device 0: cold pp4096 with no hook against cold pp4096 with a hook that does what 7c will do at each call (`save_state` into a host buffer, `save_kv` of the new range). Median ratio of 3 pairs after a warm-up pair. Also the same at pp32768 (max_len 131072). Bar: ≤ 3 % at pp4096.

- [ ] **Step 1:** write, build, run detached.
- [ ] **Step 2:** append the result to `docs/probe-prefix-cache-2026-09-27.md` (section "Write-through cost, S3").
- [ ] **Step 3:** if S3 fails: record it, and measure the variant that copies only at block ends that are **multiples of 4096** (half the snapshots); report both and hand back; do not change the design alone.
- [ ] **Step 4: Commit** `probe: write-through cost of block snapshots (spec 7 S3)`.

**Gate for the plan:** full suite green on the box, C1 (cases A-F) passing, S3 measured. Hand back with the S3 numbers.

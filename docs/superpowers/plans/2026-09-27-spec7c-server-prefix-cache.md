# Spec 7c - the prefix cache in the server

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70_serve` reuses earlier requests' work: it continues the resident session, or restores the deepest matching host snapshot, and prefills only the tail; every prefill writes its blocks through to a pinned-host store with an LRU budget. Gates C2 to C5, speed bars S1 and S2, and the record.

**Architecture:** `src/server/prefix_cache.{h,cc}` is pure host code: the store (blocks and snapshots in a tree, exact-id keys, leaf-only LRU) over an allocator interface, and `plan()`, which picks the restart point. `server.cc` asks it for a plan, drives the engine through new `EngineIface` calls (plan 7b's `Engine` API behind `EngineAdapter`), and feeds the block hook's saves back into the store.

**Tech Stack:** C++20, nlohmann::json, Level Zero host allocations, Python 3 for the replay tool.

**Spec:** `docs/superpowers/specs/2026-09-27-spec7-prefix-caching-design.md` (§3, §4 C2-C5, §5 S1-S2). Uses plan 7a (tool-call output, `--log-requests`, the recorded session in `tests/golden/opencode/session1/`, `docs/probe-prefix-cache-2026-09-27.md`) and plan 7b (`Engine::save_state`, `load_state`, `save_kv`, `load_kv`, `set_block_hook`, `kBlock`).

## Global Constraints

- `tools/box.env` is untracked: never commit it or its contents. No `rm -rf`.
- Box: `tools/box.sh sync|build|test|run`, `-j44`, device 0; long runs detached (`tools/probe/detach.sh`).
- Timing: warm-up, median of 3; A/B as interleaved pairs.
- `--prefix-cache-gb 0` is exactly today's server (`reset()` + full prefill); the default is 32.
- Numbers are **measured** unless marked **derived** or **estimated**.
- Commit on `spec7-prefix-cache`; no merge, no push.

## Review Focus

1. A request **shorter** than the resident session whose ids are a prefix of it (a client retry or an edit of the last message): restart must be a snapshot ≤ `len - 1`, never the resident `pos` (which is past the prompt). Test in Task 1.
2. The budget smaller than one session's blocks: inserting must evict older leaves, never the entries the current request is using; a store that cannot fit the new block skips storing it and the request still succeeds. Test in Task 1.
3. A request that fails mid-prefill (bad id, max_len) after a restore: the resident ids must be marked unknown so the next request cannot "continue" from a half-written state. Test in Task 2 with the mock.
4. Sampling (non-greedy) requests: the stored state is the prefill's, taken before any sampled token, and the request-end snapshot covers the tokens actually generated (`out_ids`), not the argmax. Test in Task 2.
5. Two requests with the same prompt back to back: the second restores at `len - 1` from the prompt-end snapshot and prefills exactly one id; `cached_tokens == len - 1`. Test in Task 2.

---

### Task 1: the store and the plan (host only)

**Files:**
- Create: `src/server/prefix_cache.h`, `src/server/prefix_cache.cc`; add to `b70_server` in `src/server/CMakeLists.txt`
- Test: `tests/server/prefix_cache_test.cc` (host-only, registered like `openai_test`)

**Interfaces (produces):**

```cpp
namespace server {
struct HostAlloc {                       // pinned host memory in b70_serve, malloc in tests
  virtual ~HostAlloc() = default;
  virtual void* alloc(size_t bytes) = 0;  // nullptr when it cannot
  virtual void free(void* p, size_t bytes) = 0;
};
struct Snapshot;                         // opaque to callers except through the calls below
class PrefixCache {
 public:
  PrefixCache(size_t budget_bytes, size_t state_bytes, size_t kv_bytes_per_pos, uint32_t block,
              HostAlloc& alloc);
  // The block hook's store calls. `ids` are the session's ids up to `end` (prompt, and for
  // the request end the generated ids too). Returns host pointers to fill (state first, then
  // KV of positions [kv_begin, end)) or {nullptr, nullptr} when nothing is to be stored
  // (already present, or the budget cannot fit it).
  struct Slot { void* state; void* kv; uint32_t kv_begin; };
  Slot reserve(const std::vector<uint32_t>& ids, uint32_t end, bool is_block_end);
  void commit(const Slot& slot);         // after the engine has filled it
  struct Plan {
    enum Kind { Cold, Continue, Restore } kind;
    uint32_t restart = 0;                // positions [0, restart) are reused
    uint32_t kv_from = 0;                // Restore: KV [kv_from, restart) comes from the host
    const void* state = nullptr;         // Restore
    std::vector<std::pair<uint32_t, const void*>> kv;  // Restore: (first pos, host ptr) runs
                                                        // covering [kv_from, restart)
  };
  // `resident` = ids the card's state and KV hold, `resident_valid` false after a failure.
  Plan plan(const std::vector<uint32_t>& prompt, const std::vector<uint32_t>& resident,
            bool resident_valid);        // pins the entries it returns until release()
  void release();
  size_t bytes_used() const;
};
}
```

Rules (spec §3.1, §3.3): blocks are keyed by (hash chain, exact ids of the block and of every ancestor via the parent link); a snapshot at `end` references its last block and holds the KV since that block's end. Eviction: LRU among leaves not pinned. `plan()`: `Continue` if `resident_valid`, `resident` (the ids whose state and KV the card holds, `resident.size() == pos`) is a prefix of `prompt`, and `resident.size() <= prompt.size() - 1`; else the deepest snapshot at `p <= prompt.size() - 1` whose ids equal `prompt[0:p]`, `kv_from = min(LCP(prompt, resident), p)` when `resident_valid` else 0; else `Cold`.

- [ ] **Step 1: Failing tests** with a malloc `HostAlloc` and small sizes (block = 4, state 64 B, kv 8 B/pos): continuation; restore at a prompt end; restore at a block end when the prompt diverges mid-block; `kv_from` equals the common prefix with the resident; Review Focus 1 and 2; a forced hash collision (inject a hash function through a test-only constructor argument) still never matches different ids; `release()` unpins; `bytes_used()` never exceeds the budget.
- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS.**
- [ ] **Step 5: Commit** `server: the prefix cache store and restart plan, host only (spec 7 §3.1, §3.3, C4)`.

### Task 2: the request path

**Files:**
- Modify: `src/server/deps.h` (`EngineIface` gains `state_bytes()`, `kv_bytes(n)`, `save_state`, `load_state`, `save_kv`, `load_kv`, `set_block_hook` with the same signatures as plan 7b's `Engine`), `src/cli/serve_adapters.h` (`EngineAdapter` forwards them; a `PinnedAlloc : HostAlloc` over `l0::Mem(ctx, MemKind::Host, n)`), `src/server/server.h`, `src/server/server.cc`, `src/server/openai.cc` (`usage.prompt_tokens_details.cached_tokens`), `src/cli/b70_serve.cc` (`--prefix-cache-gb N`, default 32, 0 = off)
- Modify: `tests/server/mock.h` (`MockEngine` keeps a fake state = hash of ids so far and a fake KV = the ids per position, so a wrong restore produces different generated ids)
- Test: `tests/server/prefix_server_test.cc` (host-only, mock)

Flow in `server.cc` (replacing lines 146-147):
1. `plan = cache.plan(prompt_ids, resident_ids, resident_valid)`.
2. Cold: `reset()`. Continue: nothing. Restore: `load_state(plan.state, plan.restart)` and `load_kv` for each run.
3. `resident_valid = false`; set the block hook to `reserve` → `save_state`/`save_kv` → `commit`; `prefill(prompt_ids[restart:])`; `resident_ids = prompt_ids`; `resident_valid = true`; `cache.release()`.
4. Generate as today; `resident_ids += out_ids` (with the pending-token protocol: the ids whose KV is written are the prompt plus all generated ids except the last one returned; check against `EngineIface::pos()` and use `pos()` as the truth).
5. After the last frame is sent: request-end snapshot via `reserve(resident_ids, pos(), false)`.
6. `usage.cached_tokens = plan.restart`. With `--prefix-cache-gb 0`, steps 1-2 and 5 are skipped and step 2 is always `reset()`.

- [ ] **Step 1: Failing tests** (mock): three-turn conversation with a side request in between; cached vs `--prefix-cache-gb 0` produce identical responses; `cached_tokens` values per request; Review Focus 3, 4, 5.
- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS; `golden_server_test` and all server tests unchanged.**
- [ ] **Step 5: Commit** `server: requests restart from the prefix cache (spec 7 §3.3, C4)`.

### Task 3: C2 and C5 on the card

**Files:** Create `tests/server/prefix_gpu_test.cc` (checkpoint label, `TIMEOUT 1800`, real `Engine` through `EngineAdapter`, no HTTP).

Sequences built from `long32k.ids` and the golden code prompt: (a) turn 2 extends turn 1's prompt + generated ids; (b) turn 2 drops turn 1's generated ids and appends 700 new ids (divergence at a prompt end); (c) divergence mid-block at 5000; (d) divergence at exactly 4096; (e) a side request of 300 unrelated ids between turns 1 and 2. For each: the cached run against a cold run (`reset()` + full prefill) of the same final prompt: last-row logits cosine and 32 greedy tokens under the tie-aware golden rule (reuse the golden gate's comparison helper from `tests/prefill/prefill_gate_test.cc`; the near-tie allowance is the same, 1 on `l0-int8`).

- [ ] **Step 1: write; Step 2: run on the box (both `l0` and `l0-int8`); Step 3: full suite `tools/probe/ctest_env.sh --` green (C5).**
- [ ] **Step 4: Commit** `tests: cached against cold on the card (spec 7 C2)`.

### Task 4: C3, S1, S2 - the recorded opencode session

**Files:** Create `tools/prefix/replay_log.py` (sends the logged requests of `tests/golden/opencode/session1/` in order to a running `b70_serve`, greedy: overrides `temperature` to 0 and sets `seed`; records per request the response, `cached_tokens`, time to first token (streamed) and total time; compares two runs: text equal or, where not, the first differing token and whether the golden tie rule accepts it; every `tool_calls` entry parses, same names and arguments).

- [ ] **Step 1:** replay with `--prefix-cache-gb 32` and with `0` (same server build, max_len 131072), detached on the box; C3 verdict.
- [ ] **Step 2: S1:** a synthetic continuation: a 60000-id history (`long32k.ids` extended) then a 1000-id tail, time to first token with the cache warm; bar ≤ 1.5 s. **S2:** the same after one 300-id side request; restore time (log it from the server: add a stderr line `prefix: kind restart kv_bytes restore_ms prefill_ms` per request) ≤ 1 s.
- [ ] **Step 3:** write the record: `docs/BENCHMARKS.md` section "Prefix caching (spec 7)" (P0 summary, S1, S2, S3, the C3 table of time to first token per request, cache on and off); spec 7 amendment §8 with every result; `docs/README.md` line for `probe-prefix-cache-2026-09-27.md`; `docs/04-architecture.md` "Not built: prefix caching" becomes "Prefix caching" and points to spec 7; `README.md` one paragraph under the 128k one.
- [ ] **Step 4: Commit** `spec 7: prefix caching results, C3 and S1/S2 (the record)`.

**Gate for the plan:** C2-C5 passing, full suite green, S1/S2 measured (a miss is recorded and handed back, not tuned around), the record committed. Hand back with the C3 table and the S1/S2/S3 numbers.

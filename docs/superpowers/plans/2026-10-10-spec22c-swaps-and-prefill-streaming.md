# Spec 22c - adaptive expert swaps between steps, and prefill streaming on the copy engine

**Status (2026-10-10): planned; begins after plan 22b, and only on the operator's go after plan 22a's record.** Tasks 1-2
(swaps) are built only if decision 3 admits swaps (22a's `swap` simulation shows a gain worth their cost); Task 3
(prefill streaming) is built in either case, in the form decision 7 picks. Box queue row 38 (renumbers at build time if
taken).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the static cache of 22b adapts while the model decodes and its prompts stream: (1) the route kernel counts
per-expert hits on the device; every S steps the host reads the counts and swaps up to B bytes of experts between the
pinned mirror and the VRAM slots, **between steps only**, by spec 22 §4's four-step protocol - the victim's entry
points at the host first, the incoming expert is copied on the copy engine only after the last step that used the slot
has retired, and it is admitted (entry -> slot) only once that copy's event has signalled; a step never waits for a swap,
a swap never races a reader (gate O2). (2) A prefill chunk's non-resident experts are copied layer by layer into a
dedicated staging ring on the copy engine while the previous layer computes, and the prefill's dequant reads them
through a static prefill table (decision 7); both cards stream at once under spec 16c's chunk pipeline.

**Architecture:** spec 22 §4 ("Adaptive swaps", "Prefill"), §5 O2, §7 decisions 3, 4, 7.
- **Counters:** `q4_route` gains a `_CNT` variant (`-DROUTE_COUNT=1`): work-group m (row m) adds 1 to
  `cnt[m][e]` for each of its 10 selected experts. Lane l already owns experts l and l + 256 (21c's `q4_route`), so each
  counter has one writer - no atomic, deterministic - and the route row itself is unchanged (the arithmetic is the
  same binary's but for the extra stores). Counters live per device in `u32 [4 rows][its layers + the head][512]`; the
  decode and verify lists bind `_CNT`, the prefill binds the plain route (prefill routing is not the decode working set:
  proposed, recorded).
- **The swapper** (`runtime::qwen4exp::ExpertSwapper`, host, one per device): every S steps (a step = any replay on
  that device: decode, verify, draft) it copies the counters out and resets them (both appended to the next list
  submission on the compute queue, in order), keeps a window of the last W steps' counts, and per layer picks up to
  `B / 2,611,200` pairs (incoming = the most-counted non-resident, victim = the least-counted resident, swapped only when
  `count(in) > count(victim) + margin`; S, W, B, margin from 22a's simulation). The protocol per pair, all on the
  device's queues, nothing synchronous:
  0. (`--expert-mirror misses` only) the victim has no host copy: the copy queue copies its slot into a spare host
     slot from the pool, event E0; the host checks E0 at a later step boundary and verifies the bytes' checksum;
  1. the victim's table entries := its host copy (an 8-byte copy appended before the next list on the compute queue);
  2. a barrier on the compute queue signals E1 after (1) - every step that could have read the slot has completed;
  3. the copy queue waits E1, copies the incoming expert's gate‖up and down blocks from the mirror into the slot,
     signals E2;
  4. at a later step boundary the host sees E2 (`SyncEvent::signalled()`, never a blocking wait) and appends the
     incoming entry := slot before the next list; under `misses` the incoming expert's host slot returns to the pool.
  The map (`ExpertMap`) changes with (1) and (4); 22b's checks (tags, checksums) cover every byte written.
- **Prefill streaming** (decision 7's proposal - a dedicated ring, chunk 2048): per device a staging ring of two halves,
  each `max over its layers of (512 - n_l)` expert slots; a **prefill table** (u64 `[layers][1024]`) whose resident
  entries equal the decode table's and whose non-resident entries of layer l point at ring half `l % 2`, slot j for the
  layer's j-th non-resident expert. Per chunk, a copy-queue list: for each of the device's layers in walk order, wait
  "layer l - 2's MoE done", copy layer l's non-resident experts into half `l % 2` (one copy per contiguous mirror run -
  with `misses` a layer's whole non-resident set is one run per gate‖up / down), signal "layer l staged"; the compute
  list waits "layer l staged" before layer l's dequant and signals "layer l's MoE done" after its combine. Chunks with
  fewer rows than `B70_Q4_STREAM_MIN` (from 22a's unique-experts-per-chunk table; default 512, PROPOSED) keep 22b's
  zero-copy path. Swaps pause during a prefill; a map change rebuilds the prefill table and the copy lists before the
  next prefill (an epoch counter).
- **Not built:** decision 7's other arm (borrowed cache slots refilled after the prompt) and chunk 4096 (21d's walk is
  bounded at 2048 - `kPfC`; a 4096 chunk is a change to 21d's walk and scratch, its own plan if decision 7 asks for it).

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero (`l0::SyncEvent`, `CmdList::barrier_signal` / `wait_event`, the
copy-only queue group 22a found), CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (§4, §5 O2, §7 decisions 3, 4, 7);
plan 22a's record (the swap simulation's S / B / W / margin, P0.2's copy rates per ordinal and both cards, P0.3's
replay visibility result, P0.6's prefill bytes); plan 22b (the table, the map, the mirror, `q4_page_check`). Spec 21 §14
(`q4_route`: two experts a lane), §15 (the chunk walk), spec 16c (the chunk pipeline's per-device events,
`SyncEvent::host_wait`'s bounded waits).

## Dependencies and branch points

- **Plan 22b merged** and its row 37 PASS on the box (O0, O1, the whole model on two cards).
- **Decision 3** (swaps or a static profile alone): Tasks 1-2 only if swaps are admitted; else they are recorded "not
  built: decision 3" and Task 3 starts. **Decision 4** decides step 0 of the protocol (`misses` needs the spare host pool,
  sized `--expert-swap-pool <experts>`, default `2 x B / 2,611,200`); **decision 7** the streaming form.
- **The copy-only queue group** (22a's `copy` mode found it or printed `none`): with none, the copy commands go on a second
  queue of the compute group - recorded with its measured cost.
- Row 22 / 23 (spec 16b / 16c) PASS for the two-card arms.

## Global Constraints

- Branch `spec22c-expert-swaps` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never
  printed; `oracle-out*` symlinked.
- **Commit after each step that changes a file** (signed `git commit -S`; never bypass). No `rm -rf`. No merge, no push.
- **F0:** existing binaries unchanged (`q4_route`'s define off preprocesses as main's); new binaries `_CNT` only.
- **Nothing waits on a swap.** The host never blocks on E0 / E1 / E2: it polls at step boundaries; a step's list is
  submitted whatever the swap state. Every wait that does exist (prefill staging, the pipeline) is bounded.
- **Residency never changes arithmetic** (22b's O1): every gate here is bitwise against the no-swap / zero-copy run.
- Box: `flock ~/b70-gpu.lock` (both cards), detached, polled; interleaved pairs after warm-up, median of 3, `uptime`,
  idle grade; `-j44`; the full model's RAM rule (no CPU oracle beside the pinned mirror and PLE).
- Mac checks `tools/mac_check.sh --base main --kernels`.

## Review Focus

1. **The protocol's order is the proof.** For every pair: (1) is in the compute queue before E1's barrier; the copy
   waits E1; (4) is appended only after the host saw E2; a slot is never the target of a copy while any entry points at
   it; under `misses`, (1) only after E0 and a matching checksum. A host-side model of the protocol (Task 1) checks these
   as invariants over randomised event timings.
2. **Withheld and poisoned copies.** `B70_Q4_SWAP_WITHHOLD=1` never signals E2: the incoming entry is never admitted
   and the run is bitwise; `B70_Q4_SWAP_POISON=1` fills the slot with NaN right after E1, before the real copy: any
   reader in the window would poison the logits, so a bitwise run proves there is none.
3. **Counters do not touch the route.** `_CNT`'s route rows, ids and weights are bitwise the plain binary's; its counts
   equal the host's count of the route rows over the same steps.
4. **Prefill staging reads only staged bytes.** The compute list's dequant for layer l waits "layer l staged"; the copy
   into half `l % 2` for layer l waits "layer l - 2's MoE done"; streaming prefill is bitwise the zero-copy prefill and
   the all-resident prefill (O1 for prefill), chunked and whole, one card and two.
5. **Epochs.** A swap between two prompts changes the non-resident sets: the next prefill's table and copy lists are
   the new map's (an epoch check throws if a stale list is executed).

---

### Task 1: the counters and the swap policy (host, then the kernel)

**Files:**
- Modify: `src/kernels/qwen4exp/q4_moe.cl` (`ROUTE_COUNT`), `src/kernels/qwen4exp_kernels.h` (`route_variant(M, bool
  count = false)`), `src/kernels/CMakeLists.txt` (a block `# ==== Spec 22c: expert swaps ==== (begin) / (end)`:
  `q4_route_M<1..4>_E512_T10_N528_L256_CNT`), `tests/kernels/qwen4exp_ref.h` (`q4ref::route_count`),
  `tests/kernels/qwen4exp_kernels_test.cc`, `tests/kernels/qwen4exp_variant_names_test.cc`, `tools/mac/clrun/qwen4exp_run.cc`,
  `tests/CMakeLists.txt`
- Create: `src/runtime/qwen4exp/qwen4exp_swap.{h,cc}` (`SwapPolicy`, `SwapProtocol` - the state machine, no device),
  `tests/runtime/qwen4exp_swap_test.cc` (host)
- Test: `qwen4exp_swap_test` (host), the names test, `qwen4exp_kernels_test` (card), `qwen4exp_run` (Mac GPU)

**Interfaces:**

```cpp
namespace runtime::qwen4exp {
struct SwapParams { uint32_t S = 16, W = 256, margin = 2; size_t B = 0; bool on = false; };   // 22a's values
struct SwapPair { uint32_t layer, in, victim, slot, host_slot; };
// Picks the round's pairs from a window of counts against the map (per layer, slots fixed per layer).
std::vector<SwapPair> pick(const ExpertMap& m, const std::vector<uint32_t>& window_counts, const SwapParams& p);
// The protocol without a device: events are callbacks; it emits the commands in order and refuses an out-of-order step.
class SwapProtocol {
 public:
  enum class Cmd { CopySlotToHost, PointAtHost, BarrierE1, CopyMirrorToSlot, Admit };
  struct Step { Cmd cmd; SwapPair pair; };
  SwapProtocol(MirrorPolicy mp);
  void begin(const std::vector<SwapPair>& pairs);
  std::vector<Step> at_step_boundary(const std::function<bool(uint32_t id, int e)>& signalled);  // e = 0, 1, 2
  bool idle() const;
};
}
```

- [ ] **Step 1: the failing host test** `qwen4exp_swap_test`: `pick` on a crafted window (ties to the lower id; the margin;
  never more than B; within a layer only); `SwapProtocol` under 10^4 randomised event orders and delays (both mirror
  policies) never emits `CopyMirrorToSlot` before `BarrierE1` of the same pair, never `Admit` before E2, never
  `PointAtHost` before E0 under `misses`, and each pair ends admitted or (withheld) pointing at the host; the
  `q4ref::route_count` twin. FAIL.
- [ ] **Step 2: implement; the kernel variant.** Names test PASS; preprocessed-source identity of every existing
  `q4_route` variant recorded; `qwen4exp_kernels_test`'s `count` case: route rows bitwise the plain binary, counts equal
  the twin at M = 1..4 (Review Focus 3); `qwen4exp_run` on the Mac. Mac gate. Commit `git commit -S -m "kernels: q4_route
  counts its experts' hits per row (_CNT, one writer per counter); the swap policy and its protocol as a host state
  machine (spec 22c)"`.

### Task 2: swaps on the card; O2

**Files:**
- Modify: `src/runtime/qwen4exp/qwen4exp_engine.{h,cc}` (an `ExpertSwapper` per device driven at every step boundary of
  `generate`, `verify`, `draft`; `read_expert_counts()`, `swap_stats()`), `src/runtime/qwen4exp/qwen4exp_capture.cc` (the
  `_CNT` binding and the counters' copy-out / reset when swaps are on), `src/loader/qwen4exp_experts.{h,cc}` and
  `src/loader/qwen4exp_mirror_usm.cc` (the spare host pool), `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (the counters
  and the pool in the plan), `src/cli/qwen4exp_decode.h` (`--expert-swap off|on`, `--expert-swap-every S`,
  `--expert-swap-bytes MB`, `--expert-swap-pool N`), `tests/CMakeLists.txt`
- Create: `src/runtime/qwen4exp/qwen4exp_swapper.cc` (the device half), `tests/runtime/qwen4exp_swap_gpu_test.cc`
- Test: `qwen4exp_swap_gpu_test` (card)

- [ ] **Step 1: the failing test** `qwen4exp_swap_gpu_test <ckpt> <ids> [intel ckpt]` (the synthetics at 50 % resident;
  Intel's `--layers 18` / `37`; the whole model when present): (a) **stress** - S = 1, B = 4 experts, pairs drawn at
  random (a seeded test policy), 256 decode steps after a 2200-id prefill, and verify at K = 2: every logit, id and state
  bitwise the no-swap run; (b) **withheld** and (c) **poisoned** (Review Focus 2); (d) the counters against the route rows
  read back; (e) `misses` with the pool: every evicted expert's host copy checksummed; (f) two cards: each device's
  swapper independent, bitwise `--pp 1`; (g) the table rows read back after every round equal the map. FAIL.
- [ ] **Step 2: implement**; PASS; 22b's O0 / O1 tests unchanged with `--expert-swap off`. Commit `git commit -S -m
  "runtime: expert swaps between steps - point at the host, copy after the slot's last reader, admit on the copy's
  event; O2 bitwise under stress, withheld and poisoned copies (spec 22c)"`.

### Task 3: prefill streaming

**Files:**
- Modify: `src/runtime/qwen4exp/qwen4exp_prefill.{h,cc}` and `qwen4exp_prefill_engine.cc` (the ring, the prefill table,
  the per-chunk copy list and its events, `B70_Q4_STREAM_MIN`, the epoch), `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}`
  (the ring in the prefill scratch: `2 x max(512 - n_l) x 2,611,200` B per device), `src/cli/qwen4exp_decode.h`
  (`--expert-stream on|off`, default on with the tier), `tests/CMakeLists.txt`
- Create: `tests/runtime/qwen4exp_stream_test.cc`
- Test: `qwen4exp_stream_test` (card), `qwen4exp_plan_test` (host: the ring's bytes)

- [ ] **Step 1: the failing tests.** `qwen4exp_plan_test`: the ring's bytes per device at 22b's fill, the plan with it.
  `qwen4exp_stream_test`: streaming prefill (chunks of 2048 and of 64 - the latter below the threshold, zero-copy) bitwise
  the zero-copy prefill and the all-resident prefill on the synthetics (forced 50 % / 0 %) and Intel's 18 / 37 layers,
  one card and two (Review Focus 4); a swap round between two prompts then a prefill: the epoch rebuilt, bitwise
  (Review Focus 5); a withheld staging copy is a bounded-wait failure naming the layer, never a hang. FAIL.
- [ ] **Step 2: implement**; PASS. Commit `git commit -S -m "prefill: a chunk's non-resident experts stream into a staging
  ring on the copy engine while the previous layer computes, read through a static prefill table; bitwise the zero-copy
  prefill (spec 22c)"`.

### Task 4: speed (box) and the decisions

- [ ] **Step 1:** the whole model on two cards, `--pp 2`, int8 head: decode over the 36 A4 scenarios' prompts (prefill,
  then 192 greedy ids each, scenarios back to back so the cache carries across requests) with swaps off and with 22a's
  best (S, B, W, margin) and two neighbours - the h measured from the counters, t/s per scenario, interleaved arms,
  median of 3; prefill at pp4096 / pp32768 zero-copy against streaming (paired). Against 22a's projection. `docs/BENCHMARKS.md`
  "Qwen3.8-Flash-Next (spec 21)" gains the rows (the section's tier sub-heading). Commit `git commit -S -m "spec 22c:
  swaps' and prefill streaming's measured gain on the whole model (decisions 3 and 7)"`.

### Task 5: docs, the queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` ("22c as built"; decisions 3, 7 as
  measured), `docs/superpowers/plans/box-validation-queue.md` (row 38), `tools/box_validate/stages.sh` (a `row 38`
  block), this plan's status line

- [ ] **Step 1:** stages `r38.k0` (G0: every existing `q4_route` binary unchanged), `r38.host` (`qwen4exp_swap_test`,
  `qwen4exp_plan_test`), `r38.k1` (`kbins` of the `_CNT` binaries + the count case), `r38.swap` (`qwen4exp_swap_gpu_test`:
  synthetics, Intel's 18 / 37, the whole model when present), `r38.stream` (`qwen4exp_stream_test`), opt-in `r38.speed`
  (Task 4); `test_box_validate.py` passes; `--dry-run --only r38`. Commit `git commit -S -m "box: queue row 38 - spec
  22c, expert swaps and prefill streaming"`; after the box run, "22c as built", commit `git commit -S -m "docs: spec 22
  (22c as built)"`.

**Gate for the plan:** Mac - `qwen4exp_swap_test` (the protocol's invariants under randomised timings), names tests,
`qwen4exp_run`'s count case, the preprocessed-source identity of every existing `q4_route` variant, cmdlines additions
only. Box - G0; O2 (stress, withheld, poisoned, two cards) bitwise; streaming prefill bitwise zero-copy and
all-resident; the measured gains recorded and decisions 3 / 7 written with them.

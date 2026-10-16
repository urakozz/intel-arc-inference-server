# Spec 19c - DFlash in the decode loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** speculative decoding with the drafter end to end in the engine: taps written by the verify list, the commit-side context projection, the prefill tail, verify at M = K + 1 ≤ 8 bitwise to M = 1, host acceptance with the drafter's sparse q; D2, D3, D4.

**Architecture:** spec 19 §5 "Per iteration": draft list (19b) → spec 8's verify list at M = K + 1 (writes the taps of every row) → host acceptance (spec 8's `spec_accept`, q sparse over 16 candidates) → commit list (spec 8's GDN slot index and KV length, then the projection of rows 0..j into the ring, the row count read from Control). Prefill writes the taps of the positions inside the window and fills the ring.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** spec 19 (§5, §6 D2-D4, §7 19c); spec 8 §3-§4, §8 (A4, the Control index), §10 (AdaptiveK), §11 (draft vocabulary). Needs 19b merged.

## Global Constraints

- Branch `spec19c-dflash-loop` from main; box tree automatic; `tools/box.env` never committed.
- D0 holds (kernel binaries of every path without `--spec dflash` unchanged). Lock, detached, polled.
- Spec 8's GDN slots grow to K + 1 = 8 with `--spec dflash`: memory through the planner (7 x 151 MB on Qwen3.8).
- No recursive force deletes. Commit on the branch; no merge, no push.

## Review Focus

1. **Taps come from the verify rows that get committed** (rows 0..j); rejected rows' taps are never projected.
2. **The newest token** (the target's sample after acceptance) has no tap yet; it is the next block's row 0, as in vLLM.
3. **M = 5..8 bitwise** to M = 1 for logits and GDN slots (D3) on Qwen3.8; the verify list's attention at M rows reads the KV once.
4. **Sparse q:** the acceptance rule handles q = 0 outside the candidates (as spec 8 §11 does for V′) and the residual distribution stays finite.
5. **Prefix cache interplay:** without decision 5's plane (19d) a restored request starts with a cold ring; it must still be correct (lower acceptance only).

---

### Task 1: taps and the commit projection

- [x] (Mac half; on the card: r55.engine T1) The verify list writes taps (`tap_rows[M][n_taps][hidden]` bf16); the commit list projects rows 0..j into the ring. Test: projected ring rows equal 19b's context projection of the same taps. **Commit** `runtime: taps and the commit-side projection (spec 19c)`.

### Task 2: verify at M ≤ 8 and the loop

- [x] (Mac half; on the card: r55.engine D3 / P / D2, r55.gpu, r55.serve, r55.decode) Verify variants at M = 5..8 (from 19a's probe build), GDN slots to 8, `Engine::step_many` with the drafter as the proposer, `--spec dflash` / `--draft-model PATH` / `--spec-max` in `b70-decode` (ids mode) and `b70-serve`, the AdaptiveK cost row flat in K (`--spec-cost`). D3 and D2 on the box (golden set and A4, every K). **Commit** `runtime: DFlash speculative decoding (spec 19c, D2-D3)`.

### Task 3: sampling and the prefill tail

- [x] (D4 on the Mac; the prefill tail on the card: r55.engine PF) Sparse-q acceptance host tests (D4: 10^6 seeded samples within spec 8 M4's bars; no draft outside the candidates); prefill writes the window's taps and fills the ring; `prefill_split_*` with the drafter on. **Commit** `server: DFlash sampled acceptance and prefill taps (spec 19c, D4)`.

**Gate for the plan:** D2, D3, D4 green; D0 green.

## As built (2026-10-16) - and where the tree's current patterns replaced the plan's

Branch `spec19c-dflash-engine` (the operator's name; the plan said `spec19c-dflash-loop`), written on the Mac; box
queue row 55. **Status: the Mac half is done; D2, D3 and the prefill tail are the box's** (D4 is a host test and
passes; D0 is G0). Spec 19 §11 "19c as built" is the account; each deviation from this plan, with its cost if wrong:

- **The Engine sees a drafter through `runtime::TapDrafter`** (`runtime/tap_drafter.h`) implemented by
  `runtime::dflash::DFlashDrafter` - b70_dflash_runtime already links b70_runtime, so the Engine cannot name DFlash
  types. Cost if wrong: an interface to fold away.
- **The tap verify lists are a new capture mode** (`build_verify_taps`: the decode list at M rows, the 8-slot
  SPEC_SLOTS gdn_step over slots the Engine owns, the tap copies; no MtpBuffers - its head KV, hh and logits would cost
  up to 1 GB at 131k for nothing). The plan's `tap_rows[M][n_taps][hidden]` is 19b's `DFlashBuffers::taps`. The
  plan's "commit list" is `Engine::commit`: the slot index and pos on the host, then the drafter's projection list
  (19b's, `proj_pos` / `proj_n` in DFlashCtl) replayed synchronously. Cost: a mode in capture.cc; one submit more an
  iteration (fusing projection + next draft is a lever).
- **"Verify variants at M = 5..8 (from 19a's probe build)"**: 19a built them (B70_VERIFY_M8); 19c adds only
  `gdn_step_slots_M1..4_N8` (one slot count for every M) and the bf16 head at M = 5 / 6 / 8. They land in Task 1, since
  Task 1's lists need them to capture.
- **Task 1's test** is `dflash_engine_test` T1 on the box: the taps of M rows against M = 1's and the commit's ring
  rows against the rows K + 1 plain steps project (19b's dflash_draft_test already holds the projection list to the
  twin) - end to end, the row selection included.
- **"GDN slots to 8 ... memory through the planner"**: `TapVerifyDims` (7 x 151 MB) and the prefill staging (105 MB)
  are counted by `runtime::plan` with a drafter; 19b's `dflash_plan_test` assertion was widened to include them.
- **`Engine::step_many` with the drafter as the proposer** is `EngineAdapter::step_many` (the server's `--mtp auto`
  loop; `Engine` has no step_many on main): K per iteration by `server::AdaptiveK` over `MtpCost::dflash`, whose M = 5..8
  verify rows and flat draft row are DERIVED (row 29 has not run). `--spec-cost` replaces them.
- **`b70-decode`** runs `--spec dflash` greedy at a fixed K (`--spec-max`), not the auto policy: D2 is K-independent and
  a fixed K is the reproducible box check.
- **Refused for now, by name before the device:** `--pp 2` (taps across the hand-off: 19d), `--kv-cache int8` (kv8.cl
  is built for M <= 4), `B70_DECODE_ATTN=v1`, sycl-tla prefill and `B70_PREFILL_REPLAY` (the prefill taps are copies).
  Cost: those configurations cannot run DFlash yet.
- **The prefill tail** taps the call's last 2048 positions with one 2D region copy per tap per chunk
  (`prefill::Context::copy_2d` - new on the card) and projects them 8 rows a replay (~80-100 ms for a full window,
  derived; a GEMM projection is the lever). Positions before a call keep the ring's content (a cold ring after a
  prefix-cache restore: Review Focus 5).
- **Sampling:** the walk's Gumbel key is one fresh 64-bit draw from the request's RNG per iteration; q is the softmax of
  the realized selector scores at filter_probs' T over the 16 candidates; v1 drafters serve sampled requests plainly.
- **Added beyond the plan:** `tools/mac/clrun/gdn_slots_run` (the GDN state at the last accepted token, on the Mac's
  GPU - the operator's upstream question), `dflash_gpu_test` (D2 through EngineAdapter on the golden + A4 prompts,
  truncate_to, seeded sampling), CLI refusal tests, `r55.decode` / `r55.serve` / opt-in `r55.tps` stages.
- **Not done here:** D5 and the speed record (19d), an engine-side tap dump for D1 (r54.d1 still feeds 19a's dumps),
  the replay-state lever (spec 19 §11).

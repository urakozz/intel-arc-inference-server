# Spec 16b - pipeline parallel: decode across two cards

**Status (2026-10-06): built blind on branch `spec16b-pp-decode`, before 16a** - spec 16 §8 has
what was built and the defaults chosen in 16a's place (both hand-offs behind
`--pipeline-handoff copy|peer`, copy default; the split by bytes, `--pipeline-split auto`). The
CLI spelling is `--pp N` / `--pipeline-parallel-size N`, N = 1 or 2 (vLLM's; built as
`--pipeline 1|2` and renamed 2026-10-06, when the bench's prefill length became
`--prefill-length`). Tasks 1 and 2 are written
and host-checked (planner, protocol, CLI); their on-card halves, P1 / P4 and Task 3's S1 are box
queue row 22. `docs/BENCHMARKS.md`'s "Pipeline parallel (spec 16)" section waits for row 22's
S1 numbers.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-decode` / the engine run decode with layers `[0, s)` on device 0 and `[s, L)` on device 1, one captured list per device, the residual handed off once per token by 16a's chosen mechanism; P1 (bitwise equal to one card), P4, S1.

**Architecture:** spec 16 §3.1-3.2. A multi-device `Engine`: per-device persistent buffers for its own layers (GDN state, conv ring, KV), a `Control` per device, the embedding on device 0, final norm / `lm_head` / argmax on device 1, the token returned to device 0 each step. Capture builds list 0 and list 1 from the same walk, cut at layer `s`, with the hand-off appended / prepended.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec16-pipeline-parallel-design.md` (§3.1, §3.2, §4 P1, P2 decode, P4, §5 S1). Needs 16a's doc (`docs/probe-multi-gpu-2026-10-05.md`: the hand-off, the split).

## Global Constraints

- Both cards in the box; branch `spec16b-pp-decode` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- `--pp 1` (the default) is bitwise today's engine and its kernel binaries are unchanged. PP jobs take `~/b70-gpu.lock` once for both cards; detached, polled.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **P1 is bitwise:** logits, KV of every layer, GDN state, conv ring after a prefill + 64 greedy tokens equal the single-card run on device 0, byte for byte. Any difference is a bug, not noise.
2. **The token's way back** to device 0 (argmax / sampled id) each step: no host round trip beyond the existing fence wait; sampled requests write the id where both devices read it.
3. **`pos` stays in step** on both devices across resets, prefills, MTP commits (spec 8) and spec 7 restores.
4. **Failure modes (P4):** one card missing, peer access unavailable, a hand-off timeout: clear errors, no hang (bounded waits).
5. **The split from `ModelDesc`** (`layers_on(device)`), so Agnes (72 layers) and later models split without code changes.

---

### Task 1: placement and per-device buffers

- [ ] `ModelDesc` split; the loader places each layer's weights on its device and the embedding / head on theirs; per-device buffers; `Engine` holding two devices; host tests where possible, then on the box: load with `--pp 2`, memory report per device. **Commit** `runtime: per-device placement for pipeline parallel (spec 16)`.

### Task 2: the two captured lists and the hand-off

- [ ] Capture cut at `s`; the hand-off mechanism from 16a; the token back to device 0; `tests/runtime/pp_decode_test.cc` (checkpoint label): Review Focus 1 (P1 bitwise) and 2-4; replay determinism with `--pp 2`. **Commit** `runtime: decode across two cards (spec 16, P1)`.

### Task 3: speed

- [ ] Decode at 4k and 32k depth, `--pp 2` against one card, interleaved (S1: within 2 %); record per-device step time. `docs/BENCHMARKS.md` section "Pipeline parallel (spec 16)". **Commit** `spec 16b: PP decode speed`.

**Gate for the plan:** P1 bitwise, P4 refusals, replay bitwise, S1 measured, `--pp 1` unchanged (full suite).

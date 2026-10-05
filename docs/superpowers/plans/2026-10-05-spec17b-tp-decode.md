# Spec 17b - tensor parallel: decode

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--tp 2` decode: the loader slices every linear per spec 17 §2's table, each card runs its captured list at per-card shapes, the row-parallel GEMVs' partials are folded across cards in a fixed order (17a's winning mechanism), the `lm_head` is split by vocabulary with an argmax exchange; T1, T2 (decode), T4; D1, D3.

**Architecture:** spec 17 §2-3. `ModelDesc` with `tp_rank` / `tp_size` deriving per-card shapes; the loader slices packed int4 g64 on group and head boundaries; capture per card; `prep_res_fold` (or a separate kernel, per 17a) waits on the peer flag and folds card 0's partials then card 1's.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§2, §3, §4 T1, T2, T4, §5 D1, D3). Needs 17a's verdicts and spec 16b's multi-device engine.

## Global Constraints

- Both cards; branch `spec17b-tp-decode` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- `--tp 1` (default) bitwise today's engine; its kernel binaries unchanged. `--tp 2` and `--pp 2` refused together. Lock once for both cards; detached, polled.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Slicing on boundaries:** gate‖up rows 8704 per card = 136 g64 groups; head slices whole heads; the GDN qkv layout (q and k per k-head, v per v-head) regrouped per card correctly — a host test slices a synthetic packed tensor and checks each card's slice dequantises to the right rows.
2. **T1:** both cards' residual streams byte-identical after every layer (a debug tap comparing them), golden prompts + 64 tokens.
3. **T2:** against one card under the golden tie rule (not bitwise: summation order differs); the golden gates on `l0` / `l0-int8` with `--tp 2`.
4. **The argmax exchange** picks the same id on both cards; ties to the lower id; sampled requests read both vocab halves or merge per-card top-k (spec 17 decision 3).
5. **T4:** missing card, no peer access, fold timeout: clear errors, bounded waits.

---

### Task 1: slicing and per-card shapes

- [ ] `ModelDesc` TP fields; loader slicing; host tests (Review Focus 1); kernel variants at per-card shapes (`tools/kernel_cmdlines`: one-card variants unchanged). **Commit** `loader: tensor-parallel slicing at load (spec 17)`.

### Task 2: the folded collective and the head

- [ ] The fold (17a's mechanism) in the captured lists; the vocab-split head and argmax exchange; `tests/runtime/tp_decode_test.cc` (Review Focus 2-5); replay determinism with `--tp 2`. **Commit** `runtime: decode across two cards with folded partials (spec 17, T1-T2)`.

### Task 3: speed

- [ ] D1: `--tp 2` vs one card at 4k / 32k / 128k (bar >= 1.6x at 4k); D3: `--tp 1` within 1 %. BENCHMARKS "Tensor parallel (spec 17)". **Commit** `spec 17b: TP decode speed`.

**Gate for the plan:** T1 bitwise, T2 green, T4 refusals, replay bitwise, D1 / D3 measured, the full suite green at `--tp 1`.

# Spec 10a - decode attention at depth: profile and probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure where decode attention loses time at depth (P0), then sweep spec 10's levers 1-3 in a probe kernel with a correctness check per arm, and pick the v2 design.

**Architecture:** P0 uses the existing profile machinery (`tests/runtime/profile_capture_test.cc` and whatever it drives in `src/runtime/`; read it) at max_len 131072 and depths 4k / 32k / 64k / 128k. The probe is a standalone OpenCL C kernel plus harness in `tools/probe/` (the `probe_flash_attn.{cl,cc}` pattern) on real captured q and KV, comparing each arm with the current `attn_decode` + `attn_reduce` (`src/kernels/attn.cl`) output.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero.

**Spec:** `docs/specs/2026-09-28-spec10-decode-attention-at-depth-design.md` (§1 the derived budget, §2 levers, §3 constraints, §4 A1, §5 P0, §6 10a).

## Global Constraints

- Branch `spec10a-decode-attn-probe` from main; box tree `~/b70-inference-server-spec10a` automatically. Copy `tools/box.env` from the main checkout if missing; never commit it. Symlink `oracle-out*` dirs if a golden gate is run.
- Other agents share the GPU: every GPU command under `flock ~/b70-gpu.lock`, detached, background until-loop polling for `ALLDONE rc=`. Record `uptime` with every timing table; interleaved arms, median of 3.
- Probe code only in this plan: `src/` behaviour unchanged (if a capture hook is needed to dump q/KV, it is compiled only with a probe define).
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Real data at depth:** q and KV from a real prefill of `tests/golden/prompts/long32k.ids` (repeated to 128k with `tools/probe/mk_long_ids.py`), one FA layer, not random data: softmax concentration at depth is what makes precision matter.
2. **Every arm at M = 1 and M = 4** (spec 8's verify): an arm that only wins at M = 1 is not a winner.
3. **Positions per work-group derived from `pos` on the device:** the probe must run the arm at several depths with one compiled binary, reading the stride from a control word, as production must (the captured list is fixed).
4. **Determinism:** each arm run twice gives bitwise identical output.
5. **The partial table's traffic** at 128k is measured, not only derived (bytes written and read per layer, from the arm's own counters or its buffer size and a timing split).

---

### Task 1: P0 profile

- [ ] **Step 1:** per-kernel times of `attn_prep`, `attn_decode`, `attn_reduce` at depths 4096 / 32768 / 65536 / 130816, max_len 131072, M = 1; plus M = 4 if spec 8's verify lists are on main (else say so).
- [ ] **Step 2:** effective GB/s of `attn_decode` against the KV bytes (64 KiB per position per token, spec §1), and `attn_reduce`'s share.
- [ ] **Step 3:** `docs/probe-decode-attn-2026-09-28.md` with the table; **Commit** `probe: decode attention at depth, per-kernel profile (spec 10 P0)`.

### Task 2: the probe kernel and harness

**Files:** `tools/probe/probe_decode_attn.cl`, `tools/probe/probe_decode_attn.cc`, registration in `tools/probe/CMakeLists.txt`.

Defines: `PPW` (positions per work-group: 64, 256, 512, 1024, 2048, or `0` = derived from `pos` and a target work-group count read from a control word), `LOAD` (0: today's SLM slabs, 1: `vload16` rows into registers, 2: 2D block reads as in `src/kernels/prefill/pf_flash_attn.cl`), `PREFETCH` (0/1), `REDUCE` (0: separate reduce launch over partials, 1: last-work-group reduce with a device counter and a fixed combine order), `M` (1, 4). Correctness per arm against the current kernels on the same inputs: per (q head, m) cosine >= 0.99999, max abs error; bitwise repeatability.

- [ ] **Step 1:** harness with the current kernels as arm 0 (reproduces production times within 3 %).
- [ ] **Step 2:** sweep `PPW` alone, then `LOAD` on the best `PPW`, then `PREFETCH`, then `REDUCE`, at depths 32768 and 130816, M = 1 and 4; one lever at a time, interleaved against arm 0.
- [ ] **Step 3:** table in the doc; the winner; the derived decode t/s at each depth (production step time minus arm 0's attention time plus the winner's).
- [ ] **Step 4: Commit** `probe: decode attention levers at depth (spec 10 P1)`.

### Task 3: the design verdict

- [ ] In the doc: the v2 design (defines, grid, the `Control` stride word, reduce choice), the derived A-F3 numbers against the bars (>= 23.6 / 21.0 / 17.3 t/s at 32k / 64k / 128k), and a stop if no arm reaches within 10 % of the bars (then record for the operator's call). `docs/README.md` line. **Commit** `probe: decode attention v2 design verdict (spec 10)`.

**Gate for the plan:** P0 and the sweep recorded, every arm correctness-checked, the verdict. Hand back with the tables.

# Spec 16a - multi-GPU probe: peer access, hand-off options, the TP remote-partial arm

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure what pipeline parallel (spec 16) and tensor parallel (spec 17) depend on, with both B70s in one Level Zero context: peer access and bandwidth, the three PP hand-off options at 10 KB and 20 MB, and the TP "remote partial" fold; propose the hand-off mechanism and the layer split.

**Architecture:** probe binaries in `tools/probe/` using the repo's `l0::` wrappers (`src/l0/`), one context over both devices; no engine changes.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero.

**Spec:** `docs/superpowers/specs/2026-10-05-spec16-pipeline-parallel-design.md` (§2 decisions 1-3, §5 P0 including the spec 17 arm); `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§2 decision 1).

## Global Constraints

- **Needs both cards in the box** and `ZE_AFFINITY_MASK` exposing both (the default mask today is device 0 only in several scripts: override it for these runs).
- Branch `spec16a-multi-gpu-probe` from main; box tree `~/b70-inference-server-spec16a` (automatic). Copy `tools/box.env` if missing; never commit it or its contents.
- The GPU lock `~/b70-gpu.lock` is taken once for both cards. Detached runs polled with a background until-loop. Interleaved arms, median of 5, `uptime` recorded; both cards free of other DRM holders for timing rows.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **System-scope atomics** for every flag a peer writes: verify in the generated SPIR-V or assembly that the polling atomic is cross-device scope, and run a negative control with a plain load to show the hazard (with a timeout, never a hang).
2. **The peer-written buffer is its own page-aligned allocation** holding nothing the local device writes; a control places it inside a shared allocation and checks neighbouring data for corruption (report, do not ship).
3. **Replayable handshakes:** sequence counters live in device memory and are advanced by the kernel; a captured list replayed 1000 times keeps progressing with frozen arguments.
4. **Both directions:** device 0 → 1 and 1 → 0 bandwidth and latency (the second card is 3 % slower in compute; check links are symmetric).
5. **Timeouts:** every probe that waits on the peer has a bounded wait and reports a failure instead of hanging.

---

### Task 1: capability and bandwidth

- [ ] `tools/probe/probe_p2p.{cl,cc}`: `zeDeviceCanAccessPeer` both ways; one context with both devices; copy-engine copies device→peer at 10 KB, 1 MB, 20 MB, 64 MB; kernel peer writes (a kernel on device 0 storing into device 1 memory) at the same sizes; both directions; GB/s and µs. `docs/probe-multi-gpu-2026-10-05.md`. **Commit** `probe: two-B70 peer access and bandwidth (spec 16 P0)`.

### Task 2: the PP hand-off options

- [ ] Inside replayed lists: (a) copy + cross-device event (list 0 signals, list 1 waits), (b) peer write + system-scope flag polled by device 1's first kernel, (c) host-orchestrated; for a 10 KB residual row (decode) and a 20 MB chunk (prefill); latency per hand-off and the cost added to a replayed ~800-kernel list. Review Focus 1-3, 5. **Commit** `probe: pipeline-parallel hand-off options (spec 16 P0)`.

### Task 3: the TP remote-partial arm (spec 17)

- [ ] A decode GEMV (K 5120 → split K 2560 per card, N 5120, the down / o_proj shape) on each card writing its fp32 split-K partials locally and into the peer's buffer; a `prep_res_fold`-style kernel on each card waiting for the peer flag and folding card 0's partials then card 1's (fixed order); check both cards' outputs are bitwise identical and equal to a host reference within fp32 summation tolerance; latency added per fold against (i) one card's fold, (ii) a separate all-reduce kernel. Also the 42 MB prefill-size partial transfer. **Commit** `probe: tensor-parallel remote-partial fold (spec 17 P0 arm)`.

### Task 4: proposals

- [ ] In the doc: the PP hand-off choice; the PP split proposal (`s` = 32 or 33 / 31 given the second card's 3 % slower prefill, from a per-layer timing on each card); the TP fold verdict (fold vs separate kernel) with the derived TP decode estimate; `docs/README.md` line. **Commit** `probe: multi-GPU P0 verdicts (specs 16 and 17)`.

**Gate for the plan:** every P0 number recorded with both cards idle, the safety controls run (Review Focus 1, 2, 5), the three proposals written. Hand back with the tables.

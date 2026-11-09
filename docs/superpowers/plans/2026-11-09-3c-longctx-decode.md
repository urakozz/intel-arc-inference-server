# Long-context decode probe implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Measure the attention share of Qwen3.8 plain decode and M = 4 verification at 4096, 32768, 81920 and 150000 positions, then decide whether attention merits a dedicated item.

**Architecture:** Reuse the runtime's existing captured lists and per-launch profiling API from a standalone probe. Populate real KV with the existing tiled prompt and production prefill. Compare uninstrumented wall time separately from timestamp anatomy, after warm-up and in alternating rounds; no engine or kernel changes.

**Tech Stack:** C++17, Level Zero timestamps, existing box build and GPU locks, Python result validation.

**Spec:** Operator's `codex-3c.md` brief and `docs/open-work.md` row 3c, supplied 2026-11-09.

## Global constraints

- Stay on `kv8-prefill-fast` in the existing worktree; signed `3c:` commits without trailers. No push or merge for 3c.
- Mac never compiles. Use `tools/box.sh`, JOBS=16, all box artifacts below `~/b70` and logs below `~/b70/logs/3c`.
- GPU smoke jobs use a card lock; timed comparisons hold the whole-box lock and select one device through the probe's own argument. Never nest the lock helpers (their documented protocol forbids it).
- Long jobs run detached. Retain the operator's already authorized 1–2 minute takeover polling interval.
- No hostnames or IP addresses in files, no engine flags or environment variables, no engine changes.

## Review focus

- A depth of 150000 must fit actual allocations and leave room for measured rows.
- Profile timestamps perturb execution: report their sum and attention share separately from unprofiled wall time.
- M = 4 must bind the production depth-dependent attention selection, including KV form.
- Verify is not a complete speculative cycle: report rows/s separately from accepted serving tokens/s, and inspect whether its head actually executes attention.
- Missing arms, launches or nonfinite timings must fail result validation rather than produce an incomplete verdict.

## Task 1: Reproducible anatomy probe

**Files:** Create `tools/probe/probe_longctx_decode.cc`; modify `tools/probe/CMakeLists.txt`.
**Interfaces:** Consume `runtime::build`, `runtime::build_verify`, `ProfileEvents`, `Engine::prefill` and the existing attention selection table. Emit machine-readable per-round wall and kernel-category timings with depth, KV form, M, capacity and launch count.

- [x] Confirm existing CLI limitations and select the smallest probe extension that covers both KV forms and M values without engine edits.
- [x] Implement populated KV, fixed-depth replay, warm-up, alternating M = 1 / M = 4 arms, instrumented and uninstrumented captures, and per-label attention categorization. Keep the default int8 output head consistent across arms.
- [x] Build on the box and run a short 4096-depth smoke under a card lock. Check positive finite times, complete launch categorization, correct attention variants and adequate capacity; exercise invalid capacity rejection.
- [x] Commit the verified probe with a signed `3c:` subject.

## Task 2: Complete interleaved measurements and decision

**Files:** Create `docs/probes/probe-longctx-decode-2026-11-09.md`; modify `docs/open-work.md`; append uncommitted `REPORT-3e.md`.
**Interfaces:** Consume complete per-round probe output; produce tables for all sixteen depth/KV/M combinations and the operator's >=25% / verify-scaling verdict.

- [x] Run three paired bf16/int8 rounds in alternating order, with warm-up and M-arm interleaving, on a quiet box. Populate the cache incrementally through all four depths per engine load; never keep two engines on a card.
- [x] Validate all expected rows, timestamps, launch counts and arithmetic. Compare M = 4 attention per row with M = 1 and distinguish kernel sums from wall throughput.
- [x] Record method, raw-log locations, tables and limitations. Move 3c to Closed recently. If the specified threshold is met, add a measured follow-up to Now, plan only.
- [x] Obtain one fresh final review of probe/method/results, resolve material findings, and verify documentation against the complete measurements.
- [x] Commit the probe record and work-list update with a signed `3c:` subject. Leave the report uncommitted and do not merge 3c.

Completed 2026-11-09: probe code `bfa3be0d`; full box build, both cache-form
smokes and five summary tests passed. All 48 measurements validated. Final
review found no Critical, Important or Minor findings; all 16 anatomy rows
and four paired-ratio rows were also checked against generated output.
GO under the operator's plain-step threshold; 3f proposed only. No engine
changes, no 3c merge or push, report left uncommitted.

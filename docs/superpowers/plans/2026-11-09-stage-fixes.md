# Validation stage GPU visibility fixes

> Use superpowers:executing-plans inline. The supplied task authorizes implementation and signed branch commits, with no merge or push.

**Goal:** make the affected two-card validation stages see the same two devices as their successful hand invocations, without weakening tests.

**Spec:** operator's `/private/tmp/claude-501/-Users-urakozz-CLionProjects-b70-inference-server/5db6469f-7011-486c-a972-2a428c1f57d9/scratchpad/codex-stage-fixes.md`.

**Evidence:** newest recorded JUnit output: Bolt requires its checkpoint and two GPUs; PP says Level Zero shows 1. The validation orchestrator exports its single `DEVICE` as `ZE_AFFINITY_MASK`, even when the stage takes the whole-box lock. The hand and validation builds use the same Bolt/drafter settings. Existing two-card stages pass `ZE_AFFINITY_MASK=0,1` through `run_tests`' environment argument; use that established pattern only in the affected stages.

**Constraints:** work in `.claude/worktrees/stage-fixes`; signed `stage-fixes:` commits, no trailers, merge or push. No test loosening, runtime changes, new engine options or unrelated stage changes. C++ builds only on the box with JOBS=16. GPU jobs retain the normal lock. Detached long jobs; no frequent polling. Preserve other agents' trees and jobs.

- [ ] Bolt: add a failing dry-run regression proving `r65.d2`, `r65.prefill`, and `r65.prefix` expose both cards, including under orchestrator device 1. Keep `r65.serve`'s existing two-card invocations. Observe RED; set the three `run_tests` environment arguments; observe GREEN and sign the fix.
- [ ] PP: add a failing dry-run regression for `r75.pp` under orchestrator device 1. Observe RED; set its `run_tests` environment argument, preserving the whole-box stage kind and test selection. Observe GREEN and sign the fix.
- [ ] Run shell syntax checks and the host validation tests. Build the branch on the box and run the affected tests/stages with proper locks; compute registry prerequisite closure if using the full stage driver. Record any legitimate skip rather than changing test requirements.
- [ ] Read-only final review; update only the two existing Waiting rows with measured outcomes, and write the requested uncommitted `REPORT-stage-fixes.md` with causes, fixes, results and branch head. No integration.

**Review focus:** ensure whole-box locking remains; overrides reach ctest and reuse metadata, not just the outer shell; device-1 orchestrators still select physical devices 0 and 1; server invocations already correct remain intact; failed/missing data and SKIP statuses remain honest. The dry-run regressions pin affected command generation; actual box runs cover GPU enumeration and test execution.

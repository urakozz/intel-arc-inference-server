# Spec 19e (decide) - prompt-lookup proposer for every model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a model-free speculative proposer: drafts are the continuation of the longest earlier match of the last n ids in the request's context (optionally earlier requests of the session); verify and commit as spec 8; works for every model, K2-Horizon included; D2 per model, D5 on the opencode recording.

**Architecture:** spec 19 §3 approach C. A host-side matcher (suffix automaton or rolling-hash index over the request's ids, updated as ids commit) hands draft ids to the existing verify list (MTP's or DFlash's verify path, taking externally supplied drafts). Acceptance is greedy, or sampled with the proposal as a point mass (lossless: accept with probability p(id), else sample from p with that id removed and renormalised).

**Tech Stack:** C++17 (host), Level Zero for the verify path, Python for the replay analysis.

**Spec:** spec 19 (§3 C, §4 decision 1, §7 19e, §9). Operator decision 1 decides whether this stage runs; for K2 it needs spec 18b merged.

**Status (2026-10-05, branch `spec19e-prompt-lookup`, host side only, no box):**
- Task 1 done **on A4 only** (`docs/probe-prompt-lookup-2026-10-05.md`): lookup alone 1.44x
  over plain (n = 3, K <= 3), 0.92x of projected `--mtp auto`; combined with MTP +3-6 %. The
  opencode recording, which decides, is not recorded yet: rerun the tool on it.
- Task 2, host part done: `server::PromptLookup` (`src/server/prompt_lookup.h`, the matcher),
  `server::accept_point_mass` (lossless sampling, 1e6-sample chi-square tests),
  `EngineIface::verify_k` / `step_drafts`, `b70-serve --spec off|mtp|lookup`,
  `--spec-min-match`, `--spec-max`, `--spec-cost`, `--spec-history`, AdaptiveK with free
  drafts; mock-tested (`prompt_lookup_test`, `lookup_server_test`, `spec_accept_test`).
  `EngineAdapter::step_drafts` is written (Qwen3.8: the MTP verify lists take the drafts in
  `cur_token[1..k]`, which `Engine::verify` already reads) but **has never run**: D2 per
  model is box queue row 12. A model without the MTP head (K2) needs its own verify lists
  (TODO in `serve_adapters.h`).
- `--spec mtp+lookup` not built: Task 1 does not show it paying on A4 (below spec 19's
  +10 %); the recording decides.
- Task 3 open (box).

## Global Constraints

- Branch `spec19e-prompt-lookup` from main; box tree automatic; `tools/box.env` never committed.
- D0 holds. Lock, detached, polled.
- No recursive force deletes. Commit on the branch; no merge, no push.

## Review Focus

1. **Lossless sampling** with a point-mass proposal (q = 1 on the proposed id), as above; host tests with 10^6 seeded samples.
2. **Matcher cost** is host-side and O(1) amortised per committed id; it never stalls the device list.
3. **Minimum match length and K:** a draft only when the match is ≥ n ids (tuned on the recording); K from spec 8 §10's policy with a draft cost of ~0.
4. **Agentic patterns** (file contents echoed into edits, tool output quoted back, unchanged code around a change) measured on the opencode recording before any engine change (Task 1).
5. **Composition:** with MTP or DFlash present, lookup supplies the drafts when its match is long and the model proposer otherwise (one proposer per iteration).

---

### Task 1: offline acceptance on the recording (no box)

- [x] (A4 only; the recording pending) `tools/spec/lookup_accept.py`: replay the opencode recording's requests (ids from `--log-requests`), measure tokens per verify at K = 1..7 and n = 2..6, alone and combined with MTP's measured acceptance; record. **Commit** `probe: prompt-lookup acceptance on the opencode recording (spec 19e)`.

### Task 2: the proposer and the loop

- [ ] Host matcher + `--spec lookup` (and `--spec mtp+lookup` if Task 1 shows the combination pays); external draft ids into the verify list; D2 per model; the point-mass sampling tests. **Commit** `server: prompt-lookup speculative decoding (spec 19e)`.

### Task 3: speed

- [ ] D5 rows on the recording for Qwen3.8 (vs `--mtp auto`) and, after 18b, K2-Horizon (vs plain). **Commit** `spec 19e: prompt lookup - the record`.

**Gate for the plan:** D2 per model green; the recording rows recorded; the default decided per model.

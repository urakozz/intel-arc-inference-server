# Spec 21 v2a - Qwen3.8-Flash-Next whole on two cards: the tiering probe, then the tier on `--parallel pp`

**Status (2026-11-11): planned.** Tasks 1-3 (the probe and its projection) need no operator go; Task 4 onwards begins
on the operator's go after Task 3's record. Tensor parallel for Flash-Next (spec 21 v2 §3 (b)) gets its own plan
(21v2b), written after Task 3 if the projection supports it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** the whole 48-layer Flash-Next runs and is served on two B70s - the most-used experts in VRAM by a recorded
route profile, the rest read in place from pinned host memory - bitwise equal to the resident engine wherever the
experts live, judged against an f32 reference, at ≥ 30 t/s decode.

**Spec:** [spec 21 v2](../specs/2026-11-11-spec21v2-flash-next-v2-design.md) (the design, the gates in §7 / §8, the
open questions in §9). Detail that still holds, step by step, lives in the old plans - reuse it, do not re-derive it:
[22a](2026-10-10-spec22a-offload-probe.md) Task 5 (the box runs), [22b](2026-10-10-spec22b-expert-table-and-cache.md)
Tasks 1-5 (the table, the fill, the loader, O0 / O1 / O3), [22c](2026-10-10-spec22c-swaps-and-prefill-streaming.md)
Tasks 1-3 (swaps, prefill streaming), [22d](2026-10-10-spec22d-flash-next-whole-served.md) Tasks 1-4 and 7 (serving).
Where those plans name a `--expert-*` flag, the engine decides instead (zero-config); test-only choices are
environment variables (`B70_Q4_RESIDENT`, `B70_Q4_EXPERTS`, a placement pin).

## Global Constraints

- Own worktree under `.claude/worktrees/<branch>`; branch `spec21v2a-tiering` from main. Commit (signed, no trailers)
  after every step that changes a file: `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S -m "..."`. Never push.
- The Mac never compiles C++: build and test with `tools/box.sh build|run`. Every GPU job through
  `tools/gpu_lock.sh card <N> -- ...` or `tools/gpu_lock.sh box -- ...`; CPU reference jobs take
  `/tmp/b70-cpu-model.lock`; a CPU job above 64 GB RAM also holds the box GPU lock. Long jobs detached
  (`nohup setsid`, logs under `~/b70/logs/flash-next-tier/`), checked at most every 30 min. Stop jobs only with
  `tools/box_stop.sh`. Nothing deleted with `rm -rf` (move to `~/b70/trash/<date>`). No IPs or hostnames in files.
- **Host memory both cards map: at most 4 GiB per allocation** (use 2 GiB pieces). This applies to the probe as much as
  to the engine.
- F0: every existing kernel binary keeps its name, command line and bytes (`tools/kernel_cmdlines` additions only).
- No host action between layers; captured lists do not change per step. Residency never changes arithmetic.
- Timings: interleaved pairs after warm-up, medians; say "both cards" when both are held.

---

### Task 1: the hit-rate curve on the recorded traces (box CPU, no GPU)

The 2026-11-09 traces (`$DATA/oracle-out-q4exp-traces/`, 39 sources) exist; `r36.curve` has never run.

- [ ] **Step 1: rebase the curve and the projection on Intel's forms.** `tools/oracle/offload_curve.py`: `pick_budget`
  and `cmd_project` default to `dense="int4"` and `STOP_TPS = 40.0` on int4 dense - make the bf16 (Intel's) row the
  judged one and the stopping rule spec 21 v2 §8's (`pp` < 30 and `tp` < 50 t/s -> STOP). `qwen4exp_oracle.sh curve`
  picks f from the `dense == "bf16"` budget row (two cards, 32k, head and prefill: 0.724 derived). Update
  `test_offload_curve.py`; PASS.
- [ ] **Step 2: add the `tp` layout to `project`:** per card half of D's split part plus the replicated
  hyper-connections (spec 21 v2 §3 (b)), R / 2 per card, both links at P, 0.34 ms of all-reduces; print (a) and (b)
  side by side for each f / h. Tests with spec 21 v2 §3's table as the expected numbers (± 0.5 t/s).
- [ ] **Step 3 (box CPU): run** `tools/box_validate.sh --only r36.curve` (no checkpoint read; minutes).
- **Gate:** `oracle-out-q4exp-offload/{curve,swap,mtp,prefill,profile}.txt` exist; the record states h (held out) at
  f = 0.70 and 0.724, whether swaps beat the static fill by ≥ 1 point of h, the MTP verify union per layer at K = 1..3,
  and the prefill bytes per chunk; `profile.safetensors` written.

### Task 2: the link and memory probe (box, both cards, ~1-2 h)

- [ ] **Step 1: make `probe_offload pin` safe.** Its one-range arm doubles a single pinned range from 1 GiB and reads it
  on every card: with `--dev both` a range above 4 GiB is exactly the 2026-11-11 hang. Cap any range mapped by both
  cards at 4 GiB (2 GiB pieces for the multi-range arm), run the one-range arm on one card only; the target is the
  misses mirror (~17.7 GB) + the PLE table (51.8 GB), not 116 GB.
- [ ] **Step 2: add a `peer` arm** to `probe_offload`: card 0's table kernel reading expert blocks that live in card 1's
  VRAM over P2P, and a copy-engine peer copy, at the expert sizes of P0.2 (spec 22 §3) - the evidence for spec 21 v2
  §3 (c).
- [ ] **Step 3: build** (`tools/box.sh build`), the probe's two kernels' first compile (`r36.build`).
- [ ] **Step 4 (box, both cards): run** `r36.links`, `r36.links_load` (stop and report below Gen3 x8), `r36.copy`,
  `r36.zc` (each card, then both cards at once - the `tp` case), `r36.replay`, `r36.hostbw`, `r36.pin`, `r36.alias`,
  the new peer arm, and `r32.ple_rate`.
- **Gate:** both links' generation and width recorded; host-to-device rate per card and both at once; whether a mixed
  launch costs max(VRAM part, host part) or their sum (which of spec 22 §2's two formulas holds); 10^4 replays with
  every byte checked; no alias; pinned pieces up to the target with no hang; the P2P read rate.

### Task 3: the projection and the record

- [ ] **Step 1:** `tools/box_validate.sh --only r36.project` with `PROJECT_BW` from row 32 if measured.
- [ ] **Step 2: the record** `docs/probes/probe-offload-<date>.md`: Task 1's curve, Task 2's numbers, the projection of
  (a) `pp` + tier and (b) `tp` + tier (and (c) if the peer rate allows), decode at 4k / 32k / 128k, prefill at 4k,
  against the yardsticks (Strata one B70 30-41 t/s and 980-1000 t/s prompt at 4k, two B70s 66 t/s; XeStrata ~28 t/s).
  Its last line: STOP or GO-CANDIDATE by spec 21 v2 §8.
- [ ] **Step 3:** open-work item 4 updated with the measured h and t/s; the operator decides the go and the layout.
- **Gate:** the record exists with every number marked measured / derived; the operator's go (or stop) is recorded.

### Task 4: the whole-model f32 reference (box CPU, beside Task 5)

- [ ] **Step 1:** the f32 reference of all 48 layers on `q4exp_4k` (+ 32 forced continuation steps), as the 18 / 37
  layer ones (`tools/probe/` and the plan [2026-11-08-flash-next-f32-reference.md](2026-11-08-flash-next-f32-reference.md));
  ~90 GiB estimated, so it holds `tools/gpu_lock.sh box` while it runs. Queue it with the golden-set remake
  (`#HOLD-sj q4exp-golden-scores`) so the CPU is not loaded twice.
- [ ] **Step 2:** the bf16 CPU reference's own f32-rejected tokens and routes on the same prompts (the twin count, spec
  21 v2 §7) - the bar for decode.
- **Gate:** the paired reference files published with their identity checks; the twin counts recorded.

### Task 5: the table, the mirror and the static fill on `--parallel pp` (box)

Plan 22b Tasks 1-5, with these changes: the mirror is the misses only, per layer (≤ 1.34 GB a range); the profile is
`<snapshot>-expert-profile.safetensors` from Task 1, filled with no flag (id order and a startup line without it); the
budget per layer is the profile's curve if Task 1 shows ≥ 0.5 point of h over uniform, else uniform; the MTP head's
experts resident; the planner stops refusing the whole model and refuses only when the non-routed part or the host
memory does not fit, naming the bytes.

- [ ] **Step 1:** the `_TBL` form of `q4_moe.cl` / `q4_pf_moe.cl` (`Q4_EXPERT_TABLE`), every existing variant
  preprocessed token for token as main's (22b Task 1).
- [ ] **Step 2:** the fill and the planner with a cache (host tests; 22b Task 2).
- [ ] **Step 3:** the loader - arenas, the pinned misses mirror in ≤ 2 GiB pieces, the table, the tag / checksum checks
  (22b Task 3).
- [ ] **Step 4:** the engine binds the table; `b70-decode` runs the whole model on `--parallel pp`; O0 and O1 (22b Task 4).
- [ ] **Step 5 (box):** the whole model end to end: load time, the memory line, decode and prefill at 4k, Task 4's count
  rule on 48 layers, decode's twin count.
- **Gate:** O0 bitwise (ids, logits, routes, selections, states; decode, prefill, verify); O1 bitwise at 0 / 50 / 100 %
  random and adversarial; O3 (tags, checksums, a hang-free refusal); the whole model's prefill passes the count rule and
  decode the twin count against Task 4; **decode ≥ 30 t/s** at 4k, one request, MTP off.

### Task 6: swaps (only if Task 1 says they pay) and prefill streaming

Plan 22c Tasks 1-3 with basalt's policy (spec 21 v2 §4): counters in the route kernel (`_CNT`), promotion only past an
age margin (start at basalt's 1280 lookups; set from the trace simulation), copies in 1 MB pieces between steps,
admitted after the copy's event, placement pinned by an environment variable in tests. Prefill: a staging ring on the
copy engine, filled while the previous layer computes.

- **Gate:** O2 (a swap every step bitwise the no-swap run; a withheld copy never read); decode with swaps ≥ the static
  fill's on the opencode recording; prefill at 4k recorded against 1000 t/s.

### Task 7: serving the whole model

Plan 22d Tasks 1-4 and 7 (zero-config): `--max-len auto` by spec 21 v2 §6, `--max-num-seqs` per its design, MTP with
the resident head (`--spec auto` from measured costs), the prefix cache restoring bitwise with the tier on, the
tool-call test against `r34.a4_ref` (run `r34.fixture` / `r34.accept` / `r34.a4_ref` on the CPU before this task;
`r34.accept` with the bf16 head and the RTN head, spec 21 v2 §5.5), passkey at the chosen context; the speed rows in
docs/BENCHMARKS.md and the Flash-Next section of docs/18-running-models.md.

- **Gate:** `b70-serve` serves the whole model with no `--layers`; prefix cache restores bitwise; tool-call test: every
  first difference passes the f32 judge; passkey passes; decode ≥ 30 t/s served (spec 21 v2 §8 (a)).

### Task 8 (after Task 7, each on its own measurement): decode levers

Union-aware verify MoE; int8 hyper-connections; a fused QSA decode kernel and int8 KV on the QSA layers with
OpenVINO `3b0ee89623` as the reference; the dense-bytes change if the operator allows it (spec 21 v2 §5, §9 question
1). Each lands only with its measured gain and the f32 judge passing. Tensor parallel is plan 21v2b.

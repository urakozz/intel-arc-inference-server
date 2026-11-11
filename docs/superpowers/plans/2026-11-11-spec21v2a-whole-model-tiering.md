# Spec 21 v2a - Qwen3.8-Flash-Next whole on two cards: our AutoRound export, the tiering probe, the tier on `--parallel pp`

**Status (2026-11-12): planned, revised with the operator's decisions** (spec 21 v2 §0, including the same day's
follow-up: tuned AutoRound, our export only, no comparison with Intel's checkpoint). Task 1 is done on branch
`flash-next-quant` (`tools/quantize/quantize_qwen4exp.sh`, proven on the tiny model); Tasks 2-3 are the operator's
weekend run (open-work's "Quantise Flash-Next" row). Tasks 4-6 need no further operator go except Task 6's go; Task 8
onwards begins on the operator's go after Task 6's record. Tensor parallel with the expert split is plan
[21v2b](2026-11-12-spec21v2b-tensor-parallel.md), after this plan's Task 10.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** our own AutoRound int4 g64 export of Flash-Next (head and MTP experts included), then the whole 48-layer
model run and served on two B70s with the layers split - the most-used experts in VRAM by a recorded route profile,
the rest read in place from pinned host memory - bitwise equal to the resident engine wherever the experts live,
judged against an f32 reference, at ≥ 30 t/s decode.

**Spec:** [spec 21 v2](../specs/2026-11-11-spec21v2-flash-next-v2-design.md) (our weights §2, the bytes §3, the tier
§6, the gates §10 / §11, the open questions §12). Detail that still holds lives in the old plans - reuse it, do not
re-derive it: plan 21q (`git show 9ef48bb2^:docs/superpowers/plans/2026-10-09-spec21q-qwen4exp-autoround.md`: the
script's stages, the ignore list, per-expert coverage, F6), [22a](2026-10-10-spec22a-offload-probe.md) Task 5 (the box
runs), [22b](2026-10-10-spec22b-expert-table-and-cache.md) Tasks 1-5 (the table, the fill, the loader, O0 / O1 / O3),
[22c](2026-10-10-spec22c-swaps-and-prefill-streaming.md) Tasks 1-3 (swaps, prefill streaming),
[22d](2026-10-10-spec22d-flash-next-whole-served.md) Tasks 1-4 and 7 (serving). Where those plans name a `--expert-*`
flag, the engine decides instead (zero-config); test-only choices are environment variables (`B70_Q4_RESIDENT`,
`B70_Q4_EXPERTS`, a placement pin).

## Global Constraints

- Own worktree under `.claude/worktrees/<branch>`; branch `spec21v2a-tiering` from main. Commit (signed, no trailers)
  after every step that changes a file: `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S -m "..."`. Never push.
- The Mac never compiles C++: build and test with `tools/box.sh build|run`. Every GPU job through
  `tools/gpu_lock.sh card <N> -- ...` or `tools/gpu_lock.sh box -- ...` (AutoRound on a card included); CPU reference
  jobs take `/tmp/b70-cpu-model.lock`; a CPU job above 64 GB RAM also holds the box GPU lock. Long jobs detached
  (`nohup setsid`, logs under `~/b70/logs/flash-next-tier/` and `~/b70/logs/flash-next-quant/`), checked at most every
  30 min. Stop jobs only with `tools/box_stop.sh`. Nothing deleted with `rm -rf` (move to `~/b70/trash/<date>`). No IPs
  or hostnames in files.
- **The operator runs every download** (the bf16 original included); the plan gives the exact command and checks the
  result.
- **Host RAM:** ~115 GB usable. **Host memory both cards map: at most 4 GiB per allocation** (use 2 GiB pieces) - the
  probe as much as the engine.
- F0: every existing kernel binary keeps its name, command line and bytes (`tools/kernel_cmdlines` additions only).
- No host action between layers; captured lists do not change per step. Residency never changes arithmetic.
- Timings: interleaved pairs after warm-up, medians; say "both cards" when both are held.

---

### Task 1: the bf16 original on the box and the AutoRound toolchain on the tiny model (done 2026-11-12, `flash-next-quant`)

What was built instead of the steps' first guesses (tools/quantize/README.md, "Qwen3.8-Flash-Next"): AutoRound at
the operator's fork commit `a6cf7487` (intel/auto-round `b58069ea` of 2026-11-11 + cached OpenCodeInstruct) rather
than 0.17.0 `6afaecdb` - per-block resume, the vectorised QSA indexer, its own MTP export; calibration on
nvidia/OpenCodeInstruct through the chat template (operator 2026-11-12); the tuned path only (no RTN arm, operator 2026-11-12); `lm_head` packed by
our RTN g64 rule in a `head` stage (AutoRound RTNs the head on the tuned path too, and writes its scales BF16); the MTP
head's routed experts RTN int4 g64 by AutoRound at export, its attention and shared expert bf16; the vision tower bf16
and not counted by the checks. The tiny model is `tools/quantize/qwen4exp/make_tiny_vl.py` (the fused original's
form with a vision tower and the processor files: AutoRound takes the multimodal path for this architecture).

- [ ] **Step 1 (operator): the download** (running since 2026-11-11 to `~/b70/usb/hf-hub`, linked into the HF cache).
  Check: `tools/quantize/quantize_qwen4exp.sh check-download [--sha]` - the hub's 144 files at the snapshot's
  revision, sizes (and LFS sha256), 360.02 GB. The PLE int8 file is made from our export by the script's `ple` stage
  (not reused from Intel's snapshot: one source of truth, ~30 min).
- [x] **Step 2: AutoRound on the tiny fused model.** `a6cf7487` via `ar_pin.sh` (`AR_COMMIT`, `AR_PATCH=0`), the
  tiny `make_tiny_vl.py` model, the recipe of spec 21 v2 §2, low-CPU-memory mode (AutoRound: "Fused-MoE checkpoint
  detected: building a meta skeleton and materializing weights per block"), 2 iterations; the per-block resume
  (stopped after block 0, resumed at block 1, `OK`); the PLE table file-backed (AutoRound's own load: ~190 GiB).
- [x] **Step 3: the head and the MTP head.** `lm_head`: not AutoRound's (`quant_lm_head` turns low CPU memory and
  the resume off); the `head` stage packs and declares it (`head_rtn.py`, `loader/rtn.h`'s rule). The
  MTP head: AutoRound's own export (`missing_tensors.py`: the source's `mtp.*` copied, the fused experts split per
  expert, RTN int4 g64; attention and shared expert kept bf16 by the ignore list). A tuned pass over the MTP layer is
  not available (transformers builds no MTP module); acceptance with the RTN experts is measured in Task 7 Step 4.
- [x] **Step 4:** the script `tools/quantize/quantize_qwen4exp.sh` (stages quant / bf16 / head / ple / check, `run --only` /
  `--from`, `start` / `status` / `stop` / `selftest`); `check.py` and `expected_names` extended: `lm_head` int4,
  the MTP head's experts int4; `check_gptq_export.py --skip-prefix model.visual.`.
- **Gate (passed on the tiny model, card 1 and the box CPU):** `check.py` ACCEPTED with the int8 PLE file,
  `check_gptq_export.py --lm-head quant --source` VERIFIED, per-expert int4 tensors for every routed, shared and MTP
  expert; the download check is the operator's.

### Task 2: size the run (measured on a stand-in 2026-11-11; the operator confirms on the real model)

The operator ruled tuned (spec 21 v2 §0.1), so there is no RTN arm and no CPU arm to size. A real-width stand-in (4
layers, 512 experts, random weights) on card 1: one GDN block **57 min**, peak VRAM 38.9 GiB on the 31.9 GiB card
(spilled by the driver; the tuning state alone is ~34 GiB), two cards no help; projection **~48 h on one card**
(tools/quantize/README.md, "Sizing").

- [ ] **Step 1 (operator):** after `start`, `status` once the first block is done (~1 h): the seconds per block and
  AutoRound's time left; `stop` if a block is far above ~60 min or the RAM nears 115 GB (the run resumes).
- **Gate:** the first real block's time and the RAM recorded in the run's README; the run continues (Task 3).

### Task 3: the run and the export (box, detached)

- [ ] **Step 1 (operator):** `tools/quantize/quantize_qwen4exp.sh start` - detached, resumable per block, on card 1;
  output to the cold disk (`~/b70/hdd/quant/`); `status` shows the blocks done and AutoRound's time left.
- [ ] **Step 2 (the script's last stages):** the PLE int8 file (`ple_int8.py`, beside the export), `check.py` and
  `check_gptq_export.py --lm-head quant --source`, provenance appended to its README (version, commit, flags, source
  revision, date): `FLASH-NEXT QUANT: OK`. Then `tools/hf_tier.sh` makes it hot.
- [ ] **Step 3:** the qwen4exp loader accepts the checkpoint's int4 head (`LmHeadForm::Checkpoint` for this family:
  int4 g64 layout 0 and the int4 head GEMV Qwen3.8 already has; the int8-from-bf16 path stays for Intel's form);
  `r31.load` on our export at `--layers 18` and `--layers 37 --parallel pp`.
- **Gate:** both checks `ACCEPTED`; the engine loads our export with 0 unconsumed tensors; `r32.k3` / `r33.prefill`
  run on it.

### Task 4: the hit-rate curve on the recorded traces (box CPU, no GPU; may run during Task 3)

The 2026-11-09 traces (`$DATA/oracle-out-q4exp-traces/`, 39 sources, Intel's checkpoint; the routers are bf16 in both)
exist; `r36.curve` has never run.

- [ ] **Step 1: rebase the curve and the projection on our forms.** `tools/oracle/offload_curve.py`: the judged budget
  row is ours (int4 dense, int4 head: D = 3.46 GB, spec 21 v2 §3), Intel's bf16 row printed beside it;
  `STOP_TPS = 40.0` becomes spec 21 v2 §11's rule (`pp` < 30 and `tp` < 50 t/s -> STOP). Update
  `test_offload_curve.py`; PASS.
- [ ] **Step 2: add the `tp` layout to `project`** (spec 21 v2 §5 (b): D 2.38 and R 0.627 per card, both links at P,
  0.34 ms of all-reduces); print (a) and (b) side by side for each f / h. Tests with spec 21 v2 §5's table as the
  expected numbers (± 1 t/s).
- [ ] **Step 3: the swap policies in the simulation:** static, basalt's (decay 0.85, margin 1.5, up to 96 swaps a
  window) and saber's (≤ 2 inserts a layer a step, second chance within 8 steps, read-through above 50 %), each by
  held-out h and link bytes (spec 21 v2 §6).
- [ ] **Step 4 (box CPU): run** `tools/box_validate.sh --only r36.curve`.
- **Gate:** `oracle-out-q4exp-offload/{curve,swap,mtp,prefill,profile}.txt` exist; the record states h (held out) at
  f = 0.75 / 0.80 / 0.85, each swap policy's h and link bytes against static, the MTP verify union per layer at
  K = 1..3, and the prefill bytes per chunk; `profile.safetensors` written.

### Task 5: the link and memory probe (box, both cards, ~1-2 h; after Task 3 frees the card)

- [ ] **Step 1: make `probe_offload pin` safe.** Its one-range arm doubles a single pinned range from 1 GiB and reads it
  on every card: with `--dev both` a range above 4 GiB is exactly the 2026-11-11 hang. Cap any range mapped by both
  cards at 4 GiB (2 GiB pieces for the multi-range arm), run the one-range arm on one card only; the target is the
  whole-model host side: the PLE table (51.8 GB) + each card's own misses mirror (~13 GB in all), with the ~30 GiB per
  process limit XeStrata saw as the question.
- [ ] **Step 2: add a `peer` arm** to `probe_offload`: card 0's table kernel reading expert blocks that live in card 1's
  VRAM over P2P, and a copy-engine peer copy, at the expert sizes of P0.2 (spec 22 §3) - the evidence for spec 21 v2
  §5 (c).
- [ ] **Step 3: build** (`tools/box.sh build`), the probe's two kernels' first compile (`r36.build`).
- [ ] **Step 4 (box, both cards): run** `r36.links`, `r36.links_load` (stop and report below Gen3 x8), `r36.copy`,
  `r36.zc` (each card, then both cards at once - the `tp` case), `r36.replay`, `r36.hostbw`, `r36.pin`, `r36.alias`,
  the new peer arm, and `r32.ple_rate`.
- **Gate:** both links' generation and width recorded; host-to-device rate per card and both at once, copy and
  in-kernel read; whether a mixed launch costs max(VRAM part, host part) or their sum; 10^4 replays with every byte
  checked; no alias; pinned pieces up to the target with no hang, or the first failing total named; the P2P read rate;
  the PLE gather's share of a step.

### Task 6: the projection and the record

- [ ] **Step 1:** `tools/box_validate.sh --only r36.project` with Task 5's measured rates.
- [ ] **Step 2: the record** `docs/probes/probe-offload-<date>.md`: Task 4's curve and policies, Task 5's numbers, the
  projection of (a) `pp` + tier and (b) `tp` + tier (and (c) if the peer rate allows), decode at 4k / 32k / 128k,
  prefill at 4k, against spec 21 v2 §4's yardsticks (4-bit class: 20.8 one B70, 72-78 four; the goal 66 on two). Its
  last line: STOP or GO by spec 21 v2 §11.
- [ ] **Step 3:** open-work item 4 updated with the measured h and t/s; the operator decides the go.
- **Gate:** the record exists with every number marked measured / derived; the operator's go (or stop) is recorded.

### Task 7: the quality of our export (box CPU, beside Tasks 5-8)

- [ ] **Step 1: ours against the bf16 original** (spec 21 v2 §2; no Intel comparison, operator 2026-11-12): KL and
  top-1 per held-out set (golden prompts, tool-call prompts), layer-streamed from the original in f32
  (`/tmp/b70-cpu-model.lock`; above 64 GB also the box GPU lock).
- [ ] **Step 2: the whole-model f32 reference of our weights:** all 48 layers on `q4exp_4k` (+ 32 forced continuation
  steps), as the 18 / 37 layer ones ([plan](2026-11-08-flash-next-f32-reference.md)); ~90 GiB, holds
  `tools/gpu_lock.sh box`. Queued with the golden-set remake (`#HOLD-sj q4exp-golden-scores`, now on our export) so
  the CPU is not loaded twice; the bf16 CPU reference's own f32-rejected tokens and routes (the twin count).
- [ ] **Step 3:** a short routing trace on our export (3 sources) against Intel's: the profile's top experts per layer
  overlap; below 90 % overlap, re-record the 39 sources on ours and rerun Task 4.
- [ ] **Step 4:** `r34.fixture` / `r34.accept` / `r34.a4_ref` on our export: MTP acceptance with our int4 MTP experts
  against the bf16 head.
- **Gate:** the KL / top-1 table against the bf16 original recorded; the paired reference files published with their
  identity checks; the twin counts and the acceptance pair recorded (spec 21 v2 §10's bars decide, not a
  comparison).

### Task 8: the table, the mirror and the static fill on `--parallel pp` (box)

Plan 22b Tasks 1-5, with these changes: our export; the mirror is the misses only, per layer (≤ 1.34 GB a range, 2 GiB
pieces); the profile is `<snapshot>-expert-profile.safetensors` from Task 4, filled with no flag (id order and a
startup line without it); the budget per layer is the profile's curve if Task 4 shows ≥ 0.5 point of h over uniform,
else uniform; the MTP head resident; the planner stops refusing the whole model and refuses only when the non-routed
part or the host memory does not fit, naming the bytes.

- [ ] **Step 1:** the `_TBL` form of `q4_moe.cl` / `q4_pf_moe.cl` (`Q4_EXPERT_TABLE`), every existing variant
  preprocessed token for token as main's (22b Task 1).
- [ ] **Step 2:** the fill and the planner with a cache (host tests; 22b Task 2).
- [ ] **Step 3:** the loader - arenas, the pinned misses mirror in ≤ 2 GiB pieces, the table, the tag / checksum checks
  (22b Task 3).
- [ ] **Step 4:** the engine binds the table; `b70-decode` runs the whole model on `--parallel pp`; O0 and O1 (22b Task 4).
- [ ] **Step 5 (box):** the whole model end to end: load time, the memory line (VRAM and pinned host per card), decode
  and prefill at 4k, Task 7's count rule on 48 layers, decode's twin count.
- **Gate:** O0 bitwise (ids, logits, routes, selections, states; decode, prefill, verify); O1 bitwise at 0 / 50 / 100 %
  random and adversarial; O3 (tags, checksums, a hang-free refusal); the whole model's prefill passes the count rule and
  decode the twin count against Task 7; **decode ≥ 30 t/s** at 4k, one request, MTP off.

### Task 9: swaps (the policy Task 4 picked) and the two prefill modes

Plan 22c Tasks 1-3 with the winning policy of Task 4 (spec 21 v2 §6): counters in the route kernel (`_CNT`), copies in
1 MiB pieces between steps, admitted after the copy's event, placement pinned by an environment variable in tests.
Prefill: routed-misses-only staging below the crossover, a staging ring on the copy engine above it, filled while the
previous layer computes; the crossover measured, not copied from Strata.

- **Gate:** O2 (a swap every step bitwise the no-swap run; a withheld copy never read); decode with swaps ≥ the static
  fill's on the opencode recording, h within 1 point of Task 4's simulation; prefill at 4k ≥ the bus-cap projection's
  half (spec 21 v2 §6), recorded.

### Task 10: serving the whole model

Plan 22d Tasks 1-4 and 7 (zero-config): `--max-len auto` by spec 21 v2 §9, `--max-num-seqs` per its design, MTP with
the resident head (the per-step choice of open-work item 13 when it lands, else `--spec auto`), the PLE gather moved
before the step only if Task 5 showed it in the step time, the prefix cache restoring bitwise with the tier on, the
tool-call test against `r34.a4_ref`, passkey at the chosen context; the speed rows in docs/BENCHMARKS.md and the
Flash-Next section of docs/18-running-models.md.

- **Gate:** `b70-serve` serves the whole model with no `--layers`; prefix cache restores bitwise; tool-call test: every
  first difference passes the f32 judge; passkey passes; decode ≥ 30 t/s served (spec 21 v2 §11 (a)); MTP acceptance
  on prose and code recorded against Strata's / basalt's.

Then plan 21v2b (tensor parallel, the speed gates ≥ 50 / ≥ 1000 and the goal). The decode levers of spec 21 v2 §7
(launch count, union-aware verify, QSA fusion and int8 KV, int8 hyper-connections, the int8 DPAS grouped GEMM) are
21v2b's last task, each landing only with its measured gain and the f32 judge passing.

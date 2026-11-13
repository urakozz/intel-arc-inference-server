# MoE-model int8 prefill - W4A8 experts over bf16 mixer linears, the one-launch delta-net kernel and the GEMM router on `l0-int8` (Ornith)

**Status (2026-11-13): written, not started. Starts only after phase 3
(`2026-11-12-moe-model-prefill-phase3.md`) is merged** (operator 2026-11-12: int8 is not patched around a moving
base). Open-work item 16. Main at 700c4b44 when written.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The
> gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step**
> (signed, no trailers; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file
> at the worktree root (untracked) when you stop. Box jobs: every GPU job through `tools/gpu_lock.sh card <N> -- ...`,
> CPU reference jobs under `/tmp/b70-cpu-model.lock`, long jobs detached (`nohup setsid`, logs under
> `~/b70/logs/moe-int8/`), checked at most every 30 min with one blocking wait.

**Goal:** Ornith 1.5 35B-A3B, one card, pp4096: `l0-int8` **>= 9,600 t/s and >= 1.08x `l0` of the same build**, with
the serving check (39 golden + tool-call requests through `b70-serve` with MTP, f32-judged) at **0 FAIL**. The
operator's original 11,000 t/s (372 ms) is **out of reach of this plan by ~40-50 ms (DERIVED, section 3)**; Task 6
reports what is left. `l0` (bf16) stays the default; if the quality gates fail, the plan ends with a record and a
question, not a workaround.

Every time and error below is from an existing record unless marked **DERIVED**.

## 1. Diagnosis

**Where `l0-int8`'s speed comes from today** (4k kernel ms, `l0-int8` / `l0`, `docs/probes/probe-moe-prefill-2026-11-12.md`
"Per phase at 4k"):

| phase | `l0-int8` | `l0` | int8 minus bf16 |
|---|---:|---:|---:|
| expert GEMMs (W4A8 gate\|\|up, bf16 down) | 125.2 | 169.8 | **-44.6** |
| gather | 12.0 | 17.4 | -5.4 |
| dense (mixer) linears | 74.4 | 91.1 | **-16.7** |
| GDN | 128.8 | 94.5 | **+34.3** |
| everything else | 107.3 | 109.2 | -1.9 |
| sum (wall) | 448 (481) | 482 (507) | -34 (-26) |

Two thirds of the gain is the W4A8 expert GEMM (the int4 nibble is the DPAS operand, no conversion). The int8 mixer
linears give 17 ms (3.5 %). The delta-net loses 34 ms because `l0-int8` keeps the four intra-chunk launches
(`src/runtime/prefill/gdn.cc:155-157`: `intra_one_launch` is `d.is_moe() && b == PrefillBackend::L0`).

**Where its error comes from.** The mixer linears (`step.cc:108` qkv||z, `:128` out_proj, `:132` attention q||k||v,
`:~157` o_proj) run as rotated int8 x int8: one activation scale per token row over a 1024-block Hadamard rotation
(`src/kernels/prefill/pf_int8.cl:80-102`), one weight scale per output column (`:210-246`, `int8.cc:134-148`), the
rescale in the epilogue (`:392-406`). No arithmetic defect was found by reading; ~1.2 % relative error a linear is the
form's floor (measured 1.22 % against bf16's 0.33 %). On a 2,600-id prompt against f32
(`probe-prefill-phase2-2026-11-12.md` section 8, and section 10 on branch `int8-mixer-bf16`, be894d8b):

| distance to f32, x the bf16 reference's | `l0` | `l0-int8` today | all four mixer linears bf16, experts W4A8 |
|---|---:|---:|---:|
| logits | 0.48 | 0.59 | 0.66 |
| all-rows median | 1.01 | 1.39 | **1.06** |
| attention K/V worst row | 0.73 | 2.35 | 0.58 |
| delta-net state, layers 0 / 1 / 4 / 10 | 1.57 / 0.94 / 2.30 / 0.96 | 6.38 / 5.95 / 4.94 / 2.09 | 1.57 / 0.95 / 2.02 / 1.01 |

The drift is the int8 mixer linears as a group (out_proj as much as qkv||z); the W4A8 experts add little. The serving
FAILs (6 of 21 judged, 10-145 ulp, all long tool-call prompts) are consistent with that but **not yet shown to be
caused by it**: Task 1 measures it.

**Serving evidence, 2026-11-12 evening** (the judge re-run on the references built so far, `~/b70/logs/ornith-moe-prefill/
t5c/serve_judge3.log`; all three forms have int8 mixer linears and differ only in the experts):

| form of `l0-int8` | experts | requests judged | FAIL |
|---|---|---:|---:|
| the old rotated int8 gate\|\|up | rotated int8 | 20 | 6 |
| bf16 experts | bf16 | 20 | 5 |
| W4A8 (today's) | W4A8 | 21 | 6 |

With bf16 experts the check still fails, on mostly the same tool-call requests (worst misses 18-47 ulp): the experts
are not the cause, the int8 mixer linears are what the three forms share. It does not show that form A is clean -
no run had bf16 mixer linears. The other 18-19 requests of each form wait for their f32 references (W4A8's 35
contexts are building since 19:16).

**Why it worked on dense Qwen3.8 and not here.**
1. *The prize sits elsewhere.* On Qwen3.8 the int8 linears include the MLP's gate||up and down at K 17408, 1.37-1.52x
   each; the whole prefill went +28 % (spec 5 section 8). On Ornith the MLP is the experts, which are W4A8 anyway; the
   rotated int8 path covers only the mixer linears: 17 ms.
2. *It was accepted on short prompts.* Spec 5's gates used 38-61-id prompts (spec 5 section 7.1); the long gate was
   never run. Its own smoke showed last-row cosine 0.88-0.95 on a scattered few long prefixes (section 7.3), and the
   later records are known fails (`l0-int8` + int8 KV ~3 % of exact-copy tool calls; three `mtp_gpu_kv8_test` rows
   traced to the int8 prompt pass). The dense model was never clean on long prompts; it was not looked at there.
3. *Ornith amplifies.* 40 routed layers with top-8 near-ties (the margin table: every miss on rows with 3-7 ulp f32
   margin), 30 delta-net layers whose state reads every token, and the judge is f32 with agentic-length prompts.

**The evening experiment's lead, verified.** In the `dense` arm of be894d8b the delta-net still ran the four launches:
`mixer_body` passed the unchanged `L0Int8` backend to `gdn_chunk` and the commit did not touch `gdn.cc`. So its
"~`l0` speed" (8,089 vs 7,974) carried ~61 ms that `l0` does not pay. Whether the one-launch kernel passes there is
not known: it failed on `l0-int8` on one near-tie row by cosine (0.999025 vs the twin's 0.999467) while the int8 mixer
linears were still in; the kernel itself matches fp64 as well as the four launches do (phase 2 record, section 2).

## 2. Candidate forms

`l0` after phase 3, **DERIVED**: 482 - 19.3 (router) - 5.6 (a||b) - 4.1 (combine + norm) - 8.6 (attention 0.78x) =
~444 ms kernel, ~470 ms wall, **~8,700 t/s**. `l0-int8` as it is after phase 3 (it keeps the GEMV router): ~436 ms
kernel, ~469 wall - **the same as `l0`**: today's int8 lead disappears when phase 3 lands.

| form | 4k wall, DERIVED | error vs f32 (records) | build cost | reuses |
|---|---|---|---|---|
| **A. bf16 mixer linears + W4A8 experts + the one-launch delta-net kernel + the GEMM router and a\|\|b** | **408-425 ms, 9,600-10,000 t/s** (two derivations below) | split test: median 1.06x, logits 0.66x, states within `l0`'s spread (measured, be894d8b, with four launches and the GEMV router); short gates and serving not measured | small: bindings only, no new kernel | `linear_l0`, `pf_gdn_intra`, phase 3's `pf_route_slab` + `pf_gemm`, `pf_moe_w4a8` |
| A without the one-launch kernel | ~470-485 ms, 8,450-8,700 | as A | - | - |
| B. W4A8 for the mixer linears too (int4 exact, per-row-and-64-group int8 activations) | ~`l0`'s dense time: the per-group rescale costs 3.0-3.4 of 5.44 ms, 1.017x bf16 on Qwen's gate\|\|up (`probe-w4a8-2026-09-23.md` sections 6, 13) | ~1.0 % a linear DERIVED from the experts' h (1.006 %, against rotated int8's 1.22 % and bf16's 0.33 %): the activation's 8 bits remain | a new dense kernel (`pf_moe_w4a8` is tied to the item table and the SiLU epilogue) | - |
| B'. other transforms on the rotated path (SmoothQuant, Hadamard 128, random orthogonal) | none faster | gate\|\|up 0.51-0.81 % at best against 0.56 % for today's rotation (section 14.2); none reaches bf16 | - | - |
| C. int8 per linear or per layer where it does not accumulate | at most 17 ms for all four; the attention pair alone ~4 ms (8,125 vs 8,089, inside the noise) | with only the attention pair int8: median 1.13x, states 1.75x / 2.66x at layers 3 / 5 | switch table | - |
| D. int4 weights x 16-bit activations as two int8 planes | ~bf16's: two int8 DPAS passes are one bf16 pass in operations, and the W4A8 walk is not DPAS-bound (1.76 ms against 1.72 for bf16 without conversion) | better than bf16 (DERIVED) | new kernels | - |

**Recommended: A.** It keeps the large clean gain (experts), drops the small noisy one (mixer linears, 17 ms), and
takes back what `l0-int8` gave away (delta-net 61 ms, router and a||b 25 ms). B, B', C and D are not built: B and D
are no faster than bf16 on these linears, C's whole prize is 17 ms and its error is measured.

A's two derivations: (1) `l0` after phase 3, 444 ms kernel, - 44.6 (experts) - 5.4 (gather) + ~3 (the per-group
quantiser, 40 launches) = ~397 kernel, + 25-33 wall = 422-430 ms, 9,500-9,700 t/s. (2) the measured `dense` arm,
506 ms (busy box), - 61 (one-launch kernel) - 25 (router, a||b) - 4 - 9 = 408 ms, 10,000 t/s.

## 3. The target, honestly

A is expected at 9,600-10,000 t/s, +10-15 % over `l0`. 11,000 needs 372 ms: 40-50 ms more. What could give it, none
of it int8-only except the first: the W4A8 walk's per-group rescale (the probe's per-row arm is 0.73 ms a layer-chunk
faster at C 4096, ~29 ms at 4k - Task 5, probe first), and the phases `l0` shares (delta-net scan 28, front 21; mixer
linears 86 at 75 % of peak). Flash-Next is not bound here; its turn comes with its own weights (open-work item 4).

## Dependencies and constraints

- Phase 3 merged (its `pf_route_slab` router and a||b, `pf_moe_combine_norm`, the attention arm). Branch
  `moe-int8-prefill`, own worktree. One implementing agent; probes may run on Sonnet.
- Box data: the Ornith int4 checkpoint, `oracle_ornith_f32`, the long32k reference, the serving references
  `oracle-out-ornith-f32-i8head-forced` (the 18 that were still building on 2026-11-12 must be complete: Task 1 Step 1).
- **No user-facing switch.** `--prefill-backend l0-int8` on a MoE model *is* form A after Task 3. Task 1 carries
  be894d8b's `B70_PF_I8_MIXER_IN` and two more test switches on the probe branch only; none is merged.
- **Who moves.** Ornith on `l0-int8` only. `l0` logs stay bitwise. Qwen3.8 (`int8.cc`, the rotated path) is not
  touched: every Qwen binary and gate log identical (G0). Flash-Next, K2, Kolibri, Bolt untouched.
- **Numerics kind: judged.** The f32 judge; the `l0-int8` MoE near-miss rule as it is
  (`tests/prefill/prefill_gate_test.cc:187`, `:357-363`); no new relaxation without the operator.
- **Timing:** interleaved pairs after a warm-up, median of 3, one card, each number labelled quiet or busy box.

## Stop rules

- Task 1's serving arm of form A with any FAIL that `l0`'s own control does not have: stop, operator decision D2.
- The one-launch kernel failing a gate on form A while the four launches pass: stop, D3 (form A is then no faster
  than `l0`).
- Task 1's speed below 1.05x `l0`: stop and record; item 16 goes to the operator as "no speed case".

---

### Task 1: the probes - is form A clean, and is it fast (Mac, then box)

**Files (probe branch `moe-int8-probes`, not merged):** `src/runtime/prefill/step.cc` (be894d8b's `linear_backend`
re-applied on the phase 3 base), `src/runtime/prefill/gdn.cc` (`B70_PF_I8_GDN=intra`: the one-launch kernel on
`l0-int8`), `src/runtime/prefill/moe.cc` and `step.cc` (`B70_PF_I8_ROUTE=gemm`: phase 3's router and a||b on
`l0-int8`), `tests/prefill/prefill_split_test.cc` (be894d8b's per-layer print), `tools/box_validate/stages.sh` (row 86
or the next free: `r86.split`, `r86.gates`, `r86.serve`, `r86.speed`, opt-in; `r86.serve` is the 2026-11-12 scratch
run made a stage: the 39 golden + tool-call requests, `b70-serve` with MTP, `tools/box_validate/serve_judge.py --ref
l0`).

Arms: `l0`; `l0-int8` today; **A0** = mixer linears bf16; **A1** = A0 + one-launch kernel; **A2** = A1 + GEMM router
and a||b (= form A).

- [ ] **Step 1 (box, no build):** the 18 unjudged requests of the 2026-11-12 run: check the references are complete
  and re-run the judge on the kept outputs. Record the full 39 for today's `l0-int8`.
- [ ] **Step 2 (Mac):** the switches, the stages; host tests green; `--dry-run --only r86` prints the commands. Commit.
- [ ] **Step 3 (box):** record `docs/probes/probe-moe-int8-2026-11-<day>.md`, **the gates written into it before any
  run**:
  - **Split (2,600 ids vs f32), A0 / A1 / A2:** logits and all-rows median <= 2x the bf16 reference's (the test's own
    rule, `kCosSlack`); median <= 1.15x; no printed delta-net state above 1.3x `l0`'s own multiple for that layer.
  - **Short gates, A0 / A1 / A2:** `prefill_gate_ornith_l0-int8_test` and `_c16` pass with no more rule-passed rows
    than today's (one each); every such row printed with its f32 margin and both cosines.
  - **Serving, A2 (and A0 if A2 fails):** 0 FAIL of 39. Control, at no extra cost: at every first difference the judge
    already has the f32 reference of `l0`'s own continuation. For each request, report whether `l0`'s continuation
    from that point passes the same rule. A request where `l0`'s side is rejected too is reported as "both sides
    rejected" and counted as the floor, not as form A's FAIL.
  - **Speed, A2 vs `l0`, pp4096 interleaved:** >= 1.08x; with `B70_PREFILL_PROFILE=1` the phase table.
  - Reported, not gated: which of today's six failing requests pass on A0 (the open cause question of item 16), the
    tool-call test's identical / invalid counts against bf16 (today 20 / 36, 1 invalid).
- [ ] **Step 4:** the verdict line per arm; on a stop rule, the operator's digest (D2 / D3) and nothing further.

### Task 2: bf16 mixer linears on `l0-int8` for MoE models (Mac, then box) - JUDGED

**Files:** `src/runtime/prefill/step.cc` (`pf_linear`: a MoE model's QkvZ, OutProj, Qkv, OProj take `linear_l0` on
`L0Int8`; the quantiser and launch counts follow, as be894d8b's `quantisers()`); `src/runtime/engine_prefill.cc:121-123`
(no rotated column-scale pass for a MoE model: -74 ms at load, its scale buffers not allocated);
`src/runtime/buffer_sizes.cc` (the rotated scratch not planned for MoE models); `tests/` launch-count and memory-plan
tests.

- [ ] **Step 1 (Mac):** counts and plan tests first, then the binding. Dead code this leaves for MoE models (the
  rotated int8 gather of the old expert path, if any is still reachable) is listed in the commit message, not removed.
- [ ] **Step 2 (box):** `ornith_prefill_l0-int8_test`, the two gates, the split test, `prefill_smoke`, memory-plan.
  - **Gates:** logs equal Task 1's A0 arm byte for byte (tokens, cosines); `l0` and every Qwen3.8 log bitwise main's.

### Task 3: the one-launch delta-net kernel and the GEMM router on `l0-int8` (Mac, then box) - JUDGED

**Files:** `src/runtime/prefill/gdn.cc:155-157` (`intra_one_launch`: `d.is_moe()`; the comment at `:235-237`
rewritten with Task 1's evidence); `src/runtime/prefill/moe.cc`, `step.cc` (phase 3's router and a||b binding without
its `l0`-only condition); `kGdnChunkLaunches`, `moe_chunk_launches`; `tests/prefill/prefill_split_test.cc` (`:222`:
`l0-int8` on a MoE model is no longer diagnostic - it binds at the 2x rule; the ctest `diagnostic` label dropped at
`tests/CMakeLists.txt:2004`).

- [ ] **Step 1 (Mac):** the bindings, the counts, the test's rule. Commit.
- [ ] **Step 2 (box):** the Ornith set (`tools/probe/pf_phase2_gates.sh`), the split test now binding, replay bitwise.
  - **Gates:** logs equal Task 1's A2 arm; the profile's GDN row within 3 ms of `l0`'s, route + a||b within 2 ms.

### Task 4: the serving check as the quality gate, and the tool-call test (box)

- [ ] **Step 1 (box, card 0; CPU references under the CPU lock if any context is new):** `r86.serve` on the branch
  head; the tool-call test (36 calls against bf16); `r16.mtp_rows` on `l0-int8`.
  - **Gate (final quality):** 0 FAIL of 39 by `serve_judge.py`'s rules as they are (a speculative miss beyond 8 bf16
    ulps of the f32 top fails unless the bf16 reference picks it); tool calls: no more invalid calls than `l0`, and
    identical-to-bf16 count >= `l0`'s - 1.
- [ ] **Step 2:** the result into the record and open-work item 16. A FAIL here stops the plan (D2).

### Task 5: the W4A8 walk's per-group rescale (Mac, then box) - only after Task 4 passes, probe first

The probe's per-row-and-group arm is 3.44 ms a layer-chunk at C 4096 against 2.71 for the per-row arm; production is
125 ms at 4k against the probe's ~108 + quantiser.

**Files:** `tools/probe/probe_moe_prefill.{cc,cl}` (arms on `F3.a4b2m4n8.c0x1o1`: the row's group scale folded into one
fp32 vector per (row tile, group) outside the column loop; a 128-row item at C 4096), then `pf_moe_w4a8.cl` only if
the probe passes.

- [ ] **Step 1 (box, probe):** **gate, written first:** oracle-exact against the integer host reference and
  <= 0.88x the bound arm at C 4096 and C 2048. A miss closes the task.
- [ ] **Step 2 (Mac, box):** the kernel form; `pf_moe_test` exact against the host's integer sums; expert GEMMs
  <= today's - 12 ms at 4k; the gates of Tasks 3 and 4 again (bitwise logs if the result words are unchanged).

### Task 6: the record and the clean-up (box, one idle card; then Mac)

- [ ] **Step 1 (box):** interleaved pairs, main against the branch head, median of 3: Ornith pp4096 and pp16384 on
  `l0` and `l0-int8`; the phase table.
  - **Gate (final speed):** pp4096 `l0-int8` **>= 9,600 t/s and >= 1.08x `l0`**. The gap to 11,000 by phase.
- [ ] **Step 2 (box):** `tools/box_validate.sh --quick`; G0 against `--baseline main`: every Qwen3.8, K2, Kolibri,
  Bolt and Flash-Next binary and log identical, Ornith `l0` bitwise, decode unchanged.
- [ ] **Step 3 (Mac):** BENCHMARKS' Ornith rows, `docs/17-int8-prefill.md` (what `l0-int8` means on a MoE model),
  open-work items 14 and 16; branch `int8-mixer-bf16` named to the operator as superseded. Commit; offer the ff-merge.

| task | Mac | box | numerics |
|---|---|---|---|
| 1 probes | step 2 | steps 1, 3 | judged, gates pre-registered |
| 2 bf16 mixer linears | step 1 | step 2 | judged, equal to the probe arm |
| 3 one-launch kernel, GEMM router | step 1 | step 2 | judged, equal to the probe arm |
| 4 serving check | - | step 1 | the f32 serving judge |
| 5 rescale (conditional) | step 2 | steps 1-2 | integer-exact |
| 6 record | step 3 | steps 1-2 | - |

## Operator decisions (each with a recommendation)

- **D1 - what `l0-int8` means on a MoE model.** Form A: int8 only in the experts' gate||up, everything else bf16.
  The name stays, no new flag. *Recommendation: yes;* the rotated int8 mixer path stays for Qwen3.8 only.
- **D2 - if form A still FAILs the serving check.** Options: (a) W4A8 on the layers whose error shows least (a CPU
  per-layer study with `moe_w4a8_accuracy.py` first); (b) close item 16: `l0-int8` on MoE models refused at start-up
  with a line naming `l0`, the W4A8 kernel kept for Flash-Next's study or deleted. *Recommendation: (b) unless the
  FAILs are one or two near-tie rows; then bring the rows in the margin-table form.*
- **D3 - if the one-launch kernel fails on form A.** *Recommendation: stop;* without it form A is no faster than
  `l0`, and a new relaxation of the near-miss rule for one kernel is not worth it.
- **D4 - the default.** If Tasks 4 and 6 pass, should MoE models default to `l0-int8`? *Recommendation: not yet -
  one on-demand full validation and a week of the operator's own sessions first; int8 KV with it stays a known fail
  until checked.*
- **D5 - the 11,000 target.** *Recommendation: replace it by ">= 1.10x `l0`"*; the rest of the way is the phases
  both backends share, so it belongs to item 14, not to int8.
- **D6 - Qwen3.8's `l0-int8`.** The same long-prompt noise is on record there (section 1, point 2).
  *Recommendation: out of this plan;* one split-test run against f32 on Qwen3.8 as a later, separate check.

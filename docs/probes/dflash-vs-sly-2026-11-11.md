# Probe: our DFlash against vllm-sly-radiance and saber-engine, 2026-11-11 (code and docs only)

**Question.** `--spec dflash` (Qwen3.8-27B, `z-lab/Qwen3.8-27B-DFlash2`) lost to `--mtp auto` in spec 19's D5 (0.878x on
llama-benchy tg128). A vLLM fork on one AMD R9700 (`~/PycharmProjects/vllm-sly-radiance`, "sly") reports 131 t/s at
one request with the same drafter and "11.17 tokens per step". Where does the gap come from?

**Method.** Reading only: no box run, no engine change. Theirs: sly at its 2026-11-10 night gate baseline, saber-engine
(`~/PycharmProjects/saber-engine`, the same drafter on R9700s). Ours: main at c9c311a9. Grades:
- **[R]**: read in a file;
- **[D]**: derived;
- **measured**: from a measurement recorded in our docs, with its date.

Paths in sly / saber are relative to their repo roots.

## 1. The headline number was misread

- **sly's "mean tokens per step 11.17" is not tokens per verify.** It is
  `Δvllm:iteration_tokens_total_sum / Δ..._count` over the whole BetterBench run (`ci/accept/accept.py:183-197`).
  - vLLM counts the computed **prompt tokens plus generated tokens per engine step, summed over the batch**.
  - The run covers concurrency 1-8, so it is a measure of how full the batch was. The gate's own comment says so:
    "tokens/step measures how full the batch was per step" (`ci/accept/accept.py:636-643`).
- **Their per-request yield is ~5 tokens per verify [D]:** 1 + 7 x 0.569 (acceptance = accepted / proposed,
  `accept.py:195`). An older counter run of the same setup recorded **4.20 tokens / step at acceptance 0.457**, per
  position 0.81 / 0.64 / 0.50 / 0.40 / 0.33 / 0.28 / 0.24 (`docs/BENCHMARKS.md:452-454`).
- **Their block is 8, K = 7, the same as ours.**
  - Launch: `num_speculative_tokens 7` (`README.md:199`); "the drafter's block_size is 8; 7 is the shipped default"
    (`serve-mxfp4.sh:471`).
  - K = 15 is opt-in only (`serve-mxfp4.sh:490-500`).
  - saber: `DRAFT_BLOCK` 8 (`README.md:328`), block a power of two, production 7
    (`crates/saber-engine/src/draft_tier.rs:59-68`).
  - The checkpoint's `dflash_config.block_size` is 8 (its config.json); sly measured 4.205 accepted at the trained 8
    against 3.961 at 9 (`dflash2/qwen3_dflash2.py:129-142`).
- **The gate's workload is not ours.**
  - BetterBench, **sampled** (T 0.7, top-p 0.95, top-k 20; `ci/accept/configs/full.json`).
  - 60-120-token prompts (`accept.py:20`), up to 600 output tokens.
  - Code .30 / reasoning .20 / prose .15 / json .15 / file_edit .10 / summarization .10.
  - The 131 / 231 / 350 / 407 t/s are BetterBench aggregates at concurrency 1 / 2 / 4 / 8, at 210 W
    (`ci/accept/baselines/vllm7-mxfp4.json`). Concurrency 1 spreads ~20 % run to run (`accept.py:577-581`).

So the comparison is **~5 tokens per verify (theirs) against ~4 (ours, §3)** at the same block. Tokens per step are close.
**The difference is the cost of a step.**

## 2. Side by side (one request)

| | sly (R9700, MXFP4 target) | saber (R9700) | ours (B70, int4 g64 target) |
|---|---|---|---|
| block / K / verify rows | 8 / 7 / 8 [R] | 8 / 7 / 8 chain (10-row tree at TP 2 only) [R] `main.rs:2186-2193` | 8 / 7 / 8 |
| drafter weights | W4A16 int4 (`syvai/...-W4A16`, `README.md:198`); context K / V from the int4 rows (`docs/TECHNICAL.md:551-553`) | fp8 e4m3, 128x128 block scales (`rules/dflash_gfx1201.toml:17-34`) | **int8 per-row RTN at load**, 2.05 GB (`loader/dflash_layout.h`) |
| draft head | the target's head **int4**, 656 MB, 1305 us at M = 8 (`sly/README.md:26`) | **int2** copy of the target's head, its top-80 re-ranked exactly by the real head (`toml:54-80`, `head_share.rs:19-29`) | the target's **int8** head, 1.27 GB, full vocabulary (`dflash_capture.cc:151-163`) |
| drafter KV | fp8 (`docs/TECHNICAL.md:227`) | fp8 (`dflash_graph.rs:386,498`) | bf16 ring, 2048 slots |
| taps | 5 layers -> concat 25,600 -> fc -> hidden_norm -> per-layer k / v (`radiance_dflash_capture.py:4,50`) | layers 5 / 19 / 33 / 47 / 61, copied inside the step at the tap point (`aux_tap.rs:5-31`) | the same 5 layers. vLLM collects tap i after layer i runs (aux index i + 1: upstream `eagle3_utils.py:88`, `qwen3_next.py:746-757`); ours copies layer i's output (spec 19 §11) - the same point |
| context projection | for every verify row, rejected rows tracked (`radiance_dflash_capture.py:61`) | the verify rows, rejected ones to a pad slot (`draft_inputs.rs:28-47`) | the j + 1 committed rows, after the host's accept (`serve_adapters.h:259-289`) |
| draft pass | one non-causal pass, **one FULL HIP graph** (`serve-mxfp4.sh:447-466`) | one captured graph of 90 launches + 12 eager, **0 host round trips** (`draft_plan.rs:18-29,1715-1743`); the proposal runs inside the step that made the taps (`draft.rs:200-215`) | one regular list of **87 launches** (`dflash_capture.cc:66-195`), then verify, then projection: **3 submits, 3 fences an iteration** (spec 19 §11, "Not in 19c") |
| draft pass cost | not recorded; ~3-4 ms [D] from its pieces (head 1.3 ms, drafter GEMMs, attention 5 x 38 us) | **4.40 ms** (saber) / **5.51 ms** (prod), TP 1, staged replay (`draft_tier.rs:34-36`) | **8.90 ms** (measured 2026-11-04, spec 28 §11, gemm_sm_i8w; was 20.6 ms host-timed / 16.3 ms on gemv_i8w) |
| plain step | ~30 ms (their fit's intercept) | 18.7 ms at TP 2 | 29.2-29.4 ms |
| verify at 8 rows | **T(M) = 29.9 + 0.393 M ms for the whole spec step, draft included**: 33 ms at M = 8 (`docs/TECHNICAL.md:648-651`); greedy ~35.0 ms / step (baseline `_comment`) | 8 rows ~ 1 row at batch 1 (head M 1 vs 8: 1088 vs 1131 us, `toml:328-329`) | **1.57 plain steps = 46 ms** (measured 2026-11-01, spec 28b, the M = 8 attention on v2: 11.9 ms of it). Since 28c, M = 6..8 bind DPAS attention at any depth (`runtime/verify_attn.h:48-50`): **~1.1 [D], not re-measured** |
| acceptance | greedy prefix / lossless sampling + Cactus (lossy, on by default, `README.md:451-473`) | greedy prefix / point-mass sampling | greedy prefix / lossless sparse q (spec 19 §11) |
| adaptive K | per request, widths {1, 3, 5, 7}, `E / T(M)` with T from the fit (`sly/radiance_adaptive_width.py:30-31,75-77,225-269`) | static 8; their EMA width lost when measured (`width.rs:16-23`) | AdaptiveK over `MtpCost::dflash`: verify 1.56 at M = 8 (pre-28c), draft **0.17 derived** (`adaptive_k.cc:37-45`) |
| tokens per verify | ~5.0 [D] sampled BetterBench; 4.20 measured earlier | 5.34 priced greedy on recorded lattices (`selector_rank.rs:5-9`); prod 3.25 live | 4.10 at K 7 (dflash_gpu_test, golden + A4, measured 2026-10-27; **counts 256 ids past EOS**, `tests/server/dflash_gpu_test.cc:44,98-108`) - P0's teacher-forced E_7: A4 7.21, code 6.15, prose 2.20 |

## 3. Where our iteration's time goes (K = 7, depth ~4k)

| piece | 2026-10-27 (D5's state) | today's main (spec 28b / 28c / 28d merged) | sly |
|---|---:|---:|---:|
| draft list | 20.6 ms (0.70) | 8.90 ms (0.30) measured | ~3-4 ms [D] |
| verify(8) | 92.6 ms (3.17, gemv) | 46 ms (1.57) measured before 28c; ~32 ms (1.1) [D] with 28c's DPAS attention | ~30 ms [D] |
| projection | 2.2 ms | 0.40 ms measured | inside the step |
| host round trips | 3 fences | 3 fences (~0.3-1 ms [D]) | 0 (graphs) |
| **iteration** | **~115 ms = 3.9 plain steps** | **~42-56 ms = 1.4-1.9** | **~33-35 ms = ~1.15** |

**The draft list, 8.90 ms, by bytes [D].**
- It reads 1.665 GB of drafter linears (per layer: q||k||v 31.5 M, o 21.0 M, gate||up 178.3 M, down 89.1 M, the two
  conv kernel projections 13.1 M - int8) plus the full int8 head, 1.27 GB (248,320 x 5120, at 7 rows): **2.94 GB**.
- At the target's own ~560 GB/s that is 5.3 ms, so the list runs at ~60 % of the bandwidth the target reaches.
- The rest is 87 launches with barriers, and small kernels whose grids are tiny:
  - `df_norm` (grid 1 x 8) twice a layer;
  - `df_select`, one work-group walking 7 rows;
  - two top-k stages over 7 x 248,320 logits.
- The draft vocabulary arm (32k ranked) measured 12.7 against 15.5 ms on gemv (r54.d1), so **the head is ~2.8 ms of a
  draft**.

**Bytes sly reads for the same pass [D]:** ~0.9 GB of int4 drafter linears + 0.66 GB of int4 head = **1.6 GB**, 54 % of ours.
saber reads less again (fp8 linears ~1.7 GB, but an int2 head of ~0.32 GB plus 80 exact rows).

## 4. The gap decomposition

Speed = tokens per iteration / iteration time.

**D5 (2026-10-27), llama-benchy tg128 (prose-like text), 0.878x of `--mtp auto`.**
- Tokens per iteration were higher for DFlash than for MTP: P0 prose E_2 2.13 against MTP K = 1's 1.44.
- The iteration cost more:
  - DFlash at K = 2: verify(3) 1.51 + draft 0.70 + projection 0.08 = 2.29 plain steps, so 2.13 / 2.29 = 0.93x plain.
  - MTP K = 1: 1.16 + 0.13 = 1.29, so 1.44 / 1.29 = 1.12x.
  - Ratio **0.83 [D]**, against the 0.82-0.88 measured.
- **Cost per step explains the whole loss.** The draft list alone (0.70 against MTP's 0.13) is more than the gap.
- Tokens per step were on DFlash's side.

**Against sly (one request).**
- Tokens per verify: ~5.0 against our ~4.1, a factor **~1.2**, and that comparison is not like for like:
  - their number is sampled BetterBench;
  - ours is greedy golden + A4 counted past EOS.
- Iteration time: ~34 ms against our ~115 ms in D5's state, a factor **~3.4**. Today it is ~42-56 ms [D], a factor
  **1.2-1.6**.
- **Cost per step dominated, by ~3x.** Most of that was closed by specs 28b / 28c / 28d after D5 ran, and D5 has not
  been re-measured since.

**What is left today [D], in plain-step units at K = 7:**

| term | ours | sly | gap |
|---|---:|---:|---:|
| verify(8) | 1.1 (with 28c, unmeasured) - 1.57 (measured without it) | ~1.03 | 0.07-0.54 |
| draft | 0.30 | ~0.12 | 0.18 |
| host round trips and projection | ~0.03-0.05 | ~0 | ~0.04 |

## 5. Is int8 hurting acceptance?

No.
- P0 (CPU reference, teacher-forced, 2487 anchors): int8 7.10 / bf16 7.09 / the published W4A16 7.10 at K = 7
  (`docs/probes/probe-dflash-2026-10-08.md` §2).
- D1 on the card: int8 sets 585 / walks 592 of 592, against bf16 587 / 592 (BENCHMARKS "DFlash (spec 19)").
- Both references run int4 drafters (sly) or an int2 head with exact re-rank (saber: Flash-Next's MTP head
  2.965 -> 2.956 tokens / step, `crates/saber-cli/src/flashnext/run.rs:5167-5171`).

**Precision is a cost lever, not an acceptance problem.**

**One open question on acceptance.**
- Our in-engine number (4.10 at K 7) sits far below P0's teacher-forced A4 7.21 / code 6.15.
- `dflash_gpu_test` runs 256 ids without stopping at EOS, so A4's tool calls (~77 ids each in P0) are followed by
  ~180 ids of whatever comes after the turn.
- D1 checked the drafter on **CPU-dumped** taps (19a's), never on the engine's own taps in the live loop.
- saber warns that a wrong drafter path "collapses acceptance silently while the output ids stay correct"
  (`draft_tier.rs:42-47`).
- This needs one measurement (v2 plan Task 2), not an assumption.

## 6. Conclusion in plain words

- **We did not lose because of the drafter or its block.** Our block is the same as theirs (8 rows, 7 drafts), and
  int8 costs no acceptance.
- **We lost because each draft-and-check round cost almost four plain steps when D5 measured it:**
  - checking 8 rows was 3.2x a plain step on the old per-row kernel;
  - drafting was 0.7x.
- **sly's round costs ~1.15 plain steps.** It checks 8 rows almost for the price of one, drafts in about a tenth of a
  step from int4 weights, and runs the whole thing as one GPU graph.
- **Since D5, specs 28b / 28c / 28d cut our check and our draft by ~2-3x.** No one has measured DFlash against MTP
  since. The first job is to measure it again on the opencode recording.
- **What is left:**
  - the draft pass at 0.30 (int8 weights, the full int8 head, 87 launches);
  - three host round trips a round;
  - AdaptiveK's cost table, which still believes the 0.17 draft and the pre-28c 1.56 verify;
  - the unexplained distance between our in-engine acceptance and the drafter's teacher-forced acceptance.
- **"11 tokens a step" was batch occupancy, not draft quality.** Their real yield is ~5 tokens a round, close to ours.

Design and plan: `docs/superpowers/specs/2026-11-11-spec19v2-dflash-v2-design.md`,
`docs/superpowers/plans/2026-11-11-spec19v2-dflash-speed.md`.

## 7. Spec 19v2 measurements, 2026-11-12

Task 5's RTN int4 g64 layout-1 arm at `39d54ad1`, with the original full int8 head and original engine-tap
anchors, **fails the acceptance gate**. This measures our at-load RTN form; the published quantized checkpoint's
P0 result above does not establish its quality. All 35 corpus/K rows recalibrate the original int8 baseline on
the new dumps before comparison (zero baseline drift).

| corpus, K=7 | original int8 E_K | RTN int4 E_K | change |
|---|---:|---:|---:|
| A4 | 7.08025 | 7.00551 | −0.07474 |
| code | 5.87500 | 5.93750 | +0.06250 |
| prose | 2.28646 | 2.22727 | −0.05919 |
| CJK | 1.50000 | 1.62500 | +0.12500 |
| held-out opencode | 3.68750 | 3.25000 | −0.43750 |

A4, prose and held-out opencode exceed the allowed 0.05 loss at K=7; code also loses 0.125 at K=2.
Per C2, request an AutoRound-exported drafter before promoting int4. The serving default remains int8.
Evidence: `~/b70/logs/dflash-v2/task5-real-39d54ad1.log` and
`~/b70/logs/dflash-v2/task5-real-39d54ad1/acceptance.json`. The synthetic f32 body-set gate also remains unresolved
at this source. Int4 S1 draft cost is 8.94586 ms at depth 4096 versus 8.97802 ms for the original int8 arm: memory
savings alone did not deliver the Task 5 speed gate. Split-K producer/consumer experiments continue separately;
their quality and performance are not established by this result.

The `8a16f5a4` grid split-K diagnostic (same int4 weights/head, fused consumer folds, unchanged 87/8 launches)
reduces the 4k draft to **7.84970 ms**, saving 1.12832 ms against the original int8 baseline; the 2.5 ms Task 5
gate is still unmet. At 32k it is 8.29017 ms. Projection is 0.33730/0.45911 ms, and K7 iteration is
1.52810/1.52793 plain steps (4k/32k). These are idle-card/busy-box diagnostics.
At 4k, sums of per-launch medians are 1.91240 ms for the int4 body, 2.16833 ms for the full int8 head,
2.76844 ms for attention-part, and 0.23677 ms for both top-k stages. Attention is the next measured non-GEMM target.
Evidence: `~/b70/logs/dflash-v2/task5-split-cost/iter-{4096,32768}.log`.

Task 6's 32k ranking also **fails out of sample**, measured on the unchanged int8 body to isolate the head.
Training excludes the whole `opencode-session-2`; ranked-id SHA256 is
`11a38d7470b3a2f47d75d703dcfd10e49af21aa44f6a8f7e650ccb333d18cb0e`.
The cached-tap path first reproduced all 4256 original int8/full-head engine drafts exactly. K7 A4 then falls
7.08025 → 6.29054, code 5.875 → 5.750, CJK 1.500 → 1.250; held-out opencode rises 3.68750 → 3.87500.
The ranking stays disabled. C3's next arm is the int4 candidate head with top-64 exact int8-row reranking.
Evidence: `~/b70/logs/dflash-v2/task6/{cached-taps-calibration,ranked32-int8}/` and
`task6-cached-calibration.log`, `task6-ranked-head.log`.

Task 7's fewer-heads geometry is bitwise the real drafter at short and full-window contexts. At pos4096/K7,
H4/H1/H2 attention-part sums are 2.700417/2.525833/2.872083 ms (interleaved timestamp medians, five launches,
busy box/card1). H1 saves only 0.174584 ms here, despite a larger short-context saving. Splitting rows as well
is the next experiment; the runtime geometry stays unchanged until its gates pass.

The later split-K numerical regression passes after the CPU twin mirrors GEMV's verified fused multiply-add
order (zero projection differences for both paths, aggregate f32 sets 5 versus 6 misses). RTN corpus loss remains;
this arithmetic-reference correction does not qualify RTN for promotion. Evidence: `task5-contracted-twin.log`.

C3's full-vocabulary int4 candidate head plus exact top-64 int8 rerank **passes isolated quality**: all 35
corpus/K acceptance values equal the original int8/full-head baseline (gap zero). One-load `--all-k` first
reproduced all 4256 original engine drafts. At depth4096 the head path costs 1.789271 ms (candidate GEMM
1.214896, top-64 0.365000/0.073750, rerank 0.135625), so the plan's 0.5 ms head-share gate **fails**.
Draft wall is 8.723402/8.618562 ms and iteration 1.591323/1.556495 plain steps (4k/32k), original H4 attention,
exclusive card1/busy box. Source `18d366bf`; `task6/candidate64-int8/{acceptance.json,iter-*.log}`.

The H1/B4 row split subsequently passes all primitive and real-model bitwise checks. At pos4096/K7 attention
falls 2.696458 → 2.084063 ms; at K2, 1.595521 → 1.227500 ms. Parallel-exp is slower at steady K7 despite
passing bitwise checks; both local K-cache representations fail and are rejected. H1/B4 policy integration
is the next step. Full cost targets and Task 8 remain unpassed.

H1/B4 policy integration subsequently passes four device checks with no skips (kernel, profile, engine,
golden/A4 greedy identity); the full int8 H1/B4 numerical check subsequently passes on card0
(`resume-numerical-card0.log`, DONE rc0). The int8-body candidate arm measures
8.429809 ms draft / 1.558631 plain-step iteration at 4k, and 8.527476 ms / 1.530302 at 32k. C7 targets still
fail; resource grade is exclusive card1/busy box with concurrent CPU references. See BENCHMARKS for the full
paired full-head/candidate table. The operator deferred proper AutoRound conversion to a later follow-up;
RTN cost diagnostics are complete, with the candidate 4k wall result explicitly noisy. The pre-merge main
recording's three rotations measure 0.7986995x default per-step MTP and 1.1629667x plain decode (busy box,
mixed cards; the last MTP arm overlaps CPU references). This is not a passed speed gate.

**2026-11-13:** main `0b5ef628` is merged at `b5bb4044`; the merged build and six host checks pass. Fresh G0
precedes a new Task 8 dataset (`~/b70/logs/dflash-v2/task8/main-0b5ef628`), with all three recordings,
rotated arms, corpora and the actual candidate-head PP path. The old timings are not reused across the binary
change. The operator restarted detached CPU references during G0 (69 GiB, eight threads, saved layer 26);
overlapping timings are busy-box even though both GPUs are reserved.

A further cold-request bottleneck is isolated before the new collection: DFlash rebuilds its immutable
131072-row RoPE table in every reset. Historical prefix logs show cold reset 324–332 ms versus MTP 7–8 ms,
while prompt processing is nearly equal (96 versus 93 ms). Moving table generation/upload to construction
reduces the focused buffer reset 244.426 → 0.179 ms. A new device regression fails before the change and
passes after it, checking initial RoPE against the independent twin, immutable-table preservation across
reset and zeroed ring/control/readback. Seven selected checks pass; the real-engine regression subsequently passes in 297.36 s. This addresses cold-request overhead; the measured draft/verify C7 costs remain failed.

The completed main replay (three rotated rounds, all 78 requests) measures **1.193081621x default per-step
MTP** and **1.774903067x plain decode**. All six speculative-arm/round comparisons pass the f32 judge with
0 FAIL and 0 SKIPPED after the 88 new references finish. Whole-box GPU reservation, physical card0, CPU
references active at eight threads: busy-box grade. Source `a83322bb`, merged main `0b5ef628`; evidence
`~/b70/logs/dflash-v2/task8/main-0b5ef628/opencode-session-int8/judge-mid-refs.log` and the nine saved arms.
G0 passes 2703 common compiled kernels (254 differ only in the normalised tree path), 44 drafter additions,
zero removals. This clears the main recording's 10% bar; the other recordings and corpus/benchy judgments
still gate the full result. The actual candidate-head PP check subsequently passes 39/39 requests
(37 identical, 0 FAIL / 0 SKIPPED); its four short references take 137 s at eight threads. The C7 draft/iteration targets remain failed; the default is unchanged.

The first round of the 21-request long-context control recording needs 26 new contexts of up to 102k tokens.
The current planner estimates about 138 CPU hours before chunking overhead; the operator's budget choice
is pending. The 88 main and four PP references are complete. GPU collection continues independently.

### Completed GPU collection and current verdict (2026-11-13)

All 27 recording arms and nine corpus/benchy arms finished; GPU locks are released. Main/controls DFlash/MTP
ratios are 1.193081621 / 1.226568536 / 1.319626438. Golden 1.058189106, A4 1.826279459, tg128 1.382364965 and
tg128 at depth32768 1.486807750 pass. **pp4096 fails:** 0.955104041 (1489.321 versus MTP 1559.328 t/s), a
4.49% loss against the allowed 3%. The primary speed gain does not make the full G-speed gate pass.

Saved pp4096 round-1 logs: prompt processing ~2629 versus 2510 ms, prefix storage ~65 versus 57 ms, cold reset
~3.4 versus 2.8 ms. About 42.23 ms must be removed from the median DFlash prompt time to reach the 3% limit.
The staged 8-row tail projection is the next profiling target; attribution is an inference from the path
and earlier component costs. C7 remains failed, and proper AutoRound remains deferred.

Main, all golden/A4 comparisons and candidate-head PP pass f32 judging with zero failures/skips. **94 new
references** are complete (88 main, four PP, two corpus); the corpus pair takes 74 s. The complete remaining
plan is 32 references / 26 groups, estimated 140.6 CPU hours before chunk/stream overhead, held for the
operator's budget choice. No model or GPU job is left running. No default promotion; item12 stays open.
Final machine result: `~/b70/logs/dflash-v2/task8/main-0b5ef628/gates-after-core-refs.json`. Full tables are in
BENCHMARKS; the missing-reference plan is `task8-final-ref-plan-v7/{plan.log,summary.json}` in the same log topic.

**pp4096 fixed (2026-11-13, branch `dflash-pf-projection`).** The cause is measured: `Engine::drafter_project_staged`
projected the drafter's 2,048 prompt rows 8 at a time (256 submits), 131.93 ms median per 4,097-token prompt - the
whole gap. It now runs once per prefill chunk through the prefill GEMM (`runtime::dflash::project_wide`,
`df_proj_wide.cl`). The resulting drafter context matches the 8-row path within bf16 rounding (worst row relative
L2 1.33e-3, about 99 % of the values bitwise; `dflash_draft_test` bar 2e-3). Interleaved arms after a warm-up,
3 rounds of 3 runs each, server prompt-pass medians: DFlash before 2617.75 ms, DFlash after 2497.05 ms, MTP
2498.65 ms; projection 131.93 → 13.43 ms; llama-benchy pp4096 medians 1494.58 / 1588.04 / 1566.64 t/s. DFlash's
prompt pass is now level with MTP (0.9994x its time), inside the 3 % bar. The measurement held the whole box, with
the server on device 0. The box was busy: CPU reference work was running (a `qwen_forced.py` docker job was up at
the end), and other trees' card jobs waited for the lock. Evidence: `~/b70/logs/dflash-pf-projection/timing.log`,
`timing/`, `timing-context.log`.

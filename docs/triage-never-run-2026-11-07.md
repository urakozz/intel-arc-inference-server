# Triage: box stages never run or never passed (2026-11-07)

A proposal for the operator to approve (open-work "Now 3"); nothing is deleted or edited yet. It sorts every box
validation stage that has never run, or never passed, into what to do with it. Reviewed the same day against the code,
the specs and tonight's run on the box (state `f73b2490562a-vs-f73b2490562a`, read at ~15:10 while it was at row 54).

## In short

233 stages: **74 to run**, **43 already done**, **45 speed measurements** for the quiet-box session, **67 to delete**,
**4 on hold** (plus 4 Agnes stages already removed with the model).

- **To run (74)** - in plain words:
  - **CPU reference jobs (17)**: the "right answers" the card's output is judged against, computed on the box's CPU
    from the original bf16 models. They are the bottleneck: hours to days each, one at a time. Tool-call references for
    Qwen3.8 (new: its job now exists), K2-Horizon, Bolt and Flash-Next; Bolt's golden sets and its quantisation check;
    Flash-Next's golden sets (to remake: two of its real-checkpoint tests fail until then); the bf16 references the long
    MTP tests need; the DFlash drafter's reference (running now).
  - **Tool-call quality on the card (4)** for Qwen3.8 over the int8 KV cache, K2, Kolibri and Bolt - each after its
    CPU reference.
  - **Long-context recall (5)**: hide a passkey deep in a 32k-260k prompt and ask for it back (K2, Kolibri), Bolt's
    golden sets at 32k, and int8-vs-bf16 KV over long generations.
  - **Correctness checks in the nightly set that tonight's run has not reached yet (38)**: Bolt (kernels, synthetic and
    real checkpoints, two cards, server, snapshots), its LiLiCorr draft model and prompt lookup, the small-batch GEMM in
    production, prompt lookup inside DFlash, the 4096-row prefill chunk, the batching probe's check, tensor-parallel host
    tests.
  - **Flash-Next failures and blocks from tonight (7)**, and 3 more card runs (Bolt loaded on two cards, Bolt's live
    agentic loop, Qwen3.8's speculative-decoding contexts dumped for their CPU references).
- **To delete (67)**: 31 "recap" stages that only repeat the G0 gate (below); one-off probes whose decision is made and
  written down (Ornith, K2, Kolibri and Flash-Next tuning sweeps and profiles, the small-batch GEMM probe, the tensor
  parallel probe, the 4096-chunk probe, DFlash's speed verdict); the expert-pruning half of the dropped expert-offload
  spec; and two third-party checkpoints the "one format" rule excludes. Almost no engine code goes: these are registry
  lines and probe tools.
- **On hold (4)**: the expert-offload probe's routing traces, hit-rate curves and projection, and the PLE host-read
  rate. Open-work's next probe (Now 4, Flash-Next's expert tiering) needs exactly these; decide with it.

### What needs the operator

1. **Deleting the recaps is not only deleting lines.** 12 needed stages wait on a recap (`r19.load` on `r19.r0`;
   `r60.k1`, `r60.p0`, `r61.k1`, `r63.cost`, `r64.k1`, `r65.d3`, `r68.check`, `r69.kernels`, `r70.d2`, `r75.p0`,
   `r75.gates`, `r75.pp` on their row's `k0`). A recap needs `g0.suite` to PASS; tonight it FAILs on three known tests,
   so the recaps SKIP and take those stages with them: `r19.load` SKIPped tonight for this reason, and rows 60-75
   will too. **Proposal:** delete the recaps and make those 12 wait on `g0.sha` (the build gate) instead. Seven more
   recaps outside this triage (`r31.k0`, `r36.k0`, `r56.k0`, `r62.k0`, `r64.k0`, `r66.k0`, `r77.k0`) are the same kind
   and should go with them.
2. **Row 36 (expert offload) is on hold, not deleted.** Spec 22 was dropped (open-work, 2026-11-06), but open-work's
   Now 4 - the expert-tiering probe - must "record Flash-Next's real routes, size the VRAM expert slots from them,
   measure the hit rate" with the experts in pinned host RAM read over PCIe. That is what `r30.traces` (routes),
   `r36.curve` (hit-rate curves), `r36.project` (the projection) and row 36's host-link measurements (copy, zero-copy,
   host bandwidth, pinning - all ten of its other stages PASSed today) do. Delete only the expert-pruning stages
   (lossy, not part of Now 4) now; decide the rest with Now 4. Meanwhile, row 36's timed stages could become opt-in
   (they run ~10 min every night).
   Also: spec 22's own file still says "outline"; only open-work records the drop.
3. **The opencode recording.** Spec 19 closed DFlash's speed verdict on 2026-10-28 with "no opencode recording is made".
   So `r56.d5_opencode` goes (c). Two speed stages still wait on a recording: `r2.opencode` (MTP auto on a real
   agentic session) and `r12.d5` (prompt lookup). Options: (1) record one session with `b70-serve --log-requests`
   (recommended: agentic coding is the target workload, and 13b's speed record wants an opencode-shaped replay too);
   (2) delete both.
4. **Two new Flash-Next failures tonight** (rows 32 / 33, two cards): `r32.pp` - `qwen4exp_pp_test` aborts in its
   Intel-checkpoint part: all 38 layers do not fit two cards (59.0 GB planned); the test asks for more than the cards
   hold. `r33.pp` - `qwen4exp_prefill_intel_pp_test`: expert and sparse-attention selections differ beyond the
   near-tie gap at rows 4096 and later (possibly the same old gap rule as the golden sets being remade - not
   checked). `r32.cli` / `r33.cli` SKIP behind them.
5. **Flash-Next's tool-call reference has no consumer.** `r34.a4_ref` makes it (11 of 36 done when the CPU chain
   died), but the stage that was to score the engine against it was spec 22's. Add an engine tool-call stage for
   Flash-Next (recommended - it is needed anyway once Now 4 runs the full model) or drop the reference.
6. **`r31.ple_kl` is blocked on a script never written** (plan 21q Task 3's `evaluate.py`) and on the 360 GB bf16
   original. Build it, or accept the int8 PLE table on the golden gates alone.
7. **The W4A4 probe's verdict table is not written** (`r35.grid`, `r35.per_class` ran; plan Task 2 Step 3, a Mac
   job from the JSON on the box). After it, the two stages and `w4a4_grid.sh` can go.
8. **Tonight's run is older than today's fixes** (it runs f73b249). So `r2.auto_rows`, `r12.a4` (the old "identical
   output" bar), `r6.mtp_long` (the long MTP tests before the reviewed rows) and `lilicorr_draft_test` (before
   `df_exp`) fail tonight under the old code; the fixes are on main and the next run checks them. Tonight's `g0.suite`
   FAIL is three real tests this time (the two Flash-Next sets, LiLiCorr's exact arm), not the harness.

## Words used below

**Verdicts.** **a** - needed, run it on the box. **a-done** - needed and already PASSed; re-run only if its inputs
change. **b** - a speed number: it belongs to the quiet-box session (open-work Now 5), where the box runs nothing else
and the clocks are settled; a default b still runs nightly, but only the quiet-session number counts. **c** - obsolete:
delete the stage (and its code, where nothing else uses it). **hold** - keep untouched until a named decision.

**Stages.** `rN.name` is a stage of row N of the box test list (`docs/superpowers/plans/box-validation-queue.md`;
`tools/box_validate.sh --list`). *default* stages run in every validation run, *opt-in* ones only when named
(`--with`), *manual* ones only print commands. *one card* / *two cards*: what it locks.

**Models.** Qwen3.8 (27B dense, the main model), Ornith (Ornith 1.5, 35B mixture-of-experts), K2 (K2-Horizon,
MoE with routed attention), Kolibri (Kolibri-1, German, MoE, two cards), Flash-Next (Qwen3.8-Flash-Next, a MoE too big
for two cards in full - run truncated to 18 / 38 layers), Bolt (Nemotron 3.5 Bolt, hybrid Mamba / attention MoE, two
cards).

**Checks.**
- **G0** - the "nothing else changed" gate: every kernel binary that existed before is byte-identical, the old tests'
  outputs identical line for line, and the whole test suite passes (`g0.suite`).
- **Recap** - a row's first stage (`k0`, `r0`) that re-checks G0 for that row's change. Against main itself G0 is
  trivially true, and the G0 stages are kept anyway, so a recap proves nothing of its own. The few extras some carry
  are covered elsewhere: "these kernel binaries were built" (the tests that load them fail otherwise), tests that the
  suite runs anyway, and log excerpts.
- **CPU reference** - the original bf16 model run in fp32 on the box's CPU: the answers the card is judged against.
- **Golden set / golden gate** - fixed prompts whose reference tokens and logits the engine must match (ties within a
  measured noise allowed: the **f32 judge** decides a near-tie by an fp32 reference).
- **A4** - the tool-call score: the engine's answers to 36 tool-call scenarios compared with the bf16 model's (Qwen3.8's
  bar: 25 / 36).
- **Passkey** - a number hidden at 5 / 50 / 95 % of a very long prompt; the model must repeat it (long-context recall).
- **Bitwise** - identical to the last bit; used when two code paths must compute the same thing (two cards vs one, a
  restored snapshot vs a cold run).
- **P0** - a spec's first probe: measure before building; its decision is recorded in the spec.
- **MTP** - the model's own multi-token-prediction head drafts the next tokens, the model verifies them in one step.
  **DFlash** - a separate small draft model for the same job (Qwen3.8); **LiLiCorr** - Bolt's draft model; **prompt
  lookup** - drafts copied from the prompt. **Verify(M)** - checking M drafted tokens in one step.
- **D1 / D5** (DFlash), **B2 / B3** (Bolt), **D2** (speculative) - a spec's numbered checks: D1 the drafter against its
  reference, D5 the speed verdict, B2 every kernel against its reference, B3 the golden gate, D2 speculative output
  equal to plain decoding.
- **pp** - `--parallel pp`, one model split across two cards (pipeline parallel). **TP** - tensor parallel (spec 17).
- **int8 KV** - the attention cache stored as int8 instead of bf16 (halves its memory). **PLE** - Flash-Next's
  per-layer embedding table (51.8 GB, read from host memory). **QSA** - Flash-Next's sparse attention.
- **List B** - the zero-config rulings made in the quiet-box session: for each choice (card hand-off copy vs peer,
  eager vs flash attention, int8 KV, draft vocabulary, ...) the winner stays and the rest is deleted.
- **llama-benchy** - the client that measures our server and vLLM the same way. **opencode recording** - a logged
  real coding-agent session, replayed as a speed test.

## Changed in this review

- 20 stages PASSed tonight: a -> a-done (rows 23, 25-28, 32-34).
- `r69.forced_dumps`, `r69.forced_refs`: a-done -> a (re-run: they now also make the long MTP tests' references).
- `r30.golden`, `r30.golden_layers`: a-done -> a (the golden-set remake is this stage re-run).
- `r66.check`, `r68.build`, `r34.accept`: b -> a (a correctness check, a build step and a CPU acceptance count - none
  is a timing).
- `r56.d5_opencode`, `r56.d5_dv`: b -> c (DFlash's speed verdict is closed, spec 19 D5, 2026-10-28).
- `r30.traces`, `r32.ple_rate`, `r36.curve`, `r36.project`: c -> hold (Now 4 needs them).
- `r35.grid`, `r35.per_class`: c -> a-done (their verdict table is still to write).
- `r11.a4_ref` added (new stage: Qwen3.8's tool-call reference, a).

## a - run on the box (74)

### CPU reference jobs (17) - one at a time under the CPU memory cap, outside timed sessions

| stages | what it makes, and why it matters | needs |
|---|---|---|
| `r11.a4_ref` | Qwen3.8's tool-call reference (never existed): behind `r11.a4` and `r75.gates`' tool-call half | Qwen3.8 bf16 hot; est. 10-16 h |
| `r25.a4_ref` | K2's tool-call reference (died in the 2026-11-06 crash) | K2 bf16 hot (75 GB); hours |
| `r57.golden`, `r57.ppl`, `r57.hfcheck` | Bolt's reference golden sets on our export (everything real-checkpoint Bolt waits on), the reference's perplexity and its port checked bitwise against transformers - the sanity of the reference itself | Bolt bf16 hot; hours |
| `r57.golden_bf16`, `r57.a4_ref`, `r57.a4_bf16`, `r57.gateq` | Bolt's tool-call references (int4 and bf16) and the quantisation check "our int4 export does not lower quality" (KL, top-1, A4 against bf16) | days, resumable |
| `r30.golden`, `r30.golden_layers` | Flash-Next's golden sets remade with the block scores: `r32.partial` / `r33.golden_real` fail until then | hours |
| `r34.fixture`, `r34.accept`, `r34.a4_ref` | Flash-Next's MTP-head reference (without it `r34.mtp`'s first part SKIPs), its draft acceptance (feeds list B's MTP-norm choice) and tool-call reference (11 / 36 done; see attention item 5) | hours |
| `r69.forced_refs` | bf16 references for the contexts where speculative and plain decoding part (the long MTP tests' row 68 included) | after `r69.forced_dumps` |
| `r54.d1_ref` | the DFlash drafter's only check against its reference | running tonight |
| `r31.ple_kl` | the int8 PLE table's error against bf16 | **blocked**: script not written (item 6) |

### Card runs after their references, and long context (11)

| stages | what it proves | needs |
|---|---|---|
| `r11.a4`, `r11.near_ties` | Qwen3.8's tool-call score with the int8 KV cache, and 256-token generations int8 vs bf16 KV: the quality half of the int8 KV ruling (`near_ties` is manual - script it into the same session) | `r11.a4_ref`; one card |
| `r25.a4` | K2's tool-call quality | `r25.a4_ref`; one card |
| `r28.a4` | Kolibri's tool-call quality (German set) | **blocked**: its reference needs the 156 GB bf16 download, then `a4_ref.sh kolibri ref`; two cards |
| `r62.a4`, `r62.agentic` | Bolt's tool-call quality through the server; the agentic transcript as a live loop with the prefix cache | `r57.a4_ref`; two cards |
| `r21.passkey` | K2 recall near the one-card int8-KV ceiling (~79k tokens) | FAILed on its last run: read the log first; one card |
| `r28.passkey` | Kolibri recall at 260,000 tokens | two cards, hours |
| `r60.golden32k`, `r61.golden32k` | Bolt's golden gate at 32k tokens, decode and after a 32k prefill | `r57.golden`; two cards, hours |
| `r59.load_real` | Bolt's export loaded on two cards at 262,144 tokens: the plan, nothing left unread, load time, memory | two cards, minutes |

### Flash-Next, failed or blocked tonight (7)

| stages | tonight |
|---|---|
| `r32.partial`, `r33.golden_real` | FAIL as expected: the golden sets still use the old near-tie gap rule - remake them (`r30.golden*`) |
| `r32.pp` | FAIL: `qwen4exp_pp_test`'s Intel-checkpoint part asks for all 38 layers on two cards, 59.0 GB, which does not fit (item 4) |
| `r33.pp` | FAIL: `qwen4exp_prefill_intel_pp_test` - expert / QSA selections differ beyond the near-tie gap at rows >= 4096 (item 4) |
| `r32.cli`, `r33.cli` | SKIP: wait on `r32.pp` / `r33.pp` |
| `r19.load` | SKIP: waits on the recap `r19.r0` (item 1). Ornith's real int4 checkpoint loaded and run on the card; data present |

### Correctness in the nightly set, not reached yet tonight (38)

| stages | what they prove |
|---|---|
| `r57.host`, `r59.pytest`, `r59.synth_host` | Bolt's reference and loader tests (CPU, minutes) |
| `r60.k1`, `r60.synth`, `r60.synth_b3`, `r60.synth_pp`, `r60.golden`, `r60.fuse` | Bolt decode: every kernel against its reference, the synthetic checkpoints (bitwise twice, golden gate, two cards bitwise one), our real export's golden gate on two cards, every fused kernel bitwise the unfused one |
| `r61.k1`, `r61.synth`, `r61.synth_b3`, `r61.synth_pp`, `r61.golden` | the same for Bolt's prefill |
| `r62.snap_synth`, `r62.serve`, `r62.chat_synth` | Bolt served: snapshots restore bitwise, the server starts and answers, chat output = the CLI's |
| `r63.host`, `r64.k1`, `r64.draft` | LiLiCorr's host tests, kernels and draft. `r64.draft`'s exact arm missed its bar before the `df_exp` fix (on main since 29c428e, not in tonight's build): the next run confirms |
| `r65.host`, `r65.reject`, `r65.d3`, `r65.d2`, `r65.prefill`, `r65.prefix`, `r65.serve` | Bolt's speculative decoding (lookup, LiLiCorr): verify bitwise plain steps, output equal to plain decoding at every draft length (also picks LiLiCorr's draft format), prefill and prefix-cache restores bitwise, the 36 tool-call scenarios identical through the server |
| `r66.check` | the batching probe's rows bitwise each sequence run alone - the check before its speed table (`r66.cost`) |
| `r68.build` | builds the small-batch GEMM probes that `r69.speed` uses |
| `r69.host`, `r69.kernels`, `r69.gates` | the small-batch GEMM in production: host tests, kernels against fp64, MTP / DFlash verify judged by the f32 judge |
| `r70.host`, `r70.d2` | prompt lookup inside DFlash: host tests; output equal to plain decoding (its "identical" bar moves to the f32 judge: open-work, to do) |
| `r71.host` | tensor parallel host tests that spec 17b builds on |
| `r75.host`, `r75.gates`, `r75.pp` | prefill in 4096-row chunks: planner tests, the prefill gates at 4096 (tool-call half after `r11.a4_ref`), two cards bitwise |

### Card dumps (1)

| stages | what it makes |
|---|---|
| `r69.forced_dumps` | the forced contexts for `r69.forced_refs`, now with the long MTP tests at 32k / 128k (~6-8 h card time) |

## a-done - already passed (43)

| stages | what passed |
|---|---|
| `r6.passkey120k`, `r7.passkey`, `r11.passkey120k`, `r11.passkey262k`, `r16.passkey`, `r27.p3_mtp` | long-context recall: Qwen3.8 at 120k with MTP, at 95 % of the auto length, over the int8 KV cache at 120k and 250k; Ornith at 120k / 250k; two cards at ~250k with MTP |
| `r25.passkey`, `r25.toolcall` | tonight: K2 recall at 32k / ~40k; K2's tool-call set through the server |
| `r16.a4_ref`, `r16.a4` | Ornith's tool-call reference (36 answers) and its score against it |
| `r10.oracle_f32`, `r14.oracle`, `r14.oracle_f32`, `r16.forced_dumps`, `r16.forced_refs`, `r24.oracle_synth`, `r24.oracle_real`, `r30.ppl`, `r30.hfcheck`, `r31.synth`, `r31.ple_convert`, `r32.oracle_synth` | CPU references and their data: Ornith's and K2's f32 judges, K2's real-weight reference, Kolibri's golden sets, Flash-Next's reference (perplexity, port bitwise transformers), synthetic checkpoints and the int8 PLE file |
| `r8.m3_small` | MTP output equal at draft vocabulary 32k / 64k (delete with the losing sizes after list B) |
| `r23.p1`, `r26.real`, `r28.real` | tonight: two-card Qwen3.8 prefill bitwise one card; Kolibri prefill and serving on the real checkpoint |
| `r27.mtp`, `r27.mtp_kv8`, `r27.prefix`, `r27.prefix_kv8`, `r27.serve`, `r27.golden`, `r27.ornith` | tonight: two-card serving - MTP across the split, the prefix cache (bf16 and int8 KV), start-up at 262,144, one request end to end, Ornith |
| `r32.k3`, `r32.golden`, `r33.prefill`, `r33.golden`, `r34.snapshot`, `r34.mtp`, `r34.serve`, `r34.pp` | tonight: Flash-Next decode, prefill, snapshots, MTP, serving and two cards on the synthetic checkpoints (`r34.mtp`'s first part still SKIPs until `r34.fixture`) |
| `r35.grid`, `r35.per_class` | the W4A4 accuracy grids ran (2026-10-28); their verdict table is a Mac job (item 7), then delete |

## hold - decide with open-work Now 4, the expert-tiering probe (4)

| stages | why kept |
|---|---|
| `r30.traces`, `r36.curve`, `r36.project` | Flash-Next's real routing traces, the hit-rate curves from them and the projection: Now 4's "record the real routes, size the VRAM slots, measure the hit rate". Code: `qwen4exp_oracle.sh` trace / curve / project, `offload_curve.py` |
| `r32.ple_rate` | the PLE table read from host memory by the card: Now 4 reads it from host or NVMe. Code: `qwen4exp_kernels_test --ple-rate` |

Row 36's ten other stages (host tests, build, links idle and under load, copy, zero-copy, replay, host bandwidth,
pin, alias; PASSed, so outside this triage) and `probe_offload` belong to the same hold.

## b - speed, for the quiet-box session (45)

| stages | what it times / which decision it feeds |
|---|---|
| `r6.cost`, `r6.d1`, `r7.depth`, `r8.rows_full`, `r8.auto_rows` | Qwen3.8 MTP cost tables at depth 32k / 120k (`r6.cost` ran tonight on a busy box), MTP speed at depth, decode at the auto length, the draft-vocabulary matrix and `--spec auto` with it (list B's draft-vocabulary ruling) |
| `r2.opencode`, `r12.d5` | MTP auto and prompt lookup on an opencode recording - **blocked**: no recording (item 3) |
| `r10.speed`, `r13.speed`, `r16.mtp_rows`, `r16.benchy` | Ornith decode 4k-250k, prefill, MTP rows, llama-benchy |
| `r14.speed`, `r15.speed`, `r21.speed`, `r25.benchy` | K2 decode, prefill, int8 vs bf16 KV (list B), llama-benchy beside vLLM |
| `r22.s1_32k`, `r23.s2_4k`, `r23.s2_32k`, `r23.s2_64k`, `r23.s2_128k`, `r23.split_sweep`, `r27.benchy` | two cards vs one: decode at 32k (list B: hand-off copy vs peer), prefill at 4k-128k (the three default ones ran tonight on a busy box), the split point, llama-benchy |
| `r24.speed`, `r26.speed`, `r28.benchy` | Kolibri decode and prefill with the eager arm (list B), llama-benchy |
| `r32.bench_attn`, `r32.speed`, `r33.speed` | Flash-Next: one sparse attention kernel at every depth vs two, decode and prefill with the eager arm (list B) |
| `r41.ours`, `r41.vllm`, `r41.vllm_mtp` | the llama-benchy matrix: our server, the vLLM baseline, vLLM with MTP |
| `r55.tps` | DFlash speed (informational) |
| `r60.state_arm`, `r60.speed`, `r61.speed`, `r62.benchy`, `r63.cost`, `r65.benchy` | Bolt: the bf16 SSM state vs fp32 (list B; its golden half runs in the same session), decode and prefill speed, llama-benchy beside vLLM, LiLiCorr's verify / block costs (with `r65.d2` picks the draft format), the speculative speed verdict |
| `r66.cost`, `r66.chunk` | batching: the step table and the interleave chunk cost |
| `r69.speed`, `r69.d1` | verify(M) on the small-batch GEMM vs GEMV, MTP speed at 4096 (against 1.633x) |
| `r70.speed`, `r70.verdict` | prompt lookup inside DFlash vs DFlash / auto / off; the verdict read from it (manual) |
| `r75.speed` | prefill at chunk 4096 vs 2048: the 5d default (+4.2 % measured, under the +5 % bar - the operator decides) |

## c - delete (67)

| stages | why | code that goes |
|---|---|---|
| `r5.r0`, `r10.r0`, `r11.bf16`, `r13.r0`, `r14.k0`, `r15.k0`, `r16.r0`, `r17.k0`, `r19.r0`, `r21.k0`, `r22.r0`, `r23.r0`, `r24.k0`, `r25.k0`, `r26.k0`, `r27.k0`, `r28.k0`, `r32.k0`, `r33.k0`, `r34.k0`, `r54.k0`, `r55.k0`, `r60.k0`, `r61.k0`, `r63.k0`, `r65.k0`, `r68.k0`, `r69.k0`, `r70.k0`, `r71.k0`, `r75.k0` | 31 recaps of G0 (see Words). Tonight `r16.r0` FAILed and the rest SKIPped only because `g0.suite` failed; `r17.k0` PASSed | registry lines and bodies; **first** point the 12 stages that wait on them at `g0.sha` (item 1) |
| `r68.host` | runs `gemm_sm_ref_test`, which `r69.host` runs too | the lines |
| `r10.p0`, `r10.p0_sweeps`, `r13.p0`, `r13.p0_arms`, `r14.sweeps`, `r15.p0`, `r15.p0_arms`, `r24.sweeps`, `r26.p0`, `r32.sweeps`, `r33.p0` | first probes and tuning sweeps of Ornith, K2, Kolibri and Flash-Next decode / prefill whose decisions are in their specs (15, 18, 20, 21; plans 15c-d, 18b-c, 20c-d, 21c-d): e.g. Ornith's int4 cell stays S1 L1, K2's attention stays flash (eager removed). The `p0` profiles PASSed; the sweeps need rebuilds per value, which zero-config does not want | none beyond the stages (the manual ones only print commands) |
| `r67.profile` | a prefill memory-bandwidth diagnostic that gates nothing; it ran | none |
| `r29.cost` | DFlash verify cost at M = 5..8: `r69.speed` times verify(M) for M 1..8 on today's kernels; DFlash is opt-in (D5 closed) | none (`probe_mtp_steps` stays); drop its "not run" row in BENCHMARKS.md |
| `r68.check`, `r68.shapes`, `r68.attrib`, `r68.numerics`, `r68.verdict` | the small-batch GEMM probe (spec 28a): GO, built and merged as 28b (`docs/probes/probe-small-m-gemm-2026-11-01.md`); `r69.*` checks and times the production kernels | the stand-in kernels stay only while `r69.speed`'s lever table uses them |
| `r71.build`, `r71.facts`, `r71.bar`, `r71.ar`, `r71.step`, `r71.verdict` | the tensor-parallel probe (spec 17a): GO on a quiet box 2026-11-06 (spec 17 §9, `docs/probes/probe-tp-2026-11-01.md` §8) | none yet: `probe_tp` stays for 17b |
| `r75.p0` | the 4096-row chunk probe: decided, 5d built and merged | none (`probe_chunk_rows` after `r75.speed`) |
| `r56.d5_a4`, `r56.d5_dv`, `r56.d5_benchy`, `r56.d5_pp`, `r56.d5_opencode` | DFlash's speed verdict, closed 2026-10-28 (spec 19, D5: an option, not the default; no opencode recording is made); the numbers are in BENCHMARKS.md (A4 1.033x MTP, golden 0.791x, llama-benchy 0.82-0.88x, two cards 1.033 / 0.793, the draft vocabulary +4 % in-sample) | none |
| `r36.reap`, `r36.prune`, `r36.prune_a4` | expert pruning (spec 22 P0.8, plan 22m): drops experts, so it is lossy; not part of Now 4's tiering; spec 22 dropped | the masks in `offload_curve.py` / `qwen4exp_ref.py` (masked routing) and `qwen4exp_oracle.sh` reap / prune - after Now 4, with the rest of row 36 if it goes |
| `r4.g128`, `r18.ct` | third-party g128 checkpoints (GPTQ, RedHat compressed-tensors): the "one format" rule (`docs/02-formats.md`: AutoRound int4 g64 sym, quantised by us) | none: the g128 and compressed-tensors readers stay (Intel's Flash-Next and Bolt's g128 spelling, K2 / Kolibri / Bolt loaders) |

## Removed with Agnes (drop-agnes, 2026-11-07)

`r1.a4`, `r1.prefix_benchy`, `r1.vllm`, `r27.agnes`: gone from `stages.sh` with the model (tonight's `r27.agnes` FAIL
is from the build before the removal).

## Suggested order

1. The next validation run on main (it has today's fixes): rows 55-75 after the recap rewiring (item 1); then
   `r19.load`.
2. CPU references, one at a time: `r11.a4_ref` -> `r69.forced_dumps` (card) + `r69.forced_refs` -> `r25.a4_ref` ->
   `r30.golden*` -> `r57.golden`, `r57.hfcheck`, `r57.ppl` -> `r57.golden_bf16`, `r57.a4_ref`, `r57.a4_bf16`,
   `r57.gateq` (days) -> `r34.fixture`, `r34.accept`, `r34.a4_ref` -> the Kolibri reference (after the download).
3. Card runs after their references: `r11.a4` + `r11.near_ties`, `r25.a4`, `r21.passkey` (read its FAIL first),
   `r28.passkey`, `r28.a4`, `r62.a4`, `r62.agentic`, `r60.golden32k`, `r61.golden32k`, `r59.load_real`.

## Totals

| selection | stages | a | a-done | b | c | hold |
|---|---|---|---|---|---|---|
| default, never passed | 112 | 46 | 20 | 5 | 39 | 2 |
| opt-in | 106 | 27 | 22 | 39 | 16 | 2 |
| manual | 15 | 1 | 1 | 1 | 12 | 0 |
| **all** | **233** | **74** | **43** | **45** | **67** | **4** |

Sources: `tools/box_validate.sh --list` on main 5e8147c (303 default, 106 opt-in, 15 manual stages, the 4 G0 stages and
`pre`), the stage bodies in `tools/box_validate/stages.sh`, the box test list, `docs/open-work.md`, the specs,
`docs/BENCHMARKS.md`, the box's `~/b70/logs/stage-latest.txt` (2026-11-04) and tonight's `*.status`.

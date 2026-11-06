# Triage: box stages never run or never passed (2026-11-07)

For open-work "Now 3" (everything coded but never run on the box). **A proposal for the operator to approve; no
stage is deleted or edited yet.**

**Scope.** Every opt-in and manual stage (106 + 17 = 123), plus every default stage with no PASS in
`~/b70/logs/stage-latest.txt` (the latest result of each stage over all runs, as of 2026-11-04) nor in tonight's
run (113). 236 stages in all.

**Inputs.** `tools/box_validate.sh --list` on main 6c1aacf (310 default, 106 opt-in and 17 manual stages, the 4 G0 stages and
`pre`), the stage bodies in `tools/box_validate/stages.sh`, `docs/superpowers/plans/box-validation-queue.md`,
`docs/box-day-plan.md` §3-§4 (which decision each stage feeds), `docs/open-work.md`, docs/BENCHMARKS.md (rows
measured by hand outside the runbook), the box's `stage-latest.txt`, tonight's
`0c0977669f19-vs-0c0977669f19/*.status` read at ~23:00 (only row 1 had run: 13 PASS, `r1.speed` running) and its
`have.env` (which data the box has).

**Verdicts.**

- **a** - needed: run it on the box. The reason says what it proves and what it needs.
- **a-done** - needed, and it already PASSed (mostly CPU references and passkeys): nothing to run unless its
  inputs change. Kept.
- **b** - a speed number: it belongs in the quiet-box measurement session (open-work Now 4). A default stage
  marked b still runs in the nightly set, but its number binds only from the quiet session.
- **c** - obsolete: the reason says why (a decided probe, a dropped spec, a recap with nothing left to check)
  and whether code goes with it.

"Tonight" in the reason column means the stage is in the default set of the validation run now on the box.

## What the operator should know first

1. **There is no Qwen3.8 A4 reference on the box or the Mac.** The A4 check is the tool-call score: the engine's
   answers to 36 tool-call scenarios compared with the bf16 model's (25 / 36 is Qwen3.8's recorded score). Its
   reference texts (`<name>.bf16.txt`) are not in any `oracle-out-*` directory, `A4_REF_DIR` is unset, and
   `tools/toolcall/a4_ref.sh` has no Qwen3.8 mode. So `r11.a4` (tool calls over the int8 KV cache - the quality
   half of list B's int8 KV ruling) and the A4 half of `r75.gates` (prefill chunk 4096) cannot run, and open-work's
   Waiting row "12 int8 KV: the tool-call rows, in tonight's run" will not be closed tonight (`r11.a4` is opt-in and
   its data is missing). **Fix: one CPU job** - `oracle_generate.py` on `Qwen/Qwen3.8-27B` (the bf16 original is on
   the box) over `tests/golden/toolcall`, under the new CPU memory cap; then `A4_REF_DIR=<that dir>`. Options:
   (1) add a `qwen` mode to `a4_ref.sh` and an `oracle-out-qwen-a4` data key (recommended: it then works like every
   other model's); (2) run it by hand once and set `A4_REF_DIR`.
2. **Tonight's `g0.suite` FAIL is the harness, not a test.** In the 22:03 restart, `ctest --show-only=json-v1` in the
   baseline tree `~/b70-inference-server-g0-6c1aacf3eaaa` returned rc 1 (no ctest tree there: `g0.build` was skipped
   as already PASS from the 17:02 start, whose baseline was another directory), the JSON file was empty and
   `junit.py todo` crashed (`JSONDecodeError`). The other 301 stages show SKIP "blocked: G0 not passed" from the first
   start; the restart runs row 1 on. Every recap stage below (`need_pass g0.suite`) will SKIP or FAIL tonight
   whatever the code does.
3. **Six needed stages are blocked on data nobody has made:** no opencode request recording (`r2.opencode`,
   `r12.d5`, `r56.d5_opencode` - the agentic-session speed bar; record one with `b70-serve --log-requests`), no
   Kolibri A4 reference (`r28.a4`: needs the 156 GB bf16 source), no Qwen3.8 A4 reference (`r11.a4`, above),
   and plan 21q's `evaluate.py` was never written (`r31.ple_kl`).
4. **Flash-Next's A4 reference has no consumer.** `r34.a4_ref` makes it (11 of 36 done when the CPU chain died),
   but the only stage that was to score the engine against it was spec 22's (dropped). Either add an engine A4
   stage for Flash-Next or drop the reference (recommended: add the stage - tool-call quality is the first rule).
5. **31 recap stages** (`rN.k0` / `rN.r0`, `r11.bf16`) only re-check G0 ("every pre-existing binary identical"),
   which says nothing when the nightly run compares main with itself; they never passed because G0 never passed in
   their state. Verdict c: delete the registry lines (no other code).

## Suggested order for the a's

1. Tonight's default set covers 65 of them (marked "tonight").
2. CPU references, one at a time under the memory cap, outside timed sessions: Qwen3.8 A4 reference (new) ->
   `r25.a4_ref` (K2 bf16 hot) -> `r57.golden`, `r57.hfcheck`, `r57.ppl` (Bolt) -> `r57.golden_bf16`, `r57.a4_ref`,
   `r57.a4_bf16`, `r57.gateq` (Bolt, days) -> `r34.fixture`, `r34.a4_ref` (Flash-Next) -> `r54.d1_ref` (DFlash) ->
   the Kolibri A4 reference (`a4_ref.sh kolibri ref`, after the 156 GB download) -> `r1.a4`'s Agnes reference.
3. Card runs after their references: `r11.a4` + `r11.near_ties` (one card), `r25.a4`, `r21.passkey` (look at its FAIL
   first), `r28.passkey`, `r28.a4`, `r62.a4`, `r62.agentic`, `r60.golden32k`, `r61.golden32k`, `r59.load_real`.

## What would be deleted (if approved)

- **Row 36 whole (spec 22, expert offload - dropped):** `tools/probe/probe_offload*`, its two kernels
  (`po_moe_M1_E512_T10_D2560_I640_SH4`, `po_kernels`), `mk_offload_kernel.py`, `qwen4exp_oracle.sh`'s curve / reap /
  prune / project modes, `HAVE_probe_offload_bin`, the 12 r36 default stages that PASSed. Kept: `offload_links.sh`
  (`r71.facts` uses it) and `expert_budget` if the planner still calls it (`qwen4exp_sizes.cc` does today). With it:
  `r30.traces` (the oracle's trace mode) and `r32.ple_rate` (`qwen4exp_kernels_test --ple-rate`).
- **Decided probes:** the manual sweeps / P0 arms of rows 10, 13, 14, 15, 24, 26, 32, 33 (no code: they only print
  commands); the PASSed P0 profiles `r10.p0`, `r13.p0`, `r15.p0`, `r67.profile`; row 68's stand-in checks (keep
  `r68.build` and `probe_small_m` / `probe_verify_m` for `r69.speed`); row 71's arms (keep `probe_tp` until 17b is
  decided); `r29.cost`; `r75.p0`; DFlash's measured D5 rows `r56.d5_a4`, `r56.d5_benchy`, `r56.d5_pp`; row 35's
  grid once its verdict table is written.
- **The 'one format' rule:** `r4.g128`, `r18.ct` (manual; the loaders' g128 / compressed-tensors paths stay - other
  models use them).
- **Recaps:** the 31 above plus `r68.host` (a duplicate of `r69.host`).

## The stages, by row

### Row 1 - spec 14 - Agnes 3.0 Flash, ModelDesc (2026-10-03-spec14-validation-checklist.md)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r1.a4` | manual | never run | **a** | Agnes's tool-call score against its bf16 reference (agentic quality on the second Qwen-family model). Needs the Agnes bf16 original downloaded and hours of CPU under the memory cap; after the Qwen3.8 / K2 / Kolibri / Bolt references |
| `r1.prefix_benchy` | optin, gpu-self | never run | **b** | llama-benchy prefix-cache rows on Agnes: a speed record |
| `r1.vllm` | manual | never run | **c** | Agnes's vLLM row needs an unmerged vLLM PR overlaid on the image; the quiet session's vLLM baseline is Qwen3.8's (r41.vllm). Manual: no code |

### Row 2 - spec 13b Task 1 (step_many signature) and spec 8 §10 (--spec auto)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r2.opencode` | optin, gpu-self | SKIP | **b** | the opencode replay with --spec auto: the agentic-session speed number. Blocked: no request recording (OPENCODE_LOG) exists - record one with --log-requests first |

### Row 4 - g128 symmetric loader (2210406)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r4.g128` | manual | never run | **c** | a third-party g128 GPTQ checkpoint: the 'one format' rule (AutoRound int4 g64 sym, re-quantised by us). Manual: no code; the g128 read path stays (Bolt's g128 spelling uses it) |

### Row 5 - spec 15b - ModelDesc carries every per-model width (2026-10-04-spec15b-validation-checklist.md)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r5.r0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |

### Row 6 - spec 8 §12 - MTP at every context length

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r6.cost` | default, gpu | SKIP | **b** | the MTP cost table at depth 32k / 120k (timed probe_mtp_steps): feeds the speculative cost defaults |
| `r6.passkey120k` | optin, gpu | PASS | **a-done** | passkey 3/3 at 120k with MTP: PASS; long-context recall - re-run only if the attention or MTP path changes |
| `r6.d1` | optin, gpu-self | never run | **b** | A12's MTP speed rows at depth 32k / 64k (hours) |

### Row 7 - spec 6 §10 - --max-len auto (b70-serve's default), the memory planner

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r7.passkey` | optin, gpu | PASS | **a-done** | passkey at 95 % of the auto length: PASS |
| `r7.depth` | optin, gpu | PASS | **b** | decode at depth N - 512 under --max-len auto: a speed number (PASS on the old driver; re-measure on the new one) |

### Row 8 - spec 8 §11 - --draft-vocab (int8 and bf16 heads), --spec auto draft costs scaled by |V'|

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r8.rows_full` | optin, gpu | PASS | **b** | the draft-vocabulary speed matrix (bf16 head, sampled): feeds list B's draft-vocabulary ruling (PASS on the old driver) |
| `r8.m3_small` | optin, card | PASS | **a-done** | greedy equality at draft vocab 32k / 64k: PASS; delete with the losing sizes after list B's draft-vocabulary ruling |
| `r8.auto_rows` | optin, gpu | never run | **b** | --spec auto with a draft vocabulary vs fixed K through b70-serve: list B's draft-vocabulary ruling |

### Row 10 - spec 15c - Ornith 1.5 35B-A3B decode (spec 15 §10)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r10.r0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r10.oracle_f32` | optin, cpu | PASS | **a-done** | Ornith's f32 judge data: PASS; re-run only if the golden sets change |
| `r10.p0` | optin, gpu | PASS | **c** | Ornith decode P0 profile: ran (PASS); the int4 a\|\|b cell stays S1 L1 (an S > 1 form needs an unbuilt arm), nothing decided by re-running it. No code beyond the stage |
| `r10.speed` | optin, gpu | never run | **b** | Ornith decode at 4k-250k: speed record |
| `r10.p0_sweeps` | manual | never run | **c** | a decided probe's build-variant sweeps (rebuild per define); the zero-config cleanup adds no tuning knobs. Manual: no code |

### Row 11 - spec 12b - the int8 KV cache (--kv-cache int8, rotkv; spec 12 §9, plan 12b)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r11.bf16` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r11.passkey120k` | optin, gpu | PASS | **a-done** | passkey at 120k over the int8 KV cache: PASS |
| `r11.passkey262k` | optin, gpu | PASS | **a-done** | passkey at 250k ids, max_len 262144 (int8 KV only): PASS |
| `r11.a4` | optin, card | SKIP | **a** | **the tool-call score with the int8 KV cache** against Qwen3.8's bf16 reference (>= 25/36): the quality half of list B's int8 KV ruling. Blocked: no Qwen3.8 A4 bf16 reference is on the box or the Mac (A4_REF_DIR unset, a4_ref.sh has no qwen mode) - a CPU job (oracle_generate.py on Qwen/Qwen3.8-27B, which is on the box) makes it; one card |
| `r11.near_ties` | manual | never run | **a** | 256 greedy ids per golden prompt, int8 vs bf16 KV, the first divergence judged by the tie rule: long-generation evidence for the int8 KV ruling; six b70-decode runs, one card (manual - script it into r11.a4's session) |

### Row 12 - spec 19e host side - --spec lookup, prompt lookup (plan 19e)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r12.d5` | optin, gpu-self | SKIP | **b** | D5 for prompt lookup on an opencode recording (lookup vs auto vs off). Blocked: no recording exists |

### Row 13 - spec 15d - Ornith 1.5 prefill: device routing, grouped expert GEMMs (spec 15 §11, plan 15d)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r13.r0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r13.p0` | optin, gpu | PASS | **c** | Ornith prefill P0 profile: ran (PASS); its levers are recorded, none is being built. No code |
| `r13.speed` | optin, gpu | never run | **b** | Ornith prefill pp4k-130k: speed record |
| `r13.p0_arms` | manual | never run | **c** | a decided probe's arms that need a rebuild or an unwritten probe. Manual: no code |

### Row 14 - spec 18b - K2-Horizon decode on one card (spec 18 §10, plan 18b)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r14.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r14.oracle` | optin, cpu | PASS | **a-done** | K2's real-weight CPU reference: PASS (oracle-out-k2) |
| `r14.oracle_f32` | optin, cpu | PASS | **a-done** | K2's f32 judges: PASS |
| `r14.speed` | optin, gpu | never run | **b** | K2 decode at 4k-32k: speed record |
| `r14.sweeps` | manual | never run | **c** | a decided probe's {S, layout} / K-split sweeps (rebuild per value). Manual: no code |

### Row 15 - spec 18c - K2-Horizon prefill: grouped MoVA and MoE, flash attention at head_dim 128 (spec 18 §11, plan 18c)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r15.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r15.p0` | optin, gpu | PASS | **c** | K2 prefill P0 profile: ran (PASS); K2's attention is decided (flash; B70_K2_ATTN removed 2026-11-06). No code |
| `r15.speed` | optin, gpu | never run | **b** | K2 prefill pp4k-32k: speed record |
| `r15.p0_arms` | manual | never run | **c** | a decided probe's rebuild arms (EXP2, RPW, sort). Manual: no code |

### Row 16 - spec 15e - Ornith 1.5 served: template, tool calls, the MoE MTP head (spec 15 §12, plan 15e)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r16.r0` | default, card | FAIL | **c** | a recap of G0 plus kbins and four MTP tests that g0.suite / the row's own stages run anyway; its FAIL was the recap, not a test. Only the registry lines go |
| `r16.forced_dumps` | optin, card | PASS | **a-done** | the forced contexts for the f32 judges: PASS |
| `r16.forced_refs` | optin, cpu | PASS | **a-done** | their f32 references: PASS |
| `r16.mtp_rows` | optin, gpu | never run | **b** | Ornith's MTP off / 1 / 3 / auto rows through b70-serve: speed |
| `r16.passkey` | optin, gpu | PASS | **a-done** | Ornith passkey at 120k / 250k: PASS |
| `r16.benchy` | optin, gpu-self | never run | **b** | Ornith llama-benchy rows |
| `r16.a4_ref` | manual | never run | **a-done** | Ornith's A4 set and reference: made (oracle-out-ornith-a4 holds 36 references). Manual: nothing to run |
| `r16.a4` | optin, card | PASS | **a-done** | Ornith A4 against its reference: PASS |

### Row 17 - spec 18d host side - K2-Horizon's template, tokenizer, tool calls and server dispatch (spec 18 §12, plan 18d)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r17.k0` | default, card | FAIL | **c** | a recap: the ten server / template tests it re-runs are in g0.suite and `x`; its FAIL was a test inside that the suite judges anyway. Only the registry lines go |

### Row 18 - compressed-tensors symmetric pack-quantized loader (branch ct-sym-loader; spec 20 §9 as built, docs/13)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r18.ct` | manual | never run | **c** | RedHat's g128 compressed-tensors checkpoint: the 'one format' rule. Manual: no code; the compressed-tensors reading is shared with the K2 / Kolibri / Flash-Next / Bolt loaders and stays |

### Row 19 - the real Ornith int4 checkpoint - its int4 a||b, vocab_used 248077, doc_w (branch ornith-real-checkpoint; spec

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r19.r0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r19.load` | default, card | SKIP | **a** | Ornith's real checkpoint on the card: the int4 a\|\|b classification, 0 unconsumed, the W check; one card, data present - in tonight's default set (it SKIPped before on a missing prerequisite) |

### Row 21 - spec 18e Task 1 - K2-Horizon's int8 KV cache, rotkv at head_dim 128 (--kv-cache int8; spec 18 §13, plan 18e)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r21.k0` | default, cpu | SKIP | **c** | a recap of G0; its 'pre-18e binaries' check compares against a baseline older than K2, meaningless when the baseline is main itself. Only the registry lines go |
| `r21.passkey` | optin, gpu | FAIL | **a** | **K2 passkey near the one-card int8-KV ceiling (~79k ids)**: long-context recall. FAILed on its last run - look at the log before re-running; one card, quiet not needed |
| `r21.speed` | optin, gpu | never run | **b** | K2 int8 vs bf16 KV decode speed: list B's int8 KV ruling |

### Row 22 - spec 16b - pipeline parallel decode across two B70s (--parallel pp; spec 16 §8, plan 16b)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r22.r0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r22.s1_32k` | optin, gpu | never run | **b** | S1 at depth 32k: one card vs --parallel pp copy / peer (list B: hand-off copy vs peer) |

### Row 23 - spec 16c - pipeline parallel PREFILL across two B70s (--parallel pp --prefill / --prefill-length; spec 16 §9, 

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r23.r0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r23.p1` | default, gpu | RUNNING | **a** | two-card prefill bitwise one card (pp_prefill_test, 3 forms): correctness of --parallel pp prefill; both cards; running tonight |
| `r23.s2_4k` | default, gpu | SKIP | **b** | S2 two-card prefill speed at pp4096 (default, timed: tonight's busy-box number does not bind) |
| `r23.s2_32k` | default, gpu | SKIP | **b** | S2 at pp32768 (same) |
| `r23.s2_64k` | default, gpu | SKIP | **b** | S2 at pp65536 (same) |
| `r23.s2_128k` | optin, gpu | never run | **b** | S2 at pp131072, the length only two cards hold |
| `r23.split_sweep` | optin, gpu | never run | **b** | the time-balanced cut (splits 30 / 32 / 34): a speed tuning of --pipeline-split auto |

### Row 24 - spec 20c - Kolibri-1 decode, one card then two (spec 20 §11, plan 20c)

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r24.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r24.oracle_synth` | optin, cpu | PASS | **a-done** | Kolibri synthetic golden sets + f32 judges: PASS |
| `r24.oracle_real` | optin, cpu | PASS | **a-done** | Kolibri real golden set + f32 judge: PASS |
| `r24.speed` | optin, gpu | never run | **b** | Kolibri decode, copy / peer / eager arms: list B (Kolibri eager attention, hand-off) |
| `r24.sweeps` | manual | never run | **c** | a decided probe's sweeps (rebuild per value). Manual: no code |

### Row 25 - spec 18d engine side - K2-Horizon served: K2Engine behind b70-serve, KV-only prefix snapshots, K4 (A4, passkey

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r25.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r25.passkey` | default, gpu | FAIL | **a** | **K2 passkey at 32k and ~40k with bf16 KV** (5 / 50 / 95 % placements): long context. FAILed last time; in tonight's default set; one card |
| `r25.toolcall` | default, card | SKIP | **a** | **K2's tool-call set through b70-serve**: prompt ids = the set's, finish reasons tool_calls, calls parsed. Its data (oracle-out-k2-a4/set) is now present; in tonight's default set; one card |
| `r25.a4_ref` | optin, cpu | FAIL | **a** | **K2's A4 bf16 reference** on the box CPU (hours; FAILed / died in the 2026-11-06 crash). Needs K2 bf16 hot (75 GB on the NVMe) and the CPU memory cap; no GPU |
| `r25.a4` | optin, card | SKIP | **a** | **K2 A4 against that reference**: K2's tool-call quality. After r25.a4_ref; one card |
| `r25.benchy` | optin, gpu-self | never run | **b** | K2 llama-benchy rows beside vLLM's recorded numbers |

### Row 26 - spec 20d - Kolibri-1 prefill: grouped MoE over 384 experts, windowed flash over the ring, one hand-off per chu

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r26.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r26.real` | default, gpu | FAIL | **a** | Kolibri prefill on the real checkpoint, two cards (prefill test, split, golden prefill gates): FAILed on its last run; in tonight's default set; both cards |
| `r26.speed` | optin, gpu | never run | **b** | Kolibri prefill + eager arm: list B (Kolibri eager attention) |
| `r26.p0` | manual | never run | **c** | a decided probe's profile and levers (need a rebuild). Manual: no code |

### Row 27 - spec 16d - pipeline parallel integration: b70-serve --parallel pp, MTP across the split, the prefix cache on t

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r27.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r27.mtp` | default, gpu | SKIP | **a** | MTP across the two-card split bitwise one card: open-work's spec 16 row; both cards; tonight |
| `r27.mtp_kv8` | default, gpu | SKIP | **a** | the same over the int8 KV cache: both cards; tonight |
| `r27.prefix` | default, gpu | SKIP | **a** | the prefix cache on two cards bitwise one card (spec 7's agentic sequences a-f): agentic sessions on --parallel pp; tonight |
| `r27.prefix_kv8` | default, gpu | SKIP | **a** | the same over the int8 KV cache; tonight |
| `r27.serve` | default, gpu | SKIP | **a** | b70-serve --parallel pp start-up at 262144 with MTP / int8 KV: the long-context two-card serve; tonight |
| `r27.golden` | default, gpu | SKIP | **a** | one request end to end through b70-serve --parallel pp = one-card b70-decode, plain and MTP; tonight |
| `r27.agnes` | default, gpu | SKIP | **a** | the same on Agnes (72 layers); cheap; tonight |
| `r27.ornith` | default, gpu | SKIP | **a** | the same on Ornith (MoE head on device 1); cheap; tonight |
| `r27.p3_mtp` | optin, gpu | PASS | **a-done** | passkey at ~250k with --parallel pp --spec auto: PASS |
| `r27.benchy` | optin, gpu-self | never run | **b** | one card vs --parallel pp llama-benchy rows |

### Row 28 - spec 20e - Kolibri-1 served: its template and tokenizer, reasoning and hermes JSON tool calls, b70-serve on tw

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r28.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r28.real` | default, gpu | FAIL | **a** | Kolibri served on two cards: snapshot, start-up at 262144, golden_server --chat; FAILed on its last run; tonight; both cards |
| `r28.passkey` | optin, gpu | SKIP | **a** | **Kolibri passkey at 260000 ids on two cards**: long context; its data now exists (it SKIPped before spec 20b); both cards, hours |
| `r28.a4` | optin, gpu | SKIP | **a** | **Kolibri A4 (German set) against the bf16 source's reference**. Blocked: oracle-out-kolibri-a4 has no references yet - a CPU job on Aleph-Alpha/Kolibri-1-BF16 (156 GB download; tools/toolcall/a4_ref.sh kolibri ref) under the memory cap; then both cards |
| `r28.benchy` | optin, gpu-self | never run | **b** | Kolibri llama-benchy rows on two cards |

### Row 29 - spec 19a Task 4 - DFlash's verify cost at M = 5..8 and one block's draft cost (plan 19a Task 4; box-day plan S

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r29.cost` | optin, gpu | never run | **c** | DFlash verify at M = 5..8 on the old GEMV: superseded by spec 28b's small-batch GEMM, whose verify(M) table is r69.speed. No code (probe_mtp_steps stays) |

### Row 30 - spec 21a - Qwen3.8-Flash-Next (qwen4_exp): the facts, the CPU reference (F1), golden sets, routing traces (pla

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r30.ppl` | optin, cpu | PASS | **a-done** | Flash-Next reference perplexity: PASS |
| `r30.hfcheck` | optin, cpu | PASS | **a-done** | the port's QSA indexer bitwise transformers': PASS |
| `r30.golden` | optin, cpu | PASS | **a-done** | Flash-Next golden sets: PASS (to be remade with the block scores - open-work's Waiting row; that is a re-run of this stage) |
| `r30.golden_layers` | optin, cpu | PASS | **a-done** | the --layers 4 / 18 golden sets: PASS (same remake) |
| `r30.traces` | optin, cpu | never run | **c** | routing traces for spec 22's expert-offload probe (dropped). Code: qwen4exp_oracle.sh's trace mode and its dumper go with it if nothing else reads them |

### Row 31 - spec 21b - Qwen3.8-Flash-Next (qwen4_exp): the descriptor, the loader (both checkpoint forms), the PLE int8 ta

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r31.synth` | optin, cpu | PASS | **a-done** | the synthetic real-width checkpoints: PASS |
| `r31.ple_convert` | optin, cpu | PASS | **a-done** | the int8 PLE file: PASS |
| `r31.ple_kl` | optin, cpu | never run | **a** | the int8 PLE table's KL against the bf16 table (quality evidence for the only PLE form the engine reads). Blocked: plan 21q's evaluate.py was never written and the 360 GB bf16 original is not on the box - operator: build 21q Task 3 or accept the int8 PLE on the golden gates alone |

### Row 32 - spec 21c - Qwen3.8-Flash-Next (qwen4_exp) decode: hyper-connections, QSA, the PLE gather from host USM, the 51

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r32.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r32.oracle_synth` | optin, cpu | PASS | **a-done** | the Flash-Next synthetic golden sets: PASS |
| `r32.k3` | default, card | RUNNING | **a** | Flash-Next decode on the synthetics (plan == allocation, 75 launches, bitwise twice); running tonight; one card |
| `r32.golden` | default, card | SKIP | **a** | the synthetic golden gates (tie-aware, routing diagnostic); tonight; one card |
| `r32.partial` | default, gpu | SKIP | **a** | Intel's real checkpoint at --layers 18 against the full reference; tonight (FAILs until the golden sets are remade - Waiting row); one card |
| `r32.pp` | default, gpu | SKIP | **a** | two cards bitwise one card on the synthetic; tonight |
| `r32.cli` | default, gpu | SKIP | **a** | b70-decode one card vs --parallel pp on the synthetic; tonight |
| `r32.ple_rate` | optin, gpu | never run | **c** | the PLE gather from host USM: spec 22 P0's first number (expert offload, dropped). Code: qwen4exp_kernels_test's --ple-rate mode can go |
| `r32.bench_attn` | optin, gpu | never run | **b** | decision 10's evidence: one sparse kernel at every depth vs attn_v2 below 2052 (timed) |
| `r32.speed` | optin, gpu | never run | **b** | Flash-Next decode on what fits, flash / eager: list B (Flash-Next eager attention) |
| `r32.sweeps` | manual | never run | **c** | a decided probe's sweeps. Manual: no code |

### Row 33 - spec 21d - Qwen3.8-Flash-Next (qwen4_exp) prefill: dense flash to 2050, the indexer and the sparse flash past 

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r33.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r33.prefill` | default, card | SKIP | **a** | Flash-Next prefill on the synthetics (launch counts, bitwise chunks); tonight; one card |
| `r33.golden` | default, card | SKIP | **a** | the synthetic prefill golden gates; tonight; one card |
| `r33.golden_real` | default, gpu | SKIP | **a** | the prefill gate on Intel's checkpoint at --layers 18; tonight (FAILs until the golden sets are remade); one card |
| `r33.pp` | default, gpu | SKIP | **a** | two-card prefill bitwise one card on the synthetic; tonight |
| `r33.cli` | default, gpu | SKIP | **a** | the prefill CLI on the synthetic, one card vs pp; tonight |
| `r33.speed` | optin, gpu | never run | **b** | Flash-Next prefill + eager arm: list B |
| `r33.p0` | manual | never run | **c** | a decided probe's profile and recorded levers. Manual: no code |

### Row 34 - spec 21e - Qwen3.8-Flash-Next (qwen4_exp) served truncated: b70-serve over Qwen4ExpEngine, prefix-cache snapsh

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r34.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r34.snapshot` | default, gpu | SKIP | **a** | Flash-Next prefix-cache snapshots bitwise a cold run (agentic restore); tonight; one card |
| `r34.mtp` | default, gpu | SKIP | **a** | the Flash-Next MTP head: M2 verify bitwise, M3 greedy lossless; tonight; its M1 part needs r34.fixture |
| `r34.serve` | default, card | SKIP | **a** | b70-serve on the synthetic = b70-decode, plain and MTP; tonight |
| `r34.pp` | default, gpu | SKIP | **a** | snapshots and MTP on --parallel pp, cross-layout restores; tonight; both cards |
| `r34.fixture` | optin, cpu | never run | **a** | M1's data: the MTP head against 21a's port (CPU, the oracle image); without it r34.mtp's M1 SKIPs. CPU, under the memory cap |
| `r34.accept` | optin, cpu | never run | **b** | decision 4's MTP acceptance on the CPU reference: feeds list B's Flash-Next MTP norm choice (CPU only, can run any night) |
| `r34.a4_ref` | optin, cpu | never run | **a** | **Flash-Next's A4 reference** on Intel's checkpoint (11 of 36 done in ~/b70/trees/cpu-refs, the chain died 2026-11-06). Note: no stage scores the engine against it yet (spec 22's run was to) - an engine A4 stage is needed for it to prove anything. CPU, under the memory cap |

### Row 35 - the W4A4 probe (docs/07 §3): int4 weights x int4 activations at group 256 after h8's runtime Hadamard - the ac

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r35.grid` | optin, cpu | PASS | **c** | the W4A4 accuracy grid: ran (PASS); what is left is the verdict table, written on the Mac from its output (plan Task 2 Step 3). Code stays until that table exists |
| `r35.per_class` | optin, cpu | PASS | **c** | its per-class grid: ran (PASS); same |

### Row 36 - spec 22a - the expert-offload probe (P0): the links, copy and zero-copy rates, replayed visibility, host RAM u

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r36.curve` | default, cpu | SKIP | **c** | spec 22 (expert offload) dropped. Row 36 goes whole: probe_offload + its two kernels, mk_offload_kernel.py, the oracle's curve / reap / prune / project modes, and the 12 PASSed r36 default stages (offload_links.sh stays: r71.facts uses it; expert_budget stays if the planner keeps it) |
| `r36.reap` | default, cpu | SKIP | **c** | spec 22 dropped (as r36.curve) |
| `r36.prune` | optin, cpu | never run | **c** | spec 22 dropped (as r36.curve) |
| `r36.prune_a4` | optin, cpu | never run | **c** | spec 22 dropped (as r36.curve) |
| `r36.project` | default, cpu | SKIP | **c** | spec 22 dropped (as r36.curve) |

### Row 41 - the llama-benchy comparison matrix (operator, 2026-10-13): b70-serve and vLLM through one client, one set of f

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r41.ours` | optin, gpu-self | never run | **b** | the llama-benchy matrix A / B / D for b70-serve |
| `r41.vllm` | optin, gpu-self | never run | **b** | the vLLM baseline on Qwen3.8 (Now 4) |
| `r41.vllm_mtp` | optin, gpu-self | never run | **b** | vLLM's MTP recipe on Qwen3.8 (Now 4) |

### Row 54 - spec 19b - the DFlash drafter on the card: loader::load_drafter (z-lab/Qwen3.8-27B-DFlash2 beside Qwen3.8, bf1

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r54.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r54.d1_ref` | default, cpu | FAIL | **a** | D1's reference half: the DFlash drafter's logits against dflash_ref.py on the same taps - the drafter's only check against a reference. FAILed (CPU job); open-work's Waiting row; CPU, under the memory cap |

### Row 55 - spec 19c - DFlash in the decode loop: the taps in the verify lists (runtime::build_verify_taps, M = 1..8 over 

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r55.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r55.tps` | optin, gpu-self | never run | **b** | DFlash speed record (informational) |

### Row 56 - spec 19d - DFlash in the server: the drafter's ring in the prefix cache's snapshots (decision 5 as ruled: the 

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r56.d5_a4` | optin, gpu | never run | **c** | D5 measured by hand (docs/BENCHMARKS.md: dflash / mtp A4 1.033, golden 0.791): DFlash stays opt-in behind --spec auto = MTP. No code |
| `r56.d5_dv` | optin, gpu | never run | **b** | DFlash with a draft vocabulary: measured once (ranked 1.043 A4); list B's draft-vocabulary ruling covers it |
| `r56.d5_benchy` | optin, gpu-self | never run | **c** | D5's llama-benchy rows measured by hand (dflash 0.82-0.88 x mtp): decided. No code |
| `r56.d5_opencode` | optin, gpu-self | never run | **b** | D5 proper (the bar) on an opencode recording. Blocked: no recording exists |
| `r56.d5_pp` | optin, gpu | never run | **c** | D5 on two cards measured by hand (A4 1.033, golden 0.793): decided. No code |

### Row 57 - spec 25a - Nemotron 3.5 Bolt: the facts, the CPU reference (B1), golden sets, the A4 references, the tap dumps

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r57.host` | default, cpu | SKIP | **a** | Bolt's reference tests in the box image (the bitwise claims re-checked there); CPU, minutes; tonight |
| `r57.ppl` | optin, cpu | never run | **a** | the Bolt reference's perplexity, int4 and BF16: a sanity check of the reference everything else is judged by; CPU, BF16 present |
| `r57.hfcheck` | optin, cpu | never run | **a** | the port bitwise its restated chain on real weights: proves the CPU reference; CPU |
| `r57.golden` | optin, cpu | RUNNING | **a** | **the Bolt golden sets on our g64 export** (the Bolt reference set; died in the 2026-11-06 crash, shown RUNNING): the real-checkpoint Bolt gates wait on it; CPU hours, Bolt bf16 hot, under the memory cap |
| `r57.golden_bf16` | optin, cpu | never run | **a** | the golden sets on the BF16 original (gate Q's other side); CPU hours, BF16 hot |
| `r57.a4_ref` | optin, cpu | never run | **a** | **Bolt's A4 reference on the int4 form** (days, resumable): Bolt's tool-call quality reference; CPU, under the memory cap |
| `r57.a4_bf16` | optin, cpu | never run | **a** | the A4 reference on the BF16 original (days): scores our quantisation against the true model |
| `r57.gateq` | optin, cpu | never run | **a** | gate Q: our g64 vs BF16 (KL / top-1 / A4) - 'quality not lowered' for our own quantisation; CPU, after the two BF16 runs |

### Row 59 - spec 25b - Nemotron 3.5 Bolt: BoltDesc, load_bolt (g128-expanded and native g64 into one layout, the '+:' / BF

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r59.pytest` | default, cpu | SKIP | **a** | test_bolt_synth.py in the oracle image; CPU, minutes; tonight |
| `r59.synth_host` | default, cpu | SKIP | **a** | the synthetic checkpoints' names and repacks host-side; CPU; tonight |
| `r59.load_real` | optin, gpu | never run | **a** | our Bolt export loaded on two cards at 262144: plan, 0 unconsumed, load time and RSS; both cards, minutes |

### Row 60 - spec 25c - Nemotron 3.5 Bolt decode: P0 (653 / 533 skeletons, GEMV and SSM-state floors, the S sweep), the nb_

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r60.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r60.k1` | default, card | SKIP | **a** | every Bolt decode kernel bitwise its reference (B2); one card; tonight |
| `r60.synth` | default, card | SKIP | **a** | Bolt decode on the synthetics, bitwise twice, every arm; tonight |
| `r60.synth_b3` | default, card | SKIP | **a** | the synthetic golden gate (B3); tonight |
| `r60.synth_pp` | default, gpu | SKIP | **a** | two cards bitwise one card on the synthetic; tonight |
| `r60.golden` | default, gpu | SKIP | **a** | **our g64 export, two cards: B3 short / 4k / agentic**; tonight (the real reference set is r57.golden) |
| `r60.golden32k` | optin, gpu | never run | **a** | B3 at 32k ids on the real export: Bolt long context (hours); both cards |
| `r60.fuse` | default, gpu | SKIP | **a** | every fused arm bitwise 'none' on the real checkpoint; tonight. Its A/B timing half is list B's (Bolt's fused kernels) |
| `r60.state_arm` | optin, gpu | never run | **b** | the bf16 SSM state arm vs fp32: list B (Bolt's bf16 state) |
| `r60.speed` | optin, gpu | never run | **b** | Bolt decode at 886-262000, eager / fused arms |

### Row 61 - spec 25d - Nemotron 3.5 Bolt prefill on l0: absolute SSD sub-chunks with the fp32 state, the conv seed / write

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r61.k0` | default, cpu | SKIP | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r61.k1` | default, card | SKIP | **a** | every Bolt prefill kernel against its reference (B2); one card; tonight |
| `r61.synth` | default, card | SKIP | **a** | Bolt prefill on the synthetics; tonight |
| `r61.synth_b3` | default, card | SKIP | **a** | B3 on prefill, synthetics; tonight |
| `r61.synth_pp` | default, gpu | SKIP | **a** | two-card prefill bitwise one card; tonight |
| `r61.golden` | default, gpu | SKIP | **a** | **our export, two cards: prefill + B3 after it**; tonight |
| `r61.golden32k` | optin, gpu | never run | **a** | B3 after a 32k prefill: Bolt long context; both cards |
| `r61.speed` | optin, gpu | never run | **b** | Bolt prefill rate beside the published vLLM table |

### Row 62 - spec 25e - Nemotron 3.5 Bolt served: the template / tokenizer identities, ultra_v3 reasoning over Qwen XML cal

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r62.snap_synth` | default, gpu | SKIP | **a** | Bolt snapshots bitwise a cold run (agentic restore), incl. cross one / two cards; tonight |
| `r62.serve` | default, gpu | SKIP | **a** | b70-serve on the Bolt synthetic: start-up line and requests; tonight |
| `r62.chat_synth` | default, gpu | SKIP | **a** | greedy chat through b70-serve = b70-decode; tonight |
| `r62.a4` | optin, gpu | never run | **a** | **Bolt A4 through b70-serve against the int4 (and BF16) reference**: Bolt's tool-call quality. After r57.a4_ref; both cards |
| `r62.agentic` | optin, gpu | never run | **a** | the agentic transcript as a live loop with the prefix cache: hit rate and template re-render turns - agentic correctness; both cards |
| `r62.benchy` | optin, gpu-self | never run | **b** | Bolt llama-benchy beside vLLM XPU |

### Row 67 - the prefill memory-bandwidth profile: which kernels reach ~94 % of the card's 638 GB/s (the operator saw 600+ 

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r67.profile` | optin, gpu | PASS | **c** | a diagnostic (gates nothing): ran (PASS). No code beyond the stage |

### Row 63 - spec 26a - LiLiCorr P0: the verify / block costs (probe_draft_cost --shapes bolt-lilicorr) and the projection'

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r63.k0` | default, cpu | never run | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r63.host` | default, cpu | never run | **a** | the P0 projection and expert-union host tests; CPU; tonight |
| `r63.cost` | default, gpu | never run | **b** | Bolt verify per M and one LiLiCorr block per format (timed): feeds the draft-format and cost-table defaults. Default, so it runs tonight, but only a quiet-box number binds |

### Row 64 - spec 26b - LiLiCorr on the card: DFlashForm::LiLiCorr, NVFP4 decoded exactly into exact / int8 / int4 g64, the

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r64.k1` | default, card | FAIL | **a** | lilicorr_kernels_test (FAIL 11-04; the near-zero judging fixed 11-06, then PASS by hand); tonight; one card |
| `r64.draft` | default, card | SKIP | **a** | lilicorr_draft_test: the int8 arm passes, the bf16 'exact' arm fails its log-prob bar (5.67e-3 vs 1e-3) - see docs/probes/lilicorr-exact-divergence-2026-11-07.md; one card |

### Row 65 - spec 26c - LiLiCorr in Bolt's loop and server: verify M <= 8 with taps on two cards, Mamba state slots and the

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r65.k0` | default, cpu | never run | **c** | a recap of G0 plus three host name tests that `x` runs anyway. Only the registry lines go |
| `r65.host` | default, cpu | never run | **a** | Bolt spec host tests (D2 / D4 over a fake engine); CPU; tonight |
| `r65.reject` | default, gpu | never run | **a** | the --spec refusals before the device; tonight |
| `r65.d3` | default, gpu | never run | **a** | Bolt verify lists bitwise the plain steps, the union MoE; tonight; both cards |
| `r65.d2` | default, gpu | never run | **a** | **--spec dflash greedy = --spec off at every K on the agentic transcript**, per draft format; tonight; both cards. Also the acceptance that picks the draft format |
| `r65.prefill` | default, gpu | never run | **a** | the drafter's ring after prefill, bitwise across chunkings; tonight |
| `r65.prefix` | default, gpu | never run | **a** | restores with the drafter bitwise the cold run (agentic sessions); tonight |
| `r65.serve` | default, gpu | never run | **a** | **the 36 A4 scenarios through b70-serve <bolt>: off / dflash / lookup identical**; tonight; both cards |
| `r65.benchy` | optin, gpu-self | never run | **b** | D5: Bolt's --spec auto verdict (dflash vs lookup vs off) by llama-benchy + the agentic loop |

### Row 66 - spec 13a - the batching probe: rows of different sequences in one decode step (the BATCH_ROWS row builds, runt

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r66.check` | default, gpu | SKIP | **b** | the 13a batching probe's correctness check (row builds bitwise per slot): run first in the quiet session, before r66.cost; drop its Agnes arms (spec 13 §11: Qwen3.8 only) |
| `r66.cost` | optin, gpu | SKIP | **b** | the batching step table (Now 4: batching speed); Qwen3.8 arms only |
| `r66.chunk` | optin, gpu | SKIP | **b** | the interleave chunk cost; Qwen3.8 arms only |

### Row 68 - spec 28a - the small-M GEMM P0: Qwen3.8's decode linears at M = 1 / 2 / 4 / 8 / 16 (gemv vs bf16 / i8xi4 DPAS 

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r68.k0` | default, cpu | never run | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r68.host` | default, cpu | never run | **c** | gemm_sm_ref_test - r69.host runs it too. Only the registry lines go |
| `r68.build` | default, cpu | never run | **b** | builds probe_small_m / probe_verify_m, which r69.speed needs: keep as (or fold into) r69.speed's build step |
| `r68.check` | default, card | never run | **c** | 28a's stand-in kernels checked: the probe is decided (GO; spec 28b is in production and r69.kernels checks the real kernels). The 347 psm_* stand-ins stay only for r69.speed's `prod` lever table |
| `r68.shapes` | default, gpu | never run | **c** | 28a's busy-box shapes table: superseded by r69.speed on the production kernels |
| `r68.attrib` | default, gpu | never run | **c** | probe_verify_m --verify8 for the GO line: decided; r69.speed runs it on production |
| `r68.numerics` | default, card | never run | **c** | N2 on the stand-ins: decided; r69.kernels / r69.gates judge the production kernels |
| `r68.verdict` | manual | never run | **c** | spec 28's GO line: made (28b built and merged). Manual: no code |

### Row 69 - spec 28b - gemm_sm in production: every M >= 2 decode list (MTP / lookup verify, DFlash tap verify - the draft

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r69.k0` | default, cpu | never run | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r69.host` | default, cpu | never run | **a** | gemv_kind / gemm_sm names / ref / greedy tie / AdaptiveK cost tests; CPU; tonight |
| `r69.kernels` | default, card | never run | **a** | gemm_sm_test against fp64 and gemv: every M >= 2 decode list runs on it; one card; tonight |
| `r69.gates` | default, card | never run | **a** | **MTP / DFlash verify through gemm_sm judged by the f32 judge (golden + A4)**: the speculative paths' correctness; one card; tonight |
| `r69.forced_dumps` | optin, card | never run | **a-done** | the Qwen3.8 spec tests' forced contexts: made by hand 2026-11-04 (286 contexts); re-run only when the contexts change |
| `r69.forced_refs` | optin, cpu | never run | **a-done** | their f32 judges: made 2026-11-04 (open-work 'Closed recently'); same |
| `r69.speed` | optin, gpu | never run | **b** | verify(M) M 1..8 on gemm_sm vs forced gemv, the lever table |
| `r69.d1` | optin, gpu-self | never run | **b** | MTP greedy D1 speed at 4096 against 1.633x |

### Row 70 - spec 19f - prompt lookup inside DFlash (--spec dflash-lookup: ENTER >= 8 tokens beyond the drafter's 2048 wind

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r70.k0` | default, cpu | never run | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r70.host` | default, cpu | never run | **a** | lookup_in_dflash and mock serving tests; CPU; tonight |
| `r70.d2` | default, card | never run | **a** | --spec dflash-lookup = --spec off on golden + A4 + the edit / copy set (tie rule); one card; tonight |
| `r70.speed` | optin, gpu | never run | **b** | dflash-lookup vs dflash / auto / off per depth |
| `r70.verdict` | manual | never run | **b** | spec 19 §13.1's bar read from r70.speed (replace / opt-in / drop): decided after the quiet session. Manual: no code |

### Row 75 - plan 5d - prefill chunk 4096: the int8 linears at 4096 rows (P0), the chunk cap through the planner, the gates

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r75.k0` | default, cpu | never run | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r75.host` | default, cpu | never run | **a** | buffers_test, the planner's prefill term, refusals, prefix hooks at cap 4096; CPU; tonight |
| `r75.p0` | default, gpu | never run | **c** | probe_chunk_rows at M 4096 vs 2 x 2048: the P0 is decided (5d built and merged; +4.2 % measured). No code beyond the stage (probe_chunk_rows can go when r75.speed decides) |
| `r75.gates` | default, card | never run | **a** | prefill gates / split / determinism at cap 4096; tonight; one card. Its A4 half SKIPs without A4_REF_DIR - the same Qwen3.8 A4 reference as r11.a4 |
| `r75.pp` | default, gpu | never run | **a** | --parallel pp prefill at cap 4096 bitwise; tonight; both cards |
| `r75.speed` | optin, gpu | never run | **b** | pp4096-65536 at chunk 4096 vs 2048: the 5d default decision (Now 4) |

### Row 71 - spec 17a - the TP P0, Qwen3.8 first: links / BARs / peer access, CPU stores into VRAM through the BAR, the all

| stage | kind | last result | verdict | reason |
|---|---|---|---|---|
| `r71.k0` | default, cpu | never run | **c** | a recap of G0 (`need_pass g0.*`); the nightly run compares main with itself, so 'every pre-existing binary identical' says nothing - G0 and the row's own stages are the evidence; only the registry lines go |
| `r71.host` | default, cpu | never run | **a** | tp_shapes_test / tp_crossing_protocol_test: host tests 17b will build on; CPU; tonight |
| `r71.build` | default, cpu | never run | **c** | probe_tp's build for the opt-in arms below: 17a is decided (GO, quiet box 2026-11-06, docs/probes/probe-tp-2026-11-01.md §8). probe_tp's code stays until 17b decides (its protocol / shapes are 17b's start) |
| `r71.facts` | optin, gpu | never run | **c** | 17a decided (GO); measured by hand 2026-11-01 / 11-06 |
| `r71.bar` | optin, gpu | never run | **c** | 17a decided (as r71.facts) |
| `r71.ar` | optin, gpu | never run | **c** | 17a decided: the P2P push bitwise over 10^4 replays, 3.6 us a call (open-work Closed) |
| `r71.step` | optin, card | never run | **c** | 17a decided: per-card linears 0.550 of one card's |
| `r71.verdict` | manual | never run | **c** | spec 17's GO line: made. Manual: no code |

## Totals

| selection | stages | a | a-done | b | c |
|---|---|---|---|---|---|
| default, never PASSed | 113 | 65 | 0 | 7 | 41 |
| opt-in | 106 | 21 | 24 | 43 | 18 |
| manual | 17 | 2 | 1 | 1 | 13 |
| **all** | **236** | **88** | **25** | **51** | **72** |

Of the 88 a's: 65 are in tonight's default set, 3 are blocked on data nobody has made (`r11.a4`, `r28.a4`,
`r31.ple_kl`; the b's `r2.opencode`, `r12.d5` and `r56.d5_opencode` wait on an opencode recording), the rest are
CPU references and the card runs that follow them.
Of the 72 c's: 31 G0 recaps, 5 row-36 stages (spec 22), 2 'one format' checkpoints, and decided probes.

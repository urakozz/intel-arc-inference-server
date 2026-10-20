# Spec 25q - Nemotron 3.5 Bolt: our AutoRound int4 g64 (the recipe the operator runs), RTN g64 first, F6

**Status (2026-10-20): planned; nothing built.** Box queue row **58**. Parallel to 25b once 25a's reference exists. The
operator runs the long stages himself ("I'll quant it later as g64 with coding dataset and more iterations"): this plan
delivers the script, the recipe, its tiny end to end and the evaluation; the `tune` stage is his box days.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the one weight format for Bolt made by us - AutoRound W4A16 int4 **g64** symmetric, GPTQ-packed, calibrated on
agentic coding through Bolt's own template, with more iterations and samples than SergiioB's g128 (200 iters, 128 samples
of 1536, generic data) - as one resumable command on 24q's pattern; RTN g64 from the BF16 original first (F6's baseline and
a native-g64 development checkpoint for 25b-25d); F6's table (RTN g64, SergiioB's g128 and ours against the BF16 original:
held-out KL / top-1 per set, A4); the export accepted by 25b's loader with the **same names and layout** as SergiioB's.

**Architecture:** spec 25 §5, §9 B5, §12 decisions 3, 6, 7, 8, 9, 20. `tools/quantize_bolt.sh` (stages `calib`,
`coverage`, `rtn`, `tune`, `eval`, `check`, `card`; `--only` / `--from` / `--dry-run` / `--list`; `$OUT/.done-<stage>`),
modules in `tools/quantize/bolt/`. AutoRound at a pin **at or after intel/auto-round `ec10a649`** (#2512: the Nemotron-H
Omni wrapper in plain AutoRound calls - the trunk rename `language_model.backbone` -> `language_model.model`, the remote
puzzle class preferred, `A_log` / `D` / `dt_bias` / `e_score_correction_bias` restored in fp32 after a low-precision load,
the gated norm's `group_size` restored, the disk-streamed load's chained renames) and with `e31b7728`'s generic dataset
loader. **The calibration forward uses 25a's training-kernel ops** (`tools/oracle/bolt_mamba_ops.py`: no dt clamp, fp32
SSM internals, a memory-lean chunk scan) installed as the `mamba_ssm` stub - so the block outputs AutoRound fits are the
model's real arithmetic (SergiioB's export ran the remote torch path, whose 0.001 dt clamp spec 25 §2.0 rules out, and he
had to replace four lines with einsums to fit memory).

**Tech Stack:** bash; Python 3.11+, torch XPU build (the box's `~/auto-round/.venv`), transformers 5.19.0 (25a's pin),
AutoRound at the Bolt pin; vLLM's XPU image for self-generated answers (`GEN_BACKEND`); the box (both B70s for `tune`,
44 threads, ~121 GB RAM).

**Spec:** `docs/superpowers/specs/2026-10-20-spec25-nemotron-bolt-design.md` (§1.3-1.5, §5, §9 B5, §12). Precedents: plan
24q task for task, `tools/quantize_lfm2moe.sh` / `tools/quantize_kolibri1.sh` (the script shape), `tools/quantize/ar_pin.sh`,
`tools/quantize/check_gptq_export.py`, SergiioB's process (`README.md` of his repo: AutoRound `main` 08d7085e + #2512,
`scheme W4A16, group_size 128, sym, iters 200, nsamples 128, seqlen 1536`, disk-streamed load, `low_gpu_mem_usage`,
`device_map 0,1`, **5 h 57 min** on two B70s - published).

## Dependencies and branch points

- **Plan 25a merged:** `bolt_ref.py` (`expected_names` "ar-g64" and "gptq-g128", `build_streamed(weights="int4")`),
  `bolt_mamba_ops.py`, `bolt_make_tiny.py` (`pattern12`, `int4-g128-gptq`), the A4 set and both A4 references, the BF16
  golden sets (`oracle-out-bolt-bf16`), `bolt_oracle.sh export`.
- **The AutoRound pin for Bolt** (Task 1): `ar_pin.sh` gains an opt-in `AR_PIN=bolt` that selects a second commit (the
  first commit at or after `ec10a649` on which Task 1's tiny run passes) - the Qwen / LFM2 / Kolibri scripts keep 0.17.0 @
  `6afaecdb` byte for byte. Whether `auto-round-local.patch`'s two fixes are upstream at the new commit is recorded; if not,
  the patch is re-based (`tools/quantize/auto-round-local-bolt.patch`).
- **Decision 7 (answers):** `GEN_BACKEND=vllm` serves SergiioB's g128 on the box's vLLM XPU image at TP 2 (his recipe's
  flags, `b70/patch_xpu_moe_topk_pad.py` applied - the image needs it for top-9) for the self-generated agentic sessions;
  `GEN_BACKEND=hf` (transformers on the box CPU, the g128 dequantised) is the slow fallback; the public slice is
  `openbmb/UltraData-SFT-Agent-2609` (Code-Agent, Tool-Use) through `e31b7728`'s loader **only after its licence is checked
  and recorded** (else the slice is dropped and the mix printed without it).
- **Decision 8 (budget):** defaults `ITERS` 400, `NSAMPLES` 512, `SEQLEN` 2048; `ITERS=800` is the arm.
- **If AutoRound cannot run Bolt at the pin** (Task 1 fails and one local patch does not fix it): the RTN g64 export ships as
  "ours" for development, F6 runs on it, and the operator decides. No third-party format, no g128 re-export.

## Global Constraints

- Branch `spec25q-bolt-quant` from main; signed commits; no merge, no push. Never commit weights, tokens, generated
  calibration text or `tools/box.env`.
- **Not overridable:** W4A16, `group_size` 64, symmetric, GPTQ packing (`auto_round:auto_gptq` or `gptq` - Task 1 records
  which the export writes; 25b reads both), model dtype bf16, no `--quant_lm_head`, `quant_nontext_module` false (the vision
  tower stays bf16 **in** the export, for name parity with SergiioB's and so vLLM can serve it for comparisons; our loader
  skips it by name). **Never quantised:** `mixer.gate.weight` (the routers), `e_score_correction_bias`, `conv1d`, every norm,
  `A_log` / `D` / `dt_bias`, `embeddings`, `lm_head`. The check refuses an export with any of them packed, a non-F32
  `e_score_correction_bias` / `A_log` / `D` / `dt_bias`, a scale that is not F16 (Decision 9), a non-finite or zero scale, a
  `qzeros` word other than 0x77777777, a `g_idx` other than `i // 64` (or absent), or a quantised tensor outside the set.
- One format: no asymmetric, no g128, no compressed-tensors, no `--model_free`.
- Box: `tune` holds `flock ~/b70-gpu.lock` for its whole run (both cards; **~16-40 h** - ESTIMATED in Task 3's dry run) - box
  days without other GPU rows; CPU stages check `MemAvailable` (the BF16 original is ~150 GB against ~121 GB of RAM: every
  stage reads it disk-streamed or shard by shard; `tune` refuses below 64 GB free, ESTIMATED); detached, polled; `-j44`.
- Every number measured, or marked derived / estimated / proposed / published; the card says which.
- No `rm -rf`.

## Review Focus

1. **The quantised set is exactly the language model's `nn.Linear`s** (27,920: 27,648 expert linears, 80 shared-expert
   linears, 80 latent projections, 80 Mamba in / out, 32 attention): the dry run prints the resolved set; `check.py` lists
   every packed tensor against `bolt_ref.expected_names(cfg, "ar-g64")` and refuses anything else.
2. **The calibration forward is the model's arithmetic.** AutoRound's block outputs come from the remote code with 25a's ops
   installed (no clamp); a test on the tiny model shows the block output with and without the stub differs on `dtclamp`, and
   equals `bolt_ref`'s with it.
3. **Every expert sees calibration tokens.** 240-496 experts a layer, top-9: `coverage` counts routed tokens per (layer,
   expert) with the g128 model's routes (or the BF16's through 25a's recorder) on the calibration rows and fails under
   `COVERAGE_FLOOR` (2048, PROPOSED) after `COVERAGE_ROUNDS` of targeted top-ups; whether AutoRound forces all experts
   during calibration (`force_all_experts_routing_enabled`) is recorded with the setting used.
4. **The calibration is the target workload**, and disjoint from the held-out sets (25a's agentic transcript, code, prose,
   the A4 scenarios: a hash check).
5. **F6 is like for like:** RTN g64, SergiioB's g128 and ours scored against the same BF16 reference rows (25a's, eager,
   the same tokens) and the same A4 budget.

---

### Task 1: the toolchain on the tiny model (Mac or box CPU)

**Files:**
- Create: `tools/quantize/bolt/autoround_run.py`
- Modify: `tools/quantize/ar_pin.sh` (the opt-in `AR_PIN=bolt`; the default path unchanged - `test_dry_run` of the existing
  scripts proves it), `tools/quantize/README.md` (a "Nemotron 3.5 Bolt" section)

- [ ] **Step 1:** in a venv with torch, transformers 5.19.0 and AutoRound at the Bolt pin, quantise 25a's `pattern12` tiny
  (wrapped as an Omni config so #2512's paths run) through `autoround_run.py` (the Python API: `AutoRound(model, tokenizer,
  scheme="W4A16", group_size=64, sym=True, iters=..., nsamples=8, seqlen=256, device_map=DEVICE, low_gpu_mem_usage=True)`,
  the `mamba_ssm` stub from `bolt_mamba_ops`, then `save_quantized(format="auto_round:auto_gptq")`) at `--iters 0` and
  `--iters 2`. Record in the README section, each with the file:line it rests on: (a) the export's tensor names (=
  `language_model.model.*`? the vision names?); (b) the packed set == Review Focus 1's; the kept tensors bitwise the source
  (bf16) or F32 as listed; (c) `quantization_config` (bits, group 64, sym, packing format, `dynamic` / `extra_config`; any
  `+:` rule) and whether `loader::QuantConfig::parse` + spec 25 §5.2's relaxations accept it; (d) the scale dtype of every
  linear in the export (#2512 pins `mixer.out_proj` to bf16 **in memory**; SergiioB's GPTQ export wrote F16) - Decision 9;
  (e) `force_all_experts_routing_enabled`; (f) whether `auto-round-local.patch`'s fixes are upstream at the pin.
- [ ] **Step 2:** if any of (a)-(d) fails: one local patch at most (`tools/quantize/auto-round-bolt.patch`, its reason
  beside it); else stop and report - the RTN path ships for development.
- [ ] **Step 3: commit** `git commit -S -m "quantize: AutoRound at the Bolt pin (#2512) on a tiny puzzle, the training-kernel ops in the calibration forward - names, the packed set, scale dtypes (spec 25q)"`.

### Task 2: the RTN g64 export and the check (no AutoRound in the path)

**Files:**
- Create: `tools/quantize/bolt/rtn_export.py`, `tools/quantize/bolt/check.py`, `tools/quantize/bolt/test_bolt_quant.py`

**Interfaces:**
- Produces: `rtn_export.py <BF16 snapshot> <out> [--signed-scales]` - every selected linear by `src/loader/rtn.h`'s rule (or
  AutoRound's signed variant, `--signed-scales`: the group's extreme maps to -8 - the form AutoRound's own RTN writes, so the
  RTN row compares like with like), the names as Task 1 recorded, everything else copied byte for byte (vision included),
  one source shard at a time (peak RAM ~4 GB, ESTIMATED); `quantization_config` as Task 1 recorded plus provenance
  `"rtn": "spec25q rtn_export.py"`. `check.py <export> [--source <BF16 snapshot>]`: names against `expected_names`, dtypes,
  shapes, `qzeros`, `g_idx`, the group, every scale F16 / finite / non-zero (signed allowed), `quantization_config`, the kept
  tensors bitwise the source (with `--source`), Review Focus 1; prints `ACCEPTED` or the first refusal.
- [ ] **Step 1: the failing tests** (`test_bolt_quant.py`): `test_rtn_rule` (crafted groups: all zero, an f16-rounding
  `2 amax / 15`, a `rint` tie, a negative extreme with `--signed-scales`); `test_rtn_tiny_end_to_end` (`pattern12` ->
  `rtn_export.py` -> `check.py --source` `ACCEPTED`; `bolt_ref.build_streamed(weights="int4")` loads it);
  `test_check_refuses` (a packed router, a bf16 `e_score_correction_bias`, a packed conv, a BF16 scale, a missing expert, a
  g128 group: each refused by name); `test_same_names_as_autoround` (when Task 1's export exists).
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "quantize: Bolt RTN g64 export (rtn.h's rule, AutoRound's names, signed option) and the export check (spec 25q)"`.

### Task 3: the script and its modules

**Files:**
- Create: `tools/quantize_bolt.sh`, `tools/quantize/bolt/{calib,coverage,evaluate,card}.py`, `tools/quantize/bolt/prompts/`
  (original agentic-coding task prompts written for this repo; no third-party text)
- Modify: `tools/quantize/bolt/test_bolt_quant.py`, `tools/quantize/README.md`

**Interfaces:**
- Produces: `tools/quantize_bolt.sh [--only STAGE | --from STAGE] [--dry-run] [--list]`; environment `MODEL` (the BF16
  snapshot), `G128` (SergiioB's snapshot: the answer generator and F6's row), `OUT` (`$HOME/models/bolt-w4g64`), `DEVICES`
  (`0,1`), `NSAMPLES` (512), `SEQLEN` (2048), `ITERS` (400), `COVERAGE_FLOOR` (2048), `COVERAGE_ROUNDS` (4), `GEN_BACKEND`
  (`vllm` | `hf`), `PUBLIC_SLICE` (`ultradata` | `none`), `MIX` (`agentic=60,code=25,chat=15`, PROPOSED), `EVAL_ROWS` (32 per
  held-out set), `VENVPY`, `AR_PIN=bolt`.
- Stages:
  - `calib` (`calib.py`): conversations rendered by Bolt's template (`enable_thinking` true; reasoning kept in the turns
    after the last user message as the template does), Qwen3-Coder XML calls from 25a's A4 tool schemas and new task
    prompts, tool results from this repository's own files; assistant turns generated by `GEN_BACKEND`; the public slice
    rendered through the same template; packed into `NSAMPLES` rows of `SEQLEN`; held-out hashes refused; the mix printed
    with token counts.
  - `coverage` (`coverage.py`): routed tokens per (layer, expert) on the rows (25a's `Recorder`), targeted top-ups, FAIL
    under the floor.
  - `rtn`: `rtn_export.py --signed-scales` (minutes to an hour: ~150 GB read) then `check`.
  - `tune`: AutoRound sign-SGD, `--iters $ITERS --nsamples $NSAMPLES --seqlen $SEQLEN --low_gpu_mem_usage
    --device_map $DEVICES`, the rows pre-tokenised (`AutoRound(dataset=rows)`), the disk-streamed load, the stub installed;
    under the GPU lock; after the first block the honest ESTIMATE (block time x 88) printed and logged.
  - `eval` (`evaluate.py`): F6 - per held-out set KL(BF16 ‖ int4) and top-1 agreement, perplexity, for `rtn`, `g128`
    (SergiioB's) and `export` (ours) through `bolt_ref` (eager, layer-streamed, the BF16 rows from 25a's
    `oracle-out-bolt-bf16`); A4: `oracle_generate.py --model bolt --weights int4` on each against 25a's BF16 A4 reference
    (agreement, parse failures); the int8-head row (KL of the int8 head against the bf16 head with everything else BF16,
    spec 9's rule); writes `$OUT/eval/f6.md` with Decision 6's rule marked pass / fail.
  - `check`: `check.py --source $MODEL` on each export.
  - `card` (`card.py`): base model, the recipe (AutoRound commit, patch, flags, calibration mix and token counts, coverage,
    times), F6's table beside SergiioB's published card numbers, the engine command, the licence (OpenMDW-1.1 by link, the
    derivation notice - Decision 20); refuses to render when F6 fails unless `ARM=force`.
- [ ] **Step 1: the failing tests:** `test_dry_run_prints_every_stage` (the resolved settings, the quantised set, the mix,
  ESTIMATED RAM / disk / time per stage - the tune's ~16-20 h at the defaults and ~32-40 h at `ITERS=800` derived from SergiioB's
  5 h 57 min scaled by iters x seqlen at a fixed batch, plus the per-block caching of NSAMPLES inputs); `test_tiny_end_to_end` (`--tiny <dir>`: calib -> coverage -> rtn -> eval ->
  check -> card on `pattern12` with `GEN_BACKEND=hf`, `ACCEPTED`); `test_heldout_disjoint`; `test_card_licence`.
- [ ] **Step 2: implement** (the header documents each stage as `quantize_lfm2moe.sh`'s does); PASS; `--dry-run` prints
  every stage.
- [ ] **Step 3: commit** `git commit -S -m "quantize: tools/quantize_bolt.sh - agentic calibration through Bolt's template, coverage over 240-496 x 40, RTN then AutoRound g64 on both B70s, F6 against BF16 with SergiioB's g128 as a row (spec 25q)"`.

### Task 4: the box runs (row 58; the operator schedules `tune`)

- [ ] **Step 1: `rtn` first** (`--only rtn`, `--only check`): the native-g64 development checkpoint; then
  `bolt_oracle.sh <data> export $OUT/rtn` (-> `oracle-out-bolt-rtn/`).
- [ ] **Step 2: `calib`, `coverage`, `tune`, `eval`, `check`, `card`** - detached, resumable, `tune` under the GPU lock;
  record the machine, commit, flags, times, peak RAM, token counts, coverage. F6's table and the run record go into 25a's
  facts sheet (a new section) and spec 25 ("25q as built"); Decision 6's rule applied and recorded.
- [ ] **Step 3: publish** (repo name the operator's) **after the operator's go** (the licence is ruled - Decision 20, 2026-10-20; the go is for the upload itself), with the card. Then
  `bolt_oracle.sh export $OUT/export` (-> `oracle-out-bolt-ours/`) so 25c-25e's and 26's `ours` stages have their reference.
- [ ] **Step 4: commit** `git commit -S -m "spec 25q: our Bolt AutoRound g64 - F6, the run record"`.

### Task 5: docs, the box queue row

**Files:**
- Modify: spec 25 ("25q as built"), this plan's status, `docs/superpowers/plans/box-validation-queue.md` (row 58),
  `tools/box_validate/stages.sh` (`row 58`), `tools/box_validate/data.sh` (`have bolt_rtn`, `have bolt_ours`,
  `have oracle_bolt_rtn`, `have oracle_bolt_ours`)

- [ ] **Step 1: the queue row 58** (`python3 tools/box_validate/test_box_validate.py` passes; `--dry-run --only r58`):

```text
row 58 "spec 25q - Nemotron 3.5 Bolt: our int4 g64 - RTN first (native g64 development checkpoint), then AutoRound on both B70s (~1-2 days), F6 vs BF16 with SergiioB's g128 as a row (plan 25q)"
stage r58.host     58 default cpu  oracle_image                    -        "test_bolt_quant.py (RTN rule, tiny end to end, refusals, names = AutoRound's)"
stage r58.rtn      58 default cpu  bolt_bf16                       -        "quantize_bolt.sh --only rtn, --only check: ACCEPTED -> \$DATA/bolt-w4g64/rtn"
stage r58.rtn_ref  58 optin   cpu  bolt_rtn,oracle_image           r58.rtn  "bolt_oracle.sh export rtn: the native-g64 golden sets"
stage r58.tune     58 manual  gpu  bolt_bf16,bolt_g128             r58.rtn  "calib, coverage, tune (AutoRound on devices 0,1, ~16-40 h, holds the GPU lock), check - the operator starts it"
stage r58.eval     58 optin   cpu  bolt_bf16,bolt_a4_ref           r58.tune "F6: KL / top-1 per set and A4 for rtn, g128 and ours; the int8-head row; f6.md"
stage r58.ours_ref 58 optin   cpu  bolt_ours,oracle_image          r58.eval "bolt_oracle.sh export ours -> oracle-out-bolt-ours"
rownote 58 "Publication waits for the operator's go to upload (Decision 20 ruled: OpenMDW-1.1 open weights, publishable). tune: ~16-20 h at ITERS 400 / NSAMPLES 512 / SEQLEN 2048, ~32-40 h at ITERS 800 (ESTIMATED)."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: spec 25 (25q as built), box queue row 58 - our Bolt quantisation"`.

**Gate for the plan:** the tiny end to end `ACCEPTED` (AutoRound's names, the scale dtypes and the stub's effect recorded);
the dry run complete with its estimates; on the box - the RTN export `ACCEPTED` and its golden sets written; after the tune
(the operator's) - F6's full table, Decision 6's rule applied, the export `ACCEPTED`, published only after the operator's
go.

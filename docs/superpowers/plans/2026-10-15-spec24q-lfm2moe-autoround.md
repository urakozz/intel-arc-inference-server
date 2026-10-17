# Spec 24q - LFM2.5-24B-A2B: our AutoRound int4 g64 (the script, the recipe, F6), RTN g64 first

**Status (2026-10-15): planned; nothing built.** Box queue row **49** (rows 42-47 are spec 23's). Parallel to 24b once
24a's reference exists; its `rtn` stage comes first and is the development checkpoint of 24b-24d.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the one weight format for this model - AutoRound 0.17.0 W4A16 int4 g64 symmetric `auto_round:auto_gptq`, re-quantised by us from the bf16 original and calibrated for agentic coding - as one resumable command on Kolibri's pattern (`tools/quantize_kolibri1.sh`: a script plus modules, stage by stage); RTN g64 first, both as F6's baseline row and as the development checkpoint; F6's table (ours and RTN against bf16: held-out KL / top-1 per set and A4); the export accepted by 24b's loader rules; published as `urakozz/LFM2.5-24B-A2B-W4A16-g64-AutoRound-GPTQ` after the operator's go.

**Architecture:** spec 24 §5, §7 F6, §8 24q, §10 decisions 3, 4, 5, 9. `tools/quantize_lfm2moe.sh` (stages `calib`, `coverage`, `rtn`, `tune`, `eval`, `check`, `card`; `--only` / `--from` / `--dry-run` / `--list`; a finished stage leaves `$OUT/.done-<stage>`), modules in `tools/quantize/lfm2moe/`. Model-neutral pieces of `tools/quantize/kolibri/` (the row writer, the routed-token counter, the KL / top-1 evaluator) are imported, not copied; what is this model's - the calibration mix through LFM2's template with pythonic calls, the quantised set, the expert names, the card with the LFM licence's notices - is new. AutoRound comes from `tools/quantize/ar_pin.sh` (0.17.0 @ `6afaecdb` + `auto-round-local.patch`), as the Qwen3.8 scripts install it.

**Tech Stack:** bash; Python 3.11+, torch (XPU build on the box: the `~/auto-round/.venv` the Qwen3.8 scripts use), transformers 5.19.0 (24a's pin), AutoRound 0.17.0; vLLM's XPU image or transformers for self-generated answers (`GEN_BACKEND`); the box (XPU device 1, 44 threads, ~121 GB RAM).

**Spec:** `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` (§1.4, §5, §7 F6, §8 24q, §10 decisions 3-5, 9). Precedents: plan 21q, `tools/quantize_kolibri1.sh` (its header is this script's template), `tools/quantize/kolibri/{calib,coverage,autoround_run,evaluate,check,card,common}.py`, `tools/quantize_qwen38_tuned.sh` (the XPU run), `tools/quantize/check_gptq_export.py`, `tools/quantize/README.md` ("AutoRound version"), docs/13 (`auto_round:auto_gptq` as `loader::QuantConfig::parse` reads it).

## Dependencies and branch points

- **Plan 24a merged:** `lfm2moe_ref.py` (`expected_names` both expert spellings, `build_streamed(weights="int4")` for `eval`), `lfm2moe_make_tiny.py` (`pattern8`), the A4 set (`tests/golden/toolcall-lfm2moe/`), `score.py --format pythonic`, `oracle_generate.py --model lfm2moe`, `lfm2moe_oracle.sh export`.
- **The expert names** are an open fact until Task 1: transformers 5.19.0 fuses `experts.E.{w1,w3,w2}` at load (`conversion_mapping.py:1160-1174`) and AutoRound 0.17.0's generic unfusing names the linears `gate_proj` / `up_proj` / `down_proj` (`auto_round/modeling/fused_moe/moe_experts_interface.py:94-121` at `6afaecdb`; `lfm2_moe` is not in `auto_round/modeling/unfused_moe/__init__.py`'s `MODEL_CONFIG`). Task 1 records what the export carries; `rtn_export.py` writes **the same names** so every checkpoint of this model has one name set; 24b's loader reads both spellings anyway (spec 24 §5).
- **If AutoRound cannot run `lfm2_moe`** (Task 1 fails and no recorded local patch fixes it), the RTN export (`RTN_BACKEND=ours`) is what ships, F6 is run on it alone, and the operator decides (spec 24 §0: RTN g64 is the quick fallback). No third-party format, no engine-side fused path.
- **24b already exists and shipped a placeholder `check.py` / packer (spec 24 "24b as built"), to be merged here, not duplicated.** `tools/quantize/lfm2moe/check.py` and `make_synth.py` were written self-contained in 24b (24q not yet built then), reusing `tools/oracle/lfm2moe_make_tiny.py`'s RTN packer rather than this plan's own. When this plan's Task 1+ build their own `check.py` (naming the same tensors, spec 24 §5's rules) and `rtn_export.py` / packer, **merge the two `check.py`s into one** (this plan's checks are the superset: 24b's crafted-refusal cases - unknown/missing tensor, dtype refusal, tied-head mismatch, g128 refusal - carry forward) and **repoint `tools/quantize/lfm2moe/make_synth.py` at this plan's packer** once it exists, dropping the `lfm2moe_make_tiny` reuse. Leaving two independent `check.py` modules after this plan lands is the failure mode to avoid - a future change to spec 24 §5's rules would need updating twice and could silently drift.
- **Decision 4 (where):** the plan's default is the box's XPU device 1 (`DEVICE=1`, as `quantize_qwen38_tuned.sh`), the bf16 model in host RAM; `DEVICE=cpu` works (slow); a rented CUDA machine (`DEVICE=cuda`) works too.
- **Decision 3 (the quantised set):** every `nn.Linear` (default `QSET=all`); `QSET=no-conv` / `no-attn` / `no-conv-attn` build the bf16 arms if F6 fails its bars.

## Global Constraints

- Branch `spec24q-lfm2moe-quant` from main; signed commits; no merge, no push. Never commit weights, tokens, generated calibration text or `tools/box.env`.
- **Settings that are spec 24 §5 and not overridable:** scheme W4A16, `group_size` 64, symmetric, `packing_format` / `--format` `auto_round:auto_gptq`, model dtype bf16, no `--quant_lm_head` (the head is tied to the embedding; the engine builds its int8 head at load, spec 9). **Never quantised:** `feed_forward.gate.weight` (the router), `feed_forward.expert_bias` (fp32), `conv.conv.weight`, every norm, `embed_tokens`. The check refuses an export with any of them packed, with `expert_bias` not F32, or with a quantised tensor outside the `QSET`'s set.
- One format: no asymmetric, no g128, no compressed-tensors export, no `--model_free`.
- Box: AutoRound on XPU holds `flock ~/b70-gpu.lock` for the whole `tune` stage (hours: a box day without other GPU rows); CPU stages check `MemAvailable` (the `tune` stage refuses below 80 GB, ESTIMATED need ~65 GB); detached (`tools/probe/detach.sh`), polled; `-j44` / all threads.
- Every number measured, or marked derived / estimated / proposed / published; the card says which.
- No `rm -rf`.

## Review Focus

1. **The quantised set is exactly the `nn.Linear`s.** The dry run prints the resolved set; `check.py` lists every packed tensor against `lfm2moe_ref.expected_names(cfg, "int4", <spelling>)` and refuses anything else packed (router, bias, conv taps, norms, embedding) or anything missing.
2. **Every expert of every layer sees calibration tokens.** 64 experts top-4 route ~6 % of tokens to an expert on average, and the trained `expert_bias` balances load - but `coverage` still counts routed tokens per (layer, expert) on the calibration rows with the bf16 model and fails under `COVERAGE_FLOOR` (2048, PROPOSED, Kolibri's) after `COVERAGE_ROUNDS` of targeted top-ups.
3. **The calibration is the target workload.** Agentic coding sessions with pythonic tool calls rendered by the checkpoint's own template (reasoning on - the template always opens `<think>`), code, chat; answers self-generated by the bf16 model; the held-out sets (24a's agentic transcript, code, prose, the A4 scenarios) never in the calibration rows (a hash check).
4. **F6 is like for like.** RTN and AutoRound exports scored against the same bf16 reference rows (24a's reference, the same tokens, the same eager attention); A4 against 24a's bf16 run with the same new-token budget.

---

### Task 1: the toolchain on the tiny model (Mac or box CPU)

**Files:**
- Create: `tools/quantize/lfm2moe/autoround_run.py`
- Modify: `tools/quantize/README.md` (an "LFM2.5-24B-A2B" section)

- [ ] **Step 1:** in a venv with torch, transformers 5.19.0 and AutoRound from `ar_pin.sh` (`. tools/quantize/ar_pin.sh` with `VENVPY` set), quantise 24a's `pattern8` tiny (`lfm2moe_make_tiny.py --variant pattern8`) through `autoround_run.py` (the Python API: `AutoRound(model, tokenizer, scheme="W4A16", group_size=64, sym=True, iters=..., nsamples=8, seqlen=256, device_map=DEVICE)` then `save_quantized(format="auto_round:auto_gptq")`) at `--iters 0` and `--iters 2`. Record in the README section, each with the file:line it rests on: (a) the export's expert tensor names; (b) the packed set == the `nn.Linear`s, `feed_forward.gate.weight` bf16 and bitwise the source, `expert_bias` F32 and bitwise, `conv.conv.weight` bf16 and bitwise, no `lm_head.*` (tied) - or, if present, equal to `embed_tokens` and ignored; (c) `quantization_config` (`bits` 4, `group_size` 64, `sym` true, `packing_format` `auto_round:auto_gptq`, any `extra_config`) and whether `loader::QuantConfig::parse`'s rules accept it (`src/loader/quant.h:48-53`; 24b's `check.py` repeats them); (d) whether AutoRound's MoE path forces all experts during calibration (`force_all_experts_routing_enabled`, `auto_round/modeling/fused_moe/utils.py:164` at `6afaecdb`) and the setting the real run uses; (e) on a CPU without bf16, the unquantised tensors restored to the source's bf16 bytes (Kolibri's `autoround_run.py` rule).
- [ ] **Step 2:** if any of (a)-(c) fails: one local patch at most, recorded with its reason beside `auto-round-local.patch` (`tools/quantize/auto-round-lfm2moe.patch`, applied by `autoround_run.py` only); if that does not suffice, stop and report - the RTN path (Task 2) ships.
- [ ] **Step 3: commit** `git commit -S -m "quantize: AutoRound 0.17.0 on lfm2_moe, proven on a tiny model - expert names, the packed set, the kept tensors (spec 24q)"`.

### Task 2: the RTN export and the check (no AutoRound in the path)

**Files:**
- Create: `tools/quantize/lfm2moe/rtn_export.py`, `tools/quantize/lfm2moe/check.py`, `tools/quantize/lfm2moe/test_lfm2moe_quant.py`

**Interfaces:**
- Produces: `rtn_export.py <bf16 snapshot> <out> [--expert-names w|proj] [--qset all|no-conv|no-attn|no-conv-attn]` - every selected `nn.Linear` packed by `src/loader/rtn.h:20-27`'s rule (per output column and 64-group: `s16 = f16(2 amax / 15)`, `q = clamp(rint(w / s16) + 8, 0, 15)`, `qweight` [K/8][N] with k = 8r in the lowest nibble, `scales` F16 [K/64][N], `qzeros` 0x77777777, no `g_idx`), everything else copied byte for byte; `quantization_config` as Task 1 recorded AutoRound writes it (plus `"quant_method": "auto-round"`, `"packing_format": "auto_round:auto_gptq"`, provenance `"rtn": "spec24q rtn_export.py"`); one source shard at a time (peak RAM ~3 GB, ESTIMATED); the tokenizer / template / generation config / LICENSE copied. `check.py <export> [--source <bf16 snapshot>] [--qset ...]`: names against `expected_names`, dtypes, shapes, `qzeros`, the group size, `quantization_config`, the kept tensors bitwise the source (with `--source`), Review Focus 1; prints `ACCEPTED` or the first refusal.
- [ ] **Step 1: the failing tests** (`test_lfm2moe_quant.py`, in `agnes-ref-img` with 24a's site): `test_rtn_rule` (the packer against `rtn.h`'s rule on crafted groups: an all-zero group, a group whose `2 amax / 15` rounds at f16, a tie at `rint`); `test_rtn_tiny_end_to_end` (`pattern8` -> `rtn_export.py` -> `check.py --source` `ACCEPTED`; `lfm2moe_ref.build_streamed(weights="int4")` loads it; both expert spellings); `test_check_refuses` (a packed router, a bf16 `expert_bias`, a packed conv tap, a missing expert: each refused by name); `test_same_names_as_autoround` (the RTN export's names equal Task 1's AutoRound export's names on the tiny model, when Task 1's export is present).
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "quantize: lfm2moe RTN g64 export (rtn.h's rule, AutoRound's names) and the export check (spec 24q)"`.

### Task 3: the script and its modules

**Files:**
- Create: `tools/quantize_lfm2moe.sh`, `tools/quantize/lfm2moe/calib.py`, `tools/quantize/lfm2moe/evaluate.py`, `tools/quantize/lfm2moe/card.py`, `tools/quantize/lfm2moe/prompts/` (original agentic-coding task prompts written for this repo; no third-party text)
- Modify: `tools/quantize/lfm2moe/test_lfm2moe_quant.py`, `tools/quantize/README.md`

**Interfaces:**
- Produces: `tools/quantize_lfm2moe.sh [--only STAGE | --from STAGE] [--dry-run] [--list]`; environment `MODEL` (default `LiquidAI/LFM2.5-24B-A2B`), `OUT` (`$HOME/models/lfm2moe-w4g64`), `DEVICE` (`1`: the second B70), `NSAMPLES` (512 rows of `SEQLEN` 2048, PROPOSED), `ITERS` (200), `COVERAGE_FLOOR` (2048), `COVERAGE_ROUNDS` (4), `GEN_BACKEND` (`auto` | `vllm` | `hf`), `TP` (2), `BATCH` (8), `QSET` (`all`), `RTN_BACKEND` (`autoround` | `ours`), `EVAL_ROWS` (32 per held-out set), `VENVPY`, `AR_REPO`, `AR_PATCH`. Outputs: `$OUT/calib/` (rows, the mix, the held-out hash list), `$OUT/rtn/` and `$OUT/export/` (`auto_round:auto_gptq`), `$OUT/eval/f6.md`, the card.
- Stages:
  - `calib` (`calib.py`): ~60 % agentic coding sessions (a task prompt from `prompts/`, the repo's own files as tool results, the 24a A4 tool schemas, pythonic calls), ~25 % code, ~15 % chat (PROPOSED mix, printed with token counts); each conversation rendered by `chat_template.jinja` through `apply_chat_template`, the assistant turns generated by the bf16 model (`GEN_BACKEND=vllm`: the box's XPU vLLM image at `TP` 2 if it serves `Lfm2MoeForCausalLM` - checked first, printed; `hf`: transformers on the box CPU, batched, ESTIMATED hours; `auto` picks); packed into `NSAMPLES` rows of `SEQLEN`; the held-out sets' hashes refused.
  - `coverage`: routed tokens per (layer, expert) with the bf16 model on the rows (24a's `Recorder`); experts under the floor get targeted rows (rows whose routing hit them, from more generated sessions); FAIL if any stays under it.
  - `rtn`: `RTN_BACKEND=autoround` - AutoRound `--iters 0 --disable_opt_rtn` (minutes); `ours` - `rtn_export.py`. Either output then goes through `check.py`.
  - `tune`: AutoRound sign-SGD, `--iters $ITERS --nsamples $NSAMPLES --seqlen $SEQLEN --low_gpu_mem_usage --device $DEVICE`, the rows passed pre-tokenised (`AutoRound(dataset=rows)`, Kolibri's way, so the BOS / EOS counter bug the local patch fixes is not reached); under `flock ~/b70-gpu.lock`; the first block's time x 40 printed as the honest ESTIMATE after block 1.
  - `eval` (`evaluate.py`): F6 - per held-out set (24a's agentic transcript, code, prose; `EVAL_ROWS` rows of 2048) KL(bf16 ‖ int4) and top-1 agreement, perplexity, for `rtn` and `export` (whichever exist), through `lfm2moe_ref` (bf16 vs `weights="int4"`, eager, layer-streamed); A4: `oracle_generate.py --model lfm2moe --weights int4` on each export against 24a's bf16 reference (`oracle-out-lfm2moe-a4`), `score.py --format pythonic` (agreement, parse failures); the int8 head row: KL(bf16 head ‖ int8 head) on the same rows with every other weight bf16 (spec 9's rule on the tied embedding); writes `$OUT/eval/f6.md` with the bars (decision 5) marked pass / fail.
  - `check`: `check.py --source $MODEL` on each export; `ACCEPTED` is the gate.
  - `card`: `card.py` - the model card: base model, the recipe (AutoRound version and commit, patch, flags, `QSET`, calibration mix and token counts, coverage), F6's table, the engine command, **the licence**: `license: other`, `license_name: lfm1.0`, the LICENSE file copied into the export, the modification notice ("quantised from LiquidAI/LFM2.5-24B-A2B: int4 g64 AutoRound; every other tensor unchanged") - spec 24 §1.4; refuses to render when a bar fails unless `ARM=force`.
- [ ] **Step 1: the failing tests:** `test_dry_run_prints_every_stage` (the resolved settings, the quantised set, the mix, ESTIMATED RAM / disk / time per stage); `test_tiny_end_to_end` (`--tiny <dir>`: calib -> coverage -> rtn -> eval -> check -> card on `pattern8` in minutes with `GEN_BACKEND=hf`, `check` printing `ACCEPTED`); `test_heldout_disjoint` (a held-out row planted in the calibration input is refused); `test_card_licence` (the card carries the licence fields and the notice; the LICENSE is copied).
- [ ] **Step 2: implement** (the header documents each stage as `quantize_kolibri1.sh`'s does); PASS; `tools/quantize_lfm2moe.sh --dry-run` prints every stage.
- [ ] **Step 3: commit** `git commit -S -m "quantize: tools/quantize_lfm2moe.sh - agentic calibration with pythonic calls, coverage over 64 x 38, RTN then AutoRound g64 on the B70, F6 (KL, top-1, A4), the check, the card with the LFM licence's notices (spec 24q)"`.

### Task 4: the box run (row 49)

- [ ] **Step 1: `rtn` first** (`tools/quantize_lfm2moe.sh --only rtn` then `--only check`): minutes; the export is 24b-24d's development checkpoint. Then `tools/box_validate/lfm2moe_oracle.sh <data> export $OUT/rtn` (24a's engine-format golden sets on it -> `oracle-out-lfm2moe-rtn/`).
- [ ] **Step 2: `calib`, `coverage`, `tune`, `eval`, `check`, `card`** - detached, resumable, `tune` under the GPU lock; record the machine, commit, flags, times, peak RAM, token counts, coverage. F6's table and the run record go into 24a's facts sheet (a new section) and spec 24 ("24q as built"); decision 5's bars are proposed there from RTN's row.
- [ ] **Step 3: publish** `urakozz/LFM2.5-24B-A2B-W4A16-g64-AutoRound-GPTQ` **after the operator's go** (decision 9), with the card and the LICENSE. Then `lfm2moe_oracle.sh export $OUT/export` (-> `oracle-out-lfm2moe-ours/`) so 24c-24e's `ours` stages have their reference.
- [ ] **Step 4: commit** `git commit -S -m "spec 24q: our LFM2.5-24B-A2B AutoRound g64 - F6, the run record"`.

### Task 5: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` ("24q as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 49), `tools/box_validate/stages.sh` (`row 49`), `tools/box_validate/data.sh` (`have lfm2moe_rtn`, `have lfm2moe_ours`, `have oracle_lfm2moe_rtn`, `have oracle_lfm2moe_ours`)

- [ ] **Step 1: the queue row 49** and its stages (`python3 tools/box_validate/test_box_validate.py` passes; `--dry-run --only r49` prints them):

```text
row 49 "spec 24q - LFM2.5-24B-A2B: our int4 g64 - RTN first (the development checkpoint), then AutoRound on the B70, F6 (plan 24q)"
stage r49.host     49 default cpu  oracle_image                   -        "test_lfm2moe_quant.py (RTN rule, tiny end to end, refusals, names = AutoRound's)"
stage r49.rtn      49 default cpu  lfm2moe_bf16                   -        "quantize_lfm2moe.sh --only rtn, --only check: ACCEPTED (minutes) -> \$DATA/lfm2moe-w4g64/rtn"
stage r49.rtn_ref  49 optin   cpu  lfm2moe_rtn,oracle_image       r49.rtn  "lfm2moe_oracle.sh export rtn: the engine-format golden sets 24b-24d gate against"
stage r49.tune     49 optin   gpu  lfm2moe_bf16                   r49.rtn  "calib, coverage, tune (AutoRound on device 1, HOURS, holds the GPU lock), check"
stage r49.eval     49 optin   cpu  lfm2moe_bf16,lfm2moe_a4_ref    r49.tune "F6: KL / top-1 per set and A4 for rtn and ours; the int8-head row; f6.md"
stage r49.ours_ref 49 optin   cpu  lfm2moe_ours,oracle_image      r49.eval "lfm2moe_oracle.sh export ours -> oracle-out-lfm2moe-ours"
rownote 49 "Publication (urakozz/LFM2.5-24B-A2B-W4A16-g64-AutoRound-GPTQ) waits for the operator's go (decision 9: LFM Open License v1.0 notices, the USD 10M threshold)."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: spec 24 (24q as built), box queue row 49 - our LFM2.5-24B-A2B quantisation"`.

**Gate for the plan:** the tiny end to end `ACCEPTED` (both RTN backends; AutoRound's names recorded); the dry run complete; on the box - the RTN export `ACCEPTED` and its golden sets written; after the tune - F6's full table (RTN and ours vs bf16: KL / top-1 per set, A4 agreement and parse failures, the int8-head KL) against decision 5's bars, the export `ACCEPTED`, published only after the operator's go.

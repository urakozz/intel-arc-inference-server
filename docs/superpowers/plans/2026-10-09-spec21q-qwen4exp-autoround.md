# Spec 21q - Qwen3.8-Flash-Next: our AutoRound int4 g64 (the script, the recipe, F6)

**Status (2026-10-09): planned; nothing built.** Parallel to 21b-21e once 21a's facts exist. The 360 GB run
itself waits for spec 21 decision 6 (where it runs, and the MTP head's experts). No box queue row: the run is a
big machine's; F6's PLE row runs on the box CPU and is row 31's opt-in stage.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the one weight format for this family - AutoRound W4A16 int4 g64 symmetric re-quantised by us from the 360 GB bf16 original - as one resumable command on Kolibri's 20b model (`tools/quantize_kolibri1.sh`: a script plus a recipe, stage by stage), with `ple` in the ignore list and the PLE table converted separately; F6's quality table (ours g64 vs the bf16 original, Intel's g128 vs the bf16 original, the int8 PLE table vs bf16); the export accepted by 21b's loader and check, published.

**Architecture:** spec 21 §5 "Ours", §7 F6, §8 21q, decisions 6 and 7. `tools/quantize_qwen4exp.sh` (stages `calib`, `coverage`, `rtn`, `tune`, `eval`, `ple`, `check`, `card`; `--only` / `--from` / `--dry-run` / `--list`; a finished stage leaves `$OUT/.done-<stage>`), modules in `tools/quantize/qwen4exp/` beside 21b's `make_synth.py` / `ple_int8.py` / `check.py`. Model-neutral pieces of `tools/quantize/kolibri/` (the row writer, the routed-token counter, the KL / top-1 evaluator) are imported, not copied; what is this model's (the calibration mix through its template, the ignore list, the fused-expert export, the card) is new. AutoRound is pinned by commit through `tools/quantize/ar_pin.sh`, as Kolibri's run pinned `6afaecdb` (0.17.0).

**Tech Stack:** bash; Python 3, torch, transformers 5.19.0 (21a's pin), AutoRound at a pinned commit, vLLM where it serves `qwen4_exp` for self-generated answers (`GEN_BACKEND`); a big machine (decision 6) for the 360 GB model; the box CPU for F6's PLE row.

**Spec:** `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§5, §7 F6, §8 21q, §10 decisions 6, 7). Precedents: `tools/quantize_kolibri1.sh` (its header is this script's template), `tools/quantize/kolibri/{calib,coverage,autoround_run,evaluate,check,card,common}.py`, `tools/quantize/README.md`, plan 20a Task 4, `docs/13-loader.md` (`auto_round:auto_gptq` as `loader::QuantConfig::parse` reads it).

## Dependencies and branch points

- **Plan 21a merged** (the facts: the original stores routed experts fused, `mlp.experts.gate_up_proj [512, 1280, 2560]` gate-then-up and `down_proj [512, 2560, 640]` under `model.language_model.`; the tokenizer and template; the reference for `eval`). **Plan 21b's** `check.py` and `ple_int8.py` are this plan's `check` and `ple` stages (21b Task 1); until 21b merges, those two stages refuse naming it.
- **Decision 6 (open):** where the run happens (a rented big machine, as 20b's decision 1 - the bf16 model is 360 GB, ~102 GB of it the PLE table AutoRound never touches) and the MTP head's routed experts: `MTP_EXPERTS=bf16` (left as shipped, 5.03 GB; 21b quantises them at load by RTN), `rtn` (RTN g64 in the export, 1.34 GB) or `autoround` (a second AutoRound pass over `mtp.*` with the main model's last hidden as input). The script builds all three; the operator picks.
- **Decision 7 (open):** the PLE table's scale dtype (`f32` or `bf16`) and that the int8 file ships beside our export (21b's proposal): the `ple` stage runs 21b's converter with `PLE_SCALE`, and `eval`'s PLE row measures both dtypes - the evidence for the ruling.
- **The fused experts:** AutoRound must export per-expert `qweight` / `scales` / `qzeros` (what 21b's loader reads for both checkpoints). Task 1 proves the pinned commit does so on the tiny model, from fused bf16 experts (the original's form, which the tiny model does not have: 21a's `qwen4exp_make_tiny.py` writes a fused variant); if no AutoRound commit does, stop and report - no third-party format, no engine-side fused path.

## Global Constraints

- Branch `spec21q-qwen4exp-quant` from main; signed commits; no merge, no push. Never commit weights, tokens or `tools/box.env`.
- **Settings that are spec 21 §5 and not overridable:** scheme W4A16, `group_size` 64, symmetric, `packing_format` `auto_round:auto_gptq`, model dtype bf16. Quantised: routed and shared experts, QSA `q_proj` / `k_proj` / `v_proj` / `o_proj`, GDN `in_proj_qkv` / `in_proj_z` / `out_proj`. Kept bf16 (the ignore list): `mlp.gate` (routers), `shared_expert_gate`, `in_proj_a`, `in_proj_b`, `indexer`, `hyper_connection` (and `hyper_connection_mixer`), every norm, `embed_tokens`, `lm_head` (no `--quant_lm_head`: the engine builds its int8 head at load, spec 9), `visual`, and **`ple`** (its projections stay bf16; the table is converted by the `ple` stage). `mtp` per `MTP_EXPERTS`.
- One format: no asymmetric, no g128, no compressed-tensors export.
- Every number measured, or marked derived / estimated / published; the card says which.
- No `rm -rf`.

## Review Focus

1. **`ple` in the ignore list, and nothing of the table quantised by AutoRound.** The dry run prints the resolved ignore list; the `check` stage refuses an export with any `ple.*.qweight` or a missing `ple.*` bf16 tensor, and requires the 128 `ngram_embedding.shard_*` tensors either absent (the int8 file carries the table) or bf16 as shipped.
2. **Every one of the 512 x 48 experts sees calibration tokens.** Top-10 of 512 routes ~2 % of tokens to an expert on average; `coverage` counts routed tokens per (layer, expert) on the calibration set with the bf16 model, tops up experts under `COVERAGE_FLOOR` (2048, PROPOSED, Kolibri's) with targeted text, and fails if any stays under it after `COVERAGE_ROUNDS`.
3. **The calibration is the target workload.** Agentic coding sessions with tool calls (Qwen XML through the checkpoint's own template, reasoning on), chat, and code - the mix and token counts printed by the dry run; answers self-generated by the bf16 model; held-out evaluation sets never in the calibration rows.
4. **F6 is a like-for-like table.** Ours g64 and Intel's g128 are scored against the same bf16 reference on the same held-out rows (KL, top-1, perplexity, per set); the int8 PLE row holds every other weight equal (Intel's checkpoint dequantised, bf16 PLE vs int8 PLE).

---

### Task 1: the toolchain on the tiny model

**Files:**
- Modify: `tools/quantize/ar_pin.sh` (a `qwen4exp` pin, if the script holds per-model pins; else the pin is the new script's `AR_COMMIT`), `tools/quantize/README.md`
- Test: a tiny end-to-end run (below)

- [ ] **Step 1:** in a venv with torch, transformers 5.19.0 and AutoRound at a candidate commit (the newest whose `qwen4_exp` / fused-3D-expert handling exists - read its source, record file:line): quantise the fused tiny variant (`qwen4exp_make_tiny.py --fused-experts`) with the settings above, `--iters 0` and `--iters 2`; check the export: per-expert `qweight` / `scales` / `qzeros` for every routed and shared expert, g64, `qzeros` 0x77777777, the ignore list's tensors bf16 and bitwise the source, `quantization_config` parsed by `loader::QuantConfig::parse`'s rules (21b's `check.py` once merged). Record the commit and any local patch needed (as `tools/quantize/auto-round-local.patch` was for Qwen3.8 - a patch is the last resort, recorded with its reason).
- [ ] **Step 2: commit** `git commit -S -m "quantize: the AutoRound pin for Qwen3.8-Flash-Next, proven on a fused-expert tiny model (spec 21q)"`.

### Task 2: the script and its modules

**Files:**
- Create: `tools/quantize_qwen4exp.sh`, `tools/quantize/qwen4exp/calib.py`, `tools/quantize/qwen4exp/autoround_run.py`, `tools/quantize/qwen4exp/evaluate.py`, `tools/quantize/qwen4exp/card.py`, `tools/quantize/qwen4exp/prompts/` (original agentic-coding task prompts written for this repo; no third-party text)
- Modify: `tools/quantize/qwen4exp/test_qwen4exp_quant.py` (21b's: the script's tests), `tools/quantize/README.md`

**Interfaces:**
- Produces: `tools/quantize_qwen4exp.sh [--only STAGE | --from STAGE] [--dry-run] [--list]`, environment `MODEL` (default `Qwen/Qwen3.8-Flash-Next`), `OUT` (`$HOME/models/qwen4exp-w4g64`), `DEVICE`, `NSAMPLES` (1024 rows of `SEQLEN` 2048, PROPOSED), `ITERS` (200), `COVERAGE_FLOOR` (2048), `COVERAGE_ROUNDS` (4), `GEN_BACKEND` (auto | vllm | hf), `TP`, `BATCH`, `MTP_EXPERTS` (bf16 | rtn | autoround), `PLE_SCALE` (f32 | bf16), `VENVPY`, `AR_SRC`; the export `$OUT/export/` (`auto_round:auto_gptq`), `$OUT/export-ple-int8/` (21b's file format), `$OUT/eval/f6.md` (the table), the card.

- [ ] **Step 1: the failing tests:** `test_dry_run_prints_every_stage` (the resolved settings, the ignore list with `ple`, the calibration mix, estimated RAM / disk / time per stage - ESTIMATED); `test_tiny_end_to_end` (`--tiny <dir>`: calib -> coverage -> rtn -> eval -> ple -> check -> card on the fused tiny variant in minutes, `check` printing `ACCEPTED`); `test_ignore_list` (Review Focus 1); `test_mtp_modes` (each `MTP_EXPERTS` writes the expected `mtp.*` names). Run in `agnes-ref-img` with 21a's site: FAIL.
- [ ] **Step 2: implement** (the header documents each stage as `quantize_kolibri1.sh`'s does; `eval` writes the F6 table with both checkpoints when `INTEL=<snapshot>` is set); PASS; `tools/quantize_qwen4exp.sh --dry-run` prints every stage.
- [ ] **Step 3: commit** `git commit -S -m "quantize: tools/quantize_qwen4exp.sh - agentic calibration, per-expert coverage over 512 x 48, RTN then AutoRound g64, F6, the PLE int8 file, the check (spec 21q)"`.

### Task 3: F6's PLE row (box CPU, before the big run)

- [ ] **Step 1:** `evaluate.py --ple-only` on Intel's checkpoint (21a's reference, layer-streamed): the same held-out rows through the bf16 PLE table (the original's shards) and the int8 file at `f32` and at `bf16` scales (21b's converter); KL and top-1 per set against the bf16-PLE run; the table into `docs/probe-qwen4exp-<date>.md` (21a's sheet, a new section) - decision 7's evidence. Box queue row 31's opt-in stage `r31.ple_kl` runs it.
- [ ] **Step 2: commit** `git commit -S -m "probe: the int8 PLE table against bf16 - KL per set, f32 vs bf16 scales (spec 21 F6, decision 7)"`.

### Task 4: the real run (decision 6's machine) and publication

- [ ] **Step 1 (operator, decision 6):** `tools/quantize_qwen4exp.sh` on the chosen machine, detached, resumable; `rtn` first (the baseline row of F6), then `tune`; `eval` with `INTEL=<Intel's snapshot>` for the like-for-like rows; `ple`; `check` (`ACCEPTED` is the gate); `card`.
- [ ] **Step 2:** the F6 table and the run record (machine, commit, time, peak RAM, token counts, coverage) into the facts sheet; publish `urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ` (PROPOSED name, the Kolibri pattern) with the int8 PLE file beside it, after the operator's go.
- [ ] **Step 3:** on the box, 21b's loader on our export (`qwen4exp_load_checkpoint_test` with `B70_Q4EXP_OURS_SNAPSHOT`, a stage added to row 31) and 21c-21e's real-weight gates re-run against it (their rows' `ours` stages). **Commit** `git commit -S -m "spec 21q: our Qwen3.8-Flash-Next AutoRound g64 - F6, the record"`.

**Gate for the plan:** the tiny end to end `ACCEPTED`; the dry run complete; F6's PLE row recorded (decision 7's evidence); after decision 6's run - F6's full table, the export `ACCEPTED` by `check.py` and loaded by `load_qwen4exp` with 0 unconsumed.

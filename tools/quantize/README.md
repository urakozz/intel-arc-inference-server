# Quantisation provenance

The checkpoint every gate in this repo grades against -
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, int4 **symmetric, group_size 64**,
GPTQ export (`docs/03-models.md`) - was produced by AutoRound **with two local
patches that are not upstream**. They lived only as uncommitted edits in
a local `auto-round` clone until 2026-09-22; `auto-round-local.patch` is that
diff, captured so the recipe survives the disk.

Checked against `origin/main` on 2026-09-22 (36 commits ahead of the clone's
`141e4c99`): **neither fix is upstream**, so the patch is still required.

- **`calib_dataset.py`** - resets `os_cnt, have_bos, have_eos` per sample instead
  of once before the loop, in both `get_opencode_instruct_dataset` and
  `_get_dataset_impl`. Without it the BOS/EOS bookkeeping accumulates across
  calibration samples: every sample that starts with BOS or ends with EOS adds one
  to a counter that is never reset, so every later concatenated row reserves more
  room for markers it does not carry and is cut short. Dormant when the samples
  carry no BOS/EOS (Qwen on pile-10k); active on chat-formatted data ending in
  `<|im_end|>`, Qwen's EOS (e.g. opencode-instruct). Kolibri's script passes
  pre-tokenised rows (`AutoRound(dataset=rows)`) and never reaches this code.
- **`wrapper.py`** - gives `WrapperLinear` `in_features` / `out_features`
  properties forwarding to the wrapped layer, which the pipeline queries and
  upstream does not expose.

Apply with `git apply tools/quantize/auto-round-local.patch` in an auto-round
checkout, or keep them on a local branch and rebase. Re-checked against 0.17.0 @ `6afaecdb`
(2026-10-06): still not upstream (the `os_cnt` reset is still outside the loop, `WrapperLinear`
still has no `in_features`), and the patch still applies cleanly (`git apply --check`, offsets
only). **Both Qwen3.8 scripts apply it since 2026-10-06** (`tools/quantize/ar_pin.sh`,
`AR_PATCH=1`, the default; `AR_PATCH=0` builds the commit as is, into its own package directory);
`quantize_kolibri1.sh` does not (see above). The published checkpoints below say which build made
them.

## AutoRound version (all scripts)

| script | auto-round | why |
|---|---|---|
| `quantize_qwen38_tuned.sh` | **0.17.0, intel/auto-round main `6afaecdbe4092dc803f3e91026fe81a65b64b372`** (`AR_COMMIT`) + `auto-round-local.patch` | the calibrated path packs `lm_head` on 0.17.0 (checked below) |
| `quantize_qwen38_rtn.sh` | **0.17.0 @ `6afaecdb` + `auto-round-local.patch`, with `--disable_low_cpu_mem_usage`** (since 2026-10-06; 0.14.2 before) | that flag takes the path that packs `lm_head`; the default low-CPU-memory RTN path still drops it on 0.17.0 (below). Holds the whole bf16 model in RAM: refuses below `RTN_MIN_RAM_GB` (64) available |
| `quantize_kolibri1.sh` | 0.17.0 @ `6afaecdb` | the ruling of 2026-10-05; Kolibri keeps lm_head bf16, so the bug is not exercised |
| `quantize_qwen38_mxfp4.sh` | none - llm-compressor `model_free_ptq` | only borrows the auto-round venv's python |
| `quantize/quantize_qwen4exp.sh` (Flash-Next) | **the LATEST of our fork: the hash branch `local-patches-v2` of `~/auto-round` (`AR_REPO`, `AR_REF`) has when the run starts** (2026-11-13: `8186d8b66af4bd56c9c352951786216bff7281f5` = intel/auto-round main `42db5d3a` + the cached-OpenCodeInstruct commit; version string 0.17.0), **no patch file** (`AR_PATCH=0`); the resolved hash and its upstream base go into the export's README and `qwen4exp_recipe.json`; `tools/quantize/ar_latest.sh` | per-block resume, the vectorised QSA indexer, its own MTP export, OpenCodeInstruct from the cache; the old pin `a6cf7487` (= `b58069ea` + the same commit) is the branch `pin-a6cf7487-2026-11-13`; see "Qwen3.8-Flash-Next: our AutoRound export" below |
| `quantize/quantize_dflash.sh` (the DFlash 2 drafter) | the same: the latest `local-patches-v2` head, hash and upstream base recorded in `dflash_recipe.json` and the README, `AR_PATCH=0` | the calibration rows are our own tensors, so neither fix of the patch is on its path; see "The DFlash 2 drafter" below |

**Latest, not a pin (Flash-Next and DFlash scripts).** `ar_latest.sh` runs `git rev-parse` on `AR_REF` in `AR_REPO` once when `start` / `run` begins and exports the 40-hex hash as `AR_COMMIT` (the detached run builds that same hash even if the branch moves); it never fetches, pulls or rebases `~/auto-round`. `ar_pin.sh` keys the package cache on the hash (`~/.cache/auto-round-<hash8>-plain-pkg`), so a moved branch rebuilds. A run records its hash in `ar_commit` (Flash-Next: `<OUT>/ar_commit`; DFlash: `<OUT>.work/ar_commit`) and never switches: `start` or `run` on a run whose branch has moved refuses and names the recorded hash; `AR_COMMIT=<recorded hash>` resumes on the old build (an old run with progress but no record needs the same).

**Install (Qwen3.8 scripts, `tools/quantize/ar_pin.sh`).** Clones `AR_REPO` (default
`https://github.com/intel/auto-round`, or any local clone that contains the commit) into
`~/.cache/auto-round-src`, checks out `AR_COMMIT`, applies `auto-round-local.patch` (`AR_PATCH=1`,
the default), and installs it with `--no-deps --target` into
`~/.cache/auto-round-6afaecdb-p<patch sha256[:8]>-pkg` (or `-plain-pkg` for `AR_PATCH=0`) beside
the venv, once per (commit, patch); the scripts run with `PYTHONPATH` pointing at it, so the venv
itself is not modified (`git` is needed: setup.py runs `git describe`; `auto_round.__version__`
says `0.17.0`). The check after the install also asserts the patch state
(`WrapperLinear.in_features` is a property iff `AR_PATCH=1`). They assert `auto_round.__version__ >= 0.17.0`, python >= 3.11 (0.17's floor; 0.14.2's
was 3.10) and that `datasets`, `py-cpuinfo`, `pydantic`, `accelerate` are importable (the same
requirement list as 0.14.2). Every Qwen3.8 export is refused unless
`check_gptq_export.py` passes, and gets a provenance section appended to its `README.md`
(auto-round version and commit, flags, source, date).

**The CLI did not move under these scripts.** 0.14.2 already had the refactored parser; every flag
the Qwen3.8 scripts pass means the same in 0.17.0 (checked: `--help` of both, and
`auto_round/cli/parser.py`):

| flag | 0.14.2 | 0.17.0 |
|---|---|---|
| `--scheme W4A16` | preset, default W4A16 | same |
| `--group_size 64` | int | int, or comma ints (block fp8) |
| symmetric | default; `--asym` opts out (no `--sym` since before 0.14.2) | same |
| `--bits` | int | float (an int is still required without `--schemes`); alias `--bit` |
| `--quant_lm_head` | `BooleanOptionalAction`, `auto_round*` formats only | same |
| `--format auto_round:auto_gptq` | alias `--formats` | same; not eligible for model-free auto-routing (that needs format `auto_round`) |
| `--iters 0` / `--disable_opt_rtn` | RTN / plain RTN (`--enable_opt_rtn` the other way) | same |
| `--nsamples`, `--seqlen`, `--batch_size` | default 128 / 2048 / 8 (recipe `default`) | same |
| `--dataset` | default `NeelNanda/pile-10k` in the parser | default `None` in the parser, resolved to `NeelNanda/pile-10k` in `compressors/base.py` - same data |
| `--device 1` | alias of `--device_map` | same |
| `--low_gpu_mem_usage` | flag | same |
| low CPU memory | on by default; `--disable_low_cpu_mem_usage` | same - and it decides the RTN `lm_head` bug (below) |
| layer exclusions | `--ignore_layers` / `--fp_layers` | same |
| model-free | `--model_free`, `--disable_model_free` | same |
| new in 0.17.0 | - | `--max_shard_size`, `--num_hidden_layers` (debug), `--disable_torch_compile`, `--enable_neuqi`, `--schemes` (`--options`/`--avg_bits` kept as hidden aliases) |

**The lm_head bug, re-checked on 0.17.0 (2026-10-06).** Tiny random LLaMA and Qwen3 models (2
layers, hidden 128, vocab 512, **untied** lm_head) quantised with the scripts' exact flags on CPU
(`agnes-ref-img`, torch 2.14.1+cpu, transformers 5.15.0, 3 threads), 0.14.2 against 0.17.0 @
`6afaecdb`, every export run through `check_gptq_export.py --source` and byte-compared:

| path | 0.14.2 | 0.17.0 |
|---|---|---|
| RTN (`--iters 0 --disable_opt_rtn`), LLaMA and Qwen3 | lm_head packed, VERIFIED | **lm_head NOT packed**: `lm_head.weight` shipped, while `extra_config.lm_head` still says bits 4 / g64 / sym - FAIL |
| RTN + `--disable_low_cpu_mem_usage`, LLaMA | - | packed; **byte-identical to 0.14.2** (51/51 tensors) |
| tuned (4 iters, 8 x 64 local samples), LLaMA | packed, VERIFIED | packed, VERIFIED; **byte-identical to 0.14.2** (51/51) |
| tuned, Qwen3 | - | packed, VERIFIED |

**With the patch (2026-10-06).** Both scripts on `ar_pin.sh`, the tiny Qwen3, the same container,
`AR_REPO` = the local checkout: tuned (patched) VERIFIED; RTN (patched, 0.17.0,
`--disable_low_cpu_mem_usage`) VERIFIED and **byte-identical to the 0.14.2 RTN export (55/55
tensors)**; the RTN RAM guard refused at `RTN_MIN_RAM_GB=9999`; tuned with `AR_PATCH=0` VERIFIED
from its own package directory; a rerun reused the patched package. Patched and unpatched tuned
exports are byte-identical here, as expected: the local calibration samples carry no BOS/EOS, so
the counter fix has nothing to change, and the `WrapperLinear` properties do not alter tuning.

The two scripts as committed before the patch step also ran end to end on the tiny Qwen3 (`MODEL`, `DEVICE=cpu`, fresh
`HOME`, `EXTRA` for a 4-iteration tune on local samples): tuned installed 0.17.0 from a clean
checkout at `AR_COMMIT` and VERIFIED, rtn installed 0.14.2 from PyPI and VERIFIED, both wrote the
provenance section; the tuned script forced onto RTN (`EXTRA="--iters 0 --disable_opt_rtn"`)
**refused with exit 1** ("lm_head NOT packed"), which is the guard doing its job.

Every other tensor is byte-identical between the versions in every arm; config keys, the
`packing_format` string, the `extra_config` spelling and its `lm_head` entry are the same, and the
loader's rules (`src/loader/quant.cc`) accept both. The cause, in 0.17.0's source: with low CPU
memory on (the default) RTN takes the zero-shot loop, which saves through `ShardWriter`
immediately; it quantises `lm_head` as a "remaining layer" (`compressors/orchestrator.py`,
`Quantizing remaining layer ...`) **without the `immediate_pack` call** the calibrated loop makes,
so `ShardWriter.finalize` writes the plain `lm_head.weight` from the state dict. The calibrated
path turns immediate packing off when a quantised layer lives outside the blocks and packs every
layer at export. Hence: the tuned script moved to 0.17.0, and the RTN script moved with `--disable_low_cpu_mem_usage`
, which packs at export like the calibrated path. It holds the whole model in
RAM (~54 GB bf16 for the 27B; the test machine has ~128 GB) and has not run on the 27B yet - the checker
refuses the export if `lm_head` comes out unpacked. Not exercised on CPU: XPU; the 27B's GDN layers and MTP head (the
scripts' checker guards the real run). On a CPU without bf16, both versions write unpacked tensors
as F32 (the checker WARNs with `--source`); on an XPU they stay bf16.

**Reproducibility.** The published checkpoints were not made by 0.17.0, and re-running a script
now does not reproduce them byte for byte unless the version matches:
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` (the gate) by the patched build (`~/auto-round`,
`141e4c99` plus `auto-round-local.patch`, `--format auto_gptq`, bf16 lm_head); the int4-lm_head RTN
checkpoint of docs/13 by 0.14.2 (`autoround_version` 0.14.2, `tests/loader/quant_test.cc`'s
fixture); `urakozz/Ornith-1.5-35B-A3B-W4A16-g64-AutoRound-GPTQ` by **0.15.0** (its config's
`autoround_version`). On the tiny models the 0.14.2 and 0.17.0 tuned exports are
byte-identical, which is evidence, not proof, for the 27B.

Re-run the check on any export:

```sh
python3 tools/quantize/check_gptq_export.py OUT_DIR --lm-head quant [--source UNQUANTISED_DIR]
```

## The command

auto-round has no `--sym` (absent from 0.14.2's parser already; symmetric is the default and
`--asym` opts out), and `--device` is an alias of `--device_map`. The command is unchanged on
0.17.0; `--format auto_gptq` is the gate checkpoint's export (bf16 lm_head - `--quant_lm_head`
needs an `auto_round*` format):

```sh
~/auto-round/.venv/bin/python -m auto_round \
  --model Qwen/Qwen3.8-27B \
  --bits 4 --group_size 64 \
  --format auto_gptq \
  --iters 0 \
  --device xpu \
  --output_dir ~/models/Qwen3.8-27B-W4A16-g64-rtn
```

`--iters 0` is RTN - minutes, lower quality. Drop it for real AutoRound tuning.
The loader asserts sym / g64 / gptq, so those are not optional. A fresh
checkpoint needs its own `oracle-out-*` golden set before it can gate anything.

Also present in 0.16.0 and relevant to the W4A8 probe:
`--act_bits`,
`--act_group_size`, `--act_data_type`, `--disable_act_dynamic`.

## Kolibri-1 (spec 20 §3.1): `tools/quantize_kolibri1.sh`

One command for `Aleph-Alpha/Kolibri-1-BF16` -> `urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ`, with
the recipe of spec 20 §3.1 encoded stage by stage (helpers in `kolibri/`, facts and checks in
the stage list below):

| stage | what | helper |
|---|---|---|
| `calib` | calibration rows (SEQLEN tokens, every row at a conversation boundary, chat template, reasoning on): German 50 % (chat 20, tool calls 10, Wikipedia documents 20), English chat 30 %, code 20 %; the answers self-generated by the bf16 model (vLLM + `aleph-alpha-inference` when importable, else the transformers port); the prompts' `eval` split and Wikipedia shard 00019 become the held-out sets | `kolibri/calib.py`, `kolibri/prompts/*.jsonl` |
| `coverage` | routed tokens per layer x expert on the calibration rows (the bf16 model's own router, `tools/oracle/kolibri_ref.py` layer-major); experts under `COVERAGE_FLOOR` (2048) get German articles added greedily; **exit 3** naming the starved experts if they cannot be filled | `kolibri/coverage.py` |
| `rtn` | AutoRound `iters=0`, plain RTN (`disable_opt_rtn`), per attention arm | `kolibri/autoround_run.py` |
| `tune` | AutoRound sign-SGD `iters=ITERS` (200) over every calibration row, `seqlen` 2048, `low_gpu_mem_usage` | `kolibri/autoround_run.py` |
| `eval` | KL / top-1 / perplexity vs bf16 on the held-out sets, German and English apart, against the bars | `kolibri/evaluate.py` |
| `check` | the engine's acceptance of every export (`src/loader/quant.cc`'s rules in Python, bf16 tensors bitwise the source's, a dequant sample) | `kolibri/check.py` |
| `card` | README.md for the arm that clears the bars (int4 attention preferred), none if no arm does | `kolibri/card.py` |

Fixed (not overridable): W4A16, group 64, symmetric, `auto_round:auto_gptq`, model dtype bf16; bf16
for the router `mlp.gate` (not an `nn.Linear` in our port, so nothing can quantise it), `expert_bias`,
`shared_experts` (`ignore_layers`), norms, `embed_tokens`, `lm_head`. Attention: `ATTN=int4|bf16|both`
(default both - spec 20 decision 2 is decided by the `eval` table). The model class is our
transformers port `tools/oracle/third_party/kolibri1/`, wired in through a shim directory
(`$OUT/Kolibri-1`: the snapshot's files symlinked, config.json with an `auto_map`, the two .py
files). Resumable: a finished stage writes `$OUT/.done-<stage>`; `--only STAGE`, `--from STAGE`,
`--dry-run` (prints every stage's exact commands), `--list`.

```sh
# the real run (see "Machine" below); every stage, both attention arms
MODEL=Aleph-Alpha/Kolibri-1-BF16 OUT=/data/kolibri1-w4g64 DEVICE=cuda NSAMPLES=1024 \
  VENVPY=/opt/ar/bin/python GEN_PY=/opt/vllm/bin/python TP=2 \
  setsid nohup tools/quantize_kolibri1.sh > ~/kolibri-quant.log 2>&1 < /dev/null &
tools/quantize_kolibri1.sh --dry-run          # the exact commands, nothing run
```

**AutoRound for Kolibri is NOT the Qwen3.8 pin.** The ruling (2026-10-05): intel/auto-round
main at `6afaecdbe4092dc803f3e91026fe81a65b64b372` (0.17.0), the `AR_COMMIT` in the script, installed
into `VENVPY` on first use from `AR_SRC` (a clean checkout at that commit - verified - or the pip
git URL); `kolibri_recipe.json` and the card record it. Both Qwen3.8 scripts use the same commit
since 2026-10-06, with `auto-round-local.patch` applied (see "AutoRound version" above); Kolibri's
does not need the patch's dataset fix (it passes pre-tokenised rows). Checked for Kolibri on a tiny random Kolibri (below): the 0.17.0 API names
(`scheme`, `group_size`, `sym` - the CLI has only `--asym` -, `iters`, `nsamples`, `seqlen`,
`low_gpu_mem_usage`, `device_map`/`--device`, `ignore_layers`/`--fp_layers` substring match,
`model_dtype`, `disable_opt_rtn`, `disable_model_free`); its fused-MoE unfusing only touches 3-D
expert parameters, so our per-expert `nn.Linear`s keep their names; the export has every expert's
`qweight` / `scales` / `qzeros` (0x77777777), no `g_idx`, `extra_config` bits 16 for the excluded
modules. The 0.14.2-era shard-writer bug (lm_head packing dropped) does not apply: lm_head stays bf16.
On a CPU without bf16, AutoRound writes the unquantised tensors in **fp32**; `autoround_run.py`
restores them to the source's bf16 bytes after proving each is bitwise the widened source.

**Machine (estimated, not measured on the real model).** Disk: 156 GB source + ~45 GB per arm
(4 arms ~180 GB) + AutoRound's offload directory. RAM: the bf16 model is read layer by layer by the
reference stages but loaded whole by AutoRound (low_cpu_mem_usage offloads, still budget ~200 GB);
the calibration activations are 1024+ x 2048 x 2560 bf16 = 11 GB. `calib` needs the bf16 model on
GPUs for generation (~1.4M generated tokens): vLLM + `aleph-alpha-inference` on 2 x H100 80 GB
(`TP=2`) or one B200 - tens of minutes; the transformers fallback is for tests only (days on a CPU).
`coverage` and `eval` run the reference over ~2.3M and ~1.3M tokens: hours on a 44-thread CPU, far
less with `DEVICE=cuda`. `rtn`: well under an hour per arm. `tune`: AutoRound loads one 3 GB block
of 384 experts at a time; on one H100-class GPU estimate 10-30 h per arm at 1024 x 2048 samples (the
first block's time in the log x 50 is the honest figure - spec 20 decision 1 measures it). On the
GPU machine: `DEVICE=xpu` for AutoRound works in principle; `calib` cannot run there at a useful speed.

**Tested on the Mac** (tiny random Kolibri from `kolibri/make_tiny.py`: 5 layers, 16 experts, hidden
128, the real tokenizer; in `agnes-ref-img` + a venv with auto-round 0.17.0 @ 6afaecdb, 3 threads):
every stage end to end - `calib` (transformers generation, 20 rows x 256), `coverage` (7 starved
experts failed loudly with no documents left, then filled by 8 top-up rows when given more),
`rtn` and `tune` (2 iterations: AutoRound keeps the best-loss iterate, here the RTN start, so the
exports equal RTN's; a 20-iteration run cut every block's loss by 13-28 % and changed all 240 int4
linears) for both attention arms, `eval` (the table; random weights fail the bars, so `card`
refused with exit 2 as designed, and rendered when an arm was forced), `check` (all ACCEPTED);
plus `kolibri/test_kolibri_quant.py` (packing, the tool-call turn == the template's rendering, the
greedy top-up, restore + check accepting the right export and rejecting a bad qzeros word, a
permuting g_idx, a changed expert_bias, a quantised shared expert, a column-permuted qweight and an
asymmetric config, the KL arithmetic, the card's choice, the prompt lists).

## Qwen3.8-Flash-Next: our AutoRound export (`tools/quantize/quantize_qwen4exp.sh`, spec 21-QFN-v2 §2)

The rulings (2026-11-12): our own export of the bf16 original `Qwen/Qwen3.8-Flash-Next` (360 GB), int4 g64
symmetric everywhere it quantises (the routed and shared experts, QSA q/k/v/o, GDN qkv / z / out, `lm_head`, the MTP
head's routed experts), **tuned** (sign-SGD, 200 iterations, 128 x 2048 rows, batch 8 - the Qwen3.8 tuned recipe's
sizes - of **nvidia/OpenCodeInstruct** through the checkpoint's chat template: the model is for agentic coding,
), **no comparison with Intel's checkpoint** - quality is ours against the bf16
original and the f32 judge. The commands follow.

### The commands (on the GPU machine, in the repo tree)

```sh
tools/quantize/quantize_qwen4exp.sh check-download   # DOWNLOAD OK: 144 files, 360.02 GB ... (--sha: also hashes, ~30 min)
tools/quantize/quantize_qwen4exp.sh start            # detached, card 1; refuses unless download, RAM and disks are ready
tools/quantize/quantize_qwen4exp.sh status           # blocks done / 48, s per block, AutoRound's time left, RAM, the result
tools/quantize/quantize_qwen4exp.sh stop             # TERM to the run's process group
tools/quantize/quantize_qwen4exp.sh start            # after a stop, a crash or a reboot: resumes at the first unfinished block
tools/quantize/quantize_qwen4exp.sh selftest         # the whole pipeline on the tiny model, ~5 min (card 1; DEVICE=cpu: the CPU)
```

- **Card:** one card (`CARD`, default 1): card 0 stays free for other card jobs. A job that needs both cards waits
  until the quantisation ends, so start such runs before, or `stop` and `start` around them.
- **Output and disks:** `QUANT_ROOT` = `$B70_DATA/hdd/quant/` (a large disk) holds
  `Qwen3.8-Flash-Next-W4A16-g64-<date>/` (`flash-next-current` points at it): `export/<snapshot>-w4g64/` (the
  checkpoint), `export/<snapshot>-w4g64-ple-int8/` (the engine's PLE file), `resume/`, `work/`, `RESULT`, `run.log`
  (-> `$B70_DATA/logs/flash-next-quant/run-<time>.log`). `PLE_DIR` = `$B70_DATA/scratch/flash-next-ple/` on a fast SSD holds the
  PLE table as one raw 102 GB file while `quant` runs (below). Growth (derived from the real-width probe and the tiny
  runs): `resume/` 21.5 GB from the first block on, overwritten each block (two fp32 files: 128 x 2048 x 10240 x 4 B
  each); the export ~1.4 GB a layer (512 experts 1.34 GB + the dense part), +102.4 GB of bf16 PLE table once layer 1
  is written, ~172 GB at the end (experts 64.2, MTP experts 1.3, dense ~1.7, embedding 1.3, head 0.34, vision ~0.9,
  PLE 102.4); then the int8 PLE file 51.8 GB: **~245 GB on the large disk, 102 GB on the SSD**. `start` wants
  300 GB free under `QUANT_ROOT` and 110 GB under `PLE_DIR`; after `FLASH-NEXT QUANT: OK` move `PLE_DIR` to
  `$B70_DATA/trash/<date>`.
- **Stages** (`run --only S` / `--from S` reruns one): `quant` (AutoRound, the ~2 days), `bf16` (every unquantised
  tensor bitwise the source's: reads the ~172 GB export and the same bytes of the source, ~30-40 min, estimated),
  `head` (`lm_head` packed by `loader/rtn.h`'s rule and declared 4-bit, a minute), `ple` (`ple_int8.py`, reads
  102 GB: ~30 min, estimated), `check` (`check.py` + `check_gptq_export.py`, minutes); the last line
  `FLASH-NEXT QUANT: OK <export>` or `FAIL`, also in `RESULT`.
- **Resume:** AutoRound's per-block checkpoint (`AR_RESUME_DIR`): each finished block is written to the export and
  the chained activations to `resume/`; a new process with the same settings logs "ResumeState: resuming after k/48
  already-quantized blocks" and continues (proven on the tiny model: stopped after block 0, `start` resumed at block 1
  and ended `OK`). `start` on an unfinished run resumes it; a finished one is refused (a new run needs `OUT=<dir>`).
- **Guards:** `start` refuses unless `check-download` passes, MemAvailable >= 100 GB (`RAM_MIN_GB`) and the disks have
  room; AutoRound runs inside `systemd-run --user --scope -p MemoryMax=115G -p MemorySwapMax=0`, so running out of
  RAM kills the run (then `start` resumes it), never the machine.

### What the AutoRound of `~/auto-round` needed (each found on the tiny model or the real-width probe)

Our clone `~/auto-round` (branch `local-patches-v2`; rebased 2026-11-13 to upstream main `42db5d3a` + one
local commit, head `8186d8b6`; before that `b58069ea` + `a6cf7487`, now the branch `pin-a6cf7487-2026-11-13`): the local commit's
`calib_dataset.py` reads OpenCodeInstruct's cached parquet shards (`_load_cached_parquet_dataset`, before
its hub fallback) and the model loaders try the local snapshot first. The script builds the branch head through
`ar_latest.sh` + `ar_pin.sh` (`AR_COMMIT`, `AR_PATCH=0`, `AR_REPO=~/auto-round`: the commit is only in that clone). The table below was measured on `a6cf7487`. Against the Qwen3.8
scripts' 0.17.0 `6afaecdb`:

| | 0.17.0 `6afaecdb` | `a6cf7487` (`b58069ea` 2026-11-11 + the local commit) |
|---|---|---|
| per-sample BOS/EOS reset (`auto-round-local.patch`, `calib_dataset.py`) | needs the patch | upstream (#2591, `b3d11504`); the patch no longer applies (both hunks fail) |
| `WrapperLinear.in_features` (the patch's `wrapper.py`) | needs the patch | not read on the compressors / sign-round path; not needed (tiny runs) |
| resume after a stop | none | `AR_RESUME_DIR` per block (`utils/resume.py`) |
| the 12 QSA layers | transformers' per-token indexer loop | vectorised indexer (#2579) |
| fused experts (`mlp.experts.gate_up_proj [512, 1280, 2560]`) | the same `modeling/fused_moe` code (not run here) | "Fused-MoE checkpoint detected: building a meta skeleton and materializing weights per block"; exported per expert |
| the MTP head (`mtp.*`, not a transformers module) | has `AR_DISABLE_COPY_MTP_WEIGHTS` (not run here) | `missing_tensors.py`: copied, fused experts split per expert, every linear not ignored RTN int4 g64, the rest declared bits 16 in `extra_config` - no pass of our own; ours ignores the head's attention and shared expert (spec 21-QFN-v2 §2) |
| Qwen4's predefined ignore list | `hyper_connection`, `mlp.gate`, `shared_expert` | same - ours drops `shared_expert` (our shared experts are int4, spec 21-QFN-v1 §5 "Ours"; `autoround_run.py`) |

What the script does about the rest (`autoround_run.py`, `head_rtn.py`):

- **No `quant_lm_head`.** A quantised layer outside the blocks makes AutoRound switch low CPU memory off
  (`compressors/base.py`: "`low_cpu_mem_usage` is not fully supported when there are quantized layers outside
  blocks"): every finished block would stay resident (the whole model by the end) and the resume is disabled ("
  AR_RESUME_DIR is set but neither immediate saving nor low_cpu_mem_usage is active", seen on the probe). AutoRound
  only RTNs the head anyway, and at this commit with **BF16 scales** the loader refuses (tiny run). The `head` stage
  packs it by the engine's rule and declares it (`extra_config["lm_head"]` bits 4 g64 sym).
- **The PLE table file-backed.** Materialising layer 1, AutoRound reads all 128 `ngram_embedding.shard_*` tensors
  and `torch.cat`s them (`utils/disk_stream_util.py` `_assemble_sharded_tensor`): ~95 GiB of shards + a 95 GiB
  result on a 121 GiB machine - the run would die at layer 1. `autoround_run.py` replaces that one function for the
  table: the shards are streamed once into a raw bf16 file in `PLE_DIR` and the table is `torch.from_file` of it
  (mmap), so only the rows the calibration tokens hit are read and the kernel may drop them again. On an SSD
  because the first lookups are millions of small random reads (an hour-scale stall on the HDD, estimated). The
  `bf16` stage proves the exported table bitwise the source's (tiny runs: 255 tensors).
- **The CPU path** (`DEVICE=cpu`, the tiny model only) needs `AR_DISABLE_META_LOAD=1` (the script sets it): with the
  per-block meta load AutoRound turns amp off on a CPU without bf16 but materialises the ignored layers' weights in
  bf16 ("expected m1 and m2 to have the same dtype"); and it writes the unquantised tensors as fp32, which the
  `bf16` stage restores (222 of 254 on the tiny model).

Other settings: transformers **5.19.0** (the reference's pin; `SITE`, installed `--no-deps` beside the venv), the
venv `~/auto-round/.venv` (python 3.12, torch 2.14.0+xpu), `TORCHINDUCTOR_COMPILE_THREADS=4`, offline hub and
datasets (the network can drop). **Calibration data:** `DATASET` = `nvidia/OpenCodeInstruct:apply_chat_template`
(AutoRound's registered loader `get_opencode_instruct_dataset`: 128 x 100 rows drawn, each a user turn + the
assistant's answer through Qwen's template, ending in `<|im_end|>`, packed to 2048-token rows with the per-sample
BOS/EOS reset of #2591); the 50 parquet shards come from the HF cache
(`~/.cache/huggingface/hub/datasets--nvidia--OpenCodeInstruct`, 6.4 GB, on the SSD). `autoround_run.py`
refuses to start unless all 50 are there, and offline the fork's hub fallback is an error too, so the run cannot
fall back to pile-10k: with an empty cache it exits "calibration data: nvidia/OpenCodeInstruct is not in the HF
cache ... refusing to run" (checked). AutoRound takes the multimodal path for
`Qwen4ExpForConditionalGeneration` and, with a text dataset and `quant_nontext_module` off, calibrates through the
standard text loader; the vision tower stays bf16 (the engine never loads it; `check_gptq_export.py --skip-prefix
model.visual.`).

### Proven on the tiny model (2026-11-11 evening, test machine)

`make_tiny_vl.py`: `qwen4exp_make_tiny.py`'s fused `bf16` form (hidden 128, layers lllq, 16 experts top-4, the PLE at
layer 1, the MTP head) + a 1-block vision tower + the tokenizer (qikp's tiny) + the processor files (Intel's
snapshot until the original's are downloaded). The downloaded qikp tiny itself cannot be used: hidden 16, so no
linear is a multiple of 64, and per-expert (not fused) experts. `selftest` (2 iterations, 8 x 128 rows):

| | card (XPU) | test-machine CPU (`DEVICE=cpu`) |
|---|---|---|
| `quant` | ~70 s, peak RSS 5.7 GiB; meta skeleton; 218 of 263 linears quantised; the PLE table from the file | 80 s, peak RSS 3.0 GiB; full load |
| per-expert int4 | 64 routed experts (4 layers x 16), every shared expert, QSA q/k/v/o, GDN qkv / z / out | same |
| MTP head | 16 experts x 3 RTN int4 g64, its attention and shared expert bf16 | same |
| `bf16` | 255 unquantised tensors bitwise the source's (the PLE table's 128 shards included) | 254, of which 222 restored from fp32 |
| `head` | `lm_head [248320][128]` packed, scales F16, declared 4-bit | same |
| `ple` / `check` | 16 heads; `check.py` ACCEPTED with the PLE file; `check_gptq_export.py` VERIFIED B70-READY | same |
| result | `FLASH-NEXT QUANT: OK` | `FLASH-NEXT QUANT: OK` |

Logs in `$B70_DATA/logs/flash-next-quant/`: `selftest-cpu-opencode.log` (the final code: AutoRound `a6cf7487`,
OpenCodeInstruct from the cache's 50 shards, 800 rows drawn for 8 x 128), `selftest-xpu-3.log` (card 1, the same
code with pile-10k), the resume test
in `tiny-resume/` (100 iterations, 128 x 2048 rows: stopped after block 0, the
second `start` resumed at block 1, `OK`).

### Sizing (2026-11-11, a real-width stand-in on the test machine)

The download had 4 of 131 shards, so the measurement used a stand-in at the real widths: 4 layers (GDN, GDN with a
small PLE, GDN, QSA), hidden 2560, 512 experts top-10 of width 640, 24 q heads, random bf16 weights
(`qwen4exp_make_tiny.make`, 23.4 GB), quantised with the run's settings (200 iterations, 128 x 2048, batch 8) on
card 1 (calibrated on pile-10k then; the data does not change a block's cost - the same 128 x 2048 tokens):

| | measured |
|---|---|
| block 0 (GDN, 512 experts), tuned (`measure-20261111-195942.log`) | **3441 s = 57 min** (incl. ~4 min of loading, caching 128 x 2048 rows and torch.compile); loss 4.23e-3 -> 1.66e-3 (best at iteration 177) |
| AutoRound's MoE path | "grouped tuning needs 150.7 GiB on xpu:0 (free 15.4) -- switching this run to linear_loop": the 512 experts run one after another |
| peak VRAM (`torch.xpu.memory_reserved`) | **38.88 GiB on a 31.89 GiB card** |
| peak RAM | 70 GB - on the probe's path only: a run that loads a few layers (AutoRound's debug option) skips the meta skeleton and holds the whole stand-in, unfused, in RAM |
| `resume/` | 21.5 GB after block 0 (2 x 10.74 GB fp32), overwritten per block, not growing (`utils/resume.py`) |

**The VRAM peak (read from AutoRound's `tune_memory.py`, not run again):** the block's tuning state is ~14 bytes for
each of its 2.52 G expert weights (the rounding value 4 + its gradient 4 + a bf16 copy 2 + the best-iterate
snapshot 4; AutoRound printed `state 14.5` = 6/14 of it = 33.8 GiB in all) plus the bf16 weights - ~39 GiB, more
than one card has, whatever the batch: `batch_size`, `nsamples` and `low_gpu_mem_usage` (on: the calibration
activations live on the CPU) only move the activations, not the state. On Xe the kernel driver lets the allocation
exceed VRAM and evicts the overflow to host RAM (the Xe hazard of spec 21-QFN-v2 §8), so the run works and is slowed by
it by an amount not measured; it is in the 57 min. **Two cards do not help:** `--device 0,1` (tried 2026-11-11 on
the stand-in) ran AutoRound's data-parallel path with the whole block on xpu:1 (32 GB resident, all the compute,
"Major device (xpu:1) has reached memory threshold") and xpu:0 at 7.6 GB idle; block 0 had not finished after 64 min (one card: 57); and
on the tiny model it fails at the first QSA block ("Expected all tensors to be on the same device, xpu:1 and xpu:0",
with and without torch.compile). So the recipe is one card (card 1).

**Projection (derived):** 48 blocks x ~57 min = ~46 h (the 12 QSA blocks are not measured; the 512 experts dominate
both kinds), + ~10 min for the PLE file, + the export of the MTP head and the `bf16` / `head` / `ple` / `check`
stages ~1.5 h (estimated): **~48 h on card 1**, e.g. Friday evening to Sunday evening. RAM on the real path
(estimated): the activations 21.5 GB + the next block's 10.7 GB + one block's weights and their unfused copies
~10 GB + ~10 GB of runtime, ~55 GB, plus the PLE table's touched pages (page cache, reclaimable) - under the 115 GB
cap.

**Re-measure on the real model:** `start`, then `status` after the first block (~1 h): AutoRound's bar
prints the seconds per block and its own time left; the `-- seconds per finished block` line lists each one. A
block far above ~60 min, or RAM in `status` near 115 GB, is the moment to `stop` (the run resumes later).

## Qwen3.8-Flash-Next (spec 21-QFN-v1b): synthetic checkpoints, the PLE int8 file, the export check

`tools/quantize/qwen4exp/` holds what spec 21-QFN-v1's engine needs before (and beside) 21q's own AutoRound run.
Every script runs in `agnes-ref-img` with 21-QFN-v1a's transformers 5.19.0 site first on `PYTHONPATH`
(`tools/oracle/qwen4exp_env.sh`; on the Mac `oracle-out-q4exp/site`):

- `make_synth.py <out> --form ours|intel --tokenizer <the original's small files> [--layers 4]
  [--ple-base 1000] [--mtp] [--seed 0]` - a checkpoint at the REAL widths with random weights, every tensor
  named as `qwen4exp_ref.expected_names(tc, form)` names it (checked both ways at the end): the original's
  `config.json` at N layers (2..48; 4 holds the PLE layer 1 and the QSA layer 3) with a reduced PLE base
  (the same hash and gather over small primes), our form (int4 g64 experts / shared / dense) or Intel's
  (int4 g128 experts with F16 scales, the rest bf16), the PLE table as 128 bf16 shards plus the three I64
  tensors by the formula, `--mtp` the head (bf16, per-expert `.weight` experts). The packer is Kolibri's
  `pack_rtn_g64` at group g (`(q - 8) x scale`, qzeros 0x77777777). One layer at a time; ~1.4 GB a layer
  in either form + 2.54 GB of embedding and head (+ 5.2 GB with `--mtp`), derived; measured at N = 4 with
  `--mtp` on the Mac: 13.52 GB ours (1650 s), 13.68 GB Intel's form (748 s).
- `ple_int8.py <snapshot> <out> [--scale bf16|f32]` - decision 7's file: the checkpoint's 128 PLE shards
  re-cut PER HEAD (head h's rows `[offset_h, offset_h + prime_h)` are its own tensor, head-local row r =
  global row `offset_h + r`; no shard boundary is a head boundary on the real table) and quantised by
  spec 9's row rule (`s = max|w| / 127` fp32, bf16 rounds it once; `q = clamp(rne(w / s), -127, 127)`; a
  zero row s = 0): `ple.h<h>.q` I8 `[prime_h][160]`, `ple.h<h>.s` BF16 / F32 `[prime_h]`, the I64 tensors
  copied after checking them against the formula, one file per head + an index, metadata `{source:
  <repo>@<revision>, rule: row-int8-spec9, scale}`. Streams 65,536 rows at a time; the real table: 51.2 GB
  of q + 0.64 GB of bf16 scales out (derived). The engine reads `<snapshot>-ple-int8/` (only there),
  as does `qwen4exp_ref.py --ple int8:<dir>` (the engine-format reference).
- `check.py <ckpt> [--ple <dir>]` - would the engine accept it: `quantization_config` (g64 ours, g128
  Intel's), names = `expected_names` both ways, dtypes and shapes at the real widths, qzeros, finite
  scales, no g_idx, the I64 tensors = the formula; with `--ple` the file's heads, constants and sampled
  rows re-quantised from the source bit for bit. Prints `ACCEPTED`.
- `test_qwen4exp_quant.py` - the names both ways for both forms (and `--mtp`), check.py's refusals (a
  missing tensor, a wrong qzeros word, a g64 expert in Intel's form), the packer at g64 / g128 against
  dequant.py, gate rows first as the reference reads them, the PLE file's marker rows (each head's first /
  last row, a shard boundary inside a head, a zero row, both scale dtypes, byte-equal to 21-QFN-v1a's
  `write_ple_int8`), and the hash constants for the reduced base and 20,000,000 against transformers'
  builders. Small fixtures (16 experts, a 1024-row vocabulary): ~3-4 min in an 8 GB / 4-CPU container.

```sh
docker run --rm --memory 8g --cpus 4 -e PYTHONPATH=/ws/oracle-out-q4exp/site \
  -e TORCH_EXTENSIONS_DIR=/ws/oracle-out-q4exp/torch-ext -v "$PWD":/ws -w /ws agnes-ref-img:latest \
  python3 tools/quantize/qwen4exp/test_qwen4exp_quant.py
# a synthetic checkpoint, its PLE file beside it and the check - the layout qwen4exp_load_synth_*_test reads
# (validation: r31.synth writes oracle-out-q4exp-synth/{ours,intel}/ckpt and ckpt-ple-int8 under $DATA)
python3 tools/quantize/qwen4exp/make_synth.py oracle-out-q4exp-synth/ours/ckpt --form ours --mtp \
  --tokenizer oracle-out-q4exp/orig-small
python3 tools/quantize/qwen4exp/ple_int8.py oracle-out-q4exp-synth/ours/ckpt oracle-out-q4exp-synth/ours/ckpt-ple-int8
python3 tools/quantize/qwen4exp/check.py oracle-out-q4exp-synth/ours/ckpt --ple oracle-out-q4exp-synth/ours/ckpt-ple-int8
```

## The DFlash 2 drafter (`tools/quantize/quantize_dflash.sh`, the open-work list "DFlash drafter quant")

`z-lab/Qwen3.8-27B-DFlash2` -> W4A16 g64 symmetric `auto_round:auto_gptq`, tuned (200 iterations, 128 samples, seed
42, batch 8). The commands and stages are in the script's header and in the open-work list. What is particular:

- **The model AutoRound tunes** is `dflash/model.py`: `tools/oracle/dflash_ref.py`'s non-causal forward as torch
  modules, batched (`test_dflash_quant.py` checks it equal to `dflash_ref` / `dflash_accept.forward_blocks` in
  float32). Three block lists, tuned in order on their own cached inputs: `encoder` (`fc` + `hidden_norm`: the
  taps -> the context plane), `layers.0-4` (the context plane is an input of every layer: `k_proj` / `v_proj` project
  it too), `candidate_selector` (`hidden_projection` of the final-norm mask rows). So `fc` and the selector projection
  are tuned like the layers; norms, conv base kernels and the codebooks are plain parameters and stay as loaded.
  `encoder.*` is the one name not in the checkpoint; `dflash/finalize.py` maps it back.
- **Samples** (`dflash/calib.py`): one recorded request and 32 anchors from its continuation, each a block of 8
  (anchor + 7 mask rows, positions p .. p + 7), over the quantised target's taps at positions [min p - 2047, max p)
  (2304 rows, padded). The block rows are the target's embedding rows (or `mask_embedding.pt`) times
  `input_embedding_scale`. RoPE tables are float64-computed float32 values carried as int32 bits: AutoRound casts
  every floating block input to bf16. AutoRound gets a stub tokenizer (it requires one for a model object; every row
  is valid, no pad id).
- **torch.compile is off** (`--torch-compile` turns it on): on the tiny drafter (test-machine CPU, 50 iterations) the layer
  blocks' backward fails under it ("modified by an inplace operation", a [hidden] float tensor); the encoder block
  passes. Not chased further: the drafter is small.
- **Tiny-drafter smoke (2026-11-13, test-machine CPU, 4 threads):** 20/20 linears quantised; 50 iterations cut the blocks'
  losses 4.3e-3 -> 1.9e-3 (encoder), 5.3e-2 -> 6.8e-3 and 5.7e-2 -> 9.6e-3 (layers), 7.8e-3 -> 1.8e-3 (selector); the
  export dequantised through the calibration forward is 9.5 % off the bf16 selector projection against 24 % for the
  2-iteration (RTN-start) export; `finalize` restored 16 fp32 widenings to the source's bf16 and the check VERIFIED.
- **Real-drafter smoke (2026-11-13, card 0, `quantize_dflash.sh start`):** two opencode-session-int8 requests (taps
  for 65 + 2358 positions), 4 samples x 32 blocks, 2 iterations: `DFLASH QUANT: OK`, 47/47 linears packed, 34 tensors
  bitwise the source's (22 norms, 10 conv base kernels, 2 codebooks); AutoRound 88 s, peak 9.8 GiB RAM, 7.3 GiB
  VRAM. The layer blocks' losses (5e1 .. 7e3) are MSEs of the residual stream, which grows layer by layer.
  The `taps` stage needs the whole build (the probe loads the engine's kernel binaries); it builds it when the probe
  is missing.

## Nemotron 3.5 Super: our AutoRound export (`tools/quantize/quantize_nemotron.sh`)

`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` (249 GB, OpenMDW-1.1) -> int4 g64 symmetric,
`auto_round:auto_gptq`, tuned (200 iterations, 128 x 2048 rows of nvidia/OpenCodeInstruct through the chat template,
batch 8), text only, with the MTP layer. Nemotron 3.5 Bolt (Super compressed: the same 88 blocks, 40 Mamba-2, 40
latent MoE, 8 NoPE attention) had no quantisation tool in the repo; the reference's Mamba ops and the export check
are restored from its tools (commit 55e8923f) and adapted to Super (512 x 2688 experts in every MoE block, top-22,
the MTP layer, the native `nemotron_h` config).

```sh
tools/quantize/quantize_nemotron.sh check-download   # DOWNLOAD OK ... (--sha: also hashes)
tools/quantize/quantize_nemotron.sh start            # detached, card 1; refuses unless download, RAM (80 GB) and disk (150 GB) are ready
tools/quantize/quantize_nemotron.sh status           # blocks done / 88, s per block, RAM, disk, the result
tools/quantize/quantize_nemotron.sh stop             # TERM to the run's process group; start resumes at the first unfinished block
tools/quantize/quantize_nemotron.sh selftest         # the whole pipeline on a tiny model on the CPU, then the reference
```

- **What is int4:** every block's routed experts, shared expert, `fc1/fc2_latent_proj`, Mamba `in_proj` / `out_proj`,
  attention `q/k/v/o_proj`. **Kept as the source ships them:** routers (BF16) and their F32 bias, `conv1d`, norms,
  A_log / D / dt_bias (BF16), the embedding, `lm_head` (no `quant_lm_head`: it would turn low CPU memory and the
  resume off; the engine builds its int8 head at load). The vision tower and projector are dropped (`strip`).
- **The MTP layer** (`language_model.mtp.layers.{0,1}.*`) is not a transformers module (the omni wrapper and the
  native class ignore `mtp.*` on load), so AutoRound never tunes it. At export its `missing_tensors.py` copies it
  from the source and RTNs every linear the config does not exclude; `autoround_run.py` excludes eh_proj, the
  attention, the shared expert and the latent projections (bits 16 in `extra_config`), so its 512 routed experts
  are int4 g64 (RTN) and the rest bf16. `check.py` refuses an export without every MTP tensor.
- **Names:** AutoRound writes the trunk back under the source's `language_model.backbone.*` (its `extra_config`
  entries spell `language_model.model.*`); the experts per expert; the MTP layer under its source names in
  `model_extra_tensors.safetensors`.
- **Stages:** `quant` (AutoRound, meta skeleton, one block of the original at a time, per-block resume),
  `strip` (vision out; AutoRound's own rename of that tower back to checkpoint names is lossy anyway),
  `scales` (the Mamba `out_proj` scales AutoRound keeps BF16 for this family -> F16 exactly), `bf16` (every
  unquantised tensor bitwise the source's; fp32 widenings restored), `check` (`nemotron/check.py`).
- **The CPU reference** `tools/oracle/nemotron_ref.py` (`run` / `compare` / `hfcheck`) reads the original or the export
  layer-major from the safetensors (no transformers model; a few GB of RAM for the 249 GB original), MTP drafts
  included; the selftest holds it against transformers' native class (fp32) and runs the tiny's bf16 original
  against its int4 export.
- **Proven on the tiny model** (2026-11-20, test-machine CPU, `selftest-20261120-050920.log` in `$B70_DATA/logs/nemotron-quant/`):
  `nemotron_make_tiny.py` (hidden 256, blocks mamba / moe / attention / moe, 16 experts top-4, the MTP layer, a
  one-block vision tower, the real tokenizer and remote code); AutoRound on the meta-skeleton path in 91 s, peak
  RSS 1.6 GiB; 110 int4 linears (32 of them the MTP layer's routed experts, RTN); `strip` 28 vision tensors;
  `scales` 1 BF16 out_proj scale tensor -> F16; `bf16` 33 tensors bitwise the source's (15 restored from fp32);
  `check` ACCEPTED; `NEMOTRON QUANT: OK`. Reference: `hfcheck` against transformers fp32 max |diff| 5.6e-7 of the
  largest logit, top-1 1.000; bf16 vs int4 export logits mean cosine 0.985, MTP logits 0.992, the same 4 greedy tokens.
- **The real run (derived, not measured):** 40 MoE blocks of 512 experts (2.82 G expert weights each, Flash-Next's
  2.52 G block took 57 min tuned on one card, the expert loop dominating) + 48 small blocks: **~45-65 h on one card**
  - measure the first MoE block (block 1) with `status` and extrapolate. VRAM: the MoE block's tuning state
  (~14 B a weight) + its bf16 weights ~45 GB, over the 31.9 GB card (Flash-Next ran at 38.9 GB with the Xe driver
  evicting to host RAM). RAM ~50 GB (activations 2 x 4.3 GB, a block and its unfused copy 11 GB, the spill);
  `start` waits for MemAvailable >= 80 GB, so not beside the Flash-Next run. Disk: export ~68 GB + resume 8.6 GB.

## Ripple-65B Preview (`tools/quantize/quantize_ripple.sh`)

`ricursiveAI/Ripple-65B-Preview` (bf16, 130 GB, remote code `modeling_ricursive.py`, pinned snapshot `d58dbcf4`) ->
W4A16 g64 symmetric `auto_round:auto_gptq`, tuned (200 iterations, 128 x 2048 rows of nvidia/OpenCodeInstruct
through the chat template, batch 8, seed 42) - the Flash-Next recipe. `lm_head` and the embedding stay bf16 (the
engine builds its int8 head at load). The commands and stages are in the script's header:

```sh
tools/quantize/quantize_ripple.sh selftest          # tiny random Ripple, test-machine CPU, both attention arms, ~5 min
tools/quantize/quantize_ripple.sh start             # detached, card 1; refuses unless data, RAM and disk are ready
tools/quantize/quantize_ripple.sh status            # blocks done / 80, s per block, RAM, the result
tools/quantize/quantize_ripple.sh stop              # TERM to the run's process group; start resumes
ATTN=bf16 tools/quantize/quantize_ripple.sh start   # attention bf16, only the MLP int4
```

What is particular to Ripple (`ripple/autoround_run.py`):

- **Low CPU memory for a dense model.** AutoRound builds its meta skeleton by itself only for fused-MoE
  checkpoints and would load a dense model whole (130 GB > the test machine's RAM). `AR_DISK_STREAM_MODEL=1` forces the
  skeleton: each block is read from the checkpoint when it is tuned, written and freed; the run refuses if
  AutoRound did not take that path.
- **Value residuals need no shim.** Every layer gets the same tensor, the scaled embedding rows, as the
  `value_residual` keyword. AutoRound caches the first block's tensor keywords per sample and replays them into
  every block (only the hidden states are chained; a resume re-captures them from block 0). A forward pre-hook on
  every decoder layer refuses a call without it - the remote code would silently fall back to the normed input.
  The hook is one function for all layers and records nothing while compiling: a per-layer closure made every block
  a new graph and hit dynamo's recompile limit (16) on the tiny model's fifth block.
- **Masks.** AutoRound masks each row's last token, so the remote code's `_update_causal_mask` builds a 4-D causal
  mask; the sliding layers cut it to the window (the tiny model's window 32 binds at 128-token rows as the real
  1024 does at 2048), the NoPE layers use it as is.
- **On the CPU** (`DEVICE=cpu`, the selftest) a streamed block comes out bf16 while AutoRound computes in fp32
  ("expected m1 and m2 to have the same dtype"); `autoround_run.py` casts each materialised block to the compute
  dtype there, and the `bf16` stage restores the unquantised tensors to the source's bytes.
- **`ATTN=bf16`** puts `self_attn` in AutoRound's ignore list: q / k / v / o / g ship as bf16 `.weight` with
  `extra_config` bits 16, which the loader accepts today. int8 attention would be this export with the engine
  quantising those tensors to int8 at load, as it does for `lm_head`: the loader takes int4 and bf16 linears only.
- **The export keeps the source's bytes** for the remote code, tokenizer files, chat template and
  `generation_config.json` (AutoRound re-serialises the tokenizer and writes its own generation config).
  `ripple/check.py` checks them, every source config key and the attention arm; in the selftest (`RELOAD=1`) it
  also loads the export with transformers + trust_remote_code and compares it with the engine's dequantisation.

Selftest (2026-11-20, test-machine CPU, AutoRound `8186d8b6`, transformers 5.19.0, 10 iterations on 8 x 128 OpenCodeInstruct
rows; `ripple/make_tiny.py`: hidden 256, layers 0-4 of the real pattern, the 78 tensor names equal to the real
index's; `$B70_DATA/logs/ripple-quant/selftest-20261120-043938.log`): `RIPPLE SELFTEST: OK both arms`, ~2 min per arm.

| | `ATTN=int4` | `ATTN=bf16` |
|---|---|---|
| quant | meta skeleton, 40/40 linears, block losses -10 % .. -47 %, peak RSS 1.3 GiB | 15/40 (the MLP), 26 bf16 exclusions |
| bf16 | 38 unquantised tensors bitwise the source's (restored from fp32) | 63 |
| reload vs the engine's dequantisation | relative L2 1.5e-3, argmax 30/30 | 1.7e-3, 30/30 |
| `check_gptq_export.py` | VERIFIED B70-READY | VERIFIED B70-READY |

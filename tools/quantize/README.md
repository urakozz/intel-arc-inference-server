# Quantisation provenance

The checkpoint every gate in this repo grades against -
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, int4 **symmetric, group_size 64**,
GPTQ export (`docs/03-models.md`) - was produced by AutoRound **with two local
patches that are not upstream**. They lived only as uncommitted edits in
`~/auto-round` on the box until 2026-09-22; `auto-round-local.patch` is that
diff, captured so the recipe survives the disk.

Checked against `origin/main` on 2026-09-22 (36 commits ahead of the box's
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
| `quantize_kolibri1.sh` | 0.17.0 @ `6afaecdb` | the operator's ruling of 2026-10-05; Kolibri keeps lm_head bf16, so the bug is not exercised |
| `quantize_qwen38_mxfp4.sh` | none - llm-compressor `model_free_ptq` | only borrows the auto-round venv's python |

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
(operator, 2026-10-06), which packs at export like the calibrated path. It holds the whole model in
RAM (~54 GB bf16 for the 27B; the box has ~128 GB) and has not run on the 27B yet - the checker
refuses the export if `lm_head` comes out unpacked. Not exercised on CPU: XPU; the 27B's GDN layers and MTP head (the
scripts' checker guards the real run). On a CPU without bf16, both versions write unpacked tensors
as F32 (the checker WARNs with `--source`); on the box's XPU they stay bf16.

**Reproducibility.** The published checkpoints were not made by 0.17.0, and re-running a script
now does not reproduce them byte for byte unless the version matches:
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` (the gate) by the patched box build (`~/auto-round`,
`141e4c99` plus `auto-round-local.patch`, `--format auto_gptq`, bf16 lm_head); the int4-lm_head RTN
checkpoint of docs/13 by 0.14.2 (`autoround_version` 0.14.2, `tests/loader/quant_test.cc`'s
fixture); `urakozz/Ornith-1.5-35B-A3B-W4A16-g64-AutoRound-GPTQ` by **0.15.0** (its config's
`autoround_version`); Agnes and K2 by 0.16.0 (docs/probe-agnes-2026-10-03.md,
docs/probe-k2-2026-10-05.md). On the tiny models the 0.14.2 and 0.17.0 tuned exports are
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

Also present in 0.16.0 and relevant to the W4A8 probe
(`docs/superpowers/specs/2026-09-22-w4a8-dpas-probe-design.md`): `--act_bits`,
`--act_group_size`, `--act_data_type`, `--disable_act_dynamic`.

## Kolibri-1 (spec 20 §3.1): `tools/quantize_kolibri1.sh`

One command for `Aleph-Alpha/Kolibri-1-BF16` -> `urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ`, with
the recipe of spec 20 §3.1 encoded stage by stage (helpers in `kolibri/`, facts and checks in
`docs/probe-kolibri-2026-10-05.md`):

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

**AutoRound for Kolibri is NOT the Qwen3.8 pin.** The operator's ruling (2026-10-05): intel/auto-round
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
box: `DEVICE=xpu` for AutoRound works in principle; `calib` cannot run there at a useful speed.

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

## Qwen3.8-Flash-Next (spec 21b): synthetic checkpoints, the PLE int8 file, the export check

`tools/quantize/qwen4exp/` holds what spec 21's engine needs before (and beside) 21q's own AutoRound run.
Every script runs in `agnes-ref-img` with 21a's transformers 5.19.0 site first on `PYTHONPATH`
(`tools/oracle/qwen4exp_env.sh`; on the Mac `oracle-out-q4exp/site`):

- `make_synth.py <out> --form ours|intel --tokenizer <the original's small files> [--layers 4]
  [--ple-base 1000] [--mtp] [--seed 0]` - a checkpoint at the REAL widths with random weights, every tensor
  named as `qwen4exp_ref.expected_names(tc, form)` names it (checked both ways at the end): the original's
  `config.json` at N layers (2..48; 4 holds the PLE layer 1 and the QSA layer 3) with a reduced PLE base
  (the same hash and gather over small primes), our form (int4 g64 experts / shared / dense) or Intel's
  (int4 g128 experts with F16 scales, the rest bf16), the PLE table as 128 bf16 shards plus the three I64
  tensors by the formula, `--mtp` the head (bf16, per-expert `.weight` experts). The packer is Kolibri's
  `pack_rtn_g64` at group g (`(q - 8) x scale`, qzeros 0x77777777). One layer at a time; ~1.4 GB a layer
  (Intel's; ~0.75 ours) + 2.54 GB of embedding and head (+ 5.2 GB with `--mtp`), derived.
- `ple_int8.py <snapshot> <out> [--scale bf16|f32]` - decision 7's file: the checkpoint's 128 PLE shards
  re-cut PER HEAD (head h's rows `[offset_h, offset_h + prime_h)` are its own tensor, head-local row r =
  global row `offset_h + r`; no shard boundary is a head boundary on the real table) and quantised by
  spec 9's row rule (`s = max|w| / 127` fp32, bf16 rounds it once; `q = clamp(rne(w / s), -127, 127)`; a
  zero row s = 0): `ple.h<h>.q` I8 `[prime_h][160]`, `ple.h<h>.s` BF16 / F32 `[prime_h]`, the I64 tensors
  copied after checking them against the formula, one file per head + an index, metadata `{source:
  <repo>@<revision>, rule: row-int8-spec9, scale}`. Streams 65,536 rows at a time; the real table: 51.2 GB
  of q + 0.64 GB of bf16 scales out (derived). The engine reads `<snapshot>-ple-int8/` (or `$B70_Q4_PLE`),
  as does `qwen4exp_ref.py --ple int8:<dir>` (the engine-format reference).
- `check.py <ckpt> [--ple <dir>]` - would the engine accept it: `quantization_config` (g64 ours, g128
  Intel's), names = `expected_names` both ways, dtypes and shapes at the real widths, qzeros, finite
  scales, no g_idx, the I64 tensors = the formula; with `--ple` the file's heads, constants and sampled
  rows re-quantised from the source bit for bit. Prints `ACCEPTED`.
- `test_qwen4exp_quant.py` - the names both ways for both forms (and `--mtp`), check.py's refusals (a
  missing tensor, a wrong qzeros word, a g64 expert in Intel's form), the packer at g64 / g128 against
  dequant.py, gate rows first as the reference reads them, the PLE file's marker rows (each head's first /
  last row, a shard boundary inside a head, a zero row, both scale dtypes, byte-equal to 21a's
  `write_ple_int8`), and the hash constants for the reduced base and 20,000,000 against transformers'
  builders. Small fixtures (16 experts, a 1024-row vocabulary): ~3-4 min in an 8 GB / 4-CPU container.

```sh
docker run --rm --memory 8g --cpus 4 -e PYTHONPATH=/ws/oracle-out-q4exp/site \
  -e TORCH_EXTENSIONS_DIR=/ws/oracle-out-q4exp/torch-ext -v "$PWD":/ws -w /ws agnes-ref-img:latest \
  python3 tools/quantize/qwen4exp/test_qwen4exp_quant.py
# a synthetic checkpoint, its PLE file beside it and the check - the layout qwen4exp_load_synth_*_test reads
# (box: r31.synth writes oracle-out-q4exp-synth/{ours,intel}/ckpt and ckpt-ple-int8 under $DATA)
python3 tools/quantize/qwen4exp/make_synth.py oracle-out-q4exp-synth/ours/ckpt --form ours --mtp \
  --tokenizer oracle-out-q4exp/orig-small
python3 tools/quantize/qwen4exp/ple_int8.py oracle-out-q4exp-synth/ours/ckpt oracle-out-q4exp-synth/ours/ckpt-ple-int8
python3 tools/quantize/qwen4exp/check.py oracle-out-q4exp-synth/ours/ckpt --ple oracle-out-q4exp-synth/ours/ckpt-ple-int8
```

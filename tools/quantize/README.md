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
  calibration samples.
- **`wrapper.py`** - gives `WrapperLinear` `in_features` / `out_features`
  properties forwarding to the wrapped layer, which the pipeline queries and
  upstream does not expose.

Apply with `git apply tools/quantize/auto-round-local.patch` in an auto-round
checkout, or keep them on a local branch and rebase.

## The command

auto-round 0.16.0 dropped `--sym` (symmetric is the default; `--asym` opts out),
and `--device` is an argparse prefix of `--device_map`:

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

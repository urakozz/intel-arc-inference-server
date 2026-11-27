# Development: b70-decode

`b70-decode` is the engine without the server. It reads prompt token ids from a file, runs them through the same
prompt pass and decode step as `b70-serve`, and writes the generated ids to stdout, greedy. That makes a run exactly
reproducible: the tests and the reference comparisons use it, `--bench` prints a speed row, and `--profile` times
every launch of the decode step.

```
# 128 ids after a prompt, MTP on
./build/src/cli/b70-decode <snapshot-or-repo> --ids prompt.ids --prefill --n 128 --spec mtp

# a speed row: 256 ids generated after 4,096 ids of context
./build/src/cli/b70-decode <snapshot-or-repo> --bench --depth 4096 --tg 256
```

## Flags

The usual flags:

| flag | default | what it does |
|---|---|---|
| `--ids FILE` | - | prompt ids, whitespace-separated; generated ids go to stdout |
| `--n N` | - | ids to generate |
| `--prefill` | off | run the prompt through the chunked prefill |
| `--spec off\|mtp\|dflash`, `--spec-max K` | `off` | as for `b70-serve` ([README](../README.md#flags)) |
| `--max-len L\|auto`, `--parallel auto\|1\|pp\|tp`, `--kv-cache`, `--prefill-backend` | `16384`, ... | as for `b70-serve` ([README](../README.md#flags)) |
| `--layers N\|auto` | - | load only the first N layers (Kolibri-1, Qwen3.8-Flash-Next) |
| `--host-experts auto\|gpu\|cpu` | `auto` | as for `b70-serve` ([README](../README.md#flags)) |

Developer flags:

| flag | default | what it does |
|---|---|---|
| `--bench` | - | time `--tg` generated ids after `--depth` ids of context (or a `--prefill-length N` prompt); prints a markdown row |
| `--depth D` | `4096` | context before decoding, for `--bench` / `--profile` |
| `--prefill-length N` | - | `--bench`: prefill N ids and print the prefill row too |
| `--tg N` | `256` | ids to generate in `--bench` |
| `--prefill-chunk C` | `2048` | positions per prefill chunk, at most 4096 |
| `--profile`, `--steps N`, `--repeats R` | -, `32`, `1` | per-launch timing of the decode step |
| `--pipeline-split auto\|N` | `auto` | with `pp`: where the layers are cut |
| `--dump-taps FILE` | - | Inkling 2 Mini, one card: write the drafter's tap rows (the prompt's last 2048 positions and each generated one), the head's top 64 logits and their log-sum-exp, for an offline drafter acceptance measurement |
| `--plan` | off | print the memory plan from the checkpoint's headers (both cards' with `--parallel pp` or `tp`; with `--max-len`, `--spec`), then exit |

Environment variables (test switches, tool settings, the Hugging Face cache): [environment-variables.md](environment-variables.md).

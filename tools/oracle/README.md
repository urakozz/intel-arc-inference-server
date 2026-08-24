# The oracle

The engine has to be *right* before it is fast, and "right" needs a reference
that is not the engine. That reference is `transformers` 5.15 on CPU: the same
checkpoint, dequantised by the same rule the C++ loader is bit-compared against,
run through the pure-torch gated-delta-rule that ships in
`transformers/models/qwen3_5/modeling_qwen3_5.py`. It produces, per prompt, one
safetensors file of layer-by-layer activations, the GDN states after the prompt,
every logits row, and the greedy continuation. Plan 3's golden test compares the
engine against that file.

The reference image (`vllm-xpu-env-next-p314-t214-vxkp0:latest`, doc 10)
deliberately has **no `fla`**: the fallback path *is* the contract (doc 03).

## Files

| File | What it is |
|---|---|
| `dequant.py` | The one meaning of the int4 GPTQ bits: `(q - 8) * scale`, product in fp32, one RNE cast to bf16. Also writes `tests/golden/dequant_fixture.safetensors`, which the C++ loader test matches bit-exactly. |
| `tokenize.py` | `encode` a prompt file to ids / `decode` ids back. Raw text, no chat template, no special tokens. |
| `dump.py` | Builds `Qwen3_5ForCausalLM` from the config, loads a `dequant.py`-produced bf16 state dict `strict=True`, forwards the prompt with hooks, greedy-decodes, writes one safetensors file. |
| `run_in_container.sh` | Wraps `docker run` for the box: read-only HF cache at `/hf`, repo at `/ws`, `$SNAP` resolved to the snapshot directory, **`-u $(id -u):$(id -g)`** so outputs are not root-owned, no memory limit. |

Prompts live in `tests/golden/prompts/`: `prose.txt` (plain English),
`code.txt` (a Python function), `cjk.txt` (Chinese plus two emoji - multi-byte
tokens and the tail of `lm_head`). Each must tokenize to **24-64 ids**;
`dump.py --max-prompt` enforces the ceiling.

## Running it

Everything runs on the box; the Mac never loads the model.

```bash
tools/box.sh sync
tools/box.sh run "mkdir -p oracle-out && tools/oracle/run_in_container.sh '
  python3 tools/oracle/tokenize.py \"\$SNAP\" encode tests/golden/prompts/prose.txt > /ws/oracle-out/prose.ids'"
tools/box.sh run "tools/oracle/run_in_container.sh '
  python3 tools/oracle/dump.py \"\$SNAP\" --prompt /ws/oracle-out/prose.ids \
    --out /ws/oracle-out/prose.golden.safetensors --gen 32'"
```

A run takes tens of minutes, so launch it detached (`setsid nohup … > log 2>&1
< /dev/null &`) and tail the log rather than holding an ssh session open.

**Outputs stay on the box**, in `~/b70-inference-server/oracle-out/` - hundreds
of MB per prompt. `.gitignore` has `oracle-out/` and `tools/box.sh sync` passes
`--exclude oracle-out`, so a sync from the Mac neither deletes nor commits them.

### Exact commands and measured numbers

_(filled by Task 8: per-prompt id counts, wall time, peak RSS, output size.)_

| Prompt | ids | wall | peak RSS | output | first greedy tokens |
|---|---|---|---|---|---|
| prose | _(t8)_ | _(t8)_ | _(t8)_ | _(t8)_ | _(t8)_ |
| code | _(t8)_ | _(t8)_ | _(t8)_ | _(t8)_ | _(t8)_ |
| cjk | _(t8)_ | _(t8)_ | _(t8)_ | _(t8)_ | _(t8)_ |

## What is in a golden file

Batch is always 1 and the batch dimension is squeezed out.

| Name | dtype | shape | Meaning |
|---|---|---|---|
| `resid.L{0..63}` | bf16 | `[T, 5120]` | decoder layer output (the residual stream after the MLP add) |
| `mixer.L{0..63}` | bf16 | `[T, 5120]` | `linear_attn` / `self_attn` submodule output, before the residual add |
| `mlp.L{0..63}` | bf16 | `[T, 5120]` | `mlp` submodule output, before the residual add |
| `gdn_state.L{i}` | f32 | `[48, 128, 128]` | recurrent state after the prompt, the 48 GDN layers only |
| `conv_state.L{i}` | bf16 | `[10240, 4]` | conv window after the prompt, the 48 GDN layers only |
| `logits` | f32 | `[T + gen, 248320]` | every prompt position, then one row per generated step |
| `tokens` | i32 | `[gen]` | the greedy continuation |

`logits[t]` for `t < T` is prompt position `t`, and `tokens[0] =
argmax(logits[T-1])`. `logits[T+j]` is the forward that consumed `tokens[j]`, so
`tokens[j+1] = argmax(logits[T+j])`; the final row's argmax is deliberately not
in `tokens` - it is a free extra check for the engine.

File metadata carries the snapshot path, the prompt ids, `gen`, the attention
implementation and the group size.

`dump.py` aborts rather than writing a doubtful file if: the state dict does not
load `strict=True` (it prints the full missing/unexpected lists), a meta tensor
survives the load, `tokens` is not `--gen` long, a GDN layer contributed no
state, or `resid.L63` contains a NaN/Inf.

## The trust chain

_(written by Task 8: what the golden file is trusted against, and what a
three-way disagreement between the engine, the oracle and vLLM would mean -
the dequant convention being the first suspect.)_

## Decisions worth knowing

- **Model built from the config, not `from_pretrained`.** The checkpoint is
  GPTQ; `from_pretrained` would hand it to a quantisation backend and we would be
  testing that backend, not our own dequantisation. `dump.py` instead builds the
  module tree on the `meta` device and loads our state dict with
  `assign=True` - which also means the ~55 GB of bf16 weights exist once, not
  twice. The meta build leaves the non-persistent RoPE `inv_freq` buffer empty,
  so `dump.py` rebuilds `model.model.rotary_emb` on CPU and then asserts that no
  meta tensor survived.
- **`attn_implementation="eager"`**, because that path takes the softmax in fp32,
  which is the contract doc 03 records and the kernels implement.
- **GPTQ `qweight` is `[K/8, N]` = `[in/8, out]`**, so `dequant_gptq` returns
  `[in, out]` and `dump.py` transposes it into `nn.Linear`'s `[out, in]`.
- **GDN states are read straight after the prompt forward and cloned**: the
  decode loop mutates `cache.layers[i].recurrent_states[0]` in place.
- **`tokenize.py` shadows the stdlib `tokenize` module** that `inspect` (and so
  torch and transformers) import. Both scripts drop their own directory from
  `sys.path` before importing anything third-party; do the same in any new script
  in this directory, or run it with `python -P`.
- **Never give the container a memory limit** and run one prompt at a time; the
  box has 121 GB and the dequantised model wants a large fraction of it.

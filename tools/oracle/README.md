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

Everything runs on the box; the Mac never loads the model. Sync from the Mac,
then run the container commands **on the box** - wrapping them in
`tools/box.sh run` adds a third layer of quoting around `"$SNAP"` and is easy
to get wrong.

```bash
tools/box.sh sync                                  # on the Mac
ssh user@box                          # everything below: on the box
cd ~/b70-inference-server && mkdir -p oracle-out

# ids: seconds, tokenizer only
tools/oracle/run_in_container.sh 'python3 tools/oracle/tokenize.py "$SNAP" \
  encode tests/golden/prompts/prose.txt > /ws/oracle-out/prose.ids'

# the dump: ~5.5 min, ONE PROMPT AT A TIME (61 GiB peak, 121 GB box)
tools/oracle/run_in_container.sh 'python3 tools/oracle/dump.py "$SNAP" \
  --prompt /ws/oracle-out/prose.ids \
  --out /ws/oracle-out/prose.golden.safetensors --gen 32'
```

`dump.py` takes **one** `--prompt`/`--out` pair, so three prompts are three
container invocations and the 118 s load+dequant is paid three times. Task 8
drove them from a throwaway `for p in prose code cjk` loop in
`oracle-out/golden.sh`, launched detached (`setsid nohup … > log 2>&1 </dev/null &`)
so the 17-minute serial run outlives the ssh session, and tailed the log.
Making `dump.py` accept repeated pairs would save ~4 minutes per full re-run;
it was not worth touching the numerics path for.

**Outputs stay on the box**, in `~/b70-inference-server/oracle-out/` - hundreds
of MB per prompt. `.gitignore` has `oracle-out/` and `tools/box.sh sync` passes
`--exclude oracle-out`, so a sync from the Mac neither deletes nor commits them.

### Exact commands and measured numbers

Run 2026-08-24 21:42-21:58 on the box (`box`, 121 GB),
image `vllm-xpu-env-next-p314-t214-vxkp0:latest`, serially, `--gen 32` for all
three. The three commands were exactly the two-line `run_in_container.sh` form
above with `prose` replaced by `code` / `cjk`. Every number below is from the
per-prompt logs in `oracle-out/{prose,code,cjk}.log`.

| Prompt | ids | wall | load+dequant | prefill | 32 greedy | peak RSS | output | manifest |
|---|---|---|---|---|---|---|---|---|
| prose | 42 | 314.5 s | 117.9 s | 14.8 s | 181.7 s | 61.4 GiB | 296.6 MiB (311,030,384 B) | 290 tensors |
| code  | 61 | 324.9 s | 118.1 s | 19.0 s | 187.7 s | 61.5 GiB | 350.2 MiB (367,258,272 B) | 290 tensors |
| cjk   | 38 | 316.1 s | 117.7 s | 13.9 s | 184.4 s | 61.4 GiB | 285.3 MiB (299,192,968 B) | 290 tensors |

Wall is `dump.py`'s own (`time.time()`); the `docker run` wrapper adds ~17.5 s of
container start (5:32.0 / 5:42.4 / 5:33.7 measured by `/usr/bin/time`). Peak RSS
is `dump.py`'s own `getrusage(RUSAGE_SELF)` inside the container - do **not**
read the wrapper's 28 MB, which is the docker *client*. Serial total including
the reloads: 16 min 48 s. Each state dict is the same 851 tensors, 400
dequantised from int4, 50.10 GiB; each manifest is the same 290 tensors -
64 `resid` + 64 `mixer` + 64 `mlp` + 48 `gdn_state` + 48 `conv_state` +
`logits` + `tokens` - with only `T` (42/61/38) and the `logits` row count
(`T + 32` = 74/93/70) differing. Decode is ~5.8 s/token on CPU, memory-bound on
50 GB of bf16 weights.

The file sizes reconcile exactly against that manifest, which is a cheap check
that a golden file holds what it claims: prose = 70.1 MiB `logits` + 78.8 MiB
activations + 144.0 MiB GDN states + 3.8 MiB conv = 296.6 MiB; code =
88.1 + 114.4 + 144.0 + 3.8 = 350.2; cjk = 66.3 + 71.2 + 144.0 + 3.8 = 285.3.
Note that the 48 fp32 `[48,128,128]` recurrent states are ~half of every file
and do not depend on prompt length.

#### Sanity checks on the written files

Re-read by a separate process (`oracle-out/check.sh`), not the writer's own
claim: `tokens` length, distinctness, `resid.L63` finiteness, and the
continuation decoded back through `tokenize.py decode`.

```
=== prose  (311030384 bytes)
  tokens: len=32 distinct=24 degenerate=False
  ids: [3113, 7810, 279, 1118, 479, 654, 8980, 1000, 381, 1142, 440, 279, 1834, 725, 2213, 13,
        3113, 11292, 279, 4220, 6092, 1000, 381, 6992, 11, 321, 539, 5600, 279, 72103, 1000, 381]
  resid.L63: shape=(42, 5120) finite=True min=-175.0000 max=352.0000
  logits: shape=(74, 248320) finite=True min=-15.3750 max=25.5000
  metadata: n_prompt=42 gen=32 attn=eager group_size=64
  decoded:
  |  By eight the first trawlers would be back with the day’s catch. By nine the whole town would be moving, and by ten the harbour would be
  repr: ' By eight the first trawlers would be back with the day’s catch. By nine the whole town would be moving, and by ten the harbour would be\n'
=== code  (367258272 bytes)
  tokens: len=32 distinct=24 degenerate=False
  ids: [271, 727, 40523, 17, 19490, 11, 750, 11, 15131, 1590, 198, 262, 460, 498, 1030, 8474,
        3620, 11, 750, 681, 15131, 8, 364, 343, 303, 2663, 60, 271, 727, 40523, 18, 19490]
  resid.L63: shape=(61, 5120) finite=True min=-182.0000 max=592.0000
  logits: shape=(93, 248320) finite=True min=-16.3750 max=30.0000
  metadata: n_prompt=61 gen=32 attn=eager group_size=64
  decoded:
  |
  |
  | def clamp2(values, lo, hi):
  |     return [min(max(v, lo), hi) for v in values]
  |
  | def clamp3(values
  repr: '\n\ndef clamp2(values, lo, hi):\n    return [min(max(v, lo), hi) for v in values]\n\ndef clamp3(values\n'
=== cjk  (299192968 bytes)
  tokens: len=32 distinct=28 degenerate=False
  ids: [29545, 271, 95815, 108553, 97663, 108447, 96494, 3709, 98844, 95895, 97771, 95726,
        114183, 101650, 100700, 1710, 271, 550, 220, 99737, 96863, 271, 99737, 96863, 95761,
        105064, 97463, 95793, 100830, 98252, 96019, 115534]
  resid.L63: shape=(38, 5120) finite=True min=-214.0000 max=632.0000
  logits: shape=(70, 248320) finite=True min=-18.3750 max=23.3750
  metadata: n_prompt=38 gen=32 attn=eager group_size=64
  decoded:
  | ️
  |
  | 我站在船舷边，看着对岸的轮廓慢慢清晰。
  |
  | ## 渡轮
  |
  | 渡轮是这座岛和大陆之间最古老的
  repr: '️\n\n我站在船舷边，看着对岸的轮廓慢慢清晰。\n\n## 渡轮\n\n渡轮是这座岛和大陆之间最古老的\n'
```

The three continuations as text:

- **prose** - ` By eight the first trawlers would be back with the day’s catch. By nine the whole town would be moving, and by ten the harbour would be`
- **code** - a blank line, then `def clamp2(values, lo, hi):` / `    return [min(max(v, lo), hi) for v in values]`, then a blank line and `def clamp3(values` - the model continued the file with the next function, which is what a code prompt should do.
- **cjk** - `我站在船舷边，看着对岸的轮廓慢慢清晰。` then a `## 渡轮` heading and `渡轮是这座岛和大陆之间最古老的` - grammatical Chinese, and it opens by completing the trailing emoji's variation selector (`️`), which is exactly the multi-byte behaviour this prompt exists to pin.

24 / 24 / 28 distinct ids out of 32: none degenerate. `resid.L63` and `logits`
finite in all three. `dump.py`'s own aborts (strict load, no surviving meta
tensor, `tokens` length, every GDN layer's state present, `resid.L63` finite)
also passed inside each run, or nothing would have been written.

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

Nothing here is trusted absolutely. Each link is trusted only against the next
one, and it is worth being explicit about where the chain currently ends.

1. **The bits.** `dequant.py` fixes the one meaning of the int4 nibbles -
   `(q - 8) * scale`, product in fp32, one RNE cast to bf16 - and the C++
   loader test matches it **bit-exactly** on the committed fixture. Loader and
   oracle therefore start from identical bf16 weights by construction, not by
   agreement.
2. **The math.** The oracle runs those weights through `transformers` 5.15's
   pure-torch `modeling_qwen3_5.py` on CPU, `attn_implementation="eager"` (fp32
   softmax), deliberately without `fla`. That reference implementation - not
   our kernels - decides what the correct activations are.
3. **The engine.** **Plan 3's golden test compares the engine against THESE
   files**: `oracle-out/{prose,code,cjk}.golden.safetensors` on the box, the
   ones produced by the run recorded above. Per-layer `resid`/`mixer`/`mlp`,
   the GDN and conv states after the prompt, the logits rows and the 32 greedy
   token ids. Their metadata pins the snapshot path, the prompt ids, `gen`, the
   attention implementation and the group size, so a golden file always says
   what it is a golden file *of*. Regenerate and the comparison target changes;
   that is the point of recording the exact numbers above.

What that chain does **not** yet prove is that the checkpoint was unpacked the
way its author packed it. Link 1 is exactly what makes link 3 meaningful and
also what limits it: the C++ loader and the oracle are pinned to the *same*
`dequant.py` convention (bit-exactly, by test), so a convention that is wrong
moves the engine and the oracle **together** and the golden test still passes. **The cross-check against vLLM is deferred to plan 3 deliberately** -
it needs a working engine to be worth running, and doing it now would only
compare two CPU paths that already share their input.

When it does run, read a three-way disagreement like this:

- **engine ≠ oracle** → an engine bug (kernel, layout, fusion, fp32 accumulation
  boundary). The oracle is the reference; the engine is wrong.
- **engine == oracle, both ≠ vLLM's greedy output** → **the dequant convention
  is the common suspect**, because it is the one thing the engine and the oracle
  share and vLLM does not. Check `dequant.py` first - zero-point handling
  (`q - 8` vs. an explicit `qzeros`), the group axis, the `[K/8, N]` packing
  order and nibble ordering, `desc_act`/`g_idx` - before touching a kernel.
- **all three agree** → the chain is closed.

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

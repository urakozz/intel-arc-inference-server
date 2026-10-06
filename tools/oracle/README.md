# The oracle

The engine has to be *right* before it is fast, and "right" needs a reference
that is not the engine. That reference is `transformers` 5.15 on CPU: the same
checkpoint, dequantised by the same rule the C++ loader is bit-compared against,
run through the pure-torch gated-delta-rule that ships in
`transformers/models/qwen3_5/modeling_qwen3_5.py`. It produces, per prompt, one
safetensors file of layer-by-layer activations, the GDN states after the prompt,
every logits row, and the greedy continuation. Plan 3's golden test compares the
engine against that file.

The reference image (`vllm-xpu-env-next-p314-t215-vxkp0:latest` since
2026-09-09; `…-t214-…` before that, and the sets dumped under it keep that
label) deliberately has **no `fla`**: the fallback path *is* the contract
(doc 03).

## Files

| File | What it is |
|---|---|
| `dequant.py` | The one meaning of the int4 GPTQ bits: `(q - 8) * scale`, product in fp32, one RNE cast to bf16. Also writes `tests/golden/dequant_fixture.safetensors`, which the C++ loader test matches bit-exactly. |
| `tokenize.py` | `encode` a prompt file to ids / `decode` ids back. Raw text, no chat template, no special tokens - except `encode --bos` for a tokenizer that prepends its BOS (K2-Horizon, spec 18a), which checks that BOS is the only token added. |
| `dump.py` | Builds `Qwen3_5ForCausalLM` from the config, loads a `dequant.py`-produced bf16 state dict `strict=True`, forwards the prompt with hooks, greedy-decodes, writes one safetensors file. |
| `run_in_container.sh` | Wraps `docker run` for the box: read-only HF cache at `/hf`, repo at `/ws`, `$SNAP` resolved to the snapshot directory, **`-u $(id -u):$(id -g)`** so outputs are not root-owned, no memory limit. |
| `golden.sh` | The production run: the three prompts, serially, `--gen 32`. This is the script that made the files plan 3 compares against - committed rather than retyped. `PROMPTS` names which sets to build; `--max-prompt 4096` since spec 2. |
| `make_long_prompt.sh` | Builds `tests/golden/prompts/long.ids` (2820 ids) and `long.txt` from the three committed prompts, on the Mac, with no model and no tokenizer. Spec 2 §6.2's ≥ 2048-id prompt. |
| `check.sh` | Re-reads the three written files in a separate process and prints the block quoted under "Sanity checks" below. |
| `stream.py` | Layer-streamed weights for `dump.py --stream` / `mtp_ref.py --dump --stream` (spec 14, the Mac path below): the model on `meta`, embed/norm/lm_head resident, each decoder layer materialised by a forward pre-hook and dropped after its forward; plus a bit-exact C++ `dequant_t` and the single-thread grouped conv1d. |
| `dflash_ref.py` | Spec 19a: the CPU reference of the DFlash / DFlash 2 drafters (config reader, drafter + target embed/head loader incl. the W4A16 drafter, `context_kv`, `draft_block`); `test_dflash_ref.py` checks it on tiny random weights. "The DFlash reference" below. |
| `dump_taps.py` | Spec 19a Task 2: the Qwen3.8 bf16 target (layer-streamed, as `kv_int8_probe.py run` builds it) over [prompt + recorded greedy continuation]; per source the residual after the drafter's tap layers (5, 19, 33, 47, 61), the greedy next ids and the top-64 logits. `test_dump_taps.py`: Review Focus 1 (tap i = layer i's OUTPUT) on a tiny checkpoint. "The DFlash P0" below. |
| `dflash_accept.py` | Spec 19a Task 3: teacher-forced acceptance of the DFlash 2 drafter on those dumps, K = 1..7, arms bf16 / int8 RTN / int8 + int8 head / W4A16, draft vocabularies 32k / 64k / 128k (`loader::select_draft_vocab`); batched over anchors. `test_dflash_accept.py`: the batch against `dflash_ref.draft_block`. |
| `dflash_p0.sh` | The two above on the Mac in a 28 GB container, resumable, refusing to start beside the 12a repeat; `DRY_RUN=1` plans only. |
| `k2_ref.py` | Spec 18a: the K2-Horizon CPU reference - a plain-torch port of the checkpoint's `modeling_k2_horizon.py`, layer at a time, bf16 or int4 GPTQ checkpoints by name; `run` (golden file + MoE / MoVA routing dumps), `hfcheck` (against the vendored HF model, layer-streamed), `facts`. `test_k2_ref.py` checks it on tiny random weights. "The K2-Horizon reference" below. |
| `ornith_ref.py` | Spec 15a: Ornith 1.5 35B-A3B from its **int4** checkpoint - transformers' `Qwen3_5MoeForCausalLM`, layer-streamed, the routed experts dequantised on demand; `run` (golden file + every layer's router / shared-gate logits and routes, and with `--mtp-out` the MoE MTP head's M1 reference), `facts` (the checkpoint facts spec 15 §13 rests on). `test_ornith_ref.py` checks it on tiny random weights; `ornith_golden.sh` is the Mac run. "The Ornith reference" below. |
| `kolibri_ref.py` | Spec 20a: the Kolibri-1 reference - layer at a time, bf16 or int4 GPTQ by name (sandwich norms, RoPE sliding / NoPE full layers, window 513 incl. the query, fp32 router on logit + expert_bias with sigmoid weights, ungated shared expert, fp32 head); `run`, `ppl`, `hfcheck`, `facts`, layer-major `run_batch` (the quantisation script's coverage and eval). `test_kolibri_ref.py` (KL0, 21 tests): the transformers port == it bitwise. Facts: `docs/probe-kolibri-2026-10-05.md`. |
| `third_party/kolibri1/` | Our transformers 5.x port `Kolibri1ForCausalLM` (from Aleph Alpha's vLLM plugin `aleph-alpha-inference` @ `049a6a7b`, Apache-2.0) - what AutoRound loads (`tools/quantize_kolibri1.sh`) - with the checkpoint's `config.json`, `generation_config.json`, `tokenizer_config.json` and the chat template, unmodified. |
| `k2_attn_eager_fixture.py` | Spec 18 §10.1: `k2_ref.py attention()` on hash inputs, written as `tests/kernels/k2_attn_eager_fixture.h` (scores, probabilities, output) for `k2_attn_eager_ref_test`; no checkpoint, seconds - the docker line is in its docstring. |
| `third_party/k2_horizon/` | `modeling_k2_horizon.py`, `configuration_k2_horizon.py`, `config.json` from `IFM/K2-Horizon-MoVA-36B-A4B` @ `cca48b66`, unmodified (Apache-2.0, headers kept): the semantics source of truth, imported by the tests and `hfcheck` without network. |
| `kv_int8_probe.py` | Spec 12a: int8 KV schemes (per token, KIVI, Hadamard-rotated K, and `rotkv`); `run` = one layer-streamed forward whose batch carries the variants through a patched eager attention (per-position logits metrics, q/K/V capture); `replay` = fp64 attention from a capture at real and tiled depths. Record: `docs/probe-int8-kv-2026-09-28.md`. |
| `kv_int8_probe.py` | Spec 12a: int8 KV schemes (per token, KIVI, Hadamard-rotated K, and `rotkv`); `run` = one layer-streamed forward whose batch carries the variants through a patched eager attention (per-position logits metrics, q/K/V capture); `replay` = fp64 attention from a capture at real and tiled depths. Record: `docs/probe-int8-kv-2026-09-28.md`. Takes the bf16 base checkpoint too (no `quantization_config`) and `--cont-ids` (teacher-forced ids from a file). |
| `kv8_qwen38_repeat.sh` | Spec 12 §8 "Before 12b" on the Mac: the 12a probe on `Qwen/Qwen3.8-27B` (bf16) in a 28 GB container, golden rows, A4, long32k[:4096] + replay, resumable; how to run and what decides the tolerances: plan 12b "Task A". |
| `k2_kv8_probe.py` | Spec 18e Task 1 (Review Focus 1): the int8 KV probe on K2-Horizon over `k2_ref.py` - variants bf16 (the reference), `rotkv` (the engine's head_dim-128 scheme bit for bit: `src/common/kv8.h` hd128), per token, fp32 attention, each with its own residual stream and KV cache; `run` (decision-row logits metrics + the routing diagnostic + q/K/V capture), `replay` (fp64, real and tiled depths), `summary`. `test_k2_kv8_probe.py` checks it on tiny weights (no checkpoint, seconds). |
| `kv8_k2_repeat.sh` | Spec 18e Task 1 on the Mac: `k2_kv8_probe.py` on the int4 K2 checkpoint (`hf download urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ`, 22.2 GB) in a 28 GB container - check, K2 tokenisation (BOS 0), golden rows (18a's tokens when `oracle-out-k2/` is here, else 32 greedy steps), long[:4096] + capture, replay to 64k; resumable, `STEPS=` subsets. |
| `agnes_remote_check.py` | Agnes: our reference (`agnes.py`) against the checkpoint's own `modeling_agnes.py` (trust_remote_code), per-position logits on 64 ids, both streamed. |
| `vllm_check.py` | The third implementation: vLLM's own greedy 32 tokens on the same three prompt id files, compared against the golden `tokens`. Closes the trust chain's last link - run and recorded 2026-08-25, **96/96** (`docs/14-golden-gate.md` §cross-check). |

Prompts live in `tests/golden/prompts/`: `prose.txt` (plain English, **42 ids**),
`code.txt` (a Python function, **61 ids** - 3 under the ceiling, so do not edit
it without re-tokenizing), `cjk.txt` (Chinese plus two emoji - multi-byte tokens
and the tail of `lm_head` - **38 ids**). Each must tokenize to **24-64 ids**;
`dump.py --max-prompt` enforces the ceiling. The counts are the ones the
recorded run used and are what `n_prompt` in each golden file's metadata says.

## Running it

Everything runs on the box; the Mac never loads the model. Sync from the Mac,
then run the container commands **on the box** - wrapping them in
`tools/box.sh run` adds a third layer of quoting around `"$SNAP"` and is easy
to get wrong.

```bash
tools/box.sh sync                                  # on the Mac
ssh "$BOX"                                         # everything below: on the box
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
container invocations and the 118 s load+dequant is paid three times. The full
run and its re-read are the two committed scripts - no retyping, and the recipe
is under version control:

```bash
# on the box, from the repo root, detached so the 17-min run outlives the ssh
setsid nohup tools/oracle/golden.sh > oracle-out/golden.log 2>&1 </dev/null &
tail -f oracle-out/golden.log
tools/oracle/check.sh          # re-read what was written, in a fresh process
```

`cmake --build build --target golden` prints exactly this sequence; it does not
run it (the oracle needs the container and 61 GiB, and never runs from the Mac
- see the target's comment in `CMakeLists.txt`). Making `dump.py` accept
repeated `--prompt`/`--out` pairs would save ~4 minutes per full re-run; it was
not worth touching the numerics path for.

**Outputs stay on the box**, in `~/b70-inference-server/oracle-out/` - hundreds
of MB per prompt. `.gitignore` has `oracle-out/` and `oracle-out-*/`, and
`tools/box.sh sync` excludes both, so a sync from the Mac neither deletes nor
commits them.

### One golden set per checkpoint - 2026-08-26

A golden set is the oracle's answer **for one set of weights**, so it belongs to
exactly one checkpoint and overwriting one with another's output silently
changes what the gate means. Since spec 1.6 §5.1 there are two:

| directory | checkpoint | `lm_head` |
|---|---|---|
| `oracle-out/` | `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` (HF cache) | bf16 |
| `oracle-out-rtn/` | `~/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64` | **int4 g64** |
| `oracle-out-long/` | `~/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64` | **int4 g64** - the spec 2 §6.2 long prompt only, **NOT YET RUN** (below) |
| `oracle-out-agnes/` | `urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ` (HF cache) | bf16 - spec 14; `dump.py` builds Agnes from `config.json` (`agnes.py`: Qwen3.5 + the parallel FFN, unfolded); **dumped on the Mac 2026-10-04**, layer-streamed (the Mac path below) - rsync it to the box |

`golden.sh` takes `OUT_DIR` and neither default clobbers the other. It reads the
**committed** `tests/golden/prompts/*.ids` rather than re-tokenizing: the two
checkpoints ship the identical tokenizer, so re-running `tokenize.py` could only
be a chance to change the ids by accident. Verified byte-identical to the
`oracle-out/*.ids` the 2026-08-24 run used.

`run_in_container.sh` grew two knobs for this. **`ORACLE_SNAP`** is an absolute
host path to a snapshot, mounted read-only at `/snap` - the way to reach a
self-quantised checkpoint that never went through `hf download`; it takes
precedence over `ORACLE_MODEL`. **`ORACLE_THREADS`** caps `OMP_NUM_THREADS` and
`MKL_NUM_THREADS`, because the box is shared and torch's default is every
thread on it. `dump.py` prints `torch.get_num_threads()` so the cap that was
actually applied is in the log rather than assumed - note that the reference
image capped a requested 28 at **22**.

```bash
# on the box, from the repo root
OUT_DIR=oracle-out-rtn \
ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
ORACLE_THREADS=28 tools/oracle/golden.sh

OUT_DIR=oracle-out-rtn \
ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
tools/oracle/check.sh
```

**Measured 2026-08-26**, three prompts serially, box under a 12-core vLLM XPU
kernel compile: **25 min 14 s** (prose 7:28, code 9:49, cjk 7:57) against
16 min 48 s on an idle box. The output files are within ~100 bytes of the
`oracle-out/` ones - same 290-tensor manifest, same shapes.

### Two config vocabularies, and an int4 head

`dump.py` reads what `src/loader/quant.cc` reads, deliberately in the same
shape, because the golden gate is the only thing grading the two against each
other:

- `quant_method` `"gptq"` **or** `"auto-round"`; `packing_format`, when present,
  must be `"auto_round:auto_gptq"`.
- **A missing `desc_act` is auto-round's spelling of `false`, and it is proven,
  not assumed**: `dump.py` counts the `.g_idx` tensors in the snapshot and dies
  if there are any. A permutation needs a `g_idx` to carry it.
- **`lm_head` is dequantised like any other `.qweight`** - `map_name` already
  left it top level and `Qwen3_5ForCausalLM` already wanted `lm_head.weight`,
  so this needed no new branch. What it did need is memory:
  `dequant_gptq` gained an `n_chunk` column split (bit-identical - columns are
  independent, and the fixture writer asserts it at two widths, one of them
  ragged), used above 65536 columns. That is exactly one tensor, `[5120,
  248320]`, whose unchunked intermediates are ~25 GB on top of a ~47 GB state
  dict. `build_state_dict` prints which kind of head it found.
- **`model_extra_tensors.safetensors` is skipped cleanly on both checkpoints**:
  its tensors are all `mtp.*`, the duplicate-name check still runs over them
  first (that check is the 9B lesson), and nothing in the shard is materialised.
  The RTN checkpoint has 29 `mtp` tensors against the published one's 15, and 8
  of them are int4; the skip is by name and does not care.

### Exact commands and measured numbers

Run 2026-08-24 21:42-21:58 on the box (121 GB of RAM),
image `vllm-xpu-env-next-p314-t214-vxkp0:latest`, serially, `--gen 32` for all
three. The three commands were exactly the two-line `run_in_container.sh` form
above with `prose` replaced by `code` / `cjk`. Every number below is from the
per-prompt logs in `oracle-out/{prose,code,cjk}.log`.

| Prompt | ids | wall | load+dequant | prefill | 32 greedy | peak RSS | output | manifest |
|---|---|---|---|---|---|---|---|---|
| prose | 42 | 314.5 s | 117.9 s | 14.8 s | 181.7 s | 61.4 GiB | 296.6 MiB (311,030,384 B) | 290 tensors |
| code  | 61 | 324.9 s | 118.1 s | 19.0 s | 187.7 s | 61.5 GiB | 350.2 MiB (367,258,272 B) | 290 tensors |
| cjk   | 38 | 316.1 s | 117.7 s | 13.9 s | 184.4 s | 61.4 GiB | 285.3 MiB (299,192,968 B) | 290 tensors |

### The spec 2 §6.2 long prompt (`oracle-out-long`) - staged, NOT RUN

`tools/oracle/make_long_prompt.sh` builds **`tests/golden/prompts/long.ids`,
2820 ids** (20 repetitions of prose+code+cjk = 20 × 141), committed. It builds
the ID stream, not the text: `dump.py --prompt` takes an ids file, and the
tokenizer exists only inside the reference container. The readable
`long.txt` beside it is for review and nothing consumes it. **The consequence,
stated rather than glossed: `long.ids` is the concatenation of three id
streams, which is not the tokenization of the concatenated text** - a real
tokenizer would merge differently across each seam. That is irrelevant to the
gate, which compares the engine against the oracle on the same ids.

**The oracle run itself has not been taken.** It is one `docker run` inside the
reference container, and the L1-engine stage that staged everything else here
was operationally barred from starting containers. The command is one line and
everything it needs is committed:

```bash
# on the box, from the repo root, detached so it outlives the ssh session
mkdir -p oracle-out-long
OUT_DIR=oracle-out-long PROMPTS=long \
ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
ORACLE_THREADS=28 \
  setsid nohup tools/oracle/golden.sh > oracle-out-long/golden.log 2>&1 </dev/null &
tail -f oracle-out-long/golden.log
OUT_DIR=oracle-out-long PROMPTS=long \
ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 tools/oracle/check.sh
```

**Pre-registered cost, so the run has something to be scored against**
(plan 6b Task 13 Step 4; every line labelled):

| term | figure | grade |
|---|---:|---|
| load + dequant | 118 s | **measured** (the table above; the three prompts are within 0.2 s of each other and it does not depend on `T`) |
| prefill forward | ~900 s | **estimated** - 14.8 s at 42 ids scaled linearly in `T` (the MLP is 70% of the FLOPs and is linear) plus 10% for the quadratic attention term: 14.8 × (2820/42) × 1.1 ≈ 1093 s, and the fixed part of the 14.8 s is unknown, so this is an upper-ish bound |
| 32 greedy | ~250 s | **estimated**, from ~5.8 s/token at 42 ids plus the deeper KV walk |
| **total, one prompt** | **20-40 min** | **estimated**, plus ~18 s of container start (**measured**) |
| peak RSS | ~70 GiB of the box's 121 GB | **estimated** - 61.4 GiB **measured** at 42 ids, plus the 192 hook activations (192 × 2820 × 5120 × 2 B = 5.54 GB, **derived**), `logits` ((2820+32) × 248320 × 4 B = 2.83 GB, **derived**) and the eager attention scores (24 × 2820² × 4 B = 763 MB, **derived**, transient) |
| output file | ~8.5 GB | **estimated**, the three terms above plus 151 MB of GDN states and 4 MB of conv states |

**Reconciliation with spec §6.2**, which said "hours-class - estimated from the
18-minute 32-id runs": that estimate scaled the whole 16 m 48 s three-prompt run
by prompt length; scaling the three terms separately (fixed load, linear
prefill, 32 decodes) gives the 20-40 min above. Both are estimates and neither
is measured yet; **the run's own `/usr/bin/time -v` is the number that goes in
the record**, and whichever it lands on, this sentence stays. Check
`df -h ~/b70-inference-server` first and give the container **no** memory limit.

The RTN checkpoint is the one to run: it is the gate checkpoint (spec §7 records
on RTN) and one CPU oracle run is what §6.2 budgets. A Vishva long set is the
same command with `ORACLE_SNAP` unset and `OUT_DIR=oracle-out-long-vishva`;
priced here, not run.

`ctest -R prefill_gate_long_test` SKIPs (77) until `oracle-out-long/` exists,
then gates the prefill path at **C = 1024** over 2820 ids - two chunk
boundaries crossed in attention-over-cache and in the GDN state carry.

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

Re-read by a separate process (`tools/oracle/check.sh`), not the writer's own
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

### The Mac path: Agnes in a Linux container, layer-streamed (spec 14, 2026-10-04)

Agnes 3.0 Flash dequantised to bf16 is ~62 GB, and the x86_64 Mac's own Python stops at
torch 2.2 (no transformers 5). So the Agnes oracle runs on the Mac in a plain Linux
container, with the weights **streamed one decoder layer at a time** (`stream.py`).
The container (kept between sessions; `docker start agnes-ref` after a Docker restart):

```bash
# from the repo (or worktree) root on the Mac
docker run -d --name agnes-ref -v ~/.cache/huggingface:/hf:ro -v "$PWD":/ws -e HF_HOME=/hf \
  -w /ws python:3.12-slim sleep infinity
docker update --memory 28g --memory-swap 28g agnes-ref      # see "memory" below
docker exec agnes-ref pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
docker exec agnes-ref pip install transformers==5.15.0 safetensors accelerate ninja py-spy
docker exec agnes-ref sh -c 'apt-get update && apt-get install -y g++ procps'   # stream.py's C++ dequant
```

Install torch from the CPU index **first** and never let a later `pip install` pull it
from PyPI (a bare `pip install accelerate` resolves torch to the 2.14.1+cu130 build);
`transformers==5.15.0` is the box image's version. Then, e.g. one golden prompt:

```bash
docker exec -d -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/root/torch_ext \
  agnes-ref sh -c 'S=$(ls -d /hf/hub/models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ/snapshots/*/);
  python3 tools/oracle/dump.py "$S" --stream --prompt /ws/tests/golden/prompts/prose.ids \
    --out /ws/oracle-out-agnes/prose.golden.safetensors --gen 32 --max-prompt 4096 \
    --mlp-in /ws/oracle-out-agnes/mlp_in.safetensors > /ws/oracle-out-agnes/prose.log 2>&1'
```

(`HF_HOME=/tmp/hf` for anything that loads remote code or `AutoTokenizer` on Agnes:
`/hf` is read-only and transformers copies `modeling_agnes.py` into `$HF_HOME/modules`.)

What streaming changes, and what it does not:

- **The computation is the resident one.** The model is built on `meta` exactly as
  before; `dump.convert` (the same `dequant_gptq` rule, the same names) makes each
  layer's tensors when a forward pre-hook asks, the layer runs, a forward hook returns
  it to `meta`. The model's own forward, masks, rotary, cache and greedy loop are
  untouched. A worker thread dequantises the next layer during the current one.
- **`dequant_t`** (C++, built on first use by `torch.utils.cpp_extension`) is
  `dequant_gptq(...).t().contiguous()` bit for bit (one fp32 product, one RNE cast; a
  16-entry table per group and column); ~15x faster, and every tensor's first dequant
  re-does 64 columns with `dequant.py` and must be equal.
- **Grouped bf16 `conv1d` runs on one thread.** On the Mac's CPU (i9-9980HK, AVX2,
  16 vCPUs) torch 2.14's depthwise bf16 conv is 50-4000x slower multi-threaded (Agnes's
  10240-channel causal conv did not finish in 10 min); outputs are bitwise equal across
  thread counts. A wrapper on `torch.nn.functional.conv1d` (both modeling files call it).
- **Memory.** Cap the container at 28 GB. Uncapped (Docker Desktop at 64 GB), the guest
  page cache over the 18 GB checkpoint plus kept layers grew the VM to 60 GB, macOS
  swapped 38 GB, and a decode step took 360 s instead of 43 s; Docker Desktop had to be
  restarted to give the memory back. Streamed, peak RSS is 22 GiB (embed + lm_head
  4.7 GiB resident, two layers, the mmapped shards). `--stream-keep N` keeps layers
  0..N-1 once loaded; on this Mac it does not pay.

Measured (prose, 42 ids, `--gen 32`): prompt forward 263 s, 32 greedy steps 2100 s
(~66 s each), wall 2379 s; code 3118 s, cjk 1069 s (the Mac's load varies). Record: `docs/probe-agnes-2026-10-03.md`.

### The DFlash reference (spec 19a, 2026-10-05)

`dflash_ref.py` is spec 19 §2 in plain torch: float32 compute, bf16 weights upcast (the drafter's
linears once at load; the target's embedding / `lm_head` and the selector codebooks per use, in row
chunks), deterministic. Semantics ported from vLLM's Apache-2.0 `qwen3_dflash.py`,
`qwen3_dflash2.py` and the DFlash / DFlash 2 speculators (file header).

```python
d = load_drafter(drafter_snapshot, target_snapshot)        # Qwen/Qwen3.8-27B for embed + lm_head
ctx = d.context_kv(taps_from_resid(golden, d.cfg.target_layer_ids), positions)
out = d.draft_block(anchor_id, p, K, ctx, temperature=0.0)  # out.ids, cand_ids, unary, scores, q
```

- **Taps** are a dict keyed by exactly `target_layer_ids`; `taps[i]` is the residual AFTER target
  layer i (`resid.L{i}` in our dumps, `hidden_states[i + 1]` in transformers; vLLM's i + 1).
- **Block**: row 0 = the anchor (the token at p, no hidden state yet) at position p, rows 1..K =
  the mask token at p + 1 .. p + K; mask row j drafts the token at p + j. Only context positions
  < p are read. `1 <= K <= block_size - 1`.
- **Sampling** (temperature > 0 needs a seed): Gumbel-max keyed like vLLM by (seed, P - 1 + 2^30,
  token id); `q` is softmax(scores / T) on the row's 16 candidates (DFlash 2) or over the head (v1).
- **Ornith's DFlash (v1)** resolves to 5 *causal* sliding layers + 1 non-causal full layer: vLLM's
  rule when a config has `layer_types` and no `is_causal` (`_dflash_layer_causal`).

The tests take ~10 s; the last one loads `z-lab/Qwen3.8-27B-DFlash2` (+ `syvai/...-W4A16`) and the
`Qwen/Qwen3.8-27B` embedding / head from the HF cache (~16 GB RSS, estimated: not yet run) and prints SKIP when they are not
there. On the Mac, in the image the Agnes container was committed to (torch 2.14.1+cpu,
safetensors), capped at 28 GB:

```bash
# from the repo (or worktree) root on the Mac
docker run --rm --memory 28g --memory-swap 28g -v "$PWD":/ws -w /ws \
  -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  agnes-ref-img:latest python3 tools/oracle/test_dflash_ref.py
```

On the box: `tools/oracle/run_in_container.sh 'HF_HUB_CACHE=/hf/hub python3 tools/oracle/test_dflash_ref.py'`
(the script points `HF_HOME` at a scratch dir; `HF_HUB_CACHE` sends `find_snapshot` to the
read-only cache mount). Not yet run there.

### The DFlash P0 (spec 19a Tasks 2-3, 2026-10-06)

Two scripts and a driver, on the Mac:

```bash
# from the repo (or worktree) root; detached, resumable, log in oracle-out-19a/p0.log
DRY_RUN=1 tools/oracle/dflash_p0.sh                       # the plan, ~1 min, 6 GB container
nohup tools/oracle/dflash_p0.sh > /dev/null 2>&1 &        # the run (28 GB container)
```

- **`dump_taps.py`** runs `Qwen/Qwen3.8-27B` (bf16, layer-streamed, fp32 matmuls as the 12a
  repeat) over each source's [prompt + recorded continuation] and writes
  `oracle-out-19a/dumps/<corpus>__<label>.taps.safetensors`: the residual after layers 5, 19,
  33, 47, 61 (`resid.L{i}`, the drafter's `taps[i]`) from `n_prompt - 2048` on (the drafter's
  window never reads older context from an anchor in the continuation), the bf16 greedy next id
  per position (ids < 248077, ties to the lower id), the top-64 logits and the logsumexp (a
  later sampled pass needs no re-dump). Sources are batched, right-padded, up to
  `--batch-tokens` (8192): the padding is causal-safe (bitwise, tested), but a batch is not
  bitwise a one-sequence forward - the GDN core's fp32 batched matmuls block differently with
  the batch shape (a few bf16 ulps per layer; `test_dump_taps.py` says how much).
- **`dflash_accept.py`** drafts at every anchor of the continuation (n_prompt <= p <= N - 7),
  one block per K = 1..7, and keeps the longest prefix equal to the bf16 greedy ids. A row is
  censored at the depth where the recorded text leaves the bf16 greedy path (the recorded
  continuations come from the int4 engine / oracle, so they mostly - not always - agree).
  Reports alpha per depth, `E_K = 1 + sum_i prod_{j<=i} alpha_j` tokens per verify, the
  length histogram, per corpus (the source name's prefix) and arm.
- **Sources:** the golden prompts with the 32 greedy ids of the 2026-08-24 oracle
  (`GOLDEN_CONT_DIR` takes longer recorded continuations), the A4 set (34 scenarios, 2613
  output ids: the reference text of `tests/server/toolcall_expected.json` re-encoded, bf16
  backend first, via `tools/spec/lookup_accept.py`'s `load_a4`), and `EXTRA_SOURCES`.
- **Cost (estimated by `DRY_RUN`, assumed rates):** dumps ~5 h on an idle Mac (12 streamed
  forwards at ~6.5 min each + ~4 PFLOP), peak ~12.5 GiB; acceptance ~0.4 h per drafter arm
  (2487 anchors x 35 drafter rows + 28 head rows), ~16-20 GiB.

Tests (tiny models, a minute; `dflash_p0.sh`'s `check` step runs both):

```bash
docker run --rm --memory 6g -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/oracle/test_dump_taps.py
docker run --rm --memory 6g -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/oracle/test_dflash_accept.py
```

### The K2-Horizon reference (spec 18a, 2026-10-05)

`k2_ref.py` ports `modeling_k2_horizon.py` (vendored in `third_party/k2_horizon/`) to plain torch,
one decoder layer at a time: stream.py's `_Prefetch` loads the next layer's attention, router,
norm and dense / shared tensors on a worker thread, and the MoE / MoVA experts are read on first
use and dropped with the layer - a decode step reads only the 8 + 4 experts it routes to. Mode
`bf16` (default) rounds exactly where the HF model in bf16 rounds and uses torch's own bf16 GEMMs:
on the same torch it is **bit-identical** to the vendored model (eager attention), prompt and
cached decode. Mode `f32` never rounds. Ties at the top-k cut go to the lower expert id (HF's
`torch.topk` leaves them unspecified). Semantics, rounding points, the HF-vs-vLLM differences and
the checkpoint facts: `docs/probe-k2-2026-10-05.md`.

Output (`run`): `dump.py`'s layout - `resid.L{i}` / `mixer.L{i}` / `mlp.L{i}` bf16 [T, 2560],
`logits` f32 [T + gen, V], `tokens` i32 [gen] - plus plan 8d's routing records for every sparse
layer and every forward (prompt rows, then one row per generated step): `route.moe.ids.L{i}` i32
[T + gen, 8] and `route.mova.ids.L{i}` [T + gen, 4] (ascending id), `route.*.w.L{i}` bf16 (the
multiplier each id gets), `route.*.gap.L{i}` f32 [T + gen] (selection score of the k-th minus the
(k+1)-th: 0 = a tie at the cut), and `rope.cos` / `rope.sin` bf16 [T + gen, 128].

The tests (tiny random weights; 18 tests, ~45 s of which ~30 s build stream.py's C++ dequant; `test_real_index` SKIPs unless
`K2_INDEX_BF16` / `K2_INDEX_INT4` name directories holding a real `config.json` +
`model.safetensors.index.json`):

```bash
# from the repo (or worktree) root on the Mac
docker run --rm --memory 28g --memory-swap 28g -v "$PWD":/ws -w /ws \
  -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  agnes-ref-img:latest python3 tools/oracle/test_k2_ref.py
```

**The real run (not yet done).** `run` needs torch + safetensors only (`hfcheck` also needs
transformers >= 5.13 and huggingface_hub with `dataclasses.strict`, as in `agnes-ref-img`). Memory:
embed + head 2.6 GB resident, one bf16 sparse layer 1.6 GB (two while prefetching), the experts a
prompt touches; ~6-8 GiB RSS (estimated) plus the page cache over the shards. Which checkpoint:
the **int4** one is what the engine runs (golden sets it can match token for token; 22.2 GB, fits
the Mac's 28 GB container too); the **bf16** one (74.9 GB, the box) is the quantisation-damage
reference. On the box, from the tree root, one prompt at a time, detached:

```bash
M=models--IFM--K2-Horizon-MoVA-36B-A4B          # or models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ
mkdir -p oracle-out-k2 && free -g               # >= 70 GB available for the bf16 page cache
for p in prose code cjk; do                     # K2 prompts start with BOS 0 (docs/probe-k2-2026-10-05.md)
  ORACLE_MODEL=$M tools/oracle/run_in_container.sh \
    "python3 tools/oracle/tokenize.py \"\$SNAP\" encode --bos tests/golden/prompts/$p.txt > /ws/oracle-out-k2/$p.ids"
done
ORACLE_MODEL=$M setsid nohup tools/oracle/run_in_container.sh \
  'for p in prose code cjk; do python3 tools/oracle/k2_ref.py run "$SNAP" --prompt /ws/oracle-out-k2/$p.ids \
     --out /ws/oracle-out-k2/$p.golden.safetensors --gen 32 > /ws/oracle-out-k2/$p.log 2>&1; done;
   python3 tools/oracle/k2_ref.py hfcheck "$SNAP" --prompt /ws/oracle-out-k2/prose.ids \
     --against /ws/oracle-out-k2/prose.golden.safetensors > /ws/oracle-out-k2/hfcheck.log 2>&1' \
  > ~/k2-ref.log 2>&1 < /dev/null &
```

Time (estimated from the 4-layer timing at real shapes on the Mac and the bytes read): bf16 - a
decode step reads 10.6 GB of weights (8 + 1 + 4 experts per layer, all attention, the head),
~4-6 s warm; the prompt touches ~55 of 100 experts per layer, ~1-2 min, plus the first cold read
of the 75 GB shards; **~5 min per prompt warm, ~15-20 min for the three**. int4 - 3.8 GB read
per step but dequantised on the fly (stream.py's C++ `dequant_t`, 1-3 GB/s of bf16 output on the
Mac), ~7 s per step, ~5-6 min per prompt. `hfcheck` runs the vendored model layer-streamed with
every expert of each layer resident (prompt forward only). The `run` log prints the gap
distribution (`#==0` = ties at the cut).

### The Ornith reference (spec 15a, from the int4 checkpoint, 2026-10-06)

`ornith_ref.py` runs transformers 5.15's own `Qwen3_5MoeForCausalLM` (eager attention, the
`grouped_mm` experts implementation unless `--experts eager`, bf16 - the checkpoint's top-level
`dtype: float16` is AutoRound's export label; its text config and every unquantised tensor are
bf16) on `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ` dequantised by `dequant.py`'s rule
(stream.py's C++ `dequant_t`), so the engine - which loads the same int4 weights - can match it
token for token (spec 18a's argument for K2). Layer-streamed (`stream.attach`); the per-expert GPTQ
tensors are stacked into the class's 3D `gate_up_proj` (gate rows first) / `down_proj` only for
the experts a forward routes to, in one buffer pair shared by all layers - bitwise the fully
materialised model (`test_ornith_ref.py`, both experts implementations), and a decode step
dequantises 9 experts per layer instead of 257.

Outputs: `oracle-out-ornith/<p>.golden.safetensors` - dump.py's tensors plus `router_logits.L*`
bf16 [T + gen, 256], `shared_gate_logits.L*` [T + gen, 1] (what ornith_decode_test's R2 reads),
`route.{ids,w,gap}.L*` (the reference's top-8, their weights, p8 - p9; the log prints the gap
distribution for R2's tolerance) - and `oracle-out-ornith-mtp/m1/<p>.mtp.safetensors` +
`<p>.cont256.ids` (mtp_head_ornith_test's M1: the head on this run's post-norm hidden, its 257
experts RTN-quantised exactly as loader/rtn.h does at load; `--mtp-experts bf16` for the shipped
weights). The committed `tests/golden/prompts/{prose,code,cjk}.ids` are this checkpoint's
tokenisation too (checked, spec 15 §13).

Tests (tiny random weights, ~1 min; run 2026-10-06, all pass):

```bash
# from the repo (or worktree) root on the Mac
docker run --rm --cpus 2 --memory 6g -v "$PWD":/ws -w /ws -e HF_HOME=/tmp/hf \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext agnes-ref-img:latest python3 tools/oracle/test_ornith_ref.py
```

The real run - download first (plan 15a has the disk / RAM / time figures), then
`tools/oracle/ornith_golden.sh` (detached, resumable, 28 GB cap, `facts` first) and
`tools/oracle/ornith_golden.sh --status`.

## The gate: what these files are compared against, and what it proved

`tests/golden/golden_gate_test.cc` is the consumer of everything above. The full
story - the divergence classes, the magnitudes, the `code` prompt's anomalous
position - is `docs/14-golden-gate.md`; this section is how to run it and the
one-line result.

```bash
tools/box.sh test golden_gate_test          # sync + build + ctest, on the box
# the full diagnostic log (per-layer tables, attribution traces, every step):
tools/box.sh run './build/tests/golden_gate_test "$PWD/oracle-out" "$PWD/tests/golden/prompts"'
```

The three prompt id files are **committed** as
`tests/golden/prompts/{prose,code,cjk}.ids` - the same 42 / 61 / 38 ids the
`prompt_ids` metadata above records, pulled from `oracle-out/` and small enough
to version. The `.safetensors` are not committed and never leave the box; when
they are absent the test exits **77**, which ctest reports as SKIP.

Per prompt the test ingests the ids one per replay (reading the per-layer
residual tap after each), then generates 32 ids as 32 × `generate(1)`, reading
`buffers().logits` before each step so a mismatch can be reported with both
top-5 sets. What it asserts:

| | Bar | Measured 2026-08-25 |
|---|---|---|
| 32 generated ids == `tokens` | **exact - the gate** | 32/32 on all three prompts |
| `gdn_state` vs `gdn_state.L{i}`, per GDN layer | cosine ≥ 0.999 | min 0.999537, 0 of 144 below |
| per-layer residual tap | diagnostic, `**LOW**` under 0.999 | 36 of 9024 (layer, position) pairs low, all on 3 positions of the `code` prompt |
| logits at each decision row | diagnostic | 0.999844 … 0.999989 over 96 rows |

Two things about the tap comparison are easy to get wrong and are worth
repeating here, because they are properties of *these files*:

- the engine's `tap[i]` is **not** `resid.L{i}`. It is the residual with layer
  *i*'s mixer folded in and its MLP not yet folded, so the test builds
  `bf16(resid.L{i-1}[t] + mixer.L{i}[t])` in fp32 and rounds RNE. Layer 0 has no
  `resid.L-1` - these files contain no embedding tensor - so the embedding rows
  are gathered from the loaded model instead;
- layer 63's post-MLP residual has no tap. `resid.L63` is compared against the
  engine's `b.resid` after the fence, which is that same vector.

The engine reproduced all three continuations recorded above id for id,
including the CJK prompt's completion of the trailing emoji's variation
selector.

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

1. **The bits - the *convention*, not the values.** `dequant.py` fixes the one
   meaning of the int4 nibbles: nibble order within the `[K/8, N]` u32 word,
   zero point 8 (`q - 8`, no `qzeros` stream), the group axis (64 along `K`),
   and the packing. `tests/golden/dequant_fixture.safetensors` pins **that
   convention** bit-exactly against the C++ side, and that is all it pins.

   It is **not** a claim that the engine and the oracle hold equal weight
   values, because the engine never materialises a weight: `gemv.cl` computes
   `scale · Σ(q − 8)·x` with the products and the sum in fp32 and the scale
   applied once per group of 64, while `dequant.py` materialises
   `bf16(scale · (q − 8))` and torch then multiplies bf16 weights by the
   activations. Two different accumulation orders and two different rounding
   points on the same bits. That divergence is **expected, and it is the
   engine that is the more accurate of the two** - measured at roughly
   **0.2% RMS per weight** (the bf16 mantissa is 8 bits; a per-weight RNE cast
   costs ~2⁻⁹ relative). Read a disagreement of that order as this, not as a
   bug; the golden test's gate is token equality for exactly this reason.

   The controller ruling of **2026-08-25** removed one member of this family
   deliberately: RMSNorm weights are now stored fp32 `1 + w`, so the norm
   multiplier no longer carries a bf16 rounding the reference never had (docs
   13, "What the loader bakes in"). What is left is the GEMV accumulation
   difference above, which is inherent to not materialising weights.
2. **The math.** The oracle runs those weights through `transformers` 5.15's
   pure-torch `modeling_qwen3_5.py` on CPU, `attn_implementation="eager"` (fp32
   softmax), deliberately without `fla`. That reference implementation - not
   our kernels - decides what the correct activations are.
3. **The engine.** **Plan 3's golden test compares the engine against THESE
   files** - and as of **2026-08-25 it passes, 96/96 token ids exact** (re-read
   under the 2026-08-26 ruling as 94 determined-exact + 2 tie-agreements)
   (`docs/14-golden-gate.md`): `oracle-out/{prose,code,cjk}.golden.safetensors`
   on the box, the ones produced by the run recorded above. Per-layer `resid`/`mixer`/`mlp`,
   the GDN and conv states after the prompt, the logits rows and the 32 greedy
   token ids. Their metadata pins the snapshot path, the prompt ids, `gen`, the
   attention implementation and the group size, so a golden file always says
   what it is a golden file *of*. Regenerate and the comparison target changes;
   that is the point of recording the exact numbers above.

What links 1-3 on their own do **not** prove is that the checkpoint was unpacked
the way its author packed it. Link 1 is exactly what makes link 3 meaningful and
also what limits it: the C++ loader and the oracle are pinned to the *same*
`dequant.py` convention (bit-exactly, by test), so a convention that is wrong
moves the engine and the oracle **together** and the golden test still passes.

4. **The third implementation.** `vllm_check.py` runs vLLM's own greedy
   generation on the same checkpoint and the same three prompt id files -
   vLLM's GPTQ unpack (`XPUwNa16LinearKernel`), its Triton GDN kernels, its
   FlashAttention, on the XPU: nothing shared with either of the paths above
   except the checkpoint's bytes. **Run 2026-08-25: 96/96 token ids
   element-exact against the golden `tokens`, on both the compiled and the
   `enforce_eager` path.** The chain is closed for these three prompts, and
   `dequant.py` is cross-checked rather than merely self-consistent. Full
   record - command, ids, caveats - in `docs/14-golden-gate.md` §cross-check.

> **The chain has a resolution limit, found 2026-08-26 and recorded in
> docs/14 ("The RTN-checkpoint gate").** `dump.py` records
> `out.logits[0].to(torch.float32)`, and `out.logits` from a bf16 model *is*
> bf16 - so every value in a golden `logits` tensor lands exactly on the bf16
> grid (verified: zero low 16 bits, all six files). The reference therefore
> carries ~8 mantissa bits at a decision, and **two candidates inside one ulp of
> each other are indistinguishable to it**. Measured: `oracle-out-rtn/prose` has
> **3 decision rows where the top two logits are bit-identical**, and
> `oracle-out/cjk` has **2** - the latter has been there since the sets were
> made, and the engine happened to agree with `torch.argmax`'s lowest-index
> tie-break on both. On the RTN set it does not, at the first one, and the gate
> reads 79/96. This is the point where "the engine is the more accurate of the
> two" stops being a footnote and starts deciding a token.
>
> **Resolved by ruling, 2026-08-26**: such a row is UNDETERMINED, and the gate
> asserts membership of the golden argmax set there instead of torch's
> lowest-index tie-break, at unchanged strictness everywhere else. Both gates
> are green under it - 187 of 187 determined rows element-exact across the two
> checkpoints. The reference's resolution limit is now a property the gate
> *knows about* rather than one it trips over. docs/14 has the ruling.

Read a future three-way disagreement like this:

- **engine ≠ oracle** → an engine bug (kernel, layout, fusion, fp32 accumulation
  boundary). The oracle is the reference; the engine is wrong.
- **engine == oracle, both ≠ vLLM's greedy output** → **the dequant convention
  is the common suspect**, because it is the one thing the engine and the oracle
  share and vLLM does not. Check `dequant.py` first - zero-point handling
  (`q - 8` vs. an explicit `qzeros`), the group axis, the `[K/8, N]` packing
  order and nibble ordering, `desc_act`/`g_idx` - before touching a kernel.
- **all three agree** → the chain is closed. *(This is the 2026-08-25 result.)*

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

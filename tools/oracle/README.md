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
| `tokenize.py` | `encode` a prompt file to ids / `decode` ids back. Raw text, no chat template, no special tokens. |
| `dump.py` | Builds `Qwen3_5ForCausalLM` from the config, loads a `dequant.py`-produced bf16 state dict `strict=True`, forwards the prompt with hooks, greedy-decodes, writes one safetensors file. |
| `run_in_container.sh` | Wraps `docker run` for the box: read-only HF cache at `/hf`, repo at `/ws`, `$SNAP` resolved to the snapshot directory, **`-u $(id -u):$(id -g)`** so outputs are not root-owned, no memory limit. |
| `golden.sh` | The production run: the three prompts, serially, `--gen 32`. This is the script that made the files plan 3 compares against - committed rather than retyped. `PROMPTS` names which sets to build; `--max-prompt 4096` since spec 2. |
| `make_long_prompt.sh` | Builds `tests/golden/prompts/long.ids` (2820 ids) and `long.txt` from the three committed prompts, on the Mac, with no model and no tokenizer. Spec 2 §6.2's ≥ 2048-id prompt. |
| `check.sh` | Re-reads the three written files in a separate process and prints the block quoted under "Sanity checks" below. |
| `stream.py` | Layer-streamed weights for `dump.py --stream` / `mtp_ref.py --dump --stream` (spec 14, the Mac path below): the model on `meta`, embed/norm/lm_head resident, each decoder layer materialised by a forward pre-hook and dropped after its forward; plus a bit-exact C++ `dequant_t` and the single-thread grouped conv1d. |
| `dflash_ref.py` | Spec 19a: the CPU reference of the DFlash / DFlash 2 drafters (config reader, drafter + target embed/head loader incl. the W4A16 drafter, `context_kv`, `draft_block`); `test_dflash_ref.py` checks it on tiny random weights. "The DFlash reference" below. |
| `kv_int8_probe.py` | Spec 12a: int8 KV schemes (per token, KIVI, Hadamard-rotated K, and `rotkv`); `run` = one layer-streamed forward whose batch carries the variants through a patched eager attention (per-position logits metrics, q/K/V capture); `replay` = fp64 attention from a capture at real and tiled depths. Record: `docs/probe-int8-kv-2026-09-28.md`. |
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

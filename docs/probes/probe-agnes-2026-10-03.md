# Agnes 3.0 Flash on the Mac - the CPU reference, the fold proof, the template (2026-10-03)

Spec 14 phase 1 (`docs/superpowers/plans/2026-10-03-spec14-write-phase.md`), Tasks 2 and 4.
Written **without the box**: everything here ran on the operator's Mac (x86_64, macOS,
Apple clang 21, Python 3.12, torch 2.2.2, transformers 4.57.6). What could not run here
is listed as **pending** at the end and carried into
`docs/superpowers/plans/2026-10-03-spec14-validation-checklist.md`. Nothing here is a
card measurement.

## What the checkpoint is, from its own files

`urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ`, snapshot `946012d0`. The full download
(22.9 GB) was still in progress on the Mac (~1 GB of it on disk at 00:55), so the
numbers below come from the small files (`config.json`, the index,
`chat_template.jinja`, `tokenizer_config.json`, `modeling_agnes.py`) and from the six
safetensors **headers** (HTTP range requests, ~0.6 MB), plus the MLP tensors of four
layers (range-fetched, 161.3 MB each).

| | measured from the files |
|---|---|
| `architectures[0]` | `AgnesForConditionalGeneration`, `model_type` `agnes` |
| layers | 72; `layer_types` = 54 `agnes_delta_attention` + 18 `agnes_global_attention`, global exactly at `l % 4 == 3` |
| MLP | `intermediate_size` 17408 + `parallel_ffn_intermediate_size` 2048; `modeling_agnes.py` `AgnesMLP.forward`: `y = mlp(x) + parallel_ffn(x)`, the same input |
| per-layer tensors | Qwen3.8's set (names `delta_attn.` / `global_attn.`) + 12 `mlp.parallel_ffn.*` (qweight/scales/qzeros/g_idx x 3), all int4 g64 sym |
| dtypes | I32 qweight/qzeros/g_idx, F16 scales, BF16 everything else (incl. `lm_head`, embed, norms, `in_proj_a/b`) |
| MTP head | 15 bf16 tensors in `model_extra_tensors.safetensors`, `mtp.layers.0.global_attn.*`, MLP 17408 (no parallel FFN), 849,398,784 B - Qwen3.8's byte count |
| quantisation | `quant_method` gptq, auto-round 0.16.0, bits 4, group 64, sym, `desc_act` false, `dynamic` = 108 `-:` rules (in_proj_a/b of the 54 GDN layers) |

**`W` for the loader's cross-check** (bytes a decode step streams with the bf16 head),
summed from the headers over doc 03's four categories:

| | Agnes (headers) | Qwen3.8 (docs/03, measured) |
|---|---:|---:|
| int4 qweight | 14,816,378,880 | 12.163 GB |
| f16 scales | 926,023,680 | 0.760 GB |
| bf16 per-layer tensors | 59,035,264 | 0.052 GB |
| `lm_head` (bf16) | 2,542,796,800 | 2.543 GB |
| **W** | **18.344 GB** | **15.519 GB** |

The qweight and scales sums equal the shapes' arithmetic exactly (the same formula gives
Qwen3.8's 12.163 / 0.760). With the int8 head (spec 9) Agnes reads 17.07 GB per token, the
spec's derived ~17.2 GB. `model::agnes().doc_w` carries 18.344 GB; the loader's 2% check on
the card is the first time it meets a real load (pending).

## The reference (spec 14 §3.4) - `tools/oracle/agnes.py`

Qwen3.5's `Qwen3_5ForCausalLM` from a translated text config (layer types mapped,
Agnes-only keys dropped), each layer's `mlp` replaced by `AgnesMLP` - the layer's own
gate/up/down adopted (state-dict names unchanged) plus `parallel_ffn`, summed in the module
dtype exactly as `modeling_agnes.py` does. Names mapped as vLLM PR #57003's
`WeightsMapper` (`delta_attn.` -> `linear_attn.`, `global_attn.` -> `self_attn.`).
`dump.py` and `mtp_ref.py` detect `model_type: agnes` and build it; Qwen3.8's path is
unchanged (no Qwen3.8 name contains either infix; its metadata gains no key). `dump.py
--mlp-in` also writes the listed layers' MLP inputs, for the fold proof on real activations.

`tools/oracle/test_agnes.py <snapshot>` on the Mac:

```
1. config: 72 layers (54 GDN + 18 FA at l % 4 == 3), parallel FFN 2048
2. names: 3170 language-model keys mapped, unique; 12 parallel_ffn tensors per layer; MTP head 15 keys -> self_attn
3. AgnesMLP == two-branch SwiGLU, bitwise; state-dict names as the checkpoint's
4. SKIPPED: transformers' Qwen3_5 is not importable here (ModuleNotFoundError); run in the oracle container
```

Check 4 (the build on meta at the real config, strict key match) and the comparison with
`modeling_agnes.py` need transformers >= 5, which needs torch >= 2.5; torch for x86_64 macOS
stops at 2.2.2, transformers 4.57 has no `qwen3_5`, and the Docker daemon was not running.
Both are box items.

## The fold (spec 14 §2, G1 CPU half) - `tools/oracle/agnes_fold.py`, `src/loader/fold.cc`

On packed GPTQ int4: gate' = [gate | gate_p], up' = [up | up_p] along N (columns of
`qweight [K/8][N]` and `scales [K/64][N]`); down' = [down ; down_p] along K, at `qweight`
row 2176 and `scales` row 272. `g_idx` must be the identity grouping (asserted).

**Synthetic, the shared fixture** (`tests/loader/agnes_fold_fixture.safetensors`, hidden
128, intermediate 192 + 64, written by `agnes_fold.py --write-fixture`):

- `test_agnes_fold.py`: the folded tensors dequantise to the stacked dequantisations
  **bit-exact**; gate'||up' interleave16 layout checked; folded MLP vs two-branch in fp32,
  worst row cosine 1.000000000 (fp64 cosine; max |d| 1.5e-5); a permuted `g_idx` and an
  off-boundary row join refused; the committed fixture equals the generator.
- `tests/loader/agnes_fold_test.cc` (Apple clang, host-only): `loader::fold_n` / `fold_k`
  and the loader's own gate'||up' assembly (`cols_interleave16` + `repack_int4_layout0_cols`)
  **byte-identical** to the Python fold: gate' 128x256, up' 128x256, down' 256x128, gate'||up'.

**Real packed weights, layers 0, 3, 35, 71** (the checkpoint's own int4 tensors,
range-fetched): the fold's dequant equals the stacked dequant **exactly** on every layer,
and the folded MLP against the two-branch MLP, fp32 math, 64 input rows per layer:

| layer | kind | cos min (fp64) | max \|d\| | rel L2 | mean \|y\| |
|---:|---|---:|---:|---:|---:|
| 0 | GDN | 1.000000000 | 1.53e-05 | 1.03e-07 | 70.97 |
| 3 | FA | 1.000000000 | 2.86e-06 | 2.06e-07 | 33.46 |
| 35 | FA | 1.000000000 | 9.54e-07 | 2.19e-07 | 45.65 |
| 71 | FA | 1.000000000 | 3.81e-06 | 2.23e-07 | 107.52 |

G1 (CPU) bar cosine >= 0.999999: met on these inputs. **The inputs are synthetic**: N(0,1)
rows RMS-normalised and scaled by the layer's real `post_attention_layernorm` (1 + w) - the
MLP's input scale, not its directions. Real activations need a forward through every
earlier layer, i.e. the full checkpoint: pending (`dump.py --mlp-in`, then
`agnes_fold.py --proof <snapshot> --inputs <mlp_in>`). The algebra does not depend on the
inputs; the rel L2 of ~2e-7 is fp32 accumulation order over K = 19456.

## The chat template and tool calls (spec 14 §3.5, Review Focus 5)

- Agnes's `chat_template.jinja` is **byte-identical to Qwen3.8's**: sha256
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`, the hash
  `chat::Template` already pins for its minja fallback (`is undefined` -> `is not defined`).
  `bos_token` null, `eos_token` `<|im_end|>`, as Qwen3.8's.
- The tool-call format is therefore Qwen3.8's XML (`<tool_call><function=...><parameter=...>`)
  and `src/server/toolcall` parses it unchanged; no parser addition.
- The engine's renderer (`template_test`, Apple clang) on Agnes's snapshot: Qwen3.8's three
  reference renders identical; and five new message lists
  (`tests/tokenizer/agnes_template_cases.json`: plain, thinking with reasoning in history,
  tools, a tool call in history, two tool responses) identical to transformers'
  `render_jinja_template` (4.57.6; the Mac has no `tokenizer.json`, so not through
  `AutoTokenizer`) - 156 / 450 / 1492 / 1487 / 1739 bytes.
- **The tokenizer is not Qwen3.8's.** `tokenizer.json` (19,991,676 B against Qwen3.8's
  19,989,325) has the same vocab and merges but (a) 12 added specials at ids
  248077..248088 (`<|agnes_bos|>`, `<|agnes_eos|>`, `<|agnes_system|>`, `<|agnes_user|>`,
  `<|agnes_assistant|>`, `<|agnes_tool|>`, `<|agnes_think_start|>`, `<|agnes_think_end|>`,
  `<|agnes_reserved_0..3|>`) - none used by the template, eos stays `<|im_end|>` 248046 /
  `<|endoftext|>` 248044 (`generation_config.json`) - and (b) a pre-tokenizer regex with
  `[\p{L}\p{M}]+` where Qwen3.8 has `\p{L}+` (combining marks join the letter run) and
  ByteLevel decoder flags `add_prefix_space/trim_offsets/use_regex` false. Consequences
  taken in this branch: `model::agnes().vocab_used` = 248089, so Agnes's greedy argmax
  masks from 248089 (`argmax_stage1_*_V248089`) instead of Qwen3.8's 248077. Carried to the
  box: the engine's Rust tokenizer reads the snapshot's `tokenizer.json` (regex included)
  - `parity_test` on Agnes; the committed prompt ids (`tests/golden/prompts`, the A4 set)
  were tokenised by Qwen3.8 and must be re-made for Agnes where tokenisation matters (A4,
  passkey), while the golden gates only need the oracle and the engine to see the same ids.
- Observed, not Agnes-specific: the engine holds requests as `nlohmann::json`, whose objects
  are key-sorted, so tool schemas reach the template key-sorted where transformers keeps the
  caller's order (the committed references are rendered from sorted objects, like
  `template_tools.json`). A prompt-level difference from vLLM on the same request; recorded.

## Pending (the validation checklist carries each)

- The full checkpoint on disk, then: `load_agnes_test` (device), `test_agnes.py` check 4 and
  the `modeling_agnes.py` comparison (64-id prompt, cosine >= 0.99999, argmax at every
  position), the fold proof on real MLP inputs, the golden sets `oracle-out-agnes`, the MTP
  head reference `oracle-out-agnes-mtp`, A4 references.
- Every card gate (G0, G1 card, G2, G4, G5), the GEMV tuning of the two new shapes, speed.

## Phase 1 continued on the Mac (2026-10-04, Linux container)

The items above pending "for lack of transformers 5" ran in a `python:3.12-slim` container
on the same Mac (Docker Desktop, 16 vCPUs, capped at 28 GB; torch 2.14.1+cpu,
transformers 5.15.0 - the box image's version). Agnes in bf16 (~62 GB) never fits, so the
model is **layer-streamed**: built on `meta`, each decoder layer dequantised by the same
`dump.convert` rule when its forward starts and dropped after it (`tools/oracle/stream.py`,
`dump.py --stream`; setup and caveats in `tools/oracle/README.md`, "The Mac path"). Still no
card measurement.

| item | result |
|---|---|
| `test_agnes.py` | **4/4**. Check 4 failed first: `attach_parallel_ffn` built `Qwen3_5MLP(config)`, 5.15's signature is `(config, intermediate_size)` - `dump.py` would have died on the box too. Fixed. |
| `modeling_agnes.py` vs `agnes.py` | `long.ids[:64]`, eager, no cache, both streamed: per-position cos min 1.000000000, argmax 64/64, max\|d\| 0 - **bit-identical logits**. The remote file imports `transformers.cache_utils.LAYER_TYPE_CACHE_MAPPING`, which no released transformers (5.15-5.18) nor main has; shimmed as `{}` (it only registers cache layers, and no cache is used). |
| fold proof, real MLP inputs | layers 0 / 3 / 35 / 71, the 42 prose rows captured by `dump.py --mlp-in`: cos min 1.000000000, max\|d\| 0, weights exact - PASS. fp32 folded = two-branch **bitwise** on this torch/MKL (also on the synthetic rows, where torch 2.2 gave max\|d\| 1.5e-5): its K blocking splits at 17408. |
| golden sets `oracle-out-agnes` | prose / code / cjk, `--gen 32`, unfolded, eager. Continuations: prose " By eight the first trawlers would be back with the day's catch. By nine the whole town would be awake and arguing about something.\n\nI had"; code "\n\ndef clamp2(values, lo, hi):\n    return [min(max(v, lo), hi) for v in values]\n\ndef clamp3(values"; cjk "️\n\n我站在甲板上，看着对岸的轮廓一点点清晰起来。雾气还没散尽，远处的山像被水浸湿的墨迹。". 26 / 24 / 29 distinct of 32, `resid.L71` finite. |
| wall time per prompt | prose 2379 s (prompt forward 263 s, 32 steps 2100 s); code 3118 s (895 s + 2208 s, the Mac busy with other work); cjk 1069 s (255 s + 803 s). Peak RSS 22.1-22.5 GiB. |
| prompt ids | Agnes's `tokenizer.json` gives the committed `prose/code/cjk.ids` byte-identical (42 / 61 / 38); the golden gates keep `tests/golden/prompts`. |
| MTP head reference | `oracle-out-agnes-mtp/m1`, 32 rows per prompt, 561 s for three; continuation = the oracle's greedy 33 ids (`<name>.cont256.ids`); head top-1 vs the main model's next greedy id 29/32, 22/32, 21/32. |
| A4 set | `tests/golden/toolcall-agnes` (`make_set.py --from`, the same 36 conversations): 36/36 ids identical to Qwen3.8's. |
| P6 template | `dump_agnes_template.py` through `apply_chat_template`: `git diff tests/tokenizer` empty. |
| passkey | `passkey.py` at 60000: 59963 ids, needle at 3001 / 29953 / 56929. |

Two Mac-only performance fixes, both bit-exact and both in `stream.py`: a C++ `dequant_t`
(`dequant_gptq` + transpose, ~15x faster; every tensor's first dequant re-checked against
`dequant.py` on 64 columns), and grouped bf16 `conv1d` on one thread (torch 2.14's depthwise
bf16 conv is 50-4000x slower multi-threaded on this AVX2 CPU; outputs equal across thread
counts). One operational lesson: uncapped, the VM grew to 60 GB and macOS swapped 38 GB
(a decode step 360 s instead of 43 s) until Docker Desktop was restarted; the container
is now capped at 28 GB.

Left for the box: every device gate (G0, G1 card, G2 against these sets, G4 incl. M1 against
`oracle-out-agnes-mtp`, G5), `load_agnes_test`, `parity_test` on Agnes's tokenizer, the GEMV
tuning, A4 scoring and speed.


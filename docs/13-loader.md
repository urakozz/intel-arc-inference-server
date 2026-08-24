# The loader

One job: turn a HuggingFace checkpoint on disk into **the exact device buffers
the measured kernels were benchmarked against**, and fail loudly by tensor name
if the checkpoint does not mean what this project assumes. Everything the decode
loop needs is resident and canonical when `loader::load` returns; nothing is
converted, branched on, or discovered at run time.

`src/loader/{snapshot,safetensors,quant,loader}.{h,cc}`, driven by the model
description in `src/model/qwen35.{h,cc}` (doc 03) and the canonical layouts in
`src/common/repack.h` (doc 12). Measured on the box **2026-08-24** against
`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`.

## Snapshot resolution - and why it never downloads

`resolve_snapshot(arg)`:

1. If `arg` is a directory containing `config.json`, that is the snapshot.
2. Otherwise it must be an `<org>/<name>` repo id. It becomes
   `$HF_HOME/hub/models--<org>--<name>` (or `~/.cache/huggingface/hub/...`), the
   revision is read from `refs/main`, and the snapshot is
   `snapshots/<revision>/`.
3. Anything else throws, **naming the exact path it looked for**.

The server never fetches a model. Two reasons, both goal 3 ("own the
dependencies"): a download path is a network dependency inside the serving
binary and an implicit trust decision made at the worst possible moment, and a
server that can silently populate a cache can silently serve a different
revision than the one that was benchmarked. `hf download` is a separate,
explicit act. The error message names the command to run.

## The index is the manifest

`SafetensorsSet` reads `model.safetensors.index.json` and treats its
`weight_map` as **the** list of tensors: one entry per name, and the shard named
there is the shard that is opened. Each shard is mmapped read-only
(`MADV_SEQUENTIAL`) and its header parsed once; `data_offsets` are validated
against the file's actual data section, with the length check written as
`hlen > n - 8` rather than `8 + hlen > n` so a crafted 64-bit header length
cannot wrap past it.

This dedup rule is the **one lasting lesson of the 9B that was not**
(`letechlead/Ornith-1.5-9B-INT4-W4A16-AutoRound`, doc 03): it shipped a
`model_extra_tensors.safetensors` that overlapped the numbered shards, so a
loader that globbed `*.safetensors` and merged headers got two different tensors
under one name and no way to tell which was live. The 27B ships a
`model_extra_tensors.safetensors` too - the index points at it for some tensors
- and has no duplicated names, so it does not need the rule. It is kept because
the *next* checkpoint might, and because a wrong weight is a silent failure.

## Classification: suffixes decide, labels are never trusted

`LinearSrc::classify(set, prefix)` decides what a linear *is* by which tensors
exist under its prefix, never by what `config.json` claims:

| Found | Kind | Asserted |
|---|---|---|
| `<prefix>.qweight` | int4 GPTQ | `.scales` must exist; `qweight` dtype `I32`, `scales` dtype `F16`; `scales` shape exactly `[K/64][N]` |
| `<prefix>.weight` | bf16 | dtype `BF16` |
| neither | - | throws naming the prefix |

Doc 02's rule ("format plurality is a load-time problem") is why: the same
`quant_method: "gptq"` string covers checkpoints that ship different tensors,
and the `dynamic` rules that decide *which* modules were skipped are a regex
list that has already been observed to lie. This checkpoint carries **98
exclusion rules**, all `-:` - and the loader's real check on them is stronger
than parsing them: the model description says `in_proj_a`/`in_proj_b` are bf16
and everything else int4, and `load_linear` throws if a part classifies the
other way ("the checkpoint's dynamic exclusions moved"). A regex the loader
cannot read cannot mislead it.

`QuantConfig::parse` additionally throws unless `bits == 4`,
`group_size == 64`, `sym == true`, `desc_act == false`, and throws on any `+:`
dynamic rule - the exact pattern that made a published MTP head unusable
(BENCHMARKS.md). This checkpoint's copy has been corrected to `-:`.

### The two data asserts, and what measured them

`assert_quant_invariants(set)` runs **before a single byte is repacked** and
scans every word of every `.qzeros` and `.g_idx` in the checkpoint (0.202 GB,
under a second):

- **Every `qzeros` word is `0x77777777`.** GPTQ v1 stores *zero point − 1*, so
  the packed nibble `7` means a zero point of 8 - symmetric. Measured over all
  400 `qzeros` tensors of this checkpoint 2026-08-24: no other value occurs.
  This is what licenses the dequant `w = (q − 8)·scale` with no zero-point
  stream in the kernel, and (doc 02) what keeps the native s8×s4 DPAS prefill
  path open.
- **Every `g_idx[k] == k/64`.** `desc_act: false` means no activation-order
  permutation, so the group index must be the identity. Measured identical over
  all 400 `g_idx` tensors 2026-08-24. This is what licenses the layout-1 repack
  to compute a group index arithmetically instead of carrying a per-`k` table.

Both throw with tensor name, element index and the offending value. Neither
array is uploaded: **0.202 GB dropped at load**, and the report prints the
counts so the drop is visible rather than assumed.

One more assert exists because the loader `reinterpret_cast`s mmapped bytes to
`uint32_t`/`uint16_t`: every such pointer is checked for alignment and throws
naming the tensor. The safetensors data section is 8-aligned on every file seen
here, so this has never fired - a loud failure beats undefined behaviour on a
checkpoint that pads its header differently.

## The name view

The checkpoint has three top-level namespaces. The loader builds a view of
`SafetensorsSet::tensors()` in which:

- `model.language_model.` is **stripped**, so `model::Qwen35`'s layer-relative
  names (`layers.5.mlp.gate_proj`) bind directly;
- `lm_head.weight` is kept top level (it is genuinely outside the language
  model's namespace, and `tie_word_embeddings: false`);
- `model.visual.*` (333 tensors, 0.921 GB) is **skipped** - this is a
  vision-language checkpoint served text-only;
- `mtp.*` (15 tensors, 0.849 GB) is **skipped in v1** - MTP is phase 2 (doc 03).

Both skip counts are printed. So is the count of tensors the loader never
consumed - **0**, and the test asserts it (`report.unconsumed`). Every name in
the view is either loaded or deliberately dropped, and the drop is counted.
Nothing is silently ignored; if a future checkpoint grows a tensor this loader
does not know about, the number stops being 0 and both the report and the test
say so.

## Fusion: why these weights are welded together

The device holds **one buffer per fused linear**, not one per checkpoint tensor.
The table is `model::Qwen35` (doc 03); the loader executes it.

| Fused | Parts | Fuse | Why |
|---|---|---|---|
| `QkvZ` 5120→16384 | `in_proj_qkv` (10240) ‖ `in_proj_z` (6144) | concat | Both read the same `h` in the same GDN step. One GEMV instead of two means one weight stream, one launch, and - the reason it matters here - an `N` of 16384 that layout 1 handles and layout 0 does not (below). |
| `AB` 5120→128 | `in_proj_a` (48) ‖ `in_proj_b` (48), zero-padded | concat + pad | Two 0.5 MB bf16 GEMVs are pure launch overhead; fused they are one 1.3 MB read. `N = 96` is not a multiple of the 16-wide tile, so 32 zero rows pad it to 128. |
| `GateUp` 5120→34816 | `gate_proj` ‖ `up_proj` | **interleave in 16s** | See below. |
| `Qkv` 5120→14336 | `q_proj` (12288) ‖ `k_proj` (1024) ‖ `v_proj` (1024) | concat | Same `h`, same step; GQA 6:1 makes k and v tiny on their own. |
| `OutProj`, `OProj`, `Down`, `LmHead` | one each | single | Nothing to fuse. |

**Why `gate‖up` interleaves in blocks of 16 rather than concatenating.** The MLP
is `W_down · (silu(W_gate h) ⊙ W_up h)`: every output column `n` of `gate` is
multiplied by column `n` of `up`. A subgroup owns 16 consecutive output columns
(one n-tile). Under a plain concat, `up[n]` sits 17408 columns away from
`gate[n]` - a different tile, a different work-group, a round trip through
global memory before the elementwise product can happen. Interleaved as
`gate[0:16] up[0:16] gate[16:32] up[16:32] …`, the lane holding `gate[n]` finds
`up[n]` at a **fixed +16 output columns** - the adjacent tile, which the same
work-group can hold, so `silu(·) ⊙ ·` fuses into the GEMV epilogue instead of
becoming a second kernel over 34816 floats. `common::cols_interleave16` is the
column map; the repack does not care, it reads whatever map it is given.

The mechanism is `common::ColSource`: a fused linear's output column is a
`(qweight, scales, column, part-N)` tuple, and `repack_int4_layout1_cols` walks
the map. Concat and interleave are two functions building two maps over the same
repack.

### Preconditions, checked in release

The repack helpers document what they need but cannot check it. The loader
checks it at every call site - and these are real `throw`s, not
`assert`s, because a load happens once and a wrong weight is silent:

- every part shares the fused `K`, and `K` matches the shape table;
- the parts' `N` sums to the model description's unpadded width, and that is
  ≤ `shape.N` (padding, if any, is the difference);
- the column map has exactly `shape.N` entries;
- `K % 64 == 0`, `N % 16 == 0`; interleave gets two equal, 16-divisible parts;
- the staged bytes fit the reused staging buffer.

Every message names the linear (`GateUp (layers.7.mlp.gate_proj)`) - the
doc-03 shape table is therefore enforced *by construction*: a checkpoint with
different shapes cannot get past its own linear's name.

## What the loader bakes in

The kernels should do arithmetic, not bookkeeping. Four transformations happen
once at load:

**`1 + w` for every RMSNorm weight.** `Qwen3_5RMSNorm` is Gemma-style:
`x·rsqrt(mean(x²)+ε) ⊙ (1 + w)`, and the `+1` is **not** in the checkpoint
(doc 03, "Layer math"). Storing `1 + w` makes the kernel a plain RMSNorm - one
multiply, no constant to materialise per element. The bake is **one fp32 add and
one round-to-nearest-even cast back to bf16**: a single rounding, the same
discipline as every other conversion in this project. Applied to
`input_layernorm`, `post_attention_layernorm`, and FA's `q_norm`/`k_norm`.

**`linear_attn.norm.weight` is left alone.** `Qwen3_5RMSNormGated` is plain `w`
- it is the one norm in the model *without* the `+1`, and it multiplies by
`silu(z)` after normalising. Baking `+1` into it would be a silent accuracy bug
in all 48 GDN layers, which is why it is called out here and in the offset block
in `loader.cc`.

**`A_log → −exp(A_log)`, fp32.** The GDN decay is only ever used as
`g = −exp(A_log) ⊙ softplus(a + dt_bias)`. The `exp` is a per-head constant, so
it is hoisted out of 48 layers × every token. Stored fp32 because the recurrence
runs in fp32.

**`conv1d.weight` bf16 `[10240][1][4]` → fp32 `[10240][4]`.** The 4-tap
depthwise convolution accumulates in fp32; widening once at load costs 3.93 MB
of VRAM and saves a per-tap convert in the GDN kernel.

### The per-layer small-tensor blocks

Everything that is not a GEMV weight is packed into **two allocations per
layer** with compile-time offsets (`loader.cc`, `static_assert`ed on block size
and field alignment - the assert is what fails if this file is edited without
the kernels being told):

```
norms  (both kinds, 20480 B)     [0]      input_layernorm       (1+w) bf16 [5120]
                                 [10240]  post_attention_ln     (1+w) bf16 [5120]

GDN    (164480 B)                [0]      conv1d                fp32 [10240][4]
                                 [163840] -exp(A_log)           fp32 [48]
                                 [164032] dt_bias               fp32 [48]
                                 [164224] linear_attn.norm      plain w bf16 [128]

FA     (1024 B)                  [0]      q_norm                (1+w) bf16 [256]
                                 [512]    k_norm                (1+w) bf16 [256]
```

One block per layer instead of six allocations per layer keeps the descriptor
count down for plan 3's captured command list, and makes each layer's constants
one contiguous read.

The one norm that belongs to no layer - `model.language_model.norm.weight`, the
final RMSNorm before `lm_head` - gets its own 10 240 B allocation,
`LoadedModel::final_norm`, baked `(1 + w)` like all the rest.

### The RoPE table

`rope[p][0][i] = cos(p·θ^(−2i/64))`, `[1][i] = sin(…)`, `i < 32`, fp32,
`p < max_len` (default 16384; it sizes nothing else). `θ = 1e7`,
`partial_rotary_factor 0.25` of `head_dim` 256 → 64 rotary dims, 32 frequency
pairs. `rope_parameters` declares interleaved mRoPE with
`mrope_section [11,11,10]`, but in text mode the model expands one position id
onto every stream, so `apply_interleaved_mrope` copies each frequency onto
itself: **plain RoPE** (doc 03). The angles are computed in `double` and stored
`float` - 4.19 MB at the default length, a rounding error against 18 GB, and it
removes a transcendental from the FA prologue.

The table is **resident but not per-token traffic**: a decode step reads one
position's `2 × 32` floats, ~256 B, not the 4.19 MB. It is therefore reported on
its own line next to `embed_tokens` and kept out of the read-per-token figure on
both sides of the `W` cross-check - while still counted in `total()`, because
the report's buckets must account for every byte allocated.

## Layout and split-K come from the table, not from here

The loader does not choose layouts. `model::Qwen35`'s per-linear row carries
`(layout, S)` from the measured decision block of
`docs/probe-gemv-2026-08-24.md`:

| Linear | K × N | layout | S |
|---|---|---|---|
| `QkvZ` | 5120 × 16384 | 1 | 1 |
| `Qkv` | 5120 × 14336 | 1 | 1 |
| `GateUp` | 5120 × 34816 | 1 | 4 |
| `Down` | 17408 × 5120 | 1 | 16 |
| `OutProj` / `OProj` | 6144 × 5120 | 1 | 16 |
| `AB`, `LmHead` (bf16) | - | n/a | - |

All rows are layout 1 today, and layout 1 **loses four of the five int4 shapes**
(doc 12): it wins the sum only because `QkvZ` at `N = 16384` is the one
power-of-two stride where layout 0 collapses from ~550 to 419 GB/s. The
controller ruling of **2026-08-24** kept both repack paths rather than deleting
the loser, because flipping individual rows to layout 0 is a measured +3.5%
best-per-shape option - a phase-1 tuning knob to be measured end-to-end, not a
decision to be relitigated in the loader. The mechanism supports it: change one
row in `qwen35.cc` and the loader repacks that linear differently, with no other
edit anywhere. bf16 weights (`AB`, `LmHead`) have exactly one tiled layout and
carry `layout 0` as a filler.

## Memory and the `W` cross-check - measured 2026-08-24

Repack is single-threaded, staged through one reused host buffer per kind (the
largest int4 linear is `gate‖up` at 94.7 MB; the bf16 tile buffer is sized for
`lm_head`'s 2.54 GB and allocated once), and uploaded with **one synchronous
copy per linear**. `embed_tokens` skips staging entirely - it is gathered one
row per token, so row-major *is* its canonical layout and the mmap is copied
verbatim.

```
loader: .../snapshots/2a9077667e28aa53e61d91bdee5d7962e8674668/
  quant     int4 g64 sym desc_act=false, 98 dynamic exclusion rules
  tensors   2050 language-model + lm_head; skipped 333 visual, 15 mtp;
            dropped 400 qzeros + 400 g_idx (invariants asserted), 0 unconsumed
  linears   305 fused weights, 64 layers
  int4          12163481600 B   12.163 GB
  scales          760217600 B    0.760 GB
  bf16_linear      47185920 B    0.047 GB   (a‖b real rows; 0.063 GB with padding)
  lm_head        2542796800 B    2.543 GB
  small             9232384 B    0.009 GB   (per-layer blocks + final norm)
  pad              15728640 B    0.016 GB
  read/token    15538642944 B   15.539 GB
  embed          2542796800 B    2.543 GB   (resident, gathered - not per-token)
  rope              4194304 B    0.004 GB   (resident, ~256 B per token - not per-token)
  total         18085634048 B   18.086 GB
  W check     15.539 GB vs 15.539 GB expected = 15.519 doc-03 + 0.016 pad + 0.003941 widen
              ->  -0.000%
  load        13.1 s
```

Every uploaded byte lands in exactly one bucket, and the buckets sum to the
allocation total: a layout-1 int4 tile is 136 u32 = 128 of nibbles + 8 of
scales, so `K·N/2` and `K·N/32` split it exactly; `a‖b`'s 96 real rows and 32
zero rows split its buffer exactly.

**The `W` cross-check.** Doc 03 measured `W = 15.519 GB` read per decode token
from the checkpoint headers alone. The loader recomputes the same quantity from
what it actually uploaded and **throws if the two differ by more than 2%**,
printing the table first. The expected side is doc 03's figure plus this
loader's own itemised additions - never a widened tolerance:

| Addition | Bytes | Why |
|---|---|---|
| `a‖b` zero padding | 15 728 640 | 32 rows × 5120 × 2 B × 48 GDN layers, to reach the 16-wide tile |
| `conv1d` bf16→fp32 | 3 932 160 | 40960 taps × 2 B × 48 layers |
| `A_log`, `dt_bias` bf16→fp32 | 9 216 | 48 heads × 2 B × 2 tensors × 48 layers |

The RoPE table (4 194 304 B) is *not* in this list: it is resident but not
streamed per token, so it is excluded from both sides rather than added to both.

Measured delta: **−27 072 B, −0.00017%** - the loader is 27 KB *under* doc 03's
figure, which is the rounding in "15.519". Note the three totals the report
prints and why they differ: **read/token 15.539 GB** is what the decode step
streams; **total 18.086 GB** adds `embed_tokens` (gathered one row at a time,
~0 traffic) and the RoPE table (~256 B read per token). All three are printed,
labelled.

**Load time: 13.0-13.2 s** (warm page cache; 11.3 s of it is user CPU in the
single-threaded repack), against a 120 s target. No thread pool was built: the
known lever (a `std::async` pool over per-linear repack) is not worth its
complexity at 13 s. On a cold page cache this is bounded below by reading 19 GB
off the disk, and the figure should be re-measured before anyone calls it fast.

Device VRAM is 32.6 GB (doc 01), so 18.1 GB of weights leaves room for KV, GDN
state and activations without relaxed allocation games - though `l0::Mem` does
chain `ze_relaxed_allocation_limits_exp_desc_t` automatically for the two
allocations (`lm_head`, `embed_tokens`) that exceed the device's
`maxMemAllocSize`.

## Proving the bytes: the device round-trip

`tests/loader/load_checkpoint_test` (ctest label `checkpoint`, so it can be
excluded where the model or the card is absent) loads the real checkpoint,
asserts the counts (305 fused linears = 48×5 + 16×4 + 1, 64 small-tensor
blocks) and the byte buckets, and then does the thing that actually matters:
it **reads the first four n-tiles of layer 0's `QkvZ` back off the device** and
compares them against a CPU repack of the mmapped source, computed
independently in the test. Zero mismatches.

That check rides on a property of layout 1: tiles are ordered `n_tile`-outer,
`k_group`-inner, so the first `NT·(K/64)·136` u32 of the buffer *are* n-tiles
0..3, and the test can verify a slice without repacking 16 384 columns. If the
loader ever ordered them differently the test would be wrong rather than the
loader - it is written down here so that is a decision and not a surprise.

The tile check covers the int4 path only, so three more readbacks pin the
transformations it cannot see - each one a value the kernels would otherwise
have to take on trust:

- **the `(1+w)` bake**: layer 0's `input_layernorm[0]` on the device equals
  `f32_to_bf16(1.0f + bf16_to_f32(src))` computed independently - the single
  rounding, verified rather than asserted;
- **the `a‖b` padding**: rows 96..127 of the tiled `[128][5120]` buffer read
  back as zero, all 163 840 of them, so the 32 pad rows contribute nothing to
  the GEMV;
- **the RoPE table**: `p = 0` is exactly `(1, 0)` and `p = 1`'s first frequency
  is `(cos 1, sin 1)` - `inv_freq[0] = θ⁰ = 1`, so this catches a wrong `θ`
  exponent sign or a transposed `[2][32]`.

And `report.unconsumed == 0` is asserted, which is what makes the "nothing is
skipped by accident" claim testable instead of rhetorical.

## Deliberately not loaded

- **`model.visual.*`** (333 tensors, 0.921 GB) - this checkpoint is a VLM; the
  project serves text. Phase scope, doc 03.
- **`mtp.*`** (15 tensors, 0.849 GB) - speculative decoding is phase 2. The
  head is shipped and correct in this checkpoint; v1 simply does not build a
  draft path, and counting the skip in the report is how that stays a decision
  rather than an oversight.
- **`*.qzeros`, `*.g_idx`** (800 tensors, 0.202 GB) - fully scanned to prove the
  invariants above, then dropped. They are constants; uploading them would cost
  0.2 GB of every token's bandwidth to re-read a value the kernel already knows.

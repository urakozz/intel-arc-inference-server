# The loader

This document explains how the engine turns a Hugging Face checkpoint on disk into the weights the kernels read
on the card. It is for readers who want to know what the engine assumes about a checkpoint, how those assumptions
are checked, and where every byte of device memory goes.

The loader has one job: produce exactly the device buffers the kernels were benchmarked against, and fail loudly,
naming the tensor, if the checkpoint does not mean what the engine assumes. When `loader::load` returns, everything
the decode loop needs is on the card in its final layout. Nothing is converted, branched on or discovered while the
model runs.

## Map of a load

The walk-through uses Qwen3.8-27B; the other families follow the same rules. A load runs these steps, each with its
own section below:

1. find the snapshot in the Hugging Face cache, never downloading ("Finding the snapshot");
2. read the tensor list from the index file ("The index file is the list of tensors");
3. classify every linear from the tensors present, not the config labels, and check the quantisation invariants on
   every word ("Classification", "The three data checks");
4. map checkpoint names onto the model description, skipping the vision tower and, unless asked for, the MTP head
   ("The name view");
5. fuse linears that read the same input, repack them into the layout each shape was tuned for, and bake constants
   such as RMSNorm's `1 + w` ("Fusion", "What the loader bakes in", "Layout and split-K");
6. upload, then cross-check the bytes per token against the figure computed from the headers ("Memory and the `W`
   cross-check"), with a test that reads the weights back off the card ("Proving the bytes").

The later sections cover the other checkpoint forms the loader accepts: a self-quantised checkpoint with an int4
head, g128 groups, compressed-tensors, and Qwen3.8-Flash-Next.

The code is `src/loader/{snapshot,safetensors,quant,loader}.{h,cc}` plus `src/loader/small_layout.h`, which holds
the small-tensor block offsets shared with the model description and the device kernels. The loader is driven by
the model description in `src/model/qwen35.{h,cc}` ([03-models.md](03-models.md)) and the weight layouts in
`src/common/repack.h` ([12-kernels.md](12-kernels.md)). The figures below come from real checkpoint files, as
reported by an earlier build; the load is not re-measured on every change.

## One loader per model family

Each model family has its own loader. They share the snapshot lookup, the safetensors reader, the quantisation
config parser and `LinearSrc::classify`, all described below.

| Loader | Models | Files |
|---|---|---|
| `load`, and `load_tp`, which slices every layer for tensor parallel so each card takes half | Qwen3.8-27B (dense) and Ornith 1.5 35B-A3B (a mixture of experts, or MoE: many small expert MLPs, of which a few run per token) | `loader.cc`, `moe.*` |
| `load_kolibri1` | Kolibri-1 | `kolibri1_*` |
| `load_qwen4exp` | Qwen3.8-Flash-Next (see the section near the end) | `qwen4exp_*` |
| `load_drafter` | the DFlash draft model: the bf16 original, quantised to int8 at load, or our packed int4 export, loaded as shipped | `dflash_*`, `nvfp4.h` |

### Three forms of Qwen3.8's output head

Qwen3.8 comes in two checkpoint shapes, and the loader takes both without a conditional anywhere in the walk. The
published checkpoint keeps the output head (`lm_head`, the final projection from the hidden state to one score per
vocabulary id) in bf16. A self-quantised checkpoint packs the head at int4; `tools/quantize_qwen38_rtn.sh` makes
one. Both quantise scripts run auto-round 0.17.0 at commit `6afaecdb`. `tools/quantize/README.md`, section
"AutoRound version", records why the RTN script needs `--disable_low_cpu_mem_usage`: 0.17.0's default RTN path ships
`lm_head.weight` unpacked while its `extra_config` declares it 4-bit. Where the two shapes differ, this document says
which one it means, and the section "The packed-head checkpoint" lists every difference, measured against the files.

The third form is made at load and is the one the command-line tools run. `load(..., LmHeadForm::Int8)` reads the
bf16 `lm_head` and quantises it on the host (`src/loader/lm_head_int8.h`): per row, `s = max|row| / 127` in fp32 and
`q = rne(w / s)` clamped to [-127, 127], using every host thread, in 0.35-0.75 s. The result goes into the
256-byte `[16 k][16 n]` int8 tiles that `gemv_i8w` reads, ordered `[K/16/8][N/16][8]` at an output head's width
and `[N/16][K/16]` otherwise (`loader::int8_tiled_index`),
with the fp32 `[N]` scales in `DeviceWeight::scales`.
The report's `lm_head` line names the form and the quantisation time, and the `W` cross-check (below) itemises the
-1.270 GB it saves. A checkpoint whose head is already int4 is refused.

## Finding the snapshot, and why the engine never downloads

`resolve_snapshot(arg)` works in three steps:

1. If `arg` is a directory containing `config.json`, that directory is the snapshot.
2. Otherwise `arg` must be an `<org>/<name>` repo id. It becomes `$HF_HOME/hub/models--<org>--<name>` (or
   `~/.cache/huggingface/hub/...`), the revision is read from `refs/main`, and the snapshot is
   `snapshots/<revision>/`
3. Anything else throws, naming the exact path it looked for.

The server never fetches a model, for two reasons. A download path would put a network dependency inside the serving
binary and make a trust decision at the worst possible moment. And a server that can quietly fill its own cache can
quietly serve a different revision from the one that was benchmarked. `hf download` is a separate, explicit act, and
the error message names the command to run.

## The index file is the list of tensors

`SafetensorsSet` reads `model.safetensors.index.json` and treats its `weight_map` as the list of tensors: one entry
per name, and the shard named there is the shard that is opened. Each shard is memory-mapped read-only
(`MADV_SEQUENTIAL`) and its header parsed once. Every tensor's `data_offsets` are checked against the file's actual
data section. The length check is written as `hlen > n - 8` rather than `8 + hlen > n`, so a crafted 64-bit header
length cannot wrap around it.

Trusting the index, rather than globbing `*.safetensors`, comes from a 9B checkpoint that was dropped
(`letechlead/Ornith-1.5-9B-INT4-W4A16-AutoRound`). It shipped a `model_extra_tensors.safetensors` that overlapped
the numbered shards, so a loader that merged every header found two different tensors under one name and no way to
tell which was live. The 27B also ships a `model_extra_tensors.safetensors`, and the index points at it for some
tensors, but no name appears twice. The rule stays because a wrong weight fails silently.

## Classification: the tensors decide, the config labels do not

`LinearSrc::classify(set, prefix)` decides what kind of linear layer it is looking at from the tensors that exist
under its prefix, never from what `config.json` claims:

| Found | Kind | Checked |
|---|---|---|
| `<prefix>.qweight` | int4 GPTQ | `.scales` must exist; `qweight` dtype `I32`, `scales` dtype `F16`; `scales` shape exactly `[K/64][N]` (or `[K/128][N]`, see "g128 checkpoints") |
| `<prefix>.weight_packed` | int4 compressed-tensors, converted to the GPTQ pair | `.weight_scale` and `.weight_shape` must exist, no `.weight_zero_point`; see "compressed-tensors symmetric checkpoints" |
| `<prefix>.weight` | bf16 | dtype `BF16` |
| none of these | - | throws, naming the prefix |

The reason is [02-formats.md](02-formats.md)'s rule that the variety of formats is a load-time problem. The same
`quant_method: "gptq"` string covers checkpoints that ship different tensors, and the `dynamic` rules that say which
modules were left unquantised are a list of regular expressions that has already been seen to lie. The Qwen3.8
checkpoint carries 98 exclusion rules, all of the `-:` (exclude) kind. The loader does not parse them; it does
something stronger. The model description says `in_proj_a` and `in_proj_b` are bf16 and everything else int4, and
`load_linear` throws if a part classifies the other way ("the checkpoint's dynamic exclusions moved"). A regex the
loader never reads cannot mislead it.

`a‖b`'s kind, like `lm_head`'s, belongs to the checkpoint. Ornith's int4 checkpoint
(`urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`) quantises `in_proj_a` and `in_proj_b` at int4 g64, where
Qwen3.8's export keeps them bf16. The loader classifies the first delta-net layer's `in_proj_a` by content
(`loader/ab.h`) and takes either the table's bf16 `AB` row or the descriptor's `ab_int4` row. Both keep the same
128-column contract: the real columns are zero-padded with the int4 zero (nibble 8, scale +0) and read by the
layout-1 GEMV in `gemv.cl`, which writes `ab_out` exactly where `gemv_bf16` did. Every delta-net layer must agree; a
layer in the other form throws, by name. For prefill the loader also keeps a bf16 copy, dequantised once with the
prefill dequant's arithmetic (`LoadedModel::ab_prefill`, counted in `total()` but not read per token), which
`pf_ab_proj` reads unchanged.

`QuantConfig::parse` adds its own checks. It throws unless `bits == 4`, the group size is 64 (or 128, see "g128
checkpoints"), `sym == true`, and `desc_act` is false or absent. Absence is auto-round's way of writing false, and the
loader proves it from the shipped `g_idx` count instead of taking it on trust (see "The packed-head checkpoint"). It
throws on any `+:` (include) dynamic rule that is not an exact restatement of the top-level settings; that pattern is
what made a published MTP head unusable. It also throws on an unrecognised `quant_method` or
`packing_format`, and on an `extra_config` module that claims to be packed at anything but g64 symmetric int4.

### The three data checks

`assert_quant_invariants(set)` runs before a single byte is repacked. It scans every word of every `.qzeros`,
`.g_idx` and `.scales` tensor in the checkpoint: 0.202 GB of the first two and 0.760 GB of scales, about 1 s in
total. Each check guards an assumption a kernel relies on.

Every `qzeros` word must be `0x77777777`. GPTQ v1 stores the zero point minus one, so the packed nibble `7` means a
zero point of 8, which is symmetric quantisation. Over all 400 `qzeros` tensors of the checkpoint, no other value
occurs. This is what allows the dequantisation `w = (q − 8)·scale` with no zero-point stream in the kernel, and
([02-formats.md](02-formats.md)) what keeps the native s8×s4 DPAS prefill path open. DPAS is the Xe matrix-engine
instruction that multiplies small tiles of 8-bit or 16-bit values.

Every `g_idx[k]` must equal `k/64`. `desc_act: false` means the columns were not reordered by activation size, so the
group index must be the identity. It is, over all 400 `g_idx` tensors. This is what lets the kernels compute a
group index arithmetically instead of carrying a per-`k` table.

Every `.scales` value must be a finite f16. The dequantisation `scale·(q − 8)` has no guard downstream, so one NaN or
Inf scale would silently poison a group of 64 weights. On the published checkpoint 1658 of the 380,108,800 scales
are subnormal f16 (the first is `layers.10.linear_attn.in_proj_qkv.scales[1444] = 0x00A8`, about 1.0e-5).
Subnormals are counted and reported, not rejected. They are exact f16 values, the kernel reads the f16 word
natively, and `common::f16_to_f32` decodes them exactly (`man · 2⁻²⁴`) instead of flushing them to zero, which would
make the host reference disagree with both the kernel and the CPU oracle on 1658 groups.

That last fact binds every device kernel: no kernel may be compiled with `-cl-denorms-are-zero` or any other option
that flushes f16 denormals. Flushing would zero 1658 real scales and silently diverge from both the loader's host
reference and the CPU oracle. The build makes that flag a fatal error rather than a convention.

All three checks throw with the tensor name, the element index and the offending value. Neither `qzeros` nor `g_idx`
is uploaded, which drops 0.202 GB at load. The report prints the counts, and the subnormal count, so the drop is
visible rather than assumed.

One more check exists because the loader `reinterpret_cast`s memory-mapped bytes to `uint32_t` and `uint16_t`: every
such pointer is checked for alignment, and a misaligned one throws naming the tensor. The safetensors data section has
been 8-byte aligned in every file seen so far, so this has never fired, but a loud failure beats undefined behaviour
on a checkpoint that pads its header differently.

## The name view

The checkpoint has three top-level namespaces. The loader builds a view of `SafetensorsSet::tensors()` with these
rules:

- `model.language_model.` is stripped, so `model::Qwen35`'s layer-relative names (`layers.5.mlp.gate_proj`) bind
  directly.
- `lm_head.*` stays top level. It really is outside the language model's namespace, and `tie_word_embeddings` is
  false. That is one tensor (`.weight`) on the published checkpoint and three (`.qweight`, `.qzeros`, `.scales`) on
  one that packs the head. The drop counters cover this branch too, so a packed head's `qzeros` is reported as dropped
  rather than quietly missing from the tally, and the report prints the language-model and top-level counts
  separately.
- `model.visual.*` (333 tensors, 0.921 GB) is skipped. The checkpoint is a vision-language model served text-only.
- `mtp.*` is skipped unless the caller asks for it with `load(..., mtp = true)`. MTP (multi-token prediction) is
  Qwen3.8's small built-in draft head, used for speculative decoding. It is 15 tensors (0.849 GB) on the published
  checkpoint and 29 on the self-quantised one, which also packs 8 of them; the skip is by name and does not care.
  When asked for, the published 15 are loaded into `LoadedModel::mtp` in bf16, in `gemv_bf16`'s tiled layout: `fc`;
  q, k and v concatenated like a full-attention layer's `Qkv`; gate and up interleaved in 16-column blocks like
  `GateUp`; `o`; `down`. Its five RMSNorms and its q and k norms are baked to fp32 `1 + w`. The report gains an `mtp`
  line. Any other shape of head, such as the RTN checkpoint's 29 tensors, is refused by name.

Both skip counts are printed, and so is the count of tensors the loader never consumed: 0, which the test asserts
(`report.unconsumed`). Every name in the view is either loaded or deliberately dropped, and every drop is counted. If a
future checkpoint grows a tensor the loader does not know, the number stops being 0 and the report names up to five
of the unconsumed tensors (then `…`), so a checkpoint that grew a whole family says which family it is.

## Fusion: why some weights are joined

The device holds one buffer per fused linear, not one per checkpoint tensor. The table lives in `model::Qwen35`
([03-models.md](03-models.md)); the loader carries it out. GDN below is the gated delta net, Qwen3.8's linear-attention
layer, which keeps a fixed-size recurrent state instead of a growing cache; FA is a full-attention layer.

| Fused | Parts | How | Why |
|---|---|---|---|
| `QkvZ` 5120→16384 | `in_proj_qkv` (10240) ‖ `in_proj_z` (6144) | concat | Both read the same input `h` in the same GDN step. One GEMV (matrix-vector product) instead of two means one weight stream and one launch, and it gives an `N` of 16384, the one shape where layout 1 beats layout 0 (below) |
| `AB` 5120→128 | `in_proj_a` (48) ‖ `in_proj_b` (48), zero-padded | concat and pad | Two 0.5 MB bf16 GEMVs are pure launch overhead; fused they are one 1.3 MB read. `N = 96` is not a multiple of the 16-wide tile, so 32 zero rows pad it to 128. |
| `GateUp` 5120→34816 | `gate_proj` ‖ `up_proj` | interleave in 16s | See below |
| `Qkv` 5120→14336 | `q_proj` (12288) ‖ `k_proj` (1024) ‖ `v_proj` (1024) | concat | Same `h`, same step; grouped-query attention at 6:1 makes k and v tiny on their own |
| `OutProj`, `OProj`, `Down`, `LmHead` | one each | single | Nothing to fuse |

Gate and up are interleaved in blocks of 16 rather than concatenated because of how the MLP uses them. The MLP is
`W_down · (silu(W_gate h) ⊙ W_up h)`: output column `n` of `gate` is multiplied by column `n` of `up`. A subgroup (the
16 lanes that execute together on an Xe core) owns 16 consecutive output columns. With a plain concat, `up[n]` would
sit 17408 columns away from `gate[n]`, in a different tile and a different work-group, so the product would need a
round trip through global memory. Interleaved as `gate[0:16] up[0:16] gate[16:32] up[16:32] …`, the lane holding
`gate[n]` finds `up[n]` exactly 16 output columns away, in the adjacent tile, which the same work-group holds. So
`silu(·) ⊙ ·` fuses into the GEMV's epilogue instead of becoming a second kernel over 34816 floats.
`common::cols_interleave16` builds that column map.

The mechanism is `common::ColSource`: each output column of a fused linear is a `(qweight, scales, column, part-N)`
tuple, and the repack functions (`repack_int4_layout0_cols`, `repack_int4_layout1_cols`) walk a column map. Concat and
interleave are two functions building two maps over the same repack; the repack itself does not care which map it is
given.

### Preconditions, checked in release builds

The repack helpers document what they need but cannot check it, so the loader checks it at every call site. These are
real `throw`s, not `assert`s, because a load happens once and a wrong weight is silent:

- every part shares the fused `K`, and `K` matches the shape table;
- the parts' `N` values sum to the model description's unpadded width, which is at most `shape.N` (any difference is
  padding);
- the column map has exactly `shape.N` entries;
- `K % 64 == 0` and `N % 16 == 0`, and an interleave gets two equal parts divisible by 16;
- the staged bytes fit the reused staging buffer.

Every message names the linear, for example `GateUp (layers.7.mlp.gate_proj)`. The shape table in
[03-models.md](03-models.md) is therefore enforced by construction: a checkpoint with different shapes cannot get past
its own linear's name.

## What the loader bakes in

Kernels should do arithmetic, not bookkeeping, so four transformations happen once at load.

The first is `1 + w`, stored in fp32, for every RMSNorm weight. `Qwen3_5RMSNorm` is Gemma-style,
`x·rsqrt(mean(x²)+ε) ⊙ (1 + w)`, and the `+1` is not in the checkpoint ([03-models.md](03-models.md), "Layer math").
Storing `1 + w` makes the kernel a plain RMSNorm with one multiply. The bake is one fp32 add, stored as fp32, with no
cast back to bf16. The Hugging Face reference computes the whole of `x·rsqrt(…)·(1 + w)` in fp32 and never rounds the
multiplier. Storing `1 + w` as bf16 would add a rounding the reference does not have, and it would land where bf16 is
weakest: `w` is small, so `1 + w` sits just above 1.0, where bf16's 8-bit mantissa steps by 2⁻⁸. That is up to 0.39%
error on the multiplier at about 130 sites (128 layer norms and the final norm), compounding down 64 layers. fp32
removes the rounding and costs 1,337,344 B of VRAM, 0.0086% of the weights. The bake applies to `input_layernorm`,
`post_attention_layernorm`, the full-attention layers' `q_norm` and `k_norm`, and the final `norm`

`linear_attn.norm.weight` is deliberately left alone: plain `w`, still bf16. `Qwen3_5RMSNormGated` is the one norm in
the model without the `+1`, and it multiplies by `silu(z)` after normalising. Baking `+1` into it would be a silent
accuracy bug in all 48 GDN layers. It also stays bf16, for the mirror image of the argument above: the reference's own
parameter is bf16 and it multiplies in the bf16 domain, so widening it would move away from the reference, not
towards it. `src/loader/small_layout.h` calls out both facts too.

The second is `A_log → −exp(A_log)`, in fp32. The GDN decay is only ever used as `g = −exp(A_log) ⊙ softplus(a +
dt_bias)`. The `exp` is a per-head constant, so it is hoisted out of 48 layers times every token. It is stored in fp32
because the recurrence runs in fp32.

The third is `conv1d.weight`, bf16 `[10240][1][4]`, widened to fp32 `[10240][4]`. The 4-tap depthwise convolution
accumulates in fp32; widening once at load costs 3.93 MB of VRAM and saves a convert per tap in the GDN kernel.

The fourth is the RoPE table, described below.

### The per-layer small-tensor blocks

Everything that is not a GEMV weight is packed into two allocations per layer at compile-time offsets. Those offsets
live in one file, `src/loader/small_layout.h`, with a `static_assert` on every offset, every block size and every
field alignment; the assert is what fails if the layout is edited without the kernels being told. Three consumers read
that header and none re-derives a number: the loader packs against it, the model description's small-tensor table
carries its constants as destination offsets, and the GDN, attention and norm kernel bindings include it to find each
field in the block they are given.

```
norms  (both kinds, 40960 B)     [0]      input_layernorm       (1+w) fp32 [5120]
                                 [20480]  post_attention_ln     (1+w) fp32 [5120]

GDN    (164480 B)                [0]      conv1d                fp32 [10240][4]
                                 [163840] -exp(A_log)           fp32 [48]
                                 [164032] dt_bias               fp32 [48]
                                 [164224] linear_attn.norm      plain w bf16 [128]

FA     (2048 B)                  [0]      q_norm                (1+w) fp32 [256]
                                 [1024]   k_norm                (1+w) fp32 [256]
```

One block per layer instead of six allocations per layer keeps down the number of buffers the captured command list
binds, and makes each layer's constants one contiguous read. The one norm that belongs to no layer,
`model.language_model.norm.weight` (the final RMSNorm before `lm_head`), gets its own 20,480 B allocation,
`LoadedModel::final_norm`, baked to fp32 `1 + w` like the rest.

The model description owns this table, and the loader just walks it. Each entry of `model::LayerDesc::small_tensors`
carries the layer-relative name, the exact element count, the required source dtype, which of the two blocks it lands
in, its byte offset there (a `small_layout.h` constant) and its bake kind: `one_plus_w_fp32`, `plain_bf16`,
`neg_exp_fp32`, `raw_fp32_widen` or `raw_fp32`. `loader::load_small` hardcodes no name, dtype or offset, so adding a
small tensor is a row in `qwen35.cc` plus, if it moves the layout, a constant in `small_layout.h`. Two checks keep the
pair honest at load: an entry that would overrun its block throws naming the tensor, and a block the entries do not
tile exactly throws naming the layer. `tests/model/qwen35_test` checks the six GDN and four attention rows, their
offsets and their bakes without touching a device.

### The RoPE table

RoPE (rotary position embedding) rotates pairs of query and key dimensions by an angle that depends on the position.
The loader precomputes the angles: `rope[p][0][i] = cos(p·θ^(−2i/64))` and `rope[p][1][i] = sin(…)` for `i < 32`, in
fp32, for every position `p` below the run's `max_len` (the table sizes nothing else). `θ = 1e7`, and
`partial_rotary_factor 0.25` of `head_dim` 256 gives 64 rotary dimensions, or 32 frequency pairs. `rope_parameters`
declares interleaved multi-axis RoPE with `mrope_section [11,11,10]`, but in text mode the model puts the same
position id on every axis, so `apply_interleaved_mrope` copies each frequency onto itself and the result is plain
RoPE ([03-models.md](03-models.md)). The angles are computed in `double` and stored as `float`: 4.19 MB at a 16,384
context, small against 18 GB of weights, and it removes a transcendental function from the attention kernel's
prologue.

The table is resident but not read in full per token: a decode step reads one position's `2 × 32` floats, about
256 B. So it gets its own line in the report next to `embed_tokens` and stays out of the read-per-token figure on both
sides of the `W` cross-check, while still counting in `total()`, because the report's buckets must account for every
byte allocated.

## Layout and split-K come from the model description

The loader does not choose weight layouts. Each int4 row of the model description carries a `(layout, S)` pair taken
from the measured per-shape decision in [12-kernels.md](12-kernels.md) (`kQwen38Tuning` in
`src/model/model_desc.cc`). Layout 0 is GPTQ's own: `qweight[K/8][N]` words plus a separate `scales[K/64][N]`
allocation. Layout 1 is the subgroup-tiled repack, with each 64-row group's 16 scales inline in its 136-word tile. `S`
is the split-K count: how many work-groups share one column's dot product.

| Linear | K × N | layout | S |
|---|---|---|---|
| `QkvZ` | 5120 × 16384 | 1 | 1 |
| `Qkv` | 5120 × 14336 | 0 | 2 |
| `GateUp` | 5120 × 34816 | 0 | 8 |
| `Down` | 17408 × 5120 | 0 | 4 |
| `OutProj` / `OProj` | 6144 × 5120 | 0 | 4 |
| `AB`, `LmHead` (bf16, or int8 at load) | - | n/a | - |
| `LmHead` (int4, if the checkpoint packed it) | 5120 × 248320 | 1 | 1 |

Layout 1 is kept only for `QkvZ`: at `N = 16384` layout 0's row stride is a power of two and its bandwidth collapses
to 419 GB/s, while layout 1 reaches 559 ([12-kernels.md](12-kernels.md)). For the other shapes layout 0 is faster and
needs no repack beyond the fusion column map, so most weights reach the card in the checkpoint's own geometry.

`load_linear` implements both layouts. For a layout-0 row it uploads two allocations, the words and
`DeviceWeight::scales`; for layout 1 it uploads one, and `DeviceWeight::scales` stays null. Any other layout value
throws, naming the linear, so a typo cannot silently take layout 1's bytes. bf16 weights (`AB`, a bf16 `LmHead`) have
one tiled layout and carry `layout 0` as a filler, which is why the check applies to int4 rows only.

## Memory and the `W` cross-check

The repack is single-threaded and staged through one reused host buffer per kind, sized from the linears this load
will repack (see "Staging is sized from the linears this load will repack"). With a bf16 head the largest buffers are
`gate‖up`'s 94.7 MB of int4 words and scales and `lm_head`'s 2.54 GB of bf16 tiles; with a packed head the two swap
ends. Each linear is uploaded with one synchronous copy. `embed_tokens` skips staging entirely: it is gathered one row
per token, so row-major is already its final layout and the memory-mapped bytes are copied verbatim.

The report the load prints:

```
loader: .../snapshots/<revision>/
  quant     int4 g64 sym desc_act=false, 98 dynamic exclusion rules
  tensors   2050 language-model + lm_head; skipped 333 visual, 15 mtp;
            dropped 400 qzeros + 400 g_idx (invariants asserted), 0 unconsumed
  scales    1658 subnormal f16 (exact on device and in f16_to_f32; not an error)
  linears   305 fused weights, 64 layers
  int4          12163481600 B   12.163 GB
  scales          760217600 B    0.760 GB
  bf16_linear      47185920 B    0.047 GB   (a‖b real rows; 0.063 GB with padding)
  lm_head        2542796800 B    2.543 GB
  small            10569728 B    0.011 GB   (per-layer blocks + final norm)
  pad              15728640 B    0.016 GB
  read/token    15539980288 B   15.540 GB
  embed          2542796800 B    2.543 GB   (resident, gathered - not per-token)
  rope              4194304 B    0.004 GB   (resident, ~256 B per token - not per-token)
  total         18086971392 B   18.087 GB
  W check     15.540 GB vs 15.540 GB expected = 15.519 doc-03 + 0.016 pad + 0.005279 widen
              widen = 1337344 B RMSNorm fp32 (1+w) + 3941376 B GDN fp32  ->  -0.000%
  load        13.6 s
```

Every uploaded byte lands in exactly one bucket, and the buckets sum to the allocation total. An int4 weight is
`K·N/2` bytes of nibbles plus `K·N/32` bytes of f16 scales in either layout (a layout-1 tile is 136 u32: 128 of
nibbles and 8 of scales), so the `int4` and `scales` lines split it exactly. `a‖b`'s 96 real rows and 32 zero rows
split its buffer exactly between `bf16_linear` and `pad`

`W` is the number of bytes one decode token reads, which sets the bandwidth ceiling on decode speed.
[03-models.md](03-models.md) measured `W = 15.519 GB` from the checkpoint headers alone. The loader recomputes the same
quantity from what it actually uploaded and throws if the two differ by more than 2%, printing the table first. The
expected side is the 03-models figure plus this loader's own itemised additions, never a widened tolerance:

| Addition | Bytes | Why |
|---|---|---|
| `a‖b` zero padding | 15,728,640 | 32 rows × 5120 × 2 B × 48 GDN layers, to reach the 16-wide tile |
| `conv1d` bf16→fp32 | 3,932,160 | 40960 taps × 2 B × 48 layers |
| `A_log`, `dt_bias` bf16→fp32 | 9,216 | 48 heads × 2 B × 2 tensors × 48 layers |
| both layer norms bf16→fp32 | 1,310,720 | 5120 × 2 B × 2 norms × 64 layers |
| attention `q_norm`, `k_norm` bf16→fp32 | 16,384 | 256 × 2 B × 2 norms × 16 attention layers |
| final `norm` bf16→fp32 | 10,240 | 5120 × 2 B, once |

The last three rows are the fp32 RMSNorm store, 1,337,344 B in total, which the report's `widen` line itemises as
"RMSNorm fp32 (1+w)" next to the GDN family's 3,941,376 B. Widening is a byte cost, never a tolerance: both sides of
the cross-check move by the same amount. The RoPE table (4,194,304 B) is not in the list; it is resident but not
streamed per token, so it is left out of both sides rather than added to both.

The measured difference is -27,072 B, or -0.00017%: the loader is 27 KB under the 03-models figure, which is the
rounding in "15.519". The report prints three totals, and they differ for a reason. `read/token` (15.540 GB) is what a
decode step streams. `total` (18.087 GB) adds `embed_tokens`, gathered one row at a time for almost no traffic, and
the RoPE table, about 256 B a token.

The load took 13.6 s on a warm page cache, against a 120 s budget; a cold cache is bounded below by reading 19 GB off
the disk. A `std::async` pool over the per-linear repack is the obvious way to speed it up and is not worth its
complexity at 13 s.

Device VRAM is 32.6 GB ([01-hardware.md](01-hardware.md)), so 18.1 GB of weights leaves room for the KV cache (the
stored keys and values of every past position), the delta-net state and the activations. `l0::Mem` adds
`ze_relaxed_allocation_limits_exp_desc_t` automatically for the two allocations (`lm_head`, `embed_tokens`) that
exceed the device's `maxMemAllocSize`

## Proving the bytes: the device round trip

`tests/loader/load_checkpoint_test` (ctest label `checkpoint`, so it can be excluded where the model or the card is
absent) loads the real checkpoint and checks the counts (305 fused linears = 48×5 + 16×4 + 1, and 64 small-tensor
blocks) and the byte buckets. Then it does the check that matters most: it reads weights back off the device and
compares them with bytes it computes itself from the memory-mapped source.

For layout 1 it reads the first four n-tiles of layer 0's `QkvZ` and compares them with an independent CPU repack:
zero mismatches. That relies on a property of layout 1: tiles are ordered `n_tile` outer and `k_group` inner, so the
first `NT·(K/64)·136` u32 of the buffer are n-tiles 0 to 3, and the test can check a slice without repacking 16,384
columns. If the loader ever ordered them differently, the test would be wrong rather than the loader; this paragraph
records that as a decision. For layout 0 it reads the first and last rows of words and scales at boundary columns of
several layout-0 linears, with the expected values taken straight from the checkpoint parts rather than from the
repack's column helpers, and checks that the scales are a second, separate allocation.

The small blocks are read back field by field at their `small_layout.h` offsets. Every bake kind is covered, and each
check would fail if the block layout moved:

- The norms block, both halves: `input_layernorm[0]` at `[0]` and `post_attention_layernorm[0]` at `[20480]` equal
  `1.0f + bf16_to_f32(src)` as fp32. That one comparison proves the bake, the fp32 store (a bf16 store would not
  compare equal) and the second half's offset.
- Layer 0's GDN block, one field per bake: `conv[0]` at `[0]` equals `bf16_to_f32(src conv1d[0])`; `negA[0]` at
  `[163840]` equals `−exp(bf16_to_f32(src A_log[0]))`; `dt_bias[0]` at `[164032]` equals `bf16_to_f32(src)`; and the
  gated norm at `[164224]` is the checkpoint's raw bf16 word, neither `1+w` nor widened. That last check is the one
  that would catch someone "fixing" the inconsistency.
- A full-attention layer's `k_norm`: layer 3's, fp32 `1 + w` at `[1024]`
- The final norm, fp32 `1 + w` in its own allocation.
- The `a‖b` padding: rows 96 to 127 of the tiled `[128][5120]` buffer read back as zero, all 163,840 of them, so the
  32 pad rows contribute nothing to the GEMV.
- The RoPE table: `p = 0` is exactly `(1, 0)` and `p = 1`'s first frequency is `(cos 1, sin 1)`. `inv_freq[0] = θ⁰ =
  1`, so this catches a wrong sign on the `θ` exponent or a transposed `[2][32]`
- `embed_tokens` is compared bit for bit with the checkpoint.

The test also asserts `report.unconsumed == 0`, which is what makes the claim that nothing is skipped by accident
testable.

## The packed-head checkpoint, measured against the files

A self-quantised checkpoint with an int4 `lm_head` uses a second config vocabulary and a second `lm_head` form, and
the engine takes it and the published bf16-head checkpoint through one code path. The differences below were read off
the real files before the loader code was written, and two of them change what the loader has to check.

| | published, bf16 head | self-quantised, int4 head |
|---|---|---|
| `quant_method` | `"gptq"` (with `provider: "auto-round"`) | `"auto-round"` |
| `packing_format` | absent | `"auto_round:auto_gptq"` |
| `desc_act` | `false`, declared | the key does not exist |
| exclusions | `dynamic`: 98 regexes, all `-:` | `extra_config`: 98 per-module objects, 97 at `bits: 16` plus `lm_head` at `bits: 4, group_size: 64, sym: true` |
| `.g_idx` tensors | 400 | none shipped |
| `lm_head` | `.weight`, BF16 `[248320][5120]` | `.qweight` I32 `[640][248320]` + `.qzeros` + `.scales` F16 `[80][248320]` |
| `mtp.*` | 15 tensors, all bf16 | 29 tensors, 8 of them int4 |
| language-model tensors | 2050 | 1650 (no `g_idx` family) |
| subnormal f16 scales | 1658 | 1628 |
| read per token, `W` | 15.540 GB | 13.673 GB |
| bandwidth ceiling at the measured 590 GB/s | 37.97 t/s | 43.15 t/s |

Both are int4 g64 symmetric with GPTQ v1 `qzeros` (`0x77777777` over all 401 of the second one's), the same
5120-wide model and the same tokenizer.

Neither the int4 head nor the ceiling it implies is quoted as a result anywhere in this project. vLLM cannot load a
quantised head at all, so a comparison against it would measure vLLM's missing feature rather than a difference in
kernels. The standing numbers compare like with like, with a bf16 `lm_head` on both sides
([05-perf-model.md](05-perf-model.md)). What the packed head gives the loader is one more format its classification
has to get right.

### The two config vocabularies, and the key that is missing

`QuantConfig::parse` reads both vocabularies. `bits`, `group_size` and `sym` are spelled the same in each and remain
hard requirements. Three things needed widening.

`quant_method` and `packing_format` are checked, not trusted. They decide nothing about a tensor, which is
`classify`'s job, but an unrecognised value means an unrecognised packing, and a wrong nibble order is a silently
wrong model. Both observed spellings are named, and anything else throws with the value found.

`extra_config` replaces `dynamic`, and unlike `dynamic` it can be read exactly. `dynamic` is a regex list the loader
deliberately does not interpret; `extra_config` is a per-module object, so `bits: 16` is counted as an exclusion and
`bits: 4` is checked for `group_size == 64` and `sym`. A module packed at g128 would otherwise dequantise wrong with
nothing to catch it. The `+:` hazard has no counterpart here: a claim that a module is packed is checked against the
shipped tensors at every load site, by `classify`

`desc_act` is absent, and the loader proves what the absence means instead of assuming it. The auto-round writer never
reorders columns, so it never writes the key. Inferring `false` from silence is the kind of assumption the loader
exists to refuse, so the inference is checked against the bytes: an undeclared `desc_act` together with any `g_idx`
tensor in the checkpoint throws, naming the count. A reordering needs a `g_idx` to carry it, there are none, so the
missing key cannot have meant anything else. The report prints `desc_act=false (inferred, 0 g_idx)` so the difference
reaches the log.

This changes the `g_idx` data check: with no `g_idx` tensors, the identity check has nothing to scan. The invariant is
not weakened, it is replaced by a stronger statement (the vector does not exist), but the `g_idx` count in the report
now carries weight, which is why `QuantScan` records it.

### `lm_head`: the one linear whose kind the checkpoint chooses

Everything else in the model is fixed by the architecture plus a quantiser exclusion list that is the same in both
checkpoints. `lm_head` is not. `model::Qwen35::lm_head(kind)` carries both rows, and `LinearSrc::classify(set,
"lm_head")` picks between them by content, like every other linear, so a config that lied about the head would be
caught by the tensors. That one classification drives the staging sizes, the byte buckets, the `W` cross-check, and the
shape and layout the captured command list binds.

The int4 row is `{K 5120, N 248320, S 1, layout 1}`. `src/model/qwen35.h` and [12-kernels.md](12-kernels.md) argue for
`S = 1`; in short, 3880 work-groups already fill the device, and `S = 1` lets the command list write the result
straight into `logits`

`lm_head` keeps its own byte bucket in both forms: 2,542,796,800 B in bf16 and 675,430,400 B in int4 (`K·N/2` nibbles
plus `K·N/32` scales). It has to, because it is the one row whose format differs; folding it into `int4` and `scales`
would make the two reports incomparable exactly where they differ.

### The `W` cross-check with a packed head

The 03-models `W = 15.519 GB` was measured over a checkpoint with a bf16 head. A packed head becomes one more itemised
term on the expected side, following the same rule as the padding and the fp32 widening; the 2% tolerance never
moves:

```
  W check     13.673 GB vs 13.673 GB expected = 15.519 doc-03 + 0.016 pad
                                              + 0.005279 widen -1.867 lm_head
```

`−1.867 GB` is `675,430,400 − 2,542,796,800`. The measured difference is −0.000% on both checkpoints.

### Staging is sized from the linears this load will repack

The staging buffers are not sized from a constant, because the largest linear changes with the head:

| | bf16 head | int4 head |
|---|---|---|
| largest int4 linear | `gate‖up`, 94.7 MB of words and scales | `lm_head`, 15520×80×136 u32 = 675.4 MB |
| largest bf16 tile buffer | `lm_head`, 2.54 GB | `a‖b` `[128][5120]`, 1.3 MB |

Sizing both buffers for the worst case would cost 2.5 GB of host memory for nothing. The `load_linear` bound checks
stay: a buffer that is too small produces a message naming the linear, never a heap overrun.

### The repack at 5120×248320: the overflow audit

An int4 `lm_head` is 7.1 times larger by element count than any int4 tensor the repack had seen before (1,271,398,400
against `gate‖up`'s 178,257,920). So every integer expression on the path was read for width: tile base, nibble and
scale source, column map, staging bound, byte buckets, bf16 head tiling and safetensors offsets, in
`src/common/repack.h` and `src/loader/loader.cc`. Nothing overflows; every product that can pass 2³² is computed in
`size_t`. The one expression that is not 64-bit, `g * 8 + j`, is bounded by `K/64 · 8 + 7 = 647` for any `K` this
model has, and the shape table is enforced by name before it runs. The audit did find one real defect: the int4
staging buffer had been sized from `gate‖up` alone, 7.1 times too small for a packed head. `load_linear` checks `words
> st.i4.size()` and names the linear, so it would have thrown rather than corrupted anything, and the sizing is now
the one described above.

### `mtp.*` and `model_extra_tensors.safetensors`

Both checkpoints put `mtp.*` in `model_extra_tensors.safetensors` and point the index at it, so that shard is opened
and its header parsed like any other. That is the index rule working, not a leak. The skip is by name, in
`build_view`, before the shard matters: 29 tensors counted as `mtp`, none of them read, none able to reach
`unconsumed`. The count is 0 unconsumed on both checkpoints.

`unconsumed` is only reported at load; nothing throws on it. `loader::load` counts it, names up to five and prints it,
and the assertion that it is 0 lives only in `tests/loader/load_checkpoint_test`. So a checkpoint that grew a tensor
family loads and serves, and the only evidence is a line in the log. That is deliberate (a load is not the place to
refuse a checkpoint over an unread tensor), but it matters which claims depend on it. The handling of both `lm_head`
forms does: the top-level branch of `build_view` puts every `lm_head.*` name in the view, and what proves each is
either consumed or deliberately dropped (the packed head's `.qzeros`) is the unconsumed count being 0 in the test. A
future head form with a fourth tensor would load silently and be caught only by `ctest -L checkpoint`

One consequence is not obvious. `assert_quant_invariants` walks the whole `SafetensorsSet`, so it also scans the 8
int4 `mtp` modules' `qzeros` and `scales`, tensors the engine does not load from this checkpoint. That is a deliberate
over-approximation (the invariants hold for them), and it is why the packed-head checkpoint's `qzeros` count is 401
(400 language-model plus `lm_head`) while its subnormal-scale count includes `mtp`'s.

### The report for both checkpoints

```
  quant     int4 g64 sym desc_act=false (declared), 98 dynamic exclusion rules
  tensors   2050 language-model + 1 top-level; skipped 333 visual, 15 mtp;
            dropped 400 qzeros + 400 g_idx (invariants asserted), 0 unconsumed
  lm_head        2542796800 B    2.543 GB   (bf16, by checkpoint content)
  read/token    15539980288 B   15.540 GB
  W check     15.540 GB vs 15.540 GB expected = … +0.000 lm_head  ->  -0.000%

  quant     int4 g64 sym desc_act=false (inferred, 0 g_idx), 97 extra_config exclusions + 1 explicit int4
  tensors   1650 language-model + 3 top-level; skipped 333 visual, 29 mtp;
            dropped 401 qzeros + 0 g_idx (invariants asserted), 0 unconsumed
  lm_head         675430400 B    0.675 GB   (int4 g64, by checkpoint content)
  read/token    13672613888 B   13.673 GB
  W check     13.673 GB vs 13.673 GB expected = … -1.867 lm_head  ->  -0.000%
```

## Picking the model

The model is picked from `config.json`'s `architectures[0]`, with no flag (`model::desc_for_architecture`).
`Qwen3_5ForConditionalGeneration` (Qwen3.8) and `Qwen3_5MoeForConditionalGeneration` (Ornith) get a `ModelDesc`
(`src/model/model_desc.h`: layer counts, widths, the MLP's intermediate size, `W`, the tokenizer's id count). An
unknown architecture throws, naming it and the supported ones. Agnes 3.0 Flash is no longer supported, and a
`config.json` naming `AgnesForConditionalGeneration` is refused by name. Kolibri-1 and Flash-Next have their own
loaders and descriptors, chosen by the entry point that serves them.

## g128 checkpoints

Common GPTQ exports quantise in groups of 128, symmetric. The loader accepts them (`quant.h`) and runs them on the g64
kernels unchanged. `LinearSrc::classify` reads the group size from the `.scales` row count (K/64 is g64, K/128 is
g128, anything else is refused) and, for g128, builds a `[K/64][N]` copy in which g64 rows 2j and 2j+1 both hold
g128 row j. Every weight keeps its own scale, so the expanded tensor dequantises to the g128 checkpoint's weights
exactly, and the repack, the GEMVs, the prefill dequant and the int8 prefill path all see an ordinary g64 linear.
`assert_quant_invariants` checks a `g_idx` against the identity k/group with the group taken from the same module's
`.scales`, so a `g_idx` and scales that disagree are refused. `tests/loader/quant_test.cc` pins the expansion against
a direct g128 dequantisation.

The cost is g64's bytes on the card (4.25 bits per weight instead of g128's 4.125, [02-formats.md](02-formats.md)) and
host memory for the expanded scales during the load. Asymmetric checkpoints (a zero point per group) and act-order
(`desc_act`) checkpoints stay refused.

A dedicated g128 kernel is not planned. Native g128 would read about 3% fewer weight bytes (about 2.9% faster decode
if tuned as well as g64; prefill would gain nothing), but it would need a g128 variant and a tuning sweep for every
int4 kernel, golden references per checkpoint and twice the kernel binaries. Re-quantising such a model at g64 is the
better answer.

## compressed-tensors symmetric checkpoints

llm-compressor writes its W4A16 releases (RedHatAI's `*-quantized.w4a16` and `*-INT4`) in compressed-tensors'
`pack-quantized` format. For the symmetric int4 group scheme, that format holds exactly what GPTQ's does, laid out the
other way round, so the loader reads it with a conversion and no kernel change.

The format, per linear: `weight_packed` int32 `[N][K/8]`, eight consecutive input columns per word, k = 8r in the low
nibble, each stored as q + 8. `weight_scale` `[N][K/g]`, f16 or bf16 (RedHatAI/Qwen3.8-27B-INT4 ships bf16).
`weight_shape` `[2]` = `[N, K]`, I64. Optionally `weight_g_idx` `[K]` (activation ordering) and `weight_zero_point`
(asymmetric). This was confirmed against a real checkpoint by `dequant_pack_quantized` in `tools/oracle/dflash_ref.py`.
In `config.json`: `quant_method: compressed-tensors`, `format: pack-quantized`, `config_groups.*.weights` with
`num_bits`, `type`, `symmetric`, `group_size`, `strategy` and `actorder`, and `ignore`, the modules left in bf16, which
ship `.weight` and are classified as bf16 anyway.

The conversion happens in one place and is exact. `LinearSrc::classify` finds `.weight_packed` (after looking for
`.qweight` and before `.weight`) and builds the GPTQ pair every loader path already consumes:

```
qweight[r][n] = weight_packed[n][r]              the int32 words verbatim, transposed
scales[g][n]  = f16(weight_scale[n][g])          bf16 -> f16 only if exact, else refused
then g128 -> g64 expansion (as for g128 checkpoints, above)
```

The nibble order and the +8 offset are the same in both formats, so the words are copied, never re-packed. A bf16
scale has 8 significant bits and an f16 has 11, so every bf16 value from 2⁻¹⁷ to 65504 is exactly an f16. Anything
outside that range is refused, naming the tensor and the index, because rounding a scale changes every weight in its
group. Both new buffers belong to the `LinearSrc` (`owned_qweight`, `expanded_scales`, as shared pointers), one linear
at a time. `LinearSrc::suffixes()` names the tensors read (`weight_packed`, `weight_scale`, `weight_shape`, and
`weight_g_idx` if shipped), and every loader marks those consumed. So dense linears and the MoE experts, in the main
layers and in the MTP head's MoE layer, all take compressed-tensors through the same function. The dense MTP head
stays the published 15 bf16 tensors (RedHatAI's checkpoint lists `mtp.*` under `ignore`); a quantised dense head is
refused by its tensor count, as before.

`tests/loader/ct_loader_test.cc` proves the conversion exact. It quantises known floats at g64 and g128 with f16 and
bf16 scales, writes them in both formats, and checks three things: every converted weight, read through
`common::Int4Gptq::at` (the kernels' host reference) and through the layout-1 tiles, equals the format's own
dequantisation (u − 8) × s, which equals q × s from the quantisation; the converted `qweight` and scales are
byte-equal to the GPTQ checkpoint of the same q and s; and one group's scale is subnormal in f16.
`ornith_repack_test` converts a compressed-tensors MoE layer (separate and fused experts, g128 on one) and compares
it with the GPTQ layer's device bytes.

What is refused, by field or by tensor:

- `symmetric: false` and any `.weight_zero_point` (asymmetric: there are no zero-point kernels and no exact
  conversion to symmetric int4);
- `num_bits` other than 4, `type` other than int, `strategy` other than group, and `group_size` other than 64 or 128,
  checked in the config and again from the scale shapes;
- a `format` other than pack-quantized, quantised activations, `dynamic` weights, a `quantization_status` other than
  compressed, and a non-empty `sparsity_config` or `transform_config` (rotations put the weights in another basis);
- a non-identity `weight_g_idx`, under any `actorder`;
- a compressed-tensors config over `.qweight` tensors, or a GPTQ config over `.weight_packed`, refused by
  `check_quant_scan`

`actorder` `"weight"`, `"static"` (an alias for weight) and null are accepted. They quantise in activation order but
assign groups in stored column order, which is the identity k / group the kernels compute. `"group"` and `"dynamic"`
(deprecated) are accepted only if every linear ships a `weight_g_idx` and each one is the identity.
`kv_cache_scheme` (fp8 KV calibration) is accepted: its `self_attn.{k,v}_scale` tensors are dropped by name and counted
in the report, because this engine's KV cache is bf16 or its own int8.

A compressed-tensors load prints one line to stderr and keeps it in `LoadReport::quant_note` (`ct_conversion_note` is
the source that can be tested on the host):

```
loader: converting compressed-tensors pack-quantized (g128, 400 linears): weight_packed [N][K/8] -> our qweight
[K/8][N], weight_scale [N][K/g] bf16 -> scales [K/g][N] f16 (exact) (g128 -> g64 scales expanded). The model
is supported, but expect reduced quality compared to our recommended format: AutoRound GPTQ W4A16 g64 symmetric.
```

The figures are for RedHatAI/Qwen3.8-27B-INT4 (48 × 3 GDN, 64 × 3 MLP and 16 × 4 attention linears). A GPTQ or
AutoRound checkpoint prints nothing. "Reduced quality" is relative to our own g64 AutoRound quantisation:
llm-compressor's g128 groups are twice as coarse, and its GPTQ calibration is not our recipe. The bytes on the card
are g64's either way, so `W` and the speed match a g64 checkpoint.

The configs checked so far are fixtures in `tests/loader/ct/`. Accepted: RedHatAI/Qwen3.8-27B-INT4 (symmetric int4
g128, `actorder` static, bf16 scales, an fp8 `kv_cache_scheme`). Refused: the checkpoints with `symmetric = false`
(Kolibri's W4A16-GPTQ and RedHatAI's Qwen3-8B) and one with an int8 channel group.

## Qwen3.8-Flash-Next: two checkpoint forms, the g128 expansion, the PLE file

`loader::load_qwen4exp` (`src/loader/qwen4exp_*`) is a separate loader. It reuses `resolve_snapshot`,
`QuantConfig::parse`, `SafetensorsSet`, `assert_quant_invariants` and `check_quant_scan`, `LinearSrc::classify`, the
int8 head and `rtn_int4_g64`

Checkpoint forms are read from the tensor names. `q4_forms` reads three groups, each all-or-nothing, as `.qweight`
(int4) or `.weight` (bf16): the dense projections (the delta-net `in_proj_qkv`, `in_proj_z` and `out_proj`, and the
attention `q/k/v/o_proj`), the shared experts, and the MTP head's experts. A mixed group is refused, naming the first
odd tensor. Our export is int4 everywhere AutoRound quantises. Intel's export keeps the dense layers, shared experts
and MTP head in bf16, with the routed experts at int4 g128. A bf16 or fused (`mlp.experts.gate_up_proj`) routed expert
in the main model is refused by name: the bf16 original is the reference's input, never the engine's.

The g128 expansion is `LinearSrc::classify`'s, expert by expert, so each expert's layout-1 block dequantises to `(q -
8) × scale` of its g128 source bit for bit (`qwen4exp_repack_test`). `quantization_config.group_size` must match the
experts' scales.

The layouts live in `qwen4exp_layout.h`, the one formula the memory planner and the loader share. Routed experts are
layout-1 blocks, one contiguous range per allocation, with expert `e` at `e × 1,740,800` (gate‖up, interleaved by
`cols_interleave16`) and `e × 870,400` (down). The router carries `shared_expert_gate` as row 512. The dense int4
linears are GPTQ layout 0. The delta-net small block comes from `make_small_layout`, and every `1 + w` norm is baked
to fp32.

The MTP head's experts ship in bf16 in both checkpoints and are quantised to int4 g64 at load by round to nearest
(`rtn_int4_g64`); the head only drafts, so this can change how many drafts are accepted, never the output.

The PLE table (a per-layer embedding indexed by n-grams of the input, 51.8 GB) is not in device memory. The
checkpoint's 128 bf16 shards for it are skipped by design. The engine reads `<snapshot>-ple-int8/` instead, made once
by `tools/quantize/qwen4exp/ple_int8.py`: per n-gram head, an int8 row table with per-row scales, plus three I64
constants. [17-running-models.md](17-running-models.md) has the command. A checkpoint without the file is refused with
the converter's command line, and the I64 tensors are held to the hash formula in `qwen4exp_ple_hash.h`.
`load_q4_ple` refuses a table larger than MemAvailable minus 16 GiB. It then pins the table as 16 + 16 ranges of host
USM (memory the card can read directly over PCIe), one per head and about 3.2 GB each, never as one 51.8 GB
allocation. Before copying the data it writes and reads back a tag on every 2 MiB page of every range, to catch two
ranges that alias the same memory. It then copies the rows, compares sampled rows, and uploads the device pointer
table the gather kernel indexes.

Every tensor is either read once or skipped by design and counted: `model.visual.*`, the PLE shards, `mtp.*` when the
MTP head is not loaded, and the later layers when `--layers N` loads only the first N.

The plan comes before the first allocation. `runtime::qwen4exp::plan` and `require_fits` check that weights and
persistent state fit. The whole model does not fit two cards' VRAM, so its routed experts are split between VRAM and
pinned host memory (the expert tier, [17-running-models.md](17-running-models.md)).

Serving adds no loader code. `b70-serve` loads through `load_qwen4exp` as `b70-decode` does, with `mtp = true` under
`--spec mtp`. The one file it checks besides the weights is `tokenizer.json`: it must be the
original model's, whose pre-tokenizer split includes `\p{M}` (combining marks). Intel's checkpoint ships Qwen3.8's
file, so `b70-serve` refuses it, and the original's file has to be linked beside the snapshot
([17-running-models.md](17-running-models.md)).

## Deliberately not loaded

- `model.visual.*` (333 tensors, 0.921 GB): the checkpoint is a vision-language model and this project serves text
  only ([03-models.md](03-models.md)).
- `mtp.*` (15 tensors, 0.849 GB on the published checkpoint; 29 on the self-quantised one): skipped unless the caller
  asks for the head (`load(..., mtp = true)`, which `--spec mtp` and `--spec auto` do). The skip is counted in the
  report, which keeps it a decision rather than an oversight. The self-quantised checkpoint's 29-tensor head is
  refused by name.
- `*.qzeros` and `*.g_idx` (800 tensors, 0.202 GB): scanned in full to prove the invariants above, then dropped. They
  are constants, and uploading them would spend 0.2 GB of every token's bandwidth re-reading values the kernel
  already knows.

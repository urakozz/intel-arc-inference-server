# The loader

One job: turn a HuggingFace checkpoint on disk into **the exact device buffers
the measured kernels were benchmarked against**, and fail loudly by tensor name
if the checkpoint does not mean what this project assumes. Everything the decode
loop needs is resident and canonical when `loader::load` returns; nothing is
converted, branched on, or discovered at run time.

`src/loader/{snapshot,safetensors,quant,loader}.{h,cc}` plus
`src/loader/small_layout.h` (the small-tensor block offsets, shared with the
model description and the device kernels), driven by the model description in
`src/model/qwen35.{h,cc}` (doc 03) and the canonical layouts in
`src/common/repack.h` (doc 12). Every figure below is measured against real
checkpoint files.

**Two checkpoint shapes are supported and the loader takes both without a
conditional anywhere in the walk**: a published checkpoint with a bf16
`lm_head`, and a self-quantised one with the head packed at int4
(`tools/quantize_qwen38_rtn.sh` produces the second). Everything in this
document that differs between them says which one it is; the section **"The
packed-head checkpoint"** below is the delta list, and it was measured against
the files rather than assumed.

**A third head form is made, not shipped (spec 9).** `load(..., LmHeadForm::Int8)`
reads the bf16 `lm_head`, quantises it on the host (`src/loader/lm_head_int8.h`:
`s = max|row| / 127` in fp32, `q = rne(w / s)` clamped to [-127, 127], 44 threads,
0.35-0.75 s) into `gemv_i8w`'s `[N/16][K/16][16 k][16 n]` int8 tiles, and puts the fp32
`[N]` scales in `DeviceWeight::scales`. The report's `lm_head` line names the form and
the quantisation time; the W cross-check itemises -1.270 GB. It refuses a checkpoint
whose head is already int4.

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

Doc 02's rule, that format plurality is a load-time problem, is why: the same
`quant_method: "gptq"` string covers checkpoints that ship different tensors,
and the `dynamic` rules that decide *which* modules were skipped are a regex
list that has already been observed to lie. This checkpoint carries **98
exclusion rules**, all `-:` - and the loader's real check on them is stronger
than parsing them: the model description says `in_proj_a`/`in_proj_b` are bf16
and everything else int4, and `load_linear` throws if a part classifies the
other way ("the checkpoint's dynamic exclusions moved"). A regex the loader
cannot read cannot mislead it.

`QuantConfig::parse` additionally throws unless `bits == 4`,
`group_size == 64`, `sym == true`, and `desc_act` is false **or absent** (see
"The packed-head checkpoint" below - absence is auto-round's spelling of false and is
proven from the shipped `g_idx` count, not taken on trust). It throws on any
`+:` dynamic rule - the exact pattern that made a published MTP head unusable
(BENCHMARKS.md); the published checkpoint's copy has been corrected to `-:`. It
also throws on an unrecognised `quant_method` or `packing_format`, and on an
`extra_config` module that claims to be packed at anything but g64 symmetric
int4.

### The three data asserts, and what measured them

`assert_quant_invariants(set)` runs **before a single byte is repacked** and
scans every word of every `.qzeros`, `.g_idx` and `.scales` in the checkpoint
(0.202 GB of the first two, 0.760 GB of scales; ~1 s in total):

- **Every `qzeros` word is `0x77777777`.** GPTQ v1 stores *zero point − 1*, so
  the packed nibble `7` means a zero point of 8 - symmetric. Measured over all
  400 `qzeros` tensors of this checkpoint: no other value occurs.
  This is what licenses the dequant `w = (q − 8)·scale` with no zero-point
  stream in the kernel, and (doc 02) what keeps the native s8×s4 DPAS prefill
  path open.
- **Every `g_idx[k] == k/64`.** `desc_act: false` means no activation-order
  permutation, so the group index must be the identity. Measured identical over
  all 400 `g_idx` tensors. This is what licenses the layout-1 repack
  to compute a group index arithmetically instead of carrying a per-`k` table.

- **Every `.scales` f16 is finite.** The dequant is `scale·(q − 8)` with no
  guard anywhere downstream, so one NaN or Inf scale poisons a group of 64
  weights silently. This assert was added last and it immediately found
  something: **1658 of the 380 108 800 scales in this checkpoint are
  subnormal f16** (the first is `layers.10.linear_attn.in_proj_qkv.scales[1444]
  = 0x00A8` ≈ 1.0e-5). Subnormals are *counted and reported, not rejected* -
  they are exact f16 values, the kernel reads the f16 word natively, and
  `common::f16_to_f32` was corrected the same day to decode them exactly
  (`man · 2⁻²⁴`) instead of flushing them to zero as its old "normal range
  only" comment allowed. Flushing would have made the host-side reference
  disagree with both the kernel and the oracle on 1658 groups.

**Binding constraint for every device kernel:** because real group scales are
subnormal f16, kernels must never be compiled with `-cl-denorms-are-zero`, or
any other FP16 denorm-flushing option. Flushing would zero 1 658 real scales and
silently diverge from both this loader's host reference and the CPU oracle. The
build makes that flag a fatal error rather than a convention.

All three throw with tensor name, element index and the offending value.
Neither `qzeros` nor `g_idx` is uploaded: **0.202 GB dropped at load**, and the
report prints the counts (and the subnormal count) so the drop is visible
rather than assumed.

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
- **`lm_head.*` is kept top level** (it is genuinely outside the language
  model's namespace, and `tie_word_embeddings: false`) - that is one tensor
  (`.weight`) on the published checkpoint and three (`.qweight`, `.qzeros`,
  `.scales`) on a checkpoint that packed the head. The drop counters cover this
  branch too, so a packed head's `qzeros` is reported as dropped rather than
  quietly missing from the tally, and the report prints the language-model and
  top-level counts separately instead of subtracting a hardcoded 1;
- `model.visual.*` (333 tensors, 0.921 GB) is **skipped** - this is a
  vision-language checkpoint served text-only;
- `mtp.*` is **skipped** unless the caller asks for it (`load(..., mtp = true)`, spec 8
  §3.1). 15 tensors (0.849 GB) on the published checkpoint, 29 on the self-quantised
  one, which also packs 8 of them; the skip is by name and does not care. Asked for,
  the published 15 are loaded into `LoadedModel::mtp` (bf16 in gemv_bf16's tiled
  layout: fc, q||k||v concatenated like an FA layer's Qkv, gate/up interleaved in
  16-column blocks like GateUp, o, down; the five RMSNorms and q/k norms baked fp32
  `1 + w`), the report gains an `mtp` line, and any other shape of head - the RTN
  checkpoint's 29 - is refused by name.

Both skip counts are printed. So is the count of tensors the loader never
consumed - **0**, and the test asserts it (`report.unconsumed`). Every name in
the view is either loaded or deliberately dropped, and the drop is counted.
Nothing is silently ignored; if a future checkpoint grows a tensor this loader
does not know about, the number stops being 0 and both the report and the test
say so - and the report names **up to five** of the unconsumed tensors (then
`…`), so a checkpoint that grew a whole family says which family instead of
sending the reader back with a debugger.

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

**`1 + w`, stored fp32, for every RMSNorm weight.** `Qwen3_5RMSNorm` is
Gemma-style: `x·rsqrt(mean(x²)+ε) ⊙ (1 + w)`, and the `+1` is **not** in the
checkpoint (doc 03, "Layer math"). Storing `1 + w` makes the kernel a plain
RMSNorm - one multiply, no constant to materialise per element.

The bake is **one fp32 add, stored as fp32, with no cast back to bf16.** The
original design stored bf16 and was changed for this reason: the reference HF computes the whole of `x·rsqrt(…)·(1 + w)` in fp32
and never rounds the multiplier. Storing `1 + w` as bf16 introduces a rounding
the oracle does not have, and it lands where bf16 is weakest - `w` is small, so
`1 + w` sits just above 1.0, where bf16's 8-bit mantissa steps by 2⁻⁸: up to
**0.39% error on the multiplier**, at ~130 sites (128 layernorms + the final
norm), compounding down 64 layers. Storing fp32 **removes** that rounding
entirely and closes the divergence; it costs 1 337 344 B of VRAM (below), which
is 0.0086% of the weights. Applied to `input_layernorm`,
`post_attention_layernorm`, FA's `q_norm`/`k_norm`, and the final `norm`.

**`linear_attn.norm.weight` is left alone - plain `w`, and still bf16.**
`Qwen3_5RMSNormGated` is plain `w`: it is the one norm in the model *without*
the `+1`, and it multiplies by `silu(z)` after normalising. Baking `+1` into it
would be a silent accuracy bug in all 48 GDN layers. It also keeps its bf16
storage, for the mirror image of the argument above: the reference's own
parameter dtype is bf16 and it multiplies in the bf16 domain, so widening it
would move *away* from the oracle rather than towards it. Both facts are called
out here and in `src/loader/small_layout.h`.

**`A_log → −exp(A_log)`, fp32.** The GDN decay is only ever used as
`g = −exp(A_log) ⊙ softplus(a + dt_bias)`. The `exp` is a per-head constant, so
it is hoisted out of 48 layers × every token. Stored fp32 because the recurrence
runs in fp32.

**`conv1d.weight` bf16 `[10240][1][4]` → fp32 `[10240][4]`.** The 4-tap
depthwise convolution accumulates in fp32; widening once at load costs 3.93 MB
of VRAM and saves a per-tap convert in the GDN kernel.

### The per-layer small-tensor blocks

Everything that is not a GEMV weight is packed into **two allocations per
layer** with compile-time offsets. Those offsets live in exactly one file,
**`src/loader/small_layout.h`**, `static_assert`ed on every offset, every block
size and every field alignment - the assert is what fails if the layout is
edited without the kernels being told. Three consumers read that header and
none of them re-derives a number: the loader packs against it, the model
description's small-tensor table (below) carries those constants as its
destination offsets, and **the GDN, FA and norm kernel bindings include it**
to find each field inside the block they are handed.

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

One block per layer instead of six allocations per layer keeps the descriptor
count down for the captured command list, and makes each layer's constants one
contiguous read.

The one norm that belongs to no layer - `model.language_model.norm.weight`, the
final RMSNorm before `lm_head` - gets its own 20 480 B allocation,
`LoadedModel::final_norm`, baked fp32 `(1 + w)` like all the rest.

**The table is the loader, and the model description owns it.**
`model::LayerDesc::small_tensors` is not a list of names any more: each entry
carries the layer-relative name, the exact element count, the required source
dtype, which of the two blocks it lands in, its byte offset there (a
`small_layout.h` constant) and its bake kind - `one_plus_w_fp32`, `plain_bf16`,
`neg_exp_fp32`, `raw_fp32_widen`, `raw_fp32`. `loader::load_small` **walks**
that table and hardcodes no name, no dtype and no offset, so adding a small
tensor is a row in `qwen35.cc` plus (if it moves the layout) a constant in
`small_layout.h`. Two checks keep the pair honest at load: an entry that would
overrun its block throws naming the tensor, and a block the entries do not tile
*exactly* throws naming the layer. `tests/model/qwen35_test` asserts the six GDN
and four FA rows, their offsets and their bakes without touching a device.

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
`(layout, S)` from the measured per-shape decision recorded in doc 12:

| Linear | K × N | layout | S |
|---|---|---|---|
| `QkvZ` | 5120 × 16384 | 1 | 1 |
| `Qkv` | 5120 × 14336 | 1 | 1 |
| `GateUp` | 5120 × 34816 | 1 | 4 |
| `Down` | 17408 × 5120 | 1 | 16 |
| `OutProj` / `OProj` | 6144 × 5120 | 1 | 16 |
| `AB`, `LmHead` (bf16) | - | n/a | - |
| `LmHead` (int4, if the checkpoint packed it) | 5120 × 248320 | 1 | 1 |

All rows are layout 1 today, and layout 1 **loses four of the five int4
shapes** (doc 12): it wins the sum only because `QkvZ` at `N = 16384` is the one
power-of-two stride where layout 0 collapses from about 550 to 419 GB/s. Both
repack paths were kept rather than deleting the loser, because flipping
individual rows to layout 0 is a measured +3.5% best-per-shape option - a tuning
knob to be measured end to end, not a decision to relitigate in the loader.

**Turning that knob is not a one-row edit, and the loader now says so.** Only
the *kernel* side of layout 0 exists today: `gemv.cl` compiles it, and the probe
measured it. The loader implements the layout-1 repack alone, and
`loader::DeviceWeight` holds a single `l0::Mem` - while `gemv.cl`'s `LAYOUT == 0`
binds `w` (`qweight[K/8][N]`, as shipped) and `scales[K/64][N]` as **two
separate buffers**. Flipping a row to layout 0 therefore needs (a) a loader path
that keeps the GPTQ-native arrays instead of repacking, and (b) a second device
allocation in `DeviceWeight`. Until both exist, `load_linear` **throws** on any
int4 row whose `layout` is not 1, naming the linear and both missing pieces - so
forgetting is loud at load rather than silent bytes the kernel cannot read.
(This doc previously claimed "change one row … with no other edit anywhere";
that was wrong, and the guard is what makes it stay corrected.) bf16 weights
(`AB`, `LmHead`) have exactly one tiled layout and carry `layout 0` as a filler,
which is why the guard is scoped to int4 rows.

## Memory and the `W` cross-check

Repack is single-threaded, staged through one reused host buffer per kind
(sized from the linears **this** load will repack - see "Staging is sized from
the linears this load will repack" below; with a bf16 head that is `gate‖up`'s
94.7 MB of int4 and `lm_head`'s 2.54 GB of bf16 tiles, and with a packed head
the two swap ends), and uploaded with **one synchronous copy per linear**. `embed_tokens` skips staging entirely - it is gathered one
row per token, so row-major *is* its canonical layout and the mmap is copied
verbatim.

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
| both layernorms bf16→fp32 | 1 310 720 | 5120 × 2 B × 2 norms × 64 layers |
| FA `q_norm`, `k_norm` bf16→fp32 | 16 384 | 256 × 2 B × 2 norms × 16 FA layers |
| final `norm` bf16→fp32 | 10 240 | 5120 × 2 B, once |

The last three rows are the fp32 RMSNorm store: **1 337 344 B** in total,
which is what the report's `widen` line itemises as *"RMSNorm fp32 (1+w)"*
against the GDN family's 3 941 376 B. Widening is a byte cost, never a
tolerance: both sides of the cross-check move by exactly the same amount.

The RoPE table (4 194 304 B) is *not* in this list: it is resident but not
streamed per token, so it is excluded from both sides rather than added to both.

Measured delta: **-27 072 B, -0.00017%** - the loader is 27 KB *under* doc 03's
figure, which is the rounding in "15.519". The fp32 RMSNorm store does not move
it, because the same 1 337 344 B are added to *both* sides. Note the three
totals the report prints and why they differ: **read/token 15.540 GB** is what
the decode step streams; **total 18.087 GB** adds `embed_tokens` (gathered one
row at a time, ~0 traffic) and the RoPE table (~256 B read per token). All three
are printed, labelled.

**Load time: 13.6 s** on a warm page cache, up about 0.4 s from before the
`.scales` finiteness scan was added, against a 120 s budget. No thread pool was
built: the
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

The tile check covers the int4 path only, so the small blocks are read back
field by field at their `small_layout.h` offsets - **every bake kind is
represented, and each check would fail if the block layout moved**:

- **the norms block**, both halves: `input_layernorm[0]` at `[0]` and
  `post_attention_layernorm[0]` at `[20480]` equal `1.0f + bf16_to_f32(src)` as
  **fp32** - which is simultaneously the proof of the bake, of the fp32 store
  (a bf16 store would not compare equal) and of the post half's new offset;
- **layer 0's GDN block**, one field per bake: `conv[0]` at `[0]` equals
  `bf16_to_f32(src conv1d[0])`; `negA[0]` at `[163840]` equals
  `−exp(bf16_to_f32(src A_log[0]))`; `dt_bias[0]` at `[164032]` equals
  `bf16_to_f32(src)`; and the gated norm at `[164224]` is the checkpoint's
  **raw bf16 word** - not `1+w`, not widened. That last one is the check that
  would catch someone "fixing" the inconsistency;
- **an FA layer's `k_norm`**: layer 3's, fp32 `1 + w` at `[1024]`;
- **the final norm**, fp32 `1 + w` in its own allocation;
- **the `a‖b` padding**: rows 96..127 of the tiled `[128][5120]` buffer read
  back as zero, all 163 840 of them, so the 32 pad rows contribute nothing to
  the GEMV;
- **the RoPE table**: `p = 0` is exactly `(1, 0)` and `p = 1`'s first frequency
  is `(cos 1, sin 1)` - `inv_freq[0] = θ⁰ = 1`, so this catches a wrong `θ`
  exponent sign or a transposed `[2][32]`.

And `report.unconsumed == 0` is asserted, which is what makes the "nothing is
skipped by accident" claim testable instead of rhetorical.

## The packed-head checkpoint - measured against the files

A self-quantised checkpoint with an int4 `lm_head` exercises a second config
vocabulary and a second `lm_head` classification, and the engine takes it and a
published bf16-head checkpoint through **one code path**. The deltas below were
probed off the real files before a line of loader code was written, and two of
them change what the loader has to check.

|                                   | published, bf16 head                     | self-quantised, int4 head                                                                                            |
|-----------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `quant_method`                    | `"gptq"` (with `provider: "auto-round"`) | **`"auto-round"`**                                                                                                   |
| `packing_format`                  | absent                                   | **`"auto_round:auto_gptq"`**                                                                                         |
| `desc_act`                        | `false`, declared                        | **the key does not exist**                                                                                           |
| exclusions                        | `dynamic`: 98 regexes, all `-:`          | **`extra_config`: 98 per-module objects** - 97 at `bits: 16`, plus `lm_head` at `bits: 4, group_size: 64, sym: true` |
| `.g_idx` tensors                  | 400                                      | **0 - none shipped at all**                                                                                          |
| `lm_head`                         | `.weight`, BF16 `[248320][5120]`         | **`.qweight` I32 `[640][248320]` + `.qzeros` + `.scales` F16 `[80][248320]`**                                        |
| `mtp.*`                           | 15 tensors, all bf16                     | 29 tensors, 8 of them int4                                                                                           |
| language-model tensors            | 2050                                     | 1650 (no `g_idx` family)                                                                                             |
| subnormal f16 scales              | 1658                                     | 1628                                                                                                                 |
| **read/token `W`**                | **15.540 GB**                            | **13.673 GB**                                                                                                        |
| roofline at the measured 590 GB/s | 37.97 t/s                                | **43.15 t/s**                                                                                                        |

Both are `int4 g64 sym`, GPTQ v1 `qzeros` (`0x77777777` over all 401 of the
second one's), the same 5120-wide model and the same tokenizer.

**Neither the int4 head nor the roofline it implies is quoted as a result
anywhere in this project.** vLLM cannot load a quantised head at all, so a
comparison against it would price an inability of the comparison rather than a
difference in kernels. The standing numbers are byte-matched with bf16
`lm_head` on both sides (doc 05). What the packed head buys the *loader* is the
point of this section: one more format its classification has to be right
about.

### The two config vocabularies, and the one that says nothing

`QuantConfig::parse` reads both. `bits`, `group_size` and `sym` are spelled the
same in each and are still hard requirements. The rest needed widening:

- **`quant_method` and `packing_format` are checked, not trusted.** They decide
  nothing about a tensor - that is `classify` on the shipped suffixes - but an
  unrecognised value means an unrecognised *packing*, and a wrong nibble order
  is a silently wrong model. Both observed spellings are named and anything
  else throws with the value it found.
- **`extra_config` replaces `dynamic`, and it can be read exactly.** `dynamic`
  is a regex list the loader deliberately does not interpret; `extra_config` is
  a per-module object, so `bits: 16` is counted as an exclusion and `bits: 4`
  is checked for `group_size == 64` and `sym` - a module packed at g128 would
  dequantise wrong with nothing else to catch it. The `+:` hazard has no
  analogue here: a claim that a module is packed is checked against the shipped
  tensors at every load site, by `classify`.
- **`desc_act` is absent, and absence is proven rather than assumed.** The
  auto-round writer never permutes, so it never emits the key. Inferring
  `false` from silence is exactly the kind of assumption this loader exists to
  refuse - so the inference is checked against the bytes: an undeclared
  `desc_act` with **any** `g_idx` tensor in the checkpoint throws, naming the
  count. A permutation needs a `g_idx` to carry it; there are none, so there is
  nothing the missing key could have meant. The report prints
  `desc_act=false (inferred, 0 g_idx)` so the distinction survives into the log.

Note what this costs the **second data assert**: with no `g_idx` tensors, the
identity check has nothing to scan. The invariant is not weakened - it is
replaced by a stronger statement (the vector does not exist) - but the
`g_idx` count in the report is now load-bearing rather than decorative, which
is why `QuantScan` carries it.

### `lm_head`: the one linear whose kind is the checkpoint's to choose

Everything else in this model is fixed by the architecture plus a quantiser
exclusion list that is the same in both checkpoints. `lm_head` is not.
`model::Qwen35::lm_head(kind)` carries both rows and
`LinearSrc::classify(set, "lm_head")` picks - by content, like every other
linear, so a config that lied about it would be caught by the tensors. One
classification then drives the staging sizes, the byte buckets, the `W`
cross-check, and the shape and layout the capture binds.

The int4 row is `{K 5120, N 248320, S 1, layout 1}`. `S = 1` is argued in
`src/model/qwen35.h` and in doc 12; the short version is that 3880 work-groups
already saturate the device and that `S = 1` is what lets the capture write the
result straight into `logits`.

**`lm_head` keeps its own byte bucket in both kinds** - 2 542 796 800 B bf16,
675 430 400 B int4 (`K·N/2` nibbles + `K·N/32` scales). It has to: it is the one
row whose format differs, so folding it into `int4`/`scales` would make the two
reports incomparable exactly where they differ.

### The `W` cross-check across the boundary

doc 03's `W = 15.519 GB` was measured over a checkpoint with a bf16 head. A
packed head is an **itemised term on the expected side**, the same rule the
padding and the fp32 widening follow - the 2% tolerance never moves:

```
  W check     13.673 GB vs 13.673 GB expected = 15.519 doc-03 + 0.016 pad
                                              + 0.005279 widen -1.867 lm_head
```

`−1.867 GB` is `675 430 400 − 2 542 796 800`. Measured delta **−0.000%** on both
checkpoints.

### Staging is sized from the linears this load will repack

Not from a constant, because the maximum swaps ends:

| | bf16 head | int4 head |
|---|---|---|
| largest int4 linear | `gate‖up`, 2176×80×136 u32 = **94.7 MB** | **`lm_head`, 15520×80×136 u32 = 675.4 MB** |
| largest bf16 tile buffer | **`lm_head`, 2.54 GB** | `a‖b` `[128][5120]` = **1.3 MB** |

Sizing both for the maximum would cost 2.5 GB of host RSS for nothing. The
`load_linear` bound checks stay where they are: a mis-sized buffer is a message
naming the linear, never a heap overrun.

### The repack at 5120×248320 - the overflow audit

`lm_head` at int4 is **7.1× the largest int4 tensor this repack had ever seen**
by element count (1 271 398 400 against `gate‖up`'s 178 257 920), so every
integer expression on the path was read for width before the first run. The
result: **nothing overflows, and nothing needed changing.**

| site (`src/common/repack.h`, `src/loader/loader.cc`) | expression | width | max value here |
|---|---|---|---|
| tile base | `out + (size_t(nt) * G + g) * 136` | `size_t` - `nt` is promoted first | 168 857 464 |
| nibble source | `c.qweight[size_t(g * 8 + j) * c.n_part + c.n]` | `g*8+j` is `uint32_t` but ≤ 639; the product is `size_t` | 158 876 799 |
| scale source | `c0.scales[size_t(g) * c0.n_part + c0.n]` | `size_t` | 19 859 519 |
| column map index | `cols[nt * 16 + l]` | `uint32_t` | 248 319 |
| staging bound | `size_t(sh.N / 16) * (sh.K / 64) * 136` | `size_t` | 168 857 600 |
| byte buckets | `size_t(sh.K) * sh.N / 2`, `/32` | `size_t` | 635 699 200 |
| bf16 head tiling | `out[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16]` | `size_t` | 1 271 398 399 |
| safetensors offsets | `TensorInfo::begin/end` | `uint64_t`; `bytes()` returns `size_t` | 3.2 GB shards |

The one expression that is *not* 64-bit - `g * 8 + j` - is bounded by
`K/64 · 8 + 7 = 647` for any `K` this model has, and the shape table is
enforced by name before it runs. It is called out here rather than widened so
that the next person does not have to re-derive the bound.

**The size change did surface a real defect, and it was under-sizing, not
overflow**: the int4 staging buffer was sized from `gate‖up` alone and is 7.1×
too small for a packed head. It would have thrown - `load_linear` checks
`words > st.i4.size()` and names the linear - rather than corrupting anything,
which is the behaviour that check exists for. It is fixed above.

### `mtp.*` and `model_extra_tensors.safetensors` are ignored cleanly

Both checkpoints put `mtp.*` in `model_extra_tensors.safetensors` and both point
the index at it, so the shard is opened and its header parsed like any other -
that is the dedup rule working, not a leak. The skip is by **name**, in
`build_view`, before the shard is relevant: 29 tensors counted as `mtp`, none of
them ever read, none of them able to reach `unconsumed`. Measured
**`0 unconsumed`** on both checkpoints.

**`unconsumed` is report-only at load - nothing throws on it.** `loader::load`
counts it, names up to five, and prints it; the assertion that it is 0 lives in
`tests/loader/load_checkpoint_test` and nowhere else. So on a box without the
test, a checkpoint that grew a tensor family loads and serves, and the only
evidence is a line in the log. That is a deliberate trade - a load is not the
place to refuse a checkpoint over an unread tensor - but it is worth knowing
*which* claim rides on it. The `lm_head`-in-both-forms handling does: the
top-level branch of `build_view` puts every `lm_head.*` name in the view, and
what proves that all of them are either consumed or deliberately dropped (the
packed head's `.qzeros`) is the unconsumed count being 0, checked by the test.
A future head form with a fourth tensor would load silently and be caught only
by `ctest -L checkpoint`.

One consequence worth stating because it is not obvious:
`assert_quant_invariants` walks the whole `SafetensorsSet`, so it *does* scan
the 8 int4 `mtp` modules' `qzeros` and `scales` - tensors the engine never
loads. That is a deliberate over-approximation (the invariants hold for them,
measured), and it is why the packed-head checkpoint's `qzeros` count is **401**
(400 language-model + `lm_head`) while its subnormal-scale count includes
`mtp`'s.

### The report, both checkpoints

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

Load times from that session are **not** comparable to the 13.6 s recorded
above: the machine was under a 12-core compile throughout, and a cold page cache
read 41.4 s against a warm 12.6 s. Nothing about load time was being measured
there.

## Deliberately not loaded

- **`model.visual.*`** (333 tensors, 0.921 GB) - this checkpoint is a
  vision-language model and this project serves text only (doc 03).
- **`mtp.*`** (15 tensors, 0.849 GB on the published checkpoint; 29 on the
  self-quantised one) - skipped unless the caller asks for the head
  (`load(..., mtp = true)`; `b70-serve --mtp K`, spec 8). Without it the skip
  is counted in the report, which is how that stays a decision rather than an
  oversight. The self-quantised checkpoint's 29-tensor head is refused by name.
- **`*.qzeros`, `*.g_idx`** (800 tensors, 0.202 GB) - fully scanned to prove the
  invariants above, then dropped. They are constants; uploading them would cost
  0.2 GB of every token's bandwidth to re-read a value the kernel already knows.

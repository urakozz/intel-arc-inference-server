# The target model

The engine supports one model family, `model_type: qwen3_5`, on purpose. A
general engine cannot hardcode the things this one hardcodes; specialisation is
the whole strategy.

## The checkpoint

`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` - GPTQ g64 symmetric, int4
weights, bf16 `lm_head`. Both this engine and vLLM serve the same files, which
is what makes the comparison in [BENCHMARKS.md](BENCHMARKS.md) fair.

Everything below is read from the snapshot's `config.json` and the safetensors
headers. **Measured, not assumed.**

```
model_type              qwen3_5         architectures  Qwen3_5ForConditionalGeneration
num_hidden_layers       64
hidden_size             5120
intermediate_size       17408           <- dense MLP, NO MoE
vocab_size              248320          <- lm_head rows; tokenizer has 248077 (doc 11)
head_dim                256
num_attention_heads     24
num_key_value_heads     4               <- GQA 6:1
full_attention_interval 4
attn_output_gate        true            <- q_proj emits q AND a gate: N = 2 x 24 x 256 = 12288
tie_word_embeddings     false
mtp_num_hidden_layers   1
max_position_embeddings 262144
rms_norm_eps            1e-6
dtype                   float16         <- but every unquantised tensor is stored BF16

linear_num_key_heads    16
linear_num_value_heads  48
linear_key_head_dim     128
linear_value_head_dim   128
linear_conv_kernel_dim  4
```

Layer pattern is `[LA, LA, LA, FA] x 16` - full attention at layers 3, 7, 11,
…, 63:

**48 linear-attention (GDN) layers, 16 full-attention layers.**

Quantisation: GPTQ packing produced by AutoRound, int4, **symmetric, group_size
64**, `desc_act: false`. `lm_head` is **not** quantised. 98 `dynamic` exclusion
rules keep `mtp.*` and every GDN layer's `in_proj_a`/`in_proj_b` in bf16
(doc 02).

### Per-layer tensors, from the headers

GDN layer (e.g. layer 0), K to N:

| Tensor | Shape / role | Bytes |
|---|---|---|
| `linear_attn.in_proj_qkv` | 5120 → 10240 = q 16x128 ‖ k 16x128 ‖ v 48x128, int4 | 26.2 MB + 1.6 MB scales |
| `linear_attn.in_proj_z` | 5120 → 6144 (output gate), int4 | 15.7 + 1.0 |
| `linear_attn.in_proj_a`, `in_proj_b` | 5120 → 48 each, **bf16** (decay / beta per value head) | 0.5 + 0.5 |
| `linear_attn.conv1d.weight` | `[10240, 1, 4]` bf16 - depthwise, 4 taps, over qkv | 0.08 |
| `linear_attn.A_log`, `dt_bias` | `[48]` each | - |
| `linear_attn.norm.weight` | `[128]` - gated RMSNorm per value head | - |
| `linear_attn.out_proj` | 6144 → 5120, int4 | 15.7 + 1.0 |
| `mlp.gate_proj`, `up_proj` | 5120 → 17408 each, int4 | 44.6 + 2.8 each |
| `mlp.down_proj` | 17408 → 5120, int4 | 44.6 + 2.8 |
| `input_layernorm`, `post_attention_layernorm` | `[5120]` | - |

About **204 MB per GDN layer**, times 48 = 9.8 GB.

Full-attention layer (e.g. layer 3):

| Tensor | Shape / role | Bytes |
|---|---|---|
| `self_attn.q_proj` | 5120 → 12288 = 24 heads x 256 x (q ‖ gate), int4 | 31.5 + 2.0 |
| `self_attn.k_proj`, `v_proj` | 5120 → 1024 = 4 x 256 each, int4 | 2.6 + 0.2 each |
| `self_attn.q_norm`, `k_norm` | `[256]` - per-head RMSNorm before RoPE | - |
| `self_attn.o_proj` | 6144 → 5120, int4 | 15.7 + 1.0 |
| `mlp.*` | as above | 133.7 + 8.4 |

About **198 MB per FA layer**, times 16 = 3.2 GB.

Top level: `embed_tokens` BF16 `[248320, 5120]` 2.54 GB (gathered, about zero
traffic); `lm_head.weight` BF16 `[248320, 5120]` **2.54 GB, read in full every
token**; `norm.weight`.

The MTP head, all **bf16**, 0.85 GB: `mtp.fc` `[5120, 10240]` (fusing embedding
‖ hidden), `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, one full transformer
layer at the main model's shapes, `mtp.norm`. It shares `embed_tokens` and
`lm_head` with the main model, so a draft step would read 0.85 + 2.54 GB and
`lm_head` width would matter twice. The engine loads it behind `--mtp K`
(spec 8): `b70-serve --mtp K` drafts K tokens with it and verifies them in one
step of K + 1 rows (docs/BENCHMARKS.md, "MTP speculative decoding"). Without
the flag it is skipped, as before.

### Layer math, verified against the modeling file

From `transformers` 5.15, `models/qwen3_5/modeling_qwen3_5.py`. This is the
contract the kernels and the CPU oracle (doc 14) must agree on. `x` is one
token's hidden vector, `⊙` is elementwise, all norms run in fp32.

**RMSNorm is Gemma-style `(1 + w)`.** `Qwen3_5RMSNorm` (used for
`input_layernorm`, `post_attention_layernorm`, the final `norm`, `q_norm`,
`k_norm`) computes `x · rsqrt(mean(x²) + 1e-6) ⊙ (1 + w)`. **The `+1` is not in
the weights**; the loader bakes it in and stores `1 + w` **as fp32**, because
the reference does the whole product in fp32 and rounding the multiplier back to
bf16 would add an error it does not have (doc 13). `Qwen3_5RMSNormGated` (the
GDN output norm) is plain `w` - no `+1` - and multiplies by `silu(z)` *after*
normalising.

**Decoder layer.** `x += mixer(norm₁(x)); x += mlp(norm₂(x))`, where
`mlp(h) = W_down · (silu(W_gate h) ⊙ W_up h)`.

**GDN layer** (`Qwen3_5GatedDeltaNet.forward`, decode path):

```
qkv  = W_qkv h                       # 10240 = q 16x128 | k 16x128 | v 48x128
z    = W_z   h                       # 6144  = 48x128
b, a = W_b h, W_a h                  # 48 each, bf16 weights
qkv  = silu(conv1d_4tap(qkv))        # depthwise over the last 4 tokens; state = previous 3; no bias
β    = sigmoid(b)                    # [48]
g    = -exp(A_log) ⊙ softplus(a + dt_bias)      # [48], fp32
q, k = repeat_interleave(q, 3), repeat_interleave(k, 3)   # v-head h uses k-head h // 3
q, k = l2norm(q), l2norm(k)          # per head, eps 1e-6;  q *= 1/sqrt(128)
per v-head h, state S_h in fp32[128 k x 128 v]:
  S_h  = S_h · exp(g_h)
  kv   = kᵀ S_h                      # [128 v]
  Δ    = (v - kv) ⊙ β_h
  S_h += k ⊗ Δ                       # rank-1 update
  o_h  = qᵀ S_h                      # [128 v]
o    = RMSNormGated(o, z)            # per head: w ⊙ (o · rsqrt(mean(o²)+eps)) ⊙ silu(z)
out  = W_out · o                     # 6144 → 5120
```

The whole recurrence is 48 independent 128x128 fp32 state updates per layer,
about 3 MB read and written. `torch_recurrent_gated_delta_rule` is the 51-line
reference; the chunked form is what prefill uses.

**Full-attention layer** (`Qwen3_5Attention.forward`):

```
qg = W_q h   viewed as 24 heads x [q 256 | gate 256]   # per-head interleaved, NOT two halves
k  = W_k h   →  4 x 256 ;  v = W_v h  →  4 x 256
q, k = RMSNorm_{1+w}(q), RMSNorm_{1+w}(k)                # over 256, per head
RoPE on the first 64 dims only (partial_rotary_factor 0.25), θ = 1e7,
  inv_freq_i = θ^(-2i/64), i < 32; rotate_half over the 64-slice (non-interleaved halves);
  dims 64..255 pass through
GQA 6:1: q-head j reads kv-head j // 6 ;  scores = q·k / sqrt(256) ; softmax in fp32
attn = softmax(scores) · v  →  24 x 256 = 6144
attn = attn ⊙ sigmoid(gate)                            # gate is 6144, head-major
out  = W_o · attn                                      # 6144 → 5120
```

`rope_parameters` carries interleaved mRoPE (`mrope_section [11, 11, 10]`). For
text the model expands one position id to every stream, so
`apply_interleaved_mrope` copies each frequency onto itself: **plain RoPE**. The
loader precomputes `cos/sin[max_model_len, 64]` once.

**Head.** `logits = W_lm · RMSNorm_{1+w}(x)`; `lm_head` is a separate tensor
(`tie_word_embeddings: false`), and only the last token's logits are needed.

**Per-token state the decode list owns:** per GDN layer a conv window
`[10240 x 3]` bf16 and `S[48 x 128 x 128]` fp32; per FA layer a KV ring
`[max_model_len x 4 x 256]` x 2 bf16; plus `position`.

### Byte accounting - `W` is measured

Index manifest, 2399 tensors across 6 files, **no duplicated names**, total
20.03 GB on disk:

| Group | Tensors | GB |
|---|---|---|
| `model.language_model.*.qweight` (int4) | 400 | 12.163 |
| `model.language_model.*.scales` (f16, g64) | 400 | 0.760 |
| `model.language_model.*.qzeros` | 400 | 0.190 - dropped at load |
| `model.language_model.*.g_idx` | 400 | 0.012 - dropped at load |
| norms, conv1d, A_log, dt_bias, in_proj_a/b (bf16) | 449 | 0.052 |
| `lm_head.weight` (bf16) | 1 | 2.543 |
| `embed_tokens` (bf16, gathered) | 1 | 2.543 |
| `mtp.*` (bf16) | 15 | 0.849 |
| `model.visual.*` - skipped | 333 | 0.921 |

**`W` = 12.163 + 0.760 + 0.052 + 2.543 = 15.519 GB read per decode token.** The
loader's own report of what it makes resident is **15.540 GB**, the difference
being tiling pad and fp32-widened norms, itemised in doc 13. Doc 05 uses the
loader's figure throughout.

Reproduce: `python3 tools/probe/checkpoint_bytes.py <snapshot>`. It reads the
index and the safetensors headers only, no tensor data, and prints this table,
`W`, the `lm_head` int4/int8 variants and the MTP head.

### Three facts that drive the design

1. **No MoE.** The hardest kernel family in the larger models of this family is
   simply absent. No grouped GEMM.
2. **vLLM is already at 81.7% of the roofline on this model.** 31.01 t/s times
   15.540 GB is 482 GB/s of a measured 590. Decode is not won by one big lever;
   it is won by several small ones that multiply (doc 05).
3. **GDN dominates the layer count, not the byte count.** 48 of 64 layers, and
   `sycl-tla` has no example for them - but the recurrent state is
   `48 x 128 x 128 x 4 B` = 3 MB per layer, about 150 MB per token read and
   written across 48 layers, roughly 2% of `W`. Measured in situ, the GDN step
   kernel is 1.7% of a decode step and runs at 92% of device bandwidth. **GDN is
   the kernel that must be *correct*; the linears decide the *speed*.**

### Kernel count, and why it stopped being a worry

At one kernel per op the unfused decode step is about 700 kernels, which at
design time looked like the dominant fixed cost. Both halves of that framing
were measured and both were pessimistic. The captured list is **774** launches
and **19 modules** per token, and the per-launch cost inside a replayed list is
**0.73 us in situ** - 0.473 ms, about 1.1% of a decode step. Fusion for launch
count's sake is therefore not a lever, and the launch count moved *up*
deliberately when splitting a single-work-group kernel in two bought 2.4 ms.
What those kernels do while they run is the problem; how many there are is not.

### KV and state sizes at the benchmark shape

Per token of context: 16 FA layers x 4 kv heads x 256 x 2 (K, V) x 2 B =
**64 KB**. At pp4096 + tg256 = 4352 tokens that is 285 MB; at `--max-model-len
16k` it is 1.07 GB. Attention at depth 4096 reads about 270 MB per token, around
1.7% of `W`. GDN conv state is 10240 x 3 x 2 B = 61 KB per layer. A ring KV
buffer sized to `max_model_len` is a rounding error next to the weights; there
is no memory reason for paged KV with one sequence.

### Tokenizer

`Qwen2Tokenizer`, byte-level BPE, **248 044 base + 33 added tokens = 248 077**;
`lm_head` is padded to 248 320, so the sampler must never emit ids >= 248 077.
The chat template is a separate **`chat_template.jinja`** (8.9 KB, not in
`tokenizer_config.json`) and uses macros, `namespace`, `raise_exception`, the
`is string` / `is iterable` / `is mapping` tests, thinking mode
(`enable_thinking`) and tools. `eos_token_id` is the list `[248046, 248044]` =
`<|im_end|>`, `<|endoftext|>`. Details and the pre-tokenizer regex in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).

### One lesson from a checkpoint that did not work

An earlier 9B target (32 layers, 24 GDN + 8 FA, hidden 4096, g128) never loaded
in vLLM and turned out to be broken beyond that. Its one lasting lesson is in
the loader: its `model_extra_tensors.safetensors` overlapped the numbered
shards, so **the loader deduplicates by tensor name and honours the index as the
manifest**. The 27B happens not to need that rule; the next checkpoint may.

## Benchmark command

Every vLLM number in these docs comes from this, so new ones must too:

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

Prefill is measured directly rather than through `llama-benchy`'s derived `pp`
figure; the reason, and every measured row, is in [BENCHMARKS.md](BENCHMARKS.md).

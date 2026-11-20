# The target model: Qwen3.8-27B

This document describes Qwen3.8-27B (`model_type: qwen3_5`), the engine's first and main model, in the detail the
kernels need: its shapes, the exact math of each layer, and where its bytes are. It is for a reader who wants to see
what one decode step has to compute and read, and why the engine is built the way it is.

A general engine cannot hardcode the things this one hardcodes. Specialising to the model is the whole strategy.

## The other models

Qwen3.8-27B, used for agentic coding, comes first; Qwen3.8-Flash-Next is next in line. Ornith 1.5, Kolibri-1 and
Qwen3.8-Flash-Next (`qwen4_exp`) each have their own engine, specialised the same way. Muse-Glimmer and Agnes 3.0
Flash were dropped and are no longer supported.

| model | in short |
|---|---|
| Ornith 1.5 35B-A3B | 40 layers: 30 GDN (gated delta-net, a linear-attention layer) and 10 full attention; a 256-expert MoE (mixture of experts: a router picks a few expert networks per token) with top-8 routing plus a shared expert; about 3 B parameters active per token |
| Qwen3.8-Flash-Next | 12 QSA sparse-attention layers and 36 GDN layers, a 512-expert top-10 MoE, 4-stream hyper-connections and a 51.8 GB n-gram embedding held in host memory. The whole model does not fit two 32 GB cards, so it is served on two cards with part of its routed experts read from host memory. Its chat template is Qwen3.8's byte for byte; its tokenizer is not (the original's split adds `\p{M}`) |
| Kolibri-1 | 50 MoE layers (384 experts, top 6, plus a shared expert), 40 of them with sliding-window attention; needs two cards at int4 |

What each model runs, and with which flags, is in [17-running-models.md](17-running-models.md).

## Qwen3.8-27B

### The checkpoint

`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`: GPTQ g64 symmetric, int4 weights, bf16 `lm_head`. vLLM serves
the same files, so the two can be compared on identical weights.

Everything below is read from the snapshot's `config.json` and the safetensors headers, not assumed.

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

The layer pattern is `[LA, LA, LA, FA] x 16`, with full attention at layers 3, 7, 11, …, 63. That makes 48
linear-attention layers and 16 full-attention layers. The linear-attention layers are GDN (gated delta net): in
place of a cache that grows with the context, each keeps a fixed-size state matrix that every token updates. The
full-attention layers are ordinary softmax attention with GQA (grouped-query attention: 24 query heads share 4
key/value heads, 6 to 1).

Quantisation: GPTQ packing produced by AutoRound, int4, symmetric, `group_size` 64, `desc_act: false`. `lm_head`,
the output projection from the hidden state to vocabulary scores, is not quantised in the checkpoint. 98
`dynamic` exclusion rules keep `mtp.*` and every GDN layer's `in_proj_a` and `in_proj_b` in bf16
([02-formats.md](02-formats.md)).

### Per-layer tensors, from the headers

A GDN layer (layer 0, for example), K to N:

| Tensor | Shape / role | Bytes |
|---|---|---|
| `linear_attn.in_proj_qkv` | 5120 → 10240 = q 16x128 ‖ k 16x128 ‖ v 48x128, int4 | 26.2 MB + 1.6 MB scales |
| `linear_attn.in_proj_z` | 5120 → 6144 (output gate), int4 | 15.7 + 1.0 |
| `linear_attn.in_proj_a`, `in_proj_b` | 5120 → 48 each, bf16 (decay and beta per value head) | 0.5 + 0.5 |
| `linear_attn.conv1d.weight` | `[10240, 1, 4]` bf16, depthwise, 4 taps, over qkv | 0.08 |
| `linear_attn.A_log`, `dt_bias` | `[48]` each | - |
| `linear_attn.norm.weight` | `[128]`, gated RMSNorm per value head | - |
| `linear_attn.out_proj` | 6144 → 5120, int4 | 15.7 + 1.0 |
| `mlp.gate_proj`, `up_proj` | 5120 → 17408 each, int4 | 44.6 + 2.8 each |
| `mlp.down_proj` | 17408 → 5120, int4 | 44.6 + 2.8 |
| `input_layernorm`, `post_attention_layernorm` | `[5120]` | - |

That is about 204 MB per GDN layer, times 48 = 9.8 GB.

A full-attention layer (layer 3, for example):

| Tensor | Shape / role | Bytes |
|---|---|---|
| `self_attn.q_proj` | 5120 → 12288 = 24 heads x 256 x (q ‖ gate), int4 | 31.5 + 2.0 |
| `self_attn.k_proj`, `v_proj` | 5120 → 1024 = 4 x 256 each, int4 | 2.6 + 0.2 each |
| `self_attn.q_norm`, `k_norm` | `[256]`, per-head RMSNorm before RoPE | - |
| `self_attn.o_proj` | 6144 → 5120, int4 | 15.7 + 1.0 |
| `mlp.*` | as above | 133.7 + 8.4 |

That is about 198 MB per full-attention layer, times 16 = 3.2 GB.

At the top level, `embed_tokens` is BF16 `[248320, 5120]`, 2.54 GB, but only one row is gathered per token, so it
costs about no traffic. `lm_head.weight` is BF16 `[248320, 5120]`, 2.54 GB, and is read in full on every token.
Then there is `norm.weight`

The MTP head (multi-token prediction: an extra layer trained to guess the next tokens) is all bf16, 0.85 GB:
`mtp.fc` `[5120, 10240]` (fusing embedding ‖ hidden), `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, one full
transformer layer at the main model's shapes, and `mtp.norm`. It shares `embed_tokens` and `lm_head` with the main
model, so a draft step reads 0.85 + 2.54 GB and the width of `lm_head` matters twice.

The engine loads the MTP head behind `--spec mtp`, which is also what `b70-serve`'s default `--spec auto` picks on
this model. It drafts up to K tokens and verifies them in one step of K + 1 rows (speculative decoding: the main
model checks several guessed tokens for about the cost of one, and keeps those it agrees with). K is chosen per iteration from each request's own acceptance rate, up to
`--spec-max`. With `--spec off` the head is skipped.

### Layer math, checked against the modeling file

This is from `transformers` 5.15, `models/qwen3_5/modeling_qwen3_5.py`. It is the contract the kernels and the CPU
reference ([14-golden-gate.md](14-golden-gate.md)) must both agree on. `x` is one token's hidden vector, `⊙` is
elementwise multiplication, and all norms run in fp32.

**RMSNorm is Gemma-style `(1 + w)`.** `Qwen3_5RMSNorm` (used for `input_layernorm`, `post_attention_layernorm`,
the final `norm`, `q_norm` and `k_norm`) computes `x · rsqrt(mean(x²) + 1e-6) ⊙ (1 + w)`. The `+1` is not in the
weights. The loader bakes it in and stores `1 + w` as fp32, because the reference does the whole product in fp32
and rounding the multiplier back to bf16 would add an error the reference does not have
([13-loader.md](13-loader.md)). `Qwen3_5RMSNormGated`, the GDN output norm, is plain `w` with no `+1`, and
multiplies by `silu(z)` after normalising.

**Decoder layer.** `x += mixer(norm₁(x)); x += mlp(norm₂(x))`, where `mlp(h) = W_down · (silu(W_gate h) ⊙ W_up h)`.
The mixer is GDN or full attention, depending on the layer.

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

The whole recurrence is 48 independent 128x128 fp32 state updates per layer, about 3 MB read and written.
`torch_recurrent_gated_delta_rule` is the 51-line reference; prefill uses the chunked form, which processes many
tokens at once with matrix multiplies.

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

RoPE (rotary position embedding) encodes each token's position by rotating pairs of query and key dimensions.
`rope_parameters` carries interleaved mRoPE (`mrope_section [11, 11, 10]`), a multi-stream form meant for images
and video. For text the model expands one position id to every stream, so `apply_interleaved_mrope` copies each
frequency onto itself, which is plain RoPE. The loader precomputes `cos/sin[max_model_len, 64]` once.

**Head.** `logits = W_lm · RMSNorm_{1+w}(x)`. `lm_head` is a separate tensor (`tie_word_embeddings: false`), and
only the last token's logits are needed.

**Per-token state the decode list owns.** Per GDN layer, a conv window `[10240 x 3]` bf16 and
`S[48 x 128 x 128]` fp32; per full-attention layer, a KV ring `[max_model_len x 4 x 256]` x 2 bf16 (the KV cache:
the keys and values of every earlier token); plus `position`

### Byte accounting

`W`, the bytes read per decode token, is the number the performance model divides by, so it is measured from the
files rather than estimated. The index manifest lists 2399 tensors across 6 files, with no duplicated names, 20.03
GB on disk:

| Group | Tensors | GB |
|---|---|---|
| `model.language_model.*.qweight` (int4) | 400 | 12.163 |
| `model.language_model.*.scales` (f16, g64) | 400 | 0.760 |
| `model.language_model.*.qzeros` | 400 | 0.190, dropped at load |
| `model.language_model.*.g_idx` | 400 | 0.012, dropped at load |
| norms, conv1d, A_log, dt_bias, in_proj_a/b (bf16) | 449 | 0.052 |
| `lm_head.weight` (bf16) | 1 | 2.543 |
| `embed_tokens` (bf16, gathered) | 1 | 2.543 |
| `mtp.*` (bf16) | 15 | 0.849 |
| `model.visual.*`, skipped | 333 | 0.921 |

`W` = 12.163 + 0.760 + 0.052 + 2.543 = 15.519 GB read per decode token. The loader's own report of what it makes
resident is 15.540 GB; the difference is tiling pad and norms widened to fp32, itemised in
[13-loader.md](13-loader.md). [05-perf-model.md](05-perf-model.md) uses the loader's figure throughout.

To reproduce: `python3 tools/probe/checkpoint_bytes.py <snapshot>`. It reads the index and the safetensors headers
only, no tensor data, and prints this table, `W`, the `lm_head` int4 and int8 variants and the MTP head.

The engine can read `lm_head` in three forms: bf16 as shipped (2.543 GB); int4 g64, when a self-quantised
checkpoint ships it packed (0.675 GB); or int8 with one fp32 scale per row, quantised at load from the bf16 tensor
(1,271,398,400 + 993,280 B = 1.272 GB, making `W` 14.270 GB). Both CLIs run the int8 head; the tests keep
`loader::LmHeadForm` as the seam through which they select the checkpoint's bf16 head.

### Three facts that drive the design

1. **No MoE.** The hardest kernel family in the larger models of this family is simply absent, so there is no
   grouped GEMM.
2. **vLLM was already at 81.7 % of the roofline on this model** (its early baseline, 31.01 t/s on the first
   stack): 31.01 t/s times 15.540 GB is 482 GB/s of a measured 590. Decode is not won by one big lever but by
   several small ones that multiply ([05-perf-model.md](05-perf-model.md)).
3. **GDN dominates the layer count, not the byte count.** It is 48 of 64 layers, and `sycl-tla` has no example for
   it. But its recurrent state is `48 x 128 x 128 x 4 B` = 3 MB per layer, about 150 MB per token read and
   written across 48 layers, roughly 2 % of `W`. Measured in place, the GDN step kernel is 1.7 % of a decode step
   and runs at 92 % of device bandwidth. GDN is the kernel that must be correct; the linears decide the speed.

### Kernel count, and why it stopped being a worry

At one kernel per operation, the unfused decode step is about 700 kernels, which at design time looked like the
dominant fixed cost. Both halves of that view were measured and both were pessimistic. The decode step is
recorded once into a Level Zero command list (a recorded sequence of kernel launches the GPU can replay without
the host's help) and replayed every token. That list holds 774 launches from 19 modules, and each launch inside
the replayed list costs 0.73 us in place: 0.473 ms, about 1.1 % of a decode step. Fusing kernels just to cut the
launch count is therefore not a lever, and the count went up on purpose when splitting a single-work-group kernel
in two bought 2.4 ms. What the kernels do while they run is the problem; how many there are is not.

### KV and state sizes at the benchmark shape

Per token of context the KV cache holds 16 full-attention layers x 4 KV heads x 256 x 2 (K and V) x 2 B = 64 KB.
At pp4096 + tg256 (a 4096-token prompt and 256 generated tokens), 4352 tokens, that is 285 MB; at
a 16k context it is 1.07 GB. Attention at depth 4096 reads about 270 MB per token, around 1.7 % of `W`.
The GDN conv state is 10240 x 3 x 2 B = 61 KB per layer. A ring KV buffer sized to `max_model_len` is a rounding
error next to the weights, so there is no memory reason for paged KV with one sequence.

### Tokenizer

`Qwen2Tokenizer`, byte-level BPE, 248 044 base + 33 added tokens = 248 077. `lm_head` is padded to 248 320, so the
sampler must never emit ids >= 248 077. The chat template is a separate `chat_template.jinja` (8.9 KB, not in
`tokenizer_config.json`) and uses macros, `namespace`, `raise_exception`, the `is string`, `is iterable` and
`is mapping` tests, thinking mode (`enable_thinking`) and tools. `eos_token_id` is the list `[248046, 248044]`,
that is `<|im_end|>` and `<|endoftext|>`. Details and the pre-tokenizer regex are in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).

### One lesson from a checkpoint that did not work

An earlier 9B target (32 layers, 24 GDN + 8 full attention, hidden 4096, g128) never loaded in vLLM and turned out
to be broken beyond that. Its one lasting lesson is in the loader: its `model_extra_tensors.safetensors` overlapped
the numbered shards, so the loader deduplicates by tensor name and treats the index as the manifest. The 27B
happens not to need that rule; the next checkpoint may.

## Benchmark command

Served decode speeds, ours and vLLM's alike, are measured with this command:

```bash
uvx llama-benchy --base-url http://<host>:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

Prefill is measured on the device (`b70-decode --bench --prefill-length`) rather than through `llama-benchy`'s `pp`
figure, which includes HTTP ([17-running-models.md](17-running-models.md#benchmarks-and-greedy-runs)).

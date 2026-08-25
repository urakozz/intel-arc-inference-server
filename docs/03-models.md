# Target models

All three checkpoints are `model_type: qwen3_5`. Support nothing else.

## Phase 1 and 2 - `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`

Chosen 2026-08-22 after no working 9B checkpoint could be found (`letechlead`
is broken beyond the prefix issue; see doc 10). Phases 1 and 2 now share one
checkpoint: phase 1 serves it without the MTP head, phase 2 adds the head.
Baseline for both is in [BENCHMARKS.md](BENCHMARKS.md): **pp4096 1973, tg256
31.50** without speculation; **42.56 / 45.23** with MTP at 1 / 2 draft tokens.

Everything below is read from the snapshot on the box
(`~/.cache/huggingface/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/2a90776…`),
`config.json` and the safetensors headers. **Measured, not assumed.**

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
attn_output_gate        true            <- q_proj emits q AND a gate: N = 2 × 24 × 256 = 12288
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

Layer pattern is `[LA, LA, LA, FA] × 16` - full attention at layers
3, 7, 11, …, 63:

**48 linear-attention (GDN) layers, 16 full-attention layers.**

Quantisation: GPTQ packing produced by AutoRound 0.15.0 (`quant_method:
"gptq"`, `provider: "auto-round"`), int4, **symmetric, group_size 64**,
`desc_act: false`. `lm_head` is **not** quantised. 98 `dynamic` exclusion
rules keep `mtp.*` and every GDN layer's `in_proj_a`/`in_proj_b` in bf16
(doc 02). The `"+:.*mtp.*"` bug described in BENCHMARKS.md is already fixed in
the cached `config.json` - the rules now read `-:`.

### Per-layer tensors (from the headers)

GDN layer (e.g. layer 0), K → N:

| Tensor | Shape / role | Bytes |
|---|---|---|
| `linear_attn.in_proj_qkv` | 5120 → 10240 = q 16×128 ‖ k 16×128 ‖ v 48×128, int4 | 26.2 MB + 1.6 MB scales |
| `linear_attn.in_proj_z` | 5120 → 6144 (output gate), int4 | 15.7 + 1.0 |
| `linear_attn.in_proj_a`, `in_proj_b` | 5120 → 48 each, **bf16** (decay / beta per value head) | 0.5 + 0.5 |
| `linear_attn.conv1d.weight` | `[10240, 1, 4]` bf16 - depthwise, 4 taps, over qkv | 0.08 |
| `linear_attn.A_log`, `dt_bias` | `[48]` each | - |
| `linear_attn.norm.weight` | `[128]` - gated RMSNorm per value head | - |
| `linear_attn.out_proj` | 6144 → 5120, int4 | 15.7 + 1.0 |
| `mlp.gate_proj`, `up_proj` | 5120 → 17408 each, int4 | 44.6 + 2.8 each |
| `mlp.down_proj` | 17408 → 5120, int4 | 44.6 + 2.8 |
| `input_layernorm`, `post_attention_layernorm` | `[5120]` | - |

≈ **204 MB per GDN layer**, × 48 = 9.8 GB.

Full-attention layer (e.g. layer 3):

| Tensor | Shape / role | Bytes |
|---|---|---|
| `self_attn.q_proj` | 5120 → 12288 = 24 heads × 256 × (q ‖ gate), int4 | 31.5 + 2.0 |
| `self_attn.k_proj`, `v_proj` | 5120 → 1024 = 4 × 256 each, int4 | 2.6 + 0.2 each |
| `self_attn.q_norm`, `k_norm` | `[256]` - per-head RMSNorm before RoPE | - |
| `self_attn.o_proj` | 6144 → 5120, int4 | 15.7 + 1.0 |
| `mlp.*` | as above | 133.7 + 8.4 |

≈ **198 MB per FA layer**, × 16 = 3.2 GB.

Top level: `embed_tokens` BF16 `[248320, 5120]` 2.54 GB (gathered, ~0
traffic); `lm_head.weight` BF16 `[248320, 5120]` **2.54 GB, read in full every
token**; `norm.weight`.

MTP head (phase 2), all **bf16**, 0.85 GB: `mtp.fc` `[5120, 10240]` (fuses
embedding ‖ hidden), `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, one full
transformer layer (`self_attn` q/k/v/o + `mlp` at the main model's shapes),
`mtp.norm`. It shares `embed_tokens` and `lm_head` with the main model - so a
draft step reads **0.85 + 2.54 GB**, and `lm_head` width matters twice.

### Layer math - verified in the modeling file

From `transformers 5.15.0`, `models/qwen3_5/modeling_qwen3_5.py` inside the
reference image (doc 10). This is the contract the kernels and the oracle
(doc 04, testing) must agree on. `x` is one token's hidden vector, `⊙` is
elementwise, all norms run in fp32.

**RMSNorm - Gemma-style `(1 + w)`.** `Qwen3_5RMSNorm` (used for
`input_layernorm`, `post_attention_layernorm`, the final `norm`, `q_norm`,
`k_norm`) computes `x · rsqrt(mean(x²) + 1e-6) ⊙ (1 + w)`. **The `+1` is not in
the weights**; the loader bakes it in (store `1 + w`, **as fp32** - the
reference does the whole product in fp32, so rounding the multiplier back to
bf16 would add an error it does not have; controller ruling 2026-08-25, doc 13)
so the kernel is a plain RMSNorm. `Qwen3_5RMSNormGated` (GDN output norm) is plain `w` - no `+1` - and
multiplies by `silu(z)` *after* normalising.

**Decoder layer.** `x += mixer(norm₁(x)); x += mlp(norm₂(x))`, where
`mlp(h) = W_down · (silu(W_gate h) ⊙ W_up h)`.

**GDN layer** (`Qwen3_5GatedDeltaNet.forward`, decode path):

```
qkv  = W_qkv h                       # 10240 = q 16×128 | k 16×128 | v 48×128
z    = W_z   h                       # 6144  = 48×128
b, a = W_b h, W_a h                  # 48 each, bf16 weights
qkv  = silu(conv1d_4tap(qkv))        # depthwise over the last 4 tokens; state = previous 3; no bias
β    = sigmoid(b)                    # [48]
g    = −exp(A_log) ⊙ softplus(a + dt_bias)      # [48], fp32
q, k = repeat_interleave(q, 3), repeat_interleave(k, 3)   # v-head h uses k-head h // 3
q, k = l2norm(q), l2norm(k)          # per head, eps 1e-6;  q ⋅= 1/√128
per v-head h, state S_h ∈ fp32[128 k × 128 v]:
  S_h  = S_h · exp(g_h)
  kv   = kᵀ S_h                      # [128 v]
  Δ    = (v − kv) ⊙ β_h
  S_h += k ⊗ Δ                       # rank-1 update
  o_h  = qᵀ S_h                      # [128 v]
o    = RMSNormGated(o, z)            # per head: w ⊙ (o · rsqrt(mean(o²)+eps)) ⊙ silu(z)
out  = W_out · o                     # 6144 → 5120
```

The whole recurrence is 48 independent 128×128 fp32 state updates per layer -
~3 MB read + written. `torch_recurrent_gated_delta_rule` is the 51-line
reference; the chunked form is used for prefill.

**Full-attention layer** (`Qwen3_5Attention.forward`):

```
qg = W_q h   viewed as 24 heads × [q 256 | gate 256]   # per-head interleaved, NOT two halves
k  = W_k h   →  4 × 256 ;  v = W_v h  →  4 × 256
q, k = RMSNorm_{1+w}(q), RMSNorm_{1+w}(k)                # over 256, per head
RoPE on the first 64 dims only (partial_rotary_factor 0.25), θ = 1e7,
  inv_freq_i = θ^(−2i/64), i < 32; rotate_half over the 64-slice (non-interleaved halves);
  dims 64..255 pass through
GQA 6:1: q-head j reads kv-head j // 6 ;  scores = q·k / √256 ; softmax in fp32
attn = softmax(scores) · v  →  24 × 256 = 6144
attn = attn ⊙ sigmoid(gate)                            # gate is 6144, head-major
out  = W_o · attn                                      # 6144 → 5120
```

`rope_parameters` carries interleaved mRoPE (`mrope_section [11, 11, 10]`).
For text the model expands one position id to every stream, so
`apply_interleaved_mrope` copies each frequency onto itself - **plain RoPE**.
The loader precomputes `cos/sin[max_model_len, 64]` once.

**Head.** `logits = W_lm · RMSNorm_{1+w}(x)`; `lm_head` is a separate tensor
(`tie_word_embeddings: false`); only the last token's logits are needed.

**Per-token state the decode list owns:** per GDN layer a conv window
`[10240 × 3]` bf16 and `S[48 × 128 × 128]` fp32; per FA layer a KV ring
`[max_model_len × 4 × 256]` × 2 bf16; plus `position`.

### Byte accounting - `W` is measured

Index manifest, 2399 tensors across 6 files, **no duplicated names** (unlike
the 9B), total 20.03 GB on disk:

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

**`W` = 12.163 + 0.760 + 0.052 + 2.543 = 15.52 GB per decode token** (no MTP).
See doc 05 for what follows from it.

Reproduce: `python3 tools/probe/checkpoint_bytes.py <snapshot>` - it reads the
index and the safetensors headers only (no tensor data) and prints this table,
`W`, the `lm_head` int4/int8 variants and the MTP head.

### Three facts that should drive phase 1

1. **No MoE.** The hardest kernel family in the later models is simply absent.
   No grouped GEMM until phase 3.
2. **The model is already at 81% of the roofline under vLLM.** 31.50 t/s ×
   15.52 GB = 489 GB/s of a measured 600. Phase 1 is not won by one big lever;
   it is won by four small ones that multiply (README, doc 05).
3. **GDN dominates the layer count, not the byte count.** 48 of 64 layers, and
   `sycl-tla` has no example for them - but the recurrent state is
   `48 × 128 × 128 × 4 B` = **3 MB per layer** (fp32), ~150 MB per token
   read + written across 48 layers: ~2% of `W`. GDN is the kernel that must be
   *correct*; GEMV is the kernel that decides the *speed*.
   `vllm-xpu-kernels/csrc/xpu/gdn_attn/` is a SYCL reference for both the
   recurrent and the chunked form (doc 06).

And one that drives the replay design: **~700 kernels per token** at one
kernel per op (≈ 11 per GDN layer, ≈ 9 per FA layer, plus `lm_head` and
sampling). At a few µs of fixed cost each inside a command list that is
2-3 ms of a ~26 ms step - the same order as the host overhead being removed.
(Design-time framing, 2026-08-22. Both halves were later measured and both were
pessimistic: the list is 645 kernels at 0.52 µs = 0.335 ms estimated, **0.473 ms
measured in situ 2026-08-25** (0.733 µs/launch - doc 07 #5, docs/15), and the
step is 42.14 ms - docs 07 #5 and 05. The conclusion inverted and the in-situ
number keeps it inverted: fusion is *not* on the phase-1 critical path.)
Fusion (norm into GEMV prologue, conv1d + l2norm + recurrence + gated norm
into one GDN kernel, gate ‖ up ‖ SiLU into one GEMV) is part of phase 1, not a
later optimisation. Doc 07 #5 measures the fixed cost first.

### KV and state sizes at the benchmark shape

Per token of context: 16 FA layers × 4 kv heads × 256 × 2 (K, V) × 2 B =
**64 KB**. At pp4096 + tg256 = 4352 tokens → 285 MB; at `--max-model-len 16k`
→ 1.07 GB. Attention at depth 4096 reads ~270 MB per token, ~1.7% of `W`.
GDN conv state is 10240 × 3 × 2 B = 61 KB per layer. A ring KV buffer sized to
`max_model_len` is a rounding error next to the weights; there is no memory
reason for paged KV in v1.

### Tokenizer

`Qwen2Tokenizer`, byte-level BPE, **248 044 base + 33 added tokens = 248 077**;
`lm_head` is padded to 248 320 - the sampler must never emit ids ≥ 248 077.
The chat template is a separate **`chat_template.jinja`** (8.9 KB, not in
`tokenizer_config.json`) and uses macros, `namespace`, `raise_exception`, the
`is string` / `is iterable` / `is mapping` tests, thinking mode
(`enable_thinking`) and tools. `eos_token_id` is the list `[248046, 248044]` =
`<|im_end|>`, `<|endoftext|>`. Details and the pre-tokenizer regex in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).

### The 9B that was not

`letechlead/Ornith-1.5-9B-INT4-W4A16-AutoRound` (32 layers, 24 GDN + 8 FA,
hidden 4096, g128) was the original phase-1 target. It does not load in vLLM
and turned out to be broken beyond that; no other 9B on the box works. Its
one lasting lesson is in the loader: its `model_extra_tensors.safetensors`
overlapped the shards, so **the loader deduplicates by tensor name and honours
the index as the manifest** - the 27B happens not to need it, the next
checkpoint may.

## Phase 2 - MTP on the same checkpoint

Adds **MTP speculative decoding** using the shipped 1-layer head (above).
vLLM reaches 42.56 t/s at 1 draft token and 45.23 at 2 (BENCHMARKS.md), paying
full weight traffic on every draft *and* verify step. The `M ∈ [1,8]` kernel
(doc 08) makes the verify step cost one weight read for `k + 1` tokens; doc 05
has the ceiling estimate.

MTP notes carried over from the vLLM work:

- Some published checkpoints declare the MTP tensors quantised while shipping
  them unquantised; this one did (`"+:.*mtp.*"`) and the cached `config.json`
  has been corrected. Verify `dynamic` against the tensors actually present
  before blaming the runtime.
- The head reads `lm_head` for its draft logits, so quantising `lm_head` pays
  twice per step in phase 2.

## Phase 3 - `olka-fi/Ornith-1.0-35B-MXFP4`

Adds **MoE (grouped GEMM)** and **MXFP4**. No MTP.

Grouped GEMM is the big new kernel. Notes from reading the `vllm-xpu-kernels`
v0.1.12 source, worth not re-deriving:

- Their `XpuFusedMoe` is **not fused**: per MoE layer per step it issues
  `remap_hidden_states` → grouped GEMM → `silu_and_mul` → grouped GEMM →
  `moe_gather` = **4 kernel launches and 5 allocations**. At 40 layers that is
  160 launches per decode step. This is exactly the host overhead this project
  exists to delete.
- Every work-group scans **all** experts (`for i in num_experts`, reading
  `rows_per_expert[i]` from global memory) even when only 8 are non-empty.
- MXFP4 there runs **W4A16**, not W4A4 - activations stay bf16.

## Phase 4 - `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`

MoE **and** MTP together, GPTQ packing. No new kernel families beyond phases 2-3;
the work is making speculation and expert routing coexist.

## Benchmark command

Every number in these docs comes from this, so new numbers must too:

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

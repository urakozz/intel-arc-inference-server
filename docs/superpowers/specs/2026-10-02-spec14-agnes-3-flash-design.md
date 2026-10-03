# Spec 14 - Agnes 3.0 Flash, the second model

**Status:** design, 2026-10-02, operator approved approach 1 (§2) the same day; re-scoped
2026-10-03 into write-now / validate-on-the-box (§6).

**Checkpoint:** [`urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ`](https://huggingface.co/urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ)
(the operator's quant of `Agnes-AI/Agnes-3.0-Flash`). Reference implementation: the
operator's vLLM PR [#57003](https://github.com/vllm-project/vllm/pull/57003), itself based
on the Agnes team's SGLang patch.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. What Agnes is, against what the engine runs

Read from the checkpoint's `config.json`, `quantization_config` and tensor index
(2026-10-02):

| | Qwen3.8-27B (today) | Agnes 3.0 Flash |
|---|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` | `AgnesForConditionalGeneration` |
| layers | 64: 48 GDN + 16 FA, FA at `l % 4 == 3` | **72: 54 GDN + 18 FA**, FA at `l % 4 == 3` (`global_attention_interval` 4) |
| hidden, vocab | 5120, 248320 | same |
| FA | 24 q-heads, 4 kv-heads, head_dim 256, partial rotary 0.25 | same |
| GDN | 16 k-heads, 48 v-heads, dim 128, conv 4 | same |
| MLP | SwiGLU, intermediate 17408 | the same **plus `mlp.parallel_ffn`**: a SwiGLU of intermediate 2048, its output added to the MLP's |
| tensor names | `linear_attn.*`, `self_attn.*` | `delta_attn.*`, `global_attn.*` |
| MTP head | `mtp.*`, 15 tensors, bf16 | same (15 tensors, `model_extra_tensors.safetensors`) |
| quantisation | AutoRound GPTQ int4, g64, sym; `in_proj_a/b`, `lm_head`, `mtp.*`, vision bf16 | **identical**, and the parallel FFN's three projections are int4 g64 too |
| max positions | 262144 | 262144 |

So the engine's math path applies unchanged; two things differ: the layer count, and the
parallel FFN.

## 2. The decision

**Approach 1 (operator, 2026-10-02): a model descriptor replaces the hardcoded layer counts,
and the parallel FFN is folded into the MLP at load.**

**The fold.** Per layer, `y = down(silu(gate x) * up x) + down_p(silu(gate_p x) * up_p x)`.
With `gate' = [gate ; gate_p]`, `up' = [up ; up_p]` (output rows stacked, intermediate
17408 + 2048 = **19456**) and `down' = [down | down_p]` (input columns concatenated),
`y = down'(silu(gate' x) * up' x)`: the sum happens inside the down projection's dot product.
Exact up to accumulation order. Alignment holds everywhere the engine tiles:

- g64 groups: 17408 = 272 x 64, so no group of `down'` straddles the join;
- spec 5's 1024-column slabs and 1024-blocked Hadamard: 17408 = 17 x 1024, 19456 = 19 x 1024;
- the fused gate‖up of the engine (`N` = 2 x intermediate) becomes 38912 = 38 x 1024.

The fold is a loader operation on packed int4. GPTQ stores `qweight` as `[K/8, N]` and
`scales` / `qzeros` as `[K/64, N]`: gate and up are concatenated along N (columns), down along
the packed K (rows, at `qweight` row 2176 and `scales` row 272, a group boundary). It is done at load, so the checkpoint stays as published.

Rejected: the parallel FFN as its own three launches per layer (+216 launches per token, small
GEMVs at worse bandwidth than one wider one).

## 3. Design

### 3.1 The model descriptor

`model::Qwen35`'s `static constexpr` shape (`kLayers` 64, `kIntermediate` 17408, the derived
48 / 16 layer counts, `is_fa`) becomes a `ModelDesc` chosen by the loader from `config.json`'s
architecture: `{layers, gdn_layers, fa_layers, intermediate, has_parallel_ffn, names}`. The
per-layer constants that are the same for both models (hidden, heads, head dims, vocab, rotary)
stay `constexpr`. Users today: `src/model/qwen35.{h,cc}`, `src/runtime/{buffers,capture,engine}.cc`
(17 references to the layer constants), `src/loader/loader.{h,cc}`, the prefill path
(`src/runtime/prefill/{step,int8,linear_l0,gemm_l0}`), and the intermediate size in kernels
(`src/kernels/prep.cl`, `src/kernels/prefill/{pf_gemm,pf_prep}.cl`) and the kernel CMake.

- **Buffers** are sized from the descriptor: `gdn_state [gdn_layers]`, `conv_ring
  [gdn_layers]`, `kv_k/kv_v [fa_layers]`, and spec 7's `state_bytes` / `kv_bytes`.
- **Kernels** with the intermediate size baked in get Agnes variants (`INTERMEDIATE=19456`)
  beside today's; the capture names the variant from the descriptor. New GEMV shapes for
  gate‖up `N=38912, K=5120` and down `N=5120, K=19456`, with tuning-table rows
  (`docs/probe-gemv-2026-08-24.md`'s method), and prefill GEMM / h8 slab counts (38 and 19).
- **The captured decode list** grows from 64 to 72 layers (774 kernels today; derived ~870).

### 3.2 The loader

- Name map `delta_attn.` → `linear_attn.`, `global_attn.` → `self_attn.`, as the PR's
  `WeightsMapper`.
- The fold of §2; the classification table (docs/13) gains the parallel FFN rows.
- `model_extra_tensors.safetensors` (the MTP head) as for Qwen3.8; the vision tower skipped.

### 3.3 Memory and context (derived)

| | Qwen3.8-27B | Agnes |
|---|---:|---:|
| loaded weights, bf16 / int8 `lm_head` | 18.1 / 16.8 GB | ~21.0 / ~19.7 GB |
| KV per position (bf16) | 64 KiB | 72 KiB |
| KV at 65536 / 131072 | 4.3 / 8.6 GB | 4.8 / 9.7 GB |

128k with bf16 KV is ~31.9 of 32.5 GB with the int8 head: not shipped. **max_len 65536 is
Agnes's ceiling** until spec 12 (int8 KV) lands; then 131072 fits (~4.8 GB of KV). The MTP lists
(spec 8, compiled at 16384) get Agnes variants at the same max_len.

### 3.4 The CPU reference

`tools/oracle/dump.py` builds `transformers`' `Qwen3_5ForCausalLM`. For Agnes: the same model
with the name map and the parallel FFN added to each MLP (a small subclass, as the PR does in
vLLM), checked once against the checkpoint's own `modeling_agnes.py` (remote code) on a short
prompt. Golden sets dumped into `oracle-out-agnes/`.

### 3.5 CLI

The model is picked from the checkpoint; no flag. `b70-serve` / `b70-decode` with
`urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ` resolve it from the HF cache as today. The chat
template is the checkpoint's `chat_template.jinja`; tool-call format checked against it (the
Qwen XML parser of spec 7 §3.5 if it matches; a parser addition if not).

## 4. Correctness gates

- **G0, nothing moves for Qwen3.8:** every existing registration passes, the golden gates and
  replay determinism **bitwise unchanged** (the descriptor refactor is behaviour-neutral);
  kernel binaries for Qwen3.8 byte-identical.
- **G1, the fold:** on the CPU, the folded MLP against the unfolded two-branch MLP on real layer
  inputs: cosine >= 0.999999 per row; on the card, the folded GEMVs/GEMMs against a host
  reference at the new shapes (the `tests/kernels/*` pattern).
- **G2, golden:** the golden gates on Agnes (`l0`, `l0-int8`) against its CPU reference, the
  same tie rule.
- **G3:** tool-call set A4 on Agnes, reported against Agnes's own bf16 reference (the
  unquantised `Agnes-AI/Agnes-3.0-Flash` on the CPU) — no bar until the first measurement,
  recorded for the operator.
- **G4:** spec 7's C1/C2, spec 8's M1/M2/M3, spec 9's int8 head L3, spec 10's A1 on Agnes.
- **G5:** passkey 3/3 at 60k (the max_len ceiling of §3.3).

## 5. Speed bars

Idle box, device 0, interleaved pairs, median of 3.

- **Recorded first:** decode at 4k / 32k / 60k depth, pp4096, load time, memory report.
- **Derived expectation:** decode ~26 t/s at 4k (~17.2 GB per token against Qwen3.8's ~14.3 GB
  at 31.4 t/s), pp4096 ~1850-1900 t/s (~1.18x Qwen3.8's FLOPs per token).
- **vLLM row:** the operator's PR on the box, `llama-benchy` with the operator's flags (pp4096,
  tg256, depth 1, `--no-cache --exact-tg --latency-mode generation`), against `b70-serve` on the
  same flags: recorded, no bar.

## 6. Stages

**Re-scoped 2026-10-03 (operator): write first, validate on the box later.** The box is
unavailable for a while, so the code is written now and checked as far as the Mac allows; every
gate that needs the card runs in one validation session when the box is back.

- **Phase 1, write (now, no box):** plan `2026-10-03-spec14-write-phase.md`. All of 14a-14d's code
  on one branch `spec14-agnes`: the CPU reference and the fold (Python), `ModelDesc` (the
  refactor first), the loader's name map and fold, the Agnes descriptor, kernel variants and
  CMake, **provisional** GEMV tuning rows (copied from the nearest Qwen3.8 shapes, marked),
  the chat template and tool-call check. Checked locally: host C++ compiled and its host-only
  tests run on the Mac (Apple clang; Level Zero headers from the open-source loader repo for a
  syntax check of the runtime); the Python tests and the fold proof (G1, CPU part) on the real
  checkpoint in the Mac's HF cache; the template rendered against HF's `apply_chat_template`.
- **Phase 2, validate (the box):** `2026-10-03-spec14-validation-checklist.md`, in order: build;
  **G0 first** (Qwen3.8 bitwise, binaries byte-identical); kernel tests at the new shapes; GEMV
  tuning measured and the provisional rows replaced; golden sets and tool-call references dumped
  (bf16 Agnes on the box's RAM); G1 (card), G2, G4, G5, A4 (G3); speed and the vLLM row; the
  record. The branch merges only after it.

The earlier per-stage plans (`2026-10-02-spec14a`..`14d`) are superseded by these two and kept for
their task detail, which the write plan points to.

## 7. Out of scope

- The vision tower (text only, as the PR's `--language-model-only` test).
- 128k context for Agnes (after spec 12).
- Other Qwen3.5-family checkpoints (the descriptor makes them easier; not gated here).

# Spec 6 - fused flash attention in prefill, and 128k context

**Status:** design, 2026-09-25, for operator review.

**Order, set by the operator:** spec 5 (int8 prefill linears) is the
foundation, done 2026-09-24. This spec is the first feature on top of it.
Prefix caching and MTP come after.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

The operator's workload is long agentic coding sessions (memory:
target-workload-agentic-coding). Two things stop the engine running them past
16k tokens:

1. **Prefill attention materialises its scores.** The composed path (ruling
   A14: QK^T GEMM, then `pf_softmax_causal`, then the PV GEMM) writes S (fp32)
   and P (bf16) as `[6][kC][max_len]` buffers, `pf_s` and `pf_p`. At
   max_len 131072 that is 6 x 2048 x 131072 x 6 B, **about 9.7 GB** (derived),
   which does not fit beside the model in 32 GB.
2. **Decode attention is built for one max_len.** `attn_decode` / `attn_reduce`
   bake `MAXLEN` as a stride. Only L16384 exists for M = 1, and the captured
   list's grid is `max_len / 64` blocks whatever the depth.

The target is **128k context** (operator, 2026-09-25): KV cache bf16,
16 FA layers x 4 kv heads x 256 x 2 (K, V) x 2 B = 64 KiB per position, so
**8.4 GB** at 131072 (derived). With ~18 GB of loaded model (the loader's
18.087 GB report) that leaves ~5 GB for prefill scratch, decode state and slack
once `pf_s` / `pf_p` are gone.

## 2. The decision

**A fused flash-attention kernel of our own, on the Level Zero list, bf16 DPAS
with fp32 accumulation and fp32 online softmax** (operator: approach 1, "fused
one of course"). Rejected in the design discussion:

- **The composed GEMMs tiled over KV positions:** bounded memory, but S and P
  still go through memory every tile, so speed stays where it is.
- **sycl-tla's `XeFMHAFwdKernel`:** it is already in the build tree, but it
  brings back the SYCL-L0 boundary (host waits) that spec 2.1 removed, and its
  build log shows ~500-register spills.

Attention math stays bf16. int8 QK^T or PV is out of scope, because attention
scores are the most precision-sensitive tensor in the model and spec 5's
evidence covers the linears only.

## 3. Design

### 3.1 `pf_flash_attn`

- **Inputs:** `pf_q` bf16 `[C][24][256]` (RoPE'd, from `pf_attn_prep_q16`); the
  layer's KV cache bf16 `[pos][4][256]`, into which the chunk's own K and V are
  already written (as today). **Output:** `pf_attn` bf16 `[C][24][256]`, before
  `pf_attn_gate`. Everything before and after the kernel is unchanged.
- **Math, per query row and head:** stream KV tiles; S = q k^T in fp32, then
  x `ATTN_SCALE` (1/16 = 1/sqrt(256), `pf_attn.cl`'s constant, applied to the
  fp32 scores as `pf_softmax_causal` does); causal mask
  (position <= base + row); running max m, running sum l; O is rescaled by
  exp(m_old - m_new) and accumulates P V with P = exp(s - m) rounded to bf16
  as the DPAS operand. At the end O / l, then bf16. exp is `exp`, not
  `native_exp`, matching `pf_softmax_causal`. **One rounding point differs by
  construction:** the composed path rounds the *normalised* P (already divided
  by the row sum) to bf16. Flash rounds the *unnormalised* exp(s - m) and
  divides by l at the end. Together with the summation order, this is why K1
  is a cosine bar and not bit-exactness.
- **Tiling:** one work-group per (kv head, query tile). The six q-heads of
  the kv group are one work-group, so each K and V tile is read from global
  memory once and staged in SLM for all six. KV tiles entirely past the query
  tile's last row are skipped; only the diagonal tile is masked.
- **The tile shape is not chosen here** (P1). Head dim 256 makes the O
  accumulator large: 6 heads x R rows x 256 x 4 B is 393 KB for R = 64
  (derived), against 512 KB of GRF per 32-sub-group work-group at 256 GRF.

### 3.2 Integration

- `attn_chunk` calls `pf_flash_attn` in place of QK^T, softmax and PV on the
  `l0` and `l0-int8` backends. The composed path stays selectable
  (`B70_PREFILL_ATTN=composed`) as the correctness reference, the role
  sycl-tla plays for the GEMM.
- `pf_s` and `pf_p` are allocated **only when the composed path is selected**.
  On the default path prefill scratch no longer scales with max_len.
- The launch arithmetic in `step.cc` and `attn.h` (`attn_chunk_launches`)
  follows the new kernel's count, and `prefill_smoke_test` pins it.

### 3.3 128k

- `attn_decode` and `attn_reduce` variants at L131072 (M = 1), beside L16384.
- `--max-len` accepted up to 131072 by both CLIs. `loader::load`'s RoPE table,
  the KV cache, `attn_part` (`[24][max_len/64][M][258]` fp32, 202 MB at 128k,
  derived) and any other max_len-sized buffer are checked to fit, with a load
  report line that sums them.
- **Decode grid at shallow depth.** The captured list launches `max_len / 64`
  = 2048 blocks per head group every token. Blocks past `pos` exit at once, but
  their dispatch is paid, and `attn_reduce` walks the partial table. If F4
  (§5) fails, the grid becomes **indirect**: `attn_prep` writes this step's
  group count (`ceil((pos + 1) / 64)`) into device memory, and the captured
  list launches `attn_decode` / `attn_reduce` with
  `zeCommandListAppendLaunchKernelIndirect`. The list is still captured once.

## 4. Correctness gates

- **K1, kernel.** `pf_flash_attn` against the composed path on the same Q, K
  and V (real captures from one FA layer, and random), at depths 2k, 16k and
  32k, chunks of 2048 and 300 rows. Bar: per-(row, head) output cosine
  >= 0.99999 and the max abs error recorded. Not bit-exact: online softmax
  sums in a different order.
- **K2, golden gates.** Every existing registration passes unchanged: prose,
  code and cjk on `l0` and `l0-int8`, the chunk-1000 and chunk-16 pairs, and
  consistency, determinism and replay.
- **K3, long context.** There is no CPU oracle at 128k, so:
  - flash against composed end to end at 32k context (the composed path still
    fits there): logits cosine >= 0.9999 and the same greedy tokens for 64
    steps;
  - at 128k: finite logits, determinism across two runs, replay bitwise;
  - **passkey retrieval**: a 5-digit number stated once near the start of
    ~120k tokens of filler text must be reproduced when asked for at the end,
    at three placements (5 %, 50 %, 95 % depth). This is the behavioural check
    that long attention is right.
- **K4.** `prefill_smoke_test` launch counts and `prefill_replay_test` pass on
  the new kernel.

## 5. Speed bars

Idle box, device 0, interleaved control and candidate, median of 3.

- **P0, baseline (first, before any kernel).** The attention phase of a
  4096-id prefill today (the 148.2 ms of `prefill-parity-2026-09-20.md`
  predates causal QK^T, parity S3), prefill of one chunk at 8k and 16k depth,
  and decode at depths 4k and 16k at max_len 16384.
- **F1.** The attention phase of a 4096-id prefill **<= 80 ms**. vLLM's flash
  kernel measured 63.1 ms (`prefill-parity-2026-09-20.md`). pp4096 must not
  fall below spec 5's 2104.50 t/s record.
- **F2.** One 2048-id chunk prefilled at 32k, 64k and 128k depth: recorded,
  and the attention part must reach **>= 60 % of the bf16 DPAS peak** (183.45
  TFLOP/s) on the full-tile KV range (derived from its FLOP count).
- **F3.** Decode at depths 4k, 32k, 64k and 128k with max_len 131072:
  **>= 90 % of the bandwidth-derived rate**: the effective bandwidth decode
  measures today divided by the bytes one token reads (weights plus KV up to
  that depth). About 18 t/s at 128k (derived).
- **F4.** Decode at 4k depth with max_len 131072 within **2 %** of the same
  depth at max_len 16384. If it fails, §3.3's indirect grid is required.

## 6. Stages

- **P0 (measure):** the baselines of §5.
- **P1 (probe):** a tile-shape sweep for `pf_flash_attn`: query rows
  32/64/128, KV tile 64/128, heads per work-group 6 against 2 or 3 (halving
  the O accumulator, reading K/V twice or three times), at 256 GRF. Measured on
  real shapes at 16k depth, with a K1-style correctness check per arm. Its
  winner is the kernel's tiling.
- **P2 (kernel):** `pf_flash_attn` in `src/kernels/prefill/`, K1.
- **P3 (integration):** `attn_chunk`, the composed path as a selectable
  reference, lazy `pf_s` / `pf_p`, launch arithmetic, K2 and K4, F1.
- **P4 (128k):** the L131072 variants, the max_len plumbing and load report,
  F3 and F4, and the indirect grid if F4 fails.
- **P5 (long-context gates):** K3, F2, the BENCHMARKS rows at depth, and
  this spec's amendment with results.

**Stopping rule.** If K1 fails at every P1 tile shape, stop and record: the
design assumption that fp32 online softmax matches the composed path within
0.99999 is then wrong, and that needs the operator's call before more kernel
work.

## 7. Out of scope

- int8 KV cache (it would halve decode's KV bytes at depth, but raises an
  accuracy question of its own); int8 attention math.
- Prefix caching and session continuation, MTP: next, on top of this.
- The ~0.7 s short-chunk floor of the linear path (probe-w4a8 §15.6).
- max_len beyond 131072.

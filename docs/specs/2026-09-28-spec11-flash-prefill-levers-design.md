# Spec 11 - flash prefill: levers 3 to 5

**Status:** design, 2026-09-28, for operator review.

**Order:** independent of specs 10, 12 and 13 (it touches only `pf_flash_attn`).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

Prefill attention dominates long prompts. After spec 6c (8 rows per work-group and
`exp2`, operator ruling 2026-09-27): the attention phase of a 4096-id prefill is
~81 ms (vLLM's kernel: 63.1 ms), and prefill runs 1301 t/s at 64k depth and 915 t/s
at 128k (spec 6 §9, `docs/BENCHMARKS.md` "Spec 6c"). At 64k attention was 51 % of
prefill GPU time before 6c, at 15 % of the bf16 DPAS peak.

`docs/research-flash-prefill-2026-09-27.md` ranked five levers; 1 and 2 shipped in 6c.
With the register spill gone the kernel is no longer instruction-bound: the arms that
cut instructions (diagonal-only masking, conditional rescale, hardware bf16
conversion, deferred row sums) measured -3 % to +1 %. What remains is on the memory
side, which levers 3-5 address.

## 2. The levers

| # | lever | mechanism | estimate (research doc) | risk |
|---|---|---|---|---|
| 3 | **cooperative 2D prefetch of the next K/V tile + one work-group barrier per tile** | the six sub-groups of a work-group (all at the same trip count since RPW 8) prefetch tile t+1 while computing t; vLLM, sycl-tla and our own `pf_gemm` do this; our kernel enables the barrier extension but never uses it | 1.1-1.3x | medium |
| 4 | **split-d for head_dim 256** | each sub-group takes 16 rows and a pair of sub-groups splits the output over d, exchanging P through SLM (oneDNN micro-SDPA, IPEX XeTLA); today every K/V operand feeds exactly one DPAS: 576 B per DPAS against the production GEMM's 192 B | 1.3-1.8x beyond 3, **only if P0 says operand bandwidth is the limit** | high |
| 5 | **32-key K loads** | halves the K load message count; the register layout of the 32-row transposed load must be pinned first (the research's attempt had it wrong: cosine 0.15-0.24) | 1.0-1.1x | low once pinned |

## 3. Design

All three are built as arms of `tools/probe/probe_flash_attn.{cl,cc}` (the spec 6a
harness, real shapes: 24 q-heads, 4 kv-heads, d 256, chunk 2048, depths 0 / 2048 /
30720 and 65536) first, one lever at a time, K1-checked; the winners are promoted into
`src/kernels/prefill/pf_flash_attn.cl` behind defines, the composed path and today's
kernel stay selectable.

## 4. Correctness gates

- **K1** per arm: per-(row, head) cosine >= 0.99999 against fp64, the five spec 6 cases
  plus 6c's single-row, 9-row and tail cases; bitwise repeatable.
- In production: golden gates, K3a (`flash_vs_oracle_test`, tolerances as ruled in spec 6
  §9), `flash_long_test`, passkey 3/3 at 120k, `prefill_split_*` (the continuation test),
  `prefill_replay_test`, full suite.

## 5. Speed bars

Idle box (no other GPU job; load recorded), interleaved pairs, median of 3.

- **P0 (first).** The operand-bandwidth ceiling: a DPAS loop fed by 2D loads from L1 at
  reuse 1, 2 and 4 (Intel publishes no Xe2 L1 load bandwidth). It decides whether lever
  4 is worth building: build it only if the ceiling at reuse 1 is within 1.3x of the
  kernel's current rate.
- **F1'.** pp4096 attention phase **<= 70 ms** (from ~81; vLLM 63.1).
- **F2'.** pp65536 and pp130816 recorded; attention at 64k >= 25 % of bf16 peak.
- No regression in pp4096 t/s or in decode.

## 6. Stages

- **11a, probe:** P0; levers 3 and 5 as harness arms (5 after a small unit probe pins
  the 32-row load layout); lever 4 only if P0 says so. Winner and verdict.
- **11b, build:** the winner in `pf_flash_attn`, the gates, the speed rows, the record.

## 7. Out of scope

- int8 QK^T or PV (attention math stays bf16, spec 6 §2).
- Changing the KV layout (spec 12 may; this spec reads whatever layout is current).

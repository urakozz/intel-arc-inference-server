# Spec 28c - the verify step's attention on DPAS, chosen by shape: M rows x depth

**Status (2026-10-31): DESIGN - nothing built.** Follows spec 28 (small-M verify GEMM). Operator, 2026-10-31: "let's
spec it", after the upstream review found OpenVINO choosing its speculative-verify attention kernel by shape. Builds
after 28b; the operator is asked before the plan starts. No box row is allocated until the plan exists (focus-closing,
2026-10-27).

**Grades** as spec 28: measured, derived, estimated, published by others (never a bar by itself).

## 1. Why

Spec 28 removes the GEMV part of the verify step's extra cost. What is left is mostly not GEMV: spec 28 §1 names the
attention's per-row KV walk, `gdn_step`'s M slot writes and the head. Spec 28a's P0 attributes the measured verify
step into GEMV and non-GEMV time; this spec is the lever P0 names when attention is the largest non-GEMV share.

| fact | value | grade | source |
|---|---|---|---|
| `attn_decode` at M = 4 against M = 1 (v1, before spec 10's v2) | 2.5 -> 9.2 ms | measured 2026-09-27 | docs/probe-mtp-2026-09-27.md |
| today's verify attention | `attn_decode_v2` rows variant (`kernels::attn_v2_rows_variant`): per-row walk on the vector units, K / V waves shared across rows (spec 10) - no DPAS | code | src/kernels/attn_v2.cl, src/runtime/capture.cc |
| OpenVINO `pa_small_q_opt.cl` (subgroup matrix + SLM, int4 or plain KV) for the speculative-verify stage, then a shape rule that falls back to the non-DPAS GQA path for few draft tokens or short contexts: the old small-q kernel 3.85 vs 1.78 at ctx 512, T = 3 | DPAS loses at small T x short ctx | published by others | openvino df348d6faf, e51a8dcb6d |

So on this card a DPAS verify attention wins at some shapes and loses at others. The rule must be measured, not
assumed: keep today's kernel where it wins, use the DPAS one where it wins.

## 2. The kernel

`attn_verify_dpas`: the M query rows of one KV head's GQA group (Qwen3.8: 24 q / 4 kv heads, so 6 x M rows) as the
DPAS A operand, K / V tiles from the paged KV as B, online softmax per row, causal mask inside the last M positions.
Split-K over the KV length with the existing `attn_reduce_v2` style reduce (the same block count `kAttnV2Blocks`), so
long depths fill the card. bf16 KV first; int8 KV (`kv8.cl` rotkv) as a second arm, dequant in registers.

## 3. The selection rule

`verify_attn_kind(M, depth)`: today's `attn_v2` rows variant or `attn_verify_dpas`, from a table measured on the box
(M = 2..16 x depth 512 / 4096 / 32768 / 131072, bf16 and int8 KV), stored beside the capture's `gemv_kind(M)` rule.
No engine flag; an env override (`B70_VERIFY_ATTN=v2|dpas`) for probes only. A shape where the two are within noise
keeps v2 (no new numerics without a win).

## 4. Numerics and gates

- **N1:** `attn_verify_dpas` against the host reference (fp32) per row: within the bf16 bar `attn_v2` already passes.
- **N2:** each row of an M-row verify equals that row decoded alone under the one-ulp tie rule (spec 13a's check);
  bitwise is not required, the summation order differs.
- **N3:** the MTP / DFlash golden gates and D1 unchanged or better at the shapes that switch.
- M = 1 never switches: plain decode stays bitwise (spec 28 §8, decision 1).

## 5. Speed bars

As spec 28 §5 (idle box, device 0, interleaved pairs after warm-up, median of 5 / 3).
- **S1:** per shape, the chosen kernel is never slower than v2 beyond noise (the rule's own check).
- **S2:** verify(8) at depth 32768 improves by at least the attention share P0 measured, less 25 %; depth 4096 no worse.

## 6. Stages

- **28c-P0 (inside plan 28a's attribution, no new probe run):** record the attention's share of the verify step at
  M = 2 / 4 / 8 / 16 x depth 4096 / 32768. GO for this spec when attention is >= 15 % of verify(8) at depth 32768
  after 28b's GEMM; else recorded, parked.
- **28c (build):** the kernel, the host reference, the measured table and rule, N1-N3, S1-S2, BENCHMARKS rows.

## 7. Out of scope

Prefill attention (`pf_flash_attn`); the MoE engines' own attention (Ornith / K2 / Kolibri / Bolt) - a follow-on per
engine once 28c's numbers exist; varlen M per request inside one batch (vllm-xpu-kernels 470c19f) - spec 13b's call.

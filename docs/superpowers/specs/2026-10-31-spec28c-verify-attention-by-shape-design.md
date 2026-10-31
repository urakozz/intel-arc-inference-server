# Spec 28c - the verify step's attention on DPAS, chosen by shape: M rows x depth

**Status (2026-10-31): DESIGN - nothing built.** Follows spec 28 (small-M verify GEMM). Operator, 2026-10-31: "let's
spec it", after the upstream review found OpenVINO choosing its speculative-verify attention kernel by shape. Builds
after 28b; the operator is asked before the plan starts. No box row is allocated until the plan exists (focus-closing,
2026-10-27).

**Plan:** `docs/superpowers/plans/2026-10-31-spec28c-verify-attn-build.md` (one plan: the P0 line from 28a / 28b's
records, then the build). No box row yet (operator, 2026-10-31); the plan describes its stages in prose. §8 lists what
the plan corrected in this text against the code.

**Grades** as spec 28: measured, derived, estimated, published by others (never a bar by itself).

## 1. Why

Spec 28 removes the GEMV part of the verify step's extra cost. What is left is mostly not GEMV: spec 28 §1 names the
attention's per-row KV walk, `gdn_step`'s M slot writes and the head. Spec 28a's P0 attributes the measured verify
step into GEMV and non-GEMV time; this spec is the lever P0 names when attention is the largest non-GEMV share.

| fact | value | grade | source |
|---|---|---|---|
| `attn_decode` at M = 4 against M = 1 (v1, before spec 10's v2) | 2.5 -> 9.2 ms | measured 2026-09-27 | docs/probe-mtp-2026-09-27.md |
| today's verify attention | `attn_decode_v2` + `attn_reduce_v2` at the list's M (`kernels::attn_v2_variant(M, 32)`, under `--kv-cache int8` its twin `attn_v2_kv8_variant`), bound by `attn_pair` in the capture for every M (decode, MTP / lookup verify, DFlash tap verify, `--pp 2` stage verify): per-row walk on the vector units, K / V waves shared across rows, the stride per row so every verify row is **bitwise** its M = 1 step (spec 10 §8) - no DPAS. (`attn_v2_rows_variant` is spec 13a's `BATCH_ROWS` probe build for rows of different sequences, not the verify path.) | code | src/kernels/attn_v2.cl, src/runtime/capture.cc |
| the M range that exists | `attn_v2.cl` compiles M = 1..8 (`#error` above 8); the verify lists stop at M = 8 (`B70_VERIFY_M8`); the int8-KV decode binaries stop at M = 4 (`B70_KV8`; M = 5..8 int8 only as 13a's `_ROWS`) | code | src/kernels/attn_v2.cl, src/kernels/CMakeLists.txt |
| decode q precision | `attn_q` is fp32 and v2's score dot is an fp32 `fma` on it; a bf16 DPAS A operand rounds q to bf16 (prefill's flash does, `pf_attn_prep_q16`) - a numerics change the kernel must price (§2) | code | src/kernels/attn_v2.cl, src/kernels/prefill/pf_flash_attn.cl |
| OpenVINO `pa_small_q_opt.cl` (subgroup matrix + SLM, int4 or plain KV) for the speculative-verify stage, then a shape rule that falls back to the non-DPAS GQA path for few draft tokens or short contexts: the old small-q kernel 3.85 vs 1.78 at ctx 512, T = 3 | DPAS loses at small T x short ctx | published by others | openvino df348d6faf, e51a8dcb6d |

So on this card a DPAS verify attention wins at some shapes and loses at others. The rule must be measured, not
assumed: keep today's kernel where it wins, use the DPAS one where it wins.

## 2. The kernel

`attn_verify_dpas`: the M query rows of one KV head's GQA group (Qwen3.8: 24 q / 4 kv heads, so 6 x M rows) as the
DPAS A operand, K / V tiles from the KV cache as B (`pf_flash_attn`'s verified idioms: K^T by the transposed 32-bit 2D
read, V by the VNNI-transform read, S in the A-operand layout so P feeds PV without data movement), online softmax per
row, causal mask inside the last M positions. Split-K over the KV length **in v2's own blocking**: the same grid
(4 kv heads, `kAttnV2Blocks` = 32), the same per-row stride `ppw(L)` and row groups (at most two strides a launch),
and v2's partials `[24][32][M][258]` with v2's meaning (natural-log max, sum, unnormalised acc) - so the reduce is
`attn_reduce_v2` / `attn_reduce_v2_kv8` itself, the existing binary at that M, unchanged, and a row's blocks depend only
on its own key count (property R below). `exp`, not `exp2` (spec 10 §8). bf16 KV first; int8 KV (`kv8.cl` rotkv) as a
second arm: K and V int8 are exact bf16 operands, the key's scale on the score and folded into P (`pf_flash_attn_kv8`'s
form), q from `attn_prep_kv8` (already rotated). **q's precision is a lever:** q rounded to bf16 (prefill's numerics) or
split hi + lo into two bf16 operands (two QK^T DPAS, ~16 significant bits); N1 / N3 decide, the cheaper passing form
ships.

## 3. The selection rule

`verify_attn_kind(M, keys, kv)`: today's `attn_v2` pair or `attn_verify_dpas`, from a table measured on the box
(M = 2..8 x depth 512 / 4096 / 32768 / 130816, bf16 and int8 KV; M = 16 of the DPAS kernel recorded only - there is no
v2 and no verify list at 16), stored beside the capture's `gemv_kind(M)` rule (spec 28b's). Per (M, KV form) the table
holds one number, `dpas_from`: the row-0 key count from which DPAS is bound (never, or a measured depth with every
deeper measured depth a win too). No engine flag; an env override (`B70_VERIFY_ATTN=v2|dpas`) for probes only. A shape
where the two are within noise keeps v2 (no new numerics without a win). The rule applies to Qwen3.8's heads (24 / 4,
Agnes's too); any other model's heads keep v2.

**Depth is not known when a list is captured.** A captured list replays at every position (v2 derives its stride on
the device from `Control::pos`), so a capture-time rule can only see M. Where the table's winners do not depend on
depth (`dpas_from` is 0 or never for every M), the rule is applied at capture by M alone. Where some M switches at a
depth, the engine captures that M's verify list twice (v2 and DPAS - only the 16 attention pairs differ) and the host
picks the list at submit by `pos` (`Engine::verify` knows it); launch counts are unchanged, no kernel early-outs. The
cost of the second lists (capture time, list memory) is recorded.

## 4. Numerics and gates

- **N1:** `attn_verify_dpas` against the host reference (fp64, on v2's inputs: fp32 q, bf16 or int8 + fp16-scale KV)
  per (q head, row): cosine >= 0.99999 - the bar `attn_v2` passes (spec 10 A1) - with v2's own error on the same inputs
  recorded beside it.
- **N2:** (a) **property R**: every row bitwise equal to the same query at the same key count in any other M >= 2
  launch and whatever the other rows hold (spec 28 N3's form - what AdaptiveK and spec 13's checks need at M >= 2);
  (b) each row against that row decoded alone (v2, M = 1): spec 10 A1's bar, cosine >= 0.99999 per (q head, row), max
  abs recorded. Bitwise against M = 1 is not required: the summation order differs. (The text before 2026-10-31 cited a
  "one-ulp tie rule" of spec 13a; 13a defines none - its check is bitwise row mode against the slot alone.)
- **N3:** the MTP / DFlash golden gates and D1 unchanged or better at the shapes that switch - run with the rule and
  with `B70_VERIFY_ATTN=dpas` (the golden prompts are short, so the rule alone may never switch on them), plus 256
  greedy ids at 32k from the long prompt with `--mtp 3` against v2, identical except near-ties (spec 10 A3's form).
- M = 1 never switches: plain decode stays bitwise (spec 28 §8, decision 1).

## 5. Speed bars

As spec 28 §5 (idle box, device 0, interleaved pairs after warm-up, median of 5 / 3).
- **S1:** per shape, the chosen kernel is never slower than v2 beyond noise (the rule's own check).
- **S2:** verify(8) at depth 32768 improves by at least 75 % of P0's attention **excess** - attention at M = 8 less
  attention at M = 1 at the same depth: one row's KV walk is the floor no kernel removes (v2 at M = 1 already streams
  near the device's bandwidth, spec 10 A-F3) -; depth 4096 no worse.

## 6. Stages

- **28c-P0 (inside plan 28a's attribution, no new probe run):** record the attention's share of the verify step at
  M = 2 / 4 / 8 x depth 4096 / 32768 (28a profiles M = 1..8; no list exists at 16), and its excess over M = 1. GO for
  this spec when the attention excess is >= 15 % of verify(8) at depth 32768 after 28b's GEMM (derived from 28a's
  shares on 28b's measured step); else recorded, parked.
- **28c (build):** the kernel, the host reference, the measured table and rule, N1-N3, S1-S2, BENCHMARKS rows.

## 7. Out of scope

Prefill attention (`pf_flash_attn`); the MoE engines' own attention (Ornith / K2 / Kolibri / Bolt) - a follow-on per
engine once 28c's numbers exist; varlen M per request inside one batch (vllm-xpu-kernels 470c19f) - spec 13b's call.

## 8. Amendment - 2026-10-31, against the code (with plan 28c)

What the plan found wrong in the first text, and what replaced it above:
1. Today's verify attention is `attn_v2_variant(M)` (`_KV8` under int8 KV) through `attn_pair`, not
   `attn_v2_rows_variant` - that is spec 13a's `BATCH_ROWS` probe build.
2. M = 2..16 -> 2..8: `attn_v2.cl` stops at M = 8 and so do the verify lists; DPAS at 16 is recorded only. The int8-KV
   decode binaries stop at M = 4, so the int8 table at M = 5..8 uses probe-built controls and production int8 switches
   only at M = 2..4.
3. Depth 131072 -> 130816: the deepest a 131072-row cache holds with 8 rows and the probe's steps (spec 10's depth).
4. `verify_attn_kind(M, depth)` cannot be a capture-time rule in depth (a list replays at every position); §3 now says
   how depth is applied (capture-time by M, or two lists chosen at submit).
5. "`attn_reduce_v2` style reduce": the kernel writes v2's partials in v2's blocking, so the reduce is the existing
   binary; this is also what gives property R.
6. N2's "one-ulp tie rule (spec 13a)" does not exist; N2 is now property R plus A1's bar against M = 1.
7. q is fp32 in decode; a bf16 operand rounds it - the hi + lo split is a lever (§2).
8. S2 and the GO line are measured on the attention excess over M = 1 (the floor no kernel removes); P0 at M = 16
   dropped.
9. `gemv_kind(M)` is spec 28b's and does not exist before 28b is merged; this spec builds on a main with 28b.

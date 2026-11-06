# Spec 28c - the verify step's attention on DPAS, chosen by shape: M rows x depth

**Status (2026-11-03): BUILT on branch `spec28c-verify-attn` (§10; busy-box timing; the operator's rulings of 2026-11-03 in §10).** (2026-10-31: DESIGN.) Follows spec 28 (small-M verify GEMM). Operator, 2026-10-31: "let's
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
| `attn_decode` at M = 4 against M = 1 (v1, before spec 10's v2) | 2.5 -> 9.2 ms | measured 2026-09-27 | docs/probes/probe-mtp-2026-09-27.md |
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
engine once 28c's numbers exist; varlen M per request inside one batch (vllm-xpu-kernels 470c19f) - spec 13c's call.

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

## 9. P0 line - 2026-11-01 (plan 28c Task 0; records only, no new run)

**Busy-box timing (one card), to be re-checked on a quiet box** - every number below is 28a's
(docs/probes/probe-small-m-gemm-2026-11-01.md §2, raw tables in the 28a box log `timing1.log`), measured with the other card
and the CPU on other jobs. Operator ruling 2026-11-01: 28c is built now, in parallel with 28b, so there is no
28b-measured step yet; verify(8) "after 28b" is derived as 28a Task 5 / §4 (b) did.

Attention = `attn_prep` + `attn_decode_v2` + `attn_reduce_v2` over the 16 FA layers. Its time on the unprofiled step
= its profiled share x the unprofiled step (Review Focus 3 of 28a). Excess = attention at M less attention at M = 1,
same depth. All **derived** from measured inputs.

| depth | M | profiled share | attention ms (derived) | excess over M = 1 ms | excess / today's verify(M) |
|---|---|---|---|---|---|
| 4096 | 1 | 0.030 | 0.92 | - | - |
| 4096 | 2 | 0.043 | 1.54 | 0.61 | 1.7 % |
| 4096 | 4 | 0.048 | 2.52 | 1.60 | 3.1 % |
| 4096 | 8 | 0.131 | 12.63 | 11.71 | 12.1 % |
| 32768 | 1 | 0.149 | 5.18 | - | - |
| 32768 | 2 | 0.202 | 8.68 | 3.51 | 8.2 % |
| 32768 | 4 | 0.218 | 13.87 | 8.70 | 13.7 % |
| 32768 | 8 | 0.454 | 67.34 | **62.16** | 41.9 % |

(M = 3 / 5 / 6 / 7 at 32768, same derivation: excess 6.0 / 22.6 / 41.2 / 50.7 ms - the M = 4 -> 5 step doubles
`attn_decode_v2`, 13.3 -> 26.9 ms profiled, and at 4096 the M = 7 -> 8 step does: 6.3 -> 12.1 ms.)

**verify(8) at 32768 after 28b (derived):** 28a's non-GEMV time at M = 8, depth 32768 (73.81 ms, derived) + 28a's
GEMM sum of the bf16 wB stand-in at M = 8 without a‖b (27.46 ms stand-alone, measured; 28.89 ms with 28a's 5.2 %
in-list charge) = **101.3 - 102.7 ms**. The attention excess (62.16 ms) is **60.5 - 61.4 %** of it (on today's step,
148.36 ms: 41.9 %).

**Line: GO** - the excess at M = 8, depth 32768 is >= 15 % of verify(8) there, by 4x. In the operator's summary:
today's verify(8) is 3.17 plain steps at 4096; at M = 8 attention is 12.3 ms (profiled) of the 19.35 ms non-GEMV time
and doubles from M = 7 to 8; at the attention's M = 1..7 slope verify(8) would be ~1.37 plain steps (estimated, 28a
§4). S2's target at 32768 is therefore (v2 - rule) >= 0.75 x 62.16 = **46.6 ms** on the unprofiled verify(8) (to be
re-derived from the quiet-box re-check). Provisional with 28a's line until that re-check.

## 10. 28c as built (2026-11-03, branch `spec28c-verify-attn`)

All timing is **busy box** (card 1 beside spec 28b's G0 / suite on card 0, a desktop session; load 10-30): re-check on a
quiet box. Record: docs/probes/probe-verify-attn-2026-11-03.md.

- **Kernel:** `attn_verify_dpas_M<M>_T32` (bf16, M = 2..8) and `_KV8` (int8, M = 2..4), one source, kept levers
  **KT 32, PACK g, KS 4, DSPLIT 256, QSPLIT 1, 256 GRF** for both forms (fastest passing N1 / N2 at M = 8, 32768:
  8.53x / 4.88x over v2 per layer). It writes v2's partials; v2's own reduce finishes it. The 256-GRF build spills
  (~100-140 registers, IGC's retry) - measured as built.
- **The rule (`src/runtime/verify_attn.h`)**, from the table at 512 / 1448 / 4096 / 11585 / 32768 / 130816 keys:
  bf16 M = 2 from 11585 keys, M = 3 / 4 / 5 from 4096, M = 6 / 7 / 8 always; int8 M = 2 from 11585, M = 3 / 4 from
  4096 (the operator's ruling 1 below). M = 1,
  other heads (Ornith), v1 and the 13a row builds never switch.
- **Depth by two lists** (decision 3): where `dpas_from > 1` the Engine and the PipelineEngine capture the verify list
  at that M twice (the rule's v2 list, a DPAS one) and `verify()` submits the DPAS list from row 0's keys
  `pos + 1 >= dpas_from`. MTP engines: 3 extra lists (M = 2..4; 33 - 42 ms capture at engine start); a DFlash engine
  4 (M = 2..5). The capture's override is `CaptureOptions::force_attn` (tests / probes only) - **not** the plan's
  `B70_VERIFY_ATTN` (operator 2026-11-03: no new environment variables in the engine). Tests take `--vattn
  v2|dpas|rule` (`mtp_gpu_test`, `mtp_verify_test`'s third argument) through `Engine::recapture_verify`.
- **Deviations, each with its cost if wrong:** (1) int8 at 4096: one row 0.99998797 < 0.99999 (`pf_flash_attn_kv8`
  0.99998792 there) - **accepted** (ruling 1 below): int8 M = 3 / 4 from 4096. (2) M = 5 bf16 is not monotone (512 dpas 1.10, 1448 v2 0.80): dpas_from 4096 by the plan's
  rule; a 512-1448 window stays v2 - cost under 0.1 ms a step. (3) One lever line for every M: at 4096, `S2_D128` is
  10-18 % faster at M = 2 / 4 but slower at 32768; not built.

### Gates (card 1, busy box)

- **N1 / N2 on the card** (`attn_verify_dpas_test`, the production binaries + v2's reduce, partials NaN-filled before
  every run): part S (synthetic, 11 positions, M = 2..8 bf16, 2..4 int8) N1 worst 0.9999913 (int8) / 0.9999922
  (bf16), N2a 1000 / 1000 bitwise at every shape, two runs bitwise; part D (real KV) N1 0.9999937 - 0.9999985 except
  int8 at 4096 (0.99998797, recorded: the rule keeps it on v2); `--sweep` every position 0 .. 4088 at M = 2 / 3 / 4 /
  8 against v2's M = 1 step: worst 0.9999890, no (position, head, row) below 0.9999. **PASS.**
- **With the rule** (ctest, the engines' own lists): `mtp_gpu_test` and `mtp_gpu_kv8_test` M3 0 divergences over 39
  prompts x 3 K, RF1 / M5 / M4 as before; `mtp_verify_test` / `_kv8` M2 PASS; `prefix_gpu_mtp_test`,
  `replay_determinism_test` PASS; golden_server_test through `--spec lookup` identical. The golden / A4 prompts sit
  below 4096 keys, where the rule keeps M = 2..4 on v2 - so these runs mostly exercise v2; DPAS is exercised by the
  forced runs and the DFlash tests below.
- **DFlash with the rule** (M = 6..8 bind DPAS at every depth): `dflash_engine_test`, `dflash_gpu_test`,
  `dflash_lookup_gpu_test` **FAIL their bitwise / 0.99999 bars** - D3 rows at M >= 6 logits cosine 0.99991 - 0.99998
  against M = 1, the drafter's ring rows from DPAS rows 0.99989 (bar 0.9999), D2 streams parting from plain at
  near-ties. This is the class spec 28 §10's ruling covers for 28b (the f32 judge; 28b's `tests/server/spec_judge.h`
  re-grades exactly these tests). M <= 4 rows (v2 at ~2000 keys) stay bitwise.
- **Forced DPAS** (`--vattn dpas`: every M = 2..4 list on DPAS at every depth): `mtp_verify_test` M2a logits cosine
  0.99996 - 0.99999 (below the 0.99999 bar), GDN slots >= 0.9999998, M2b 0 of 64 ids differ at every (k, j), RF3-5
  PASS. `mtp_gpu_test` M3: 22 of 39 prompts part from plain (the same position at every K); the plain row's margin
  at the parting step: 16 prompts <= 1 bf16 ulp (near-ties), 4 within 1.2 - 3.4 ulp, **2 beyond 8 ulp: 11.3 ulp at 45
  and 27.1 ulp at 191** - the second is the position 28b measured with gemm_sm (t3_rename-step at 191, 27.1 ulp): a
  position the model amplifies whichever kernel changes the rounding. The f32 judge needs Qwen3.8 f32 references of
  these contexts (f32_refs.sh serves Ornith / K2 only) - **not run; open**.
- **A3 (long32k, 256 greedy ids, `--mtp 3`, `--max-len 65536`):** forced v2 identical to the plain loop; the rule
  (DPAS at 32k for every M) parts from it at id 133 at a **0.15 bf16 ulp** near-tie in the plain row.
- **Not run:** the int8-KV passkey at 120k with `--mtp 3` (hours, opt-in); the two-card PP gates (`pp_mtp_test`:
  the PipelineEngine's deep lists are built but untested on the box); the forced-DPAS int8-head / int8-KV
  `mtp_gpu_test` runs were graded by the first (one-step-late) classifier and not re-run.

### Speed (busy box)

- **S1 holds** at every shape the rule binds DPAS (in-list paired v2 / rule >= 1.006); **S2 met**: verify(8) at 32768
  145.6 -> 91.2 ms (-54.4 ms against the 46.6 ms bar). Verify(M) at 120000 keys: M = 2 60.4 -> 48.9 ms, M = 4 94.6 ->
  67.0, M = 8 297.3 -> 106.5.
- **End to end:** tool-call geomean unchanged (P0 prompts at 2.2-3.0k keys: v2 by the rule); golden at 32768 keys
  +3 - 10 % (K = 3 +9.5 %). Record §4.

### The operator's rulings (2026-11-03, "accept all")

1. **int8 at 4096 keys:** accepted at the prefill kernel's distance. The int8 N1 bar is "no further from the fp64
   reference than `pf_flash_attn_kv8` on the same row" (0.99998792 on the dump's (q head 19, row 0) at 4096), stated
   as such in `attn_verify_dpas_test` part D - 0.99999 is not lowered elsewhere. int8 M = 3 / 4 bind DPAS from 4096.
2. **DFlash M >= 6 non-bitwise, and the forced-DPAS prompts beyond 8 ulp:** accepted under spec 28's ruling for 28b
   (the f32 judge; near-misses like 28b's t5 / t6 accepted as near-ties; contexts without a Qwen3.8 judge stay open
   for the CPU lane). The verify-path tests grade by 28b's `tests/server/spec_judge.h`; their contexts are dumped
   (`B70_DUMP_FORCED`) to `~/b70/logs/spec28c/forced`.
3. **On 28b:** 28c rebased onto main with 28b; `force_attn` joined 28b's `CaptureOptions` (runtime/gemv_kind.h).
4. **M = 6..8 back on v2 (operator 2026-11-04).** The CPU lane's f32 check (tools/oracle/dump_taps.py, top-64 f32
   logits, teacher-forced on each stream) of the one new divergence beyond 8 ulp on main - t6_usage-loader, DFlash
   K = 7, first difference at generated position 103: plain's id is the f32 top on all 9 rows 103..111; the DPAS
   path's is f32 rank 3, **4.0 below the top** (~32 ulp; the top's own margin 3.44). Not a rounding near-tie, so
   `kVerifyAttnTable`'s bf16 M = 6 / 7 / 8 rows are kDpasNever until the kernel is fixed (queued: attn_verify_dpas
   at 8 rows on this prompt). Then every row (MTP M = 2..4, int8 too) went to v2 the same day: the kernel is one code at every M and the cause is unknown. Log: box `~/b70/logs/cpu-1104/verdict.txt`.
5. **Fixed and back on (2026-11-04, branch `fix-attn-verify-dpas`).** The cause was P in bf16 (docs/07 §11): not
   IGC (2.42.6 gave the same token; 128 GRF, `-cl-opt-disable` and the EAGER walk give the same rows). The kernel
   takes a seventh lever, **PSPLIT 1** (P as bf16 hi + lo, two PV DPAS per key-atom pair), on both production lines;
   real-data layers are at v2's level (worst cos 0.9999989 against fp64, v2 0.9999989), DFlash K = 7 with DPAS at
   every M is token-identical to plain on t6_usage-loader, `attn_verify_dpas_test`'s N1 bar is 0.999997 (int8 4096
   now 0.99999994: deviation (1) and ruling 1 are retired), and the re-timed table (record §5) keeps every
   `dpas_from`. Probe: `tools/probe/probe_vattn_ab.cc` (CaptureOptions::attn_dump, test / probe only).

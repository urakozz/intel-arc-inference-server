# Spec 12 - an int8 KV cache

**Status:** design, 2026-09-28, for operator review. Open decisions are marked
**(decide)**.

**Order:** after spec 10 (decode attention v2): the int8 KV decode kernel is v2 with
int8 loads, and its speed is measured against v2, not v1.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

- **Decode at depth reads the KV cache every token.** 64 KiB per position (16 FA layers
  x 4 kv-heads x 256 x 2 B x K and V): 2.1 GB at 32k, 8.57 GB at 128k, against 15.5 GB
  of weights (14.3 with the int8 `lm_head`). Once spec 10's v2 streams KV near device
  bandwidth, KV bytes are the largest term at depth: int8 halves them (derived at
  128k: ~23 GB -> ~18.6 GB per token, about 1.2x decode).
- **Memory.** The KV cache is 8.6 GB at 128k (spec 6 §8: 28.1 of 32.5 GB used). Half of
  it frees ~4.3 GB: room for spec 13's second and third sequences, or MTP's lists at
  every max_len, or 256k context.
- **Spec 7's snapshots** halve too: KV blocks 128 -> 64 MiB, restore at 60k 348 -> ~175 ms
  (derived).

## 2. The decision (proposed)

**int8 K and V with fp16 or fp32 scales, quantised when written (in `attn_prep` and
`pf_attn_prep`), dequantised inside the attention kernels** (decode v2 and
`pf_flash_attn`); attention math stays bf16 / fp32 as today. `--kv-cache bf16|int8`,
default bf16 until the gates pass.

**(decide) Scale granularity**, chosen by P0:

| scheme | K | V | scale bytes per position | notes |
|---|---|---|---:|---|
| per token, per head | 1 scale / 256 | 1 scale / 256 | 32 B (0.05 %) | simplest; K's outlier channels may hurt |
| KIVI-style | per channel over a group of 64 positions | per token | small | K outliers are per channel; needs a group buffer for the last < 64 positions |
| rotated (as spec 5) | Hadamard-rotate K per head before per-token int8 | per token | 32 B | spreads K's outlier channels; q gets the same rotation (it cancels in q·k) |

## 3. Design

- **Layout:** `kv_k`, `kv_v` int8 `[16][max_len][4][256]` plus scale arrays
  `[16][max_len][4]`; the MTP head's KV (spec 8) follows the same form.
- **Writers:** `attn_prep` (decode) and `pf_attn_prep` (prefill) quantise their rows;
  the rotation, if chosen, is applied there.
- **Readers:** decode v2 (spec 10) and `pf_flash_attn` load int8 and scale while building
  the bf16 DPAS operands; the composed prefill path stays bf16-KV only (reference).
- **Spec 7:** `save_kv` / `load_kv` copy the int8 rows and the scales; `kv_bytes()` halves.
- **max_len:** at int8, a 262144 variant becomes possible (derived: 8.6 GB); it is out
  of scope here but the memory report says so.

## 4. Correctness gates

- **Q1, P0 on the CPU:** from oracle dumps (q, K, V per FA layer at depths 2k / 16k /
  32k), attention output cosine per scheme against bf16 KV, and end-to-end logits
  cosine with each scheme simulated.
- **Q2:** golden gates on `l0` and `l0-int8` with int8 KV; the tie rule's allowance
  unchanged.
- **Q3, long context:** `flash_long_test`'s oracle-closeness at 32k (int8 KV no further
  from the oracle than bf16 KV minus a tolerance P0 proposes); passkey 3/3 at 120k.
- **Q4:** tool-call set A4 >= 25/36; the golden-prompt decode of 256 tokens diverging
  only at near-ties.
- **Q5:** determinism and replay bitwise; spec 7's snapshot tests and spec 8's M2 with
  int8 KV.

## 5. Speed bars

- Decode at 32k / 64k / 128k: >= 1.1x / 1.15x / 1.2x v2 with bf16 KV (derived from the
  bytes; recorded if missed).
- Prefill at depth: no regression beyond 3 % (the flash kernel dequantises K and V).
- The memory report line at 131072.

## 6. Stages

- **12a, probe (CPU first):** Q1 for the three schemes; choose the scheme, or stop.
- **12b, build:** writers, readers, flag, spec 7/8 interplay, Q2-Q5, speed, record.

## 7. Out of scope

- int4 KV; fp8 KV (no FP8 hardware on the B70, spec 5).
- 256k context (enabled by this, a separate change).

## 8. Amendment - 2026-10-05: the scheme (12a, Agnes stand-in)

**Operator ruling (2026-10-05): `rotkv`.** K and V are both rotated with the same 256-point
Hadamard (random signs) per kv head and quantised to int8 with one fp16 scale per token per head;
the attention output is un-rotated per head **before** the sigmoid output gate (the gate is
channel-wise, so the inverse cannot be folded into `o_proj`). q gets the same rotation as K, which
cancels in q·k.

Evidence (`docs/probe-int8-kv-2026-09-28.md` on branch `spec12a-int8-kv-probe`, Agnes 3.0 Flash as
the stand-in for Qwen3.8, layer-streamed CPU reference): attention-only relative L2 4.7e-3 against
6.1e-3 (KIVI), 6.8e-3 (rotated K only) and 8.8e-3 (per token), below the reference's own bf16
arithmetic (5.5e-3); flat with depth to 32k; end to end on the golden decision rows 1 - cos
5.3e-5 and KL 1.9e-4, the closest to the fp32-attention control (4.0e-5). Per-token K fails on K's
outlier channels; once K is fixed, per-token V is the largest remaining error, which rotating V
removes.

**Before 12b** (on the box, Qwen3.8 itself): the golden decision-row table for all schemes against
the stop bar (`l0-int8`'s 0.999931742; rotkv was 0.9999467 on Agnes), the long32k capture at 8192
and its replay, the two A4 prompts, and the Q3 tolerances re-derived from those (12a's proposal:
golden mean logit cosine drop <= 1e-4, `flash_long_test` at 32k <= 5e-4, unfiltered KL mean
<= 1e-3, argmax differences only at near-ties).


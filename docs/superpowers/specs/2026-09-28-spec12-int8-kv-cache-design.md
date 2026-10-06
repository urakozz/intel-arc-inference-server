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


## 9. Amendment - 2026-10-05: 12b as built, blind (the Mac; nothing has run on the card)

The operator had 12b written before the Qwen3.8 repeat of 12a (plan 12b Review Focus 0 set
aside); the repeat runs on the Mac CPU (`tools/oracle/kv8_qwen38_repeat.sh`, plan 12b
"Task A") and its numbers replace the PROVISIONAL tolerances below. Branch `spec12b-int8-kv`.

**The flag.** `--kv-cache bf16|int8` on `b70-serve` and `b70-decode`, default **bf16**, which
is today's engine bit for bit: no existing kernel source changed (`tools/kernel_cmdlines`:
the 280 existing command lines identical, 15 added), bf16 allocates and copies exactly what
it did, and every bf16 code path is the old one. `B70_KV_CACHE=bf16|int8` sets the default
(`runtime::default_kv_cache()`, read at each call as `B70_DECODE_ATTN` is), which is how the
gate tests run their own binaries over the int8 cache. int8 is refused, with the reason,
with decode attention v1 (at capture), with the composed prefill path or sycl-tla (at the
first prefill; the CLIs before the load).

**The scheme (§8's `rotkv`), bit-defined.** `src/common/kv8.h` (host) and
`src/kernels/kv8.cl` (device) are one definition:

- R = H_256 diag(s) / 16 with **the probe's own s** (`hadamard(256, 0)`, torch's generator
  seeded 0; `kv8_test` pins the bit string). rotate(x)[j] = s[j] FWHT(x)[j] / 16;
  unrotate(y)[i] = FWHT(s ⊙ y)[i] / 16; the FWHT in ascending stages.
- What is rotated is what the bf16 cache would hold: K = rne_bf16(roped k), V = rne_bf16(v).
  q is rotated from attn_prep's fp32 value (decode keeps fp32; prefill rounds to bf16 as
  `pf_attn_prep_q16` does).
- Per (position, kv head): s16 = fp16_rne(amax / 127) (subnormals kept, not flushed);
  q8 = clamp(rint(y / f16(s16)), ±127) - divided by the ROUNDED scale; deq = q8 · f16(s16),
  exact in fp32.
- Readers: decode v2's kv8 twin dequantises K and V exactly in fp32 (the attention
  arithmetic and every order are v2's); the flash kernel takes int8 as an exact bf16 DPAS
  operand, multiplies each score by its key's K scale, and folds the V scale into P
  (rne_bf16(p · s_v); the row sum stays Σ p). Both un-rotate each head's output in fp32
  before the sigmoid gate (decode: in `attn_reduce_v2_kv8`; prefill: `pf_attn_gate_kv8`).

**Layout (§3, refined).** One K and one V allocation as before: the int8 rows
`[layers][max_len][4][256]` where the bf16 rows were, then every layer's fp16 scales
`[layers][max_len][4]` (`runtime::KvLayout`; the MTP head's own layer the same at
`layers` = 1). 32 KiB + 256 B per position on Qwen3.8 (16 FA layers), 36 KiB + 288 B on
Agnes. Spec 7: `save_kv` / `load_kv` copy the int8 rows and then the scales per tensor
(`kv_bytes` = n · 2 · layers · 4 · 258); the prefix cache's hash chains start from the
engine's KV form, so a bf16 entry can never restore into an int8 engine (Review Focus 2).

**Memory (derived, `memory_plan_test`; 1.5 GB reserve, 32.53 GB card).** `--max-len auto`
with `--kv-cache int8`:

| model / head | no MTP | `--mtp` |
|---|---:|---:|
| Qwen3.8, int8 head | **262144** (bf16 KV 201216) | **262144** (169984) |
| Qwen3.8, bf16 head | **262144** (181760) | **262144** (151808) |
| Agnes, int8 head | **262144** (139520) | **225792** (114176) |

Qwen3.8 at 262144: KV 8.657 GB, total 26.51 GB + reserve (int8 head, no MTP).

**Validated on the Mac (indicative, not the gates).** Host: `kv8_test` (the signs, every
fp16, the rotation against fp64 R, the quantiser's bounds, a synthetic attention where
rotkv beats per token), `memory_plan_test`, `prefix_cache_test`. Device code: every touched
host file syntax-checked against the Level Zero headers; `kv8.cl` clang-checked per family;
the writer (`attn_prep_kv8`, decode M = 2 / QKV_S = 2 and prefill PF = 1) and the gate
(`pf_attn_gate_kv8`) RUN on the Mac's OpenCL GPU (UHD 630) bitwise equal to the host
reference (q, gate, int8 rows, scales; the gate 18432/18432 exact). The decode and flash
readers use Intel sub-group / DPAS built-ins and have not run anywhere.

**Tolerances, from the Qwen3.8 repeat of 12a** (2026-10-06,
`docs/probe-int8-kv-qwen38-2026-10-06.md`; rotkv holds): golden mean logit cos drop ≤ 1e-4
(`kv8_vs_oracle_test`, confirmed); 1 - cos(int8 KV, bf16 KV) ≤ 2e-3 at each of 4k-32k and
≤ 5e-4 averaged over them (`flash_long_kv8_test`; was ≤ 5e-4 per depth, which one flat-logit
row can exceed with no defect); the kernel test's gated flash rows ≥ 0.9999 (still
PROVISIONAL, for the card's first run). The golden and prefill gates keep their tie rule
unchanged (Q2).

**Open for the box** (box-validation-queue row 11): every kernel binary's first ocloc
compile and first run; Q2-Q5; the speed bars (§5) - the readers are written for
correctness first: decode loads V a byte per lane per position and the flash loads K
as 16-byte rows and V a byte per lane per key, where the bf16 kernels use 2D block reads,
so the prefill-at-depth ≤ 3 % bar is at risk and is recorded, not assumed.

# Probe: an int8 KV cache against bf16 KV (spec 12 P0), run 2026-10-04/05 - **Agnes stand-in**

**Every number in this record is measured on Agnes 3.0 Flash, not Qwen3.8**, on the Mac in a
Docker container. The box was unavailable and Qwen3.8's checkpoint is not on the Mac. Agnes
(`urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ`) is the same architecture (Qwen3.5 hybrid).
Its FA layers have identical shapes: 24 q-heads, 4 kv-heads, head_dim 256, partial rotary
0.25. It has **18** FA layers (3, 7, ..., 71 of 72) where Qwen3.8 has 16. KV quantisation
error depends on those layers' K/V statistics, so Agnes is a proxy. It is not the gate. §6
lists what must be repeated on Qwen3.8 on the box.

**Verdict (Agnes stand-in): not a stop. The chosen scheme is the rotated one, extended to V
(`rotkv`).**

- **Per token without rotation is rejected.** K's outlier channels make it 3-4x worse than
  the others on the per-head attention tails.
- **KIVI and rotated K tie end to end,** and both are V-limited. Per-token V is their
  largest error term.
- **Rotating V with the same Hadamard halves that term.** `rotkv` is then the smallest int8
  error on every end-to-end table, and its attention error is below the bf16 eager path's
  own distance from fp64.
- **`rotkv` is outside spec §2's table** (operator **(decide)**). Inside the table, the tie
  between KIVI and rotated K goes to the simpler one: **rotated K, V per token.**

Every number is **measured** unless marked **derived**. This agent ran with the container
capped at **28 GB** (`--memory 28g`) and the model layer-streamed (`stream.py`). It never
instantiated the full model.

## 1. What was run

**The schemes.** All are symmetric: `q = rne(x / s)` clamped to [-127, 127], `s = amax / 127`,
fp32 scales unless named. Dequantised K/V are rounded to bf16, as the engine builds bf16 DPAS
operands. K is quantised **after the k-norm and RoPE** (what the cache stores) and V as
stored. q is never quantised.

| name | K | V |
|---|---|---|
| `pt` | per (position, kv head), over 256 | per (position, kv head) |
| `pt16` | `pt` with fp16 scales (replay only) | same |
| `kivi` | per channel over groups of 64 positions, aligned at 0. A decoder holding L rows has rows >= floor(L/64)*64 in bf16: the partial group is simulated, not padded. | per token |
| `rot` | `x @ R` per kv head, then per token. q gets `q @ R` rounded to bf16. R = H_256 diag(random signs) / 16. | per token |
| `rotkv` (beyond spec) | as `rot` | `v @ R` per token; the attention output is un-rotated (`o @ R^T`, fp32) before the output gate |

The rotation is checked on real rows in fp64. Review Focus 2: max |(qR)·(kR) - q·k| / (|q||k|)
is **4.4e-15** over every FA layer of every capture.

**Attention-only replay** (`kv_int8_probe.py replay`, Task 1). The capture holds q, K, V of the
bf16 forward for all 18 FA layers.

- Per depth bucket: 32 query positions, drawn at random from the last 256 positions before
  the depth.
- Each query sees keys 0..p, with decode semantics for KIVI's tail.
- Attention is computed in **fp64**. The reference is fp64 attention over bf16 K/V. The
  metrics are per (head, query) over the 256-wide output: cosine, max abs error, and
  relative L2.
- **Tiled** depths repeat the 4096-position capture. The query keeps its real local context
  at the end. The earlier copies stand in for distant context, but their RoPE phases are
  those of near keys, so tiling is a softmax-dilution stress, not a real 32k context.
- **Controls:**
  - `ctl_bf16out`: the fp64 reference rounded to bf16.
  - `ctl_eager`: the oracle's own eager math, which has bf16 scores, an fp32 softmax rounded
    to bf16, and a bf16 P·V. This is "how far bf16 itself is from fp32 on the same path".
- **Diagnostics:** `K:x` quantises only K and `V:x` only V. `KV:rot` is the replay's name for
  `rotkv`.

**End to end** (`kv_int8_probe.py run`, Task 2).

- **The forward.** It is one teacher-forced, layer-streamed forward whose **batch dimension
  carries the variants**. `eager_attention_forward` is replaced so that element b quantises
  its own K/V by scheme b. The rest of the model (GDN layers, MLPs, norms) is untouched, so
  each element's error propagates through the whole residual stream, the prefix positions'
  K/V included.
- **The reference element.** Element 0 (`bf16`) is the eager math op for op, chunked over
  queries. The test proves it bitwise against `eager_attention_forward`.
- **The logits.** They are computed as fp32(h) @ fp32(lm_head)^T over `vocab_used` 248089.
  Per position, against the bf16 element: cosine, argmax, unfiltered KL(p_bf16 || p_x) at
  T = 1, and the reference's own logit gap at x's argmax. "Near-tie" means a gap <= 0.05.
- **Controls:**
  - `f32attn`: the attention in fp32 (scores, softmax, P·V) with bf16 K/V. This is the
    end-to-end "bf16 vs fp32 on the same path" control.
  - `ctl_bf16out`: the reference's fp32 logits rounded to bf16, plan 9a's control.
- **Golden prompts.** prose / code / cjk, each with its **32 golden greedy tokens
  teacher-forced** (from `oracle-out-agnes/*.golden.safetensors`). This gives the 33 decision
  rows per prompt in one forward instead of 32 decode steps per variant.
- **Tool-call prompts.** Two A4 prompts: `toolcall-agnes/t1_define-linear_l0` (963 ids) and
  `t2_explain-box` (1023 ids). Their first ~900 ids (system prompt + tool schemas) are
  shared, so their rows 0-511 are the same rows.
- **Long prompt.** `long32k.ids[:4096]`.

**--fp32-matmul.** The Mac's i9-9980HK has AVX2 and no bf16 hardware. Measured on a
[2560 x 5120] x [5120 x 17408] matmul, torch's bf16 matmul runs at **64 GFLOPS** and fp32
sgemm at **346**. So every bf16 matmul (Linear and attention) runs as fp32 sgemm on the same
bf16 values, rounded once to bf16. That is the same math (exact products, fp32 accumulation,
one RNE) in another summation order.

- **The bf16 element against the dumped golden logits** (batch 1, the exact bf16 path):
  - cos min: 0.99979 (prose), 0.99939 (code), 0.99979 (cjk);
  - argmax: **74/74, 93/93, 70/70** equal (the same order).
- **The batch composition also moves fp32 sgemm's blocking.** prose's bf16 element is 0.99984
  against golden at batch 6 and 0.99979 at batch 5. The shared-prefix rows 0-511 of t1 and
  t2 reproduce bitwise at the same batch size, but differ in places between a 5- and a
  6-variant batch (`pt` cos min 0.845 vs 0.892).
- **Per-position minima are therefore fragile.** The means and the KL means are the
  comparison, and every comparison is made within one forward.

**Tokenizer.** Agnes's `tokenizer.json` round-trips `long32k.ids` (32768 ids), the golden ids
and the A4 ids **identically**: decode, then re-encode, gives the same ids.

- The decoded long32k text has **0 combining marks** and **no ids >= 248077**.
- Qwen3.8's tokenizer has the same vocab and merges. It differs only in the regex on
  `\p{M}` and in Agnes's added specials (`docs/probe-agnes-2026-10-03.md`).
- So Qwen3.8 gives the same ids for long32k as Agnes (derived from the above).

**The 8192-id plan was cut to 4096.**

- The first runs measured 3.5 ms per row-layer. The Mac then slowed to 5.6-7.2 ms per
  row-layer.
- At that rate, 8192 ids x 4 variants extrapolated to ~4.7 h, over the brief's 3 h line.
- So the long run is 4096 ids x 5 variants. Its capture comes from the same forward and is
  tiled to 8k / 16k / 32k for the replay. **No end-to-end number exists at 8k.**

## 2. The attention-only replay (Task 1)

`long32k.ids[:4096]` has real depths 1k / 2k / 4k and tiled depths 8k / 16k / 32k. The table
gives the spec schemes, the controls and the decisive diagnostics. `pt16` tracks `pt` on every
row: cos mean within 1e-6, worst head within 1e-4. fp16 scales cost nothing measurable.

| depth | scheme | cos min (layer) | cos p0.1 | cos mean | rel L2 mean | rel L2 max |
|---|---|---:|---:|---:|---:|---:|
| 2048 | pt | 0.9961116 (L35) | 0.9981694 | 0.99994136 | 8.47e-03 | 9.94e-02 |
| 2048 | kivi | 0.9993632 (L43) | 0.9994434 | 0.99997042 | 6.05e-03 | 3.66e-02 |
| 2048 | rot | 0.9991766 (L39) | 0.9994336 | 0.99996607 | 6.66e-03 | 4.65e-02 |
| 2048 | **rotkv** | 0.9994737 (L39) | 0.9998077 | 0.99998655 | **4.60e-03** | 3.50e-02 |
| 2048 | ctl_eager | 0.9993690 (L43) | 0.9997385 | 0.99998297 | 5.29e-03 | 3.77e-02 |
| 2048 | ctl_bf16out | 0.9999968 (L67) | 0.9999975 | 0.99999870 | 1.65e-03 | 2.92e-03 |
| 4096 | pt | 0.9963140 (L31) | 0.9979761 | 0.99993404 | 8.76e-03 | 1.19e-01 |
| 4096 | kivi | 0.9993411 (L35) | 0.9994181 | 0.99996952 | 6.09e-03 | 3.68e-02 |
| 4096 | rot | 0.9989625 (L39) | 0.9994065 | 0.99996439 | 6.76e-03 | 4.66e-02 |
| 4096 | **rotkv** | 0.9993096 (L39) | 0.9997840 | 0.99998590 | **4.68e-03** | 3.92e-02 |
| 4096 | ctl_eager | 0.9989898 (L63) | 0.9996482 | 0.99998095 | 5.51e-03 | 4.63e-02 |
| 8192 tiled | pt | 0.9971953 (L31) | 0.9983071 | 0.99995031 | 7.79e-03 | 8.92e-02 |
| 8192 tiled | kivi | 0.9993465 (L35) | 0.9994635 | 0.99997589 | 5.61e-03 | 3.70e-02 |
| 8192 tiled | rot | 0.9993198 (L39) | 0.9994518 | 0.99997228 | 6.12e-03 | 4.02e-02 |
| 8192 tiled | **rotkv** | 0.9995896 (L43) | 0.9998281 | 0.99998718 | **4.48e-03** | 2.87e-02 |
| 8192 tiled | ctl_eager | 0.9992142 (L43) | 0.9996540 | 0.99998109 | 5.49e-03 | 4.23e-02 |
| 32768 tiled | pt | 0.9965033 (L35) | 0.9983590 | 0.99994813 | 7.98e-03 | 9.47e-02 |
| 32768 tiled | kivi | 0.9993434 (L35) | 0.9994584 | 0.99997365 | 5.89e-03 | 3.69e-02 |
| 32768 tiled | rot | 0.9990474 (L39) | 0.9994314 | 0.99997073 | 6.29e-03 | 5.13e-02 |
| 32768 tiled | **rotkv** | 0.9995028 (L39) | 0.9998307 | 0.99998681 | **4.57e-03** | 4.19e-02 |
| 32768 tiled | ctl_eager | 0.9989627 (L43) | 0.9996200 | 0.99998060 | 5.54e-03 | 4.55e-02 |
| 32768 tiled | ctl_bf16out | 0.9999970 (L19) | 0.9999976 | 0.99999870 | 1.65e-03 | 3.02e-03 |

Where the error comes from, at 32768 tiled (the other depths agree to within the row's
scatter):

| part | cos min | cos mean | rel L2 mean | rel L2 max |
|---|---:|---:|---:|---:|
| K:pt (V bf16) | 0.9966977 | 0.99996943 | 5.72e-03 | 9.17e-02 |
| K:kivi | 0.9998219 | 0.99999481 | 2.72e-03 | 3.06e-02 |
| K:rot | 0.9995064 | 0.99999190 | 3.41e-03 | 4.16e-02 |
| V:pt (K bf16) | 0.9993551 | 0.99997883 | 4.97e-03 | 3.59e-02 |
| V:rot | 0.9999507 | 0.99999491 | 2.73e-03 | 9.95e-03 |

The tool-call captures (t1 963 ids, t2 1023 ids, at their own end) give the same order. At
t2's end:

| depth | kivi | rot | rotkv | pt | ctl_eager |
|---|---:|---:|---:|---:|---:|
| cos min | 0.99938 | 0.99938 | 0.99973 | 0.99572 | 0.99940 |
| rel L2 mean | 6.78e-03 | 7.29e-03 | 4.92e-03 | 9.11e-03 | 5.27e-03 |

Readings:

1. **Depth does not amplify.** The attention error is flat from 1k to 32k (tiled). Every
   scheme's cos mean and rel L2 mean move by less than their scatter. Review Focus 4's worry
   (softmax at depth amplifying small K errors) does not show on Agnes's K. The tiling
   caveat in §1 applies.
2. **`pt` fails on K.** K:pt's worst head is 0.9964-0.9976, and its rel L2 max is
   0.09-0.12, 3-4x the other schemes. These are K's outlier channels. Per channel (KIVI)
   and the Hadamard both fix it.
3. **With K fixed, V per token is the largest term.** `kivi` and `rot` share their worst
   case with V:pt (0.99934-0.99938, the same layer 35), whatever K does. Rotating V removes
   it: V:rot's worst is 0.99995.
4. **`rotkv`'s attention error is below the oracle's own bf16 arithmetic error.** Its
   rel L2 mean is 4.5-4.7e-3, against `ctl_eager`'s 5.3-5.5e-3. `kivi` and `rot` sit just
   above it (5.6-6.8e-3).

## 3. End to end (Task 2)

Golden prompts, decision rows (33 per prompt, 99 pooled):

| variant | cos min | cos mean | 1 - cos mean | argmax diff (near-tie) | KL mean | KL p99 | KL max |
|---|---:|---:|---:|---:|---:|---:|---:|
| pt | 0.9991037 | 0.9999155 | 8.5e-05 | 0 (0) | 2.50e-04 | 1.25e-03 | 1.28e-03 |
| kivi | 0.9994424 | 0.9999308 | 6.9e-05 | 0 (0) | 2.46e-04 | 1.36e-03 | 1.46e-03 |
| rot | 0.9991699 | 0.9999269 | 7.3e-05 | 1 (1) | 2.34e-04 | 9.46e-04 | 1.30e-03 |
| **rotkv** | 0.9995273 | **0.9999467** | **5.3e-05** | 0 (0) | **1.92e-04** | 1.12e-03 | 1.59e-03 |
| f32attn (control) | 0.9997248 | 0.9999599 | 4.0e-05 | 0 (0) | 1.56e-04 | 7.27e-04 | 1.47e-03 |
| ctl_bf16out (control) | 0.9999986 | 0.9999986 | 1.4e-06 | 0 (0) | 2.32e-04 | 1.35e-03 | 1.47e-03 |

`long32k.ids[:4096]`, every position:

| rows | variant | cos min | cos mean | argmax diff (near-tie) | KL mean | KL max |
|---|---|---:|---:|---:|---:|---:|
| 512-2047 | pt | 0.9048416 | 0.9990354 | 15 (9) | 1.25e-03 | 2.02e-01 |
| 512-2047 | kivi | 0.8899878 | 0.9992472 | 16 (12) | 8.07e-04 | 5.86e-02 |
| 512-2047 | rot | 0.9025964 | 0.9992501 | 15 (11) | 9.03e-04 | 9.12e-02 |
| 512-2047 | **rotkv** | 0.9480504 | **0.9994174** | 14 (10) | 8.15e-04 | 1.11e-01 |
| 512-2047 | ctl_bf16out | 0.9999986 | 0.9999986 | 15 (13) | 2.09e-04 | 1.35e-03 |
| 2048-4095 | pt | 0.9443733 | 0.9996478 | 17 (15) | 7.76e-04 | 7.48e-02 |
| 2048-4095 | kivi | 0.9434597 | 0.9996110 | 24 (16) | 6.74e-04 | 4.74e-02 |
| 2048-4095 | rot | 0.9357923 | 0.9996682 | 24 (18) | 6.65e-04 | 3.39e-02 |
| 2048-4095 | **rotkv** | 0.9697259 | **0.9996853** | 19 (16) | **6.08e-04** | 2.26e-02 |
| 2048-4095 | ctl_bf16out | 0.9999986 | 0.9999986 | 31 (20) | 2.45e-04 | 1.52e-03 |
| last row | pt / kivi / rot / rotkv | 0.99984 / 0.99984 / 0.99988 / 0.99987 | | 0 | 7.3 / 7.8 / 7.9 / 4.0 e-04 | |

A4 tool-call t1 (963 ids, all six variants in one forward):

| rows | variant | cos mean | argmax diff (near-tie) | KL mean | KL max |
|---|---|---:|---:|---:|---:|
| 0-511 | pt / kivi / rot / rotkv | 0.99694 / 0.99756 / 0.99650 / 0.99706 | 15 / 11 / 15 / 12 | 2.6 / 1.5 / 2.5 / 2.4 e-02 | 3.4 / 1.5 / 4.3 / 4.6 |
| 0-511 | **f32attn** | 0.99719 | 14 | 2.8e-02 | 6.6 |
| 512-962 | pt / kivi / rot / rotkv | 0.99872 / 0.99878 / 0.99867 / 0.99893 | 10 / 12 / 11 / 14 | 5.5 / 6.3 / 9.0 / 6.4 e-03 | 0.27 / 0.22 / 1.08 / 0.51 |
| 512-962 | **f32attn** | 0.99895 | 11 | 6.0e-03 | 0.25 |

t2 (1023 ids, 5 variants) agrees: rows 512-1022 have KL mean 5.8 / 5.1 / 5.7e-3 (pt / kivi /
rot) against f32attn's 5.5e-3.

Readings:

1. **int8 KV moves the logits about as far as fp32 attention math does.** On the tool-call
   prompts, every scheme sits inside the `f32attn` control's envelope. That holds for the cos
   mean, the argmax differences and the KL mean, and it includes the chaotic rows 0-511 of
   the system prompt, where even `f32attn` reaches cos 0.71 and KL 6.6. Agnes's logits are
   that sensitive to any perturbation of the attention arithmetic. The per-position minima
   say more about that sensitivity than about the schemes.
2. **On the golden decision rows the ranking is clean.** By 1 - cos mean: rotkv 5.3e-5,
   kivi 6.9e-5, rot 7.3e-5, pt 8.5e-5, against the f32attn control's 4.0e-5. By KL mean:
   rotkv 1.9e-4, against 2.3-2.5e-4 for the rest.
3. **At depth (2k-4k), rotkv has the best mean cosine and KL mean.** Most argmax differences
   are near-ties: 16 of 19. bf16 output rounding alone produces 31 differences in the same
   2048 rows, because Agnes's top logits tie often at bf16 resolution.

## 4. The choice, and the stop rule

**The stop rule** (plan 12a Task 2 Step 2): stop if every scheme's logits cosine "at 8k" is
below the `l0-int8` path's own distance to the oracle. That distance is **0.999931742**, the
mean logit cosine over the golden set's 96 decision rows on Qwen3.8 (BENCHMARKS,
`flash_vs_oracle_test`).

- **The like-for-like Agnes number is the golden decision rows:** rotkv 0.9999467, kivi
  0.9999308 (tied), rot 0.9999269, pt 0.9999155. One scheme is above the line and one is
  level with it, so **this is not a stop.**
- No 8k end-to-end number exists (see §1).
- The 2k-4k bucket has no `l0-int8` counterpart. Its 1 - cos means of 3.1-3.9e-4 are the
  same order as `l0-int8` flash against composed at 32k (1 - 0.999662 = 3.4e-4).
- The replay shows no growth in attention error from 4k to 32k.
- **The rule must be evaluated literally on Qwen3.8 (§6).**

**The choice.** The plan's rule is the smallest error, with ties going to the simplest:

- **`rotkv`** has the smallest error on every end-to-end table and in the replay.
  - Cost against `rot`: V gets the same 256-point Hadamard in `attn_prep` / `pf_attn_prep`,
    and the attention output is un-rotated per head (an inverse FWHT on the 256-wide row).
    The un-rotation goes in the decode v2 / `pf_flash_attn` epilogue, **before** the sigmoid
    output gate.
  - It cannot be folded into `o_proj`, because the gate multiplies channel-wise in the
    un-rotated basis.
- **Within spec §2's table,** KIVI and rotated K tie end to end (6.9 vs 7.3e-5 golden; 0.99961
  vs 0.99967 cos mean at 2k-4k; KL 6.7e-4 both). KIVI is slightly better on K alone in the
  replay. The tie goes to **rotated K, the simpler one:**
  - every row is written and read the same way;
  - there is no 64-row bf16 group buffer and no prefill/decode asymmetry;
  - spec 7's snapshots copy rows + scales as they are;
  - the cost is a 256-point Hadamard on q and K in `attn_prep`.
- **Proposed: rotated K and V (`rotkv`), with fp32 or fp16 scales** (`pt16` tracks `pt`).
  It is a superset of the spec's rotated row. **(decide)**: if the operator keeps to spec
  §2's table, the pick is rotated K with V per token.
- Not measured, and possibly better still: KIVI K + rotated V. It has the best K term and the
  best V term, but it carries KIVI's group buffer.

## 5. Proposed tolerances for spec 12's Q3 (Agnes-derived, to be re-derived on Qwen3.8)

These are calibrated on `rotkv`. The bf16-KV engine's distance to the oracle is the
baseline, so each tolerance is a **drop** allowed when the KV goes int8.

| gate | measured (rotkv, Agnes) | proposed tolerance |
|---|---|---|
| golden set, mean logit cos vs oracle, int8 KV vs bf16 KV | int8-vs-bf16 1 - cos 5.3e-5 (pt 8.5e-5) | drop <= **1.0e-4** |
| `flash_long_test` 32k, logit cos vs oracle (last row / decision rows) | 1 - cos 3.1e-4 mean at 2k-4k, 1.3e-4 last row; replay flat 4k -> 32k | drop <= **5e-4** |
| unfiltered KL at T = 1, mean over the gate's rows | 1.9e-4 golden, 6.1e-4 at 2k-4k | <= **1e-3** |
| argmax | 0 golden; at depth 19/2048, 16 of them near-ties (bf16 rounding alone: 31) | near-tie rule unchanged; non-near-tie differences <= 0.5 % of rows |

Passkey 3/3 at 120k and Q4 (A4 >= 25/36) stay as written. The tool-call e2e above suggests
A4 will see int8 KV as noise of the f32attn size, but the A4 count itself is a box
measurement.

## 6. What must be repeated on Qwen3.8 on the box

1. **The end-to-end golden table (§3, first table)** on the three golden prompts and their
   32 golden tokens, all six variants (`kv_int8_probe.py run ... --cont oracle-out/<p>.golden.safetensors`).
   - This is the literal stop-rule check against `l0-int8`'s 0.999931742, on the same
     model.
   - `--fp32-matmul` should pay there too. The T5810's Xeon is AVX2 without bf16
     (derived, not measured). Within-run comparisons do not need the reference element to
     be bitwise the oracle.
2. **The long capture and replay** on `long32k.ids`.
   - Capture at **8192** ids (the box has the RAM; batch 1 is the "day-class" 11359 s to
     8192 ids per BENCHMARKS, so variants x 8192 is too long). At least 4096 x the variants
     end to end.
   - Then the replay at 2k / 4k / 8k real and 16k / 32k tiled. This confirms that K's
     outlier channels (pt) and V's per-token term behave as on Agnes.
3. **The two A4 tool-call prompts end to end** with Qwen3.8's A4 ids (`tests/golden/toolcall/`).
4. **§5's tolerances,** re-derived from those numbers.
5. Then 12b's own gates Q2-Q5 on the engine. Nothing in this record replaces them.

## 7. Timings and resources (Mac, i9-9980HK, Docker container capped at 28 GB, 16 threads)

| run | rows (ids x variants) | forward | wall | peak RSS |
|---|---:|---:|---:|---:|
| prose / code / cjk (74 / 93 / 70 ids x 6) | 444 / 558 / 420 | 588 / 529 / 302 s | 638 / 573 / 339 s | 24.2-24.7 GiB |
| t1 963 x 5 | 4815 | 1332 s | 1466 s | 24.2 GiB |
| t2 1023 x 5 | 5115 | 2625 s (Mac slowed) | 2896 s | 24.5 GiB |
| t1 963 x 6 | 5778 | 3382 s | 3670 s | 24.3 GiB |
| long32k 4096 x 5 + capture | 20480 | 9176 s | 9952 s (logits metrics 740 s) | 25.4 GiB |
| replay t1 / t2 (4 threads) | | | 65 s each | |
| replay long4k, 6 depths, 16 comparisons (6 threads, beside a run) | | | 4778 s | |

Peak RSS includes the mmapped shards (~18 GB checkpoint) and the 5 GB fp32 copy of
`lm_head` for the metrics. The captures are 1.21 GB (4096) and 0.28 / 0.30 GB (t1 / t2). All
outputs are in `oracle-out-12a/` (git-ignored).

## 8. Reproduce

```bash
# in the oracle container (the Mac path: tools/oracle/README.md), from the repo root
python3 tools/oracle/test_kv_int8_probe.py                     # seconds, no weights
S=$(ls -d /hf/hub/models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ/snapshots/*/)
python3 tools/oracle/kv_int8_probe.py run "$S" --fp32-matmul --ids tests/golden/prompts/prose.ids \
  --cont /golden-agnes/prose.golden.safetensors --out oracle-out-12a/prose6.pt
python3 tools/oracle/kv_int8_probe.py run "$S" --fp32-matmul --ids tests/golden/prompts/long32k.ids \
  --n 4096 --variants bf16,pt,kivi,rot,rotkv --capture oracle-out-12a/long4k.cap.safetensors \
  --out oracle-out-12a/long4k.pt
python3 tools/oracle/kv_int8_probe.py replay oracle-out-12a/long4k.cap.safetensors \
  --depths 1024,2048,4096 --tiles 8192,16384,32768
python3 tools/oracle/kv_int8_probe.py summary oracle-out-12a/*.pt
python3 tools/oracle/kv_int8_probe.py tokcheck "$S" tests/golden/prompts/long32k.ids
```

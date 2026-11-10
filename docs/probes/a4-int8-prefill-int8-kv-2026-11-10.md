# A4 `t3_rename-linear_l0`: `uint3_t` under l0-int8 prefill x int8 KV (open-work 3o, 2026-11-10)

Box card 0, driver intel-ocloc / intel-opencl-icd / libze-intel-gpu1 26.40.40196.15, libze1 1.34.0 (the upgrade at
19:57; the tree rebuilt clean after it). Every number below was made on that driver; the rows made before it were
bitwise the same (the logits of row 76 identical to the last digit), so nothing here depends on the upgrade.

The decision row is row 76 of `t3_rename-linear_l0` (prompt 2737 ids + 76 forced ids of the bf16 reference): the
reference writes `2` (id 17) after `uint3`, the failing run `_t` (516). Int8 lm head throughout (A4's).

## The row under each configuration, against the f32 references

| run | logit 17 - logit 516 |
|---|---:|
| f32, the engine's int4 weights exact (row_taps_ref.py --dtype f32 --exact-dequant) | +1.85 (14.8 bf16 ulp) |
| f32, the oracle's bf16-rounded dequant | +1.67 |
| l0 + bf16 KV | +1.48 |
| l0 + int8 KV | +2.95 |
| l0-int8 + bf16 KV | +1.14 |
| **l0-int8 + int8 KV** | **-1.34** (the wrong digit) |

Every configuration but the combined one is within 1.1 of the f32 value; the combined one is 3.2 off. Its per-layer
residual at the row leaves f32 like the others up to layer ~30 and then grows superadditively (relative L2 at layer
48: 0.064 l0 + bf16, 0.093 l0 + int8, 0.101 l0-int8 + bf16, 0.301 combined).

## Where the error enters: the prompt's state, written by the prefill

`tools/probe/probe_kv_state` saves the prompt's KV and GDN state after prefill, converts the KV between the forms on
the host (bf16 -> int8: `common::kv8::encode`, the writer's chain; int8 -> bf16: dequantise, un-rotate, round) and
restores it before the 76 decode steps.

| prompt state from | decoded with | 17 - 516 |
|---|---|---:|
| l0-int8 + int8 KV prefill (restored as is) | int8 KV | -1.34 (= the prefilled run, bitwise) |
| l0-int8 + bf16 KV prefill, KV quantised on the host | int8 KV | **+1.03** |
| l0-int8 + int8 KV prefill, KV dequantised on the host | bf16 KV | **-0.39** |
| l0 + bf16 KV prefill, KV quantised on the host | int8 KV | +2.00 |
| l0 + int8 KV prefill, KV dequantised on the host | bf16 KV | +2.84 |

So the decode reading int8 rows is not it (quantising a good prompt for decode keeps the right digit), and the
prompt state of the combined prefill is wrong even when decoded in bf16: the drift is made inside the prefill, where
each FA layer's flash attention reads the int8 K / V the chunk just wrote (the 3e widening: raw codes, scales applied
in the kernel) - the same path under `l0`.

## The writer is exact; the int8 prefill amplifies the int8 KV's perturbation

Relative L2 of the prompt's K over all 2737 positions, int8-KV prefill (dequantised) against bf16-KV prefill of the
same backend, and the two backends against each other (bf16 KV):

| FA layer (model layer) | l0-int8: int8 vs bf16 KV | l0: int8 vs bf16 KV | l0-int8 vs l0 |
|---|---:|---:|---:|
| 0 (3) | 0.0070 | 0.0070 | 0.032 |
| 1 (7) | 0.0184 | 0.0084 | 0.029 |
| 3 (15) | 0.0215 | 0.0099 | 0.026 |
| 7 (31) | 0.0323 | 0.0141 | 0.035 |
| 11 (47) | 0.0847 | 0.0426 | 0.103 |
| 15 (63) | 0.1338 | 0.0829 | 0.161 |

- FA layer 0 (nothing upstream reads the cache): the combined run's int8 rows and scales are byte-identical to
  `kv8::encode` of the l0-int8 + bf16 run's rows - the writer, the rotation and the scales are exact; the 0.0070 is
  the quantisation itself, the same for both backends.
- From the next FA layer on, the same int8-KV perturbation moves l0-int8's K 1.6-2.6x as far as l0's. Between them sit
  l0-int8's int8 linears: each re-quantises its input per token (rotation, per-token int8 codes), a step function -
  a perturbation smaller than the code step re-draws the rounding of the codes near a boundary, so it comes out
  roughly doubled (a perturbation of ~0.2 code steps flips ~20 % of the codes by a whole step). The two noise
  sources do not add; the int8 linears amplify the int8 cache's.
- Chunk boundaries: the second chunk's (2048..2736) error is larger than the first's at every layer (FA 7: 0.043 vs
  0.028) - it attends over more int8 rows - but it grows smoothly, no step at the boundary.

## A4 on the new driver (36 scenarios, 384 ids, score.py --min 25 with the validity check)

| prefill + KV | matches | invalid calls | bar |
|---|---:|---:|---|
| l0 + int8 KV | 28 / 36 | 0 | PASS |
| l0-int8 + bf16 KV | 30 / 36 | 0 | PASS |
| l0-int8 + int8 KV | 29 / 36 | 1 (t3_rename-linear_l0, `uint3_t`) | FAIL |

## Verdict

Not a defect of one kernel: quantisation drift - the int8 cache's error inside the prefill's attention, amplified by
the int8 prefill linears - landing on a copy token whose f32 margin is 15 bf16 ulp. A fix would have the prefill's
attention read bf16 K / V (the second row of the restore table: +1.03): a bf16 shadow of the prompt's K / V during
prefill, 64 KiB a position (2 GiB at 32k, 16 GiB at 262k - the memory the int8 cache exists to save).

# T6 - direct M=2048 widened-small-kernel measurements

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held).

`tools/probe/probe_prefill_small` compiles probe-only M=2048 variants from the
unchanged production `src/kernels/prep.cl` source. No runtime variant helper
names these binaries. Inputs are finite, incompressible random data; timing is
the Level Zero kernel timestamp median of eight replays after dropping the
first three (measured, iterate grade).

## One harness defect, one fix, one measurement

The first run returned `ZE_RESULT_ERROR_DEVICE_LOST` at the fence and produced
no timing. Its cause was concrete: the probe allocated `gdn_o` as bf16, while
`prep_gated_head` declares it `const float*` (source-verified, iterate grade).
The probe allocation and random fill were changed to fp32; the following table
is the one post-fix measurement, not a retry of the invalid run.

| kernel / M=2048 variant | grid work-groups | us/launch | calls/chunk | ms/chunk | grade |
|---|---:|---:|---:|---:|---|
| `prep_res_fold`, SP4 | 20 × 2048 = 40960 | 381.354 | 129 | 49.195 | measured launch time; derived chunk time, iterate |
| `prep_norm_finish` | 20 × 2048 = 40960 | 133.333 | 129 | 17.200 | measured launch time; derived chunk time, iterate |
| `prep_silu_mul` | 5 × 2048 = 10240 | 4392.396 | 64 | 281.113 | measured launch time; derived chunk time, iterate |
| `prep_gated_head` | 48 × 2048 = 98304 | 278.750 | 48 | 13.380 | measured launch time; derived chunk time, iterate |
| **four directly measured terms** | - | - | - | **360.888** | derived sum of measured terms, iterate |

`prep_res_fold` uses the measured SP4 row for all 129 calls, a conservative
derived price because only the initial no-previous-partials call is SP0. The
four rows remove 2419.669 ms from the former literal-decode upper bound
(derived from its 2532.002 ms upper endpoint and the 112.333 ms residual below).

`attn_prep` and `embed_gather` remain unmeasured at M=2048. Their retained
literal-decode upper bound is 112.333 ms (derived from 3.2 µs × 2048 × 16 plus
3.65 µs × 2048); their lower bound remains 0 ms (derived). The small-kernel
term is consequently tightened from 0.000-2532.002 ms to
360.888-473.221 ms (mixed: direct measurements plus the stated derived
residual range, iterate grade).

# Probe: where a full Qwen3.8 prefill chunk's time goes (open-work 3k) - 2026-11-10

**Question.** Is a new prefill multiply kernel worth a spec? Split a 2048-row chunk and a cold 15,257-id prompt
into the linear multiplies, the int4 conversion, attention, GDN (the chunked linear-attention recurrence of the 48
GDN layers) and the rest, on both prefill backends - `l0` (int4 -> bf16 slab, bf16 GEMM) and `l0-int8` (int4 ->
rotated int8 slab, int8 x int8 GEMM, spec 5) - and set the multiply's achieved rate against the card's measured
DPAS peak (DPAS: the matrix-engine instruction).

**Rule, fixed in the row before measuring:** GO for a spec if the multiply is the largest share and below ~70 % of
the measured peak; else a recorded no-go with the split.

**Verdict: GO, narrowly, for the `l0-int8` multiply only** (`pf_gemm_i8`). On `l0-int8` the multiply is the largest
share (44 % of a 2048-row chunk, 38 % of the 15k prompt) and runs at **67.6 % / 67.1 %** of the int8 peak. On `l0`
the multiply is the largest share too (61 % / 54 %) but already runs at **77.6 % / 75.7 %** of the bf16 peak: a
no-go for a new bf16 kernel. The payoff of the int8 spec is bounded: at 85 % of peak the multiply saves ~80 ms of a
914 ms chunk (~9 %). The proposed item is in open-work (Now, after 3l), plan only.

Every number is **measured** unless marked **derived**.

## 1. Method

One Arc Pro B70 (device 0), the whole-box lock (`tools/gpu_lock.sh box`) held for the whole run, 22:58-23:07 on
2026-11-09 (box clock). No CPU reference job ran; load average ~1.4-2.9 (a desktop indexer and `hw-smi`). One
script, `tools/probe/prefill_split_3k.sh`, logs in `~/b70/logs/3k/`:

1. **DPAS peak**: `probe_dpas_rate --sweep` (the 2026-09-22 probe: register-resident chains of 64 independent
   `matrix_mad`, assembly-checked) before and after the run.
2. **The linears alone, 3i's distinct-layer walk**: a new probe, `tools/probe/probe_prefill_split.cc` - production
   kernels only (`pf_dequant_slab` + `gemm_l0`; `pf_quant_had` + `pf_requant_rot` + `pf_gemm_i8`, each layer's own
   int8 column scales by `pf_colmax_rot` as `Int8State` builds them), one live allocation of independent synthetic
   weights per layer of each of the five int4 linear families, visited in ascending order, one GPU span per whole
   family walk, a warm-up pass, 7 rounds with the arms rotating, medians. Arms: each backend's whole walk and each of
   its kernels alone. Sanity before timing: layer 0's int8 output against bf16's of the same linear, relative RMS
   0.013-0.019 on all five shapes (an int8 arm computing nothing would fail it).
3. **The engine**: `b70-decode --bench --prefill-length 2048 | 15257 --prefill-backend l0 | l0-int8 --kv-cache bf16
   --max-len 32768 --tg 16`, real weights (`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`), one warm-up per backend,
   then 3 interleaved rounds unprofiled (the walls), then 2 interleaved rounds with `B70_PREFILL_PROFILE=1` (per-phase
   L0 kernel timestamps; the two rounds agree within 1 % on every phase above 5 ms, the tables use their mean).

**FLOP (derived).** One prompt row through the int4 linears is 2 x (gate||up 5120 x 34816 x 64 + down 17408 x 5120 x 64
+ GDN qkv||z 5120 x 16384 x 48 + out / o_proj 6144 x 5120 x 64 + FA q||k||v 5120 x 14336 x 16) = **48.65 GFLOP**:
**99.64 TFLOP** a 2048-row chunk, **742.3 TFLOP** for 15,257 ids.

**One accounting correction.** On `l0-int8` the profile's `norm` row also carries the one-time column-scale pass
(`pf_colmax_rot`, 256 launches, ~69-77 ms) that `prepare_prefill` runs at setup, outside the timed window: its
launches are timestamped before `profile_reset` and collected by the first phase wait, which is a norm. The
tables give `l0-int8` the norm time without it (the same kernels as `l0`'s norm row; the call's wall does not
contain it).

## 2. The DPAS peaks

| type | before the run | after the run | 2026-09-22 |
|---|---:|---:|---:|
| bf16 (`bf16_bf16_k16`) | **183.45 TFLOP/s** | 183.35 | 183.45 |
| int8 (`i8_i8_k32`) | **366.90 TIOP/s** | 366.90 | 366.90 |

Unchanged since September, and the same after 9 minutes of prefill (no thermal drift in the peak).

## 3. The bar: l0 vs l0-int8, unprofiled walls

| prompt | `l0` (3 rounds) | `l0-int8` (3 rounds) | `l0` median | `l0-int8` median | int8 gain |
|---|---|---|---:|---:|---:|
| 2048 ids (one chunk) | 1185.4 / 1190.2 / 1189.7 ms | 912.1 / 913.9 / 913.7 ms | **1189.7 ms, 1,721 t/s** | **913.7 ms, 2,241 t/s** | 1.302x |
| 15,257 ids (cold, 7 chunks + 921 rows) | 10111.4 / 10120.1 / 10126.6 ms | 8012.8 / 7992.8 / 8008.6 ms | **10120.1 ms, 1,508 t/s** | **8008.6 ms, 1,905 t/s** | 1.264x |

The earlier 2,247 vs 1,727 t/s at 2048 reproduces (2,241 vs 1,721). Spread within a backend: 0.4 % or less.

## 4. The split (engine, real weights, L0 kernel time, ms)

Groups: **multiply** = `slab_gemm` / `i8_gemm`; **int4 conversion** = `slab_dequant` / `i8_requant` + `i8_quant` (the
per-slab int4 -> int8 rotated requant and the per-linear activation quantiser); **attention** = `attn_prep` +
`attn_flash` + `attn_gate` (16 FA layers); **GDN** = `ab_proj` + the ten `gdn_*` kernels (48 GDN layers);
**rest** = `norm` + `head`.

### 2048-row chunk

| group | `l0` ms | share | `l0-int8` ms | share |
|---|---:|---:|---:|---:|
| **multiply** | **700.3** | **60.8 %** | **401.6** | **44.1 %** |
| int4 conversion | 125.8 | 10.9 % | 189.7 (requant 161.0 + quant 28.7) | 20.8 % |
| attention | 33.2 | 2.9 % | 33.0 | 3.6 % |
| GDN | 237.0 (scan 63.3, wu 42.3, solve 37.4, ab_proj 23.4) | 20.6 % | 231.7 | 25.4 % |
| norms / rest | 54.9 | 4.8 % | 55.0 | 6.0 % |
| **kernel sum** | **1151.2** | | **911.0** | |
| unprofiled wall | 1189.7 | | 913.7 | |

### Cold 15,257-id prompt

| group | `l0` ms | share | `l0-int8` ms | share |
|---|---:|---:|---:|---:|
| **multiply** | **5344.3** | **54.3 %** | **3017.1** | **38.1 %** |
| int4 conversion | 992.1 | 10.1 % | 1494.2 (requant 1279.4 + quant 214.8) | 18.9 % |
| attention | 1321.1 (flash 1235.4) | 13.4 % | 1259.8 (flash 1175.5) | 15.9 % |
| GDN | 1793.6 (scan 477.2, wu 317.8, solve 283.8) | 18.2 % | 1751.9 | 22.1 % |
| norms / rest | 399.2 | 4.1 % | 399.2 | 5.0 % |
| **kernel sum** | **9850.2** | | **7922.0** | |
| unprofiled wall | 10120.1 | | 8008.6 | |

Attention and GDN run the same kernels on both backends; their 2-5 % differences between the two columns are
run-to-run (flash attention 1235 vs 1176 ms on identical work), not the backend.

### The multiply's achieved rate

| | `l0` (bf16 GEMM) | of 183.45 | `l0-int8` (i8 GEMM) | of 366.90 |
|---|---:|---:|---:|---:|
| 2048-row chunk, engine | **142.3 TFLOP/s** | **77.6 %** | **248.1 TIOP/s** | **67.6 %** |
| 15,257 ids, engine | 138.9 | 75.7 % | 246.0 | 67.1 % |
| 2048 rows, probe walk (synthetic) | 127.2 | 69.3 % | 232.6 | 63.4 % |
| the whole linear (multiply + conversion), engine 2048 | 120.6 | 65.8 % | 168.5 (bf16-equivalent FLOP) | - |

The 15k prompt's rate is slightly lower: its last 921-row chunk runs at ~111 / 217 (probe), below the 2048 rate.

## 5. The probe walk (3i's protocol, synthetic distinct-layer weights)

Sum over the five families of whole-walk GPU medians, ms (7 rounds):

| arm | 2048 rows | 921 rows |
|---|---:|---:|
| `l0` walk | 915.5 | 525.4 |
| `l0` dequant alone | 116.9 | 116.9 |
| `l0` GEMM alone | 783.2 | 405.1 |
| `l0-int8` walk | 658.0 | 424.1 |
| `l0-int8` quantiser alone | 29.8 | 14.8 |
| `l0-int8` requant alone | 168.9 | 165.4 |
| `l0-int8` GEMM alone | 428.3 | 206.8 |

Per family at 2048 rows, GEMM alone: bf16 123-133 TFLOP/s, int8 226-242 TIOP/s (gate||up lowest on both).

The probe walk is ~10 % slower than the engine's own linears (bf16: 915.5 vs 826.1 ms; int8: 658.0 vs 591.3 ms) with
the same kernels and shapes. The difference is the data, not the walk: uniformly random int4 bits and normal
activations toggle the DPAS array harder than the real weights and activations do (higher power at the same work);
the engine also leaves small gaps between kernels (profiled run). The engine's rate is the in-situ one the rule is
read on; the probe confirms the ranking and the conversion costs.

## 6. Verdict

- **`l0`: no-go for a new bf16 multiply.** Largest share (61 % / 54 %), but at 77.6 % / 75.7 % of the measured bf16
  peak - above the ~70 % bar. A perfect kernel at 100 % would save ~160 ms of 1190; nothing near that is realistic.
  The 2026-09-22 fused-dequant rejection stands (its 156 TFLOP/s "ceiling" was the in-situ GEMM rate with the
  dequant free; the GEMM alone now measures 142 in the engine).
- **`l0-int8`: GO, narrowly.** The multiply is the largest share (44.1 % / 38.1 %) and runs at 67.6 % / 67.1 % of the
  measured int8 peak - below ~70 %. It is the smaller tile (`pf_gemm_i8`: 256 x 128 per work-group, 32 x 32 per
  sub-group, the probe_w8a8 gsweep tile) against bf16's 256 x 256. Bound (derived): at 85 % of peak (312 TIOP/s)
  the chunk's multiply is 319 ms, **-82 ms of 914 (~9 %, ~2,460 t/s at 2048)**; at 75 % (bf16's level), -40 ms (~4 %).
- **What the rule does not cover, recorded for whoever picks next:** on `l0-int8` the int4 -> int8 conversion is the
  second cost (161 ms requant + 29 ms quantiser a chunk, 21 %), fixed per chunk per weight (the requant rotates every
  weight every chunk; the 4096-row chunk cap halves it per id), and GDN is 21-25 % of the chunk on both backends
  (`gdn_scan` alone 63 ms). Attention is small at 15k (13-16 %) and grows with context.

## 7. Files

- `tools/probe/probe_prefill_split.cc` - the distinct-layer linear split on both backends (new; probe only).
- `tools/probe/prefill_split_3k.sh` - the whole run under one box lock.
- Logs: `~/b70/logs/3k/` (`run.log`, `split.log`, `dpas-pre.log` / `dpas-post.log`, `plain-*`, `profile-*`).

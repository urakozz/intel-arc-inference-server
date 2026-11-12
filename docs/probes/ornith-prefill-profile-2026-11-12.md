# Ornith 1.5 prefill profile, 2026-11-12

Where the prefill time of Ornith 1.5 35B-A3B (urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ) goes on one B70.
Test run, no code changed. Build at main 238f0582, `B70_PREFILL_PROFILE=1 b70-decode --bench --prefill-length N
--tg 32 --max-len 32768 --parallel 1 --prefill-backend l0|l0-int8`, under `tools/gpu_lock.sh card 0` (a first set
ran on card 1, which is ~3.4% slower on prefill; it was dropped, the numbers below are card 0 only).
Two rounds per row; the rounds agree within 1%. Logs: `~/b70/logs/ornith-prefill/` on the box.
The profiler adds a host wait per phase, so the t/s below are ~4% under the README's plain runs (3362 / 3937).
Percentages are of the sum of the kernel (L0 gpu) times, round 2.

Prefill tokens/s (instrumented): 4k l0 3187 / 3226, 4k l0-int8 3707 / 3717; 16k l0 3137 / 3132, 16k l0-int8 3475 / 3479.
`--prefill-chunk 4096` is refused for this model ("pf_moe bakes KC = 2048 into its SLM sort staging"), so there is no
4096-chunk row. As a substitute, `--prefill-chunk 1024` at 4k, l0: 2276 t/s (see below).

## Phase table (ms and % of kernel time)

| phase | 4k l0 | % | 4k int8 | % | 16k l0 | % | 16k int8 | % |
|---|---|---|---|---|---|---|---|---|
| MoE expert weights to the GEMM form (dequant/requant, all 257 blocks) | 390.7 | 32.9 | 331.3 | 30.2 | 1421.7 | 28.7 | 1281.8 | 28.5 |
| MoE grouped GEMMs (gate-up+SiLU, down) | 277.0 | 23.3 | 200.2 | 18.2 | 1049.8 | 21.2 | 764.5 | 17.0 |
| MoE sort (1 work-group) | 122.9 | 10.3 | 120.5 | 11.0 | 493.3 | 10.0 | 484.8 | 10.8 |
| MoE route (router GEMV + moe_route) | 35.6 | 3.0 | 34.8 | 3.2 | 143.1 | 2.9 | 139.7 | 3.1 |
| MoE gather + combine | 28.2 | 2.4 | 21.1 | 1.9 | 112.5 | 2.3 | 85.0 | 1.9 |
| dense linears (dequant or requant + GEMM, + quantiser) | 88.7 | 7.5 | 72.3 | 6.6 | 354.5 | 7.2 | 289.5 | 6.4 |
| GDN (ab_proj, conv, l2norm, gate, A, solve, wu, A2, scan, head) | 155.8 | 13.1 | 154.4 | 14.1 | 623.4 | 12.6 | 618.6 | 13.8 |
| attention (prep, flash, gate) | 46.9 | 3.9 | 46.9 | 4.3 | 589.7 | 11.9 | 590.2 | 13.1 |
| norm | 41.3 | 3.5 | 114.3 | 10.4 | 166.0 | 3.3 | 234.9 | 5.2 |
| head | 0.9 | 0.1 | 0.9 | 0.1 | 0.9 | 0 | 0.9 | 0 |
| total | 1187.9 | | 1097.1 | | 4954.8 | | 4490.0 | |

The whole MoE FFN is 72% of the l0 time and 66% of the int8 time (4k). The int8 "norm" row is inflated against l0
(114 vs 41 ms, 120 more launches): something extra is timed in that bucket in int8 mode; its wait column shows 39.6 ms,
so read it with care.
GDN sub-phases at 4k l0 (ms): wu 31.9, solve 26.4, scan 26.4, conv 16.3, head 12.6, A2 11.5, ab_proj 10.9, A 10.8,
l2norm 8.4.

Chunk size (4k, l0): chunk 1024 gives MoE weights 705 ms (vs 391 at 2048) and MoE GEMM 419 ms (vs 277), total 1799 ms
vs 1270 ms: the weight pass costs the same per chunk whatever the chunk holds (~4.9 ms a layer, 196 ms a chunk), so
fewer, bigger chunks pay it fewer times. A 4096 chunk would save about another 196 ms at 4k (est. ~3800 t/s) but the
sort's 2048 limit forbids it today.

## Efficiency by phase

Peaks: 183 TFLOP/s bf16, 367 int8, ~590 GB/s memory.

- MoE grouped GEMM, 4k: useful FLOPs = 4096 tokens x 9 experts (8 routed + shared) x 3 x 2048 x 512 x 2 x 40 layers
  = 9.27 TFLOP (rows padded to tiles of 32 add roughly 20% more work done than useful). l0: 277 ms = 33.5 TFLOP/s,
  18% of the bf16 peak. 16k: 37.1 TFLOP in 1050 ms = 35.3 TFLOP/s (19%). int8 mode, 4k: 200 ms for the same FLOPs
  = 46 TFLOP/s blended; the gate-up half is int8 and, if down keeps its 33 TFLOP/s bf16 rate (~92 ms), gate-up runs
  ~108 ms = ~57 TFLOP/s, 16% of the int8 peak.
- MoE weight pass: each layer and chunk turns all 257 expert blocks (808M weights) from int4 into bf16 (reads 0.4 GB,
  writes 1.6 GB): 4.9 ms = ~410 GB/s, 70% of the measured bandwidth. It is a bandwidth-bound pass over experts that
  get about 64 tokens each per chunk, so it does not shrink with the tokens. int8 mode (gate-up requant to int8,
  down still bf16): 4.15 ms, ~360 GB/s.
- Dense linears (1.28 G int4 weights, 10.5 TFLOP at 4k): l0 76 ms GEMM = 138 TFLOP/s, 75% of the bf16 peak (plus
  12.7 ms dequant); int8 mode 50.5 ms = 208 TFLOP/s, 57% of the int8 peak. These are fine; the MoE GEMMs are the
  weak ones.
- Attention (flash, 10 layers, causal, 16 heads x 256): 4k 1.37 TFLOP in 38 ms = 36 TFLOP/s (20%); 16k 22 TFLOP in
  554 ms = 40 TFLOP/s (22%).
- GDN: no FLOP count taken; it is 13% of the time and scales linearly with tokens (4x tokens, 4x time). The
  delta-rule solve, wu and scan are the largest pieces.

## MoE prefill today (src/runtime/prefill/moe.cc, src/kernels/prefill/pf_moe*.cl)

All 40 layers have the MoE FFN; 30 layers are gated delta net and 10 are full attention (every 4th). 256 experts,
top-8, hidden 2048, expert width 512, one shared expert (the 257th weight block). Chunks of 2048 tokens (a hard cap).
Per layer and chunk (l0: 11 launches; l0-int8: 10):
1. Router: `pf_ab_proj` GEMV (router || shared gate) over the chunk, then decode's `moe_route` (grid 1 x C).
2. `pf_moe_sort`: ONE work-group of 256 lanes (lane = expert): each lane scans all C x 8 expert ids serially, makes
   expert-major rows padded to 32-row tiles, a tile table padded to a fixed maximum (no host readback), and the shared
   expert's tiles after them.
3. Gather: copy the normed activations into sorted order (int8: the Hadamard-rotated, quantised rows).
4. Weights: the whole expert array of the layer is converted EVERY chunk (a separate pass writing a full bf16 copy
   into a scratch region): bf16 gate-up in 2 batches and down in 1 batch (3 dequant+GEMM pairs); int8 mode requantises
   all gate-up blocks into one int8 VNNI array (1 launch), then one GEMM.
5. GEMM: `pf_moe_gemm`, one launch per batch covers every expert: a work-group = 32 rows x 256 columns of one tile,
   DPAS bf16 (4 sub-groups) or int8 (8 sub-groups), SiLU fused for gate-up. Down is bf16 on both backends (K = 512
   is not whole 1024-k rotation blocks). Padding tiles return at once.
6. `pf_moe_combine`: weighted sum of the 8 slots in fixed order + shared expert by its gate + residual fold.

## The 3 biggest time sinks

1. The expert-weight pass, 29-33% of the prefill (391 ms of 1188 at 4k): all 257 experts are converted to bf16/int8
   in a separate memory-bound pass every chunk (~410 GB/s), at the same cost for any chunk size (the 1024-chunk run
   doubles it). Fix directions: dequantise inside the GEMM from int4 (no 1.6 GB round trip), and/or fewer, bigger
   chunks (needs the sort to stop being capped at 2048).
2. The MoE GEMMs, 17-23% (277 ms at 4k l0) at only ~18% of the bf16 peak (~16% of the int8 peak for gate-up): a
   32-row tile with ~64 tokens per expert per chunk, so each B tile is reloaded for few rows; padding adds ~20% work.
3. The sort, ~10% (123 ms at 4k, 493 ms at 16k, both backends): a single work-group of 256 lanes, every lane
   re-scanning all C x 8 ids, 1.5 ms per layer and chunk for what is a histogram and a scan. Pure latency; a
   multi-work-group histogram/scan should remove most of it. Next after these: GDN (13%) and, at 16k, attention
   (12-13%, 20% of peak).

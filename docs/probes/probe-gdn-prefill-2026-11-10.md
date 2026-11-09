# Probe: GDN prefill per kernel on Qwen3.8 (open-work 3m) - 2026-11-10

**Question.** 3k ([record](probe-prefill-chunk-split-2026-11-10.md)) found the 48 GDN layers' chunk recurrence
(GDN: Gated DeltaNet, the model's linear-attention mixer) to be the second share of a prefill chunk on the default
`l0` backend: 237 ms of a 1,190 ms 2048-row chunk, 1.8 s of a cold 15,257-id prompt. Which of its eleven kernels
(the `a||b` projection and `gdn_chunk`'s ten launches) is worth a build item?

**Rule, fixed in the row and written here before measuring.** GO for a build item only for a kernel that is
**>= 5 % of the chunk's time** and **below ~60 % of its bound**. The bound is bandwidth or compute, whichever binds
for that kernel: its bytes and FLOPs per launch are computed from the shapes, the time each would take at the
card's measured peak (590 GB/s; 183.45 TFLOP/s bf16 DPAS for a DPAS kernel, the measured fp32 vector rate for a
vector kernel) is computed, and the larger of the two is the bound. Each GO carries an estimated saving. An exact
(bitwise) change comes first; a reassociation only with the f32 judge (the engine judged against an f32 reference,
memory ruling 2026-10-23).

**GDN's numerics are not on the table.** The operator's ruling of 2026-10-31 is that `pf_gdn_wu` keeps its four
intermediates Q1-Q4 in fp32 (accuracy over FLA parity); that change is parked on branch `g0-fix-gdn-fp32` (main
still stores `w` / `u` bf16). Nothing proposed here may add a rounding, and anything proposed must not stand in
the way of the parked fp32 `w` / `u`.

Every number is **measured** unless marked **derived**.

## 1. Method

**Locks.** Building is CPU only (`tools/box.sh build`, no GPU). The smoke checks (the new probe's checksum, one
profiled engine run) ran under `tools/gpu_lock.sh card <N> -- ...` (one card; other agents may hold the other).
The final timed numbers ran under `tools/gpu_lock.sh box -- tools/probe/gdn_prefill_3m.sh`: both cards idle,
nothing else on the GPU, the arms interleaved after a warm-up. `ZE_AFFINITY_MASK` was never set by hand.

**The engine, real weights.** `b70-decode --bench --prefill-length 2048 | 15257 --prefill-backend l0
--kv-cache bf16 --max-len 32768 --tg 16 --parallel 1` on `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`.
`B70_PREFILL_PROFILE=1` gives every kernel of the walk its own L0 GPU timestamp bucket: `ab_proj` and the ten
`gdn_*` launches are separate rows, summed over the 48 GDN layers, so the per-kernel split is the engine's own,
over all 48 layers' real tensors (distinct layers by construction). One warm-up, then three rounds of four arms
(2048 and 15,257 ids, each plain and profiled), the arm order rotating each round. The plain runs give the walls;
the profiled runs give the per-kernel times (median of three).

**The peaks.** `probe_dpas_rate --sweep` (183.45 TFLOP/s bf16 since 2026-09-22) and a new
`tools/probe/probe_fma_rate` (the sustained fp32 vector `fma` rate: 16 independent register chains per work-item,
probe_dpas_rate's calibration and replay protocol), before the run and the fma rate again after it.

**Bytes and FLOPs (derived)** per launch from the shapes in the kernels (`src/kernels/prefill/pf_gdn_conv.cl`,
`pf_gdn_wy.cl`, `pf_gdn_scan.cl`, `pf_gated_head.cl`, `pf_gemv_bf16.cl`) and the scratch layout
(`src/runtime/buffer_sizes.cc`). Bytes are the minimum DRAM traffic: every distinct tensor read once and written
once (re-reads that hit the 24 MB L2 are not counted), so an achieved GB/s is a lower bound on the kernel's real
traffic rate. FLOPs count each `fma` as 2.

## 2. Results

(filled in after the run)

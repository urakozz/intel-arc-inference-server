# P5 - Intel CuTe chunked gated delta rule vs our state (spec 2 §4)

grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held)

sycl-tla: `91e5bd735517d8e79591b41e0d0cd37a7bacdca7` content-verified by
`include/cute/util/compat/launch_policy.hpp` SHA-256
`5460b1e5119308896ab65ca99127179c2b47c2ca8173db4349b938d680fea661`.

## Pin and source-line audit - first action under ruling A11

The sycl-tla pin was re-verified before this pre-registration. P5's plan has no
separate sycl-tla file:line assertion beyond the CuTe/CUTLASS headers it
consumes; `cute/tensor.hpp` SHA-256 was recorded as
`21ce110e2a7c030a7fdfb3540d7c4ff69adfc441e93e986eba2b554738d0f8c0` at this
pin.

The plan's `~/PycharmProjects/vllm-xpu-kernels` path does not exist on the
box. The read-only checkout is `~/vllm-xpu-kernels`, at
`5085fddea4350d16426300f1b7467e24bf5e3f83`, not the plan's
`a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`. This source drift is recorded
before build: any result names the actual revision and is not silently
attributed to the older one.

## Layout map (source-derived before measurement)

| their contract | our layout | consequence |
|---|---|---|
| q/k `[T][16][128]`, v/out `[T][48][128]` | fused qkvz rows, q/k/v at 0/2048/4096 in `[C][16384]` | gather or copy into their contiguous stride |
| b/a fp32 head-major `[48][T]`; a mutated | ab partials token-major `[C][128]` | transpose and re-materialise a before every replay |
| state `[batch][48][128 v][128 k]` | state `[48][128 k][128 v]` | transpose each 128x128 fp32 head in and out |
| fixed chunk 64 | C is a multiple of 64 for timing | preserve 64-token chunks |
| fp32 state, fp32 accumulation | fp32 state | dtype match |
| caller-owned A/w/u scratch | no pre-existing equivalent | allocate and record extra scratch |
| five queue submissions with no explicit dependencies | Context's queue is in-order | valid only on that queue |

The source audit also confirms the BMG branch, the raw-pointer launcher, the
chunk-prepare in-place a mutation, and the unchecked head-dimension multiples
of 64 noted in plan 6a. The probe will assert 128/128 and 48 % 16 == 0.

## Pre-registered predictions (written 2026-09-05, BEFORE any P5 measurement)

1. The header builds against the actual box checkout once the sole host
   predicate it needs is supplied and caller-owned scratch is allocated
   (medium confidence).
2. State is transposed between the implementation's V-major and our K-major
   layouts. A device transpose round trip is bit-exact.
3. After 4096 positions, max relative state difference versus
   `gdn_ref::step` is <= `5e-2`, mean <= `5e-3`; the growth at 64, 256,
   1024, and 4096 positions is recorded.
4. Delta-rule time per layer scales approximately with C and remains under 2%
   of the corrected P2 GEMM chunk at C=1024/2048/4096.
5. The 151 MB all-layer state relayout is not throughput dominant (derived
   traffic estimate: about 0.5 ms at 590 GB/s).

If the header needs torch, cannot compile against the pinned headers, or cannot
be adapted with O(state) relayout, P5 records the first diagnostic and the
priced own implementation as its complete deliverable.


## 2026-09-05 result - blocked by an unconditional PyTorch header

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held). The
compile-only target was attempted through `tools/box.sh build` against the
actual box checkout `5085fddea4350d16426300f1b7467e24bf5e3f83`.

The first diagnostic was:

```text
/home/user/vllm-xpu-kernels/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp:5:10:
fatal error: 'torch/all.h' file not found
#include <torch/all.h>
         ^~~~~~~~~~~~~
```

This is Step 2 branch (b). The header's raw-pointer
`gdn::kernel_launcher` is not separable from its torch-facing entry point in
this revision: `torch/all.h` is unconditionally included before
`gemm.hpp`, the raw launcher, or `csrc/utils.h`. Per the planned retry, a
local, include-path-precedent `csrc/utils.h` shim supplying only
`vllm::xpu::is_bmg()` was added. The second build produced the same first
`torch/all.h` diagnostic. No PyTorch installation, checkout mutation, or
header-copying workaround was attempted.

The optional `probe_gdn_chunk` target is therefore gated by
`B70_P5_CUTE_GDN=OFF` by default. Enabling it reproduces the diagnostic;
leaving it off preserves the ordinary prefill build and test suite. This is a
blocked P5 deliverable, not a throughput or numerics measurement.

## Priced own implementation - estimate for the ruling request

All costs in this section are estimated or derived, iterate grade; none is a
P5 measurement.

| term at C=4096 | price | grade |
|---|---:|---|
| batched depthwise 4-tap conv, 48 layers | 8.06 GB / 590 GB/s = 13.7 ms | derived lower bound |
| one 64x128x128 bf16 GEMM per head/chunk across 48 layers | 6.442 GFLOP / 164.21 TFLOP/s = 0.039 ms | derived lower bound using corrected P2 peak cell |
| three such GEMM-shaped passes (A/W/U lower-bound proxy) | 0.118 ms | derived estimate |
| chunk-to-chunk state scan, one layer | 64 x 3.15 MB / 590 GB/s = 0.341 ms | derived lower bound |
| chunk-to-chunk state scan, 48 layers | 16.4 ms | derived lower bound |
| state transpose at one chunk boundary, all layers | 151 MB / 590 GB/s = 0.256 ms | derived lower bound |

The intra-chunk price is explicitly a lower bound: the source's inverse and
triangular work is not a count of exactly three GEMMs, and its vector-fp32
component must be measured in an own kernel. The table is sufficient to show
the design shape, not to promise a replacement rate.

An own implementation needs: (1) batched causal depthwise conv seeded from the
ring's final three values and writing the last three values back; (2) a
64-token fp32 GDN block using cumulative gates, K·K^T, triangular solve, and
W/U recompute; (3) a sequential 64-step chunk scan over the fp32
`[48][128][128]` state; and (4) O(state) K-major/V-major transposes at the
boundary. It remains a correctness problem first: the required acceptance
measurement is the `gdn_ref::step` 4096-position state/output band. Estimate:
8-12 engineer-days before that gate, then a separate performance measurement.


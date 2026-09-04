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


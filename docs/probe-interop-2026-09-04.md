# P1 - SYCL↔Level Zero interop smoke (spec 2 §4)

grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)

All pre-registered predictions and estimated/derived values below are
iterate-grade. No measurement has been made for this record.

## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

| quantity | prediction | basis |
|---|---|---|
| USM round-trip | a `zeMemAllocDevice` buffer written by a SYCL kernel and read by an OpenCL C kernel on the L0 immediate list holds the SYCL kernel's values, **bit-exact** | the L0 backend extension: a pointer from `zeMemAllocDevice` on the SAME `ze_context` is a valid interop USM pointer (`sycl_ext_oneapi_backend_level_zero.md`, "Construct a SYCL object from a Level-Zero handle") |
| `sycl::get_pointer_type` on that pointer | **unknown** - recorded, not asserted | the extension does not promise the runtime tracks a foreign allocation; the probe reports what this driver says |
| L0 immediate-list per-launch overhead (`noop`) | ≤ 10 µs at N = 1, ≤ 2 µs amortised at N = 1024 | `probe_replay`, docs/07 #5: 9.9 µs for a 1-launch regular-list replay, 0.52 µs/kernel at N = 700 |
| SYCL in-order queue per-launch overhead (empty `single_task`) | ≤ 10 µs at N = 1, ≤ 2 µs at N = 1024 | same class of work; **no published figure for this queue on this box - this is the first** |
| cross-queue handoff, both directions | **≤ 20 µs** | spec §4's P1 row |

## The budget this has to fit inside (derived, iterate-grade, and it is tight)

A prefill chunk alternates our OpenCL C kernels and `sycl-tla` GEMMs about
**6 times per layer** (norm → GEMM → activation → GEMM → attention/GDN →
GEMM), so **≈ 384 handoffs per chunk over 64 layers**. Spec §3.6 requires the
interop cost under **1 %** of chunk time. Against a C = 2048 chunk of ~1.0 s
(half of vLLM's 2.076 s for pp4096, `explorer-2 §4`), 1 % is 10 ms - i.e.
**26 µs per handoff**. The 20 µs prediction is therefore the design's headroom
and not a comfortable margin. **If the measurement lands above 26 µs the
finding is that the first cut must batch our OpenCL C kernels between GEMMs
(fewer, larger handoffs), and P1 records that arithmetic explicitly rather
than only a pass/fail.**

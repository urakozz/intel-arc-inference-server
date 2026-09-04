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

## Build inputs and interop spelling

The controller-provisioned box copy is `/home/user/sycl-tla`, is read-only,
and intentionally has no `.git`. The Mac checkout reports
`91e5bd735517d8e79591b41e0d0cd37a7bacdca7`; the box copy's
`include/cute/util/compat/launch_policy.hpp` SHA-256 is
`5460b1e5119308896ab65ca99127179c2b47c2ca8173db4349b938d680fea661`, equal
to that pinned Mac checkout. Its `include/cutlass/version.h` reports CUTLASS
4.2.1. This is a content verification, not a Git query against the box copy.

The installed oneAPI 2026.1 headers define the interop inputs below; this is
the declaration order used by `src/sycl/context.cc`:

```cpp
template <> struct BackendInput<backend::ext_oneapi_level_zero, context> {
  struct type {
    interop<backend::ext_oneapi_level_zero, context>::type NativeHandle;
    std::vector<device> DeviceList;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <> struct BackendInput<backend::ext_oneapi_level_zero, device> {
  using type = ze_device_handle_t;
};

enum class ownership { transfer, keep };
```

## Measured (2026-09-04; iterate-grade)

Conditions: `ZE_AFFINITY_MASK=1`; L0 device `Intel(R) Arc(TM) Pro B70
Graphics`, driver `17012946`; `renderD` holder count **8 (measured,
iterate-grade)**. The probe used the four IGC variables printed below. All
timing rows are **measured, iterate-grade**, with one discarded warm-up and
eight replays, dropping the first three and taking the median. `libb70_prefill.so`
was **30,440 bytes (measured, iterate-grade)**. The first green
`context_test` build took **4.3 s (measured, iterate-grade)** on the cached
tree.

```text
sycl-tla sha 91e5bd735517d8e79591b41e0d0cd37a7bacdca7-content-verified (pin 91e5bd735517d8e79591b41e0d0cd37a7bacdca7)
IGC env: SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file | IGC_VISAOptions=-perfmodel | IGC_VectorAliasBBThreshold=10000 | IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread
L0 device: Intel(R) Arc(TM) Pro B70 Graphics (driver 17012946)
# P1 results (iterate-grade; 8 replays, discard first 3, median)

## USM round-trip
| check | result | detail |
|---|---|---|
| SYCL -> L0 -> host | pass | bit-exact, sum 36028797010575360 |
| sycl::get_pointer_type | device | foreign zeMemAllocDevice |
| sycl::get_pointer_device | Intel(R) Arc(TM) Pro B70 Graphics | foreign zeMemAllocDevice |
| L0 -> SYCL -> host | pass | ctrl_read value 1 |

## Per-launch overhead
| queue | N | us/launch (8 replays, drop 3, median; iterate-grade) |
|---|---:|---:|
| L0 immediate list | 1 | 8.319 |
| L0 immediate list | 64 | 2.244 |
| L0 immediate list | 1024 | 2.149 |
| SYCL in-order queue | 1 | 9.877 |
| SYCL in-order queue | 64 | 4.622 |
| SYCL in-order queue | 1024 | 4.468 |

## Cross-queue handoff
| ordering | us/handoff (8 replays, drop 3, median; iterate-grade) | result |
|---|---:|---|
| SYCL -> wait -> L0 -> wait | 8.569 | pass |
| L0 event -> SYCL | 14.797 | pass |

| budget quantity | value (derived, iterate-grade) |
|---|---:|
| handoffs per chunk | 384 |
| wait handoffs at C=2048 | 3.290 ms (0.329% of 1.0 s) |
| wait handoffs at C=4096 | 3.290 ms (0.150% of 2.2 s) |
If a measurement lands above 26 us/handoff, the first cut must batch OpenCL C kernels between GEMMs (fewer, larger handoffs).
```

### Prediction versus measurement (iterate-grade)

| quantity | result | hit / miss |
|---|---|---|
| USM round-trip | bit-exact in both directions | hit |
| `sycl::get_pointer_type` | `device`; this was record-only, not an asserted value | hit (record-only) |
| L0 immediate list | 8.319 µs at N=1; 2.149 µs at N=1024 | hit / miss |
| SYCL in-order queue | 9.877 µs at N=1; 4.468 µs at N=1024 | hit / miss |
| cross-queue handoff | 8.569 µs wait path; 14.797 µs L0-event path | hit |

The derived **3.290 ms (iterate-grade)** wait-path cost for 384 handoffs is
0.329% of the 1.0 s C=2048 budget and 0.150% of the 2.2 s C=4096 budget. It
is below the 1% design limit; no batching ruling is requested.

## Decode-isolation verification (iterate-grade)

`B70_PREFILL=OFF` configured and built on the box, then passed **35/35
(measured, iterate-grade)** tests with the five checkpoint-labelled tests
excluded. The normal `B70_PREFILL=AUTO` build enabled the icpx component and
passed **41/41 (measured, iterate-grade)** tests, including `context_test` and
all five checkpoint-labelled tests. Both suites ran with `ZE_AFFINITY_MASK=1`.

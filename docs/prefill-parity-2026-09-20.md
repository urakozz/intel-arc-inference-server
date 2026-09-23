# Prefill parity: where the time actually went (2026-09-20)

A matched, controlled comparison of this engine's prefill against vLLM's on the
same box, same checkpoint, same prompt ids and same chunk width, plus a GPU
trace of both sides so the remaining gap could be attributed to kernels rather
than guessed at.

**These numbers are the state of play on 2026-09-20.** The engine has moved
since: the current standing rows are in [BENCHMARKS.md](BENCHMARKS.md), and
prefill is now ahead of the vLLM row measured here. What stands unchanged is
the attribution: this document is why the work that followed went where it did.

Every number is labelled measured or derived.

## The matched comparison

| path | median latency | prefill t/s |
|---|---:|---:|
| this engine, immediate | 2748.157 ms | 1490.45 |
| this engine, recorded chunks | 2734.409 ms | 1497.95 |
| vLLM, same checkpoint, ids and chunk | **2544.043 ms** | **1610.04** |

Three warm requests per path after one first-use request. GPU work serialized,
no compiler running during the comparison. All paths returned first token
**383**; vLLM reported zero cached prompt tokens.

The warm gap was **204.1 ms** (immediate) or 190.4 ms (recorded), not the
roughly 657 ms that an older, uncontrolled 1973 t/s vLLM figure implied. That
older figure is retained elsewhere as a separate external result, but its exact
vLLM and oneDNN revisions, device and GDN dtype were never fully recorded, so
it does not explain anything causally and 1610.04 is the number this comparison
uses.

Configuration, both sides:

- Checkpoint `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, snapshot
  `84575a18f209992ef96d819b31f924b489e3d55d`.
- 4096 ids made by repeating the `prose` golden prompt; uint32-LE SHA256
  `342eace92c4214eadd1ccfda421d3a576ae753f9be4ee61abbca167316237e67`.
- Chunk and `max_num_batched_tokens` 2048, max length 16384, bf16 activations
  and KV, fp32 GDN state, one sequence, TP1/PP1, no prefix caching, no
  speculation.
- vLLM `0.29.1rc1.dev380+g23e26e058.d20260919.xpu`, torch
  `2.15.0.dev20260913+xpu`, kernels `0.1.15.dev22+g3d74ec9.d20260918`,
  oneDNN `3.13.0`.
- Our scope: the entire `Engine::prefill()` call through the first id landing
  in shared control memory, reset outside timing. vLLM's scope:
  `LLM.generate()` through one returned id, including scheduling and sampling,
  excluding HTTP, tokenization and model loading.

## The linear path, itemised

The single `linear_l0` phase was split into `slab_dequant` and `slab_gemm`, one
wait per slab, so the dequant round trip could be attributed rather than
estimated. One instrumented run, same checkpoint, 4096 ids:

| phase | L0 GPU ms | launches |
|---|---:|---:|
| `slab_dequant` | **274.5** | 7,616 |
| `slab_gemm` | **1344.3** | 7,616 |
| sum | 1618.8 | 15,232 |

The sum reproduces the unsplit `linear_l0` row (1635.8 ms) within the
run-to-run spread of an instrumented process, so this is an attribution of the
same work rather than a change to it. Closing a phase per slab costs 15,232
extra waits, so the instrumented wall inflates accordingly and is not a
throughput figure.

**Derived floor for the GEMM half.** Total linear FLOPs per request are
2 x M x SUM(K.N) over 64 layers = 199 TFLOP (derived); at the 156.53 TFLOP/s
this GEMM measures on its own that is **1278 ms**. The measured 1344.3 ms is
about 5% above its own floor, so the matrix math is close to the rate already
demonstrated.

**What this makes the dequant round trip worth.** The slab pass writes about
48.7 GB of bf16 weights per chunk and the GEMM reads them back (derived from
SUM(K.N) = 24.3 G weights); a fused path would read about 12.2 GB of int4
instead. The measured 274.5 ms is therefore the **ceiling** on a dequant-fusion
win, less whatever an in-GEMM dequant costs. It is the largest single attributed
block in the walk that is not matrix multiplication.

That ceiling was tested directly two days later and the fusion lost: the dequant
pass already runs near memory bandwidth, so there is very little to reclaim and
the in-mainloop ALU costs several times more than the traffic it removes. See
[probe-fused-dequant-2026-09-22.md](probe-fused-dequant-2026-09-22.md).

## Where vLLM spends its GPU time

A separate trace of one uncached 4096-id request, same container, same card.
Wall time 2567.373 ms. Kernel events counted once: the profiler's operator rows
and graph annotations contain the same kernels and must not be added again.

| disjoint GPU kernel family | kernel time | launches |
|---|---:|---:|
| GEMM (int4 linears plus AB and head GEMMs) | 2130.553 ms | 610 |
| GDN kernels, including conv and update | 177.270 ms | 672 |
| flash attention | 63.091 ms | 64 |
| Triton pointwise and reduction | 125.253 ms | 706 |
| other, including fills and cache operations | 17.860 ms | 973 |
| **total** | **2514.027 ms** | 3025 |

Kernel-interval union 2514.002 ms; first kernel to last kernel 2537.572 ms.
Only **23.569 ms** of that span is outside kernel execution. The trace contains
38 host-device copies totalling **14,056 bytes and 0.070 ms**, so there is no
evidence of large intermediate tensors shuttling through the CPU between torch,
Triton and oneDNN.

The int4 operator alone accounts for 2116.9 ms across 512 calls. GDN's core
prepare, A, inverse, WU and forward kernels total 141.833 ms across 96
layer-chunk instances, of which the forward and state scan is 47.570 ms.

## Side by side, and why the shorter path can still lose

Different instrumentation on the two sides and different operator boundaries,
so this is a bottleneck comparison, not an end-to-end subtraction:

| work | this engine | vLLM |
|---|---:|---:|
| int4 linear path | 1620.1 ms | 2116.9 ms |
| GDN core: gate, A, solve, WU, A2, scan | 527.6 ms | 141.8 ms |
| attention QK, softmax, PV against flash attention | 148.2 ms | 63.1 ms |
| SiLU and multiply | 98.7 ms | 51.3 ms |

The GDN scan alone was 304.7 ms against the vendor forward kernel's 47.6 ms.
Our residual and norm work is another 104.2 ms; vLLM fuses several such
operations, so there is no single matching trace row.

As an approximate partition: our L0 kernel sum was 2651.4 ms against vLLM's
2514.027 ms. **Our int4 linears are about 497 ms faster, and everything else is
about 634 ms slower** (1031.3 against 397.1 ms), for a net deficit of roughly
137 ms of GPU time. That is how the path with fewer layers of software still
loses, and it is what pointed the next round of work at the GDN scan, at
attention and at pointwise fusion rather than at the GEMM.

Every one of those three landed. The scan went to a split-BF16 DPAS kernel
worth 2.33x on its own row, the attention QK kernel stopped computing the
masked half, and SiLU folded into the gate‖up epilogue. Those three changes are
the difference between the 1490 t/s here and the current 1670.72.

## Profiling repair

`B70_PREFILL_PROFILE=1` reports L0 GPU timestamps, host launch submission and
residual host waits separately. It no longer labels the instrumented call's
wall time as an uninstrumented run, nor derives an instrumentation overhead
from that comparison. Timestamp collection is outside the wait timer, SYCL work
is explicitly outside L0 timestamp coverage, and the PV boundary closes each
attention group so PV is not attributed to the following QK.

An instrumented run on the series card measured 2761.1 ms wall, 2651.4 ms L0
GPU sum, 67.5 ms host submission, 17,383 launches, with QK and PV each
accounting for exactly 128 launches.

## Vendor probes: what passed and what did not

### Intel's chunked GDN kernel

Three real probe defects were found and repaired first: out-of-order dependent
submissions, a missing sigmoid on beta, and reuse of the gate input after the
vendor prepare kernel had overwritten it. Raw a and b now use the CPU
reference's bf16 rounding points, Q and K are normalized before the launcher,
timing excludes reset, and one layer is reported separately from its 48-layer
projection.

The vendor source that vLLM's pinned build carries **passes** the CPU-reference
bands after those repairs:

| fixture | state max / mean rel err | output max / mean rel err |
|---|---:|---:|
| 129 tokens padded to 192, nonzero initial state | 0.04048 / 0.001559 | 0.05004 / 0.002262 |
| 2048 tokens, zero initial state | 0.05558 / 0.001608 | 0.05071 / 0.002330 |

But a strengthened full-chunk test finds **differing state or output bytes
across independent repetitions of the same call**. That fails the determinism
gate outright, and it means the attractive 1.847 ms/layer timing is not an
adoption-ready target. No tolerance was relaxed to make it pass.

A newer upstream revision of the same kernel **passes** the same checks: eight
repetitions byte-identical, bands unchanged, across four fixtures. The relevant
source change replaces local-space barriers with `sycl::group_barrier` and
explicitly types the split-barrier scopes, which supports the synchronisation
hypothesis without proving which individual barrier caused it. Model-level
golden gates for adopting vendor arithmetic were never run.

### oneDNN's int4 primitive

Built the exact revision vLLM pins, in an isolated prefix rather than over the
system library, GPU SYCL and CPU none. All shapes dispatch the real JIT GEMM
generator, not a reference fallback.

Exact identity extraction still fails on **33,198,358 of 83,886,080 weights**,
plus 5,372,359 sign-of-zero differences. The diagnostic that explains it:
**every extracted weight equals a truncation of the fp32 dequant product to
bf16**, ignoring zero signs, at most one bf16 ULP from our round-to-nearest
oracle. Rounding the scale to bf16 first does not account for all of it. This
supersedes an earlier speculative explanation about separately rounded multiply
and subtract steps. The strict correctness gate still fails, so this is not an
adoption.

The rate side of the same question is in
[probe-prefill-vllm-parity-2026-09-14.md](probe-prefill-vllm-parity-2026-09-14.md):
the vendor's own fused int4 W4A16 primitive measured **below** the break-even an
in-kernel int4 GEMM of ours would have to clear.

## The replay prototype

`B70_PREFILL_REPLAY=1` selects recorded L0 chunks;
`Engine::set_prefill_replay(bool)` overrides the environment. The default is
still immediate. An eight-entry FIFO cache keys frozen arguments by position and
active row count. Input ids and persistent buffers stay dynamic. Recordings are
destroyed before their kernels and context, and replay synchronizes before any
shared-buffer update. SYCL, profiling and nested recording are rejected rather
than silently captured wrongly.

Model replay tests compare exact bytes of GDN state, convolution history, both
KV caches, control and logits, across reset with dirty scratch, changed prompts,
incremental positions, two full chunks plus a one-token tail, several ragged
widths, cache eviction and recapture.

**Kept experimental.** A 13.75 ms saving does not explain a 204 ms gap, and the
complexity is real.

## What this investigation concluded

- Removing SYCL and framework layers did **not** by itself make the GPU kernels
  equivalent. Our linear path is competitive and then some; the vendor's
  remaining advantage was in GDN, flash attention and fused pointwise work.
- Whole-chunk L0 replay is numerically exact in every exercised case and saves
  about 0.5%. Not the lever.
- Do not adopt the exact-pin int4 primitive: it fails the strict identity gate
  and is slower than the two-pass path it would replace.
- Do not adopt the nondeterministic vendor GDN revision. The newer one is a
  usable numerical and performance reference, but a diagnostic band is not a
  model-level adoption gate.
- The next substantial target is a higher-precision GDN rewrite, split-BF16
  specifically, then attention, then pointwise and reduction fusion. The earlier
  single-BF16 DPAS scan failed a real golden token gate, and its speed is not
  permission to reintroduce its rounding.

Holding everything else fixed, substituting the vendor GDN core rate was a
diagnostic projection only: 2748.157 - (527.6 - 141.833) = 2362.390 ms, about
1734 t/s. Derived, not measured, and it leaves every other kernel and all host
work unchanged.

There was measured GPU-work headroom larger than the latency gap of the day.
That made parity plausible rather than proven. It turned out to be there.

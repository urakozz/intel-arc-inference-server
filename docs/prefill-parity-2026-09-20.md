# Prefill parity: matched measurements and remaining work

Box: `user@box`. Follow-up to spec 2.1 / plans 9a-9e.
Execution ledger: [approved four-point follow-up](superpowers/plans/2026-09-19-prefill-parity-followup.md).

## What the current measurements establish

Removing SYCL/framework layers did not make the GPU kernels equivalent.
Our linear path is competitive; the remaining advantage in the measured vLLM
pipeline comes from GDN, flash attention and fused pointwise/reduction kernels.
Whole-chunk L0 replay is numerically exact in the exercised model cases, but
only saves about 0.5% here. There is no measured 1900+ tokens/s native result.

### Matched warm requests, GPU 0

| Path | Median latency | Prefill tokens/s |
|---|---:|---:|
| Native L0, immediate | 2748.157 ms | 1490.45 |
| Native L0, recorded chunks | 2734.409 ms | 1497.95 |
| vLLM, same checkpoint/IDs/chunk | 2544.043 ms | 1610.04 |

Measured: three warm requests per path, after one first-use request. GPU work
was serialized; no compiler ran during this comparison. All returned first
token **383**; vLLM reported zero cached prompt tokens. The current warm
latency gap is 204.1 ms (immediate) or 190.4 ms (recorded), not the historical
~657 ms inferred from 1973 tokens/s. This does not invalidate the older run:
its software/settings are not the current controlled comparison.

The historical 1973 tokens/s reference is retained as a separate result. Its
[dense benchmark record](BENCHMARKS.md#dense-4-bit-qwen38-27b--two-quantizations)
records `p314-t214-vxkp0`, Python 3.14, Torch 2.14.0, source kernels patch 00,
`pp4096` and two 2048-token chunks, with an HTTP-inclusive scope; the
[spec-2 re-gate table](BENCHMARKS.md#the-spec-2-re-gate-rows--977a31c-2026-09-09--the-first-record-grade-rows-this-project-has-taken)
also labels it external. Exact historical vLLM/oneDNN revisions, device and
GDN dtype are not fully recorded. Consequently the records do not establish a
causal explanation for 1973 versus the current 1610.04 tokens/s. A 1900+
target remains unachieved and requires a separately specified rebenchmark.

Configuration:

- Checkpoint `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, snapshot
  `84575a18f209992ef96d819b31f924b489e3d55d`.
- 4096 IDs obtained by repeating `tests/golden/prompts/prose.ids`; uint32-LE
  SHA256 `342eace92c4214eadd1ccfda421d3a576ae753f9be4ee61abbca167316237e67`.
- Chunk and `max_num_batched_tokens` 2048, maximum length 16384, BF16
  activations/KV, FP32 GDN state, one sequence, TP1/PP1, no prefix caching or
  speculation.
- vLLM `0.29.1rc1.dev380+g23e26e058.d20260919.xpu`, torch
  `2.15.0.dev20260913+xpu`, kernels `0.1.15.dev22+g3d74ec9.d20260918`,
  oneDNN `3.13.0.0e2a5bfeef1bfbffc3137464606540233086ce9b`.
- Native scope: entire `Engine::prefill()` call through first ID in shared
  control memory. vLLM scope: `LLM.generate()` through one returned ID,
  including scheduling/sampling, excluding HTTP/tokenization/model loading.
  Native reset is outside timing. vLLM receives independent uncached requests.

First native immediate request was 2729.918 ms. First recorded request was
2748.672 ms, **after immediate warmup**, including capture cost but not cold
module/scratch initialization. vLLM's first request in the idle rerun was
2542.564 ms, after engine warmup and with a retained compilation cache.
These are first-request labels, not interchangeable cold-start measurements.

Logs on the box: `prefill-followup-replay-bench.log`,
`prefill-followup-vllm-idle.log`. The earlier independent native process runs
were 2717.3/2714.2/2727.0 ms (median 1507.36 tokens/s); do not select those
colder results over the explicitly matched warm protocol.

### Reproducing the measured paths

The native replay benchmark can be run with
`build-nosycl/tests/prefill_replay_test urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench`.
The vLLM driver is `tools/probe/bench_vllm_prefill.py --model PATH`. For the
vendor probes, configure the probe build with the exact isolated oneDNN root
`-DB70_ONEDNN_ROOT=/home/user/b70-onednn-v3.13-followup/install` and
`-DB70_VLLM_XPU_KERNELS_DIR=/home/user/b70-vllm-xpu-followup-d7c35d2`.
Those are diagnostic build inputs only: no production dependency was changed.

## Where vLLM spends GPU time

Separate diagnostic trace of one uncached 4096-ID request, same container and
GPU 0. Wall time was 2567.373 ms. Count only Chrome-trace `cat=kernel` events
once: the profiler's operator rows and graph annotations contain those same
kernels and must not be added again.

| Disjoint GPU kernel family | Kernel time | Launches |
|---|---:|---:|
| GEMM (INT4 linears plus AB/head GEMMs) | 2130.553 ms | 610 |
| GDN kernels (including conv/update) | 177.270 ms | 672 |
| Flash attention | 63.091 ms | 64 |
| Triton pointwise/reduction | 125.253 ms | 706 |
| Other (including fills/cache operations) | 17.860 ms | 973 |
| Total | 2514.027 ms | 3025 |

Kernel-interval union: 2514.002 ms; first-kernel-to-last-kernel span:
2537.572 ms. Only **23.569 ms** of that span is outside kernel execution.
The trace also contains 38 H2D/D2H copies totaling **14,056 bytes and 0.070 ms**.
This run provides no evidence of large intermediate tensors shuttling through
the CPU between torch, Triton and oneDNN.

The INT4 operator alone accounts for 2116.9 ms across 512 calls. GDN's core
prepare/A/inverse/WU/forward kernels total 141.833 ms across 96 layer/chunk
instances; forward/state scan is 47.570 ms. The larger GDN operator total is
185.952 ms because it also includes child operations such as scratch fills.

Raw trace: `/home/user/prefill-followup-trace/rank0.1789897479237655575.pt.trace.json.gz`;
text table: `profiler_out_0.txt` in the same directory. Log:
`prefill-followup-vllm-trace-retained.log`. The original image ID was removed
from Docker's store during the investigation; its retained, owned baseline
container was restarted with profiling to preserve exact software parity.
Normal driver defaults were restored immediately after argument parsing.

## Native profiling repair

`B70_PREFILL_PROFILE=1` now reports L0 GPU timestamps, host launch submission
and residual host waits separately. It no longer labels the instrumented
call's wall time as an uninstrumented run or derives fictitious instrumentation
overhead from it. Timestamp collection itself is outside the wait timer.
SYCL work is explicitly outside L0 timestamp coverage. The PV boundary now
closes each attention group, so PV is not attributed to the following QK.

Initial GPU-1 diagnostic: 2824.3 ms instrumented wall, 2694.5 ms L0 GPU sum,
89.7 ms host submission, 17383 launches. Linear kernels: 1649.8 ms; GDN scan:
310.6 ms; all GDN stages: approximately 614 ms. These overlap with host work,
and GPU 1 is not interchangeable with GPU 0 for exact percentage comparisons.
A corrected GPU-0 run measured 2761.1 ms instrumented wall, 2651.4 ms L0 GPU
sum, and 67.5 ms host submission. QK and PV now each account for exactly 128
launches, with the unchanged 17383-launch total.

Same-GPU diagnostic comparison (different instrumentation; not a precise
end-to-end subtraction):

| Work | Native L0 GPU time | vLLM GPU time |
|---|---:|---:|
| INT4 linear path | 1620.1 ms | 2116.9 ms |
| GDN core: gate/A/solve/WU/A2/scan | 527.6 ms | 141.8 ms |
| Attention QK/softmax/PV vs flash attention | 148.2 ms | 63.1 ms |
| SiLU/multiply | 98.7 ms | 51.3 ms |

The native GDN scan alone is 304.7 ms versus the vendor forward kernel's
47.6 ms. Operator boundaries and rounding points differ; these are bottleneck
measurements, not drop-in replacement guarantees. Native residual/norm is
another 104.2 ms; vLLM fuses several such operations, so there is no single
exactly matching trace row. Log: `prefill-followup-profile-gpu0.log`.

As an approximate diagnostic partition, native L0 kernel sums are 2651.4 ms
versus 2514.027 ms for vLLM. Native INT4 linears are about 497 ms faster
(1620.1 versus 2116.9 ms), but the remaining work is about 634 ms slower
(1031.3 versus 397.1 ms), yielding a net roughly 137 ms GPU deficit. This
explains how the path with fewer layers can still lose. These GPU-event
categories have different boundaries and their host times overlap, so they
are not an end-to-end decomposition. The vLLM trace's copies total only
14,056 bytes / 0.070 ms.

## Vendor probes: what passed and what did not

### GDN

Repaired three actual probe defects: out-of-order dependent submissions,
missing sigmoid on beta, and reuse of the gate input after the vendor prepare
kernel overwrote it. Raw a/b now use the CPU reference's BF16 rounding points;
Q/K are normalized before the launcher. Timing excludes reset and reports
one layer separately from its 48-layer projection.

The original remote vendor source (`5085fddea4350d16426300f1b7467e24bf5e3f83`)
passes the CPU-reference bands after those repairs:

| Fixture | State max/mean relative error | Output max/mean relative error |
|---|---:|---:|
| 129 tokens, padded to 192, nonzero initial state | 0.04048 / 0.001559 | 0.05004 / 0.002262 |
| 2048 tokens, zero initial state | 0.05558 / 0.001608 | 0.05071 / 0.002330 |

But the strengthened full-chunk test detects differing state or output bytes
across independent repetitions. **That fails the determinism gate.** The
earlier 1.847 ms/layer timing is consequently not an adoption-ready target.
The 129-token case passes eight independent repetitions (0.682 ms/layer,
diagnostic only). No tolerance was relaxed.

An isolated copy of current upstream headers (`d7c35d2`) **passes** the same
checks without modifying the remote vendor checkout. Full-chunk output/state
bands are unchanged, and eight repetitions are byte-identical. Separate runs
also pass 129 tokens/nonzero state, 2048 tokens/nonzero state and another 2048
tokens/zero state, each with eight repetitions. The newest case logs report
129-token/nonzero-state at 1.206 ms/layer (diagnostic, not a rate claim),
2048-token/nonzero-state at 1.134 ms/layer, and 2048-token/zero-state at
1.120 ms/layer versus the first 1.123 ms/layer result. These exclude conv,
packing, normalization and gated output norm, and are diagnostic rates.

The relevant source delta changes local-space barriers to
`sycl::group_barrier` and explicitly types split-barrier scopes. Passing after
that vendor delta supports the synchronization hypothesis; it is not proof
of which individual barrier caused the older nondeterminism. Model golden
gates for adopting vendor arithmetic have **not** been run. Logs:
`prefill-followup-gdn-current.log`, `prefill-followup-gdn-current-cases.log`.

### oneDNN

Built the actual vLLM pin `0e2a5bfeef1bfbffc3137464606540233086ce9b` in
`/home/user/b70-onednn-v3.13-followup/install`, not over the system library.
GPU SYCL / CPU NONE; shared packaging here versus static in vLLM. The source
archive's version reports 3.13.0 with hash N/A; the archive pin is recorded
separately rather than inferred from that missing hash.

All shapes dispatch `jit:gemm:any`. Diagnostic M2048 times for
qkvz/out/gate-up/down/qkv were 3.201/1.185/7.637/2.784/2.777 ms. This does not
establish a useful GEMM advantage over the native slab path; the vLLM trace
also spends more on INT4 linears than the initial native linear profile.

Exact identity extraction still fails on **33,198,358 / 83,886,080 weights**,
plus 5,372,359 sign-of-zero differences. New diagnostics establish that every
extracted weight equals **truncation of the FP32 dequant product to BF16**,
ignoring zero signs, with maximum difference one BF16 ULP from our round-to-nearest
oracle. Rounding the scale to BF16 first does not explain all results. This
supersedes the older speculative explanation about separate multiply/subtract
rounding; the strict correctness gate still fails and this is not an adoption.

## Delivered replay prototype and verification

`B70_PREFILL_REPLAY=1` selects experimental recorded L0 chunks;
`Engine::set_prefill_replay(bool)` overrides the environment. Default behavior
remains immediate. An eight-entry FIFO cache keys frozen arguments by
`(position, active rows)`. Input IDs and persistent buffers remain dynamic.
Recordings are destroyed before their kernels/context; replay synchronizes
before any shared-buffer update. SYCL, profiling and nested recording are
rejected rather than silently captured incorrectly.

Model replay tests compare exact bytes of GDN state, convolution history,
both KV caches, control and logits. Covered: reset with dirty scratch,
changed prompts, incremental positions, two full chunks plus a one-token tail,
several ragged widths, cache eviction and recapture. The low-level test also
covers pending immediate work, frozen scalars, changed buffer contents and
capture exception recovery. Profiling tests cross the 256-event pool boundary.

Fresh review found no Critical/Important issues. Complete no-SYCL regression:
**65/65 passed**, 467.89 seconds, including decode/server golden tests,
prefill golden, consistency, determinism and replay. The additional SYCL-enabled
full build/CTest passed **76/76**, with zero failed or skipped tests, in 790.99
seconds. Log: `/home/user/prefill-followup-full-regression.log`.
Deferred review minors: profile-mode vLLM medians mix instrumented/plain rows
(not used above), and repeated GDN output is not poisoned before each run
(consistency is checked, complete overwrite is not independently proven).

## Decision

Keep replay experimental: a 13.75 ms saving does not explain the gap. Do not
adopt the exact-pin INT4 primitive or the older nondeterministic GDN path.
The current vendor GDN probe is a usable numerical/performance reference,
but its diagnostic bands do not replace model-level adoption gates.
Fused INT4 GEMM/epilogue work is explicitly deferred: the exact-pin oneDNN
path fails the strict identity gate and the current vLLM linear path is slower.
This is not a claim that all kernel optimization is exhausted. The next
substantial target is a separately designed higher-precision GDN rewrite (for
example, split-BF16), followed by flash attention and pointwise/reduction fusion. The earlier
single-BF16 DPAS scan failed a real golden-token gate; its speed is not
permission to reintroduce its rounding changes. Any higher-precision GDN
design remains subject to the existing numerical and model gates.

Holding all non-GDN work unchanged, substituting the vendor GDN core rate is
only a diagnostic projection: `2748.157 - (527.6 - 141.833) = 2362.390 ms`,
or about 1734 tokens/s. It is not measured and leaves other kernels and host
work unchanged.

There is measured GPU-work headroom larger than today's ~190-204 ms latency
gap. That makes parity plausible, not proven. Reaching the historical
1900-1973 tokens/s would require another 592-672 ms reduction from
the measured native immediate warm latency, so it is a separate, harder target.

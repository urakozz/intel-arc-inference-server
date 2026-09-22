# Prefill parity follow-up implementation and experiment plan

**Goal:** Execute the four follow-up points approved by the operator on 2026-09-19: repair the evidence, establish a matched vLLM comparison, measure valid vendor kernels, and measure replay before selecting further kernel changes.

**Design basis:** The approved in-chat investigation and four-point recommendation. Existing execution paths are retained; diagnostics are opt-in. Kernel adoption is conditional on correctness and measured end-to-end benefit. Matching 1973 tokens/s is a target, not an assumed outcome.

**Execution:** Native execution in the existing `spec1.7-codex-exp` experiment checkout; preserve the operator's local source changes and the newly updated box address. No push or merge. Keep this file as the experiment ledger.

## Constraints and measurement rules

- Box: `user@box`. Compile and execute GPU work there, not on the Mac.
- Do not stop the operator's existing image build or modify the patched vLLM checkout.
- Use device 1 for diagnostics; use the same device for paired comparisons. Serialize GPU workloads.
- Long jobs run detached with unique logs and PID records; do not delete remote work.
- Record exact checkpoint, software/image revision, KV dtype, chunk width, token IDs and cache behavior.
- Preserve golden, determinism, backend-equivalence and decode invariants. Never weaken correctness thresholds to admit a faster kernel.
- Label source findings, measured times and projections separately. GPU-event sums, host submission, waits and end-to-end latency are different quantities and must not be added when they overlap.
- Existing CPU compilation makes initial timings diagnostic; record its presence. A final idle comparison is required for a record-grade claim.

## Task 1: Repair the GDN probe and profiling

Files: `tools/probe/probe_pf_gdn_xe2.cc`, `src/runtime/prefill/{context,profile}.{h,cc}`, `tests/prefill/context_test.cc`, relevant test registration and CLI reporting.

- [x] Reproduce the GDN correctness failure with an executable numerical assertion. Test independent repeated runs, not just finiteness.
- [x] Give dependent vendor submissions an in-order queue; restore the mutable raw gate input before every run; separate preparation from timed kernel execution.
- [x] Validate both output and state, investigate remaining discrepancies stage by stage, and exercise nonzero initial state and ragged/multiple chunks before treating the result as a usable target.
- [x] Report one-layer time and explicitly multiplied 48-layer core time; exclude unmatched conv/norm stages from speedup claims.
- [x] Add opt-in L0 timestamps and host launch timing, test timestamp reuse and argument correctness, and retain diagnostics outside the production fast path.
- [x] Report the actual instrumented call's wall time, residual wait time and L0-only GPU coverage. Remove the unsupported upper-bound and negative instrument-cost claims.

## Task 2: Matched vLLM baseline

Files: a standalone comparison driver under `tools/probe/` and the resulting measurement document.

- [x] Inspect the available image by immutable ID and verify its model/kernel/oneDNN versions without rebuilding or editing vLLM.
- [x] Use checkpoint `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, snapshot `84575a18f209992ef96d819b31f924b489e3d55d`, BF16 KV, 4096 identical IDs, chunk 2048, concurrency 1, TP1, no speculation, no prefix caching.
- [x] Measure comparable prefill/first-token scopes and preserve server-inclusive and engine-only labels. Capture the kernel profile as a separate diagnostic run.
- [x] Validate generated results and record cold/warm distinctions and at least three timed repetitions.

## Task 3: Vendor reference kernels

Files: existing `probe_gemm_onednn_int4`, `probe_pf_gdn_xe2`, isolated dependency build configuration as needed.

- [x] Build oneDNN at the actual vLLM pin in a fresh remote directory; do not replace system oneDNN or vendor checkouts.
- [x] Run the existing identity-extraction correctness check and all production GEMM shapes. Report the version and implementation selected by oneDNN.
- [x] Compare the corrected GDN stages with the engine's matching stages, and include normalization, conversion and scratch costs in any end-to-end projection.
- [x] Select subsequent kernel work only from these measurements; document a negative result as a completed experiment, not a reason to force adoption.

## Task 4: Whole-chunk replay and measured kernel improvements

Files: prefill context/engine and existing context/equivalence tests; kernel files only when Task 3 supplies evidence.

- [x] Prototype an opt-in recorded L0 chunk, preserving dynamic position, ragged length, buffer lifetimes and queue ordering. Account for first-use recording cost separately from repeated replay.
- [x] Test changing input data and positions, reset and repeated calls, two chunks and a ragged tail against immediate execution.
- [x] Benchmark immediate versus recorded execution on identical inputs, including first-request and warmed-request latency.
- [x] If the vendor measurements justify fused INT4 GEMM or epilogue work, implement the smallest candidate and run its numerical and model gates before timing adoption. The measurements do not justify it: exact-pin oneDNN fails identity and current vLLM linears are slower, so fused INT4/epilogue work is explicitly deferred.
- [x] Run the relevant complete regression checks, obtain a fresh code review, and publish measured results and remaining limitations. The no-SYCL suite and fresh review are complete; the additional full SYCL-enabled build/CTest passed 76/76 with zero failed or skipped tests in 790.99 s.

## Progress and rulings

- 2026-09-19: Address updated in runtime helpers, current connection docs and CMake instructions; shell syntax and diff checks pass.
- 2026-09-19: Remote inspection found an operator-owned vLLM image build at `-j10`, approximately 93 GiB available RAM, no running Docker containers or inference processes. Existing installed image ID begins `4e8293325229`; do not rely on the mutable image tag while its replacement builds.
- Ruling: Continue the explicitly approved four-point workflow without another permission handoff. Start with bounded diagnostic repairs; kernel adoption and replay remain evidence-driven.
- Ruling: The current branch is already the project's experiment branch and holds the requested address edits. Keep changes here and use isolated remote dependency/build directories where needed.
- New finding: `profile_report("--pp", ingest_ms)` receives the same instrumented run's wall time. The previous report's "plain" label and calculated instrumentation overhead were invalid, independently of the wait-only attribution defect.
- 2026-09-20: Opt-in L0 timestamp/host-submit implementation passed `context_test` after the missing-API RED build. Diagnostic pp4096: wall 2824.3 ms, GPU-event sum 2694.5 ms, host launch submission 89.7 ms, 17383 launches. Linear GPU time 1649.8 ms; GDN scan 310.6 ms. These overlapping times are not additive. Attention sub-phase attribution still needs correction (PV submissions currently drain under the following QK boundary).
- 2026-09-20: Immutable vLLM image `4e8293325229`: vLLM `0.29.1rc1.dev380+g23e26e058.d20260919.xpu`, torch `2.15.0.dev20260913+xpu`, kernels `0.1.15.dev22+g3d74ec9.d20260918`, oneDNN `3.13.0.0e2a5bfeef1bfbffc3137464606540233086ce9b`. GPU 0, exact checkpoint and 4096 IDs (uint32-LE SHA256 `342eace92c4214eadd1ccfda421d3a576ae753f9be4ee61abbca167316237e67`), no cached tokens, stable first token 383. First request 2673.51 ms; three warm requests 2542.84/2550.24/2554.80 ms, median 1606.12 tokens/s. CPU dependency build was active; idle recheck and separate trace remain pending. Historical 1973 is not a matched current baseline.
- 2026-09-20: Idle GPU-0 native first-request runs: 2717.3/2714.2/2727.0 ms, median 1507.36 tokens/s. Same checkpoint/chunk/IDs. Native scope ends at first ID in device memory; vLLM includes scheduler/sampling/result return. Native warm repeated requests still pending.
- 2026-09-20: Isolated oneDNN at the exact image pin built and installed in `/home/user/b70-onednn-v3.13-followup/install`. Shared library packaging differs from vLLM's static build. Targeted probe rebuild uses this root; production backend is unchanged.
- 2026-09-20: GDN in-order-only correction still failed (state max/mean RMS-floored relative error 56.63/1.015). Source tracing found the missing activation: vendor conv/reorder computes sigmoid(b), whereas the probe supplied raw b. Vendor prepare transforms a in place. Corrected probe now provides sigmoid(bf16(b)), bf16-rounded raw a, and restores a before each independent run. Numerical gate remains unchanged; build/run pending. Removed invalid one-layer-versus-48-layer speedup calculation.
- Ruling: GPU 0 is the paired-comparison device, with GPU 1 reserved for diagnostic probes. Serialize measured inference; compilation makes concurrent timing diagnostic, not record grade.
- 2026-09-20: Context capture/replay RED (missing API) → GREEN: frozen scalar arguments, changed buffer contents, pending immediate work ordering, 300-launch recording, failed-capture recovery, profiling/SYCL/nested-capture rejection. Timestamp reuse now crosses the 256-event pool boundary (300 events, twice).
- 2026-09-20: Engine replay RED (missing setter) → GREEN on the real GPTQ checkpoint: eight cases × immediate/capture/reuse with exact persistent state, control and logits. Includes reversed prompt, incremental calls, 4097 IDs (two full chunks + ragged tail), widths 16/17/18/19 and cache eviction/revisit. Cache is bounded to eight `(pos,C)` recordings. Off by default; `set_prefill_replay(bool)` overrides `B70_PREFILL_REPLAY=1`.
- 2026-09-20: Idle sequential warm comparison on GPU 0: native immediate median 2748.157 ms (1490.45 tokens/s), native recorded median 2734.409 ms (1497.95 tokens/s), vLLM median 2544.043 ms (1610.04 tokens/s). All return first token 383. Replay saves only 13.75 ms / 0.50%; keep experimental and off by default. Native first request includes lazy scratch/modules; recorded first use follows immediate warmup and includes capture, so those first-use rows are not cold-engine equivalents.
- 2026-09-20: Corrected vendor GDN state gate passes: max/mean relative 0.05558/0.001608, 1.847 ms/layer, 88.678 ms projected across 48 layers per 2048-token chunk. This run restored mutable input each time but did not yet validate output/repetition bytes/nonzero initial state/ragged input; extended checks queued. Initial attempted run used conflicting device selectors and failed before GPU execution; the successful run unsets `ONEAPI_DEVICE_SELECTOR` and selects GPU 1 via `ZE_AFFINITY_MASK` only.
- 2026-09-20: Exact-pin oneDNN dispatches `jit:gemm:any`; M2048 qkvz/out/gate-up/down/qkv = 3.201/1.185/7.637/2.784/2.777 ms. Identity extraction reproduces exactly 33,198,358 nonzero mismatches / 83,886,080 weights from the older-library experiment. No direct adoption; extra rounding diagnostics queued, gate unchanged. During this diagnostic the model replay test was on the other GPU, so rates are not record grade.
- Ruling: Do not implement fused INT4 GEMM from this measurement: the newer vendor reference neither clears the identity gate nor demonstrates a compelling linear speed advantage. Pursue the verified GDN execution gap as the next kernel-design target; the earlier failed single-bf16 DPAS model gate remains binding.
- 2026-09-20: vLLM trace launch found the original image removed from Docker's image store, although our stopped baseline container remains restartable. Preserve that owned container as a new local diagnostic image (including its compile cache); do not substitute the operator's newer mutable image. Extended vendor validation and the complete no-SYCL regression suite are queued serially.
- Trace recovery: `docker commit` also failed because an original image content digest was missing. Instead temporarily defaulted our comparison driver to profiling and restarted the retained baseline container (PID 1528746, log `/home/user/prefill-followup-vllm-trace-retained.log`). Restore the driver's normal default after its process has parsed arguments. Our validation shell PID 1527809 is SIGSTOP-paused while its compiler child finishes, to prevent GPU test overlap; SIGCONT it after the trace exits. No operator-owned process was paused.
- Trace recovery completed: profiling arguments were observed in the running process log, then the normal driver default was restored locally and remotely. Trace container exited; validation PID 1527809 resumed. Profiler reports 2.514 s self XPU, including 2.117 s INT4 GEMM, 185.952 ms GDN and 63.201 ms flash attention (hierarchical operator totals, not additive with their child kernels). Full trace copied from the retained container's `/results` to the box's `/home/user/prefill-followup-trace/`.
- Final fresh-context review: no Critical/Important findings; opt-in implementation acceptable subject to queued validation. Replay ordering/lifetimes/cache tests and timestamp calibration reviewed.
- Final: minor (deferred): profile-mode vLLM summary mixes its instrumented request with subsequent uninstrumented requests. It remains labeled diagnostic; use per-request values and the separate non-profile baseline, never this median, for comparisons.
- Final: minor (deferred): GDN repeated-output checks do not poison the output allocation before each invocation, so they establish consistency but do not independently prove complete overwrite. State/scratch/raw-gate reset, CPU-reference checks and finite checks remain active.
- Final: Ruling: review set aside remote regression/extended numerics and trace attribution because they were pending, not because they are optional. Controller will report their actual outcomes; production vendor adoption remains excluded without model-level gates. Cost of proceeding without those gates would be incorrect performance/correctness claims, so no such adoption is made.
- 2026-09-20: Complete no-SYCL build and CTest: **65/65 passed**, 467.89 s; no skipped/failed tests. Includes `prefill_replay_test`, prefill golden/consistency/determinism, decode golden and golden server. Log `/home/user/prefill-followup-regression.log`.
- 2026-09-20: Corrected GPU-0 phase profile: QK/PV each 128 launches (was 224/32), total 17383. Wall 2761.1 ms; L0 GPU sum 2651.4 ms; host submission 67.5 ms. Linears 1620.1 ms, GDN core 527.6 ms, attention QK/softmax/PV 148.2 ms, SiLU 98.7 ms. Report and raw-log pointers in `docs/prefill-parity-2026-09-20.md`.
- 2026-09-20: Extended old-vendor GDN test caught a real repeatability failure despite passing reference bands. Isolated updated `d7c35d2` headers (source delta: group-barrier semantics/typed split-barrier scopes) pass the same unchanged gates: initial 2048 case plus 129/nonzero, 2048/nonzero and another 2048/zero, each eight bit-identical repetitions. Full-chunk core 1.120-1.134 ms/layer. Source synchronization change is a supported explanation, not isolated per-barrier proof. No production vendor kernel adopted.
- 2026-09-20: oneDNN diagnostic: all 83,886,080 extracted weights match FP32-product-to-BF16 truncation (zero signs ignored); max distance from RNE oracle is one BF16 ULP. This supersedes the prior speculative two-multiply/subtract explanation. Exact gate is unchanged and still fails.
- 2026-09-20: Additional full SYCL-enabled build/CTest verified the reference backend and both-build linkage: **76/76 passed**, zero failed or skipped, 790.99 s. Log `/home/user/prefill-followup-full-regression.log`. No commit, push or merge was performed.

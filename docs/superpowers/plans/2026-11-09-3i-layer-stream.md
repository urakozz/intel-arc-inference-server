# 3i: representative fused-linear weight walks

> Use superpowers:executing-plans. The operator requests implementation and local merge, with focused checks only.

**Goal:** extend probe_fused_small to measure distinct per-layer weight allocations instead of extrapolating one repeatedly reused matrix.

**Design:** default to a `layers` weight mode: one independent synthetic weight/scale allocation for every real layer of the selected Qwen3.8 shape (64 gate/up matrices, 6,060,769,280 bytes). Retain explicit `resident` mode for reproducing old measurements. Use the same allocation order and activations for every arm; gate every layer's output against its original slab walk before timing that shape. Warm complete walks and rotate arms between rounds. One timestamp pair and one fence surround the whole layer walk; record host wall separately. Normalized per-linear times are measured walk totals divided by layer count; only resident-mode chunk totals are extrapolations. No production code or kernels change.

**Limits:** distinct weights have real shapes and addresses, but synthetic values. This isolates a linear family, not surrounding model operators, dependencies or complete serving. Compare both candidates with the existing recorded CLI/server evidence; report any ranking disagreement rather than forcing a desired result.

- [ ] Host-only CLI/protocol tests: plan counts/bytes, explicit modes, zero extrapolation in layers mode, invalid inputs fail before GPU initialization. Observe RED against old probe.
- [ ] Implement bank allocation, per-layer bitwise checks, complete-walk warm-up/timing, structured raw samples and explicit resident estimates.
- [ ] Run focused GPU checks: GU resident/layers at 237/250 with split and SLM, one layout-1 family, and a wrong-last-layer negative control.
- [ ] Compare recorded engine evidence, document results/limits, static review, signed local merge and move existing 3i to done.

**Verification limit:** no full/end-of-branch suite, no repeated model-serving campaign, no new work items. Builds on the box with JOBS=16; card lock for correctness, whole-box lock for timed comparisons; detached jobs and masked output. Preserve unrelated worktrees and untracked reports.

# 3g: lazy setup for fused prefill

> **For agentic workers:** use superpowers:executing-plans; preserve the operator's dedicated-test-only limit.

**Goal:** fused short linears perform common validation without allocating the unused dequant slab or loading its module.

**Design:** split the common checks in linear_l0.cc from slab/module preparation. Both plain and SiLU entry points validate their output contracts, try the existing fused selector, and prepare the slab only on fallback. Keep all kernels, selectors, arithmetic, launch counts and runtime options unchanged. The operator requested this existing work item directly; no further design decisions need approval.

**Scope and verification:** only dedicated tests and focused before/after measurements. No full build/test campaign, box_check, broad prefill/MTP suite or end-of-branch suite. All builds on the box with JOBS=16. Preserve the existing uncommitted report and unrelated main reports.

- [x] Add a dedicated real-GPU regression and observe RED: a fresh fused call must leave lazy bytes zero and cache only its fused module.
- [x] Cover both weight layouts, plain/SiLU, selector boundaries, validation failures, warm reuse and short→slab→short bitwise output against the original slab walk.
- [x] Split the setup, build only relevant targets and observe dedicated GREEN.
- [x] Compare a small fixed real-checkpoint sequence before/after: cold/warm short prefill, later fallback, byte-identical logits/ids, allocation and timing.
- [x] Review the small change, record results and update existing item 3g. No new work-list items.

**Review focus:** common checks still precede every launch; output checks do not allocate; both fallback forms initialize their slab/module on demand; cached modules and existing slabs remain reusable; profile/launch accounting and other users of slab_buffer are unchanged.

Completed and merged locally as `af09c1a7`. Dedicated tests and focused paired measurements only; no full-suite or box_check run. Short requests avoid 34 MiB and unnecessary dequant modules; no meaningful wall-time gain was measured. Later fallback allocates and reuses the slab, with byte-identical outputs.

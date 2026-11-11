# TP int8 prefill implementation plan

Goal: enable the opt-in l0-int8 backend on TP without changing decode or bf16 prefill.

Architecture: use the existing int8 walk with direct row-parallel output. Keep 1024-wide rotations for divisible K; use 512-wide rotations for K=8704. Both operands use the same orthonormal transform. Full-width rotation would mix the two down shards at their boundary and require additional exchanges and weight preparation. The 512 option keeps TP scheduling intact; its changed numerics require independent f32 gates.

Constraints: own worktree, no Mac C++ compilation, no box tests in this session, no push. The supplied implementation brief is the spec. Tests are written before implementation, with execution deferred by operator request.

- [x] Add kernel CPU mirrors for K=8704, both int4 layouts, zero-column scales, quantisation and complete GEMM; keep existing 1024 cases.
- [x] Add rotation variants and direct output to the int8 walk. Check every TP linear K/N before chunks; precompute scales while both lists are idle.
- [x] Add backend selection to TpEngine, memory planning and both CLIs. Retain the int8 KV warning.
- [x] Register a TP int8 golden/tool-call gate with f32 judging and explicit missing-reference skips.
- [x] Review diff and kernel indexing locally. Read-only independent review found no blocking issue; corrected int8 memory-log categorisation. `git diff --check` and a local staging/index coverage audit passed. Build, negative controls, quality gates, serve checks and median-of-three measurements remain deferred; item 10 stays open in docs/open-work.md.

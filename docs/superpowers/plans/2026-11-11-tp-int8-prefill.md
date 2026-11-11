# TP int8 prefill implementation plan

Goal: enable the opt-in l0-int8 backend on TP without changing decode or bf16 prefill.

Architecture: use the existing int8 walk with direct row-parallel output. Keep 1024-wide rotations for divisible K; use 512-wide rotations for K=8704. Both operands use the same orthonormal transform. Full-width rotation would mix the two down shards at their boundary and require additional exchanges and weight preparation. The 512 option keeps TP scheduling intact; its changed numerics require independent f32 gates.

Constraints: own worktree, no Mac C++ compilation, no push. Box validation authorised after rebasing onto main. The supplied implementation brief is the spec. Tests are written before implementation, with execution deferred by operator request.

- [x] Add kernel CPU mirrors for K=8704, both int4 layouts, zero-column scales, quantisation and complete GEMM; keep existing 1024 cases.
- [x] Add rotation variants and direct output to the int8 walk. Check every TP linear K/N before chunks; precompute scales while both lists are idle.
- [x] Add backend selection to TpEngine, memory planning and both CLIs. Retain the int8 KV warning.
- [x] Register a TP int8 golden/tool-call gate with f32 judging and explicit missing-reference skips.
- [x] Review diff and kernel indexing locally. Read-only independent review found no blocking issue; corrected int8 memory-log categorisation. `git diff --check` and a local staging/index coverage audit passed. Build and focused tests started on the box after rebasing onto main `02ec4323`; logs: `~/b70/logs/tp-int8-prefill/validation-324952f5/`. Target dependencies now include the 3072/8704 quantisers and 512-wide requant variants. Negative controls, quality results, serve checks and median-of-three measurements remain pending; item 10 stays open in docs/open-work.md.

Validation started: targeted build PASS; `pf_int8_test`, `linear_i8_test`, `pipeline_args_test` and `prefill_int8_test` PASS. First model run aborted because the shared prefill test dependency list omitted bf16 short-chunk kernels. Added the variants selected by `pf_q4_tile` / `pf_small_tile`; rebuilt and reran affected tests only. TP gates are running in `tests-2.log`; the separate product build is in `product-build.log`. Missing f32 references will be listed by the gate; reference creation remains the operator's step.

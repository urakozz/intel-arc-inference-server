# TP int8 prefill implementation plan

Goal: enable the opt-in l0-int8 backend on TP without changing decode or bf16 prefill.

Architecture: retain 1024-wide rotations, zero-extending TP down's local K=8704 to 9216 in activation/weight kernels and int8 GEMM. Raw checkpoint addressing stays 8704. Scratch planning includes the padding. The 512-wide alternative failed f32 quality and was removed; full-width rotation would require additional crossings and weight preparation.

Constraints: own worktree, no Mac C++ compilation, no push. Box validation authorised after rebasing onto main. The supplied implementation brief is the spec. Tests are written before implementation, with execution deferred by operator request.

- [x] Add kernel CPU mirrors for K=8704, both int4 layouts, zero-column scales, quantisation and complete GEMM; keep existing 1024 cases.
- [x] Add rotation variants and direct output to the int8 walk. Check every TP linear K/N before chunks; precompute scales while both lists are idle.
- [x] Add backend selection to TpEngine, memory planning and both CLIs. Retain the int8 KV warning.
- [x] Register a TP int8 golden/tool-call gate with f32 judging and explicit missing-reference skips.
- [x] Review diff and kernel indexing locally; independent review found no blocking issue.
- [x] Build and validate on the box. `pipeline_args_test`, `pf_int8_test`, `linear_i8_test`, `memory_plan_test`, `prefill_int8_test` and `tp_prefill_test` pass. The fresh targeted build exposed missing bf16 short-chunk kernel dependencies; those were corrected.
- [x] Generate missing f32 references detached under the CPU-model lock. The final five-context batch completed successfully.
- [x] Apply the documented 8-ulp near-miss rule. `tp_prefill_int8_test` passes all 40 contexts, with no SKIP; worst differing row 6.70 ulp. The negative control rejects the bad judge.
- [x] Serve 39 golden/tool-call requests, smoke-test default MTP on three golden prompts, and verify the int8 KV known-fail warning.
- [x] Measure three interleaved idle runs at 4k / 16k: TP int8 3322.73 / 3047.73 t/s versus bf16 2934.67 / 2673.31, gains 13.2% / 14.0%. The operator confirmed the earlier figures were estimates, not thresholds.

Code commit: `0e41b221` (signed), branch `tp-int8-prefill`, rebased onto main `02ec4323`; never pushed. Full results and log paths: [validation record](../../probes/tp-int8-prefill-2026-11-11.md). Item 10 moves to Closed recently in `docs/open-work.md`.

The extra 256-token server-stream comparison produced 54 contexts. Its dumps are preserved for follow-up; it is outside the requested 64-token quality gate and is not claimed judged.

# Box validation queue

Work merged into `main` without running on the card, in the order to validate it when the box is
back. Each entry names its checklist or its tests. Remove an entry when it has passed on the box.

| # | merged | what is unvalidated | how to validate |
|---|---|---|---|
| 1 | spec 14 (Agnes 3.0 Flash, `ModelDesc`), commit range ending `c104247` | everything that runs on the card; the descriptor refactor's bitwise neutrality for Qwen3.8 | `2026-10-03-spec14-validation-checklist.md`, **G0 first** (Qwen3.8 kernel binaries checksum-identical, full suite, golden and replay bitwise) |
| 2 | spec 13c Task 1 (batching scheduler) and spec 8 §10 (`--mtp auto`), commits `6628218`, `da53c68` | the `EngineIface::step_many(sampling, k)` signature change in `EngineAdapter`, `b70_serve.cc`, `mtp_gpu_test`, `prefix_gpu_test`, `golden_server_test` (syntax-checked only); `--mtp auto` on the real engine | build; the full suite (especially `mtp_gpu_test`, `mtp_server_test`, `golden_server_test`, `prefix_gpu_*`); calibrate the int8-head cost table with `probe_mtp_steps` (M = 1..4, k = 1..3) and update `MtpCost`'s defaults; D1-style rows for `--mtp auto` against fixed K = 1 and K = 3 on the golden and A4 prompts |
| 3 | README rows | the headline and depth rows were measured on v1 decode attention and the bf16 head | re-measure on current `main` (v2 attention, int8 head), idle box, RECORD grade |
| 4 | g128 symmetric loader (branch `g128-loader`, `2757bfd`) | host-tested on the Mac (`quant_test`); the g64 path's `classify` was rewritten | `quant_test`, `load_checkpoint_test` and the golden gate on the g64 checkpoint (unchanged bytes); optionally load a public g128 sym GPTQ checkpoint of Qwen3.8 and run its golden prompts against its own CPU reference |
| 5 | spec 15b (`ModelDesc` carries every per-model width; Ornith's row; shape-suffixed variant names), 6 commits ending at the 15b comment fix on main | the routed device code has never built for the card; Qwen3.8 / Agnes bitwise neutrality | `2026-10-04-spec15b-validation-checklist.md` after entry 1 (spec 14) passes: kernel sha256 identical to main's, the full suite bitwise (774 launches, buffers_test byte table, golden / prefill / replay gates, `load_checkpoint_test` 849,398,784 MTP bytes), Agnes gates, `b70-decode` on an Ornith config fails with the MoE message |

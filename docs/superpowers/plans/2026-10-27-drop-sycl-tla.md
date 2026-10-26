# Drop the sycl-tla prefill backend (operator, 2026-10-27)

**Ruling:** "Drop sycl-tla backend. l0 should be the default one, l0-int8 optional." Direct GPU calls (our Level Zero /
OpenCL C kernels) over library layers. oneDNN is NOT part of this plan: `tools/probe/probe_gemm_onednn_int4.cc` stays.

**Status (2026-10-27): IN PROGRESS.** Inventory done (§1); the three things outside the sycl-tla backend that depended
on the SYCL component were ruled by the operator the same day (§2).

> **For agentic workers:** one implementing agent, gates below are the review. Commit after every step (signed; `WIP:`
> when it does not build). Handoff file at the worktree root when you stop.

## 1. Inventory (main 981c6be)

**The icpx component.** `cmake/prefill.cmake` (`B70_ICPX`, `B70_PREFILL` AUTO/ON/OFF, `B70_SYCL_AOT_256_GRF`,
`B70_SYCL_TLA_SRC_DIR`, `B70_SYCL_TLA_REVISION`, `B70_PREFILL_ENABLED`, `b70_prefill_ext`, `b70_link_prefill`);
`src/sycl/` (CMakeLists.txt, sycl_side.cc, gemm_batched.cc, xe_gemm_config.h, tla_pin.h.in, tla_smoke.cc,
vllm_shim/); `cmake/host_only.cmake:59-60`; `CMakePresets.json:13` (`B70_PREFILL=OFF`); `.gitignore` `.syclcache/`.

**Built by `src/sycl/CMakeLists.txt` (icpx only):** libb70_prefill.so (SyclSide + sycl-tla GEMM), b70_tla_smoke, and the
SYCL probes `probe_interop`, `probe_prefill_gemm`, `probe_prefill_attn`, `probe_gemm_int4_mixed`,
**`probe_gemm_onednn_int4`**, `probe_gdn_chunk` (opt-in, vllm_shim), `probe_pf_gdn_xe2` (vllm_shim).

**Runtime.** `prefill_backend.h` (`SyclTla`, parse / name); `prefill/backend.h`, `backend_sycl.cc`,
`backend_sycl_absent.cc`, `gemm_sycl.h`, `sycl_side.h`, `context_sycl.h`; `gemm.h`'s `gemm_bf16`,
`gemm_bf16_batched`, `gemm_bf16_supports_transb`, `gemm_bf16_batched_grid` (GemmBatch / GemmDims stay: `gemm_l0.h` uses
them); `Context::sycl()` / `has_sycl()` and their guards in `context.cc`; `Engine::prefill_sycl_side_created`;
sycl arms in `step.cc` (`linear_sycl`, launch / gemm / wait arithmetic), `attn.cc` / `attn.h` (`attn_qk_sycl` /
`attn_pv_sycl`, `attn_chunk_launches_sycl`, the 8-padding branch), `engine_prefill.cc`, `pipeline_prefill.cc`;
`memory_plan.cc:59-60` (the dequant scratch), `buffers.h:181` / `buffer_sizes.cc:259` (`dequant_buffer`, sycl-tla only).

**No model prefills only on sycl-tla.** Every model has an L0 walk; MoE (Ornith, K2), Bolt, Kolibri, Qwen3.8-Flash-Next
and `--pp 2` already refuse sycl-tla. Attention has the L0 composed and flash paths; no batched GEMM is sycl-only.
`vllm_shim` serves only the two CuTe-GDN reference probes; there is no vLLM prefill bridge in the engine.

**CLIs.** `b70_decode.cc` (usage 103/110, help 152/168, parse error 984, K2 message 1152, banner 1301), `b70_serve.cc`
(help 129/137/143, parse errors 1011/1127, K2 message 1258, banner 1377), `bolt_decode.h:110`, `kolibri_decode.h:108`,
`qwen4exp_decode.h:132`, `qwen4exp_serve.h:25`, `pipeline_args.h:102`, `pipeline_serve.h`, `dflash_args.h:66-73`,
`max_len.h:108`; `src/cli/CMakeLists.txt` (`b70_link_prefill` for b70-decode / b70-serve).

**Tests.** sycl-only: `gemm_batched_test`, `prefill_backend_equivalence_test`, the `sycl-tla` registrations of
`prefill_gate_test` / `prefill_consistency_test` / `prefill_determinism_test`, `prefill_smoke_test` sections 2-7
(G0_BITWISE_RE member). Using sycl-tla as the ORACLE of an L0 kernel: `pf_gemm_test` (spec 2.1 bar 1, bitwise),
`attn_chunk_test` (its sycl grid). Refusal rows: `cli_reject_{k2,kolibri,qwen4exp,bolt}_prefill_sycl`,
`cli_reject_pipeline_{prefill,bench_prefill}_sycl`, `cli_reject_serve_{pp,qwen4exp_prefill,bolt_prefill,dflash}_sycl`;
host tests `pipeline_args_test`, `dflash_engine_host_test`, `memory_plan_test`, `pipeline_prefill_plan_test`,
`bolt_serve_args_test`. L0-only tests that sit inside `if(B70_PREFILL_ENABLED)` only for the guard: `pf_int8_test`,
`pf_flash_attn_test`, `linear_i8_test`, `gdn_conv_test`, `gdn_wy_test`, `gdn_chunk_test` (+ selector rows); and ~40
`b70_link_prefill` calls under the guard.

**Probes (g++).** sycl-tla `gemm_bf16` as the C2 control: `probe_pf_gemm`, `probe_dequant_overlap`,
`probe_dequant_slab`. Guard-only (no SYCL use): `probe_pf_gemm_slab`, `probe_fused_dequant_gemm`, `probe_w4a8`,
`probe_w4a8_gsweep`, `probe_w8a8`, `probe_flash_attn`, `probe_tail`, `probe_writethrough`, `probe_prefix_s12`,
`probe_tie_judge`, `probe_verify_m`, `probe_mtp_steps`, `probe_batch_step`, `probe_decode_attn`, `probe_lm_head_l1`.

**Tooling / docs.** `tools/mac/l0_syntax.sh` (src/sycl + b70_sycl_probe skip), `tools/mac_check.sh:13,149`,
`tools/box.sh:11,26` (build-nosycl), `tools/bench_decode.sh:20`, `tools/toolcall/engine_generate.sh:7`;
`stages.sh` r13.cli (796-801), r15.k1 (956-959), r23.host / r23.reject (1457-1471), r26.reject (1831-1833),
r27.reject (1930-1932), r33.reject, r34.reject, r61.reject, r62.reject; README.md 109-113, 177-178, 254, 324, 347;
46 docs files name sycl-tla (older specs / plans stay as written; only live docs change).

**Defaults today -> after.**

| Model / path | today | after | output changes? |
|---|---|---|---|
| Qwen3.8, Agnes, Ornith (runtime::Engine, `default_prefill_backend()`) | l0-int8 | l0 | yes (prefill GEMMs bf16 instead of h8) |
| DFlash on Qwen3.8 (follows Engine) | l0-int8 | l0 | yes |
| `--pp 2` (pipeline_args.h, pipeline_serve.h, pipeline_engine.h, b70_serve PipeServe) | l0-int8 | l0 | yes |
| `memory_plan.h` PrefillPath default | l0-int8 | l0 | plan sizes (slab instead of Int8State) |
| K2-Horizon, Bolt, Kolibri-1, Qwen3.8-Flash-Next | l0 | l0 | no |

Tests that take `default_prefill_backend()` when unpinned: `prefill_gate_test` (argv[6]), `prefill_consistency_test`,
`prefill_determinism_test`, `prefill_split_test`, `prefix_gpu_test`.

## 2. Operator rulings (2026-10-27)

1. **The oneDNN probe** gets its own small stand-alone icpx build, `tools/probe/onednn/CMakeLists.txt`. It makes its
   own `sycl::queue` (no `Context::sycl()`, no libb70_prefill.so) and links the engine build's loader / model / l0
   archives. The engine build has no SYCL / icpx left at all. The stand-alone build is not part of mac_check or the box
   rows. To build it on the box, after the engine build in `build/`:
   `cmake -S tools/probe/onednn -B build-onednn -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx
   -DB70_BUILD_DIR=$PWD/build && cmake --build build-onednn`.
2. **The L0 GEMM's oracle:** `pf_gemm_test` compares against a host reference (fp64 accumulate of the same bf16
   inputs) at the same shapes, with a stated cos / max-ulp tolerance derived from the reference's own error and written
   in the test. `attn_chunk_test` drops its sycl-tla part (flash vs composed and the host reference stay).
3. **The other SYCL probes** (probe_interop, probe_prefill_gemm, probe_prefill_attn, probe_gemm_int4_mixed,
   probe_gdn_chunk, probe_pf_gdn_xe2) and `src/sycl/vllm_shim` are deleted. probe_pf_gemm, probe_dequant_overlap and
   probe_dequant_slab lose their sycl-tla comparison arm and keep the rest.
4. **Defaults:** l0 for every model, even if slower; l0-int8 stays opt-in. The box report records (not a gate) the l0 vs
   l0-int8 prefill rate for Qwen3.8 at pp2048 and pp8192: interleaved pairs after warm-up, median of 3.
5. **G0 / golden diffs** are reported, never accepted here.
6. `--prefill-backend sycl-tla` is an unknown value like any other; no dedicated test, stage or note; older specs and
   plans are not touched.

## 3. Tasks

1. CMake: delete `cmake/prefill.cmake` (or slim it, §2.1), `src/sycl/` (per §2.1/§2.3), the guard and every
   `b70_link_prefill` call (L0-only tests / probes become unconditional), `host_only.cmake`'s check, the preset entry.
   Gate: fresh configure in a new build dir and with `-U 'B70_PREFILL*'`.
2. Runtime: drop `SyclTla` and every sycl arm; `default_prefill_backend()` -> L0 (one definition, no absent twin);
   pipeline / memory_plan / PipeServe defaults -> L0. `parse_prefill_backend` accepts `l0` and `l0-int8` only;
   `sycl-tla` is an unknown value like any other ("--prefill-backend expects l0 or l0-int8, got 'sycl-tla'"), with no
   special note (operator, 2026-10-27).
3. CLIs and help text; the per-model refusals lose their sycl-tla clause.
4. Tests: delete the sycl-only tests and registrations and every `*_sycl` reject row (no dedicated sycl-tla test; the
   existing generic bad-value row, `--prefill-backend cuda`, stays with its new message); host tests drop their SyclTla cases;
   `prefill_smoke_test` keeps its L0 sections (its G0 output changes - listed for the operator).
5. Tooling and docs: stages.sh (the sycl reject rows / xfails go away), box-validation-queue.md, README.md,
   docs/README.md and the live docs, CLI help, l0_syntax.sh, mac_check.sh, box.sh, bench_decode.sh. Older specs and
   plans are not touched (operator, 2026-10-27).

## 4. Gates

- `tools/mac_check.sh --base main` PASS; every changed / removed command line listed.
- Box (own tree, `gpu_lock.sh`): build -j44 from a fresh build dir; r13.cli; Qwen3.8 and Ornith l0 / l0-int8 prefill
  rows; the changed reject rows; G0 (`g0.bitwise`) and the Qwen3.8 / Agnes / Ornith golden rows for the l0-int8 -> l0
  default move - a G0 diff is reported, never accepted (the operator's call).

## 5. As built (2026-10-27)

- Tasks 1-5 done on branch `drop-sycl-tla`. The engine build configures from scratch with no `B70_PREFILL*`, icpx or
  SYCL cache entry (a fresh host-only configure has none; the box tree is new, so its first build is from scratch).
- Beyond §1: `PrefillScratch::dequant_buffer()` and `PrefillScratchSizes::dequant` (the sycl-tla bf16 scratch) are
  gone; `is_l0()` is gone (both backends are L0); `step_chunk_gemms` / `step_chunk_waits` are gone; the per-model
  refusals and the pipeline checks lose their sycl-tla branch. `runtime/prefill/dequant.cc` (`dequant_to_bf16`, the
  whole-matrix `pf_dequant_tile` walk) stays: L0 code, no runtime caller left - a follow-up may delete it.
- `pf_gemm_test`: per element `|got - ref| / ulp32(sum |a b|) <= K` (the fp32 summation bound for any order) and per
  case cos >= 0.999999, on a row sample (both ends of every 64-row block) and every column; the silu and determinism
  checks stay device against device, bitwise. `attn_chunk_test`'s composed grid runs on the L0 backend.
- The default move also moves derived memory: `batch_rows_test`'s spec 13 §9.1 layouts lose Int8State (Qwen3.8
  0.087 GB, Agnes 0.099 GB of shared scratch; the largest batches are unchanged: Qwen3.8 MTP 4, Agnes MTP 2 /
  `--mtp off` 4); `memory_plan_test`'s spec 6 line pins the l0-int8 path it was measured on.
- `--pp-backend` is not accepted by any CLI: it is refused naming `--prefill-backend` (`cli/renamed_flags.h`).
- Box: row 76 (r76.tests, r76.speed) in stages.sh and the queue; r23.reject is gone (it held only the sycl refusal).
- The stand-alone oneDNN probe build: `tools/probe/onednn/CMakeLists.txt` (§2.1 has the command); built on the box
  2026-10-27 with icpx 2026.2 and oneDNN 2026.1 (configure + build clean, libs resolve; not run).
- On the box the SYCL component was already off on main: `B70_ICPX` pinned 2026.1, which is no longer installed
  (2026.2 is), so `B70_PREFILL=AUTO` resolved to OFF - main's box build had no sycl-tla backend before this branch.

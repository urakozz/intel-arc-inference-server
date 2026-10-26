# Drop the sycl-tla prefill backend (operator, 2026-10-27)

**Ruling:** "Drop sycl-tla backend. l0 should be the default one, l0-int8 optional." Direct GPU calls (our Level Zero /
OpenCL C kernels) over library layers. oneDNN is NOT part of this plan: `tools/probe/probe_gemm_onednn_int4.cc` stays.

**Status (2026-10-27): DRAFT - BLOCKED on the operator (§2).** Inventory done (§1); nothing deleted. The inventory found
three things outside the sycl-tla backend that depend on the SYCL component; each needs a ruling before Task 1.

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

## 2. Open (operator) - blocking

1. **The oneDNN probe.** `probe_gemm_onednn_int4` is built only by the icpx sub-build and takes its `sycl::queue` from
   `Context::sycl()` (libb70_prefill.so's SyclSide). Removing the icpx component unbuilds it. (A) keep a slim icpx
   sub-build: `sycl_side.cc` + `Context::sycl()` + this one probe, no sycl-tla, no runtime backend; (B) remove the
   component, keep the .cc in the tree unbuilt (or move its build to a stand-alone icpx CMake under tools/probe).
2. **The L0 GEMM's oracle.** `pf_gemm_test` (spec 2.1 bar 1) and `attn_chunk_test` compare L0 bitwise to sycl-tla.
   Proposal: `pf_gemm_test` against a host fp32 reference (fp64 accumulate, bound by the bf16 product error) at the same
   shapes; `attn_chunk_test` drops its sycl grid (flash vs composed and the host reference stay).
3. **The other SYCL probes** (probe_interop, probe_prefill_gemm, probe_prefill_attn, probe_gemm_int4_mixed,
   probe_gdn_chunk, probe_pf_gdn_xe2 + vllm_shim) and the C2 arms of probe_pf_gemm / probe_dequant_overlap /
   probe_dequant_slab. Proposal: delete the SYCL probes and vllm_shim (records stay in docs and git); drop the C2 arm
   of the three g++ probes (P-A / P-B already recorded), or delete those three probes.

## 3. Tasks (after §2)

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

# MoE-model prefill, phase 3 - the route on the matrix engine, the expert GEMM's conversion, the shared expert, the attention tile (Ornith, then Flash-Next)

**Status (2026-11-12): CLOSED.** Landed: Tasks 2 (router + a||b on pf_gemm, slabs made once per engine), 5
(combine + next norm, bitwise), 7 (32-row flash attention for Ornith only, then the output gate fused into its
epilogue, bitwise), and the pipeline-parallel launch-count fix for MoE models. Dropped by their probes: Tasks 3, 4, 6.
Not done: Task 8 (Flash-Next binding; waits for Flash-Next work). Record (Task 9, busy box against main's build): 4k
`l0` 8,463 -> 9,263 t/s, 16k +10 %; the quiet-box server record is the README's Ornith table and open-work's Closed
recently. 9,500 missed by ~11 ms. Follows phase 1 (`2026-11-12-moe-model-prefill-phase1.md`) and phase 2
(`2026-11-12-moe-model-prefill-phase2.md`), both closed. Main at 14dd0260.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The
> gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step**
> (signed, no trailers; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file
> at the worktree root (untracked) when you stop. Box jobs: every GPU job through `tools/gpu_lock.sh card 0` (speed) or
> `card <N>` (correctness), long jobs detached (`nohup setsid`, logs under `~/b70/logs/moe-pf-phase3/`), checked at most
> every 30 min with one blocking wait.

**Goal:** Ornith prefill on one card at pp4096 on `l0` (bf16) **>= 9,500 t/s** (~430 ms per 4096 ids; today 8,080 t/s,
507 ms). 16k recorded, not gated. Decode unchanged. `l0-int8` is parked (open-work item 16) and only has to stay as it is.

**Where the ideas come from:** the 2026-11-12 review of vllm-xpu-kernels, XeStrata, Strata, saber-engine, basalt,
oneDNN, sycl-tla and OpenVINO against our own launch map. Each task names its source. Every saving below is
**DERIVED, not measured**; Task 1 measures before anything is built.

## The budget (kernel ms at 4k `l0`; the wall is ~30 ms above the kernel sum: 507 vs 475 today)

| phase | today (phase 2's record, section 9) | after this plan (derived) | task |
|---|---:|---:|---|
| route (router + `moe_route`) | 27.9 | ~9 | 2 |
| GDN (a\|\|b 7, front 21, intra 23, scan 28, head 12) | 92.9 | ~87 | 2 (a\|\|b only) |
| dense linears | 85.7 | 85.7 | - |
| attention | 46.9 | ~30 | 7 |
| MoE sort + gather + expert GEMMs + combine | 203.6 | ~160-170 | 3, 4, 6 |
| norms | 16.5 | ~9 | 5 |
| head | 0.9 | 0.9 | - |
| **sum** | 474.6 | **~385-400** | |

That lands at ~415-430 ms wall, 9,500-9,900 t/s: **in reach only if Tasks 2, 4 and 7 all deliver**. Task 9 reports the
measured split against this table.

| task | what | 4k saving (derived) | numerics | source |
|---|---|---:|---|---|
| 2 | router (N 272) and a\|\|b (N 128) logits by the dense DPAS GEMM `pf_gemm` | ~18-24 | judged | Strata, XeStrata, basalt run the router as a GEMM |
| 3 | the shared expert (C rows) through the dense slab GEMMs instead of 64-row items | ~8-11 | judged | our profile: its block is converted once per 64-row item |
| 4 | expert weights fed to DPAS as exact integers, the g64 scale applied to the accumulator | ~15-25 | judged | vllm-xpu-kernels' block-FP8 path; our W4A8 kernel's shape |
| 5 | MoE combine fused with the next layer's residual fold + norm | ~6-8 | bitwise | basalt / XeStrata `gr_write_norm_rs` |
| 6 | the expert GEMM reads its rows through `row_tok` (no gather copy) - only if Task 1 shows a gain | ~0-10 | bitwise | saber `moe_grouped.hip` |
| 7 | flash attention with more query rows a sub-group | ~15-20 (16k: ~-250) | bitwise if the per-row key order is kept | vllm-xpu-kernels 256-row Q tile; sycl-tla 45-78 % of peak |

**Not in this plan, and why:**
- Strip-ordered work-group numbering (operator 2026-11-12: removed). XeStrata uses it for its dense GEMM only, at
  N >= 10240; ours walks 1024-column slabs already and sits at 75 % of peak.
- bf16 instead of fp32 stores for the linears' outputs: measured a wash on 2026-09 (`pf_gemm.cl`'s note: the epilogue is
  bound by the store-message count, not the bytes).
- Software-pipelining the int4 conversion ahead of its DPAS: saber measured neutral to -5 %; our 2026-09-23 finding is
  that conversion and DPAS time add.
- More scan work-groups: phase 2 Task 4 (128 work-groups fill every Xe-core's thread slots).
- Anything on `l0-int8`: open-work item 16.

**Order and why:** measure first (Task 1: five probe arms and one correctness check). Then the bitwise one (Task 5).
Then the judged MoE tasks, smallest first (2, 3, 4), each gated alone so a judge miss has one cause. Task 6 only by its
probe. Task 7 touches only the attention files and may run in a second worktree beside Tasks 2-6. Flash-Next (Task 8),
the record (Task 9).

**Stop rules:**
- A task whose Task 1 arm misses its pre-registered speed gate does not start; the plan reports its budget row missed.
- A bitwise gate that fails stops its task until the first differing value is explained.
- A judged task that misses the f32 judge on `l0` stops and goes to the operator with the row, its f32 margin in bf16
  ulps and the twin's verdict (the margin-table form of phase 1's record).

## Dependencies

- Main at or after 14dd0260. Branch `moe-pf-phase3`, own worktree (`.claude/worktrees/moe-pf-phase3`); a second agent on
  Task 7 takes `moe-pf-phase3-attn`. Tasks 2-6 all touch `src/runtime/prefill/moe.cc`: one agent, in order.
- Box data: the Ornith int4 checkpoint and its f32 references (`oracle_ornith_f32`, the long32k reference
  `prefill_split_ornith` uses); Flash-Next's synthetic checkpoints and Intel's at `--layers 18`.
- Cards: both until the Flash-Next quantisation starts on card 1 (Saturday 2026-11-14); after that card 0 only.

## Global constraints

- **Additive kernels.** New entries go beside the old ones; every existing binary and command line stays byte for byte
  until Task 9 (`tools/kernel_cmdlines`: + only; G0's `g0.sha` shows every pre-existing binary identical).
- **Who moves.** Ornith binds the new kernels on `l0`. `l0-int8` keeps today's path in every task (the DPAS router
  missed its judge there in phase 2). Flash-Next binds in Task 8. Qwen3.8 changes only in Task 7 and only if the arm is
  bitwise; otherwise it stays on today's attention. K2, Kolibri and Bolt are untouched (host and binaries; G0).
- **A/B:** no env switch and no user-facing flag. A control build of main against the branch's build
  (`tools/probe/pf_phase2_ab.sh`, `pf_phase2_bitwise.sh` - reused as they are).
- **Numerics.** Bitwise tasks: the extended G0 compare (`prefill_gate_ornith_{l0,l0-int8}_test`, `_c16`,
  `ornith_prefill_{l0,l0-int8}_test`, `--baseline main`). Judged tasks: the f32 judge with the 1-bf16-ulp near-tie rule
  on every Ornith `l0` prefill gate - the engine's determined-row misses and differing MoE sets no more than the bf16
  golden's against f32, both counts printed; `prefill_split_ornith_l0`'s logits cosine distance <= 2x the bf16 twin's.
  GDN state stays fp32. Deterministic: no atomics, fixed reduction orders, replay bitwise.
- **Prefill-vs-decode routes.** Tasks 2-4 move last bits of what the router sees or of the residual. The
  prefill-vs-decode route diagnostic of `ornith_prefill_l0_test` is printed before and after each; it is recorded, the
  f32 judge decides.
- **Timing:** interleaved pairs after a warm-up, median of rounds, card 0; each number labelled "quiet box" or "busy
  box"; probe ms are L0 kernel timestamps.
- **Launch counts:** `moe_chunk_launches` (`moe.cc`) and `kGdnChunkLaunches` stay the homes of the counts; the tests
  asserting them read them.

## Review focus

Conditions no single task's gate exercises, most likely first; each has its test in the task named.
1. A chunk whose length is not a multiple of 256 or of the GEMM tile (C = 37, C = 2600): padded rows must not reach the
   route, the combine or the residual. Task 2 (`pf_moe_test` at C 37), Task 3 (C 37 and 2600), Task 5.
2. A short chunk deep in a session (prefix cache hit, pos > 0, C <= 128): the small-chunk paths must still bind the old
   or the new kernels consistently with decode. Task 2 and Task 7 (`prefill_split_ornith_l0`, `attn_chunk_test`).
3. An expert with zero rows and an expert with more rows than one item (hot expert, > 64 rows). Task 4 (`pf_moe_test`).
4. The last layer: its combine has no next layer's norm, the final norm takes it. Task 5.
5. 4096-row and 2048-row chunk caps both (the memory plan picks either). Every task's speed and gate runs at both.

---

### Task 1: the probes and the gate-span check (Mac, then box)

**Files:**
- Modify `tools/probe/probe_route.cc`: arm `gemm` - the router's bf16 weight [272][2048] re-laid into `pf_gemm`'s B
  slab form, zero-padded to N 512, then `pf_gemm` at M 2048 / 4096, K 2048; the same for a||b at N 128 padded to 256.
  Control: `pf_gemv_rows` (MT16 TC2, what is bound). Printed: ms each, max |logit - control| and the count of rows
  whose top-8 set differs from the control's.
- Modify `tools/probe/probe_moe_prefill.{cc,cl}` (the phase 1 harness, real route dumps), three arms beside the chosen
  `F1.a8b2m1n8.c2x0o1`:
  - `c4`: B as `(float)(q - 8)` exactly in bf16 (no scale, no rounding), a zeroed fp32 tile per 64-k group and
    `acc = fma(tile, scale[group][column], acc)` after each group - `pf_moe_w4a8.cl`'s group loop with a bf16 A.
    Checked against a host oracle of the same order on two tiles' rows x every column.
  - `shd`: the shared expert's C rows out of the item walk: `pf_dequant_slab` + `pf_gemm` (SiLU epilogue) for gate||up
    and `pf_dequant_slab` + `pf_gemm` for down, A read from the chunk's rows in token order; the routed items alone in
    the item walk. Timed as the sum.
  - `o8`: A read through `row_tok` (one row read per row) instead of the gathered `xg`; the gather's own ms subtracted
    in the report. Bitwise the chosen arm.
- Modify `tools/probe/probe_flash_attn.{cc,cl}`: arms RPW 16 and 32 (query rows a sub-group; today 8), each with Q held
  in registers across key tiles (`QREG 1`) and without; C 4096 at pos 0 and C 4096 at pos 12288. Per arm: ms, and
  whether o / l are bitwise today's.
- Create `tools/probe/gdn_gate_span.py`: over `B70_PF_GDN_DUMP` dumps (Ornith layers 0 / 12 / 28; Flash-Next at
  `--layers 18`, layers 0 / 8 / 16), the largest |gl - gc_i| inside one 64-row sub-chunk, in nats, per layer.
- Modify `tools/box_validate/stages.sh`: row 85, stages `r85.probe`, `r85.span` (opt-in, timed).

- [ ] **Step 1 (Mac):** the arms, the script, row 85; `tools/box_validate.sh --dry-run --only r85` prints the commands;
  host tests green. Commit `probes: the router on pf_gemm, exact-integer expert weights, the shared expert as dense, the
  indexed A, flash attention row tiles (phase 3 Task 1)`.
- [ ] **Step 2 (box, card 0):** `r85.probe`, `r85.span`. Record `docs/probes/probe-prefill-phase3-2026-11-13.md`, the
  gates written into it before the run:
  - **Task 2 gate:** router + a||b on `pf_gemm` <= 0.35x the control per layer-chunk at C 4096 (control ~0.5 + 0.2 ms).
  - **Task 3 gate:** `shd` saves >= 0.15 ms a layer-chunk at C 4096 against the chosen arm (4.35 ms).
  - **Task 4 gate:** `c4` <= 0.88x the chosen arm at C 4096 and at C 2048, oracle-exact.
  - **Task 6 gate:** `o8` minus the gather saves >= 0.1 ms a layer-chunk at C 4096, bitwise.
  - **Task 7 gate:** the best arm <= 0.7x today's `pf_flash_attn` at both positions.
  - **Span:** a sub-chunk span above 80 nats on any layer is reported to the operator with the layer and the value
    (sly-radiance and saber measured their factorised scan wrong above ~160); it stops nothing here.

### Task 2: the router and a||b on the dense GEMM (Mac, then box) - JUDGED

Today `pf_gemv_rows` (`pf_route.cl:39`) computes 0.18 TFLOP of router logits in ~22 ms at 4k (~9 TFLOP/s; `pf_gemm`
runs at 138) and a||b in ~7. Phase 2's DPAS router (`d20de347`, reverted) was a 45-line kernel on the GEMV's tile
layout, one 16-column tile a sub-group - not the dense GEMM. Operator decision D3 of phase 2 (a DPAS router whose last
bits differ from decode's, if the f32 judge passes) covers this task; it passed on `l0` then.

**Files:** `src/kernels/prefill/pf_route.cl` (new entry `pf_route_slab`: router or a||b weight -> `pf_gemm`'s B slab
form, N padded with zero columns to a whole 256); `src/runtime/prefill/moe.cc` (router: `pf_route_slab` + `pf_gemm`
into the logits scratch, `l0` only), `src/runtime/prefill/step.cc` (a||b the same); `src/runtime/buffer_sizes.cc`
(logits pitch 512, a||b pitch 256, one [2048][512] bf16 slab); `src/kernels/moe.cl`'s `moe_route` and
`pf_gdn_conv.cl`'s `pf_gdn_gate` read their rows at the new pitch (a prefill-only build define if the pitch is not
already an argument - decode's binaries stay byte for byte); `tests/kernels/pf_moe_test.cc`,
`tests/prefill/pf_gemv_test.cc`.

**Interfaces - produces:** router logits fp32 [C][512] (columns 0..271 meaningful, as today: 256 experts, the shared
gate at 256); a||b fp32 [C][256] (columns 0..127).

- [x] **Step 1 (Mac):** tests first: `pf_moe_test` case `router_gemm` - logits within 1e-5 relative of the host's
  double-precision sum of the same bf16 operands, at C 2048 and C 37, pad columns never read by `moe_route`;
  `pf_gemv_test` case `ab_gemm` the same for a||b. Then the entry and the bindings. Host tests green. Commit.
- [x] **Step 2 (box, card 0):** `pf_moe_test`, `pf_gemv_test`, the judged Ornith `l0` set (`tools/probe/
  pf_phase2_gates.sh`), the `l0-int8` gates bitwise main's, `B70_PREFILL_PROFILE=1` 4k `l0`.
  - **Gates:** judged PASS on `l0`; `l0-int8` logs identical to main's; route row <= 10 ms and a||b <= 2 ms at 4k.
  - **Result (2026-11-12, card 1, busy box): PASS.** The weights are re-laid per chunk by `pf_route_slab` into a
    [2048][P] slab in xg's region (no new bytes), `tiled_gemm` runs them on `pf_gemm_T0`; `moe_route` /
    `pf_gdn_gate` read pitch 512 / 256 from their own builds (`_R512`, `_AB256`; the old binaries md5-identical).
    Unit: slab exact, max err 7.7e-8 of sum|w x|, the consumers bitwise the old ones with NaN pads. Judged: gate 90/90
    determined rows, `_c16` 89/90 (cjk 15, the accepted row), golden 90/90; split logits cosine distance x0.54-0.79
    of the bf16 twin's, median x0.99-1.02; R2 1068 differing sets (base 1098, bf16 golden 1140). Route diagnostic vs
    decode: 4040 equal / 1410 near-tie / 895 differing (base 4027 / 1421 / 927). `l0-int8` logs identical (two
    differ only in the tree path). 4k profile, 3 interleaved pairs: route 8.4 ms (base 27.2-27.7), a||b 1.9 (7.5),
    wall median 501.5 ms vs 512.2, launches 1777 vs 1707.
  - **Follow-up (2026-11-12, rebased on be5f5418, card 1, busy box): the slabs made once.** `make_route_slabs`
    (Engine / PipelineEngine `prepare_prefill`, l0) builds every router / a||b B slab per weight in
    `PrefillScratch::gemm_slab`; `runtime::route_slab_bytes` adds them to both memory plans (Ornith 115,343,360 B,
    a stage its own layers). Per-chunk launches back to the base's (moe_route 80, ab_proj 30 at 4k); the 70 slab
    launches run once per engine, inside the first prefill. Unit, the five judged gates (same counts as above) and
    the four `l0-int8` logs identical again. 4k, 3 interleaved pairs against be5f5418: route 8.5-8.7 ms (27.1-27.6),
    a||b 1.9 (7.4-7.5), kernel sum 450.2-451.9 (474.3-475.0), wall median 477.7 ms (8,574 t/s) vs 520.8 (7,865).

### Task 3: the shared expert through the dense GEMMs (Mac, then box) - JUDGED

The shared expert has C rows in token order but runs as 64-row items of `pf_moe_q4gemm`, so its block is converted
C / 64 times. As dense it needs no sort entry and no gather.

**Files:** `src/runtime/prefill/moe.cc` (the shared block: `pf_dequant_slab` + `pf_gemm` with the SiLU epilogue ->
`h_sh` bf16 [C][512], then `pf_dequant_slab` + `pf_gemm` -> `y_sh`; `l0` only); `src/kernels/prefill/
pf_moe_sort_mw.cl` (a build define that leaves the shared expert's tiles out of the item and tile tables);
`src/kernels/prefill/pf_moe.cl` (`pf_moe_combine` reads the shared row from `y_sh` at the token's row);
`src/runtime/buffer_sizes.cc` (`h_sh`, `y_sh` in `moe_prefill_layout`; `xg` / `h` shrink by the shared rows);
`tests/kernels/pf_moe_test.cc`.

- [ ] **Step 1 (Mac):** confirm by reading that the shared block's int4 layout is one `pf_dequant_slab` accepts (layout
  0 or 1, gate||up in the 16-column interleave the SiLU epilogue expects); if it is not, stop and report - no new
  dequant kernel in this task. Test case `shared_dense`: the combine's output against `pf_moe_ref` within the judged
  band at C 2048, 2600 and 37. Then the code. Commit.
- [ ] **Step 2 (box, card 0):** `pf_moe_test`, the judged Ornith `l0` set, `l0-int8` bitwise main's, the profile.
  - **Gates:** judged PASS; MoE expert GEMMs + gather <= today's - 8 ms at 4k `l0`.

### Task 4: exact-integer expert weights, the scale on the accumulator (Mac, then box) - JUDGED

Today each weight becomes `rne_bf16((q - 8) * scale)` in registers (~5.5 ALU instructions a value, ~65 of the 170 ms).
`q - 8` is exact in bf16, so the DPAS can take it as is and the group's scale multiplies the group's partial sum once.
The weights the model sees are then the checkpoint's exactly - closer to the f32 reference than today's rounded
products, but different bits.

**Files:** `src/kernels/prefill/pf_moe_q4gemm.cl` (a `CVT 4` form: Task 1's `c4` arm, the epilogue unchanged);
`src/kernels/prefill/CMakeLists.txt` (new binaries at Ornith's two shapes, beside today's); `src/runtime/prefill/
moe.cc` (bind on `l0`); `tests/kernels/pf_moe_test.cc`.

- [ ] **Step 1 (Mac):** test case `q4gemm_c4`: h and y against a host oracle that sums each 64-k group in the kernel's
  order and applies the scale per group, exact; an expert with 0 rows; an expert with 200 rows (4 items, the last
  short). Then the kernel form and the binding. Commit.
- [ ] **Step 2 (box, card 0):** `pf_moe_test`, the judged Ornith `l0` set, `prefill_split_ornith_l0`, replay bitwise,
  the profile at chunk 4096 and 2048.
  - **Gates:** judged PASS; expert GEMMs <= 0.88x Task 3's at 4k `l0`.

### Task 5: combine + the next layer's fold and norm in one launch (Mac, then box) - BITWISE

`pf_moe_combine` (`pf_moe.cl:260`) writes the residual; the next layer's `pf_res_norm` (`pf_norm_fused.cl:45`,
`S_PREV 0`) reads it straight back to write x.

**Files:** `src/kernels/prefill/pf_moe.cl` (entry `pf_moe_combine_norm`: one work-group a row; the combine's sums in
today's slot order, then `pf_res_norm`'s 20 group sums and finish in today's order, writing the residual and x);
`src/runtime/prefill/moe.cc` and `step.cc` (the layer's leading norm is skipped when the previous layer's combine wrote
x; the last layer's combine feeds the final norm the same way); `moe_chunk_launches` and the layer count in `step.cc`;
`tests/kernels/pf_moe_test.cc` (case `combine_norm`: residual and x equal `pf_moe_combine` then `pf_res_norm`, every
byte, C 2048 / 37).

- [ ] **Step 1 (Mac):** the test, the entry, the bindings, the launch counts. Commit.
- [ ] **Step 2 (box, card 0):** `pf_moe_test`, `tools/probe/pf_phase2_bitwise.sh` against main (Ornith `l0`, `l0-int8`,
  `_c16`), the launch-count tests, the profile.
  - **Gates:** bitwise; combine + norms <= today's - 6 ms at 4k `l0` (today 15.3 + 16.5).

### Task 6: the expert GEMM reads rows through `row_tok` (Mac, then box) - BITWISE - only if Task 1's `o8` passed

**Files:** `pf_moe_q4gemm.cl` (A base and row index from `row_tok`), `moe.cc` (no `pf_moe_gather` launch on `l0`; down
still writes y into `xg`), `buffer_sizes.cc`, `pf_moe_test`.

- [ ] **Step 1 (Mac):** test case `indexed_a` (h bitwise the gathered form's), the code. Commit.
- [ ] **Step 2 (box, card 0):** the bitwise compare; gather + expert GEMMs <= Task 4's - 4 ms at 4k `l0`, else revert.

### Task 7: flash attention, more query rows a sub-group (Mac, then box) - second worktree allowed

Today a sub-group is one q-head x 8 query rows and reloads Q every key tile (`pf_flash_attn.cl:116-152`, `RPW 8`,
`QREG 0`); each 64-key tile is read once per 8 query rows. 1.37 TFLOP in 38 ms at 4k is 20 % of peak.

**Files:** `src/kernels/prefill/pf_flash_attn.cl` and its CMake stanza (Task 1's winning arm as new binaries);
`src/runtime/prefill/attn.cc` (bind for Ornith; grid from the new RPW); `tests/kernels/attn_chunk_test.cc`,
`flash_long_test.cc`.

- [x] **Step 1 (Mac):** tests: o and l against `attn_ref.h` at C 4096 pos 0, C 300 pos 5000, C 16 pos 20000 (the
  split-K path keeps today's binary); bitwise today's when the arm kept each row's key order. Then the binding. Commit.
- [x] **Step 2 (box, card 0):** the attention tests, the Ornith `l0` set (bitwise compare if the arm is bitwise, the
  judge otherwise), `--kv-cache int8`'s prefill gates unchanged (they keep today's kernel), pp4096 and pp16384.
  - **Gates:** its numerics kind; attention <= 33 ms at 4k and <= 0.7x today's at 16k (`l0`).
- [x] **Step 3 (only if bitwise):** bind for Qwen3.8 too; its G0 stays PASS, its pp16384 not slower.
- **Result (2026-11-12, card 1, busy box; branch `moe-pf-phase3-t7`): bitwise, Ornith only.** Bound: the probe's arm
  `pfao_KT64_R32_H8_Q1` as `pf_flash_attn_r32_Q16KV2` (KT 64 RPW 32 HPW 8 QREG 1 EXP2 1, grid ceil(C / 32)), both
  backends; split-K, int8 KV and the TP card keep today's binaries. Operator ruled 2026-11-12 to take it though it
  missed Task 1's 0.7x bar, because it is bitwise.
  - `attn_chunk_test` PASS: 0 differing words against today's binary at C 4096 pos 0, C 300 pos 5000, C 37 pos 0.
    `flash_long_test` PASS. `pf_phase2_bitwise.sh` against a7a75e61: the Ornith `prefill_gate_*` (l0, l0-int8, c16)
    identical; `ornith_prefill_l0` / `l0-int8` differ only in the f32 reference's tree path.
  - Ornith `l0` 4k (`B70_PREFILL_PROFILE=1`, 3 interleaved pairs): attn_flash 38.0 -> 29.3 ms (<= 33: PASS); wall
    within noise (526-531 vs 520-528 ms). pp16384: 6948 -> 7272 t/s (+4.7 %).
  - **Qwen3.8 not bound (Step 3 fails):** with the same form at its HPW 6 (`pf_flash_attn_r32`, bitwise on the attention
    test and its prefill gates) pp16384 fell 1563 -> 1544 t/s (-1.2 %, 2 interleaved pairs) and the profiled attn_flash
    median rose 1243 / 1267 -> 1493 / 1500 ms (+19 %). The binary was removed; Qwen3.8 keeps `pf_flash_attn`.

### Task 8: Flash-Next on the phase 3 kernels (Mac, then box)

Router (K 2560, N 528 -> padded 768) and a||b by Task 2; the shared expert by Task 3; `CVT 4` at Flash-Next's shapes
(E 512, H 2560, I 640) by Task 4; Task 7's attention does not apply (its attention is sparse). Its combine is
`q4_hc_combine_norm` already: Task 5 does not apply.

**Files:** `src/runtime/qwen4exp/qwen4exp_prefill.cc`, `qwen4exp_plan_test`,
`tests/kernels/qwen4exp_pf_kernels_test.cc`, CMake stanzas.

- [ ] **Step 1 (Mac):** the bindings and the tests' new arms. Commit.
- [ ] **Step 2 (box):** `r33.k1`, `r33.prefill`, `r33.golden`, `r33.speed`'s pp4096 at `--layers 18` interleaved
  against main. `qwen4exp_prefill_intel_test` compares routes bitwise against a recorded engine trace: it will differ
  by construction; report its differing-route count and leave the re-made reference to open-work's Flash-Next row.
  - **Gates:** synthetic and golden gates PASS; pp4096 at `--layers 18` >= 1.05x main's.
  - **Result (2026-11-12, branch `flashnext-pf-router`):** the router (N 528 -> pitch 768) and a||b (N 128 -> 256) on
    pf_gemm_T0, the weight re-laid into the walk's existing slab buffer each chunk (no new memory; Ornith's persistent
    slabs would cost ~236 MB here); q4_route `_P768`, pf_gdn_conv `_AB256` (Qwen3.8's binaries unchanged); MoE 10
    launches, a||b 2. Synthetic gates PASS (qwen4exp_pf_kernels, prefill synth ours / intel, golden synth ours /
    intel), plan and names tests PASS. Intel's checkpoint at `--layers 18` (card 0, interleaved, median of 3): pp4096
    872.0 -> **848.9 ms** (4,697 -> 4,825 t/s, -23 ms, +2.7 %; under the 1.05x bar - the router is a smaller share at
    18 layers). `qwen4exp_golden_prefill_intel_test` fails on both builds (its old gap rule, as on main): determined
    rows 71 / 95 against main's 74, logits worst cosine higher on all three prompts (0.956 / 0.980 / 0.978 against
    0.947 / 0.961 / 0.973), 20 sets beyond a near-tie on both. Only the f32 judge settles it: waits for the re-made
    Flash-Next f32 reference (open-work, Waiting). `qwen4exp_prefill_intel_test` differs from its recorded trace, by
    construction. A Fable review found no runtime defect; its one test-wiring defect is fixed; a second found none.
    Expected kernel diff against main: +4 (`pf_route_slab_K2560_N528_P768`, `pf_route_slab_K2560_N128_P256`,
    `q4_route_M1_E512_T10_N528_L256_P768`, `pf_gdn_conv_AB256`) / -2 (`pf_route_ab_D2560`,
    `pf_route_router_K2560_N528`, no longer bound): G0's compare needs `G0_ALLOW_REMOVED=1` and `tools/box_check.sh`
    `--allow-kernel-changes` for the removals.

### Task 9: the record, the clean-up, the final gate (box idle, then Mac)

- [ ] **Step 1 (box, card 0, idle):** interleaved pairs, main's build against the branch head, median of 3,
  `b70-decode --bench --tg 32`: Ornith pp4096 and pp16384 on `l0` (and `l0-int8`, recorded); then
  `B70_PREFILL_PROFILE=1` for the phase table against the budget above.
  - **Gate (final):** pp4096 `l0` **>= 9,500 t/s**. A miss keeps open-work item 14 open with the measured table.
- [ ] **Step 2 (box):** decode unchanged (`g0.sha` on every decode binary, `ornith_decode_test` PASS, plain decode at
  depth 4096 within 1 % of main's, interleaved); G0 against `--baseline main`: Qwen3.8 bitwise unless Task 7 Step 3
  bound it, every K2 / Kolibri / Bolt binary identical; Ornith rows r13 / r10 and Flash-Next's r33 PASS.
- [ ] **Step 3 (Mac):** delete the losing probe arms' production twins and the binaries nothing binds any more;
  BENCHMARKS' Ornith and Flash-Next rows, `docs/12-kernels.md`'s MoE and attention sections, open-work item 14 ->
  Closed recently (or its measured gap). Commit; offer the ff-merge to main.

**Which tasks need the box:**

| task | Mac | box | numerics |
|---|---|---|---|
| 1 probes | step 1 | step 2 (card 0) | per arm |
| 2 router, a\|\|b | step 1 | step 2 | judged |
| 3 shared expert | step 1 | step 2 | judged |
| 4 exact-integer weights | step 1 | step 2 | judged |
| 5 combine + norm | step 1 | step 2 | bitwise |
| 6 indexed A (conditional) | step 1 | step 2 | bitwise |
| 7 attention | step 1 | steps 2-3 | bitwise or judged, per arm |
| 8 Flash-Next | step 1 | step 2 | as the tasks it binds |
| 9 record | step 3 | steps 1-2 (idle) | - |

## Operator decisions

- **D1 (before Task 2):** phase 2's D3 ruling (a DPAS router is allowed on `l0` if the f32 judge passes) is taken to
  cover Task 2. Say so if it does not.
- **D2 (if a judged task misses on `l0`):** the row goes to the operator in the margin-table form; the near-miss rule
  is `l0-int8`'s only and is not applied to `l0` without a ruling.
- **D3 (after Task 7):** if the winning attention arm is not bitwise, whether Qwen3.8 moves to it (its own gates, G0's
  baseline moves) - recommendation: yes, as its own small task, for the 16k and longer prompts.
- **If Task 9 misses 9,500:** what is left is the delta-net scan (28 ms) and front (21 ms) and the dense GEMMs (86 ms at
  75 % of peak); the measured table says which.

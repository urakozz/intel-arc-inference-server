# Spec 20d - Kolibri-1 prefill: grouped MoE over 384 experts, windowed flash attention

**Status (2026-10-06): Tasks 1-4 built blind on branch `spec20d-kolibri-prefill` (Task 5's speed is box-only:
its stages are r26.speed / r26.p0)** - spec 20 §12 has what was built and where the build departs from
this plan (the bf16 slab takes the slab width as an argument; `pf_res_fold_K2560_SP1_G20` is not built for
Kolibri - nothing binds it; `prefill_split_kolibri_test` is `kolibri_prefill_test`'s split mode; the plan's
all-to-6 adversary is the sort's per-lane worst case, not the tile bound's - a separate adversary reaches
the bound). Box queue row 26.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `KolibriEngine::prefill` runs a prompt in chunks of 2048 positions: the attention linears through spec 5's slab + `pf_gemm` path (both arms of decision 2), flash attention with a 513-key window over the sliding ring and causal over the full layers' KV, the MoE block through spec 15d's grouped expert GEMM with Kolibri's router, its 384-expert sort and its combine; the chunk crosses the two cards once; KL2 and KL3 on the prefill path; the prefill speed rows.

**Architecture:** spec 20 §4 and §6 20d; spec 18 §11 is the closest precedent (K2's prefill: own sort, `pf_moe_gemm.cl` unchanged, experts dequantised to bf16 per chunk, decode's chains row for row). Per chunk on each device, one Level Zero list (no host wait inside it): embed (device 0) · per layer: input norm · q||k||v (slabs + `pf_gemm`) · `kol_attn_prep` at M = 2048 (q/k norm, RoPE in sliding layers, every row's k / v into the ring or the full KV) · `kol_pf_flash_attn` · o_proj (slabs + `pf_gemm`) · sandwich `pf_res_fold ..._Z` + `kol_post_add` · post-attention norm · router (`pf_gemv_bf16`, fp32 logits) · decode's `kol_route` on grid (1, C) · `kol_pf_sort` · `kol_pf_gather` · per weight batch `kol_pf_dequant_gu` + `pf_moe_gemm` (SiLU) · per batch `kol_pf_dequant_dn` + `pf_moe_gemm` · `kol_pf_moe_combine` (into `mo`) · `pf_res_fold` SP0 + `kol_post_add`. The chunk's residual rows and their norm sums cross from device 0 to device 1 once, by spec 16b's `copy` hand-off at chunk size (no overlap: spec 16c's chunk pipeline is a later lever, not a dependency); the head runs on the last row with decode's binaries.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec20-kolibri-1-design.md` (§4, §5 KL2-KL3, §6 20d). Facts: `docs/probe-kolibri-2026-10-05.md`. Plan 20c (`docs/superpowers/plans/2026-10-06-spec20c-kolibri-decode.md`) and its as-built §11 in the spec. Precedents: `src/kernels/k2/k2_pf_moe.cl`, `k2_pf_attn.cl`, `k2_pf_linear.cl`, `src/runtime/k2/k2_prefill.{h,cc}`, `k2_prefill_engine.cc`, `tests/kernels/k2_pf_ref.h`, `tests/prefill/prefill_split_test.cc`, spec 18 §11.

## Dependencies and branch points

- **Plan 20c merged** (the descriptor, loader, decode kernels, `KolibriEngine`, the synthetic checkpoints and golden sets). Its names are used as 20c defines them; if 20c's as-built §11 renamed anything, follow §11.
- **Spec 16b is merged** (`0792415..dd2fc87`) and 20c Task 6 built Kolibri's two-card decode on its pieces. The two-card prefill (Task 3 Step 4) reuses the same ones - no new mechanism: a second `runtime::PipelineLink` in `copy` mode built with 20c's descriptor-free constructor over `runtime::pp_landing_layout(kPfC x 2560 x 2, 20 x kPfC x 4)` (10.5 MB of rows + 160 KB of sums: 16b's two regions at chunk size), its `l0::SyncEvent`, `CmdList::barrier_signal` / `wait_event`, `Fence::wait_for`, and the split 20c's `pp_split` chose (`runtime::pp_balance` over `pp_layer_bytes`, now with the prefill scratch in both devices' fixed bytes). The chunk always crosses by `copy`, whatever `--pipeline-handoff` says for decode: `pp_handoff.cl`'s peer kernels are one 256-lane work-group, sized for a 5 KB row, not 10.5 MB (recorded in the as-built section). The switches are 16b's as renamed on 2026-10-06 - `--pp N` / `--pipeline-parallel-size N` (Kolibri's default 2), `--pipeline-split auto|N`, `--pipeline-handoff copy|peer` - and the bench's prefill flags `--prefill-length N`, `--prefill-chunk C`, `--prefill-backend B`. Everything else runs on one card with the synthetic checkpoints (`--pp 1`). Spec 16c (the overlapped chunk pipeline) is **not** needed; adopting it is Task 5's recorded lever.
- **Decision 2 (attention int4 vs bf16):** both arms built. Int4: `k2_pf_dequant_slab` (layout 0) slabs; bf16: `kol_pf_bf16_slab` (Task 2: `pf_bf16_slab.cl` cannot write o_proj's 512-column tail). Both feed the same `pf_gemm_T0` binary.
- **Spec 20b** (real checkpoint): real-weight gates SKIP 77 until it and `oracle-out-kolibri/` exist.
- **Backend:** `l0` only. `l0-int8` (spec 5's h8) rotates in 1024-k Hadamard blocks and Kolibri's hidden is 2560; sycl-tla has no Kolibri walk. Both refused by name (as K2).

## Global Constraints

- Branch `spec20d-kolibri-prefill` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked.
- **K0:** every existing binary keeps its name and command line (`tools/kernel_cmdlines` additions only; sha256 on the box); no existing `.cl` edited; 20c's decode list and launch counts unchanged; Qwen3.8 / Agnes / Ornith / K2 suites unchanged.
- One weight format (int4 g64 sym `auto_round:auto_gptq`, bf16 for the rest); no asymmetric or third-party path.
- **Determinism:** no atomic in any sum that feeds a result; the sort is one work-group in (token, slot) order; a grouped-GEMM row's output is independent of the other rows of its tile; the combine sums in ascending expert id; flash key tiles start at an absolute multiple of KT, and a wholly masked tile adds exact zeros. Prefill replay bitwise; a split prefill bitwise at multiples of 64.
- Kolibri-only kernels in `src/kernels/kolibri/`, `kol_*` names with every shape define.
- Every number measured, or marked derived / estimated / proposed. Box: `flock ~/b70-gpu.lock` (once for both cards), detached, polled; interleaved pairs, median of 3, `uptime`; `-j44`.
- Mac checks `tools/mac_check.sh --base main --kernels`. No `rm -rf`. Signed commits on the branch; no merge, no push.

## Review Focus

1. **The window inside one chunk and across the chunk boundary.** Row t of a chunk starting at position c0 sees keys `(c0 + t - 513, c0 + t]`: some from the ring (positions before c0, written by the previous chunk) and some from this chunk (written by this chunk's `kol_attn_prep` first). Task 1's host reference and Task 2's kernel test cover c0 = 0, 1000 and 4096 + 300 (the window wrapped in the ring) with rows 0, 1, 511, 512, 513 and C - 1; the ring holds 4096 slots, so a chunk of 2048 never overwrites a key a row of it still needs.
2. **384 experts in the sort.** Ids up to 383 do not fit the byte staging of `k2_pf_sort` (`PF_E <= 255`); Kolibri's sort stages them as `ushort` and gives each of 256 lanes two experts. Task 1's adversaries: every token to the same 6 experts (the tile bound's worst case), experts 255 / 256 / 383 (the lane boundaries), an expert with exactly 32 rows (one full tile, no padding) and one with 0.
3. **The bf16 shared expert inside the grouped GEMM.** Block 384 of each weight batch is the shared expert, copied from its bf16 tiles (not dequantised), its rows the C tokens in order; the combine adds it after the routed sum with one rounding. Task 2 tests it alone (routed weights zero).
4. **Prefill against decode.** The KV rows and ring slots a prefill writes, and its route rows, against 20c's M = 1 decode fill of the same tokens: route sets equal except near-ties; KV cosine bars proposed as spec 18c's (rows >= 0.999, median >= 0.9998, p01 >= 0.99), printed per layer.
5. **Continuation.** A prompt prefilled as two calls (split at 64, 1000, 2048, 4097) equals one call: bitwise at multiples of 64, within the split test's bars elsewhere; on two cards the same as on one.

---

### Task 1: the host reference for the prefill kernels

**Files:**
- Create: `tests/kernels/kolibri_pf_ref.h`, `tests/kernels/kolibri_pf_ref_test.cc`
- Modify: `tests/CMakeLists.txt` (a block `# ==== Spec 20d: Kolibri-1 prefill (labels kolibri;prefill) ==== (begin)` at the end)
- Test: `kolibri_pf_ref_test` (host)

**Interfaces:**
- Consumes: 20c's `tests/kernels/kolibri_ref.h` (decode chains: `route`, `moe_down`'s combine, `attn_prep`, eager attention, `ring_rows`), `kernels::kolibri::route` word indices.
- Produces (namespace `kolibri_pf_ref`, used by Task 2's kernel test and the Mac driver):

```cpp
struct Sorted { std::vector<uint32_t> hdr, tiles, row_tok, pair_row; uint32_t tiles_used; };
uint32_t tmax(uint32_t C);                                    // floor((C*6 + 384*31) / 32) + ceil(C / 32): 820 at C = 2048
Sorted sort(const uint32_t* route, uint32_t C);               // expert-major, ascending token, tiles of 32, shared block 384 last
void combine(const uint32_t* route, const Sorted& s, const uint16_t* y, uint16_t* mo, uint32_t C);   // == kolibri_ref combine per row
void flash_window(const float* q, const uint16_t* ring_k, const uint16_t* ring_v, uint32_t c0, uint32_t C,
                  bool sliding, double* o);                   // fp64, keys by ring_rows / linear rows, window 513
void eager_window(const float* q, const uint16_t* ring_k, const uint16_t* ring_v, uint32_t c0, uint32_t C,
                  bool sliding, uint16_t* o);                 // the reference's rounding points (scores twice, p once, o once)
```

- [ ] **Step 1: the failing test** `kolibri_pf_ref_test.cc`: (a) `sort` on random routes, on the Review Focus 2 adversaries and with exact selection ties at the cut, checking every (token, slot) pair appears once, each expert's rows ascending in token, padding rows `NONE`, `tiles_used <= tmax(C)`, and the all-to-6 adversary reaching the bound's routed term; (b) row independence: permuting rows inside a tile leaves `combine`'s per-token output bitwise unchanged; (c) `combine` == `kolibri_ref.h`'s decode combine (the 20c chain: fp32 ascending, mul then add, `+ shared`, one rounding) bit for bit on every token; (d) `flash_window` against a direct fp64 softmax over the visible keys of Review Focus 1's cases, including keys read through the ring's wrap; `eager_window` against 20c's `kolibri_fixture.h` eager rows bitwise where the positions coincide (5, 512, 513, 700).
- [ ] **Step 2: register and run, expect FAIL.**

```cmake
add_executable(kolibri_pf_ref_test kernels/kolibri_pf_ref_test.cc)
target_include_directories(kolibri_pf_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME kolibri_pf_ref_test COMMAND kolibri_pf_ref_test)
```
`cmake --preset mac-host && cmake --build --preset mac-host && ctest --preset mac-host -R '^kolibri_pf_ref_test$'`
- [ ] **Step 3: implement `kolibri_pf_ref.h`** until PASS.
- [ ] **Step 4: commit** `git commit -S -m "tests: Kolibri-1 prefill host reference - 384-expert sort, combine == decode's chain, windowed flash over the ring (spec 20d)"`.

### Task 2: the prefill kernels

**Files:**
- Create: `src/kernels/kolibri/kol_pf_moe.cl`, `src/kernels/kolibri/kol_pf_attn.cl`, `src/kernels/kolibri/kol_pf_linear.cl`, `tests/kernels/kolibri_pf_kernels_test.cc`, `tests/kernels/kolibri_pf_variant_names_test.cc`
- Modify: `src/kernels/kolibri_kernels.h` (a spec 20d block), `src/kernels/CMakeLists.txt` (a block `# ==== Spec 20d: Kolibri-1 prefill ==== (begin) / (end)` after 20c's), `tools/mac/clrun/kolibri_run.cc` (the portable prefill kernels), `tests/CMakeLists.txt`
- Test: `kolibri_pf_variant_names_test` (host), `kolibri_pf_kernels_test` (card), `kolibri_run` (Mac GPU, indicative)

**Interfaces:**
- Consumes: Task 1's reference; 20c's `kol_prep.cl` and `kol_moe.cl` (built again at M = 2048 / reused by name).
- Produces (`kernels::kolibri`, spec 20d block):

```cpp
inline constexpr unsigned kPfC = 2048, kPfTm = 32, kPfSlab = 1024;
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;   // one weight batch of bf16 expert blocks
inline constexpr unsigned kPfKt = 64, kPfRpw = 8, kPfHpw = 4;
std::string pf_moe_variant();                       // "kol_pf_moe_E384_T6_D2560_I512_L256"
std::string pf_flash_variant(bool sliding, bool eager);   // "kol_pf_flash_attn_Q48KV4_W513_R4096[_EAGER]" | "_F[_EAGER]"
std::string pf_bf16_slab_variant(unsigned K, unsigned N); // "kol_pf_bf16_slab_K6144_N2560"
std::vector<std::string> prefill_variants(model::KolAttnForm a, bool eager);
```

The kernels:

```text
kol_pf_moe.cl  (k2_pf_moe.cl's design at 384 experts; no sub-group function but the dequants' block reads, which the Mac shim emulates)
  kol_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)   grid (1), WG 256: lane l owns experts l, l + 256;
      ids staged as ushort [C][6] in SLM; counts, lane 0's serial prefix of ceil(c_e / 32) over e ascending, the scatter in
      (token, slot) order, the shared block 384's C rows last, the NONE tail; header [0] tiles, [1] shared first row, [2] rows,
      [3] C, [4 + e] counts (e <= 384)
  kol_pf_gather(x, hdr, row_tok, xg)                 grid (rows), WG 64: xg[row] = x[row_tok[row]], NONE -> zeros
  kol_pf_dequant_gu(w_gu, w_sh_gu, hdr, out, b0, b1) grid (2I / 16, 2560 / 64, b1 - b0), WG 16: block e in [b0, b1) with
      rows: e < 384 -> int4 layout-1 dequant ((q - 8) x scale, bf16) into [2560][1024] row-major (gate||up interleave16 kept);
      e == 384 -> the shared expert's bf16 tiles copied; a block with no rows returns at once
  kol_pf_dequant_dn(w_dn, w_sh_dn, hdr, out, b0, b1) the same for down: [512][2560]
  kol_pf_moe_combine(route, hdr, pair_row, y, mo, C) grid (2560 / 256, C), WG 256;  #pragma OPENCL FP_CONTRACT OFF
      acc = 0.f; for j = 0..5 (route row order = ascending id): acc = acc + f32(y[pair_row[t][j]][n]) · w_j
      mo[t][n] = rne(acc + f32(y[shared row of t][n]))            == kol_moe_down's chain, row for row
kol_pf_attn.cl  (k2_pf_attn.cl's DPAS / 2D-block structure; card only)
  kol_pf_flash_attn(q, kv_k, kv_v, out, pos, C)      grid (ceil(C / RPW), 4, 12 / HPW), 256 GRF
      row t at position p = pos + t; visible keys k: k <= p and (SLIDING: k > p - 513); key k at row SLIDING ? k & 4095 : k
      key tiles from KT·floor(max(0, pos + t0 - 512) / KT) (SLIDING; 0 for full layers) to the work-group's last row:
      absolute multiples of KT, and 4096 % KT == 0, so no 2D read crosses the ring's end; a wholly masked tile adds exact zeros
      out bf16 [C][6144] = rne(o / l)  (no gate); EAGER 1: k2_pf_attn's two passes, s = rne(f32(rne(q·k)) x 128^-0.5),
      p = rne(exp(s - m) / l), o rounded once
kol_pf_linear.cl  (one sub-group block read, as pf_bf16_slab.cl; runs on the Mac)
  kol_pf_bf16_slab(w, out, n0)                       pf_bf16_slab's copy for a slab of width pf_slab_width(N, n0), zero-filled to
                                                     pad256 - the same slab / tail contract as k2_pf_dequant_slab
```

Reused and new lines (`src/kernels/CMakeLists.txt`, spec 20d block, `if(B70_KOLIBRI)`; targets K2 already defines are built here only `if(NOT B70_K2)`):

```cmake
  set(KOL_PF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kolibri)
  set(KOL_PF_REUSE ${CMAKE_CURRENT_SOURCE_DIR}/prefill)
  add_ocloc_kernel(kol_pf_embed_gather_D2560_V128000 SOURCE ${KOL_PF_REUSE}/pf_embed.cl DEFINES HIDDEN=2560 VOCAB=128000)
  if(NOT B70_K2)        # K2's spec 18c block builds these two names
    foreach(SP 0 1)
      add_ocloc_kernel(pf_res_fold_K2560_SP${SP}_G20 SOURCE ${KOL_PF_REUSE}/pf_prep.cl DEFINES K=2560 S_PREV=${SP} FOLD_G=20)
    endforeach()
    add_ocloc_kernel(k2_pf_dequant_slab_K6144_N2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_pf_linear.cl DEFINES K=6144 N=2560)
  endif()
  add_ocloc_kernel(pf_res_fold_K2560_SP1_G20_Z SOURCE ${KOL_PF_REUSE}/pf_prep.cl DEFINES K=2560 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
  add_ocloc_kernel(k2_pf_dequant_slab_K2560_N7168 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_pf_linear.cl DEFINES K=2560 N=7168)
  foreach(row "2560 7168" "6144 2560")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(kol_pf_bf16_slab_K${K}_N${N} SOURCE ${KOL_PF_SRC}/kol_pf_linear.cl DEFINES K=${K} N=${N})
  endforeach()
  add_ocloc_kernel(kol_norm_M2048_K2560_G20_W20 SOURCE ${KOL_PF_SRC}/kol_prep.cl DEFINES M=2048 K=2560 NORM_G=20 NORM_WGS=20)
  add_ocloc_kernel(kol_post_add_M2048_K2560_G20 SOURCE ${KOL_PF_SRC}/kol_prep.cl DEFINES M=2048 K=2560 POST_G=20)
  set(KOL_PF_HEADS Q_HEADS=48 KV_HEADS=4 HD=128)
  add_ocloc_kernel(kol_attn_prep_M2048_N7168_S1_Q48KV4_R4096 SOURCE ${KOL_PF_SRC}/kol_prep.cl
                   DEFINES ${CTRL_DEFINES} M=2048 QKV_N=7168 QKV_S=1 ${KOL_PF_HEADS} SLIDING=1 RING=4096)
  add_ocloc_kernel(kol_attn_prep_M2048_N7168_S1_Q48KV4_F SOURCE ${KOL_PF_SRC}/kol_prep.cl
                   DEFINES ${CTRL_DEFINES} M=2048 QKV_N=7168 QKV_S=1 ${KOL_PF_HEADS} SLIDING=0 RING=0)
  foreach(SL 1 0)
    foreach(EG 0 1)
      if(SL)
        set(_n kol_pf_flash_attn_Q48KV4_W513_R4096)
        set(_w WINDOW=513 RING=4096)
      else()
        set(_n kol_pf_flash_attn_Q48KV4_F)
        set(_w WINDOW=0 RING=0)
      endif()
      if(EG)
        string(APPEND _n _EAGER)
      endif()
      add_ocloc_kernel(${_n} SOURCE ${KOL_PF_SRC}/kol_pf_attn.cl
                       DEFINES KT=64 RPW=8 HPW=4 Q_HEADS=48 KV_HEADS=4 EXP2=0 EAGER=${EG} ${_w}
                       OPTIONS -cl-intel-256-GRF-per-thread)
    endforeach()
  endforeach()
  add_ocloc_kernel(kol_pf_moe_E384_T6_D2560_I512_L256 SOURCE ${KOL_PF_SRC}/kol_pf_moe.cl
                   DEFINES PF_E=384 PF_K=6 PF_WG=256 PF_EPL=2 PF_D=2560 PF_INTER=512 TM=32)
  add_ocloc_kernel(pf_moe_router_K2560_N512 SOURCE ${KOL_PF_REUSE}/pf_gemv_bf16.cl DEFINES K=2560 N=512 COLS_PER_WG=16 KSPLIT=16)
  add_ocloc_kernel(pf_moe_gemm_K2560_N1024_SILU SOURCE ${KOL_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=2560 N=1024 TM=32 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K512_N2560 SOURCE ${KOL_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=512 N=2560 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
```
(The route is 20c's decode binary `kol_route_M1_E384_T6_N512_L256` on grid (1, C): it reads row m of `[C][512]` logits and nothing depends on its M. `pf_gemm_T0` is the existing shape-free binary.)

- [ ] **Step 1: the failing host test** `kolibri_pf_variant_names_test`: every name `prefill_variants` returns, for both arms and both attention forms, is a target of the block (`${B70_KOLIBRI_PREFILL_KERNELS}` as argv). Run on the Mac: FAIL (names undefined).
- [ ] **Step 2: write the three sources, the header block and the CMake block;** the names test PASSES.
- [ ] **Step 3: Mac GPU.** `kolibri_run` gains `kol_pf_sort` (random, adversarial, tied routes at C = 2048), `kol_pf_gather`, `kol_pf_dequant_gu` / `_dn` (an empty expert skipped, block 384 the bf16 copy), `kol_pf_moe_combine`, `kol_pf_bf16_slab` (the 512-column tail zero-filled), `kol_norm_finish` / `kol_post_add` at M = 2048 - each exact against `kolibri_pf_ref.h` / `kolibri_ref.h`.
- [ ] **Step 4: the card test** `kolibri_pf_kernels_test` (label `kolibri;prefill`, no checkpoint): everything of Step 3 on the card; the grouped GEMMs bitwise equal to a dense `pf_gemm` per expert on the same rows, and Task 1's row permutation leaving outputs bitwise unchanged (Review Focus 2-3); `kol_pf_flash_attn` against `flash_window` (fp64, cosine >= 0.99999, spec 6 K1) at full-layer depths 0, 2048, 30000, 60000, sliding at c0 = 0, 1000, 4396 (wrapped), a tail chunk of 300 rows and single rows; the EAGER form against `eager_window` (rounding points; not bitwise in sum order, spec 18 §11).
- [ ] **Step 5: Mac gate** `tools/mac_check.sh --base main --kernels` (cmdlines additions only; record the count) and **commit** `git commit -S -m "kernels: Kolibri-1 prefill - 384-expert sort, bf16 shared block, windowed ring flash, bf16 slab tail (spec 20d)"`.

### Task 3: the prefill walk and `KolibriEngine::prefill`

**Files:**
- Create: `src/runtime/kolibri/kolibri_prefill.{h,cc}`, `src/runtime/kolibri/kolibri_prefill_engine.cc`, `tests/runtime/kolibri_prefill_test.cc`, `tests/prefill/prefill_split_kolibri_test.cc`; `tests/golden/kolibri_golden_test.cc` (20c's) gains a `prefill` mode argument, as `k2_golden_test.cc` did - the `kolibri_golden_prefill_*` tests run it
- Modify: `src/runtime/kolibri/CMakeLists.txt` (archive `b70_kolibri_prefill` beside `b70_kolibri_runtime`), `src/runtime/kolibri/kolibri_engine.h` (the prefill members, declared as K2Engine declares its), `src/runtime/kolibri/kolibri_sizes.{h,cc}` (prefill sizes and launches), `src/cli/kolibri_decode.h`, `src/cli/b70_decode.cc`, `tests/CMakeLists.txt` (20c's `cli_reject_kolibri_prefill` removed; `cli_reject_kolibri_prefill_int8`, `cli_reject_kolibri_prefill_sycl` added)
- Test: `kolibri_plan_test` (host, extended); `kolibri_prefill_test`, `kolibri_golden_prefill_test`, `prefill_split_kolibri_test` (card)

**Interfaces:**
- Consumes: Task 2's kernels; 20c's `KolibriEngine`, `KolibriBuffers`, `kolibri_sizes`.
- Produces (used by 20e's server adapter and prefix cache):

```cpp
namespace runtime::kolibri {
struct PrefillSizes { size_t partials, slab, xg, h, y, mo, attn_q, attn_out, routes, sort, batch, total; };
PrefillSizes prefill_sizes(const model::Kolibri1Desc& d);          // per device, C = kPfC
uint32_t pf_batches_gu(const model::Kolibri1Desc& d);              // 4: 385 blocks of 5,242,880 B in 512 MiB batches of 102
uint32_t pf_batches_dn(const model::Kolibri1Desc& d);              // 2: 385 blocks of 2,621,440 B in batches of 204
size_t prefill_chunk_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p);
//   per layer 45 = norm 1, q||k||v 7 slabs x 2, prep 1, flash 1, o_proj 3 slabs x 2, fold_Z + post_add 2, norm 1,
//   router 1, route 1, sort 1, gather 1, gate||up 4 x 2, down 2 x 2, combine 1, fold + post_add 2;
//   + embed + fold (device 0): 2252 at 50 layers on one card and on two - the cut is 16b's (device 0 ends with layer
//   s-1's kol_post_add, whose rows and sums cross; device 1 starts at layer s's norm), so nothing is recomputed
inline constexpr size_t kPrefillHeadLaunches = 5;                // fold + norm over the last row, lm_head, argmax x 2
}
// KolibriEngine (kolibri_engine.h), defined in b70_kolibri_prefill:
void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);   // 0 = kPfC; leaves pos += ids.size(), cur_token pending
void prepare_prefill();
void set_prefill_replay(bool enabled);
size_t prefill_launches() const;
std::vector<uint32_t> read_prefill_routes();                          // u32 [layers][kPfC][32] of the last chunk
// Spec 20e's prefix cache hooks into this walk: the chunk loop calls
using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
void set_block_hook(BlockHook hook);                                  // fired after each chunk's last layer on the last device
```

- [ ] **Step 1: failing host test** (`kolibri_plan_test` extended): `prefill_chunk_launches` 2252 on one card and on two (`KolPlacement{2, 25}`); `pf_batches_gu` 4, `pf_batches_dn` 2; `prefill_sizes(...).total` per device (derived, the test asserts the formula; about 1.0 GB at kPfC 2048, the 512 MiB weight batch its largest term); `max_len_that_fits` with prefill planned still reaches 262144 on two cards. Run on the Mac: FAIL; implement in `kolibri_sizes.cc`; PASS.
- [ ] **Step 2: the walk** `kolibri_prefill.cc`: the chunk's list per device in the order of the Architecture line, asserting `prefill_chunk_launches`; the route rows of every layer kept for `read_prefill_routes`; the last chunk's last row copied into the decode `resid` of the last device and the head's 5 launches run there, so `cur_token` holds the first generated id and `pos` advanced on every device (20c's bookkeeping). B70_PREFILL_REPLAY's arrangement as K2's: record each chunk's list once per (pos, rows, attention variant) and replay.
- [ ] **Step 3: the card gates** (one card, both synthetic arms; label `checkpoint;kolibri;prefill`, SKIP 77 without data):
  - `kolibri_prefill_test <ckpt> <ids>`: K3 - two prefills bitwise equal (KV, rings, routes, first token); recorded replay == direct; chunk 2048 vs chunk 16 bitwise; Review Focus 4 (prefill KV / ring / routes against 20c's decode fill of the same ids, bars printed per layer); `_eager` twin (`B70_KOLIBRI_ATTN=eager`).
  - `prefill_split_kolibri_test <ckpt> <ids>`: Review Focus 5 - splits 64, 1000, 2048, 4097 against one call; multiples of 64 bitwise, others within `prefill_split_test`'s bars; prints which were bitwise.
  - `kolibri_golden_prefill_test` (= `kolibri_golden_test <ckpt> <oracle dir> prefill [int8]`): KL2 on prefill - the prompt by `prefill`, then 32 greedy tokens; 20c's tie-aware rule and routing diagnostic on the prompt rows (`B70_KOL_TIE_TOL`); `_c16` (chunks of 16) and `_i8head` twins; registered for `int4attn` and `bf16attn` synths and, SKIP until 20b, the real checkpoint.
- [ ] **Step 4: two cards** (16b's pieces, as 20c Task 6): device 0's chunk list ends after layer s-1's `kol_post_add` with two device-to-device copies - the chunk's `resid` rows `[rows][2560]` and their `sumsq_r` `[20][rows]` - into the prefill link's landing buffer and `barrier_signal(event)`; device 1's chunk list starts with `wait_event(event)` and two local copies into its prefill `resid` / `sumsq_r`, then layer s's `kol_norm_finish`. The host resets the event per chunk, submits both lists and waits on device 1's fence with `wait_for(timeout_ms)`; a lost hand-off host-signals the event, throws naming it and marks the engine until `reset()` (PipelineEngine's rule). The last chunk's head runs on device 1, then the Control mirror (device 1's block into device 0's). Gates: `kolibri_prefill_test` and `prefill_split_kolibri_test` with `--pp 2 --pipeline-split 3` (the engine options) on the synthetic checkpoint bitwise equal to `--pp 1` (KV, rings, routes, first token); the real checkpoint's `kolibri_golden_prefill_test` on two cards; `--max-len auto` with prefill planned through `pp_split_and_len`.
- [ ] **Step 5: CLI.** `b70-decode <kolibri> --prefill` and `--bench --prefill-length N [--prefill-chunk C]` on `--prefill-backend l0` (the default), with `--pp 2` (Kolibri's default) or `--pp 1` where the model fits one card; 20c's `cli_reject_kolibri_prefill` removed; `--prefill-chunk` above 2048 refused; `--prefill-backend l0-int8 | sycl-tla` refused naming the 1024-k Hadamard blocks / the missing walk (`cli_reject_kolibri_prefill_int8`, `cli_reject_kolibri_prefill_sycl`); `--max-len auto` plans the prefill scratch when a run prefills.
- [ ] **Step 6: Mac gate and commit** `git commit -S -m "prefill: Kolibri-1 chunks - grouped experts, windowed flash, one hand-off per chunk; KL2, KL3 on prefill (spec 20d)"`.

### Task 4: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-05-spec20-kolibri-1-design.md` (new section "20d as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md`, `tools/box_validate/stages.sh`, `docs/19-running-models.md` (Kolibri row: `--prefill`)

- [ ] **Step 1:** the as-built section: the chunk table (kernels, rounding chains), launches 2252 (one card and two) + 5, the prefill scratch per device, the decisions taken blind (l0 only; the separate dequant pass, 512 MiB batches; the ring's 4096 slots as the reason a chunk's keys fit; the sequential two-card chunk), precision against the reference (flash's fp32 probabilities vs eager; `B70_KOLIBRI_ATTN=eager` covers prefill too, one parser `runtime::kolibri::kolibri_attn()`).
- [ ] **Step 2: the queue row** (next free number, after 20c's): every binary of the 20d block never compiled by ocloc; the DPAS flash at GQA 12 with the window; the grouped GEMMs at Kolibri's shapes; the 2252-launch walk; Mac-side what passed; runbook stages `rN.k1` (`kolibri_pf_kernels_test`), `rN.prefill` (`kolibri_prefill_test`, `_eager`), `rN.split`, `rN.golden` (synths; real after 20b), `rN.pp` (two cards), `rN.speed` (opt-in); `python3 tools/box_validate/test_box_validate.py` passes.
- [ ] **Step 3: commit** `git commit -S -m "docs: spec 20 (20d as built), box queue row N - Kolibri-1 prefill"`.

### Task 5: prefill speed (box)

- [ ] Two cards, real checkpoint, `l0`: `b70-decode <snap> --pp 2 --bench --prefill-length N --max-len 140000` for N = 4096, 32768, 131072, interleaved pairs, median of 3, `uptime`; per-device busy time per chunk; flash vs eager prefill at pp32768; the share of the dequant pass (a profile of one chunk). Derived estimate (unmeasured): ~6.2 GFLOP per token (attention row 7168 + o_proj 6144 columns, 6 experts + shared at 3 x 2560 x 512, x 50 layers) - pp4096 ~25 TFLOP of DPAS; the per-chunk dequant pass writes and reads ~3 GB of bf16 expert blocks per layer (~300 GB of traffic per chunk over both cards, ~0.5 s); pp4096 ~1 s, ~4,000 t/s; at 131072 the full layers' flash adds ~2.1 PFLOP (10 layers x 48 heads x 131072² / 2 pairs x 512 FLOP; ~20 s at ~100 TFLOP/s). Recorded levers, not built here: spec 16c's overlapped chunk pipeline (up to ~2x), an SLM-fused int4 grouped GEMM (spec 15d Task 1's arm) removing the dequant pass. `docs/BENCHMARKS.md` "Kolibri-1 (spec 20)" prefill rows. **Commit** `git commit -S -m "spec 20d: Kolibri-1 prefill speed"`.

**Gate for the plan:** Mac - host tests and `kolibri_run` green, cmdlines additions only. Box - K0; `kolibri_pf_kernels_test`; K3 on prefill (replay, chunking, split at multiples of 64 bitwise); KL2 on prefill for both synthetic arms (real after 20b); two cards bitwise equal to one card (after 16b); speed rows recorded.

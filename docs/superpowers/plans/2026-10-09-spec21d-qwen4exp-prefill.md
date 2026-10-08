# Spec 21d - Qwen3.8-Flash-Next prefill: dense flash below 2052, the indexer and sparse flash above, grouped MoE at 512

**Status (2026-10-09): planned; nothing built.** Built blind on the Mac; box queue row 33 (it renumbers at build
time if taken). Real-weight gates SKIP (77) until Intel's checkpoint and 21a's sets exist on the box.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Qwen4ExpEngine::prefill` runs a prompt in chunks of up to 2048 positions on the `l0` backend: the linears as slab GEMMs (int4 dequant slabs or bf16 slabs), the hyper-connections over a chunk, PLE over a chunk, GDN through Qwen3.8's WY-chunked binaries with the sigmoid gate, QSA as spec 6's flash for rows with at most 2051 visible positions and as a sparse flash over each row's own selection above, the MoE through a 512-expert sort with `ushort` ids and spec 15d's grouped GEMMs; the chunk crosses the two cards once; F3 and F4 on prefill; the prefill speed rows.

**Architecture:** spec 21 §4.2 "Prefill", §8 21d; spec 20d / 18c's arrangement (own walk, `runtime::prefill::Context` + `KernelCache` + `gemm_l0`, the family's kernels named `q4_pf_*`, decode's chains row for row where a kernel is shared). Per chunk on each device, one Level Zero list:

```text
device 0 first:  pf_embed_gather (hidden 2560) · [layer 0's combine_norm _E at M = C]
every layer:     HC: q4_hc_combine_norm (M = C) · bf16 slab {10240, 336} + pf_gemm · q4_hc_up_mix (M = C)
  GDN:           qkv||z (16 int4 L1 dequant slabs, or bf16 slabs) + pf_gemm · a||b pf_ab_proj (hidden 2560) · gdn_chunk_q4 (Qwen3.8's ten binaries,
                 the tenth pf_gated_head _SIG) · out_proj slabs + pf_gemm
  QSA:           q||gate||k||v slabs + pf_gemm · indexer bf16 slab {2560, 640} + pf_gemm · pf_attn_prep_q16 _Q24KV2 · q4_qsa_prep (M = C:
                 the chunk's raw keys, its complete blocks' compressed keys, the q heads) · rows <= 2050: pf_flash_attn _Q24KV2 + pf_attn _Q24KV2's
                 gate; rows >= 2051: q4_qsa_score + q4_qsa_select (decode's binaries at M = C) · q4_pf_sparse_attn + the same gate · o_proj
  MLP side:      HC (3 + slab) · router pf_gemv_bf16 (528) · q4_route (decode's binary on grid (1, C)) · q4_pf_sort · q4_pf_gather ·
                 per weight batch q4_pf_dequant_gu + pf_moe_gemm (SiLU) · per batch q4_pf_dequant_dn + pf_moe_gemm · q4_pf_moe_combine (y)
  PLE (layer 1): combine_norm _NN · q4_ple_gather (ids from the chunk) · key||value slabs + pf_gemm · q4_pf_ple_gate · q4_pf_ple_conv ·
                 q4_pf_ple_ring · combine_norm _X
last row:        the final mixer and the head with decode's binaries on the last row (pending id, pos advanced on every device)
```

**One deliberate departure from spec 21 §4.2:** the prefill **scores** run decode's `q4_qsa_score` at M = C (fp32 FMA: ~8.6 GFLOP a QSA layer a chunk at 32k context, ~34 at 128k, ESTIMATED) instead of a DPAS GEMM, so a prefill row's scores are bitwise the decode kernel's for the same query and keys and the selection is exactly decode's rule; the GEMM form is Task 5's recorded lever if the scores bind.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§2.3, §4.2, §4.3, §4.4, §7 F3 F4, §8 21d). Plan 21c and its "21c as built". Precedents: plan 20d (`2026-10-06-spec20d-kolibri-prefill.md`), `src/kernels/kolibri/{kol_pf_moe,kol_pf_attn,kol_pf_linear}.cl`, `src/runtime/kolibri/kolibri_prefill{.h,.cc,_engine.cc}`, `tests/kernels/kolibri_pf_ref.h`, `tests/runtime/kolibri_prefill_test.cc`, `src/runtime/prefill/gdn.{h,cc}` (`gdn_chunk`'s ten launches), `src/kernels/prefill/pf_flash_attn.cl` (`FA_Q_HEADS` / `FA_KV_HEADS`, HD 256), `src/kernels/prefill/pf_moe.cl` (the `uchar` staging this family cannot use, `:117-124`).

## Dependencies and branch points

- **Plan 21c merged** (its kernels, `Qwen4ExpEngine`, `qwen4exp_sizes`, the golden test, the synthetic golden sets). Two of 21c's kernels are reused at M = C and must be M-agnostic in their row indexing (`get_group_id(1)` is the row; nothing else depends on M - Kolibri's `kol_route` precedent): `q4_route_M1_*` on grid (1, C) as built, and `q4_hc_*`, `q4_qsa_{prep,score,select}` rebuilt at `M = 2048` from their sources. If 21c's as-built section says otherwise, follow it.
- **`runtime::prefill::gdn_chunk` cannot be called** for this family: it takes a `PrefillScratch` sized by a `model::ModelDesc` (Qwen3.8's ~GB of scratch) and binds `pf_gated_head`'s silu form by name (`src/runtime/prefill/gdn.cc:153-229`). This plan writes a sibling, `runtime::qwen4exp::gdn_chunk_q4`, launching the same ten binaries in `gdn.cc`'s order over this family's own buffers, the tenth being `pf_gated_head_SIG`. `gdn.cc` is not edited.
- **Spec 16b / 16c:** the chunk crosses by spec 16b's `copy` hand-off at chunk size - a second `runtime::PipelineLink` over `runtime::pp_landing_layout(kPfC x 20480, 0)` (42 MB of `H` rows), sequentially (spec 16c's overlapped chunk pipeline is a later lever, as for Kolibri). `pp_handoff.cl`'s peer kernels are sized for a row, not a chunk: the chunk always crosses by `copy`, whatever `--pipeline-handoff` says for decode (Kolibri's rule, recorded).
- **Backend:** `l0` only. `l0-int8` (spec 5's h8 path rotates in 1024-k Hadamard blocks; the hidden is 2560) and `sycl-tla` (no walk for this family) are refused by name, as K2 and Kolibri refuse them.
- **Spec 21e** hooks the prefix cache into this walk (`set_block_hook`), and spec 22's prefill streaming (spec 22 §4) will read the grouped GEMM's expert blocks through its table: the dequant kernels take the layer's block base and an id, never a computed layer offset (21c's `q4_expert.h` macro).

## Global Constraints

- Branch `spec21d-qwen4exp-prefill` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked.
- **F0:** every existing binary keeps its name, command line and bytes; 21c's decode list and launch counts unchanged. **One existing source changes:** `src/kernels/prefill/pf_gated_head.cl` gains the same `GDN_GATE_SIGMOID` define 21c gave `prep.cl` (9a67889's rule; the fallback - a copy in `src/kernels/qwen4exp/q4_pf_gdn.cl` - if G0 shows `pf_gated_head` moved). Binaries other blocks define are bound by name and built here only when that block is off (`k2_pf_dequant_slab_K6144_N2560`: K2's / Kolibri's).
- **Determinism:** no atomic in any sum that feeds a result; the sort one work-group in (token, slot) order; a grouped-GEMM row independent of the other rows of its tile; the combine in decode's order; flash key tiles at absolute multiples of KT; a split prefill bitwise at multiples of 64.
- One weight format; the forms 21b reads. Family kernels in `src/kernels/qwen4exp/`, `q4_pf_*`, every shape define in the name; reused sources' binaries named by their existing helpers (`pf_kernels.h`) or with a `q4_` prefix where the source's own name carries another model's vocabulary (`kol_pf_linear.cl`'s slab at this family's shapes is `q4_pf_bf16_slab_K<K>_N<N>`).
- Every number measured, or marked derived / estimated / proposed. Box: `flock ~/b70-gpu.lock` (once for both cards), detached, polled; interleaved pairs, median of 3, `uptime`; `-j44`.
- Mac checks `tools/mac_check.sh --base main --kernels`. No `rm -rf`. Signed commits on the branch; no merge, no push.

## Review Focus

1. **The 2051 boundary inside one chunk and across chunks.** A row at position p has `p + 1` visible positions; rows `p <= 2050` run dense flash, rows `p >= 2051` the sparse kernel over their own list. The first chunk (positions 0..2047) is wholly dense; a chunk starting at 2048 has three dense rows (2048, 2049, 2050) then sparse ones. Task 1's reference and Task 2's kernel test cover chunk starts 0, 1000, 2000 (straddling), 2048, 4096 and a tail chunk of 300 rows; the dense rows' output is bitwise what the same rows give through the sparse kernel's identity list only within the flash bar (two kernels), never assumed bitwise.
2. **A chunk's compressed keys before any row reads them.** `q4_qsa_prep` at M = C writes every complete block of the chunk (from this launch's raw keys, and the tail ring for the open block before `pos`), then `q4_qsa_score` reads them in the next launch: a row never sees a block that is not complete at its own position (`n = (p + 1) / 4`). The 8-slot tail ring after the chunk holds the open block's raw keys - checked against decode's ring after the same ids.
3. **512 experts in the sort.** Ids up to 511 do not fit `pf_moe_sort`'s `uchar` staging (`pf_moe.cl:61`, `:117-124`); `q4_pf_sort` stages `ushort` and gives each of 256 lanes two experts; block 512 of every weight batch is the shared expert (int4 dequantised, or Intel's bf16 copied). Adversaries: every token to the same 10 experts (the tile bound), experts 255 / 256 / 511 (the lane boundaries), an expert with exactly 32 rows and one with 0, exact route ties at the 10th. `tmax(2048) = floor((2048 x 10 + 512 x 31) / 32) + 64 = 1200` tiles (derived).
4. **PLE over a chunk reads only what came before.** Row m's history is the chunk's ids at m - 1, m - 2 or the position-indexed id ring for positions before `pos`; its conv reads normed gated rows at p - 3, p - 6, p - 9 from this chunk or from the ring - which is why gate, conv and the ring update are three launches (a row's conv must not read a ring slot this launch overwrites).
5. **Prefill against decode, and against itself.** The KV, compressed keys, GDN state and PLE rings a prefill writes against 21c's M = 1 fill of the same ids: within cosine bars (rows >= 0.999, median >= 0.9998, p01 >= 0.99 - spec 18c's, PROPOSED), selections and routes equal except near-ties; a prefill split at multiples of 64 bitwise equal to one call; chunk 2048 vs chunk 16 bitwise; on two cards the same as on one.

---

### Task 1: the host reference for the prefill kernels

**Files:**
- Create: `tests/kernels/qwen4exp_pf_ref.h`, `tests/kernels/qwen4exp_pf_ref_test.cc`
- Modify: `tests/CMakeLists.txt` (a block `# ==== Spec 21d: Qwen3.8-Flash-Next prefill (labels qwen4exp;prefill) ==== (begin)` after 21c's)
- Test: `qwen4exp_pf_ref_test` (host)

**Interfaces:**
- Consumes: 21c's `tests/kernels/qwen4exp_ref.h` (`route`, `moe_combine`, `qsa_prep`, `qsa_scores`, `qsa_select`, `qsa_attn_eager`, `qsa_attn_fp64`, `ple_ids`, `hc_*`), `kernels::qwen4exp::route` word indices.
- Produces (namespace `q4pf`):

```cpp
struct Sorted { std::vector<uint32_t> hdr, tiles, row_tok, pair_row; uint32_t tiles_used; };
uint32_t tmax(uint32_t C);                          // floor((C*10 + 512*31) / 32) + ceil(C / 32): 1200 at C = 2048
Sorted sort(const uint32_t* route, uint32_t C);     // expert-major, ascending token, tiles of 32, shared block 512 last
void combine(const uint32_t* route, const Sorted& s, const uint16_t* y, uint16_t* out, uint32_t C);   // == decode's chain per row
uint32_t dense_rows(uint32_t pos, uint32_t C);      // rows of the chunk with <= 2051 visible positions
void sparse_attn_fp64(const float* q, const uint16_t* k, const uint16_t* v, const uint32_t* lists, const uint32_t* counts,
                      uint32_t C, double* o);       // per row over its own list, the 12 q heads of each kv head
void ple_chunk(/* ids, ring, table rows, H */);     // gate, conv, ring update in three steps (Review Focus 4)
```

- [ ] **Step 1: the failing test** `qwen4exp_pf_ref_test.cc`: `sort` on random routes and Review Focus 3's adversaries (every (token, slot) once, rows ascending per expert, padding `NONE`, `tiles_used <= tmax(C)`, the all-to-10 case reaching the routed term); row independence (permuting rows inside a tile leaves `combine` unchanged); `combine` == decode's `moe_combine` per row bit for bit; `dense_rows` at the starts of Review Focus 1; `sparse_attn_fp64` against a direct softmax over the row's list; `ple_chunk` == 21c's M = 1 `ple` chain applied row by row (bitwise: the same arithmetic).
- [ ] **Step 2: register, run, expect FAIL; implement; PASS** (`ctest --preset mac-host -R '^qwen4exp_pf_ref_test$'`).
- [ ] **Step 3: commit** `git commit -S -m "tests: Qwen3.8-Flash-Next prefill host reference - 512-expert sort, combine == decode, sparse attention over row lists, PLE over a chunk (spec 21d)"`.

### Task 2: the prefill kernels

**Files:**
- Create: `src/kernels/qwen4exp/q4_pf_moe.cl`, `src/kernels/qwen4exp/q4_pf_attn.cl`, `src/kernels/qwen4exp/q4_pf_ple.cl`, `tests/kernels/qwen4exp_pf_kernels_test.cc`, `tests/kernels/qwen4exp_pf_variant_names_test.cc`
- Modify: `src/kernels/prefill/pf_gated_head.cl` (the `GDN_GATE_SIGMOID` define), `src/kernels/qwen4exp_kernels.h` (a spec 21d block), `src/kernels/CMakeLists.txt` (a block `# ==== Spec 21d: Qwen3.8-Flash-Next prefill ==== (begin) / (end)` after 21c's), `tools/mac/clrun/qwen4exp_run.cc` (the portable prefill kernels), `tests/CMakeLists.txt`
- Test: `qwen4exp_pf_variant_names_test` (host), `qwen4exp_pf_kernels_test` (card), `qwen4exp_run` (Mac GPU, indicative)

**Interfaces:**
- Produces (`kernels::qwen4exp`, spec 21d block):

```cpp
inline constexpr unsigned kPfC = 2048, kPfTm = 32, kPfSlab = 1024;
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;   // gate||up: 513 blocks of 6,553,600 B, 81 a batch, 7 batches;
                                                             // down: 3,276,800 B, 163 a batch, 4 batches (derived)
std::string pf_moe_variant();                                // "q4_pf_moe_E512_T10_D2560_I640_L256"
std::string pf_sparse_attn_variant(bool eager);              // "q4_pf_sparse_attn_Q24KV2" | "_EAGER"
std::string pf_ple_variant();                                // "q4_pf_ple_C2048"
std::string pf_bf16_slab_variant(unsigned K, unsigned N);    // "q4_pf_bf16_slab_K10240_N336" ...
std::vector<std::string> prefill_variants(const model::Qwen4ExpDesc& d, bool eager);
```

```text
q4_pf_moe.cl  (kol_pf_moe.cl's design at 512 experts; no sub-group function but the dequants' block reads, which the Mac shim emulates)
  q4_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)   grid (1), WG 256: lane l owns experts l, l + 256; ids staged as ushort
      [C][10] in SLM; counts, lane 0's serial prefix of ceil(c_e / 32) over e ascending, the scatter in (token, slot) order, the shared
      block 512's C rows last, the NONE tail
  q4_pf_gather(x, hdr, row_tok, xg)                 xg[row] = x[row_tok[row]], NONE -> zeros
  q4_pf_dequant_gu(w_gu, w_sh, hdr, out, b0, b1)    block e in [b0, b1) with rows: e < 512 -> layout-1 dequant ((q - 8) x scale, bf16)
      into [2560][1280] (the gate||up interleave kept), addressed by Q4_EXPERT_GU; e == 512 -> the shared expert (int4 or bf16 copy)
  q4_pf_dequant_dn(...)                             the same for down: [640][2560]
  q4_pf_moe_combine(route, hdr, pair_row, y, out, C)  #pragma OPENCL FP_CONTRACT OFF; == q4_moe_down's combine, row for row
q4_pf_attn.cl  (pf_flash_attn.cl's DPAS / 2D-block idioms; card only)
  q4_pf_sparse_attn(Q, Kc, Vc, lists, counts, O, pos, C)   grid (rows >= 2051 of the chunk, 2 kv heads): the row's 12 q heads padded
      to 16 are the DPAS M; keys and values gathered by the row's list in tiles of KT positions (ascending), fp32 online softmax at
      1/16; O fp32 [24][rows][256] in pf_o's layout, so pf_attn _Q24KV2's gate applies unchanged
      EAGER 1: the reference's bf16 chain (s rounded twice, p once, o once), not bitwise in sum order (spec 18 §11's rule)
q4_pf_ple.cl  (plain OpenCL C; runs on the Mac)
  q4_pf_ple_gate(kv_f32, H, w, gated, gn)           grid (4 streams, C): the decode block's gate and norm_conv, per row
  q4_pf_ple_conv(gated, gn, ring, H)                grid (2560 / 64, C): rows p, p-3, p-6, p-9 from gn (this chunk) or the ring; H += out
  q4_pf_ple_ring(gn, ids, ring, id_ring)            the chunk's last 16 positions' rows and ids into the position-indexed rings
```

Reused at this family's shapes (new lines in the block): `pf_embed.cl` and `pf_ab_proj`'s source at hidden 2560 (the names `pf_embed_gather_variant(2560)` and `pf_ab_proj_variant(2560)` return, VOCAB 248320; K2's and Kolibri's embeds carry their own prefixes, so no target collides), `pf_gemm_T0` (shape-free, as built), `pf_attn_prep_q16_Q24KV2` and `pf_attn_Q24KV2` (`pf_attn_prep_q16_variant(24, 2)`, `pf_attn_variant(24, 2)`: the q/k norms, RoPE, KV write, and the gate kernel `attn_gate_chunk` binds), `pf_flash_attn_Q24KV2` (KT 64, RPW 8, HPW 6 or 4 - a divisor of GQA 12 - EXP2 as the existing binary, `-cl-intel-256-GRF-per-thread`), `pf_gdn_conv` / `pf_gdn_wy` / `pf_gdn_scan` (Qwen3.8's binaries, by name: the GDN shape is Qwen3.8's), `pf_gated_head_SIG`, the dequant slabs (`pf_dequant_slab_variant(2560, 16384, 1)` for qkv‖z, `(2560, 13312, 0)` for q‖gate‖k‖v; `k2_pf_dequant_slab_K6144_N2560` for out_proj / o_proj, `if(NOT B70_K2 AND NOT B70_KOLIBRI)`), the bf16 slabs from `kol_pf_linear.cl` (`q4_pf_bf16_slab_K{2560_N16384, 2560_N13312, 6144_N2560, 2560_N640, 10240_N336, 2560_N12800}`), `pf_moe_router_K2560_N528` (`pf_gemv_bf16.cl`), `pf_moe_gemm_K2560_N1280_SILU`, `pf_moe_gemm_K640_N2560` (`pf_moe_gemm.cl`, TM 32, 256 GRF), and 21c's `q4_hc_*`, `q4_qsa_prep`, `q4_qsa_score`, `q4_qsa_select`, `q4_ple_gather` sources at `M=2048`. The block's count is recorded in the commit message.

- [ ] **Step 1: the failing host test** `qwen4exp_pf_variant_names_test`: every name `prefill_variants` returns, both forms, both attention forms, is a target of the block (`${B70_Q4EXP_PREFILL_KERNELS}`). FAIL.
- [ ] **Step 2: write the three sources, the header block, the CMake block, the `pf_gated_head.cl` define;** the names test PASSES.
- [ ] **Step 3: Mac GPU.** `qwen4exp_run` gains `q4_pf_sort` (random, adversarial, tied routes at C = 2048), `q4_pf_gather`, `q4_pf_dequant_gu` / `_dn` (an empty expert skipped, block 512 both shared forms), `q4_pf_moe_combine`, `q4_pf_ple_{gate,conv,ring}`, 21c's HC / indexer kernels at M = 2048 - each exact against `qwen4exp_pf_ref.h` / `qwen4exp_ref.h`.
- [ ] **Step 4: the card test** `qwen4exp_pf_kernels_test` (labels `qwen4exp;prefill`, no checkpoint): Step 3's set on the card; the grouped GEMMs bitwise equal to a dense `pf_gemm` per expert on the same rows and the row permutation leaving outputs unchanged; `q4_pf_sparse_attn` against `sparse_attn_fp64` (cosine >= 0.99999, spec 6 K1) at chunk starts 2048, 4096, 30000, 131072 and a 300-row tail; `pf_flash_attn_Q24KV2` against fp64 on the dense rows; the EAGER form against the eager chain (rounding points; not bitwise in sum order); `gdn_chunk_q4`'s binaries with `pf_gated_head_SIG` against 21c's decode `gdn_step` + `prep_gated_head_M1_SIG` state over the same 64 positions within spec 6 §6.3's self-consistency bar.
- [ ] **Step 5: Mac gate** `tools/mac_check.sh --base main --kernels` (cmdlines additions only; the count recorded) and **commit** `git commit -S -m "kernels: Qwen3.8-Flash-Next prefill - 512-expert ushort sort, sparse flash over row lists, PLE over a chunk, the sigmoid gated head (spec 21d)"`.

### Task 3: the prefill walk and `Qwen4ExpEngine::prefill`

**Files:**
- Create: `src/runtime/qwen4exp/qwen4exp_prefill.{h,cc}`, `src/runtime/qwen4exp/qwen4exp_prefill_gdn.{h,cc}` (`gdn_chunk_q4`), `src/runtime/qwen4exp/qwen4exp_prefill_engine.cc`, `tests/runtime/qwen4exp_prefill_test.cc`
- Modify: `src/runtime/qwen4exp/CMakeLists.txt` (archive `b70_qwen4exp_prefill` beside `b70_qwen4exp_runtime`, Kolibri's arrangement: a decode-only target links what it always linked), `src/runtime/qwen4exp/qwen4exp_engine.h` (the prefill members, declared as `KolibriEngine` declares its), `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (prefill sizes and launches), `tests/runtime/qwen4exp_plan_test.cc`, `tests/golden/qwen4exp_golden_test.cc` (21c's: a `prefill` mode argument, as `kolibri_golden_test` has), `src/cli/qwen4exp_decode.h`, `src/cli/b70_decode.cc`, `tests/CMakeLists.txt` (21c's `cli_reject_qwen4exp_prefill` removed; `cli_reject_qwen4exp_prefill_int8`, `_prefill_sycl` added)
- Test: `qwen4exp_plan_test` (host, extended); `qwen4exp_prefill_test`, `qwen4exp_golden_prefill_*` (card)

**Interfaces:**
- Consumes: Task 2's kernels; 21c's engine, buffers, sizes; `runtime::prefill::Context`, `KernelCache`, `gemm_l0` (`src/runtime/prefill/gemm_l0.h`), `runtime::prefill_chunk_rows` (`src/runtime/prefill_chunks.h`).
- Produces (used by 21e's server adapter and prefix cache):

```cpp
namespace runtime::qwen4exp {
struct PrefillSizes { size_t ids, H, xn, x, down_f32, slab, partials, idx_f32, idx_q, scores, lists, counts, q16, o, gate, attn_out,
                      gdn /* gdn_chunk_q4's ten scratch buffers */, logits, routes, hdr, tiles, row_tok, pair_row, xg, h, y,
                      w /* the expert weight batch */, ple; size_t total() const; };
PrefillSizes prefill_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len);   // scores [kPfC][max_len / 4] fp32 is the term that scales
size_t prefill_chunk_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t pos, uint32_t C);
                                                     // depends on how many rows are dense / sparse; asserted per chunk
PpLandingLayout pf_landing_layout(const model::Qwen4ExpDesc& d);   // pp_landing_layout(kPfC x 20480, 0)
// gdn_chunk_q4: gdn.cc's ten launches over explicit buffers (no PrefillScratch, no ModelDesc), the tenth pf_gated_head_SIG.
void gdn_chunk_q4(prefill::Context& cx, prefill::KernelCache& kc, const GdnChunkBuffers& b, uint32_t pos, uint32_t C);
inline constexpr size_t kGdnChunkLaunches = 10;
}
// Qwen4ExpEngine (qwen4exp_engine.h), defined in b70_qwen4exp_prefill:
void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);   // 0 = kPfC; pos += ids.size(), the first id pending
void prepare_prefill();  void set_prefill_replay(bool enabled);  size_t prefill_launches() const;
std::vector<uint32_t> read_prefill_routes();                          // u32 [layers][kPfC][32], the last chunk
std::vector<uint32_t> read_prefill_selection(uint32_t layer);         // the last chunk's lists + counts
static constexpr uint32_t kBlock = 2048;
using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
void set_block_hook(BlockHook hook);                                  // fired after each chunk's last layer on the last device
```

- [ ] **Step 1: failing host test** (`qwen4exp_plan_test` extended): `prefill_chunk_launches` for a chunk at pos 0 (all dense), at 2048 (3 dense rows), at 4096 (all sparse), at `--layers 4` and 48 (the test prints and asserts the formula); `prefill_sizes(...).total` at max_len 32768 / 131072 (about 1.5-2 GB, ESTIMATED; the 512 MiB weight batch and the score rows the largest terms); `max_len_that_fits` with prefill planned at `--layers 18` (one card) and 38 (two). FAIL; implement in `qwen4exp_sizes.cc`; PASS.
- [ ] **Step 2: the walk** `qwen4exp_prefill.cc` in the Architecture's order, asserting `prefill_chunk_launches`; the injected selection (21c's) honoured in prefill too (each chunk's lists from the host buffer); the last chunk's last row into the decode `H` of the last device and the head there (`cur_token` holds the first generated id, `pos` advanced on every device); B70_PREFILL_REPLAY's arrangement as Kolibri's (record each chunk's list once per (pos, rows) and replay).
- [ ] **Step 3: the card gates** (one card, both synthetics; labels `checkpoint;qwen4exp;prefill`, SKIP 77 without data):
  - `qwen4exp_prefill_test <ckpt> <ids>` (F4 on prefill): two prefills bitwise (KV, keys, tail rings, GDN state, PLE rings, routes, selections, first token); replay == direct; chunk 2048 vs chunk 16 bitwise; splits at 64, 2048, 2051, 4097 against one call - multiples of 64 bitwise, the others within `prefill_split_test`'s bars, printed which were bitwise; Review Focus 5's prefill-vs-decode bars per layer; the injected run chunked == whole bitwise; an `_eager` twin.
  - `qwen4exp_golden_prefill_*` (= `qwen4exp_golden_test <ckpt> <oracle dir> prefill [int8] [inject]`): F3 on prefill - the prompt by `prefill`, then 32 greedy tokens; 21c's token rule, routing diagnostic and gate S on the prompt rows; `_c16`, `_i8head`, `_inject` twins; registered for both synthetics and (SKIP until the data exists) Intel's checkpoint at `--layers 18`.
- [ ] **Step 4: two cards** (16b's pieces, as 21c Task 4): device 0's chunk list ends after layer s-1 with the combine-only `_NN` over the chunk's rows and a device-to-device copy of `H` `[rows][10240]` into the prefill link's landing buffer, then `barrier_signal(event)`; device 1's chunk list starts with `wait_event(event)`, the local copy into its prefill `H`, layer s's `_X`. Host resets the event per chunk, waits on device 1's fence with `wait_for(timeout_ms)`; a lost hand-off host-signals the event, throws naming it, marks the engine until `reset()`. Gates: `qwen4exp_prefill_test` with `--pp 2 --pipeline-split 2` (synthetic) bitwise equal to `--pp 1`; Intel's at `--layers 38` on two cards; `--max-len auto` with prefill planned.
- [ ] **Step 5: CLI.** `b70-decode <qwen4exp> --layers N --prefill` and `--bench --prefill-length N [--prefill-chunk C]` on `--prefill-backend l0` (the default) with `--pp 1` or `--pp 2`; `--prefill-chunk` above 2048 refused; `--prefill-backend l0-int8 | sycl-tla` refused naming the 1024-k Hadamard blocks / the missing walk (`cli_reject_qwen4exp_prefill_int8`, `_prefill_sycl`); `--max-len auto` plans the prefill scratch when a run prefills.
- [ ] **Step 6: Mac gate and commit** `git commit -S -m "prefill: Qwen3.8-Flash-Next chunks - dense flash to 2050, the indexer and sparse flash beyond, grouped experts at 512, one hand-off per chunk; F3, F4 on prefill (spec 21d)"`.

### Task 4: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (section "21d as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 33), `tools/box_validate/stages.sh` (`row 33` block), `docs/19-running-models.md` (the Qwen3.8-Flash-Next row: `--prefill`)

- [ ] **Step 1:** "21d as built": the chunk table (kernels, rounding chains, launches per kind of chunk), the scores' departure from spec 21 §4.2 and its cost, `gdn_chunk_q4`, the prefill scratch per device, precision against the reference (flash's fp32 probabilities; `B70_Q4_ATTN=eager` covers prefill too).
- [ ] **Step 2: the queue row** (33): every binary of the 21d block never compiled by ocloc; the DPAS sparse flash; the grouped GEMMs at this family's shapes; the walk; stages `r33.k0` (G0 incl. `pf_gated_head`), `r33.host`, `r33.k1` (`qwen4exp_pf_kernels_test`), `r33.prefill` (`qwen4exp_prefill_test`, `_eager`), `r33.golden` (synthetics; Intel's `--layers 18` after 21a's data), `r33.pp` (two cards), `r33.speed` (opt-in, Task 5). `python3 tools/box_validate/test_box_validate.py` passes.
- [ ] **Step 3: commit** `git commit -S -m "docs: spec 21 (21d as built), box queue row 33 - Qwen3.8-Flash-Next prefill"`.

### Task 5: prefill speed (box)

- [ ] Intel's checkpoint `--layers 18` (one card) and `--layers 38` (`--pp 2`), `l0`: `b70-decode <snap> --layers N --bench --prefill-length P` for P = 4096, 32768, 131072, interleaved pairs, median of 3, `uptime`; per-device busy time per chunk; flash vs eager at 32768; a profile of one chunk: the dequant pass, the HC kernels at M = C (the HC up is ~6.7 GFLOP a chunk an HC on plain FMA, ESTIMATED ~0.5 ms), the scores (fp32 FMA; the DPAS GEMM form is the lever), the sparse flash. Recorded levers, not built here: spec 16c's overlapped chunk pipeline, HC up through DPAS slabs, the scores as one GEMM, an SLM-fused int4 grouped GEMM (spec 15d Task 1's arm) removing the dequant pass, and spec 22's prefill streaming once the full model runs. `docs/BENCHMARKS.md` "Qwen3.8-Flash-Next (spec 21)" prefill rows. **Commit** `git commit -S -m "spec 21d: Qwen3.8-Flash-Next prefill speed on the truncated model"`.

**Gate for the plan:** Mac - host tests and `qwen4exp_run` green, cmdlines additions only. Box - F0 (incl. `pf_gated_head` unchanged); `qwen4exp_pf_kernels_test`; F4 on prefill (replay, chunking, splits at multiples of 64 bitwise, injected chunked == whole); F3 on prefill for both synthetics (Intel's `--layers 18` when its data exists); two cards bitwise equal to one; speed rows recorded.

# Spec 21c - Qwen3.8-Flash-Next decode: hyper-connections, QSA, PLE zero-copy, 512-expert MoE, one card then two

**Status (2026-10-09): built on the Mac (branch `spec21c-qwen4exp-decode`; spec 21 §14 "21c as built"); nothing has run on a card - box queue row 32.** Built blind on the Mac; every kernel's first compile and run is
the box's. Box queue row 32 (it renumbers at build time if taken). Every real-weight gate SKIPs (77) until
Intel's checkpoint and 21a's golden sets exist on the box.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the `qwen4_exp` family decodes through replayed Level Zero lists on its own engine (K2's / Kolibri's pattern): the kernels of spec 21 §4.1-4.4, the decode list, `b70-decode <qwen4exp> --layers N`, first on one card (the synthetic real-width checkpoints, Intel's checkpoint truncated), then across two with spec 16b's pieces; F2 on every kernel, F3 and F4 on decode; the PLE gather's zero-copy rate measured once for spec 22's P0; the decode speed rows on what fits.

**Architecture:** spec 21 §4, §6. `runtime::qwen4exp::Qwen4ExpEngine` (`src/runtime/qwen4exp/`) owns 21b's `loader::Q4LoadedModel`, one buffer set and one captured list per device. The residual is `H` bf16 [M][4][2560]. **One gated residual = three launches:** `q4_hc_combine_norm` (the pending `H = H0 + y ⊗ inj` of the block before - vLLM's fusion, `V/nvidia/hyperconnection.py:177-203` - written back to `H`, then the grouped `(1 + w)` norm -> `xn`), `gemv_bf16` at `{10240, 336}` (down's 320 rows and `block_inject`'s 4, padded; `src/kernels/gemv_bf16.cl` unchanged, fp32 out), `q4_hc_up_mix` (the bf16 rounding of down's rows, `silu(/4)`, `2 · sigmoid(/4)` -> `inj`, the 320 -> 10240 up, `sigmoid`, `⊙ xn`, the mean over streams -> the block input `x` bf16 [M][2560]). Per layer:

```text
GDN layer (36), 15 launches:  combine_norm · hc_down · up_mix · qkvz GEMV (S 1) · a||b gemv_bf16 · gdn_step (Qwen3.8's) ·
                              prep_gated_head _SIG · out_proj GEMV · combine_norm · hc_down · up_mix · router gemv_bf16 (528) ·
                              q4_route · q4_moe_gate_up · q4_moe_down
QSA layer (12), 19 launches:  combine_norm · hc_down · up_mix · q||gate||k||v GEMV · indexer gemv_bf16 (640) · attn_prep _Q24KV2
                              (attn.cl, unchanged: (1 + w) q / k norms, partial RoPE 64 dims, the KV write, the gate) ·
                              q4_qsa_prep (the indexer) · q4_qsa_score · q4_qsa_select · q4_qsa_attn · q4_qsa_reduce · o_proj GEMV ·
                              combine_norm · hc_down · up_mix · router · q4_route · q4_moe_gate_up · q4_moe_down
PLE (layer 1), +4:            combine_norm _NN (materialise H, no norm) · q4_ple_gather · key||value gemv_bf16 (12800) ·
                              q4_ple_block (gate, norms, the dilated conv over its ring, H += out), then layer 1's attn side
                              starts with the norm-only combine_norm _X
head, 7:                      embed_gather (device 0, before layer 0; layer 0's combine_norm _E repeats it into 4 streams) ·
                              final mixer combine_norm · gemv_bf16 {10240, 320} · up_mix (no inject) · lm_head · argmax x 2
```

779 launches per token at 48 layers on one card (derived: 1 + 36 x 15 + 12 x 19 + 4 + 6; the plan test asserts the formula, spec 21 §7 estimated ~770). Two cards add one: device 0 ends with a combine-only `_NN` so the **materialised** `H` crosses (10240 bf16, 20 KB, as vLLM's PP carries it), device 1 starts with the norm-only `_X`. Experts are addressed by id inside the MoE kernels through ONE macro over the layer's base pointer (spec 15 decision 3), which spec 22 replaces with its indirection table without touching the rest.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest; Python 3 + torch in `agnes-ref-img` with 21a's 5.19.0 site (the fixture generator, the synthetic golden sets).

**Spec:** `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§2, §3, §4.1-4.4, §4.6, §6, §7 F0, F2, F3, F4 and speed, §8 21c, §10 decisions 2, 3, 8, 10). Spec 22 §3 P0.3, §4 (the indirection table, the miss path reusing the PLE mechanism). Facts: 21a's `docs/probe-qwen4exp-<date>.md`. Precedents: plan 20c (`2026-10-06-spec20c-kolibri-decode.md`) task for task, `src/kernels/kolibri/*.cl`, `src/kernels/kolibri_kernels.h`, `src/runtime/kolibri/kolibri_{sizes,buffers,capture,engine}.*`, `src/cli/kolibri_decode.h`, `tests/kernels/kolibri_{ref.h,ref_test.cc,kernels_test.cc,variant_names_test.cc}`, `tests/golden/{kolibri_golden_test,kolibri_partial_test}.cc`, `tests/runtime/{kolibri_plan_test,kolibri_decode_test,kolibri_pp_test}.cc`, `tools/mac/clrun/kolibri_run.cc`, `src/kernels/attn.cl` (`attn_prep` at `FA_Q_HEADS` / `FA_KV_HEADS`, Ornith's `_Q16KV2`), spec 18 §10.1 (the flash / eager switch rule).

## Dependencies and branch points

- **Plans 21a and 21b merged.** 21a: `qwen4exp_ref.py` (the fixture generator imports its ops), the golden layout, the facts sheet's rounding points and tie rule. 21b: `model::Qwen4ExpDesc`, `loader::load_qwen4exp` and its layouts, `runtime::qwen4exp::qwen4exp_sizes` (this plan adds the decode scratch, the hand-off and the launch counts to it), the synthetic checkpoints.
- **Spec 16b is merged** and Kolibri (20c Task 6) added the descriptor-free pieces this plan needs: `runtime::pp_landing_layout(size_t resid_bytes, size_t sumsq_bytes)` (`src/runtime/pipeline_plan.h:118`) and `PipelineLink(l0::Context&, l0::Context&, const PpLandingLayout&, PpHandoff)` (`src/runtime/pipeline_engine.h:116`), plus `l0::Context(const Context&, uint32_t)`, `SyncEvent`, `CmdList::barrier_signal` / `wait_event`, `Fence::wait_for`, `runtime::pp_balance`, `PipelineOptions`, the `pp_handoff` binary, `cli::PipelineArgs` / `parse_pipeline_split` / `parse_pipeline_handoff` (`src/cli/pipeline_args.h`), `require_two_devices`. Nothing in 16b's files changes: the hand-off here is one region (`H`, 20 KB) with no norm sums (`sumsq_bytes` 0 -> the layout's second region empty; checked in Task 4). The switches are 16b's as renamed on 2026-10-06: `--pp N` / `--pipeline-parallel-size N`, `--pipeline-split auto|N`, `--pipeline-handoff copy|peer`.
- **Decision 10 (decode attention):** this plan builds the proposal - one sparse kernel at every depth (below 2052 visible positions the selection writes the identity list) - and, for the evidence only, `attn_v2.cl` at `_Q24KV2` (one new CMake line) timed against it in the kernel test's bench mode. The engine binds only the sparse kernel.
- **Decision 8 (KV):** bf16. `--kv-cache int8` is refused naming decision 8 (`kv8.cl` hard-codes 24 q / 4 kv heads, `kv8.cl:79-84`).
- **Decision 3 (τ):** the selection gate S compares block sets except where the 512th / 513th scores are within `B70_Q4_SEL_TOL` (PROPOSED from 21a's printed gap distribution); the routing diagnostic likewise with `B70_Q4_TIE_TOL` for the 10th / 11th.
- **Decision 2 (HC precision):** bf16 only.
- **Not here:** prefill (21d: `--prefill*` refused naming it), MTP (21e: `--mtp` refused naming it), serving (21e), the full model (spec 22: refused by 21b's `require_fits`).

## Global Constraints

- Branch `spec21c-qwen4exp-decode` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked (`tools/box_validate/data.sh`).
- **F0, nothing moves:** every existing kernel binary keeps its name, command line and bytes (`tools/kernel_cmdlines` additions only; G0's sha256 on the box). Reused sources run at this family's shapes from new CMake lines. **One existing source changes:** `src/kernels/prep.cl` gains a `GDN_GATE_SIGMOID` define that selects `sigmoid` for `silu` in `prep_gated_head` (spec 21 §4.4) - the precedent is 9a67889 (Ornith's overridable head defines): every existing variant's command line unchanged and its preprocessed source unchanged; if G0 shows any existing binary moved, the variant moves to a copy of the kernel in `src/kernels/qwen4exp/q4_gdn.cl` and `prep.cl` is restored (recorded). Binaries other spec blocks already define are bound by name and built here only when that block is off (`if(NOT B70_K2)`, `if(NOT B70_KOLIBRI)`: `gemv_bf16_M1_K2560_N128_C16_S16` is K2's router, `gemv_M1_K6144_N2560_S4_L0` K2's / Kolibri's, `gemv_bf16_M1_K6144_N2560` Kolibri's).
- **Semantics are 21a's, pinned** (the facts sheet): every rounding point of spec 21 §2 in the kernels and in `tests/kernels/qwen4exp_ref.h`, edited together; exact ties to the lower block index (decision 3) and the lower expert id; the experts combine in the oracle's order.
- Family kernels in `src/kernels/qwen4exp/`, named `q4_*`, every shape define in the binary name; host half `src/kernels/qwen4exp_kernels.h` (namespace `kernels::qwen4exp`).
- **Determinism:** no atomic in any sum that feeds a result; every reduction a fixed tree; the selection's ascending list independent of work-group timing.
- Every number measured, or marked derived / estimated / proposed. Box: every GPU command under `flock ~/b70-gpu.lock` (once for both cards in a two-card job), detached, polled; interleaved pairs after warm-up, median of 3, `uptime` and idle grade recorded; `-j44` builds.
- Mac checks: `tools/mac_check.sh --base main --kernels` (host tests, Level Zero syntax, kernel command lines, OpenCL syntax, Mac GPU runs of the kernels without DPAS or sub-group built-ins). Python tests in `agnes-ref-img` with `--memory 28g`.
- No `rm -rf`. Signed commits on the branch; no merge, no push.

## Review Focus

1. **The pending combine is never lost or applied twice.** Every block's `y` and `inj` are folded into `H` exactly once, by the NEXT `combine_norm` (or the PLE layer's `_NN`, or the final mixer's), and `H0` is the materialised `H` before that block's HC. Task 1's host test runs three layers both ways - the reference's per-layer order and the fused kernel order - and requires bitwise equality of `H` after every layer; a crafted `inj = 0` makes a block vanish, `inj = 2` doubles it.
2. **The selection at its edges.** Row positions 2049, 2050 (identity: 2050 and 2051 visible), 2051 (513 complete blocks: the first real cut), `(p + 1) % 4 == 0` (p's own block a candidate; p attended only if its block wins), exact ties at the 512 / 513 cut (lower block index), a tail of 0 and 3 positions; the list ascending, the count `2048 + tail`. Task 1's reference and Task 2's kernel test cover each; under replay the grid spans `max_len / 4` blocks with an early-out past the row's complete blocks.
3. **The compressed key is formed once, from raw keys, at the block's first position.** `(p + 1) % 4 == 0` writes block `(p + 1) / 4 - 1`: fp32 mean of the four raw (un-normed, un-roped) keys -> bf16 -> `k_layernorm` -> RoPE at `p - 3`. The raw keys of the open block live in an **8-slot** tail ring (`p % 8`): with up to 4 rows in one launch (21e's verify), rows write positions `pos..pos + 3` while a completing row reads `pos - 3..pos - 1` - 4 slots would collide, 8 cannot (spec 21 §4.2 says 4; recorded).
4. **The MoE at 512 / top-10.** 512 experts on 256 lanes (two a lane, Kolibri's `kol_route`), `ushort`-free decode ids in u32 words; renormalisation in the reference's order; ties at the 10th / 11th to the lower id; the shared expert's gate as row 512 of the router GEMV; slot 10 the shared expert (int4 for ours, bf16 for Intel's); the combine order the oracle's. Task 2's kernel test runs each slot alone (others' weights zero) and a tie at the cut.
5. **The PLE gather reads host memory by a device-computed index and nothing else.** No host doorbell, no per-token copy: the 16 ids are hashed on the device from Control's token and the position-indexed id ring, the rows read from the 16 host-USM ranges through the device pointer table (21b's `Q4PleTable::ptrs`), dequantised, and the ring advanced - all inside the replayed list. The EOS rule, a history before position 2, and a token at id 248319 are Task 1's cases; the gate is bitwise ids and bitwise rows.
6. **The two-card cut is invisible.** Logits, routes, selections, KV, compressed keys, GDN state, PLE state after a prompt and 32 greedy tokens: `--pp 1` and `--pp 2 --pipeline-split 2` (synthetic, 4 layers) bitwise equal, under `copy` and `peer`.

---

### Task 1: the fixture, the synthetic golden sets and the host reference (no card)

**Files:**
- Create: `tools/oracle/qwen4exp_fixture.py`, `tests/kernels/qwen4exp_fixture.h` (generated, committed), `tests/kernels/qwen4exp_ref.h`, `tests/kernels/qwen4exp_ref_test.cc`
- Modify: `tools/box_validate/qwen4exp_oracle.sh` (21a's: a `synth` mode), `tools/box_validate/data.sh` (`have oracle_q4exp_synth` checks the golden files too), `tests/CMakeLists.txt` (a block `# ==== Spec 21c: Qwen3.8-Flash-Next decode (label qwen4exp) ==== (begin)` after 21b's)
- Test: `qwen4exp_ref_test` (host)

**Interfaces:**
- Consumes: `tools/oracle/qwen4exp_ref.py` (its restated ops: the HC chain, the PLE hash and block, the indexer, the selection, the route, the combine, the gated GDN norm), 21b's synthetic checkpoints.
- Produces: `tests/kernels/qwen4exp_fixture.h` (seeded inputs and the reference's outputs at the real widths, as `uint16_t` / `float` / `uint32_t` / `uint64_t` arrays: (a) three layers' HC chains with `inj` 0 / 1 / 2 and random; (b) PLE: 64 histories -> ids, and one block's output for a random `H` and table rows; (c) the indexer: q heads normed + roped at positions 0, 2047, 2050, 2051, 9000, compressed keys of blocks completing at 3, 2047, 9003; (d) scores and selections at 600 / 513 / 512 complete blocks with planted exact ties at the cut; (e) the route at 512 / top-10 with a tie at the 10th and an all-`-1e30`-but-10 row; (f) the combine of 10 experts + the shared slot; (g) the gated GDN norm with `sigmoid`). `qwen4exp_oracle.sh synth`: builds 21b's two synthetic checkpoints if absent, then `qwen4exp_ref.py run` on each for `short`, `4k`, `agentic` with `--gen 32` and the engine-format PLE (`--ple int8:<dir>`) -> `oracle-out-q4exp-synth/{ours,intel}/<p>.{ids,golden.safetensors,log}`.
- Produces (`tests/kernels/qwen4exp_ref.h`, namespace `q4ref`, used by Tasks 2-3 and 21d / 21e's references): one function per kernel chain below, the selection as a `std::partial_sort` with the (score desc, block asc) comparator, the id-ring PLE history rule `t1 = id[p-1] (EOS if p < 1); t2 = t1 == EOS ? EOS : id[p-2] (EOS if p < 2)` (21a's reading of M:1107-1121, restated), `ple_ids`, `hc_combine_norm`, `hc_up_mix`, `qsa_prep`, `qsa_scores`, `qsa_select`, `qsa_attn_eager`, `qsa_attn_fp64`, `route`, `moe_combine`, `gated_head_sig`.

- [ ] **Step 1: the failing test** `qwen4exp_ref_test.cc`: every fixture case bit for bit against `qwen4exp_ref.h`, plus Review Focus 1's two-order run and Review Focus 2's edge rows on the host; the ids exact.
- [ ] **Step 2: register, run, expect FAIL** (`ctest --preset mac-host -R '^qwen4exp_ref_test$'`); write `qwen4exp_ref.h` until PASS.
- [ ] **Step 3: generate the fixture** (`agnes-ref-img` with the 5.19.0 site: `python3 tools/oracle/qwen4exp_fixture.py > tests/kernels/qwen4exp_fixture.h`; the header's comment records the command and the torch / transformers versions) and the `synth` mode (`DRY_RUN=1` prints its plan; the run itself is the box's `r32.oracle_synth`).
- [ ] **Step 4: commit** `git commit -S -m "tests: Qwen3.8-Flash-Next decode host reference and fixture - HC chain, PLE ids, indexer, selection, route at 512 / top-10 (spec 21c)"`.

### Task 2: the kernels

**Files:**
- Create: `src/kernels/qwen4exp/q4_hc.cl`, `src/kernels/qwen4exp/q4_ple.cl`, `src/kernels/qwen4exp/q4_qsa.cl`, `src/kernels/qwen4exp/q4_qsa_attn.cl`, `src/kernels/qwen4exp/q4_qsa_attn_eager.cl`, `src/kernels/qwen4exp/q4_moe.cl`, `src/kernels/qwen4exp/q4_expert.h` (the address macro), `src/kernels/qwen4exp_kernels.h`, `tests/kernels/qwen4exp_variant_names_test.cc`, `tests/kernels/qwen4exp_kernels_test.cc`, `tools/mac/clrun/qwen4exp_run.cc`
- Modify: `src/kernels/prep.cl` (the `GDN_GATE_SIGMOID` define, F0's rule above), `src/kernels/CMakeLists.txt` (a block `# ==== Spec 21c: Qwen3.8-Flash-Next decode (model::qwen4exp()) ==== (begin) / (end)` after spec 20d's, option `B70_Q4EXP` ON), `tools/mac_check.sh` (section 5's driver loop gains `qwen4exp`, its extra host sources like kolibri's), `tools/mac/opencl/intel_shim.h` only if a new block-read shape is used (said in the commit), `tests/CMakeLists.txt`
- Test: `qwen4exp_variant_names_test` (host), `qwen4exp_kernels_test` (card), `qwen4exp_run` (Mac GPU, indicative)

**Interfaces:**
- Consumes: Task 1; 21b's layouts (`loader/qwen4exp_layout.h`).
- Produces (`src/kernels/qwen4exp_kernels.h`):

```cpp
namespace kernels::qwen4exp {
inline constexpr unsigned kTailSlots = 8;                       // Review Focus 3
inline constexpr unsigned kListMax = 2052;                       // 2048 + 3 tail + 1 (u32 positions per row)
inline constexpr unsigned kScoreWg = 256, kSelectWg = 1024, kAttnTgt = 32, kRouteLanes = 256;
inline constexpr unsigned kUpKs = 4, kDnKs = 2;                  // q4_moe K splits, PROVISIONAL (Task 7 sweeps)
namespace route {                                                // the route row, u32 words per (layer, row)
inline constexpr unsigned kWords = 32, kIds = 0 /* 10 ids, the combine's order */, kWeights = 16 /* f32 of the bf16 weight */,
                          kSharedGate = 26 /* f32 of bf16 sigmoid */, kP10 = 27, kP11 = 28 /* diagnostics */;
}
enum class HcSrc { Embed, Slices, Y, None };                     // what the combine folds: the embedding (layer 0),
                                                                 // a GEMV's split-K slices, the MoE's bf16 y, nothing
std::string hc_combine_norm_variant(unsigned M, HcSrc src, unsigned S, bool norm);   // "q4_hc_combine_norm_M1_S4" | "_Y" | "_E" | "_X" | ..."_NN"
std::string hc_up_mix_variant(unsigned M, bool inject);         // "q4_hc_up_mix_M1_I" | "q4_hc_up_mix_M1"
std::string ple_gather_variant(unsigned M, bool bf16_scale);    // "q4_ple_gather_M1_F32" | "_BF16"
std::string ple_block_variant(unsigned M);                      // "q4_ple_block_M1"
std::string qsa_prep_variant(unsigned M);  std::string qsa_score_variant(unsigned M);  std::string qsa_select_variant(unsigned M);
std::string qsa_attn_variant(unsigned M, bool eager);           // "q4_qsa_attn_M1_T32" | "q4_qsa_attn_eager_M1"
std::string route_variant(unsigned M);                          // "q4_route_M1_E512_T10_N528_L256"
std::string moe_variant(unsigned M, bool shared_bf16);          // "q4_moe_M1_E512_T10_D2560_I640_SH4" | "_SHB"
std::vector<std::string> decode_variants(const model::Qwen4ExpDesc& d, bool int8_head, bool eager, bool two_cards);
}
```

The kernels (each chain is the reference's, `qwen4exp_ref.h` repeats it):

```text
q4_hc.cl  (plain OpenCL C: runs on the Mac)
  q4_hc_combine_norm(ctrl, H, src, inj, norm_w, xn)   grid (4 streams, M), WG 256
      _E: H[m][s][k] = x_embed[m][k];  _S<s>: y = rne(Σ_{j<S} slices[j][m][k]);  _Y: y = y_bf16[m][k];  _X: nothing
      H[m][s][k] = rne(f32(H0) + f32(rne(f32(y[m][k]) · f32(inj[m][s]))))          (product rounded, then the add)
      unless _NN: xn[m][s][k] = rne(f32(H) · rstd_s · (1 + w)[s·2560 + k]),  rstd_s over stream s (a fixed tree), fp32
  q4_hc_up_mix(down_f32, up_w, xn, x, inj)            grid (2560 / 16, M), WG 64
      d = rne(down_f32[m][j]) for j < 320 (+ 4 inject rows):  a_j = rne(silu(f32(rne(f32(d_j) / 4))));  inj_s = rne(2 · rne(sigmoid(rne(d_{320+s} / 4))))
      g[s][c] = rne(sigmoid(f32(rne(Σ_j a_j · up[j][s·2560 + c]))))  (the up linear's bf16 output first)
      x[m][c] = rne(mean_s(f32(rne(f32(g[s][c]) · f32(xn[s][c])))))   (21a pins the mean's order); work-group 0 writes inj[m][4]
q4_ple.cl  (plain loads of host USM; runs on the Mac against host memory)
  q4_ple_gather(ctrl, ple_ids_ring, ptrs, e)           grid (16 heads, M), WG 160
      t0 = cur_token[m]; t1, t2 by the id-ring rule; mixed = (t0·m0) ^ (t1·m1) [^ (t2·m2)] (uint64);  id_h = mixed % prime_h + offset_h
      e[m][h·160 + i] = rne(f32(q_h[id_h - offset_h][i]) · scale_h[id_h - offset_h]);  ple_ids_ring[(pos + m) % 16] = t0
  q4_ple_block(H, kv_f32, w, conv_ring, H_out)         grid (4 streams, M): key = grouped_norm(rne(kv[..10240])), value = rne(kv[10240..]),
      s = <key_s, grouped_norm(H)_s> / √2560; gate = sigmoid(sign(s) · √max(|s|, 1e-6)); gated = rne(value · gate);
      conv_ring[(pos + m) % 16] = norm_conv(gated);  out = gated + rne(silu(dwconv over rows p, p-3, p-6, p-9));  H += out
q4_qsa.cl  (plain OpenCL C)
  q4_qsa_prep(ctrl, idx_f32, small, rope, idx_q, tail, idx_keys)   grid (5, M), WG 128
      heads 0..3: idx_q[m][h] = rope64(rne(rne(idx_f32) · rstd · (1 + w_q)))  (fp32 of bf16 values);  WG 4: raw key -> tail[(pos+m) % 8];
      if (pos + m + 1) % 4 == 0: block b = (pos + m + 1) / 4 - 1 from raw keys (this launch's rows from idx_f32, older from the tail):
      idx_keys[b] = rope64_at(4b)(rne(rne(mean_f32(4 keys)) · rstd · (1 + w_k)))
  q4_qsa_score(ctrl, idx_q, idx_keys, scores)          grid (max_len / 4 / 256, M), WG 256; early-out past n = (pos + m + 1) / 4
      scores[m][b] = (Σ_h relu(Σ_i q[h][i] · k[b][i])) / √128   (fp32, a fixed order)
  q4_qsa_select(ctrl, scores, list, count, diag)       grid (1, M), WG 1024
      n <= 512: list = 0..p;  else the top 512 by (score desc, block asc) - radix select on the order-preserving key - then the
      blocks ascending, expanded to 4 positions each, then the tail 4n..p;  count = 2048 + (p + 1 - 4n);  diag = {s_512, s_513}
q4_qsa_attn.cl  (spec 10 v2's structure: sub-groups, block reads; card only)
  q4_qsa_attn(ctrl, attn_q, kv_k, kv_v, list, count, part)   grid (2 kv heads, kAttnTgt, M): each work-group a contiguous slice of the
      list, its 12 q heads, fp32 online softmax at scale 1/16, positions in ascending order
  q4_qsa_reduce(ctrl, part, attn_gate, attn_out)       grid (24, M): slices merged ascending, o / l, rne, × rne(sigmoid(gate)) (attn.cl's reduce)
q4_qsa_attn_eager.cl  (the reference's bf16 chain over the list: s = rne(rne(q·k) / 16), p = rne(softmax_f32), o = rne(Σ p·v), × the gate;
                       plain OpenCL C, runs on the Mac; B70_Q4_ATTN=eager binds it instead of attn + reduce)
q4_moe.cl  (Kolibri's kol_moe.cl structure at 512 / top-10 / 640; sub-group block reads under cl_intel_subgroups, plain loads otherwise)
  q4_route(logits_f32, route)                          grid (1, M), WG 256, lane l owns experts l and l + 256
      lg = rne(logits) (the router's bf16 output); p = softmax_f32(lg[0..511]); top 10 by rank (ties to the lower id); renormalised in
      the reference's order; w = rne(p_j / Σ); shared = rne(sigmoid(lg[512])); kP10 / kP11
  q4_moe_gate_up(route, x, w_gu, w_sh_gu, h)           11 slots: 0..9 at Q4_EXPERT_GU(w_gu, id) (q4_expert.h), 10 the shared expert
      h[m][slot][i] = rne(f32(rne(silu(f32(rne(Σ gate))))) · f32(rne(Σ up)))
  q4_moe_down(route, h, w_dn, w_sh_dn, y)              #pragma OPENCL FP_CONTRACT OFF; y_e = rne(Σ down_e); the combine in 21a's pinned
      order (eager: acc bf16, ascending id, acc = rne(acc + rne(y_e · w_e))); y = rne(acc + rne(shared · y_sh))   NOT folded into H
q4_expert.h
  #define Q4_EXPERT_GU(base, id) ((base) + (size_t)(id) * GU_BLK_U32)   - the only expert address; spec 22 swaps it for a table
q4_ple_check(ptrs, tags, out)                          one lane per 2 MiB page of each host range: the tag the loader wrote, read by the device
```

Reused at this family's shapes (new lines in the block): `embed_gather_M1_D2560` (VOCAB 248320), `gdn_step_M1` and `argmax_stage1_M1` / `argmax_stage2` (Qwen3.8's binaries, bound by name: the GDN shape is Qwen3.8's - 16 / 48 heads, 10240 conv rows, qkv‖z at S 1, the small block of `make_small_layout(…, 10240, 48)`; the argmax only if 21a found `vocab_used` 248077, else an `argmax_stage1_M1_V<n>` line), `prep_gated_head_M1_SIG` (`GDN_GATE_SIGMOID=1`), `attn_prep_M1_Q24KV2` (QKV_S 2) and `attn_prep_M1_S1_Q24KV2` (the bf16 arm), `gemv_M1_K2560_N16384_S1_L1` (qkv‖z int4), `gemv_M1_K2560_N13312_S2_L0` (q‖gate‖k‖v int4), `gemv_M1_K6144_N2560_S4_L0` (out_proj / o_proj int4; `if(NOT B70_K2 AND NOT B70_KOLIBRI)`), `gemv_bf16_M1_K2560_N16384`, `gemv_bf16_M1_K2560_N13312`, `gemv_bf16_M1_K6144_N2560` (Intel's dense arm; the last `if(NOT B70_KOLIBRI)`), `gemv_bf16_M1_K10240_N336_C16_S16` (HC down‖inject), `gemv_bf16_M1_K10240_N320_C16_S16` (the final mixer's down), `gemv_bf16_M1_K2560_N640_C16_S16` (indexer), `gemv_bf16_M1_K2560_N528_C16_S16` (router + shared gate), `gemv_bf16_M1_K2560_N128_C16_S16` (a‖b; `if(NOT B70_K2)`), `gemv_bf16_M1_K2560_N12800` (PLE key‖value), `gemv_bf16_M1_K2560_N248320` and `gemv_i8w_M1_K2560_N248320` (the head), and for decision 10's bench only `attn_v2_M1_T${ATTN_V2_TGT}_Q24KV2`. The block's count is recorded in the commit message (cmdlines `+N / -0 / ~0`).

- [ ] **Step 1: the failing host test** `qwen4exp_variant_names_test`: every name `decode_variants` returns for (int4 / bf16 dense) x (int8 / bf16 head) x (flash / eager) x (one / two cards) is a target of the block (`${B70_Q4EXP_DECODE_KERNELS}` as argv, as `kolibri_variant_names_test`). FAIL.
- [ ] **Step 2: write the sources, `q4_expert.h`, the header, the CMake block, the `prep.cl` define;** the names test PASSES.
- [ ] **Step 3: Mac GPU driver** `tools/mac/clrun/qwen4exp_run.cc`: builds `q4_hc.cl`, `q4_ple.cl` (host memory as the "table": the Mac's shared memory, indicative of nothing but the arithmetic), `q4_qsa.cl`, `q4_qsa_attn_eager.cl`, `q4_moe.cl` (plain-load path) and `prep.cl`'s `_SIG` at the real shapes, against `qwen4exp_ref.h` on random inputs and the fixture's cases; prints `qwen4exp_run: <kernel> exact` or the first differing index. Added to section 5's loop in `tools/mac_check.sh`.
- [ ] **Step 4: the card test** `qwen4exp_kernels_test` (label `qwen4exp`, no checkpoint): every binary of the block at the real shapes against `qwen4exp_ref.h` - bitwise for the HC kernels, the PLE gather (ids and rows, from real host-USM ranges) and block, the indexer prep and scores, the selection (Review Focus 2's rows, exact ties, 512 / 513), the route, the MoE (each slot alone, both shared forms), the eager attention, `prep_gated_head_M1_SIG`; flash over the list against fp64 at cosine >= 0.99999 (spec 6 K1's bar) at depths 1, 2050, 2051, 9000, 131072; `q4_qsa_attn` at depth < 2052 equal to `attn_v2_M1_T32_Q24KV2`'s dense result within the same bar. `--bench-attn` (opt-in): the sparse kernel against `attn_v2` at depths 512 / 1024 / 2048, interleaved pairs - decision 10's evidence. `--ple-rate` (opt-in): see Task 5.
- [ ] **Step 5: Mac gate** `tools/mac_check.sh --base main --kernels`: host PASS (incl. `qwen4exp_ref_test`, `qwen4exp_variant_names_test`), l0 PASS, cmdlines `+N / -0 / ~0`, opencl PASS, `qwen4exp_run` exact on every kernel it runs.
- [ ] **Step 6: commit** `git commit -S -m "kernels: Qwen3.8-Flash-Next decode - hyper-connections, PLE gather from host USM, QSA indexer / selection / sparse attention, 512-expert route and MoE, the sigmoid gated head (spec 21c)"`.

### Task 3: `Qwen4ExpEngine` on one card, the CLI, F3 / F4 on decode

**Files:**
- Create: `src/runtime/qwen4exp/qwen4exp_buffers.{h,cc}`, `src/runtime/qwen4exp/qwen4exp_capture.{h,cc}`, `src/runtime/qwen4exp/qwen4exp_engine.{h,cc}`, `src/cli/qwen4exp_decode.h`, `tests/runtime/qwen4exp_decode_test.cc`, `tests/golden/qwen4exp_golden_test.cc`, `tests/golden/qwen4exp_partial_test.cc`
- Modify: `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (21b's: decode scratch, launches), `src/runtime/qwen4exp/CMakeLists.txt` (library `b70_qwen4exp_runtime`), `tests/runtime/qwen4exp_plan_test.cc` (21b's: the launch counts), `src/cli/b70_decode.cc` (dispatch; the `--layers` refusal's text), `src/cli/b70_serve.cc` (refusal naming 21e), `src/cli/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `qwen4exp_plan_test` (host); `qwen4exp_decode_test`, `qwen4exp_golden_test`, `qwen4exp_partial_test`, `cli_reject_qwen4exp_*` (card / binary)

**Interfaces:**
- Consumes: Tasks 1-2; 21b's `loader::load_qwen4exp`, `Q4LoadedModel`, `runtime::qwen4exp::plan` / `require_fits` / `layers_that_fit`.
- Produces (21d adds `prefill`; 21e the snapshots and MTP):

```cpp
namespace runtime::qwen4exp {
enum class Q4Attn { Flash, Eager };
inline constexpr Q4Attn kDefaultQ4Attn = Q4Attn::Flash;      // until the box A/B (spec 18 §10.1's rule)
Q4Attn q4_attn();                                            // B70_Q4_ATTN=flash|eager; anything else throws
struct ScratchSizes { size_t H, xn, x, slices, down_f32, inj, hc_up, idx_f32, idx_q, scores, list, count, diag,
                      attn_q, attn_gate, attn_part, attn_out, logits_r, routes, moe_h, y, ple_e, ple_kv, logits,
                      argmax_part; size_t total() const; };
ScratchSizes scratch_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len, Q4Attn a);   // scores: [M][max_len / 4] fp32
size_t device_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, Q4Attn a, PpHandoff h);
size_t decode_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, Q4Attn a, PpHandoff h);
//   779 at 48 layers, one card, flash (1 + 36 x 15 + 12 x 19 + 4 + 6); eager: each QSA layer's attn + reduce become eager's
//   launches; two cards +1 (the _NN / _X pair), peer +2 more (pp_send, pp_recv)
PpLandingLayout landing_layout(const model::Qwen4ExpDesc& d);   // pp_landing_layout(10240 x 2, 0)
class Qwen4ExpEngine {
 public:
  Qwen4ExpEngine(std::vector<l0::Context*> devices, loader::Q4LoadedModel model, uint32_t max_len,
                 bool debug_tap = false, const PipelineOptions& opt = {});
  void reset();
  void ingest(const std::vector<uint32_t>& ids);
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);  uint32_t pending() const;  uint32_t vocab() const;
  // The injected run (spec 21 F3): while set, every QSA layer reads its list from this host-USM buffer
  // ([layer][kListMax] positions + count, written by the caller before each replay) and score / select are
  // not in the list - a separate capture, its launch count printed. Debug only.
  void set_injected_selection(bool on);  uint32_t* injected_list(uint32_t qsa_index);
  std::vector<uint16_t> read_debug_H();                 // bf16 [layers][10240] of the last replay (debug_tap)
  std::vector<uint32_t> read_routes();                  // u32 [layers][32]
  std::vector<uint32_t> read_selection(uint32_t layer); // the last replay's list (positions) and count
  std::vector<float> read_selection_diag();             // [12][2]: s_512, s_513
  std::vector<float> read_logits();  void read_logits_into(float* host);
  std::vector<uint16_t> read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v);
  std::vector<uint16_t> read_idx_keys(uint32_t layer, uint32_t first_block, uint32_t count);
  MemoryComponents memory_use(uint32_t dev) const;  std::string memory_line() const;
  uint32_t pos() const;  uint32_t max_len() const;  size_t launches() const;  uint32_t devices() const;
  uint32_t split() const;  PpHandoff handoff() const;  Q4Attn attention() const;
  void drop_next_handoff();                             // P4's test hook (Task 4)
  const loader::Q4LoadedModel& model() const;
};
}
// src/cli/qwen4exp_decode.h, namespace cli::qwen4exp: is_qwen4exp(path), DecodeArgs, check_args, settle (placement + max_len +
// --layers auto through layers_that_fit), run_decode<Redirect>(const DecodeArgs&) - cli/kolibri_decode.h's shape.
```

- [ ] **Step 1: failing host test** (`qwen4exp_plan_test` extended): `decode_launches` 779 / one card / flash, the eager count, +1 on two cards, +2 more with `peer`; `--layers 4` (the synthetic) 1 + 3 x 15 + 19 + 4 + 6 = 75; the scratch at max_len 262144 (the score row `[M][65536]` fp32 is the term that scales); the plan at `--layers 18` (one card) and 38 (two) at 32k context fits, at 48 refused (21b's `require_fits`).
- [ ] **Step 2: buffers, capture, engine (one device).** `capture` walks layers `[first, end)`, asserts `device_launches`, writes each layer's route row and selection diag, the debug tap after each layer's last launch. `generate` = `KolibriEngine::generate`'s contract. At construction: the PLE ranges' device readback (`q4_ple_check`) - a mismatch throws naming the range and page (the xe alias hazard, spec 22 §1).
- [ ] **Step 3: CLI.** `b70-decode` dispatches on `model_type` `qwen4_exp` (`cli::qwen4exp::is_qwen4exp`, placed beside Kolibri's dispatch and before `cli::check_pipeline`) to `cli::qwen4exp::run_decode`: `--ids`, `--n`, `--bench [--depth N] [--tg N]`, `--lm-head bf16|int8` (default bf16, as Kolibri's b70-decode), `--max-len N|auto`, `--layers N|auto` (**required** until spec 22: without it the full model is refused naming the bytes and spec 22), `--ple-dir DIR` (default 21b's `q4_ple_dir`), and 16b's `--pp N` / `--pipeline-parallel-size N` (default 1), `--pipeline-split auto|N`, `--pipeline-handoff copy|peer` (until Task 4 lands, `--pp 2` refused naming Task 4). Refused before the device, by name: `--prefill` / `--prefill-length` / `--prefill-chunk` / `--prefill-backend` (spec 21d), `--mtp` (spec 21e), `--kv-cache int8` (decision 8), `--profile`, `--device N` with `--pp 2`. The Qwen engine's `--layers` refusal (`b70_decode.cc`) keeps the substring `--layers belongs to Kolibri-1` that `cli_reject_kolibri_layers_qwen` expects and names this family too. `b70-serve` refuses `qwen4_exp` before the device naming spec 21e. Registered `b70_cli_reject` tests over `tests/model/qwen4exp` (config only): `cli_reject_qwen4exp_full` (no `--layers`: "spec 22"), `_prefill` ("spec 21d"), `_mtp` ("spec 21e"), `_kv8` ("decision 8"), `_serve` ("spec 21e").
- [ ] **Step 4: the card gates** (label `checkpoint;golden;qwen4exp`, SKIP 77 without data):
  - `qwen4exp_decode_test <ckpt> <ids>` (F4 on decode): plan == allocation; two `reset` + `ingest` + `generate(32)` runs bitwise (logits, routes, selections and diag, KV, compressed keys, tail rings, GDN state, PLE rings); the injected capture's launch count printed.
  - `qwen4exp_golden_test <ckpt> <oracle dir> [int8] [inject]` (F3, kolibri_golden_test's copy): the prompt by M = 1 replays, then `generate(32)`; golden_common.h's tie-aware token rule; the routing diagnostic per layer against `route.ids/w/gap.L*` (a set difference beyond `B70_Q4_TIE_TOL` fails); **gate S** per QSA layer and row against `qsa.sel.L*` (a block-set difference where `qsa.gap.L*` exceeds `B70_Q4_SEL_TOL` fails; near-tie rows counted and printed); **`inject`**: the reference's lists fed in, which must pass the determined-row gate on every row. The free run passes S and the greedy gate on rows with no near-tie in any layer and reports the others. Registered for both synthetics (`ours`, `intel`) x (`short`, `4k`, `agentic`), `_i8head`, `_eager`, `_inject` twins.
  - `qwen4exp_partial_test <intel ckpt> <oracle-out-q4exp> <oracle-out-q4exp-L18> 18` (development mode on the real checkpoint, one card): the tap of layers 0..17 against the full reference's `H.L*` (cosine per row: median >= 0.9998, min >= 0.99, PROPOSED, printed per layer), the routing diagnostic and gate S for layers < 18, and the token gate against the truncated reference `oracle-out-q4exp-L18`.
- [ ] **Step 5: Mac gate and commit** `git commit -S -m "runtime: Qwen4ExpEngine decodes on one card - synthetic checkpoints and Intel's truncated; F3 (tokens, routes, selection gate S, injected run), F4 replay (spec 21c)"`.

### Task 4: two cards (`--pp 2`, on spec 16b's pieces)

**Files:**
- Modify: `src/runtime/qwen4exp/qwen4exp_engine.{h,cc}`, `src/runtime/qwen4exp/qwen4exp_capture.{h,cc}`, `src/cli/qwen4exp_decode.h`, `tests/CMakeLists.txt`
- Create: `tests/runtime/qwen4exp_pp_test.cc`
- Test: `qwen4exp_pp_test`; K0: `pipeline_plan_test`, `pp_protocol_test`, `kolibri_pp_test` unchanged

**Interfaces:**
- Consumes: `PipelineLink(l0::Context&, l0::Context&, const PpLandingLayout&, PpHandoff)`, `runtime::pp_landing_layout(size_t, size_t)`, `StageLink`, `PipelineLink::binding`, `zero`, `SyncEvent`, `Fence::wait_for`, the `pp_handoff` binary (`pp_send` / `pp_recv` with `resid_words` 5120 and `sumsq_words` 0 - if `pp_handoff.cl` needs a non-zero count, one dummy word crosses, recorded), `require_two_devices`.
- Produces: the engine's two-device path (`set_token` writes both Control blocks; after device 1's fence the host copies its Control into device 0's; a lost hand-off host-signals the event, throws naming it and marks the engine until `reset()` - PipelineEngine's rule); `cli::qwen4exp::run_decode` with `--pp 2` (`require_two_devices`, the view, the peer check before the load, `runtime::qwen4exp::pp_split` for `--pipeline-split auto`, `describe` printed).

- [ ] **Step 1: the failing test** `qwen4exp_pp_test <synth ckpt> <intel ckpt> <ids>`: (a) synthetic 4 layers, `--pp 1` vs `--pp 2 --pipeline-split 2`, `copy` and `peer`: Review Focus 6, every state bitwise, both Control blocks equal after every phase; (b) Intel's checkpoint at `--layers 38`, split 19 vs 20: bitwise; (c) P4: `drop_next_handoff()` throws within the bounds and later steps throw until `reset()`; a pair without peer access refused naming `zeDeviceCanAccessPeer`; (d) `pp_landing_layout(20480, 0)`'s second region is empty and the flag still has its own page.
- [ ] **Step 2: register** (`LABELS "checkpoint;qwen4exp;pp"`, SKIP 77 with one GPU or no data); **implement**; `kolibri_pp_test`'s and 16b's host tests unchanged.
- [ ] **Step 3: Mac gate and commit** `git commit -S -m "runtime: Qwen3.8-Flash-Next across two cards - the materialised 4-stream H crosses, bitwise across splits (spec 21c)"`.

### Task 5: the PLE zero-copy rate (box; the number spec 22's P0 starts from)

**Files:**
- Modify: `tests/kernels/qwen4exp_kernels_test.cc` (`--ple-rate`), `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (one line in §1 "The facts it starts from": where the measured rate is recorded, once it is)

- [ ] **Step 1:** `qwen4exp_kernels_test --ple-rate [GB]`: allocates 16 host-USM ranges per 21b's loader rule (real primes, `GB` total, default 16 for a quick run, 51.8 for the real table's size) on device 0, then device 1, then both concurrently; times `q4_ple_gather` alone in a replayed list at M = 1 (16 rows, 2,560 B + scales a token) over 10^4 replays with random ids, the readback check every replay; prints the median µs per token, the implied bus rate, and whether every row read back equal. Also a "wide" mode reading 256 rows per launch, the closest thing this kernel has to an expert's access pattern (spec 22 P0.3 does the real one).
- [ ] **Step 2 (box):** run it (stage `r32.ple_rate`); the numbers go into row 32's summary and, by reference, into spec 22 §1 when they exist. **Commit** `git commit -S -m "spec 21c: the PLE gather's zero-copy rate, the first host-USM number for spec 22's P0"`.

### Task 6: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (section "21c as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 32), `tools/box_validate/stages.sh` (`row 32` block), `docs/19-running-models.md` (a Qwen3.8-Flash-Next row: decode only, `--layers N`)

- [ ] **Step 1:** "21c as built": the list per layer and its launch counts, the 8-slot tail ring, the reuse of `attn_prep` / `gdn_step` / `gemv_bf16`, the `prep.cl` define and what G0 must show, known deviations (flash's fp32 probabilities vs the eager bf16 chain; the mean's order), the expert macro spec 22 replaces.
- [ ] **Step 2: the queue row** (32): merged = branch and "21c as built"; unvalidated = every binary of the block never compiled by ocloc (listed), `q4_qsa_attn.cl`'s sub-group path, the host-USM gather on the card, the 779-launch list, two cards; how = stages `r32.k0` (G0: incl. `prep_gated_head_M1` unchanged), `r32.host`, `r32.k1` (`kbins` + `qwen4exp_kernels_test`), `r32.oracle_synth` (opt-in cpu: `qwen4exp_oracle.sh synth`), `r32.load` and `r32.k3` (`qwen4exp_decode_test` on the synthetics), `r32.golden` (synthetics: free, `_i8head`, `_eager`, `_inject`), `r32.partial` (Intel's, `--layers 18`; needs 21a's real sets), `r32.pp` (two cards; after row 22), `r32.ple_rate` (opt-in), `r32.bench_attn` (opt-in: decision 10), `r32.speed` (opt-in, Task 7); `rownote`s for the data each needs and the proposed tolerances. `python3 tools/box_validate/test_box_validate.py` passes; `--dry-run --only r32` prints the stages.
- [ ] **Step 3: commit** `git commit -S -m "docs: spec 21 (21c as built), box queue row 32 - Qwen3.8-Flash-Next decode"`.

### Task 7: decode speed (box)

- [ ] On what fits: Intel's checkpoint `--layers 18` on one card and `--layers 38` on two (`--pp 2`, `copy` and `peer`), our synthetic at the same N, `--bench --depth D --tg 256` for D = 1024, 4096, 32768 (the selection active from 2052), int8 and bf16 heads, `B70_Q4_ATTN` flash and eager - interleaved pairs, median of 3, `uptime` and idle grade; launches per token, per-device step time, the per-launch floor at this family's small kernels (the HC three are ~9 % of the launches; spec 21 §7: HC launch fusion is the first lever if the floor binds); the MoE kernels' `UP_KS` / `DN_KS` and the int4 rows' `{S, layout}` sweep (`qkvg`, `out_proj`, `o_proj`; `qkvz` stays S 1 for `gdn_step`). Against the derived roofline per layer (spec 21 §3: ~4.97 GB a token for the whole model at our formats; Intel's bf16 dense ~9.3 GB) - the full-model rows are spec 22's. `docs/BENCHMARKS.md` section "Qwen3.8-Flash-Next (spec 21)" decode rows. **Commit** `git commit -S -m "spec 21c: Qwen3.8-Flash-Next decode speed on the truncated model"`.

**Gate for the plan:** Mac - every host test, `qwen4exp_run` exact, cmdlines additions only, every syntax check. Box - F0 (G0: every pre-existing binary's sha256 unchanged, `prep_gated_head_M1` included); F2 on every kernel; F4 replay bitwise; F3 on both synthetics (tokens tie-aware, routing, gate S, the injected run passing every determined row); the partial forward on Intel's checkpoint within the proposed bars; two cards bitwise equal to one; the PLE rate and the speed rows recorded.

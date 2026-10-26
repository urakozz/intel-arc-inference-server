# Spec 24c - LFM2.5-24B-A2B decode: the launch-count P0 first, then the short conv, hd64 attention, the sigmoid-bf16 router and the 64 x 1536 MoE on one card, the fused arms, `--pp 2`

> DROPPED (operator, 2026-10-27): support removed from the engine in 4822e3e; kept for the record.
>
> Reusable later (revive from git: `git show 4c491f8:<path>` - 4c491f8, this branch's parent on main, is the last
> commit that has all of it):
> - the MoE router: `src/kernels/lfm2moe/lfm_moe.cl` (`lfm_router`; `lfm_route` - sigmoid, the expert bias for the selection only, renormalised weights; `lfm_moe_gate_up` / `lfm_moe_down`), `src/kernels/lfm2moe/lfm_pf_moe.cl`, `src/kernels/lfm2moe/lfm_expert.h`; twins `tests/kernels/lfm2moe_ref.h`, `tests/kernels/lfm2moe_pf_ref.h`; tests `tests/kernels/lfm2moe_ref_test.cc`, `tests/kernels/lfm2moe_kernels_test.cc`, `tests/kernels/lfm2moe_pf_kernels_test.cc`
> - the pythonic tool-call parser: `src/server/toolcall_lfm2.h`, `src/server/toolcall_lfm2.cc`, `tests/server/toolcall_lfm2_test.cc`; `tools/toolcall/score.py --format pythonic`, `tools/toolcall/lfm2moe_rules.cc`
> - the rest: the short conv `src/kernels/lfm2moe/lfm_conv.cl` / `lfm_pf_conv.cl`, head_dim-64 attention `src/kernels/lfm2moe/lfm_attn.cl` / `lfm_pf_attn.cl`, `src/model/lfm2moe.h`, `src/model/lfm2moe.cc`, `src/loader/lfm2moe_*`, `src/runtime/lfm2moe/` (`lfm_prep.cl` still live in `src/kernels/bolt/`)

**Status (2026-10-20): built on the Mac (branch `spec24c-lfm2moe-decode`), Tasks 1-7; not run on a card.** Box queue
row **51** (rows 42-47 are spec 23's). Task 1 Step 4 (P0 on the card, `r51.p0`) and Task 8 (speed, `r51.speed`) are
the box's, prepared as stages; every kernel's first compile and run is the box's. As built: spec 24 §14.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** measure first where an LFM2 decode step's time goes - a 383-launch skeleton list from existing binaries at LFM2's shapes, the same at 277 launches (every fusion applied), and a GEMV-only list, interleaved on the card (P0) - then decode the model through replayed Level Zero lists on its own engine (Kolibri's pattern): the family kernels of spec 24 §4.1-4.4, the list of §4.5 on one card, `b70-decode <lfm2moe>`, the per-launch anatomy (`--profile`), the three fused arms behind `B70_LFM_FUSE` (bitwise the as-built list), and `--pp 2` on spec 16b's pieces; F2 on every kernel, F3 and F4 on decode; the speed rows at depth.

**Architecture:** spec 24 §3, §4, §6. `runtime::lfm2moe::Lfm2MoeEngine` (`src/runtime/lfm2moe/`) owns 24b's `loader::LfmLoaded`, one buffer set and one captured list per device. The residual `h` is bf16 [M][2048] in **two** buffers (ping-pong across norm sites, spec 24 §4.4). Per layer:

```text
conv + MoE (28), 9:      res_norm _S<s> · in_proj GEMV · lfm_conv · out_proj GEMV · res_norm _S<s> · lfm_router · lfm_route ·
                         lfm_moe_gate_up · lfm_moe_down (folds y into h)
attention + MoE (10), 11: res_norm _NF · q‖k‖v GEMV · lfm_attn_prep · lfm_attn · lfm_attn_reduce · o GEMV · res_norm _S<s> ·
                         lfm_router · lfm_route · lfm_moe_gate_up · lfm_moe_down
dense conv (0-1), 8:     res_norm · in_proj · lfm_conv · out_proj · res_norm _S<s> · gate‖up GEMV · lfm_silu_mul · down GEMV
head, 5:                 lfm_embed_gather (device 0, first) · res_norm _NF + embedding_norm · head GEMV (int8 | bf16) ·
                         lfm_argmax stage 1 · stage 2
```

383 launches a token, one card (derived: 2 x 8 + 28 x 9 + 10 x 11 + 5; the plan test asserts the formula); eager attention +2 a layer (403); the arms `route` -38, `router` -38, `conv` -30 (277 with all three); two cards +0 with `copy`, +2 with `peer`. The first norm of a layer folds what the previous layer left unfolded (a dense layer's down slices) or only normalises (`_NF`, after `lfm_moe_down` folded); the operator norm of layer 0 normalises the gathered embedding.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake / ctest; Python 3 + torch in `agnes-ref-img` with 24a's 5.19.0 site (the fixture generator).

**Spec:** `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` (§2, §3, §4.1-4.5, §4.7, §6, §7 F0, F2, F3, F4 and speed, §8 24c, §10 decisions 6, 10, 11, 12, 13). Facts: 24a's sheet (the combine order 5.19.0 ran, the conv-accumulation count, the tie order, the routing gap distribution). Precedents: plan 20c and plan 21c task for task, `src/kernels/kolibri/{kol_prep,kol_moe,kol_attn,kol_attn_eager}.cl`, `src/kernels/k2/k2_moe.cl` (`k2_route`), `src/kernels/moe.cl` (the combine, `tile_dot`), `src/kernels/gdn_step.cl` (the conv ring), `src/kernels/kolibri_kernels.h`, `src/runtime/kolibri/kolibri_{sizes,buffers,capture,engine}.*`, `src/cli/kolibri_decode.h`, `tests/kernels/kolibri_{ref.h,ref_test.cc,kernels_test.cc,variant_names_test.cc}`, `tests/golden/kolibri_golden_test.cc`, `tests/runtime/{kolibri_plan_test,kolibri_decode_test,kolibri_pp_test}.cc`, `tools/mac/clrun/kolibri_run.cc`, `tools/probe/probe_replay.cc`, `src/runtime/capture.h:39-48` (`ProfileEvents`), docs/15-step-anatomy.md (the instrument).

## Dependencies and branch points

- **Plans 24a and 24b merged.** 24a: `lfm2moe_ref.py`'s restated ops (the fixture), the golden layout, the tie order, the gap distribution (`B70_LFM_TIE_TOL`). 24b: `model::Lfm2MoeDesc`, `loader::load_lfm2moe`, `lfm2moe_layout.h` (the expert address, the small block, `kConvRing`), `runtime::lfm2moe::plan` (this plan adds the decode scratch and the launch counts), the synthetic checkpoints. Real-weight gates need 24q's RTN export and its golden sets (`have lfm2moe_rtn`, `have oracle_lfm2moe_rtn`); `ours` stages need 24q's AutoRound export.
- **Per-linear layout is PROVISIONAL (24b review fix, I1) - this plan is where it gets swept.** 24b set, per `LfmLinear`, layout 1 (the shared subgroup-tiled form) for `in_proj`, attention `out_proj` / the dense down-projection and the routed experts, layout 0 (GPTQ-native) for the fused `q||k||v` and dense `gate||up`, with `S` left at 1 everywhere - a first guess, not a measured choice (`model::Lfm2MoeDesc::linears(l)` is the one place that would change). Every GEMV kernel this plan writes or calls must read `desc.linears(l)[...].shape.layout` (or the equivalent fixed-shape constant it already compiled for) and dispatch `gemv.cl`'s layout-0 or layout-1 form accordingly; `GEMV_BLOCK2D` is layout-0-only (`gemv.cl:29`), so a linear 24b provisionally set to layout 1 cannot use that kernel variant without first being re-provisioned to layout 0. Task 8's A/B (the fused arms) is a natural point to also measure whether swapping any of 24b's provisional picks changes decode throughput; if so, the change is 24b's descriptor, not a kernel hack.
- **Spec 16b is merged** with Kolibri's descriptor-free pieces: `runtime::pp_landing_layout(size_t, size_t)`, `PipelineLink`, `SyncEvent`, `require_two_devices`, `cli::PipelineArgs` (`src/cli/pipeline_args.h`).
- **Decision 6 (the default list):** built here: the as-built list and the three arms behind `B70_LFM_FUSE` (`none` default until Task 8 records the box's A/B; then the commit that changes the default names its rows).
- **Decision 10 (`--pp 2`):** Task 6 builds it on 16b's pieces; `--pp 1` stays the default (the model and 131072 positions fit one card).
- **Decision 11 (KV):** bf16. `--kv-cache int8` refused naming decision 11.
- **Decision 12:** every Kolibri- / K2-derived kernel is a family copy in `src/kernels/lfm2moe/`; no existing `.cl` is edited.
- **Not here:** prefill (24d: `--prefill*` refused naming it), serving (24e: `b70-serve` refuses `lfm2_moe` naming it), MTP (none: `--mtp` refused naming "no MTP head").

## Global Constraints

- Branch `spec24c-lfm2moe-decode` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked (`tools/box_validate/data.sh`).
- **F0, nothing moves:** every existing kernel binary keeps its name, command line and bytes (`tools/kernel_cmdlines` additions only; G0's sha256 on the box); no existing `.cl` edited. Binaries other spec blocks already define (e.g. Ornith's `prep_res_fold_M1_K2048_SP0_G20`) are bound by name and built here only when that block is off (`if(NOT B70_ORNITH)` etc.).
- **Semantics are 24a's, pinned:** every rounding point of spec 24 §2 in the kernels and in `tests/kernels/lfm2moe_ref.h`, edited together; exact ties to the lower expert id; the combine in the order 24a recorded.
- Family kernels in `src/kernels/lfm2moe/`, named `lfm_*`, every shape define in the binary name; host half `src/kernels/lfm2moe_kernels.h` (namespace `kernels::lfm2moe`); a CMake block `# ==== Spec 24c: LFM2.5-24B-A2B decode (model::lfm2moe()) ==== (begin) / (end)`, option `B70_LFM2MOE` ON.
- **Determinism:** no atomic in any sum that feeds a result; every reduction a fixed tree; the fused arms bitwise the as-built list (same op order, only the launch boundary moves).
- Every number measured, or marked derived / estimated / proposed. Box: every GPU command under `flock ~/b70-gpu.lock` (once for both cards in a two-card job), detached, polled; timed arms as interleaved pairs after a warm-up, median of 3, `uptime` and the idle grade recorded; `-j44` builds.
- Mac checks: `tools/mac_check.sh --base main --kernels`. Python in `agnes-ref-img` (`--memory 8g --cpus 4`).
- No `rm -rf`. Signed commits; no merge, no push.

## Review Focus

1. **The conv ring holds `Bx`, at `pos % 16`, and a row's taps come from the right positions.** `lfm_conv` writes `Bx` of row m at slot `(pos + m) % 16` and convolves `w0 · Bx[p-2] + w1 · Bx[p-1] + w2 · Bx[p]` (taps oldest first, spec 24 §2.5); positions < 0 read 0 (a fresh sequence, after `reset`); rows of one launch never read a slot another row of the launch writes (M ≤ 14). Task 2's reference and Task 3's kernel test cover positions 0, 1, 2, a wrap at 15 / 16 / 17, and M = 1..4 rows in one launch (the M-row form 24d and a future verify list use).
2. **The router's bf16 chain, ties and the bias.** `s_b = rne(sigmoid(rne(l)))`, `sel = s_b + bias` (fp32), rank with exact ties to the lower id, `w = rne(s_b / rne(rne(Σ s_b) + 1e-6))`; the route row in **rank** order; the 5th `sel` written for the near-tie diagnostic. Task 3 tests a bias that reorders, a tie at the 4th / 5th, all-equal logits, and a NaN-free row with one dominant expert.
3. **The norm's two roundings and the ping-pong.** `x = rne(rne(h · rstd) · w)` with `rstd = 1 / sqrt(Σ / 2048 + 1e-5)` (never `rsqrt`); every work-group of a `res_norm` launch computes the identical Σ (one fixed tree over the folded row) and writes only its chunk; the fold reads `h_in` and writes `h_out` (a different buffer), so no work-group reads a chunk another rewrites. The capture asserts the buffer alternation per site; Task 4's decode test reads both buffers after every layer.
4. **The fused arms are the same arithmetic.** `route` (the route recomputed in every gate‖up work-group), `router` (the 64 router rows computed inside `ffn_norm`'s launch, from the identical normalised row), `conv` (the conv as the in_proj GEMV's epilogue on the conv-tile layout, both at split-K 1): each arm's logits, routes, conv rings and KV bitwise the as-built list's after a prompt and 32 greedy tokens (Task 5).
5. **The tied head and the vocabulary edge.** The int8 head is the embedding quantised (24b); `--lm-head bf16` reads the same tiles the gather reads; the argmax never returns an id ≥ 125017 even when a crafted row makes 127999 the largest logit.
6. **The two-card cut is invisible.** Logits, routes, KV and conv rings after a prompt and 32 greedy tokens: `--pp 1` and `--pp 2` (synthetic N 4, split 2; RTN export, split 20) bitwise equal under `copy` and `peer`.

---

### Task 1: P0 - where the step's time goes, before any LFM2 kernel (box first)

**Files:**
- Create: `tools/probe/probe_lfm_step.cc`, `tools/probe/lfm_step_shapes.h` (the skeleton's launch list as data)
- Modify: `tools/probe/CMakeLists.txt`, `src/kernels/CMakeLists.txt` (the block's first lines: the existing kernels at LFM2's shapes - also the engine's binaries - and the stand-ins), `tools/kernel_cmdlines` (regenerated: additions only)

**Interfaces:**
- Produces: `probe_lfm_step [--depth D] [--rounds R] [--list a|fused|gemv|noop] [--sweep-s]` printing one markdown table per run: per launch family the in-situ µs (kernel-timestamp events, docs/15's instrument), Σ durations, the fence wall, the gap per launch; the whole step for each list; the derived roofline (bytes / 590 GB/s).
- The lists (`lfm_step_shapes.h`), from **existing** sources at LFM2's shapes (no LFM2 kernel exists yet):
  - `a` (383): spec 24 §4.5's sequence with stand-ins - the int4 GEMVs as `gemv.cl` at K 2048 N 6144 / 2048 / 3072 / 23552 and K 11776 N 2048 (`gemv_M1_K..._N..._S<s>_L<l>`, the engine's own binaries); the router `gemv_bf16_M1_K2048_N64_C16_S16`; the head `gemv_i8w_M1_K2048_N128000`; the MoE as `moe.cl` at `EXPERTS=64 TOP_K=4 HIDDEN=2048 INTER=1536 ROUTER_N=80` (`moe_M1_E64_T4_D2048_I1536`: its route, gate‖up, down - **5 slots**, the shared one pointed at expert 0, so its gate‖up / down read 25 % more than LFM2's; the table prints the raw time and x 4/5 PROPOSED as the 4-slot estimate); the attention as Kolibri's `kol_attn` full path at `Q_HEADS=32 KV_HEADS=8` (head_dim **128**: twice LFM2's KV bytes; printed raw and x 1/2 PROPOSED) with `kol_attn_prep` at the same heads; the norm as Ornith's `prep_res_fold_M1_K2048_SP0_G20` + a `kol_norm_finish` line at K 2048; `noop` (`src/kernels/noop.cl`) in the conv, silu and embed slots; `argmax.cl` at `VOCAB=128000 VOCAB_USED=125017`.
  - `fused` (277): `a` without one `noop` per conv layer (arm `conv`), one route launch per MoE layer (arm `route`) and one router launch per MoE layer (arm `router`).
  - `gemv`: only the GEMV and MoE gate‖up / down launches of `a` (the bytes floor of the list).
  - `noop`: 383 `noop`s (`probe_replay`'s floor at this count).
- [x] **Step 1: write the shapes as data and the tool;** the weights random (incompressible, `probe_gemv`'s rule), one buffer set; each list captured once, replayed `R` rounds interleaved (`a`, `fused`, `gemv`, `noop` rotated per round) after a 20-replay warm-up; `--sweep-s` times every int4 GEMV shape at S in {1, 2, 4, 8, 16} (where S divides K / 64) and layouts 0 / 1, `probe_gemv`'s 3 % rule picks S. Host test `probe_lfm_step_shapes_test`: the list lengths 383 / 277 and the per-family counts (2 x 8 + 28 x 9 + 10 x 11 + 5).
- [x] **Step 2: CMake lines** for every binary the lists bind (recorded: `+N / -0 / ~0`); Mac gate `tools/mac_check.sh --base main --kernels` (the OpenCL syntax of the new lines).
- [x] **Step 3: commit** `git commit -S -m "probe: probe_lfm_step - LFM2.5-24B-A2B's 383- and 277-launch skeleton lists from existing kernels at its shapes, the GEMV-only floor, the S sweep (spec 24c P0)"`.
- [ ] **Step 4 (box, row 51's first GPU stage `r51.p0`):** `probe_lfm_step --depth 4096 --rounds 9` and `--depth 32768`, `--sweep-s`; the table into `docs/probe-lfm2moe-<date>.md` (24a's sheet, a "P0" section): step ms for `a` / `fused` / `gemv` / `noop`, the gap per launch, the small-kernel floor, the S picks. **The decision rule** (decision 6, PROPOSED): every arm whose skeleton saving is ≥ 1 % of the step is built and A/B'd in Task 8; the S picks become `kernels::lfm2moe`'s table (`PROVISIONAL` until then). Commit `git commit -S -m "spec 24c P0: the LFM2.5-24B-A2B step on the card - launches, gaps, GEMV floors, the S picks"`.

### Task 2: the fixture and the host reference (no card)

**Files:**
- Create: `tools/oracle/lfm2moe_fixture.py`, `tests/kernels/lfm2moe_fixture.h` (generated, committed), `tests/kernels/lfm2moe_ref.h`, `tests/kernels/lfm2moe_ref_test.cc`
- Modify: `tests/CMakeLists.txt` (a block `# ==== Spec 24c: LFM2.5-24B-A2B decode (label lfm2moe) ==== (begin)` after 24b's)

**Interfaces:**
- Consumes: `tools/oracle/lfm2moe_ref.py` (`rms_norm`, `route`, `combine`, `short_conv`, the attention chain, RoPE).
- Produces: `lfm2moe_fixture.h` - seeded inputs and the reference's outputs at the real widths, as `uint16_t` / `float` / `uint32_t` arrays: (a) `res_norm` on 8 rows incl. one with a huge and one with a tiny Σ; (b) the conv at positions 0, 1, 2, 15, 16, 17 with M = 1 and an M = 4 launch, the ring before and after; (c) the route on 64 rows of a real-width layer with the trained-like bias distribution, plus rows with a tie at the 4th / 5th, all-equal logits, a bias that reorders; (d) one MoE layer's 4-slot output for 4 rows (int4 experts, dequantised by 24b's rule); (e) the attention prep at positions 0, 1, 4095, 131071 (q / k norm, RoPE) and eager attention at 6 / 300 / 3000 keys; (f) `silu_mul` at I 11776; (g) the argmax with planted maxima at 125016 and 127999.
- Produces (`tests/kernels/lfm2moe_ref.h`, namespace `lfmref`, used by Tasks 3-5 and 24d / 24e): `res_norm(h_in, slices, S, w, eps) -> {h_out, x}`, `conv_step(slices, S, ring, taps, pos, M) -> {y, ring'}`, `router(x, w_bf16) -> logits_f32` (THE dot order the kernels use: per row, 2048 products in 8 lanes of 256 then a fixed pairwise tree - one function, so the `router` arm can match it), `route(logits, bias) -> {ids (rank), w, sel5, sum}`, `moe(x, route, blocks) -> y` (24a's order), `attn_prep`, `attn_eager`, `attn_fp64`, `silu_mul`, `argmax(logits, vocab_used)`.
- [x] **Step 1: the failing test** `lfm2moe_ref_test.cc`: every fixture case bit for bit against `lfm2moe_ref.h` (where the reference's own fp32 order differs from the engine's chosen order - the router GEMV, the conv's accumulation per 24a's count - the fixture carries the reference's value and the test asserts equality on the rows 24a found equal and prints the others); Review Focus 1-3 on the host.
- [x] **Step 2: register, FAIL; write `lfm2moe_ref.h` until PASS** (`ctest --preset mac-host -R '^lfm2moe_ref_test$'`).
- [x] **Step 3: generate the fixture** (`agnes-ref-img` + 24a's site: `python3 tools/oracle/lfm2moe_fixture.py > tests/kernels/lfm2moe_fixture.h`; the header records the command and the torch / transformers versions). **Commit** `git commit -S -m "tests: LFM2.5-24B-A2B decode host reference and fixture - norm, conv ring, sigmoid-bf16 route, MoE, hd64 attention, argmax edge (spec 24c)"`.

### Task 3: the kernels

**Files:**
- Create: `src/kernels/lfm2moe/lfm_prep.cl`, `src/kernels/lfm2moe/lfm_conv.cl`, `src/kernels/lfm2moe/lfm_moe.cl`, `src/kernels/lfm2moe/lfm_expert.h` (the address macro), `src/kernels/lfm2moe/lfm_attn.cl`, `src/kernels/lfm2moe/lfm_attn_eager.cl`, `src/kernels/lfm2moe_kernels.h`, `tests/kernels/lfm2moe_variant_names_test.cc`, `tests/kernels/lfm2moe_kernels_test.cc`, `tools/mac/clrun/lfm2moe_run.cc`
- Modify: `src/kernels/CMakeLists.txt` (the block), `tools/mac_check.sh` (section 5's driver loop gains `lfm2moe`), `tests/CMakeLists.txt`
- Test: `lfm2moe_variant_names_test` (host), `lfm2moe_kernels_test` (card), `lfm2moe_run` (Mac GPU, indicative)

**Interfaces:**
- Produces (`src/kernels/lfm2moe_kernels.h`):

```cpp
namespace kernels::lfm2moe {
inline constexpr unsigned kHidden = 2048, kQHeads = 32, kKvHeads = 8, kHd = 64, kQkvN = 3072, kInProjN = 6144;
inline constexpr unsigned kExperts = 64, kTopK = 4, kInter = 1536, kDenseInter = 11776, kVocab = 128000, kVocabUsed = 125017;
inline constexpr unsigned kConvRing = 16;                          // = loader::lfm::kConvRing
inline constexpr unsigned kNormWgs = 8, kNormWg = 256;             // res_norm: 8 work-groups of 256 (PROVISIONAL, Task 8)
inline constexpr unsigned kUpKs = 4, kDnKs = 2, kAttnTgt = 32;     // PROVISIONAL (Task 8 sweeps)
namespace route { inline constexpr unsigned kWords = 16, kIds = 0 /* 4 ids, rank order */, kW = 4 /* f32 of bf16 */,
                  kSel = 8 /* 5 sel, rank 0..4 */, kSum = 13 /* f32 of bf16 Σ + eps */; }
struct Gemv { unsigned K, N, S, L; };                              // the int4 shapes and their P0 picks (PROVISIONAL)
Gemv in_proj(), out_proj(), qkv(), o_proj(), dense_gate_up(), dense_down();
std::string res_norm_variant(unsigned M, unsigned S_fold, bool router_fused);   // "lfm_res_norm_M1_K2048_S4" | "_NF" | "..._R"
std::string conv_variant(unsigned M, unsigned S);                  // "lfm_conv_M1_S1"
std::string conv_in_variant(unsigned M);                           // "lfm_conv_in_M1_K2048" (arm conv: GEMV + conv epilogue)
std::string router_variant(unsigned M);                            // "lfm_router_M1_K2048_E64"
std::string route_variant(unsigned M);                             // "lfm_route_M1_E64_T4"
std::string moe_variant(unsigned M, bool route_fused);             // "lfm_moe_M1_E64_T4_D2048_I1536" | "..._R"
std::string attn_prep_variant(unsigned M, unsigned S);             // "lfm_attn_prep_M1_S1_Q32KV8"
std::string attn_variant(unsigned M, bool eager);                  // "lfm_attn_M1_T32_Q32KV8" | "lfm_attn_eager_M1_Q32KV8"
std::string silu_mul_variant(unsigned M, unsigned S);              // "lfm_silu_mul_M1_I11776_S4"
std::string embed_variant(unsigned M);                             // "lfm_embed_gather_M1_K2048_V128000"
std::string argmax1_variant(unsigned M);                           // "lfm_argmax_stage1_M1_V128000_U125017" (argmax.cl)
std::string argmax2_variant();                                     // "lfm_argmax_stage2_V128000"
std::vector<std::string> decode_variants(const model::Lfm2MoeDesc& d, bool int8_head, bool eager, bool two_cards);
}
```

The kernels (each chain is the reference's; `lfm2moe_ref.h` repeats it):

```text
lfm_prep.cl  (plain OpenCL C: runs on the Mac)
  lfm_res_norm(h_in, slices, w, h_out, x)       grid (kNormWgs, M), WG 256
      every WG: r[k] = S ? rne(f32(h_in[k]) + f32(rne(Σ_{s<S} slices[s][m][k]))) : h_in[k]   for all 2048 k
                Σ = tree(r²) (fixed, identical in every WG);  rstd = 1 / sqrt(Σ / 2048 + 1e-5)
      own chunk: h_out[k] = r[k]  (S > 0 only);  x[k] = rne(f32(rne(f32(r[k]) · rstd)) · w[k])
      _R: + 64 router rows from x (lfmref::router's order) -> logits_f32 (arm router; WGs kNormWgs..kNormWgs+3)
  lfm_silu_mul(slices, x)                       prep_silu_mul's chain at I 11776 (cols_interleave16 pairs)
  lfm_attn_prep(ctrl, slices, qkn, rope, attn_q, kv_k, kv_v)   kol_attn_prep at HD 64, eps 1e-5, RoPE always (rotate-half,
      the table [max_len][2][32]); grid (32 + 2 x 8, M), WG 64
  lfm_embed_gather(ctrl, tiles, x)              x[m][k] = tiles[lfm_embed_row_index(2048, k, cur_token[m])]
lfm_conv.cl  (plain OpenCL C)
  lfm_conv(ctrl, slices, taps, ring, y)         grid (2048 / 256, 1), WG 256; one work-item a channel, rows m = 0..M-1 in order:
      B, C, x = rne(Σ_s ...) at columns c, 2048 + c, 4096 + c;  bx = rne(f32(B) · f32(x));  ring[(pos+m) % 16][c] = bx
      acc = fma(w2, bx, fma(w1, f1, fma(w0, f2, 0)))  with f1 / f2 the channel's Bx at p-1 / p-2 (registers within the launch,
      the ring before it; 0 below position 0);  y[m][c] = rne(f32(C) · f32(rne(acc)))
  lfm_conv_in (arm conv; Task 5)
lfm_moe.cl  (moe.cl's structure; sub-group block reads under cl_intel_subgroups, plain loads otherwise)
  lfm_router(x, w_tiles, logits)                64 rows, lfmref::router's order
  lfm_route(logits, bias, route)                grid (1, M), WG 64, lane e owns expert e: spec 24 §2.3's chain; rank by count with
      ties to the lower id; lane 0: Σ of the 4 s_b in rank order (fp32), d = rne(rne(Σ) + 1e-6f), w_j = rne(s_j / d)
  lfm_moe_gate_up(route, x, w_gu, h)            4 slots x (2 x 1536 / 16 n-tiles); expert id from the route row, its block at
      LFM_EXPERT_GU(base, id) (lfm_expert.h); h[m][slot][i] = rne(f32(rne(silu(f32(rne(Σ gate))))) · f32(rne(Σ up)))
  lfm_moe_down(route, h, w_dn, h_res)           #pragma OPENCL FP_CONTRACT OFF; d_k = rne(Σ down_k); t_k = rne(f32(d_k) · w_k);
      y = rne(t_0 + t_1 + t_2 + t_3) in fp32, rank order (or 24a's other order); h_res[m][n] = rne(f32(h_res) + f32(y))
lfm_attn.cl     kol_attn.cl's full path at HD 64 (NSG 4, PPS 4, PER_LANE 4, WG 64), scale 0.125, + lfm_attn_reduce (merge ascending)
lfm_attn_eager.cl  k2_attn_eager's four launches at HD 64 / GQA 4 (exp_torch, the 8-lane softmax sum), B70_LFM_ATTN=eager
lfm_expert.h    #define LFM_EXPERT_GU(base, id) ((base) + (size_t)(id) * GU_BLK_U32)   the only expert address (and _DN)
```

Reused at this family's shapes (Task 1's lines): `gemv_M1_K2048_N6144_S*_L*`, `gemv_M1_K2048_N2048_S*_L*`, `gemv_M1_K2048_N3072_S*_L*`, `gemv_M1_K2048_N23552_S*_L*`, `gemv_M1_K11776_N2048_S*_L*` (the P0 picks), `gemv_i8w_M1_K2048_N128000`, `gemv_bf16_M1_K2048_N128000` (the bf16 head over the embedding tiles).

- [x] **Step 1: the failing host test** `lfm2moe_variant_names_test`: every name `decode_variants` returns for (int8 / bf16 head) x (flash / eager) x (one / two cards) x (each arm) is a target of the block (`${B70_LFM2MOE_DECODE_KERNELS}` as argv). FAIL.
- [x] **Step 2: write the sources, `lfm_expert.h`, the header, the CMake block;** the names test PASSES; `tools/kernel_cmdlines` additions only.
- [x] **Step 3: Mac GPU driver** `tools/mac/clrun/lfm2moe_run.cc`: `lfm_prep.cl`, `lfm_conv.cl`, `lfm_moe.cl` (plain-load path), `lfm_attn_eager.cl` at the real shapes against `lfm2moe_ref.h` on the fixture and random inputs; prints `lfm2moe_run: <kernel> exact` or the first differing index; added to `tools/mac_check.sh` section 5.
- [x] **Step 4: the card test** `lfm2moe_kernels_test` (label `lfm2moe`, no checkpoint): every binary of the block at the real shapes against `lfm2moe_ref.h` - bitwise for the norm (both variants), the conv (Review Focus 1's positions and M = 1..4), the router and route (Review Focus 2's rows), the MoE (each slot alone, the others' weights zero; the full 4-slot combine), the attention prep, `silu_mul`, the eager attention, the gather, the argmax (Review Focus 5); flash against fp64 at cosine ≥ 0.99999 at depths 1, 2, 300, 4096, 131072.
- [x] **Step 5: Mac gate** `tools/mac_check.sh --base main --kernels`: host PASS, l0 PASS, cmdlines `+N / -0 / ~0`, opencl PASS, `lfm2moe_run` exact on every kernel it runs. **Commit** `git commit -S -m "kernels: LFM2.5-24B-A2B decode - short conv over a Bx ring, the sigmoid-bf16 route with renormalisation, the 64 x 1536 top-4 MoE, hd64 attention prep / flash / eager, the one-launch norm, the tied gather (spec 24c)"`.

### Task 4: `Lfm2MoeEngine` on one card, the CLI, `--profile`, F3 / F4

**Files:**
- Create: `src/runtime/lfm2moe/lfm2moe_buffers.{h,cc}`, `src/runtime/lfm2moe/lfm2moe_capture.{h,cc}`, `src/runtime/lfm2moe/lfm2moe_engine.{h,cc}`, `src/cli/lfm2moe_decode.h`, `tests/runtime/lfm2moe_decode_test.cc`, `tests/golden/lfm2moe_golden_test.cc`
- Modify: `src/runtime/lfm2moe/lfm2moe_sizes.{h,cc}` (24b's: decode scratch, launch counts), `src/runtime/lfm2moe/CMakeLists.txt` (library `b70_lfm2moe_runtime`), `tests/runtime/lfm2moe_plan_test.cc`, `src/cli/b70_decode.cc` (dispatch), `src/cli/b70_serve.cc` (refusal naming 24e), `src/cli/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `lfm2moe_plan_test` (host); `lfm2moe_decode_test`, `lfm2moe_golden_test`, `cli_reject_lfm2moe_*` (card / binary)

**Interfaces:**
- Consumes: Tasks 2-3; 24b's `load_lfm2moe`, `LfmLoaded`, `runtime::lfm2moe::plan` / `max_len_auto` / `require_fits`.
- Produces (24d adds `prefill`, 24e the snapshots):

```cpp
namespace runtime::lfm2moe {
enum class LfmAttn { Flash, Eager };
LfmAttn lfm_attn();                                    // B70_LFM_ATTN=flash|eager; anything else throws
struct LfmFuse { bool route = false, router = false, conv = false; };
LfmFuse lfm_fuse();                                    // B70_LFM_FUSE=none|all|<comma list of route,router,conv>; else throws
struct ScratchSizes { size_t h0, h1, x, slices, conv_y, logits_r, routes, moe_h, attn_q, attn_part, attn_out, s_mlp,
                      logits, argmax_part; size_t total() const; };
ScratchSizes scratch_sizes(const model::Lfm2MoeDesc& d, uint32_t max_len, LfmAttn a);
size_t device_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, LfmAttn a, LfmFuse f,
                       PpHandoff h);
size_t decode_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, LfmAttn a, LfmFuse f, PpHandoff h);
//   383 one card, flash, no arm (2 x 8 + 28 x 9 + 10 x 11 + 5); eager +20; all arms 277; synthetic N 4: 8 + 8 + 11 + 9 + 5 = 41
class Lfm2MoeEngine {
 public:
  Lfm2MoeEngine(std::vector<l0::Context*> devices, loader::LfmLoaded model, uint32_t max_len,
                bool debug_tap = false, const PipelineOptions& opt = {}, bool profile = false);   // profile: ProfileEvents
  void reset();                                        // pos 0; the conv rings need no clearing (positions < 0 read 0)
  void ingest(const std::vector<uint32_t>& ids);       // one replay per id (the prompt, until 24d)
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);  uint32_t pending() const;  uint32_t vocab() const;
  std::vector<uint16_t> read_debug_h();                // bf16 [layers][2048] of the last replay (debug_tap)
  std::vector<uint32_t> read_routes();                 // u32 [moe layers][route::kWords]
  std::vector<uint16_t> read_conv_ring(uint32_t conv_index);   // [16][2048]
  std::vector<uint16_t> read_kv(uint32_t attn_index, uint32_t first, uint32_t count, bool v);
  std::vector<float> read_logits();  void read_logits_into(float* host);
  std::string profile_table();                         // --profile: per-family µs, Σ, fence wall, gap per launch
  MemoryComponents memory_use(uint32_t dev) const;  std::string memory_line() const;
  uint32_t pos() const;  uint32_t max_len() const;  size_t launches() const;  uint32_t devices() const;
  LfmAttn attention() const;  LfmFuse fuse() const;  const loader::LfmLoaded& model() const;
};
}
// src/cli/lfm2moe_decode.h, namespace cli::lfm2moe: is_lfm2moe(path), DecodeArgs, check_args, settle, run_decode -
// cli/kolibri_decode.h's shape
```

- [x] **Step 1: failing host test** (`lfm2moe_plan_test` extended): `decode_launches` 383 / eager 403 / each arm alone 345 / 345 / 353 / all 277 / synthetic N 4 = 41; the decode scratch at max_len 131072 (eager's score row `[M][32][max_len]` fp32 = 16 MiB is the term that scales); the plan at 131072 on one card fits; `ProfileEvents::kProfileCapacity` (1024) ≥ every count.
- [x] **Step 2: buffers, capture, engine (one device).** `capture` walks layers `[first, end)`, asserts `device_launches`, labels every launch (`"L<l> <kernel> <variant>"`, `CapturedStep::labels`' convention), alternates the residual buffers per norm site and asserts the pattern (Review Focus 3), writes each MoE layer's route row, the debug tap after each layer's last launch. `generate` = `KolibriEngine::generate`'s contract. `--profile` builds the list with `ProfileEvents` (docs/15's instrument, `src/runtime/capture.h:39-48`).
- [x] **Step 3: CLI.** `b70-decode` dispatches on `model_type` `lfm2_moe` (`cli::lfm2moe::is_lfm2moe`, beside Kolibri's dispatch and before `cli::check_pipeline`) to `cli::lfm2moe::run_decode`: `--ids`, `--n`, `--bench [--depth N] [--tg N]`, `--profile [--steps N] [--repeats R]`, `--lm-head bf16|int8` (default int8; bench rows say which), `--max-len N|auto` (default 16384 for bench comparability; `auto` = 24b's `max_len_auto`), `--pp N` / `--pipeline-split` / `--pipeline-handoff` (until Task 6, `--pp 2` refused naming Task 6). Refused before the device, by name: `--prefill` / `--prefill-length` / `--prefill-chunk` / `--prefill-backend` ("spec 24d"), `--mtp` ("LFM2.5-24B-A2B has no MTP head"), `--kv-cache int8` ("spec 24 decision 11"), `--layers` (the whole model fits: "--layers belongs to Kolibri-1 / Qwen3.8-Flash-Next" - keep the substring `cli_reject_kolibri_layers_qwen` expects). `b70-serve` refuses `lfm2_moe` before the device naming spec 24e. Registered `b70_cli_reject` tests over `tests/model/lfm2moe` (config only): `cli_reject_lfm2moe_prefill`, `_mtp`, `_kv8`, `_layers`, `_serve`.
- [x] **Step 4: the card gates** (label `checkpoint;golden;lfm2moe`, SKIP 77 without data):
  - `lfm2moe_decode_test <ckpt> <ids>` (F4 on decode): plan == allocation; two `reset` + `ingest` + `generate(32)` runs bitwise (logits, routes, conv rings, KV, both residual buffers); the launch count printed.
  - `lfm2moe_golden_test <ckpt> <oracle dir> [int8] [eager]` (F3, kolibri_golden_test's copy): the prompt by M = 1 replays, then `generate(32)`; `golden_common.h`'s tie-aware token rule; per MoE layer the route against `route.ids/w/gap.L*` (a set difference beyond `B70_LFM_TIE_TOL` fails; an exact 0 gap undetermined - torch's order, spec 24 §2.3); the conv state after the prompt against `conv.L*` (bitwise, the input of the ring); `h.L*` per layer (cosine per row, median ≥ 0.9998, min ≥ 0.99, PROPOSED, printed). Registered for the synthetic (`synth`, both spellings) x (`lfm2moe_short`, `lfm2moe_4k`, `lfm2moe_agentic`), the RTN export (`rtn`: the same prompts, `+ _32k` opt-in), `ours` when 24q's export exists; `_i8head` (default) and `_bf16head`, `_eager` twins.
- [x] **Step 5: Mac gate and commit** `git commit -S -m "runtime: Lfm2MoeEngine decodes on one card - 383 launches, the per-launch profile, F3 (tokens, routes, conv state) and F4 replay on the synthetics and the RTN export (spec 24c)"`.

### Task 5: the fused arms (`B70_LFM_FUSE`)

**Files:**
- Modify: `src/kernels/lfm2moe/lfm_prep.cl` (`_R`), `src/kernels/lfm2moe/lfm_moe.cl` (`_R`), `src/kernels/lfm2moe/lfm_conv.cl` (`lfm_conv_in`), `src/kernels/lfm2moe_kernels.h`, `src/kernels/CMakeLists.txt`, `src/runtime/lfm2moe/lfm2moe_capture.cc`, `tests/kernels/lfm2moe_kernels_test.cc`, `tests/runtime/lfm2moe_decode_test.cc`
- Test: `lfm2moe_kernels_test` (each arm's kernel against the as-built pair), `lfm2moe_decode_test --fuse-all` (Review Focus 4)

- [x] **Step 1: the failing tests:** kernel level - `lfm_res_norm_..._R`'s logits equal `lfm_router`'s on the same normalised row bitwise; `lfm_moe_M1_..._R`'s gate‖up output and route row equal `lfm_route` + `lfm_moe_gate_up`; `lfm_conv_in` on the conv-tile `in_proj` (24b's `repack_in_proj(..., true)`) equals `gemv` (S 1) + `lfm_conv` bitwise. Engine level - `lfm2moe_decode_test <ckpt> <ids> --fuse route|router|conv|all`: logits, routes, conv rings and KV after a prompt and 32 greedy tokens bitwise the `none` run (Review Focus 4); the launch counts 345 / 345 / 353 / 277.
- [x] **Step 2: implement** (`lfm_conv_in`: `gemv.cl`'s layout-1 inner loop - the same `tile_dot`, the same K order at S 1 - with 3 n-tiles a work-group (B, C, x of one 16-channel tile) and the conv as the epilogue; the loader uploads `in_proj` in the conv-tile layout when the arm is on). PASS on the Mac GPU where the kernels are plain OpenCL; the card test is the box's.
- [x] **Step 3: commit** `git commit -S -m "kernels: the LFM2.5-24B-A2B fused arms - route in gate||up, router in ffn_norm, conv as the in_proj epilogue; bitwise the as-built list, 277 launches with all three (spec 24c)"`.

### Task 6: two cards (`--pp 2`, on spec 16b's pieces; decision 10)

**Files:**
- Modify: `src/runtime/lfm2moe/lfm2moe_engine.{h,cc}`, `src/runtime/lfm2moe/lfm2moe_capture.{h,cc}`, `src/cli/lfm2moe_decode.h`, `tests/CMakeLists.txt`
- Create: `tests/runtime/lfm2moe_pp_test.cc`

- [x] **Step 1: the failing test** `lfm2moe_pp_test <synth> <rtn> <ids>`: (a) synthetic N 4, `--pp 1` vs `--pp 2 --pipeline-split 2`, `copy` and `peer`: Review Focus 6, both Control blocks equal after every phase; (b) the RTN export split 20 vs 19: bitwise; (c) a lost hand-off throws within the bounds and later steps throw until `reset()`; (d) `pp_landing_layout(4096, 0)` - `h` only (the receiving `res_norm _NF` recomputes Σ).
- [x] **Step 2: register** (`LABELS "checkpoint;lfm2moe;pp"`, SKIP 77 with one GPU or no data); **implement** (device 0 ends with layer s-1's `lfm_moe_down`; device 1 starts with `res_norm _NF`; the head on device 1; the embedding tiles on device 0 for the gather and on device 1 for a bf16 head); `kolibri_pp_test` and 16b's host tests unchanged.
- [x] **Step 3: commit** `git commit -S -m "runtime: LFM2.5-24B-A2B across two cards - h crosses (4 KB), bitwise across splits (spec 24c)"`.

### Task 7: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` ("24c as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 51), `tools/box_validate/stages.sh` (`row 51`), `docs/19-running-models.md` (an "LFM2.5-24B-A2B (spec 24)" section: decode only until 24d / 24e, every command marked "never run on a B70" until the row passes), `README.md` (the env-var table: `B70_LFM_ATTN`, `B70_LFM_FUSE`)

- [x] **Step 1:** "24c as built": the list and its counts per arm, the stand-ins P0 used, the S picks, every departure.
- [x] **Step 2: the queue row 51** (`test_box_validate.py` passes; `--dry-run --only r51` prints it):

```text
row 51 "spec 24c - LFM2.5-24B-A2B decode: the launch-count P0, short conv / hd64 attention / sigmoid-bf16 route / 64 x 1536 MoE, the fused arms, one card then two (plan 24c)"
stage r51.k0       51 default cpu -                           g0.sha,g0.bitwise,g0.suite "K0 (F0): every pre-existing binary identical; kernel_cmdlines +N / -0 / ~0"
stage r51.p0       51 default gpu -                           r51.k0     "P0: probe_lfm_step --depth 4096 / 32768, interleaved a / fused / gemv / noop, --sweep-s (the table decides decision 6's arms and the S picks)"
stage r51.host     51 default cpu -                           -          "host: lfm2moe_ref_test, lfm2moe_variant_names_test, lfm2moe_plan_test, probe_lfm_step_shapes_test"
stage r51.k1       51 default gpu -                           r51.k0     "kbins of the block + lfm2moe_kernels_test (F2 bitwise; flash at cosine 0.99999; each fused arm against its pair)"
stage r51.reject   51 default cpu -                           -          "cli_reject_lfm2moe_* (prefill, mtp, kv8, layers, serve)"
stage r51.k3       51 default gpu lfm2moe_synth               r51.k1     "lfm2moe_decode_test on the synthetics: plan == allocation, 41 launches, two runs bitwise, --fuse all bitwise none"
stage r51.golden   51 default gpu lfm2moe_synth,oracle_lfm2moe_synth r51.k3 "lfm2moe_golden_test on the synthetics: tokens tie-aware, routes, conv state, h taps; _bf16head, _eager"
stage r51.rtn      51 default gpu lfm2moe_rtn,oracle_lfm2moe_rtn r51.k3  "the RTN export: decode_test (383 launches) and golden_test short / 4k / agentic"
stage r51.ours     51 optin   gpu lfm2moe_ours,oracle_lfm2moe_ours r51.k3 "24q's AutoRound export: the same gates"
stage r51.pp       51 default gpu lfm2moe_synth,lfm2moe_rtn   r51.k3     "lfm2moe_pp_test: two cards bitwise one (copy, peer)"
stage r51.speed    51 optin   gpu lfm2moe_rtn                 r51.rtn    "Task 8: decode at depth 1024 / 4096 / 32768 / 131072, each fused arm A/B, --profile; int8 / bf16 head"
rownote 51 "Tolerances PROPOSED until row 48 prints the gap distribution: B70_LFM_TIE_TOL 1e-3 on the 4th / 5th sel; an exact 0 gap is undetermined. The h-tap bars (median >= 0.9998, min >= 0.99) are PROPOSED. P0's stand-ins: moe.cl reads 5 slots (x 4/5), kol_attn is head_dim 128 (x 1/2) - the corrections are PROPOSED approximations, the engine's --profile in r51.speed is the measurement."
```
- [x] **Step 3: commit** `git commit -S -m "docs: spec 24 (24c as built), box queue row 51 - LFM2.5-24B-A2B decode"`.

### Task 8: decode speed and decision 6 (box)

- [ ] On the RTN export (then ours): `b70-decode <lfm2moe> --bench --depth D --tg 256` for D = 1024, 4096, 32768, 131072 (`--max-len auto`), int8 and bf16 heads, flash and eager, `B70_LFM_FUSE=none` vs each arm vs `all` as interleaved pairs, median of 3, `uptime` and the idle grade; `--profile --repeats 5` at 4096 (the in-situ anatomy against P0's skeleton: per family µs, the gap per launch); the MoE kernels' `UP_KS` / `DN_KS`, `res_norm`'s work-group count and the attention's `TGT` swept. Against spec 24 §3's estimate (~4.5 ms, ~220 t/s at 4096; ~319 t/s at 77.6 % of roofline). `docs/BENCHMARKS.md` section "LFM2.5-24B-A2B (spec 24)" decode rows; **decision 6's evidence** (the arms' measured savings) recorded and the default proposed. **Commit** `git commit -S -m "spec 24c: LFM2.5-24B-A2B decode speed, the fused arms measured (decision 6's evidence)"`.

**Gate for the plan:** Mac - every host test, `lfm2moe_run` exact, cmdlines additions only, every syntax check. Box - F0 (G0); P0's table written before any engine stage; F2 on every kernel and arm; F4 replay bitwise, every arm bitwise `none`; F3 on the synthetics and the RTN export (tokens tie-aware, routes, conv state); two cards bitwise one; the speed rows and the arms' A/B recorded.

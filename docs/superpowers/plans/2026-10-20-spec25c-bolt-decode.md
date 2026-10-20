# Spec 25c - Nemotron 3.5 Bolt decode: the launch-count and state-bandwidth P0 first, then Mamba-2, the latent top-9 MoE and NoPE GQA-16 attention on two cards, the fused arms

**Status (2026-10-20): planned; nothing built.** Box queue row **60**.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** measure first where a Bolt decode step's time goes - a 653-launch skeleton from existing binaries at Bolt's
shapes, the 533-launch fused skeleton, a GEMV-only list and the SSM state's read / write traffic, interleaved on the card
(P0) - then decode the model through replayed Level Zero lists on its own engine: the `nb_*` family kernels of spec 25
§6.1, the list on two cards (`--pp 2`, split by bytes) and on one card for synthetic / truncated checkpoints, `b70-decode
<bolt>`, `--profile`, the three fused arms behind `B70_BOLT_FUSE` (bitwise the as-built list), the fp32 / bf16 SSM-state
arm (Decision 2); B2 on every kernel, B3 / B4 on decode; the speed rows at depth.

**Architecture:** spec 25 §2, §3, §4, §6.1, §8. `runtime::bolt::BoltEngine` (`src/runtime/bolt/`) owns 25b's
`loader::BoltLoaded`, one buffer set and one captured list per device. The residual `h` is bf16 [M][4096] in two buffers
(ping-pong across norm sites, 24c's rule); every block leaves its output's split-K slices for the next norm to fold:

```text
Mamba (40), 6:      nb_res_norm _S · in_proj GEMV (K 4096, N 18560) · nb_conv · nb_ssm · nb_gnorm · out_proj GEMV (K 8192)
attention (8), 6:   nb_res_norm _S · q‖k‖v GEMV (N 4608) · nb_attn_prep · attn · attn_reduce · o GEMV
MoE (40), 9:        nb_res_norm _S · nb_router · nb_route · fc1‖sh_up GEMV (N 6400) · nb_sh_act · nb_moe_up · nb_moe_down ·
                    fc2 GEMV (K 1024) · sh_down GEMV (K 5376)           (the next norm folds o and sh: _S2)
head, 5:            nb_embed_gather (device 0, first) · nb_res_norm _S + norm_f · head GEMV (int8 | bf16) · argmax x 2
```
**653 launches a token** (derived: 40 x 6 + 8 x 6 + 40 x 9 + 5; the plan test asserts the formula), eager +2 per attention
layer (669), the arms `router` -40, `route` -40, `conv` -40 (**533** with all three), two cards +0 (`copy`) / +2 (`peer`).

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake / ctest; Python 3 + torch in `agnes-ref-img` with 25a's site
(the fixture generator).

**Spec:** `docs/superpowers/specs/2026-10-20-spec25-nemotron-bolt-design.md` (§2, §3, §4, §6.1, §8, §9 B0, B2-B4 and speed,
§12 decisions 1, 2, 4, 10, 11, 15, 16, 18, 19). Facts: 25a's sheet (the rounding chain, the tie order, the gap
distributions, the dt-clamp census). Precedents: plan 24c task for task (its as-built §14: the norm-site rule, the arms'
binaries, the P0 stand-ins and their scale factors, `ProfileEvents` at 1024), plan 23c (Muse's `mg_attn` at GQA 16 / head_dim
128 - Decision 15's reuse), `src/kernels/kolibri/kol_moe.cl` (`kol_route`: 256 lanes x 2 experts, the fp32 sigmoid router,
the ascending-id combine), `src/kernels/lfm2moe/lfm_{prep,moe,conv}.cl` (the one-launch norm site, the `_R` arms),
`src/kernels/gdn_step.cl` (the ring discipline; the slot build 19c added), docs/15-step-anatomy.md (the instrument).

## Dependencies and branch points

- **Plans 25a and 25b merged.** 25a: `bolt_ref.py` / `bolt_mamba_ops.py` (the fixture), the golden layout, the gap
  distributions (`B70_BOLT_TIE_TOL`, Decision 11). 25b: `BoltDesc` (`linears(l)`'s PROVISIONAL `{S, layout}` - swept here,
  the descriptor is the only place a re-pick changes), `load_bolt`, `bolt_layout.h`, `runtime::bolt::plan` (this plan adds
  the decode scratch and the launch counts), the synthetic checkpoints. Real-weight gates need SergiioB's checkpoint and
  25a's `oracle-out-bolt` golden sets; `rtn` / `ours` stages need 25q's exports.
- **Spec 16b's pieces** are merged (`PipelineLink`, `pp_landing_layout`, `require_two_devices`, `cli::PipelineArgs`); the real
  model always runs on two cards; one card serves the synthetics and `--layers N` (N <= 24 fits one card at short
  `--max-len`, derived in Task 4).
- **Decision 15:** the attention binaries are Muse's (`mg_attn` / `mg_attn_eager`, 23c) bound by name from new CMake lines at
  their existing defines if the KV layout matches (Task 3 Step 1 checks), with a ones gate buffer; else `nb_attn` copies.
- **Decision 2:** the SSM state is fp32 by default; a `B70_BOLT_SSM=bf16` build is built and measured (Task 8), the default
  changes only on the operator's ruling with Task 8's numbers.
- **Not here:** prefill (25d: `--prefill*` refused naming it), serving (25e), the verify lists at M = 2..8 and the state
  slots (spec 26c - but every kernel takes `M` as a define with `M <= 8` and the twins test M = 1..4 in one launch).

## Global Constraints

- Branch `spec25c-bolt-decode` from main; box tree automatic; `tools/box.env` never committed or printed.
- **B0:** every existing kernel binary keeps its name, command line and bytes (`tools/kernel_cmdlines` additions only; G0's
  sha256 on the box); no existing `.cl` edited (Muse's binaries bound by name - if a needed one is defined only inside the
  Muse CMake block, it is built here when that block is off, 24c's rule).
- **Semantics are 25a's, pinned:** every rounding point of spec 25 §2 in the kernels and in `tests/kernels/bolt_ref.h`,
  edited together; exact ties to the lower expert id; the combine in ascending id; no dt clamp.
- Family kernels in `src/kernels/bolt/`, named `nb_*`, every shape define in the binary name; host half
  `src/kernels/bolt_kernels.h` (`kernels::bolt`); a CMake block `# ==== Spec 25c: Nemotron 3.5 Bolt decode ==== (begin) /
  (end)`, option `B70_BOLT` ON.
- **Determinism:** no atomic in any sum that feeds a result; every reduction a fixed tree or a fixed sequential order; the
  fused arms bitwise the as-built list; every M-row launch runs exactly M real rows (no padding row reaches an expert).
- Alignment: KV rows (256 B), ring rows (20,480 B), SSM state slices, expert blocks - 64 B aligned bases, asserted at capture.
- Every number measured, or marked derived / estimated / proposed. Box: every GPU command under `flock ~/b70-gpu.lock` (once
  for both cards), detached, polled; timed arms as interleaved pairs after a warm-up, median of 3; `-j44`.
- Mac checks: `tools/mac_check.sh --base main --kernels`. No `rm -rf`. Signed commits; no merge, no push.

## Review Focus

1. **The Mamba chain is reading 1 (spec 25 §2.2) and nothing else.** z ‖ xBC ‖ dt split at 8192 / 18432; the ring holds RAW
   xBC at `pos % 16`; taps oldest first, bias, SiLU, one rounding; `softplus` with threshold 20 and no clamp; heads to groups
   `h / 16`; S in fp32 (or the bf16 arm's explicit rounding after the update); y rounded once; the gated norm `v = y · silu(z)`
   normed over groups of 1024. Task 2's twin and Task 3's kernel test cover positions 0, 1, 2, 3, the ring wrap at 15 / 16 /
   17, M = 1..4 rows in one launch, and a head with `dt_bias` -10.5 (the clamp census's case).
2. **The router's fp32 chain and top-9.** fp32 logits (never rounded), `s = sigmoid`, `sel = s + bias`, rank by count over
   SLM with exact ties to the lower id, the selected set `rank < 9`, `w = s / (Σ + 1e-20) x 5.0` with Σ in the order 25a
   pinned; the route row of 48 words; E 240..496 on 256 lanes x 2; a lane whose expert index >= E never ranks. Tests: a bias
   that reorders, a tie at the 9th / 10th, all-equal logits, E = 496 (the second lane slot live) and E = 240.
3. **The latent MoE and the shared expert.** `fc1` and `shared_up` share one GEMV launch (columns 0..1023 latent, 1024..6399
   shared up); `nb_sh_act` rounds the latent to bf16 and applies relu² to the shared half; the experts work in the 1024 latent;
   `nb_moe_down`'s combine is ascending expert id, product rounded to fp32 then added, ONE rounding; the next norm folds
   `rne(rne(Σ fc2) + rne(Σ sh_down))` then `rne(h + ·)` - three roundings in that order.
4. **The norm sites and the ping-pong** (24c's rule): one launch a site, every work-group computes the identical Σ over the
   folded row, writes only its chunk; folds read `h[cur]` and write `h[1 - cur]`; the capture asserts the alternation; the
   two-source fold at every block after a MoE block.
5. **The arms are the same arithmetic:** `router` (the E rows inside the MoE norm launch from the identical normalised row),
   `route` (the rank recomputed in every `nb_moe_up` work-group), `conv` (each `nb_ssm` work-group computes the conv for its
   head's 64 x-channels and its group's 256 B / C channels): logits, routes, SSM states, conv rings and KV bitwise the as-built
   list's after a prompt and 32 greedy tokens.
6. **The two-card cut is invisible:** `--pp 2` at split 48 and at another split bitwise each other (and bitwise one card on a
   truncated model that fits one card), under `copy` and `peer`.

---

### Task 1: P0 - where the step's time goes, before any Bolt kernel (box first)

**Files:**
- Create: `tools/probe/probe_bolt_step.cc`, `tools/probe/bolt_step_shapes.h` (the skeleton's launch list as data)
- Modify: `tools/probe/CMakeLists.txt`, `src/kernels/CMakeLists.txt` (the block's first lines: existing kernels at Bolt's
  shapes - also the engine's binaries - and the stand-ins), `tools/kernel_cmdlines` (additions only)

**Interfaces:**
- Produces: `probe_bolt_step [--depth D] [--rounds R] [--list a|fused|gemv|noop|state] [--sweep-s]` printing per launch family
  the in-situ µs (kernel-timestamp events), Σ, the fence wall, the gap per launch, the whole step per list, the derived
  roofline (bytes / 590 GB/s).
- The lists (from **existing** sources at Bolt's shapes, **one card**: every dense linear its own random buffer (5.3 GB), the
  experts as a pool of 64 distinct blocks per MoE layer (40 x 64 x ~2.5 MB = 6.3 GB) with 9 random distinct ids a step, so no
  block is re-read inside a step):
  - `a` (653): the GEMVs at their real shapes (`gemv.cl` at K 4096 N 18560 / 4608 / 6400, K 8192 N 4096, K 1024 N 4096, K 5376
    N 4096, each `S` x layout), the routers `gemv_bf16` at K 4096 N E, the head `gemv_i8w` at K 4096 N 131072, `argmax.cl` at
    131072; **attention: Muse's `mg_attn` at its defines - Bolt's exact shape**; the experts: `moe.cl` at `EXPERTS 256 TOP_K 8
    HIDDEN 1024 INTER 2048` (gated: its gate||up reads 2x the up bytes and it has a 9th shared slot - printed raw and scaled
    PROPOSED x 9/8 slots x 3/4 bytes); the Mamba SSM step as `gdn_step` at Qwen3.8's shape (3.1 MB of fp32 state read and
    written a layer against Bolt's 4 MiB - printed raw and scaled x 4/3.1, PROPOSED); `noop` in the conv, gnorm, route, act,
    attn-prep and embed slots; the norm as Ornith's `prep_res_fold` + `kol_norm_finish` at K 4096 as ONE launch a slot (keeps
    653; `--pair` times the two-launch form).
  - `fused` (533): `a` without one router GEMV, one route and one conv slot a layer.
  - `gemv`: only the GEMV and expert launches (the bytes floor).
  - `noop`: 653 noops.
  - `state`: 40 launches of a read-modify-write over 4 MiB fp32 each (`bw_sum.cl`'s pattern) - the SSM state's floor on its own,
    and at 2 MiB (the bf16 arm).
- [ ] **Step 1: write the shapes as data and the tool;** each list captured once, replayed `R` rounds interleaved (`a`,
  `fused`, `gemv`, `noop`, `state` rotated) after a 20-replay warm-up; `--sweep-s` times every int4 GEMV shape at S in {1, 2,
  4, 8, 16} (where S divides K / 64) and layouts 0 / 1, `probe_gemv`'s 3 % rule picks S. Host test
  `probe_bolt_step_shapes_test`: the list lengths 653 / 533 and the per-family counts.
- [ ] **Step 2: CMake lines** for every binary the lists bind (`+N / -0 / ~0` recorded); Mac gate `--kernels`.
- [ ] **Step 3: commit** `git commit -S -m "probe: probe_bolt_step - Bolt's 653- and 533-launch skeleton lists from existing kernels, the GEMV floor, the SSM-state floor, the S sweep (spec 25c P0)"`.
- [ ] **Step 4 (box, `r60.p0`):** `--depth 4096 --rounds 9` and `--depth 65536`, `--sweep-s`; the table into 25a's facts
  sheet (a "P0" section). **The decision rule** (Decision 16, PROPOSED): every arm whose skeleton saving is >= 1 % of the step is
  built and A/B'd in Task 8; the S picks become `BoltDesc::linears`' values (PROVISIONAL until then). Commit
  `git commit -S -m "spec 25c P0: the Bolt step on the card - launches, gaps, GEMV floors, the SSM-state floor, the S picks"`.

### Task 2: the fixture and the host reference (no card)

**Files:**
- Create: `tools/oracle/bolt_fixture.py`, `tests/kernels/bolt_fixture.h` (generated, committed), `tests/kernels/bolt_ref.h`,
  `tests/kernels/bolt_ref_test.cc`
- Modify: `tests/CMakeLists.txt` (`# ==== Spec 25c: Nemotron 3.5 Bolt decode (label bolt) ==== (begin)`)

**Interfaces:**
- Consumes: `bolt_ref.py`'s restated ops and `bolt_mamba_ops.py`.
- Produces: `bolt_fixture.h` - seeded inputs and the reference's outputs at the real widths: (a) `res_norm` on 8 rows (a huge
  and a tiny Σ; the one-source and two-source folds); (b) the conv at positions 0, 1, 2, 3, 15, 16, 17, M = 1 and an M = 4
  launch, the ring before / after; (c) the SSM update for 3 steps from a random fp32 state on all 128 heads (incl. `dt_bias`
  -10.5 and +8.5, `A_log` -4.66 / +4.41 - the checkpoint's extremes), in fp32 and the bf16-state arm; (d) the gated norm on 4
  rows (one group with a dominant element); (e) the route on 64 rows of an E = 496 and an E = 240 layer with a bias around 4.5
  (incl. a 9th / 10th tie, all-equal logits, a reordering bias); (f) one MoE block's output for 4 rows (int4 experts and
  latent / shared linears, dequantised by 25b's rule - incl. negative scales); (g) the KV write and eager attention at 6 / 300
  / 3000 / 262143 keys (NoPE); (h) the argmax with planted maxima at 0 and 131071.
- Produces (`tests/kernels/bolt_ref.h`, namespace `boltref`, used by Tasks 3-6, 25d, 25e, 26): `res_norm`, `conv_step`,
  `ssm_step` (fp32 / bf16 arms), `gnorm`, `router` (THE dot order the kernels use: 8 lanes of 512 then a fixed pairwise tree -
  one function, so the `router` arm matches it), `route`, `moe` (ascending-id combine), `sh_act`, `attn_eager`, `attn_fp64`,
  `argmax`.
- [ ] **Step 1: the failing test** `bolt_ref_test.cc`: every fixture case bit for bit against `bolt_ref.h` (where the
  reference's own fp32 order differs from the engine's chosen order - the router GEMV, the SSM's Σ_n, the gated norm's mean -
  the fixture carries the reference's value, the test asserts equality on the rows 25a found equal and prints the others with
  their ulp distance); Review Focus 1-3 on the host.
- [ ] **Step 2: register, FAIL; write `bolt_ref.h` until PASS** (`ctest --preset mac-host -R '^bolt_ref_test$'`).
- [ ] **Step 3: generate the fixture** (`agnes-ref-img` + 25a's site; the header records the command and versions). **Commit**
  `git commit -S -m "tests: Bolt decode host reference and fixture - norm folds, conv ring, SSM update (fp32 / bf16), gated norm, fp32 top-9 route, latent MoE, NoPE attention, argmax edge (spec 25c)"`.

### Task 3: the kernels

**Files:**
- Create: `src/kernels/bolt/nb_prep.cl` (`nb_res_norm` + `_R` (router arm) + `_S2` (two-source fold), `nb_embed_gather`,
  `nb_attn_prep`, `nb_sh_act`), `src/kernels/bolt/nb_mamba.cl` (`nb_conv`, `nb_ssm` (+ `_C`: the conv arm), `nb_gnorm`),
  `src/kernels/bolt/nb_moe.cl` (`nb_router`, `nb_route`, `nb_moe_up` (+ `_R`), `nb_moe_down`), `src/kernels/bolt/nb_expert.h`
  (the one expert-address macro), `src/kernels/bolt_kernels.h`, `tests/kernels/bolt_variant_names_test.cc`,
  `tests/kernels/bolt_kernels_test.cc` (card), `tools/mac/clrun/bolt_run.cc`
- Modify: `src/kernels/CMakeLists.txt`, `tools/mac_check.sh` (section 5's driver loop gains `bolt`), `tests/CMakeLists.txt`

**Interfaces** (`src/kernels/bolt_kernels.h`):

```cpp
namespace kernels::bolt {
inline constexpr unsigned kHidden = 4096, kMHeads = 128, kMHeadDim = 64, kMGroups = 8, kMState = 128, kMConv = 4;
inline constexpr unsigned kInProjN = 18560, kXbc = 10240, kInner = 8192, kQHeads = 32, kKvHeads = 2, kHd = 128;
inline constexpr unsigned kLatent = 1024, kShared = 5376, kTopK = 9, kVocab = 131072, kConvRing = 16, kGnormGroup = 1024;
namespace route { inline constexpr unsigned kWords = 48, kIds = 0 /* 9, ascending */, kW = 9 /* fp32 */, kSel = 18
                  /* 10: ranks 0..9 */, kSum = 28, kSelMin = 29; }
std::string res_norm_variant(unsigned M, unsigned S_fold, unsigned sources, bool router_fused, unsigned E);
std::string conv_variant(unsigned M);                              // "nb_conv_M1_C10240"
std::string ssm_variant(unsigned M, bool conv_fused, bool bf16_state);   // "nb_ssm_M1_H128_P64_N128_G8" (+ _C, _B16)
std::string gnorm_variant(unsigned M);                             // "nb_gnorm_M1_D8192_G1024"
std::string router_variant(unsigned M, unsigned E);                // "nb_router_M1_K4096_E496"
std::string route_variant(unsigned M, unsigned E);                 // "nb_route_M1_E496_T9"
std::string moe_up_variant(unsigned M, unsigned I, bool route_fused, unsigned E);  // "nb_moe_up_M1_I2560_L1024_T9"
std::string moe_down_variant(unsigned M, unsigned I);              // "nb_moe_down_M1_I2560_L1024_T9"
std::string sh_act_variant(unsigned M);                            // "nb_sh_act_M1_L1024_SH5376"
std::string attn_prep_variant(unsigned M);                         // "nb_attn_prep_M1_KV2"
std::string attn_variant(unsigned M, bool eager);                  // Muse's binary names (Decision 15) or nb_attn copies
std::string embed_variant(unsigned M);                             // "nb_embed_gather_M1_K4096_V131072"
std::vector<std::string> decode_variants(const model::BoltDesc&, bool int8_head, bool eager, bool two_cards, bool bf16_state);
}
```

The kernels (each chain is spec 25 §2's; `bolt_ref.h` repeats it):

```text
nb_prep.cl (plain OpenCL C: runs on the Mac)
  nb_res_norm(h_in, slices[, slices2], w, h_out, x)   grid (kNormWgs, M), WG 256: fold (one or two sources, §2.4's
      three roundings for _S2), Σ r² by one fixed tree in every WG, rstd = 1 / sqrt(Σ / 4096 + 1e-5), x = rne(r · rstd · w)
      (ONE rounding, fp32 product), h_out = r (own chunk); _R: + the E router rows from x (boltref::router's order)
  nb_sh_act(slices, lat, sh)            lat = rne(Σ_s slices[..1024)); sh = rne(rne(relu(rne(Σ_s slices[1024..6400))))²)
  nb_attn_prep(ctrl, slices, kv_k, kv_v, q)   rne of q / k / v, the KV write at pos (no RoPE, no norm)
  nb_embed_gather(ctrl, tiles, x)
nb_mamba.cl
  nb_conv(ctrl, slices, taps, bias, ring, xc)    grid (10240 / 256, 1): per channel, rows m = 0..M-1 in order: xbc_raw =
      rne(Σ_s slices[m][8192 + c]); ring[(pos+m) % 16][c] = xbc_raw; acc = bias + fma-chain of the 4 taps (older taps from the
      ring before the launch / registers within it; 0 below position 0); xc[m][c] = rne(silu(acc))
  nb_ssm(ctrl, slices, xc, A, D, dt_bias, state, y)   grid (128 heads x 4 slices of 16 dims, 1), WG 256: lane (p, n-block)
      holds S[h][p][n..] in registers; per row m: dt = softplus(f32(rne(Σ_s slices[m][18432 + h])) + dt_bias[h]); dA =
      exp(dt · A[h]); S = S · dA + (dt · x[m][h][p]) · B[m][g][n]; y[m][h][p] = rne(Σ_n S · C[m][g][n] + D[h] · x[m][h][p])
      (Σ_n: a fixed sub-group tree); S written back once at the end (the slot build of spec 26c writes after every row)
  nb_gnorm(slices, y, w, out)           v = f32(y) · silu(f32(rne(Σ_s slices z))); per 1024-group: rstd by one fixed tree
      (every WG of a group recomputes it); out = rne(v · rstd · w)
nb_moe.cl  (kol_moe.cl's structure: sub-group block reads under cl_intel_subgroups, plain loads otherwise)
  nb_router(x, w_tiles, logits)         E rows, fp32 out, boltref::router's order
  nb_route(logits, bias, route)         grid (1, M), WG 256, lane l owns experts l and l + 256 (index >= E never ranks): §2.4's
      chain, rank by count with ties to the lower id, the route row (ids ascending, w, sel of ranks 0..9, Σ, min sel)
  nb_moe_up(route, lat, w_up, a)        9 slots x (I / 16 n-tiles): u = rne(Σ up); a = rne(rne(relu(u))²); expert block at
      NB_EXPERT_UP(base, id, I) (nb_expert.h)
  nb_moe_down(route, a, w_dn, r)        #pragma OPENCL FP_CONTRACT OFF; d_k = rne(Σ down_k); acc = 0; for k ascending id:
      acc = acc + (f32(d_k) · w_k) (the product rounded to fp32 first); r = rne(acc)
```
Reused at Bolt's shapes (Task 1's lines): `gemv_M1_K4096_N18560_*`, `gemv_M1_K8192_N4096_*`, `gemv_M1_K4096_N4608_*`,
`gemv_M1_K4096_N6400_*`, `gemv_M1_K1024_N4096_*`, `gemv_M1_K5376_N4096_*` (the P0 picks), `gemv_bf16_M1_K4096_N{240,304,368,
432,496}` (only if the `router` arm is not the default - the `_R` norm replaces them), `gemv_i8w_M1_K4096_N131072`,
`gemv_bf16_M1_K4096_N131072`, `argmax_stage{1,2}` at 131072 / 131072, Muse's `mg_attn` / `mg_attn_eager` at their defines.
- [ ] **Step 1: the failing host test** `bolt_variant_names_test`: every name `decode_variants` returns for (int8 / bf16 head)
  x (flash / eager) x (one / two cards) x (each arm) x (fp32 / bf16 state) is a target of the block. Decision 15's check: the
  KV layout `mg_attn` reads (`[pos][kv][hd]` K and V buffers, 23c) equals `nb_attn_prep`'s writes; if not, `nb_attn` /
  `nb_attn_eager` copies are added and the test names them. FAIL.
- [ ] **Step 2: write the kernels and the CMake lines;** `bolt_run` (Mac GPU, indicative) runs every plain-OpenCL kernel
  against the twin on the fixture - exact or the ulp distance printed; `bolt_kernels_test` (card) B2: every kernel and every
  arm against `bolt_ref.h`, bitwise where the arithmetic is the same, the attention flash against fp64 at cosine >= 0.99999.
- [ ] **Step 3: Mac gates** (`--kernels`: syntax, cmdlines additions only). **Commit** `git commit -S -m "kernels: the Bolt decode family - one-launch norm sites with the two-source fold, the conv ring, the fp32 SSM step, the 1024-group gated norm, the fp32 top-9 router on 256 x 2 lanes, the latent relu² experts with the ascending-id combine (spec 25c)"`.

### Task 4: `BoltEngine`, the CLI, `--profile`, B3 / B4

**Files:**
- Create: `src/runtime/bolt/bolt_{buffers,capture,engine}.{h,cc}`, `src/cli/bolt_decode.h`, `tests/runtime/bolt_decode_test.cc`
  (card), `tests/golden/bolt_golden_test.cc` (card), `tests/runtime/bolt_plan_test.cc` (extended)
- Modify: `src/runtime/bolt/bolt_sizes.{h,cc}` (the decode scratch; the launch counts), `src/cli/b70_decode.cc` (dispatch on
  `nemotron_h_omni` + `nemotron_h_puzzle` before the Qwen rules), CMake files

- [ ] **Step 1: the failing tests:** `bolt_plan_test` (host) - `device_launches` = 653 / 669 / 533 and per arm, +0 / +2 on two
  cards, the synthetic N 12's count, the decode scratch formula; `bolt_decode_test` (card) - replay twice bitwise (B4), the
  residual buffers after every layer (Review Focus 4), `--pp 2` at two splits bitwise (Review Focus 6) on the N 12 synthetic
  (one card vs two); `bolt_golden_test` (card) - B3 against the engine-format reference: tokens tie-aware (golden_common.h),
  routes as sets per (row, MoE layer) with one-expert differences inside `B70_BOLT_TIE_TOL` counted as near-ties and a 0 gap
  undetermined, h taps median >= 0.9998 / min >= 0.99 a layer, the SSM state and conv columns after the prompt by cosine (>=
  0.999, PROPOSED - not bitwise: the int4 GEMV's fp32 order vs torch's bf16 linear, 24c's finding) with the exact fraction
  printed.
- [ ] **Step 2: implement** the engine (one buffer set and list per device; the landing on device 1 carries h and the pending
  slices, spec 25 §4.1), `b70-decode <bolt>` (`--ids`, `--bench`, `--profile`, `--lm-head bf16|int8` (int8 default),
  `--max-len N|auto`, `--pp 1|2` (2 the default for the real model; 1 refused naming the bytes when it does not fit),
  `--pipeline-split`, `--layers N`; refused before the device by name: `--prefill*` (25d), `--mtp` (no MTP head), `--spec`
  (spec 26), `--kv-cache int8` (Decision 17)); PASS on the synthetics (box), host tests on the Mac.
- [ ] **Step 3: commit** `git commit -S -m "runtime: BoltEngine decodes on one or two cards - 653 launches, --profile, B3 / B4 on the synthetics (spec 25c)"`.

### Task 5: the fused arms (`B70_BOLT_FUSE`)

- [ ] **Step 1:** `router` (`nb_res_norm_R` at every MoE site), `route` (`nb_moe_up_R`), `conv` (`nb_ssm_C`); `B70_BOLT_FUSE`
  = `none` | any comma list | `all`; the capture's counts per arm. Test (card): each arm and `all` bitwise `none` after a prompt
  and 32 greedy tokens (Review Focus 5) on the synthetic and on the real checkpoint. **Commit** `git commit -S -m "runtime: the Bolt fused arms - router in the norm, route in the up, conv in the SSM step - bitwise the as-built list (spec 25c)"`.

### Task 6: the real checkpoint on two cards

- [ ] **Step 1:** B3 on SergiioB's checkpoint against `oracle-out-bolt` (short, 4k, agentic; 32k optin), split 48 and one other
  split bitwise each other; the `rtn` / `ours` stages when 25q's exports exist. **Commit** `git commit -S -m "tests: Bolt golden on SergiioB's g128 on two cards (spec 25c)"`.

### Task 7: docs, the box queue row

**Files:**
- Modify: spec 25 ("25c as built"), this plan's status, `docs/superpowers/plans/box-validation-queue.md` (row 60),
  `tools/box_validate/stages.sh` (`row 60`)

- [ ] **Step 1:** "25c as built" (the list as built, the arms' binaries, P0's stand-ins and scale factors, B3's bars as built,
  every departure with its cost if wrong). The queue row:

```text
row 60 "spec 25c - Nemotron 3.5 Bolt decode: P0 (653 / 533 skeletons, GEMV and SSM-state floors, the S sweep), the nb_* kernels, BoltEngine on two cards, the fused arms, the fp32 / bf16 state arm (plan 25c)"
stage r60.p0          60 default gpu  -                  -          "probe_bolt_step at depth 4096 / 65536, --sweep-s: the table into the facts sheet"
stage r60.k1          60 default gpu  -                  -          "bolt_kernels_test: B2 on every kernel and arm"
stage r60.synth       60 default gpu  bolt_synth         -          "bolt_decode_test + bolt_golden_test on the N 12 synthetics, one card and two"
stage r60.golden      60 default gpu  bolt_g128,oracle_bolt -       "B3 on SergiioB's checkpoint, two cards, split 48 and another (bitwise each other)"
stage r60.fuse        60 default gpu  bolt_g128          r60.golden "every arm bitwise none; the arms' A/B (interleaved)"
stage r60.state_arm   60 optin   gpu  bolt_g128,oracle_bolt r60.golden "B70_BOLT_SSM=bf16: B3 and the A4 subset beside fp32 (Decision 2's evidence)"
stage r60.speed       60 optin   gpu  bolt_g128          r60.golden "decode at depth 886 / 8192 / 32000 / 65536 / 131072 / 262000, int8 / bf16 head, flash / eager, arms"
```
- [ ] **Step 2: commit** `git commit -S -m "docs: spec 25 (25c as built), box queue row 60 - Bolt decode"`.

### Task 8: decode speed, Decisions 2 and 16 (box)

- [ ] **Step 1 (box, `r60.speed`, `r60.state_arm`):** the speed rows beside spec 25 §3's roofline and SergiioB's published
  table; the arms' interleaved A/B; the fp32 / bf16 state arm's speed and quality. The default list (Decision 16) and the state
  precision (Decision 2) are changed only by a commit naming these rows and the operator's ruling.

**Gate for the plan:** Mac - every host test, `bolt_run` exact (or its ulp distances recorded), cmdlines additions only, every
syntax check. Box - B0 (G0); P0's table written before any engine stage; B2 on every kernel and arm; B4 replay bitwise, every
arm bitwise `none`, splits bitwise each other; B3 on the synthetics and SergiioB's checkpoint; the speed rows and the arms' A/B
recorded.

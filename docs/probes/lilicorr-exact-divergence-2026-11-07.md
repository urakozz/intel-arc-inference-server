# LiLiCorr draft test: why the exact arm misses and the int8 arm passes (2026-11-07)

Code reading only. Nothing was built or run. Branch `diag-lilicorr-exact` adds a test-only `--diag` print to
`lilicorr_draft_test`, plus a swappable exp in the CPU twin, so tomorrow's run can confirm the cause.

File:line references are to main at 6c1aacf. On the branch, lilicorr_ref.h's lines after :116 are 5 further down,
because of the new exp hook.

## What the box printed (validation run `0c0977669f19-vs-0c0977669f19`, `g0.suite.log`)

| arm | load | step 1 (projection) | step 2 (draft at p 1100) |
|---|---|---|---|
| int8 | 25 NVFP4 linears, the head exact | ring rows 0 over 1 bf16 ulp, anchor cosine 1.000000000 | sets 7 / 7, in order 7, log-probs within **7.93e-4**, walk = |
| exact | 25 NVFP4 linears, the head exact | ring rows 0 over 1 bf16 ulp, anchor cosine 1.000000000 | sets 7 / 7, in order **6**, log-probs within **5.67e-3**, walk = |

The check is `sets == 7 && lc >= 0.99999 && walk && lpd <= 1e-3` (tests/runtime/lilicorr_draft_test.cc:210 on main).
Only `lpd`, the largest log-prob difference, fails. "Log-prob" here means a candidate token's logit minus the row's
log-sum-exp over all 131072 vocabulary entries. Two side notes:

- The printed "lattice cosine 1.000000000" in the exact arm was **not computed**. `lc` starts at 1 and is computed only
  when all 7 slots list their candidates in the same order (line 207: `if (same == 7)`); the exact arm had 6. The print
  should say "n/a" in that case. The check does not depend on it.
- One slot listing its candidates in a different order does **not** inflate `lpd`. The test compares log-probs rank by
  rank. When two near-tied candidates swap, the card's k-th largest logit and the twin's k-th largest logit still differ
  by at most the largest per-logit error. So 5.67e-3 is a real difference in the logits or the log-sum-exp, about 7x
  the int8 arm's.

## Where the card and its CPU twin can compute differently

Each arm is held to its own CPU twin (tests/kernels/lilicorr_ref.h `llref::draft`, with the weights built by
tests/kernels/lilicorr_fixture_io.h `drafter_weights`). The twin mirrors the kernels' operation order and turns off fma
contraction (fusing a multiply and an add into one rounding). The kernels turn it off too (`#pragma OPENCL
FP_CONTRACT OFF` in df_prep.cl:25, ll_attn.cl:20, ll_topk.cl:18, ll_lattice.cl:31). Every kernel is built with
`-cl-fp32-correctly-rounded-divide-sqrt` (cmake/ocloc.cmake:45), so `/` and `sqrt` match libm bit for bit. That leaves
two kinds of difference: summation order, and the transcendental functions (`exp`, `log`).

### Ruled out by the code

| place | the exact arm | the int8 arm | why it is not the cause |
|---|---|---|---|
| NVFP4 decode at load (NVFP4: NVIDIA's 4-bit float weights, a 1-in-16 scale and one global scale g) | src/loader/lilicorr_repack.cc:313-319: W' = s_b x e as bf16 tiles + g per column | :320-326: int8 of W', row scale fp32(g x s'_r) | the twin calls the same `LiliCheckpoint::layer_linear` / `fc` (lilicorr_fixture_io.h:162-176) and copies its bytes (`g.lin.bf16 = dev->bf16; g.gs = dev->gs`, :77-80). Same bytes on both sides |
| backbone GEMVs (GEMV: matrix x a few vectors) | `ll_gemv_bf16_g` for fc / gate‖up / down (lilicorr_drafter.cc:382-389), `gemv_bf16` for q‖k‖v / o / the conv projections (:390-395) | `gemv_i8w` for all (:374-380) | bitwise by construction. A bf16 x bf16 product has at most 16 significant bits, and int8 x bf16 at most 15, so each product is exact in fp32 (24 bits). Contraction therefore cannot change it, and the sum order is the twin's: k ascending in 8-wide steps (ll_gemv_bf16_g.cl:60-71 = dflash_ref.h:390-415). The split-K choice matches: the kernel's `gemv_bf16_tiling(N)` (kernels.h:70, 32 Xe-cores) and the twin's `bf16_ksplit(N)` (lilicorr_fixture_io.h:37) both give KSPLIT 16 only for N 1024 (the conv projections), with the same pairwise tree. g multiplies once after the sum on both sides (ll_gemv_bf16_g.cl:74 = lilicorr_ref.h:71-76) |
| the target head (the 131072 x 4096 vocabulary projection) | int8, synthetic | int8, synthetic | the same head for both arms (test :118-127, `HeadForm::Int8` :144) |
| the LiLiCorr head (the small lattice network that scores candidate pairs) | exact | exact | exact in every arm (lilicorr_loader.cc:115). It does not feed the log-probs either: they come out of `ll_topk` before the head runs |
| log-softmax (`ll_topk`: top-8 + log-sum-exp) | builtin `exp` / `log` | the same | its order matches the twin (ll_topk.cl:108-118, 157-165 = lilicorr_ref.h:212-239). The builtin exp/log are 1-2 fp32 ulps off libm. On a log-sum-exp of about 10, that is ~1e-6, three orders of magnitude below 5.67e-3, and the same in both arms |
| the projection (step 1) | k‖v at N 512 runs `gemv_bf16` with **KSPLIT 16** (lilicorr_drafter.cc:300-305). The twin sums the whole q‖k‖v at N 4608 with KSPLIT 1 (lilicorr_ref.h:433) | `gemv_i8w`, no split, the same as the twin | a real exact-only order mismatch, but step 1 allows 1 ulp and saw 0 rows over it. From step 2 on, the twin reads the **card's** ring and anchor (test :175-179), so it cannot reach step 2 |

So no arithmetic is arm-specific. Both arms run the same non-GEMV kernels, and the GEMVs are bitwise in both.

### Ranked candidates

1. **The attention's `exp`: the device's builtin against libm's `expf`, amplified through bf16 rounding.**
   ll_attn.cl uses the OpenCL builtin `exp` at :94, :161, :166 and :170. Its twin `attention_sink` uses `std::exp`
   (lilicorr_ref.h, now via `attn_sink_exp`, at the same four sites). Spec 19f measured this on the card on 2026-10-27
   and wrote it down in tests/kernels/dflash_ref.h:64-70: the builtin differs from glibc's expf in ~45 % of softmax
   arguments by 1-2 fp32 ulps. When an attention output cancels toward 0, that turned into **2-45 bf16 ulps**. DFlash's
   fix was its own exp (`df_exp` in df_attn.cl:58, twin `attn_exp`), built from + - x only so the kernel and twin agree
   bit for bit. **The LiLiCorr attention was written after that fix but did not take it.** ll_attn.cl and
   `llref::attention_sink` still use the builtin and libm. Each layer's attention output is rounded to bf16
   (lilicorr_ref.h:507-508, "ll_attn_reduce writes bf16") before o_proj. A one-ulp flip there goes through 5 layers of
   residual stream and reaches the final-norm rows, which feed the 4096-wide target head. Both arms carry this mechanism.
   Its size depends on where the outputs happen to cancel, and those places differ between the two weight sets (exact W'
   against its int8 rounding). That would explain why one arm lands at 7.9e-4 and the other at 5.7e-3. It also means the
   int8 arm's pass, at 79 % of the 1e-3 bar, is luck rather than margin.
2. **SiLU's `exp` in `df_silu_mul`** (df_prep.cl:130, builtin) against the twin's `dfref::silu_exp` (dflash_ref.h:259-260,
   libm). SiLU is the MLP's activation, g / (1 + e^-g). Same mechanism, one site per layer: the activation is rounded to
   bf16 (lilicorr_ref.h:525) before `down`. It is probably smaller than candidate 1 because SiLU output does not cancel
   toward 0 the way attention does, but it is not excluded.
3. **The 1e-3 absolute bar is tighter than a 5-layer bf16 backbone with non-bitwise exp can hold.** The test's other
   checks are bitwise or tie-aware; this one assumes the backbone is close to bitwise. If candidates 1 and 2 are
   confirmed, the choice is between making the exp bitwise (candidate 1's fix) and pricing the bar. See the operator
   decision below.
4. **Something exact-specific that reading missed.** For example, a miscompile in one of the three `ll_gemv_bf16_g`
   real-shape binaries (CMakeLists.txt:1712-1718). Step 1 tests only fc's binary (K 24576), not gate‖up's (K 4096,
   N 24576) or down's (K 12288). lilicorr_kernels_test runs the kernel only at the fixture's tiny shapes. Unlikely, but
   `--diag` line (a) would expose it as a jump in error.

## The one box run that decides it (tomorrow, quiet card)

The `--diag` argument on branch `diag-lilicorr-exact` prints only. It changes no check and no default: the twin's new
`llref::attn_sink_exp` defaults to libm, exactly as before. After step 2 it prints:

- **(a)** per row of the 8-row block: the card's final-norm rows against the twin's (cosine, max |diff|, how many bf16
  inputs to the head differ) - is the backbone where they part?
- **(b)** the target head + top-k + log-softmax run in the twin over the **card's** rows - logits bitwise and log-probs
  within ~1e-6 clears the head and the log-softmax;
- **(c)** the worst log-prob split into logit and log-sum-exp, plus each slot's 8 candidates on both sides;
- **(d)** the twin against itself, with only one exp swapped from libm's `expf` to `attn_exp` (an exp 1-2 ulps away,
  the same size as the builtin's distance from libm): SiLU's, the attention's, then both. This gives the backbone's
  sensitivity to exp ulps, measured on the host.

From the Mac, on the branch (`tools/box.sh` syncs this checkout's tree; build with -j44):

```
git switch diag-lilicorr-exact
JOBS=44 tools/box.sh build
tools/box.sh run 'tools/gpu_lock.sh card 0 -- build/tests/lilicorr_draft_test nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr exact --diag 2>&1 | tee ~/b70/logs/lilicorr-diag-exact.txt; tools/gpu_lock.sh card 0 -- build/tests/lilicorr_draft_test nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr int8 --diag 2>&1 | tee ~/b70/logs/lilicorr-diag-int8.txt'
```

About 1 minute per arm on the card, plus a few seconds of CPU per twin draft (four twin drafts per arm). The host holds
the twin's weights plus a 1 GB synthetic embedding and head, as the test already does.

**Reading it:**

| result | means |
|---|---|
| (b) bitwise / ~1e-6, (a) cosine < 1 with a few bf16 flips, and (d) "the attention's exp" moves the twin's log-probs by about the same 5e-3 in exact (and ~1e-3 in int8) | **candidate 1 confirmed**: exp ulps amplified, no engine defect |
| (d) "SiLU's exp" carries most of it | candidate 2 |
| (b) not bitwise | the target head GEMV or `ll_topk` - not predicted by the reading; look there |
| (a) large error and (d) small (< 1e-3) | something exact-specific (candidate 4): dump per layer next |

## Decision for the operator (after the run)

If candidate 1 is confirmed:

- **Option A (recommended): give the LiLiCorr attention DFlash's bitwise exp.** In ll_attn.cl, replace the four `exp`
  calls with `df_exp`, copied from df_attn.cl:58. In the twin, set `attn_sink_exp = dfref::attn_exp` as the default. The
  kernel and twin then agree bit for bit there, as DFlash already does since 19f. The bar stays at 1e-3. This is a
  kernel change, so its binary name and bytes change. It needs a rebuild and `lilicorr_kernels_test` +
  `lilicorr_draft_test` on the card. If (d) shows SiLU matters as well, do the same for `df_silu_mul`. That one is
  shared with DFlash's df_prep.cl, so it needs a family copy or DFlash's tests re-run.
- Option B: keep the kernels and widen the log-prob bar to a measured one, for example 2x the larger of (d)'s numbers.
  This is cheaper, but the bar then hides a real 1-2 ulp drift and stays luck-dependent.
- Option C: judge step 2 by an f32 reference, as the Qwen3.8 speculative tests are. This is heavier and not needed for
  a drafter whose outputs only propose tokens: verification is bitwise elsewhere.

Either way, the draft-format choice (int8 if `r65.d2`'s acceptance holds, else exact) does not depend on this. The exact
arm's miss is the test's tolerance against exp ulps, not a sign that exact drafts worse.

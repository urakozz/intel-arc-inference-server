# Probe: Muse-Glimmer-30B's DFlash drafter - P0 (spec 23f, 2026-10-16)

Spec 23 §8.3 / plan 23f Tasks 1-5: is `meta-models/Muse-Glimmer-30B-assistant` worth its verify on our card? Every
number below is **measured on the Mac's CPU** unless marked ESTIMATED or DERIVED, and the Mac's sample is SMALL: the
two golden continuations 23a made (`muse_short_code`, `muse_short_prose`: 32 greedy ids each, 18 anchors each at
K_max 15). The deciding corpora - the 36 A4 scenarios (512-id int4 continuations) and the agentic session (13.4k ids +
512) - are the box's (queue row 47: `r47.taps`, `r47.accept`); so are the verify costs (`r47.cost`). The operator
ruled (2026-10-15) that the engine half is built whatever P0 shows: P0 picks the defaults and sets the expectation.

## 1. The reference (Task 1)

`tools/oracle/dflash_ref.py`'s Muse form is **bitwise** transformers 5.19.0's `MuseGlimmerAssistantModel` +
`DFlashTokenCandidateGenerator` (bf16 and fp32) on 23a's tiny pair (`l8` target, `assist` / `assist16` drafter):
anchors 20..59, a 3-accepted continuation with random rejected rows and every layer's cache after the crop, and the
real window's edge at p = 2047 / 2048 / 2049 / 3000 (row 0 sees p - 2048, not p - 2049). Removing each of Review Focus
4's rules flips drafts (the one-rounding norm 29 / 600, the embedding norm 600 / 600, q x 3.87 592 / 600, theta 1e6
240 / 600). `tools/oracle/README.md` "The Muse-Glimmer form".

## 2. Teacher-forced acceptance (Task 3, the Mac's two short dumps)

`tools/oracle/dflash_p0.sh` (MODEL=muse) over `oracle-out-muse-taps/dumps` (`muse_taps.py from-golden` of 23a's two
golden sets: the int4 reference's own greedy continuation, raw greedy = the recorded text at 32 / 32 rows), all arms,
float32 compute, 577 s wall (12 threads, 36 GB cap). Greedy E_K (tokens per verify, the target's own token included):

| corpus | arm | E_1 | E_3 | E_5 | E_7 | E_10 | E_15 |
|---|---|---:|---:|---:|---:|---:|---:|
| code (18 anchors) | bf16 | 1.89 | 3.44 | 4.83 | 6.06 | 7.72 | **9.33** |
| | int8 | 1.89 | 3.50 | 4.89 | 6.11 | 7.78 | 9.39 |
| | int8h (int8 head) | 1.89 | 3.50 | 4.89 | 6.11 | 7.89 | 9.78 |
| | int4g64 | 1.89 | 3.44 | 4.83 | 6.00 | 7.50 | 8.83 |
| | int4g64h (int8 head) | 1.89 | 3.44 | 4.83 | 6.06 | 7.72 | 9.33 |
| prose (18 anchors) | bf16 | 1.50 | 1.78 | 1.78 | 1.78 | 1.78 | 1.78 |
| | int8h / int4g64h | 1.56 | 1.83 | 1.83 | 1.83 | 1.83 | 1.83 |
| all (36) | bf16 | 1.69 | 2.61 | 3.31 | 3.92 | 4.75 | 5.56 |
| | int4g64h | 1.72 | 2.64 | 3.33 | 3.94 | 4.78 | 5.58 |

- **Code accepts deep**: per-depth alpha stays 0.83-1.00 through depth 14 at every format - the block of 16 pays where
  the text is predictable; prose saturates at ~1.8 (the drafter's first two drafts) - 19a's pattern on Qwen3.8.
- **Formats** (decision 9: int4 g64 RTN if within 0.05 of bf16's E_K at the chosen K): with the int8 head the engine
  serves, int4 g64 equals bf16 on every row of both corpora at K 15 (5.58 vs 5.56 over all; 9.33 = 9.33 on code); the
  bf16-head int4 arm is 0.25 below at K 15 (one anchor's tail: 18 anchors, a draft is worth ~0.06). **Provisional
  default: int4 g64** (1.36 GB against 5.11), to be confirmed on A4 + the agentic session (r47.accept).
- **Sampled, the q form (decision 11)** - the expected-acceptance formula at T 1.0 / top-k 64 / top-p 0.95 (the
  model's defaults), q = the draft row's top 16 at T:

| corpus | q | E_1 | E_3 | E_7 | E_15 |
|---|---|---:|---:|---:|---:|
| code | **raw** | 1.87 | 3.24 | 4.82 | **5.58** |
| code | capped | 1.27 | 1.35 | 1.35 | 1.35 |
| prose | **raw** | 1.66 | 1.96 | 1.97 | 1.97 |
| prose | capped | 1.36 | 1.47 | 1.47 | 1.47 |

  **The raw form wins by far**: the drafter was trained in transformers' form (drafts from softmax(raw logits), G:1680-
  1683), so its raw rows are what tracks the target's capped p; capping them again flattens q ~5x (the multiplier
  0.196) and the overlap falls to ~0.3. Decision 11 recommended the capped form; **the measurement rules raw as the
  provisional default** (`--draft-q raw`; both stay selectable; lossless either way).
- **Prompt lookup on the same continuations** (`lookup_accept.py --dumps`, ESTIMATED verify costs past M = 4): ~1.04x
  at n = 3 (verify share 0.09) - short golden prompts give lookup almost nothing to copy; the A4 / agentic corpora are
  where it matters (19e measured 1.44x on Qwen3.8's A4).

## 3. The verify cost (Task 4) - the box's

`probe_draft_cost --shapes muse --mmax 16 --depth 4096,32768` (r47.cost) times the 52 layers' int4 GEMVs + the int8 head
+ mg_argmax at M = 1..16 as interleaved pairs against M = 1 (and the two-pass form at M = 9..16), the attention term
ESTIMATED from the KV bytes, and one drafter block at 16 rows in bf16 / int8 / int4 g64. Not run on the Mac (no card).

## 4. The projection (Task 5)

`tools/spec/muse_p0_project.py` - DFlash tokens per plain step = E_K / (verify(K + 1) + draft), lookup's the same
verify costs with a free draft (AdaptiveK, n = 3) - with the DERIVED cost table (verify 1 / 1.18 / 1.56 / 1.79, then
+0.305 a row to 5.45 at M = 16; the block 0.18 int4 g64, 0.26 int8, 0.43 bf16 plain steps):

| arm | corpus | best K | E_K | DFlash x plain | lookup x plain |
|---|---|---:|---:|---:|---:|
| int4g64h | code | 8 | 6.67 | 1.906 | 1.091 |
| int4g64h | prose | 1 | 1.56 | 1.142 | 1.000 |
| int8h | code | 9 | 7.33 | 1.889 | 1.091 |
| bf16 | code | 9 | 7.22 | 1.781 | 1.091 |
| bf16 | prose | 1 | 1.50 | 0.929 | 1.000 |

Sampled (q raw, the expected-acceptance E_K): int4g64h code 1.64x at K 4, prose 1.22x at K 1. The stopping rule is
UNDECIDED on the Mac (no agentic corpus); spec 23 §8.5 records the verdict and the provisional defaults (int4 g64 + the
int8 head, K_max 15 under AdaptiveK, q raw). Outputs (git-ignored, the main checkout): `oracle-out-muse-taps/dumps/` (2
dumps, 10.7 MB), `oracle-out-muse-p0/` (accept/, lookup.{txt,json}, project.*).

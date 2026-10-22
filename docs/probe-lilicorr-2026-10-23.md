# Probe: LiLiCorr on our Bolt - P0 (spec 26a, 2026-10-23)

Spec 26 §8 / plan 26a: what NVIDIA's LiLiCorr drafter (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` @
`449d2414`) is worth on OUR Bolt (SergiioB's g128 dequantised - what we serve; LiLiCorr was trained against NVIDIA's 2-bit
24GB release, Decision 9). **P0 sets the defaults and the expectation; it gates nothing** (the operator's 2026-10-15
ruling: 26b / 26c are built whatever it shows). Every number is **measured on the Mac** unless marked ESTIMATED,
DERIVED or PENDING (the box's: queue row 63). Nothing of the real drafter has been read: its 962 MB of weights were not
downloaded when this was written (**PENDING-WEIGHTS**: `hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr
--revision 449d24140160b8e51871bfc0f0a7338abf640aa1`), and the real Bolt tap dumps (row 57's `r57.taps`) are the box's.

## 1. The reference (Task 1, Mac)

`tools/oracle/dflash_ref.py`'s LiLiCorr form (vLLM 3901d56800's `LiLiCorrHead` + the DFlash 2 backbone with sinks; spec 26
§2; `tools/oracle/README.md` "The LiLiCorr form"), on a tiny random drafter (`lilicorr_make_tiny.py`: 2 layers at 256, the
head at 64, window 16, NVFP4-packed with a non-trivial global scale) - `test_dflash_ref.py`'s `test_lilicorr_*`, 12 tests,
~10 s in `agnes-ref-img`:

- **NVFP4 decodes exactly** (Review Focus 3): all 16 x 254 finite (e2m1 code, e4m3 scale) products are exact in bf16 AND
  f16, |W'| in [2^-10, 2688]; the 2 NaN scale codes are refused (16 x 2 pairs) naming the tensor and element; the e4m3
  table equals torch's `float8_e4m3fn` on every finite code; k = 2i in the low nibble, one scale per 16 k. g on the
  output: `g x (x W'^T)` bitwise; folding g into W' moves 2125 / 2304 fc outputs and the lattice at 20 / 20 anchors.
- **Every Review Focus item is a mutation that moves the lattice or the drafts** (20 anchors x 7 drafts, window 16):

| mutation | lattice moved | drafts flipped (of 140) |
|---|---:|---:|
| anchor = the anchor token's embedding (RF 1) | 20 | 107 |
| anchor = fc's output before hidden_norm (RF 1) | 20 | 18 |
| anchor = the row at p - 2 (RF 1) | 20 | 119 |
| anchor_valid false (RF 1) | 20 | 104 |
| slot 0 = start transposed (RF 2) | 20 | 33 |
| slot s scored against itself (pair index shift, RF 2) | 20 | 51 |
| the slot bias's sign flipped (RF 2) | 20 | 12 |
| no same-slot bias (RF 2) | 20 | 5 |
| log-probs added to the walk's scores (RF 2) | 0 | 103 |
| log-probs normalised over the top 8 only (RF 2) | 20 | 10 |
| W_cross on h instead of h x a (RF 2) | 20 | 115 |
| no sink (RF 4) | 20 | 57 |
| the sink scaled by head_dim^-0.5 (RF 4) | 20 | 56 |
| g folded into W' (RF 3) | 20 | 0 |
| the sampled walk without the step seed | - | 109 (T 1.0) |

  The slot bias's trained offset 7: a 3-slot lattice (vLLM's short-draft prefix form) moves under the offset n_slots, the
  7-slot one cannot (equal by construction). x 8.0 is exact (a power of two: the unscaled lattice x 8 is bitwise the
  lattice). The walk's start index cannot matter (slot 0 is a broadcast; every start gives the same drafts - asserted).
  The taps in another order move the lattice (RF 5's API half; the tiny pair below holds the other half).
- **The window** (16 here; 1024 real; vLLM's `|q - k| <= window - 1`, Decision 11): row 0 at p sees p - 15 .. p - 1, row 7
  sees p - 8 .. p - 1 (p = 15, 16, 17); positions older than p - 15 change nothing (<= 1e-5: masked keys add exact zeros),
  p - 15 does (> 1e-3); the inclusive reading moves the block.
- **K = 1..7 are prefixes of one walk** (one 8-row block, the head over 7 slots - spec 26 §5.3; vLLM runs a (1 + K)-row
  block at K < 7, not this form).
- **Candidates**: top 8 by logit, exact ties to the lower id; `torch.topk` agreed with that order on 4 of 5 rows (2 with
  planted exact ties) - recorded, not required (24a's lesson).
- **The cross-check against vLLM's own `LiLiCorrHead`** (Decision 1; `tools/oracle/lilicorr_vllm_check.py`): vLLM is NOT
  importable in `agnes-ref-img` (16 missing dependencies, and stubbing pydantic / msgspec breaks vLLM's own classes -
  measured 2026-10-23), so it runs on the box in the vLLM image: **PENDING (r63.vllmref)**.

## 2. The tiny pair (Task 2, Mac)

25a's tiny Bolt (`pattern12`, 12 blocks at hidden 256, vocab 131072) + the tiny LiLiCorr at taps 1 / 2 / 4 / 6 / 9 / 10
(the real 1 / 18 / 35 / 51 / 68 / 85's kinds: MoE first and last, Mamba between): `bolt_ref.py run` (48 random prompt
ids + 24 greedy) and `bolt_ref.py taps --tap-layers 1,2,4,6,9,10` (new: `--tap-layers`) write the dump; **the dump's
tap.L{i} is bitwise `run`'s h.L{i}** over the same 72 ids at every tap (the residual after block i; `test_bolt_ref.py`
ties h.L{i} to `output_hidden_states[i + 1]`) - RF 5; taps one block early flip 94 / 126 drafts. `dflash_accept.py run
--model bolt-lilicorr` over it: the batched drafts = `draft_block` on all 18 anchors; E_7 = 1.00 in every arm (random
weights: the drafter never predicts the tiny target - a plumbing check, not a measurement). On the synthetic target: the
batch = `draft_block` per anchor (candidates and ids of every K bitwise, the lattice within 9.5e-6 - batched GEMMs); the
int8 arm's codes are independent of g (x 3.7: the same q) with the row scale fp32(g x s'); the int4 arm is rtn.h's rule
on g x W' in fp64 (on bf16 inputs the fp64 and fp32 rules agreed on 256 / 256 scales and 16384 / 16384 codes); a text
that follows the drafter's own drafts accepts to depth 7; `lattice+lp` gives other sampled sums than `lattice`.

## 3. Acceptance on our Bolt, lookup, the expert union (Task 3) - PENDING (box)

`r63.accept` (`MODEL=lilicorr tools/oracle/dflash_p0.sh`): per corpus (the 36 A4 continuations' first 512 ids, prose /
code at 512 greedy ids, the agentic session) and arm (`exact`, `int8`, `int4g64`, and the `h` arms with the int8 head
b70-serve runs): greedy E_K for K = 1..7, the per-depth acceptance, the sampled estimate at T 1.0 / top-p 0.95 / no top-k
for q `lattice` and `lattice+lp`; lookup's E_K on the same continuations (`lookup_accept.py --dumps --max-k 7`).
`r63.union` (`tools/spec/expert_union.py` over the golden runs' route records): u, the distinct experts of M = 2..8
consecutive rows per MoE block - mean / p50 / p90 per M and per E bucket (240 .. 496) - and the routed bytes a verify
reads, union-aware against the plain M-row form. DERIVED from the config: an expert block is 2 x I x 544 B (2.500 MB on
average over the 40 MoE blocks); a plain row reads 0.900 GB of routed experts, so the plain M-row form reads 7.2 GB at
M = 8 (spec 26 §4's figure).

## 4. The verify and block costs (Task 4) - PENDING (box)

`r63.cost`: `probe_draft_cost --shapes bolt-lilicorr --mmax 8 --depth 4096,65536` on one card - Bolt's 88 blocks' GEMVs at
the descriptor's cells, 25c's `nb_router` and the int8 head + argmax stage 1 at M = 1..8, the routed experts in the plain
M-row form (9 M stand-in blocks a MoE block, each its own allocation), by part and whole, against M = 1 as interleaved
pairs; the experts at 9 and 72 blocks a MoE block (the time per block - the union form's cost at u is 40 x u x it); the SSM
state (`nb_state_rmw` over 40 x 4 MiB) and from its rate the slot form's (1 + M) x 4 MiB and the replay form's 3 x 4 MiB a
Mamba block; the attention ESTIMATED from 8,192 B of KV a position; one LiLiCorr block per drafter format (the drafter at
8 rows, the int8 head at 7 + argmax stage 1 as the top-8 + logsumexp stand-in, the bf16 LiLiCorr head as 7 launches of 8
rows) and the LiLiCorr head alone (spec 26 §5.2 ESTIMATED it at ~0.1-0.2 ms). kernel_cmdlines +99 / -0 / ~0; none of the
new binaries has been compiled by ocloc; `lilicorr_cost_names_test` holds the 110 names the probe binds to the 99 the block
builds + 11 of 25c's.

## 5. The projection and the defaults (Task 5)

`tools/spec/lilicorr_p0_project.py` (r63.verdict, by hand on the box's numbers): LiLiCorr at K = E_K / (verify(K + 1) +
block), lookup over the same verify costs with a free draft, plain = 1; SPEED-Bench's published acceptance beside ours; the
defaults. Until the box's numbers exist it runs on the DERIVED cost table (spec 26 §11; depth 4096):

| | M = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| verify, plain M-row MoE (plain steps, DERIVED) | 1.000 | 1.193 | 1.387 | 1.580 | 1.773 | 1.966 | 2.160 | 2.353 |
| slot writes' share of it (DERIVED) | - | 5.1 % | 6.6 % | 7.7 % | 8.6 % | 9.3 % | 9.8 % | 10.3 % |

One block: exact 0.500 / int8 0.306 / int4 g64 0.214 plain steps (DERIVED). ESTIMATED expectation with SPEED-Bench's
published acceptance (the 24GB target - Decision 9's gap is what r63.accept measures) and the int8 drafter: coding 5.26 ->
1.98x plain (plain M-row MoE) / 2.77x (union MoE at u = 30, §4's ESTIMATE); overall 4.30 -> 1.62x / 2.27x; writing 3.41 ->
1.28x / 1.80x. **The defaults until then (PROVISIONAL, spec 26 §11):** int8 per row, K_max 7 under AdaptiveK, q `lattice`, the
replay lever not built, the union MoE built in 26c regardless. Outputs on the box: `oracle-out-lilicorr-p0/` (accept/,
lookup.{txt,json}, union{,.gen}.{txt,json}, project.{md,json}).

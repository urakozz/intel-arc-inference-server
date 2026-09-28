# Prefill continuation: two calls against one, and against M = 1 decode (2026-09-28)

**Verdict: not a bug.** A prefill that starts at a non-zero `pos` is correct. What looked
like a continuation bug is the model amplifying a rounding-sized perturbation at a few
positions, and the CPU oracle shows the same amplification from chunking alone.

## The suspicion

1. Plan 8b (`docs/probe-mtp-2026-09-27.md` §6, "Prefill fills the head's KV"): the main
   model's post-final-norm hidden after l0-int8 prefill against plain M = 1 decode of the
   same ids had median cosine 0.99971, 15 rows below 0.9, and **0.27 at row 2091**, just
   past the first 2048-chunk edge.
2. Plan 7c (branch `spec7c-server-prefix-cache`, `prefix_gpu_test` sequence g): the same
   prompt prefilled in two `Engine::prefill` calls split at 3000 gave last-row logits
   **cosine 0.91** on l0 against a one-call prefill.

## Localising it: `prefill_split_test`

`tests/prefill/prefill_split_test.cc` prefills long32k[0, 2600) in one call, then in two
calls split at 1, 7, 16, 63, 64, 1000, 2047, 2048 and 2049, then by M = 1 decode (the last
id by a one-id prefill). It compares every FA layer's K and V cache rows (K/V row p of FA
layer f is a function of the hidden state entering that layer at p, so this localises by
layer and row), every GDN layer's state, and the last-row logits.

Device 0, flash attention, 2026-09-28:

| run (l0) | last logits | FA 0-3 worst row | all (layer, row) median | p01 | rows < 0.9 | worst | GDN L0 state |
|---|---:|---:|---:|---:|---:|---|---:|
| split 1 | 0.999920 | 0.999832 | 0.9999395 | 0.99579 | 35 | 0.148 (fa 11, row 1565) | 0.9999980 |
| split 7 | 0.999897 | 0.999810 | 0.9999415 | 0.99669 | 26 | 0.181 (fa 13, row 987) | 0.9999974 |
| split 16 | 0.999882 | 0.999873 | 0.9999394 | 0.99632 | 39 | 0.143 (fa 11, row 1565) | 0.9999976 |
| split 63 | 0.999910 | 0.999797 | 0.9999427 | 0.99694 | 19 | 0.199 (fa 13, row 987) | 0.9999982 |
| split 64 | bitwise | | | | | | |
| split 1000 | 0.999890 | 0.999764 | 0.9999721 | 0.99813 | 18 | 0.122 (fa 10, row 2091) | 0.9999977 |
| split 2047 | 0.999904 | 0.999851 | 1.0000000 | 0.99975 | 12 | 0.085 (fa 11, row 2091) | 0.9999984 |
| split 2048 | bitwise | | | | | | |
| split 2049 | 0.999929 | 0.999870 | 1.0000000 | 0.99980 | 0 | 0.921 (fa 13, row 2094) | 0.9999982 |
| decode M = 1 | 0.999911 | 0.999810 | 0.9999345 | 0.99552 | 39 | 0.171 (fa 13, row 987) | 0.9999923 |

l0-int8 has the same shape one step noisier: last logits 0.99951 to 0.99973, FA 0-3 worst
0.99854 (decode 0.99793), median 0.99961 (decode 0.99953), GDN L0 state 0.9999972 (decode
0.9999763); splits at 64 and 2048 bitwise. `B70_PREFILL_ATTN=composed` on l0 gives the same
picture (split 1000: worst 0.121 at fa 13 row 2091), so flash attention is not involved.

Reading it:

* **Splits at a multiple of 64 are bitwise the one-call run**, KV, GDN state, live conv
  slots and logits. Everything on the walk is either row-local or keyed to absolute
  positions (RoPE, KV writes, flash attention's tiles and causal bound, the conv ring);
  the only thing a split changes is where the chunked gated delta rule cuts its 64-id
  chunks, which is relative to the call's first id. So `(pos + m) % 16`, RoPE offsets,
  the KV write rows and the flash-attention bound are all proven by the bitwise rows;
  any other split can only re-round GDN.
* **Other splits re-round GDN and nothing else, by GDN's own rounding size.** GDN layer
  0's state, whose inputs are identical, moves by 2-3e-6 in cosine; FA layers 0-3 (model
  layers 3-15) stay >= 0.9998 on every row. A real hand-off bug (lost conv seed or state,
  wrong RoPE/KV offset) moves every row after the split from FA layer 0 on (see the fault
  injection below).
* **The low rows are a handful of positions, the same ones in every run** (987, 1050,
  1212, 1565, 2091 of long32k), and they collapse only from FA layer ~7 on. M = 1 decode
  hits the same positions. Positions, not offsets: whatever the split, the same rows.

## Ground truth: the CPU oracle

`tools/oracle/dump.py` on long32k[0, 2100) in one forward (every position's logits; 1894 s,
40 threads, 71.4 GiB peak), and `tools/oracle/last_logits.py` on the same ids through one
cache in chunks of 1000 (`--at 988,1213,2092 --chunk 1000`, 1471 s): the oracle's own
split. Outputs in `oracle-out-fixpf/` on the box.

| row | engine one call | engine split 7 | engine split 1000 | engine split 2047 | engine decode M = 1 | **oracle chunk 1000 vs oracle one forward** |
|---|---:|---:|---:|---:|---:|---:|
| 203 | 0.999876 | 0.999918 | | | 0.999961 | |
| 987 | **0.937957** | 0.999066 | | | 0.999713 | 1.000000 (row 987 is before the first cut) |
| 1050 | 0.988117 | 0.987296 | 0.985094 | | 0.999317 | |
| 1212 | 0.981983 | 0.890661 | 0.984102 | | 0.911408 | **0.998730** |
| 1565 | 0.996263 | 0.991429 | 0.997470 | | 0.993316 | |
| 2091 | 0.995550 | 0.981985 | **0.755507** | **0.757822** | 0.981372 | **0.759646** |
| 2099 | 0.999704 | 0.999100 | 0.998155 | 0.998579 | 0.999599 | |

(Engine columns: l0 flash, last-row logits of a prefill of ids[0, row] against the one-forward
oracle; top-1 agrees with the oracle in every cell.)

**The oracle itself, chunked at 1000 instead of one forward, is at cosine 0.7596 at row
2091** -- the engine's split-1000 row is 0.7555. The fp32-softmax bf16 reference moves that
row by the same amount from the same cause (a different reduction grouping), so the 0.27
hidden / 0.08 KV at row 2091 is a property of the model at that position, not of the
engine's continuation. Row 987 makes the same point the other way round: there the
ONE-call engine run is the outlier (0.938) and the split and decode runs are close to
the oracle. No path is uniformly closer; each rounding picks different victims.

Plan 7c's 0.91 is the same class: its split is at 3000 (3000 % 64 = 56, a re-rounded GDN
chunking) and its last row is one prompt position; 0.91 sits between the oracle's own
0.9987 and 0.7596 above. Plan 8b's 0.27 is prefill (int8) against decode, the decode row
of the table above at row 2091 (0.98 on logits after the final norm and lm_head).

## Fault injection: the test catches a real continuation bug

`pf_gdn_seed` launched with `pos = 0` (every chunk after the first loses its conv
window -- the kind of hand-off bug that was suspected), l0, splits 7, 64, 1000:

| run | last logits | FA 0-3 worst row | median | p01 | GDN L0 state | verdict |
|---|---:|---|---:|---:|---:|---|
| split 7 | 0.999683 | **0.449** (fa 1, row 2048) | 0.99977 | **0.932** | **0.99981** | FAIL |
| split 64 | 0.999154 | **0.449** (fa 1, row 2048) | 0.99976 | **0.903** | **0.99852** | FAIL (not bitwise) |
| split 1000 | 0.999546 | **-0.011** (fa 3, row 1000) | 0.99990 | **0.842** | **0.99976** | FAIL |

The injected bug shows at exactly the rows where a chunk starts (1000, 2048) from FA layer
0-3 on, and in GDN layer 0's state by 20-60x the clean runs' distance. The last-row logits
alone would NOT have caught it (0.9997, the clean runs' band), which is why neither the
golden gates nor K3a saw anything either way: the test's discriminating bars are the
shallow-layer rows and GDN L0's state.

## The regression test

`prefill_split_l0_test` and `prefill_split_l0-int8_test` (label `checkpoint;prefill`,
`tests/CMakeLists.txt`). Gated: one call twice bitwise; splits at multiples of 64 bitwise
(KV, GDN state, the three live conv slots, logits); every other split and M = 1 decode
against bars a few times below the measured values above (l0 split: logits >= 0.9995,
FA 0-3 every row >= 0.999, median >= 0.9998, p01 >= 0.99, GDN L0 state >= 0.99999;
l0-int8 and decode looser, in the source). Not gated: deep-layer single rows, for the
oracle reason above.

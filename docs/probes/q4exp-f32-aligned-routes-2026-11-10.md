# Flash-Next f32-judged expert replay

Both registered gates completed and **FAIL** with the unchanged 2x bar. The requested near-tie rule is
implemented in signed `a44d0a43`: replay the engine's expert set and slot order only when the actual
exchanged experts fit within one bf16 ulp of the f32 router cut. Recompute mixing weights from the
f32 softmax. Rejected choices remain unforced and fail routing.

The full runs still contain rejected routes on both paths, already at layer 0. Consequently, the
remaining state-distance failures do **not** establish a prefill-specific conv/GDN defect. The next
isolated diagnostic is layer-0 decode GDN/HC/router at the first rejected position, with identical
intermediate inputs. No engine arithmetic or acceptance threshold changed; the branch stays unmerged.

## Captures and reference identity

Separate prefill/decode captures cover all 4500 prompt positions plus 32 forced continuation steps at
18 and 37 layers. Both f32 companions use the original unaligned f32 continuation, avoiding a circular
dependency between reference token generation and engine route capture. Each companion uses its own
engine route trace and independently recomputed f32 mixing weights. The engine's int8 head, physical
int8 PLE table, prompt, vocabulary and capture dimensions are checked.

Staging v3 files cannot be passed directly to the gate. Paired v2 files preserve both capture identities
and all routing verdicts. The registered gates checked every prompt/generated engine route against
its capture. Prefill's chunk-call snapshot also matched its normal whole-prompt call bitwise.

## Router verdicts

Counts are layer-position decisions, each selecting ten experts, over prompt **and** continuation.

| layers | path | exact set | accepted near-tie | rejected | prompt / continuation rejections |
|---|---|---:|---:|---:|---:|
| 18 | prefill | 70451 | 9846 | 1279 | 1260 / 19 |
| 18 | decode | 70588 | 9899 | 1089 | 1064 / 25 |
| 37 | prefill | 138122 | 20677 | 8885 | 8832 / 53 |
| 37 | decode | 138684 | 20587 | 8413 | 8333 / 80 |

At layer 0, decode has 14 rejected rows and prefill 18, at both truncations. The first decode rejection
is position 62: exchanged-expert logit span 0.0338935852 exceeds the 0.03125 window. Prefill's first is
position 156: span 0.0460858345 exceeds 0.03125. This disagreement already exists in decode at the first
MoE layer; downstream route divergence cannot explain that first-layer observation.

Full per-layer counts, first rejected positions and worst rejected spans:
[110-row route audit](q4exp-f32-route-judgment-2026-11-10.csv).

## Gate results

| gate | result / elapsed | state kinds above 2x | f32-accepted tokens, decode / prefill |
|---|---|---|---|
| one card, 18 layers | FAIL / 345.63 s | 4 conv kinds; last logits | 32/32 / 32/32 |
| two cards, 37 layers | FAIL / 755.44 s | 15 conv kinds; 2 GDN kinds; 3 QSA tails; last logits | 28/32 / 29/32 |

Both also fail six last-chunk route-distance ratios (layers 2, 3, 4, 5, 8, 12), and the all-position
routing verdict audit. No selection-distance ratio fails. Token agreement improved from the initial
unaligned reference: 24/27 accepted at 18 layers, 20/20 at 37 layers (decode/prefill).

Conv failures at 18 layers are L13 2.0331x, L14 2.2692x, L16 3.2533x, L17 4.8139x. Last logits are
0.000836546031 / 0.0001721885 (prefill/decode distance), 4.8583x.

At 37 layers the conv failures are L13,14,16,17,18,20,21,22,25,28,29,32,33,34,36; worst is L17
4.8139x. GDN L33 is 0.00204410458 / 0.0008303673 (2.4617x); L36 is
0.00381972117 / 0.00180192691 (2.1198x). Last logits are 0.0235612881 / 0.00961167543 (2.4513x).
QSA tails fail at L19,31,35. All 208 per-layer/kind measurements, including passing rows, are in the
[distance CSV](q4exp-f32-aligned-judgment-2026-11-10.csv).

These are full-kind cosine distances, not a bound on every individual state row. Set distance is the
mean missing-member fraction over the last 404 prompt rows. Ratios compare each engine path against
its corresponding f32 companion; rejected routing prevents treating them as fully aligned evidence.

## Bounded validation and resource measurements

Four layers/eight tokens: 27 exact decisions, five accepted near-ties, **zero rejected**. Row-7 logits
distance to decode falls from 8.01857e-5 to 6.26643e-5; relative L2 falls from 0.0126665 to 0.0112788.
The replay takes 24.8 s, peak process RSS 15.5 GiB. The original bounded launch had an older capture
without `trace.spec` and failed before execution; the detached runner regenerated a complete trace
and successfully ran this validation before either full model. The failed launch log is retained.

The CPU lock was acquired at 12:51:50 UTC. Both full paired references were published by 13:51:20 UTC.

| layers | f32 companion | wall seconds | peak process RSS, GiB |
|---|---|---:|---:|
| 18 | prefill | 578.8 | 45.1 |
| 18 | decode | 516.7 | 45.2 |
| 37 | prefill | 1111.9 | 68.2 |
| 37 | decode | 991.3 | 68.2 |

All model runs used the shared CPU-model lock, 16 threads, and the forced 64g Docker memory/no-swap
limit. Process RSS exceeds 64 GiB for L37; cgroup peak was not retained, so do not claim RSS below
the configured limit. GPU work used the existing card/whole-box GPU locks; these are correctness
checks, not performance benchmarks.

Fresh verification: full box build PASS (JOBS=16), four focused host CTests PASS. Oracle host tests:
35 PASS / one optional real-index fixture SKIP (seven route/packaging, 19 reference, nine MTP pass).
The two wrapper tests pass, including overriding inherited 110g with 64g. Negative controls reject
distant swaps, dropped high-score experts, malformed sets, stale/swapped capture identity and invalid staging
formats. Review has no remaining actionable findings.

Completed box logs: `~/b70/logs/q4exp-f32/aligned-{full,r30-host,one-card,r33-pp}.log`.
Published pairs: `~/b70-inference-server/oracle-out-q4exp-f32-aligned-L{18,37}/q4exp_4k.state.safetensors`.

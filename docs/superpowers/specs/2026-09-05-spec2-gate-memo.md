# Spec 2 - the gate, and what it did and did not settle

Date: 2026-09-05. Gate row 1 at sha `e44c40c` - which is `1cb79ae`, the sha the
gate was ordered at, plus one SDD-only commit and this task's one-file harness
fix; **no engine code differs and the kernels last changed at `42921d2`**.
Authority: spec `2026-09-04-spec2-prefill-design.md` incl. its §3.0c amendment;
plan `2026-09-04-plan6e-spec2-gate.md`; ledger
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/progress.md` and its rulings
A20-A28. The rows and their conditions: `gate-preflight.md`; the arithmetic:
`gate-arithmetic.md`, both in that workspace.

**This is the SHORT path.** No tag. A re-gate follows the two GDN tasks the
operator has ruled for (§5), and it is that re-gate - not this memo - that
closes spec 2.

## 1. The verdict

| bar | source | result |
|---|---|---|
| **pp4096 ≥ 1973 t/s** (beat vLLM) | spec §2, phase-1 target | **SHORT - 1377.20 t/s device-side, 69.8%**, −30.2% |
| prefill at **70-79%** of vLLM | spec §3.0c amendment, 2026-09-05 | **MET at the floor - 69.8%** (device 0); 67.4% on device 1 |
| **replace the 121 s ingest** | spec §1, the reason the spec exists | **MET - 2.974 s for 4096 ids, a 40.7× cut** |
| **decode untouched** (≤ 0.09% drift) | spec §6 bar 5 | **MET - RTN 32.21 vs 32.22 (−0.031%), Vishva 29.31 vs 29.33 (−0.068%)** |
| golden / determinism / consistency / launch invariants | spec §6 bars 1, 3, 4, 6 | **MET - 60/60 tests, 774 kernels, 19 modules** |
| **multi-chunk golden gate ≥ 2048 ids** | spec §6 bar 2 | **NOT RUN - skipped, oracle owed** (§6) |

Every row is **iterate grade**: two desktop processes hold DRM fds on both cards
(§6.1). Nothing here is record grade and none of it is claimed to be.

## 2. What the 30% gap actually is, and it is not skill

**The composed ceiling itself sits under vLLM.** That is the "skill or silicon"
answer for prefill, and it is arithmetic rather than opinion. Per chunk at
C = 2048, measured in situ:

```
GEMM 656  +  dequant 205  +  small kernels 131  +  attention 83  =  1075 ms
                                                       before any GDN at all
2048 tokens at vLLM's 1973 t/s allows                     1038 ms
```

**With GDN at exactly zero this design is still 3.6% slower than vLLM.** The
four terms above are not ours to fix: the GEMM is `sycl-tla`'s stock bf16
mainloop running at Intel's rate, and every dequant lever is measured dead
(A23: the int4 mixed-input mainloop is 3× slower on this driver; A24: the device
exposes one compute queue, so overlap is unavailable, and the L2 slab buys
−70 ms only by paying a cross-runtime handoff). vLLM reaches 1973 on a
*different* mainloop - `MainloopIntelXeXMX16MixedPrecision` via
oneDNN/`gemmstone`, which upconverts int4 in-register and never materialises a
bf16 scratch. **Our 205 ms/chunk of dequant is the price of not having that
mainloop**, and inheriting it is a redesign, not a lever.

So: **the implementation is at 98.8% of its own ceiling and the ceiling is at
70% of vLLM.** The gap is the design, and the design was chosen because A21's
premise - that the same 4-bit mainloop family runs at the bf16 rate - was
measured false.

## 3. Per-stage yields, all measured

| stage | what landed | `--pp 4096` | of vLLM |
|---|---|---:|---:|
| L1-engine (`a663f7b`) | first device-side prefill | 978.07 | 50% |
| GDN scan rewrite (`bb5f1d5`, A25/A27) | output tiles, 640 barriers → O(4) | 1304.06 | 66% |
| conv + wu rewrites (`42921d2`, A28) | blocking; mirror-paired output tile | 1375.65 | 70% |
| **this gate (`e44c40c`)** | no engine change | **1377.20** | **69.8%** |

The gate row reproduces `42921d2`'s to **0.11%** on the same device, which is
the point of taking it: the engine did not move, the instrument did not drift,
and the number is real.

GDN went **984.1 → 347.0 ms/chunk** across two pre-registered rewrites (2.84×)
and is **still 22.5× the 15.4 ms Stage 0 priced it at**. Five of the six ledger
terms came in at or under their Stage-0 price; GDN is the only miss, and A25
records that the controller had labelled that projection "measured".

## 4. Predictions that died here

1. **"92% of vLLM, every term measured" (A24).** It carried one term that was
   never measured. Retired by A25; the ceiling it rested on never existed.
2. **The int4 mixed-input route (A21).** Measured 3.4× slower than the bf16
   scratch it was meant to replace (A23). Plan 6f's 2220-2234 t/s went with it.
3. **`--pp-chunk 4096`.** Refused by the build - `PrefillScratch::kC` is 2048
   (A13) - and the sweep says the flag would not have helped anyway: 512 → 899,
   1024 → 1185, 2048 → 1331 t/s, monotone. Wider chunks, not narrower, are where
   the fixed 411 ms/chunk gets amortised, and 2048 is the widest that fits.
4. **"The two cards are identical."** Falsified in this gate, and it is a new
   finding: same binary, same checkpoint, same ten minutes, **prefill measures
   1377.20 on device 0 and 1330.82 on device 1 (−3.37%)** while **decode measures
   32.37 and 32.31 (−0.19%)**. A compute-throughput difference between two
   nominally identical B70s, visible only to the compute-bound workload.
   Unattributed: neither card drives a display, both report identical
   unprivileged PCIe link fields, and the `xe` frequency sysfs needs root.
5. **The byte-matched caveat transfers from decode to prefill.** It does not.
   Prefill runs `lm_head` once per prompt, so RTN and Vishva measure **0.17%
   apart** on pp (1377.20 / 1374.87) against **9.9% apart** on tg (32.21 /
   29.31, the control rows). The checkpoint asymmetry that decides the decode
   comparison is immaterial here.

## 5. What is left, and what is queued behind it

**Two tasks are queued, by operator ruling of 2026-09-05**, and they are the
reason this memo does not close the spec:

- **(b) `gdn-a2a-simd32-brief.md`** - `pf_gdn_wu` at SIMD32 (a one-line
  attribute the wu task copied from the scan and derived under its own bar,
  −45 ms/4096) plus `pf_gdn_A2` and `pf_gdn_A` output tiles (−70 ms/4096, the
  same mechanism and the same measured 0.79 TFLOP/s wu had before its rewrite).
  **−115 ms/4096 derived → ~1430 t/s.**
- **(c) `gdn-scan-dpas-brief.md`** - a DPAS scan. `pf_gdn_scan` is 3.40
  ms/layer/chunk and 11% of the walk; the rewrite task's own verdict is that
  under 1.5 ms needs DPAS, not another vector mapping.

Together, with the rest of the priced menu, **≈ 1500-1550 t/s = 76-79%**
(derived). **Nothing reaches 1973** - §2's arithmetic holds independently of how
far GDN is driven. A **re-gate on a strict-idle box follows (b) and (c)**, and
spec 3 (tokenizer + HTTP) follows that.

Priced and *not* queued: `pf_gdn_solve` is sequential by algorithm (0.18
TFLOP/s, 0.75 ms/layer/chunk) - a different problem from the mapping ones. The
attention and GEMM re-tunings remain post-spec-3 candidates. The one genuinely
unexplored idea is a device-side L0↔SYCL dependency that removes the host wait
at every boundary (A24's remainder); it is worth a probe only if the operator
wants the last few percent of the ceiling, and the ceiling is 70%.

## 6. Owed, named, not silently dropped

1. **The long-prompt oracle.** Spec §6 bar 2 - the multi-chunk golden gate at
   C = 1024 over a ≥ 2048-id prompt - **has never run**. `long.ids` (2820 ids)
   is staged and `prefill_gate_long_test` skips cleanly, which is exactly what
   the suite reported here (60 tests, 60 pass, 1 skip). The dump is a
   `docker run` and it is the **operator's** to start
   (`tools/oracle/README.md`, "long prompt"; 20-40 min, ~70 GiB RSS estimated).
   **A hard correctness bar of this spec is unmet, and no gate row substitutes
   for it.**
2. **The strict-idle re-gate.** Every row in this memo is iterate grade.
   `baobab` (285644) and `ptyxis` (285678) **re-acquire** render-node fds
   periodically - the fds present during this gate were opened at 19:58:59,
   three minutes after the box was verified clean at 19:55 - so a strict-idle
   window cannot simply be waited for; the processes have to be closed. They
   hold zero VRAM and have submitted zero engine cycles, so the *numbers* are
   very likely unaffected (the 0.21-0.45% spreads say the instrument is quiet),
   but the *grade* is not claimable. The re-gate after (b) and (c) carries this.
3. **The device asymmetry** (§4.4) is measured and unattributed. Every historical
   `bench_decode` row in docs/BENCHMARKS was taken on device 0 because the
   harness dropped `ZE_AFFINITY_MASK` until `e44c40c`; the probe rows in this
   spec's docs that claim `ZE_AFFINITY_MASK=1` in their headers **ran on device 0
   too** where they went through `tools/bench_decode.sh`. Direct probe binaries
   read the variable themselves and are unaffected.
4. **The one-time prefill warm-up** (~170 ms, derived at `a663f7b`) has not been
   re-measured since the GDN rewrites; the gate row sits 1.2% above the profiled
   per-chunk composition and that is where the difference is presumed to live.
5. `docs/07-open-questions.md`'s driver/IGC entry - the box runs 26.27 / IGC 2.38
   against `sycl-tla`'s validated 26.01 / IGC 2.27 - remains unpriced.

## 7. Ruling requested

Spec 2 does **not** close here. This memo records **gate row 1**: the spec's
headline deliverable (a 40.7× cut in prefill, 121 s → 2.974 s) is met and
carried; the §2 bar is short by 30.2% for a reason that is arithmetic and
recorded; §3.0c's band is met at its floor. The sequence the operator ruled
stands - **(b), then (c), then a strict-idle re-gate**, and the long-prompt
oracle whenever the operator starts it. No tag.

## 8. Closed, 2026-09-14 - the operator ruled "run the spec 2 golden gates and tag it"

Everything §7 queued has run:

| item | outcome |
|---|---|
| (b) GDN wu SIMD32 + A2/A tiles | A2 1.75×, A 1.87× adopted; the SIMD32 lever measured 1.24× slower and died (A29) |
| (c) DPAS scan | hit its time bar at 4.0×, flipped one determined golden token, reverted by its own rule (A30) |
| strict-idle re-gate | **`pp4096` 1406.18 t/s, RECORD grade**, 2026-09-09 at `977a31c` - the first record-grade row this project took |
| golden gates | **93/93 + 93/93 determined rows exact** on `urakozz@84575a1`, 2026-09-14 at `a6ce30e` |

Two things changed underneath the gate and both are recorded where they
happened: the gate checkpoint became `urakozz@84575a1` after both earlier
checkpoints were deleted from the box (A31), and the int4-`lm_head` decode
comparison was retired, since vLLM would also be faster with one (A33).

**Verdict.** The §2 bar - beat vLLM's 1973 - is **not met: 71.3%**, and §3.0c
had already shown it unreachable by this design (the non-GDN terms alone exceed
the chunk budget 1973 t/s allows). The deliverable that is met is the one the
spec set out to build: a real device-side prefill, **41.5× the decode-replay
ingest it replaced** (121 s → 2.913 s per 4096 ids), correct against the CPU
oracle on both code paths.

**Owed, and not claimed by the tag:** §6 bar 2, the multi-chunk golden gate over
`long.ids` at C = 1024. It needs a `PROMPTS=long` oracle dump for `84575a1` and
its test registration re-pointed off the deleted RTN checkpoint.

Tag `spec2-done` is placed on the commit that records the golden gates.

# Docs

Reference material for an LLM inference engine written from scratch for the
Intel Arc Pro B70, against Level Zero and OpenCL C. The project README has the
short version; this is where the reasoning and the evidence live.

## Start here

- **[08-decode-vs-prefill.md](08-decode-vs-prefill.md)** - why decode and
  prefill want opposite kernels. If you read one file, read this one; everything
  else follows from it.
- **[01-hardware.md](01-hardware.md)** - what this card actually does, measured:
  bandwidth, the DPAS rate by data type, and what is simply not available on it.
- **[BENCHMARKS.md](BENCHMARKS.md)** - every measured row, how it was taken and
  what grade it earned.
- **[coding-benchmarks.md](coding-benchmarks.md)** - the published coding-agent scores
  (SWE-bench, Terminal-Bench) of the models the engine supports or plans, and the models
  considered and not supported.
- **[18-running-models.md](18-running-models.md)** - the commands to download, serve,
  decode and benchmark each model (Qwen3.8, Ornith 1.5, K2-Horizon, Kolibri-1, Nemotron 3.5 Bolt, Flash-Next),
  what each supports, its limits, the speculative-decoding modes and how their output is judged.
- **[environment-variables.md](environment-variables.md)** - what the CLIs, tests and tools read from the
  environment (none of it needed to run the engine).

## The design

| | |
|---|---|
| [02-formats.md](02-formats.md) | why int4 W4A16, and what the alternatives cost |
| [03-models.md](03-models.md) | the target model: shapes, layer math, byte accounting |
| [04-architecture.md](04-architecture.md) | the captured decode list, the prefill walk, the server |
| [05-perf-model.md](05-perf-model.md) | where the time goes, which levers are left, what was measured and rejected |
| [13-loader.md](13-loader.md) | checkpoint to device buffers, and every assert on the way |
| [11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md) | BPE, Jinja and UTF-8-safe streaming detokenisation |

## The kernels

| | |
|---|---|
| [12-kernels.md](12-kernels.md) | every kernel, its layout decisions and its measured rate |
| [15-step-anatomy.md](15-step-anatomy.md) | one decode step, per launch, in situ |

## Correctness

| | |
|---|---|
| [14-golden-gate.md](14-golden-gate.md) | the trust chain: engine against a CPU oracle against vLLM, and why tokens gate while tensors only diagnose |

## Context and what is unresolved

| | |
|---|---|
| [06-prior-art.md](06-prior-art.md) | what to borrow from sycl-tla, oneDNN, OpenVINO and vLLM, and what not to |
| [09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md) | months of patching the wrong layer, and the result |
| [07-open-questions.md](07-open-questions.md) | what is genuinely not known |
| [16-know-how.md](16-know-how.md) | the transferable findings, condensed |
| [17-int8-prefill.md](17-int8-prefill.md) | how 2104.50 t/s prefill is computed: the int8 path on the W4A16 file, and the 12 things rejected on the way |
| [10-the-box.md](10-the-box.md) | the box is the check: box_check, the GPU lock, the toolchain, the two streams, the idle protocol and why an idle box matters for a benchmark |

## Probes (evidence behind decisions)

Dated, self-contained records of one experiment each, in [probes/](probes/). They are
evidence, not narrative, and the numbered guides cite them where they matter.

- [probe-prefix-cache-2026-09-27.md](probes/probe-prefix-cache-2026-09-27.md) - prefix
  caching (spec 7): pinned host copies at 12-14 GB/s, the tail floor at depth, the
  write-through cost (S3), S1/S2 at 60k and the replayed opencode-shaped session
- [probe-dpas-rates-2026-09-22.md](probes/probe-dpas-rates-2026-09-22.md) - DPAS rate
  by data type, with the assembly that proves the loops are DPAS bound
- [probe-fused-dequant-2026-09-22.md](probes/probe-fused-dequant-2026-09-22.md) -
  fusing int4 dequant into the GEMM: bitwise correct, slower, **rejected**
- [probe-flash-attn-2026-09-25.md](probes/probe-flash-attn-2026-09-25.md) - the
  fused flash-attention probe (spec 6 P0/P1): the attention baselines, the
  composed path against fp64, and the tile sweep whose winner,
  `pfa_KT64_R16_H6_Q0`, is the production `pf_flash_attn`
- [probe-mtp-2026-09-27.md](probes/probe-mtp-2026-09-27.md) - spec 8 P0: the MTP
  head's reference wiring (post-norm hidden, embed first), draft acceptance by
  depth, the verify step at M = 1..4 (1.17 / 1.52 / 1.74 steps), K = 3
  provisional and the `Control` index for commit; opencode rows pending the log
- [probe-lm-head-2026-09-28.md](probes/probe-lm-head-2026-09-28.md) - spec 9 P0: an int8
  per-row `lm_head` against bf16 on the CPU. Cosine and argmax pass. The top-20 set
  (92.0 %) and the filtered KL mean fail as written, and bf16 output rounding fails
  them too; the head stays bf16 pending the operator's ruling. Host quantisation
  takes 3.3 s
- [probe-int8-kv-2026-09-28.md](probes/probe-int8-kv-2026-09-28.md) - spec 12a, **Agnes
  stand-in** on the Mac: int8 KV schemes against bf16 KV, by attention replay (1k-32k
  tiled) and end to end. Not a stop. Per-token K fails on outlier channels; KIVI and
  rotated K tie; rotating V too (`rotkv`) is best and below the bf16 eager path's own
  error. Proposed `rotkv` (decide), Q3 tolerances, and the Qwen3.8 repeats for the box
- [probe-dflash-2026-10-08.md](probes/probe-dflash-2026-10-08.md) - spec 19a P0 on the Mac CPU:
  DFlash2's acceptance on Qwen3.8 (A4 tool calls 7.21 tokens per verify at K = 7, code 6.15,
  prose 2.2); int8 / 4-bit drafters cost nothing; a ranked 32k draft vocabulary too (in-sample ranking: an upper bound). Provisional
  go on code and agentic output (+54-65 % over `--mtp auto`, derived); verify at M = 5..8 is
  the box's to measure (plan 19a Task 4).
- [probe-batching-2026-10-23.md](probes/probe-batching-2026-10-23.md) - spec 13a: rows of different sequences in one
  decode step - the audit (six per-sequence kernels, file:line), the `B70_BATCH_PROBE` row builds and
  `probe_batch_step` (bitwise on the Mac for gdn_step / attn_prep; the box's check PENDING), the step and chunk
  tables (derived; measured PENDING - queue row 66), the layouts' memory from the code (Qwen3.8 4 slots 1.94 GB
  spare; Agnes 2 with MTP 1.76, 4 with `--mtp off` 1.22), the row table for 13c.
- [probe-small-m-gemm-2026-11-01.md](probes/probe-small-m-gemm-2026-11-01.md) - spec 28a: Qwen3.8's decode linears at
  M = 1..16 through gemv and small-M DPAS stand-ins (`probe_small_m`, 347 binaries), the verify step at M = 1..8 split
  into GEMV and non-GEMV (`probe_verify_m --verify8`), numerics and property R on layer 62 / 63's weights. Busy-box
  timing on one card (provisional): bf16 weights-on-B at M = 8 is 1.08x a token's M = 1 GEMV sum (today 3.10x);
  verify(8) today 3.17 plain steps, derived with the stand-in 1.54-1.58 - the operator's call; attention at M = 8 is the
  next lever (spec 28c). Quiet-box re-check set in §6.
- spec 28b (no separate record; spec 28 §10 "28b as built" and BENCHMARKS "The verify step on gemm_sm"): `gemm_sm`
  in production for every M >= 2 decode list (Qwen3.8, Ornith's dense projections), M = 1 bitwise; verify(8)
  1.57 plain steps (gemv 3.17), D1 tool calls 2.12x at K = 3 (busy-box timing); N4 / N6 as written not met - the
  operator's ruling pending.
- [probe-tp-2026-11-01.md](probes/probe-tp-2026-11-01.md) - spec 17a: tensor parallel over the two cards, Qwen3.8 first
  (`probe_tp`): links Gen3 x16 both, 32 GB BARs, peer access; the CPU into VRAM through a dma-buf BAR mapping; the
  all-reduce on every route - p2p-push 3.51 us a call / 0.45 ms a token at M = 1, bitwise over 10^4 replays; host-bar
  bitwise but 31.6-52.9 ms a token; p2p-pull and host-polling routes fail (a card's poll of memory it does not own is
  cached); the per-card GEMVs 0.550 of one card's. Quiet-box re-check 2026-11-06 (§8): 0.47 ms a token, one card
  33.42 t/s, `--parallel pp` 34.59, derived plain TP 56.3 t/s (1.68x one card, 1.63x pp) - GO.
- [probe-qwen4exp-2026-10-09.md](probes/probe-qwen4exp-2026-10-09.md) - spec 21a: Qwen3.8-Flash-Next (`qwen4_exp`)
  from its small files, headers and transformers 5.19.0 / vLLM source - the rounding chain, both checkpoints,
  the PLE layout and hash constants (= the formula), the MTP head's wiring (decision 5's reuse is vLLM's
  opt-in), the tokenizer (the pre-tokenizer adds `\p{M}`), torch's tie rule, the derived bytes.
- [probe-bolt-2026-10-21.md](probes/probe-bolt-2026-10-21.md) - spec 25a: Nemotron 3.5 Bolt 74B-A9B - the rounding chain
  cited into the remote code (63 anchors), both checkpoints from their headers (the third-party g128 export's 112,940 tensors - served since 2026-10-29: our g64 export -, the BF16
  original's `backbone` names and 74,042,720,768 parameters), signed / subnormal scales, the dt-clamp census (212 of
  5,120 heads), tokens and the template (14 renders in `tests/server/bolt_template_cases.json`), torch's tie order and
  fp32-GEMM row-count dependence, B1's bounds, the bf16 table, and R's fallback state-passing bug past 128 rows.
- [probe-lilicorr-2026-10-23.md](probes/probe-lilicorr-2026-10-23.md) - spec 26a: NVIDIA's LiLiCorr draft model on our Bolt
  (dropped 2026-10-27, restored 2026-11-01) - its shapes and NVFP4 weights read on the Mac, the derived verify and block
  costs, the expected acceptance from SPEED-Bench; the box's numbers are queue row 63.
- [probe-eagle3-k2-2026-10-08.md](probes/probe-eagle3-k2-2026-10-08.md) - EAGLE-3 on K2-Horizon (Mac CPU P0):
  **no-go** - the community drafter is undertrained (golden: 0.23 / 0.06 acceptance, below plain decode at
  every K); prompt lookup wins (to 1.68x at 4k). K2 verify costs 1.26x / 1.63x a plain step at M = 2 / 4.
- [probe-int8-kv-qwen38-2026-10-06.md](probes/probe-int8-kv-qwen38-2026-10-06.md) - the 12a
  repeat on **Qwen3.8** (Mac CPU, plan 12b Task A): `rotkv` holds (golden cos mean
  0.99995-0.99997, above the stop rule; replay flat to 32k, below bf16 eager's own error).
  Sets 12b's tolerances: Q2 1e-4 confirmed; Q3 2e-3 per depth and 5e-4 averaged.
- [probe-agnes-2026-10-03.md](probes/probe-agnes-2026-10-03.md) - spec 14 phase 1 (Agnes dropped
  2026-11-07; kept as the record), on the Mac: Agnes 3.0 Flash from its own files (72 layers, W 18.344 GB from the headers), the
  CPU reference, the parallel-FFN fold (bit-exact on the weights, C++ byte-identical to
  Python, real layers 0/3/35/71 on real activations: max |d| 0), the reference bit-identical
  to `modeling_agnes.py`, the golden and MTP dumps made layer-streamed in Docker, the
  chat template identical to Qwen3.8's; what is pending for the box
- [probe-verify-attn-2026-11-03.md](probes/probe-verify-attn-2026-11-03.md) - spec
  28c: the verify step's attention on DPAS (attn_verify_dpas) against v2 at M = 2..8 x
  depth 512..130816, bf16 and int8 KV - the levers, the table the rule is built from,
  numerics, the in-list and end-to-end speed (busy box)
- [dflash-vs-sly-2026-11-11.md](probes/dflash-vs-sly-2026-11-11.md) - spec 19 v2's
  starting point: our DFlash against vllm-sly-radiance and saber-engine (same drafter, same
  8-row block), from code and docs - the gap was the cost of a round, not tokens per round
- [probe-decode-attn-2026-09-28.md](probes/probe-decode-attn-2026-09-28.md) - spec
  10a: decode attention at depth. P0 (69% of the step at 128k, 149 GB/s), the
  lever sweep at M = 1 and 4 on real KV, and the v2 design: device-derived
  stride, block-read V, m-inner single-barrier waves, exp2; 4.4x at 128k,
  derived 28.3 / 25.8 / 21.7 t/s against the A-F3 bars
- [probe-w4a8-2026-09-23.md](probes/probe-w4a8-2026-09-23.md) - W4A8 on the mixed
  4-by-8 DPAS: 0.965x the control and 2.79% relative error, **rejected**; the
  group-size sweep in §13 shows why, and that per-channel scales would reach
  1.752x while every group size a real int4 checkpoint uses does not
- The September 2026 prefill investigations (the matched vLLM comparison, our own GEMM,
  oneDNN's int4 matmul, Intel's delta net kernel, replay, continuation, the split-BF16
  scan, the barrier-free solve, the flash-attention levers) are folded into
  [08-decode-vs-prefill.md](08-decode-vs-prefill.md#how-prefill-got-here) and
  [12-kernels.md](12-kernels.md#how-the-prefill-kernels-got-here); their standalone
  write-ups were deleted when the work closed.

## Designs

[superpowers/specs/](superpowers/specs/) holds the design documents behind the
larger changes: the decode core, the prefill path, the tokenizer and HTTP layer,
the Level Zero prefill GEMM, and the prefill parity programme.

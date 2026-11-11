# Coding benchmarks of the models we run or considered

Published scores of the source models (from their model cards), not runs of this engine and not of our int4
re-quants. A dash means no published result was found, not zero. Scores from different harnesses, budgets or
benchmark versions are not strictly comparable. `A` = parameters active per token (MoE).

| Model | Parameters / architecture | SWE-bench Verified | SWE-bench Pro | Terminal-Bench 2.1 | In this engine |
|---|---|---:|---:|---:|---|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense | — | 61.7% | 73.0% (Terminus) | **focus**: one card, TP, replica, MTP |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 125B-A6B MoE (+ 51B n-gram embeddings) | — | 62.5% | — | **next**: being finished (expert tiering over two cards) |
| [Ornith 1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | 35B-A3B MoE | 79.0% | 59.6% | 67.8% (Terminus-2), 68.5% (Claude Code) | supported |
| [Nemotron 3.5 Bolt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16) | 74B-A9B hybrid Mamba-2 / Transformer LatentMoE | — | — | 53.93% (bf16; 50.98% IQ2) | built, **parked** until Flash-Next is done |
| [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) | 78B-A3.46B MoE | 66.4% | — | 27.7% (Harbor) | built, parked |
| Recursive Flint | 39.8B-A2.5B MoE (sliding-window + NoPE attention, 256 experts top-6 + shared) | 74.2% (mini-swe-agent) | — | 54.9% (Terminus 2) | watched; design written, parked ([spec 27](superpowers/specs/2026-11-01-spec27-flint-design.md)) |
| [Xing4.0](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) | 29B-A4B MoE (MLA + Sinkhorn residual mixing + MTP) | 75.0% (SWE-agent) | — | 57.5% (Terminus-2) | not supported: below Ornith with more active parameters, needs three new families |
| [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 30B dense | 76.0% | 51.2% | 51.7% | dropped 2026-10-27: below Ornith / Qwen3.8 |
| [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) | 24B-A2.45B MoE | 51.0% (OpenCode) | — | — | dropped 2026-10-27: below Ornith / Qwen3.8 |

This engine has not run these coding benchmarks itself; [BENCHMARKS.md](BENCHMARKS.md) measures speed, not task
success. An engine-side score would need a fixed harness, the exact quantized checkpoint, the inference settings
and the budget recorded with it.

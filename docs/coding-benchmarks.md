# Supported models: coding benchmark snapshot

This table covers the three checkpoint architectures accepted by
`src/model/model_desc.cc` (`desc_for_architecture`), plus Nemotron 3.5 Bolt,
which is planned for support. Other model families in the repository use
separate engine implementations and are outside this descriptor table. Parameter notation
uses total parameters and (for MoE) active parameters per token, e.g.
`35B-A3B`. Model benchmark scores describe the published source model, not the
engine's int4 checkpoints or a run performed by this repository.

| Engine model | Parameters / architecture | SWE-bench Verified | SWE-bench Pro | Terminal-Bench 2.1 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 27B dense | — | 61.7% | 73.0% |
| Agnes 3.0 Flash | 33B dense | — | — | — |
| Ornith 1.5 35B-A3B | 35B-A3B MoE | 79.0% | 59.6% | 67.8% (Terminus-2); 68.5% (Claude Code) |
| Kolibri-1 | 78B-A3.46B MoE | 66.4% | — | 27.7% |
| Qwen3.8-Flash-Next | 125B-A6B MoE (plus 51B n-gram embeddings) | — | 62.5% | — |
| Nemotron 3.5 Bolt *(planned)* | 74B-A9B hybrid Mamba-2/Transformer LatentMoE | — | — | 53.93% (BF16) |

## Watched candidates

| Model | Parameters / architecture | SWE-bench Verified | SWE-bench Pro | Terminal-Bench 2.1 | Status |
|---|---:|---:|---:|---:|---|
| Recursive Flint | 39.8B-A2.5B MoE (35 sliding-window + 14 NoPE attention layers, 256 experts top-6 + shared) | 74.2% (mini-swe-agent) | — (its "Pro v2 89.7" is a different benchmark version) | 54.9% (Terminus 2, web-retrieved counted as failures) | Watched (operator, 2026-10-23): decided against LFM2.5 on our A4 numbers; transformers support merged on main 2026-10-21 (not yet released). `docs/superpowers/specs/2026-10-23-spec27-flint-watch.md` |

## Considered, not supported

| Model | Parameters / architecture | SWE-bench Verified | SWE-bench Pro | Terminal-Bench 2.1 | Why not |
|---|---:|---:|---:|---:|---|
| Muse-Glimmer-30B | 30B dense | 76.0% | 51.2% | 51.7% | dropped 2026-10-27: quality below Ornith / Qwen3.8 |
| LFM2.5-24B-A2B | 24B-A2.45B MoE | 51.0% (OpenCode) | — | — | dropped 2026-10-27: quality below Ornith / Qwen3.8 |
| Xing4.0-29B-A4B | 29B-A4B MoE (MLA + mHC + MTP) | 75.0% | — | 57.5% (Terminus-2) | Below Ornith 1.5 35B-A3B on both reported coding scores with more active parameters; needs three new families (MLA attention and cache, Sinkhorn mHC residual mixing, a SentencePiece-only tokenizer) |

## Sources and interpretation

- Qwen3.8 model card: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card reports SWE-bench Pro 61.7 and Terminal-Bench 2.1 (Terminus) 73.0. Both use the model card's reported setup.
- Ornith model card: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). It reports SWE-bench Verified 79, SWE-bench Pro 59.6, Terminal-Bench 2.1 Terminus-2 67.8, and Terminal-Bench 2.1 Claude Code 68.5.
- Muse-Glimmer model card: [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B). Meta reports SWE-bench Verified 76.0, SWE-bench Pro 51.2, and Terminal-Bench 2.1 51.7.
- Kolibri-1 model card: [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1). Aleph Alpha reports SWE-bench Verified 66.4 and TerminalBench 2.1 27.7 using Harbor; its card documents 3–4B active parameters and the model has about 78B total / 3.46B active parameters.
- Qwen3.8-Flash-Next model card: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Qwen reports 125B total / 6B active parameters (plus 51B n-gram embeddings) and SWE-bench Pro 62.5. No Terminal-Bench result is listed here.
- LFM2.5 model card: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B). It reports 24B total / 2.45B active parameters and SWE-bench Verified (OpenCode) 51.0.
- Nemotron 3.5 Bolt model card: [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16). NVIDIA reports 74B total / 9B active parameters and Terminal-Bench 2.1 53.93 for BF16 (the 24GB IQ2 variant reports 50.98). SWE-bench results are not listed in that model card.
- Xing4.0 model card: [XingChen-AGI/Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B). It reports 29B total / 4B active parameters, SWE-bench Verified 75.0 (SWE-agent, 210K context) and Terminal-Bench 2.1 57.5 (terminus-2, mean of 3 runs). Its `config.json` is DeepSeek-V3-style MLA (`kv_lora_rank` 512, `q_lora_rank` 768, rope head 64) with manifold-constrained hyper-connections (`hc_mult` 4, `hc_sinkhorn_iters` 20) and one MTP layer; the repository ships `tokenizer.model` and no `tokenizer.json`. Revisit if a later release passes Ornith or Qwen3.8 on coding, or if MLA arrives with another model worth supporting.
- A dash means no result located for that exact source model and benchmark during this research; it does not mean zero. The two Terminal-Bench Ornith scores use different agent harnesses and are shown separately. Published scores across models are not strictly comparable when harness, sampling, budget, or benchmark version differs.
- Parameter labels follow the upstream model names and model cards; `A` denotes active parameters per token. The descriptor source records tensor shapes and architecture identifiers, not authoritative total parameter counts. Qwen3.8, Agnes, and Ornith are the three descriptors in `src/model/model_desc.cc`; the other supported families have separate engine implementations. Nemotron is planned and is not yet accepted by `desc_for_architecture`.

## Engine benchmark status

These coding-agent benchmarks have not been run by this engine. Existing engine
measurements in `docs/BENCHMARKS.md` measure inference throughput and are not
SWE-bench or Terminal-Bench task success rates. A valid engine-side score would
need a fixed harness, exact quantized checkpoint, inference settings, and
budget recorded alongside the result.

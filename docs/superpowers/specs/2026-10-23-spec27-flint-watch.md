# Spec 27 - Recursive Flint (`flint`): a watched candidate

**Status (2026-10-23): WATCHED - nothing planned or built.** The operator's ruling (2026-10-23, option 1): Flint is
recorded as a candidate beside LFM2.5 (spec 24), which stays; the decision between them is made on OUR numbers - the
A4 tool-call set (plus a small SWE-style set) run through b70-serve on the card for both - not on the model cards.
A full spec (facts, CPU reference, quant, descriptor / loader, decode, prefill, serving - the spec 24 shape) is written
when the trigger below holds and the operator gives the go.

Every number is **published** (Recursive's model card / config), **derived** (computed from the config) or
**estimated**; nothing here is measured.

## 1. The model (published unless marked)

- `RecursiveSI/flint` - `FlintForCausalLM`, `model_type: flint`; Apache 2.0; BF16 only (~80 GB); config written with
  transformers 5.12.1.
- 39.8B total, ~2.5B active per token including the output head (~2.2B body); 262,144 context; vocab 150,144;
  hidden 2048.
- 49 layers: 2 dense, then 47 MoE - 256 routed experts (expert width 512), top-6, plus an always-active shared expert.
- Attention: 35 sliding-window layers (513 tokens: the current one and 512 before it, RoPE) and 14 full-attention layers
  without position encoding (NoPE); GQA 32 q / 4 kv heads, head dim 128; per-head q / k RMSNorm; learned per-head sinks.
- No MTP / draft head published. Tool calls OpenAI-compatible (`<tool_call>name<arg_key>...<arg_value>`, K2's form -
  to verify against the template). Sampling defaults: T 1.0, top-p 0.95, top-k 20; reasoning kept in history.
- Scores (Recursive's; pass@1 averaged over >= 5 attempts): SWE-bench Verified 74.2 (mini-swe-agent), Terminal-Bench
  2.1 54.9 (Terminus 2; web-retrieved solutions counted as failures). "SWE-bench Pro v2 89.7" is a newer benchmark
  version, not comparable with the SWE-bench Pro column of docs/coding-benchmarks.md.

## 2. What it would cost here (derived / estimated)

- **Bytes per token:** ~1.2 GB body at int4 g64 + the output head (150,144 x 2048 = 0.31B params; ~0.31 GB at int8)
  = ~1.3-1.5 GB - about LFM2.5's 1.43 GB, ~15-20 % below Ornith's.
- **KV:** only the 14 full-attention layers grow with context: 14 x 4 x 128 x 2 x 2 B = 28 KiB a position at bf16
  (~7.5 GB at 262,144); the 35 sliding layers hold 513 rows each. Cheap long contexts and cheap batching slots (spec 13).
- **Size:** ~22 GB at int4 g64 (estimated) - one card.
- **Parts that exist:** sliding window + sinks (Muse, spec 23), per-head q / k norms (Qwen3.8), top-k routing with a
  shared expert (Kolibri top-6 / Ornith), the tool-call parser (K2). New: the combination, a 2-dense-then-MoE layout,
  our own AutoRound g64 quant of the BF16 checkpoint (the one-format rule), its CPU reference.

## 3. The trigger and the pin

- **Reference code exists since 2026-10-21:** huggingface/transformers PR #49824 ("Add Flint model support",
  `models/flint/modeling_flint.py`, configuration, modular file, tests) merged at **`0ea108daf169dd66a1800e48a56d2251d4ff76bd`**,
  after v5.20.2 (2026-10-19) - expected in the next minor release (v5.21, estimated). A CPU reference can pin that
  commit now, or the release when it ships (our other references pin 5.19.0).
- **Trigger for writing the full spec:** the operator's go, after the A4 comparison basis exists - LFM2.5's A4 run on
  the card (spec 24's rows; it needs 24q's AutoRound export, queue row 49, then rows 50-53, and the bf16 original's A4
  reference on the box CPU, `lfm2moe_oracle.sh a4`).

## 4. Against the MoE models already in the engine (published)

| model | total / active | SWE-bench Verified | Terminal-Bench 2.1 | one card |
|---|---|---|---|---|
| Ornith 1.5 35B-A3B | 35B / 3B | 79.0 | 67.8 (Terminus-2) | yes |
| **Flint** | 39.8B / 2.5B | 74.2 | 54.9 (Terminus 2, strict) | yes (est.) |
| Kolibri-1 | 78B / 3.46B | 66.4 | 27.7 | no |
| LFM2.5-24B-A2B | 24B / 2.45B | 51.0 (OpenCode) | - | yes |
| Nemotron 3.5 Bolt | 74B / 9B | - | 53.93 (BF16) | no |

LFM2.5 was chosen (spec 24, 2026-10-15) for its published agentic tool use (LiveClaw 48.09 vs Qwen3.6-35B-A3B's 38.19),
OpenCode-harness RL, the lowest bytes per token, and an existing transformers reference - not for SWE-bench. Flint
publishes no tool-use benchmark; hence the decision on our own A4 numbers.

# EAGLE-3 on K2-Horizon, P0 (Mac CPU), 2026-10-08 - STUB: semantics pinned, tools built

**Status.** Step 1 (the drafter's semantics, pinned from source) and the tools are done and
tested on tiny weights; **no result yet**. The run needs the drafter's `model.safetensors`
(1.67 GB) and the K2 int4 checkpoint (22.2 GB) in the HF cache - the operator is downloading
both - and K2's A4 reference continuations (`tools/toolcall/a4_ref.sh k2 ref`, hours). The
results section below is to be filled from `oracle-out-eagle3-k2/summary.md`.

**The question** (the operator chose "probe first"): is EAGLE-3
([`Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3`](https://huggingface.co/Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3),
community, Apache-2.0, vLLM Speculators 0.8.0, trained against `Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4`)
worth building for K2 in this engine? Decided from **measured** acceptance (teacher-forced,
greedy, against K2's own argmax on our int4 weights) and a **derived** verify cost (bytes read
by a verify of M rows, from the recorded MoE / MoVA routes), against what K2 can already do
(spec 19e's prompt lookup, replayed on the same anchors). No engine work before it.

**A derived prior** (`tools/oracle/eagle3_cost.py model`, before any route is recorded): K2's
plain step reads 3.145 GB (int8 head). If consecutive tokens drew their experts
independently, a verify of 2 rows would touch 15.4 of 100 MoE and 7.7 of 64 MoVA experts per
layer and read **1.41x** a plain step's bytes (M = 3: 1.78x, M = 4: 2.12x); the drafter costs
0.045-0.11 plain steps per draft (int4h - bf16). With the model card's per-position acceptance
0.44 / 0.17 (E_1 = 1.44, E_2 = 1.61), K = 1 would break even and K = 2 lose at depth 0. What
decides it is how much consecutive tokens **share** experts (the `cost` step measures that
from the routes) and how much better than UltraChat the agentic text accepts.

## Pre-run facts (MEASURED from the drafter's files, 2026-10-08)

- **The checkpoint** (`model.safetensors`, 1,669,096,896 B): 17 tensors, exactly the expected
  set (`eagle3_ref.py facts`: missing none, extra none), all bf16 except `d2t` I64 [32768] and
  `t2d` BOOL [250624]; `embed_tokens` shipped, sha256 `f198e179...a1f73e`.
- **`d2t`:** draft + d2t[draft] is strictly increasing and equals t2d's 32768 ids (draft 0 ->
  target 2, last -> 250030): the offset reading holds on the real file.
- **The draft vocabulary misses K2's agentic markers.** In t2d: `<|ifm|im_start|>` /
  `<|ifm|im_end|>` (250018 / 250019, the EOS) and `<ifm|think>` / `</ifm|think>` (250029 / 30) -
  but **not** the tool-call markers (250043 / 44, 250054-250061: `<ifm|tool_calls>`,
  `<ifm|tool_call>`, `<ifm|arg_key>`, `<ifm|arg_value>` and their closers), **not**
  `<ifm|think_fast>` / `<ifm|think_faster>` (250050-250053; the A4 set renders reasoning effort
  low, i.e. `think_faster`), and not BOS 0 / EOS 1. Only 4 of the 624 ids >= 250000 are in it
  (a trace of the UltraChat training data). So on tool-call text **every marker position is a
  guaranteed miss**, and a chain cannot run through one. The summary prints, per corpus, the
  share of the continuation's ids inside the draft vocabulary (a ceiling on a_1).
- **Smoke test** (the real drafter, random taps, 2 CPUs beside another container): every arm
  runs (int8 / int8h / int4h quantise 8 / 9 / 9 linears), propose_batch == propose, every
  draft inside V'; 40-58 anchors/s for the 5-draft chain (int4h 17/s in that run: every arm
  computes in float32, so the spread is the shared CPUs'), 65-420 context rows/s.
- **The embedding against K2's:** pending - K2's `model-00004-of-00005.safetensors` (it holds
  `model.embed_tokens.weight`) is still downloading; `eagle3_ref.py facts --k2` compares the
  sha256 of the two tensors' bytes the moment it is complete.

## Sources read (Review Focus 1)

- **speculators 0.8.0** (the PyPI wheel, unpacked and read in a scratch directory):
  `speculators/models/eagle3/{core,model_definitions,data,attention}.py`,
  `speculators/train/vocab_mapping.py`, `speculators/model.py`; the checkpoint's own
  `config.py` (a copy of `Eagle3SpeculatorConfig`).
- **vLLM v0.28.0** (the plugin's pinned version; single files fetched at the tag):
  `vllm/model_executor/models/{llama_eagle3,interfaces,llama}.py`,
  `vllm/v1/spec_decode/{eagle,llm_base_proposer,extract_hidden_states}.py`,
  `vllm/v1/worker/gpu_model_runner.py`, `vllm/transformers_utils/configs/speculators/algos.py`,
  `vllm/v1/attention/backends/{flash_attn,triton_attn}.py`.
- **github.com/stefanskiasan/k2-horizon-vllm @ 13ee389** (2026-09-06): `k2_horizon_vllm/model.py`
  (the target, `SupportsEagle3`, the aux capture).
- The drafter's `config.json` (vendored: `tools/oracle/third_party/eagle3_k2/config.json`),
  `README.md`, `val_metrics.json`.

## The pinned semantics

| # | what | pinned | source (file:line) |
|---|---|---|---|
| S1 | "aux layer 2 / 24 / 45" | the residual stream at the **input** of target decoder layer a = the **output of layer a - 1**, pre-norm (a = 0 would be the embedding). Our dumps: `aux.{a}` = k2_ref's `resid.L{a-1}`. | plugin `model.py:340-345` (`_maybe_add_hidden_state(aux, 0, h)` before layer 0, `(aux, idx + 1, h)` after layer idx; `self.norm` only after, :345); vLLM `interfaces.py:1499-1509`; `gpu_model_runner.py:5629-5633` adds 1 to DFlash's output-of-layer ids to get this convention |
| S1' | the final hidden state | **training target only** (`verifier_last_hidden_states` -> `verifier_lm_head(verifier_norm(.))`); inference never reads it | `core.py:248-253`; README "plus the final hidden state as the distillation target" |
| S1'' | concat order | layer order, `[L2 | L24 | L45]` (7680) | `gpu_model_runner.py:5326-5328`; the plugin appends while traversing |
| S2 | fc fusion | `fc(input_norm(cat aux))`: ONE RMSNorm over all 7680 (`norm_before_fc`), no bias; `fc_norm` (per-chunk norms) false | `core.py:236-244`; `llama_eagle3.py:150-154, 189-195, 367-379` |
| S3 | the decoder layer | `e = input_layernorm(embed(token))`, `h = hidden_norm(hidden)`; residual = the **normed** h (`norm_before_residual`); `x = cat([e, h])` - **embedding first** - into q [4096, 5120], k / v [1024, 5120]; `h = residual + o_proj(attn)`, `h += mlp(post_attention_layernorm(h))`; `out = norm(h)`. Plain Llama RMSNorm (`w * x_hat`, one group, eps 1e-6), RoPE rotate_half over 128 dims at theta 1e4, GQA 32 / 8, SiLU-GLU 6144, no biases | `model_definitions.py:29-52, 71-110`; `llama_eagle3.py:50, 66-72, 99-119` |
| S4 | what step 2..K feeds back | the **post-norm** `out` (`norm_output` true); logits = `lm_head(out)` | `core.py:302-306`; `llama_eagle3.py:218, 249-254`; `llm_base_proposer.py:603, 722, 760` |
| S5 | row alignment, positions | the row of token x_s carries the aux of position s - 1. Training RoPE position = s (`data.py` shifts, "position_ids now start at 1"); vLLM = s - 1 (rotates the ids, keeps the target positions). Relative RoPE: the same drafts (tested) | `data.py:24-35`; `llm_base_proposer.py:854-864` |
| S6 | draft steps | step 0 = the anchor's row (pending token x[p], aux of p - 1) -> d_0; step j: embed(d_{j-1}) (a **target** id), the fed-back hidden, position + 1 -> d_j; greedy = argmax (first max) | `core.py:269-357`; `llm_base_proposer.py:436-446, 690-770` |
| S7 | the drafter's KV over the context | step 0 over **every** previous position from the **target's** taps: vLLM re-runs the drafter over each verify's rows, so committed positions never hold draft-derived K/V. Causal, sliding window 2048 (keys with q - k <= 2047). Step j attends the step-0 keys of positions <= the anchor plus its own chain. **So the probe runs the drafter over the whole prefix** (its last 2048 positions: the dumps keep aux from `n_prompt - 2049`) | `llm_base_proposer.py:510-603`; `attention.py:16-34, 115-132`; `core.py:216` (one cache across steps); `llama.py:183-217` -> `flash_attn.py:850`, `triton_attn.py:534` |
| S8 | `d2t` | **target id = draft id + d2t[draft id]** (an offset); `t2d` [250624] bool marks the 32768 selected ids, ascending in draft id. vLLM scatters the 32k logits into the full vocabulary (-inf elsewhere): the same argmax, ties to the lower id | `vocab_mapping.py:64-96` (:86 sort, :88-91 offset, :93-94 t2d); `interfaces.py:1465-1469`; `llama_eagle3.py:304-307, 335-357, 386-391` |
| S9 | embedding / head | the drafter ships `embed_tokens` [250624, 2560] (1.283 of the file's 1.669 GB; `embed_requires_grad` false: the verifier's, loaded when uninitialised) and its own 32k `lm_head` (initialised from the verifier's head rows `[t2d]`). vLLM swaps in the target's embedding only when byte-identical, else keeps the drafter's - so the drafter's own is the semantics; `facts --k2` checks identity with OUR checkpoint, `K2_EMBED=1` measures the K2-table variant | `model.py:156-187`; `core.py:68`; `llm_base_proposer.py:1449-1516` |
| S10 | training | 1 layer, 8 epochs, ~4,500 UltraChat conversations, TTT 3 steps; val cond. acc 0.494 / 0.467 / 0.465, full acc 0.494 / 0.230 / 0.108; vLLM per-position acceptance 0.44 / 0.17, mean accept length 1.61 at 2 drafts, 1.17x on an L40S | `val_metrics.json`; the model card |

### Ambiguities, and how the sources resolved them

1. **Layer id: input or output of the layer?** Resolved: the INPUT (the output of layer
   a - 1) - the plugin's capture order and vLLM's explicit +1 for DFlash's ids, which ARE
   output-of-layer (`dflash_ref.py`'s taps). `test_k2_taps.py` holds `aux.{a}` bitwise to the
   plugin's capture emulated on the vendored HF model, and asserts it is NOT layer a's output.
2. **Pre- or post-norm feedback.** Resolved by `norm_output: true`: post-norm, in training
   and in vLLM (the `aux_output` branch).
3. **Positions.** Training and vLLM differ by one absolute position; attention sees only
   differences, so they agree - tested (same drafts, hidden within 1e-6).
4. **The window for draft steps >= 1.** NOT the same in the two sources: training masks the
   step-0 keys relative to the anchor row (`attention.py` builds the mask once and only
   appends diagonals), vLLM relative to each query's own position (p + j). They differ only
   in the j oldest keys of a full window. Both are implemented (`--window query|anchor`); the
   default is vLLM's (`query`, what a served engine would do).
5. **`d2t`: offset or direct id?** Resolved: offset (`vocab_mapping.py`'s comment and
   construction, vLLM's `base + draft_id_to_target_id`). `eagle3_ref.check_vocab_maps` holds
   the real file to it (strictly increasing, = t2d's set) and refuses a direct-id reading;
   `facts` prints the result on the real checkpoint (pending the download).
6. **Does the drafter attend over all previous positions?** Yes (S7), within its 2048
   window - so a probe that only fed the anchor's own row would be wrong.
7. **The training targets' head (an observation, not a blocker).** `verifier_norm` is a
   one-group `LlamaRMSNorm(2560)` loaded with K2's `model.norm.weight`, while K2's final norm
   is GROUPED (2 x 1280, spec 18 §3). If the extraction stored the pre-norm residual (the
   plugin's capture point for index 48 is before `self.norm`), the drafter learned K2's
   logits through a slightly different norm. The dumps record `greedy_flatnorm` (K2's argmax
   through a one-group norm) and the summary prints how often it equals K2's own; a low
   number would explain part of the low published acceptance (and be fixable by retraining).
8. **Verifier mismatch.** Trained on hidden states of a GPTQ-Int4 K2 served by vLLM; our
   engine runs the operator's AutoRound int4 g64 checkpoint. The probe measures on OUR
   weights (k2_ref's reference on the int4 checkpoint, dequantised), which is the number that
   matters here.

## Method (as built)

- **Taps** (`tools/oracle/k2_taps.py`): one `k2_ref.K2Ref` forward per source over
  [prompt + K2's recorded greedy continuation] on the int4 checkpoint (dequantised), mode
  bf16 with fp32 GEMMs (one bf16 rounding of each output; torch's bf16 GEMM is ~6x slower on
  this Mac's AVX2, measured; `--matmul bf16` is bitwise k2_ref). Per source: `aux.{2,24,45}`
  from `n_prompt - 2049`, the pre-norm final hidden, K2's greedy next id, top-16 logits,
  logsumexp, `greedy_flatnorm`, and every continuation position's routed MoE (8) and MoVA
  (4) experts per sparse layer.
- **Acceptance** (`tools/oracle/eagle3_accept.py run`): the drafter (`eagle3_ref.py`) runs
  step 0 over every row, then at every anchor n_prompt <= p <= N - 5 a chain of 5 greedy
  drafts (K = 1..5 are its prefixes); dflash_accept.py's protocol (accepted prefix against
  K2's argmax, censored where the recorded text leaves it). E_K, the per-position acceptance
  a_j (the card's metric), conditional alpha_j, per corpus. Arms bf16 / int8 / int8h / int4h.
- **Lookup** (`eagle3_accept.py lookup`): `tools/spec/lookup_accept.py`'s matcher (= the C++
  one, no candidate cap), min match 3 (b70-serve's default), the same anchors and scoring.
- **Cost** (`tools/oracle/eagle3_cost.py`): the bytes a verify of M = 1..6 rows reads, from the
  distinct experts the M consecutive recorded rows route to, over a plain step's (int8 head,
  int8 KV at depth 0 / 4k / 32k); the drafter's bytes per draft step per arm; S_K =
  E_K / (V(K + 1) + D(K)). Bytes only: launches, the drafter's serial steps and compute at M
  rows are not in it - an upper bound on the gain.
- **Sources**: K2's A4 set (36 scenarios, `tools/toolcall/a4_ref.sh k2 set|ref`: the
  Qwen3.8 conversations re-rendered by K2's template, 512 new ids, reasoning effort low) and
  the three golden prompts with K2 greedy continuations (18a's 32 ids from `oracle-out-k2`
  if present, else 128 made by the driver).

## How to run

```bash
# from the repo root on the Mac, once both downloads are complete
DRY_RUN=1 tools/oracle/eagle3_k2_p0.sh            # the plan + estimates; no container, no model
nohup tools/oracle/eagle3_k2_p0.sh > /dev/null 2>&1 &   # starts the A4 reference (a4-ref-k2) and stops (exit 4)
tools/toolcall/a4_ref.sh k2 status                 # ... until DONE (hours), then:
nohup tools/oracle/eagle3_k2_p0.sh > /dev/null 2>&1 &   # check facts golden sources dump accept lookup cost summary
tail -f oracle-out-eagle3-k2/p0.log
```

A golden-only first look (no A4 reference, ~1 h): `A4=0 nohup tools/oracle/eagle3_k2_p0.sh > /dev/null 2>&1 &`.

**Estimates (ESTIMATED, not measured):** the A4 reference ~1-1.5 days (36 scenarios x up to
512 layer-streamed int4 decode steps at ~7 s, 28 GB container); golden continuations ~15 min
per prompt at 128 ids; the dump ~1.6 h for 39 sources (~87k tokens at the 512-id cap: 60 s of
dequant per forward + ~1.05 PFLOP at 300 GFLOP/s), peak RSS ~12 GiB, dumps ~1.3 GB; acceptance
~5-10 min per arm (18.7k anchors, 74k context rows: the smoke test's rates on 2 CPUs, faster at
16 threads), ~3-4 GiB; lookup and cost seconds. Docker cap 54 GB (one oracle at a time:
the driver refuses beside any other `agnes-ref-img` container unless `FORCE=1`).

## Results

Pending: `oracle-out-eagle3-k2/summary.md` (acceptance per corpus and arm, lookup, verify /
draft bytes, S_K at 4k and every depth) and `facts.log` (the drafter's tensors, the `d2t`
check on the real file, its embedding against K2's bytes).

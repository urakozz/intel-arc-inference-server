# Spec 24a - LFM2.5-24B-A2B (`lfm2_moe`): the facts, the CPU reference, golden sets, the A4 reference

> DROPPED (operator, 2026-10-27): support removed from the engine in 4822e3e; kept for the record.
>
> Reusable later (revive from git: `git show 4c491f8:<path>` - 4c491f8, this branch's parent on main, is the last
> commit that has all of it):
> - the MoE router: `src/kernels/lfm2moe/lfm_moe.cl` (`lfm_router`; `lfm_route` - sigmoid, the expert bias for the selection only, renormalised weights; `lfm_moe_gate_up` / `lfm_moe_down`), `src/kernels/lfm2moe/lfm_pf_moe.cl`, `src/kernels/lfm2moe/lfm_expert.h`; twins `tests/kernels/lfm2moe_ref.h`, `tests/kernels/lfm2moe_pf_ref.h`; tests `tests/kernels/lfm2moe_ref_test.cc`, `tests/kernels/lfm2moe_kernels_test.cc`, `tests/kernels/lfm2moe_pf_kernels_test.cc`
> - the pythonic tool-call parser: `src/server/toolcall_lfm2.h`, `src/server/toolcall_lfm2.cc`, `tests/server/toolcall_lfm2_test.cc`; `tools/toolcall/score.py --format pythonic`, `tools/toolcall/lfm2moe_rules.cc`
> - the rest: the short conv `src/kernels/lfm2moe/lfm_conv.cl` / `lfm_pf_conv.cl`, head_dim-64 attention `src/kernels/lfm2moe/lfm_attn.cl` / `lfm_pf_attn.cl`, `src/model/lfm2moe.h`, `src/model/lfm2moe.cc`, `src/loader/lfm2moe_*`, `src/runtime/lfm2moe/` (`lfm_prep.cl` still live in `src/kernels/bolt/`)

**Status (2026-10-15): built on the Mac (branch `spec24a-lfm2moe-reference`); spec 24 §12 "24a as built".** F1 bitwise on the tiny models; on the real bf16 weights (the Mac, 40 GB container): `hfcheck --layers 6` BITWISE, `ppl`, golden `lfm2moe_short` / `4k` / `agentic`. Box queue row **48** (rows 42-47 are spec 23's): the box image's tests, `lfm2moe_32k` and the A4 reference (above the Mac's 6 h rule).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the reference every later spec 24 stage is graded against: a facts sheet with every rounding point of spec 24 §2 pinned from transformers 5.19.0 source (file:line); `tools/oracle/lfm2moe_ref.py`, transformers' own `Lfm2MoeForCausalLM` layer-streamed, **bitwise** equal to the un-streamed transformers forward on the tiny random model and its variants (F1); golden sets on short and long prompts and an agentic transcript with pythonic tool calls; the A4 set rendered by LFM2's template and its reference run on the bf16 original (box CPU).

**Architecture:** spec 24 §2, §8 24a. Spec 21a's arrangement (`tools/oracle/qwen4exp_ref.py`): transformers' own model class built on `meta`, `stream.py`'s `attach` materialising one decoder layer at a time, routed experts read lazily (bf16 from the original, or dequantised from our int4 g64 export by `stream.dequant_t`), routes recorded by hooks, `layer_major` for batches. LFM2 adds no new mechanism to that arrangement - its traps are rounding points (the router's bf16 chain, the conv's tap order and state, the two-rounding norm, the final `embedding_norm`, the tied head), each pinned by a test.

**Tech Stack:** Python 3, torch 2.14.1 CPU, **transformers 5.19.0** through spec 21a's site (`tools/oracle/qwen4exp_env.sh <dir>`: `pip --no-deps --target`, tokenizers 0.23.2; the `agnes-ref-img` image keeps 5.15.0 and is never changed), safetensors; the Mac's `agnes-ref-img` container (8 GB, 4 CPUs - the operator's cap) for tests and tiny runs; the box CPU (`tools/oracle/run_in_container.sh`) for real weights.

**Spec:** `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` (§1, §2, §5, §7 F1 / F6's A4 half, §8 24a, §9, §10 decisions 5, 13). Precedents: plan 21a (`2026-10-09-spec21a-qwen4exp-facts-and-reference.md`) task for task; `tools/oracle/qwen4exp_ref.py` + `test_qwen4exp_ref.py`, `tools/oracle/ornith_ref.py` (`LazyExperts`, `layer_major`), `tools/oracle/kolibri_ref.py` (`run` / `ppl` / `facts`), `tools/box_validate/qwen4exp_oracle.sh` (resumable golden runs, `MemAvailable` floor, exit 77), `tools/toolcall/{make_set,score,oracle_generate}.py`, `tools/toolcall/a4_ref.sh`.

## Dependencies and branch points

- **None on the engine.** No C++, no kernel. First of spec 24's plans: 24q (the recipe's names and `eval`), 24b (`expected_names`, `make_synth.py`), 24c (the fixture imports this reference's restated ops) depend on it.
- **Weights.** The tiny model `yujiepan/lfm2-moe-tiny-random` (10.17 MB, bf16; identical to `tiny-random/lfm2-moe`: 3 layers `conv, conv, full_attention` - the real model's first three -, 2 dense layers, 32 experts top-4, hidden 64, 2 q / 1 kv heads of **32**, `expert_bias` all zero, vocab 65536 (LFM2's first tokenizer), `tie_word_embeddings` absent from config.json - the config class's default `True`, C:66) is fetched into the HF cache by Task 1, never vendored. It is the closest of the four public tiny models to the real config (layer pattern, dense count, GQA, top-4, tied); the other three were read and set aside (`optimum-intel-internal-testing/tiny-random-lfm2-moe`: 64 experts but `attn, attn, conv`, one dense layer, no GQA, fp32; `hf-tiny-v2/tiny-random-Lfm2MoeForCausalLM`: untied, 8 experts top-2, 2 layers). What it lacks - a conv + MoE layer, head_dim 64, non-zero biases, the 128000 vocab - Task 2's variants add.
- **The real checkpoint** (`LiquidAI/LFM2.5-24B-A2B`, 47.96 GB bf16, 10 shards) is box-side only; on the Mac only its small files (config, index, generation config, tokenizer files, chat template, README, LICENSE - already in the HF cache) are read. Every real-weight step SKIPs (exit 77, "missing data") until the operator downloads it on the box (`df -h ~` first).
- **Our int4 export** (24q) does not exist yet: the engine-format path is tested on tiny exports Task 2 writes in the `auto_round:auto_gptq` form; Task 7's `export` mode runs on 24q's RTN / AutoRound outputs when they exist.

## Global Constraints

- Branch `spec24a-lfm2moe-reference` from main; signed commits; no merge, no push. Box tree automatic (`tools/box.sh dir`); `tools/box.env` copied from the main checkout if missing, never read into a log, never printed, never committed.
- **One reference version:** transformers 5.19.0. `lfm2moe_ref.py` refuses any other installed version, naming it (`PINNED_TRANSFORMERS = "5.19.0"`).
- **The reference's attention is eager** (`attn_implementation="eager"`; spec 24 §2.6, decision 13): every reference run pins it; the facts sheet records sdpa's difference on the tiny model (max |Δlogit|), never uses it.
- Python tests in `agnes-ref-img` with `--memory 8g --cpus 4 -e OMP_NUM_THREADS=3` and `PYTHONPATH=/ws/oracle-out-q4exp/site` (spec 21a's site; created by `tools/oracle/qwen4exp_env.sh oracle-out-q4exp/site` if absent). One oracle container at a time on the Mac.
- Box CPU runs: the oracle container, `MemAvailable` >= the script's floor, one at a time, detached (`tools/probe/detach.sh`), polled in bounded foreground loops; no GPU, no GPU lock.
- Every number in docs is measured, or marked derived / estimated / proposed / published. Facts carry `M:line` / `C:line` / `file.py:line` citations into the installed 5.19.0.
- No `rm -rf`. No mention of any third-party engine in the repo beyond what spec 24 cites.

## Review Focus

1. **Bitwise, not close.** The streamed port equals the un-streamed transformers forward bit for bit in bf16 eager and in fp32, prompt and cached decode, and a prompt fed in chunks equals one forward. A tolerance anywhere in F1 hides exactly the rounding points the engine must copy.
2. **The router's bf16 chain.** `s_b = bf16(sigmoid(l_b))`, selection on `f32(s_b) + bias` (fp32), the weights from `s_b` (not `sel`), `bf16(bf16(Σ) + 1e-6)`, a bf16 division (M:183-194). A test where the bias reorders experts (an expert with a lower sigmoid selected) and a test where a non-zero bias would change the weights if it leaked into them; ties planted at the 4th / 5th `sel`, with torch's CPU tie order recorded (spec 21 §12 finding 2: not the lower id), the recorder flagging an exact 0 gap.
3. **The conv's state is `Bx`, its taps oldest first, its history zero.** A cached decode step reads only the newest two state columns; a prompt split at every offset 1..5 across a chunk boundary equals one forward; position 0 and 1 convolve against zeros (M:357-397, `cache_utils.py:1100-1137`).
4. **The combine's order is what the run used.** `grouped_mm` (slot order = descending `sel`, fp32 sum, one rounding, `integrations/moe.py:484-495`) or the eager fallback (ascending id, bf16 accumulator, M:156-166): Task 1 records which one 5.19.0 ran on torch 2.14.1 CPU, and the restated combine in `lfm2moe_ref.py` matches it bitwise; the other one's difference is printed.
5. **The end of the model.** `embedding_norm` runs after layer 39 (M:622), not after the embedding; the head is `embed_tokens.weight` (M:632); logits rows 125017..127999 exist (the 128000-row head) and are never a target id.

---

### Task 1: the tiny model end to end in 5.19.0 (Mac)

**Files:**
- Create: `tools/oracle/lfm2moe_tiny_check.py`
- Test: its exit code

- [x] **Step 1: write `lfm2moe_tiny_check.py <snapshot>`:** `AutoConfig` (print `layer_types`, `tie_word_embeddings`, `num_dense_layers`, `use_expert_bias`); `Lfm2MoeForCausalLM.from_pretrained(..., dtype=bf16, attn_implementation="eager")` and fp32; print `model.config._experts_implementation` (`grouped_mm` or `eager`, `modeling_utils.py:1887-1909`) and whether `causal_conv1d_update` / `causal_conv1d_fn` resolved to the torch functions (`integrations/hub_kernels.py:984-1019`: `is_new_implementation` false); a 24-id prompt with `generate(max_new_tokens=8, do_sample=False)` then the same 32 ids as one uncached forward: the last 8 rows' argmax equal and the logits bitwise in fp32 (bf16: print max |diff|, recorded); `lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()` (tied); a sdpa twin of the bf16 run: print max |Δlogit| (the facts sheet's sdpa row). Exit 0 only if all hold.
- [x] **Step 2: run on the Mac:**

```sh
uvx --from huggingface_hub hf download yujiepan/lfm2-moe-tiny-random
docker run --rm --memory 8g --memory-swap 8g --cpus 4 -e OMP_NUM_THREADS=3 -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/tmp/hf -v "$PWD":/ws -w /ws agnes-ref-img:latest sh -c \
  'tools/oracle/qwen4exp_env.sh oracle-out-q4exp/site && PYTHONPATH=/ws/oracle-out-q4exp/site \
   python3 tools/oracle/lfm2moe_tiny_check.py $(ls -d /hf/hub/models--yujiepan--lfm2-moe-tiny-random/snapshots/*/)'
```
Expected: `tiny check: OK` and the recorded lines. If 5.19.0 refuses the tiny config, stop and report (no silent version switch; 5.0.0.dev0 - the tiny's `transformers_version` - may cross-check, never be the reference).
- [x] **Step 3: commit** `git commit -S -m "oracle: the lfm2_moe tiny model end to end in transformers 5.19.0 (spec 24a)"`.

### Task 2: tiny variants the downloaded model cannot give (Mac)

**Files:**
- Create: `tools/oracle/lfm2moe_make_tiny.py`, `tools/oracle/test_lfm2moe_make_tiny.py`

**Interfaces:**
- Produces: `lfm2moe_make_tiny.py <out> --variant NAME [--seed 0] [--tokenizer <LFM2.5 snapshot>]`, writing a `save_pretrained`-shaped directory with the real checkpoint's tensor names (`model.layers.L.feed_forward.experts.E.{w1,w2,w3}.weight`, per expert, as the original stores them; `expert_bias` F32):
  - `pattern8`: 8 layers `conv, conv, full_attention, conv, conv, conv, full_attention, conv` (the real model's first eight), `num_dense_layers` 2, hidden 256, 4 q / 1 kv heads of **64** (GQA 4, the real head_dim), 16 experts top-4, `moe_intermediate_size` 128, `intermediate_size` 384, vocab 128000 with the real `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` copied (the embedding is 65.5 MB bf16), `expert_bias` ~ N(0, 0.05) fp32 (non-zero: the bias reorders), θ 1e7, `max_position_embeddings` 131072, bf16;
  - `pattern8-fp32`: the same in fp32;
  - `ties`: `pattern8` with experts 3 and 11's router rows and biases identical in layer 2 and experts 5 / 6 in layer 4 (exact `sel` ties at the cut on most tokens);
  - `int4-g64`: `pattern8` with every `nn.Linear` (experts, dense `w1/w2/w3`, conv `in_proj` / `out_proj`, attention `q/k/v/out_proj`) packed RTN int4 g64 sym in the `auto_round:auto_gptq` bytes (`qweight` / `scales` F16 / `qzeros` 0x77777777, `src/loader/rtn.h:20-27`'s rule) and `quantization_config` as AutoRound 0.17.0 writes it; `--expert-names gate_up_down` writes the experts as `experts.E.{gate_proj,up_proj,down_proj}` (AutoRound's unfused spelling, spec 24 §5), the default keeps `w1 / w3 / w2`.
- [x] **Step 1: the failing test** `test_lfm2moe_make_tiny.py`: each variant loads in 5.19.0 (`from_pretrained`, eager); `pattern8`'s layer types and the dense / MoE split are the real model's for layers 0..7; `ties` really ties (`sel` equal at the planted pairs on > 50 % of 64 random tokens); the int4 variant's dequant equals `(q - 8) x scale` (`stream.dequant_t`) and its names follow `--expert-names`. Run (Task 1's container, `python3 tools/oracle/test_lfm2moe_make_tiny.py`): FAIL (`ModuleNotFoundError`).
- [x] **Step 2: implement; PASS. Commit** `git commit -S -m "oracle: lfm2_moe tiny variants - the real layer pattern at head_dim 64, biased routers, planted ties, int4 g64 exports in both expert spellings (spec 24a)"`.

### Task 3: `lfm2moe_ref.py` and F1 (Mac)

**Files:**
- Create: `tools/oracle/lfm2moe_ref.py`, `tools/oracle/test_lfm2moe_ref.py`
- Modify: `tools/oracle/README.md` (a section "The LFM2.5-24B-A2B reference" and the Files table rows)

**Interfaces:**
- Consumes: `tools/oracle/stream.py` (`attach`, `dequant_t`, `checkpoint_reader`, the hold mode - imported, not edited), `qwen4exp_ref.py`'s patterns (copied where model-specific).
- Produces (imported by Tasks 4-7, 24q's `evaluate.py`, 24b's `make_synth.py` / `check.py`, 24c's fixture generator):

```python
# tools/oracle/lfm2moe_ref.py
PINNED_TRANSFORMERS = "5.19.0"
def config(snapshot: str, layers: int | None = None) -> "Lfm2MoeConfig"
    # layers=N truncates num_hidden_layers and layer_types[:N]; N >= 3 (the first attention layer is 2: transformers'
    # DynamicCache takes its length from an attention layer - spec 21 §12 finding 4 - so N < 3 is refused by name)
def expected_names(cfg, form: str, expert_names: str = "w") -> list[str]
    # every tensor of a checkpoint in form "bf16" (the original) or "int4" (our export: every nn.Linear packed);
    # expert_names "w" (experts.E.w1/w3/w2) | "proj" (experts.E.gate_proj/up_proj/down_proj)
def build_streamed(snapshot: str, cfg, weights: str = "bf16") -> "Lfm2MoeForCausalLM"
    # weights: "bf16" (the original, experts read lazily) | "int4" (an export dequantised: the engine-format reference)
class Recorder:            # per layer and row: h after the layer, the mixer output, the FFN output, the conv state (Bx of
                           # the last 2 positions), route ids (rank order) / w / sel / the 4th-5th sel gap; logits rows
def layer_major(model, pf)                     # ornith_ref.layer_major's restatement for this class
# the restated ops (24c's host twin is generated from these; each is tested bitwise against the module it restates)
def rms_norm(x_bf16, w_bf16, eps) -> Tensor                # M:60-65, two roundings
def route(x_bf16, gate_w, bias_f32, k=4) -> tuple[ids, w_bf16, sel_f32, gap]    # M:183-194
def combine(proj_out_bf16, w_bf16, order: str) -> Tensor   # "grouped_mm" | "eager" (Task 1's finding picks the default)
def short_conv(x_bf16, in_w, conv_w, out_w, state_bx) -> tuple[y, new_state]  # M:437-465
def main()  # subcommands: run | ppl | facts | hfcheck
```

`run <snapshot> --prompt <ids> --out <p>.golden.safetensors --gen 32 [--layers N] [--weights bf16|int4] [--chunk C]`: the golden layout - `h.L<l>` bf16 [T][2048] (for T > 4096: the last `--act-tail` 256 rows), `route.ids.L<l>` i32 [T + gen][4] (rank order), `route.w.L<l>` f32 (bf16 values), `route.sel.L<l>` f32 [T + gen][5] (the top 5 `sel`), `route.gap.L<l>` f32 (4th - 5th), `conv.L<l>` bf16 [2][2048] (the state after the prompt), `logits` f32 (the last `--logits-tail` 1024 rows and every generated row), `tokens`, `nll` f32 [T - 1]; a log with the gap summary per layer (min, p01, p1, median, exact ties).

- [x] **Step 1: the failing tests** (`test_lfm2moe_ref.py`; each against transformers 5.19.0 un-streamed; skipped with a message when a snapshot is absent):
  - `test_streamed_equals_hf_bf16` / `_fp32` on `pattern8` and the downloaded tiny: a 40-id and a 600-id prompt, then 8 cached decode steps - logits and every `h.L*` bitwise (Review Focus 1);
  - `test_chunked_equals_whole`: the 600-id prompt fed as chunks of 1, 2, 3, 64 and 255 through the cache equals one forward bitwise, conv state included (Review Focus 3);
  - `test_router_chain`: `route()` equals `Lfm2MoeTopKRouter` bitwise on 512 random rows of `pattern8` layer 2; a crafted row where the bias selects an expert with a lower sigmoid; with the bias zeroed the weights are unchanged and the selection may change (Review Focus 2);
  - `test_route_ties`: on `ties`, the recorded route follows torch's CPU order and `route.gap` is exactly 0 on the planted rows;
  - `test_combine_order`: `combine(..., Task 1's order)` equals the experts module bitwise; the other order differs on > 0 rows (printed);
  - `test_short_conv`: `short_conv` equals `Lfm2MoeShortConv` bitwise for a prompt and a decode step; positions 0 and 1 convolve against zeros; swapping two taps changes the output (the order is live);
  - `test_norm_two_roundings`: `rms_norm` equals `Lfm2MoeRMSNorm` bitwise; a one-rounding variant differs on > 0 elements;
  - `test_final_norm_and_tie` (Review Focus 5): dropping `embedding_norm` changes the logits; the head is the embedding;
  - `test_int4_export`: on `int4-g64` (both expert spellings) `build_streamed(..., weights="int4")` equals transformers on the dequantised twin bitwise;
  - `test_layers_truncation`: `config(..., layers=3)` equals transformers with `num_hidden_layers=3` bitwise; `layers=2` refused naming the attention layer.
- [x] **Step 2: run, expect FAIL** (`ModuleNotFoundError: lfm2moe_ref`):

```sh
docker run --rm --memory 8g --memory-swap 8g --cpus 4 -e OMP_NUM_THREADS=3 -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/ws/oracle-out-q4exp/site -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/tmp/hf \
  -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/oracle/test_lfm2moe_ref.py
```
- [x] **Step 3: implement until every test passes;** `facts` prints Task 4's tables; `hfcheck <snapshot> --layers N` runs transformers' own model (layer-streamed, `stream.attach`) against the port on real weights (Task 7).
- [x] **Step 4: README section** (the site rule, the subcommands, RAM / time per prompt ESTIMATED until Task 7 measures, what is bitwise against what). **Commit** `git commit -S -m "oracle: lfm2moe_ref.py - transformers 5.19.0's Lfm2Moe layer-streamed, restated router / conv / norm / combine; F1 bitwise on the tiny models (spec 24a)"`.

### Task 4: the facts sheet (Mac: source and small files)

**Files:**
- Create: `docs/probe-lfm2moe-2026-10-<dd>.md` (the day it is written), `tools/oracle/lfm2moe_facts.py`
- Modify: `docs/README.md` (one line)

- [x] **Step 1: the rounding chain**, one row per point with `M:line`: the norm (M:60-65, two roundings, eps 1e-5, the q / k norms over 64), RoPE (inv_freq fp32 M:104, `freqs` fp32 M:110, cos / sin cast to bf16 M:115, rotate-half M:216-220, the two products and the add each rounded M:244-245), eager attention (M:274-280: the bf16 QKᵀ, `x 0.125`, the fp32 softmax cast to bf16, the bf16 PV), the conv (M:437-465 and the cache's state rule, `cache_utils.py:1100-1137`), the router (M:183-194), the combine of Task 1's implementation (`integrations/moe.py:484-495` or M:156-166), the dense MLP (M:127-128), the layer and the model loop (M:495-515, M:608-627), the tie (M:632). Torch's CPU bf16 `conv1d` accumulation: measure on 10^5 random real-width rows whether `fma` taps-ascending in fp32 + one rounding equals torch's output (count the rows that differ; spec 24 §2.5 says the engine copies the reference where it can and documents the rest).
- [x] **Step 2: the checkpoint facts** (small files only): the 7610 names grouped (spec 24 §1), dtypes and shapes from the index and shard headers by range request (`expert_bias` F32 [64], `conv.conv.weight` BF16 [2048, 1, 3], no `lm_head.weight`), the shard map, the derived parameter counts (23,971,587,712 with the biases; the index's 23,971,585,280 without), the derived bytes of spec 24 §3 re-derived, `vocab_used` 125017 (`tokenizer.json`'s 124,893 + 124 added, contiguous), the special / added tokens and their `special` flags, the licence summary (spec 24 §1.4; LICENSE sha256 `4d28ca14...`).
- [x] **Step 3: tokens and template:** `chat_template.jinja` sha256 `ea663864...` and its features minja must render (`namespace`, macros, `is mapping` / `is iterable` / `is string`, `tojson`, `replace`, slicing, `.endswith`, `.split`, `{% generation %}`, `raise_exception`); the pythonic call grammar the template writes (`format_arg_value`, template lines 4-12: strings single-quoted with `\\ \' \n \r` escaped, mappings / lists `tojson`, everything else `| string`); a rendered example per feature (system + tools, a call with each argument type, a tool result, reasoning before / after the last user turn) through `apply_chat_template` - 24e's byte-identity fixtures; whether `tests/golden/prompts/{prose,code,long}.txt` re-tokenise (ids differ from Qwen's: different tokenizer).
- [x] **Step 4: torch's CPU `topk` on tied `sel`** at the 4th / 5th of 64 (planted ties, 50 rows): which index wins, stable or not, at torch 2.14.1 - the engine's rule is the lower id; the recorder flags exact ties (`route.gap == 0`).
- [x] **Step 5: commit** `git commit -S -m "probe: LFM2.5-24B-A2B facts - rounding chain, checkpoint, tokens, template, tie order (spec 24a)"`.

### Task 5: the golden prompts (Mac)

**Files:**
- Create: `tools/oracle/lfm2moe_prompts.py`, `tests/golden/prompts/lfm2moe_agentic.json`, the ids it writes (`tests/golden/prompts/lfm2moe_{short,4k,32k,agentic}.ids`)
- Modify: `tools/oracle/README.md`

- [x] **Step 1:** `lfm2moe_prompts.py <LFM2.5 snapshot>`: `short` = `prose.txt` + `code.txt` encoded by LFM2's tokenizer with BOS (`tools/oracle/tokenize.py`'s rule); `4k` / `32k` = the first 4096 / 32768 ids of `long.txt` repeated as `long32k.ids` was made (`tools/probe/mk_long_ids.py`'s recipe, re-tokenised); `agentic` = `lfm2moe_agentic.json` - an original multi-turn coding session written for this repo (system prompt, a user task about this repository, assistant turns with reasoning and **pythonic** tool calls given as structured `tool_calls`, tool results holding `src/runtime/kolibri/kolibri_sizes.h` and `src/loader/kolibri1_layout.h`), rendered with the checkpoint's own template (`apply_chat_template(..., tools=..., add_generation_prompt=True)`) to ~12k ids. The script prints each length and refuses a `32k` shorter than 32768.
- [x] **Step 2: commit** the script, the JSON and the ids: `git commit -S -m "tests: LFM2.5-24B-A2B golden prompts - short, 4k, 32k and an agentic session with pythonic tool calls (spec 24a)"`.

### Task 6: the A4 set and its scorer (Mac)

**Files:**
- Modify: `tools/toolcall/make_set.py` (`--model lfm2moe`), `tools/toolcall/score.py` (`--format pythonic`), `tools/toolcall/oracle_generate.py` (`--model lfm2moe` -> `lfm2moe_ref`), `tools/toolcall/a4_ref.sh` (`lfm2moe set|ref|status`), `tools/toolcall/test_score.py`, `tools/toolcall/test_make_set.py`
- Create: `tests/golden/toolcall-lfm2moe/` (the 36 scenarios' `.ids` / `.json`, made by `make_set.py`)

- [x] **Step 1: the failing tests:** `test_score.py` gains pythonic cases - a call with each literal type, single and double quotes, escapes, nested lists / dicts, two calls in one block (the first wins), text before and after the block, a positional argument (`incomplete`), an unterminated block (`incomplete`), output ending inside `<think>` (`reasoning`; the prompt ends in `<think>`, so the output opens inside reasoning); values compared as JSON values (`True` == `true`); `test_make_set.py`: a scenario re-rendered by LFM2's template equals `apply_chat_template` byte for byte and its ids `tokenizer(text)`'s.
- [x] **Step 2: implement:** `make_set.py --model lfm2moe --from tests/golden/toolcall --tok <LFM2.5 snapshot>` re-renders the 36 conversations (their structured messages and tools) with LFM2's template - the calls in history turns become pythonic through the template itself - into `tests/golden/toolcall-lfm2moe/`; `score.py --format pythonic`; `oracle_generate.py --model lfm2moe` (greedy through the reference's cache, `--weights bf16|int4`, `--batch`, the default 2048 new ids - reasoning comes first; ESTIMATED lengths printed); `a4_ref.sh lfm2moe set|ref|status` (`ref` -> `$OUT/<scenario>.bf16.{ids,txt}`, resumable).
- [x] **Step 3: PASS; commit** `git commit -S -m "toolcall: the A4 set through LFM2's template, the pythonic scorer, the lfm2moe reference generator (spec 24a)"`.

### Task 7: the box CPU runner and the real-weight runs

**Files:**
- Create: `tools/box_validate/lfm2moe_oracle.sh`
- Modify: `tools/box_validate/data.sh` (`have lfm2moe_bf16`, `have oracle_lfm2moe`, `have lfm2moe_a4_ref`), `tools/oracle/README.md`

**Interfaces:**
- Produces: `lfm2moe_oracle.sh <data tree> tests|ppl|hfcheck|golden|export <dir>|a4` (qwen4exp_oracle.sh's shape: per prompt resumable into `*.partial/`, moved when complete; `MemAvailable` floor `LFM_REF_MIN_GB` - 8 tiny, 64 real; exit 77 when short or when the snapshot is absent; `DRY_RUN=1` prints every command, the floor and an ESTIMATED time per prompt). `golden` -> `oracle-out-lfm2moe/<p>.{ids,golden.safetensors,log}` on the bf16 original for every Task 5 prompt (`--gen 32`); `export <dir>` -> `oracle-out-lfm2moe-<name>/` on an int4 export dequantised (24q's RTN / AutoRound, 24b's synthetics: the engine-format reference every F3 gate reads); `ppl` -> perplexity on `prose`, `code` and the agentic transcript; `a4` -> `a4_ref.sh lfm2moe ref` into `oracle-out-lfm2moe-a4/`. Input `SNAP_LFM2MOE` (default `models--LiquidAI--LFM2.5-24B-A2B`).

- [x] **Step 1: write the script and the `have` lines;** `DRY_RUN=1` prints the plan (the bf16 model is 47.96 GB: layer-streamed it needs ~6 GB resident plus the activations, ESTIMATED; whole it fits the box's RAM too - the streamed path is the one used, as for every reference).
- [x] **Step 2 (box CPU; run on the Mac): F1's real-weight half** (hfcheck BITWISE; ppl - the bound moved to code < 10, spec 24 §12) - `ppl` sane (finite, below 20 on prose - an ESTIMATED ceiling - recorded) and `hfcheck --layers 6` (transformers' own model on real weights against the port, 2048 ids + 4 cached steps, bitwise).
- [x] **Step 3 (box CPU; partly the Mac): the golden sets and the A4 reference** (Mac: short / 4k / agentic and their gap lines; box: 32k and the A4 reference - row 48) - `golden` on every prompt; `a4` (36 scenarios, greedy, hours ESTIMATED: reasoning precedes every call); each log's gap summary (the 4th / 5th `sel` per layer - the routing tolerance `B70_LFM_TIE_TOL` for 24c / 24d is set from it); times and RSS.
- [x] **Step 4: commit** `git commit -S -m "oracle: lfm2moe_oracle.sh - resumable golden runs on the bf16 original and on int4 exports, perplexity, the A4 reference (spec 24a)"` (outputs are git-ignored data).

### Task 8: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` (a section "24a as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 48), `tools/box_validate/stages.sh` (its `row 48` block)

- [x] **Step 1:** "24a as built": the experts implementation and conv path 5.19.0 ran, the tie order, the conv-accumulation finding, every departure from this plan, the measured gap distributions (the tolerance's evidence), per-prompt times.
- [x] **Step 2: the queue row 48** and its stages (stages.sh's header says how; `python3 tools/box_validate/test_box_validate.py` passes; `tools/box_validate.sh --dry-run --only r48` prints them):

```text
row 48 "spec 24a - LFM2.5-24B-A2B (lfm2_moe): the facts, the CPU reference (F1), golden sets, the A4 reference (plan 24a)"
stage r48.host     48 default cpu  oracle_image                 -  "lfm2moe_oracle.sh tests: test_lfm2moe_ref.py, test_lfm2moe_make_tiny.py, the toolcall tests (tiny models: hf download yujiepan/lfm2-moe-tiny-random)"
stage r48.ppl      48 optin   cpu  lfm2moe_bf16,oracle_image    -  "perplexity on prose / code / agentic (finite, < 20 ESTIMATED ceiling; recorded)"
stage r48.hfcheck  48 optin   cpu  lfm2moe_bf16,oracle_image    -  "hfcheck --layers 6: transformers' own model vs the port on real weights, BITWISE"
stage r48.golden   48 optin   cpu  lfm2moe_bf16,oracle_image    -  "golden sets on the bf16 original -> oracle-out-lfm2moe (gap lines per layer)"
stage r48.a4_ref   48 optin   cpu  lfm2moe_bf16,oracle_image    -  "the A4 reference on the bf16 original -> oracle-out-lfm2moe-a4 (hours)"
rownote 48 "Download first: hf download LiquidAI/LFM2.5-24B-A2B (47.96 GB; df -h ~ first; MemAvailable >= 64 GB for the real-weight stages)."
```
- [x] **Step 3: commit** `git commit -S -m "docs: spec 24 (24a as built), box queue row 48 - the LFM2.5-24B-A2B reference"`.

**Gate for the plan:** Mac - Task 1's tiny check OK; every test of `test_lfm2moe_ref.py`, `test_lfm2moe_make_tiny.py` and the toolcall tests green in `agnes-ref-img` with the 5.19.0 site; the facts sheet complete (every §2 rounding point cited, the checkpoint / token / template facts, the tie order, the conv-accumulation count). Box CPU (when the weights exist) - perplexity sane, `hfcheck` bitwise, the golden sets and the A4 reference written, the gap distributions printed. Hand back the sheet's tables, the gap summaries, the experts implementation 5.19.0 ran and the conv finding.

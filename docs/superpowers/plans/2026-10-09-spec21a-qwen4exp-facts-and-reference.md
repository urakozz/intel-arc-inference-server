# Spec 21a - Qwen3.8-Flash-Next (`qwen4_exp`): the facts, the CPU reference, golden sets, routing traces

**Status (2026-10-09): built on the Mac, branch `spec21a-qwen4exp-reference`; the box CPU runs pending (queue
row 30).** Tasks 1-5 and 8 done and their Mac gates green (F1 bitwise on the tiny model in bf16 and fp32; the
MTP head = an independent build bitwise); Tasks 6-7's scripts written, `DRY_RUN=1` checked - the real-weight
steps (6.2, 6.3, 7.2) wait for Intel's checkpoint on the box. As built, departures and findings: spec 21 §12
"21a as built" (decision 5's reuse is vLLM's opt-in; torch's tie rule is not the lower index; the original's
tokenizer adds `\p{M}`; `--layers N` needs N >= 4 cached). The operator approved spec 21 on 2026-10-09.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the reference every later 21 stage is graded against: a facts sheet with every rounding point of spec 21 §2 pinned from transformers 5.19.0 / vLLM source (file:line); `tools/oracle/qwen4exp_ref.py`, transformers' own text model layer-streamed, bitwise equal to the un-streamed transformers forward on the tiny random model (F1); the MTP head ported from vLLM and held to an independent build; golden sets on short (< 2052 positions, QSA dense) and long (4k / 8k / 32k, the selection active) prompts with per-layer routes, selections and gaps; teacher-forced routing traces for spec 22's P0.6.

**Architecture:** spec 21 §2, §8 21a. Ornith's arrangement (`tools/oracle/ornith_ref.py`: transformers' own model class built on `meta`, `stream.py`'s `attach` materialising one decoder layer at a time, routed experts dequantised on demand by a `LazyExperts`-style module, routes recorded by hooks, `layer_major` for batches) applied to `Qwen4ExpForCausalLM`, plus three pieces Ornith did not need: (1) the PLE n-gram table read by mmap, 16 rows a token, from the bf16 shards or from 21b's int8 file (the engine-format reference); (2) the QSA indexer with each complete block's compressed key computed once and cached (transformers recomputes every block for every query - the same per-block ops, so bitwise, and `O(T)` instead of `O(T²)` per layer), recording the selected blocks and the 512th / 513th scores; (3) the MTP head, which transformers drops (`_keys_to_ignore_on_load_unexpected`, M:1314), ported from vLLM `V/nvidia/mtp.py:164-357` onto transformers' own modules.

**Tech Stack:** Python 3, torch 2.14.1 CPU, **transformers 5.19.0** (the version spec 21 cites; not the 5.15.0 of `agnes-ref-img` - Task 1 installs it beside the image, never into it), safetensors; the Mac's `agnes-ref-img` container for tests and tiny runs; the box CPU (`tools/oracle/run_in_container.sh`) for real weights.

**Spec:** `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§1, §2, §5, §7 F1, §8 21a, §9, §10 decisions 1, 3, 4, 5). Spec 22's P0.6 (`docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` §3) consumes Task 7's traces. Precedents: `tools/oracle/ornith_ref.py` + `test_ornith_ref.py` (transformers model, streamed, `LazyExperts`, `layer_major`), `tools/oracle/kolibri_ref.py` (`run` / `ppl` / `facts` subcommands, golden layout), `tools/oracle/eagle3_ref.py` + `test_eagle3_ref.py` (a port held to an independent build from transformers modules), `tools/oracle/mtp_accept.py` (teacher-forced acceptance, `--opencode`), `tools/toolcall/oracle_generate.py` (`--batch`, `--resident`), `tools/box_validate/kolibri_oracle.sh` (resumable golden runs, `MemAvailable` check, exit 77), plan 15a Task 0, plan 18a, plan 20a.

## Dependencies and branch points

- **None on the engine.** This plan touches no C++ and no kernel. It is the first of spec 21's plans; 21b (the synthetic checkpoints call `expected_names`), 21c (the kernel fixture imports this reference's ops) and 21q (the recipe reads the facts sheet) depend on it.
- **Weights.** The tiny model (`qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next`, 124 MB, fp32, 4 layers: 3 GDN + 1 QSA, 4 experts top-2, hidden 16, `model_type: qwen4_exp_text`, saved by transformers 5.16.1) is fetched into the HF cache by Task 1; it is never vendored. The real checkpoints (`Qwen/Qwen3.8-Flash-Next` bf16, 360 GB; `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`, 181 GB) are box-side only and only their small files (config, index, generation config, tokenizer files, model card) are read on the Mac. Every real-weight step SKIPs (exit 77, "missing data") until the operator downloads them on the box (`df -h ~` first; the PLE shards alone are 102.4 GB).
- **Decision 6 (where the 360 GB runs) is 21q's**, not this plan's: this plan's box runs use Intel's 181 GB checkpoint (the engine-format reference: g128 experts dequantised exactly, bf16 dense as shipped) and, for the bf16-PLE KL of F6, the original's 128 PLE shards only.
- **Decision 7 (the PLE table form)** reaches this plan through the reader: Task 3 reads bf16 shards or an int8 file whose format 21b Task 1 defines; until 21b exists the int8 path is tested on a table this plan's own test writes in that format.

## Global Constraints

- Branch `spec21a-qwen4exp-reference` from main; signed commits there; no merge, no push. Box tree automatic (`tools/box.sh dir`); `tools/box.env` copied from the main checkout if missing, never read into a log, never committed.
- **One reference version.** transformers 5.19.0 for the main model (decision 1), vLLM v0.31.1rc0-138 (`b52ae2aa7f`, the local checkout `/Users/urakozz/PycharmProjects/vllm`) for the MTP head. `qwen4exp_ref.py` refuses to run on any other transformers version (ornith_ref's "prologue restated, refused on another version" rule), naming the installed one.
- **Never change `agnes-ref-img`** (every other oracle depends on its transformers 5.15.0): 5.19.0 lives in a separate `--target` site directory put first on `PYTHONPATH` (Task 1).
- Python tests run in `agnes-ref-img` (`--memory 28g`, `OMP_NUM_THREADS=3`) with that `PYTHONPATH`. One oracle container at a time on the Mac (the existing rule: refuse to start while another `agnes-ref-img` container runs).
- Box CPU runs: the oracle container, `MemAvailable` >= the script's floor, one at a time, detached (`tools/probe/detach.sh`), polled; no GPU, no GPU lock needed.
- Golden prompts must reach past position 2051: below it QSA is exactly causal attention (spec 21 §2.3), so only long prompts test the indexer.
- Every number in docs is measured, or marked derived / estimated / published. Facts carry file:line citations into the installed transformers 5.19.0 (`M:` / `C:` as spec 21 abbreviates) and vLLM (`V/...`).
- No `rm -rf`. Nothing about third-party engines in the repo beyond what spec 21 already cites.

## Review Focus

1. **The tiny model really loads, in the version the reference uses.** Spec 21 §1 says the tiny model's `qwen_sparse_attention` layer type is remapped (`configuration_utils.py:96`); a static reading of 5.19.0 shows the remap at the end of `PreTrainedConfig.__post_init__` (`configuration_utils.py:398`), which `Qwen4ExpTextConfig.__post_init__` reaches through `super().__post_init__` before `validate_architecture` (C:189) rejects unknown types - but nothing has run it (the Mac's scratch venv has no torch). Task 1 Step 2 runs it, and the facts sheet says which version loaded it and how.
2. **Bitwise, not close.** The streamed port must equal the un-streamed transformers forward bit for bit in bf16 eager and in fp32, prompt and cached decode, at prompt lengths that cross 2051 (the tiny model is cheap enough for 2100-3000 positions). A tolerance anywhere in F1 hides exactly the rounding points the engine must copy.
3. **The indexer cache is the reference's arithmetic.** The port caches each block's compressed key (fp32 mean of 4 raw keys -> bf16 -> `k_layernorm` -> RoPE at the block's first position, M:735-741) once; transformers recomputes it per query from the raw keys. Same ops per block, so the selected sets, scores and attention must be bitwise transformers'; Task 3's test proves it at positions 2047..2060 and at every `(p + 1) % 4 == 0` boundary in that range.
4. **The MTP port is vLLM's semantics, not transformers' (which has none).** `pre_fc_norm_hidden` is one RMS over 10240 (`V/nvidia/mtp.py:227-229`), `fc_hidden` is applied per stream, the embedding is added to every stream with unit injection, draft steps after the first reuse step 0's QSA selection (`skip_topk`, `:258-261`), the head returns both its single stream (logits) and its pre-mixer 4-stream hidden. Task 4 holds the port to an independent build and to each of these points by a crafted test.
5. **The PLE ids are exact integers.** The hash (`(t0·m0) XOR (t1·m1) [XOR (t2·m2)]` in int64, `mod prime + offset`), the EOS history rule (a missing predecessor, or one at or before an earlier EOS 248044, reads as 248044; padding too) and the multipliers / primes recomputed from the formula (seed 1234, C:155) against the checkpoint's I64 tensors. No float anywhere; any difference is a bug.

---

### Task 1: the 5.19.0 environment and the tiny model end to end (Mac)

**Files:**
- Create: `tools/oracle/qwen4exp_requirements.txt` (exact pins: `transformers==5.19.0` and the `huggingface_hub` / `tokenizers` / `safetensors` versions 5.19.0 requires - read from its `dist-info/METADATA`, not guessed), `tools/oracle/qwen4exp_env.sh`, `tools/oracle/qwen4exp_tiny_check.py`
- Test: `qwen4exp_tiny_check.py` (its exit code)

**Interfaces:**
- Produces: `tools/oracle/qwen4exp_env.sh <dir>` - `pip install --no-deps --target <dir> -r tools/oracle/qwen4exp_requirements.txt` (no dependency resolution, so torch 2.14.1+cpu of the image is never replaced) and prints `PYTHONPATH=<dir>`; idempotent (skips when `<dir>/transformers-5.19.0.dist-info` exists). On the Mac `<dir>` is `oracle-out-q4exp/site` (git-ignored; created inside the container, so the wheels match its Python); on the box `$DATA/q4exp-site` in the oracle image. Every later command of this plan runs with `-e PYTHONPATH=/ws/<dir>` (Mac) or the same through `run_in_container.sh`.

- [ ] **Step 1: the environment.** Write the pins and the script; in a fresh `agnes-ref-img` container run it, then `python3 -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'` -> `2.14.1+cpu 5.19.0`. If 5.19.0 needs a newer torch than 2.14.1 (its METADATA says), stop and report (the operator picks: a second image, or torch upgraded in a copy).
- [ ] **Step 2: load the tiny model end to end** - `qwen4exp_tiny_check.py <snapshot>`: `AutoConfig` (prints `layer_types` as remapped), `Qwen4ExpForCausalLM.from_pretrained(..., dtype=bf16, attn_implementation="eager")` and fp32; a 24-id prompt: `generate(max_new_tokens=8, do_sample=False)` with the cache, then the same 32 ids as one uncached forward - the logits of the last 8 rows equal the cached steps' (bitwise in fp32; in bf16 print the max |diff|, recorded); a 2100-id random prompt forward (the QSA selection active past 2051) completes and its selection mask differs from causal for at least one row past 2051 (printed count). Exit 0 only if all hold.
  - **If 5.19.0 refuses the tiny config** (`Unsupported Qwen4-Exp layer types`): first fallback - a copy of the tiny `config.json` with `layer_types` rewritten `qwen_sparse_attention -> indexed_attention` (the remap's own target; a data edit, not a code edit) in `oracle-out-q4exp/tiny-config/`, the weights symlinked; recorded in the facts sheet as "5.19.0 does not apply its legacy remap to this config; the tests load the rewritten copy". Second, only if the first fails: report and stop. **Never** switch the reference to another transformers version silently: 5.16.1 (the tiny model's `transformers_version`) may be used to cross-check that it loads there, never as the reference.
  - Also record: the experts implementation 5.19.0 uses by default for this class (`@use_experts_implementation`, M:912; `grouped_mm` vs `eager`), and that the tiny checkpoint's per-expert `mlp.experts.N.{gate,up,down}_proj.weight` load through `conversion_mapping.py:1918-1928` (qwen2_moe's per-expert -> fused converter plus the `ngram_embedding.shard_*` concatenation).
- [ ] **Step 3: run on the Mac:**

```sh
uvx --from huggingface_hub hf download qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next
docker run --rm --memory 28g --memory-swap 28g -e OMP_NUM_THREADS=3 -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/tmp/hf -v "$PWD":/ws -w /ws agnes-ref-img:latest sh -c \
  'tools/oracle/qwen4exp_env.sh oracle-out-q4exp/site && PYTHONPATH=/ws/oracle-out-q4exp/site \
   python3 tools/oracle/qwen4exp_tiny_check.py $(ls -d /hf/hub/models--qikp--tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next/snapshots/*/)'
```
Expected: `tiny check: OK` and the lines Step 2 records.
- [ ] **Step 4: commit** `git commit -S -m "oracle: transformers 5.19.0 beside the reference image, the qwen4_exp tiny model end to end (spec 21a)"`.

### Task 2: the facts sheet (Mac: source and small files only)

**Files:**
- Create: `docs/probe-qwen4exp-2026-10-09.md` (date = the day it is written), `tools/oracle/qwen4exp_facts.py`
- Modify: `docs/README.md` (one line)

**Interfaces:**
- Produces: the sheet 21b-21e cite for every rounding point and every name; `qwen4exp_facts.py <snapshot-or-dir>` (small files only: config, index, shard headers by range request, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, the I64 hash tensors fetched by byte range) printing the tables the sheet quotes.

- [ ] **Step 1: the rounding chain**, one row per point, with M:line: the RMSNorm `(1 + w)` in fp32 with one cast (M:144-169), the grouped HC norm (group 2560), the gated GDN norm `w · x̂` then `× sigmoid(z.float())` and its casts (M:172-188, `output_gate_type` sigmoid M:492), every bf16 point of the gated residual (`down` -> `/4` -> `silu` -> `up` -> `sigmoid` -> `⊙ xn` -> `mean` over streams; `block_inject` -> `/4` -> `sigmoid` -> `×2`; M:995-1030) and of the caller's `H0 + y ⊗ inj` (M:1294-1301: the product rounded, then the add), the QSA chain (q / gate split per head M:859-864, q / k norms, partial NEOX RoPE on 64 dims at θ 1e7, scale 1/16 M:820, `× sigmoid(gate)` M:890), the indexer (M:665-771: q norm + RoPE; raw keys cached un-normed; block mean in fp32 -> bf16 -> `k_layernorm` -> RoPE at the block start; `relu` scores summed over 4 heads, `/ √128`, fp32; `topk(min(512, n))` then the tail), the router (fp32 softmax over 512, top-10, renormalised, cast to bf16, M:961-970), the experts combine in the default implementation found in Task 1 (eager: bf16 accumulator, ascending expert id, `index_add_` of `y_e · w_e` rounded, M:925-949) and the shared expert (`sigmoid(shared_expert_gate · x) · shared(x)`, then `+`, M:981-992), PLE (M:1072-1247: the EOS shift, the hash, the 16 rows -> 2560, `key_proj` -> grouped norm, `value_proj`, the gate `sigmoid(sign(s) · √max(|s|, 1e-6))` with `s / √2560`, `norm_conv`, the dilated depthwise conv with SiLU over a 9-row history, `gated + conv`), the model loop (M:1392-1490: embedding repeated into 4 streams, no layer norms, the final mixer without inject, `lm_head` on its output).
- [ ] **Step 2: the checkpoint facts** (small files by range request, nothing downloaded whole): the original's and Intel's tensor lists grouped (the original stores routed experts fused, `mlp.experts.gate_up_proj [512, 1280, 2560]` gate-then-up, `down_proj [512, 2560, 640]`, under `model.language_model.`; Intel's per-expert `qweight` / `scales` F16 / `qzeros`, with their exact names), Intel's `quantization_config` (packing format, group 128, sym, `ignore_layers`, `extra_config`) and whether `loader::QuantConfig::parse` accepts it as is (g128 sym is accepted, `src/loader/quant.h:49-60`); the 128 PLE shards `[2500012, 160]` and how the 16 heads' row ranges `[offset_h, offset_h + prime_h)` fall across them (a table: head, prime, offset, first / last shard); the I64 tensors `layer_multipliers [3]`, `ngram_heads_vocab_sizes [16]`, `ngram_heads_offsets [16]` fetched by range and equal to `_build_layer_multipliers` / `_find_nth_prime_after` recomputed (M:1033-1105); the MTP tensors (`mtp.*`, 1.4 GB derived with int4 experts, 5.03 GB bf16) and vLLM's mapper dropping `mtp.hyper_connection_mixer.block_inject_weight` (`V/nvidia/mtp.py` `load_weights`); `generation_config.json` (EOS [248046, 248044], T 1.0, top-p 0.95, top-k 20); the licence from the original's model card.
- [ ] **Step 3: tokens and template.** `tokenizer.json` against Qwen3.8's (`tok::default_tokenizer_json()`'s file): byte-identical or the diff (vocab, merges, added tokens 248044-248076, `len(tok)`); the chat template's sha256 against `c3cf9e34...` (`src/tokenizer/chat_template.cc:142`, Qwen3.8's / Agnes's); whether `tests/golden/prompts/{prose,code,cjk,long,long32k}.ids` are valid ids for this tokenizer (they are Qwen3.8's - equal if the tokenizers are). If they differ, every later step re-makes ids from text with `tools/oracle/tokenize.py` (and `long32k.ids`, which has no `.txt`, by decoding with Qwen3.8's tokenizer first) - say which.
- [ ] **Step 4: torch's CPU `topk` on tied scores** (decision 3's ruled rule is "exact ties to the lower block index"): `torch.topk` on fp32 vectors of 600 with planted exact ties at the 512 / 513 cut - which index wins, stable or not, at torch 2.14.1; the same for the router's top-10 at 512 with a tie at the 10th. If torch does not prefer the lower index, the sheet says the engine's rule and the reference differ only on exact ties, and Task 3's recorder flags exact ties (`gap == 0`) so the gates treat those rows as undetermined.
- [ ] **Step 5: the derived bytes** of spec 21 §3 re-derived from the headers (routed experts, shared, routers, HC, GDN, QSA + indexer, PLE projections, `lm_head` int8) and the per-layer bytes the planner uses (21b Task 2): Intel's bf16 dense arm ~1.49 GB a layer, ours at int4. Commit `git commit -S -m "probe: Qwen3.8-Flash-Next facts - rounding chain, checkpoints, PLE layout, tokens (spec 21a)"`.

### Task 3: `qwen4exp_ref.py` and F1 on the tiny model

**Files:**
- Create: `tools/oracle/qwen4exp_ref.py`, `tools/oracle/test_qwen4exp_ref.py`, `tools/oracle/qwen4exp_make_tiny.py` (tiny variants the downloaded model cannot give: more experts and QSA layers, planted router ties, int4-packed experts at g64 / g128, an int8 PLE table, `mtp.*` (Task 4), and `--fused-experts` - the original's fused `gate_up_proj` / `down_proj` storage, which 21q's AutoRound pin is proven on)
- Modify: `tools/oracle/README.md` (a section "The Qwen3.8-Flash-Next reference" and the Files table rows)

**Interfaces:**
- Consumes: `tools/oracle/stream.py` (`attach`, `dequant_t` with `group_size` 64 or 128, `checkpoint_reader`, `conv1d_single_thread`, the hold mode), `tools/oracle/ornith_ref.py`'s patterns (copied into the new file where they are Ornith-specific; `stream.py` is imported, not edited).
- Produces (imported by Task 4, Task 7, 21b's `make_synth.py` / `check.py`, 21c's fixture generator):

```python
# tools/oracle/qwen4exp_ref.py
PINNED_TRANSFORMERS = "5.19.0"
def text_config(snapshot: str, layers: int | None = None, experts: str = "<Task 1's default>") -> "Qwen4ExpTextConfig"
    # config.json's text_config (or the tiny's top level); layers=N truncates num_hidden_layers and
    # layer_types[:N] (N >= 2: ple_layer_ids [2] must stay inside, C:239-245) - spec 21's --layers N
def expected_names(tc, form: str) -> list[str]
    # every tensor of the export in form "ours" (spec 21 §5 quantised set, per-expert int4 g64 names),
    # "intel" (Intel's names: g128 experts, bf16 dense) or "bf16" (the original, fused experts)
class PleTable:                       # 16 rows a token by mmap: bf16 shards, or the int8 file (21b Task 1)
    def __init__(self, snapshot_or_file: str, tc): ...
    def rows(self, ids: torch.Tensor) -> torch.Tensor   # int64 [T, 16] -> bf16 [T, 16, 160]
    def ids(self, history: torch.Tensor, multipliers, sizes, offsets) -> torch.Tensor  # exact int64 hash
def build_streamed(snapshot: str, tc, ple: PleTable, keep_dense=(), keep_experts=()) -> "Qwen4ExpForCausalLM"
class Recorder:                       # per layer and row: H (4-stream residual after the layer), mixer out,
                                      # MoE out, route ids / w / p10-p11 gap, QSA selected blocks, the
                                      # 512th / 513th score gap, PLE ids; the logits rows
def layer_major(model, pf)            # ornith_ref.layer_major's restatement for this class
def main()  # subcommands: run | ppl | facts | trace (Task 7) | hfcheck
```

`run <snapshot> --prompt <ids> --out <p>.golden.safetensors --gen 32 [--layers N] [--ple bf16|int8:<file>]`: the golden layout - `H.L<l>` bf16 [T][10240] (prompt rows), `mixer.L<l>`, `moe.L<l>` bf16 [T][2560], `route.ids.L<l>` i32 [T + gen][10], `route.w.L<l>` f32 (the bf16 weights widened), `route.gap.L<l>` f32 (p10 - p11), `qsa.sel.L<l>` i32 [rows past 2050][512] block ids ascending, -1 padded (rows <= 2050 are the identity and not stored), `qsa.gap.L<l>` f32 (512th - 513th score; +inf where n <= 512), `ple.ids` i64 [T + gen][16], `logits` f32 [T + gen][248320], `tokens`, `nll`; a log with the gap summaries (`MoE 10th/11th gap`, `QSA 512th/513th gap`: min, p01, p1, median, count of exact ties).

- [ ] **Step 1: the failing tests** in `test_qwen4exp_ref.py` (each against transformers 5.19.0 un-streamed on the downloaded tiny model unless named; skipped with a message when the tiny snapshot is absent):
  - `test_streamed_equals_hf_bf16` / `_fp32`: prompt of 40 ids and of 2100 ids, then 8 cached decode steps: logits and every `H.L*` bitwise (Review Focus 2);
  - `test_indexer_cache_bitwise`: the port's cached compressed keys vs transformers' per-query recomputation - selections, `qsa.gap`, attention output bitwise for rows 2047..2060 and each `(p + 1) % 4 == 0` boundary there (Review Focus 3); a row at 2050 selects everything (2051 visible), 2051 does not;
  - `test_hc_chain`: one gated residual through the port's restated ops equals `Qwen4ExpTextGatedResidual` bitwise, with `block_inject` absent (the final mixer) and present;
  - `test_ple_ids`: the port's `PleTable.ids` equals `Qwen4ExpTextNGramEmbedding`'s ids on sequences with EOS at 0, 1, 5 and twice, with padding, across a cached decode boundary; multipliers / sizes / offsets recomputed equal the tiny checkpoint's I64 tensors (Review Focus 5);
  - `test_sigmoid_gate` (a GDN layer with `output_gate_type` silu gives a different output - the gate is live), `test_v_head_map` (v head h reads k head h // 3: a k head zeroed silences exactly v heads 3k..3k+2);
  - `test_route_ties`: `qwen4exp_make_tiny.py` makes a 16-expert top-4 variant with two experts' router rows identical; the recorded route follows Task 2 Step 4's tie rule and `route.gap` is 0 there;
  - `test_layers_truncation`: `text_config(..., layers=2)` equals transformers with `num_hidden_layers=2` bitwise; `layers=1` refused naming `ple_layer_ids`;
  - `test_int4_experts`: a tiny variant whose experts are RTN-packed g64 and g128 (the `auto_round:auto_gptq` bytes, `qzeros` 0x77777777) - the dequant equals `(q - 8) · scale` (stream.py's rule) and the forward equals transformers on the dequantised weights bitwise; `test_ple_int8`: an int8 table in 21b's file format written by the test equals `round(w / s)` rows and the forward on it equals transformers on the dequantised table.
- [ ] **Step 2: run, expect FAIL** (`ModuleNotFoundError: qwen4exp_ref`):

```sh
docker run --rm --memory 28g --memory-swap 28g -e OMP_NUM_THREADS=3 -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/ws/oracle-out-q4exp/site -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/tmp/hf \
  -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/oracle/test_qwen4exp_ref.py
```
- [ ] **Step 3: implement** until every test passes; `facts` prints Task 2's tables from a snapshot; `hfcheck <snapshot> --layers N` runs transformers' own model layer-streamed against the port on real weights (the box, Task 6).
- [ ] **Step 4: README** section: the image + `PYTHONPATH` rule, the subcommands, RAM / time per prompt (ESTIMATED until Task 6 measures), what is bitwise against what. **Commit** `git commit -S -m "oracle: qwen4exp_ref.py - transformers 5.19.0's Qwen4Exp layer-streamed, cached indexer keys, PLE by mmap; F1 bitwise on the tiny model (spec 21a)"`.

### Task 4: the MTP head (vLLM's semantics) and its independent check

**Files:**
- Create: `tools/oracle/qwen4exp_mtp.py`, `tools/oracle/test_qwen4exp_mtp.py`
- Modify: `tools/oracle/qwen4exp_make_tiny.py` (a tiny with `mtp.*` tensors: the downloaded tiny has none), `tools/oracle/README.md`

**Interfaces:**
- Consumes: Task 3 (`build_streamed`, `Recorder`, the main model's pre-mixer 4-stream hidden `R` - the materialised `H` after the last layer's MLP-side combine, before `hyper_connection_mixer`).
- Produces:

```python
class MtpHead:                        # mtp.* on transformers' own modules (Qwen4ExpTextDecoderLayer with
                                      # layer_type "indexed_attention", its GatedResidual pair, RMSNorm)
    def __init__(self, src, tc, norm: str = "single"): ...   # norm: "single" (vLLM: one RMS over 10240,
                                                             # V/nvidia/mtp.py:227-229) | "per_stream"
    def step(self, R: Tensor, tok: Tensor, pos: int, cache, sel=None) -> tuple[Tensor, Tensor, Tensor]
        # -> (logits [V] via the shared lm_head, the head's own pre-mixer 4-stream hidden, its QSA selection)
def chain(main, head, ids, k: int) -> dict    # draft steps 1..k from every row; step >= 2 reuses step 0's sel
def accept(main, head, ctx_ids, cont_ids, k_max=3) -> dict   # teacher-forced greedy acceptance by depth (mtp_accept.py's shape)
```

- [ ] **Step 1: the failing tests:** `test_against_independent` - an independent build of the head straight from vLLM's text (written in the test from `Qwen4ExpTextGatedResidual`, `Qwen4ExpTextAttention`, `Qwen4ExpTextSparseMoeBlock`, `Qwen4ExpTextRMSNorm`, wired as `V/nvidia/mtp.py:283-357` reads) equals `MtpHead.step` bitwise on the MTP tiny, both `norm` forms; `test_unit_injection` (the head layer's attn-side input is `h` with `e` added to every stream, no inject weights); `test_fc_hidden_per_stream` (`fc_hidden` applied to each 2560 stream of the normed 10240 - a permuted stream permutes the output); `test_skip_topk` (step 2's selection is step 0's, bitwise, even when its own scores would choose otherwise - a crafted head); `test_returns_premixer` (step i + 1 consumes step i's pre-mixer hidden; the final mixer is used only for logits); `test_norm_forms_differ` (the two `pre_fc_norm_hidden` forms give different logits on random weights: the switch is live).
- [ ] **Step 2: run, expect FAIL; implement; PASS** (the Task 3 command with `test_qwen4exp_mtp.py`).
- [ ] **Step 3: commit** `git commit -S -m "oracle: the Qwen3.8-Flash-Next MTP head from vLLM's semantics, held to an independent build (spec 21a)"`.

### Task 5: the golden prompts (Mac)

**Files:**
- Create: `tools/oracle/qwen4exp_prompts.py`, `tests/golden/prompts/q4exp_agentic.json`, the ids files it writes (`tests/golden/prompts/q4exp_{short,4k,8k,32k,agentic}.ids`)
- Modify: `tools/oracle/README.md`

- [ ] **Step 1:** the prompt set (Global Constraints: past 2051): `short` = `prose.ids` + `code.ids` (< 2052, QSA dense: the dense-attention gate); `4k`, `8k`, `32k` = the first 4096 / 8192 / 32768 ids of `long32k.ids` (re-made for this tokenizer if Task 2 Step 3 found it differs); `agentic` = `q4exp_agentic.json` - an original multi-turn coding session written for this repo (system prompt, a user task about this repository, assistant turns with Qwen XML tool calls, tool results holding this repo's own source files `src/runtime/kolibri/kolibri_sizes.h` and `src/loader/kolibri1_layout.h`), rendered with the checkpoint's own template (`apply_chat_template`, `enable_thinking=True`) to ~12k ids. `qwen4exp_prompts.py <snapshot>` writes every `.ids` and prints each length; it refuses a set where `4k` / `8k` / `32k` / `agentic` are not all > 2051.
- [ ] **Step 2: commit** the script, the JSON and the ids: `git commit -S -m "tests: Qwen3.8-Flash-Next golden prompts - dense short, 4k / 8k / 32k and an agentic session past the QSA cut (spec 21a)"`.

### Task 6: the golden-set runner and the real-weight runs (box CPU)

**Files:**
- Create: `tools/box_validate/qwen4exp_oracle.sh`
- Modify: `tools/box_validate/data.sh` (`have q4exp_intel`, `have q4exp_bf16`, `have oracle_q4exp`, `have oracle_q4exp_layers`), `tools/oracle/README.md`

**Interfaces:**
- Produces: `qwen4exp_oracle.sh <data tree> tiny|intel|intel-layers N|ours|ppl` (kolibri_oracle.sh's shape: per prompt resumable into `oracle-out-q4exp*.partial/`, moved when complete; `MemAvailable` floor `Q4_REF_MIN_GB` (default 24 for tiny, 64 for real), exit 77 when short or when the snapshot is absent): `intel` -> `oracle-out-q4exp/<p>.{ids,golden.safetensors,log}` for every Task 5 prompt (`--gen 32`); `intel-layers N` -> `oracle-out-q4exp-L<N>/` (the reference truncated identically: what 21c's `--layers N` golden gate reads); `ppl` -> a perplexity line per text on `prose.txt`, `code.txt` and the agentic transcript. Inputs: `SNAP_Q4EXP_INTEL` (default `models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound`), the PLE from `Q4_PLE` (the original's shards, `models--Qwen--Qwen3.8-Flash-Next`, or 21b's int8 file once it exists - the engine-format reference uses the int8 file, as spec 21 F3 says).

- [ ] **Step 1: write the script and the `have` lines;** `DRY_RUN=1` prints every command, the RAM floor and an ESTIMATED time per prompt (the dequant of one layer's 512 experts, ~2.7 GB bf16, per forward; the 32k prompt's indexer and attention at 12 layers) with no container.
- [ ] **Step 2 (box CPU): F1's third bullet** - `ppl` on Intel's checkpoint: perplexity sane (finite, below a stated sanity ceiling of 30 on prose, ESTIMATED, the number recorded); `hfcheck --layers 4` (transformers' own model on real weights against the port, bitwise).
- [ ] **Step 3 (box CPU): the golden sets** - `intel` for every prompt, then `intel-layers N` for N = 4 and 18 (one card's fit with Intel's bf16 dense layers, spec 21 §3) on `short`, `4k`, `agentic`; each log's gap summaries (decision 3's τ evidence: the 512th / 513th gap distribution per QSA layer; R2's for the 10th / 11th). Record times and RSS.
- [ ] **Step 4: commit** `git commit -S -m "oracle: qwen4exp_oracle.sh - resumable golden runs, --layers N sets, perplexity (spec 21a)"` (the outputs are git-ignored data).

### Task 7: routing traces for spec 22's P0.6

**Files:**
- Modify: `tools/oracle/qwen4exp_ref.py` (`trace`), `tools/oracle/test_qwen4exp_ref.py` (`test_trace_equals_run`), `tools/box_validate/qwen4exp_oracle.sh` (`trace` mode), `tools/oracle/README.md`

**Interfaces:**
- Produces: `qwen4exp_ref.py trace <snapshot> --source NAME:IDS [...] [--opencode DIR] --out <dir>` - one teacher-forced forward per source (no generation), layer-major batched (`layer_major`, as oracle_generate's `--batch`), writing `<dir>/<name>.routes.safetensors`: `ids` i32 [T][48][10] (each row's top-10, ascending id), `p` f32 [T][48][10] (renormalised probabilities), `mtp_ids` i32 [T][10] (the MTP head's step-1 routes, when `mtp.*` exists), and metadata `{model, checkpoint revision, transformers, source, T}`. This is spec 22's `tools/oracle/offload_curve.py` input; its format is fixed here and documented in the README section.

- [ ] **Step 1:** `test_trace_equals_run`: on the tiny model, `trace` routes equal `run`'s `route.ids.L*` for the prompt rows bitwise; batching 3 sources equals each alone.
- [ ] **Step 2 (box CPU):** `qwen4exp_oracle.sh <data> trace` - sources: the 36 A4 scenarios (`tests/golden/toolcall/*.ids`, ids valid per Task 2 Step 3), the agentic prompt, `code`, `prose`, and the recorded opencode session when `OPENCODE_LOG` is set (`--opencode`, mtp_accept.py's reader) -> `oracle-out-q4exp-traces/`; print tokens per source and the per-layer distinct-experts count.
- [ ] **Step 3: commit** `git commit -S -m "oracle: Qwen3.8-Flash-Next teacher-forced routing traces, the input of spec 22's hit-rate curve (spec 21a)"`.

### Task 8: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (a section "21a as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 30), `tools/box_validate/stages.sh` (its `row 30` block)

- [ ] **Step 1:** "21a as built": the environment and the version that loaded the tiny model, every departure from this plan, the measured gap distributions, τ's evidence for decision 3, the per-prompt times.
- [ ] **Step 2: the queue row** (30, or the next free number at build time): merged = the reference and its tooling, nothing on the card; how = stages `r30.host` (the Python tests in the oracle image with the 5.19.0 site), `r30.ppl` and `r30.hfcheck` (opt-in cpu: Intel's checkpoint), `r30.golden` (opt-in cpu: `qwen4exp_oracle.sh intel`, hours), `r30.golden_layers` (opt-in: `intel-layers 4`, `18`), `r30.traces` (opt-in: spec 22 P0.6's input); `rownote`s for the downloads (Intel 181 GB; the original's 128 PLE shards 102.4 GB for the bf16-PLE KL; `df -h` first) and the RAM floor. `python3 tools/box_validate/test_box_validate.py` passes; `tools/box_validate.sh --dry-run --only r30` prints the stages.
- [ ] **Step 3: commit** `git commit -S -m "docs: spec 21 (21a as built), box queue row 30 - the Qwen3.8-Flash-Next reference"`.

**Gate for the plan:** Mac - Task 1's tiny check OK, every test of `test_qwen4exp_ref.py` and `test_qwen4exp_mtp.py` green in `agnes-ref-img` with the 5.19.0 site, the facts sheet complete (every §2 rounding point cited, the checkpoint and token facts, the tie rule). Box CPU (when the weights exist) - perplexity sane, `hfcheck` bitwise, the golden sets and `--layers` sets written with their gap distributions, the traces written. Hand back with the sheet's tables, the gap summaries and the version that loaded the tiny model.

# Spec 29b - Nanbeige4.3-33B: the dense descriptor, the loader (AutoRound g64 under Llama names), synthetic checkpoints, the 1- and 2-stage memory plan

**Status (2026-11-01): PLANNED - after 29a.** Nothing built. No box queue row.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed; `WIP:` if needed); handoff file when you stop.

**Goal:** the dense family as data and on the cards: `model::Dense` held to `config.json` key by key (validates spec 29 §1's dimensions, refuses any other Llama by name), the loader for the checkpoint's AutoRound g64 tensors, the dense memory plan for one and two stages (the bytes-balanced split), and `b70-decode` / `b70-serve` dispatch on `model_type: llama` as far as "loads and prints its plan" - no kernel, no decode.

**Architecture:** spec 29 §3 (descriptor / loader rows), §4, §5. Precedents: plan 25b (Bolt's descriptor, loader, plan), K2's `k2_horizon.{h,cc}` / `k2_loader` / `k2_repack`, `loader/int4_layout.h` (layout 0 / 1 byte formulas), `loader/lm_head_int8` (generic tiling), `runtime/pipeline_plan.cc` (`best_split_at`, `pp_balance` - the logic, rewritten for an all-attention model because `require_split` refuses it). Optional structural read (not a copy): the dense descriptor / loader of the old Muse code at `git show 99adbb0:src/model/muse_glimmer.h`, `...src/loader/muse_loader.cc` (the revival parent: `99adbb0` = `98c13ef^`, 98c13ef being the Muse drop; the dropped spec 23 docs' `c57e08e` carries the same Muse files byte for byte; `4c491f8` does not resolve).

**Tech Stack:** C++17, CMake / ctest, Python for the synthetic checkpoints (29a's builders).

**Spec:** `docs/superpowers/specs/2026-11-01-spec29-nanbeige-design.md` (§1-§5, §11 R12).

## Dependencies and branch points

- 29a merged (tiny builders, facts sheet). Branch `spec29b-nanbeige-loader`. No shared source is edited except: `src/cli/b70_decode.cc` / `b70_serve.cc` dispatch lines, `src/cli/CMakeLists.txt`, `tests/CMakeLists.txt` (additions); `model_desc_test`'s "LlamaForCausalLM is refused" assertion stays true (the dense family dispatches on `model_type` before `desc_for_architecture`, like K2).
- **Box:** one card for Task 5's first half, both cards for its second; the real quantised checkpoint.

## Global Constraints

- Signed commits after every step; no merge, no push; no `rm -rf`; box tree automatic.
- **G0:** `tools/kernel_cmdlines` shows no change in this plan (it adds no kernel); the whole existing suite is green.
- No new CLI flag and no environment variable.
- Uploads of tensors >= 512 MB are chunked to <= 256 MiB (fix round A's rule; `loader.cc` upload helpers) - the head and embedding are 1.2 / 2.39 GB.

## Review Focus

1. **The descriptor is held to the config** key by key (a test per key, a test per refusal: wrong layer count, q != hidden, tied embeddings, rope scaling set, hidden act, bias true, attention bias, `head_dim` != 128).
2. **Bytes.** The loader's resident bytes equal spec 29 §2's table (16.32 GB int4 linears + 1.20 int8 head + 2.39 bf16 embed, plus norms) within 0.1 %; each stage's bytes follow.
3. **The int4 bytes are Qwen's bytes.** The same `auto_round:auto_gptq` packing the existing loader reads (qweight `[K/8][N]`, scales `[K/64][N]` fp16, sym zero 8, **signed scales** allowed); gate-up concatenated per `Interleave16` rows; qkv concatenated along N; `g_idx` absent or identity (checked, refused otherwise by name).
4. **The planner prints its line** (`memory: model ..., kv ..., decode state ..., prefill scratch ..., total ... of 32.530 GB`) per stage, and the lengths in spec 29 §4 are what it computes (within 3 %), or the doc is corrected.

---

### Task 1: the descriptor

**Files:** Create `src/model/dense.h`, `src/model/dense.cc`, `tests/model/dense_test.cc`.

- [ ] **Step 1: failing tests**, **Step 2: implement.** `struct Dense { layers 52, hidden 7168, q_heads 56, kv_heads 8, head_dim 128, ffn 22016, vocab 166912, vocab_used 166107, rope_theta, eps 1e-5, eos / bos ids; derived widths: qkv N 9216, gate-up N 44032, per-layer bytes; per-linear shape table with the split-K S candidates; `view(tp_rank, tp_size)` returning head counts / N / K per rank (identity at 1 - the hook for spec 17, tested only at 1) }`, `check_dense_config(json)` refusing by name. **Step 3: commit** `model: Dense - the plain Llama descriptor held to Nanbeige's config (spec 29b)`.

### Task 2: the checkpoint layout and the host repack

**Files:** Create `src/loader/dense_layout.h`, `dense_repack.{h,cc}`, `tests/loader/dense_layout_test.cc`, `tests/loader/dense_repack_test.cc`.

- [ ] **Step 1: failing tests** on 29a's `int4-g64` tiny export: names (`model.layers.N.self_attn.{q,k,v,o}_proj.{qweight,qzeros,scales}`, `mlp.{gate,up,down}_proj...`, `input_layernorm`, `post_attention_layernorm`, `embed_tokens`, `lm_head`, `model.norm`); the repack into layout 1 reproduces `(q - 8) x scale` bitwise against the host dequant; qkv / gate-up fusion order; a tensor with a wrong shape, a missing name, an extra name (the 471 / the quantised repo's counts) is refused by name. **Step 2: implement** (reuse `common/repack.h`'s `repack_int4_layout1`, `cols_concat`, `cols_interleave16`). **Step 3: commit.**

### Task 3: the loader (host half)

**Files:** Create `src/loader/dense_loader.{h,cc}`, `tests/loader/dense_load_checkpoint_test.cc` (SKIPs without the snapshot).

- [ ] **Step 1: failing tests**; **Step 2: implement:** read the 4 shards, repack per layer, norms to fp32 (plain `w`, no `1 + w`), bf16 embed kept bf16 (a gather, never read per token), the head: int8 by `lm_head_int8` (default; bf16 optional by the existing `--lm-head` flag) with the per-column scales, padded rows kept (masked by `VOCAB_USED`), uploads chunked <= 256 MiB; the loaded-bytes accounting. Synthetic checkpoints (`tools/oracle/dense_make_tiny.py` variants written to a temp dir) load through the same code. **Step 3: commit.**

### Task 4: the dense memory plan and the two-stage placement

**Files:** Create `src/runtime/dense/dense_plan.{h,cc}`, `tests/runtime/dense_plan_test.cc`.

- [ ] **Step 1: failing tests:** the plan's terms for 1 stage and 2 stages at max_len 4096 / 84k / 242k / 330k / 345k, bf16 and int8 KV, against spec 29 §4's table (derived by hand in the test with the formulas); the bytes-balanced `auto` split (min over s of max over stages of weights + KV + scratch + hand-off) and an explicit `--pipeline-split N`; the stage with the embedding and the stage with the head; the refusals (a split that does not fit, `--pp 2` with `--device`). **Step 2: implement** (the logic of `pp_balance` / `best_split_at` without `require_split`'s GDN / FA minimum; `kMaxLenQuantum` 256; the 1.5 GB reserve; auto max-len bisection). **Step 3: commit.**

### Task 5: the cards (box)

- [ ] **Step 1 (one card):** load the real quantised checkpoint's first 8 layers + head to device 0 and read every tensor back: **D1 bitwise** against the host dequant (all 8 layers, embed rows sampled, head rows sampled and the padded rows' mask). **Step 2 (the 10 GB buffer, spec 29 R12):** allocate one 10 GB device buffer through `l0::Mem` (it chains the relaxed-limits descriptor above `maxMemAllocSize`; the largest device allocation on record is the 8 GB bandwidth probe), fill it in 256 MiB chunks with a pattern, read back the first / last / random chunks and a kernel read at the far end. **Step 3 (two cards):** load the whole model split by `auto` for `--max-len 330000` int8 KV; print each stage's planned line and the real allocation (L0 residency query); D1 on a sample of layers per stage. **Step 4: commit** the records `docs/probe-nanbeige-load-2026-11-<dd>.md`.

### Task 6: dispatch and docs

- [ ] **Step 1:** `b70-decode` / `b70-serve` dispatch on `model_type: llama` to a `dense` path that loads, prints the plan and exits with "decode not built" until 29c; `docs/19-running-models.md` gets a Nanbeige column (never run on a B70 until 29c); spec 29 gets "29b as built". **Step 2: commit.**

**Gate for the plan:** Mac - every host test; `tools/box_check.sh --base main` green; `tools/kernel_cmdlines` unchanged. Box - D1 bitwise on both cards; the 10 GB allocation works (or the per-layer-group fallback is built and D1 re-run); the planner's lines match spec 29 §4 within 3 % (or the spec is corrected with the printed numbers).

**Box stages (prose):** GPU lock for every card job; idle-box check at dispatch. One card first (the load of 8 layers + head), then the 10 GB buffer test, then the whole model on two cards. Each stage is minutes; the real-checkpoint load reads 20 GB from disk (the cold-copy USB path exists).

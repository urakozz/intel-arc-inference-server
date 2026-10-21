# Spec 25b - Nemotron 3.5 Bolt: the descriptor, the loader (g128-expanded and native g64, one layout), synthetic checkpoints, the two-card memory plan

**Status (2026-10-22): built on the Mac (branch `spec25b-bolt-loader`; as built: spec 25 §14); the card stages are box queue row 59.**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** Bolt as data and on the cards: `model::BoltDesc` held to `llm_config` key by key (the heterogeneous per-layer
types, E and I), `loader::load_bolt` reading **either** SergiioB's g128 GPTQ checkpoint (through the existing exact g128 ->
g64 expansion) **or** our g64 export into one device layout, spec 25 §5.2's two exact loader relaxations (a restating `+:`
rule; signed scales audited; BF16 GPTQ scales exact-or-refused), synthetic real-width checkpoints in both spellings, and the
memory plan that places 40.46 GB on two cards by bytes and reserves spec 26's drafter.

**Architecture:** spec 25 §1.3, §3, §5, §8, §9 B0. 24b's pattern (`model::Lfm2MoeDesc` / `loader::load_lfm2moe` /
`runtime::lfm2moe::plan`), its as-built lessons carried over (spec 24 §13): names from 25a's `expected_names`, not from
guesses; **per-linear `{S, layout}` in the descriptor** (`BoltDesc::linears(l)`, the one source of truth, PROVISIONAL until
25c's P0 sweep: layout 1 for every routed expert and every N = 4096 linear, layout 0 for `in_proj` and the fused
`q‖k‖v` / `fc1‖shared_up`; S 1 everywhere); no test types a grand total (the planner's figure is the sum of its own
per-layer outputs, `CHECK_EQ`'d against an independent accumulation); the bytes formula shared by the loader and the
planner (`src/loader/bolt_layout.h`).

**Tech Stack:** C++17, Level Zero, CMake / ctest (`mac-host` preset for host tests); Python 3 for the synthetic checkpoints.

**Spec:** `docs/superpowers/specs/2026-10-20-spec25-nemotron-bolt-design.md` (§1, §3, §4.1, §5, §8, §9 B0, §12 decisions 4,
5, 9, 14, 18, 20). Precedents: plan 24b task for task; `src/loader/quant.{h,cc}` (`QuantConfig::parse`,
`LinearSrc::classify`'s g128 expansion, `bf16_to_f16_exact`, `assert_quant_invariants`), `src/loader/int4_layout.h`,
`src/loader/moe_layout.h`, `src/loader/lm_head_int8.h`, docs/13 ("g128 checkpoints", spec 21b's section),
`runtime::pp_balance`, `runtime::pp_landing_layout`.

## Dependencies and branch points

- **Plan 25a merged:** `bolt_ref.expected_names` ("gptq-g128", "ar-g64", "bf16"), `bolt_make_tiny.py`'s packers (the
  synthetic generator reuses them until 25q's lands - 24b's precedent; 25q merges the two `check.py`s).
- **No kernel.** The device half uploads bytes only; 25c binds them. kernel_cmdlines +0.
- **Decision 20 (ruled 2026-10-20: vendoring allowed):** `tests/model/bolt/` may hold a copied `config.json`; the
  **generator** (`make_config.py`, writing the config from the published values in code) is optional.
- **The relaxations touch shared code** (`src/loader/quant.cc`): every existing loader test stays green and every existing
  refusal still refuses (B0).

## Global Constraints

- Branch `spec25b-bolt-loader` from main; signed commits; no merge, no push.
- **B0:** no kernel binary changes; `tools/kernel_cmdlines` +0 / -0 / ~0; `tools/mac_check.sh --base main` green.
- Alignment: every device allocation base and every expert block base a multiple of 64 B (the LSC 2D-load rule,
  vllm-xpu-kernels 6f5935c) - asserted by the layout test, not assumed.
- Every number in docs measured or derived (from `bolt_layout.h`'s formulas, printed by a test).
- No `rm -rf`.

## Review Focus

1. **One layout from two sources.** A linear loaded from the g128 checkpoint and the same weights packed g64 with each
   g128 scale written to both g64 rows give **byte-identical** device buffers (the expansion is exact); the test builds both
   from one random matrix.
2. **The `+:` relaxation is exactly a restatement.** SergiioB's `dynamic` object parses; a `+:` rule changing bits, group or
   sym, or carrying any other key, still throws naming the rule; a `-:` rule behaves as before.
3. **Signed scales survive every path** (the repack, the g128 expansion, `assert_quant_invariants`, the quant scan,
   `check.py`) - a synthetic with half its scales negative round-trips bitwise and the dequant is `(q - 8) x scale`.
4. **The heterogeneous MoE tables.** E and I per layer from `block_configs`, never from the top-level `num_experts` (512 -
   a max, not any layer's count) or `intermediate_size` (2688 - unused by any block); the expert address of (layer, id) from
   one formula; the router rows per layer = E (a multiple of 16 in every layer: no padding).
5. **What is skipped is named.** The 849 vision / projector tensors skipped by prefix and counted; every other tensor
   consumed (unconsumed = 0); `language_model.backbone.*` accepted as an alias only if a checkpoint ships it.

---

### Task 1: the synthetic checkpoints (Python, no card)

**Files:**
- Create: `tools/quantize/bolt/make_synth.py`, `tests/model/bolt/make_config.py`
- Modify: `tools/quantize/bolt/check.py` (if 25q has not landed: a self-contained check, 24b's precedent - merged later)

- [ ] **Step 1:** `make_synth.py <out> --layers N --form gptq-g128|ar-g64 [--seed 0]`: real widths (hidden 4096, Mamba
  18560 / 8192, attention 4096 / 256, latent 1024, shared 5376, the real E and I of the first N blocks), random weights
  packed by 25a's packers (signed scales on half the groups), the real names, `quantization_config` in that form's spelling
  (the g128 form with `g_idx` and SergiioB's `dynamic` object incl. the `+:` rule), the kept tensors at their dtypes
  (`A_log` / `D` / `dt_bias` / `e_score_correction_bias` F32), a vision tensor or two to be skipped, the tokenizer and
  template copied from the HF cache at run time. N 8 (the first attention block is 7) is the smallest; N 12 the default
  (~5.5 GB, ESTIMATED). `make_config.py` writes the real `config.json` (omni wrapper + `llm_config`) from values typed from
  the published config, each annotated with its source key.
- [ ] **Step 2:** both forms at N 8 `ACCEPTED` by `check.py`; `bolt_ref.build_streamed(weights="int4")` loads both and
  their logits on 16 ids are bitwise equal between the forms when the g64 form is the g128 form's expansion (`--expand-from`).
- [ ] **Step 3: commit** `git commit -S -m "tools: Bolt synthetic real-width checkpoints in SergiioB's g128 GPTQ and our g64 spellings (spec 25b)"`.

### Task 2: the model table

**Files:**
- Create: `src/model/bolt.{h,cc}`, `tests/model/bolt_test.cc`
- Modify: `src/model/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**

```cpp
namespace model {
enum class BoltBlock : uint8_t { Mamba, Moe, Attention };
struct BoltLinear { uint32_t K, N, S, layout; };                 // per-linear {S, layout}: PROVISIONAL (25c sweeps)
struct BoltDesc {
  uint32_t layers = 88, hidden = 4096, vocab = 131072, vocab_used = 131072, max_pos = 262144;
  uint32_t m_heads = 128, m_head_dim = 64, m_groups = 8, m_state = 128, m_conv = 4, m_inproj = 18560, m_inner = 8192;
  uint32_t q_heads = 32, kv_heads = 2, head_dim = 128;
  uint32_t latent = 1024, shared_inter = 5376, top_k = 9;
  float eps = 1e-5f, routed_scale = 5.0f;
  std::vector<BoltBlock> block;                                  // [88]
  std::vector<uint32_t> experts, inter;                          // [88], 0 for non-MoE blocks
  bool int8_head = true;
  std::vector<BoltLinear> linears(uint32_t l) const;             // in_proj, out_proj | q||k||v, o | fc1||shared_up,
                                                                 // fc2, shared_down, expert up (K 1024, N I), expert down
};
BoltDesc check_bolt_config(const common::json::Value& config_json);   // the omni wrapper + llm_config
}
```
Refused by name (each throw names the key and its value): another `model_type` (wrapper or `llm_config`), another hidden /
vocab / head count, `num_nextn_predict_layers` > 0 or non-empty `mtp_block_configs`, `residual_in_fp32` true,
`use_qk_norm` true, `n_group` != 1 or `topk_group` != 1, `scoring_func` != sigmoid, `norm_topk_prob` false, a block
`num_experts_per_tok` != 9, an `n_routed_experts` not a multiple of 16 or above 512, a `moe_intermediate_size` not a
multiple of 64, a `block_type` other than the three, `layers_block_type` disagreeing with `block_configs`, a non-null
`sliding_window`, `mamba_proj_bias` / `attention_bias` / `mlp_bias` true, `use_conv_bias` false, `hidden_act` /
`mlp_hidden_act` != relu2, `mamba_hidden_act` != silu, `tie_word_embeddings` true at either level.
- [ ] **Step 1: the failing test** `bolt_test.cc`: the published config (from `make_config.py`'s output) parses to the
  table of spec 25 §1.1 (88 blocks, the eight attention indices, E / I per layer equal to 25a's facts sheet list); each
  refusal above on an in-memory edit names its key; `linears(l)` per block type; the `--layers N` truncation keeps
  `block[0..N)`.
- [ ] **Step 2: implement; PASS** (`ctest --preset mac-host -R '^bolt_test$'`). **Commit** `git commit -S -m "model: BoltDesc held to llm_config key by key - heterogeneous E / I, per-linear {S, layout} (spec 25b)"`.

### Task 3: the loader relaxations (shared code, exact)

**Files:**
- Modify: `src/loader/quant.{h,cc}`, `tests/loader/quant_test.cc`, docs/13 (a "Nemotron 3.5 Bolt (spec 25b)" section)

- [ ] **Step 1: the failing tests** (`quant_test.cc`): (a) SergiioB's `quantization_config` (typed in the test from the
  published file) parses; `dynamic_restating == 1`; (b) `+:` rules with `bits 8`, `group_size 64` under a g128 top level,
  `sym false`, or an extra key each throw naming the rule (the broken-MTP guard intact); (c) a GPTQ linear with BF16
  `.scales` converts exactly (values in f16's range) and throws naming the element for a value below 2^-17 or above 65504;
  (d) a GPTQ linear with negative scales: `classify` + the g128 expansion keep the sign bit; `assert_quant_invariants` and
  `check_quant_scan` accept it; (e) every existing case unchanged.
- [ ] **Step 2: implement** (the narrowest change: the `+:` branch compares the rule's three values with the top level;
  the GPTQ scales branch routes BF16 through `bf16_to_f16_exact`); PASS; the whole loader suite green. **Commit**
  `git commit -S -m "loader: a restating '+:' dynamic rule accepted, BF16 GPTQ scales exact-or-refused, signed scales tested (spec 25b)"`.

### Task 4: the layouts and the host repack

**Files:**
- Create: `src/loader/bolt_layout.h`, `src/loader/bolt_repack.{h,cc}`, `tests/loader/bolt_repack_test.cc`,
  `tests/loader/bolt_synth_host_test.cc`
- Modify: `src/loader/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- `bolt_layout.h` (`loader::bolt`): `expert_up_bytes(I)` / `expert_dn_bytes(I)` (layout 1: I x 544 B each), `expert_block_bytes(I)`,
  `layer_expert_bytes(E, I)`, the dense linears' bytes per `{S, layout}` (`int4_layout0_bytes` / `int4_l1_bytes` from
  `int4_layout.h`), `router_bytes(E)` (gemv_bf16 tiles, E rows), `mamba_small_bytes` (conv taps fp32 [10240][4], bias,
  A / D / dt_bias fp32 [128], gated-norm weight fp32 [8192]), `norm_bytes`, `embed_tiled_bytes`, `head_int8_bytes`,
  `layer_weight_bytes(desc, l)`, `weight_bytes(desc, int8_head)`; `kConvRing` 16.
- `bolt_repack` (`loader::bolt`): per block, from `LinearSrc` (g128 expanded or g64 as shipped) to the descriptor's layout:
  `in_proj` (N 18560), `out_proj`; `q‖k‖v` fused (N 4608); `fc1‖shared_up` fused (N 6400), `fc2`, `shared_down`; every
  expert's up / down to layout-1 blocks at `base + id x block(I)`; the router to bf16 tiles; the kept tensors widened to
  fp32; `A = -exp(A_log)` with the function 25a pinned (torch's fp32 `exp`, `exp_torch` in `common/`).
- [ ] **Step 1: the failing tests:** `bolt_repack_test` - Review Focus 1 (one random matrix, both spellings, byte-identical
  buffers), Review Focus 3, every block base and expert block a multiple of 64 B, the bytes of spec 25 §3 re-derived word for
  word from the formulas (printed: routed experts 35,139,092,480 B, the rest per category) with the total never typed;
  `bolt_synth_host_test` - Task 1's two synthetics read through the served constructor's host half: every tensor consumed but
  the vision ones (counted), 0 unconsumed, the repacked words equal the shards' (sampled), both spellings to the same bytes.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "loader: Bolt layouts and host repack - one device layout from g128-expanded or native g64, heterogeneous expert blocks, 64 B aligned (spec 25b)"`.

### Task 5: the loader (device half) and the memory plan

**Files:**
- Create: `src/loader/bolt_loader.{h,cc}`, `src/runtime/bolt/bolt_sizes.{h,cc}`, `tests/loader/bolt_load_checkpoint_test.cc`
  (card), `tests/runtime/bolt_plan_test.cc`
- Modify: `src/loader/CMakeLists.txt`, `src/runtime/CMakeLists.txt`, `tests/CMakeLists.txt`, `cmake/host_only.cmake` (the card
  test excluded on the Mac)

**Interfaces:**

```cpp
namespace runtime::bolt {
struct BoltReserve { size_t decode_scratch = 0, prefill_scratch = 0, drafter = 0, drafter_embed_replica = 0,
                     verify_slots = 0; };   // placeholders 25c / 25d / spec 26 fill; 0 = none
struct BoltPlan { std::vector<uint32_t> split; size_t max_len; std::vector<size_t> device_bytes; /* per category */ };
BoltPlan plan(const model::BoltDesc&, size_t max_len /* 0 = auto */, const std::vector<size_t>& device_free,
              const BoltReserve&, std::optional<uint32_t> split = {}, size_t reserve_bytes = 1.5 GB);
}
namespace loader {
struct BoltLoaded;   // per device: the layer weights it owns, the embedding (device 0), norm_f + head (the last device)
BoltLoaded load_bolt(const std::vector<l0::Device*>& devices, const std::string& snapshot, size_t max_len,
                     const runtime::bolt::BoltPlan& placement, bool int8_head, std::optional<uint32_t> layers_override);
}
```
The plan: per device its layers' weights (`layer_weight_bytes`), KV for its attention layers at `max_len` (8,192 B a position
in all, split by layer), the SSM state (one slot; `verify_slots` adds spec 26's), the conv rings, the reserve; split by bytes
(`pp_balance`; 48 on the real model, derived) or given; `--max-len auto` the largest multiple of 256 <= 262144 that fits both
devices (`require_fits` before the first allocation, naming the device and the bytes); one device only for synthetic /
truncated checkpoints that fit.
- [ ] **Step 1: the failing tests:** `bolt_plan_test` (host): the real descriptor on two 32 GB cards places split 48 (19.93 /
  20.54 GB of weights, derived), 262144 positions fit with the drafter and replica reserved on device 1; one 32 GB card is
  refused naming the bytes and Decision 4; a 24 GB pair reaches a smaller auto length, printed; the plan's bytes equal the
  loader's per-device allocation formula. `bolt_load_checkpoint_test` (card): the N 12 synthetic in both spellings uploads on
  one card, read-backs of sampled words equal the host repack, the plan equals the allocation; the real g128 checkpoint on two
  cards (SKIP without it).
- [ ] **Step 2: implement; PASS on the Mac (host tests); the card test compiles (Level Zero syntax). Commit**
  `git commit -S -m "loader: load_bolt on one or two devices, the vision skip by name; runtime::bolt::plan - split by bytes, 262144 auto, spec 26's reserve (spec 25b)"`.

### Task 6: docs, the box queue row

**Files:**
- Modify: spec 25 ("25b as built"), this plan's status, `docs/superpowers/plans/box-validation-queue.md` (row 59),
  `tools/box_validate/stages.sh` (`row 59`), `tools/box_validate/data.sh` (`have bolt_synth`)

- [ ] **Step 1:** "25b as built" (the measured host numbers, every departure with its cost if wrong); the queue row:

```text
row 59 "spec 25b - Nemotron 3.5 Bolt: BoltDesc, load_bolt (g128-expanded and native g64 into one layout, the '+:' / BF16-scale / signed-scale relaxations), synthetic checkpoints, the two-card plan (plan 25b)"
stage r59.synth      59 default cpu  oracle_image          -          "make_synth.py --layers 12 in both spellings; check.py ACCEPTED"
stage r59.load_synth 59 default gpu  bolt_synth            r59.synth  "bolt_load_checkpoint_test on the synthetics (one card): read-backs, plan == allocation"
stage r59.load_real  59 optin   gpu  bolt_g128             -          "load SergiioB's g128 on two cards (split 48): read-backs sampled, plan == allocation, time and RSS"
stage r59.load_rtn   59 optin   gpu  bolt_rtn              -          "load our native-g64 RTN export (25q) on two cards: the same device bytes as g128-expanded where the weights agree (sampled)"
```
- [ ] **Step 2: commit** `git commit -S -m "docs: spec 25 (25b as built), box queue row 59 - the Bolt loader"`.

**Gate for the plan:** Mac - `tools/mac_check.sh --base main` green (every host test incl. the existing loader suite;
kernel_cmdlines +0 / -0 / ~0); both synthetics `ACCEPTED` and read with 0 unconsumed. Box - the synthetics and SergiioB's
checkpoint load on their cards with the plan equal to the allocation.

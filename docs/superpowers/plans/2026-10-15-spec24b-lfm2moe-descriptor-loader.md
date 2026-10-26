# Spec 24b - LFM2.5-24B-A2B: the descriptor, the loader, synthetic checkpoints, the memory plan

> DROPPED (operator, 2026-10-27): support removed from the engine in 77d07ad; kept for the record.
>
> Reusable later (revive from git: `git show c57e08e:<path>` - c57e08e, this branch's parent on main, is the last
> commit that has all of it):
> - the MoE router: `src/kernels/lfm2moe/lfm_moe.cl` (`lfm_router`; `lfm_route` - sigmoid, the expert bias for the selection only, renormalised weights; `lfm_moe_gate_up` / `lfm_moe_down`), `src/kernels/lfm2moe/lfm_pf_moe.cl`, `src/kernels/lfm2moe/lfm_expert.h`; twins `tests/kernels/lfm2moe_ref.h`, `tests/kernels/lfm2moe_pf_ref.h`; tests `tests/kernels/lfm2moe_ref_test.cc`, `tests/kernels/lfm2moe_kernels_test.cc`, `tests/kernels/lfm2moe_pf_kernels_test.cc`
> - the pythonic tool-call parser: `src/server/toolcall_lfm2.h`, `src/server/toolcall_lfm2.cc`, `tests/server/toolcall_lfm2_test.cc`; `tools/toolcall/score.py --format pythonic`, `tools/toolcall/lfm2moe_rules.cc`
> - the rest: the short conv `src/kernels/lfm2moe/lfm_conv.cl` / `lfm_pf_conv.cl`, head_dim-64 attention `src/kernels/lfm2moe/lfm_attn.cl` / `lfm_pf_attn.cl`, `src/model/lfm2moe.h`, `src/model/lfm2moe.cc`, `src/loader/lfm2moe_*`, `src/runtime/lfm2moe/` (`lfm_prep.cl` still live in `src/kernels/bolt/`)

**Status (2026-10-17): Tasks 1-6 built on the Mac; box queue row 50 added.** Branch `spec24b-lfm2moe-loader`.
Everything host-side is green (`tools/mac_check.sh`); nothing has run on a card (24b's own row, and 24q's RTN /
AutoRound export, are the box's job). See spec 24 "24b as built" for the measured facts.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the model as data and its bytes on the card: `model::Lfm2MoeDesc` (the published model, held to config.json key by key), `loader::load_lfm2moe` reading our export (24q's RTN or AutoRound, either expert spelling) into per-expert int4 layout-1 blocks, int4 dense / conv / attention linears, bf16 router tiles, the fp32 small blocks (norms, conv taps, `expert_bias`), the embedding as bf16 tiles and the int8 head built from it (tied); the synthetic real-width checkpoints; the memory plan (`--max-len auto`, one card, the two-card split).

**Architecture:** spec 24 §3, §4.4, §5, §6, §8 24b; Kolibri's pattern (plan 20c Tasks 1-3, plan 21b): a descriptor, a header-only layout (the one formula the planner and the loader share), a host-only repack half, a loader library, a host-only planner - beside `qwen3_5`, K2, Kolibri and `qwen4_exp`, their files untouched. Reused unchanged: `loader::resolve_snapshot`, `QuantConfig::parse`, `SafetensorsSet`, `assert_quant_invariants`, `LinearSrc::classify`, `DeviceWeight`, `quantise_int8_tiled` (`src/loader/lm_head_int8.h:44-47`), `common::repack_int4_layout1_cols`, `cols_interleave16`, `repack_bf16_tiled` (`src/common/repack.h`), `make_small_layout` (`src/loader/small_layout.h`) where its shape fits, spec 16b's `runtime::pp_balance` (`src/runtime/pipeline_plan.h`).

**Tech Stack:** C++17, Level Zero (the upload only), CMake / ctest; Python 3 + torch + safetensors in `agnes-ref-img` with 24a's 5.19.0 site for the synthetic checkpoints.

**Spec:** `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` (§1, §3, §4.1 (the arm-`c` permutation), §4.4, §5, §6, §8 24b, §10 decisions 9, 10). Facts: 24a's `docs/probe-lfm2moe-<date>.md` (names, shapes, `vocab_used`), 24q Task 1 (the export's expert names and `quantization_config`). Precedents: `src/model/kolibri1.{h,cc}`, `src/loader/kolibri1_{layout.h,repack.*,loader.*,rope.cc}`, `src/runtime/kolibri/kolibri_sizes.{h,cc}`, `src/model/qwen4exp.{h,cc}`, `src/loader/qwen4exp_{layout.h,repack.*,loader.*}`, `tools/quantize/kolibri/make_synth.py`, `tests/model/kolibri1_test.cc`, `tests/loader/kolibri1_repack_test.cc`, `tests/runtime/kolibri_plan_test.cc`.

## Dependencies and branch points

- **Plan 24a merged** (`lfm2moe_ref.expected_names` both forms and spellings, the facts sheet). **Plan 24q Tasks 1-2 merged** (`rtn_export.py`'s packer - `make_synth.py` packs with it - and `check.py`). Real-weight loads need 24q's RTN export on the box (`have lfm2moe_rtn`); they SKIP 77 until then.
- **One format, two expert spellings:** the loader reads `feed_forward.experts.E.{w1,w3,w2}` or `feed_forward.experts.E.{gate_proj,up_proj,down_proj}` (24q Task 1 records which AutoRound writes), all-or-nothing per checkpoint (a mix is refused naming the first odd tensor). A bf16 expert, a bf16 linear the export should have packed, or a g128 group is refused by name: the bf16 original is the reference's input, never the engine's.
- **The tied head:** no `lm_head.*` is read; if an export carries `lm_head.weight` it must equal `embed_tokens.weight` byte for byte (checked, then skipped) - anything else is refused naming both.
- **Decision 9 (the config file):** `tests/model/lfm2moe/config.json` is the original's `config.json` copied unchanged, beside a `tests/model/lfm2moe/SOURCE` file naming `LiquidAI/LFM2.5-24B-A2B@1769516a` and the LFM Open License v1.0; if the operator rules against copying it, `lfm2moe_test.cc` builds the same JSON in code and the file is not created.
- **Decision 10 (`--pp 2`):** the placement and the split are built here (cheap, host only); whether 24c's engine uses them is 24c's last task.

## Global Constraints

- Branch `spec24b-lfm2moe-loader` from main (after 24a); box tree automatic; `tools/box.env` copied if missing, never committed, never printed.
- **F0, nothing moves:** no kernel touched (`tools/kernel_cmdlines` +0 / -0 / ~0); `loader::load`, `load_k2`, `load_kolibri1`, `load_qwen4exp` and their files unchanged; every existing suite green.
- **One weight format** (the operator's rule): AutoRound W4A16 int4 g64 symmetric `auto_round:auto_gptq` (`w = (q - 8) x scale`, `qzeros` 0x77777777, no `g_idx` or the identity) for every `nn.Linear`; bf16 / fp32 as shipped for everything else. No conversion of any other format.
- The descriptor is held to config.json key by key; every refusal names the key or the tensor.
- Every byte figure below is **derived**; each test asserts its formula, not a constant typed from this plan.
- Mac checks: `tools/mac_check.sh --base main` (host tests, Level Zero syntax, kernel command lines unchanged). Python in `agnes-ref-img` (`--memory 8g --cpus 4`).
- No `rm -rf`. Signed commits; no merge, no push.

## Review Focus

1. **gate‖up is w1 then w3, interleaved in 16s.** The device expert block (and the dense gate‖up) is `cols_interleave16(w1, w3)` - gate tile t then up tile t (Kolibri's / Ornith's) - whichever spelling the export uses (`gate_proj` = `w1`, `up_proj` = `w3`, `down_proj` = `w2`: `Lfm2MoeMLP` M:123-128, the fusion order `conversion_mapping.py:1163-1167` w1 then w3). A test builds a checkpoint where w1 and w3 differ by a sign and checks the block, for both spellings.
2. **`in_proj`'s row order is B, C, x.** The GEMV's output columns 0..2047 are B, 2048..4095 C, 4096..6143 x (M:437-438); the arm-`c` permutation (spec 24 §4.1: 16-channel tiles `[B 16 | C 16 | x 16]`) is a second, separate layout whose inverse is tested to give the first back word for word.
3. **The tied head and the embedding are one tensor.** The embedding is uploaded once in `gemv_bf16`'s tiled layout (`repack_bf16_tiled`), the int8 head built from the same host bytes (`quantise_int8_tiled`, spec 9's rule); the row-from-tiles index the gather kernel will use (`lfm_embed_row_index(k, n)`) is defined here and tested against the row-major tensor for every k of 64 random ids including 0, 124894 and 127999.
4. **The plan refuses what does not fit, by bytes, before any device work, and `auto` stops at 131072.** Weights 13.39 GB + KV 20,480 B a position + the conv rings + the scratch terms 24c / 24d add; `--max-len auto` = the largest multiple of 256 that fits **and** ≤ `trained_max_len` (131072); an explicit `--max-len` above 131072 is refused naming the trained context.
5. **Names both ways.** Every tensor of the checkpoint is read exactly once or listed as skipped by design (the `lm_head.weight` duplicate; `--layers N`'s later layers on a synthetic); an unknown or a missing tensor throws naming it; `expert_bias` must be F32 [64], `conv.conv.weight` BF16 [2048, 1, 3].

---

### Task 1: the synthetic checkpoints (Python, no card)

**Files:**
- Create: `tools/quantize/lfm2moe/make_synth.py`
- Modify: `tools/quantize/lfm2moe/test_lfm2moe_quant.py` (24q's), `tools/quantize/README.md`, `tools/box_validate/data.sh` (`have lfm2moe_synth`)

**Interfaces:**
- Consumes: `tools/oracle/lfm2moe_ref.py` (`config`, `expected_names`), `tools/quantize/lfm2moe/rtn_export.py`'s packer, `check.py`.
- Produces: `make_synth.py <out> --layers N [--expert-names w|proj] --tokenizer <LFM2.5 snapshot> [--seed 0]` - a real-width checkpoint: the original's `config.json` with `num_hidden_layers = N` and `layer_types[:N]` (N in 3..40; the default 4 holds two dense conv layers, the first attention + MoE layer and a conv + MoE layer - every layer kind), `quantization_config` as 24q's export writes it; weights drawn at the real widths (normal, std 0.02 for linears, norms ~1, conv taps ~N(0, 0.3), `expert_bias` ~N(0, 0.05) F32, embedding std 0.02) and packed by `rtn_export.py`'s packer; one layer generated, packed and written at a time (peak RAM ~2 GB, ESTIMATED); `model.safetensors.index.json`; the tokenizer, template, generation config and LICENSE copied from `--tokenizer`. Size at N = 4: ~1.28 GB (derived: the two dense conv layers 47.3 MB each, the attention + MoE layer 326.7 MB, the conv + MoE layer 330.0 MB, the embedding 524.3 MB).
- [x] **Step 1: the failing tests:** `test_synth_names` (both spellings: the file's names equal `expected_names(cfg, "int4", spelling)` both ways); `test_synth_checked` (`check.py` `ACCEPTED`); `test_synth_loads_in_reference` (`lfm2moe_ref.build_streamed(weights="int4")` runs 16 ids through it); `test_gate_up_sign` (Review Focus 1 at the source: a `--plant-sign` option writes w3 = -w1 for expert 0 of the first MoE layer).
- [x] **Step 2: implement; PASS.** Then one synthetic per spelling on the Mac (`/data/lfm2moe-synth/{w,proj}`, N 4) and `check.py`: `ACCEPTED` twice.
- [x] **Step 3: commit** `git commit -S -m "quantize: LFM2.5-24B-A2B synthetic real-width checkpoints in our format, both expert spellings (spec 24b)"`.

### Task 2: the model table

**Files:**
- Create: `src/model/lfm2moe.h`, `src/model/lfm2moe.cc`, `tests/model/lfm2moe_test.cc`, `tests/model/lfm2moe/config.json` + `tests/model/lfm2moe/SOURCE` (decision 9), `tests/model/lfm2moe/synth4.json` (Task 1's synthetic config)
- Modify: `src/model/CMakeLists.txt` (library `b70_lfm2moe_model`), `tests/CMakeLists.txt` (a block `# ==== Spec 24b: LFM2.5-24B-A2B descriptor and loader (label lfm2moe) ==== (begin)` at the end)
- Test: `lfm2moe_test`

**Interfaces:**
- Consumes: `model::GemvShape`, `model::WeightKind`, `model::Fuse` (`src/model/qwen35.h`), `common::json::Value`.
- Produces (every later task and 24c-24e):

```cpp
namespace model {
enum class LfmKind { Conv, Attention };
enum class LfmExpertNames { W, Proj };                 // experts.E.{w1,w3,w2} | {gate_proj,up_proj,down_proj}
enum class LfmLinearId { InProj, OutProj, Qkv, OProj, DenseGateUp, DenseDown, Router, Head, kCount };
struct LfmLinear { LfmLinearId id; GemvShape shape; WeightKind kind; Fuse fuse; std::vector<std::string> parts; };
struct Lfm2MoeDesc {
  std::string name, architecture, model_type, prefix;      // "lfm2.5-24b-a2b", "Lfm2MoeForCausalLM", "lfm2_moe", "model."
  uint32_t layers = 0, hidden = 0, vocab = 0, vocab_used = 0, trained_max_len = 0;   // 40, 2048, 128000, 125017, 131072
  uint32_t q_heads = 0, kv_heads = 0, head_dim = 0;        // 32, 8, 64
  double rope_theta = 0;                                   // 1e7 (rope_type "default" only)
  uint32_t conv_taps = 0;                                  // 3 (conv_L_cache; conv_bias false only)
  uint32_t dense_layers = 0, dense_inter = 0;              // 2, 11776
  uint32_t experts = 0, top_k = 0, moe_inter = 0;          // 64, 4, 1536 (use_expert_bias, norm_topk_prob true only;
                                                           //   routed_scaling_factor 1.0 only)
  float norm_eps = 0;                                      // 1e-5
  uint32_t bos = 0, eos = 0, pad = 0;                      // 124894, 124900, 124893
  std::vector<LfmKind> kinds;                              // from layer_types; must equal the published prefix [0, layers)
  bool is_attention(uint32_t l) const;  bool is_dense(uint32_t l) const;
  uint32_t attn_index(uint32_t l) const;  uint32_t conv_index(uint32_t l) const;   // dense index of that kind
  uint32_t attn_layers() const;  uint32_t conv_layers() const;  uint32_t moe_layers() const;
  std::vector<LfmLinear> linears(uint32_t l) const;       // the GEMVs of layer l, shapes from the fields
};
Lfm2MoeDesc lfm2moe();                                     // the published model
void check_lfm2moe_config(const common::json::Value& config);   // throws std::invalid_argument naming the key
Lfm2MoeDesc lfm2moe_desc(const common::json::Value& config, uint32_t vocab_used);  // vocab_used from tokenizer.json
bool is_lfm2moe(const common::json::Value& config);        // model_type == "lfm2_moe"
struct LfmPlacement {                                      // layers [0, split) on device 0, [split, layers) on 1
  uint32_t devices = 1, split = 0, layers = 0;
  uint32_t first(uint32_t dev) const; uint32_t end(uint32_t dev) const; uint32_t device_of(uint32_t l) const;
  static LfmPlacement one(const Lfm2MoeDesc& d); static LfmPlacement two(const Lfm2MoeDesc& d, uint32_t split);
};
void validate(const LfmPlacement& p, const Lfm2MoeDesc& d);
}
```

- [x] **Step 1: the failing test** `lfm2moe_test.cc`: `lfm2moe_desc(config.json)` == `lfm2moe()` field by field; `kinds` = the 30 / 10 pattern (attention at `l % 4 == 2`, layer 39 conv); `is_dense` true for 0 and 1 only; every `linears(l)` shape (in_proj {2048, 6144}, out_proj / o_proj {2048, 2048}, q‖k‖v {2048, 3072}, dense gate‖up {2048, 23552} fused `Interleave16`, dense down {11776, 2048}, router {2048, 64} bf16); refusals, each naming its key: another `model_type`, `conv_bias` true, `use_expert_bias` false, `norm_topk_prob` false, `routed_scaling_factor` 2.0, `rope_type` other than default, `tie_word_embeddings` false, a `layer_types` entry off the published pattern, a width other than the published one (hidden, heads, head_dim, experts, top_k, intermediates, vocab), `num_dense_layers` 1; the synthetic (`synth4.json`) accepted with `layers` 4; `validate` refuses a split of 0 or 40.
- [x] **Step 2: register, run, FAIL; implement; PASS** (`ctest --preset mac-host -R '^lfm2moe_test$'`). **Commit** `git commit -S -m "model: Lfm2MoeDesc - LFM2.5-24B-A2B held to config.json key by key, the layer kinds, the GEMV shapes, the placement (spec 24b)"`.

### Task 3: the layouts and the host repack half

**Files:**
- Create: `src/loader/lfm2moe_layout.h`, `src/loader/lfm2moe_repack.h`, `src/loader/lfm2moe_repack.cc`, `src/loader/lfm2moe_rope.cc`, `tests/loader/lfm2moe_repack_test.cc`, `tests/loader/lfm2moe_rope_test.cc`, `tools/oracle/lfm2moe_rope_fixture.py` -> `tests/loader/lfm2moe_rope_fixture.h` (generated, committed)
- Modify: `src/loader/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces (`lfm2moe_layout.h`, header-only; 24c's kernels' host half reads the same constants):

```cpp
namespace loader::lfm {
inline constexpr size_t kTileBytes = 544;                            // layout 1: 128 u32 nibbles + 8 u32 f16 scales
size_t int4_l1_bytes(uint32_t K, uint32_t N);                         // N/16 x K/64 x 544
size_t expert_gate_up_bytes(const model::Lfm2MoeDesc&);               // {2048, 3072}: 3,342,336
size_t expert_down_bytes(const model::Lfm2MoeDesc&);                  // {1536, 2048}: 1,671,168
size_t expert_gate_up_offset(const model::Lfm2MoeDesc&, uint32_t e); // e x 3,342,336 - THE expert address (24c's macro)
size_t expert_down_offset(const model::Lfm2MoeDesc&, uint32_t e);    // e x 1,671,168
size_t moe_layer_bytes(const model::Lfm2MoeDesc&);                    // 64 x 5,013,504 = 320,864,256
size_t router_bytes(const model::Lfm2MoeDesc&);                       // bf16 tiles {2048, 64}: 262,144
struct SmallBlock { size_t operator_norm, ffn_norm, conv_taps, q_norm, k_norm, expert_bias, bytes; };  // fp32 offsets
SmallBlock small_block(const model::Lfm2MoeDesc&, uint32_t layer);    // conv: 2 x 8192 + 24,576; attention: + 2 x 256;
                                                                      // MoE layers + 256 (the bias)
size_t embed_tiled_bytes(const model::Lfm2MoeDesc&);                  // bf16 {2048, 128000}: 524,288,000
size_t head_int8_bytes(const model::Lfm2MoeDesc&);                    // lm_head_int8_bytes(2048, 128000): 262,656,000
size_t lfm_embed_row_index(uint32_t K, uint32_t k, uint32_t n);       // element (k, n) in gemv_bf16's tiles
size_t rope_table_bytes(uint32_t max_len);                            // [max_len][2][32] fp32 (bf16 values)
size_t kv_bytes_per_position(const model::Lfm2MoeDesc&);              // 10 x 8 x 64 x 2 x 2 = 20,480
size_t conv_ring_bytes(const model::Lfm2MoeDesc&);                    // 30 x 16 x 2048 x 2 = 1,966,080
inline constexpr uint32_t kConvRing = 16;
size_t layer_weight_bytes(const model::Lfm2MoeDesc&, uint32_t layer); // what the split balances
size_t weight_bytes(const model::Lfm2MoeDesc&, bool int8_head);       // 13.39 GB with the int8 head (derived)
}
```

- Produces (`lfm2moe_repack.h`, host only): `lfm_expected_names(desc, LfmExpertNames)`, `detect_expert_names(const SafetensorsSet&)` (all-or-nothing), `repack_expert(w1, w3, w2 int4 GPTQ) -> {gate_up layout-1, down layout-1}` (`cols_interleave16` then `repack_int4_layout1_cols`), `repack_in_proj(qweight, scales, bool conv_tiles)` (Review Focus 2's permutation when `conv_tiles`), `widen_bf16(const uint16_t*, size_t) -> std::vector<float>` (norms, taps, exact), `lfm_rope_table(desc, max_len)` (`kolibri1_rope.cc`'s method at dim 64, θ 1e7).

- [x] **Step 1: the failing tests** `lfm2moe_repack_test.cc`: every byte formula above (asserted, printed); `repack_expert` on a crafted expert (Review Focus 1: w3 = -w1 lands in the up tiles; both spellings through `detect_expert_names`; a mixed checkpoint refused); `repack_in_proj` with `conv_tiles` false = `repack_int4_layout1_cols` of the GPTQ tensor, with `conv_tiles` true its inverse permutation gives the false layout back word for word (Review Focus 2); `lfm_embed_row_index` against the row-major tensor (Review Focus 3); `small_block` offsets disjoint and 64-byte aligned. `lfm2moe_rope_test.cc`: the table equals transformers 5.19.0's `Lfm2MoeRotaryEmbedding` cos / sin (bf16) bitwise at positions 0, 1, 2, 4095, 131071 and 1000 random ones (the fixture, generated in `agnes-ref-img` with 24a's site).
- [x] **Step 2: register, FAIL; implement; PASS. Commit** `git commit -S -m "loader: lfm2moe layouts and the host repack half - per-expert layout-1 blocks, in_proj B/C/x (and the conv-tile permutation), the tied embedding tiles, the RoPE table (spec 24b)"`.

### Task 4: the loader

**Files:**
- Create: `src/loader/lfm2moe_loader.h`, `src/loader/lfm2moe_loader.cc`, `tests/loader/lfm2moe_synth_host_test.cc`, `tests/loader/lfm2moe_load_checkpoint_test.cc`
- Modify: `src/loader/CMakeLists.txt` (library `b70_lfm2moe_loader`), `tests/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
namespace loader {
struct LfmLayer {                       // one decoder layer on its device
  DeviceWeight in_proj, out_proj;       // conv layers (in_proj in the B/C/x layout, or conv tiles when asked)
  DeviceWeight qkv, o_proj;             // attention layers (q||k||v fused, N 3072)
  DeviceWeight dense_gate_up, dense_down;   // layers 0-1
  l0::Mem* experts_gate_up = nullptr;   // MoE layers: 64 x 3,342,336 B (one allocation per layer)
  l0::Mem* experts_down = nullptr;      // 64 x 1,671,168 B
  DeviceWeight router;                  // bf16 tiles {2048, 64}
  l0::Mem* small = nullptr;             // lfm::small_block(desc, l)
};
struct LfmLoaded {
  model::Lfm2MoeDesc desc;  model::LfmPlacement place;  model::LfmExpertNames names;
  std::vector<LfmLayer> layers;
  l0::Mem* embed_tiles = nullptr;       // on device 0 (the gather) - and, as the bf16 head, on the last device
  DeviceWeight head_int8;               // on the last device, when int8
  l0::Mem* final_norm = nullptr;        // embedding_norm, fp32, last device
  l0::Mem* rope = nullptr;              // per device that holds an attention layer
  std::vector<std::string> skipped;     // by design (the tied lm_head duplicate, --layers' later layers)
};
struct LfmLoadOptions { bool int8_head = true; bool conv_tiles = false; uint32_t max_len = 0; uint32_t layers = 0; };
LfmLoaded load_lfm2moe(std::vector<l0::Context*> devices, const std::string& snapshot, const LfmLoadOptions& o,
                       const model::LfmPlacement& place);
}
```

- [x] **Step 1: the failing tests:** `lfm2moe_synth_host_test` (host, over Task 1's synthetics when present, SKIP 77 otherwise): every tensor read once or listed skipped (Review Focus 5); the repacked expert / in_proj / embedding bytes equal Task 3's functions over the file's tensors; the int8 head = `quantise_int8_tiled` of the embedding; an export with `lm_head.weight` equal to the embedding skips it, one with a different one is refused naming both. `lfm2moe_load_checkpoint_test <snapshot>` (card, label `checkpoint;lfm2moe`, SKIP 77 without data): the load on one device, `memory_use` per component equal to Task 5's plan, a sampled expert block read back equal to the host repack, both spellings; on two devices with `two(desc, 20)`.
- [x] **Step 2: register, FAIL; implement; PASS (host); the card test is the box's** (row 50). **Commit** `git commit -S -m "loader: load_lfm2moe - both expert spellings, per-expert layout-1 blocks, the tied int8 head from the embedding tiles, fp32 small blocks, placement over two cards (spec 24b)"`.

### Task 5: the memory plan

**Files:**
- Create: `src/runtime/lfm2moe/lfm2moe_sizes.h`, `src/runtime/lfm2moe/lfm2moe_sizes.cc`, `src/runtime/lfm2moe/CMakeLists.txt` (library `b70_lfm2moe_plan`, host only), `tests/runtime/lfm2moe_plan_test.cc`
- Modify: `src/runtime/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces (24c adds the decode scratch and the launch counts, 24d the prefill scratch):

```cpp
namespace runtime::lfm2moe {
struct Plan { size_t weights, kv, conv_rings, rope, decode_scratch, prefill_scratch, reserve; size_t total() const;
              uint32_t max_len; };
Plan plan(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, uint32_t max_len, bool int8_head);
uint32_t max_len_auto(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, size_t device_bytes,
                      double reserve_gb, bool int8_head);          // multiple of 256, <= trained_max_len
void require_fits(const Plan& p, size_t device_bytes);              // throws naming every term
uint32_t pp_split(const model::Lfm2MoeDesc& d, bool int8_head);     // runtime::pp_balance over layer_weight_bytes
std::string describe(const Plan& p);
}
```

- [x] **Step 1: the failing test** `lfm2moe_plan_test`: one card, int8 head: weights 13.39 GB (the formula), KV 2.684 GB at 131072, the total under a 32 GiB device with the 1.5 GB reserve; `max_len_auto` = 131072 on a 32 GiB device (capped by the trained context, Review Focus 4) and the largest fitting multiple of 256 on a crafted 15 GiB device; an explicit 262144 refused naming `max_position_embeddings`; the bf16-head plan (no int8 copy: the embedding tiles serve as the head); `pp_split` balanced within one layer's bytes; synthetic N = 4 plans.
- [x] **Step 2: register, FAIL; implement; PASS. Commit** `git commit -S -m "runtime: the LFM2.5-24B-A2B memory plan - one card at 131072, max-len auto capped at the trained context, the two-card split (spec 24b)"`.

### Task 6: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` ("24b as built"), this plan's status line, `docs/13-loader.md` (an "LFM2.5-24B-A2B" paragraph: the two expert spellings, the tied head from the embedding tiles), `docs/superpowers/plans/box-validation-queue.md` (row 50), `tools/box_validate/stages.sh` (`row 50`)

- [x] **Step 1: Mac gate:** `tools/mac_check.sh --base main` exit 0 (host PASS incl. `lfm2moe_test`, `lfm2moe_repack_test`, `lfm2moe_rope_test`, `lfm2moe_synth_host_test`, `lfm2moe_plan_test`; Level Zero syntax; kernel_cmdlines +0 / -0 / ~0).
- [x] **Step 2: the queue row 50** (`test_box_validate.py` passes; `--dry-run --only r50` prints it):

```text
row 50 "spec 24b - LFM2.5-24B-A2B: the descriptor, the loader (both expert spellings, the tied int8 head), the memory plan (plan 24b)"
stage r50.host       50 default cpu -                         -        "host: lfm2moe_test, lfm2moe_repack_test, lfm2moe_rope_test, lfm2moe_plan_test, lfm2moe_synth_host_test"
stage r50.synth      50 default cpu oracle_image              -        "make_synth.py N 4, both spellings -> \$DATA/lfm2moe-synth/{w,proj}; check.py ACCEPTED"
stage r50.load_synth 50 default gpu lfm2moe_synth             r50.synth "lfm2moe_load_checkpoint_test on both synthetics: plan == allocation, blocks read back"
stage r50.load_real  50 default gpu lfm2moe_rtn               -        "lfm2moe_load_checkpoint_test on 24q's RTN export (one card; two with --split 20 when two GPUs)"
```
- [x] **Step 3: commit** `git commit -S -m "docs: spec 24 (24b as built), box queue row 50 - the LFM2.5-24B-A2B descriptor and loader"`.

**Gate for the plan:** Mac - every host test green, both synthetics `ACCEPTED`, kernel_cmdlines unchanged. Box - both synthetics and the RTN export load on one card with the plan equal to the allocation (two cards when present).

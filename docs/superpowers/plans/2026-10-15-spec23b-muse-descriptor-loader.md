# Spec 23b - Muse-Glimmer-30B: the descriptor, the loader, synthetic checkpoints, the memory plan

**Status (2026-10-15): built on the Mac (branch `spec23b-muse-loader`): Tasks 1-6 done, every host test green, the synthetic 2-layer checkpoint ACCEPTED and read by the C++ host half, the real -AutoRound checkpoint through the host half (1459 text, 809 vision skipped, 0 unconsumed); the card's loads are box queue row 43. As built: spec 23 §13.**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit after each step** (signed, `git commit -S`; a step that does not build or pass yet goes in as `WIP: ...`) - never leave a step's files uncommitted. When you stop for any reason, leave `HANDOFF-spec23b.md` (untracked, worktree root): the last step done, what is half-done, the next command.

**Goal:** Muse-Glimmer as data and bytes on the card: `model::MuseGlimmerDesc` held to `config.json` key by key; `loader::load_muse_glimmer` reading the GPTQ checkpoint (int4 g64 linears through the existing `LinearSrc::classify`, bf16 norms / embedding / head, spec 9's int8 head at load, the 809 vision tensors skipped by name, 0 unconsumed); `make_synth.py` (N layers at real width in exactly that format); the memory plan for one card and two (`--max-len auto`, `--pp 2` by bytes).

**Architecture:** spec 23 §2, §3, §5.1, §5.5. Kolibri's arrangement (spec 20 §11: `model::Kolibri1Desc`, `loader/kolibri1_layout.h` as the one byte formula for the planner, the loader and the kernels' host half, `loader::load_kolibri1`, `runtime::kolibri::kolibri_sizes`) for a dense model: no experts, no router - the layer is eight int4 linears (q, k, v, the attention gate_proj, o; gate, up, down - 23a's count) fused into four GEMVs, four centered norms, and the attention's ring or full cache.

**Tech Stack:** C++17, CMake / ctest, Level Zero (the load), Python 3 (`make_synth.py`, `check.py` in the oracle image with spec 21a's 5.19.0 site).

**Spec:** `docs/superpowers/specs/2026-10-15-spec23-muse-glimmer-design.md` (§1, §2, §3, §5.1, §5.4, §5.5, §11 decisions 6, 7). Precedents: plan 20c Tasks 1-2 (Kolibri's descriptor and loader), plan 21b (`2026-10-09-spec21b-qwen4exp-descriptor-loader-formats.md`: `make_synth.py`, `check.py`, the plan test), `src/loader/quant.h` (`QuantConfig::parse`, `LinearSrc`), `src/loader/lm_head_int8.h`, `src/runtime/pipeline_plan.h` (`pp_balance`).

## Dependencies and branch points

- **23a merged** (`muse_ref.expected_names`, `vision_names`, `muse_make_tiny.py --variant int4`; the facts sheet). If 23a is not merged, branch from its branch and say so in the status line.
- **No kernel** in this plan (`kernel_cmdlines` +0 / -0 / ~0). The device half is uploads and read-backs; the GEMV layouts are the existing `gemv.cl` layouts (0 / 1) whose host repack exists (`loader/loader.cc`).

## Global Constraints

- Branch `spec23b-muse-loader` from main; signed commits; no merge, no push. Box tree automatic; `tools/box.env` never read into a log, never committed. Every GPU command under `flock ~/b70-gpu.lock`, detached, polled.
- **Our format only:** int4 g64 symmetric GPTQ (`qweight` / `scales` f16 / `qzeros` 0x77777777 / identity `g_idx`). **Either Vishva007 repo** (23a's facts sheet, 2026-10-15: one export, sampled tensors byte-identical): `-AutoRound` (`quant_method` auto-round, `packing_format` auto_round:auto_gptq, NO `g_idx` - the repo the Mac downloaded, 1459 text names) or `-AutoRound-GPTQ` (`quant_method` gptq, `g_idx` shipped and checked `== i // 64`, 1875 text names); `expected_names` takes both (23a's `muse_ref.expected_names(tc, "autoround" | "gptq")`). Anything else (g128 aside, which `LinearSrc::classify` expands exactly) is refused by name; no new kernel family.
- Every buffer through `loader/muse_layout.h` and the planner; the plan runs before the first allocation.
- The vendored config: `tests/model/muse_glimmer/config.json` (+ `generation_config.json`) copied byte for byte from `meta-models/Muse-Glimmer-30B` at `a4e59da5` (Apache-2.0).
- No `rm -rf`. No third-party engine named.

## Review Focus

1. **The fused GEMVs' column order.** `q || gate || k || v` (8704 rows) and `gate || up` (39936) concatenate the checkpoint's tensors in exactly that order; a test reads back column 0, 4095, 4096, 8191, 8192, 8447, 8448, 8703 (and 0 / 19967 / 19968 / 39935) and compares with the dequantised source rows. An off-by-one here is a silent wrong model.
2. **The vision tensors are skipped by name, not by accident:** the consumed set is the 1875 text names exactly; a stray unknown text name throws; the 809 vision names are counted in the load line.
3. **`layer_rope_theta`'s switch rule** (spec 23 §1): non-zero entries must equal `rope_parameters.rope_theta`, zeros exactly at `layer_types == full_attention`; a config breaking either is refused naming the layer.
4. **The planner and the loader agree to the byte** (`muse_device_weight_bytes` = what the loader allocates, per device, for 1 / 4 / 52 layers and two placements), and `--max-len auto` reaches 131072 on one card with the drafter's reservation (decision 6, spec 23 §2).
5. **`make_synth.py`'s checkpoint is accepted by the same C++ host half that reads the real one** (names both ways, 0 unconsumed) - the Python writer and the C++ reader held to each other on real files (21b's `qwen4exp_synth_host_test` precedent).

---

### Task 1: `MuseGlimmerDesc`

**Files:**
- Create: `src/model/muse_glimmer.h`, `src/model/muse_glimmer.cc`, `tests/model/muse_glimmer_test.cc`, `tests/model/muse_glimmer/config.json`, `tests/model/muse_glimmer/generation_config.json`
- Modify: `src/model/CMakeLists.txt`, `tests/CMakeLists.txt` (or the model tests' CMake file - follow `kolibri1_test`'s registration)

**Interfaces:**
- Consumes: `model::GemvShape`, `WeightKind`, `Fuse` (`src/model/qwen35.h`), `common::json`.
- Produces:

```cpp
namespace model {
enum class MuseLinearId { QGKV, OProj, GateUp, Down, LmHead, kCount };
struct MuseLinear { MuseLinearId id; GemvShape shape; WeightKind kind; Fuse fuse;
                    std::vector<std::string> parts; };          // layer-relative prefixes, column order
struct MusePlacement {                                         // KolPlacement's form
  uint32_t devices = 1, split = 0, layers = 0;
  uint32_t first(uint32_t dev) const; uint32_t end(uint32_t dev) const;
  uint32_t device_of(uint32_t layer) const; uint32_t count(uint32_t dev) const;
};
struct MuseGlimmerDesc {
  uint32_t layers = 52, hidden = 6656, q_heads = 32, kv_heads = 2, head_dim = 128,
           inter = 19968, vocab = 202048, window = 2048, max_pos = 131072;
  float rms_eps = 1e-5f, post_eps = 1e-8f, qk_scale = 3.87f, rope_theta = 5e5f,
        out_mult = 0.19611613513818404f, softcap = 20.f;
  uint32_t bos = 200000, pad = 200018; std::vector<uint32_t> eos = {200001, 200008};
  std::vector<bool> full;                       // per layer: full attention (NoPE)
  uint32_t full_layers() const; uint32_t sliding_layers() const;
  uint32_t qgkv_rows() const { return 2 * q_heads * head_dim + 2 * kv_heads * head_dim; }  // 8704
  MuseLinear linear(MuseLinearId id) const;
};
MuseGlimmerDesc muse_glimmer();                                     // the published model
MuseGlimmerDesc check_muse_config(const common::json::Value& config_json, uint32_t layers_override = 0);
bool is_muse_model_type(const std::string& model_type);             // "muse_glimmer" | "muse_glimmer_text"
}
```

- [x] **Step 1: the failing test** `muse_glimmer_test`: `muse_glimmer()` equals `check_muse_config(tests/model/muse_glimmer/config.json)` field by field; `full` is true exactly at `l % 4 == 3` (13 of 52); `qgkv_rows() == 8704`; `linear(QGKV).parts == {"self_attn.q_proj", "self_attn.gate_proj", "self_attn.k_proj", "self_attn.v_proj"}`, `GateUp == {"mlp.gate_proj", "mlp.up_proj"}`; refusals, each naming the key and value, on in-memory edits of the vendored config: `head_dim` 64, `num_key_value_heads` 4 (the kernels' GQA 16 is a define - refuse until built), `hidden_activation` gelu, `tie_word_embeddings` true, a non-zero `layer_rope_theta` != the global theta, a zero at a sliding layer, `rope_type` not default, `attention_bias` true, `final_logit_softcapping` null (the host cap needs it), `vocab_size` changed; `layers_override` 4 keeps `full = {0,0,0,1}`; `is_muse_model_type`.
- [x] **Step 2: run, expect FAIL** (`ctest --test-dir build -R muse_glimmer_test --output-on-failure`: target not found).
- [x] **Step 3: implement**, PASS. **Commit** `model: MuseGlimmerDesc held to config.json key by key (spec 23b)`.

### Task 2: the layout header (bytes, the one formula)

**Files:**
- Create: `src/loader/muse_layout.h`, `tests/loader/muse_layout_test.cc`

**Interfaces:**
- Produces:

```cpp
namespace loader {
struct MuseLayerBytes { size_t qgkv = 0, oproj = 0, gate_up = 0, down = 0, norms = 0;
                        size_t total() const; };
MuseLayerBytes muse_layer_bytes(const model::MuseGlimmerDesc& d);     // int4 GEMV layouts + 4 fp32 norms
size_t muse_embed_bytes(const model::MuseGlimmerDesc& d);             // vocab x hidden x 2
size_t muse_lm_head_bytes(const model::MuseGlimmerDesc& d, bool int8); // lm_head_int8_bytes or bf16
size_t muse_final_norm_bytes(const model::MuseGlimmerDesc& d);
size_t muse_ring_slots();                                             // 4096 (spec 23 §5.2)
size_t muse_ring_bytes_per_layer(const model::MuseGlimmerDesc& d);     // 4096 x kv x 128 x 2 x 2
size_t muse_kv_bytes_per_position(const model::MuseGlimmerDesc& d);   // full layers only: 13,312
size_t muse_device_weight_bytes(const model::MuseGlimmerDesc& d, const model::MusePlacement& p,
                                uint32_t dev, bool int8_head);       // embed on dev 0, head on the last
size_t muse_layer_read_per_token(const model::MuseGlimmerDesc& d);
}
```

- [x] **Step 1: failing test** printing and asserting spec 23 §2's derived numbers from the formulas: per-layer int4 (qweight + scales) 257,081,344 B ± the layout's padding (assert the formula, print the value), 52 layers 13.368 GB, int8 head 1,345,639,680 B, embed 2,689,662,976 B, KV 13,312 B a position, ring 4,194,304 B a sliding layer (39 -> 163,577,856 B), read per token 14.72 GB (derived).
- [x] **Step 2: implement** on `kol_layout0_bytes` / `kol_layout1_bytes`' formulas (shared, not copied: if they are Kolibri-local, move them to a common header with a byte-identical check - Kolibri's own test must stay green), PASS. **Commit** `loader: muse_layout.h - every Muse-Glimmer allocation's bytes (spec 23b)`.

### Task 3: the loader (host half, then the device half)

**Files:**
- Create: `src/loader/muse_loader.h`, `src/loader/muse_loader.cc`, `src/loader/muse_repack.h`, `src/loader/muse_repack.cc`, `tests/loader/muse_repack_test.cc`, `tests/loader/muse_load_checkpoint_test.cc`
- Modify: `src/loader/CMakeLists.txt`, the loader tests' CMake

**Interfaces:**
- Consumes: `loader::QuantConfig::parse`, `LinearSrc::classify` (the GPTQ / g128 / compressed-tensors forms - only g64 GPTQ is expected), `quantise_row_int8` (`lm_head_int8.h`), the existing GEMV repack (layout 0 / 1 tiles), `Safetensors` (`loader/safetensors.h`).
- Produces:

```cpp
namespace loader {
struct MuseHostLayer {               // one layer's device-form bytes, host side (tests read them)
  std::vector<uint8_t> qgkv, oproj, gate_up, down;   // the GEMV layouts, columns fused in Desc order
  std::vector<float> in_norm, post_attn, pre_ff, post_ff;   // fp32 copies of the bf16 weights
};
struct MuseCheckpoint {              // the host half: open, check, repack one layer at a time
  explicit MuseCheckpoint(const std::string& snapshot, uint32_t layers_override = 0);
  const model::MuseGlimmerDesc& desc() const;
  MuseHostLayer layer(uint32_t l) const;
  const uint16_t* embed() const; const uint16_t* lm_head() const; std::vector<float> final_norm() const;
  size_t vision_skipped() const;     // 809 on the real checkpoint
  std::vector<std::string> unconsumed() const;   // empty or the load throws
};
struct MuseWeights;                  // device allocations per placement (l0::Mem / DeviceWeight)
MuseWeights load_muse_glimmer(l0::Context& ctx0, l0::Context* ctx1, const std::string& snapshot,
                              const model::MusePlacement& p, bool int8_head, uint32_t layers_override = 0);
}
```

- [x] **Step 1: failing host test** `muse_repack_test` over 23a's tiny `int4` variant, and over a 2-layer real-width checkpoint written by Task 4's `make_synth.py` when present (SKIP 77 otherwise): Review Focus 1's column read-backs (every sampled weight `(q - 8) x scale` of the source row), Review Focus 2 (names both ways: consumed = `expected_names(gptq)`, the vision names counted; one extra text name throws naming it), the data asserts (a `qzeros` word != 0x77777777 throws naming the tensor and index; a non-identity `g_idx` throws), the norms' fp32 copies bit-exact widenings, refusals by name (`desc_act` true, group 32, bits 8, asym).
- [x] **Step 2: implement the host half**, PASS. **Commit** `loader: Muse-Glimmer host half - the fused GEMVs, the norms, the vision skip, the data asserts (spec 23b)`.
- [x] **Step 3: the device half** - `load_muse_glimmer`: the plan first (Task 5's `require_fits`), then per layer straight onto its device (`device_of`), the embedding on device 0, the final norm and the head (int8 built on the host with `quantise_row_int8`, 0.35-0.75 s on 44 threads for Qwen's - recorded here) on the last device; a load line `Muse-Glimmer-30B: 52 layers (int4 g64), 809 vision tensors skipped, head int8, devices 1, <bytes>`. Card test `muse_load_checkpoint_test` (label `gpu`; SKIP 77 without a checkpoint): every allocation's bytes = `muse_device_weight_bytes`, the read-backs of Review Focus 1's columns on the device equal the host repack, 0 unconsumed. **Commit** `loader: load_muse_glimmer - placement, the int8 head at load (spec 23b)`.

### Task 4: synthetic checkpoints and `check.py`

**Files:**
- Create: `tools/quantize/muse/make_synth.py`, `tools/quantize/muse/check.py`, `tools/quantize/muse/test_muse_quant.py`, `tests/loader/muse_synth_host_test.cc`
- Modify: `tools/quantize/README.md`, `tools/box_validate/muse_oracle.sh` (23a's; mode `synth-ckpt`), `tools/box_validate/data.sh` (`have oracle_muse_synth`)

**Interfaces:**
- Produces: `make_synth.py <out> --layers N --tokenizer <bf16-snapshot> [--seed S]`: a GPTQ-form checkpoint of the first N layers at real widths (`layer_types` / `layer_rope_theta` truncated, random bf16 weights with spec 20b's init scales, RTN-packed with 23a's packer, `quantization_config` copied from the real GPTQ config), real-shape `embed_tokens` / `lm_head` / `norm`, the original's `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (the ORIGINAL's), `generation_config.json`; `check.py <ckpt>`: names = `muse_ref.expected_names(text_config(ckpt), "gptq")`, every int4 tensor dequantises to the stored bf16 twin within RTN's bound, prints `ACCEPTED` or the first failure. Bytes (derived): 4 layers ~6.41 GB (embed + head 5.38 GB + 4 x 257 MB).

- [x] **Step 1: failing tests** `test_muse_quant.py` (a 1-layer checkpoint at a test-only reduced vocab of 1024 through `make()` arguments: names, `check.py` accepts it, a corrupted `qzeros` word is refused) and `muse_synth_host_test` (label `checkpoint`: the C++ host half over a `make_synth.py --layers 2` checkpoint - Review Focus 5).
- [x] **Step 2: implement, PASS** (Python tests in `agnes-ref-img` with the 5.19.0 site). Make the 2-layer real-width checkpoint on the Mac (~5.9 GB; time recorded) and run `muse_synth_host_test` by hand. **Commit** `quantize: Muse-Glimmer synthetic real-width checkpoints and check.py (spec 23b)`.
- [x] **Step 3: the box mode** `muse_oracle.sh <data> synth-ckpt` -> `$DATA/oracle-out-muse-synth/ckpt4` (`--layers 4`) + `check.py`; 23c's golden sets live beside it. **Commit** `box: muse_oracle.sh synth-ckpt (spec 23b)`.

### Task 5: the memory plan

**Files:**
- Create: `src/runtime/muse/muse_sizes.h`, `src/runtime/muse/muse_sizes.cc`, `src/runtime/muse/CMakeLists.txt`, `tests/runtime/muse_plan_test.cc`
- Modify: `src/runtime/CMakeLists.txt`

**Interfaces:**
- Consumes: `muse_layout.h`, `runtime::pp_balance` (`src/runtime/pipeline_plan.h`), the reserve rule of the other planners (1.5 GB), the card capacity read as the other planners read it.
- Produces:

```cpp
namespace runtime::muse {
struct MuseReserve { size_t drafter = 0; size_t prefill_scratch = 0; size_t decode_scratch = 0; };
struct MuseDevicePlan { size_t weights = 0, kv = 0, rings = 0, scratch = 0, reserve = 0, drafter = 0;
                        size_t total() const; };
struct MusePlan { model::MusePlacement placement; uint32_t max_len = 0;
                  std::vector<MuseDevicePlan> devices; bool fits = false; std::string why; };
MusePlan plan_muse(const model::MuseGlimmerDesc& d, uint32_t devices, uint32_t max_len /*0 = auto*/,
                   bool int8_head, const MuseReserve& r, const std::vector<size_t>& capacity);
void require_fits(const MusePlan& p);       // throws naming each device's bytes and capacity
}
```

`max_len` auto: the largest multiple of 256 up to `d.max_pos` (decision 6) that fits; the split for two devices by `pp_balance` over per-layer bytes (weights + ring or `max_len x 1024 B` per full layer).

- [x] **Step 1: failing test** `muse_plan_test`: one card, int8 head, no drafter -> `max_len` 131072, total printed (~19.3 GB + scratch placeholders); with a 1.358 GB drafter reservation still 131072 (Review Focus 4); bf16 head and a 5.112 GB bf16 drafter still 131072 (printed); two cards: the split by bytes and both totals printed, `--pp 2` with `split` given; `max_len` 262144 refused naming decision 6; a 16 GB card capacity -> the largest fitting `max_len` or a refusal naming the bytes; the plan's weights term = `muse_device_weight_bytes` per device.
- [x] **Step 2: implement, PASS.** The decode / prefill scratch terms are placeholders here (23c / 23d fill them; the test asserts they are added, not their values). **Commit** `runtime: the Muse-Glimmer memory plan, one card and two (spec 23b)`.

### Task 6: docs, the queue row

**Files:**
- Modify: spec 23 ("23b as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 43), `tools/box_validate/stages.sh` (row 43 block), `docs/13-loader.md` (a "Muse-Glimmer" paragraph: the GPTQ spelling accepted, the vision skip)

- [x] **Step 1:** "23b as built" with the measured Mac results and departures.
- [x] **Step 2: queue row 43. Stage list (sketch):**
  - `r43.k0` default cpu (after `g0.*`): MG0 - no kernel changed, the suites;
  - `r43.host` default cpu: `muse_glimmer_test`, `muse_layout_test`, `muse_repack_test`, `muse_plan_test`, `test_muse_quant.py`;
  - `r43.synth` optin cpu `oracle_image`: `muse_oracle.sh synth-ckpt` + `check.py` (~6.4 GB, then re-links `oracle-out*`);
  - `r43.synth_host` default cpu `oracle_muse_synth`: `muse_synth_host_test`;
  - `r43.load` default gpu `oracle_muse_synth`: `muse_load_checkpoint_test` on the synthetic (one card, then `--pp 2` split 2);
  - `r43.load_real` default gpu `muse_gptq`: the real checkpoint's load (18.96 GB read, 17.41 GB on the card), its line and time, 0 unconsumed.
- [x] **Step 3: commit** `docs: spec 23 (23b as built), box queue row 43 - the Muse-Glimmer loader`.

**Gate for the plan:** Mac - every host test green; the synthetic 2-layer checkpoint `ACCEPTED` and read by the C++ host half with 0 unconsumed; `tools/mac_check.sh --base main --quick` exit 0 (`kernel_cmdlines` +0 / -0 / ~0). Box - the loads on the synthetic and the real checkpoint with every part's bytes = the plan.

# Spec 21b - Qwen3.8-Flash-Next: the descriptor, the loader, the formats, synthetic checkpoints, the memory plan

**Status (2026-10-09): built on the Mac, branch `spec21b-qwen4exp-loader`; the card runs pending (queue row 31).**
Tasks 1-7 done and their Mac gates green: the four host tests, `test_qwen4exp_quant.py` 6 / 6, both synthetic
checkpoints `ACCEPTED`, Level Zero syntax, kernel_cmdlines +0 / -0 / ~0, `tools/mac_check.sh --base main --quick`
exit 0. As built (layouts, bytes, the planner's N, departures): spec 21 §13 "21b as built".
21q (our AutoRound run) is its own plan, `2026-10-09-spec21q-qwen4exp-autoround.md`, parallel to this one.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the model as data and its bytes on the card: `model::Qwen4ExpDesc` (the published model, held to config.json key by key), `loader::load_qwen4exp` reading both checkpoints the engine accepts - Intel's interim export (g128 sym routed experts through the exact g64 expansion, its bf16 dense layers as shipped) and ours (21q: AutoRound int4 g64 sym) - with per-expert int4 layout-1 blocks, the hyper-connection weights, the PLE projections, the I64 hash tensors checked against the formula, the int8 PLE table in host USM, placement over two cards and `--layers N`; the synthetic real-width checkpoints and the PLE int8 converter; the memory plan that sizes `N` per card and refuses the full model until spec 22.

**Architecture:** spec 21 §5, §6, §8 21b; K2's / Kolibri's pattern (spec 18 §5.1, spec 20 §11): a descriptor, a loader and (21c) an engine beside `qwen3_5`, K2 and Kolibri, their files untouched. `src/model/qwen4exp.{h,cc}` (library `b70_qwen4exp_model`), `src/loader/qwen4exp_layout.h` (header-only sizes: the one formula the planner and the loader share), `src/loader/qwen4exp_repack.{h,cc}` (the host half, no Level Zero), `src/loader/qwen4exp_ple.{h,cc}` (the hash, the table file, the host-USM ranges), `src/loader/qwen4exp_loader.{h,cc}` (library `b70_qwen4exp_loader`), `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (library `b70_qwen4exp_plan`, host only). Reused unchanged: `loader::resolve_snapshot`, `QuantConfig::parse`, `SafetensorsSet`, `assert_quant_invariants`, `check_quant_scan`, `LinearSrc::classify` (its g128 -> g64 scale expansion, `src/loader/quant.h:55-60`), `DeviceWeight`, `rtn_int4_g64` (`src/loader/rtn.h`), `quantise_row_int8` / `quantise_int8_tiled` (`src/loader/lm_head_int8.h`), `make_small_layout` (`src/loader/small_layout.h`), `common::repack_int4_layout1_cols`, `cols_interleave16`, `repack_bf16_tiled` (`src/common/repack.h`), `l0::Mem` with `l0::MemKind::Host` (`zeMemAllocHost`, `src/l0/memory.cc:28`), spec 16b's `runtime::pp_balance` (`src/runtime/pipeline_plan.h`).

**Tech Stack:** C++17, Level Zero (the loader's upload only), CMake/ctest; Python 3 + torch + safetensors in `agnes-ref-img` (with 21a's 5.19.0 site on `PYTHONPATH` where `qwen4exp_ref` is imported) for the synthetic checkpoints and the converter.

**Spec:** `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§1, §3, §4.3, §4.4, §5, §6, §8 21b, §10 decisions 6, 7, 9). Facts: 21a's `docs/probe-qwen4exp-<date>.md` (names, the PLE head / shard map, Intel's `quantization_config`, the per-layer bytes). Spec 22 §3 P0.5, §4 (what the host-USM ranges and the expert blocks must allow). Precedents: `src/model/kolibri1.{h,cc}`, `src/loader/kolibri1_{layout.h,repack.*,loader.*}`, `src/runtime/kolibri/kolibri_sizes.{h,cc}`, `tools/quantize/kolibri/{make_synth,check}.py`, `tests/model/kolibri1_test.cc`, `tests/loader/kolibri1_repack_test.cc`, `tests/runtime/kolibri_plan_test.cc`, plan 20c Tasks 1-3 and 5's planner half.

## Dependencies and branch points

- **Plan 21a merged** (Tasks 1-3 at least): `qwen4exp_ref.expected_names` (both forms), the facts sheet's names and its PLE map. Real-weight steps also need 21a's facts on Intel's exact tensor names; until the operator downloads Intel's checkpoint on the box, its gates SKIP 77.
- **Two checkpoint forms, read from the tensor names, all-or-nothing per group** (Kolibri's `kol_attn_form` rule): `Q4Form::Int4` (`.qweight` / `.scales` / `.qzeros`) or `Q4Form::Bf16` (`.weight`) separately for (a) the dense projections (GDN `in_proj_qkv` / `in_proj_z` / `out_proj`, QSA `q/k/v/o_proj`), (b) the shared experts, (c) the MTP head's routed experts. Routed experts are always int4 (g64, or g128 expanded exactly to g64 by `LinearSrc::classify`); a bf16 routed expert in the main model is refused by name (the bf16 original is the reference's input, never the engine's). Intel's export is `{dense Bf16, shared Bf16, mtp Bf16}`; ours is 21q's (decision 6 decides its MTP form).
- **The MTP head's bf16 experts** (Intel's: 5.03 GB) are quantised at load to int4 g64 by `loader::rtn_int4_g64` - spec 15e's precedent for Ornith's head (`src/loader/rtn.h:15-19`: the head only drafts; the verify decides every token, so it moves acceptance, never output). Recorded as the interim; decision 6 may replace it with 21q's quantised head.
- **Decision 7 (the PLE table):** this plan builds the proposal - a one-time int8 file beside the checkpoint (Task 1's converter), read by the loader - with the scale dtype a field of the file (`F32`, spec 9's rule, or `BF16`). Converting 102 GB at every load is not built; a checkpoint without the file is refused naming the converter's command.
- **Decision 9 (context against cache)** reaches this plan as a `max_len` argument only.
- **Spec 22:** nothing here waits for it, but two choices are made for it: every expert's gate‖up block and down block is one contiguous range at `e x block` (a host mirror copies or reads one range per expert), and the PLE table lives in **16 host-USM ranges, one per n-gram head** (head h's rows `[offset_h, offset_h + prime_h)` are contiguous in the hash's id space). One 51.8 GB allocation would exceed the largest pinned allocation ever measured (48 GiB, `docs/probe-prefix-cache-2026-09-27.md` §1); ~3.2 GB ranges are what spec 22's P0.5 calls "the PLE table's shards".

## Global Constraints

- Branch `spec21b-qwen4exp-loader` from main (after 21a); box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked into the box tree (`tools/box_validate/data.sh`).
- **F0, nothing moves:** no kernel touched (`tools/kernel_cmdlines` +0 / -0 / ~0); `loader::load`, `load_k2`, `load_kolibri1` and their files unchanged; every existing suite green.
- **One weight format** (the operator's rule): AutoRound W4A16 int4 g64 symmetric in `auto_round:auto_gptq` packing (`w = (q - 8) x scale`, `qzeros` 0x77777777, no `g_idx` or the identity), with the one exact-conversion exception - Intel's g128 sym experts expanded to g64 at load - and bf16 for everything AutoRound leaves bf16. No asymmetric path, no compressed-tensors path for this family, no third-party layout of any tensor (spec 21 §11: no GGUF-derived order; the GDN v-head map is `h // 3`).
- The descriptor is held to config.json key by key; every refusal names the key or the tensor.
- Every number in docs is measured, or marked derived / estimated / proposed. The byte figures below are **derived**; each test asserts its formula, not a typed constant from this plan.
- Mac checks: `tools/mac_check.sh --base main` (host tests, Level Zero syntax, kernel command lines unchanged). Python tests in `agnes-ref-img` with `--memory 28g`.
- No `rm -rf`. Signed commits on the branch; no merge, no push.

## Review Focus

1. **The g128 expansion is exact and per expert.** An Intel expert's g128 scales become g64 pairs (`2j`, `2j + 1` both take group j's) inside `LinearSrc::classify`; the dequantised words equal `(q - 8) x scale` of the g128 source bit for bit, expert by expert - test expert slices individually (plan 15c's Review Focus 2): a wrong stride shows as right experts chosen and wrong outputs.
2. **gate‖up is gate then up, interleaved in 16s on the device.** The original fuses `gate_up_proj [512, 1280, 2560]` with gate rows first (M:943 `.chunk(2)`); Intel and our exports ship per-expert `gate_proj` / `up_proj`. The device block is `cols_interleave16(gate, up)` (Kolibri's), so a column tile t of gate is followed by up's tile t. The test builds a checkpoint where gate and up differ by a sign and checks the block.
3. **The PLE ids never touch a float, and the table is never wrongly offset.** `q4_ple_multipliers` / `q4_ple_primes` recompute 21a's formula in C++ (`uint64_t`, splitmix64, the primality walk) and equal the checkpoint's I64 tensors; the converter's head h row r is the source's global row `offset_h + r`, across shard boundaries (shard 0..127 of 2,500,012 rows each do not align with heads - 21a's map). A test plants marker rows at every head's first and last row and at a shard boundary inside a head.
4. **The plan refuses what does not fit, by bytes, before any device work.** The full model (48 layers + MTP) is refused naming the bytes and spec 22; `--layers N` and the synthetic checkpoints are planned per card; two cards split by `runtime::pp_balance` over this family's per-layer bytes (16b's rule, not a second one). The host side too: the pinned PLE table against `MemAvailable` minus a 16 GiB margin (the measured pinned cap's rule), refused naming both numbers.
5. **Names both ways.** Every tensor of the checkpoint is read exactly once or listed as skipped by design (`--layers N`'s later layers; `model.visual.*`; Intel's `mtp.*` when MTP is off); an unknown or a missing tensor throws naming it.

---

### Task 1: the synthetic checkpoints and the PLE int8 converter (Python, no card)

**Files:**
- Create: `tools/quantize/qwen4exp/make_synth.py`, `tools/quantize/qwen4exp/ple_int8.py`, `tools/quantize/qwen4exp/check.py`, `tools/quantize/qwen4exp/test_qwen4exp_quant.py`
- Modify: `tools/quantize/README.md` (a "Qwen3.8-Flash-Next" section), `tools/box_validate/data.sh` (`have oracle_q4exp_synth`)
- Test: `test_qwen4exp_quant.py`

**Interfaces:**
- Consumes: `tools/oracle/qwen4exp_ref.py` (`text_config`, `expected_names`, `PleTable`), `tools/oracle/stream.py` (`dequant_t`), `tools/quantize/kolibri/make_synth.py`'s packer as the model to copy (`pack_rtn_g64`'s byte rule).
- Produces:
  - `make_synth.py <out> --layers N --form ours|intel --tokenizer <dir> [--ple-base 1000] [--mtp] [--seed 0]`: a real-width checkpoint - the original's `config.json` with `text_config.num_hidden_layers = N`, `layer_types[:N]` (N in 2..48; the default 4 holds three GDN layers, the PLE layer and one QSA layer), `ngram_vocab_size_base` = `--ple-base` (the reduced-prime table: the same kernel, small primes; real widths everywhere else), `quantization_config` as the form's export writes it; weights drawn as 21a's `qwen4exp_make_tiny.py` draws them at the REAL widths; `--form ours` packs routed and shared experts and the dense projections int4 g64 (`auto_round:auto_gptq`), `--form intel` packs routed experts int4 **g128** (F16 scales, Intel's per-expert names and `extra_config` / `ignore_layers`) and leaves dense, shared, MTP bf16; the PLE table as 128 bf16 shards like the original's (rows = the padded total for the reduced base) plus the three I64 tensors computed by the formula; `--mtp` adds `mtp.*`; one layer generated, packed and written at a time (peak RAM ~6 GB, ESTIMATED); `model.safetensors.index.json`; the tokenizer files copied from `--tokenizer`.
  - `ple_int8.py <snapshot> <out dir> [--scale f32|bf16]`: the decision-7 file. For each head h: `ple.h<h>.q` I8 `[prime_h][160]` (`q = clamp(rne(w / s), -127, 127)`, `s = max|w| / 127` in fp32 - spec 9's rule, `src/loader/lm_head_int8.h:4-8`; a zero row has `s = 0`) and `ple.h<h>.s` F32 or BF16 `[prime_h]`; the three I64 tensors copied; metadata `{source: <repo>@<revision>, rule: "row-int8-spec9", scale: "f32"|"bf16"}`; written as `<out dir>/ple_int8-0000N-of-0000M.safetensors` + `model.safetensors.index.json` (so `SafetensorsSet` reads it; it refuses index-less dirs, `src/loader/safetensors.cc:65-68`). Streams one source shard at a time (peak RAM ~1 GB, ESTIMATED); on the real table ~51.8 GB out (derived).
  - `check.py <ckpt> [--ple <dir>]`: names, dtypes, shapes, `qzeros`, scales' group (64, or 128 for Intel's experts), `quantization_config`, the I64 tensors against the formula, and the int8 file's rows against the source by sampled rows; prints `ACCEPTED`.

- [ ] **Step 1: the failing tests** in `test_qwen4exp_quant.py`: `test_synth_names` (`make_synth --layers 4 --form ours` and `intel` at a tiny `--ple-base`: the file's names equal `qwen4exp_ref.expected_names(tc, form)` both ways); `test_pack_g64_g128` (the packers against `dequant_t`'s rule at both group sizes; `qzeros` 0x77777777; nibble order); `test_gate_up_order` (Review Focus 2 at the source: gate rows first); `test_ple_int8_rows` (Review Focus 3: marker rows at each head's first / last row and a shard boundary inside a head land at head-local rows; scales by spec 9's rule; a zero row; `--scale bf16` rounds the scale once); `test_hash_tensors` (the I64 tensors equal the formula for the reduced base and for 20,000,000).
- [ ] **Step 2: run, expect FAIL** (the `make_synth` import fails):

```sh
docker run --rm --memory 28g --memory-swap 28g -e OMP_NUM_THREADS=3 -e PYTHONPATH=/ws/oracle-out-q4exp/site \
  -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/quantize/qwen4exp/test_qwen4exp_quant.py
```
- [ ] **Step 3: implement; PASS.** Then one synthetic per form and the check (Mac or box CPU; ~6 GB each at N = 4, derived: 4 x ~1.40 GB of layers + the 1.27 GB embedding + the 1.27 GB head):

```sh
docker run --rm --memory 28g -e OMP_NUM_THREADS=3 -e PYTHONPATH=/ws/oracle-out-q4exp/site -v "$PWD":/ws \
  -v "$HOME/b70-data":/data -w /ws agnes-ref-img:latest sh -c \
  'for f in ours intel; do python3 tools/quantize/qwen4exp/make_synth.py /data/q4exp-synth/$f --layers 4 --form $f --mtp \
     --tokenizer <the original snapshot small files> && python3 tools/quantize/qwen4exp/ple_int8.py /data/q4exp-synth/$f /data/q4exp-synth/$f-ple && \
     python3 tools/quantize/qwen4exp/check.py /data/q4exp-synth/$f --ple /data/q4exp-synth/$f-ple; done'
```
Expected: `ACCEPTED` twice.
- [ ] **Step 4: README, `have` line, commit** `git commit -S -m "quantize: Qwen3.8-Flash-Next synthetic real-width checkpoints (ours / Intel's form), the PLE int8 table converter, the export check (spec 21b)"`.

### Task 2: the model table

**Files:**
- Create: `src/model/qwen4exp.h`, `src/model/qwen4exp.cc`, `tests/model/qwen4exp_test.cc`, `tests/model/qwen4exp/config.json` (the original's `config.json`; if 21a's licence fact forbids redistributing the file, the test builds the same JSON in code instead and this file is not created), `tests/model/qwen4exp/synth4.json` (Task 1's synthetic config)
- Modify: `src/model/CMakeLists.txt` (library `b70_qwen4exp_model`), `tests/CMakeLists.txt` (a block `# ==== Spec 21b: Qwen3.8-Flash-Next descriptor and loader (label qwen4exp) ==== (begin)` at the end)
- Test: `qwen4exp_test`

**Interfaces:**
- Consumes: `model::GemvShape`, `model::WeightKind`, `model::Fuse` (`src/model/qwen35.h`), `common::json::Value`.
- Produces (used by every later task and by 21c-21e):

```cpp
namespace model {
enum class Q4Form { Int4, Bf16 };                     // a tensor group's form, from the names
struct Q4Forms { Q4Form dense = Q4Form::Int4, shared = Q4Form::Int4, mtp_experts = Q4Form::Int4;
                 uint32_t expert_group = 64; };       // 64 (ours) | 128 (Intel's, expanded at load)
enum class Q4LinearId { GdnQkvz, GdnAb, GdnOut, QsaQkvg, QsaIdx, QsaO, Router, PleKv, LmHead, kCount };
struct Q4Linear { Q4LinearId id; GemvShape shape; WeightKind kind; Fuse fuse; std::vector<std::string> parts; };
struct Qwen4ExpDesc;
struct Q4Placement {                                  // layers [0, split) on device 0, [split, layers) on 1
  uint32_t devices = 1, split = 0, layers = 0;
  uint32_t first(uint32_t dev) const; uint32_t end(uint32_t dev) const; uint32_t device_of(uint32_t l) const;
  uint32_t count(uint32_t dev) const;
  static Q4Placement one(const Qwen4ExpDesc& d); static Q4Placement two(const Qwen4ExpDesc& d, uint32_t split);
};
void validate(const Q4Placement& p, const Qwen4ExpDesc& d);    // throws std::invalid_argument
struct Qwen4ExpDesc {
  std::string name, architecture, model_type, prefix;   // "qwen3.8-flash-next", "Qwen4ExpForConditionalGeneration",
                                                        // "qwen4_exp", "model.language_model."
  uint32_t layers = 0, hidden = 0, hc = 0, hc_low = 0;  // 48, 2560, 4, 320
  uint32_t vocab = 0, vocab_used = 0, trained_max_len = 0;   // 248320, 21a's tokenizer count (248077 if = Qwen3.8's), 262144
  uint32_t gdn_k_heads = 0, gdn_v_heads = 0, gdn_head = 0, conv_taps = 0;   // 16, 48, 128, 4
  uint32_t q_heads = 0, kv_heads = 0, head_dim = 0, rope_dims = 0;          // 24, 2, 256, 64
  double rope_theta = 0;                                                    // 1e7
  uint32_t idx_heads = 0, idx_dim = 0, idx_budget = 0, idx_compress = 0;    // 4, 128, 2048, 4
  uint32_t experts = 0, top_k = 0, moe_inter = 0, shared_inter = 0;         // 512, 10, 640, 640
  uint32_t ple_layer = 0, ngram = 0, ple_heads = 0, ple_dim = 0, ple_conv_taps = 0;   // 1 (zero-indexed), 3, 16, 160, 4
  uint64_t ple_base = 0, ple_seed = 0;                  // 20,000,000 (a synthetic: config's value), 1234
  uint32_t ple_eos = 0, ple_pad = 0;                    // 248044, 128
  uint32_t mtp_layers = 0;                              // 1
  std::vector<uint32_t> eos;                            // {248046, 248044} (generation_config.json)
  float rms_eps = 0;
  Q4Forms forms;
  uint32_t gdn_out_s = 0, qkvg_s = 0, o_s = 0;   // int4 split-K, PROVISIONAL until 21c's sweep; qkv||z is S 1, fixed:
                                                 // gdn_step and prep_gated_head read S 1 (prep.cl:118-123)
  bool is_qsa(uint32_t l) const { return l % 4 == 3; }
  uint32_t qsa_before(uint32_t l) const; uint32_t gdn_before(uint32_t l) const;
  uint32_t hc_n() const { return hc * hidden; }                        // 10240
  uint32_t hc_down_n() const { return hc_low + hc; }                   // 324 (down's 320 + block_inject's 4)
  uint32_t conv_rows() const;                                          // 10240
  uint32_t qkvz_n() const;                                             // 16384: qkv 10240 | z 6144
  uint32_t ab_n() const { return 2 * gdn_v_heads; }                    // 96, padded to 128 on the device
  uint32_t q_n() const { return q_heads * head_dim; }                  // 6144
  uint32_t kv_n() const { return kv_heads * head_dim; }                // 512
  uint32_t qkvg_n() const { return 2 * q_n() + 2 * kv_n(); }           // 13312: q||gate per head | k | v
  uint32_t idx_n() const { return (idx_heads + 1) * idx_dim; }         // 640: 4 query heads + 1 raw key
  uint32_t gqa() const { return q_heads / kv_heads; }                  // 12
  uint32_t block_topk() const { return idx_budget / idx_compress; }    // 512
  uint32_t max_visible() const { return idx_budget + idx_compress - 1; }   // 2051
  uint32_t router_n() const;                                           // 528: 512 + the shared gate's row, to 16
  static constexpr uint32_t kRouteLanes = 256;
  uint32_t ple_e() const { return ple_heads * ple_dim; }               // 2560
  uint32_t ple_ring() const { return (ple_conv_taps - 1) * ngram; }    // 9
  static std::string layer_prefix(uint32_t l);                         // "model.language_model.layers.<l>."
  Q4Linear linear(Q4LinearId id) const;
};
const Qwen4ExpDesc& qwen4exp();                         // the published model, forms = ours
Qwen4ExpDesc qwen4exp_desc(const common::json::Value& config, const Q4Forms& forms);   // layers, ple_base read; the rest held
void check_qwen4exp_config(const Qwen4ExpDesc& d, const common::json::Value& config);  // throws naming the key
bool is_qwen4exp_model_type(const std::string& model_type);  // "qwen4_exp"
}
```

- [ ] **Step 1: the failing test** `tests/model/qwen4exp_test.cc` (argv: the real config, the synth one), `tests/check.h`'s `CHECK`: every derived width above; `is_qsa` true exactly at 3, 7, ..., 47; `qsa_before(48) == 12`; `check_qwen4exp_config` passes the real config and refuses, each naming its key: `num_experts: 256`, `num_experts_per_tok: 8`, `norm_topk_prob: false`, `output_gate_type: "silu"`, `hc_count: 2`, `indexer_budget: 1024`, `indexer_kv_heads: 2`, `ple_layer_ids: [3]`, `layer_types` with a GDN layer at 3, `partial_rotary_factor: 0.5`, `rope_theta: 1e6`, `tie_word_embeddings: true`, `head_dim: 128`, `num_hidden_layers: 49`; `qwen4exp_desc(synth4)` gives layers 4 and the synth's `ple_base`; placements as Kolibri's test checks them (`two(d, 24)`: `device_of(23) == 0`, `first(1) == 24`; a split of 0 or `layers` with two devices throws).
- [ ] **Step 2: register, run, expect FAIL** (link error): `ctest --preset mac-host -R '^qwen4exp_test$'`.
- [ ] **Step 3: implement** (`check_qwen4exp_config` in `check_kolibri1_config`'s style; the text config read from `text_config` or, for a text-only export, the top level). **PASS.**
- [ ] **Step 4: commit** `git commit -S -m "model: Qwen3.8-Flash-Next table - HC, QSA, PLE, 512-expert MoE, placement (spec 21b)"`.

### Task 3: the layout and the host repack

**Files:**
- Create: `src/loader/qwen4exp_layout.h`, `src/loader/qwen4exp_repack.h`, `src/loader/qwen4exp_repack.cc`, `tests/loader/qwen4exp_repack_test.cc`
- Modify: `src/loader/CMakeLists.txt` (library `b70_qwen4exp_loader`), `tests/CMakeLists.txt` (the 21b block)
- Test: `qwen4exp_repack_test` (host)

**Interfaces:**
- Consumes: Task 2; `SafetensorsSet`, `TensorInfo`, `LinearSrc::classify`, `QuantConfig::parse`, `rtn_int4_g64`, `make_small_layout`, `common::repack_*`, `cols_interleave16`.
- Produces:

```cpp
namespace loader {
// qwen4exp_layout.h - device-free sizes, header only (the planner's and the loader's one formula)
inline constexpr uint32_t kQ4TileU32 = 136;                 // layout 1: 128 nibble + 8 scale u32 per tile
inline constexpr size_t q4_layout1_bytes(uint32_t K, uint32_t N);   // N/16 x K/64 x 136 x 4
inline size_t q4_gate_up_block_bytes(const model::Qwen4ExpDesc& d); // K 2560, N 1280: 1,740,800 (derived)
inline size_t q4_down_block_bytes(const model::Qwen4ExpDesc& d);    // K 640, N 2560: 870,400
struct Q4LayerBytes {
  size_t hc_attn = 0, hc_mlp = 0;        // per HC: down||inject bf16 tiled {10240, 336} + up {320, 10240} + norm fp32 (1+w) [10240]
  size_t gdn_qkvz = 0, gdn_ab = 0, gdn_out = 0, gdn_small = 0;          // GDN layers
  size_t qsa_qkvg = 0, qsa_idx = 0, qsa_o = 0, qsa_small = 0;           // QSA layers: q/k norms, idx q/k norms, fp32 (1+w)
  size_t router = 0, gate_up = 0, down = 0, shared_gate_up = 0, shared_down = 0;
  size_t ple = 0;                        // the PLE layer: key||value bf16 {2560, 12800} + 3 norms + conv fp32 [10240][4]
  size_t total() const; size_t experts() const { return gate_up + down; }
};
Q4LayerBytes q4_layer_bytes(const model::Qwen4ExpDesc& d, uint32_t layer);
size_t q4_embed_bytes(const model::Qwen4ExpDesc& d);                 // bf16 [vocab][hidden]
size_t q4_lm_head_bytes(const model::Qwen4ExpDesc& d, bool int8);    // bf16 tiled, or int8 + fp32 row scales
size_t q4_final_mixer_bytes(const model::Qwen4ExpDesc& d);           // down {10240, 320} + up + norm
size_t q4_mtp_bytes(const model::Qwen4ExpDesc& d);                   // the head: fc x 2, pre-norms, one QSA layer + its MoE, 2 HCs, mixer
size_t q4_device_weight_bytes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev,
                              bool int8_head, bool mtp);
size_t q4_layer_read_per_token(const model::Qwen4ExpDesc& d, uint32_t layer);   // spec 21 §3's rows, derived
// qwen4exp_repack.h - host only
model::Q4Forms q4_forms(const SafetensorsSet& st, const model::Qwen4ExpDesc& d);   // all-or-nothing per group, throws naming
std::vector<std::string> q4_expected_names(const model::Qwen4ExpDesc& d, const model::Q4Forms& f, bool mtp);
std::vector<float> q4_rope_table(const model::Qwen4ExpDesc& d, uint32_t max_len);   // attn.cl's [max_len][2][32] form
struct Q4HostLayer { /* one vector per Q4LayerBytes field, in the device layout */ size_t src_int4_bytes = 0, src_bf16_bytes = 0; };
class Q4Checkpoint {
 public:
  Q4Checkpoint(const model::Qwen4ExpDesc& d, const SafetensorsSet& set);
  void check_names(bool mtp) const;
  void repack_layer(uint32_t layer, Q4HostLayer& out);
  void repack_mtp(Q4HostLayer& out);                 // Bf16 experts -> rtn_int4_g64 (spec 15e's rule)
  const uint16_t* embed(); const uint16_t* lm_head(); void final_mixer(Q4HostLayer& out);
  size_t skip_layers_from(uint32_t from); size_t skip_prefix(const std::string& p);   // --layers N; model.visual.
  size_t unconsumed(std::string* names = nullptr) const;
};
}
```

Device layouts (the kernels 21c writes read exactly these): routed experts as layout-1 blocks, block e at `e x block` in one per-layer allocation for gate‖up (`cols_interleave16(gate, up)`) and one for down; the shared expert int4 as one more block (ours) or `repack_bf16_tiled` (Intel's); dense int4 projections in GPTQ layout 0 at the descriptor's S (GDN `in_proj_qkv ‖ in_proj_z` fused as Qwen3.8's qkv‖z, `Fuse::Concat`; QSA `q ‖ k ‖ v` with q's per-head gate interleave kept, so `attn_prep`'s column map holds: k at 12288, v at 12800); bf16 projections `repack_bf16_tiled`; `in_proj_a ‖ in_proj_b` bf16 tiled to 128 columns; the GDN small block by `make_small_layout(2560, 10240, 48)` (conv taps, `-exp(A_log)`, `dt_bias`, the gated norm plain `w`); every `(1 + w)` norm baked to fp32 at load (Qwen's rule); the router bf16 tiled `{2560, 528}`: rows 0..511 the router, row 512 `shared_expert_gate`, 513..527 zero.

- [ ] **Step 1: the failing test** `qwen4exp_repack_test.cc`: a synthetic 4-layer checkpoint in memory (real widths, 8 experts: a test copy of the descriptor with `experts = 8`) in both forms through `SafetensorsSet` on a temp dir, checking: `q4_forms` per group (a mixed group throws naming the first odd tensor); every expert's block dequantises to `(q - 8) x scale` of its source, g64 and g128 (Review Focus 1); gate‖up interleave (Review Focus 2); the router's row 512 is the shared gate and 513..527 are zero; the GDN small block offsets are `make_small_layout`'s; `(1 + w)` baked bit for bit (`float(1) + float(w)`); the MTP head's bf16 experts equal `rtn_int4_g64` of the source; names both ways (a missing and an extra tensor each refused by name; `skip_layers_from(2)` consumes layers 2..3 and `skip_prefix("model.visual.")` the tower); `q4_layer_bytes` formulas: experts 1,336,934,400 B a layer, HC 26,951,680 B a layer, the final mixer 13,148,160 B, the int8 head 636,692,480 B, Intel's bf16 GDN layer ~1.49 GB (derived; the test asserts the formula and prints the values).
- [ ] **Step 2: register, run, expect FAIL; implement; PASS** (`ctest --preset mac-host -R '^qwen4exp_repack_test$'`).
- [ ] **Step 3: commit** `git commit -S -m "loader: Qwen3.8-Flash-Next host repack - per-expert layout-1 blocks (g128 expanded exactly), HC tiles, both forms, names both ways (spec 21b)"`.

### Task 4: the PLE host table

**Files:**
- Create: `src/loader/qwen4exp_ple.h`, `src/loader/qwen4exp_ple.cc`, `tests/loader/qwen4exp_ple_test.cc`, `tools/oracle/qwen4exp_ple_fixture.py` (writes `tests/loader/qwen4exp_ple_fixture.h`, committed)
- Modify: `src/loader/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `qwen4exp_ple_test` (host)

**Interfaces:**
- Consumes: Task 1's file format; `l0::Mem` (`MemKind::Host`) for the upload half only.
- Produces (21c's `q4_ple_gather` reads the table through `Q4PleTable::ptrs`; spec 22 reuses the allocation, pointer passing and checks):

```cpp
namespace loader {
std::array<uint64_t, 3> q4_ple_multipliers(uint32_t vocab, uint32_t ngram, uint32_t ple_index, uint64_t seed);  // M:1040-1049
std::vector<uint64_t> q4_ple_primes(uint64_t base, uint32_t heads, uint32_t ple_index);                         // M:1052-1105
// The hash of one position (host twin of the kernel; exact integers): history (t2, t1) and t0 with the
// EOS rule applied by the caller; returns the 16 global row ids.
std::array<uint64_t, 16> q4_ple_ids(uint32_t t0, uint32_t t1, uint32_t t2, const std::array<uint64_t, 3>& mult,
                                    const std::vector<uint64_t>& sizes, const std::vector<uint64_t>& offsets);
enum class Q4PleScale { F32, Bf16 };
struct Q4PleHost {                         // the int8 file, mmapped (SafetensorsSet over its dir)
  explicit Q4PleHost(const std::string& dir);
  Q4PleScale scale() const; uint64_t rows(uint32_t h) const;
  const int8_t* q(uint32_t h) const; const void* s(uint32_t h) const;
  std::array<uint64_t, 3> multipliers; std::vector<uint64_t> sizes, offsets;   // the file's I64 tensors
};
struct Q4PleTable {                        // 16 host-USM ranges (q) + 16 (scales); device-visible
  std::vector<std::unique_ptr<l0::Mem>> q, s;
  std::unique_ptr<l0::Mem> ptrs;           // device u64 [32]: q[0..15], s[0..15] - the kernel's table
  size_t bytes = 0; Q4PleScale scale = Q4PleScale::F32;
  uint64_t tag_checksum = 0;               // host-side: a tag per 2 MiB written then read back (21c adds the device read)
};
// Checks the file against the descriptor (heads, dims, the I64 tensors == the formula for d.ple_base),
// refuses when bytes > MemAvailable - 16 GiB (naming both), then copies each head into its own host range.
Q4PleTable load_q4_ple(l0::Context& ctx, const model::Qwen4ExpDesc& d, const std::string& ple_dir);
std::string q4_ple_dir(const std::string& snapshot_dir);   // "<snapshot>-ple-int8/" or $B70_Q4_PLE; refusal names ple_int8.py
}
```

- [ ] **Step 1: the failing test:** `qwen4exp_ple_fixture.py` (run once in `agnes-ref-img` with 21a's site; its output committed) writes the multipliers, primes, sizes and offsets for base 20,000,000 and for the synthetic base, and 64 histories (EOS at every slot, a missing predecessor, ids 0 and 248319) with their 16 ids each, from `qwen4exp_ref` / transformers; `qwen4exp_ple_test` asserts `q4_ple_multipliers`, `q4_ple_primes`, `q4_ple_ids` equal them exactly (Review Focus 3), and that `Q4PleHost` over a test-written file (Task 1's format, 2 heads of small primes) returns marker rows at head-local indices, both scale dtypes.
- [ ] **Step 2: register, run, expect FAIL; implement; PASS.**
- [ ] **Step 3: commit** `git commit -S -m "loader: the PLE table - the hash in exact integers, the int8 file, 16 host-USM ranges per head (spec 21b)"`.

### Task 5: the memory plan (weights and state; host only)

**Files:**
- Create: `src/runtime/qwen4exp/qwen4exp_sizes.h`, `src/runtime/qwen4exp/qwen4exp_sizes.cc`, `src/runtime/qwen4exp/CMakeLists.txt` (library `b70_qwen4exp_plan`, host only), `tests/runtime/qwen4exp_plan_test.cc`
- Modify: `src/runtime/CMakeLists.txt` (`add_subdirectory(qwen4exp)`), `tests/CMakeLists.txt`
- Test: `qwen4exp_plan_test` (host)

**Interfaces:**
- Consumes: Tasks 2-3; `runtime::pp_balance`, `PpBalance`, `PpChoice`, `kPpDevices` (`src/runtime/pipeline_plan.h`), `runtime::MemoryComponents`, `kMaxLenQuantum` (`src/runtime/memory_plan.h`).
- Produces (21c adds the decode scratch, the hand-off and the launch counts to this file; 21d the prefill scratch; 21e the MTP buffers):

```cpp
namespace runtime::qwen4exp {
// One device's persistent state at max_len (zeroed by reset):
//   kv_k / kv_v   bf16 [its QSA layers][max_len][2][256]            2048 B a position a layer (K + V)
//   idx_keys      bf16 [its QSA layers][max_len / 4][128]           the compressed indexer keys, 64 B a position a layer
//   idx_tail      bf16 [its QSA layers][4][128]                     the raw keys of the open block
//   gdn_state     fp32 [its GDN layers][48][128][128]               3,145,728 B a layer
//   conv_ring     bf16 [its GDN layers][16][10240]                  Qwen3.8's ring (gdn_step reads it)
//   ple_state     u32 [2] ids + bf16 [9][10240] conv rows           device 0 only (the PLE layer)
struct PersistentSizes { size_t control = 0, kv = 0, idx_keys = 0, idx_tail = 0, gdn_state = 0, conv_ring = 0, ple = 0;
                         size_t total() const; };
PersistentSizes persistent_sizes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, uint32_t max_len);
size_t kv_bytes_per_pos(const model::Qwen4ExpDesc& d);          // 12 x 2048 + 12 x 64 = 25,344 (derived)
size_t host_ple_bytes(const model::Qwen4ExpDesc& d, loader::Q4PleScale s);   // ~51.8 GB with bf16 scales (derived)
std::vector<size_t> pp_layer_bytes(const model::Qwen4ExpDesc& d, uint32_t max_len);   // weights + state, per layer
PpBalance pp_split(const model::Qwen4ExpDesc& d, uint32_t max_len, bool int8_head, bool mtp);   // 16b's pp_balance
struct DevicePlan : MemoryComponents { uint32_t device = 0; size_t weights = 0, rope = 0, kv = 0, state = 0; };
std::vector<DevicePlan> plan(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                             bool int8_head, bool mtp);
uint32_t layers_that_fit(const model::Qwen4ExpDesc& d, uint32_t devices, uint32_t max_len, bool int8_head,
                         bool mtp, size_t device_bytes, size_t reserve);   // the planner's N for --layers auto
// The largest multiple of kMaxLenQuantum, at most min(cap, 262144), whose plan + reserve fits every device (`--max-len auto`;
// 21c adds the decode scratch to plan(), 21d the prefill scratch behind a `prefill` flag - Kolibri's arrangement).
uint32_t max_len_that_fits(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, bool int8_head, bool mtp,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap = 0);
// Throws naming the bytes, both devices' capacity and spec 22 when the placement does not fit - the full
// model's refusal ("Qwen3.8-Flash-Next holds ~69 GB of weights at int4 g64 and two B70s ~62 GB: it runs
// whole only with spec 22's expert-offload tier; use --layers N").
void require_fits(const std::vector<DevicePlan>& p, const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);
std::string describe(const std::vector<DevicePlan>& p, const model::Q4Placement& pl, uint32_t max_len,
                     const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);
}
```

- [ ] **Step 1: the failing test** `qwen4exp_plan_test`: the persistent sizes' formulas (`kv_bytes_per_pos` 25,344; the GDN state 113.2 MB for 36 layers; a 262144-position plan ~6.6 GB of KV + keys, derived); `layers_that_fit` at max_len 32768 with the int8 head and Intel's forms on 32.53 GB cards with a 1.5 GB reserve - about 18 on one card and 38 on two (spec 21 §3, derived: the test prints the exact N and asserts it equals the formula's, not a typed number); `require_fits` throws for the full model on two cards naming spec 22 and for N = 38 on one card; passes N = 4 (the synthetic) on one card; `pp_split` is `runtime::pp_balance` over `pp_layer_bytes` (no second rule); `describe`'s text.
- [ ] **Step 2: register, run, expect FAIL; implement; PASS** (`ctest --preset mac-host -R '^qwen4exp_plan_test$'`).
- [ ] **Step 3: commit** `git commit -S -m "runtime: the Qwen3.8-Flash-Next memory plan - N per card, the split by bytes, the full model refused until spec 22 (spec 21b)"`.

### Task 6: the loader on the card

**Files:**
- Create: `src/loader/qwen4exp_loader.h`, `src/loader/qwen4exp_loader.cc`, `tests/loader/qwen4exp_load_checkpoint_test.cc`
- Modify: `src/loader/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `qwen4exp_load_checkpoint_test` (card, label `checkpoint;qwen4exp`, SKIP 77 without data)

**Interfaces:**
- Consumes: Tasks 2-5; `loader::LmHeadForm`, `quantise_int8_tiled`.
- Produces (21c's engine consumes it unchanged):

```cpp
namespace loader {
struct Q4Layer {                                   // one decoder layer on its device
  std::unique_ptr<l0::Mem> hc_attn, hc_mlp;        // per HC: down||inject tiles | up tiles | norm (offsets: q4_hc_offsets)
  std::unique_ptr<DeviceWeight> gdn_qkvz, gdn_ab, gdn_out;   // GDN layers
  std::unique_ptr<l0::Mem> gdn_small;
  std::unique_ptr<DeviceWeight> qsa_qkvg, qsa_idx, qsa_o;    // QSA layers
  std::unique_ptr<l0::Mem> qsa_small;
  std::unique_ptr<DeviceWeight> router;            // {2560, 528}
  std::unique_ptr<l0::Mem> gate_up, down;          // 512 layout-1 blocks each, block e at e x block (spec 22's ranges)
  std::unique_ptr<l0::Mem> shared;                 // int4 block pair or bf16 tiles (forms.shared)
  std::unique_ptr<l0::Mem> ple;                    // the PLE layer only: key||value tiles, norms, conv
};
struct Q4DevicePart { uint32_t device = 0, first = 0, end = 0; std::vector<Q4Layer> layers;
                      std::unique_ptr<l0::Mem> embed, rope;                     // device 0 (embed); every device with a QSA layer (rope)
                      std::unique_ptr<l0::Mem> final_mixer; std::unique_ptr<DeviceWeight> lm_head;   // last device
                      std::unique_ptr<Q4Layer> mtp; std::unique_ptr<l0::Mem> mtp_fc;            // last device, when loaded
                      size_t bytes = 0; const Q4Layer& layer(uint32_t l) const; };
struct Q4LoadedModel {
  model::Qwen4ExpDesc desc; model::Q4Placement placement; std::vector<Q4DevicePart> parts;
  Q4PleTable ple;                                  // host USM, device-visible from every device of the context
  uint32_t max_len = 0, checkpoint_layers = 0; bool int8_head = false, mtp = false;
  size_t unconsumed = 0, read_per_token = 0; double seconds = 0;
};
bool is_qwen4exp_checkpoint(const std::string& snapshot_dir);
model::Qwen4ExpDesc qwen4exp_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit = 0);
Q4LoadedModel load_qwen4exp(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                            uint32_t max_len, const model::Q4Placement& placement,
                            LmHeadForm lm_head = LmHeadForm::Int8, uint32_t layers_limit = 0, bool mtp = false,
                            const std::string& ple_dir = "");
}
```

`load_qwen4exp`: config held (`check_qwen4exp_config`), `max_len <= 262144`, quant invariants (`assert_quant_invariants` + `check_quant_scan`: g64 everywhere for ours, g128 for Intel's experts), forms from the names, names both ways, the plan (`runtime::qwen4exp::plan` + `require_fits`) before the first allocation, each layer repacked and uploaded straight onto its device (no load-then-place), the PLE table loaded once, the int8 head built at load (spec 9), zero unconsumed, each part's `bytes` equal to `q4_device_weight_bytes`; one report line per device plus one for the host table.

- [ ] **Step 1: the card test** `qwen4exp_load_checkpoint_test <snapshot> [layers] [mtp]`: loads, prints the report, asserts 0 unconsumed, `bytes` per part, the last expert's block and the router's row 512 read back equal to the host repack, the PLE ranges' tags read back on the host. Registered: `qwen4exp_load_synth_ours_test`, `qwen4exp_load_synth_intel_test` (Task 1's checkpoints, `${B70_Q4EXP_SYNTH_DIR}`), `qwen4exp_load_intel_layers_test` (Intel's checkpoint, `layers` 18, `${B70_Q4EXP_INTEL_SNAPSHOT}`), labels `checkpoint;qwen4exp`, `SKIP_RETURN_CODE 77`, `TIMEOUT 3600`.
- [ ] **Step 2: implement; Mac gate** `tools/mac_check.sh --base main` (host PASS incl. every test above, l0 PASS, cmdlines +0 / -0 / ~0).
- [ ] **Step 3: commit** `git commit -S -m "loader: load_qwen4exp - both checkpoint forms, two cards at load, --layers N, the PLE table pinned per head (spec 21b)"`.

### Task 7: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (section "21b as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 31), `tools/box_validate/stages.sh` (`row 31` block), `tools/box_validate/data.sh` (`have q4exp_intel_ple` for `<Intel snapshot>-ple-int8/`), `docs/13-loader.md` (a "Qwen3.8-Flash-Next" section: forms, the expansion, the PLE file)

- [ ] **Step 1:** "21b as built": the layouts and their derived bytes, the forms detection, the interim MTP RTN, the PLE file format (decision 7's proposal as built), departures from this plan.
- [ ] **Step 2: the queue row** (31): no kernel (G0's sha is the check); stages `r31.host` (`qwen4exp_test`, `qwen4exp_repack_test`, `qwen4exp_ple_test`, `qwen4exp_plan_test`), `r31.synth` (opt-in cpu: Task 1's two synthetics + their PLE files into `$DATA/oracle-out-q4exp-synth/`), `r31.ple_convert` (opt-in cpu: `ple_int8.py` on the original's 128 shards -> `<Intel snapshot>-ple-int8/`, ~51.8 GB written, `df -h` first), `r31.load` (`qwen4exp_load_synth_*`), `r31.load_real` (`qwen4exp_load_intel_layers_test` at 18 layers on one card, and the pinned table: the time to pin 51.8 GB, `MemAvailable` before / after - the number spec 22's P0.5 starts from), `r31.plan` (the planner's N printed for one and two cards at 32k / 128k), `r31.ple_kl` (opt-in cpu: 21q Task 3's int8-vs-bf16 PLE KL, decision 7's evidence); `rownote`s for the downloads and the converter's disk. `python3 tools/box_validate/test_box_validate.py` passes.
- [ ] **Step 3: commit** `git commit -S -m "docs: spec 21 (21b as built), box queue row 31 - the Qwen3.8-Flash-Next loader"`.

**Gate for the plan:** Mac - every host test green (repack word for word incl. g128 expansion per expert, refusals by name, the PLE hash exact, the plan at N = 4 / 18 / 38 and the full-model refusal), `kernel_cmdlines` +0 / -0 / ~0, the Python tests and both synthetic checkpoints `ACCEPTED`. Box - both synthetic loads and the 18-layer Intel load with 0 unconsumed and every part's bytes equal to the plan; the 51.8 GB table pinned with its time and `MemAvailable` recorded; the planner's N for one and two cards recorded.

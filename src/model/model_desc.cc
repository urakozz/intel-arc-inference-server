#include "model/model_desc.h"

#include <stdexcept>

#include "kernels/shape_suffix.h"

namespace model {
namespace {

static_assert(kLinearCount == 8, "LinearId grew: add the row below and re-check the shape table");

// a||b's device width: the source 2 x v-heads (96 on Qwen3.8, 64 on Ornith) is
// zero-padded to the bf16 GEMV's 128-column tile, which is also gdn_step's
// AB_STRIDE (b at column v-heads). One width for every model with <= 64 v-heads,
// and for both of a||b's forms (bf16, or a checkpoint's int4 - make_ab_rows).
constexpr uint32_t kAbPaddedN = 128;

// One GEMV row's tuned configuration: the split-K count S and the int4 layout.
struct RowTuning {
  uint32_t S, layout;
};
// The six int4 rows' tuning (the bf16 rows have none: a||b and lm_head are S1 L0).
struct TableTuning {
  RowTuning qkvz, out_proj, gate_up, down, qkv, o_proj;
};

// Qwen3.8's (layout, S): the decision block of docs/probes/probe-gemv-2026-08-24.md,
// measured on the box:
//
//   Task 4 production map: out/o_proj L0 S4, q||k||v L0 S2,
//   qkv||z L1 S1, gate||up L0 S8, down L0 S4.
constexpr TableTuning kQwen38Tuning = {{1, 1}, {4, 0}, {8, 0}, {4, 0}, {2, 0}, {4, 0}};

// The linear table, indexed by LinearId ordinal, from the descriptor's shapes
// (hidden, heads, intermediate) and a tuning. Qwen3.8's shapes are read from
// the checkpoint headers (docs/03-models.md); every K / N below is the
// descriptor's derived width, so Qwen3.8's rows are the numbers in the comments.
//
// The two bf16 rows (AB, LmHead) have no layout choice - one tiled layout
// (common::repack_bf16_tiled) - so they carry layout 0 as a filler.
// Fuse maps onto the loader's column maps: Concat -> common::cols_concat,
// Interleave16 -> common::cols_interleave16 (gate||up in 16-column blocks).
// Part names are layer-relative engine names; a consumer joins them with
// Qwen35::layer_prefix(). LmHead is a top-level tensor and appears in no
// layer's list; the capture binds it from this row directly.
//
// A FfnKind::Moe model's GateUp / Down rows are its SHARED expert (spec 15b):
// the same SwiGLU shape at `intermediate` = the shared expert's width. The
// routed experts are `ModelDesc::moe`, run by nothing before spec 15c.
std::array<FusedLinear, kLinearCount> make_table(const ModelDesc& d, const TableTuning& t) {
  const uint32_t H = d.hidden;
  const bool moe = d.ffn == FfnKind::Moe;
  const std::vector<std::string> gate_up =
      moe ? std::vector<std::string>{"mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"}
          : std::vector<std::string>{"mlp.gate_proj", "mlp.up_proj"};
  const std::vector<std::string> down = moe ? std::vector<std::string>{"mlp.shared_expert.down_proj"}
                                            : std::vector<std::string>{"mlp.down_proj"};
  if (d.gdn_ab_n() > kAbPaddedN)
    throw std::logic_error(d.name + ": a||b (" + std::to_string(d.gdn_ab_n()) +
                           " columns) exceeds the bf16 GEMV's " + std::to_string(kAbPaddedN));
  return {{
      // GDN: in_proj_qkv (conv dim = q k-heads x 128 | k k-heads x 128 | v v-heads x 128;
      // Qwen3.8 10240) || in_proj_z (v-heads x 128; 6144). Qwen3.8 5120 x 16384.
      {LinearId::QkvZ, {H, d.gdn_qkvz_n(), t.qkvz.S, t.qkvz.layout}, WeightKind::Int4,
       Fuse::Concat, {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z"}, 0},
      // GDN decay/beta, bf16 (kept out of the quantisation by the checkpoint's
      // dynamic rules). Source N is 2 x v-heads (Qwen3.8 48+48 = 96), zero-padded
      // to the kernel's 128.
      {LinearId::AB, {H, kAbPaddedN, 1, 0}, WeightKind::Bf16, Fuse::Concat,
       {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}, d.gdn_ab_n()},
      // K = v-heads x 128 (Qwen3.8 6144 = 48 x 128).
      {LinearId::OutProj, {d.gdn_value_dim(), H, t.out_proj.S, t.out_proj.layout},
       WeightKind::Int4, Fuse::Single, {"linear_attn.out_proj"}, 0},
      // gate || up interleaved so the lane holding gate[n] finds up[n] at +16.
      // Qwen3.8 5120 x 34816.
      {LinearId::GateUp, {H, 2 * d.intermediate, t.gate_up.S, t.gate_up.layout},
       WeightKind::Int4, Fuse::Interleave16, gate_up, 0},
      // Qwen3.8 17408 x 5120.
      {LinearId::Down, {d.intermediate, H, t.down.S, t.down.layout}, WeightKind::Int4,
       Fuse::Single, down, 0},
      // FA: q_proj is 2 x q-heads x 256 (q || gate per head; Qwen3.8 12288 = 24 x
      // 512), k/v kv-heads x 256 each (1024). Qwen3.8 5120 x 14336.
      {LinearId::Qkv, {H, d.fa_qkv_n(), t.qkv.S, t.qkv.layout}, WeightKind::Int4, Fuse::Concat,
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}, 0},
      // K = q-heads x 256 (Qwen3.8 6144 = 24 x 256).
      {LinearId::OProj, {d.fa_value_dim(), H, t.o_proj.S, t.o_proj.layout}, WeightKind::Int4,
       Fuse::Single, {"self_attn.o_proj"}, 0},
      // Top level. The bf16 row: 2.543 GB read in full every token on Qwen3.8.
      // This is the row `linear(LinearId::LmHead)` returns and the one the
      // shipped bf16 checkpoint needs; `lm_int4` below is its counterpart for a
      // checkpoint that packed this tensor.
      {LinearId::LmHead, {H, Qwen35::kVocab, 1, 0}, WeightKind::Bf16, Fuse::Single,
       {"lm_head"}, 0},
  }};
}

// The per-layer non-GEMV tensors (docs/03-models.md), with the shape, source
// dtype, device placement and bake of each - the loader walks exactly this and
// hardcodes nothing (fix I3). Offsets are the descriptor's small_layout(); the
// entries fill their block exactly and the loader throws if they do not.
//
// Ruling 2026-08-25: the RMSNorm family (input/post layernorms, q_norm,
// k_norm, and the top-level final norm) is stored fp32 `1 + w` - the HF
// reference multiplies in fp32, so a bf16 multiplier would add a rounding it
// never had. `linear_attn.norm` is RMSNormGated: plain `w`, and it stays bf16
// because the reference's own parameter dtype is bf16 and it multiplies in the
// bf16 domain.
void make_small_tables(ModelDesc& d) {
  const loader::SmallLayout l = d.small_layout();
  d.small_gdn = {
      {"input_layernorm.weight", d.hidden, "BF16", SmallBlock::Norms,
       uint32_t(l.norms_off_input), SmallBake::OnePlusWFp32},
      {"post_attention_layernorm.weight", d.hidden, "BF16", SmallBlock::Norms,
       uint32_t(l.norms_off_post), SmallBake::OnePlusWFp32},
      // bf16 [conv dim][1][4] in the checkpoint; the 4-tap depthwise state is
      // accumulated in fp32, so the taps are widened once at load.
      {"linear_attn.conv1d.weight", uint32_t(d.gdn_conv_dim() * loader::kConvTaps), "BF16",
       SmallBlock::Kind, uint32_t(l.gdn_off_conv), SmallBake::RawFp32Widen},
      // Only ever used as exp(g) with g = -exp(A_log)*softplus(...): hoisted.
      {"linear_attn.A_log", d.gdn_v_heads, "BF16", SmallBlock::Kind, uint32_t(l.gdn_off_nega),
       SmallBake::NegExpFp32},
      {"linear_attn.dt_bias", d.gdn_v_heads, "BF16", SmallBlock::Kind,
       uint32_t(l.gdn_off_dtbias), SmallBake::RawFp32Widen},
      {"linear_attn.norm.weight", Qwen35::kGdnHeadDim, "BF16", SmallBlock::Kind,
       uint32_t(l.gdn_off_gated_norm), SmallBake::PlainBf16},
  };
  d.small_fa = {
      {"input_layernorm.weight", d.hidden, "BF16", SmallBlock::Norms,
       uint32_t(l.norms_off_input), SmallBake::OnePlusWFp32},
      {"post_attention_layernorm.weight", d.hidden, "BF16", SmallBlock::Norms,
       uint32_t(l.norms_off_post), SmallBake::OnePlusWFp32},
      {"self_attn.q_norm.weight", Qwen35::kFaHeadDim, "BF16", SmallBlock::Kind,
       uint32_t(loader::kFaOffQNorm), SmallBake::OnePlusWFp32},
      {"self_attn.k_norm.weight", Qwen35::kFaHeadDim, "BF16", SmallBlock::Kind,
       uint32_t(loader::kFaOffKNorm), SmallBake::OnePlusWFp32},
  };
}

// **`lm_head` is the ONE linear whose kind is a property of the checkpoint,
// not of the model.** The shipped `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`
// leaves it bf16 (2.543 GB read per token); the self-quantised
// `qwen38-27b-w4g64-rtn` packs it int4 g64 sym (0.675 GB). The loader picks by
// which tensors the checkpoint ships (`loader::LinearSrc::classify`), never by
// a config label (docs/02, docs/13's "Classification" table).
//
// The int4 row is `{K hidden, N 248320, S 1, layout 1}`: layout 1 because it is
// the only int4 repack the loader implements and the measured winner at large
// N (docs/probes/probe-gemv-2026-08-24.md); S = 1 because N/64 = 3880 work-groups
// already saturate the device, and because S = 1 lets the capture bind the
// output straight at `logits` (gemv.cl writes `out[(s*M + m)*N + n]`).
// Deliberately NOT a ninth table entry: `LinearId::LmHead` must stay one ordinal.
//
// Spec 9: the int8 `lm_head` - quantised from the bf16 tensor at load, 1.272 GB
// on Qwen3.8. S = 1 for the same reason (gemv_i8w writes [M][N] straight at
// `logits`); `layout` is a filler, the int8 tiling is fixed (loader/lm_head_int8.h).
void make_lm_head_rows(ModelDesc& d) {
  d.lm_int4 = {LinearId::LmHead, {d.hidden, Qwen35::kVocab, 1, 1}, WeightKind::Int4,
               Fuse::Single,     {"lm_head"},                    0};
  d.lm_int8 = {LinearId::LmHead, {d.hidden, Qwen35::kVocab, 1, 0}, WeightKind::Int8,
               Fuse::Single,     {"lm_head"},                    0};
}

// Spec 15 §13: a||b's int4 row, for a checkpoint that quantised in_proj_a / in_proj_b
// (the published Ornith int4 export; Qwen3.8's keeps them bf16). The bf16 row's
// contract kept whole: K hidden, the 2 x v-heads real columns zero-padded to kAbPaddedN, so
// ab_out, gdn_step's AB_STRIDE and the prefill's pf_ab_proj (which reads a bf16 copy the
// loader dequantises, loader/ab.h) see what they always saw. gemv.cl at S = 1 - its output
// IS ab_out, [M][128] fp32, no partials for a fold to sum - and layout 1, qkv||z's (the
// other int4 GEMV of the GDN mixer). PROVISIONAL like every Ornith cell until P0: 128
// columns are 8 sub-groups at S 1 (gemv_bf16's L2 lesson, kernels.h), and an S > 1 form
// would need gdn_step to sum the slices.
void make_ab_rows(ModelDesc& d) {
  d.ab_int4 = {LinearId::AB, {d.hidden, kAbPaddedN, 1, 1}, WeightKind::Int4, Fuse::Concat,
               {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}, d.gdn_ab_n()};
}

void finish(ModelDesc& d, const TableTuning& t) {
  d.table = make_table(d, t);
  make_small_tables(d);
  make_lm_head_rows(d);
  make_ab_rows(d);
}

// Qwen3.8's widths.
void set_qwen38_shapes(ModelDesc& d) {
  d.hidden = 5120;
  d.fa_q_heads = 24;
  d.fa_kv_heads = 4;
  d.gdn_k_heads = 16;
  d.gdn_v_heads = 48;
  d.ffn = FfnKind::Dense;
  d.tied_embeddings = false;
  d.mtp_intermediate = 17408;
}

ModelDesc make_qwen38() {
  ModelDesc d;
  d.name = "qwen3.8";
  d.architecture = "Qwen3_5ForConditionalGeneration";
  d.layers = 64;
  d.fa_layers = 16;   // l % 4 == 3
  d.gdn_layers = 48;
  set_qwen38_shapes(d);
  d.intermediate = 17408;
  d.doc_w = 15.519e9;   // docs/03-models.md, measured 2026-08-24
  d.vocab_used = Qwen35::kVocabUsed;
  finish(d, kQwen38Tuning);
  return d;
}

// Spec 15: Ornith 1.5 35B-A3B (`ornith-ai/Ornith-1.5-35B-A3B`), every number from
// its config.json (`text_config`) and model.safetensors.index.json (spec 15 §1):
// hidden_size 2048, num_hidden_layers 40 (full_attention_interval 4), 16 q-heads /
// 2 kv-heads with attn_output_gate, linear_num_key_heads 16 / linear_num_value_heads
// 32, num_experts 256, num_experts_per_tok 8, moe_intermediate_size 512,
// shared_expert_intermediate_size 512 (+ `mlp.shared_expert_gate`), vocab 248320,
// tie_word_embeddings false, mtp_num_hidden_layers 1 (a MoE layer). Tensor names
// are Qwen3.5's (`linear_attn.` / `self_attn.`), so no name map.
//
// **The int4 checkpoint is `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`** (AutoRound
// 0.15.0, GPTQ packing, int4 g64 sym, desc_act false; spec 15 §13 has every fact read from
// its index and shard headers). What it settles here:
//   * vocab_used 248077 (Qwen35::kVocabUsed): its tokenizer.json defines ids 0..248076 -
//     the base checkpoint's file stopped its added tokens at </think> (248069), which the
//     descriptor masked from (248070) until this checkpoint was read; vocab, merges and
//     the committed golden prompts' ids are the same. The greedy argmax binds Qwen3.8's
//     argmax_stage1 binaries; the `_V248070` ones stay built, unbound.
//   * in_proj_a / in_proj_b ship int4 (its dynamic rules exclude only the routers): the
//     loader picks `ab_int4` by content (`ab()`, loader/ab.h).
//   * doc_w 2,344,862,976 B: the read-per-token bytes with a bf16 head, summed from the
//     shard headers - int4 qweight + f16 scales of the mixer linears (a||b's 2,088,960
//     included) and the shared experts 748,544,000, 8 routed experts per layer
//     534,773,760, the bf16 routers + shared gates 42,106,880, the bf16 small tensors
//     2,319,616, lm_head 1,017,118,720. The device adds the cross-check's itemised terms:
//     a||b's int4 padding, the fp32 widening and the router's 15 zero rows.
//
// **Loadable for decode from spec 15c, prefill from spec 15d** (the grouped experts on
// the L0 backends). The int4 rows' tuning is PROVISIONAL - copied from Qwen3.8's map so every
// S divides K/64 (2048 / 64 = 32, 4096 / 64 = 64, 512 / 64 = 8) and
// Capture::check_sizes's baked S pairings (qkv||z S1, gate||up S8, qkv S2) hold -
// until 15c's P0 measures them. The GateUp / Down rows (the shared expert) are not
// bound by the decode list: the loader puts the shared expert into the expert
// blocks' last slot (MoeDesc), where moe.cl runs it as the ninth slot.
ModelDesc make_ornith() {
  ModelDesc d;
  d.name = "ornith-1.5-35b-a3b";
  d.architecture = "Qwen3_5MoeForConditionalGeneration";
  d.layers = 40;
  d.fa_layers = 10;   // l % 4 == 3
  d.gdn_layers = 30;
  d.hidden = 2048;
  d.fa_q_heads = 16;
  d.fa_kv_heads = 2;
  d.gdn_k_heads = 16;
  d.gdn_v_heads = 32;
  d.ffn = FfnKind::Moe;
  d.moe = {256, 8, 512, 512, true};
  d.tied_embeddings = false;
  d.mtp_intermediate = 0;          // the MTP head is one MoE layer (spec 15e)
  d.intermediate = d.moe.shared_intermediate;
  d.doc_w = 2.344862976e9;            // the int4 checkpoint's headers (above), derived
  d.vocab_used = Qwen35::kVocabUsed;  // 248077: the int4 checkpoint's tokenizer.json
  d.provisional_tuning = true;
  finish(d, kQwen38Tuning);        // PROVISIONAL (copied), see above
  return d;
}

// Spec 17b: Qwen3.8's per-card descriptor under TP (model_desc.h, tp_card_desc). The column-parallel rows keep the
// one-card S and layout - each card's column is the one-card column, K for K - and the row-parallel rows halve K and,
// under Slices, S too, so a card's slices ARE one-card slices (K / S unchanged).
ModelDesc make_tp_card(const ModelDesc& one, TpCrossing c) {
  if (one.architecture != "Qwen3_5ForConditionalGeneration" || one.is_moe())
    throw std::runtime_error(one.name + ": tensor parallel is built for Qwen3.8 only (spec 17b)");
  ModelDesc d;
  d.name = one.name + "-tp2" + (c == TpCrossing::Folded ? "-folded" : "");
  d.architecture = one.architecture;
  d.layers = one.layers;
  d.fa_layers = one.fa_layers;
  d.gdn_layers = one.gdn_layers;
  set_qwen38_shapes(d);
  d.fa_q_heads /= kTpSize;
  d.fa_kv_heads /= kTpSize;
  d.gdn_k_heads /= kTpSize;
  d.gdn_v_heads /= kTpSize;
  d.mtp_intermediate = 0;   // no MTP head under TP before 17d
  d.intermediate = one.intermediate / kTpSize;
  d.doc_w = 0;
  d.vocab_used = one.vocab_used;   // the global count; each card's argmax masks its own half (runtime/tp_*)
  TableTuning t = kQwen38Tuning;
  if (c == TpCrossing::Slices) {
    t.out_proj.S /= kTpSize;
    t.o_proj.S /= kTpSize;
    t.down.S /= kTpSize;
  }
  finish(d, t);
  const uint32_t vocab_card = Qwen35::kVocab / kTpSize;   // 124160: whole 64-column tiles
  d.table[size_t(LinearId::LmHead)].shape.N = vocab_card;
  d.lm_int4.shape.N = vocab_card;
  d.lm_int8.shape.N = vocab_card;
  return d;
}

}  // namespace

const char* tp_crossing_name(TpCrossing c) { return c == TpCrossing::Slices ? "slices" : "folded"; }

const ModelDesc& tp_card_desc(const ModelDesc& one, TpCrossing c) {
  if (&one != &qwen38()) make_tp_card(one, c);   // throws, naming the model
  static const ModelDesc s = make_tp_card(qwen38(), TpCrossing::Slices);
  static const ModelDesc f = make_tp_card(qwen38(), TpCrossing::Folded);
  return c == TpCrossing::Slices ? s : f;
}

const FusedLinear& ModelDesc::linear(LinearId id) const {
  const size_t i = static_cast<size_t>(id);
  if (i >= kLinearCount)   // kCount, or an out-of-range cast from a dispatch
    throw std::out_of_range("ModelDesc::linear: LinearId ordinal " + std::to_string(i) +
                            " is out of range (table has " + std::to_string(kLinearCount) + ")");
  return table[i];
}

const FusedLinear& ModelDesc::lm_head(WeightKind kind) const {
  if (kind == WeightKind::Int8) return lm_int8;
  return kind == WeightKind::Int4 ? lm_int4 : linear(LinearId::LmHead);
}

const FusedLinear& ModelDesc::ab(WeightKind kind) const {
  if (kind == WeightKind::Int8)
    throw std::invalid_argument(name + ": a||b has no int8 form (bf16, or a checkpoint's int4)");
  return kind == WeightKind::Int4 ? ab_int4 : linear(LinearId::AB);
}

// Built per call rather than cached: layer_descs() returns by value, and the
// only per-layer state is index/kind.
std::vector<LayerDesc> ModelDesc::layer_descs() const {
  std::vector<LayerDesc> out;
  out.reserve(layers);
  for (uint32_t i = 0; i < layers; ++i) {
    const LayerKind kind = is_fa(i) ? LayerKind::FA : LayerKind::GDN;
    std::vector<FusedLinear> lin;
    for (LinearId id : Qwen35::linear_order(kind)) lin.push_back(linear(id));
    out.push_back({i, kind, std::move(lin), small_tensors(kind)});
  }
  return out;
}

size_t ModelDesc::mtp_checkpoint_bytes() const {
  if (mtp_intermediate == 0 && !mtp_head_moe()) return 0;
  const size_t H = hidden;
  // The FFN: the dense MLP, or (spec 15e) the router, the shared gate and experts + 1
  // SwiGLU experts (the shared one included) of expert_intermediate.
  const size_t ffn = mtp_head_moe()
                         ? size_t(moe.experts) * H + H +
                               size_t(moe.blocks()) * 3 * moe.expert_intermediate * H
                         : 3 * size_t(mtp_intermediate) * H;   // gate, up, down
  const size_t elems = H * 2 * H                                   // fc
                       + size_t(fa_qkv_n()) * H                    // q_proj, k_proj, v_proj
                       + H * fa_value_dim()                        // o_proj
                       + ffn
                       + 5 * H                                     // the five RMSNorms
                       + 2 * size_t(Qwen35::kFaHeadDim);           // q_norm, k_norm
  return elems * 2;                                                // all bf16
}

size_t ModelDesc::mtp_checkpoint_tensors() const {
  if (mtp_head_moe()) return 14 + 3 * size_t(moe.blocks());   // + router, shared gate
  return mtp_intermediate == 0 ? 0 : 15;
}

std::string ModelDesc::intermediate_suffix() const {
  return intermediate == 17408 ? std::string() : "_I" + std::to_string(intermediate);
}
std::string ModelDesc::hidden_suffix() const { return kernels::hidden_suffix(hidden); }
std::string ModelDesc::gdn_suffix() const { return kernels::gdn_suffix(gdn_k_heads, gdn_v_heads); }
std::string ModelDesc::fa_suffix() const { return kernels::fa_suffix(fa_q_heads, fa_kv_heads); }

const ModelDesc& qwen38() {
  static const ModelDesc d = make_qwen38();
  return d;
}

const ModelDesc& ornith() {
  static const ModelDesc d = make_ornith();
  return d;
}

const ModelDesc& desc_for_architecture(const std::string& architecture) {
  for (const ModelDesc* d : {&qwen38(), &ornith()})
    if (d->architecture == architecture) return *d;
  // Dropped models are refused by name (spec 14: Agnes 3.0 Flash, dropped 2026-11-07).
  if (architecture == "AgnesForConditionalGeneration")
    throw std::runtime_error("config.json architectures[0] is 'AgnesForConditionalGeneration': Agnes 3.0 Flash "
                             "was dropped (2026-11-07, spec 14) and is not supported; this engine runs "
                             "Qwen3_5ForConditionalGeneration (Qwen3.8) and Qwen3_5MoeForConditionalGeneration "
                             "(Ornith)");
  throw std::runtime_error("config.json architectures[0] is '" + architecture +
                           "'; this engine runs Qwen3_5ForConditionalGeneration (Qwen3.8) and "
                           "Qwen3_5MoeForConditionalGeneration (Ornith, from spec 15c)");
}

uint32_t ModelDesc::ffn_fold_s() const { return is_moe() ? 0u : shape(LinearId::Down).S; }

void require_loadable(const ModelDesc& d) {
  if (d.ffn == FfnKind::Moe) {
    // What src/kernels/moe.cl is written for (spec 15c): the shared expert is the
    // last slot of the same kernels, so it must exist, be gated and be one routed
    // expert's width; the route row holds at most 8 slots; the router kernel puts
    // one expert per lane of a work-group of `experts` lanes and sums the softmax by a
    // pairwise tree - moe.cl's `#error`: a POWER OF TWO in [16, 256], checked here too
    // (spec 18b) so another model fails at load naming its shape, not at capture naming
    // a binary nobody could build. (K2-Horizon's 100 experts run K2's own route kernel,
    // src/kernels/k2/k2_moe.cl, padded to 128 lanes; K2 is not a ModelDesc.)
    const MoeDesc& m = d.moe;
    const bool pow2 = m.experts != 0 && (m.experts & (m.experts - 1)) == 0;
    if (!m.has_shared_gate || m.shared_intermediate != m.expert_intermediate || m.top_k == 0 ||
        m.top_k > 8 || m.experts < m.top_k + 1 || !pow2 || m.experts < 16 || m.experts > 256 ||
        m.expert_intermediate % 64 != 0 || d.hidden % 64 != 0)
      throw std::runtime_error(d.name + " (" + d.architecture + "): this MoE shape (" +
                               std::to_string(m.experts) + " experts, top-" +
                               std::to_string(m.top_k) + ", expert " +
                               std::to_string(m.expert_intermediate) + ", shared " +
                               std::to_string(m.shared_intermediate) +
                               (m.has_shared_gate ? " gated" : " ungated") +
                               ") is not implemented: the decode kernels (spec 15c) take a gated "
                               "shared expert of the routed width, top-k <= 8, experts a "
                               "power of two in [16, 256] (moe.cl's route work-group)");
  }
  if (d.tied_embeddings)
    throw std::runtime_error(d.name + ": tied embeddings are not implemented");
}

void require_prefill(const ModelDesc& d) {
  if (!d.is_moe()) return;
  // Spec 15d: what the prefill MoE kernels are written for (src/kernels/prefill/pf_moe.cl,
  // pf_moe_gemm.cl) on top of require_loadable's shape: the hidden size whole 1024-k
  // rotation blocks (the h8 gate||up) and 256-column work-groups (down, the combine), a
  // block's gate||up whole 256-column work-groups, the expert width whole k-groups.
  const MoeDesc& m = d.moe;
  if (d.hidden % 1024 != 0 || (2 * m.expert_intermediate) % 256 != 0 ||
      m.expert_intermediate % 64 != 0)
    throw std::runtime_error(d.name + " (" + d.architecture + "): this MoE shape (hidden " +
                             std::to_string(d.hidden) + ", expert " +
                             std::to_string(m.expert_intermediate) +
                             ") is not implemented on the prefill path: spec 15d's grouped "
                             "GEMMs take a hidden size of whole 1024-k blocks and an expert "
                             "width of whole 128-column pairs; decode it instead (Engine::ingest)");
}

}  // namespace model

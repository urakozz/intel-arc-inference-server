#include "model/model_desc.h"

#include <stdexcept>

#include "kernels/shape_suffix.h"

namespace model {
namespace {

static_assert(kLinearCount == 8, "LinearId grew: add the row below and re-check the shape table");

// a||b's device width: the source 2 x v-heads (96 on Qwen3.8, 64 on Ornith) is
// zero-padded to the bf16 GEMV's 128-column tile, which is also gdn_step's
// AB_STRIDE (b at column v-heads). One width for every model with <= 64 v-heads.
constexpr uint32_t kAbPaddedN = 128;

// One GEMV row's tuned configuration: the split-K count S and the int4 layout.
struct RowTuning {
  uint32_t S, layout;
};
// The six int4 rows' tuning (the bf16 rows have none: a||b and lm_head are S1 L0).
struct TableTuning {
  RowTuning qkvz, out_proj, gate_up, down, qkv, o_proj;
};

// Qwen3.8's (layout, S): the decision block of docs/probe-gemv-2026-08-24.md,
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
       Fuse::Concat, {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z"}, 0, Fold::None, {}},
      // GDN decay/beta, bf16 (kept out of the quantisation by the checkpoint's
      // dynamic rules). Source N is 2 x v-heads (Qwen3.8 48+48 = 96), zero-padded
      // to the kernel's 128.
      {LinearId::AB, {H, kAbPaddedN, 1, 0}, WeightKind::Bf16, Fuse::Concat,
       {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}, d.gdn_ab_n(), Fold::None, {}},
      // K = v-heads x 128 (Qwen3.8 6144 = 48 x 128).
      {LinearId::OutProj, {d.gdn_value_dim(), H, t.out_proj.S, t.out_proj.layout},
       WeightKind::Int4, Fuse::Single, {"linear_attn.out_proj"}, 0, Fold::None, {}},
      // gate || up interleaved so the lane holding gate[n] finds up[n] at +16.
      // Qwen3.8 5120 x 34816.
      {LinearId::GateUp, {H, 2 * d.intermediate, t.gate_up.S, t.gate_up.layout},
       WeightKind::Int4, Fuse::Interleave16, gate_up, 0, Fold::None, {}},
      // Qwen3.8 17408 x 5120.
      {LinearId::Down, {d.intermediate, H, t.down.S, t.down.layout}, WeightKind::Int4,
       Fuse::Single, down, 0, Fold::None, {}},
      // FA: q_proj is 2 x q-heads x 256 (q || gate per head; Qwen3.8 12288 = 24 x
      // 512), k/v kv-heads x 256 each (1024). Qwen3.8 5120 x 14336.
      {LinearId::Qkv, {H, d.fa_qkv_n(), t.qkv.S, t.qkv.layout}, WeightKind::Int4, Fuse::Concat,
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}, 0, Fold::None, {}},
      // K = q-heads x 256 (Qwen3.8 6144 = 24 x 256).
      {LinearId::OProj, {d.fa_value_dim(), H, t.o_proj.S, t.o_proj.layout}, WeightKind::Int4,
       Fuse::Single, {"self_attn.o_proj"}, 0, Fold::None, {}},
      // Top level. The bf16 row: 2.543 GB read in full every token on Qwen3.8.
      // This is the row `linear(LinearId::LmHead)` returns and the one the
      // shipped bf16 checkpoint needs; `lm_int4` below is its counterpart for a
      // checkpoint that packed this tensor.
      {LinearId::LmHead, {H, Qwen35::kVocab, 1, 0}, WeightKind::Bf16, Fuse::Single,
       {"lm_head"}, 0, Fold::None, {}},
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
// N (docs/probe-gemv-2026-08-24.md); S = 1 because N/64 = 3880 work-groups
// already saturate the device, and because S = 1 lets the capture bind the
// output straight at `logits` (gemv.cl writes `out[(s*M + m)*N + n]`).
// Deliberately NOT a ninth table entry: `LinearId::LmHead` must stay one ordinal.
//
// Spec 9: the int8 `lm_head` - quantised from the bf16 tensor at load, 1.272 GB
// on Qwen3.8. S = 1 for the same reason (gemv_i8w writes [M][N] straight at
// `logits`); `layout` is a filler, the int8 tiling is fixed (loader/lm_head_int8.h).
void make_lm_head_rows(ModelDesc& d) {
  d.lm_int4 = {LinearId::LmHead, {d.hidden, Qwen35::kVocab, 1, 1}, WeightKind::Int4,
               Fuse::Single,     {"lm_head"},                    0, Fold::None, {}};
  d.lm_int8 = {LinearId::LmHead, {d.hidden, Qwen35::kVocab, 1, 0}, WeightKind::Int8,
               Fuse::Single,     {"lm_head"},                    0, Fold::None, {}};
}

void finish(ModelDesc& d, const TableTuning& t) {
  d.table = make_table(d, t);
  make_small_tables(d);
  make_lm_head_rows(d);
}

// Qwen3.8 and Agnes share every width but the layer counts and the MLP.
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

// Spec 14: Agnes 3.0 Flash. Qwen3.8's table with the two MLP rows widened by the
// parallel FFN fold (§2): gate||up N = 2 x (17408 + 2048) = 38912, down K = 19456.
//
// **PROVISIONAL tuning (spec 14 §6).** `{S, layout}` of both rows are COPIED from
// the nearest Qwen3.8 shapes - gate||up 5120x34816 L0 S8, down 17408x5120 L0 S4 -
// not measured; and the gemv binaries at the new shapes take the copied cells'
// extra defines (src/kernels/CMakeLists.txt). The box sweep
// (docs/probe-gemv-2026-08-24.md's method) replaces both rows by name:
// validation checklist "GEMV tuning". S must keep dividing K/64: 80 / 8 and 304 / 4.
// The S = 8 of gate||up is ALSO baked into prep.cl's SILU_S, which
// Capture::check_sizes asserts - a retune to another S needs a SILU_S variant.
ModelDesc make_agnes() {
  ModelDesc d;
  d.name = "agnes-3.0-flash";
  d.architecture = "AgnesForConditionalGeneration";
  d.layers = 72;
  d.fa_layers = 18;   // l % 4 == 3 (global_attention_interval 4)
  d.gdn_layers = 54;
  set_qwen38_shapes(d);   // the head's MLP stays 17408: no parallel FFN in it (spec 14 §1)
  d.parallel_ffn = 2048;
  d.intermediate = 17408 + 2048;   // 19456 = 19 x 1024 = 304 x 64
  d.name_map = {{"delta_attn.", "linear_attn."}, {"global_attn.", "self_attn."}};
  d.doc_w = 18.344234624e9;        // derived from the headers, see model_desc.h
  d.vocab_used = 248089;           // tokenizer.json: 248077 + 12 Agnes specials
  d.provisional_tuning = true;
  finish(d, kQwen38Tuning);        // gate||up and down: PROVISIONAL S8 L0 / S4 L0 (copied)
  FusedLinear& gu = d.table[static_cast<size_t>(LinearId::GateUp)];
  gu.fold = Fold::N;
  gu.fold_parts = {"mlp.parallel_ffn.gate_proj", "mlp.parallel_ffn.up_proj"};
  FusedLinear& dn = d.table[static_cast<size_t>(LinearId::Down)];
  dn.fold = Fold::K;
  dn.fold_parts = {"mlp.parallel_ffn.down_proj"};
  return d;
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
  d.parallel_ffn = 0;
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
// are Qwen3.5's (`linear_attn.` / `self_attn.`), so no name map. tokenizer_config's
// added tokens end at 248076: vocab_used 248077, as Qwen3.8's.
//
// **Not loadable before spec 15c** (`require_loadable` throws). The int4 rows'
// tuning is PROVISIONAL - copied from Qwen3.8's map so every S divides K/64 (2048
// / 64 = 32, 4096 / 64 = 64, 512 / 64 = 8) and Capture::check_sizes's baked S
// pairings (qkv||z S1, gate||up S8, qkv S2) hold - until 15c's P0 measures them.
// `doc_w` is 0: no int4 checkpoint exists yet (spec 15 decision 1, 15a).
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
  d.parallel_ffn = 0;
  d.doc_w = 0;
  d.vocab_used = 248077;
  d.provisional_tuning = true;
  finish(d, kQwen38Tuning);        // PROVISIONAL (copied), see above
  return d;
}

}  // namespace

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
  if (mtp_intermediate == 0) return 0;
  const size_t H = hidden;
  const size_t elems = H * 2 * H                                   // fc
                       + size_t(fa_qkv_n()) * H                    // q_proj, k_proj, v_proj
                       + H * fa_value_dim()                        // o_proj
                       + 3 * size_t(mtp_intermediate) * H          // gate, up, down
                       + 5 * H                                     // the five RMSNorms
                       + 2 * size_t(Qwen35::kFaHeadDim);           // q_norm, k_norm
  return elems * 2;                                                // all bf16
}

std::string ModelDesc::to_engine(const std::string& n) const {
  for (const auto& [ckpt, eng] : name_map) {
    const size_t at = n.find(ckpt);
    if (at != std::string::npos) return n.substr(0, at) + eng + n.substr(at + ckpt.size());
  }
  return n;
}

std::string ModelDesc::to_checkpoint(const std::string& n) const {
  for (const auto& [ckpt, eng] : name_map) {
    const size_t at = n.find(eng);
    if (at != std::string::npos) return n.substr(0, at) + ckpt + n.substr(at + eng.size());
  }
  return n;
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

const ModelDesc& agnes() {
  static const ModelDesc d = make_agnes();
  return d;
}

const ModelDesc& ornith() {
  static const ModelDesc d = make_ornith();
  return d;
}

const ModelDesc& desc_for_architecture(const std::string& architecture) {
  for (const ModelDesc* d : {&qwen38(), &agnes(), &ornith()})
    if (d->architecture == architecture) return *d;
  throw std::runtime_error("config.json architectures[0] is '" + architecture +
                           "'; this engine runs Qwen3_5ForConditionalGeneration (Qwen3.8), "
                           "AgnesForConditionalGeneration (Agnes 3.0 Flash) and "
                           "Qwen3_5MoeForConditionalGeneration (Ornith, from spec 15c)");
}

void require_loadable(const ModelDesc& d) {
  if (d.ffn == FfnKind::Moe)
    throw std::runtime_error(d.name + " (" + d.architecture + "): MoE not implemented (spec 15c)");
  if (d.tied_embeddings)
    throw std::runtime_error(d.name + ": tied embeddings are not implemented");
}

}  // namespace model

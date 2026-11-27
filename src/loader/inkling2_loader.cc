#include "loader/inkling2_loader.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <thread>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <stdexcept>

#include "common/bf16.h"
#include "loader/expert_fp.h"
#include "loader/lm_head_int8.h"
#include "model/dflash.h"

namespace loader {
namespace {

using D = model::Inkling2Desc;
using L = Inkling2WeightLayout;
constexpr const char* kHead = "model.llm.unembed.weight";
constexpr const char* kFoldWeight = "markov_head.hidden_out.weight";
constexpr const char* kFoldBias = "markov_head.hidden_out.bias";

size_t aligned(size_t bytes) { return (bytes + 63) & ~size_t(63); }

size_t elements(const std::vector<uint64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>{});
}

size_t source_bytes(const std::string& dtype) { return dtype == "F32" ? 4 : dtype == "BF16" ? 2 : 1; }

std::vector<uint16_t> bf16(const uint8_t* p, size_t n) {
  std::vector<uint16_t> out(n);
  std::memcpy(out.data(), p, n * 2);
  return out;
}

}  // namespace

size_t Inkling2WeightLayout::layer_bytes() const {
  return std::accumulate(bytes.begin() + kFirstLayer, bytes.begin() + kDraft, size_t(0));
}

size_t Inkling2WeightLayout::total(bool draft) const {
  return bytes[kModel] + layer_bytes() + (draft ? bytes[kDraft] : 0);
}

void Inkling2Checkpoint::add(std::string name, Source source, uint32_t padded_rows) {
  const bool is_draft = source.arena == L::kDraft;
  const auto& tensors = is_draft ? draft_tensors_ : text_.tensors();
  const auto it = tensors.find(name);
  if (it == tensors.end()) throw std::runtime_error("Missing Inkling tensor " + name);
  const auto& info = it->second;
  const size_t count = elements(source.shape);
  if (info.dtype != source.dtype || info.shape != source.shape || info.end - info.begin != count * source_bytes(source.dtype))
    throw std::runtime_error("Invalid Inkling tensor " + name + ". Expected " + source.dtype + " and the release shape.");
  const auto kind = source.encoding;
  auto& arena = layout_.bytes[source.arena];
  Inkling2Weight w{source.arena, arena};
  switch (kind) {
    case Encoding::Raw: w.bytes = count * source_bytes(source.dtype); break;
    case Encoding::Float: w.bytes = count * sizeof(float); break;
    case Encoding::Bf16Tile:
    case Encoding::Int8Tile:
      w.K = uint32_t(source.shape[1]);
      w.N = padded_rows ? padded_rows : uint32_t(source.shape[0]);
      w.bytes = size_t(w.K) * w.N * (kind == Encoding::Int8Tile ? 1 : 2);
      break;
    case Encoding::Expert: {
      const auto layer = source.arena - L::kFirstLayer;
      const auto format = desc_.expert_format[layer];
      const bool nv = format == common::ExpertFpFormat::Nvfp4;
      w.N = uint32_t(source.shape[1]);
      w.K = uint32_t(source.shape[2]) * (nv ? 2 : 1);
      w.bytes = D::experts * (size_t(w.N / 16) * (w.K / 64) * common::expert_fp_tile_words(format) + nv) * 4;
      break;
    }
    case Encoding::FoldSource: break;
  }
  if (!sources_.emplace(name, std::move(source)).second)
    throw std::logic_error("Duplicate Inkling tensor layout for " + name);
  if (w.bytes) {
    arena = aligned(arena + w.bytes);
    if (kind == Encoding::Int8Tile) {
      w.scales = arena;
      arena = aligned(arena + size_t(w.N) * sizeof(float));
    }
    layout_.weights.emplace(std::move(name), w);
  }
}

void Inkling2Checkpoint::open_draft(const std::string& file) {
  draft_ = std::make_unique<MappedFile>(file);
  auto entries = SafetensorsSet::parse_header(draft_->data(), draft_->size());
  uint64_t header = 0;
  std::memcpy(&header, draft_->data(), sizeof(header));
  draft_data_start_ = sizeof(header) + header;
  for (auto& [name, info] : entries) {
    if (info.begin > info.end || info.end > draft_->size() - draft_data_start_)
      throw std::runtime_error("Invalid drafter safetensors offsets for " + name);
    if (!draft_tensors_.emplace(name, std::move(info)).second)
      throw std::runtime_error("Duplicate drafter safetensors tensor " + name);
  }
}

// Plan 41k: the DFlash 2 drafter's tensors, its descriptor held to its kernels (model::check_inkling_dflash2).
void Inkling2Checkpoint::add_dflash(const std::string& dir) {
  const auto d = model::dflash_desc_from_config(read_json(dir + "config.json"));
  model::check_inkling_dflash2(d);
  dflash_ = std::make_unique<DFlashCheckpoint>(d, dir, /*require_packed_g64=*/true);
  if (dflash_->target_hidden() != D::hidden)
    throw std::runtime_error("Inkling DFlash 2 drafter fc.weight must take four 2304-channel target taps");
  desc_.draft = d;
  const DraftFormat format = dflash_->packed() ? DraftFormat::Int4G64 : DraftFormat::Int8;
  layout_.draft_format = format;
  if (dflash_->packed()) layout_.packed_draft_key = dflash_->content_key();
  constexpr uint32_t a = L::kDraft;
  const uint32_t H = d.hidden, G = d.kproj_n();
  auto& size = layout_.bytes[a];
  auto raw = [&](const std::string& name, size_t bytes) {
    layout_.weights.emplace(name, Inkling2Weight{a, size, bytes});
    size = aligned(size + bytes);
  };
  auto linear = [&](const std::string& name, uint32_t N, uint32_t K) {
    const bool int8 = format == DraftFormat::Int8;
    Inkling2Weight w{a, size, int8 ? size_t(N) * K : dflash_linear_bytes(K, N, format), 0, K, N};
    size = aligned(size + w.bytes);
    if (int8) {
      w.scales = size;
      size = aligned(size + size_t(N) * sizeof(float));
    }
    layout_.weights.emplace(name, w);
    return w;
  };
  linear("fc.weight", H, d.fc_k(H));
  raw("hidden_norm.weight", H * sizeof(float));
  raw("norm.weight", H * sizeof(float));
  for (uint32_t l = 0; l < d.layers; ++l) {
    const auto p = "layers." + std::to_string(l) + ".", s = p + "self_attn.";
    for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight"}) raw(p + n, H * sizeof(float));
    raw(s + "qk_norm", 2 * d.head_dim * sizeof(float));
    auto kv = linear(s + "qkv_proj.weight", d.qkv_n(), H);
    // The context projection reads the k || v tiles of q || k || v in place.
    const size_t skip = kv.scales ? size_t(d.q_n()) * H : dflash_linear_bytes(H, d.q_n(), format);
    kv.offset += skip;
    kv.bytes -= skip;
    if (kv.scales) kv.scales += d.q_n() * sizeof(float);
    kv.N = 2 * d.kv_n();
    layout_.weights.emplace(s + "kv_proj.weight", kv);
    linear(s + "o_proj.weight", H, d.q_n());
    linear(p + "mlp.gate_up_proj.weight", d.gate_up_n(), H);
    linear(p + "mlp.down_proj.weight", H, d.inter);
    for (const char* c : {"attention_conv.", "mlp_conv."}) {
      raw(p + c + "base_kernel", 2 * d.conv_taps * H * sizeof(float));
      linear(p + c + "kernel_projection.weight", G, H);
    }
  }
  linear("candidate_selector.hidden_projection.weight", d.sel_rank, H);
  for (const char* n : {"candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"})
    raw(n, size_t(d.vocab) * d.sel_rank * sizeof(uint16_t));
  raw("df2.mask_embed", H * sizeof(uint16_t));
}

Inkling2Checkpoint::Inkling2Checkpoint(const std::string& arg, const std::string& dflash)
    : dir_(resolve_snapshot(arg)),
      desc_(model::inkling2_desc_from_config(read_json(dir_ + "config.json"), read_json(dir_ + "hf_quant_config.json"))),
      text_(dir_) {
  using E = Encoding;
  const std::string root = model::kInkling2Prefix;
  auto raw = [&](std::string n, std::vector<uint64_t> shape, uint32_t arena, bool fp32 = false) {
    add(std::move(n), {E::Raw, std::move(shape), fp32 ? "F32" : "BF16", arena});
  };
  auto floats = [&](std::string n, std::vector<uint64_t> shape, uint32_t arena) {
    add(std::move(n), {E::Float, std::move(shape), "BF16", arena});
  };
  auto linear = [&](std::string n, uint32_t rows, uint32_t k, uint32_t arena, E encoding = E::Int8Tile) {
    const uint32_t width = encoding == E::Int8Tile ? (rows + 63) / 64 * 64 : rows;
    add(std::move(n), {encoding, {rows, k}, "BF16", arena}, width);
  };
  // The physical head has six zero padding rows after the live vocabulary.
  add(kHead, {E::Int8Tile, {D::vocab, D::hidden}, "BF16", L::kModel}, D::head_rows);
  raw(root + "embed.weight", {D::vocab, D::hidden}, L::kModel);
  for (const char* n : {"embed_norm.weight", "norm.weight"}) floats(root + n, {D::hidden}, L::kModel);
  raw(root + "attn_res.output_query", {D::hidden}, L::kModel, true);
  for (const char* n : {"middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"})
    raw(root + n, {2 * (D::core_end - D::core_start), D::hidden}, L::kModel, true);

  for (uint32_t l = 0; l < D::layers; ++l) {
    const auto p = model::inkling2_layer_prefix(l), a = p + "attn.";
    const uint32_t arena = l + L::kFirstLayer;
    const bool local = desc_.local[l];
    if (l) raw(root + "attn_res.attn_queries." + std::to_string(l - 1),
               local ? std::vector<uint64_t>{D::hidden} : std::vector<uint64_t>{3, D::hidden}, arena, true);
    raw(root + "attn_res.mlp_queries." + std::to_string(l), {D::hidden}, arena, true);
    for (const char* n : {"wq_du.weight", "gate_attn.weight"}) linear(a + n, D::q_n, D::hidden, arena);
    for (const char* n : {"wk_dv.weight", "wv_dv.weight"}) linear(a + n, D::kv_n, D::hidden, arena);
    linear(a + "wo_ud.weight", D::hidden, D::q_n, arena);
    linear(a + "wr_du.weight", D::q_heads * D::relative_dim, D::hidden, arena);
    linear(a + "attn_sink_proj.weight", D::q_heads, D::hidden, arena);
    floats(a + "q_l2_clip.scale", {D::q_heads, D::head_dim}, arena);
    floats(a + "rel_l2_clip.scale", {D::q_heads, D::relative_dim}, arena);
    floats(a + "rel_logits_proj.proj", {D::relative_dim, D::relative_extent}, arena);
    for (const char* n : {"k_sconv.", "v_sconv."}) {
      if (local) {
        floats(a + n + "weight", {D::kv_n, 1, D::conv_taps}, arena);
      } else {
        floats(a + n + "static_w", {D::conv_taps, D::kv_n}, arena);
        linear(a + n + "weight_proj", D::kv_heads * D::conv_taps, D::hidden, arena);
      }
    }
    if (local) {
      floats(p + "attn_norm.weight", {D::hidden}, arena);
    } else {
      for (const char* n : {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"})
        floats(p + n, {D::hidden}, arena);
    }
    floats(p + "mlp_norm.weight", {D::hidden}, arena);
    for (const char* n : {"attn_sconv.weight", "mlp_sconv.weight"})
      floats(p + n, {D::hidden, 1, D::conv_taps}, arena);
    linear(p + "mlp.gate.weight", D::experts, D::hidden, arena, E::Bf16Tile);
    raw(p + "mlp.gate.bias", {D::experts}, arena, true);
    if (l >= D::core_start && l < D::core_end)
      for (const char* stream : {"attn.", "k.", "v.", "mlp."}) {
        const auto c = p + "controllers." + stream;
        linear(c + "down_weight", D::controller_rank, 2 * D::hidden, arena, E::Bf16Tile);
        for (const char* n : {"gain_weight", "forget_weight"})
          linear(c + n, D::hidden, D::controller_rank, arena, E::Bf16Tile);
      }

    const auto f = desc_.expert_format[l];
    const bool nv = f == common::ExpertFpFormat::Nvfp4;
    for (const bool up : {true, false}) {
      const std::string n = p + (up ? "mlp.experts.w13_weight" : "mlp.experts.w2_weight");
      const uint32_t rows = up ? 2 * D::inter : D::hidden, k = up ? D::hidden : D::inter;
      add(n, {E::Expert, {D::experts, rows, k / (nv ? 2u : 1u)}, nv ? "U8" : "F8_E4M3", arena});
      add(n + "_scale", {E::FoldSource, {D::experts, rows, k / common::expert_fp_group(f)}, nv ? "F8_E4M3" : "U8", arena});
      if (nv) {
        add(n + "_amax", {E::FoldSource, {D::experts}, "F32", arena});
        add(n + "_outer_scale_denominator", {E::FoldSource, {1}, "F32", arena});
      }
    }
  }
  for (const auto& [name, info] : text_.tensors()) {
    if (name.starts_with("model.visual.")) ++vision_;
    else if (!sources_.contains(name)) throw std::runtime_error("Unexpected Inkling text tensor " + name);
  }

  if (!dflash.empty()) {
    add_dflash(resolve_snapshot(dflash));
    return;
  }
  const auto file = dir_ + "mtp.safetensors";
  if (!std::filesystem::exists(file)) return;
  open_draft(file);

  constexpr uint32_t a = L::kDraft;
  floats("fuse_weight", {D::draft_layers + 1, desc_.draft.taps.size()}, a);
  raw("mask_embed", {D::hidden}, a);
  floats("norm.weight", {D::hidden}, a);
  for (uint32_t l = 0; l < D::draft_layers; ++l) {
    const auto p = "layers." + std::to_string(l) + ".", s = p + "self_attn.";
    for (const char* n : {"dflare_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight"})
      floats(p + n, {D::hidden}, a);
    for (const char* n : {"q_norm.weight", "k_norm.weight", "k_norm_ctx.weight"}) floats(s + n, {D::head_dim}, a);
    linear(s + "q_proj.weight", D::q_n, D::hidden, a, E::Bf16Tile);
    for (const char* n : {"k_proj.weight", "v_proj.weight", "k_proj_ctx.weight", "v_proj_ctx.weight"})
      linear(s + n, D::kv_n, D::hidden, a, E::Bf16Tile);
    linear(s + "o_proj.weight", D::hidden, D::q_n, a, E::Bf16Tile);
    for (const char* n : {"mlp.gate_proj.weight", "mlp.up_proj.weight"})
      linear(p + n, D::draft_inter, D::hidden, a, E::Bf16Tile);
    linear(p + "mlp.down_proj.weight", D::hidden, D::draft_inter, a, E::Bf16Tile);
  }
  floats("markov_head.dflare_norm.weight", {D::hidden}, a);
  raw("markov_head.markov_w1.weight", {D::vocab, D::markov_rank}, a);
  linear("markov_head.rnn_proj.weight", 3 * D::markov_rank, 2 * D::markov_rank + 2 * D::hidden, a, E::Bf16Tile);
  floats("markov_head.rnn_proj.bias", {3 * D::markov_rank}, a);
  add(kFoldWeight, {E::FoldSource, {D::hidden, D::markov_rank}, "BF16", a});
  add(kFoldBias, {E::FoldSource, {D::hidden}, "BF16", a});
  for (const auto& [name, info] : draft_tensors_)
    if (!sources_.contains(name) || sources_.at(name).arena != a)
      throw std::runtime_error("Unexpected mtp.safetensors tensor " + name);

  auto& size = layout_.bytes[a];
  Inkling2Weight fold{a, size, size_t(D::head_rows) * D::markov_rank, 0, D::markov_rank, D::head_rows};
  fold.scales = aligned(fold.offset + fold.bytes);
  layout_.weights.emplace("markov_head.fold.weight", fold);
  size = aligned(fold.scales + size_t(D::head_rows) * sizeof(float));
  layout_.weights.emplace("markov_head.fold.bias", Inkling2Weight{a, size, size_t(D::head_rows) * sizeof(float)});
  size = aligned(size + size_t(D::head_rows) * sizeof(float));
}

const uint8_t* Inkling2Checkpoint::data(const std::string& name, const Source& source) const {
  return source.arena == L::kDraft ? draft_->data() + draft_data_start_ + draft_tensors_.at(name).begin
                                  : text_.data(text_.tensors().at(name));
}

void Inkling2Checkpoint::stream(const Sink& write, bool with_draft, unsigned threads) const {
  if (with_draft && !has_drafter())
    throw std::runtime_error("Inkling drafter weights are missing. Use --spec off or restore mtp.safetensors.");
  const auto emit = [&](uint32_t a, size_t off, const auto& values) {
    write(a, off, std::as_bytes(std::span(values)));
  };
  // The checkpoint read ahead of the repack, in file order: kReaders threads in large preads fill the page cache, so
  // the repack faults on cached pages instead of reading the mapping a page fault at a time (Flash-Next's loader's rule).
  constexpr unsigned kReaders = 4;
  std::vector<const TensorInfo*> order;
  for (const auto& [name, t] : text_.tensors()) order.push_back(&t);
  auto ahead = std::async(std::launch::async, [&] { return text_.read_ahead(text_.ranges(order), kReaders); });
  struct Join {
    std::future<size_t>& f;
    ~Join() { f.wait(); }
  } join{ahead};

  const auto int8 = [&](const Inkling2Weight& w, uint32_t row, uint32_t n, const uint16_t* source) {
    std::vector<int8_t> q(size_t(n) * w.K);
    std::vector<float> scales(n);
    quantise_int8_tiled(source, w.K, n, q.data(), scales.data(), threads);
    emit(w.arena, w.offset + size_t(row) * w.K, q);
    emit(w.arena, w.scales + size_t(row) * sizeof(float), scales);
  };

  for (const auto& [name, source] : sources_) {
    if ((!with_draft && source.arena == L::kDraft) || source.encoding == Encoding::FoldSource ||
        source.encoding == Encoding::Expert)
      continue;
    const auto& w = layout_.weights.at(name);
    const auto* src = data(name, source);
    const auto count = elements(source.shape);
    if (source.encoding == Encoding::Raw) {
      write(w.arena, w.offset, std::as_bytes(std::span(src, w.bytes)));
    } else if (source.encoding == Encoding::Float) {
      auto words = bf16(src, count);
      std::vector<float> out(count);
      std::transform(words.begin(), words.end(), out.begin(), common::bf16_to_f32);
      emit(w.arena, w.offset, out);
    } else if (source.encoding == Encoding::Bf16Tile) {
      auto words = bf16(src, count);
      std::vector<uint16_t> out(count);
      common::repack_bf16_tiled(words.data(), w.K, w.N, out.data());
      emit(w.arena, w.offset, out);
    } else {
      const uint32_t live = name == kHead ? D::vocab_used : uint32_t(source.shape[0]);
      // PERF-111: a head-width tensor's tiles are k-major across its whole N (loader::int8_k_major), so it is
      // quantised whole; the others in 1024-row slices (n-major: a slice's tiles are contiguous).
      const uint32_t kRows = int8_k_major(w.N) ? w.N : 1024;
      for (uint32_t row = 0; row < w.N; row += kRows) {
        const uint32_t n = std::min(kRows, w.N - row), copy = std::min(n, live - row);
        std::vector<uint16_t> input(size_t(n) * w.K);
        std::memcpy(input.data(), src + size_t(row) * w.K * 2, size_t(copy) * w.K * 2);
        int8(w, row, n, input.data());
      }
    }
  }

  ExpertFpCheckpoint experts(dir_);
  for (uint32_t l = 0; l < D::layers; ++l) {
    const auto p = model::inkling2_layer_prefix(l) + "mlp.experts.";
    const auto& up = layout_.weights.at(p + "w13_weight");
    const auto& dn = layout_.weights.at(p + "w2_weight");
    for (uint32_t e = 0; e < D::experts; ++e) {
      const auto pair = experts.load(l, e);
      emit(up.arena, up.offset + e * (up.bytes / D::experts), pair.gate_up.words);
      emit(dn.arena, dn.offset + e * (dn.bytes / D::experts), pair.down.words);
    }
  }
  if (!with_draft) return;
  if (dflash()) {
    const auto& d = desc_.draft;
    const auto format = layout_.draft_format;
    const auto tensor = [&](const std::string& name, const auto& values) {
      const auto& w = layout_.weights.at(name);
      emit(w.arena, w.offset, values);
    };
    const auto linear = [&](const std::string& name, const DFlashHostLinear& values) {
      const auto& w = layout_.weights.at(name);
      if (format == DraftFormat::Int4G64) {
        emit(w.arena, w.offset, values.int4);
      } else {
        emit(w.arena, w.offset, values.i8);
        emit(w.arena, w.scales, values.scales);
      }
    };
    linear("fc.weight", dflash_->fc(format, threads));
    tensor("hidden_norm.weight", dflash_->hidden_norm());
    tensor("norm.weight", dflash_->final_norm());
    const auto qk = dflash_->qk_norms(), bases = dflash_->conv_bases();
    const size_t base_size = size_t(2) * d.conv_taps * d.hidden;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const auto p = "layers." + std::to_string(l) + ".";
      const auto norms = dflash_->layer_norms(l);
      tensor(p + "input_layernorm.weight", std::span(norms).first(d.hidden));
      tensor(p + "post_attention_layernorm.weight", std::span(norms).subspan(d.hidden));
      tensor(p + "self_attn.qk_norm", std::span(qk).subspan(size_t(l) * 2 * d.head_dim, 2 * d.head_dim));
      for (const auto& [name, which] : {
               std::pair{"self_attn.qkv_proj.weight", DFlashLinear::Qkv}, {"self_attn.o_proj.weight", DFlashLinear::O},
               {"mlp.gate_up_proj.weight", DFlashLinear::GateUp}, {"mlp.down_proj.weight", DFlashLinear::Down},
               {"attention_conv.kernel_projection.weight", DFlashLinear::AttnKproj},
               {"mlp_conv.kernel_projection.weight", DFlashLinear::MlpKproj}})
        linear(p + name, dflash_->layer_linear(l, which, format, threads));
      tensor(p + "attention_conv.base_kernel", std::span(bases).subspan(2 * l * base_size, base_size));
      tensor(p + "mlp_conv.base_kernel", std::span(bases).subspan((2 * l + 1) * base_size, base_size));
    }
    linear("candidate_selector.hidden_projection.weight", dflash_->selector_projection(format, threads));
    const size_t codes = size_t(d.vocab) * d.sel_rank;
    tensor("candidate_selector.predecessor_codebook", std::span(dflash_->codebook(false), codes));
    tensor("candidate_selector.successor_codebook", std::span(dflash_->codebook(true), codes));
    // the block's mask rows: embed_norm(embed[mask]) as ink_draft_embed computes row 0, rounded to bf16
    const auto* row = text_.data(text_.tensors().at(model::kInkling2Prefix + std::string("embed.weight"))) +
                      size_t(desc_.draft.mask_id) * D::hidden * 2;
    const auto x = bf16(row, D::hidden);
    const auto w = bf16(text_.data(text_.tensors().at(model::kInkling2Prefix + std::string("embed_norm.weight"))),
                        D::hidden);
    float ss = 0.f;
    for (const auto v : x) ss += common::bf16_to_f32(v) * common::bf16_to_f32(v);
    const float inv = 1.f / std::sqrt(ss / D::hidden + desc_.eps);
    std::vector<uint16_t> mask(D::hidden);
    for (uint32_t i = 0; i < D::hidden; ++i)
      mask[i] = common::f32_to_bf16((common::bf16_to_f32(x[i]) * inv) * common::bf16_to_f32(w[i]));
    const auto& m = layout_.weights.at("df2.mask_embed");
    emit(m.arena, m.offset, mask);
    return;
  }

  // Fold the recurrent latent and output bias into the target head, from
  // checkpoint BF16 values with FP64 reduction, then use the int8 head layout.
  const auto projection = bf16(data(kFoldWeight, sources_.at(kFoldWeight)), size_t(D::hidden) * D::markov_rank);
  const auto bias = bf16(data(kFoldBias, sources_.at(kFoldBias)), D::hidden);
  std::vector<double> matrix(projection.size()), offset(bias.size());
  std::transform(projection.begin(), projection.end(), matrix.begin(), common::bf16_to_f32);
  std::transform(bias.begin(), bias.end(), offset.begin(), common::bf16_to_f32);
  const auto& fw = layout_.weights.at("markov_head.fold.weight");
  const auto& fb = layout_.weights.at("markov_head.fold.bias");
  const auto* head = data(kHead, sources_.at(kHead));
  // Every head row's fold is its own (the same sums in the same order whatever the threads); the int8 head layout is
  // written whole (PERF-111: k-major across its 200,064 rows).
  std::vector<uint16_t> folded(size_t(D::head_rows) * D::markov_rank);
  std::vector<float> folded_bias(D::head_rows);
  constexpr uint32_t kRows = 256;
  const unsigned workers = threads ? threads : std::max(1u, std::thread::hardware_concurrency());
  std::atomic<uint32_t> next_block{0};
  const auto fold_rows = [&] {
    for (uint32_t row; (row = next_block.fetch_add(kRows)) < D::head_rows;) {
      const uint32_t n = std::min(kRows, D::head_rows - row);
      const uint32_t live = row < D::vocab_used ? std::min(n, D::vocab_used - row) : 0;
      auto raw = bf16(head + size_t(row) * D::hidden * 2, size_t(live) * D::hidden);
      for (uint32_t i = 0; i < live; ++i) {
        std::array<double, D::markov_rank> latent{};
        double folded_offset = 0.;
        // Accumulate every output in checkpoint channel order. Independent
        // latents can share a vector instruction without changing that order.
        for (uint32_t k = 0; k < D::hidden; ++k) {
          const double w = common::bf16_to_f32(raw[size_t(i) * D::hidden + k]);
          const auto* rank_row = matrix.data() + size_t(k) * D::markov_rank;
          for (uint32_t r = 0; r < D::markov_rank; ++r) latent[r] += w * rank_row[r];
          folded_offset += w * offset[k];
        }
        for (uint32_t r = 0; r < D::markov_rank; ++r)
          folded[size_t(row + i) * D::markov_rank + r] = common::f32_to_bf16(float(latent[r]));
        folded_bias[row + i] = float(folded_offset);
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < workers; ++t) pool.emplace_back(fold_rows);
  for (auto& t : pool) t.join();
  int8(fw, 0, D::head_rows, folded.data());
  emit(fb.arena, fb.offset, folded_bias);
}

}  // namespace loader

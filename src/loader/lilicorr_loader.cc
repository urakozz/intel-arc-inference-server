#include "loader/lilicorr_loader.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "loader/lilicorr_repack.h"

// lilicorr_loader.cc - the upload half of the LiLiCorr drafter loader (lilicorr_loader.h). Each piece is built on the host
// by LiliCheckpoint (decoded from NVFP4, then put in its format) and uploaded at once on a synchronous immediate list, so
// the host holds one linear at a time beside the mmapped checkpoint.
namespace loader {
namespace {

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem m(ctx, l0::MemKind::Device, bytes);
  imm.copy(m.ptr(), src, bytes);
  return m;
}
std::unique_ptr<l0::Mem> upload_u(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  return std::make_unique<l0::Mem>(upload(ctx, imm, src, bytes));
}
l0::Mem upload_f32(l0::Context& ctx, l0::CmdList& imm, const std::vector<float>& v, size_t& bytes) {
  bytes += v.size() * 4;
  return upload(ctx, imm, v.data(), v.size() * 4);
}
LiliLinear upload_linear(l0::Context& ctx, l0::CmdList& imm, const DFlashHostLinear& h, size_t& bytes) {
  bytes += h.bytes();
  if (h.kind == model::WeightKind::Int4)   // layout-1 tiles, the scales inline
    return LiliLinear{DeviceWeight{upload(ctx, imm, h.int4.data(), h.int4.size() * 4), nullptr, h.shape(), h.kind}, nullptr};
  if (h.kind == model::WeightKind::Int8)
    return LiliLinear{DeviceWeight{upload(ctx, imm, h.i8.data(), h.i8.size()),
                                   upload_u(ctx, imm, h.scales.data(), h.scales.size() * 4), h.shape(), h.kind},
                      nullptr};
  return LiliLinear{DeviceWeight{upload(ctx, imm, h.bf16.data(), h.bf16.size() * 2), nullptr, h.shape(), h.kind},
                    h.gs.empty() ? nullptr : upload_u(ctx, imm, h.gs.data(), h.gs.size() * 4)};
}
std::unique_ptr<LiliHeadWeight> upload_head(l0::Context& ctx, l0::CmdList& imm, const LiliHostHead& h, size_t& bytes) {
  std::vector<float> gb(h.gs);
  gb.insert(gb.end(), h.bias.begin(), h.bias.end());
  bytes += h.bytes();
  auto o = std::make_unique<LiliHeadWeight>(
      LiliHeadWeight{upload(ctx, imm, h.wt.data(), h.wt.size() * 2), upload(ctx, imm, gb.data(), gb.size() * 4), h.K, h.N});
  return o;
}

}  // namespace

const LiliHeadWeight& LiliWeights::head_layer(uint32_t l, LiliHead w) const {
  const uint32_t i = uint32_t(w) - uint32_t(LiliHead::InProj);
  if (i >= kLiliHeadPerLayer) throw std::runtime_error("LiliWeights::head_layer: not a per-layer head linear");
  return *head_layers.at(l).at(i);
}

l0::Mem lilicorr_embed_replica(l0::Context& ctx, const uint16_t* rows, uint32_t vocab, uint32_t hidden) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  return upload(ctx, imm, rows, size_t(vocab) * hidden * 2);
}

LiliWeights load_lilicorr(l0::Context& ctx, const std::string& snapshot_or_repo, const model::DFlashTargetShape& target,
                          const l0::Mem& target_embed, DraftFormat format, unsigned threads) {
  const auto t0 = std::chrono::steady_clock::now();
  const LiliCheckpoint ck(snapshot_or_repo);
  model::check_lilicorr(ck.desc());
  ck.check_target(target);
  if (format == DraftFormat::Bf16)
    throw std::runtime_error("load_lilicorr: --draft-format bf16 is not a LiLiCorr format (exact | int8 | int4)");
  if (target_embed.size() < size_t(target.vocab_rows) * target.hidden * 2)
    throw std::runtime_error("load_lilicorr: the target embedding replica is smaller than [vocab rows][hidden] bf16");
  const model::DFlashDesc& d = ck.desc();
  LiliWeights w;
  w.desc = d;
  w.format = format;
  w.target_hidden = ck.target_hidden();
  w.nvfp4_rows = ck.backbone_nvfp4_rows();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  size_t bytes = 0;
  double quant = 0;
  auto timed = [&](auto&& build) {
    const auto q0 = std::chrono::steady_clock::now();
    auto h = build();
    quant += std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
    return h;
  };

  w.fc = std::make_unique<LiliLinear>(
      upload_linear(ctx, imm, timed([&] { return ck.fc(format, threads); }), bytes));
  w.hidden_norm = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.hidden_norm(), bytes));
  w.final_norm = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.final_norm(), bytes));
  for (uint32_t l = 0; l < d.layers; ++l) {
    using L = DFlashLinear;
    auto lin = [&](L which) {
      return upload_linear(ctx, imm, timed([&] { return ck.layer_linear(l, which, format, threads); }), bytes);
    };
    LiliLinear qkv = lin(L::Qkv), o = lin(L::O), gu = lin(L::GateUp), dn = lin(L::Down), ak = lin(L::AttnKproj),
               mk = lin(L::MlpKproj);
    w.layers.push_back(LiliLayer{std::move(qkv), std::move(o), std::move(gu), std::move(dn), std::move(ak), std::move(mk),
                                 upload_f32(ctx, imm, ck.layer_norms(l), bytes)});
  }
  w.qk_norm = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.qk_norms(), bytes));
  w.conv_base = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.conv_bases(), bytes));
  w.sinks = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.sinks(), bytes));
  {   // the mask row: the target's embedding row mask_id (bf16 -> fp32, exact)
    l0::Mem stage(ctx, l0::MemKind::Host, size_t(d.hidden) * 2);
    imm.copy(stage.ptr(), static_cast<const uint8_t*>(target_embed.ptr()) + size_t(d.mask_id) * d.hidden * 2,
             size_t(d.hidden) * 2);
    const uint16_t* row = stage.as<uint16_t>();
    std::vector<float> mask(d.hidden);
    for (uint32_t i = 0; i < d.hidden; ++i) mask[i] = common::bf16_to_f32(row[i]);
    w.mask_row = std::make_unique<l0::Mem>(upload_f32(ctx, imm, mask, bytes));
  }
  // the head (exact in every arm, Decision 3)
  w.head.resize(size_t(LiliHead::AnchorOut) + 1);
  for (LiliHead h : {LiliHead::TokenProj, LiliHead::PassHidden, LiliHead::FeatUp, LiliHead::FeatDown,
                     LiliHead::ContextProj, LiliHead::Factor, LiliHead::OutIn, LiliHead::AnchorOut}) {
    if (h == LiliHead::TokenProj && !lili_has_token_proj(d)) continue;
    w.head[size_t(h)] = upload_head(ctx, imm, timed([&] { return ck.head_linear(h); }), bytes);
  }
  for (uint32_t l = 0; l < d.lil_layers; ++l) {
    std::vector<std::unique_ptr<LiliHeadWeight>> v;
    for (LiliHead h : {LiliHead::InProj, LiliHead::OutProj, LiliHead::MlpUp, LiliHead::MlpDown})
      v.push_back(upload_head(ctx, imm, timed([&] { return ck.head_linear(h, l); }), bytes));
    w.head_layers.push_back(std::move(v));
  }
  w.head_small = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.head_small(), bytes));

  const size_t want = lilicorr_weight_bytes(d, w.target_hidden, format, w.nvfp4_rows).total();
  if (bytes != want)
    throw std::runtime_error("load_lilicorr: allocated " + std::to_string(bytes) + " B, the layout formula says " +
                             std::to_string(want) + " B (loader/lilicorr_layout.h)");
  w.bytes = bytes;
  w.quant_seconds = quant;
  w.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::string taps;
  for (uint32_t t : d.taps) taps += (taps.empty() ? "" : "/") + std::to_string(t);
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "LiLiCorr drafter: %u layers, taps %s, block %u, %s (decoded%s at load %.1f s), %zu NVFP4 linears, the "
                "head exact, %zu B, %.1f s",
                d.layers, taps.c_str(), d.block, draft_format_name(format),
                format == DraftFormat::Exact ? "" : " + quantised", quant, ck.nvfp4_linears(), bytes, w.seconds);
  w.line = buf;
  std::printf("%s\n", buf);
  return w;
}

}  // namespace loader

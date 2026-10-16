#include "loader/dflash_loader.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "loader/dflash_repack.h"
#include "model/qwen35.h"

// dflash_loader.cc - the upload half of the DFlash drafter loader (dflash_loader.h). Each piece is built on the host
// by DFlashCheckpoint and uploaded at once on a synchronous immediate list, so the host holds one linear at a time.
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
DeviceWeight upload_linear(l0::Context& ctx, l0::CmdList& imm, const DFlashHostLinear& h, size_t& bytes) {
  bytes += h.bytes();
  if (h.kind == model::WeightKind::Int4)   // spec 23f: layout-1 tiles, the scales inline (no separate allocation)
    return DeviceWeight{upload(ctx, imm, h.int4.data(), h.int4.size() * 4), nullptr, h.shape(), model::WeightKind::Int4};
  if (h.kind == model::WeightKind::Int8) {
    DeviceWeight w{upload(ctx, imm, h.i8.data(), h.i8.size()), upload_u(ctx, imm, h.scales.data(), h.scales.size() * 4),
                   h.shape(), model::WeightKind::Int8};
    return w;
  }
  return DeviceWeight{upload(ctx, imm, h.bf16.data(), h.bf16.size() * 2), nullptr, h.shape(), model::WeightKind::Bf16};
}
std::unique_ptr<DeviceWeight> upload_linear_u(l0::Context& ctx, l0::CmdList& imm, const DFlashHostLinear& h,
                                              size_t& bytes) {
  return std::make_unique<DeviceWeight>(upload_linear(ctx, imm, h, bytes));
}
l0::Mem upload_f32(l0::Context& ctx, l0::CmdList& imm, const std::vector<float>& v, size_t& bytes) {
  bytes += v.size() * 4;
  return upload(ctx, imm, v.data(), v.size() * 4);
}

}  // namespace

DFlashWeights load_drafter(l0::Context& ctx, const std::string& snapshot_or_repo, const LoadedModel& target,
                           DraftFormat format, unsigned threads) {
  if (!target.desc) throw std::runtime_error("load_drafter: the target model is not loaded");
  return load_drafter(ctx, snapshot_or_repo, *target.desc,
                      target.linears.at({kTopLevel, model::LinearId::LmHead}).shape.N, target.embed, format, threads);
}

DFlashWeights load_drafter(l0::Context& ctx, const std::string& snapshot_or_repo, const model::ModelDesc& target,
                           uint32_t vocab_rows, const l0::Mem& target_embed, DraftFormat format, unsigned threads) {
  return load_drafter(ctx, snapshot_or_repo, model::DFlashTargetShape{target.name, target.layers, target.hidden, vocab_rows},
                      target_embed, format, threads);
}

DFlashWeights load_drafter(l0::Context& ctx, const std::string& snapshot_or_repo, const model::DFlashTargetShape& target,
                           const l0::Mem& target_embed, DraftFormat format, unsigned threads) {
  const auto t0 = std::chrono::steady_clock::now();
  const uint32_t vocab_rows = target.vocab_rows;
  const DFlashCheckpoint ck(snapshot_or_repo);
  ck.check_target(target);
  if (target_embed.size() < size_t(vocab_rows) * target.hidden * 2)
    throw std::runtime_error("load_drafter: the target embedding is smaller than [vocab rows][hidden] bf16");
  model::DFlashDesc d = ck.desc();
  if (d.muse() && d.vocab == 0) d.vocab = vocab_rows;   // the assistant's config has none: the target's (spec 23f)

  DFlashWeights w;
  w.desc = d;
  w.format = format;
  w.target_hidden = ck.target_hidden();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  size_t bytes = 0;
  double quant = 0;
  auto timed = [&](auto&& build) {
    const auto q0 = std::chrono::steady_clock::now();
    auto h = build();
    quant += std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
    return h;
  };

  w.fc = upload_linear_u(ctx, imm, timed([&] { return ck.fc(format, threads); }), bytes);
  w.hidden_norm = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.hidden_norm(), bytes));
  w.final_norm = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.final_norm(), bytes));
  for (uint32_t l = 0; l < d.layers; ++l) {
    using L = DFlashLinear;
    auto lin = [&](L which) { return timed([&] { return ck.layer_linear(l, which, format, threads); }); };
    DeviceWeight qkv = upload_linear(ctx, imm, lin(L::Qkv), bytes);
    DeviceWeight o = upload_linear(ctx, imm, lin(L::O), bytes);
    DeviceWeight gu = upload_linear(ctx, imm, lin(L::GateUp), bytes);
    DeviceWeight dn = upload_linear(ctx, imm, lin(L::Down), bytes);
    l0::Mem norms = upload_f32(ctx, imm, ck.layer_norms(l), bytes);
    w.layers.push_back(DFlashLayerWeights{std::move(qkv), std::move(o), std::move(gu), std::move(dn), nullptr,
                                          nullptr, std::move(norms)});
    if (d.v2()) {
      w.layers.back().attn_kproj = upload_linear_u(ctx, imm, lin(L::AttnKproj), bytes);
      w.layers.back().mlp_kproj = upload_linear_u(ctx, imm, lin(L::MlpKproj), bytes);
    }
  }
  w.qk_norm = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.qk_norms(), bytes));
  if (d.v2()) {
    w.conv_base = std::make_unique<l0::Mem>(upload_f32(ctx, imm, ck.conv_bases(), bytes));
    w.sel_proj = upload_linear_u(ctx, imm, timed([&] { return ck.selector_projection(format, threads); }), bytes);
    const size_t cb = size_t(d.vocab) * d.sel_rank * 2;
    w.pred = upload_u(ctx, imm, ck.codebook(false), cb);
    w.succ = upload_u(ctx, imm, ck.codebook(true), cb);
    bytes += 2 * cb;
  }
  // The mask row: the checkpoint's, else the target's embedding row (bf16 -> fp32, exact).
  std::vector<float> mask = ck.mask_embedding();
  w.mask_from_checkpoint = !mask.empty();
  if (mask.empty()) {
    l0::Mem stage(ctx, l0::MemKind::Host, size_t(d.hidden) * 2);
    imm.copy(stage.ptr(), static_cast<const uint8_t*>(target_embed.ptr()) + size_t(d.mask_id) * d.hidden * 2,
             size_t(d.hidden) * 2);
    const uint16_t* row = stage.as<uint16_t>();
    mask.resize(d.hidden);
    for (uint32_t i = 0; i < d.hidden; ++i) mask[i] = common::bf16_to_f32(row[i]);
  }
  w.mask_row = std::make_unique<l0::Mem>(upload_f32(ctx, imm, mask, bytes));

  const size_t want = dflash_weight_bytes(d, w.target_hidden, format).total();
  if (bytes != want)
    throw std::runtime_error("load_drafter: allocated " + std::to_string(bytes) + " B, the layout formula says " +
                             std::to_string(want) + " B (loader/dflash_layout.h)");
  w.bytes = bytes;
  w.quant_seconds = quant;
  w.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::string taps;
  for (uint32_t t : d.taps) taps += (taps.empty() ? "" : "/") + std::to_string(t);
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "DFlash drafter: %s, %u layers, taps %s, block %u, %s (%s %.1f s), mask row from the %s, %zu B, %.1f s",
                d.architecture.c_str(), d.layers, taps.c_str(), d.block, draft_format_name(format),
                format == DraftFormat::Bf16 ? "tiled at load" : "quantised at load", quant,
                w.mask_from_checkpoint ? "checkpoint" : "target's embedding", bytes, w.seconds);
  w.line = buf;
  std::printf("%s\n", buf);
  return w;
}

}  // namespace loader

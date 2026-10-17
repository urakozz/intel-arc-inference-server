// Spec 24b Task 4: load_lfm2moe on the card (box only; label `checkpoint;lfm2moe`), one device or two.
//   lfm2moe_load_checkpoint_test <snapshot> [int8|bf16] [layers] [split]
// Loads the checkpoint (one of Task 1's synthetic ones, or the real int4 export truncated to its
// first `layers` - development mode); `split` > 0 (I4): two devices, layers [0, split) | [split, L)).
// Prints the report, and asserts: zero unconsumed tensors, each part's weight bytes equal to
// loader::lfm::device_weight_bytes (the planner's figure), the RoPE table's size on every device that
// holds an attention layer, and a sampled expert / in_proj block on EACH device read back equal to
// the host repack (muse_load_checkpoint_test.cc's pattern). Exit 77 (SKIP) when the snapshot is not
// on this machine, or `split` asks for two devices on a one-GPU host.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/lfm2moe_layout.h"
#include "loader/lfm2moe_loader.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

namespace {
std::vector<uint8_t> readback(l0::CmdList& imm, const l0::Mem& m, size_t off, size_t bytes) {
  std::vector<uint8_t> v(bytes);
  imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, bytes);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/LFM2.5-24B-A2B-W4A16-g64-AutoRound-GPTQ";
  const bool bf16_head = argc > 2 && std::string(argv[2]) == "bf16";
  const uint32_t layers = argc > 3 ? uint32_t(std::stoul(argv[3])) : 0;
  const uint32_t split = argc > 4 ? uint32_t(std::stoul(argv[4])) : 0;
  std::string snap;
  try {
    snap = loader::resolve_snapshot(arg);
    if (!loader::is_lfm2moe_checkpoint(snap)) throw std::runtime_error("not an lfm2_moe checkpoint");
  } catch (const std::exception& e) {
    std::printf("SKIP: no LFM2.5-24B-A2B checkpoint at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  if (split != 0 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: split %u asks for two devices; this host has %u\n", split, l0::Context::gpu_count());
    return 77;
  }
  const model::Lfm2MoeDesc pre = loader::lfm2moe_checkpoint_desc(snap, layers);
  l0::Context ctx0(0);
  std::unique_ptr<l0::Context> ctx1;
  std::vector<l0::Context*> devs = {&ctx0};
  if (split != 0) {
    ctx1 = std::make_unique<l0::Context>(ctx0, 1);
    devs.push_back(ctx1.get());
  }
  const model::LfmPlacement pl = split ? model::LfmPlacement::two(pre, split) : model::LfmPlacement::one(pre);
  loader::LfmLoadOptions o;
  o.int8_head = !bf16_head;
  o.max_len = 8192;
  o.layers = layers;
  loader::LfmLoaded m = loader::load_lfm2moe(devs, snap, o, pl);
  const model::Lfm2MoeDesc& d = m.desc;
  CHECK_EQ(d.layers, pre.layers);
  CHECK_EQ(m.unconsumed, size_t(0));
  CHECK_EQ(m.parts.size(), size_t(pl.devices));
  for (uint32_t dev = 0; dev < pl.devices; ++dev) {
    const loader::LfmDevicePart& P = m.parts[dev];
    CHECK_EQ(P.layers.size(), size_t(pl.count(dev)));
    CHECK_EQ(P.bytes, loader::lfm::device_weight_bytes(d, pl, dev, o.int8_head));
    const bool has_attn = loader::lfm::device_has_attention(d, pl, dev);
    CHECK_EQ(bool(P.rope), has_attn);
    if (has_attn) CHECK_EQ(P.rope_bytes, loader::lfm::rope_table_bytes(d, o.max_len));
    if (dev == 0) CHECK(P.embed_tiles);
    if (dev + 1 == pl.devices) {
      CHECK(P.final_norm);
      CHECK((o.int8_head && P.head_int8) || (!o.int8_head && !P.head_int8));
      // the bf16 head's own replica of the embedding tiles on a last device that is not device 0.
      if (!o.int8_head && dev != 0) CHECK(P.embed_tiles);
    }
  }

  loader::SafetensorsSet set(snap);
  // A sampled block on EACH device read back equal to the host repack, for both spellings' names.
  for (uint32_t l : {0u, d.layers - 1}) {
    const uint32_t dev = pl.device_of(l);
    const loader::LfmDevicePart& P = m.parts[dev];
    l0::CmdList imm = l0::CmdList::immediate(*devs[dev]);
    const std::string lp = d.layer_prefix(l);
    const loader::LfmLayer& L = P.layer(l);
    if (!d.is_dense(l)) {
      const model::LfmExpertNames names = loader::detect_expert_names(set);
      const char* g = names == model::LfmExpertNames::Proj ? "gate_proj" : "w1";
      const char* u = names == model::LfmExpertNames::Proj ? "up_proj" : "w3";
      const char* dn = names == model::LfmExpertNames::Proj ? "down_proj" : "w2";
      const std::string ep = lp + "feed_forward.experts." + std::to_string(d.experts - 1) + ".";
      const loader::LinearSrc wg = loader::LinearSrc::classify(set, ep + g);
      const loader::LinearSrc wu = loader::LinearSrc::classify(set, ep + u);
      const loader::LinearSrc wd = loader::LinearSrc::classify(set, ep + dn);
      const loader::ExpertRepacked want_e = loader::repack_expert(wg, wu, wd);
      const size_t off = loader::lfm::expert_gate_up_offset(d, d.experts - 1);
      const std::vector<uint8_t> got = readback(imm, *L.experts_gate_up, off, want_e.gate_up.size() * 4);
      CHECK(std::memcmp(got.data(), want_e.gate_up.data(), got.size()) == 0);
    }
    if (d.is_attention(l)) {
      const loader::LinearSrc o = loader::LinearSrc::classify(set, lp + "self_attn.out_proj");
      const model::LfmLinear oproj_ln = *std::find_if(d.linears(l).begin(), d.linears(l).end(),
                                                       [](const model::LfmLinear& x) { return x.id == model::LfmLinearId::OProj; });
      const loader::HostLinear want_o = loader::repack_fused_linear(oproj_ln, {o});
      const std::vector<uint8_t> got_o = readback(imm, L.o_proj->mem, 0, want_o.words.size() * 4);
      CHECK(std::memcmp(got_o.data(), want_o.words.data(), got_o.size()) == 0);
    } else {
      const loader::LinearSrc in = loader::LinearSrc::classify(set, lp + "conv.in_proj");
      const model::LfmLinear in_ln = *std::find_if(d.linears(l).begin(), d.linears(l).end(),
                                                    [](const model::LfmLinear& x) { return x.id == model::LfmLinearId::InProj; });
      const loader::HostLinear want_in =
          loader::repack_in_proj(in.qweight, in.scales, in.K, in.N, o.conv_tiles, in_ln.shape.layout);
      const std::vector<uint8_t> got_in = readback(imm, L.in_proj->mem, 0, want_in.words.size() * 4);
      CHECK(std::memcmp(got_in.data(), want_in.words.data(), got_in.size()) == 0);
    }
  }
  std::printf("lfm2moe_load_checkpoint_test OK: %s, %u layers, split %u, %zu total B, %.1f s\n", snap.c_str(),
             d.layers, split, m.total_bytes(), m.seconds);
  return 0;
}

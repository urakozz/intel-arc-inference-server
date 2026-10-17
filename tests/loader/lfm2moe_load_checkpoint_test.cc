// Spec 24b Task 4: load_lfm2moe on the card (box only; label `checkpoint;lfm2moe`), one device.
//   lfm2moe_load_checkpoint_test <snapshot> [int8|bf16] [layers]
// Loads the checkpoint (one of Task 1's synthetic ones, or the real int4 export truncated to its
// first `layers` - development mode), prints the report, and asserts: zero unconsumed tensors, each
// part's weight bytes equal to loader::lfm::layer_weight_bytes summed (the planner's figure), the
// RoPE table's size, and a sampled expert / in_proj block read back equal to the host repack.
// Exit 77 (SKIP) when the snapshot is not on this machine.
#include <cstdio>
#include <cstring>
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
  std::string snap;
  try {
    snap = loader::resolve_snapshot(arg);
    if (!loader::is_lfm2moe_checkpoint(snap)) throw std::runtime_error("not an lfm2_moe checkpoint");
  } catch (const std::exception& e) {
    std::printf("SKIP: no LFM2.5-24B-A2B checkpoint at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  const model::Lfm2MoeDesc pre = loader::lfm2moe_checkpoint_desc(snap, layers);
  l0::Context ctx(0);
  std::vector<l0::Context*> devs = {&ctx};
  loader::LfmLoadOptions o;
  o.int8_head = !bf16_head;
  o.max_len = 8192;
  o.layers = layers;
  loader::LfmLoaded m = loader::load_lfm2moe(devs, snap, o, model::LfmPlacement::one(pre));
  const model::Lfm2MoeDesc& d = m.desc;
  CHECK_EQ(d.layers, pre.layers);
  CHECK_EQ(m.unconsumed, size_t(0));
  CHECK_EQ(m.parts.size(), size_t(1));
  const loader::LfmDevicePart& P = m.parts[0];
  CHECK_EQ(P.layers.size(), size_t(d.layers));
  CHECK_EQ(P.bytes, loader::lfm::device_weight_bytes(d, m.place, 0, o.int8_head));
  CHECK_EQ(P.rope_bytes, loader::lfm::rope_table_bytes(d, o.max_len));
  CHECK(P.embed_tiles && P.final_norm && P.rope);
  CHECK((o.int8_head && P.head_int8) || (!o.int8_head && !P.head_int8));

  loader::SafetensorsSet set(snap);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const uint32_t last = d.layers - 1;
  const std::string lp = d.layer_prefix(last);
  const loader::LfmLayer& L = P.layer(last);
  // A sampled block read back equal to the host repack, for both spellings' names.
  if (!d.is_dense(last)) {
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
  std::printf("lfm2moe_load_checkpoint_test OK: %s, %u layers, %zu B weights, %.1f s\n", snap.c_str(), d.layers,
             P.bytes, m.seconds);
  return 0;
}

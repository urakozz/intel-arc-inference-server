// Spec 23b Task 3: load_muse_glimmer on the card (box only; label `checkpoint;muse`).
//   muse_load_checkpoint_test <snapshot> [layers] [split]
// Loads the checkpoint (make_synth.py's synthetic one - muse_oracle.sh synth-ckpt -, or the real one, either
// Vishva007 spelling; `layers` > 0: development mode; `split` > 0: two devices, layers [0, split) | [split, L))
// with the int8 head, prints the load line, and asserts: zero unconsumed tensors, every part's bytes equal to
// muse_device_weight_bytes (the planner's figure), and device bytes read back against the host repack at the
// columns a wrong fusion breaks (Review Focus 1: q || gate || k || v at 0 / 4095 / 4096 / 8191 / 8192 / 8447 / 8448
// / 8703, gate || up at 0 / 19967 / 19968 / 39935 - every k-row word and every scale of each), every layer's norm
// block, the embedding's first and last rows, the final norm. Exit 77 (SKIP) when the snapshot is not on this
// machine, or two devices are asked of a one-GPU host.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/muse_layout.h"
#include "loader/muse_loader.h"
#include "loader/muse_repack.h"
#include "loader/snapshot.h"

namespace {

template <class T>
std::vector<T> readback(l0::CmdList& imm, const l0::Mem& m, size_t off_elems, size_t n) {
  std::vector<T> v(n);
  imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off_elems * sizeof(T), n * sizeof(T));
  return v;
}

std::vector<uint32_t> columns(const model::MuseLinear& ln) {
  std::vector<uint32_t> c;
  uint32_t at = 0;
  for (uint32_t n : ln.part_n) {
    c.push_back(at);
    c.push_back(at + n - 1);
    at += n;
  }
  return c;
}

// Column c of a layout-0 device weight (every k-row word, every group's scale) against the host repack's.
void check_linear(l0::CmdList& imm, const loader::DeviceWeight& w, const loader::MuseHostLinear& h,
                  const model::MuseLinear& ln) {
  const uint32_t K = ln.shape.K, N = ln.shape.N;
  CHECK(w.shape.K == K && w.shape.N == N && w.shape.layout == 0 && w.scales);
  CHECK_EQ(w.mem.size(), h.words.size() * 4);
  CHECK_EQ(w.scales->size(), h.scales.size() * 2);
  for (uint32_t c : columns(ln)) {
    for (uint32_t r = 0; r < K / 8; ++r) CHECK_EQ(readback<uint32_t>(imm, w.mem, size_t(r) * N + c, 1)[0], h.words[size_t(r) * N + c]);
    for (uint32_t g = 0; g < K / 64; ++g)
      CHECK_EQ(readback<uint16_t>(imm, *w.scales, size_t(g) * N + c, 1)[0], h.scales[size_t(g) * N + c]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "Vishva007/Muse-Glimmer-30B-W4A16-AutoRound";
  const uint32_t layers = argc > 2 ? uint32_t(std::stoul(argv[2])) : 0;
  const uint32_t split = argc > 3 ? uint32_t(std::stoul(argv[3])) : 0;
  std::string snap;
  try {
    snap = loader::resolve_muse_snapshot(arg);
    if (!loader::is_muse_checkpoint(snap)) throw std::runtime_error("not a muse_glimmer checkpoint");
  } catch (const std::exception& e) {
    std::printf("SKIP: no Muse-Glimmer checkpoint at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  if (split != 0 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: split %u asks for two devices; this host has %u\n", split, l0::Context::gpu_count());
    return 77;
  }
  const model::MuseGlimmerDesc pre = loader::muse_checkpoint_desc(snap, layers);
  l0::Context ctx0(0);
  std::unique_ptr<l0::Context> ctx1;
  std::vector<l0::Context*> devs = {&ctx0};
  if (split != 0) {
    ctx1 = std::make_unique<l0::Context>(ctx0, 1);
    devs.push_back(ctx1.get());
  }
  const model::MusePlacement pl = split ? model::MusePlacement::two(pre, split) : model::MusePlacement::one(pre);
  const uint32_t max_len = 8192;
  const loader::MuseWeights m = loader::load_muse_glimmer(devs, snap, max_len, pl, true, layers);
  const model::MuseGlimmerDesc& d = m.desc;
  CHECK_EQ(d.layers, pre.layers);
  CHECK_EQ(m.unconsumed, size_t(0));
  CHECK_EQ(m.parts.size(), size_t(pl.devices));
  for (uint32_t dev = 0; dev < pl.devices; ++dev) {
    CHECK_EQ(m.parts[dev].bytes, loader::muse_device_weight_bytes(d, pl, dev, true));
    CHECK_EQ(m.parts[dev].layers.size(), size_t(pl.count(dev)));
  }
  CHECK_EQ(m.total_bytes(), loader::muse_device_weight_bytes(d, model::MusePlacement::one(d), 0, true));

  // The device bytes against the host repack (an independent read of the same shards).
  const loader::MuseCheckpoint ck(snap, layers);
  for (uint32_t l : {0u, d.layers - 1}) {
    const loader::MuseDevicePart& P = m.parts[pl.device_of(l)];
    l0::CmdList imm = l0::CmdList::immediate(*devs[P.device]);
    const loader::MuseLayer& L = P.layer(l);
    const loader::MuseHostLayer h = ck.layer(l);
    check_linear(imm, *L.qgkv, h.qgkv, d.linear(model::MuseLinearId::QGKV));
    check_linear(imm, *L.oproj, h.oproj, d.linear(model::MuseLinearId::OProj));
    check_linear(imm, *L.gate_up, h.gate_up, d.linear(model::MuseLinearId::GateUp));
    check_linear(imm, *L.down, h.down, d.linear(model::MuseLinearId::Down));
    const std::vector<float> nb = h.norm_block(), dn = readback<float>(imm, *L.norms, 0, nb.size());
    CHECK(std::memcmp(nb.data(), dn.data(), nb.size() * 4) == 0);
  }
  {
    l0::CmdList imm = l0::CmdList::immediate(*devs[0]);
    for (uint32_t row : {0u, d.vocab - 1}) {
      const std::vector<uint16_t> e = readback<uint16_t>(imm, *m.parts[0].embed, size_t(row) * d.hidden, d.hidden);
      CHECK(std::memcmp(e.data(), ck.embed() + size_t(row) * d.hidden, size_t(d.hidden) * 2) == 0);
    }
    l0::CmdList imm1 = l0::CmdList::immediate(*devs.back());
    const std::vector<float> fn = ck.final_norm(), df = readback<float>(imm1, *m.parts.back().final_norm, 0, fn.size());
    CHECK(std::memcmp(fn.data(), df.data(), fn.size() * 4) == 0);
    CHECK(m.parts.back().lm_head->kind == model::WeightKind::Int8);
    CHECK_EQ(m.parts.back().lm_head->mem.size() + m.parts.back().lm_head->scales->size(), loader::muse_lm_head_bytes(d, true));
  }
  std::printf("muse_load_checkpoint_test OK: %s; %.1f s (int8 head %.2f s)\n", m.line.c_str(), m.seconds,
              m.lm_head_quant_seconds);
  return 0;
}

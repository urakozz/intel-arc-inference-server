// Spec 23c Task 4 (MG4 on decode): MuseEngine on one card. Box only (labels checkpoint;muse).
//   muse_decode_test <checkpoint> <ids> [bf16]
// <checkpoint>: the 4-layer synthetic one (tools/box_validate/muse_oracle.sh synth-ckpt: real widths, random weights)
// or the real one.
//
//   1. plan == allocation, per device and component: the weights (loader::muse_device_weight_bytes), the full KV and
//      the rings (runtime::muse::plan_at's kv / rings), the decode scratch (decode_scratch_bytes); the list's launches
//      == decode_launches (733 / 681 at 52 layers, here scaled to the layers) and the formula 1 + 14 L + 4 printed
//   2. replay determinism: two reset + ingest + generate(32) runs give bitwise equal tokens, logits, every full
//      layer's KV rows and every sliding layer's whole ring
//   3. the ring: after 4201 positions (0 .. 4200) a sliding layer's 4096-slot ring holds exactly positions 105 .. 4200
//      in their slots (p & 4095), each the K row written at its own step (read back through read_kv)
//   4. --layers 2 (development mode): the engine loads layers [0, 2), its list is 1 + 2 x 14 + 4 launches, and its
//      residual tap at layers 0 and 1 equals the whole model's (the same layers, the same rows) - bitwise
// Exit 77 (SKIP) when the checkpoint or the ids are absent.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "check.h"
#include "loader/muse_layout.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/muse_rig.h"

namespace {
constexpr uint32_t kGen = 32, kRingPos = 4201;
namespace rm = runtime::muse;

struct Snapshot {
  std::vector<float> logits;
  std::vector<uint32_t> tokens;
  std::vector<std::vector<uint16_t>> kv;   // per layer: a full layer's K then V over [0, end); a sliding layer's rings
};

Snapshot run(rm::MuseEngine& e, const std::vector<uint32_t>& ids) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  e.reset();
  e.ingest(ids);
  Snapshot s;
  s.tokens = e.generate(kGen);
  s.logits = e.read_logits();
  const uint32_t end = e.pos();
  for (uint32_t l = 0; l < d.layers; ++l) {
    std::vector<uint16_t> k, v;
    if (d.is_full(l)) {
      k = e.read_kv(l, 0, end, false);
      v = e.read_kv(l, 0, end, true);
    } else {
      k = e.read_ring(l, false);
      v = e.read_ring(l, true);
    }
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = muse_rig::muse_snapshot(argc > 1 ? argv[1] : "");
  if (snap.empty()) {
    std::printf("SKIP: no Muse-Glimmer checkpoint at '%s' (muse_oracle.sh synth-ckpt)\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  bool bf16 = false;
  for (int i = 3; i < argc; ++i) bf16 = bf16 || std::string(argv[i]) == "bf16";
  const model::MuseGlimmerDesc d0 = loader::muse_checkpoint_desc(snap);
  std::vector<uint32_t> ids = muse_rig::read_ids(argc > 2 ? argv[2] : "", d0.vocab);
  if (ids.empty()) {
    std::printf("SKIP: no ids at '%s'\n", argc > 2 ? argv[2] : "");
    return 77;
  }
  muse_rig::Rig rig;
  rm::MuseOptions o;
  o.max_len = 8192;
  o.int8_head = !bf16;
  o.debug_tap = true;
  muse_rig::build(rig, snap, 1, o);
  rm::MuseEngine& e = *rig.eng;
  const model::MuseGlimmerDesc& d = e.weights().desc;

  // 1. plan == allocation; the launch formula
  {
    const rm::MusePlan& p = e.plan();
    for (uint32_t dev = 0; dev < e.devices(); ++dev) {
      const rm::MuseDevicePlan& dp = p.devices[dev];
      CHECK_EQ(e.weights().parts[dev].bytes, dp.weights);
      CHECK_EQ(e.weights().parts[dev].bytes, loader::muse_device_weight_bytes(d, e.weights().placement, dev, !bf16));
      CHECK_EQ(e.buffers(dev).kv_bytes(), dp.kv);
      CHECK_EQ(e.buffers(dev).ring_bytes(), dp.rings);
      CHECK_EQ(e.buffers(dev).scratch_bytes(), rm::decode_scratch_bytes(d, e.max_len(), rm::kM, e.attention()));
      CHECK_EQ(dp.scratch, rm::decode_reserve(d, e.max_len(), e.devices(), e.attention()).decode_scratch);
      CHECK(e.buffers(dev).scratch_bytes() <= dp.scratch);
    }
    const size_t per = e.attention() == rm::MuseAttn::Eager ? 13 : 14;
    CHECK_EQ(e.launches(), rm::decode_launches(d, e.weights().placement, e.attention(), e.handoff()));
    CHECK_EQ(e.launches(), 1 + per * d.layers + 4);
    std::printf("plan == allocation; %zu launches a token = 1 + %zu x %u + 4 (%s attention; 52 layers: %zu)\n%s\n",
                e.launches(), per, d.layers, rm::muse_attn_name(e.attention()), 1 + per * 52 + 4, e.memory_line().c_str());
  }
  // 2. replay determinism
  {
    const Snapshot a = run(e, ids), b = run(e, ids);
    CHECK(a.tokens == b.tokens);
    CHECK(a.logits == b.logits);
    CHECK(a.kv == b.kv);
    std::printf("MG4: two runs of %zu ids + %u generated bitwise equal (tokens, logits, full KV, rings)\n", ids.size(), kGen);
  }
  // 3. the ring at 4200
  {
    e.reset();
    uint32_t sl = 0;
    while (d.is_full(sl)) ++sl;
    std::vector<uint16_t> written(size_t(kRingPos) * d.kv_n());
    for (uint32_t p = 0; p < kRingPos; ++p) {
      e.ingest({ids[p % ids.size()]});
      const std::vector<uint16_t> r = e.read_kv(sl, p, 1, false);
      std::memcpy(written.data() + size_t(p) * d.kv_n(), r.data(), size_t(d.kv_n()) * 2);
    }
    CHECK_EQ(e.pos(), kRingPos);
    const uint32_t lo = kRingPos - model::MuseGlimmerDesc::kRing;   // 105
    const std::vector<uint16_t> ring = e.read_ring(sl, false);
    for (uint32_t p = lo; p < kRingPos; ++p)
      CHECK(std::memcmp(ring.data() + size_t(p & (model::MuseGlimmerDesc::kRing - 1)) * d.kv_n(),
                        written.data() + size_t(p) * d.kv_n(), size_t(d.kv_n()) * 2) == 0);
    std::printf("ring: layer %u at pos %u holds positions %u..%u in their slots (p & 4095)\n", sl, kRingPos, lo,
                kRingPos - 1);
  }
  // 4. --layers 2: the same ids and the whole model's own continuation fed to both; layers 0..1 bitwise
  if (d.layers > 2) {
    const uint32_t hidden = d.hidden;   // `d` and `e` belong to the engine rebuilt below
    const Snapshot a = run(e, ids);     // generate(32) fed tokens[0..31]: the last replay consumed tokens[31]
    const std::vector<uint16_t> whole = e.read_debug_resid();
    rm::MuseOptions o2 = o;
    o2.layers = 2;
    muse_rig::build(rig, snap, 1, o2);
    rm::MuseEngine& e2 = *rig.eng;
    CHECK_EQ(e2.weights().desc.layers, 2u);
    const size_t per = e2.attention() == rm::MuseAttn::Eager ? 13 : 14;
    CHECK_EQ(e2.launches(), 1 + per * 2 + 4);
    e2.reset();
    e2.ingest(ids);
    e2.ingest(a.tokens);
    const std::vector<uint16_t> part = e2.read_debug_resid();
    CHECK(std::memcmp(part.data(), whole.data(), size_t(2) * hidden * 2) == 0);
    std::printf("--layers 2: %zu launches; layers 0..1's residual == the whole model's, bitwise\n", e2.launches());
  }
  std::puts("muse_decode_test OK");
  return 0;
}

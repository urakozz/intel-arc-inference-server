// Spec 20c Task 5 (KL3 on decode): KolibriEngine on one card. Box only (label checkpoint;golden;kolibri).
//   kolibri_decode_test <checkpoint> <ids> [int8]
// <checkpoint>: a synthetic one (tools/box_validate/kolibri_oracle.sh synth) or the real one (with two
// cards, spec 20c Task 6, kolibri_pp_test covers it; one card here takes a synthetic or a --layers
// checkpoint: the real model does not fit one card - SKIP).
//
//   1. the plan == the allocation, component by component (runtime::kolibri::plan vs memory_use);
//      the list's launches == decode_launches (756 / 856 scaled to the layers)
//   2. K3 replay determinism: two reset + ingest + generate(32) runs give bitwise equal logits, route
//      rows, every full layer's KV rows and every sliding layer's ring
//   3. the ring: after 600 positions a sliding layer's ring holds exactly the keys of positions 88..599
//      in their slots (p & 4095), each the K row written at its own step (read back through read_kv)
// Exit 77 (SKIP) when the checkpoint is absent or does not fit one card.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "check.h"
#include "kolibri_rig.h"
#include "loader/kolibri1_layout.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace {
constexpr uint32_t kMaxLen = 4096, kGen = 32, kRingPos = 600;

struct Snapshot {
  std::vector<float> logits;
  std::vector<uint32_t> routes, tokens;
  std::vector<std::vector<uint16_t>> kv;   // per layer: K then V over the positions it holds
};

Snapshot run(runtime::kolibri::KolibriEngine& e, const std::vector<uint32_t>& ids) {
  const model::Kolibri1Desc& d = e.model().desc;
  e.reset();
  e.ingest(ids);
  Snapshot s;
  s.tokens = e.generate(kGen);
  s.logits = e.read_logits();
  s.routes = e.read_routes();
  const uint32_t end = e.pos();
  for (uint32_t l = 0; l < d.layers; ++l) {
    const uint32_t first = d.is_sliding(l) && end > d.window ? end - d.window : 0, count = end - first;
    std::vector<uint16_t> k = e.read_kv(l, first, count, false), v = e.read_kv(l, first, count, true);
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = kolibri_rig::kolibri_snapshot(argc > 1 ? argv[1] : "");
  if (snap.empty()) {
    std::printf("SKIP: no Kolibri-1 checkpoint at '%s'\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  const bool int8 = argc > 3 && std::string(argv[3]) == "int8";
  const model::Kolibri1Desc d0 = loader::kolibri1_checkpoint_desc(snap);
  if (!runtime::kolibri::fits_one_card(d0, int8, size_t(32530000000ull), size_t(1.5e9))) {
    std::printf("SKIP: %s (%u layers) does not fit one card - kolibri_pp_test runs it on two\n", snap.c_str(), d0.layers);
    return 77;
  }
  std::vector<uint32_t> ids = kolibri_rig::read_ids(argc > 2 ? argv[2] : "", d0.vocab);
  CHECK(!ids.empty());
  kolibri_rig::Rig rig;
  kolibri_rig::Options o;
  o.max_len = kMaxLen;
  o.int8_head = int8;
  kolibri_rig::build(rig, snap, o);
  runtime::kolibri::KolibriEngine& e = *rig.eng;
  const model::Kolibri1Desc& d = e.model().desc;

  // 1. plan == allocation; the launch count
  {
    const std::vector<runtime::kolibri::DevicePlan> pl =
        runtime::kolibri::plan(d, e.model().placement, kMaxLen, int8, false, e.attention());
    const runtime::MemoryComponents got = e.memory_use(0);
    CHECK_EQ(got.model, pl[0].model);
    CHECK_EQ(got.kv, pl[0].kv);
    CHECK_EQ(got.decode_state, pl[0].decode_state);
    CHECK_EQ(e.launches(), runtime::kolibri::decode_launches(d, e.model().placement, e.attention(), runtime::kDefaultPpHandoff));
    std::printf("plan == allocation; %zu launches a token (%s attention); %s\n", e.launches(),
                runtime::kolibri::kol_attn_name(e.attention()), e.memory_line().c_str());
  }
  // 2. K3: replay determinism
  {
    const Snapshot a = run(e, ids), b = run(e, ids);
    CHECK(a.tokens == b.tokens);
    CHECK(a.logits == b.logits);
    CHECK(a.routes == b.routes);
    CHECK(a.kv == b.kv);
    std::printf("K3: two runs of %zu ids + %u generated bitwise equal (logits, routes, full KV, rings)\n", ids.size(),
                kGen);
  }
  // 3. the ring at 600: every sliding layer's slot of position p holds the K row written at step p
  {
    e.reset();
    std::vector<uint32_t> cyc(kRingPos);
    for (uint32_t i = 0; i < kRingPos; ++i) cyc[i] = ids[i % ids.size()];
    uint32_t sl = 0;
    while (!d.is_sliding(sl)) ++sl;
    std::map<uint32_t, std::vector<uint16_t>> written;
    for (uint32_t p = 0; p < kRingPos; ++p) {
      e.ingest({cyc[p]});
      written[p] = e.read_kv(sl, p, 1, false);
    }
    CHECK_EQ(e.pos(), kRingPos);
    const uint32_t lo = kRingPos - (d.window - 1);   // 88: the keys position 600's query will see, less itself
    const std::vector<uint16_t> ring = e.read_kv(sl, lo, kRingPos - lo, false);
    for (uint32_t p = lo; p < kRingPos; ++p)
      CHECK(std::memcmp(ring.data() + size_t(p - lo) * d.kv_n(), written[p].data(), size_t(d.kv_n()) * 2) == 0);
    std::printf("ring: layer %u at pos %u holds positions %u..%u in their slots\n", sl, kRingPos, lo, kRingPos - 1);
  }
  std::puts("kolibri_decode_test OK");
  return 0;
}

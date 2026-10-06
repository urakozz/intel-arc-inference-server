// Spec 16b: the pipeline-parallel planner on the host - the stages, the weights by device,
// the per-device memory plan, the auto split and the auto length - for Qwen3.8, Agnes and
// Ornith, and K2-Horizon's bytes per card through the descriptor-free balance (K2 has its
// own engine; PP for it is future work). No device, no checkpoint.
#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "loader/k2_layout.h"
#include "model/k2_horizon.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/k2/k2_sizes.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"

namespace {
using model::WeightKind;
using runtime::KvCache;

// The measured Qwen3.8 load (memory_plan_test's constant: docs/13's printout less the RoPE
// table) and the derived int8-head / Agnes figures beside it.
constexpr size_t kQwenBf16Weights = 18082777088ull;
constexpr size_t kQwenInt8Weights = 16812371968ull;
constexpr size_t kAgnesInt8Weights = 19640258304ull;
// One B70 as memory_line() prints it (memory_plan_test's kDevice), and the default reserve.
constexpr size_t kDevice = 32530000000ull;
constexpr size_t kReserve = size_t(runtime::kDefaultReserveGb * 1e9);
constexpr std::array<size_t, 2> kTwo{kDevice, kDevice};
constexpr uint32_t kTrained = 262144;

bool throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void check_stage_sizes() {
  // The stage overload at the model's own counts is the single-card formula.
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()})
    for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
      for (uint32_t len : {4096u, 16384u, 131072u}) {
        const runtime::PersistentSizes a = runtime::PersistentDims::sizes(len, *d, kv);
        const runtime::PersistentSizes b =
            runtime::PersistentDims::stage_sizes(len, *d, kv, d->gdn_layers, d->fa_layers);
        CHECK_EQ(a.control, b.control);
        CHECK_EQ(a.gdn_state, b.gdn_state);
        CHECK_EQ(a.conv_ring, b.conv_ring);
        CHECK_EQ(a.kv_k, b.kv_k);
        CHECK_EQ(a.kv_v, b.kv_v);
      }
}

void check_weights() {
  // The descriptor's device forms sum to the loads: Qwen3.8's measured total exactly, the
  // derived int8-head and Agnes figures exactly.
  CHECK_EQ(runtime::pp_weights(model::qwen38(), WeightKind::Bf16).total(), kQwenBf16Weights);
  CHECK_EQ(runtime::pp_weights(model::qwen38(), WeightKind::Int8).total(), kQwenInt8Weights);
  CHECK_EQ(runtime::pp_weights(model::agnes(), WeightKind::Int8).total(), kAgnesInt8Weights);
  const runtime::PpWeights q = runtime::pp_weights(model::qwen38(), WeightKind::Bf16);
  CHECK_EQ(q.layer.size(), size_t{64});
  CHECK_EQ(q.layer[0], size_t{204841600});   // a GDN layer: qkv||z, a||b, out, gate||up, down, small
  CHECK_EQ(q.layer[3], size_t{197797888});   // an FA layer: qkv, o, gate||up, down, small
  CHECK_EQ(q.embed, size_t{2542796800});
  CHECK_EQ(q.head, size_t{2542796800} + 20480);   // bf16 lm_head + the final norm
  // Ornith: every layer carries its 257 expert blocks (loader::moe_bytes), no dense MLP rows.
  const runtime::PpWeights o = runtime::pp_weights(model::ornith(), WeightKind::Bf16);
  CHECK_EQ(o.layer.size(), size_t{40});
  CHECK(o.layer[0] > size_t{400} * 1000 * 1000);
}

void check_stages() {
  const model::ModelDesc& q = model::qwen38();
  const std::array<runtime::PpStage, 2> s = runtime::pp_stages(q, 32);
  CHECK_EQ(s[0].first, 0u);
  CHECK_EQ(s[0].last, 32u);
  CHECK_EQ(s[0].gdn, 24u);
  CHECK_EQ(s[0].fa, 8u);
  CHECK_EQ(s[1].first, 32u);
  CHECK_EQ(s[1].gdn_first, 24u);
  CHECK_EQ(s[1].fa_first, 8u);
  CHECK_EQ(s[1].gdn, 24u);
  CHECK_EQ(s[1].fa, 8u);
  // An uneven split: layers [0, 29) = 22 GDN + 7 FA (FA at 3, 7, ..., 27).
  const std::array<runtime::PpStage, 2> u = runtime::pp_stages(q, 29);
  CHECK_EQ(u[0].gdn, 22u);
  CHECK_EQ(u[0].fa, 7u);
  CHECK_EQ(u[1].gdn_first, 22u);
  CHECK_EQ(u[1].fa_first, 7u);
  CHECK_EQ(u[1].gdn, 26u);
  CHECK_EQ(u[1].fa, 9u);
  // The refusals: no layer for a device, or a device without one kind.
  for (uint32_t bad : {0u, 1u, 2u, 3u, 63u, 64u, 100u})
    CHECK(throws([&] { runtime::require_split(q, bad); }));
  for (uint32_t ok : {4u, 32u, 62u}) runtime::require_split(q, ok);
  // Every model, every legal split: the two stages' compute launches are the single-card
  // list's (runtime::decode_launches: 774 / 870 / 526), cut in two.
  struct M {
    const model::ModelDesc* d;
    size_t launches;
  };
  for (const M m : {M{&model::qwen38(), 774}, M{&model::agnes(), 870}, M{&model::ornith(), 526}}) {
    size_t legal = 0;
    for (uint32_t sp = 1; sp < m.d->layers; ++sp) {
      if (throws([&] { runtime::require_split(*m.d, sp); })) continue;
      ++legal;
      const std::array<runtime::PpStage, 2> st = runtime::pp_stages(*m.d, sp);
      CHECK_EQ(st[0].gdn + st[1].gdn, m.d->gdn_layers);
      CHECK_EQ(st[0].fa + st[1].fa, m.d->fa_layers);
      CHECK_EQ(runtime::pp_stage_launches(*m.d, st[0]) + runtime::pp_stage_launches(*m.d, st[1]),
               m.launches);
    }
    CHECK_EQ(legal, size_t{m.d->layers - 5});   // [4, layers - 2]: layer layers-1 is FA
  }
}

// One model's planner numbers at `len`, printed (they are the report's table) and checked
// against a brute force over every legal split.
uint32_t check_auto(const char* label, const model::ModelDesc& d, WeightKind head, uint32_t len,
                    KvCache kv) {
  const runtime::PpWeights w = runtime::pp_weights(d, head);
  const uint32_t s = runtime::pp_auto_split(d, w, len, kv);
  const runtime::PpPlan p = runtime::pp_plan(d, s, len, w, kv);
  for (uint32_t t = 1; t < d.layers; ++t) {
    if (throws([&] { runtime::require_split(d, t); })) continue;
    CHECK(runtime::pp_plan(d, t, len, w, kv).max_total() >= p.max_total());
  }
  // Both devices together hold the model, its state at `len`, two scratches, two RoPE
  // tables and the hand-off buffers.
  const runtime::MemoryPlan one = runtime::plan(d, len, false, w.total(), runtime::PrefillPath{false},
                                                {}, kv);
  CHECK_EQ(p.dev[0].kv + p.dev[1].kv, one.kv);
  CHECK_EQ(p.dev[0].weights + p.dev[1].weights, w.total());
  // = one card's plan + a second RoPE table, control block and decode scratch + the link.
  CHECK_EQ(p.dev[0].total() + p.dev[1].total(),
           one.total() + p.dev[0].rope + runtime::PersistentDims::sizes(len, d, kv).control +
               runtime::DecodeScratchDims::sizes(len, d).total() + runtime::pp_link_bytes(d, 0) +
               runtime::pp_link_bytes(d, 1));
  std::printf("%-28s len %6u: split %2u (%u + %u layers), device 0 %.3f GB, device 1 %.3f GB"
              " (weights %.3f / %.3f, kv %.3f / %.3f)\n",
              label, len, s, p.dev[0].stage.layers(), p.dev[1].stage.layers(),
              p.dev[0].total() / 1e9, p.dev[1].total() / 1e9, p.dev[0].weights / 1e9,
              p.dev[1].weights / 1e9, p.dev[0].kv / 1e9, p.dev[1].kv / 1e9);
  return s;
}

void check_auto_splits() {
  const model::ModelDesc& q = model::qwen38();
  // Qwen3.8 with its bf16 head: the embedding and lm_head are the same 2.54 GB, so the even
  // split is the balanced one.
  CHECK_EQ(check_auto("qwen3.8 bf16 head", q, WeightKind::Bf16, 16384, KvCache::Bf16), 32u);
  // The int8 head (1.27 GB on device 1) moves layers to device 1: the lm_head card is not
  // the heavier one any more, device 0's embedding is.
  // (Derived numbers, pinned so a change to a size function shows up here: 29 / 35 layers,
  // 9.009 / 9.101 GB.)
  CHECK_EQ(check_auto("qwen3.8 int8 head", q, WeightKind::Int8, 16384, KvCache::Bf16), 29u);
  // At 262144 a FA layer carries 1.07 GB of KV: the lumps decide, and 32 stays best.
  CHECK_EQ(check_auto("qwen3.8 int8 head 262144", q, WeightKind::Int8, 262144, KvCache::Bf16), 32u);
  CHECK_EQ(check_auto("qwen3.8 bf16 head 262144", q, WeightKind::Bf16, 262144, KvCache::Bf16), 32u);
  CHECK_EQ(check_auto("qwen3.8 int8 head int8 KV", q, WeightKind::Int8, 262144, KvCache::Int8), 31u);
  CHECK_EQ(check_auto("agnes int8 head", model::agnes(), WeightKind::Int8, 16384, KvCache::Bf16), 33u);
  CHECK_EQ(check_auto("agnes int8 head 131072", model::agnes(), WeightKind::Int8, 131072,
                      KvCache::Bf16), 35u);
  CHECK_EQ(check_auto("ornith bf16 head", model::ornith(), WeightKind::Bf16, 16384, KvCache::Bf16), 20u);
  CHECK_EQ(check_auto("ornith int8 head 262144", model::ornith(), WeightKind::Int8, 262144,
                      KvCache::Bf16), 20u);
}

void check_auto_len() {
  const model::ModelDesc& q = model::qwen38();
  const runtime::PpWeights w = runtime::pp_weights(q, WeightKind::Bf16);
  // Spec 16 P3 / S3: Qwen3.8 at 262144 with bf16 KV fits two cards (one card's auto is
  // ~131k at most with the bf16 head).
  const runtime::PpChoice c = runtime::pp_auto_split_and_len(q, w, kTwo, kReserve, kTrained);
  CHECK(c.split != 0);
  CHECK_EQ(c.max_len, kTrained);
  const uint32_t one = runtime::max_len_that_fits(q, false, w.total(), kDevice, kReserve, kTrained,
                                                  runtime::PrefillPath{false});
  CHECK(one < kTrained);
  std::printf("qwen3.8 bf16 head auto: one card %u, --pp 2 %u at split %u\n", one,
              c.max_len, c.split);
  // The min over devices: a length that fits both, the next quantum not, at a fixed split;
  // a smaller device 1 sets the length.
  const std::array<size_t, 2> small{kDevice, size_t{12} * 1000 * 1000 * 1000};
  const uint32_t at = runtime::pp_max_len_that_fits(q, 32, w, small, kReserve, kTrained);
  CHECK(at != 0 && at < kTrained);
  const auto fits = [&](uint32_t len) {
    const runtime::PpPlan p = runtime::pp_plan(q, 32, len, w);
    return p.dev[0].total() + kReserve <= small[0] && p.dev[1].total() + kReserve <= small[1];
  };
  CHECK(fits(at));
  CHECK(!fits(at + runtime::kMaxLenQuantum));
  // Nothing fits: 0, and both-auto says {0, 0}.
  const std::array<size_t, 2> tiny{size_t{1} << 30, size_t{1} << 30};
  CHECK_EQ(runtime::pp_max_len_that_fits(q, 32, w, tiny, kReserve, kTrained), 0u);
  CHECK_EQ(runtime::pp_auto_split_and_len(q, w, tiny, kReserve, kTrained).split, 0u);
  CHECK(throws([&] { runtime::pp_max_len_that_fits(q, 32, w, kTwo, kReserve, 100); }));
  const std::string line = runtime::pp_describe(runtime::pp_plan(q, 32, 16384, w), kTwo, kReserve);
  CHECK(line.find("split 32 (device 0 layers [0, 32): 24 GDN + 8 FA, device 1 layers [32, 64)") !=
        std::string::npos);
  CHECK(line.find("\n  device 1: model ") != std::string::npos);
  std::printf("%s\n", line.c_str());
}

void check_landing() {
  for (const model::ModelDesc* d : {&model::qwen38(), &model::ornith()}) {
    const runtime::PpLandingLayout l = runtime::pp_landing_layout(*d);
    const runtime::DecodeScratchSizes s = runtime::DecodeScratchDims::sizes(4096, *d);
    CHECK_EQ(l.resid_bytes, s.resid);
    CHECK_EQ(l.sumsq_bytes, s.norm_sumsq);
    CHECK_EQ(l.sumsq_off % runtime::kPpPage, size_t{0});
    CHECK(l.sumsq_off >= l.resid_bytes);
    CHECK_EQ(l.stamp_off, l.sumsq_off + l.sumsq_bytes);
    CHECK_EQ(l.flag_off % runtime::kPpPage, size_t{0});   // the flag's own page
    CHECK(l.flag_off > l.stamp_off);
    CHECK_EQ(l.total, l.flag_off + runtime::kPpPage);
  }
  // Spec 20c: the descriptor-free form is the ModelDesc form at the same two sizes, field for
  // field (Qwen3.8, Agnes, Ornith); at Kolibri-1's (one 2560-wide bf16 row, 20 fp32 sums) the
  // regions are page-aligned and the flag has a page of its own.
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()}) {
    const runtime::PpLandingLayout a = runtime::pp_landing_layout(*d);
    const runtime::PpLandingLayout b = runtime::pp_landing_layout(a.resid_bytes, a.sumsq_bytes);
    CHECK(a.resid_bytes == b.resid_bytes && a.sumsq_bytes == b.sumsq_bytes && a.sumsq_off == b.sumsq_off &&
          a.stamp_off == b.stamp_off && a.flag_off == b.flag_off && a.total == b.total);
  }
  {
    const runtime::PpLandingLayout k = runtime::pp_landing_layout(size_t{2560} * 2, size_t{20} * 4);
    CHECK(k.sumsq_off % runtime::kPpPage == 0 && k.sumsq_off >= k.resid_bytes);
    CHECK_EQ(k.stamp_off, k.sumsq_off + size_t{80});
    CHECK(k.flag_off % runtime::kPpPage == 0 && k.flag_off > k.stamp_off);
    CHECK_EQ(k.total, k.flag_off + runtime::kPpPage);
    CHECK_EQ(k.total, size_t{4} * runtime::kPpPage);   // rows 5120 B -> sums at 8192, flag at 12288
  }
  const runtime::PpLandingLayout q = runtime::pp_landing_layout(model::qwen38());
  CHECK_EQ(q.resid_bytes, size_t{8} * 5120 * 2);   // kM rows
  CHECK_EQ(q.sumsq_bytes, size_t{640});
  CHECK_EQ(q.total, size_t{90112});
}

// K2-Horizon: no ModelDesc and its own engine (runtime/k2), so PP does not apply to it in
// 16b. Its bytes per card through the descriptor-free balance, for the record: every layer
// full attention (KV per layer at the length), embed on device 0, lm_head + norm on device 1.
void check_k2_balance() {
  const model::K2Desc& d = model::k2();
  const loader::K2ExpertBytes e = loader::k2_expert_bytes(d);
  for (uint32_t len : {32768u, 131072u}) {
    std::vector<size_t> layer(d.layers, 0);
    for (uint32_t l = 0; l < d.layers; ++l) {
      for (model::K2LinearId id : d.layer_linears(l)) layer[l] += loader::k2_linear_bytes(d, id);
      layer[l] += d.norms_bytes();
      if (!d.is_dense(l)) layer[l] += e.router + e.experts() + d.route_bytes();
      layer[l] += 2 * runtime::k2::kv_layer_bytes(d, len);   // K and V
    }
    const loader::K2WeightBytes wb = loader::k2_weight_bytes(d, /*int8_head=*/true);
    size_t sum = 0;
    for (size_t b : layer) sum += b;
    CHECK_EQ(sum - 2 * runtime::k2::kv_layer_bytes(d, len) * d.layers + wb.embed + wb.lm_head +
                 size_t(d.hidden) * 4,
             wb.total());
    const runtime::PpBalance b =
        runtime::pp_balance(layer, wb.embed, wb.lm_head + size_t(d.hidden) * 4);
    CHECK_EQ(b.split, 24u);   // derived: 24 + 24 at both lengths (every layer holds KV)
    std::printf("k2-horizon int8 head len %6u: split %2u (%u + %u layers), device 0 %.3f GB,"
                " device 1 %.3f GB (weights + KV)\n",
                len, b.split, b.split, d.layers - b.split, b.dev0 / 1e9, b.dev1 / 1e9);
  }
  // The balance itself: three equal layers and a heavy device 1 put two layers on device 0.
  const runtime::PpBalance t = runtime::pp_balance({10, 10, 10}, 0, 15);
  CHECK_EQ(t.split, 2u);
  CHECK_EQ(t.dev0, size_t{20});
  CHECK_EQ(t.dev1, size_t{25});
}

void check_handoff_names() {
  runtime::PpHandoff h = runtime::PpHandoff::Peer;
  CHECK(runtime::parse_pp_handoff("copy", h) && h == runtime::PpHandoff::Copy);
  CHECK(runtime::parse_pp_handoff("peer", h) && h == runtime::PpHandoff::Peer);
  CHECK(!runtime::parse_pp_handoff("p2p", h));
  CHECK_EQ(std::string(runtime::pp_handoff_name(runtime::kDefaultPpHandoff)), std::string("copy"));
}
}  // namespace

int main() {
  check_stage_sizes();
  check_weights();
  check_stages();
  check_auto_splits();
  check_auto_len();
  check_landing();
  check_k2_balance();
  check_handoff_names();
  std::printf("pipeline_plan_test: OK\n");
  return 0;
}

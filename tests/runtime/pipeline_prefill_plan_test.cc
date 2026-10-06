// Spec 16c: the two-card prefill pipeline's device-free half on the host - the chunks (the
// single card's, exactly), the host's order and its rules (the schedule passes its check,
// every mutation of it that drops a rule is caught), the landing slots, and the per-device
// memory plan with a prefill path (16b's decode-only plan unchanged without one). No device.
#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"
#include "runtime/pipeline_prefill_plan.h"
#include "runtime/prefill_chunks.h"

namespace {
using model::WeightKind;
using runtime::KvCache;
using runtime::PpChunk;
using runtime::PpPfOp;
using runtime::PpPfStep;
using runtime::PrefillBackend;

constexpr uint32_t kBlock = runtime::kPpPfBlock;
constexpr size_t kDevice = 32530000000ull;
constexpr size_t kReserve = size_t(runtime::kDefaultReserveGb * 1e9);
constexpr std::array<size_t, 2> kTwo{kDevice, kDevice};

bool throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

// Engine::prefill's loop, restated from engine_prefill.cc as it was before the chunk rule
// moved into runtime/prefill_chunks.h: the (pos, rows) it walks and where its hook fires
// mid-prompt.
std::vector<PpChunk> engine_chunks(uint32_t base, size_t n, uint32_t chunk, bool hooked) {
  std::vector<PpChunk> out;
  uint32_t C = 0;
  for (size_t off = 0; off < n; off += C) {
    C = uint32_t(std::min<size_t>(chunk, n - off));
    if (hooked) C = std::min(C, kBlock - (base + uint32_t(off)) % kBlock);
    PpChunk c;
    c.pos = base + uint32_t(off);
    c.rows = C;
    const uint32_t end = base + uint32_t(off) + C;
    c.hook = hooked && off + C < n && end % kBlock == 0;
    out.push_back(c);
  }
  return out;
}

bool same(const std::vector<PpChunk>& a, const std::vector<PpChunk>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].pos != b[i].pos || a[i].rows != b[i].rows || a[i].hook != b[i].hook) return false;
  return true;
}

void check_chunks() {
  size_t cases = 0;
  for (uint32_t base : {0u, 1u, 300u, 2047u, 2048u, 4095u, 10000u})
    for (size_t n : {size_t{1}, size_t{2}, size_t{63}, size_t{2047}, size_t{2048}, size_t{2049},
                     size_t{4096}, size_t{5000}, size_t{32768}, size_t{65537}})
      for (uint32_t chunk : {1u, 7u, 256u, 1000u, 2048u})
        for (bool hooked : {false, true}) {
          if (chunk == 1 && n > 5000) continue;   // 65k one-row chunks prove nothing more
          CHECK(same(runtime::pp_prefill_chunks(base, n, chunk, hooked), engine_chunks(base, n, chunk, hooked)));
          ++cases;
        }
  // snapshot_test's cases D and E (spec 7): the hooks at 2048 and 4096, the end not one.
  const std::vector<PpChunk> d = runtime::pp_prefill_chunks(0, 5000, 2048, true);
  CHECK_EQ(d.size(), size_t{3});
  CHECK(d[0].hook && d[0].end() == 2048 && d[1].hook && d[1].end() == 4096 && !d[2].hook &&
        d[2].end() == 5000);
  const std::vector<PpChunk> e = runtime::pp_prefill_chunks(300, 4000, 2048, true);
  CHECK_EQ(e.size(), size_t{3});
  CHECK(e[0].rows == 1748 && e[0].hook && e[1].end() == 4096 && e[1].hook && e[2].end() == 4300 &&
        !e[2].hook);
  // A prompt ending on a block end gets one call (the end's, live): its last chunk no hook.
  const std::vector<PpChunk> f = runtime::pp_prefill_chunks(0, 4096, 2048, true);
  CHECK(f[0].hook && !f[1].hook);
  CHECK(throws([] { runtime::pp_prefill_chunks(0, 0, 2048, false); }));
  CHECK(throws([] { runtime::pp_prefill_chunks(0, 10, 0, false); }));
  std::printf("chunks: %zu (base, n, chunk, hooked) cases equal Engine::prefill's loop\n", cases);
}

std::string ops(const std::vector<PpPfStep>& s) {
  std::string o;
  for (const PpPfStep& x : s) o += std::string(runtime::pp_pf_op_name(x.op)) + std::to_string(x.chunk) + " ";
  return o;
}

void check_schedule() {
  // Its shape for n = 3, a hook after chunk 0: the order the header describes.
  std::vector<PpChunk> c3 = runtime::pp_prefill_chunks(0, 5000, 2048, true);
  CHECK_EQ(ops(runtime::pp_prefill_schedule(c3)),
           std::string("Run00 Run10 Run01 Run11 Wait00 Wait10 Hook0 Run02 Run12 Wait01 Wait11 "
                       "Hook1 Head2 Wait02 Wait12 Drain3 "));
  CHECK_EQ(ops(runtime::pp_prefill_schedule(runtime::pp_prefill_chunks(0, 5, 2048, false))),
           std::string("Run00 Run10 Head0 Wait00 Wait10 Drain1 "));
  // Every length, hooks or not: the schedule keeps every rule.
  size_t checked = 0;
  for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{4095}, size_t{4096}, size_t{4097},
                   size_t{20000}, size_t{65536}, size_t{131072}})
    for (uint32_t chunk : {512u, 2048u})
      for (bool hooked : {false, true}) {
        const std::vector<PpChunk> c = runtime::pp_prefill_chunks(hooked ? 100 : 0, n, chunk, hooked);
        const std::vector<PpPfStep> s = runtime::pp_prefill_schedule(c);
        const std::string err = runtime::pp_prefill_check(s, c);
        if (!err.empty()) std::fprintf(stderr, "n %zu chunk %u hooked %d: %s\n", n, chunk, hooked, err.c_str());
        CHECK(err.empty());
        ++checked;
      }
  // Every single step dropped is caught (each one is required), on hooked and plain runs.
  size_t dropped = 0;
  for (bool hooked : {false, true}) {
    const std::vector<PpChunk> c = runtime::pp_prefill_chunks(0, 9000, 2048, hooked);
    const std::vector<PpPfStep> s = runtime::pp_prefill_schedule(c);
    for (size_t i = 0; i < s.size(); ++i) {
      std::vector<PpPfStep> m = s;
      m.erase(m.begin() + long(i));
      CHECK(!runtime::pp_prefill_check(m, c).empty());
      ++dropped;
    }
  }
  // The rules by name: device 0 three chunks ahead (Run0 of chunk 2 before the waits of 0);
  // the hook after the shadow it reads is overwritten; the head before the last hook; device
  // 1 ahead of device 0.
  const std::vector<PpChunk> c = runtime::pp_prefill_chunks(0, 9000, 2048, true);
  const std::vector<PpPfStep> s = runtime::pp_prefill_schedule(c);
  const auto index = [&](const std::vector<PpPfStep>& v, PpPfOp op, uint32_t j) {
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i].op == op && v[i].chunk == j) return i;
    CHECK(false);
    return size_t{0};
  };
  const auto move_before = [&](PpPfOp op, uint32_t j, PpPfOp before, uint32_t k) {
    std::vector<PpPfStep> m = s;
    const PpPfStep x = m[index(m, op, j)];
    m.erase(m.begin() + long(index(m, op, j)));
    m.insert(m.begin() + long(index(m, before, k)), x);
    return runtime::pp_prefill_check(m, c);
  };
  const std::string ahead = move_before(PpPfOp::Run0, 2, PpPfOp::Wait1, 0);
  CHECK(ahead.find("back-pressure") != std::string::npos);
  const std::string ahead1 = move_before(PpPfOp::Wait1, 0, PpPfOp::Run1, 3);   // waits after Run 2
  CHECK(!ahead1.empty());
  const std::string late_hook = move_before(PpPfOp::Hook, 0, PpPfOp::Wait0, 1);
  CHECK(late_hook.find("shadow") != std::string::npos);
  const std::string early_head = move_before(PpPfOp::Head, 4, PpPfOp::Hook, 3);
  CHECK(early_head.find("after the Head") != std::string::npos);
  const std::string d1_first = move_before(PpPfOp::Run1, 1, PpPfOp::Run0, 1);
  CHECK(d1_first.find("device 0 has not") != std::string::npos);
  const std::string hook_early = move_before(PpPfOp::Hook, 0, PpPfOp::Wait1, 0);
  CHECK(hook_early.find("before both devices") != std::string::npos);
  std::printf("schedule: %zu lengths keep every rule; %zu single-step drops and 6 named "
              "reorderings caught (e.g. \"%s\")\n", checked, dropped, ahead.c_str());
}

void check_link() {
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()}) {
    const runtime::PpPfLinkLayout l = runtime::pp_prefill_link_layout(*d);
    const runtime::PrefillScratchSizes s = runtime::PrefillScratchDims::sizes(16384, *d);
    CHECK_EQ(l.resid_bytes, s.resid);
    CHECK_EQ(l.sumsq_bytes, s.norm_sumsq);
    CHECK_EQ(l.resid_bytes, size_t{runtime::PrefillScratchDims::kC} * d->hidden * 2);
    CHECK_EQ(l.sumsq_off % runtime::kPpPage, size_t{0});
    CHECK(l.sumsq_off >= l.resid_bytes);
    CHECK_EQ(l.stamp_off, l.sumsq_off + l.sumsq_bytes);
    CHECK_EQ(l.flag_off % runtime::kPpPage, size_t{0});
    CHECK(l.flag_off > l.stamp_off);
    CHECK(l.slot_bytes >= l.flag_off + runtime::kPpPage);
    CHECK_EQ(l.slot_bytes % runtime::kPpLandingAlign, size_t{0});
    CHECK_EQ(l.total, 2 * l.slot_bytes);
    CHECK_EQ(l.slot(0), size_t{0});
    CHECK_EQ(l.slot(7), l.slot_bytes);
    // pp_send's word counts are 32-bit kernel arguments.
    CHECK(l.resid_bytes / 4 < (size_t{1} << 32));
  }
  const runtime::PpPfLinkLayout q = runtime::pp_prefill_link_layout(model::qwen38());
  CHECK_EQ(q.resid_bytes, size_t{20971520});   // 2048 x 5120 bf16
  CHECK_EQ(q.sumsq_bytes, size_t{163840});     // [20][2048] fp32
  CHECK_EQ(q.slot_bytes, size_t{21168128});
  CHECK_EQ(q.total, size_t{42336256});         // spec 16 §3.3's ~40 MB, derived
  const size_t common = 2 * sizeof(runtime::Control) + 32;
  CHECK_EQ(runtime::pp_prefill_link_bytes(model::qwen38(), 0), common + 2 * runtime::kPpStateWords * 4);
  CHECK_EQ(runtime::pp_prefill_link_bytes(model::qwen38(), 1), q.total + common + 2 * runtime::kPpStateWords * 4);
  // The shadows: two copies of each device's GDN state + conv ring; together, two of the
  // model's (Qwen3.8 166.72 MB once).
  const std::array<runtime::PpStage, 2> st = runtime::pp_stages(model::qwen38(), 32);
  const runtime::PersistentSizes p = runtime::PersistentDims::sizes(4096, model::qwen38(), KvCache::Bf16);
  CHECK_EQ(runtime::pp_prefill_shadow_bytes(model::qwen38(), st[0]) +
               runtime::pp_prefill_shadow_bytes(model::qwen38(), st[1]),
           2 * (p.gdn_state + p.conv_ring));
  std::printf("landing: Qwen3.8 two slots of %zu B (%.1f MB in all); shadows %.1f MB per device\n",
              q.slot_bytes, q.total / 1e6, runtime::pp_prefill_shadow_bytes(model::qwen38(), st[0]) / 1e6);
}

runtime::PrefillPath path(PrefillBackend b) {
  runtime::PrefillPath p;
  p.prefill = true;
  p.backend = b;
  return p;
}

void check_plan() {
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()})
    for (PrefillBackend b : {PrefillBackend::L0, PrefillBackend::L0Int8})
      for (uint32_t len : {16384u, 131072u}) {
        const runtime::PpWeights w = runtime::pp_weights(*d, WeightKind::Int8);
        const uint32_t s = runtime::pp_auto_split(*d, w, len, KvCache::Bf16);
        const runtime::PpPlan dec = runtime::pp_plan(*d, s, len, w, KvCache::Bf16);
        const runtime::PpPlan pf = runtime::pp_plan(*d, s, len, w, KvCache::Bf16, path(b));
        // One card's prefill terms (memory_plan.cc), per device, plus the hand-off.
        const runtime::MemoryPlan one = runtime::plan(*d, len, false, w.total(), path(b), {}, KvCache::Bf16);
        size_t int8 = 0;
        for (uint32_t i = 0; i < 2; ++i) {
          CHECK_EQ(pf.dev[i].model, dec.dev[i].model);
          CHECK_EQ(pf.dev[i].kv, dec.dev[i].kv);
          CHECK_EQ(pf.dev[i].prefill_scratch, one.prefill_scratch);
          CHECK_EQ(pf.dev[i].decode_state, dec.dev[i].decode_state + runtime::pp_prefill_link_bytes(*d, i));
          CHECK_EQ(pf.dev[i].prefill_link, runtime::pp_prefill_link_bytes(*d, i));
          CHECK_EQ(dec.dev[i].prefill_scratch + dec.dev[i].int8 + dec.dev[i].prefill_link, size_t{0});
          int8 += pf.dev[i].int8;
        }
        // The scales split by layer: the two devices' Int8States hold the one card's scales
        // once and the h8 scratch twice.
        if (b == PrefillBackend::L0Int8)
          CHECK_EQ(int8, one.int8 + runtime::int8_scratch_sizes(runtime::prefill_int8_max_k(*d)).total());
        else
          CHECK_EQ(int8, size_t{0});
      }
  CHECK_EQ(runtime::int8_scale_bytes(model::qwen38()),
           runtime::int8_scale_bytes(model::qwen38(), 0, 29) + runtime::int8_scale_bytes(model::qwen38(), 29, 64));
  CHECK_EQ(runtime::int8_scale_bytes(model::ornith()),
           runtime::int8_scale_bytes(model::ornith(), 0, 20) + runtime::int8_scale_bytes(model::ornith(), 20, 40));
  // The refusals: no composed attention and no sycl-tla under the pipeline.
  const runtime::PpWeights w = runtime::pp_weights(model::qwen38(), WeightKind::Bf16);
  runtime::PrefillPath composed = path(PrefillBackend::L0);
  composed.composed_attn = true;
  CHECK(throws([&] { runtime::pp_plan(model::qwen38(), 32, 16384, w, KvCache::Bf16, composed); }));
  CHECK(throws([&] {
    runtime::pp_plan(model::qwen38(), 32, 16384, w, KvCache::Bf16, path(PrefillBackend::SyclTla));
  }));
  // b70-decode's prefill runs: auto split and length with the prefill planned (the bf16 head,
  // the default l0-int8 backend) - derived, printed for the report.
  const runtime::PpChoice c = runtime::pp_auto_split_and_len(model::qwen38(), w, kTwo, kReserve, 262144,
                                                             KvCache::Bf16, path(PrefillBackend::L0Int8));
  CHECK(c.split != 0 && c.max_len == 262144);
  const runtime::PpPlan p = runtime::pp_plan(model::qwen38(), c.split, c.max_len, w, KvCache::Bf16,
                                             path(PrefillBackend::L0Int8));
  const std::string line = runtime::pp_describe(p, kTwo, kReserve);
  CHECK(line.find(", prefill l0-int8") != std::string::npos);
  CHECK(line.find("prefill hand-off") != std::string::npos);
  std::printf("%s\n", line.c_str());
  for (uint32_t len : {16384u, 65536u})
    for (WeightKind head : {WeightKind::Bf16, WeightKind::Int8}) {
      const runtime::PpWeights hw = runtime::pp_weights(model::qwen38(), head);
      const uint32_t s = runtime::pp_auto_split(model::qwen38(), hw, len, KvCache::Bf16,
                                                path(PrefillBackend::L0Int8));
      const runtime::PpPlan q = runtime::pp_plan(model::qwen38(), s, len, hw, KvCache::Bf16,
                                                 path(PrefillBackend::L0Int8));
      std::printf("qwen3.8 %s head, prefill l0-int8, len %6u: split %u (%u + %u layers), %.3f / %.3f GB\n",
                  head == WeightKind::Bf16 ? "bf16" : "int8", len, s, q.dev[0].stage.layers(),
                  q.dev[1].stage.layers(), q.dev[0].total() / 1e9, q.dev[1].total() / 1e9);
    }
}
}  // namespace

int main() {
  check_chunks();
  check_schedule();
  check_link();
  check_plan();
  std::printf("pipeline_prefill_plan_test: OK\n");
  return 0;
}

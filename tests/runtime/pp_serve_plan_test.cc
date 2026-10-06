// Spec 16d on the host: the planner's server terms (runtime::PpExtras - the MTP head on device
// 1 with its embedding replica and buffers, device 0's verify slots, the prefix cache's block
// shadows), the stage verify lists' launch arithmetic, and spec 7's KV snapshot over two
// devices byte for byte the one-card layout (runtime::pp_kv_runs, MTP's head layer included).
// No device, no checkpoint. The table it prints is spec 16 §10's derived memory rows.
//
//   1. PpExtras{} is spec 16b / 16c's plan exactly, for every model, split, length, KV form and
//      prefill path;
//   2. each term where it belongs: device 1 gains the head's weights + the embedding replica
//      (model), MtpBuffers (decode state), the draft vocabulary (its own term), the head's
//      prefill rows (decode state) and on l0-int8 the L0 slab (prefill scratch); device 0 gains
//      only its verify slots - pp_gdn_spec_bytes, against the binaries' SPEC_SLOT_STRIDE;
//   3. the block shadows: two of each device's GDN state + conv ring (+ device 1's hidden row
//      with MTP), only with a prefill path;
//   4. the stage verify lists' launches add up to verify_launches (decode_launches + 10) at
//      every legal split of Qwen3.8, Agnes and Ornith;
//   5. pp_kv_runs: a two-device "save" of synthetic KV equals the one-card save (Engine's
//      kv_runs, restated here) byte for byte - bf16 and int8, with and without the head's
//      layer, three splits, several ranges; a load of those bytes restores both stages;
//   6. b70-serve --pp 2's auto choices with the server's terms (derived): Qwen3.8, Agnes and
//      Ornith with and without MTP, both KV forms - and each beats one card's auto length.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "loader/draft_vocab.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"
#include "runtime/pipeline_prefill_plan.h"

namespace {
using model::WeightKind;
using runtime::KvCache;

constexpr size_t kDevice = 32530000000ull;   // one B70 as memory_line() prints it
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

bool legal(const model::ModelDesc& d, uint32_t s) {
  return !throws([&] { runtime::require_split(d, s); });
}

// The head's device bytes for a descriptor-only plan: the dense head's checkpoint bytes (its
// bf16 linears are repacked, not resized); the CLI plans from the loaded allocations instead.
runtime::PpWeights weights_with_head(const model::ModelDesc& d, WeightKind head) {
  runtime::PpWeights w = runtime::pp_weights(d, head);
  w.mtp = d.mtp_checkpoint_bytes();
  return w;
}

void same(const runtime::PpDevicePlan& a, const runtime::PpDevicePlan& b) {
  CHECK_EQ(a.model, b.model);
  CHECK_EQ(a.kv, b.kv);
  CHECK_EQ(a.decode_state, b.decode_state);
  CHECK_EQ(a.prefill_scratch, b.prefill_scratch);
  CHECK_EQ(a.int8, b.int8);
  CHECK_EQ(a.draft_vocab, b.draft_vocab);
  CHECK_EQ(a.weights, b.weights);
}

const std::vector<runtime::PrefillPath>& paths() {
  static const std::vector<runtime::PrefillPath> p = [] {
    std::vector<runtime::PrefillPath> v{runtime::pp_no_prefill()};
    v.push_back({true, runtime::PrefillBackend::L0Int8, false});
    v.push_back({true, runtime::PrefillBackend::L0, false});
    return v;
  }();
  return p;
}

// ---- 1 -------------------------------------------------------------------------------------
void check_extras_off() {
  size_t n = 0;
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()})
    for (WeightKind h : {WeightKind::Bf16, WeightKind::Int8}) {
      const runtime::PpWeights w = weights_with_head(*d, h);   // w.mtp ignored without PpExtras::mtp
      for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
        for (const runtime::PrefillPath& pf : paths())
          for (uint32_t len : {4096u, 65536u})
            for (uint32_t s : {d->layers / 2, d->layers / 2 + 3}) {
              const runtime::PpPlan a = runtime::pp_plan(*d, s, len, w, kv, pf);
              const runtime::PpPlan b = runtime::pp_plan(*d, s, len, w, kv, pf, runtime::PpExtras{});
              for (uint32_t i = 0; i < 2; ++i) same(a.dev[i], b.dev[i]);
              ++n;
            }
      CHECK_EQ(runtime::pp_auto_split(*d, w, 16384, KvCache::Bf16, paths()[1]),
               runtime::pp_auto_split(*d, w, 16384, KvCache::Bf16, paths()[1], runtime::PpExtras{}));
    }
  // A hook without a prefill path plans nothing (b70-decode never hooks).
  runtime::PpExtras hook_only;
  hook_only.hook = true;
  const runtime::PpWeights w = runtime::pp_weights(model::qwen38(), WeightKind::Int8);
  const runtime::PpPlan a = runtime::pp_plan(model::qwen38(), 32, 16384, w, KvCache::Bf16, runtime::pp_no_prefill());
  const runtime::PpPlan b =
      runtime::pp_plan(model::qwen38(), 32, 16384, w, KvCache::Bf16, runtime::pp_no_prefill(), hook_only);
  for (uint32_t i = 0; i < 2; ++i) same(a.dev[i], b.dev[i]);
  std::printf("extras off: %zu plans identical to spec 16b / 16c's\n", n);
}

// ---- 2 -------------------------------------------------------------------------------------
void check_gdn_spec() {
  // The binaries' SPEC_SLOT_STRIDE (floats per slot, src/kernels/CMakeLists.txt): Qwen3.8
  // 37748736 (48 GDN layers), Agnes 42467328 (54), Ornith 15728640 (30).
  struct M {
    const model::ModelDesc* d;
    size_t slot_floats;
  };
  for (const M m : {M{&model::qwen38(), 37748736}, M{&model::agnes(), 42467328}, M{&model::ornith(), 15728640}}) {
    const size_t slot = m.slot_floats * 4, layer = slot / m.d->gdn_layers;
    CHECK_EQ(layer * m.d->gdn_layers, slot);
    for (uint32_t g = 1; g < m.d->gdn_layers; ++g) CHECK_EQ(runtime::pp_gdn_spec_bytes(*m.d, g), 2 * slot + g * layer);
    // At the whole model's layer count it is one card's gdn_spec (MtpDims: kSlots - 1 slots).
    CHECK_EQ(runtime::pp_gdn_spec_bytes(*m.d, m.d->gdn_layers),
             runtime::MtpDims::sizes(4096, *m.d).gdn_spec);
  }
  std::printf("verify slots: (kSlots - 2) whole-model slots + the stage's layers; Qwen3.8 at split 32: "
              "%.3f GB on device 0 (one card's: %.3f GB)\n",
              runtime::pp_gdn_spec_bytes(model::qwen38(), 24) / 1e9,
              runtime::MtpDims::sizes(4096, model::qwen38()).gdn_spec / 1e9);
}

void check_mtp_terms() {
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()})
    for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
      for (const runtime::PrefillPath& pf : paths())
        for (uint32_t dv : {0u, 32768u}) {
          const runtime::PpWeights w = weights_with_head(*d, WeightKind::Int8);
          const uint32_t s = d->layers / 2, len = 16384;
          runtime::PpExtras x;
          x.mtp = true;
          x.draft_vocab = dv;
          const runtime::PpPlan base = runtime::pp_plan(*d, s, len, w, kv, pf);
          const runtime::PpPlan p = runtime::pp_plan(*d, s, len, w, kv, pf, x);
          const runtime::PpDevicePlan &b0 = base.dev[0], &b1 = base.dev[1], &p0 = p.dev[0], &p1 = p.dev[1];
          // Device 0: its verify slots only.
          CHECK_EQ(p0.model, b0.model);
          CHECK_EQ(p0.kv, b0.kv);
          CHECK_EQ(p0.decode_state, b0.decode_state + runtime::pp_gdn_spec_bytes(*d, p0.stage.gdn));
          CHECK_EQ(p0.prefill_scratch, b0.prefill_scratch);
          CHECK_EQ(p0.draft_vocab, size_t{0});
          // Device 1: the head, the embedding replica, MtpBuffers, the head's prefill state.
          CHECK_EQ(p1.model, b1.model + w.mtp + w.embed);
          CHECK_EQ(p1.embed_replica, w.embed);
          CHECK_EQ(p1.kv, b1.kv);
          // The head's prefill rows and its two Control blocks (one per chunk in flight).
          const size_t mtp_pf =
              pf.prefill ? runtime::mtp_prefill_hidden_bytes(*d) + runtime::kPpPfDepth * sizeof(runtime::Control) : 0;
          CHECK_EQ(p1.decode_state, b1.decode_state + runtime::MtpDims::sizes(len, *d, dv, kv).total() + mtp_pf);
          const size_t slab = pf.prefill && pf.backend == runtime::PrefillBackend::L0Int8
                                  ? runtime::PrefillScratchDims::sizes(len, *d).slab
                                  : 0;
          CHECK_EQ(p1.prefill_scratch, b1.prefill_scratch + slab);
          CHECK_EQ(p1.draft_vocab, dv ? loader::draft_vocab_bytes(dv, d->hidden, true).total() : size_t{0});
        }
  CHECK(throws([] {
    runtime::PpExtras x;
    x.draft_vocab = 32768;   // without the head
    runtime::pp_plan(model::qwen38(), 32, 4096, runtime::pp_weights(model::qwen38(), WeightKind::Int8),
                     KvCache::Bf16, runtime::pp_no_prefill(), x);
  }));
  // The describe line names the head and the replica only when they are there.
  const runtime::PpWeights w = weights_with_head(model::qwen38(), WeightKind::Int8);
  runtime::PpExtras x;
  x.mtp = true;
  x.hook = true;
  const std::string on = runtime::pp_describe(
      runtime::pp_plan(model::qwen38(), 32, 16384, w, KvCache::Bf16, paths()[1], x), kTwo, kReserve);
  const std::string off = runtime::pp_describe(
      runtime::pp_plan(model::qwen38(), 32, 16384, w, KvCache::Bf16, paths()[1]), kTwo, kReserve);
  CHECK(on.find("the MTP head on device 1") != std::string::npos);
  CHECK(on.find("embedding replica 2.543 GB") != std::string::npos);
  CHECK(on.find("prefix-cache block shadows") != std::string::npos);
  CHECK(off.find("MTP") == std::string::npos && off.find("shadows") == std::string::npos);
  std::printf("%s\n", on.c_str());
}

// ---- 3 -------------------------------------------------------------------------------------
void check_shadows() {
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()})
    for (bool mtp : {false, true}) {
      const runtime::PpWeights w = weights_with_head(*d, WeightKind::Int8);
      runtime::PpExtras x;
      x.mtp = mtp;
      runtime::PpExtras xh = x;
      xh.hook = true;
      const uint32_t s = d->layers / 2 + 1, len = 32768;
      const runtime::PpPlan a = runtime::pp_plan(*d, s, len, w, KvCache::Bf16, paths()[1], x);
      const runtime::PpPlan b = runtime::pp_plan(*d, s, len, w, KvCache::Bf16, paths()[1], xh);
      for (uint32_t i = 0; i < 2; ++i) {
        const runtime::PpStage& st = a.dev[i].stage;
        const runtime::PersistentSizes ps =
            runtime::PersistentDims::stage_sizes(len, *d, KvCache::Bf16, st.gdn, st.fa);
        const size_t want = 2 * (ps.gdn_state + ps.conv_ring + (mtp && i == 1 ? size_t(d->hidden) * 2 : 0));
        CHECK_EQ(b.dev[i].shadows, want);
        CHECK_EQ(b.dev[i].prefill_scratch, a.dev[i].prefill_scratch + want);
        CHECK_EQ(b.dev[i].total(), a.dev[i].total() + want);
      }
    }
  const runtime::PpPlan q = runtime::pp_plan(model::qwen38(), 32, 16384, runtime::pp_weights(model::qwen38(), WeightKind::Int8),
                                             KvCache::Bf16, paths()[1], runtime::PpExtras{false, 0, true, true});
  std::printf("block shadows (b70-serve's prefix cache), Qwen3.8 split 32: %.3f / %.3f GB\n",
              q.dev[0].shadows / 1e9, q.dev[1].shadows / 1e9);
}

// ---- 4 -------------------------------------------------------------------------------------
void check_verify_launches() {
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes(), &model::ornith()}) {
    const size_t decode = 1 + size_t(d->layers) * (d->is_moe() ? 13 : 12) + 5;   // decode_launches
    for (uint32_t s = 1; s < d->layers; ++s) {
      if (!legal(*d, s)) continue;
      const std::array<runtime::PpStage, 2> st = runtime::pp_stages(*d, s);
      CHECK_EQ(runtime::pp_stage_verify_launches(*d, st[0]), runtime::pp_stage_launches(*d, st[0]));
      CHECK_EQ(runtime::pp_stage_verify_launches(*d, st[1]), runtime::pp_stage_launches(*d, st[1]) + 10);
      CHECK_EQ(runtime::pp_stage_verify_launches(*d, st[0]) + runtime::pp_stage_verify_launches(*d, st[1]),
               decode + 10);   // verify_launches: 784 / 880 / 536
    }
  }
  std::printf("verify lists: the stages add up to 784 / 880 / 536 launches at every split\n");
}

// ---- 5 -------------------------------------------------------------------------------------
// A synthetic KV byte: tensor t, model FA layer l (the head's layer is fa_layers), position p,
// byte i of the row (or, scale = true, of the scale row).
uint8_t kv_byte(uint32_t t, uint32_t l, uint32_t p, size_t i, bool scale) {
  uint64_t h = (uint64_t(t) * 1315423911u) ^ (uint64_t(l) << 40) ^ (uint64_t(p) << 20) ^ i ^ (scale ? 0xABCDEFull : 0);
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  return uint8_t(h >> 32);
}

// One allocation of `L` (layers `first`.. of the model's FA order), filled with kv_byte.
std::vector<uint8_t> fill(const runtime::KvLayout& L, uint32_t t, uint32_t first) {
  std::vector<uint8_t> m(L.bytes());
  for (uint32_t l = 0; l < L.layers; ++l)
    for (uint32_t p = 0; p < L.max_len; ++p) {
      for (size_t i = 0; i < L.row_bytes(); ++i)
        m[L.rows_offset(l) + size_t(p) * L.row_bytes() + i] = kv_byte(t, first + l, p, i, false);
      for (size_t i = 0; i < L.scale_row_bytes(); ++i)
        m[L.scales_offset(l) + size_t(p) * L.scale_row_bytes() + i] = kv_byte(t, first + l, p, i, true);
    }
  return m;
}

// Engine::kv_runs (engine.cc), restated: per tensor every layer's rows (the head's last),
// then at int8 every layer's scales (the head's last). Saved from one card's allocations.
std::vector<uint8_t> one_card_save(const runtime::KvLayout& L, const runtime::KvLayout* head,
                                   const std::array<std::vector<uint8_t>, 2>& kv,
                                   const std::array<std::vector<uint8_t>, 2>& hkv, uint32_t b, uint32_t e) {
  std::vector<uint8_t> out;
  const size_t n = e - b;
  const auto put = [&](const std::vector<uint8_t>& m, size_t off, size_t bytes) {
    out.insert(out.end(), m.begin() + off, m.begin() + off + bytes);
  };
  for (int t = 0; t < 2; ++t) {
    for (uint32_t l = 0; l < L.layers; ++l) put(kv[t], L.rows_offset(l) + b * L.row_bytes(), n * L.row_bytes());
    if (head) put(hkv[t], head->rows_offset(0) + b * L.row_bytes(), n * L.row_bytes());
    if (L.scale_row_bytes() == 0) continue;
    for (uint32_t l = 0; l < L.layers; ++l)
      put(kv[t], L.scales_offset(l) + b * L.scale_row_bytes(), n * L.scale_row_bytes());
    if (head) put(hkv[t], head->scales_offset(0) + b * L.scale_row_bytes(), n * L.scale_row_bytes());
  }
  return out;
}

void check_kv_runs() {
  const model::ModelDesc& d = model::qwen38();
  const uint32_t len = 64;
  size_t cases = 0;
  for (KvCache form : {KvCache::Bf16, KvCache::Int8})
    for (bool with_head : {false, true})
      for (uint32_t split : {4u, 32u, 62u}) {
        const std::array<runtime::PpStage, 2> st = runtime::pp_stages(d, split);
        const runtime::KvLayout full = runtime::kv_layout(len, d, d.fa_layers, form);
        const runtime::KvLayout hl = runtime::kv_layout(len, d, 1, form);
        const std::array<runtime::KvLayout, 2> sl{runtime::kv_layout(len, d, st[0].fa, form),
                                                  runtime::kv_layout(len, d, st[1].fa, form)};
        const std::array<std::vector<uint8_t>, 2> one{fill(full, 0, 0), fill(full, 1, 0)};
        const std::array<std::vector<uint8_t>, 2> head{fill(hl, 0, d.fa_layers), fill(hl, 1, d.fa_layers)};
        // stage[device][tensor]
        std::array<std::array<std::vector<uint8_t>, 2>, 2> stage{};
        for (uint32_t dev = 0; dev < 2; ++dev)
          for (uint32_t t = 0; t < 2; ++t) stage[dev][t] = fill(sl[dev], t, st[dev].fa_first);
        for (const auto& [b, e] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 64}, {0, 1}, {17, 49}, {63, 64}, {5, 5}}) {
          const std::vector<runtime::PpKvRun> runs = runtime::pp_kv_runs(sl, with_head ? &hl : nullptr, b, e);
          std::vector<uint8_t> two;
          for (const runtime::PpKvRun& r : runs) {
            const std::vector<uint8_t>& m = r.head ? head[r.tensor] : stage[r.device][r.tensor];
            CHECK(r.offset + r.bytes <= m.size());
            CHECK(!r.head || r.device == 1);
            two.insert(two.end(), m.begin() + r.offset, m.begin() + r.offset + r.bytes);
          }
          const std::vector<uint8_t> want = one_card_save(full, with_head ? &hl : nullptr, one, head, b, e);
          CHECK_EQ(two.size(), want.size());
          CHECK(two == want);
          // The size is the engines' kv_bytes(e - b): 2 x pos_bytes x (fa [+ 1]) x n.
          CHECK_EQ(two.size(), 2 * full.pos_bytes() * (d.fa_layers + (with_head ? 1 : 0)) * size_t(e - b));
          // ... and a load through the same runs puts every byte back where it came from.
          std::array<std::array<std::vector<uint8_t>, 2>, 2> back{};
          std::array<std::vector<uint8_t>, 2> hback{std::vector<uint8_t>(hl.bytes()), std::vector<uint8_t>(hl.bytes())};
          for (uint32_t dev = 0; dev < 2; ++dev)
            for (uint32_t t = 0; t < 2; ++t) back[dev][t].assign(sl[dev].bytes(), 0);
          size_t at = 0;
          for (const runtime::PpKvRun& r : runs) {
            std::vector<uint8_t>& m = r.head ? hback[r.tensor] : back[r.device][r.tensor];
            std::memcpy(m.data() + r.offset, want.data() + at, r.bytes);
            at += r.bytes;
          }
          for (const runtime::PpKvRun& r : runs) {
            const std::vector<uint8_t>& m = r.head ? hback[r.tensor] : back[r.device][r.tensor];
            const std::vector<uint8_t>& o = r.head ? head[r.tensor] : stage[r.device][r.tensor];
            CHECK(std::memcmp(m.data() + r.offset, o.data() + r.offset, r.bytes) == 0);
          }
          ++cases;
        }
      }
  const std::array<runtime::KvLayout, 2> sl{runtime::kv_layout(64, d, 8, KvCache::Bf16),
                                            runtime::kv_layout(64, d, 8, KvCache::Bf16)};
  CHECK(throws([&] { runtime::pp_kv_runs(sl, nullptr, 10, 9); }));
  CHECK(throws([&] { runtime::pp_kv_runs(sl, nullptr, 0, 65); }));
  const runtime::KvLayout wrong = runtime::kv_layout(64, d, 1, KvCache::Int8);
  CHECK(throws([&] { runtime::pp_kv_runs(sl, &wrong, 0, 8); }));
  std::printf("kv snapshot: %zu cases - two devices' runs == the one-card layout byte for byte "
              "(bf16 / int8, with / without the head's layer, splits 4 / 32 / 62)\n", cases);
}

// ---- 6 -------------------------------------------------------------------------------------
void check_serve_auto() {
  std::printf("b70-serve --pp 2 --max-len auto (derived; prefill l0-int8, the prefix cache's shadows, "
              "1.5 GB reserve per card):\n");
  struct Row {
    const char* name;
    const model::ModelDesc* d;
    WeightKind head;
    bool mtp;
    KvCache kv;
  };
  for (const Row r : {Row{"qwen3.8 int8 head", &model::qwen38(), WeightKind::Int8, false, KvCache::Bf16},
                      Row{"qwen3.8 int8 head, mtp", &model::qwen38(), WeightKind::Int8, true, KvCache::Bf16},
                      Row{"qwen3.8 int8 head, mtp, int8 KV", &model::qwen38(), WeightKind::Int8, true, KvCache::Int8},
                      Row{"qwen3.8 bf16 head, mtp", &model::qwen38(), WeightKind::Bf16, true, KvCache::Bf16},
                      Row{"agnes int8 head", &model::agnes(), WeightKind::Int8, false, KvCache::Bf16},
                      Row{"agnes int8 head, mtp", &model::agnes(), WeightKind::Int8, true, KvCache::Bf16},
                      Row{"ornith int8 head", &model::ornith(), WeightKind::Int8, false, KvCache::Bf16}}) {
    const runtime::PpWeights w = weights_with_head(*r.d, r.head);
    runtime::PpExtras x;
    x.mtp = r.mtp;
    x.hook = true;
    const runtime::PrefillPath pf = paths()[1];
    const uint32_t cap = kTrained;
    const runtime::PpChoice c = runtime::pp_auto_split_and_len(*r.d, w, kTwo, kReserve, cap, r.kv, pf, x);
    CHECK(c.split != 0 && c.max_len != 0);
    const runtime::PpPlan p = runtime::pp_plan(*r.d, c.split, c.max_len, w, r.kv, pf, x);
    for (uint32_t i = 0; i < 2; ++i) CHECK(p.dev[i].total() + kReserve <= kDevice);
    // One card at the same flags (its planner: no shadows - one card snapshots straight to the
    // host - and the head's buffers with its weights in model_bytes).
    const size_t one_weights = w.total() + (r.mtp ? w.mtp : 0);
    const uint32_t one = runtime::max_len_that_fits(*r.d, r.mtp, one_weights, kDevice, kReserve, cap, pf, {}, r.kv);
    CHECK(c.max_len >= one);
    std::printf("  %-34s split %2u, max_len %6u (one card %6u); device 0 %.3f GB, device 1 %.3f GB\n", r.name,
                c.split, c.max_len, one, p.dev[0].total() / 1e9, p.dev[1].total() / 1e9);
  }
}

}  // namespace

int main() {
  check_extras_off();
  check_gdn_spec();
  check_mtp_terms();
  check_shadows();
  check_verify_launches();
  check_kv_runs();
  check_serve_auto();
  std::printf("pp_serve_plan_test OK\n");
  return 0;
}

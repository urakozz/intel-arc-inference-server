// Spec 19b Task 3 (host): the DFlash drafter's buffers through the planner (`--max-len auto` must count the
// drafter's weights, ring and taps - plan 19b's Global Constraints).
//
//   1. runtime::dflash::sizes for z-lab/Qwen3.8-27B-DFlash2 beside Qwen3.8: spec 19 §2 / §5's derived numbers - the
//      ring 5 layers x 2048 positions x 20 KB = 41,943,040 B (K and V), the taps 8 x 25,600 bf16 = 409,600 B - and
//      every other buffer from its stated layout; a draft vocabulary shrinks only the head's three.
//   2. drafter_plan: weights = loader::dflash_weight_bytes; the per-position term is the RoPE rows (128 fp32 = 512 B)
//      and fixed + per-position x L = sizes(L).total() at several L.
//   3. runtime::plan / max_len_that_fits: MemoryComponents::drafter = the drafter's bytes at the plan's length; the
//      memory line gains "drafter X GB," only with a drafter (byte-identical without); auto with the drafter is the
//      largest length whose plan + reserve fits, and never above auto without it.
//   4. rope_table == the host twin's (bitwise) at positions 0..300 and 131000..131071; the lists' launch formulas
//      (87 for DFlash 2, 66 for v1, 8 for the projection).
//   5. spec 31 (plan 31 Task 4): the KV page pool with the drafter (--spec dflash, MTP off) on the 32.53 GB card -
//      every further sequence's drafter session at the per-request cap instead of a short slot's 32768; printed
//      against the fixed slots at 2 / 3 / 4 requests, bf16 and int8.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "kernels/dflash_ref.h"
#include "loader/dflash_layout.h"
#include "model/dflash.h"
#include "model/model_desc.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/memory_plan.h"

namespace {

model::DFlashDesc read_desc(const std::string& path) {
  std::ifstream f(path);
  std::stringstream s;
  s << f.rdbuf();
  return model::dflash_desc_from_config(common::json::parse(s.str()));
}

}  // namespace

int main(int argc, char** argv) {
  const model::DFlashDesc d = read_desc(argc > 1 ? argv[1] : "tests/model/dflash/qwen38_dflash2_config.json");
  namespace rd = runtime::dflash;
  const uint32_t V = 248320;

  // 1. the buffers
  const rd::DFlashSizes s = rd::sizes(d, 5120, V, 16384);
  CHECK_EQ(s.ring_k + s.ring_v, size_t(41943040));          // 5 x 2048 x (8 x 128 x 2 B) x 2
  CHECK_EQ(s.taps, size_t(409600));                          // 8 x 25600 x 2 B
  CHECK_EQ(s.rope, size_t(16384 + 8) * 128 * 4);
  CHECK_EQ(s.resid, size_t(8) * 5120 * 4);
  CHECK_EQ(s.xa, size_t(8) * 17408 * 2);
  CHECK_EQ(s.gout, size_t(8) * 34816 * 4);
  CHECK_EQ(s.co, size_t(8) * 1280 * 4);
  CHECK_EQ(s.part, size_t(32) * 16 * 8 * 130 * 4);
  CHECK_EQ(s.kv, size_t(5) * 8 * 2048 * 4);
  CHECK_EQ(s.logits, size_t(7) * V * 4);
  CHECK_EQ(s.tpart, size_t(7) * 61 * 16 * 8);
  CHECK_EQ(s.hid, size_t(8) * 256 * 4);
  CHECK_EQ(s.out, size_t(1616));
  const rd::DFlashSizes s32 = rd::sizes(d, 5120, 32768, 16384);
  const auto s4 = rd::sizes(d, 5120, V, 16384, loader::DraftFormat::Int4G64);
  CHECK_EQ(s4.gout, 8 * s.gout);
  CHECK_EQ(s4.co, 16 * s.co);
  CHECK_EQ(s4.kv, 8 * s.kv);
  CHECK_EQ(s4.hid, 16 * s.hid);
  CHECK_EQ(s4.ring_k, s.ring_k);
  CHECK_EQ(s4.ring_v, s.ring_v);
  const auto hc = rd::sizes(d, 5120, V, 16384, loader::DraftFormat::Int8, true);
  CHECK_EQ(hc.tpart, 5 * s.tpart);                  // spec 19 v3d: room for 80 candidates a row
  CHECK_EQ(hc.candidate, size_t(7) * 80 * 4);
  CHECK_EQ(hc.logits, s.logits);
  CHECK_EQ(hc.ring_k + hc.ring_v, s.ring_k + s.ring_v);
  const auto hp = rd::drafter_plan(d, 5120, V, loader::DraftFormat::Int8, true);
  CHECK_EQ(hp.weights, loader::dflash_weight_bytes(d, 5120, loader::DraftFormat::Int8).total() +
                          size_t(V / 16) * (5120 / 64) * 72 * 4);   // the 2-bit candidate head (spec 19 v3d)
  CHECK_EQ(hp.fixed + hp.per_position * 16384, hc.total());
  CHECK_EQ(rd::draft_launches(d, true), 88u);
  CHECK_EQ(s.total() - s32.total(), (s.logits - s32.logits) + (s.tpart - s32.tpart));
  std::printf("buffers at max_len 16384: %zu B (ring %zu, taps %zu, rope %zu, logits %zu); with a 32k draft "
              "vocabulary %zu B\n",
              s.total(), s.ring_k + s.ring_v, s.taps, s.rope, s.logits, s32.total());

  // 2. the planner's term
  for (loader::DraftFormat f : {loader::DraftFormat::Int8, loader::DraftFormat::Bf16, loader::DraftFormat::Int4G64}) {
    const runtime::DrafterPlan p = rd::drafter_plan(d, 5120, V, f);
    CHECK_EQ(p.weights, loader::dflash_weight_bytes(d, 5120, f).total());
    CHECK_EQ(p.per_position, size_t(512));
    for (uint32_t L : {4096u, 16384u, 131072u, 262144u})
      CHECK_EQ(p.fixed + p.per_position * L, rd::sizes(d, 5120, V, L, f).total());
    std::printf("drafter %s: weights %zu B + buffers %zu B + 512 B / position = %.3f GB at 131072\n",
                loader::draft_format_name(f), p.weights, p.fixed, p.bytes(131072) / 1e9);
  }

  // 3. the plan
  const model::ModelDesc& q = model::qwen38();
  const size_t model_bytes = size_t(17'000'000'000ull);   // a stand-in for the loaded target's bytes
  const runtime::DrafterPlan dr = rd::drafter_plan(d, 5120, V, loader::DraftFormat::Int8);
  const runtime::MemoryPlan off = runtime::plan(q, 32768, false, model_bytes);
  const runtime::MemoryPlan on =
      runtime::plan(q, 32768, false, model_bytes, {}, {}, runtime::default_kv_cache(), dr);
  CHECK_EQ(off.drafter, size_t(0));
  CHECK_EQ(on.drafter, dr.bytes(32768));
  // Spec 19c: and the engine's side - 7 more GDN slots (decode state) and, prefilling, the tap staging.
  CHECK_EQ(dr.n_taps, 5u);
  CHECK_EQ(on.drafter_slots, runtime::TapVerifyDims::gdn_spec_bytes(q));
  CHECK_EQ(on.drafter_taps, runtime::TapVerifyDims::prefill_taps_bytes(q, 5, runtime::PrefillScratchDims::kCDefault));
  // Spec 39: and one sequence's deferred GDN rows (Engine's gdn_pend: two parities at Qwen3.8's shapes), planned with
  // a drafter since the engine makes them with either proposer.
  CHECK_EQ(on.deep, size_t{103022592});
  // PERF-123: + the verify lists' head choice word (64 B, EngineSizes::head_gate)
  CHECK_EQ(on.total(), off.total() + dr.bytes(32768) + on.drafter_slots + on.drafter_taps + on.deep + 64);
  const std::string loff = runtime::format_memory("x", off, 32'000'000'000ull),
                    lon = runtime::format_memory("x", on, 32'000'000'000ull);
  CHECK(loff.find("drafter") == std::string::npos && lon.find(" drafter ") != std::string::npos);
  std::printf("%s\n", lon.c_str());
  const size_t device = 32'000'000'000ull, reserve = 1'500'000'000ull;
  const uint32_t a_off = runtime::max_len_that_fits(q, false, model_bytes, device, reserve, 262144);
  const uint32_t a_on = runtime::max_len_that_fits(q, false, model_bytes, device, reserve, 262144, {}, {},
                                                   runtime::default_kv_cache(), dr);
  CHECK(a_on <= a_off && a_on > 0);
  CHECK(runtime::plan(q, a_on, false, model_bytes, {}, {}, runtime::default_kv_cache(), dr).total() + reserve <= device);
  if (a_on + runtime::kMaxLenQuantum <= 262144)
    CHECK(runtime::plan(q, a_on + runtime::kMaxLenQuantum, false, model_bytes, {}, {}, runtime::default_kv_cache(), dr)
                  .total() + reserve > device);
  std::printf("max_len auto on 32 GB with a 17 GB target: %u without the drafter, %u with it (int8)\n", a_off, a_on);

  // Spec 19v3a: the loaded target with its int8 head (memory_plan_test's derived weights), candidate64 head,
  // unchanged drafter and eight GDN slots. Only the target's KV and its prefill widening change form.
  {
    constexpr size_t target = 16'812'371'968ull, card = 32'530'000'000ull;
    const auto bf = runtime::plan(q, 32768, false, target, {}, {}, runtime::KvCache::Bf16, hp);
    const auto i8 = runtime::plan(q, 32768, false, target, {}, {}, runtime::KvCache::Int8, hp);
    CHECK_EQ(i8.drafter, bf.drafter);
    CHECK_EQ(i8.drafter_slots, bf.drafter_slots);
    CHECK_EQ(i8.drafter_taps, bf.drafter_taps);
    const auto persistent = runtime::PersistentDims::sizes(32768, q, runtime::KvCache::Int8);
    CHECK_EQ(i8.kv, persistent.kv_k + persistent.kv_v);   // each int8 allocation includes its scales
    CHECK(i8.kv < bf.kv);
    const uint32_t len = runtime::max_len_that_fits(q, false, target, card, reserve, 262144, {}, {},
                                                  runtime::KvCache::Int8, hp);
    CHECK(len >= 200000);
    CHECK(runtime::plan(q, len, false, target, {}, {}, runtime::KvCache::Int8, hp).total() + reserve <= card);
    CHECK(len == 262144 || runtime::plan(q, len + runtime::kMaxLenQuantum, false, target, {}, {},
                                       runtime::KvCache::Int8, hp).total() + reserve > card);
    std::printf("DFlash int8 KV + candidate64 --max-len auto on 32.530 GB: %u (target %zu B, reserve %zu B)\n",
                len, target, reserve);
    // The packed int4 drafter (the default since 2026-11-14): 0.84 GB fewer drafter bytes, so as long or longer.
    const auto hp4 = rd::drafter_plan(d, 5120, V, loader::DraftFormat::Int4G64, true);
    for (runtime::KvCache kv : {runtime::KvCache::Bf16, runtime::KvCache::Int8}) {
      const uint32_t l4 = runtime::max_len_that_fits(q, false, target, card, reserve, 262144, {}, {}, kv, hp4),
                     l8 = runtime::max_len_that_fits(q, false, target, card, reserve, 262144, {}, {}, kv, hp);
      CHECK(l4 >= l8);
      std::printf("DFlash %s KV --max-len auto on 32.530 GB: packed int4 drafter %u, int8 drafter %u\n",
                  kv == runtime::KvCache::Int8 ? "int8" : "bf16", l4, l8);
    }
  }

  // Plan 19v3b Task 2: several sequences with the drafter (spec 31's pool): every further sequence adds its drafter
  // session (ring, taps, control, readback) and its GDN state; --max-len auto at 2 and 4 sequences.
  {
    constexpr size_t target = 16'812'371'968ull, card = 32'530'000'000ull;
    const size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
    const auto hp4 = rd::drafter_plan(d, 5120, V, loader::DraftFormat::Int4G64, true);
    const rd::DFlashSizes s4 = rd::sizes(d, 5120, V, 4096, loader::DraftFormat::Int4G64, true);
    CHECK_EQ(hp4.session_per_position, size_t(0));   // no full-attention layer: a session does not grow
    CHECK_EQ(hp4.session(32768), s4.ring_k + s4.ring_v + s4.taps + s4.dctl + s4.out);
    CHECK_EQ(s4.ring_k + s4.ring_v, rd::ring_snapshot_bytes(d, 4096));
    for (runtime::KvCache kv : {runtime::KvCache::Bf16, runtime::KvCache::Int8}) {
      const auto one = runtime::plan(q, 65536, false, target, {}, {}, kv, hp4);
      const runtime::PoolShape ps{3, 64, 65536};   // 131072 positions, a request up to 65536
      const auto three = runtime::plan_pool(q, ps, false, target, {}, kv, hp4);
      // Plan 40a Task 2: every sequence of several one GDN state, its committed rows pending (EngineSizes::gdn_pend).
      const size_t states = size_t(2) * runtime::PersistentDims::sizes(0, q).gdn_state;
      CHECK_EQ(three.slot_state, runtime::SlotDims::sizes(q, 3, false, runtime::pool_table_pages(ps)).total() +
                                     2 * hp4.session(65536) + hp4.rows);
      // Spec 39: the row tables (two of [32 + 8 x 4] u32 since plan 40a Task 4) and the row lists' taps, once beside
      // several slots - the allocations DFlashBuffers::add_sessions makes, unplanned before; plan 40a Task 3: the taps
      // of 32 rows; Task 4: the draft scratch grown from 8 to the batched pass's 32 rows.
      const rd::DFlashSizes w4 = rd::sizes(d, 5120, V, 4096, loader::DraftFormat::Int4G64, true, 32);
      CHECK_EQ(rd::seqs_rows(loader::DraftFormat::Int4G64, true), 32u);
      CHECK_EQ(rd::seqs_rows(loader::DraftFormat::Int8, false), 8u);
      CHECK_EQ(w4.taps, s4.taps);
      CHECK_EQ(w4.logits, s4.logits / 7 * 31);
      CHECK_EQ(hp4.rows, size_t{2} * (32 + 8 * 4) * 4 + s4.taps * 4 + (w4.total() - s4.total()));
      std::printf("the batched pass's scratch at 32 rows: %.1f MB more than the block's 8\n",
                  double(w4.total() - s4.total()) / 1e6);
      CHECK_EQ(runtime::SlotDims::sizes(q, 3, false, 0).states, states);
      CHECK_EQ(three.drafter, hp4.bytes(65536));   // the first sequence's drafter at the cap
      CHECK_EQ(three.drafter_slots, size_t(0));   // no verify slots beside several sequences
      CHECK_EQ(three.deep, runtime::EngineDims::sizes(q, kv, 0, 3, false, true).gdn_pend);   // three sequences' rows
      CHECK_EQ(one.drafter_slots, runtime::TapVerifyDims::gdn_spec_bytes(q));   // one sequence's 7 more
      for (uint32_t seqs : {2u, 4u}) {
        const runtime::PoolShape s0 = runtime::pool_that_fits(q, seqs, false, target, card, reserve, 262144, {}, kv, hp4);
        const runtime::PoolShape m0 = runtime::pool_that_fits(q, seqs, false, target, card, reserve, 262144, {}, kv);
        CHECK(s0.pages <= m0.pages);
        std::printf("DFlash --max-num-seqs %u --max-len auto on 32.530 GB, %s KV, packed int4 drafter: %u pages, one "
                    "request up to %u (without the drafter %u pages)\n", seqs,
                    kv == runtime::KvCache::Int8 ? "int8" : "bf16", s0.pages, s0.cap, m0.pages);
      }
    }
  }

  // 4. the table and the launches
  const std::vector<float> t = rd::rope_table(1e7, 128, 131072);
  const std::vector<float> tw = dfref::rope_table(1e7, 128, 131072);
  CHECK(std::memcmp(t.data(), tw.data(), 301 * 128 * 4) == 0);
  CHECK(std::memcmp(t.data() + size_t(131000) * 128, tw.data() + size_t(131000) * 128, 72 * 128 * 4) == 0);
  CHECK_EQ(rd::draft_launches(d), size_t(87));
  CHECK_EQ(rd::project_launches(d), size_t(8));
  model::DFlashDesc v1 = d;
  v1.version = 1;
  CHECK_EQ(rd::draft_launches(v1), size_t(66));

  // 5. the draft's parameters (the only way the host sets K / T / seed): K 1..7; DFlash v1 drafts greedily only (its
  //    sampled q is a full-vocabulary softmax the 16-candidate readback cannot carry) - T != 0 refused by name.
  {
    runtime::dflash::DFlashCtl c{};
    rd::set_draft_params(c, d, 7, 0.8f, 0x0123456789ABCDEFull);
    CHECK(c.k == 7 && c.temperature == 0.8f && c.seed_lo == 0x89ABCDEFu && c.seed_hi == 0x01234567u);
    bool threw = false;
    try {
      rd::set_draft_params(c, d, 8, 0.0f, 0);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
      rd::set_draft_params(c, v1, 3, 0.7f, 1);
    } catch (const std::invalid_argument& e) {
      threw = std::string(e.what()).find("v1") != std::string::npos;
    }
    CHECK(threw);
    rd::set_draft_params(c, v1, 3, 0.0f, 1);   // greedy v1 is fine
    CHECK(c.k == 3 && c.temperature == 0.0f);
  }
  // 6. spec 19d (decision 5): a snapshot carries the whole ring - sizes()' ring_k + ring_v at any max_len (every
  //    layer slides) - and the store's key names the drafter: its shapes, its format and its norm weights.
  for (uint32_t L : {4096u, 16384u, 131072u})
    CHECK_EQ(rd::ring_snapshot_bytes(d, L), rd::sizes(d, 5120, V, L).ring_k + rd::sizes(d, 5120, V, L).ring_v);
  CHECK_EQ(rd::ring_snapshot_bytes(d, 16384), size_t(41943040));
  {
    std::vector<float> norms(2 * 5120, 1.0f);
    const uint64_t k8 = rd::drafter_key(d, loader::DraftFormat::Int8, norms.data(), norms.size() * 4);
    const uint64_t k16 = rd::drafter_key(d, loader::DraftFormat::Bf16, norms.data(), norms.size() * 4);
    const uint64_t k4 = rd::drafter_key(d, loader::DraftFormat::Int4G64, norms.data(), norms.size() * 4);
    CHECK(k4 != 0 && k4 != k8 && k4 != k16);
    // the packed int4 drafter and int4 quantised at load never share a key
    const uint64_t k4p = rd::drafter_key(d, loader::DraftFormat::Int4G64, norms.data(), norms.size() * 4, true);
    CHECK(k4p != 0 && k4p != k4 && k4p != k8 && k4p != k16);
    norms[17] = 1.0078125f;   // one norm weight another checkpoint has
    const uint64_t k8b = rd::drafter_key(d, loader::DraftFormat::Int8, norms.data(), norms.size() * 4);
    model::DFlashDesc other = d;
    other.taps.back() = 60;
    const uint64_t kt = rd::drafter_key(other, loader::DraftFormat::Int8, norms.data(), norms.size() * 4);
    CHECK(k8 != 0 && k16 != 0 && k8b != 0 && kt != 0);
    CHECK(k8 != k16 && k8 != k8b && k8b != kt);
    CHECK_EQ(k8, rd::drafter_key(d, loader::DraftFormat::Int8, std::vector<float>(2 * 5120, 1.0f).data(),
                                 2 * 5120 * 4));
    std::printf("snapshot ring %zu B; the drafter's key moves with its format, a norm weight and a tap\n",
                rd::ring_snapshot_bytes(d, 16384));
  }
  // 5. spec 31: the page pool with the drafter
  {
    const size_t w = size_t{16812371968};   // Qwen3.8 int8 weights, int8 lm_head (slot_plan_test's kQwenInt8Weights)
    const size_t device = 32530000000ull, reserve = size_t(runtime::kDefaultReserveGb * 1e9);
    const runtime::DrafterPlan dp = rd::drafter_plan(d, 5120, V, loader::DraftFormat::Int8);
    std::printf("drafter session: %zu B + %zu B a position\n", dp.session_fixed, dp.session_per_position);
    std::printf("| --max-num-seqs | KV | pool (DFlash, MTP off) | one request up to | sessions at the cap GB |\n");
    for (runtime::KvCache kv : {runtime::KvCache::Bf16, runtime::KvCache::Int8})
      for (uint32_t seqs = 2; seqs <= runtime::SlotDims::kMaxSeqs; ++seqs) {
        const runtime::PoolShape ps = runtime::pool_that_fits(q, seqs, false, w, device, reserve, 262144, {}, kv, dp);
        CHECK(ps.pages != 0);
        const runtime::MemoryPlan pp = runtime::plan_pool(q, ps, false, w, {}, kv, dp);
        CHECK(pp.total() + reserve <= device);
        CHECK_EQ(pp.drafter, dp.bytes(ps.cap));   // the first sequence's drafter at the cap
        std::printf("| %u | %s | %u pages = %llu positions | %u | %.3f |\n", seqs, runtime::kv_cache_name(kv), ps.pages,
                    static_cast<unsigned long long>(ps.positions()), ps.cap, (seqs - 1) * dp.session(ps.cap) / 1e9);
      }
  }
  std::printf("dflash_plan_test: PASS\n");
  return 0;
}

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
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

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
  CHECK_EQ(s.total() - s32.total(), (s.logits - s32.logits) + (s.tpart - s32.tpart));
  std::printf("buffers at max_len 16384: %zu B (ring %zu, taps %zu, rope %zu, logits %zu); with a 32k draft "
              "vocabulary %zu B\n",
              s.total(), s.ring_k + s.ring_v, s.taps, s.rope, s.logits, s32.total());

  // 2. the planner's term
  for (loader::DraftFormat f : {loader::DraftFormat::Int8, loader::DraftFormat::Bf16}) {
    const runtime::DrafterPlan p = rd::drafter_plan(d, 5120, V, f);
    CHECK_EQ(p.weights, loader::dflash_weight_bytes(d, 5120, f).total());
    CHECK_EQ(p.per_position, size_t(512));
    for (uint32_t L : {4096u, 16384u, 131072u, 262144u})
      CHECK_EQ(p.fixed + p.per_position * L, rd::sizes(d, 5120, V, L).total());
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
  CHECK_EQ(on.drafter_taps, runtime::TapVerifyDims::prefill_taps_bytes(q, 5));
  CHECK_EQ(on.total(), off.total() + dr.bytes(32768) + on.drafter_slots + on.drafter_taps);
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
  std::printf("dflash_plan_test: PASS\n");
  return 0;
}

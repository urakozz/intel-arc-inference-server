// Spec 18b Task 3: K2's runtime sizes, launch count and memory planner (runtime/k2/
// k2_sizes.h), host only; spec 18c, the prefill chunk's too (launches, scratch, the plan
// term). The numbers are derived here from the descriptor's shapes and the kernels'
// constants, independently of the code under test; the box's k2_decode_test /
// k2_prefill_test then hold the plan to what a K2Engine actually allocates.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "check.h"
#include "loader/k2_layout.h"
#include "model/k2_horizon.h"
#include "runtime/k2/k2_sizes.h"
#include "runtime/memory_plan.h"

int main() {
  namespace k2 = runtime::k2;
  const model::K2Desc& d = model::k2();
  unsetenv("B70_K2_ATTN");   // the defaults below are flash's (spec 18 §10's switch is tested after)

  // Review Focus 5: 1 + 3 x 12 + 45 x 15 + 5 against spec 18 §2's ~1000 estimate.
  CHECK_EQ(k2::decode_launches(d), size_t(717));

  const k2::PersistentSizes ps = k2::persistent_sizes(d, 32768);
  CHECK_EQ(ps.kv_k, size_t(48) * 32768 * 8 * 128 * 2);
  CHECK_EQ(ps.kv(), size_t(32768) * 196608);   // 192 KiB per position (spec 18 §1)
  CHECK_EQ(k2::kv_layer_bytes(d, 16384), size_t(16384) * 1024 * 2);

  const k2::ScratchSizes s = k2::scratch_sizes(d);
  CHECK_EQ(k2::partials_floats(d), size_t(4) * 12288);   // the dense gate||up's S4 x N
  CHECK_EQ(s.x, size_t(6144) * 2);
  CHECK_EQ(s.attn_part, size_t(32) * 32 * 130 * 4);
  CHECK_EQ(s.router, size_t(128) * 4);
  CHECK_EQ(s.routes, size_t(48) * 2 * 32 * 4);
  CHECK_EQ(s.moe_h, size_t(9) * 768 * 2);
  CHECK_EQ(s.argmax_part, size_t(245) * 2 * 4);   // ceil(250624 / 1024) = 245 groups
  CHECK_EQ(s.total(), size_t(1818616));
  CHECK_EQ(k2::moe_route_at(3) - k2::mova_route_at(3), size_t(128));
  CHECK_EQ(k2::mova_route_at(4), size_t(8) * 128);

  // The planner at the checkpoint's weights (loader::k2_weight_bytes; the loader's total
  // without the RoPE table).
  const size_t w_bf16 = loader::k2_weight_bytes(d, false).total();
  const size_t w_int8 = loader::k2_weight_bytes(d, true).total();
  const k2::Plan p = k2::plan(d, 32768, w_bf16);
  CHECK_EQ(p.rope, size_t(32768) * 512);
  CHECK_EQ(p.model, w_bf16 + p.rope);
  CHECK_EQ(p.kv, ps.kv());
  CHECK_EQ(p.decode_state, size_t(128) + s.total());
  CHECK_EQ(p.prefill_scratch, size_t(0));
  CHECK_EQ(p.total(), size_t(28262548344ull));   // spec 18 decision 2 (A): 32k bf16 KV on one card
  CHECK_EQ(k2::plan(d, 32768, w_int8).total(), size_t(27621953400ull));
  CHECK_EQ(k2::plan(d, 32768, w_bf16, true).decode_state, p.decode_state + k2::tap_bytes(d));

  // Spec 18 §10: B70_K2_ATTN. Flash is the default and changes nothing above; eager adds two
  // launches a layer and a [M][q_heads][max_len] fp32 score row to the decode state.
  unsetenv("B70_K2_ATTN");
  CHECK(k2::k2_attn() == k2::K2Attn::Flash);
  CHECK_EQ(k2::attn_scores_bytes(d, 32768), size_t(0));
  CHECK_EQ(k2::decode_launches(d, k2::K2Attn::Eager), size_t(717 + 2 * 48));
  CHECK_EQ(k2::attn_scores_bytes(d, 32768, k2::K2Attn::Eager), size_t(32) * 32768 * 4);   // 4 MiB
  CHECK_EQ(k2::plan(d, 32768, w_bf16, false, false, k2::K2Attn::Eager).decode_state,
           p.decode_state + size_t(32) * 32768 * 4);
  setenv("B70_K2_ATTN", "eager", 1);
  CHECK(k2::k2_attn() == k2::K2Attn::Eager);
  CHECK_EQ(k2::decode_launches(d), size_t(813));   // the defaults follow the variable, as capture does
  CHECK_EQ(k2::plan(d, 32768, w_bf16).decode_state, p.decode_state + size_t(32) * 32768 * 4);
  setenv("B70_K2_ATTN", "flash", 1);
  CHECK(k2::k2_attn() == k2::K2Attn::Flash);
  setenv("B70_K2_ATTN", "", 1);
  CHECK(k2::k2_attn() == k2::K2Attn::Flash);
  setenv("B70_K2_ATTN", "fp32", 1);
  bool bad = false;
  try {
    (void)k2::k2_attn();
  } catch (const std::runtime_error& e) {
    bad = std::string(e.what()).find("expected flash or eager") != std::string::npos;
  }
  CHECK(bad);
  unsetenv("B70_K2_ATTN");
  CHECK_EQ(std::string(k2::k2_attn_name(k2::K2Attn::Eager)), std::string("eager"));

  // --max-len auto on a 32.53 GB card with the default 1.5 GB reserve: the largest quantum
  // that fits, and one more quantum does not.
  const size_t dev = 32530000000ull, res = size_t(runtime::kDefaultReserveGb * 1e9);
  const uint32_t fit_b = k2::max_len_that_fits(d, w_bf16, dev, res, 524288);
  const uint32_t fit_i = k2::max_len_that_fits(d, w_int8, dev, res, 524288);
  CHECK_EQ(fit_b, 46592u);
  CHECK_EQ(fit_i, 49920u);
  CHECK(k2::plan(d, fit_b, w_bf16).total() + res <= dev);
  CHECK(k2::plan(d, fit_b + runtime::kMaxLenQuantum, w_bf16).total() + res > dev);
  CHECK_EQ(k2::max_len_that_fits(d, w_bf16, dev, res, 16384), 16384u);   // capped
  CHECK_EQ(k2::max_len_that_fits(d, w_bf16, size_t(20e9), res, 524288), 0u);   // nothing fits
  bool threw = false;
  try {
    (void)k2::max_len_that_fits(d, w_bf16, dev, res, 100);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  const std::string line = k2::describe(p, dev, res);
  CHECK(line.find("plan at max_len 32768") != std::string::npos);
  std::printf("%s\n", line.c_str());

  // ---- spec 18c: the prefill chunk ------------------------------------------------------
  // 1 + 3 x (2 + 20 + 2 + 6 + 2 + 24 + 6) + 45 x (2 + 20 + 6 + 2 + 6 + 2 + 11) = 2392.
  CHECK_EQ(k2::pf_batches(d, k2::PfGroup::Value), 1u);
  CHECK_EQ(k2::pf_batches(d, k2::PfGroup::GateUp), 2u);
  CHECK_EQ(k2::pf_batches(d, k2::PfGroup::Down), 1u);
  CHECK_EQ(k2::prefill_chunk_launches(d), size_t(2392));
  CHECK_EQ(k2::pf_ld_max(d), 10240u);
  const k2::PrefillSizes pf = k2::prefill_sizes(d);
  CHECK_EQ(pf.partials, size_t(2048) * 10240 * 4);
  CHECK_EQ(pf.slab, size_t(6144) * 1024 * 2);          // the dense down's K
  CHECK_EQ(pf.routes, size_t(48) * 2 * 2048 * 32 * 4);
  CHECK_EQ(pf.hdr, size_t(112) * 4);                    // 4 + 100 + 1 words, padded to 16
  CHECK_EQ(pf.xg, size_t(672) * 32 * 2560 * 2);         // tmax(2048) = 672 MoE tiles
  CHECK_EQ(pf.h, size_t(672) * 32 * 768 * 2);
  CHECK_EQ(pf.w, size_t(51) * 2560 * 1536 * 2);         // half the gate||up blocks, the largest
  CHECK_EQ(pf.total(), size_t(797247168));              // derived by hand, k2_sizes.h's table
  const k2::Plan pp = k2::plan(d, 32768, w_bf16, false, true);
  CHECK_EQ(pp.prefill_scratch, pf.total());
  CHECK_EQ(pp.total(), p.total() + pf.total());
  const uint32_t fit_pf = k2::max_len_that_fits(d, w_bf16, dev, res, 524288, true);
  CHECK_EQ(fit_pf, 42752u);   // 3840 positions fewer than decode-only: 0.797 GB / 196,608 B (+ RoPE)
  CHECK(k2::plan(d, fit_pf, w_bf16, false, true).total() + res <= dev);
  CHECK(k2::plan(d, fit_pf + runtime::kMaxLenQuantum, w_bf16, false, true).total() + res > dev);
  CHECK(k2::describe(pp, dev, res).find("prefill scratch planned") != std::string::npos);
  std::printf("prefill: %zu launches per chunk + %zu, scratch %.3f GB; auto with prefill -> %u (bf16 head)\n",
              k2::prefill_chunk_launches(d), k2::kPrefillHeadLaunches, pf.total() / 1e9, fit_pf);
  std::printf("k2_plan_test OK: 717 launches (813 eager); auto -> %u (bf16 head) / %u (int8 head) on "
              "a %.2f GB card with a %.1f GB reserve\n", fit_b, fit_i, dev / 1e9, res / 1e9);
  return 0;
}

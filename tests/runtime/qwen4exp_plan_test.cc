// Spec 21b Task 5: Qwen3.8-Flash-Next's memory plan as device-free arithmetic (runtime/qwen4exp/
// qwen4exp_sizes.h). Host only. Every number is DERIVED from the descriptor and the loader's formulas
// (loader/qwen4exp_layout.h); the test asserts the formulas and prints the values (plan 21b Task 5 Step 1):
//
//   1. the persistent sizes: KV + indexer keys a position, the GDN state of 36 layers, a 262144-position plan,
//      the PLE state on the PLE layer's device only, empty groups one line;
//   2. layers_that_fit at 32768 / 131072 with the int8 head, Intel's forms and ours, one and two 32.53 GB cards
//      and a 1.5 GB reserve: N fits and N + 1 does not (under any split on two cards);
//   3. require_fits: the full model refused on two cards naming spec 22 (both forms), --layers 38 refused on
//      one card naming its bytes, the synthetic's N = 4 planned on one card;
//   4. pp_split is runtime::pp_balance over pp_layer_bytes (16b's rule, no second one);
//   5. max_len_that_fits, describe's text, the host PLE table's bytes.
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "loader/qwen4exp_layout.h"
#include "model/qwen4exp.h"
#include "runtime/control.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {

namespace rq = runtime::qwen4exp;
using model::Q4Form;
using model::Q4Placement;
using model::Qwen4ExpDesc;

constexpr size_t kCard = 32530000000ull;   // memory_line()'s device total (spec 6 §8.4)
constexpr size_t kReserve = 1500000000ull;  // runtime::kDefaultReserveGb

template <class F>
void throws_naming(const std::string& what, F fn) {
  bool ok = false;
  try {
    fn();
  } catch (const std::exception& e) {
    ok = std::string(e.what()).find(what) != std::string::npos;
    if (!ok) std::fprintf(stderr, "threw, but without '%s': %s\n", what.c_str(), e.what());
  }
  if (!ok) std::fprintf(stderr, "expected a refusal naming '%s'\n", what.c_str());
  CHECK(ok);
}

Qwen4ExpDesc intel_desc() {
  Qwen4ExpDesc d = model::qwen4exp();
  d.forms = {Q4Form::Bf16, Q4Form::Bf16, Q4Form::Bf16, 128};
  return d;
}

// Whether N layers fit `devices` cards under ANY placement (the brute force layers_that_fit must agree with).
bool any_fit(const Qwen4ExpDesc& d, uint32_t n, uint32_t devices, uint32_t max_len, bool mtp) {
  const Qwen4ExpDesc t = rq::truncated(d, n);
  const std::array<size_t, runtime::kPpDevices> caps = {kCard, kCard};
  if (devices == 1) return rq::fits(rq::plan(t, Q4Placement::one(t), max_len, true, mtp), caps, kReserve);
  for (uint32_t s = 1; s < n; ++s)
    if (rq::fits(rq::plan(t, Q4Placement::two(t, s), max_len, true, mtp), caps, kReserve)) return true;
  return false;
}

}  // namespace

int main() {
  const Qwen4ExpDesc& ours = model::qwen4exp();
  const Qwen4ExpDesc intel = intel_desc();
  const std::array<size_t, runtime::kPpDevices> caps = {kCard, kCard};

  // --- 1. the persistent sizes ---------------------------------------------------------------------------
  CHECK_EQ(rq::kv_bytes_per_pos(ours), size_t(12) * (2 * 512 * 2 + 128 * 2 / 4));
  CHECK_EQ(rq::kv_bytes_per_pos(ours), size_t(25344));
  CHECK_EQ(rq::gdn_state_bytes_per_layer(ours), size_t(48) * 128 * 128 * 4);
  CHECK_EQ(rq::conv_ring_bytes_per_layer(ours), size_t(16) * 10240 * 2);
  CHECK_EQ(rq::ple_state_bytes(ours), size_t(8) + size_t(9) * 10240 * 2);
  {
    const uint32_t L = 262144;
    const rq::PersistentSizes ps = rq::persistent_sizes(ours, Q4Placement::one(ours), 0, L);
    CHECK_EQ(ps.gdn_state, size_t(36) * rq::gdn_state_bytes_per_layer(ours));   // 113.2 MB
    CHECK_EQ(ps.kv, size_t(12) * L * 2048);
    CHECK_EQ(ps.idx_keys, size_t(12) * (L / 4) * 256);
    CHECK_EQ(ps.kv + ps.idx_keys, rq::kv_bytes_per_pos(ours) * L);           // ~6.6 GB
    CHECK_EQ(ps.ple, rq::ple_state_bytes(ours));
    CHECK_EQ(ps.control, sizeof(runtime::Control));
    // two cards split at 2: device 0 holds layers 0, 1 (the PLE layer, no QSA), device 1 the rest
    const Q4Placement two = Q4Placement::two(ours, 2);
    const rq::PersistentSizes p0 = rq::persistent_sizes(ours, two, 0, L), p1 = rq::persistent_sizes(ours, two, 1, L);
    CHECK(p0.kv == 64 && p0.idx_keys == 64 && p0.ple == rq::ple_state_bytes(ours) && p1.ple == 64);
    CHECK_EQ(p0.gdn_state + p1.gdn_state, ps.gdn_state);
    CHECK_EQ(p1.kv, ps.kv);
    std::printf("persistent at 262144 (one card): KV + indexer keys %zu B (%.3f GB), GDN state %zu B (%.1f MB), "
                "conv rings %zu B, PLE state %zu B\n", ps.kv_total(), ps.kv_total() / 1e9, ps.gdn_state,
                ps.gdn_state / 1e6, ps.conv_ring, ps.ple);
  }

  // --- 2. N per card ---------------------------------------------------------------------------------------
  uint32_t n_intel_1 = 0, n_intel_2 = 0;
  for (const Qwen4ExpDesc* d : {&intel, &ours})
    for (uint32_t L : {32768u, 131072u})
      for (uint32_t dev : {1u, 2u}) {
        const uint32_t n = rq::layers_that_fit(*d, dev, L, true, false, kCard, kReserve);
        CHECK(n >= 4 && n < 48);
        CHECK(any_fit(*d, n, dev, L, false));
        CHECK(!any_fit(*d, n + 1, dev, L, false));
        const Qwen4ExpDesc t = rq::truncated(*d, n);
        const Q4Placement pl = rq::placement_for(t, dev, L, true, false, kCard, kReserve);
        const std::vector<rq::DevicePlan> p = rq::plan(t, pl, L, true, false);
        CHECK(rq::fits(p, caps, kReserve));
        rq::require_fits(p, caps, kReserve);
        const uint32_t n_mtp = rq::layers_that_fit(*d, dev, L, true, true, kCard, kReserve);
        CHECK(n_mtp <= n && any_fit(*d, n_mtp, dev, L, true) && !any_fit(*d, n_mtp + 1, dev, L, true));
        std::printf("layers_that_fit (%s forms, int8 head, max_len %u, %u card(s) of %.2f GB, reserve %.1f GB): N = %u "
                    "(%u with the MTP head)\n  %s\n",
                    d == &intel ? "Intel's" : "ours", L, dev, kCard / 1e9, kReserve / 1e9, n, n_mtp,
                    rq::describe(p, pl, L, caps, kReserve).c_str());
        if (d == &intel && L == 32768) (dev == 1 ? n_intel_1 : n_intel_2) = n;
      }

  // --- 3. the refusals --------------------------------------------------------------------------------------
  for (const Qwen4ExpDesc* d : {&intel, &ours}) {
    const std::vector<rq::DevicePlan> full = rq::plan(*d, Q4Placement::two(*d, rq::pp_split(*d, 32768, true, false).split),
                                                      32768, true, false);
    CHECK(full[0].whole && !rq::fits(full, caps, kReserve));
    throws_naming("spec 22", [&] { rq::require_fits(full, caps, kReserve); });
    throws_naming("expert-offload tier", [&] { rq::require_fits(full, caps, kReserve); });
    throws_naming(std::to_string(kCard) + " B", [&] { rq::require_fits(full, caps, kReserve); });
  }
  {
    const Qwen4ExpDesc t38 = rq::truncated(intel, 38);
    const std::vector<rq::DevicePlan> p = rq::plan(t38, Q4Placement::one(t38), 32768, true, false);
    CHECK(!p[0].whole);
    throws_naming("layers [0, 38)", [&] { rq::require_fits(p, caps, kReserve); });
    throws_naming(std::to_string(p[0].total()) + " B", [&] { rq::require_fits(p, caps, kReserve); });
    const Qwen4ExpDesc t4 = rq::truncated(ours, 4);   // the synthetic checkpoints
    for (const Qwen4ExpDesc* d : {&t4}) {
      const std::vector<rq::DevicePlan> p4 = rq::plan(*d, Q4Placement::one(*d), 32768, true, true);
      rq::require_fits(p4, caps, kReserve);
      CHECK_EQ(rq::max_len_that_fits(*d, Q4Placement::one(*d), true, true, caps, kReserve), 262144u);
      std::printf("the synthetic (4 layers, ours, MTP): %s\n", rq::describe(p4, Q4Placement::one(*d), 32768, caps, kReserve).c_str());
    }
    throws_naming("--layers 1", [&] { rq::truncated(ours, 1); });
    throws_naming("--layers 49", [&] { rq::truncated(ours, 49); });
  }

  // --- 4. the split by bytes is 16b's ---------------------------------------------------------------------------
  {
    const uint32_t L = 32768;
    const Qwen4ExpDesc t = rq::truncated(intel, n_intel_2);
    const runtime::PpBalance b = rq::pp_split(t, L, true, false);
    const std::vector<size_t> lb = rq::pp_layer_bytes(t, L);
    CHECK_EQ(lb.size(), size_t(t.layers));
    CHECK_EQ(lb[0], loader::q4_layer_bytes(t, 0).total() + rq::gdn_state_bytes_per_layer(t) + rq::conv_ring_bytes_per_layer(t));
    CHECK_EQ(lb[1], loader::q4_layer_bytes(t, 1).total() + rq::gdn_state_bytes_per_layer(t) + rq::conv_ring_bytes_per_layer(t) +
                        rq::ple_state_bytes(t));
    CHECK_EQ(lb[3], loader::q4_layer_bytes(t, 3).total() + size_t(L) * 2048 + size_t(L / 4) * 256 + 4 * 256);
    const size_t ctl = sizeof(runtime::Control), rope = t.rope_table_bytes(L);
    const runtime::PpBalance direct = runtime::pp_balance(
        lb, loader::q4_embed_bytes(t) + rope + ctl, loader::q4_final_mixer_bytes(t) + loader::q4_lm_head_bytes(t, true) + rope + ctl);
    CHECK(direct.split == b.split && direct.dev0 == b.dev0 && direct.dev1 == b.dev1);
    // the plan's totals add up the same terms
    const std::vector<rq::DevicePlan> p = rq::plan(t, Q4Placement::two(t, b.split), L, true, false);
    CHECK_EQ(p[0].weights + p[1].weights, loader::q4_device_weight_bytes(t, Q4Placement::one(t), 0, true, false));
    std::printf("pp_split (Intel's forms, %u layers, max_len %u): split %u, device 0 %zu B, device 1 %zu B (16b's "
                "pp_balance)\n", t.layers, L, b.split, b.dev0, b.dev1);
  }

  // --- 5. max_len_that_fits, describe, the host table --------------------------------------------------------
  {
    const Qwen4ExpDesc t = rq::truncated(intel, n_intel_1);
    const uint32_t len = rq::max_len_that_fits(t, Q4Placement::one(t), true, false, caps, kReserve);
    CHECK(len >= 32768 && len % runtime::kMaxLenQuantum == 0);
    CHECK(rq::fits(rq::plan(t, Q4Placement::one(t), len, true, false), caps, kReserve));
    if (len < 262144) CHECK(!rq::fits(rq::plan(t, Q4Placement::one(t), len + runtime::kMaxLenQuantum, true, false), caps, kReserve));
    const std::string s = rq::describe(rq::plan(t, Q4Placement::one(t), len, true, false), Q4Placement::one(t), len, caps, kReserve);
    CHECK(s.find("plan at max_len " + std::to_string(len) + " (one card, layers [0, " + std::to_string(n_intel_1) + "))") == 0);
    CHECK(s.find("device 0") != std::string::npos && s.find("weights") != std::string::npos);
    std::printf("max_len_that_fits (Intel's forms, %u layers, one card): %u\n", n_intel_1, len);
  }
  const size_t b16 = rq::host_ple_bytes(ours, loader::Q4PleScale::Bf16), f32 = rq::host_ple_bytes(ours, loader::Q4PleScale::F32);
  CHECK_EQ(b16, size_t(320001446) * (160 + 2));
  CHECK_EQ(f32, size_t(320001446) * (160 + 4));
  std::printf("qwen4exp_plan_test OK: N at 32768 with Intel's forms = %u on one card, %u on two; the host PLE table "
              "%zu B (%.2f GB, bf16 scales) / %zu B (%.2f GB, f32); the full model refused naming spec 22\n",
              n_intel_1, n_intel_2, b16, b16 / 1e9, f32, f32 / 1e9);
  return 0;
}

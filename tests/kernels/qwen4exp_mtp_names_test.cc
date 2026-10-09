// Spec 21e Task 3 Step 1 (host): the binaries Qwen3.8-Flash-Next's MTP lists bind - kernels::qwen4exp's spec 21e
// block (verify_variants at M = 1..4, draft_variants, mtp_prefill_variants: the names runtime/qwen4exp/
// qwen4exp_capture.cc and qwen4exp_prefill.cc form) for both dense arms (ours int4 / Intel's bf16), both attention
// forms, both PLE scale forms, both head forms and both pre_fc_norm_hidden forms (decision 4) - against what the build
// makes: tests/CMakeLists.txt's B70_Q4EXP_MTP_KERNELS (src/kernels/CMakeLists.txt's spec 21e block, `kernel_<name>`)
// then, after `--reused`, the names other blocks build that the lists bind by name (21c's decode list, 21d's prefill
// binaries, Qwen3.8's argmax_stage1_M<2..4>). Every bound name is built; every 21e-block binary is bound (nothing
// dead). Then the launch counts the capture asserts (runtime::qwen4exp::verify_device_launches / draft_launches /
// mtp_prefill_launches) at 48 layers and the synthetic's 4, one card and two:
//   verify at M: the decode list at M (779 at 48 layers) + 2 on the PLE layer at M > 1 (q4_pf_ple's three for
//     q4_ple_block) + the head's KV pass 10 + M on the last device;
//   a draft step: 27 (flash) / 26 (eager), + 2 when it selects (step 0; every step under B70_Q4_MTP_SELECT=fresh);
//   the prefill's head pass a chunk: 1 + 26 (just the _Y_NN when the chunk has no head row: pos 0, one id).
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/qwen4exp_kernels.h"
#include "model/qwen4exp.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

int main(int argc, char** argv) {
  namespace kq = kernels::qwen4exp;
  namespace rq = runtime::qwen4exp;
  std::set<std::string> block, reused;
  bool after = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reused") {
      after = true;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (after ? reused : block).insert(a.substr(7));
  }
  CHECK(!block.empty() && !reused.empty());
  CHECK_EQ(kq::kMtpVerifyRows, rq::kVerifyRows);
  CHECK_EQ(rq::kVerifyRows, rq::kMaxDraft + 1);
  CHECK_EQ(rq::kGdnSlots, 4u);
  CHECK_EQ(kq::mtp_variant(1, true), std::string("q4_mtp_M1_SINGLE"));
  CHECK_EQ(kq::mtp_variant(4, false), std::string("q4_mtp_M4_STREAM"));
  CHECK_EQ(kq::mtp_variant(2048, true, true), std::string("q4_mtp_M2048_SINGLE_PF"));
  CHECK_EQ(kq::gdn_slots_variant(3), std::string("gdn_step_slots_M3_G1"));
  CHECK_EQ(kq::fc_variant(4), std::string("gemv_bf16_M4_K2560_N2560"));
  CHECK_EQ(kq::pf_ple_verify_variant(), std::string("q4_pf_ple_C4"));
  CHECK_EQ(kq::qkvg_m_variant(3, true), std::string("gemv_M3_K2560_N13312_S2_L0"));
  CHECK_EQ(kq::lm_head_m_variant(2, true), std::string("gemv_i8w_M2_K2560_N248320"));

  std::set<std::string> bound;
  model::Qwen4ExpDesc intel = model::qwen4exp();
  intel.forms = {model::Q4Form::Bf16, model::Q4Form::Bf16, model::Q4Form::Bf16, 128};
  for (const model::Qwen4ExpDesc* f : {&model::qwen4exp(), static_cast<const model::Qwen4ExpDesc*>(&intel)})
    for (bool eager : {false, true})
      for (bool ple_bf16 : {false, true})
        for (bool int8 : {false, true})
          for (bool single : {false, true}) {
            for (unsigned M = 1; M <= kq::kMtpVerifyRows; ++M)
              for (const std::string& v : kq::verify_variants(*f, M, int8, eager, ple_bf16, single)) bound.insert(v);
            for (const std::string& v : kq::draft_variants(int8, eager, single)) bound.insert(v);
            for (const std::string& v : kq::mtp_prefill_variants(single)) bound.insert(v);
          }
  for (const std::string& v : bound)
    if (!block.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the Qwen3.8-Flash-Next MTP lists bind %s, which no CMake block builds (B70_Q4EXP_MTP_KERNELS)\n",
                   v.c_str());
      return 1;
    }
  for (const std::string& b : block)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built by the spec 21e block but no MTP list binds it\n", b.c_str());
      return 1;
    }
  std::printf("names: %zu bound, %zu built by the spec 21e block, %zu reused by name\n", bound.size(), block.size(),
              reused.size());

  // The launch counts (derived).
  const model::Qwen4ExpDesc& full = model::qwen4exp();
  const model::Q4Placement one = model::Q4Placement::one(full), two = model::Q4Placement::two(full, 24);
  const rq::Q4Attn fl = rq::Q4Attn::Flash, eg = rq::Q4Attn::Eager;
  const runtime::PpHandoff cp = runtime::PpHandoff::Copy;
  CHECK_EQ(rq::decode_launches(full, one, fl, cp), size_t(779));
  for (uint32_t M = 1; M <= rq::kVerifyRows; ++M) {
    const size_t want = 779 + (M > 1 ? 2 : 0) + 10 + M;
    CHECK_EQ(rq::verify_launches(full, one, M, fl, cp), want);
    CHECK_EQ(rq::verify_launches(full, one, M, eg, cp), want - 12);   // eager: one launch fewer a QSA layer
    // two cards: device 0 holds the PLE layer (+2 at M > 1) and ends with _Y_NN; device 1 the head pass
    CHECK_EQ(rq::verify_device_launches(full, two, 0, M, fl, cp), rq::device_launches(full, two, 0, fl, cp) + (M > 1 ? 2 : 0));
    CHECK_EQ(rq::verify_device_launches(full, two, 1, M, fl, cp), rq::device_launches(full, two, 1, fl, cp) + 10 + M);
    std::printf("verify at M = %u: %zu launches at 48 layers on one card (%zu eager), %zu + %zu on two\n", M,
                rq::verify_launches(full, one, M, fl, cp), rq::verify_launches(full, one, M, eg, cp),
                rq::verify_device_launches(full, two, 0, M, fl, cp), rq::verify_device_launches(full, two, 1, M, fl, cp));
  }
  const model::Qwen4ExpDesc d4 = rq::truncated(full, 4);
  CHECK_EQ(rq::decode_launches(d4, model::Q4Placement::one(d4), fl, cp), size_t(75));
  CHECK_EQ(rq::verify_launches(d4, model::Q4Placement::one(d4), 1, fl, cp), size_t(75 + 11));
  CHECK_EQ(rq::verify_launches(d4, model::Q4Placement::one(d4), 4, fl, cp), size_t(75 + 2 + 14));
  CHECK_EQ(rq::draft_launches(fl, true), size_t(29));
  CHECK_EQ(rq::draft_launches(fl, false), size_t(27));
  CHECK_EQ(rq::draft_launches(eg, true), size_t(28));
  CHECK_EQ(rq::draft_launches(eg, false), size_t(26));
  CHECK_EQ(rq::mtp_prefill_launches(0, 1), size_t(1));
  CHECK_EQ(rq::mtp_prefill_launches(0, 2048), size_t(27));
  CHECK_EQ(rq::mtp_prefill_launches(2048, 2048), size_t(27));
  CHECK_EQ(rq::mtp_prefill_rows(0, 2048), 2047u);
  CHECK_EQ(rq::mtp_prefill_rows(2048, 7), 7u);
  CHECK_EQ(rq::prefill_chunk_launches(d4, model::Q4Placement::one(d4), 0, 2048, false, true),
           rq::prefill_chunk_launches(d4, model::Q4Placement::one(d4), 0, 2048) + 27);
  std::printf("--layers 4: decode 75, verify at M = 1 / 4: %zu / %zu; a draft step 29 / 27 (step 0 / later, flash); "
              "the prefill's head pass 27 a chunk\n", rq::verify_launches(d4, model::Q4Placement::one(d4), 1, fl, cp),
              rq::verify_launches(d4, model::Q4Placement::one(d4), 4, fl, cp));
  // an iteration at K drafts: K draft steps + the verify at M = K + 1 (K = 3 at 48 layers, one card, flash)
  const size_t it3 = rq::draft_launches(fl, true) + 2 * rq::draft_launches(fl, false) + rq::verify_launches(full, one, 4, fl, cp);
  std::printf("an iteration at K = 3 (48 layers, one card, flash): %zu launches for up to 4 tokens (plain: 779 a token)\n", it3);
  std::puts("qwen4exp_mtp_names_test OK");
  return 0;
}

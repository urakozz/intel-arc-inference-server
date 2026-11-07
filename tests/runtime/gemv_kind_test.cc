// gemv_kind_test - spec 28b's capture rule (runtime/gemv_kind.h, kernels::kSmallMTuning). Host only.
//
//   - M = 1 -> Gemv for every list kind: plain decode, ingest / generate, the MTP draft, the prefill tail's head,
//     the draft vocabulary's head all bind today's gemv / gemv_i8w names, on every model's shapes;
//   - M >= 2 -> SmallM: every tuned shape of Qwen3.8 (its int4 linears and the int8 head) binds
//     gemm_sm at M = 2..8, and the shape's S is the one its consumer folds (the S table is one row per shape -
//     the same at every M, property R's precondition); Ornith's dense projections on gemm_sm at M 2..4; an untuned
//     shape keeps gemv;
//   - the override (CaptureOptions::force_linear) forces either kernel at any M;
//   - spec 28d: the DFlash drafter's int8 linears and draft heads on gemm_sm_i8w (gemv_i8w under the override);
//     Flash-Next's dense projections and int8 head at M 2..4.
#include <cstdio>
#include <string>

#include "check.h"
#include "kernels/dflash_kernels.h"
#include "kernels/kernels.h"
#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/gemv_kind.h"

using kernels::LinearKind;
using model::LinearId;

int main() {
  // The rule.
  static_assert(runtime::gemv_kind(1) == LinearKind::Gemv, "M = 1 keeps gemv");
  static_assert(runtime::gemv_kind(2) == LinearKind::SmallM, "M >= 2 binds gemm_sm");
  for (uint32_t M = 2; M <= 16; ++M) CHECK(runtime::gemv_kind(M) == LinearKind::SmallM);
  // The override.
  const runtime::CaptureOptions aut{}, gv{runtime::LinearForce::Gemv}, sm{runtime::LinearForce::SmallM};
  for (uint32_t M = 1; M <= 16; ++M) {
    CHECK(runtime::linear_kind(M, aut) == runtime::gemv_kind(M));
    CHECK(runtime::linear_kind(M, gv) == LinearKind::Gemv);
    CHECK(runtime::linear_kind(M, sm) == LinearKind::SmallM);
  }

  size_t tuned = 0;
  for (const model::ModelDesc* d : {&model::qwen38()}) {
    const LinearId int4[] = {LinearId::QkvZ, LinearId::OutProj, LinearId::Qkv, LinearId::OProj, LinearId::GateUp,
                             LinearId::Down};
    for (LinearId id : int4) {
      const model::GemvShape& s = d->shape(id);
      // M = 1: today's binary, whatever the list
      CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(1), 1, s.K, s.N, s.S, s.layout),
               kernels::gemv_variant(1, s.K, s.N, s.S, s.layout));
      const kernels::SmallMTuning* t = kernels::small_m_tuning(s.K, s.N, s.S, s.layout);
      CHECK(t != nullptr);   // every Qwen3.8 int4 decode linear is tuned
      // the consumer's fold is unchanged - except gate||up's one-slice 2x8 work-group, whose SiLU is the _S1 build
      if (id == LinearId::GateUp && kernels::gemm_sm_out_s(*t) != s.S) {
        CHECK_EQ(kernels::gemm_sm_out_s(*t), 1u);
        for (uint32_t M = 2; M <= 8; ++M)
          CHECK_EQ(kernels::gemm_sm_silu_variant(M, d->intermediate, 1, s.S),
                   kernels::prep_silu_mul_s1_variant(M, d->intermediate));
      } else {
        CHECK_EQ(kernels::gemm_sm_out_s(*t), s.S);
        CHECK_EQ(kernels::gemm_sm_silu_variant(2, d->intermediate, s.S, s.S),
                 kernels::prep_silu_mul_variant(2, d->intermediate));
      }
      ++tuned;
      for (uint32_t M = 2; M <= 8; ++M)
        CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(M), M, s.K, s.N, s.S, s.layout),
                 kernels::gemm_sm_variant(M, *t));
    }
    // the int8 head (spec 9): gemv_i8w at M = 1, gemm_sm_i8w at M >= 2
    CHECK_EQ(kernels::int8_linear_variant(runtime::gemv_kind(1), 1, d->hidden, model::Qwen35::kVocab),
             kernels::gemv_i8w_variant(1, d->hidden, model::Qwen35::kVocab));
    for (uint32_t M = 2; M <= 8; ++M)
      CHECK(kernels::int8_linear_variant(runtime::gemv_kind(M), M, d->hidden, model::Qwen35::kVocab)
                .rfind("gemm_sm_i8w_M", 0) == 0);
  }
  // The S table is one row per shape: the same row (so the same S, row block, work-group) at every M.
  for (const kernels::SmallMTuning& t : kernels::kSmallMTuning)
    for (unsigned M : kernels::gemm_sm_built_m(t)) {
      const std::string v = kernels::gemm_sm_variant(M, t);
      CHECK(v.find("_M" + std::to_string(M) + "_") != std::string::npos);
      CHECK_EQ(kernels::small_m_tuning(t.K, t.N, t.S, t.layout), &t);
    }
  // Ornith (the operator's 28b addition): its dense int4 projections and int8 head on gemm_sm at its verify
  // lists' M = 2..4, gemv at M = 1; the consumer's S unchanged.
  const model::ModelDesc& o = model::ornith();
  for (LinearId id : {LinearId::Qkv, LinearId::OProj, LinearId::QkvZ, LinearId::OutProj, LinearId::AB}) {
    const model::GemvShape& s = id == LinearId::AB ? o.ab(model::WeightKind::Int4).shape : o.shape(id);
    const kernels::SmallMTuning* t = kernels::small_m_tuning(s.K, s.N, s.S, s.layout);
    CHECK(t != nullptr && t->mset == kernels::kSmVerify4 && kernels::gemm_sm_out_s(*t) == s.S);
    CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(1), 1, s.K, s.N, s.S, s.layout),
             kernels::gemv_variant(1, s.K, s.N, s.S, s.layout));
    for (uint32_t M = 2; M <= 4; ++M)
      CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(M), M, s.K, s.N, s.S, s.layout),
               kernels::gemm_sm_variant(M, *t));
  }
  CHECK(kernels::int8_linear_variant(runtime::gemv_kind(2), 2, o.hidden, model::Qwen35::kVocab)
            .rfind("gemm_sm_i8w_M2_K2048", 0) == 0);
  // An untuned shape keeps gemv at every M (a made-up one).
  CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(4), 4, 3072, 1024, 2, 0), kernels::gemv_variant(4, 3072, 1024, 2, 0));
  // The draft vocabulary's head (M = 1) and the MTP draft (M = 1): gemv_i8w.
  for (unsigned nv : {32768u, 65536u, 131072u})
    CHECK_EQ(kernels::int8_linear_variant(runtime::gemv_kind(1), 1, 5120, nv), kernels::gemv_i8w_variant(1, 5120, nv));
  // Spec 28d: the DFlash drafter's int8 linears (M 8) and the int8 head at its 7 draft rows (full and every draft
  // vocabulary) on gemm_sm_i8w; under the override (Gemv) every one is gemv_i8w as before; the bf16 forms gemv_bf16.
  size_t drafter = 0;
  for (bool v2 : {true, false})
    for (unsigned dv : {0u, 32768u, 65536u, 131072u}) {
      for (const std::string& v : kernels::dflash::draft_variants(v2, true, true, dv)) {
        CHECK(v.rfind("gemv_i8w", 0) != 0);
        drafter += v.rfind("gemm_sm_i8w_M", 0) == 0;
      }
      for (const std::string& v : kernels::dflash::draft_variants(v2, true, true, dv, LinearKind::Gemv))
        CHECK(v.rfind("gemm_sm", 0) != 0);
      for (const std::string& v : kernels::dflash::draft_variants(v2, false, false, dv))
        CHECK(v.rfind("gemm_sm", 0) != 0 && v.rfind("gemv_i8w", 0) != 0);
    }
  CHECK_EQ(drafter, size_t(4 * (9 + 7)));   // v2: 8 linears + the head; v1: 6 + the head
  // Spec 28d: Qwen3.8-Flash-Next's dense int4 projections and its int8 head at M 2..4 (the verify lists).
  {
    const unsigned H = 2560;
    const struct { unsigned K, N, S; } q4[] = {{H, 16384, 1}, {H, 13312, 2}, {6144, H, 4}};
    for (const auto& r : q4) {
      const kernels::SmallMTuning* t = kernels::small_m_tuning(r.K, r.N, r.S, 0);
      CHECK(t != nullptr && t->mset == kernels::kSmVerify4 && kernels::gemm_sm_out_s(*t) == r.S);
      for (uint32_t M = 2; M <= 4; ++M)
        CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(M), M, r.K, r.N, r.S, 0), kernels::gemm_sm_variant(M, *t));
    }
    CHECK_EQ(kernels::int8_linear_variant(runtime::gemv_kind(4), 4, H, model::Qwen35::kVocab),
             std::string("gemm_sm_i8w_M4_K2560_N248320_wg"));
  }
  std::printf("gemv_kind_test OK: %zu tuned int4 decode linears (Qwen3.8), M = 1 on gemv, M >= 2 on "
              "gemm_sm, Ornith and Flash-Next at M 2..4, the override, the drafter on gemm_sm_i8w (%zu names)\n",
              tuned, drafter);
  return 0;
}

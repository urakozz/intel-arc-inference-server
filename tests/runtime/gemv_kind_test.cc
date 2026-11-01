// gemv_kind_test - spec 28b's capture rule (runtime/gemv_kind.h, kernels::kSmallMTuning). Host only.
//
//   - M = 1 -> Gemv for every list kind: plain decode, ingest / generate, the MTP draft, the prefill tail's head,
//     the draft vocabulary's head all bind today's gemv / gemv_i8w names, on every model's shapes;
//   - M >= 2 -> SmallM: every tuned shape of Qwen3.8 and Agnes (their int4 linears and the int8 head) binds
//     gemm_sm at M = 2..8, and the shape's S is the one its consumer folds (the S table is one row per shape -
//     the same at every M, property R's precondition); untuned shapes (Ornith's) keep gemv;
//   - the override (CaptureOptions::force_linear) forces either kernel at any M;
//   - the DFlash drafter's name list binds gemm_sm_i8w at its 8 / 7 rows.
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
  for (const model::ModelDesc* d : {&model::qwen38(), &model::agnes()}) {
    const LinearId int4[] = {LinearId::QkvZ, LinearId::OutProj, LinearId::Qkv, LinearId::OProj, LinearId::GateUp,
                             LinearId::Down};
    for (LinearId id : int4) {
      const model::GemvShape& s = d->shape(id);
      // M = 1: today's binary, whatever the list
      CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(1), 1, s.K, s.N, s.S, s.layout),
               kernels::gemv_variant(1, s.K, s.N, s.S, s.layout));
      const kernels::SmallMTuning* t = kernels::small_m_tuning(s.K, s.N, s.S, s.layout);
      CHECK(t != nullptr);   // every Qwen3.8 / Agnes int4 decode linear is tuned
      CHECK_EQ(kernels::gemm_sm_out_s(*t), s.S);   // the consumer's fold is unchanged
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
  // Untuned shapes keep gemv at every M (Ornith's int4 a||b and attention, its 2048-wide int8 head).
  const model::ModelDesc& o = model::ornith();
  for (LinearId id : {LinearId::Qkv, LinearId::OProj, LinearId::QkvZ}) {
    const model::GemvShape& s = o.shape(id);
    for (uint32_t M = 1; M <= 4; ++M)
      CHECK_EQ(kernels::int4_linear_variant(runtime::gemv_kind(M), M, s.K, s.N, s.S, s.layout),
               kernels::gemv_variant(M, s.K, s.N, s.S, s.layout));
  }
  // The draft vocabulary's head (M = 1) and the MTP draft (M = 1): gemv_i8w.
  for (unsigned nv : {32768u, 65536u, 131072u})
    CHECK_EQ(kernels::int8_linear_variant(runtime::gemv_kind(1), 1, 5120, nv), kernels::gemv_i8w_variant(1, 5120, nv));
  // The DFlash drafter: every int8 linear of the lists at 8 rows, the head at 7, on gemm_sm_i8w.
  size_t drafter = 0;
  for (bool v2 : {true, false})
    for (unsigned dv : {0u, 32768u})
      for (const std::string& v : kernels::dflash::draft_variants(v2, true, true, dv)) {
        CHECK(v.rfind("gemv_i8w", 0) != 0);
        if (v.rfind("gemm_sm_i8w", 0) == 0) ++drafter;
      }
  CHECK(drafter > 0);
  for (const std::string& v : kernels::dflash::draft_variants(true, true, true, 0, true))
    CHECK(v.rfind("gemm_sm", 0) != 0);   // pre28b: today's names
  std::printf("gemv_kind_test OK: %zu tuned int4 decode linears (Qwen3.8 + Agnes), M = 1 on gemv, M >= 2 on "
              "gemm_sm, the override, %zu drafter bindings on gemm_sm_i8w\n",
              tuned, drafter);
  return 0;
}

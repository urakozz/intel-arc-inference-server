// Spec 17b (plan 17b's review, Task 1): a tensor-parallel card's descriptor, plan and kernel names - host only.
//
//   desc    model::tp_card_desc(qwen38, crossing) against spec 17 §R2.5's per-card table: heads, widths, every
//           linear's K / N / S / layout (the row-parallel S halved under slices, kept under folded), a||b's padding,
//           the head's 124160 rows, the small blocks' offsets (gdn_step's NEGA_OFF / DTBIAS_OFF in the CMake block),
//           the suffixes that name the per-card binaries; Ornith refused by name.
//   plan    a card's KV and GDN state are half one card's at every max_len and KV form; the crossing's bytes; the
//           list's launches (774 + 128); --max-len auto under TP is limited by the tighter card.
//   names   kernels::tp_cross_variant / tp_argmax_variant: the CMake block's names.
#include <array>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "check.h"
#include "kernels/tp_kernels.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/tp_plan.h"
#include "runtime/tp_protocol.h"

namespace {
using model::LinearId;

void check_shape(const model::ModelDesc& d, LinearId id, uint32_t K, uint32_t N, uint32_t S, uint32_t L) {
  const model::GemvShape& s = d.shape(id);
  if (s.K != K || s.N != N || s.S != S || s.layout != L)
    std::fprintf(stderr, "%s linear %d: %u x %u S%u L%u, want %u x %u S%u L%u\n", d.name.c_str(), int(id), s.K, s.N,
                 s.S, s.layout, K, N, S, L);
  CHECK(s.K == K && s.N == N && s.S == S && s.layout == L);
}

bool throws(const std::function<void()>& f, const char* what) {
  try {
    f();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(what) != std::string::npos;
  }
  return false;
}
}  // namespace

int main() {
  const model::ModelDesc& one = model::qwen38();
  for (model::TpCrossing c : {model::TpCrossing::Slices, model::TpCrossing::Folded}) {
    const model::ModelDesc& d = model::tp_card_desc(one, c);
    CHECK(&d == &model::tp_card_desc(one, c));   // a process-lifetime singleton
    const uint32_t rs = c == model::TpCrossing::Slices ? 2 : 4;   // the row-parallel S
    CHECK_EQ(d.layers, 64u);
    CHECK_EQ(d.gdn_layers, 48u);
    CHECK_EQ(d.fa_layers, 16u);
    CHECK_EQ(d.hidden, 5120u);
    CHECK_EQ(d.gdn_k_heads, 8u);
    CHECK_EQ(d.gdn_v_heads, 24u);
    CHECK_EQ(d.fa_q_heads, 12u);
    CHECK_EQ(d.fa_kv_heads, 2u);
    CHECK_EQ(d.intermediate, 8704u);
    CHECK_EQ(d.mtp_intermediate, 0u);
    CHECK(d.doc_w == 0);
    CHECK_EQ(d.vocab_used, one.vocab_used);
    check_shape(d, LinearId::QkvZ, 5120, 8192, 1, 1);    // q 1024 | k 1024 | v 3072 | z 3072
    check_shape(d, LinearId::AB, 5120, 128, 1, 0);
    CHECK_EQ(d.linear(LinearId::AB).pad_n, 48u);         // 24 a + 24 b, zero-padded to 128 as one card's 96
    check_shape(d, LinearId::OutProj, 3072, 5120, rs, 0);
    check_shape(d, LinearId::GateUp, 5120, 17408, 8, 0);
    check_shape(d, LinearId::Down, 8704, 5120, rs, 0);
    check_shape(d, LinearId::Qkv, 5120, 7168, 2, 0);     // q||gate 6144 | k 512 | v 512
    check_shape(d, LinearId::OProj, 3072, 5120, rs, 0);
    CHECK_EQ(d.lm_head(model::WeightKind::Int8).shape.N, 124160u);
    CHECK_EQ(d.lm_head(model::WeightKind::Int4).shape.N, 124160u);
    CHECK_EQ(d.ffn_fold_s(), rs);
    // Every row-parallel slice is the one-card slice length (the slices form's bitwise premise).
    if (c == model::TpCrossing::Slices)
      for (LinearId id : {LinearId::OutProj, LinearId::OProj, LinearId::Down})
        CHECK_EQ(d.shape(id).K / d.shape(id).S, one.shape(id).K / one.shape(id).S);
    const loader::SmallLayout sl = d.small_layout();
    CHECK_EQ(sl.gdn_off_nega, size_t(81920));    // NEGA_OFF 20480 floats
    CHECK_EQ(sl.gdn_off_dtbias, size_t(82016));  // DTBIAS_OFF 20504 floats
    CHECK_EQ(d.gdn_suffix(), std::string("_GK8V24"));
    CHECK_EQ(d.fa_suffix(), std::string("_Q12KV2"));
    CHECK_EQ(d.intermediate_suffix(), std::string("_I8704"));
    CHECK_EQ(d.hidden_suffix(), std::string());

    // The plan: half the per-position KV and half the GDN state at every length and form.
    for (runtime::KvCache kv : {runtime::KvCache::Bf16, runtime::KvCache::Int8})
      for (uint32_t len : {4096u, 16384u, 131072u}) {
        const runtime::PersistentSizes p1 = runtime::PersistentDims::sizes(len, one, kv);
        const runtime::PersistentSizes pc = runtime::PersistentDims::sizes(len, d, kv);
        CHECK_EQ(2 * pc.kv_k, p1.kv_k);
        CHECK_EQ(2 * pc.gdn_state, p1.gdn_state);
        CHECK_EQ(2 * pc.conv_ring, p1.conv_ring);
        const runtime::MemoryPlan mp = runtime::tp_card_plan(d, c, len, size_t(11e9), kv);
        CHECK_EQ(mp.kv, pc.kv_k + pc.kv_v);
        CHECK(mp.decode_state > runtime::tp_crossing_bytes(d, c));
      }
    const runtime::tp::Layout L = runtime::tp::layout(d, c);
    CHECK_EQ(L.s_msg, c == model::TpCrossing::Slices ? 2u : 1u);
    std::printf("%s: crossing %zu B a card (landing %zu B)\n", d.name.c_str(), runtime::tp_crossing_bytes(d, c),
                L.total_bytes());
    // --max-len auto: the tighter card decides.
    const std::array<size_t, 2> w{size_t(11e9), size_t(11e9)};
    const uint32_t both = runtime::tp_max_len_that_fits(d, c, w, {size_t(32e9), size_t(32e9)}, size_t(1.5e9), 262144);
    const uint32_t tight =
        runtime::tp_max_len_that_fits(d, c, w, {size_t(32e9), size_t(24e9)}, size_t(1.5e9), 262144);
    CHECK(both >= tight && tight > 0);
    CHECK_EQ(runtime::tp_max_len_that_fits(d, c, {size_t(31e9), size_t(11e9)}, {size_t(32e9), size_t(32e9)},
                                           size_t(1.5e9), 262144),
             0u);
    std::printf("%s: --max-len auto at 11 GB of weights a card: %u (32 + 32 GB), %u (32 + 24 GB)\n", d.name.c_str(),
                both, tight);
  }
  CHECK(throws([] { model::tp_card_desc(model::ornith(), model::TpCrossing::Slices); }, "built for Qwen3.8 only"));

  // The binaries the CMake block builds.
  CHECK_EQ(kernels::tp_cross_variant(1, 5120, 20, 2, false, 0), std::string("tp_cross_M1_K5120_G20_S2_X_R0"));
  CHECK_EQ(kernels::tp_cross_variant(1, 5120, 20, 4, true, 1), std::string("tp_cross_M1_K5120_G20_S4_F_R1"));
  CHECK_EQ(kernels::tp_argmax_variant(1, 124160, 123917, 124160), std::string("tp_argmax_M1_N124160_U123917_B124160"));
  CHECK_EQ(kernels::kCtrlOutValue * 4, uint32_t(offsetof(runtime::Control, out_value)));
  CHECK_EQ(runtime::tp::kMRows, kernels::kTpMRows);
  CHECK_EQ(runtime::tp::kFoldG, runtime::DecodeScratchDims::kNormGroups);
  std::printf("tp_desc_test: OK\n");
  return 0;
}

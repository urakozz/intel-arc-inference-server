#include "runtime/qwen4exp/qwen4exp_prefill_gdn.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill/profile.h"

// gdn_chunk_q4 - runtime/prefill/gdn.cc's launches (a MoE model's one-launch intra form: six), statement for
// statement in their order, grids and arguments, over GdnChunkBuffers instead of a PrefillScratch, on Qwen3.8's one-card
// bitwise forms (PERF-20 / PERF-21: front _PL, scan _OB; a tensor-parallel card's own _GK8V24 forms); the one binding
// that differs is the head (pf_gated_head _SIG_OB). The comments of gdn.cc hold for every launch here (ordering is the in-order list's alone;
// launch 9 is the only writer of the state).
namespace runtime::qwen4exp {
namespace {

using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;
namespace kq = kernels::qwen4exp;

// gdn.cc's constants (Qwen3.8's GDN shape, which this family shares: kq::kGdnHeads 48 v-heads, 16 k-heads, 10240
// conv channels; a tensor-parallel card's 8 / 24 heads, 5120 channels, come in through GdnChunkBuffers).
constexpr uint32_t kHd = kq::kGdnHd;
constexpr uint32_t kConvWg = 256, kStateColChunks = 4, kConvBlock = 128;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::qwen4exp::gdn_chunk_q4: " + what);
}

}  // namespace

void gdn_chunk_q4(prefill::Context& cx, prefill::KernelCache& kc, const GdnChunkBuffers& b, uint32_t pos, uint32_t C) {
  require(C > 0 && C <= kq::kPfCWide,
          "C = " + std::to_string(C) + " is outside (0, " + std::to_string(kq::kPfCWide) + "]");
  require(b.qkvz && b.ab && b.state && b.ring && b.small && b.y && b.xb && b.seed && b.g && b.beta && b.A && b.A2 && b.w &&
              b.u && b.o,
          "a buffer is missing");
  const char* const scan_entry = prefill::gdn_scan_entry_name();   // gdn.cc's (gdn.h)
  const uint32_t kKHeads = b.k_heads, kVHeads = b.v_heads, kConvDim = (2 * kKHeads + kVHeads) * kHd;
  require((kKHeads == 16 && kVHeads == kq::kGdnHeads) || (kKHeads == kq::kTpGdnK && kVHeads == kq::kTpGdnV),
          "the GDN heads are one card's (16 / 48) or a tensor-parallel card's (8 / 24)");
  require(kConvDim % kConvWg == 0, "the conv channels are whole work-groups");
  const uint32_t nch = (C + kGdnChunk - 1) / kGdnChunk;
  const uint32_t conv_groups = kConvDim / kConvWg;
  const void* gated_w = static_cast<const uint8_t*>(b.small) + b.gated_norm_off;
  const std::string conv = kernels::pf_gdn_conv_variant(kKHeads, kVHeads);
  const std::string scan = kernels::pf_gdn_scan_variant(kKHeads, kVHeads) + "_OB";
  const std::string pb = b.qkvz_bf16 ? "_PB" : "";   // the qkv||z partials as bf16 words (gdn.cc's partials_bf16)
  //  1 - lift the ring's three older slots (pos-3..pos-1) into a flat seed.
  cx.launch(kc(conv, "pf_gdn_seed"), conv_groups, 1, 1, {PtrArg(b.ring), PtrArg(b.seed), arg_val(pos)});
  prefill::profile_wait(cx, Phase::kGdnSeed);
  //  2-3 - conv + SiLU + q / k l2norm in one launch (phase 2 Task 2, gdn.cc's pf_gdn_front; PERF-20: _PL, a lane's
  //        partials read first), bitwise.
  cx.launch(kc(kernels::pf_gdn_front_variant(kKHeads, kVHeads) + "_PL" + pb, "pf_gdn_front"), conv_groups, (C + 31) / 32, 1,
            {PtrArg(b.qkvz), PtrArg(b.seed), PtrArg(b.small), PtrArg(b.xb), PtrArg(b.ring), arg_val(pos), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnConv);
  //  4 - head scalars and the intra-chunk cumulative gate.
  require(b.ab_pitch == 128 || b.ab_pitch == kq::kPfAbPitch, "a||b's row pitch is neither 128 nor the GEMM's");
  const std::string gate = b.ab_pitch == 128 ? conv : kernels::pf_gdn_gate_gemm_variant(kKHeads, kVHeads, kq::kPfAbPitch);
  cx.launch(kc(gate, "pf_gdn_gate"), kVHeads, nch, 1,
            {PtrArg(b.ab), PtrArg(b.small), PtrArg(b.g), PtrArg(b.beta), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnGate);
  //  5-8 (PERF-19, as Ornith's walk and Qwen3.8's one-card walk) - A, T, w / u and A2 in one launch: gdn.cc's
  //      pf_gdn_intra, one work-group per (v-head, 64-row sub-chunk), the scan's inputs in its layouts. Judged, not
  //      bitwise against pf_gdn_wy's four (DPAS dots, a blocked solve, T as hi / lo limbs; Q1-Q4 bf16 as before).
  //      Every walk binds it: one card and a pipeline stage at 16 / 48 heads (Qwen3.8's binary), a tensor-parallel
  //      card at 8 / 24 (pf_gdn_intra_GK8V24), INTRA_WU2 both. b.A is not read.
  cx.launch(kc(kernels::pf_gdn_intra_variant(kKHeads, kVHeads), "pf_gdn_intra"), kVHeads, nch, 1,
            {PtrArg(b.xb), PtrArg(b.g), PtrArg(b.beta), PtrArg(b.w), PtrArg(b.u), PtrArg(b.A2), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnWu);
  //  9 - the sequential chunk-to-chunk state scan; the only writer of the state. PERF-20 (_OB): o written as the bf16
  //      word the head rounds it to, q.S beside w.S, two barriers a step - bitwise.
  cx.launch(kc(scan, scan_entry), kVHeads, kStateColChunks, 1,
            {PtrArg(b.xb), PtrArg(b.w), PtrArg(b.u), PtrArg(b.A2), PtrArg(b.g), PtrArg(b.state), PtrArg(b.o), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnScan);
  // 10 - the gated head with the SIGMOID gate, one sub-group a (position, head), 8 heads a work-group, on the bf16 o
  //      (PERF-21, pf_gated_head _SIG_OB, bitwise _SIG): the mixer's last launch, y post-gate.
  cx.launch(kc(kernels::pf_gated_head_variant(kKHeads, kVHeads) + "_SIG_OB" + pb, "pf_gated_head_sg"), kVHeads / 8, C, 1,
            {PtrArg(b.qkvz), PtrArg(b.o), PtrArg(gated_w), PtrArg(b.y), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnHead);
}

}  // namespace runtime::qwen4exp

#include "runtime/qwen4exp/qwen4exp_prefill_gdn.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill/profile.h"

// gdn_chunk_q4 - runtime/prefill/gdn.cc's ten launches, statement for statement in their order, grids and
// arguments, over GdnChunkBuffers instead of a PrefillScratch; the one binding that differs is the tenth
// (pf_gated_head_SIG). The comments of gdn.cc hold for every launch here (ordering is the in-order list's alone;
// launch 9 is the only writer of the state).
namespace runtime::qwen4exp {
namespace {

using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;
namespace kq = kernels::qwen4exp;

// gdn.cc's constants (Qwen3.8's GDN shape, which this family shares: kq::kGdnHeads 48 v-heads, 16 k-heads, 10240
// conv channels).
constexpr uint32_t kKHeads = 16, kVHeads = kq::kGdnHeads, kHd = kq::kGdnHd, kConvDim = 10240;
constexpr uint32_t kConvWg = 256, kStateColChunks = 4, kConvBlock = 128;
static_assert((2 * kKHeads + kVHeads) * kHd == kConvDim, "the conv channels are q, k and v heads x 128");
static_assert(kConvDim % kConvWg == 0, "the conv channels are whole work-groups");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::qwen4exp::gdn_chunk_q4: " + what);
}

// gdn.cc's B70_PREFILL_GDN_SOLVE selector, the same rule (its static is private to gdn.cc): resolved once.
const char* solve_entry() {
  static const char* const entry = [] {
    const char* const v = std::getenv("B70_PREFILL_GDN_SOLVE");
    if (!v || !*v || std::string(v) == "vector") return "pf_gdn_solve";
    if (std::string(v) == "register") return "pf_gdn_solve_register";
    throw std::runtime_error("runtime::qwen4exp::gdn_chunk_q4: B70_PREFILL_GDN_SOLVE must be unset, 'vector', or "
                             "'register' (got '" + std::string(v) + "')");
  }();
  return entry;
}

}  // namespace

void gdn_chunk_q4(prefill::Context& cx, prefill::KernelCache& kc, const GdnChunkBuffers& b, uint32_t pos, uint32_t C) {
  require(C > 0 && C <= kq::kPfC, "C = " + std::to_string(C) + " is outside (0, " + std::to_string(kq::kPfC) + "]");
  require(b.qkvz && b.ab && b.state && b.ring && b.small && b.y && b.xb && b.seed && b.g && b.beta && b.A && b.A2 && b.w &&
              b.u && b.o,
          "a buffer is missing");
  const char* const scan_entry = prefill::gdn_scan_entry_name();   // the process-wide selector (gdn.h)
  const char* const solve = solve_entry();
  const uint32_t nch = (C + kGdnChunk - 1) / kGdnChunk;
  const uint32_t conv_groups = kConvDim / kConvWg;
  const void* gated_w = static_cast<const uint8_t*>(b.small) + b.gated_norm_off;
  const std::string conv = kernels::pf_gdn_conv_variant(kKHeads, kVHeads);
  const std::string wy = kernels::pf_gdn_wy_variant(kKHeads, kVHeads);
  const std::string scan = kernels::pf_gdn_scan_variant(kKHeads, kVHeads);
  //  1 - lift the ring's three older slots (pos-3..pos-1) into a flat seed.
  cx.launch(kc(conv, "pf_gdn_seed"), conv_groups, 1, 1, {PtrArg(b.ring), PtrArg(b.seed), arg_val(pos)});
  prefill::profile_wait(cx, Phase::kGdnSeed);
  //  2 - the batched conv1d + SiLU over the chunk, and the ring writeback.
  cx.launch(kc(conv, "pf_gdn_conv"), conv_groups, (C + kConvBlock - 1) / kConvBlock, 1,
            {PtrArg(b.qkvz), PtrArg(b.seed), PtrArg(b.small), PtrArg(b.xb), PtrArg(b.ring), arg_val(pos), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnConv);
  //  3 - l2norm q and k in place.
  cx.launch(kc(conv, "pf_gdn_l2norm"), 2 * kKHeads, C, 1, {PtrArg(b.xb), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnL2);
  //  4 - head scalars and the intra-chunk cumulative gate.
  cx.launch(kc(conv, "pf_gdn_gate"), kVHeads, nch, 1,
            {PtrArg(b.ab), PtrArg(b.small), PtrArg(b.g), PtrArg(b.beta), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnGate);
  //  5 - A = beta_i (k_i . k_j) exp(gc_i - gc_j), i > j.
  cx.launch(kc(wy, "pf_gdn_A"), kVHeads, nch, 4, {PtrArg(b.xb), PtrArg(b.g), PtrArg(b.beta), PtrArg(b.A), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnA);
  //  6 - T = (I + A)^-1 in place over A (gdn.cc's selector: pf_gdn_solve | pf_gdn_solve_register).
  cx.launch(kc(wy, solve), kVHeads, nch, 1, {PtrArg(b.A), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnSolve);
  //  7 - u = T vb and w = T kb.
  cx.launch(kc(wy, "pf_gdn_wu"), kVHeads, nch, 2,
            {PtrArg(b.xb), PtrArg(b.A), PtrArg(b.g), PtrArg(b.beta), PtrArg(b.w), PtrArg(b.u), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnWu);
  //  8 - A2 = (q_i . k_j) exp(gc_i - gc_j), j <= i.
  cx.launch(kc(wy, "pf_gdn_A2"), kVHeads, nch, 4, {PtrArg(b.xb), PtrArg(b.g), PtrArg(b.A2), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnA2);
  //  9 - the sequential chunk-to-chunk state scan; the only writer of the state.
  cx.launch(kc(scan, scan_entry), kVHeads, kStateColChunks, 1,
            {PtrArg(b.xb), PtrArg(b.w), PtrArg(b.u), PtrArg(b.A2), PtrArg(b.g), PtrArg(b.state), PtrArg(b.o), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnScan);
  // 10 - the gated head with the SIGMOID gate (pf_gated_head_SIG): the mixer's last launch, y post-gate.
  cx.launch(kc(kq::pf_gated_head_sig_variant(), "pf_gated_head"), kVHeads, C, 1,
            {PtrArg(b.qkvz), PtrArg(b.o), PtrArg(gated_w), PtrArg(b.y), arg_val(C)});
  prefill::profile_wait(cx, Phase::kGdnHead);
}

}  // namespace runtime::qwen4exp

#include "runtime/prefill/gdn.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"
#include "runtime/prefill/profile.h"

// The ten launches of one GDN layer's chunk, in order. Every one takes `C` as a
// RUNTIME argument, so one binary set serves every chunk width `--pp-chunk` can
// ask for (interfaces.md, "Layout conventions").
//
// **Ordering is the in-order immediate list and nothing else.** Launch 2 reads
// the ring slots launch 1 lifted; launch 3 rewrites `xb` in place after launch
// 2 wrote it; launch 6 overwrites `A` with `T` and launch 7 reads that `T`;
// launch 9 reads launches 7 and 8's outputs and is the only writer of
// `gdn_state`. No events, no fences: `src/sycl/context.cc` creates the list
// with ZE_COMMAND_QUEUE_FLAG_IN_ORDER and `tests/prefill/context_test.cc`'s
// 1024-launch chain is the empirical check on that guarantee.
//
// **Note for plan 6c (L2).** Launches 5, 7, 8 and the two contractions inside 9
// are 64x128x128-shaped per head and are plain vector code here. 6c MAY route
// 5, 7 and 8 through `gemm_bf16` - spec §3.4's "goes through the §3.2 interface
// where the shape suits DPAS, plain vector code where it does not". **Launch 6,
// the triangular solve, stays vector code**: it is inherently sequential in `i`
// and has no DPAS shape. Any such reroute changes the reduction order and
// therefore the band `tests/prefill/gdn_chunk_test.cc` records, so it is a
// re-measure, not a refactor.
namespace runtime::prefill {
namespace {
using Q = model::Qwen35;

constexpr uint32_t kConvRows = 10240;
constexpr uint32_t kConvGroups = kConvRows / 256;      // 40 work-groups of 256
constexpr uint32_t kKHeads = Q::kGdnKHeads;            // 16
constexpr uint32_t kStateColChunks = 4;                // gdn_step.cl's grid.y

// Ruling A27: `pf_gdn_conv`'s position range is blocked, and this is the only
// place the block count is decided. The kernel derives its own block WIDTH from
// `get_num_groups(1)` rather than sharing this literal, so the two cannot
// disagree about coverage; what this number buys is the grid. 128 positions per
// block puts C = 2048 at 16 blocks = 640 work-groups = 10,240 threads (5 waves
// of the 2048-slot machine, ~640 KB of loads in flight against the ~295 KB
// Little's law asks for at 590 GB/s) while the three-position halo each block
// re-reads stays at 2.3% of the block's work.
constexpr uint32_t kConvBlock = 128;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::gdn_chunk: " + what);
}
}  // namespace

void gdn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
               const float* qkvz_partials, const float* ab_out, float* gdn_state,
               uint16_t* conv_ring, const void* small, uint16_t* y) {
  require(C > 0, "C must be > 0");
  require(C <= PrefillScratch::kC,
          "C = " + std::to_string(C) + " exceeds PrefillScratch::kC = " +
              std::to_string(PrefillScratch::kC) + " - the scratch is sized for kC");

  const uint32_t nch = (C + PrefillScratch::kGdnChunk - 1) / PrefillScratch::kGdnChunk;
  const uint32_t heads = Q::kGdnVHeads;                 // 48
  const void* gated_w = static_cast<const uint8_t*>(small) + loader::kGdnOffGatedNorm;

  void* p_xb = s.gdn_xb.ptr();
  void* p_seed = s.gdn_seed.ptr();
  void* p_g = s.gdn_g.ptr();
  void* p_beta = s.gdn_beta.ptr();
  void* p_A = s.gdn_A.ptr();
  void* p_A2 = s.gdn_A2.ptr();
  void* p_w = s.gdn_w.ptr();
  void* p_u = s.gdn_u.ptr();
  void* p_o = s.gdn_o.ptr();

  const std::string conv = kernels::pf_gdn_conv_variant();
  const std::string wy = kernels::pf_gdn_wy_variant();
  const std::string scan = kernels::pf_gdn_scan_variant();

  //  1 - lift the ring's three older slots (pos-3..pos-1) into a flat seed.
  cx.launch(kc(conv, "pf_gdn_seed"), kConvGroups, 1, 1,
            {PtrArg(conv_ring), PtrArg(p_seed), arg_val(pos)});
  profile_wait(cx, Phase::kGdnSeed);
  //  2 - the batched conv1d + SiLU over the chunk, and the ring writeback.
  //      Grid.y is A27's position blocking; grid.x and the work-group are
  //      unchanged.
  const uint32_t conv_blocks = (C + kConvBlock - 1) / kConvBlock;
  cx.launch(kc(conv, "pf_gdn_conv"), kConvGroups, conv_blocks, 1,
            {PtrArg(qkvz_partials), PtrArg(p_seed), PtrArg(small), PtrArg(p_xb),
             PtrArg(conv_ring), arg_val(pos), arg_val(C)});
  profile_wait(cx, Phase::kGdnConv);
  //  3 - l2norm q and k in place.
  cx.launch(kc(conv, "pf_gdn_l2norm"), 2 * kKHeads, C, 1, {PtrArg(p_xb), arg_val(C)});
  profile_wait(cx, Phase::kGdnL2);
  //  4 - head scalars and the intra-chunk cumulative gate.
  cx.launch(kc(conv, "pf_gdn_gate"), heads, nch, 1,
            {PtrArg(ab_out), PtrArg(small), PtrArg(p_g), PtrArg(p_beta), arg_val(C)});
  profile_wait(cx, Phase::kGdnGate);
  //  5 - A = beta_i (k_i . k_j) exp(gc_i - gc_j), i > j.
  cx.launch(kc(wy, "pf_gdn_A"), heads, nch, 1,
            {PtrArg(p_xb), PtrArg(p_g), PtrArg(p_beta), PtrArg(p_A), arg_val(C)});
  profile_wait(cx, Phase::kGdnA);
  //  6 - T = (I - A)^-1, IN PLACE over A.
  cx.launch(kc(wy, "pf_gdn_solve"), heads, nch, 1, {PtrArg(p_A), arg_val(C)});
  profile_wait(cx, Phase::kGdnSolve);
  //  7 - vb/kb, then u = T vb and w = T kb.
  cx.launch(kc(wy, "pf_gdn_wu"), heads, nch, 1,
            {PtrArg(p_xb), PtrArg(p_A), PtrArg(p_g), PtrArg(p_beta), PtrArg(p_w), PtrArg(p_u),
             arg_val(C)});
  profile_wait(cx, Phase::kGdnWu);
  //  8 - A2 = (q_i . k_j) exp(gc_i - gc_j), j <= i. NOTE the diagonal.
  cx.launch(kc(wy, "pf_gdn_A2"), heads, nch, 1,
            {PtrArg(p_xb), PtrArg(p_g), PtrArg(p_A2), arg_val(C)});
  profile_wait(cx, Phase::kGdnA2);
  //  9 - the sequential chunk-to-chunk state scan; the only writer of gdn_state.
  cx.launch(kc(scan, "pf_gdn_scan"), heads, kStateColChunks, 1,
            {PtrArg(p_xb), PtrArg(p_w), PtrArg(p_u), PtrArg(p_A2), PtrArg(p_g),
             PtrArg(gdn_state), PtrArg(p_o), arg_val(C)});
  profile_wait(cx, Phase::kGdnScan);
  // 10 - the gated head. Ruling R3: it belongs to the mixer, not to the caller.
  cx.launch(kc.get(kernels::pf_gated_head_variant(), "pf_gated_head"), heads, C, 1,
            {PtrArg(qkvz_partials), PtrArg(p_o), PtrArg(gated_w), PtrArg(y), arg_val(C)});
  profile_wait(cx, Phase::kGdnHead);
}

}  // namespace runtime::prefill

#pragma once
#include <cstddef>
#include <cstdint>

#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"   // kGdnChunkLaunches, kGdnChunk, gdn_scratch_sizes

// Spec 21d: gdn_chunk_q4 - the WY-chunked gated delta rule for ONE Qwen3.8-Flash-Next GDN layer over `C` new
// positions: runtime::prefill::gdn_chunk's ten launches (src/runtime/prefill/gdn.cc), in its order and with its
// binaries, over this family's own buffers - the tenth launch is pf_gated_head_SIG (the sigmoid output gate,
// `output_gate_type: sigmoid`) where gdn_chunk binds pf_gated_head's silu form by name.
//
// Why a sibling and not gdn_chunk itself (plan 21d "Dependencies"): gdn_chunk takes a runtime::PrefillScratch sized by
// a model::ModelDesc (Qwen3.8's ~GB of scratch, its descriptor's GDN shape) and binds pf_gated_head by name.
// gdn.cc is not edited. The GDN shape is Qwen3.8's (16 k / 48 v heads x 128, 10240 conv channels; the small block's
// conv / negA / dt_bias at Qwen3.8's offsets: loader::make_small_layout(2560, 10240, 48) == kQwen38Small's GDN part),
// so Qwen3.8's pf_gdn_conv / pf_gdn_wy / pf_gdn_scan binaries apply as built; the scan and solve entries follow
// the same process-wide selectors gdn.cc reads (B70_PREFILL_GDN_SCAN, B70_PREFILL_GDN_SOLVE: gdn_scan_entry_name()).
//
// It reads and writes the SAME gdn_state and conv_ring decode's gdn_step uses, in decode's layouts, so decode
// continues from pos + C with no translation; the gated head's output `y` is the out_proj input.
namespace runtime::qwen4exp {

// Every buffer gdn_chunk_q4 reads or writes (PrefillScratch's gdn_* fields, explicit):
//   qkvz      fp32 [C][16384]   the qkv||z linear at S = 1 (rounded to bf16 inside, gdn_step's single point)
//   ab        fp32 [C][128]     a at [0, 48), b at [48, 96)
//   state     fp32 [48][128][128]  the layer's decode state (in place)
//   ring      bf16 [16][10240]  the layer's decode conv ring (in place)
//   small     the layer's GDN small block (conv weights, negA, dt_bias, the gated norm at gated_norm_off)
//   y         bf16 [C][6144]    the gated head's output (out_proj's A)
//   the chain's scratch at gdn_scratch_sizes(C) (qwen4exp_sizes.h)
struct GdnChunkBuffers {
  const float* qkvz = nullptr;
  const float* ab = nullptr;
  float* state = nullptr;
  uint16_t* ring = nullptr;
  const void* small = nullptr;
  size_t gated_norm_off = 0;
  uint16_t* y = nullptr;
  void *xb = nullptr, *seed = nullptr, *g = nullptr, *beta = nullptr, *A = nullptr, *A2 = nullptr, *w = nullptr,
       *u = nullptr, *o = nullptr;
};

// kGdnChunkLaunches (10) launches on cx, in order; C in (0, kPfC]; throws otherwise or on a missing buffer.
void gdn_chunk_q4(prefill::Context& cx, prefill::KernelCache& kc, const GdnChunkBuffers& b, uint32_t pos, uint32_t C);

}  // namespace runtime::qwen4exp

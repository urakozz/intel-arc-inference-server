#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "loader/moe_layout.h"
#include "loader/quant.h"
#include "model/model_desc.h"

// Spec 15c: one mixture-of-experts layer repacked on the host into the device form
// loader/moe_layout.h defines - no Level Zero, so tests/loader/ornith_repack_test.cc
// runs it on any host over a synthetic checkpoint. loader::load uploads the result.
namespace loader {

// Where the repack finds a layer's tensors, by LAYER-RELATIVE engine part name
// ("mlp.experts.3.gate_proj", "mlp.gate", "mlp.shared_expert.down_proj"). The loader
// binds these to LinearSrc::classify on the checkpoint (and marks the names
// consumed); a test binds them to its fixture.
struct MoeSource {
  // True when the checkpoint has this linear in either form (`.qweight` or `.weight`).
  std::function<bool(const std::string& part)> has;
  // The linear, classified by content (loader/quant.h); throws if absent.
  std::function<LinearSrc(const std::string& part)> linear;
};

// The host copy of one layer, sized by moe_layer_bytes (reused across layers).
struct MoeHost {
  std::vector<uint16_t> router;    // bf16 tiled [router_n][hidden]
  std::vector<uint32_t> gate_up;   // blocks x gate_up_block bytes, layout-1 tiles
  std::vector<uint32_t> down;      // blocks x down_block bytes, layout-1 tiles
  // Which per-expert form the checkpoint used: separate `experts.E.gate_proj` /
  // `up_proj` (AutoRound's export) or one `experts.E.gate_up_proj` whose first half of
  // output columns is gate (vLLM's per-expert fused form). Decided per layer from
  // expert 0 and required of every expert.
  bool fused_gate_up = false;
  // The checkpoint bytes this layer consumed (int4 qweight + f16 scales of every
  // expert and the shared expert, bf16 router and gate) - for the load report.
  size_t int4_src_bytes = 0, bf16_src_bytes = 0;
  // Spec 15e: the expert linears (shared expert included) that arrived bf16 and were
  // quantised here (loader/rtn.h) - their bf16 bytes are in bf16_src_bytes. 0 unless
  // repack_moe_layer was asked to.
  size_t rtn_linears = 0;
};

// Repacks layer `what` (a label for error messages, e.g. "layer 3") of the MoE model
// `d`: the router || shared-gate rows, then every expert block (shared expert last).
// Throws by name on a missing tensor, a wrong kind (an int4 router, a bf16 expert),
// a shape that is not the descriptor's, or mixed per-expert forms.
//
// `rtn_bf16_experts` (spec 15e, the MTP head's MoE layer only): an expert linear - routed
// or shared - that the checkpoint ships bf16 is quantised to int4 g64 sym on the host
// (loader::rtn_int4_g64) and repacked from that, instead of refused; an int4 one is
// repacked as shipped, linear by linear. The main layers keep the refusal: their experts
// must be the quantiser's (spec 15 decision 1).
void repack_moe_layer(const model::ModelDesc& d, const MoeSource& src, MoeHost& out,
                      const std::string& what, bool rtn_bf16_experts = false);

}  // namespace loader

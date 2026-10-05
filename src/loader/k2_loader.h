#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/k2_layout.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"   // DeviceWeight - reused, not re-declared
#include "model/k2_horizon.h"

// Spec 18b: K2-Horizon on one B70 - the checkpoint's tensors in the layouts
// src/kernels/k2/*.cl and the reused gemv / gemv_bf16 / gemv_i8w kernels read
// (loader/k2_layout.h sizes every allocation; loader/k2_repack.h is the host half).
// A second loader BESIDE loader::load (spec 18 §5.1): loader.cc is untouched; the public
// pieces are reused - resolve_snapshot, QuantConfig::parse, SafetensorsSet,
// assert_quant_invariants, LinearSrc::classify, the int8 lm_head quantiser.
namespace loader {

// One layer on the card. The sparse-only members are null on a dense layer.
struct K2Layer {
  std::vector<DeviceWeight> linears;   // K2Desc::layer_linears(layer) order, int4 layout 0
  std::unique_ptr<l0::Mem> norms;      // fp32 plain w: input [hidden] || post [hidden]
  std::unique_ptr<l0::Mem> route;      // fp32: MoE bias [router_n] || MoVA bias [value_experts]
  std::unique_ptr<DeviceWeight> router;   // bf16 tiled {hidden, router_n}, rows >= experts zero
  std::unique_ptr<l0::Mem> value;      // int4 layout-1 blocks [value_experts][hidden x kv_n]
  std::unique_ptr<l0::Mem> gate_up;    // int4 layout-1 blocks [experts + 1][hidden x 2 I]
  std::unique_ptr<l0::Mem> down;       // int4 layout-1 blocks [experts + 1][I x hidden]
};

struct K2LoadReport {
  K2WeightBytes bytes;          // what was allocated, bucket by bucket (k2_layout.h's formula)
  size_t rope_bytes = 0;        // the RoPE table (K2Desc::rope_table_bytes(max_len))
  size_t src_int4_bytes = 0;    // checkpoint int4 words + scales read
  size_t src_bf16_bytes = 0;    // checkpoint bf16 / f16 read (routers, norms, biases, embed, head)
  // What ONE decode token reads (derived): every non-expert linear, the routers, top-8 +
  // shared of the MoE blocks and top-4 of the value blocks per sparse layer, the norms
  // and route blocks, the head. Not the embedding (one row) nor the RoPE table.
  size_t read_per_token = 0;
  size_t unconsumed = 0;        // must be 0
  size_t subnormal_scales = 0;
  std::string quant_note;       // loader::ct_conversion_note (compressed-tensors only)
  double lm_head_quant_seconds = 0, seconds = 0;
  size_t total() const { return bytes.total() + rope_bytes; }
};

struct K2LoadedModel {
  const model::K2Desc* desc = nullptr;
  std::vector<K2Layer> layers;
  std::unique_ptr<l0::Mem> embed;        // bf16 [vocab][hidden] row-major
  std::unique_ptr<l0::Mem> final_norm;   // fp32 plain w [hidden]
  std::unique_ptr<l0::Mem> rope;         // fp32 [max_len][2][head_dim / 2], bf16-valued
  std::unique_ptr<DeviceWeight> lm_head; // bf16 tiled, or int8 tiled + fp32 row scales
  K2LoadReport report;
  uint32_t max_len = 0;
  uint32_t trained_max_len = 0;          // config.json max_position_embeddings (524288)
};

// True when `snapshot_dir`'s config.json says model_type "k2_horizon" (the CLIs' dispatch).
// Throws when config.json cannot be read.
bool is_k2_checkpoint(const std::string& snapshot_dir);

// Loads the K2-Horizon checkpoint (resolve_snapshot rules) onto ctx's device. Before a
// tensor is read: config.json held to model::k2() (check_k2_checkpoint_config), max_len
// within the trained context, the quantisation invariants, every tensor name accounted
// for (K2Checkpoint::check_names). After: zero unconsumed tensors, and every bucket equal
// to loader::k2_weight_bytes (the planner's figure). `lm_head` Int8 quantises the bf16
// head at load (spec 9); the checkpoint's bf16 head otherwise. Prints its report.
K2LoadedModel load_k2(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len,
                      LmHeadForm lm_head = LmHeadForm::Checkpoint);

// `--max-len auto` (spec 6 §10, as loader::set_max_len): a new RoPE table of `max_len`
// positions, the old freed. Only before anything captured `m.rope`.
void set_max_len_k2(l0::Context& ctx, K2LoadedModel& m, uint32_t max_len);

}  // namespace loader

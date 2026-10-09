#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"   // DeviceWeight - reused, not re-declared
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_ple.h"
#include "model/qwen4exp.h"

// Spec 21b Task 6: Qwen3.8-Flash-Next on one or two B70s - the checkpoint's tensors in the layouts
// loader/qwen4exp_layout.h sizes (loader/qwen4exp_repack.h is the host half), the PLE table pinned in host USM
// (loader/qwen4exp_ple.h). A loader BESIDE loader::load, load_k2 and load_kolibri1 (spec 21 §6): their files
// are untouched; the public pieces are reused - resolve_snapshot, QuantConfig::parse, SafetensorsSet,
// assert_quant_invariants, check_quant_scan, LinearSrc::classify (its exact g128 -> g64 expansion), the int8
// lm_head quantiser, rtn_int4_g64.
//
// **Both checkpoints the engine accepts:** Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (int4 g128 routed experts
// through the expansion, its bf16 dense / shared / MTP as shipped - the interim, spec 21 §5) and ours (21q:
// AutoRound int4 g64 sym). The forms are read from the names (loader::q4_forms).
//
// **Placement at load.** Each layer is repacked from the shards and uploaded straight onto the device the
// placement gives it (no load-then-place). Device 0 holds the embedding, the last device the final mixer,
// lm_head and (mtp) the MTP head, every device with a QSA layer its RoPE table; the PLE table is host USM,
// allocated once through the context of the device that holds the PLE layer and visible to every device of
// the context (spec 16 decision 1: one context).
namespace loader {

struct Q4Layer {                                   // one decoder layer on its device
  std::unique_ptr<l0::Mem> hc_attn, hc_mlp;        // per HC: down||inject tiles | up tiles | norm (q4_hc_offsets)
  std::unique_ptr<DeviceWeight> gdn_qkvz, gdn_ab, gdn_out;   // GDN layers
  std::unique_ptr<l0::Mem> gdn_small;              // q4_gdn_small (make_small_layout)
  std::unique_ptr<DeviceWeight> qsa_qkvg, qsa_idx, qsa_o;    // QSA layers
  std::unique_ptr<l0::Mem> qsa_small;              // q4_qsa_small
  std::unique_ptr<DeviceWeight> router;            // {2560, 528}: rows 0..511 the router, 512 the shared gate
  std::unique_ptr<l0::Mem> gate_up, down;          // 512 layout-1 blocks each, block e at q4_gate_up_offset /
                                                   // q4_down_offset (spec 22's ranges)
  std::unique_ptr<l0::Mem> shared;                 // gate||up then down: int4 layout-1 blocks or bf16 tiles
  size_t shared_down_offset = 0;                   // = q4_shared_gate_up_bytes
  std::unique_ptr<l0::Mem> ple;                    // the PLE layer only: key||value tiles, norms, conv (q4_ple_offsets)
};

struct Q4DevicePart {
  uint32_t device = 0, first = 0, end = 0;         // layers [first, end) live here
  std::vector<Q4Layer> layers;                     // index = layer - first
  std::unique_ptr<l0::Mem> embed;                  // device 0 only
  std::unique_ptr<l0::Mem> rope;                   // every device with a QSA layer (or the MTP head's)
  std::unique_ptr<l0::Mem> final_mixer;            // last device: q4_hc_offsets(inject = false)
  std::unique_ptr<DeviceWeight> lm_head;           // last device: int8 + fp32 row scales, or bf16 tiles
  std::unique_ptr<Q4Layer> mtp;                    // last device, when loaded: the head's QSA layer + MoE
  std::unique_ptr<l0::Mem> mtp_fc;                 // q4_mtp_fc_offsets
  std::unique_ptr<l0::Mem> mtp_mixer;              // the head's own final mixer
  size_t bytes = 0;                                // weights, == q4_device_weight_bytes (RoPE apart)
  size_t rope_bytes = 0;
  const Q4Layer& layer(uint32_t l) const { return layers.at(l - first); }
};

struct Q4LoadedModel {
  model::Qwen4ExpDesc desc;                        // layers = layers_limit when one was given
  model::Q4Placement placement;
  std::vector<Q4DevicePart> parts;                 // one per device
  Q4PleTable ple;                                  // host USM, device-visible from every device of the context
  uint32_t ple_device = 0;                         // whose context allocated it and holds its pointer table
  uint32_t max_len = 0, checkpoint_layers = 0;
  bool int8_head = false, mtp = false;
  size_t unconsumed = 0, read_per_token = 0, subnormal_scales = 0, skipped = 0;
  double seconds = 0, lm_head_quant_seconds = 0;
  size_t total_bytes() const;                      // every part's weights + RoPE (the PLE table apart)
};

// True when `snapshot_dir`'s config.json says model_type "qwen4_exp" (or "qwen4_exp_text") - the CLIs' dispatch.
bool is_qwen4exp_checkpoint(const std::string& snapshot_dir);

// The descriptor load_qwen4exp would build, from config.json and the tensor names alone (no device, no tensor
// read): the CLI's planner needs it before the load (--layers auto, --pp, --max-len auto).
model::Qwen4ExpDesc qwen4exp_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit = 0);

// Loads the checkpoint (resolve_snapshot rules) onto `devices` (devices.size() == placement.devices; device 1 a
// view of device 0's context). Before a byte is allocated: config.json held to model::qwen4exp() at its layer
// count, max_len within the trained 262144, the quantisation invariants (assert_quant_invariants +
// check_quant_scan; the experts' group = quantization_config.group_size: 64 ours, 128 Intel's), the forms from
// the names, every name accounted for both ways, the plan (runtime::qwen4exp::plan + require_fits at the
// default 1.5 GB reserve - the full model is refused naming spec 22). `layers_limit` > 0 loads only layers
// [0, layers_limit) (development mode; the later layers' tensors consumed by design, one note line). `mtp`
// loads the MTP head (its bf16 experts RTN-quantised to int4 g64). `ple_dir` empty: q4_ple_dir(snapshot).
// After: zero unconsumed tensors and each part's bytes equal to q4_device_weight_bytes. One report line per
// device plus one for the host table.
Q4LoadedModel load_qwen4exp(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                            uint32_t max_len, const model::Q4Placement& placement,
                            LmHeadForm lm_head = LmHeadForm::Int8, uint32_t layers_limit = 0, bool mtp = false,
                            const std::string& ple_dir = "");

}  // namespace loader

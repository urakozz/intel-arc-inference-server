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
#include "runtime/qwen4exp/qwen4exp_sizes.h"   // ExpertTier

// Spec 21-QFN-v1b Task 6: Qwen3.8-Flash-Next on one or two B70s - the checkpoint's tensors in the layouts
// loader/qwen4exp_layout.h sizes (loader/qwen4exp_repack.h is the host half), the PLE table pinned in host USM
// (loader/qwen4exp_ple.h). A loader BESIDE loader::load and load_kolibri1 (spec 21-QFN-v1 §6): their files
// are untouched; the public pieces are reused - resolve_snapshot, QuantConfig::parse, SafetensorsSet,
// assert_quant_invariants, check_quant_scan, LinearSrc::classify (its exact g128 -> g64 expansion), the int8
// lm_head quantiser, rtn_int4_g64.
//
// **Both checkpoints the engine accepts:** Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (int4 g128 routed experts
// through the expansion, its bf16 dense / shared / MTP as shipped - the interim, spec 21-QFN-v1 §5) and ours (21q:
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
                                                   // q4_down_offset (spec 22's ranges); with the expert tier the
                                                   // layer's VRAM arena: its resident experts' blocks in id order
                                                   // (null when none is resident)
  std::unique_ptr<l0::Mem> mirror;                 // the tier: the non-resident experts' gate||up blocks then their
                                                   // down blocks, in id order, pinned host memory (null: all resident)
  void* expert_row = nullptr;                      // the table form (_TBL): the layer's u64 row on its device - block
                                                   // addresses, gate||up at [0, 512), down at [512, 1024); null: the
                                                   // direct form (the MTP head, B70_Q4_EXPERTS=direct)
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
  std::unique_ptr<l0::Mem> expert_table;           // the table form: u64 [its layers][1024] (Q4Layer::expert_row)
  // The table form's load check (the engine runs q4_expert_check at construction): sample g reads the u64 at table
  // entry check_idx[g] ((layer - first) x 1024 + slot) + check_off[g] bytes; check_words[g] is the checkpoint's word
  // there. Each block's first and last word and one every 1 MiB: every 2 MiB page of every arena and mirror.
  std::vector<uint32_t> check_idx, check_off;
  std::vector<uint64_t> check_words;
  size_t bytes = 0;                               // weights, == q4_device_weight_bytes less the experts in host
                                                   // memory (RoPE and the expert table apart)
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
  uint32_t pf_rows = runtime::qwen4exp::kPfC;      // the engines' prefill chunk rows (the planner's, settle / pf_rows)
  bool int8_head = false, mtp = false;
  runtime::qwen4exp::ExpertTier tier;              // where each routed expert lives (all resident without a tier);
                                                   // under tp card 0's halves
  runtime::qwen4exp::ExpertTier tier1;             // plan 21-QFN-v2b, tp: card 1's resident halves
  std::unique_ptr<l0::Mem> ple_ptrs1;              // tp: the PLE pointer table on card 1 (card 0's is ple.ptrs)
  bool expert_table = false;                       // the layers bind the table form (B70_Q4_EXPERTS, default table)
  size_t mirror_bytes = 0;                         // the tier's pinned host bytes
  size_t unconsumed = 0, read_per_token = 0, subnormal_scales = 0, skipped = 0;
  double seconds = 0, lm_head_quant_seconds = 0;
  double expert_check_ms = 0;                      // the table check's cost: the loader's sampling + the device pass
  size_t total_bytes() const;                     // every part's weights + RoPE (the PLE table apart)
};

// True when `snapshot_dir`'s config.json says model_type "qwen4_exp" (or "qwen4_exp_text") - the CLIs' dispatch.
bool is_qwen4exp_checkpoint(const std::string& snapshot_dir);

// The descriptor load_qwen4exp would build, from config.json and the tensor names alone (no device, no tensor
// read): the CLI's planner needs it before the load (--layers auto, --parallel, --max-len auto).
model::Qwen4ExpDesc qwen4exp_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit = 0);

// Loads the checkpoint (resolve_snapshot rules) onto `devices` (devices.size() == placement.devices; device 1 a
// view of device 0's context). Before a byte is allocated: config.json held to model::qwen4exp() at its layer
// count, max_len within the trained 262144, the quantisation invariants (assert_quant_invariants +
// check_quant_scan; the experts' group = quantization_config.group_size: 64 ours, 128 Intel's), the forms from
// the names, every name accounted for both ways, the plan (runtime::qwen4exp::plan + require_fits at the
// default 1.5 GB reserve - the full model is refused naming spec 22). `layers_limit` > 0 loads only layers
// [0, layers_limit) (development mode; the later layers' tensors consumed by design, one note line). `mtp`
// loads the MTP head (its bf16 experts RTN-quantised to int4 g64). The PLE table: q4_ple_dir(snapshot).
// After: zero unconsumed tensors and each part's bytes equal to q4_device_weight_bytes. One report line per
// device plus one for the host table.
// Plan 21-QFN-v2a Task 8, `tier` (null: every expert resident): the plan less the experts in host memory, each layer's
// VRAM arena of its resident experts, its pinned host mirror of the rest (refused before the PLE table's pin when the
// table, the mirrors and a 16 GiB margin exceed MemAvailable, naming the bytes: check_q4_host_fit), each device's
// table of block addresses read back equal, and the load check's samples (Q4DevicePart::check_*; the engine reads
// them through the table). The layers bind the table form unless B70_Q4_EXPERTS=direct (refused with any expert in
// host memory). `pf_rows`: the prefill scratch's chunk rows the caller planned (runtime::qwen4exp::kPfC or kPfCWide:
// pf_chunk_choice), kept for the engine; a B70_Q4_RESIDENT tier is filled from the budget at those rows.
// Plan 21-QFN-v2b, `placement.tp`: each layer repacked once and cut into its two halves (loader::q4_tp_layer), each half
// uploaded onto its card with that card's tier (`tier` card 0's, `tier1` card 1's; null: B70_Q4_RESIDENT's placement
// through tp_fill_tier, else every expert resident); the final mixer and lm_head's vocabulary half (rows [124160 r,
// + 124160), quantised per row: one card's rows' bytes) on both cards, the embedding where placement.holds_embed, the
// PLE table pinned once with a pointer table on each card. Each part's bytes are the card plan's (runtime::qwen4exp::
// tp_plan). The MTP head under tp is not built yet (refused by name).
Q4LoadedModel load_qwen4exp(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                            uint32_t max_len, const model::Q4Placement& placement,
                            LmHeadForm lm_head = LmHeadForm::Int8, uint32_t layers_limit = 0, bool mtp = false,
                            const runtime::qwen4exp::ExpertTier* tier = nullptr,
                            const runtime::qwen4exp::ExpertTier* tier1 = nullptr,
                            uint32_t pf_rows = runtime::qwen4exp::kPfC);

// The routing profile (`<snapshot>/expert-profile.safetensors`, shipped in our model repo, else
// `<snapshot>-expert-profile.safetensors` beside the checkpoint: offload_curve.py profile's
// u32 counts [layers][experts], format "spec22-profile-1"): its counts, or empty when there is no such file; a file
// of another shape or format is refused by name.
std::vector<uint32_t> load_q4_profile(const std::string& snapshot_or_repo, const model::Qwen4ExpDesc& full,
                                      std::string* path = nullptr);

}  // namespace loader

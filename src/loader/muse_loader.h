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
#include "loader/muse_layout.h"
#include "loader/muse_repack.h"
#include "model/muse_glimmer.h"
#include "runtime/muse/muse_sizes.h"

// Spec 23b: Muse-Glimmer-30B on one or two B70s - the checkpoint's tensors in the layouts the reused gemv.cl
// (int4 layout 0, the descriptor's S), gemv_i8w / gemv_bf16 (the head) and 23c's kernels read
// (loader/muse_layout.h sizes every allocation; loader/muse_repack.h is the host half). A loader BESIDE the others
// (spec 23 §5.1): their files are untouched; the public pieces are reused - resolve_snapshot, QuantConfig::parse,
// SafetensorsSet, assert_quant_invariants, LinearSrc::classify, the int8 lm_head quantiser.
//
// **The plan first.** runtime::muse::plan_at + require_fits at the default 1.5 GB reserve against each device's
// memory before the first allocation; then each layer is repacked from the shards and uploaded straight onto the
// device the placement gives it. Device 0 also holds the embedding, the last device the final norm and lm_head
// (int8 built on the host at load - spec 9 - or bf16 tiled).
namespace loader {

struct MuseLayer {
  std::unique_ptr<DeviceWeight> qgkv, oproj, gate_up, down;   // int4 layout 0 (words + scales), the desc's S
  std::unique_ptr<l0::Mem> norms;                             // fp32 [4][hidden] (MuseGlimmerDesc::norm_off_*)
};

struct MuseDevicePart {
  uint32_t device = 0, first = 0, end = 0;   // layers [first, end) live here
  std::vector<MuseLayer> layers;             // index = layer - first
  std::unique_ptr<l0::Mem> embed;            // device 0: bf16 [vocab][hidden] row-major (gathered)
  std::unique_ptr<l0::Mem> final_norm;       // last device: fp32 plain w [hidden]
  std::unique_ptr<DeviceWeight> lm_head;     // last device: int8 tiled + fp32 row scales, or bf16 tiled
  size_t bytes = 0;                          // == muse_device_weight_bytes
  const MuseLayer& layer(uint32_t l) const { return layers.at(l - first); }
};

struct MuseWeights {
  model::MuseGlimmerDesc desc;               // layers = layers_override when one was given
  model::MusePlacement placement;
  std::vector<MuseDevicePart> parts;         // one per device
  uint32_t max_len = 0;
  bool int8_head = true;
  MuseQuantForm form = MuseQuantForm::AutoRound;
  uint32_t checkpoint_layers = 0;
  size_t tensors = 0, vision_skipped = 0, skipped_layer_tensors = 0, unconsumed = 0, subnormal_scales = 0;
  size_t read_per_token = 0;                 // derived: layers + final norm + head
  double seconds = 0, lm_head_quant_seconds = 0;
  std::string line;                          // the load line (also printed)
  size_t total_bytes() const;
};

// True when `snapshot_dir`'s config.json says model_type "muse_glimmer" (the CLIs' dispatch, 23c).
bool is_muse_checkpoint(const std::string& snapshot_dir);
// The descriptor load_muse_glimmer would build, from config.json alone (no shard): the CLI's planner needs it first.
model::MuseGlimmerDesc muse_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_override = 0);

// Loads the checkpoint (resolve_snapshot rules) onto `devices` (devices.size() == placement.devices; device 1 a view
// of device 0's context). Before a byte is allocated: the host half's checks (config.json, the quantisation, the
// names both ways, the data asserts), the placement over desc.layers, the plan (require_fits with `reserve`'s
// placeholders). After: zero unconsumed tensors, each part's bytes equal to muse_device_weight_bytes. One load line:
//   Muse-Glimmer-30B: 52 layers (int4 g64), 809 vision tensors skipped, head int8, devices 1, 17409096960 B
MuseWeights load_muse_glimmer(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                              uint32_t max_len, const model::MusePlacement& placement, bool int8_head = true,
                              uint32_t layers_override = 0, const runtime::muse::MuseReserve& reserve = {});

}  // namespace loader

#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/kolibri1_layout.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"   // DeviceWeight - reused, not re-declared
#include "model/kolibri1.h"

// Spec 20c: Kolibri-1 on one or two B70s - the checkpoint's tensors in the layouts
// src/kernels/kolibri/*.cl and the reused gemv / gemv_bf16 / gemv_i8w kernels read
// (loader/kolibri1_layout.h sizes every allocation; loader/kolibri1_repack.h is the host half).
// A loader BESIDE loader::load and load_k2 (spec 20 §4): their files are untouched; the public
// pieces are reused - resolve_snapshot, QuantConfig::parse, SafetensorsSet,
// assert_quant_invariants, LinearSrc::classify, the int8 lm_head quantiser.
//
// **Placement at load.** Each layer is repacked from the shards and uploaded straight onto the
// device the placement gives it (spec 16b's load-then-place, runtime::place_stages, would need
// the whole ~42 GB model on device 0 first - the case pipeline_place.h leaves to "that model's
// spec"). Device 0 also holds the embedding, the last device the final norm and lm_head, and
// every device with a sliding layer its own RoPE table.
namespace loader {

struct KolLayer {
  std::unique_ptr<DeviceWeight> qkv, oproj;        // int4 layout 0 (the desc's S) or bf16 tiled
  std::unique_ptr<l0::Mem> norms;                  // fp32 Kolibri1Desc::norm_floats(), plain w
  std::unique_ptr<DeviceWeight> router;            // bf16 tiled {hidden, router_n}, rows >= experts zero
  std::unique_ptr<l0::Mem> bias;                   // fp32 [router_n], 0 past the experts
  std::unique_ptr<l0::Mem> gate_up, down;          // int4 layout-1 blocks, block e at e x block
  std::unique_ptr<l0::Mem> shared_gate_up, shared_down;   // bf16 tiled {hidden, 2I}, {I, hidden}
};

struct KolDevicePart {
  uint32_t device = 0, first = 0, end = 0;         // layers [first, end) live here
  std::vector<KolLayer> layers;                    // index = layer - first
  std::unique_ptr<l0::Mem> embed;                  // device 0 only
  std::unique_ptr<l0::Mem> final_norm;             // last device only
  std::unique_ptr<DeviceWeight> lm_head;           // last device only
  std::unique_ptr<l0::Mem> rope;                   // any device holding a sliding layer
  size_t bytes = 0;                                // weights, == kol_device_weight_bytes (RoPE apart)
  size_t rope_bytes = 0;
  const KolLayer& layer(uint32_t l) const { return layers.at(l - first); }
};

struct KolLoadedModel {
  model::Kolibri1Desc desc;                        // layers = layers_limit when one was given
  model::KolPlacement placement;
  std::vector<KolDevicePart> parts;                // one per device
  uint32_t max_len = 0, trained_max_len = 0;
  bool int8_head = false;
  uint32_t checkpoint_layers = 0;                  // the checkpoint's own count (>= desc.layers)
  size_t unconsumed = 0, read_per_token = 0;       // read_per_token: derived, every device
  size_t subnormal_scales = 0;
  double seconds = 0, lm_head_quant_seconds = 0;
  size_t total_bytes() const;                      // every part's weights + RoPE
};

// True when `snapshot_dir`'s config.json says model_type "kolibri1" (the CLIs' dispatch).
bool is_kolibri1_checkpoint(const std::string& snapshot_dir);

// Loads the checkpoint (resolve_snapshot rules) onto `devices` (devices.size() ==
// placement.devices; device 1 a view of device 0's context). Before a tensor is read:
// config.json held to model::kolibri1() at its layer count, max_len within the trained 262144
// (else refused naming spec 20 decision 3), the quantisation invariants, the attention arm from
// the names (kol_attn_form), every name accounted for both ways. `layers_limit` > 0 loads only
// layers [0, layers_limit) (development mode, spec 20 §4: the real checkpoint truncated to fit one
// card) and makes desc.layers = layers_limit - the later layers' tensors are consumed by design and
// listed in one note line. The placement is over desc.layers (after the limit). After: zero
// unconsumed tensors, and each part's bytes equal to kol_device_weight_bytes. One report line per
// device.
KolLoadedModel load_kolibri1(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                             uint32_t max_len, const model::KolPlacement& placement,
                             LmHeadForm lm_head = LmHeadForm::Checkpoint, uint32_t layers_limit = 0);

// The descriptor load_kolibri1 would build, from config.json and the tensor names alone (no
// device): the CLI's planner needs it before the load (`--pp`, `--max-len auto`).
model::Kolibri1Desc kolibri1_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit = 0);

}  // namespace loader

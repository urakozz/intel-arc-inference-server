#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/pipeline_plan.h"

// Spec 16b (pipeline parallel): the loaded model split between the two devices.
//
// **Load, then place.** loader::load() loads the whole checkpoint onto device 0, as it
// always has; place_stages() then moves layers [s, layers), the final norm and lm_head to
// device 1 (allocate there, copy through a bounded host staging buffer, free on device 0)
// and leaves device 0 with the embedding and layers [0, s). The loader is untouched, every
// weight on either device is byte-for-byte what a single-card load makes (the copy moves
// bytes; nothing is re-quantised or re-packed), and placement needs no peer access. The
// cost is the load's peak: device 0 briefly holds the whole model - 18.1 GB on Qwen3.8
// with its bf16 head, under one card's 32.5 GB for every supported model (spec 16b as
// built). A model larger than one card needs the loader to place each layer directly
// (16d or that model's spec).
//
// The stage models are ordinary LoadedModels with three conventions build_stage() and
// PipelineEngine rely on:
//   - `linears` holds exactly the stage's layers (keyed by the model's layer index) and,
//     on device 1 only, lm_head;
//   - `layer_small` and `moe` hold the stage's layers in order: entry i is layer first + i;
//   - device 0 keeps `embed`; device 1 holds a 64-byte placeholder there (LoadedModel's
//     member is not optional; no stage list binds it) - or, with the MTP head (spec 16d), a
//     replica of the embedding, which the head's drafts and KV fill gather. Device 1 holds
//     the final norm, device 0 a 64-byte placeholder. Both hold the RoPE table (every FA
//     layer reads it);
//   - spec 16d: the MTP head (`mtp`) and a draft vocabulary (`draft_vocab`) are device 1's;
//     device 0's model holds neither.
namespace runtime {

// Device bytes of the weights a LOADED model holds, by where pipeline parallel puts them
// (PpWeights' fields) - exact for whatever the checkpoint shipped. The RoPE table is not a
// weight (the planner adds it per device at the session's max_len). Requires the whole
// model (one card's load).
PpWeights pp_weights(const loader::LoadedModel& m);

// Every device byte a model holds: weights, small tensors, the embedding, the final norm,
// the RoPE table, placeholders included. A stage model's memory_line() `model` term.
size_t model_device_bytes(const loader::LoadedModel& m);

// Splits `full` (loaded on `d0`) at `split`: returns {stage 0 on d0, stage 1 on d1}.
// Throws for a split outside [1, layers - 1]. `d1` must be a view of d0's context
// (l0::Context's view constructor). Spec 16d: the MTP head and a draft vocabulary move to
// device 1 with lm_head, and the embedding is replicated there.
std::vector<loader::LoadedModel> place_stages(l0::Context& d0, l0::Context& d1,
                                              loader::LoadedModel full, uint32_t split);

}  // namespace runtime

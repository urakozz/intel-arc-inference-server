#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_layout.h"
#include "loader/loader.h"   // DeviceWeight, LoadedModel - reused
#include "model/dflash.h"

// Spec 19b Task 1: a DFlash / DFlash 2 drafter on the card BESIDE a loaded target (spec 19 §5 "Memory"): its own
// linears, norms, conv bases and codebooks; the target's embedding and lm_head are SHARED - read through the target's
// LoadedModel at capture, never copied (Review Focus 3). Only the mask row is the drafter's: the target's embedding
// row `mask_id` (copied once at load), or the checkpoint's mask_embedding.safetensors (Review Focus 4).
//
// The layouts are loader/dflash_layout.h's (the planner's formula); load_drafter asserts its allocations add up to
// loader::dflash_weight_bytes exactly.
namespace loader {

struct DFlashLayerWeights {
  DeviceWeight qkv, o, gate_up, down;                  // K / N: hidden -> qkv_n, q_n -> hidden, ... (layout.h)
  std::unique_ptr<DeviceWeight> attn_kproj, mlp_kproj; // DFlash 2 only: hidden -> kproj_n
  l0::Mem norms;                                       // fp32 [input_layernorm | post_attention_layernorm][hidden]
};

struct DFlashWeights {
  model::DFlashDesc desc;
  DraftFormat format = DraftFormat::Int8;
  uint32_t target_hidden = 0;
  std::unique_ptr<DeviceWeight> fc;                  // taps x target hidden -> hidden
  std::unique_ptr<l0::Mem> hidden_norm, final_norm;  // fp32 [hidden] each
  std::vector<DFlashLayerWeights> layers;
  std::unique_ptr<l0::Mem> qk_norm;                  // fp32 [layers][q | k][head_dim]
  std::unique_ptr<l0::Mem> conv_base;                // DFlash 2: fp32 [layers][attn, mlp][side][tap][hidden]
  std::unique_ptr<DeviceWeight> sel_proj;            // DFlash 2: hidden -> sel_rank
  std::unique_ptr<l0::Mem> pred, succ;               // DFlash 2: bf16 [vocab][sel_rank] each
  std::unique_ptr<l0::Mem> mask_row;                 // fp32 [hidden]
  bool mask_from_checkpoint = false;
  size_t bytes = 0;                                  // == dflash_weight_bytes(desc, target_hidden, format).total()
  double seconds = 0, quant_seconds = 0;
  std::string line;                                  // the load line (also printed)
};

// Loads the drafter at `snapshot_or_repo` (resolve_snapshot rules) onto `ctx`'s device beside `target` (loaded on
// the same device). Before a byte is allocated: the host half's checks (config, names, dtypes, shapes,
// dflash_repack.h), the drafter held to the target (model::check_dflash_target + fc's K, against the target's
// lm_head rows). One load line:
//   DFlash drafter: DFlash2DraftModel, 5 layers, taps 5/19/33/47/61, block 8, int8 (quantised at load 3.1 s),
//   mask row from the target's embedding, 2015363072 B, 4.2 s
DFlashWeights load_drafter(l0::Context& ctx, const std::string& snapshot_or_repo, const LoadedModel& target,
                           DraftFormat format, unsigned threads = 0);
// The same against a target given by its parts: its descriptor, its lm_head rows and its embedding (bf16
// [rows][hidden] on this device) - what the overload above reads off a LoadedModel; a device test passes a synthetic
// target (tests/runtime/dflash_draft_test.cc).
DFlashWeights load_drafter(l0::Context& ctx, const std::string& snapshot_or_repo, const model::ModelDesc& target,
                           uint32_t target_vocab_rows, const l0::Mem& target_embed, DraftFormat format,
                           unsigned threads = 0);
// Spec 23f: the same against a target given by its shape alone (Muse-Glimmer's MuseGlimmerDesc is not a ModelDesc): its
// name, layers, hidden and head rows. A Muse-form drafter's vocab (0 in its config) becomes the target's head rows;
// DraftFormat::Int4G64 quantises every linear at load (loader/rtn.h) into gemv.cl's layout 1.
DFlashWeights load_drafter(l0::Context& ctx, const std::string& snapshot_or_repo, const model::DFlashTargetShape& target,
                           const l0::Mem& target_embed, DraftFormat format, unsigned threads = 0);

}  // namespace loader

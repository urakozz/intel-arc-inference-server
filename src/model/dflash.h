#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/json.h"
#include "model/model_desc.h"

// Spec 19b: a DFlash / DFlash 2 drafter (spec 19 §2) as data - the checkpoint's config.json held to the semantics
// tools/oracle/dflash_ref.py ports from vLLM (`config_from_dict`, `resolve_layer_attention`), and the drafter held
// to the target it drafts for. Host only (no device, no file I/O beyond what the caller parsed).
//
// One descriptor covers both versions, switched by the checkpoint's config (Review Focus 5): DFlash 2
// (`DFlash2DraftModel`, or a dflash_config naming `selector_rank` / `conv_kernel_size`) adds the two grouped
// convolutions per layer and the candidate selector; DFlash (v1, `DFlashDraftModel`) has neither and drafts each
// mask row from its own logits.
//
// **Refused by name** (each throw names the key): a quantization_config (our formats only: the drafter's own bf16,
// quantised by us at load), an activation other than silu, a rope_type other than default or partial rotary,
// `sample_from_anchor` (not DFlash), `use_aux_hidden_state` false (no fc), target_layer_ids not strictly increasing,
// a draft vocabulary of the drafter's own (`draft_vocab_size` != vocab_size: d2t - undefined for DFlash 2 upstream,
// not built for v1), attention biases, an attention value scale or sink bias (no checkpoint we serve has one), a
// conv group that does not divide the hidden size, a block below 2 rows.
namespace model {

// Spec 23f: whose semantics a drafter follows. Vllm: spec 19's (DFlash / DFlash 2 from vLLM's qwen3_dflash*.py - every
// field below). Muse: transformers 5.19.0's MuseGlimmerAssistantModel + DFlashTokenCandidateGenerator
// (meta-models/Muse-Glimmer-30B-assistant; tools/oracle/dflash_ref.py's "muse" form, bitwise transformers): a DFlash
// v1 shape whose block is always `block` rows (row 0 the anchor, rows 1..block-1 the mask; K drafts are rows 1..K), the
// window INCLUDES |q - k| == window (vLLM's is <= window - 1; spec 23 decision 13), the RMSNorm rounds x-hat to bf16
// before x w (two roundings), the embedding is the target's RAW row and the drafts the RAW head rows (no multiplier,
// no softcap). Its config.json carries no vocab_size: `vocab` is 0 until the target sets it (check_dflash_target).
enum class DFlashForm : uint32_t { Vllm = 0, Muse = 1 };

struct DFlashDesc {
  DFlashForm form = DFlashForm::Vllm;
  bool window_inclusive = false;        // Muse: |q - k| <= window; Vllm: <= window - 1
  bool norm_two_roundings = false;      // Muse: w x bf16(x-hat); Vllm: bf16(x-hat x w)
  uint32_t version = 2;                 // 1 = DFlash, 2 = DFlash 2
  uint32_t layers = 0, hidden = 0, q_heads = 0, kv_heads = 0, head_dim = 0, inter = 0;
  uint32_t vocab = 0;                   // config.json vocab_size: the target's embedding / head rows
  uint32_t block = 0;                   // dflash_config.block_size: 1 anchor + (block - 1) mask rows
  uint32_t mask_id = 0;                 // dflash_config.mask_token_id
  uint32_t num_target_layers = 0;       // config.json num_target_layers (0 = not declared)
  std::vector<uint32_t> taps;           // dflash_config.target_layer_ids: the residual AFTER each target layer
  std::vector<uint32_t> window;         // per layer: the sliding window, 0 = full attention
  std::vector<bool> causal;             // per layer (`is_causal`, then dflash_config.causal, then layer types)
  float eps = 1e-6f;
  double theta = 1e6;                   // rope_parameters.rope_theta (default rope, full rotary)
  bool neox = true;
  // DFlash 2 only (0 / 1.0 / 0 on v1):
  uint32_t conv_group = 0, conv_taps = 0, sel_rank = 0, sel_top_k = 0;
  float in_scale = 1.0f;                // input_embedding_scale
  float out_mult = 1.0f;                // v2: output_multiplier; v1: logit_scale (the head's scale)
  float softcap = 0.0f;                 // v2: final_logit_softcapping (0 = none)
  std::string architecture;

  uint32_t q_n() const { return q_heads * head_dim; }
  uint32_t kv_n() const { return kv_heads * head_dim; }
  uint32_t qkv_n() const { return q_n() + 2 * kv_n(); }      // q || k || v, the block's projection
  uint32_t gate_up_n() const { return 2 * inter; }           // gate || up, concatenated
  uint32_t groups() const { return conv_group ? hidden / conv_group : 0; }
  // kernel_projection's rows: [2 sides (prepare, finish)][conv_taps][groups].
  uint32_t kproj_n() const { return 2 * conv_taps * groups(); }
  uint32_t max_k() const { return block - 1; }
  uint32_t fc_k(uint32_t target_hidden) const { return uint32_t(taps.size()) * target_hidden; }
  bool v2() const { return version == 2; }
  bool all_sliding() const;             // every layer has a window (a ring suffices)
  uint32_t max_window() const;          // the largest window (0 when none is sliding)
  bool muse() const { return form == DFlashForm::Muse; }
  // The context positions a block row reads back from the anchor: window (Muse: row 0 sees p - window .. p - 1) or
  // window - 1 (Vllm).
  uint32_t context_reach(uint32_t layer) const { return window.at(layer) - (window_inclusive ? 0 : 1); }
};

// The target a drafter is held to, by its parts (spec 23f: Muse-Glimmer's MuseGlimmerDesc is not a ModelDesc).
struct DFlashTargetShape {
  std::string name;
  uint32_t layers = 0, hidden = 0;
  uint32_t vocab_rows = 0;              // its lm_head's (and embedding's) rows
};

// config.json -> the descriptor (refusals above). `architectures[0]` must be DFlashDraftModel or DFlash2DraftModel
// - or, spec 23f, MuseGlimmerAssistantModel (`model_type: muse_glimmer_assistant`): block_size, mask_token_id and
// target_layer_ids at the top level (MuseGlimmerAssistantConfig, AC:62-89), every layer sliding (a full layer is
// refused), non-causal, RoPE default at rope_parameters.rope_theta, rms_norm_eps; form Muse.
DFlashDesc dflash_desc_from_config(const common::json::Value& config_json);
// True for those three architectures (a CLI's dispatch reads only this).
bool is_dflash_architecture(const std::string& architecture);
inline constexpr const char* kMuseAssistantArchitecture = "MuseGlimmerAssistantModel";

// The drafter held to its target (spec 19 §2, plan 19b Task 1): it shares the target's embedding and lm_head, so
// its hidden size must be the target's and its vocab_size the target's head rows (`target_vocab_rows`); every tap
// layer must exist (target_layer_ids[i] < target.layers - the residual after the last layer is the pre-final-norm
// hidden, a legal tap); a declared num_target_layers must equal target.layers; the mask id must be a row of the shared
// embedding (< target_vocab_rows; Ornith's drafter masks with a padded row). Throws naming the key.
void check_dflash_target(const DFlashDesc& d, const ModelDesc& target, uint32_t target_vocab_rows);
// The same rules against a target given by its parts (spec 23f: Muse-Glimmer). A Muse-form drafter's vocab 0 means
// "the target's" (its config has no vocab_size); every other rule as above.
void check_dflash_target(const DFlashDesc& d, const DFlashTargetShape& target);
// Spec 23f: the drafter is the one the Muse kernels are built for (kernels::dflash::kMuse) - before the device, each
// refusal naming the key: the Muse form, 5 layers, hidden 6656, 32 / 8 heads x 128, intermediate 19968, 5 taps, a
// block other than 16, a window other than 2048 (inclusive), rms_norm_eps other than 1e-5.
void check_muse_assistant(const DFlashDesc& d);

}  // namespace model

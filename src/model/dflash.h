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
// **Refused by name** (each throw names the key): a quantization_config other than an AutoRound int4 export's
// (quant_method auto-round or gptq; its arithmetic held by loader::DFlashCheckpoint), an activation other than silu,
// a rope_type other than default or partial rotary, `sample_from_anchor` (not DFlash), `use_aux_hidden_state` false (no fc), target_layer_ids not strictly increasing,
// a draft vocabulary of the drafter's own (`draft_vocab_size` != vocab_size: d2t - undefined for DFlash 2 upstream,
// not built for v1), attention biases, an attention value scale or sink bias (no checkpoint we serve has one), a conv
// group that does not divide the hidden size, a block below 2 rows.
namespace model {

// Whose semantics a drafter follows. Vllm: spec 19's (DFlash / DFlash 2 from vLLM's qwen3_dflash*.py - every field
// below). (1 was spec 23f's Muse form, dropped with Muse-Glimmer on 2026-10-27; 2 another dropped model's form.)
//
// Kolibri DSpark: DSpark (`DSparkDraftModel`, a vllm-project/speculators 0.8 checkpoint - intuit/Kolibri-1-dspark beside
// Kolibri-1; vLLM's qwen3_dspark.py + spec_decode/dspark/speculator.py, its config through transformers_utils/configs/
// speculators/algos.py update_dspark): the DFlash v1 backbone (no convolutions) with three differences -
//   * anchor-as-first-prediction (`sample_from_anchor`): block row r at p + r predicts position p + r + 1, so K drafts are
//     rows 0 .. K - 1 (DFlash's mask rows sit AT the position they predict, rows 1 .. K);
//   * causal attention inside the block (its sliding layers, vLLM's _dflash_layer_causal with no dflash_config.causal);
//   * the Markov head: row r's logits + markov_w2[c] . markov_w1[prev] (prev the anchor, then row r - 1's pick), drafted
//     left to right - DFlash 2's selector with the hidden projection taken as ones (pred = markov_w1, succ = markov_w2,
//     sel_rank = markov_rank), over the head's top 16 per row (vLLM's dspark_draft_topk form at 16).
// The config is the speculators form: the layer config under `transformer_layer_config`, the taps as
// aux_hidden_state_layer_ids (each the INPUT of that target layer: taps = ids - 1, update_dspark's rule), block_size,
// mask_token_id and the DSpark keys at the top level. Its confidence head (a per-position acceptance estimate) is read
// for its shape only: the engine's AdaptiveK picks K from the observed acceptance.
enum class DFlashForm : uint32_t { Vllm = 0, DSpark = 3 };

struct DFlashDesc {
  DFlashForm form = DFlashForm::Vllm;
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
  // Kolibri DSpark only: the confidence head's input width (hidden, + markov_rank with confidence_head_with_markov); 0 =
  // none shipped. The Markov head's rank is sel_rank (the codebooks pred / succ are markov_w1 / markov_w2).
  uint32_t conf_in = 0;

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
  bool dspark() const { return form == DFlashForm::DSpark; }
  // The context positions a block row reads back from the anchor: window - 1 (the vLLM rule |q - k| <= window - 1).
  uint32_t context_reach(uint32_t layer) const { return window.at(layer) - 1; }
};

// The target a drafter is held to, by its parts (Kolibri DSpark: Kolibri1Desc is not a ModelDesc).
struct DFlashTargetShape {
  std::string name;
  uint32_t layers = 0, hidden = 0;
  uint32_t vocab_rows = 0;              // its lm_head's (and embedding's) rows
};

// config.json -> the descriptor (refusals above). `architectures[0]` must be DFlashDraftModel or DFlash2DraftModel
// - or DSparkDraftModel.
// Kolibri DSpark: DSparkDraftModel (the speculators form above), refused by name besides 19b's rules: no
// transformer_layer_config, sample_from_anchor not true (absent is refused too: speculators' default is true, vLLM's
// update_dspark reads false), a markov_head_type other than vanilla, markov_rank 0, a quantization_config (bf16 first),
// sliding_window_non_causal true, tie_word_embeddings, aux_hidden_state_layer_ids holding 0, and a target_hidden_size that
// is not the drafter's hidden size.
DFlashDesc dflash_desc_from_config(const common::json::Value& config_json);
// True for those three architectures (a CLI's dispatch reads only this).
bool is_dflash_architecture(const std::string& architecture);
inline constexpr const char* kDSparkArchitecture = "DSparkDraftModel";

// The drafter held to its target (spec 19 §2, plan 19b Task 1): it shares the target's embedding and lm_head, so
// its hidden size must be the target's and its vocab_size the target's head rows (`target_vocab_rows`); every tap
// layer must exist (target_layer_ids[i] < target.layers - the residual after the last layer is the pre-final-norm
// hidden, a legal tap); a declared num_target_layers must equal target.layers; the mask id must be a row of the shared
// embedding (< target_vocab_rows; Ornith's drafter masks with a padded row). Throws naming the key.
void check_dflash_target(const DFlashDesc& d, const ModelDesc& target, uint32_t target_vocab_rows);
// The same rules against a target given by its parts (Kolibri DSpark: Kolibri-1).
void check_dflash_target(const DFlashDesc& d, const DFlashTargetShape& target);
// Kolibri DSpark: the drafter is the one the DSpark kernels are built for (kernels::dflash::kDSpark) - before the device,
// each refusal naming the key: the DSpark form, hidden 2560, 20 / 4 heads x 128, intermediate 6144, every layer sliding
// at 2048 (|q - k| <= 2047) and causal, block 8, 5 taps, Markov rank 256, rms_norm_eps 1e-6, vocabulary 128000.
void check_dspark(const DFlashDesc& d);
// Spec 19 v3c: one `--parallel tp` card's half of a drafter (runtime::dflash::TpDFlashDrafter) - half the q and kv heads
// and half the MLP; every other field the whole drafter's (its hidden, layers, taps, window, convolutions, selector,
// vocabulary). Throws unless the heads and the MLP halve into whole 16-row tiles.
DFlashDesc dflash_tp_card_desc(const DFlashDesc& d);

}  // namespace model

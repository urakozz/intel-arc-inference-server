#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/draft_vocab.h"
#include "loader/lm_head_int8.h"
#include "loader/quant.h"
#include "loader/small_layout.h"
#include "loader/snapshot.h"
#include "model/model_desc.h"
#include "model/qwen35.h"

namespace loader {

// `LoadedModel::linears` is keyed by (layer, id); tensors that belong to no
// layer (lm_head) use this sentinel instead of a second map.
constexpr uint32_t kTopLevel = 65535;

struct DeviceWeight {
  l0::Mem mem;                 // the canonical bytes on device
  // Layout 0's real, independent f16 [K/64][N] allocation. Null for layout 1
  // (scales are inline in mem) and for bf16 weights. For the int8 `lm_head`
  // (spec 9) it is the fp32 [N] per-row scale allocation.
  std::unique_ptr<l0::Mem> scales;
  model::GemvShape shape;      // K, N, S, layout as the kernel variant needs
  model::WeightKind kind;
};

struct SmallTensors {          // per layer: everything that is not a GEMV weight
  l0::Mem norms;               // input_ln(1+w) ‖ post_ln(1+w) fp32 [2][hidden]
  l0::Mem gdn;                 // the layer-kind block: GDN = conv fp32
                               //   [conv dim][4] ‖ negA fp32 [v-heads] ‖ dt_bias fp32
                               //   [v-heads] ‖ gated-norm plain w bf16 [128] (Qwen3.8:
                               //   conv dim 10240, 48 v-heads);
                               //   FA = q_norm ‖ k_norm (1+w) fp32 [2][256].
                               // Offsets: loader/small_layout.h (one place).
};

// Spec 8 §3.1: the checkpoint's multi-token-prediction head, loaded only when asked
// (`load(..., mtp = true)`). All bf16, in gemv_bf16's tiled layout
// (common::repack_bf16_tiled), [N][K] logically, S = 1, layout 0 (a filler for bf16).
// It shares `embed` and the `lm_head` linear with the main model.
struct MtpHead {
  // Shapes from the descriptor (spec 15b); Qwen3.8's and Agnes's in the comments.
  DeviceWeight fc;        // K 2 x hidden 10240 -> N 5120, over cat(pre_fc_norm_embedding(e), pre_fc_norm_hidden(h))
  DeviceWeight qkv;       // K 5120 -> N fa_qkv_n 14336: q_proj (q||gate per head) || k_proj || v_proj
  DeviceWeight o;         // K fa_value_dim 6144 -> N 5120
  DeviceWeight gate_up;   // K 5120 -> N 2 x mtp_intermediate 34816, gate/up in alternating 16-column blocks
  DeviceWeight down;      // K mtp_intermediate 17408 -> N 5120
  l0::Mem norms;          // fp32 (1 + w) [5][hidden], mtp_norm_off below
  l0::Mem fa;             // q_norm || k_norm fp32 (1 + w), the FA block (small_layout.h)
};
// The five RMSNorms inside MtpHead::norms, in row order; row i sits at byte
// i x hidden x 4 (spec 15b: hidden is the descriptor's, 5120 on Qwen3.8).
enum MtpNorm : uint32_t {
  kMtpNormPreE,    // mtp.pre_fc_norm_embedding
  kMtpNormPreH,    // mtp.pre_fc_norm_hidden
  kMtpNormInput,   // mtp.layers.0.input_layernorm
  kMtpNormPost,    // mtp.layers.0.post_attention_layernorm
  kMtpNormFinal,   // mtp.norm
  kMtpNormCount
};
inline size_t mtp_norm_off(const model::ModelDesc& d, MtpNorm n) {
  return size_t(n) * d.hidden * 4;
}
inline size_t mtp_norms_bytes(const model::ModelDesc& d) {
  return size_t(kMtpNormCount) * d.hidden * 4;
}
// A dense MTP head's tensor count; its bf16 bytes are ModelDesc::mtp_checkpoint_bytes()
// (849,398,784 on Qwen3.8 and Agnes, docs/03: 0.849 GB).
constexpr size_t kMtpTensors = 15;

// Spec 8 §11: the MTP draft's reduced vocabulary V' (`b70-serve --draft-vocab`), built
// only when asked (`load(..., draft_vocab.size > 0)`, which needs the MTP head and an int8
// or bf16 lm_head). The head's rows ids[0..|V'|) - and for int8 their fp32 scales -
// gathered at load from the host copy the loader just made of the head (the int8
// quantisation's, or the bf16 tiling's) into the same tiled layout at N = |V'|
// (loader::gather_int8_tiled_rows / gather_bf16_tiled_rows), and the id table the draft's
// argmax maps its compact index through. The verify list never reads any of it.
//
// **The compact head is the head's own form**, a byte-exact subset of what the verify
// list reads, so the compact GEMV's column j is the full GEMV's column ids[j], bitwise
// (draft_vocab_kernels_test, both forms): gemv_i8w or gemv_bf16 at N = |V'|. An int4
// (`--quant_lm_head`) checkpoint head is refused - no int4 gather and no binaries for it.
struct DraftVocab {
  // int8: [|V'|/16][K/16][16][16] + fp32 scales [|V'|]; bf16: [|V'|/16][K/8][8][16], no
  // scales (2x the int8 bytes). Kind Int8 or Bf16, shape {K = hidden, N = |V'|, S 1, layout 0}.
  DeviceWeight head;
  l0::Mem ids;                    // u32 [|V'|], ascending - the compact index -> id table
  std::vector<uint32_t> host_ids; // the same table on the host
  uint32_t size() const { return uint32_t(host_ids.size()); }
  size_t bytes() const {
    return head.mem.size() + (head.scales ? head.scales->size() : 0) + ids.size();
  }
};

struct LoadReport {            // printed by load(); asserted by the checkpoint test
  size_t int4_bytes = 0, scale_bytes = 0, bf16_linear_bytes = 0;
  size_t embed_bytes = 0, lm_head_bytes = 0, pad_bytes = 0;
  // Every non-GEMV byte allocated, the RoPE table included. The printed
  // `small` line nets rope out (it is resident but not per-token traffic) and
  // reports it on its own line; total() and this field do not.
  size_t small_bytes = 0;
  // The RoPE table's share of small_bytes (model::Qwen35::rope_table_bytes(max_len)):
  // `total() - rope_bytes` is what the memory planner calls the model's bytes, the
  // part that does not scale with max_len (spec 6 §10).
  size_t rope_bytes = 0;
  // Spec 8: the MTP head's device bytes (0 unless loaded) - in total(), not in
  // read_per_token (the main step never reads it).
  size_t mtp_bytes = 0;
  size_t mtp_checkpoint_bytes = 0;  // what those came from: desc.mtp_checkpoint_bytes() when loaded
  size_t mtp_tensors = 0;           // mtp.* tensors consumed (kMtpTensors when loaded)
  size_t total() const;             // the eight byte fields above
  // **`W`, as this load actually measured it** - the bytes a decode step
  // streams: everything in total() except `embed_tokens` (gathered one row per
  // token) and the RoPE table (~256 B per token). It is a FIELD rather than a
  // constant because it is no longer one number: the published checkpoint's
  // bf16 `lm_head` makes it 15.540 GB and a packed one makes it 13.673 GB
  // (both measured 2026-08-26). Anything that divides by W - the MBU line the
  // bench prints, the roofline - has to read it from here, or it reports one
  // checkpoint's efficiency against another's denominator.
  size_t read_per_token = 0;
  size_t unconsumed = 0;            // checkpoint tensors nothing loaded (must be 0)
  // Spec 9: the host quantisation of an int8 `lm_head` (0 unless LmHeadForm::Int8).
  double lm_head_quant_seconds = 0;
  // Spec 8 §11: the draft vocabulary's device bytes (DraftVocab::bytes(), 0 when off) and
  // where its ids came from. NOT in total(): those eight fields are the model as loaded
  // before §11, and Engine::memory_line() prints this as its own term.
  size_t draft_vocab_bytes = 0;
  DraftVocabCounts draft_vocab_counts;
  double draft_vocab_seconds = 0;
  double seconds = 0;
};

struct LoadedModel {
  std::map<std::pair<uint32_t, model::LinearId>, DeviceWeight> linears;  // layer kTopLevel = lm_head
  std::vector<SmallTensors> layer_small;   // [desc->layers]
  l0::Mem embed;                            // bf16 [248320][hidden] row-major (gathered)
  l0::Mem final_norm;                       // pre-lm_head RMSNorm, (1+w) fp32 [hidden]
  l0::Mem rope;                             // fp32 [max_len][2][32] cos/sin pairs
  LoadReport report;
  uint32_t max_len = 0;                     // the RoPE table's length (load, set_max_len)
  // config.json's max_position_embeddings (loader/trained_context.h); 0 = not declared.
  uint32_t trained_max_len = 0;
  std::unique_ptr<MtpHead> mtp;             // null unless load(..., mtp = true) (spec 8)
  // Spec 14 §3.1: the model this checkpoint is, chosen from config.json's
  // architectures[0] (model::desc_for_architecture). A process-lifetime singleton.
  const model::ModelDesc* desc = nullptr;
  // Spec 8 §11: null unless a draft vocabulary was asked for. Last, so load()'s
  // aggregate initialiser of the members above is unchanged.
  std::unique_ptr<DraftVocab> draft_vocab;
};

// Loads the qwen3_5 checkpoint at `snapshot_or_repo` (resolve_snapshot rules)
// into ctx's device. Skips model.visual.* and (v1) mtp.*. Strips the
// "model.language_model." prefix so model::Qwen35's layer-relative names bind.
// Asserts quant invariants and the doc-03 shape table; throws by name on any
// mismatch. max_len sizes the RoPE table only (default 16384); it must be at least 1
// and at most config.json's max_position_embeddings (spec 6 §10: positions the model
// was never trained on are refused here, before a byte is read). Whether the KV cache
// at that max_len FITS is not the loader's question - runtime/memory_plan.h answers it
// from the loaded bytes (spec 14 §3.3's fixed Agnes ceiling is gone).
// `mtp` (spec 8 §3.1): also load the MTP head into LoadedModel::mtp. Without it the
// head's tensors are skipped by name, counted, and never read - exactly as before.
// `lm_head` (spec 9 §3): Checkpoint loads the head as shipped; Int8 quantises a bf16
// head to int8 rows + fp32 row scales on the host (loader/lm_head_int8.h) and throws
// if the checkpoint's head is not bf16.
// `draft_vocab` (spec 8 §11): size > 0 builds LoadedModel::draft_vocab - V' chosen by
// loader::select_draft_vocab over the spec's lists at the model's vocab_used, gathered
// from the head in its loaded form (int8 or bf16). Throws before a byte is read unless
// `mtp` is on, the head is not the checkpoint's int4, and the size is one the kernels
// are compiled for (kDraftVocabSizes).
LoadedModel load(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len = 16384,
                 bool mtp = false, LmHeadForm lm_head = LmHeadForm::Checkpoint,
                 const DraftVocabSpec& draft_vocab = DraftVocabSpec{});

// Spec 6 §10 (`--max-len auto`): the planner needs the loaded byte count, and load()
// needs a max_len. So auto loads at a small max_len, plans, and then re-tables the
// model here: a RoPE table of `max_len` positions is built and uploaded, the old one is
// freed, and m.max_len / m.report (small_bytes, rope_bytes) say so - exactly what
// load(..., max_len) would have produced. Same bound as load(): 1..trained_max_len.
// **Only before anything has captured `m.rope.ptr()`** - before the Engine is built:
// a captured list bakes the old pointer. Nothing reads the table's length from the
// allocation; every kernel indexes it by position (capture.cc, prefill/step.cc).
void set_max_len(l0::Context& ctx, LoadedModel& m, uint32_t max_len);

}  // namespace loader

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/memory_plan.h"

// Spec 16b (pipeline parallel decode across two B70s): the split and the per-device
// memory plan, as device-free arithmetic - the planner half of spec 16, beside
// runtime/memory_plan.h and over the same size functions (runtime/buffer_sizes.h), so
// tests/runtime/pipeline_plan_test.cc runs on any host.
//
// **The split comes from the descriptor.** Layers [0, s) run on device 0 with the
// embedding; layers [s, layers) on device 1 with the final norm, lm_head and the argmax.
// Which layers are GDN and which FA, and every byte, are ModelDesc's - so Qwen3.8 (64
// layers), Agnes (72) and Ornith (40) split without code changes (spec 16b Review Focus 5).
//
// **`--pipeline-split auto` balances BYTES, not layers** (spec 16 decision 2, and vLLM's
// PP = 2 lesson on these cards: an even layer split leaves the lm_head card heavier, and
// the smaller remainder sets the KV both cards can hold). The bytes are everything the
// device will hold at the session's max_len: its weights, the RoPE table, its layers' KV
// and GDN state, the decode scratch, the hand-off buffers.
namespace runtime {

// --- the hand-off (spec 16 §2 decision 3; 16a would have chosen, 16b builds both) --------

// copy: device 0's list ends with a device-to-device copy of the residual into device 1's
//       landing buffer and a barrier signalling a cross-device event; device 1's list
//       starts by waiting on it and copying the landing buffer into its own scratch.
// peer: device 0's last kernel (pp_send) writes the residual into device 1's landing buffer
//       and raises a flag with a system-scope release store; device 1's first kernel
//       (pp_recv) spins on it with system-scope acquire loads, bounded.
enum class PpHandoff { Copy, Peer };
inline constexpr PpHandoff kDefaultPpHandoff = PpHandoff::Copy;
const char* pp_handoff_name(PpHandoff h);
bool parse_pp_handoff(const std::string& v, PpHandoff& out);   // "copy" | "peer"

// --- the stages --------------------------------------------------------------------------

inline constexpr uint32_t kPpDevices = 2;   // spec 16 builds and gates two (§7)

// The layers one device runs, and where its GDN / FA layers sit in the model's per-kind
// order (the index the persistent state is sliced by: GDN 0..gdn_layers-1, FA
// 0..fa_layers-1). A stage's state holds exactly its own layers, kind-major as the
// single-card buffers are, so stage 0's slices followed by stage 1's ARE the single-card
// layout - which is what makes save_state / save_kv identical across --pp 1 and 2.
struct PpStage {
  uint32_t first = 0, last = 0;      // layers [first, last)
  uint32_t gdn_first = 0, gdn = 0;   // its GDN layers: [gdn_first, gdn_first + gdn)
  uint32_t fa_first = 0, fa = 0;     // its FA layers:  [fa_first, fa_first + fa)
  uint32_t layers() const { return last - first; }
  bool has_embed() const { return first == 0; }
  bool has_head(const model::ModelDesc& d) const { return last == d.layers; }
};
PpStage pp_stage(const model::ModelDesc& d, uint32_t first, uint32_t last);
// Throws std::invalid_argument unless 1 <= split <= layers - 1 and each device holds at least
// one GDN and one FA layer (its state is one allocation of each kind, sliced per layer).
void require_split(const model::ModelDesc& d, uint32_t split);
std::array<PpStage, kPpDevices> pp_stages(const model::ModelDesc& d, uint32_t split);

// The compute launches of a stage's decode list (the hand-off's own launches - pp_send /
// pp_recv under `peer` - excluded): stage 0 is embed_gather, its layers, and layer s's
// prep_res_fold (the cut: the fold needs no weight, so device 0 runs it and hands off the
// folded residual and its sums - 10.9 KB on Qwen3.8 instead of 90 KB of split-K partials);
// stage 1 is its layers less that fold, and the 5 head launches. The two add up to
// runtime::decode_launches(d) (capture.cc asserts it at every stage capture).
size_t pp_stage_launches(const model::ModelDesc& d, const PpStage& st);

// --- the bytes ---------------------------------------------------------------------------

// The loaded weights' device bytes by where they go.
struct PpWeights {
  std::vector<size_t> layer;   // per layer: its linears, small blocks, MoE blocks
  size_t embed = 0;            // device 0: embed_tokens
  size_t head = 0;             // the last device: lm_head and the final norm
  size_t total() const;
  // A stage's share: its layers, and the embedding / head when it holds them.
  size_t stage(const model::ModelDesc& d, const PpStage& st) const;
};
// From the descriptor, in the device forms loader::load allocates (int4 layout 0: GPTQ
// words + f16 scales; layout 1: 136-word tiles with inline scales; bf16 a||b padded to
// 128 columns; the small blocks of loader/small_layout.h; a MoE layer's blocks of
// loader/moe_layout.h; lm_head in `lm_head`'s form). Qwen3.8 with its bf16 head sums to
// the measured load, 18,082,777,088 B. The CLI plans from the LOADED model's allocations
// instead (runtime::pp_weights(const loader::LoadedModel&), pipeline_place.h) - exact for
// whatever the checkpoint shipped.
PpWeights pp_weights(const model::ModelDesc& d, model::WeightKind lm_head);

// The hand-off buffers (runtime::PipelineLink). One landing allocation on device 1 that only
// device 0 writes: the residual rows (kM x hidden bf16), then on their own page the norm
// sums (norm_sumsq's size) and the peer path's stamp word, then the flag on a page of its
// own (spec 16 §2: the peer-written buffer is its own page-aligned allocation, nothing the
// local device writes in it). Device 1 also keeps pp_recv's state words, device 0 its
// send counter.
inline constexpr size_t kPpPage = 4096;
inline constexpr size_t kPpLandingAlign = 65536;
inline constexpr size_t kPpStateWords = 16;   // pp_recv's state[] (pp_handoff.cl) and spare
struct PpLandingLayout {
  size_t resid_bytes = 0;   // kM rows x hidden bf16 - the allocation; a step moves M rows
  size_t sumsq_bytes = 0;   // DecodeScratch::norm_sumsq's size
  size_t sumsq_off = 0;     // page-aligned, after the rows
  size_t stamp_off = 0;     // the word right after the sums
  size_t flag_off = 0;      // a page of its own
  size_t total = 0;
};
PpLandingLayout pp_landing_layout(const model::ModelDesc& d);
// Spec 20c: the same layout from the two regions' sizes alone - the form a model without a
// ModelDesc (Kolibri-1, its own engine) builds its hand-off with. The ModelDesc form above is
// this one at (kM rows x hidden bf16, norm_sumsq's size), byte for byte.
PpLandingLayout pp_landing_layout(size_t resid_bytes, size_t sumsq_bytes);
// Device d's hand-off bytes: device 0 its send counter, device 1 the landing + state words.
size_t pp_link_bytes(const model::ModelDesc& d, uint32_t device);

// One device's plan, in memory_line()'s five components: model = its weights + the RoPE
// table (both devices hold one: every FA layer's attn_prep reads it); kv = its FA layers'
// K and V; decode_state = control + its GDN state + conv ring + the decode scratch (each
// device has the whole DecodeScratch: the lists are per device and scratch is per list) +
// the hand-off buffers. No MTP (16d).
//
// Spec 16c, with a prefill path (`pf.prefill`): each device also holds a whole
// PrefillScratch (eager, and the l0 backend's slab) in prefill_scratch, its own Int8State
// on l0-int8 (the scratch for the model's widest h8 K, and the column scales of ITS layers'
// linears only) in int8, and the prefill hand-off (runtime/pipeline_prefill_plan.h:
// pp_prefill_link_bytes - device 1's two landing slots) in decode_state, as `prefill_link`.
// The block-end shadows a prefix cache's hook needs are not planned (b70-serve's pipeline is
// spec 16d). The composed attention and sycl-tla have no pipeline walk: refused.
// `pp_no_prefill()` (the default everywhere) is spec 16b's decode-only plan, unchanged.
inline PrefillPath pp_no_prefill() {
  PrefillPath p;
  p.prefill = false;
  return p;
}
struct PpDevicePlan : MemoryComponents {
  PpStage stage;
  size_t weights = 0, rope = 0, link = 0, prefill_link = 0;
};
struct PpPlan {
  uint32_t max_len = 0, split = 0;
  KvCache kv_cache = KvCache::Bf16;
  PrefillPath prefill = pp_no_prefill();
  std::array<PpDevicePlan, kPpDevices> dev{};
  size_t max_total() const;
};
PpPlan pp_plan(const model::ModelDesc& d, uint32_t split, uint32_t max_len, const PpWeights& w,
               KvCache kv = default_kv_cache(), const PrefillPath& pf = pp_no_prefill());

// The descriptor-free core of the balance: `layer_bytes[l]` is everything layer l puts on
// its device (weights + its state at the session's length), `dev0_fixed` / `dev1_fixed`
// what each device holds whatever the split (the embedding, the head, the scratch). The
// split in [1, n - 1] whose heavier side is smallest, ties to the more even, then the
// smaller s. pp_auto_split adds the Qwen-family constraint (a GDN and an FA layer per
// device); K2-Horizon, which has no ModelDesc and its own engine, is planned with this
// alone (tests/runtime/pipeline_plan_test.cc - PP for K2 is future work).
struct PpBalance {
  uint32_t split = 0;
  size_t dev0 = 0, dev1 = 0;
};
PpBalance pp_balance(const std::vector<size_t>& layer_bytes, size_t dev0_fixed, size_t dev1_fixed);

// `--pipeline-split auto` at an explicit max_len: the split whose heavier device holds the
// fewest bytes; ties go to the smaller difference between the two, then to the smaller s.
uint32_t pp_auto_split(const model::ModelDesc& d, const PpWeights& w, uint32_t max_len,
                       KvCache kv = default_kv_cache(), const PrefillPath& pf = pp_no_prefill());

// `--max-len auto` under a split: the largest multiple of kMaxLenQuantum up to `cap` whose
// plan + `reserve` fits EACH device (the min over the devices); 0 when not even
// min(kMinAutoMaxLen, cap) fits. Throws std::invalid_argument when cap < one quantum.
uint32_t pp_max_len_that_fits(const model::ModelDesc& d, uint32_t split, const PpWeights& w,
                              const std::array<size_t, kPpDevices>& device_bytes,
                              size_t reserve, uint32_t cap, KvCache kv = default_kv_cache(),
                              const PrefillPath& pf = pp_no_prefill());

// Both auto: the split whose min-over-devices max_len is the largest, ties broken as
// pp_auto_split breaks them at that length. {0, 0} when no split fits min(4096, cap).
struct PpChoice {
  uint32_t split = 0, max_len = 0;
};
PpChoice pp_auto_split_and_len(const model::ModelDesc& d, const PpWeights& w,
                               const std::array<size_t, kPpDevices>& device_bytes,
                               size_t reserve, uint32_t cap, KvCache kv = default_kv_cache(),
                               const PrefillPath& pf = pp_no_prefill());

// "pipeline plan at max_len N, split s (layers [0, s) | [s, L)):" and one memory_line()-style
// line per device ("device 0: model ... total ... of ...; + reserve").
std::string pp_describe(const PpPlan& p, const std::array<size_t, kPpDevices>& device_bytes,
                        size_t reserve);

}  // namespace runtime

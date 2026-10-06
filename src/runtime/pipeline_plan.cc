#include "runtime/pipeline_plan.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include "loader/draft_vocab.h"
#include "loader/moe_layout.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/pipeline_prefill_plan.h"

namespace runtime {
namespace {
using model::LinearId;
using model::Qwen35;
using model::WeightKind;

size_t round_up(size_t n, size_t q) { return (n + q - 1) / q * q; }

// One linear's device bytes as loader::load_linear allocates them.
size_t linear_bytes(const model::FusedLinear& fl) {
  const model::GemvShape& s = fl.shape;
  switch (fl.kind) {
    case WeightKind::Int4:
      // Layout 0: GPTQ words [K/8][N] + a separate f16 [K/64][N] scale allocation.
      // Layout 1: (N/16) x (K/64) tiles of 136 u32, scales inline.
      return s.layout == 0 ? size_t(s.K / 8) * s.N * 4 + size_t(s.K / 64) * s.N * 2
                           : loader::int4_layout1_bytes(s.K, s.N);
    case WeightKind::Bf16:
      return size_t(s.K) * s.N * 2;   // a||b at its padded 128 columns
    case WeightKind::Int8:
      return size_t(s.K) * s.N + size_t(s.N) * 4;   // int8 rows + fp32 row scales
  }
  return 0;
}

// The per-layer launches of the decode walk (runtime::decode_launches' terms): 12 for a
// dense layer, 13 for a MoE one; the boundary is embed_gather (1) and the head (5).
size_t layer_launches(const model::ModelDesc& d) { return d.is_moe() ? 13 : 12; }
constexpr size_t kEmbedLaunches = 1, kHeadLaunches = 5;

}  // namespace

const char* pp_handoff_name(PpHandoff h) { return h == PpHandoff::Peer ? "peer" : "copy"; }

bool parse_pp_handoff(const std::string& v, PpHandoff& out) {
  if (v == "copy") {
    out = PpHandoff::Copy;
    return true;
  }
  if (v == "peer") {
    out = PpHandoff::Peer;
    return true;
  }
  return false;
}

PpStage pp_stage(const model::ModelDesc& d, uint32_t first, uint32_t last) {
  if (first >= last || last > d.layers)
    throw std::invalid_argument("runtime::pp_stage: layers [" + std::to_string(first) + ", " +
                                std::to_string(last) + ") are not a non-empty range of " +
                                d.name + "'s " + std::to_string(d.layers));
  PpStage st;
  st.first = first;
  st.last = last;
  for (uint32_t l = 0; l < last; ++l) {
    const bool fa = model::ModelDesc::is_fa(l);
    if (l < first) {
      (fa ? st.fa_first : st.gdn_first) += 1;
    } else {
      (fa ? st.fa : st.gdn) += 1;
    }
  }
  return st;
}

void require_split(const model::ModelDesc& d, uint32_t split) {
  if (split < 1 || split >= d.layers)
    throw std::invalid_argument("--pipeline-split " + std::to_string(split) + " is not a split of " +
                                d.name + "'s " + std::to_string(d.layers) +
                                " layers: device 0 runs layers [0, s) and device 1 [s, " +
                                std::to_string(d.layers) + "), so s must be in [1, " +
                                std::to_string(d.layers - 1) + "]");
  // Each device's state is one GDN allocation and one KV pair, sliced per layer: a stage
  // without a layer of either kind would need a zero-byte allocation. With FA at every
  // fourth layer that rules out only the outermost splits (Qwen3.8: s < 4 or s > 62).
  const PpStage a = pp_stage(d, 0, split), b = pp_stage(d, split, d.layers);
  if (a.gdn == 0 || a.fa == 0 || b.gdn == 0 || b.fa == 0)
    throw std::invalid_argument("--pipeline-split " + std::to_string(split) + " leaves device " +
                                (a.gdn == 0 || a.fa == 0 ? "0" : "1") +
                                " without a GDN or a full-attention layer (" + d.name +
                                ": FA at every fourth layer); each device must hold at least one"
                                " of each");
}

std::array<PpStage, kPpDevices> pp_stages(const model::ModelDesc& d, uint32_t split) {
  require_split(d, split);
  return {pp_stage(d, 0, split), pp_stage(d, split, d.layers)};
}

size_t pp_stage_launches(const model::ModelDesc& d, const PpStage& st) {
  size_t n = size_t(st.layers()) * layer_launches(d);
  if (st.has_embed())
    n += kEmbedLaunches;
  else
    n -= 1;   // layer `first`'s leading prep_res_fold ran on the previous device
  if (st.has_head(d))
    n += kHeadLaunches;
  else
    n += 1;   // the next layer's prep_res_fold: the cut
  return n;
}

size_t PpWeights::total() const {
  size_t t = embed + head;
  for (size_t b : layer) t += b;
  return t;
}

size_t PpWeights::stage(const model::ModelDesc& d, const PpStage& st) const {
  if (layer.size() != d.layers)
    throw std::invalid_argument("runtime::PpWeights: " + std::to_string(layer.size()) +
                                " layer entries for " + d.name + "'s " + std::to_string(d.layers));
  size_t t = 0;
  for (uint32_t l = st.first; l < st.last; ++l) t += layer[l];
  if (st.has_embed()) t += embed;
  if (st.has_head(d)) t += head;
  return t;
}

PpWeights pp_weights(const model::ModelDesc& d, WeightKind lm_head) {
  PpWeights w;
  const loader::SmallLayout sl = d.small_layout();
  const size_t moe = loader::moe_layer_bytes(d).total();
  for (const model::LayerDesc& ld : d.layer_descs()) {
    size_t b = 0;
    for (const model::FusedLinear& fl : ld.linears) {
      // A MoE model's GateUp / Down rows are its shared expert, loaded into the expert
      // blocks (loader/moe_layout.h), not as linears of their own.
      if (d.is_moe() && (fl.id == LinearId::GateUp || fl.id == LinearId::Down)) continue;
      b += linear_bytes(fl);
    }
    b += sl.norms_block_bytes;
    b += ld.kind == model::LayerKind::FA ? loader::kFaBlockBytes : sl.gdn_block_bytes;
    b += moe;
    w.layer.push_back(b);
  }
  w.embed = size_t(Qwen35::kVocab) * d.hidden * 2;
  w.head = linear_bytes(d.lm_head(lm_head)) + sl.final_norm_bytes;
  return w;
}

PpLandingLayout pp_landing_layout(const model::ModelDesc& d) {
  const DecodeScratchSizes s = DecodeScratchDims::sizes(kMinAutoMaxLen, d);   // rows and sums: max_len-free
  return pp_landing_layout(s.resid, s.norm_sumsq);
}

PpLandingLayout pp_landing_layout(size_t resid_bytes, size_t sumsq_bytes) {
  PpLandingLayout l;
  l.resid_bytes = resid_bytes;
  l.sumsq_bytes = sumsq_bytes;
  l.sumsq_off = round_up(l.resid_bytes, kPpPage);
  l.stamp_off = l.sumsq_off + l.sumsq_bytes;
  l.flag_off = round_up(l.stamp_off + 4, kPpPage);
  l.total = l.flag_off + kPpPage;
  return l;
}

size_t pp_link_bytes(const model::ModelDesc& d, uint32_t device) {
  // Device 0: pp_send's counter; device 1: the landing allocation and pp_recv's state.
  return device == 0 ? kPpStateWords * 4 : pp_landing_layout(d).total + kPpStateWords * 4;
}

size_t PpPlan::max_total() const {
  size_t m = 0;
  for (const PpDevicePlan& p : dev) m = std::max(m, p.total());
  return m;
}

PpPlan pp_plan(const model::ModelDesc& d, uint32_t split, uint32_t max_len, const PpWeights& w,
               KvCache kv, const PrefillPath& pf) {
  return pp_plan(d, split, max_len, w, kv, pf, PpExtras{});
}

size_t pp_gdn_spec_bytes(const model::ModelDesc& d, uint32_t stage_gdn) {
  const size_t layer = size_t(d.gdn_v_heads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim * 4;
  return size_t(MtpDims::kSlots - 2) * d.gdn_layers * layer + size_t(stage_gdn) * layer;
}

size_t pp_mtp_prefill_bytes(const model::ModelDesc& d) {
  return mtp_prefill_hidden_bytes(d) + kPpPfDepth * sizeof(Control);
}

size_t pp_stage_verify_launches(const model::ModelDesc& d, const PpStage& st) {
  // The head's KV fill (capture.cc head_front): embed, the two pre-fc norm pairs, fc, the
  // zero-residual fold pair, q||k||v, attn_prep - 10, as runtime::verify_launches adds.
  return pp_stage_launches(d, st) + (st.has_head(d) ? 10 : 0);
}

std::vector<PpKvRun> pp_kv_runs(const std::array<KvLayout, kPpDevices>& stage, const KvLayout* head,
                                uint32_t begin, uint32_t end) {
  if (begin > end)
    throw std::invalid_argument("runtime::pp_kv_runs: begin " + std::to_string(begin) + " > end " +
                                std::to_string(end));
  for (const KvLayout& L : stage)
    if (end > L.max_len || L.form != stage[0].form)
      throw std::invalid_argument("runtime::pp_kv_runs: [" + std::to_string(begin) + ", " +
                                  std::to_string(end) + ") is not inside both stages' caches of "
                                  "one form");
  if (head && (head->form != stage[0].form || head->layers != 1 || end > head->max_len))
    throw std::invalid_argument("runtime::pp_kv_runs: the head's KV is not one layer in the "
                                "stages' form covering the range");
  std::vector<PpKvRun> runs;
  const size_t n = end - begin;
  if (n == 0) return runs;
  const uint32_t last = kPpDevices - 1;
  for (uint32_t t = 0; t < 2; ++t) {
    for (uint32_t dev = 0; dev < kPpDevices; ++dev) {
      const KvLayout& L = stage[dev];
      for (uint32_t l = 0; l < L.layers; ++l)
        runs.push_back({dev, t, false, L.rows_offset(l) + size_t(begin) * L.row_bytes(), n * L.row_bytes()});
    }
    if (head)   // the head's layer, after the last FA layer (Engine's kv_runs)
      runs.push_back({last, t, true, head->rows_offset(0) + size_t(begin) * head->row_bytes(),
                      n * head->row_bytes()});
    if (stage[0].scale_row_bytes() == 0) continue;
    for (uint32_t dev = 0; dev < kPpDevices; ++dev) {
      const KvLayout& L = stage[dev];
      for (uint32_t l = 0; l < L.layers; ++l)
        runs.push_back({dev, t, false, L.scales_offset(l) + size_t(begin) * L.scale_row_bytes(),
                        n * L.scale_row_bytes()});
    }
    if (head)
      runs.push_back({last, t, true, head->scales_offset(0) + size_t(begin) * head->scale_row_bytes(),
                      n * head->scale_row_bytes()});
  }
  return runs;
}

PpPlan pp_plan(const model::ModelDesc& d, uint32_t split, uint32_t max_len, const PpWeights& w,
               KvCache kv, const PrefillPath& pf, const PpExtras& x) {
  if (x.draft_vocab != 0 && !x.mtp)
    throw std::invalid_argument("runtime::pp_plan: a draft vocabulary drafts with the MTP head");
  if (pf.prefill && (pf.composed_attn || !is_l0(pf.backend)))
    throw std::invalid_argument(std::string("runtime::pp_plan: the two-card prefill runs the L0 "
                                            "backends' flash attention only, not ") +
                                (pf.composed_attn ? "the composed attention"
                                                  : prefill_backend_name(pf.backend)));
  PpPlan p;
  p.max_len = max_len;
  p.split = split;
  p.kv_cache = kv;
  p.prefill = pf;
  p.extras = x;
  const std::array<PpStage, kPpDevices> st = pp_stages(d, split);
  // The decode scratch is max_len-dependent only through v1's attn_part; the planner
  // follows B70_DECODE_ATTN as runtime::plan does.
  const size_t scratch = DecodeScratchDims::sizes(max_len, d).total();
  for (uint32_t i = 0; i < kPpDevices; ++i) {
    PpDevicePlan& dp = p.dev[i];
    dp.stage = st[i];
    dp.weights = w.stage(d, st[i]);
    dp.rope = Qwen35::rope_table_bytes(max_len);
    dp.link = pp_link_bytes(d, i);
    dp.model = dp.weights + dp.rope;
    const PersistentSizes ps = PersistentDims::stage_sizes(max_len, d, kv, st[i].gdn, st[i].fa);
    dp.kv = ps.kv_k + ps.kv_v;
    dp.decode_state = ps.total() - dp.kv + scratch + dp.link;
    if (pf.prefill) {   // spec 16c: memory_plan.cc's prefill terms, per device
      const PrefillScratchSizes s = PrefillScratchDims::sizes(max_len, d);
      dp.prefill_scratch = s.eager() + (pf.backend == PrefillBackend::L0 ? s.slab : 0);
      if (pf.backend == PrefillBackend::L0Int8)
        dp.int8 = int8_scratch_sizes(prefill_int8_max_k(d)).total() +
                  int8_scale_bytes(d, st[i].first, st[i].last);
      dp.prefill_link = pp_prefill_link_bytes(d, i);
      dp.decode_state += dp.prefill_link;
    }
    // Spec 16d (PpExtras, pipeline_plan.h).
    const bool last = st[i].has_head(d);
    if (x.mtp) {
      if (last) {
        dp.embed_replica = w.embed;
        dp.mtp_weights = w.mtp;
        dp.model += dp.embed_replica + dp.mtp_weights;
        dp.mtp_buffers = MtpDims::sizes(max_len, d, x.draft_vocab, kv).total();
        if (x.draft_vocab != 0)
          dp.draft_vocab = w.draft_vocab != 0
                               ? w.draft_vocab
                               : loader::draft_vocab_bytes(x.draft_vocab, d.hidden, x.draft_vocab_int8).total();
        if (pf.prefill) {
          dp.mtp_prefill = pp_mtp_prefill_bytes(d);
          // step_mtp_kv's bf16 linears walk the L0 slab; the l0 backend already holds it.
          if (pf.backend == PrefillBackend::L0Int8)
            dp.prefill_scratch += PrefillScratchDims::sizes(max_len, d).slab;
        }
      } else {
        dp.mtp_buffers = pp_gdn_spec_bytes(d, st[i].gdn);
      }
      dp.decode_state += dp.mtp_buffers + dp.mtp_prefill;
    }
    if (x.hook && pf.prefill) {
      const PersistentSizes hs = PersistentDims::stage_sizes(max_len, d, kv, st[i].gdn, st[i].fa);
      dp.shadows = kPpPfDepth * (hs.gdn_state + hs.conv_ring +
                                 (x.mtp && last ? size_t(d.hidden) * 2 : 0));
      dp.prefill_scratch += dp.shadows;
    }
  }
  return p;
}

namespace {
bool split_ok(const model::ModelDesc& d, uint32_t s) {
  try {
    require_split(d, s);
    return true;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

// The split choice at one length: the heavier device as small as possible, then the two as
// even as possible, then the smaller s. `allowed` filters candidates (all when empty).
uint32_t best_split_at(const model::ModelDesc& d, const PpWeights& w, uint32_t max_len, KvCache kv,
                       const PrefillPath& pf, const PpExtras& x, const std::vector<uint32_t>& allowed) {
  uint32_t best = 0;
  size_t best_max = 0, best_diff = 0;
  const auto consider = [&](uint32_t s) {
    const PpPlan p = pp_plan(d, s, max_len, w, kv, pf, x);
    const size_t a = p.dev[0].total(), b = p.dev[1].total();
    const size_t mx = std::max(a, b), diff = a > b ? a - b : b - a;
    if (best == 0 || mx < best_max || (mx == best_max && diff < best_diff)) {
      best = s;
      best_max = mx;
      best_diff = diff;
    }
  };
  if (allowed.empty()) {
    for (uint32_t s = 1; s < d.layers; ++s)
      if (split_ok(d, s)) consider(s);
  } else {
    for (uint32_t s : allowed) consider(s);
  }
  return best;
}
}  // namespace

PpBalance pp_balance(const std::vector<size_t>& layer_bytes, size_t dev0_fixed, size_t dev1_fixed) {
  PpBalance best;
  const size_t n = layer_bytes.size();
  size_t total = 0;
  for (size_t b : layer_bytes) total += b;
  size_t before = 0;   // layers [0, s)
  for (size_t s = 1; s < n; ++s) {
    before += layer_bytes[s - 1];
    const size_t a = dev0_fixed + before, b = dev1_fixed + (total - before);
    const size_t mx = std::max(a, b), diff = a > b ? a - b : b - a;
    const size_t bmx = std::max(best.dev0, best.dev1);
    const size_t bdiff = best.dev0 > best.dev1 ? best.dev0 - best.dev1 : best.dev1 - best.dev0;
    if (best.split == 0 || mx < bmx || (mx == bmx && diff < bdiff)) best = {uint32_t(s), a, b};
  }
  return best;
}

uint32_t pp_auto_split(const model::ModelDesc& d, const PpWeights& w, uint32_t max_len, KvCache kv,
                       const PrefillPath& pf) {
  return pp_auto_split(d, w, max_len, kv, pf, PpExtras{});
}

uint32_t pp_auto_split(const model::ModelDesc& d, const PpWeights& w, uint32_t max_len, KvCache kv,
                       const PrefillPath& pf, const PpExtras& x) {
  if (d.layers < 2)
    throw std::invalid_argument(d.name + " has fewer than two layers: nothing to split");
  const uint32_t s = best_split_at(d, w, max_len, kv, pf, x, {});
  if (s == 0)
    throw std::invalid_argument(d.name + ": no split leaves each device a GDN and an FA layer");
  return s;
}

uint32_t pp_max_len_that_fits(const model::ModelDesc& d, uint32_t split, const PpWeights& w,
                              const std::array<size_t, kPpDevices>& device_bytes, size_t reserve,
                              uint32_t cap, KvCache kv, const PrefillPath& pf) {
  return pp_max_len_that_fits(d, split, w, device_bytes, reserve, cap, kv, pf, PpExtras{});
}

uint32_t pp_max_len_that_fits(const model::ModelDesc& d, uint32_t split, const PpWeights& w,
                              const std::array<size_t, kPpDevices>& device_bytes, size_t reserve,
                              uint32_t cap, KvCache kv, const PrefillPath& pf, const PpExtras& x) {
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("pp_max_len_that_fits: the cap " + std::to_string(cap) +
                                " is below one " + std::to_string(kMaxLenQuantum) +
                                "-position quantum");
  require_split(d, split);
  const auto fits = [&](uint32_t len) {
    const PpPlan p = pp_plan(d, split, len, w, kv, pf, x);
    for (uint32_t i = 0; i < kPpDevices; ++i)
      if (p.dev[i].total() + reserve > device_bytes[i]) return false;
    return true;
  };
  // Every term is non-decreasing in max_len on both devices (memory_plan.cc's argument),
  // so the quanta that fit both are a prefix: bisect its end.
  uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum;
  uint32_t hi = cap / kMaxLenQuantum;
  if (!fits(lo * kMaxLenQuantum)) return 0;
  while (lo < hi) {   // invariant: lo fits
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid * kMaxLenQuantum))
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo * kMaxLenQuantum;
}

PpChoice pp_auto_split_and_len(const model::ModelDesc& d, const PpWeights& w,
                               const std::array<size_t, kPpDevices>& device_bytes, size_t reserve,
                               uint32_t cap, KvCache kv, const PrefillPath& pf) {
  return pp_auto_split_and_len(d, w, device_bytes, reserve, cap, kv, pf, PpExtras{});
}

PpChoice pp_auto_split_and_len(const model::ModelDesc& d, const PpWeights& w,
                               const std::array<size_t, kPpDevices>& device_bytes, size_t reserve,
                               uint32_t cap, KvCache kv, const PrefillPath& pf, const PpExtras& x) {
  uint32_t best_len = 0;
  std::vector<uint32_t> at_best;
  for (uint32_t s = 1; s < d.layers; ++s) {
    if (!split_ok(d, s)) continue;
    const uint32_t len = pp_max_len_that_fits(d, s, w, device_bytes, reserve, cap, kv, pf, x);
    if (len == 0) continue;
    if (len > best_len) {
      best_len = len;
      at_best.clear();
    }
    if (len == best_len) at_best.push_back(s);
  }
  if (best_len == 0) return {};
  return {best_split_at(d, w, best_len, kv, pf, x, at_best), best_len};
}

std::string pp_describe(const PpPlan& p, const std::array<size_t, kPpDevices>& device_bytes,
                        size_t reserve) {
  const double gb = 1e9;
  const PpStage& s0 = p.dev[0].stage;
  const PpStage& s1 = p.dev[1].stage;
  char head[256];
  std::snprintf(head, sizeof head,
                "pipeline plan at max_len %u, split %u (device 0 layers [0, %u): %u GDN + %u FA, "
                "device 1 layers [%u, %u): %u GDN + %u FA)%s%s%s%s",
                p.max_len, p.split, s0.last, s0.gdn, s0.fa, s1.first, s1.last, s1.gdn, s1.fa,
                p.kv_cache == KvCache::Int8 ? ", int8 KV cache" : "",
                p.prefill.prefill ? ", prefill " : "",
                p.prefill.prefill ? prefill_backend_name(p.prefill.backend) : "",
                p.extras.mtp ? ", the MTP head on device 1" : "");
  std::string out = head;
  for (uint32_t i = 0; i < kPpDevices; ++i) {
    const std::string label = "\n  device " + std::to_string(i);
    out += format_memory(label.c_str(), p.dev[i], device_bytes[i]);
    char tail[224];
    std::snprintf(tail, sizeof tail, "; + reserve %.3f GB = %.3f GB (weights %.3f GB, RoPE %.3f GB, "
                  "hand-off %zu B)",
                  reserve / gb, (p.dev[i].total() + reserve) / gb, p.dev[i].weights / gb,
                  p.dev[i].rope / gb, p.dev[i].link);
    out += tail;
    if (p.dev[i].prefill_link != 0) {   // spec 16c; 16b's line is unchanged without it
      std::snprintf(tail, sizeof tail, " + prefill hand-off %zu B", p.dev[i].prefill_link);
      out += tail;
    }
    // Spec 16d's terms, only when present: 16b's and 16c's lines are unchanged without them.
    if (p.dev[i].embed_replica + p.dev[i].mtp_weights != 0) {
      std::snprintf(tail, sizeof tail,
                    "; MTP: head %.3f GB + embedding replica %.3f GB in model, %.3f GB in decode state",
                    p.dev[i].mtp_weights / gb, p.dev[i].embed_replica / gb,
                    (p.dev[i].mtp_buffers + p.dev[i].mtp_prefill) / gb);
      out += tail;
    } else if (p.dev[i].mtp_buffers != 0) {
      std::snprintf(tail, sizeof tail, "; MTP: its layers' verify slots %.3f GB in decode state",
                    p.dev[i].mtp_buffers / gb);
      out += tail;
    }
    if (p.dev[i].shadows != 0) {
      std::snprintf(tail, sizeof tail, "; prefix-cache block shadows %.3f GB in prefill scratch",
                    p.dev[i].shadows / gb);
      out += tail;
    }
  }
  return out;
}

}  // namespace runtime

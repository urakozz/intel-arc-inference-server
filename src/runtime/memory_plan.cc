#include "runtime/memory_plan.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include "loader/draft_vocab.h"
#include "loader/moe_layout.h"
#include "runtime/buffer_sizes.h"

namespace runtime {

std::string format_memory(const char* label, const MemoryComponents& c, size_t device_bytes) {
  const double gb = 1e9;
  char dv[48] = "", dr[48] = "";
  if (c.draft_vocab) std::snprintf(dv, sizeof dv, " draft vocab %.3f GB,", c.draft_vocab / gb);
  if (c.drafter) std::snprintf(dr, sizeof dr, " drafter %.3f GB,", c.drafter / gb);   // spec 19b
  char buf[384];
  std::snprintf(buf, sizeof buf,
                "%s: model %.3f GB, kv %.3f GB, decode state %.3f GB, prefill scratch %.3f GB,"
                " int8 %.3f GB,%s%s total %.3f GB of %.3f GB",
                label, c.model / gb, c.kv / gb, c.decode_state / gb, c.prefill_scratch / gb,
                c.int8 / gb, dv, dr, c.total() / gb, device_bytes / gb);
  return buf;
}

MemoryPlan plan(const model::ModelDesc& desc, uint32_t max_len, bool mtp, size_t model_bytes,
                const PrefillPath& path, const DraftVocabPlan& dv, KvCache kv, const DrafterPlan& drafter) {
  // Spec 8 §11 / spec 19c: a draft vocabulary narrows the MTP head's or the DFlash drafter's head.
  if (dv.rows != 0 && !mtp && !drafter.on())
    throw std::invalid_argument("runtime::plan: a draft vocabulary drafts with the MTP head or a drafter");
  MemoryPlan p;
  p.drafter = drafter.bytes(max_len);   // spec 19b: 0 without a drafter
  p.max_len = max_len;
  p.mtp = mtp;
  p.kv_cache = kv;
  p.rope = model::Qwen35::rope_table_bytes(max_len);
  p.model = model_bytes + p.rope;

  const PersistentSizes ps = PersistentDims::sizes(max_len, desc, kv);
  p.kv = ps.kv_k + ps.kv_v;
  p.decode_state = ps.total() - p.kv + DecodeScratchDims::sizes(max_len, desc).total();
  // Spec 15c: the MoE split (both already counted above - model_bytes is the loaded total,
  // DecodeScratchSizes::moe is in decode_state).
  p.moe_weights = loader::moe_bytes(desc);
  p.moe_scratch = moe_scratch_layout(desc).total;
  // The loader's compact head and id table, outside model_bytes (LoadReport::total()) - for the MTP head's drafts or
  // (spec 19c) the drafter's, whose own logits are in its buffers (DrafterPlan).
  if (dv.rows != 0) p.draft_vocab = loader::draft_vocab_bytes(dv.rows, desc.hidden, dv.int8).total();
  if (mtp) {
    // With a draft vocabulary MtpBuffers also holds its compact logits (spec 8 §11).
    p.mtp_buffers = MtpDims::sizes(max_len, desc, dv.rows, kv).total();
    // Allocated on the first prefill of an engine with the head (engine_prefill.cc).
    p.mtp_hidden = path.prefill ? mtp_prefill_hidden_bytes(desc) : 0;
    p.decode_state += p.mtp_buffers + p.mtp_hidden;
  }

  if (path.prefill) {
    const PrefillScratchSizes s = PrefillScratchDims::sizes(max_len, desc);
    // linear_l0 walks the slab for every int4 linear; on l0-int8 only the MTP head's
    // bf16 linears do (step_mtp_kv).
    if (path.backend == PrefillBackend::L0 || mtp) p.prefill_lazy += s.slab;
    if (path.composed_attn) p.prefill_lazy += s.pf_s + s.pf_p;
    p.prefill_scratch = s.eager() + p.prefill_lazy;
    // Engine::prepare_prefill: Int8State for the MLP intermediate (down's K), then every
    // int4 linear's rotated column scales, all before the first chunk.
    if (path.backend == PrefillBackend::L0Int8)
      p.int8 = int8_scratch_sizes(prefill_int8_max_k(desc)).total() + int8_scale_bytes(desc);
    p.moe_prefill = s.moe;   // spec 15d: inside s.eager(), 0 when dense
  }
  // Spec 19c: the engine's side of a drafter - 7 more GDN slots, and the prefill's tap staging.
  if (drafter.on()) {
    p.drafter_slots = TapVerifyDims::gdn_spec_bytes(desc);
    p.decode_state += p.drafter_slots;
    if (path.prefill) {
      p.drafter_taps = TapVerifyDims::prefill_taps_bytes(desc, drafter.n_taps);
      p.prefill_scratch += p.drafter_taps;
    }
  }
  return p;
}

uint32_t max_len_that_fits(const model::ModelDesc& desc, bool mtp, size_t model_bytes,
                           size_t device_bytes, size_t reserve_bytes, uint32_t cap,
                           const PrefillPath& path, const DraftVocabPlan& dv, KvCache kv,
                           const DrafterPlan& drafter) {
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("max_len_that_fits: the cap " + std::to_string(cap) +
                                " is below one " + std::to_string(kMaxLenQuantum) +
                                "-position quantum");
  const auto fits = [&](uint32_t len) {
    return plan(desc, len, mtp, model_bytes, path, dv, kv, drafter).total() + reserve_bytes <= device_bytes;
  };
  // Every term of the plan is non-decreasing in max_len (the KV, attn_part's block
  // count, the RoPE table, the MTP head's KV, the composed pf_s / pf_p), so the
  // quanta that fit are a prefix and a bisection finds its end.
  uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum;   // in quanta
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

std::string describe(const MemoryPlan& p, size_t device_bytes, size_t reserve_bytes) {
  const double gb = 1e9;
  const std::string label = "plan at max_len " + std::to_string(p.max_len);
  std::string s = format_memory(label.c_str(), p, device_bytes);
  char buf[256];
  std::snprintf(buf, sizeof buf, "; + reserve %.3f GB = %.3f GB (RoPE %.3f GB in model",
                reserve_bytes / gb, (p.total() + reserve_bytes) / gb, p.rope / gb);
  s += buf;
  if (p.mtp) {
    std::snprintf(buf, sizeof buf, ", MTP %.3f GB in decode state", (p.mtp_buffers + p.mtp_hidden) / gb);
    s += buf;
  }
  if (p.moe_weights != 0) {
    std::snprintf(buf, sizeof buf, ", MoE experts %.3f GB in model", p.moe_weights / gb);
    s += buf;
  }
  if (p.moe_prefill != 0) {
    std::snprintf(buf, sizeof buf, ", MoE prefill %.3f GB in prefill scratch", p.moe_prefill / gb);
    s += buf;
  }
  if (p.prefill_lazy != 0) {
    std::snprintf(buf, sizeof buf, ", lazy %.3f GB in prefill scratch", p.prefill_lazy / gb);
    s += buf;
  }
  if (p.drafter_slots != 0) {   // spec 19c
    std::snprintf(buf, sizeof buf, ", drafter's verify slots %.3f GB in decode state", p.drafter_slots / gb);
    s += buf;
  }
  if (p.drafter_taps != 0) {
    std::snprintf(buf, sizeof buf, ", drafter's prefill taps %.3f GB in prefill scratch", p.drafter_taps / gb);
    s += buf;
  }
  if (p.kv_cache == KvCache::Int8) s += ", int8 KV cache";   // spec 12b; bf16 says nothing
  return s + ")";
}

}  // namespace runtime

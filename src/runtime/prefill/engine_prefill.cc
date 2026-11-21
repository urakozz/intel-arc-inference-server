// `Engine::prefill` and its two accessors -- the ONLY members of
// `runtime::Engine` that are not compiled into `b70_runtime`.
//
// **Why they live here.** `b70_runtime` is linked by every decode binary and
// every decode test; `Engine::prefill` calls into `runtime::prefill::Context` and the
// prefill walk. Defining it in this archive keeps the split exact: a target that never
// calls `prefill()` links what it always linked.
// The member declarations stay in `runtime/engine.h`; only the definitions move.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill_chunks.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill/linear_l0.h"
#include "runtime/prefill/moe.h"
#include "runtime/prefill/profile.h"
#include "runtime/prefill/step.h"

namespace runtime {

struct PrefillEngine {
  prefill::Context cx;
  prefill::KernelCache kc;
  // Spec 5's int8 state (signs, per-linear column scales, int8 scratch), created on the
  // first l0-int8 prefill. Declared before `chunks`: recordings that name its buffers are
  // destroyed first.
  std::unique_ptr<prefill::Int8State> int8;
  explicit PrefillEngine(l0::Context& c) : cx(c), kc(c) {}
};

namespace {
void destroy_prefill(PrefillEngine* p) { delete p; }
}  // namespace

size_t Engine::prefill_launches() const { return pfx_ ? pfx_->cx.launches() : 0; }

PrefillBackend Engine::prefill_backend() const {
  return pf_backend_ ? *pf_backend_ : prefill::default_prefill_backend();
}

uint32_t Engine::prefill_chunk_cap() const {
  if (pf_) return pf_->chunk_cap();   // sized once: the cap the buffers have
  return pf_chunk_cap_ ? *pf_chunk_cap_ : default_prefill_chunk_cap();
}

void Engine::set_prefill_chunk_cap(uint32_t cap) {
  check_prefill_chunk_cap(*model_.desc, cap);
  if (pf_ && pf_->chunk_cap() != cap)
    throw std::logic_error("runtime::Engine::set_prefill_chunk_cap: the prefill scratch already holds " +
                           std::to_string(pf_->chunk_cap()) + "-row chunks; set the cap (" +
                           std::to_string(cap) + ") before prepare_prefill()");
  pf_chunk_cap_ = cap;
}

MemoryComponents Engine::memory_use() const {
  // Spec 39: every allocation of the engine is in a paired buffers struct (none of its own members is one).
  static_assert(gs::buffers_problem<l0::Mem, Engine>().empty(), gs::buffers_problem<l0::Mem, Engine>());
  MemoryComponents c;
  c.model = model_.report.total();
  c.kv = persist_.kv_k.size() + persist_.kv_v.size();
  // Spec 8: the MTP head's buffers (its KV, the GDN slots, hh/dh, draft logits) count as
  // decode state; the head's weights are in `model` (LoadReport::mtp_bytes). Spec 13c: so do
  // the further sequence slots' (SlotBuffers; their MTP part is in MtpBuffers already).
  c.decode_state = persist_.bytes() - c.kv + decode_scratch_.bytes() +
                   (mtp_ ? mtp_->bytes() : 0) + (pf_own_.mtp_hid ? pf_own_.mtp_hid->size() : 0) +
                   (slot_bufs_ ? slot_bufs_->bytes() : 0);
  // Spec 39: Engine's own allocations (EngineBuffers) are decode state - spec 19c's drafter verify slots, spec 32's
  // deferred GDN rows and deep chain scratch, and the debug tap (which this sum left out before). A drafter's prefill
  // tap staging is prefill scratch, and the drafter itself (weights, ring, taps, scratch) its own term.
  c.decode_state += own_.bytes();
  // EnginePrefillSizes by name: the plan puts the head's hidden rows in decode state and the tap staging in prefill
  // scratch (MemoryPlan::mtp_hidden, drafter_taps).
  c.prefill_scratch = (pf_ ? pf_->bytes() + pf_->lazy_bytes() : 0) + (pf_own_.df_taps ? pf_own_.df_taps->size() : 0);
  c.drafter = drafter_ ? drafter_->bytes() : 0;
  c.int8 = pfx_ && pfx_->int8 ? pfx_->int8->bytes() : 0;
  // Spec 8 §11: the draft vocabulary's compact head and id table - the loader's, but
  // outside LoadReport::total(), so `model` reads as before; its compact logits are in
  // MtpBuffers, i.e. decode state.
  c.draft_vocab = model_.report.draft_vocab_bytes;
  return c;
}

std::string Engine::memory_line() const {
  return format_memory("memory", memory_use(), ctx_.memory_bytes());
}

void Engine::prepare_prefill() {
  // A model the prefill path cannot run is refused here, before any prefill allocation,
  // by name (Engine::prefill calls this first). Spec 15d: a MoE model prefills through the
  // grouped experts.
  model::require_prefill(*model_.desc);
  // Ruling R7: both allocations are lazy, so a decode-only Engine's device
  // residency is byte-identical to what it was before the buffer split.
  // Plan 5d: every per-chunk buffer at the session's chunk cap (checked against the model).
  const uint32_t cap = prefill_chunk_cap();
  check_prefill_chunk_cap(*model_.desc, cap);
  // 3j: int8's widening scratch at the longest slot (one sequence prefills at a time; runtime::plan_slots).
  uint32_t longest = 0;
  for (uint32_t q = 0; q < slots(); ++q) longest = std::max(longest, slot_len(q));
  if (!pf_) pf_.reset(new PrefillScratch(ctx_, buffers_.max_len, *model_.desc, cap, longest));
  if (kv_cache() == KvCache::Int8) pf_->kv8_raw_buffer();
  if (!pfx_)
    pfx_ = std::unique_ptr<PrefillEngine, void (*)(PrefillEngine*)>(new PrefillEngine(ctx_),
                                                                    &destroy_prefill);
  // Phase 3 Task 2: a MoE model's router / a||b B slabs (Qwen3.8 one card: a||b's), once, outside any chunk (counted
  // by the memory plan).
  prefill::make_route_slabs(pfx_->cx, pfx_->kc, *pf_, model_, 0, model_.desc->layers);
  // Open-work 38: the mixer linears' bf16 slabs (counted by the plan too), finished here so no timed prefill pays them.
  if (pf_dense_bf16_) {
    prefill::make_dense_slabs(pfx_->cx, pfx_->kc, *pf_, model_);
    pfx_->cx.wait();
  }
  if (prefill_backend() != PrefillBackend::L0Int8) return;
  if (!pfx_->int8)   // the largest int4 K the h8 walk runs: down's on a dense model (spec
                     // 14), the hidden size on a MoE model (the W4A8 experts' quantiser, open-work 16)
    pfx_->int8 = std::make_unique<prefill::Int8State>(ctx_, prefill_int8_max_k(*model_.desc), pf_->chunk_cap());
  if (model_.desc->is_moe()) return;   // open-work 16: no h8 linear on a MoE model (W4A8 routed experts only)
  // Every h8 linear's rotated column scales (spec 5 T2), outside any chunk and
  // any recording: cached by weight, so after the first call this loop only
  // looks them up, and no recorded list ever holds the host finish of
  // pf_colmax_rot.
  // Not a||b: prefill runs it in bf16 (the dense GEMM, or pf_ab_proj on a tensor-parallel card) on every backend,
  // over the checkpoint's bf16 weight or (spec 15 §13) an int4 a||b's bf16 copy - never through h8.
  // Not a MoE model's mixer linears either: l0-int8 runs them on the l0 walk there (open-work 16).
  for (const auto& [key, w] : model_.linears)
    if (key.second != model::LinearId::LmHead && key.second != model::LinearId::AB &&
        w.kind == model::WeightKind::Int4 && !l0_int8_bf16_linear(*model_.desc, key.second))
      pfx_->int8->scales(pfx_->cx, pfx_->kc, w);
}

void Engine::drafter_project_staged(uint32_t pos, uint32_t n) {
  if (drafter_->project_wide(pfx_->cx, pfx_->kc, *pf_, pf_own_.df_taps->ptr(), pos, n)) {
    prefill::timed_wait(pfx_->cx, prefill::Phase::kDraftProj);   // the ring's rows have landed: a block hook's
                                                                  // snapshot, the next draft
    return;
  }
  const size_t row = drafter_->tap_layers().size() * size_t(model_.desc->hidden) * 2;
  for (uint32_t r = 0; r < n; r += 8) {
    const uint32_t g = std::min<uint32_t>(8, n - r);
    imm_.copy(drafter_->taps().ptr(), pf_own_.df_taps->as<uint8_t>() + size_t(r) * row, size_t(g) * row);
    replay_aux(drafter_->projection(pos + r, g));
  }
}

void Engine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  if (ids.empty()) throw std::runtime_error("runtime::Engine::prefill: no ids");
  const uint32_t cap = prefill_chunk_cap();
  if (chunk == 0) chunk = cap;
  if (chunk > cap)
    throw std::runtime_error("runtime::Engine::prefill: chunk " + std::to_string(chunk) +
                             " exceeds the session's prefill chunk cap " + std::to_string(cap) +
                             " (set_prefill_chunk_cap: 2048 or 4096)");
  for (uint32_t id : ids)
    if (id >= model::Qwen35::kVocab)
      throw std::runtime_error(
          "runtime::Engine::prefill: id " + std::to_string(id) +
          " is outside the vocabulary (" + std::to_string(model::Qwen35::kVocab) +
          " rows) -- pf_embed_gather has no debug_flag channel, so the host is the only bound");
  flush_projection();
  const uint32_t base = control_->pos;
  if (size_t(base) + ids.size() > size_t(max_len()))
    throw std::runtime_error(
        "runtime::Engine::prefill: pos " + std::to_string(base) + " + " +
        std::to_string(ids.size()) + " ids exceeds max_len " +
        std::to_string(max_len()) +
        " -- the KV cache and the RoPE table stop there. Start a new session with reset(), or"
        " load the model and the engine with a larger max_len.");
  need_pages(cur_, base + uint32_t(ids.size()), "prefill");   // spec 31: the call's positions, before a chunk writes

  prepare_prefill();

  const PrefillBackend backend = prefill_backend();

  prefill::Int8State* q = backend == PrefillBackend::L0Int8 ? pfx_->int8.get() : nullptr;
  // Spec 12b: an int8 cache is written and read by the flash path's kv8 twins only; refuse
  // the composed path here, before a chunk writes anything.
  if (persist_.kv_lay.form == KvCache::Int8) prefill::require_kv8_path(backend);

  // With a block hook (spec 7 §3.2) every chunk ends at a block end or at the prompt
  // end, so each completed block is a storable one; without one, uniform chunks.
  const bool hooked = static_cast<bool>(block_hook_);
  // Spec 8: with the MTP head, every chunk also fills the head's KV (step_mtp_kv), which
  // needs the chunk's post-final-norm hidden rows and h_{pos-1} (MtpBuffers::hh row 0).
  // The prefill GDN path reads slot 0 directly, so a live slot left by a commit is
  // copied there first.
  uint16_t* hid = nullptr;
  // Spec 19c: with a drafter, slot 0 holds the state the prefill GDN path reads and writes, and every chunk taps
  // its rows inside the drafter's window of this call's end (the positions its ring reads back) into the staging,
  // projected after the chunk (drafter_project_staged). Positions before this call keep the ring's own rows: the
  // session's, or since spec 19d a restored snapshot's (save_state / load_state carry the ring - spec 19 decision 5).
  // A mid-call block end's snapshot (the block hook below) holds the ring as it stands there: its window's positions
  // that are not inside this call's last window() have not been projected yet - lower acceptance after a restore
  // there followed by a tail shorter than the window, never a different output (spec 19 §12).
  std::vector<std::vector<uint32_t>> df_sites;
  uint32_t df_from = 0;   // the first position tapped
  if (drafter_) {
    df_sites = tap_sites(*model_.desc, drafter_->tap_layers());
    if (!df_sites.back().empty())
      throw std::runtime_error("runtime::Engine::prefill: the drafter taps the last layer's output, whose fold "
                               "prefill runs for the last row only - not built (no published drafter does)");
    mtp_normalise_live();
    seq().verify_k = kNoVerify;
    if (!pf_own_.df_taps)
      pf_own_.df_taps = std::make_unique<l0::Mem>(
          ctx_, l0::MemKind::Device,
          engine_prefill_sizes(*model_.desc, pf_->chunk_cap(), false, uint32_t(drafter_->tap_layers().size())).df_taps);
    const uint64_t end = uint64_t(base) + ids.size(), w = drafter_->window();
    df_from = uint32_t(end > w ? std::max<uint64_t>(base, end - w) : base);
  }
  if (mtp_) {
    mtp_normalise_live();
    seq().verify_k = kNoVerify;
    if (!pf_own_.mtp_hid)
      pf_own_.mtp_hid = std::make_unique<l0::Mem>(ctx_, l0::MemKind::Device,
                                                  engine_prefill_sizes(*model_.desc, pf_->chunk_cap(), true, 0).mtp_hid);
    hid = pf_own_.mtp_hid->as<uint16_t>();
  }
  const size_t hid_row = size_t(model_.desc->hidden) * 2;
  // Spec 13c: the selected slot's state, ring and KV range (slot 0 of a one-slot engine: the buffers' own, unchanged).
  const Seq& sq = seq();
  prefill::PrefillSlot pslot;
  pslot.gdn_state = sq.state0;
  pslot.conv_ring = sq.ring;
  pslot.kv_base = sq.base;
  pslot.pages = sq.bind.pages;   // plan 19v3b Task 0: several slots' KV is paged
  if (pool_) pslot.positions = layout_.total();   // spec 31: the sequences' table regions
  const prefill::PrefillSlot* slot = slots() > 1 ? &pslot : nullptr;
  // PERF-89: the copies a mid-prompt hook leaves on the copy engine end before this call returns or throws.
  struct SnapDrain {
    const Engine& e;
    ~SnapDrain() {
      try {
        e.snap_wait();
      } catch (...) {
      }
    }
  } snap_drain{*this};
  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    C = prefill_chunk_rows(base + uint32_t(off), ids.size() - off, chunk, hooked, kBlock, /*wide_stride=*/true,
                           prefill_tail());
    pfx_->cx.wait();                       // before touching `ids` or `Control`
    std::memcpy(pf_->ids.ptr(), ids.data() + off, size_t(C) * 4);
    control_->pos = base + uint32_t(off);
    control_->n_active = C;
    note_kv(base + uint32_t(off) + C);   // PERF-90: before the chunk writes, so a throw leaves it counted
    if (mtp_) {
      const uint32_t pos = base + uint32_t(off);
      hctl_->pos = pos == 0 ? 0 : pos - 1;
      hctl_->n_active = prefill::mtp_kv_rows(pos, C);
      imm_.copy(hid, sq.hh, hid_row);   // h_{pos-1} into row 0
    }
    // Spec 19c: this chunk's rows from df_from on (r0 = C: none).
    const uint32_t cpos = base + uint32_t(off);
    prefill::PrefillTaps df_taps;
    if (drafter_) {
      df_taps.dst = pf_own_.df_taps->ptr();
      df_taps.sites = &df_sites;
      df_taps.n_taps = uint32_t(drafter_->tap_layers().size());
      df_taps.r0 = df_from > cpos ? std::min(df_from - cpos, C) : 0;
    }
    prefill::step_chunk(pfx_->cx, pfx_->kc, *pf_, model_, buffers_.max_len, control_,
                      base + uint32_t(off), C, persist_.gdn_state, persist_.conv_ring,
                      persist_.kv_k, persist_.kv_v, persist_.kv_lay, backend, q,
                      drafter_ ? &df_taps : nullptr, slot);
    if (mtp_)
      prefill::step_mtp_kv(pfx_->cx, pfx_->kc, *pf_, model_, hctl_, base + uint32_t(off), C, hid,
                           mtp_->kv_lay.at_base(mtp_->kv_lay.layer(mtp_->kv_k.ptr(), mtp_->kv_v.ptr(), 0),
                                                pslot.pages ? 0 : sq.base),
                           pslot.pages);
    pfx_->cx.wait();                       // the chunk's state has landed
    if (drafter_ && df_taps.r0 < C) drafter_project_staged(cpos + df_taps.r0, C - df_taps.r0);   // spec 19c
    if (mtp_)   // h_{end-1}: the next chunk's row 0, a snapshot's hidden, the first draft's h
      imm_.copy(sq.hh, reinterpret_cast<uint8_t*>(hid) + size_t(C) * hid_row, hid_row);
    const uint32_t end = base + uint32_t(off) + C;
    if (hooked && off + C < ids.size() && end % kBlock == 0) {
      // Mid-prompt block end: the state is exactly at `end`; say so in Control before
      // the hook runs, so a hook that throws leaves pos matching the chunks written.
      control_->pos = end;
      control_->n_active = 0;
      snap_wait();   // PERF-89: the previous hook's copies, whose staging this one reuses
      snap_async_ = true;
      try {
        block_hook_(end, true);
      } catch (...) {
        snap_async_ = false;
        throw;
      }
      snap_async_ = false;
    }
  }

  // The tail: pos = base + L - 1 and n_active = 1 make `argmax_stage2` leave
  // pos = base + L and cur_token[0] = the first generated id. `last` is the last
  // row of the last chunk (C rows).
  const uint32_t last = C - 1;
  control_->pos = base + uint32_t(ids.size()) - 1;
  control_->n_active = 1;
  prefill::step_head(pfx_->cx, pfx_->kc, *pf_, model_, control_, last,
                     mtp_ ? reinterpret_cast<uint8_t*>(hid) + size_t(C) * hid_row : nullptr);
  pfx_->cx.wait();
  snap_wait();   // PERF-89: the last mid-prompt hook's copies ran beside the last chunk; this hook's are blocking
  if (hooked) {
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
}

}  // namespace runtime

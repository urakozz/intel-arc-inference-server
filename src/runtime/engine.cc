#include "runtime/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "model/qwen35.h"

namespace runtime {
namespace {
using model::Qwen35;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// The captured list's M. It is runtime/capture.cc's `kCapM` - 1 in this plan -
// and it sizes the tap the same way capture.cc's check_sizes() expects it, so
// the two constants have to move together. (capture.cc keeps it file-local; it
// is the M the kernel *variants* were compiled for, not DecodeBuffers::kM,
// which is the allocation capacity.)
constexpr uint32_t kCapM = 1;
// The tap is bf16 [layers][kCapM][hidden] - the descriptor's layer count (spec 14)
// and hidden size (spec 15b; 5120 on Qwen3.8).
size_t tap_elems(const model::ModelDesc& d) { return size_t(d.layers) * kCapM * d.hidden; }
// One bf16 hidden row: the descriptor's hidden x 2 (spec 15b; 10240 B on Qwen3.8).
size_t hid_bytes(const model::ModelDesc& d) { return size_t(d.hidden) * 2; }

// Checked in the member-initialiser list so it throws before DecodeBuffers
// allocates ~1.2 GB. runtime::build() checks the same identity, but only after
// the buffers exist and the weights are already resident.
uint32_t checked_max_len(const loader::LoadedModel& m, uint32_t max_len) {
  if (m.max_len != max_len)
    throw std::runtime_error("runtime::Engine: model was loaded with max_len " +
                             std::to_string(m.max_len) + " but the engine was asked for " +
                             std::to_string(max_len) +
                             " - the RoPE table, the KV allocation and the attention variants'"
                             " baked MAXLEN are one number");
  return max_len;
}

// Spec 13c: the slot lengths, checked against max_len before anything is allocated (empty: one slot of max_len).
std::vector<uint32_t> checked_slots(const loader::LoadedModel& m, uint32_t max_len, std::vector<uint32_t> slots) {
  if (slots.empty()) return {max_len};
  uint64_t sum = 0;
  for (uint32_t l : slots) {
    if (l == 0) throw std::runtime_error("runtime::Engine: a sequence slot of 0 positions");
    sum += l;
  }
  if (sum != max_len)
    throw std::runtime_error("runtime::Engine: the sequence slots hold " + std::to_string(sum) +
                             " positions, but max_len is " + std::to_string(max_len) +
                             " - the KV caches are one allocation at the slots' sum");
  if (slots.size() > SlotDims::kMaxSeqs)
    throw std::runtime_error("runtime::Engine: " + std::to_string(slots.size()) + " sequence slots; at most " +
                             std::to_string(SlotDims::kMaxSeqs));
  if (slots.size() > 1) {
    // The row builds bake one whole-model GDN state and ring of Qwen3.8's shape (src/kernels/CMakeLists.txt,
    // BR_GDN_QWEN), and the batched lists are the dense model's.
    const model::ModelDesc& d = *m.desc;
    if (d.is_moe() || d.gdn_layers != 48 || d.gdn_v_heads != 48)
      throw std::runtime_error("runtime::Engine: sequence slots are built for Qwen3.8 (spec 13c); " + d.name +
                               " serves one sequence");
    if (m.draft_vocab)
      throw std::runtime_error("runtime::Engine: a draft vocabulary beside sequence slots is not built");
  }
  return slots;
}
}  // namespace

Engine::Engine(l0::Context& ctx, loader::LoadedModel model, uint32_t max_len, bool debug_resid,
               KvCache kv, TapDrafter* drafter, std::vector<uint32_t> slot_lens)
    : ctx_(ctx),
      model_(std::move(model)),
      persist_(ctx, checked_max_len(model_, max_len), *model_.desc, kv),
      decode_scratch_(ctx, persist_.max_len, *model_.desc),
      buffers_(persist_, decode_scratch_),
      tap_(debug_resid ? std::unique_ptr<l0::Mem>(
                             new l0::Mem(ctx, l0::MemKind::Device, tap_elems(*model_.desc) * 2))
                       : nullptr),
      mtp_(model_.mtp ? std::make_unique<MtpBuffers>(
                            ctx, persist_.max_len, *model_.desc,
                            model_.draft_vocab ? model_.draft_vocab->size() : 0u,   // spec 8 §11
                            kv, MtpBuffers::kSlots,
                            static_cast<uint32_t>(checked_slots(model_, max_len, slot_lens).size()))   // spec 13c
                      : nullptr),
      step_(build(ctx, model_, buffers_, tap_.get())),
      queue_(ctx),
      fence_(queue_),
      imm_(l0::CmdList::immediate(ctx)),
      control_(buffers_.control.as<Control>()) {
  setup_slots(checked_slots(model_, max_len, std::move(slot_lens)));
  if (drafter && seqs_.size() > 1)
    throw std::runtime_error("runtime::Engine: a tap drafter beside sequence slots is not built (spec 13c: MTP or "
                             "prompt lookup per slot)");
  if (drafter) {
    // Spec 19c: one proposer per engine, and the drafter's verify lists at M = 1 .. max_k + 1 over its 8 slots.
    if (mtp_)
      throw std::runtime_error("runtime::Engine: a tap drafter and the MTP head are two proposers - load the "
                               "model without its head (one proposer per server, spec 19 decision 3)");
    if (drafter->max_k() == 0 || drafter->max_k() + 1 > TapVerifyDims::kSlots)
      throw std::runtime_error("runtime::Engine: the drafter drafts " + std::to_string(drafter->max_k()) +
                               " ids a block; the verify lists take 1.." +
                               std::to_string(TapVerifyDims::kSlots - 1));
    drafter_ = drafter;
    df_slots_ = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device,
                                          TapVerifyDims::gdn_spec_bytes(*model_.desc));
    df_tv_.gdn_spec = df_slots_.get();
    df_tv_.slots = TapVerifyDims::kSlots;
    df_tv_.taps = &drafter->taps();
    df_tv_.layers = drafter->tap_layers();
    seqs_[0].spec = df_slots_->as<uint8_t>();
    for (uint32_t M = 1; M <= drafter->max_k() + 1; ++M) capture_verify(M, {});
  }
  if (mtp_) {
    // Spec 8: the draft lists (one per draft index, each writing its own logits row)
    // and the verify lists at M = 1..kSlots. Only the head's model carries these.
    hctl_ = mtp_->hctl.as<Control>();
    seqs_[0].hctl = hctl_;
    for (uint32_t i = 0; i < kMaxDraft; ++i)
      seqs_[0].draft_steps.push_back(build_draft(ctx, model_, buffers_, *mtp_, i));
    for (uint32_t M = 1; M <= MtpBuffers::kSlots; ++M) capture_verify(M, {});
  }
  // Spec 13c: every further slot's own lists, then the row lists; then every slot reset (slot 0 last: selected).
  for (uint32_t q = 1; q < slots(); ++q) capture_slot(q);
  if (slots() > 1)
    for (uint32_t M = 2; M <= kMaxRows; ++M) rows_steps_.push_back(build_rows(ctx, model_, buffers_, rows_bind_, M));
  if (slots() > 1 && mtp_)
    for (uint32_t M = 2; M <= std::min(slots(), kMaxDraftRows); ++M)
      draft_rows_steps_.push_back(build_draft_rows(ctx, model_, buffers_, *mtp_, draft_bind_, M));
  for (uint32_t q = slots(); q-- > 0;) {
    select(q);
    reset();
  }
}

// --- spec 13c: sequence slots ----------------------------------------------------------------------------------------

void Engine::setup_slots(const std::vector<uint32_t>& lens) {
  const model::ModelDesc& d = *model_.desc;
  const uint32_t S = static_cast<uint32_t>(lens.size());
  const size_t state = persist_.gdn_state.size(), ring = persist_.conv_ring.size();
  const size_t hid = size_t(d.hidden) * 2;
  seqs_.resize(S);
  uint32_t base = 0;
  for (uint32_t q = 0; q < S; ++q) {
    seqs_[q].base = base;
    seqs_[q].len = lens[q];
    layout_.slots.push_back({lens[q], mtp_ != nullptr});
    base += lens[q];
  }
  Seq& z = seqs_[0];   // slot 0: the buffers' own state, today's binding (bind all null)
  z.control = control_;
  z.state0 = persist_.gdn_state.as<uint8_t>();
  z.ring = persist_.conv_ring.as<uint8_t>();
  if (mtp_) {
    z.hctl = mtp_->hctl.as<Control>();
    z.spec = mtp_->gdn_spec.as<uint8_t>();
    z.hh = mtp_->hh.as<uint8_t>();
  }
  if (S == 1) return;
  slot_bufs_ = std::make_unique<SlotBuffers>(ctx_, d, S, mtp_ != nullptr);
  // GDN index i >= 1 lives at pool + (i - 1) whole states: with MTP MtpBuffers::gdn_spec (slot 0's three, then every
  // further slot's four - MtpDims' `seqs`), without SlotBuffers::states (one a further slot).
  uint8_t* pool = mtp_ ? mtp_->gdn_spec.as<uint8_t>() : slot_bufs_->states->as<uint8_t>();
  const uint32_t G = gdn_slots();
  for (uint32_t q = 1; q < S; ++q) {
    Seq& s = seqs_[q];
    s.control = reinterpret_cast<Control*>(slot_bufs_->control_block(SlotDims::control_at(S, q)));
    s.state0 = pool + (size_t(q) * G - 1) * state;
    s.ring = slot_bufs_->rings.as<uint8_t>() + size_t(q - 1) * ring;
    if (mtp_) {
      s.hctl = reinterpret_cast<Control*>(slot_bufs_->control_block(SlotDims::head_control_at(S, q)));
      s.spec = pool + size_t(q) * G * state;
      s.hh = mtp_->hh.as<uint8_t>() + size_t(q) * (DecodeScratchDims::kM + 1) * hid;
    }
    s.bind.control = s.control;
    s.bind.gdn_state = s.state0;
    s.bind.gdn_spec = s.spec;
    s.bind.conv_ring = s.ring;
    s.bind.kv_base = s.base;
    s.bind.hctl = s.hctl;
    s.bind.hh = s.hh;
  }
  // The row lists' bindings (capture.h, build_rows): their own Control blocks and tables, the pools above, and with
  // MTP the gather over MtpBuffers::hh - every slot's rows (slot q's h_{pos-1} at row q x (kM + 1)), then the saved
  // rows, then the head's input rows (MtpDims::hh_rows).
  BatchRowsBind& b = rows_bind_;
  b.rows = slot_bufs_->control_block(SlotDims::rows_table_at(S));
  b.control = slot_bufs_->control_block(SlotDims::rows_control_at(S));
  b.state_pool = pool;
  b.ring_pool = slot_bufs_->rings.ptr();
  if (mtp_) {
    b.mtp = mtp_.get();
    b.head_rows = slot_bufs_->control_block(SlotDims::head_table_at(S));
    b.hctl = slot_bufs_->control_block(SlotDims::rows_head_control_at(S));
    b.gctl = slot_bufs_->control_block(SlotDims::gather_control_at(S));
    b.table = mtp_->hh.ptr();
    b.saved = mtp_->hh.as<uint8_t>() + size_t(S) * (DecodeScratchDims::kM + 1) * hid;
    b.hin = static_cast<uint8_t*>(b.saved) + size_t(DecodeScratchDims::kM) * hid;
    // The row draft lists: their own head Control and table, the chain rows after the gather's, the logits rows of
    // the decode scratch (read before the verify that rewrites them).
    draft_bind_.hctl = slot_bufs_->control_block(SlotDims::draft_control_at(S));
    draft_bind_.head_rows = slot_bufs_->control_block(SlotDims::draft_table_at(S));
    draft_bind_.dh = static_cast<uint8_t*>(b.hin) + size_t(DecodeScratchDims::kM) * hid;
    draft_bind_.logits = buffers_.logits.ptr();
  }
}

void Engine::capture_slot(uint32_t q) {
  // Slot q's own copy of every list slot 0 has (the constructor's), bound to slot q's state (Seq::bind): the same
  // kernels, the same order, its own pointers.
  select(q);
  Seq& s = seq();
  s.plain.push_back(build(ctx_, model_, buffers_, nullptr, nullptr, 1, {}, s.bind));
  if (!mtp_) return;
  for (uint32_t i = 0; i < kMaxDraft; ++i) s.draft_steps.push_back(build_draft(ctx_, model_, buffers_, *mtp_, i, s.bind));
  for (uint32_t M = 1; M <= MtpBuffers::kSlots; ++M) capture_verify(M, {});
}

void Engine::select(uint32_t slot) {
  if (slot >= seqs_.size())
    throw std::out_of_range("runtime::Engine::select: no slot " + std::to_string(slot) + " (the engine has " +
                            std::to_string(seqs_.size()) + ")");
  cur_ = slot;
  control_ = seqs_[slot].control;
  hctl_ = seqs_[slot].hctl;
}

void Engine::zero_slot() {
  // Spec 13c: exactly the selected slot's persistent state - what reset() zeroes for one slot.
  Seq& s = seq();
  const size_t state = persist_.gdn_state.size();
  imm_.fill(s.control, 0u, sizeof(Control));
  imm_.fill(s.state0, 0u, state);
  imm_.fill(s.ring, 0u, persist_.conv_ring.size());
  if (mtp_) {
    imm_.fill(s.hctl, 0u, sizeof(Control));
    imm_.fill(s.spec, 0u, state * (MtpBuffers::kSlots - 1));
    imm_.fill(s.hh, 0u, size_t(DecodeScratchDims::kM + 1) * model_.desc->hidden * 2);
  }
  // Its KV range, every layer (K, V and at int8 the scales; the head's layer too).
  const auto range = [&](const KvLayout& L, l0::Mem& k, l0::Mem& v) {
    for (uint32_t l = 0; l < L.layers; ++l) {
      const KvLayer x = L.at_base(L.layer(k.ptr(), v.ptr(), l), s.base);
      imm_.fill(x.k, 0u, size_t(s.len) * L.row_bytes());
      imm_.fill(x.v, 0u, size_t(s.len) * L.row_bytes());
      if (x.int8()) {
        imm_.fill(x.ks, 0u, size_t(s.len) * L.scale_row_bytes());
        imm_.fill(x.vs, 0u, size_t(s.len) * L.scale_row_bytes());
      }
    }
  };
  range(persist_.kv_lay, persist_.kv_k, persist_.kv_v);
  if (mtp_) range(mtp_->kv_lay, mtp_->kv_k, mtp_->kv_v);
}

void Engine::verify_rows(const std::vector<Run>& runs) {
  if (slots() < 2) throw std::runtime_error("runtime::Engine::verify_rows: the engine has one sequence slot");
  const uint32_t G = gdn_slots();
  std::vector<RowRun> rr;
  uint32_t M = 0;
  for (const Run& r : runs) {
    const Seq& s = seqs_.at(r.slot);
    const uint32_t pos = s.control->pos;
    if (r.k > max_draft() || (mtp_ && pos == 0) || size_t(pos) + r.k + 1 > s.len)
      throw std::runtime_error("runtime::Engine::verify_rows: slot " + std::to_string(r.slot) + ": k " +
                               std::to_string(r.k) + " at pos " + std::to_string(pos) + " (k <= " +
                               std::to_string(max_draft()) + ", pos >= 1 with the MTP head, pos + k + 1 <= its " +
                               std::to_string(s.len) + " positions)");
    for (uint32_t j = 0; j <= r.k; ++j)
      if (s.control->cur_token[j] >= Qwen35::kVocab)
        throw std::runtime_error("runtime::Engine::verify_rows: slot " + std::to_string(r.slot) + "'s id " +
                                 std::to_string(s.control->cur_token[j]) + " in row " + std::to_string(j) +
                                 " is outside the vocabulary");
    RowRun x;
    x.slot = r.slot;
    x.pos = pos;
    x.rows = r.k + 1;
    const uint32_t live = s.control->gdn_live;
    x.src = gdn_index(r.slot, live);
    for (uint32_t j = 0; j <= r.k; ++j) x.dst.push_back(gdn_index(r.slot, (live + j) % G));   // verify()'s slots
    rr.push_back(x);
    M += x.rows;
  }
  if (M < 2 || M > kMaxRows)
    throw std::runtime_error("runtime::Engine::verify_rows: " + std::to_string(M) + " rows; a row step takes 2.." +
                             std::to_string(kMaxRows) + " (one slot's lone step is its own verify / generate)");
  const std::vector<BatchRow> t = make_rows(layout_, rr);   // check_rows: one run a slot, every rule
  // The host's writes: the tables, the rows' ids, the head's pairs (h_{p-1}, x_p) at p - 1, the gather's table rows.
  auto* table = static_cast<BatchRowTable*>(rows_bind_.rows);
  auto* rc = static_cast<Control*>(rows_bind_.control);
  rc->n_active = M;
  uint32_t m = 0;
  for (const RowRun& x : rr)
    for (uint32_t j = 0; j < x.rows; ++j, ++m) {
      table->row[m] = t[m];
      rc->cur_token[m] = seqs_[x.slot].control->cur_token[j];
    }
  if (mtp_) {
    auto* ht = static_cast<BatchRowTable*>(rows_bind_.head_rows);
    auto* hc = static_cast<Control*>(rows_bind_.hctl);
    auto* gc = static_cast<Control*>(rows_bind_.gctl);
    hc->n_active = M;
    gc->n_active = M;
    const uint32_t row_hh = DecodeScratchDims::kM + 1, saved = slots() * row_hh;
    for (uint32_t i = 0; i < M; ++i) {
      ht->row[i] = t[i];
      ht->row[i].pos = t[i].pos - 1;
      hc->cur_token[i] = rc->cur_token[i];
      const bool first = i == 0 || t[i - 1].slot != t[i].slot;
      gc->cur_token[i] = first ? t[i].slot * row_hh : saved + i - 1;
    }
  }
  queue_.execute(rows_steps_.at(M - 2).list, &fence_);
  fence_.wait();
  rows_m_ = M;
  // Per run: committable as after its own verify(k) - row j's hidden is the saved row r0 + j.
  const size_t hid = size_t(model_.desc->hidden) * 2;
  uint32_t r0 = 0;
  for (const RowRun& x : rr) {
    Seq& s = seqs_[x.slot];
    s.verify_k = x.rows - 1;
    s.verify_pos = x.pos;
    s.verify_hidden = mtp_ ? static_cast<const uint8_t*>(rows_bind_.saved) + size_t(r0) * hid : nullptr;
    s.control->n_active = x.rows;
    r0 += x.rows;
  }
}

void Engine::draft_rows(const std::vector<Run>& runs,
                        const std::function<uint32_t(size_t, uint32_t, uint32_t)>& pick) {
  require_mtp("draft_rows");
  const uint32_t M = static_cast<uint32_t>(runs.size());
  if (M < 2 || M - 2 >= draft_rows_steps_.size())
    throw std::runtime_error("runtime::Engine::draft_rows: " + std::to_string(M) + " runs; 2.." +
                             std::to_string(draft_rows_steps_.size() + 1) + " (one is the slot's own draft())");
  uint32_t K = 0;
  for (const Run& r : runs) K = std::max(K, r.k);
  const size_t hid = hid_bytes(*model_.desc);
  auto* hc = static_cast<Control*>(draft_bind_.hctl);
  auto* table = static_cast<BatchRowTable*>(draft_bind_.head_rows);
  for (size_t m = 0; m < runs.size(); ++m) {
    Seq& s = seqs_.at(runs[m].slot);
    const uint32_t pos = s.control->pos;
    if (runs[m].k == 0 || runs[m].k > kMaxDraft || pos == 0 || size_t(pos) + K + 1 > s.len)
      throw std::runtime_error("runtime::Engine::draft_rows: slot " + std::to_string(runs[m].slot) + ": k " +
                               std::to_string(runs[m].k) + " (the step drafts " + std::to_string(K) + ") at pos " +
                               std::to_string(pos) + " of its " + std::to_string(s.len) + " positions");
    for (size_t n = 0; n < m; ++n)
      if (runs[n].slot == runs[m].slot) throw std::runtime_error("runtime::Engine::draft_rows: a slot twice");
    // The chain's first hidden is the slot's h_{pos-1}; its first input the pending id (draft()'s start).
    imm_.copy(static_cast<uint8_t*>(draft_bind_.dh) + m * hid, s.hh, hid);
    hc->cur_token[m] = s.control->cur_token[0];
    s.draft_ids.clear();
  }
  hc->n_active = M;
  CapturedStep& list = draft_rows_steps_[M - 2];
  for (uint32_t i = 0; i < K; ++i) {
    for (size_t m = 0; m < runs.size(); ++m) {
      const Seq& s = seqs_[runs[m].slot];
      BatchRow row;
      row.pos = s.control->pos - 1 + i;   // draft i of the slot at position pos - 1 + i (draft())
      row.kv_base = s.base;
      row.slot = runs[m].slot;
      table->row[m] = row;
    }
    queue_.execute(list.list, &fence_);
    fence_.wait();
    for (size_t m = 0; m < runs.size(); ++m) {
      Seq& s = seqs_[runs[m].slot];
      uint32_t d = hc->out_token[m];
      if (pick) {
        d = pick(m, i, d);
        if (d >= Qwen35::kVocab)
          throw std::runtime_error("runtime::Engine::draft_rows: picked id " + std::to_string(d) +
                                   " is outside the vocabulary");
      }
      hc->cur_token[m] = d;   // the next draft index's input
      if (i < runs[m].k) {
        s.draft_ids.push_back(d);
        s.control->cur_token[1 + i] = d;
      }
    }
  }
}

const uint32_t* Engine::rows_ids() const {
  if (slots() < 2) throw std::runtime_error("runtime::Engine::rows_ids: the engine has one sequence slot");
  return static_cast<const Control*>(rows_bind_.control)->out_token;
}
const float* Engine::rows_logits_device() const { return buffers_.logits.as<float>(); }

// --- the single-sequence calls, on the selected slot -----------------------------------------------------------------

void Engine::capture_verify(uint32_t M, const CaptureOptions& opt) {
  Seq& s = seq();
  auto one = [&](const CaptureOptions& o) {
    return drafter_ ? build_verify_taps(ctx_, model_, buffers_, df_tv_, M, o)
                    : build_verify(ctx_, model_, buffers_, *mtp_, M, nullptr, o, s.bind);
  };
  // Spec 28c: the deep list where the rule switches this M at a depth (and nothing forces one kind).
  const model::ModelDesc& d = *model_.desc;
  const uint32_t from = opt.force_attn == VerifyAttnForce::Auto
                            ? verify_attn_deep_from(M, kv_cache() == KvCache::Int8, d.fa_q_heads, d.fa_kv_heads)
                            : kDpasNever;
  const bool by_depth = from != kDpasNever;
  CaptureOptions shallow = opt;
  if (by_depth && opt.profile_keys >= from) shallow.profile = nullptr;
  s.verify_steps.push_back(one(shallow));
  if (by_depth) {
    const auto t0 = std::chrono::steady_clock::now();
    CaptureOptions deep = attn_forced(VerifyAttnForce::Dpas);
    if (opt.profile_keys >= from) {
      deep.profile = opt.profile;
      deep.profile_m = opt.profile_m;
    }
    s.verify_deep.push_back(std::make_unique<CapturedStep>(one(deep)));
    verify_deep_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    s.verify_deep_from.push_back(from);
  } else {
    s.verify_deep.push_back(nullptr);
    s.verify_deep_from.push_back(kDpasNever);
  }
}

void Engine::recapture_verify(const CaptureOptions& opt) {
  // The constructor's loops, with `opt`: index M - 1 is the list at M rows (a CapturedStep moves, it does not
  // assign: the old lists go first). Spec 13c: every slot's.
  const uint32_t keep = cur_;
  verify_deep_ms_ = 0;
  for (uint32_t q = 0; q < slots(); ++q) {
    select(q);
    Seq& s = seq();
    const uint32_t n = uint32_t(s.verify_steps.size());
    s.verify_steps.clear();
    s.verify_deep.clear();
    s.verify_deep_from.clear();
    for (uint32_t M = 1; M <= n; ++M) capture_verify(M, opt);
  }
  select(keep);
}

uint32_t Engine::verify_deep_lists() const {
  uint32_t n = 0;
  for (const auto& p : seq().verify_deep) n += p != nullptr;
  return n;
}

const CapturedStep& Engine::verify_step_at(uint32_t M, uint32_t keys) const {
  require_spec("verify_step_at");
  const size_t i = M - 1;
  const Seq& s = seq();
  return s.verify_deep.at(i) && keys >= s.verify_deep_from[i] ? *s.verify_deep[i] : s.verify_steps.at(i);
}

void Engine::reset() {
  if (slots() > 1) {
    // Spec 13c: the selected slot only; the shared scratch (dh, the draft logits) is written before it is read.
    zero_slot();
    seq().verify_k = kNoVerify;
    seq().draft_ids.clear();
    return;
  }
  // Exactly the persistent group, in exactly the order PersistentBuffers'
  // constructor filled it - one function, so "reset writes what construction
  // wrote" is a property of the code and not of two lists agreeing. Scratch is
  // left alone on purpose - see the header.
  persist_.zero(imm_);
  if (mtp_) mtp_->zero(imm_);
  // Spec 19c: the drafter's slots (as MtpBuffers::zero does the head's) and a cold drafter.
  if (df_slots_) imm_.fill(df_slots_->ptr(), 0u, df_slots_->size());
  if (drafter_) drafter_->reset(imm_);
  seq().verify_k = kNoVerify;
  seq().draft_ids.clear();
}

void Engine::replay() {
  const size_t pos = control_->pos, n = control_->n_active;
  if (pos + n > size_t(max_len()))
    throw std::runtime_error(
        "runtime::Engine: pos " + std::to_string(pos) + " + n_active " + std::to_string(n) +
        " exceeds max_len " + std::to_string(max_len()) +
        " - the KV cache and the RoPE table stop there. Start a new session with reset(), or"
        " load the model and the engine with a larger max_len.");
  queue_.execute(plain().list, &fence_);
  fence_.wait();
}

void Engine::ingest(const std::vector<uint32_t>& ids) {
  control_->n_active = 1;
  for (uint32_t id : ids) {
    control_->cur_token[0] = id;
    if (mtp_ || drafter_)
      mtp_step1();
    else
      replay();
  }
}

std::vector<uint32_t> Engine::generate(uint32_t n,
                                       const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  control_->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    // Produced by the previous fence; replay i is what turns it into i + 1.
    const uint32_t id = control_->cur_token[0];
    const Clock::time_point f0 = Clock::now();
    if (mtp_ || drafter_)
      mtp_step1();
    else
      replay();
    fence_ms += ms_since(f0);
    out.push_back(id);
    if (on_token) on_token(id);
  }
  last_gen_ms_ = ms_since(t0);
  last_fence_ms_ = fence_ms;
  last_tok_per_s_ = (n != 0 && last_gen_ms_ > 0.0) ? double(n) * 1000.0 / last_gen_ms_ : 0.0;
  return out;
}

// --- spec 8: MTP -------------------------------------------------------------------


void Engine::require_mtp(const char* what) const {
  if (!mtp_)
    throw std::runtime_error(std::string("runtime::Engine::") + what +
                             ": MTP is off - load the model with its head (loader::load(..., mtp"
                             " = true), b70 --spec mtp)");
}

void Engine::require_spec(const char* what) const {
  if (!mtp_ && !drafter_)
    throw std::runtime_error(std::string("runtime::Engine::") + what +
                             ": no proposer - load the model with its MTP head (b70 --spec mtp) or build the engine "
                             "with a drafter (--spec dflash)");
}

uint32_t Engine::max_draft() const { return mtp_ ? kMaxDraft : drafter_ ? drafter_->max_k() : 0; }

uint32_t Engine::gdn_slots() const {
  return mtp_ ? MtpBuffers::kSlots : drafter_ ? TapVerifyDims::kSlots : 1;
}

void Engine::replay_aux(l0::CmdList& list) {
  queue_.execute(list, &fence_);
  fence_.wait();
}

uint32_t Engine::max_verify_k() const {
  const uint32_t pos = control_->pos;
  // Spec 19c: a drafter's verify needs no h_{pos-1}, so pos 0 drafts too.
  if ((!mtp_ && !drafter_) || (mtp_ && pos == 0) || pos + 1 >= max_len()) return 0;
  return std::min<uint32_t>(max_draft(), max_len() - pos - 1);
}

const float* Engine::mtp_logits_device() const {
  require_mtp("mtp_logits_device");
  return mtp_->logits.as<float>();
}
const float* Engine::verify_logits_device() const {
  require_spec("verify_logits_device");
  return buffers_.logits.as<float>();
}
const CapturedStep& Engine::verify_step(uint32_t M) const {
  require_spec("verify_step");
  return seq().verify_steps.at(M - 1);
}
const CapturedStep& Engine::draft_step(uint32_t i) const {
  require_mtp("draft_step");
  return seq().draft_steps.at(i);
}

void Engine::draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick) {
  require_mtp("draft");
  Seq& s = seq();
  const uint32_t pos = control_->pos;
  if (k == 0 || k > kMaxDraft || k > max_verify_k())
    throw std::runtime_error("runtime::Engine::draft: k " + std::to_string(k) +
                             " is outside [1, max_verify_k() = " + std::to_string(max_verify_k()) +
                             "] at pos " + std::to_string(pos));
  // The chain's first hidden is h_{pos-1} (hh row 0); each draft list writes its own
  // output hidden back into dh. Draft i runs at position pos - 1 + i (P0 §1).
  imm_.copy(mtp_->dh.ptr(), s.hh, hid_bytes(*model_.desc));
  hctl_->pos = pos - 1;
  hctl_->n_active = 1;
  hctl_->cur_token[0] = control_->cur_token[0];
  s.draft_ids.clear();
  for (uint32_t i = 0; i < k; ++i) {
    queue_.execute(s.draft_steps[i].list, &fence_);
    fence_.wait();
    uint32_t d = hctl_->out_token[0];
    if (pick) {
      d = pick(i);
      if (d >= Qwen35::kVocab)
        throw std::runtime_error("runtime::Engine::draft: picked id " + std::to_string(d) +
                                 " is outside the vocabulary");
      hctl_->cur_token[0] = d;   // the next draft list's input
    }
    s.draft_ids.push_back(d);
    control_->cur_token[1 + i] = d;
  }
}

void Engine::verify(uint32_t k) {
  require_spec("verify");
  Seq& s = seq();
  const uint32_t pos = control_->pos;
  if (k > max_draft() || (mtp_ && pos == 0) || size_t(pos) + k + 1 > max_len())
    throw std::runtime_error("runtime::Engine::verify: k " + std::to_string(k) + " at pos " +
                             std::to_string(pos) + " (max_verify_k() = " +
                             std::to_string(max_verify_k()) + "; k <= " + std::to_string(max_draft()) +
                             ", pos >= 1 with the MTP head, and pos + k + 1 <= max_len " +
                             std::to_string(max_len()) + ")");
  for (uint32_t r = 0; r <= k; ++r)
    if (control_->cur_token[r] >= Qwen35::kVocab)
      throw std::runtime_error("runtime::Engine::verify: id " +
                               std::to_string(control_->cur_token[r]) + " in row " +
                               std::to_string(r) + " is outside the vocabulary");
  control_->n_active = k + 1;
  if (mtp_) {
    // The head's KV fill of the same rows at pos - 1 .. pos + k - 1: row r is the pair
    // (hh[r], cur_token[r]) = (h_{pos-1+r}, x[pos+r]) - capture.h, build_verify.
    hctl_->pos = pos - 1;
    hctl_->n_active = k + 1;
    for (uint32_t r = 0; r <= k; ++r) hctl_->cur_token[r] = control_->cur_token[r];
  }
  s.verify_pos = pos;
  s.verify_k = k;
  s.verify_hidden = mtp_ ? s.hh + hid_bytes(*model_.desc) : nullptr;   // hh row 1 + j: row j's hidden
  // Spec 28c: the deep list (DPAS bound) once row 0's keys reach the table's threshold.
  CapturedStep& list =
      s.verify_deep[k] && pos + 1 >= s.verify_deep_from[k] ? *s.verify_deep[k] : s.verify_steps[k];
  queue_.execute(list.list, &fence_);
  fence_.wait();
}

void Engine::commit(uint32_t j, uint32_t next_token) {
  Seq& s = seq();
  // Spec 13c: a slot of a verify_rows step commits without a proposer too (its plain row, j = 0).
  if (s.verify_k == kNoVerify || (!mtp_ && !drafter_ && slots() == 1)) require_spec("commit");
  if (s.verify_k == kNoVerify || j > s.verify_k)
    throw std::runtime_error("runtime::Engine::commit: j " + std::to_string(j) +
                             (s.verify_k == kNoVerify ? " with no verify() pending"
                                                      : " exceeds the last verify's k " +
                                                            std::to_string(s.verify_k)));
  if (next_token >= Qwen35::kVocab)
    throw std::runtime_error("runtime::Engine::commit: next_token " + std::to_string(next_token) +
                             " is outside the vocabulary");
  // Row j's GDN state is slot (live + j) % slots (gdn_step.cl SPEC_SLOTS; spec 13c's row
  // lists write the same indices): make it live. Row j's hidden is the next iteration's
  // h_{pos-1}.
  control_->gdn_live = (control_->gdn_live + j) % gdn_slots();
  if (mtp_) imm_.copy(s.hh, s.verify_hidden + size_t(j) * hid_bytes(*model_.desc), hid_bytes(*model_.desc));
  control_->pos = s.verify_pos + j + 1;
  control_->n_active = 1;
  control_->cur_token[0] = next_token;
  s.verify_k = kNoVerify;
  // Spec 19c: the committed rows' taps (rows 0..j: positions verify_pos .. + j) into the drafter's ring. The
  // projection list reads only the taps and its own control block, so Control's new pos / cur_token above do not
  // reach it; it runs before anything can rewrite the taps (the next verify).
  if (drafter_) replay_aux(drafter_->projection(s.verify_pos, j + 1));
}

void Engine::mtp_step1() {
  if (drafter_) {   // spec 19c: no hidden bookkeeping; the projection rides in commit()
    verify(0);
    commit(0, control_->out_token[0]);
    return;
  }
  if (control_->pos == 0) {
    // No h_{-1}: the plain list, which reads and writes slot 0 (live is 0 at pos 0 -
    // reset() and load_state() both leave it so), then its hidden into hh row 0.
    if (control_->gdn_live != 0) mtp_normalise_live();
    replay();
    imm_.copy(seq().hh, buffers_.x.ptr(), hid_bytes(*model_.desc));
    return;
  }
  verify(0);
  commit(0, control_->out_token[0]);
}

const void* Engine::live_gdn_slot() const {
  const uint32_t live = (mtp_ || drafter_) ? control_->gdn_live : 0;
  const Seq& s = seq();
  if (live == 0) return s.state0;
  return s.spec + size_t(live - 1) * persist_.gdn_state.size();
}

void Engine::mtp_normalise_live() {
  const uint32_t live = control_->gdn_live;
  if ((!mtp_ && !drafter_) || live == 0) return;
  imm_.copy(seq().state0, live_gdn_slot(), persist_.gdn_state.size());
  control_->gdn_live = 0;
}

// --- spec 7 snapshots -----------------------------------------------------------

namespace {
void check_range(uint32_t begin, uint32_t end, uint32_t max_len, const char* what) {
  if (begin > end || end > max_len)
    throw std::runtime_error(std::string("runtime::Engine::") + what + ": range [" +
                             std::to_string(begin) + ", " + std::to_string(end) +
                             ") is not within [0, max_len " + std::to_string(max_len) + ")");
}
}  // namespace

size_t Engine::state_bytes() const {
  // Spec 19d: with a drafter, its ring follows (TapDrafter::ring_bytes; never with the MTP head - one proposer).
  return persist_.gdn_state.size() + persist_.conv_ring.size() + (mtp_ ? hid_bytes(*model_.desc) : 0) +
         (drafter_ ? drafter_->ring_bytes() : 0);
}

size_t Engine::kv_bytes(uint32_t n_pos) const {
  const size_t fa = model_.desc->fa_layers;   // 16 on Qwen3.8, 10 on Ornith
  // One position of one layer: the rows, and at int8 their scales (spec 12b).
  return 2 * persist_.kv_lay.pos_bytes() * (fa + (mtp_ ? 1 : 0)) * n_pos;   // K and V
}

void Engine::save_state(void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  const Seq& s = seq();
  // The LIVE GDN slot (spec 8: after a commit it may be one of gdn_spec's; spec 19c: of the
  // drafter's slots). Spec 19d (spec 19 decision 5, as ruled there): with a drafter its whole ring
  // follows, so a restore resumes the drafter bitwise where this snapshot found it.
  imm_.copy(h, live_gdn_slot(), persist_.gdn_state.size());
  imm_.copy(h + persist_.gdn_state.size(), s.ring, persist_.conv_ring.size());
  if (mtp_)
    imm_.copy(h + persist_.gdn_state.size() + persist_.conv_ring.size(), s.hh, hid_bytes(*model_.desc));
  if (drafter_) drafter_->save_ring(imm_, h + persist_.gdn_state.size() + persist_.conv_ring.size());
}

void Engine::load_state(const void* host, uint32_t pos) {
  if (pos > max_len())
    throw std::runtime_error("runtime::Engine::load_state: pos " + std::to_string(pos) +
                             " exceeds max_len " + std::to_string(max_len()));
  const auto* h = static_cast<const uint8_t*>(host);
  Seq& s = seq();
  imm_.copy(s.state0, h, persist_.gdn_state.size());
  imm_.copy(s.ring, h + persist_.gdn_state.size(), persist_.conv_ring.size());
  if (mtp_) {
    imm_.copy(s.hh, h + persist_.gdn_state.size() + persist_.conv_ring.size(), hid_bytes(*model_.desc));
    control_->gdn_live = 0;   // the state went into slot 0
    s.verify_k = kNoVerify;
  }
  if (drafter_) {   // spec 19c: the state went into slot 0; spec 19d: and the ring is the snapshot's
    drafter_->load_ring(imm_, h + persist_.gdn_state.size() + persist_.conv_ring.size());
    control_->gdn_live = 0;
    s.verify_k = kNoVerify;
  }
  control_->pos = pos;
  control_->n_active = 0;
}

// kv_k / kv_v are KvLayout's: [fa_layers][max_len][4][256] rows (bf16, or int8 then the
// [fa_layers][max_len][4] fp16 scales). A position range is one contiguous run per layer,
// so fa_layers (16 on Qwen3.8) copies each for K and V, plus as many scale
// runs at int8 (spec 12b). The two loops below walk the same (allocation, offset, bytes)
// list in the same order, one copying out and the other in. Spec 13c: the selected slot's
// positions, from its KV base.
namespace {
struct KvRun {
  uint8_t* dev;
  size_t bytes;
};
// Every device run of positions [begin, end), in host order: per tensor (K, V), the rows of
// every layer (the head's last), then at int8 the scales the same way.
std::vector<KvRun> kv_runs(const PersistentBuffers& p, const MtpBuffers* mtp, uint32_t begin,
                           uint32_t end) {
  std::vector<KvRun> runs;
  const KvLayout& L = p.kv_lay;
  const size_t n = end - begin;
  for (int kv = 0; kv < 2; ++kv) {
    uint8_t* base = (kv == 0 ? p.kv_k : p.kv_v).as<uint8_t>();
    uint8_t* hbase = mtp ? (kv == 0 ? mtp->kv_k : mtp->kv_v).as<uint8_t>() : nullptr;
    for (uint32_t l = 0; l < L.layers; ++l)
      runs.push_back({base + L.rows_offset(l) + begin * L.row_bytes(), n * L.row_bytes()});
    if (mtp)   // the head's layer, after the last FA layer
      runs.push_back({hbase + mtp->kv_lay.rows_offset(0) + begin * L.row_bytes(), n * L.row_bytes()});
    if (L.scale_row_bytes() == 0) continue;
    for (uint32_t l = 0; l < L.layers; ++l)
      runs.push_back({base + L.scales_offset(l) + begin * L.scale_row_bytes(),
                      n * L.scale_row_bytes()});
    if (mtp)
      runs.push_back({hbase + mtp->kv_lay.scales_offset(0) + begin * L.scale_row_bytes(),
                      n * L.scale_row_bytes()});
  }
  return runs;
}
}  // namespace

void Engine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  check_range(begin, end, max_len(), "save_kv");
  if (begin == end) return;
  auto* h = static_cast<uint8_t*>(host);
  const uint32_t b = seq().base;
  for (const KvRun& r : kv_runs(persist_, mtp_.get(), b + begin, b + end)) {
    imm_.copy(h, r.dev, r.bytes);
    h += r.bytes;
  }
}

void Engine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  check_range(begin, end, max_len(), "load_kv");
  if (begin == end) return;
  const auto* h = static_cast<const uint8_t*>(host);
  const uint32_t b = seq().base;
  for (const KvRun& r : kv_runs(persist_, mtp_.get(), b + begin, b + end)) {
    imm_.copy(r.dev, h, r.bytes);
    h += r.bytes;
  }
}

std::vector<uint16_t> Engine::read_debug_resid() {
  if (!tap_)
    throw std::runtime_error(
        "runtime::Engine::read_debug_resid: this engine was constructed with debug_resid=false,"
        " so no per-layer tap was captured");
  std::vector<uint16_t> out(tap_elems(*model_.desc));
  imm_.copy(out.data(), tap_->ptr(), out.size() * 2);
  return out;
}

}  // namespace runtime

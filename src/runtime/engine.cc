#include "runtime/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/kernels.h"
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
// Spec 31: with a page pool, every sequence's cap - all equal, at most the pool - and max_len the pool's positions.
std::vector<uint32_t> checked_slots(const loader::LoadedModel& m, uint32_t max_len, std::vector<uint32_t> slots,
                                    uint32_t pool_pages) {
  if (slots.empty() && pool_pages != 0) throw std::runtime_error("runtime::Engine: a KV page pool without sequences");
  if (slots.empty()) return {max_len};
  for (uint32_t l : slots)
    if (l == 0) throw std::runtime_error("runtime::Engine: a sequence slot of 0 positions");
  if (pool_pages != 0) {
    if (slots.size() < 2 || std::count(slots.begin(), slots.end(), slots[0]) != std::ptrdiff_t(slots.size()))
      throw std::runtime_error("runtime::Engine: a KV page pool serves two or more sequences of one cap");
    if (uint64_t(pool_pages) * kKvPage != max_len || slots[0] > max_len)
      throw std::runtime_error("runtime::Engine: a pool of " + std::to_string(pool_pages) + " pages of " +
                               std::to_string(kKvPage) + " is max_len " + std::to_string(uint64_t(pool_pages) * kKvPage) +
                               " (asked " + std::to_string(max_len) + "), and a request's cap " +
                               std::to_string(slots[0]) + " at most that");
  } else if (const uint32_t sum = slots_kv_len(slots); sum != max_len) {   // plan 19v3b Task 0: whole KV pages each
    throw std::runtime_error("runtime::Engine: the sequence slots hold " + std::to_string(sum) +
                             " positions (runtime::slots_kv_len), but max_len is " + std::to_string(max_len) +
                             " - the KV caches are one allocation of the slots' pages");
  }
  if (slots.size() > SlotDims::kMaxSeqs)
    throw std::runtime_error("runtime::Engine: " + std::to_string(slots.size()) + " sequence slots; at most " +
                             std::to_string(SlotDims::kMaxSeqs));
  if (slots.size() > 1) {
    // The row builds bake one whole-model GDN state and ring of Qwen3.8's shape (src/kernels/CMakeLists.txt,
    // BR_GDN_QWEN), and the batched lists are the dense model's.
    const model::ModelDesc& d = *m.desc;
    if (d.is_moe() || d.gdn_layers != 48 || d.gdn_v_heads != 48)
      throw std::runtime_error("runtime::Engine: sequence slots are built for Qwen3.8; " + d.name +
                               " serves one sequence");
    if (m.draft_vocab)
      throw std::runtime_error("runtime::Engine: a draft vocabulary beside sequence slots is not built");
  }
  return slots;
}

// Spec 32e: where capture_wide builds the 16-, 32- and 64-row lists - DecodeScratchDims::rows' conditions and the
// int8 head (gemm_sm_i8w's wide builds; the bf16 head has none). Only there is the decode scratch sized for 64 rows.
bool wide_lists(const loader::LoadedModel& m, KvCache kv, size_t seqs) {
  return DecodeScratchDims::rows(*m.desc, m.mtp != nullptr, kv, uint32_t(seqs), m.draft_vocab != nullptr) ==
             DecodeScratchDims::kDeepM &&
         m.linears.at({loader::kTopLevel, model::LinearId::LmHead}).kind == model::WeightKind::Int8;
}
}  // namespace

Engine::Engine(l0::Context& ctx, loader::LoadedModel model, uint32_t max_len, bool debug_resid,
               KvCache kv, TapDrafter* drafter, std::vector<uint32_t> slot_lens, uint32_t pool_pages)
    : ctx_(ctx),
      model_(std::move(model)),
      persist_(ctx, checked_max_len(model_, max_len), *model_.desc, kv),
      decode_scratch_(ctx, persist_.max_len, *model_.desc,
                      wide_lists(model_, kv, checked_slots(model_, max_len, slot_lens, pool_pages).size()) ? DecodeScratchDims::kDeepM
                                                                                               : DecodeScratchDims::kWideM),
      buffers_(persist_, decode_scratch_),
      own_{.tap = debug_resid ? std::make_unique<l0::Mem>(
                                    ctx, l0::MemKind::Device, EngineDims::sizes(*model_.desc, kv, 0, 1, false, false, true).tap)
                              : nullptr,
            .df_slots = nullptr, .gdn_pend = nullptr, .gdn_x = nullptr, .conv_x = nullptr},
      mtp_(model_.mtp ? std::make_unique<MtpBuffers>(
                            ctx, persist_.max_len, *model_.desc,
                            model_.draft_vocab ? model_.draft_vocab->size() : 0u,   // spec 8 §11
                            kv, MtpBuffers::kSlots,
                            static_cast<uint32_t>(checked_slots(model_, max_len, slot_lens, pool_pages).size()))   // spec 13c
                      : nullptr),
      step_(build(ctx, model_, buffers_, own_.tap.get())),
      queue_(ctx),
      fence_(queue_),
      imm_(l0::CmdList::immediate(ctx)),
      control_(buffers_.control.as<Control>()) {
  const std::vector<uint32_t> lens = checked_slots(model_, max_len, std::move(slot_lens), pool_pages);
  // Spec 39: the drafter's slots and the deferred GDN rows are made exactly when the plan counts them.
  const EngineSizes es = EngineDims::sizes(*model_.desc, kv, model_.draft_vocab ? model_.draft_vocab->size() : 0u,
                                           uint32_t(lens.size()), mtp_ != nullptr, drafter != nullptr, debug_resid);
  if (drafter) {
    // Spec 19c: one proposer per engine, and the drafter's verify lists at M = 1 .. max_k + 1 over its 8 slots.
    if (mtp_)
      throw std::runtime_error("runtime::Engine: a tap drafter and the MTP head are two proposers - load the "
                               "model without its head (one proposer per server)");
    if (drafter->max_k() == 0 || drafter->max_k() + 1 > TapVerifyDims::kSlots)
      throw std::runtime_error("runtime::Engine: the drafter drafts " + std::to_string(drafter->max_k()) +
                               " ids a block; the verify lists take 1.." +
                               std::to_string(TapVerifyDims::kSlots - 1));
    drafter_ = drafter;
    // Plan 19v3b: slot 0's 7 speculative states, then every further slot's SlotDims::kDrafterStates (setup_slots).
    own_.df_slots = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, es.df_slots);
  }
  setup_slots(lens, pool_pages);
  {
    // One slot with a proposer on a DEFERRED build's shapes (PERF-51: Ornith's too) - EngineSizes::gdn_pend.
    if (es.gdn_pend) {
      own_.gdn_pend = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, es.gdn_pend);
      seqs_[0].bind.gdn_pend = own_.gdn_pend->ptr();
      seqs_[0].tv.gdn_pend = own_.gdn_pend.get();
      gdn_flush_ = std::make_unique<CapturedStep>(build_gdn_flush(ctx, model_, buffers_, own_.gdn_pend->ptr()));
    }
  }
  if (drafter) {
    // Plan 19v3b: a drafter session per slot; slot 0's verify lists here, the further slots' in capture_slot.
    if (slots() > 1) {
      drafter->add_sessions(ctx, slots());
      rows_bind_.taps = &drafter->rows_taps();   // the row lists copy every row's taps
      rows_bind_.tap_layers = drafter->tap_layers();
    }
    Seq& z = seqs_[0];
    z.tv.gdn_spec = z.bind.gdn_spec ? nullptr : own_.df_slots.get();   // several slots: the bound pool (setup_slots)
    z.tv.slots = TapVerifyDims::kSlots;
    z.tv.taps = &drafter->taps();
    z.tv.layers = drafter->tap_layers();
    for (uint32_t M = 1; M <= drafter->max_k() + 1; ++M) capture_verify(M, {});
  }
  if (mtp_) {
    // Spec 8: the draft lists (one per draft index, each writing its own logits row)
    // and the verify lists at M = 1..kSlots. Only the head's model carries these.
    hctl_ = mtp_->hctl.as<Control>();
    seqs_[0].hctl = hctl_;
    for (uint32_t i = 0; i < kMaxDraft; ++i)   // plan 19v3b Task 0: slot 0 of several binds its KV pages
      seqs_[0].draft_steps.push_back(build_draft(ctx, model_, buffers_, *mtp_, i, seqs_[0].bind));
    // Open-work 13: the same lists ending with the draft's probability, for the serial server's per-step K (an
    // engine of one slot: several slots run under the scheduler, whose K stays per request).
    if (slots() == 1)
      for (uint32_t i = 0; i < kMaxDraft; ++i)
        seqs_[0].draft_prob_steps.push_back(build_draft(ctx, model_, buffers_, *mtp_, i, {}, true));
    for (uint32_t M = 1; M <= MtpBuffers::kSlots; ++M) capture_verify(M, {});
    capture_wide({});
  }
  // Spec 13c: every further slot's own lists, then the row lists; then every slot reset (slot 0 last: selected).
  // Plan 19v3b Task 0: slot 0 of several has its own paged plain list too (step_ stays the one-slot engine's).
  if (slots() > 1) seqs_[0].plain.push_back(build(ctx, model_, buffers_, nullptr, nullptr, 1, {}, seqs_[0].bind));
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

void Engine::setup_slots(const std::vector<uint32_t>& lens, uint32_t pool_pages) {
  const model::ModelDesc& d = *model_.desc;
  const uint32_t S = static_cast<uint32_t>(lens.size());
  const size_t state = persist_.gdn_state.size(), ring = persist_.conv_ring.size();
  const size_t hid = size_t(d.hidden) * 2;
  seqs_.resize(S);
  for (uint32_t q = 0; q < S; ++q) {
    seqs_[q].len = lens[q];
    layout_.slots.push_back({lens[q], mtp_ != nullptr});
    seqs_[q].base = layout_.kv_base(q);   // plan 19v3b Task 0: with several slots, a page boundary
  }
  // The GDN states a slot holds: MTP's kSlots, a drafter's 8 in slot 0 and kDrafterStates in a further slot, else 1;
  // `first` numbers them as the row lists' pool does (index i >= 1 at pool + (i - 1) whole states).
  for (uint32_t q = 0; q < S; ++q) {
    Seq& s = seqs_[q];
    s.states = mtp_ ? MtpBuffers::kSlots : drafter_ ? (q == 0 ? TapVerifyDims::kSlots : SlotDims::kDrafterStates) : 1;
    s.first = q == 0 ? 0 : seqs_[q - 1].first + seqs_[q - 1].states;
  }
  Seq& z = seqs_[0];   // slot 0: the buffers' own state, today's binding (bind all null but, of several, its pages)
  z.control = control_;
  z.state0 = persist_.gdn_state.as<uint8_t>();
  z.ring = persist_.conv_ring.as<uint8_t>();
  if (mtp_) {
    z.hctl = mtp_->hctl.as<Control>();
    z.spec = mtp_->gdn_spec.as<uint8_t>();
    z.hh = mtp_->hh.as<uint8_t>();
  }
  if (drafter_) z.spec = own_.df_slots->as<uint8_t>();
  if (S == 1) return;
  // Plan 19v3b Task 0: the KV page table - the identity, written once: slot q's pages are its own range. Spec 31, a
  // pool: every sequence's region of kv_pages(cap) entries (its base, layout_'s), all page 0 until it takes pages.
  const uint32_t n_pages = pool_pages != 0 ? layout_.total() / kKvPage : persist_.max_len / kKvPage;
  // The further slots' GDN states live in MtpBuffers::gdn_spec (MTP) or own_.df_slots (a drafter): SlotBuffers holds none.
  slot_bufs_ = std::make_unique<SlotBuffers>(ctx_, d, S, mtp_ != nullptr || drafter_ != nullptr, n_pages);
  pages_.resize(n_pages);
  for (uint32_t p = 0; p < n_pages; ++p) pages_[p] = pool_pages != 0 ? 0 : p;
  if (pool_pages != 0) pool_ = std::make_unique<KvPool>(pool_pages, S, kv_pages(lens[0]));
  imm_.copy(slot_bufs_->pages.ptr(), pages_.data(), size_t(n_pages) * 4);
  const uint32_t* table = slot_bufs_->pages.as<uint32_t>();
  z.bind.pages = table;
  // GDN index i >= 1 lives at pool + (i - 1) whole states: with MTP MtpBuffers::gdn_spec (slot 0's three, then every
  // further slot's four - MtpDims' `seqs`), without SlotBuffers::states (one a further slot).
  // Plan 19v3b: beside a drafter own_.df_slots - slot 0's 7, then each further slot's 4.
  uint8_t* pool = mtp_      ? mtp_->gdn_spec.as<uint8_t>()
                  : drafter_ ? own_.df_slots->as<uint8_t>()
                             : slot_bufs_->states->as<uint8_t>();
  if (drafter_) z.bind.gdn_spec = z.spec;   // slot 0's verify lists bind the pool (its first 7 states)
  for (uint32_t q = 1; q < S; ++q) {
    Seq& s = seqs_[q];
    s.control = reinterpret_cast<Control*>(slot_bufs_->control_block(SlotDims::control_at(S, q)));
    s.state0 = pool + (size_t(s.first) - 1) * state;
    s.ring = slot_bufs_->rings.as<uint8_t>() + size_t(q - 1) * ring;
    if (s.states > 1) s.spec = s.state0 + state;
    if (mtp_) {
      s.hctl = reinterpret_cast<Control*>(slot_bufs_->control_block(SlotDims::head_control_at(S, q)));
      s.hh = mtp_->hh.as<uint8_t>() + size_t(q) * (DecodeScratchDims::kM + 1) * hid;
    }
    s.bind.control = s.control;
    s.bind.gdn_state = s.state0;
    s.bind.gdn_spec = s.spec;
    s.bind.conv_ring = s.ring;
    s.bind.kv_base = s.base;
    s.bind.hctl = s.hctl;
    s.bind.hh = s.hh;
    s.bind.pages = table + s.base / kKvPage;
  }
  // The row lists' bindings (capture.h, build_rows): their own Control blocks and tables, the pools above, and with
  // MTP the gather over MtpBuffers::hh - every slot's rows (slot q's h_{pos-1} at row q x (kM + 1)), then the saved
  // rows, then the head's input rows (MtpDims::hh_rows).
  BatchRowsBind& b = rows_bind_;
  b.rows = slot_bufs_->control_block(SlotDims::rows_table_at(S));
  b.control = slot_bufs_->control_block(SlotDims::rows_control_at(S));
  b.state_pool = pool;
  b.ring_pool = slot_bufs_->rings.ptr();
  b.pages = table;
  draft_bind_.pages = table;
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
  if (drafter_) {   // plan 19v3b: its verify lists over its 4 states and its session's taps, M = 1 .. 4
    s.tv.slots = s.states;
    s.tv.taps = &drafter_->taps();
    s.tv.layers = drafter_->tap_layers();
    for (uint32_t M = 1; M <= max_draft() + 1; ++M) capture_verify(M, {});
  }
  if (!mtp_) return;
  for (uint32_t i = 0; i < kMaxDraft; ++i) s.draft_steps.push_back(build_draft(ctx_, model_, buffers_, *mtp_, i, s.bind));
  for (uint32_t M = 1; M <= MtpBuffers::kSlots; ++M) capture_verify(M, {});
}

void Engine::select(uint32_t slot) {
  cur_ = slot;
  control_ = seqs_[slot].control;
  hctl_ = seqs_[slot].hctl;
  if (drafter_) drafter_->select(slot);   // plan 19v3b: the drafter's calls act on the slot's session
}

void Engine::zero_slot() {
  // Spec 13c: exactly the selected slot's persistent state - what reset() zeroes for one slot.
  Seq& s = seq();
  // Spec 31: what the sequence wrote (PERF-90's bound, as reset() takes it for one slot), read before Control is zeroed.
  const uint32_t written = s.control->pos + kKvAhead;
  const size_t state = persist_.gdn_state.size();
  imm_.fill(s.control, 0u, sizeof(Control));
  imm_.fill(s.state0, 0u, state);
  imm_.fill(s.ring, 0u, persist_.conv_ring.size());
  if (mtp_) {
    imm_.fill(s.hctl, 0u, sizeof(Control));
    imm_.fill(s.spec, 0u, state * (MtpBuffers::kSlots - 1));
    imm_.fill(s.hh, 0u, size_t(DecodeScratchDims::kM + 1) * model_.desc->hidden * 2);
  }
  if (drafter_) {   // plan 19v3b: its speculative states and its drafter session
    imm_.fill(s.spec, 0u, state * (s.states - 1));
    drafter_->reset(imm_);
  }
  // Its KV range, every layer (K, V and at int8 the scales; the head's layer too) - plan 19v3b Task 0: its pages.
  // Spec 31: a pool's sequence gives its pages back - only their rows below its written bound are zeroed (the rest are
  // still zero), and its table entries point at page 0 again.
  std::vector<std::pair<uint32_t, uint32_t>> rows;
  if (pool_) {
    const uint32_t held = uint32_t(pool_->pages_of(cur_).size());
    const uint32_t bound = uint32_t(std::min<uint64_t>(std::max(s.kv_end, written), uint64_t(held) * kKvPage));
    for (const KvPool::Zero& z : pool_->release(cur_, bound)) rows.push_back({z.page * kKvPage, z.rows});
    s.kv_end = 0;
    if (held != 0) {
      const uint32_t first = s.base / kKvPage;
      std::fill(pages_.begin() + first, pages_.begin() + first + held, 0u);
      imm_.copy(slot_bufs_->pages.as<uint32_t>() + first, pages_.data() + first, size_t(held) * 4);
    }
  } else {
    rows = kv_rows(s, 0, s.len);
  }
  const auto range = [&](const KvLayout& L, l0::Mem& k, l0::Mem& v) {
    for (uint32_t l = 0; l < L.layers; ++l)
      for (const auto& [row, n] : rows) {
        const KvLayer x = L.at_base(L.layer(k.ptr(), v.ptr(), l), row);
        imm_.fill(x.k, 0u, size_t(n) * L.row_bytes());
        imm_.fill(x.v, 0u, size_t(n) * L.row_bytes());
        if (x.int8()) {
          imm_.fill(x.ks, 0u, size_t(n) * L.scale_row_bytes());
          imm_.fill(x.vs, 0u, size_t(n) * L.scale_row_bytes());
        }
      }
  };
  range(persist_.kv_lay, persist_.kv_k, persist_.kv_v);
  if (mtp_) range(mtp_->kv_lay, mtp_->kv_k, mtp_->kv_v);
}

std::vector<std::pair<uint32_t, uint32_t>> Engine::kv_rows(const Seq& s, uint32_t begin, uint32_t end) const {
  std::vector<std::pair<uint32_t, uint32_t>> runs;
  if (pages_.empty()) {   // one slot: its positions are the rows
    if (begin < end) runs.push_back({s.base + begin, end - begin});
    return runs;
  }
  for (uint32_t p = begin; p < end;) {
    const uint32_t n = std::min(end, (p / kKvPage + 1) * kKvPage) - p;   // to the page's end
    const uint32_t row = pages_[(s.base + p) / kKvPage] * kKvPage + p % kKvPage;
    if (!runs.empty() && runs.back().first + runs.back().second == row)
      runs.back().second += n;
    else
      runs.push_back({row, n});
    p += n;
  }
  return runs;
}

uint32_t Engine::kv_row(uint32_t slot, uint32_t p) const {
  return kv_rows(seqs_[slot], p, p + 1).at(0).first;
}

bool Engine::take_pages_of(uint32_t slot, uint32_t upto) {
  contract_assert(pool_ != nullptr);
  const uint32_t before = uint32_t(pool_->pages_of(slot).size());
  if (!pool_->take(slot, upto)) return false;
  const std::vector<uint32_t>& t = pool_->pages_of(slot);
  if (t.size() == before) return true;
  const uint32_t first = seqs_.at(slot).base / kKvPage + before;
  std::copy(t.begin() + before, t.end(), pages_.begin() + first);
  imm_.copy(slot_bufs_->pages.as<uint32_t>() + first, pages_.data() + first, (t.size() - before) * 4);
  return true;
}

void Engine::need_pages(uint32_t slot, uint32_t upto, const char* what) {
  if (!pool_ || take_pages_of(slot, upto)) return;
  throw std::runtime_error(std::string("runtime::Engine::") + what + ": the KV pool has no free page for slot " +
                           std::to_string(slot) + "'s positions up to " + std::to_string(upto) + " (" +
                           std::to_string(pool_->pages_total()) + " pages, " + std::to_string(pool_->pages_free()) +
                           " free) - nothing was written");
}

void Engine::set_kv_page_table(const std::vector<uint32_t>& table) {
  std::vector<bool> seen(pages_.size(), false);
  for (uint32_t p : table) {
    contract_assert(p < seen.size() && !seen[p]);   // a permutation of the KV's pages
    seen[p] = true;
  }
  pages_ = table;
  imm_.copy(slot_bufs_->pages.ptr(), pages_.data(), pages_.size() * 4);
}

void Engine::verify_rows(const std::vector<Run>& runs) {
  flush_projection();   // plan 19v3b: the rows' tap buffer is rewritten below
  RowRuns rr;
  uint32_t M = 0;
  for (const Run& r : runs) {
    const Seq& s = seqs_.at(r.slot);
    const uint32_t pos = s.control->pos;
    if (r.k > max_draft_of(s) || (mtp_ && pos == 0) || size_t(pos) + r.k + 1 > s.len)
      throw std::runtime_error("runtime::Engine::verify_rows: slot " + std::to_string(r.slot) + ": k " +
                               std::to_string(r.k) + " at pos " + std::to_string(pos) + " (k <= " +
                               std::to_string(max_draft_of(s)) + ", pos >= 1 with the MTP head, pos + k + 1 <= its " +
                               std::to_string(s.len) + " positions)");
    for (uint32_t j = 0; j <= r.k; ++j)
      if (s.control->cur_token[j] >= Qwen35::kVocab)
        throw std::runtime_error("runtime::Engine::verify_rows: slot " + std::to_string(r.slot) + "'s id " +
                                 std::to_string(s.control->cur_token[j]) + " in row " + std::to_string(j) +
                                 " is outside the vocabulary");
    need_pages(r.slot, pos + r.k + 1, "verify_rows");   // spec 31: the rows' KV (main and head)
    RowRun x;
    x.slot = r.slot;
    x.pos = pos;
    x.rows = r.k + 1;
    const uint32_t live = s.control->gdn_live;
    x.src = gdn_index(r.slot, live);
    for (uint32_t j = 0; j <= r.k; ++j) x.dst.push_back(gdn_index(r.slot, (live + j) % s.states));   // verify()'s
    rr.push_back(x);
    M += x.rows;
  }
  if (M < 2 || M > kMaxRows)
    throw std::runtime_error("runtime::Engine::verify_rows: " + std::to_string(M) + " rows; a row step takes 2.." +
                             std::to_string(kMaxRows) + " (one slot's lone step is its own verify / generate)");
  const RowTable t = make_rows(layout_, rr);   // check_rows: one run a slot, every rule
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
  // Per run: committable as after its own verify(k) - row j's hidden is the saved row r0 + j.
  const size_t hid = size_t(model_.desc->hidden) * 2;
  uint32_t r0 = 0;
  for (const RowRun& x : rr) {
    Seq& s = seqs_[x.slot];
    s.verify_k = x.rows - 1;
    s.verify_pos = x.pos;
    s.verify_hidden = mtp_ ? static_cast<const uint8_t*>(rows_bind_.saved) + size_t(r0) * hid : nullptr;
    s.from_rows = true;
    s.rows_row0 = r0;
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
    need_pages(runs[m].slot, pos + K, "draft_rows");   // spec 31: the head's rows pos - 1 .. pos + K - 2
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
  return static_cast<const Control*>(rows_bind_.control)->out_token;
}
const float* Engine::rows_logits_device() const { return buffers_.logits.as<float>(); }

// --- the single-sequence calls, on the selected slot -----------------------------------------------------------------

void Engine::capture_verify(uint32_t M, const CaptureOptions& opt) {
  Seq& s = seq();
  auto one = [&](const CaptureOptions& o) {
    return drafter_ ? build_verify_taps(ctx_, model_, buffers_, s.tv, M, o, s.bind)
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

// Open-work 28: the wide list where it is built - one slot with the deferred GDN rows, the bf16 KV cache, the int8 head
// (gemm_sm_i8w's 16-row build; the bf16 head has none) and Qwen3.8's attention heads (PERF-56: or Ornith's). One list
// for every depth: the 16-row attention is DPAS from the first key (runtime/verify_attn.h).
static_assert(MtpDims::kGdnPendParity == kernels::kGdnPendParityBytes &&
                  MtpDims::kGdnConvXLayer == kernels::kGdnConvXLayerBytes,
              "MtpDims' deep terms are the kernels' pending rows and conv scratch");
static_assert(MtpDims::kGdnPendParityOrnith == kernels::gdn_pend_parity_bytes(30, 16, 32) &&
                  MtpDims::kGdnConvXLayerOrnith == kernels::gdn_conv_x_layer_bytes(8192),
              "MtpDims' Ornith pending rows and conv scratch are the kernels'");
void Engine::capture_wide(const CaptureOptions& opt) {
  const model::ModelDesc& d = *model_.desc;
  if (!own_.gdn_pend || !wide_lists(model_, kv_cache(), slots())) return;
  // A probe's per-launch events (CaptureOptions::profile) go to the list at profile_m rows.
  const auto prof = [&](uint32_t M) { return opt.profile_m == M ? opt.profile : nullptr; };
  seqs_[0].wide = std::make_unique<CapturedStep>(
      build_verify(ctx_, model_, buffers_, *mtp_, DecodeScratchDims::kWideM, prof(16), opt, seqs_[0].bind));
  // Spec 32: the 32- and 64-row lists, their chain scratch and the ring copy (allocated once; a recapture reuses them).
  if (!own_.gdn_x) {
    // PERF-56: the ring copy at the model's conv channels (Qwen3.8 10240, Ornith 8192)
    const uint32_t conv = d.gdn_conv_dim();
    const EngineSizes es = EngineDims::sizes(d, kv_cache(), 0, 1, true, false);   // wide_lists: one slot, the head
    own_.gdn_x = std::make_unique<l0::Mem>(ctx_, l0::MemKind::Device, es.gdn_x);
    own_.conv_x = std::make_unique<l0::Mem>(ctx_, l0::MemKind::Device, es.conv_x);
    ring_mod_ = std::make_unique<l0::Module>(ctx_, kernels::path(kernels::gdn_ring_commit_variant(conv)));
    ring_k_ = std::make_unique<l0::Kernel>(*ring_mod_, "gdn_ring_commit");
    ring_k_->group_size(256);
    ring_k_->arg_ptr(0, control_);
    ring_k_->arg_ptr(1, seqs_[0].ring);
    ring_k_->arg_ptr(2, own_.conv_x->ptr());
    ring_list_ = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    ring_list_->launch(*ring_k_, conv / 256, 3, d.gdn_layers);
    ring_list_->close();
    if (const char* v = std::getenv("B70_SPEC_LOOKUP_ROWS")) {
      const std::string r = v;
      if (r != "16" && r != "32" && r != "64")
        throw std::runtime_error("B70_SPEC_LOOKUP_ROWS=" + r + ": 16, 32 or 64 (the widest lookup verify)");
      deep_rows_ = uint32_t(std::stoul(r));
    }
  }
  SlotBind b = seqs_[0].bind;
  b.gdn_x = own_.gdn_x->ptr();
  b.conv_x = own_.conv_x->ptr();
  const auto t0 = std::chrono::steady_clock::now();
  seqs_[0].deep32 = std::make_unique<CapturedStep>(build_verify(ctx_, model_, buffers_, *mtp_, 32, prof(32), opt, b));
  const auto t1 = std::chrono::steady_clock::now();
  seqs_[0].deep64 = std::make_unique<CapturedStep>(build_verify(ctx_, model_, buffers_, *mtp_, 64, prof(64), opt, b));
  deep_capture_ms_[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  deep_capture_ms_[1] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
}

uint32_t Engine::max_wide_k() const {
  const uint32_t pos = control_->pos;
  if (!seq().wide || pos == 0) return 0;
  for (const uint32_t k : {kDeepK64, kDeepK32, kWideK})   // spec 32e: 64 rows up to kDeep64MaxKeys (verify_attn.h)
    if ((k == kWideK || deep_built(k)) && k + 1 <= deep_rows_ && size_t(pos) + k + 1 <= max_len() &&
        (k != kDeepK64 || pos + 1 <= kDeep64MaxKeys))
      return k;
  return 0;
}

void Engine::recapture_verify(const CaptureOptions& opt) {
  flush_projection();
  profile_projection_ = opt.profile_projection;
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
  if (seqs_[0].wide) capture_wide(opt);
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
  seq().pending = nullptr;
  drop_rows_commits();
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
  // PERF-90: the KV at the positions written since the last zero only.
  note_kv(control_->pos + kKvAhead);
  persist_.zero(imm_, kv_end_);
  if (mtp_) mtp_->zero(imm_, kv_end_);
  kv_end_ = 0;
  // Spec 19c: the drafter's slots (as MtpBuffers::zero does the head's) and a cold drafter.
  own_.zero(imm_);   // spec 39: EngineSizes' zero_on_reset buffers - the drafter's slots
  if (drafter_) drafter_->reset(imm_);
  control_->gdn_pend = 0;
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
  need_pages(cur_, uint32_t(pos + n), "replay");   // spec 31
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

uint32_t Engine::max_draft_of(const Seq& s) const {
  return mtp_ ? kMaxDraft : drafter_ ? std::min(drafter_->max_k(), s.states - 1) : 0;
}
uint32_t Engine::max_draft() const { return max_draft_of(seq()); }

void Engine::drop_rows_commits() {
  rows_commits_.erase(std::remove_if(rows_commits_.begin(), rows_commits_.end(),
                                     [&](const TapDrafter::RowsCommit& c) { return c.slot == cur_; }),
                      rows_commits_.end());
}

void Engine::replay_aux(l0::CmdList& list) {
  flush_projection();
  queue_.execute(list, &fence_);
  fence_.wait();
}

void Engine::flush_projection() const {
  for (const Seq& s : seqs_) {   // plan 19v3b: every slot's - each reads only its own session
    if (!s.pending) continue;
    queue_.execute(*s.pending, &fence_);
    fence_.wait();
    s.pending = nullptr;
  }
  if (!rows_commits_.empty()) {   // plan 19v3b: every committed run of the row steps, one replay
    queue_.execute(drafter_->rows_projection(rows_commits_), &fence_);
    fence_.wait();
    rows_commits_.clear();
  }
}

uint32_t Engine::max_verify_k() const {
  const uint32_t pos = control_->pos;
  // Spec 19c: a drafter's verify needs no h_{pos-1}, so pos 0 drafts too.
  // Spec 31: with a pool, no further than the pages the sequence holds and the free ones.
  const uint64_t room = pool_ ? std::min<uint64_t>(max_len(), (uint64_t(pages_held(cur_)) + pages_free()) * kKvPage)
                              : max_len();
  if ((!mtp_ && !drafter_) || (mtp_ && pos == 0) || pos + 1 >= room) return 0;
  return uint32_t(std::min<uint64_t>(max_draft(), room - pos - 1));
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

void Engine::draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick,
                   const std::function<bool(uint32_t, float)>& more) {
  require_mtp("draft");
  Seq& s = seq();
  if (more && s.draft_prob_steps.empty())
    throw std::runtime_error("runtime::Engine::draft: no draft lists with the probability on this slot (draft_probs())");
  const uint32_t pos = control_->pos;
  if (k == 0 || k > kMaxDraft || k > max_verify_k())
    throw std::runtime_error("runtime::Engine::draft: k " + std::to_string(k) +
                             " is outside [1, max_verify_k() = " + std::to_string(max_verify_k()) +
                             "] at pos " + std::to_string(pos));
  need_pages(cur_, pos + k, "draft");   // spec 31: the head's rows pos - 1 .. pos + k - 2
  // The chain's first hidden is h_{pos-1} (hh row 0); each draft list writes its own
  // output hidden back into dh. Draft i runs at position pos - 1 + i (P0 §1).
  imm_.copy(mtp_->dh.ptr(), s.hh, hid_bytes(*model_.desc));
  hctl_->pos = pos - 1;
  hctl_->n_active = 1;
  hctl_->cur_token[0] = control_->cur_token[0];
  s.draft_ids.clear();
  for (uint32_t i = 0; i < k; ++i) {
    queue_.execute((more ? s.draft_prob_steps : s.draft_steps)[i].list, &fence_);
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
    if (more) {
      float norm;
      std::memcpy(&norm, &hctl_->draft_stat[1], 4);
      if (!more(i, 1.0f / norm)) break;
    }
  }
}

bool Engine::draft_probs() const { return mtp_ && !seq().draft_prob_steps.empty(); }

void Engine::verify(uint32_t k, l0::CmdList* draft) {
  require_spec("verify");
  if (!draft || !rows_commits_.empty()) flush_projection();
  Seq& s = seq();
  const uint32_t pos = control_->pos;
  // Open-work 28: the 16-row list; spec 32: the 32- and 64-row lists.
  CapturedStep* const wl = draft ? nullptr
                           : k == kWideK  ? s.wide.get()
                           : k == kDeepK32 ? s.deep32.get()
                           : k == kDeepK64 ? s.deep64.get()
                                           : nullptr;
  const bool wide = wl != nullptr;
  if ((k > max_draft() && !wide) || (mtp_ && pos == 0) || size_t(pos) + k + 1 > max_len())
    throw std::runtime_error("runtime::Engine::verify: k " + std::to_string(k) + " at pos " +
                             std::to_string(pos) + " (max_verify_k() = " +
                             std::to_string(max_verify_k()) + "; k <= " + std::to_string(max_draft()) +
                             ", pos >= 1 with the MTP head, and pos + k + 1 <= max_len " +
                             std::to_string(max_len()) + ")");
  need_pages(cur_, pos + k + 1, "verify");   // spec 31: the rows' KV, before anything of the step is written
  // df_select writes the other rows on the device in the batched path.
  // Open-work 28: the wide list's rows are Control::wide_cur (its embed_gather's CTRL_CUR 32), row 0 the pending id.
  uint32_t* const ids = !wide ? control_->cur_token : k == kWideK ? control_->wide_cur : control_->deep_cur;
  if (wide) ids[0] = control_->cur_token[0];
  for (uint32_t r = 0; r <= (draft ? 0 : k); ++r)
    if (ids[r] >= Qwen35::kVocab)
      throw std::runtime_error("runtime::Engine::verify: id " + std::to_string(ids[r]) + " in row " +
                               std::to_string(r) + " is outside the vocabulary");
  control_->n_active = k + 1;
  if (mtp_) {
    // The head's KV fill of the same rows at pos - 1 .. pos + k - 1: row r is the pair
    // (hh[r], cur_token[r]) = (h_{pos-1+r}, x[pos+r]) - capture.h, build_verify.
    hctl_->pos = pos - 1;
    hctl_->n_active = k + 1;
    uint32_t* const hids = !wide ? hctl_->cur_token : k == kWideK ? hctl_->wide_cur : hctl_->deep_cur;
    for (uint32_t r = 0; r <= k; ++r) hids[r] = ids[r];
  }
  s.verify_pos = pos;
  s.verify_k = k;
  s.from_rows = false;
  s.verify_hidden = mtp_ ? s.hh + hid_bytes(*model_.desc) : nullptr;   // hh row 1 + j: row j's hidden
  // Spec 28c: the deep list (DPAS bound) once row 0's keys reach the table's threshold.
  CapturedStep& list = wide ? *wl
                       : s.verify_deep[k] && pos + 1 >= s.verify_deep_from[k] ? *s.verify_deep[k]
                                                                              : s.verify_steps[k];
  if (draft) {
    ze_command_list_handle_t lists[3];
    uint32_t count = 0;
    if (s.pending) lists[count++] = (profile_projection_ ? profile_projection_ : s.pending)->handle();
    lists[count++] = draft->handle();
    lists[count++] = list.list.handle();
    queue_.execute(count, lists, &fence_);
  } else {
    queue_.execute(list.list, &fence_);
  }
  fence_.wait();
  s.pending = nullptr;
  control_->gdn_pend &= ~0xFFu;   // the list replayed the committed rows: a second replay must not (gdn_step.cl, DEFERRED)
}

void Engine::commit(uint32_t j, uint32_t next_token, bool defer_projection) {
  Seq& s = seq();
  // Spec 13c: a slot of a verify_rows step commits without a proposer too (its plain row, j = 0).
  if (s.verify_k == kNoVerify || (!mtp_ && !drafter_ && slots() == 1)) require_spec("commit");
  if (s.verify_k == kNoVerify || j > s.verify_k)
    throw std::runtime_error("runtime::Engine::commit: j " + std::to_string(j) +
                             (s.verify_k == kNoVerify ? " with no verify() pending"
                                                      : " exceeds the last verify's k " +
                                                            std::to_string(s.verify_k)));
  // Row j's GDN state is slot (live + j) % slots (gdn_step.cl SPEC_SLOTS; spec 13c's row
  // lists write the same indices): make it live. Row j's hidden is the next iteration's
  // h_{pos-1}.
  // verify-cost-m8: with the pending rows, rows 0..j are what the next launch replays - they sit in the parity the
  // verify wrote, the other one (gdn_step.cl, DEFERRED).
  if (own_.gdn_pend)
    control_->gdn_pend = (j + 1) | ((((control_->gdn_pend >> 8) & 1u) ^ 1u) << 8);
  else
    control_->gdn_live = (control_->gdn_live + j) % gdn_slots();
  if (mtp_) imm_.copy(s.hh, s.verify_hidden + size_t(j) * hid_bytes(*model_.desc), hid_bytes(*model_.desc));
  control_->pos = s.verify_pos + j + 1;
  control_->n_active = 1;
  control_->cur_token[0] = next_token;
  // Spec 32: a deep verify's accepted rows 16.. sit in the conv scratch - the ones the next window reads go to the
  // ring now (gdn_ring_commit.cl), before anything reads it.
  if ((s.verify_k == kDeepK32 || s.verify_k == kDeepK64) && j >= 16) {
    control_->ring_commit = j + 1;
    queue_.execute(*ring_list_, &fence_);
    fence_.wait();
    control_->ring_commit = 0;
  }
  s.verify_k = kNoVerify;
  // Spec 19c: the committed rows' taps (rows 0..j: positions verify_pos .. + j) into the drafter's ring. The
  // projection list reads only the taps and its own control block, so Control's new pos / cur_token above do not
  // reach it; it runs before anything can rewrite the taps (the next verify).
  if (drafter_) {
    if (s.from_rows)   // plan 19v3b: its rows' taps are in the row lists' buffer - the rows projection
      rows_commits_.push_back({cur_, s.rows_row0, j + 1, s.verify_pos});
    else
      s.pending = &drafter_->projection(s.verify_pos, j + 1);
    if (!defer_projection) flush_projection();
  }
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
    mtp_normalise_live();
    replay();
    imm_.copy(seq().hh, buffers_.x.ptr(), hid_bytes(*model_.desc));
    return;
  }
  verify(0);
  commit(0, control_->out_token[0]);
}

void Engine::flush_gdn() const {
  if (!own_.gdn_pend || (control_->gdn_pend & 0xFFu) == 0) return;
  const uint32_t n = control_->n_active;
  control_->n_active = 0;
  queue_.execute(gdn_flush_->list, &fence_);
  fence_.wait();
  control_->gdn_pend &= ~0xFFu;
  control_->n_active = n;
}

const void* Engine::live_gdn_slot() const {
  flush_gdn();
  const uint32_t live = (mtp_ || drafter_) ? control_->gdn_live : 0;
  const Seq& s = seq();
  if (live == 0) return s.state0;
  return s.spec + size_t(live - 1) * persist_.gdn_state.size();
}

void Engine::mtp_normalise_live() {
  flush_gdn();
  const uint32_t live = control_->gdn_live;
  if ((!mtp_ && !drafter_) || live == 0) return;
  imm_.copy(seq().state0, live_gdn_slot(), persist_.gdn_state.size());
  control_->gdn_live = 0;
}

// --- spec 7 snapshots -----------------------------------------------------------

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

void Engine::snap_copy(void* host, const void* dev, size_t bytes) const {
  if (!snap_bcs_) snap_bcs_ = std::make_unique<l0::CmdList>(l0::CmdList::immediate_copy(ctx_));
  snap_bcs_->copy(host, dev, bytes);
  snap_pending_ = true;
}

void Engine::snap_wait() const {
  if (!snap_pending_) return;
  snap_bcs_->host_synchronize();
  snap_pending_ = false;
}

void Engine::save_state(void* host) const {
  flush_projection();
  auto* h = static_cast<uint8_t*>(host);
  const Seq& s = seq();
  // PERF-89: inside a mid-prompt hook with the MTP head the next chunk rewrites the state, the ring and hh row 0, so
  // they are copied in the host layout into the sequence's verify slots (from state 1 on; a prefill keeps slot 0 live,
  // mtp_normalise_live) and go to the host from there on the copy engine.
  if (snap_async_ && mtp_) {
    const size_t gdn = persist_.gdn_state.size(), ring = persist_.conv_ring.size();
    imm_.copy(s.spec, live_gdn_slot(), gdn);
    imm_.copy(s.spec + gdn, s.ring, ring);
    imm_.copy(s.spec + gdn + ring, s.hh, hid_bytes(*model_.desc));
    snap_copy(h, s.spec, state_bytes());
    return;
  }
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
  seq().pending = nullptr;
  drop_rows_commits();
  const auto* h = static_cast<const uint8_t*>(host);
  Seq& s = seq();
  imm_.copy(s.state0, h, persist_.gdn_state.size());
  imm_.copy(s.ring, h + persist_.gdn_state.size(), persist_.conv_ring.size());
  if (mtp_) {
    imm_.copy(s.hh, h + persist_.gdn_state.size() + persist_.conv_ring.size(), hid_bytes(*model_.desc));
    control_->gdn_live = 0;   // the state went into slot 0
    s.verify_k = kNoVerify;
  }
  control_->gdn_pend = 0;   // verify-cost-m8: no committed rows over a loaded state
  if (drafter_) {   // spec 19c: the state went into slot 0; spec 19d: and the ring is the snapshot's
    drafter_->load_ring(imm_, h + persist_.gdn_state.size() + persist_.conv_ring.size());
    control_->gdn_live = 0;
    s.verify_k = kNoVerify;
  }
  note_kv(control_->pos + kKvAhead);   // PERF-90: the rows written up to here stay written
  control_->pos = pos;
  control_->n_active = 0;
}

// kv_k / kv_v are KvLayout's: [fa_layers][max_len][4][256] rows (bf16, or int8 then the
// [fa_layers][max_len][4] fp16 scales). A position range is one contiguous run per layer,
// so fa_layers (16 on Qwen3.8) copies each for K and V, plus as many scale
// runs at int8 (spec 12b). The two loops below walk the same (allocation, offset, bytes)
// list in the same order, one copying out and the other in. Spec 13c: the selected slot's
// positions, from its KV base. Plan 19v3b Task 0: its rows through the page table (Engine::kv_rows) - a layer's
// range one copy per run of consecutive pages, in position order, so the host image is the same whatever the table.
namespace {
struct KvRun {
  uint8_t* dev;
  size_t bytes;
};
// Every device run of the rows `rows` ((first row, rows) runs, position order), in host order: per tensor (K, V),
// the rows of every layer (the head's last), then at int8 the scales the same way.
std::vector<KvRun> kv_runs(const PersistentBuffers& p, const MtpBuffers* mtp,
                           const std::vector<std::pair<uint32_t, uint32_t>>& rows) {
  std::vector<KvRun> runs;
  const KvLayout& L = p.kv_lay;
  for (int kv = 0; kv < 2; ++kv) {
    uint8_t* base = (kv == 0 ? p.kv_k : p.kv_v).as<uint8_t>();
    uint8_t* hbase = mtp ? (kv == 0 ? mtp->kv_k : mtp->kv_v).as<uint8_t>() : nullptr;
    for (uint32_t l = 0; l < L.layers; ++l)
      for (const auto& [begin, n] : rows)
        runs.push_back({base + L.rows_offset(l) + size_t(begin) * L.row_bytes(), size_t(n) * L.row_bytes()});
    if (mtp)   // the head's layer, after the last FA layer
      for (const auto& [begin, n] : rows)
        runs.push_back({hbase + mtp->kv_lay.rows_offset(0) + size_t(begin) * L.row_bytes(), size_t(n) * L.row_bytes()});
    if (L.scale_row_bytes() == 0) continue;
    for (uint32_t l = 0; l < L.layers; ++l)
      for (const auto& [begin, n] : rows)
        runs.push_back({base + L.scales_offset(l) + size_t(begin) * L.scale_row_bytes(),
                        size_t(n) * L.scale_row_bytes()});
    if (mtp)
      for (const auto& [begin, n] : rows)
        runs.push_back({hbase + mtp->kv_lay.scales_offset(0) + size_t(begin) * L.scale_row_bytes(),
                        size_t(n) * L.scale_row_bytes()});
  }
  return runs;
}
}  // namespace

void Engine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  flush_projection();
  if (begin == end) return;
  auto* h = static_cast<uint8_t*>(host);
  const std::vector<KvRun> runs = kv_runs(persist_, mtp_.get(), kv_rows(seq(), begin, end));
  if (!snap_async_) {
    for (const KvRun& r : runs) {
      imm_.copy(h, r.dev, r.bytes);
      h += r.bytes;
    }
    return;
  }
  // PERF-89, inside a mid-prompt hook: later chunks write rows from the hook's end on, so the runs go to the host on
  // the copy engine as they stand - all but the MTP head's row end - 1, which the next chunk's step_mtp_kv rewrites
  // (its own last position): that row is copied now, after save_state's image in the verify slots, and written over
  // its place in the host image after the runs (the list is in order).
  for (const KvRun& r : runs) {
    snap_copy(h, r.dev, r.bytes);
    h += r.bytes;
  }
  const uint32_t at = control_->pos;   // the hook's end
  if (!mtp_ || begin >= at || at > end) return;
  const KvLayout& H = mtp_->kv_lay;   // kv_runs' addressing of the head's row
  const size_t row = kv_rows(seq(), at - 1, at)[0].first, rb = persist_.kv_lay.row_bytes(),
               sb = persist_.kv_lay.scale_row_bytes();
  const std::pair<const uint8_t*, size_t> parts[4] = {
      {mtp_->kv_k.as<uint8_t>() + H.rows_offset(0) + row * rb, rb},
      {mtp_->kv_v.as<uint8_t>() + H.rows_offset(0) + row * rb, rb},
      {mtp_->kv_k.as<uint8_t>() + H.scales_offset(0) + row * sb, sb},
      {mtp_->kv_v.as<uint8_t>() + H.scales_offset(0) + row * sb, sb}};
  uint8_t* stage = seq().spec + state_bytes();
  for (const auto& [dev, bytes] : parts) {
    if (bytes == 0) continue;   // bf16: no scales
    imm_.copy(stage, dev, bytes);
    h = static_cast<uint8_t*>(host);
    for (const KvRun& r : runs) {
      if (dev >= r.dev && dev + bytes <= r.dev + r.bytes) snap_copy(h + (dev - r.dev), stage, bytes);
      h += r.bytes;
    }
    stage += bytes;
  }
}

void Engine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  if (begin == end) return;
  need_pages(cur_, end, "load_kv");   // spec 31
  note_kv(end);   // PERF-90
  const auto* h = static_cast<const uint8_t*>(host);
  for (const KvRun& r : kv_runs(persist_, mtp_.get(), kv_rows(seq(), begin, end))) {
    imm_.copy(r.dev, h, r.bytes);
    h += r.bytes;
  }
}

std::vector<uint16_t> Engine::read_debug_resid() {
  if (!own_.tap)
    throw std::runtime_error(
        "runtime::Engine::read_debug_resid: this engine was constructed with debug_resid=false,"
        " so no per-layer tap was captured");
  std::vector<uint16_t> out(tap_elems(*model_.desc));
  imm_.copy(out.data(), own_.tap->ptr(), out.size() * 2);
  return out;
}

}  // namespace runtime

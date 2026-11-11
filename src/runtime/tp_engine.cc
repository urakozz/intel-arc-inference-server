#include "runtime/tp_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "model/qwen35.h"
#include "runtime/pipeline_place.h"
#include "runtime/tp_engine_card.h"
#include "runtime/tp_gather.h"
#include "runtime/verify_attn.h"

namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

const char* status_name(uint32_t s) {
  return s == tp::kTimeout ? "timed out (the flag never reached the call)"
                           : s == tp::kStale ? "stale (the flag came without its data's trailer)" : "failed";
}
}  // namespace

TpEngine::Card& TpEngine::cd(uint32_t card) const {
  if (card >= model::kTpSize)
    throw std::out_of_range("runtime::TpEngine: card " + std::to_string(card) + " (two are built)");
  return *cards_[card];
}

TpEngine::TpEngine(l0::Context& d0, l0::Context& d1, std::array<loader::LoadedModel, 2> cards, uint32_t max_len,
                   const TpOptions& opt, KvCache kv)
    : max_len_(max_len), opt_(opt), kv_(kv), pf_cap_(default_prefill_chunk_cap()) {
  if (d1.handle() != d0.handle())
    throw std::invalid_argument("runtime::TpEngine: card 1 is not a view of card 0's Level Zero context");
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0))
    throw std::runtime_error("--parallel tp: the two cards cannot both access each other's memory "
                             "(zeDeviceCanAccessPeer 0 -> 1 / 1 -> 0), and every crossing writes the other card's "
                             "VRAM. Peer access needs the P2P-capable kernel and both cards under one root complex "
                             "(docs/10-the-box.md)");
  const model::ModelDesc* desc = cards[0].desc;
  if (!desc || cards[1].desc != desc)
    throw std::invalid_argument("runtime::TpEngine: the two cards' models are not one TP descriptor");
  if (desc == &model::tp_card_desc(model::qwen38(), model::TpCrossing::Slices))
    crossing_ = model::TpCrossing::Slices;
  else if (desc == &model::tp_card_desc(model::qwen38(), model::TpCrossing::Folded))
    crossing_ = model::TpCrossing::Folded;
  else
    throw std::invalid_argument("runtime::TpEngine: " + desc->name + " is not a TP card's descriptor (loader::load_tp)");
  for (const loader::LoadedModel& m : cards) {
    if (m.max_len != max_len)
      throw std::runtime_error("runtime::TpEngine: a card's model was loaded with max_len " + std::to_string(m.max_len) +
                               " but the engine was asked for " + std::to_string(max_len));
    if (m.draft_vocab) throw std::invalid_argument("runtime::TpEngine: no draft vocabulary under TP (spec 17d)");
  }
  // Spec 17d: the MTP head on both cards or on neither; the folded crossing (the verify lists' binaries).
  if (bool(cards[0].mtp) != bool(cards[1].mtp))
    throw std::invalid_argument("runtime::TpEngine: the MTP head is loaded on one card only");
  mtp_ = cards[0].mtp != nullptr;
  if (mtp_ && crossing_ != model::TpCrossing::Folded)
    throw std::invalid_argument("runtime::TpEngine: MTP under tensor parallel is built for the folded crossing "
                                "(runtime::kTpDefaultCrossing); the slices form runs plain decode");
  if (opt.timeout_ms == 0 || opt.spin_limit == 0)
    throw std::invalid_argument("runtime::TpEngine: a crossing needs non-zero bounds");
  layout_ = tp::layout(*desc, crossing_);
  const bool folded = crossing_ == model::TpCrossing::Folded;
  cards_[0] = std::make_unique<Card>(d0, std::move(cards[0]), max_len, kv, layout_, folded, opt.debug_resid);
  cards_[1] = std::make_unique<Card>(d1, std::move(cards[1]), max_len, kv, layout_, folded, opt.debug_resid);
  const uint32_t rows = cd(0).model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    Card& peer = cd(1 - r);
    TpLink link;
    link.rank = r;
    link.folded = folded;
    link.peer_land = peer.land.ptr();
    link.land = c.land.ptr();
    link.send_ctr = c.send_ctr.ptr();
    link.fold_ctr = c.fold_ctr.ptr();
    link.own = c.own ? c.own->ptr() : nullptr;
    link.state = c.state.ptr();
    link.spin_limit = opt.spin_limit;
    // The card's real rows: card 0's half is all real, card 1's ends at the tokenizer's count (argmax.cl's mask).
    const uint32_t first = r * rows;
    link.vocab_used = desc->vocab_used > first ? std::min(rows, desc->vocab_used - first) : 0;
    // Spec 17d: the head's descriptor - the one-card model's (the head is whole on each card; tp_card_desc is cut
    // from model::qwen38(), the one model TP is built for).
    link.head = mtp_ ? &model::qwen38() : nullptr;
    c.link = link;
    c.step.emplace(build_tp(c.ctx, c.model, c.buffers, link, c.tap.get()));
    if (mtp_) capture_mtp_lists(r);
  }
  reset();
}

// Spec 17d: a card's MTP buffers and lists - the draft lists i = 0..kMaxDraft - 1 and the verify lists at
// M = 1..kSlots, each M captured twice where spec 28c's table switches its attention by depth (Engine::capture_verify).
void TpEngine::capture_mtp_lists(uint32_t card) {
  Card& c = cd(card);
  c.mtp = std::make_unique<MtpBuffers>(c.ctx, max_len_, *c.model.desc, model::qwen38(), kv_);
  c.hctl = c.mtp->hctl.as<Control>();
  for (uint32_t i = 0; i < kMaxDraft; ++i) c.drafts.push_back(build_tp_draft(c.ctx, c.model, c.buffers, *c.mtp, c.link, i));
  for (uint32_t i = 0; i < kMaxDraft; ++i)   // open-work 13: the same with the card's half of the draft probability
    c.draft_probs.push_back(build_tp_draft(c.ctx, c.model, c.buffers, *c.mtp, c.link, i, true));
  const model::ModelDesc& d = *c.model.desc;
  for (uint32_t M = 1; M <= MtpBuffers::kSlots; ++M) {
    c.verifies.push_back(build_tp_verify(c.ctx, c.model, c.buffers, *c.mtp, c.link, M));
    const uint32_t from = verify_attn_deep_from(M, kv_ == KvCache::Int8, d.fa_q_heads, d.fa_kv_heads);
    c.verify_deep.push_back(from != kDpasNever ? std::make_unique<CapturedStep>(build_tp_verify(
                                                     c.ctx, c.model, c.buffers, *c.mtp, c.link, M,
                                                     attn_forced(VerifyAttnForce::Dpas)))
                                               : nullptr);
    c.verify_deep_from.push_back(from);
  }
}

TpEngine::~TpEngine() {
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    try {
      if (cards_[r]) settle(r);
    } catch (...) {
    }
  }
}

bool TpEngine::settle(uint32_t card) {
  if (!pending_[card]) return true;
  if (cd(card).fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull)) pending_[card] = false;
  return !pending_[card];
}

void TpEngine::reset() {
  // A failed step may have left a list running (a wait up to its spin bound): it finishes before its buffers move.
  for (uint32_t r = 0; r < model::kTpSize; ++r)
    if (!settle(r))
      throw std::runtime_error("runtime::TpEngine::reset: card " + std::to_string(r) + "'s last step is still running "
                               "after " + std::to_string(opt_.timeout_ms) + " ms");
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    cd(r).persist.zero(cd(r).imm);
    cd(r).zero_crossing();
    if (cd(r).mtp) cd(r).mtp->zero(cd(r).imm);   // spec 17d: the head's KV, slots, hidden rows and Control
  }
  broken_ = false;
  drop_ = -1;
  verify_k_ = kNoVerify;
  draft_ids_.clear();
}

void TpEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("tensor parallel: " + what + ". The session's state is not valid; reset() before the next "
                           "step");
}

void TpEngine::step_once() {
  const size_t pos = cd(0).ctl->pos, n = cd(0).ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::TpEngine: pos " + std::to_string(pos) + " + n_active " + std::to_string(n) +
                             " exceeds max_len " + std::to_string(max_len_) +
                             " - the KV cache and the RoPE table stop there. Start a new session with reset().");
  run_lists({&cd(0).step->list, &cd(1).step->list}, true, "step");
}

void TpEngine::run_lists(const std::array<l0::CmdList*, model::kTpSize>& lists, bool crossings, const char* what) {
  if (broken_) throw std::runtime_error("tensor parallel: a crossing failed earlier in this session; reset() first");
  const int drop = drop_;
  drop_ = -1;
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    if (int(r) == drop) continue;
    cd(r).queue.execute(*lists[r], &cd(r).fence);
    pending_[r] = true;
  }
  const std::string dropped =
      drop >= 0 ? " [card " + std::to_string(drop) + "'s list was not submitted: drop_next()]" : std::string();
  for (uint32_t r = 0; r < model::kTpSize; ++r)
    if (!settle(r))
      fail("card " + std::to_string(r) + "'s " + what + " did not finish within " + std::to_string(opt_.timeout_ms) +
           " ms" + dropped);
  if (!crossings) {
    if (drop >= 0) fail("card " + std::to_string(drop) + "'s " + what + " list was not submitted (drop_next())");
    return;
  }
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    const uint32_t* w = cd(r).state.as<uint32_t>();
    if (w[tp::kStatus] != tp::kOk)
      fail("card " + std::to_string(r) + "'s wait for card " + std::to_string(1 - r) + "'s part " +
           status_name(w[tp::kStatus]) + ": call " + std::to_string(w[tp::kWant]) + ", chunk " +
           std::to_string(w[tp::kChunk]) + ", flag " + std::to_string(w[tp::kFlag]) + ", trailer " +
           std::to_string(w[tp::kStamp]) + "; " + std::to_string(w[tp::kFailed]) + " failed chunk(s)" + dropped);
  }
  if (drop >= 0) fail("card " + std::to_string(drop) + "'s list was not submitted (drop_next()) but no wait failed");
  merge_heads();
}

void TpEngine::merge_heads() {
  Control& a = *cd(0).ctl;
  Control& b = *cd(1).ctl;
  if (a.pos != b.pos || a.n_active != b.n_active)
    fail("the two Controls disagree after the step (pos " + std::to_string(a.pos) + " / " + std::to_string(b.pos) + ")");
  // The head's merge (D2): argmax.cl's comparator over the two halves' winners.
  for (uint32_t m = 0; m < a.n_active; ++m) {
    float va, vb;
    std::memcpy(&va, &a.out_value[m], 4);
    std::memcpy(&vb, &b.out_value[m], 4);
    const uint32_t ia = a.out_token[m], ib = b.out_token[m];
    const bool b_wins = (vb > va) || (vb == va && ib < ia);
    a.out_token[m] = b.out_token[m] = b_wins ? ib : ia;
  }
  a.cur_token[0] = b.cur_token[0] = a.out_token[a.n_active - 1];
}

void TpEngine::set_token(uint32_t id) {
  if (id >= model::Qwen35::kVocab)
    throw std::runtime_error("runtime::TpEngine::set_token: id " + std::to_string(id) + " is outside the vocabulary");
  cd(0).ctl->cur_token[0] = id;
  cd(1).ctl->cur_token[0] = id;
}

uint32_t TpEngine::pending() const { return cd(0).ctl->cur_token[0]; }

void TpEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (uint32_t r = 0; r < model::kTpSize; ++r) cd(r).ctl->n_active = 1;
    set_token(id);
    if (mtp_)
      mtp_step1();
    else
      step_once();
  }
}

std::vector<uint32_t> TpEngine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  for (uint32_t r = 0; r < model::kTpSize; ++r) cd(r).ctl->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = cd(0).ctl->cur_token[0];   // == card 1's after the merge
    const Clock::time_point f0 = Clock::now();
    if (mtp_)
      mtp_step1();
    else
      step_once();
    fence_ms += ms_since(f0);
    out.push_back(id);
    if (on_token) on_token(id);
  }
  last_gen_ms_ = ms_since(t0);
  last_fence_ms_ = fence_ms;
  last_tok_per_s_ = (n != 0 && last_gen_ms_ > 0.0) ? double(n) * 1000.0 / last_gen_ms_ : 0.0;
  return out;
}

uint32_t TpEngine::pos() const { return cd(0).ctl->pos; }
const CapturedStep& TpEngine::step(uint32_t card) const { return *cd(card).step; }
DecodeBuffers& TpEngine::buffers(uint32_t card) { return cd(card).buffers; }
const Control& TpEngine::control(uint32_t card) const { return *cd(card).ctl; }
const loader::LoadedModel& TpEngine::model(uint32_t card) const { return cd(card).model; }
l0::Context& TpEngine::context(uint32_t card) const { return cd(card).ctx; }

// --- spec 7's host layouts of one card ----------------------------------------------------------------------------

namespace {
size_t hid_bytes(const model::ModelDesc& d) { return size_t(d.hidden) * 2; }
}  // namespace

size_t TpEngine::state_bytes() const {
  size_t t = 0;
  for (uint32_t r = 0; r < model::kTpSize; ++r) t += cd(r).persist.gdn_state.size() + cd(r).persist.conv_ring.size();
  return t + (mtp_ ? hid_bytes(*cd(0).model.desc) : 0);   // spec 17d: the head's h_{pos-1}
}

size_t TpEngine::kv_bytes(uint32_t n_pos) const {
  // Spec 17d: with MTP the head's layer, at its own kv heads (the head is whole on each card).
  const size_t head = mtp_ ? 2 * cd(0).mtp->kv_lay.pos_bytes() * size_t(n_pos) : 0;
  return model::kTpSize * 2 * cd(0).persist.kv_lay.pos_bytes() * size_t(cd(0).model.desc->fa_layers) * n_pos + head;
}

void TpEngine::save_state(void* host) const {
  std::array<std::vector<uint8_t>, 2> gdn, conv;
  const uint8_t* gp[2];
  const uint8_t* cp[2];
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    const PersistentBuffers& p = cd(r).persist;
    gdn[r].resize(p.gdn_state.size());
    conv[r].resize(p.conv_ring.size());
    // Spec 17d: the LIVE GDN slot (after a commit it may be one of the head's speculative slots, Engine's rule).
    cd(r).imm.copy(gdn[r].data(), cd(r).live_gdn(), p.gdn_state.size());
    cd(r).imm.copy(conv[r].data(), p.conv_ring.ptr(), p.conv_ring.size());
    gp[r] = gdn[r].data();
    cp[r] = conv[r].data();
  }
  tp::gather_state(*cd(0).model.desc, gp, cp, static_cast<uint8_t*>(host));
  if (mtp_) {   // the head's hidden row, card 0's (the cards' are the same bytes)
    const size_t off = state_bytes() - hid_bytes(*cd(0).model.desc);
    cd(0).imm.copy(static_cast<uint8_t*>(host) + off, cd(0).mtp->hh.ptr(), hid_bytes(*cd(0).model.desc));
  }
}

void TpEngine::load_state(const void* host, uint32_t pos) {
  if (pos > max_len_)
    throw std::runtime_error("runtime::TpEngine::load_state: pos " + std::to_string(pos) + " exceeds max_len " +
                             std::to_string(max_len_));
  std::array<std::vector<uint8_t>, 2> gdn, conv;
  uint8_t* gp[2];
  uint8_t* cp[2];
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    gdn[r].resize(cd(r).persist.gdn_state.size());
    conv[r].resize(cd(r).persist.conv_ring.size());
    gp[r] = gdn[r].data();
    cp[r] = conv[r].data();
  }
  const auto* h = static_cast<const uint8_t*>(host);
  tp::scatter_state(*cd(0).model.desc, h, gp, cp);
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    c.imm.copy(c.persist.gdn_state.ptr(), gdn[r].data(), gdn[r].size());   // slot 0
    c.imm.copy(c.persist.conv_ring.ptr(), conv[r].data(), conv[r].size());
    if (mtp_) {
      c.imm.copy(c.mtp->hh.ptr(), h + state_bytes() - hid_bytes(*c.model.desc), hid_bytes(*c.model.desc));
      c.ctl->gdn_live = 0;   // the state went into slot 0
    }
    c.ctl->pos = pos;
    c.ctl->n_active = 0;
  }
  verify_k_ = kNoVerify;
}

namespace {
// One card's KV runs of positions [begin, end) in Engine::save_kv's per-card layout (at its own heads): per tensor
// every layer's rows, then (int8) every layer's scales. `head`: the MTP head's one layer the same way.
struct KvRun {
  uint8_t* dev;
  size_t bytes;
};
std::vector<KvRun> card_kv_runs(const l0::Mem& k, const l0::Mem& v, const KvLayout& L, uint32_t begin, uint32_t end) {
  std::vector<KvRun> runs;
  const size_t n = end - begin;
  for (const l0::Mem* t : {&k, &v}) {
    for (uint32_t l = 0; l < L.layers; ++l)
      runs.push_back({t->as<uint8_t>() + L.rows_offset(l) + size_t(begin) * L.row_bytes(), n * L.row_bytes()});
    if (L.form == KvCache::Int8)
      for (uint32_t l = 0; l < L.layers; ++l)
        runs.push_back(
            {t->as<uint8_t>() + L.scales_offset(l) + size_t(begin) * L.scale_row_bytes(), n * L.scale_row_bytes()});
  }
  return runs;
}
}  // namespace

void TpEngine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  if (begin > end || end > max_len_)
    throw std::runtime_error("runtime::TpEngine::save_kv: range [" + std::to_string(begin) + ", " +
                             std::to_string(end) + ") is not within [0, max_len " + std::to_string(max_len_) + ")");
  if (begin == end) return;
  const uint32_t n = end - begin;
  std::array<std::vector<uint8_t>, 2> card;
  const uint8_t* ptrs[2];
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    const PersistentBuffers& p = cd(r).persist;
    card[r].resize(2 * size_t(p.kv_lay.layers) * n * p.kv_lay.pos_bytes());
    uint8_t* h = card[r].data();
    for (const KvRun& run : card_kv_runs(p.kv_k, p.kv_v, p.kv_lay, begin, end)) {
      cd(r).imm.copy(h, run.dev, run.bytes);
      h += run.bytes;
    }
    ptrs[r] = card[r].data();
  }
  const model::ModelDesc& d = *cd(0).model.desc;
  if (!mtp_) {
    tp::gather_kv(d, kv_, n, ptrs, static_cast<uint8_t*>(host));
    return;
  }
  // Spec 17d: the FA layers gathered, the head's layer (card 0's) spliced in after them - Engine's MTP layout.
  const MtpBuffers& m = *cd(0).mtp;
  const tp::KvSplit ks = tp::kv_split(d.fa_layers, model::kTpSize * d.fa_kv_heads, m.kv_lay.kv_heads, kv_, n);
  std::vector<uint8_t> main(2 * ks.main_tensor()), head(2 * ks.head_tensor());
  tp::gather_kv(d, kv_, n, ptrs, main.data());
  uint8_t* h = head.data();
  for (const KvRun& run : card_kv_runs(m.kv_k, m.kv_v, m.kv_lay, begin, end)) {
    cd(0).imm.copy(h, run.dev, run.bytes);
    h += run.bytes;
  }
  tp::splice_head_kv(ks, main.data(), head.data(), static_cast<uint8_t*>(host));
}

void TpEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  if (begin > end || end > max_len_)
    throw std::runtime_error("runtime::TpEngine::load_kv: range [" + std::to_string(begin) + ", " +
                             std::to_string(end) + ") is not within [0, max_len " + std::to_string(max_len_) + ")");
  if (begin == end) return;
  const uint32_t n = end - begin;
  const model::ModelDesc& d = *cd(0).model.desc;
  const uint8_t* main = static_cast<const uint8_t*>(host);
  std::vector<uint8_t> main_buf, head;
  if (mtp_) {
    const MtpBuffers& m = *cd(0).mtp;
    const tp::KvSplit ks = tp::kv_split(d.fa_layers, model::kTpSize * d.fa_kv_heads, m.kv_lay.kv_heads, kv_, n);
    main_buf.resize(2 * ks.main_tensor());
    head.resize(2 * ks.head_tensor());
    tp::split_head_kv(ks, main, main_buf.data(), head.data());
    main = main_buf.data();
  }
  std::array<std::vector<uint8_t>, 2> card;
  uint8_t* ptrs[2];
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    card[r].resize(2 * size_t(cd(r).persist.kv_lay.layers) * n * cd(r).persist.kv_lay.pos_bytes());
    ptrs[r] = card[r].data();
  }
  tp::scatter_kv(d, kv_, n, main, ptrs);
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    const uint8_t* h = card[r].data();
    for (const KvRun& run : card_kv_runs(c.persist.kv_k, c.persist.kv_v, c.persist.kv_lay, begin, end)) {
      c.imm.copy(run.dev, h, run.bytes);
      h += run.bytes;
    }
    if (mtp_) {   // the head's layer on both cards
      const uint8_t* hh = head.data();
      for (const KvRun& run : card_kv_runs(c.mtp->kv_k, c.mtp->kv_v, c.mtp->kv_lay, begin, end)) {
        c.imm.copy(run.dev, hh, run.bytes);
        hh += run.bytes;
      }
    }
  }
}

// --- spec 17d: MTP ------------------------------------------------------------------------------------------------

void TpEngine::require_mtp(const char* what) const {
  if (!mtp_)
    throw std::runtime_error(std::string("runtime::TpEngine::") + what +
                             ": MTP is off - load both cards with the head (loader::load_tp's mtp, --spec mtp)");
}

uint32_t TpEngine::max_verify_k() const {
  const uint32_t pos = cd(0).ctl->pos;
  if (!mtp_ || pos == 0 || pos + 1 >= max_len_) return 0;
  return std::min<uint32_t>(kMaxDraft, max_len_ - pos - 1);
}

const uint32_t* TpEngine::verify_ids() const { return cd(0).ctl->out_token; }

void TpEngine::set_draft_input(uint32_t i, uint32_t id) {
  if (i >= kMaxDraft || id >= model::Qwen35::kVocab)
    throw std::runtime_error("runtime::TpEngine::set_draft_input: draft " + std::to_string(i) + ", id " +
                             std::to_string(id));
  for (uint32_t r = 0; r < model::kTpSize; ++r) cd(r).ctl->cur_token[1 + i] = id;
}

void TpEngine::draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick,
                     const std::function<bool(uint32_t, float)>& more) {
  require_mtp("draft");
  const uint32_t pos = cd(0).ctl->pos;
  if (k == 0 || k > kMaxDraft || k > max_verify_k())
    throw std::runtime_error("runtime::TpEngine::draft: k " + std::to_string(k) + " is outside [1, max_verify_k() = " +
                             std::to_string(max_verify_k()) + "] at pos " + std::to_string(pos));
  // Engine::draft per card: the chain's first hidden is h_{pos-1} (hh row 0); draft i runs at pos - 1 + i.
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    c.imm.copy(c.mtp->dh.ptr(), c.mtp->hh.ptr(), hid_bytes(*c.model.desc));
    c.hctl->pos = pos - 1;
    c.hctl->n_active = 1;
    c.hctl->cur_token[0] = c.ctl->cur_token[0];
  }
  draft_ids_.clear();
  for (uint32_t i = 0; i < k; ++i) {
    if (more)
      run_lists({&cd(0).draft_probs[i].list, &cd(1).draft_probs[i].list}, false, "draft list");
    else
      run_lists({&cd(0).drafts[i].list, &cd(1).drafts[i].list}, false, "draft list");
    // The head's merge (D2's rule over the two vocabulary halves' winners), into both head Controls.
    Control& a = *cd(0).hctl;
    Control& b = *cd(1).hctl;
    float va, vb;
    std::memcpy(&va, &a.out_value[0], 4);
    std::memcpy(&vb, &b.out_value[0], 4);
    uint32_t d = (vb > va) || (vb == va && b.out_token[0] < a.out_token[0]) ? b.out_token[0] : a.out_token[0];
    if (pick) {
      a.out_token[0] = b.out_token[0] = d;   // what the device drafted, for a sampler that reads it
      d = pick(i);
      if (d >= model::Qwen35::kVocab)
        throw std::runtime_error("runtime::TpEngine::draft: picked id " + std::to_string(d) +
                                 " is outside the vocabulary");
    }
    a.out_token[0] = b.out_token[0] = d;
    a.cur_token[0] = b.cur_token[0] = d;   // the next draft list's input
    draft_ids_.push_back(d);
    for (uint32_t r = 0; r < model::kTpSize; ++r) cd(r).ctl->cur_token[1 + i] = d;
    if (more) {
      // Open-work 13: the two halves' softmax (max, normaliser) merged - the head's probability for the merged winner.
      float ma, sa, mb, sb;
      std::memcpy(&ma, &a.draft_stat[0], 4);
      std::memcpy(&sa, &a.draft_stat[1], 4);
      std::memcpy(&mb, &b.draft_stat[0], 4);
      std::memcpy(&sb, &b.draft_stat[1], 4);
      const float m = std::max(ma, mb);
      if (!more(i, 1.0f / (sa * std::exp(ma - m) + sb * std::exp(mb - m)))) break;
    }
  }
}

bool TpEngine::draft_probs() const { return mtp_ && !cd(0).draft_probs.empty(); }

void TpEngine::verify(uint32_t k) {
  require_mtp("verify");
  const uint32_t pos = cd(0).ctl->pos;
  if (k > kMaxDraft || pos == 0 || size_t(pos) + k + 1 > max_len_)
    throw std::runtime_error("runtime::TpEngine::verify: k " + std::to_string(k) + " at pos " + std::to_string(pos) +
                             " (max_verify_k() = " + std::to_string(max_verify_k()) + "; k <= " +
                             std::to_string(kMaxDraft) + ", pos >= 1, and pos + k + 1 <= max_len " +
                             std::to_string(max_len_) + ")");
  for (uint32_t r = 0; r <= k; ++r)
    if (cd(0).ctl->cur_token[r] >= model::Qwen35::kVocab || cd(1).ctl->cur_token[r] != cd(0).ctl->cur_token[r])
      throw std::runtime_error("runtime::TpEngine::verify: row " + std::to_string(r) + "'s id " +
                               std::to_string(cd(0).ctl->cur_token[r]) +
                               " is outside the vocabulary or differs between the cards");
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    c.ctl->n_active = k + 1;
    // The head's KV fill of the same rows at pos - 1 .. pos + k - 1 (Engine::verify).
    c.hctl->pos = pos - 1;
    c.hctl->n_active = k + 1;
    for (uint32_t m = 0; m <= k; ++m) c.hctl->cur_token[m] = c.ctl->cur_token[m];
  }
  verify_pos_ = pos;
  verify_k_ = k;
  // Spec 28c: the deep list (DPAS bound) once row 0's keys reach the table's threshold - the same M on both cards.
  std::array<l0::CmdList*, model::kTpSize> lists{};
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    lists[r] = c.verify_deep[k] && pos + 1 >= c.verify_deep_from[k] ? &c.verify_deep[k]->list : &c.verifies[k].list;
  }
  run_lists(lists, true, "verify step");
}

void TpEngine::commit(uint32_t j, uint32_t next_token) {
  require_mtp("commit");
  if (verify_k_ == kNoVerify || j > verify_k_)
    throw std::runtime_error("runtime::TpEngine::commit: j " + std::to_string(j) +
                             (verify_k_ == kNoVerify ? " with no verify() pending"
                                                     : " exceeds the last verify's k " + std::to_string(verify_k_)));
  if (next_token >= model::Qwen35::kVocab)
    throw std::runtime_error("runtime::TpEngine::commit: next_token " + std::to_string(next_token) +
                             " is outside the vocabulary");
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    const size_t row = hid_bytes(*c.model.desc);
    c.ctl->gdn_live = (c.ctl->gdn_live + j) % MtpBuffers::kSlots;   // row j's state live (Engine::commit)
    c.imm.copy(c.mtp->hh.ptr(), c.mtp->hh.as<uint8_t>() + (1 + size_t(j)) * row, row);   // row j's hidden
    c.ctl->pos = verify_pos_ + j + 1;
    c.ctl->n_active = 1;
    c.ctl->cur_token[0] = next_token;
  }
  verify_k_ = kNoVerify;
}

void TpEngine::mtp_normalise_live() {
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    Card& c = cd(r);
    if (!c.mtp || c.ctl->gdn_live == 0) continue;
    c.imm.copy(c.persist.gdn_state.ptr(), c.live_gdn(), c.persist.gdn_state.size());
    c.ctl->gdn_live = 0;
  }
}

void TpEngine::mtp_step1() {
  if (cd(0).ctl->pos == 0) {
    // No h_{-1}: the plain list, which reads and writes slot 0, then its hidden into hh row 0 (Engine::mtp_step1).
    mtp_normalise_live();
    step_once();
    for (uint32_t r = 0; r < model::kTpSize; ++r)
      cd(r).imm.copy(cd(r).mtp->hh.ptr(), cd(r).buffers.x.ptr(), hid_bytes(*cd(r).model.desc));
    return;
  }
  verify(0);
  commit(0, verify_ids()[0]);
}

void TpEngine::read_logits_into(float* host, uint32_t rows) const {
  if (rows == 0 || rows > MtpBuffers::kSlots)
    throw std::runtime_error("runtime::TpEngine::read_logits_into: rows " + std::to_string(rows));
  // D9: the gather - each card's [rows][N] half into the full [rows][kVocab] row.
  const uint32_t N = cd(0).model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  for (uint32_t m = 0; m < rows; ++m)
    for (uint32_t r = 0; r < model::kTpSize; ++r)
      cd(r).imm.copy(host + size_t(m) * model::Qwen35::kVocab + size_t(r) * N,
                     cd(r).buffers.logits.as<float>() + size_t(m) * N, size_t(N) * sizeof(float));
}

void TpEngine::read_draft_logits_into(float* host, uint32_t i) const {
  require_mtp("read_draft_logits_into");
  if (i >= kMaxDraft) throw std::runtime_error("runtime::TpEngine::read_draft_logits_into: row " + std::to_string(i));
  const uint32_t N = cd(0).model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  for (uint32_t r = 0; r < model::kTpSize; ++r)
    cd(r).imm.copy(host + size_t(r) * N, cd(r).mtp->logits.as<float>() + size_t(i) * model::Qwen35::kVocab,
                   size_t(N) * sizeof(float));
}

float* TpEngine::host_rows(size_t floats) {
  if (!host_rows_ || host_rows_->size() < floats * sizeof(float))
    host_rows_ = std::make_unique<l0::Mem>(cd(0).ctx, l0::MemKind::Host, floats * sizeof(float));
  return host_rows_->as<float>();
}

const CapturedStep& TpEngine::verify_step(uint32_t card, uint32_t M) const {
  require_mtp("verify_step");
  return cd(card).verifies.at(M - 1);
}
const CapturedStep& TpEngine::draft_step(uint32_t card, uint32_t i) const {
  require_mtp("draft_step");
  return cd(card).drafts.at(i);
}
const MtpBuffers* TpEngine::mtp_buffers(uint32_t card) const { return cd(card).mtp.get(); }

uint32_t TpEngine::verify_deep_lists() const {
  uint32_t n = 0;
  for (const auto& p : cd(0).verify_deep) n += p != nullptr;
  return n;
}

std::vector<float> TpEngine::read_logits() const {
  const uint32_t rows = cd(0).model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  std::vector<float> out(size_t(rows) * model::kTpSize);
  for (uint32_t r = 0; r < model::kTpSize; ++r)
    cd(r).imm.copy(out.data() + size_t(r) * rows, cd(r).buffers.logits.ptr(), size_t(rows) * sizeof(float));
  return out;
}

std::vector<uint16_t> TpEngine::read_debug_resid(uint32_t card) const {
  const Card& c = cd(card);
  if (!c.tap) throw std::runtime_error("runtime::TpEngine::read_debug_resid: built without debug_resid");
  std::vector<uint16_t> out(c.tap->size() / 2);
  c.imm.copy(out.data(), c.tap->ptr(), c.tap->size());
  return out;
}

MemoryComponents TpEngine::memory_use(uint32_t card) const {
  const Card& c = cd(card);
  MemoryComponents m;
  m.model = model_device_bytes(c.model);
  m.kv = c.persist.kv_k.size() + c.persist.kv_v.size();
  m.decode_state = c.persist.bytes() - m.kv + c.scratch.bytes() + c.bytes_crossing() + (c.tap ? c.tap->size() : 0) +
                   (c.mtp ? c.mtp->bytes() : 0);   // spec 17d: the head's buffers
  m.prefill_scratch = pf_bytes_[card];   // spec 17c: after prepare_prefill()
  m.int8 = pf_int8_bytes_[card];
  return m;
}

std::string TpEngine::memory_line(uint32_t card) const {
  const std::string label = "memory, card " + std::to_string(card);
  return format_memory(label.c_str(), memory_use(card), cd(card).ctx.memory_bytes());
}

}  // namespace runtime

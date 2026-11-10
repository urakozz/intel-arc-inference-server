#include "runtime/tp_engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "model/qwen35.h"
#include "runtime/pipeline_place.h"
#include "runtime/tp_gather.h"

namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

const char* status_name(uint32_t s) {
  return s == tp::kTimeout ? "timed out (the flag never reached the call)"
                           : s == tp::kStale ? "stale (the flag came without its data's trailer)" : "failed";
}
}  // namespace

// One card's half (Engine's members for one device, PipelineEngine::Stage's shape) and its crossing buffers.
struct TpEngine::Card {
  Card(l0::Context& c, loader::LoadedModel m, uint32_t max_len, KvCache kv, const tp::Layout& L, bool folded,
       bool tap_on)
      : ctx(c),
        model(std::move(m)),
        persist(c, max_len, *model.desc, kv),
        scratch(c, max_len, *model.desc),
        buffers(persist, scratch),
        queue(c),
        fence(queue),
        imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()),
        land(c, l0::MemKind::Device, L.total_bytes(), 4096),
        send_ctr(c, l0::MemKind::Device, L.counter_bytes()),
        fold_ctr(c, l0::MemKind::Device, L.counter_bytes()),
        state(c, l0::MemKind::Shared, tp::kStateWords * 4),
        own(folded ? std::make_unique<l0::Mem>(c, l0::MemKind::Device,
                                               size_t(tp::kMRows) * model.desc->hidden * 4)
                   : nullptr),
        tap(tap_on ? std::make_unique<l0::Mem>(c, l0::MemKind::Device,
                                                size_t(model.desc->layers) * model.desc->hidden * 2)
                   : nullptr) {}
  l0::Context& ctx;
  loader::LoadedModel model;
  // Declaration order is construction order: `buffers` binds references into the two above it (Engine's rule).
  PersistentBuffers persist;
  DecodeScratch scratch;
  DecodeBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  l0::Mem land;                    // written by the OTHER card's tp_send, read by this card's tp_res_fold
  l0::Mem send_ctr, fold_ctr;      // this card's call counters (device, advanced by its kernels)
  l0::Mem state;                   // shared: the first failure, read after the fence
  std::unique_ptr<l0::Mem> own;    // folded: this card's folded row
  std::unique_ptr<l0::Mem> tap;    // debug_resid
  std::optional<CapturedStep> step;

  size_t bytes_crossing() const {
    return land.size() + send_ctr.size() + fold_ctr.size() + state.size() + (own ? own->size() : 0);
  }
  void zero_crossing() {
    imm.fill(land.ptr(), 0u, land.size());
    imm.fill(send_ctr.ptr(), 0u, send_ctr.size());
    imm.fill(fold_ctr.ptr(), 0u, fold_ctr.size());
    std::memset(state.ptr(), 0, state.size());
  }
};

TpEngine::Card& TpEngine::cd(uint32_t card) const {
  if (card >= model::kTpSize)
    throw std::out_of_range("runtime::TpEngine: card " + std::to_string(card) + " (two are built)");
  return *cards_[card];
}

TpEngine::TpEngine(l0::Context& d0, l0::Context& d1, std::array<loader::LoadedModel, 2> cards, uint32_t max_len,
                   const TpOptions& opt, KvCache kv)
    : max_len_(max_len), opt_(opt), kv_(kv) {
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
    if (m.mtp || m.draft_vocab) throw std::invalid_argument("runtime::TpEngine: no MTP head under TP before 17d");
  }
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
    c.step.emplace(build_tp(c.ctx, c.model, c.buffers, link, c.tap.get()));
  }
  reset();
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
  }
  broken_ = false;
  drop_ = -1;
}

void TpEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("tensor parallel: " + what + ". The session's state is not valid; reset() before the next "
                           "step");
}

void TpEngine::step_once() {
  if (broken_) throw std::runtime_error("tensor parallel: a crossing failed earlier in this session; reset() first");
  const size_t pos = cd(0).ctl->pos, n = cd(0).ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::TpEngine: pos " + std::to_string(pos) + " + n_active " + std::to_string(n) +
                             " exceeds max_len " + std::to_string(max_len_) +
                             " - the KV cache and the RoPE table stop there. Start a new session with reset().");
  const int drop = drop_;
  drop_ = -1;
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    if (int(r) == drop) continue;
    cd(r).queue.execute(cd(r).step->list, &cd(r).fence);
    pending_[r] = true;
  }
  const std::string dropped =
      drop >= 0 ? " [card " + std::to_string(drop) + "'s list was not submitted: drop_next()]" : std::string();
  for (uint32_t r = 0; r < model::kTpSize; ++r)
    if (!settle(r))
      fail("card " + std::to_string(r) + "'s step did not finish within " + std::to_string(opt_.timeout_ms) + " ms" +
           dropped);
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    const uint32_t* w = cd(r).state.as<uint32_t>();
    if (w[tp::kStatus] != tp::kOk)
      fail("card " + std::to_string(r) + "'s wait for card " + std::to_string(1 - r) + "'s part " +
           status_name(w[tp::kStatus]) + ": call " + std::to_string(w[tp::kWant]) + ", chunk " +
           std::to_string(w[tp::kChunk]) + ", flag " + std::to_string(w[tp::kFlag]) + ", trailer " +
           std::to_string(w[tp::kStamp]) + "; " + std::to_string(w[tp::kFailed]) + " failed chunk(s)" + dropped);
  }
  if (drop >= 0) fail("card " + std::to_string(drop) + "'s list was not submitted (drop_next()) but no wait failed");
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

void TpEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (uint32_t r = 0; r < model::kTpSize; ++r) cd(r).ctl->n_active = 1;
    set_token(id);
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

size_t TpEngine::state_bytes() const {
  size_t t = 0;
  for (uint32_t r = 0; r < model::kTpSize; ++r) t += cd(r).persist.gdn_state.size() + cd(r).persist.conv_ring.size();
  return t;
}

size_t TpEngine::kv_bytes(uint32_t n_pos) const {
  return model::kTpSize * 2 * cd(0).persist.kv_lay.pos_bytes() * size_t(cd(0).model.desc->fa_layers) * n_pos;
}

void TpEngine::save_state(void* host) const {
  std::array<std::vector<uint8_t>, 2> gdn, conv;
  const uint8_t* gp[2];
  const uint8_t* cp[2];
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    const PersistentBuffers& p = cd(r).persist;
    gdn[r].resize(p.gdn_state.size());
    conv[r].resize(p.conv_ring.size());
    cd(r).imm.copy(gdn[r].data(), p.gdn_state.ptr(), p.gdn_state.size());
    cd(r).imm.copy(conv[r].data(), p.conv_ring.ptr(), p.conv_ring.size());
    gp[r] = gdn[r].data();
    cp[r] = conv[r].data();
  }
  tp::gather_state(*cd(0).model.desc, gp, cp, static_cast<uint8_t*>(host));
}

void TpEngine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  if (begin > end || end > max_len_)
    throw std::runtime_error("runtime::TpEngine::save_kv: range [" + std::to_string(begin) + ", " +
                             std::to_string(end) + ") is not within [0, max_len " + std::to_string(max_len_) + ")");
  if (begin == end) return;
  const uint32_t n = end - begin;
  std::array<std::vector<uint8_t>, 2> card;
  const uint8_t* ptrs[2];
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    // The card's bytes in Engine::save_kv's layout at its own heads: per tensor every layer's rows, then (int8)
    // every layer's scales.
    const PersistentBuffers& p = cd(r).persist;
    const KvLayout& L = p.kv_lay;
    card[r].resize(2 * size_t(L.layers) * n * L.pos_bytes());
    uint8_t* h = card[r].data();
    for (const l0::Mem* t : {&p.kv_k, &p.kv_v}) {
      for (uint32_t l = 0; l < L.layers; ++l) {
        cd(r).imm.copy(h, t->as<uint8_t>() + L.rows_offset(l) + size_t(begin) * L.row_bytes(), size_t(n) * L.row_bytes());
        h += size_t(n) * L.row_bytes();
      }
      if (L.form == KvCache::Int8)
        for (uint32_t l = 0; l < L.layers; ++l) {
          cd(r).imm.copy(h, t->as<uint8_t>() + L.scales_offset(l) + size_t(begin) * L.scale_row_bytes(),
                         size_t(n) * L.scale_row_bytes());
          h += size_t(n) * L.scale_row_bytes();
        }
    }
    ptrs[r] = card[r].data();
  }
  tp::gather_kv(*cd(0).model.desc, kv_, n, ptrs, static_cast<uint8_t*>(host));
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
  m.decode_state = c.persist.bytes() - m.kv + c.scratch.bytes() + c.bytes_crossing() + (c.tap ? c.tap->size() : 0);
  return m;
}

std::string TpEngine::memory_line(uint32_t card) const {
  const std::string label = "memory, card " + std::to_string(card);
  return format_memory(label.c_str(), memory_use(card), cd(card).ctx.memory_bytes());
}

}  // namespace runtime

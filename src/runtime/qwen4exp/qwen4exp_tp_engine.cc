#include "runtime/qwen4exp/qwen4exp_tp_engine.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "loader/qwen4exp_ple_hash.h"
#include "model/qwen4exp.h"
#include "runtime/qwen4exp/qwen4exp_capture.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

// qwen4exp_tp_engine.cc - plan 21v2b Task 2: the header has the contract.
namespace runtime::qwen4exp {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
}  // namespace

// One card: its descriptor (the buffers keep a reference to it, so it is declared first), buffers, queue, crossing.
struct Qwen4ExpTpEngine::Card {
  Card(l0::Context& c, const model::Qwen4ExpDesc& one, uint32_t rank, uint32_t max_len)
      : ctx(c),
        desc(model::q4_tp_card(one, rank)),
        buffers(c, desc, model::Q4Placement::one(desc), 0, max_len, Q4Attn::Flash, false),
        queue(c),
        fence(queue),
        imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()),
        lo{one.hidden, kTpOutSlices, tp::kFoldG},
        lm{one.hidden, one.top_k + 1, tp::kFoldG},
        land_o(c, l0::MemKind::Device, lo.total_bytes(), 4096),
        land_m(c, l0::MemKind::Device, lm.total_bytes(), 4096),
        ctr_so(c, l0::MemKind::Device, lo.counter_bytes()),
        ctr_fo(c, l0::MemKind::Device, lo.counter_bytes()),
        ctr_sm(c, l0::MemKind::Device, lm.counter_bytes()),
        ctr_fm(c, l0::MemKind::Device, lm.counter_bytes()),
        slots(c, l0::MemKind::Device, size_t(one.top_k + 1) * one.hidden * 4),
        state(c, l0::MemKind::Shared, tp::kStateWords * 4) {}
  l0::Context& ctx;
  const model::Qwen4ExpDesc desc;
  Qwen4ExpBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  tp::Layout lo, lm;   // out_proj / o_proj's crossing, the MoE's
  l0::Mem land_o, land_m, ctr_so, ctr_fo, ctr_sm, ctr_fm, slots, state;
  TpLink link, moe;
  Q4TpBinding binding;
  std::optional<CapturedStep> step;
  bool pending = false;
};

Qwen4ExpTpEngine::Qwen4ExpTpEngine(l0::Context& d0, l0::Context& d1, loader::Q4LoadedModel model, uint32_t max_len,
                                   const Q4TpOptions& opt)
    : model_(std::move(model)), max_len_(max_len), opt_(opt) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (!model_.placement.tp || model_.parts.size() != 2)
    throw std::invalid_argument("runtime::qwen4exp::Qwen4ExpTpEngine: the model was not loaded at the tp placement");
  if (model_.max_len != max_len)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpTpEngine: the model was loaded with max_len " +
                             std::to_string(model_.max_len) + ", the engine asked for " + std::to_string(max_len));
  if (d1.handle() != d0.handle())
    throw std::invalid_argument("runtime::qwen4exp::Qwen4ExpTpEngine: card 1 is not a view of card 0's context");
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0))
    throw std::runtime_error("--parallel tp: the cards cannot access each other's memory (zeDeviceCanAccessPeer is "
                             "false), and each card's all-reduce writes the other's. Peer access needs the P2P-capable "
                             "kernel and both cards under one root complex (docs/10-the-box.md)");
  if (opt_.timeout_ms == 0 || opt_.spin_limit == 0)
    throw std::invalid_argument("runtime::qwen4exp::Qwen4ExpTpEngine: the waits need non-zero bounds");
  if (!model_.parts[0].embed) throw std::runtime_error("runtime::qwen4exp::Qwen4ExpTpEngine: card 0 holds no embedding");
  cards_[0] = std::make_unique<Card>(d0, d, 0, max_len);
  cards_[1] = std::make_unique<Card>(d1, d, 1, max_len);
  for (uint32_t r = 0; r < 2; ++r) {
    Card& c = *cards_[r];
    Card& peer = *cards_[1 - r];
    for (TpLink* t : {&c.link, &c.moe}) {
      const bool m = t == &c.moe;
      t->rank = r;
      t->folded = false;
      t->peer_land = (m ? peer.land_m : peer.land_o).ptr();
      t->land = (m ? c.land_m : c.land_o).ptr();
      t->send_ctr = (m ? c.ctr_sm : c.ctr_so).ptr();
      t->fold_ctr = (m ? c.ctr_fm : c.ctr_fo).ptr();
      t->state = c.state.ptr();   // one record of the first failure
      t->spin_limit = opt_.spin_limit;
      t->vocab_used = c.desc.vocab_used;
    }
    c.binding.link = &c.link;
    c.binding.moe = &c.moe;
    c.binding.slots = c.slots.ptr();
    c.binding.card = &c.desc;
    // card 1 gathers its embedding row from card 0's table (peer access), unless it holds its own
    c.binding.embed = (model_.parts[r].embed ? model_.parts[r].embed : model_.parts[0].embed)->ptr();
  }
  ple_init();
  for (uint32_t r = 0; r < 2; ++r) cards_[r]->step.emplace(build_card(r, nullptr));
  reset();
}

Qwen4ExpTpEngine::~Qwen4ExpTpEngine() {
  for (auto& c : cards_)
    if (c && c->pending) c->fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull);
}

CapturedStep Qwen4ExpTpEngine::build_card(uint32_t r, ProfileEvents* prof) {
  Card& c = *cards_[r];
  ListSpec spec;
  spec.prof = prof;
  return build(c.ctx, model_, model_.parts[r], c.buffers, nullptr, nullptr, nullptr, spec, nullptr, &c.binding);
}

// The PLE hash constants into both cards' ple_consts (the one-card engine's values; the table's page check is the
// one-card engine's, on card 0's pointer table - card 1's is its copy).
void Qwen4ExpTpEngine::ple_init() {
  const model::Qwen4ExpDesc& d = model_.desc;
  const std::array<uint64_t, 3> mult = loader::q4_ple_multipliers(d.vocab, d.ngram, 0, d.ple_seed);
  const std::vector<uint64_t> sizes = loader::q4_ple_primes(d.ple_base, d.ple_heads, 0);
  const std::vector<uint64_t> offs = loader::q4_ple_offsets(sizes);
  std::vector<uint64_t> consts(kPleConsts, 0);
  for (uint32_t i = 0; i < 3; ++i) consts[i] = mult[i];
  for (uint32_t h = 0; h < d.ple_heads; ++h) {
    consts[3 + h] = sizes[h];
    consts[3 + d.ple_heads + h] = offs[h];
  }
  if (!model_.ple_ptrs1) throw std::runtime_error("runtime::qwen4exp::Qwen4ExpTpEngine: card 1 has no PLE pointer table");
  for (auto& c : cards_) c->imm.copy(c->buffers.ple_consts.ptr(), consts.data(), consts.size() * 8);
}

void Qwen4ExpTpEngine::reset() {
  for (uint32_t r = 0; r < 2; ++r) {
    Card& c = *cards_[r];
    if (c.pending && !c.fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull))
      throw std::runtime_error("runtime::qwen4exp::Qwen4ExpTpEngine::reset: card " + std::to_string(r) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
    c.pending = false;
  }
  for (auto& c : cards_) {
    c->buffers.zero(c->imm);
    for (l0::Mem* m : {&c->land_o, &c->land_m, &c->ctr_so, &c->ctr_fo, &c->ctr_sm, &c->ctr_fm})
      c->imm.fill(m->ptr(), 0u, m->size());
    std::memset(c->state.ptr(), 0, c->state.size());
  }
  broken_ = false;
}

void Qwen4ExpTpEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("tensor parallel: " + what + ". The session's state is not valid; reset() before the next step");
}

void Qwen4ExpTpEngine::submit(CapturedStep* l0p, CapturedStep* l1p) {
  if (broken_) throw std::runtime_error("tensor parallel: a crossing failed earlier in this session; reset() first");
  const Control& c0 = *cards_[0]->ctl;
  if (size_t(c0.pos) + c0.n_active > max_len_)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpTpEngine: pos " + std::to_string(c0.pos) +
                             " exceeds max_len " + std::to_string(max_len_));
  CapturedStep* lists[2] = {l0p, l1p};
  for (uint32_t r = 0; r < 2; ++r) {
    cards_[r]->queue.execute(lists[r]->list, &cards_[r]->fence);
    cards_[r]->pending = true;
  }
  for (uint32_t r = 0; r < 2; ++r) {
    Card& c = *cards_[r];
    if (!c.fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull))
      fail("card " + std::to_string(r) + "'s step did not finish within " + std::to_string(opt_.timeout_ms) + " ms");
    c.pending = false;
  }
  for (uint32_t r = 0; r < 2; ++r) {
    const uint32_t* s = cards_[r]->state.as<uint32_t>();
    if (s[tp::kStatus] != tp::kOk)
      fail("card " + std::to_string(r) + "'s all-reduce " + (s[tp::kStatus] == tp::kTimeout ? "timed out" : "read a stale "
           "chunk") + " (chunk " + std::to_string(s[tp::kChunk]) + ", call " + std::to_string(s[tp::kWant]) + ", flag " +
           std::to_string(s[tp::kFlag]) + ", stamp " + std::to_string(s[tp::kStamp]) + ", " +
           std::to_string(s[tp::kFailed]) + " chunks failed)");
  }
  // The head's merge: the larger value, ties to card 0 (the lower ids).
  Control& a = *cards_[0]->ctl;
  Control& b = *cards_[1]->ctl;
  float va = 0.f, vb = 0.f;
  std::memcpy(&va, &a.out_value[0], 4);
  std::memcpy(&vb, &b.out_value[0], 4);
  const uint32_t id = vb > va ? b.out_token[0] : a.out_token[0];
  for (Control* c : {&a, &b}) {
    c->out_token[0] = id;
    c->cur_token[0] = id;
  }
  if (a.pos != b.pos) fail("the cards' positions differ (" + std::to_string(a.pos) + " / " + std::to_string(b.pos) + ")");
}

void Qwen4ExpTpEngine::set_token(uint32_t id) {
  if (id >= model_.desc.vocab)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpTpEngine: id " + std::to_string(id) +
                             " is outside the vocabulary (" + std::to_string(model_.desc.vocab) + ")");
  for (auto& c : cards_) c->ctl->cur_token[0] = id;
}

void Qwen4ExpTpEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (auto& c : cards_) c->ctl->n_active = 1;
    set_token(id);
    submit(&*cards_[0]->step, &*cards_[1]->step);
  }
}

std::vector<uint32_t> Qwen4ExpTpEngine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  for (auto& c : cards_) c->ctl->n_active = 1;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = cards_[0]->ctl->cur_token[0];
    submit(&*cards_[0]->step, &*cards_[1]->step);
    out.push_back(id);
    if (on_token) on_token(id);
  }
  last_gen_ms_ = ms_since(t0);
  last_tok_per_s_ = n != 0 && last_gen_ms_ > 0.0 ? double(n) * 1000.0 / last_gen_ms_ : 0.0;
  return out;
}

uint32_t Qwen4ExpTpEngine::pending() const { return cards_[0]->ctl->cur_token[0]; }
uint32_t Qwen4ExpTpEngine::pos() const { return cards_[0]->ctl->pos; }
size_t Qwen4ExpTpEngine::launches(uint32_t card) const { return cards_.at(card)->step->kernel_count; }
const CapturedStep& Qwen4ExpTpEngine::step(uint32_t card) const { return *cards_.at(card)->step; }

std::vector<uint16_t> Qwen4ExpTpEngine::read_H(uint32_t card) {
  Card& c = *cards_.at(card);
  std::vector<uint16_t> h(model_.desc.hc_n());
  c.imm.copy(h.data(), c.buffers.H.ptr(), h.size() * 2);
  return h;
}

std::vector<float> Qwen4ExpTpEngine::read_logits() {
  std::vector<float> out(model_.desc.vocab);
  size_t at = 0;
  for (auto& c : cards_) {
    c->imm.copy(out.data() + at, c->buffers.logits.ptr(), size_t(c->desc.vocab) * 4);
    at += c->desc.vocab;
  }
  return out;
}

std::vector<uint32_t> Qwen4ExpTpEngine::read_routes(uint32_t card) {
  Card& c = *cards_.at(card);
  std::vector<uint32_t> r(size_t(model_.desc.layers) * kRouteWords);
  c.imm.copy(r.data(), c.buffers.routes.ptr(), r.size() * 4);
  return r;
}

std::array<uint32_t, tp::kStateWords> Qwen4ExpTpEngine::state_words(uint32_t card) const {
  std::array<uint32_t, tp::kStateWords> s{};
  std::memcpy(s.data(), cards_.at(card)->state.ptr(), sizeof(s));
  return s;
}

std::vector<double> Qwen4ExpTpEngine::profile_decode(uint32_t card, uint32_t replays) {
  ProfileEvents prof(cards_.at(card)->ctx);
  CapturedStep p = build_card(card, &prof);
  std::vector<double> us(p.kernel_count, 0.0);
  for (uint32_t i = 0; i < replays; ++i) {
    for (l0::Event& e : prof.events) e.reset();
    if (card == 0)
      submit(&p, &*cards_[1]->step);
    else
      submit(&*cards_[0]->step, &p);
    for (size_t j = 0; j < us.size(); ++j) us[j] += prof.events[j].duration_us() / replays;
  }
  return us;
}

}  // namespace runtime::qwen4exp

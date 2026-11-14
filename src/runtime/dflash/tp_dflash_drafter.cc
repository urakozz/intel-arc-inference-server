#include "runtime/dflash/tp_dflash_drafter.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/dflash_dims.h"
#include "model/qwen35.h"
#include "runtime/dflash/dflash_sizes.h"

namespace runtime::dflash {

namespace {
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::dflash::TpDFlashDrafter: " + what);
}
// The whole drafter's descriptor from a card's half (model::dflash_tp_card_desc's inverse) - the snapshot key's.
model::DFlashDesc whole_desc(model::DFlashDesc d) {
  d.q_heads *= 2;
  d.kv_heads *= 2;
  d.inter *= 2;
  return d;
}
}  // namespace

TpDFlashDrafter::Half::Half(l0::Context& c, loader::DFlashWeights wt, uint32_t max_len, uint32_t max_k,
                            const CaptureOptions& opt)
    : ctx(c),
      w(std::move(wt)),
      b(c, w.desc, w.target_hidden, w.candidate_head ? w.candidate_head->shape.N : 0, max_len, w.format, true, true),
      k_max(max_k),
      project(build_dflash_project(c, w, b, opt)),
      imm(l0::CmdList::immediate(c)) {}

uint32_t TpDFlashDrafter::Half::window() const {
  uint32_t win = 0;
  for (uint32_t l = 0; l < w.desc.layers; ++l)
    win = std::max(win, w.desc.window[l] ? w.desc.window[l] : std::numeric_limits<uint32_t>::max());
  return win;
}

l0::CmdList& TpDFlashDrafter::Half::projection(uint32_t pos, uint32_t n) {
  require(n >= 1 && n <= kernels::dflash::kM, "a projection takes 1.." + std::to_string(kernels::dflash::kM) +
                                                  " rows, not " + std::to_string(n));
  require(size_t(pos) + n <= size_t(b.max_len) + kernels::dflash::kM,
          "positions " + std::to_string(pos) + " + " + std::to_string(n) + " are past the RoPE table");
  b.ctl()->proj_pos = pos;
  b.ctl()->proj_n = n;
  return project.list;
}

void TpDFlashDrafter::Half::save_ring(l0::CmdList& i, void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  i.copy(h, b.ring_k.ptr(), b.ring_k.size());
  i.copy(h + b.ring_k.size(), b.ring_v.ptr(), b.ring_v.size());
}

void TpDFlashDrafter::Half::load_ring(l0::CmdList& i, const void* host) {
  const auto* h = static_cast<const uint8_t*>(host);
  i.copy(b.ring_k.ptr(), h, b.ring_k.size());
  i.copy(b.ring_v.ptr(), h + b.ring_k.size(), b.ring_v.size());
}

TpDFlashDrafter::TpDFlashDrafter(std::array<l0::Context*, 2> ctx, std::array<loader::DFlashWeights, 2> halves,
                                 uint32_t max_len, uint32_t max_k, const CaptureOptions& opt)
    : opt_(opt) {
  for (uint32_t r = 0; r < 2; ++r) {
    const loader::DFlashWeights& w = halves[r];
    require(ctx[r] != nullptr && w.desc.v2() && w.candidate_head && w.format != loader::DraftFormat::Bf16,
            "card " + std::to_string(r) + "'s half is not a DFlash 2 drafter with the candidate head in int8 or int4 "
            "(the TP lists are built for those)");
    require(max_k >= 1 && max_k <= w.desc.max_k() && max_k <= kernels::dflash::kR,
            "max_k " + std::to_string(max_k) + " is outside 1.." + std::to_string(w.desc.max_k()));
  }
  require(halves[0].desc.q_heads == halves[1].desc.q_heads && halves[0].format == halves[1].format &&
              halves[0].candidate_bits == halves[1].candidate_bits &&
              halves[0].candidates == halves[1].candidates,
          "the two halves are not one drafter's");
  for (uint32_t r = 0; r < 2; ++r)
    half_[r] = std::make_unique<Half>(*ctx[r], std::move(halves[r]), max_len, max_k, opt);
  // The store's name: the whole drafter's (DFlashDrafter's key: its shapes, format and norms) - the snapshot's ring is
  // one card's layout, so a snapshot moves between one card and TP.
  const loader::DFlashWeights& w = half_[0]->w;
  std::vector<uint8_t> norms(w.hidden_norm->size() + w.final_norm->size());
  half_[0]->imm.copy(norms.data(), w.hidden_norm->ptr(), w.hidden_norm->size());
  half_[0]->imm.copy(norms.data() + w.hidden_norm->size(), w.final_norm->ptr(), w.final_norm->size());
  key_ = drafter_key(whole_desc(w.desc), w.format, norms.data(), norms.size());
}

void TpDFlashDrafter::attach(TpEngine& e) {
  require(half_[0]->draft == nullptr, "attach() twice");
  require(e.tap_drafter() == this, "the engine was not built with this drafter (TpEngine(..., &drafter))");
  for (uint32_t r = 0; r < 2; ++r) {
    Half& h = *half_[r];
    require(e.max_len() == h.b.max_len, "the engine's max_len is not the drafter's buffers'");
    const loader::LoadedModel& m = e.model(r);
    DFlashTarget t;
    t.embed = &m.embed;
    t.head = &m.linears.at({loader::kTopLevel, model::LinearId::LmHead});
    t.vocab_rows = model::Qwen35::kVocab;   // the whole embedding; the head is the card's half
    t.vocab_used = m.desc->vocab_used;
    h.draft = std::make_unique<CapturedStep>(
        build_dflash_draft_tp(h.ctx, h.w, t, h.b, e.buffers(r).control, e.link(r), opt_));
    h.tail = std::make_unique<CapturedStep>(build_dflash_tail(h.ctx, h.w, h.b, e.buffers(r).control, opt_));
  }
}

TapSink& TpDFlashDrafter::card(uint32_t r) {
  require(r < 2, "card " + std::to_string(r));
  return *half_[r];
}

std::array<l0::CmdList*, 2> TpDFlashDrafter::draft_lists(uint32_t k, float temperature, uint64_t seed) {
  require(half_[0]->draft != nullptr, "draft before attach");
  for (uint32_t r = 0; r < 2; ++r) set_draft_params(*half_[r]->b.ctl(), half_[r]->w.desc, k, temperature, seed);
  return {&half_[0]->draft->list, &half_[1]->draft->list};
}

std::array<l0::CmdList*, 2> TpDFlashDrafter::merge() {
  constexpr uint32_t R = kernels::dflash::kR, T = kernels::dflash::kTopK;
  struct C {
    uint32_t id;
    float v;
  };
  const auto better = [](const C& a, const C& b) {
    const bool ra = a.id < model::Qwen35::kVocab, rb = b.id < model::Qwen35::kVocab;   // a real id beats none
    if (ra != rb) return ra;
    return a.v > b.v || (a.v == b.v && a.id < b.id);
  };
  uint32_t* ids[2] = {half_[0]->b.cand.as<uint32_t>(), half_[1]->b.cand.as<uint32_t>()};
  float* vals[2] = {half_[0]->b.unary.as<float>(), half_[1]->b.unary.as<float>()};
  C row[2 * T];
  for (uint32_t j = 0; j < R; ++j) {
    for (uint32_t r = 0; r < 2; ++r)
      for (uint32_t i = 0; i < T; ++i) row[r * T + i] = {ids[r][j * T + i], vals[r][j * T + i]};
    std::stable_sort(row, row + 2 * T, better);
    for (uint32_t r = 0; r < 2; ++r)
      for (uint32_t i = 0; i < T; ++i) {
        ids[r][j * T + i] = row[i].id;
        vals[r][j * T + i] = row[i].v;
      }
  }
  return {&half_[0]->tail->list, &half_[1]->tail->list};
}

size_t TpDFlashDrafter::ring_bytes() const { return half_[0]->ring_bytes() + half_[1]->ring_bytes(); }

// One card's ring layout: [layers][slots][kv heads][head_dim] bf16, K then V - card 0's heads first in every slot.
void TpDFlashDrafter::save_ring(void* host) const {
  const size_t half = half_[0]->ring_bytes(), head_run = size_t(half_[0]->w.desc.kv_n()) * 2;
  std::vector<uint8_t> h[2] = {std::vector<uint8_t>(half), std::vector<uint8_t>(half)};
  for (uint32_t r = 0; r < 2; ++r) half_[r]->save_ring(half_[r]->imm, h[r].data());
  auto* out = static_cast<uint8_t*>(host);
  for (size_t run = 0; run < half / head_run; ++run)
    for (uint32_t r = 0; r < 2; ++r) {
      std::memcpy(out, h[r].data() + run * head_run, head_run);
      out += head_run;
    }
}

void TpDFlashDrafter::load_ring(const void* host) {
  const size_t half = half_[0]->ring_bytes(), head_run = size_t(half_[0]->w.desc.kv_n()) * 2;
  std::vector<uint8_t> h[2] = {std::vector<uint8_t>(half), std::vector<uint8_t>(half)};
  const auto* in = static_cast<const uint8_t*>(host);
  for (size_t run = 0; run < half / head_run; ++run)
    for (uint32_t r = 0; r < 2; ++r) {
      std::memcpy(h[r].data() + run * head_run, in, head_run);
      in += head_run;
    }
  for (uint32_t r = 0; r < 2; ++r) half_[r]->load_ring(half_[r]->imm, h[r].data());
}

}  // namespace runtime::dflash

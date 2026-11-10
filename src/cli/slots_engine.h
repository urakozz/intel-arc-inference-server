#pragma once
// Spec 13 §11 (plan 13c Task 4): several requests at once on ONE card - runtime::Engine's sequence slots behind
// server::BatchEngineIface and the scheduler (plan 13b's interface; cli::ReplicaEngine is its two-card twin).
//
// A step over ONE row (one request decoding) is that slot's own EngineAdapter (cli/serve_adapters.h) on the selected
// slot: today's lists and today's host code - sampling, MTP drafts and acceptance, prompt lookup, the deferred commit,
// truncate_to - so a lone request is bitwise what the one-slot server computes. A step over SEVERAL rows is one
// runtime::Engine::verify_rows replay over all of them (the row lists, the small-batch kernels): with MTP each row's
// drafts come from its slot's own draft lists first (M = 1 per slot, spec 13 §9.1 P7's first form), with prompt lookup
// from its request's proposer; the acceptance is EngineAdapter's rule per row (greedy: the target's argmax; sampled:
// accept_sampled / accept_point_mass with the slot's own generator, so a seeded request draws the same ids whatever
// shares the step), and each row's commit is deferred exactly as EngineAdapter's (truncate_to may keep fewer). A
// batched row's output is judged by the f32 judge against its lone run (spec 13 §11.3: B1 - the M-row kernels round
// differently from the M = 1 GEMV); the slots' isolation is bitwise (B3).
//
// The row budget (row_budget): one row decoding - none (the slot's own verify lists, K <= 3); several - 8 rows a step
// (runtime::Engine::kMaxRows), and with more than mtp_active_max rows decoding no drafts at all (plain batched decode:
// the measured costs decide, docs/probes/probe-batching-2026-10-23.md "Engine"). The scheduler clips the rows' K to it.
//
// One device, one thread: every call comes from the scheduler's tick thread (under --parallel replica, from its card's
// worker thread - cli::ReplicaEngine); nothing here is thread-safe.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/serve_adapters.h"
#include "runtime/engine.h"
#include "runtime/prefill_chunks.h"
#include "server/batch_engine.h"
#include "server/spec_accept.h"

namespace cli {

class SlotsEngine : public server::BatchEngineIface {
 public:
  // `mtp_k` / `lookup`: the server's --spec (EngineAdapter's arguments, every slot alike). `hooked`: the prefix cache
  // is on - a prompt is then cut where the serial server's hooked Engine::prefill call cuts it (prefill_cut).
  // `mtp_active_max`: above this many decoding rows a step drafts nothing.
  SlotsEngine(runtime::Engine& engine, uint32_t vocab_used, uint32_t mtp_k, bool lookup, bool hooked,
              uint32_t mtp_active_max = runtime::Engine::kMaxRows)
      : e_(engine), vocab_used_(vocab_used), hooked_(hooked), mtp_active_max_(mtp_active_max) {
    if (e_.slots() < 2) throw std::invalid_argument("SlotsEngine: the engine has one sequence slot");
    for (uint32_t s = 0; s < e_.slots(); ++s) {
      e_.select(s);
      adapters_.push_back(std::make_unique<EngineAdapter>(e_, vocab_used, mtp_k, lookup));
    }
    e_.select(0);
    pending_.resize(e_.slots());
    constexpr size_t V = model::Qwen35::kVocab;
    // Pinned readback rows (EngineAdapter's): q [kMaxRows][V] (each row's drafts), p [kMaxRows][V].
    host_ = std::make_unique<l0::Mem>(engine.context(), l0::MemKind::Host, size_t(2) * runtime::Engine::kMaxRows * V * 4);
  }

  // Tests and the record.
  EngineAdapter& adapter(uint32_t slot) { return *adapters_.at(slot); }
  uint64_t batched_steps() const { return batched_steps_; }
  uint64_t lone_steps() const { return lone_steps_; }

  uint32_t slots() override { return e_.slots(); }
  uint32_t max_len() override {
    uint32_t m = 0;
    for (uint32_t s = 0; s < e_.slots(); ++s) m = std::max(m, e_.slot_len(s));
    return m;
  }
  uint32_t slot_max_len(uint32_t slot) override { return e_.slot_len(slot); }
  uint32_t pos(uint32_t slot) override {
    const Pending& p = pending_.at(slot);
    if (p.on) return p.pos + p.j + 1;
    return on(slot).pos();
  }
  void reset(uint32_t slot) override {
    pending_.at(slot).on = false;   // discarded, as EngineAdapter::reset discards its own
    on(slot).reset();
  }

  // The serial server's cut (ReplicaEngine's): the whole prompt in the engine's own chunks, so a prompt prefilled
  // alone is bitwise the one-slot server's. A chunk stalls every decoding slot for its duration (the record measures
  // it).
  uint32_t prefill_cut(uint32_t /*slot*/, uint32_t pos, uint32_t remaining) override {
    return runtime::prefill_chunk_rows(pos, remaining, e_.prefill_chunk_cap(), hooked_, runtime::Engine::kBlock,
                                       /*wide_stride=*/true);
  }
  void prefill(const std::vector<Chunk>& chunks) override {
    for (const Chunk& c : chunks) {
      EngineAdapter& a = on(c.slot);
      std::vector<uint32_t> ids(c.ids, c.ids + c.n);
      if (c.replay)
        a.ingest(ids);   // --prefix-split-last: the last prompt id through the decode replay
      else
        a.prefill(ids);
    }
  }

  std::vector<uint32_t> step_batch(const std::vector<Row>& rows) override {
    if (rows.size() == 1) {
      ++lone_steps_;
      return {on(rows[0].slot).step(*rows[0].sampling)};
    }
    std::vector<SpecRow> spec;
    for (const Row& r : rows) spec.push_back({r.slot, r.sampling, r.rng, 0, nullptr});
    std::vector<uint32_t> ids;
    for (const std::vector<uint32_t>& run : batched(spec)) ids.push_back(run.at(0));
    flush_all();   // a plain step keeps every id: nothing to truncate
    return ids;
  }

  uint32_t spec_k() override { return adapters_[0]->mtp_k(); }
  uint32_t verify_k() override { return adapters_[0]->verify_k(); }
  uint32_t row_budget(uint32_t active) override {
    if (active <= 1) return 0;   // the slot's own verify lists
    return active > mtp_active_max_ ? active : runtime::Engine::kMaxRows;
  }
  std::vector<std::vector<uint32_t>> step_spec(const std::vector<SpecRow>& rows) override {
    if (rows.size() == 1) {
      ++lone_steps_;
      const SpecRow& r = rows[0];
      EngineAdapter& a = on(r.slot);
      if (r.propose != nullptr) return {a.step_drafts(*r.sampling, *r.propose)};
      if (a.mtp_k() > 0) return {a.step_many(*r.sampling, r.k)};
      return {{a.step(*r.sampling)}};
    }
    return batched(rows);
  }
  void truncate_to(uint32_t slot, uint32_t p) override {
    Pending& q = pending_.at(slot);
    if (!q.on) {
      on(slot).truncate_to(p);
      return;
    }
    if (p < q.pos + 1 || p > q.pos + q.j + 1)
      throw std::logic_error("SlotsEngine::truncate_to: pos " + std::to_string(p) + " is outside slot " +
                             std::to_string(slot) + "'s last run [" + std::to_string(q.pos + 1) + ", " +
                             std::to_string(q.pos + q.j + 1) + "]");
    const uint32_t m = p - q.pos - 1;
    q.on = false;
    e_.select(slot);
    e_.commit(m, m < q.j ? q.drafts[m] : q.next);
  }

  // Spec 7 per slot: the slot's own adapter (its pending commit first).
  uint32_t block() override { return runtime::Engine::kBlock; }
  // The same for every slot, and asking commits nothing.
  size_t state_bytes() override { return adapters_[0]->state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return adapters_[0]->kv_bytes(n); }
  uint64_t kv_form() override { return adapters_[0]->kv_form(); }
  void save_state(uint32_t slot, void* host) override { on(slot).save_state(host); }
  void load_state(uint32_t slot, const void* host, uint32_t p) override {
    pending_.at(slot).on = false;
    on(slot).load_state(host, p);
  }
  void save_kv(uint32_t slot, uint32_t b, uint32_t en, void* host) override { on(slot).save_kv(b, en, host); }
  void load_kv(uint32_t slot, uint32_t b, uint32_t en, const void* host) override { on(slot).load_kv(b, en, host); }

 private:
  // A batched run's commit, deferred as EngineAdapter defers its own (truncate_to may keep fewer).
  struct Pending {
    bool on = false;
    uint32_t pos = 0, j = 0, next = 0;
    std::vector<uint32_t> drafts;
  };

  // The slot selected, its batched commit made (the adapter flushes its own at its next call).
  EngineAdapter& on(uint32_t slot) {
    EngineAdapter& a = *adapters_.at(slot);
    flush(slot);
    e_.select(slot);
    return a;
  }
  void flush(uint32_t slot) {
    Pending& q = pending_.at(slot);
    if (!q.on) return;
    q.on = false;
    e_.select(slot);
    e_.commit(q.j, q.next);
  }
  void flush_all() {
    for (uint32_t s = 0; s < e_.slots(); ++s) {
      flush(s);
      e_.select(s);
      adapters_[s]->flush_commit();
    }
  }

  // One batched iteration over several rows (the class comment). Returns per row its pending id and the kept drafts.
  std::vector<std::vector<uint32_t>> batched(const std::vector<SpecRow>& rows) {
    ++batched_steps_;
    flush_all();   // every slot committed: the row lists' saved rows are rewritten below
    constexpr size_t V = model::Qwen35::kVocab;
    float* q_host = host_->as<float>();
    float* p_host = q_host + runtime::Engine::kMaxRows * V;
    std::vector<runtime::Engine::Run> runs;
    std::vector<std::vector<uint32_t>> drafts(rows.size());
    std::vector<uint32_t> xs(rows.size()), ks(rows.size(), 0);
    std::vector<size_t> q_row(rows.size(), 0);   // each sampled MTP row's first q row in q_host
    size_t q_used = 0;
    std::vector<size_t> mtp;                     // the rows the MTP head drafts for
    for (size_t i = 0; i < rows.size(); ++i) {
      const SpecRow& r = rows[i];
      EngineAdapter& a = *adapters_.at(r.slot);
      e_.select(r.slot);
      runtime::Control* c = e_.control();
      xs[i] = c->cur_token[0];
      if (r.propose != nullptr) {   // prompt lookup: the request's own drafts
        const std::vector<uint32_t> d = (*r.propose)(xs[i]);
        ks[i] = std::min<uint32_t>({static_cast<uint32_t>(d.size()), runtime::Engine::kMaxDraft, e_.max_verify_k()});
        drafts[i].assign(d.begin(), d.begin() + ks[i]);
        for (uint32_t t = 0; t < ks[i]; ++t) c->cur_token[1 + t] = drafts[i][t];
      } else if (a.mtp_k() > 0) {
        ks[i] = std::min({r.k, a.mtp_k(), e_.max_verify_k()});
        if (ks[i] > 0) {
          mtp.push_back(i);
          if (!r.sampling->greedy) {
            a.seed_rng(*r.sampling);
            q_row[i] = q_used;
            q_used += ks[i];
          }
        }
      }
    }
    // The MTP drafts: two or more rows drafting - ONE draft_rows step per draft index (the head and lm_head read once
    // for all of them; every row drafts the deepest K and keeps its own); one row - its slot's own draft lists. A
    // sampled row draws its own k drafts from its own generator (the extra ones keep the argmax: no draw).
    uint32_t deepest = 0;
    for (size_t i : mtp) deepest = std::max(deepest, ks[i]);
    for (size_t i : mtp) {   // every row needs room for the deepest K
      e_.select(rows[i].slot);
      if (e_.max_verify_k() < deepest) deepest = 0;
    }
    auto sample_q = [&](size_t i, uint32_t t, const float* dev) {
      const SpecRow& r = rows[i];
      EngineAdapter& a = *adapters_.at(r.slot);
      float* q = q_host + (q_row[i] + t) * V;
      a.imm.copy(q, dev, V * sizeof(float));
      return server::sample_draft(q, vocab_used_, *r.sampling, a.rng);
    };
    if (mtp.size() >= 2 && deepest > 0 && mtp.size() <= runtime::Engine::kMaxDraftRows) {
      std::vector<runtime::Engine::Run> dr;
      for (size_t i : mtp) dr.push_back({rows[i].slot, ks[i]});
      e_.draft_rows(dr, [&](size_t m, uint32_t t, uint32_t argmax) {
        const size_t i = mtp[m];
        if (rows[i].sampling->greedy || t >= ks[i]) return argmax;
        return sample_q(i, t, e_.rows_logits_device() + m * V);
      });
    } else {
      for (size_t i : mtp) {
        e_.select(rows[i].slot);
        if (rows[i].sampling->greedy)
          e_.draft(ks[i]);
        else
          e_.draft(ks[i], [&](uint32_t t) { return sample_q(i, t, e_.mtp_logits_device() + t * V); });
      }
    }
    for (size_t i : mtp) {
      e_.select(rows[i].slot);
      drafts[i] = e_.draft_ids();
    }
    for (size_t i = 0; i < rows.size(); ++i) {
      ++adapters_.at(rows[i].slot)->mtp_iters_at_k[ks[i]];
      runs.push_back({rows[i].slot, ks[i]});
    }
    e_.verify_rows(runs);
    const uint32_t* ids = e_.rows_ids();
    std::vector<std::vector<uint32_t>> out(rows.size());
    uint32_t r0 = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
      const SpecRow& r = rows[i];
      EngineAdapter& a = *adapters_.at(r.slot);
      const uint32_t k = runs[i].k;
      server::AcceptResult acc{};
      if (r.sampling->greedy) {
        acc = server::accept_greedy(ids + r0, drafts[i].data(), k);
      } else {
        a.seed_rng(*r.sampling);
        a.imm.copy(p_host, e_.rows_logits_device() + size_t(r0) * V, (k + 1) * V * sizeof(float));
        if (k == 0) {
          acc.accepted = 0;
          acc.next_token = sample(p_host, vocab_used_, *r.sampling, a.rng);
        } else if (r.propose != nullptr) {
          acc = server::accept_point_mass(p_host, drafts[i].data(), k, vocab_used_, V, *r.sampling, a.rng);
        } else {
          acc = server::accept_sampled(p_host, q_host + q_row[i] * V, drafts[i].data(), k, vocab_used_, V,
                                       *r.sampling, a.rng);
        }
      }
      Pending& pd = pending_[r.slot];
      pd.on = true;
      e_.select(r.slot);
      pd.pos = e_.pos();
      pd.j = acc.accepted;
      pd.next = acc.next_token;
      pd.drafts = drafts[i];
      if (k > 0) {
        ++a.mtp_iters;
        a.mtp_drafted += k;
        a.mtp_accepted += acc.accepted;
      }
      out[i].push_back(xs[i]);
      out[i].insert(out[i].end(), drafts[i].begin(), drafts[i].begin() + acc.accepted);
      a.gen_tokens_ += static_cast<uint32_t>(out[i].size());
      r0 += k + 1;
    }
    return out;
  }

  runtime::Engine& e_;
  uint32_t vocab_used_;
  bool hooked_;
  uint32_t mtp_active_max_;
  std::vector<std::unique_ptr<EngineAdapter>> adapters_;
  std::vector<Pending> pending_;
  std::unique_ptr<l0::Mem> host_;
  uint64_t batched_steps_ = 0, lone_steps_ = 0;
};

}  // namespace cli

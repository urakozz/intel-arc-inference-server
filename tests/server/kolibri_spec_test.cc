// Kolibri DSpark (host): cli::kolibri::KolibriEngineAdapterT's speculative loop over a device-free engine with KolibriEngine's
// speculative calls (verify / commit / read_verify_logits / max_verify_k) and a fake DSpark drafter. No device.
//
// The fake target's greedy next id is a fixed function of (previous id, position) over a 64-row head of which the tokenizer
// defines 60 (Kolibri's 128000 / 127998 in small): its argmax lands on an undefined row now and then, which the adapter must
// mask exactly as a plain step does. The fake drafter proposes the target's chain for the first a(pos) drafts, then wrong
// ids - every acceptance length 0..7 occurs.
//
//   1. Greedy is lossless: step_many at K = 1..7 (and AdaptiveK-like varying K) emits the plain step()'s ids, token for token,
//      over 400 ids from three prompts - with the masked bonus ids among them.
//   2. truncate_to inside a run (a stop the server keeps short) leaves the session where plain decoding would be: the next
//      ids equal the plain run's continuation.
//   3. The deferred commit: pos() counts the kept ids before the commit; a save flushes it.
//   4. Sampled: the walk's seed comes from the request's RNG, every emitted id is a defined token, a seeded request is
//      reproducible.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

#include "check.h"
#include "cli/kolibri_serve.h"

namespace {

constexpr uint32_t kV = 64, kUsed = 60;

uint32_t next_of(uint32_t prev, uint32_t pos) { return (prev * 37u + pos * 11u + 5u) % kV; }
void row_for(uint32_t argmax, float* row) {
  for (uint32_t i = 0; i < kV; ++i) row[i] = -std::fabs(float(i) - float(argmax)) - 0.001f * float(i);
}

struct FakeEngine {
  static constexpr uint32_t kBlock = 2048;
  uint32_t pos_ = 0, pending_ = 0, max_len_ = 4096;
  int logits_row_ = -1;
  std::vector<float> logits_ = std::vector<float>(kV), vlogits_ = std::vector<float>(8 * kV);
  uint32_t verify_ids_[8] = {}, vk_ = 0xFFFFFFFFu, vpos_ = 0;
  uint64_t verifies = 0;

  uint32_t vocab() const { return kV; }
  uint32_t max_len() const { return max_len_; }
  uint32_t pos() const { return pos_; }
  uint32_t pending() const { return pending_; }
  void reset() { pos_ = pending_ = 0; logits_row_ = -1; vk_ = 0xFFFFFFFFu; }
  void set_token(uint32_t id) { pending_ = id; }
  void prefill(const std::vector<uint32_t>& ids) {
    pos_ += uint32_t(ids.size());
    pending_ = next_of(ids.back(), pos_ - 1);
    row_for(pending_, logits_.data());
    logits_row_ = -1;
  }
  void ingest(const std::vector<uint32_t>& ids) { prefill(ids); }
  std::vector<uint32_t> generate(uint32_t n) {
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < n; ++i) {
      out.push_back(pending_);
      pending_ = next_of(pending_, pos_);
      row_for(pending_, logits_.data());
      ++pos_;
      logits_row_ = -1;
    }
    return out;
  }
  void read_logits_into(float* host) {
    std::memcpy(host, logits_row_ >= 0 ? vlogits_.data() + size_t(logits_row_) * kV : logits_.data(), kV * 4);
  }
  // KolibriEngine's speculative calls.
  uint32_t max_verify_k() const { return std::min<uint32_t>(7, max_len_ - pos_ - 1); }
  void verify(const uint32_t* drafts, uint32_t k) {
    uint32_t t = pending_;
    for (uint32_t r = 0; r <= k; ++r) {
      verify_ids_[r] = next_of(t, pos_ + r);
      row_for(verify_ids_[r], vlogits_.data() + size_t(r) * kV);
      if (r < k) t = drafts[r];
    }
    vk_ = k;
    vpos_ = pos_;
    ++verifies;
  }
  const uint32_t* verify_ids() const { return verify_ids_; }
  void read_verify_logits(float* host, uint32_t rows) {
    CHECK(vk_ != 0xFFFFFFFFu && rows >= 1 && rows <= vk_ + 1);
    std::memcpy(host, vlogits_.data(), size_t(rows) * kV * 4);
  }
  void commit(uint32_t j, uint32_t t) {
    CHECK(vk_ != 0xFFFFFFFFu && vpos_ == pos_ && j <= vk_ && t < kV);
    pos_ += j + 1;
    pending_ = t;
    logits_row_ = int(j);
    vk_ = 0xFFFFFFFFu;
  }
  uint64_t drafter_key() const { return 7; }
  // 20e's snapshot calls (unused here).
  size_t state_bytes() const { return 0; }
  size_t kv_bytes(uint32_t) const { return 0; }
  void save_state(void*) {}
  void load_state(const void*, uint32_t p) { pos_ = p; }
  void save_kv(uint32_t, uint32_t, void*) {}
  void load_kv(uint32_t, uint32_t, const void*) {}
  void set_block_hook(server::EngineIface::BlockHook) {}
};

// The fake drafter: the target's chain from the pending id for the first a(pos) drafts, then wrong ids; one candidate a row.
struct FakeDrafter {
  FakeEngine& e;
  runtime::dflash::DFlashDraftOut out{};
  uint64_t last_seed = 0;
  const runtime::dflash::DFlashDraftOut& draft(uint32_t k, float, uint64_t seed) {
    last_seed = seed;
    out = {};
    out.k = 7;
    const uint32_t a = (e.pos_ * 5u) % 9u;   // 0..8: every acceptance length occurs
    uint32_t t = e.pending_;
    for (uint32_t j = 0; j < 7; ++j) {
      const uint32_t truth = next_of(t, e.pos_ + j);
      const uint32_t id = (j < a && truth < kUsed) ? truth : (truth + 1) % kUsed;
      out.ids[j] = id;
      out.cand[j][0] = id;
      out.scores[j][0] = 0.0f;
      for (uint32_t c = 1; c < runtime::dflash::DFlashDraftOut::kTopK; ++c) {
        out.cand[j][c] = (id + c) % kUsed;
        out.scores[j][c] = -INFINITY;
      }
      t = id;
    }
    (void)k;
    return out;
  }
};

using Adapter = cli::kolibri::KolibriEngineAdapterT<FakeEngine>;

server::Sampling greedy() {
  server::Sampling s;
  s.greedy = true;
  return s;
}

std::vector<uint32_t> plain(const std::vector<uint32_t>& prompt, uint32_t n) {
  FakeEngine e;
  Adapter a(e, kUsed);
  a.prefill(prompt);
  std::vector<uint32_t> out;
  while (out.size() < n) out.push_back(a.step(greedy()));
  return out;
}

}  // namespace

int main() {
  static_assert(Adapter::kSpec, "the fake engine has KolibriEngine's speculative calls");
  const std::vector<std::vector<uint32_t>> prompts = {{1, 2, 3}, {17, 4, 40, 9, 9}, {59}};
  const uint32_t N = 400;

  // --- 1. greedy is lossless ------------------------------------------------------------------------------------------
  uint64_t masked_total = 0;
  for (const auto& prompt : prompts) {
    const std::vector<uint32_t> ref = plain(prompt, N);
    for (uint32_t K = 0; K <= 8; ++K) {   // K 8: a varying K 1..7 by iteration
      FakeEngine e;
      FakeDrafter fd{e};
      cli::kolibri::KolibriDraftHooks h;
      h.draft = [&fd](uint32_t k, float T, uint64_t seed) -> const runtime::dflash::DFlashDraftOut& { return fd.draft(k, T, seed); };
      Adapter a(e, kUsed, 7, h);
      a.prefill(prompt);
      std::vector<uint32_t> out;
      uint32_t it = 0;
      while (out.size() < N) {
        const uint32_t k = K == 8 ? 1 + (it++ % 7) : K;
        for (uint32_t id : a.step_many(greedy(), k)) out.push_back(id);
      }
      out.resize(N);
      CHECK(out == ref);
      for (uint32_t id : out) CHECK(id < kUsed);
      masked_total += a.masked();
      if (K == 7) CHECK(a.spec_accepted() > 0 && a.spec_iterations() > 0);
    }
  }
  CHECK(masked_total > 0);
  std::printf("1. greedy step_many == plain step() over %u ids x 3 prompts at K 0..7 and a varying K (%llu bonus ids masked)\n",
              N, (unsigned long long)masked_total);

  // --- 2. truncate_to inside a run ------------------------------------------------------------------------------------
  {
    const std::vector<uint32_t> ref = plain(prompts[0], N);
    FakeEngine e;
    FakeDrafter fd{e};
    cli::kolibri::KolibriDraftHooks h;
    h.draft = [&fd](uint32_t k, float T, uint64_t seed) -> const runtime::dflash::DFlashDraftOut& { return fd.draft(k, T, seed); };
    Adapter a(e, kUsed, 7, h);
    a.prefill(prompts[0]);
    std::vector<uint32_t> out;
    uint32_t truncations = 0;
    while (out.size() < N) {
      const std::vector<uint32_t> run = a.step_many(greedy(), 7);
      if (run.size() >= 3) {   // the server keeps two of them
        a.truncate_to(a.pos() - uint32_t(run.size()) + 2);
        out.insert(out.end(), run.begin(), run.begin() + 2);
        ++truncations;
      } else {
        out.insert(out.end(), run.begin(), run.end());
      }
    }
    out.resize(N);
    CHECK(truncations > 0 && out == ref);
    std::printf("2. truncate_to inside %u runs: the plain continuation\n", truncations);
  }

  // --- 3. the deferred commit -----------------------------------------------------------------------------------------
  {
    FakeEngine e;
    FakeDrafter fd{e};
    cli::kolibri::KolibriDraftHooks h;
    h.draft = [&fd](uint32_t k, float T, uint64_t seed) -> const runtime::dflash::DFlashDraftOut& { return fd.draft(k, T, seed); };
    Adapter a(e, kUsed, 7, h);
    a.prefill(prompts[1]);
    const uint32_t p0 = a.pos();
    const std::vector<uint32_t> run = a.step_many(greedy(), 7);
    CHECK(a.pos() == p0 + run.size());
    CHECK(e.pos_ == p0);   // not committed yet
    a.save_state(nullptr);
    CHECK(e.pos_ == p0 + run.size());
    std::printf("3. the commit is deferred until the next engine call (a save flushes it)\n");
  }

  // --- 4. sampled ----------------------------------------------------------------------------------------------------
  {
    auto run_sampled = [&](uint64_t seed, std::vector<uint64_t>* seeds) {
      FakeEngine e;
      FakeDrafter fd{e};
      cli::kolibri::KolibriDraftHooks h;
      h.draft = [&](uint32_t k, float T, uint64_t s) -> const runtime::dflash::DFlashDraftOut& {
        CHECK(T > 0.0f);
        if (seeds) seeds->push_back(s);
        return fd.draft(k, T, s);
      };
      Adapter a(e, kUsed, 7, h);
      a.prefill(prompts[2]);
      server::Sampling s;
      s.greedy = false;
      s.temperature = 0.7f;
      s.top_p = 1.0f;
      s.top_k = 0;
      s.has_seed = true;
      s.seed = seed;
      std::vector<uint32_t> out;
      while (out.size() < 200)
        for (uint32_t id : a.step_many(s, 7)) out.push_back(id);
      for (uint32_t id : out) CHECK(id < kUsed);
      return out;
    };
    std::vector<uint64_t> seeds;
    const std::vector<uint32_t> a1 = run_sampled(42, &seeds), a2 = run_sampled(42, nullptr), b = run_sampled(43, nullptr);
    CHECK(a1 == a2 && a1 != b);
    CHECK(!seeds.empty() && std::count(seeds.begin(), seeds.end(), seeds[0]) < int(seeds.size()));
    std::printf("4. sampled: defined ids only, a seeded request reproducible, the walk's seed from the request's RNG\n");
  }
  // --- 5. the AdaptiveK cost table (DERIVED) ----------------------------------------------------------------------------
  for (bool i8 : {true, false}) {
    const server::MtpCost c = server::MtpCost::kolibri_dspark(i8);
    CHECK(c.verify.size() == 8 && c.draft.size() == 7 && c.verify[0] == 1.0 && c.max_k() == 7);
    for (size_t m = 1; m < c.verify.size(); ++m) CHECK(c.verify[m] > c.verify[m - 1]);
    for (double d : c.draft) CHECK(d == c.draft[0] && d > 0.0 && d < 1.0);
    std::printf("5. cost (%s head, DERIVED): verify M 8 %.3f, a draft %.3f plain steps\n", i8 ? "int8" : "bf16", c.verify[7],
                c.draft[0]);
  }
  std::printf("kolibri_spec_test: OK\n");
  return 0;
}

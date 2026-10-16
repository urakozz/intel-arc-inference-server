// Spec 19d (`b70-serve --spec dflash --pp 2`) without a device: cli::pp::PipelineEngineAdapterT's DFlash path -
// EngineAdapter::step_dflash's logic over runtime::PipelineEngine's host calls - over FakeTapPipe, a device-free
// engine with PipelineEngine's interface and a tap drafter's contract:
//
//   - the target: one session (the state a hash of every id consumed, the KV the ids per position), the next id the
//     argmax of a logits row over a word list - a function of the state and the KV;
//   - the drafter: a ring of `kW` slots, slot p % kW holding position p's projection (a hash of the ids up to p),
//     filled as PipelineEngine fills its drafter's: a prefill call projects the window of its end, commit(j) the
//     verify's rows 0..j, reset() zeroes it; the snapshot carries it (spec 19d) and drafter_key() names it;
//     draft_tap(k, T, seed): with a ring that holds the true projections of the window before the anchor, draft
//     j >= 2 is the true continuation unless mix(ring, pos, j) % 4 == 0, draft 1 always; with a stale ring every
//     draft misses. The readback carries each row's 16 candidates (the first 16 words, their scores), the walk's
//     pick greedy or a seeded draw over softmax(scores / T);
//   - verify(k) at M = k + 1 rows (k <= 7, pos 0 included), commit(j, t) as Engine's contract.
//
//   1. greedy is lossless through the adapter: step_many at K = 7, 3 and AdaptiveK over MtpCost::dflash give the
//      plain step() ids, with truncate_to inside a burst; speculation happens (fewer iterations than ids);
//   2. seeded sampling: a sampled K = 7 run twice from fresh adapters is identical, every id a word;
//   3. spec 7's request path (server::PrefixSession over the adapter): a turn restored from the previous turn's
//      request-end snapshot after a side request drafts exactly as the uninterrupted session (the per-iteration
//      acceptance equal), its ids the plain run's; the store's root is store_root(kv form, the drafter's key);
//   4. the adapter's refusals: --spec dflash without a drafter, --spec-max past the drafter's block, with lookup.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "cli/pipeline_serve_adapter.h"
#include "model/qwen35.h"
#include "runtime/buffer_sizes.h"
#include "runtime/dflash/dflash_control.h"
#include "server/adaptive_k.h"
#include "server/prefix_cache.h"

namespace {

using Ids = std::vector<uint32_t>;
constexpr uint32_t kV = model::Qwen35::kVocab;
constexpr uint32_t kW = 8;   // the drafter's ring (its window)

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

struct FakeTapPipe {
  static constexpr uint32_t kBlock = 16;
  static constexpr uint32_t kMaxDraft = 3;   // the MTP head's (none here)
  using BlockHook = std::function<void(uint32_t, bool)>;

  FakeTapPipe(uint32_t max_len, bool with_drafter, uint32_t drafter_k = 7) : maxlen(max_len), dk(drafter_k) {
    drafter = with_drafter;
    for (uint32_t i = 0; i < 24; ++i) words.push_back(1000 + i * 7);
    kv.assign(maxlen + 8, 0);
    reset();
  }
  uint32_t maxlen, dk;
  bool drafter = true;
  Ids words, kv;
  uint32_t p = 0, cur = 0;
  std::vector<uint64_t> ring = std::vector<uint64_t>(kW, 0);
  BlockHook hook;
  // verify state
  static constexpr uint32_t kNone = 0xFFFFFFFFu;
  uint32_t verify_pos = 0, verify_k = kNone, cur_rows[8] = {}, out_rows[8] = {};
  std::vector<float> rows;
  runtime::dflash::DFlashDraftOut out{};
  std::vector<std::pair<uint32_t, uint32_t>> log;   // (pos, drafts right) per draft_tap

  uint64_t prefix_hash(uint32_t n) const {   // the "state": every id consumed
    uint64_t h = 0x5EED;
    for (uint32_t i = 0; i < n; ++i) h = mix(h, kv[i]);
    return h;
  }
  float score(uint64_t h, uint32_t w) const { return float(mix(h, w) % 1000) / 250.0f; }
  uint32_t argmax_after(uint32_t n) const {   // the next id after positions [0, n)
    const uint64_t h = prefix_hash(n);
    uint32_t best = words[0];
    float bv = -1;
    for (uint32_t w : words) {
      const float s = score(h, w);
      if (s > bv || (s == bv && w < best)) {
        bv = s;
        best = w;
      }
    }
    return best;
  }
  uint64_t projection(uint32_t at) const { return mix(prefix_hash(at + 1), 0xD5); }
  void project(uint32_t at) { ring[at % kW] = projection(at); }
  bool ring_true(uint32_t n) const {
    for (uint32_t q = n > kW ? n - kW : 0; q < n; ++q)
      if (ring[q % kW] != projection(q)) return false;
    return true;
  }

  // --- the session ------------------------------------------------------------------------------------------------
  void reset() {
    std::fill(kv.begin(), kv.end(), 0);
    p = cur = 0;
    ring.assign(kW, 0);
    verify_k = kNone;
  }
  void prefill(const Ids& ids) {
    CHECK(!ids.empty() && p + ids.size() <= maxlen);
    const uint32_t base = p, end = p + uint32_t(ids.size());
    const uint32_t from = end > kW ? std::max(base, end - kW) : base;
    for (size_t i = 0; i < ids.size(); ++i) {
      kv[p++] = ids[i];
      if (drafter && p - 1 >= from) project(p - 1);
      if (hook && i + 1 < ids.size() && p % kBlock == 0) hook(p, true);
    }
    cur = argmax_after(p);
    verify_k = kNone;
    if (hook) hook(p, p % kBlock == 0);
  }
  void step1() {   // verify(0) + commit(0) with a drafter, a plain step without
    kv[p++] = cur;
    if (drafter) project(p - 1);
    cur = argmax_after(p);
  }
  void ingest(const Ids& ids) {
    for (uint32_t id : ids) {
      cur = id;
      step1();
    }
  }
  Ids generate(uint32_t n) {
    Ids o;
    for (uint32_t i = 0; i < n; ++i) {
      o.push_back(cur);
      step1();
    }
    return o;
  }
  uint32_t pos() const { return p; }
  uint32_t max_len() const { return maxlen; }
  runtime::KvCache kv_cache() const { return runtime::KvCache::Bf16; }
  uint32_t pending() const { return cur; }
  void set_token(uint32_t id) {
    CHECK(id < kV);
    cur = id;
  }
  void set_block_hook(BlockHook h) { hook = std::move(h); }

  // --- the proposers -----------------------------------------------------------------------------------------------
  bool mtp() const { return false; }
  uint32_t max_draft() const { return drafter ? dk : 0; }
  uint32_t max_verify_k() const {
    if (!drafter || p + 1 >= maxlen) return 0;
    return std::min<uint32_t>(dk, maxlen - p - 1);
  }
  void draft(uint32_t, const std::function<uint32_t(uint32_t)>& = {}) { CHECK(false); }
  const Ids& draft_ids() const {
    CHECK(false);
    return words;
  }
  void set_draft_input(uint32_t i, uint32_t id) {
    CHECK(i < 7 && id < kV);
    cur_rows[1 + i] = id;
  }
  bool tap_samples() const { return true; }
  uint64_t drafter_key() const { return drafter ? 0xDF1A5Bull : 0; }
  const runtime::dflash::DFlashDraftOut& draft_tap(uint32_t k, float T, uint64_t seed) {
    CHECK(drafter && k >= 1 && k <= max_verify_k());
    const bool good = ring_true(p);
    uint64_t rh = 0;
    for (uint64_t r : ring) rh = mix(rh, r);
    // The true continuation from a copy of the session.
    FakeTapPipe copy = *this;
    copy.hook = {};
    Ids truth;
    for (uint32_t i = 0; i < k; ++i) {
      copy.step1();
      truth.push_back(copy.cur);
    }
    std::mt19937_64 g(seed);
    out = {};
    out.k = k;
    out.anchor = cur;
    out.pos = p;
    uint32_t right = 0;
    bool missed = false;
    for (uint32_t j = 0; j < k; ++j) {
      const bool hit = good && (j == 0 || mix(rh, uint64_t(p) * 8 + j) % 4 != 0);
      const uint32_t guess = hit ? truth[j] : words[(std::find(words.begin(), words.end(), truth[j]) - words.begin() + 5) % words.size()];
      for (uint32_t c = 0; c < 16; ++c) {
        out.cand[j][c] = words[c];
        out.unary[j][c] = out.scores[j][c] = words[c] == guess ? 9.0f : float(mix(rh, c + j * 31) % 100) / 50.0f;
      }
      // the candidates must hold the guess: put it first if it is not among the 16
      if (std::find(out.cand[j], out.cand[j] + 16, guess) == out.cand[j] + 16) {
        out.cand[j][0] = guess;
        out.scores[j][0] = out.unary[j][0] = 9.0f;
      }
      uint32_t pick = guess;
      if (T > 0.0f) {   // a draw from softmax(scores / T) over the 16 - the walk's q
        double z = 0, w[16];
        for (uint32_t c = 0; c < 16; ++c) z += w[c] = std::exp(double(out.scores[j][c]) / T);
        double u = std::uniform_real_distribution<double>(0, z)(g);
        uint32_t c = 0;
        while (c < 15 && u >= w[c]) u -= w[c++];
        pick = out.cand[j][c];
        out.chosen[j] = c;
      }
      out.ids[j] = pick;
      if (!missed && pick == truth[j]) ++right;
      else missed = true;
    }
    cur_rows[0] = cur;
    for (uint32_t j = 0; j < k; ++j) cur_rows[1 + j] = out.ids[j];
    log.push_back({p, right});
    return out;
  }
  void verify(uint32_t k) {
    CHECK(drafter && k <= dk && p + k + 1 <= maxlen);
    cur_rows[0] = cur;
    verify_pos = p;
    for (uint32_t m = 0; m <= k; ++m) {
      kv[p + m] = cur_rows[m];
      out_rows[m] = argmax_after(p + m + 1);
    }
    verify_k = k;
  }
  const uint32_t* verify_ids() const { return out_rows; }
  void commit(uint32_t j, uint32_t next) {
    CHECK(verify_k != kNone && j <= verify_k && next < kV);
    p = verify_pos + j + 1;
    for (uint32_t m = 0; m <= j; ++m) project(verify_pos + m);
    cur = next;
    verify_k = kNone;
  }

  // --- the host reads ----------------------------------------------------------------------------------------------
  float* host_rows(size_t floats) {
    rows.assign(floats, 0.0f);
    return rows.data();
  }
  void read_logits_into(float* host, uint32_t n) const {
    for (uint32_t r = 0; r < n; ++r) {
      float* row = host + size_t(r) * kV;
      std::fill(row, row + kV, -1e30f);
      const uint64_t h = prefix_hash(verify_k == kNone ? p : verify_pos + r + 1);
      for (uint32_t w : words) row[w] = score(h, w);
    }
  }
  void read_draft_logits_into(float*, uint32_t) const { CHECK(false); }

  // --- spec 7 (state = the ring; the KV = the ids) -------------------------------------------------------------------
  size_t state_bytes() const { return 8 + (drafter ? kW * 8 : 0); }
  size_t kv_bytes(uint32_t n) const { return size_t(n) * 4; }
  void save_state(void* host) const {
    const uint64_t marker = prefix_hash(p);
    std::memcpy(host, &marker, 8);
    if (drafter) std::memcpy(static_cast<uint8_t*>(host) + 8, ring.data(), kW * 8);
  }
  void load_state(const void* host, uint32_t at) {
    uint64_t marker = 0;
    std::memcpy(&marker, host, 8);
    p = at;
    if (drafter) std::memcpy(ring.data(), static_cast<const uint8_t*>(host) + 8, kW * 8);
    verify_k = kNone;
    loaded_marker = marker;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) const { std::memcpy(host, kv.data() + b, size_t(e - b) * 4); }
  void load_kv(uint32_t b, uint32_t e, const void* host) { std::memcpy(kv.data() + b, host, size_t(e - b) * 4); }
  uint64_t loaded_marker = 0;
};

using Adapter = cli::pp::PipelineEngineAdapterT<FakeTapPipe>;

struct MallocAlloc : server::HostAlloc {
  void* alloc(size_t n) override { return std::malloc(n); }
  void free(void* p, size_t) override { std::free(p); }
};

Ids prompt_of(uint32_t key, uint32_t n) {
  Ids v;
  for (uint32_t i = 0; i < n; ++i) v.push_back(1000 + ((key * 131 + i * 7) % 24) * 7);
  return v;
}

struct Gen {
  Ids ids;
  uint64_t iters = 0;
};
// server.cc's loop minus the text: step() (k = 0, no policy) or step_many(K) until n ids, the last burst truncated.
Gen run(Adapter& a, uint32_t n, const server::Sampling& s, uint32_t k_fixed, bool policy) {
  Gen g;
  std::unique_ptr<server::AdaptiveK> ak;
  if (policy) {
    server::AdaptiveKOptions o;
    o.max_k = 7;
    o.cost = server::MtpCost::dflash(true, true);
    ak = std::make_unique<server::AdaptiveK>(o);
  }
  while (g.ids.size() < n) {
    if (!policy && k_fixed == 0) {
      g.ids.push_back(a.step(s));
      continue;
    }
    const uint32_t k = ak ? ak->next() : k_fixed;
    const Ids burst = a.step_many(s, k);
    if (ak) ak->observe(k, std::min<uint32_t>(k, uint32_t(burst.size() - 1)));
    ++g.iters;
    const size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    g.ids.insert(g.ids.end(), burst.begin(), burst.begin() + keep);
    if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
  }
  return g;
}

void test_lossless() {
  FakeTapPipe e(512, true);
  Adapter a(e, kV, kV, 7, false, true);
  server::Sampling greedy;
  for (uint32_t key : {1u, 2u, 3u}) {
    const Ids pr = prompt_of(key, 37);
    a.reset();
    a.prefill(pr);
    const Gen plain = run(a, 120, greedy, 0, false);
    for (int arm = 0; arm < 3; ++arm) {
      a.reset();
      a.prefill(pr);
      const Gen g = run(a, 120, greedy, arm == 0 ? 7 : arm == 1 ? 3 : 0, arm == 2);
      CHECK(g.ids == plain.ids);
      CHECK(g.iters < 120);
    }
  }
  std::printf("1. greedy step_many at K 7 / 3 / auto (truncated bursts included) = the plain ids OK\n");
}

void test_sampled() {
  server::Sampling s;
  s.greedy = false;
  s.temperature = 0.7f;
  s.top_k = 20;
  s.top_p = 0.95f;
  s.seed = 42;
  s.has_seed = true;
  Gen g[2];
  for (int r = 0; r < 2; ++r) {
    FakeTapPipe e(512, true);
    Adapter a(e, kV, kV, 7, false, true);
    a.reset();
    a.prefill(prompt_of(5, 30));
    g[r] = run(a, 96, s, 7, false);
  }
  CHECK(g[0].ids == g[1].ids);
  FakeTapPipe probe(512, true);
  for (uint32_t id : g[0].ids) CHECK(std::find(probe.words.begin(), probe.words.end(), id) != probe.words.end());
  std::printf("2. a seeded sampled K = 7 run twice: identical, every id a word OK\n");
}

// A session of turns over PrefixSession; returns turn 2's per-iteration (pos, drafts right) and its ids.
struct Turn2 {
  std::vector<std::pair<uint32_t, uint32_t>> log;
  Ids ids;
  server::PrefixCache::Plan::Kind kind;
};
Turn2 session(bool side_request) {
  FakeTapPipe e(1024, true);
  Adapter a(e, kV, kV, 7, false, true);
  MallocAlloc alloc;
  server::PrefixSession s(a, 1 << 22, &alloc);
  CHECK_EQ(s.cache()->kv_form(), server::store_root(0, e.drafter_key()));
  server::Sampling greedy;
  const auto turn = [&](const Ids& prompt, uint32_t n, server::PrefixCache::Plan::Kind* kind) {
    const server::PrefixSession::Report r = s.begin(prompt);
    if (kind) *kind = r.kind;
    Ids out;
    while (out.size() < n) {
      const Ids burst = a.step_many(greedy, 7);
      const size_t keep = std::min<size_t>(burst.size(), n - out.size());
      for (size_t i = 0; i < keep; ++i) {
        out.push_back(burst[i]);
        s.fed(burst[i]);
      }
      if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
    }
    s.end();
    return out;
  };
  const Ids A = prompt_of(9, 40);
  const Ids g1 = turn(A, 24, nullptr);
  if (side_request) (void)turn(prompt_of(11, 33), 12, nullptr);
  Ids t2 = A;
  t2.insert(t2.end(), g1.begin(), g1.end());
  for (uint32_t x : {1007u, 1014u, 1021u}) t2.push_back(x);
  Turn2 r;
  const size_t before = e.log.size();
  r.ids = turn(t2, 40, &r.kind);
  r.log.assign(e.log.begin() + long(before), e.log.end());
  return r;
}

void test_prefix() {
  const Turn2 ref = session(false), got = session(true);
  CHECK(ref.kind == server::PrefixCache::Plan::Continue && got.kind == server::PrefixCache::Plan::Restore);
  CHECK(got.ids == ref.ids);
  CHECK(got.log == ref.log);
  CHECK(!got.log.empty() && got.log.front().second > 0);
  std::printf("3. a turn restored after a side request drafts as the uninterrupted session (%zu iterations, the "
              "first keeping %u), ids equal; the store keyed by the drafter OK\n",
              got.log.size(), got.log.front().second);
}

void test_refusals() {
  int refused = 0;
  const auto expect = [&](const std::function<void()>& f, const char* what) {
    try {
      f();
    } catch (const std::runtime_error& x) {
      refused += std::string(x.what()).find(what) != std::string::npos;
    }
  };
  FakeTapPipe none(64, false), with(64, true);
  expect([&] { Adapter a(none, kV, kV, 7, false, true); }, "not built with a drafter");
  expect([&] { Adapter a(with, kV, kV, 8, false, true); }, "--spec-max 8");
  expect([&] { Adapter a(with, kV, kV, 7, true, true); }, "pick one proposer");
  CHECK_EQ(refused, 3);
  std::printf("4. refused: no drafter, --spec-max past the block, with lookup OK\n");
}

}  // namespace

int main() {
  test_lossless();
  test_sampled();
  test_prefix();
  test_refusals();
  std::printf("pp_dflash_serve_test OK\n");
  return 0;
}

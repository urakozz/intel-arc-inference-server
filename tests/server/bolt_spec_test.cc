// Spec 26c Tasks 4 and 6, host only (D2's and D4's loop half, Review Focus 3, 4 and 5's bookkeeping; no device):
// cli::bolt::BoltEngineAdapterT's speculative paths over FakeBoltSpec - an engine with runtime::bolt::BoltEngine's speculative
// interface (verify(drafts, k) / verify_ids / read_verify_logits / commit(j, t) / max_verify_k / drafter_key, prefill / ingest /
// generate / read_logits_into, the snapshots) whose "model" is a deterministic next-token row over a 128-id vocabulary (a
// function of the last three ids) and whose attached drafter keeps a 1024-slot "ring" of projected positions and an anchor
// exactly as BoltEngine's contract says (commit projects verify rows 0..j; a plain token its row; a prefill the rows
// runtime::bolt::lili_prefill_r0 names, each chunk before its block hook; snapshots carry ring and anchor after the target's
// state) - and a fake LiLiCorr readback (runtime::dflash::LiliDraftOut: per slot the 8 best ids of the model's row, their
// log-probs, a lattice [7][8][8] whose rows depend on the previous pick AND on the ring's contents and the anchor, the device's
// greedy walk, planted misses).
//
//   1. D2 (greedy is lossless): --spec dflash at every fixed K 1..7, at K varying per iteration, with truncate_to cutting runs
//      at random kept lengths - every id equals --spec off's plain steps; the same for --spec lookup; drafts were accepted and
//      rejected both.
//   2. Review Focus 3: after every iteration of (1) (the commit flushed) the drafter's ring and anchor equal a plain run's at
//      the same position - the anchor is always position pos - 1 (Review Focus 5's anchor: the newest committed row).
//   3. D4 (sampled is exact): 10^6 seeded iterations from one state at K 4 for both q forms (`lattice`: softmax(realized / T)
//      over the 8 candidates, the host's walk over the readback's lattice; `lattice+lp`) and two samplings (Bolt's T 1.0 /
//      top-p 0.95 with no top-k - cli::bolt::filter_probs_top_p's path - and T 0.7 / top-k 12 / top-p 0.8): the token at n + 1
//      and the token at n + 2 given d_1 kept are distributed as the model's filtered p (chi-square p-value >= 0.01, spec 8 M4's
//      bar); every accepted draft is one of its slot's candidates; lookup's point mass the same way. A MUTANT residual (the
//      correction drawn from p instead of normalise(max(0, p - q))) fails the same bar.
//   4. The deferred commit: made before a snapshot, discarded by a restore; a sampled request with a repetition penalty
//      decodes plainly.
//   5. Review Focus 4 (snapshots carry the drafter): a prompt of 5000 prefilled with the block hook (snapshots at 2048 and
//      4096): a restore at 4096 + the rest gives the cold run's ring and anchor, and the drafts after it are the cold run's; a
//      sibling diverging one token after the 4096 block end (vLLM a608b6e6e0) restores there and its ring equals its own cold
//      prefill's; a snapshot without the drafter's ring (zeros) restores, the drafts change and the output does not; the
//      prefix store's root differs with and without the drafter (server::store_root).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "cli/bolt_serve.h"
#include "runtime/bolt/bolt_sizes.h"
#include "server/prefix_cache.h"
#include "server/spec_accept.h"

namespace {

using Ids = std::vector<uint32_t>;
using runtime::dflash::LiliDraftOut;
constexpr uint32_t V = 128, kWindow = 1024, kNone = 0xFFFFFFFFu;

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// The "model": the raw logits of the token after `ctx` - a function of its last three ids and its length's parity.
void model_row(const Ids& ctx, float* out) {
  uint64_t h = 0x26C ^ (ctx.size() & 1);
  for (size_t i = ctx.size() >= 3 ? ctx.size() - 3 : 0; i < ctx.size(); ++i) h = mix(h, ctx[i]);
  for (uint32_t v = 0; v < V; ++v) out[v] = float(mix(h, v) % 4000) / 400.0f - 5.0f;
}
uint32_t argmax(const float* r) { return uint32_t(std::max_element(r, r + V) - r); }

// The ring's one host layout (FakeBoltSpec::save_state appends it after the target's state, as BoltEngine appends
// LiliDrafter::save_ring): per slot (pos + 1, tap) - 0 an empty slot -, then the anchor (pos + 1, tap).
constexpr size_t kRingBytes = size_t(kWindow) * 12 + 12;

struct FakeBoltSpec {
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t, bool)>;
  Ids hist;
  std::vector<uint64_t> taps;   // taps[p]: the target's taps at position p (a hash of the prefix through p)
  uint32_t cur = 0, max_len_ = 1u << 20, verified_k = kNone;
  std::vector<float> last = std::vector<float>(V), vlog;
  Ids vids, vdrafts;
  bool drafter = false;
  std::vector<uint32_t> ring_pos = std::vector<uint32_t>(kWindow, 0);   // pos + 1, 0 = empty
  std::vector<uint64_t> ring_tap = std::vector<uint64_t>(kWindow, 0);
  uint32_t anchor_pos = 0;   // pos + 1, 0 = none
  uint64_t anchor = 0;
  uint64_t commits = 0, saves = 0, loads = 0, verifies = 0;
  BlockHook hook;

  void push(uint32_t id) {
    hist.push_back(id);
    taps.push_back(mix(taps.empty() ? 0x7Aull : taps.back(), id));
  }
  void project(uint32_t p) {   // the drafter's projection of position p (its taps) into slot p % 1024 and the anchor
    CHECK(p < taps.size());
    ring_pos[p % kWindow] = p + 1;
    ring_tap[p % kWindow] = taps[p];
    anchor_pos = p + 1;
    anchor = taps[p];
  }
  void refresh() {
    model_row(hist, last.data());
    cur = argmax(last.data());
  }
  void reset() {
    hist.clear();
    taps.clear();
    std::fill(ring_pos.begin(), ring_pos.end(), 0u);
    std::fill(ring_tap.begin(), ring_tap.end(), 0ull);
    anchor_pos = 0;
    anchor = 0;
    cur = 0;
    verified_k = kNone;
  }
  // BoltEngine::prefill's chunks (runtime::bolt::prefill_chunk_rows at 2048, ending at block ends with a hook) and, with a
  // drafter, Decision 7's rows of each chunk projected before its hook.
  void prefill(const Ids& ids) {
    const uint32_t base = pos(), end = base + uint32_t(ids.size());
    const bool hooked = static_cast<bool>(hook);
    size_t off = 0;
    while (off < ids.size()) {
      const uint32_t p0 = base + uint32_t(off);
      const uint32_t C = runtime::bolt::prefill_chunk_rows(p0, ids.size() - off, 2048, hooked, kBlock);
      for (uint32_t i = 0; i < C; ++i) push(ids[off + i]);
      if (drafter)
        for (uint32_t r = runtime::bolt::lili_prefill_r0(p0, C, end, kWindow, hooked, kBlock); r < C; ++r) project(p0 + r);
      off += C;
      if (hooked && off < ids.size() && (p0 + C) % kBlock == 0) hook(p0 + C, true);
    }
    refresh();
    if (hooked) hook(end, end % kBlock == 0);
  }
  Ids generate(uint32_t n) {   // a plain token a step (BoltEngine with the verify lists: verify(0) + commit(0))
    Ids out;
    for (uint32_t i = 0; i < n; ++i) {
      out.push_back(cur);
      push(cur);
      if (drafter) project(pos() - 1);
      refresh();
    }
    return out;
  }
  void ingest(const Ids& ids) {
    for (uint32_t id : ids) {
      cur = id;
      generate(1);
    }
  }
  uint32_t pos() const { return uint32_t(hist.size()); }
  uint32_t max_len() const { return max_len_; }
  uint32_t vocab() const { return V; }
  uint32_t pending() const { return cur; }
  void set_token(uint32_t id) { cur = id; }
  void read_logits_into(float* host) const { std::copy(last.begin(), last.end(), host); }
  void set_block_hook(BlockHook h) { hook = std::move(h); }
  bool bf16_state() const { return false; }
  size_t state_bytes() const { return 16 + (drafter ? kRingBytes : 0); }
  size_t kv_bytes(uint32_t) const { return 0; }
  void save_state(void* host) {
    ++saves;
    uint8_t* h = static_cast<uint8_t*>(host);
    std::memset(h, 0, 16);
    const uint32_t p = pos();
    std::memcpy(h, &p, 4);
    if (!drafter) return;
    h += 16;
    for (uint32_t s = 0; s < kWindow; ++s, h += 12) {
      std::memcpy(h, &ring_pos[s], 4);
      std::memcpy(h + 4, &ring_tap[s], 8);
    }
    std::memcpy(h, &anchor_pos, 4);
    std::memcpy(h + 4, &anchor, 8);
  }
  // `full` holds the prompt the snapshot was taken over (the fake's "KV and state" of positions < p).
  void load_state(const void* host, uint32_t p) {
    ++loads;
    CHECK(p <= hist.size() || p <= full.size());
    if (hist.size() < p) hist.assign(full.begin(), full.begin() + p);
    hist.resize(p);
    taps.clear();
    {
      const Ids keep = hist;
      hist.clear();
      for (uint32_t id : keep) push(id);
    }
    verified_k = kNone;
    if (!drafter) return;
    const uint8_t* h = static_cast<const uint8_t*>(host) + 16;
    for (uint32_t s = 0; s < kWindow; ++s, h += 12) {
      std::memcpy(&ring_pos[s], h, 4);
      std::memcpy(&ring_tap[s], h + 4, 8);
    }
    std::memcpy(&anchor_pos, h, 4);
    std::memcpy(&anchor, h + 4, 8);
  }
  Ids full;
  void save_kv(uint32_t, uint32_t, void*) {}
  void load_kv(uint32_t, uint32_t, const void*) {}

  // --- BoltEngine's speculative calls ---
  uint32_t max_verify_k() const { return 7; }
  void verify(const uint32_t* d, uint32_t k) {
    CHECK(verified_k == kNone && k <= 7);
    ++verifies;
    vlog.assign(size_t(k + 1) * V, 0.0f);
    vids.assign(k + 1, 0);
    vdrafts.assign(d, d + k);
    Ids ctx = hist;
    ctx.push_back(cur);
    for (uint32_t r = 0; r <= k; ++r) {
      model_row(ctx, &vlog[size_t(r) * V]);
      vids[r] = argmax(&vlog[size_t(r) * V]);
      if (r < k) ctx.push_back(d[r]);
    }
    verified_k = k;
  }
  const uint32_t* verify_ids() const { return vids.data(); }
  void read_verify_logits(float* host, uint32_t rows) const {
    CHECK(rows <= verified_k + 1);
    std::copy(vlog.begin(), vlog.begin() + size_t(rows) * V, host);
  }
  void commit(uint32_t j, uint32_t t) {
    CHECK(verified_k != kNone && j <= verified_k);
    const uint32_t n = pos();
    push(cur);
    for (uint32_t i = 0; i < j; ++i) push(vdrafts[i]);
    if (drafter)
      for (uint32_t r = 0; r <= j; ++r) project(n + r);   // verify rows 0..j: positions n .. n + j
    std::copy(vlog.begin() + size_t(j) * V, vlog.begin() + size_t(j + 1) * V, last.begin());
    cur = t;
    verified_k = kNone;
    ++commits;
  }
  uint64_t drafter_key() const { return drafter ? 0x26C26Cull : 0; }
};

// The fake LiLiCorr readback: slot s's 8 candidates are the 8 best ids of the model's row after [hist, cur, the greedy walk's
// first s ids] (the block's candidates do not follow a sampled walk: the device drafts all 7 slots at once), their full log-
// probs, a lattice whose next-row scores are the candidate's logit plus a term of (slot, previous pick, candidate) and of the
// ring and the anchor - so the drafts depend on what the drafter holds - and every `miss`-th position a planted preference for
// the 4th candidate; ids / chosen the greedy walk (ties to the lower index).
struct FakeLili {
  FakeBoltSpec& e;
  uint32_t miss = 5;
  LiliDraftOut out{};
  uint64_t calls = 0;
  uint64_t ring_hash() const {
    uint64_t h = mix(e.anchor_pos, e.anchor);
    for (uint32_t s = 0; s < kWindow; ++s) h = mix(h, mix(e.ring_pos[s], e.ring_tap[s]));
    return h;
  }
  const LiliDraftOut& draft(uint32_t k) {
    ++calls;
    CHECK(k >= 1 && k <= 7);
    if (e.drafter) CHECK(e.anchor_pos == e.pos());   // the anchor is position pos - 1 (Review Focus 5)
    const uint64_t rh = ring_hash();
    Ids ctx = e.hist;
    ctx.push_back(e.cur);
    out = LiliDraftOut{};
    out.k = 7;
    out.anchor = e.cur;
    out.pos = e.pos();
    out.anchor_valid = 1;
    std::vector<float> row(V);
    uint32_t prev = 0;
    for (uint32_t s = 0; s < 7; ++s) {   // the device always walks all 7 slots
      model_row(ctx, row.data());
      std::vector<uint32_t> ord(V);
      for (uint32_t v = 0; v < V; ++v) ord[v] = v;
      std::partial_sort(ord.begin(), ord.begin() + 8, ord.end(),
                        [&](uint32_t a, uint32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
      double m = row[ord[0]], z = 0;
      for (uint32_t v = 0; v < V; ++v) z += std::exp(double(row[v]) - m);
      const float lse = float(m + std::log(z));
      const bool planted = (ctx.size() + s) % miss == 0;
      for (uint32_t c = 0; c < 8; ++c) {
        out.cand[s][c] = ord[c];
        out.logit[s][c] = row[ord[c]];
        out.lp[s][c] = row[ord[c]] - lse;
      }
      for (uint32_t p = 0; p < 8; ++p)
        for (uint32_t c = 0; c < 8; ++c) {
          const uint32_t pp = s == 0 ? 0 : p;   // slot 0: the start row, broadcast over the previous index
          const float noise = float(mix(mix(rh, s * 64 + pp * 8 + c), 0x5EED) % 1000) / 2000.0f;   // [0, 0.5)
          out.lattice[s][p][c] = 2.0f * (row[ord[c]] - row[ord[0]]) + noise + (planted && c == 3 ? 4.0f : 0.0f);
        }
      uint32_t best = 0;
      for (uint32_t c = 1; c < 8; ++c)
        if (out.lattice[s][prev][c] > out.lattice[s][prev][best]) best = c;
      for (uint32_t c = 0; c < 8; ++c) out.scores[s][c] = out.lattice[s][prev][c];
      out.chosen[s] = best;
      out.ids[s] = ord[best];
      prev = best;
      ctx.push_back(ord[best]);
    }
    return out;
  }
};

using Adapter = cli::bolt::BoltEngineAdapterT<FakeBoltSpec>;
cli::bolt::BoltDraftHooks hooks(FakeLili& d) {
  cli::bolt::BoltDraftHooks h;
  h.draft = [&d](uint32_t k) -> const LiliDraftOut& { return d.draft(k); };
  return h;
}

int g_iters = 1000000;
uint64_t g_seed = 20261023;
const Ids kPrompt = {3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26, 43, 38, 32, 79, 50, 28, 84, 19};

Ids plain_run(uint32_t n) {
  FakeBoltSpec e;
  e.drafter = true;
  Adapter a(e, V);
  a.prefill(kPrompt);
  server::Sampling g;
  Ids out;
  while (out.size() < n) out.push_back(a.step(g));
  return out;
}

void same_ring(const FakeBoltSpec& a, const FakeBoltSpec& b, const char* what) {
  if (a.ring_pos != b.ring_pos || a.ring_tap != b.ring_tap || a.anchor_pos != b.anchor_pos || a.anchor != b.anchor) {
    std::fprintf(stderr, "%s: the drafter's ring / anchor differ (pos %u vs %u)\n", what, a.pos(), b.pos());
    CHECK(false);
  }
}

void greedy_case() {
  const uint32_t N = 600;
  const Ids want = plain_run(N);
  // modes: 0..6 fixed K = mode + 1; 7 K varying 1..7; 8 K 7 with truncated runs
  for (int mode = 0; mode < 9; ++mode) {
    FakeBoltSpec e, twin;
    e.drafter = twin.drafter = true;
    FakeLili d{e};
    Adapter a(e, V, 7, false, hooks(d));
    a.prefill(kPrompt);
    twin.prefill(kPrompt);
    server::Sampling g;
    std::mt19937_64 rng(42);
    Ids got;
    uint32_t it = 0;
    while (got.size() < N) {
      const uint32_t k = mode < 7 ? uint32_t(mode + 1) : mode == 7 ? 1 + it % 7 : 7;
      const Ids run = a.step_many(g, k);
      size_t keep = run.size();
      if (mode == 8 && run.size() > 1 && rng() % 3 == 0) {   // a stop inside the run: keep a prefix
        keep = 1 + rng() % run.size();
        a.truncate_to(a.pos() - uint32_t(run.size() - keep));
      }
      a.flush_commit();
      got.insert(got.end(), run.begin(), run.begin() + keep);
      // Review Focus 3: the ring and the anchor after the commit = a plain run's to the same position
      twin.ingest(Ids(run.begin(), run.begin() + keep));
      CHECK(twin.pos() == e.pos());
      same_ring(e, twin, "greedy dflash");
      ++it;
    }
    got.resize(N);
    std::printf("greedy dflash (%s): %llu iterations, %llu / %llu drafts accepted; ring and anchor = the plain run's after "
                "every commit\n",
                mode < 7 ? ("K " + std::to_string(mode + 1)).c_str() : mode == 7 ? "K 1..7" : "K 7, truncated runs",
                (unsigned long long)a.spec_iterations(), (unsigned long long)a.spec_accepted(),
                (unsigned long long)a.spec_drafted());
    CHECK(got == want);
    CHECK(a.spec_accepted() > 0 && a.spec_accepted() < a.spec_drafted());
  }
  {   // lookup: a proposer with planted misses
    FakeBoltSpec e;
    e.drafter = true;   // its ring follows every commit too
    Adapter a(e, V, 7, true);
    a.prefill(kPrompt);
    server::Sampling g;
    Ids got;
    while (got.size() < N) {
      const Ids run = a.step_drafts(g, [&](uint32_t x) {
        Ids ctx = e.hist, d;
        ctx.push_back(x);
        std::vector<float> row(V);
        for (uint32_t j = 0; j < 6; ++j) {
          model_row(ctx, row.data());
          uint32_t id = argmax(row.data());
          if ((ctx.size() + j) % 5 == 0) id = (id + 1) % V;
          d.push_back(id);
          ctx.push_back(id);
        }
        return d;
      });
      got.insert(got.end(), run.begin(), run.end());
    }
    got.resize(N);
    std::printf("greedy lookup: %llu / %llu drafts accepted\n", (unsigned long long)a.spec_accepted(),
                (unsigned long long)a.spec_drafted());
    CHECK(got == want);
  }
}

double gammq(double a, double x) {   // Numerical Recipes gammq, as muse_spec_test's
  if (x <= 0) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; ++n) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (int i = 1; i < 1000; ++i) {
    const double an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (std::fabs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1) < 1e-15) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

double chi2_pvalue(const std::vector<uint64_t>& counts, const server::FilteredProbs& f, const std::string& what) {
  uint64_t n = 0;
  for (uint64_t c : counts) n += c;
  if (n == 0) {
    std::printf("  %s: no sample  <-- the case never reached this token\n", what.c_str());
    return 0.0;
  }
  double chi2 = 0;
  int bins = 0;
  bool outside = false;
  for (uint32_t id = 0; id < counts.size(); ++id) {
    const double p = f.prob(id);
    if (p == 0) {
      outside = outside || counts[id] != 0;
      continue;
    }
    const double e = p * double(n);
    chi2 += (double(counts[id]) - e) * (double(counts[id]) - e) / e;
    ++bins;
  }
  const double pv = outside ? 0.0 : bins > 1 ? gammq(0.5 * (bins - 1), 0.5 * chi2) : 1.0;
  std::printf("  %s: n %llu, bins %d, chi2 %.2f, p-value %.4f%s\n", what.c_str(), (unsigned long long)n, bins, chi2, pv,
              outside ? " (an id outside p's support)" : "");
  return pv;
}

server::FilteredProbs target_p(const Ids& ctx, const server::Sampling& s) {
  std::vector<float> row(V);
  model_row(ctx, row.data());
  return cli::bolt::filter_probs_top_p(row.data(), V, s);
}

// D4 from one state: the token at n + 1, and the token at n + 2 given d_1 = the greedy walk's first id was kept, against the
// model's filtered p. Each run is cut back by truncate_to (keep x, or x and d_1), which leaves the next token pending - d_1 /
// d_2 when kept, the correction otherwise - and the state is rewound. The first iteration (the anchor drawn from the prefill's
// row) is a warm-up.
void sampled_case(const server::Sampling& s0, cli::bolt::BoltDraftQ q, bool lookup, const char* name) {
  const int kIters = g_iters;
  FakeBoltSpec e;
  FakeLili d{e};
  Adapter a(e, V, 4, lookup, lookup ? cli::bolt::BoltDraftHooks{} : hooks(d), q);
  e.prefill(kPrompt);
  server::Sampling s = s0;
  s.greedy = false;
  s.has_seed = true;
  s.seed = g_seed;
  // lookup's point-mass drafts: the model's greedy chain from the anchor (the same every iteration: the state is rewound)
  const auto propose = [&](uint32_t anchor) {
    Ids ctx = e.hist, d4;
    ctx.push_back(anchor);
    std::vector<float> row(V);
    for (int j = 0; j < 4; ++j) {
      model_row(ctx, row.data());
      d4.push_back(argmax(row.data()));
      ctx.push_back(d4.back());
    }
    return d4;
  };
  Ids base, prop;
  uint32_t x = 0, n = 0, d1 = 0;
  std::vector<uint64_t> first(V, 0), second(V, 0);
  uint64_t outside = 0;
  for (int i = 0; i <= kIters; ++i) {
    const Ids run = lookup ? a.step_drafts(s, [&](uint32_t anchor) { return prop = propose(anchor); }) : a.step_many(s, 4);
    if (i == 0) {   // the warm-up: the anchor x drawn from the prefill's row; then every iteration starts from (base, x)
      a.truncate_to(uint32_t(kPrompt.size()) + 1);
      base = e.hist;
      base.pop_back();
      x = run[0];
      n = uint32_t(kPrompt.size());
      d1 = lookup ? prop[0] : d.out.ids[0];
      e.hist = base;
      e.taps.resize(base.size());
      e.cur = x;
      continue;
    }
    CHECK(run.size() >= 1 && run[0] == x);
    for (size_t j = 1; j < run.size() && !lookup; ++j) {   // every accepted draft is one of its slot's candidates
      bool in = false;
      for (uint32_t c = 0; c < 8; ++c) in = in || d.out.cand[j - 1][c] == run[j];
      outside += in ? 0 : 1;
    }
    if (run.size() >= 2) {
      a.truncate_to(n + 2);   // keep x, d_1: pending the token at n + 2
      ++first[run[1]];
      if (run[1] == d1) ++second[e.cur];
    } else {
      a.truncate_to(n + 1);   // keep x: pending the correction at n + 1
      ++first[e.cur];
    }
    e.hist = base;
    e.taps.resize(base.size());
    e.cur = x;
  }
  CHECK_EQ(outside, uint64_t(0));
  Ids ctx = base;
  ctx.push_back(x);
  std::printf("%s:\n", name);
  CHECK(chi2_pvalue(first, target_p(ctx, s), "the token at n + 1 vs p") >= 0.01);
  ctx.push_back(d1);
  CHECK(chi2_pvalue(second, target_p(ctx, s), "the token at n + 2 | d_1 kept vs p") >= 0.01);
}

// The mutant: on a rejection the correction is drawn from p itself, not from normalise(max(0, p - q)) - the output then
// over-weights the drafts' ids; the same chi-square bar must catch it (the test has the power the real rule passes with).
void mutant_case(const server::Sampling& s0) {
  const int kIters = std::max(200000, g_iters / 5);
  FakeBoltSpec e;
  FakeLili d{e};
  e.prefill(kPrompt);
  server::Sampling s = s0;
  s.greedy = false;
  const float T = server::dflash_temperature(s);
  const LiliDraftOut& o = d.draft(1);
  float v[8];
  for (uint32_t c = 0; c < 8; ++c) v[c] = o.lattice[0][0][c];
  const server::SparseQ q = server::dflash_q(o.cand[0], v, 8, T);
  std::mt19937_64 rng(g_seed);
  std::vector<uint64_t> first(V, 0);
  Ids ctx = e.hist;
  ctx.push_back(e.cur);
  std::vector<float> rows(size_t(2) * V);
  model_row(ctx, rows.data());
  for (int i = 0; i < kIters; ++i) {
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double acc = 0;
    uint32_t pick = 0;
    for (uint32_t c = 0; c < 8; ++c) {
      acc += q.probs[c];
      if (u < acc) {
        pick = c;
        break;
      }
    }
    const uint32_t d1 = q.ids[pick];
    const server::FilteredProbs p = cli::bolt::filter_probs_top_p(rows.data(), V, s);
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < std::min(1.0, p.prob(d1) / q.prob(d1))) {
      ++first[d1];
    } else {
      ++first[server::draw(p, rng)];   // the MUTANT residual
    }
  }
  std::printf("mutant residual (the correction from p, not max(0, p - q)):\n");
  CHECK(chi2_pvalue(first, cli::bolt::filter_probs_top_p(rows.data(), V, s), "the token at n + 1 vs p") < 0.01);
}

// 4: the deferred commit - flushed before a snapshot, discarded by a restore; a penalised sampled request decodes plainly.
void deferred_case() {
  FakeBoltSpec e;
  e.drafter = true;
  FakeLili d{e};
  Adapter a(e, V, 7, false, hooks(d));
  a.prefill(kPrompt);
  e.full = e.hist;
  server::Sampling g;
  const Ids run = a.step_many(g, 7);
  CHECK(e.commits == 0 && a.pos() == uint32_t(kPrompt.size() + run.size()));
  std::vector<uint8_t> host(a.state_bytes());
  CHECK_EQ(host.size(), size_t(16) + kRingBytes);   // the ring and the anchor after the target's state
  a.save_state(host.data());
  CHECK(e.commits == 1 && e.saves == 1 && e.pos() == a.pos());
  a.step_many(g, 7);
  CHECK(e.commits == 1);
  a.load_state(host.data(), uint32_t(kPrompt.size()));
  CHECK(e.commits == 1 && e.loads == 1 && a.pos() == kPrompt.size());   // the pending run was discarded
  CHECK(a.drafter_key() == 0x26C26Cull);
  // a sampled request with a repetition penalty: plain steps (its p depends on ids a run has not committed)
  server::Sampling pen;
  pen.greedy = false;
  pen.repetition_penalty = 1.3f;
  server::SeenIds seen;
  pen.seen = &seen;
  const uint64_t v0 = e.verifies;
  for (int i = 0; i < 20; ++i) CHECK(a.step_many(pen, 7).size() == 1);
  CHECK(e.verifies == v0);
  std::printf("deferred commit: flushed by save_state, discarded by load_state; a penalised sampled request steps plainly\n");
}

// 5: Review Focus 4 - the drafter in the snapshots.
void snapshot_case() {
  Ids prompt;
  std::mt19937 g(7);
  for (int i = 0; i < 5000; ++i) prompt.push_back(g() % V);
  std::map<uint32_t, std::vector<uint8_t>> snaps;
  // the cold run: prefill with the hook (snapshots at 2048, 4096 and the prompt end), then 40 greedy iterations
  FakeBoltSpec cold;
  cold.drafter = true;
  cold.full = prompt;
  FakeLili cd{cold};
  Adapter ca(cold, V, 7, false, hooks(cd));
  ca.set_block_hook([&](uint32_t end, bool block_end) {
    if (!block_end) return;
    std::vector<uint8_t> h(ca.state_bytes());
    ca.save_state(h.data());
    snaps[end] = h;
  });
  ca.prefill(prompt);
  CHECK(snaps.count(2048) && snaps.count(4096));
  server::Sampling greedy;
  Ids cold_out, cold_drafts;
  while (cold_out.size() < 120) {
    const Ids run = ca.step_many(greedy, 7);
    for (uint32_t s = 0; s < 7; ++s) cold_drafts.push_back(cd.out.ids[s]);
    cold_out.insert(cold_out.end(), run.begin(), run.end());
  }
  ca.flush_commit();
  // a restore at 4096, the rest of the prompt prefilled: the cold run's ring / anchor and its drafts
  FakeBoltSpec warm;
  warm.drafter = true;
  warm.full = prompt;
  FakeLili wd{warm};
  Adapter wa(warm, V, 7, false, hooks(wd));
  wa.load_state(snaps[4096].data(), 4096);
  {
    FakeBoltSpec ref;   // the cold run at the prompt's end
    ref.drafter = true;
    ref.prefill(prompt);
    wa.prefill(Ids(prompt.begin() + 4096, prompt.end()));
    same_ring(warm, ref, "restore at 4096 + the rest");
  }
  Ids warm_out, warm_drafts;
  while (warm_out.size() < 120) {
    const Ids run = wa.step_many(greedy, 7);
    for (uint32_t s = 0; s < 7; ++s) warm_drafts.push_back(wd.out.ids[s]);
    warm_out.insert(warm_out.end(), run.begin(), run.end());
  }
  CHECK(warm_out == cold_out && warm_drafts == cold_drafts);
  // a608b6e6e0: a sibling sharing the prompt to 4097, then another token - it restores at the block end 4096 (the largest
  // <= the common prefix) and its ring equals its own cold prefill's
  Ids sib(prompt.begin(), prompt.begin() + 4097);
  sib.push_back((prompt[4097] + 1) % V);
  for (int i = 0; i < 300; ++i) sib.push_back(g() % V);
  {
    FakeBoltSpec s1, s2;
    s1.drafter = s2.drafter = true;
    s1.full = prompt;
    Adapter sa(s1, V);
    sa.load_state(snaps[4096].data(), 4096);
    sa.prefill(Ids(sib.begin() + 4096, sib.end()));
    s2.prefill(sib);
    same_ring(s1, s2, "the sibling restored at its junction's block end");
  }
  // a snapshot without the drafter's ring: zeros - the drafts change, the output does not
  FakeBoltSpec bare;
  bare.drafter = true;
  bare.full = prompt;
  FakeLili bd{bare};
  Adapter ba(bare, V, 7, false, hooks(bd));
  std::vector<uint8_t> zeroed = snaps[4096];
  std::fill(zeroed.begin() + 16, zeroed.end(), uint8_t(0));
  ba.load_state(zeroed.data(), 4096);
  bare.anchor_pos = 0;   // (the zero anchor slot: the fake checks the anchor after the prefill re-projects it)
  ba.prefill(Ids(prompt.begin() + 4096, prompt.end()));
  Ids bare_out, bare_drafts;
  while (bare_out.size() < 120) {
    const Ids run = ba.step_many(greedy, 7);
    for (uint32_t s = 0; s < 7; ++s) bare_drafts.push_back(bd.out.ids[s]);
    bare_out.insert(bare_out.end(), run.begin(), run.end());
  }
  bare_out.resize(120);
  Ids cold120 = cold_out;
  cold120.resize(120);
  CHECK(bare_out == cold120);
  CHECK(bare_drafts != cold_drafts);
  // the prefix store's root: the drafter's key moves it
  const uint64_t key = cli::bolt::store_key(false);
  CHECK(server::store_root(key, 0x26C26Cull) != server::store_root(key, 0));
  std::printf("snapshots: a restore at 4096 + the rest = the cold ring / anchor and drafts; the junction's sibling = its cold "
              "prefill; a ring of zeros changes the drafts, not the output; the store root differs with the drafter\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_iters = std::atoi(argv[1]);
  if (argc > 2) g_seed = std::strtoull(argv[2], nullptr, 10);
  greedy_case();
  deferred_case();
  snapshot_case();
  server::Sampling bolt;   // Bolt's published defaults (Decision 12): T 1.0, top-p 0.95, no top-k
  bolt.temperature = 1.0f;
  bolt.top_k = 0;
  bolt.top_p = 0.95f;
  server::Sampling tight;
  tight.temperature = 0.7f;
  tight.top_k = 12;
  tight.top_p = 0.8f;
  using Q = cli::bolt::BoltDraftQ;
  sampled_case(bolt, Q::Lattice, false, "D4 dflash, q lattice, T 1.0 / top-p 0.95 (no top-k)");
  sampled_case(bolt, Q::LatticeLp, false, "D4 dflash, q lattice+lp, T 1.0 / top-p 0.95 (no top-k)");
  sampled_case(tight, Q::Lattice, false, "D4 dflash, q lattice, T 0.7 / top-k 12 / top-p 0.8");
  sampled_case(tight, Q::LatticeLp, false, "D4 dflash, q lattice+lp, T 0.7 / top-k 12 / top-p 0.8");
  sampled_case(bolt, Q::Lattice, true, "D4 lookup (point mass), T 1.0 / top-p 0.95 (no top-k)");
  mutant_case(bolt);
  std::printf("bolt_spec_test: PASS\n");
  return 0;
}

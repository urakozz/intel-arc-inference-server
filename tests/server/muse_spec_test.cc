// Spec 23f Task 8, host only (D2's and D4's loop half, no device): cli::muse::MuseEngineAdapterT's speculative paths
// over FakeSpec - an engine with MuseEngine's speculative interface (verify(drafts, k) / verify_ids /
// read_verify_logits / commit(j, t) / max_verify_k, prefill / ingest / generate / read_logits_into) whose "model" is a
// deterministic next-token row over a 64-id vocabulary (a function of the last three ids), and a fake drafter filling
// runtime::dflash::MuseDraftOut's 15 rows (the target's argmax chain with planted misses, the scores the row's own raw
// logits, so acceptance is partial and the order the device's).
//
//   1. D2 (greedy is lossless): --spec dflash at a fixed K 15, at K varying 1..15 per iteration and with truncate_to
//      cutting runs at random kept lengths (a stop inside a run) - every id equals --spec off's plain steps, the same
//      for --spec lookup's step_drafts over a proposer with planted misses; drafts were accepted and rejected both.
//   2. D4 (sampled is exact): 10^6 seeded iterations from one state at K 4, for both q forms (raw: softmax(z / T) over
//      the 16 candidates' raw logits; capped: over 20 tanh(z x 0.196 / 20)) and two samplings (T 1.0 / top-k 64 /
//      top-p 0.95 - Muse's defaults - and T 0.7 / top-k 12 / top-p 0.8) - the token at n + 1 (d_1 kept, or the
//      correction) is distributed as the model's capped, filtered p (chi-square p-value >= 0.01, spec 8 M4's bar), and
//      the second token given d_1 kept as p at n + 2. Also lookup's point mass (sampled) the same way.
//   3. The commit is deferred, and made before a snapshot (save_state flushes it) or discarded by a restore / reset.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "cli/muse_serve.h"
#include "server/spec_accept.h"

namespace {

using Ids = std::vector<uint32_t>;
using runtime::dflash::MuseDraftOut;
constexpr uint32_t V = 64;
constexpr float kMult = 0.19611613513818404f, kCap = 20.0f;

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// The "model": the raw logits of the token after `ctx` - a function of its last three ids and its length's parity.
void model_row(const Ids& ctx, float* out) {
  uint64_t h = 0x5EED ^ (ctx.size() & 1);
  for (size_t i = ctx.size() >= 3 ? ctx.size() - 3 : 0; i < ctx.size(); ++i) h = mix(h, ctx[i]);
  for (uint32_t v = 0; v < V; ++v) out[v] = float(mix(h, v) % 4000) / 100.0f - 20.0f;   // raw, in the cap's bend
}
uint32_t argmax(const float* r) { return uint32_t(std::max_element(r, r + V) - r); }

struct FakeSpec {
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t, bool)>;
  Ids hist;
  uint32_t cur = 0, max_len_ = 1u << 20, verified_k = 0xFFFFFFFFu;
  std::vector<float> last = std::vector<float>(V), vlog;
  Ids vids, vdrafts;
  uint64_t commits = 0, saves = 0, loads = 0;

  void reset() {
    hist.clear();
    cur = 0;
    verified_k = 0xFFFFFFFFu;
  }
  void refresh() {
    model_row(hist, last.data());
    cur = argmax(last.data());
  }
  void prefill(const Ids& ids) {
    hist.insert(hist.end(), ids.begin(), ids.end());
    refresh();
  }
  void ingest(const Ids& ids) { prefill(ids); }
  Ids generate(uint32_t n) {
    Ids out;
    for (uint32_t i = 0; i < n; ++i) {
      out.push_back(cur);
      hist.push_back(cur);
      refresh();
    }
    return out;
  }
  uint32_t pos() const { return uint32_t(hist.size()); }
  uint32_t max_len() const { return max_len_; }
  uint32_t vocab() const { return V; }
  uint32_t pending() const { return cur; }
  void set_token(uint32_t id) { cur = id; }
  void read_logits_into(float* host) const { std::copy(last.begin(), last.end(), host); }
  void set_block_hook(BlockHook) {}
  size_t state_bytes() const { return 8; }
  size_t kv_bytes(uint32_t) const { return 0; }
  void save_state(void*) { ++saves; }
  void load_state(const void*, uint32_t p) {
    hist.resize(p);
    ++loads;
  }
  void save_kv(uint32_t, uint32_t, void*) {}
  void load_kv(uint32_t, uint32_t, const void*) {}

  // --- MuseEngine's speculative calls ---
  uint32_t spec_rows() const { return 16; }
  uint32_t max_verify_k() const { return 15; }
  void verify(const uint32_t* d, uint32_t k) {
    CHECK(verified_k == 0xFFFFFFFFu);
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
    CHECK(verified_k != 0xFFFFFFFFu && j <= verified_k);
    hist.push_back(cur);
    for (uint32_t i = 0; i < j; ++i) hist.push_back(vdrafts[i]);
    std::copy(vlog.begin() + size_t(j) * V, vlog.begin() + size_t(j + 1) * V, last.begin());
    cur = t;
    verified_k = 0xFFFFFFFFu;
    ++commits;
  }
  uint64_t drafter_key() const { return 0x23F; }
};

// The fake drafter: row j's candidates are the 16 best of the model's row after [hist, cur, d_1 .. d_j] - the device's
// order (raw logit desc) - except that every `miss`-th position puts a wrong id first; scores raw or capped.
struct FakeDrafter {
  FakeSpec& e;
  bool capped = false;
  uint32_t miss = 4;
  MuseDraftOut out{};
  uint64_t calls = 0;
  const MuseDraftOut& draft(uint32_t k) {
    ++calls;
    Ids ctx = e.hist;
    ctx.push_back(e.cur);
    out.k = 15;
    out.anchor = e.cur;
    out.pos = e.pos();
    std::vector<float> row(V);
    for (uint32_t j = 0; j < k && j < 15; ++j) {   // the rows the caller reads (the device always fills 15)
      model_row(ctx, row.data());
      std::vector<uint32_t> ord(V);
      for (uint32_t v = 0; v < V; ++v) ord[v] = v;
      std::partial_sort(ord.begin(), ord.begin() + 16, ord.end(),
                        [&](uint32_t a, uint32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
      if ((ctx.size() + j) % miss == 0) std::swap(ord[0], ord[5]);   // a planted miss: the 6th best ranked first
      for (uint32_t c = 0; c < 16; ++c) {
        out.cand[j][c] = ord[c];
        const float z = row[ord[c]];
        out.unary[j][c] = out.scores[j][c] = capped ? std::tanh(z * kMult / kCap) * kCap : z;
      }
      out.ids[j] = ord[0];
      out.chosen[j] = 0;
      ctx.push_back(ord[0]);
    }
    return out;
  }
};

using Adapter = cli::muse::MuseEngineAdapterT<FakeSpec>;

cli::muse::MuseDraftHooks hooks(FakeDrafter& d) {
  cli::muse::MuseDraftHooks h;
  h.draft = [&d](uint32_t k) -> const MuseDraftOut& { return d.draft(k); };
  return h;
}

int g_iters = 1000000;
uint64_t g_seed = 20261016;

const Ids kPrompt = {3, 14, 15, 9, 26, 53, 58, 9, 7, 9, 32, 38, 46, 26, 43};

Ids plain_run(uint32_t n) {
  FakeSpec e;
  Adapter a(e, V, kMult, kCap);
  a.prefill(kPrompt);
  server::Sampling g;
  Ids out;
  while (out.size() < n) out.push_back(a.step(g));
  return out;
}

void greedy_case() {
  const uint32_t N = 600;
  const Ids want = plain_run(N);
  for (int mode = 0; mode < 3; ++mode) {
    FakeSpec e;
    FakeDrafter d{e};
    Adapter a(e, V, kMult, kCap, 15, false, hooks(d));
    a.prefill(kPrompt);
    server::Sampling g;
    std::mt19937_64 rng(42);
    Ids got;
    uint32_t it = 0;
    while (got.size() < N) {
      const uint32_t k = mode == 1 ? 1 + it % 15 : 15;
      const Ids run = a.step_many(g, k);
      size_t keep = run.size();
      if (mode == 2 && run.size() > 1 && rng() % 3 == 0) {   // a stop inside the run: keep a prefix
        keep = 1 + rng() % run.size();
        a.truncate_to(a.pos() - uint32_t(run.size() - keep));
      }
      got.insert(got.end(), run.begin(), run.begin() + keep);
      ++it;
    }
    got.resize(N);
    std::printf("greedy dflash (%s): %llu iterations, %llu / %llu drafts accepted\n",
                mode == 0 ? "K 15" : mode == 1 ? "K 1..15" : "K 15, truncated runs",
                (unsigned long long)a.spec_iterations(), (unsigned long long)a.spec_accepted(),
                (unsigned long long)a.spec_drafted());
    CHECK(got == want);
    CHECK(a.spec_accepted() > 0 && a.spec_accepted() < a.spec_drafted());
  }
  {   // lookup: a proposer with planted misses
    FakeSpec e;
    Adapter a(e, V, kMult, kCap, 15, true);
    a.prefill(kPrompt);
    server::Sampling g;
    Ids got;
    uint64_t n_prop = 0;
    while (got.size() < N) {
      const Ids run = a.step_drafts(g, [&](uint32_t x) {
        ++n_prop;
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
    std::printf("greedy lookup: %llu proposals, %llu / %llu drafts accepted\n", (unsigned long long)n_prop,
                (unsigned long long)a.spec_accepted(), (unsigned long long)a.spec_drafted());
    CHECK(got == want);
  }
}

double gammq(double a, double x) {   // Numerical Recipes gammq, as dflash_accept_test's
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
  double chi2 = 0;
  int bins = 0;
  for (uint32_t id = 0; id < counts.size(); ++id) {
    const double p = f.prob(id);
    if (p == 0) {
      CHECK(counts[id] == 0);
      continue;
    }
    const double e = p * double(n);
    chi2 += (double(counts[id]) - e) * (double(counts[id]) - e) / e;
    ++bins;
  }
  const double pv = bins > 1 ? gammq(0.5 * (bins - 1), 0.5 * chi2) : 1.0;
  std::printf("  %s: n %llu, bins %d, chi2 %.2f, p-value %.4f\n", what.c_str(), (unsigned long long)n, bins, chi2, pv);
  return pv;
}

// D4 from one state: the token at n + 1, and the token at n + 2 given d_1 = the drafter's (or proposer's) first id
// was kept, against the model's capped, filtered p. Each iteration's run is cut back by truncate_to (keep x, or x and
// d_1), which leaves the next token pending - d_1 / d_2 when kept, the correction otherwise - and the state is rewound.
void sampled_case(const server::Sampling& s0, bool capped, bool lookup, const char* name) {
  const int kIters = g_iters;
  FakeSpec e;
  FakeDrafter d{e, capped};
  Adapter a(e, V, kMult, kCap, 4, lookup, lookup ? cli::muse::MuseDraftHooks{} : hooks(d));
  e.prefill(kPrompt);   // the engine directly: no "fresh" first-id resample, the state stays fixed
  const Ids base = e.hist;
  const uint32_t x = e.cur, n = e.pos();
  const Ids prop = {7, 5, 9, 12};   // lookup's fixed point-mass drafts
  const uint32_t d1 = lookup ? prop[0] : d.draft(4).ids[0];
  server::Sampling s = s0;
  s.greedy = false;
  s.has_seed = true;
  s.seed = g_seed;
  std::vector<uint64_t> first(V, 0), second(V, 0);
  for (int i = 0; i < kIters; ++i) {
    const Ids run = lookup ? a.step_drafts(s, [&](uint32_t) { return prop; }) : a.step_many(s, 4);
    CHECK(run.size() >= 1 && run[0] == x);
    if (run.size() >= 2) {
      a.truncate_to(n + 2);   // keep x, d_1: pending the token at n + 2
      ++first[run[1]];
      if (run[1] == d1) ++second[e.cur];
    } else {
      a.truncate_to(n + 1);   // keep x: pending the correction at n + 1
      ++first[e.cur];
    }
    e.hist = base;
    e.cur = x;
  }
  server::Sampling f = s;
  f.logit_mult = kMult;
  f.softcap = kCap;
  std::vector<float> row(V);
  Ids ctx = base;
  ctx.push_back(x);
  model_row(ctx, row.data());
  std::printf("%s:\n", name);
  CHECK(chi2_pvalue(first, server::filter_probs(row.data(), V, f), "the token at n + 1 vs p") >= 0.01);
  ctx.push_back(d1);
  model_row(ctx, row.data());
  CHECK(chi2_pvalue(second, server::filter_probs(row.data(), V, f), "the token at n + 2 | d_1 kept vs p") >= 0.01);
}

// 3: the deferred commit - flushed before a snapshot, discarded by a restore.
void deferred_case() {
  FakeSpec e;
  FakeDrafter d{e};
  Adapter a(e, V, kMult, kCap, 15, false, hooks(d));
  a.prefill(kPrompt);
  server::Sampling g;
  const Ids run = a.step_many(g, 15);
  CHECK(e.commits == 0 && a.pos() == uint32_t(kPrompt.size() + run.size()));
  std::vector<uint8_t> host(a.state_bytes());
  a.save_state(host.data());
  CHECK(e.commits == 1 && e.saves == 1 && e.pos() == a.pos());
  a.step_many(g, 15);
  CHECK(e.commits == 1);
  a.load_state(host.data(), uint32_t(kPrompt.size()));
  CHECK(e.commits == 1 && e.loads == 1 && a.pos() == kPrompt.size());   // the pending run was discarded
  CHECK(a.drafter_key() == 0x23F);
  std::printf("deferred commit: flushed by save_state, discarded by load_state\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_iters = std::atoi(argv[1]);
  if (argc > 2) g_seed = std::strtoull(argv[2], nullptr, 10);
  greedy_case();
  deferred_case();
  server::Sampling muse;   // Muse-Glimmer's defaults (decision 4)
  muse.temperature = 1.0f;
  muse.top_k = 64;
  muse.top_p = 0.95f;
  server::Sampling tight;
  tight.temperature = 0.7f;
  tight.top_k = 12;
  tight.top_p = 0.8f;
  sampled_case(muse, false, false, "D4 dflash, q raw, T 1.0 / top-k 64 / top-p 0.95");
  sampled_case(muse, true, false, "D4 dflash, q capped, T 1.0 / top-k 64 / top-p 0.95");
  sampled_case(tight, false, false, "D4 dflash, q raw, T 0.7 / top-k 12 / top-p 0.8");
  sampled_case(tight, true, false, "D4 dflash, q capped, T 0.7 / top-k 12 / top-p 0.8");
  sampled_case(muse, false, true, "D4 lookup (point mass), T 1.0 / top-k 64 / top-p 0.95");
  std::printf("muse_spec_test: PASS\n");
  return 0;
}

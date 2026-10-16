// Spec 23e (plan 23e Task 4, host side): Muse-Glimmer-30B behind b70-serve without a device.
// cli::muse::MuseEngineAdapterT - the adapter b70-serve builds over runtime::muse::MuseEngine - runs here over FakeMuse,
// an engine with MuseEngine's interface whose rings and full-layer KV are host buffers in MuseBuffers' real layout
// (runtime::muse::persistent_sizes per device of a placement: one card or two) and whose snapshot calls are
// MuseEngine's own loops over runtime::muse::state_runs / kv_runs, memcpy for the device copies. Its "model" reads what
// Muse's attention reads: the query at position q sees the sliding layers' ring rows of [q - 2047, q] (through the
// slots q & 4095) and the full layers' rows [0, q]; position q's K / V rows of every layer are a function of the id, q,
// the layer and the keys the query at q attends to before its own, and the next id a function of what the last query
// attends to (its own key included) - so a restore that misplaces one row of one ring or one full layer, or leaves a
// needed ring row stale, changes what it generates, while rows outside the window (stale after a restore, as on the
// card) are never read.
//
//   1. the snapshot layout on Muse's real shapes: 81,748,992 B of state at every pos and 13,312 B of KV a position;
//      the zero runs of a snapshot below 2047; the ring's wrap; every run inside its device allocation and disjoint;
//      the host order the same under --pp 1 and --pp 2 at splits 13 and 26 (spec 16b's rule); bad ranges refused;
//   2. save / load on host buffers: the state's ring rows (and the zero rows below position 0) restored bit for bit,
//      everything else untouched; KV ranges likewise; adjacent saves compose;
//   3. through spec 7's request path (server::PrefixSession over the adapter, the real block 2048 and the 2048-key
//      window): restores at block ends 2048 and 4096, at request ends 2049 and 5000 and at 300 (< 2047: zero rows) -
//      every cached turn's ids equal a cold run's; the store's root is Muse's key;
//   4. 16b's rule: a state and KV saved under --pp 2 (split 4) restore under --pp 1 and continue as the two-card session
//      does, and the reverse;
//   5. the server (HTTP, Muse's chat format, its sampling defaults with the softcap): a greedy chat's ids equal the
//      decode run on the response's prompt_token_ids (b70-decode --prefill's flow) - the box stage r46.serve does the
//      same on the card; the same request again restores from the cache at 2048 (cached_tokens); a seeded sampled
//      request is reproducible, draws its FIRST id from the prefill's row too, through the model's cap (the adapter
//      writes it into the Sampling), and only words the row allows.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/muse_serve.h"
#include "model/muse_glimmer.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/prefill_chunks.h"
#include "server/mock.h"
#include "server/openai.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using Ids = std::vector<uint32_t>;
namespace rm = runtime::muse;
using model::MusePlacement;
using rm::SnapRun;
using rm::SnapTensor;

constexpr uint32_t kVocabUsed = 202048;   // the release's tokenizer.json (muse_tokenizer_test)
constexpr float kMult = 0.19611613513818404f, kCap = 20.0f;

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// Muse's pattern (sliding x3, full) twice at a tiny width: 8 layers (full at 3 and 7), one kv head of 8 dims (a 16-byte
// row), the real window (2048) and ring (4096 slots) - everything state_runs / kv_runs read.
model::MuseGlimmerDesc small_desc() {
  model::MuseGlimmerDesc d = model::muse_glimmer();
  d.layers = 8;
  d.full.resize(8);
  d.kv_heads = 1;
  d.head_dim = 8;
  return d;
}

// MuseEngine's interface (what MuseEngineAdapterT calls), device-free.
struct FakeMuse {
  static constexpr uint32_t kBlock = 2048;   // MuseEngine::kBlock (= kPfC)
  using BlockHook = std::function<void(uint32_t, bool)>;

  struct Dev {
    std::vector<uint8_t> full_k, full_v, ring_k, ring_v;
  };

  FakeMuse(const model::MuseGlimmerDesc& desc, MusePlacement placement, uint32_t len, Ids vocab_words)
      : d(desc), p(placement), max_len_(len), words(std::move(vocab_words)) {
    model::validate(p, d);
    for (uint32_t i = 0; i < p.devices; ++i) {
      const rm::PersistentSizes s = rm::persistent_sizes(d, p, i, max_len_);
      devs.push_back({std::vector<uint8_t>(s.full_k), std::vector<uint8_t>(s.full_v), std::vector<uint8_t>(s.ring_k),
                      std::vector<uint8_t>(s.ring_v)});
    }
  }

  // --- the session ----------------------------------------------------------------------------------------------
  void reset() {
    for (Dev& v : devs)
      for (auto* m : {&v.full_k, &v.full_v, &v.ring_k, &v.ring_v}) std::fill(m->begin(), m->end(), uint8_t(0));
    pos_ = 0;
    cur = 0;
    memo_ = -1;
  }
  // MuseEngine::prefill's control flow (muse_prefill_engine.cc): chunks by runtime::prefill_chunk_rows (kPfC 2048),
  // the mid-prompt hook at block ends with pos == end, the prompt-end hook after the first generated id is pending.
  void prefill(const Ids& ids) {
    CHECK(!ids.empty());
    const bool hooked = static_cast<bool>(hook);
    const uint32_t base = pos_;
    uint32_t C = 0;
    for (size_t off = 0; off < ids.size(); off += C) {
      C = runtime::prefill_chunk_rows(base + uint32_t(off), ids.size() - off, rm::kPfC, hooked, kBlock);
      for (uint32_t i = 0; i < C; ++i) write(ids[off + i]);
      const uint32_t end = base + uint32_t(off) + C;
      if (hooked && off + C < ids.size() && end % kBlock == 0) hook(end, true);
    }
    cur = next();
    if (hooked) hook(pos_, pos_ % kBlock == 0);
  }
  void ingest(const Ids& ids) {
    for (uint32_t id : ids) write(id);
    cur = next();
  }
  Ids generate(uint32_t n) {
    Ids out;
    for (uint32_t i = 0; i < n; ++i) {
      out.push_back(cur);
      write(cur);
      cur = next();
    }
    return out;
  }
  uint32_t pos() const { return pos_; }
  uint32_t max_len() const { return max_len_; }
  uint32_t vocab() const { return d.vocab; }
  uint32_t pending() const { return cur; }
  void set_token(uint32_t id) {
    CHECK(id < d.vocab);
    cur = id;
  }
  void set_block_hook(BlockHook h) { hook = std::move(h); }
  // The head's RAW row of the last position: the words scored from the last view (the argmax word highest), everything
  // else -1e30 - next() is this row's argmax, the device's.
  void read_logits_into(float* host) {
    ++logits_reads;
    std::fill(host, host + d.vocab, -1e30f);
    const uint64_t v = attend(pos_ == 0 ? 0 : pos_ - 1, true);
    uint64_t h = v;
    for (uint32_t w : words) {
      h = mix(h, w);
      host[w] = float(h % 1000) / 25.0f;   // < 40: raw logits in the range the cap bends
    }
    host[words[v % words.size()]] = 60.0f;
  }

  // --- MuseEngine's snapshot calls (muse_engine.cc: the same run lists, memcpy for the device) ---------------------
  size_t state_bytes() const { return rm::state_snapshot_bytes(d); }
  size_t kv_bytes(uint32_t n) const { return rm::kv_snapshot_bytes(d, n); }
  std::vector<uint8_t>& mem(uint32_t dev, SnapTensor t) {
    Dev& v = devs.at(dev);
    return t == SnapTensor::FullK ? v.full_k : t == SnapTensor::FullV ? v.full_v : t == SnapTensor::RingK ? v.ring_k : v.ring_v;
  }
  void save_state(void* host) {
    auto* h = static_cast<uint8_t*>(host);
    for (const SnapRun& r : rm::state_runs(d, p, pos_)) {
      std::vector<uint8_t>& m = mem(r.device, r.tensor);
      CHECK(r.offset + r.bytes <= m.size());
      if (r.zero) std::memset(h, 0, r.bytes);
      else std::memcpy(h, m.data() + r.offset, r.bytes);
      h += r.bytes;
    }
  }
  void load_state(const void* host, uint32_t at) {
    CHECK(at <= max_len_);
    const auto* h = static_cast<const uint8_t*>(host);
    for (const SnapRun& r : rm::state_runs(d, p, at)) {
      std::vector<uint8_t>& m = mem(r.device, r.tensor);
      CHECK(r.offset + r.bytes <= m.size());
      std::memcpy(m.data() + r.offset, h, r.bytes);
      h += r.bytes;
    }
    pos_ = at;
    memo_ = -1;
    ++state_loads;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) {
    auto* h = static_cast<uint8_t*>(host);
    for (const SnapRun& r : rm::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(h, mem(r.device, r.tensor).data() + r.offset, r.bytes);
      h += r.bytes;
    }
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) {
    const auto* h = static_cast<const uint8_t*>(host);
    for (const SnapRun& r : rm::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(mem(r.device, r.tensor).data() + r.offset, h, r.bytes);
      h += r.bytes;
    }
    memo_ = -1;
    kv_loaded += e - b;
  }

  // --- the "model" ------------------------------------------------------------------------------------------------
  // Layer l's K (t 0) or V (t 1) row of position q, addressed as MuseBuffers::k_layer does - independent of state_runs
  // / kv_runs, so the layout functions are checked against it.
  uint8_t* row(uint32_t l, uint32_t q, int t) {
    const uint32_t dev = p.device_of(l);
    const size_t rb = size_t(d.kv_n()) * 2;
    if (!d.is_full(l)) {
      const size_t i = d.sliding_before(l) - d.sliding_before(p.first(dev));
      return (t ? devs[dev].ring_v : devs[dev].ring_k).data() + i * size_t(model::MuseGlimmerDesc::kRing) * rb +
             size_t(q & (model::MuseGlimmerDesc::kRing - 1)) * rb;
    }
    const size_t i = d.full_before(l) - d.full_before(p.first(dev));
    return (t ? devs[dev].full_v : devs[dev].full_k).data() + i * size_t(max_len_) * rb + size_t(q) * rb;
  }
  // What the query at position q reads: every sliding layer's rows of (q - 2048, q], every full layer's [0, q] - with
  // `self` false, without its own row (what position q's rows are computed from).
  uint64_t attend(uint32_t q, bool self) {
    const int64_t key = int64_t(q) * 2 + (self ? 1 : 0);
    if (key == memo_) return memo_value_;
    uint64_t h = 0x51ED;
    const uint32_t lo = q >= d.window - 1 ? q - (d.window - 1) : 0;
    const uint32_t end = self ? q + 1 : q;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const uint32_t first = d.is_full(l) ? 0 : lo;
      for (uint32_t j = first; j < end; ++j) {
        uint64_t a, b;
        std::memcpy(&a, row(l, j, 0), 8);
        std::memcpy(&b, row(l, j, 1) + 8, 8);
        h = mix(h, a ^ (b << 1) ^ (uint64_t(l) << 59));
      }
    }
    memo_ = key;
    memo_value_ = h;
    return h;
  }
  void write(uint32_t id) {
    CHECK(pos_ < max_len_);
    const uint64_t prev = pos_ == 0 ? 0x1234 : attend(pos_, false);
    const size_t rb = size_t(d.kv_n()) * 2;
    for (uint32_t l = 0; l < d.layers; ++l)
      for (int t = 0; t < 2; ++t) {
        uint64_t h = mix(mix(mix(prev, id), uint64_t(pos_) * 131 + l), uint64_t(t));
        uint8_t* r = row(l, pos_, t);
        for (size_t i = 0; i < rb; i += 8) {
          h = mix(h, i);
          std::memcpy(r + i, &h, std::min<size_t>(8, rb - i));
        }
      }
    ++pos_;
    memo_ = -1;
  }
  uint32_t next() { return words[attend(pos_ - 1, true) % words.size()]; }

  model::MuseGlimmerDesc d;
  MusePlacement p;
  uint32_t max_len_;
  std::vector<Dev> devs;
  Ids words;
  uint32_t pos_ = 0, cur = 0;
  BlockHook hook;
  int64_t memo_ = -1;
  uint64_t memo_value_ = 0;
  size_t state_loads = 0, kv_loaded = 0, logits_reads = 0;
};

using Adapter = cli::muse::MuseEngineAdapterT<FakeMuse>;

struct MallocAlloc : server::HostAlloc {
  size_t live = 0;
  void* alloc(size_t n) override {
    live += n;
    return std::malloc(n);
  }
  void free(void* q, size_t n) override {
    live -= n;
    std::free(q);
  }
};

Ids some_words() {
  Ids w;
  for (uint32_t i = 0; i < 24; ++i) w.push_back(1000 + 7 * i);
  return w;
}

// ---- 1. the layout on Muse's real shapes ----------------------------------------------------------------------------
void check_disjoint_inside(const model::MuseGlimmerDesc& d, const MusePlacement& p, uint32_t max_len,
                           const std::vector<SnapRun>& runs) {
  for (size_t i = 0; i < runs.size(); ++i) {
    const SnapRun& r = runs[i];
    const rm::PersistentSizes s = rm::persistent_sizes(d, p, r.device, max_len);
    const size_t cap = r.tensor == SnapTensor::FullK ? s.full_k : r.tensor == SnapTensor::FullV ? s.full_v
                                                     : r.tensor == SnapTensor::RingK ? s.ring_k : s.ring_v;
    CHECK(r.bytes > 0);
    CHECK(r.offset + r.bytes <= cap);
    for (size_t q = 0; q < i; ++q)
      if (runs[q].device == r.device && runs[q].tensor == r.tensor)
        CHECK(runs[q].offset + runs[q].bytes <= r.offset || r.offset + r.bytes <= runs[q].offset);
  }
}

// The host order of a run list, placement-free: (tensor, the run's layer-relative offset, bytes, zero).
std::vector<std::vector<size_t>> host_order(const model::MuseGlimmerDesc& d, uint32_t max_len,
                                            const std::vector<SnapRun>& runs) {
  std::vector<std::vector<size_t>> out;
  const size_t row = size_t(d.kv_n()) * 2;
  for (const SnapRun& r : runs) {
    const bool ring = r.tensor == SnapTensor::RingK || r.tensor == SnapTensor::RingV;
    const size_t per = (ring ? size_t(model::MuseGlimmerDesc::kRing) : size_t(max_len)) * row;
    out.push_back({size_t(r.tensor), r.offset % per, r.bytes, size_t(r.zero)});
  }
  return out;
}

void case_layout() {
  const model::MuseGlimmerDesc d = model::muse_glimmer();
  const uint32_t len = 131072;
  CHECK_EQ(rm::state_positions(d), 2047u);
  CHECK_EQ(rm::state_snapshot_bytes(d), size_t(81748992));
  CHECK_EQ(rm::kv_snapshot_bytes(d, 1), size_t(13312));
  CHECK_EQ(rm::kv_snapshot_bytes(d, 2048), size_t(27262976));
  for (uint32_t dev = 0; dev < 2; ++dev) {
    const MusePlacement p2 = MusePlacement::two(d, 26);
    const rm::PersistentSizes s = rm::persistent_sizes(d, p2, dev, len);
    CHECK_EQ(s.full_k + s.full_v, rm::device_kv_bytes(d, p2, dev, len));
    CHECK_EQ(s.ring_k + s.ring_v, rm::device_ring_bytes(d, p2, dev));
  }
  const std::vector<MusePlacement> places = {MusePlacement::one(d), MusePlacement::two(d, 13), MusePlacement::two(d, 26)};
  const size_t row = size_t(d.kv_n()) * 2;
  for (uint32_t pos : {0u, 1u, 300u, 2046u, 2047u, 2048u, 2049u, 4095u, 4096u, 4097u, 5000u, 6143u, 6144u, 131071u,
                       131072u}) {
    std::vector<std::vector<size_t>> first;
    for (const MusePlacement& p : places) {
      const std::vector<SnapRun> runs = rm::state_runs(d, p, pos);
      size_t total = 0, zero = 0;
      for (const SnapRun& r : runs) {
        total += r.bytes;
        zero += r.zero ? r.bytes : 0;
        CHECK(r.tensor == SnapTensor::RingK || r.tensor == SnapTensor::RingV);
      }
      CHECK_EQ(total, rm::state_snapshot_bytes(d));
      CHECK_EQ(zero, size_t(pos < 2047 ? 2047 - pos : 0) * row * 39 * 2);
      // One run a layer and tensor, plus one where the slots wrap (positions crossing a multiple of 4096) or the
      // window reaches below 0.
      const bool split = pos < 2047 ? pos > 0 : (pos - 2047) / 4096 != (pos - 1) / 4096;
      CHECK_EQ(runs.size(), size_t(78) * (split ? 2 : 1));
      check_disjoint_inside(d, p, len, runs);
      if (first.empty()) first = host_order(d, len, runs);
      else CHECK(host_order(d, len, runs) == first);   // 16b's rule: one host layout
    }
  }
  for (const MusePlacement& p : places) {
    CHECK(rm::kv_runs(d, p, len, 100, 100).empty());
    const std::vector<SnapRun> runs = rm::kv_runs(d, p, len, 2048, 4096);
    CHECK_EQ(runs.size(), size_t(26));
    size_t total = 0;
    for (const SnapRun& r : runs) total += r.bytes;
    CHECK_EQ(total, rm::kv_snapshot_bytes(d, 2048));
    check_disjoint_inside(d, p, len, runs);
    CHECK(host_order(d, len, runs) == host_order(d, len, rm::kv_runs(d, places[0], len, 2048, 4096)));
    bool threw = false;
    try {
      (void)rm::kv_runs(d, p, len, 10, len + 1);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  std::printf("layout: state 81748992 B (78 runs, 156 across a wrap or below 0), KV 13312 B a position (26 runs), "
              "inside the allocations, one host layout for --pp 1 and --pp 2 at splits 13 and 26\n");
}

// ---- 2. save / load on host buffers -------------------------------------------------------------------------------
void scramble(FakeMuse& e, uint32_t seed) {
  std::mt19937_64 rng(seed);
  for (auto& v : e.devs)
    for (auto* m : {&v.full_k, &v.full_v, &v.ring_k, &v.ring_v})
      for (auto& b : *m) b = uint8_t(rng());
}

void case_round_trip() {
  const model::MuseGlimmerDesc d = small_desc();
  const uint32_t W = rm::state_positions(d);
  for (const MusePlacement& p : {MusePlacement::one(d), MusePlacement::two(d, 4)}) {
    for (uint32_t at : {300u, 2047u, 4096u, 5000u}) {
      FakeMuse e(d, p, 8192, some_words());
      scramble(e, at);
      e.pos_ = at;
      std::vector<uint8_t> state(e.state_bytes());
      e.save_state(state.data());
      // positions before 0 are zero on the host
      const size_t row = size_t(d.kv_n()) * 2, below = at < W ? W - at : 0;
      for (size_t i = 0; i < below * row; ++i) CHECK_EQ(state[i], uint8_t(0));
      std::vector<std::vector<uint8_t>> r0;
      for (uint32_t l = 0; l < d.layers; ++l)
        for (uint32_t q = (at >= W ? at - W : 0); q < at; ++q)
          for (int t = 0; t < 2; ++t) r0.emplace_back(e.row(l, q, t), e.row(l, q, t) + row);
      FakeMuse f(d, p, 8192, some_words());
      scramble(f, at + 1);
      const FakeMuse before = f;
      f.load_state(state.data(), at);
      CHECK_EQ(f.pos(), at);
      size_t k = 0;
      for (uint32_t l = 0; l < d.layers; ++l)
        for (uint32_t q = (at >= W ? at - W : 0); q < at; ++q)
          for (int t = 0; t < 2; ++t) {
            if (!d.is_full(l)) CHECK(std::equal(r0[k].begin(), r0[k].end(), f.row(l, q, t)));
            ++k;
          }
      // the full layers and every ring slot outside [at - 2047, at) and the zero slots are untouched
      for (uint32_t dv = 0; dv < p.devices; ++dv) {
        CHECK(f.devs[dv].full_k == before.devs[dv].full_k && f.devs[dv].full_v == before.devs[dv].full_v);
        for (int t = 0; t < 2; ++t) {
          const std::vector<uint8_t>& now = t ? f.devs[dv].ring_v : f.devs[dv].ring_k;
          const std::vector<uint8_t>& was = t ? before.devs[dv].ring_v : before.devs[dv].ring_k;
          std::vector<bool> in(now.size(), false);
          for (const SnapRun& r : rm::state_runs(d, p, at))
            if (r.device == dv && (r.tensor == SnapTensor::RingV) == bool(t))
              for (size_t i = 0; i < r.bytes; ++i) in[r.offset + i] = true;
          for (size_t i = 0; i < now.size(); ++i)
            if (!in[i]) CHECK_EQ(now[i], was[i]);
        }
      }
      // KV: [17, at) restored bit for bit, the rest untouched; adjacent saves compose.
      std::vector<uint8_t> kv(e.kv_bytes(at - 17)), a(e.kv_bytes(100)), b(e.kv_bytes(at - 117));
      e.save_kv(17, at, kv.data());
      e.save_kv(17, 117, a.data());
      e.save_kv(117, at, b.data());
      FakeMuse x(d, p, 8192, some_words()), y(d, p, 8192, some_words());
      x.load_kv(17, 117, a.data());
      x.load_kv(117, at, b.data());
      y.load_kv(17, at, kv.data());
      for (uint32_t dv = 0; dv < p.devices; ++dv)
        CHECK(x.devs[dv].full_k == y.devs[dv].full_k && x.devs[dv].full_v == y.devs[dv].full_v);
      for (uint32_t l = 0; l < d.layers; ++l)
        if (d.is_full(l))
          for (uint32_t q = 17; q < at; ++q)
            for (int t = 0; t < 2; ++t) CHECK(std::equal(e.row(l, q, t), e.row(l, q, t) + row, y.row(l, q, t)));
      e.save_kv(5, 5, nullptr);   // empty: no copy
    }
  }
  std::printf("round trip: the ring rows of [pos - 2047, pos) (zeros below 0) and KV ranges restored bit for bit, "
              "nothing else touched, on one card and two\n");
}

// ---- 3. through spec 7's request path -----------------------------------------------------------------------------
Ids turn(server::PrefixSession& s, Adapter& ad, const Ids& prompt, uint32_t n, server::PrefixSession::Report* rep = nullptr) {
  const auto r = s.begin(prompt);
  if (rep) *rep = r;
  Ids out;
  server::Sampling greedy;
  for (uint32_t i = 0; i < n; ++i) {
    out.push_back(ad.step(greedy));
    s.fed(out.back());
  }
  s.end();
  return out;
}

Ids cold(const model::MuseGlimmerDesc& d, const MusePlacement& p, const Ids& prompt, uint32_t n) {
  FakeMuse e(d, p, 8192, some_words());
  e.prefill(prompt);
  return e.generate(n);
}

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

void case_prefix() {
  const model::MuseGlimmerDesc d = small_desc();
  Ids L;
  for (uint32_t i = 0; i < 6000; ++i) L.push_back(2000 + (i * 7919) % 50000);
  const Ids C(L.rbegin(), L.rbegin() + 60);
  using K = server::PrefixCache::Plan;
  struct Want {
    const char* name;
    uint32_t restart;
    uint32_t first;   // turn 1's prompt length (it generates 8)
  };
  MallocAlloc alloc;
  for (const Want& w : {Want{"block end 2048", 2048, 2100}, Want{"block end 4096", 4096, 4200},
                        Want{"request end 2049", 2049, 2041}, Want{"request end 5000", 5000, 4992},
                        Want{"request end 300 (< 2047)", 300, 292}}) {
    for (const MusePlacement& p : {MusePlacement::one(d), MusePlacement::two(d, 4)}) {
      FakeMuse eng(d, p, 8192, some_words());
      Adapter ad(eng, kVocabUsed, kMult, kCap);
      CHECK_EQ(ad.block(), 2048u);
      CHECK_EQ(ad.state_bytes(), rm::state_snapshot_bytes(d));
      server::PrefixSession s(ad, size_t(1) << 30, &alloc);
      CHECK_EQ(s.cache()->kv_form(), cli::muse::kMuseStoreKey);
      const Ids t1 = slice(L, 0, w.first);
      const Ids g1 = turn(s, ad, t1, 8);
      (void)turn(s, ad, slice(L, 5900, 5960), 4);   // a side request: the next turn must restore
      Ids final_prompt;
      if (w.restart % 2048 == 0) final_prompt = cat(slice(L, 0, w.restart + 7), C);   // diverges past the block end
      else final_prompt = cat(cat(t1, g1), slice(C, 0, 30));                         // continues the request end
      server::PrefixSession::Report rep;
      const Ids got = turn(s, ad, final_prompt, 16, &rep);
      if (rep.kind != K::Restore || rep.restart != w.restart)
        std::fprintf(stderr, "%s: plan %s at %u, expected restore at %u\n", w.name, server::plan_kind_name(rep.kind),
                     rep.restart, w.restart);
      CHECK(rep.kind == K::Restore);
      CHECK_EQ(rep.restart, w.restart);
      CHECK(eng.state_loads > 0);
      const Ids want = cold(d, MusePlacement::one(d), final_prompt, 16);
      CHECK(got == want);   // a restore across the window continues as the cold run
      std::printf("  %s, %s: restore at %u (%zu B of KV), 16 ids == cold\n", w.name, p.devices == 2 ? "--pp 2" : "--pp 1",
                  rep.restart, rep.kv_bytes);
    }
  }
  CHECK_EQ(alloc.live, size_t(0));
}

// ---- 4. 16b's rule: one host layout for one card and two -----------------------------------------------------------
void case_pp() {
  const model::MuseGlimmerDesc d = small_desc();
  Ids prompt;
  for (uint32_t i = 0; i < 2600; ++i) prompt.push_back(3000 + (i * 104729) % 40000);
  for (int dir = 0; dir < 2; ++dir) {
    const MusePlacement from = dir == 0 ? MusePlacement::two(d, 4) : MusePlacement::one(d);
    const MusePlacement to = dir == 0 ? MusePlacement::one(d) : MusePlacement::two(d, 4);
    FakeMuse a(d, from, 8192, some_words());
    a.prefill(prompt);
    (void)a.generate(5);
    const uint32_t at = a.pos();
    std::vector<uint8_t> state(a.state_bytes()), kv(a.kv_bytes(at));
    a.save_state(state.data());
    a.save_kv(0, at, kv.data());
    FakeMuse b(d, to, 8192, some_words());
    CHECK_EQ(b.state_bytes(), a.state_bytes());
    CHECK_EQ(b.kv_bytes(at), a.kv_bytes(at));
    b.load_kv(0, at, kv.data());
    b.load_state(state.data(), at);
    b.set_token(a.pending());
    CHECK(b.generate(24) == a.generate(24));
  }
  std::printf("pp: a snapshot taken on two cards (split 4) continues on one bitwise, and the reverse\n");
}

// ---- 5. the server -------------------------------------------------------------------------------------------------
struct WordTemplate : server::TemplateIface {
  std::string render(const json& messages, const json&, bool) override { return body(messages); }
  std::string render_with_kwargs(const json& messages, const json&, bool, const json&) override { return body(messages); }
  static std::string body(const json& messages) {
    std::string text = "<|begin_of_text|> ";
    for (const auto& m : messages)
      text += "<|start|> " + m.at("role").get<std::string>() + " " + m.at("content").get<std::string>() + " <|eot|> ";
    return text + "<|start|> assistant ";
  }
};

server::Sampling muse_defaults() {
  server::Sampling s = server::generation_sampling({{"do_sample", true}, {"temperature", 1.0}, {"top_p", 0.95},
                                                    {"top_k", 64}});
  s.logit_mult = kMult;
  s.softcap = kCap;
  return s;
}

struct ServeFixture {
  explicit ServeFixture(bool cache)
      : eng(small_desc(), MusePlacement::two(small_desc(), 4), 8192, words()),
        adapter(eng, kVocabUsed, kMult, kCap),
        server({tok, tmpl, adapter}, options(cache)) {
    CHECK(server.start());
  }
  ~ServeFixture() { server.stop(); }
  Ids words() {
    Ids w;
    for (uint32_t i = 0; i < 24; ++i) w.push_back(tok.id_of("w" + std::to_string(i)));
    return w;
  }
  server::Options options(bool cache) {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "muse";
    o.eos_ids = {200001, 200008};
    o.chat_format = server::ChatFormat::muse();
    o.sampling_defaults = muse_defaults();
    if (cache) {
      o.prefix_cache_bytes = size_t(512) << 20;
      o.prefix_alloc = &alloc;
    }
    return o;
  }
  json chat(const json& extra) {
    std::string long_text;
    for (int i = 0; i < 2200; ++i) long_text += "word" + std::to_string(i % 97) + " ";
    json b = {{"model", "muse"},
              {"messages", json::array({{{"role", "user"}, {"content", long_text + "explain the cache please"}}})},
              {"max_tokens", 24},
              {"return_token_ids", true}};
    for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
    httplib::Client c("127.0.0.1", server.bound_port());
    const auto r = c.Post("/v1/chat/completions", b.dump(), "application/json");
    CHECK(r);
    if (r->status != 200) std::fprintf(stderr, "status %d: %s\n", r->status, r->body.c_str());
    CHECK_EQ(r->status, 200);
    return json::parse(r->body);
  }

  MockTok tok;   // declared first: words() needs it
  WordTemplate tmpl;
  MallocAlloc alloc;
  FakeMuse eng;
  Adapter adapter;
  server::Server server;
};

void case_server() {
  ServeFixture f(/*cache=*/true);
  const json greedy = {{"temperature", 0}};
  const json r1 = f.chat(greedy);
  const Ids prompt = r1.at("prompt_token_ids").get<Ids>();
  const Ids served = r1.at("choices").at(0).at("token_ids").get<Ids>();
  CHECK(prompt.size() > 2048);
  CHECK_EQ(served.size(), size_t(24));
  // b70-decode --prefill's run of the same ids
  FakeMuse decode(small_desc(), MusePlacement::one(small_desc()), 8192, f.eng.words);
  decode.prefill(prompt);
  CHECK(decode.generate(24) == served);
  // The same request again: restored from the block-end snapshot (rings + KV), the same ids.
  const json r2 = f.chat(greedy);
  CHECK(r2.at("prompt_token_ids").get<Ids>() == prompt);
  CHECK(r2.at("choices").at(0).at("token_ids").get<Ids>() == served);
  const uint32_t cached = r2.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
  CHECK_EQ(cached, 2048u);   // the block-end snapshot: the prompt is past 2048, its own end cannot restore
  CHECK(f.eng.state_loads > 0);
  std::printf("server: greedy chat of %zu prompt ids -> 24 ids == the decode run; again: restored at %u, same ids\n",
              prompt.size(), cached);

  // Sampled with the model's defaults (no temperature in the request), seeded: reproducible on a fresh server, the
  // first id drawn from the prefill's row too (n + 1 rows read for n ids), only allowed words; the adapter draws
  // through the CAPPED distribution: its draw equals filter_probs with the model's cap on the same row.
  const json sampled = {{"seed", 1234}};
  Ids a, b;
  size_t reads = 0;
  {
    ServeFixture g(/*cache=*/false);
    a = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
    reads = g.eng.logits_reads;
    CHECK_EQ(g.adapter.sampled(), uint64_t(a.size() + 1));
  }
  {
    ServeFixture g(/*cache=*/false);
    b = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
  }
  CHECK(a == b);
  CHECK_EQ(reads, a.size() + 1);
  // A seed is honoured per request: the same seeded request twice on ONE server draws the same ids (the RNG is
  // reseeded from the request's seed at its first sampled step - not only once per process).
  {
    ServeFixture g(/*cache=*/false);
    const Ids first = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
    const Ids second = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
    CHECK(first == a);
    CHECK(second == first);
  }
  for (uint32_t id : a) CHECK(std::find(f.eng.words.begin(), f.eng.words.end(), id) != f.eng.words.end());
  // The cap applies even when a request's Sampling lacks it (the adapter writes the model's constants): a sampled
  // step's draw on a fixed row is filter_probs(row, cap)'s draw, not the raw row's.
  {
    FakeMuse e(small_desc(), MusePlacement::one(small_desc()), 8192, f.eng.words);
    Adapter ad(e, kVocabUsed, kMult, kCap);
    e.prefill(slice(prompt, 0, 100));
    server::Sampling bare;   // softcap 0 in the request's Sampling
    bare.greedy = false;
    bare.top_k = 0;
    bare.top_p = 1.0f;
    bare.seed = 77;
    bare.has_seed = true;
    std::vector<float> row(e.vocab());
    e.read_logits_into(row.data());
    server::Sampling capped = bare;
    capped.logit_mult = kMult;
    capped.softcap = kCap;
    std::mt19937_64 rng(77);
    const uint32_t want = server::draw(server::filter_probs(row.data(), kVocabUsed, capped), rng);
    ad.sample_into_pending(bare);
    CHECK_EQ(e.pending(), want);
  }
  std::printf("server: a seeded request with Muse's default sampling reproducible (%zu ids, %zu rows read, each through "
              "the cap), every id a vocabulary word\n", a.size(), reads);
}

}  // namespace

int main() {
  case_layout();
  case_round_trip();
  case_prefix();
  case_pp();
  case_server();
  std::printf("muse_snapshot_test OK\n");
  return 0;
}

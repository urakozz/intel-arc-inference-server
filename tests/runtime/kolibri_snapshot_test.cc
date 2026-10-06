// Spec 20e (plan 20e Task 3, host side): Kolibri-1 behind b70-serve without a device.
// cli::kolibri::KolibriEngineAdapterT - the adapter b70-serve builds over runtime::kolibri::KolibriEngine -
// runs here over FakeKolibri, an engine with KolibriEngine's interface whose rings and full-layer KV are host
// buffers in KolibriBuffers' real layout (runtime::kolibri::persistent_sizes per device of a placement: one
// card or two) and whose snapshot calls are KolibriEngine's own loops over runtime::kolibri::state_runs /
// kv_runs, memcpy for the device copies. Its "model" reads what Kolibri's attention reads: the query at
// position q sees the sliding layers' ring rows of [q - 512, q] (through the slots q & 4095) and the full
// layers' rows [0, q]; position q's K / V rows of every layer are a function of the id, q, the layer and the
// keys the query at q attends to before its own (the window's [q - 512, q - 1], every earlier full row),
// and the next id a function of what the last query attends to (its own key included) - so a restore that
// misplaces one row of one ring or one full layer, or leaves a needed ring row stale, changes what it
// generates, while rows outside the window (stale after a restore, as on the card) are never read.
//
//   1. the snapshot layout on Kolibri's real shapes: 41,943,040 B of state at every pos and 20,480 B of KV a
//      position; the zero runs of a snapshot below 512; the ring's wrap; every run inside its device
//      allocation and disjoint; the host order the same under --pp 1 and --pp 2 at splits 3 and 25 (spec
//      16b's rule); bad ranges refused;
//   2. save / load on host buffers: the state's ring rows (and the zero rows below position 0) restored bit
//      for bit, everything else untouched; KV ranges likewise; adjacent saves compose;
//   3. Review Focus 4 through spec 7's request path (server::PrefixSession over the adapter, the real block
//      2048 and the 512-position window): restores at block ends 2048 and 4096, at request ends 2049 and
//      5000 and at 300 (< 512: zero rows) - every cached turn's ids equal a cold run's; the store's root is
//      Kolibri's key;
//   4. 16b's rule: a state and KV saved under --pp 2 (split 3) restore under --pp 1 and continue as the
//      two-card session does, and the reverse;
//   5. the server (HTTP, Kolibri's chat format): a greedy chat's ids equal the decode run on the response's
//      prompt_token_ids (b70-decode --prefill's flow) - the box stage r28.serve does the same on the card;
//      the same request again restores from the cache; a seeded sampled request is reproducible, draws its
//      FIRST id from the prefill's row too, and only words the row allows; a greedy argmax on a head row
//      the tokenizer does not define (127998 / 127999) is replaced by the masked row's argmax (Review
//      Focus 3).
#include <algorithm>
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
#include "cli/kolibri_serve.h"
#include "model/kolibri1.h"
#include "runtime/kolibri/kolibri_sizes.h"
#include "runtime/prefill_chunks.h"
#include "server/mock.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using Ids = std::vector<uint32_t>;
namespace rk = runtime::kolibri;
using model::KolPlacement;
using rk::SnapRun;
using rk::SnapTensor;

constexpr uint32_t kVocabUsed = 127998;   // the release's tokenizer.json (kolibri_tokenizer_test)

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// Kolibri's pattern (sliding x4, full) at a tiny width: 5 layers, one kv head of 8 dims (a 16-byte row),
// the real window (513) and ring (4096 slots) - everything state_runs / kv_runs read.
model::Kolibri1Desc small_desc() {
  model::Kolibri1Desc d = model::kolibri1();
  d.layers = 5;
  d.kv_heads = 1;
  d.head_dim = 8;
  return d;
}

// KolibriEngine's interface (what KolibriEngineAdapterT calls), device-free.
struct FakeKolibri {
  static constexpr uint32_t kBlock = 2048;   // KolibriEngine::kBlock (= kPfC)
  using BlockHook = std::function<void(uint32_t, bool)>;

  struct Dev {
    std::vector<uint8_t> full_k, full_v, ring_k, ring_v;
  };

  FakeKolibri(const model::Kolibri1Desc& desc, KolPlacement placement, uint32_t len, Ids vocab_words)
      : d(desc), p(placement), max_len_(len), words(std::move(vocab_words)) {
    model::validate(p, d);
    for (uint32_t i = 0; i < p.devices; ++i) {
      const rk::PersistentSizes s = rk::persistent_sizes(d, p, i, max_len_);
      devs.push_back({std::vector<uint8_t>(s.full_k), std::vector<uint8_t>(s.full_v),
                      std::vector<uint8_t>(s.ring_k), std::vector<uint8_t>(s.ring_v)});
    }
  }

  // --- the session ----------------------------------------------------------------------------------
  void reset() {
    for (Dev& v : devs)
      for (auto* m : {&v.full_k, &v.full_v, &v.ring_k, &v.ring_v}) std::fill(m->begin(), m->end(), uint8_t(0));
    pos_ = 0;
    cur = 0;
    memo_ = -1;
  }
  // KolibriEngine::prefill's control flow (kolibri_prefill_engine.cc): chunks by runtime::prefill_chunk_rows
  // (kPfC 2048), the mid-prompt hook at block ends with pos == end, the prompt-end hook after the first
  // generated id is pending.
  void prefill(const Ids& ids) {
    CHECK(!ids.empty());
    const bool hooked = static_cast<bool>(hook);
    const uint32_t base = pos_;
    uint32_t C = 0;
    for (size_t off = 0; off < ids.size(); off += C) {
      C = runtime::prefill_chunk_rows(base + uint32_t(off), ids.size() - off, rk::kPfC, hooked, kBlock);
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
  // The head's row of the last position: the words scored from the last view (the argmax word highest),
  // everything else -1e30; with `poison` the two rows without a token score highest, as an untrained row
  // might - next() is this row's argmax, the device's.
  void read_logits_into(float* host) {
    ++logits_reads;
    std::fill(host, host + d.vocab, -1e30f);
    const uint64_t v = attend(pos_ == 0 ? 0 : pos_ - 1, true);
    uint64_t h = v;
    for (uint32_t w : words) {
      h = mix(h, w);
      host[w] = float(h % 1000) / 250.0f;   // < 4
    }
    host[words[v % words.size()]] = 10.0f;
    if (poison) host[127998] = host[127999] = 100.0f;
  }

  // --- KolibriEngine's snapshot calls (kolibri_engine.cc: the same run lists, memcpy for the device) ----
  size_t state_bytes() const { return rk::state_snapshot_bytes(d); }
  size_t kv_bytes(uint32_t n) const { return rk::kv_snapshot_bytes(d, n); }
  std::vector<uint8_t>& mem(uint32_t dev, SnapTensor t) {
    Dev& v = devs.at(dev);
    return t == SnapTensor::FullK ? v.full_k : t == SnapTensor::FullV ? v.full_v : t == SnapTensor::RingK ? v.ring_k : v.ring_v;
  }
  void save_state(void* host) {
    auto* h = static_cast<uint8_t*>(host);
    for (const SnapRun& r : rk::state_runs(d, p, pos_)) {
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
    for (const SnapRun& r : rk::state_runs(d, p, at)) {
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
    for (const SnapRun& r : rk::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(h, mem(r.device, r.tensor).data() + r.offset, r.bytes);
      h += r.bytes;
    }
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) {
    const auto* h = static_cast<const uint8_t*>(host);
    for (const SnapRun& r : rk::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(mem(r.device, r.tensor).data() + r.offset, h, r.bytes);
      h += r.bytes;
    }
    memo_ = -1;
    kv_loaded += e - b;
  }

  // --- the "model" --------------------------------------------------------------------------------------
  // Layer l's K (t 0) or V (t 1) row of position q, addressed as KolibriBuffers::k_layer does - independent
  // of state_runs / kv_runs, so the layout functions are checked against it.
  uint8_t* row(uint32_t l, uint32_t q, int t) {
    const uint32_t dev = p.device_of(l);
    const size_t rb = size_t(d.kv_n()) * 2;
    if (d.is_sliding(l)) {
      const size_t i = d.sliding_before(l) - d.sliding_before(p.first(dev));
      return (t ? devs[dev].ring_v : devs[dev].ring_k).data() + i * rk::ring_layer_rows_bytes(d) +
             size_t(q & (model::Kolibri1Desc::kRing - 1)) * rb;
    }
    const size_t i = d.full_before(l) - d.full_before(p.first(dev));
    return (t ? devs[dev].full_v : devs[dev].full_k).data() + i * rk::full_layer_rows_bytes(d, max_len_) + size_t(q) * rb;
  }
  // What the query at position q reads: every sliding layer's rows of (q - 513, q], every full layer's
  // [0, q] - with `self` false, without its own row (what position q's rows are computed from).
  uint64_t attend(uint32_t q, bool self) {
    const int64_t key = int64_t(q) * 2 + (self ? 1 : 0);
    if (key == memo_) return memo_value_;
    uint64_t h = 0x51ED;
    const uint32_t lo = q >= d.window - 1 ? q - (d.window - 1) : 0;
    const uint32_t end = self ? q + 1 : q;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const uint32_t first = d.is_sliding(l) ? lo : 0;
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
  uint32_t next() { return poison ? 127999u : words[attend(pos_ - 1, true) % words.size()]; }

  model::Kolibri1Desc d;
  KolPlacement p;
  uint32_t max_len_;
  std::vector<Dev> devs;
  Ids words;
  uint32_t pos_ = 0, cur = 0;
  BlockHook hook;
  bool poison = false;
  int64_t memo_ = -1;
  uint64_t memo_value_ = 0;
  size_t state_loads = 0, kv_loaded = 0, logits_reads = 0;
};

using Adapter = cli::kolibri::KolibriEngineAdapterT<FakeKolibri>;

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

// ---- 1. the layout on Kolibri's real shapes ---------------------------------------------------------------
void check_disjoint_inside(const model::Kolibri1Desc& d, const KolPlacement& p, uint32_t max_len,
                           const std::vector<SnapRun>& runs) {
  for (size_t i = 0; i < runs.size(); ++i) {
    const SnapRun& r = runs[i];
    const rk::PersistentSizes s = rk::persistent_sizes(d, p, r.device, max_len);
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
std::vector<std::vector<size_t>> host_order(const model::Kolibri1Desc& d, uint32_t max_len, const std::vector<SnapRun>& runs) {
  std::vector<std::vector<size_t>> out;
  for (const SnapRun& r : runs) {
    const bool ring = r.tensor == SnapTensor::RingK || r.tensor == SnapTensor::RingV;
    const size_t per = ring ? rk::ring_layer_rows_bytes(d) : rk::full_layer_rows_bytes(d, max_len);
    out.push_back({size_t(r.tensor), r.offset % per, r.bytes, size_t(r.zero)});
  }
  return out;
}

void case_layout() {
  const model::Kolibri1Desc& d = model::kolibri1();
  const uint32_t len = 262144;
  CHECK_EQ(rk::state_positions(d), 512u);
  CHECK_EQ(rk::state_snapshot_bytes(d), size_t(41943040));
  CHECK_EQ(rk::kv_snapshot_bytes(d, 1), size_t(20480));
  CHECK_EQ(rk::kv_snapshot_bytes(d, 1), rk::full_kv_bytes_per_pos(d));
  const std::vector<KolPlacement> places = {KolPlacement::one(d), KolPlacement::two(d, 3), KolPlacement::two(d, 25)};
  const size_t row = size_t(d.kv_n()) * 2;
  for (uint32_t pos : {0u, 1u, 300u, 511u, 512u, 513u, 2048u, 2049u, 4095u, 4096u, 4097u, 4600u, 5000u, 262143u, 262144u}) {
    std::vector<std::vector<size_t>> first;
    for (const KolPlacement& p : places) {
      const std::vector<SnapRun> runs = rk::state_runs(d, p, pos);
      size_t total = 0, zero = 0;
      for (const SnapRun& r : runs) {
        total += r.bytes;
        zero += r.zero ? r.bytes : 0;
        CHECK(r.tensor == SnapTensor::RingK || r.tensor == SnapTensor::RingV);
      }
      CHECK_EQ(total, rk::state_snapshot_bytes(d));
      CHECK_EQ(zero, size_t(pos < 512 ? 512 - pos : 0) * row * 40 * 2);
      // One run a layer and tensor, plus one where the slots wrap (positions crossing a multiple of 4096) or
      // the window reaches below 0.
      const bool split = pos < 512 ? pos > 0 : (pos - 512) / 4096 != (pos - 1) / 4096;
      CHECK_EQ(runs.size(), size_t(80) * (split ? 2 : 1));
      check_disjoint_inside(d, p, len, runs);
      if (first.empty()) first = host_order(d, len, runs);
      else CHECK(host_order(d, len, runs) == first);   // 16b's rule: one host layout
    }
  }
  for (const KolPlacement& p : places) {
    CHECK(rk::kv_runs(d, p, len, 100, 100).empty());
    const std::vector<SnapRun> runs = rk::kv_runs(d, p, len, 2048, 4096);
    CHECK_EQ(runs.size(), size_t(20));
    size_t total = 0;
    for (const SnapRun& r : runs) total += r.bytes;
    CHECK_EQ(total, rk::kv_snapshot_bytes(d, 2048));
    check_disjoint_inside(d, p, len, runs);
    CHECK(host_order(d, len, runs) == host_order(d, len, rk::kv_runs(d, places[0], len, 2048, 4096)));
    bool threw = false;
    try {
      (void)rk::kv_runs(d, p, len, 10, len + 1);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  std::printf("layout: state 41943040 B (80 runs, 160 across a wrap or below 0), KV 20480 B a position (20 runs), "
              "inside the allocations, one host layout for --pp 1 and --pp 2 at splits 3 and 25\n");
}

// ---- 2. save / load on host buffers --------------------------------------------------------------------------
void scramble(FakeKolibri& e, uint32_t seed) {
  std::mt19937_64 rng(seed);
  for (auto& v : e.devs)
    for (auto* m : {&v.full_k, &v.full_v, &v.ring_k, &v.ring_v})
      for (auto& b : *m) b = uint8_t(rng());
}

void case_round_trip() {
  const model::Kolibri1Desc d = small_desc();
  for (const KolPlacement& p : {KolPlacement::one(d), KolPlacement::two(d, 3)}) {
    for (uint32_t at : {300u, 512u, 4096u, 4600u}) {
      FakeKolibri e(d, p, 8192, some_words());
      scramble(e, at);
      e.pos_ = at;
      std::vector<uint8_t> state(e.state_bytes());
      e.save_state(state.data());
      // positions before 0 are zero on the host
      const size_t row = size_t(d.kv_n()) * 2, below = at < 512 ? 512 - at : 0;
      for (size_t i = 0; i < below * row; ++i) CHECK_EQ(state[i], uint8_t(0));
      std::vector<std::vector<uint8_t>> rk0;
      for (uint32_t l = 0; l < d.layers; ++l)
        for (uint32_t q = (at >= 512 ? at - 512 : 0); q < at; ++q)
          for (int t = 0; t < 2; ++t) rk0.emplace_back(e.row(l, q, t), e.row(l, q, t) + row);
      FakeKolibri f(d, p, 8192, some_words());
      scramble(f, at + 1);
      const FakeKolibri before = f;
      f.load_state(state.data(), at);
      CHECK_EQ(f.pos(), at);
      size_t k = 0;
      for (uint32_t l = 0; l < d.layers; ++l)
        for (uint32_t q = (at >= 512 ? at - 512 : 0); q < at; ++q)
          for (int t = 0; t < 2; ++t) {
            if (d.is_sliding(l)) CHECK(std::equal(rk0[k].begin(), rk0[k].end(), f.row(l, q, t)));
            ++k;
          }
      // the full layers and every ring slot outside [at - 512, at) and the zero slots are untouched
      for (uint32_t dv = 0; dv < p.devices; ++dv) {
        CHECK(f.devs[dv].full_k == before.devs[dv].full_k && f.devs[dv].full_v == before.devs[dv].full_v);
        for (int t = 0; t < 2; ++t) {
          const std::vector<uint8_t>& now = t ? f.devs[dv].ring_v : f.devs[dv].ring_k;
          const std::vector<uint8_t>& was = t ? before.devs[dv].ring_v : before.devs[dv].ring_k;
          std::vector<bool> in(now.size(), false);
          for (const SnapRun& r : rk::state_runs(d, p, at))
            if (r.device == dv && (r.tensor == SnapTensor::RingV) == bool(t))
              for (size_t i = 0; i < r.bytes; ++i) in[r.offset + i] = true;
          for (size_t i = 0; i < now.size(); ++i) {
            if (!in[i]) CHECK_EQ(now[i], was[i]);
          }
        }
      }
      // KV: [17, at) restored bit for bit, the rest untouched; adjacent saves compose.
      std::vector<uint8_t> kv(e.kv_bytes(at - 17)), a(e.kv_bytes(100)), b(e.kv_bytes(at - 117));
      e.save_kv(17, at, kv.data());
      e.save_kv(17, 117, a.data());
      e.save_kv(117, at, b.data());
      FakeKolibri x(d, p, 8192, some_words()), y(d, p, 8192, some_words());
      x.load_kv(17, 117, a.data());
      x.load_kv(117, at, b.data());
      y.load_kv(17, at, kv.data());
      for (uint32_t dv = 0; dv < p.devices; ++dv) CHECK(x.devs[dv].full_k == y.devs[dv].full_k && x.devs[dv].full_v == y.devs[dv].full_v);
      for (uint32_t l = 0; l < d.layers; ++l)
        if (!d.is_sliding(l))
          for (uint32_t q = 17; q < at; ++q)
            for (int t = 0; t < 2; ++t) CHECK(std::equal(e.row(l, q, t), e.row(l, q, t) + row, y.row(l, q, t)));
      e.save_kv(5, 5, nullptr);   // empty: no copy
    }
  }
  std::printf("round trip: the ring rows of [pos - 512, pos) (zeros below 0) and KV ranges restored bit for bit, "
              "nothing else touched, on one card and two\n");
}

// ---- 3. Review Focus 4 through spec 7's request path ---------------------------------------------------------
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

Ids cold(const model::Kolibri1Desc& d, const KolPlacement& p, const Ids& prompt, uint32_t n) {
  FakeKolibri e(d, p, 8192, some_words());
  e.prefill(prompt);
  return e.generate(n);
}

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

void case_prefix() {
  const model::Kolibri1Desc d = small_desc();
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
                        Want{"request end 300 (< 512)", 300, 292}}) {
    for (const KolPlacement& p : {KolPlacement::one(d), KolPlacement::two(d, 3)}) {
      FakeKolibri eng(d, p, 8192, some_words());
      Adapter ad(eng, kVocabUsed);
      CHECK_EQ(ad.block(), 2048u);
      CHECK_EQ(ad.state_bytes(), rk::state_snapshot_bytes(d));
      server::PrefixSession s(ad, size_t(1) << 30, &alloc);
      CHECK_EQ(s.cache()->kv_form(), cli::kolibri::kKolibriStoreKey);
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
      const Ids want = cold(d, KolPlacement::one(d), final_prompt, 16);
      CHECK(got == want);   // a restore across the window continues as the cold run
      std::printf("  %s, %s: restore at %u (%zu B of KV), 16 ids == cold\n", w.name, p.devices == 2 ? "--pp 2" : "--pp 1",
                  rep.restart, rep.kv_bytes);
    }
  }
  CHECK_EQ(alloc.live, size_t(0));
}

// ---- 4. 16b's rule: one host layout for one card and two --------------------------------------------------
void case_pp() {
  const model::Kolibri1Desc d = small_desc();
  Ids prompt;
  for (uint32_t i = 0; i < 2600; ++i) prompt.push_back(3000 + (i * 104729) % 40000);
  for (int dir = 0; dir < 2; ++dir) {
    const KolPlacement from = dir == 0 ? KolPlacement::two(d, 3) : KolPlacement::one(d);
    const KolPlacement to = dir == 0 ? KolPlacement::one(d) : KolPlacement::two(d, 3);
    FakeKolibri a(d, from, 8192, some_words());
    a.prefill(prompt);
    (void)a.generate(5);
    const uint32_t at = a.pos();
    std::vector<uint8_t> state(a.state_bytes()), kv(a.kv_bytes(at));
    a.save_state(state.data());
    a.save_kv(0, at, kv.data());
    FakeKolibri b(d, to, 8192, some_words());
    CHECK_EQ(b.state_bytes(), a.state_bytes());
    CHECK_EQ(b.kv_bytes(at), a.kv_bytes(at));
    b.load_kv(0, at, kv.data());
    b.load_state(state.data(), at);
    b.set_token(a.pending());
    CHECK(b.generate(24) == a.generate(24));
  }
  std::printf("pp: a snapshot taken on two cards (split 3) continues on one bitwise, and the reverse\n");
}

// ---- 5. the server -------------------------------------------------------------------------------------------
struct KolibriTemplate : server::TemplateIface {
  std::string render(const json& messages, const json&, bool) override { return body(messages); }
  std::string render_with_kwargs(const json& messages, const json&, bool, const json&) override { return body(messages); }
  static std::string body(const json& messages) {
    std::string text;
    for (const auto& m : messages)
      text += "<|im_start|> " + m.at("role").get<std::string>() + " " + m.at("content").get<std::string>() + " <|im_end|> ";
    return text + "<|im_start|> assistant ";
  }
};

struct ServeFixture {
  explicit ServeFixture(bool cache)
      : eng(small_desc(), KolPlacement::two(small_desc(), 3), 8192, words()),
        adapter(eng, kVocabUsed),
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
    o.served_model = "kolibri";
    o.eos_ids = {127906, 127901};
    o.chat_format = server::ChatFormat::kolibri();
    o.sampling_defaults = server::generation_sampling({{"do_sample", true}, {"temperature", 1.0}, {"top_p", 0.97},
                                                       {"top_k", 128}});
    if (cache) {
      o.prefix_cache_bytes = size_t(256) << 20;
      o.prefix_alloc = &alloc;
    }
    return o;
  }
  json chat(const json& extra) {
    std::string long_text;
    for (int i = 0; i < 2200; ++i) long_text += "wort" + std::to_string(i % 97) + " ";
    json b = {{"model", "kolibri"},
              {"messages", json::array({{{"role", "user"}, {"content", long_text + "erkläre den Cache bitte"}}})},
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
  KolibriTemplate tmpl;
  MallocAlloc alloc;
  FakeKolibri eng;
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
  FakeKolibri decode(small_desc(), KolPlacement::one(small_desc()), 8192, f.eng.words);
  decode.prefill(prompt);
  CHECK(decode.generate(24) == served);
  // The same request again: restored from the prompt-end snapshot (rings + KV), the same ids.
  const json r2 = f.chat(greedy);
  CHECK(r2.at("prompt_token_ids").get<Ids>() == prompt);
  CHECK(r2.at("choices").at(0).at("token_ids").get<Ids>() == served);
  const uint32_t cached = r2.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
  CHECK_EQ(cached, 2048u);   // the block-end snapshot: the prompt is past 2048, its own end cannot restore
  CHECK(f.eng.state_loads > 0);
  std::printf("server: greedy chat of %zu prompt ids -> 24 ids == the decode run; again: restored at %u, same ids\n",
              prompt.size(), cached);

  // Sampled with the model's defaults (no temperature in the request), seeded: reproducible on a fresh
  // server, the first id drawn from the prefill's row too (n + 1 rows read for n ids), only allowed words.
  const json sampled = {{"seed", 1234}};
  Ids a, b;
  size_t reads = 0;
  {
    ServeFixture g(/*cache=*/false);
    a = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
    reads = g.eng.logits_reads;
  }
  {
    ServeFixture g(/*cache=*/false);
    b = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
  }
  CHECK(a == b);
  CHECK_EQ(reads, a.size() + 1);
  for (uint32_t id : a) CHECK(std::find(f.eng.words.begin(), f.eng.words.end(), id) != f.eng.words.end());
  std::printf("server: a seeded request with Kolibri's default sampling reproducible (%zu ids, %zu rows read), "
              "every id a vocabulary word\n", a.size(), reads);

  // Review Focus 3: the device argmax lands on a row without a token - greedy takes the masked argmax.
  {
    ServeFixture g(/*cache=*/false);
    g.eng.poison = true;
    const Ids ids = g.chat(greedy).at("choices").at(0).at("token_ids").get<Ids>();
    CHECK_EQ(ids.size(), size_t(24));
    for (uint32_t id : ids) CHECK(id < kVocabUsed);
    CHECK(ids == served);                         // the masked argmax is the argmax among the words
    CHECK_EQ(g.adapter.masked(), uint64_t(25));   // the prefill's pending id and every step's
    // sampled: the filter masks them (vocab_used)
    const Ids s = g.chat({{"temperature", 1.5}, {"top_k", 0}, {"top_p", 1.0}, {"seed", 9}})
                      .at("choices").at(0).at("token_ids").get<Ids>();
    for (uint32_t id : s) CHECK(id < kVocabUsed);
  }
  std::printf("server: no id the tokenizer does not define is ever emitted (greedy masked 25 / 25, sampled filtered)\n");
}

}  // namespace

int main() {
  case_layout();
  case_round_trip();
  case_prefix();
  case_pp();
  case_server();
  std::printf("kolibri_snapshot_test OK\n");
  return 0;
}

// Spec 24e (plan 24e Task 4, host side): LFM2.5-24B-A2B behind b70-serve without a device.
// cli::lfm2moe::Lfm2MoeEngineAdapterT - the adapter b70-serve builds over runtime::lfm2moe::Lfm2MoeEngine - runs here over
// FakeLfm, an engine with Lfm2MoeEngine's interface whose conv rings and attention-layer KV are host buffers in the real
// layout (runtime::lfm2moe::persistent_sizes per device of a placement: one card or two) and whose snapshot calls are
// Lfm2MoeEngine's own loops over runtime::lfm2moe::state_runs / kv_runs, memcpy for the device copies. Its "model" reads what
// LFM2's layers read: the query at position q sees every conv layer's ring rows of q - 2, q - 1 and q (slots % 16; a
// position below 0 reads as zero, as lfm_conv does) and every attention layer's K / V rows [0, q]; position q's rows of every
// layer are a function of the id, q, the layer and what the query at q read before its own rows, and the next id a function of
// what the last query reads - so a restore that misplaces one row of one ring or one attention layer, or leaves a needed
// ring row stale, changes what it generates, while ring slots outside the two the state names (stale after a restore, as on
// the card) are never read.
//
//   1. the snapshot layout on LFM2.5's real shapes: 245,760 B of state at every pos (60 runs of 4096 B: 2 per conv layer), the
//      slot arithmetic at pos 0, 1, 2, 15, 16, 17, 2048 (p - 1 and p - 2 mod 16, zero rows below 0), 20,480 B of KV a position
//      (20 runs); every run inside its device allocation and disjoint; the host order the same under --pp 1 and --pp 2 (spec
//      16b's rule); bad ranges refused;
//   2. save / load on host buffers: the two ring rows per conv layer (and the zero rows below position 0) restored bit for
//      bit, everything else untouched; KV ranges likewise; adjacent saves compose;
//   3. Review Focus 5: a restore at positions 0, 1, 2, 15, 16, 17 continues bitwise as the cold run (KV, conv rings, the ids),
//      and through spec 7's request path (server::PrefixSession over the adapter, the real block 2048): restores at the block
//      ends 2048 and 4096, at the request ends 2049, 5000 and 300 - every cached turn's ids equal a cold run's; the store's
//      root is LFM2's key;
//   4. 16b's rule: a state and KV saved under --pp 2 restore under --pp 1 and continue as the two-card session does, and the
//      reverse;
//   5. the server (HTTP, LFM2's chat format): a greedy chat's ids equal the decode run; the same request again restores from
//      the cache (cached_tokens 2048); a seeded sampled request with the model's defaults is reproducible; the repetition
//      penalty reaches the sampler through the server's SeenIds (the prompt's ids, then every consumed id) - the ids equal an
//      independent HF-ordered simulation - and a greedy request ignores it (the device argmax) unless the operator's
//      B70_LFM_PENALTY_GREEDY route is on; a greedy argmax on a head row the tokenizer does not define (125017 ..) is replaced
//      by the masked row's argmax.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/lfm2moe_serve.h"
#include "model/lfm2moe.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"
#include "runtime/prefill_chunks.h"
#include "server/mock.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using Ids = std::vector<uint32_t>;
namespace rl = runtime::lfm2moe;
using model::LfmPlacement;
using rl::SnapRun;
using rl::SnapTensor;

constexpr uint32_t kVocabUsed = 125017;   // LFM2.5's tokenizer.json (lfm2moe_tokenizer_test)

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// LFM2.5's layer pattern (conv conv attention conv conv conv attention conv: 6 conv, 2 attention) at a tiny width: a 32-byte ring
// row and a 16-byte KV row, the real ring (16 slots) and block - everything state_runs / kv_runs read.
model::Lfm2MoeDesc small_desc() {
  model::Lfm2MoeDesc d = model::lfm2moe();
  d.layers = 8;
  d.kinds.resize(8);
  d.hidden = 16;
  d.kv_heads = 1;
  d.head_dim = 8;
  return d;
}

// Lfm2MoeEngine's interface (what Lfm2MoeEngineAdapterT calls), device-free.
struct FakeLfm {
  static constexpr uint32_t kBlock = 2048;   // Lfm2MoeEngine::kBlock (= kPfC)
  using BlockHook = std::function<void(uint32_t, bool)>;

  struct Dev {
    std::vector<uint8_t> kv_k, kv_v, ring;
  };

  FakeLfm(const model::Lfm2MoeDesc& desc, LfmPlacement placement, uint32_t len, Ids vocab_words)
      : d(desc), p(placement), max_len_(len), words(std::move(vocab_words)) {
    model::validate(p, d);
    for (uint32_t i = 0; i < p.devices; ++i) {
      const rl::PersistentSizes s = rl::persistent_sizes(d, p, i, max_len_);
      devs.push_back({std::vector<uint8_t>(s.kv_k), std::vector<uint8_t>(s.kv_v), std::vector<uint8_t>(s.conv_ring)});
    }
  }

  // --- the session ----------------------------------------------------------------------------------
  void reset() {
    for (Dev& v : devs)
      for (auto* m : {&v.kv_k, &v.kv_v, &v.ring}) std::fill(m->begin(), m->end(), uint8_t(0));
    pos_ = 0;
    cur = 0;
  }
  // Lfm2MoeEngine::prefill's control flow (lfm2moe_prefill_engine.cc): chunks by runtime::prefill_chunk_rows (kPfC 2048), the
  // mid-prompt hook at block ends with pos == end, the prompt-end hook after the first generated id is pending.
  void prefill(const Ids& ids) {
    CHECK(!ids.empty());
    const bool hooked = static_cast<bool>(hook);
    const uint32_t base = pos_;
    uint32_t C = 0;
    for (size_t off = 0; off < ids.size(); off += C) {
      C = runtime::prefill_chunk_rows(base + uint32_t(off), ids.size() - off, rl::kPfC, hooked, kBlock);
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
  // The head's row of the last position: the words scored from the last view (the argmax word highest), everything else -1e30;
  // `poison` makes the two rows without a token score highest, as an untrained row might - next() is this row's argmax, the
  // device's. `tie` is a fixed row: word A 10, word B 9.9 - the repetition penalty's test.
  void read_logits_into(float* host) {
    ++logits_reads;
    std::fill(host, host + d.vocab, -1e30f);
    if (tie) {
      host[tie_a] = 10.0f;
      host[tie_b] = 9.9f;
      return;
    }
    const uint64_t v = attend(pos_ == 0 ? 0 : pos_ - 1, true);
    uint64_t h = v;
    for (uint32_t w : words) {
      h = mix(h, w);
      host[w] = float(h % 1000) / 250.0f;   // < 4
    }
    host[words[v % words.size()]] = 10.0f;
    if (poison) host[kVocabUsed] = host[kVocabUsed + 1] = 100.0f;
  }

  // --- Lfm2MoeEngine's snapshot calls (lfm2moe_engine.cc: the same run lists, memcpy for the device) ----
  size_t state_bytes() const { return rl::state_snapshot_bytes(d); }
  size_t kv_bytes(uint32_t n) const { return rl::kv_snapshot_bytes(d, n); }
  std::vector<uint8_t>& mem(uint32_t dev, SnapTensor t) {
    Dev& v = devs.at(dev);
    return t == SnapTensor::KvK ? v.kv_k : t == SnapTensor::KvV ? v.kv_v : v.ring;
  }
  void save_state(void* host) {
    auto* h = static_cast<uint8_t*>(host);
    for (const SnapRun& r : rl::state_runs(d, p, pos_)) {
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
    for (const SnapRun& r : rl::state_runs(d, p, at)) {
      std::vector<uint8_t>& m = mem(r.device, r.tensor);
      CHECK(r.offset + r.bytes <= m.size());
      std::memcpy(m.data() + r.offset, h, r.bytes);
      h += r.bytes;
    }
    pos_ = at;
    ++state_loads;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) {
    auto* h = static_cast<uint8_t*>(host);
    for (const SnapRun& r : rl::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(h, mem(r.device, r.tensor).data() + r.offset, r.bytes);
      h += r.bytes;
    }
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) {
    const auto* h = static_cast<const uint8_t*>(host);
    for (const SnapRun& r : rl::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(mem(r.device, r.tensor).data() + r.offset, h, r.bytes);
      h += r.bytes;
    }
    kv_loaded += e - b;
  }

  // --- the "model" --------------------------------------------------------------------------------------
  // Layer l's conv ring row of position q / attention K (t 0) or V (t 1) row, addressed as Lfm2MoeBuffers::ring_layer /
  // k_layer do (by counting the layers of the kind on the device) - independent of state_runs / kv_runs, so the layout
  // functions are checked against it.
  uint8_t* ring_row(uint32_t l, uint32_t q) {
    const uint32_t dev = p.device_of(l);
    size_t i = 0;
    for (uint32_t k = p.first(dev); k < l; ++k) i += d.is_attention(k) ? 0 : 1;
    return devs[dev].ring.data() + i * rl::ring_layer_bytes(d) + size_t(q % 16) * d.hidden * 2;
  }
  uint8_t* kv_row(uint32_t l, uint32_t q, int t) {
    const uint32_t dev = p.device_of(l);
    size_t i = 0;
    for (uint32_t k = p.first(dev); k < l; ++k) i += d.is_attention(k) ? 1 : 0;
    return (t ? devs[dev].kv_v : devs[dev].kv_k).data() + i * rl::kv_layer_bytes(d, max_len_) + size_t(q) * d.kv_n() * 2;
  }
  // What the query at position q reads: every conv layer's ring rows of q - 2 and q - 1 (below 0: zero, as lfm_conv) and, with
  // `self`, q; every attention layer's K / V rows [0, q) and, with `self`, q.
  uint64_t attend(uint32_t q, bool self) {
    uint64_t h = 0x51ED;
    const size_t ring_words = size_t(d.hidden) * 2 / 8;
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (!d.is_attention(l)) {
        for (int back = 2; back >= (self ? 0 : 1); --back) {
          if (int64_t(q) - back < 0) {
            h = mix(h, 0x2E80 + uint64_t(l) * 8 + uint64_t(back));
            continue;
          }
          const uint8_t* r = ring_row(l, q - uint32_t(back));
          for (size_t w = 0; w < ring_words; ++w) {
            uint64_t a;
            std::memcpy(&a, r + w * 8, 8);
            h = mix(h, a ^ (uint64_t(l) << 59) ^ uint64_t(back));
          }
        }
      } else {
        const uint32_t end = self ? q + 1 : q;
        for (uint32_t j = 0; j < end; ++j) {
          uint64_t a, b;
          std::memcpy(&a, kv_row(l, j, 0), 8);
          std::memcpy(&b, kv_row(l, j, 1) + 8, 8);
          h = mix(h, a ^ (b << 1) ^ (uint64_t(l) << 59));
        }
      }
    }
    return h;
  }
  void write(uint32_t id) {
    CHECK(pos_ < max_len_);
    const uint64_t prev = attend(pos_, false);
    const size_t kvb = size_t(d.kv_n()) * 2, rb = size_t(d.hidden) * 2;
    for (uint32_t l = 0; l < d.layers; ++l) {
      for (int t = 0; t < 2; ++t) {
        if (!d.is_attention(l) && t == 1) continue;
        uint64_t h = mix(mix(mix(prev, id), uint64_t(pos_) * 131 + l), uint64_t(t));
        uint8_t* r = d.is_attention(l) ? kv_row(l, pos_, t) : ring_row(l, pos_);
        const size_t bytes = d.is_attention(l) ? kvb : rb;
        for (size_t i = 0; i < bytes; i += 8) {
          h = mix(h, i);
          std::memcpy(r + i, &h, std::min<size_t>(8, bytes - i));
        }
      }
    }
    ++pos_;
  }
  uint32_t next() { return poison ? uint32_t(kVocabUsed + 1) : tie ? tie_a : words[attend(pos_ - 1, true) % words.size()]; }

  model::Lfm2MoeDesc d;
  LfmPlacement p;
  uint32_t max_len_;
  std::vector<Dev> devs;
  Ids words;
  uint32_t pos_ = 0, cur = 0;
  BlockHook hook;
  bool poison = false, tie = false;
  uint32_t tie_a = 0, tie_b = 0;
  size_t state_loads = 0, kv_loaded = 0, logits_reads = 0;
};

using Adapter = cli::lfm2moe::Lfm2MoeEngineAdapterT<FakeLfm>;

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

// ---- 1. the layout on LFM2.5's real shapes --------------------------------------------------------------------
void check_disjoint_inside(const model::Lfm2MoeDesc& d, const LfmPlacement& p, uint32_t max_len,
                           const std::vector<SnapRun>& runs) {
  for (size_t i = 0; i < runs.size(); ++i) {
    const SnapRun& r = runs[i];
    const rl::PersistentSizes s = rl::persistent_sizes(d, p, r.device, max_len);
    const size_t cap = r.tensor == SnapTensor::KvK ? s.kv_k : r.tensor == SnapTensor::KvV ? s.kv_v : s.conv_ring;
    CHECK(r.bytes > 0);
    CHECK(r.offset + r.bytes <= cap);
    for (size_t q = 0; q < i; ++q)
      if (runs[q].device == r.device && runs[q].tensor == r.tensor)
        CHECK(runs[q].offset + runs[q].bytes <= r.offset || r.offset + r.bytes <= runs[q].offset);
  }
}

// The host order of a run list, placement-free: (tensor, the run's offset within its layer's block, bytes, zero).
std::vector<std::vector<size_t>> host_order(const model::Lfm2MoeDesc& d, uint32_t max_len, const std::vector<SnapRun>& runs) {
  std::vector<std::vector<size_t>> out;
  for (const SnapRun& r : runs) {
    const size_t per = r.tensor == SnapTensor::ConvRing ? rl::ring_layer_bytes(d) : rl::kv_layer_bytes(d, max_len);
    out.push_back({size_t(r.tensor), r.offset % per, r.bytes, size_t(r.zero)});
  }
  return out;
}

void case_layout() {
  const model::Lfm2MoeDesc& d = model::lfm2moe();
  const uint32_t len = 131072;
  CHECK_EQ(d.conv_layers(), 30u);
  CHECK_EQ(d.attn_layers(), 10u);
  CHECK_EQ(rl::kStatePositions, 2u);
  CHECK_EQ(rl::state_snapshot_bytes(d), size_t(245760));
  CHECK_EQ(rl::kv_snapshot_bytes(d, 1), size_t(20480));
  CHECK_EQ(rl::kv_snapshot_bytes(d, 2048), size_t(41943040));
  const std::vector<LfmPlacement> places = {LfmPlacement::one(d), LfmPlacement::two(d, 3), LfmPlacement::two(d, 20),
                                            LfmPlacement::two(d, 39)};
  const size_t row = size_t(d.hidden) * 2;
  CHECK_EQ(row, size_t(4096));
  for (uint32_t pos : {0u, 1u, 2u, 3u, 15u, 16u, 17u, 18u, 2047u, 2048u, 2049u, 4096u, 5000u, 131071u, 131072u}) {
    std::vector<std::vector<size_t>> first;
    for (const LfmPlacement& p : places) {
      const std::vector<SnapRun> runs = rl::state_runs(d, p, pos);
      CHECK_EQ(runs.size(), size_t(60));   // two rows a conv layer
      size_t total = 0, zero = 0;
      uint32_t layer = 0;
      for (size_t i = 0; i < runs.size(); ++i) {
        const SnapRun& r = runs[i];
        total += r.bytes;
        zero += r.zero ? r.bytes : 0;
        CHECK(r.tensor == SnapTensor::ConvRing);
        CHECK_EQ(r.bytes, row);
        // Run 2c + j is conv layer c's row of position pos - 1 - j, in slot (pos - 1 - j) mod 16 of its layer's ring.
        const uint32_t c = uint32_t(i / 2), j = uint32_t(i % 2);
        const uint32_t slot = uint32_t((int64_t(pos) - 1 - int64_t(j) + 32) % 16);
        CHECK_EQ(r.offset % rl::ring_layer_bytes(d), size_t(slot) * row);
        CHECK_EQ(r.zero, int64_t(pos) - 1 - int64_t(j) < 0);
        // the allocation index of conv layer c on its device, counted the long way
        while (d.is_attention(layer)) ++layer;
        CHECK_EQ(r.device, p.device_of(layer));
        size_t idx = 0;
        for (uint32_t k = p.first(r.device); k < layer; ++k) idx += d.is_attention(k) ? 0 : 1;
        CHECK_EQ(r.offset / rl::ring_layer_bytes(d), idx);
        if (j == 1) ++layer;
        (void)c;
      }
      CHECK_EQ(total, rl::state_snapshot_bytes(d));
      CHECK_EQ(zero, size_t(pos == 0 ? 2 : pos == 1 ? 1 : 0) * 30 * row);   // the bytes below position 0 are meaningless: zeros
      check_disjoint_inside(d, p, len, runs);
      if (first.empty()) first = host_order(d, len, runs);
      else CHECK(host_order(d, len, runs) == first);   // 16b's rule: one host layout
    }
  }
  for (const LfmPlacement& p : places) {
    CHECK(rl::kv_runs(d, p, len, 100, 100).empty());
    const std::vector<SnapRun> runs = rl::kv_runs(d, p, len, 2048, 4096);
    CHECK_EQ(runs.size(), size_t(20));   // K then V of 10 attention layers
    size_t total = 0;
    for (size_t i = 0; i < runs.size(); ++i) {
      total += runs[i].bytes;
      CHECK(runs[i].tensor == (i < 10 ? SnapTensor::KvK : SnapTensor::KvV));
      CHECK_EQ(runs[i].bytes, size_t(2048) * 1024);
    }
    CHECK_EQ(total, rl::kv_snapshot_bytes(d, 2048));
    check_disjoint_inside(d, p, len, runs);
    CHECK(host_order(d, len, runs) == host_order(d, len, rl::kv_runs(d, places[0], len, 2048, 4096)));
    for (const auto& range : {std::make_pair(10u, len + 1), std::make_pair(5u, 4u)}) {
      bool threw = false;
      try {
        (void)rl::kv_runs(d, p, len, range.first, range.second);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      CHECK(threw);
    }
  }
  // A placement of another model's layer count is refused, not silently misaddressed.
  bool threw = false;
  try {
    (void)rl::state_runs(d, LfmPlacement::one(small_desc()), 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("layout: state 245760 B (60 runs of 4096 B, slots (pos - 1) and (pos - 2) mod 16, zeros below 0) at every pos, "
              "KV 20480 B a position (20 runs), inside the allocations, one host layout for --pp 1 and --pp 2 at splits 3, "
              "20 and 39\n");
}

// ---- 2. save / load on host buffers --------------------------------------------------------------------------
void scramble(FakeLfm& e, uint32_t seed) {
  std::mt19937_64 rng(seed);
  for (auto& v : e.devs)
    for (auto* m : {&v.kv_k, &v.kv_v, &v.ring})
      for (auto& b : *m) b = uint8_t(rng());
}

void case_round_trip() {
  const model::Lfm2MoeDesc d = small_desc();
  const size_t rb = size_t(d.hidden) * 2, kvb = size_t(d.kv_n()) * 2;
  for (const LfmPlacement& p : {LfmPlacement::one(d), LfmPlacement::two(d, 3)}) {
    for (uint32_t at : {0u, 1u, 2u, 15u, 16u, 17u, 300u, 2048u, 4097u}) {
      FakeLfm e(d, p, 8192, some_words());
      scramble(e, at);
      e.pos_ = at;
      std::vector<uint8_t> state(e.state_bytes());
      e.save_state(state.data());
      // The layout: per conv layer, the row of position at - 1 then the row of at - 2; zeros for a position below 0.
      size_t k = 0;
      std::vector<std::vector<uint8_t>> want_rows;
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (d.is_attention(l)) continue;
        for (uint32_t j = 0; j < 2; ++j) {
          const int64_t q = int64_t(at) - 1 - int64_t(j);
          if (q < 0) {
            for (size_t i = 0; i < rb; ++i) CHECK_EQ(state[k * rb + i], uint8_t(0));
            want_rows.push_back(std::vector<uint8_t>(rb, 0));
          } else {
            CHECK(std::equal(state.begin() + k * rb, state.begin() + (k + 1) * rb, e.ring_row(l, uint32_t(q))));
            want_rows.emplace_back(e.ring_row(l, uint32_t(q)), e.ring_row(l, uint32_t(q)) + rb);
          }
          ++k;
        }
      }
      CHECK_EQ(k * rb, state.size());
      FakeLfm f(d, p, 8192, some_words());
      scramble(f, at + 1);
      const FakeLfm before = f;
      f.load_state(state.data(), at);
      CHECK_EQ(f.pos(), at);
      k = 0;
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (d.is_attention(l)) continue;
        for (uint32_t j = 0; j < 2; ++j) {
          const int64_t q = int64_t(at) - 1 - int64_t(j);
          const uint32_t slot_q = uint32_t((q + 32) % 16);
          CHECK(std::equal(want_rows[k].begin(), want_rows[k].end(), f.ring_row(l, slot_q)));   // ring_row indexes by q % 16
          ++k;
        }
      }
      // The KV and every ring slot other than the two (a position below 0 included) are untouched.
      for (uint32_t dv = 0; dv < p.devices; ++dv) {
        CHECK(f.devs[dv].kv_k == before.devs[dv].kv_k && f.devs[dv].kv_v == before.devs[dv].kv_v);
        std::vector<bool> in(f.devs[dv].ring.size(), false);
        for (const SnapRun& r : rl::state_runs(d, p, at))
          if (r.device == dv)
            for (size_t i = 0; i < r.bytes; ++i) in[r.offset + i] = true;
        for (size_t i = 0; i < in.size(); ++i)
          if (!in[i]) CHECK_EQ(f.devs[dv].ring[i], before.devs[dv].ring[i]);
      }
      // KV: [17, at) restored bit for bit, the rest untouched; adjacent saves compose. (at <= 17: nothing to copy.)
      if (at > 17) {
        std::vector<uint8_t> kv(e.kv_bytes(at - 17)), a(e.kv_bytes(100 > at - 17 ? 1 : 100)),
            b(e.kv_bytes(at - 17 - (100 > at - 17 ? 1 : 100)));
        const uint32_t mid = 17 + (100 > at - 17 ? 1 : 100);
        e.save_kv(17, at, kv.data());
        e.save_kv(17, mid, a.data());
        e.save_kv(mid, at, b.data());
        FakeLfm x(d, p, 8192, some_words()), y(d, p, 8192, some_words());
        x.load_kv(17, mid, a.data());
        x.load_kv(mid, at, b.data());
        y.load_kv(17, at, kv.data());
        for (uint32_t dv = 0; dv < p.devices; ++dv) CHECK(x.devs[dv].kv_k == y.devs[dv].kv_k && x.devs[dv].kv_v == y.devs[dv].kv_v);
        for (uint32_t l = 0; l < d.layers; ++l)
          if (d.is_attention(l))
            for (uint32_t q = 0; q < at; ++q)
              for (int t = 0; t < 2; ++t) {
                if (q >= 17) CHECK(std::equal(e.kv_row(l, q, t), e.kv_row(l, q, t) + kvb, y.kv_row(l, q, t)));
                else
                  for (size_t i = 0; i < kvb; ++i) CHECK_EQ(y.kv_row(l, q, t)[i], uint8_t(0));
              }
      }
      e.save_kv(5, 5, nullptr);   // empty: no copy
    }
  }
  std::printf("round trip: the two ring rows per conv layer (zeros below 0) and KV ranges restored bit for bit, nothing else "
              "touched, on one card and two\n");
}

// ---- 3. Review Focus 5 ---------------------------------------------------------------------------------------
Ids cold_tail(FakeLfm& e, const Ids& tail, uint32_t n) {
  e.prefill(tail);
  return e.generate(n);
}

void case_tiny_positions() {
  // A restore at positions 0, 1, 2 (where a conv row is a zero row), 15, 16, 17 (the ring's wrap) continues bitwise as the cold
  // session: the ring rows the state names, every KV row and the generated ids.
  const model::Lfm2MoeDesc d = small_desc();
  Ids all;
  for (uint32_t i = 0; i < 60; ++i) all.push_back(3000 + (i * 7919) % 5000);
  for (const LfmPlacement& p : {LfmPlacement::one(d), LfmPlacement::two(d, 5)}) {
    for (uint32_t at : {0u, 1u, 2u, 3u, 15u, 16u, 17u, 18u}) {
      const Ids head(all.begin(), all.begin() + at), tail(all.begin() + at, all.end());
      FakeLfm a(d, p, 1024, some_words());
      if (at > 0) a.prefill(head);
      CHECK_EQ(a.pos(), at);
      std::vector<uint8_t> state(a.state_bytes()), kv(a.kv_bytes(at));
      a.save_state(state.data());
      a.save_kv(0, at, kv.data());
      FakeLfm b(d, p, 1024, some_words());
      scramble(b, 40 + at);   // another session's leftovers everywhere
      b.load_kv(0, at, kv.data());
      b.load_state(state.data(), at);
      const Ids ga = cold_tail(a, tail, 16), gb = cold_tail(b, tail, 16);
      CHECK(ga == gb);
      CHECK_EQ(a.pos(), b.pos());
      // The two ring rows and every KV row [0, pos) agree at the end.
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (d.is_attention(l)) {
          for (uint32_t q = 0; q < a.pos(); ++q)
            for (int t = 0; t < 2; ++t)
              CHECK(std::equal(a.kv_row(l, q, t), a.kv_row(l, q, t) + 16, b.kv_row(l, q, t)));
        } else {
          for (uint32_t back = 1; back <= 2; ++back)
            CHECK(std::equal(a.ring_row(l, a.pos() - back), a.ring_row(l, a.pos() - back) + 32, b.ring_row(l, a.pos() - back)));
        }
      }
    }
  }
  std::printf("restores at positions 0, 1, 2, 3, 15, 16, 17, 18 continue bitwise as the cold session (KV, rings, ids), on one "
              "card and two\n");
}

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

Ids cold(const model::Lfm2MoeDesc& d, const LfmPlacement& p, const Ids& prompt, uint32_t n) {
  FakeLfm e(d, p, 8192, some_words());
  e.prefill(prompt);
  return e.generate(n);
}

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

void case_prefix() {
  const model::Lfm2MoeDesc d = small_desc();
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
                        Want{"request end 300", 300, 292}}) {
    for (const LfmPlacement& p : {LfmPlacement::one(d), LfmPlacement::two(d, 3)}) {
      FakeLfm eng(d, p, 8192, some_words());
      Adapter ad(eng, kVocabUsed);
      CHECK_EQ(ad.block(), 2048u);
      CHECK_EQ(ad.state_bytes(), rl::state_snapshot_bytes(d));
      server::PrefixSession s(ad, size_t(1) << 30, &alloc);
      CHECK_EQ(s.cache()->kv_form(), cli::lfm2moe::kLfm2MoeStoreKey);
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
      const Ids want = cold(d, LfmPlacement::one(d), final_prompt, 16);
      CHECK(got == want);   // a restore continues as the cold run
      std::printf("  %s, %s: restore at %u (%zu B of KV), 16 ids == cold\n", w.name, p.devices == 2 ? "--pp 2" : "--pp 1",
                  rep.restart, rep.kv_bytes);
    }
  }
  CHECK_EQ(alloc.live, size_t(0));
}

// ---- 4. 16b's rule: one host layout for one card and two --------------------------------------------------
void case_pp() {
  const model::Lfm2MoeDesc d = small_desc();
  Ids prompt;
  for (uint32_t i = 0; i < 2600; ++i) prompt.push_back(3000 + (i * 104729) % 40000);
  for (int dir = 0; dir < 2; ++dir) {
    const LfmPlacement from = dir == 0 ? LfmPlacement::two(d, 3) : LfmPlacement::one(d);
    const LfmPlacement to = dir == 0 ? LfmPlacement::one(d) : LfmPlacement::two(d, 3);
    FakeLfm a(d, from, 8192, some_words());
    a.prefill(prompt);
    (void)a.generate(5);
    const uint32_t at = a.pos();
    std::vector<uint8_t> state(a.state_bytes()), kv(a.kv_bytes(at));
    a.save_state(state.data());
    a.save_kv(0, at, kv.data());
    FakeLfm b(d, to, 8192, some_words());
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
struct LfmTemplate : server::TemplateIface {
  std::string render(const json& messages, const json&, bool) override { return body(messages); }
  std::string render_with_kwargs(const json& messages, const json&, bool, const json&) override { return body(messages); }
  static std::string body(const json& messages) {
    std::string text = "<|startoftext|> ";
    for (const auto& m : messages)
      text += "<|im_start|> " + m.at("role").get<std::string>() + " " + m.at("content").get<std::string>() + " <|im_end|> ";
    return text + "<|im_start|> assistant <think>";   // the real template's generation prompt
  }
};

const json kLfmGeneration = {{"do_sample", true}, {"temperature", 0.1}, {"top_k", 50}, {"repetition_penalty", 1.05}};

struct ServeFixture {
  explicit ServeFixture(bool cache, bool penalty_greedy = false, bool tie_mode = false)
      : eng(small_desc(), LfmPlacement::two(small_desc(), 3), 8192, words()),
        adapter(eng, kVocabUsed, penalty_greedy),
        server({tok, tmpl, adapter}, options(cache)) {
    if (tie_mode) {
      eng.tie = true;
      eng.tie_a = tok.id_of("alpha");
      eng.tie_b = tok.id_of("beta");
    }
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
    o.served_model = "lfm";
    o.eos_ids = {124900, 248046};   // 124900 is LFM2.5's; 248046 is the mock tokenizer's
    o.chat_format = server::ChatFormat::lfm2();
    o.sampling_defaults = server::generation_sampling(kLfmGeneration);
    if (cache) {
      o.prefix_cache_bytes = size_t(256) << 20;
      o.prefix_alloc = &alloc;
    }
    return o;
  }
  json chat_text(const std::string& user, const json& extra) {
    json b = {{"model", "lfm"},
              {"messages", json::array({{{"role", "user"}, {"content", user}}})},
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
  json chat(const json& extra) {
    std::string long_text;
    for (int i = 0; i < 2200; ++i) long_text += "wort" + std::to_string(i % 97) + " ";
    return chat_text(long_text + "explain the cache please", extra);
  }
  Ids out_ids(const json& response) { return response.at("choices").at(0).at("token_ids").get<Ids>(); }

  MockTok tok;   // declared first: words() needs it
  LfmTemplate tmpl;
  MallocAlloc alloc;
  FakeLfm eng;
  Adapter adapter;
  server::Server server;
};

// HF's loop, independently: at each step the next id is the argmax of the row {A 10, B 9.9} with the repetition penalty
// applied to the ids seen so far (the prompt's, then every id produced).
Ids hf_tie_simulation(uint32_t a, uint32_t b, std::set<uint32_t> seen, float penalty, uint32_t n) {
  Ids out;
  for (uint32_t i = 0; i < n; ++i) {
    float la = 10.0f, lb = 9.9f;
    if (seen.count(a)) la = la < 0 ? la * penalty : la / penalty;
    if (seen.count(b)) lb = lb < 0 ? lb * penalty : lb / penalty;
    const uint32_t id = la >= lb ? a : b;
    out.push_back(id);
    seen.insert(id);
  }
  return out;
}

void case_server() {
  ServeFixture f(/*cache=*/true);
  const json greedy = {{"temperature", 0}};
  const json r1 = f.chat(greedy);
  const Ids prompt = r1.at("prompt_token_ids").get<Ids>();
  const Ids served = f.out_ids(r1);
  CHECK(prompt.size() > 2048);
  CHECK_EQ(served.size(), size_t(24));
  // b70-decode --prefill's run of the same ids
  FakeLfm decode(small_desc(), LfmPlacement::one(small_desc()), 8192, f.eng.words);
  decode.prefill(prompt);
  CHECK(decode.generate(24) == served);
  // The same request again: restored from the block-end snapshot (conv rings + KV), the same ids.
  const json r2 = f.chat(greedy);
  CHECK(r2.at("prompt_token_ids").get<Ids>() == prompt);
  CHECK(f.out_ids(r2) == served);
  const uint32_t cached = r2.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
  CHECK_EQ(cached, 2048u);   // the block-end snapshot: the prompt is past 2048, its own end cannot restore
  CHECK(f.eng.state_loads > 0);
  std::printf("server: greedy chat of %zu prompt ids -> 24 ids == the decode run; again: restored at %u, same ids\n",
              prompt.size(), cached);

  // Sampled with the model's defaults (T 0.1, top-k 50, penalty 1.05; the request names none), seeded: reproducible on a
  // fresh server, the first id drawn from the prefill's row too (n reads for n ids), only words the row allows.
  const json sampled = {{"seed", 1234}};
  Ids a, b;
  size_t reads = 0;
  {
    ServeFixture g(/*cache=*/false);
    a = g.out_ids(g.chat(sampled));
    reads = g.eng.logits_reads;
  }
  {
    ServeFixture g(/*cache=*/false);
    b = g.out_ids(g.chat(sampled));
  }
  CHECK(a == b);
  CHECK_EQ(reads, a.size());
  for (uint32_t id : a) CHECK(std::find(f.eng.words.begin(), f.eng.words.end(), id) != f.eng.words.end());
  std::printf("server: a seeded request with LFM2's default sampling reproducible (%zu ids, %zu rows read), every id a "
              "vocabulary word\n", a.size(), reads);

  // Decision 1: the repetition penalty reaches the host sampler through the server's SeenIds. The row is {A 10, B 9.9}, T 0.01,
  // top-k 2: the id is the argmax of the penalised row - the prompt's ids and every id produced so far are seen.
  {
    ServeFixture g(/*cache=*/false, /*penalty_greedy=*/false, /*tie_mode=*/true);
    const uint32_t A = g.eng.tie_a, B = g.eng.tie_b;
    const json sharp = {{"temperature", 0.01}, {"top_k", 2}, {"top_p", 1.0}, {"seed", 5}};
    json with = sharp;
    with["repetition_penalty"] = 1.05;
    // a prompt that holds A: the first id is already B
    const Ids got_a = g.out_ids(g.chat_text("alpha question", with));
    const Ids prompt_a = g.tok.encode("<|startoftext|> <|im_start|> user alpha question <|im_end|> <|im_start|> assistant <think>");
    CHECK(hf_tie_simulation(A, B, std::set<uint32_t>(prompt_a.begin(), prompt_a.end()), 1.05f, 24) == got_a);
    CHECK_EQ(got_a[0], B);
    // a prompt without A or B: A, then (A seen) B, then both seen: A
    const Ids got_b = g.out_ids(g.chat_text("gamma question", with));
    const Ids prompt_b = g.tok.encode("<|startoftext|> <|im_start|> user gamma question <|im_end|> <|im_start|> assistant <think>");
    CHECK(hf_tie_simulation(A, B, std::set<uint32_t>(prompt_b.begin(), prompt_b.end()), 1.05f, 24) == got_b);
    CHECK_EQ(got_b[0], A);
    CHECK_EQ(got_b[1], B);
    // the same request with the model's default penalty (1.05, the request names none) is the same as naming it
    json dflt = sharp;
    CHECK(g.out_ids(g.chat_text("gamma question", dflt)) == got_b);
    // penalty 1.0 in the request: no penalty, A every time
    json off = sharp;
    off["repetition_penalty"] = 1.0;
    const Ids got_off = g.out_ids(g.chat_text("gamma question", off));
    CHECK(got_off == Ids(24, A));
    // a greedy request ignores the penalty: the device argmax (A every time), no row read
    json greedy_pen = {{"temperature", 0}, {"repetition_penalty", 1.05}};
    const size_t reads0 = g.eng.logits_reads;
    CHECK(g.out_ids(g.chat_text("gamma question", greedy_pen)) == Ids(24, A));
    CHECK_EQ(g.eng.logits_reads, reads0);
  }
  {
    ServeFixture g(/*cache=*/false, /*penalty_greedy=*/true, /*tie_mode=*/true);   // B70_LFM_PENALTY_GREEDY=1
    const uint32_t A = g.eng.tie_a, B = g.eng.tie_b;
    const json greedy_pen = {{"temperature", 0}, {"repetition_penalty", 1.05}};
    const Ids prompt_b = g.tok.encode("<|startoftext|> <|im_start|> user gamma question <|im_end|> <|im_start|> assistant <think>");
    CHECK(g.out_ids(g.chat_text("gamma question", greedy_pen)) ==
          hf_tie_simulation(A, B, std::set<uint32_t>(prompt_b.begin(), prompt_b.end()), 1.05f, 24));
    CHECK(g.adapter.penalised_greedy() > 0);
    // a request that turns the model's default penalty (1.05) off: the route changes nothing
    const uint64_t routed = g.adapter.penalised_greedy();
    CHECK(g.out_ids(g.chat_text("gamma question", {{"temperature", 0}, {"repetition_penalty", 1.0}})) == Ids(24, A));
    CHECK_EQ(g.adapter.penalised_greedy(), routed);
  }
  std::printf("server: the repetition penalty is HF's over the prompt's ids and every consumed id (sampled), ignored by a "
              "greedy request unless B70_LFM_PENALTY_GREEDY routes it\n");

  // Review Focus 3 of 24e: the device argmax lands on a row without a token - greedy takes the masked argmax.
  {
    ServeFixture g(/*cache=*/false);
    g.eng.poison = true;
    const Ids ids = g.out_ids(g.chat(greedy));
    CHECK_EQ(ids.size(), size_t(24));
    for (uint32_t id : ids) CHECK(id < kVocabUsed);
    CHECK(ids == served);                         // the masked argmax is the argmax among the words
    CHECK_EQ(g.adapter.masked(), uint64_t(24));   // one per consumed id: the prefill's pending id and every step's
    // sampled: the filter masks them (vocab_used)
    const Ids s = g.out_ids(g.chat({{"temperature", 1.5}, {"top_k", 0}, {"top_p", 1.0}, {"seed", 9}}));
    for (uint32_t id : s) CHECK(id < kVocabUsed);
  }
  std::printf("server: no id the tokenizer does not define is ever emitted (greedy masked 24 / 24, sampled filtered)\n");
}

}  // namespace

int main() {
  case_layout();
  case_round_trip();
  case_tiny_positions();
  case_prefix();
  case_pp();
  case_server();
  std::printf("lfm2moe_snapshot_test OK\n");
  return 0;
}

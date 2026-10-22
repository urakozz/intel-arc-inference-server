// Spec 25e (plan 25e Task 4, host side): Nemotron 3.5 Bolt behind b70-serve without a device.
// cli::bolt::BoltEngineAdapterT - the adapter b70-serve builds over runtime::bolt::BoltEngine - runs here over FakeBolt, an
// engine with BoltEngine's interface whose attention KV, Mamba conv rings and SSM states are host buffers in the real layout
// (runtime::bolt::persistent_sizes per device of a placement: one card or two) and whose snapshot calls are BoltEngine's own
// loops over runtime::bolt::state_runs / kv_runs, memcpy for the device copies. Its "model" reads what Bolt's blocks read: the
// query at position q sees every Mamba block's SSM state (the state AFTER q - 1) and its raw-xBC ring columns of q - 3 .. q - 1
// (slots % 16; a position below 0 reads as zero, as nb_conv does), and every attention block's K / V rows [0, q); it then
// updates each state and writes position q's ring column and KV rows, as functions of the id, q, the block and what the query
// read - and the next id is a function of what the last query read - so a restore that misplaces one column of one ring, one
// KV row of one block or one byte of one SSM state, or leaves a needed column stale, changes what it generates, while ring slots
// outside the three the state names (stale after a restore, as on the card) are never read.
//
//   1. the snapshot layout on Bolt's real shapes: 170,229,760 B of state = 167,772,160 B of fp32 SSM states + 2,457,600 B of
//      conv columns (83,886,080 + 2,457,600 with the bf16 arm), 160 runs (an SSM state and three columns per Mamba block), 8,192 B
//      of KV a position (16 runs: K then V of 8 attention blocks); every run inside its device allocation and disjoint; the host
//      order the same under --pp 1 and --pp 2 at several splits (spec 16b's rule); the slot arithmetic at pos 0, 1, 2, 3, 15, 16,
//      17, 2048 (p - 1 .. p - 3 mod 16, zero columns below 0); bad ranges refused;
//   2. save / load on host buffers: the SSM states and the three columns per block (zeros below position 0) restored bit for bit,
//      everything else untouched; KV ranges likewise, adjacent saves compose; the bf16 arm's states half the size;
//   3. Review Focus 4: a restore at positions 0, 1, 2, 3, 4, 15, 16, 17, 18 continues bitwise as the cold run (KV, rings, states,
//      ids), and through spec 7's request path (server::PrefixSession over the adapter, the real block 2048): restores at the
//      block ends 2048, 4096 and 6144 and at the request ends 2049, 5000 and 300 - every cached turn's ids and the whole memory
//      equal a cold run's; the store's root is Bolt's key (the bf16 arm's another);
//   4. 16b's rule: a state and KV saved under --pp 2 restore under --pp 1 and under another split, and the reverse - bitwise;
//   5. the server (HTTP, Bolt's chat format): a greedy chat's ids equal the decode run; the same request again restores from
//      the cache (cached_tokens 2048); a seeded sampled request is reproducible (on the same server and on a fresh one), an
//      unseeded one is not a seeded one's continuation; a greedy argmax on a head row the tokenizer does not define is replaced
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
#include "cli/bolt_serve.h"
#include "model/bolt.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/prefill_chunks.h"
#include "server/mock.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using Ids = std::vector<uint32_t>;
namespace rb = runtime::bolt;
using model::BoltBlock;
using model::BoltPlacement;
using rb::BoltState;
using rb::SnapRun;
using rb::SnapTensor;

constexpr uint32_t kVocabUsed = 131072;   // Bolt's tokenizer.json (bolt_tokenizer_test)

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// Bolt's block pattern in miniature (Mamba, MoE, Mamba, attention ... : 4 Mamba, 2 attention, 2 MoE blocks) at a tiny width: a
// 48-byte conv column, a 16-byte KV row, a 128-byte fp32 SSM state; the real ring (16 slots) and block (2048) - everything
// state_runs / kv_runs read.
model::BoltDesc small_desc() {
  model::BoltDesc d = model::bolt();
  d.layers = 8;
  d.block = {BoltBlock::Mamba, BoltBlock::Moe, BoltBlock::Mamba, BoltBlock::Attention,
             BoltBlock::Mamba, BoltBlock::Moe, BoltBlock::Mamba, BoltBlock::Attention};
  d.experts = {0, 16, 0, 0, 0, 16, 0, 0};
  d.inter = {0, 64, 0, 0, 0, 64, 0, 0};
  d.hidden = 16;
  d.m_heads = 2;
  d.m_head_dim = 4;
  d.m_state = 4;
  d.m_groups = 1;
  d.m_conv_dim = 24;
  d.q_heads = 2;
  d.kv_heads = 1;
  d.head_dim = 8;
  return d;
}

// BoltEngine's interface (what BoltEngineAdapterT calls), device-free.
struct FakeBolt {
  static constexpr uint32_t kBlock = 2048;   // BoltEngine::kBlock (= kPfC)
  using BlockHook = std::function<void(uint32_t, bool)>;

  struct Dev {
    std::vector<uint8_t> kv_k, kv_v, ring, ssm;
  };

  FakeBolt(const model::BoltDesc& desc, BoltPlacement placement, uint32_t len, Ids vocab_words, BoltState state = BoltState::Fp32)
      : d(desc), p(placement), st(state), max_len_(len), words(std::move(vocab_words)) {
    model::validate(p, d);
    for (uint32_t i = 0; i < p.devices; ++i) {
      const rb::PersistentSizes s = rb::persistent_sizes(d, p, i, max_len_, st);
      devs.push_back({std::vector<uint8_t>(s.kv_k), std::vector<uint8_t>(s.kv_v), std::vector<uint8_t>(s.conv_ring),
                      std::vector<uint8_t>(s.ssm)});
    }
  }

  // --- the session ----------------------------------------------------------------------------------
  void reset() {
    for (Dev& v : devs)
      for (auto* m : {&v.kv_k, &v.kv_v, &v.ring, &v.ssm}) std::fill(m->begin(), m->end(), uint8_t(0));
    pos_ = 0;
    cur = 0;
  }
  // BoltEngine::prefill's control flow (bolt_prefill_engine.cc): chunks by runtime::prefill_chunk_rows (kPfC 2048), the mid-prompt
  // hook at block ends (not the last chunk's) with pos == end, the prompt-end hook after the first generated id is pending.
  void prefill(const Ids& ids) {
    CHECK(!ids.empty());
    const bool hooked = static_cast<bool>(hook);
    const uint32_t base = pos_;
    uint32_t C = 0;
    for (size_t off = 0; off < ids.size(); off += C) {
      C = runtime::prefill_chunk_rows(base + uint32_t(off), ids.size() - off, 2048, hooked, kBlock);
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
  // `poison` makes the two rows without a token score highest, as an untrained row might (the tokenizer defines fewer rows
  // here: the adapter's vocab_used is what the test sets).
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
    if (poison) host[poison_at] = host[poison_at + 1] = 100.0f;
  }

  // --- BoltEngine's snapshot calls (bolt_engine.cc: the same run lists, memcpy for the device) ----
  size_t state_bytes() const { return rb::state_snapshot_bytes(d, st); }
  size_t kv_bytes(uint32_t n) const { return rb::kv_snapshot_bytes(d, n); }
  bool bf16_state() const { return st == BoltState::Bf16; }
  std::vector<uint8_t>& mem(uint32_t dev, SnapTensor t) {
    Dev& v = devs.at(dev);
    return t == SnapTensor::KvK ? v.kv_k : t == SnapTensor::KvV ? v.kv_v : t == SnapTensor::Ssm ? v.ssm : v.ring;
  }
  void save_state(void* host) {
    auto* h = static_cast<uint8_t*>(host);
    for (const SnapRun& r : rb::state_runs(d, p, pos_, st)) {
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
    for (const SnapRun& r : rb::state_runs(d, p, at, st)) {
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
    for (const SnapRun& r : rb::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(h, mem(r.device, r.tensor).data() + r.offset, r.bytes);
      h += r.bytes;
    }
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) {
    const auto* h = static_cast<const uint8_t*>(host);
    for (const SnapRun& r : rb::kv_runs(d, p, max_len_, b, e)) {
      std::memcpy(mem(r.device, r.tensor).data() + r.offset, h, r.bytes);
      h += r.bytes;
    }
    kv_loaded += e - b;
  }

  // --- the "model" --------------------------------------------------------------------------------------
  // Block l's conv column of position q, SSM state, attention K (t 0) or V (t 1) row, addressed as BoltBuffers::ring_layer /
  // ssm_layer / k_layer do (by counting the blocks of the kind on the device) - independent of state_runs / kv_runs, so the
  // layout functions are checked against it.
  size_t kind_index(uint32_t l) const {
    const uint32_t dev = p.device_of(l);
    size_t i = 0;
    for (uint32_t k = p.first(dev); k < l; ++k) i += d.block.at(k) == d.block.at(l) ? 1 : 0;
    return i;
  }
  uint8_t* ring_row(uint32_t l, uint32_t q) {
    return devs[p.device_of(l)].ring.data() + kind_index(l) * rb::ring_layer_bytes(d) + size_t(q % 16) * d.m_conv_dim * 2;
  }
  uint8_t* ssm_state(uint32_t l) {
    return devs[p.device_of(l)].ssm.data() + kind_index(l) * rb::ssm_layer_bytes(d, st);
  }
  uint8_t* kv_row(uint32_t l, uint32_t q, int t) {
    const uint32_t dev = p.device_of(l);
    return (t ? devs[dev].kv_v : devs[dev].kv_k).data() + kind_index(l) * rb::kv_layer_bytes(d, max_len_) +
           size_t(q) * d.kv_n() * 2;
  }
  // What the query at position q reads: every Mamba block's SSM state and its ring columns of q - 3 .. q - 1 (below 0: zero, as
  // nb_conv) and, with `self`, q; every attention block's K / V rows [0, q) and, with `self`, q.
  uint64_t attend(uint32_t q, bool self) {
    uint64_t h = 0x51ED;
    const size_t col_words = size_t(d.m_conv_dim) * 2 / 8, ssm_bytes = rb::ssm_layer_bytes(d, st);
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (d.is_mamba(l)) {
        const uint8_t* s = ssm_state(l);
        for (size_t i = 0; i < ssm_bytes; i += 8) {
          uint64_t a;
          std::memcpy(&a, s + i, std::min<size_t>(8, ssm_bytes - i));
          h = mix(h, a ^ (uint64_t(l) << 57) ^ uint64_t(i));
        }
        for (int back = 3; back >= (self ? 0 : 1); --back) {
          if (int64_t(q) - back < 0) {
            h = mix(h, 0x2E80 + uint64_t(l) * 8 + uint64_t(back));
            continue;
          }
          const uint8_t* r = ring_row(l, q - uint32_t(back));
          for (size_t w = 0; w < col_words; ++w) {
            uint64_t a;
            std::memcpy(&a, r + w * 8, 8);
            h = mix(h, a ^ (uint64_t(l) << 59) ^ uint64_t(back));
          }
        }
      } else if (d.is_attention(l)) {
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
    const size_t kvb = size_t(d.kv_n()) * 2, rb_bytes = size_t(d.m_conv_dim) * 2, sb = rb::ssm_layer_bytes(d, st);
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (d.is_attention(l)) {
        for (int t = 0; t < 2; ++t) {
          uint64_t h = mix(mix(mix(prev, id), uint64_t(pos_) * 131 + l), uint64_t(t));
          uint8_t* r = kv_row(l, pos_, t);
          for (size_t i = 0; i < kvb; i += 8) {
            h = mix(h, i);
            std::memcpy(r + i, &h, std::min<size_t>(8, kvb - i));
          }
        }
      } else if (d.is_mamba(l)) {
        uint64_t h = mix(mix(mix(prev, id), uint64_t(pos_) * 131 + l), 7);
        uint8_t* r = ring_row(l, pos_);
        for (size_t i = 0; i < rb_bytes; i += 8) {   // the raw xBC column of this position
          h = mix(h, i);
          std::memcpy(r + i, &h, std::min<size_t>(8, rb_bytes - i));
        }
        uint8_t* s = ssm_state(l);   // the recurrence: the new state depends on the old one, in place
        for (size_t i = 0; i < sb; ++i) s[i] = uint8_t(s[i] * 31 + (h >> (8 * (i % 8))) + id + i);
      }
    }
    ++pos_;
  }
  uint32_t next() { return poison ? poison_at : words[attend(pos_ - 1, true) % words.size()]; }

  model::BoltDesc d;
  BoltPlacement p;
  BoltState st;
  uint32_t max_len_;
  std::vector<Dev> devs;
  Ids words;
  uint32_t pos_ = 0, cur = 0;
  BlockHook hook;
  bool poison = false;
  uint32_t poison_at = 0;
  size_t state_loads = 0, kv_loaded = 0, logits_reads = 0;
};

using Adapter = cli::bolt::BoltEngineAdapterT<FakeBolt>;

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

// ---- 1. the layout on Bolt's real shapes ----------------------------------------------------------------------
void check_disjoint_inside(const model::BoltDesc& d, const BoltPlacement& p, uint32_t max_len, BoltState st,
                           const std::vector<SnapRun>& runs) {
  for (size_t i = 0; i < runs.size(); ++i) {
    const SnapRun& r = runs[i];
    const rb::PersistentSizes s = rb::persistent_sizes(d, p, r.device, max_len, st);
    const size_t cap = r.tensor == SnapTensor::KvK ? s.kv_k : r.tensor == SnapTensor::KvV ? s.kv_v
                       : r.tensor == SnapTensor::Ssm ? s.ssm : s.conv_ring;
    CHECK(r.bytes > 0);
    CHECK(r.offset + r.bytes <= cap);
    for (size_t q = 0; q < i; ++q)
      if (runs[q].device == r.device && runs[q].tensor == r.tensor)
        CHECK(runs[q].offset + runs[q].bytes <= r.offset || r.offset + r.bytes <= runs[q].offset);
  }
}

// The host order of a run list, placement-free: (tensor, the run's offset within its block's allocation, bytes, zero).
std::vector<std::vector<size_t>> host_order(const model::BoltDesc& d, uint32_t max_len, BoltState st,
                                            const std::vector<SnapRun>& runs) {
  std::vector<std::vector<size_t>> out;
  for (const SnapRun& r : runs) {
    const size_t per = r.tensor == SnapTensor::ConvRing ? rb::ring_layer_bytes(d)
                       : r.tensor == SnapTensor::Ssm    ? rb::ssm_layer_bytes(d, st)
                                                        : rb::kv_layer_bytes(d, max_len);
    out.push_back({size_t(r.tensor), r.offset % per, r.bytes, size_t(r.zero)});
  }
  return out;
}

void case_layout() {
  const model::BoltDesc& d = model::bolt();
  const uint32_t len = 262144;
  CHECK_EQ(d.mamba_layers(), 40u);
  CHECK_EQ(d.attn_layers(), 8u);
  CHECK_EQ(rb::kStateColumns, 3u);
  const size_t ssm_fp32 = 167772160, ssm_bf16 = 83886080, cols = 2457600;
  CHECK_EQ(rb::state_snapshot_bytes(d, BoltState::Fp32), ssm_fp32 + cols);
  CHECK_EQ(rb::state_snapshot_bytes(d, BoltState::Fp32), size_t(170229760));
  CHECK_EQ(rb::state_snapshot_bytes(d, BoltState::Bf16), ssm_bf16 + cols);
  CHECK_EQ(rb::kv_snapshot_bytes(d, 1), size_t(8192));
  CHECK_EQ(rb::kv_snapshot_bytes(d, 2048), size_t(16777216));
  const std::vector<BoltPlacement> places = {BoltPlacement::one(d), BoltPlacement::two(d, 3), BoltPlacement::two(d, 48),
                                             BoltPlacement::two(d, 87)};
  const size_t col = size_t(d.m_conv_dim) * 2;
  CHECK_EQ(col, size_t(20480));
  for (BoltState st : {BoltState::Fp32, BoltState::Bf16}) {
    const size_t ssm = st == BoltState::Fp32 ? size_t(4194304) : size_t(2097152);
    CHECK_EQ(rb::ssm_layer_bytes(d, st), ssm);
    for (uint32_t pos : {0u, 1u, 2u, 3u, 4u, 15u, 16u, 17u, 18u, 2047u, 2048u, 2049u, 4096u, 5000u, 262143u, 262144u}) {
      std::vector<std::vector<size_t>> first;
      for (const BoltPlacement& p : places) {
        const std::vector<SnapRun> runs = rb::state_runs(d, p, pos, st);
        CHECK_EQ(runs.size(), size_t(160));   // an SSM state and three columns a Mamba block
        size_t total = 0, zero = 0;
        uint32_t layer = 0;
        for (size_t i = 0; i < runs.size(); ++i) {
          const SnapRun& r = runs[i];
          total += r.bytes;
          zero += r.zero ? r.bytes : 0;
          const uint32_t j = uint32_t(i % 4);
          while (!d.is_mamba(layer)) ++layer;
          CHECK_EQ(r.device, p.device_of(layer));
          size_t idx = 0;   // the allocation index of Mamba block `layer` on its device, counted the long way
          for (uint32_t k = p.first(r.device); k < layer; ++k) idx += d.is_mamba(k) ? 1 : 0;
          if (j == 0) {   // run 4m is block m's SSM state
            CHECK(r.tensor == SnapTensor::Ssm);
            CHECK_EQ(r.bytes, ssm);
            CHECK_EQ(r.offset, idx * ssm);
            CHECK(!r.zero);
          } else {        // runs 4m + 1 .. 3 are its columns of positions pos - 1, pos - 2, pos - 3
            CHECK(r.tensor == SnapTensor::ConvRing);
            CHECK_EQ(r.bytes, col);
            const int64_t q = int64_t(pos) - int64_t(j);
            const uint32_t slot = uint32_t((q % 16 + 16) % 16);
            CHECK_EQ(r.offset % rb::ring_layer_bytes(d), size_t(slot) * col);
            CHECK_EQ(r.offset / rb::ring_layer_bytes(d), idx);
            CHECK_EQ(r.zero, q < 0);
            if (j == 3) ++layer;
          }
        }
        CHECK_EQ(total, rb::state_snapshot_bytes(d, st));
        CHECK_EQ(zero, size_t(pos == 0 ? 3 : pos == 1 ? 2 : pos == 2 ? 1 : 0) * 40 * col);   // below position 0: zeros
        check_disjoint_inside(d, p, len, st, runs);
        if (first.empty()) first = host_order(d, len, st, runs);
        else CHECK(host_order(d, len, st, runs) == first);   // 16b's rule: one host layout
      }
    }
  }
  for (const BoltPlacement& p : places) {
    CHECK(rb::kv_runs(d, p, len, 100, 100).empty());
    const std::vector<SnapRun> runs = rb::kv_runs(d, p, len, 2048, 4096);
    CHECK_EQ(runs.size(), size_t(16));   // K then V of 8 attention blocks
    size_t total = 0;
    for (size_t i = 0; i < runs.size(); ++i) {
      total += runs[i].bytes;
      CHECK(runs[i].tensor == (i < 8 ? SnapTensor::KvK : SnapTensor::KvV));
      CHECK_EQ(runs[i].bytes, size_t(2048) * 512);
    }
    CHECK_EQ(total, rb::kv_snapshot_bytes(d, 2048));
    check_disjoint_inside(d, p, len, BoltState::Fp32, runs);
    CHECK(host_order(d, len, BoltState::Fp32, runs) == host_order(d, len, BoltState::Fp32, rb::kv_runs(d, places[0], len, 2048, 4096)));
    for (const auto& range : {std::make_pair(10u, len + 1), std::make_pair(5u, 4u)}) {
      bool threw = false;
      try {
        (void)rb::kv_runs(d, p, len, range.first, range.second);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      CHECK(threw);
    }
  }
  // A placement of another model's block count is refused, not silently misaddressed.
  bool threw = false;
  try {
    (void)rb::state_runs(d, BoltPlacement::one(small_desc()), 0, BoltState::Fp32);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("layout: state %zu B = %zu B of fp32 SSM states + %zu B of conv columns (%zu + %zu with the bf16 arm) in 160 runs "
              "(an SSM state and the columns of pos - 1 .. pos - 3, slots mod 16, zeros below 0) at every pos, KV %zu B a position "
              "(16 runs), inside the allocations, one host layout for --pp 1 and --pp 2 at splits 3, 48 and 87\n",
              rb::state_snapshot_bytes(d, BoltState::Fp32), ssm_fp32, cols, ssm_bf16, cols, rb::kv_snapshot_bytes(d, 1));
}

// ---- 2. save / load on host buffers --------------------------------------------------------------------------
void scramble(FakeBolt& e, uint32_t seed) {
  std::mt19937_64 rng(seed);
  for (auto& v : e.devs)
    for (auto* m : {&v.kv_k, &v.kv_v, &v.ring, &v.ssm})
      for (auto& b : *m) b = uint8_t(rng());
}

void case_round_trip() {
  const model::BoltDesc d = small_desc();
  const size_t cb = size_t(d.m_conv_dim) * 2, kvb = size_t(d.kv_n()) * 2;
  for (BoltState st : {BoltState::Fp32, BoltState::Bf16}) {
    const size_t sb = rb::ssm_layer_bytes(d, st);
    CHECK_EQ(sb, st == BoltState::Fp32 ? size_t(128) : size_t(64));
    for (const BoltPlacement& p : {BoltPlacement::one(d), BoltPlacement::two(d, 3)}) {
      for (uint32_t at : {0u, 1u, 2u, 3u, 15u, 16u, 17u, 300u, 2048u, 4097u}) {
        FakeBolt e(d, p, 8192, some_words(), st);
        scramble(e, at);
        e.pos_ = at;
        std::vector<uint8_t> state(e.state_bytes());
        e.save_state(state.data());
        // The layout: per Mamba block, its SSM state, then the columns of positions at - 1, at - 2, at - 3; zeros below 0.
        size_t k = 0;
        for (uint32_t l = 0; l < d.layers; ++l) {
          if (!d.is_mamba(l)) continue;
          CHECK(std::equal(state.begin() + k, state.begin() + k + sb, e.ssm_state(l)));
          k += sb;
          for (uint32_t j = 1; j <= 3; ++j) {
            const int64_t q = int64_t(at) - int64_t(j);
            if (q < 0) {
              for (size_t i = 0; i < cb; ++i) CHECK_EQ(state[k + i], uint8_t(0));
            } else {
              CHECK(std::equal(state.begin() + k, state.begin() + k + cb, e.ring_row(l, uint32_t(q))));
            }
            k += cb;
          }
        }
        CHECK_EQ(k, state.size());
        FakeBolt f(d, p, 8192, some_words(), st);
        scramble(f, at + 1);
        const FakeBolt before = f;
        f.load_state(state.data(), at);
        CHECK_EQ(f.pos(), at);
        for (uint32_t l = 0; l < d.layers; ++l) {
          if (!d.is_mamba(l)) continue;
          CHECK(std::equal(e.ssm_state(l), e.ssm_state(l) + sb, f.ssm_state(l)));
          for (uint32_t j = 1; j <= 3; ++j) {
            const int64_t q = int64_t(at) - int64_t(j);
            if (q >= 0) CHECK(std::equal(e.ring_row(l, uint32_t(q)), e.ring_row(l, uint32_t(q)) + cb, f.ring_row(l, uint32_t(q))));
          }
        }
        // The KV and every ring slot other than the three (a position below 0 included) are untouched; so is every state byte
        // the runs do not name (none: the states are whole).
        for (uint32_t dv = 0; dv < p.devices; ++dv) {
          CHECK(f.devs[dv].kv_k == before.devs[dv].kv_k && f.devs[dv].kv_v == before.devs[dv].kv_v);
          std::vector<bool> in(f.devs[dv].ring.size(), false);
          for (const SnapRun& r : rb::state_runs(d, p, at, st))
            if (r.device == dv && r.tensor == SnapTensor::ConvRing)
              for (size_t i = 0; i < r.bytes; ++i) in[r.offset + i] = true;
          for (size_t i = 0; i < in.size(); ++i)
            if (!in[i]) CHECK_EQ(f.devs[dv].ring[i], before.devs[dv].ring[i]);
        }
        // KV: [17, at) restored bit for bit, the rest untouched; adjacent saves compose. (at <= 17: nothing to copy.)
        if (at > 17) {
          const uint32_t mid = 17 + (100 > at - 17 ? 1 : 100);
          std::vector<uint8_t> kv(e.kv_bytes(at - 17)), a(e.kv_bytes(mid - 17)), b(e.kv_bytes(at - mid));
          e.save_kv(17, at, kv.data());
          e.save_kv(17, mid, a.data());
          e.save_kv(mid, at, b.data());
          FakeBolt x(d, p, 8192, some_words(), st), y(d, p, 8192, some_words(), st);
          x.load_kv(17, mid, a.data());
          x.load_kv(mid, at, b.data());
          y.load_kv(17, at, kv.data());
          for (uint32_t dv = 0; dv < p.devices; ++dv)
            CHECK(x.devs[dv].kv_k == y.devs[dv].kv_k && x.devs[dv].kv_v == y.devs[dv].kv_v);
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
  }
  std::printf("round trip: the SSM states and three columns a Mamba block (zeros below 0) and KV ranges restored bit for bit, "
              "nothing else touched, on one card and two, fp32 and bf16 states\n");
}

// ---- 3. Review Focus 4 ---------------------------------------------------------------------------------------
Ids cold_tail(FakeBolt& e, const Ids& tail, uint32_t n) {
  e.prefill(tail);
  return e.generate(n);
}

// Everything a session's future reads agrees: every attention row [0, pos), the ring columns of pos - 3 .. pos - 1, the states.
void check_same_memory(FakeBolt& a, FakeBolt& b) {
  CHECK_EQ(a.pos(), b.pos());
  const model::BoltDesc& d = a.d;
  const size_t cb = size_t(d.m_conv_dim) * 2, kvb = size_t(d.kv_n()) * 2;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.is_attention(l)) {
      for (uint32_t q = 0; q < a.pos(); ++q)
        for (int t = 0; t < 2; ++t) CHECK(std::equal(a.kv_row(l, q, t), a.kv_row(l, q, t) + kvb, b.kv_row(l, q, t)));
    } else if (d.is_mamba(l)) {
      CHECK(std::equal(a.ssm_state(l), a.ssm_state(l) + rb::ssm_layer_bytes(d, a.st), b.ssm_state(l)));
      for (uint32_t back = 1; back <= 3; ++back)
        if (a.pos() >= back)
          CHECK(std::equal(a.ring_row(l, a.pos() - back), a.ring_row(l, a.pos() - back) + cb, b.ring_row(l, a.pos() - back)));
    }
  }
}

void case_tiny_positions() {
  // A restore at positions 0, 1, 2, 3 (where a column is a zero column), 4, 15, 16, 17, 18 (the ring's wrap) continues bitwise as
  // the cold session: the ring columns the state names, every KV row, every SSM state and the generated ids.
  const model::BoltDesc d = small_desc();
  Ids all;
  for (uint32_t i = 0; i < 60; ++i) all.push_back(3000 + (i * 7919) % 5000);
  for (BoltState st : {BoltState::Fp32, BoltState::Bf16}) {
    for (const BoltPlacement& p : {BoltPlacement::one(d), BoltPlacement::two(d, 5)}) {
      for (uint32_t at : {0u, 1u, 2u, 3u, 4u, 15u, 16u, 17u, 18u}) {
        const Ids head(all.begin(), all.begin() + at), tail(all.begin() + at, all.end());
        FakeBolt a(d, p, 1024, some_words(), st);
        if (at > 0) a.prefill(head);
        CHECK_EQ(a.pos(), at);
        std::vector<uint8_t> state(a.state_bytes()), kv(a.kv_bytes(at));
        a.save_state(state.data());
        a.save_kv(0, at, kv.data());
        FakeBolt b(d, p, 1024, some_words(), st);
        scramble(b, 40 + at);   // another session's leftovers everywhere
        b.load_kv(0, at, kv.data());
        b.load_state(state.data(), at);
        const Ids ga = cold_tail(a, tail, 16), gb = cold_tail(b, tail, 16);
        CHECK(ga == gb);
        check_same_memory(a, b);
      }
    }
  }
  std::printf("restores at positions 0, 1, 2, 3, 4, 15, 16, 17, 18 continue bitwise as the cold session (KV, rings, states, ids), "
              "on one card and two, fp32 and bf16 states\n");
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

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

void case_prefix() {
  const model::BoltDesc d = small_desc();
  Ids L;
  for (uint32_t i = 0; i < 6600; ++i) L.push_back(2000 + (i * 7919) % 50000);
  const Ids C(L.rbegin(), L.rbegin() + 60);
  using K = server::PrefixCache::Plan;
  struct Want {
    const char* name;
    uint32_t restart;
    uint32_t first;   // turn 1's prompt length (it generates 8)
  };
  MallocAlloc alloc;
  size_t checked = 0;
  for (const Want& w : {Want{"block end 2048", 2048, 2100}, Want{"block end 4096", 4096, 4200}, Want{"block end 6144", 6144, 6200},
                        Want{"request end 2049", 2049, 2041}, Want{"request end 5000", 5000, 4992},
                        Want{"request end 300", 300, 292}}) {
    for (const BoltPlacement& p : {BoltPlacement::one(d), BoltPlacement::two(d, 3)}) {
      FakeBolt eng(d, p, 8192, some_words());
      Adapter ad(eng, kVocabUsed);
      CHECK_EQ(ad.block(), 2048u);
      CHECK_EQ(ad.state_bytes(), rb::state_snapshot_bytes(d, BoltState::Fp32));
      server::PrefixSession s(ad, size_t(1) << 30, &alloc);
      CHECK_EQ(s.cache()->kv_form(), cli::bolt::kBoltStoreKey);
      const Ids t1 = slice(L, 0, w.first);
      const Ids g1 = turn(s, ad, t1, 8);
      (void)turn(s, ad, slice(L, 6500, 6560), 4);   // a side request: the next turn must restore
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
      // The cold run of the same prompt on one card: the ids AND the whole memory (every KV row, every state) agree.
      FakeBolt cold_eng(d, BoltPlacement::one(d), 8192, some_words());
      cold_eng.prefill(final_prompt);
      const Ids want = cold_eng.generate(16);
      CHECK(got == want);   // a restore continues as the cold run
      check_same_memory(eng, cold_eng);
      ++checked;
      std::printf("  %s, %s: restore at %u (%zu B of KV), 16 ids and the whole memory == cold\n", w.name,
                  p.devices == 2 ? "--pp 2" : "--pp 1", rep.restart, rep.kv_bytes);
    }
  }
  CHECK_EQ(alloc.live, size_t(0));
  // The bf16 arm's snapshots are another store root: a chain written under one arm can never be restored under the other.
  FakeBolt b16(d, BoltPlacement::one(d), 8192, some_words(), BoltState::Bf16);
  Adapter ab(b16, kVocabUsed);
  FakeBolt f32(d, BoltPlacement::one(d), 8192, some_words(), BoltState::Fp32);
  Adapter af(f32, kVocabUsed);
  CHECK(ab.kv_form() != af.kv_form());
  CHECK_EQ(af.kv_form(), cli::bolt::kBoltStoreKey);
  CHECK_EQ(cli::bolt::kBoltStoreKey, 0x4E454D4F424F4C54ull);   // "NEMOBOLT"
  CHECK(ab.state_bytes() < af.state_bytes());
  std::printf("prefix: %zu restores (block ends 2048 / 4096 / 6144, request ends 2049 / 5000 / 300; one card and two) equal the cold "
              "run; the store key is \"NEMOBOLT\", the bf16 arm's another\n", checked);
}

// ---- 4. 16b's rule: one host layout for every split --------------------------------------------------------
void case_pp() {
  const model::BoltDesc d = small_desc();
  Ids prompt;
  for (uint32_t i = 0; i < 2600; ++i) prompt.push_back(3000 + (i * 104729) % 40000);
  const std::vector<BoltPlacement> places = {BoltPlacement::two(d, 3), BoltPlacement::one(d), BoltPlacement::two(d, 5),
                                             BoltPlacement::two(d, 1), BoltPlacement::two(d, 7)};
  size_t moves = 0;
  for (const BoltPlacement& from : places) {
    for (const BoltPlacement& to : places) {
      FakeBolt a(d, from, 8192, some_words());
      a.prefill(prompt);
      (void)a.generate(5);
      const uint32_t at = a.pos();
      std::vector<uint8_t> state(a.state_bytes()), kv(a.kv_bytes(at));
      a.save_state(state.data());
      a.save_kv(0, at, kv.data());
      FakeBolt b(d, to, 8192, some_words());
      CHECK_EQ(b.state_bytes(), a.state_bytes());
      CHECK_EQ(b.kv_bytes(at), a.kv_bytes(at));
      b.load_kv(0, at, kv.data());
      b.load_state(state.data(), at);
      b.set_token(a.pending());
      CHECK(b.generate(24) == a.generate(24));
      check_same_memory(a, b);
      ++moves;
    }
  }
  std::printf("pp: a snapshot taken under any of --pp 2 splits 1 / 3 / 5 / 7 and --pp 1 continues bitwise under every other (%zu "
              "moves)\n", moves);
}

// ---- 5. the server -------------------------------------------------------------------------------------------
struct BoltTemplate : server::TemplateIface {
  std::string render(const json& messages, const json&, bool) override { return body(messages); }
  std::string render_with_kwargs(const json& messages, const json&, bool, const json&) override { return body(messages); }
  static std::string body(const json& messages) {
    std::string text = "<|im_start|> system <|im_end|> ";   // the real template always opens the empty system block
    for (const auto& m : messages)
      text += "<|im_start|> " + m.at("role").get<std::string>() + " " + m.at("content").get<std::string>() + " <|im_end|> ";
    return text + "<|im_start|> assistant <think>";
  }
};

struct ServeFixture {
  explicit ServeFixture(bool cache)
      : eng(small_desc(), BoltPlacement::two(small_desc(), 3), 8192, words()),
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
    o.served_model = "bolt";
    const cli::bolt::Generation g = cli::bolt::published_generation();
    o.eos_ids = g.eos;
    o.eos_ids.push_back(248046);   // the mock tokenizer's <|im_end|>
    o.chat_format = server::ChatFormat::bolt();
    o.sampling_defaults = g.sampling;
    if (cache) {
      o.prefix_cache_bytes = size_t(256) << 20;
      o.prefix_alloc = &alloc;
    }
    return o;
  }
  json chat_text(const std::string& user, const json& extra) {
    json b = {{"model", "bolt"},
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
  BoltTemplate tmpl;
  MallocAlloc alloc;
  FakeBolt eng;
  Adapter adapter;
  server::Server server;
};

void case_server() {
  ServeFixture f(/*cache=*/true);
  const json greedy = {{"temperature", 0}};
  const json r1 = f.chat(greedy);
  const Ids prompt = r1.at("prompt_token_ids").get<Ids>();
  const Ids served = f.out_ids(r1);
  CHECK(prompt.size() > 2048);
  CHECK_EQ(served.size(), size_t(24));
  // b70-decode --prefill's run of the same ids
  FakeBolt decode(small_desc(), BoltPlacement::one(small_desc()), 8192, f.eng.words);
  decode.prefill(prompt);
  CHECK(decode.generate(24) == served);
  // The same request again: restored from the block-end snapshot (SSM states + conv columns + KV), the same ids.
  const json r2 = f.chat(greedy);
  CHECK(r2.at("prompt_token_ids").get<Ids>() == prompt);
  CHECK(f.out_ids(r2) == served);
  const uint32_t cached = r2.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
  CHECK_EQ(cached, 2048u);   // the block-end snapshot: the prompt is past 2048, its own end cannot restore
  CHECK(f.eng.state_loads > 0);
  std::printf("server: greedy chat of %zu prompt ids -> 24 ids == the decode run; again: restored at %u, same ids\n",
              prompt.size(), cached);

  // Sampled (a flat row: T 8, so the draws matter), seeded: reproducible on a fresh server and on the same one, the first id
  // drawn from the prefill's row too (n reads for n ids), only words the row allows.
  const json sampled = {{"seed", 1234}, {"temperature", 8.0}, {"top_k", 24}, {"top_p", 1.0}};
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
    // The same seeded request again on the SAME server draws the same ids (the RNG is re-seeded per request); another seed and
    // an unseeded request differ.
    CHECK(g.out_ids(g.chat(sampled)) == b);
    json other = sampled;
    other["seed"] = 99;
    CHECK(g.out_ids(g.chat(other)) != b);
    json unseeded = sampled;
    unseeded.erase("seed");
    CHECK(g.out_ids(g.chat(unseeded)) != b);
  }
  // An unseeded request after a seeded one is not that seed's continuation: two fresh servers that each serve the seeded request
  // and then the same unseeded one draw different ids (each unseeded request re-seeds from the random device).
  Ids u1, u2;
  json unseeded_req = sampled;
  unseeded_req.erase("seed");
  {
    ServeFixture g(/*cache=*/false);
    (void)g.out_ids(g.chat(sampled));
    u1 = g.out_ids(g.chat(unseeded_req));
  }
  {
    ServeFixture g(/*cache=*/false);
    (void)g.out_ids(g.chat(sampled));
    u2 = g.out_ids(g.chat(unseeded_req));
  }
  CHECK(u1.size() >= 8 && u1.size() == u2.size());
  CHECK(u1 != u2);
  CHECK(a == b);
  CHECK_EQ(reads, a.size());
  for (uint32_t id : a) CHECK(std::find(f.eng.words.begin(), f.eng.words.end(), id) != f.eng.words.end());
  std::printf("server: a seeded request (T 8) reproducible on a fresh server and on the same one (%zu ids, %zu rows read), every id a "
              "vocabulary word\n", a.size(), reads);

  // The head's rows past what the tokenizer defines: the adapter built with a smaller vocab_used masks them - greedy takes the
  // masked row's argmax, sampled never draws them.
  {
    ServeFixture g(/*cache=*/false);
    Adapter masked(g.eng, g.eng.vocab() - 2);
    g.eng.poison = true;
    g.eng.poison_at = g.eng.vocab() - 2;
    server::Sampling gr;
    g.eng.prefill(Ids{1010, 1011, 1012});
    const uint32_t first = masked.step(gr);
    CHECK(first < g.eng.vocab() - 2);   // the pending id was an undefined row: replaced by the masked argmax
    CHECK_EQ(masked.masked(), uint64_t(1));
    server::Sampling s;
    s.greedy = false;
    s.has_seed = true;
    s.seed = 5;
    s.top_k = 0;
    for (int i = 0; i < 20; ++i) CHECK(masked.step(s) < g.eng.vocab() - 2);
  }
  std::printf("server: no id the tokenizer does not define is ever emitted (greedy masked, sampled filtered)\n");
}

}  // namespace

int main() {
  case_layout();
  case_round_trip();
  case_tiny_positions();
  case_prefix();
  case_pp();
  case_server();
  std::printf("bolt_snapshot_test OK\n");
  return 0;
}

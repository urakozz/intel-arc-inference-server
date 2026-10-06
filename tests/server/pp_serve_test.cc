// Spec 16d (b70-serve --pp 2) without a device: cli::pp::PipelineEngineAdapterT - the adapter
// b70-serve builds over runtime::PipelineEngine - over FakeEngine, a device-free engine with
// PipelineEngine's interface and its two-device structure:
//
//   - the model's layers split over "devices" (12 layers, FA at every fourth as ModelDesc; one
//     device holds them all, two hold [0, 5) and [5, 12)); each device keeps its own layers'
//     GDN state in four slots (slot 0 live, the MTP verify slots), its conv ring and its KV
//     cache in a real runtime::KvLayout (bf16 rows, or int8 rows + fp16 scales), and its own
//     Control block - device 1's is the truth, device 0's mirrors it after every step (checked
//     equal after every call);
//   - a row is the embedding, every layer in order (device 0's, the hand-off of the residual,
//     device 1's), each reading what it wrote at earlier positions (the KV rows, the ring, the
//     state), and the head; the next id is the argmax of a logits row over a word list;
//   - MTP as Engine's contract: verify(k) runs k + 1 rows writing row m's state into slot
//     (live + m) % 4, the head's KV rows at pos - 1 .. pos + k - 1 and the hidden rows;
//     commit(j) moves live and pos on both devices; draft(k) chains the head over its own KV
//     (right most of the time, by construction);
//   - spec 7's snapshots in Engine's layouts: the live slot of every GDN layer (device 0's
//     then device 1's), the conv rings, the head's hidden row; the KV runs of
//     runtime::pp_kv_runs (two devices) or Engine's kv_runs restated (one device) - so a
//     snapshot taken on one "card" must restore on two and vice versa.
//
//   1. one device and two: prefill + 48 ids, ids and every snapshot byte equal;
//   2. snapshots move: one device's snapshot at a block end restored on two (and back), the
//      continuation equal to the uninterrupted run, both KV forms, with and without the head;
//   3. spec 8 M2 on the fake: draft / verify / commit with K = 1, 2, 3 and a varying K give the
//      plain greedy ids, and the session (state, live slot, hidden row, KV with the head's
//      layer) equal to the plain run's;
//   4. spec 7's request path (server::PrefixSession over the adapter, block 16): restore at the
//      prompt end / mid-block / on a block / after a side request / the same prompt - every
//      cached turn's ids equal to a cold one-device run, without and with MTP (K = 2);
//   5. the server (HTTP, the Qwen chat format): a greedy chat = the one-device decode run on
//      the response's prompt_token_ids; again from the cache (cached_tokens > 0) with the same
//      ids; --mtp 2 and --spec lookup give the same ids; --prefix-split-last; a seeded sampled
//      request reproducible on a fresh server (plain and MTP), every id a word.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/pipeline_serve_adapter.h"
#include "model/qwen35.h"
#include "runtime/buffer_sizes.h"
#include "runtime/pipeline_plan.h"
#include "runtime/prefill_chunks.h"
#include "server/mock.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using Ids = std::vector<uint32_t>;
using runtime::KvCache;

constexpr uint32_t kLayers = 12, kSplit = 5;   // FA at 3, 7, 11 (ModelDesc::is_fa)
constexpr uint32_t kRing = 16;
constexpr uint32_t kV = model::Qwen35::kVocab;

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}
bool is_fa(uint32_t l) { return l % 4 == 3; }

struct Ctl {
  uint32_t pos = 0, n_active = 0, cur[4] = {}, out[4] = {}, live = 0;
  bool operator==(const Ctl& o) const { return std::memcmp(this, &o, sizeof(Ctl)) == 0; }
};

// One device's layers and state.
struct Dev {
  uint32_t first = 0, last = 0;
  std::vector<uint32_t> gdn, fa;   // the model's layer numbers, in order
  std::array<std::vector<uint64_t>, 4> slot;   // [slot][local GDN layer]
  std::vector<uint64_t> conv;                  // [local GDN layer][kRing]
  runtime::KvLayout lay;
  std::vector<uint8_t> k, v;
  Ctl ctl;
};

// PipelineEngine's interface (what PipelineEngineAdapterT calls), device-free.
struct FakeEngine {
  static constexpr uint32_t kBlock = 16;   // PipelineEngine::kBlock is 2048; small for the test
  static constexpr uint32_t kChunk = 24;
  static constexpr uint32_t kMaxDraft = 3;
  using BlockHook = std::function<void(uint32_t, bool)>;

  FakeEngine(uint32_t devices, uint32_t max_len, KvCache form, bool with_mtp, Ids vocab_words)
      : words(std::move(vocab_words)), mtp_on(with_mtp) {
    const uint32_t cut[3] = {0, devices == 1 ? kLayers : kSplit, kLayers};
    for (uint32_t d = 0; d < devices; ++d) {
      Dev dv;
      dv.first = cut[d];
      dv.last = cut[d + 1];
      for (uint32_t l = dv.first; l < dv.last; ++l) (is_fa(l) ? dv.fa : dv.gdn).push_back(l);
      dv.lay = layout(max_len, form, uint32_t(dv.fa.size()));
      dev.push_back(std::move(dv));
    }
    head_lay = layout(max_len, form, 1);
    reset();
  }
  static runtime::KvLayout layout(uint32_t max_len, KvCache form, uint32_t layers) {
    runtime::KvLayout L;
    L.form = form;
    L.max_len = max_len;
    L.kv_heads = 1;
    L.head_dim = 8;
    L.layers = layers;
    return L;
  }

  // --- the session ----------------------------------------------------------------------
  void reset() {
    for (Dev& d : dev) {
      for (auto& s : d.slot) s.assign(d.gdn.size(), 0);
      d.conv.assign(d.gdn.size() * kRing, 0);
      d.k.assign(d.lay.bytes(), 0);
      d.v.assign(d.lay.bytes(), 0);
      d.ctl = Ctl{};
    }
    hk.assign(head_lay.bytes(), 0);
    hv.assign(head_lay.bytes(), 0);
    hh.assign(5, 0);
    verify_pos = 0;
    verify_k = kNone;
    drafts.clear();
    ++resets;
  }
  Ctl& c() { return dev.back().ctl; }
  const Ctl& c() const { return dev.back().ctl; }
  // Review Focus 3 on the fake: every device's Control block is device 1's at a boundary.
  void mirror() {
    for (Dev& d : dev) d.ctl = c();
  }
  void check_in_step() const {
    for (const Dev& d : dev) CHECK(d.ctl == c());
  }

  // One row at `pos` with id `x`: GDN state from slot `in` into slot `out`. Returns the
  // final-normed hidden; writes the layers' KV rows and ring entries at `pos`.
  uint64_t row(uint32_t pos, uint32_t x, uint32_t in, uint32_t out) {
    CHECK(pos < head_lay.max_len);
    uint64_t h = mix(0xE11Bu, x);   // the embedding
    for (Dev& d : dev) {            // device 0's layers, the hand-off of h, device 1's
      uint32_t g = 0, f = 0;
      for (uint32_t l = d.first; l < d.last; ++l) {
        if (!is_fa(l)) {
          const uint64_t s = mix(d.slot[in][g], h ^ l);
          d.slot[out][g] = s;
          const uint64_t prev = d.conv[g * kRing + (pos + kRing - 1) % kRing];
          d.conv[g * kRing + pos % kRing] = h;
          h = mix(mix(h, s), prev);
          ++g;
        } else {
          write_kv(d.lay, d.k, d.v, f, pos, mix(h, l));
          h = mix(h, read_kv(d.lay, d.k, d.v, f, pos));
          ++f;
        }
      }
    }
    return mix(h, 0xF1Fu);   // the final norm
  }
  static void write_kv(const runtime::KvLayout& L, std::vector<uint8_t>& k, std::vector<uint8_t>& v,
                       uint32_t layer, uint32_t pos, uint64_t h) {
    for (int t = 0; t < 2; ++t) {
      std::vector<uint8_t>& m = t == 0 ? k : v;
      uint64_t r = mix(h, uint64_t(t) + 7);
      for (size_t i = 0; i < L.row_bytes(); ++i) m[L.rows_offset(layer) + pos * L.row_bytes() + i] = uint8_t(mix(r, i));
      for (size_t i = 0; i < L.scale_row_bytes(); ++i)
        m[L.scales_offset(layer) + pos * L.scale_row_bytes() + i] = uint8_t(mix(r, i + 99));
    }
  }
  // "Attention": rows 0, pos / 2, pos - 1 and pos of K and V (and their scales).
  static uint64_t read_kv(const runtime::KvLayout& L, const std::vector<uint8_t>& k,
                          const std::vector<uint8_t>& v, uint32_t layer, uint32_t pos) {
    uint64_t h = 0x5EED;
    for (uint32_t q : {0u, pos / 2, pos == 0 ? 0u : pos - 1, pos})
      for (const std::vector<uint8_t>* m : {&k, &v}) {
        for (size_t i = 0; i < L.row_bytes(); ++i) h = mix(h, (*m)[L.rows_offset(layer) + q * L.row_bytes() + i]);
        for (size_t i = 0; i < L.scale_row_bytes(); ++i)
          h = mix(h, (*m)[L.scales_offset(layer) + q * L.scale_row_bytes() + i]);
      }
    return h;
  }
  // The logits row of a hidden: -1e30 outside the words; the argmax is the next id.
  float score(uint64_t h, uint32_t w) const { return float(mix(h, w) % 1000) / 250.0f; }
  uint32_t argmax(uint64_t h) const {
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
  void logits_of(uint64_t h, float* row) const {
    std::fill(row, row + kV, -1e30f);
    for (uint32_t w : words) row[w] = score(h, w);
  }
  // The head's KV row at q from (h_q, x_{q+1}).
  void head_fill(uint32_t q, uint64_t hq, uint32_t x) { write_kv(head_lay, hk, hv, 0, q, mix(hq, x)); }

  void normalise_live() {
    const uint32_t live = c().live;
    if (live != 0)
      for (Dev& d : dev) d.slot[0] = d.slot[live];
    c().live = 0;
    mirror();
  }

  // A plain step at pos with the pending id (Engine::replay / PipelineEngine::step_once).
  void plain() {
    const uint32_t p = c().pos, x = c().cur[0];
    CHECK(c().live == 0 || mtp_on);
    const uint64_t h = row(p, x, c().live, c().live);
    last_h[0] = h;
    c().out[0] = argmax(h);
    c().cur[0] = c().out[0];
    c().pos = p + 1;
    mirror();
    if (mtp_on) hh[0] = h;   // pos 0 only (Engine::mtp_step1)
  }
  void step1() {   // one token: plain, or verify(0) + commit(0) with MTP on
    if (!mtp_on) return plain();
    if (c().pos == 0) {
      if (c().live != 0) normalise_live();
      return plain();
    }
    verify(0);
    commit(0, c().out[0]);
  }

  void prefill(const Ids& ids) {
    CHECK(!ids.empty());
    if (mtp_on) {
      normalise_live();
      verify_k = kNone;
    }
    const bool hooked = static_cast<bool>(hook);
    const uint32_t base = c().pos;
    uint32_t C = 0;
    for (size_t off = 0; off < ids.size(); off += C) {
      C = runtime::prefill_chunk_rows(base + uint32_t(off), ids.size() - off, kChunk, hooked, kBlock);
      for (uint32_t i = 0; i < C; ++i) {
        const uint32_t p = base + uint32_t(off) + i, x = ids[off + i];
        const uint64_t h = row(p, x, 0, 0);
        if (mtp_on) {
          if (p > 0) head_fill(p - 1, hh[0], x);
          hh[0] = h;
        }
        last_h[0] = h;
      }
      const uint32_t end = base + uint32_t(off) + C;
      c().pos = end;
      c().n_active = 0;
      mirror();
      if (hooked && off + C < ids.size() && end % kBlock == 0) hook(end, true);
    }
    c().out[0] = argmax(last_h[0]);
    c().cur[0] = c().out[0];
    c().n_active = 1;
    mirror();
    prefilled += ids.size();
    if (hooked) hook(c().pos, c().pos % kBlock == 0);
  }
  void ingest(const Ids& ids) {
    for (uint32_t id : ids) {
      set_token(id);
      step1();
    }
  }
  Ids generate(uint32_t n) {
    Ids out;
    for (uint32_t i = 0; i < n; ++i) {
      out.push_back(c().cur[0]);
      step1();
    }
    check_in_step();
    return out;
  }
  uint32_t pos() const { return c().pos; }
  uint32_t max_len() const { return head_lay.max_len; }
  KvCache kv_cache() const { return head_lay.form; }
  uint32_t pending() const { return c().cur[0]; }
  void set_token(uint32_t id) {
    CHECK(id < kV);
    for (Dev& d : dev) d.ctl.cur[0] = id;
  }

  // --- MTP: Engine's contract ----------------------------------------------------------------
  bool mtp() const { return mtp_on; }
  uint32_t max_verify_k() const {
    const uint32_t p = c().pos;
    if (!mtp_on || p == 0 || p + 1 >= max_len()) return 0;
    return std::min<uint32_t>(kMaxDraft, max_len() - p - 1);
  }
  // The true continuation (what plain steps would give), from a copy: the drafts are right
  // three times in four.
  Ids truth(uint32_t k) const {
    FakeEngine copy = *this;
    copy.hook = {};
    Ids t;
    copy.step1();
    for (uint32_t i = 0; i < k; ++i) {
      t.push_back(copy.c().cur[0]);
      copy.step1();
    }
    return t;
  }
  void draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick = {}) {
    CHECK(mtp_on && k >= 1 && k <= max_verify_k());
    const Ids t = truth(k);
    uint64_t dh = hh[0];
    uint32_t in = c().cur[0];
    const uint32_t p = c().pos;
    drafts.clear();
    for (uint32_t i = 0; i < k; ++i) {
      head_fill(p - 1 + i, dh, in);   // the draft writes the head's row at p - 1 + i
      dh = mix(mix(dh, in), read_kv(head_lay, hk, hv, 0, p - 1 + i));
      uint32_t d = t[i];
      if (mix(dh, 3) % 4 == 0) d = words[(std::find(words.begin(), words.end(), d) - words.begin() + 1) % words.size()];
      q_h[i] = dh;
      q_top[i] = d;
      if (pick) d = pick(i);
      CHECK(d < kV);
      drafts.push_back(d);
      for (Dev& dv : dev) dv.ctl.cur[1 + i] = d;
      in = d;
    }
  }
  const Ids& draft_ids() const { return drafts; }
  void set_draft_input(uint32_t i, uint32_t id) {
    CHECK(i < kMaxDraft && id < kV);
    for (Dev& d : dev) d.ctl.cur[1 + i] = id;
  }
  void verify(uint32_t k) {
    CHECK(mtp_on);
    const uint32_t p = c().pos;
    CHECK(k <= kMaxDraft && p >= 1 && p + k + 1 <= max_len());
    for (uint32_t r = 0; r <= k; ++r) CHECK(dev.front().ctl.cur[r] == c().cur[r]);   // both devices
    const uint32_t live = c().live;
    for (uint32_t m = 0; m <= k; ++m) {
      const uint32_t in = m == 0 ? live : (live + m - 1) % 4, out = (live + m) % 4;
      const uint64_t h = row(p + m, c().cur[m], in, out);
      last_h[m] = h;
      hh[1 + m] = h;
      c().out[m] = argmax(h);
    }
    for (uint32_t r = 0; r <= k; ++r) head_fill(p - 1 + r, hh[r], c().cur[r]);   // hh[0] = h_{p-1}
    c().pos = p + k + 1;
    c().n_active = k + 1;
    mirror();
    verify_pos = p;
    verify_k = k;
  }
  const uint32_t* verify_ids() const { return c().out; }
  void commit(uint32_t j, uint32_t next) {
    CHECK(verify_k != kNone && j <= verify_k && next < kV);
    hh[0] = hh[1 + j];
    c().live = (c().live + j) % 4;
    c().pos = verify_pos + j + 1;
    c().n_active = 1;
    c().cur[0] = next;
    mirror();
    verify_k = kNone;
  }

  // --- the host reads ------------------------------------------------------------------------
  float* host_rows(size_t floats) {
    rows.assign(floats, 0.0f);
    return rows.data();
  }
  void read_logits_into(float* host, uint32_t n) const {
    for (uint32_t r = 0; r < n; ++r) logits_of(last_h[r], host + size_t(r) * kV);
  }
  void read_draft_logits_into(float* host, uint32_t i) const {
    logits_of(q_h[i], host);
    float top = -1e30f;
    for (uint32_t w : words) top = std::max(top, host[w]);
    host[q_top[i]] = top + 1.0f;   // the head's own argmax is its draft
  }

  // --- spec 7, Engine's layouts ------------------------------------------------------------
  size_t n_gdn() const {
    size_t n = 0;
    for (const Dev& d : dev) n += d.gdn.size();
    return n;
  }
  size_t state_bytes() const { return n_gdn() * 8 * (1 + kRing) + (mtp_on ? 8 : 0); }
  size_t kv_bytes(uint32_t n) const {
    return 2 * head_lay.pos_bytes() * (3 + (mtp_on ? 1 : 0)) * size_t(n);
  }
  void save_state(void* host) const {
    auto* h = static_cast<uint8_t*>(host);
    for (const Dev& d : dev) {   // the live slot of every GDN layer, device 0's first
      std::memcpy(h, d.slot[c().live].data(), d.gdn.size() * 8);
      h += d.gdn.size() * 8;
    }
    for (const Dev& d : dev) {
      std::memcpy(h, d.conv.data(), d.conv.size() * 8);
      h += d.conv.size() * 8;
    }
    if (mtp_on) std::memcpy(h, &hh[0], 8);
  }
  void load_state(const void* host, uint32_t at) {
    CHECK(at <= max_len());
    const auto* h = static_cast<const uint8_t*>(host);
    for (Dev& d : dev) {
      std::memcpy(d.slot[0].data(), h, d.gdn.size() * 8);
      h += d.gdn.size() * 8;
    }
    for (Dev& d : dev) {
      std::memcpy(d.conv.data(), h, d.conv.size() * 8);
      h += d.conv.size() * 8;
    }
    if (mtp_on) std::memcpy(&hh[0], h, 8);
    c().live = 0;
    c().pos = at;
    c().n_active = 0;
    verify_k = kNone;
    mirror();
    ++state_loads;
  }
  struct Run {
    uint8_t* p;
    size_t n;
  };
  std::vector<Run> runs(uint32_t b, uint32_t e) {
    std::vector<Run> out;
    if (dev.size() == 2) {   // runtime::pp_kv_runs: what PipelineEngine's save_kv / load_kv walk
      for (const runtime::PpKvRun& r : runtime::pp_kv_runs({dev[0].lay, dev[1].lay}, mtp_on ? &head_lay : nullptr, b, e)) {
        std::vector<uint8_t>& m = r.head ? (r.tensor == 0 ? hk : hv) : (r.tensor == 0 ? dev[r.device].k : dev[r.device].v);
        CHECK(r.offset + r.bytes <= m.size());
        out.push_back({m.data() + r.offset, r.bytes});
      }
      return out;
    }
    // One device: Engine::kv_runs (engine.cc) restated - per tensor every layer's rows (the
    // head's last), then at int8 every layer's scales (the head's last).
    const runtime::KvLayout& L = dev[0].lay;
    const size_t n = e - b;
    for (int t = 0; t < 2; ++t) {
      std::vector<uint8_t>& m = t == 0 ? dev[0].k : dev[0].v;
      std::vector<uint8_t>& hm = t == 0 ? hk : hv;
      for (uint32_t l = 0; l < L.layers; ++l) out.push_back({m.data() + L.rows_offset(l) + b * L.row_bytes(), n * L.row_bytes()});
      if (mtp_on) out.push_back({hm.data() + b * L.row_bytes(), n * L.row_bytes()});
      if (L.scale_row_bytes() == 0) continue;
      for (uint32_t l = 0; l < L.layers; ++l)
        out.push_back({m.data() + L.scales_offset(l) + b * L.scale_row_bytes(), n * L.scale_row_bytes()});
      if (mtp_on) out.push_back({hm.data() + head_lay.scales_offset(0) + b * L.scale_row_bytes(), n * L.scale_row_bytes()});
    }
    return out;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) {
    CHECK(b <= e && e <= max_len());
    auto* h = static_cast<uint8_t*>(host);
    for (const Run& r : runs(b, e)) {
      std::memcpy(h, r.p, r.n);
      h += r.n;
    }
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) {
    CHECK(b <= e && e <= max_len());
    const auto* h = static_cast<const uint8_t*>(host);
    for (const Run& r : runs(b, e)) {
      std::memcpy(r.p, h, r.n);
      h += r.n;
    }
    kv_loaded += e - b;
  }
  void set_block_hook(BlockHook h) { hook = std::move(h); }

  Ids words;
  bool mtp_on;
  std::vector<Dev> dev;
  runtime::KvLayout head_lay;
  std::vector<uint8_t> hk, hv;   // the head's KV layer (the last device's)
  std::vector<uint64_t> hh;      // [0] h_{pos-1}, [1..4] the verify rows' hiddens
  uint64_t last_h[4] = {}, q_h[3] = {};
  uint32_t q_top[3] = {};
  static constexpr uint32_t kNone = 0xFFFFFFFFu;
  uint32_t verify_pos = 0, verify_k = kNone;
  Ids drafts;
  std::vector<float> rows;
  BlockHook hook;
  size_t resets = 0, state_loads = 0, kv_loaded = 0, prefilled = 0;
};

using Adapter = cli::pp::PipelineEngineAdapterT<FakeEngine>;

Ids some_words() {
  Ids w;
  for (uint32_t i = 0; i < 24; ++i) w.push_back(1000 + 7 * i);
  return w;
}
Ids prompt_of(uint32_t n, uint32_t salt) {
  Ids p;
  for (uint32_t i = 0; i < n; ++i) p.push_back(2000 + uint32_t(mix(salt, i) % 5000));
  return p;
}
std::vector<uint8_t> state_of(FakeEngine& e) {
  std::vector<uint8_t> s(e.state_bytes());
  e.save_state(s.data());
  return s;
}
std::vector<uint8_t> kv_of(FakeEngine& e, uint32_t b, uint32_t end) {
  std::vector<uint8_t> s(e.kv_bytes(end - b));
  e.save_kv(b, end, s.data());
  return s;
}

// ---- 1. one device and two ---------------------------------------------------------------------
void case_one_and_two() {
  for (KvCache form : {KvCache::Bf16, KvCache::Int8})
    for (bool mtp : {false, true}) {
      FakeEngine one(1, 512, form, mtp, some_words()), two(2, 512, form, mtp, some_words());
      CHECK_EQ(two.dev.size(), size_t{2});
      CHECK_EQ(two.dev[0].gdn.size() + two.dev[1].gdn.size(), one.dev[0].gdn.size());
      const Ids p = prompt_of(70, 1);
      one.prefill(p);
      two.prefill(p);
      CHECK(one.generate(48) == two.generate(48));
      CHECK(state_of(one) == state_of(two));
      CHECK(kv_of(one, 0, one.pos()) == kv_of(two, 0, two.pos()));
      CHECK_EQ(one.kv_bytes(10), two.kv_bytes(10));
    }
  std::printf("one device and two: 48 ids, the state and the KV bytes equal (bf16 / int8, +- MTP)\n");
}

// ---- 2. snapshots move between one device and two -----------------------------------------------
void case_snapshot_moves() {
  for (KvCache form : {KvCache::Bf16, KvCache::Int8})
    for (bool mtp : {false, true})
      for (int dir = 0; dir < 2; ++dir) {
        const uint32_t from = dir == 0 ? 1 : 2, to = dir == 0 ? 2 : 1;
        const Ids p = prompt_of(64, 2);
        FakeEngine ref(1, 512, form, mtp, some_words());
        ref.prefill(Ids(p.begin(), p.begin() + 48));   // a block end (16 x 3)
        ref.ingest(Ids(p.begin() + 48, p.end()));
        const Ids want = ref.generate(32);
        FakeEngine src(from, 512, form, mtp, some_words());
        src.prefill(Ids(p.begin(), p.begin() + 48));
        const std::vector<uint8_t> st = state_of(src), kv = kv_of(src, 0, 48);
        FakeEngine dst(to, 512, form, mtp, some_words());
        dst.load_state(st.data(), 48);
        dst.load_kv(0, 48, kv.data());
        dst.ingest(Ids(p.begin() + 48, p.end()));   // at least one id after a restore (spec 7 §3.3)
        CHECK(dst.generate(32) == want);
      }
  std::printf("snapshots move: a snapshot at 48 taken on one device continues on two and back, "
              "bitwise the uninterrupted run (bf16 / int8, +- MTP)\n");
}

// ---- 3. spec 8 M2 on the fake --------------------------------------------------------------------
void case_mtp_m2() {
  for (uint32_t devices : {1u, 2u})
    for (int mode = 0; mode < 4; ++mode) {   // K = 1, 2, 3, then a varying K
      FakeEngine plain(devices, 512, KvCache::Bf16, true, some_words());
      FakeEngine spec(devices, 512, KvCache::Bf16, true, some_words());
      const Ids p = prompt_of(41, 3);
      plain.prefill(p);
      spec.prefill(p);
      const uint32_t n = 96;
      const Ids want = plain.generate(n);
      Ids got;
      uint64_t iters = 0, drafted = 0, accepted = 0;
      while (got.size() < n) {
        const uint32_t want_k = mode < 3 ? uint32_t(mode + 1) : uint32_t(iters % 4);
        const uint32_t k = std::min(want_k, spec.max_verify_k());
        const uint32_t x = spec.pending();
        if (k == 0) {
          spec.verify(0);
          spec.commit(0, spec.verify_ids()[0]);
          got.push_back(x);
        } else {
          spec.draft(k);
          spec.verify(k);
          uint32_t j = 0;
          while (j < k && spec.draft_ids()[j] == spec.verify_ids()[j]) ++j;
          got.push_back(x);
          for (uint32_t i = 0; i < j; ++i) got.push_back(spec.draft_ids()[i]);
          spec.commit(j, spec.verify_ids()[j]);
          drafted += k;
          accepted += j;
        }
        spec.check_in_step();
        ++iters;
      }
      // The ids past n are a speculative run's tail: compare the first n, then bring the
      // plain run to the same position and compare the whole session.
      CHECK(Ids(got.begin(), got.begin() + n) == want);
      plain.generate(uint32_t(got.size() - n));
      CHECK_EQ(plain.pos(), spec.pos());
      CHECK(state_of(plain) == state_of(spec));   // the LIVE slot, the hidden row
      CHECK(kv_of(plain, 0, plain.pos()) == kv_of(spec, 0, spec.pos()));   // with the head's layer
      if (mode == 1)
        std::printf("  M2, %u device%s, K = 2: %zu ids in %llu iterations, %llu / %llu drafts accepted\n",
                    devices, devices == 1 ? "" : "s", got.size(), (unsigned long long)iters,
                    (unsigned long long)accepted, (unsigned long long)drafted);
    }
  std::printf("spec 8 M2: K = 1 / 2 / 3 / varying give the plain ids and the plain session, one "
              "device and two\n");
}

// ---- 4. spec 7's request path through the adapter -------------------------------------------------
struct MallocAlloc : server::HostAlloc {
  size_t live = 0;
  void* alloc(size_t n) override {
    live += n;
    return std::malloc(n);
  }
  void free(void* p, size_t n) override {
    live -= n;
    std::free(p);
  }
};

Ids turn(server::PrefixSession& s, Adapter& ad, const Ids& prompt, uint32_t n, uint32_t mtp_k,
         server::PrefixSession::Report* rep = nullptr) {
  const auto r = s.begin(prompt);
  if (rep) *rep = r;
  Ids out;
  server::Sampling greedy;
  while (out.size() < n) {
    const Ids burst = mtp_k ? ad.step_many(greedy, mtp_k) : Ids{ad.step(greedy)};
    for (uint32_t id : burst) {
      if (out.size() == n) break;
      out.push_back(id);
      s.fed(id);
    }
    if (out.size() == n && mtp_k) ad.truncate_to(uint32_t(prompt.size() + n));   // keep what the server keeps
  }
  s.end();
  return out;
}
Ids cold(const Ids& prompt, uint32_t n) {
  FakeEngine e(1, 512, KvCache::Bf16, false, some_words());
  e.prefill(prompt);
  return e.generate(n);
}
Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

void case_prefix() {
  const Ids L = prompt_of(400, 4);
  const Ids C(L.rbegin(), L.rbegin() + 60);
  using K = server::PrefixCache::Plan;
  MallocAlloc alloc;
  for (uint32_t mtp_k : {0u, 2u})
    for (const char* name : {"b prompt end", "c mid-block", "d block boundary", "e side request", "f same prompt"}) {
      FakeEngine eng(2, 512, KvCache::Bf16, mtp_k != 0, some_words());
      Adapter ad(eng, 248077, kV, mtp_k);
      CHECK_EQ(ad.block(), FakeEngine::kBlock);
      CHECK_EQ(ad.kv_form(), uint64_t{0});   // the one-card store key: the bytes are the one-card layout's
      server::PrefixSession s(ad, size_t(64) << 20, &alloc);
      Ids final_prompt, turn1;
      const char n = name[0];
      if (n == 'b') {
        const Ids t1 = slice(L, 0, 100);
        (void)turn(s, ad, t1, 8, mtp_k);
        final_prompt = cat(t1, slice(C, 0, 30));
      } else if (n == 'c' || n == 'd') {
        (void)turn(s, ad, slice(L, 0, 200), 8, mtp_k);
        final_prompt = cat(slice(L, 0, n == 'c' ? 150 : 144), slice(C, 0, 30));
      } else if (n == 'e') {
        const Ids t1 = slice(L, 0, 158);
        const Ids g1 = turn(s, ad, t1, 8, mtp_k);
        (void)turn(s, ad, slice(L, 300, 340), 4, mtp_k);
        final_prompt = cat(cat(t1, g1), slice(L, 158, 190));
      } else {
        const Ids t1 = slice(L, 0, 120);
        turn1 = turn(s, ad, t1, 16, mtp_k);
        (void)turn(s, ad, slice(L, 300, 330), 4, mtp_k);
        final_prompt = t1;
      }
      server::PrefixSession::Report rep;
      const Ids got = turn(s, ad, final_prompt, 16, mtp_k, &rep);
      CHECK(rep.kind == K::Restore);
      CHECK(got == cold(final_prompt, 16));
      if (n == 'f') CHECK(got == turn1);
      CHECK(eng.state_loads > 0);
      std::printf("  %s, mtp %u: %s at %u, 16 ids == a cold one-device run\n", name, mtp_k,
                  server::plan_kind_name(rep.kind), rep.restart);
    }
  CHECK_EQ(alloc.live, size_t(0));
}

// ---- 5. the server ---------------------------------------------------------------------------------
struct ServeFixture {
  ServeFixture(bool cache, uint32_t mtp_k, bool lookup, bool split_last = false)
      : eng(2, 512, KvCache::Bf16, mtp_k != 0 || lookup, words()),
        adapter(eng, 248077, kV, mtp_k, lookup),
        server({tok, tmpl, adapter}, options(cache, lookup, split_last)) {
    CHECK(server.start());
  }
  ~ServeFixture() { server.stop(); }
  Ids words() {
    Ids w;
    for (uint32_t i = 0; i < 24; ++i) w.push_back(tok.id_of("w" + std::to_string(i)));
    return w;
  }
  server::Options options(bool cache, bool lookup, bool split_last) {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "qwen";
    o.eos_ids = {248046};
    o.spec_lookup = lookup;
    o.lookup_min_match = 2;
    o.lookup_adaptive.cost = server::MtpCost::int8_head().with_free_drafts();   // b70-serve's
    o.prefix_split_last = split_last;
    if (cache) {
      o.prefix_cache_bytes = size_t(256) << 20;
      o.prefix_alloc = &alloc;
    }
    return o;
  }
  json chat(const json& extra) {
    json b = {{"model", "qwen"},
              {"messages", json::array({{{"role", "user"},
                                         {"content", "explain the pipeline of the server across two "
                                                     "cards in a few words please and then stop"}}})},
              {"max_tokens", 40},
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
  MockTemplate tmpl;
  MallocAlloc alloc;
  FakeEngine eng;
  Adapter adapter;
  server::Server server;
};

void case_server() {
  const json greedy = {{"temperature", 0}};
  Ids prompt, served;
  {
    ServeFixture f(/*cache=*/true, 0, false);
    const json r1 = f.chat(greedy);
    prompt = r1.at("prompt_token_ids").get<Ids>();
    served = r1.at("choices").at(0).at("token_ids").get<Ids>();
    CHECK(!prompt.empty());
    CHECK(!served.empty());
    // b70-decode on one card: --ids <prompt_token_ids> --n <served> --prefill
    FakeEngine decode(1, 512, KvCache::Bf16, false, f.eng.words);
    decode.prefill(prompt);
    CHECK(decode.generate(uint32_t(served.size())) == served);
    const json r2 = f.chat(greedy);
    CHECK(r2.at("choices").at(0).at("token_ids").get<Ids>() == served);
    const uint32_t cached = r2.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
    CHECK(cached > 0 && cached < prompt.size());
    CHECK(f.eng.state_loads > 0);
    std::printf("server --pp 2: greedy chat, %zu prompt ids -> %zu ids == one card's decode run; again: "
                "restored at %u, the same ids\n", prompt.size(), served.size(), cached);
  }
  for (const auto& [mtp_k, lookup] : std::vector<std::pair<uint32_t, bool>>{{2, false}, {3, false}, {0, true}}) {
    ServeFixture f(/*cache=*/true, mtp_k, lookup);
    CHECK(f.chat(greedy).at("choices").at(0).at("token_ids").get<Ids>() == served);
    CHECK(f.chat(greedy).at("choices").at(0).at("token_ids").get<Ids>() == served);   // from the cache
    std::printf("server --pp 2 %s: the same ids, twice (the second from the cache)\n",
                lookup ? "--spec lookup" : ("--mtp " + std::to_string(mtp_k)).c_str());
  }
  {
    ServeFixture f(/*cache=*/true, 2, false, /*split_last=*/true);
    CHECK(f.chat(greedy).at("choices").at(0).at("token_ids").get<Ids>() == served);
    const json r = f.chat(greedy);
    CHECK(r.at("choices").at(0).at("token_ids").get<Ids>() == served);
    std::printf("server --pp 2 --mtp 2 --prefix-split-last: the same ids; again restored at %u\n",
                r.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>());
  }
  // Sampled, seeded: reproducible on a fresh server, every id a word - plain and with MTP.
  const json sampled = {{"temperature", 0.8}, {"top_k", 8}, {"seed", 1234}};
  for (uint32_t mtp_k : {0u, 2u}) {
    Ids a, b, words;
    {
      ServeFixture g(/*cache=*/false, mtp_k, false);
      a = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
      words = g.eng.words;
    }
    {
      ServeFixture g(/*cache=*/false, mtp_k, false);
      b = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
    }
    CHECK(a == b);
    CHECK(!a.empty());
    for (uint32_t id : a) CHECK(std::find(words.begin(), words.end(), id) != words.end() || id == 248046);
    std::printf("server --pp 2%s: a seeded sampled request reproducible (%zu ids), every id a word\n",
                mtp_k ? " --mtp 2" : "", a.size());
  }
}

}  // namespace

int main() {
  case_one_and_two();
  case_snapshot_moves();
  case_mtp_m2();
  case_prefix();
  case_server();
  std::printf("pp_serve_test OK\n");
  return 0;
}

// Spec 18d (plan 18d Task 1's wiring and Task 2's Review Focus 3, host side): K2-Horizon behind
// b70-serve without a device. cli::k2::K2EngineAdapterT - the adapter b70-serve builds over
// runtime::k2::K2Engine - runs here over FakeK2, an engine with K2Engine's interface whose KV
// cache is host memory in K2's real layout (runtime::k2::kv_layout of model::k2(): 48 layers x 8
// heads x 128, bf16 or int8 rows + fp16 scales) and whose snapshot calls are K2Engine's own
// arithmetic (runtime::k2::kv_snapshot_runs). Its "model" is exact and context-dependent: each
// position's K / V rows of every layer are a function of the id, the position and the previous
// position's last-layer row, and the next id a function of the last row - so a restore that
// misplaces one byte of one layer's rows (or scales) changes what it generates.
//
//   1. the snapshot layout on K2's real shapes (max_len 4096): 96 runs at bf16, 192 at int8 (rows
//      then scales, K then V), 196,608 / 99,840 B a position, every run inside its allocation,
//      the runs of a range disjoint and exactly the range of every layer; empty and bad ranges;
//   2. save / load on host buffers, both forms: [b, e) restored bit for bit, everything outside
//      untouched; two adjacent saves restore what one save of their union does;
//   3. Review Focus 3 through spec 7's request path (server::PrefixSession over the adapter, the
//      block 16 for the test): KV-only snapshots - state_bytes() 0, the empty block-end snapshots
//      as restore points - continue / restore at the prompt end / mid-block / on a block /
//      after a side request / the same prompt, every cached turn's ids equal to a cold run's
//      bit for bit; the store's root is K2's key in its form, never Qwen's 0 / 1;
//   4. the server (HTTP, K2's chat format): a greedy chat's generated ids equal FakeK2 run as
//      b70-decode runs it on the response's prompt_token_ids (prefill, then generate) - the box
//      stage r25.chat does the same against the real engine; the second identical request
//      restores from the cache (usage cached_tokens > 0) and generates the same ids; a seeded
//      sampled request is reproducible and draws only ids the logits row allows.
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
#include "cli/k2_serve_adapter.h"
#include "model/k2_horizon.h"
#include "runtime/k2/k2_sizes.h"
#include "runtime/prefill_chunks.h"
#include "server/mock.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using Ids = std::vector<uint32_t>;
using runtime::KvCache;

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// K2Engine's interface (what K2EngineAdapterT calls), device-free.
struct FakeK2 {
  static constexpr uint32_t kBlock = 16;   // K2Engine::kBlock is 2048; small for the test
  static constexpr uint32_t kChunk = 24;   // the prefill chunk (kPfC's role)
  using BlockHook = std::function<void(uint32_t, bool)>;

  FakeK2(uint32_t len, KvCache form, std::vector<uint32_t> vocab_words)
      : lay(runtime::k2::kv_layout(model::k2(), len, form)),
        k(lay.bytes()),
        v(lay.bytes()),
        words(std::move(vocab_words)) {}

  // --- the session ---------------------------------------------------------------------------
  void reset() {
    std::fill(k.begin(), k.end(), uint8_t(0));
    std::fill(v.begin(), v.end(), uint8_t(0));
    p = 0;
    cur = 0;
    ++resets;
  }
  // K2Engine::prefill's control flow: chunks by runtime::prefill_chunk_rows, the mid-prompt hook
  // at block ends (pos = end), the prompt-end hook after the first generated id is pending.
  void prefill(const Ids& ids) {
    CHECK(!ids.empty());
    const bool hooked = static_cast<bool>(hook);
    const uint32_t base = p;
    uint32_t C = 0;
    for (size_t off = 0; off < ids.size(); off += C) {
      C = runtime::prefill_chunk_rows(base + uint32_t(off), ids.size() - off, kChunk, hooked, kBlock);
      for (uint32_t i = 0; i < C; ++i) write(ids[off + i]);
      const uint32_t end = base + uint32_t(off) + C;
      if (hooked && off + C < ids.size() && end % kBlock == 0) hook(end, true);
    }
    cur = next();
    prefilled += ids.size();
    if (hooked) hook(p, p % kBlock == 0);
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
  uint32_t pos() const { return p; }
  uint32_t max_len() const { return lay.max_len; }
  uint32_t vocab() const { return model::k2().vocab; }
  KvCache kv_cache() const { return lay.form; }

  // --- K2Engine's snapshot calls (k2_engine.cc: the same run list, host memory for device) ---
  size_t state_bytes() const { return 0; }
  size_t kv_bytes(uint32_t n) const { return runtime::k2::kv_snapshot_bytes(lay, n); }
  void save_state(void*) const {}
  void load_state(const void*, uint32_t at) {
    CHECK(at <= lay.max_len);
    p = at;
    ++state_loads;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) const {
    auto* h = static_cast<uint8_t*>(host);
    for (const auto& r : runtime::k2::kv_snapshot_runs(lay, b, e)) {
      std::memcpy(h, (r.tensor == 0 ? k : v).data() + r.offset, r.bytes);
      h += r.bytes;
    }
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) {
    const auto* h = static_cast<const uint8_t*>(host);
    for (const auto& r : runtime::k2::kv_snapshot_runs(lay, b, e)) {
      std::memcpy((r.tensor == 0 ? k : v).data() + r.offset, h, r.bytes);
      h += r.bytes;
    }
    kv_loaded += e - b;
  }
  void set_block_hook(BlockHook h) { hook = std::move(h); }
  void set_pending(uint32_t id) {
    CHECK(id < vocab());
    cur = id;
  }
  // A row that allows exactly the vocabulary words (everything else -1e30), shaped by the
  // last position's row - the sampler's input.
  void read_logits_into(float* host) {
    std::fill(host, host + vocab(), -1e30f);
    uint64_t h = digest(p == 0 ? 0 : p - 1);
    for (uint32_t w : words) {
      h = mix(h, w);
      host[w] = float(h % 1000) / 250.0f;
    }
  }

  // --- the "model" -----------------------------------------------------------------------------
  // Position q's row of layer l (K: salt 0, V: salt 1), every byte from (id, q, l, the previous
  // position's last-layer K row); at int8 the scales too.
  void write(uint32_t id) {
    CHECK(p < lay.max_len);
    const uint64_t prev = p == 0 ? 0x1234 : digest(p - 1);
    for (uint32_t l = 0; l < lay.layers; ++l)
      for (int t = 0; t < 2; ++t) {
        std::vector<uint8_t>& m = t == 0 ? k : v;
        uint64_t h = mix(mix(mix(prev, id), uint64_t(p) * 131 + l), uint64_t(t));
        uint8_t* row = m.data() + lay.rows_offset(l) + size_t(p) * lay.row_bytes();
        for (size_t i = 0; i < lay.row_bytes(); i += 8) {
          h = mix(h, i);
          std::memcpy(row + i, &h, std::min<size_t>(8, lay.row_bytes() - i));
        }
        if (lay.scale_row_bytes() != 0) {
          uint8_t* sc = m.data() + lay.scales_offset(l) + size_t(p) * lay.scale_row_bytes();
          for (size_t i = 0; i < lay.scale_row_bytes(); ++i) sc[i] = uint8_t(mix(h, i + 977));
        }
      }
    ++p;
  }
  // What the next position reads of position q: its rows of the first and last layers (K and
  // V) and, at int8, their scales - a restore that drops or moves either changes the hash.
  uint64_t digest(uint32_t q) const {
    uint64_t h = 0x51ED;
    for (uint32_t l : {0u, lay.layers / 2, lay.layers - 1})
      for (const std::vector<uint8_t>* m : {&k, &v}) {
        const uint8_t* row = m->data() + lay.rows_offset(l) + size_t(q) * lay.row_bytes();
        for (size_t i = 0; i < lay.row_bytes(); i += 64) h = mix(h, row[i] | (uint64_t(row[i + 63]) << 8));
        if (lay.scale_row_bytes() != 0)
          h = mix(h, m->data()[lay.scales_offset(l) + size_t(q) * lay.scale_row_bytes() + 3]);
      }
    return h;
  }
  uint32_t next() const { return words[digest(p - 1) % words.size()]; }

  runtime::KvLayout lay;
  std::vector<uint8_t> k, v;
  std::vector<uint32_t> words;
  uint32_t p = 0, cur = 0;
  BlockHook hook;
  size_t resets = 0, state_loads = 0, kv_loaded = 0, prefilled = 0;
};

using Adapter = cli::k2::K2EngineAdapterT<FakeK2>;

struct MallocAlloc : server::HostAlloc {
  size_t live = 0, zero_requests = 0;
  void* alloc(size_t n) override {
    if (n == 0) ++zero_requests;   // PrefixCache must never ask (spec 18d's empty snapshots)
    live += n;
    return std::malloc(n);
  }
  void free(void* p, size_t n) override {
    live -= n;
    std::free(p);
  }
};

Ids some_words() {
  Ids w;
  for (uint32_t i = 0; i < 24; ++i) w.push_back(1000 + 7 * i);
  return w;
}

// ---- 1. the layout on K2's real shapes ----------------------------------------------------------
void case_layout() {
  const model::K2Desc& d = model::k2();
  for (KvCache form : {KvCache::Bf16, KvCache::Int8}) {
    const runtime::KvLayout lay = runtime::k2::kv_layout(d, 4096, form);
    const bool int8 = form == KvCache::Int8;
    CHECK_EQ(lay.layers, 48u);
    CHECK_EQ(runtime::k2::kv_snapshot_bytes(lay, 1), int8 ? size_t(99840) : size_t(196608));
    CHECK(runtime::k2::kv_snapshot_runs(lay, 100, 100).empty());
    bool threw = false;
    try {
      (void)runtime::k2::kv_snapshot_runs(lay, 10, 4097);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
    for (const auto& [b, e] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {0, 2048}, {2048, 4096}, {777, 3001}}) {
      const auto runs = runtime::k2::kv_snapshot_runs(lay, b, e);
      CHECK_EQ(runs.size(), size_t(int8 ? 192 : 96));
      size_t total = 0;
      for (size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        total += r.bytes;
        CHECK(r.offset + r.bytes <= lay.bytes());
        // host order: tensor K (runs [0, n/2)), then V; within each the rows of layers 0..47,
        // then (int8) the scales of layers 0..47
        const size_t per = runs.size() / 2, j = i % per;
        CHECK_EQ(r.tensor, uint32_t(i < per ? 0 : 1));
        const uint32_t l = uint32_t(j % 48);
        const bool scale = j >= 48;
        CHECK_EQ(r.offset, scale ? lay.scales_offset(l) + size_t(b) * lay.scale_row_bytes()
                                 : lay.rows_offset(l) + size_t(b) * lay.row_bytes());
        CHECK_EQ(r.bytes, size_t(e - b) * (scale ? lay.scale_row_bytes() : lay.row_bytes()));
        for (size_t q = 0; q < i; ++q)   // disjoint within one allocation
          if (runs[q].tensor == r.tensor)
            CHECK(runs[q].offset + runs[q].bytes <= r.offset || r.offset + r.bytes <= runs[q].offset);
      }
      CHECK_EQ(total, runtime::k2::kv_snapshot_bytes(lay, e - b));
    }
  }
  std::printf("layout: 96 / 192 runs, 196608 / 99840 B a position, disjoint, inside the allocation\n");
}

// ---- 2. save / load on host buffers ---------------------------------------------------------------
void case_round_trip() {
  for (KvCache form : {KvCache::Bf16, KvCache::Int8}) {
    FakeK2 e(96, form, some_words());
    std::mt19937_64 rng(form == KvCache::Int8 ? 9 : 4);
    for (auto& b : e.k) b = uint8_t(rng());
    for (auto& b : e.v) b = uint8_t(rng());
    const std::vector<uint8_t> k0 = e.k, v0 = e.v;
    std::vector<uint8_t> host(e.kv_bytes(96));
    e.save_kv(17, 61, host.data());
    for (auto& b : e.k) b = uint8_t(rng());
    for (auto& b : e.v) b = uint8_t(rng());
    const std::vector<uint8_t> k1 = e.k, v1 = e.v;
    e.load_kv(17, 61, host.data());
    // [17, 61) is k0 / v0 again in every layer (and scale), everything else stays k1 / v1
    std::vector<bool> in(e.k.size(), false);
    for (const auto& r : runtime::k2::kv_snapshot_runs(e.lay, 17, 61))
      if (r.tensor == 0)
        for (size_t i = 0; i < r.bytes; ++i) in[r.offset + i] = true;
    size_t restored = 0;
    for (size_t i = 0; i < e.k.size(); ++i) {
      CHECK_EQ(e.k[i], in[i] ? k0[i] : k1[i]);
      CHECK_EQ(e.v[i], in[i] ? v0[i] : v1[i]);
      restored += in[i];
    }
    CHECK_EQ(2 * restored, e.kv_bytes(44));
    // two adjacent saves = one save of their union
    std::vector<uint8_t> a(e.kv_bytes(30)), b(e.kv_bytes(50)), ab(e.kv_bytes(80));
    e.save_kv(0, 30, a.data());
    e.save_kv(30, 80, b.data());
    e.save_kv(0, 80, ab.data());
    FakeK2 x(96, form, some_words()), y(96, form, some_words());
    x.load_kv(0, 30, a.data());
    x.load_kv(30, 80, b.data());
    y.load_kv(0, 80, ab.data());
    CHECK(x.k == y.k && x.v == y.v);
    e.save_kv(5, 5, nullptr);   // empty: no copy, the host pointer is not touched
  }
  std::printf("round trip: [17, 61) restored bit for bit in both forms, the rest untouched; "
              "adjacent saves compose\n");
}

// ---- 3. Review Focus 3 through spec 7's request path -------------------------------------------
Ids turn(server::PrefixSession& s, Adapter& ad, const Ids& prompt, uint32_t n,
         server::PrefixSession::Report* rep = nullptr) {
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

Ids cold(KvCache form, const Ids& prompt, uint32_t n) {
  FakeK2 e(512, form, some_words());
  e.prefill(prompt);
  return e.generate(n);
}

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

void case_prefix(KvCache form) {
  Ids L;
  for (uint32_t i = 0; i < 400; ++i) L.push_back(2000 + (i * 7919) % 5000);
  const Ids C(L.rbegin(), L.rbegin() + 60);
  using K = server::PrefixCache::Plan;
  struct Want {
    const char* name;
    K::Kind kind;
    uint32_t restart;
  };
  MallocAlloc alloc;
  for (const Want& w : {Want{"a continue", K::Continue, 108}, Want{"b prompt end", K::Restore, 100},
                        Want{"c mid-block", K::Restore, 144}, Want{"d block boundary", K::Restore, 144},
                        Want{"e side request", K::Restore, 166}, Want{"f same prompt", K::Restore, 112}}) {
    FakeK2 eng(512, form, some_words());
    Adapter ad(eng, 248077);
    CHECK_EQ(ad.state_bytes(), size_t(0));
    CHECK_EQ(ad.block(), FakeK2::kBlock);
    server::PrefixSession s(ad, size_t(64) << 20, &alloc);
    CHECK_EQ(s.cache()->kv_form(), cli::k2::k2_store_key(form));
    CHECK(s.cache()->kv_form() != 0 && s.cache()->kv_form() != 1);   // never Qwen's roots
    Ids final_prompt, turn1;
    const char n = w.name[0];
    if (n == 'a' || n == 'b') {
      const Ids t1 = slice(L, 0, 100);
      const Ids g1 = turn(s, ad, t1, 8);
      final_prompt = n == 'a' ? cat(cat(t1, g1), slice(L, 100, 140)) : cat(t1, slice(C, 0, 30));
    } else if (n == 'c' || n == 'd') {
      (void)turn(s, ad, slice(L, 0, 200), 8);
      final_prompt = cat(slice(L, 0, n == 'c' ? 150 : 144), slice(C, 0, 30));
    } else if (n == 'e') {
      const Ids t1 = slice(L, 0, 158);
      const Ids g1 = turn(s, ad, t1, 8);
      (void)turn(s, ad, slice(L, 300, 340), 4);
      final_prompt = cat(cat(t1, g1), slice(L, 158, 190));
    } else {
      const Ids t1 = slice(L, 0, 120);
      turn1 = turn(s, ad, t1, 16);
      (void)turn(s, ad, slice(L, 300, 330), 4);
      final_prompt = t1;
    }
    server::PrefixSession::Report rep;
    const Ids got = turn(s, ad, final_prompt, 16, &rep);
    if (rep.kind != w.kind || rep.restart != w.restart)
      std::fprintf(stderr, "%s: plan %s at %u, expected %s at %u\n", w.name, server::plan_kind_name(rep.kind),
                   rep.restart, server::plan_kind_name(w.kind), w.restart);
    CHECK(rep.kind == w.kind);
    CHECK_EQ(rep.restart, w.restart);
    const Ids want = cold(form, final_prompt, 16);
    CHECK(got == want);   // KV-only snapshots restore the session exactly
    if (n == 'f') CHECK(got == turn1);
    if (w.kind == K::Restore) CHECK(eng.state_loads > 0);
    std::printf("  %s %s: %s at %u (kv %zu B restored), 16 ids == cold\n", runtime::kv_cache_name(form), w.name,
                server::plan_kind_name(rep.kind), rep.restart, rep.kv_bytes);
  }
  CHECK_EQ(alloc.zero_requests, size_t(0));
  CHECK_EQ(alloc.live, size_t(0));   // every PrefixSession freed its store
  // c / d restored at 144 = 9 blocks of 16: the restore point is an empty block-end snapshot
  // (state_bytes() 0, no KV of its own) and every KV byte came from the blocks.
}

// ---- 4. the server ----------------------------------------------------------------------------------
struct K2Template : server::TemplateIface {
  std::string render(const json& messages, const json&, bool) override { return body(messages, json::object()); }
  std::string render_with_kwargs(const json& messages, const json&, bool, const json& kwargs) override {
    return body(messages, kwargs);
  }
  static std::string body(const json& messages, const json& kwargs) {
    std::string text = "<|ifm|begin_of_text|> ";
    for (const auto& m : messages)
      text += "<|ifm|im_start|>" + m.at("role").get<std::string>() + " " + m.at("content").get<std::string>() +
              " <|ifm|im_end|> ";
    const std::string effort = kwargs.value("reasoning_effort", std::string("high"));
    return text + "<|ifm|im_start|>assistant\n" +
           (effort == "low" ? "<ifm|think_faster>" : effort == "medium" ? "<ifm|think_fast>" : "<ifm|think>") +
           "\n";
  }
};

struct ServeFixture {
  explicit ServeFixture(bool cache)
      : eng(512, KvCache::Bf16, words()), adapter(eng, 248077), server({tok, tmpl, adapter}, options(cache)) {
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
    o.served_model = "k2";
    o.eos_ids = {1, 250019};
    o.chat_format = server::ChatFormat::for_model_type("k2_horizon");
    if (cache) {
      o.prefix_cache_bytes = size_t(256) << 20;
      o.prefix_alloc = &alloc;
    }
    return o;
  }
  json chat(const json& extra) {
    json b = {{"model", "k2"},
              {"messages", json::array({{{"role", "user"},
                                         {"content", "explain the prefix cache of the server in a few words "
                                                     "please and then stop"}}})},
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
  K2Template tmpl;
  MallocAlloc alloc;
  FakeK2 eng;
  Adapter adapter;
  server::Server server;
};

void case_server() {
  ServeFixture f(/*cache=*/true);
  const json greedy = {{"temperature", 0}};
  const json r1 = f.chat(greedy);
  const Ids prompt = r1.at("prompt_token_ids").get<Ids>();
  const Ids served = r1.at("choices").at(0).at("token_ids").get<Ids>();
  CHECK(!prompt.empty());
  CHECK_EQ(served.size(), size_t(24));
  // b70-decode's run of the same ids: --ids <prompt_token_ids> --n 24 --prefill
  FakeK2 decode(512, KvCache::Bf16, f.eng.words);
  decode.prefill(prompt);
  CHECK(decode.generate(24) == served);
  // The same request again: restored from the prompt-end snapshot (KV only), the same ids.
  const json r2 = f.chat(greedy);
  CHECK(r2.at("prompt_token_ids").get<Ids>() == prompt);
  CHECK(r2.at("choices").at(0).at("token_ids").get<Ids>() == served);
  const uint32_t cached = r2.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
  CHECK(cached > 0 && cached < prompt.size());
  CHECK(f.eng.state_loads > 0);
  std::printf("server: greedy chat of %zu prompt ids -> 24 ids == the decode run; again: restored at %u, "
              "same ids\n", prompt.size(), cached);

  // Sampled, seeded: reproducible on a fresh server, and only ids the logits row allows.
  const json sampled = {{"temperature", 0.8}, {"top_k", 8}, {"seed", 1234}};
  Ids a, b;
  {
    ServeFixture g(/*cache=*/false);
    a = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
  }
  {
    ServeFixture g(/*cache=*/false);
    b = g.chat(sampled).at("choices").at(0).at("token_ids").get<Ids>();
  }
  CHECK(a == b);
  for (uint32_t id : a) CHECK(std::find(f.eng.words.begin(), f.eng.words.end(), id) != f.eng.words.end());
  std::printf("server: a seeded sampled request reproducible (%zu ids), every id a vocabulary word\n", a.size());
}

}  // namespace

int main() {
  case_layout();
  case_round_trip();
  case_prefix(KvCache::Bf16);
  case_prefix(KvCache::Int8);
  case_server();
  std::printf("k2_serve_test OK\n");
  return 0;
}

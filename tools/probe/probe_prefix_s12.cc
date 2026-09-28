// Spec 7 S1 and S2 (plan 7c Task 4 step 2): time to first token of a turn at 60k tokens
// of history, through the server's request path (server::PrefixSession over EngineAdapter,
// the code b70-serve runs; no HTTP, no tokenizer).
//   S1  continuation: turn A = H (60000 ids, long32k.ids repeated) + 16 generated; turn B =
//       A + a 1000-id tail. The card holds A: nothing is copied, the tail is prefilled.
//   S2  the same after a 300-id side request: B restores A's request-end snapshot (state +
//       the whole KV from the host), then prefills the tail.
// Each measured turn is begin() until it returns: the first generated id is then pending
// (what a streamed response's first chunk waits for). One warm-up of each, then 3 runs,
// median; every run has its own tail, so no run is served by an earlier one's snapshot.
// Before each S1 run the resident session is re-made A (a restore at the last block end
// and a short prefill, untimed).
//   probe_prefix_s12 <snapshot> <long32k.ids> [--history 60000] [--tail 1000] [--gb 32]
//                    [--split 0|1]   (1: PrefixSession's opt-in split_last)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "cli/serve_adapters.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
#include "server/prefix_cache.h"

namespace {
using Ids = std::vector<uint32_t>;

Ids read_ids(const std::string& path) {
  std::ifstream in(path);
  Ids ids;
  uint32_t id = 0;
  while (in >> id) ids.push_back(id);
  return ids;
}
Ids repeat_to(const Ids& src, size_t n) {
  Ids out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
Ids slice(const Ids& v, size_t a, size_t n) { return Ids(v.begin() + a, v.begin() + a + n); }

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

Ids gen(server::PrefixSession& s, EngineAdapter& ad, uint32_t n) {
  Ids out;
  server::Sampling greedy;
  for (uint32_t i = 0; i < n; ++i) {
    out.push_back(ad.step(greedy));
    s.fed(out.back());
  }
  return out;
}

void print(const char* tag, const server::PrefixSession::Report& r, size_t len) {
  std::printf("  %-10s %zu ids: %-8s at %6u  kv %7.1f MB  restore %7.1f ms  prefill %7.1f ms"
              "  store %6.1f ms  total %7.1f ms\n", tag, len, server::plan_kind_name(r.kind),
              r.restart, r.kv_bytes / 1e6, r.restore_ms, r.prefill_ms, r.store_ms,
              r.restore_ms + r.prefill_ms + r.store_ms);
  std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: probe_prefix_s12 <snapshot> <ids> [--history N] [--tail T]"
                         " [--gb G]\n");
    return 2;
  }
  uint32_t history = 60000, tail = 1000, gb = 32, split = 0;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const uint32_t v = uint32_t(std::stoul(argv[i + 1]));
    if (a == "--history") history = v;
    else if (a == "--tail") tail = v;
    else if (a == "--gb") gb = v;
    else if (a == "--split") split = v;
  }
  const Ids L = read_ids(argv[2]);
  const Ids H = repeat_to(L, history);
  constexpr uint32_t kMaxLen = 131072;
  l0::Context ctx(0);
  runtime::Engine eng(ctx, loader::load(ctx, loader::resolve_snapshot(argv[1]), kMaxLen), kMaxLen);
  eng.prepare_prefill();
  EngineAdapter ad(eng, model::Qwen35::kVocabUsed);
  PinnedAlloc alloc(ctx, size_t(gb) << 30);
  server::PrefixSession s(ad, alloc.mem.size(), &alloc);
  const bool sp = split != 0;
  const auto begin = [&](const Ids& p) { return s.begin(p, sp); };
  std::printf("probe_prefix_s12: history %u, tail %u, cache %u GiB, split_last %u, backend %s\n",
              history, tail, gb, split, runtime::prefill_backend_name(eng.prefill_backend()));

  auto r = begin(H);
  print("A (cold)", r, H.size());
  const Ids A = cat(H, gen(s, ad, 16));
  std::printf("  request end: store %.1f ms\n", s.end());
  const Ids side = slice(L, 20000, 300);

  std::vector<double> s1, s2, s2_restore;
  for (int run = 0; run < 4; ++run) {   // run 0 is the warm-up
    // S1: re-make the resident session A, then the continuation.
    r = begin(A);
    print("re-make A", r, A.size());
    s.end();
    const Ids b1 = cat(A, slice(L, 2000 + 2 * tail * run, tail));
    r = begin(b1);
    print(run ? "S1" : "S1 warm", r, b1.size());
    if (r.kind != server::PrefixCache::Plan::Continue) std::printf("  ** S1 did not continue **\n");
    if (run) s1.push_back(r.restore_ms + r.prefill_ms + r.store_ms);
    (void)gen(s, ad, 4);
    s.end();
    // S2: a side request, then the main session back.
    r = begin(side);
    print("side", r, side.size());
    (void)gen(s, ad, 8);
    s.end();
    const Ids b2 = cat(A, slice(L, 2000 + 2 * tail * run + tail, tail));
    r = begin(b2);
    print(run ? "S2" : "S2 warm", r, b2.size());
    if (r.kind != server::PrefixCache::Plan::Restore) std::printf("  ** S2 did not restore **\n");
    if (run) {
      s2.push_back(r.restore_ms + r.prefill_ms + r.store_ms);
      s2_restore.push_back(r.restore_ms);
    }
    (void)gen(s, ad, 4);
    s.end();
  }
  std::printf("S1 time to first token, median of 3: %.1f ms (bar 1500)\n", median(s1));
  std::printf("S2 restore, median of 3: %.1f ms (bar 1000); time to first token %.1f ms\n",
              median(s2_restore), median(s2));
  std::printf("store: %.2f GB of %.2f GB used, %zu entries\n", s.cache()->bytes_used() / 1e9,
              s.cache()->budget() / 1e9, s.cache()->entries());
  return 0;
}

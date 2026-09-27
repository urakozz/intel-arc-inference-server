// Spec 7 S3 (plan 7b Task 3): the write-through cost of block snapshots.
// Cold prefill of N ids with no block hook against the same prefill with a hook that does
// what plan 7c's store will do at each call: save_state into a pinned host buffer and
// save_kv of the positions since the previous call. Interleaved pairs after a warm-up pair
// (the order alternates pair to pair), median of the per-pair ratios.
//   probe_writethrough <snapshot> <ids file> [--n 4096,32768] [--pairs 3] [--every 2048]
// --every 4096: copy only at block ends that are multiples of 4096 (and at the prompt
// end), the half-snapshot variant of plan 7b Task 3 step 3.
// The model is loaded at max_len 131072 (spec 6's serving length).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/engine.h"
#include "timer.h"

namespace {

std::vector<uint32_t> list_arg(const std::string& s) {
  std::vector<uint32_t> out;
  std::stringstream in(s);
  std::string item;
  while (std::getline(in, item, ',')) out.push_back(static_cast<uint32_t>(std::stoul(item)));
  return out;
}

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream in(path);
  std::vector<uint32_t> ids;
  uint32_t id = 0;
  while (in >> id) ids.push_back(id);
  return ids;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: probe_writethrough <snapshot> <ids file> [--n a,b] [--pairs k]"
                         " [--every 2048|4096]\n");
    return 2;
  }
  std::vector<uint32_t> ns = {4096, 32768};
  int pairs = 3;
  uint32_t every = runtime::Engine::kBlock;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--n") ns = list_arg(argv[i + 1]);
    if (flag == "--pairs") pairs = std::stoi(argv[i + 1]);
    if (flag == "--every") every = static_cast<uint32_t>(std::stoul(argv[i + 1]));
  }
  constexpr uint32_t kMaxLen = 131072;
  const std::vector<uint32_t> source = read_ids(argv[2]);
  if (source.empty()) {
    std::fprintf(stderr, "no ids in %s\n", argv[2]);
    return 2;
  }
  const uint32_t need = *std::max_element(ns.begin(), ns.end());
  std::vector<uint32_t> ids;
  while (ids.size() < need) ids.insert(ids.end(), source.begin(), source.end());
  ids.resize(need);

  const std::string snapshot = loader::resolve_snapshot(argv[1]);
  l0::Context ctx(0);
  runtime::Engine engine(ctx, loader::load(ctx, snapshot, kMaxLen), kMaxLen);
  engine.prepare_prefill();
  std::printf("# probe_writethrough: %s, max_len %u, backend default, copy every %u\n",
              ctx.name().c_str(), kMaxLen, every);

  // Pinned host store: one state buffer (overwritten per call, as 7c's per-call entries
  // cost the same copy) and the KV of the whole prompt, each range at its own offset.
  l0::Mem state(ctx, l0::MemKind::Host, engine.state_bytes());
  l0::Mem kv(ctx, l0::MemKind::Host, engine.kv_bytes(need));
  uint32_t prev = 0;
  int copies = 0;
  double hook_ms = 0.0;
  const runtime::Engine::BlockHook hook = [&](uint32_t end, bool block) {
    if (block && end % every != 0) return;   // --every 4096: skip the odd block ends
    Timer t;
    t.start();
    engine.save_state(state.ptr());
    engine.save_kv(prev, end, kv.as<uint8_t>() + engine.kv_bytes(prev));
    hook_ms += t.ms();
    prev = end;
    ++copies;
  };

  const auto run = [&](uint32_t n, bool hooked) {
    engine.reset();
    engine.set_block_hook(hooked ? hook : runtime::Engine::BlockHook{});
    prev = 0;
    copies = 0;
    hook_ms = 0.0;
    const std::vector<uint32_t> prompt(ids.begin(), ids.begin() + n);
    Timer timer;
    timer.start();
    engine.prefill(prompt);
    const double ms = timer.ms();
    if (engine.pos() != n) std::fprintf(stderr, "pos %u != %u\n", engine.pos(), n);
    engine.set_block_hook({});
    return ms;
  };

  for (uint32_t n : ns) {
    std::vector<double> ratios, plain_ms, hooked_ms, inside;
    for (int p = 0; p <= pairs; ++p) {   // p == 0 is the warm-up pair
      double a = 0, b = 0;
      if (p % 2 == 0) {
        a = run(n, false);
        b = run(n, true);
      } else {
        b = run(n, true);
        const double h = hook_ms;
        const int c = copies;
        a = run(n, false);
        hook_ms = h;
        copies = c;
      }
      std::printf("n=%u pair=%d%s plain_ms=%.1f hooked_ms=%.1f ratio=%.4f copies=%d hook_ms=%.1f\n",
                  n, p, p == 0 ? " (warm-up)" : "", a, b, b / a, copies, hook_ms);
      std::fflush(stdout);
      if (p == 0) continue;
      ratios.push_back(b / a);
      plain_ms.push_back(a);
      hooked_ms.push_back(b);
      inside.push_back(hook_ms);
    }
    std::printf("RESULT n=%u every=%u plain_median_ms=%.1f hooked_median_ms=%.1f"
                " median_ratio=%.4f cost=%.2f%% hook_median_ms=%.1f\n",
                n, every, median(plain_ms), median(hooked_ms), median(ratios),
                (median(ratios) - 1.0) * 100.0, median(inside));
    std::fflush(stdout);
  }
}

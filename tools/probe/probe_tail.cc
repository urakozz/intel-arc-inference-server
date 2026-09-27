// Spec 7 P0: the tail-prefill floor at depth. Load the model at max_len 131072,
// prefill N ids (long32k.ids repeated to length), then time prefill of T more ids:
// the time to the first generated id of a cached turn whose restart point is N.
//   probe_tail <snapshot> <ids file> [--n 0,30000,60000] [--t 16,256,1024,2048]
// Each timed run is a fresh reset() + base prefill of N (untimed) + the timed tail;
// per N one warm-up run, then per T the median of 3.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "l0/context.h"
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: probe_tail <snapshot> <ids file> [--n a,b] [--t a,b]\n");
    return 2;
  }
  std::vector<uint32_t> ns = {0, 30000, 60000}, ts = {16, 256, 1024, 2048};
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--n") ns = list_arg(argv[i + 1]);
    if (flag == "--t") ts = list_arg(argv[i + 1]);
  }
  constexpr uint32_t kMaxLen = 131072;
  const std::vector<uint32_t> source = read_ids(argv[2]);
  if (source.empty()) {
    std::fprintf(stderr, "no ids in %s\n", argv[2]);
    return 2;
  }
  const uint32_t need = *std::max_element(ns.begin(), ns.end()) + *std::max_element(ts.begin(), ts.end());
  std::vector<uint32_t> ids;
  while (ids.size() < need) ids.insert(ids.end(), source.begin(), source.end());
  ids.resize(need);

  const std::string snapshot = loader::resolve_snapshot(argv[1]);
  l0::Context ctx(0);
  runtime::Engine engine(ctx, loader::load(ctx, snapshot, kMaxLen), kMaxLen);
  engine.prepare_prefill();
  std::printf("# probe_tail: %s, max_len %u, backend default, source %zu ids\n", ctx.name().c_str(),
              kMaxLen, source.size());

  const auto run = [&](uint32_t n, uint32_t t) {
    engine.reset();
    if (n > 0) engine.prefill(std::vector<uint32_t>(ids.begin(), ids.begin() + n));
    const std::vector<uint32_t> tail(ids.begin() + n, ids.begin() + n + t);
    Timer timer;
    timer.start();
    engine.prefill(tail);
    const uint32_t pos = engine.pos();
    const double ms = timer.ms();
    if (pos != n + t) std::fprintf(stderr, "pos %u != %u\n", pos, n + t);
    return ms;
  };

  std::printf("\n| N (restart point) | T (tail) | runs ms | median ms | tail t/s |\n|---|---|---|---|---|\n");
  for (uint32_t n : ns) {
    (void)run(n, ts.front());  // warm-up at this depth
    for (uint32_t t : ts) {
      std::vector<double> ms;
      for (int r = 0; r < 3; ++r) ms.push_back(run(n, t));
      std::vector<double> sorted = ms;
      std::sort(sorted.begin(), sorted.end());
      std::printf("| %u | %u | %.1f, %.1f, %.1f | %.1f | %.0f |\n", n, t, ms[0], ms[1], ms[2], sorted[1],
                  t / (sorted[1] / 1000.0));
      std::fflush(stdout);
    }
  }
  return 0;
}

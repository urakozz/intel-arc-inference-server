// probe_turn_chunks - 3l (2): the whole-chunk wall time of one short turn deep in a session, the split-K
// prefill attention (runtime::prefill::FlashSplit::Auto, the engine's rule) against today's one-pass launch
// (FlashSplit::Off, the test seam), on one engine.
//
//   probe_turn_chunks SNAPSHOT KV [depths=2048,8192,32768,131072] [rows=16,64,128,2048] [rounds=3]
//                     [max_len=133376]
//
// tests/golden/prompts/long32k.ids tiled is prefilled (the engine's own chunks) to each depth in turn; the
// depth's state is saved (Engine::save_state) outside timing. Per (depth, rows): a turn of the next `rows` ids
// is ONE Engine::prefill call (chunk = rows) - every launch of the chunk, the head and the waits - timed on
// the host clock. Before every timed call the state is restored (load_state at the depth: the turn's own KV
// rows are rewritten identically). Two warm-up calls an arm, then `rounds` rounds of one pair each, the arm
// order alternating by round. Printed: each arm's median, the median paired gain (1 - auto / off) and its
// range, and whether the rule split the chunk (runtime::prefill::flash_split_span). int8 head, no MTP.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/model_desc.h"
#include "runtime/engine.h"
#include "runtime/prefill/attn.h"

namespace {
using Clock = std::chrono::steady_clock;
using runtime::prefill::FlashSplit;

std::vector<uint32_t> list_of(const std::string& s) {
  std::vector<uint32_t> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) v.push_back(uint32_t(std::stoul(t)));
  return v;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 7)
    throw std::runtime_error("usage: probe_turn_chunks SNAPSHOT KV [depths] [rows] [rounds] [max_len]");
  const auto kv = runtime::parse_kv_cache(argv[2]);
  const auto depths = list_of(argc > 3 ? argv[3] : "2048,8192,32768,131072");
  const auto rowsv = list_of(argc > 4 ? argv[4] : "16,64,128,2048");
  const uint32_t rounds = argc > 5 ? std::stoul(argv[5]) : 3;
  const uint32_t max_len = argc > 6 ? std::stoul(argv[6]) : 133376;
  std::ifstream input("tests/golden/prompts/long32k.ids");
  std::vector<uint32_t> base;
  for (uint32_t id; input >> id;) base.push_back(id);
  if (base.empty()) throw std::runtime_error("missing tests/golden/prompts/long32k.ids");
  l0::Context ctx(0);
  runtime::Engine engine(ctx, loader::load(ctx, argv[1], max_len, false, loader::LmHeadForm::Int8), max_len,
                         false, kv);
  engine.prepare_prefill();
  const model::ModelDesc& d = *engine.model().desc;
  l0::Mem state(ctx, l0::MemKind::Host, engine.state_bytes());
  std::printf("# 3l (2) probe_turn_chunks kv=%s max_len=%u rounds=%u\n", argv[2], max_len, rounds);
  for (uint32_t depth : depths) {
    std::vector<uint32_t> ids;
    for (uint32_t p = engine.pos(); p < depth; ++p) ids.push_back(base[p % base.size()]);
    runtime::prefill::set_flash_split_for_test(FlashSplit::Off);   // the prefix as today, every arm alike
    if (!ids.empty()) engine.prefill(ids);
    if (engine.pos() != depth) throw std::runtime_error("unexpected depth");
    engine.save_state(state.ptr());
    for (uint32_t C : rowsv) {
      if (depth + C > max_len) throw std::runtime_error("max_len too small");
      std::vector<uint32_t> turn;
      for (uint32_t p = depth; p < depth + C; ++p) turn.push_back(base[p % base.size()]);
      const FlashSplit arms[2] = {FlashSplit::Off, FlashSplit::Auto};
      auto once = [&](FlashSplit m) {
        engine.load_state(state.ptr(), depth);
        runtime::prefill::set_flash_split_for_test(m);
        const auto t0 = Clock::now();
        engine.prefill(turn, C);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        runtime::prefill::set_flash_split_for_test(FlashSplit::Off);
        return ms;
      };
      for (int w = 0; w < 2; ++w)
        for (FlashSplit m : arms) once(m);
      std::vector<double> off, on, gain;
      for (uint32_t r = 0; r < rounds; ++r) {
        double t[2];
        for (int i = 0; i < 2; ++i) {
          const int a = (int(r) + i) % 2;
          t[a] = once(arms[a]);
        }
        off.push_back(t[0]);
        on.push_back(t[1]);
        gain.push_back(100.0 * (1.0 - t[1] / t[0]));
      }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
      runtime::prefill::set_flash_split_for_test(FlashSplit::Auto);
      const uint32_t span = runtime::prefill::flash_split_span(d, depth, C);
      runtime::prefill::set_flash_split_for_test(FlashSplit::Off);
      std::printf("turn kv=%s depth=%u rows=%u split=%u spans=%u off_ms=%.3f auto_ms=%.3f gain_pct=%.2f [%.2f, %.2f]"
                  " rounds_faster=%zu/%u\n",
                  argv[2], depth, C, span, runtime::prefill::flash_split_count(depth, span), med(off), med(on),
                  med(gain), *std::min_element(gain.begin(), gain.end()), *std::max_element(gain.begin(), gain.end()),
                  size_t(std::count_if(gain.begin(), gain.end(), [](double g) { return g > 0; })), rounds);
      std::fflush(stdout);
    }
    engine.load_state(state.ptr(), depth);
  }
  std::puts("probe_turn_chunks: done");
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "probe_turn_chunks: %s\n", e.what());
  return 1;
}

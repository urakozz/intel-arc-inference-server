#pragma once
// Spec 17b: `b70-decode --parallel tp` - Qwen3.8 with every layer split over two B70s (runtime::TpEngine).
// b70_decode.cc validates every flag first (cli/pipeline_args.h check_tp) and hands over here before it opens a
// device:
//
//   1. both cards in one Level Zero context (GPUs 0 and 1 of what the driver shows) and peer access both ways -
//      refused before any weight is read (T4);
//   2. each card loads its own slice straight from the checkpoint (loader::load_tp; no whole-model load first),
//      at the asked max_len, or at a small one under --max-len auto;
//   3. --max-len auto: the largest length both cards fit (runtime::tp_max_len_that_fits over the loaded bytes),
//      then both RoPE tables re-tabled; an explicit length is held to each card's plan;
//   4. the prompt by decode replays, or (spec 17c, --prefill / --bench --prefill-length) by TpEngine::prefill - the
//      wavefront of both cards' prefill walks with the crossings on the copy engines; then generate (--ids) or time
//      --tg ids (--bench; with --prefill-length the pp row too, one card's columns);
//   5. spec 17d: --spec mtp / auto (--ids): the MTP head loaded whole on each card, planned on each, and the ids by
//      b70-decode's greedy speculative loop (generate_mtp over TpEngine's draft / verify / commit).
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/trained_context.h"
#include "runtime/memory_plan.h"
#include "runtime/tp_engine.h"
#include "runtime/tp_plan.h"

namespace cli {

struct TpDecodeArgs {
  std::string path;
  std::vector<uint32_t> ids;   // the prompt (--ids) or the synthetic depth (--bench)
  bool bench = false;
  uint32_t n = 0, depth = 0, tg = 0;
  MaxLenArg max_len;
  uint32_t trained = 0;        // config.json's context, read before the load for auto
  size_t reserve = 0;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  runtime::KvCache kv = runtime::KvCache::Bf16;
  std::function<void(uint32_t)> check_len;   // b70-decode's prompt + n <= max_len bound
  model::TpCrossing crossing = runtime::kTpDefaultCrossing;
  bool prefill = false;         // spec 17c: the prompt by prefill
  bool bench_prefill = false;   // --prefill-length: print the pp row
  uint32_t chunk = 0;           // --prefill-chunk (0: the cap)
  bool mtp = false;             // spec 17d: --spec mtp / auto
  std::function<int(runtime::TpEngine&)> run_mtp;   // b70_decode.cc's generate_mtp
};

template <class Redirect>
int run_tp_decode(const TpDecodeArgs& a) {
  require_two_devices_for_tp(l0::Context::gpu_count());
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--parallel tp: one context]\n", d0.name().c_str(),
               d0.eu_count(), d1.name().c_str(), d1.eu_count());
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0))
    throw std::runtime_error(
        "--parallel tp: the cards cannot access each other's memory (zeDeviceCanAccessPeer 0 -> 1 and 1 -> 0), and "
        "every crossing writes the other card's VRAM. Peer access needs the P2P-capable kernel and both cards under "
        "one root complex (docs/10-the-box.md)");

  // Spec 17c: the prefill's chunk cap (2048, or 4096 when --prefill-chunk asks for more) is planned on each card.
  runtime::PrefillPath path;
  path.prefill = a.prefill;
  path.chunk_cap = a.chunk > runtime::PrefillScratchDims::kCDefault ? runtime::PrefillScratchDims::kCMax
                                                                     : runtime::PrefillScratchDims::kCDefault;
  const uint32_t load_len = cli::load_len(a.max_len, a.trained);
  std::array<loader::LoadedModel, 2> cards = [&] {
    Redirect redirect;
    return std::array<loader::LoadedModel, 2>{
        loader::load_tp(d0, a.path, loader::TpSlice{0, a.crossing}, load_len, a.lm_head, a.mtp),
        loader::load_tp(d1, a.path, loader::TpSlice{1, a.crossing}, load_len, a.lm_head, a.mtp)};
  }();
  const model::ModelDesc& card = *cards[0].desc;
  const std::array<size_t, 2> weights{cards[0].report.total() - cards[0].report.rope_bytes,
                                      cards[1].report.total() - cards[1].report.rope_bytes};
  const std::array<size_t, 2> dev{d0.memory_bytes(), d1.memory_bytes()};
  uint32_t len = a.max_len.value;
  if (a.max_len.is_auto) {
    len = runtime::tp_max_len_that_fits(card, a.crossing, weights, dev, a.reserve, cards[0].trained_max_len, a.kv,
                                        path, a.mtp);
    if (len == 0)
      throw std::runtime_error("--parallel tp --max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
                               " positions fit both cards beside the weights");
    for (uint32_t r = 0; r < 2; ++r) loader::set_max_len(r == 0 ? d0 : d1, cards[r], len);
  } else {
    for (uint32_t r = 0; r < 2; ++r) {
      const runtime::MemoryPlan p = runtime::tp_card_plan(card, a.crossing, len, weights[r], a.kv, path, a.mtp);
      if (p.total() + a.reserve > dev[r])
        throw std::runtime_error("--parallel tp --max-len " + std::to_string(len) + " does not fit card " +
                                 std::to_string(r) + ": " + runtime::describe(p, dev[r], a.reserve) +
                                 " - --max-len auto gives the largest that fits");
    }
  }
  a.check_len(len);

  runtime::TpEngine eng(d0, d1, std::move(cards), len, {}, a.kv);
  std::fprintf(stderr,
               "engine: --parallel tp, crossing %s; %zu launches a card (%zu modules / %zu), max_len %u, kv cache %s%s\n",
               model::tp_crossing_name(eng.crossing()), eng.step(0).kernel_count, eng.step(0).modules.size(),
               eng.step(1).modules.size(), eng.max_len(), runtime::kv_cache_name(eng.kv_cache()),
               eng.mtp() ? ", MTP head on both cards (verify lists M = 1..4, draft lists on each card's vocabulary half)"
                         : "");
  if (a.prefill) {   // outside the timed window, as one card's prepare_prefill
    eng.set_prefill_chunk_cap(path.chunk_cap);
    eng.prepare_prefill();
    std::fprintf(stderr, "prefill backend: l0, chunk cap %u, crossings %s\n", eng.prefill_chunk_cap(),
                 eng.prefill_overlaps() ? "on the copy engines (the wavefront)" : "on the compute lists (no overlap)");
  }
  for (uint32_t r = 0; r < 2; ++r) std::fprintf(stderr, "%s\n", eng.memory_line(r).c_str());

  const size_t launches0 = eng.prefill_launches();
  const auto t0 = std::chrono::steady_clock::now();
  if (a.prefill) eng.prefill(a.ids, a.chunk);
  else eng.ingest(a.ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (a.prefill)
    std::fprintf(stderr,
                 "pp: %zu ids in %.1f ms (%.2f t/s) -- both cards, first prefill launch to the first generated id in "
                 "cur_token; chunk %u, %zu L0 launches (both cards), pos %u\n",
                 a.ids.size(), ingest_ms, ingest_ms > 0.0 ? double(a.ids.size()) * 1000.0 / ingest_ms : 0.0,
                 a.chunk ? a.chunk : eng.prefill_chunk_cap(), eng.prefill_launches() - launches0, eng.pos());
  else
    std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token), pos %u\n", a.ids.size(), ingest_ms,
                 a.ids.empty() ? 0.0 : ingest_ms / double(a.ids.size()), eng.pos());
  const auto report = [&](uint32_t n) {
    std::fprintf(stderr, "generate: %u ids, %.2f t/s, %.2f ms/token (%.1f%% of it inside the fences)\n", n,
                 eng.last_tok_per_s(), n ? eng.last_gen_ms() / n : 0.0,
                 eng.last_gen_ms() > 0.0 ? 100.0 * eng.last_fence_ms() / eng.last_gen_ms() : 0.0);
  };
  if (!a.bench && a.mtp) return a.run_mtp(eng);   // spec 17d
  if (!a.bench) {
    eng.generate(a.n, [](uint32_t id) {
      std::printf("%u\n", id);
      std::fflush(stdout);
    });
    report(a.n);
    return 0;
  }
  eng.generate(a.tg);
  report(a.tg);
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  // The single-card row's columns; the tag names the configuration, so interleaved arms sort apart.
  const std::string tags = std::string(a.lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "") +
                           (a.kv == runtime::KvCache::Int8 ? " int8-kv" : "") + " tp2-" +
                           model::tp_crossing_name(eng.crossing());
  std::printf("| b70-decode %s%s | %u | %u | %.2f | %.2f |\n", sha, tags.c_str(), a.depth, a.tg, eng.last_tok_per_s(),
              eng.last_gen_ms() / double(a.tg));
  if (a.bench_prefill)   // spec 17c: one card's pp row, the configuration in the tag
    std::printf("| b70-decode %s%s l0 pp | %u | %u | %.1f | %.2f |\n", sha, tags.c_str(), a.depth,
                a.chunk ? a.chunk : eng.prefill_chunk_cap(), ingest_ms,
                ingest_ms > 0.0 ? double(a.ids.size()) * 1000.0 / ingest_ms : 0.0);
  std::fprintf(stderr,
               "(measured, one run. docs/BENCHMARKS.md records the median of three on an idle box, interleaved with\n"
               " the one-card row.)\n");
  return 0;
}

}  // namespace cli

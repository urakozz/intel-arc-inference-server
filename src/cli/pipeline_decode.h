#pragma once
// Spec 16b: `b70-decode --pp 2` - the model's layers over two B70s (runtime::PipelineEngine);
// spec 16c: and the prompt's prefill through the two-card chunk pipeline. b70_decode.cc validates every flag first (cli/pipeline_args.h)
// and hands over here before it opens a device; the flow is b70-decode's own, per device:
//
//   1. both devices in one Level Zero context (GPUs 0 and 1 of what the driver shows), and
//      peer access from device 0 to device 1 - refused before the 18 GB load (P4);
//   2. the whole checkpoint loaded onto device 0, as one card loads it;
//   3. the split and max_len: the planner over the LOADED weights (runtime::pp_weights),
//      each device's total + the reserve held to that device's memory; auto picks the
//      split that balances bytes and the length both devices fit;
//   4. layers [s, L), the final norm and lm_head moved to device 1 (runtime::place_stages),
//      the RoPE table re-tabled on both if auto changed the length;
//   5. the prompt: with --prefill / --bench --prefill-length the two-card prefill (spec 16c;
//      each device's prefill scratch planned in step 3), else one decode replay per id; then
//      generate (--ids) or time --tg ids (--bench).
//
// Spec 16d: steps 3 and 4 are cli/pipeline_settle.h's (shared with b70-serve --pp 2), and
// --mtp K|auto (--ids, greedy) loads the head, plans it on device 1 (PpExtras) and runs
// b70-decode's own speculative loop over the pipeline (`run_mtp`).
#include <algorithm>
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
#include "cli/pipeline_settle.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"

namespace cli {

struct PipelineDecodeArgs {
  std::string path;
  std::vector<uint32_t> ids;   // the prompt (--ids) or the synthetic depth (--bench)
  bool bench = false;
  uint32_t n = 0, depth = 0, tg = 0;
  MaxLenArg max_len;
  uint32_t trained = 0;        // config.json's context, read before the load for auto
  size_t reserve = 0;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  runtime::KvCache kv = runtime::KvCache::Bf16;
  PipelineArgs pipe;
  std::function<void(uint32_t)> check_len;   // b70-decode's prompt + n <= max_len bound
  // Spec 16c: --prefill (the prompt through the chunk pipeline) or --bench --prefill-length N
  // (the timed prefill of `depth` synthetic ids, the pp row); `pf` is what both run on and
  // what the plan counts, `chunk` --prefill-chunk (0: PrefillScratch::kC).
  bool prefill = false, bench_prefill = false;
  uint32_t chunk = 0;
  runtime::PrefillPath pf = runtime::pp_no_prefill();
  // Spec 16d: --mtp K|auto (--ids only): the head is loaded and placed on device 1, and
  // run_mtp (b70-decode's greedy speculative loop) generates instead of the plain loop.
  bool mtp = false;
  std::function<int(runtime::PipelineEngine&)> run_mtp;
};

template <class Redirect>
int run_pipeline_decode(const PipelineDecodeArgs& a) {
  require_two_devices(l0::Context::gpu_count());
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--pp 2: one context]\n",
               d0.name().c_str(), d0.eu_count(), d1.name().c_str(), d1.eu_count());
  if (!d0.can_access_peer(d1))
    throw std::runtime_error(
        "--pp 2: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer 0 -> 1 "
        "is false), and both hand-offs write it. Peer access needs the P2P-capable kernel and "
        "both cards under one root complex (docs/10-the-box.md)");

  loader::LoadedModel full = [&] {
    Redirect redirect;
    return loader::load(d0, a.path, load_len(a.max_len, a.trained), /*mtp=*/a.mtp, a.lm_head);
  }();
  const model::ModelDesc& d = *full.desc;
  const runtime::PpWeights w = runtime::pp_weights(full);
  const std::array<size_t, runtime::kPpDevices> dev{d0.memory_bytes(), d1.memory_bytes()};
  // Steps 3 and 4 (cli/pipeline_settle.h): no hook here - b70-decode has no prefix cache.
  const PpSettled settled = pp_settle(d, w, dev, a.max_len, full.trained_max_len, a.reserve, a.kv,
                                      a.pf, pp_extras(full, /*hook=*/false), a.pipe);
  const uint32_t split = settled.split, len = settled.max_len;
  a.check_len(len);

  std::vector<loader::LoadedModel> stages = runtime::place_stages(d0, d1, std::move(full), split);
  if (stages[0].max_len != len) {   // auto loaded at a small length: re-table both
    loader::set_max_len(d0, stages[0], len);
    loader::set_max_len(d1, stages[1], len);
  }
  runtime::PipelineOptions opt;
  opt.handoff = a.pipe.handoff;
  runtime::PipelineEngine eng(d0, d1, std::move(stages), len, opt, a.kv);
  std::fprintf(stderr,
               "engine: --pp 2, hand-off %s, split %u; device 0 layers [0, %u): %zu launches,"
               " %zu modules; device 1 layers [%u, %u): %zu launches, %zu modules; max_len %u,"
               " decode attention %s, kv cache %s\n",
               runtime::pp_handoff_name(eng.handoff()), eng.split(), eng.split(),
               eng.step(0).kernel_count, eng.step(0).modules.size(), eng.stage(1).first,
               eng.stage(1).last, eng.step(1).kernel_count, eng.step(1).modules.size(),
               eng.max_len(), runtime::decode_attn_name(runtime::decode_attn()),
               runtime::kv_cache_name(eng.kv_cache()));
  if (eng.mtp())   // spec 16d: the head on device 1, the verify lists on both
    std::fprintf(stderr,
                 "mtp: the head on device 1; verify lists M = 1..%u (%zu + %zu launches at every M),"
                 " %u draft lists of %zu launches on device 1\n",
                 runtime::MtpBuffers::kSlots, eng.verify_step(0, 1).kernel_count,
                 eng.verify_step(1, 1).kernel_count, runtime::PipelineEngine::kMaxDraft,
                 eng.draft_step(0).kernel_count);
  // Spec 16c: the prefill set up before the window (each device's scratch and context, the
  // l0-int8 column scales), so the memory lines below hold it, as one card's do.
  const bool prefill = a.prefill || a.bench_prefill;
  if (prefill) {
    eng.set_prefill_backend(a.pf.backend);
    std::fprintf(stderr, "prefill backend: %s, both devices (spec 16c: the chunk pipeline)\n",
                 runtime::prefill_backend_name(eng.prefill_backend()));
    eng.prepare_prefill();
  }
  for (uint32_t i = 0; i < runtime::kPpDevices; ++i)
    std::fprintf(stderr, "%s\n", eng.memory_line(i).c_str());

  const auto t0 = std::chrono::steady_clock::now();
  if (prefill)
    eng.prefill(a.ids, a.chunk);
  else
    eng.ingest(a.ids);
  const double ingest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const uint32_t chunk = a.chunk ? a.chunk : runtime::PrefillScratch::kC;
  if (prefill) {
    const runtime::PipelineEngine::PrefillStats& st = eng.last_prefill();
    std::fprintf(stderr,
                 "pp: %zu ids in %.1f ms (%.2f t/s) -- --pp 2, %u chunks of %u, the first "
                 "prefill launch to the first generated id in cur_token; pos %u\n",
                 a.ids.size(), ingest_ms,
                 ingest_ms > 0.0 ? double(a.ids.size()) * 1000.0 / ingest_ms : 0.0, st.chunks,
                 chunk, eng.pos());
    // Plan 16c Review Focus 5: each card's busy time (its chunks' walks, device timestamps)
    // against the wall - the pipeline's rate is the slower card's.
    for (uint32_t i = 0; i < runtime::kPpDevices; ++i)
      std::fprintf(stderr,
                   "pp: device %u busy %.1f ms (%.1f%% of the wall), %.2f ms per chunk, the "
                   "longest %.2f ms; %zu launches (%zu expected)%s\n",
                   i, st.busy_ms[i], st.wall_ms > 0.0 ? 100.0 * st.busy_ms[i] / st.wall_ms : 0.0,
                   st.chunks ? st.busy_ms[i] / st.chunks : 0.0, st.busy_max_ms[i], st.launches[i],
                   st.expected_launches[i], i == 1 ? " (with the head)" : "");
  } else {
    std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token), pos %u\n", a.ids.size(),
                 ingest_ms, a.ids.empty() ? 0.0 : ingest_ms / double(a.ids.size()), eng.pos());
  }

  const auto report = [&](uint32_t n) {
    std::fprintf(stderr,
                 "generate: %u ids, %.2f t/s, %.2f ms/token (%.1f%% of it inside the fences)\n", n,
                 eng.last_tok_per_s(), n ? eng.last_gen_ms() / n : 0.0,
                 eng.last_gen_ms() > 0.0 ? 100.0 * eng.last_fence_ms() / eng.last_gen_ms() : 0.0);
  };
  if (!a.bench && a.mtp && a.run_mtp) return a.run_mtp(eng);   // spec 16d
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
  const double ms_per_token = eng.last_gen_ms() / double(a.tg);
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  // The single-card row's columns; the tag names the configuration, so the rows of an
  // interleaved one-card / --pp 2 pair sort apart (spec 16b S1).
  const std::string tags = std::string(a.lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "") +
                           (a.kv == runtime::KvCache::Int8 ? " int8-kv" : "") + " pipeline2-" +
                           runtime::pp_handoff_name(eng.handoff()) + "-s" +
                           std::to_string(eng.split());
  std::printf("| b70-decode %s%s | %u | %u | %.2f | %.2f |\n", sha, tags.c_str(), a.depth, a.tg,
              eng.last_tok_per_s(), ms_per_token);
  // Spec 16c: the pp row, one card's columns (tools/box_validate/summary.py reads a field
  // ending in " pp"), the configuration in the tag.
  if (a.bench_prefill)
    std::printf("| b70-decode %s%s %s pp | %u | %u | %.1f | %.2f |\n", sha, tags.c_str(),
                runtime::prefill_backend_name(eng.prefill_backend()), a.depth, chunk, ingest_ms,
                ingest_ms > 0.0 ? double(a.ids.size()) * 1000.0 / ingest_ms : 0.0);
  std::fprintf(stderr,
               "(measured, one run. docs/BENCHMARKS.md records the median of three on an idle box;\n"
               " spec 16b S1 compares it with the one-card row, interleaved.)\n");
  return 0;
}

}  // namespace cli

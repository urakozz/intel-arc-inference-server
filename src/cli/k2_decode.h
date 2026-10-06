#pragma once
// Spec 18b: b70-decode's K2-Horizon path - the CLI dispatches on config.json's model_type
// (loader::is_k2_checkpoint) and, for "k2_horizon", runs runtime::k2::K2Engine instead of
// runtime::Engine. `--ids` and `--bench` exactly as the Qwen path prints them (ids on stdout,
// one per line; the bench row on stdout); the prompt goes through the decode list one replay
// per id, or (spec 18c, `--prefill` / `--prefill-length N`) through K2Engine::prefill in
// chunks of `--prefill-chunk` (default kPfC = 2048) on the l0 backend. `--max-len auto|N` plans
// with K2's own planner (runtime/k2/k2_sizes.h) under the trained context (524288), the
// prefill scratch included when this run prefills (it is lazy on the engine, as Qwen's ruling
// R7). Spec 18e:
// `--kv-cache int8` (or B70_KV_CACHE=int8) plans and builds the int8 rotkv cache.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/max_len.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "loader/lm_head_int8.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"
#include "runtime/k2/k2_engine.h"
#include "runtime/k2/k2_prefill.h"   // spec 18c: prefill_attn_eager (b70_k2_prefill)
#include "runtime/k2/k2_sizes.h"

namespace cli::k2 {

// True when `path` resolves to a snapshot whose config.json says model_type "k2_horizon".
// A path that does not resolve is not K2: the caller's own flow then reports it (so every
// argument rejection keeps its message and its order).
inline bool is_k2(const std::string& path) {
  try {
    return loader::is_k2_checkpoint(loader::resolve_snapshot(path));
  } catch (const std::exception&) {
    return false;
  }
}

struct DecodeArgs {
  std::string path, ids_path;
  uint32_t n = 0, depth = 4096, tg = 256;
  bool bench = false;
  MaxLenArg max_len{false, 16384};
  size_t reserve = 0;
  uint32_t device = l0::Context::kFromEnv;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  const uint32_t* bench_prompt = nullptr;   // cycled to --depth (b70_decode.cc's kBenchPrompt)
  size_t bench_prompt_len = 0;
  // Spec 18c: `--prefill` (ingest by K2Engine::prefill) or `--prefill-length N` (--bench at
  // depth N, prefilled and timed - the pp row); pp_chunk 0 = kPfC.
  bool prefill = false, pp = false;
  uint32_t pp_chunk = 0;
  // Spec 18e: `--kv-cache bf16|int8` (B70_KV_CACHE's default) - planned and built in that form.
  runtime::KvCache kv = runtime::KvCache::Bf16;
};

inline std::vector<uint32_t> read_ids(const std::string& path, uint32_t vocab) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --ids file '" + path + "'");
  std::vector<uint32_t> ids;
  std::string w;
  while (f >> w) {
    if (w.empty() || w.find_first_not_of("0123456789") != std::string::npos || w.size() > 10)
      throw std::runtime_error("--ids file '" + path + "' is not whitespace-separated ids");
    const unsigned long long v = std::stoull(w);
    if (v >= vocab)
      throw std::runtime_error("--ids file '" + path + "': id " + w + " is outside K2-Horizon's " +
                               "vocabulary (" + std::to_string(vocab) + " rows)");
    ids.push_back(uint32_t(v));
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

// cli::settle for K2: auto plans the largest max_len that fits and re-tables the model; an
// explicit N is held to the same plan. `prefill` (spec 18c): the plan carries the prefill
// scratch this run will allocate. `kv` (spec 18e): the KV term in the cache's form.
inline uint32_t settle(l0::Context& ctx, loader::K2LoadedModel& m, const MaxLenArg& a, size_t reserve,
                       bool prefill = false, runtime::KvCache kv = runtime::KvCache::Bf16) {
  const model::K2Desc& d = *m.desc;
  const size_t weights = m.report.bytes.total(), device = ctx.memory_bytes();
  const auto plan_at = [&](uint32_t n) {
    return runtime::k2::plan(d, n, weights, false, prefill, runtime::k2::k2_attn(), kv);
  };
  uint32_t len = a.value;
  if (a.is_auto) {
    const uint32_t fit =
        runtime::k2::max_len_that_fits(d, weights, device, reserve, m.trained_max_len, prefill, kv);
    if (fit == 0)
      throw std::runtime_error("--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
                               " positions fit - " +
                               runtime::k2::describe(plan_at(runtime::kMinAutoMaxLen), device, reserve));
    len = fit;
    loader::set_max_len_k2(ctx, m, len);
    std::fprintf(stderr, "max_len: auto -> %u (K2-Horizon, %s KV; the largest multiple of %u that fits "
                 "%.3f GB with a %.3f GB reserve; trained context %u)\n", len, runtime::kv_cache_name(kv),
                 runtime::kMaxLenQuantum, device / 1e9, reserve / 1e9, m.trained_max_len);
  } else {
    const runtime::k2::Plan p = plan_at(len);
    if (p.total() + reserve > device)
      throw std::runtime_error("--max-len " + std::to_string(len) + " does not fit: " +
                               runtime::k2::describe(p, device, reserve) + ". The largest that fits is " +
                               std::to_string(runtime::k2::max_len_that_fits(
                                   d, weights, device, reserve, m.trained_max_len ? m.trained_max_len : len,
                                   prefill, kv)) +
                               " (--max-len auto)");
    std::fprintf(stderr, "max_len: %u (--max-len)\n", len);
  }
  std::fprintf(stderr, "%s\n", runtime::k2::describe(plan_at(len), device, reserve).c_str());
  return len;
}

// The loader prints its report to stdout (its convention); `redirect` is the caller's
// StdoutToStderr so the report joins the diagnostics on stderr.
template <class Redirect>
int run_decode(const DecodeArgs& a) {
  std::vector<uint32_t> ids;
  if (!a.bench) ids = read_ids(a.ids_path, model::k2().vocab);
  uint32_t trained = 0;
  if (a.max_len.is_auto) {
    trained = loader::trained_context(loader::resolve_snapshot(a.path));
    check_before_load(a.max_len, trained, /*require_quantum=*/false);
  }
  const uint32_t after = a.bench ? a.tg : a.n;
  const size_t need = (a.bench ? size_t(a.depth) : ids.size()) + after;
  if (!a.max_len.is_auto && need > a.max_len.value)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) +
                             " generated ids exceeds --max-len " + std::to_string(a.max_len.value));
  if (a.bench) {
    ids.resize(a.depth);
    for (uint32_t i = 0; i < a.depth; ++i) ids[i] = a.bench_prompt[i % a.bench_prompt_len];
  }

  l0::Context ctx(a.device);
  std::fprintf(stderr, "device: %s (%u EUs)%s\n", ctx.name().c_str(), ctx.eu_count(),
               a.device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  loader::K2LoadedModel model = [&] {
    Redirect redirect;
    return loader::load_k2(ctx, a.path, load_len(a.max_len, trained), a.lm_head);
  }();
  const bool prefill = a.prefill || a.pp;
  const uint32_t max_len = settle(ctx, model, a.max_len, a.reserve, prefill, a.kv);
  if (need > max_len)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) +
                             " generated ids exceeds max_len " + std::to_string(max_len));
  runtime::k2::K2Engine eng(ctx, std::move(model), max_len, /*debug_tap=*/false, a.kv);
  std::fprintf(stderr, "engine: K2-Horizon, %zu kernels, %zu modules, max_len %u, %.2f GB of persistent "
               "state (%s KV%s)\n", eng.step().kernel_count, eng.step().modules.size(), eng.max_len(),
               eng.buffers().persistent_bytes() / 1e9, runtime::kv_cache_name(eng.kv_cache()),
               eng.kv_cache() == runtime::KvCache::Int8 ? ", rotkv at head_dim 128, spec 18e" : "");
  // Spec 18c: the prefill setup (scratch, Context, the binaries' check) happens here, outside
  // the timed window, which is then the whole prefill() call: first launch to the first
  // generated id in cur_token (argmax_stage2 is its last launch).
  size_t pp_launches0 = 0;
  if (prefill) {
    eng.prepare_prefill();
    pp_launches0 = eng.prefill_launches();
    std::fprintf(stderr, "prefill: l0 backend, chunk %u, attention %s (B70_K2_ATTN), %zu launches per "
                 "chunk + %zu for the head\n", a.pp_chunk ? a.pp_chunk : runtime::k2::kPfC,
                 runtime::k2::prefill_attn_eager() ? "eager" : "flash",
                 runtime::k2::prefill_chunk_launches(*eng.model().desc), runtime::k2::kPrefillHeadLaunches);
  }
  std::fprintf(stderr, "%s\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  if (prefill)
    eng.prefill(ids, a.pp_chunk);
  else
    eng.ingest(ids);
  const double ingest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (prefill)
    std::fprintf(stderr, "%s: %zu ids in %.1f ms (%.2f t/s) - device-side, loader excluded, first prefill "
                 "launch to the first generated id; chunk %u, %zu L0 launches, pos %u\n",
                 a.pp ? "pp" : "prefill", ids.size(), ingest_ms,
                 ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0,
                 a.pp_chunk ? a.pp_chunk : runtime::k2::kPfC, eng.prefill_launches() - pp_launches0, eng.pos());
  else
    std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token, one decode replay per id), pos %u\n",
                 ids.size(), ingest_ms, ingest_ms / double(ids.size()), eng.pos());
  const uint32_t n = a.bench ? a.tg : a.n;
  eng.generate(n, [&](uint32_t id) {
    if (a.bench) return;
    std::printf("%u\n", id);
    std::fflush(stdout);
  });
  std::fprintf(stderr, "generate: %u ids, %.2f t/s, %.2f ms/token (%.1f%% of it inside the fence)\n", n,
               eng.last_tok_per_s(), n ? eng.last_gen_ms() / n : 0.0,
               eng.last_gen_ms() > 0.0 ? 100.0 * eng.last_fence_ms() / eng.last_gen_ms() : 0.0);
  if (!a.bench) return 0;
  const double ms_per_token = eng.last_gen_ms() / double(a.tg);
  const double gb = double(eng.model().report.read_per_token) / 1e9;
  std::fprintf(stderr, "  MBU: %.2f t/s x %.3f GB = %.0f GB/s (W = this load's derived read/token: "
               "top-8 + shared MoE, top-4 value experts); %zu launches/token\n",
               eng.last_tok_per_s(), gb, eng.last_tok_per_s() * gb, eng.step().kernel_count);
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  // Spec 18e: an int8-KV run's rows carry `int8-kv`, as the Qwen path's do.
  const char* kvtag = eng.kv_cache() == runtime::KvCache::Int8 ? " int8-kv" : "";
  std::printf("| b70-decode %s k2%s%s | %u | %u | %.2f | %.2f |\n", sha,
              a.lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "", kvtag, a.depth, a.tg,
              eng.last_tok_per_s(), ms_per_token);
  // Spec 18c: the pp row, a second line as the Qwen path prints it (depth, chunk, ms, t/s).
  if (a.pp)
    std::printf("| b70-decode %s k2%s%s l0 pp | %u | %u | %.1f | %.2f |\n", sha,
                a.lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "", kvtag, a.depth,
                a.pp_chunk ? a.pp_chunk : runtime::k2::kPfC, ingest_ms,
                ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0);
  return 0;
}

}  // namespace cli::k2

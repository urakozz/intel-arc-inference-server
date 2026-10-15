#pragma once
// Spec 23c: b70-decode's Muse-Glimmer-30B path - the CLI dispatches on config.json's model_type
// (loader::is_muse_checkpoint) and, for "muse_glimmer", runs runtime::muse::MuseEngine on one card (the default,
// spec 23 decision 7) or two. `--ids` and `--bench` exactly as the Qwen path prints them (ids on stdout, one per
// line; the bench row on stdout); the prompt goes through the decode list one replay per id (the chunked prefill is
// spec 23d's).
//
//   --pp N / --pipeline-parallel-size N   1 (the default: 17.4 GB of weights with the int8 head, one card) or 2
//                      (capacity, not speed - decision 7)
//   --pipeline-split auto|N, --pipeline-handoff copy|peer   spec 16b's switches and parsers; auto is
//                      runtime::pp_balance over Muse's per-layer bytes at the session's max_len
//   --layers N         development mode: load only layers [0, N) of the checkpoint (the head reads layer N-1's residual)
//   --lm-head int8|bf16 (int8 the default, spec 9), --max-len N|auto (b70-decode's 16384 the default; at most
//                      131072, a multiple of 256)
// Refused before the device, each by name: --prefill / --prefill-length / --prefill-chunk / --prefill-backend
// (spec 23d - box queue row 45), --mtp (Muse has no MTP head), --spec (Muse's drafter is spec 23f's DFlash),
// --kv-cache int8 (decision 8: bf16 KV), --max-len above 131072 (decision 6), --profile (not built for Muse),
// --device N with --pp 2 (16b's rule: GPUs 0 and 1), --layers auto, --ple-dir.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/muse_loader.h"
#include "loader/snapshot.h"
#include "runtime/muse/muse_engine.h"
#include "runtime/muse/muse_sizes.h"

namespace cli::muse {

// The bench prompt, cycled to --depth: tests/golden/prompts/muse_short_prose.ids - the 42 ids of
// tests/golden/prompts/prose.txt through Muse-Glimmer's own tokenizer, no special tokens (23a's
// tools/oracle/muse_prompts.py; the golden set muse_short_prose's prompt).
inline constexpr uint32_t kMuseBenchPrompt[] = {
    954, 174266, 552, 96034, 812, 27591, 7177, 328, 17189, 290, 135587, 1417, 447, 5511,
    290, 154016, 3579, 26, 477, 132410, 64753, 10143, 436, 290, 2926, 15539, 2467, 290,
    98556, 185943, 1492, 38492, 26, 4719, 8115, 290, 4224, 1808, 446, 42204, 3357, 26};
inline constexpr size_t kMuseBenchPromptLen = sizeof(kMuseBenchPrompt) / sizeof(kMuseBenchPrompt[0]);

// True when `path` resolves to a snapshot whose config.json says model_type "muse_glimmer". A path that does not
// resolve is not Muse: the caller's own flow then reports it.
inline bool is_muse(const std::string& path) {
  try {
    return loader::is_muse_checkpoint(loader::resolve_muse_snapshot(path));   // either Vishva007 spelling
  } catch (const std::exception&) {
    return false;
  }
}

struct DecodeArgs {
  std::string path, ids_path;
  uint32_t n = 0, depth = 4096, tg = 256;
  bool bench = false;
  MaxLenArg max_len{false, 16384};   // b70-decode's default (bench rows comparable); auto plans the largest
  uint32_t device = l0::Context::kFromEnv;
  bool int8_head = true;        // --lm-head (int8 unless bf16 is asked for)
  uint32_t layers = 0;          // --layers N (0: the checkpoint's)
  PipelineArgs pipe;            // devices as parsed (1 when --pp was not given)
  // What else was asked for, for the refusals.
  bool prefill = false, prefill_len = false, prefill_chunk = false, prefill_backend = false;
  bool mtp = false, spec = false, kv8 = false, profile = false, layers_auto = false, ple_dir = false;
};

// Every refusal that needs no device - in this order, each by name (tests/CMakeLists.txt's cli_reject_muse_*). `d` is
// the checkpoint's descriptor (config.json only).
inline void check_args(const DecodeArgs& a, const model::MuseGlimmerDesc& d) {
  if (a.prefill || a.prefill_len || a.prefill_chunk || a.prefill_backend)
    throw std::runtime_error("Muse-Glimmer-30B has no prefill path yet: spec 23d builds it (box queue row 45) - feed the "
                             "prompt through the decode list (drop --prefill, --prefill-length, --prefill-chunk and "
                             "--prefill-backend)");
  if (a.mtp)
    throw std::runtime_error("Muse-Glimmer-30B has no MTP head (spec 23 §1); its drafter is the DFlash assistant, "
                             "spec 23f - drop --mtp");
  if (a.spec)
    throw std::runtime_error("--spec: Muse-Glimmer-30B's speculative decoding is the DFlash drafter of spec 23f "
                             "(--spec dflash, b70-serve) - not built yet; drop --spec");
  if (a.kv8)
    throw std::runtime_error("--kv-cache int8: Muse-Glimmer-30B's KV is bf16 only (spec 23 decision 8: 13,312 B a "
                             "position on its 13 full layers; int8 KV is recorded for later, §10)");
  if (!a.max_len.is_auto && a.max_len.value > d.max_pos)
    throw std::runtime_error("--max-len " + std::to_string(a.max_len.value) + " exceeds Muse-Glimmer-30B's context " +
                             std::to_string(d.max_pos) + " (max_position_embeddings) - spec 23 decision 6: above it is "
                             "refused");
  if (!a.max_len.is_auto && a.max_len.value % 256 != 0)
    throw std::runtime_error("--max-len " + std::to_string(a.max_len.value) + " is not a multiple of 256 (the planner's "
                             "grid, runtime::kMaxLenQuantum) - or --max-len auto");
  if (a.profile)
    throw std::runtime_error("--profile is not built for Muse-Glimmer-30B (spec 23c); --bench times it");
  if (a.layers_auto)
    throw std::runtime_error("--layers auto is Qwen3.8-Flash-Next's (spec 21c); Muse-Glimmer-30B takes --layers N");
  if (a.ple_dir) throw std::runtime_error("--ple-dir belongs to Qwen3.8-Flash-Next (spec 21c)");
  if (a.pipe.devices == 1) {
    if (a.pipe.have_split) throw std::runtime_error("--pipeline-split belongs to --pp 2 (the layer device 1 starts at)");
    if (a.pipe.have_handoff) throw std::runtime_error("--pipeline-handoff belongs to --pp 2 (how the residual crosses)");
    return;
  }
  if (a.device != l0::Context::kFromEnv)
    throw std::runtime_error("--pp 2 runs on GPUs 0 and 1 of what Level Zero shows (ZE_AFFINITY_MASK picks which "
                             "two); --device names one card - drop it");
  if (d.layers < 2)
    throw std::runtime_error("--pp 2 needs at least two layers to split (--layers " + std::to_string(d.layers) + ")");
  if (!a.pipe.split_auto && (a.pipe.split < 1 || a.pipe.split >= d.layers))
    throw std::runtime_error("--pipeline-split " + std::to_string(a.pipe.split) + " leaves a device no layer: it is the "
                             "first layer of device 1, in [1, " + std::to_string(d.layers - 1) + "]");
}

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
      throw std::runtime_error("--ids file '" + path + "': id " + w + " is outside Muse-Glimmer-30B's vocabulary (" +
                               std::to_string(vocab) + " rows)");
    ids.push_back(uint32_t(v));
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

// The loader prints its report to stdout (its convention); `Redirect` is the caller's StdoutToStderr so the report
// joins the diagnostics on stderr.
template <class Redirect>
int run_decode(const DecodeArgs& a) {
  const model::MuseGlimmerDesc pre = loader::muse_checkpoint_desc(a.path, a.layers);
  check_args(a, pre);
  const uint32_t devs = a.pipe.devices;
  std::vector<uint32_t> ids;
  if (!a.bench) {
    ids = read_ids(a.ids_path, pre.vocab);
  } else {
    ids.resize(a.depth);
    for (uint32_t i = 0; i < a.depth; ++i) ids[i] = kMuseBenchPrompt[i % kMuseBenchPromptLen];
  }
  const uint32_t after = a.bench ? a.tg : a.n;
  const size_t need = ids.size() + after;
  if (!a.max_len.is_auto && need > a.max_len.value)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds --max-len " +
                             std::to_string(a.max_len.value));
  if (need > pre.max_pos)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds Muse-Glimmer's "
                             "context " + std::to_string(pre.max_pos) + " (spec 23 decision 6)");
  if (devs == 2) require_two_devices(l0::Context::gpu_count());

  // The contexts: one card (--device / the environment), or GPUs 0 and 1 (device 1 a view of 0's).
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(devs == 2 ? 0u : a.device);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)%s\n", c0->name().c_str(), c0->eu_count(),
                 a.device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  else
    std::fprintf(stderr, "devices: %s + %s (--pp 2)\n", c0->name().c_str(), c1->name().c_str());

  runtime::muse::MuseOptions o;
  o.max_len = a.max_len.is_auto ? 0 : a.max_len.value;
  o.int8_head = a.int8_head;
  o.handoff = a.pipe.handoff;
  o.split = devs == 2 && !a.pipe.split_auto ? a.pipe.split : 0;
  o.layers = a.layers;
  std::unique_ptr<runtime::muse::MuseEngine> eng;
  {
    Redirect redirect;
    eng = std::make_unique<runtime::muse::MuseEngine>(ctx, a.path, o);
  }
  if (need > eng->max_len())
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds max_len " +
                             std::to_string(eng->max_len()));
  std::fprintf(stderr, "max_len: %s%u (decision 6: at most %u)\n", a.max_len.is_auto ? "auto -> " : "", eng->max_len(),
               pre.max_pos);
  if (devs == 2)
    std::fprintf(stderr, "split: %s%u (layers [0, %u) on device 0, [%u, %u) and the head on device 1)\n",
                 a.pipe.split_auto ? "auto -> " : "", eng->split(), eng->split(), eng->split(), eng->weights().desc.layers);
  if (a.layers)
    std::fprintf(stderr, "layers: %u of the checkpoint's %u (--layers: development mode; the head reads layer %u's "
                 "residual)\n", eng->weights().desc.layers, eng->weights().checkpoint_layers,
                 eng->weights().desc.layers - 1);
  std::fprintf(stderr, "%s\n", eng->plan().describe().c_str());
  std::fprintf(stderr, "engine: Muse-Glimmer-30B, %u device(s)%s, %zu launches a token (%s attention, B70_MUSE_ATTN; "
               "runtime::muse::decode_launches %zu), max_len %u, head %s\n", eng->devices(),
               devs == 2 ? (std::string(", split ") + std::to_string(eng->split()) + ", hand-off " +
                            runtime::pp_handoff_name(eng->handoff())).c_str() : "",
               eng->launches(), runtime::muse::muse_attn_name(eng->attention()),
               runtime::muse::decode_launches(eng->weights().desc, eng->weights().placement, eng->attention(),
                                              eng->handoff()),
               eng->max_len(), a.int8_head ? "int8" : "bf16");
  std::fprintf(stderr, "%s\n", eng->memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  eng->ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token, one decode replay per id), pos %u\n", ids.size(),
               ingest_ms, ingest_ms / double(ids.size()), eng->pos());
  const uint32_t n = a.bench ? a.tg : a.n;
  eng->generate(n, [&](uint32_t id) {
    if (a.bench) return;
    std::printf("%u\n", id);
    std::fflush(stdout);
  });
  std::fprintf(stderr, "generate: %u ids, %.2f t/s, %.2f ms/token (%.1f%% of it inside the fence)\n", n,
               eng->last_tok_per_s(), n ? eng->last_gen_ms() / n : 0.0,
               eng->last_gen_ms() > 0.0 ? 100.0 * eng->last_fence_ms() / eng->last_gen_ms() : 0.0);
  if (!a.bench) return 0;
  const double ms_per_token = eng->last_gen_ms() / double(a.tg);
  const double gb = double(eng->weights().read_per_token) / 1e9;
  std::fprintf(stderr, "  MBU: %.2f t/s x %.3f GB = %.0f GB/s (W = this load's derived read/token: the layers' int4 "
               "linears and norms, the head; the KV and windows not counted); %zu launches/token\n",
               eng->last_tok_per_s(), gb, eng->last_tok_per_s() * gb, eng->launches());
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode %s muse%s%s%s | %u | %u | %.2f | %.2f |\n", sha, a.int8_head ? " int8-head" : " bf16-head",
              eng->attention() == runtime::muse::MuseAttn::Eager ? " eager" : "",
              devs == 2 ? (std::string(" pipeline2-") + runtime::pp_handoff_name(eng->handoff()) + "-s" +
                           std::to_string(eng->split())).c_str() : "",
              a.depth, a.tg, eng->last_tok_per_s(), ms_per_token);
  return 0;
}

}  // namespace cli::muse

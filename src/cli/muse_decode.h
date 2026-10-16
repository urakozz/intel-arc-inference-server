#pragma once
// Spec 23c: b70-decode's Muse-Glimmer-30B path - the CLI dispatches on config.json's model_type
// (loader::is_muse_checkpoint) and, for "muse_glimmer", runs runtime::muse::MuseEngine on one card (the default,
// spec 23 decision 7) or two. `--ids` and `--bench` exactly as the Qwen path prints them (ids on stdout, one per
// line; the bench row on stdout); the prompt goes through the decode list one replay per id, or (spec 23d) through
// MuseEngine::prefill in chunks:
//
//   --prefill          (--ids) the prompt prefilled, then --n greedy ids
//   --prefill-length N (--bench) depth N prefilled and timed (the pp row), then --tg decoded
//   --prefill-chunk C  the chunk width, <= 2048 (runtime::muse::kPfC; the default)
//   --prefill-backend l0  the only backend
//
//   --pp N / --pipeline-parallel-size N   1 (the default: 17.4 GB of weights with the int8 head, one card) or 2
//                      (capacity, not speed - decision 7)
//   --pipeline-split auto|N, --pipeline-handoff copy|peer   spec 16b's switches and parsers; auto is
//                      runtime::pp_balance over Muse's per-layer bytes at the session's max_len
//   --layers N         development mode: load only layers [0, N) of the checkpoint (the head reads layer N-1's residual)
//   --lm-head int8|bf16 (int8 the default, spec 9), --max-len N|auto (b70-decode's 16384 the default; at most
//                      131072, a multiple of 256)
// Refused before the device, each by name: --prefill-backend l0-int8 (spec 23 §10: the h8 linears rotate in whole
// 1024-wide Hadamard blocks and 6656 = 6.5 x 1024) and sycl-tla (no Muse walk), --prefill-chunk above 2048 (the
// 4096-slot ring holds a chunk of 2048 and the 2047 keys before it), --mtp (Muse has no MTP head),
// --kv-cache int8 (decision 8: bf16 KV), --max-len above 131072 (decision 6), --profile (not built for Muse),
// --device N with --pp 2 (16b's rule: GPUs 0 and 1), --layers auto, --ple-dir.
//
// Spec 23f: `--spec dflash` (--ids: greedy, through cli::muse::MuseEngineAdapterT's speculative loop at --spec-max K
// every iteration - D2: the ids equal --spec off's) with --draft-model / --draft-format / --draft-q (cli/muse_spec.h has
// the flags and their refusals); --spec lookup is b70-serve's (the server's prompt lookup proposes), --bench with --spec
// is refused (b70-serve's llama-benchy times the loop, D5).
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
#include "cli/muse_spec.h"   // spec 23f
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/muse_loader.h"
#include "loader/snapshot.h"
#include "runtime/muse/muse_engine.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/prefill_backend.h"

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
  // Spec 23d: --prefill (--ids), --prefill-length N (--bench at depth N: the pp row), --prefill-chunk C (0 = kPfC),
  // --prefill-backend as parsed (l0 only).
  bool prefill = false, pp = false, pp_backend_given = false;
  uint32_t pp_chunk = 0;
  runtime::PrefillBackend pp_backend = runtime::PrefillBackend::L0;
  bool prefills() const { return prefill || pp; }
  // What else was asked for, for the refusals.
  bool mtp = false, spec = false, kv8 = false, profile = false, layers_auto = false, ple_dir = false;
  SpecArgs spec_args;   // spec 23f: --spec dflash / lookup and the drafter's flags (`spec` above: --spec mtp)
};

// Every refusal that needs no device - in this order, each by name (tests/CMakeLists.txt's cli_reject_muse_*). `d` is
// the checkpoint's descriptor (config.json only).
inline void check_args(const DecodeArgs& a, const model::MuseGlimmerDesc& d) {
  if (a.pp_backend_given && a.pp_backend == runtime::PrefillBackend::L0Int8)
    throw std::runtime_error("Muse-Glimmer-30B prefills on the l0 backend only (spec 23d), not l0-int8: spec 23 §10 - its "
                             "h8 linears rotate in whole 1024-wide Hadamard blocks and 6656 = 6.5 x 1024 (a later probe) - "
                             "--prefill-backend l0, or omit it");
  if (a.pp_backend_given && a.pp_backend == runtime::PrefillBackend::SyclTla)
    throw std::runtime_error("Muse-Glimmer-30B prefills on the l0 backend only (spec 23d), not sycl-tla: sycl-tla has no "
                             "Muse walk - --prefill-backend l0, or omit it");
  if (a.pp_chunk > runtime::muse::kPfC)
    throw std::runtime_error("--prefill-chunk " + std::to_string(a.pp_chunk) + " exceeds Muse-Glimmer's prefill chunk " +
                             std::to_string(runtime::muse::kPfC) + " (the 4096-slot ring holds a chunk of 2048 and the "
                             "2047 keys before it: 4095 <= 4096)");
  if (a.mtp)
    throw std::runtime_error("Muse-Glimmer-30B has no MTP head (spec 23 §1); its drafter is the DFlash assistant, "
                             "spec 23f - drop --mtp");
  if (a.spec)
    throw std::runtime_error("--spec mtp: Muse-Glimmer-30B has no MTP head (spec 23 §1); its proposers are --spec "
                             "dflash (the assistant, spec 23f) and --spec lookup");
  check_spec(a.spec_args, a.pipe.devices, a.prefills());   // spec 23f
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
  if (a.spec_args.on() && a.bench)
    throw std::runtime_error("--bench times plain decode; --spec on Muse-Glimmer-30B is timed through b70-serve "
                             "(llama-benchy, spec 23f D5) - drop --bench or --spec");
  if (a.spec_args.lookup())
    throw std::runtime_error("--spec lookup is b70-serve's (the server's prompt lookup proposes the drafts); b70-decode "
                             "runs --spec dflash");
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
  o.prefill = a.prefills();   // spec 23d: plan the prefill scratch (and its link) into max_len and the fit
  o.spec = engine_spec(a.spec_args, pre);   // spec 23f: the verify lists' rows and the drafter's term
  std::unique_ptr<runtime::muse::MuseEngine> eng;
  std::unique_ptr<runtime::dflash::MuseDrafter> drafter;
  {
    Redirect redirect;
    eng = std::make_unique<runtime::muse::MuseEngine>(ctx, a.path, o);
    if (a.spec_args.dflash()) drafter = attach_drafter(*eng, a.spec_args);
  }
  if (drafter)
    std::fprintf(stderr, "drafter: %s, --draft-q %s, --spec-max %u, %zu B (%s)\n", a.spec_args.draft_model.c_str(),
                 runtime::dflash::muse_q_name(a.spec_args.q), a.spec_args.max_k, drafter->bytes(),
                 drafter->weights().line.c_str());
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
  // Spec 23d: the prefill setup (every device's scratch, Context, the prefill hand-off, the binaries' check) happens
  // here, outside the timed window, which is then the whole prefill() call: first launch to the first generated id in
  // cur_token (argmax_stage2 is its last launch).
  const uint32_t chunk = a.pp_chunk ? a.pp_chunk : runtime::muse::kPfC;
  size_t pp_launches0 = 0;
  if (a.prefills()) {
    eng->prepare_prefill();
    pp_launches0 = eng->prefill_launches();
    std::fprintf(stderr, "prefill: l0 backend, chunk %u, attention %s (B70_MUSE_ATTN), %zu launches per chunk (%s) + %zu "
                 "for the head%s\n", chunk, runtime::muse::muse_attn_name(eng->attention()),
                 runtime::muse::prefill_chunk_launches(eng->weights().desc, eng->weights().placement),
                 devs == 2 ? "both cards" : "one card", runtime::muse::kPrefillHeadLaunches,
                 devs == 2 ? "; the chunk crosses by copy, sequentially" : "");
  }
  std::fprintf(stderr, "%s\n", eng->memory_line().c_str());
  if (drafter) {   // spec 23f: greedy through the adapter's speculative loop (D2), K = --spec-max every iteration
    MuseEngineAdapterT<runtime::muse::MuseEngine> ad(*eng, pre.vocab, pre.out_mult, pre.softcap, a.spec_args.max_k,
                                                     false, draft_hooks(*eng, *drafter));
    const auto s0 = std::chrono::steady_clock::now();
    if (a.prefills()) ad.prefill(ids);
    else ad.ingest(ids);
    const double in_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
    std::fprintf(stderr, "%s: %zu ids in %.1f ms (the drafter's ring filled: the prompt's last %u positions), pos %u\n",
                 a.prefills() ? "prefill" : "ingest", ids.size(), in_ms, drafter->window(), eng->pos());
    server::Sampling greedy;
    greedy.greedy = true;
    uint32_t printed = 0;
    const auto g0 = std::chrono::steady_clock::now();
    while (printed < a.n) {
      const std::vector<uint32_t> run = ad.step_many(greedy, a.spec_args.max_k);
      for (uint32_t id : run) {
        if (printed == a.n) break;
        std::printf("%u\n", id);
        ++printed;
      }
      std::fflush(stdout);
    }
    const double g_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
    std::fprintf(stderr, "generate (--spec dflash): %u ids in %.1f ms (%.2f t/s); %llu iterations, %llu / %llu drafts "
                 "accepted (%.3f)\n", printed, g_ms, g_ms > 0 ? 1000.0 * printed / g_ms : 0.0,
                 (unsigned long long)ad.spec_iterations(), (unsigned long long)ad.spec_accepted(),
                 (unsigned long long)ad.spec_drafted(),
                 ad.spec_drafted() ? double(ad.spec_accepted()) / double(ad.spec_drafted()) : 0.0);
    return 0;
  }
  const auto t0 = std::chrono::steady_clock::now();
  if (a.prefills())
    eng->prefill(ids, chunk);
  else
    eng->ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (a.prefills())
    std::fprintf(stderr, "%s: %zu ids in %.1f ms (%.2f t/s) - device-side, loader excluded, first prefill launch to the "
                 "first generated id; chunk %u, %zu L0 launches, pos %u\n", a.pp ? "pp" : "prefill", ids.size(),
                 ingest_ms, ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0, chunk,
                 eng->prefill_launches() - pp_launches0, eng->pos());
  else
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
  // Spec 23d: the pp row, a second line as the Kolibri / K2 / Qwen paths print it (depth, chunk, ms, t/s) - and the
  // linears' derived rate (50.3 GFLOP a token at 52 layers: 2 x the linears' parameters, spec 23 §5.3).
  if (a.pp) {
    const double tps = ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0;
    const double gflop_tok = 2.0 * double(eng->weights().desc.layers) * 483917824.0 / 1e9;
    std::fprintf(stderr, "  pp: %.2f t/s x %.1f GFLOP a token (the linears, derived) = %.1f TFLOP/s\n", tps, gflop_tok,
                 tps * gflop_tok / 1e3);
    std::printf("| b70-decode %s muse%s%s%s l0 pp | %u | %u | %.1f | %.2f |\n", sha, a.int8_head ? " int8-head" : " bf16-head",
                eng->attention() == runtime::muse::MuseAttn::Eager ? " eager" : "",
                devs == 2 ? (std::string(" pipeline2-copy-s") + std::to_string(eng->split())).c_str() : "", a.depth,
                chunk, ingest_ms, tps);
  }
  return 0;
}

}  // namespace cli::muse

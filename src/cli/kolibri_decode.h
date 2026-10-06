#pragma once
// Spec 20c: b70-decode's Kolibri-1 path - the CLI dispatches on config.json's model_type
// (loader::is_kolibri1_checkpoint) and, for "kolibri1", runs runtime::kolibri::KolibriEngine on one or
// two cards. `--ids` and `--bench` exactly as the Qwen path prints them (ids on stdout, one per line;
// the bench row on stdout); the prompt goes through the decode list one replay per id - or, spec 20d,
// through KolibriEngine::prefill in chunks:
//
//   --prefill          (--ids) the prompt prefilled, then --n greedy ids
//   --prefill-length N (--bench) depth N prefilled and timed (the pp row), then --tg decoded
//   --prefill-chunk C  the chunk width, <= 2048 (kernels::kolibri::kPfC; the default)
//   --prefill-backend l0  the only backend: l0-int8 (its h8 linears rotate in 1024-k Hadamard blocks and
//                      Kolibri's hidden is 2560) and sycl-tla (no Kolibri walk) are refused by name
//
//   --pp N / --pipeline-parallel-size N   2 (Kolibri's DEFAULT: the real model holds ~42.5 GB of
//                      weights at int4, one card 32.5 GB) or 1 - accepted only when the planner says the
//                      model fits one card (a synthetic checkpoint, --layers N), else refused before the
//                      device naming the bytes and --pp 2
//   --pipeline-split auto|N, --pipeline-handoff copy|peer   spec 16b's switches and parsers; auto is
//                      runtime::pp_balance over Kolibri's per-layer bytes at the session's max_len
//   --layers N         development mode (spec 20 §4): load only layers [0, N) of the checkpoint
//   --lm-head bf16|int8, --max-len N|auto (under the trained 262144; above it is spec 20 decision 3)
// Refused before the device, each by name: --prefill-backend l0-int8 / sycl-tla, --prefill-chunk above
// 2048 (spec 20d), --mtp (Kolibri has no MTP head), --kv-cache int8 (20 KiB a position: bf16 only),
// --profile (Task 8), --device N with --pp 2 (16b's rule: GPUs 0 and 1). A prefilling run plans the
// prefill scratch (and, on two cards, the prefill hand-off) into --max-len and the fit.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/kolibri1_layout.h"
#include "loader/kolibri1_loader.h"
#include "loader/lm_head_int8.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"
#include "runtime/kolibri/kolibri_engine.h"
#include "runtime/kolibri/kolibri_sizes.h"
#include "runtime/prefill_backend.h"

namespace cli::kolibri {

// The bench prompt, cycled to --depth: tests/golden/prompts/kolibri_bench.ids - the first 42 ids of
// tests/golden/prompts/de_prose.txt through Kolibri-1's own tokenizer (tools/oracle/tokenize.py on
// Aleph-Alpha/Kolibri-1-BF16 at 8c8b3489's tokenizer.json, which 20b's export copies unchanged; spec 20e
// replaced 20c's placeholder ids). "Als im Frühjahr 1871 die ersten Vermessungstrupps in das Tal von
// Rabenhausen kamen, glaubte kaum jemand im Ort, dass hier je ein Zug halten würde. Die Gemeinde lebte vom
// Holz". tools/box_validate/kolibri_oracle.sh synth prints the same 42 and diffs them against the file.
inline constexpr uint32_t kKolibriBenchPrompt[] = {
    27498, 2625,  19919, 32,    49,    56,    55,    49,    346,   7524, 744,   38897, 573,  122340,
    622,   518,   23578, 493,   18022, 266,   22233, 20353, 44,    64304, 14358, 11536, 2625, 6565,
    44,    1469,  2618,  3408,  922,   5958,  7737,  4325,  46,    1047,  8190,  65429, 4561, 7462};
inline constexpr size_t kKolibriBenchPromptLen = sizeof(kKolibriBenchPrompt) / sizeof(kKolibriBenchPrompt[0]);

// The B70's device memory as Level Zero reports it (32.530 GB, l0/context.cc): the --pp 1 fit check
// runs before any device is opened, against this; the device's own figure is used after.
inline constexpr size_t kB70Bytes = size_t(32530000000ull);

// True when `path` resolves to a snapshot whose config.json says model_type "kolibri1". A path that
// does not resolve is not Kolibri: the caller's own flow then reports it.
inline bool is_kolibri(const std::string& path) {
  try {
    return loader::is_kolibri1_checkpoint(loader::resolve_snapshot(path));
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
  uint32_t layers = 0;          // --layers N (0: the checkpoint's)
  PipelineArgs pipe;            // devices as parsed (1 when --pp was not given)
  bool pp_given = false;        // --pp / --pipeline-parallel-size on the command line
  // Spec 20d: --prefill (--ids), --prefill-length N (--bench at depth N: the pp row), --prefill-chunk C
  // (0 = kPfC), --prefill-backend as parsed (l0 only).
  bool prefill = false, pp = false, pp_backend_given = false;
  uint32_t pp_chunk = 0;
  runtime::PrefillBackend pp_backend = runtime::PrefillBackend::L0;
  // What else was asked for, for the refusals.
  bool mtp = false, kv8 = false, profile = false;
  bool prefills() const { return prefill || pp; }
};

// Kolibri's device count: --pp as given, else 2.
inline uint32_t devices_of(const DecodeArgs& a) { return a.pp_given ? a.pipe.devices : 2u; }

// Every refusal that needs no device - in this order, each by name (tests/CMakeLists.txt's
// cli_reject_kolibri_*). `d` is the checkpoint's descriptor (kolibri1_checkpoint_desc: config.json
// and the index's names only).
inline void check_args(const DecodeArgs& a, const model::Kolibri1Desc& d) {
  if (a.pp_backend_given && a.pp_backend == runtime::PrefillBackend::L0Int8)
    throw std::runtime_error("Kolibri-1 prefills on the l0 backend only (spec 20d), not l0-int8: its h8 linears "
                             "rotate in whole 1024-k Hadamard blocks and Kolibri's hidden is 2560 - "
                             "--prefill-backend l0, or omit it");
  if (a.pp_backend_given && a.pp_backend == runtime::PrefillBackend::SyclTla)
    throw std::runtime_error("Kolibri-1 prefills on the l0 backend only (spec 20d), not sycl-tla: sycl-tla has no "
                             "Kolibri walk - --prefill-backend l0, or omit it");
  if (a.pp_chunk > runtime::kolibri::kPfC)
    throw std::runtime_error("--prefill-chunk " + std::to_string(a.pp_chunk) + " exceeds Kolibri-1's prefill chunk " +
                             std::to_string(runtime::kolibri::kPfC) + " (the scratch and the 4096-slot ring's "
                             "headroom are sized for it)");
  if (a.mtp) throw std::runtime_error("Kolibri-1 has no MTP head (spec 20 §1); drop --mtp");
  if (a.kv8)
    throw std::runtime_error("--kv-cache int8: Kolibri-1's KV is bf16 only - 20 KiB a position on its 10 full "
                             "layers, 5.4 GB at 262144 (spec 20 §2)");
  if (a.profile)
    throw std::runtime_error("--profile is not built for Kolibri-1 (spec 20c Task 8, the box's speed work); "
                             "--bench times it");
  const uint32_t devs = devices_of(a);
  const bool int8 = a.lm_head == loader::LmHeadForm::Int8;
  if (devs == 1) {
    if (a.pipe.have_split)
      throw std::runtime_error("--pipeline-split belongs to --pp 2 (the layer device 1 starts at)");
    if (a.pipe.have_handoff)
      throw std::runtime_error("--pipeline-handoff belongs to --pp 2 (how the residual crosses)");
    if (!runtime::kolibri::fits_one_card(d, int8, kB70Bytes, a.reserve, a.prefills())) {
      const size_t w = loader::kol_device_weight_bytes(d, model::KolPlacement::one(d), 0, int8);
      char msg[400];
      std::snprintf(msg, sizeof msg,
                    "--pp 1: Kolibri-1 holds ~%.1f GB of weights at int4 (%u layers, %s attention, %s head), one card "
                    "%.1f GB: run it with --pp 2 (the default), or load fewer layers with --layers N (development "
                    "mode)", w / 1e9, d.layers, model::kol_attn_form_name(d.attn), int8 ? "int8" : "bf16",
                    kB70Bytes / 1e9);
      throw std::runtime_error(msg);
    }
    return;
  }
  if (a.device != l0::Context::kFromEnv)
    throw std::runtime_error("--pp 2 runs on GPUs 0 and 1 of what Level Zero shows (ZE_AFFINITY_MASK picks which "
                             "two); --device names one card - drop it");
  if (d.layers < 2) throw std::runtime_error("--pp 2 needs at least two layers to split (--layers " +
                                             std::to_string(d.layers) + ")");
  if (!a.pipe.split_auto && (a.pipe.split < 1 || a.pipe.split >= d.layers))
    throw std::runtime_error("--pipeline-split " + std::to_string(a.pipe.split) + " leaves a device no layer: it is "
                             "the first layer of device 1, in [1, " + std::to_string(d.layers - 1) + "]");
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
      throw std::runtime_error("--ids file '" + path + "': id " + w + " is outside Kolibri-1's vocabulary (" +
                               std::to_string(vocab) + " rows)");
    ids.push_back(uint32_t(v));
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

// The placement and max_len for this run, from the planner (before the load: each layer goes
// straight onto its device). Prints the choice and the plan to stderr.
inline std::pair<model::KolPlacement, uint32_t> settle(const model::Kolibri1Desc& d, const DecodeArgs& a,
                                                       const std::array<size_t, runtime::kPpDevices>& dev) {
  const bool int8 = a.lm_head == loader::LmHeadForm::Int8, pf = a.prefills();
  const uint32_t devs = devices_of(a), cap = d.trained_max_len;
  namespace rk = runtime::kolibri;
  const rk::KolAttn at = rk::kolibri_attn();
  model::KolPlacement p = devs == 1 ? model::KolPlacement::one(d) : model::KolPlacement::two(d, a.pipe.split ? a.pipe.split : 1);
  uint32_t len = a.max_len.value;
  if (a.max_len.is_auto) {
    if (devs == 2 && a.pipe.split_auto) {
      const runtime::PpChoice c = rk::pp_split_and_len(d, int8, dev, a.reserve, cap, pf);
      if (c.max_len == 0)
        throw std::runtime_error("--max-len auto: no split fits " + std::to_string(runtime::kMinAutoMaxLen) +
                                 " positions on both cards - " +
                                 rk::describe(rk::plan(d, model::KolPlacement::two(d, d.layers / 2), runtime::kMinAutoMaxLen, int8,
                                                       false, at, pf),
                                              model::KolPlacement::two(d, d.layers / 2), runtime::kMinAutoMaxLen, dev, a.reserve));
      p = model::KolPlacement::two(d, c.split);
      len = c.max_len;
    } else {
      len = rk::max_len_that_fits(d, p, int8, dev, a.reserve, cap, pf);
      if (len == 0)
        throw std::runtime_error("--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
                                 " positions fit - " +
                                 rk::describe(rk::plan(d, p, runtime::kMinAutoMaxLen, int8, false, at, pf), p,
                                              runtime::kMinAutoMaxLen, dev, a.reserve));
    }
    std::fprintf(stderr, "max_len: auto -> %u (the largest multiple of %u that fits every card with a %.3f GB "
                 "reserve; trained context %u)\n", len, runtime::kMaxLenQuantum, a.reserve / 1e9, cap);
  } else {
    if (len > cap)
      throw std::runtime_error("--max-len " + std::to_string(len) + " exceeds Kolibri-1's trained context " +
                               std::to_string(cap) + " - a longer context is spec 20 decision 3");
    if (devs == 2 && a.pipe.split_auto)
      p = model::KolPlacement::two(d, rk::pp_split(d, len, int8, a.pipe.handoff, at, pf).split);
    const std::vector<rk::DevicePlan> pl = rk::plan(d, p, len, int8, false, at, pf);
    for (uint32_t i = 0; i < p.devices; ++i)
      if (pl[i].total() + a.reserve > dev[i])
        throw std::runtime_error("--max-len " + std::to_string(len) + " does not fit device " + std::to_string(i) +
                                 ": " + rk::describe(pl, p, len, dev, a.reserve) + ". The largest that fits is " +
                                 std::to_string(rk::max_len_that_fits(d, p, int8, dev, a.reserve, cap, pf)) +
                                 " (--max-len auto)");
    std::fprintf(stderr, "max_len: %u (--max-len)\n", len);
  }
  if (devs == 2)
    std::fprintf(stderr, "split: %s%u (layers [0, %u) on device 0, [%u, %u) and the head on device 1)\n",
                 a.pipe.split_auto ? "auto -> " : "", p.split, p.split, p.split, d.layers);
  std::fprintf(stderr, "%s\n", rk::describe(rk::plan(d, p, len, int8, false, at, pf), p, len, dev, a.reserve).c_str());
  return {p, len};
}

// The loader prints its report to stdout (its convention); `Redirect` is the caller's StdoutToStderr so
// the report joins the diagnostics on stderr.
template <class Redirect>
int run_decode(const DecodeArgs& a) {
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(a.path, a.layers);
  check_args(a, pre);
  const uint32_t devs = devices_of(a);
  std::vector<uint32_t> ids;
  if (!a.bench) {
    ids = read_ids(a.ids_path, pre.vocab);
  } else {
    ids.resize(a.depth);
    for (uint32_t i = 0; i < a.depth; ++i) ids[i] = kKolibriBenchPrompt[i % kKolibriBenchPromptLen];
  }
  const uint32_t after = a.bench ? a.tg : a.n;
  const size_t need = ids.size() + after;
  if (!a.max_len.is_auto && need > a.max_len.value)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds --max-len " +
                             std::to_string(a.max_len.value));
  if (devs == 2) require_two_devices(l0::Context::gpu_count());

  // The contexts: one card (--device / the environment), or GPUs 0 and 1 (device 1 a view of 0's).
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(devs == 2 ? 0u : a.device);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  std::array<size_t, runtime::kPpDevices> dev{c0->memory_bytes(), c1 ? c1->memory_bytes() : 0};
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)%s\n", c0->name().c_str(), c0->eu_count(),
                 a.device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  else
    std::fprintf(stderr, "devices: %s + %s (--pp 2)\n", c0->name().c_str(), c1->name().c_str());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--pp 2: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), and "
                             "both hand-offs write it (docs/10-the-box.md)");
  const std::pair<model::KolPlacement, uint32_t> pl = settle(pre, a, dev);
  const model::KolPlacement placement = pl.first;
  const uint32_t max_len = pl.second;
  if (need > max_len)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds max_len " +
                             std::to_string(max_len));
  loader::KolLoadedModel model = [&] {
    Redirect redirect;
    return loader::load_kolibri1(ctx, a.path, max_len, placement, a.lm_head, a.layers);
  }();
  if (a.layers)
    std::fprintf(stderr, "layers: %u of the checkpoint's %u (--layers: development mode; the head reads layer %u's "
                 "residual)\n", model.desc.layers, model.checkpoint_layers, model.desc.layers - 1);
  runtime::PipelineOptions opt;
  opt.handoff = a.pipe.handoff;
  runtime::kolibri::KolibriEngine eng(ctx, std::move(model), max_len, /*debug_tap=*/false, opt);
  std::fprintf(stderr, "engine: Kolibri-1, %u device(s)%s, %zu launches a token (%s attention, B70_KOLIBRI_ATTN), "
               "max_len %u\n", eng.devices(),
               devs == 2 ? (std::string(", split ") + std::to_string(eng.split()) + ", hand-off " +
                            runtime::pp_handoff_name(eng.handoff())).c_str() : "",
               eng.launches(), runtime::kolibri::kol_attn_name(eng.attention()), eng.max_len());
  // Spec 20d: the prefill setup (every device's scratch, Context, the prefill hand-off, the binaries' check)
  // happens here, outside the timed window, which is then the whole prefill() call: first launch to the
  // first generated id in cur_token (argmax_stage2 is its last launch).
  const uint32_t chunk = a.pp_chunk ? a.pp_chunk : runtime::kolibri::kPfC;
  size_t pp_launches0 = 0;
  if (a.prefills()) {
    eng.prepare_prefill();
    pp_launches0 = eng.prefill_launches();
    std::fprintf(stderr, "prefill: l0 backend, chunk %u, attention %s (B70_KOLIBRI_ATTN), %zu launches per chunk "
                 "(%s) + %zu for the head%s\n", chunk, runtime::kolibri::kol_attn_name(eng.attention()),
                 runtime::kolibri::prefill_chunk_launches(eng.model().desc, eng.model().placement),
                 devs == 2 ? "both cards" : "one card", runtime::kolibri::kPrefillHeadLaunches,
                 devs == 2 ? "; the chunk crosses by copy, sequentially (spec 16c's overlap is a later lever)" : "");
  }
  std::fprintf(stderr, "%s\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  if (a.prefills())
    eng.prefill(ids, chunk);
  else
    eng.ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (a.prefills())
    std::fprintf(stderr, "%s: %zu ids in %.1f ms (%.2f t/s) - device-side, loader excluded, first prefill launch to "
                 "the first generated id; chunk %u, %zu L0 launches, pos %u\n", a.pp ? "pp" : "prefill", ids.size(),
                 ingest_ms, ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0, chunk,
                 eng.prefill_launches() - pp_launches0, eng.pos());
  else
    std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token, one decode replay per id), pos %u\n", ids.size(),
                 ingest_ms, ingest_ms / double(ids.size()), eng.pos());
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
  const double gb = double(eng.model().read_per_token) / 1e9;
  std::fprintf(stderr, "  MBU: %.2f t/s x %.3f GB = %.0f GB/s (W = this load's derived read/token: top-6 + shared "
               "experts, attention, routers, head); %zu launches/token\n",
               eng.last_tok_per_s(), gb, eng.last_tok_per_s() * gb, eng.launches());
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode %s kolibri%s%s%s | %u | %u | %.2f | %.2f |\n", sha,
              a.lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "",
              eng.model().desc.attn == model::KolAttnForm::Bf16 ? " bf16-attn" : "",
              devs == 2 ? (std::string(" pipeline2-") + runtime::pp_handoff_name(eng.handoff()) + "-s" +
                           std::to_string(eng.split())).c_str() : "",
              a.depth, a.tg, eng.last_tok_per_s(), ms_per_token);
  // Spec 20d: the pp row, a second line as the K2 / Qwen paths print it (depth, chunk, ms, t/s).
  if (a.pp)
    std::printf("| b70-decode %s kolibri%s%s%s l0 pp | %u | %u | %.1f | %.2f |\n", sha,
                a.lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "",
                eng.model().desc.attn == model::KolAttnForm::Bf16 ? " bf16-attn" : "",
                devs == 2 ? (std::string(" pipeline2-copy-s") + std::to_string(eng.split())).c_str() : "", a.depth,
                chunk, ingest_ms, ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0);
  return 0;
}

}  // namespace cli::kolibri

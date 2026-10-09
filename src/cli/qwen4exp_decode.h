#pragma once
// Spec 21c: b70-decode's Qwen3.8-Flash-Next path - the CLI dispatches on config.json's model_type
// (loader::is_qwen4exp_checkpoint: qwen4_exp / qwen4_exp_text) and runs runtime::qwen4exp::Qwen4ExpEngine on one or
// two cards (cli/kolibri_decode.h's shape). `--ids` and `--bench` exactly as the other paths print them (ids on
// stdout, one per line; the bench row on stdout); the prompt goes through the decode list one replay per id.
//
//   --layers N|auto    REQUIRED until spec 22: the development mode (spec 21 §6) - load only layers [0, N) of the
//                      checkpoint, then the final mixer and the head; `auto` is the planner's largest N that fits
//                      (runtime::qwen4exp::layers_that_fit). Without it the whole model is refused naming its
//                      bytes and spec 22 (the expert-offload tier): 512 experts x 48 layers do not fit two B70s
//   --ple-dir DIR      the PLE int8 file (default: <snapshot>-ple-int8/ or $B70_Q4_PLE - loader::q4_ple_dir)
//   --pp N / --pipeline-parallel-size N   1 (the default) or 2 (spec 16b's pieces: the materialised H crosses);
//   --pipeline-split auto|N, --pipeline-handoff copy|peer   16b's switches and parsers; auto is
//                      runtime::qwen4exp::pp_split (pp_balance over the per-layer bytes at the session's max_len)
//   --lm-head bf16|int8 (bf16 by default, as b70-decode's other paths), --max-len N|auto (<= the trained 262144)
//   --bench [--depth N] [--tg N]; B70_Q4_ATTN=flash|eager picks the decode attention
// Refused before the device, each by name: --prefill / --prefill-length / --prefill-chunk / --prefill-backend
// (spec 21d builds the prefill), --mtp (spec 21e), --kv-cache int8 (decision 8: bf16 first; kv8.cl hard-codes 24 q
// / 4 kv heads), --profile (Task 7 times it with --bench), --device N with --pp 2 (16b's rule: GPUs 0 and 1), and
// the whole model without --layers (spec 22).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/lm_head_int8.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_loader.h"
#include "loader/snapshot.h"
#include "model/qwen4exp.h"
#include "runtime/qwen4exp/qwen4exp_engine.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace cli::qwen4exp {

// The bench prompt, cycled to --depth: the first 42 ids of tests/golden/prompts/q4exp_short.ids (prose.txt through
// the original's tokenizer - 21a: Qwen3.8's committed ids re-encode identically).
inline constexpr uint32_t kQwen4ExpBenchPrompt[] = {
    760,   72103, 506,   37119, 557,  11012, 3213, 310,   6512, 279,   61789, 272,  1072, 2272,
    279,   197616, 2271, 13,    469,  68042, 29123, 7247, 383,  279,   1387,  12615, 1345, 279,
    49813, 78911, 1141,  20459, 13,   3113,  7840, 279,  2981, 1000,  381,   16850, 1495, 13};
inline constexpr size_t kQwen4ExpBenchPromptLen = sizeof(kQwen4ExpBenchPrompt) / sizeof(kQwen4ExpBenchPrompt[0]);

// The B70's device memory as Level Zero reports it (32.530 GB, l0/context.cc): the fit checks before any device
// is opened run against this; the device's own figure is used after.
inline constexpr size_t kB70Bytes = size_t(32530000000ull);

// True when `path` resolves to a snapshot whose config.json says model_type qwen4_exp (or qwen4_exp_text). A path
// that does not resolve is not this family: the caller's own flow then reports it.
inline bool is_qwen4exp(const std::string& path) {
  try {
    return loader::is_qwen4exp_checkpoint(loader::resolve_snapshot(path));
  } catch (const std::exception&) {
    return false;
  }
}

struct DecodeArgs {
  std::string path, ids_path, ple_dir;
  uint32_t n = 0, depth = 4096, tg = 256;
  bool bench = false;
  MaxLenArg max_len{false, 16384};
  size_t reserve = 0;
  uint32_t device = l0::Context::kFromEnv;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  uint32_t layers = 0;          // --layers N (0: not given)
  bool layers_auto = false;     // --layers auto
  PipelineArgs pipe;            // devices as parsed (1 when --pp was not given)
  bool pp_given = false;
  // What else was asked for, for the refusals.
  bool prefill = false, prefill_length = false, prefill_chunk = false, prefill_backend = false;
  bool mtp = false, kv8 = false, profile = false;
  bool int8_head() const { return lm_head == loader::LmHeadForm::Int8; }
};

inline uint32_t devices_of(const DecodeArgs& a) { return a.pp_given ? a.pipe.devices : 1u; }

// The whole published model's refusal (no --layers): the planner's own message (require_fits names the bytes,
// each card's capacity and spec 22) on its best two-card split.
[[noreturn]] inline void refuse_whole(const DecodeArgs& a) {
  namespace rq = runtime::qwen4exp;
  const model::Qwen4ExpDesc& d = model::qwen4exp();
  const uint32_t len = a.max_len.is_auto ? runtime::kMinAutoMaxLen : a.max_len.value;
  const model::Q4Placement p = model::Q4Placement::two(d, rq::pp_split(d, len, a.int8_head(), false).split);
  const std::array<size_t, runtime::kPpDevices> caps = {kB70Bytes, kB70Bytes};
  try {
    rq::require_fits(rq::plan(d, p, len, a.int8_head(), false), caps, a.reserve);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(e.what()) + ". Without --layers b70-decode would load all 48 layers: pass "
                             "--layers N or --layers auto (development mode, spec 21c)");
  }
  throw std::runtime_error("Qwen3.8-Flash-Next's whole model runs only with spec 22's expert-offload tier: pass "
                           "--layers N or --layers auto (development mode, spec 21c)");
}

// Every refusal that needs no device and no tensor - in this order, each by name (tests/CMakeLists.txt's
// cli_reject_qwen4exp_*): the stages that build what was asked for first, then the whole model.
inline void check_args(const DecodeArgs& a) {
  if (a.prefill || a.prefill_length || a.prefill_chunk || a.prefill_backend)
    throw std::runtime_error("Qwen3.8-Flash-Next prefills in spec 21d (the indexer GEMM, the row-wise selection, "
                             "sparse flash): this build decodes the prompt one replay per id - drop --prefill / "
                             "--prefill-length / --prefill-chunk / --prefill-backend");
  if (a.mtp)
    throw std::runtime_error("Qwen3.8-Flash-Next's MTP head (its own QSA layer and 512 experts) is spec 21e's: drop --mtp");
  if (a.kv8)
    throw std::runtime_error("--kv-cache int8: Qwen3.8-Flash-Next's KV is bf16 (spec 21 decision 8: int8 later - kv8.cl "
                             "hard-codes 24 q / 4 kv heads and this model has 2)");
  if (a.profile)
    throw std::runtime_error("--profile is not built for Qwen3.8-Flash-Next (spec 21c Task 7 times it with --bench)");
  const uint32_t devs = devices_of(a);
  if (devs == 1) {
    if (a.pipe.have_split) throw std::runtime_error("--pipeline-split belongs to --pp 2 (the layer device 1 starts at)");
    if (a.pipe.have_handoff) throw std::runtime_error("--pipeline-handoff belongs to --pp 2 (how H crosses)");
  } else if (a.device != l0::Context::kFromEnv) {
    throw std::runtime_error("--pp 2 runs on GPUs 0 and 1 of what Level Zero shows (ZE_AFFINITY_MASK picks which "
                             "two); --device names one card - drop it");
  }
  if (a.layers == 0 && !a.layers_auto) refuse_whole(a);
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
      throw std::runtime_error("--ids file '" + path + "': id " + w + " is outside Qwen3.8-Flash-Next's vocabulary (" +
                               std::to_string(vocab) + " rows)");
    ids.push_back(uint32_t(v));
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

struct Settled {
  model::Qwen4ExpDesc desc;   // the truncated descriptor
  model::Q4Placement placement;
  uint32_t max_len = 0;
};

// The layer count, placement and max_len for this run, from the planner (before the load: each layer goes straight
// onto its device). `full` is the checkpoint's descriptor (all its layers). Prints the choice and the plan.
inline Settled settle(const model::Qwen4ExpDesc& full, const DecodeArgs& a,
                      const std::array<size_t, runtime::kPpDevices>& dev) {
  namespace rq = runtime::qwen4exp;
  const bool int8 = a.int8_head();
  const uint32_t devs = devices_of(a), cap = full.trained_max_len;
  Settled s;
  uint32_t layers = a.layers;
  if (a.layers_auto) {
    const uint32_t len = a.max_len.is_auto ? runtime::kMinAutoMaxLen : a.max_len.value;
    layers = rq::layers_that_fit(full, devs, len, int8, false, std::min(dev[0], devs == 2 ? dev[1] : dev[0]), a.reserve);
    if (layers == 0)
      throw std::runtime_error("--layers auto: not even " + std::to_string(full.ple_layer + 1) + " layers fit " +
                               std::to_string(devs) + " card(s) at max_len " + std::to_string(len));
    std::fprintf(stderr, "layers: auto -> %u of %u (the largest N whose truncated model fits %u card(s) at max_len %u "
                 "with a %.3f GB reserve)\n", layers, full.layers, devs, len, a.reserve / 1e9);
  }
  if (layers > full.layers)
    throw std::runtime_error("--layers " + std::to_string(layers) + " exceeds the checkpoint's " + std::to_string(full.layers));
  s.desc = rq::truncated(full, layers);
  const model::Qwen4ExpDesc& d = s.desc;
  model::Q4Placement p = devs == 1 ? model::Q4Placement::one(d) : model::Q4Placement::two(d, a.pipe.split ? a.pipe.split : 1);
  if (devs == 2 && d.layers < 2) throw std::runtime_error("--pp 2 needs at least two layers to split (--layers " +
                                                          std::to_string(d.layers) + ")");
  if (devs == 2 && !a.pipe.split_auto && (a.pipe.split < 1 || a.pipe.split >= d.layers))
    throw std::runtime_error("--pipeline-split " + std::to_string(a.pipe.split) + " leaves a device no layer: it is the "
                             "first layer of device 1, in [1, " + std::to_string(d.layers - 1) + "]");
  uint32_t len = a.max_len.value;
  if (a.max_len.is_auto) {
    if (devs == 2 && a.pipe.split_auto)
      p = model::Q4Placement::two(d, rq::pp_split(d, runtime::kMinAutoMaxLen, int8, false).split);
    len = rq::max_len_that_fits(d, p, int8, false, dev, a.reserve, cap);
    if (len == 0)
      throw std::runtime_error("--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) + " positions fit - " +
                               rq::describe(rq::plan(d, p, runtime::kMinAutoMaxLen, int8, false), p,
                                            runtime::kMinAutoMaxLen, dev, a.reserve));
    std::fprintf(stderr, "max_len: auto -> %u (the largest multiple of %u that fits every card with a %.3f GB "
                 "reserve; trained context %u)\n", len, runtime::kMaxLenQuantum, a.reserve / 1e9, cap);
  } else {
    if (len > cap)
      throw std::runtime_error("--max-len " + std::to_string(len) + " exceeds Qwen3.8-Flash-Next's trained context " +
                               std::to_string(cap));
    if (devs == 2 && a.pipe.split_auto) p = model::Q4Placement::two(d, rq::pp_split(d, len, int8, false).split);
    std::fprintf(stderr, "max_len: %u (--max-len)\n", len);
  }
  rq::require_fits(rq::plan(d, p, len, int8, false), dev, a.reserve);
  if (devs == 2)
    std::fprintf(stderr, "split: %s%u (layers [0, %u) on device 0, [%u, %u) and the head on device 1)\n",
                 a.pipe.split_auto ? "auto -> " : "", p.split, p.split, p.split, d.layers);
  std::fprintf(stderr, "%s\n", rq::describe(rq::plan(d, p, len, int8, false), p, len, dev, a.reserve).c_str());
  s.placement = p;
  s.max_len = len;
  return s;
}

// The loader prints its report to stdout (its convention); `Redirect` is the caller's StdoutToStderr so the report
// joins the diagnostics on stderr.
template <class Redirect>
int run_decode(const DecodeArgs& a) {
  check_args(a);   // before any tensor is read
  const uint32_t devs = devices_of(a);
  const model::Qwen4ExpDesc full = loader::qwen4exp_checkpoint_desc(a.path, 0);
  std::vector<uint32_t> ids;
  if (!a.bench) {
    ids = read_ids(a.ids_path, full.vocab);
  } else {
    ids.resize(a.depth);
    for (uint32_t i = 0; i < a.depth; ++i) ids[i] = kQwen4ExpBenchPrompt[i % kQwen4ExpBenchPromptLen];
  }
  const uint32_t after = a.bench ? a.tg : a.n;
  const size_t need = ids.size() + after;
  if (!a.max_len.is_auto && need > a.max_len.value)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds --max-len " +
                             std::to_string(a.max_len.value));
  if (devs == 2) require_two_devices(l0::Context::gpu_count());
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(devs == 2 ? 0u : a.device);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  std::array<size_t, runtime::kPpDevices> dev{c0->memory_bytes(), c1 ? c1->memory_bytes() : c0->memory_bytes()};
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)%s\n", c0->name().c_str(), c0->eu_count(),
                 a.device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  else
    std::fprintf(stderr, "devices: %s + %s (--pp 2)\n", c0->name().c_str(), c1->name().c_str());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--pp 2: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), and "
                             "both hand-offs write it (docs/10-the-box.md)");
  const Settled s = settle(full, a, dev);
  if (need > s.max_len)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds max_len " +
                             std::to_string(s.max_len));
  loader::Q4LoadedModel model = [&] {
    Redirect redirect;
    return loader::load_qwen4exp(ctx, a.path, s.max_len, s.placement, a.lm_head, s.desc.layers, false, a.ple_dir);
  }();
  std::fprintf(stderr, "layers: %u of the checkpoint's %u (--layers: development mode; the final mixer reads layer %u's "
               "H)\n", model.desc.layers, model.checkpoint_layers, model.desc.layers - 1);
  runtime::PipelineOptions opt;
  opt.handoff = a.pipe.handoff;
  runtime::qwen4exp::Qwen4ExpEngine eng(ctx, std::move(model), s.max_len, /*debug_tap=*/false, opt);
  std::fprintf(stderr, "engine: Qwen3.8-Flash-Next, %u device(s)%s, %zu launches a token (%s attention, B70_Q4_ATTN), "
               "max_len %u; the PLE table's %zu host pages read back by the device\n", eng.devices(),
               devs == 2 ? (std::string(", split ") + std::to_string(eng.split()) + ", hand-off " +
                            runtime::pp_handoff_name(eng.handoff())).c_str() : "",
               eng.launches(), runtime::qwen4exp::q4_attn_name(eng.attention()), eng.max_len(), eng.ple_pages_checked());
  std::fprintf(stderr, "%s\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  eng.ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
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
  std::fprintf(stderr, "  MBU: %.2f t/s x %.3f GB = %.0f GB/s (W = this load's derived read/token: both HCs, the "
               "mixers, the top-10 + shared experts, the routers, the head; the 16 PLE rows from host memory apart); %zu "
               "launches/token\n", eng.last_tok_per_s(), gb, eng.last_tok_per_s() * gb, eng.launches());
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode %s qwen4exp L%u%s%s %s%s | %u | %u | %.2f | %.2f |\n", sha, eng.model().desc.layers,
              a.int8_head() ? " int8-head" : "",
              eng.model().desc.forms.dense == model::Q4Form::Bf16 ? " bf16-dense" : "",
              runtime::qwen4exp::q4_attn_name(eng.attention()),
              devs == 2 ? (std::string(" pipeline2-") + runtime::pp_handoff_name(eng.handoff()) + "-s" +
                           std::to_string(eng.split())).c_str() : "",
              a.depth, a.tg, eng.last_tok_per_s(), ms_per_token);
  return 0;
}

}  // namespace cli::qwen4exp

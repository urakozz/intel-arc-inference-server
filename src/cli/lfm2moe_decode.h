#pragma once
// Spec 24c: b70-decode's LFM2.5-24B-A2B path - the CLI dispatches on config.json's model_type
// (loader::is_lfm2moe_checkpoint) and, for "lfm2_moe", runs runtime::lfm2moe::Lfm2MoeEngine on one card (the default:
// the model and its whole 131072-position context fit one B70, spec 24 §3) or two. cli/kolibri_decode.h's shape:
//
//   --ids F --n N        the prompt one decode replay per id (24d brings prefill), then N greedy ids on stdout
//   --bench [--depth N] [--tg N]   depth N of the bench prompt ingested, N decoded and timed (the bench row on stdout)
//   --profile [--depth N] [--steps N] [--repeats R]   depth N ingested, then R x N profiled replays: the per-launch
//                        anatomy (kernel-timestamp events, docs/15's instrument; never a bench row)
//   --lm-head bf16|int8  int8 the default (spec 9: built from the tied embedding); bf16 reads the embedding tiles
//   --max-len N|auto     16384 the default (bench rows comparable); auto = runtime::lfm2moe::max_len_auto, at most
//                        the trained 131072
//   --pp N / --pipeline-parallel-size N   1 (the default) or 2 (spec 16b's pieces, decision 10)
//   --pipeline-split auto|N, --pipeline-handoff copy|peer   --pp 2's
// B70_LFM_ATTN=flash|eager and B70_LFM_FUSE=none|all|<route,router,conv> pick the list (read at construction).
// Refused before the device, each by name (tests/CMakeLists.txt's cli_reject_lfm2moe_*): --prefill / --prefill-length
// / --prefill-chunk / --prefill-backend (spec 24d), --mtp / --spec (no MTP head), --kv-cache int8 (spec 24 decision
// 11), --layers (Kolibri-1's / Qwen3.8-Flash-Next's development mode), --device with --pp 2.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/lfm2moe_layout.h"
#include "loader/lfm2moe_loader.h"
#include "loader/snapshot.h"
#include "runtime/lfm2moe/lfm2moe_engine.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace cli::lfm2moe {

// The bench prompt, cycled to --depth: the first 46 ids of tests/golden/prompts/lfm2moe_short.ids (spec 24a: BOS then
// tests/golden/prompts/prose.txt through LFM2.5's own tokenizer) - "The harbour at dawn was quiet enough to hear the
// ropes creak against the bollards. Gulls traded places on the breakwater while the ferry warmed its engines. By
// seven the market would be loud again."
inline constexpr uint32_t kBenchPrompt[] = {
    124894, 597,  52593, 462,  28206, 452, 14086, 3374, 310,  7997, 278,   731,  11136, 1339, 454,   2222,
    278,    293,  754,   2391, 22,    417, 58613, 38950, 5953, 383, 278,   3781, 6621,  1657, 278,   49785,
    5747,   290,  851,   17354, 22,   3359, 7077,  278,  3630, 998, 373,   18977, 1529, 22};
inline constexpr size_t kBenchPromptLen = sizeof(kBenchPrompt) / sizeof(kBenchPrompt[0]);

// The B70's device memory as Level Zero reports it (32.530 GB): the planner's check before any device is opened.
inline constexpr size_t kB70Bytes = size_t(32530000000ull);

// True when `path` resolves to a snapshot whose config.json says model_type "lfm2_moe". A path that does not
// resolve is not LFM2: the caller's own flow then reports it.
inline bool is_lfm2moe(const std::string& path) {
  try {
    return loader::is_lfm2moe_checkpoint(loader::resolve_snapshot(path));
  } catch (const std::exception&) {
    return false;
  }
}

struct DecodeArgs {
  std::string path, ids_path;
  uint32_t n = 0, depth = 4096, tg = 256, steps = 32, repeats = 1;
  bool bench = false, profile = false;
  MaxLenArg max_len{false, 16384};
  size_t reserve = 0;
  uint32_t device = l0::Context::kFromEnv;
  bool int8_head = true;        // --lm-head (spec 9's int8 head unless bf16)
  PipelineArgs pipe;            // devices as parsed (1 when --pp was not given)
  // What else was asked for, for the refusals.
  bool prefill = false, mtp = false, kv8 = false, layers = false;
};

// Every refusal that needs no device - in this order, each by name (tests/CMakeLists.txt's cli_reject_lfm2moe_*).
inline void check_args(const DecodeArgs& a, const model::Lfm2MoeDesc& d) {
  if (a.prefill)
    throw std::runtime_error("LFM2.5-24B-A2B's prefill is spec 24d (not built yet): --prefill, --prefill-length, "
                             "--prefill-chunk and --prefill-backend are refused - the prompt goes through the decode "
                             "list one replay per id");
  if (a.mtp)
    throw std::runtime_error("LFM2.5-24B-A2B has no MTP head (spec 24 §11): drop --mtp / --spec");
  if (a.kv8)
    throw std::runtime_error("--kv-cache int8: LFM2.5-24B-A2B's KV is bf16 (spec 24 decision 11: int8 KV at head_dim "
                             "64 is a later spec)");
  if (a.layers)
    throw std::runtime_error("--layers belongs to Kolibri-1 / Qwen3.8-Flash-Next (their development modes); "
                             "LFM2.5-24B-A2B's whole model fits one card");
  if (a.pipe.devices == 1) {
    if (a.pipe.have_split) throw std::runtime_error("--pipeline-split belongs to --pp 2 (the layer device 1 starts at)");
    if (a.pipe.have_handoff) throw std::runtime_error("--pipeline-handoff belongs to --pp 2 (how the residual crosses)");
    return;
  }
  if (a.device != l0::Context::kFromEnv)
    throw std::runtime_error("--pp 2 runs on GPUs 0 and 1 of what Level Zero shows (ZE_AFFINITY_MASK picks which "
                             "two); --device names one card - drop it");
  if (d.layers < 2) throw std::runtime_error("--pp 2 needs at least two layers to split");
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
      throw std::runtime_error("--ids file '" + path + "': id " + w + " is outside LFM2.5-24B-A2B's vocabulary (" +
                               std::to_string(vocab) + " rows)");
    ids.push_back(uint32_t(v));
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

// The bytes a decode step reads (derived): every layer's int4 linears and router, top_k experts' blocks, the small
// blocks, the head (int8 or the bf16 tiles), the embedding row - the MBU line's W (KV excluded).
inline size_t read_per_token(const model::Lfm2MoeDesc& d, bool int8_head) {
  size_t b = size_t(d.hidden) * 2 + size_t(d.hidden) * 4;   // the gathered row, embedding_norm
  for (uint32_t l = 0; l < d.layers; ++l) {
    for (const model::LfmLinear& ln : d.linears(l))
      b += ln.kind == model::WeightKind::Int4 ? loader::lfm::int4_bytes(ln.shape) : size_t(ln.shape.K) * ln.shape.N * 2;
    if (!d.is_dense(l))
      b += size_t(d.top_k) * (loader::lfm::expert_gate_up_bytes(d) + loader::lfm::expert_down_bytes(d));
    b += loader::lfm::small_block(d, l).bytes;
  }
  return b + (int8_head ? loader::lfm::head_int8_bytes(d) : loader::lfm::embed_tiled_bytes(d));
}

// The placement and max_len for this run, from the planner (before the load). Prints the choice and the plan.
inline std::pair<model::LfmPlacement, uint32_t> settle(const model::Lfm2MoeDesc& d, const DecodeArgs& a,
                                                       const std::array<size_t, runtime::kPpDevices>& dev) {
  namespace rl = runtime::lfm2moe;
  const rl::LfmAttn at = rl::lfm_attn();
  model::LfmPlacement p = model::LfmPlacement::one(d);
  if (a.pipe.devices == 2)
    p = model::LfmPlacement::two(d, a.pipe.split_auto ? rl::pp_split(d, a.int8_head) : a.pipe.split);
  const double reserve_gb = (a.reserve ? double(a.reserve) : runtime::kDefaultReserveGb * 1e9) / 1e9;
  size_t smallest = dev[0];
  for (uint32_t i = 1; i < p.devices; ++i) smallest = std::min(smallest, dev[i]);
  uint32_t len = a.max_len.value;
  if (a.max_len.is_auto) {
    len = rl::max_len_auto(d, p, smallest, reserve_gb, a.int8_head, at);
    if (len == 0)
      throw std::runtime_error("--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
                               " positions fit - " + rl::describe(rl::plan(d, p, 0, runtime::kMinAutoMaxLen, a.int8_head, at), smallest));
    std::fprintf(stderr, "max_len: auto -> %u (the largest multiple of %u that fits every card with a %.3f GB "
                 "reserve; trained context %u)\n", len, runtime::kMaxLenQuantum, reserve_gb, d.trained_max_len);
  } else {
    if (len > d.trained_max_len)
      throw std::runtime_error("--max-len " + std::to_string(len) + " exceeds LFM2.5-24B-A2B's trained context " +
                               std::to_string(d.trained_max_len) + " (max_position_embeddings)");
    std::fprintf(stderr, "max_len: %u (--max-len)\n", len);
  }
  for (uint32_t i = 0; i < p.devices; ++i) {
    rl::Plan pl = rl::plan(d, p, i, len, a.int8_head, at);
    pl.reserve = size_t(reserve_gb * 1e9);
    rl::require_fits(pl, dev[i]);
    std::fprintf(stderr, "plan, device %u: %s\n", i, rl::describe(pl, dev[i]).c_str());
  }
  if (p.devices == 2)
    std::fprintf(stderr, "split: %s%u (layers [0, %u) on device 0, [%u, %u) and the head on device 1)\n",
                 a.pipe.split_auto ? "auto -> " : "", p.split, p.split, p.split, d.layers);
  return {p, len};
}

// The loader prints its report to stdout (its convention); `Redirect` is the caller's StdoutToStderr so the report
// joins the diagnostics on stderr.
template <class Redirect>
int run_decode(const DecodeArgs& a) {
  namespace rl = runtime::lfm2moe;
  const model::Lfm2MoeDesc pre = loader::lfm2moe_checkpoint_desc(a.path);
  check_args(a, pre);
  // The switches, read once (a bad value throws before the device).
  const rl::LfmAttn attn = rl::lfm_attn();
  const rl::LfmFuse fuse = rl::lfm_fuse();
  std::vector<uint32_t> ids;
  if (!a.bench && !a.profile) {
    ids = read_ids(a.ids_path, pre.vocab);
  } else {
    ids.resize(a.depth);
    for (uint32_t i = 0; i < a.depth; ++i) ids[i] = kBenchPrompt[i % kBenchPromptLen];
  }
  const uint32_t after = a.profile ? a.steps * a.repeats : (a.bench ? a.tg : a.n);
  const size_t need = ids.size() + after;
  if (!a.max_len.is_auto && need > a.max_len.value)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds --max-len " +
                             std::to_string(a.max_len.value));
  const uint32_t devs = a.pipe.devices;
  if (devs == 2) require_two_devices(l0::Context::gpu_count());
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
  const std::pair<model::LfmPlacement, uint32_t> pl = settle(pre, a, dev);
  const uint32_t max_len = pl.second;
  if (need > max_len)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds max_len " +
                             std::to_string(max_len));
  loader::LfmLoadOptions lo;
  lo.int8_head = a.int8_head;
  lo.conv_tiles = fuse.conv;   // the arm conv reads in_proj in the conv-tile layout
  lo.max_len = max_len;
  loader::LfmLoaded model = [&] {
    Redirect redirect;
    return loader::load_lfm2moe(ctx, a.path, lo, pl.first);
  }();
  runtime::PipelineOptions opt;
  opt.handoff = a.pipe.handoff;
  rl::Lfm2MoeEngine eng(ctx, std::move(model), max_len, /*debug_tap=*/false, opt, a.profile, attn, fuse);
  std::fprintf(stderr, "engine: LFM2.5-24B-A2B, %u device(s)%s, %zu launches a token (%s attention B70_LFM_ATTN, fuse %s "
               "B70_LFM_FUSE), %s head, max_len %u\n", eng.devices(),
               devs == 2 ? (std::string(", split ") + std::to_string(eng.split()) + ", hand-off " +
                            runtime::pp_handoff_name(eng.handoff())).c_str() : "",
               eng.launches(), rl::lfm_attn_name(eng.attention()), rl::lfm_fuse_name(eng.fuse()).c_str(),
               a.int8_head ? "int8" : "bf16", eng.max_len());
  std::fprintf(stderr, "%s\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  eng.ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token, one decode replay per id), pos %u\n", ids.size(),
               ingest_ms, ingest_ms / double(ids.size()), eng.pos());
  if (a.profile) {
    eng.clear_profile();
    for (uint32_t r = 0; r < a.repeats; ++r) eng.generate(a.steps);
    std::printf("# b70-decode --profile - LFM2.5-24B-A2B in-situ step anatomy\ndepth %u, steps %u, repeats %u, %zu "
                "launches/step, max_len %u, %s head (a profiled list signals a kernel-timestamp event per launch: never "
                "a bench row)\n\n%s", a.depth, a.steps, a.repeats, eng.launches(), max_len, a.int8_head ? "int8" : "bf16",
                eng.profile_table().c_str());
    return 0;
  }
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
  const double gb = double(read_per_token(eng.model().desc, a.int8_head)) / 1e9;
  std::fprintf(stderr, "  MBU: %.2f t/s x %.3f GB = %.0f GB/s (W = the derived weights read a token, KV excluded); %zu "
               "launches/token\n", eng.last_tok_per_s(), gb, eng.last_tok_per_s() * gb, eng.launches());
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode %s lfm2moe %s-head %s fuse-%s%s | %u | %u | %.2f | %.2f |\n", sha,
              a.int8_head ? "int8" : "bf16", rl::lfm_attn_name(eng.attention()), rl::lfm_fuse_name(eng.fuse()).c_str(),
              devs == 2 ? (std::string(" pipeline2-") + runtime::pp_handoff_name(eng.handoff()) + "-s" +
                           std::to_string(eng.split())).c_str() : "",
              a.depth, a.tg, eng.last_tok_per_s(), eng.last_gen_ms() / double(a.tg));
  return 0;
}

}  // namespace cli::lfm2moe

#pragma once
// Spec 25c: b70-decode's Nemotron 3.5 Bolt path - the CLI dispatches on config.json's model_type nemotron_h_omni over
// llm_config.model_type nemotron_h_puzzle (loader::is_bolt_checkpoint) and runs runtime::bolt::BoltEngine on two cards
// (the real model's default: 40.46 GB of weights - Decision 4) or one (a synthetic or --layers N checkpoint that fits).
// cli/lfm2moe_decode.h's shape:
//
//   --ids F --n N        the prompt one decode replay per id, then N greedy ids on stdout
//   --bench [--depth N] [--tg N]   depth N of the bench prompt ingested, N decoded and timed (the bench row on stdout)
//   --profile [--depth N] [--steps N] [--repeats R]   depth N ingested, then R x N profiled replays: the per-launch
//                        anatomy (kernel-timestamp events, docs/15's instrument; never a bench row)
//   --lm-head bf16|int8  int8 the default (Decision 18: built from lm_head at load); bf16 reads lm_head's own tiles
//   --max-len N|auto     16384 the default (bench rows comparable); auto = the largest multiple of 256 that fits every card,
//                        at most the trained 262144 (Decision 14)
//   --pp N / --pipeline-parallel-size N   2 the default for the whole model, 1 for a truncated one; --pp 1 on the whole
//                        model is refused by the plan naming the bytes and Decision 4
//   --pipeline-split auto|N, --pipeline-handoff copy|peer   --pp 2's (auto = runtime::bolt::pp_split: 48)
//   --layers N           development mode: load only blocks [0, N) (the head reads block N-1's residual)
//   --prefill            (--ids) spec 25d: the prompt through BoltEngine::prefill in chunks, then --n greedy ids
//   --bench --prefill-length N    depth N PREFILLED and timed (the pp row, a second stdout line, with its TFLOP/s against
//                        the descriptor's matmul parameters a token), then --tg decoded
//   --prefill-chunk C    the chunk width: a multiple of 64, at most 2048 (the default) - every chunk ends on the absolute
//                        64-grid (runtime::bolt::prefill_chunk_rows)
//   --prefill-backend l0 the only backend (spec 25d)
// B70_BOLT_ATTN=flash|eager, B70_BOLT_FUSE=none|all|<router,route,conv> and B70_BOLT_SSM=fp32|bf16 pick the list (read at
// construction; the prefill follows the attention form and the state precision). B70_PREFILL_PROFILE=1 prints the prefill's
// phase table (runtime/prefill/profile.h; the Mamba launches under the gdn_* rows) after the timed call. Refused before the
// device, each by name (tests/CMakeLists.txt's cli_reject_bolt_*): --prefill-backend l0-int8 / sycl-tla and a
// --prefill-chunk that is not a multiple of 64 or above 2048 (spec 25d), --mtp (Bolt has no MTP head), --spec (no
// drafter either), --kv-cache int8 (Decision 17), --device with --pp 2.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/bolt_layout.h"
#include "loader/bolt_loader.h"
#include "loader/snapshot.h"
#include "runtime/bolt/bolt_engine.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/prefill/profile.h"
#include "runtime/prefill_backend.h"

namespace cli::bolt {

// The bench prompt, cycled to --depth: tests/golden/prompts/bolt_short_prose.ids (spec 25a: tests/golden/prompts/prose.txt
// through Bolt's own tokenizer, no BOS - the template writes none) - "The harbour at dawn was quiet enough to hear the
// ropes creak against the bollards. ..."
inline constexpr uint32_t kBenchPrompt[] = {
    1784,  78493, 1513, 54507, 1486, 17417, 6171,  1317,  12459, 1278,   105888, 2446,  1489,  3675, 1278,
    1289,  1932,  4058, 1046,  114904, 1115, 52995, 12421, 1408, 1278,  4323,   17083, 3016,  1278, 78272,
    108813, 2246, 16853, 1046, 5652,  11282, 1278,  7232,  2168, 1402,  25402,  2790,  1046};
inline constexpr size_t kBenchPromptLen = sizeof(kBenchPrompt) / sizeof(kBenchPrompt[0]);

// The B70's device memory as Level Zero reports it (32.530 GB): the planner's check before any device is opened.
inline constexpr size_t kB70Bytes = size_t(32530000000ull);

// True when `path` resolves to a snapshot whose config.json is Bolt's. A path that does not resolve is not Bolt: the
// caller's own flow then reports it.
inline bool is_bolt(const std::string& path) {
  try {
    return loader::is_bolt_checkpoint(loader::resolve_snapshot(path));
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
  bool int8_head = true;        // --lm-head (Decision 18's int8 head unless bf16)
  PipelineArgs pipe;            // devices as parsed (1 when --pp was not given)
  bool pp_given = false;        // --pp / --pipeline-parallel-size on the command line
  uint32_t layers = 0;          // --layers N (0: the checkpoint's)
  // Spec 25d: --prefill (--ids), --prefill-length N (--bench at depth N: the pp row), --prefill-chunk C (0 = kPfC),
  // --prefill-backend as parsed (l0 only).
  bool prefill = false, pp = false, pp_backend_given = false;
  uint32_t pp_chunk = 0;
  runtime::PrefillBackend pp_backend = runtime::PrefillBackend::L0;
  // What else was asked for, for the refusals.
  bool mtp = false, spec = false, kv8 = false;
  bool prefills() const { return prefill || pp; }
};

// --pp's default: two cards for the whole model (Decision 4), one for a truncated / synthetic one.
inline uint32_t devices_of(const DecodeArgs& a, const model::BoltDesc& d) {
  if (a.pp_given) return a.pipe.devices;
  return d.layers == model::bolt().layers ? 2u : 1u;
}

// Every refusal that needs no device - in this order, each by name (tests/CMakeLists.txt's cli_reject_bolt_*).
inline void check_args(const DecodeArgs& a, const model::BoltDesc& d) {
  if (a.pp_backend_given && a.pp_backend == runtime::PrefillBackend::L0Int8)
    throw std::runtime_error("Nemotron 3.5 Bolt prefills on the l0 backend only (spec 25d), not l0-int8: int8 prefill for "
                             "this family is out of scope (spec 25 §11) - --prefill-backend l0, or omit it");
  if (a.pp_backend_given && a.pp_backend == runtime::PrefillBackend::SyclTla)
    throw std::runtime_error("Nemotron 3.5 Bolt prefills on the l0 backend only (spec 25d), not sycl-tla: sycl-tla has no "
                             "Bolt walk (no SSD scan, no latent MoE) - --prefill-backend l0, or omit it");
  if (a.pp_chunk > runtime::bolt::kPfC)
    throw std::runtime_error("--prefill-chunk " + std::to_string(a.pp_chunk) + " exceeds Nemotron 3.5 Bolt's prefill chunk " +
                             std::to_string(runtime::bolt::kPfC) + " (the scratch is sized for it)");
  if (a.pp_chunk && a.pp_chunk % runtime::bolt::kPfSub)
    throw std::runtime_error("--prefill-chunk " + std::to_string(a.pp_chunk) + " is not a multiple of " +
                             std::to_string(runtime::bolt::kPfSub) + ": the SSD's sub-chunks are absolute 64-row windows "
                             "and every chunk ends on that grid (spec 25d)");
  if (a.mtp) throw std::runtime_error("Nemotron 3.5 Bolt has no MTP head (spec 25 §0: num_nextn_predict_layers 0): drop --mtp");
  if (a.spec)
    throw std::runtime_error("--spec: Nemotron 3.5 Bolt has no speculative decoding (no MTP head, no drafter)");
  if (a.kv8)
    throw std::runtime_error("--kv-cache int8: Nemotron 3.5 Bolt's KV is bf16 (spec 25 Decision 17: 8 KiB a position)");
  const uint32_t devs = devices_of(a, d);
  if (devs == 1) {
    if (a.pipe.have_split) throw std::runtime_error("--pipeline-split belongs to --pp 2 (the block device 1 starts at)");
    if (a.pipe.have_handoff) throw std::runtime_error("--pipeline-handoff belongs to --pp 2 (how the residual crosses)");
    return;
  }
  if (a.device != l0::Context::kFromEnv)
    throw std::runtime_error("--pp 2 runs on GPUs 0 and 1 of what Level Zero shows (ZE_AFFINITY_MASK picks which two); "
                             "--device names one card - drop it");
  if (d.layers < 2) throw std::runtime_error("--pp 2 needs at least two blocks to split (--layers " + std::to_string(d.layers) + ")");
  if (!a.pipe.split_auto && (a.pipe.split < 1 || a.pipe.split >= d.layers))
    throw std::runtime_error("--pipeline-split " + std::to_string(a.pipe.split) + " leaves a device no block: it is the "
                             "first block of device 1, in [1, " + std::to_string(d.layers - 1) + "]");
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
      throw std::runtime_error("--ids file '" + path + "': id " + w + " is outside Nemotron 3.5 Bolt's vocabulary (" +
                               std::to_string(vocab) + " rows)");
    ids.push_back(uint32_t(v));
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

// The bytes a decode step reads (derived): every block's int4 linears and router, top_k experts' blocks, the small
// blocks, the head (int8 or bf16 tiles), the embedding row - the MBU line's W (the KV and the SSM state excluded).
inline size_t read_per_token(const model::BoltDesc& d, bool int8_head) {
  size_t b = size_t(d.hidden) * 2 + size_t(d.hidden) * 4;   // the gathered row, norm_f
  for (uint32_t l = 0; l < d.layers; ++l) {
    for (const model::BoltLinear& ln : d.linears(l)) {
      if (ln.id == model::BoltLinearId::ExpertUp || ln.id == model::BoltLinearId::ExpertDown) continue;
      b += ln.kind == model::WeightKind::Int4 ? loader::bolt::int4_bytes(ln.shape) : size_t(ln.shape.K) * ln.shape.N * 2;
    }
    if (d.is_moe(l)) b += size_t(d.top_k) * loader::bolt::expert_block_bytes(d.latent, d.inter.at(l));
    b += loader::bolt::small_block(d, l).bytes;
  }
  return b + (int8_head ? loader::bolt::head_int8_bytes(d) : loader::bolt::head_bf16_bytes(d));
}

// The plan's BoltReserve for this run: the decode scratch the engine allocates (the largest device's, at the length
// bound - eager's score row scales with max_len) and the SSM state's precision.
inline runtime::bolt::BoltReserve reserve_for(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t len,
                                              runtime::bolt::BoltAttn at, runtime::bolt::BoltState st, bool prefills = false) {
  runtime::bolt::BoltReserve r;
  for (uint32_t dev = 0; dev < p.devices; ++dev) {
    r.decode_scratch = std::max(r.decode_scratch, runtime::bolt::decode_scratch_bytes(d, p, dev, len, at, st));
    if (prefills)   // spec 25d: the prefill scratch (+ the prefill link on two cards), the largest device's
      r.prefill_scratch = std::max(r.prefill_scratch, runtime::bolt::prefill_scratch_bytes(d, p, dev));
  }
  r.bf16_state = st == runtime::bolt::BoltState::Bf16;
  return r;
}

// The matmul parameters a prefilled token multiplies through (derived from the descriptor): every int4 linear but the
// routed experts, the routers, top_k routed experts' up + down a MoE block - the pp row's TFLOP/s is 2 x this x t/s (the
// attention's depth-dependent q K^T / P V and the last row's head excluded).
inline double prefill_params_per_token(const model::BoltDesc& d) {
  double n = 0;
  for (uint32_t l = 0; l < d.layers; ++l)
    for (const model::BoltLinear& ln : d.linears(l)) {
      const double kn = double(ln.shape.K) * ln.shape.N;
      if (ln.id == model::BoltLinearId::ExpertUp || ln.id == model::BoltLinearId::ExpertDown)
        n += d.top_k * kn;
      else
        n += kn;
    }
  return n;
}

// The placement and max_len for this run, from the planner (before the load). Prints the choice and the plan.
inline runtime::bolt::BoltPlan settle(const model::BoltDesc& d, const DecodeArgs& a, const std::vector<size_t>& dev) {
  namespace rb = runtime::bolt;
  const rb::BoltAttn at = rb::bolt_attn();
  const rb::BoltState st = rb::bolt_state();
  const uint32_t devs = devices_of(a, d);
  const model::BoltPlacement place = devs == 2
                                         ? model::BoltPlacement::two(d, a.pipe.split_auto ? rb::pp_split(d, a.int8_head)
                                                                                          : a.pipe.split)
                                         : model::BoltPlacement::one(d);
  const uint32_t bound = a.max_len.is_auto ? d.max_pos : a.max_len.value;
  const size_t reserve = a.reserve ? a.reserve : size_t(runtime::kDefaultReserveGb * 1e9);
  const rb::BoltPlan p = rb::plan(d, a.max_len.is_auto ? 0 : a.max_len.value, dev,
                                  reserve_for(d, place, bound, at, st, a.prefills()),
                                  devs == 2 ? std::optional<uint32_t>(place.split) : std::nullopt, reserve, a.int8_head);
  if (a.max_len.is_auto)
    std::fprintf(stderr, "max_len: auto -> %u (the largest multiple of 256 that fits every card; trained context %u)\n",
                 p.max_len, d.max_pos);
  else
    std::fprintf(stderr, "max_len: %u (--max-len)\n", p.max_len);
  for (uint32_t i = 0; i < p.place.devices; ++i)
    std::fprintf(stderr, "plan, device %u: %s\n", i, rb::describe(p.device[i], dev[i]).c_str());
  if (p.place.devices == 2)
    std::fprintf(stderr, "split: %s%u (blocks [0, %u) on device 0, [%u, %u) and the head on device 1)\n",
                 a.pipe.split_auto ? "auto -> " : "", p.place.split, p.place.split, p.place.split, d.layers);
  return p;
}

// The loader prints its report to stdout (its convention); `Redirect` is the caller's StdoutToStderr so the report
// joins the diagnostics on stderr.
template <class Redirect>
int run_decode(const DecodeArgs& a) {
  namespace rb = runtime::bolt;
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(a.path, a.layers);
  check_args(a, pre);
  // The switches, read once (a bad value throws before the device).
  const rb::BoltAttn attn = rb::bolt_attn();
  const rb::BoltFuse fuse = rb::bolt_fuse();
  const rb::BoltState state = rb::bolt_state();
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
  const uint32_t devs = devices_of(a, pre);
  if (devs == 2) require_two_devices(l0::Context::gpu_count());
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(devs == 2 ? 0u : a.device);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  std::vector<size_t> dev = {c0->memory_bytes()};
  if (c1) dev.push_back(c1->memory_bytes());
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)%s\n", c0->name().c_str(), c0->eu_count(),
                 a.device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  else
    std::fprintf(stderr, "devices: %s + %s (--pp 2)\n", c0->name().c_str(), c1->name().c_str());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--pp 2: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), and both "
                             "hand-offs write it (docs/10-the-box.md)");
  const rb::BoltPlan pl = settle(pre, a, dev);
  if (need > pl.max_len)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds max_len " +
                             std::to_string(pl.max_len));
  loader::BoltLoadOptions lo;
  lo.int8_head = a.int8_head;
  lo.max_len = pl.max_len;
  lo.layers = a.layers;
  loader::BoltLoaded model = [&] {
    Redirect redirect;
    return loader::load_bolt(ctx, a.path, lo, pl.place);
  }();
  runtime::PipelineOptions opt;
  opt.handoff = a.pipe.handoff;
  rb::BoltEngine eng(ctx, std::move(model), pl.max_len, /*debug_tap=*/false, opt, a.profile, attn, fuse, state);
  std::fprintf(stderr, "engine: Nemotron 3.5 Bolt, %u block(s), %u device(s)%s, %zu launches a token (%s attention "
               "B70_BOLT_ATTN, fuse %s B70_BOLT_FUSE, %s SSM state B70_BOLT_SSM), %s head, max_len %u\n",
               eng.model().desc.layers, eng.devices(),
               devs == 2 ? (std::string(", split ") + std::to_string(eng.split()) + ", hand-off " +
                            runtime::pp_handoff_name(eng.handoff())).c_str() : "",
               eng.launches(), rb::bolt_attn_name(eng.attention()), rb::bolt_fuse_name(eng.fuse()).c_str(),
               rb::bolt_state_name(eng.state()), a.int8_head ? "int8" : "bf16", eng.max_len());
  if (a.layers)
    std::fprintf(stderr, "layers: %u of the checkpoint's %u (--layers: development mode; the head reads block %u's "
                 "residual)\n", eng.model().desc.layers, eng.model().checkpoint_layers, eng.model().desc.layers - 1);
  // Spec 25d: the prefill setup (every device's scratch, Context, the prefill hand-off, the binaries' check) happens here,
  // outside the timed window, which is then the whole prefill() call: first launch to the first generated id in cur_token.
  const uint32_t chunk = a.pp_chunk ? a.pp_chunk : rb::kPfC;
  size_t pp_launches0 = 0;
  if (a.prefills()) {
    eng.prepare_prefill();
    pp_launches0 = eng.prefill_launches();
    std::fprintf(stderr, "prefill: l0 backend, chunk %u (every chunk ends on the absolute 64-grid), attention %s "
                 "(B70_BOLT_ATTN), %s SSM state (B70_BOLT_SSM), %zu launches per chunk (%s) + %zu for the head%s\n", chunk,
                 rb::bolt_attn_name(eng.attention()), rb::bolt_state_name(eng.state()),
                 rb::prefill_chunk_launches(eng.model().desc, eng.model().place), devs == 2 ? "both cards" : "one card",
                 rb::kPrefillHeadLaunches,
                 devs == 2 ? "; the chunk crosses by copy, sequentially (spec 16c's overlap is a later lever)" : "");
    runtime::prefill::profile_reset();
  }
  std::fprintf(stderr, "%s\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  if (a.prefills())
    eng.prefill(ids, chunk);
  else
    eng.ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const double pp_tps = ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0;
  const double pp_params = prefill_params_per_token(eng.model().desc);
  if (a.prefills()) {
    std::fprintf(stderr, "%s: %zu ids in %.1f ms (%.2f t/s, %.2f TFLOP/s at 2 x %.3f B matmul parameters a token - "
                 "derived, attention's depth term excluded) - device-side, loader excluded, first prefill launch to the "
                 "first generated id; chunk %u, %zu L0 launches, pos %u\n", a.pp ? "pp" : "prefill", ids.size(), ingest_ms,
                 pp_tps, pp_tps * 2.0 * pp_params / 1e12, pp_params / 1e9, chunk, eng.prefill_launches() - pp_launches0,
                 eng.pos());
    if (runtime::prefill::profile_enabled()) runtime::prefill::profile_report("Nemotron 3.5 Bolt prefill", ingest_ms);
  } else {
    std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token, one decode replay per id), pos %u\n", ids.size(),
                 ingest_ms, ingest_ms / double(ids.size()), eng.pos());
  }
  if (a.profile) {
    eng.clear_profile();
    for (uint32_t r = 0; r < a.repeats; ++r) eng.generate(a.steps);
    std::printf("# b70-decode --profile - Nemotron 3.5 Bolt in-situ step anatomy\ndepth %u, steps %u, repeats %u, %zu "
                "launches/step, max_len %u, %s head (a profiled list signals a kernel-timestamp event per launch: never a "
                "bench row)\n\n%s", a.depth, a.steps, a.repeats, eng.launches(), pl.max_len, a.int8_head ? "int8" : "bf16",
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
  std::fprintf(stderr, "  MBU: %.2f t/s x %.3f GB = %.0f GB/s (W = the derived weights read a token, KV and SSM state "
               "excluded); %zu launches/token\n", eng.last_tok_per_s(), gb, eng.last_tok_per_s() * gb, eng.launches());
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode %s bolt %s-head %s fuse-%s ssm-%s%s | %u | %u | %.2f | %.2f |\n", sha,
              a.int8_head ? "int8" : "bf16", rb::bolt_attn_name(eng.attention()), rb::bolt_fuse_name(eng.fuse()).c_str(),
              rb::bolt_state_name(eng.state()),
              devs == 2 ? (std::string(" pipeline2-") + runtime::pp_handoff_name(eng.handoff()) + "-s" +
                           std::to_string(eng.split())).c_str() : "",
              a.depth, a.tg, eng.last_tok_per_s(), eng.last_gen_ms() / double(a.tg));
  // Spec 25d: the pp row, a second line as the LFM2 / Kolibri / Qwen paths print it (depth, chunk, ms, t/s).
  if (a.pp)
    std::printf("| b70-decode %s bolt %s-head %s ssm-%s%s l0 pp | %u | %u | %.1f | %.2f |\n", sha,
                a.int8_head ? "int8" : "bf16", rb::bolt_attn_name(eng.attention()), rb::bolt_state_name(eng.state()),
                devs == 2 ? (std::string(" pipeline2-copy-s") + std::to_string(eng.split())).c_str() : "", a.depth, chunk,
                ingest_ms, pp_tps);
  return 0;
}

}  // namespace cli::bolt

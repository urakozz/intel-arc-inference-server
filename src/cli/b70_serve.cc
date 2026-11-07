// b70-serve -- the OpenAI-compatible server over the replayed runtime::Engine (Qwen3.8,
// Ornith), runtime::k2::K2Engine (K2-Horizon, spec 18d) or runtime::kolibri::KolibriEngine
// (Kolibri-1, spec 20e: two cards by default), picked by config.json's model_type; with --parallel pp
// (spec 16d) over runtime::PipelineEngine, the Qwen-family model split over two B70s
// (cli/pipeline_serve.h has what it serves and refuses).
#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "cli/bolt_decode.h"    // spec 25c: is_bolt, check_args, settle
#include "cli/bolt_serve.h"     // spec 25e: BoltEngineAdapterT, the generation defaults, the refusals
#include "cli/bolt_spec.h"      // spec 26c: --spec dflash / lookup for Bolt (LiLiCorr)
#include "cli/dflash_args.h"    // spec 19c: --spec dflash
#include "cli/k2_decode.h"
#include "cli/hardware_gate.h"
#include "cli/k2_serve_adapter.h"
#include "cli/kolibri_decode.h"
#include "cli/kolibri_serve.h"
#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "cli/pipeline_serve.h"
#include "cli/pipeline_serve_adapter.h"
#include "cli/pipeline_settle.h"
#include "cli/prefix_cache_size.h"
#include "cli/qwen4exp_chat.h"
#include "cli/qwen4exp_serve.h"
#include "cli/replica_engine.h"     // plan 13b: --parallel replica
#include "cli/served_name.h"
#include "cli/spec_args.h"
#include "cli/serve_adapters.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/qwen4exp_loader.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"
#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/backend.h"
#include "server/lookup_in_dflash.h"   // spec 19f: --spec dflash-lookup's constants (the startup line)
#include "server/server.h"

namespace {

class StdoutToStderr {
 public:
  StdoutToStderr() {
    std::fflush(stdout);
    saved_ = ::dup(1);
    if (saved_ >= 0 && ::dup2(2, 1) < 0) {
      ::close(saved_);
      saved_ = -1;
    }
  }
  ~StdoutToStderr() {
    std::fflush(stdout);
    if (saved_ >= 0) {
      ::dup2(saved_, 1);
      ::close(saved_);
    }
  }
  StdoutToStderr(const StdoutToStderr&) = delete;
  StdoutToStderr& operator=(const StdoutToStderr&) = delete;

 private:
  int saved_ = -1;
};

server::Server* g_server = nullptr;

void stop_server(int) {
  if (g_server != nullptr) g_server->stop();
}

void usage() {
  std::fprintf(stderr,
               "usage: b70-serve <snapshot-or-repo> [--spec auto|off|mtp|lookup|dflash] [--max-len auto|N]\n"
               "                 [--parallel auto|1|pp|replica] [--host 0.0.0.0] [--port 8000]\n"
               "\n"
               "  The model is chosen from the checkpoint's config.json (Qwen3.8, Ornith 1.5,\n"
               "  K2-Horizon, Kolibri-1, Qwen3.8-Flash-Next, Nemotron 3.5 Bolt) and served under its name\n"
               "  (the repo id, or the directory's). Everything else - the device memory the plan leaves free,\n"
               "  the int8 lm_head, the drafter's form, the speculative cost tables, the tokenizer, the chat\n"
               "  template and the generation defaults (the snapshot's) - the engine decides; what does not fit\n"
               "  is refused at startup, naming why. Cards: Intel Arc Pro B70, one or two (peer access between\n"
               "  them); ZE_AFFINITY_MASK picks which the driver shows.\n"
               "\n"
               "  --spec S        speculative decoding (default auto: the model's own MTP head with an adaptive\n"
               "                  K per request where it has one and its cost table is measured - Qwen3.8,\n"
               "                  Ornith; off elsewhere). off; mtp; lookup (prompt lookup, verified\n"
               "                  through the MTP lists); dflash (Qwen3.8's DFlash drafter, from the HF cache);\n"
               "                  dflash-lookup (lookup inside DFlash, opt-in). Greedy output unchanged.\n"
               "                  Nemotron 3.5 Bolt (spec 26c, opt-in): lookup and dflash (its LiLiCorr\n"
               "                  drafter on the last card), both over its own verify lists.\n"
               "  --spec-max K    the proposer's cap: 1..3 drafts (mtp, lookup), 1..7 (dflash; Bolt's lookup\n"
               "                  too). Default the most.\n"
               "  --draft-model P --spec dflash's drafter (default z-lab/Qwen3.8-27B-DFlash2; Bolt's\n"
               "                  nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr).\n"
               "  --max-len L     auto (default): the largest context that fits beside the weights, at most\n"
               "                  the trained one; N: any multiple of 256 up to it, refused if it does not fit.\n"
               "  --parallel P    auto (default): one card when the model and the asked context fit it, else\n"
               "                  pp; 1: one card; pp: the layers split over two cards (spec 16). Kolibri-1 and\n"
               "                  Nemotron 3.5 Bolt run on two cards by default (the model needs both).\n"
               "                  replica (Qwen3.8): the model on both cards, one engine each - two\n"
               "                  requests at once, each exactly what it is alone on its card.\n"
               "  --pipeline-split auto|N   pp: device 0 runs layers [0, N) (auto: the split whose heavier card\n"
               "                  holds the fewest bytes). An override, for experiments; it implies pp.\n"
               "  --host H, --port P   where to listen (0.0.0.0:8000).\n"
               "  --log-requests DIR   write DIR/NNNNNN.json per request (timings, ids, text).\n"
               "  --prefix-cache-gb auto|N   the prefix cache in SYSTEM RAM (pinned host memory), GiB. auto\n"
               "                  (default): min(32, half the RAM, available - 8), off below 4; 0 = off.\n"
               "  --prefix-split-last   chat requests: prefill to len - 1 and run the last id as one decode\n"
               "                  step, so a thinking model's next turn restores right there (agentic sessions;\n"
               "                  off for benchmarks and the golden gates).\n"
               "  --layers N|auto Qwen3.8-Flash-Next only, REQUIRED until spec 22: load its first N layers\n"
               "                  (development mode); its PLE int8 file is <snapshot>-ple-int8/.\n"
               "\n"
               "  Still being measured (they go when their numbers decide, docs/open-work.md):\n"
               "  --kv-cache bf16|int8, --prefill-backend l0|l0-int8, --pipeline-handoff copy|peer,\n"
               "  --draft-vocab off|32k|64k|128k [--draft-vocab-ids FILE].\n");
}

uint32_t parse_u32(const char* what, const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos ||
      value.size() > 10) {
    throw std::runtime_error(std::string(what) + " expects a non-negative integer, got '" +
                             value + "'");
  }
  const unsigned long long parsed = std::stoull(value);
  if (parsed > 0xFFFFFFFFull)
    throw std::runtime_error(std::string(what) + " is out of range: " + value);
  return static_cast<uint32_t>(parsed);
}

std::vector<uint32_t> eos_ids(const std::string& snapshot_dir) {
  std::ifstream input(snapshot_dir + "generation_config.json");
  if (!input) throw std::runtime_error("cannot open '" + snapshot_dir + "generation_config.json'");
  nlohmann::json config;
  input >> config;
  try {
    return server::generation_eos_ids(config);   // an integer or a list (spec 24e moved the reading out for a test)
  } catch (const std::invalid_argument& error) {
    throw std::runtime_error(error.what());
  }
}

void print_eos(const std::vector<uint32_t>& ids) {
  std::fputc('[', stderr);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) std::fputs(", ", stderr);
    std::fprintf(stderr, "%u", ids[i]);
  }
  std::fputc(']', stderr);
}

// Spec 7: the prefix cache's pinned system RAM, sized after the model is loaded (so
// "available" already excludes the process's own load) - shared by the Qwen-family and the
// K2 paths. Returns null when the cache is off; sets options.prefix_cache_bytes / prefix_alloc.
std::unique_ptr<PinnedAlloc> make_prefix_alloc(l0::Context& context, bool prefix_auto,
                                               uint32_t prefix_cache_gb, server::Options& options) {
  std::unique_ptr<PinnedAlloc> prefix_alloc;
  const cli::HostMemory host = cli::host_memory();
  std::string prefix_why = "explicit";
  if (prefix_auto) {
    const cli::PrefixCacheChoice c = cli::auto_prefix_cache(host);
    prefix_cache_gb = c.gib;
    prefix_why = c.why;
  } else if (prefix_cache_gb > 0 && host.known && (uint64_t(prefix_cache_gb) << 30) > host.available) {
    std::fprintf(stderr,
                 "warning: --prefix-cache-gb %u pins more host RAM than is available (%.0f GiB);"
                 " pinned pages cannot be swapped\n",
                 prefix_cache_gb, double(host.available) / double(cli::kGiB));
  }
  if (prefix_cache_gb > 0) {
    const auto t0 = std::chrono::steady_clock::now();
    prefix_alloc = std::make_unique<PinnedAlloc>(context, size_t(prefix_cache_gb) << 30);
    options.prefix_cache_bytes = prefix_alloc->mem.size();
    options.prefix_alloc = prefix_alloc.get();
    std::fprintf(stderr, "prefix cache: %u GiB pinned system RAM (%s), allocated in %.0f ms\n",
                 prefix_cache_gb, prefix_why.c_str(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                     .count());
  } else {
    std::fprintf(stderr, "prefix cache: off (%s)\n", prefix_why.c_str());
  }
  return prefix_alloc;
}

int listen_until_stopped(server::Server& server, const server::Options& options) {
  g_server = &server;
  std::signal(SIGTERM, stop_server);
  std::signal(SIGINT, stop_server);
  const bool listened = server.listen();
  g_server = nullptr;
  if (!listened)
    throw std::runtime_error("failed to bind http://" + options.host + ":" +
                             std::to_string(options.port));
  return 0;
}

// Spec 18d: K2-Horizon served - runtime::k2::K2Engine behind cli::k2::K2EngineAdapterT (KV-only
// prefix-cache snapshots, host sampling), K2's chat format (server::ChatFormat, set by run()
// from model_type), its EOS [1, 250019] from generation_config.json. The refusals are run()'s,
// before this is called (before the device).
struct K2Serve {
  std::string path, snapshot_dir;
  cli::MaxLenArg max_len_arg;
  size_t mem_reserve = 0;
  runtime::KvCache kv_cache = runtime::KvCache::Bf16;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
};

int serve_k2(const K2Serve& a, server::Options options) {
  const std::vector<uint32_t> eos = eos_ids(a.snapshot_dir);
  const uint32_t trained = loader::trained_context(a.snapshot_dir);
  cli::check_before_load(a.max_len_arg, trained, /*require_quantum=*/true);   // spec 6 §10
  l0::Context context(0u);
  std::fprintf(stderr, "device: %s (%u EUs)\n", context.name().c_str(), context.eu_count());
  loader::K2LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load_k2(context, a.path, cli::load_len(a.max_len_arg, trained), a.lm_head);
  }();
  // The server prefills every request: the plan carries K2's prefill scratch (spec 18c).
  const uint32_t max_len =
      cli::k2::settle(context, model, a.max_len_arg, a.mem_reserve, /*prefill=*/true, a.kv_cache);
  runtime::k2::K2Engine engine(context, std::move(model), max_len, /*debug_tap=*/false, a.kv_cache);
  engine.prepare_prefill();   // the scratch and the binaries' check at load, not in a request
  std::fprintf(stderr, "%s\n", engine.memory_line().c_str());
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  if (tokenizer.vocab_used() != engine.vocab())
    std::fprintf(stderr, "note: tokenizer.json defines %u ids, K2-Horizon's vocabulary is %u; "
                         "sampling masks from %u\n",
                 tokenizer.vocab_used(), engine.vocab(), tokenizer.vocab_used());
  TemplateAdapter chat_template(a.snapshot_dir);
  cli::k2::K2EngineAdapterT<runtime::k2::K2Engine> engine_adapter(engine, tokenizer.vocab_used());
  options.eos_ids = eos;
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(context, a.prefix_auto, a.prefix_cache_gb, options);
  options.parallel_desc = "1 card";
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::fprintf(stderr, ", K2-Horizon (%s chat format), prefill backend l0, attention flash, "
                       "mtp none, lm_head %s, kv cache %s, prefix snapshots KV-only\n",
               options.chat_format.name(), loader::lm_head_form_name(a.lm_head), runtime::kv_cache_name(engine.kv_cache()));
  return listen_until_stopped(server, options);
}

// Spec 20e: Kolibri-1 served - runtime::kolibri::KolibriEngine (one card or, the default, two: spec 16b's
// pieces as 20c / 20d built them) behind cli::kolibri::KolibriEngineAdapterT (the rings + full-KV prefix
// snapshots, host sampling at the tokenizer's id count), Kolibri's chat format (server::ChatFormat,
// set by run() from model_type), its EOS [127906, 127901] and its sampling defaults from
// generation_config.json. b70-decode's flow (cli/kolibri_decode.h run_decode): the refusals before the
// device (cli::kolibri::check_args: --spec mtp, --kv-cache int8, the prefill backends, a --parallel 1 that does not
// fit), both cards in one Level Zero context with peer access, the placement and
// max_len planned before the load with the prefill scratch (every request prefills), each layer loaded
// straight onto its device.
struct KolibriServe {
  std::string snapshot_dir;
  cli::kolibri::DecodeArgs args;   // the planner's and the refusals' inputs (prefill on)
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
};

int serve_kolibri(const KolibriServe& a, server::Options options) {
  const cli::kolibri::DecodeArgs& da = a.args;
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(a.snapshot_dir, 0);
  cli::kolibri::check_args(da, pre);
  // Spec 20 decision 3: the served context is capped at the trained 262144 - refused here, before the device
  // (settle() refuses it too, after the cards are open).
  if (!da.max_len.is_auto && da.max_len.value > pre.trained_max_len)
    throw std::runtime_error("--max-len " + std::to_string(da.max_len.value) + " exceeds Kolibri-1's trained context " +
                             std::to_string(pre.trained_max_len) + " - a longer context is spec 20 decision 3");
  const std::vector<uint32_t> eos = eos_ids(a.snapshot_dir);
  server::Sampling defaults;
  {
    std::ifstream g(a.snapshot_dir + "generation_config.json");
    if (!g) throw std::runtime_error("cannot open '" + a.snapshot_dir + "generation_config.json'");
    nlohmann::json gj;
    g >> gj;
    try {
      defaults = server::generation_sampling(gj);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(error.what());
    }
  }
  const uint32_t devs = cli::kolibri::devices_of(da);
  if (devs == 2) cli::require_two_devices(l0::Context::gpu_count());
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(0u);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  const std::array<size_t, runtime::kPpDevices> dev{c0->memory_bytes(), c1 ? c1->memory_bytes() : 0};
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)\n", c0->name().c_str(), c0->eu_count());
  else
    std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--parallel pp: one context]\n", c0->name().c_str(),
                 c0->eu_count(), c1->name().c_str(), c1->eu_count());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), "
                             "and both hand-offs write it (docs/10-the-box.md)");
  const std::pair<model::KolPlacement, uint32_t> pl = cli::kolibri::settle(pre, da, dev);
  const uint32_t max_len = pl.second;
  loader::KolLoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load_kolibri1(ctx, a.snapshot_dir, max_len, pl.first, da.lm_head, 0);
  }();
  runtime::PipelineOptions opt;
  opt.handoff = da.pipe.handoff;
  runtime::kolibri::KolibriEngine engine(ctx, std::move(model), max_len, /*debug_tap=*/false, opt);
  engine.prepare_prefill();   // every card's prefill scratch and the binaries' check at load, not in a request
  std::fprintf(stderr, "%s\n", engine.memory_line().c_str());
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  if (tokenizer.vocab_used() != engine.vocab())
    std::fprintf(stderr, "note: tokenizer.json defines %u ids, Kolibri-1's head has %u rows; sampling masks from "
                         "%u, and a greedy argmax on a row without a token takes the masked row's argmax\n",
                 tokenizer.vocab_used(), engine.vocab(), tokenizer.vocab_used());
  TemplateAdapter chat_template(a.snapshot_dir);
  cli::kolibri::KolibriEngineAdapterT<runtime::kolibri::KolibriEngine> engine_adapter(engine, tokenizer.vocab_used());
  options.eos_ids = eos;
  options.sampling_defaults = defaults;
  // Pinned host memory of the one context both cards share: either device's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(*c0, a.prefix_auto, a.prefix_cache_gb, options);
  options.parallel_desc = "pp - the layers split over 2 cards (Kolibri-1 needs both)";
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  const std::string placement =
      devs == 2 ? std::string("--parallel pp (hand-off ") + runtime::pp_handoff_name(engine.handoff()) + ", split " +
                      std::to_string(engine.split()) + ": device 0 layers [0, " + std::to_string(engine.split()) +
                      "), device 1 [" + std::to_string(engine.split()) + ", " +
                      std::to_string(engine.model().desc.layers) + "))"
                : std::string("--parallel 1");
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::fprintf(stderr,
               ", Kolibri-1 (%s chat format), %s, prefill backend l0 (chunk %u), attention %s (B70_KOLIBRI_ATTN), "
               "mtp none, lm_head %s, kv cache bf16, sampling defaults %s T %.2f top-p %.2f top-k %u, prefix "
               "snapshots: the rings' last %u positions + the full layers' KV\n",
               options.chat_format.name(), placement.c_str(), runtime::kolibri::kPfC,
               runtime::kolibri::kol_attn_name(engine.attention()), loader::lm_head_form_name(da.lm_head),
               defaults.greedy ? "greedy" : "sampled", defaults.temperature, defaults.top_p, defaults.top_k,
               runtime::kolibri::state_positions(engine.model().desc));
  return listen_until_stopped(server, options);
}

// Spec 25e: Nemotron 3.5 Bolt 74B-A9B served - runtime::bolt::BoltEngine (two cards by default, spec 25 Decision 4: the whole
// model is 40.46 GB of weights; or one for a synthetic / small checkpoint) behind cli::bolt::BoltEngineAdapterT (the hybrid
// prefix snapshots: per Mamba block the SSM state and three conv columns, the attention blocks' KV; host sampling), Bolt's chat
// format (server::ChatFormat::bolt(): the template's kwargs, the ultra_v3 reasoning rules, Qwen3-Coder XML calls), the snapshot's
// chat_template.jinja (or --chat-template FILE), NVIDIA's sampling defaults and EOS [2, 11] unless the snapshot's
// generation_config.json carries sampling keys or --generation-config FILE says otherwise (cli::bolt::resolve_generation).
// b70-decode's flow (cli/bolt_decode.h run_decode): the refusals before the device (cli::bolt::check_args and check_serve_flags,
// the trained-context cap), both cards in one Level Zero context with peer access when --parallel pp, the placement and max_len
// planned before the load with the prefill scratch (every request prefills), each block loaded straight onto its device.
struct BoltServe {
  std::string snapshot_dir;
  cli::bolt::DecodeArgs args;   // the planner's and the refusals' inputs (prefill on)
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  cli::bolt::SpecArgs spec_args;   // spec 26c: --spec dflash / lookup (cli/bolt_spec.h)
};

int serve_bolt(const BoltServe& a, server::Options options) {
  namespace rb = runtime::bolt;
  const cli::bolt::DecodeArgs& da = a.args;
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(a.snapshot_dir, 0);
  cli::bolt::check_args(da, pre);
  cli::bolt::check_serve_max_len(da.max_len.is_auto, da.max_len.value, pre.max_pos);   // the trained 262144, before the device
  // The switches, read once (a bad value throws before the device).
  const rb::BoltAttn attn = rb::bolt_attn();
  const rb::BoltFuse fuse = rb::bolt_fuse();
  const rb::BoltState state = rb::bolt_state();
  // Spec 26c: --spec dflash / lookup - the refusals that need the placement, the drafter's plan terms (its config.json and
  // safetensors header), all before the device. The verify's routed experts are the union-aware form (Decision 10).
  const cli::bolt::SpecArgs& sa = a.spec_args;
  cli::bolt::check_spec(sa, cli::bolt::devices_of(da, pre), da.pipe.handoff);
  const rb::VerifyMoe vmoe = rb::VerifyMoe::Union;
  cli::bolt::DecodeArgs dplan = da;
  if (sa.on()) cli::bolt::plan_terms(sa, pre, da.max_len.is_auto ? pre.max_pos : da.max_len.value, dplan);
  // The generation defaults (Decision 12): the snapshot's generation_config.json when it carries sampling keys, else NVIDIA's
  // published values.
  cli::bolt::Generation gen;
  {
    const auto read = [](const std::string& path, nlohmann::json& out) {
      std::ifstream g(path);
      if (!g) return false;
      try {
        g >> out;
      } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("'" + path + "' is not JSON: " + error.what());
      }
      return true;
    };
    nlohmann::json snap;
    const std::string snap_path = a.snapshot_dir + "generation_config.json";
    const bool have_snap = read(snap_path, snap);
    try {
      gen = cli::bolt::resolve_generation(have_snap ? &snap : nullptr, nullptr, "", snap_path);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(error.what());
    }
  }
  TemplateAdapter chat_template(a.snapshot_dir);
  const std::string template_sha = chat_template.tmpl.source_sha256();
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  if (tokenizer.vocab_used() != pre.vocab_used)
    std::fprintf(stderr, "note: %stokenizer.json defines %u ids, Nemotron 3.5 Bolt's published tokenizer %u (the head has %u "
                         "rows); sampling masks from %u, and a greedy argmax on a row without a token takes the masked row's "
                         "argmax\n", a.snapshot_dir.c_str(), tokenizer.vocab_used(), pre.vocab_used, pre.vocab,
                 tokenizer.vocab_used());
  const uint32_t devs = cli::bolt::devices_of(da, pre);
  if (devs == 2) cli::require_two_devices(l0::Context::gpu_count());
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(0u);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  std::vector<size_t> dev = {c0->memory_bytes()};
  if (c1) dev.push_back(c1->memory_bytes());
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)\n", c0->name().c_str(), c0->eu_count());
  else
    std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--parallel pp: one context]\n", c0->name().c_str(), c0->eu_count(),
                 c1->name().c_str(), c1->eu_count());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), and both "
                             "hand-offs write it (docs/10-the-box.md)");
  const rb::BoltPlan pl = cli::bolt::settle(pre, dplan, dev);
  loader::BoltLoadOptions lo;
  lo.int8_head = da.int8_head;
  lo.max_len = pl.max_len;
  lo.layers = da.layers;
  loader::BoltLoaded model = [&] {
    StdoutToStderr redirect;
    return loader::load_bolt(ctx, a.snapshot_dir, lo, pl.place);
  }();
  runtime::PipelineOptions opt;
  opt.handoff = da.pipe.handoff;
  rb::BoltEngine engine(ctx, std::move(model), pl.max_len, /*debug_tap=*/false, opt, /*profile=*/false, attn, fuse, state);
  // Spec 26c: the verify lists (their buffers; the lists on first use) and the drafter, before the adapter: snapshots carry
  // its ring and anchor.
  cli::bolt::BoltDrafter drafter;
  if (sa.on()) engine.enable_spec(vmoe);
  if (sa.dflash()) {
    StdoutToStderr redirect;
    drafter = cli::bolt::attach_drafter(engine, sa, a.snapshot_dir);
  }
  engine.prepare_prefill();   // every card's prefill scratch and the binaries' check at load, not in a request
  std::fprintf(stderr, "%s\n", engine.memory_line().c_str());
  if (sa.dflash()) {   // the server's adaptive loop over the drafter's cost table (AdaptiveK, K to --spec-max)
    options.mtp_auto = true;
    options.mtp_adaptive.max_k = sa.max_k;
    options.mtp_adaptive.cost = cli::bolt::spec_cost(sa, da.int8_head, vmoe);
    std::fprintf(stderr, "spec dflash: LiLiCorr %s, drafter %s (+ the head exact), q %s, --spec-max %u, the verify's MoE %s, "
                 "drafter key %016llx, %zu B (%s) + the embedding replica %zu B on device %u - NOT Bolt's auto (spec 26 D5: box "
                 "row 65)\n", sa.draft_model.c_str(), loader::draft_format_name(sa.format), cli::bolt::bolt_draft_q_name(sa.q),
                 sa.max_k, rb::verify_moe_name(vmoe), (unsigned long long)engine.drafter_key(), drafter.drafter->bytes(),
                 drafter.drafter->weights().line.c_str(), drafter.replica->size(), engine.devices() - 1);
  } else if (sa.lookup()) {
    options.spec_lookup = true;
    options.lookup_min_match = server::Options{}.lookup_min_match;   // 3 (spec 19e)
    options.lookup_history = 0;
    options.lookup_adaptive.max_k = sa.max_k;
    options.lookup_adaptive.cost = cli::bolt::spec_cost(sa, da.int8_head, vmoe);
    std::fprintf(stderr, "spec lookup: --spec-max %u, the verify's MoE %s\n", sa.max_k, rb::verify_moe_name(vmoe));
  }
  cli::bolt::BoltEngineAdapterT<rb::BoltEngine> engine_adapter(
      engine, tokenizer.vocab_used(), sa.on() ? sa.max_k : 0, sa.lookup(),
      drafter.drafter ? cli::bolt::draft_hooks(engine, *drafter.drafter) : cli::bolt::BoltDraftHooks{}, sa.q);
  options.chat_format = server::ChatFormat::bolt();
  options.eos_ids = gen.eos;
  options.sampling_defaults = gen.sampling;
  options.stop_strings = gen.stops;
  // Pinned host memory of the one context both cards share: either device's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc = make_prefix_alloc(*c0, a.prefix_auto, a.prefix_cache_gb, options);
  options.parallel_desc = "pp - the layers split over 2 cards (Nemotron 3.5 Bolt needs both)";
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  const std::string placement =
      devs == 2 ? std::string("--parallel pp (hand-off ") + runtime::pp_handoff_name(engine.handoff()) + ", split " +
                      std::to_string(engine.split()) + ": device 0 blocks [0, " + std::to_string(engine.split()) +
                      "), device 1 [" + std::to_string(engine.split()) + ", " + std::to_string(engine.model().desc.layers) +
                      "))"
                : std::string("--parallel 1");
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(), options.host.c_str(),
               options.port, pl.max_len);
  print_eos(gen.eos);
  const server::Sampling& s = gen.sampling;
  std::fprintf(stderr,
               ", Nemotron 3.5 Bolt (%s chat format, template sha256 %.8s%s), %s, prefill backend l0 (chunk %u), attention %s "
               "(B70_BOLT_ATTN), fuse %s (B70_BOLT_FUSE), SSM state %s (B70_BOLT_SSM), mtp none, lm_head %s, kv cache bf16, "
               "sampling defaults %s: %s T %.2f top-p %.2f top-k %s, prefix snapshots at aligned %u-position block ends: %u "
               "SSM states + conv columns %.1f MB, attention KV %zu B a position\n",
               options.chat_format.name(), template_sha.c_str(),
               template_sha == cli::bolt::kTemplateSha256 ? " = Bolt's" : " (NOT Bolt's 6521afc7: another snapshot's)",
               placement.c_str(), rb::kPfC, rb::bolt_attn_name(engine.attention()), rb::bolt_fuse_name(engine.fuse()).c_str(),
               rb::bolt_state_name(engine.state()),
               loader::lm_head_form_name(da.int8_head ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint),
               gen.source.c_str(), s.greedy ? "greedy" : "sampled", s.temperature, s.top_p,
               s.top_k == 0 ? "none" : std::to_string(s.top_k).c_str(), rb::BoltEngine::kBlock,
               engine.model().desc.mamba_layers(), double(engine.state_bytes()) / 1e6, engine.kv_bytes(1));
  return listen_until_stopped(server, options);
}

// Spec 21e: Qwen3.8-Flash-Next served (--layers N until spec 22) - runtime::qwen4exp::Qwen4ExpEngine (one card or two:
// spec 16b's pieces as 21c / 21d built them) behind spec 16d's cli::pp::PipelineEngineAdapterT (EngineAdapter's logic
// over the engine's host calls: greedy and sampled steps, spec 7's prefix cache over the engine's snapshots, spec 8's
// MTP - draft / verify / commit - and spec 19e's lookup through the same verify lists), the Qwen family's request path
// (cli::qwen4exp::chat_options: ChatFormat Qwen, EOS [248046, 248044], decision 11's greedy-unless-asked). b70-decode's
// flow (cli/qwen4exp_decode.h run_decode): the refusals before the device (cli::qwen4exp::check_serve), the served
// tokenizer.json held to the original's (check_tokenizer), both cards in one context with peer access, the layers,
// placement and max_len planned before the load with the prefill scratch and (--spec mtp / --spec) the head, each layer
// loaded straight onto its device.
int serve_qwen4exp(const cli::qwen4exp::ServeArgs& a, server::Options options) {
  namespace cq = cli::qwen4exp;
  cq::check_serve(a);
  const std::vector<uint32_t> eos = eos_ids(a.snapshot_dir);
  const std::string tok_path = a.snapshot_dir + "tokenizer.json";
  cq::check_tokenizer(tok_path);   // a file read, before the device: the snapshot's must be the original's
  const cq::DecodeArgs& da = a.args;
  const model::Qwen4ExpDesc full = loader::qwen4exp_checkpoint_desc(a.snapshot_dir, 0);
  if (!da.max_len.is_auto && da.max_len.value > full.trained_max_len)
    throw std::runtime_error("--max-len " + std::to_string(da.max_len.value) +
                             " exceeds Qwen3.8-Flash-Next's trained context " + std::to_string(full.trained_max_len));
  const uint32_t devs = cq::devices_of(da);
  if (devs == 2) cli::require_two_devices(l0::Context::gpu_count());
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(0u);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  const std::array<size_t, runtime::kPpDevices> dev{c0->memory_bytes(), c1 ? c1->memory_bytes() : c0->memory_bytes()};
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)\n", c0->name().c_str(), c0->eu_count());
  else
    std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--parallel pp: one context]\n", c0->name().c_str(),
                 c0->eu_count(), c1->name().c_str(), c1->eu_count());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), "
                             "and both hand-offs write it (docs/10-the-box.md)");
  const cq::Settled s = cq::settle(full, da, dev);
  loader::Q4LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load_qwen4exp(ctx, a.snapshot_dir, s.max_len, s.placement, da.lm_head, s.desc.layers, a.loads_head());
  }();
  std::fprintf(stderr, "layers: %u of the checkpoint's %u (--layers: development mode until spec 22)\n",
               model.desc.layers, model.checkpoint_layers);
  runtime::PipelineOptions opt;
  opt.handoff = da.pipe.handoff;
  runtime::qwen4exp::Qwen4ExpEngine engine(ctx, std::move(model), s.max_len, /*debug_tap=*/false, opt);
  engine.prepare_prefill();   // every card's prefill scratch and the binaries' check at load, not in a request
  std::fprintf(stderr, "%s\n", engine.memory_line().c_str());
  TokAdapter tokenizer(tok_path);
  if (tokenizer.vocab_used() != engine.model().desc.vocab_used)
    std::fprintf(stderr, "note: %s defines %u ids, the greedy argmax masks from %u; sampling masks from %u\n",
                 tok_path.c_str(), tokenizer.vocab_used(), engine.model().desc.vocab_used, tokenizer.vocab_used());
  TemplateAdapter chat_template(a.snapshot_dir);
  cli::pp::PipelineEngineAdapterT<runtime::qwen4exp::Qwen4ExpEngine> engine_adapter(
      engine, tokenizer.vocab_used(), engine.vocab(), a.mtp_k, a.spec_lookup);
  cq::chat_options(options, eos);
  // Pinned host memory of the one context both cards share: either device's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc = make_prefix_alloc(*c0, a.prefix_auto, a.prefix_cache_gb, options);
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  const std::string placement =
      devs == 2 ? std::string("--parallel pp (hand-off ") + runtime::pp_handoff_name(engine.handoff()) + ", split " +
                      std::to_string(engine.split()) + ")"
                : std::string("--parallel 1");
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, s.max_len);
  print_eos(eos);
  std::fprintf(stderr,
               ", Qwen3.8-Flash-Next --layers %u (%s chat format), %s, prefill backend l0 (chunk %u), attention %s "
               "(B70_Q4_ATTN), mtp %u%s, lm_head %s, kv cache bf16, sampling defaults: greedy unless the request "
               "samples (decision 11 open), prefix snapshots: state %zu B (GDN, PLE, indexer tails%s), blocks %zu B a "
               "position\n",
               engine.model().desc.layers, options.chat_format.name(), placement.c_str(), runtime::qwen4exp::kPfC,
               runtime::qwen4exp::q4_attn_name(engine.attention()), a.mtp_k,
               engine.mtp() ? (std::string(" (the head, fixed K: ") + runtime::qwen4exp::mtp_norm_name(engine.mtp_norm()) +
                               " norm, reuse selection" +
                               (a.spec_lookup ? ", --spec lookup" : "") + ")").c_str()
                            : "",
               loader::lm_head_form_name(da.lm_head), engine.state_bytes(), engine.mtp() ? ", the head's" : "",
               engine.kv_bytes(4) / 4);
  return listen_until_stopped(server, options);
}

// The speculative proposer's startup lines (spec 19e's lookup, spec 8 §10's auto policy) -
// one card's and --parallel pp's.
void print_proposer(const server::Options& options, bool spec_lookup, uint32_t spec_min_match,
                    uint32_t spec_max, uint32_t spec_history, bool mtp_auto, bool moe_default_costs,
                    const std::string& model_name) {
  if (spec_lookup) {
    const server::MtpCost& c = options.lookup_adaptive.cost;
    std::fprintf(stderr, "spec lookup: min match %u, K <= %u, history %u, cost verify M=1..%zu",
                 spec_min_match, spec_max, spec_history, c.verify.size());
    for (double v : c.verify) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, ", draft k=1..%zu", c.draft.size());
    for (double v : c.draft) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, " (plain steps)\n");
  }
  if (mtp_auto) {
    const server::MtpCost& c = options.mtp_adaptive.cost;
    std::fprintf(stderr, "mtp auto: cost verify M=1..%zu", c.verify.size());
    for (double v : c.verify) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, ", draft k=1..%zu", c.draft.size());
    for (double v : c.draft) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, " (plain steps)\n");
    // Spec 15e Review Focus 2: the defaults are Qwen3.8's measured table. A MoE model's
    // verify at M rows touches up to top_k x M distinct experts, so its costs are its own -
    // unmeasured until the box records them (then they become the engine's table).
    if (moe_default_costs)
      std::fprintf(stderr, "mtp auto: %s's own costs are not measured yet; the table above is "
                           "Qwen3.8's\n",
                   model_name.c_str());
  }
}

// Spec 16d: b70-serve --parallel pp - the Qwen-family model over two B70s (runtime::PipelineEngine
// behind cli::pp::PipelineEngineAdapterT). b70-decode --parallel pp's flow (cli/pipeline_decode.h):
// both cards in one Level Zero context and peer access, the whole checkpoint (with the MTP
// head when a proposer needs it) loaded onto device 0, the split and max_len planned over the
// loaded bytes with the server's terms (the prefill scratch per card; the MTP head, its
// embedding replica and buffers on device 1; the prefix cache's block shadows when the cache
// may be on - cli/pipeline_settle.h), then placed, then the two-card prefill prepared before
// the first request. --parallel's refusals ran before this (cli/pipeline_serve.h).
struct PipeServe {
  std::string snapshot_dir;
  cli::MaxLenArg max_len_arg;
  uint32_t trained = 0;
  size_t mem_reserve = 0;
  runtime::KvCache kv_cache = runtime::KvCache::Bf16;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  loader::DraftVocabSpec dv_spec;
  uint32_t mtp_k = 0;
  bool mtp_auto = false, spec_lookup = false;
  uint32_t spec_max = 3;
  bool have_pp_backend = false;
  runtime::PrefillBackend pp_backend = runtime::PrefillBackend::L0;
  runtime::PrefillPath pp_path;
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  cli::PipelineArgs pipe;
  std::vector<uint32_t> eos;
  cli::dflash::Args dflash;   // spec 19d: --spec dflash --parallel pp (the drafter on device 1)
};

int serve_pipeline(l0::Context& d0, const PipeServe& a, server::Options options) {
  cli::require_two_devices(l0::Context::gpu_count());
  l0::Context d1(d0, 1u);
  std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--parallel pp: one context]\n",
               d0.name().c_str(), d0.eu_count(), d1.name().c_str(), d1.eu_count());
  if (!d0.can_access_peer(d1))
    throw std::runtime_error(
        "--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer 0 -> 1 is false), "
        "and both hand-offs write it. Peer access needs the P2P-capable kernel and both cards "
        "under one root complex (docs/10-the-box.md)");
  loader::LoadedModel full = [&] {
    StdoutToStderr redirect;
    return loader::load(d0, a.snapshot_dir, cli::load_len(a.max_len_arg, a.trained),
                        /*mtp=*/a.mtp_k > 0 || a.spec_lookup, a.lm_head, a.dv_spec);
  }();
  model::require_prefill(*full.desc);   // spec 15c, as on one card
  // Spec 19d: the drafter's planner term, read off its checkpoint against the whole target (it loads onto device 1
  // after the placement, beside lm_head and the embedding replica).
  runtime::DrafterPlan drafter_plan;
  if (a.dflash.on) drafter_plan = cli::dflash::plan_only(a.dflash, full);
  const bool draft_vocab_on = full.draft_vocab != nullptr;
  const model::ModelDesc& d = *full.desc;
  const runtime::PpWeights w = runtime::pp_weights(full);
  const std::array<size_t, runtime::kPpDevices> dev{d0.memory_bytes(), d1.memory_bytes()};
  // The block hook's shadows are planned whenever the cache may be on: auto decides from the
  // host's RAM only after the load (make_prefix_alloc), and an unplanned shadow would be an
  // allocation in the first request.
  const bool hook = a.prefix_auto || a.prefix_cache_gb > 0;
  runtime::PpExtras extras = cli::pp_extras(full, hook);
  extras.drafter = drafter_plan;   // spec 19d
  // Plan 5d: the chunk cap (the default 2048; cli/max_len.h), planned on both devices.
  runtime::PrefillPath pf = a.pp_path;
  pf.chunk_cap = cli::planned_chunk_cap(0, d);
  const cli::PpSettled settled = cli::pp_settle(d, w, dev, a.max_len_arg, full.trained_max_len,
                                                a.mem_reserve, a.kv_cache, pf, extras, a.pipe);
  std::vector<loader::LoadedModel> stages =
      runtime::place_stages(d0, d1, std::move(full), settled.split, /*drafter=*/a.dflash.on);
  if (stages[0].max_len != settled.max_len) {   // auto loaded at a small length: re-table both
    loader::set_max_len(d0, stages[0], settled.max_len);
    loader::set_max_len(d1, stages[1], settled.max_len);
  }
  // Spec 19d: the drafter on device 1 beside the stage model (its lm_head, the embedding replica).
  std::unique_ptr<runtime::dflash::DFlashDrafter> drafter;
  if (a.dflash.on) {
    cli::dflash::Loaded l = cli::dflash::load(d1, a.dflash, stages[1]);
    drafter = std::make_unique<runtime::dflash::DFlashDrafter>(d1, std::move(l.w), l.head_rows, settled.max_len,
                                                               a.dflash.max_k);
  }
  runtime::PipelineOptions opt;
  opt.handoff = a.pipe.handoff;
  runtime::PipelineEngine engine(d0, d1, std::move(stages), settled.max_len, opt, a.kv_cache, drafter.get());
  if (drafter)
    drafter->attach_target(d1, engine.model(1), engine.buffers(1).control, settled.max_len, draft_vocab_on);
  if (a.have_pp_backend) engine.set_prefill_backend(a.pp_backend);
  engine.set_prefill_chunk_cap(pf.chunk_cap);   // plan 5d: the cap the plan counted
  // The two-card prefill set up at load, not in a request (each card's scratch, context and
  // l0-int8 column scales; with the head, its prefill rows on device 1).
  engine.prepare_prefill();
  for (uint32_t i = 0; i < runtime::kPpDevices; ++i)
    std::fprintf(stderr, "%s\n", engine.memory_line(i).c_str());
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  const model::ModelDesc& served = *engine.model(0).desc;
  if (tokenizer.vocab_used() != served.vocab_used)
    std::fprintf(stderr,
                 "note: tokenizer.json defines %u ids, %s's greedy argmax masks from %u; sampling "
                 "masks from %u\n",
                 tokenizer.vocab_used(), served.name.c_str(), served.vocab_used, tokenizer.vocab_used());
  TemplateAdapter chat_template(a.snapshot_dir);
  cli::pp::PipelineEngineAdapterT<runtime::PipelineEngine> engine_adapter(
      engine, tokenizer.vocab_used(), model::Qwen35::kVocab, a.dflash.on ? a.dflash.max_k : a.mtp_k, a.spec_lookup,
      a.dflash.on);
  options.eos_ids = a.eos;
  // Pinned host memory of the one context both cards share: either device's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(d0, a.prefix_auto, a.prefix_cache_gb, options);
  options.parallel_desc = "pp - the layers split over 2 cards";
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, settled.max_len);
  print_eos(a.eos);
  std::fprintf(stderr,
               ", --parallel pp (hand-off %s, split %u: device 0 layers [0, %u), device 1 [%u, %u)), prefill "
               "backend %s, mtp %s%u%s, lm_head %s%s%s, kv cache %s\n",
               runtime::pp_handoff_name(engine.handoff()), engine.split(), engine.split(),
               engine.stage(1).first, engine.stage(1).last,
               runtime::prefill_backend_name(engine.prefill_backend()), a.mtp_auto ? "auto, max " : "",
               a.mtp_k,
               engine.mtp() ? " (the head on device 1)" : engine.has_drafter() ? " (dflash: the drafter on device 1)" : "",
               loader::lm_head_form_name(a.lm_head),
               engine.draft_vocab() ? ", draft vocab " : "",
               engine.draft_vocab() ? loader::draft_vocab_name(engine.draft_vocab()).c_str() : "",
               runtime::kv_cache_name(engine.kv_cache()));
  print_proposer(options, a.spec_lookup, options.lookup_min_match, a.spec_max, options.lookup_history, a.mtp_auto,
                 served.is_moe(), served.name);
  if (drafter) {   // spec 19d: the drafter's policy line, as on one card
    const server::MtpCost& c = options.mtp_adaptive.cost;
    std::fprintf(stderr, "spec dflash (--parallel pp, device 1): %s drafter, K <= %u, cost verify M=1..%zu",
                 loader::draft_format_name(a.dflash.format), a.dflash.max_k, c.verify.size());
    for (double v : c.verify) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, ", draft k=1..%zu", c.draft.size());
    for (double v : c.draft) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, " (plain steps; derived until the box measures them)\n");
  }
  return listen_until_stopped(server, options);
}

// Plan 13b (spec 13 §10 / §11): b70-serve --parallel replica - Qwen3.8 on both cards, one runtime::Engine each (each
// card's own plan: its --max-len auto, its prefill chunk cap, MTP / lookup per --spec), behind cli::ReplicaEngine and
// server::Scheduler: two requests at once, each bitwise what it is alone on its card. Both cards are views of ONE
// Level Zero context (spec 16 decision 1) so the prefix cache's pinned host store is device-visible to both - a
// snapshot saved by card 0 restores on card 1; nothing else crosses (no peer access is used). The refusals
// (another model, one visible card, the pp override flags, DFlash) ran in run(), before the device.
struct ReplicaServe {
  std::string snapshot_dir;
  cli::MaxLenArg max_len_arg;
  uint32_t trained = 0;
  size_t mem_reserve = 0;
  runtime::KvCache kv_cache = runtime::KvCache::Bf16;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  loader::DraftVocabSpec dv_spec;
  uint32_t mtp_k = 0;
  bool mtp_auto = false, spec_lookup = false;
  uint32_t spec_max = 0;
  bool have_pp_backend = false;
  runtime::PrefillBackend pp_backend{};
  runtime::PrefillPath pp_path;
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  std::vector<uint32_t> eos;
};

int serve_replica(const ReplicaServe& a, server::Options options) {
  cli::require_two_devices_for_replica(l0::Context::gpu_count());
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--parallel replica: one engine per card, one "
                       "context; no peer access used]\n",
               d0.name().c_str(), d0.eu_count(), d1.name().c_str(), d1.eu_count());
  std::array<l0::Context*, 2> ctx{&d0, &d1};
  std::vector<std::unique_ptr<runtime::Engine>> engines;
  std::vector<uint32_t> lens, caps;
  for (uint32_t i = 0; i < 2; ++i) {
    l0::Context& c = *ctx[i];
    loader::LoadedModel model = [&] {
      StdoutToStderr redirect;
      return loader::load(c, a.snapshot_dir, cli::load_len(a.max_len_arg, a.trained),
                          /*mtp=*/a.mtp_k > 0 || a.spec_lookup, a.lm_head, a.dv_spec);
    }();
    model::require_prefill(*model.desc);
    if (!cli::replica_architecture(model.desc->architecture))   // run() read config.json; the loader is the truth
      throw std::runtime_error("--parallel replica serves Qwen3.8 only (spec 13b); this checkpoint loads as " +
                               model.desc->name);
    runtime::PrefillPath pf = a.pp_path;
    pf.chunk_cap = cli::planned_chunk_cap(0, *model.desc);
    std::fprintf(stderr, "card %u: ", i);
    const uint32_t max_len = cli::settle(c, model, a.max_len_arg, a.mem_reserve, pf, a.kv_cache);
    const uint32_t chunk_cap = cli::settle_chunk_cap(c, model, max_len, a.mem_reserve, pf, a.kv_cache, {}, 0);
    auto engine = std::make_unique<runtime::Engine>(c, std::move(model), max_len, /*debug_resid=*/false, a.kv_cache);
    if (a.have_pp_backend) engine->set_prefill_backend(a.pp_backend);
    engine->set_prefill_chunk_cap(chunk_cap);   // plan 5d
    engine->prepare_prefill();
    std::fprintf(stderr, "card %u: max_len %u, prefill chunk %u, %s\n", i, max_len, chunk_cap,
                 engine->memory_line().c_str());
    lens.push_back(max_len);
    caps.push_back(chunk_cap);
    engines.push_back(std::move(engine));
  }
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  TemplateAdapter chat_template(a.snapshot_dir);
  std::vector<std::unique_ptr<EngineAdapter>> adapters;
  for (auto& e : engines)
    adapters.push_back(std::make_unique<EngineAdapter>(*e, tokenizer.vocab_used(), a.mtp_k, a.spec_lookup));
  options.eos_ids = a.eos;
  // The prefix store: one pinned allocation in the shared context, sized after both loads.
  const std::unique_ptr<PinnedAlloc> prefix_alloc = make_prefix_alloc(d0, a.prefix_auto, a.prefix_cache_gb, options);
  // The prompt's cut is the serial server's: hooked (every chunk ends at a stride end) when the cache is on.
  cli::ReplicaEngine replica({{adapters[0].get(), caps[0]}, {adapters[1].get(), caps[1]}},
                             /*hooked=*/options.prefix_cache_bytes != 0, /*wide_stride=*/true);
  options.parallel_desc = "replica - one copy of the model per card, two requests at once";
  server::Server server({tokenizer, chat_template, *adapters[0]}, options, &replica);
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, --parallel replica: 2 requests at once, max_len %u / %u, eos ",
               options.served_model.c_str(), options.host.c_str(), options.port, lens[0], lens[1]);
  print_eos(a.eos);
  std::fprintf(stderr, ", prefill backend %s, mtp %s%u, lm_head %s, kv cache %s\n",
               runtime::prefill_backend_name(engines[0]->prefill_backend()), a.mtp_auto ? "auto, max " : "", a.mtp_k,
               loader::lm_head_form_name(a.lm_head), runtime::kv_cache_name(engines[0]->kv_cache()));
  print_proposer(options, a.spec_lookup, options.lookup_min_match, a.spec_max, options.lookup_history, a.mtp_auto,
                 false, engines[0]->model().desc->name);
  return listen_until_stopped(server, options);
}

int run(int argc, char** argv) {
  std::string path;
  server::Options options;
  cli::MaxLenArg max_len_arg;   // spec 6 §10: auto unless --max-len N
  runtime::KvCache kv_cache = runtime::default_kv_cache();   // spec 12b: --kv-cache
  // The zero-config cleanup (operator 2026-11-06): the engine decides these - the device memory the plan
  // leaves free (runtime::kDefaultReserveGb), the int8 lm_head (spec 9, every served model), the request
  // queue (server::Options::queue_depth) and the served name (cli::served_name, from the model).
  const size_t mem_reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  std::string pp_backend_arg;
  bool have_pp_backend = false;
  // Spec 7: system RAM, not VRAM; auto by default (cli/prefix_cache_size.h).
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  // --spec auto|off|mtp|lookup|dflash|dflash-lookup (cli/spec_args.h); --spec-max K its one cap.
  std::string spec_arg = "auto";
  uint32_t spec_max = 0;
  bool have_spec_max = false;
  cli::dflash::Args dflash;   // spec 19c: --spec dflash's drafter (--draft-model)
  // Spec 8 §11: off until the box rows decide (§11 "Gates").
  uint32_t draft_vocab = 0;
  std::string draft_vocab_ids;
  cli::PipelineArgs pipe;   // --parallel, --pipeline-split, --pipeline-handoff
  // Spec 21e: Qwen3.8-Flash-Next only - its development mode (--layers N|auto until spec 22).
  uint32_t q4_layers = 0;
  bool q4_layers_auto = false, have_q4_layers = false;

  auto value = [&](int& i, const char* flag) -> std::string {
    if (++i >= argc) throw std::runtime_error(std::string(flag) + " needs a value");
    return argv[i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else if (arg == "--host") {
      options.host = value(i, "--host");
    } else if (arg == "--port") {
      const uint32_t port = parse_u32("--port", value(i, "--port"));
      if (port > 65535) throw std::runtime_error("--port is out of range");
      options.port = static_cast<int>(port);
    } else if (arg == "--max-len") {
      max_len_arg = cli::parse_max_len(value(i, "--max-len"));
    } else if (arg == "--log-requests") {
      options.log_requests_dir = value(i, "--log-requests");
    } else if (arg == "--prefix-cache-gb") {
      const std::string v = value(i, "--prefix-cache-gb");
      prefix_auto = v == "auto";
      if (!prefix_auto) prefix_cache_gb = parse_u32("--prefix-cache-gb", v);
    } else if (arg == "--prefix-split-last") {
      options.prefix_split_last = true;
    } else if (arg == "--spec") {
      spec_arg = cli::parse_spec(value(i, "--spec"));
    } else if (arg == "--spec-max") {
      spec_max = parse_u32("--spec-max", value(i, "--spec-max"));
      have_spec_max = true;   // its range is the proposer's: checked below
    } else if (arg == "--draft-model") {   // spec 19c
      dflash.model = value(i, "--draft-model");
      dflash.have_model = true;
    } else if (arg == "--prefill-backend") {
      pp_backend_arg = value(i, "--prefill-backend");
      have_pp_backend = true;
    } else if (arg == "--draft-vocab") {
      const std::string v = value(i, "--draft-vocab");
      if (!loader::parse_draft_vocab(v, draft_vocab))
        throw std::runtime_error("--draft-vocab expects off, 32k, 64k or 128k, got '" + v + "'");
    } else if (arg == "--draft-vocab-ids") {
      draft_vocab_ids = value(i, "--draft-vocab-ids");
    } else if (arg == "--kv-cache") {   // spec 12b
      kv_cache = cli::parse_kv_cache_arg(value(i, "--kv-cache"));
    } else if (arg == "--parallel") {
      cli::parse_parallel(value(i, "--parallel"), pipe);
    } else if (arg == "--pipeline-split") {
      cli::parse_pipeline_split(value(i, "--pipeline-split"), pipe);
    } else if (arg == "--pipeline-handoff") {
      cli::parse_pipeline_handoff(value(i, "--pipeline-handoff"), pipe);
    } else if (arg == "--layers") {   // spec 21e: Qwen3.8-Flash-Next only (also `auto`)
      const std::string v = value(i, "--layers");
      have_q4_layers = true;
      if (v == "auto") {
        q4_layers_auto = true;
      } else {
        q4_layers = parse_u32("--layers", v);
        if (q4_layers == 0) throw std::runtime_error("--layers 0 would load no layer");
      }
    } else if (!arg.empty() && arg[0] == '-') {
      usage();
      throw std::runtime_error("unknown option '" + arg + "'");
    } else if (path.empty()) {
      path = arg;
    } else {
      usage();
      throw std::runtime_error("unexpected extra argument '" + arg + "'");
    }
  }
  if (path.empty()) {
    usage();
    throw std::runtime_error("a snapshot directory or HF repo id is required");
  }
  if (options.host.empty()) throw std::runtime_error("--host must not be empty");
  options.served_model = cli::served_name(path);
  cli::resolve_overrides(pipe);   // --pipeline-split / --pipeline-handoff under --parallel auto: pp
  if (pipe.mode == cli::Parallel::One && pipe.have_split)   // before the model is even resolved
    throw std::runtime_error("--pipeline-split belongs to --parallel pp (the layer device 1 starts at)");
  if (pipe.mode == cli::Parallel::One && pipe.have_handoff)
    throw std::runtime_error("--pipeline-handoff belongs to --parallel pp (how the hidden state crosses)");
  if (pipe.replica()) {   // plan 13b: Qwen3.8 only, by name, before any device or weight file
    cli::check_replica_flags(pipe, /*dflash=*/false);
    std::string architecture, name;
    try {
      const std::string snap = loader::resolve_snapshot(path);
      if (std::ifstream cf(snap + "config.json"); cf) {
        std::stringstream cs;
        cs << cf.rdbuf();
        const nlohmann::json cj = nlohmann::json::parse(cs.str(), nullptr, /*allow_exceptions=*/false);
        if (cj.is_object() && cj.contains("architectures") && cj.at("architectures").is_array() &&
            !cj.at("architectures").empty() && cj.at("architectures").at(0).is_string())
          architecture = cj.at("architectures").at(0).get<std::string>();
        if (cj.is_object()) name = cj.value("model_type", std::string());
      }
    } catch (const std::exception&) {
    }
    if (!architecture.empty()) {
      try {
        name = model::desc_for_architecture(architecture).name;
      } catch (const std::exception&) {
        name = (name.empty() ? std::string("this model") : name) + " (" + architecture + ")";
      }
    }
    cli::check_replica_model(architecture, name);
  }
  const bool pp_given = pipe.given();
  if (have_q4_layers && !cli::qwen4exp::is_qwen4exp(path))
    throw std::runtime_error("--layers belongs to Qwen3.8-Flash-Next (model_type qwen4_exp, spec 21e); this checkpoint "
                             "is not one");
  runtime::PrefillBackend pp_backend{};
  if (have_pp_backend && !runtime::parse_prefill_backend(pp_backend_arg, pp_backend))
    throw std::runtime_error("--prefill-backend expects l0 or l0-int8, got '" + pp_backend_arg + "'");
  if (!draft_vocab_ids.empty() && draft_vocab == 0)
    throw std::runtime_error("--draft-vocab-ids needs --draft-vocab 32k, 64k or 128k");
  if (!options.log_requests_dir.empty()) std::filesystem::create_directories(options.log_requests_dir);
  // The hardware gate (cli/hardware_gate.h): one or two Intel Arc Pro B70s, peer access between two - before
  // any model file is read. Its card memory is what the fit checks before a device opens plan against.
  const cli::Hardware hw = cli::hardware_gate();
  if (pipe.replica()) cli::require_two_devices_for_replica(hw.cards);   // plan 13b: one engine per card

  // Spec 25e: Nemotron 3.5 Bolt (model_type nemotron_h_omni over nemotron_h_puzzle) runs runtime::bolt::BoltEngine behind
  // cli::bolt::BoltEngineAdapterT (two cards by default) - dispatched here, before every other family's rules (its own:
  // cli::bolt::check_args and check_serve_flags replace them), as b70-decode dispatches it. A path that does not resolve is not
  // Bolt: the flow below reports it.
  if (cli::bolt::is_bolt(path)) {
    cli::bolt::ServeFlags flags;
    flags.draft_vocab = draft_vocab != 0 || !draft_vocab_ids.empty();
    // Spec 26c: --spec / --spec-max / --draft-model through cli::resolve_spec (SpecFamily::Bolt), refused by name before the
    // device; then the compact-head flags.
    BoltServe bs;
    bs.spec_args = cli::bolt::spec_args(spec_arg, have_spec_max, spec_max, dflash.have_model, dflash.model);
    cli::bolt::check_serve_flags(flags);
    options.chat_format = server::ChatFormat::bolt();
    bs.snapshot_dir = loader::resolve_snapshot(path);
    cli::bolt::DecodeArgs& ba = bs.args;
    ba.path = bs.snapshot_dir;
    ba.max_len = max_len_arg;   // auto by default: the largest that fits, at most 262144 (Decision 14)
    ba.reserve = mem_reserve;
    ba.int8_head = true;        // int8 (Decision 18), as every served model
    ba.pipe = pipe;
    ba.pp_given = pp_given;
    ba.prefill = true;          // every request prefills: the plan carries the prefill scratch
    ba.pp_backend_given = have_pp_backend;
    ba.pp_backend = pp_backend;
    ba.mtp = false;             // refused above
    ba.spec = false;            // the plan's proposer terms come from bs.spec_args (plan_terms)
    ba.kv8 = kv_cache == runtime::KvCache::Int8;
    ba.layers = 0;              // --layers is Flash-Next's (refused above)
    bs.prefix_auto = prefix_auto;
    bs.prefix_cache_gb = prefix_cache_gb;
    return serve_bolt(bs, options);
  }

  // The model family decides what `--spec auto` is and which proposers exist (cli/spec_args.h).
  const bool is_q4 = cli::qwen4exp::is_qwen4exp(path);
  const bool is_kol = !is_q4 && cli::kolibri::is_kolibri(path);
  // A path that does not resolve is the Qwen family here: every flag refusal below keeps its order, and the
  // snapshot's own error comes after them.
  std::string model_type;
  if (!is_q4 && !is_kol) {
    std::string snap;
    try {
      snap = loader::resolve_snapshot(path);
    } catch (const std::exception&) {
    }
    if (std::ifstream cf(snap + "config.json"); !snap.empty() && cf) {
      std::stringstream cs;
      cs << cf.rdbuf();
      const nlohmann::json cj = nlohmann::json::parse(cs.str(), nullptr, /*allow_exceptions=*/false);
      if (cj.is_object()) model_type = cj.value("model_type", std::string());
    }
  }
  const bool is_k2 = model_type == "k2_horizon";
  const cli::SpecFamily family = is_q4 ? cli::SpecFamily::Qwen4Exp
                                 : is_kol ? cli::SpecFamily::None
                                 : is_k2 ? cli::SpecFamily::None
                                         : cli::SpecFamily::Qwen;
  const std::string model_name = is_q4 ? "Qwen3.8-Flash-Next" : is_kol ? "Kolibri-1" : is_k2 ? "K2-Horizon" : "this model";
  const cli::Spec spec = cli::resolve_spec(spec_arg, family, model_name, have_spec_max, spec_max, dflash.have_model);
  const bool mtp_on = spec.kind == cli::SpecKind::Mtp;
  const bool spec_lookup = spec.kind == cli::SpecKind::Lookup;
  const bool dflash_lookup = spec.kind == cli::SpecKind::DFlashLookup;
  dflash.on = spec.kind == cli::SpecKind::DFlash || dflash_lookup;   // spec 19c / 19f
  dflash.max_k = dflash.on ? spec.max_k : dflash.max_k;
  options.spec_dflash_lookup = dflash_lookup;
  uint32_t mtp_k = mtp_on ? spec.max_k : 0;   // the head's drafts per iteration at most
  cli::dflash::check_flags(dflash);
  // Spec 19c: bf16 KV, attention v2; spec 19d: pipeline parallel on the copy hand-off.
  cli::dflash::check_engine(dflash, pipe.on(), kv_cache, /*pipeline_served=*/true,
                            pipe.handoff == runtime::PpHandoff::Peer);
  cli::check_replica_flags(pipe, dflash.on);   // plan 13b: --spec dflash / dflash-lookup are not built for replica
  if (dflash_lookup && pipe.on())   // spec 19f: EngineAdapter's step_drafts_or_many; the pipeline adapter has none
    throw std::runtime_error("--spec dflash-lookup is built for one card (spec 19f); --parallel pp serves --spec dflash");
  if (spec_lookup) {
    options.spec_lookup = true;
    options.lookup_min_match = server::Options{}.lookup_min_match;   // 3 (spec 19e)
    options.lookup_history = 0;
    options.lookup_adaptive.max_k = spec.max_k;
    options.lookup_adaptive.cost = server::MtpCost::int8_head().with_free_drafts();
  }
  // Spec 8 §10: MTP chooses K per iteration from each request's own acceptance, over the int8 head's measured
  // cost table (the Qwen family; Flash-Next drafts a fixed K - its adaptive policy waits for spec 22's numbers).
  if (mtp_on && spec.adaptive) {
    options.mtp_auto = true;
    options.mtp_adaptive.max_k = spec.max_k;
    options.mtp_adaptive.cost = server::MtpCost::int8_head();
    // Spec 8 §11: a draft over V' reads |V'| / 248320 of the head, so the drafts' head share scales with it
    // (derived; MtpCost::kInt8DraftHeadShare).
    if (draft_vocab != 0)
      options.mtp_adaptive.cost = options.mtp_adaptive.cost.with_draft_vocab(
          server::MtpCost::kInt8DraftHeadShare, double(draft_vocab) / model::Qwen35::kVocab);
  }
  // Spec 8 §11: the draft vocabulary narrows the MTP head's or the drafter's drafts.
  if ((draft_vocab != 0 || !draft_vocab_ids.empty()) && !mtp_on && !dflash.on)
    throw std::runtime_error("--draft-vocab and --draft-vocab-ids need --spec mtp (the default where the model has "
                             "an MTP head) or --spec dflash");
  // Spec 19c: the drafter's AdaptiveK policy (the server's adaptive loop, over the drafter's cost table).
  if (dflash.on) {
    options.mtp_auto = true;
    options.mtp_adaptive.max_k = dflash.max_k;
    options.mtp_adaptive.cost = cli::dflash::cost(dflash, lm_head, draft_vocab);
  }

  // Spec 21e: Qwen3.8-Flash-Next (model_type qwen4_exp) runs runtime::qwen4exp::Qwen4ExpEngine behind spec 16d's
  // PipelineEngineAdapterT (one card or two) - dispatched here, before the Qwen family's kv-cache and parallel rules
  // (its own, cli::qwen4exp::check_serve, replace them), as b70-decode dispatches it.
  if (is_q4) {
    cli::qwen4exp::ServeArgs qs;
    qs.snapshot_dir = loader::resolve_snapshot(path);
    cli::qwen4exp::DecodeArgs& qa = qs.args;
    qa.path = qs.snapshot_dir;
    qa.max_len = max_len_arg;
    qa.reserve = mem_reserve;
    qa.card_bytes = hw.card_bytes;
    qa.lm_head = lm_head;   // int8 (spec 9), as every served model
    qa.layers = q4_layers;
    qa.layers_auto = q4_layers_auto;
    qa.pipe = pipe;
    qa.pp_given = pp_given;
    qa.prefill = true;      // every request prefills: the plan carries the prefill scratch
    qa.pp_backend_given = have_pp_backend;
    qa.pp_backend = pp_backend;
    qa.kv8 = kv_cache == runtime::KvCache::Int8;
    qs.mtp_k = mtp_k;       // a fixed K (cli::resolve_spec: Flash-Next's MTP is not adaptive yet)
    qs.spec_lookup = spec_lookup;
    qs.draft_vocab = draft_vocab != 0 || !draft_vocab_ids.empty();
    qa.load_mtp = qs.loads_head();
    qs.prefix_auto = prefix_auto;
    qs.prefix_cache_gb = prefix_cache_gb;
    return serve_qwen4exp(qs, options);
  }
  // Spec 20e: Kolibri-1 (model_type kolibri1) runs runtime::kolibri::KolibriEngine on two cards by default -
  // dispatched here, before the Qwen family's kv-cache and parallel rules (its own, cli::kolibri::check_args,
  // replace them: a --pipeline-split is its default two cards), as b70-decode dispatches it.
  if (is_kol) {
    options.chat_format = server::ChatFormat::kolibri();
    KolibriServe ks;
    ks.snapshot_dir = loader::resolve_snapshot(path);
    cli::kolibri::DecodeArgs& ka = ks.args;
    ka.path = ks.snapshot_dir;
    ka.max_len = max_len_arg;
    ka.reserve = mem_reserve;
    ka.card_bytes = hw.card_bytes;
    ka.lm_head = lm_head;   // int8 (spec 9), as every served model
    ka.pipe = pipe;
    ka.pp_given = pp_given;
    ka.prefill = true;      // every request prefills: the plan carries the prefill scratch
    ka.pp_backend_given = have_pp_backend;
    ka.pp_backend = pp_backend;
    ka.mtp = false;         // cli::resolve_spec refused every proposer
    ka.kv8 = kv_cache == runtime::KvCache::Int8;
    ks.prefix_auto = prefix_auto;
    ks.prefix_cache_gb = prefix_cache_gb;
    return serve_kolibri(ks, options);
  }

  runtime::PrefillPath pp_path = cli::prefill_path(
      true, have_pp_backend ? pp_backend : runtime::prefill::default_prefill_backend());

  // Spec 16d: the pipeline's refusals that need no checkpoint (cli/pipeline_serve.h) - the override flags on
  // one card and prefill profiling - before the snapshot is even resolved; the model's own follow below.
  cli::ServePipelineContext pipe_ctx;
  pipe_ctx.mtp = mtp_on || spec_lookup;
  {
    const char* profile = std::getenv("B70_PREFILL_PROFILE");
    pipe_ctx.prefill_profile = profile != nullptr && std::string(profile) == "1";
  }
  const std::string snapshot_dir = loader::resolve_snapshot(path);
  // Spec 18b / 18d: dispatch on config.json's model_type. The chat format (template variables,
  // reasoning tags, tool-call syntax) follows it (server/chat_format.h: anything but
  // k2_horizon is the Qwen path Qwen3.8 and Ornith share); K2-Horizon runs its own
  // engine (runtime/k2, cli::k2::K2EngineAdapterT) - what that engine does not have is refused
  // here by name, before the device.
  options.chat_format = server::ChatFormat::for_model_type(model_type);
  // Spec 16d: the model's parallel refusals, before the model dispatch below - so K2 under --parallel pp is
  // refused by name rather than served on one card.
  pipe_ctx.model_type = model_type;
  cli::check_serve_pipeline(pipe, pipe_ctx);
  if (is_k2) {
    if (have_pp_backend && pp_backend != runtime::PrefillBackend::L0)
      throw std::runtime_error(std::string("K2-Horizon prefills on the l0 backend only (spec 18c), "
                                           "not ") + runtime::prefill_backend_name(pp_backend) +
                               ": l0-int8's h8 linears need K in whole 1024-k Hadamard blocks and "
                               "K2's hidden is 2560 - --prefill-backend "
                               "l0, or omit it");
    K2Serve k2;
    k2.path = path;
    k2.snapshot_dir = snapshot_dir;
    k2.max_len_arg = max_len_arg;
    k2.mem_reserve = mem_reserve;
    k2.kv_cache = kv_cache;
    k2.lm_head = lm_head;
    k2.prefix_auto = prefix_auto;
    k2.prefix_cache_gb = prefix_cache_gb;
    return serve_k2(k2, options);
  }
  const std::vector<uint32_t> eos = eos_ids(snapshot_dir);
  const uint32_t trained = loader::trained_context(snapshot_dir);
  cli::check_before_load(max_len_arg, trained, /*require_quantum=*/true);   // spec 6 §10
  // Spec 8 §11: what V' is built from - tokenizer.json's added tokens, the EOS ids, the
  // ranked file. Read before the device is touched, so a bad file fails fast.
  loader::DraftVocabSpec dv_spec;
  if (draft_vocab != 0) {
    dv_spec.size = draft_vocab;
    dv_spec.for_drafter = dflash.on;   // spec 19c: the drafter's head, no MTP head
    dv_spec.added = loader::added_token_ids_file(snapshot_dir + "tokenizer.json");
    dv_spec.eos = eos;
    if (!draft_vocab_ids.empty()) dv_spec.ranked = loader::read_ranked_ids(draft_vocab_ids);
  }
  if (pipe.replica()) {   // plan 13b: one engine per card behind the scheduler
    ReplicaServe rs;
    rs.snapshot_dir = snapshot_dir;
    rs.max_len_arg = max_len_arg;
    rs.trained = trained;
    rs.mem_reserve = mem_reserve;
    rs.kv_cache = kv_cache;
    rs.lm_head = lm_head;
    rs.dv_spec = dv_spec;
    rs.mtp_k = mtp_k;
    rs.mtp_auto = mtp_on && spec.adaptive;
    rs.spec_lookup = spec_lookup;
    rs.spec_max = spec.max_k;
    rs.have_pp_backend = have_pp_backend;
    rs.pp_backend = pp_backend;
    rs.pp_path = pp_path;
    rs.prefix_auto = prefix_auto;
    rs.prefix_cache_gb = prefix_cache_gb;
    rs.eos = eos;
    return serve_replica(rs, options);
  }
  // --parallel auto (the zero-config cleanup): one card when the model and the asked context fit it, else
  // pipeline parallel (cli::parallel_auto, from the checkpoint's descriptor before the load).
  if (pipe.mode == cli::Parallel::Auto) {
    pipe.devices = cli::parallel_auto(snapshot_dir, max_len_arg, mtp_on || spec_lookup, kv_cache, pp_path,
                                      hw.card_bytes, mem_reserve);
    std::fprintf(stderr, "parallel: auto -> %s\n",
                 pipe.on() ? "pp (the model and the context do not fit one card)" : "1 (one card)");
    if (pipe.on()) {
      if (hw.cards < 2)
        throw std::runtime_error("--parallel auto: this model at this --max-len does not fit one card and Level Zero "
                                 "shows one - a shorter --max-len, or both cards visible (ZE_AFFINITY_MASK=0,1)");
      cli::dflash::check_engine(dflash, true, kv_cache, /*pipeline_served=*/true, pipe.handoff == runtime::PpHandoff::Peer);
      if (dflash_lookup)
        throw std::runtime_error("--spec dflash-lookup is built for one card (spec 19f), and this model and context "
                                 "need two: --spec dflash, or a shorter --max-len");
    }
  }
  std::unique_ptr<l0::Context> d0 = std::make_unique<l0::Context>(0u);
  if (pipe.on()) {   // spec 16d: two cards, serve_pipeline's own flow
    PipeServe ps;
    ps.snapshot_dir = snapshot_dir;
    ps.max_len_arg = max_len_arg;
    ps.trained = trained;
    ps.mem_reserve = mem_reserve;
    ps.kv_cache = kv_cache;
    ps.lm_head = lm_head;
    ps.dv_spec = dv_spec;
    ps.mtp_k = mtp_k;
    ps.mtp_auto = mtp_on;
    ps.spec_lookup = spec_lookup;
    ps.spec_max = spec.max_k;
    ps.have_pp_backend = have_pp_backend;
    ps.pp_backend = pp_backend;
    ps.pp_path = pp_path;
    ps.prefix_auto = prefix_auto;
    ps.prefix_cache_gb = prefix_cache_gb;
    ps.pipe = pipe;
    ps.eos = eos;
    ps.dflash = dflash;   // spec 19d
    return serve_pipeline(*d0, ps, options);
  }
  l0::Context& context = *d0;
  std::fprintf(stderr, "device: %s (%u EUs)\n", context.name().c_str(), context.eu_count());
  loader::LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load(context, snapshot_dir, cli::load_len(max_len_arg, trained),
                        /*mtp=*/mtp_k > 0 || spec_lookup, lm_head, dv_spec);
  }();
  // Spec 19c: the drafter beside the target, before the plan (its weights and buffers, and the engine's verify
  // slots and prefill taps for it, come out of what auto can give the context).
  std::unique_ptr<cli::dflash::Loaded> drafter_load;
  if (dflash.on) drafter_load = std::make_unique<cli::dflash::Loaded>(cli::dflash::load(context, dflash, model));
  // Spec 15c: the server prefills every request; a model whose prefill is not built is
  // refused here by name. Spec 15e serves Ornith (a MoE model): its chat template is the
  // checkpoint's (chat::Template, tests/tokenizer/ornith_*), its tool calls the Qwen XML
  // format src/server/toolcall.cc parses, its EOS ids generation_config.json's, and the
  // prefix cache, --max-len auto and MTP run on the descriptor like every model's.
  model::require_prefill(*model.desc);
  // Spec 6 §10: auto plans the largest max_len that fits and re-tables the model; an
  // explicit N is held to the same plan. Both print the plan's breakdown.
  // Plan 5d: the prefill chunk cap - the default, or kPrefillChunkCapRule's choice (cli/max_len.h) -
  // planned with max_len and given to the engine.
  pp_path.chunk_cap = cli::planned_chunk_cap(0, *model.desc);
  const runtime::DrafterPlan drafter_plan = drafter_load ? drafter_load->plan : runtime::DrafterPlan{};
  const uint32_t max_len = cli::settle(context, model, max_len_arg, mem_reserve, pp_path, kv_cache, drafter_plan);
  const uint32_t chunk_cap =
      cli::settle_chunk_cap(context, model, max_len, mem_reserve, pp_path, kv_cache, drafter_plan, 0);
  std::unique_ptr<runtime::dflash::DFlashDrafter> drafter;
  if (drafter_load)
    drafter = std::make_unique<runtime::dflash::DFlashDrafter>(context, std::move(drafter_load->w),
                                                               drafter_load->head_rows, max_len, dflash.max_k);
  const bool draft_vocab_on = model.draft_vocab != nullptr;
  runtime::Engine engine(context, std::move(model), max_len, /*debug_resid=*/false, kv_cache, drafter.get());
  if (drafter) drafter->attach(engine, draft_vocab_on);
  if (have_pp_backend) engine.set_prefill_backend(pp_backend);
  engine.set_prefill_chunk_cap(chunk_cap);   // plan 5d
  // Prefill setup at load, not in the first request: on l0-int8 this is the
  // one-time rotated column-scale pass (spec 5 T2).
  engine.prepare_prefill();
  std::fprintf(stderr, "%s\n", engine.memory_line().c_str());   // spec 6
  TokAdapter tokenizer(snapshot_dir + "tokenizer.json");
  // Spec 15e: the sampler masks at the tokenizer's count, the greedy argmax at the
  // descriptor's vocab_used (argmax.cl VOCAB_USED). They agree on every model served:
  // Qwen3.8 248077 and Ornith 248077 - the int4 checkpoint's tokenizer.json
  // defines Qwen3.8's ids (spec 15 §13; the base checkpoint's stopped at </think>, 248070,
  // which the descriptor masked from between 11ab54f and §13). A checkpoint whose
  // tokenizer.json and descriptor disagree is said once at startup
  // rather than discovered as an undecodable id.
  const model::ModelDesc& served = *engine.model().desc;
  if (tokenizer.vocab_used() != served.vocab_used)
    std::fprintf(stderr,
                 "note: tokenizer.json defines %u ids, %s's greedy argmax masks from %u; sampling "
                 "masks from %u\n",
                 tokenizer.vocab_used(), served.name.c_str(), served.vocab_used,
                 tokenizer.vocab_used());
  TemplateAdapter chat_template(snapshot_dir);
  EngineAdapter engine_adapter(engine, tokenizer.vocab_used(), dflash.on ? dflash.max_k : mtp_k, spec_lookup,
                               dflash.on, dflash_lookup);
  options.eos_ids = eos;
  // Sized after the model is loaded, so "available" already excludes the process's own load.
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(context, prefix_auto, prefix_cache_gb, options);
  options.parallel_desc = "1 card";
  server::Server server({tokenizer, chat_template, engine_adapter}, options);

  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::fprintf(stderr, ", prefill backend %s, mtp %s%u, lm_head %s%s%s, kv cache %s\n",
               runtime::prefill_backend_name(engine.prefill_backend()), mtp_on && spec.adaptive ? "auto, max " : "",
               mtp_k, loader::lm_head_form_name(lm_head),
               engine.draft_vocab() ? ", draft vocab " : "",
               engine.draft_vocab() ? loader::draft_vocab_name(engine.draft_vocab()).c_str() : "",
               runtime::kv_cache_name(engine.kv_cache()));
  print_proposer(options, spec_lookup, options.lookup_min_match, spec.max_k, options.lookup_history,
                 mtp_on && spec.adaptive, served.is_moe(), served.name);
  if (drafter) {   // spec 19c: the drafter's policy line (print_proposer's format)
    const server::MtpCost& c = options.mtp_adaptive.cost;
    std::fprintf(stderr, "spec dflash: %s drafter, K <= %u, cost verify M=1..%zu",
                 loader::draft_format_name(dflash.format), dflash.max_k, c.verify.size());
    for (double v : c.verify) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, ", draft k=1..%zu", c.draft.size());
    for (double v : c.draft) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, " (plain steps; derived until the box measures them)\n");
    if (dflash_lookup)   // spec 19f
      std::fprintf(stderr,
                   "spec dflash-lookup: prompt lookup inside DFlash - ENTER a match of >= %u ids beyond the drafter's "
                   "window %u, STAY while a verify keeps >= %u (and a match of >= %u remains), K %u lookup drafts\n",
                   server::LookupInDFlash::kEnterMatch, drafter->window(), server::LookupInDFlash::kStayAccepted,
                   server::LookupInDFlash::kStayMatch, dflash.max_k);
  }

  return listen_until_stopped(server, options);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "b70-serve: %s\n", error.what());
    return 1;
  }
}

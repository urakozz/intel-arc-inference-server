// b70-serve -- the OpenAI-compatible server over the replayed runtime::Engine (Qwen3.8, Agnes,
// Ornith), runtime::k2::K2Engine (K2-Horizon, spec 18d) or runtime::kolibri::KolibriEngine
// (Kolibri-1, spec 20e: two cards by default), picked by config.json's model_type; with --pp 2
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

#include "cli/k2_decode.h"
#include "cli/k2_serve_adapter.h"
#include "cli/kolibri_decode.h"
#include "cli/kolibri_serve.h"
#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "cli/pipeline_serve.h"
#include "cli/pipeline_serve_adapter.h"
#include "cli/pipeline_settle.h"
#include "cli/prefix_cache_size.h"
#include "cli/renamed_flags.h"
#include "cli/serve_adapters.h"
#include "l0/context.h"
#include "loader/loader.h"
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
               "usage: b70-serve <snapshot-or-repo> [--host 0.0.0.0] [--port 8000]\n"
               "                 [--max-len auto|N] [--device N] [--served-name NAME] [--queue 4]\n"
               "                 --max-len: auto (default) = the largest context that fits the card\n"
               "                 beside the weights (spec 6 §10); N = any multiple of 256 up to the\n"
               "                 trained context (config.json max_position_embeddings), refused at\n"
               "                 startup if it does not fit (B70_DECODE_ATTN=v1: the compiled lengths)\n"
               "                 [--mem-reserve-gb G]   device memory the plan leaves free (default\n"
               "                                        1.5: driver, kernels, slack; box-unconfirmed)\n"
               "                 The model is picked from the checkpoint's config.json: Qwen3.8,\n"
               "                 Agnes 3.0 Flash (spec 14), Ornith 1.5 35B-A3B (spec 15, MoE) or\n"
               "                 K2-Horizon MoVA 36B-A4B (spec 18: its own engine, prefill on l0\n"
               "                 only, no MTP head - --mtp, --spec mtp|lookup and --draft-vocab are\n"
               "                 refused; its chat_template_kwargs tool_call_format / reasoning_effort\n"
               "                 reach the template) or Kolibri-1 (spec 20e: its own engine on TWO cards\n"
               "                 by default - --pp 1 only when the model fits one card; prefill on l0\n"
               "                 only; no MTP head and bf16 KV only - --mtp, --spec mtp|lookup and\n"
               "                 --kv-cache int8 are refused; its chat_template_kwargs reasoning_effort\n"
               "                 none|minimal|low|medium|high|xhigh|max reach the template; requests\n"
               "                 without temperature / top_p / top_k sample with generation_config.json's\n"
               "                 1.0 / 0.97 / 128). UNVALIDATED on the cards (queue row 28)\n"
               "                 [--pp 1|2] [--pipeline-parallel-size 1|2]   2 (spec 16d): the model's\n"
               "                             layers over GPUs 0 and 1 of what Level Zero shows (Qwen3.8,\n"
               "                             Agnes, Ornith, Kolibri-1 - its default; K2-Horizon is refused).\n"
               "                             --max-len auto fits both cards; the prefix cache, --mtp /\n"
               "                             --spec (the MTP head on device 1), --kv-cache int8 and\n"
               "                             --lm-head work as on one card. Not with --device or a\n"
               "                             sycl-tla prefill. UNVALIDATED on the cards (queue row 27)\n"
               "                 [--pipeline-split auto|N]   --pp 2: device 0 runs layers [0, N)\n"
               "                             (auto, the default: the split whose heavier card holds the\n"
               "                             fewest bytes, with the MTP head and the cache's shadows)\n"
               "                 [--pipeline-handoff copy|peer]   --pp 2: how the residual crosses\n"
               "                             (copy, the default: a device copy and a cross-device event)\n"
               "                 [--prefill-backend sycl-tla|l0|l0-int8]   Default: l0-int8.\n"
               "                 [--log-requests DIR]   write DIR/NNNNNN.json per request\n"
               "                 [--prefix-cache-gb auto|N]  the prefix cache, in SYSTEM RAM (pinned\n"
               "                                        host memory, not VRAM), GiB. auto (default):\n"
               "                                        min(32, half the RAM, available - 8), off\n"
               "                                        below 4. 0 = off: every request prefills\n"
               "                                        in full (spec 7)\n"
               "                 [--prefix-split-last]  a flag (present = on), chat requests only.\n"
               "                                        A thinking model's next turn diverges at this\n"
               "                                        prompt's LAST id (<think>\\n re-rendered as\n"
               "                                        <think>\\n\\n</think>); this prefills to len - 1,\n"
               "                                        snapshots there and runs the last id as one\n"
               "                                        decode step, so the next turn restores right\n"
               "                                        there instead of up to 2047 ids earlier (~1 s).\n"
               "                                        Cost: a near-tie first token may differ from an\n"
               "                                        uncached run. On for agentic sessions; off for\n"
               "                                        benchmarks and the golden gates.\n"
               "                 [--mtp off|1|2|3|auto]   speculative decoding with the MTP\n"
               "                             head: K guesses per step (off/0 = none, the default;\n"
               "                             auto = 0..3 per step from the request's own hit rate). Loads the\n"
               "                             head (+0.85 GB weights - Ornith's MoE head +0.50 GB -\n"
               "                             + its KV and state slots);\n"
               "                             any --max-len (spec 8 §12).\n"
               "                 [--mtp auto [--mtp-max 3] [--mtp-cost SPEC]]   K per iteration\n"
               "                             from each request's own acceptance (spec 8 §10).\n"
               "                             SPEC: \"verify=1,1.17,1.52,1.74;draft=0.19,0.37,0.55\"\n"
               "                             (plain-step units; default: the --lm-head form's)\n"
               "                 [--lm-head bf16|int8]   Default: int8 (spec 9: quantised at load\n"
               "                 from the bf16 head; bf16 is the checkpoint's own head)\n"
               "                 [--draft-vocab off|32k|64k|128k]   MTP drafts over a reduced\n"
               "                             vocabulary (spec 8 §11; default off): the head's\n"
               "                             rows for the added tokens, then\n"
               "                             --draft-vocab-ids, then the lowest ids. Output\n"
               "                             unchanged (verify is full); needs --mtp. With\n"
               "                             --mtp auto the draft costs scale with |V'|\n"
               "                             (derived; an explicit --mtp-cost draft= wins)\n"
               "                 [--draft-vocab-ids FILE]   ranked ids, most frequent first\n"
               "                             (tools/draft_vocab/rank.py over --log-requests logs)\n"
               "                 [--kv-cache bf16|int8]   the KV cache's storage (spec 12):\n"
               "                             bf16 (default) or int8 - every K and V row rotated\n"
               "                             (256-point Hadamard) and stored as int8 with one\n"
               "                             fp16 scale per token and head: half the KV memory,\n"
               "                             so --max-len auto reaches about twice the context\n"
               "                             (Qwen3.8: the trained 262144). Accuracy gates on\n"
               "                             the card pending; needs the l0 / l0-int8 backends\n"
               "                 [--spec off|mtp|lookup]   the speculative proposer (spec 19\n"
               "                             decision 3; default: --mtp's). mtp = --mtp auto unless\n"
               "                             --mtp is given. lookup (spec 19e) = prompt lookup: the\n"
               "                             drafts are the continuation of the longest earlier match\n"
               "                             of the request's own ids, verified by the MTP verify lists\n"
               "                             (loads the head for them; Qwen3.8 only so far); K per\n"
               "                             iteration by --mtp auto's policy with a free draft.\n"
               "                             Greedy output unchanged; sampled output lossless.\n"
               "                             UNVALIDATED on the card (plan 19e Task 2)\n"
               "                 [--spec-min-match N]   lookup drafts only after a match of >= N ids\n"
               "                             (2..64, default 3)\n"
               "                 [--spec-max K]   lookup's largest K (1..3, default 3)\n"
               "                 [--spec-cost SPEC]   lookup's cost table, --mtp-cost's syntax\n"
               "                             (default: the --lm-head form's verify rows, free drafts)\n"
               "                 [--spec-history N]   also match the generated ids of the last N\n"
               "                             requests (default 0)\n");
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
  if (!config.contains("eos_token_id"))
    throw std::runtime_error("generation_config.json has no eos_token_id");

  const auto add = [](const nlohmann::json& value, std::vector<uint32_t>& ids) {
    if (!value.is_number_unsigned() && !value.is_number_integer())
      throw std::runtime_error("generation_config.json eos_token_id is not an integer");
    const int64_t id = value.get<int64_t>();
    if (id < 0 || static_cast<uint64_t>(id) > 0xFFFFFFFFull)
      throw std::runtime_error("generation_config.json eos_token_id is out of range");
    ids.push_back(static_cast<uint32_t>(id));
  };

  std::vector<uint32_t> ids;
  const nlohmann::json& value = config.at("eos_token_id");
  if (value.is_array()) {
    for (const auto& id : value) add(id, ids);
  } else {
    add(value, ids);
  }
  if (ids.empty()) throw std::runtime_error("generation_config.json eos_token_id is empty");
  return ids;
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
  uint32_t device = l0::Context::kFromEnv;
  runtime::KvCache kv_cache = runtime::KvCache::Bf16;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
};

int serve_k2(const K2Serve& a, server::Options options) {
  const std::vector<uint32_t> eos = eos_ids(a.snapshot_dir);
  const uint32_t trained = loader::trained_context(a.snapshot_dir);
  cli::check_before_load(a.max_len_arg, trained, /*require_quantum=*/true);   // spec 6 §10
  l0::Context context(a.device);
  std::fprintf(stderr, "device: %s (%u EUs)%s\n", context.name().c_str(), context.eu_count(),
               a.device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
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
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::fprintf(stderr, ", K2-Horizon (%s chat format), prefill backend l0, attention %s "
                       "(B70_K2_ATTN), mtp none, lm_head %s, kv cache %s, prefix snapshots KV-only\n",
               options.chat_format.name(), runtime::k2::k2_attn_name(runtime::k2::k2_attn()),
               loader::lm_head_form_name(a.lm_head), runtime::kv_cache_name(engine.kv_cache()));
  return listen_until_stopped(server, options);
}

// Spec 20e: Kolibri-1 served - runtime::kolibri::KolibriEngine (one card or, the default, two: spec 16b's
// pieces as 20c / 20d built them) behind cli::kolibri::KolibriEngineAdapterT (the rings + full-KV prefix
// snapshots, host sampling at the tokenizer's id count), Kolibri's chat format (server::ChatFormat,
// set by run() from model_type), its EOS [127906, 127901] and its sampling defaults from
// generation_config.json. b70-decode's flow (cli/kolibri_decode.h run_decode): the refusals before the
// device (cli::kolibri::check_args: --mtp, --kv-cache int8, the prefill backends, a --pp 1 that does not
// fit, --device with --pp 2), both cards in one Level Zero context with peer access, the placement and
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
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(devs == 2 ? 0u : da.device);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  const std::array<size_t, runtime::kPpDevices> dev{c0->memory_bytes(), c1 ? c1->memory_bytes() : 0};
  if (devs == 1)
    std::fprintf(stderr, "device: %s (%u EUs)%s\n", c0->name().c_str(), c0->eu_count(),
                 da.device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  else
    std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--pp 2: one context]\n", c0->name().c_str(),
                 c0->eu_count(), c1->name().c_str(), c1->eu_count());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--pp 2: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), "
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
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  const std::string placement =
      devs == 2 ? std::string("--pp 2 (hand-off ") + runtime::pp_handoff_name(engine.handoff()) + ", split " +
                      std::to_string(engine.split()) + ": device 0 layers [0, " + std::to_string(engine.split()) +
                      "), device 1 [" + std::to_string(engine.split()) + ", " +
                      std::to_string(engine.model().desc.layers) + "))"
                : std::string("--pp 1");
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

// The speculative proposer's startup lines (spec 19e's lookup, spec 8 §10's auto policy) -
// one card's and --pp 2's.
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
    // unmeasured until the box records them; --mtp-cost replaces the table.
    if (moe_default_costs)
      std::fprintf(stderr, "mtp auto: %s's own costs are not measured yet; the table above is "
                           "Qwen3.8's (pass --mtp-cost once the box has them)\n",
                   model_name.c_str());
  }
}

// Spec 16d: b70-serve --pp 2 - the Qwen-family model over two B70s (runtime::PipelineEngine
// behind cli::pp::PipelineEngineAdapterT). b70-decode --pp 2's flow (cli/pipeline_decode.h):
// both cards in one Level Zero context and peer access, the whole checkpoint (with the MTP
// head when a proposer needs it) loaded onto device 0, the split and max_len planned over the
// loaded bytes with the server's terms (the prefill scratch per card; the MTP head, its
// embedding replica and buffers on device 1; the prefix cache's block shadows when the cache
// may be on - cli/pipeline_settle.h), then placed, then the two-card prefill prepared before
// the first request. --pp's refusals ran before this (cli/pipeline_serve.h).
struct PipeServe {
  std::string snapshot_dir;
  cli::MaxLenArg max_len_arg;
  uint32_t trained = 0;
  size_t mem_reserve = 0;
  runtime::KvCache kv_cache = runtime::KvCache::Bf16;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  loader::DraftVocabSpec dv_spec;
  uint32_t mtp_k = 0;
  bool mtp_auto = false, spec_lookup = false, default_mtp_costs = true;
  uint32_t spec_min_match = 3, spec_max = 3, spec_history = 0;
  bool have_pp_backend = false;
  runtime::PrefillBackend pp_backend = runtime::PrefillBackend::L0Int8;
  runtime::PrefillPath pp_path;
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  cli::PipelineArgs pipe;
  std::vector<uint32_t> eos;
};

int serve_pipeline(const PipeServe& a, server::Options options) {
  cli::require_two_devices(l0::Context::gpu_count());
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  std::fprintf(stderr, "devices: 0 %s (%u EUs), 1 %s (%u EUs) [--pp 2: one context]\n",
               d0.name().c_str(), d0.eu_count(), d1.name().c_str(), d1.eu_count());
  if (!d0.can_access_peer(d1))
    throw std::runtime_error(
        "--pp 2: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer 0 -> 1 is false), "
        "and both hand-offs write it. Peer access needs the P2P-capable kernel and both cards "
        "under one root complex (docs/10-the-box.md)");
  loader::LoadedModel full = [&] {
    StdoutToStderr redirect;
    return loader::load(d0, a.snapshot_dir, cli::load_len(a.max_len_arg, a.trained),
                        /*mtp=*/a.mtp_k > 0 || a.spec_lookup, a.lm_head, a.dv_spec);
  }();
  model::require_prefill(*full.desc);   // spec 15c, as on one card
  const model::ModelDesc& d = *full.desc;
  const runtime::PpWeights w = runtime::pp_weights(full);
  const std::array<size_t, runtime::kPpDevices> dev{d0.memory_bytes(), d1.memory_bytes()};
  // The block hook's shadows are planned whenever the cache may be on: auto decides from the
  // host's RAM only after the load (make_prefix_alloc), and an unplanned shadow would be an
  // allocation in the first request.
  const bool hook = a.prefix_auto || a.prefix_cache_gb > 0;
  const cli::PpSettled settled = cli::pp_settle(d, w, dev, a.max_len_arg, full.trained_max_len,
                                                a.mem_reserve, a.kv_cache, a.pp_path,
                                                cli::pp_extras(full, hook), a.pipe);
  std::vector<loader::LoadedModel> stages = runtime::place_stages(d0, d1, std::move(full), settled.split);
  if (stages[0].max_len != settled.max_len) {   // auto loaded at a small length: re-table both
    loader::set_max_len(d0, stages[0], settled.max_len);
    loader::set_max_len(d1, stages[1], settled.max_len);
  }
  runtime::PipelineOptions opt;
  opt.handoff = a.pipe.handoff;
  runtime::PipelineEngine engine(d0, d1, std::move(stages), settled.max_len, opt, a.kv_cache);
  if (a.have_pp_backend) engine.set_prefill_backend(a.pp_backend);
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
      engine, tokenizer.vocab_used(), model::Qwen35::kVocab, a.mtp_k, a.spec_lookup);
  options.eos_ids = a.eos;
  // Pinned host memory of the one context both cards share: either device's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(d0, a.prefix_auto, a.prefix_cache_gb, options);
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, settled.max_len);
  print_eos(a.eos);
  std::fprintf(stderr,
               ", --pp 2 (hand-off %s, split %u: device 0 layers [0, %u), device 1 [%u, %u)), prefill "
               "backend %s, mtp %s%u%s, lm_head %s%s%s, kv cache %s\n",
               runtime::pp_handoff_name(engine.handoff()), engine.split(), engine.split(),
               engine.stage(1).first, engine.stage(1).last,
               runtime::prefill_backend_name(engine.prefill_backend()), a.mtp_auto ? "auto, max " : "",
               a.mtp_k, engine.mtp() ? " (the head on device 1)" : "", loader::lm_head_form_name(a.lm_head),
               engine.draft_vocab() ? ", draft vocab " : "",
               engine.draft_vocab() ? loader::draft_vocab_name(engine.draft_vocab()).c_str() : "",
               runtime::kv_cache_name(engine.kv_cache()));
  print_proposer(options, a.spec_lookup, a.spec_min_match, a.spec_max, a.spec_history, a.mtp_auto,
                 served.is_moe() && a.default_mtp_costs, served.name);
  return listen_until_stopped(server, options);
}

int run(int argc, char** argv) {
  std::string path;
  server::Options options;
  cli::MaxLenArg max_len_arg;   // spec 6 §10: auto unless --max-len N
  runtime::KvCache kv_cache = runtime::default_kv_cache();   // spec 12b: --kv-cache
  size_t mem_reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  uint32_t device = l0::Context::kFromEnv;
  std::string pp_backend_arg;
  bool have_pp_backend = false;
  // Spec 7: system RAM, not VRAM; auto by default (cli/prefix_cache_size.h).
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  uint32_t mtp_k = 0;
  bool mtp_auto = false, have_mtp_tuning = false;   // spec 8 §10
  uint32_t mtp_max = 3;
  std::string mtp_cost_arg;
  // Spec 19 decision 3 / spec 19e: --spec; unset keeps --mtp's meaning exactly.
  std::string spec_arg;
  bool have_mtp_arg = false, have_spec_tuning = false;
  uint32_t spec_min_match = 3, spec_max = runtime::Engine::kMaxDraft, spec_history = 0;
  std::string spec_cost_arg;
  // Spec 9 §3: the serving default is the gated int8 head (amendment §8, L3).
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  // Spec 8 §11: off until the box rows decide (§11 "Gates").
  uint32_t draft_vocab = 0;
  std::string draft_vocab_ids;
  cli::PipelineArgs pipe;   // spec 16d: --pp, --pipeline-split, --pipeline-handoff
  bool pp_given = false;    // spec 20e: Kolibri-1's --pp defaults to 2, so "given" matters

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
    } else if (arg == "--mem-reserve-gb") {
      mem_reserve = cli::parse_reserve_gb(value(i, "--mem-reserve-gb"));
    } else if (arg == "--device") {
      device = parse_u32("--device", value(i, "--device"));
      if (device == l0::Context::kFromEnv) throw std::runtime_error("--device is out of range");
    } else if (arg == "--served-name") {
      options.served_model = value(i, "--served-name");
    } else if (arg == "--queue") {
      options.queue_depth = parse_u32("--queue", value(i, "--queue"));
    } else if (arg == "--log-requests") {
      options.log_requests_dir = value(i, "--log-requests");
    } else if (arg == "--prefix-cache-gb") {
      const std::string v = value(i, "--prefix-cache-gb");
      prefix_auto = v == "auto";
      if (!prefix_auto) prefix_cache_gb = parse_u32("--prefix-cache-gb", v);
    } else if (arg == "--prefix-split-last") {
      options.prefix_split_last = true;
    } else if (arg == "--mtp") {
      const std::string v = value(i, "--mtp");
      have_mtp_arg = true;
      mtp_auto = v == "auto";
      if (!mtp_auto) {
        mtp_k = v == "off" ? 0 : parse_u32("--mtp", v);
        if (mtp_k > runtime::Engine::kMaxDraft)
          throw std::runtime_error("--mtp expects off, 0..3 or auto");
      }
    } else if (arg == "--mtp-max") {
      mtp_max = parse_u32("--mtp-max", value(i, "--mtp-max"));
      if (mtp_max == 0 || mtp_max > runtime::Engine::kMaxDraft)
        throw std::runtime_error("--mtp-max expects 1..3");
      have_mtp_tuning = true;
    } else if (arg == "--mtp-cost") {
      mtp_cost_arg = value(i, "--mtp-cost");
      have_mtp_tuning = true;
    } else if (arg == "--spec") {
      spec_arg = value(i, "--spec");
      if (spec_arg != "off" && spec_arg != "mtp" && spec_arg != "lookup")
        throw std::runtime_error("--spec expects off, mtp or lookup, got '" + spec_arg + "'");
    } else if (arg == "--spec-min-match") {
      spec_min_match = parse_u32("--spec-min-match", value(i, "--spec-min-match"));
      if (spec_min_match < 2 || spec_min_match > 64)
        throw std::runtime_error("--spec-min-match expects 2..64");
      have_spec_tuning = true;
    } else if (arg == "--spec-max") {
      spec_max = parse_u32("--spec-max", value(i, "--spec-max"));
      if (spec_max == 0 || spec_max > runtime::Engine::kMaxDraft)
        throw std::runtime_error("--spec-max expects 1..3");
      have_spec_tuning = true;
    } else if (arg == "--spec-cost") {
      spec_cost_arg = value(i, "--spec-cost");
      have_spec_tuning = true;
    } else if (arg == "--spec-history") {
      spec_history = parse_u32("--spec-history", value(i, "--spec-history"));
      have_spec_tuning = true;
    } else if (arg == "--prefill-backend") {
      pp_backend_arg = value(i, "--prefill-backend");
      have_pp_backend = true;
    } else if (arg == "--lm-head") {
      const std::string v = value(i, "--lm-head");
      if (!loader::parse_lm_head_form(v, lm_head))
        throw std::runtime_error("--lm-head expects bf16 or int8, got '" + v + "'");
    } else if (arg == "--draft-vocab") {
      const std::string v = value(i, "--draft-vocab");
      if (!loader::parse_draft_vocab(v, draft_vocab))
        throw std::runtime_error("--draft-vocab expects off, 32k, 64k or 128k, got '" + v + "'");
    } else if (arg == "--draft-vocab-ids") {
      draft_vocab_ids = value(i, "--draft-vocab-ids");
    } else if (arg == "--kv-cache") {   // spec 12b
      kv_cache = cli::parse_kv_cache_arg(value(i, "--kv-cache"));
    } else if (arg == "--pp" || arg == "--pipeline-parallel-size") {   // spec 16d
      pipe.devices = cli::parse_pipeline_devices(arg, value(i, arg.c_str()));
      pp_given = true;
    } else if (arg == "--pipeline-split") {
      cli::parse_pipeline_split(value(i, "--pipeline-split"), pipe);
    } else if (arg == "--pipeline-handoff") {
      cli::parse_pipeline_handoff(value(i, "--pipeline-handoff"), pipe);
    } else if (!arg.empty() && arg[0] == '-') {
      usage();
      throw std::runtime_error(cli::unknown_option(arg));
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
  // Spec 19 decision 3: --spec names the proposer; --mtp keeps its meaning (--spec mtp).
  const bool spec_lookup = spec_arg == "lookup";
  if (spec_arg == "mtp" && !have_mtp_arg) mtp_auto = true;
  if ((spec_arg == "off" || spec_lookup) && have_mtp_arg && (mtp_auto || mtp_k > 0))
    throw std::runtime_error("--spec " + spec_arg + " and --mtp: one proposer per server");
  if (have_spec_tuning && !spec_lookup)
    throw std::runtime_error("--spec-min-match, --spec-max, --spec-cost and --spec-history need "
                             "--spec lookup");
  if (spec_lookup) {
    options.spec_lookup = true;
    options.lookup_min_match = spec_min_match;
    options.lookup_history = spec_history;
    options.lookup_adaptive.max_k = spec_max;
    options.lookup_adaptive.cost = (lm_head == loader::LmHeadForm::Int8 ? server::MtpCost::int8_head()
                                                                       : server::MtpCost::bf16_head())
                                       .with_free_drafts();
    if (!spec_cost_arg.empty()) {
      try {
        options.lookup_adaptive.cost =
            server::MtpCost::parse(spec_cost_arg, options.lookup_adaptive.cost);
      } catch (const std::invalid_argument& error) {
        throw std::runtime_error(std::string(error.what()) + " (--spec-cost)");
      }
    }
    if (options.lookup_adaptive.cost.max_k() < spec_max)
      throw std::runtime_error("--spec-cost covers K up to " +
                               std::to_string(options.lookup_adaptive.cost.max_k()) +
                               ", below --spec-max " + std::to_string(spec_max));
  }
  // Spec 8 §10: --mtp auto loads the head for --mtp-max drafts and chooses K per iteration;
  // the cost table defaults to the head form's, --mtp-cost overrides it.
  if (have_mtp_tuning && !mtp_auto)
    throw std::runtime_error("--mtp-max and --mtp-cost need --mtp auto");
  if (mtp_auto) {
    mtp_k = mtp_max;
    options.mtp_auto = true;
    options.mtp_adaptive.max_k = mtp_max;
    options.mtp_adaptive.cost = lm_head == loader::LmHeadForm::Int8 ? server::MtpCost::int8_head()
                                                                    : server::MtpCost::bf16_head();
    // Spec 8 §11: a draft over V' reads |V'| / 248320 of the head, so the drafts' head
    // share scales with it (derived; MtpCost::kInt8DraftHeadShare). Applied to the base
    // --mtp-cost parses over, so an explicit draft= wins and a verify= alone keeps it.
    if (draft_vocab != 0)
      options.mtp_adaptive.cost = options.mtp_adaptive.cost.with_draft_vocab(
          lm_head == loader::LmHeadForm::Int8 ? server::MtpCost::kInt8DraftHeadShare
                                              : server::MtpCost::kBf16DraftHeadShare,
          double(draft_vocab) / model::Qwen35::kVocab);
    if (!mtp_cost_arg.empty()) {
      try {
        options.mtp_adaptive.cost = server::MtpCost::parse(mtp_cost_arg, options.mtp_adaptive.cost);
      } catch (const std::invalid_argument& error) {
        throw std::runtime_error(error.what());
      }
    }
    if (options.mtp_adaptive.cost.max_k() < mtp_max)
      throw std::runtime_error("--mtp-cost covers K up to " +
                               std::to_string(options.mtp_adaptive.cost.max_k()) +
                               ", below --mtp-max " + std::to_string(mtp_max));
  }
  // Spec 8 §11: the draft vocabulary narrows the MTP head's drafts, so both flags need
  // --mtp (K or auto), and the ranked list needs a size to fill. Checked here, before the
  // device is touched. The compact head is gathered from the head in either --lm-head
  // form; the loader refuses only a checkpoint that ships the head int4.
  if ((draft_vocab != 0 || !draft_vocab_ids.empty()) && mtp_k == 0)
    throw std::runtime_error("--draft-vocab and --draft-vocab-ids need --mtp (K or auto)");
  if (!draft_vocab_ids.empty() && draft_vocab == 0)
    throw std::runtime_error("--draft-vocab-ids needs --draft-vocab 32k, 64k or 128k");
  if (options.host.empty()) throw std::runtime_error("--host must not be empty");
  if (options.served_model.empty()) throw std::runtime_error("--served-name must not be empty");
  if (!options.log_requests_dir.empty()) std::filesystem::create_directories(options.log_requests_dir);
  runtime::PrefillBackend pp_backend{};
  if (have_pp_backend && !runtime::parse_prefill_backend(pp_backend_arg, pp_backend))
    throw std::runtime_error("--prefill-backend expects sycl-tla, l0 or l0-int8, got '" + pp_backend_arg +
                             "'");

  // Spec 20e: Kolibri-1 (model_type kolibri1) runs runtime::kolibri::KolibriEngine on two cards by default -
  // dispatched here, before the Qwen family's kv-cache and --pp rules (its own, cli::kolibri::check_args,
  // replace them: a --pipeline-split without --pp is its default --pp 2), as b70-decode dispatches it. A
  // path that does not resolve is not Kolibri: the flow below reports it, so every rejection keeps its order.
  if (cli::kolibri::is_kolibri(path)) {
    if (spec_lookup)
      throw std::runtime_error("--spec lookup verifies its drafts through the MTP verify lists, which Kolibri-1's "
                               "engine does not capture (no MTP head, spec 20 §1): drop --spec lookup");
    options.chat_format = server::ChatFormat::kolibri();
    KolibriServe ks;
    ks.snapshot_dir = loader::resolve_snapshot(path);
    cli::kolibri::DecodeArgs& ka = ks.args;
    ka.path = ks.snapshot_dir;
    ka.max_len = max_len_arg;
    ka.reserve = mem_reserve;
    ka.device = device;
    ka.lm_head = lm_head;   // int8 by default (spec 9), as every served model
    ka.pipe = pipe;
    ka.pp_given = pp_given;
    ka.prefill = true;      // every request prefills: the plan carries the prefill scratch
    ka.pp_backend_given = have_pp_backend;
    ka.pp_backend = pp_backend;
    ka.mtp = mtp_k > 0 || mtp_auto;
    ka.kv8 = kv_cache == runtime::KvCache::Int8;
    ks.prefix_auto = prefix_auto;
    ks.prefix_cache_gb = prefix_cache_gb;
    return serve_kolibri(ks, options);
  }

  const runtime::PrefillPath pp_path = cli::prefill_path(
      true, have_pp_backend ? pp_backend : runtime::prefill::default_prefill_backend());

  // Spec 16d: --pp's refusals that need no checkpoint (cli/pipeline_serve.h) - the sub-flags
  // without --pp 2, --device, a sycl-tla prefill, the composed attention, prefill replay and
  // profiling - before the snapshot is even resolved; the model's own follow below.
  cli::ServePipelineContext pipe_ctx;
  pipe_ctx.device = device != l0::Context::kFromEnv;
  pipe_ctx.mtp = mtp_k > 0 || mtp_auto || spec_lookup;
  pipe_ctx.prefill_backend = pp_path.backend;
  pipe_ctx.composed_attn = pp_path.composed_attn;
  {
    const char* replay = std::getenv("B70_PREFILL_REPLAY");
    const char* profile = std::getenv("B70_PREFILL_PROFILE");
    pipe_ctx.prefill_replay = replay != nullptr && std::string(replay) == "1";
    pipe_ctx.prefill_profile = profile != nullptr && std::string(profile) == "1";
  }
  cli::check_serve_pipeline(pipe, pipe_ctx);
  const std::string snapshot_dir = loader::resolve_snapshot(path);
  // Spec 18b / 18d: dispatch on config.json's model_type. The chat format (template variables,
  // reasoning tags, tool-call syntax) follows it (server/chat_format.h: anything but
  // k2_horizon is the Qwen path Qwen3.8, Agnes and Ornith share); K2-Horizon runs its own
  // engine (runtime/k2, cli::k2::K2EngineAdapterT) - what that engine does not have is refused
  // here by name, before the device.
  std::string model_type;
  if (std::ifstream cf(snapshot_dir + "config.json"); cf) {
    std::stringstream cs;
    cs << cf.rdbuf();
    const nlohmann::json cj = nlohmann::json::parse(cs.str(), nullptr, /*allow_exceptions=*/false);
    if (cj.is_object()) model_type = cj.value("model_type", std::string());
  }
  options.chat_format = server::ChatFormat::for_model_type(model_type);
  // Spec 16d: the model's --pp refusals, before the model dispatch below - so K2 under --pp 2 is
  // refused by name rather than served on one card (Kolibri-1 was dispatched above, spec 20e).
  pipe_ctx.model_type = model_type;
  cli::check_serve_pipeline(pipe, pipe_ctx);
  if (model_type == "k2_horizon") {
    if (mtp_k > 0 || mtp_auto)
      throw std::runtime_error("K2-Horizon has no MTP head (spec 18 §9): drop --mtp / --spec mtp");
    if (spec_lookup)
      throw std::runtime_error("--spec lookup verifies its drafts through the MTP verify lists, "
                               "which K2-Horizon's engine does not capture (no head, spec 18 §9; "
                               "plan 19e Task 2's engine part): drop --spec lookup");
    if (have_pp_backend && pp_backend != runtime::PrefillBackend::L0)
      throw std::runtime_error(std::string("K2-Horizon prefills on the l0 backend only (spec 18c), "
                                           "not ") + runtime::prefill_backend_name(pp_backend) +
                               ": l0-int8's h8 linears need K in whole 1024-k Hadamard blocks and "
                               "K2's hidden is 2560; sycl-tla has no K2 walk - --prefill-backend "
                               "l0, or omit it");
    K2Serve k2;
    k2.path = path;
    k2.snapshot_dir = snapshot_dir;
    k2.max_len_arg = max_len_arg;
    k2.mem_reserve = mem_reserve;
    k2.device = device;
    k2.kv_cache = kv_cache;
    k2.lm_head = lm_head;
    k2.prefix_auto = prefix_auto;
    k2.prefix_cache_gb = prefix_cache_gb;
    return serve_k2(k2, options);
  }
  // Spec 12b: the Qwen3.5 family's int8-KV refusals, before the device is touched (K2 above has
  // its own engine's int8 twins; Kolibri, earlier, refuses int8 KV itself).
  cli::check_kv_cache(kv_cache, pp_path);
  const std::vector<uint32_t> eos = eos_ids(snapshot_dir);
  const uint32_t trained = loader::trained_context(snapshot_dir);
  cli::check_before_load(max_len_arg, trained, /*require_quantum=*/true);   // spec 6 §10
  // Spec 8 §11: what V' is built from - tokenizer.json's added tokens, the EOS ids, the
  // ranked file. Read before the device is touched, so a bad file fails fast.
  loader::DraftVocabSpec dv_spec;
  if (draft_vocab != 0) {
    dv_spec.size = draft_vocab;
    dv_spec.added = loader::added_token_ids_file(snapshot_dir + "tokenizer.json");
    dv_spec.eos = eos;
    if (!draft_vocab_ids.empty()) dv_spec.ranked = loader::read_ranked_ids(draft_vocab_ids);
  }
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
    ps.mtp_auto = mtp_auto;
    ps.spec_lookup = spec_lookup;
    ps.spec_min_match = spec_min_match;
    ps.spec_max = spec_max;
    ps.spec_history = spec_history;
    ps.default_mtp_costs = mtp_cost_arg.empty();
    ps.have_pp_backend = have_pp_backend;
    ps.pp_backend = pp_backend;
    ps.pp_path = pp_path;
    ps.prefix_auto = prefix_auto;
    ps.prefix_cache_gb = prefix_cache_gb;
    ps.pipe = pipe;
    ps.eos = eos;
    return serve_pipeline(ps, options);
  }
  l0::Context context(device);
  std::fprintf(stderr, "device: %s (%u EUs)%s\n", context.name().c_str(), context.eu_count(),
               device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  loader::LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load(context, snapshot_dir, cli::load_len(max_len_arg, trained),
                        /*mtp=*/mtp_k > 0 || spec_lookup, lm_head, dv_spec);
  }();
  // Spec 15c: the server prefills every request; a model whose prefill is not built is
  // refused here by name. Spec 15e serves Ornith (a MoE model): its chat template is the
  // checkpoint's (chat::Template, tests/tokenizer/ornith_*), its tool calls the Qwen XML
  // format src/server/toolcall.cc parses, its EOS ids generation_config.json's, and the
  // prefix cache, --max-len auto and MTP run on the descriptor like every model's.
  model::require_prefill(*model.desc);
  // Spec 6 §10: auto plans the largest max_len that fits and re-tables the model; an
  // explicit N is held to the same plan. Both print the plan's breakdown.
  const uint32_t max_len = cli::settle(context, model, max_len_arg, mem_reserve, pp_path, kv_cache);
  runtime::Engine engine(context, std::move(model), max_len, /*debug_resid=*/false, kv_cache);
  if (have_pp_backend) engine.set_prefill_backend(pp_backend);
  // Prefill setup at load, not in the first request: on l0-int8 this is the
  // one-time rotated column-scale pass (spec 5 T2).
  engine.prepare_prefill();
  std::fprintf(stderr, "%s\n", engine.memory_line().c_str());   // spec 6
  TokAdapter tokenizer(snapshot_dir + "tokenizer.json");
  // Spec 15e: the sampler masks at the tokenizer's count, the greedy argmax at the
  // descriptor's vocab_used (argmax.cl VOCAB_USED). They agree on every model served:
  // Qwen3.8 248077, Agnes 248089, and Ornith 248077 - the int4 checkpoint's tokenizer.json
  // defines Qwen3.8's ids (spec 15 §13; the base checkpoint's stopped at </think>, 248070,
  // which the descriptor masked from between 66a8923 and §13). A checkpoint whose
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
  EngineAdapter engine_adapter(engine, tokenizer.vocab_used(), mtp_k, spec_lookup);
  options.eos_ids = eos;
  // Sized after the model is loaded, so "available" already excludes the process's own load.
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(context, prefix_auto, prefix_cache_gb, options);
  server::Server server({tokenizer, chat_template, engine_adapter}, options);

  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::fprintf(stderr, ", prefill backend %s (SYCL component %s), mtp %s%u, lm_head %s%s%s, kv cache %s\n",
               runtime::prefill_backend_name(engine.prefill_backend()),
               runtime::prefill::sycl_available() ? "on" : "off", mtp_auto ? "auto, max " : "",
               mtp_k, loader::lm_head_form_name(lm_head),
               engine.draft_vocab() ? ", draft vocab " : "",
               engine.draft_vocab() ? loader::draft_vocab_name(engine.draft_vocab()).c_str() : "",
               runtime::kv_cache_name(engine.kv_cache()));
  print_proposer(options, spec_lookup, spec_min_match, spec_max, spec_history, mtp_auto,
                 served.is_moe() && mtp_cost_arg.empty(), served.name);

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

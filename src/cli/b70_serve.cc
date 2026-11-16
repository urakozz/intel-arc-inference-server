// b70-serve -- the OpenAI-compatible server over the replayed runtime::Engine (Qwen3.8,
// Ornith) or runtime::kolibri::KolibriEngine
// (Kolibri-1, spec 20e: two cards by default), picked by config.json's model_type; with --parallel pp
// (spec 16d) over runtime::PipelineEngine, the Qwen-family model split over two B70s
// (cli/pipeline_serve.h has what it serves and refuses); with --parallel tp (spec 17d) over runtime::TpEngine, Qwen3.8
// with every layer split over two B70s.
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <print>
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

#include "cli/dflash_args.h"    // spec 19c: --spec dflash
#include "cli/flags.h"
#include "cli/hardware_gate.h"
#include "cli/stdout_to_stderr.h"
#include "cli/kolibri_decode.h"
#include "cli/kolibri_serve.h"
#include "cli/kolibri_spec.h"     // Kolibri DSpark: --spec dflash with the DSpark drafter
#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "cli/pipeline_serve.h"
#include "cli/pipeline_serve_adapter.h"
#include "cli/pipeline_settle.h"
#include "cli/prefix_cache_size.h"
#include "cli/qwen4exp_chat.h"
#include "cli/qwen4exp_serve.h"
#include "cli/replica_engine.h"     // plan 13b: --parallel replica
#include "cli/slots_engine.h"      // plan 13c: sequence slots on one card
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
#include "runtime/tp_engine.h"   // spec 17d: --parallel tp
#include "runtime/tp_plan.h"
#include "runtime/verify_attn.h"   // spec 32e: the 64-row list's per-depth cap (the startup line)
#include "cli/tp_dflash.h"
#include "runtime/dflash/tp_dflash_drafter.h"
#include "server/lookup_in_dflash.h"   // spec 19f: --spec dflash-lookup's constants (the startup line)
#include "server/server.h"

namespace {

using cli::StdoutToStderr;

server::Server* g_server = nullptr;

void stop_server(int) {
  if (g_server != nullptr) g_server->stop();
}

// The flags (cli/flags.h: each member is its flag, its default and its --help text).
using cli::flags::Arg;
using cli::flags::Help;
using cli::flags::Section;
struct Flags {
  [[=Help("S", "speculative decoding: auto, off, mtp, dflash. auto: Qwen3.8's DFlash\n"
               "drafter on one card (and replica) when it is in the HF cache, else the model's\n"
               "MTP head (Qwen3.8 under --parallel tp or pp or with slots; Ornith), off\n"
               "elsewhere; the startup log says which and why. dflash on one card\n"
               "also copies earlier repeats (prompt lookup). Greedy output unchanged.")]]
  std::string spec = "auto";   // + the developer values lookup, dflash-lookup (cli/spec_args.h)
  [[=Help("K", "the proposer's cap: 1..3 drafts (mtp, lookup), 1..7 (dflash).", "the most")]]
  Arg<uint32_t> spec_max;   // its range is the proposer's: checked below
  [[=Help("P", "--spec dflash's drafter. Qwen3.8: required, urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound\n"
               "(packed int4).")]]
  Arg<std::string> draft_model;
  [[=Help("N", "how many requests run at once (vLLM's flag); the context follows. 1:\n"
               "one request at the largest context that fits. N > 1 (Qwen3.8): a\n"
               "card holds up to 4 slots - slot 0 the largest context that fits beside N - 1\n"
               "slots of 32768 positions (e.g. 131072 + 3 x 32768); one card up to 4, two cards\n"
               "(replica, the slots split between the cards) up to 8. A request that reaches its\n"
               "slot's end finishes with \"length\" and is logged. Ornith: 2 on two cards\n"
               "(replica, one request a card).")]]
  Arg<uint32_t> max_num_seqs{1};   // operator 2026-11-11 (cli/pipeline_args.h)
  [[=Help("auto|L", "auto: the largest context that fits beside the weights (beside the\n"
                    "other slots with --max-num-seqs), at most the trained one; L: slot 0 exactly L,\n"
                    "any multiple of 256 up to the trained context, refused if it does not fit.")]]
  std::string max_len = "auto";   // spec 6 §10
  [[=Help("P", "auto: Qwen3.8 on two cards runs tp at --max-num-seqs 1 with --max-len\n"
               "auto and replica at --max-num-seqs above 1; otherwise one card when the model\n"
               "and the asked context fit it, else pp; 1: one card; pp: the\n"
               "layers split over two cards. Kolibri-1 runs on two cards by default\n"
               "(the model needs both).\n"
               "replica (Qwen3.8, Ornith): the model on both cards, one engine each -\n"
               "--max-num-seqs 2 to 8 (Ornith: 2; 2 when not given: one request a card), each\n"
               "exactly what it is alone; Qwen3.8's --spec dflash: a drafter on each card,\n"
               "--max-num-seqs 2.\n"
               "tp (Qwen3.8): every layer split over both cards (needs peer\n"
               "access) - one request at a time, faster; --spec auto / mtp / lookup.")]]
  std::string parallel = "auto";
  [[=Help("auto|N", "pp: device 0 runs layers [0, N) (auto: the split whose heavier card\n"
                    "holds the fewest bytes). An override, for experiments; it implies pp.")]]
  Arg<std::string> pipeline_split{"auto"};
  [[=Help("H", "the address to listen on.")]] std::string host = "0.0.0.0";
  [[=Help("P", "the port to listen on.")]] uint32_t port = 8000;
  [[=Help("DIR", "write DIR/NNNNNN.json per request (timings, ids, text).")]] std::string log_requests;
  [[=Help("auto|N", "the prefix cache in SYSTEM RAM (pinned host memory), GiB. auto:\n"
                    "min(32, half the RAM, available - 8), off below 4; 0 = off.")]]
  std::string prefix_cache_gb = "auto";   // spec 7 (cli/prefix_cache_size.h)
  [[=Help("", "chat requests: prefill to len - 1 and run the last id as one decode\n"
              "step, so a thinking model's next turn is a prefix cache hit right there (agentic sessions;\n"
              "off for benchmarks and the golden gates).")]]
  bool prefix_split_last = false;
  [[=Help("N|auto", "Qwen3.8-Flash-Next only: load its first N layers (development mode); without it\n"
                    "the whole model is served, the experts that do not fit the cards read from pinned\n"
                    "host memory. Its PLE int8 file is <snapshot>-ple-int8/.")]]
  Arg<std::string> layers;   // spec 21e (until spec 22)
  [[=Section("\n  Still being measured (they go when their numbers decide, docs/open-work.md):\n"),
    =Help("bf16|int8", "the KV cache's form; int8 halves it.", "bf16")]]
  std::string kv_cache;   // spec 12b; empty: runtime::default_kv_cache()
  [[=Help("l0|l0-int8", "the prompt pass: bf16, or int8 on the matrix engines.", "l0")]] Arg<std::string> prefill_backend;
  [[=Help("off|32k|64k|128k", "the MTP head's (or DFlash drafter's) draft vocabulary.")]]
  std::string draft_vocab = "off";   // spec 8 §11: off until the box rows decide
  [[=Help("FILE", "the draft vocabulary's ids (with --draft-vocab).")]] std::string draft_vocab_ids;
};

void usage() {
  std::print(stderr,
             "usage: b70-serve <snapshot-or-repo> [--spec auto|off|mtp|lookup|dflash] [--max-num-seqs N]\n"
             "                 [--max-len auto|L] [--parallel auto|1|pp|replica|tp] [--host 0.0.0.0] [--port 8000]\n"
             "\n"
             "  The model is chosen from the checkpoint's config.json (Qwen3.8, Ornith 1.5,\n"
             "  Kolibri-1, Qwen3.8-Flash-Next) and served under its name\n"
             "  (the repo id, or the directory's). Everything else - the device memory the plan leaves free,\n"
             "  the int8 lm_head, the drafter's form, the speculative cost tables, the tokenizer, the chat\n"
             "  template and the generation defaults (the snapshot's) - the engine decides; what does not fit\n"
             "  is refused at startup, naming why. Cards: Intel Arc Pro B70, one or two (peer access between\n"
             "  them); ZE_AFFINITY_MASK picks which the driver shows.\n"
             "\n"
             "{}",
             cli::flags::help<Flags>(18));
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
    std::print(stderr, "{:d}", ids[i]);
  }
  std::fputc(']', stderr);
}

// Spec 7: the prefix cache's pinned system RAM, sized after the model is loaded (so
// "available" already excludes the process's own load) - shared by the Qwen-family and the
// other families' paths. Returns null when the cache is off; sets options.prefix_cache_bytes / prefix_alloc.
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
    std::print(stderr,
               "warning: --prefix-cache-gb {:d} pins more host RAM than is available ({:.0f} GiB);"
               " pinned pages cannot be swapped\n",
               prefix_cache_gb, double(host.available) / double(cli::kGiB));
  }
  if (prefix_cache_gb > 0) {
    const auto t0 = std::chrono::steady_clock::now();
    prefix_alloc = std::make_unique<PinnedAlloc>(context, size_t(prefix_cache_gb) << 30);
    options.prefix_cache_bytes = prefix_alloc->size();
    options.prefix_alloc = prefix_alloc.get();
    std::print(stderr, "prefix cache: {:d} GiB pinned system RAM ({}), allocated in {:.0f} ms\n",
               prefix_cache_gb, prefix_why.c_str(),
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                   .count());
  } else {
    std::print(stderr, "prefix cache: off ({})\n", prefix_why.c_str());
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
  cli::kolibri::SpecArgs spec;     // Kolibri DSpark: --spec dflash (cli/kolibri_spec.h)
};

int serve_kolibri(const KolibriServe& a, server::Options options) {
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(a.snapshot_dir, 0);
  // Kolibri DSpark: the drafter's checks and its plan terms (its config.json and safetensors header), before the device.
  cli::kolibri::DecodeArgs da = a.args;
  da.spec = cli::kolibri::spec_reserve(a.spec, pre, da.max_len.is_auto ? pre.trained_max_len : da.max_len.value);
  cli::kolibri::check_args(da, pre);
  // Spec 20 decision 3: the served context is capped at the trained 262144 - refused here, before the device
  // (settle() refuses it too, after the cards are open).
  if (!da.max_len.is_auto && da.max_len.value > pre.trained_max_len)
    throw std::runtime_error("--max-len " + std::to_string(da.max_len.value) + " exceeds Kolibri-1's trained context " +
                             std::to_string(pre.trained_max_len) + " - a longer context is not supported");
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
    std::print(stderr, "device: {} ({:d} EUs)\n", c0->name().c_str(), c0->eu_count());
  else
    std::print(stderr, "devices: 0 {} ({:d} EUs), 1 {} ({:d} EUs) [--parallel pp: one context]\n", c0->name().c_str(),
               c0->eu_count(), c1->name().c_str(), c1->eu_count());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), "
                             "and the hand-off writes it (docs/10-the-box.md)");
  const std::pair<model::KolPlacement, uint32_t> pl = cli::kolibri::settle(pre, da, dev);
  const uint32_t max_len = pl.second;
  loader::KolLoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load_kolibri1(ctx, a.snapshot_dir, max_len, pl.first, da.lm_head, 0);
  }();
  runtime::PipelineOptions opt;
  runtime::kolibri::KolibriEngine engine(ctx, std::move(model), max_len, /*debug_tap=*/false, opt);
  // Kolibri DSpark: the verify lists (their buffers; the lists on first use) and the drafter, before the adapter: snapshots
  // carry its ring.
  cli::kolibri::KolibriDrafter drafter;
  if (a.spec.on) {
    engine.enable_spec();
    StdoutToStderr redirect;
    drafter = cli::kolibri::attach_drafter(engine, a.spec);
  }
  engine.prepare_prefill();   // every card's prefill scratch and the binaries' check at load, not in a request
  std::print(stderr, "{}\n", engine.memory_line().c_str());
  if (a.spec.on) {   // the server's adaptive loop over the derived cost table (AdaptiveK, K to --spec-max)
    options.mtp_auto = true;
    options.mtp_adaptive.max_k = a.spec.max_k;
    options.mtp_adaptive.cost = server::MtpCost::kolibri_dspark(da.lm_head == loader::LmHeadForm::Int8);
    std::print(stderr, "spec dflash: DSpark {} (bf16), --spec-max {:d}, drafter key {:016x}, {:d} B + the embedding replica "
               "{:d} B on device {:d} - NOT YET RUN ON THE CARDS (docs/18-running-models.md)\n", a.spec.draft_model.c_str(),
               a.spec.max_k, (unsigned long long)engine.drafter_key(), drafter.drafter->bytes(), drafter.replica->size(),
               engine.devices() - 1);
  }
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  if (tokenizer.vocab_used() != engine.vocab())
    std::print(stderr, "note: tokenizer.json defines {:d} ids, Kolibri-1's head has {:d} rows; sampling masks from "
                       "{:d}, and a greedy argmax on a row without a token takes the masked row's argmax\n",
               tokenizer.vocab_used(), engine.vocab(), tokenizer.vocab_used());
  TemplateAdapter chat_template(a.snapshot_dir);
  cli::kolibri::KolibriEngineAdapterT<runtime::kolibri::KolibriEngine> engine_adapter(
      engine, tokenizer.vocab_used(), a.spec.on ? a.spec.max_k : 0,
      drafter.drafter ? cli::kolibri::draft_hooks(engine, *drafter.drafter) : cli::kolibri::KolibriDraftHooks{});
  options.eos_ids = eos;
  options.sampling_defaults = defaults;
  // Pinned host memory of the one context both cards share: either device's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(*c0, a.prefix_auto, a.prefix_cache_gb, options);
  options.parallel_desc = "pp - the layers split over 2 cards (Kolibri-1 needs both)";
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  const std::string placement =
      devs == 2 ? std::string("--parallel pp (split ") +
                      std::to_string(engine.split()) + ": device 0 layers [0, " + std::to_string(engine.split()) +
                      "), device 1 [" + std::to_string(engine.split()) + ", " +
                      std::to_string(engine.model().desc.layers) + "))"
                : std::string("--parallel 1");
  std::print(stderr, "b70-serve: {} on http://{}:{:d}, max_len {:d}, eos ", options.served_model.c_str(),
             options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::print(stderr,
             ", Kolibri-1 ({} chat format), {}, prefill backend l0 (chunk {:d}), attention {} (B70_KOLIBRI_ATTN), "
             "spec {}, lm_head {}, kv cache bf16, sampling defaults {} T {:.2f} top-p {:.2f} top-k {:d}, prefix "
             "snapshots: the rings' last {:d} positions + the full layers' KV\n",
             options.chat_format.name(), placement.c_str(), runtime::kolibri::kPfC,
             runtime::kolibri::kol_attn_name(engine.attention()), a.spec.on ? "dflash (DSpark)" : "none",
             loader::lm_head_form_name(da.lm_head),
             defaults.greedy ? "greedy" : "sampled", defaults.temperature, defaults.top_p, defaults.top_k,
             runtime::kolibri::state_positions(engine.model().desc));
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
    std::print(stderr, "device: {} ({:d} EUs)\n", c0->name().c_str(), c0->eu_count());
  else
    std::print(stderr, "devices: 0 {} ({:d} EUs), 1 {} ({:d} EUs) [--parallel pp: one context]\n", c0->name().c_str(),
               c0->eu_count(), c1->name().c_str(), c1->eu_count());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), "
                             "and the hand-off writes it (docs/10-the-box.md)");
  const cq::Settled s = cq::settle(full, da, dev, a.snapshot_dir);
  loader::Q4LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load_qwen4exp(ctx, a.snapshot_dir, s.max_len, s.placement, da.lm_head,
                                 s.desc.layers == full.layers ? 0 : s.desc.layers, a.loads_head(),
                                 s.tier ? &*s.tier : nullptr);
  }();
  if (model.desc.layers < model.checkpoint_layers)
    std::print(stderr, "layers: {:d} of the checkpoint's {:d} (--layers: development mode)\n", model.desc.layers,
               model.checkpoint_layers);
  runtime::PipelineOptions opt;
  runtime::qwen4exp::Qwen4ExpEngine engine(ctx, std::move(model), s.max_len, /*debug_tap=*/false, opt);
  engine.prepare_prefill();   // every card's prefill scratch and the binaries' check at load, not in a request
  std::print(stderr, "{}\n", engine.memory_line().c_str());
  TokAdapter tokenizer(tok_path);
  if (tokenizer.vocab_used() != engine.model().desc.vocab_used)
    std::print(stderr, "note: {} defines {:d} ids, the greedy argmax masks from {:d}; sampling masks from {:d}\n",
               tok_path.c_str(), tokenizer.vocab_used(), engine.model().desc.vocab_used, tokenizer.vocab_used());
  TemplateAdapter chat_template(a.snapshot_dir);
  cli::pp::PipelineEngineAdapterT<runtime::qwen4exp::Qwen4ExpEngine> engine_adapter(
      engine, tokenizer.vocab_used(), engine.vocab(), a.mtp_k, a.spec_lookup);
  cq::chat_options(options, eos);
  // Pinned host memory of the one context both cards share: either device's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc = make_prefix_alloc(*c0, a.prefix_auto, a.prefix_cache_gb, options);
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  const std::string placement =
      devs == 2 ? std::string("--parallel pp (split ") +
                      std::to_string(engine.split()) + ")"
                : std::string("--parallel 1");
  std::print(stderr, "b70-serve: {} on http://{}:{:d}, max_len {:d}, eos ", options.served_model.c_str(),
             options.host.c_str(), options.port, s.max_len);
  print_eos(eos);
  std::print(stderr,
             ", Qwen3.8-Flash-Next {} ({} chat format), {}, prefill backend l0 (chunk {:d}), attention {} "
             "(B70_Q4_ATTN), mtp {:d}{}, lm_head {}, kv cache bf16, sampling defaults: greedy unless the request "
             "samples, prefix snapshots: state {:d} B (GDN, PLE, indexer tails{}), blocks {:d} B a "
             "position\n",
             engine.model().desc.layers < engine.model().checkpoint_layers
                 ? "--layers " + std::to_string(engine.model().desc.layers)
                 : std::string("whole"),
             options.chat_format.name(), placement.c_str(), runtime::qwen4exp::kPfC,
             runtime::qwen4exp::q4_attn_name(engine.attention()), a.mtp_k,
             engine.mtp() ? (std::string(" (the head: ") + runtime::qwen4exp::mtp_norm_name(engine.mtp_norm()) +
                             " norm, reuse selection" +
                             (a.spec_lookup ? ", --spec lookup" : "") + ")").c_str()
                          : "",
             loader::lm_head_form_name(da.lm_head), engine.state_bytes(), engine.mtp() ? ", the head's" : "",
             engine.kv_bytes(4) / 4);
  return listen_until_stopped(server, options);
}

// The tail of a speculative policy's startup line: its AdaptiveK cost table, in plain steps.
void print_cost_table(const server::MtpCost& c) {
  std::print(stderr, "cost verify M=1..{:d}", c.verify.size());
  for (double v : c.verify) std::print(stderr, " {:.3f}", v);
  std::print(stderr, ", draft k=1..{:d}", c.draft.size());
  for (double v : c.draft) std::print(stderr, " {:.3f}", v);
  std::print(stderr, " (plain steps)\n");
}

// The speculative proposer's startup lines (spec 19e's lookup, spec 8 §10's auto policy) -
// one card's and --parallel pp's.
void print_proposer(const server::Options& options, bool spec_lookup, uint32_t spec_min_match,
                    uint32_t spec_max, uint32_t spec_history, bool mtp_auto, bool moe_default_costs,
                    const std::string& model_name) {
  if (spec_lookup) {
    std::print(stderr, "spec lookup: min match {:d}, K <= {:d}, history {:d}, ", spec_min_match, spec_max, spec_history);
    print_cost_table(options.lookup_adaptive.cost);
  }
  if (mtp_auto) {
    std::print(stderr, "mtp auto: ");
    print_cost_table(options.mtp_adaptive.cost);
    // A MoE model's verify at M rows touches up to top_k x M distinct experts, so its costs differ from Qwen3.8's;
    // Ornith's measured table (docs/BENCHMARKS.md) was 6-10 % slower in the speed A/B than Qwen3.8's, so it keeps these.
    if (moe_default_costs)
      std::print(stderr, "mtp auto: {} runs on Qwen3.8's table (its own measured one decoded slower, "
                         "docs/BENCHMARKS.md)\n",
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
  std::print(stderr, "devices: 0 {} ({:d} EUs), 1 {} ({:d} EUs) [--parallel pp: one context]\n",
             d0.name().c_str(), d0.eu_count(), d1.name().c_str(), d1.eu_count());
  if (!d0.can_access_peer(d1))
    throw std::runtime_error(
        "--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer 0 -> 1 is false), "
        "and the hand-off writes it. Peer access needs the P2P-capable kernel and both cards "
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
  runtime::PipelineEngine engine(d0, d1, std::move(stages), settled.max_len, opt, a.kv_cache, drafter.get());
  if (drafter)
    drafter->attach_target(d1, engine.model(1), engine.buffers(1).control, settled.max_len, draft_vocab_on);
  if (a.have_pp_backend) engine.set_prefill_backend(a.pp_backend);
  engine.set_prefill_chunk_cap(pf.chunk_cap);   // plan 5d: the cap the plan counted
  // The two-card prefill set up at load, not in a request (each card's scratch, context and
  // l0-int8 column scales; with the head, its prefill rows on device 1).
  engine.prepare_prefill();
  for (uint32_t i = 0; i < runtime::kPpDevices; ++i)
    std::print(stderr, "{}\n", engine.memory_line(i).c_str());
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  const model::ModelDesc& served = *engine.model(0).desc;
  if (tokenizer.vocab_used() != served.vocab_used)
    std::print(stderr,
               "note: tokenizer.json defines {:d} ids, {}'s greedy argmax masks from {:d}; sampling "
               "masks from {:d}\n",
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
  std::print(stderr, "b70-serve: {} on http://{}:{:d}, max_len {:d}, eos ", options.served_model.c_str(),
             options.host.c_str(), options.port, settled.max_len);
  print_eos(a.eos);
  std::print(stderr,
             ", --parallel pp (split {:d}: device 0 layers [0, {:d}), device 1 [{:d}, {:d})), prefill "
             "backend {}, mtp {}{:d}{}, lm_head {}{}{}, kv cache {}\n",
             engine.split(), engine.split(),
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
    std::print(stderr, "spec dflash (--parallel pp, device 1): {} drafter, K <= {:d}, ",
               loader::draft_format_name(a.dflash.format), a.dflash.max_k);
    print_cost_table(options.mtp_adaptive.cost);
  }
  return listen_until_stopped(server, options);
}

// Spec 17d: b70-serve --parallel tp - Qwen3.8 with every layer split over two B70s (runtime::TpEngine) behind spec 16d's
// cli::pp::PipelineEngineAdapterT, whose engine interface TpEngine has (MTP, prompt lookup, the host sampler over the
// gathered logits rows - operator decision D9 - and the snapshots). b70-decode --parallel tp's flow (cli/tp_decode.h):
// both cards in one Level Zero context with peer access both ways, each card's slice loaded straight onto it (with the
// MTP head whole on each card when a proposer needs it), max_len planned on both cards with the prefill scratch and
// the head, the two-card prefill prepared before the first request. The prefix cache's entries are the one-card host
// layouts (runtime/tp_gather.h), keyed like one card's, and its hook fires at the TP pair stride (tp_prefill_chunks).
// --parallel's refusals ran in run(), before the device.
struct TpServe {
  runtime::PrefillBackend backend = runtime::PrefillBackend::L0;
  std::string snapshot_dir;
  cli::MaxLenArg max_len_arg;
  size_t mem_reserve = 0;
  runtime::KvCache kv_cache = runtime::KvCache::Bf16;
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  uint32_t mtp_k = 0;
  bool mtp_auto = false, spec_lookup = false;
  uint32_t spec_max = 0;
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  uint32_t trained = 0;
  std::vector<uint32_t> eos;
  std::string auto_why;   // set when --parallel auto chose tp (D3): the reason, printed
  cli::dflash::Args dflash;   // spec 19 v3c: --spec dflash, the drafter split over both cards
  bool dflash_lookup = false;   // spec 19 v3c: --spec dflash-lookup (spec 19f's policy under TP)
};

int serve_tp(const TpServe& a, server::Options options) {
  cli::require_two_devices_for_tp(l0::Context::gpu_count());
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  std::print(stderr, "devices: 0 {} ({:d} EUs), 1 {} ({:d} EUs) [--parallel tp: one context]\n", d0.name().c_str(),
             d0.eu_count(), d1.name().c_str(), d1.eu_count());
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0))
    throw std::runtime_error(
        "--parallel tp: the cards cannot access each other's memory (zeDeviceCanAccessPeer 0 -> 1 and 1 -> 0), and "
        "every crossing writes the other card's VRAM. Peer access needs the P2P-capable kernel and both cards under "
        "one root complex (docs/10-the-box.md)");
  const bool head = a.mtp_k > 0 || a.spec_lookup;
  const model::TpCrossing crossing = runtime::kTpDefaultCrossing;
  runtime::PrefillPath path;
  path.prefill = true;   // every request prefills, with flash attention
  path.backend = a.backend;
  path.chunk_cap = runtime::PrefillScratchDims::kCDefault;
  std::array<loader::LoadedModel, 2> cards = [&] {
    StdoutToStderr redirect;
    const uint32_t len = cli::load_len(a.max_len_arg, a.trained);
    return std::array<loader::LoadedModel, 2>{
        loader::load_tp(d0, a.snapshot_dir, loader::TpSlice{0, crossing}, len, a.lm_head, head),
        loader::load_tp(d1, a.snapshot_dir, loader::TpSlice{1, crossing}, len, a.lm_head, head)};
  }();
  const model::ModelDesc& card = *cards[0].desc;
  const std::array<size_t, 2> weights{cards[0].report.total() - cards[0].report.rope_bytes,
                                      cards[1].report.total() - cards[1].report.rope_bytes};
  const std::array<size_t, 2> dev{d0.memory_bytes(), d1.memory_bytes()};
  // Spec 19 v3c: the drafter's halves beside the cards' slices, planned with them.
  std::unique_ptr<cli::dflash::TpLoaded> halves;
  if (a.dflash.on) {
    StdoutToStderr redirect;
    halves = std::make_unique<cli::dflash::TpLoaded>(cli::dflash::load_tp({&d0, &d1}, a.dflash, cards));
  }
  const runtime::DrafterPlan drafter_plan = halves ? halves->plan : runtime::DrafterPlan{};
  uint32_t len = a.max_len_arg.value;
  if (a.max_len_arg.is_auto) {
    len = runtime::tp_max_len_that_fits(card, crossing, weights, dev, a.mem_reserve, cards[0].trained_max_len,
                                        a.kv_cache, path, head, drafter_plan);
    if (len == 0)
      throw std::runtime_error("--parallel tp --max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
                               " positions fit both cards beside the weights");
    std::print(stderr,
               "max_len: auto -> {:d} (the largest multiple of {:d} that fits BOTH cards, {:.3f} and {:.3f} GB, with a "
               "{:.3f} GB reserve each; trained context {:d})\n",
               len, runtime::kMaxLenQuantum, dev[0] / 1e9, dev[1] / 1e9, a.mem_reserve / 1e9,
               cards[0].trained_max_len);
    for (uint32_t r = 0; r < 2; ++r) loader::set_max_len(r == 0 ? d0 : d1, cards[r], len);
  } else {
    for (uint32_t r = 0; r < 2; ++r) {
      const runtime::MemoryPlan p =
          runtime::tp_card_plan(card, crossing, len, weights[r], a.kv_cache, path, head, drafter_plan);
      if (p.total() + a.mem_reserve > dev[r])
        throw std::runtime_error("--parallel tp --max-len " + std::to_string(len) + " does not fit card " +
                                 std::to_string(r) + ": " + runtime::describe(p, dev[r], a.mem_reserve) +
                                 " - --max-len auto gives the largest that fits");
    }
  }
  std::unique_ptr<runtime::dflash::TpDFlashDrafter> drafter;
  if (halves)
    drafter = std::make_unique<runtime::dflash::TpDFlashDrafter>(
        std::array<l0::Context*, 2>{&d0, &d1}, std::move(halves->halves), len, a.dflash.max_k);
  runtime::TpEngine engine(d0, d1, std::move(cards), len, {}, a.kv_cache, drafter.get());
  if (drafter) drafter->attach(engine);
  engine.set_prefill_backend(a.backend);
  engine.set_prefill_chunk_cap(path.chunk_cap);
  engine.prepare_prefill();   // the two cards' prefill scratch, crossings and head rows at load, not in a request
  for (uint32_t r = 0; r < 2; ++r) std::print(stderr, "{}\n", engine.memory_line(r).c_str());
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  if (tokenizer.vocab_used() != card.vocab_used)
    std::print(stderr, "note: tokenizer.json defines {:d} ids, Qwen3.8's greedy argmax masks from {:d}; sampling masks "
                       "from {:d}\n",
               tokenizer.vocab_used(), card.vocab_used, tokenizer.vocab_used());
  TemplateAdapter chat_template(a.snapshot_dir);
  cli::pp::PipelineEngineAdapterT<runtime::TpEngine> engine_adapter(
      engine, tokenizer.vocab_used(), model::Qwen35::kVocab, a.dflash.on ? a.dflash.max_k : a.mtp_k, a.spec_lookup,
      a.dflash.on, a.dflash_lookup);
  options.eos_ids = a.eos;
  // Pinned host memory of the one context both cards share: either card's copies reach it.
  const std::unique_ptr<PinnedAlloc> prefix_alloc = make_prefix_alloc(d0, a.prefix_auto, a.prefix_cache_gb, options);
  options.parallel_desc = "tp - every layer split over 2 cards";
  server::Server server({tokenizer, chat_template, engine_adapter}, options);
  std::print(stderr, "b70-serve: {} on http://{}:{:d}, max_len {:d}, eos ", options.served_model.c_str(),
             options.host.c_str(), options.port, len);
  print_eos(a.eos);
  std::print(stderr,
             ", --parallel tp{} (crossing {}, prefill backend {}, chunk cap {:d}, crossings {}), mtp {}{:d}{}, lm_head "
             "{}, kv cache {}\n",
             a.auto_why.c_str(), model::tp_crossing_name(engine.crossing()),
             runtime::prefill_backend_name(engine.prefill_backend()), engine.prefill_chunk_cap(),
             engine.prefill_overlaps() ? "on the copy engines" : "on the compute lists",
             a.mtp_auto ? "auto, max " : "", a.mtp_k,
             engine.mtp()            ? " (the head whole on each card)"
             : engine.has_drafter() ? ", DFlash drafter split over both cards"
                                     : "",
             loader::lm_head_form_name(a.lm_head), runtime::kv_cache_name(engine.kv_cache()));
  print_proposer(options, a.spec_lookup, options.lookup_min_match, a.spec_max, options.lookup_history, a.mtp_auto,
                 false, card.name);
  if (engine.has_drafter())   // operator 2026-11-14: kept as an explicit choice, with the advice
    std::print(stderr, "note: DFlash under --parallel tp is not faster than DFlash on one card for one session; for "
                       "speed use --parallel 1 --spec dflash (one card), --parallel replica --spec dflash (a drafter "
                       "per card) or --spec mtp on two cards\n");
  return listen_until_stopped(server, options);
}

// Plan 13b (spec 13 §10 / §11): b70-serve --parallel replica - Qwen3.8 or Ornith (one request a card: its engine has
// no slots) on both cards, one runtime::Engine each (each
// card's own plan: its --max-len auto, its prefill chunk cap, MTP / lookup per --spec), behind cli::ReplicaEngine and
// server::Scheduler: two requests at once, each bitwise what it is alone on its card. Both cards are views of ONE
// Level Zero context (spec 16 decision 1) so the prefix cache's pinned host store is device-visible to both - a
// snapshot saved by card 0 gives a prefix cache hit on card 1; nothing else crosses (no peer access is used). The refusals
// (another model, one visible card, the pp override flags, lookup inside DFlash) ran in run(), before the device.
// --spec dflash: each card loads its own drafter beside its engine, as on one card (one slot a card).
// Replica x slots (2026-11-10; --max-num-seqs, operator 2026-11-11): --max-num-seqs N splits between the cards (card 0
// the odd one); a card's slots are plan 13c's (cli::settle_slots over that card's own memory), and a card with
// several is a cli::SlotsEngine behind the replica engine - N requests at once.
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
  uint32_t max_num_seqs = 2;
  cli::dflash::Args dflash;   // --spec dflash: a drafter on each card
  bool mtp_lookup = false;    // open-work 27: lookup inside MTP on each card (one request a card; B70_SPEC_LOOKUP=0 off)
};

int serve_replica(const ReplicaServe& a, server::Options options) {
  cli::require_two_devices_for_replica(l0::Context::gpu_count());
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  std::print(stderr, "devices: 0 {} ({:d} EUs), 1 {} ({:d} EUs) [--parallel replica: one engine per card, one "
                     "context; no peer access used]\n",
             d0.name().c_str(), d0.eu_count(), d1.name().c_str(), d1.eu_count());
  std::array<l0::Context*, 2> ctx{&d0, &d1};
  std::vector<std::unique_ptr<runtime::dflash::DFlashDrafter>> drafters;   // outlive the engines, as on one card
  std::vector<std::unique_ptr<runtime::Engine>> engines;
  std::vector<uint32_t> lens, caps;
  std::vector<std::vector<uint32_t>> layouts;   // per card: its slots' lengths
  for (uint32_t i = 0; i < 2; ++i) {
    l0::Context& c = *ctx[i];
    loader::LoadedModel model = [&] {
      StdoutToStderr redirect;
      return loader::load(c, a.snapshot_dir, cli::load_len(a.max_len_arg, a.trained),
                          /*mtp=*/a.mtp_k > 0 || a.spec_lookup, a.lm_head, a.dv_spec);
    }();
    model::require_prefill(*model.desc);
    if (!cli::replica_architecture(model.desc->architecture))   // run() read config.json; the loader is the truth
      throw std::runtime_error("--parallel replica serves Qwen3.8 and Ornith only; this checkpoint loads as " +
                               model.desc->name);
    // Spec 19c on this card: the drafter beside the target, before the card's plan.
    std::unique_ptr<cli::dflash::Loaded> drafter_load;
    if (a.dflash.on) drafter_load = std::make_unique<cli::dflash::Loaded>(cli::dflash::load(c, a.dflash, model));
    const runtime::DrafterPlan drafter_plan = drafter_load ? drafter_load->plan : runtime::DrafterPlan{};
    runtime::PrefillPath pf = a.pp_path;
    pf.chunk_cap = cli::planned_chunk_cap(0, *model.desc);
    const uint32_t seqs = (a.max_num_seqs + 1 - i) / 2;   // card 0 the odd one
    std::vector<uint32_t> slot_lens;
    uint32_t max_len = 0;
    std::print(stderr, "card {:d}: ", i);
    if (seqs > 1) {
      slot_lens = cli::settle_slots(c, model, a.max_len_arg, seqs, a.mem_reserve, pf, a.kv_cache, drafter_plan);
      max_len = runtime::slots_kv_len(slot_lens);   // one RoPE table, one KV allocation of the slots' pages
    } else {
      max_len = cli::settle(c, model, a.max_len_arg, a.mem_reserve, pf, a.kv_cache, drafter_plan);
      slot_lens = {max_len};
    }
    const uint32_t chunk_cap =
        cli::settle_chunk_cap(c, model, slot_lens, a.mem_reserve, pf, a.kv_cache, drafter_plan, 0);
    std::print(stderr, "card {:d}: slots: {:d} ({})\n", i, slot_lens.size(),
               runtime::slot_layout_name(slot_lens).c_str());
    std::unique_ptr<runtime::dflash::DFlashDrafter> drafter;
    if (drafter_load)
      drafter = std::make_unique<runtime::dflash::DFlashDrafter>(c, std::move(drafter_load->w), drafter_load->head_rows,
                                                                 max_len, a.dflash.max_k);
    const bool draft_vocab_on = model.draft_vocab != nullptr;
    auto engine = std::make_unique<runtime::Engine>(c, std::move(model), max_len, /*debug_resid=*/false, a.kv_cache,
                                                    drafter.get(),
                                                    slot_lens.size() > 1 ? slot_lens : std::vector<uint32_t>{});
    if (drafter) drafter->attach(*engine, draft_vocab_on);
    if (a.have_pp_backend) engine->set_prefill_backend(a.pp_backend);
    engine->set_prefill_chunk_cap(chunk_cap);   // plan 5d
    engine->prepare_prefill();
    std::print(stderr, "card {:d}: max_len {:d}, prefill chunk {:d}, {}\n", i, max_len, chunk_cap,
               engine->memory_line().c_str());
    lens.push_back(slot_lens[0]);
    caps.push_back(chunk_cap);
    layouts.push_back(slot_lens);
    drafters.push_back(std::move(drafter));
    engines.push_back(std::move(engine));
  }
  TokAdapter tokenizer(a.snapshot_dir + "tokenizer.json");
  TemplateAdapter chat_template(a.snapshot_dir);
  options.eos_ids = a.eos;
  // The prefix store: one pinned allocation in the shared context, sized after both loads.
  const std::unique_ptr<PinnedAlloc> prefix_alloc = make_prefix_alloc(d0, a.prefix_auto, a.prefix_cache_gb, options);
  // The prompt's cut is the serial server's: hooked (every chunk ends at a stride end) when the cache is on. A card
  // with one slot is today's EngineAdapter; one with several, 13c's SlotsEngine (an EngineAdapter per slot).
  const bool hooked = options.prefix_cache_bytes != 0;
  std::vector<std::unique_ptr<EngineAdapter>> adapters;
  std::vector<std::unique_ptr<cli::SlotsEngine>> slots_engines;
  std::vector<cli::ReplicaEngine::Card> cards;
  uint32_t requests = 0;
  for (size_t i = 0; i < engines.size(); ++i) {
    runtime::Engine& e = *engines[i];
    if (e.slots() > 1) {
      slots_engines.push_back(
          std::make_unique<cli::SlotsEngine>(e, tokenizer.vocab_used(), a.dflash.on ? a.dflash.max_k : a.mtp_k,
                                             a.spec_lookup, hooked, runtime::Engine::kMaxRows, a.dflash.on));
      cards.push_back({nullptr, caps[i], slots_engines.back().get(), e.prefill_tail()});
    } else {
      adapters.push_back(std::make_unique<EngineAdapter>(e, tokenizer.vocab_used(),
                                                         a.dflash.on ? a.dflash.max_k : a.mtp_k, a.spec_lookup,
                                                         a.dflash.on));
      cards.push_back({adapters.back().get(), caps[i], nullptr, e.prefill_tail()});
    }
    requests += e.slots();
  }
  cli::ReplicaEngine replica(cards, hooked, /*wide_stride=*/true);
  options.spec_mtp_lookup = a.mtp_lookup && slots_engines.empty();   // open-work 27: one request a card, as one card
  // The serial server's view (startup line, refusals): card 0's engine - its slot 0's adapter when it has slots.
  server::EngineIface& first = cards[0].engine != nullptr ? *cards[0].engine : slots_engines[0]->adapter(0);
  options.parallel_desc = "replica - one copy of the model per card, " + std::to_string(requests) + " requests at once";
  if (!slots_engines.empty())
    options.parallel_desc += " (card 0 " + runtime::slot_layout_name(layouts[0]) + ", card 1 " +
                             runtime::slot_layout_name(layouts[1]) + ")";
  server::Server server({tokenizer, chat_template, first}, options, &replica);
  std::print(stderr, "b70-serve: {} on http://{}:{:d}, --parallel replica: {:d} requests at once, max_len {:d} / {:d}, eos ",
             options.served_model.c_str(), options.host.c_str(), options.port, requests, lens[0], lens[1]);
  print_eos(a.eos);
  std::print(stderr, ", prefill backend {}, mtp {}{:d}, lm_head {}, kv cache {}\n",
             runtime::prefill_backend_name(engines[0]->prefill_backend()), a.mtp_auto ? "auto, max " : "", a.mtp_k,
             loader::lm_head_form_name(a.lm_head), runtime::kv_cache_name(engines[0]->kv_cache()));
  print_proposer(options, a.spec_lookup, options.lookup_min_match, a.spec_max, options.lookup_history, a.mtp_auto,
                 engines[0]->model().desc->is_moe(), engines[0]->model().desc->name);
  if (options.spec_mtp_lookup)   // open-work 27 / 28 on each card, one card's lines
    std::print(stderr,
               "spec mtp: prompt lookup inside MTP on each card - ENTER a match of >= {:d} ids, STAY while a verify "
               "keeps >= {:d} (and a match of >= {:d} remains), K {:d} lookup drafts; long runs in one {:d}-row verify on "
               "card 0 {}, card 1 {}\n",
               server::LookupInDFlash::kEnterMatch, server::LookupInDFlash::kStayAccepted,
               server::LookupInDFlash::kStayMatch, a.mtp_k, runtime::Engine::kWideK + 1,
               engines[0]->wide_built() ? "yes" : "no", engines[1]->wide_built() ? "yes" : "no");
  else if (a.mtp_k > 0 && slots_engines.empty())
    std::print(stderr, "spec mtp: prompt lookup inside MTP off (B70_SPEC_LOOKUP=0)\n");
  for (uint32_t i = 0; a.dflash.on && i < 2; ++i) {   // spec 19c: each card's drafter policy line, as on one card
    std::print(stderr, "card {:d}: spec dflash: {} drafter, K <= {:d}, ", i, loader::draft_format_name(a.dflash.format),
               a.dflash.max_k);
    print_cost_table(options.mtp_adaptive.cost);
  }
  return listen_until_stopped(server, options);
}

int run(int argc, char** argv) {
  std::string path;
  Flags f;
  if (!cli::flags::parse(argc, argv, f, path, usage)) {
    usage();
    return 0;
  }
  server::Options options;
  options.host = f.host;
  if (f.port > 65535) throw std::runtime_error("--port is out of range");
  options.port = static_cast<int>(f.port);
  options.log_requests_dir = f.log_requests;
  options.prefix_split_last = f.prefix_split_last;
  cli::MaxLenArg max_len_arg = cli::parse_max_len(f.max_len);
  runtime::KvCache kv_cache = f.kv_cache.empty() ? runtime::default_kv_cache() : cli::parse_kv_cache_arg(f.kv_cache);
  // The zero-config cleanup (operator 2026-11-06): the engine decides these - the device memory the plan
  // leaves free (runtime::kDefaultReserveGb), the int8 lm_head (spec 9, every served model), the request
  // queue (server::Options::queue_depth) and the served name (cli::served_name, from the model).
  const size_t mem_reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  const std::string pp_backend_arg = f.prefill_backend.value;
  const bool have_pp_backend = f.prefill_backend.given;
  // Spec 7: system RAM, not VRAM; auto by default (cli/prefix_cache_size.h).
  const bool prefix_auto = f.prefix_cache_gb == "auto";
  const uint32_t prefix_cache_gb = prefix_auto ? 0 : cli::flags::parse_u32("--prefix-cache-gb", f.prefix_cache_gb);
  std::string spec_arg = cli::parse_spec(f.spec);
  uint32_t spec_max = f.spec_max.value;
  const bool have_spec_max = f.spec_max.given;
  cli::dflash::Args dflash;   // spec 19c: --spec dflash's drafter (--draft-model)
  dflash.model = f.draft_model.value;
  dflash.have_model = f.draft_model.given;
  uint32_t draft_vocab = 0;
  if (!loader::parse_draft_vocab(f.draft_vocab, draft_vocab))
    throw std::runtime_error("--draft-vocab expects off, 32k, 64k or 128k, got '" + f.draft_vocab + "'");
  const std::string draft_vocab_ids = f.draft_vocab_ids;
  cli::PipelineArgs pipe;
  cli::parse_parallel(f.parallel, pipe);
  if (f.pipeline_split.given) cli::parse_pipeline_split(f.pipeline_split.value, pipe);
  if (f.max_num_seqs.value == 0) throw std::runtime_error("--max-num-seqs 0 would serve no request");
  uint32_t max_num_seqs = f.max_num_seqs.value;
  const bool have_max_num_seqs = f.max_num_seqs.given;
  // Spec 21e: Qwen3.8-Flash-Next only - its development mode (--layers N|auto until spec 22).
  const bool have_q4_layers = f.layers.given, q4_layers_auto = have_q4_layers && f.layers.value == "auto";
  const uint32_t q4_layers = have_q4_layers && !q4_layers_auto ? cli::flags::parse_u32("--layers", f.layers.value) : 0;
  if (have_q4_layers && !q4_layers_auto && q4_layers == 0) throw std::runtime_error("--layers 0 would load no layer");
  if (options.host.empty()) throw std::runtime_error("--host must not be empty");
  options.served_model = cli::served_name(path);
  cli::resolve_overrides(pipe);   // --pipeline-split under --parallel auto: pp
  if (pipe.mode == cli::Parallel::One && pipe.have_split)   // before the model is even resolved
    throw std::runtime_error("--pipeline-split belongs to --parallel pp (the layer device 1 starts at)");
  max_num_seqs = cli::check_max_num_seqs(pipe, max_num_seqs, have_max_num_seqs);
  // Plan 13b / spec 17d / --max-num-seqs above 1: Qwen3.8 (replica: and Ornith) only, by name, before any device or
  // weight file.
  std::string architecture, name;
  if (pipe.replica() || pipe.tp() || max_num_seqs > 1) {
    if (pipe.replica()) cli::check_replica_flags(pipe, /*dflash_lookup=*/spec_arg == "dflash-lookup");
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
    if (pipe.replica())
      cli::check_replica_model(architecture, name);
    else if (pipe.tp())
      cli::check_tp_model(architecture, name);
    else if (!cli::replica_architecture(architecture))
      throw std::runtime_error("--max-num-seqs above 1 is built for Qwen3.8 (its sequence slots) and "
                               "Ornith (2: one request a card, replica); " +
                               (name.empty() ? std::string("this checkpoint (no readable config.json)") : name) +
                               " serves one request at a time - drop --max-num-seqs");
    cli::check_one_request_a_card(pipe, max_num_seqs, architecture, name);   // Ornith: no slots
  }
  const bool pp_given = pipe.given();
  if (have_q4_layers && !cli::qwen4exp::is_qwen4exp(path))
    throw std::runtime_error("--layers belongs to Qwen3.8-Flash-Next (model_type qwen4_exp); this checkpoint "
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
  if (pipe.mode == cli::Parallel::Auto && max_num_seqs > 1)
    std::print(stderr, "parallel: auto -> {} (--max-num-seqs {:d})\n",
               hw.cards >= 2 ? "replica (two B70s: each card its slots)" : "1 (one card: its slots)", max_num_seqs);
  cli::resolve_max_num_seqs(pipe, max_num_seqs, hw.cards);
  cli::check_one_request_a_card(pipe, max_num_seqs, architecture, name);   // Ornith's 2 on one card: auto -> 1
  if (pipe.replica()) cli::require_two_devices_for_replica(hw.cards);   // plan 13b: one engine per card

  // The model family decides what `--spec auto` is and which proposers exist (cli/spec_args.h).
  const bool is_q4 = cli::qwen4exp::is_qwen4exp(path);
  const bool is_kol = !is_q4 && cli::kolibri::is_kolibri(path);
  // A path that does not resolve is the Qwen family here: every flag refusal below keeps its order, and the
  // snapshot's own error comes after them.
  std::string model_type, q_snap, q_arch;
  if (!is_q4 && !is_kol) {
    try {
      q_snap = loader::resolve_snapshot(path);
    } catch (const std::exception&) {
    }
    if (std::ifstream cf(q_snap + "config.json"); !q_snap.empty() && cf) {
      std::stringstream cs;
      cs << cf.rdbuf();
      const nlohmann::json cj = nlohmann::json::parse(cs.str(), nullptr, /*allow_exceptions=*/false);
      if (cj.is_object()) model_type = cj.value("model_type", std::string());
      if (cj.is_object() && cj.contains("architectures") && cj.at("architectures").is_array() &&
          !cj.at("architectures").empty() && cj.at("architectures").at(0).is_string())
        q_arch = cj.at("architectures").at(0).get<std::string>();
    }
  }
  const cli::SpecFamily family = is_q4 ? cli::SpecFamily::Qwen4Exp
                                 : is_kol ? cli::SpecFamily::Kolibri
                                         : cli::SpecFamily::Qwen;
  const std::string model_name = is_q4 ? "Qwen3.8-Flash-Next" : is_kol ? "Kolibri-1" : "this model";
  // --spec auto = the MTP head (operator 2026-11-14: no extra VRAM by default; DFlash only on an explicit --spec dflash).
  const cli::Spec spec = cli::resolve_spec(spec_arg, family, model_name, have_spec_max, spec_max, dflash.have_model);
  const bool mtp_on = spec.kind == cli::SpecKind::Mtp;
  const bool spec_lookup = spec.kind == cli::SpecKind::Lookup;
  const bool dflash_lookup = spec.kind == cli::SpecKind::DFlashLookup;
  dflash.on = spec.kind == cli::SpecKind::DFlash || dflash_lookup;   // spec 19c / 19f
  dflash.max_k = dflash.on ? spec.max_k : dflash.max_k;
  // Operator ruling 2026-11-14: prompt lookup is folded into --spec dflash where lookup inside DFlash is built (one card,
  // one slot - the engine below; --parallel tp - serve_tp); B70_SPEC_LOOKUP=0 keeps plain DFlash there (A/B runs).
  // --spec dflash-lookup asks it explicitly and is refused where it is not built. Open-work 27: the same inside MTP
  // (--spec auto / mtp, one card, one slot; --parallel replica, one request a card), B70_SPEC_LOOKUP=0 plain MTP.
  const bool lookup_env = [] {
    const char* env = std::getenv("B70_SPEC_LOOKUP");
    return !(env != nullptr && std::string(env) == "0");
  }();
  const bool lookup_fold = spec.kind == cli::SpecKind::DFlash && lookup_env;
  uint32_t mtp_k = mtp_on ? spec.max_k : 0;   // the head's drafts per iteration at most
  // Spec 17d: --parallel tp's refusals that need no device (cli/pipeline_args.h check_tp) - before the families'
  // dispatch, so another model under --parallel tp is refused by name.
  cli::check_tp(pipe, {false, false, is_q4 || is_kol,   // spec 19 v3c: dflash and dflash-lookup are served
                       draft_vocab != 0 || !draft_vocab_ids.empty()});
  // Kolibri DSpark: Kolibri-1's drafter by name, and no draft vocabulary (DSpark reads the whole lm_head).
  if (is_kol && dflash.on && !dflash.have_model)
    throw std::runtime_error("--spec dflash on Kolibri-1 needs --draft-model: --draft-model intuit/Kolibri-1-dspark (the "
                             "DSpark drafter, bf16, from the HF cache)");
  if (is_kol && dflash.on && (draft_vocab != 0 || !draft_vocab_ids.empty()))
    throw std::runtime_error("--draft-vocab and --draft-vocab-ids narrow an MTP head's or a DFlash drafter's vocabulary; "
                             "Kolibri-1's DSpark drafter reads the whole lm_head");
  cli::dflash::check_flags(dflash);
  // Spec 19c: bf16 KV, attention v2; spec 19d: pipeline parallel.
  cli::dflash::check_engine(dflash, pipe.on(), kv_cache, /*pipeline_served=*/true);
  cli::check_replica_flags(pipe, dflash_lookup);   // replica serves --spec dflash, not lookup inside it
  // --max-num-seqs: a card's several slots run beside no draft vocabulary and no lookup inside DFlash (not built);
  // plan 19v3b: --spec dflash serves up to kDflashSlots requests a card (above it MTP is faster).
  const uint32_t per_card = pipe.replica() ? (max_num_seqs + 1) / 2 : max_num_seqs;
  if (per_card > 1 && (dflash_lookup || draft_vocab != 0 || !draft_vocab_ids.empty()))
    throw std::runtime_error("--max-num-seqs " + std::to_string(max_num_seqs) +
                             " gives a card several slots, and --spec dflash-lookup / --draft-vocab beside slots are "
                             "not built - --spec dflash, auto, mtp or lookup, or --max-num-seqs " +
                             (pipe.replica() ? "2" : "1"));
  if (dflash.on && per_card > cli::dflash::kDflashSlots)
    throw std::runtime_error("--max-num-seqs " + std::to_string(max_num_seqs) + " gives a card " +
                             std::to_string(per_card) + " requests at once; --spec dflash serves up to " +
                             std::to_string(cli::dflash::kDflashSlots) +
                             " a card (more are faster with MTP) - use --spec mtp, or --max-num-seqs " +
                             std::to_string(pipe.replica() ? 2 * cli::dflash::kDflashSlots : cli::dflash::kDflashSlots));
  if (dflash_lookup && pipe.on())   // spec 19f: EngineAdapter's step_drafts_or_many; the pipeline adapter has none
    throw std::runtime_error("--spec dflash-lookup runs on one card; with --parallel pp use --spec dflash");
  if (spec_lookup) {
    options.spec_lookup = true;
    options.lookup_min_match = server::Options{}.lookup_min_match;   // 3 (spec 19e)
    options.lookup_history = 0;
    options.lookup_adaptive.max_k = spec.max_k;
    options.lookup_adaptive.cost = server::MtpCost::int8_head().with_free_drafts();
  }
  // Spec 8 §10: MTP chooses K per iteration from each request's own acceptance, over the int8 head's measured
  // cost table (the Qwen family) or Qwen3.8-Flash-Next's own (its verify rows read their own experts).
  // B70_MTP_ADAPTIVE=0 drafts a fixed K = --spec-max instead (A/B runs, the cost table's measurement).
  const bool adaptive_env = [] {
    const char* env = std::getenv("B70_MTP_ADAPTIVE");
    return !(env != nullptr && std::string(env) == "0");
  }();
  if (mtp_on && spec.adaptive && adaptive_env) {
    options.mtp_auto = true;
    options.mtp_adaptive.max_k = spec.max_k;
    options.mtp_adaptive.cost = is_q4 ? server::MtpCost::flash_next() : server::MtpCost::int8_head();
    // Spec 17d: tensor parallel's own table (its verify rows cross at M rows, its drafts read half the head).
    if (pipe.tp()) options.mtp_adaptive.cost = server::MtpCost::tp_int8_head();
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
  // Spec 19c: the drafter's AdaptiveK policy (the server's adaptive loop, over the drafter's cost table). Kolibri DSpark sets
  // its own (serve_kolibri).
  if (dflash.on && !is_kol) {
    cli::dflash::resolve_format(dflash);
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
    qs.mtp_k = mtp_k;       // the deepest K; the server picks each iteration's (options.mtp_adaptive)
    qs.spec_lookup = spec_lookup;
    qs.draft_vocab = draft_vocab != 0 || !draft_vocab_ids.empty();
    qa.load_mtp = qs.loads_head();
    qs.prefix_auto = prefix_auto;
    qs.prefix_cache_gb = prefix_cache_gb;
    qa.prefix_bytes = prefix_auto ? 0 : size_t(prefix_cache_gb) << 30;   // settle counts an explicit cache
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
    ka.mtp = false;         // cli::resolve_spec refused --spec mtp
    ka.kv8 = kv_cache == runtime::KvCache::Int8;
    ks.prefix_auto = prefix_auto;
    ks.prefix_cache_gb = prefix_cache_gb;
    ks.spec.on = dflash.on;   // Kolibri DSpark: --spec dflash (cli::resolve_spec, SpecFamily::Kolibri)
    ks.spec.draft_model = dflash.model;
    ks.spec.max_k = dflash.on ? spec.max_k : 7;
    return serve_kolibri(ks, options);
  }

  runtime::PrefillPath pp_path = cli::prefill_path(
      true, have_pp_backend ? pp_backend : runtime::prefill::default_prefill_backend());
  cli::warn_int8_prefill_kv8(pp_path.backend, kv_cache);

  // Spec 16d: the pipeline's refusals that need no checkpoint (cli/pipeline_serve.h) - the override flags on
  // one card and prefill profiling - before the snapshot is even resolved; the model's own follow below.
  cli::ServePipelineContext pipe_ctx;
  pipe_ctx.mtp = mtp_on || spec_lookup;
  {
    const char* profile = std::getenv("B70_PREFILL_PROFILE");
    pipe_ctx.prefill_profile = profile != nullptr && std::string(profile) == "1";
  }
  const std::string snapshot_dir = loader::resolve_snapshot(path);
  // The chat format (template variables, reasoning tags, tool-call syntax) follows config.json's model_type
  // (server/chat_format.h).
  options.chat_format = server::ChatFormat::for_model_type(model_type);
  // Spec 16d: the model's parallel refusals, before the device.
  pipe_ctx.model_type = model_type;
  cli::check_serve_pipeline(pipe, pipe_ctx);
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
  // Spec 17d (operator decision D3): --parallel auto on two B70s serves Qwen3.8 under tensor parallel (cli::tp_auto,
  // on since the twin count, cli::kTpAuto) - one request at a time, faster; --max-num-seqs above 1 chose replica
  // (or one card's slots) above, cli::resolve_max_num_seqs.
  std::string tp_why;
  if (pipe.mode == cli::Parallel::Auto &&
      cli::tp_auto(snapshot_dir, max_len_arg, hw.cards,
                   dflash.on || draft_vocab != 0 || !draft_vocab_ids.empty() ||
                       (have_pp_backend && pp_backend != runtime::PrefillBackend::L0))) {
    pipe.mode = cli::Parallel::Tp;
    tp_why = " (auto: two B70s with peer access, Qwen3.8)";
    std::print(stderr, "parallel: auto -> tp (two B70s with peer access, Qwen3.8: every layer split over both)\n");
    if (mtp_on && spec.adaptive) options.mtp_adaptive.cost = server::MtpCost::tp_int8_head();
  }
  if (pipe.tp()) {   // spec 17d: tensor parallel, serve_tp's own flow
    if (dflash.on) options.mtp_adaptive.cost = server::MtpCost::tp_dflash();   // spec 19 v3c: TP's own table
    TpServe ts;
    ts.backend = pp_path.backend;
    ts.auto_why = tp_why;
    ts.snapshot_dir = snapshot_dir;
    ts.max_len_arg = max_len_arg;
    ts.trained = trained;
    ts.mem_reserve = mem_reserve;
    ts.kv_cache = kv_cache;
    ts.lm_head = lm_head;
    ts.mtp_k = mtp_k;
    ts.mtp_auto = mtp_on && spec.adaptive;
    ts.spec_lookup = spec_lookup;
    ts.spec_max = spec.max_k;
    ts.prefix_auto = prefix_auto;
    ts.prefix_cache_gb = prefix_cache_gb;
    ts.eos = eos;
    ts.dflash = dflash;   // spec 19 v3c
    ts.dflash_lookup = dflash_lookup || lookup_fold;
    return serve_tp(ts, options);
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
    rs.max_num_seqs = max_num_seqs;
    rs.dflash = dflash;
    rs.mtp_lookup = mtp_on && lookup_env;   // open-work 27: the cards' one-slot engines; serve_replica checks
    return serve_replica(rs, options);
  }
  // --parallel auto (the zero-config cleanup): one card when the model and the asked context fit it, else
  // pipeline parallel (cli::parallel_auto, from the checkpoint's descriptor before the load).
  if (pipe.mode == cli::Parallel::Auto) {
    pipe.devices = cli::parallel_auto(snapshot_dir, max_len_arg, mtp_on || spec_lookup, kv_cache, pp_path,
                                      hw.card_bytes, mem_reserve);
    std::print(stderr, "parallel: auto -> {}\n",
               pipe.on() ? "pp (the model and the context do not fit one card)" : "1 (one card)");
    if (pipe.on()) {
      if (hw.cards < 2)
        throw std::runtime_error("--parallel auto: this model at this --max-len does not fit one card and Level Zero "
                                 "shows one - a shorter --max-len, or both cards visible (ZE_AFFINITY_MASK=0,1)");
      cli::dflash::check_engine(dflash, true, kv_cache, /*pipeline_served=*/true);
      if (dflash_lookup)
        throw std::runtime_error("--spec dflash-lookup runs on one card, and this model and context "
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
  std::print(stderr, "device: {} ({:d} EUs)\n", context.name().c_str(), context.eu_count());
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
  // Plan 5d: the prefill chunk cap - 2048, or 4096 when the plan has room (cli/settle_chunk_cap) -
  // planned with max_len and given to the engine.
  pp_path.chunk_cap = cli::planned_chunk_cap(0, *model.desc);
  const runtime::DrafterPlan drafter_plan = drafter_load ? drafter_load->plan : runtime::DrafterPlan{};
  // --max-num-seqs (operator 2026-11-11; plan 13c's slots): N > 1 slots on this card - slot 0 the largest context that
  // fits beside N - 1 slots of 32768 (or the asked --max-len), several requests at once (Qwen3.8; not beside a
  // drafter or a draft vocabulary: refused above). 1: settle()'s one slot.
  std::vector<uint32_t> slot_lens;
  uint32_t max_len = 0;
  if (max_num_seqs > 1) {
    slot_lens = cli::settle_slots(context, model, max_len_arg, max_num_seqs, mem_reserve, pp_path, kv_cache,
                                  drafter_plan);
    max_len = runtime::slots_kv_len(slot_lens);   // one RoPE table, one KV allocation of the slots' pages
  } else {
    max_len = cli::settle(context, model, max_len_arg, mem_reserve, pp_path, kv_cache, drafter_plan);
    slot_lens = {max_len};
  }
  const uint32_t chunk_cap =
      cli::settle_chunk_cap(context, model, slot_lens, mem_reserve, pp_path, kv_cache, drafter_plan, 0);
  std::print(stderr, "slots: {:d} ({}) - {}\n", slot_lens.size(), runtime::slot_layout_name(slot_lens).c_str(),
             slot_lens.size() > 1 ? "several requests at once on this card" : "one request at a time");
  std::unique_ptr<runtime::dflash::DFlashDrafter> drafter;
  if (drafter_load)
    drafter = std::make_unique<runtime::dflash::DFlashDrafter>(context, std::move(drafter_load->w),
                                                               drafter_load->head_rows, max_len, dflash.max_k);
  const bool draft_vocab_on = model.draft_vocab != nullptr;
  runtime::Engine engine(context, std::move(model), max_len, /*debug_resid=*/false, kv_cache, drafter.get(),
                         slot_lens.size() > 1 ? slot_lens : std::vector<uint32_t>{});
  if (drafter) drafter->attach(engine, draft_vocab_on);
  if (have_pp_backend) engine.set_prefill_backend(pp_backend);
  engine.set_prefill_chunk_cap(chunk_cap);   // plan 5d
  // Prefill setup at load, not in the first request: on l0-int8 this is the
  // one-time rotated column-scale pass (spec 5 T2).
  engine.prepare_prefill();
  std::print(stderr, "{}\n", engine.memory_line().c_str());   // spec 6
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
    std::print(stderr,
               "note: tokenizer.json defines {:d} ids, {}'s greedy argmax masks from {:d}; sampling "
               "masks from {:d}\n",
               tokenizer.vocab_used(), served.name.c_str(), served.vocab_used,
               tokenizer.vocab_used());
  TemplateAdapter chat_template(snapshot_dir);
  const bool lookup_inside = dflash_lookup || (lookup_fold && engine.slots() == 1);   // spec 19f, folded into dflash
  options.spec_dflash_lookup = lookup_inside;
  options.spec_mtp_lookup = mtp_on && lookup_env && engine.slots() == 1;   // open-work 27
  EngineAdapter engine_adapter(engine, tokenizer.vocab_used(), dflash.on ? dflash.max_k : mtp_k, spec_lookup,
                               dflash.on, lookup_inside);
  options.eos_ids = eos;
  // Sized after the model is loaded, so "available" already excludes the process's own load.
  const std::unique_ptr<PinnedAlloc> prefix_alloc =
      make_prefix_alloc(context, prefix_auto, prefix_cache_gb, options);
  options.parallel_desc = "1 card";
  // Plan 13c: several slots - the scheduler over cli::SlotsEngine (the serial server's EngineAdapter per slot, the
  // row lists for a step of several requests).
  std::unique_ptr<cli::SlotsEngine> slots_engine;
  if (engine.slots() > 1) {
    slots_engine = std::make_unique<cli::SlotsEngine>(engine, tokenizer.vocab_used(), dflash.on ? dflash.max_k : mtp_k,
                                                      spec_lookup, /*hooked=*/options.prefix_cache_bytes != 0,
                                                      runtime::Engine::kMaxRows, dflash.on);
    options.parallel_desc = "1 card, " + std::to_string(engine.slots()) + " sequence slots (" +
                            runtime::slot_layout_name(slot_lens) + ")";
  }
  server::Server server({tokenizer, chat_template, slots_engine ? slots_engine->adapter(0) : engine_adapter}, options,
                        slots_engine.get());

  std::print(stderr, "b70-serve: {} on http://{}:{:d}, max_len {:d}, eos ", options.served_model.c_str(),
             options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::print(stderr, ", prefill backend {}, mtp {}{:d}, lm_head {}{}{}, kv cache {}\n",
             runtime::prefill_backend_name(engine.prefill_backend()), mtp_on && spec.adaptive ? "auto, max " : "",
             mtp_k, loader::lm_head_form_name(lm_head),
             engine.draft_vocab() ? ", draft vocab " : "",
             engine.draft_vocab() ? loader::draft_vocab_name(engine.draft_vocab()).c_str() : "",
             runtime::kv_cache_name(engine.kv_cache()));
  print_proposer(options, spec_lookup, options.lookup_min_match, spec.max_k, options.lookup_history,
                 mtp_on && spec.adaptive, served.is_moe(), served.name);
  if (options.spec_mtp_lookup)   // open-work 27
    std::print(stderr,
               "spec mtp: prompt lookup inside MTP - ENTER a match of >= {:d} ids, STAY while a verify keeps >= {:d} (and "
               "a match of >= {:d} remains), K {:d} lookup drafts\n",
               server::LookupInDFlash::kEnterMatch, server::LookupInDFlash::kStayAccepted,
               server::LookupInDFlash::kStayMatch, mtp_k);
  if (options.spec_mtp_lookup && engine.wide_built()) {   // open-work 28; spec 32: the 32-row list where it is built
    const uint32_t wk = engine.widest_k();
    std::print(stderr,
               "spec mtp: long lookup runs - up to {:d} drafts in one {:d}-row verify while a run keeps them{}; a match "
               "of >= {:d} ids enters when the head's first two drafts agree with it; the 32- and 64-row lists captured in {:.0f} ms\n",
               wk, wk + 1,
               wk == runtime::Engine::kDeepK64
                   ? std::format(" (64 rows up to {:d} keys, 32 rows beyond)", runtime::kDeep64MaxKeys)
                   : std::string(),
               server::LookupInDFlash::kConfirmMatch, engine.deep_capture_ms()[0] + engine.deep_capture_ms()[1]);
  }
  else if (mtp_on && engine.slots() == 1 && !options.spec_mtp_lookup)
    std::print(stderr, "spec mtp: prompt lookup inside MTP off (B70_SPEC_LOOKUP=0)\n");
  if (drafter) {   // spec 19c: the drafter's policy line (print_proposer's format)
    std::print(stderr, "spec dflash: {} drafter, K <= {:d}, ", loader::draft_format_name(dflash.format), dflash.max_k);
    print_cost_table(options.mtp_adaptive.cost);
    if (lookup_inside)   // spec 19f
      std::print(stderr,
                 "spec dflash-lookup: prompt lookup inside DFlash - ENTER a match of >= {:d} ids beyond the drafter's "
                 "window {:d}, STAY while a verify keeps >= {:d} (and a match of >= {:d} remains), K {:d} lookup drafts\n",
                 server::LookupInDFlash::kEnterMatch, drafter->window(), server::LookupInDFlash::kStayAccepted,
                 server::LookupInDFlash::kStayMatch, dflash.max_k);
    else if (spec.kind == cli::SpecKind::DFlash)
      std::print(stderr, "spec dflash: prompt lookup inside DFlash off (B70_SPEC_LOOKUP=0)\n");
  }

  return listen_until_stopped(server, options);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::print(stderr, "b70-serve: {}\n", error.what());
    return 1;
  }
}

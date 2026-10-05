// b70-serve -- the OpenAI-compatible server over the replayed runtime::Engine.
#include <chrono>
#include <filesystem>
#include <memory>
#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "cli/max_len.h"
#include "cli/prefix_cache_size.h"
#include "cli/serve_adapters.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"
#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
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
               "                 The model is picked from the checkpoint's config.json: Qwen3.8 or\n"
               "                 Agnes 3.0 Flash (spec 14).\n"
               "                 [--pp-backend sycl-tla|l0|l0-int8]   Default: l0-int8.\n"
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
               "                             head (+0.85 GB weights, + its KV and state slots);\n"
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
               "                             (tools/draft_vocab/rank.py over --log-requests logs)\n");
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

int run(int argc, char** argv) {
  std::string path;
  server::Options options;
  cli::MaxLenArg max_len_arg;   // spec 6 §10: auto unless --max-len N
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
  // Spec 9 §3: the serving default is the gated int8 head (amendment §8, L3).
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  // Spec 8 §11: off until the box rows decide (§11 "Gates").
  uint32_t draft_vocab = 0;
  std::string draft_vocab_ids;

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
    } else if (arg == "--pp-backend") {
      pp_backend_arg = value(i, "--pp-backend");
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
    throw std::runtime_error("--pp-backend expects sycl-tla, l0 or l0-int8, got '" + pp_backend_arg + "'");

  const std::string snapshot_dir = loader::resolve_snapshot(path);
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
  l0::Context context(device);
  std::fprintf(stderr, "device: %s (%u EUs)%s\n", context.name().c_str(), context.eu_count(),
               device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  loader::LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load(context, snapshot_dir, cli::load_len(max_len_arg, trained),
                        /*mtp=*/mtp_k > 0, lm_head, dv_spec);
  }();
  // Spec 15c: the server prefills every request; a model whose prefill is not built
  // (Ornith: spec 15d, its serving 15e) is refused here by name.
  model::require_prefill(*model.desc);
  // Spec 6 §10: auto plans the largest max_len that fits and re-tables the model; an
  // explicit N is held to the same plan. Both print the plan's breakdown.
  const uint32_t max_len = cli::settle(
      context, model, max_len_arg, mem_reserve,
      cli::prefill_path(true, have_pp_backend ? pp_backend : runtime::prefill::default_prefill_backend()));
  runtime::Engine engine(context, std::move(model), max_len);
  if (have_pp_backend) engine.set_prefill_backend(pp_backend);
  // Prefill setup at load, not in the first request: on l0-int8 this is the
  // one-time rotated column-scale pass (spec 5 T2).
  engine.prepare_prefill();
  std::fprintf(stderr, "%s\n", engine.memory_line().c_str());   // spec 6
  TokAdapter tokenizer(snapshot_dir + "tokenizer.json");
  TemplateAdapter chat_template(snapshot_dir);
  EngineAdapter engine_adapter(engine, tokenizer.vocab_used(), mtp_k);
  options.eos_ids = eos;
  std::unique_ptr<PinnedAlloc> prefix_alloc;
  // Sized after the model is loaded, so "available" already excludes the process's own load.
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
  server::Server server({tokenizer, chat_template, engine_adapter}, options);

  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::fprintf(stderr, ", prefill backend %s (SYCL component %s), mtp %s%u, lm_head %s%s%s\n",
               runtime::prefill_backend_name(engine.prefill_backend()),
               runtime::prefill::sycl_available() ? "on" : "off", mtp_auto ? "auto, max " : "",
               mtp_k, loader::lm_head_form_name(lm_head),
               engine.draft_vocab() ? ", draft vocab " : "",
               engine.draft_vocab() ? loader::draft_vocab_name(engine.draft_vocab()).c_str() : "");
  if (mtp_auto) {
    const server::MtpCost& c = options.mtp_adaptive.cost;
    std::fprintf(stderr, "mtp auto: cost verify M=1..%zu", c.verify.size());
    for (double v : c.verify) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, ", draft k=1..%zu", c.draft.size());
    for (double v : c.draft) std::fprintf(stderr, " %.3f", v);
    std::fprintf(stderr, " (plain steps)\n");
  }

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

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "b70-serve: %s\n", error.what());
    return 1;
  }
}

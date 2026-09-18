// b70-serve -- the OpenAI-compatible server over the replayed runtime::Engine.
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

#include "cli/serve_adapters.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
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
               "                 [--max-len 16384] [--device N] [--served-name NAME] [--queue 4]\n"
               "                 [--pp-backend sycl-tla|l0]   Default: l0.\n");
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
  uint32_t max_len = 16384;
  uint32_t device = l0::Context::kFromEnv;
  std::string pp_backend_arg;
  bool have_pp_backend = false;

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
      max_len = parse_u32("--max-len", value(i, "--max-len"));
    } else if (arg == "--device") {
      device = parse_u32("--device", value(i, "--device"));
      if (device == l0::Context::kFromEnv) throw std::runtime_error("--device is out of range");
    } else if (arg == "--served-name") {
      options.served_model = value(i, "--served-name");
    } else if (arg == "--queue") {
      options.queue_depth = parse_u32("--queue", value(i, "--queue"));
    } else if (arg == "--pp-backend") {
      pp_backend_arg = value(i, "--pp-backend");
      have_pp_backend = true;
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
  if (max_len == 0) throw std::runtime_error("--max-len 0 is not a model length");
  if (options.host.empty()) throw std::runtime_error("--host must not be empty");
  if (options.served_model.empty()) throw std::runtime_error("--served-name must not be empty");
  runtime::PrefillBackend pp_backend{};
  if (have_pp_backend && !runtime::parse_prefill_backend(pp_backend_arg, pp_backend))
    throw std::runtime_error("--pp-backend expects sycl-tla or l0, got '" + pp_backend_arg + "'");

  const std::string snapshot_dir = loader::resolve_snapshot(path);
  const std::vector<uint32_t> eos = eos_ids(snapshot_dir);
  l0::Context context(device);
  std::fprintf(stderr, "device: %s (%u EUs)%s\n", context.name().c_str(), context.eu_count(),
               device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");
  loader::LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load(context, snapshot_dir, max_len);
  }();
  runtime::Engine engine(context, std::move(model), max_len);
  if (have_pp_backend) engine.set_prefill_backend(pp_backend);
  TokAdapter tokenizer(snapshot_dir + "tokenizer.json");
  TemplateAdapter chat_template(snapshot_dir);
  EngineAdapter engine_adapter(engine, tokenizer.vocab_used());
  options.eos_ids = eos;
  server::Server server({tokenizer, chat_template, engine_adapter}, options);

  std::fprintf(stderr, "b70-serve: %s on http://%s:%d, max_len %u, eos ", options.served_model.c_str(),
               options.host.c_str(), options.port, max_len);
  print_eos(eos);
  std::fprintf(stderr, ", prefill backend %s (SYCL component %s)\n",
               runtime::prefill_backend_name(engine.prefill_backend()),
               runtime::prefill::sycl_available() ? "on" : "off");

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

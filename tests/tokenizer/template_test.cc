// Spec 3 §2 bar 2: byte-for-byte against transformers.apply_chat_template.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "check.h"
#include "tokenizer/chat_template.h"

#include <nlohmann/json.hpp>

namespace {

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

// Same resolution as tok::default_tokenizer_json: no home directory baked in.
std::string snapshot() {
  if (const char* env = std::getenv("B70_SNAPSHOT_DIR")) return env;
  std::string root;
  if (const char* hf = std::getenv("HF_HOME")) {
    root = hf;
  } else if (const char* home = std::getenv("HOME")) {
    root = std::string(home) + "/.cache/huggingface";
  } else {
    root = ".cache/huggingface";
  }
  return root + "/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d";
}

void diff_report(const char* name, const std::string& got, const std::string& want) {
  size_t index = 0;
  while (index < got.size() && index < want.size() && got[index] == want[index]) ++index;
  std::fprintf(stderr, "%s: differs at byte %zu of %zu/%zu\n  got : %s\n  want: %s\n", name,
               index, got.size(), want.size(), got.substr(index, 80).c_str(),
               want.substr(index, 80).c_str());
}

// Spec 14: `template_test <dir> <snapshot> <cases.json>` renders every case of the JSON
// file (name, messages, tools, think) through <snapshot>'s template and compares with
// <dir>/agnes_template_<name>.txt (tools/tokenizer/dump_agnes_template.py, transformers).
// Spec 15e: an optional fifth argument names the reference files' prefix (`ornith` ->
// <dir>/ornith_template_<name>.txt). A snapshot without its two template files is a
// SKIP (77), not a failure - the test is about the renderer, not about the cache.
// Spec 18d: the cases file may also be an object (k2_template_cases.json, prefix `k2`):
// "bos_token" / "eos_token" the snapshot's tokenizer_config must name, "tools" a table of tool
// objects by name, and "cases" whose "tools" lists names from it and whose "kwargs" are the
// request's chat_template_kwargs (tool_call_format, reasoning_effort, ...); "think" defaults to
// true (tools/tokenizer/dump_k2.py writes the references).
int run_cases(const std::string& dir, const std::string& snap, const std::string& cases_file,
              const std::string& prefix) {
  for (const char* f : {"/tokenizer_config.json", "/chat_template.jinja"})
    if (!std::ifstream(snap + f).good()) {
      std::printf("SKIP: %s%s is absent\n", snap.c_str(), f);
      return 77;
    }
  chat::Template tmpl(snap);
  const auto spec = nlohmann::json::parse(slurp(dir + "/" + cases_file));
  const bool table = spec.is_object();
  const auto& cases = table ? spec.at("cases") : spec;
  if (table) {
    CHECK_EQ(tmpl.bos_token(), spec.at("bos_token").get<std::string>());
    CHECK_EQ(tmpl.eos_token(), spec.at("eos_token").get<std::string>());
  } else {
    CHECK_EQ(tmpl.eos_token(), std::string("<|im_end|>"));
  }
  int bad = 0;
  for (const auto& c : cases) {
    const std::string file = prefix + "_template_" + c.at("name").get<std::string>() + ".txt";
    const std::string want = slurp(dir + "/" + file);
    nlohmann::json tools = c.at("tools");
    if (table && tools.is_array()) {
      nlohmann::json named = nlohmann::json::array();
      for (const auto& name : tools) named.push_back(spec.at("tools").at(name.get<std::string>()));
      tools = std::move(named);
    }
    const bool think = c.contains("think") ? c.at("think").get<bool>() : true;
    const nlohmann::json kwargs = c.contains("kwargs") ? c.at("kwargs") : nlohmann::json::object();
    std::string got;
    try {
      got = tmpl.render(c.at("messages"), tools, think, kwargs);
    } catch (const std::exception& error) {
      // minja appends the template location of every enclosing node: the first lines say it.
      const std::string what = error.what();
      size_t cut = 0;
      for (int line = 0; line < 4 && cut != std::string::npos; ++line) cut = what.find('\n', cut + 1);
      std::fprintf(stderr, "%s: the render threw: %s\n", file.c_str(), what.substr(0, cut).c_str());
      ++bad;
      continue;
    }
    if (got != want) {
      diff_report(file.c_str(), got, want);
      ++bad;
    } else {
      std::printf("%s: %zu bytes, identical\n", file.c_str(), got.size());
    }
  }
  CHECK_EQ(bad, 0);
  std::printf("template_test OK (%zu cases from %s)\n", cases.size(), cases_file.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  if (argc > 3) return run_cases(dir, argv[2], argv[3], argc > 4 ? argv[4] : "agnes");
  chat::Template tmpl(snapshot());
  CHECK_EQ(tmpl.eos_token(), std::string("<|im_end|>"));
  const auto messages = nlohmann::json::parse(slurp(dir + "/template_messages.json"));
  const auto tools = nlohmann::json::parse(slurp(dir + "/template_tools.json"));
  struct Case {
    const char* file;
    nlohmann::json tools;
    bool think;
  };
  const Case cases[] = {
      {"template_think_on.txt", nullptr, true},
      {"template_think_off.txt", nullptr, false},
      {"template_tools.txt", tools, false},
  };
  int bad = 0;
  for (const Case& test_case : cases) {
    const std::string want = slurp(dir + "/" + test_case.file);
    const std::string got = tmpl.render(messages, test_case.tools, test_case.think);
    if (got != want) {
      diff_report(test_case.file, got, want);
      ++bad;
    } else {
      std::printf("%s: %zu bytes, identical\n", test_case.file, got.size());
    }
  }
  CHECK_EQ(bad, 0);
  std::printf("template_test OK\n");
  return 0;
}

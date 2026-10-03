// Spec 3 §2 bar 2: byte-for-byte against transformers.apply_chat_template.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

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
int run_cases(const std::string& dir, const std::string& snap, const std::string& cases_file) {
  chat::Template tmpl(snap);
  CHECK_EQ(tmpl.eos_token(), std::string("<|im_end|>"));
  const auto cases = nlohmann::json::parse(slurp(dir + "/" + cases_file));
  int bad = 0;
  for (const auto& c : cases) {
    const std::string file = "agnes_template_" + c.at("name").get<std::string>() + ".txt";
    const std::string want = slurp(dir + "/" + file);
    const std::string got = tmpl.render(c.at("messages"), c.at("tools"), c.at("think").get<bool>());
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
  if (argc > 3) return run_cases(dir, argv[2], argv[3]);
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

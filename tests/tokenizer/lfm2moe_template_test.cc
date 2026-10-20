// Spec 24e (plan 24e Task 1): LFM2.5-24B-A2B's chat template through the engine's renderer (chat::Template, minja
// with the b70 patches) against transformers 5.19.0's apply_chat_template, byte for byte, from
// tools/tokenizer/dump_lfm2moe.py:
//   lfm2moe_template_test <tests/tokenizer> <tests/golden/prompts> <snapshot dir>
// <snapshot dir> is LiquidAI/LFM2.5-24B-A2B at 1769516a (tokenizer_config.json + chat_template.jinja; no weights are
// read); B70_LFM_SNAPSHOT_DIR replaces the third argument. No snapshot: SKIP (77).
//
// Every case of lfm2moe_template_cases.json is rendered with the case's chat_template_kwargs, plus `agentic` - the
// frozen coding session of tests/golden/prompts/lfm2moe_agentic.json (six tools, 14 messages). The request's objects
// reach the template key-sorted (nlohmann::json), and so they do on HF's side (the dump sorts them): the comparison
// grades the renderer, not key order - spec 24 section 4.6's known limitation.
//
// Also checked: the template's sha256 is the one the references were made from (a changed template must name itself,
// not surface as 31 diffs); the bos / eos tokens; and that the cases COVER what they claim to - the rendered texts hold
// a tool call, a kept and a dropped reasoning block, a CONTINUE_FINAL_MESSAGE_TAG, so a renderer that drops those
// branches cannot pass on empty ground.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "check.h"
#include "tokenizer/chat_template.h"

namespace {

using json = nlohmann::json;

bool exists(const std::string& path) { return std::ifstream(path).good(); }

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void diff_report(const std::string& name, const std::string& got, const std::string& want) {
  size_t index = 0;
  while (index < got.size() && index < want.size() && got[index] == want[index]) ++index;
  std::fprintf(stderr, "%s: differs at byte %zu of %zu/%zu\n  got : %s\n  want: %s\n", name.c_str(), index, got.size(),
               want.size(), json(got.substr(index, 100)).dump().c_str(), json(want.substr(index, 100)).dump().c_str());
}

bool contains(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: lfm2moe_template_test <tests/tokenizer> <tests/golden/prompts> <snapshot dir>\n");
    return 2;
  }
  const std::string dir = argv[1], prompts = argv[2];
  const char* env = std::getenv("B70_LFM_SNAPSHOT_DIR");
  const std::string snap = env != nullptr && *env != '\0' ? env : argv[3];
  if (!exists(snap + "/tokenizer_config.json") || !exists(snap + "/chat_template.jinja")) {
    std::printf("lfm2moe_template_test: SKIP (%s has no tokenizer_config.json / chat_template.jinja; set "
                "B70_LFM_SNAPSHOT_DIR)\n", snap.c_str());
    return 77;
  }
  const json facts = json::parse(slurp(dir + "/lfm2moe_tokenizer.json"));
  chat::Template tmpl(snap);
  CHECK_EQ(tmpl.source_sha256(), facts.at("chat_template_sha256").get<std::string>());
  const json spec = json::parse(slurp(dir + "/lfm2moe_template_cases.json"));
  CHECK_EQ(tmpl.bos_token(), spec.at("bos_token").get<std::string>());
  CHECK_EQ(tmpl.eos_token(), spec.at("eos_token").get<std::string>());

  json cases = spec.at("cases");
  json agentic = json::parse(slurp(prompts + "/lfm2moe_agentic.json"));
  cases.push_back({{"name", "agentic"}, {"messages", agentic.at("messages")}, {"tools", agentic.at("tools")},
                   {"kwargs", json::object()}, {"inline_tools", true}});

  int bad = 0;
  std::string all;
  for (const json& c : cases) {
    const std::string file = "lfm2moe_template_" + c.at("name").get<std::string>() + ".txt";
    const std::string want = slurp(dir + "/" + file);
    json tools = c.at("tools");
    if (tools.is_array() && !c.contains("inline_tools")) {
      json named = json::array();
      for (const json& n : tools) named.push_back(spec.at("tools").at(n.get<std::string>()));
      tools = named;
    }
    std::string got;
    try {
      got = tmpl.render(c.at("messages"), tools, true, c.at("kwargs"));
    } catch (const std::exception& error) {
      const std::string what = error.what();
      size_t cut = 0;
      for (int line = 0; line < 4 && cut != std::string::npos; ++line) cut = what.find('\n', cut + 1);
      std::fprintf(stderr, "%s: the render threw: %s\n", file.c_str(), what.substr(0, cut).c_str());
      ++bad;
      continue;
    }
    if (got != want) {
      diff_report(file, got, want);
      ++bad;
    } else {
      std::printf("%s: %zu bytes, identical\n", file.c_str(), got.size());
    }
    all += want;
  }
  CHECK_EQ(bad, 0);

  // The references exercise the branches the plan names.
  CHECK(contains(all, "<|tool_call_start|>[") && contains(all, ")]<|tool_call_end|>"));
  CHECK(contains(all, "<think>\nkept, with its newlines\n</think>"));
  CHECK(!contains(all, "dropped: before the last user"));
  CHECK(contains(all, "CONTINUE_FINAL_MESSAGE_TAG <|im_end|>"));
  CHECK(contains(all, "kept by preserve_thinking"));
  CHECK(contains(all, "List of tools: ["));
  CHECK(contains(all, "<|im_start|>assistant\n<think>"));
  std::printf("lfm2moe_template_test OK (%zu cases, template sha256 %s)\n", cases.size(), tmpl.source_sha256().c_str());
  return 0;
}

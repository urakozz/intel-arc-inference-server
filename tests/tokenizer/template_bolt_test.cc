// Spec 25e Task 1: Nemotron 3.5 Bolt's chat template (chat_template.jinja, sha256 6521afc7..., 12,514 B, NVIDIA's,
// byte-identical in our export) through chat::Template (minja) against transformers 5.19.0's apply_chat_template, from
// tools/tokenizer/dump_bolt.py's tests/tokenizer/bolt_templates.json:
//   template_bolt_test <tests/tokenizer> <snapshot dir> [<tests/server/bolt_template_cases.json>]
// <snapshot dir> (tokenizer_config.json + chat_template.jinja; B70_BOLT_SNAPSHOT replaces it) holds the template: it is read
// at run time, never copied into the repository (spec 25 Decision 20: vendoring is allowed, generated fixtures are the
// default). Absent: SKIP (77). Checked:
//   - every case renders byte for byte as transformers' (no system message / system / empty / list parts, tools with nested
//     parameters, enums, extra keys, `required`, no parameters, a bare function, calls with string / number / bool / null /
//     list / dict arguments in Python's spelling, tool results grouped into one user turn, reasoning before and after the last
//     user turn with truncate_history_thinking true and false, think tags inside content, enable_thinking false,
//     medium_effort, list-of-parts content with media parts, the generation prompt on and off, the frozen agentic session);
//   - Review Focus 1: the system block is ALWAYS there - with no system message every render opens with
//     "<|im_start|>system\n<|im_end|>\n" (its `system_message` is set to "", so a render without it would shift every prompt
//     by four tokens and miss every prefix-cache entry);
//   - the generation prompt: "<|im_start|>assistant\n<think>\n", or "<|im_start|>assistant\n<think></think>" with
//     enable_thinking false;
//   - spec 25a's own 14 fixtures (the optional third argument) render byte for byte too;
//   - the source's sha256 is Bolt's; --chat-template FILE (chat::Template's second constructor) renders the same file from
//     another path identically.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "check.h"
#include "tokenizer/chat_template.h"

namespace {

using Json = nlohmann::json;

bool exists(const std::string& path) { return std::ifstream(path).good(); }

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void diff_report(const std::string& name, const std::string& got, const std::string& want) {
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  std::fprintf(stderr, "%s: differs at byte %zu of %zu / %zu\n  got : %s\n  want: %s\n", name.c_str(), i, got.size(),
               want.size(), Json(got.substr(i, 120)).dump().c_str(), Json(want.substr(i, 120)).dump().c_str());
}

// The case's kwargs, with add_generation_prompt as a template variable (it overrides chat::Template's default true).
Json kwargs_of(const Json& c) {
  Json kw = c.at("kwargs");
  kw["add_generation_prompt"] = c.at("add_generation_prompt").get<bool>();
  return kw;
}

bool thinking_of(const Json& c) {
  const Json& kw = c.at("kwargs");
  return !kw.contains("enable_thinking") || kw.at("enable_thinking").get<bool>();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "usage: template_bolt_test <tests/tokenizer> <snapshot dir> [tests/server/bolt_template_cases.json]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const char* env_snap = std::getenv("B70_BOLT_SNAPSHOT");
  const std::string snapshot = env_snap != nullptr && *env_snap != '\0' ? env_snap : argv[2];
  if (!exists(snapshot + "/chat_template.jinja") || !exists(snapshot + "/tokenizer_config.json")) {
    std::printf("template_bolt_test: SKIP (no %s/chat_template.jinja; set B70_BOLT_SNAPSHOT)\n", snapshot.c_str());
    return 77;
  }
  const Json facts = Json::parse(slurp(dir + "/bolt_templates.json"));
  chat::Template tmpl(snapshot);
  CHECK_EQ(tmpl.source_sha256(), facts.at("chat_template_sha256").get<std::string>());
  CHECK_EQ(tmpl.source_sha256().substr(0, 8), std::string("6521afc7"));
  CHECK_EQ(slurp(snapshot + "/chat_template.jinja").size(), size_t(12514));
  CHECK_EQ(tmpl.bos_token(), facts.at("bos_token").get<std::string>());
  CHECK_EQ(tmpl.eos_token(), facts.at("eos_token").get<std::string>());

  int bad = 0;
  size_t identical = 0, bytes = 0, bare_prompts = 0;
  const std::string system_block = "<|im_start|>system\n";
  for (const Json& c : facts.at("cases")) {
    const std::string name = c.at("name");
    std::string got;
    try {
      got = tmpl.render(c.at("messages"), c.at("tools"), thinking_of(c), kwargs_of(c));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s: the render threw: %.300s\n", name.c_str(), e.what());
      ++bad;
      continue;
    }
    const std::string want = c.at("render");
    if (got != want) {
      diff_report(name, got, want);
      ++bad;
      continue;
    }
    ++identical;
    bytes += got.size();
    // Review Focus 1: no system message -> the empty system block still opens the prompt (the first message is not a
    // system message: the case may carry a later one, which the template renders in its place).
    const Json& m = c.at("messages");
    if (!m.empty() && m.at(0).at("role") != "system") {
      CHECK_EQ(got.compare(0, system_block.size(), system_block), 0);
      if (c.at("tools").is_null() || c.at("tools").empty()) {
        CHECK_EQ(got.compare(0, system_block.size() + 11, system_block + "<|im_end|>\n"), 0);
        ++bare_prompts;
      }
    }
    if (c.at("add_generation_prompt").get<bool>()) {
      const std::string tail = thinking_of(c) ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think></think>";
      CHECK(got.size() >= tail.size() && got.compare(got.size() - tail.size(), tail.size(), tail) == 0);
    }
  }
  std::printf("template cases: %zu of %zu byte-identical to transformers 5.19.0 (%zu bytes), %zu with no system message "
              "and no tools opening on the empty system block\n", identical, facts.at("cases").size(), bytes, bare_prompts);
  CHECK_EQ(bad, 0);
  CHECK_EQ(identical, facts.at("cases").size());
  CHECK(bare_prompts >= 20);

  // Spec 25a's own fixtures (tests/server/bolt_template_cases.json, tools/oracle/bolt_facts.py: 14 cases rendered by
  // transformers' apply_chat_template) hold here too - a second, independent corpus of the same renders.
  if (argc == 4 && exists(argv[3])) {
    const Json a25 = Json::parse(slurp(argv[3]));
    size_t same = 0, skipped = 0;
    for (const Json& c : a25.at("cases")) {
      // The fixture was rendered from Python dicts in insertion order; the server's JSON objects are key-sorted (spec 24 §4.6's
      // known limitation, shared by every model), so a case whose call arguments are not written in sorted order cannot match
      // it byte for byte - Task 1's corpus (key-sorted before rendering) holds those.
      if (c.at("name") == "a call: string / number / bool / null / list / dict arguments") {
        ++skipped;
        continue;
      }
      Json kw = c.at("kwargs");
      kw["add_generation_prompt"] = c.at("add_generation_prompt").get<bool>();
      const bool thinking = !kw.contains("enable_thinking") || kw.at("enable_thinking").get<bool>();
      const std::string got = tmpl.render(c.at("messages"), c.at("tools"), thinking, kw);
      if (got != c.at("text").get<std::string>()) {
        diff_report("25a: " + c.at("name").get<std::string>(), got, c.at("text"));
        ++bad;
      } else {
        ++same;
      }
    }
    std::printf("25a's fixtures: %zu of %zu cases byte-identical (%zu skipped: arguments in a Python dict's insertion order)\n", same,
                a25.at("cases").size(), skipped);
    CHECK_EQ(bad, 0);
  }

  // --chat-template FILE: the same source from another path renders identically.
  {
    chat::Template other(snapshot, snapshot + "/chat_template.jinja");
    CHECK_EQ(other.source_sha256(), tmpl.source_sha256());
    const Json& c = facts.at("cases").back();
    CHECK_EQ(other.render(c.at("messages"), c.at("tools"), thinking_of(c), kwargs_of(c)), c.at("render").get<std::string>());
  }
  std::printf("template_bolt_test OK (%zu cases)\n", facts.at("cases").size());
  return 0;
}

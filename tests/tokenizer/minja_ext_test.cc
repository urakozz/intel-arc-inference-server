// Spec 18d: the renderer extensions (the b70 patches to third_party/minja, listed in
// third_party/VERSIONS), each on its own small template, against Jinja2's own render.
//   minja_ext_test <tests/tokenizer>
// minja_ext_cases.json is written by tools/tokenizer/dump_minja_ext.py (Jinja2 configured as
// transformers' chat-template renderer). The last case is the chat_template capability probe:
// a template that writes argument names between tags (K2-Horizon's <ifm|arg_key>) supports
// tool calls, so its tool-call history is rendered by the template, not polyfilled as JSON.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <nlohmann/json.hpp>

#include "check.h"

namespace {

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

std::string render(const std::string& source, const nlohmann::ordered_json& context) {
  const auto root = minja::Parser::parse(source, {/*trim_blocks=*/true, /*lstrip_blocks=*/true,
                                                  /*keep_trailing_newline=*/false});
  return root->render(minja::Context::make(minja::Value(context)));
}

int run_jinja_cases(const std::string& dir) {
  const auto spec = nlohmann::ordered_json::parse(slurp(dir + "/minja_ext_cases.json"));
  int bad = 0;
  for (const auto& c : spec.at("cases")) {
    const std::string name = c.at("name");
    const std::string want = c.at("expected");
    std::string got;
    try {
      got = render(c.at("template"), spec.at("context"));
    } catch (const std::exception& error) {
      got = std::string("<threw: ") + error.what() + ">";
    }
    if (got != want) {
      std::fprintf(stderr, "%s:\n  got : %s\n  want: %s\n", name.c_str(), got.c_str(), want.c_str());
      ++bad;
    } else {
      std::printf("%s: %s\n", name.c_str(), got.c_str());
    }
  }
  CHECK(spec.at("cases").size() >= 10U);
  return bad;
}

// The probe in minja::chat_template's constructor decides supports_tool_calls; when it says
// no, apply() rewrites every tool call into a JSON content blob. A template in K2's shape
// (arguments as an object, names between tags, a string `arguments` refused) must probe as
// supporting them.
void case_tool_call_probe() {
  const std::string source =
      "{%- for m in messages -%}"
      "{{- '<' + m.role + '>' -}}"
      "{%- if m.content is string -%}{{- m.content -}}{%- endif -%}"
      "{%- for tc in (m.tool_calls or []) -%}"
      "{%- if tc.function.arguments is string -%}{{- raise_exception('arguments must be a dict') -}}{%- endif -%}"
      "{{- '<call>' + tc.function.name -}}"
      "{%- for k, v in tc.function.arguments | items -%}{{- '<key>' + k + '</key><value>' + v + '</value>' -}}{%- endfor -%}"
      "{{- '</call>' -}}"
      "{%- endfor -%}"
      "{%- endfor -%}";
  minja::chat_template tmpl(source, "", "");
  CHECK(tmpl.original_caps().supports_tool_calls);
  CHECK(tmpl.original_caps().requires_object_arguments);
  minja::chat_template_inputs inputs;
  inputs.messages = nlohmann::ordered_json::parse(R"([
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "function": {"name": "read", "arguments": {"path": "/a"}}}]}])");
  inputs.add_generation_prompt = false;
  const std::string got = tmpl.apply(inputs);
  CHECK_EQ(got, std::string("<user>go<assistant><call>read<key>path</key><value>/a</value></call>"));
  std::printf("tool_call_probe: %s\n", got.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  const int bad = run_jinja_cases(dir);
  CHECK_EQ(bad, 0);
  case_tool_call_probe();
  std::printf("minja_ext_test OK\n");
  return 0;
}

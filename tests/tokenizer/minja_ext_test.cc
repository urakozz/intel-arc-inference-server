// Spec 18d: the renderer extensions (the b70 patches to third_party/minja, listed in
// third_party/VERSIONS), each on its own small template, against Jinja2's own render.
//   minja_ext_test <tests/tokenizer>
// minja_ext_cases.json is written by tools/tokenizer/dump_minja_ext.py (Jinja2 configured as
// transformers' chat-template renderer). The last case is the chat_template capability probe:
// a template that writes argument names between tags (K2-Horizon's <ifm|arg_key>) supports
// tool calls, so its tool-call history is rendered by the template, not polyfilled as JSON.
#include <unistd.h>

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

// Spec 23e: the same probe for a template that writes argument names as a double-quoted ATTRIBUTE -
// Muse-Glimmer's ATEM `<atem:parameter name="KEY">VALUE</atem:parameter>` (its chat_template.jinja T:31-49; a
// string `arguments` raises, T:28-30). Without the patch the probe finds no argument name it knows, and apply()
// rewrites the call history into JSON content.
void case_tool_call_probe_attribute() {
  const std::string source =
      "{%- for m in messages -%}"
      "{{- '<' + m.role + '>' -}}"
      "{%- if m.content is string -%}{{- m.content -}}{%- endif -%}"
      "{%- for tc in (m.tool_calls or []) -%}"
      "{%- if tc.function.arguments is not mapping -%}{{- raise_exception('arguments must be a mapping') -}}{%- endif -%}"
      "{{- '<atem:invoke name=\"' + tc.function.name + '\">' -}}"
      "{%- for k, v in tc.function.arguments.items() -%}"
      "{{- '<atem:parameter name=\"' + k + '\">' -}}{{- v -}}{{- '</atem:parameter>' -}}"
      "{%- endfor -%}"
      "{{- '</atem:invoke>' -}}"
      "{%- endfor -%}"
      "{%- endfor -%}";
  minja::chat_template tmpl(source, "", "");
  CHECK(tmpl.original_caps().supports_tool_calls);
  CHECK(tmpl.original_caps().requires_object_arguments);
  minja::chat_template_inputs inputs;
  inputs.messages = nlohmann::ordered_json::parse(R"([
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "function": {"name": "fs.read", "arguments": {"path": "/a"}}}]}])");
  inputs.add_generation_prompt = false;
  const std::string got = tmpl.apply(inputs);
  CHECK_EQ(got, std::string("<user>go<assistant><atem:invoke name=\"fs.read\"><atem:parameter name=\"path\">/a"
                            "</atem:parameter></atem:invoke>"));
  std::printf("tool_call_probe_attribute: %s\n", got.c_str());
}

// Spec 24e: the same probe for a template that writes a call as a PYTHONIC keyword argument - LFM2.5's
// `<|tool_call_start|>[fn(k='v', ...)]<|tool_call_end|>` (its chat_template.jinja's render_tool_calls: a mapping
// `arguments` becomes `name=value` pairs, a non-empty string raises). Without the patch the probe finds no argument
// name it knows (`argument_needle=` is none of the shapes above), and apply() rewrites the call history into JSON
// content. The second half: the rewrite that stays on when `arguments` must be an object (a string one is parsed
// first) must not throw on a call that carries no `arguments` key - the template reads it with .get() and renders `fn()`.
void case_tool_call_probe_pythonic() {
  const std::string source =
      "{%- for m in messages -%}"
      "{{- '<' + m.role + '>' -}}"
      "{%- if m.content is string -%}{{- m.content -}}{%- endif -%}"
      "{%- if m.tool_calls -%}"
      "{%- set ns = namespace(calls=[]) -%}"
      "{%- for tc in m.tool_calls -%}"
      "{%- set fn = tc['function'] if 'function' in tc else tc -%}"
      "{%- set args = fn.get('arguments') -%}"
      "{%- set pairs = namespace(items=[]) -%}"
      "{%- if args is mapping -%}"
      "{%- for k, v in args.items() -%}{%- set pairs.items = pairs.items + [k + '=' + (v | string)] -%}{%- endfor -%}"
      "{%- elif args is string and (args | trim) not in ['', '{}', 'null'] -%}"
      "{{- raise_exception('arguments must be a mapping') -}}"
      "{%- endif -%}"
      "{%- set ns.calls = ns.calls + [fn.name + '(' + (pairs.items | join(', ')) + ')'] -%}"
      "{%- endfor -%}"
      "{{- '[' + (ns.calls | join(', ')) + ']' -}}"
      "{%- endif -%}"
      "{%- endfor -%}";
  minja::chat_template tmpl(source, "", "");
  CHECK(tmpl.original_caps().supports_tool_calls);
  CHECK(tmpl.original_caps().requires_object_arguments);
  minja::chat_template_inputs inputs;
  inputs.messages = nlohmann::ordered_json::parse(R"([
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "function": {"name": "read", "arguments": {"path": "/a", "n": 3}}},
      {"type": "function", "function": {"name": "bare"}},
      {"type": "function", "function": {"name": "empty", "arguments": "{}"}},
      {"type": "function", "function": {"name": "json", "arguments": "{\"k\": 1}"}},
      {"type": "function", "function": {"name": "blank", "arguments": ""}},
      {"type": "function", "function": {"name": "nul", "arguments": "null"}},
      {"type": "function", "function": {"name": "padded", "arguments": "  {} "}}]}])");
  inputs.add_generation_prompt = false;
  const std::string got = tmpl.apply(inputs);
  std::printf("tool_call_probe_pythonic: %s\n", got.c_str());
  CHECK_EQ(got, std::string("<user>go<assistant>[read(path=/a, n=3), bare(), empty(), json(k=1), blank(), nul(), padded()]"));

  // The rewrite's "Failed to parse arguments" notice (the blank call) is printed once per process, not once per call: a
  // long-running server rendering a history with empty arguments must not repeat it. Two more renders of that
  // history print at most one line between them (zero if an earlier case in this process already printed it).
  std::fflush(stderr);
  std::FILE* tmp = std::tmpfile();
  CHECK(tmp != nullptr);
  const int saved = dup(2);
  dup2(fileno(tmp), 2);
  for (int i = 0; i < 2; ++i) (void)tmpl.apply(inputs);
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::rewind(tmp);
  std::string err;
  char buf[256];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, tmp)) > 0) err.append(buf, n);
  std::fclose(tmp);
  size_t notices = 0;
  for (size_t at = err.find("Failed to parse arguments"); at != std::string::npos; at = err.find("Failed to parse arguments", at + 1))
    ++notices;
  CHECK(notices <= 1);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  const int bad = run_jinja_cases(dir);
  CHECK_EQ(bad, 0);
  case_tool_call_probe();
  case_tool_call_probe_attribute();
  case_tool_call_probe_pythonic();
  std::printf("minja_ext_test OK\n");
  return 0;
}

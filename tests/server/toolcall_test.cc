// Spec 7 C6: the Qwen XML tool-call parser, in one piece and streamed.
//   toolcall_test <toolcall_expected.json> <tests/golden/toolcall dir>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "server/toolcall.h"

namespace {

using json = nlohmann::json;
using server::Delta;
using server::OutputStream;
using server::ParsedOutput;

json read_json(const std::string& path) {
  std::ifstream in(path);
  CHECK(in.good());
  return json::parse(in);
}

ParsedOutput collect(const std::vector<Delta>& deltas) {
  ParsedOutput out;
  for (const Delta& d : deltas) {
    if (d.kind == Delta::Reasoning) out.reasoning += d.text;
    if (d.kind == Delta::Content) out.content += d.text;
    if (d.kind == Delta::Call) {
      CHECK_EQ(d.index, static_cast<uint32_t>(out.tool_calls.size()));
      out.tool_calls.push_back(d.call);
    }
  }
  return out;
}

bool same(const ParsedOutput& a, const ParsedOutput& b) {
  if (a.reasoning != b.reasoning || a.content != b.content) return false;
  if (a.tool_calls.size() != b.tool_calls.size()) return false;
  for (size_t i = 0; i < a.tool_calls.size(); ++i) {
    if (a.tool_calls[i].name != b.tool_calls[i].name) return false;
    if (a.tool_calls[i].arguments != b.tool_calls[i].arguments) return false;
  }
  return true;
}

ParsedOutput stream(const std::string& text, bool thinking, const json& tools,
                    const std::vector<size_t>& sizes) {
  OutputStream s(thinking, tools);
  std::vector<Delta> all;
  size_t at = 0;
  for (size_t i = 0; at < text.size(); ++i) {
    const size_t n = sizes[i % sizes.size()];
    const std::vector<Delta> d = s.push(text.substr(at, n));
    all.insert(all.end(), d.begin(), d.end());
    at += n;
  }
  const std::vector<Delta> d = s.finish();
  all.insert(all.end(), d.begin(), d.end());
  return collect(all);
}

// The one-piece parse equals byte-at-a-time and random 1-7 byte streaming.
ParsedOutput parse_checked(const std::string& text, bool thinking, const json& tools) {
  const ParsedOutput whole = server::parse_output(text, thinking, tools);
  for (const server::ToolCall& c : whole.tool_calls) {
    CHECK(c.arguments.is_object());
    CHECK_EQ(c.id.size(), 29U);
    CHECK(c.id.rfind("call_", 0) == 0);
  }
  if (!same(whole, stream(text, thinking, tools, {1}))) {
    std::fprintf(stderr, "byte stream differs for: %s\n", text.c_str());
    CHECK(false);
  }
  std::mt19937 rng(7);
  std::vector<size_t> sizes(64);
  for (size_t& n : sizes) n = 1 + rng() % 7;
  CHECK(same(whole, stream(text, thinking, tools, sizes)));
  return whole;
}

// score.py's raw parameter string against the typed value.
bool param_equal(const json& got, const std::string& raw) {
  if (got.is_string()) return got.get<std::string>() == raw;
  return got == json::parse(raw);
}

const json kTools = json::parse(R"([
  {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{
    "filePath":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}}}}},
  {"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{
    "filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"},
    "replaceAll":{"type":"boolean"},"opts":{"type":"object"}}}}}
])");

void case_golden(const std::string& expected_path, const std::string& set_dir) {
  const json cases = read_json(expected_path);
  CHECK(cases.size() >= 36U);
  size_t calls = 0;
  for (const json& c : cases) {
    const json scenario = read_json(set_dir + "/" + c.at("scenario").get<std::string>() + ".json");
    const std::string text = c.at("text");
    const ParsedOutput p = parse_checked(text, c.at("thinking"), scenario.at("tools"));
    const std::string kind = c.at("kind");
    if (kind == "call") {
      CHECK_EQ(p.tool_calls.size(), c.at("n_calls").get<size_t>());
      const server::ToolCall& first = p.tool_calls.at(0);
      CHECK_EQ(first.name, c.at("name").get<std::string>());
      CHECK_EQ(first.arguments.size(), c.at("params").size());
      for (const auto& [key, raw] : c.at("params").items()) {
        if (!param_equal(first.arguments.at(key), raw.get<std::string>())) {
          std::fprintf(stderr, "%s %s: param %s differs\n", c.at("scenario").get<std::string>().c_str(),
                       c.at("backend").get<std::string>().c_str(), key.c_str());
          CHECK(false);
        }
      }
      ++calls;
    } else {
      // Cut by the 192-id budget inside a call: its function tag is complete, so the
      // call is still emitted, with the parameters that were complete.
      CHECK_EQ(kind, std::string("incomplete"));
      CHECK_EQ(p.tool_calls.size(), 1U);
      CHECK(!p.tool_calls[0].name.empty());
    }
  }
  CHECK(calls >= 36U);
}

void case_value_with_tags() {
  // Review Focus 1: a value holding "<", "</b>" and even "</parameter" text.
  const std::string text =
      "<tool_call>\n<function=edit>\n<parameter=filePath>\n/w/a.xml\n</parameter>\n"
      "<parameter=oldString>\na </b> <c>\n\n</parameter>\n"
      "<parameter=newString>\nx</parameter>y <parameter\n</parameter>\n</function>\n</tool_call>";
  const ParsedOutput p = parse_checked(text, false, kTools);
  CHECK_EQ(p.tool_calls.size(), 1U);
  CHECK_EQ(p.tool_calls[0].name, std::string("edit"));
  CHECK_EQ(p.tool_calls[0].arguments.at("oldString"), std::string("a </b> <c>\n"));
  CHECK_EQ(p.tool_calls[0].arguments.at("newString"), std::string("x</parameter>y <parameter"));
  CHECK_EQ(p.content, std::string(""));
}

void case_reasoning_mentions_tags() {
  // Review Focus 2: reasoning is never parsed for calls.
  const std::string text =
      "\nI should emit <tool_call> with <function=read> here.\n</think>\n\nReading.\n"
      "<tool_call>\n<function=read>\n<parameter=filePath>\n/w/x.cc\n</parameter>\n"
      "<parameter=offset>\n12\n</parameter>\n</function>\n</tool_call>";
  const ParsedOutput p = parse_checked(text, true, kTools);
  CHECK_EQ(p.reasoning, std::string("I should emit <tool_call> with <function=read> here."));
  CHECK_EQ(p.content, std::string("Reading.\n"));
  CHECK_EQ(p.tool_calls.size(), 1U);
  CHECK_EQ(p.tool_calls[0].arguments, json({{"filePath", "/w/x.cc"}, {"offset", 12}}));
  // No </think>: everything is reasoning.
  const ParsedOutput q = parse_checked("thinking <tool_call> still\n", true, kTools);
  CHECK_EQ(q.reasoning, std::string("thinking <tool_call> still"));
  CHECK(q.content.empty() && q.tool_calls.empty());
}

void case_truncated() {
  // Review Focus 3: cut before the function tag is complete -> content, not dropped.
  const ParsedOutput a = parse_checked("Let me look.\n<tool_call>\n<function=re", false, kTools);
  CHECK(a.tool_calls.empty());
  CHECK_EQ(a.content, std::string("Let me look.\n<tool_call>\n<function=re"));
  const ParsedOutput b = parse_checked("<tool_c", false, kTools);
  CHECK(b.tool_calls.empty());
  CHECK_EQ(b.content, std::string("<tool_c"));
  // Function tag complete: emitted with the complete parameters only.
  const ParsedOutput c = parse_checked(
      "<tool_call>\n<function=edit>\n<parameter=filePath>\n/w/y\n</parameter>\n<parameter=oldString>\nha",
      false, kTools);
  CHECK_EQ(c.tool_calls.size(), 1U);
  CHECK_EQ(c.tool_calls[0].arguments, json({{"filePath", "/w/y"}}));
  CHECK(c.content.empty());
}

void case_split_tags() {
  // Review Focus 4: tags split across pieces; streamed forms are checked in parse_checked.
  const std::string text =
      "ok\n<tool_call>\n<function=read>\n<parameter=filePath>\n/a\n</parameter>\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=read>\n<parameter=filePath>\n/b\n</parameter>\n</function>\n</tool_call>";
  const ParsedOutput p = parse_checked(text, false, kTools);
  CHECK_EQ(p.tool_calls.size(), 2U);
  CHECK_EQ(p.tool_calls[1].arguments.at("filePath"), std::string("/b"));
  CHECK(p.tool_calls[0].id != p.tool_calls[1].id);
  CHECK_EQ(p.content, std::string("ok\n"));
  OutputStream s(false, kTools);
  const std::vector<Delta> d1 = s.push("ok\n<tool_");
  CHECK_EQ(collect(d1).content, std::string("ok\n"));
  CHECK(collect(s.push("call>\n<function=read>\n")).content.empty());
}

void case_no_tools() {
  // Review Focus 5: tools absent -> still parsed, values typed by guessing JSON.
  const std::string text =
      "<tool_call>\n<function=anything>\n<parameter=n>\n42\n</parameter>\n<parameter=s>\nhello world\n"
      "</parameter>\n<parameter=o>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=q>\n\"quoted\"\n</parameter>\n"
      "<parameter=b>\ntrue\n</parameter>\n</function>\n</tool_call>";
  const ParsedOutput p = parse_checked(text, false, json(nullptr));
  CHECK_EQ(p.tool_calls.size(), 1U);
  CHECK_EQ(p.tool_calls[0].arguments,
           json({{"n", 42}, {"s", "hello world"}, {"o", {{"a", {1, 2}}}}, {"q", "quoted"}, {"b", true}}));
  // Schema: string stays a string even when it looks like JSON; integer parses.
  CHECK_EQ(server::convert_value("123", "edit", "oldString", kTools), json("123"));
  CHECK_EQ(server::convert_value("123", "read", "limit", kTools), json(123));
  CHECK_EQ(server::convert_value("not json", "read", "limit", kTools), json("not json"));
  CHECK_EQ(server::convert_value("{\"k\":1}", "edit", "opts", kTools), json({{"k", 1}}));
}

void case_thinking_off() {
  const std::string text = "plain\n</think>\n\nnot split";
  const ParsedOutput p = parse_checked(text, false, kTools);
  CHECK(p.reasoning.empty());
  CHECK_EQ(p.content, text);
  const ParsedOutput q = parse_checked("a b c ", false, kTools);
  CHECK_EQ(q.content, std::string("a b c "));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: toolcall_test <toolcall_expected.json> <golden toolcall dir>\n");
    return 2;
  }
  case_golden(argv[1], argv[2]);
  case_value_with_tags();
  case_reasoning_mentions_tags();
  case_truncated();
  case_split_tags();
  case_no_tools();
  case_thinking_off();
  std::printf("toolcall_test OK: 7 cases\n");
  return 0;
}

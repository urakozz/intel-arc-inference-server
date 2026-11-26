// Spec 41f Task 2: Inkling 2 Mini's reasoning, text and JSON tool-call blocks (server/toolcall_inkling2.h).
//   toolcall_inkling2_test <tests/tokenizer>
// Every parse is checked whole, byte at a time and in random 1-7 byte pieces (the deltas of any split concatenate to
// the whole-text parse). Cases:
//   - the round trip: the assistant turns of the HF renders (inkling2_template_call_history / tool_result_name /
//     thinking_history / reasoning_and_call) parse back to the messages the template rendered;
//   - several calls a turn, content and reasoning around them, keys in the model's order, schema typing (a string
//     argument written as a number becomes its JSON text), raw newlines inside string arguments repaired;
//   - drift: "arguments" for "args", a string holding the args object, the JSON's name over the block's, text outside
//     any block, an unknown marker after the header, a name closed with no call, a stray turn end;
//   - not a call: JSON that does not parse, args not an object, no name - content verbatim;
//   - cut by the end of generation: reasoning / text keep what arrived; a call cut in its name or JSON is dropped;
//   - a fuzz: 1000 generated outputs (0-2 thinking / text blocks, 0-3 valid calls) parse to the generated structure,
//     and cut at a random byte parse identically under every split.
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "server/chat_format.h"
#include "server/toolcall.h"
#include "server/toolcall_inkling2.h"

namespace {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;
using server::Delta;
using server::Inkling2OutputStream;
using server::ParsedOutput;

const std::string H = "<|message_model|>", TH = "<|content_thinking|>", TX = "<|content_text|>",
                  TJ = "<|content_invoke_tool_json|>", E = "<|end_message|>", ET = "<|content_model_end_sampling|>";

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
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
  if (a.reasoning != b.reasoning || a.content != b.content || a.tool_calls.size() != b.tool_calls.size()) return false;
  for (size_t i = 0; i < a.tool_calls.size(); ++i)
    if (a.tool_calls[i].name != b.tool_calls[i].name || a.tool_calls[i].arguments.dump() != b.tool_calls[i].arguments.dump())
      return false;
  return true;
}

std::string show(const ParsedOutput& p) {
  ojson calls = ojson::array();
  for (const auto& c : p.tool_calls) calls.push_back({{"name", c.name}, {"arguments", c.arguments}});
  return ojson({{"reasoning", p.reasoning}, {"content", p.content}, {"calls", calls}}).dump();
}

ParsedOutput split_parse(const std::string& text, const json& tools, const std::vector<size_t>& cuts) {
  Inkling2OutputStream s(tools);
  std::vector<Delta> all;
  size_t at = 0;
  for (size_t c : cuts) {
    for (auto& d : s.push(text.substr(at, c - at))) all.push_back(d);
    at = c;
  }
  for (auto& d : s.push(text.substr(at))) all.push_back(d);
  for (auto& d : s.finish()) all.push_back(d);
  return collect(all);
}

// Whole, byte at a time, and in random pieces: every split parses as the whole.
ParsedOutput parse_all_ways(const std::string& text, const json& tools, std::mt19937& rng) {
  const ParsedOutput whole = server::parse_output_inkling2(text, tools);
  std::vector<size_t> bytes;
  for (size_t i = 1; i < text.size(); ++i) bytes.push_back(i);
  const ParsedOutput by_byte = split_parse(text, tools, bytes);
  if (!same(whole, by_byte)) {
    std::fprintf(stderr, "byte split differs on %s\n  whole %s\n  bytes %s\n", json(text).dump().c_str(),
                 show(whole).c_str(), show(by_byte).c_str());
    CHECK(false);
  }
  for (int r = 0; r < 4; ++r) {
    std::vector<size_t> cuts;
    for (size_t at = 0; at < text.size();) {
      at += 1 + rng() % 7;
      if (at < text.size()) cuts.push_back(at);
    }
    CHECK(same(whole, split_parse(text, tools, cuts)));
  }
  return whole;
}

ParsedOutput want(const std::string& reasoning, const std::string& content,
                  std::vector<std::pair<std::string, ojson>> calls = {}) {
  ParsedOutput p;
  p.reasoning = reasoning;
  p.content = content;
  for (auto& [n, a] : calls) p.tool_calls.push_back({"", n, a});
  return p;
}

void expect(const std::string& name, const std::string& text, const ParsedOutput& w, const json& tools,
            std::mt19937& rng) {
  const ParsedOutput got = parse_all_ways(text, tools, rng);
  if (!same(got, w)) {
    std::fprintf(stderr, "%s:\n  got  %s\n  want %s\n", name.c_str(), show(got).c_str(), show(w).c_str());
    CHECK(false);
  }
}

std::string call(const std::string& name, const std::string& args) {
  return H + name + TJ + "{\"name\":\"" + name + "\",\"args\":" + args + "}" + E;
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc == 2);
  const std::string dir = argv[1];
  std::mt19937 rng(41);
  const json tools = json::parse(R"([
    {"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number"}}}}},
    {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"limit":{"type":"integer"}}}}},
    {"type":"function","function":{"name":"grep","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}}}}}])");

  // The round trip: the template's own assistant blocks (from the HF renders) parse back to the messages.
  {
    const std::string r = slurp(dir + "/inkling2_template_call_history.txt");
    const std::string first_turn = r.substr(r.find(H + "grep"), r.find(ET) - r.find(H + "grep"));
    expect("call_history turn 1", first_turn,
           want("", "", {{"grep", ojson::parse(R"({"Include":null,"path":"/work/src","pattern":"kBlock\\b"})")},
                         {"read", ojson::parse(R"({"filePath":"/work/src/runtime/engine.h","limit":20,"offset":140})")}}),
           tools, rng);
    const size_t b = r.find(H + "bash");
    expect("call_history turn 2", r.substr(b, r.find(ET, b) - b),
           want("", "", {{"bash", ojson::parse(R"({"command":"sed -i 's/kBlock/kStoreBlock/g' src/runtime/engine.h && echo \"done ✓\"","timeout":30.5})")}}),
           tools, rng);
    const size_t c = r.find(H + TX);
    expect("call_history turn 3", r.substr(c, r.find(ET, c) - c), want("", "Renamed in src/runtime/engine.h."), tools, rng);
    const std::string n = slurp(dir + "/inkling2_template_tool_result_name.txt");
    const size_t d = n.find(H + TX);
    expect("tool_result_name", n.substr(d, n.find(ET, d) - d),
           want("", "Listing it.", {{"bash", ojson::parse(R"({"command":"ls"})")}}), tools, rng);
    const std::string t = slurp(dir + "/inkling2_template_thinking_history.txt");
    const size_t f = t.find(H + TH);
    expect("thinking_history", t.substr(f, t.find(ET, f) - f), want("91 = 7 x 13.", "No: 91 = 7 x 13."), tools, rng);
    const std::string rc = slurp(dir + "/inkling2_template_reasoning_and_call.txt");
    const size_t g = rc.find(H + TH);
    expect("reasoning_and_call", rc.substr(g, rc.find(ET, g) - g),
           want("I should read it first.", "", {{"read", ojson::parse(R"({"filePath":"/work/README.md"})")}}), tools, rng);
  }

  // Several calls, keys in the model's order, schema typing, repaired raw newlines, the turn end after them.
  expect("calls", H + TH + "plan" + E + H + TX + "Doing it." + E + call("bash", R"({"timeout":5,"command":"ls -la"})") +
                      call("read", R"({"filePath":"/a","limit":3})") + ET,
         want("plan", "Doing it.", {{"bash", ojson::parse(R"({"timeout":5,"command":"ls -la"})")},
                                    {"read", ojson::parse(R"({"filePath":"/a","limit":3})")}}),
         tools, rng);
  expect("schema string", call("bash", R"({"command":12})"), want("", "", {{"bash", ojson::parse(R"({"command":"12"})")}}),
         tools, rng);
  expect("raw newline", H + "bash" + TJ + "{\"name\":\"bash\",\"args\":{\"command\":\"echo a\nb\"}}" + E,
         want("", "", {{"bash", ojson::parse(R"({"command":"echo a\nb"})")}}), tools, rng);

  // Drift.
  expect("arguments key", H + "read" + TJ + R"({"name":"read","arguments":{"filePath":"/x"}})" + E,
         want("", "", {{"read", ojson::parse(R"({"filePath":"/x"})")}}), tools, rng);
  expect("string args", H + "read" + TJ + R"({"name":"read","args":"{\"filePath\":\"/y\"}"})" + E,
         want("", "", {{"read", ojson::parse(R"({"filePath":"/y"})")}}), tools, rng);
  expect("json name wins", H + "rea" + TJ + R"({"name":"read","args":{}})" + E, want("", "", {{"read", ojson::object()}}),
         tools, rng);
  expect("block name", H + " read " + TJ + R"({"args":{"filePath":"/z"}})" + E,
         want("", "", {{"read", ojson::parse(R"({"filePath":"/z"})")}}), tools, rng);
  expect("text outside", "Sure. " + H + TX + "x" + E + "tail", want("", "Sure. xtail"), tools, rng);
  expect("other marker", H + "<|content_xml|>abc" + E, want("", "<|content_xml|>abc"), tools, rng);
  expect("tool error block", H + "<|content_tool_error|>boom" + E, want("", "boom"), tools, rng);
  expect("name only", H + "hello" + E + H + TX + "x" + E, want("", "hellox"), tools, rng);
  expect("stray turn end", H + TX + "a" + E + ET + H + TX + "b" + E + ET, want("", "ab"), tools, rng);

  // Not a call: content verbatim.
  expect("broken json", H + "bash" + TJ + R"({"name":"bash","args":{"command":"ls")" + E,
         want("", "bash" + TJ + R"({"name":"bash","args":{"command":"ls")"), tools, rng);
  expect("args array", H + "bash" + TJ + R"({"name":"bash","args":[1]})" + E,
         want("", "bash" + TJ + R"({"name":"bash","args":[1]})"), tools, rng);
  expect("no args", H + "bash" + TJ + R"({"name":"bash"})" + E, want("", "bash" + TJ + R"({"name":"bash"})"), tools, rng);
  expect("name with space", H + "a b" + TJ + R"({"name":"a b","args":{}})" + E,
         want("", "a b" + TJ + R"({"name":"a b","args":{}})"), tools, rng);

  // Cut by the end of generation.
  expect("cut reasoning", H + TH + "still thinking", want("still thinking", ""), tools, rng);
  expect("cut text", H + TX + "half an ans", want("", "half an ans"), tools, rng);
  expect("cut in json", H + TX + "ok" + E + H + "bash" + TJ + R"({"name":"bash","args":{"comm)", want("", "ok"), tools, rng);
  expect("cut in name", H + TX + "ok" + E + H + "ba", want("", "ok"), tools, rng);
  expect("cut in marker", H + TX + "ok" + E + "<|message_mo", want("", "ok<|message_mo"), tools, rng);
  expect("empty", "", want("", ""), tools, rng);

  // The chat format picks this parser for model_type inkling2_loop.
  {
    auto p = server::make_output_parser(server::ChatFormat::for_model_type("inkling2_loop"), "", json::object(), tools);
    std::vector<Delta> d = p->push(call("read", R"({"filePath":"/q"})"));
    for (auto& x : p->finish()) d.push_back(x);
    CHECK(collect(d).tool_calls.size() == 1);
  }

  // The fuzz: generated outputs parse to their structure; cut at a random byte, every split parses alike.
  const std::vector<std::string> words = {"a", "b c", "ü", "{", "}", "\"q\"", "<|", "|>", "\n", "x<|end", "text"};
  for (int n = 0; n < 1000; ++n) {
    std::string text;
    ParsedOutput w;
    const auto phrase = [&] {
      std::string s;
      for (int k = 1 + rng() % 4; k > 0; --k) s += words[rng() % words.size()];
      return s;
    };
    if (rng() % 2) {
      const std::string r = phrase();
      text += H + TH + r + E;
      w.reasoning += r;
    }
    if (rng() % 2) {
      const std::string c = phrase();
      text += H + TX + c + E;
      w.content += c;
    }
    for (int k = rng() % 4; k > 0; --k) {
      ojson args = ojson::object();
      args["command"] = phrase();
      if (rng() % 2) args["timeout"] = int(rng() % 100);
      text += call("bash", args.dump());
      w.tool_calls.push_back({"", "bash", args});
    }
    if (rng() % 2) text += ET;
    CHECK(same(parse_all_ways(text, tools, rng), w));
    (void)parse_all_ways(text.substr(0, rng() % (text.size() + 1)), tools, rng);
  }
  std::printf("toolcall_inkling2_test OK\n");
  return 0;
}

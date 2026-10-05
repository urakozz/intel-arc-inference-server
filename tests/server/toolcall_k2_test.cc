// Spec 18d: K2-Horizon's tool-call and reasoning parser (server/toolcall_k2.h).
//   toolcall_k2_test <tests/tokenizer>
// Every parse is checked whole, byte at a time and in random 1-7 byte pieces (spec 7a's
// streaming rule: the deltas of any split concatenate to the whole-text parse). Cases:
//   - the round trip: every assistant turn of the HF renders of k2_template_cases.json
//     (xml, xml_typed and json history, reasoning, content, parallel calls) parses back to
//     the message the template rendered - reasoning, content, names, typed arguments;
//   - the template's own format examples, reasoning closed by its tag / by a call block /
//     not at all, think_fast, a repeated open tag, tags inside values, schema typing, the
//     stated xml_typed types, json bodies, bare calls, no-argument calls, malformed and cut
//     calls, whitespace-only content;
//   - a fuzz: 3000 generated outputs (random reasoning and content around tag fragments,
//     0-3 calls in a random format with random values, random cut points) - the parse equals
//     the generated structure when uncut, and every split equals the whole when cut.
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "server/chat_format.h"
#include "server/toolcall_k2.h"

namespace {

using json = nlohmann::json;
using server::Delta;
using server::K2CallFormat;
using server::K2OutputStream;
using server::ParsedOutput;

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
  if (a.reasoning != b.reasoning || a.content != b.content) return false;
  if (a.tool_calls.size() != b.tool_calls.size()) return false;
  for (size_t i = 0; i < a.tool_calls.size(); ++i) {
    if (a.tool_calls[i].name != b.tool_calls[i].name) return false;
    if (a.tool_calls[i].arguments != b.tool_calls[i].arguments) return false;
  }
  return true;
}

std::string show(const ParsedOutput& p) {
  json calls = json::array();
  for (const auto& c : p.tool_calls) calls.push_back({{"name", c.name}, {"arguments", c.arguments}});
  return json({{"reasoning", p.reasoning}, {"content", p.content}, {"calls", calls}}).dump();
}

ParsedOutput stream(const std::string& text, const std::string& open, K2CallFormat f, const json& tools,
                    const std::vector<size_t>& sizes) {
  K2OutputStream s(open, f, tools);
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

uint32_t g_seed = 7;

// The whole-text parse equals byte-at-a-time and random 1-7 byte streaming.
ParsedOutput parse_checked(const std::string& text, const std::string& open, K2CallFormat f,
                           const json& tools) {
  const ParsedOutput whole = server::parse_output_k2(text, open, f, tools);
  for (const server::ToolCall& c : whole.tool_calls) {
    CHECK(c.arguments.is_object());
    CHECK_EQ(c.id.size(), 29U);
    CHECK(c.id.rfind("call_", 0) == 0);
  }
  const ParsedOutput bytes = stream(text, open, f, tools, {1});
  if (!same(whole, bytes)) {
    std::fprintf(stderr, "byte stream differs for %s\n  whole: %s\n  bytes: %s\n", json(text).dump().c_str(),
                 show(whole).c_str(), show(bytes).c_str());
    CHECK(false);
  }
  std::mt19937 rng(g_seed++);
  std::vector<size_t> sizes(64);
  for (size_t& n : sizes) n = 1 + rng() % 7;
  const ParsedOutput pieces = stream(text, open, f, tools, sizes);
  if (!same(whole, pieces)) {
    std::fprintf(stderr, "random pieces differ for %s\n  whole: %s\n  split: %s\n", json(text).dump().c_str(),
                 show(whole).c_str(), show(pieces).c_str());
    CHECK(false);
  }
  return whole;
}

const json kTools = json::parse(R"([
  {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{
    "filePath":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}}}}},
  {"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{
    "filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"},
    "replaceAll":{"type":"boolean"},"opts":{"type":"object"},"note":{"type":["string","null"]}}}}},
  {"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}
])");

// --- the round trip through the HF renders ------------------------------------------------
void case_round_trip(const std::string& dir) {
  const json spec = json::parse(slurp(dir + "/k2_template_cases.json"));
  const std::string turn = "<|ifm|im_start|>assistant\n";
  size_t turns = 0, calls = 0;
  for (const json& c : spec.at("cases")) {
    const std::string name = c.at("name");
    K2CallFormat format = K2CallFormat::Xml;
    if (c.at("kwargs").contains("tool_call_format"))
      CHECK(server::parse_k2_call_format(c.at("kwargs").at("tool_call_format"), format));
    json tools = nullptr;
    if (c.at("tools").is_array()) {
      tools = json::array();
      for (const json& n : c.at("tools")) tools.push_back(spec.at("tools").at(n.get<std::string>()));
    }
    const std::string text = slurp(dir + "/k2_template_" + name + ".txt");
    std::vector<const json*> assistants;
    for (const json& m : c.at("messages")) {
      if (m.at("role") == "assistant") assistants.push_back(&m);
    }
    size_t at = 0, k = 0;
    for (size_t i = text.find(turn); i != std::string::npos; i = text.find(turn, at)) {
      const size_t body = i + turn.size();
      const size_t end = text.find("<|ifm|im_end|>", body);
      if (end == std::string::npos) break;   // the generation prompt
      // The prompt of the next turn would end in the think tag; the generated text follows it.
      std::string open;
      for (const char* tag : {"<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"}) {
        const std::string t = std::string(tag) + "\n";
        if (text.compare(body, t.size(), t) == 0) open = tag;
      }
      CHECK(!open.empty());
      const std::string generated = text.substr(body + open.size() + 1, end - body - open.size() - 1);
      CHECK(k < assistants.size());
      const json& m = *assistants[k++];
      const ParsedOutput p = parse_checked(generated, open, format, tools);
      std::string want_reasoning;
      for (const char* f : {"think", "think_fast", "think_faster", "reasoning_content", "reasoning"}) {
        if (want_reasoning.empty() && m.contains(f) && m.at(f).is_string()) want_reasoning = m.at(f);
      }
      const std::string content = m.at("content").is_string() ? m.at("content").get<std::string>() : "";
      bool ok = p.reasoning == want_reasoning && p.content == content;
      const json want_calls = m.contains("tool_calls") ? m.at("tool_calls") : json::array();
      ok = ok && p.tool_calls.size() == want_calls.size();
      for (size_t j = 0; ok && j < want_calls.size(); ++j) {
        ok = p.tool_calls[j].name == want_calls[j].at("function").at("name") &&
             p.tool_calls[j].arguments == want_calls[j].at("function").at("arguments");
      }
      if (!ok) {
        std::fprintf(stderr, "round trip %s turn %zu: %s\n  from %s\n", name.c_str(), k, show(p).c_str(),
                     json(generated).dump().c_str());
        CHECK(false);
      }
      calls += p.tool_calls.size();
      ++turns;
      at = end;
    }
    CHECK_EQ(k, assistants.size());
  }
  CHECK(turns >= 9U);
  CHECK(calls >= 7U);
  std::printf("round trip: %zu assistant turns, %zu calls parse back to their messages\n", turns, calls);
}

// --- the format's examples and edges -------------------------------------------------------
void case_reasoning() {
  const auto p = parse_checked("Let me think.\nOk.</ifm|think>The answer is 4.", "<ifm|think>",
                               K2CallFormat::Xml, nullptr);
  CHECK_EQ(p.reasoning, std::string("Let me think.\nOk."));
  CHECK_EQ(p.content, std::string("The answer is 4."));
  // No close tag: all reasoning (cut while thinking).
  const auto q = parse_checked("still thinking </ifm|thi", "<ifm|think>", K2CallFormat::Xml, nullptr);
  CHECK_EQ(q.reasoning, std::string("still thinking </ifm|thi"));
  CHECK(q.content.empty());
  // A repeated open tag at the very start is dropped; elsewhere it is text.
  const auto r = parse_checked("<ifm|think>\nx <ifm|think> y</ifm|think>z", "<ifm|think>", K2CallFormat::Xml, nullptr);
  CHECK_EQ(r.reasoning, std::string("\nx <ifm|think> y"));
  CHECK_EQ(r.content, std::string("z"));
  // think_fast closes on its own tag only.
  const auto s = parse_checked("a</ifm|think>b</ifm|think_fast>c", "<ifm|think_fast>", K2CallFormat::Xml, nullptr);
  CHECK_EQ(s.reasoning, std::string("a</ifm|think>b"));
  CHECK_EQ(s.content, std::string("c"));
  // A call block ends reasoning (vLLM's reasoning parser does the same).
  const auto t = parse_checked("I will read it.<ifm|tool_calls>\n<ifm|tool_call>read\n<ifm|arg_key>filePath"
                               "</ifm|arg_key>\n<ifm|arg_value>/a</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
                               "<ifm|think>", K2CallFormat::Xml, kTools);
  CHECK_EQ(t.reasoning, std::string("I will read it."));
  CHECK_EQ(t.tool_calls.size(), 1U);
  CHECK_EQ(t.tool_calls[0].arguments, json({{"filePath", "/a"}}));
  // No reasoning opened: the text is content; a close tag in it is text.
  const auto u = parse_checked("plain </ifm|think> text", "", K2CallFormat::Xml, nullptr);
  CHECK(u.reasoning.empty());
  CHECK_EQ(u.content, std::string("plain </ifm|think> text"));
}

void case_xml() {
  // The template's instruction example, two calls, content before, whitespace between.
  const std::string text =
      "</ifm|think>Reading both.\n<ifm|tool_calls>\n<ifm|tool_call>read\n<ifm|arg_key>filePath</ifm|arg_key>\n"
      "<ifm|arg_value>/w/a.cc</ifm|arg_value>\n<ifm|arg_key>offset</ifm|arg_key>\n<ifm|arg_value>12</ifm|arg_value>\n"
      "</ifm|tool_call>\n<ifm|tool_call>edit\n<ifm|arg_key>filePath</ifm|arg_key>\n<ifm|arg_value>/w/b.xml"
      "</ifm|arg_value>\n<ifm|arg_key>oldString</ifm|arg_key>\n<ifm|arg_value>  a </ifm|arg_value> <b>\n"
      "</ifm|arg_value>\n<ifm|arg_key>newString</ifm|arg_key>\n<ifm|arg_value>123</ifm|arg_value>\n"
      "<ifm|arg_key>replaceAll</ifm|arg_key>\n<ifm|arg_value>true</ifm|arg_value>\n<ifm|arg_key>opts"
      "</ifm|arg_key>\n<ifm|arg_value>{\"k\": [1, 2]}</ifm|arg_value>\n<ifm|arg_key>note</ifm|arg_key>\n"
      "<ifm|arg_value>null</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>";
  const auto p = parse_checked(text, "<ifm|think>", K2CallFormat::Xml, kTools);
  CHECK(p.reasoning.empty());
  CHECK_EQ(p.content, std::string("Reading both.\n"));
  CHECK_EQ(p.tool_calls.size(), 2U);
  CHECK_EQ(p.tool_calls[0].name, std::string("read"));
  CHECK_EQ(p.tool_calls[0].arguments, json({{"filePath", "/w/a.cc"}, {"offset", 12}}));
  CHECK_EQ(p.tool_calls[1].name, std::string("edit"));
  // A string value is verbatim (whitespace, a "</ifm|arg_value>" not followed by a key, tags);
  // a schema string stays a string even when it reads as JSON ("123", "null" for string|null).
  CHECK_EQ(p.tool_calls[1].arguments,
           json({{"filePath", "/w/b.xml"}, {"oldString", "  a </ifm|arg_value> <b>\n"}, {"newString", "123"},
                 {"replaceAll", true}, {"opts", {{"k", {1, 2}}}}, {"note", "null"}}));
  CHECK(p.tool_calls[0].id != p.tool_calls[1].id);
  // Non-string schema values are trimmed before the JSON parse; unparseable ones stay text.
  const auto q = parse_checked("<ifm|tool_calls>\n<ifm|tool_call>read\n<ifm|arg_key> offset </ifm|arg_key>\n"
                               "<ifm|arg_value>\n 7 \n</ifm|arg_value>\n<ifm|arg_key>limit</ifm|arg_key>\n"
                               "<ifm|arg_value>ten</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
                               "", K2CallFormat::Xml, kTools);
  CHECK_EQ(q.tool_calls.size(), 1U);
  CHECK_EQ(q.tool_calls[0].arguments, json({{"offset", 7}, {"limit", "ten"}}));
  // No tools: values typed by guessing JSON (server::convert_value's rule).
  const auto r = parse_checked("<ifm|tool_calls><ifm|tool_call>anything\n<ifm|arg_key>n</ifm|arg_key>"
                               "<ifm|arg_value>42</ifm|arg_value><ifm|arg_key>s</ifm|arg_key><ifm|arg_value>hello "
                               "world</ifm|arg_value><ifm|arg_key>q</ifm|arg_key><ifm|arg_value>\"quoted\""
                               "</ifm|arg_value></ifm|tool_call></ifm|tool_calls>",
                               "", K2CallFormat::Xml, nullptr);
  CHECK_EQ(r.tool_calls.size(), 1U);
  CHECK_EQ(r.tool_calls[0].arguments, json({{"n", 42}, {"s", "hello world"}, {"q", "quoted"}}));
  CHECK(r.content.empty());
}

void case_xml_typed() {
  // The stated type decides where the schema has none (no tools here).
  const std::string text =
      "<ifm|tool_calls>\n<ifm|tool_call>todo\n<ifm|arg_key>items</ifm|arg_key>\n<ifm|arg_type>array[object]"
      "</ifm|arg_type>\n<ifm|arg_value>[{\"id\": \"1\"}]</ifm|arg_value>\n<ifm|arg_key>n</ifm|arg_key>\n"
      "<ifm|arg_type>integer</ifm|arg_type>\n<ifm|arg_value>40</ifm|arg_value>\n<ifm|arg_key>s</ifm|arg_key>\n"
      "<ifm|arg_type>string</ifm|arg_type>\n<ifm|arg_value>40</ifm|arg_value>\n<ifm|arg_key>a</ifm|arg_key>\n"
      "<ifm|arg_type>any</ifm|arg_type>\n<ifm|arg_value>[1]</ifm|arg_value>\n<ifm|arg_key>z</ifm|arg_key>\n"
      "<ifm|arg_type>null</ifm|arg_type>\n<ifm|arg_value>null</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>";
  const auto p = parse_checked(text, "", K2CallFormat::XmlTyped, nullptr);
  CHECK_EQ(p.tool_calls.size(), 1U);
  CHECK_EQ(p.tool_calls[0].arguments,
           json({{"items", json::array({{{"id", "1"}}})}, {"n", 40}, {"s", "40"}, {"a", "[1]"}, {"z", nullptr}}));
  // The schema wins over a stated type.
  const auto q = parse_checked("<ifm|tool_calls><ifm|tool_call>read\n<ifm|arg_key>filePath</ifm|arg_key>"
                               "<ifm|arg_type>integer</ifm|arg_type><ifm|arg_value>12</ifm|arg_value>"
                               "</ifm|tool_call></ifm|tool_calls>",
                               "", K2CallFormat::XmlTyped, kTools);
  CHECK_EQ(q.tool_calls[0].arguments, json({{"filePath", "12"}}));
}

void case_json() {
  const std::string text =
      "Done thinking</ifm|think><ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"read\", \"arguments\": "
      "{\"filePath\": \"/a\", \"limit\": 40}}</ifm|tool_call>\n<ifm|tool_call>{\"name\": \"edit\", \"arguments\": "
      "\"{\\\"newString\\\": 5, \\\"replaceAll\\\": false}\"}</ifm|tool_call>\n</ifm|tool_calls>";
  const auto p = parse_checked(text, "<ifm|think>", K2CallFormat::Json, kTools);
  CHECK_EQ(p.reasoning, std::string("Done thinking"));
  CHECK(p.content.empty());
  CHECK_EQ(p.tool_calls.size(), 2U);
  CHECK_EQ(p.tool_calls[0].arguments, json({{"filePath", "/a"}, {"limit", 40}}));
  // arguments as a JSON string; a schema-string argument written as a number becomes its text.
  CHECK_EQ(p.tool_calls[1].arguments, json({{"newString", "5"}, {"replaceAll", false}}));
  // Invalid JSON: the call is content, verbatim with its tags.
  const auto q = parse_checked("<ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"read\", </ifm|tool_call>\n</ifm|tool_calls>",
                               "", K2CallFormat::Json, kTools);
  CHECK(q.tool_calls.empty());
  CHECK_EQ(q.content, std::string("<ifm|tool_call>{\"name\": \"read\", </ifm|tool_call>"));
  // A JSON body under the xml format is still read as JSON.
  const auto r = parse_checked("<ifm|tool_calls><ifm|tool_call>{\"name\": \"now\"}</ifm|tool_call></ifm|tool_calls>",
                               "", K2CallFormat::Xml, kTools);
  CHECK_EQ(r.tool_calls.size(), 1U);
  CHECK_EQ(r.tool_calls[0].name, std::string("now"));
  CHECK_EQ(r.tool_calls[0].arguments, json::object());
}

void case_edges() {
  // No arguments, with and without the name's newline; a bare call outside a block.
  const auto p = parse_checked("<ifm|tool_calls>\n<ifm|tool_call>now\n</ifm|tool_call>\n<ifm|tool_call>now"
                               "</ifm|tool_call>\n</ifm|tool_calls>",
                               "", K2CallFormat::Xml, kTools);
  CHECK_EQ(p.tool_calls.size(), 2U);
  CHECK(p.content.empty());
  const auto q = parse_checked("ok <ifm|tool_call>read\n<ifm|arg_key>filePath</ifm|arg_key>\n<ifm|arg_value>/x"
                               "</ifm|arg_value>\n</ifm|tool_call> after",
                               "", K2CallFormat::Xml, kTools);
  CHECK_EQ(q.tool_calls.size(), 1U);
  CHECK_EQ(q.content, std::string("ok  after"));
  // Whitespace-only content is dropped; content around a block is kept verbatim.
  const auto r = parse_checked("x</ifm|think>\n\n<ifm|tool_calls>\n<ifm|tool_call>now\n</ifm|tool_call>\n"
                               "</ifm|tool_calls>\n",
                               "<ifm|think>", K2CallFormat::Xml, kTools);
  CHECK_EQ(r.reasoning, std::string("x"));
  CHECK(r.content.empty());
  CHECK_EQ(r.tool_calls.size(), 1U);
  const auto s = parse_checked("  Hi\n<ifm|tool_calls><ifm|tool_call>now</ifm|tool_call> stray </ifm|tool_calls>bye",
                               "", K2CallFormat::Xml, kTools);
  CHECK_EQ(s.content, std::string("  Hi\n stray bye"));
  // Malformed (text between arguments, a name with a space): content, verbatim with tags.
  const std::string bad1 = "<ifm|tool_call>read\nhello <ifm|arg_key>filePath</ifm|arg_key><ifm|arg_value>/a"
                           "</ifm|arg_value></ifm|tool_call>";
  const auto t = parse_checked("<ifm|tool_calls>" + bad1 + "</ifm|tool_calls>", "", K2CallFormat::Xml, kTools);
  CHECK(t.tool_calls.empty());
  CHECK_EQ(t.content, bad1);
  const auto u = parse_checked("<ifm|tool_calls><ifm|tool_call>read file\n</ifm|tool_call></ifm|tool_calls>", "",
                               K2CallFormat::Xml, kTools);
  CHECK(u.tool_calls.empty());
  CHECK_EQ(u.content, std::string("<ifm|tool_call>read file\n</ifm|tool_call>"));
  // A value holding "</ifm|tool_call>" text: the call closes at the real end.
  const auto v = parse_checked("<ifm|tool_calls><ifm|tool_call>edit\n<ifm|arg_key>oldString</ifm|arg_key>"
                               "<ifm|arg_value>a</ifm|tool_call>b</ifm|arg_value>\n</ifm|tool_call></ifm|tool_calls>",
                               "", K2CallFormat::Xml, kTools);
  CHECK_EQ(v.tool_calls.size(), 1U);
  CHECK_EQ(v.tool_calls[0].arguments, json({{"oldString", "a</ifm|tool_call>b"}}));
}

void case_truncated() {
  // Cut before the name line ends: content from the call's tag.
  const auto a = parse_checked("Let me look.\n<ifm|tool_calls>\n<ifm|tool_call>rea", "", K2CallFormat::Xml, kTools);
  CHECK(a.tool_calls.empty());
  CHECK_EQ(a.content, std::string("Let me look.\n<ifm|tool_call>rea"));
  // Name complete: the call, with the complete arguments only (cut mid-value, mid-tag).
  for (const std::string tail : {"<ifm|arg_value>ha", "<ifm|arg_val", "", "<ifm|arg_type>str", "<ifm|ar"}) {
    const auto b = parse_checked("<ifm|tool_calls>\n<ifm|tool_call>edit\n<ifm|arg_key>filePath</ifm|arg_key>\n"
                                 "<ifm|arg_value>/w/y</ifm|arg_value>\n<ifm|arg_key>oldString</ifm|arg_key>\n" + tail,
                                 "", K2CallFormat::XmlTyped, kTools);
    CHECK_EQ(b.tool_calls.size(), 1U);
    CHECK_EQ(b.tool_calls[0].arguments, json({{"filePath", "/w/y"}}));
    CHECK(b.content.empty());
  }
  // Cut in json: content.
  const auto c = parse_checked("<ifm|tool_calls><ifm|tool_call>{\"name\": \"re", "", K2CallFormat::Json, kTools);
  CHECK(c.tool_calls.empty());
  CHECK_EQ(c.content, std::string("<ifm|tool_call>{\"name\": \"re"));
  // Cut inside a tag at the very end: kept as text.
  const auto d = parse_checked("answer <ifm|tool_", "", K2CallFormat::Xml, kTools);
  CHECK_EQ(d.content, std::string("answer <ifm|tool_"));
}

void case_factory() {
  using server::ChatFormat;
  const ChatFormat k2 = ChatFormat::for_model_type("k2_horizon");
  CHECK(k2.kind == ChatFormat::Kind::K2Horizon && k2.template_kwargs());
  CHECK(ChatFormat::for_model_type("qwen3_5").kind == ChatFormat::Kind::Qwen);
  CHECK(!ChatFormat::for_model_type("").template_kwargs());
  const auto run = [](server::OutputParser& p, const std::string& text) {
    std::vector<Delta> d = p.push(text);
    const std::vector<Delta> t = p.finish();
    d.insert(d.end(), t.begin(), t.end());
    return collect(d);
  };
  // The prompt's ending picks the reasoning tag; kwargs pick the call format.
  auto p1 = server::make_output_parser(k2, "...<|ifm|im_start|>assistant\n<ifm|think_faster>\n", json::object(), kTools);
  CHECK_EQ(run(*p1, "r</ifm|think_faster>c").reasoning, std::string("r"));
  auto p2 = server::make_output_parser(k2, "...assistant\n<ifm|think>\n", json({{"tool_call_format", "json"}}), kTools);
  const ParsedOutput o2 = run(*p2, "</ifm|think><ifm|tool_calls><ifm|tool_call>{\"name\": \"now\", \"arguments\": {}}"
                                   "</ifm|tool_call></ifm|tool_calls>");
  CHECK_EQ(o2.tool_calls.size(), 1U);
  // Qwen keeps its parser and its "<think>\n" rule.
  auto p3 = server::make_output_parser(ChatFormat{}, "x<think>\n", json::object(), nullptr);
  CHECK_EQ(run(*p3, "r\n</think>\n\nc").reasoning, std::string("r"));
  // The kwargs check: bad values are named.
  bool threw = false;
  try {
    server::check_template_kwargs(k2, json({{"tool_call_format", "yaml"}}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    server::check_template_kwargs(k2, json({{"reasoning_effort", "max"}}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  server::check_template_kwargs(k2, json({{"tool_call_format", "xml_typed"}, {"reasoning_effort", "low"}}));
  server::check_template_kwargs(ChatFormat{}, json({{"tool_call_format", "yaml"}}));   // Qwen: not read
}

// --- the fuzz --------------------------------------------------------------------------------
struct Gen {
  std::mt19937 rng;
  explicit Gen(uint32_t seed) : rng(seed) {}
  size_t below(size_t n) { return rng() % n; }
  // Text drawn from fragments that resemble the tags without being them.
  std::string text(size_t max_parts) {
    static const char* parts[] = {"a", "b c", " ", "\n", "\t", "<", ">", "/", "ifm|", "<ifm|", "</ifm|", "think",
                                  "tool_call", "arg_", "value>", "{", "}", "\"", "x=1", "é", "日本", "<b>", "</"};
    std::string s;
    const size_t n = below(max_parts + 1);
    for (size_t i = 0; i < n; ++i) s += parts[below(sizeof(parts) / sizeof(parts[0]))];
    return s;
  }
};

bool has_any(const std::string& s, std::initializer_list<const char*> tags) {
  for (const char* t : tags) {
    if (s.find(t) != std::string::npos) return true;
  }
  return false;
}

void case_fuzz() {
  const json tools = json::parse(R"([
    {"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{
      "a":{"type":"string"},"b":{"type":"string"},"c":{"type":"string"}}}}},
    {"type":"function","function":{"name":"g","parameters":{"type":"object","properties":{
      "a":{"type":"string"},"b":{"type":"string"}}}}}])");
  Gen gen(2026);
  size_t exact = 0, cut = 0;
  for (int iter = 0; iter < 3000; ++iter) {
    const K2CallFormat format = static_cast<K2CallFormat>(gen.below(3));
    const bool thinking = gen.below(4) != 0;
    std::string reasoning, before, after;
    do reasoning = gen.text(8);
    while (has_any(reasoning, {"</ifm|think>", "<ifm|tool_calls>"}) || reasoning.rfind("<ifm|think>", 0) == 0);
    do before = gen.text(6);
    while (has_any(before, {"<ifm|tool_call"}) || (!thinking && false));
    do after = gen.text(4);
    while (has_any(after, {"<ifm|tool_call"}));
    json calls = json::array();
    std::string block;
    const size_t ncalls = gen.below(4);
    if (ncalls > 0) block = "<ifm|tool_calls>";
    for (size_t i = 0; i < ncalls; ++i) {
      const std::string name = gen.below(2) ? "f" : "g";
      json args = json::object();
      const size_t nargs = gen.below(3);
      std::string body = format == K2CallFormat::Json ? "" : name + "\n";
      for (size_t k = 0; k < nargs; ++k) {
        const std::string key = std::string(1, char('a' + k));
        std::string value;
        do value = gen.text(5);
        while (has_any(value, {"<ifm|arg_", "</ifm|arg_", "</ifm|tool_call", "<ifm|tool_call"}));
        args[key] = value;
        if (format != K2CallFormat::Json) {
          body += "<ifm|arg_key>" + key + "</ifm|arg_key>\n";
          if (format == K2CallFormat::XmlTyped) body += "<ifm|arg_type>string</ifm|arg_type>\n";
          body += "<ifm|arg_value>" + value + "</ifm|arg_value>\n";
        }
      }
      if (format == K2CallFormat::Json) body = json({{"name", name}, {"arguments", args}}).dump();
      block += "\n<ifm|tool_call>" + body + "</ifm|tool_call>";
      calls.push_back({{"name", name}, {"arguments", args}});
    }
    if (ncalls > 0) block += "\n</ifm|tool_calls>";
    const std::string text = (thinking ? reasoning + "</ifm|think>" : "") + before + block + after;
    const std::string open = thinking ? "<ifm|think>" : "";
    if (gen.below(3) == 0) {   // a random cut: only the streaming rule is checked
      parse_checked(text.substr(0, gen.below(text.size() + 1)), open, format, tools);
      ++cut;
      continue;
    }
    const ParsedOutput p = parse_checked(text, open, format, tools);
    const std::string content = before + after;
    const bool blank = content.find_first_not_of(" \n\t\r") == std::string::npos;
    bool ok = p.reasoning == (thinking ? reasoning : "") && p.content == (blank ? "" : content) &&
              p.tool_calls.size() == calls.size();
    for (size_t i = 0; ok && i < calls.size(); ++i) {
      ok = p.tool_calls[i].name == calls[i].at("name") && p.tool_calls[i].arguments == calls[i].at("arguments");
    }
    if (!ok) {
      std::fprintf(stderr, "fuzz %d (format %d): %s\n  from %s\n", iter, int(format), show(p).c_str(),
                   json(text).dump().c_str());
      CHECK(false);
    }
    ++exact;
  }
  std::printf("fuzz: %zu generated outputs parse to their structure, %zu cut ones stream = whole\n", exact, cut);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  case_round_trip(dir);
  case_reasoning();
  case_xml();
  case_xml_typed();
  case_json();
  case_edges();
  case_truncated();
  case_factory();
  case_fuzz();
  std::printf("toolcall_k2_test OK: 9 cases\n");
  return 0;
}

// Spec 20e: Kolibri-1's reasoning and hermes JSON tool-call parser (server/toolcall_kolibri.h).
//   toolcall_kolibri_test <tests/tokenizer>
// Every parse is checked whole, byte at a time and in random 1-7 byte pieces (spec 7a's streaming
// rule: the deltas of any split concatenate to the whole-text parse). Cases:
//   - the round trip: every assistant turn of the HF renders of kolibri_template_cases.json (reasoning
//     before and after the last query, content, one and two calls, German arguments, <think> inside
//     content) parses back to the message the template rendered;
//   - Review Focus 1: reasoning opened by the model (with thinking on the prompt ends in
//     "<|im_start|>assistant\n"), thinking off (the output is the answer), whitespace before <think>,
//     reasoning cut by the end of generation, a lone </think>, <think> later in the text;
//   - Review Focus 2: a call that is not valid JSON (a missing brace) or not a call (no name, a name
//     with whitespace, array arguments) becomes content verbatim with its tags; a call cut by
//     max_tokens becomes content; "arguments" as a JSON string holding an object; two calls separated
//     by "\n"; content around calls; schema typing; whitespace-only content dropped;
//   - Review Focus 5: German argument strings keep their bytes (also from \u escapes), and the
//     server's arguments text (nlohmann's dump) writes them as UTF-8, not re-escaped;
//   - the shared parse_json_call (also K2's json format - toolcall_k2_test keeps K2's cases);
//   - a fuzz: 2000 generated outputs (reasoning or not, content with tag fragments, 0-3 valid calls
//     with random arguments) parse to the generated structure, and the same outputs cut at a random
//     byte, with broken calls spliced in, parse identically under every split.
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
#include "server/toolcall_kolibri.h"

namespace {

using json = nlohmann::json;
using server::Delta;
using server::KolibriOutputStream;
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

ParsedOutput stream(const std::string& text, const json& tools, const std::vector<size_t>& sizes) {
  KolibriOutputStream s(tools);
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

uint32_t g_seed = 11;
size_t g_parses = 0;

// The whole-text parse equals byte-at-a-time and random 1-7 byte streaming.
ParsedOutput parse_checked(const std::string& text, const json& tools) {
  const ParsedOutput whole = server::parse_output_kolibri(text, tools);
  for (const server::ToolCall& c : whole.tool_calls) {
    CHECK(c.arguments.is_object());
    CHECK_EQ(c.id.size(), 29U);
    CHECK(c.id.rfind("call_", 0) == 0);
  }
  const ParsedOutput bytes = stream(text, tools, {1});
  if (!same(whole, bytes)) {
    std::fprintf(stderr, "byte stream differs for %s\n  whole: %s\n  bytes: %s\n", json(text).dump().c_str(),
                 show(whole).c_str(), show(bytes).c_str());
    CHECK(false);
  }
  std::mt19937 rng(g_seed++);
  std::vector<size_t> sizes(64);
  for (size_t& n : sizes) n = 1 + rng() % 7;
  const ParsedOutput pieces = stream(text, tools, sizes);
  if (!same(whole, pieces)) {
    std::fprintf(stderr, "random pieces differ for %s\n  whole: %s\n  split: %s\n", json(text).dump().c_str(),
                 show(whole).c_str(), show(pieces).c_str());
    CHECK(false);
  }
  ++g_parses;
  return whole;
}

void expect(const std::string& text, const json& tools, const std::string& reasoning, const std::string& content,
            const json& calls) {
  const ParsedOutput p = parse_checked(text, tools);
  bool ok = p.reasoning == reasoning && p.content == content && p.tool_calls.size() == calls.size();
  for (size_t i = 0; ok && i < calls.size(); ++i)
    ok = p.tool_calls[i].name == calls[i].at(0) && p.tool_calls[i].arguments == calls[i].at(1);
  if (!ok) {
    std::fprintf(stderr, "parse of %s\n  got:  %s\n  want: %s\n", json(text).dump().c_str(), show(p).c_str(),
                 json({{"reasoning", reasoning}, {"content", content}, {"calls", calls}}).dump().c_str());
    CHECK(false);
  }
}

const json kTools = json::parse(R"([
  {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{
    "filePath":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}}}}},
  {"type":"function","function":{"name":"grep","parameters":{"type":"object","properties":{
    "pattern":{"type":"string"},"path":{"type":"string"},"include":{"type":["string","null"]}}}}},
  {"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"integer"}}}}}
])");

// --- the round trip through the HF renders --------------------------------------------------------
std::string strip_nl(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && s[b] == '\n') ++b;
  while (e > b && s[e - 1] == '\n') --e;
  return s.substr(b, e - b);
}

void case_round_trip(const std::string& dir) {
  const json spec = json::parse(slurp(dir + "/kolibri_template_cases.json"));
  const std::string turn = "<|im_start|>assistant\n";
  size_t turns = 0, calls = 0, with_reasoning = 0;
  for (const json& c : spec.at("cases")) {
    const std::string name = c.at("name");
    json tools = nullptr;
    if (c.at("tools").is_array()) {
      tools = json::array();
      for (const json& n : c.at("tools")) tools.push_back(spec.at("tools").at(n.get<std::string>()));
    }
    const std::string text = slurp(dir + "/kolibri_template_" + name + ".txt");
    std::vector<const json*> assistants;
    for (const json& m : c.at("messages"))
      if (m.at("role") == "assistant") assistants.push_back(&m);
    size_t at = 0, k = 0;
    for (size_t i = text.find(turn); i != std::string::npos; i = text.find(turn, at)) {
      const size_t body = i + turn.size();
      const size_t end = text.find("<|im_end|>", body);
      if (end == std::string::npos) break;   // the generation prompt
      // What the model generated for this turn is everything after the generation prompt.
      const std::string generated = text.substr(body, end - body);
      CHECK(k < assistants.size());
      const json& m = *assistants[k++];
      const ParsedOutput p = parse_checked(generated, tools);
      // The template's own reading of the message (its reasoning / content split).
      std::string content = m.at("content").is_string() ? m.at("content").get<std::string>() : "";
      std::string reasoning;
      if (m.contains("reasoning") && m.at("reasoning").is_string()) {
        reasoning = m.at("reasoning");
      } else if (m.contains("reasoning_content") && m.at("reasoning_content").is_string()) {
        reasoning = m.at("reasoning_content");
      } else if (content.find("</think>") != std::string::npos) {
        const std::string before = content.substr(0, content.find("</think>"));
        reasoning = before.substr(before.rfind("<think>") == std::string::npos ? 0 : before.rfind("<think>") + 7);
        content = content.substr(content.rfind("</think>") + 8);
      }
      while (!content.empty() && content.front() == '\n') content.erase(0, 1);
      // A turn before the last user query is rendered without its reasoning.
      const bool rendered = generated.rfind("<think>\n", 0) == 0;
      const std::string want_reasoning = rendered ? strip_nl(reasoning) : "";
      bool ok = p.reasoning == want_reasoning && p.content == content;
      const json want_calls = m.contains("tool_calls") ? m.at("tool_calls") : json::array();
      ok = ok && p.tool_calls.size() == want_calls.size();
      for (size_t j = 0; ok && j < want_calls.size(); ++j)
        ok = p.tool_calls[j].name == want_calls[j].at("function").at("name") &&
             p.tool_calls[j].arguments == want_calls[j].at("function").at("arguments");
      if (!ok) {
        std::fprintf(stderr, "round trip %s turn %zu: %s\n  from %s\n", name.c_str(), k, show(p).c_str(),
                     json(generated).dump().c_str());
        CHECK(false);
      }
      calls += p.tool_calls.size();
      with_reasoning += rendered ? 1 : 0;
      ++turns;
      at = end;
    }
    CHECK_EQ(k, assistants.size());
  }
  CHECK_EQ(turns, 8U);
  CHECK_EQ(calls, 5U);
  CHECK_EQ(with_reasoning, 7U);
  std::printf("round trip: %zu assistant turns (%zu with reasoning), %zu calls parse back to their messages\n",
              turns, with_reasoning, calls);
}

// --- Review Focus 1: reasoning opened by the model -------------------------------------------------
void case_reasoning() {
  const json none = json::array();
  // Thinking on: the output opens <think>; the newline after it and the whitespace before </think>
  // are the template's, not the reasoning's; the blank line after </think> is not content.
  expect("<think>\nLet me think.\nOk.\n</think>\n\nThe answer is 4.", nullptr, "Let me think.\nOk.",
         "The answer is 4.", none);
  // Thinking off (the prompt ended in "<think>\n\n</think>\n\n"): the output is the answer.
  expect("The answer is 4.", nullptr, "", "The answer is 4.", none);
  // Whitespace before <think> is not content.
  expect("\n \n<think>\nx\n</think>\n\ny", nullptr, "x", "y", none);
  // Reasoning cut by the end of generation is reasoning (trailing whitespace dropped).
  expect("<think>\nstill thinking </thi", nullptr, "still thinking </thi", "", none);
  expect("<think>\nstill thinking\n\n", nullptr, "still thinking", "", none);
  // A prefix of <think> at the very end is content; an empty reasoning block is none.
  expect("<thi", nullptr, "", "<thi", none);
  expect("<think></think>answer", nullptr, "", "answer", none);
  expect("<think>\n\n</think>\n\nanswer", nullptr, "", "answer", none);
  // A close tag without an open one, and <think> after other text, are content.
  expect("plain </think> text", nullptr, "", "plain </think> text", none);
  expect("Sure. <think>no</think> done", nullptr, "", "Sure. <think>no</think> done", none);
  // A call inside reasoning is reasoning text (calls live in the body).
  expect("<think>\nmaybe <tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>\n</think>\n\nok", kTools,
         "maybe <tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>", "ok", none);
  // Reasoning keeps inner whitespace verbatim.
  expect("<think>\n  a\n\n  b  \n</think>\n\nc", nullptr, "  a\n\n  b", "c", none);
  std::printf("reasoning: opened by the model, thinking off, whitespace before <think>, cut, lone tags\n");
}

// --- Review Focus 2: calls that are not valid JSON, or cut -----------------------------------------
void case_calls() {
  const json none = json::array();
  // The template's own form: content, "\n", the call.
  expect("<think>\nRead it.\n</think>\n\nReading.\n<tool_call>\n{\"name\": \"read\", \"arguments\": "
         "{\"filePath\": \"/w/a.cc\", \"offset\": 12}}\n</tool_call>",
         kTools, "Read it.", "Reading.", json::array({json::array({"read", {{"filePath", "/w/a.cc"}, {"offset", 12}}})}));
  // A missing brace: content, verbatim with its tags and the newline before it.
  const std::string broken = "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}\n</tool_call>";
  expect("Calling.\n" + broken, kTools, "", "Calling.\n" + broken, none);
  expect(broken, kTools, "", broken, none);
  // Cut by max_tokens inside the call: content.
  expect("Ok.\n<tool_call>\n{\"name\": \"read\", \"argu", kTools, "", "Ok.\n<tool_call>\n{\"name\": \"read\", \"argu",
         none);
  expect("<tool_call>", kTools, "", "<tool_call>", none);
  expect("x <tool_ca", kTools, "", "x <tool_ca", none);
  // "arguments" as a JSON string holding an object; absent; null.
  expect("<tool_call>\n{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}\n</tool_call>", kTools, "", "",
         json::array({json::array({"f", {{"a", 1}}})}));
  expect("<tool_call>{\"name\": \"f\"}</tool_call>", kTools, "", "", json::array({json::array({"f", json::object()})}));
  expect("<tool_call>{\"name\": \"f\", \"arguments\": null}</tool_call>", kTools, "", "",
         json::array({json::array({"f", json::object()})}));
  // Not a call: no name, a name with whitespace, array arguments, a string that is not an object.
  for (const std::string& body : {std::string("{\"arguments\": {}}"), std::string("{\"name\": \"a b\", \"arguments\": {}}"),
                                  std::string("{\"name\": \"f\", \"arguments\": [1]}"),
                                  std::string("{\"name\": \"f\", \"arguments\": \"[1]\"}"), std::string("not json")}) {
    const std::string raw = "<tool_call>" + body + "</tool_call>";
    expect(raw, kTools, "", raw, none);
  }
  // Two calls separated by "\n" (the template's parallel form): no content between them.
  expect("<tool_call>\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"TODO\"}}\n</tool_call>\n<tool_call>\n"
         "{\"name\": \"read\", \"arguments\": {\"filePath\": \"/r\"}}\n</tool_call>",
         kTools, "", "",
         json::array({json::array({"grep", {{"pattern", "TODO"}}}), json::array({"read", {{"filePath", "/r"}}})}));
  // A valid call, then a broken one: one call, the broken one content.
  expect("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 2}}</tool_call>\n<tool_call>{\"name\": </tool_call>",
         kTools, "", "<tool_call>{\"name\": </tool_call>", json::array({json::array({"f", {{"a", 2}}})}));
  // Content after a call; whitespace after a call dropped.
  expect("<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>\n\nDone.", kTools, "", "Done.",
         json::array({json::array({"f", json::object()})}));
  // The first </tool_call> closes (vLLM's hermes regex is non-greedy).
  expect("<tool_call>{\"name\": \"read\", \"arguments\": {\"filePath\": \"</tool_call>\"}}</tool_call>", kTools, "",
         "<tool_call>{\"name\": \"read\", \"arguments\": {\"filePath\": \"</tool_call>\"}}</tool_call>", none);
  // Schema typing: a schema string written as a number is its JSON text; integers stay integers;
  // a string | null schema written as null becomes "null" (the rule K2's json format has).
  expect("<tool_call>{\"name\": \"read\", \"arguments\": {\"filePath\": 123, \"limit\": 5}}</tool_call>", kTools, "",
         "", json::array({json::array({"read", {{"filePath", "123"}, {"limit", 5}}})}));
  expect("<tool_call>{\"name\": \"grep\", \"arguments\": {\"pattern\": \"x\", \"include\": null}}</tool_call>", kTools,
         "", "", json::array({json::array({"grep", {{"pattern", "x"}, {"include", "null"}}})}));
  // Whitespace-only content is dropped; content whitespace at the very end is kept.
  expect("\n\n", nullptr, "", "", none);
  expect("<think>\nx\n</think>\n\n", nullptr, "x", "", none);
  expect("Hello\n", nullptr, "", "Hello\n", none);
  expect("  Hello", nullptr, "", "  Hello", none);
  std::printf("calls: valid, broken (verbatim with tags), cut, string arguments, two calls, schema typing\n");
}

// --- Review Focus 5: German bytes -------------------------------------------------------------------
void case_german() {
  const std::string pattern = "Größe „Straße“ – Donaudampfschifffahrtsgesellschaftskapitän ẞ";
  const json args = {{"pattern", pattern}, {"path", "/w/maß"}};
  const std::string text = "<think>\nIch suche „Größe“.\n</think>\n\nIch suche nach „Größe“.\n<tool_call>\n" +
                           json({{"name", "grep"}, {"arguments", args}}).dump() + "\n</tool_call>";
  const ParsedOutput p = parse_checked(text, kTools);
  CHECK_EQ(p.reasoning, std::string("Ich suche „Größe“."));
  CHECK_EQ(p.content, std::string("Ich suche nach „Größe“."));
  CHECK_EQ(p.tool_calls.size(), 1U);
  CHECK_EQ(p.tool_calls[0].arguments.at("pattern").get<std::string>(), pattern);
  // The server sends arguments as call.arguments.dump(): UTF-8 bytes, not \u escapes.
  const std::string dumped = p.tool_calls[0].arguments.dump();
  CHECK(dumped.find("Größe „Straße“") != std::string::npos);
  CHECK(dumped.find("\\u") == std::string::npos);
  // \u escapes in the model's JSON decode to the same bytes.
  const ParsedOutput q =
      parse_checked("<tool_call>{\"name\": \"grep\", \"arguments\": {\"pattern\": \"Gr\\u00f6\\u00dfe\"}}</tool_call>", kTools);
  CHECK_EQ(q.tool_calls.size(), 1U);
  CHECK_EQ(q.tool_calls[0].arguments.at("pattern").get<std::string>(), std::string("Größe"));
  std::printf("german: umlauts, ß, „…“ and a 42-letter compound keep their bytes in reasoning, content and arguments\n");
}

// --- parse_json_call, shared with K2 ----------------------------------------------------------------
void case_parse_json_call() {
  server::ToolCall c;
  CHECK(server::parse_json_call("  {\"name\": \"f\", \"arguments\": {\"a\": 1}}\n", kTools, c));
  CHECK_EQ(c.name, std::string("f"));
  CHECK_EQ(c.arguments, json({{"a", 1}}));
  CHECK(c.id.empty());   // ids are the stream's
  CHECK(!server::parse_json_call("{\"name\": \"\", \"arguments\": {}}", kTools, c));
  CHECK(!server::parse_json_call("{\"name\": 3}", kTools, c));
  CHECK(!server::parse_json_call("[]", kTools, c));
  CHECK(!server::parse_json_call("{\"name\": \"f\", \"arguments\": \"nope\"}", kTools, c));
  std::printf("parse_json_call: the shared JSON body rule\n");
}

// --- the chat format's dispatch ----------------------------------------------------------------------
void case_format() {
  const server::ChatFormat f = server::ChatFormat::for_model_type("kolibri1");
  CHECK(f.kind == server::ChatFormat::Kind::Kolibri);
  CHECK(f.template_kwargs());
  CHECK(f.string_content());
  CHECK_EQ(std::string(f.name()), std::string("kolibri1"));
  CHECK(server::ChatFormat::for_model_type("qwen3_5").kind == server::ChatFormat::Kind::Qwen);
  CHECK(!server::ChatFormat::for_model_type("k2_horizon").string_content());
  // The parser ignores the prompt's ending (the model opens reasoning itself).
  auto parser = server::make_output_parser(f, "<|im_start|>assistant\n", json::object(), nullptr);
  std::vector<Delta> d = parser->push("<think>\nr\n</think>\n\nc");
  const std::vector<Delta> tail = parser->finish();
  d.insert(d.end(), tail.begin(), tail.end());
  const ParsedOutput p = collect(d);
  CHECK_EQ(p.reasoning, std::string("r"));
  CHECK_EQ(p.content, std::string("c"));
  // Kwargs: the seven efforts and null pass, anything else is the client's error.
  for (const char* e : {"none", "minimal", "low", "medium", "high", "xhigh", "max"})
    server::check_template_kwargs(f, {{"reasoning_effort", e}});
  server::check_template_kwargs(f, {{"reasoning_effort", nullptr}, {"preserve_thinking", true}});
  for (const json& bad : {json({{"reasoning_effort", "extreme"}}), json({{"reasoning_effort", 3}}),
                          json({{"preserve_thinking", "yes"}})}) {
    bool threw = false;
    try {
      server::check_template_kwargs(f, bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  // List content becomes a string (the template concatenates content).
  const json msgs = json::array({{{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", "a"}},
                                                                              {{"type", "text"}, {"text", "b"}}})}},
                                 {{"role", "assistant"}, {"content", nullptr}}});
  const json flat = server::template_messages(f, msgs);
  CHECK_EQ(flat.at(0).at("content"), json("a\nb"));
  CHECK(flat.at(1).at("content").is_null());
  CHECK_EQ(server::template_messages(server::ChatFormat{}, msgs), msgs);
  std::printf("format: kolibri1 -> Kolibri (kwargs, string content, the output parser)\n");
}

// --- the fuzz ----------------------------------------------------------------------------------------
std::string pick(std::mt19937& rng, const std::vector<std::string>& from) { return from[rng() % from.size()]; }

// Text that never forms a tag, starting and ending with a non-whitespace byte.
std::string words(std::mt19937& rng, size_t n) {
  static const std::vector<std::string> alphabet = {"a", "Größe", " ", "\n", "x y", "<too", "<thi", "</thin", "<",
                                                    ">", "{", "}", "\"", "ß", "„", "“", "tool_calls", "thinks", "\t"};
  std::string s = pick(rng, {"A", "Ö", "z", "1"});
  for (size_t i = 0; i < n; ++i) s += pick(rng, alphabet);
  return s + pick(rng, {".", "!", "b", "ä"});
}

json random_args(std::mt19937& rng) {
  json a = json::object();
  const size_t n = rng() % 4;
  for (size_t i = 0; i < n; ++i) {
    const std::string key = pick(rng, {"a", "path", "pattern", "n", "flag", "nested"});
    switch (rng() % 4) {
      case 0: a[key] = words(rng, rng() % 6); break;
      case 1: a[key] = int(rng() % 1000) - 500; break;
      case 2: a[key] = bool(rng() % 2); break;
      default: a[key] = {{"k", words(rng, 2)}, {"v", json::array({1, "two"})}};
    }
  }
  return a;
}

void case_fuzz() {
  std::mt19937 rng(20251006);
  size_t calls = 0, cut = 0;
  for (int it = 0; it < 2000; ++it) {
    std::string text, reasoning, content;
    json want = json::array();
    if (rng() % 2) {
      reasoning = words(rng, rng() % 8);
      text += pick(rng, {"", "\n", " \n"}) + "<think>\n" + reasoning + pick(rng, {"\n", "", "\n\n "}) + "</think>\n\n";
    }
    const size_t n = rng() % 4;
    if (n == 0 || rng() % 2) {
      content = words(rng, rng() % 8);
      text += content;
      if (n) text += "\n";
    }
    for (size_t i = 0; i < n; ++i) {
      const std::string name = pick(rng, {"read", "grep", "edit_file", "now"});
      const json args = random_args(rng);
      text += (i ? "\n" : "") + std::string("<tool_call>\n") + json({{"name", name}, {"arguments", args}}).dump() +
              "\n</tool_call>";
      want.push_back(json::array({name, args}));
    }
    if (n && rng() % 3 == 0) {
      const std::string after = words(rng, rng() % 4);
      text += "\n" + after;
      content += after;
    }
    expect(text, nullptr, reasoning, content, want);
    calls += n;
    // The same output cut at a random byte, or with a broken call spliced in: every split equals the whole.
    std::string other = text.substr(0, rng() % (text.size() + 1));
    if (rng() % 2) other += "\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"a\": 1}\n</tool_call>" + words(rng, 2);
    (void)parse_checked(other, kTools);
    ++cut;
  }
  std::printf("fuzz: 2000 outputs (%zu calls) parse to their structure; %zu cut / broken variants split-invariant\n",
              calls, cut);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  case_round_trip(dir);
  case_reasoning();
  case_calls();
  case_german();
  case_parse_json_call();
  case_format();
  case_fuzz();
  std::printf("toolcall_kolibri_test OK: %zu parses, each whole = byte stream = random pieces\n", g_parses);
  return 0;
}

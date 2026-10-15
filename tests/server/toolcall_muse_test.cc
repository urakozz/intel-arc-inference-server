// Spec 23e Task 2: Muse-Glimmer-30B's channel / ATEM output parser (server/toolcall_muse.h).
//   toolcall_muse_test <tests/server>
// Every parse is checked whole, byte at a time and in random 1-7 byte pieces (spec 7a's streaming rule: the deltas
// of any split concatenate to the whole-text parse). The text a case feeds is what the server feeds: the output after
// the prompt's `<|start|>assistant`, WITHOUT the EOS id's text (<|eot|> / <|end_of_text|> stop the generation and
// are never pushed - server.cc); <|eom|> is not EOS and stays.
//   (a) muse_parse_cases.json (tools/tokenizer/dump_muse.py parse - transformers 5.19.0's
//       tokenizer.parse_response(text, prefix=prompt, tools=...) on complete turns): every case equals
//       transformers' message except the documented differences, where the case's `ours` is the expected value -
//       reasoning_join (decision 2: every to=self block, joined with "\n"; transformers keeps the last - also
//       compared by its last block), typing (decision 15: values typed by the tool's schema first), name_collapse
//       (§6.5: ns.ns -> a registered bare ns); the assistant turns of the renders parse back to their messages;
//   (b) damaged turns (vLLM's parsers' documented cases): a to=self body without <|eom|> followed by a bare
//       ` to=<tool><|message|>` header (the call survives), <atem:invoke> echoed in reasoning / in to=user content
//       (not a call), a truncated invoke (content only), no framing at all (the whole text is content), the doubled
//       name collapsed / an unknown leaf passed through, the invoke's name winning over the recipient; generation
//       cut inside <atem:parameter>, inside `<atem:invoke name="`, and after </atem:invoke> before
//       </atem:function_calls> (Review Focus 6: a complete invoke is a call, an incomplete one is neither a call nor
//       content);
//   (c) Review Focus 2 (` to=self` split across pieces never leaks), 3 (<|eom|> ends a message, a reasoning block
//       then calls is one generation), 7 (schema-first typing, the invoke's name), a piece ending in `<|` held back;
//   (d) a fuzz: 2000 random outputs (channel sequences, values, markers inside values, cuts) - every whole parse
//       equals its byte-at-a-time and 1-7-byte streams.
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
#include "server/toolcall_muse.h"

namespace {

using json = nlohmann::json;
using server::Delta;
using server::MuseOutputStream;
using server::MuseParsed;

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

MuseParsed collect(const std::vector<Delta>& deltas) {
  MuseParsed out;
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

bool same(const MuseParsed& a, const MuseParsed& b) {
  if (a.reasoning != b.reasoning || a.content != b.content) return false;
  if (a.tool_calls.size() != b.tool_calls.size()) return false;
  for (size_t i = 0; i < a.tool_calls.size(); ++i) {
    if (a.tool_calls[i].name != b.tool_calls[i].name) return false;
    if (a.tool_calls[i].arguments != b.tool_calls[i].arguments) return false;
  }
  return true;
}

json calls_json(const MuseParsed& p) {
  json calls = json::array();
  for (const auto& c : p.tool_calls) calls.push_back({{"name", c.name}, {"arguments", c.arguments}});
  return calls;
}

std::string show(const MuseParsed& p) {
  return json({{"reasoning", p.reasoning}, {"content", p.content}, {"calls", calls_json(p)}}).dump();
}

MuseParsed stream(const std::string& text, const json& tools, const std::vector<size_t>& sizes) {
  MuseOutputStream s(tools);
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

uint32_t g_seed = 23;
size_t g_parses = 0;

// The whole-text parse equals byte-at-a-time and random 1-7 byte streaming.
MuseParsed parse_checked(const std::string& text, const json& tools) {
  const MuseParsed whole = server::parse_muse_output(text, tools);
  for (const server::ToolCall& c : whole.tool_calls) {
    CHECK(c.arguments.is_object());
    CHECK_EQ(c.id.size(), 29U);
    CHECK(c.id.rfind("call_", 0) == 0);
  }
  const MuseParsed bytes = stream(text, tools, {1});
  if (!same(whole, bytes)) {
    std::fprintf(stderr, "byte stream differs for %s\n  whole: %s\n  bytes: %s\n", json(text).dump().c_str(),
                 show(whole).c_str(), show(bytes).c_str());
    CHECK(false);
  }
  std::mt19937 rng(g_seed++);
  std::vector<size_t> sizes(64);
  for (size_t& n : sizes) n = 1 + rng() % 7;
  const MuseParsed pieces = stream(text, tools, sizes);
  if (!same(whole, pieces)) {
    std::fprintf(stderr, "random pieces differ for %s\n  whole: %s\n  split: %s\n", json(text).dump().c_str(),
                 show(whole).c_str(), show(pieces).c_str());
    CHECK(false);
  }
  ++g_parses;
  return whole;
}

// The server never feeds the EOS id's text.
std::string fed(std::string text) {
  for (const char* eos : {"<|eot|>", "<|end_of_text|>"}) {
    const std::string e = eos;
    if (text.size() >= e.size() && text.compare(text.size() - e.size(), e.size(), e) == 0) text.erase(text.size() - e.size());
  }
  return text;
}

void expect(const char* what, const MuseParsed& got, const std::string& reasoning, const std::string& content,
            const json& calls) {
  if (got.reasoning != reasoning || got.content != content || calls_json(got) != calls) {
    std::fprintf(stderr, "%s:\n  got : %s\n  want: %s\n", what, show(got).c_str(),
                 json({{"reasoning", reasoning}, {"content", content}, {"calls", calls}}).dump().c_str());
    CHECK(false);
  }
}

std::string last_block(const std::string& joined) {
  const size_t nl = joined.rfind('\n');
  return nl == std::string::npos ? joined : joined.substr(nl + 1);
}

// ---- (a) transformers' parse_response on complete turns ------------------------------------------------------------
void case_fixture(const std::string& dir) {
  const json facts = json::parse(slurp(dir + "/muse_parse_cases.json"));
  size_t differs = 0, round_trips = 0;
  for (const json& c : facts.at("cases")) {
    const std::string name = c.at("name");
    const json& tools = c.at("tools");
    const MuseParsed got = parse_checked(fed(c.at("text")), tools);
    const json& tf = c.at("transformers");
    std::string reasoning = tf.at("reasoning"), content = tf.at("content");
    json calls = tf.at("tool_calls");
    if (c.contains("ours")) {
      ++differs;
      const json& ours = c.at("ours");
      for (const json& d : ours.at("differs")) {
        const std::string why = d;
        if (why == "reasoning_join") {
          // transformers keeps the LAST to=self block; ours joins them all with "\n" (decision 2).
          CHECK_EQ(last_block(ours.at("reasoning").get<std::string>()), reasoning);
          reasoning = ours.at("reasoning");
        } else if (why == "typing") {
          for (size_t i = 0; i < calls.size(); ++i) calls[i]["arguments"] = ours.at("arguments").at(i);
        } else if (why == "name_collapse") {
          for (size_t i = 0; i < calls.size(); ++i) calls[i]["name"] = ours.at("names").at(i);
        } else {
          std::fprintf(stderr, "%s: unknown difference %s\n", name.c_str(), why.c_str());
          CHECK(false);
        }
      }
    }
    char what[160];
    std::snprintf(what, sizeof what, "fixture %s", name.c_str());
    expect(what, got, reasoning, content, calls);
    // The renders' assistant turns parse back to their messages.
    if (c.contains("messages")) {
      std::string r, t;
      json want_calls = json::array();
      for (const json& m : c.at("messages")) {
        if (m.contains("reasoning_content") && !m.at("reasoning_content").get<std::string>().empty())
          r += (r.empty() ? "" : "\n") + m.at("reasoning_content").get<std::string>();
        if (m.contains("content") && m.at("content").is_string() && !m.at("content").get<std::string>().empty())
          t += (t.empty() ? "" : "\n") + m.at("content").get<std::string>();
        if (m.contains("tool_calls"))
          for (const json& tc : m.at("tool_calls"))
            want_calls.push_back({{"name", tc.at("function").at("name")}, {"arguments", tc.at("function").at("arguments")}});
      }
      std::snprintf(what, sizeof what, "round trip %s", name.c_str());
      expect(what, got, r, t, want_calls);
      ++round_trips;
    }
  }
  CHECK(facts.at("cases").size() >= 20U);
  std::printf("fixture: %zu cases equal transformers 5.19.0's parse_response (%zu by a documented difference: "
              "decisions 2 / 15, the ns.ns collapse); %zu rendered turns parse back to their messages\n",
              facts.at("cases").size(), differs, round_trips);
}

// ---- (b) damaged turns, (c) the Review Focus cases ------------------------------------------------------------
std::string invoke(const std::string& name, const std::vector<std::pair<std::string, std::string>>& params) {
  std::string s = "<atem:function_calls>\n<atem:invoke name=\"" + name + "\">\n";
  for (const auto& [k, v] : params) s += "<atem:parameter name=\"" + k + "\">" + v + "</atem:parameter>\n";
  return s + "</atem:invoke>\n</atem:function_calls>";
}

json tool(const std::string& name, const json& props) {
  return {{"type", "function"},
          {"function", {{"name", name}, {"description", ""}, {"parameters", {{"type", "object"}, {"properties", props}}}}}};
}

void case_damaged() {
  const json str = {{"type", "string"}};
  const json tools = json::array({tool("fs.read_file", {{"path", str}, {"offset", {{"type", "integer"}}}}),
                                  tool("read_file", {{"path", str}}), tool("calendar.get", {{"id", str}})});
  const std::string call_fs = invoke("fs.read_file", {{"path", "a.cc"}, {"offset", "3"}});
  const json fs_call = json::array({{{"name", "fs.read_file"}, {"arguments", {{"path", "a.cc"}, {"offset", 3}}}}});

  // A reasoning block with no <|eom|>, then a bare header: the call survives.
  expect("unterminated reasoning, bare tool header",
         parse_checked(" to=self<|message|>I will read it. to=fs.read_file<|message|>" + call_fs, tools),
         "I will read it.", "", fs_call);
  // ... and the same with the next message's <|start|> but no <|eom|>.
  expect("unterminated reasoning, <|start|> header",
         parse_checked(" to=self<|message|>Plan<|start|>assistant to=fs.read_file<|message|>" + call_fs, tools), "Plan",
         "", fs_call);
  // An invoke echoed in reasoning / content is not a call.
  expect("invoke in reasoning",
         parse_checked(" to=self<|message|>maybe " + call_fs + "<|eom|><|start|>assistant to=user<|message|>no", tools),
         "maybe " + call_fs, "no", json::array());
  expect("invoke in content", parse_checked(" to=user<|message|>" + call_fs, tools), "", call_fs, json::array());
  // A truncated invoke: content only, no call, no markup.
  expect("truncated invoke",
         parse_checked(" to=user<|message|>Answer<|eom|><|start|>assistant to=fs.read_file<|message|><atem:function_calls>"
                       "\n<atem:invoke name=\"fs.read_file\">\n<atem:param", tools),
         "", "Answer", json::array());
  // No framing at all: the whole text is content.
  expect("no framing", parse_checked("Just text, <atem:invoke name=\"x\"> and all.", tools), "",
         "Just text, <atem:invoke name=\"x\"> and all.", json::array());
  expect("bare <|message|> header", parse_checked("<|message|>Untagged content.", tools), "", "Untagged content.",
         json::array());
  // Names: ns.ns collapsed to a registered bare ns; an unknown leaf passed through; the invoke's name wins.
  expect("doubled name",
         parse_checked(" to=read_file.read_file<|message|>" + invoke("read_file.read_file", {{"path", "x"}}), tools), "", "",
         json::array({{{"name", "read_file"}, {"arguments", {{"path", "x"}}}}}));
  expect("unknown leaf", parse_checked(" to=weather.get<|message|>" + invoke("weather.get", {{"id", "1"}}), tools), "",
         "", json::array({{{"name", "weather.get"}, {"arguments", {{"id", 1}}}}}));
  expect("bare recipient", parse_checked(" to=read_file<|message|>" + invoke("read_file", {{"path", "y"}}), tools), "",
         "", json::array({{{"name", "read_file"}, {"arguments", {{"path", "y"}}}}}));
  expect("namespace recipient", parse_checked(" to=fs<|message|>" + call_fs, tools), "", "", fs_call);
  expect("recipient differs from the invoke",
         parse_checked(" to=calendar.get<|message|>" + call_fs, tools), "", "", fs_call);
  // Review Focus 6: generation cut inside a parameter, inside the invoke's name, after </atem:invoke>.
  const std::string head = " to=self<|message|>Read both.<|eom|><|start|>assistant to=fs.read_file<|message|>";
  expect("cut inside <atem:parameter>",
         parse_checked(head + "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n<atem:parameter name=\"path\">"
                       "src/ma", tools), "Read both.", "", json::array());
  expect("cut inside the name",
         parse_checked(head + "<atem:function_calls>\n<atem:invoke name=\"fs.re", tools), "Read both.", "", json::array());
  expect("cut after </atem:invoke>",
         parse_checked(head + "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n<atem:parameter name=\"path\">"
                       "a.cc</atem:parameter>\n<atem:parameter name=\"offset\">3</atem:parameter>\n</atem:invoke>\n<atem:fun",
                       tools), "Read both.", "", fs_call);
  expect("cut in the second of two invokes",
         parse_checked(head + "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n<atem:parameter name=\"path\">"
                       "a.cc</atem:parameter>\n<atem:parameter name=\"offset\">3</atem:parameter>\n</atem:invoke>\n"
                       "<atem:invoke name=\"fs.read_file\">\n<atem:parameter name=\"path\">b.c", tools),
         "Read both.", "", fs_call);
  expect("cut mid-header", parse_checked(head + call_fs + "<|eom|><|start|>assistant to=us", tools), "Read both.", "",
         fs_call);
  std::printf("damaged turns: unterminated reasoning keeps the call, echoes are not calls, truncated / cut invokes "
              "vanish (no call, no markup), unframed text is content, names collapse / pass through / come from "
              "the invoke\n");
}

void case_review_focus() {
  const json tools = json::array({tool("typed.probe", {{"s", {{"type", "string"}}}, {"i", {{"type", "integer"}}},
                                                       {"n", {{"type", "number"}}}, {"b", {{"type", "boolean"}}},
                                                       {"sn", {{"type", json::array({"string", "null"})}}},
                                                       {"o", {{"type", "object"}}}, {"a", {{"type", "array"}}},
                                                       {"u", {{"anyOf", json::array({{{"type", "integer"}},
                                                                                      {{"type", "string"}}})}}}})});
  // Review Focus 7: schema first.
  const MuseParsed p = parse_checked(
      " to=typed.probe<|message|>" +
          invoke("typed.probe", {{"s", "007"}, {"i", "007"}, {"n", "1e3"}, {"b", " TRUE "}, {"sn", "null"}, {"o", "{\"k\": 1}"},
                                 {"a", "[1]"}, {"u", " x "}}) +
          "<|eom|><|start|>assistant to=typed.probe<|message|>" +
          invoke("typed.probe", {{"s", "true"}, {"i", "\"12\""}, {"sn", "\"q\""}, {"s2", " raw "}, {"n", "2.50"}, {"u", "5"}}),
      tools);
  expect("schema-first typing", p, "", "",
         json::array({{{"name", "typed.probe"},
                       {"arguments", {{"s", "007"}, {"i", 7}, {"n", 1000.0}, {"b", true}, {"sn", nullptr}, {"o", {{"k", 1}}},
                                      {"a", {1}}, {"u", " x "}}}},
                      {{"name", "typed.probe"},
                       {"arguments", {{"s", "true"}, {"i", 12}, {"sn", "\"q\""}, {"s2", " raw "}, {"n", 2.5}, {"u", 5}}}}}));
  CHECK(p.tool_calls[0].arguments.at("n").is_number_float());   // JSON-decoded first: 1e3 is a float, as json.loads
  CHECK(p.tool_calls[1].arguments.at("n").is_number_float());
  CHECK(server::muse_convert_value(" x ", "typed.probe", "s", tools) == json(" x "));
  CHECK(server::muse_convert_value(" 42 ", "nope", "k", tools) == json(42));
  CHECK(server::muse_convert_value(" hi ", "nope", "k", tools) == json(" hi "));
  CHECK_EQ(server::muse_normalize_name("x.x", json::array({tool("x", json::object())})), std::string("x"));
  CHECK_EQ(server::muse_normalize_name("x.y", json::array({tool("x", json::object())})), std::string("x.y"));
  CHECK_EQ(server::muse_normalize_name("x.x", json(nullptr)), std::string("x.x"));

  // Review Focus 2: the first header has no <|start|> - " to=self" split anywhere never leaks.
  const std::string text = " to=self<|message|>Think.<|eom|><|start|>assistant to=user<|message|>Done.";
  for (size_t cut = 1; cut < text.size(); ++cut) {
    MuseOutputStream s(json(nullptr));
    std::vector<Delta> all = s.push(text.substr(0, cut));
    for (const Delta& d : all) {
      CHECK(d.text.find(" to") == std::string::npos);
      CHECK(d.text.find("self") == std::string::npos);
      CHECK(d.text.find("<|") == std::string::npos);
      CHECK(d.text.find("user") == std::string::npos);
    }
    const std::vector<Delta> rest = s.push(text.substr(cut));
    all.insert(all.end(), rest.begin(), rest.end());
    const std::vector<Delta> tail = s.finish();
    all.insert(all.end(), tail.begin(), tail.end());
    expect("split header", collect(all), "Think.", "Done.", json::array());
  }
  // A piece ending in "<|" is held until resolved; a growing " to=..." tail too.
  {
    MuseOutputStream s(json(nullptr));
    (void)s.push(" to=user<|message|>");
    const std::vector<Delta> a = s.push("Hello <|");
    std::string emitted;
    for (const Delta& d : a) emitted += d.text;
    CHECK_EQ(emitted, std::string("Hello"));
    const std::vector<Delta> b = s.push("eom|>");
    for (const Delta& d : b) CHECK(d.text.find('<') == std::string::npos);
    MuseOutputStream t(json(nullptr));
    (void)t.push(" to=self<|message|>abc to=fs.re");
    const std::vector<Delta> c = t.push("ad");
    for (const Delta& d : c) CHECK(d.text.find("to") == std::string::npos);
  }
  // Review Focus 3: reasoning, then two calls (<|eom|> between), then the answer - one generation.
  const json fs = json::array({tool("fs.read_file", {{"path", {{"type", "string"}}}})});
  expect("one generation",
         parse_checked(" to=self<|message|>R1<|eom|><|start|>assistant to=fs.read_file<|message|>" +
                           invoke("fs.read_file", {{"path", "a"}}) + "<|eom|><|start|>assistant to=fs.read_file<|message|>" +
                           invoke("fs.read_file", {{"path", "b"}}) + "<|eom|><|start|>assistant to=self<|message|>R2<|eom|>"
                           "<|start|>assistant to=user<|message|>All read.", fs),
         "R1\nR2", "All read.",
         json::array({{{"name", "fs.read_file"}, {"arguments", {{"path", "a"}}}},
                      {{"name", "fs.read_file"}, {"arguments", {{"path", "b"}}}}}));
  std::printf("review focus: schema-first typing (Review Focus 7), the split first header never leaks (2), held "
              "markers, one generation through <|eom|> (3)\n");
}

// ---- (d) the fuzz -------------------------------------------------------------------------------------------------
std::string pick(std::mt19937& rng, const std::vector<std::string>& from) { return from[rng() % from.size()]; }

std::string words(std::mt19937& rng, size_t n) {
  static const std::vector<std::string> alphabet = {
      "a", "Größe", " ", "\n", "x y", "<", "<|", "<|eo", "|>", "to", " to", " to=", "to=x", "self", "user", "<atem:",
      "<atem:invoke", "</atem:invoke>", "\"", "=", ">", "\t", "東京", "😀", "<|message", "<|start", "assistant", ".", "_"};
  std::string s = pick(rng, {"A", "Ö", "z", "1"});
  for (size_t i = 0; i < n; ++i) s += pick(rng, alphabet);
  return s;
}

std::string value(std::mt19937& rng) {
  switch (rng() % 6) {
    case 0: return std::to_string(int(rng() % 2000) - 1000);
    case 1: return pick(rng, {"true", "false", "null", " x ", "007", "1.5", "\"q\""});
    case 2: return "[1, \"two\", null]";
    case 3: return "{\"k\": \"v\"}";
    default: return words(rng, rng() % 8);
  }
}

void case_fuzz() {
  const json tools = json::array({tool("fs.read_file", {{"path", {{"type", "string"}}}, {"n", {{"type", "integer"}}}}),
                                  tool("bash", {{"command", {{"type", "string"}}}}), tool("x", json::object())});
  std::mt19937 rng(2023);
  size_t calls = 0, cut = 0;
  for (int i = 0; i < 2000; ++i) {
    std::string text;
    const int msgs = 1 + int(rng() % 5);
    for (int m = 0; m < msgs; ++m) {
      if (m > 0) text += pick(rng, {"<|eom|><|start|>assistant", "<|start|>assistant", "<|eom|>", ""});
      const std::string rcpt = pick(rng, {" to=self", " to=user", "", " to=fs.read_file", " to=bash", " to=x.x", " to=fs"});
      text += rcpt + "<|message|>";
      if (rcpt == " to=self" || rcpt == " to=user" || rcpt.empty()) {
        text += pick(rng, {"", " ", "\n"}) + words(rng, rng() % 12) + pick(rng, {"", " ", "\n "});
      } else {
        text += "<atem:function_calls>\n";
        const int inv = 1 + int(rng() % 2);
        for (int k = 0; k < inv; ++k) {
          text += "<atem:invoke name=\"" + pick(rng, {"fs.read_file", "bash", "x.x", "zz"}) + "\">\n";
          const int np = int(rng() % 3);
          for (int q = 0; q < np; ++q)
            text += "<atem:parameter name=\"" + pick(rng, {"path", "n", "command", "k"}) + "\">" + value(rng) +
                    "</atem:parameter>\n";
          text += "</atem:invoke>\n";
        }
        text += "</atem:function_calls>";
      }
    }
    if (rng() % 3 == 0) {
      text.resize(rng() % (text.size() + 1));   // generation cut anywhere
      ++cut;
    }
    if (rng() % 7 == 0) text = words(rng, 20);  // unframed
    calls += parse_checked(text, tools).tool_calls.size();
  }
  std::printf("fuzz: 2000 outputs (%zu cut, %zu calls) - every whole parse equals its byte and 1-7 byte streams\n",
              cut, calls);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/server";
  case_fixture(dir);
  case_damaged();
  case_review_focus();
  case_fuzz();
  std::printf("toolcall_muse_test OK (%zu parses, each whole + byte stream + random pieces)\n", g_parses);
  return 0;
}

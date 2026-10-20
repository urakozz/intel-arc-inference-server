// Spec 24e: LFM2.5-24B-A2B's reasoning and pythonic tool-call parser (server/toolcall_lfm2.h).
//   toolcall_lfm2_test <tests/tokenizer> <tests/golden/prompts> <snapshot dir>
// Every parse is checked whole, split in two at EVERY byte, byte at a time and in random 1-7 byte pieces (spec 7a's
// rule: the deltas of any split concatenate to the whole-text parse, and the per-request rule counters agree too).
// The output a model writes starts INSIDE reasoning (the prompt ends in "<think>"), so every case below does.
// Cases, by the plan's Review Focus:
//   1. the round trip: every assistant turn of the HF renders of lfm2moe_template_cases.json parses back to the message
//      the template rendered (reasoning, content, calls with every literal type, quotes, escapes, nested values), and -
//      with the snapshot's template - the parsed turn re-renders to the same bytes;
//   2. output the template would never write that still yields the call the model meant: unescaped quotes, delimiters
//      and code inside strings, quoted names, separators, empty calls, a bare call, a lone positional dict; and the
//      invalid ones, which make the block content verbatim with its tags;
//   3. generation that stops inside a block: a closed call is emitted once, an unfinished one never and its text is not
//      content; a stray end tag is dropped;
//   4. reasoning kept verbatim, keyed on the prompt's <think>;
//   5. the limits (depth 64, 1 MiB, 10^5 quotes), the per-context escapes, the counters, and a fuzz.
#include <chrono>
#include <deque>
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
#include "server/toolcall_lfm2.h"
#include "tokenizer/chat_template.h"

namespace {

using json = nlohmann::json;
using server::Delta;
using server::Lfm2OutputStream;
using server::ParsedOutput;
using server::PythonicRules;

const std::string kStart = "<|tool_call_start|>";
const std::string kEnd = "<|tool_call_end|>";

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

json calls_json(const ParsedOutput& p) {
  json calls = json::array();
  for (const auto& c : p.tool_calls) calls.push_back(json::array({c.name, c.arguments}));
  return calls;
}

// (A damaged output cut inside a character holds invalid UTF-8: dump it with replacement.)
std::string dump_safe(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

std::string show(const ParsedOutput& p) {
  return dump_safe({{"reasoning", p.reasoning}, {"content", p.content}, {"calls", calls_json(p)}});
}

bool same_rules(const PythonicRules& a, const PythonicRules& b) {
  return a.quote_recovered == b.quote_recovered && a.quoted_name == b.quoted_name && a.bare_call == b.bare_call &&
         a.json_literal == b.json_literal && a.positional_dict == b.positional_dict &&
         a.dropped_unfinished == b.dropped_unfinished && a.stray_end == b.stray_end;
}

std::string show(const PythonicRules& r) {
  return json({{"quote_recovered", r.quote_recovered}, {"quoted_name", r.quoted_name}, {"bare_call", r.bare_call},
               {"json_literal", r.json_literal}, {"positional_dict", r.positional_dict},
               {"dropped_unfinished", r.dropped_unfinished}, {"stray_end", r.stray_end}}).dump();
}

struct Run {
  ParsedOutput out;
  PythonicRules rules;
};

Run stream(const std::string& text, const std::vector<size_t>& sizes, bool reasoning_first = true) {
  Lfm2OutputStream s(json::array(), reasoning_first);
  std::vector<Delta> deltas;
  size_t at = 0, k = 0;
  while (at < text.size()) {
    const size_t n = std::min(sizes.empty() ? text.size() : sizes[k++ % sizes.size()], text.size() - at);
    for (Delta& d : s.push(text.substr(at, n))) deltas.push_back(std::move(d));
    at += n;
  }
  for (Delta& d : s.finish()) deltas.push_back(std::move(d));
  return {collect(deltas), s.rules()};
}

// `expected`: {"reasoning", "content", "calls": [[name, args], ...]}; `rules` (optional): the counters that must fire
// (the others zero).
struct Expect {
  std::string reasoning, content;
  json calls = json::array();
};

void expect_one(const char* what, const std::string& text, const Expect& e, const PythonicRules* want_rules,
                const Run& got, const char* how) {
  const bool ok = got.out.reasoning == e.reasoning && got.out.content == e.content && calls_json(got.out) == e.calls &&
                  (want_rules == nullptr || same_rules(got.rules, *want_rules));
  if (!ok) {
    std::fprintf(stderr, "%s [%s]\n  text : %s\n  got  : %s rules %s\n  want : %s%s%s\n", what, how,
                 dump_safe(text).c_str(), show(got.out).c_str(), show(got.rules).c_str(),
                 dump_safe({{"reasoning", e.reasoning}, {"content", e.content}, {"calls", e.calls}}).c_str(),
                 want_rules ? " rules " : "", want_rules ? show(*want_rules).c_str() : "");
    std::exit(1);
  }
}

// Whole, every two-way split, byte at a time, random pieces: all identical (outputs and counters).
void expect(const char* what, const std::string& text, const Expect& e, const PythonicRules* want_rules = nullptr,
            bool reasoning_first = true) {
  const Run whole = stream(text, {}, reasoning_first);
  expect_one(what, text, e, want_rules, whole, "whole");
  for (size_t cut = 1; cut < text.size(); ++cut) {
    Lfm2OutputStream s(json::array(), reasoning_first);
    std::vector<Delta> deltas = s.push(text.substr(0, cut));
    for (Delta& d : s.push(text.substr(cut))) deltas.push_back(std::move(d));
    for (Delta& d : s.finish()) deltas.push_back(std::move(d));
    const Run got{collect(deltas), s.rules()};
    expect_one(what, text, e, &whole.rules, got, ("split at " + std::to_string(cut)).c_str());
    expect_one(what, text, e, want_rules, got, ("split at " + std::to_string(cut)).c_str());
  }
  expect_one(what, text, e, &whole.rules, stream(text, {1}, reasoning_first), "byte at a time");
  std::mt19937 rng(7);
  for (int i = 0; i < 6; ++i) {
    std::vector<size_t> sizes;
    for (int k = 0; k < 16; ++k) sizes.push_back(1 + rng() % 7);
    expect_one(what, text, e, &whole.rules, stream(text, sizes, reasoning_first), "random pieces");
  }
}

json C(const std::string& name, json args = json::object()) { return json::array({name, std::move(args)}); }

// R + "</think>" + the rest: a generation with this reasoning, closed.
std::string gen(const std::string& reasoning, const std::string& rest) { return reasoning + "</think>" + rest; }
std::string block(const std::string& inner) { return kStart + inner + kEnd; }

PythonicRules rules(uint32_t quote = 0, uint32_t qname = 0, uint32_t bare = 0, uint32_t jlit = 0, uint32_t pdict = 0,
                    uint32_t dropped = 0, uint32_t stray = 0) {
  PythonicRules r;
  r.quote_recovered = quote;
  r.quoted_name = qname;
  r.bare_call = bare;
  r.json_literal = jlit;
  r.positional_dict = pdict;
  r.dropped_unfinished = dropped;
  r.stray_end = stray;
  return r;
}

// The same, kept alive for the call that takes its address.
const PythonicRules* R(uint32_t quote = 0, uint32_t qname = 0, uint32_t bare = 0, uint32_t jlit = 0, uint32_t pdict = 0,
                       uint32_t dropped = 0, uint32_t stray = 0) {
  static std::deque<PythonicRules> pool;
  pool.push_back(rules(quote, qname, bare, jlit, pdict, dropped, stray));
  return &pool.back();
}

// One call whose single argument is `a=<literal>`: the literal's JSON value.
void lit(const char* what, const std::string& literal, const json& value, const PythonicRules* r = nullptr) {
  expect(what, gen("", block("[f(a=" + literal + ")]")), {"", "", json::array({C("f", {{"a", value}})})}, r);
}

// A block that is no call: content verbatim with its tags.
void invalid(const char* what, const std::string& inner) {
  const std::string b = block(inner);
  expect(what, gen("r", "before" + b + "after"), {"r", "before" + b + "after", json::array()});
}

// ---------------------------------------------------------------------------------------------------------------
// Review Focus 1: the template's own output.
// ---------------------------------------------------------------------------------------------------------------

const char* const kSkipRoundTrip[] = {"think_in_content", "cfm_tag", "cfm_tag_with_call", "content_list_odd_parts",
                                      "content_mapping", "reasoning_field_names"};

std::string parts_text(const json& content) {
  if (content.is_null()) return "";
  if (content.is_string()) return content.get<std::string>();
  std::string s;
  for (const json& p : content) s += p.at("text").get<std::string>();
  return s;
}

void round_trip(const std::string& dir, const std::string& prompts, const std::string& snapshot) {
  const json spec = json::parse(slurp(dir + "/lfm2moe_template_cases.json"));
  json all_cases = spec.at("cases");
  const json agentic = json::parse(slurp(prompts + "/lfm2moe_agentic.json"));
  all_cases.push_back({{"name", "agentic"}, {"messages", agentic.at("messages")}, {"tools", agentic.at("tools")},
                       {"kwargs", json::object()}, {"inline_tools", true}});
  std::unique_ptr<chat::Template> tmpl;
  if (std::ifstream(snapshot + "/chat_template.jinja").good()) tmpl = std::make_unique<chat::Template>(snapshot);
  else std::printf("round trip: no snapshot at %s - the re-render half is skipped\n", snapshot.c_str());
  size_t turns = 0, calls = 0, rerendered = 0;
  for (const json& c : all_cases) {
    const std::string name = c.at("name");
    bool skip = false;
    for (const char* s : kSkipRoundTrip) skip = skip || name == s;
    if (skip) continue;
    const std::string text = slurp(dir + "/lfm2moe_template_" + name + ".txt");
    json messages = c.at("messages");
    // The assistant turns of the render, in order (the last "<|im_start|>assistant\n<think>" is the generation prompt).
    size_t at = 0, assistant_index = 0;
    std::vector<std::string> segments;
    const std::string open = "<|im_start|>assistant\n";
    while ((at = text.find(open, at)) != std::string::npos) {
      at += open.size();
      const size_t end = text.find("<|im_end|>\n", at);
      if (end == std::string::npos) break;   // the generation prompt
      segments.push_back(text.substr(at, end - at));
      at = end;
    }
    json rebuilt = messages;
    for (json& m : rebuilt) {
      if (m.at("role") != "assistant") continue;
      CHECK(assistant_index < segments.size());
      std::string seg = segments[assistant_index++];
      const bool thought = seg.rfind("<think>", 0) == 0;
      const std::string generated = thought ? seg.substr(7) : "</think>" + seg;
      const ParsedOutput got = server::parse_output_lfm2(generated, json::array());
      ++turns;
      const std::string want_content = parts_text(m.contains("content") ? m.at("content") : json());
      std::string want_reasoning;
      if (thought) want_reasoning = seg.substr(7, seg.find("</think>") - 7);
      if (got.reasoning != want_reasoning || got.content != want_content) {
        std::fprintf(stderr, "round trip %s turn %zu: reasoning %s content %s\n  want reasoning %s content %s\n",
                     name.c_str(), assistant_index, json(got.reasoning).dump().c_str(), json(got.content).dump().c_str(),
                     json(want_reasoning).dump().c_str(), json(want_content).dump().c_str());
        std::exit(1);
      }
      json want_calls = json::array();
      if (m.contains("tool_calls")) {
        for (const json& tc : m.at("tool_calls")) {
          const json& fn = tc.contains("function") ? tc.at("function") : tc;
          json args = fn.contains("arguments") ? fn.at("arguments") : json();
          if (args.is_null() || args.is_string()) args = json::object();   // fn(): none, "", "{}", "null"
          want_calls.push_back(C(fn.at("name"), args));
        }
      }
      if (calls_json(got) != want_calls) {
        std::fprintf(stderr, "round trip %s turn %zu: calls %s\n  want %s\n", name.c_str(), assistant_index,
                     calls_json(got).dump().c_str(), want_calls.dump().c_str());
        std::exit(1);
      }
      calls += want_calls.size();
      // The same turn, as the OpenAI message a client would send back.
      m["content"] = got.content;
      m.erase("reasoning_content");
      m.erase("reasoning");
      m.erase("thinking");
      if (!got.reasoning.empty()) m["reasoning_content"] = got.reasoning;
      m.erase("tool_calls");
      if (!got.tool_calls.empty()) {
        json tcs = json::array();
        for (const auto& tc : got.tool_calls)
          tcs.push_back({{"id", tc.id}, {"type", "function"}, {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
        m["tool_calls"] = tcs;
      }
    }
    CHECK_EQ(assistant_index, segments.size());
    if (!tmpl) continue;
    json tools = c.at("tools");
    if (tools.is_array() && !c.contains("inline_tools")) {
      json named = json::array();
      for (const json& n : tools) named.push_back(spec.at("tools").at(n.get<std::string>()));
      tools = named;
    }
    const std::string again = tmpl->render(rebuilt, tools, true, c.at("kwargs"));
    if (again != text) {
      size_t i = 0;
      while (i < again.size() && i < text.size() && again[i] == text[i]) ++i;
      std::fprintf(stderr, "round trip %s: the parsed turns re-render differently at byte %zu\n  got : %s\n  want: %s\n",
                   name.c_str(), i, json(again.substr(i, 100)).dump().c_str(), json(text.substr(i, 100)).dump().c_str());
      std::exit(1);
    }
    ++rerendered;
  }
  CHECK(turns >= 25 && calls >= 20);
  std::printf("round trip: %zu assistant turns, %zu calls parsed back to their messages; %zu cases re-rendered to the "
              "same bytes\n", turns, calls, rerendered);
}

// ---------------------------------------------------------------------------------------------------------------
// Review Focus 2: the literals, the idioms and the invalid ones.
// ---------------------------------------------------------------------------------------------------------------

void literals() {
  lit("int", "3", 3);
  lit("negative int", "-7", -7);
  lit("plus int", "+4", 4);
  lit("big int", "12345678901234", 12345678901234LL);
  lit("float", "2.5", 2.5);
  lit("float whole", "1.0", 1.0);
  lit("leading dot", ".5", 0.5);
  lit("exponent", "1e3", 1000.0);
  lit("signed exponent", "-2.5E-3", -0.0025);
  lit("python exponent as the template writes it", "1e-07", 1e-07);
  lit("python big float", "1e+20", 1e20);
  lit("True", "True", true);
  lit("False", "False", false);
  lit("None", "None", nullptr);
  const PythonicRules jl = rules(0, 0, 0, 1);
  lit("json true", "true", true, &jl);
  lit("json false", "false", false, &jl);
  lit("json null", "null", nullptr, &jl);
  lit("empty list", "[]", json::array());
  lit("list with every type", "[1, \"a\", None, True, 2.5, null, true, false, 'x']",
      json::array({1, "a", nullptr, true, 2.5, nullptr, true, false, "x"}));
  lit("nested lists", "[[1, [2]], []]", json::parse("[[1,[2]],[]]"));
  lit("tuple", "(1, 'a')", json::array({1, "a"}));
  lit("one-tuple", "(1,)", json::array({1}));
  lit("empty tuple", "()", json::array());
  lit("parenthesised scalar", "(1)", 1);
  lit("trailing comma in a list", "[1, 2, ]", json::array({1, 2}));
  lit("dict", "{\"k\": 1, 'j': [True]}", json::parse("{\"k\":1,\"j\":[true]}"));
  lit("empty dict", "{}", json::object());
  lit("dict with int and None keys", "{1: 'a', None: 2, True: 3}", json::parse("{\"1\":\"a\",\"null\":2,\"true\":3}"));
  lit("nested dict", "{\"a\": {\"b\": [{\"c\": \"d'e\"}]}, \"z\": []}", json::parse("{\"a\":{\"b\":[{\"c\":\"d'e\"}]},\"z\":[]}"));
  lit("dict trailing comma", "{'a': 1,}", json::parse("{\"a\":1}"));
  lit("spaces and newlines in containers", "[ 1 ,\n 2 ,\t{ 'a' :\n 3 } ]", json::parse("[1,2,{\"a\":3}]"));
  lit("single quotes", "'x'", "x");
  lit("double quotes", "\"x\"", "x");
  lit("triple single", "'''a'b\"c'''", "a'b\"c");
  lit("triple double", "\"\"\"a\"b'c\"\"\"", "a\"b'c");
  lit("empty string", "''", "");
  lit("empty double string", "\"\"", "");
  lit("unicode", "'Gr\xc3\xb6\xc3\x9f" "e \xe6\x9d\xb1\xe4\xba\xac \xf0\x9f\x98\x80'", "Gr\xc3\xb6\xc3\x9f" "e \xe6\x9d\xb1\xe4\xba\xac \xf0\x9f\x98\x80");
  lit("a raw tab and newline", "'a\tb\nc'", "a\tb\nc");
  // Several arguments, any spacing.
  expect("two arguments", gen("", block("[f(a=1, b='x')]")), {"", "", json::array({C("f", {{"a", 1}, {"b", "x"}})})});
  expect("no spaces", gen("", block("[f(a=1,b='x')]")), {"", "", json::array({C("f", {{"a", 1}, {"b", "x"}})})});
  expect("loose spaces", gen("", block("[ f ( a = 1 ,\n b = 'x' , ) ]")),
         {"", "", json::array({C("f", {{"a", 1}, {"b", "x"}})})});
  expect("a repeated key keeps the last", gen("", block("[f(a=1, a=2)]")), {"", "", json::array({C("f", {{"a", 2}})})});
  expect("names with dots, dashes and digits", gen("", block("[fs.read-file2(my-arg.x=1, _y=2)]")),
         {"", "", json::array({C("fs.read-file2", {{"my-arg.x", 1}, {"_y", 2}})})});
}

void escapes() {
  // Top-level strings: exactly \\ , the string's own quote escape, \n and \r are decoded; every other backslash stays.
  lit("top-level \\\\", "'a\\\\b'", "a\\b");
  lit("top-level \\' in single quotes", "'a\\'b'", "a'b");
  lit("top-level \\\" in double quotes", "\"a\\\"b\"", "a\"b");
  lit("top-level \\n", "'a\\nb'", "a\nb");
  lit("top-level \\r", "'a\\rb'", "a\rb");
  lit("top-level \\t stays", "'a\\tb'", "a\\tb");
  lit("top-level \\u stays", "'\\u00e9'", "\\u00e9");
  lit("top-level \\\" in single quotes stays", "'a\\\"b'", "a\\\"b");
  lit("windows path", "'C:\\Users\\test\\file.txt'", "C:\\Users\\test\\file.txt");
  lit("regex", "'\\d+\\s*\\.'", "\\d+\\s*\\.");
  lit("backslash before the closing quote is the pair \\\\", "'C:\\\\'", "C:\\");
  lit("triple-quoted escapes", "'''a\\nb\\'c\\\\d\\te'''", "a\nb'c\\d\\te");
  // Strings inside a list or dict: JSON's set, plus their own quote's escape; every other backslash stays.
  lit("nested JSON escapes", "[\"\\\"q\\\" \\\\ \\/ \\b \\f \\n \\r \\t\"]", json::array({"\"q\" \\ / \b \f \n \r \t"}));
  lit("nested \\u", "[\"\\u00e9\\u4e1c\"]", json::array({"\xc3\xa9\xe4\xb8\x9c"}));
  lit("nested surrogate pair", "[\"\\ud83d\\ude00\"]", json::array({"\xf0\x9f\x98\x80"}));
  lit("nested lone surrogate", "[\"\\ud83dx\"]", json::array({"\xef\xbf\xbdx"}));
  lit("nested other backslashes stay", "[\"C:\\Users\\d\"]", json::array({"C:\\Users\\d"}));
  lit("nested single-quoted own quote", "['it\\'s']", json::array({"it's"}));
  lit("nested dict value escapes", "{\"k\": \"a\\nb\\\\c\"}", json::parse("{\"k\":\"a\\nb\\\\c\"}"));
  lit("nested key escapes", "{\"a\\\"b\": 1}", json::parse("{\"a\\\"b\":1}"));
}

void recovery() {
  const PythonicRules q1 = rules(1), q2 = rules(2);
  expect("it's a test", gen("", block("[f(a='it's a test')]")),
         {"", "", json::array({C("f", {{"a", "it's a test"}})})}, &q1);
  expect("Johns' car", gen("", block("[f(a='That's Johns' car.')]")),
         {"", "", json::array({C("f", {{"a", "That's Johns' car."}})})}, &q2);
  expect("an argument after a recovered string", gen("", block("[f(a='it's', b='x')]")),
         {"", "", json::array({C("f", {{"a", "it's"}, {"b", "x"}})})}, &q1);
  expect("a quote before a comma that starts no argument", gen("", block("[f(a='x', 'y', z')]")),
         {"", "", json::array({C("f", {{"a", "x', 'y', z"}})})}, R(3));
  // Code with an unescaped "__main__" inside a double-quoted argument, then another argument.
  const std::string code = "if __name__ == \"__main__\":\\n    main()";
  expect("unescaped quotes in code", gen("", block("[run(code=\"" + code + "\", n=2)]")),
         {"", "", json::array({C("run", {{"code", "if __name__ == \"__main__\":\n    main()"}, {"n", 2}})})}, &q2);
  expect("unescaped quotes in code, last argument", gen("", block("[run(code=\"" + code + "\")]")),
         {"", "", json::array({C("run", {{"code", "if __name__ == \"__main__\":\n    main()"}})})}, &q2);
  expect("a python list of strings inside a string", gen("", block("[f(a='x = ['a', 'b']; y')]")),
         {"", "", json::array({C("f", {{"a", "x = ['a', 'b']; y"}})})}, R(4));
  // Delimiters inside strings are content.
  expect("delimiters in a string", gen("", block("[f(a='x, y) ] { = >= \\n {z} (w', b=1)]")),
         {"", "", json::array({C("f", {{"a", "x, y) ] { = >= \n {z} (w"}, {"b", 1}})})});
  expect("a call inside a string", gen("", block("[f(a='g(b=1)', c=\"h('d')\")]")),
         {"", "", json::array({C("f", {{"a", "g(b=1)"}, {"c", "h('d')"}})})}, R());
  // The end tag ends the block wherever it is: the unterminated string makes the first block content, the rest is text.
  expect("the end tag inside a string ends the block", gen("", block("[f(a='x") + "y')]" + kEnd),
         {"", kStart + "[f(a='x" + kEnd + "y')]", json::array()}, R(0, 0, 0, 0, 0, 0, 1));
}

void idioms() {
  const PythonicRules qn1 = rules(0, 1);
  expect("quoted key", gen("", block("[f(\"path\"=\"x\")]")), {"", "", json::array({C("f", {{"path", "x"}})})}, &qn1);
  expect("single-quoted key", gen("", block("[f('path'='x', 'n'=2)]")),
         {"", "", json::array({C("f", {{"path", "x"}, {"n", 2}})})}, R(0, 2));
  expect("quoted function name", gen("", block("['read_file'(path='x')]")),
         {"", "", json::array({C("read_file", {{"path", "x"}})})}, &qn1);
  expect("double-quoted function name", gen("", block("[\"read_file\"(path='x')]")),
         {"", "", json::array({C("read_file", {{"path", "x"}})})}, &qn1);
  expect("two calls", gen("", block("[a(x=1), b(y=2)]")),
         {"", "", json::array({C("a", {{"x", 1}}), C("b", {{"y", 2}})})});
  expect("comma without a space", gen("", block("[a(x=1),b(y=2)]")),
         {"", "", json::array({C("a", {{"x", 1}}), C("b", {{"y", 2}})})});
  expect("newlines between calls", gen("", block("[a(x=1),\n\n  b(y=2)\n]")),
         {"", "", json::array({C("a", {{"x", 1}}), C("b", {{"y", 2}})})});
  expect("trailing comma after a call", gen("", block("[a(x=1),]")), {"", "", json::array({C("a", {{"x", 1}})})});
  expect("trailing comma in the arguments", gen("", block("[a(x=1,)]")), {"", "", json::array({C("a", {{"x", 1}})})});
  expect("no arguments", gen("", block("[fn()]")), {"", "", json::array({C("fn")})});
  expect("no arguments, spaced", gen("", block("[fn( )]")), {"", "", json::array({C("fn")})});
  expect("empty string argument", gen("", block("[fn(name=\"\")]")), {"", "", json::array({C("fn", {{"name", ""}})})});
  expect("three calls", gen("", block("[a(), b(), c()]")), {"", "", json::array({C("a"), C("b"), C("c")})});
  // A bare call without the brackets (decision 7).
  const PythonicRules bare = rules(0, 0, 1);
  expect("bare call", gen("", block("fn(a=1)")), {"", "", json::array({C("fn", {{"a", 1}})})}, &bare);
  expect("bare calls", gen("", block("a(x=1), b()")), {"", "", json::array({C("a", {{"x", 1}}), C("b")})}, &bare);
  // A lone positional dict is the arguments object.
  const PythonicRules pd = rules(0, 0, 0, 0, 1);
  expect("positional dict", gen("", block("[f({\"a\": 1, \"b\": [1, 2]})]")),
         {"", "", json::array({C("f", {{"a", 1}, {"b", json::array({1, 2})}})})}, &pd);
  expect("positional dict, python quotes", gen("", block("[f({'a': 'x'})]")),
         {"", "", json::array({C("f", {{"a", "x"}})})}, &pd);
  expect("missing ] is harmless", gen("", block("[f(a=1)")), {"", "", json::array({C("f", {{"a", 1}})})});
  expect("text around calls", gen("think", "Let me look. " + block("[a()]") + " Done." + block("[b()]") + "!"),
         {"think", "Let me look.  Done.!", json::array({C("a"), C("b")})});
}

void invalids() {
  invalid("positional scalar", "[f(1)]");
  invalid("positional string", "[f(\"x\")]");
  invalid("positional after keyword", "[f(a=1, 2)]");
  invalid("keyword after positional", "[f(1, a=2)]");
  invalid("two positional dicts", "[f({}, {})]");
  invalid("positional dict and a keyword", "[f({\"a\": 1}, b=2)]");
  invalid("name without =", "[f(a)]");
  invalid("= without a value", "[f(a=)]");
  invalid("= without a value, then more", "[f(a=, b=1)]");
  invalid("inf", "[f(a=inf)]");
  invalid("-inf", "[f(a=-inf)]");
  invalid("nan", "[f(a=nan)]");
  invalid("bare name", "[f(a=foo)]");
  invalid("1e400", "[f(a=1e400)]");
  invalid("unterminated string", "[f(a='unterminat)]");
  invalid("unterminated list", "[f(a=[1, 2)]");
  invalid("unterminated dict", "[f(a={'k': 1)]");
  invalid("missing comma between arguments", "[f(a=1 b=2)]");
  invalid("missing paren", "[f(a=1]");
  invalid("no name", "[(a=1)]");
  invalid("empty list", "[]");
  invalid("empty block", "");
  invalid("whitespace block", " \n ");
  invalid("a name only", "[f]");
  invalid("a number", "[1]");
  invalid("a dict", "[{}]");
  invalid("number glued to a name", "[f(a=12abc)]");
  invalid("dict with a non-scalar key", "[f(a={[1]: 2})]");
  invalid("dict without a colon", "[f(a={'k'})]");
  invalid("set literal", "[f(a={1, 2})]");
  invalid("garbage", "hello world");
  // The first call decides: an invalid first call makes the block content even if a valid call follows.
  invalid("invalid then valid", "[f(1), g(a=1)]");
  // A valid call, then an invalid one: the valid call stays, the rest is dropped (never content).
  expect("valid then invalid", gen("", block("[a(x=1), b(1)]")), {"", "", json::array({C("a", {{"x", 1}})})},
         R(0, 0, 0, 0, 0, 1));
  expect("valid then garbage", gen("", block("[a(x=1)] trailing words")), {"", "", json::array({C("a", {{"x", 1}})})},
         R(0, 0, 0, 0, 0, 1));
  expect("valid, missing comma, valid", gen("", block("[a(x=1) b(y=2)]")), {"", "", json::array({C("a", {{"x", 1}})})},
         R(0, 0, 0, 0, 0, 1));
}

// ---------------------------------------------------------------------------------------------------------------
// Review Focus 3: generation that ends inside a block.
// ---------------------------------------------------------------------------------------------------------------

void cut_blocks() {
  const json one = json::array({C("f", {{"a", 1}})});
  // The call's ')' closed; the generation stopped before ']' and before the end tag: emitted once.
  expect("cut after ')'", gen("r", kStart + "[f(a=1)"), {"r", "", one});
  expect("cut after ')]'", gen("r", kStart + "[f(a=1)]"), {"r", "", one});
  expect("cut after ')' with a string argument", gen("r", kStart + "[f(a=\"x\")"),
         {"r", "", json::array({C("f", {{"a", "x"}})})});
  expect("cut after ')' of a bare call", gen("r", kStart + "f(a=1)"), {"r", "", one}, R(0, 0, 1));
  expect("cut after the comma", gen("r", kStart + "[f(a=1),"), {"r", "", one});
  expect("cut after ', '", gen("r", kStart + "[f(a=1), "), {"r", "", one});
  expect("cut inside the second call", gen("r", kStart + "[f(a=1), g(b="), {"r", "", one}, R(0, 0, 0, 0, 0, 1));
  expect("cut inside the second call's name", gen("r", kStart + "[f(a=1), g"), {"r", "", one}, R(0, 0, 0, 0, 0, 1));
  expect("cut inside an unterminated string", gen("r", kStart + "[f(a=\"unterminat"), {"r", "", json::array()},
         R(0, 0, 0, 0, 0, 1));
  expect("cut after the opening bracket", gen("r", kStart + "["), {"r", "", json::array()}, R(0, 0, 0, 0, 0, 1));
  expect("cut right after the start tag", gen("r", kStart), {"r", "", json::array()});
  expect("cut inside the name", gen("r", kStart + "[read_fi"), {"r", "", json::array()}, R(0, 0, 0, 0, 0, 1));
  expect("cut inside a list argument", gen("r", kStart + "[f(a=[1, 2"), {"r", "", json::array()},
         R(0, 0, 0, 0, 0, 1));
  expect("cut inside the end tag", gen("r", kStart + "[f(a=1)]<|tool_call_e"), {"r", "", one});
  expect("cut block keeps the content before it", gen("r", "Okay. " + kStart + "[f(a=\"x"),
         {"r", "Okay. ", json::array()}, R(0, 0, 0, 0, 0, 1));
  expect("cut invalid first call is dropped, not content", gen("r", kStart + "[f(1)"), {"r", "", json::array()},
         R(0, 0, 0, 0, 0, 1));
  // A closed string argument cut right after its quote: the quote closes (the block ended).
  expect("cut right after a closing quote", gen("r", kStart + "[f(a='x'"), {"r", "", json::array()},
         R(0, 0, 0, 0, 0, 1));
  // A stray end tag is dropped; the end tag of a closed block is the block's.
  expect("stray end tag", gen("r", "a" + kEnd + "b"), {"r", "ab", json::array()}, R(0, 0, 0, 0, 0, 0, 1));
  expect("stray end tag after a block", gen("r", block("[f(a=1)]") + kEnd), {"r", "", one}, R(0, 0, 0, 0, 0, 0, 1));
  expect("stray end tag in reasoning is reasoning", "x" + kEnd + "y</think>z", {"x" + kEnd + "y", "z", json::array()});
  expect("a second start tag inside a block is block text", gen("r", block("[f(a=1)]" + kStart)), {"r", "", one},
         R(0, 0, 0, 0, 0, 1));
  // Two blocks, the second cut.
  expect("two blocks, the second cut", gen("r", block("[f(a=1)]") + "mid" + kStart + "[g(b=2)"),
         {"r", "mid", json::array({C("f", {{"a", 1}}), C("g", {{"b", 2}})})});
}

// ---------------------------------------------------------------------------------------------------------------
// Review Focus 4: reasoning.
// ---------------------------------------------------------------------------------------------------------------

void reasoning() {
  expect("plain", "abc</think>def", {"abc", "def", json::array()});
  expect("verbatim whitespace", "\n think \n</think>\n\nanswer\n", {"\n think \n", "\n\nanswer\n", json::array()});
  expect("empty reasoning", "</think>x", {"", "x", json::array()});
  expect("empty everything", "", {"", "", json::array()});
  expect("no close: reasoning only", "still thinking...", {"still thinking...", "", json::array()});
  expect("tool call start inside reasoning is reasoning", "maybe " + kStart + "[f()]" + kEnd + " no",
         {"maybe " + kStart + "[f()]" + kEnd + " no", "", json::array()});
  expect("a call after the close", "think</think>" + block("[f()]"), {"think", "", json::array({C("f")})});
  expect("<think> inside reasoning stays", "a<think>b</think>c", {"a<think>b", "c", json::array()});
  expect("a second </think> is content", "a</think>b</think>c", {"a", "b</think>c", json::array()});
  expect("partial close tag at the end of generation", "a</thi", {"a</thi", "", json::array()});
  expect("partial tag at the end of generation, in the body", "a</think>b<|tool_ca", {"a", "b<|tool_ca", json::array()});
  expect("content with angle brackets", "a</think>1 < 2 and <b> and <|x|>", {"a", "1 < 2 and <b> and <|x|>", json::array()});
  expect("content verbatim around a call", "a</think>\n\nI will check.\n" + block("[f()]") + "\n",
         {"a", "\n\nI will check.\n\n", json::array({C("f")})});
  expect("whitespace-only content is kept", "a</think>\n" + block("[f()]"), {"a", "\n", json::array({C("f")})});
  // Without the prompt's <think> (reasoning_first false) the output is content from the first byte.
  expect("no reasoning", "plain answer " + block("[f()]"), {"", "plain answer ", json::array({C("f")})}, nullptr, false);
}

// ---------------------------------------------------------------------------------------------------------------
// Review Focus 5: limits and the counters' bookkeeping.
// ---------------------------------------------------------------------------------------------------------------

std::string nested(int depth) {
  return std::string(static_cast<size_t>(depth), '[') + "1" + std::string(static_cast<size_t>(depth), ']');
}

void limits() {
  // Depth: a list inside the argument counts 1; 64 containers deep parse, 65 do not.
  json deep = 1;
  for (int i = 0; i < 64; ++i) deep = json::array({deep});
  expect("depth 64", gen("", block("[f(a=" + nested(64) + ")]")), {"", "", json::array({C("f", {{"a", deep}})})});
  invalid("depth 65", "[f(a=" + nested(65) + ")]");
  invalid("depth 5000 does not overflow the stack", "[f(a=" + nested(5000) + ")]");
  // Size: the first 1 MiB of a block is read; a call that closes later is not a call.
  const size_t mib = size_t(1) << 20;
  const std::string tail(mib, 'x');
  const std::string big_arg = block("[f(a='" + tail + "')]");   // the call closes past 1 MiB
  {
    const Run r = stream(gen("r", big_arg), {mib / 3});
    CHECK(r.out.tool_calls.empty());
    CHECK_EQ(r.out.content, big_arg);
  }
  {
    // A call that closed before the limit stays; the padding after it is dropped.
    const std::string padded = block("[f(a=1)]" + std::string(mib, ' ') + "junk");
    const Run r = stream(gen("r", padded), {mib / 3});
    CHECK_EQ(r.out.tool_calls.size(), 1u);
    CHECK_EQ(r.out.content, std::string());
    // Exactly 1 MiB of block still parses a call that closes at its last byte.
    const std::string call = "[f(a='";
    const std::string exact = call + std::string(mib - call.size() - 3, 'y') + "')]";
    CHECK_EQ(exact.size(), mib);
    const Run e = stream(gen("r", block(exact)), {mib / 5});
    CHECK_EQ(e.out.tool_calls.size(), 1u);
    CHECK_EQ(e.out.tool_calls[0].arguments.at("a").get<std::string>().size(), mib - call.size() - 3);
  }
  // 10^5 quotes: no backtracking blow-up, linear and fast.
  {
    std::string quotes = "[f(a='x";
    for (int i = 0; i < 100000; ++i) quotes += "'";
    quotes += "x')]";
    const auto t0 = std::chrono::steady_clock::now();
    const Run r = stream(gen("", block(quotes)), {});
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK_EQ(r.out.tool_calls.size(), 1u);
    CHECK_EQ(r.out.tool_calls[0].arguments.at("a").get<std::string>().size(), size_t(100002));
    std::printf("100000 quotes in one string: %.2f ms\n", ms);
    CHECK(ms < 200.0);
    // Quotes that look like closers and are not (each is followed by ", 'a'" - no argument name): all but the last one
    // are part of the string; the last is followed by ", b=1)".
    std::string pairs = "[f(a=";
    for (int i = 0; i < 100000; ++i) pairs += "'a', ";
    pairs += "b=1)]";
    const auto t1 = std::chrono::steady_clock::now();
    const Run p = stream(gen("", block(pairs)), {});
    const double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    CHECK_EQ(p.out.tool_calls.size(), 1u);
    CHECK_EQ(p.out.tool_calls[0].arguments.at("a").get<std::string>().size(), size_t(1 + 5 * 99999));
    CHECK_EQ(p.out.tool_calls[0].arguments.at("b").get<int>(), 1);
    CHECK_EQ(p.rules.quote_recovered, 2u * 99999);
    std::printf("100000 comma-separated strings in one argument: %.2f ms\n", ms2);
    CHECK(ms2 < 200.0);
    // The same with no argument after them: the string never closes properly, the block is content - and fast.
    std::string no_end = "[f(a=";
    for (int i = 0; i < 100000; ++i) no_end += "'a', ";
    const auto t3 = std::chrono::steady_clock::now();
    const Run ne = stream(gen("", block(no_end)), {});
    const double ms4 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t3).count();
    CHECK(ne.out.tool_calls.empty());
    CHECK_EQ(ne.out.content, block(no_end));
    std::printf("100000 comma-separated strings, no end: %.2f ms\n", ms4);
    CHECK(ms4 < 200.0);
    // A long streamed call (the byte-at-a-time attempts must not go quadratic).
    std::string longcall = "[f(a='";
    for (int i = 0; i < 20000; ++i) longcall += "word) ";
    longcall += "')]";
    const auto t2 = std::chrono::steady_clock::now();
    const Run l = stream(gen("", block(longcall)), {1});
    const double ms3 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
    CHECK_EQ(l.out.tool_calls.size(), 1u);
    std::printf("a 120 KB argument streamed byte by byte: %.2f ms\n", ms3);
    CHECK(ms3 < 2000.0);
  }
}

void counters() {
  // Counters add up across calls and blocks, and only successful parses count.
  const std::string text = gen("", block("[f(a='it's'), 'g'(\"b\"=true)]") + "x" + block("h(c={'d': 1})") + kEnd);
  const Run r = stream(text, {});
  CHECK_EQ(r.rules.quote_recovered, 1u);
  CHECK_EQ(r.rules.quoted_name, 2u);
  CHECK_EQ(r.rules.json_literal, 1u);
  CHECK_EQ(r.rules.bare_call, 1u);
  CHECK_EQ(r.rules.stray_end, 1u);
  CHECK_EQ(r.rules.positional_dict, 0u);
  expect("counters", text, {"", "x", json::array({C("f", {{"a", "it's"}}), C("g", {{"b", true}}), C("h", {{"c", {{"d", 1}}}})})},
         &r.rules);
  // An invalid block counts nothing.
  const Run bad = stream(gen("", block("['g'(1)]")), {});
  CHECK(same_rules(bad.rules, rules()));
  // The public block parser.
  uint64_t seed = 1;
  PythonicRules pr;
  auto none = server::parse_pythonic_calls("[f(1)]", true, seed, pr);
  CHECK(!none.has_value());
  auto cut = server::parse_pythonic_calls("[f(a='x", false, seed, pr);
  CHECK(cut.has_value() && cut->empty());
  auto two = server::parse_pythonic_calls("[f(a=1), g()]", true, seed, pr);
  CHECK(two.has_value() && two->size() == 2);
  CHECK_EQ((*two)[0].name, std::string("f"));
  CHECK((*two)[0].id != (*two)[1].id);
  CHECK((*two)[0].id.rfind("call_", 0) == 0 && (*two)[0].id.size() == 29);
}

// ---------------------------------------------------------------------------------------------------------------
// A fuzz: generated outputs parse to the generated structure; damaged ones parse identically under every split.
// ---------------------------------------------------------------------------------------------------------------

std::string py_string(std::mt19937& rng, bool nested_ctx) {
  static const char* const pieces[] = {"a", "b", " ", "x y", "'", "\"", "\\", "\n", "\t", "\r", ",", ")", "]", "}", "(", "=",
                                       "\xc3\xa9", "\xe4\xb8\x9c", "\xf0\x9f\x98\x80", "0", "<", "|", "True", "tool_call"};
  std::string s;
  const int n = rng() % 8;
  for (int i = 0; i < n; ++i) s += pieces[rng() % (sizeof(pieces) / sizeof(*pieces))];
  (void)nested_ctx;
  return s;
}

json random_value(std::mt19937& rng, int depth) {
  switch (rng() % (depth > 2 ? 6 : 8)) {
    case 0: return int(rng() % 2000) - 1000;
    case 1: return (int(rng() % 20000) - 10000) / 8.0;
    case 2: return rng() % 2 == 0;
    case 3: return nullptr;
    case 4:
    case 5: return py_string(rng, false);
    case 6: {
      json a = json::array();
      for (unsigned i = rng() % 4; i > 0; --i) a.push_back(random_value(rng, depth + 1));
      return a;
    }
    default: {
      json o = json::object();
      for (unsigned i = rng() % 4; i > 0; --i) o[py_string(rng, true)] = random_value(rng, depth + 1);
      return o;
    }
  }
}

// The template's writer (chat_template.jinja's format_arg_value), reimplemented here for the fuzz.
std::string write_arg(const json& v) {
  auto replace_all = [](std::string s, const std::string& from, const std::string& to) {
    for (size_t at = 0; (at = s.find(from, at)) != std::string::npos; at += to.size()) s.replace(at, from.size(), to);
    return s;
  };
  if (v.is_string()) {
    std::string s = v.get<std::string>();
    s = replace_all(s, "\\", "\\\\");
    s = replace_all(s, "'", "\\'");
    s = replace_all(s, "\n", "\\n");
    s = replace_all(s, "\r", "\\r");
    return "'" + s + "'";
  }
  if (v.is_array() || v.is_object()) return v.dump(-1, ' ', false);   // tojson (ensure_ascii off)
  if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
  if (v.is_null()) return "None";
  return v.dump();
}

void fuzz() {
  std::mt19937 rng(20241020);
  int generated = 0, damaged = 0;
  for (int it = 0; it < 1500; ++it) {
    json args = json::object();
    json calls = json::array();
    std::string inner;
    const int ncalls = rng() % 4;
    for (int c = 0; c < ncalls; ++c) {
      json a = json::object();
      std::string text;
      for (unsigned i = rng() % 4; i > 0; --i) {
        const std::string key = "k" + std::to_string(rng() % 100);
        a[key] = random_value(rng, 0);
      }
      for (const auto& [k, v] : a.items()) text += (text.empty() ? "" : ", ") + k + "=" + write_arg(v);
      const std::string name = "fn" + std::to_string(c);
      calls.push_back(C(name, a));
      inner += (c ? ", " : "") + name + "(" + text + ")";
    }
    const std::string reasoning = py_string(rng, false);
    const std::string content = py_string(rng, false);
    std::string content_clean;
    for (size_t i = 0; i < content.size(); ++i) content_clean += content[i];
    std::string text = reasoning + "</think>" + content;
    if (ncalls > 0) text += kStart + "[" + inner + "]" + kEnd;
    // The generated reasoning / content never hold a tag.
    Expect e{reasoning, content, calls};
    const Run whole = stream(text, {});
    expect_one("fuzz generated", text, e, nullptr, whole, "whole");
    expect_one("fuzz generated", text, e, nullptr, stream(text, {1}), "byte at a time");
    ++generated;
    // Damaged: cut at a random byte, or a byte flipped; every split agrees with the whole parse.
    for (int d = 0; d < 2; ++d) {
      std::string bad = text;
      if (!bad.empty()) {
        if (d == 0) bad.resize(rng() % bad.size());
        else bad[rng() % bad.size()] = "'\"()[]{},=\\<|"[rng() % 14];
      }
      const Run w = stream(bad, {});
      for (int k = 0; k < 4; ++k) {
        std::vector<size_t> sizes;
        for (int j = 0; j < 8; ++j) sizes.push_back(1 + rng() % 9);
        const Run p = stream(bad, sizes);
        if (show(p.out) != show(w.out) || !same_rules(p.rules, w.rules)) {
          std::fprintf(stderr, "fuzz damaged: split-dependent parse of %s\n  whole: %s %s\n  split: %s %s\n",
                       dump_safe(bad).c_str(), show(w.out).c_str(), show(w.rules).c_str(), show(p.out).c_str(),
                       show(p.rules).c_str());
          std::exit(1);
        }
      }
      ++damaged;
    }
  }
  std::printf("fuzz: %d generated outputs parsed to their structure, %d damaged ones split-invariant\n", generated,
              damaged);
}


// ---------------------------------------------------------------------------------------------------------------
// The chat format (plan 24e Task 3): what b70-serve selects for config.json's model_type lfm2_moe.
// ---------------------------------------------------------------------------------------------------------------

template <class F>
bool throws_naming(F&& f, const char* what) {
  try {
    f();
  } catch (const std::invalid_argument& e) {
    return std::string(e.what()).find(what) != std::string::npos;
  }
  return false;
}

void chat_format() {
  using server::ChatFormat;
  const ChatFormat f = ChatFormat::for_model_type("lfm2_moe");
  CHECK(f.kind == ChatFormat::Kind::Lfm2);
  CHECK_EQ(std::string(f.name()), std::string("lfm2_moe"));
  CHECK(f.template_kwargs());     // the request's chat_template_kwargs reach the template (preserve_thinking)
  CHECK(!f.string_content());     // content lists go to the template as they are (its parse_content joins the parts)
  // Every other model type keeps its format; the other LFM2 family members are not this one.
  CHECK(ChatFormat::for_model_type("qwen3_5").kind == ChatFormat::Kind::Qwen);
  CHECK(ChatFormat::for_model_type("lfm2").kind == ChatFormat::Kind::Qwen);
  CHECK(ChatFormat::for_model_type("kolibri1").kind == ChatFormat::Kind::Kolibri);
  CHECK(ChatFormat::for_model_type("muse_glimmer").kind == ChatFormat::Kind::Muse);
  CHECK(ChatFormat::for_model_type("k2_horizon").kind == ChatFormat::Kind::K2Horizon);

  const json messages = json::parse(R"([{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
                                        {"role":"assistant","content":null,"tool_calls":[]}])");
  CHECK_EQ(server::template_messages(f, messages), messages);

  // chat_template_kwargs: preserve_thinking must be a boolean; enable_thinking is accepted (any boolean, ignored: the
  // template always reasons); anything else passes through to the template unchecked.
  server::check_template_kwargs(f, json::object());
  server::check_template_kwargs(f, json::parse(R"({"preserve_thinking": true})"));
  server::check_template_kwargs(f, json::parse(R"({"preserve_thinking": false, "enable_thinking": false, "x": 1})"));
  server::check_template_kwargs(f, json(nullptr));
  CHECK(throws_naming([&] { server::check_template_kwargs(f, json::parse(R"({"preserve_thinking": "yes"})")); },
                      "preserve_thinking"));
  CHECK(throws_naming([&] { server::check_template_kwargs(f, json::parse(R"({"preserve_thinking": null})")); },
                      "preserve_thinking"));
  CHECK_EQ(server::request_template_kwargs(f, json::parse(R"({"preserve_thinking": true})"), json("high")),
           json::parse(R"({"preserve_thinking": true})"));   // the OpenAI reasoning_effort is not this model's

  // The output parser: the generation prompt ends in <think>, so the output opens in reasoning.
  const std::string prompt = "<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>";
  const auto run = [&](const std::string& p, const std::string& text) {
    std::unique_ptr<server::OutputParser> parser = server::make_output_parser(f, p, json::object(), json::array());
    std::vector<Delta> deltas = parser->push(text);
    for (Delta& d : parser->finish()) deltas.push_back(std::move(d));
    return collect(deltas);
  };
  const ParsedOutput a = run(prompt, "think</think>answer" + block("[f(a=1)]"));
  CHECK_EQ(a.reasoning, std::string("think"));
  CHECK_EQ(a.content, std::string("answer"));
  CHECK_EQ(a.tool_calls.size(), size_t(1));
  CHECK_EQ(a.tool_calls[0].name, std::string("f"));
  // A prompt that does not end in <think> (not what the template writes): the output is content from its first byte.
  const ParsedOutput b = run("<|im_start|>assistant\n", "answer</think>");
  CHECK_EQ(b.reasoning, std::string());
  CHECK_EQ(b.content, std::string("answer</think>"));
  // The parser's counters are reachable through the OutputParser interface (--log-requests).
  std::unique_ptr<server::OutputParser> parser = server::make_output_parser(f, prompt, json::object(), json::array());
  (void)parser->push("</think>" + block("['g'(\"b\"=true)]"));
  (void)parser->finish();
  const json stats = parser->stats();
  CHECK_EQ(stats.at("rules").at("quoted_name").get<int>(), 2);
  CHECK_EQ(stats.at("rules").at("json_literal").get<int>(), 1);
  CHECK(server::make_output_parser(ChatFormat{}, "<think>\n", json::object(), json::array())->stats().is_null());
  std::printf("chat format: ok\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: toolcall_lfm2_test <tests/tokenizer> <tests/golden/prompts> <snapshot dir>\n");
    return 2;
  }
  const std::string dir = argv[1], prompts = argv[2], snapshot = argv[3];
  round_trip(dir, prompts, snapshot);
  literals();
  escapes();
  recovery();
  idioms();
  invalids();
  cut_blocks();
  reasoning();
  limits();
  counters();
  fuzz();
  chat_format();
  std::printf("toolcall_lfm2_test OK\n");
  return 0;
}

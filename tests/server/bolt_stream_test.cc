// Spec 25e Task 2: Nemotron 3.5 Bolt's output stream (server/toolcall_bolt.h) - the `ultra_v3` reasoning rules over the Qwen3-Coder
// XML calls of server/toolcall.h.
//   bolt_stream_test <tests/server>
// Every parse is checked whole, byte at a time, at every two-way split and in random 1-7 byte pieces (spec 7a's streaming rule:
// the deltas of any split concatenate to the whole-text parse - call ids aside, each stream draws its own).
//   (a) tests/server/bolt_parse_cases.json (tools/tokenizer/dump_bolt.py parse: expectations written by hand from
//       ultra_v3_reasoning_parser.py's rules and the qwen3_coder format, each cross-checked in the dump against a restatement of
//       the plugin and against score.py's `qwen-think` reader): every case parses as expected - Review Focus 2 (reasoning follows
//       the prompt's end, verbatim; thinking off makes a stray </think> content; force_nonempty_content turns reasoning that
//       never closed into content) and 3 (the Qwen parser's calls; a call inside reasoning is reasoning; a call cut by the end of
//       generation is dropped and stripped); for the cases carrying the scorer's first call, the server's equals it by name and
//       by every raw argument (a string equals the raw text, any other value is the raw text's JSON);
//   (b) streaming facts: reasoning streams verbatim up to the longest tail that may still become </think>; with
//       force_nonempty_content nothing streams until the first byte behind </think> (or the end); thinking off streams content;
//   (c) the Qwen / Agnes OutputStream is unchanged (a cut call is still emitted when its function tag is complete);
//   (d) a fuzz: 2000 random outputs (reasoning, tags, calls complete and cut, values with markers, multi-byte text; thinking on /
//       off, force on / off) - every whole parse equals its byte-at-a-time, two-way-split and 1-7-byte streams, and an independent
//       model of the reasoning rule (text before the first </think>, verbatim) holds.
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
#include "server/toolcall_bolt.h"

namespace {

using json = nlohmann::json;
using server::BoltOutputStream;
using server::Delta;
using server::ParsedOutput;

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

// What a stream yielded, as an ordered list: adjacent text deltas of a kind merged, a call as its name and arguments.
json merged(const std::vector<Delta>& deltas) {
  json out = json::array();
  uint32_t next_index = 0;
  for (const Delta& d : deltas) {
    if (d.kind == Delta::Call) {
      CHECK_EQ(d.index, next_index);
      ++next_index;
      out.push_back({{"call", d.call.name}, {"arguments", d.call.arguments}});
      continue;
    }
    const char* kind = d.kind == Delta::Reasoning ? "reasoning" : "content";
    if (d.text.empty()) {
      std::fprintf(stderr, "an empty %s delta\n", kind);
      std::exit(1);
    }
    if (!out.empty() && out.back().contains(kind)) out.back()[kind] = out.back()[kind].get<std::string>() + d.text;
    else out.push_back({{kind, d.text}});
  }
  return out;
}

std::vector<Delta> stream(const std::string& text, const std::vector<size_t>& cuts, bool thinking, bool force,
                          const json& tools) {
  BoltOutputStream s(thinking, force, tools);
  std::vector<Delta> out;
  size_t at = 0;
  const auto push = [&](size_t end) {
    for (Delta& d : s.push(text.substr(at, end - at))) out.push_back(std::move(d));
    at = end;
  };
  for (size_t c : cuts) push(c);
  push(text.size());
  for (Delta& d : s.finish()) out.push_back(std::move(d));
  return out;
}

// Whole == byte at a time == random pieces (and, for short texts, every two-way split).
size_t check_streams(const std::string& what, const std::string& text, bool thinking, bool force, const json& tools,
                     std::mt19937_64& rng, bool every_split) {
  const json whole = merged(stream(text, {}, thinking, force, tools));
  size_t runs = 1;
  std::vector<size_t> bytes;
  for (size_t i = 1; i < text.size(); ++i) bytes.push_back(i);
  if (merged(stream(text, bytes, thinking, force, tools)) != whole) {
    std::fprintf(stderr, "%s: byte-at-a-time differs from the whole parse\n  text: %s\n  whole: %s\n  bytes: %s\n", what.c_str(),
                 json(text).dump().substr(0, 400).c_str(), whole.dump().substr(0, 600).c_str(),
                 merged(stream(text, bytes, thinking, force, tools)).dump().substr(0, 600).c_str());
    std::exit(1);
  }
  ++runs;
  if (every_split) {
    for (size_t c = 1; c < text.size(); ++c) {
      if (merged(stream(text, {c}, thinking, force, tools)) != whole) {
        std::fprintf(stderr, "%s: a split at byte %zu differs from the whole parse\n  text: %s\n", what.c_str(), c,
                     json(text).dump().substr(0, 400).c_str());
        std::exit(1);
      }
      ++runs;
    }
  }
  for (int rep = 0; rep < 3; ++rep) {
    std::vector<size_t> cuts;
    for (size_t at = 0;;) {
      at += 1 + rng() % 7;
      if (at >= text.size()) break;
      cuts.push_back(at);
    }
    if (merged(stream(text, cuts, thinking, force, tools)) != whole) {
      std::fprintf(stderr, "%s: random 1-7 byte pieces differ from the whole parse\n  text: %s\n", what.c_str(),
                   json(text).dump().substr(0, 400).c_str());
      std::exit(1);
    }
    ++runs;
  }
  return runs;
}

// ---- (a) the cases ----------------------------------------------------------------------------------------------------
void check_cases(const std::string& dir, std::mt19937_64& rng) {
  const json facts = json::parse(slurp(dir + "/bolt_parse_cases.json"));
  size_t n = 0, runs = 0, scored = 0;
  for (const json& c : facts.at("cases")) {
    const std::string name = c.at("name");
    const std::string text = c.at("text");
    const bool thinking = c.at("thinking"), force = c.at("force_nonempty_content");
    const json& tools = c.at("tools");
    runs += check_streams(name, text, thinking, force, tools, rng, true);
    const ParsedOutput got = server::parse_bolt_output(text, thinking, force, tools);
    const json& e = c.at("expect");
    if (got.reasoning != e.at("reasoning").get<std::string>() || got.content != e.at("content").get<std::string>() ||
        got.tool_calls.size() != e.at("tool_calls").size()) {
      std::fprintf(stderr, "%s: reasoning %s content %s calls %zu\n  want reasoning %s content %s calls %zu\n", name.c_str(),
                   json(got.reasoning).dump().c_str(), json(got.content).dump().c_str(), got.tool_calls.size(),
                   e.at("reasoning").dump().c_str(), e.at("content").dump().c_str(), e.at("tool_calls").size());
      std::exit(1);
    }
    for (size_t i = 0; i < got.tool_calls.size(); ++i) {
      CHECK_EQ(got.tool_calls[i].name, e.at("tool_calls").at(i).at("name").get<std::string>());
      if (got.tool_calls[i].arguments != e.at("tool_calls").at(i).at("arguments")) {
        std::fprintf(stderr, "%s: call %zu arguments %s, want %s\n", name.c_str(), i, got.tool_calls[i].arguments.dump().c_str(),
                     e.at("tool_calls").at(i).at("arguments").dump().c_str());
        std::exit(1);
      }
      CHECK(got.tool_calls[i].id.empty() || got.tool_calls[i].id.rfind("call_", 0) == 0);
    }
    // The A4 scorer's first call: its name and every raw argument.
    if (c.contains("scorer")) {
      const json& s = c.at("scorer");
      CHECK(!got.tool_calls.empty());
      CHECK_EQ(got.tool_calls[0].name, s.at("name").get<std::string>());
      for (const auto& [key, raw] : s.at("params").items()) {
        CHECK(got.tool_calls[0].arguments.contains(key));
        const json& v = got.tool_calls[0].arguments.at(key);
        const std::string r = raw.get<std::string>();
        const json parsed = json::parse(r, nullptr, false);
        CHECK((v.is_string() && v.get<std::string>() == r) || (!parsed.is_discarded() && v == parsed));
      }
      CHECK_EQ(got.tool_calls[0].arguments.size(), s.at("params").size());
      ++scored;
    }
    ++n;
  }
  std::printf("parse cases: %zu equal to their expectations (%zu runs: whole, bytes, every two-way split, random pieces), %zu "
              "first calls equal to score.py's qwen-think reading\n", n, runs, scored);
}

// ---- (b) streaming facts ----------------------------------------------------------------------------------------------
std::string kinds_of(const std::vector<Delta>& d) {
  std::string s;
  for (const Delta& x : d) s += x.kind == Delta::Reasoning ? 'R' : (x.kind == Delta::Content ? 'C' : 'T');
  return s;
}

void check_streaming_facts() {
  const json none;
  {
    // Reasoning streams verbatim, holding back only what may still be </think>.
    BoltOutputStream s(true, false, none);
    std::vector<Delta> d = s.push("  a b ");
    CHECK_EQ(kinds_of(d), std::string("R"));
    CHECK_EQ(d[0].text, std::string("  a b "));
    d = s.push("</th");
    CHECK(d.empty());
    d = s.push("inking> x</thi");
    CHECK_EQ(kinds_of(d), std::string("R"));
    CHECK_EQ(d[0].text, std::string("</thinking> x"));
    d = s.push("nk>Hello");
    CHECK_EQ(kinds_of(d), std::string("C"));
    CHECK_EQ(d[0].text, std::string("Hello"));
    d = s.push(" world");
    CHECK_EQ(d[0].text, std::string(" world"));
    CHECK(s.finish().empty());
  }
  {
    // force_nonempty_content: nothing streams until a byte follows </think>; then the reasoning, then the content.
    BoltOutputStream s(true, true, none);
    CHECK(s.push("hold this ").empty());
    CHECK(s.push("</think>").empty());
    std::vector<Delta> d = s.push("x");
    CHECK_EQ(kinds_of(d), std::string("RC"));
    CHECK_EQ(d[0].text, std::string("hold this "));
    CHECK_EQ(d[1].text, std::string("x"));
    CHECK(s.finish().empty());
    BoltOutputStream t(true, true, none);
    CHECK(t.push("only reasoning</think>").empty());
    d = t.finish();
    CHECK_EQ(kinds_of(d), std::string("C"));
    CHECK_EQ(d[0].text, std::string("only reasoning"));
  }
  {
    // Thinking off: content from the first byte, a stray </think> included.
    BoltOutputStream s(false, true, none);
    std::vector<Delta> d = s.push("Answer</think> tail");
    std::string all;
    for (const Delta& x : d) {
      CHECK(x.kind == Delta::Content);
      all += x.text;
    }
    for (const Delta& x : s.finish()) all += x.text;
    CHECK_EQ(all, std::string("Answer</think> tail"));
  }
  {
    // A cut call and the content before it: the call is dropped at the end, the content already streamed stays.
    BoltOutputStream s(true, false, none);
    std::vector<Delta> all = s.push("r</think>Ok.\n<tool_call>\n<function=f>\n<parameter=a>\nv");
    for (Delta& x : s.finish()) all.push_back(x);
    json m = merged(all);
    CHECK_EQ(m.dump(), std::string("[{\"reasoning\":\"r\"},{\"content\":\"Ok.\\n\"}]"));
  }
  std::printf("streaming facts: ok\n");
}

// ---- (c) the Qwen stream is unchanged ----------------------------------------------------------------------------------
void check_qwen_unchanged() {
  const std::string cut = "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\nabc";
  const ParsedOutput q = server::parse_output(cut, false, json());
  CHECK_EQ(q.tool_calls.size(), size_t(1));   // the Qwen / Agnes behaviour: the cut call is emitted with what closed
  CHECK_EQ(q.tool_calls[0].name, std::string("f"));
  CHECK_EQ(q.tool_calls[0].arguments.dump(), std::string("{\"a\":1}"));
  const ParsedOutput b = server::parse_bolt_output(cut, false, false, json());
  CHECK(b.tool_calls.empty());
  CHECK(b.content.empty());
  std::printf("the Qwen stream is unchanged (a cut call emitted), Bolt's drops it\n");
}

// ---- (d) the fuzz ------------------------------------------------------------------------------------------------------
std::string pick(std::mt19937_64& rng, const std::vector<std::string>& v) { return v[rng() % v.size()]; }

std::string random_value(std::mt19937_64& rng) {
  return pick(rng, {"x", "42", "true", "null", "[1, 2]", "{\"k\": \"v\"}", "a </parameter> b", "line1\nline2", "  padded  ",
                    "日本語 😀", "<tool_call>", "</function>", "007", ""});
}

std::string random_call(std::mt19937_64& rng) {
  std::string s = "<tool_call>\n<function=" + pick(rng, {"read", "edit", "mystery"}) + ">\n";
  for (size_t p = rng() % 3; p > 0; --p)
    s += "<parameter=" + pick(rng, {"filePath", "oldString", "n", "flag"}) + ">\n" + random_value(rng) + "\n</parameter>\n";
  return s + "</function>\n</tool_call>";
}

std::string random_text(std::mt19937_64& rng, bool thinking) {
  std::string s;
  if (thinking) {
    for (size_t i = rng() % 4; i > 0; --i)
      s += pick(rng, {"Let me think ", "x = 1\n", "…東京😀 ", " ", "\n", "<tool_call>", "</thin", "</think", "<think>", "</thinking>",
                      "a</", "<function=f>", random_call(rng)});
    if (rng() % 5 != 0) s += "</think>";
  }
  for (size_t i = rng() % 5; i > 0; --i) {
    const size_t k = rng() % 6;
    if (k == 0) s += random_call(rng);
    else if (k == 1) s += pick(rng, {"\n", "\n\n", " ", "  \n"});
    else if (k == 2) s += pick(rng, {"</think>", "<think>", "<tool_c", "<tool_call>\n<function=f>\n", "</tool_call>"});
    else s += pick(rng, {"Hello ", "Grüße ", "code: `x`", "😀", "answer.\n", "…"});
  }
  if (rng() % 4 == 0 && !s.empty()) s.resize(s.size() - 1 - rng() % std::min<size_t>(s.size(), 12));   // a cut anywhere
  return s;
}

void check_fuzz(std::mt19937_64& rng) {
  const json tools = json::array({{{"type", "function"}, {"function", {{"name", "edit"}, {"parameters", {{"type", "object"},
      {"properties", {{"oldString", {{"type", "string"}}}, {"n", {{"type", "integer"}}}}}}}}}}});
  size_t runs = 0, with_calls = 0, with_reasoning = 0, forced_moves = 0;
  for (int i = 0; i < 2000; ++i) {
    const bool thinking = rng() % 3 != 0, force = rng() % 3 == 0;
    const std::string text = random_text(rng, thinking);
    runs += check_streams("fuzz " + std::to_string(i), text, thinking, force, tools, rng, text.size() < 60);
    const ParsedOutput p = server::parse_bolt_output(text, thinking, force, tools);
    with_calls += p.tool_calls.empty() ? 0 : 1;
    with_reasoning += p.reasoning.empty() ? 0 : 1;
    // The independent model of the reasoning rule: thinking off - none; thinking on - the text before the first </think>
    // (all of it when there is none), verbatim - unless force_nonempty_content moves it to content (no </think>, or nothing
    // after it).
    if (!thinking) {
      CHECK(p.reasoning.empty());
      continue;
    }
    const size_t at = text.find("</think>");
    const bool closed = at != std::string::npos;
    const bool moved = force && (!closed || at + 8 == text.size());
    if (moved) {
      CHECK(p.reasoning.empty());
      CHECK_EQ(p.content, closed ? text.substr(0, at) : text);
      CHECK(p.tool_calls.empty());
      ++forced_moves;
    } else {
      CHECK_EQ(p.reasoning, closed ? text.substr(0, at) : text);
      if (!closed) CHECK(p.content.empty() && p.tool_calls.empty());
    }
  }
  std::printf("fuzz: 2000 outputs, %zu stream runs, %zu with calls, %zu with reasoning, %zu moved by force_nonempty_content - "
              "whole == bytes == random pieces\n", runs, with_calls, with_reasoning, forced_moves);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: bolt_stream_test <tests/server>\n");
    return 2;
  }
  std::mt19937_64 rng(25025);
  check_cases(argv[1], rng);
  check_streaming_facts();
  check_qwen_unchanged();
  check_fuzz(rng);
  // The format's dispatch: Bolt's parser follows the prompt's end.
  {
    const server::ChatFormat f = server::ChatFormat::for_model_type("nemotron_h_omni");
    CHECK(f.kind == server::ChatFormat::Kind::Bolt);
    auto thinking = server::make_output_parser(f, "<|im_start|>assistant\n<think>\n", json::object(), json());
    auto off = server::make_output_parser(f, "<|im_start|>assistant\n<think></think>", json::object(), json());
    std::vector<Delta> d = thinking->push("abc</think>");
    CHECK_EQ(kinds_of(d), std::string("R"));
    d = off->push("abc</think>");
    CHECK_EQ(kinds_of(d), std::string("C"));
    auto forced = server::make_output_parser(f, "<|im_start|>assistant\n<think>\n", {{"force_nonempty_content", true}}, json());
    CHECK(forced->push("abc").empty());
    CHECK_EQ(kinds_of(forced->finish()), std::string("C"));
  }
  std::printf("bolt_stream_test OK\n");
  return 0;
}

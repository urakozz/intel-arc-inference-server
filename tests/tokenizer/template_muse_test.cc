// Spec 23e Task 1: Muse-Glimmer-30B's chat template (the ORIGINAL's chat_template.jinja, sha256 cfc67e5f... - spec 23
// decision 1; vendored at tests/tokenizer/muse/ with its tokenizer_config.json and generation_config.json, Apache-2.0)
// through chat::Template (minja) against transformers 5.19.0's apply_chat_template, from
// tools/tokenizer/dump_muse.py's tests/tokenizer/muse_templates.json:
//   template_muse_test <tests/tokenizer> <tests/tokenizer/muse>
//   - every case renders byte for byte as transformers' (no system message / a system message with and without a
//     reasoning line, the four "Reasoning effort" casings, reasoning_strength low / medium / high / xhigh,
//     knowledge_cutoff, tools with dotted and bare names, tool_namespace_descriptions, call history with one and with
//     parallel calls carrying every ATEM value type, tool results by name and by tool_call_id, reasoning_content,
//     recipient / end_turn overrides, consecutive assistant messages, add_generation_prompt off, list content,
//     Unicode); every case passes current_date, so no render reads the clock;
//   - a tool call whose arguments are a JSON STRING makes the template itself raise (T:28-30; minja without
//     polyfills, as transformers' Jinja); chat::Template's minja polyfill renders a string holding an object as the
//     object (the probe finds requires_object_arguments - third_party/minja's spec 23e patch); the server parses
//     string arguments before rendering anyway (server/openai.cc: a string that is not an object is a 400);
//   - strftime_now is provided: without current_date, with the clock fixed at 2026-10-15 (local time, as
//     transformers' strftime_now), the render equals the case rendered with current_date "2026-10-15";
//   - the source's sha256 is the original's; --chat-template FILE (chat::Template's second constructor) renders the
//     same file from another path identically.
#include <chrono>
#include <cstdio>
#include <ctime>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>

#include "check.h"
#include "tokenizer/chat_template.h"

namespace {

using Json = nlohmann::json;

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

// The case's kwargs, with add_generation_prompt as a template variable (it overrides chat::Template's default true:
// minja sets the request's variables after its own).
Json kwargs_of(const Json& c) {
  Json kw = c.at("kwargs");
  kw["add_generation_prompt"] = c.at("add_generation_prompt").get<bool>();
  return kw;
}

std::chrono::system_clock::time_point local_noon(int y, int m, int d) {
  std::tm tm{};
  tm.tm_year = y - 1900;
  tm.tm_mon = m - 1;
  tm.tm_mday = d;
  tm.tm_hour = 12;
  tm.tm_isdst = -1;
  return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: template_muse_test <tests/tokenizer> <tests/tokenizer/muse>\n");
    return 2;
  }
  const std::string dir = argv[1], muse = argv[2];
  const Json facts = Json::parse(slurp(dir + "/muse_templates.json"));
  chat::Template tmpl(muse);
  CHECK_EQ(tmpl.source_sha256(), facts.at("chat_template_sha256").get<std::string>());
  CHECK_EQ(tmpl.source_sha256().substr(0, 8), std::string("cfc67e5f"));
  CHECK_EQ(tmpl.bos_token(), facts.at("bos_token").get<std::string>());
  CHECK_EQ(tmpl.eos_token(), facts.at("eos_token").get<std::string>());

  int bad = 0;
  for (const Json& c : facts.at("cases")) {
    const std::string name = c.at("name");
    std::string got;
    try {
      got = tmpl.render(c.at("messages"), c.at("tools"), true, kwargs_of(c));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s: the render threw: %.300s\n", name.c_str(), e.what());
      ++bad;
      continue;
    }
    const std::string want = c.at("render");
    if (got != want) {
      diff_report(name, got, want);
      ++bad;
    } else {
      std::printf("%s: %zu bytes, identical\n", name.c_str(), got.size());
    }
  }
  CHECK_EQ(bad, 0);

  // String arguments: the template itself raises (minja without polyfills, as transformers' Jinja does); chat::Template
  // (minja's object-arguments polyfill, the probe finding requires_object_arguments) renders a string that holds an
  // object exactly as the object - and the server parses string arguments before rendering anyway, refusing one
  // that is not a JSON object with a 400 (server/openai.cc).
  {
    const Json& s = facts.at("string_arguments_case");
    bool threw = false;
    minja::chat_template raw(slurp(muse + "/chat_template.jinja"), tmpl.bos_token(), tmpl.eos_token());
    CHECK(raw.original_caps().supports_tool_calls);
    CHECK(raw.original_caps().requires_object_arguments);
    try {
      minja::chat_template_inputs in;
      in.messages = s.at("messages");
      in.tools = s.at("tools");
      in.extra_context = {{"current_date", "2026-10-15"}};
      minja::chat_template_options no_polyfills;
      no_polyfills.apply_polyfills = false;
      (void)raw.apply(in, no_polyfills);
    } catch (const std::exception& e) {
      threw = std::string(e.what()).find("mapping") != std::string::npos;
      if (!threw) std::fprintf(stderr, "string arguments: threw without naming the mapping: %.200s\n", e.what());
    }
    CHECK(threw);
    Json parsed = s.at("messages");
    for (Json& m : parsed)
      if (m.contains("tool_calls"))
        for (Json& tc : m.at("tool_calls"))
          tc.at("function")["arguments"] = Json::parse(tc.at("function").at("arguments").get<std::string>());
    const Json kw = {{"current_date", "2026-10-15"}};
    CHECK_EQ(tmpl.render(s.at("messages"), s.at("tools"), true, kw), tmpl.render(parsed, s.at("tools"), true, kw));
    std::printf("string arguments: the template raises naming the mapping (transformers: %.60s...); chat::Template "
                "renders a string holding an object as the object\n", s.at("error").get<std::string>().c_str());
  }

  // strftime_now: the clock fixed at 2026-10-15, no current_date -> the current_date render.
  {
    chat::Template clocked(muse);
    clocked.set_clock([] { return local_noon(2026, 10, 15); });
    for (const Json& c : facts.at("cases")) {
      if (c.at("name") != "no_system" && c.at("name") != "tools_dotted") continue;
      Json kw = kwargs_of(c);
      CHECK_EQ(kw.at("current_date").get<std::string>(), std::string("2026-10-15"));
      kw.erase("current_date");
      const std::string got = clocked.render(c.at("messages"), c.at("tools"), true, kw);
      if (got != c.at("render").get<std::string>()) {
        diff_report(c.at("name").get<std::string>() + " (strftime_now)", got, c.at("render"));
        CHECK(false);
      }
    }
    clocked.set_clock([] { return local_noon(2027, 1, 2); });
    const Json& c = facts.at("cases").at(0);
    Json kw = kwargs_of(c);
    kw.erase("current_date");
    const std::string got = clocked.render(c.at("messages"), c.at("tools"), true, kw);
    CHECK(got.find("Current date: 2027-01-02.") != std::string::npos);
    std::printf("strftime_now: provided, the render reads the (fixed) clock when current_date is absent\n");
  }

  // --chat-template FILE: the same source from another path renders identically.
  {
    chat::Template other(muse, muse + "/chat_template.jinja");
    CHECK_EQ(other.source_sha256(), tmpl.source_sha256());
    const Json& c = facts.at("cases").back();
    CHECK_EQ(other.render(c.at("messages"), c.at("tools"), true, kwargs_of(c)), c.at("render").get<std::string>());
  }
  std::printf("template_muse_test OK (%zu cases)\n", facts.at("cases").size());
  return 0;
}

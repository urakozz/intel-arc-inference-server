// Spec 25e Task 3: Nemotron 3.5 Bolt's request path - the template kwargs, OpenAI's reasoning_effort, the sampling defaults and
// EOS {2, 11} - through the server over a mock engine (no device, no checkpoint), the chat-format half of the wiring b70-serve
// uses for it (the engine half, BoltEngine behind cli::bolt::BoltEngineAdapterT, is bolt_snapshot_test's).
//   bolt_request_test <tests/tokenizer> <tests/model/bolt> <snapshot dir>
// <snapshot dir> (chat_template.jinja + tokenizer_config.json; B70_BOLT_SNAPSHOT replaces it) holds the REAL template, which the
// server renders with here; absent: SKIP (77). With Options::chat_format = ChatFormat::bolt():
//   - every case of tests/tokenizer/bolt_templates.json (Task 1's 80 renders, byte-checked against transformers 5.19.0) with a
//     generation prompt, POSTed as a chat request with the case's messages, tools and chat_template_kwargs, reaches the template
//     as exactly that render - each kwarg (enable_thinking, medium_effort, truncate_history_thinking, an unknown one passed
//     through) is the template's;
//   - Decision 13's reasoning_effort table: none / minimal -> enable_thinking false, low / medium -> medium_effort true, high /
//     xhigh -> the default; an explicit kwarg wins; anything else is a 400 naming reasoning_effort; each result is a render of
//     Task 1's, byte for byte;
//   - a non-boolean medium_effort / truncate_history_thinking / force_nonempty_content / enable_thinking is a 400 naming it;
//   - image / video / audio parts are refused by name (23e's list); text parts of every kind pass;
//   - Decision 12: the served export's generation_config.json (NVIDIA's keys: sampled, T 1.0, top-p 0.95, EOS [2, 11]) gives the
//     published set - no top-k -, NVIDIA's own file the same, a file with no sampling key (EOS 11 only) the published set with its
//     eos added, an explicit file its own, a malformed one a named error; EOS on either id ends a generation (finish_reason stop), a request's temperature / top_p / top_k override
//     the defaults, temperature 0 is greedy, a seeded request carries its seed;
//   - the output stream end to end (the body and the streamed frames): reasoning verbatim, content, a Qwen XML call
//     (tool_calls); thinking off (everything is content, a stray </think> included); force_nonempty_content; a call cut by
//     max_tokens dropped and stripped (finish_reason length), generation that ends inside reasoning is reasoning only.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/bolt_serve.h"
#include "server/chat_format.h"
#include "server/mock.h"
#include "server/openai.h"
#include "server/server.h"
#include "tokenizer/chat_template.h"

namespace {

using json = nlohmann::json;

bool exists(const std::string& path) { return std::ifstream(path).good(); }

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

// The real template, recording what the server asked it to render.
struct RecordingTemplate : server::TemplateIface {
  explicit RecordingTemplate(const std::string& snapshot) : tmpl(snapshot) {}
  std::string render(const json& messages, const json& tools, bool thinking) override {
    return render_with_kwargs(messages, tools, thinking, json::object());
  }
  std::string render_with_kwargs(const json& messages, const json& tools, bool thinking, const json& kwargs) override {
    ++renders;
    last_thinking = thinking;
    last_kwargs = kwargs;
    return last_prompt = tmpl.render(messages, tools, thinking, kwargs);
  }
  chat::Template tmpl;
  int renders = 0;
  bool last_thinking = true;
  json last_kwargs;
  std::string last_prompt;
};

// MockEngine with a long context, recording the sampling of every step.
struct RecordingEngine : MockEngine {
  std::vector<server::Sampling> sampling;
  uint32_t max_len() override { return 1u << 20; }
  uint32_t step(const server::Sampling& s) override {
    sampling.push_back(s);
    return MockEngine::step(s);
  }
};

std::string g_snapshot, g_fixture_dir, g_model_dir;
cli::bolt::Generation g_generation;   // the export's file, resolved - what b70-serve sets

struct Fixture {
  Fixture() : tmpl(g_snapshot), options(make_options()), server({tok, tmpl, engine}, options) { CHECK(server.start()); }
  ~Fixture() { server.stop(); }
  static server::Options make_options() {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "bolt";
    o.eos_ids = g_generation.eos;
    o.eos_ids.push_back(248046);   // the mock tokenizer's <|im_end|>: the script's end
    o.chat_format = server::ChatFormat::for_model_type("nemotron_h_omni");
    o.sampling_defaults = g_generation.sampling;
    return o;
  }
  httplib::Client client() { return httplib::Client("127.0.0.1", server.bound_port()); }
  // The engine generates `text` a few UTF-8 characters per token, then `eos` (0: none; the mock engine ends the script itself).
  void script(const std::string& text, size_t piece_chars = 3, uint32_t eos = 248046) {
    engine.script.clear();
    engine.at = 0;
    for (size_t at = 0; at < text.size();) {
      size_t n = 0;
      for (size_t chars = 0; chars < piece_chars && at + n < text.size(); ++chars) {
        ++n;
        while (at + n < text.size() && (static_cast<unsigned char>(text[at + n]) & 0xC0) == 0x80) ++n;
      }
      const std::vector<uint32_t> id = tok.raw(text.substr(at, n), n);
      engine.script.insert(engine.script.end(), id.begin(), id.end());
      at += n;
    }
    if (eos != 0) engine.script.push_back(eos);
  }

  MockTok tok;
  RecordingTemplate tmpl;
  RecordingEngine engine;
  server::Options options;
  server::Server server;
};

json base_body(const json& extra = json::object(), bool stream = false) {
  json b = {{"model", "bolt"}, {"messages", json::array({{{"role", "user"}, {"content", "Read the file a.cc please"}}})}};
  if (stream) {
    b["stream"] = true;
    b["stream_options"] = {{"include_usage", true}};
  }
  for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
  return b;
}

httplib::Result raw_post(Fixture& f, const json& b) {
  httplib::Client c = f.client();
  return c.Post("/v1/chat/completions", b.dump(), "application/json");
}

json post(Fixture& f, const json& b) {
  const auto r = raw_post(f, b);
  CHECK(r);
  if (r->status != 200) std::fprintf(stderr, "status %d: %s\n", r->status, r->body.c_str());
  CHECK_EQ(r->status, 200);
  return json::parse(r->body);
}

std::string error_of(Fixture& f, const json& b) {
  const auto r = raw_post(f, b);
  CHECK(r);
  CHECK_EQ(r->status, 400);
  return json::parse(r->body).at("error").at("message").get<std::string>();
}

struct Streamed {
  std::string reasoning, content, finish;
  std::vector<json> calls;
  json usage;
  size_t frames = 0;
};

Streamed post_stream(Fixture& f, const json& b) {
  const auto r = raw_post(f, b);
  CHECK(r);
  CHECK_EQ(r->status, 200);
  Streamed out;
  std::istringstream lines(r->body);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.rfind("data: ", 0) != 0) continue;
    const std::string data = line.substr(6);
    if (data == "[DONE]") continue;
    const json frame = json::parse(data);
    ++out.frames;
    if (frame.contains("usage") && !frame.at("usage").is_null()) out.usage = frame.at("usage");
    if (frame.at("choices").empty()) continue;
    const json& choice = frame.at("choices").at(0);
    if (!choice.at("finish_reason").is_null()) out.finish = choice.at("finish_reason");
    const json& delta = choice.at("delta");
    if (delta.contains("reasoning_content")) out.reasoning += delta.at("reasoning_content").get<std::string>();
    if (delta.contains("content") && !delta.at("content").is_null()) out.content += delta.at("content").get<std::string>();
    if (delta.contains("tool_calls"))
      for (const json& tc : delta.at("tool_calls")) out.calls.push_back(tc);
  }
  return out;
}

// (the name is a const char*, not a const std::string&: g++ 15's -Wdangling-reference flags a const& bound to the result of a
// function that takes a temporary std::string by reference)
const json& fixture_case(const json& facts, const char* name) {
  for (const json& c : facts.at("cases"))
    if (c.at("name") == name) return c;
  std::fprintf(stderr, "no fixture case %s\n", name);
  std::exit(1);
}

// ---- Task 1's renders through the server --------------------------------------------------------------------------------------
void case_fixture_renders(const json& facts) {
  Fixture f;
  size_t n = 0, n_refused = 0, thinking_off = 0, kwargs_cases = 0;
  for (const json& c : facts.at("cases")) {
    if (!c.at("add_generation_prompt").get<bool>()) continue;   // the server always adds the generation prompt
    json b = base_body();
    b["messages"] = c.at("messages");
    if (!c.at("tools").is_null()) b["tools"] = c.at("tools");
    if (!c.at("kwargs").empty()) b["chat_template_kwargs"] = c.at("kwargs");
    f.script("ok");
    const auto r = raw_post(f, b);
    CHECK(r);
    // The server refuses a content part that is not a text part (the media parts by name, 23e's list): the template renders
    // them (Task 1), the request path never lets one reach it.
    bool refused = false;
    for (const json& m : c.at("messages"))
      if (m.contains("content") && m.at("content").is_array())
        for (const json& part : m.at("content"))
          refused = refused || !part.is_object() || !part.contains("text") || !part.at("text").is_string() ||
                    (part.at("type") != "text" && part.at("type") != "input_text" && part.at("type") != "tool_use");
    if (refused) {
      CHECK_EQ(r->status, 400);
      ++n_refused;
      continue;
    }
    if (r->status != 200) {
      std::fprintf(stderr, "%s: status %d: %s\n", c.at("name").get<std::string>().c_str(), r->status, r->body.c_str());
      std::exit(1);
    }
    if (f.tmpl.last_prompt != c.at("render").get<std::string>()) {
      std::fprintf(stderr, "%s: the server's prompt differs from transformers' render\n  got : %s\n  want: %s\n",
                   c.at("name").get<std::string>().c_str(), json(f.tmpl.last_prompt.substr(0, 300)).dump().c_str(),
                   json(c.at("render").get<std::string>().substr(0, 300)).dump().c_str());
      std::exit(1);
    }
    ++n;
    thinking_off += c.at("kwargs").contains("enable_thinking") && !c.at("kwargs").at("enable_thinking").get<bool>() ? 1 : 0;
    kwargs_cases += c.at("kwargs").empty() ? 0 : 1;
  }
  std::printf("fixtures: %zu cases POSTed with their messages, tools and kwargs reach the template as transformers' render "
              "(%zu with a kwarg, %zu with thinking off); %zu more carry a media / non-text part and are refused with a 400\n", n,
              kwargs_cases, thinking_off, n_refused);
  CHECK(n >= 60);
}

// ---- Decision 13 --------------------------------------------------------------------------------------------------------------
void case_reasoning_effort(const json& facts) {
  Fixture f;
  const json& plain = fixture_case(facts, "system_plain");
  const json& off = fixture_case(facts, "generation_prompt_thinking_off");
  const json& medium = fixture_case(facts, "medium_effort");
  const auto prompt_for = [&](const json& extra) {
    json b = base_body(extra);
    b["messages"] = plain.at("messages");
    f.script("ok");
    post(f, b);
    return f.tmpl.last_prompt;
  };
  const std::string want_plain = plain.at("render"), want_off = off.at("render"), want_medium = medium.at("render");
  for (const char* e : {"none", "minimal"}) CHECK_EQ(prompt_for({{"reasoning_effort", e}}), want_off);
  for (const char* e : {"low", "medium"}) CHECK_EQ(prompt_for({{"reasoning_effort", e}}), want_medium);
  for (const char* e : {"high", "xhigh"}) CHECK_EQ(prompt_for({{"reasoning_effort", e}}), want_plain);
  CHECK_EQ(prompt_for({}), want_plain);
  CHECK_EQ(prompt_for({{"reasoning_effort", nullptr}}), want_plain);
  // An explicit kwarg wins over the mapping.
  CHECK_EQ(prompt_for({{"reasoning_effort", "none"}, {"chat_template_kwargs", {{"enable_thinking", true}}}}), want_plain);
  CHECK_EQ(prompt_for({{"reasoning_effort", "low"}, {"chat_template_kwargs", {{"medium_effort", false}}}}), want_plain);
  CHECK_EQ(prompt_for({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"medium_effort", true}}}}), want_medium);
  CHECK_EQ(prompt_for({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}), want_off);
  // The kwargs alone, and a mapping that adds a kwarg to the request's others.
  CHECK_EQ(prompt_for({{"chat_template_kwargs", {{"medium_effort", true}}}}), want_medium);
  const json& both = fixture_case(facts, "medium_effort_thinking_off");
  json b = base_body({{"reasoning_effort", "medium"}, {"chat_template_kwargs", {{"enable_thinking", false}}}});
  b["messages"] = both.at("messages");
  f.script("ok");
  post(f, b);
  CHECK_EQ(f.tmpl.last_prompt, both.at("render").get<std::string>());
  // The unmapped values.
  for (const json& bad : {json("max"), json("ultra"), json(""), json(3), json(true)}) {
    const std::string msg = error_of(f, base_body({{"reasoning_effort", bad}}));
    CHECK(msg.find("reasoning_effort must be one of none, minimal, low, medium, high, xhigh") != std::string::npos);
  }
  std::printf("reasoning_effort: none / minimal -> thinking off, low / medium -> medium_effort, high / xhigh -> the default, an "
              "explicit kwarg wins; each prompt is a fixture's render; anything else a 400\n");
}

void case_kwarg_errors() {
  Fixture f;
  for (const char* key : {"medium_effort", "truncate_history_thinking", "force_nonempty_content"}) {
    for (const json& bad : {json("yes"), json(1), json(nullptr), json::array()}) {
      const std::string msg = error_of(f, base_body({{"chat_template_kwargs", {{key, bad}}}}));
      CHECK(msg.find(std::string("chat_template_kwargs.") + key + " must be a boolean") != std::string::npos);
    }
    f.script("ok");
    post(f, base_body({{"chat_template_kwargs", {{key, true}}}}));
    f.script("ok");
    post(f, base_body({{"chat_template_kwargs", {{key, false}}}}));
  }
  const auto r = raw_post(f, base_body({{"chat_template_kwargs", {{"enable_thinking", "no"}}}}));
  CHECK(r && r->status == 400);
  const auto r2 = raw_post(f, base_body({{"chat_template_kwargs", "x"}}));
  CHECK(r2 && r2->status == 400);
  // An unknown kwarg is passed through unchecked (23e's rule).
  f.script("ok");
  post(f, base_body({{"chat_template_kwargs", {{"some_other_flag", {1, 2}}}}}));
  CHECK_EQ(f.tmpl.last_kwargs.at("some_other_flag"), json({1, 2}));
  std::printf("kwarg errors: a non-boolean medium_effort / truncate_history_thinking / force_nonempty_content / enable_thinking "
              "is a 400 naming it; an unknown kwarg passes through\n");
}

void case_media() {
  Fixture f;
  for (const char* type : {"image", "image_url", "input_image", "video", "video_url", "input_video", "audio", "input_audio"}) {
    json b = base_body();
    b["messages"] = json::array({{{"role", "user"},
                                  {"content", json::array({{{"type", "text"}, {"text", "look"}}, {{"type", type}, {"x", 1}}})}}});
    const std::string msg = error_of(f, b);
    CHECK(msg.find(type) != std::string::npos && msg.find("text only") != std::string::npos);
  }
  for (const char* type : {"text", "input_text"}) {
    json b = base_body();
    b["messages"] = json::array({{{"role", "user"}, {"content", json::array({{{"type", type}, {"text", "a"}}, {{"type", type}, {"text", "b"}}})}}});
    f.script("ok");
    post(f, b);
    CHECK(f.tmpl.last_prompt.find("<|im_start|>user\nab<|im_end|>") != std::string::npos);
  }
  std::printf("media: image / video / audio parts refused by name (text only); text and input_text parts concatenated by the "
              "template\n");
}

// ---- Decision 12 --------------------------------------------------------------------------------------------------------------
void check_published(const cli::bolt::Generation& g) {
  CHECK(!g.sampling.greedy);
  CHECK_EQ(g.sampling.temperature, 1.0f);
  CHECK_EQ(g.sampling.top_p, 0.95f);
  CHECK_EQ(g.sampling.top_k, 0u);
  CHECK_EQ(g.sampling.repetition_penalty, 1.0f);
  CHECK((g.eos == std::vector<uint32_t>{2, 11}));
}

void case_generation() {
  const json nvidia = json::parse(slurp(g_model_dir + "/generation_config.nvidia.json"));
  const json ours = json::parse(slurp(g_model_dir + "/generation_config.json"));
  // NVIDIA's file carries sampling keys: used as it is (top_k absent -> none, not HF's 50).
  {
    const cli::bolt::Generation g = cli::bolt::resolve_generation(&nvidia, nullptr, "", "nv.json");
    check_published(g);
    CHECK(g.source.find("nv.json (the snapshot's file carries sampling keys)") != std::string::npos);
  }
  // The served export's: NVIDIA's keys and values - the same set through its own keys.
  {
    const cli::bolt::Generation g = cli::bolt::resolve_generation(&ours, nullptr, "", "ours.json");
    check_published(g);
    CHECK(g.source.find("ours.json (the snapshot's file carries sampling keys)") != std::string::npos);
    CHECK(g_generation.eos == g.eos);
  }
  // A file with EOS 11 only and no sampling key (an export that writes no sampling defaults): the published set (Decision
  // 12), its eos added to it.
  {
    const json bare = {{"_from_model_config", true}, {"bos_token_id", 1}, {"eos_token_id", 11}, {"pad_token_id", 0},
                       {"use_cache", true}};
    const cli::bolt::Generation g = cli::bolt::resolve_generation(&bare, nullptr, "", "bare.json");
    check_published(g);
    CHECK(g.source.find("published defaults") != std::string::npos);
    CHECK(g.source.find("names no sampling key (its eos_token_id is added)") != std::string::npos);
  }
  // No file at all: the published values.
  {
    const cli::bolt::Generation g = cli::bolt::resolve_generation(nullptr, nullptr);
    check_published(g);
    CHECK_EQ(g.source, cli::bolt::published_generation().source);
  }
  // An export listing another end id keeps it beside the published two.
  {
    const json odd = {{"eos_token_id", 99}};
    const cli::bolt::Generation g = cli::bolt::resolve_generation(&odd, nullptr);
    CHECK((g.eos == std::vector<uint32_t>{2, 11, 99}));
    CHECK(!g.sampling.greedy);
  }
  // An explicit file is the operator's word: its sampling, its EOS (absent -> the published two), its stop strings.
  {
    const json file = {{"do_sample", true}, {"temperature", 0.6}, {"top_p", 0.9}, {"top_k", 20}, {"eos_token_id", 11},
                       {"stop_strings", json::array({"</tool_response>"})}};
    const cli::bolt::Generation g = cli::bolt::resolve_generation(&nvidia, &file, "my.json");
    CHECK(!g.sampling.greedy);
    CHECK_NEAR(g.sampling.temperature, 0.6, 1e-6);
    CHECK_NEAR(g.sampling.top_p, 0.9, 1e-6);
    CHECK_EQ(g.sampling.top_k, 20u);
    CHECK((g.eos == std::vector<uint32_t>{11}));
    CHECK((g.stops == std::vector<std::string>{"</tool_response>"}));
    CHECK_EQ(g.source, std::string("--generation-config my.json"));
    const json greedy = {{"do_sample", false}};
    const cli::bolt::Generation g2 = cli::bolt::resolve_generation(&nvidia, &greedy, "g.json");
    CHECK(g2.sampling.greedy);
    CHECK((g2.eos == std::vector<uint32_t>{2, 11}));
  }
  // A partial explicit file overlays the published values (review I2): temperature alone keeps sampling on, top-p 0.95, EOS [2, 11].
  {
    const json partial = {{"temperature", 0.6}};
    const cli::bolt::Generation g = cli::bolt::resolve_generation(&nvidia, &partial, "p.json");
    CHECK(!g.sampling.greedy);
    CHECK_NEAR(g.sampling.temperature, 0.6, 1e-6);
    CHECK_NEAR(g.sampling.top_p, 0.95, 1e-6);
    CHECK_EQ(g.sampling.top_k, 0u);
    CHECK((g.eos == std::vector<uint32_t>{2, 11}));
    const json top_k_only = {{"top_k", 40}};
    const cli::bolt::Generation g2 = cli::bolt::resolve_generation(nullptr, &top_k_only, "k.json");
    CHECK(!g2.sampling.greedy);
    CHECK_EQ(g2.sampling.top_k, 40u);
    CHECK_NEAR(g2.sampling.temperature, 1.0, 1e-6);
    CHECK_NEAR(g2.sampling.top_p, 0.95, 1e-6);
  }
  // A malformed field is a named error.
  for (const json& bad : {json{{"temperature", "hot"}}, json{{"top_p", 2.0}}, json{{"do_sample", "yes"}},
                          json{{"eos_token_id", "eleven"}}, json{{"eos_token_id", json::array()}}}) {
    bool threw = false;
    try {
      (void)cli::bolt::resolve_generation(&bad, nullptr);
    } catch (const std::invalid_argument& e) {
      threw = std::string(e.what()).find("generation_config.json") != std::string::npos;
    }
    CHECK(threw);
  }
  std::printf("generation: NVIDIA's file and the export's (T 1.0, top-p 0.95, EOS [2, 11], top-k none), a bare one (EOS 11, no "
              "sampling key) + the published defaults = the same set, an explicit file its own, malformed values named\n");
}

void case_sampling_and_eos() {
  Fixture f;
  // The defaults reach every step; a request's own fields override them; temperature 0 is greedy; a seed is carried.
  f.script("ok");
  post(f, base_body());
  CHECK(!f.engine.sampling.empty());
  for (const server::Sampling& s : f.engine.sampling) {
    CHECK(!s.greedy);
    CHECK_EQ(s.temperature, 1.0f);
    CHECK_EQ(s.top_p, 0.95f);
    CHECK_EQ(s.top_k, 0u);
    CHECK(!s.has_seed);
  }
  f.engine.sampling.clear();
  f.script("ok");
  post(f, base_body({{"temperature", 0.3}, {"top_p", 0.5}, {"top_k", 7}, {"seed", 1234}}));
  CHECK_NEAR(f.engine.sampling.front().temperature, 0.3, 1e-6);
  CHECK_NEAR(f.engine.sampling.front().top_p, 0.5, 1e-6);
  CHECK_EQ(f.engine.sampling.front().top_k, 7u);
  CHECK(f.engine.sampling.front().has_seed && f.engine.sampling.front().seed == 1234);
  f.engine.sampling.clear();
  f.script("ok");
  post(f, base_body({{"temperature", 0}}));
  CHECK(f.engine.sampling.front().greedy);
  // The same seeded request twice hands the engine the same sampling both times.
  f.engine.sampling.clear();
  for (int i = 0; i < 2; ++i) {
    f.script("hello there");
    post(f, base_body({{"seed", 77}}));
  }
  const size_t half = f.engine.sampling.size() / 2;
  CHECK(half > 0 && f.engine.sampling.size() == 2 * half);
  for (size_t i = 0; i < half; ++i) {
    CHECK(f.engine.sampling[i].has_seed && f.engine.sampling[i].seed == 77 && f.engine.sampling[half + i].seed == 77);
    CHECK_EQ(f.engine.sampling[i].temperature, f.engine.sampling[half + i].temperature);
  }
  // EOS on either id ends the generation: stop, and the text before it is the whole content.
  for (uint32_t eos : {2u, 11u}) {
    f.script("</think>done", 3, eos);
    const json r = post(f, base_body());
    CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("stop"));
    CHECK_EQ(r.at("choices").at(0).at("message").at("content").get<std::string>(), std::string("done"));
  }
  std::printf("sampling: the published defaults reach every step, a request's fields override them, temperature 0 is greedy, a "
              "seed is carried; EOS 2 and EOS 11 both stop a generation\n");
}

// ---- the output stream through the server ----------------------------------------------------------------------------------
const std::string kCall = "<tool_call>\n<function=read>\n<parameter=filePath>\n/a.cc\n</parameter>\n</function>\n</tool_call>";

void case_output() {
  Fixture f;
  const std::string text = "Need the file first,\nthen read it.\n</think>Reading it.\n" + kCall;
  f.script(text);
  const json r = post(f, base_body());
  const json& choice = r.at("choices").at(0);
  const json& msg = choice.at("message");
  CHECK_EQ(msg.at("reasoning_content").get<std::string>(), std::string("Need the file first,\nthen read it.\n"));
  CHECK_EQ(msg.at("content").get<std::string>(), std::string("Reading it.\n"));
  CHECK_EQ(msg.at("tool_calls").size(), size_t(1));
  CHECK_EQ(msg.at("tool_calls").at(0).at("function").at("name").get<std::string>(), std::string("read"));
  CHECK_EQ(json::parse(msg.at("tool_calls").at(0).at("function").at("arguments").get<std::string>()),
           json::parse(R"({"filePath": "/a.cc"})"));
  CHECK_EQ(choice.at("finish_reason").get<std::string>(), std::string("tool_calls"));
  CHECK(f.tmpl.last_prompt.size() > 10 && f.tmpl.last_prompt.rfind("<think>\n") == f.tmpl.last_prompt.size() - 8);
  // The same turn streamed: the frames concatenate to the same message, the reasoning verbatim.
  f.script(text);
  const Streamed s = post_stream(f, base_body({}, true));
  CHECK_EQ(s.reasoning, msg.at("reasoning_content").get<std::string>());
  CHECK_EQ(s.content, msg.at("content").get<std::string>());
  CHECK_EQ(s.calls.size(), size_t(1));
  CHECK_EQ(s.finish, std::string("tool_calls"));
  CHECK(s.frames > 8);
  // Thinking off: the prompt ends <think></think>, everything is content, a stray </think> included.
  f.script("Just an answer</think> and more");
  const json off = post(f, base_body({{"chat_template_kwargs", {{"enable_thinking", false}}}}));
  CHECK(f.tmpl.last_prompt.size() > 20 && f.tmpl.last_prompt.rfind("<think></think>") == f.tmpl.last_prompt.size() - 15);
  CHECK_EQ(off.at("choices").at(0).at("message").at("content").get<std::string>(), std::string("Just an answer</think> and more"));
  CHECK(!off.at("choices").at(0).at("message").contains("reasoning_content") ||
        off.at("choices").at(0).at("message").at("reasoning_content").is_null());
  // ... and through the reasoning_effort mapping.
  f.script("Plain");
  const json off2 = post(f, base_body({{"reasoning_effort", "none"}}));
  CHECK_EQ(off2.at("choices").at(0).at("message").at("content").get<std::string>(), std::string("Plain"));
  // force_nonempty_content: generation cut inside reasoning is content; without it, reasoning only.
  {
    json b = base_body({{"max_tokens", 100}});
    f.script("Still thinking when the budget ended", 3, 0);
    b["max_tokens"] = int(f.engine.script.size());
    const json plain = post(f, b);
    CHECK_EQ(plain.at("choices").at(0).at("message").at("reasoning_content").get<std::string>(), std::string("Still thinking when the budget ended"));
    CHECK(plain.at("choices").at(0).at("message").at("content").is_null() ||
          plain.at("choices").at(0).at("message").at("content").get<std::string>().empty());
    CHECK_EQ(plain.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("length"));
    f.script("Still thinking when the budget ended", 3, 0);
    b["chat_template_kwargs"] = {{"force_nonempty_content", true}};
    const json forced = post(f, b);
    CHECK_EQ(forced.at("choices").at(0).at("message").at("content").get<std::string>(), std::string("Still thinking when the budget ended"));
    CHECK(!forced.at("choices").at(0).at("message").contains("reasoning_content") ||
          forced.at("choices").at(0).at("message").at("reasoning_content").is_null());
  }
  // A call cut by max_tokens is dropped and stripped, finish_reason length (body and stream).
  {
    const std::string cut = "r</think>Let me edit.\n<tool_call>\n<function=read>\n<parameter=filePath>\n/a.c";
    f.script(cut, 3, 0);
    json b = base_body();
    b["max_tokens"] = int(f.engine.script.size());
    const json r2 = post(f, b);
    const json& m2 = r2.at("choices").at(0).at("message");
    CHECK(!m2.contains("tool_calls"));
    CHECK_EQ(m2.at("content").get<std::string>(), std::string("Let me edit.\n"));
    CHECK_EQ(r2.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("length"));
    f.script(cut, 3, 0);
    json sb = base_body({}, true);
    sb["max_tokens"] = int(f.engine.script.size());
    const Streamed cs = post_stream(f, sb);
    CHECK(cs.calls.empty());
    CHECK_EQ(cs.content, std::string("Let me edit.\n"));
    CHECK_EQ(cs.finish, std::string("length"));
  }
  // A stop string applies to content only: one in the content before a call ends the generation (the parser is never
  // finished), the call after it never comes; a call is never cut by a stop string (its text is the parser's, not content).
  {
    f.script("r</think>Doing it. STOP<tool_call>\n<function=read>\n<parameter=filePath>\n/a.cc\n</parameter>\n</function>\n</tool_call>");
    const json r3 = post(f, base_body({{"stop", "STOP"}}));
    CHECK(!r3.at("choices").at(0).at("message").contains("tool_calls"));
    CHECK_EQ(r3.at("choices").at(0).at("message").at("content").get<std::string>(), std::string("Doing it. "));
    CHECK_EQ(r3.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("stop"));
    f.script("r</think>" + kCall);
    const json r4 = post(f, base_body({{"stop", "</paramet"}}));
    CHECK_EQ(r4.at("choices").at(0).at("message").at("tool_calls").size(), size_t(1));
    CHECK_EQ(r4.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("tool_calls"));
  }
  std::printf("output: reasoning verbatim + content + a Qwen XML call (body and stream), thinking off (stray </think> content), "
              "force_nonempty_content, a cut call dropped and stripped (length), a stop string in content ends the generation "
              "before any call\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: bolt_request_test <tests/tokenizer> <tests/model/bolt> <snapshot dir>\n");
    return 2;
  }
  g_fixture_dir = argv[1];
  g_model_dir = argv[2];
  const char* env_snap = std::getenv("B70_BOLT_SNAPSHOT");
  g_snapshot = env_snap != nullptr && *env_snap != '\0' ? env_snap : argv[3];
  if (!exists(g_snapshot + "/chat_template.jinja") || !exists(g_snapshot + "/tokenizer_config.json")) {
    std::printf("bolt_request_test: SKIP (no %s/chat_template.jinja; set B70_BOLT_SNAPSHOT)\n", g_snapshot.c_str());
    return 77;
  }
  const json ours = json::parse(slurp(g_model_dir + "/generation_config.json"));
  g_generation = cli::bolt::resolve_generation(&ours, nullptr);
  check_published(g_generation);
  const json facts = json::parse(slurp(g_fixture_dir + "/bolt_templates.json"));
  case_fixture_renders(facts);
  case_reasoning_effort(facts);
  case_kwarg_errors();
  case_media();
  case_generation();
  case_sampling_and_eos();
  case_output();
  std::printf("bolt_request_test OK\n");
  return 0;
}

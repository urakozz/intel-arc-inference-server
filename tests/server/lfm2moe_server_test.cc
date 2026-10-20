// Spec 24e (plan 24e Task 4): the server's LFM2.5-24B-A2B dispatch over a mock engine (no device, no checkpoint) - the
// chat-format half of the wiring b70-serve uses for it (the engine half, Lfm2MoeEngine behind
// cli::lfm2moe::Lfm2MoeEngineAdapterT, is lfm2moe_snapshot_test's).
//   lfm2moe_server_test <tests/model/lfm2moe>
// With Options::chat_format = ChatFormat::for_model_type("lfm2_moe"):
//   - a turn that reasons, speaks and calls a tool: the body carries reasoning_content, content, tool_calls (arguments a JSON
//     string of the object) and finish_reason tool_calls; the streamed frames (a few bytes a token) concatenate to the same
//     message, the call as one frame with index 0, and the usage;
//   - Review Focus 3 end to end: max_tokens inside a block - after `fn(a=1)` the call is emitted once and finish_reason is
//     length; inside `g(b="unterminat` the unfinished call is not emitted and its text is not content; EOS inside a block likewise
//     (a closed call: tool_calls; an unfinished one: stop and nothing); generation that ends inside reasoning is reasoning only;
//   - Review Focus 4: reasoning and content verbatim (newlines kept), keyed on the prompt's <think>;
//   - decision 8: enable_thinking false is accepted, the prompt is the same, and one stderr line says it was ignored;
//     chat_template_kwargs.preserve_thinking reaches the template and a non-boolean is a 400 naming it;
//   - decision 1 / 2: the sampling a request carries - the vendored generation_config.json's defaults (sampled, T 0.1, top-k 50,
//     top-p 1.0, penalty 1.05) unless the request names a field, temperature 0 greedy - and the SeenIds a penalised request
//     hands the engine: the prompt's distinct ids, then every consumed id; none when the penalty is 1;
//   - --log-requests: the record carries parser.rules (the lenient readings that fired).
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/openai.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;

// LFM2.5's template in the shape the server sees: BOS, ChatML, the generation prompt "<|im_start|>assistant\n<think>".
struct LfmTemplate : server::TemplateIface {
  int plain_calls = 0, kwargs_calls = 0;
  json last_kwargs, last_messages;
  std::string last_prompt;
  std::string render(const json& messages, const json&, bool) override {
    ++plain_calls;
    return last_prompt = body(messages);
  }
  std::string render_with_kwargs(const json& messages, const json&, bool, const json& kwargs) override {
    ++kwargs_calls;
    last_kwargs = kwargs;
    last_messages = messages;
    return last_prompt = body(messages);
  }
  static std::string body(const json& messages) {
    std::string text = "<|startoftext|>";
    for (const auto& m : messages) {
      const json& c = m.at("content");
      std::string content;   // the real template's parse_content: a string, or the text parts concatenated
      if (c.is_string()) content = c.get<std::string>();
      else
        for (const json& part : c) content += part.at("text").get<std::string>();
      text += "<|im_start|>" + m.at("role").get<std::string>() + "\n" + content + "<|im_end|>\n";
    }
    return text + "<|im_start|>assistant\n<think>";
  }
};

// MockEngine, recording the sampling of every step and the ids the SeenIds held at that step.
struct RecordingEngine : MockEngine {
  std::vector<server::Sampling> sampling;
  std::vector<std::vector<uint32_t>> seen_at_step;
  uint32_t step(const server::Sampling& s) override {
    sampling.push_back(s);
    seen_at_step.push_back(s.seen ? s.seen->distinct() : std::vector<uint32_t>{});
    return MockEngine::step(s);
  }
};

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

server::Sampling g_defaults;   // from the vendored generation_config.json, read in main()

struct Fixture {
  explicit Fixture(const std::string& log_dir = "") : options(make_options(log_dir)), server({tok, tmpl, engine}, options) {
    CHECK(server.start());
  }
  ~Fixture() { server.stop(); }
  static server::Options make_options(const std::string& log_dir) {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "lfm";
    o.eos_ids = {124900, 248046};   // generation_config.json's [124900]; 248046 is the mock tokenizer's <|im_end|>
    o.chat_format = server::ChatFormat::for_model_type("lfm2_moe");
    o.sampling_defaults = g_defaults;
    o.log_requests_dir = log_dir;
    return o;
  }
  httplib::Client client() { return httplib::Client("127.0.0.1", server.bound_port()); }
  // The engine generates `text` a few UTF-8 characters per token (a real streamer never emits half a character, and a
  // streamed frame must be valid JSON), then `eos` (0: none, the script ends and the mock engine answers EOS).
  void script(const std::string& text, size_t piece_chars = 3, uint32_t eos = 248046) {
    engine.script.clear();
    engine.at = 0;
    std::vector<std::string> pieces;
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
  LfmTemplate tmpl;
  RecordingEngine engine;
  server::Options options;
  server::Server server;
};

json body(const json& extra = json::object(), bool stream = false) {
  json b = {{"model", "lfm"},
            {"messages", json::array({{{"role", "user"}, {"content", "Read the file a.cc please"}}})}};
  if (stream) {
    b["stream"] = true;
    b["stream_options"] = {{"include_usage", true}};
  }
  for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
  return b;
}

json post(Fixture& f, const json& b) {
  httplib::Client c = f.client();
  const auto r = c.Post("/v1/chat/completions", b.dump(), "application/json");
  CHECK(r);
  if (r->status != 200) std::fprintf(stderr, "status %d: %s\n", r->status, r->body.c_str());
  CHECK_EQ(r->status, 200);
  return json::parse(r->body);
}

// A streamed response as one message: reasoning, content and the calls by index, the finish reason, the usage.
struct Streamed {
  std::string reasoning, content, finish;
  std::vector<json> calls;
  json usage;
  size_t frames = 0;
};

Streamed post_stream(Fixture& f, const json& b) {
  httplib::Client c = f.client();
  const auto r = c.Post("/v1/chat/completions", b.dump(), "application/json");
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

const std::string kStart = "<|tool_call_start|>", kEnd = "<|tool_call_end|>";

void case_turn() {
  Fixture f;
  const std::string text = "Need the file first,\nthen read it.</think>Reading it.\n" + kStart +
                           "[read_file(path='/a.cc', limit=50, opts={\"deep\": true})]" + kEnd;
  f.script(text);
  const json r = post(f, body());
  const json& choice = r.at("choices").at(0);
  const json& msg = choice.at("message");
  CHECK_EQ(msg.at("reasoning_content").get<std::string>(), std::string("Need the file first,\nthen read it."));
  CHECK_EQ(msg.at("content").get<std::string>(), std::string("Reading it.\n"));
  CHECK_EQ(msg.at("tool_calls").size(), size_t(1));
  const json& call = msg.at("tool_calls").at(0);
  CHECK_EQ(call.at("type").get<std::string>(), std::string("function"));
  CHECK_EQ(call.at("function").at("name").get<std::string>(), std::string("read_file"));
  CHECK_EQ(json::parse(call.at("function").at("arguments").get<std::string>()),
           json::parse(R"({"path": "/a.cc", "limit": 50, "opts": {"deep": true}})"));
  CHECK_EQ(choice.at("finish_reason").get<std::string>(), std::string("tool_calls"));
  CHECK(f.tmpl.last_prompt.size() > 10 && f.tmpl.last_prompt.rfind("<think>") == f.tmpl.last_prompt.size() - 7);

  // The same turn streamed, three characters a token: the frames concatenate to the same message.
  f.script(text);
  const Streamed s = post_stream(f, body({}, true));
  CHECK_EQ(s.reasoning, msg.at("reasoning_content").get<std::string>());
  CHECK_EQ(s.content, msg.at("content").get<std::string>());
  CHECK_EQ(s.calls.size(), size_t(1));
  CHECK_EQ(s.calls[0].at("index").get<int>(), 0);
  CHECK_EQ(s.calls[0].at("function").at("name").get<std::string>(), std::string("read_file"));
  CHECK_EQ(json::parse(s.calls[0].at("function").at("arguments").get<std::string>()),
           json::parse(call.at("function").at("arguments").get<std::string>()));
  CHECK_EQ(s.finish, std::string("tool_calls"));
  CHECK(!s.usage.is_null() && s.usage.at("completion_tokens").get<int>() == int(f.engine.script.size()) - 1);
  CHECK(s.frames > 8);
  std::printf("turn: reasoning + content + one call, finish tool_calls; streamed in %zu frames to the same message\n", s.frames);
}

void case_cut_blocks() {
  Fixture f;
  // max_tokens right after the first call's ')' - before ']' and the end tag: emitted once, finish length.
  {
    const std::string text = "r</think>" + kStart + "[read_file(path='/a.cc')";
    f.script(text, 3, 0);
    json b = body();
    b["max_tokens"] = int(f.engine.script.size());   // exactly the script: the generation is cut, not stopped
    const json r = post(f, b);
    const json& msg = r.at("choices").at(0).at("message");
    CHECK_EQ(msg.at("tool_calls").size(), size_t(1));
    CHECK_EQ(msg.at("tool_calls").at(0).at("function").at("name").get<std::string>(), std::string("read_file"));
    CHECK(msg.at("content").is_null() || msg.at("content").get<std::string>().empty());
    CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("length"));
    f.script(text, 3, 0);
    const Streamed s = post_stream(f, [&] { json x = body({}, true); x["max_tokens"] = int(f.engine.script.size()); return x; }());
    CHECK_EQ(s.calls.size(), size_t(1));   // once: the end of generation does not emit it again
    CHECK_EQ(s.finish, std::string("length"));
  }
  // max_tokens inside the second call: the first is emitted, the unfinished one is not, and its text is not content.
  {
    const std::string text = "r</think>Doing it. " + kStart + "[read_file(path='/a.cc'), grep(pattern=\"unterminat";
    f.script(text, 3, 0);
    json b = body();
    b["max_tokens"] = int(f.engine.script.size());
    const json r = post(f, b);
    const json& msg = r.at("choices").at(0).at("message");
    CHECK_EQ(msg.at("tool_calls").size(), size_t(1));
    CHECK_EQ(msg.at("content").get<std::string>(), std::string("Doing it. "));
    CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("length"));
  }
  // EOS inside a block: a closed call is emitted (tool_calls); an unfinished one is not (stop, no content).
  {
    f.script("r</think>" + kStart + "[read_file(path='/a.cc')", 3, 248046);
    const json r = post(f, body());
    CHECK_EQ(r.at("choices").at(0).at("message").at("tool_calls").size(), size_t(1));
    CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("tool_calls"));
    f.script("r</think>" + kStart + "[read_file(path='/a.c", 3, 248046);
    const json r2 = post(f, body());
    CHECK(!r2.at("choices").at(0).at("message").contains("tool_calls"));
    CHECK(r2.at("choices").at(0).at("message").at("content").is_null() ||
          r2.at("choices").at(0).at("message").at("content").get<std::string>().empty());
    CHECK_EQ(r2.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("stop"));
  }
  // Generation that ends inside reasoning is reasoning (a `<|tool_call_start|>` in it is reasoning text).
  {
    const std::string text = "still thinking " + kStart + "[x()]";
    f.script(text, 3, 0);
    json b = body();
    b["max_tokens"] = int(f.engine.script.size());
    const json r = post(f, b);
    const json& msg = r.at("choices").at(0).at("message");
    CHECK_EQ(msg.at("reasoning_content").get<std::string>(), text);
    CHECK(!msg.contains("tool_calls"));
    CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("length"));
  }
  // A block with no valid call is content, verbatim with its tags.
  {
    const std::string blk = kStart + "[f(1)]" + kEnd;
    f.script("r</think>before " + blk + " after");
    const json r = post(f, body());
    CHECK_EQ(r.at("choices").at(0).at("message").at("content").get<std::string>(), "before " + blk + " after");
    CHECK(!r.at("choices").at(0).at("message").contains("tool_calls"));
  }
  std::printf("cut blocks: a closed call once (body and stream), an unfinished one never and not as content, EOS inside a "
              "block likewise, reasoning-only on length, an invalid block as content\n");
}

// Run `f` with stderr captured into a string.
template <class F>
std::string capture_stderr(F&& f) {
  std::fflush(stderr);
  FILE* tmp = std::tmpfile();
  CHECK(tmp != nullptr);
  const int saved = dup(2);
  dup2(fileno(tmp), 2);
  f();
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::rewind(tmp);
  std::string out;
  char buf[512];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, tmp)) > 0) out.append(buf, n);
  std::fclose(tmp);
  return out;
}

void case_kwargs() {
  Fixture f;
  f.script("r</think>ok");
  // enable_thinking false: accepted, the same prompt, one line on stderr.
  const json plain = post(f, body());
  const std::string prompt = f.tmpl.last_prompt;
  f.script("r</think>ok");
  json off = body();
  off["chat_template_kwargs"] = {{"enable_thinking", false}};
  json r;
  const std::string err = capture_stderr([&] { r = post(f, off); });
  CHECK_EQ(f.tmpl.last_prompt, prompt);
  CHECK(r.at("choices").at(0).at("message").contains("reasoning_content"));   // the reasoning is still returned
  size_t lines = 0;
  for (size_t at = err.find("enable_thinking ignored: the template always reasons"); at != std::string::npos;
       at = err.find("enable_thinking ignored: the template always reasons", at + 1))
    ++lines;
  CHECK_EQ(lines, size_t(1));
  // and a request that does not say so prints nothing of it
  f.script("r</think>ok");
  CHECK(capture_stderr([&] { (void)post(f, body()); }).find("enable_thinking ignored") == std::string::npos);

  // preserve_thinking reaches the template; a non-boolean is a 400 naming it, and nothing is rendered.
  f.script("r</think>ok");
  json keep = body();
  keep["chat_template_kwargs"] = {{"preserve_thinking", true}};
  (void)post(f, keep);
  CHECK_EQ(f.tmpl.last_kwargs.at("preserve_thinking"), json(true));
  const int renders = f.tmpl.kwargs_calls;
  json bad = body();
  bad["chat_template_kwargs"] = {{"preserve_thinking", "yes"}};
  httplib::Client c = f.client();
  const auto res = c.Post("/v1/chat/completions", bad.dump(), "application/json");
  CHECK(res);
  CHECK_EQ(res->status, 400);
  CHECK(res->body.find("preserve_thinking") != std::string::npos);
  CHECK_EQ(f.tmpl.kwargs_calls, renders);
  // the OpenAI top-level reasoning_effort is not this model's: ignored
  f.script("r</think>ok");
  json effort = body();
  effort["reasoning_effort"] = "high";
  (void)post(f, effort);
  CHECK(!f.tmpl.last_kwargs.contains("reasoning_strength"));
  // a list of text parts reaches the template as it is (its parse_content concatenates them)
  f.script("r</think>ok");
  json parts = body();
  parts["messages"][0]["content"] = json::array({{{"type", "text"}, {"text", "a"}}, {{"type", "text"}, {"text", "b"}}});
  (void)post(f, parts);
  CHECK(f.tmpl.last_messages.at(0).at("content").is_array());   // not joined by the server
  CHECK(f.tmpl.last_prompt.find("Read the file a.cc please") == std::string::npos);
  CHECK(f.tmpl.last_prompt.find("<|im_start|>user\nab<|im_end|>") != std::string::npos);
  std::printf("kwargs: enable_thinking ignored with one stderr line, preserve_thinking validated and passed, lists passed through\n");
}

void case_sampling() {
  Fixture f;
  // The model's defaults: the vendored generation_config.json.
  f.script("r</think>one two three", 3);
  (void)post(f, body());
  CHECK(!f.engine.sampling.empty());
  server::Sampling s = f.engine.sampling.front();
  CHECK(!s.greedy);
  CHECK_NEAR(s.temperature, 0.1, 1e-7);
  CHECK_EQ(s.top_k, 50u);
  CHECK_NEAR(s.top_p, 1.0, 0.0);
  CHECK_NEAR(s.repetition_penalty, 1.05, 1e-7);
  CHECK(s.seen != nullptr);
  // A request's own fields win; temperature 0 is greedy.
  f.engine.sampling.clear();
  f.script("r</think>ok", 3);
  (void)post(f, body({{"temperature", 0.7}, {"top_k", 7}, {"top_p", 0.9}, {"repetition_penalty", 1.2}}));
  s = f.engine.sampling.front();
  CHECK_NEAR(s.temperature, 0.7, 1e-6);
  CHECK_EQ(s.top_k, 7u);
  CHECK_NEAR(s.top_p, 0.9, 1e-6);
  CHECK_NEAR(s.repetition_penalty, 1.2, 1e-6);
  f.engine.sampling.clear();
  f.script("r</think>ok", 3);
  (void)post(f, body({{"temperature", 0}}));
  CHECK(f.engine.sampling.front().greedy);
  CHECK_NEAR(f.engine.sampling.front().repetition_penalty, 1.05, 1e-7);   // handed over; the adapter ignores it when greedy
  // A bad penalty is a 400 naming it.
  httplib::Client c = f.client();
  const auto bad = c.Post("/v1/chat/completions", body({{"repetition_penalty", 0}}).dump(), "application/json");
  CHECK(bad && bad->status == 400 && bad->body.find("repetition_penalty") != std::string::npos);

  // The SeenIds: the prompt's distinct ids at the first step, then every consumed id (the engine sees step k's set before id k).
  f.engine.sampling.clear();
  f.engine.seen_at_step.clear();
  f.script("r</think>ab", 1);   // one character a token: 'r', '<', '/', ... ids 200000+
  json b = body();
  b["max_tokens"] = 6;
  (void)post(f, b);
  const std::vector<uint32_t> prompt_ids = f.engine.last_prompt;
  CHECK(!prompt_ids.empty());
  std::vector<uint32_t> distinct;
  for (uint32_t id : prompt_ids)
    if (std::find(distinct.begin(), distinct.end(), id) == distinct.end()) distinct.push_back(id);
  CHECK(f.engine.seen_at_step.size() >= 4);
  CHECK(f.engine.seen_at_step[0] == distinct);   // the first step: the prompt's ids only
  for (size_t k = 1; k < f.engine.seen_at_step.size(); ++k) {
    // step k sees the prompt's ids plus the ids steps 0 .. k - 1 returned (the scripted ids are all new)
    CHECK_EQ(f.engine.seen_at_step[k].size(), distinct.size() + k);
    CHECK(std::equal(distinct.begin(), distinct.end(), f.engine.seen_at_step[k].begin()));
    for (size_t j = 0; j < k; ++j) CHECK_EQ(f.engine.seen_at_step[k][distinct.size() + j], f.engine.script[j]);
  }
  // Penalty 1.0: no set is handed over at all.
  f.engine.sampling.clear();
  f.script("r</think>ok", 3);
  (void)post(f, body({{"repetition_penalty", 1.0}}));
  for (const server::Sampling& x : f.engine.sampling) CHECK(x.seen == nullptr);
  std::printf("sampling: the model's defaults, request fields win, the SeenIds grow with the consumed ids, none at penalty 1\n");
}

void case_log() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("lfm2moe_server_test_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  {
    Fixture f(dir.string());
    f.script("r</think>" + kStart + "['read_file'(\"path\"='it's here', flag=true)]" + kEnd, 4);
    const json r = post(f, body());
    CHECK_EQ(r.at("choices").at(0).at("message").at("tool_calls").size(), size_t(1));
    CHECK_EQ(json::parse(r.at("choices").at(0).at("message").at("tool_calls").at(0).at("function").at("arguments").get<std::string>()),
             json::parse(R"({"path": "it's here", "flag": true})"));
  }
  size_t files = 0;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    const json rec = json::parse(slurp(e.path().string()));
    const json& rules = rec.at("parser").at("rules");
    CHECK_EQ(rules.at("quote_recovered").get<int>(), 1);
    CHECK_EQ(rules.at("quoted_name").get<int>(), 2);
    CHECK_EQ(rules.at("json_literal").get<int>(), 1);
    CHECK_EQ(rules.at("bare_call").get<int>(), 0);
    ++files;
  }
  CHECK_EQ(files, size_t(1));
  std::filesystem::remove_all(dir);
  std::printf("log: --log-requests carries parser.rules (quote_recovered 1, quoted_name 2, json_literal 1)\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/model/lfm2moe";
  g_defaults = server::generation_sampling(json::parse(slurp(dir + "/generation_config.json")));
  case_turn();
  case_cut_blocks();
  case_kwargs();
  case_sampling();
  case_log();
  std::printf("lfm2moe_server_test OK\n");
  return 0;
}

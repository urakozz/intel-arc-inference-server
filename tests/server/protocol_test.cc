#include <chrono>
#include <fstream>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/server.h"
#include "server/mock.h"

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Fixture {
  explicit Fixture(size_t queue_depth = 4, const std::string& log_dir = "")
      : options(make_options(queue_depth, log_dir)), server({tok, tmpl, engine}, options) {
    CHECK(server.start());
    CHECK(server.bound_port() > 0);
  }

  ~Fixture() { server.stop(); }

  static server::Options make_options(size_t queue_depth, const std::string& log_dir = "") {
    server::Options result;
    result.log_requests_dir = log_dir;
    result.host = "127.0.0.1";
    result.port = 0;
    result.served_model = "mock-model";
    result.queue_depth = queue_depth;
    result.eos_ids = {248046};
    return result;
  }

  httplib::Client client() { return httplib::Client("127.0.0.1", server.bound_port()); }

  MockTok tok;
  MockTemplate tmpl;
  MockEngine engine;
  server::Options options;
  server::Server server;
};

json messages() {
  return json::array({{{"role", "user"}, {"content", "What is the capital of France?"}}});
}

json chat_body() {
  return {{"model", "mock-model"}, {"messages", messages()}};
}

std::vector<uint32_t> ids(Fixture& fixture, const std::string& text) {
  return fixture.tok.encode(text);
}

std::vector<std::string> sse_frames(const std::string& body) {
  std::vector<std::string> frames;
  size_t begin = 0;
  while (begin < body.size()) {
    const size_t end = body.find("\n\n", begin);
    if (end == std::string::npos) break;
    frames.push_back(body.substr(begin, end - begin));
    begin = end + 2;
  }
  return frames;
}

json frame_json(const std::string& frame) {
  CHECK(frame.rfind("data: ", 0) == 0);
  return json::parse(frame.substr(6));
}

std::string streamed_text(const std::vector<std::string>& frames) {
  std::string text;
  for (const std::string& frame : frames) {
    if (frame == "data: [DONE]") continue;
    const json body = frame_json(frame);
    if (body.at("choices").empty()) continue;
    const json& choice = body.at("choices").at(0);
    if (choice.contains("delta") && choice.at("delta").contains("content")) {
      text += choice.at("delta").at("content").get<std::string>();
    } else if (choice.contains("text")) {
      text += choice.at("text").get<std::string>();
    }
  }
  return text;
}

struct StreamReply {
  int status = 0;
  std::string body;
  std::vector<std::string> chunks;
  std::vector<Clock::time_point> arrivals;
};

StreamReply stream(httplib::Client& client, const std::string& path, const json& body) {
  StreamReply reply;
  const auto response = client.Post(
      path, httplib::Headers(), body.dump(), "application/json",
      [&reply](const char* data, size_t size) {
        reply.chunks.emplace_back(data, size);
        reply.arrivals.push_back(Clock::now());
        reply.body.append(data, size);
        return true;
      },
      nullptr);
  CHECK(response);
  reply.status = response->status;
  return reply;
}

std::string response_text(const json& response) {
  const json& choice = response.at("choices").at(0);
  return choice.contains("message") ? choice.at("message").at("content").get<std::string>()
                                    : choice.at("text").get<std::string>();
}

void case_models() {
  Fixture fixture;
  auto client = fixture.client();
  const auto response = client.Get("/v1/models");
  CHECK(response);
  CHECK_EQ(response->status, 200);
  const json body = json::parse(response->body);
  CHECK_EQ(body.at("data").at(0).at("id"), std::string("mock-model"));
}

void case_non_stream_chat() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "Paris is the capital <|im_end|>");
  auto client = fixture.client();
  const auto response = client.Post("/v1/chat/completions", chat_body().dump(), "application/json");
  CHECK(response);
  CHECK_EQ(response->status, 200);
  const json body = json::parse(response->body);
  CHECK_EQ(response_text(body), std::string("Paris is the capital "));
  CHECK_EQ(body.at("choices").at(0).at("finish_reason"), std::string("stop"));
  const std::string prompt = fixture.tmpl.render(messages(), nullptr, true);
  CHECK_EQ(body.at("usage").at("prompt_tokens"), fixture.tok.encode(prompt).size());
  CHECK_EQ(body.at("usage").at("completion_tokens"), 4);
  CHECK_EQ(fixture.engine.last_prompt, fixture.tok.encode(prompt));
}

void case_max_tokens() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "Paris is the capital <|im_end|>");
  json request = chat_body();
  request["max_tokens"] = 2;
  auto client = fixture.client();
  const auto response = client.Post("/v1/chat/completions", request.dump(), "application/json");
  CHECK(response);
  const json body = json::parse(response->body);
  CHECK_EQ(response_text(body), std::string("Paris is "));
  CHECK_EQ(body.at("choices").at(0).at("finish_reason"), std::string("length"));
}

void case_stream_usage() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "Paris is the capital <|im_end|>");
  json request = chat_body();
  request["stream"] = true;
  request["stream_options"] = {{"include_usage", true}};
  auto client = fixture.client();
  const StreamReply reply = stream(client, "/v1/chat/completions", request);
  CHECK_EQ(reply.status, 200);
  const std::vector<std::string> frames = sse_frames(reply.body);
  CHECK_EQ(frames.size(), 8U);
  CHECK_EQ(frame_json(frames[0]).at("choices").at(0).at("delta").at("role"),
           std::string("assistant"));
  const std::vector<std::string> pieces = {"Paris ", "is ", "the ", "capital "};
  for (size_t i = 0; i < pieces.size(); ++i) {
    CHECK_EQ(frame_json(frames[i + 1]).at("choices").at(0).at("delta").at("content"), pieces[i]);
  }
  const json finish = frame_json(frames[5]);
  CHECK_EQ(finish.at("choices").at(0).at("finish_reason"), std::string("stop"));
  CHECK(finish.at("choices").at(0).at("delta").empty());
  const json usage = frame_json(frames[6]);
  CHECK(usage.at("choices").empty());
  CHECK(usage.contains("usage"));
  CHECK_EQ(frames[7], std::string("data: [DONE]"));
}

void case_unbuffered_first_frame() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "one two three four five six");
  fixture.engine.step_ms = 50;
  json request = chat_body();
  request["stream"] = true;
  request["max_tokens"] = 6;
  auto client = fixture.client();
  const Clock::time_point started = Clock::now();
  const StreamReply reply = stream(client, "/v1/chat/completions", request);
  CHECK_EQ(reply.status, 200);

  std::vector<Clock::time_point> content_arrivals;
  for (size_t i = 0; i < reply.chunks.size(); ++i) {
    if (reply.chunks[i].find("\"content\"") != std::string::npos) {
      content_arrivals.push_back(reply.arrivals[i]);
    }
  }
  CHECK_EQ(content_arrivals.size(), 6U);
  const auto first_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            content_arrivals.front() - started)
                            .count();
  const auto last_gap_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               content_arrivals.back() - content_arrivals.front())
                               .count();
  CHECK(first_ms < 120);
  CHECK(last_gap_ms >= 250);
  CHECK_EQ(fixture.engine.step_times.size(), 6U);
  for (size_t i = 0; i + 1 < content_arrivals.size(); ++i) {
    CHECK(content_arrivals[i] < fixture.engine.step_times[i + 1]);
  }
  std::printf("protocol_test case 5: first_frame_latency_ms=%lld last_frame_gap_ms=%lld\n",
              static_cast<long long>(first_ms), static_cast<long long>(last_gap_ms));
}

void case_stop_strings() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "alpha beta STOP gamma <|im_end|>");
  json request = chat_body();
  request["stream"] = true;
  request["stop"] = json::array({"STOP"});
  auto client = fixture.client();
  const StreamReply reply = stream(client, "/v1/chat/completions", request);
  CHECK_EQ(reply.status, 200);
  CHECK(reply.body.find("STOP") == std::string::npos);
  const std::vector<std::string> frames = sse_frames(reply.body);
  CHECK_EQ(streamed_text(frames), std::string("alpha beta "));
  const json finish = frame_json(frames[frames.size() - 2]);
  CHECK_EQ(finish.at("choices").at(0).at("finish_reason"), std::string("stop"));

  Fixture split_fixture;
  split_fixture.engine.script = ids(split_fixture, "alpha beta STOP gamma <|im_end|>");
  json split_request = chat_body();
  split_request["stop"] = json::array({"beta STOP"});
  auto split_client = split_fixture.client();
  const auto response = split_client.Post("/v1/chat/completions", split_request.dump(), "application/json");
  CHECK(response);
  const json split_body = json::parse(response->body);
  CHECK_EQ(response_text(split_body), std::string("alpha "));
  CHECK_EQ(split_body.at("choices").at(0).at("finish_reason"), std::string("stop"));
}

void case_min_tokens_and_ignore_eos() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "x <|im_end|> y z <|im_end|>");
  json request = chat_body();
  request["min_tokens"] = 3;
  auto client = fixture.client();
  const auto response = client.Post("/v1/chat/completions", request.dump(), "application/json");
  CHECK(response);
  const json body = json::parse(response->body);
  CHECK_EQ(response_text(body), std::string("x y z "));
  CHECK_EQ(body.at("choices").at(0).at("finish_reason"), std::string("stop"));
  CHECK_EQ(body.at("usage").at("completion_tokens"), 4);

  Fixture ignored_fixture;
  ignored_fixture.engine.script = ids(ignored_fixture, "x <|im_end|> y z <|im_end|>");
  json ignored = chat_body();
  ignored["ignore_eos"] = true;
  ignored["max_tokens"] = 4;
  auto ignored_client = ignored_fixture.client();
  const auto ignored_response =
      ignored_client.Post("/v1/chat/completions", ignored.dump(), "application/json");
  CHECK(ignored_response);
  const json ignored_body = json::parse(ignored_response->body);
  CHECK_EQ(response_text(ignored_body), std::string("x y z "));
  CHECK_EQ(ignored_body.at("usage").at("completion_tokens"), 4);
  CHECK_EQ(ignored_body.at("choices").at(0).at("finish_reason"), std::string("length"));
}

void case_completions_token_ids() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "Paris is <|im_end|>");
  const json request = {{"model", "mock-model"},
                        {"prompt", "hello world"},
                        {"return_token_ids", true}};
  auto client = fixture.client();
  const auto response = client.Post("/v1/completions", request.dump(), "application/json");
  CHECK(response);
  CHECK_EQ(response->status, 200);
  const json body = json::parse(response->body);
  const std::vector<uint32_t> prompt_ids = fixture.tok.encode("hello world");
  const std::vector<uint32_t> out_ids = ids(fixture, "Paris is");
  CHECK_EQ(fixture.engine.last_prompt, prompt_ids);
  CHECK_EQ(body.at("prompt_token_ids"), prompt_ids);
  CHECK_EQ(body.at("choices").at(0).at("token_ids"), out_ids);
  CHECK_EQ(body.at("object"), std::string("text_completion"));
}

void case_errors() {
  Fixture fixture;
  auto client = fixture.client();
  const auto malformed = client.Post("/v1/chat/completions", "{", "application/json");
  CHECK(malformed);
  CHECK_EQ(malformed->status, 400);
  CHECK_EQ(json::parse(malformed->body).at("error").at("type"),
           std::string("invalid_request_error"));

  json bad_n = chat_body();
  bad_n["n"] = 2;
  const auto n_response = client.Post("/v1/chat/completions", bad_n.dump(), "application/json");
  CHECK(n_response);
  CHECK_EQ(n_response->status, 400);

  std::string long_prompt;
  for (int i = 0; i < 1030; ++i) long_prompt += "word ";
  const json too_long = {{"model", "mock-model"}, {"prompt", long_prompt}};
  const auto length_response = client.Post("/v1/completions", too_long.dump(), "application/json");
  CHECK(length_response);
  CHECK_EQ(length_response->status, 400);
  CHECK(json::parse(length_response->body).at("error").at("message").get<std::string>()
            .find("max_model_len") != std::string::npos);

  json content_array = chat_body();
  content_array["messages"][0]["content"] = json::array({{"type", "text"}});
  const auto content_response =
      client.Post("/v1/chat/completions", content_array.dump(), "application/json");
  CHECK(content_response);
  CHECK_EQ(content_response->status, 400);
}

void case_fifo_and_overload() {
  Fixture fixture(2);
  fixture.engine.script = ids(fixture, "a b c d e f g h i j k l m n o p q r s t");
  fixture.engine.step_ms = 30;
  const std::string full_text = fixture.tok.decode(fixture.engine.script);
  json request = chat_body();
  request["max_tokens"] = 20;
  constexpr int kRequests = 6;
  std::vector<int> statuses(kRequests, 0);
  std::vector<std::string> bodies(kRequests);
  std::vector<std::thread> workers;
  workers.reserve(kRequests);
  for (int i = 0; i < kRequests; ++i) {
    workers.emplace_back([&fixture, &request, &statuses, &bodies, i] {
      auto client = fixture.client();
      const auto response = client.Post("/v1/chat/completions", request.dump(), "application/json");
      CHECK(response);
      statuses[i] = response->status;
      bodies[i] = response->body;
    });
  }
  for (auto& worker : workers) worker.join();

  int overloaded = 0;
  for (int status : statuses) {
    CHECK(status == 200 || status == 503);
    if (status == 503) ++overloaded;
  }
  for (int i = 0; i < kRequests; ++i) {
    if (statuses[i] == 200) CHECK_EQ(response_text(json::parse(bodies[i])), full_text);
  }
  CHECK(overloaded >= 1);
  CHECK(overloaded <= 4);
  CHECK_EQ(fixture.engine.max_active.load(), 1);
}

void case_unknown_fields() {
  Fixture fixture;
  fixture.engine.script = ids(fixture, "ok <|im_end|>");
  json request = chat_body();
  request["cache_prompt"] = true;
  request["return_token_ids"] = true;
  request["logprobs"] = nullptr;
  auto client = fixture.client();
  const auto response = client.Post("/v1/chat/completions", request.dump(), "application/json");
  CHECK(response);
  CHECK_EQ(response->status, 200);
}


// Spec 7 §3.5: reasoning_content and tool_calls in chat responses; completions unchanged.
const char* kToolText =
    "plan it</think>\n\nI'll read it.\n<tool_call>\n<function=read>\n<parameter=filePath>\n"
    "/w/x.cc\n</parameter>\n</function>\n</tool_call>";

void check_call(const json& call, uint32_t index, bool streamed) {
  if (streamed) CHECK_EQ(call.at("index"), index);
  CHECK(call.at("id").get<std::string>().rfind("call_", 0) == 0);
  CHECK_EQ(call.at("type"), std::string("function"));
  CHECK_EQ(call.at("function").at("name"), std::string("read"));
  CHECK_EQ(json::parse(call.at("function").at("arguments").get<std::string>()),
           json({{"filePath", "/w/x.cc"}}));
}

void case_tool_calls() {
  Fixture fixture;
  fixture.tmpl.think_tag = true;
  std::vector<uint32_t> script = fixture.tok.raw(kToolText, 3);
  script.push_back(248046);
  fixture.engine.script = script;
  auto client = fixture.client();

  // Thinking on, non-streaming.
  auto response = client.Post("/v1/chat/completions", chat_body().dump(), "application/json");
  CHECK(response);
  CHECK_EQ(response->status, 200);
  json body = json::parse(response->body);
  json choice = body.at("choices").at(0);
  CHECK_EQ(choice.at("finish_reason"), std::string("tool_calls"));
  CHECK_EQ(choice.at("message").at("role"), std::string("assistant"));
  CHECK_EQ(choice.at("message").at("reasoning_content"), std::string("plan it"));
  CHECK_EQ(choice.at("message").at("content"), std::string("I'll read it.\n"));
  CHECK_EQ(choice.at("message").at("tool_calls").size(), 1U);
  check_call(choice.at("message").at("tool_calls").at(0), 0, false);
  CHECK_EQ(body.at("usage").at("prompt_tokens_details").at("cached_tokens"), 0);

  // Thinking off: no reasoning split, the call still parsed.
  json off = chat_body();
  off["chat_template_kwargs"] = {{"enable_thinking", false}};
  response = client.Post("/v1/chat/completions", off.dump(), "application/json");
  CHECK(response);
  choice = json::parse(response->body).at("choices").at(0);
  CHECK(!choice.at("message").contains("reasoning_content"));
  CHECK_EQ(choice.at("message").at("content"), std::string("plan it</think>\n\nI'll read it.\n"));
  check_call(choice.at("message").at("tool_calls").at(0), 0, false);
  CHECK_EQ(choice.at("finish_reason"), std::string("tool_calls"));

  // Streaming, thinking on.
  json streamed = chat_body();
  streamed["stream"] = true;
  const StreamReply reply = stream(client, "/v1/chat/completions", streamed);
  CHECK_EQ(reply.status, 200);
  std::string reasoning, content, finish;
  std::vector<json> calls;
  for (const std::string& frame : sse_frames(reply.body)) {
    if (frame == "data: [DONE]") continue;
    const json c = frame_json(frame).at("choices").at(0);
    const json& delta = c.at("delta");
    if (delta.contains("reasoning_content")) reasoning += delta.at("reasoning_content").get<std::string>();
    if (delta.contains("content")) content += delta.at("content").get<std::string>();
    if (delta.contains("tool_calls")) {
      CHECK_EQ(delta.at("tool_calls").size(), 1U);
      calls.push_back(delta.at("tool_calls").at(0));
    }
    if (!c.at("finish_reason").is_null()) finish = c.at("finish_reason").get<std::string>();
  }
  CHECK_EQ(reasoning, std::string("plan it"));
  CHECK_EQ(content, std::string("I'll read it.\n"));
  CHECK_EQ(calls.size(), 1U);
  check_call(calls[0], 0, true);
  CHECK_EQ(finish, std::string("tool_calls"));

  // /v1/completions: the raw text, as before.
  const json completion = {{"model", "mock-model"}, {"prompt", "hi"}};
  response = client.Post("/v1/completions", completion.dump(), "application/json");
  CHECK(response);
  body = json::parse(response->body);
  choice = body.at("choices").at(0);
  CHECK_EQ(choice.at("text"), std::string(kToolText));
  CHECK_EQ(choice.at("finish_reason"), std::string("stop"));
  CHECK(!choice.contains("message"));
  CHECK(!body.at("usage").contains("prompt_tokens_details"));
}

// Spec 7 P0: --log-requests writes each request with its ids.
void case_request_log() {
  char dir_template[] = "/tmp/b70_reqlog_XXXXXX";
  const char* dir = ::mkdtemp(dir_template);
  CHECK(dir != nullptr);
  {
    Fixture fixture(4, dir);
    fixture.engine.script = ids(fixture, "Paris is <|im_end|>");
    auto client = fixture.client();
    json streamed = chat_body();
    streamed["stream"] = true;
    CHECK_EQ(stream(client, "/v1/chat/completions", streamed).status, 200);
    const json completion = {{"model", "mock-model"}, {"prompt", "hi there"}};
    const auto response = client.Post("/v1/completions", completion.dump(), "application/json");
    CHECK(response);
    CHECK_EQ(response->status, 200);
  }
  const std::string first_path = std::string(dir) + "/000001.json";
  const std::string second_path = std::string(dir) + "/000002.json";
  std::ifstream first_in(first_path), second_in(second_path);
  CHECK(first_in.good() && second_in.good());
  const json first = json::parse(first_in), second = json::parse(second_in);
  CHECK_EQ(first.at("endpoint"), std::string("/v1/chat/completions"));
  CHECK_EQ(first.at("request").at("messages"), messages());
  CHECK(first.at("request").at("stream").get<bool>());
  CHECK_EQ(first.at("response_text"), std::string("Paris is "));
  CHECK_EQ(first.at("out_ids").size(), 2U);
  CHECK(!first.at("prompt_ids").empty());
  const double t0 = first.at("t_start"), t1 = first.at("t_first_token"), t2 = first.at("t_end");
  CHECK(t0 <= t1 && t1 <= t2);
  CHECK_EQ(second.at("endpoint"), std::string("/v1/completions"));
  CHECK_EQ(second.at("prompt_ids").get<std::vector<uint32_t>>().size(), 2U);
  ::unlink(first_path.c_str());
  ::unlink(second_path.c_str());
  ::rmdir(dir);
}
}  // namespace

int main() {
  case_models();
  case_non_stream_chat();
  case_max_tokens();
  case_stream_usage();
  case_unbuffered_first_frame();
  case_stop_strings();
  case_min_tokens_and_ignore_eos();
  case_completions_token_ids();
  case_errors();
  case_fifo_and_overload();
  case_unknown_fields();
  case_tool_calls();
  case_request_log();
  std::printf("protocol_test OK: 13 cases\n");
  return 0;
}

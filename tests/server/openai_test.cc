#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "server/openai.h"
#include "server/toolcall.h"

namespace {

bool rejects(const std::string& body, bool chat) {
  try {
    (void)server::parse_request(body, chat);
  } catch (const server::BadRequest&) {
    return true;
  }
  return false;
}

nlohmann::json chat_request() {
  return {
      {"model", "mock-model"},
      {"messages", {{{"role", "user"}, {"content", "hello"}}}},
  };
}

}  // namespace

int main() {
  nlohmann::json body = chat_request();
  body["max_tokens"] = 5;
  body["stop"] = nlohmann::json::array({"\n"});
  body["stream"] = true;
  body["stream_options"] = {{"include_usage", true}};
  body["temperature"] = 0.7;
  body["top_k"] = 40;
  body["cache_prompt"] = true;
  body["return_token_ids"] = true;

  const server::Request r = server::parse_request(body.dump(), true);
  CHECK(r.chat);
  CHECK_EQ(r.messages, body.at("messages"));
  CHECK(r.tools.is_null());
  CHECK(r.enable_thinking);
  CHECK(r.max_tokens.has_value());
  CHECK_EQ(*r.max_tokens, 5U);
  CHECK_EQ(r.stop.size(), 1U);
  CHECK_EQ(r.stop[0], std::string("\n"));
  CHECK(r.stream);
  CHECK(r.include_usage);
  CHECK(r.return_token_ids);
  CHECK_EQ(r.model, std::string("mock-model"));
  CHECK(!r.sampling.greedy);
  CHECK_NEAR(r.sampling.temperature, 0.7, 1e-6);
  CHECK_EQ(r.sampling.top_k, 40U);
  CHECK_NEAR(r.sampling.top_p, 0.95, 1e-6);

  nlohmann::json bad_n = chat_request();
  bad_n["n"] = 2;
  CHECK(rejects(bad_n.dump(), true));
  CHECK(rejects("{", true));

  const server::Request absent_temperature = server::parse_request(chat_request().dump(), true);
  CHECK(absent_temperature.sampling.greedy);
  nlohmann::json zero_temperature = chat_request();
  zero_temperature["temperature"] = 0;
  CHECK(server::parse_request(zero_temperature.dump(), true).sampling.greedy);

  const std::vector<uint32_t> prompt_ids = {1, 2};
  const std::vector<uint32_t> out_ids = {3, 4};
  const nlohmann::json completion = nlohmann::json::parse(server::completion_body(
      r, "chatcmpl-test", 123, "hello", "stop", {2, 2}, &prompt_ids, &out_ids));
  CHECK_EQ(completion.at("object"), std::string("chat.completion"));
  CHECK_EQ(completion.at("choices").at(0).at("message").at("content"), std::string("hello"));
  CHECK_EQ(completion.at("usage").at("prompt_tokens"), 2);
  CHECK_EQ(completion.at("usage").at("completion_tokens"), 2);
  CHECK_EQ(completion.at("prompt_token_ids"), prompt_ids);
  CHECK_EQ(completion.at("choices").at(0).at("token_ids"), out_ids);

  const std::string frame = server::stream_frame_text(r, "chatcmpl-test", 123, "piece");
  CHECK(frame.rfind("data: ", 0) == 0);
  CHECK(frame.size() >= 8);
  CHECK_EQ(frame.substr(frame.size() - 2), std::string("\n\n"));
  const nlohmann::json frame_json = nlohmann::json::parse(frame.substr(6, frame.size() - 8));
  CHECK_EQ(frame_json.at("choices").at(0).at("delta").at("content"), std::string("piece"));


  // Spec 7 §3.5: a chat body from a parsed output.
  server::ParsedOutput parsed;
  parsed.reasoning = "think";
  parsed.tool_calls.push_back({"call_x", "read", {{"filePath", "/a"}, {"limit", 3}}});
  const nlohmann::json with_calls = nlohmann::json::parse(server::completion_body(
      r, "chatcmpl-test", 123, "", "tool_calls", {2, 2}, nullptr, nullptr, &parsed));
  const nlohmann::json& message = with_calls.at("choices").at(0).at("message");
  CHECK(message.at("content").is_null());
  CHECK_EQ(message.at("reasoning_content"), std::string("think"));
  CHECK_EQ(message.at("tool_calls").at(0).at("id"), std::string("call_x"));
  CHECK_EQ(message.at("tool_calls").at(0).at("type"), std::string("function"));
  CHECK_EQ(message.at("tool_calls").at(0).at("function").at("name"), std::string("read"));
  CHECK_EQ(message.at("tool_calls").at(0).at("function").at("arguments"),
           std::string(R"({"filePath":"/a","limit":3})"));
  CHECK_EQ(with_calls.at("choices").at(0).at("finish_reason"), std::string("tool_calls"));
  CHECK_EQ(with_calls.at("usage").at("prompt_tokens_details").at("cached_tokens"), 0);
  server::ParsedOutput plain;
  plain.content = "just text";
  const nlohmann::json no_calls = nlohmann::json::parse(server::completion_body(
      r, "chatcmpl-test", 123, "", "stop", {2, 2}, nullptr, nullptr, &plain));
  CHECK_EQ(no_calls.at("choices").at(0).at("message").at("content"), std::string("just text"));
  CHECK(!no_calls.at("choices").at(0).at("message").contains("tool_calls"));
  CHECK(!no_calls.at("choices").at(0).at("message").contains("reasoning_content"));

  const nlohmann::json reasoning_frame = nlohmann::json::parse(
      server::stream_frame_reasoning(r, "chatcmpl-test", 123, "hm").substr(6));
  CHECK_EQ(reasoning_frame.at("choices").at(0).at("delta").at("reasoning_content"), std::string("hm"));
  const std::string call_frame_text =
      server::stream_frame_tool_call(r, "chatcmpl-test", 123, parsed.tool_calls[0], 2);
  const nlohmann::json call_frame = nlohmann::json::parse(call_frame_text.substr(6));
  const nlohmann::json& call = call_frame.at("choices").at(0).at("delta").at("tool_calls").at(0);
  CHECK_EQ(call.at("index"), 2);
  CHECK_EQ(call.at("id"), std::string("call_x"));
  CHECK_EQ(call.at("function").at("name"), std::string("read"));

  // /v1/completions body: byte-identical to the pre-spec-7 form.
  nlohmann::json completion_request = {{"model", "m"}, {"prompt", "p"}};
  const server::Request cr = server::parse_request(completion_request.dump(), false);
  CHECK_EQ(server::completion_body(cr, "cmpl-1", 5, "a<tool_call>b", "stop", {1, 2}, nullptr, nullptr),
           std::string(R"({"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"text":"a<tool_call>b"}],)"
                       R"("created":5,"id":"cmpl-1","model":"m","object":"text_completion",)"
                       R"("usage":{"completion_tokens":2,"prompt_tokens":1,"total_tokens":3}})"));

  // Messages as OpenAI clients send them back: null content beside tool_calls, text
  // parts, and arguments as a JSON string (the template needs an object).
  nlohmann::json history = chat_request();
  history["messages"].push_back({{"role", "assistant"}, {"content", nullptr},
                                 {"tool_calls", {{{"id", "call_1"}, {"type", "function"},
                                                  {"function", {{"name", "read"},
                                                                {"arguments", R"({"filePath":"/a"})"}}}}}}});
  history["messages"].push_back({{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "x"}});
  history["messages"].push_back(
      {{"role", "user"}, {"content", {{{"type", "text"}, {"text", "a"}}, {{"type", "text"}, {"text", "b"}}}}});
  const server::Request hr = server::parse_request(history.dump(), true);
  CHECK(hr.messages.at(1).at("content").is_null());
  CHECK_EQ(hr.messages.at(1).at("tool_calls").at(0).at("function").at("arguments"),
           nlohmann::json({{"filePath", "/a"}}));
  CHECK(hr.messages.at(3).at("content").is_array());
  nlohmann::json bad_args = history;
  bad_args["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{not json";
  CHECK(rejects(bad_args.dump(), true));
  nlohmann::json bad_part = chat_request();
  bad_part["messages"][0]["content"] = nlohmann::json::array({{{"type", "text"}}});
  CHECK(rejects(bad_part.dump(), true));

  const nlohmann::json error = nlohmann::json::parse(
      server::error_body("bad field", "invalid_request_error"));
  CHECK_EQ(error.at("error").at("message"), std::string("bad field"));
  CHECK_EQ(error.at("error").at("type"), std::string("invalid_request_error"));
  CHECK(error.at("error").at("param").is_null());
  CHECK(error.at("error").at("code").is_null());

  return 0;
}

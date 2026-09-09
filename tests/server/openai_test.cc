#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "server/openai.h"

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

  const nlohmann::json error = nlohmann::json::parse(
      server::error_body("bad field", "invalid_request_error"));
  CHECK_EQ(error.at("error").at("message"), std::string("bad field"));
  CHECK_EQ(error.at("error").at("type"), std::string("invalid_request_error"));
  CHECK(error.at("error").at("param").is_null());
  CHECK(error.at("error").at("code").is_null());

  return 0;
}

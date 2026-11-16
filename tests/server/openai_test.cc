#include <format>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "server/chat_format.h"
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
  CHECK_EQ(nlohmann::json(r.messages), body.at("messages"));
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

  // Spec 23e: a text-bearing part of another type (input_text) is still a text part; only media parts are refused.
  {
    nlohmann::json parts = chat_request();
    parts["messages"][0]["content"] = nlohmann::json::array({{{"type", "input_text"}, {"text", "hi"}}, {{"text", "x"}}});
    CHECK(!rejects(parts.dump(), true));
    for (const char* media : {"image_url", "image", "input_image", "video", "video_url", "input_audio", "audio"}) {
      parts["messages"][0]["content"] = nlohmann::json::array({{{"type", media}, {media, {{"url", "x"}}}}});
      CHECK(rejects(parts.dump(), true));
    }
  }

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
  CHECK_EQ(nlohmann::json(hr.messages.at(1).at("tool_calls").at(0).at("function").at("arguments")),
           nlohmann::json({{"filePath", "/a"}}));
  CHECK(hr.messages.at(3).at("content").is_array());
  // Key order is the body's (the template prints tools with tojson and arguments with | items, as transformers does
  // over a client's dicts): tools, message fields and a string's arguments - nested objects too - are not sorted.
  {
    const std::string ordered_body =
        R"({"model": "m", "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],)"
        R"( "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": null, "tool_calls": [)"
        R"({"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"z\": 1, \"a\": {\"y\": 2, \"b\": 3}}"}}]}]})";
    const server::Request o = server::parse_request(ordered_body, true);
    CHECK_EQ(o.tools.dump(), std::string(R"([{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}])"));
    CHECK_EQ(o.messages.at(1).dump(),
             std::string(R"({"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":)"
                         R"({"name":"f","arguments":{"z":1,"a":{"y":2,"b":3}}}}]})"));
    // ... and a parsed call's arguments go back to the client in the order the model wrote them.
    server::ParsedOutput out;
    out.tool_calls.push_back({"call_y", "f", nlohmann::ordered_json::parse(R"({"z": 1, "a": {"y": 2, "b": 3}})")});
    const nlohmann::json resp = nlohmann::json::parse(server::completion_body(o, "c", 1, "", "tool_calls", {1, 1}, nullptr,
                                                                              nullptr, &out));
    CHECK_EQ(resp.at("choices").at(0).at("message").at("tool_calls").at(0).at("function").at("arguments"),
             std::string(R"({"z":1,"a":{"y":2,"b":3}})"));
  }
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

  // Every refusal's text, odd values that are accepted, and every body's bytes - pinned before the request, the response
  // and the stream chunks moved to structs mapped by reflection (server/json_struct.h).
  for (const auto& [body, chat, text] : std::vector<std::tuple<std::string, bool, std::string>>{
           {"{", true, "invalid JSON body"},
           {"[]", true, "request body must be an object"},
           {R"({"model":5,"messages":[]})", true, "model must be a string"},
           {R"({"model":null,"messages":[]})", true, "model must be a string"},
           {R"({"n":2})", true, "n must be 1"},
           {R"({"n":"1"})", true, "n must be 1"},
           {R"({"n":1.0})", true, "n must be 1"},
           {R"({"n":-1})", true, "n must be 1"},
           {R"({})", true, "missing messages"},
           {R"({"messages":{}})", true, "messages must be an array"},
           {R"({"messages":[]})", true, "messages must not be empty"},
           {R"({"messages":[1]})", true, "messages entries must be objects"},
           {R"({"messages":[{"content":"x"}]})", true, "message role must be a string"},
           {R"({"messages":[{"role":"user"}]})", true, "content must be a string"},
           {R"({"messages":[{"role":"user","content":5}]})", true, "content must be a string"},
           {R"({"messages":[{"role":"user","content":[{"type":"text"}]}]})", true, "content parts must be text parts"},
           {R"({"messages":[{"role":"user","content":[{"type":"image_url","text":"x"}]}]})", true,
            "content part of type 'image_url' is refused: this server serves text only - the vision encoder (images, "
            "video) and any audio encoder are not loaded"},
           {R"({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","function":{"name":"f","arguments":"[1]"}}]}]})",
            true, "tool call arguments must be a JSON object (or a JSON string holding one): the call 'f' (id 'c1')"},
           {R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":[]})", true,
            "chat_template_kwargs must be an object"},
           {R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"enable_thinking":1}})", true,
            "enable_thinking must be a boolean"},
           {R"({})", false, "missing prompt"},
           {R"({"prompt":["a"]})", false, "prompt must be a string"},
           {R"({"prompt":"p","max_tokens":0})", false, "max_tokens must be a positive integer"},
           {R"({"prompt":"p","max_tokens":-1})", false, "max_tokens must be a positive integer"},
           {R"({"prompt":"p","max_tokens":1.5})", false, "max_tokens must be a positive integer"},
           {R"({"prompt":"p","max_tokens":"5"})", false, "max_tokens must be a positive integer"},
           {R"({"prompt":"p","max_tokens":4294967296})", false, "max_tokens must be a positive integer"},
           {R"({"prompt":"p","min_tokens":-1})", false, "min_tokens must be a non-negative integer"},
           {R"({"prompt":"p","ignore_eos":"true"})", false, "ignore_eos must be a boolean"},
           {R"({"prompt":"p","stream":null})", false, "stream must be a boolean"},
           {R"({"prompt":"p","return_token_ids":0})", false, "return_token_ids must be a boolean"},
           {R"({"prompt":"p","stream_options":true})", false, "stream_options must be an object"},
           {R"({"prompt":"p","stream_options":{"include_usage":"y"}})", false, "include_usage must be a boolean"},
           {R"({"prompt":"p","stop":5})", false, "stop must be a string or array of strings"},
           {R"({"prompt":"p","stop":null})", false, "stop must be a string or array of strings"},
           {R"({"prompt":"p","stop":["a","b","c","d","e"]})", false, "stop must contain at most 4 strings"},
           {R"({"prompt":"p","stop":["a",1]})", false, "stop entries must be strings"},
           {R"({"prompt":"p","temperature":"x"})", false, "temperature must be a number"},
           {R"({"prompt":"p","temperature":-0.5})", false, "temperature must be non-negative"},
           {R"({"prompt":"p","top_k":-1})", false, "top_k must be a non-negative integer"},
           {R"({"prompt":"p","top_k":1.0})", false, "top_k must be a non-negative integer"},
           {R"({"prompt":"p","top_p":0})", false, "top_p must be in (0, 1]"},
           {R"({"prompt":"p","top_p":1.5})", false, "top_p must be in (0, 1]"},
           {R"({"prompt":"p","top_p":"1"})", false, "top_p must be a number"},
           {R"({"prompt":"p","repetition_penalty":0})", false, "repetition_penalty must be a finite number > 0"},
           {R"({"prompt":"p","repetition_penalty":"2"})", false, "repetition_penalty must be a finite number > 0"},
           {R"({"prompt":"p","seed":-1})", false, "seed must be a non-negative integer"},
           {R"({"prompt":"p","seed":1.5})", false, "seed must be a non-negative integer"},
           // Two faults: the first in the order the fields are checked.
           {R"({"model":5,"n":2})", true, "model must be a string"},
           {R"({"n":2,"max_tokens":0})", true, "n must be 1"},
           {R"({"max_tokens":0,"messages":5})", true, "messages must be an array"},
           {R"({"prompt":"p","stop":5,"temperature":"x"})", false, "stop must be a string or array of strings"},
           {R"({"prompt":"p","seed":-1,"max_tokens":0})", false, "max_tokens must be a positive integer"},
       }) {
    std::string got = "(accepted)";
    try {
      (void)server::parse_request(body, chat);
    } catch (const server::BadRequest& e) {
      got = e.what();
    }
    if (got != text) std::fprintf(stderr, "body %s: '%s', want '%s'\n", body.c_str(), got.c_str(), text.c_str());
    CHECK(got == text);
  }

  // What a body is read as: every Request field, odd values included.
  const auto dump = [](const server::Request& q) {
    std::string stop;
    for (const std::string& s : q.stop) stop += "'" + s + "' ";
    return std::format(
        "chat {} model '{}' prompt '{}' max {} min {} ignore_eos {} stream {} usage {} ids {} stop [{}] greedy {} T {} "
        "k {} p {} rep {} seed {} {} think {} kwargs {} effort {} tools {} messages {}",
        q.chat, q.model, q.prompt, q.max_tokens ? std::to_string(*q.max_tokens) : "-", q.min_tokens, q.ignore_eos,
        q.stream, q.include_usage, q.return_token_ids, stop, q.sampling.greedy, q.sampling.temperature,
        q.sampling.top_k, q.sampling.top_p, q.sampling.repetition_penalty, q.sampling.seed, q.sampling.has_seed,
        q.enable_thinking, q.template_kwargs.dump(), q.reasoning_effort.dump(), q.tools.dump(), q.messages.dump());
  };
  for (const auto& [body, chat, want] : std::vector<std::tuple<std::string, bool, std::string>>{
           {R"({"prompt":"p"})", false,
            "chat false model '' prompt 'p' max - min 0 ignore_eos false stream false usage false ids false stop [] "
            "greedy true T 1 k 20 p 0.95 rep 1 seed 0 false think true kwargs {} effort null tools null messages null"},
           {R"({"prompt":"p","model":"m","n":1,"max_tokens":4294967295,"min_tokens":0,"ignore_eos":true,"stream":true,)"
            R"("stream_options":{"include_usage":true,"x":1},"return_token_ids":false,"stop":"\n","temperature":2,)"
            R"("top_k":0,"top_p":1,"repetition_penalty":null,"seed":18446744073709551615,"unknown":[1],)"
            R"("chat_template_kwargs":5,"reasoning_effort":"high","messages":7})",
            false,
            "chat false model 'm' prompt 'p' max 4294967295 min 0 ignore_eos true stream true usage true ids false "
            "stop ['\n' ] greedy false T 2 k 0 p 1 rep 1 seed 18446744073709551615 true think true kwargs {} effort "
            "null tools null messages null"},
           {R"({"prompt":"p","max_tokens":1,"max_tokens":2,"stop":[],"temperature":0,"repetition_penalty":1.5,"seed":0})",
            false,
            "chat false model '' prompt 'p' max 2 min 0 ignore_eos false stream false usage false ids false stop [] "
            "greedy true T 0 k 20 p 0.95 rep 1.5 seed 0 true think true kwargs {} effort null tools null messages null"},
           {R"({"prompt":"p","stop":["a","b","c","d"],"stream_options":{},"temperature":0.25})", false,
            "chat false model '' prompt 'p' max - min 0 ignore_eos false stream false usage false ids false "
            "stop ['a' 'b' 'c' 'd' ] greedy false T 0.25 k 20 p 0.95 rep 1 seed 0 false think true kwargs {} effort "
            "null tools null messages null"},
           {R"({"messages":[{"z":1,"role":"user","content":"x"}],"tools":{"b":1,"a":2},"prompt":5,)"
            R"("chat_template_kwargs":{"z":true,"enable_thinking":false},"reasoning_effort":5})",
            true,
            "chat true model '' prompt '' max - min 0 ignore_eos false stream false usage false ids false stop [] "
            "greedy true T 1 k 20 p 0.95 rep 1 seed 0 false think false kwargs {\"enable_thinking\":false,\"z\":true} "
            "effort 5 tools {\"b\":1,\"a\":2} messages [{\"z\":1,\"role\":\"user\",\"content\":\"x\"}]"},
           {R"({"messages":[{"role":"user","content":null},{"role":"user","content":[{"text":"a"}]}],)"
            R"("chat_template_kwargs":{}})",
            true,
            "chat true model '' prompt '' max - min 0 ignore_eos false stream false usage false ids false stop [] "
            "greedy true T 1 k 20 p 0.95 rep 1 seed 0 false think true kwargs {} effort null tools null messages "
            "[{\"role\":\"user\",\"content\":null},{\"role\":\"user\",\"content\":[{\"text\":\"a\"}]}]"},
       }) {
    const std::string got = dump(server::parse_request(body, chat));
    if (got != want) std::fprintf(stderr, "body %s:\n got  %s\n want %s\n", body.c_str(), got.c_str(), want.c_str());
    CHECK(got == want);
  }
  {
    // The model's sampling defaults reach a request that names none; a seed is the request's alone.
    server::Sampling d;
    d.greedy = false;
    d.temperature = 0.6f;
    d.top_k = 7;
    d.top_p = 0.5f;
    d.has_seed = true;
    d.seed = 9;
    const server::Request q = server::parse_request(R"({"prompt":"p"})", false, d);
    CHECK(!q.sampling.greedy && q.sampling.temperature == 0.6f && q.sampling.top_k == 7 && q.sampling.top_p == 0.5f);
    CHECK(!q.sampling.has_seed && q.sampling.seed == 0);
    CHECK(server::parse_request(R"({"prompt":"p","temperature":0})", false, d).sampling.greedy);
  }

  // The bytes of every body and chunk (keys sorted: nlohmann::json's object order).
  {
    const server::Request chat_r = server::parse_request(R"({"model":"m","messages":[{"role":"user","content":"x"}]})", true);
    server::Request chat_ids = chat_r;
    chat_ids.return_token_ids = true;
    const server::Request text_r = server::parse_request(R"({"model":"m","prompt":"p"})", false);
    const std::vector<uint32_t> pids = {1, 2}, oids = {3};
    server::ParsedOutput calls;
    calls.reasoning = "r";
    calls.tool_calls.push_back({"call_a", "f", nlohmann::ordered_json::parse(R"({"z":"\u00e9","a":[1]})")});
    calls.tool_calls.push_back({"call_b", "g", nlohmann::ordered_json::object()});
    calls.tool_calls[1].arguments["s"] = std::string("cut \xe2\x82");   // a character cut by the model's output
    server::ParsedOutput empty;
    server::ParsedOutput text_and_call;
    text_and_call.content = "c";
    text_and_call.tool_calls.push_back({"call_c", "h", nlohmann::ordered_json::object()});
    const server::Usage u{3, 2, 1};
    for (const auto& [got, want] : std::vector<std::pair<std::string, std::string>>{
             {server::completion_body(chat_r, "id1", 7, "hi", "stop", u, nullptr, nullptr),
              R"({"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],)"
              R"("created":7,"id":"id1","model":"m","object":"chat.completion","usage":{"completion_tokens":2,)"
              R"("prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":1},"total_tokens":5}})"},
             {server::completion_body(chat_ids, "id1", 7, "hi", "length", u, &pids, &oids, &calls),
              R"({"choices":[{"finish_reason":"length","index":0,"message":{"content":null,"reasoning_content":"r",)"
              R"("role":"assistant","tool_calls":[{"function":{"arguments":"{\"z\":\"é\",\"a\":[1]}","name":"f"},)"
              R"("id":"call_a","type":"function"},{"function":{"arguments":"{\"s\":\"cut �\"}","name":"g"},)"
              R"("id":"call_b","type":"function"}]},"token_ids":[3]}],"created":7,"id":"id1","model":"m",)"
              R"("object":"chat.completion","prompt_token_ids":[1,2],"usage":{"completion_tokens":2,"prompt_tokens":3,)"
              R"("prompt_tokens_details":{"cached_tokens":1},"total_tokens":5}})"},
             {server::completion_body(chat_r, "id1", 7, "hi", "stop", u, &pids, &oids, &empty),
              R"({"choices":[{"finish_reason":"stop","index":0,"message":{"content":"","role":"assistant"}}],)"
              R"("created":7,"id":"id1","model":"m","object":"chat.completion","usage":{"completion_tokens":2,)"
              R"("prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":1},"total_tokens":5}})"},
             {server::completion_body(chat_r, "id1", 7, "", "tool_calls", u, nullptr, nullptr, &text_and_call),
              R"({"choices":[{"finish_reason":"tool_calls","index":0,"message":{"content":"c","role":"assistant",)"
              R"("tool_calls":[{"function":{"arguments":"{}","name":"h"},"id":"call_c","type":"function"}]}}],)"
              R"("created":7,"id":"id1","model":"m","object":"chat.completion","usage":{"completion_tokens":2,)"
              R"("prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":1},"total_tokens":5}})"},
             {server::completion_body(text_r, "id2", 8, "t", "stop", u, &pids, &oids),
              R"({"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"text":"t"}],"created":8,"id":"id2",)"
              R"("model":"m","object":"text_completion","usage":{"completion_tokens":2,"prompt_tokens":3,)"
              R"("total_tokens":5}})"},
             {server::stream_frame_role(chat_r, "id1", 7),
              "data: " R"({"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":7,)"
              R"("id":"id1","model":"m","object":"chat.completion.chunk"})" "\n\n"},
             {server::stream_frame_role(text_r, "id2", 8), ""},
             {server::stream_frame_text(chat_r, "id1", 7, "a\"b"),
              "data: " R"({"choices":[{"delta":{"content":"a\"b"},"finish_reason":null,"index":0}],"created":7,)"
              R"("id":"id1","model":"m","object":"chat.completion.chunk"})" "\n\n"},
             {server::stream_frame_text(text_r, "id2", 8, "a"),
              "data: " R"({"choices":[{"finish_reason":null,"index":0,"logprobs":null,"text":"a"}],"created":8,)"
              R"("id":"id2","model":"m","object":"text_completion"})" "\n\n"},
             {server::stream_frame_reasoning(chat_r, "id1", 7, "r"),
              "data: " R"({"choices":[{"delta":{"reasoning_content":"r"},"finish_reason":null,"index":0}],)"
              R"("created":7,"id":"id1","model":"m","object":"chat.completion.chunk"})" "\n\n"},
             {server::stream_frame_tool_call(chat_r, "id1", 7, calls.tool_calls[0], 1),
              "data: " R"({"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"z\":\"é\",\"a\":[1]}",)"
              R"("name":"f"},"id":"call_a","index":1,"type":"function"}]},"finish_reason":null,"index":0}],)"
              R"("created":7,"id":"id1","model":"m","object":"chat.completion.chunk"})" "\n\n"},
             {server::stream_frame_finish(chat_r, "id1", 7, "stop"),
              "data: " R"({"choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":7,"id":"id1",)"
              R"("model":"m","object":"chat.completion.chunk"})" "\n\n"},
             {server::stream_frame_finish(text_r, "id2", 8, "length"),
              "data: " R"({"choices":[{"finish_reason":"length","index":0,"logprobs":null,"text":""}],"created":8,)"
              R"("id":"id2","model":"m","object":"text_completion"})" "\n\n"},
             {server::stream_frame_usage(chat_r, "id1", 7, u),
              "data: " R"({"choices":[],"created":7,"id":"id1","model":"m","object":"chat.completion.chunk",)"
              R"("usage":{"completion_tokens":2,"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":1},)"
              R"("total_tokens":5}})" "\n\n"},
             {server::stream_frame_usage(text_r, "id2", 8, u),
              "data: " R"({"choices":[],"created":8,"id":"id2","model":"m","object":"text_completion",)"
              R"("usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}})" "\n\n"},
             {server::error_body("bad \"x\"", "invalid_request_error"),
              R"({"error":{"code":null,"message":"bad \"x\"","param":null,"type":"invalid_request_error"}})"},
             {server::models_body("served", 9),
              R"({"data":[{"created":9,"id":"served","object":"model","owned_by":"b70"}],"object":"list"})"},
         }) {
      if (got != want) std::fprintf(stderr, "got  %s\nwant %s\n", got.c_str(), want.c_str());
      CHECK(got == want);
    }
  }

  // The chat formats' chat_template_kwargs and reasoning_effort refusals (a 400, std::invalid_argument).
  {
    const auto refusal = [](const server::ChatFormat& f, const nlohmann::json& kwargs, const nlohmann::json& effort) {
      try {
        server::check_template_kwargs(f, kwargs);
        (void)server::request_template_kwargs(f, kwargs, effort);
      } catch (const std::invalid_argument& e) {
        return std::string(e.what());
      }
      return std::string("(accepted)");
    };
    const server::ChatFormat bolt = server::ChatFormat::bolt(), kol = server::ChatFormat::kolibri(), qwen{};
    CHECK_EQ(refusal(bolt, {{"medium_effort", 1}}, nullptr),
             std::string("chat_template_kwargs.medium_effort must be a boolean"));
    CHECK_EQ(refusal(bolt, {{"enable_thinking", "no"}, {"force_nonempty_content", 0}}, nullptr),
             std::string("chat_template_kwargs.enable_thinking must be a boolean"));
    CHECK_EQ(refusal(bolt, {{"truncate_history_thinking", true}, {"x", 1}}, nullptr), std::string("(accepted)"));
    CHECK_EQ(refusal(bolt, {{"force_nonempty_content", nullptr}}, nullptr),
             std::string("chat_template_kwargs.force_nonempty_content must be a boolean"));
    CHECK_EQ(refusal(bolt, nlohmann::json::object(), "huge"),
             std::string("reasoning_effort must be one of none, minimal, low, medium, high, xhigh (Nemotron 3.5 Bolt: "
                         "none / minimal -> enable_thinking false, low / medium -> medium_effort true, high -> the "
                         "default)"));
    CHECK_EQ(refusal(kol, {{"reasoning_effort", "huge"}}, nullptr),
             std::string("chat_template_kwargs.reasoning_effort must be one of none, minimal, low, medium, high, xhigh, "
                         "max"));
    CHECK_EQ(refusal(kol, {{"reasoning_effort", 3}}, nullptr),
             std::string("chat_template_kwargs.reasoning_effort must be one of none, minimal, low, medium, high, xhigh, "
                         "max"));
    CHECK_EQ(refusal(kol, {{"reasoning_effort", nullptr}, {"preserve_thinking", false}}, nullptr),
             std::string("(accepted)"));
    CHECK_EQ(refusal(kol, {{"preserve_thinking", "yes"}}, nullptr),
             std::string("chat_template_kwargs.preserve_thinking must be a boolean"));
    CHECK_EQ(refusal(kol, nlohmann::json::object(), "max!"),
             std::string("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, max (Kolibri-1)"));
    CHECK_EQ(refusal(qwen, {{"enable_thinking", 5}}, "anything"), std::string("(accepted)"));
    CHECK_EQ(server::request_template_kwargs(bolt, {{"medium_effort", false}}, "low").dump(),
             std::string(R"({"medium_effort":false})"));
    CHECK_EQ(server::request_template_kwargs(bolt, nlohmann::json::object(), "none").dump(),
             std::string(R"({"enable_thinking":false})"));
    CHECK_EQ(server::request_template_kwargs(kol, nlohmann::json::object(), "low").dump(),
             std::string(R"({"reasoning_effort":"low"})"));
  }

  return 0;
}

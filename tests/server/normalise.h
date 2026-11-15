#pragma once
// The mock-server tests' reply in a comparable form: tool-call ids to call_X and the "created" stamp dropped, from
// the JSON body or (sse) from every SSE data frame.
#include <regex>
#include <string>

#include <nlohmann/json.hpp>

inline std::string normalise(const std::string& raw, bool sse) {
  using json = nlohmann::json;
  const std::string body = std::regex_replace(raw, std::regex("call_[0-9a-f]+"), "call_X");
  if (!sse) {
    json j = json::parse(body);
    j.erase("created");
    return j.dump();
  }
  std::string out;
  size_t begin = 0;
  while (begin < body.size()) {
    const size_t end = body.find("\n\n", begin);
    if (end == std::string::npos) break;
    std::string frame = body.substr(begin, end - begin);
    begin = end + 2;
    if (frame.rfind("data: {", 0) == 0) {
      json j = json::parse(frame.substr(6));
      j.erase("created");
      frame = "data: " + j.dump();
    }
    out += frame + "\n";
  }
  return out;
}

// Spec 25e: the server's reading of Nemotron 3.5 Bolt's generated texts, for tools/toolcall/bolt_parser_check.py (the A4 gate
// of row 62: every reference output the scorer reads a call from, the server's parser reads the same call from).
//   bolt_parse_dump [--no-thinking] [--force-nonempty-content] <text file>...
// One JSON object a line on stdout, in argument order: {"file", "reasoning", "content", "tool_calls": [{"name", "arguments"}]}.
// The text is what the model wrote after the prompt's "<think>\n" (the A4 run files, special tokens kept) - exactly what
// server::BoltOutputStream is fed - parsed whole with no tools (the values are compared by the checker against the scorer's
// raw strings, so the schema's typing does not matter).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "server/toolcall_bolt.h"

int main(int argc, char** argv) {
  bool thinking = true, force = false;
  int i = 1;
  for (; i < argc && argv[i][0] == '-' && argv[i][1] == '-'; ++i) {
    const std::string a = argv[i];
    if (a == "--no-thinking") thinking = false;
    else if (a == "--force-nonempty-content") force = true;
    else {
      std::fprintf(stderr, "bolt_parse_dump: unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (i == argc) {
    std::fprintf(stderr, "usage: bolt_parse_dump [--no-thinking] [--force-nonempty-content] <text file>...\n");
    return 2;
  }
  for (; i < argc; ++i) {
    std::ifstream f(argv[i], std::ios::binary);
    if (!f) {
      std::fprintf(stderr, "bolt_parse_dump: cannot read %s\n", argv[i]);
      return 1;
    }
    std::stringstream text;
    text << f.rdbuf();
    const server::ParsedOutput p = server::parse_bolt_output(text.str(), thinking, force, nlohmann::json());
    nlohmann::json calls = nlohmann::json::array();
    for (const server::ToolCall& c : p.tool_calls) calls.push_back({{"name", c.name}, {"arguments", c.arguments}});
    const nlohmann::json line = {{"file", argv[i]}, {"reasoning", p.reasoning}, {"content", p.content}, {"tool_calls", calls}};
    std::printf("%s\n", line.dump().c_str());
  }
  return 0;
}

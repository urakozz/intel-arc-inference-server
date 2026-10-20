// Spec 24e Task 5: tools/toolcall/lfm2moe_rules (decision 7's evidence) on generated files.
//   a.txt  a generation with one call whose string holds an unescaped quote, then <|im_end|> and what the model wrote after
//          it (b70-decode runs on past EOS: the tool must cut there, so the second block is never read);
//   b.txt  a block with an invalid call: left as content, which the tool counts as a failure;
//   c.txt  a bare call, a quoted name and a JSON literal: three counters, no failure.
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "check.h"
#include "../../tools/toolcall/lfm2moe_rules.h"

namespace {

std::string write(const std::filesystem::path& dir, const char* name, const std::string& text) {
  const std::string path = (dir / name).string();
  std::ofstream(path, std::ios::binary) << text;
  return path;
}

std::string report(const std::vector<std::string>& files, size_t& failures) {
  std::FILE* tmp = std::tmpfile();
  CHECK(tmp != nullptr);
  failures = lfm2moe_rules::run(files, tmp);
  std::rewind(tmp);
  std::string out;
  char buf[512];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, tmp)) > 0) out.append(buf, n);
  std::fclose(tmp);
  return out;
}

bool has(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

}  // namespace

int main() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("lfm2moe_rules_test_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  const std::string a = write(dir, "a.txt",
      "thinking</think>ok <|tool_call_start|>[f(a='it's')]<|tool_call_end|><|im_end|>more <|tool_call_start|>[g(1)]<|tool_call_end|>");
  const std::string b = write(dir, "b.txt", "x</think><|tool_call_start|>[f(1)]<|tool_call_end|>");
  const std::string c = write(dir, "c.txt", "x</think><|tool_call_start|>'g'(flag=true)<|tool_call_end|>");

  size_t failures = 0;
  std::string out = report({a}, failures);
  CHECK_EQ(failures, size_t(0));
  CHECK(has(out, "1 files, 1 with calls (1 calls), 0 with a block left as content"));
  CHECK(has(out, "quote_recovered 1,"));
  CHECK(!has(out, "[a tool-call block was left as content]"));

  out = report({a, b}, failures);
  CHECK_EQ(failures, size_t(1));
  CHECK(has(out, "2 files, 1 with calls (1 calls), 1 with a block left as content"));
  CHECK(has(out, "[a tool-call block was left as content]"));

  out = report({c}, failures);
  CHECK_EQ(failures, size_t(0));
  CHECK(has(out, "quoted_name 1, bare_call 1, json_literal 1,"));

  bool threw = false;
  try {
    (void)report({(dir / "missing.txt").string()}, failures);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  std::filesystem::remove_all(dir);
  std::printf("lfm2moe_rules_test OK\n");
  return 0;
}

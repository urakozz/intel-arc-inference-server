#include <cstdlib>
#include <string>

#include "check.h"
#include "tokenizer/chat_template.h"

namespace {

std::string snapshot() {
  const char* env = std::getenv("B70_SNAPSHOT_DIR");
  return env ? env
             : "/home/user/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/"
               "snapshots/84575a18f209992ef96d819b31f924b489e3d55d";
}

}  // namespace

int main() {
  chat::Template t(snapshot());
  const auto text = t.render(nlohmann::json::array({{{"role", "user"}, {"content", "Hello"}}}),
                             nullptr, false);
  CHECK(text.rfind("<|im_start|>", 0) == 0);
  const std::string bare = "<|im_start|>assistant\n";
  const std::string with_think = bare + "<think>\n\n</think>\n\n";
  const bool ends_bare = text.size() >= bare.size() &&
                         text.compare(text.size() - bare.size(), bare.size(), bare) == 0;
  const bool ends_with_think = text.size() >= with_think.size() &&
                               text.compare(text.size() - with_think.size(), with_think.size(),
                                            with_think) == 0;
  CHECK(ends_bare || ends_with_think);
  return 0;
}

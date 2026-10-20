// Spec 24e (plan 24e Task 5): lfm2moe_rules <generated .txt>... - see lfm2moe_rules.h. The exit status is the number of
// files whose tool-call block was left as content (0: every block was read as calls), 2 on a usage or read error.
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "lfm2moe_rules.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: lfm2moe_rules <generated .txt>...\n");
    return 2;
  }
  try {
    const size_t failures = lfm2moe_rules::run(std::vector<std::string>(argv + 1, argv + argc), stdout);
    return failures > 255 ? 255 : int(failures);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
}

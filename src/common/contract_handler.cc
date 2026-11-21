// The contract violation handler every binary links (the top CMakeLists.txt): GCC 16 leaves it to the program unless
// libstdc++exp is linked. One line on stderr - the failed predicate, where, in which function - then abort: an
// enforced contract never returns to the caller (tests/contract_death.h waits for the SIGABRT).
#include <contracts>
#include <cstdio>
#include <cstdlib>

void handle_contract_violation(const std::contracts::contract_violation& v) {
  const std::source_location at = v.location();
  std::fprintf(stderr, "contract violated: %s at %s:%u in %s\n", v.comment(), at.file_name(), unsigned(at.line()),
               at.function_name());
  std::abort();
}

// Plan 5d's test seam: `--chunk-cap 2048|4096` anywhere on a test's command line sets the prefill
// chunk cap every engine of this process uses when nobody set one
// (runtime::testing::set_default_prefill_chunk_cap) and is removed from argv, so the test's own
// positional arguments are unchanged. tests/CMakeLists.txt registers the `_cap4096` runs with it.
//
// It replaces the B70_PREFILL_CHUNK_CAP environment variable of the first build (operator,
// 2026-11-03: no new engine env vars): only a test binary can move the default; the CLIs and the
// server choose the cap themselves (cli/max_len.h).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "runtime/buffer_sizes.h"

namespace b70test {

// Returns the cap now in force (2048 when the argument is absent).
inline uint32_t take_chunk_cap_arg(int& argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--chunk-cap") != 0) continue;
    if (i + 1 >= argc) throw std::invalid_argument("--chunk-cap needs 2048 or 4096");
    const unsigned long v = std::strtoul(argv[i + 1], nullptr, 10);
    runtime::testing::set_default_prefill_chunk_cap(uint32_t(v));
    for (int j = i; j + 2 <= argc; ++j) argv[j] = argv[j + 2];   // drop the pair (argv[argc] stays null)
    argc -= 2;
    std::printf("prefill chunk cap: %lu (--chunk-cap, the test seam)\n", v);
    break;
  }
  return runtime::default_prefill_chunk_cap();
}

}  // namespace b70test

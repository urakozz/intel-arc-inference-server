// cli::auto_prefix_cache (b70-serve --prefix-cache-gb auto): the prefix cache is pinned
// SYSTEM RAM, sized min(32 GiB, half the RAM, available - 8 GiB), off below 4 GiB.
#include <cstdint>
#include <cstdio>
#include <string>
#include "check.h"
#include "cli/prefix_cache_size.h"

namespace {
cli::HostMemory gib(uint64_t total, uint64_t available) {
  return {total * cli::kGiB, available * cli::kGiB, true};
}
}  // namespace

int main() {
  // /proc/meminfo's own spelling, kB.
  const cli::HostMemory m = cli::parse_meminfo(
      "MemTotal:       131072000 kB\nMemFree:          1000000 kB\n"
      "MemAvailable:   104857600 kB\nBuffers:           12345 kB\n");
  CHECK(m.known);
  CHECK_EQ(m.total, uint64_t{131072000} * 1024);
  CHECK_EQ(m.available, uint64_t{104857600} * 1024);
  CHECK(!cli::parse_meminfo("MemTotal: 1000 kB\n").known);   // no MemAvailable: unknown

  CHECK_EQ(cli::auto_prefix_cache(gib(128, 100)).gib, 32u);   // the 32 GiB cap
  CHECK_EQ(cli::auto_prefix_cache(gib(48, 40)).gib, 24u);     // half the RAM
  CHECK_EQ(cli::auto_prefix_cache(gib(64, 20)).gib, 12u);     // available - 8
  CHECK_EQ(cli::auto_prefix_cache(gib(32, 11)).gib, 0u);      // 3 GiB to spare: off
  CHECK_EQ(cli::auto_prefix_cache(gib(16, 6)).gib, 0u);       // less than the headroom: off
  CHECK_EQ(cli::auto_prefix_cache(gib(32, 12)).gib, 4u);      // exactly the floor: on
  CHECK_EQ(cli::auto_prefix_cache({}).gib, 0u);               // unknown: off
  CHECK(cli::auto_prefix_cache({}).why.find("unknown") != std::string::npos);
  CHECK(cli::auto_prefix_cache(gib(32, 11)).why.find("off") != std::string::npos);
  std::puts("prefix_cache_size_test OK");
  return 0;
}

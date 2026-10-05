#pragma once
// b70-serve --prefix-cache-gb auto|N: how much SYSTEM RAM the prefix cache pins (spec 7).
//
// The prefix cache is pinned host memory, not GPU memory: every prefill writes through to it
// and a restore copies back from it, so the card's size never limits it - the machine's RAM
// does. Pinned pages cannot be swapped, so a cache sized past what the machine can spare
// starves everything else on it rather than paging. `auto` therefore takes the smallest of
//   - 32 GiB (the old fixed default, enough for ~500k positions of Qwen3.8 KV + snapshots),
//   - half of the machine's RAM,
//   - the RAM available when the cache is allocated (after the model is loaded) minus 8 GiB
//     of headroom for the server, the tokenizer, the OS and whatever else runs there,
// and turns the cache off below 4 GiB, where it would hold too little history to matter.
// Header-only and device-free: tests/cli/prefix_cache_size_test.cc feeds it /proc/meminfo
// text.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace cli {

struct HostMemory {
  uint64_t total = 0;       // bytes; MemTotal
  uint64_t available = 0;   // bytes; MemAvailable (Linux), else `total`
  bool known = false;
};

// MemTotal / MemAvailable from /proc/meminfo's text ("MemTotal:  131900000 kB").
inline HostMemory parse_meminfo(const std::string& text) {
  HostMemory m;
  std::istringstream in(text);
  std::string key, unit;
  uint64_t kb = 0;
  bool have_total = false, have_avail = false;
  while (in >> key >> kb) {
    std::getline(in, unit);
    if (key == "MemTotal:") {
      m.total = kb * 1024;
      have_total = true;
    } else if (key == "MemAvailable:") {
      m.available = kb * 1024;
      have_avail = true;
    }
  }
  m.known = have_total && have_avail;
  return m;
}

inline HostMemory host_memory() {
  std::ifstream f("/proc/meminfo");
  if (f) {
    std::stringstream s;
    s << f.rdbuf();
    const HostMemory m = parse_meminfo(s.str());
    if (m.known) return m;
  }
  // Not Linux (a host build): the total only, and nothing about what is free.
  const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
  HostMemory m;
  if (pages > 0 && page > 0) {
    m.total = uint64_t(pages) * uint64_t(page);
    m.available = m.total;
    m.known = true;
  }
  return m;
}

struct PrefixCacheChoice {
  uint32_t gib = 0;   // 0 = off
  std::string why;    // for the startup line
};

constexpr uint64_t kGiB = uint64_t(1) << 30;
constexpr uint32_t kPrefixCacheMaxGiB = 32;
constexpr uint32_t kPrefixCacheMinGiB = 4;
constexpr uint64_t kPrefixCacheHeadroom = 8 * kGiB;

inline PrefixCacheChoice auto_prefix_cache(const HostMemory& m) {
  if (!m.known) return {0, "auto: host RAM unknown, off"};
  const uint64_t half = m.total / 2;
  const uint64_t spare = m.available > kPrefixCacheHeadroom ? m.available - kPrefixCacheHeadroom : 0;
  const uint64_t pick = std::min<uint64_t>({kPrefixCacheMaxGiB * kGiB, half, spare}) / kGiB;
  char buf[160];
  std::snprintf(buf, sizeof buf, "auto: %.0f GiB RAM, %.0f GiB available", double(m.total) / kGiB,
                double(m.available) / kGiB);
  std::string why = buf;
  if (pick < kPrefixCacheMinGiB) return {0, why + "; under " + std::to_string(kPrefixCacheMinGiB) +
                                                " GiB to spare, off"};
  return {uint32_t(pick), why};
}

}  // namespace cli

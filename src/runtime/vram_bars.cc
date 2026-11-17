#include "runtime/vram_bars.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <format>
#include <string>
#include <vector>

namespace runtime {

std::vector<VramPart> vram_parts(const MemoryComponents& c, const std::string& state, const std::string& drafter) {
  const size_t dv_weights = c.drafter ? 0 : c.draft_vocab;
  return {{"weights", c.model + dv_weights},
          {"KV cache", c.kv},
          {state, c.decode_state},
          {drafter, c.drafter + (c.draft_vocab - dv_weights)},
          {"int8 prefill state", c.int8},
          {"prefill scratch", c.prefill_scratch}};
}

std::string vram_state_name(const MemoryPlan& p) {
  const std::string slots = p.slot_lens.size() > 1 ? std::format(" × {:d} slots", p.slot_lens.size()) : "";
  if (p.mtp) return slots.empty() ? "MTP head" : "MTP" + slots;
  return "decode state" + slots;
}

namespace {
size_t free_bytes(const VramCard& c) {
  size_t used = c.reserve;
  for (const VramPart& p : c.used) used += p.bytes;
  return used < c.total ? c.total - used : 0;
}

// The non-zero parts in the bar's order, free and the reserve last.
std::vector<VramPart> shown(const VramCard& c) {
  std::vector<VramPart> out;
  for (const VramPart& p : c.used)
    if (p.bytes) out.push_back(p);
  if (const size_t f = free_bytes(c)) out.push_back({"free", f});
  if (c.reserve) out.push_back({"reserve", c.reserve});
  return out;
}
}  // namespace

std::vector<int> vram_cells(const VramCard& c) {
  const std::vector<VramPart> parts = shown(c);
  const size_t n_used = std::count_if(c.used.begin(), c.used.end(), [](const VramPart& p) { return p.bytes != 0; });
  const bool has_free = free_bytes(c) != 0;
  std::vector<int> cells(parts.size());
  int sum = 0;
  for (size_t i = 0; i < parts.size(); ++i) {
    cells[i] = std::max(1, int(std::lround(double(parts[i].bytes) / double(c.total) * kVramCells)));
    if (!(has_free && i == n_used)) sum += cells[i];
  }
  // Free stretches to the width; without free (or when rounding leaves it under one cell) the largest used part
  // gives or takes the difference.
  int rest = kVramCells - sum;
  if (has_free) {
    cells[n_used] = std::max(1, rest);
    rest = std::min(0, rest - 1);
  }
  if (rest != 0 && n_used != 0) {
    size_t big = 0;
    for (size_t i = 1; i < n_used; ++i)
      if (parts[i].bytes > parts[big].bytes) big = i;
    cells[big] = std::max(1, cells[big] + rest);
  }
  return cells;
}

bool vram_utf8() {
  for (const char* var : {"LC_ALL", "LC_CTYPE", "LANG"}) {
    const char* v = std::getenv(var);
    if (v && *v) {
      std::string s(v);
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return std::toupper(ch); });
      return s.find("UTF-8") != std::string::npos || s.find("UTF8") != std::string::npos;
    }
  }
  return false;
}

namespace {
// "a, b, c" broken before an item that would pass kVramWrap, each broken line ending in ",".
std::string wrap(const std::vector<std::string>& items) {
  std::string out, line;
  for (size_t i = 0; i < items.size(); ++i) {
    const std::string sep = i ? ", " : "";
    if (!line.empty() && line.size() + sep.size() + items[i].size() > kVramWrap) {
      out += line + ",\n";
      line = items[i];
    } else {
      line += sep + items[i];
    }
  }
  return out + line + "\n";
}

std::string ascii(std::string s) {
  for (size_t at; (at = s.find("×")) != std::string::npos;) s.replace(at, std::string("×").size(), "x");
  return s;
}
}  // namespace

std::string vram_block(const VramCard& c, uint32_t card, uint32_t cards, bool utf8) {
  static const char* const kShades[] = {"█", "▓", "▒", "░", "▚", "▞"};
  static const char* const kAscii[] = {"#", "="};
  const std::vector<VramPart> parts = shown(c);
  const std::vector<int> cells = vram_cells(c);
  const std::string title = std::format("VRAM, card {:d} of {:d}", card, cards);
  const std::string total = std::format("{:.1f} GB", c.total / 1e9);
  std::string out = title + std::string(std::max<int>(1, kVramCells - int(title.size() + total.size())), ' ') + total +
                    "\n";
  const size_t n_used = parts.size() - (free_bytes(c) != 0) - (c.reserve != 0);
  std::vector<std::string> items;
  for (size_t i = 0; i < parts.size(); ++i) {
    const bool is_reserve = i == parts.size() - 1 && c.reserve != 0;
    const char* cell = is_reserve   ? "|"
                       : i >= n_used ? (utf8 ? "·" : ".")
                       : utf8       ? kShades[i % 6]
                                    : kAscii[i % 2];
    for (int k = 0; k < cells[i]; ++k) out += cell;
    const std::string name = utf8 ? parts[i].name : ascii(parts[i].name);
    items.push_back(std::format("{} {:.1f}{}", name, parts[i].bytes / 1e9, i == 0 ? " GB" : ""));
  }
  return out + "\n" + wrap(items);
}

std::string vram_host_line(const std::vector<VramPart>& host) {
  std::string s;
  for (const VramPart& p : host)
    if (p.bytes) s += std::format("{}{} {:.1f} GB", s.empty() ? "host memory: " : ", ", p.name, p.bytes / 1e9);
  return s.empty() ? s : s + " (pinned)\n";
}

}  // namespace runtime

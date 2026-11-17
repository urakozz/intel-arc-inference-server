// The startup log's VRAM chart (runtime/vram_bars.h), on the host: every bar is 64 cells and free is the part that
// stretches, so two cards with different usage line up; a zero part is left out of the bar and the breakdown;
// neighbouring used parts differ in shade; the breakdown wraps at about 90 characters; the ASCII fallback.
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>
#include "check.h"
#include "runtime/vram_bars.h"

namespace {
using runtime::VramCard;
using runtime::VramPart;

constexpr size_t kGB = 1000000000ull;
constexpr size_t kCard = 32530000000ull;

// The code points of a UTF-8 line.
std::vector<std::string> glyphs(const std::string& line) {
  std::vector<std::string> out;
  for (size_t i = 0; i < line.size();) {
    const unsigned char c = line[i];
    const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    out.push_back(line.substr(i, n));
    i += n;
  }
  return out;
}

std::vector<std::string> lines(const std::string& s) {
  std::vector<std::string> out;
  size_t at = 0;
  for (size_t nl; (nl = s.find('\n', at)) != std::string::npos; at = nl + 1) out.push_back(s.substr(at, nl - at));
  return out;
}

VramCard card(size_t kv, size_t drafter) {
  VramCard c;
  c.used = {{"weights", 18200000000ull}, {"KV cache", kv},    {"MTP × 2 slots", 900000000ull},
            {"DFlash drafter", drafter}, {"prefill scratch", 900000000ull}};
  c.reserve = 1500000000ull;
  c.total = kCard;
  return c;
}

void check_card(const VramCard& c, bool utf8) {
  const std::vector<int> cells = runtime::vram_cells(c);
  CHECK_EQ(std::accumulate(cells.begin(), cells.end(), 0), runtime::kVramCells);
  for (int n : cells) CHECK(n >= 1);
  const std::string block = runtime::vram_block(c, 0, 2, utf8);
  std::fputs(block.c_str(), stdout);
  const std::vector<std::string> ls = lines(block);
  CHECK(ls.size() >= 3);
  const std::vector<std::string> bar = glyphs(ls[1]);
  CHECK_EQ(bar.size(), size_t(runtime::kVramCells));
  // Neighbouring cells of different parts never share a glyph.
  size_t at = 0;
  for (size_t i = 0; i + 1 < cells.size(); ++i) {
    at += cells[i];
    CHECK(bar[at - 1] != bar[at]);
  }
  for (size_t i = 2; i < ls.size(); ++i) CHECK(ls[i].size() <= runtime::kVramWrap + 1);
}
}  // namespace

int main() {
  // Two cards, different usage: both bars 64 cells, the reserve the same width on both (free stretched).
  const VramCard a = card(3400000000ull, 2100000000ull), b = card(7300000000ull, 0);
  check_card(a, true);
  check_card(b, true);
  const std::vector<int> ca = runtime::vram_cells(a), cb = runtime::vram_cells(b);
  CHECK_EQ(ca.back(), cb.back());   // reserve: round(1.5 / 32.53 x 64) = 3 on both
  CHECK_EQ(ca.back(), 3);
  CHECK_EQ(glyphs(lines(runtime::vram_block(a, 0, 2, true))[1]).size(),
           glyphs(lines(runtime::vram_block(b, 1, 2, true))[1]).size());

  // The parts sum to the total: used + free + reserve.
  {
    const std::string bd = runtime::vram_block(a, 0, 2, true);
    CHECK(bd.find("free 5.5") != std::string::npos);   // 32.53 - 18.2 - 3.4 - 0.9 - 2.1 - 0.9 - 1.5 = 5.53
  }

  // A zero part (b's drafter) is in neither the bar nor the breakdown: 4 used + free + reserve.
  CHECK_EQ(cb.size(), size_t(6));
  CHECK(runtime::vram_block(b, 1, 2, true).find("drafter") == std::string::npos);
  CHECK(runtime::vram_block(a, 0, 2, true).find("DFlash drafter 2.1") != std::string::npos);

  // Only the first number carries " GB".
  {
    const std::string bd = lines(runtime::vram_block(a, 0, 2, true))[2];
    CHECK(bd.rfind("weights 18.2 GB, KV cache 3.4, ", 0) == 0);
  }

  // Seven neighbours: the 6-shade cycle keeps every pair distinct; wrapping at 90 makes two breakdown lines.
  {
    VramCard c;
    c.total = kCard;
    c.reserve = 1500000000ull;
    for (int i = 0; i < 7; ++i) c.used.push_back({"resident prefill weights " + std::to_string(i), 2 * kGB});
    check_card(c, true);
    CHECK(lines(runtime::vram_block(c, 0, 1, true)).size() >= 4);
  }

  // Rounding pushes the used parts past 64 with no free: the largest used part gives the excess back.
  {
    VramCard c;
    c.total = 64 * kGB;
    c.used = {{"weights", 49600000000ull}};   // 50 cells, and 24 parts of 0.6 cells rounded up to 1: 74
    for (int i = 0; i < 24; ++i) c.used.push_back({"x" + std::to_string(i), 600000000ull});
    check_card(c, true);
    CHECK_EQ(runtime::vram_cells(c)[0], 40);
  }

  // ASCII: # and = alternate, . is free, | the reserve, × becomes x; the bar is 64 bytes.
  {
    const std::string blk = runtime::vram_block(a, 0, 2, false);
    check_card(a, false);
    const std::string bar = lines(blk)[1];
    CHECK_EQ(bar.size(), size_t(runtime::kVramCells));
    CHECK(bar.find_first_not_of("#=.|") == std::string::npos);
    CHECK(bar.front() == '#' && bar.back() == '|' && bar.find('.') != std::string::npos);
    CHECK(blk.find("MTP x 2 slots") != std::string::npos);
    CHECK(blk.find("×") == std::string::npos);
  }

  // The host line: only non-zero parts, empty when nothing is pinned.
  CHECK_EQ(runtime::vram_host_line({{"prefix cache", 0}}), std::string());
  CHECK_EQ(runtime::vram_host_line({{"experts", 16700000000ull}, {"PLE table", 51800000000ull}}),
           std::string("host memory: experts 16.7 GB, PLE table 51.8 GB (pinned)\n"));
  std::puts("vram_bars_test: OK");
  return 0;
}

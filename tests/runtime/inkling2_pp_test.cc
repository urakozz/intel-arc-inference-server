// `--parallel pp` against one card (runtime/inkling2/inkling2_pp_engine.h), bitwise: the same launches on the same
// inputs, the split only moves where they run.
//   inkling2_pp_test SNAPSHOT PROMPT [kv8|int8]
// One card (card 0) first: a long prompt (synthetic ids, then PROMPT's) in 4,096-row chunks, then every cache and
// convolution history, the head's logits and first id; greedy plain ids after it; the same prompt again and drafted
// ids; a short prompt (PROMPT alone) after a reset and its plain ids. Then the two-card engine on cards 0 and 1 does
// the same with the prompt in 4,096- and in 2,048-row chunks (two chunks in flight): every cache (card 0's slots then
// card 1's), the logits and every id must equal one card's at the same chunk width. Then the prompt once more with the prefix cache's hook
// (its chunks end at the snapshot points; under pp each mid-prompt snapshot comes from the cards' shadows while the
// next chunks run): every snapshot's state and KV bytes, and the caches after it, equal one card's. kv8: the int8
// cache; int8: the int8 prompt pass on it.
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "check.h"
#include "runtime/inkling2/inkling2_pp_engine.h"
#include "runtime/inkling2_prefill_test_access.h"

namespace {
namespace ink = runtime::inkling2;
using D = model::Inkling2Desc;
constexpr uint32_t kMaxLen = 8192, kPlain = 48, kDrafted = 64, kShort = 16;

std::vector<uint32_t> read_prompt(const char* path) {
  std::ifstream file(path);
  std::vector<uint32_t> prompt;
  for (uint32_t id; file >> id;) prompt.push_back(id);
  CHECK(file.eof() && !prompt.empty());
  return prompt;
}

struct Run {
  uint32_t first = 0;
  std::vector<float> logits;
  std::vector<std::vector<std::byte>> state;   // each cache tensor, the cards' allocations in card order
  std::vector<uint32_t> plain, drafted, brief;
  std::vector<std::vector<std::byte>> snapshots;   // the hooked prompt's: each block end's state then its KV
  std::vector<std::vector<std::byte>> hooked_state;
};

void download(l0::Context& ctx, const l0::Mem& src, std::vector<std::byte>& dst) {
  auto list = l0::CmdList::immediate(ctx);
  l0::Mem staging(ctx, l0::MemKind::Host, 32ul << 20);
  const size_t at = dst.size();
  dst.resize(at + src.size());
  for (size_t off = 0; off < src.size(); off += staging.size()) {
    const size_t bytes = std::min(staging.size(), src.size() - off);
    list.copy(staging.ptr(), static_cast<const std::byte*>(src.ptr()) + off, bytes);
    std::memcpy(dst.data() + at + off, staging.ptr(), bytes);
  }
}

// Each tensor of PrefillTestAccess::state (and the int8 cache's scales), card after card.
template <class Card>
std::vector<std::vector<std::byte>> state(std::span<ink::Engine* const> cards, Card context) {
  std::vector<std::vector<std::byte>> out;
  for (uint32_t c = 0; c < cards.size(); ++c) {
    auto mems = ink::PrefillTestAccess::state(*cards[c]);
    const auto scales = ink::PrefillTestAccess::scales(*cards[c]);
    std::vector<const l0::Mem*> all(mems.begin(), mems.end());
    all.insert(all.end(), scales.begin(), scales.end());
    if (out.empty()) out.resize(all.size());
    for (size_t i = 0; i < all.size(); ++i)
      if (all[i]) download(context(c), *all[i], out[i]);
  }
  return out;
}

template <class E, class Card>
Run run(E& engine, std::span<ink::Engine* const> cards, Card context, const std::vector<uint32_t>& prompt,
        const std::vector<uint32_t>& brief, uint32_t chunk) {
  Run r;
  engine.reset();
  r.first = engine.prefill(prompt, chunk);
  r.logits.resize(D::vocab_used);
  engine.read_prefill_logits(r.logits);
  r.state = state(cards, context);
  uint32_t next = r.first;
  for (uint32_t i = 0; i < kPlain; ++i) {
    r.plain.push_back(next);
    next = engine.append(next);
  }
  engine.reset();
  next = engine.prefill(prompt, chunk);
  while (r.drafted.size() < kDrafted)
    next = engine.step_drafted(next, std::min(7u, kDrafted - uint32_t(r.drafted.size()) - 1), r.drafted);
  r.drafted.resize(kDrafted);
  engine.reset();   // after a long session, a short one from the start
  next = engine.prefill(brief, chunk);
  for (uint32_t i = 0; i < kShort; ++i) {
    r.brief.push_back(next);
    next = engine.append(next);
  }
  engine.reset();
  uint32_t stored = 0;
  engine.set_block_hook([&](uint32_t end, bool) {
    auto& state = r.snapshots.emplace_back(engine.state_bytes());
    engine.save_state(state.data());
    auto& kv = r.snapshots.emplace_back(engine.kv_bytes(end - stored));
    engine.save_kv(stored, end, kv.data());
    stored = end;
  });
  CHECK_EQ(engine.prefill(prompt, chunk), r.first);
  engine.set_block_hook({});
  r.hooked_state = state(cards, context);
  return r;
}

void same(const Run& one, const Run& pp, const char* what) {
  constexpr std::array names{"full k", "full v", "ring k", "ring v", "conv k", "conv v", "conv attention", "conv MoE",
                             "drafter full k", "drafter full v", "drafter ring k", "drafter ring v", "full k scales",
                             "full v scales"};
  CHECK_EQ(one.state.size(), pp.state.size());
  bool equal = true;
  for (size_t i = 0; i < one.state.size(); ++i) {
    CHECK_EQ(one.state[i].size(), pp.state[i].size());
    const auto miss = std::mismatch(one.state[i].begin(), one.state[i].end(), pp.state[i].begin());
    if (miss.first == one.state[i].end()) continue;
    equal = false;
    size_t bytes = 0;   // how many differ, and the first one's two values
    for (size_t b = 0; b < one.state[i].size(); ++b) bytes += one.state[i][b] != pp.state[i][b];
    std::fprintf(stderr, "%s: %s differs at byte %zu of %zu (0x%02x one card, 0x%02x pp; %zu bytes differ)\n", what,
                 names[i], size_t(miss.first - one.state[i].begin()), one.state[i].size(), unsigned(*miss.first),
                 unsigned(*miss.second), bytes);
  }
  CHECK(equal);
  for (size_t i = 0; i < one.logits.size(); ++i)
    CHECK_EQ(std::bit_cast<uint32_t>(one.logits[i]), std::bit_cast<uint32_t>(pp.logits[i]));
  CHECK_EQ(one.first, pp.first);
  CHECK(one.plain == pp.plain);
  CHECK(one.drafted == pp.drafted);
  CHECK(one.brief == pp.brief);
  CHECK_EQ(one.snapshots.size(), pp.snapshots.size());
  for (size_t i = 0; i < one.snapshots.size(); ++i)
    if (one.snapshots[i] != pp.snapshots[i]) {
      std::fprintf(stderr, "%s: hooked snapshot %zu (%s) differs\n", what, i / 2, i % 2 ? "KV" : "state");
      equal = false;
    }
  CHECK(one.hooked_state == pp.hooked_state);
  CHECK(equal);
  std::printf("%s: every cache, the logits, %u plain, %u drafted and %u short-session ids, %zu hooked snapshots equal "
              "one card's\n", what, kPlain, kDrafted, kShort, pp.snapshots.size() / 2);
}

}  // namespace

int main(int argc, char** argv) {
  // A last argument dflash=DIR (plan 41k): the DFlash 2 drafter in DSpark's place, on card 1 under pp.
  std::string dflash;
  if (argc >= 4 && std::string_view(argv[argc - 1]).starts_with("dflash=")) dflash = argv[--argc] + 7;
  CHECK(argc == 3 || (argc == 4 && (std::string_view(argv[3]) == "kv8" || std::string_view(argv[3]) == "int8")));
  const bool int8_pass = argc == 4 && std::string_view(argv[3]) == "int8";
  const bool kv8 = argc == 4;
  std::setvbuf(stdout, nullptr, _IOLBF, 0);   // each line as it is printed, a crash included
  const loader::Inkling2Checkpoint checkpoint(argv[1], dflash);
  const auto brief = read_prompt(argv[2]);
  std::vector<uint32_t> prompt(7000);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = 20 + (i * 131) % 509;
  prompt.insert(prompt.end(), brief.begin(), brief.end());
  CHECK(prompt.size() + kDrafted + 8 <= kMaxLen);

  // One card at each chunk width, then pp at the same width against it.
  constexpr std::array<uint32_t, 2> widths{ink::kPrefillRows, ink::kPrefillAttentionRows};
  l0::Context card0(0);
  std::array<Run, 2> one;
  {
    ink::Engine engine(card0, checkpoint, kMaxLen, true, ink::kPrefillRows, kv8, int8_pass);
    ink::Engine* cards[] = {&engine};
    for (uint32_t i = 0; i < widths.size(); ++i)
      one[i] = run(engine, cards, [&](uint32_t) -> l0::Context& { return card0; }, prompt, brief, widths[i]);
  }
  std::printf("one card: first id %u, %zu prompt ids; %u- and %u-row chunks %s\n", one[0].first, prompt.size(),
              widths[0], widths[1],
              one[0].state == one[1].state && one[0].plain == one[1].plain ? "agree bitwise" : "DIFFER (one card)");
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards are needed\n");
    return 77;
  }
  l0::Context card1(card0, 1);
  ink::PpEngine pp(card0, card1, checkpoint, kMaxLen, true, ink::kPrefillRows, kv8, int8_pass);
  ink::Engine* cards[] = {&pp.card(0), &pp.card(1)};
  const auto context = [&](uint32_t c) -> l0::Context& { return c ? card1 : card0; };
  for (uint32_t i = 0; i < widths.size(); ++i) {
    const auto label = "pp, " + std::to_string(widths[i]) + "-row chunks";
    same(one[i], run(pp, cards, context, prompt, brief, widths[i]), label.c_str());
  }
}

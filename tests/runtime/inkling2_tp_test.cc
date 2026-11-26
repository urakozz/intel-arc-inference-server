// `--parallel tp` (runtime/inkling2/inkling2_tp_engine.h) against itself, bitwise where the order is kept:
//   inkling2_tp_test SNAPSHOT PROMPT [kv8|int8]
// The prompt (synthetic ids, then PROMPT's) in 2,048-row chunks, two in flight with the all-reduces on the copy engine,
// against the same ids one chunk a prefill call (one chunk in flight): every cache and history on both cards, the
// head's logits and first id; the tensors both cards hold whole equal between the cards after the prompt, after
// plain steps and after drafted steps; then the prompt with the prefix cache's hook (each mid-prompt block end from
// the cards' shadows while the next chunks run), and from each stored block end a restore that prefills the rest:
// every cache and history but the rings (a restore brings back their window's rows only) and the logits equal the
// uncached run. Ids against one card are the f32 judge's (inkling2_decode_test
// ... tp). kv8: the int8 cache; int8: the int8 prompt pass on it. Two cards; skipped (77) with one.
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "check.h"
#include "runtime/inkling2/inkling2_tp_engine.h"
#include "runtime/inkling2_prefill_test_access.h"

namespace {
namespace ink = runtime::inkling2;
using D = model::Inkling2Desc;
using Access = ink::PrefillTestAccess;
constexpr uint32_t kMaxLen = 12288, kChunk = 2048, kPlain = 32, kDrafted = 48;
// PrefillTestAccess::state's order: the cards' halves, the tensors both hold whole, the drafter's caches (the cards'
// heads, its layers split).
constexpr std::array kNames{"full k", "full v", "ring k", "ring v", "conv k", "conv v", "conv attention", "conv MoE",
                            "drafter full k", "drafter full v", "drafter ring k", "drafter ring v"};
constexpr bool kWhole[] = {false, false, false, false, false, false, true, true, false, false, false, false};

std::vector<uint32_t> read_prompt(const char* path) {
  std::ifstream file(path);
  std::vector<uint32_t> prompt;
  for (uint32_t id; file >> id;) prompt.push_back(id);
  CHECK(file.eof() && !prompt.empty());
  return prompt;
}

std::vector<std::byte> download(l0::Context& ctx, const l0::Mem* src) {
  std::vector<std::byte> out(src ? src->size() : 0);
  if (src) l0::CmdList::immediate(ctx).copy(out.data(), src->ptr(), out.size());
  return out;
}

// Each card's state tensors (and the int8 scales): [card][tensor].
using State = std::array<std::vector<std::vector<std::byte>>, 2>;
State state(ink::TpEngine& tp) {
  State out;
  for (uint32_t c = 0; c < 2; ++c) {
    auto& e = tp.card(c);
    l0::Context& ctx = *Access::context(e);
    for (const auto* m : Access::state(e)) out[c].push_back(download(ctx, m));
    for (const auto* m : Access::scales(e)) out[c].push_back(download(ctx, m));
  }
  return out;
}

std::vector<float> logits(ink::TpEngine& tp) {
  std::vector<float> out(D::vocab_used);
  tp.read_prefill_logits(out);
  return out;
}

// The tensors both cards hold whole are the same bytes on both.
void replicated(const State& s, const char* when) {
  for (size_t i = 0; i < std::size(kWhole); ++i)
    if (kWhole[i] && s[0][i] != s[1][i]) {
      std::fprintf(stderr, "%s: %s differs between the cards\n", when, kNames[i]);
      CHECK(false);
    }
  std::printf("%s: the replicated tensors equal on both cards\n", when);
}

// rings: false after a restore - a restore brings back the window's rows, not the older ones a ring still holds.
void same(const State& a, const State& b, const std::vector<float>& la, const std::vector<float>& lb, const char* what,
          bool rings = true) {
  bool equal = true;
  for (uint32_t c = 0; c < 2; ++c)
    for (size_t i = 0; i < a[c].size(); ++i) {
      if (!rings && (i == 2 || i == 3 || i == 10 || i == 11)) continue;
      const auto miss = std::mismatch(a[c][i].begin(), a[c][i].end(), b[c][i].begin());
      if (miss.first == a[c][i].end()) continue;
      equal = false;
      std::fprintf(stderr, "%s: card %u's %s differs at byte %zu of %zu\n", what, c,
                   i < kNames.size() ? kNames[i] : "int8 scales", size_t(miss.first - a[c][i].begin()), a[c][i].size());
    }
  for (size_t i = 0; i < la.size(); ++i) equal = equal && std::bit_cast<uint32_t>(la[i]) == std::bit_cast<uint32_t>(lb[i]);
  CHECK(equal);
  std::printf("%s: every cache, history and logit equal\n", what);
}
}  // namespace

int main(int argc, char** argv) {
  // A last argument dflash=DIR (plan 41k): the DFlash 2 drafter in DSpark's place, whole on both cards.
  std::string dflash;
  if (argc >= 4 && std::string_view(argv[argc - 1]).starts_with("dflash=")) dflash = argv[--argc] + 7;
  CHECK(argc == 3 || (argc == 4 && (std::string_view(argv[3]) == "kv8" || std::string_view(argv[3]) == "int8")));
  const bool int8_pass = argc == 4 && std::string_view(argv[3]) == "int8";
  const bool kv8 = argc == 4;
  std::setvbuf(stdout, nullptr, _IOLBF, 0);   // each line as it is printed, a crash included
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards are needed\n");
    return 77;
  }
  const loader::Inkling2Checkpoint checkpoint(argv[1], dflash);
  const auto brief = read_prompt(argv[2]);
  std::vector<uint32_t> prompt(9000);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = 20 + (i * 131) % 509;
  prompt.insert(prompt.end(), brief.begin(), brief.end());
  CHECK(prompt.size() + kDrafted + 8 <= kMaxLen);

  l0::Context card0(0);
  l0::Context card1(card0, 1);
  ink::TpEngine tp(card0, card1, checkpoint, kMaxLen, true, kChunk, kv8, int8_pass);
  std::printf("tp: %zu prompt ids, %u-row chunks\n", prompt.size(), kChunk);

  // Two chunks in flight against one chunk a call.
  tp.reset();
  const uint32_t first = tp.prefill(prompt, kChunk);
  const auto overlapped = state(tp);
  const auto top = logits(tp);
  replicated(overlapped, "after the prompt");
  tp.reset();
  uint32_t alone = 0;
  for (size_t at = 0; at < prompt.size(); at += kChunk)
    alone = tp.prefill({prompt.data() + at, std::min<size_t>(kChunk, prompt.size() - at)}, kChunk);
  CHECK_EQ(alone, first);
  same(overlapped, state(tp), top, logits(tp), "two chunks in flight against one at a time");

  // Plain then drafted steps: the cards stay equal (TpEngine also checks every step's ids on both).
  uint32_t next = first;
  std::vector<uint32_t> plain;
  for (uint32_t i = 0; i < kPlain; ++i) {
    plain.push_back(next);
    next = tp.append(next);
  }
  replicated(state(tp), "after plain steps");
  tp.reset();
  next = tp.prefill(prompt, kChunk);
  std::vector<uint32_t> drafted;
  while (drafted.size() < kDrafted)
    next = tp.step_drafted(next, std::min(7u, kDrafted - uint32_t(drafted.size()) - 1), drafted);
  replicated(state(tp), "after drafted steps");
  uint32_t agree = 0;
  while (agree < kPlain && plain[agree] == drafted[agree]) ++agree;
  std::printf("drafted ids agree with plain ids for the first %u of %u\n", agree, kPlain);

  // The prefix cache: each mid-prompt block end stored from the shadows, then restored and the rest prefilled.
  struct Stored {
    uint32_t end;
    std::vector<std::byte> state, kv;
  };
  std::vector<Stored> stored;
  tp.reset();
  tp.set_block_hook([&](uint32_t end, bool block_end) {
    if (!block_end || end == prompt.size()) return;
    auto& s = stored.emplace_back(Stored{end, std::vector<std::byte>(tp.state_bytes()), {}});
    tp.save_state(s.state.data());
    s.kv.resize(tp.kv_bytes(end));
    tp.save_kv(0, end, s.kv.data());
  });
  CHECK_EQ(tp.prefill(prompt, kChunk), first);
  tp.set_block_hook({});
  same(overlapped, state(tp), top, logits(tp), "the hooked prompt");
  CHECK(stored.size() >= 2);
  for (const auto& s : stored) {
    tp.reset();
    tp.load_state(s.state.data(), s.end);
    tp.load_kv(0, s.end, s.kv.data());
    CHECK_EQ(tp.prefill({prompt.data() + s.end, prompt.size() - s.end}, kChunk), first);
    const auto label = "restored at " + std::to_string(s.end) + ", the rest prefilled";
    same(overlapped, state(tp), top, logits(tp), label.c_str(), false);
  }
  std::printf("inkling2_tp_test: PASS\n");
}

// Plan 41h (both cards): the intermittent pp mismatch at 2,048-row chunks (one convolution-history element of visit
// 37), cornered. inkling2_pp_test's long prompt (7,000 synthetic ids, then PROMPT's) is prefilled `rounds` times in
// each arm and every cache and convolution history is compared byte for byte with one card's (card 0, 2,048-row
// chunks); each differing tensor prints its first byte, both values and the count of differing bytes.
//   probe_ink_pp_race SNAPSHOT PROMPT [rounds=10] [kv8]
// Arms: one card on card 1 (the same walk on the other card), pp at 2,048-row chunks (two chunks in flight), pp one
// chunk a prefill() call (the same chunks, never two in flight), pp at 4,096-row chunks against one card's at 4,096;
// then pp at 2,048 after each part of inkling2_pp_test's 4,096-row session (its order: 48 plain ids, the drafted ids,
// the short session, the prefix cache's hooked prompt) and after all of them.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#include "runtime/inkling2/inkling2_pp_engine.h"
#include "runtime/inkling2_prefill_test_access.h"

namespace {
namespace ink = runtime::inkling2;
constexpr uint32_t kMaxLen = 8192;
using State = std::vector<std::vector<std::byte>>;

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

// inkling2_pp_test's state(): each tensor, the cards' allocations in card order.
State state(std::vector<std::pair<ink::Engine*, l0::Context*>> cards) {
  State out;
  for (auto [engine, ctx] : cards) {
    auto mems = ink::PrefillTestAccess::state(*engine);
    const auto scales = ink::PrefillTestAccess::scales(*engine);
    std::vector<const l0::Mem*> all(mems.begin(), mems.end());
    all.insert(all.end(), scales.begin(), scales.end());
    if (out.empty()) out.resize(all.size());
    for (size_t i = 0; i < all.size(); ++i)
      if (all[i]) download(*ctx, *all[i], out[i]);
  }
  return out;
}

constexpr const char* kNames[] = {"full k", "full v", "ring k", "ring v", "conv k", "conv v", "conv attention",
                                  "conv MoE", "drafter full k", "drafter full v", "drafter ring k", "drafter ring v",
                                  "full k scales", "full v scales"};

bool compare(const State& want, const State& got, const std::string& what) {
  bool same = true;
  for (size_t i = 0; i < want.size(); ++i) {
    const auto miss = std::mismatch(want[i].begin(), want[i].end(), got[i].begin());
    if (miss.first == want[i].end()) continue;
    same = false;
    size_t bytes = 0;
    for (size_t b = 0; b < want[i].size(); ++b) bytes += want[i][b] != got[i][b];
    const size_t at = size_t(miss.first - want[i].begin());
    float a = 0, b = 0;
    std::memcpy(&a, want[i].data() + at / 4 * 4, 4);
    std::memcpy(&b, got[i].data() + at / 4 * 4, 4);
    std::printf("  %s: %s differs at byte %zu (word %zu: %.9g one card, %.9g here; %zu bytes differ)\n", what.c_str(),
                kNames[i], at, at / 4, a, b, bytes);
  }
  return same;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: probe_ink_pp_race SNAPSHOT PROMPT [rounds] [kv8]\n");
    return 2;
  }
  const uint32_t rounds = argc > 3 ? uint32_t(std::atoi(argv[3])) : 10;
  const bool kv8 = argc > 4 && std::string(argv[4]) == "kv8";
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  std::vector<uint32_t> prompt(7000);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = 20 + (i * 131) % 509;
  {
    std::ifstream file(argv[2]);
    for (uint32_t id; file >> id;) prompt.push_back(id);
  }
  l0::Context card0(0);
  l0::Context card1(card0, 1);
  State want2048, want4096;
  {
    ink::Engine one(card0, checkpoint, kMaxLen, true, ink::kPrefillRows, kv8);
    one.reset();
    one.prefill(prompt, 4096);
    want4096 = state({{&one, &card0}});
    one.reset();
    one.prefill(prompt, 2048);
    want2048 = state({{&one, &card0}});
  }
  std::printf("one card (card 0): %zu prompt ids, the reference at 2,048 and 4,096-row chunks\n", prompt.size());
  const auto arm = [&](const std::string& what, const State& want, const std::function<State()>& run) {
    uint32_t bad = 0;
    for (uint32_t r = 0; r < rounds; ++r) bad += !compare(want, run(), what + " round " + std::to_string(r));
    std::printf("%s: %u of %u rounds differ\n", what.c_str(), bad, rounds);
  };
  {
    ink::Engine one(card1, checkpoint, kMaxLen, true, ink::kPrefillRows, kv8);
    arm("one card on card 1, 2,048", want2048, [&] {
      one.reset();
      one.prefill(prompt, 2048);
      return state({{&one, &card1}});
    });
  }
  ink::PpEngine pp(card0, card1, checkpoint, kMaxLen, true, ink::kPrefillRows, kv8);
  const auto pp_state = [&] { return state({{&pp.card(0), &card0}, {&pp.card(1), &card1}}); };
  arm("pp 2,048, two in flight", want2048, [&] {
    pp.reset();
    pp.prefill(prompt, 2048);
    return pp_state();
  });
  arm("pp 2,048, one chunk a call", want2048, [&] {
    pp.reset();
    for (size_t at = 0; at < prompt.size(); at += 2048)
      pp.prefill(std::span(prompt).subspan(at, std::min<size_t>(2048, prompt.size() - at)), 2048);
    return pp_state();
  });
  arm("pp 4,096, two in flight", want4096, [&] {
    pp.reset();
    pp.prefill(prompt, 4096);
    return pp_state();
  });
  const std::vector<uint32_t> brief(prompt.begin() + 7000, prompt.end());
  const auto plain = [&] {
    pp.reset();
    uint32_t next = pp.prefill(prompt, 4096);
    for (uint32_t i = 0; i < 48; ++i) next = pp.append(next);
  };
  const auto drafted = [&] {
    pp.reset();
    uint32_t next = pp.prefill(prompt, 4096);
    std::vector<uint32_t> out;
    while (out.size() < 64) next = pp.step_drafted(next, std::min<uint32_t>(7, 63 - uint32_t(out.size())), out);
  };
  const auto short_session = [&] {
    pp.reset();
    uint32_t next = pp.prefill(brief, 4096);
    for (uint32_t i = 0; i < 16; ++i) next = pp.append(next);
  };
  const auto hooked = [&] {
    pp.reset();
    uint32_t stored = 0;
    std::vector<std::byte> host;
    pp.set_block_hook([&](uint32_t end, bool) {
      host.resize(pp.state_bytes());
      pp.save_state(host.data());
      host.resize(pp.kv_bytes(end - stored));
      pp.save_kv(stored, end, host.data());
      stored = end;
    });
    pp.prefill(prompt, 4096);
    pp.set_block_hook({});
  };
  const auto after = [&](const std::string& what, const std::function<void()>& before) {
    arm("pp 2,048 after " + what, want2048, [&] {
      before();
      pp.reset();
      pp.prefill(prompt, 2048);
      return pp_state();
    });
  };
  // The hooked prompt's host writes, guarded: each save_state / save_kv into a fresh buffer with 4 MiB of 0xA5 before
  // and after it; the guards checked when the call returns, after the prompt and 200 ms later (a late or stray write).
  for (uint32_t r = 0; r < rounds; ++r) {
    constexpr size_t kGuard = 4u << 20;
    std::vector<std::vector<std::byte>> saved;
    std::vector<std::string> names;
    const auto guards_ok = [&](const char* when) {
      for (size_t i = 0; i < saved.size(); ++i) {
        const auto& b = saved[i];
        size_t bad = 0, first = 0;
        for (size_t j = 0; j < kGuard; ++j)
          for (const size_t at : {j, b.size() - kGuard + j})
            if (b[at] != std::byte{0xA5} && !bad++) first = at;
        if (bad)
          std::printf("  hooked round %u: %s's guard has %zu bytes written %s (first at %zd from the data)\n", r,
                      names[i].c_str(), bad, when, first < kGuard ? ptrdiff_t(first) - ptrdiff_t(kGuard)
                                                                   : ptrdiff_t(first - (b.size() - kGuard)));
      }
    };
    pp.reset();
    uint32_t stored = 0;
    pp.set_block_hook([&](uint32_t end, bool) {
      for (const bool kv : {false, true}) {
        const size_t bytes = kv ? pp.kv_bytes(end - stored) : pp.state_bytes();
        auto& b = saved.emplace_back(bytes + 2 * kGuard, std::byte{0xA5});
        names.push_back((kv ? "save_kv to " : "save_state at ") + std::to_string(end));
        if (kv) pp.save_kv(stored, end, b.data() + kGuard);
        else pp.save_state(b.data() + kGuard);
      }
      stored = end;
      guards_ok("by the call's return");
    });
    pp.prefill(prompt, 4096);
    pp.set_block_hook({});
    guards_ok("after the prompt");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    guards_ok("200 ms after the prompt");
  }
  std::printf("hooked prompt guards checked, %u rounds\n", rounds);
  after("plain ids", plain);
  after("drafted ids", drafted);
  after("the short session", short_session);
  after("the hooked prompt", hooked);
  after("the whole session", [&] {
    plain();
    drafted();
    short_session();
    hooked();
  });
  return 0;
}

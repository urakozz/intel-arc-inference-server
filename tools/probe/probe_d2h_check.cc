// Plan 41h (both cards): do a card's copies into host memory land before the host sees them done? Each card's device
// buffer holds a counter pattern that changes every round; the round copies it to host memory - an immediate list
// (synchronous), or an asynchronous immediate list and its host_synchronize - into pinned host USM (one allocation
// reused, as a staging buffer is) or into plain heap memory (a fresh std::vector each round), and compares every
// byte at once. Card 1 is a view of card 0's context (l0::Context(card0, 1)), as `--parallel pp` runs it.
//   probe_d2h_check [rounds=200] [MiB=8]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"

int main(int argc, char** argv) {
  const uint32_t rounds = argc > 1 ? uint32_t(std::atoi(argv[1])) : 200;
  const size_t bytes = (argc > 2 ? size_t(std::atoi(argv[2])) : 8) << 20, words = bytes / 4;
  if (l0::Context::gpu_count() < 2) {
    std::fprintf(stderr, "probe_d2h_check: needs both cards\n");
    return 77;
  }
  l0::Context card0(0);
  l0::Context card1(card0, 1);
  std::vector<uint32_t> pattern(words);
  for (uint32_t c = 0; c < 2; ++c) {
    l0::Context& ctx = c ? card1 : card0;
    l0::Mem dev(ctx, l0::MemKind::Device, bytes);
    l0::Mem pinned(ctx, l0::MemKind::Host, bytes);
    auto up = l0::CmdList::immediate(ctx);
    for (uint32_t form = 0; form < 4; ++form) {   // {sync, async} x {pinned, heap}
      const bool async = form & 1, heap = form & 2;
      size_t bad_rounds = 0, bad_words = 0;
      for (uint32_t r = 0; r < rounds; ++r) {
        for (size_t i = 0; i < words; ++i) pattern[i] = uint32_t(i * 2654435761u) ^ (r * 0x9E3779B9u + c);
        up.copy(dev.ptr(), pattern.data(), bytes);
        std::vector<uint32_t> fresh;
        uint32_t* host = static_cast<uint32_t*>(pinned.ptr());
        if (heap) {
          fresh.assign(words, 0);
          host = fresh.data();
        } else {
          std::memset(host, 0, bytes);
        }
        if (async) {
          auto list = l0::CmdList::immediate_async(ctx);
          list.copy(host, dev.ptr(), bytes);
          list.host_synchronize();
        } else {
          auto list = l0::CmdList::immediate(ctx);
          list.copy(host, dev.ptr(), bytes);
        }
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < words; ++i)
          if (host[i] != pattern[i] && !bad++) first = i;
        if (bad) {
          if (bad_rounds < 5)
            std::printf("card %u %s %s round %u: %zu words differ, first at byte %zu (0x%08x, want 0x%08x)\n", c,
                        async ? "async" : "sync", heap ? "heap" : "pinned", r, bad, first * 4, host[first],
                        pattern[first]);
          ++bad_rounds;
          bad_words += bad;
        }
      }
      std::printf("card %u, %s list into %s memory: %zu of %u rounds differ (%zu words)\n", c,
                  async ? "an asynchronous" : "a synchronous", heap ? "heap" : "pinned", bad_rounds, rounds, bad_words);
    }
  }
  return 0;
}

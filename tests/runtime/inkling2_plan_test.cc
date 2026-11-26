// Plan 41f Task 3: Inkling 2 Mini's served context on one card - cli::inkling2::served_max_len, what b70-serve plans
// with --max-len auto (one card, and on each card under replica) from the checkpoint's headers, no device:
//   inkling2_plan_test <snapshot>
// On a 32 GB card (b70-decode --plan's capacity) with the prompt pass's 4096-row chunks (every served request
// prefills): with the drafter 248,064 positions in bf16 and 436,480 with the int8 cache, without it 308,736 / 608,000;
// b70-decode --plan's 302,080 / 531,456 are the same plan without the prompt pass's scratch; every plan fits and the
// next 256 positions do not; an explicit --max-len that does not fit is refused naming what fits (the int8 cache's reach too).
// Plan 41k: with a second argument (the DFlash 2 drafter's folder) the same plans with that drafter in DSpark's place -
// its weights, five 2048-position rings and no per-position cache - printed against DSpark's; they hold more positions.
#include <cstdio>
#include <string>

#include "check.h"
#include "cli/inkling2_decode.h"

int main(int argc, char** argv) {
  CHECK(argc == 2 || argc == 3);
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  const auto& d = checkpoint.desc();
  const auto& w = checkpoint.layout();
  namespace ink = runtime::inkling2;
  constexpr size_t capacity = 32'000'000'000ull;
  constexpr size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  constexpr uint32_t chunk = ink::kPrefillRows;
  const cli::MaxLenArg automatic{true, 0};
  uint32_t got[2][2] = {};
  for (const bool draft : {true, false}) {
    for (const bool kv8 : {false, true}) {
      const uint32_t len = cli::inkling2::served_max_len(d, w, capacity, reserve, automatic, draft, chunk, kv8);
      CHECK(ink::sizes(d, w, len, draft, chunk, kv8).memory().total() <= capacity - reserve);
      CHECK(ink::sizes(d, w, len + 256, draft, chunk, kv8).memory().total() > capacity - reserve);
      got[draft][kv8] = len;
      std::printf("one card, drafter %s, %s cache: %u positions\n", draft ? "on" : "off", kv8 ? "int8" : "bf16", len);
    }
  }
  CHECK_EQ(got[1][0], 243456u);
  CHECK_EQ(got[1][1], 428544u);
  CHECK_EQ(got[0][0], 303872u);
  CHECK_EQ(got[0][1], 598528u);
  std::printf("b70-decode --plan (no prompt pass): %u bf16, %u int8\n",
              ink::max_len_that_fits(d, w, capacity, reserve, true, 0, false),
              ink::max_len_that_fits(d, w, capacity, reserve, true, 0, true));
  // An explicit length: what fits is served as asked, what does not is refused with the reach of each cache.
  CHECK_EQ(cli::inkling2::served_max_len(d, w, capacity, reserve, {false, 65536}, true, chunk, false), 65536u);
  std::string why;
  try {
    (void)cli::inkling2::served_max_len(d, w, capacity, reserve, {false, 400000}, true, chunk, false);
  } catch (const std::runtime_error& e) {
    why = e.what();
  }
  std::printf("refusal: %s\n", why.c_str());
  CHECK(why.find("Add --kv-cache int8 (it holds up to " + std::to_string(got[1][1]) + " positions)") != std::string::npos);
  CHECK(why.find("--max-len " + std::to_string(got[1][0]) + " or less") != std::string::npos);
  if (argc == 3) {
    const loader::Inkling2Checkpoint dflash(argv[1], argv[2]);
    for (const bool kv8 : {false, true}) {
      const uint32_t len = cli::inkling2::served_max_len(dflash.desc(), dflash.layout(), capacity, reserve, automatic,
                                                         true, chunk, kv8);
      std::printf("one card, DFlash 2 drafter (%.2f GB), %s cache: %u positions (DSpark's %u, %+d)\n",
                  dflash.layout().bytes[loader::Inkling2WeightLayout::kDraft] / 1e9, kv8 ? "int8" : "bf16", len,
                  got[1][kv8], int(len) - int(got[1][kv8]));
      CHECK(len > got[1][kv8]);
    }
  }
  std::printf("inkling2_plan_test OK\n");
  return 0;
}

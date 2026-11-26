// Plan 41h (both cards): the host's access to a sync event after the immediate list that signalled it is gone. A
// pool of each kind (flags x the devices it is made for) is created, its events signalled once by each card's
// immediate list, 512 MiB of plain heap filled with 0xA5 (1 MiB chunks), then every event reset, signalled, waited on,
// queried and reset again from the host; any heap byte changed is a write by the driver into memory the process owns.
// On compute-runtime 26.40 the IPC and kernel-timestamp kinds change a heap byte from the third round on when the
// lists were destroyed before the host's accesses (kinds 0, 1, 3); kept lists (kind 5) or a reset while the lists
// still exist (kind 6) stay clean; plain host-visible events (kind 2) turn counter-based and stay clean.
//   probe_event_pool_check [rounds=4] [kind]   (kind 0..6, 4 unused: one kind only, so a fault in one does not hide
//                                               the rest; each step is printed before it runs)
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <level_zero/ze_api.h>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/error.h"

int main(int argc, char** argv) {
  const uint32_t rounds = argc > 1 ? uint32_t(std::atoi(argv[1])) : 4;
  const int only = argc > 2 ? std::atoi(argv[2]) : -1;
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (l0::Context::gpu_count() < 2) return 77;
  l0::Context card0(0);
  l0::Context card1(card0, 1);
  struct Kind {
    const char* name;
    ze_event_pool_flags_t flags;
    bool both, keep_lists = false, reset_early = false;
  };
  const Kind kinds[] = {
      {"host-visible | IPC, two devices", ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC, true},
      {"host-visible | IPC, card 0 only", ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC, false},
      {"host-visible, two devices", ZE_EVENT_POOL_FLAG_HOST_VISIBLE, true},
      {"host-visible | kernel timestamp, two devices",
       ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP, true},
      {"(unused)", 0, false},
      {"host-visible | IPC, two devices, lists kept", ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC, true, true},
      {"host-visible | IPC, two devices, reset early", ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC, true,
       false, true},
  };
  constexpr uint32_t kEvents = 64;
  constexpr size_t kChunk = 1u << 20, kChunks = 512;
  std::array<l0::CmdList, 2> kept{l0::CmdList::immediate_async(card0), l0::CmdList::immediate_async(card1)};
  for (int k = 0; k < 7; ++k)
    for (uint32_t r = 0; r < rounds && (only < 0 || only == k) && k != 4; ++r) {
      const Kind& kind = kinds[k];
      const auto step = [&](const char* what) { std::printf("%s round %u: %s\n", kind.name, r, what); };
      std::vector<ze_device_handle_t> devs{card0.device()};
      if (kind.both) devs.push_back(card1.device());
      ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, kind.flags, kEvents};
      ze_event_pool_handle_t pool = nullptr;
      ZE_CHECK(zeEventPoolCreate(card0.handle(), &pd, uint32_t(devs.size()), devs.data(), &pool));
      std::vector<ze_event_handle_t> ev(kEvents);
      for (uint32_t i = 0; i < kEvents; ++i) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
        ZE_CHECK(zeEventCreate(pool, &ed, &ev[i]));
      }
      step("the cards signal");
      for (uint32_t c = 0; c < (kind.both ? 2u : 1u); ++c) {   // each card signals its half once, as the lists do
        auto own = l0::CmdList::immediate_async(c ? card1 : card0);
        auto& list = kind.keep_lists ? kept[c] : own;
        for (uint32_t i = c; i < kEvents; i += 2) list.barrier_signal(ev[i]);
        list.host_synchronize();
        if (kind.reset_early)
          for (uint32_t i = c; i < kEvents; i += 2) zeEventHostReset(ev[i]);
      }
      std::vector<std::unique_ptr<unsigned char[]>> heap;
      for (size_t i = 0; i < kChunks; ++i) {
        heap.emplace_back(new unsigned char[kChunk]);
        std::memset(heap.back().get(), 0xA5, kChunk);
      }
      step("host reset");
      for (auto e : ev) zeEventHostReset(e);
      step("host signal");
      for (auto e : ev) zeEventHostSignal(e);
      step("host wait, query");
      for (auto e : ev) zeEventHostSynchronize(e, 1000000);
      for (auto e : ev) zeEventQueryStatus(e);
      for (auto e : ev) zeEventHostReset(e);
      size_t bad = 0, chunks = 0;
      for (const auto& h : heap) {
        size_t here = 0;
        for (size_t i = 0; i < kChunk; ++i)
          if (h[i] != 0xA5 && !here++)
            std::printf("  byte %zu of a chunk is 0x%02x (chunk at %p)\n", i, h[i], static_cast<void*>(h.get()));
        bad += here;
        chunks += here != 0;
      }
      std::printf("%-46s round %u: %zu heap bytes changed in %zu chunks\n", kind.name, r, bad, chunks);
      for (auto e : ev) zeEventDestroy(e);
      zeEventPoolDestroy(pool);
    }
  return 0;
}

// probe_tp - spec 17a (plan 17a): the tensor-parallel P0 on two B70s, Qwen3.8 first (spec 17 §R2).
//
//   probe_tp facts                                   both cards: name, PCI address and max link, memory, peer access
//                                                    and P2P properties both ways, whether a device allocation can be
//                                                    exported (dma-buf) and mapped by the CPU through the BAR
//   probe_tp bar  [--card 0] (one card) [--words 16,5120,40960] [--iters 2000] [--paths dmabuf,host,shared] [--ack host,dmabuf]
//                                                    the CPU into card memory: per path, a persistent one-work-group
//                                                    kernel (tp_ping) waits for the CPU's row + stamp and acks; the
//                                                    CPU times the round trip (median, p10 / p90) and its own store
//                                                    time; the row's words checked on the card
//   probe_tp ar   [--model qwen38|agnes] [--route all|p2p-push,p2p-pull,host-mbox,host-bar,host-read]
//                 [--rows 1,8] [--calls N] [--rounds 5] [--execs 20] [--replays N] [--inject skip-data|skip-all]
//                 [--inject-at N] [--cpu C] [--chunk 1024] [--s 4] [--flag plain|atomic]
//                                                    the all-reduce (fold-then-send) inside one replayed list per card:
//                                                    `calls` calls a list (default: the model's all-reduces a token,
//                                                    128 / 144), each call tp_send + tp_wait; timed against the same
//                                                    lists without the crossing (interleaved pairs after a warm-up,
//                                                    median of `rounds`); with --replays the check lists instead:
//                                                    every output word against the host formula on both cards, the
//                                                    two cards' output hashes equal, stamps; with --inject a stale
//                                                    (skip-data) or dropped (skip-all) write by card 1 must be caught
//   probe_tp step [--model qwen38|agnes] [--card 0] [--rounds 5] [--launches 40] [--shape all|qkvz,...]
//                                                    one card: each linear at its one-card shape (the production
//                                                    binary) against its per-card TP shape (spec 17 §R2.5, ptp_*
//                                                    binaries from the production sources and cells), M = 1,
//                                                    interleaved; the per-token GEMV sum both ways
//
// Every wait is bounded: a fence that does not signal within --timeout seconds prints the arm and exits 1; every
// kernel spin is bounded and records why it stopped (tp_protocol.h's state words). Exit 1 on any failure.
//
// Routes (spec 17 §R2.6; tp_protocol.h has the protocol):
//   p2p-push   each card's tp_send writes its row into the PEER's landing buffer (a device allocation on the peer)
//   p2p-pull   each card writes its own exposed buffer; the peer's tp_wait reads it over PCIe
//   host-mbox  each card writes a pinned host mailbox; the peer reads it (no CPU in the loop) - recorded, not gating
//   host-bar   (H) cards write pinned host mailboxes; a CPU thread on one pinned core sums in the fixed order and
//              writes the sum into each card's OWN VRAM through the BAR (an exported device allocation, mmapped)
//   host-read  (H') as host-bar, but the CPU writes the sum into one pinned host buffer both cards read
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <time.h>

#include <algorithm>
#include <cerrno>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/error.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "tp_protocol.h"
#include "tp_shapes.h"

namespace {

using Clock = std::chrono::steady_clock;
namespace tpp = tp_protocol;

double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}
double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = size_t(p * double(v.size() - 1) + 0.5);
  return v[std::min(i, v.size() - 1)];
}
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) out.push_back(t);
  return out;
}
std::vector<uint32_t> split_u(const std::string& s) {
  std::vector<uint32_t> v;
  for (const std::string& t : split(s)) v.push_back(uint32_t(std::stoul(t)));
  return v;
}
uint32_t xorshift(uint32_t& st) {
  st ^= st << 13;
  st ^= st >> 17;
  st ^= st << 5;
  return st;
}

// The CPU's store fence before a flag: WC (BAR) stores are not ordered by x86's TSO - an sfence is.
void cpu_store_fence() {
#if defined(__x86_64__)
  _mm_sfence();
#endif
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}
void cpu_relax() {
#if defined(__x86_64__)
  _mm_pause();
#endif
}
double thread_cpu_s() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
}
void pin_to_core(int core) {
#if defined(__linux__)
  if (core < 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (sched_setaffinity(0, sizeof set, &set) != 0) std::fprintf(stderr, "warning: could not pin to core %d\n", core);
#else
  (void)core;
#endif
}

struct Opts {
  std::string mode;
  std::string model = "qwen38";
  std::vector<std::string> routes = {"p2p-push", "p2p-pull", "host-mbox", "host-bar", "host-read"};
  std::vector<uint32_t> rows = {1, 8};
  uint32_t calls = 0;          // 0: the model's all-reduces a token
  uint32_t rounds = 5, execs = 20, replays = 0;
  uint32_t chunk = 1024, S = 4;
  uint32_t spin_limit = 1u << 24;
  uint32_t flag_mode = 1;      // 0: system-scope atomic store; 1: plain store + release fence (probe_tp.cl tp_publish)
  std::string inject = "none";
  uint32_t inject_at = 0;
  int cpu = -1;
  uint32_t card = 0;
  std::vector<uint32_t> words = {16, 5120, 40960};
  uint32_t iters = 2000;
  std::vector<std::string> paths = {"dmabuf", "host", "shared"};
  std::vector<std::string> acks = {"host", "dmabuf"};
  uint32_t launches = 40;
  std::vector<std::string> shapes;
  double timeout_s = 60;
};

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "probe_tp: %s\n", m.c_str());
  std::exit(1);
}

void wait_fence(l0::Fence& f, const Opts& o, const std::string& arm) {
  if (!f.wait_for(uint64_t(o.timeout_s * 1e9))) {
    std::printf("TIMEOUT: %s - a fence did not signal within %.0f s\n", arm.c_str(), o.timeout_s);
    std::fflush(stdout);
    std::_Exit(1);
  }
}

// --- the two cards -------------------------------------------------------------------------------------------------
struct Card {
  std::unique_ptr<l0::Context> owned;   // card 0 owns the context; card 1 is a view in it (spec 16 decision 1)
  l0::Context* ctx = nullptr;
  std::unique_ptr<l0::Queue> q;
  std::unique_ptr<l0::Fence> f;
  std::unique_ptr<l0::Module> mod;
};
void open_cards(Card (&c)[2], bool kernels_too) {
  if (l0::Context::gpu_count() < 2) die("two GPUs needed, " + std::to_string(l0::Context::gpu_count()) + " visible");
  c[0].owned = std::make_unique<l0::Context>(0u);
  c[0].ctx = c[0].owned.get();
  c[1].owned = std::make_unique<l0::Context>(*c[0].ctx, 1u);
  c[1].ctx = c[1].owned.get();
  for (Card& x : c) {
    x.q = std::make_unique<l0::Queue>(*x.ctx);
    x.f = std::make_unique<l0::Fence>(*x.q);
    if (kernels_too) x.mod = std::make_unique<l0::Module>(*x.ctx, kernels::path("ptp_kernels"));
  }
}

// --- device memory the CPU maps through the BAR ----------------------------------------------------------------------
// Candidate (spec 17 §R2.6): a device allocation exported as a dma-buf, the fd mmapped by the CPU. Its pages are VRAM;
// with a 32 GB BAR every page is CPU-visible, so the CPU's stores go over PCIe straight into the card's memory.
struct BarMem {
  ze_context_handle_t ctx = nullptr;
  void* dev = nullptr;
  void* cpu = nullptr;
  size_t bytes = 0;
  int fd = -1;
  std::string err;
  BarMem() = default;
  BarMem(const BarMem&) = delete;
  BarMem& operator=(const BarMem&) = delete;
  ~BarMem() {
    if (cpu && cpu != MAP_FAILED) munmap(cpu, bytes);
    if (dev) zeMemFree(ctx, dev);
  }
};
std::unique_ptr<BarMem> bar_alloc(l0::Context& ctx, size_t bytes) {
  auto b = std::make_unique<BarMem>();
  b->ctx = ctx.handle();
  b->bytes = tpp::round_up(bytes, 65536);
  ze_external_memory_export_desc_t exp{};
  exp.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;
  exp.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  ze_device_mem_alloc_desc_t dd{};
  dd.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  dd.pNext = &exp;
  ze_result_t r = zeMemAllocDevice(ctx.handle(), &dd, b->bytes, 65536, ctx.device(), &b->dev);
  if (r != ZE_RESULT_SUCCESS) {
    b->dev = nullptr;
    b->err = "zeMemAllocDevice(export dma-buf) = " + std::to_string(int(r));
    return b;
  }
  ze_external_memory_export_fd_t efd{};
  efd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
  efd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  efd.fd = -1;
  ze_memory_allocation_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  props.pNext = &efd;
  r = zeMemGetAllocProperties(ctx.handle(), b->dev, &props, nullptr);
  if (r != ZE_RESULT_SUCCESS || efd.fd < 0) {
    b->err = "zeMemGetAllocProperties(export fd) = " + std::to_string(int(r)) + " fd " + std::to_string(efd.fd);
    return b;
  }
  b->fd = efd.fd;   // owned by the driver: never closed here
  void* m = mmap(nullptr, b->bytes, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
  if (m == MAP_FAILED) {
    b->err = std::string("mmap(dma-buf fd) failed: ") + std::strerror(errno);
    b->cpu = nullptr;
    return b;
  }
  b->cpu = m;
  return b;
}

// --- facts ------------------------------------------------------------------------------------------------------------
int run_facts(const Opts& o) {
  (void)o;
  Card c[2];
  open_cards(c, false);
  ze_driver_properties_t dp{};
  dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  zeDriverGetProperties(c[0].ctx->driver(), &dp);
  std::printf("## facts\n\ndriver version %u (0x%08x)\n\n", dp.driverVersion, dp.driverVersion);
  std::printf("| card | name | PCI | max link (L0) | memory GB | EUs |\n|---|---|---|---|---|---|\n");
  for (int i = 0; i < 2; ++i) {
    ze_pci_ext_properties_t pci{};
    pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    const ze_result_t r = zeDevicePciGetPropertiesExt(c[i].ctx->device(), &pci);
    char bdf[64] = "?";
    char link[96] = "?";
    if (r == ZE_RESULT_SUCCESS) {
      std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", pci.address.domain, pci.address.bus, pci.address.device,
                    pci.address.function);
      std::snprintf(link, sizeof link, "Gen%d x%d (%.2f GB/s)", pci.maxSpeed.genVersion, pci.maxSpeed.width,
                    double(pci.maxSpeed.maxBandwidth) / 1e9);
    } else {
      std::snprintf(link, sizeof link, "zeDevicePciGetPropertiesExt = %d", int(r));
    }
    std::printf("| %d | %s | %s | %s | %.3f | %u |\n", i, c[i].ctx->name().c_str(), bdf, link,
                double(c[i].ctx->memory_bytes()) / 1e9, c[i].ctx->eu_count());
  }
  std::printf("\n| from -> to | zeDeviceCanAccessPeer | P2P ACCESS | P2P ATOMICS |\n|---|---|---|---|\n");
  for (int i = 0; i < 2; ++i) {
    const int j = 1 - i;
    ze_device_p2p_properties_t pp{};
    pp.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
    const ze_result_t r = zeDeviceGetP2PProperties(c[i].ctx->device(), c[j].ctx->device(), &pp);
    std::printf("| %d -> %d | %s | %s | %s |\n", i, j, c[i].ctx->can_access_peer(*c[j].ctx) ? "yes" : "NO",
                r == ZE_RESULT_SUCCESS ? ((pp.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS) ? "yes" : "no") : "error",
                r == ZE_RESULT_SUCCESS ? ((pp.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS) ? "yes" : "no") : "error");
  }
  std::printf("\n| card | dma-buf export + CPU mmap of a device allocation (the BAR path) |\n|---|---|\n");
  for (int i = 0; i < 2; ++i) {
    auto b = bar_alloc(*c[i].ctx, 1 << 20);
    if (b->cpu) {
      // a CPU write and read-back through the mapping
      volatile uint32_t* p = static_cast<volatile uint32_t*>(b->cpu);
      p[0] = 0x12345678u;
      cpu_store_fence();
      const uint32_t back = p[0];
      std::printf("| %d | works: mapped %zu B, CPU store + load-back %s |\n", i, b->bytes,
                  back == 0x12345678u ? "ok" : "MISMATCH");
    } else {
      std::printf("| %d | FAILS: %s |\n", i, b->err.c_str());
    }
  }
  return 0;
}

// --- bar: the CPU into card memory, the round trip -------------------------------------------------------------------
struct PathMem {   // one buffer the CPU writes and the card reads (or the ack word the other way)
  std::string name;
  void* cpu = nullptr;   // the CPU's pointer
  void* dev = nullptr;   // the card's pointer
  std::unique_ptr<BarMem> bar;
  std::unique_ptr<l0::Mem> mem;
  std::string err;
};
std::unique_ptr<PathMem> make_path(l0::Context& ctx, const std::string& kind, size_t bytes) {
  auto p = std::make_unique<PathMem>();
  p->name = kind;
  bytes = tpp::round_up(bytes, 65536);
  try {
    if (kind == "dmabuf") {
      p->bar = bar_alloc(ctx, bytes);
      if (!p->bar->cpu) {
        p->err = p->bar->err;
        return p;
      }
      p->cpu = p->bar->cpu;
      p->dev = p->bar->dev;
    } else if (kind == "host") {
      p->mem = std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, bytes, 65536);
      p->cpu = p->dev = p->mem->ptr();
    } else if (kind == "shared") {
      p->mem = std::make_unique<l0::Mem>(ctx, l0::MemKind::Shared, bytes, 65536);
      p->cpu = p->dev = p->mem->ptr();
    } else {
      p->err = "unknown path " + kind;
    }
  } catch (const std::exception& e) {
    p->err = e.what();
  }
  if (p->cpu) std::memset(p->cpu, 0, bytes);
  return p;
}

int run_bar(const Opts& o) {
  // one card: the visible device o.card (under tools/gpu_lock.sh card N it is index 0)
  Card card;
  card.owned = std::make_unique<l0::Context>(o.card);
  card.ctx = card.owned.get();
  card.q = std::make_unique<l0::Queue>(*card.ctx);
  card.f = std::make_unique<l0::Fence>(*card.q);
  card.mod = std::make_unique<l0::Module>(*card.ctx, kernels::path("ptp_kernels"));
  l0::Context& ctx = *card.ctx;
  l0::Kernel kping(*card.mod, "tp_ping");
  kping.group_size(256);
  l0::Mem state(ctx, l0::MemKind::Device, 64);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  constexpr uint32_t kPeriod = 64;   // tp_ping checks words against tp_gen_bits(i % 64, 0, e)
  const uint32_t warm = 100;
  int failures = 0;
  std::printf("## bar - visible card %u (%s), %u iterations (+%u warm-up) a cell, ack flag %s\n\n", o.card,
              ctx.name().c_str(), o.iters, warm, o.flag_mode ? "plain" : "atomic");
  std::printf("| CPU writes into | card acks into | row | RTT median us | p10 | p90 | CPU store us (median) | "
              "card errors |\n|---|---|---|---|---|---|---|---|\n");
  for (const std::string& path : o.paths)
    for (const std::string& ackk : o.acks)
      for (uint32_t words : o.words) {
        const tpp::Layout L = tpp::Layout::make(words, words);
        auto land = make_path(ctx, path, L.total_bytes());
        auto ack = make_path(ctx, ackk, 4096);
        if (!land->cpu || !ack->cpu) {
          std::printf("| %s | %s | %u B | FAILS: %s%s |  |  |  |  |\n", path.c_str(), ackk.c_str(), words * 4,
                      land->err.c_str(), ack->err.c_str());
          continue;
        }
        std::vector<std::vector<uint32_t>> rows(kPeriod, std::vector<uint32_t>(words));
        for (uint32_t i = 0; i < kPeriod; ++i)
          for (uint32_t e = 0; e < words; ++e) rows[i][e] = tpp::gen_bits(i, 0, e);
        imm.fill(state.ptr(), 0, 64);
        const uint32_t total = warm + o.iters;
        kping.arg_ptr(0, land->dev);
        kping.arg_ptr(1, ack->dev);
        kping.arg_ptr(2, state.ptr());
        kping.arg(3, words);
        kping.arg(4, uint32_t(L.flag_base()));
        kping.arg(5, total);
        kping.arg(6, o.spin_limit);
        kping.arg(7, o.flag_mode);
        l0::CmdList list = l0::CmdList::regular(ctx);
        list.launch(kping, 1);
        list.close();
        card.q->execute(list, card.f.get());
        uint32_t* lw = static_cast<uint32_t*>(land->cpu);
        volatile uint32_t* aw = static_cast<volatile uint32_t*>(ack->cpu);
        std::vector<double> rtt, st;
        bool lost = false;
        for (uint32_t i = 1; i <= total && !lost; ++i) {
          const uint32_t slot = i & 1u;
          const auto t0 = Clock::now();
          std::memcpy(lw + L.data(slot), rows[i % kPeriod].data(), size_t(words) * 4);
          __atomic_store_n(lw + L.trailer(slot, 0), i, __ATOMIC_RELAXED);
          cpu_store_fence();
          const double ts = us_since(t0);
          __atomic_store_n(lw + L.flag(slot, 0), i, __ATOMIC_RELEASE);
          cpu_store_fence();
          const auto deadline = Clock::now() + std::chrono::seconds(2);
          while (aw[0] != i) {
            cpu_relax();
            if (Clock::now() > deadline) {
              lost = true;
              break;
            }
          }
          if (i > warm && !lost) {
            rtt.push_back(us_since(t0));
            st.push_back(ts);
          }
        }
        wait_fence(*card.f, o, "bar " + path + "/" + ackk);
        uint32_t sw[8] = {};
        imm.copy(sw, state.ptr(), sizeof sw);
        const bool bad = lost || sw[0] != 0 || sw[6] != 0;
        failures += bad;
        char err[160];
        std::snprintf(err, sizeof err, "%s status %u (want %u flag %u trailer %u) words %u", lost ? "LOST ack;" : "",
                      sw[0], sw[3], sw[1], sw[2], sw[6]);
        std::printf("| %s | %s | %u B | %.2f | %.2f | %.2f | %.2f | %s |\n", path.c_str(), ackk.c_str(), words * 4,
                    median(rtt), pct(rtt, 0.1), pct(rtt, 0.9), median(st), bad ? err : "0");
        std::fflush(stdout);
      }
  return failures ? 1 : 0;
}

// --- ar: the all-reduce ---------------------------------------------------------------------------------------------
struct Side {   // one card's buffers for one (route, rows)
  std::unique_ptr<l0::Mem> P, own, out, sctr, wctr, csctr, cwctr, state, hash, cstate, chash;
};

struct ArSetup {
  tpp::Route route;
  tpp::Layout L;
  uint32_t S = 4;
  Side side[2];
  // route buffers
  std::unique_ptr<l0::Mem> dev_land[2];   // p2p: a landing / exposed buffer on each card
  std::unique_ptr<l0::Mem> mbox[2];       // pinned host mailboxes (host-*)
  std::unique_ptr<BarMem> bar[2];         // host-bar: each card's VRAM, CPU-mapped
  std::unique_ptr<l0::Mem> hsum;          // host-read: the sum, pinned host
  void* dst[2] = {nullptr, nullptr};      // tp_send's destination per card
  void* src[2] = {nullptr, nullptr};      // tp_wait's source per card
  uint32_t* cpu_dst[2] = {nullptr, nullptr};
  uint32_t cpu_ndst = 0;
  bool cpu = false, take = false;
  std::string err;
};

void zero_dev(l0::Context& ctx, void* p, size_t bytes) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.fill(p, 0, bytes);
}

std::unique_ptr<ArSetup> make_ar(Card (&c)[2], tpp::Route r, uint32_t words, const Opts& o) {
  auto a = std::make_unique<ArSetup>();
  a->route = r;
  a->L = tpp::Layout::make(words, o.chunk);
  a->S = o.S;
  const tpp::Layout& L = a->L;
  const size_t G = L.chunks();
  // the partials: the same values on both cards (each adds its own per-call term)
  std::vector<float> P(size_t(o.S) * words);
  uint32_t st = 2463534242u;
  for (float& v : P) v = tpp::bits_to_float((xorshift(st) >> 9) | 0x3e000000u) - 0.25f;
  for (int i = 0; i < 2; ++i) {
    l0::Context& ctx = *c[i].ctx;
    Side& s = a->side[i];
    auto dev = [&](size_t b) { return std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, std::max<size_t>(b, 64)); };
    s.P = dev(P.size() * 4);
    s.own = dev(size_t(words) * 4);
    s.out = dev(size_t(words) * 4);
    s.sctr = dev(G * 4);
    s.wctr = dev(G * 4);
    s.csctr = dev(G * 4);
    s.cwctr = dev(G * 4);
    s.state = dev(64);
    s.hash = dev(G * 4);
    s.cstate = dev(64);
    s.chash = dev(G * 4);
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    imm.copy(s.P->ptr(), P.data(), P.size() * 4);
  }
  const size_t lb = L.total_bytes();
  auto need_peer = [&]() {
    if (!c[0].ctx->can_access_peer(*c[1].ctx) || !c[1].ctx->can_access_peer(*c[0].ctx)) {
      a->err = "zeDeviceCanAccessPeer refused (0->1 " + std::string(c[0].ctx->can_access_peer(*c[1].ctx) ? "yes" : "no") +
               ", 1->0 " + std::string(c[1].ctx->can_access_peer(*c[0].ctx) ? "yes" : "no") + ")";
      return false;
    }
    return true;
  };
  auto host = [&](size_t b) { return std::make_unique<l0::Mem>(*c[0].ctx, l0::MemKind::Host, b, 65536); };
  switch (r) {
    case tpp::Route::P2pPush:
    case tpp::Route::P2pPull:
      if (!need_peer()) return a;
      for (int i = 0; i < 2; ++i)
        a->dev_land[i] = std::make_unique<l0::Mem>(*c[i].ctx, l0::MemKind::Device, lb, 65536);
      if (r == tpp::Route::P2pPush) {
        a->dst[0] = a->dev_land[1]->ptr(); a->dst[1] = a->dev_land[0]->ptr();
        a->src[0] = a->dev_land[0]->ptr(); a->src[1] = a->dev_land[1]->ptr();
      } else {
        a->dst[0] = a->dev_land[0]->ptr(); a->dst[1] = a->dev_land[1]->ptr();
        a->src[0] = a->dev_land[1]->ptr(); a->src[1] = a->dev_land[0]->ptr();
      }
      break;
    case tpp::Route::HostMbox:
      for (int i = 0; i < 2; ++i) a->mbox[i] = host(lb);
      a->dst[0] = a->mbox[0]->ptr(); a->dst[1] = a->mbox[1]->ptr();
      a->src[0] = a->mbox[1]->ptr(); a->src[1] = a->mbox[0]->ptr();
      break;
    case tpp::Route::HostBar:
      for (int i = 0; i < 2; ++i) {
        a->mbox[i] = host(lb);
        a->bar[i] = bar_alloc(*c[i].ctx, lb);
        if (!a->bar[i]->cpu) {
          a->err = "card " + std::to_string(i) + ": " + a->bar[i]->err;
          return a;
        }
        a->dst[i] = a->mbox[i]->ptr();
        a->src[i] = a->bar[i]->dev;
        a->cpu_dst[i] = static_cast<uint32_t*>(a->bar[i]->cpu);
      }
      a->cpu_ndst = 2;
      a->cpu = a->take = true;
      break;
    case tpp::Route::HostRead:
      for (int i = 0; i < 2; ++i) a->mbox[i] = host(lb);
      a->hsum = host(lb);
      a->dst[0] = a->mbox[0]->ptr(); a->dst[1] = a->mbox[1]->ptr();
      a->src[0] = a->src[1] = a->hsum->ptr();
      a->cpu_dst[0] = a->hsum->as<uint32_t>();
      a->cpu_ndst = 1;
      a->cpu = a->take = true;
      break;
  }
  return a;
}

// Zero the landing buffers, counters, states and hashes (before every measurement: counters restart at call 1).
void reset_ar(Card (&c)[2], ArSetup& a) {
  const size_t lb = a.L.total_bytes();
  const size_t G = a.L.chunks();
  for (int i = 0; i < 2; ++i) {
    Side& s = a.side[i];
    for (l0::Mem* m : {s.sctr.get(), s.wctr.get(), s.csctr.get(), s.cwctr.get(), s.hash.get(), s.chash.get()})
      zero_dev(*c[i].ctx, m->ptr(), G * 4);
    zero_dev(*c[i].ctx, s.state->ptr(), 64);
    zero_dev(*c[i].ctx, s.cstate->ptr(), 64);
    if (a.dev_land[i]) zero_dev(*c[i].ctx, a.dev_land[i]->ptr(), lb);
    if (a.mbox[i]) std::memset(a.mbox[i]->ptr(), 0, lb);
    if (a.bar[i]) std::memset(a.bar[i]->cpu, 0, lb);
  }
  if (a.hsum) std::memset(a.hsum->ptr(), 0, lb);
}

struct Lists {
  std::unique_ptr<l0::Kernel> send, wait, csend, cwait;
  std::unique_ptr<l0::CmdList> cross, ctl;
};

// One card's two lists: `calls` x (tp_send, tp_wait), with the crossing and without (the control).
Lists build_lists(Card& card, ArSetup& a, int rank, uint32_t calls, const Opts& o, bool check, uint32_t inject_kind) {
  Lists l;
  Side& s = a.side[rank];
  const tpp::Layout& L = a.L;
  auto mk = [&](const char* n) {
    auto k = std::make_unique<l0::Kernel>(*card.mod, n);
    k->group_size(256);
    return k;
  };
  l.send = mk("tp_send");
  l.wait = mk("tp_wait");
  l.csend = mk("tp_send");
  l.cwait = mk("tp_wait");
  auto bind_send = [&](l0::Kernel& k, uint32_t cross, l0::Mem& ctr) {
    k.arg_ptr(0, s.P->ptr());
    k.arg_ptr(1, s.own->ptr());
    k.arg_ptr(2, cross ? a.dst[rank] : s.own->ptr());   // the control never writes dst
    k.arg_ptr(3, ctr.ptr());
    k.arg(4, L.words);
    k.arg(5, L.chunk);
    k.arg(6, a.S);
    k.arg(7, uint32_t(L.flag_base()));
    k.arg(8, uint32_t(rank));
    k.arg(9, cross);
    k.arg(10, (cross && rank == 1) ? o.inject_at : 0u);
    k.arg(11, inject_kind);
    k.arg(12, o.flag_mode);
  };
  auto bind_wait = [&](l0::Kernel& k, uint32_t cross, l0::Mem& ctr, l0::Mem& state, l0::Mem& hash) {
    k.arg_ptr(0, cross ? a.src[rank] : s.own->ptr());
    k.arg_ptr(1, s.own->ptr());
    k.arg_ptr(2, s.out->ptr());
    k.arg_ptr(3, ctr.ptr());
    k.arg_ptr(4, state.ptr());
    k.arg_ptr(5, hash.ptr());
    k.arg_ptr(6, s.P->ptr());
    k.arg(7, L.words);
    k.arg(8, L.chunk);
    k.arg(9, a.S);
    k.arg(10, uint32_t(L.flag_base()));
    k.arg(11, uint32_t(rank));
    k.arg(12, cross);
    k.arg(13, uint32_t(a.take ? 1 : 0));
    k.arg(14, o.spin_limit);
    k.arg(15, uint32_t(check ? 1 : 0));
  };
  bind_send(*l.send, 1, *s.sctr);
  bind_wait(*l.wait, 1, *s.wctr, *s.state, *s.hash);
  bind_send(*l.csend, 0, *s.csctr);
  bind_wait(*l.cwait, 0, *s.cwctr, *s.cstate, *s.chash);
  const uint32_t G = L.chunks();
  l.cross = std::make_unique<l0::CmdList>(l0::CmdList::regular(*card.ctx));
  l.ctl = std::make_unique<l0::CmdList>(l0::CmdList::regular(*card.ctx));
  for (uint32_t i = 0; i < calls; ++i) {
    l.cross->launch(*l.send, G);
    l.cross->launch(*l.wait, G);
    l.ctl->launch(*l.csend, G);
    l.ctl->launch(*l.cwait, G);
  }
  l.cross->close();
  l.ctl->close();
  return l;
}

// The CPU-sum thread of routes host-bar / host-read: `calls` calls, every chunk, on one pinned core.
struct CpuRun {
  uint32_t state[tpp::kStateWords] = {};
  double cpu_s = 0, wall_s = 0;
};
void cpu_sum_loop(ArSetup& a, uint64_t calls, uint32_t spin, int core, CpuRun& out) {
  pin_to_core(core);
  const tpp::Layout& L = a.L;
  std::vector<uint32_t> ctr(L.chunks(), 0), scratch(L.chunk);
  const uint32_t* mb0 = a.mbox[0]->as<uint32_t>();
  const uint32_t* mb1 = a.mbox[1]->as<uint32_t>();
  const double c0 = thread_cpu_s();
  const auto t0 = Clock::now();
  for (uint64_t n = 1; n <= calls; ++n)
    for (uint32_t j = 0; j < L.chunks(); ++j)
      tpp::cpu_sum_chunk(mb0, mb1, a.cpu_dst, a.cpu_ndst, L, j, ctr[j], spin, out.state, scratch.data(),
                         cpu_store_fence);
  out.cpu_s = thread_cpu_s() - c0;
  out.wall_s = us_since(t0) * 1e-6;
}

// Execute both cards' lists `execs` times (each execution waits for both fences); the CPU thread runs alongside
// when the route needs it. Returns the wall time in us per execution.
double exec_both(Card (&c)[2], ArSetup& a, Lists (&l)[2], bool cross, uint32_t calls, uint32_t execs, const Opts& o,
                 CpuRun* cpu, const std::string& arm) {
  std::thread th;
  if (cross && a.cpu)
    th = std::thread([&] { cpu_sum_loop(a, uint64_t(calls) * execs, o.spin_limit, o.cpu, *cpu); });
  const auto t0 = Clock::now();
  for (uint32_t e = 0; e < execs; ++e) {
    c[1].q->execute(cross ? *l[1].cross : *l[1].ctl, c[1].f.get());
    c[0].q->execute(cross ? *l[0].cross : *l[0].ctl, c[0].f.get());
    wait_fence(*c[0].f, o, arm);
    wait_fence(*c[1].f, o, arm);
  }
  const double us = us_since(t0) / execs;
  if (th.joinable()) th.join();
  return us;
}

struct CardState {
  uint32_t w[8];
};
CardState read_state(Card& c, l0::Mem& m) {
  CardState s{};
  l0::CmdList imm = l0::CmdList::immediate(*c.ctx);
  imm.copy(s.w, m.ptr(), sizeof s.w);
  return s;
}
uint32_t read_hash(Card& c, l0::Mem& m, uint32_t G) {
  std::vector<uint32_t> h(G);
  l0::CmdList imm = l0::CmdList::immediate(*c.ctx);
  imm.copy(h.data(), m.ptr(), G * 4);
  uint32_t t = 0;
  for (uint32_t v : h) t += v;
  return t;
}
std::string state_str(const uint32_t* w) {
  char b[200];
  std::snprintf(b, sizeof b, "status %u (flag %u trailer %u want %u chunk %u) failed %u mism %u", w[0], w[1], w[2],
                w[3], w[4], w[5], w[6]);
  return b;
}

uint32_t inject_kind(const std::string& s) {
  if (s == "none") return tpp::kInjNone;
  if (s == "skip-data") return tpp::kInjSkipData;
  if (s == "skip-all") return tpp::kInjSkipAll;
  die("--inject " + s + " (none|skip-data|skip-all)");
}

int run_ar(const Opts& o) {
  const model::ModelDesc& d = tp::model_by_name(o.model);
  const tp::Shapes sh = tp::tp_shapes(d, 0, 2);
  const uint32_t calls = o.calls ? o.calls : sh.calls;
  Card c[2];
  open_cards(c, true);
  const uint32_t inj = inject_kind(o.inject);
  const bool check = o.replays > 0 || inj != tpp::kInjNone;
  int failures = 0;
  std::printf("## ar - %s, %u calls a list (the model's all-reduces a token: %u), %u B a row, chunk %u words, S %u, "
              "flag %s\n\n",
              d.name.c_str(), calls, sh.calls, sh.row_bytes, o.chunk, o.S, o.flag_mode ? "plain" : "atomic");
  if (check)
    std::printf("| route | rows | lists replayed | calls checked | card 0 | card 1 | CPU | hashes equal | verdict |\n"
                "|---|---|---|---|---|---|---|---|---|\n");
  else
    std::printf("| route | rows | ctl us/list | cross us/list | per call us (range) | per token at %u calls ms | "
                "CPU core busy | status |\n|---|---|---|---|---|---|---|---|\n",
                sh.calls);
  for (const std::string& rn : o.routes) {
    const tpp::Route r = tpp::route_from_name(rn);
    for (uint32_t M : o.rows) {
      auto a = make_ar(c, r, M * sh.hidden, o);
      if (!a->err.empty()) {
        std::printf("| %s | %u | NOT AVAILABLE: %s |\n", rn.c_str(), M, a->err.c_str());
        std::fflush(stdout);
        continue;
      }
      Lists l[2] = {build_lists(c[0], *a, 0, calls, o, check, inj), build_lists(c[1], *a, 1, calls, o, check, inj)};
      const std::string arm = std::string("ar ") + rn + " M" + std::to_string(M);
      if (check) {
        const uint32_t reps = o.replays ? o.replays : 4;
        reset_ar(c, *a);
        CpuRun cpu;
        exec_both(c, *a, l, true, calls, reps, o, &cpu, arm);
        const CardState s0 = read_state(c[0], *a->side[0].state), s1 = read_state(c[1], *a->side[1].state);
        const uint32_t h0 = read_hash(c[0], *a->side[0].hash, a->L.chunks());
        const uint32_t h1 = read_hash(c[1], *a->side[1].hash, a->L.chunks());
        bool ok;
        if (inj == tpp::kInjNone) {
          ok = s0.w[0] == 0 && s1.w[0] == 0 && s0.w[6] == 0 && s1.w[6] == 0 && h0 == h1 && cpu.state[0] == 0;
        } else {
          // the injected write must be caught as stale (skip-data) or a timeout (skip-all) - by card 0 or the CPU -
          // at the injected call, and never be consumed silently: a wrong word without a recorded failure fails.
          const uint32_t want = inj == tpp::kInjSkipData ? tpp::kStale : tpp::kTimeout;
          const bool caught = (s0.w[0] == want && s0.w[3] == o.inject_at) ||
                              (cpu.state[0] == want && cpu.state[3] == o.inject_at);
          const bool silent = (s0.w[6] != 0 && s0.w[0] == 0) || (s1.w[6] != 0 && s1.w[0] == 0);
          ok = caught && !silent;
        }
        failures += !ok;
        std::printf("| %s | %u | %u | %llu | %s | %s | %s | %s | %s |\n", rn.c_str(), M, reps,
                    (unsigned long long)reps * calls, state_str(s0.w).c_str(), state_str(s1.w).c_str(),
                    a->cpu ? state_str(cpu.state).c_str() : "-", h0 == h1 ? "yes" : "NO",
                    ok ? (inj == tpp::kInjNone ? "PASS bitwise" : "PASS caught") : "FAIL");
        std::fflush(stdout);
        continue;
      }
      // timed: warm-up, then interleaved pairs (order alternating), median of rounds
      std::vector<double> ctl, crs, per;
      double cpu_busy = 0;
      bool bad = false;
      std::string why;
      for (uint32_t w = 0; w < 2; ++w) {
        reset_ar(c, *a);
        CpuRun cpu;
        exec_both(c, *a, l, false, calls, 2, o, nullptr, arm);
        exec_both(c, *a, l, true, calls, 2, o, &cpu, arm);
      }
      for (uint32_t round = 0; round < o.rounds; ++round) {
        reset_ar(c, *a);
        CpuRun cpu;
        double tc, tx;
        if (round % 2 == 0) {
          tc = exec_both(c, *a, l, false, calls, o.execs, o, nullptr, arm);
          tx = exec_both(c, *a, l, true, calls, o.execs, o, &cpu, arm);
        } else {
          tx = exec_both(c, *a, l, true, calls, o.execs, o, &cpu, arm);
          tc = exec_both(c, *a, l, false, calls, o.execs, o, nullptr, arm);
        }
        const CardState s0 = read_state(c[0], *a->side[0].state), s1 = read_state(c[1], *a->side[1].state);
        if (s0.w[0] || s1.w[0] || cpu.state[0]) {
          bad = true;
          why = "card0 " + state_str(s0.w) + "; card1 " + state_str(s1.w) +
                (a->cpu ? "; cpu " + state_str(cpu.state) : "");
        }
        if (a->cpu && cpu.wall_s > 0) cpu_busy = std::max(cpu_busy, cpu.cpu_s / cpu.wall_s);
        ctl.push_back(tc);
        crs.push_back(tx);
        per.push_back((tx - tc) / calls);
      }
      failures += bad;
      const double pc = median(per);
      char busy[32] = "-";
      if (a->cpu) std::snprintf(busy, sizeof busy, "%.0f %%", 100.0 * cpu_busy);
      std::printf("| %s | %u | %.1f | %.1f | %.2f (%.2f-%.2f) | %.3f | %s | %s |\n", rn.c_str(), M, median(ctl),
                  median(crs), pc, *std::min_element(per.begin(), per.end()), *std::max_element(per.begin(), per.end()),
                  pc * sh.calls / 1000.0, busy, bad ? why.c_str() : "ok");
      std::fflush(stdout);
    }
  }
  return failures ? 1 : 0;
}

// --- step: the per-card linears at TP shapes ------------------------------------------------------------------------
struct GemvArm {
  std::string name;
  uint32_t K = 0, N = 0, S = 1, L = 0;
  bool i8 = false;
  size_t wbytes = 0;
  std::unique_ptr<l0::Module> mod;
  std::unique_ptr<l0::Kernel> k;
  std::vector<std::unique_ptr<l0::Mem>> w, sc;
  std::unique_ptr<l0::Mem> x, out;
  std::unique_ptr<l0::CmdList> list;
};
std::string card_variant(const tp::Linear& l) {
  if (l.kind == model::WeightKind::Int8)
    return "ptp_gemv_i8w_M1_K" + std::to_string(l.card.K) + "_N" + std::to_string(l.card.N);
  return "ptp_gemv_M1_K" + std::to_string(l.card.K) + "_N" + std::to_string(l.card.N) + "_S" +
         std::to_string(l.card.S) + "_L" + std::to_string(l.card.layout);
}
std::string one_variant(const tp::Linear& l) {
  if (l.kind == model::WeightKind::Int8) return kernels::gemv_i8w_variant(1, l.one.K, l.one.N);
  return kernels::gemv_variant(1, l.one.K, l.one.N, l.one.S, l.one.layout);
}
bool file_exists(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return bool(f);
}

std::unique_ptr<GemvArm> make_gemv(l0::Context& ctx, const std::string& name, const tp::Gemv& g, bool i8,
                                   uint32_t launches, uint32_t seed) {
  auto a = std::make_unique<GemvArm>();
  a->name = name;
  a->K = g.K; a->N = g.N; a->S = g.S; a->L = g.layout; a->i8 = i8;
  if (!file_exists(kernels::path(name))) return a;
  a->mod = std::make_unique<l0::Module>(ctx, kernels::path(name));
  a->k = std::make_unique<l0::Kernel>(*a->mod, i8 ? "gemv_i8w" : "gemv");
  a->k->group_size(64);
  // random, incompressible weights of the shape (the B70 compresses constant fills); bytes as the layouts store them
  size_t wb, sb;
  if (i8) {
    wb = size_t(g.K) * g.N;
    sb = size_t(g.N) * 4;
  } else {
    wb = size_t(g.K) * g.N / 2 + (g.layout == 1 ? size_t(g.K / 64) * g.N * 2 : 0);
    sb = g.layout == 1 ? 64 : size_t(g.K / 64) * g.N * 2;
  }
  a->wbytes = wb + (sb > 64 ? sb : 0);
  std::vector<uint32_t> wh(wb / 4);
  uint32_t st = seed * 2654435761u + 7;
  for (uint32_t& v : wh) v = xorshift(st);
  std::vector<uint16_t> sh(sb / 2);
  if (i8) {
    std::vector<float> f(g.N);
    for (float& v : f) v = 1e-3f * (1.0f + float(xorshift(st) % 1000) / 1000.f);
    std::memcpy(sh.data(), f.data(), sb);
  } else {
    for (uint16_t& v : sh) v = uint16_t(0x2c00u | (xorshift(st) & 0x3ffu));
  }
  if (!i8 && g.layout == 1)   // tiled: the scales are inside the tiles; keep them finite f16
    for (size_t t = 0; t < wh.size(); t += 136)
      for (size_t q = 128; q < 136 && t + q < wh.size(); ++q) wh[t + q] = 0x2c002c00u | (xorshift(st) & 0x03ff03ffu);
  const int NB = std::max(2, int((72u << 20) / a->wbytes) + 1);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  for (int i = 0; i < NB; ++i) {
    a->w.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, wb));
    imm.copy(a->w.back()->ptr(), wh.data(), wb);
    a->sc.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, std::max<size_t>(sb, 64)));
    if (sb > 64) imm.copy(a->sc.back()->ptr(), sh.data(), sb);
  }
  std::vector<uint16_t> x(g.K);
  for (uint16_t& v : x) v = uint16_t(0x3c00u + (xorshift(st) & 0xffu));   // bf16 ~ 0.0078..
  a->x = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(g.K) * 2);
  imm.copy(a->x->ptr(), x.data(), x.size() * 2);
  a->out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(g.S) * g.N * 4);
  a->list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx));
  for (uint32_t i = 0; i < launches; ++i) {
    const int cpy = int(i % a->w.size());
    a->k->arg_ptr(0, a->w[cpy]->ptr());
    a->k->arg_ptr(1, a->sc[cpy]->ptr());
    a->k->arg_ptr(2, a->x->ptr());
    a->k->arg_ptr(3, a->out->ptr());
    a->list->launch(*a->k, g.N / 64, i8 ? 1 : g.S);
  }
  a->list->close();
  return a;
}

int run_step(const Opts& o) {
  const model::ModelDesc& d = tp::model_by_name(o.model);
  const tp::Shapes sh = tp::tp_shapes(d, 0, 2);
  l0::Context ctx(o.card);
  l0::Queue q(ctx);
  l0::Fence f(q);
  int failures = 0;
  std::printf("## step - %s, card %u (one card: ZE_AFFINITY_MASK decides which), M = 1, %u launches a list, "
              "rounds %u\n\n",
              d.name.c_str(), o.card, o.launches, o.rounds);
  std::printf("| linear (layers) | one card | us | per card (TP 2) | us | per card / one card (range) | "
              "GB/s one | GB/s card |\n|---|---|---|---|---|---|---|---|\n");
  double sum_one = 0, sum_card = 0;
  for (const tp::Linear& l : sh.linears) {
    if (l.kind == model::WeightKind::Bf16) continue;   // a‖b: unchanged per card (zero-padded to the same tile)
    if (!o.shapes.empty() && std::find(o.shapes.begin(), o.shapes.end(), l.key) == o.shapes.end()) continue;
    const bool i8 = l.kind == model::WeightKind::Int8;
    const uint32_t n_one = i8 ? std::max(1u, o.launches / 5) : o.launches;   // the head: fewer, 1.27 GB each
    auto one = make_gemv(ctx, one_variant(l), l.one, i8, n_one, 11);
    auto card = make_gemv(ctx, card_variant(l), l.card, i8, n_one, 13);
    if (!one->k || !card->k) {
      std::printf("| %s | %s | MISSING %s |\n", l.name, one->name.c_str(), (!one->k ? one->name : card->name).c_str());
      ++failures;
      continue;
    }
    auto time_list = [&](GemvArm& a) {
      const auto t0 = Clock::now();
      q.execute(*a.list, &f);
      wait_fence(f, o, "step " + a.name);
      return us_since(t0) / n_one;
    };
    for (int w = 0; w < 3; ++w) {
      time_list(*one);
      time_list(*card);
    }
    std::vector<double> to, tc, ratio;
    for (uint32_t r = 0; r < o.rounds; ++r) {
      double a, b;
      if (r % 2 == 0) {
        a = time_list(*one);
        b = time_list(*card);
      } else {
        b = time_list(*card);
        a = time_list(*one);
      }
      to.push_back(a);
      tc.push_back(b);
      ratio.push_back(b / a);
    }
    const double mo = median(to), mc = median(tc);
    sum_one += l.layers * mo;
    sum_card += l.layers * mc;
    std::printf("| %s (%u) | %ux%u S%u | %.1f | %ux%u S%u | %.1f | %.3f (%.3f-%.3f) | %.0f | %.0f |\n", l.name, l.layers,
                l.one.K, l.one.N, l.one.S, mo, l.card.K, l.card.N, l.card.S, mc, median(ratio),
                *std::min_element(ratio.begin(), ratio.end()), *std::max_element(ratio.begin(), ratio.end()),
                double(one->wbytes) / mo / 1e3, double(card->wbytes) / mc / 1e3);
    std::fflush(stdout);
  }
  std::printf("\nGEMV sum of a token (int4 linears x layers + the int8 head once; a‖b excluded): one card %.0f us, "
              "per card at TP 2 %.0f us, ratio %.3f\n",
              sum_one, sum_card, sum_one > 0 ? sum_card / sum_one : 0.0);
  return failures ? 1 : 0;
}

Opts parse(int argc, char** argv) {
  Opts o;
  if (argc < 2) die("usage: probe_tp facts|bar|ar|step [options] (see the header of tools/probe/probe_tp.cc)");
  o.mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) die(a + " needs a value");
      return argv[++i];
    };
    if (a == "--model") o.model = val();
    else if (a == "--route") { const std::string v = val(); if (v != "all") o.routes = split(v); }
    else if (a == "--rows") o.rows = split_u(val());
    else if (a == "--calls") o.calls = uint32_t(std::stoul(val()));
    else if (a == "--rounds") o.rounds = uint32_t(std::stoul(val()));
    else if (a == "--execs") o.execs = uint32_t(std::stoul(val()));
    else if (a == "--replays") o.replays = uint32_t(std::stoul(val()));
    else if (a == "--chunk") o.chunk = uint32_t(std::stoul(val()));
    else if (a == "--s") o.S = uint32_t(std::stoul(val()));
    else if (a == "--spin") o.spin_limit = uint32_t(std::stoul(val()));
    else if (a == "--inject") o.inject = val();
    else if (a == "--inject-at") o.inject_at = uint32_t(std::stoul(val()));
    else if (a == "--cpu") o.cpu = std::stoi(val());
    else if (a == "--card") o.card = uint32_t(std::stoul(val()));
    else if (a == "--words") o.words = split_u(val());
    else if (a == "--iters") o.iters = uint32_t(std::stoul(val()));
    else if (a == "--paths") o.paths = split(val());
    else if (a == "--ack") o.acks = split(val());
    else if (a == "--launches") o.launches = uint32_t(std::stoul(val()));
    else if (a == "--shape") { const std::string v = val(); if (v != "all") o.shapes = split(v); }
    else if (a == "--timeout") o.timeout_s = std::stod(val());
    else if (a == "--flag") {
      const std::string v = val();
      if (v == "atomic") o.flag_mode = 0;
      else if (v == "plain") o.flag_mode = 1;
      else die("--flag " + v + " (atomic|plain)");
    }
    else die("unknown option " + a);
  }
  if (o.inject != "none" && o.inject_at == 0) o.inject_at = 37;
  if (o.rounds == 0 || o.execs == 0) die("--rounds and --execs must be >= 1");
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Opts o = parse(argc, argv);
    if (o.mode == "facts") return run_facts(o);
    if (o.mode == "bar") return run_bar(o);
    if (o.mode == "ar") return run_ar(o);
    if (o.mode == "step") return run_step(o);
    die("unknown mode " + o.mode + " (facts|bar|ar|step)");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_tp: %s\n", e.what());
    return 1;
  }
}

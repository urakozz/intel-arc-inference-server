// probe_offload - spec 22 P0 (plan 22a Task 2): everything the expert-offload tier depends on, measured on the box
// before any engine work. One Level Zero binary on the repo's l0:: wrappers; device 1 is a view of device 0's
// context (spec 16 decision 1: one context over both cards), so a host-USM range pinned once is read by both.
//
//   probe_offload <mode> [--dev 0|1|both] [options]      every result is one line `po <mode> dev=<d> key=value ...`
//
//   copy    P0.2  pinned host -> device copies: 0.87 MB (one down), 1.74 MB (gate||up), 2.61 MB (an expert), 16 / 64 /
//                 256 experts as one copy and as per-expert copies, 1 GiB; on an immediate list and inside a regular
//                 list replayed; on the compute ordinal and on the copy-only queue group (COPY without COMPUTE,
//                 `copy_ordinal=none` when absent); the first-byte latency of one 2.6 MB and of one 4-byte copy
//                 (submit to fence). `--dev both`: one thread per card behind a start barrier, each and the aggregate.
//   zc      P0.3  M = 1, a regular list of P launch pairs (gate||up then down, pair j routed to 10 distinct experts of
//                 the card's 512-expert layer image, the pairs' experts disjoint); arms: A 21c's binary over the VRAM
//                 image (control); B the table kernel (probe_offload_moe.cl) with an identity table over the same
//                 VRAM; C every routed slot's entry a host-USM pointer (a byte copy of the image, pinned); D1-D3 1 / 2
//                 / 3 of a pair's 10 slots on host; S the misses of D1-D3 / C staged by the copy queue into a VRAM
//                 bounce buffer (an event), then B's kernel (decision 1's comparison, an upper bound: the host knows
//                 the misses); K the realisable in-list staging (po_stage gathers host-entry slots into a bounce
//                 buffer, then B's kernel over the launch table); the ceiling po_read_sum over the same bytes, host
//                 and VRAM. A-vs-each as interleaved pairs after a ~0.5 s warm-up, median of --reps with the range;
//                 every run's y compared bitwise with A's. Then the mixed-launch law (sum: serial; max: overlapped).
//   replay  P0.3  arms D2 and C in a regular list replayed --n (10^4) times: y read back and compared every replay;
//                 every 100th replay the host rewrites one host expert with a new pattern (its expected y computed
//                 once by arm A over a VRAM copy) and the next replay must match; the host image's page words read
//                 back by po_page_check every 1000 replays. Any mismatch prints the replay and page and stops the arm.
//   hostbw  P0.4  a STREAM triad over 3 x 2 GiB on 1 and on all hardware threads (GPUs idle); then arm C on both
//                 cards (the zero-copy aggregate) with the CPU idle, beside a 64 MiB copy loop on one thread
//                 (PROPOSED stand-in for the server's per-token host work), and beside the all-thread triad.
//   pin     P0.5  the largest pinned host allocation as one range (doubling from 1 GiB, then bisection to 1 GiB),
//                 then 48 per-layer ranges of 1,336,934,400 B + the PLE's 16 + 16 shards (21b's rule): where it
//                 fails, MemAvailable before / after, seconds per GB; every range tagged per 2 MiB page by the host
//                 and read back by po_page_check on each card. Never above MemAvailable - 16 GiB (spec 7's rule).
//   alias   P0.5  a model-sized allocation history on the card (18 layers' allocations at 21b's sizes, the int8 head,
//                 the PLE pointer table; then freed), then one device arena of --arena-gb (22): po_page_tag writes a
//                 salted tag at every 2 MiB page, po_page_check gathers them, the host compares; a second salt
//                 confirms. A page holding another page's tag is an alias, printed with both addresses.
//   hold    P0.1  copy's 1 GiB copy in a loop for --seconds on the named cards (offload_links.sh's LnkSta under load)
//
// Options: --dev 0|1|both (0), --pairs 1,2,7,26, --reps 5, --n 10000, --replay-pairs 7, --arena-gb 22, --seconds 20,
// --max-gb G (pin: a cap below MemAvailable - 16 GiB), --timeout-s 20 (every wait; a hang is a FAIL line).
// Exit 0 when every arm ran and matched, 1 on a mismatch, a hang or an error, 77 when a precondition is missing.
// Sizes only from loader/qwen4exp_layout.h and model::qwen4exp(). Record: docs/probe-offload-<date>.md.
#include <level_zero/ze_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/error.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "l0/sync_event.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_ple_hash.h"
#include "model/qwen4exp.h"

namespace {

namespace qk = kernels::qwen4exp;

constexpr size_t kPage = size_t(2) << 20;   // the alias check's page (spec 22 §1: xe mapped two 2 MiB pages)
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr size_t kMarginBytes = size_t(16) << 30;   // spec 7's rule: never above MemAvailable - 16 GiB
constexpr uint32_t kBf16Tenth = 0x3DCD0000u;         // fp32 bits of rne_bf16(0.1f): every routed weight
constexpr const char* kTableVariant = "po_moe_M1_E512_T10_D2560_I640_SH4";
constexpr const char* kHelperVariant = "po_kernels";

struct Hang : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Skip : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Opts {
  std::string mode, dev = "0";
  std::vector<uint32_t> pairs = {1, 2, 7, 26};
  uint32_t reps = 5, n = 10000, replay_pairs = 7;
  double arena_gb = 22, seconds = 20, max_gb = 0, timeout_s = 20;
  uint64_t timeout_ns() const { return uint64_t(timeout_s * 1e9); }
};
Opts g_opt;
int g_fail = 0;

// --- sizes (loader/qwen4exp_layout.h) ------------------------------------------------------------------------------
struct Sizes {
  size_t gu = 0, dn = 0, expert = 0, layer_gu = 0, layer = 0, sh_gu = 0, sh_dn = 0;
  uint32_t E = 0, K = 0, hidden = 0, inter = 0;
};
Sizes sizes() {
  const model::Qwen4ExpDesc& d = model::qwen4exp();
  Sizes s;
  s.gu = loader::q4_gate_up_block_bytes(d);   // 1,740,800
  s.dn = loader::q4_down_block_bytes(d);      // 870,400
  s.expert = s.gu + s.dn;
  s.E = d.experts;
  s.K = d.top_k;
  s.layer_gu = size_t(s.E) * s.gu;
  s.layer = loader::q4_layer_bytes(d, 0).experts();   // 1,336,934,400
  s.sh_gu = loader::q4_shared_gate_up_bytes(d);       // ours: int4 layout-1 (the _SH4 binary)
  s.sh_dn = loader::q4_shared_down_bytes(d);
  s.hidden = d.hidden;
  s.inter = d.moe_inter;
  if (s.layer != s.layer_gu + size_t(s.E) * s.dn) throw std::logic_error("probe_offload: the layer image is not 512 x (gu + dn)");
  return s;
}

uint64_t mem_available() {
  std::ifstream in("/proc/meminfo");
  std::string key, unit;
  uint64_t kb = 0;
  while (in >> key >> kb >> unit)
    if (key == "MemAvailable:") return kb * 1024;
  return 0;
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stat {
  double med = 0, lo = 0, hi = 0;
};
Stat stat(std::vector<double> v) {
  Stat s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.med = v[v.size() / 2];
  s.lo = v.front();
  s.hi = v.back();
  return s;
}

uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// int4 layout-1 blocks (TILE_U32 136: 128 words of nibbles, 8 words of f16 scales): random nibbles, finite scales
// in [2^-8, 2^-7) - the kernels' outputs stay finite, so bitwise comparisons compare numbers.
void fill_blocks(uint32_t* w, size_t words, uint64_t seed) {
  constexpr size_t kTile = loader::kQ4TileU32;
  uint64_t x = seed;
  for (size_t i = 0; i < words; ++i) {
    const uint64_t r = splitmix(x);
    if (i % kTile < 128) {
      w[i] = uint32_t(r);
    } else {
      const uint32_t a = 0x1C00u | uint32_t(r & 0x3FFu), b = 0x1C00u | uint32_t((r >> 10) & 0x3FFu);
      w[i] = a | (b << 16);
    }
  }
}

// --- one card ------------------------------------------------------------------------------------------------------
struct Ordinals {
  uint32_t compute = 0;
  int copy = -1;   // the copy-only group (COPY without COMPUTE); -1 when absent
};
Ordinals find_ordinals(l0::Context& c) {
  uint32_t n = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(c.device(), &n, nullptr));
  std::vector<ze_command_queue_group_properties_t> p(n);
  for (auto& g : p) {
    g = {};
    g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  }
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(c.device(), &n, p.data()));
  Ordinals o;
  bool have_compute = false;
  for (uint32_t i = 0; i < n; ++i) {
    const bool compute = p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
    const bool copy = p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    if (compute && !have_compute) {
      o.compute = i;
      have_compute = true;
    }
    if (copy && !compute && o.copy < 0) o.copy = int(i);
  }
  return o;
}

// An ASYNCHRONOUS immediate list (l0::CmdList::immediate is synchronous: an append would be an unbounded wait).
// Copies are appended, then a barrier signals `done`, which the host waits on with a bound.
class AsyncImm {
 public:
  AsyncImm(l0::Context& c, uint32_t ordinal) {
    ze_command_queue_desc_t qd{};
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.ordinal = ordinal;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandListCreateImmediate(c.handle(), c.device(), &qd, &l_));
  }
  ~AsyncImm() {
    if (l_) zeCommandListDestroy(l_);
  }
  AsyncImm(const AsyncImm&) = delete;
  AsyncImm& operator=(const AsyncImm&) = delete;
  void copy(void* dst, const void* src, size_t bytes) {
    ZE_CHECK(zeCommandListAppendMemoryCopy(l_, dst, src, bytes, nullptr, 0, nullptr));
  }
  void signal(l0::SyncEvent& e) { ZE_CHECK(zeCommandListAppendBarrier(l_, e.handle(), 0, nullptr)); }

 private:
  ze_command_list_handle_t l_ = nullptr;
};

void wait_event(l0::SyncEvent& e, const std::string& what) {
  if (!e.host_wait(g_opt.timeout_ns())) throw Hang(what + ": no completion within " + std::to_string(g_opt.timeout_s) + " s");
}
void wait_fence(l0::Fence& f, const std::string& what) {
  if (!f.wait_for(g_opt.timeout_ns())) throw Hang(what + ": no completion within " + std::to_string(g_opt.timeout_s) + " s");
}

struct Card {
  uint32_t index = 0;
  l0::Context& ctx;
  Ordinals ord;
  l0::Queue q;
  l0::Fence f;
  std::unique_ptr<l0::Queue> cq;   // the copy-only group's queue (when present)
  std::unique_ptr<l0::Fence> cf;
  l0::SyncEvent ev;                // host-waited completions (bounded copies) and the S arm's copy -> compute event
  AsyncImm imm;
  l0::TimerCalib calib;
  Card(uint32_t i, l0::Context& c)
      : index(i), ctx(c), ord(find_ordinals(c)), q(c, ord.compute), f(q), ev(c, {&c}), imm(c, ord.compute),
        calib(l0::TimerCalib::query(c)) {
    if (ord.copy >= 0) {
      cq = std::make_unique<l0::Queue>(c, uint32_t(ord.copy));
      cf = std::make_unique<l0::Fence>(*cq);
    }
  }
  // A bounded copy on the compute ordinal (setup and readbacks).
  void copy(void* dst, const void* src, size_t bytes, const char* what) {
    ev.host_reset();
    imm.copy(dst, src, bytes);
    imm.signal(ev);
    wait_event(ev, std::string("dev ") + std::to_string(index) + " " + what);
  }
  void run(l0::CmdList& l, const std::string& what) {
    q.execute(l, &f);
    wait_fence(f, "dev " + std::to_string(index) + " " + what);
  }
  std::string ordinal_name() const { return ord.copy >= 0 ? std::to_string(ord.copy) : "none"; }
};

struct Devices {
  std::unique_ptr<l0::Context> c0, c1;
  std::vector<std::unique_ptr<Card>> cards;   // the cards --dev names, in index order
  std::string label;                          // "0", "1" or "both"
};
Devices open_devices(const std::string& dev) {
  Devices d;
  if (dev != "0" && dev != "1" && dev != "both") throw std::invalid_argument("--dev " + dev + ": 0, 1 or both");
  d.label = dev;
  d.c0 = std::make_unique<l0::Context>(0);
  const uint32_t n = l0::Context::gpu_count();
  if (dev != "0") {
    if (n < 2) throw Skip("--dev " + dev + " needs two GPUs, the driver enumerates " + std::to_string(n));
    d.c1 = std::make_unique<l0::Context>(*d.c0, 1);
    if (dev == "both" && !d.c0->can_access_peer(*d.c1))
      std::printf("po info dev=both peer=no (the arms here need none: every buffer is the card's own or host USM)\n");
  }
  if (dev == "0" || dev == "both") d.cards.push_back(std::make_unique<Card>(0, *d.c0));
  if (dev == "1" || dev == "both") d.cards.push_back(std::make_unique<Card>(1, *d.c1));
  for (const auto& c : d.cards)
    std::printf("po info dev=%u name=\"%s\" compute_ordinal=%u copy_ordinal=%s eus=%u memory_gb=%.2f\n", c->index,
                c->ctx.name().c_str(), c->ord.compute, c->ordinal_name().c_str(), c->ctx.eu_count(),
                c->ctx.memory_bytes() / 1e9);
  return d;
}

void append_ts(l0::CmdList& l, uint64_t* dst) {
  ZE_CHECK(zeCommandListAppendWriteGlobalTimestamp(l.handle(), dst, nullptr, 0, nullptr));
}

// --- the shared host data -------------------------------------------------------------------------------------------
struct HostImage {   // one layer's 512 experts in pinned host USM (gate||up blocks, then down blocks): the mirror
  std::unique_ptr<l0::Mem> mem;
  const Sizes* s = nullptr;
  uint8_t* gu(uint32_t e) const { return mem->as<uint8_t>() + size_t(e) * s->gu; }
  uint8_t* dn(uint32_t e) const { return mem->as<uint8_t>() + s->layer_gu + size_t(e) * s->dn; }
};
HostImage make_host_image(l0::Context& c0, const Sizes& s) {
  const uint64_t avail = mem_available();
  if (avail && s.layer + kMarginBytes > avail)
    throw Skip("host RAM: the 1.34 GB host image needs MemAvailable >= " + std::to_string((s.layer + kMarginBytes) >> 30) +
               " GiB, have " + std::to_string(avail >> 30));
  HostImage h;
  h.s = &s;
  h.mem = std::make_unique<l0::Mem>(c0, l0::MemKind::Host, s.layer, kPage);
  fill_blocks(h.mem->as<uint32_t>(), s.layer / 4, 0x22A0001);
  return h;
}

// --- the expert arms (zc, replay, hostbw) --------------------------------------------------------------------------
enum class Arm { A, B, C, D, S, K };
const char* arm_name(Arm a) {
  switch (a) {
    case Arm::A: return "A";
    case Arm::B: return "B";
    case Arm::C: return "C";
    case Arm::D: return "D";
    case Arm::S: return "S";
    case Arm::K: return "K";
  }
  return "?";
}

// The pairs' experts: pair j's slot k routes expert perm[10 j + k] - disjoint across pairs, at most 51 pairs.
std::vector<uint32_t> expert_perm(uint32_t E) {
  std::vector<uint32_t> p(E);
  std::iota(p.begin(), p.end(), 0u);
  std::mt19937 rng(2210);
  std::shuffle(p.begin(), p.end(), rng);
  return p;
}

struct Rig {   // one card's buffers for up to P pairs
  Card& card;
  const Sizes& s;
  const HostImage& host;
  uint32_t P;
  std::vector<uint32_t> perm;
  l0::Module direct, table, helper;
  l0::Mem vram, sh_gu, sh_dn, x, routes, h, y, bounce, ltable, out, ts;
  l0::Mem y_host;
  std::vector<uint16_t> y_ref;   // arm A's y, the bitwise reference
  std::vector<std::unique_ptr<l0::Mem>> tables;   // device u64 [1024] each, by table_id()
  std::vector<std::unique_ptr<l0::Kernel>> ks;     // every launch's own kernel object (its arguments)
  Rig(Card& c, const Sizes& sz, const HostImage& hi, uint32_t pairs)
      : card(c), s(sz), host(hi), P(pairs), perm(expert_perm(sz.E)),
        direct(c.ctx, kernels::path(qk::moe_variant(1, false))), table(c.ctx, kernels::path(kTableVariant)),
        helper(c.ctx, kernels::path(kHelperVariant)), vram(c.ctx, l0::MemKind::Device, sz.layer, kPage),
        sh_gu(c.ctx, l0::MemKind::Device, sz.sh_gu), sh_dn(c.ctx, l0::MemKind::Device, sz.sh_dn),
        x(c.ctx, l0::MemKind::Device, size_t(sz.hidden) * 2), routes(c.ctx, l0::MemKind::Device, size_t(pairs) * qk::route::kWords * 4),
        h(c.ctx, l0::MemKind::Device, size_t(pairs) * qk::kSlots * sz.inter * 2),
        y(c.ctx, l0::MemKind::Device, size_t(pairs) * sz.hidden * 2),
        bounce(c.ctx, l0::MemKind::Device, size_t(pairs) * sz.K * sz.expert, 4096),
        ltable(c.ctx, l0::MemKind::Device, size_t(2) * sz.E * 8), out(c.ctx, l0::MemKind::Device, 1 << 20),
        ts(c.ctx, l0::MemKind::Host, 64 * 2 * 8), y_host(c.ctx, l0::MemKind::Host, size_t(pairs) * sz.hidden * 2) {
    if (size_t(pairs) * sz.K > sz.E) throw std::invalid_argument("probe_offload: at most 51 pairs (disjoint experts)");
    c.copy(vram.ptr(), host.mem->ptr(), sz.layer, "the VRAM image");
    std::vector<uint32_t> w(std::max(sz.sh_gu, sz.sh_dn) / 4);
    fill_blocks(w.data(), sz.sh_gu / 4, 0x22A0002);
    c.copy(sh_gu.ptr(), w.data(), sz.sh_gu, "the shared gate||up");
    fill_blocks(w.data(), sz.sh_dn / 4, 0x22A0003);
    c.copy(sh_dn.ptr(), w.data(), sz.sh_dn, "the shared down");
    std::vector<uint16_t> xv(sz.hidden);
    uint64_t sx = 0x22A0004;
    for (auto& v : xv) v = uint16_t(0x3E00u | (splitmix(sx) & 0x7Fu) | ((splitmix(sx) & 1u) << 15));   // +-[0.125, 0.25)
    c.copy(x.ptr(), xv.data(), xv.size() * 2, "x");
    std::vector<uint32_t> rr(size_t(pairs) * qk::route::kWords, 0u);
    for (uint32_t j = 0; j < pairs; ++j)
      for (uint32_t k = 0; k < sz.K; ++k) {
        rr[size_t(j) * qk::route::kWords + qk::route::kIds + k] = perm[j * sz.K + k];
        rr[size_t(j) * qk::route::kWords + qk::route::kWeights + k] = kBf16Tenth;
      }
    // the shared gate 0 (route::kSharedGate already zero): the shared slot is computed and weighted out
    c.copy(routes.ptr(), rr.data(), rr.size() * 4, "the route rows");
  }
  uint32_t expert(uint32_t j, uint32_t k) const { return perm[j * s.K + k]; }
  uint64_t vram_gu(uint32_t e) const { return uint64_t(reinterpret_cast<uintptr_t>(vram.as<uint8_t>() + size_t(e) * s.gu)); }
  uint64_t vram_dn(uint32_t e) const {
    return uint64_t(reinterpret_cast<uintptr_t>(vram.as<uint8_t>() + s.layer_gu + size_t(e) * s.dn));
  }
  uint8_t* bounce_gu(uint32_t j, uint32_t k) const { return bounce.as<uint8_t>() + size_t(j) * s.K * s.expert + size_t(k) * s.gu; }
  uint8_t* bounce_dn(uint32_t j, uint32_t k) const {
    return bounce.as<uint8_t>() + size_t(j) * s.K * s.expert + size_t(s.K) * s.gu + size_t(k) * s.dn;
  }
  // Which slots of a pair miss (host): arm C all 10, D d of them (slots 0 .. d-1), else none.
  static uint32_t misses(Arm a, uint32_t d, uint32_t K) { return a == Arm::C ? K : (a == Arm::D || a == Arm::S || a == Arm::K) ? d : 0; }
  // The table for an arm: identity VRAM, with each pair's missing slots at their host addresses (C / D / K) or at
  // their bounce slots (S, staged by the copy queue).
  l0::Mem& make_table(Arm a, uint32_t d) {
    std::vector<uint64_t> t(size_t(2) * s.E);
    for (uint32_t e = 0; e < s.E; ++e) {
      t[e] = vram_gu(e);
      t[s.E + e] = vram_dn(e);
    }
    const uint32_t m = misses(a, d, s.K);
    for (uint32_t j = 0; j < P; ++j)
      for (uint32_t k = 0; k < m; ++k) {
        const uint32_t e = expert(j, k);
        if (a == Arm::S) {
          t[e] = uint64_t(reinterpret_cast<uintptr_t>(bounce_gu(j, k)));
          t[s.E + e] = uint64_t(reinterpret_cast<uintptr_t>(bounce_dn(j, k)));
        } else {
          t[e] = uint64_t(reinterpret_cast<uintptr_t>(host.gu(e)));
          t[s.E + e] = uint64_t(reinterpret_cast<uintptr_t>(host.dn(e)));
        }
      }
    tables.push_back(std::make_unique<l0::Mem>(card.ctx, l0::MemKind::Device, t.size() * 8));
    card.copy(tables.back()->ptr(), t.data(), t.size() * 8, "a table");
    return *tables.back();
  }
  l0::Kernel& kernel(l0::Module& m, const char* name, uint32_t wg) {
    ks.push_back(std::make_unique<l0::Kernel>(m, name));
    ks.back()->group_size(wg);
    return *ks.back();
  }
  // One arm's compute list: [ts0] (K: po_stage) gate||up, down per pair [ts1], y -> y_host. `ts_slot` picks the
  // timestamp pair in `ts`. S's copies are a separate list (stage_list); without a copy group they lead this one.
  l0::CmdList arm_list(Arm a, uint32_t d, uint32_t ts_slot, bool s_waits_event) {
    l0::CmdList l = l0::CmdList::regular(card.ctx, card.ord.compute);
    const uint32_t m = misses(a, d, s.K);
    if (a == Arm::S && !s_waits_event) append_stage_copies(l, m);
    if (a == Arm::S && s_waits_event) l.wait_event(card.ev.handle());
    l0::Mem* tb = a == Arm::A ? nullptr : &make_table(a, d);
    uint64_t* t = ts.as<uint64_t>() + 2 * ts_slot;
    append_ts(l, t);
    for (uint32_t j = 0; j < P; ++j) {
      const uint32_t* route = routes.as<uint32_t>() + size_t(j) * qk::route::kWords;
      uint8_t* hj = h.as<uint8_t>() + size_t(j) * qk::kSlots * s.inter * 2;
      uint8_t* yj = y.as<uint8_t>() + size_t(j) * s.hidden * 2;
      const void* w_gu = vram.ptr();
      const void* w_dn = vram.as<uint8_t>() + s.layer_gu;
      if (tb) {
        w_gu = tb->ptr();
        w_dn = tb->as<uint64_t>() + s.E;
      }
      if (a == Arm::K) {
        l0::Kernel& st = kernel(helper, "po_stage", 256);
        st.arg_ptr(0, route);
        st.arg_ptr(1, tb->ptr());
        st.arg(2, uint64_t(reinterpret_cast<uintptr_t>(vram.ptr())));
        st.arg(3, uint64_t(reinterpret_cast<uintptr_t>(vram.as<uint8_t>() + s.layer)));
        st.arg_ptr(4, bounce.as<uint8_t>() + size_t(j) * s.K * s.expert);
        st.arg_ptr(5, ltable.ptr());
        l.launch(st, s.K, kPoChunks);
        w_gu = ltable.ptr();
        w_dn = ltable.as<uint64_t>() + s.E;
      }
      l0::Module& mod = a == Arm::A ? direct : table;
      l0::Kernel& gu = kernel(mod, "q4_moe_gate_up", qk::moe_gate_up_wg());
      gu.arg_ptr(0, route);
      gu.arg_ptr(1, x.ptr());
      gu.arg_ptr(2, w_gu);
      gu.arg_ptr(3, sh_gu.ptr());
      gu.arg_ptr(4, hj);
      l.launch(gu, qk::moe_gate_up_groups(), 1);
      l0::Kernel& dn = kernel(mod, "q4_moe_down", qk::moe_down_wg());
      dn.arg_ptr(0, route);
      dn.arg_ptr(1, hj);
      dn.arg_ptr(2, w_dn);
      dn.arg_ptr(3, sh_dn.ptr());
      dn.arg_ptr(4, yj);
      l.launch(dn, s.hidden / 16, 1);
    }
    append_ts(l, t + 1);
    l.copy(y_host.ptr(), y.ptr(), size_t(P) * s.hidden * 2);
    l.close();
    return l;
  }
  void append_stage_copies(l0::CmdList& l, uint32_t m) {
    for (uint32_t j = 0; j < P; ++j)
      for (uint32_t k = 0; k < m; ++k) {
        const uint32_t e = expert(j, k);
        l.copy(bounce_gu(j, k), host.gu(e), s.gu);
        l.copy(bounce_dn(j, k), host.dn(e), s.dn);
      }
  }
  // S's copy list on the copy-only group: the misses into the bounce slots, then a barrier signalling the event.
  l0::CmdList stage_list(uint32_t m) {
    l0::CmdList l = l0::CmdList::regular(card.ctx, uint32_t(card.ord.copy));
    append_stage_copies(l, m);
    l.barrier_signal(card.ev.handle());
    l.close();
    return l;
  }
  // po_read_sum over `bytes` from `ptr` (the ceiling): its own list with timestamps.
  l0::CmdList ceiling_list(const void* ptr, size_t bytes, uint32_t ts_slot) {
    l0::CmdList l = l0::CmdList::regular(card.ctx, card.ord.compute);
    l0::Kernel& k = kernel(helper, "po_read_sum", 256);
    k.arg_ptr(0, ptr);
    k.arg(1, uint64_t(bytes / 16));
    k.arg_ptr(2, out.ptr());
    uint64_t* t = ts.as<uint64_t>() + 2 * ts_slot;
    append_ts(l, t);
    l.launch(k, card.ctx.eu_count() * 8 * 16 / 256 * 4);
    append_ts(l, t + 1);
    l.close();
    return l;
  }
  double dev_us(uint32_t ts_slot) const {
    const uint64_t* t = ts.as<uint64_t>() + 2 * ts_slot;
    return double((t[1] & card.calib.mask()) - (t[0] & card.calib.mask())) / card.calib.cycles_per_us;
  }
  std::vector<uint16_t> y_now() const {
    const uint16_t* p = y_host.as<uint16_t>();
    return std::vector<uint16_t>(p, p + size_t(P) * s.hidden);
  }
  static constexpr uint32_t kPoChunks = 32;   // = PO_CHUNKS (tools/probe/CMakeLists.txt)
};

// One prepared arm on one card: its compute list (+ S's copy list) and the timestamp slot.
struct ArmRun {
  Rig* rig = nullptr;
  Arm arm = Arm::A;
  uint32_t d = 0, slot = 0;
  std::unique_ptr<l0::CmdList> list, stage;
  bool ceiling = false;
};
std::unique_ptr<ArmRun> prepare(Rig& r, Arm a, uint32_t d, uint32_t slot) {
  auto run = std::make_unique<ArmRun>();
  run->rig = &r;
  run->arm = a;
  run->d = d;
  run->slot = slot;
  const bool copy_group = r.card.ord.copy >= 0;
  run->list = std::make_unique<l0::CmdList>(r.arm_list(a, d, slot, a == Arm::S && copy_group));
  if (a == Arm::S && copy_group) run->stage = std::make_unique<l0::CmdList>(r.stage_list(Rig::misses(a, d, r.s.K)));
  return run;
}
std::unique_ptr<ArmRun> prepare_ceiling(Rig& r, const void* ptr, size_t bytes, uint32_t slot) {
  auto run = std::make_unique<ArmRun>();
  run->rig = &r;
  run->slot = slot;
  run->ceiling = true;
  run->list = std::make_unique<l0::CmdList>(r.ceiling_list(ptr, bytes, slot));
  return run;
}

// Submits every run (one per card) and waits for all: the wall seconds (submit to the last fence).
double execute(const std::vector<ArmRun*>& runs, const std::string& what) {
  const double t0 = now_s();
  for (ArmRun* r : runs) {
    Card& c = r->rig->card;
    if (r->stage) {
      c.ev.host_reset();
      c.cq->execute(*r->stage, c.cf.get());
    }
    c.q.execute(*r->list, &c.f);
  }
  for (ArmRun* r : runs) {
    Card& c = r->rig->card;
    wait_fence(c.f, "dev " + std::to_string(c.index) + " " + what);
    if (r->stage) wait_fence(*c.cf, "dev " + std::to_string(c.index) + " " + what + " (copy queue)");
  }
  return now_s() - t0;
}

// The first index where two y rows differ, or -1.
long first_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return long(i);
  return a.size() == b.size() ? -1 : long(n);
}

// --- zc ------------------------------------------------------------------------------------------------------------
struct ArmResult {
  Stat dev_us, wall_us, ratio;
  bool exact = true;
};

// A-vs-X as interleaved pairs after a ~0.5 s warm-up burst, on every card of `rigs` concurrently. Arm A's device
// times of every pair are appended to `a_dev` (per card): A's own median comes from all of them.
std::vector<ArmResult> paired(std::vector<std::unique_ptr<ArmRun>>& A, std::vector<std::unique_ptr<ArmRun>>& X,
                              bool use_wall, const std::string& what, std::vector<std::vector<double>>& a_dev) {
  std::vector<ArmRun*> a, x;
  for (auto& r : A) a.push_back(r.get());
  for (auto& r : X) x.push_back(r.get());
  const double warm_end = now_s() + 0.5;
  for (int i = 0; i < 1000 && now_s() < warm_end; ++i) {
    execute(a, what + " warm-up A");
    execute(x, what + " warm-up");
  }
  const size_t n = A.size();
  std::vector<std::vector<double>> dev(n), wall(n), ratio(n);
  std::vector<ArmResult> res(n);
  for (uint32_t rep = 0; rep < g_opt.reps; ++rep) {
    const double wa = execute(a, what + " A");
    std::vector<double> ta(n);
    for (size_t i = 0; i < n; ++i) {
      ta[i] = A[i]->rig->dev_us(A[i]->slot);
      a_dev[i].push_back(ta[i]);
      if (first_diff(A[i]->rig->y_now(), A[i]->rig->y_ref) >= 0) res[i].exact = false;   // A itself must repeat
    }
    const double wx = execute(x, what);
    for (size_t i = 0; i < n; ++i) {
      Rig& r = *X[i]->rig;
      const double tx = r.dev_us(X[i]->slot);
      const double basis_x = use_wall ? wx * 1e6 : tx, basis_a = use_wall ? wa * 1e6 : ta[i];
      dev[i].push_back(tx);
      wall[i].push_back(wx * 1e6);
      ratio[i].push_back(basis_x / basis_a);
      if (!X[i]->ceiling && first_diff(r.y_now(), r.y_ref) >= 0) res[i].exact = false;
    }
  }
  for (size_t i = 0; i < n; ++i) {
    res[i].dev_us = stat(dev[i]);
    res[i].wall_us = stat(wall[i]);
    res[i].ratio = stat(ratio[i]);
  }
  return res;
}

int zc_mode() {
  const Sizes s = sizes();
  Devices dv = open_devices(g_opt.dev);
  HostImage host = make_host_image(*dv.c0, s);
  for (uint32_t P : g_opt.pairs) {
    if (P == 0 || size_t(P) * s.K > s.E) throw std::invalid_argument("--pairs: 1..51 pairs (disjoint experts)");
    std::vector<std::unique_ptr<Rig>> rigs;
    for (auto& c : dv.cards) rigs.push_back(std::make_unique<Rig>(*c, s, host, P));
    // arm A first: its y is every other arm's bitwise reference
    std::vector<std::unique_ptr<ArmRun>> A;
    for (auto& r : rigs) A.push_back(prepare(*r, Arm::A, 0, 0));
    {
      std::vector<ArmRun*> a;
      for (auto& r : A) a.push_back(r.get());
      execute(a, "arm A (reference)");
      for (auto& r : rigs) r->y_ref = r->y_now();
    }
    struct Spec {
      Arm arm;
      uint32_t d;
    };
    const std::vector<Spec> specs = {{Arm::B, 0}, {Arm::C, 0}, {Arm::D, 1}, {Arm::D, 2}, {Arm::D, 3},
                                     {Arm::S, 1}, {Arm::S, 2}, {Arm::S, 3}, {Arm::S, s.K}, {Arm::K, 1},
                                     {Arm::K, 2}, {Arm::K, 3}, {Arm::K, s.K}};
    const double routed_bytes = double(P) * s.K * s.expert;
    std::vector<std::vector<double>> t_of(rigs.size(), std::vector<double>(16, 0.0));   // dev us: [0] A, [1] B, [2] C, [3+d] D
    std::vector<double> tA(rigs.size(), 0.0);
    std::vector<std::vector<double>> a_dev(rigs.size());
    uint32_t slot = 1;
    for (const Spec& sp : specs) {
      std::vector<std::unique_ptr<ArmRun>> X;
      for (auto& r : rigs) X.push_back(prepare(*r, sp.arm, sp.d, slot));
      ++slot;
      const bool wall = sp.arm == Arm::S;
      const std::string what = std::string("arm ") + arm_name(sp.arm) + " P=" + std::to_string(P);
      const std::vector<ArmResult> res = paired(A, X, wall, what, a_dev);
      for (size_t i = 0; i < rigs.size(); ++i) {
        const ArmResult& r = res[i];
        const uint32_t m = Rig::misses(sp.arm, sp.d, s.K);
        const double host_bytes = double(P) * m * s.expert;
        const double t = wall ? r.wall_us.med : r.dev_us.med;
        std::printf("po zc dev=%s card=%u P=%u arm=%s d=%u basis=%s us=%.2f range=%.2f-%.2f ratio_vs_A=%.4f "
                    "ratio_range=%.4f-%.4f host_bytes=%.0f host_gbps=%.3f routed_gbps=%.3f exact=%d\n",
                    dv.label.c_str(), rigs[i]->card.index, P, arm_name(sp.arm), m, wall ? "wall" : "dev", t,
                    wall ? r.wall_us.lo : r.dev_us.lo, wall ? r.wall_us.hi : r.dev_us.hi, r.ratio.med, r.ratio.lo,
                    r.ratio.hi, host_bytes, host_bytes > 0 ? host_bytes / t / 1e3 : 0.0, routed_bytes / t / 1e3,
                    r.exact ? 1 : 0);
        if (!r.exact) {
          ++g_fail;
          std::printf("po zc dev=%s card=%u P=%u arm=%s d=%u FAIL y differs from arm A's\n", dv.label.c_str(),
                      rigs[i]->card.index, P, arm_name(sp.arm), m);
        }
        if (!wall) {
          if (sp.arm == Arm::B) t_of[i][1] = r.dev_us.med;
          if (sp.arm == Arm::C) t_of[i][2] = r.dev_us.med;
          if (sp.arm == Arm::D) t_of[i][3 + sp.d] = r.dev_us.med;
        }
      }
    }
    // the ceilings: po_read_sum over the routed bytes, contiguous, host and VRAM
    for (int where = 0; where < 2; ++where) {
      std::vector<std::unique_ptr<ArmRun>> X;
      for (auto& r : rigs) X.push_back(prepare_ceiling(*r, where == 0 ? host.mem->ptr() : r->vram.ptr(), size_t(routed_bytes), slot));
      ++slot;
      const std::vector<ArmResult> res = paired(A, X, false, where == 0 ? "ceiling host" : "ceiling vram", a_dev);
      for (size_t i = 0; i < rigs.size(); ++i)
        std::printf("po zc dev=%s card=%u P=%u arm=ceiling_%s bytes=%.0f us=%.2f range=%.2f-%.2f gbps=%.3f\n",
                    dv.label.c_str(), rigs[i]->card.index, P, where == 0 ? "host" : "vram", routed_bytes,
                    res[i].dev_us.med, res[i].dev_us.lo, res[i].dev_us.hi, routed_bytes / res[i].dev_us.med / 1e3);
    }
    // arm A's own time: the median of its runs in every pair above
    for (size_t i = 0; i < rigs.size(); ++i) {
      const Stat st = stat(a_dev[i]);
      tA[i] = st.med;
      std::printf("po zc dev=%s card=%u P=%u arm=A d=0 basis=dev us=%.2f range=%.2f-%.2f runs=%zu routed_gbps=%.3f\n",
                  dv.label.c_str(), rigs[i]->card.index, P, tA[i], st.lo, st.hi, a_dev[i].size(), routed_bytes / tA[i] / 1e3);
    }
    // the mixed-launch law (spec 22 §2: sum = serial, max = overlapped)
    for (size_t i = 0; i < rigs.size(); ++i) {
      const double a = tA[i], c = t_of[i][2];
      const double hslot = (c - a) / s.K;   // a slot moved to host adds this under the sum law
      double alpha_sum = 0;
      uint32_t n_alpha = 0;
      for (uint32_t d = 1; d <= 3; ++d) {
        const double t = t_of[i][3 + d];
        const double pred_sum = a + d * hslot;
        const double pred_max = std::max(a * double(s.K + 1 - d) / double(s.K + 1), c * double(d) / double(s.K));
        const double den = pred_sum - pred_max;
        const double alpha = den > 1e-9 ? std::min(1.0, std::max(0.0, (t - pred_max) / den)) : 1.0;
        alpha_sum += alpha;
        ++n_alpha;
        std::printf("po zc dev=%s card=%u P=%u law d=%u us=%.2f pred_sum=%.2f pred_max=%.2f alpha=%.3f\n",
                    dv.label.c_str(), rigs[i]->card.index, P, d, t, pred_sum, pred_max, alpha);
      }
      const double alpha = alpha_sum / n_alpha;
      const char* law = alpha > 0.75 ? "sum" : alpha < 0.25 ? "max" : "mix";
      std::printf("po zc dev=%s card=%u P=%u law_summary law=%s alpha=%.3f tA_us=%.2f tC_us=%.2f P_gbps=%.3f "
                  "P_marginal_gbps=%.3f\n",
                  dv.label.c_str(), rigs[i]->card.index, P, law, alpha, a, c, routed_bytes / c / 1e3,
                  c > a ? routed_bytes / (c - a) / 1e3 : 0.0);
    }
  }
  return g_fail ? 1 : 0;
}

// --- replay ----------------------------------------------------------------------------------------------------------
// The first u64 of every 2 MiB page of every range, read on the card through po_page_check, against `expect(range,
// page)`: the first mismatching global page (described in *where) or -1; *nbad counts every mismatch.
long page_check(Card& c, l0::Module& helper, const std::vector<const uint8_t*>& ranges, const std::vector<size_t>& bytes,
                const std::function<uint64_t(uint32_t, size_t)>& expect, std::string* where, size_t* nbad = nullptr);

int replay_mode() {
  const Sizes s = sizes();
  Devices dv = open_devices(g_opt.dev);
  HostImage host = make_host_image(*dv.c0, s);
  const uint32_t P = g_opt.replay_pairs;
  std::vector<std::unique_ptr<Rig>> rigs;
  for (auto& c : dv.cards) rigs.push_back(std::make_unique<Rig>(*c, s, host, P));
  const uint32_t e = rigs[0]->expert(0, 0);   // pair 0 slot 0: on host in both D2 and C
  // 4 patterns of expert e's blocks (0 = the image's own) and their expected y by arm A over a VRAM copy
  constexpr uint32_t kPatterns = 4;
  std::vector<std::vector<uint8_t>> pat(kPatterns, std::vector<uint8_t>(s.expert));
  std::memcpy(pat[0].data(), host.gu(e), s.gu);
  std::memcpy(pat[0].data() + s.gu, host.dn(e), s.dn);
  for (uint32_t p = 1; p < kPatterns; ++p) fill_blocks(reinterpret_cast<uint32_t*>(pat[p].data()), s.expert / 4, 0x22A0100 + p);
  std::vector<std::vector<std::vector<uint16_t>>> want(rigs.size(), std::vector<std::vector<uint16_t>>(kPatterns));
  for (size_t i = 0; i < rigs.size(); ++i) {
    Rig& r = *rigs[i];
    std::unique_ptr<ArmRun> a = prepare(r, Arm::A, 0, 0);
    for (uint32_t p = 0; p < kPatterns; ++p) {
      r.card.copy(r.vram.as<uint8_t>() + size_t(e) * s.gu, pat[p].data(), s.gu, "pattern gate||up");
      r.card.copy(r.vram.as<uint8_t>() + s.layer_gu + size_t(e) * s.dn, pat[p].data() + s.gu, s.dn, "pattern down");
      std::vector<ArmRun*> v = {a.get()};
      execute(v, "arm A (pattern reference)");
      want[i][p] = r.y_now();
    }
    r.card.copy(r.vram.as<uint8_t>() + size_t(e) * s.gu, pat[0].data(), s.gu, "pattern 0 gate||up");
    r.card.copy(r.vram.as<uint8_t>() + s.layer_gu + size_t(e) * s.dn, pat[0].data() + s.gu, s.dn, "pattern 0 down");
    if (first_diff(want[i][1], want[i][0]) < 0) {
      std::printf("po replay dev=%s card=%u FAIL the rewrite patterns do not change y (the check would be blind)\n",
                  dv.label.c_str(), r.card.index);
      return 1;
    }
  }
  const std::vector<const uint8_t*> ranges = {host.mem->as<uint8_t>()};
  const std::vector<size_t> rbytes = {s.layer};
  const auto host_word = [&](uint32_t, size_t page) { return *reinterpret_cast<const uint64_t*>(host.mem->as<uint8_t>() + page * kPage); };
  struct Spec {
    Arm arm;
    uint32_t d;
  };
  for (const Spec& sp : {Spec{Arm::D, 2}, Spec{Arm::C, 0}}) {
    std::memcpy(host.gu(e), pat[0].data(), s.gu);   // every arm starts from pattern 0
    std::memcpy(host.dn(e), pat[0].data() + s.gu, s.dn);
    std::vector<std::unique_ptr<ArmRun>> X;
    for (auto& r : rigs) X.push_back(prepare(*r, sp.arm, sp.d, 1));
    std::vector<ArmRun*> x;
    for (auto& r : X) x.push_back(r.get());
    const uint32_t m = Rig::misses(sp.arm, sp.d, s.K);
    std::vector<std::vector<double>> dev(rigs.size());
    uint32_t p = 0, rewrites = 0, checks = 0;
    bool ok = true;
    uint32_t i = 0;
    for (; i < g_opt.n && ok; ++i) {
      if (i > 0 && i % 100 == 0) {   // between replays: the host rewrites expert e; the next replay must see it
        p = (p + 1) % kPatterns;
        std::memcpy(host.gu(e), pat[p].data(), s.gu);
        std::memcpy(host.dn(e), pat[p].data() + s.gu, s.dn);
        ++rewrites;
      }
      execute(x, std::string("replay arm ") + arm_name(sp.arm) + " #" + std::to_string(i));
      for (size_t k = 0; k < rigs.size() && ok; ++k) {
        dev[k].push_back(rigs[k]->dev_us(1));
        const long dfi = first_diff(rigs[k]->y_now(), want[k][p]);
        if (dfi >= 0) {
          ok = false;
          ++g_fail;
          std::printf("po replay dev=%s card=%u arm=%s d=%u FAIL replay=%u pattern=%u y_index=%ld (pair %ld, column %ld)\n",
                      dv.label.c_str(), rigs[k]->card.index, arm_name(sp.arm), m, i, p, dfi, dfi / long(s.hidden),
                      dfi % long(s.hidden));
        }
      }
      if (ok && (i % 1000 == 0 || i + 1 == g_opt.n)) {
        for (size_t k = 0; k < rigs.size() && ok; ++k) {
          std::string where;
          const long bad = page_check(rigs[k]->card, rigs[k]->helper, ranges, rbytes, host_word, &where);
          ++checks;
          if (bad >= 0) {
            ok = false;
            ++g_fail;
            std::printf("po replay dev=%s card=%u arm=%s d=%u FAIL replay=%u page=%ld %s\n", dv.label.c_str(),
                        rigs[k]->card.index, arm_name(sp.arm), m, i, bad, where.c_str());
          }
        }
      }
    }
    for (size_t k = 0; k < rigs.size(); ++k) {
      const Stat st = stat(dev[k]);
      std::printf("po replay dev=%s card=%u arm=%s d=%u P=%u replays=%u rewrites=%u page_checks=%u us=%.2f range=%.2f-%.2f "
                  "result=%s\n",
                  dv.label.c_str(), rigs[k]->card.index, arm_name(sp.arm), m, P, i, rewrites, checks, st.med, st.lo, st.hi,
                  ok ? "exact" : "MISMATCH");
    }
  }
  std::memcpy(host.gu(e), pat[0].data(), s.gu);
  std::memcpy(host.dn(e), pat[0].data() + s.gu, s.dn);
  return g_fail ? 1 : 0;
}

// --- page checks (replay, pin) ---------------------------------------------------------------------------------------
long page_check(Card& c, l0::Module& helper, const std::vector<const uint8_t*>& ranges, const std::vector<size_t>& bytes,
                const std::function<uint64_t(uint32_t, size_t)>& expect, std::string* where, size_t* nbad) {
  std::vector<uint64_t> ptrs;
  std::vector<uint32_t> first = {0};
  for (size_t i = 0; i < ranges.size(); ++i) {
    ptrs.push_back(uint64_t(reinterpret_cast<uintptr_t>(ranges[i])));
    first.push_back(first.back() + uint32_t((bytes[i] + kPage - 1) / kPage));
  }
  const uint32_t pages = first.back();
  l0::Mem dp(c.ctx, l0::MemKind::Device, ptrs.size() * 8), df(c.ctx, l0::MemKind::Device, first.size() * 4),
      dout(c.ctx, l0::MemKind::Device, size_t(pages) * 8);
  c.copy(dp.ptr(), ptrs.data(), ptrs.size() * 8, "page table");
  c.copy(df.ptr(), first.data(), first.size() * 4, "page prefix");
  l0::Kernel k(helper, "po_page_check");
  k.group_size(64);
  k.arg_ptr(0, dp.ptr());
  k.arg_ptr(1, df.ptr());
  k.arg_ptr(2, dout.ptr());
  k.arg(3, pages);
  k.arg(4, uint32_t(ranges.size()));
  l0::CmdList l = l0::CmdList::regular(c.ctx, c.ord.compute);
  l.launch(k, (pages + 63) / 64);
  l.close();
  c.run(l, "po_page_check");
  std::vector<uint64_t> got(pages);
  c.copy(got.data(), dout.ptr(), got.size() * 8, "page words");
  long first_bad = -1;
  size_t bad = 0;
  for (size_t ri = 0; ri < ranges.size(); ++ri)
    for (uint32_t g = first[ri]; g < first[ri + 1]; ++g) {
      const uint64_t w = expect(uint32_t(ri), g - first[ri]);
      if (got[g] == w) continue;
      ++bad;
      if (first_bad < 0) {
        char b[200];
        std::snprintf(b, sizeof b, "range=%zu range_page=%u device_word=0x%016llx host_word=0x%016llx", ri, g - first[ri],
                      (unsigned long long)got[g], (unsigned long long)w);
        *where = b;
        first_bad = long(g);
      }
    }
  if (nbad) *nbad = bad;
  return first_bad;
}

// --- copy (P0.2) and hold (P0.1) -------------------------------------------------------------------------------------
struct CopyShape {
  std::string name;
  size_t bytes = 0;    // one copy's bytes
  uint32_t count = 1;  // copies (per-expert form: count experts of s.expert each)
};

struct CopyRig {   // one card's source (pinned host) and destination (VRAM), 1 GiB each
  Card& card;
  l0::Mem src, dst;
  CopyRig(Card& c, l0::Context& c0)
      : card(c), src(c0, l0::MemKind::Host, size_t(1) << 30, kPage), dst(c.ctx, l0::MemKind::Device, size_t(1) << 30, kPage) {
    uint32_t* p = src.as<uint32_t>();
    uint64_t x = 0x22A0200 + c.index;
    for (size_t i = 0; i < src.size() / 4; ++i) p[i] = uint32_t(splitmix(x));
  }
};

// A start barrier for the per-card threads (C++17: no std::barrier).
class StartGate {
 public:
  explicit StartGate(size_t n) : n_(n) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lk(m_);
    if (++k_ == n_) {
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return k_ >= n_; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  size_t n_, k_ = 0;
};

// One timed copy round of `shape` on a card: kind 0 = async immediate, 1 = a regular list (built once, replayed).
struct CopyTimer {
  CopyRig& r;
  const CopyShape& sh;
  bool regular;
  uint32_t ordinal;
  std::unique_ptr<AsyncImm> imm;
  std::unique_ptr<l0::CmdList> list;
  std::unique_ptr<l0::Queue> q;
  std::unique_ptr<l0::Fence> f;
  CopyTimer(CopyRig& rr, const CopyShape& s, bool reg, uint32_t ord) : r(rr), sh(s), regular(reg), ordinal(ord) {
    if (regular) {
      list = std::make_unique<l0::CmdList>(l0::CmdList::regular(r.card.ctx, ordinal));
      append(*list);
      list->close();
      q = std::make_unique<l0::Queue>(r.card.ctx, ordinal);
      f = std::make_unique<l0::Fence>(*q);
    } else {
      imm = std::make_unique<AsyncImm>(r.card.ctx, ordinal);
    }
  }
  template <class L>
  void append(L& l) {
    for (uint32_t i = 0; i < sh.count; ++i)
      l.copy(r.dst.as<uint8_t>() + size_t(i) * sh.bytes, r.src.as<uint8_t>() + size_t(i) * sh.bytes, sh.bytes);
  }
  double once() {   // seconds, submit to completion
    const double t0 = now_s();
    if (regular) {
      q->execute(*list, f.get());
      wait_fence(*f, "dev " + std::to_string(r.card.index) + " copy " + sh.name);
    } else {
      r.card.ev.host_reset();
      append(*imm);
      imm->signal(r.card.ev);
      wait_event(r.card.ev, "dev " + std::to_string(r.card.index) + " copy " + sh.name);
    }
    return now_s() - t0;
  }
};

int copy_mode() {
  const Sizes s = sizes();
  Devices dv = open_devices(g_opt.dev);
  std::vector<std::unique_ptr<CopyRig>> rigs;
  for (auto& c : dv.cards) rigs.push_back(std::make_unique<CopyRig>(*c, *dv.c0));
  std::vector<CopyShape> shapes = {{"down", s.dn, 1}, {"gate_up", s.gu, 1}, {"expert", s.expert, 1}};
  for (uint32_t n : {16u, 64u, 256u}) {
    shapes.push_back({std::to_string(n) + "_experts_one", size_t(n) * s.expert, 1});
    shapes.push_back({std::to_string(n) + "_experts_each", s.expert, n});
  }
  shapes.push_back({"1gib", size_t(1) << 30, 1});
  for (int reg = 0; reg < 2; ++reg)
    for (int on_copy = 0; on_copy < 2; ++on_copy) {
      if (on_copy && dv.cards[0]->ord.copy < 0) {
        std::printf("po copy dev=%s list=%s ordinal=copy copy_ordinal=none SKIP no copy-only queue group\n",
                    dv.label.c_str(), reg ? "regular" : "immediate");
        continue;
      }
      for (const CopyShape& sh : shapes) {
        std::vector<std::unique_ptr<CopyTimer>> ts;
        for (auto& r : rigs)
          ts.push_back(std::make_unique<CopyTimer>(*r, sh, reg != 0,
                                                   on_copy ? uint32_t(r->card.ord.copy) : r->card.ord.compute));
        const size_t bytes = sh.bytes * sh.count;
        std::vector<std::vector<double>> per(ts.size());
        std::vector<double> agg;
        for (uint32_t rep = 0; rep < g_opt.reps + 2; ++rep) {   // 2 warm-ups
          std::vector<double> t(ts.size(), 0.0);
          StartGate gate(ts.size());
          std::vector<std::thread> th;
          std::vector<std::string> err(ts.size());
          const double t0 = now_s();
          for (size_t i = 0; i < ts.size(); ++i)
            th.emplace_back([&, i] {
              gate.arrive_and_wait();
              try {
                t[i] = ts[i]->once();
              } catch (const std::exception& e) {
                err[i] = e.what();
              }
            });
          for (auto& x : th) x.join();
          const double wall = now_s() - t0;
          for (const std::string& e : err)
            if (!e.empty()) throw Hang(e);
          if (rep < 2) continue;
          for (size_t i = 0; i < ts.size(); ++i) per[i].push_back(t[i]);
          agg.push_back(wall);
        }
        for (size_t i = 0; i < ts.size(); ++i) {
          const Stat st = stat(per[i]);
          std::printf("po copy dev=%s card=%u list=%s ordinal=%s shape=%s bytes=%zu copies=%u us=%.1f range=%.1f-%.1f "
                      "gbps=%.3f\n",
                      dv.label.c_str(), rigs[i]->card.index, reg ? "regular" : "immediate", on_copy ? "copy" : "compute",
                      sh.name.c_str(), bytes, sh.count, st.med * 1e6, st.lo * 1e6, st.hi * 1e6, bytes / st.med / 1e9);
        }
        if (ts.size() > 1) {
          const Stat st = stat(agg);
          std::printf("po copy dev=%s card=all list=%s ordinal=%s shape=%s bytes=%zu us=%.1f aggregate_gbps=%.3f\n",
                      dv.label.c_str(), reg ? "regular" : "immediate", on_copy ? "copy" : "compute", sh.name.c_str(),
                      bytes * ts.size(), st.med * 1e6, bytes * ts.size() / st.med / 1e9);
        }
      }
    }
  // first-byte latency: one 2.6 MB copy and one 4-byte copy, submit to fence, on each ordinal
  for (auto& r : rigs)
    for (int on_copy = 0; on_copy < 2; ++on_copy) {
      if (on_copy && r->card.ord.copy < 0) continue;
      for (const CopyShape& sh : {CopyShape{"latency_expert", s.expert, 1}, CopyShape{"latency_4b", 4, 1}}) {
        CopyTimer t(*r, sh, true, on_copy ? uint32_t(r->card.ord.copy) : r->card.ord.compute);
        std::vector<double> v;
        for (int i = 0; i < 25; ++i) v.push_back(t.once());
        v.erase(v.begin(), v.begin() + 5);
        const Stat st = stat(v);
        std::printf("po copy dev=%s card=%u list=regular ordinal=%s shape=%s bytes=%zu latency_us=%.1f range=%.1f-%.1f\n",
                    dv.label.c_str(), r->card.index, on_copy ? "copy" : "compute", sh.name.c_str(), sh.bytes, st.med * 1e6,
                    st.lo * 1e6, st.hi * 1e6);
      }
    }
  return 0;
}

int hold_mode() {
  Devices dv = open_devices(g_opt.dev);
  std::vector<std::unique_ptr<CopyRig>> rigs;
  for (auto& c : dv.cards) rigs.push_back(std::make_unique<CopyRig>(*c, *dv.c0));
  const CopyShape sh{"1gib", size_t(1) << 30, 1};
  std::vector<std::unique_ptr<CopyTimer>> ts;
  for (auto& r : rigs)
    ts.push_back(std::make_unique<CopyTimer>(*r, sh, true, r->card.ord.copy >= 0 ? uint32_t(r->card.ord.copy) : r->card.ord.compute));
  std::vector<uint32_t> n(ts.size(), 0);
  std::vector<double> busy(ts.size(), 0.0);
  std::vector<std::string> err(ts.size());
  const double end = now_s() + g_opt.seconds;
  StartGate gate(ts.size());
  std::vector<std::thread> th;
  for (size_t i = 0; i < ts.size(); ++i)
    th.emplace_back([&, i] {
      gate.arrive_and_wait();
      try {
        while (now_s() < end) {
          busy[i] += ts[i]->once();
          ++n[i];
        }
      } catch (const std::exception& e) {
        err[i] = e.what();
      }
    });
  for (auto& x : th) x.join();
  for (size_t i = 0; i < ts.size(); ++i) {
    if (!err[i].empty()) throw Hang(err[i]);
    std::printf("po hold dev=%s card=%u seconds=%.1f copies=%u gbps=%.3f\n", dv.label.c_str(), rigs[i]->card.index,
                g_opt.seconds, n[i], n[i] * double(sh.bytes) / busy[i] / 1e9);
  }
  return 0;
}

// --- hostbw (P0.4) ---------------------------------------------------------------------------------------------------
struct Triad {
  std::vector<double> a, b, c;
  explicit Triad(size_t n) : a(n, 0.0), b(n, 1.0), c(n, 2.0) {}
  // one pass over [lo, hi): a = b + 3 c (STREAM's triad)
  void pass(size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) a[i] = b[i] + 3.0 * c[i];
  }
  // seconds for one pass on `threads` threads
  double run(uint32_t threads) {
    const size_t n = a.size();
    std::vector<std::thread> th;
    const double t0 = now_s();
    for (uint32_t t = 0; t < threads; ++t)
      th.emplace_back([&, t] { pass(n * t / threads, n * (t + 1) / threads); });
    for (auto& x : th) x.join();
    return now_s() - t0;
  }
};

int hostbw_mode() {
  const Sizes s = sizes();
  const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
  const size_t n = (size_t(2) << 30) / sizeof(double);
  const uint64_t avail = mem_available();
  if (avail && 3 * n * sizeof(double) + s.layer + kMarginBytes > avail)
    throw Skip("host RAM: the triad's 6 GiB + the host image need MemAvailable - 16 GiB");
  Triad tr(n);
  const double bytes = 3.0 * n * sizeof(double);   // STREAM's count: 24 B an element
  for (uint32_t threads : {1u, hw}) {
    tr.run(threads);
    std::vector<double> v;
    for (uint32_t r = 0; r < g_opt.reps; ++r) v.push_back(tr.run(threads));
    const Stat st = stat(v);
    std::printf("po hostbw dev=none load=triad threads=%u gpus=idle gbps=%.2f range=%.2f-%.2f\n", threads,
                bytes / st.med / 1e9, bytes / st.hi / 1e9, bytes / st.lo / 1e9);
  }
  Devices dv = open_devices(g_opt.dev);
  HostImage host = make_host_image(*dv.c0, s);
  const uint32_t P = g_opt.pairs.back();
  std::vector<std::unique_ptr<Rig>> rigs;
  std::vector<std::unique_ptr<ArmRun>> C;
  for (auto& c : dv.cards) rigs.push_back(std::make_unique<Rig>(*c, s, host, P));
  for (auto& r : rigs) C.push_back(prepare(*r, Arm::C, 0, 1));
  std::vector<ArmRun*> cv;
  for (auto& r : C) cv.push_back(r.get());
  const double host_bytes = double(P) * s.K * s.expert * rigs.size();
  // the loads beside arm C: none, one 64 MiB copy loop, the all-thread triad
  for (int load = 0; load < 3; ++load) {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> moved{0};
    std::vector<std::thread> bg;
    std::vector<char> buf_a, buf_b;
    const double tl0 = now_s();
    if (load == 1) {
      buf_a.assign(size_t(64) << 20, 1);
      buf_b.assign(size_t(64) << 20, 0);
      bg.emplace_back([&] {
        while (!stop.load(std::memory_order_relaxed)) {
          std::memcpy(buf_b.data(), buf_a.data(), buf_a.size());
          moved += buf_a.size() * 2;
        }
      });
    } else if (load == 2) {
      for (uint32_t t = 0; t < hw; ++t)
        bg.emplace_back([&, t] {
          while (!stop.load(std::memory_order_relaxed)) {
            tr.pass(n * t / hw, n * (t + 1) / hw);
            moved += uint64_t(24) * (n * (t + 1) / hw - n * t / hw);
          }
        });
    }
    execute(cv, "hostbw warm-up");
    std::vector<double> v;
    for (uint32_t r = 0; r < g_opt.reps; ++r) v.push_back(execute(cv, "hostbw arm C"));
    stop = true;
    for (auto& x : bg) x.join();
    const double tl = now_s() - tl0;
    const Stat st = stat(v);
    const char* name = load == 0 ? "idle" : load == 1 ? "copy64mib_1thread" : "triad_all_threads";
    std::printf("po hostbw dev=%s load=%s arm=C P=%u cards=%zu zc_aggregate_gbps=%.3f range=%.3f-%.3f cpu_gbps=%.2f\n",
                dv.label.c_str(), name, P, rigs.size(), host_bytes / st.med / 1e9, host_bytes / st.hi / 1e9,
                host_bytes / st.lo / 1e9, load ? double(moved.load()) / tl / 1e9 : 0.0);
  }
  return 0;
}

// --- pin (P0.5) ------------------------------------------------------------------------------------------------------
uint64_t host_tag(uint32_t range, uint64_t page) { return 0x504F48505400000ull ^ (uint64_t(range) << 40) ^ page; }

int pin_mode() {
  const model::Qwen4ExpDesc& md = model::qwen4exp();
  const Sizes s = sizes();
  const uint64_t avail0 = mem_available();
  if (avail0 == 0) throw Skip("no /proc/meminfo: the pinned arms run on the box only");
  double cap = double(avail0 > kMarginBytes ? avail0 - kMarginBytes : 0);
  if (g_opt.max_gb > 0) cap = std::min(cap, g_opt.max_gb * 1e9);
  std::printf("po pin dev=%s mem_available_gib=%.1f cap_gib=%.1f (MemAvailable - 16 GiB%s)\n", g_opt.dev.c_str(),
              avail0 / kGiB, cap / kGiB, g_opt.max_gb > 0 ? ", --max-gb" : "");
  Devices dv = open_devices(g_opt.dev);
  std::vector<std::unique_ptr<l0::Module>> helpers;
  for (auto& c : dv.cards) helpers.push_back(std::make_unique<l0::Module>(c->ctx, kernels::path(kHelperVariant)));
  // every range's page tags read back on every card (the host wrote them)
  const auto check_all = [&](const std::vector<const uint8_t*>& rr, const std::vector<size_t>& bytes) {
    std::string out;
    for (size_t ci = 0; ci < dv.cards.size(); ++ci) {
      Card& c = *dv.cards[ci];
      std::string where;
      size_t bad = 0;
      const long first_bad = page_check(c, *helpers[ci], rr, bytes, host_tag, &where, &bad);
      out += " card" + std::to_string(c.index) + "=" + (bad ? "MISMATCH" : "ok");
      if (bad) {
        ++g_fail;
        std::printf("po pin dev=%s card=%u FAIL %zu pages read back another word (first: global page %ld, %s)\n",
                    dv.label.c_str(), c.index, bad, first_bad, where.c_str());
      }
    }
    return out;
  };
  const auto tag = [&](uint8_t* p, size_t bytes, uint32_t range) {
    for (size_t g = 0; g * kPage < bytes; ++g) *reinterpret_cast<uint64_t*>(p + g * kPage) = host_tag(range, g);
  };
  // 1. one range: doubling from 1 GiB, then bisection to 1 GiB
  {
    double good = 0, bad = 0;
    const auto attempt = [&](double gib) {
      if (gib * kGiB > cap) {
        std::printf("po pin dev=%s single gib=%.0f ok=0 reason=cap\n", dv.label.c_str(), gib);
        return false;
      }
      const uint64_t before = mem_available();
      const double t0 = now_s();
      try {
        l0::Mem m(*dv.c0, l0::MemKind::Host, size_t(gib * kGiB), kPage);
        const double t_alloc = now_s() - t0;
        tag(m.as<uint8_t>(), m.size(), 0);
        const double t_all = now_s() - t0;
        const uint64_t after = mem_available();
        const std::string chk = check_all({m.as<uint8_t>()}, {m.size()});
        std::printf("po pin dev=%s single gib=%.0f ok=1 alloc_s=%.2f s_per_gb=%.4f mem_before_gib=%.1f mem_after_gib=%.1f "
                    "check=%s\n",
                    dv.label.c_str(), gib, t_alloc, t_all / (gib * kGiB / 1e9), before / kGiB, after / kGiB, chk.c_str());
        return true;
      } catch (const l0::Error& e) {
        std::printf("po pin dev=%s single gib=%.0f ok=0 reason=\"%s\" mem_before_gib=%.1f\n", dv.label.c_str(), gib, e.what(),
                    before / kGiB);
        return false;
      }
    };
    for (double gib = 1; ; gib *= 2) {
      if (attempt(gib)) {
        good = gib;
      } else {
        bad = gib;
        break;
      }
    }
    while (bad - good > 1) {   // bisection to 1 GiB
      const double mid = std::floor((good + bad) / 2);
      if (attempt(mid)) good = mid; else bad = mid;
    }
    std::printf("po pin dev=%s single_largest_gib=%.0f first_refused_gib=%.0f\n", dv.label.c_str(), good, bad);
  }
  // 2. 48 per-layer ranges + the PLE's 16 + 16 shards (bf16 scales: decision 7's default)
  {
    struct Range {
      std::string name;
      size_t bytes;
    };
    std::vector<Range> want;
    for (uint32_t l = 0; l < md.layers; ++l) want.push_back({"layer" + std::to_string(l), s.layer});
    const std::vector<uint64_t> primes = loader::q4_ple_primes(md.ple_base, md.ple_heads, 0);
    for (uint32_t h = 0; h < md.ple_heads; ++h) want.push_back({"ple_q" + std::to_string(h), size_t(primes[h]) * md.ple_dim});
    for (uint32_t h = 0; h < md.ple_heads; ++h) want.push_back({"ple_s" + std::to_string(h), size_t(primes[h]) * 2});
    double total_want = 0;
    for (const Range& r : want) total_want += double(r.bytes);
    std::vector<std::unique_ptr<l0::Mem>> got;
    std::vector<const uint8_t*> ptrs;
    std::vector<size_t> bytes;
    const uint64_t before = mem_available();
    const double t0 = now_s();
    double sum = 0;
    std::string stop = "none";
    for (size_t i = 0; i < want.size(); ++i) {
      if (sum + double(want[i].bytes) > cap) {
        stop = want[i].name + "(cap)";
        break;
      }
      try {
        got.push_back(std::make_unique<l0::Mem>(*dv.c0, l0::MemKind::Host, want[i].bytes, kPage));
      } catch (const l0::Error& e) {
        stop = want[i].name + "(\"" + e.what() + "\")";
        break;
      }
      tag(got.back()->as<uint8_t>(), want[i].bytes, uint32_t(i));
      ptrs.push_back(got.back()->as<uint8_t>());
      bytes.push_back(want[i].bytes);
      sum += double(want[i].bytes);
    }
    const double secs = now_s() - t0;
    const uint64_t after = mem_available();
    const std::string chk = ptrs.empty() ? std::string(" none") : check_all(ptrs, bytes);
    std::printf("po pin dev=%s ranges allocated=%zu of %zu bytes=%.0f of %.0f (experts 48 x %zu + PLE 32 shards) stopped_at=%s "
                "s_per_gb=%.4f mem_before_gib=%.1f mem_after_gib=%.1f check=%s\n",
                dv.label.c_str(), got.size(), want.size(), sum, total_want, s.layer, stop.c_str(),
                sum > 0 ? secs / (sum / 1e9) : 0.0, before / kGiB, after / kGiB, chk.c_str());
  }
  return g_fail ? 1 : 0;
}

// --- alias (P0.5) ----------------------------------------------------------------------------------------------------
int alias_mode() {
  const model::Qwen4ExpDesc& md = model::qwen4exp();
  Devices dv = open_devices(g_opt.dev);
  for (auto& c : dv.cards) {
    {   // the allocation history: 18 layers at 21b's per-tensor sizes, the int8 head, the PLE pointer table; freed
      std::vector<std::unique_ptr<l0::Mem>> hist;
      size_t total = 0;
      for (uint32_t l = 0; l < 18; ++l) {
        const loader::Q4LayerBytes b = loader::q4_layer_bytes(md, l);
        for (size_t sz : {b.hc_attn, b.hc_mlp, b.gdn_qkvz, b.gdn_ab, b.gdn_out, b.gdn_small, b.qsa_qkvg, b.qsa_idx, b.qsa_o,
                          b.qsa_small, b.router, b.gate_up, b.down, b.shared_gate_up, b.shared_down, b.ple})
          if (sz) {
            hist.push_back(std::make_unique<l0::Mem>(c->ctx, l0::MemKind::Device, sz, kPage));
            total += sz;
          }
      }
      hist.push_back(std::make_unique<l0::Mem>(c->ctx, l0::MemKind::Device, loader::q4_lm_head_bytes(md, true), kPage));
      hist.push_back(std::make_unique<l0::Mem>(c->ctx, l0::MemKind::Device, 32 * 8));
      total += loader::q4_lm_head_bytes(md, true) + 32 * 8;
      std::printf("po alias dev=%s card=%u history allocations=%zu bytes=%zu (freed before the arena)\n", dv.label.c_str(),
                  c->index, hist.size(), total);
    }
    const size_t bytes = size_t(g_opt.arena_gb * kGiB) / kPage * kPage;
    const uint32_t pages = uint32_t(bytes / kPage);
    l0::Mem arena(c->ctx, l0::MemKind::Device, bytes, kPage);
    l0::Module helper(c->ctx, kernels::path(kHelperVariant));
    l0::Mem dp(c->ctx, l0::MemKind::Device, 8), df(c->ctx, l0::MemKind::Device, 8), dout(c->ctx, l0::MemKind::Device, size_t(pages) * 8);
    const uint64_t base = uint64_t(reinterpret_cast<uintptr_t>(arena.ptr()));
    const uint32_t first[2] = {0, pages};
    c->copy(dp.ptr(), &base, 8, "arena pointer");
    c->copy(df.ptr(), first, 8, "arena pages");
    size_t aliases = 0, corrupt = 0;
    for (uint32_t salt : {0x5A17u, 0xC0DEu}) {
      l0::Kernel kt(helper, "po_page_tag"), kc(helper, "po_page_check");
      kt.group_size(64);
      kt.arg_ptr(0, arena.ptr());
      kt.arg(1, pages);
      kt.arg(2, salt);
      kc.group_size(64);
      kc.arg_ptr(0, dp.ptr());
      kc.arg_ptr(1, df.ptr());
      kc.arg_ptr(2, dout.ptr());
      kc.arg(3, pages);
      kc.arg(4, 1u);
      l0::CmdList l = l0::CmdList::regular(c->ctx, c->ord.compute);
      l.launch(kt, (pages + 63) / 64);
      l.launch(kc, (pages + 63) / 64);
      l.close();
      const double t0 = now_s();
      c->run(l, "alias tag + check");
      const double secs = now_s() - t0;
      std::vector<uint64_t> got(pages);
      c->copy(got.data(), dout.ptr(), got.size() * 8, "alias tags");
      size_t a = 0, k = 0;
      for (uint32_t g = 0; g < pages; ++g) {
        const uint64_t want = 0xA11A500000000000ull ^ (uint64_t(salt) << 32) ^ g;
        if (got[g] == want) continue;
        const uint64_t other = got[g] ^ 0xA11A500000000000ull ^ (uint64_t(salt) << 32);
        if (other < pages) {
          ++a;
          if (a <= 16)
            std::printf("po alias dev=%s card=%u ALIAS salt=0x%x page=%u addr=0x%llx holds page=%llu addr=0x%llx\n",
                        dv.label.c_str(), c->index, salt, g, (unsigned long long)(base + uint64_t(g) * kPage),
                        (unsigned long long)other, (unsigned long long)(base + other * kPage));
        } else {
          ++k;
          if (k <= 16)
            std::printf("po alias dev=%s card=%u CORRUPT salt=0x%x page=%u word=0x%016llx\n", dv.label.c_str(), c->index,
                        salt, g, (unsigned long long)got[g]);
        }
      }
      std::printf("po alias dev=%s card=%u arena_gib=%.1f pages=%u salt=0x%x aliases=%zu corrupt=%zu seconds=%.3f\n",
                  dv.label.c_str(), c->index, bytes / kGiB, pages, salt, a, k, secs);
      aliases += a;
      corrupt += k;
    }
    std::printf("po alias dev=%s card=%u result=%s\n", dv.label.c_str(), c->index, aliases + corrupt ? "FAIL" : "clean");
    if (aliases + corrupt) ++g_fail;
  }
  return g_fail ? 1 : 0;
}

std::vector<uint32_t> parse_list(const std::string& v) {
  std::vector<uint32_t> out;
  size_t i = 0;
  while (i < v.size()) {
    size_t j = v.find(',', i);
    if (j == std::string::npos) j = v.size();
    out.push_back(uint32_t(std::stoul(v.substr(i, j - i))));
    i = j + 1;
  }
  if (out.empty()) throw std::invalid_argument("an empty list");
  return out;
}

int usage() {
  std::fprintf(stderr,
               "usage: probe_offload copy|zc|replay|hostbw|pin|alias|hold [--dev 0|1|both] [--pairs 1,2,7,26] [--reps 5] "
               "[--n 10000] [--replay-pairs 7] [--arena-gb 22] [--seconds 20] [--max-gb G] [--timeout-s 20]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  g_opt.mode = argv[1];
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      const auto val = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--dev") g_opt.dev = val();
      else if (a == "--pairs") g_opt.pairs = parse_list(val());
      else if (a == "--reps") g_opt.reps = uint32_t(std::stoul(val()));
      else if (a == "--n") g_opt.n = uint32_t(std::stoul(val()));
      else if (a == "--replay-pairs") g_opt.replay_pairs = uint32_t(std::stoul(val()));
      else if (a == "--arena-gb") g_opt.arena_gb = std::stod(val());
      else if (a == "--seconds") g_opt.seconds = std::stod(val());
      else if (a == "--max-gb") g_opt.max_gb = std::stod(val());
      else if (a == "--timeout-s") g_opt.timeout_s = std::stod(val());
      else return usage();
    }
    if (g_opt.reps == 0) throw std::invalid_argument("--reps 0");
    std::printf("po info mode=%s dev=%s pairs=%zu reps=%u timeout_s=%.0f\n", g_opt.mode.c_str(), g_opt.dev.c_str(),
                g_opt.pairs.size(), g_opt.reps, g_opt.timeout_s);
    if (g_opt.mode == "copy") return copy_mode();
    if (g_opt.mode == "zc") return zc_mode();
    if (g_opt.mode == "replay") return replay_mode();
    if (g_opt.mode == "hostbw") return hostbw_mode();
    if (g_opt.mode == "pin") return pin_mode();
    if (g_opt.mode == "alias") return alias_mode();
    if (g_opt.mode == "hold") return hold_mode();
    return usage();
  } catch (const Skip& e) {
    std::printf("po %s dev=%s SKIP %s\n", g_opt.mode.c_str(), g_opt.dev.c_str(), e.what());
    return 77;
  } catch (const Hang& e) {
    std::printf("po %s dev=%s FAIL hang: %s\n", g_opt.mode.c_str(), g_opt.dev.c_str(), e.what());
    std::fflush(stdout);
    std::_Exit(1);   // a queue may still be busy: no destructor waits on it
  } catch (const std::exception& e) {
    std::printf("po %s dev=%s FAIL %s\n", g_opt.mode.c_str(), g_opt.dev.c_str(), e.what());
    return 1;
  }
}

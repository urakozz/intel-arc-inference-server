// probe_tp_pf - spec 17c step 1 (plan 17c, Review 2026-11-10 point 3; operator decision D8): the rate of tensor-
// parallel prefill's crossing on two B70s. A TP prefill chunk of C rows adds the two cards' halves after every
// out_proj / o_proj / down: 128 crossings a Qwen3.8 chunk, each C x 5120 fp32 (40 MiB at C = 2048) each way, or half
// that in bf16. The decode probe (17a) measured M <= 8 only; this measures the chunk-sized crossing.
//
//   probe_tp_pf [--rows 1024,2048] [--reps 8] [--rounds 5] [--gemms 24] [--timeout 20]
//
// Prints one `ptpf` line per measurement (key=value), interleaved rounds after a warm-up, the median of --rounds:
//   groups    both cards' command queue groups (the copy-only ordinal the copy arms use)
//   rate      per crossing: engine (compute = zeCommandListAppendMemoryCopy on the compute list, copy = on the
//             copy-only queue group, push = a kernel on the source card storing into the peer, pull = a kernel on the
//             destination card loading from the peer), direction (0>1, 1>0, both at once), rows, dtype: ms a
//             crossing (the wall of --reps back-to-back crossings / reps) and GB/s (a direction's bytes / its ms)
//   event     a cross-device event round trip: card 0 signals, card 1's list waits and signals back, --reps times
//   overlap   --gemms pf_gemm launches (2048 x 5120 x 1024, a slab of the per-card linears) on each card's compute
//             list, alone; --reps fp32 crossings both ways on the copy engines, alone; both together - whether the
//             copy engine hides the crossing behind the GEMMs and what each costs the other
//
// Every wait is bounded (--timeout seconds); a miss prints the arm and exits 1. The data is checked once per arm
// (the last crossing's bytes equal the source's). Run under `tools/gpu_lock.sh box --`.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <level_zero/ze_api.h>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/error.h"
#include "l0/memory.h"
#include "l0/sync_event.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {

using Clock = std::chrono::steady_clock;
namespace pf = runtime::prefill;

constexpr uint32_t kHidden = 5120;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}
std::vector<uint32_t> split_u(const std::string& s) {
  std::vector<uint32_t> v;
  size_t a = 0;
  while (a < s.size()) {
    size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    if (b > a) v.push_back(uint32_t(std::stoul(s.substr(a, b - a))));
    a = b + 1;
  }
  return v;
}

uint64_t g_timeout_ns = 20ull * 1000000000ull;

[[noreturn]] void die(const std::string& what) {
  std::printf("ptpf FAIL %s\n", what.c_str());
  std::fflush(stdout);
  std::exit(1);
}

// The copy-only queue group's ordinal (COPY without COMPUTE), or UINT32_MAX.
uint32_t copy_ordinal(ze_device_handle_t dev, uint32_t card) {
  uint32_t n = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &n, nullptr));
  std::vector<ze_command_queue_group_properties_t> p(n);
  for (auto& q : p) q.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &n, p.data()));
  uint32_t found = UINT32_MAX;
  for (uint32_t i = 0; i < n; ++i) {
    const bool compute = p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
    const bool copy = p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    std::printf("ptpf groups card=%u ordinal=%u compute=%d copy=%d queues=%u\n", card, i, int(compute), int(copy),
                p[i].numQueues);
    if (copy && !compute && found == UINT32_MAX) found = i;
  }
  return found;
}

// One card: its compute list (prefill::Context), a copy-engine immediate list, the crossing's buffers and the GEMM
// load's.
struct Card {
  Card(l0::Context& c, uint32_t idx, size_t bytes)
      : ctx(c),
        cx(c),
        kc(c),
        src(c, l0::MemKind::Device, bytes, 4096),
        land(c, l0::MemKind::Device, bytes, 4096),
        a(c, l0::MemKind::Device, size_t(2048) * kHidden * 2, 4096),
        b(c, l0::MemKind::Device, size_t(kHidden) * 1024 * 2, 4096),
        out(c, l0::MemKind::Device, size_t(2048) * 1024 * 4, 4096),
        imm(l0::CmdList::immediate(c)) {
    copy_ord = copy_ordinal(c.device(), idx);
    if (copy_ord != UINT32_MAX) {
      ze_command_queue_desc_t qd{};
      qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
      qd.ordinal = copy_ord;
      qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
      qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
      qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
      ZE_CHECK(zeCommandListCreateImmediate(c.handle(), c.device(), &qd, &bcs));
    }
    // A recognisable source pattern; the landing zeroed.
    std::vector<uint32_t> h(bytes / 4);
    for (size_t i = 0; i < h.size(); ++i) h[i] = uint32_t(i * 2654435761u) ^ (idx << 31);
    imm.copy(src.ptr(), h.data(), bytes);
    imm.fill(land.ptr(), 0u, bytes);
    imm.fill(a.ptr(), 0x3c003c00u, a.size());   // bf16 1.0-ish words: the GEMM's values do not matter
    imm.fill(b.ptr(), 0x3c003c00u, b.size());
  }
  ~Card() {
    if (bcs) {
      zeCommandListHostSynchronize(bcs, UINT64_MAX);
      zeCommandListDestroy(bcs);
    }
  }
  void bcs_copy(void* dst, const void* s, size_t n) {
    ZE_CHECK(zeCommandListAppendMemoryCopy(bcs, dst, s, n, nullptr, 0, nullptr));
  }
  bool bcs_wait() { return zeCommandListHostSynchronize(bcs, g_timeout_ns) == ZE_RESULT_SUCCESS; }
  void kernel_copy(const void* s, void* d, size_t n) {
    const uint32_t n4 = uint32_t(n / 16);
    cx.launch(kc("ptpf_kernels", "ptpf_copy"), 1024, 1, 1, {pf::PtrArg(s), pf::PtrArg(d), pf::arg_val(n4)});
  }
  void gemm() {
    const pf::GemmBatch g{2048, kHidden, 1024, 1, kHidden, 1024, 1024, 0, 0, 0};
    pf::gemm_l0(cx, kc, g, a.as<uint16_t>(), b.as<uint16_t>(), out.as<float>(), false);
  }

  l0::Context& ctx;
  pf::Context cx;
  pf::KernelCache kc;
  l0::Mem src, land;
  l0::Mem a, b, out;
  l0::CmdList imm;
  uint32_t copy_ord = UINT32_MAX;
  ze_command_list_handle_t bcs = nullptr;
};

struct Probe {
  std::array<Card*, 2> c{};
  uint32_t reps = 8, rounds = 5;

  void wait_compute(uint32_t r, const char* arm) {
    if (!c[r]->cx.wait_for(g_timeout_ns)) die(std::string(arm) + ": card " + std::to_string(r) + " compute list");
  }
  void wait_copy(uint32_t r, const char* arm) {
    if (!c[r]->bcs_wait()) die(std::string(arm) + ": card " + std::to_string(r) + " copy list");
  }

  // One crossing of `bytes` from card `from` into card 1 - from's landing, on `engine`.
  void cross(const std::string& engine, uint32_t from, size_t bytes) {
    Card& s = *c[from];
    Card& d = *c[1 - from];
    if (engine == "compute") s.cx.copy(d.land.ptr(), s.src.ptr(), bytes);
    else if (engine == "copy") s.bcs_copy(d.land.ptr(), s.src.ptr(), bytes);
    else if (engine == "push") s.kernel_copy(s.src.ptr(), d.land.ptr(), bytes);
    else if (engine == "pull") d.kernel_copy(s.src.ptr(), d.land.ptr(), bytes);
    else die("unknown engine " + engine);
  }
  // The list a crossing from `from` ran on (pull runs on the destination).
  uint32_t runs_on(const std::string& engine, uint32_t from) const { return engine == "pull" ? 1 - from : from; }

  // reps crossings in each named direction, then the waits: the wall a crossing.
  double time_rate(const std::string& engine, const std::vector<uint32_t>& froms, size_t bytes) {
    const Clock::time_point t0 = Clock::now();
    for (uint32_t i = 0; i < reps; ++i)
      for (uint32_t f : froms) cross(engine, f, bytes);
    for (uint32_t f : froms) {
      const uint32_t r = runs_on(engine, f);
      if (engine == "copy") wait_copy(r, engine.c_str());
      else wait_compute(r, engine.c_str());
    }
    return ms_since(t0) / reps;
  }

  bool check(uint32_t from, size_t bytes) {
    std::vector<uint8_t> a(bytes), b(bytes);
    c[from]->imm.copy(a.data(), c[from]->src.ptr(), bytes);
    c[1 - from]->imm.copy(b.data(), c[1 - from]->land.ptr(), bytes);
    const bool ok = std::memcmp(a.data(), b.data(), bytes) == 0;
    c[1 - from]->imm.fill(c[1 - from]->land.ptr(), 0u, bytes);
    return ok;
  }

  void rate(const std::vector<uint32_t>& rows_list) {
    const std::vector<std::string> engines{"compute", "copy", "push", "pull"};
    const std::vector<std::pair<std::string, std::vector<uint32_t>>> dirs{{"0>1", {0}}, {"1>0", {1}}, {"both", {0, 1}}};
    for (uint32_t rows : rows_list)
      for (uint32_t bpe : {4u, 2u}) {
        const size_t bytes = size_t(rows) * kHidden * bpe;
        for (const auto& [dname, froms] : dirs)
          for (const std::string& e : engines) {
            if (e == "copy" && (c[0]->bcs == nullptr || c[1]->bcs == nullptr)) continue;
            time_rate(e, froms, bytes);   // warm-up
            bool ok = true;
            for (uint32_t f : froms) ok = check(f, bytes) && ok;
            std::vector<double> ms;
            for (uint32_t k = 0; k < rounds; ++k) ms.push_back(time_rate(e, froms, bytes));
            const double m = median(ms);
            std::printf("ptpf rate engine=%s dir=%s rows=%u dtype=%s bytes=%zu ms=%.3f gbps=%.2f check=%s\n",
                        e.c_str(), dname.c_str(), rows, bpe == 4 ? "fp32" : "bf16", bytes, m,
                        double(bytes) / (m * 1e6), ok ? "ok" : "MISMATCH");
            std::fflush(stdout);
            if (!ok) die("rate: the landed bytes differ from the source");
          }
      }
  }

  void event() {
    std::vector<std::unique_ptr<l0::SyncEvent>> e01, e10;
    for (uint32_t i = 0; i < reps; ++i) {
      e01.push_back(std::make_unique<l0::SyncEvent>(c[0]->ctx, std::vector<const l0::Context*>{&c[0]->ctx, &c[1]->ctx}));
      e10.push_back(std::make_unique<l0::SyncEvent>(c[1]->ctx, std::vector<const l0::Context*>{&c[0]->ctx, &c[1]->ctx}));
    }
    auto once = [&] {
      for (uint32_t i = 0; i < reps; ++i) {
        e01[i]->host_reset();
        e10[i]->host_reset();
      }
      const Clock::time_point t0 = Clock::now();
      for (uint32_t i = 0; i < reps; ++i) {
        c[0]->cx.signal(e01[i]->handle());
        c[0]->cx.wait_event(e10[i]->handle());
        c[1]->cx.wait_event(e01[i]->handle());
        c[1]->cx.signal(e10[i]->handle());
      }
      wait_compute(0, "event");
      wait_compute(1, "event");
      return ms_since(t0) * 1000.0 / reps;
    };
    once();
    std::vector<double> us;
    for (uint32_t k = 0; k < rounds; ++k) us.push_back(once());
    std::printf("ptpf event round_trip_us=%.2f reps=%u\n", median(us), reps);
    std::fflush(stdout);
  }

  void overlap(uint32_t gemms) {
    if (c[0]->bcs == nullptr || c[1]->bcs == nullptr) {
      std::printf("ptpf overlap SKIP no copy-only queue group\n");
      return;
    }
    const size_t bytes = size_t(2048) * kHidden * 4;
    auto run = [&](bool g, bool x) {
      const Clock::time_point t0 = Clock::now();
      if (x)
        for (uint32_t i = 0; i < reps; ++i)
          for (uint32_t f : {0u, 1u}) cross("copy", f, bytes);
      if (g)
        for (uint32_t i = 0; i < gemms; ++i)
          for (uint32_t r : {0u, 1u}) c[r]->gemm();
      double tg = 0, tx = 0;
      if (g) {
        wait_compute(0, "overlap");
        wait_compute(1, "overlap");
        tg = ms_since(t0);
      }
      if (x) {
        wait_copy(0, "overlap");
        wait_copy(1, "overlap");
        tx = ms_since(t0);
      }
      return std::make_pair(tg, tx);
    };
    run(true, true);   // warm-up
    std::vector<double> g_alone, x_alone, g_both, x_both, wall_both;
    for (uint32_t k = 0; k < rounds; ++k) {
      g_alone.push_back(run(true, false).first);
      x_alone.push_back(run(false, true).second);
      const auto b = run(true, true);
      g_both.push_back(b.first);
      x_both.push_back(b.second);
      wall_both.push_back(std::max(b.first, b.second));
    }
    std::printf("ptpf overlap gemms=%u crossings=%u gemm_alone_ms=%.2f copy_alone_ms=%.2f gemm_with_copy_ms=%.2f "
                "copy_with_gemm_ms=%.2f wall_both_ms=%.2f sum_alone_ms=%.2f\n",
                gemms, reps, median(g_alone), median(x_alone), median(g_both), median(x_both), median(wall_both),
                median(g_alone) + median(x_alone));
    std::fflush(stdout);
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<uint32_t> rows{1024, 2048};
  uint32_t reps = 8, rounds = 5, gemms = 24;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--rows") rows = split_u(val());
    else if (a == "--reps") reps = uint32_t(std::stoul(val()));
    else if (a == "--rounds") rounds = uint32_t(std::stoul(val()));
    else if (a == "--gemms") gemms = uint32_t(std::stoul(val()));
    else if (a == "--timeout") g_timeout_ns = uint64_t(std::stoul(val())) * 1000000000ull;
    else {
      std::fprintf(stderr, "usage: probe_tp_pf [--rows 1024,2048] [--reps 8] [--rounds 5] [--gemms 24] [--timeout 20]\n");
      return 2;
    }
  }
  try {
    if (l0::Context::gpu_count() < 2) die("two visible GPUs are needed (run under tools/gpu_lock.sh box)");
    l0::Context d0(0);
    l0::Context d1(d0, 1);
    if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0)) die("no peer access both ways");
    const size_t max_bytes = size_t(*std::max_element(rows.begin(), rows.end())) * kHidden * 4;
    Card c0(d0, 0, max_bytes), c1(d1, 1, max_bytes);
    Probe p;
    p.c = {&c0, &c1};
    p.reps = reps;
    p.rounds = rounds;
    std::printf("ptpf info cards=%s,%s reps=%u rounds=%u\n", d0.name().c_str(), d1.name().c_str(), reps, rounds);
    p.event();
    p.rate(rows);
    p.overlap(gemms);
    std::printf("ptpf DONE\n");
  } catch (const std::exception& e) {
    die(e.what());
  }
  return 0;
}

// Does the int4->bf16 dequant hide behind the GEMM? (spec 2, plan 6c Task 4,
// promoted to a standalone probe by the controller's probe-first ruling.)
//
// The question is NOT whether double-buffering removes a data dependency --
// it obviously does -- but whether the driver runs an L0 immediate command
// list and a SYCL in-order queue CONCURRENTLY on one compute engine. Two
// queues on one engine may serialise; this probe measures whether they do.
// A measured recovery of 0.0 is a finding, not a failure.
//
// Pre-registration: docs/probe-dequant-overlap-2026-09-05.md (recovers
// 0.6-0.9 of the dequant time; accept at >= 0.3), committed before this file.
//
// 2026-09-05 EXTENSION (PROBE B, the same document's second dated section).
// The battery below ran the dequant on prefill::Context's own immediate list --
// src/sycl/context.cc creates it with `qd.ordinal = 0` and no `index` -- and
// the GEMM on a sycl::queue whose ordinal this project has never inspected. So
// the verdict "the two queues serialise" is strictly "two queues that may both
// be queue 0 of group 0 serialise". `--ordinal O [--index I]` binds the
// dequant to a DIFFERENT hardware queue and re-runs the identical battery;
// `--enumerate-only` prints the queue groups and stops (no timing at all).
// With no arguments the original single-list path runs unchanged, so the
// record in that document stays reproducible from this same binary.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"

namespace {

// gate|up -- the largest prefill linear and the one whose dequant P3 measured
// at 1.5393 ms. Layout 0 is what the model selects for this matrix.
constexpr uint32_t kK = 5120;
constexpr uint32_t kN = 34816;
constexpr uint32_t kM = 2048;  // PrefillScratch::kC (ruling A13)
constexpr size_t kScratchBytes = size_t(kK) * kN * sizeof(uint16_t);  // 356,515,840
constexpr int kLinears = 8;    // linears per battery
constexpr int kReplays = 8;
constexpr int kDropped = 3;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// 8 replays, first 3 discarded, median of the last 5 -- P2's and P3's protocol.
template <class F>
double replay(F&& body) {
  body();  // discarded warm-up
  std::vector<double> samples;
  for (int r = 0; r < kReplays; ++r) {
    const auto begin = std::chrono::steady_clock::now();
    body();
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (r >= kDropped) samples.push_back(ms);
  }
  return median(std::move(samples));
}

std::vector<uint32_t> host_words(const void* p, size_t bytes) {
  std::vector<uint32_t> out(bytes / sizeof(uint32_t));
  std::memcpy(out.data(), p, bytes);
  return out;
}

// --- PROBE B: the queue groups, read verbatim from the driver ---------------
// This function creates nothing, submits nothing and times nothing. Whatever
// zeDeviceGetCommandQueueGroupProperties reports IS the record; the probe does
// not assert in advance how many groups a B70 has.
std::vector<ze_command_queue_group_properties_t> queue_groups(l0::Context& ctx) {
  uint32_t n = 0;
  if (zeDeviceGetCommandQueueGroupProperties(ctx.device(), &n, nullptr) != ZE_RESULT_SUCCESS)
    return {};
  std::vector<ze_command_queue_group_properties_t> p(n);
  for (auto& g : p) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  if (zeDeviceGetCommandQueueGroupProperties(ctx.device(), &n, p.data()) != ZE_RESULT_SUCCESS)
    return {};
  return p;
}

std::string decode_flags(ze_command_queue_group_property_flags_t f) {
  std::string s;
  auto add = [&](ze_command_queue_group_property_flag_t bit, const char* name) {
    if (f & bit) { if (!s.empty()) s += " | "; s += name; }
  };
  add(ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE, "COMPUTE");
  add(ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY, "COPY");
  add(ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS, "COOPERATIVE_KERNELS");
  add(ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS, "METRICS");
  if (s.empty()) s = "(none)";
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  // --- PROBE B's arguments. No argument = the original battery, unchanged. ---
  int opt_ordinal = -1;   // -1 = use prefill::Context's own list (the original)
  uint32_t opt_index = 0;
  bool enumerate_only = false;
  bool auto_bind = false;  // apply the pre-registered binding rule
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--enumerate-only") enumerate_only = true;
    else if (a == "--auto") auto_bind = true;
    else if (a == "--ordinal" && i + 1 < argc) opt_ordinal = std::atoi(argv[++i]);
    else if (a == "--index" && i + 1 < argc) opt_index = uint32_t(std::atoi(argv[++i]));
    else {
      std::fprintf(stderr,
                   "usage: %s [--enumerate-only] [--auto | --ordinal O [--index I]]\n", argv[0]);
      return 2;
    }
  }

  l0::Context ctx(0);
  std::printf("# P-overlap: does the dequant hide behind the GEMM? (measured, iterate-grade)\n");
  std::printf("# L0 device: %s\n", ctx.name().c_str());

  // --- PROBE B step 1: enumerate. Read-only, printed verbatim. --------------
  const std::vector<ze_command_queue_group_properties_t> groups = queue_groups(ctx);
  std::printf("\n## zeDeviceGetCommandQueueGroupProperties (measured, verbatim)\n");
  std::printf("| ordinal | flags | numQueues | maxMemoryFillPatternSize |\n|---:|---|---:|---:|\n");
  int compute_groups = 0, first_compute = -1, second_compute = -1;
  for (uint32_t i = 0; i < groups.size(); ++i) {
    std::printf("| %u | %s | %u | %zu |\n", i, decode_flags(groups[i].flags).c_str(),
                groups[i].numQueues, size_t(groups[i].maxMemoryFillPatternSize));
    if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ++compute_groups;
      if (first_compute < 0) first_compute = int(i);
      else if (second_compute < 0) second_compute = int(i);
    }
  }
  std::printf("compute groups: %d; first compute ordinal: %d; second compute ordinal: %d\n",
              compute_groups, first_compute, second_compute);

  // --- PROBE B step 2: the binding rule, fixed in the pre-registration ------
  // 1. a second COMPUTE group  -> that ordinal, index 0
  // 2. else numQueues >= 2     -> the compute ordinal, index 1
  // 3. else                    -> one compute queue on the device; dead by
  //                               enumeration, and the probe says so.
  if (auto_bind) {
    if (second_compute >= 0) {
      opt_ordinal = second_compute; opt_index = 0;
      std::printf("binding rule 1: a second COMPUTE group exists -> ordinal %d index %u\n",
                  opt_ordinal, opt_index);
    } else if (first_compute >= 0 && groups[size_t(first_compute)].numQueues >= 2) {
      opt_ordinal = first_compute; opt_index = 1;
      std::printf("binding rule 2: one COMPUTE group with numQueues = %u -> ordinal %d index %u\n",
                  groups[size_t(first_compute)].numQueues, opt_ordinal, opt_index);
    } else {
      std::printf("binding rule 3: exactly ONE compute queue on this device. The alternate "
                  "binding is impossible and the lever is DEAD BY ENUMERATION.\n");
      return 0;
    }
    std::printf("pre-registered branch: a second compute queue EXISTS -> predict recovery "
                ">= 0.5 of D\n");
  } else if (opt_ordinal < 0) {
    std::printf("no --ordinal/--auto: the dequant runs on prefill::Context's own list "
                "(ordinal 0, index 0) -- the original 2026-09-05 battery\n");
  }
  if (enumerate_only) return 0;

  runtime::prefill::Context cx(ctx);

  // The alternate immediate list: the same shape prefill::Context builds
  // (IN_ORDER + ASYNCHRONOUS) on a different ordinal/index. Created here rather
  // than through l0::CmdList::immediate because that helper is SYNCHRONOUS,
  // which would serialise the host on every append and destroy the premise.
  ze_command_list_handle_t alt = nullptr;
  if (opt_ordinal >= 0) {
    ze_command_queue_desc_t qd{};
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.ordinal = uint32_t(opt_ordinal);
    qd.index = opt_index;
    qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    const ze_result_t r = zeCommandListCreateImmediate(ctx.handle(), ctx.device(), &qd, &alt);
    if (r != ZE_RESULT_SUCCESS) {
      std::printf("zeCommandListCreateImmediate(ordinal=%d, index=%u) FAILED: 0x%X -- the "
                  "driver refuses this binding; reported, not worked around.\n",
                  opt_ordinal, opt_index, unsigned(r));
      return 3;
    }
    std::printf("alternate immediate list created on ordinal %d index %u "
                "(IN_ORDER | ASYNCHRONOUS)\n", opt_ordinal, opt_index);
  }
  std::printf("# shape: gate|up K=%u N=%u layout 0; GEMM at M=%u; scratch %zu B x 2\n", kK, kN, kM,
              kScratchBytes);
  std::printf("# %d replays, discard first %d, median of last %d; one discarded warm-up.\n",
              kReplays, kDropped, kReplays - kDropped);

  // --- allocations ----------------------------------------------------------
  // TWO weight copies with DIFFERENT contents. P3 used NB = 2 at this shape to
  // push a timing list past the 24 MB L2; here the second purpose is teeth: if
  // the double-buffered sequence ever let a GEMM read a slot whose dequant had
  // not landed, identical copies would hide it and different ones cannot.
  const common::Int4Gptq w0 = common::Int4Gptq::random(kK, kN, 42);
  const common::Int4Gptq w1 = common::Int4Gptq::random(kK, kN, 43);
  const common::Int4Gptq* weights[2] = {&w0, &w1};

  l0::Mem dq[2] = {l0::Mem(ctx, l0::MemKind::Device, w0.qweight.size() * sizeof(uint32_t)),
                   l0::Mem(ctx, l0::MemKind::Device, w1.qweight.size() * sizeof(uint32_t))};
  l0::Mem ds[2] = {l0::Mem(ctx, l0::MemKind::Device, w0.scales.size() * sizeof(uint16_t)),
                   l0::Mem(ctx, l0::MemKind::Device, w1.scales.size() * sizeof(uint16_t))};
  l0::Mem scratch[2] = {l0::Mem(ctx, l0::MemKind::Device, kScratchBytes),
                        l0::Mem(ctx, l0::MemKind::Device, kScratchBytes)};
  const size_t a_bytes = size_t(kM) * kK * sizeof(uint16_t);
  const size_t c_bytes = size_t(kM) * kN * sizeof(float);
  l0::Mem da(ctx, l0::MemKind::Device, a_bytes);
  l0::Mem dc(ctx, l0::MemKind::Device, c_bytes);
  {
    std::vector<uint16_t> a(size_t(kM) * kK);
    uint32_t s = 0x9e3779b9u;
    for (uint16_t& v : a) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      v = uint16_t((s & 0x807fu) | 0x3e00u);  // exponent 124: |x| in [0.125, 0.25)
    }
    l0::CmdList up = l0::CmdList::immediate(ctx);
    up.copy(da.ptr(), a.data(), a_bytes);
    for (int i = 0; i < 2; ++i) {
      up.copy(dq[i].ptr(), weights[i]->qweight.data(),
              weights[i]->qweight.size() * sizeof(uint32_t));
      up.copy(ds[i].ptr(), weights[i]->scales.data(),
              weights[i]->scales.size() * sizeof(uint16_t));
    }
  }

  l0::Module module(ctx, kernels::path("pf_dequant_tile_K" + std::to_string(kK) + "_N" +
                                       std::to_string(kN) + "_L0_T0"));
  l0::Kernel kernel = module.kernel("pf_dequant_tile");

  // Appends one dequant of weight copy `i&1` into scratch slot `slot` on the L0
  // immediate list. Returns immediately: the list is ASYNCHRONOUS (context.cc),
  // which is the whole premise of the lever.
  //
  // PROBE B: when `alt` exists the append goes there instead. The two paths do
  // exactly the same three things prefill::Context::launch does -- set three
  // pointer arguments, set the kernel's own reqd_work_group_size, append the
  // launch -- so the only difference between them is the command list, which is
  // the whole point of the extension.
  auto append_dequant = [&](int i, int slot) {
    void* w = dq[i & 1].ptr();
    void* s = ds[i & 1].ptr();
    void* out = scratch[slot].ptr();
    if (!alt) {
      cx.launch(kernel, kN / 16, kK / 64, 1,
                {{&w, sizeof w}, {&s, sizeof s}, {&out, sizeof out}});
      return;
    }
    ze_kernel_handle_t k = kernel.handle();
    if (zeKernelSetArgumentValue(k, 0, sizeof w, &w) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(k, 1, sizeof s, &s) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(k, 2, sizeof out, &out) != ZE_RESULT_SUCCESS)
      std::abort();
    ze_kernel_properties_t kp{};
    kp.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    if (zeKernelGetProperties(k, &kp) != ZE_RESULT_SUCCESS) std::abort();
    if (zeKernelSetGroupSize(k, kp.requiredGroupSizeX, kp.requiredGroupSizeY,
                             kp.requiredGroupSizeZ) != ZE_RESULT_SUCCESS)
      std::abort();
    ze_group_count_t g{kN / 16, kK / 64, 1};
    if (zeCommandListAppendLaunchKernel(alt, k, &g, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS)
      std::abort();
  };
  auto submit_gemm = [&](int slot) {
    runtime::prefill::gemm_bf16(cx, {kM, kK, kN}, da.as<uint16_t>(),
                                scratch[slot].as<uint16_t>(), dc.as<float>());
  };
  // Drains BOTH queues. Context::wait() covers the SYCL queue and the Context's
  // own list; the alternate list is this probe's and nothing else waits on it.
  auto wait_all = [&] {
    cx.wait();
    if (alt && zeCommandListHostSynchronize(alt, UINT64_MAX) != ZE_RESULT_SUCCESS) std::abort();
  };

  // --- control A: the dequant alone ----------------------------------------
  const double t_dequant = replay([&] {
                             for (int i = 0; i < kLinears; ++i) append_dequant(i, 0);
                             wait_all();
                           }) /
                           kLinears;
  const size_t deq_read = w0.bytes();
  const double deq_gbps = double(deq_read + kScratchBytes) / (t_dequant * 1e6);

  // --- control B: the GEMM alone -------------------------------------------
  const double t_gemm = replay([&] {
                          for (int i = 0; i < kLinears; ++i) submit_gemm(0);
                          wait_all();
                        }) /
                        kLinears;
  const double gemm_tflops = 2.0 * double(kM) * kK * kN / (t_gemm * 1e9);

  std::printf("\n## controls (measured, iterate-grade)\n");
  std::printf("| control | ms/launch | rate | reference |\n|---|---:|---:|---|\n");
  std::printf("| dequant gate‖up L0 [K][N] | %.3f | %.1f GB/s (r+w) | P3 measured 1.539 ms / "
              "293.1 GB/s |\n", t_dequant, deq_gbps);
  std::printf("| gemm_bf16 gate‖up M=%u | %.3f | %.2f TFLOP/s | P2 measured 150.19 TFLOP/s |\n",
              kM, t_gemm, gemm_tflops);

  // --- battery 1: serial, one scratch (today's ordering) --------------------
  const double t_serial = replay([&] {
    for (int i = 0; i < kLinears; ++i) {
      append_dequant(i, 0);
      wait_all();
      submit_gemm(0);
      wait_all();
    }
  });
  const std::vector<uint32_t> serial_c = [&] {
    l0::Mem host(ctx, l0::MemKind::Host, c_bytes);
    l0::CmdList::immediate(ctx).copy(host.ptr(), dc.ptr(), c_bytes);
    return host_words(host.ptr(), c_bytes);
  }();

  // --- battery 2: double-buffered (plan 6c Task 4 Step 4's exact sequence) ---
  // dequant `i` into slot `i&1` on the L0 list; GEMM `i-1` reading slot
  // `(i-1)&1` on the SYCL queue; then ONE Context::wait() per iteration that
  // drains both. The slot the GEMM reads is never the slot being written.
  const double t_overlap = replay([&] {
    append_dequant(0, 0);
    for (int i = 1; i < kLinears; ++i) {
      append_dequant(i, i & 1);
      submit_gemm((i - 1) & 1);
      wait_all();
    }
    submit_gemm((kLinears - 1) & 1);
    wait_all();
  });
  const std::vector<uint32_t> overlap_c = [&] {
    l0::Mem host(ctx, l0::MemKind::Host, c_bytes);
    l0::CmdList::immediate(ctx).copy(host.ptr(), dc.ptr(), c_bytes);
    return host_words(host.ptr(), c_bytes);
  }();
  const bool bitwise = serial_c == overlap_c;

  // --- battery 3 (diagnostic, NOT the pre-registered cell) -------------------
  // The same dependencies, the same two slots, only the SUBMISSION order
  // swapped: the GEMM goes to the SYCL queue before the dequant is appended to
  // the L0 list. If the device serialises whatever it sees first, this cell
  // moves; if it serialises the two queues outright, it does not. It exists to
  // name the mechanism behind battery 2's number, not to improve it -- the
  // verdict against the pre-registration is battery 2's and stays battery 2's.
  const double t_overlap_gemm_first = replay([&] {
    append_dequant(0, 0);
    for (int i = 1; i < kLinears; ++i) {
      submit_gemm((i - 1) & 1);
      append_dequant(i, i & 1);
      wait_all();
    }
    submit_gemm((kLinears - 1) & 1);
    wait_all();
  });

  // --- the verdict ----------------------------------------------------------
  const double hidden = t_serial - t_overlap;
  const double recovery = hidden / (kLinears * t_dequant);
  std::printf("\n## batteries (%d linears each; measured, iterate-grade)\n", kLinears);
  if (alt)
    std::printf("dequant queue: **ordinal %d index %u** (alternate immediate list); "
                "GEMM queue: the sycl::queue, as before\n", opt_ordinal, opt_index);
  else
    std::printf("dequant queue: prefill::Context's own list (ordinal 0, index 0) -- "
                "the original battery\n");
  std::printf("| battery | waits | ms total | ms/linear |\n|---|---:|---:|---:|\n");
  std::printf("| serial, one scratch | %d | %.3f | %.3f |\n", 2 * kLinears, t_serial,
              t_serial / kLinears);
  std::printf("| double-buffered, two scratches | %d | %.3f | %.3f |\n", kLinears, t_overlap,
              t_overlap / kLinears);
  std::printf("| double-buffered, GEMM submitted first (diagnostic) | %d | %.3f | %.3f |\n",
              kLinears, t_overlap_gemm_first, t_overlap_gemm_first / kLinears);
  std::printf("\ndequant time in the batteries: %d x %.3f = %.3f ms\n", kLinears, t_dequant,
              kLinears * t_dequant);
  std::printf("hidden: %.3f ms; **recovery = %.3f** (pre-registered 0.6-0.9, accept >= 0.3)\n",
              hidden, recovery);
  std::printf("sum vs max model: serial ~ %d x (%.3f + %.3f) = %.3f ms; perfect overlap ~ "
              "%d x max = %.3f ms (both derived)\n",
              kLinears, t_dequant, t_gemm, kLinears * (t_dequant + t_gemm), kLinears,
              kLinears * std::max(t_dequant, t_gemm));
  std::printf("wait-count difference is %d fewer handoffs = %.3f ms at P1's measured 8.569 us "
              "(derived) -- subtract it before crediting overlap\n",
              kLinears, kLinears * 0.008569);
  std::printf("diagnostic (GEMM-first submission order): recovery = %.3f -- reported to name the "
              "mechanism, NOT as the verdict\n",
              (t_serial - t_overlap_gemm_first) / (kLinears * t_dequant));
  std::printf("overlapped battery output vs serial: %s\n",
              bitwise ? "bitwise identical" : "**DIFFER -- this is a race, not a speed-up**");
  if (alt) zeCommandListDestroy(alt);
  return bitwise ? 0 : 1;
}

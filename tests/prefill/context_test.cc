// The prefill execution context: one ze_context, one lazy SYCL side (spec 2.1
// §3.5). Needs a B70 and the `noop` / `pf_probe_chain` binaries and nothing
// else -- no checkpoint, so no ctest label, and it is built and run in EVERY
// configuration: `Context` itself is g++-compiled and links only b70_l0, and
// nothing in this file ever calls `sycl()`, so linking libb70_prefill.so (when
// the component is on) proves has_sycl() still reads false rather than
// exercising the ABI boundary.
//
// **The 9,000-launch chain is the point of this file.** `gdn_chunk` appends ten
// kernels that feed each other through device buffers with no events between
// them, and `Engine::prefill()` will append thousands. Two properties have to
// hold and neither is provable by reading the driver header:
//
//   (a) arguments are resolved at EACH append, not frozen at the first -- the
//       whole reason this path exists instead of a captured list;
//   (b) appended launches execute IN ORDER with no overlap -- what
//       ZE_COMMAND_QUEUE_FLAG_IN_ORDER buys (ze_api.h:3450) and what replaces
//       decode's `M + 3 <= RING` ring-ownership argument.
//
// `pf_chain_step(buf, n, step)` does `buf[i] += step`. Launched 1024 times with
// `step = launch index`, every element must equal sum_{i<1024} i = 523776. A
// frozen argument gives 0; a lost or duplicated increment gives anything else.
//
// **Where the teeth actually are, measured rather than assumed (2026-09-05,
// card 1).** Two mutations were run and reverted:
//
//   * `arg_val(steps[i])` -> `arg_val(steps[0])` in `chain()`: FAILS at line
//     "CHECK_EQ failed: host[i] != kExpected". Property (a) is covered.
//   * `ZE_COMMAND_QUEUE_FLAG_IN_ORDER` removed from src/sycl/context.cc: still
//     PASSES, 523776. This driver already dispatches a legacy immediate list to
//     one hardware queue in submission order, so this test does NOT distinguish
//     the flag being set from it being absent, and must not be cited as proof
//     that it is. The flag stays for the documented guarantee; src/sycl/
//     context.cc says so in the same words.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace {
constexpr uint32_t kN = 4096;        // elements in the chained buffer
// 9,000 is the margin (spec §8 risk 7): S3's finished L0 backend will append
// 8,689 launches per chunk between two waits, and this chain has to cover that
// without the pass-2 doubling (2 x kExpected = 80,991,000) overflowing uint32_t.
constexpr uint32_t kLaunches = 9000;
constexpr uint32_t kExpected = kLaunches * (kLaunches - 1) / 2;  // 40495500

// One pass of `kLaunches` appends, each with its own `step` argument. Every
// `step` lives in a vector that outlives the loop: `zeKernelSetArgumentValue`
// copies at append time, so a loop variable would in fact be fine, but the
// vector makes the lifetime rule visible rather than incidental.
void chain(runtime::prefill::Context& cx, l0::Kernel& k, void* buf) {
  using runtime::prefill::arg_val;
  using runtime::prefill::PtrArg;
  std::vector<uint32_t> steps(kLaunches);
  const uint32_t n = kN;
  for (uint32_t i = 0; i < kLaunches; ++i) {
    steps[i] = i;
    cx.launch(k, kN / 256, 1, 1, {PtrArg(buf), arg_val(n), arg_val(steps[i])});
  }
  cx.wait();
}
}  // namespace

int main() {
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  CHECK(cx.ze_context() == ctx.handle());
  CHECK(cx.ze_device() == ctx.device());
  CHECK(cx.l0_list() != nullptr);
  CHECK_EQ(cx.launches(), size_t{0});
  cx.wait();
  CHECK(!cx.has_sycl());   // a pure-L0 Context never builds the SYCL side

  l0::Module mod(ctx, kernels::path("noop"));
  l0::Kernel k = mod.kernel("noop");
  l0::Mem out(ctx, l0::MemKind::Shared, 64);
  out.as<uint32_t>()[0] = 0;
  void* out_ptr = out.ptr();
  cx.launch(k, 1, 1, 1, {{&out_ptr, sizeof out_ptr}});
  cx.wait();
  CHECK_EQ(out.as<uint32_t>()[0], 42u);
  CHECK_EQ(cx.launches(), size_t{1});

  // --- the module/kernel cache ----------------------------------------------
  runtime::prefill::KernelCache kc(ctx);
  l0::Kernel& chain_k = kc.get(kernels::pf_probe_chain_variant(), "pf_chain_step");
  CHECK_EQ(kc.modules(), size_t{1});
  CHECK_EQ(kc.kernels(), size_t{1});
  // Asking again returns THE SAME object: one binary load per variant for a
  // chunk that runs the same ten kernels over 48 layers.
  CHECK(&kc(kernels::pf_probe_chain_variant(), "pf_chain_step") == &chain_k);
  CHECK_EQ(kc.modules(), size_t{1});
  CHECK_EQ(kc.kernels(), size_t{1});

  // --- (a) per-launch arguments and (b) in-order execution ------------------
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem buf(ctx, l0::MemKind::Device, size_t{kN} * sizeof(uint32_t));
  imm.fill(buf.ptr(), 0u, buf.size());
  std::vector<uint32_t> host(kN);

  chain(cx, chain_k, buf.ptr());
  imm.copy(host.data(), buf.ptr(), buf.size());
  for (uint32_t i = 0; i < kN; ++i) CHECK_EQ(host[i], kExpected);
  CHECK_EQ(cx.launches(), size_t{1} + kLaunches);
  std::printf("chain pass 1: every element = %u (= sum_{i<%u} i)\n", kExpected, kLaunches);

  // The same 9,000 launches again on the same Context and the same cached
  // Kernel: the value must DOUBLE, which proves the cache is reusable and that
  // a second pass does not inherit the first pass's frozen arguments.
  chain(cx, chain_k, buf.ptr());
  imm.copy(host.data(), buf.ptr(), buf.size());
  for (uint32_t i = 0; i < kN; ++i) CHECK_EQ(host[i], 2u * kExpected);
  CHECK_EQ(cx.launches(), size_t{1} + 2 * kLaunches);
  std::printf("chain pass 2: every element = %u\n", 2u * kExpected);

  cx.reset_launches();
  CHECK_EQ(cx.launches(), size_t{0});

  // Profiling observes real device work and can reuse its event slots after a
  // drain. It must not alter argument updates or count unprofiled launches.
  cx.set_profiling(true);
  for (uint32_t pass = 0; pass < 2; ++pass) {
    const uint32_t n = kN, increment = pass + 1;
    for (uint32_t launch = 0; launch < 300; ++launch)
      cx.launch(chain_k, kN / 256, 1, 1,
              {runtime::prefill::PtrArg(buf.ptr()), runtime::prefill::arg_val(n),
               runtime::prefill::arg_val(increment)});
    cx.wait();
    const auto metrics = cx.take_launch_metrics();
    CHECK_EQ(metrics.launches, size_t{300});
    CHECK(metrics.gpu_ms > 0.0);
    CHECK(metrics.host_submit_ms > 0.0);
    CHECK_EQ(cx.take_launch_metrics().launches, size_t{0});
  }
  imm.copy(host.data(), buf.ptr(), buf.size());
  for (uint32_t i = 0; i < kN; ++i) CHECK_EQ(host[i], 2u * kExpected + 900u);
  cx.set_profiling(false);
  cx.launch(k, 1, 1, 1, {{&out_ptr, sizeof out_ptr}});
  cx.wait();
  CHECK_EQ(cx.take_launch_metrics().launches, size_t{0});

  // Recording must freeze scalar arguments, not buffer contents; capture is
  // not execution, and later immediate launches must not mutate the recording.
  imm.fill(buf.ptr(), 0u, buf.size());
  cx.reset_launches();
  auto recorded = cx.capture([&] {
    for (uint32_t step = 1; step <= 300; ++step)
      cx.launch(chain_k, kN / 256, 1, 1,
                {runtime::prefill::PtrArg(buf.ptr()), runtime::prefill::arg_val(kN),
                 runtime::prefill::arg_val(step)});
  });
  CHECK_EQ(cx.launches(), size_t{0});
  imm.copy(host.data(), buf.ptr(), buf.size());
  for (auto value : host) CHECK_EQ(value, 0u);
  cx.replay(*recorded);
  CHECK_EQ(cx.launches(), size_t{300});
  imm.copy(host.data(), buf.ptr(), buf.size());
  for (auto value : host) CHECK_EQ(value, 45150u);
  imm.fill(buf.ptr(), 7u, buf.size());
  cx.launch(chain_k, kN / 256, 1, 1,
            {runtime::prefill::PtrArg(buf.ptr()), runtime::prefill::arg_val(kN),
             runtime::prefill::arg_val(11u)});
  cx.replay(*recorded);  // orders pending immediate work before regular-list work
  imm.copy(host.data(), buf.ptr(), buf.size());
  for (auto value : host) CHECK_EQ(value, 45168u);
  CHECK_EQ(cx.launches(), size_t{601});
  for (int misuse = 0; misuse < 3; ++misuse) {
    bool threw = false;
    try {
      auto invalid = cx.capture([&] {
        if (misuse == 0) cx.wait();
        if (misuse == 1) (void)cx.sycl();
        if (misuse == 2) (void)cx.capture([] {});
      });
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    cx.wait();  // a failed capture must restore the immediate path
  }
  cx.set_profiling(true);
  bool profile_capture_threw = false;
  try { (void)cx.capture([] {}); }
  catch (const std::exception&) { profile_capture_threw = true; }
  CHECK(profile_capture_threw);
  cx.set_profiling(false);

  // --- misuse is loud -------------------------------------------------------
  // One argument too many: the driver rejects index 3 on a three-argument
  // kernel and the throw names the kernel and the index, not just a hex code.
  {
    bool threw = false;
    const uint32_t n = kN, step = 0, extra = 0;
    try {
      cx.launch(chain_k, kN / 256, 1, 1,
                {runtime::prefill::PtrArg(buf.ptr()), runtime::prefill::arg_val(n),
                 runtime::prefill::arg_val(step), runtime::prefill::arg_val(extra)});
    } catch (const std::exception& e) {
      threw = true;
      const std::string m = e.what();
      std::printf("expected throw: %s\n", m.c_str());
      CHECK(m.find("pf_chain_step") != std::string::npos);
      CHECK(m.find("argument 3") != std::string::npos);
    }
    CHECK(threw);
  }
  // A variant that was never compiled: the message carries the path, because
  // the failure is always a missing add_ocloc_kernel row or a missing
  // add_dependencies, never a run-time condition.
  {
    bool threw = false;
    try {
      kc.get("pf_no_such_variant", "nope");
    } catch (const std::exception& e) {
      threw = true;
      const std::string m = e.what();
      CHECK(m.find("pf_no_such_variant") != std::string::npos);
    }
    CHECK(threw);
  }

  std::printf("context_test OK\n");
  return 0;
}

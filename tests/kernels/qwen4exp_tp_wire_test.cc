// PERF-44: Qwen3.8-Flash-Next tp decode's out_proj / o_proj crossing with the send folded into the gemv
// (gemv_M<M>_K3072_N2560_S2_L0_TP + tp_cross_..._sum_W64) is BITWISE today's unfused path (the plain gemv, tp_send,
// tp_cross_..._sum) on one card: two "cards" A (rank 0) and B (rank 1) on the same device, each its own random int4
// weights and rows, each sending into the other's landing buffer (the fake peer). Per side, after each of two replays
// (call 1 slot 1, call 2 slot 0): the buffer [2][M][2560] byte-identical fused against unfused (slice 0 the sum, slice
// 1 the gemv's own partial - TP_SUM 1 writes slice 0 only, the combine _S1 reads only it), A's sum (slice 0)
// byte-identical to B's (card 0's part first on both), the landing's data words
// identical, the state words clear and every fused counter at the call number (rows M .. 7 kept level). The fused
// landing is sized by the host's tp::Layout at kTpOutFoldG, as Qwen4ExpTpEngine sizes it.
// Arguments: the M values to run (1 always built; 2..4 with B70_MTP). Needs a B70, no checkpoint; exit 77 without one.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/tp_protocol.h"

namespace {
namespace qk = kernels::qwen4exp;
namespace tp = runtime::tp;
constexpr uint32_t kK = 3072, kN = qk::kHidden, kS = qk::kTpOutS, kSpin = 1u << 22;

int failures = 0;
void report(const std::string& what, bool ok) {
  std::printf("  %s%s\n", ok ? "" : "FAIL ", what.c_str());
  if (!ok) ++failures;
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  template <class T>
  l0::Mem up(const std::vector<T>& v) {
    l0::Mem m(ctx, l0::MemKind::Device, v.size() * sizeof(T));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  l0::Mem zeros(size_t bytes, l0::MemKind kind = l0::MemKind::Device) {
    l0::Mem m(ctx, kind, bytes, 4096);
    imm.fill(m.ptr(), 0u, bytes);
    return m;
  }
  std::vector<uint32_t> read(const l0::Mem& m) {
    std::vector<uint32_t> v(m.size() / 4);
    imm.copy(v.data(), m.ptr(), m.size());
    return v;
  }
};

// One side's (card's) buffers for one path: its partials (summed in place), the landing the PEER writes, the send's
// and the sum's counters, the state words.
struct Side {
  l0::Mem part, land, sctr, fctr, state;
  Side(Dev& d, uint32_t M, const tp::Layout& L)
      : part(d.zeros(size_t(kS) * M * kN * 4)),
        land(d.zeros(L.total_bytes())),
        sctr(d.zeros(L.counter_bytes())),
        fctr(d.zeros(L.counter_bytes())),
        state(d.zeros(tp::kStateWords * 4, l0::MemKind::Shared)) {}
};

struct Weights {
  l0::Mem w, sc, x;
};

void set_gemv(l0::Kernel& k, const Weights& W, const Side& me) {
  k.group_size(64);
  k.arg_ptr(0, W.w.ptr());
  k.arg_ptr(1, W.sc.ptr());
  k.arg_ptr(2, W.x.ptr());
  k.arg_ptr(3, me.part.ptr());
}
void set_sum(l0::Kernel& k, const Side& me) {
  k.group_size(qk::kTpWg);
  k.arg_ptr(0, me.part.ptr());
  k.arg_ptr(1, me.land.ptr());
  k.arg_ptr(2, me.part.ptr());
  k.arg_ptr(3, me.fctr.ptr());
  k.arg_ptr(4, me.state.ptr());
  k.arg<uint32_t>(5, kSpin);
}

void run_m(Dev& d, uint32_t M) {
  std::printf("M %u\n", M);
  std::vector<Weights> W;
  W.reserve(2);
  for (uint32_t r = 0; r < 2; ++r) {
    const common::Int4Gptq g = common::Int4Gptq::random(kK, kN, 100 * M + r);
    std::vector<uint16_t> x(size_t(M) * kK);
    std::mt19937 rng(200 * M + r);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);
    for (auto& v : x) v = common::f32_to_bf16(xd(rng));
    W.push_back(Weights{d.up(g.qweight), d.up(g.scales), d.up(x)});
  }
  const tp::Layout lc{kN, kS, tp::kFoldG}, lf{kN, kS, qk::kTpOutFoldG};
  Side c[2] = {Side(d, M, lc), Side(d, M, lc)}, f[2] = {Side(d, M, lf), Side(d, M, lf)};

  l0::Module gm(d.ctx, kernels::path(kernels::gemv_variant(M, kK, kN, kS, 0)));
  l0::Module gtm(d.ctx, kernels::path(qk::tp_out_gemv_variant(M, kK)));
  std::vector<std::unique_ptr<l0::Module>> sm, swm;
  for (uint32_t r = 0; r < 2; ++r) {
    sm.push_back(std::make_unique<l0::Module>(d.ctx, kernels::path(qk::tp_sum_variant(r, M))));
    swm.push_back(std::make_unique<l0::Module>(d.ctx, kernels::path(qk::tp_sum_variant(r, M, true))));
  }
  std::vector<std::unique_ptr<l0::Kernel>> ks;
  const auto mk = [&](l0::Module& m, const char* e) -> l0::Kernel& {
    ks.push_back(std::make_unique<l0::Kernel>(m, e));
    return *ks.back();
  };
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  // unfused: gemv, tp_send into the peer's landing (both sides), then each side's tp_sum
  for (uint32_t r = 0; r < 2; ++r) {
    l0::Kernel& g = mk(gm, "gemv");
    set_gemv(g, W[r], c[r]);
    list.launch(g, kN / 64, kS);
    l0::Kernel& s = mk(*sm[r], "tp_send");
    s.group_size(qk::kTpWg);
    s.arg_ptr(0, c[r].part.ptr());
    s.arg_ptr(1, c[1 - r].land.ptr());
    s.arg_ptr(2, c[r].sctr.ptr());
    s.arg_ptr(3, c[r].part.ptr());
    list.launch(s, qk::kTpFoldG, M);
  }
  for (uint32_t r = 0; r < 2; ++r) {
    l0::Kernel& s = mk(*sm[r], "tp_sum");
    set_sum(s, c[r]);
    list.launch(s, qk::kTpFoldG, M);
  }
  // fused: the send-folded gemv (both sides), then each side's wired tp_sum
  for (uint32_t r = 0; r < 2; ++r) {
    l0::Kernel& g = mk(gtm, "gemv");
    set_gemv(g, W[r], f[r]);
    g.arg_ptr(4, f[1 - r].land.ptr());
    g.arg_ptr(5, f[r].sctr.ptr());
    list.launch(g, kN / 64, kS);
  }
  for (uint32_t r = 0; r < 2; ++r) {
    l0::Kernel& s = mk(*swm[r], "tp_sum");
    set_sum(s, f[r]);
    list.launch(s, qk::kTpFoldG, M);
  }
  list.close();

  for (uint32_t call = 1; call <= 2; ++call) {
    d.q.execute(list, &d.f);
    d.f.wait();
    const std::string at = "call " + std::to_string(call) + ": ";
    const uint32_t slot = call & 1u;
    std::vector<uint32_t> pc[2], pf[2];
    for (uint32_t r = 0; r < 2; ++r) {
      pc[r] = d.read(c[r].part);
      pf[r] = d.read(f[r].part);
      const std::vector<uint32_t> lcw = d.read(c[r].land), lfw = d.read(f[r].land);
      const size_t dw = size_t(kS) * M * kN;   // the call's [S][M][N] in the slot
      const bool land_same = std::equal(lcw.begin() + lc.data(slot), lcw.begin() + lc.data(slot) + dw,
                                        lfw.begin() + lf.data(slot));
      const std::string side = std::string(r ? "B (rank 1)" : "A (rank 0)");
      const bool written = std::any_of(pf[r].begin(), pf[r].begin() + size_t(M) * kN, [](uint32_t v) { return v != 0; });
      report(at + side + ": sum bitwise the unfused path", pc[r] == pf[r] && written);
      report(at + side + ": landing data words bitwise", land_same);
      const std::vector<uint32_t> stc = d.read(c[r].state), stf = d.read(f[r].state);
      report(at + side + ": no wait failed",
             std::all_of(stc.begin(), stc.end(), [](uint32_t v) { return v == 0; }) &&
                 std::all_of(stf.begin(), stf.end(), [](uint32_t v) { return v == 0; }));
      const std::vector<uint32_t> sc = d.read(f[r].sctr), fc = d.read(f[r].fctr);
      report(at + side + ": every fused counter (" + std::to_string(lf.chunks()) + ") at the call",
             std::all_of(sc.begin(), sc.end(), [&](uint32_t v) { return v == call; }) &&
                 std::all_of(fc.begin(), fc.end(), [&](uint32_t v) { return v == call; }));
    }
    const size_t row = size_t(M) * kN;   // slice 0: the sum
    report(at + "A's sum bitwise B's", std::equal(pf[0].begin(), pf[0].begin() + row, pf[1].begin()));
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (l0::Context::gpu_count() < 1) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  Dev d;
  for (int i = 1; i < argc; ++i) run_m(d, uint32_t(std::atoi(argv[i])));
  std::printf("qwen4exp_tp_wire_test: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}

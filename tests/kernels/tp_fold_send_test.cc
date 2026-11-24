// PERF-44 on Qwen3.8 tp: the folded crossing with tp_send as the row-parallel linear's tail (gemv _TF at M 1, gemm_sm
// _tf at the verify rows M 2..8: TP_FOLD_SEND, src/kernels/tp_fold_send.h) is BITWISE today's path (the plain linear,
// tp_send, tp_res_fold) on one card: two "cards" A (rank 0) and B (rank 1) on the same device, each its own random int4
// weights and rows at a TP card's out_proj / o_proj (3072 x 5120) and down (8704 x 5120), S 4, each sending into the
// other's landing buffer (the fake peer), both folding into the same starting residual. After each of three replays
// (slots 1, 0, 1), per side: the residual, the sums of squares, the folded row (`own`) and the call's landing data words
// byte-identical fused against unfused, the residual moved, the state words clear, every call counter at the call and
// every arrival counter at 16 x the call; and A's residual bitwise B's (card 0's part first on both).
// Arguments: the M values to run (1; 2..4 with B70_MTP; 5..8 with B70_VERIFY_M8). Needs a B70, no checkpoint; exit 77
// without one.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "kernels/tp_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/tp_protocol.h"

namespace {
namespace tp = runtime::tp;
constexpr uint32_t kN = 5120, kS = 4, kSpin = 1u << 22, kCalls = 3;
constexpr uint32_t kWgsPerChunk = kN / tp::kFoldG / 64 * kS;   // tp_fold_send.h TPF_WGS

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
    std::vector<uint32_t> v((m.size() + 3) / 4);
    imm.copy(v.data(), m.ptr(), m.size());
    return v;
  }
};

// One side's (card's) buffers for one path: partials, its residual and sums of squares, the landing the PEER writes,
// the send's counters (+ the arrival counters) and the fold's, the state words, the folded row.
struct Side {
  l0::Mem part, resid, sumsq, land, sctr, fctr, state, own;
  Side(Dev& d, uint32_t M, const tp::Layout& L, const std::vector<uint16_t>& r0)
      : part(d.zeros(size_t(kS) * M * kN * 4)),
        resid(d.up(r0)),
        sumsq(d.zeros(size_t(tp::kFoldG) * M * 4)),
        land(d.zeros(L.total_bytes())),
        sctr(d.zeros(L.counter_bytes() + L.fold_g * 4)),
        fctr(d.zeros(L.counter_bytes())),
        state(d.zeros(tp::kStateWords * 4, l0::MemKind::Shared)),
        own(d.zeros(size_t(tp::kMRows) * kN * 4)) {}
};

void run_shape(Dev& d, uint32_t M, uint32_t K) {
  std::printf("M %u, %u x %u\n", M, K, kN);
  const tp::Layout L{kN, 1, tp::kFoldG};
  std::vector<uint16_t> r0(size_t(M) * kN);
  {
    std::mt19937 rng(7 * M + K);
    std::uniform_real_distribution<float> rd(-4.f, 4.f);
    for (auto& v : r0) v = common::f32_to_bf16(rd(rng));
  }
  std::vector<l0::Mem> w, sc, x;
  for (uint32_t r = 0; r < 2; ++r) {
    const common::Int4Gptq g = common::Int4Gptq::random(K, kN, 100 * M + 10 * r + K);
    std::vector<uint16_t> xv(size_t(M) * K);
    std::mt19937 rng(200 * M + r + K);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);
    for (auto& v : xv) v = common::f32_to_bf16(xd(rng));
    w.push_back(d.up(g.qweight));
    sc.push_back(d.up(g.scales));
    x.push_back(d.up(xv));
  }
  std::vector<std::unique_ptr<Side>> c, f;   // unfused, fused
  for (uint32_t r = 0; r < 2; ++r) {
    c.push_back(std::make_unique<Side>(d, M, L, r0));
    f.push_back(std::make_unique<Side>(d, M, L, r0));
  }

  // The linear the capture binds at M (gemv at 1, gemm_sm at the verify rows), plain and with the send as its tail.
  const kernels::SmallMTuning* t = M > 1 ? kernels::small_m_tuning(K, kN, kS, 0) : nullptr;
  if (M > 1 && !t) {
    report("no gemm_sm row for " + std::to_string(K) + " x " + std::to_string(kN), false);
    return;
  }
  const std::string lin = t ? kernels::gemm_sm_variant(M, *t) : kernels::gemv_variant(M, K, kN, kS, 0);
  const char* entry = t ? "gemm_sm" : "gemv";
  l0::Module lm(d.ctx, kernels::path(lin));
  l0::Module lfm(d.ctx, kernels::path(kernels::tp_fold_send_variant(lin, t != nullptr)));
  std::vector<std::unique_ptr<l0::Module>> xm;
  for (uint32_t r = 0; r < 2; ++r)
    xm.push_back(std::make_unique<l0::Module>(d.ctx, kernels::path(kernels::tp_cross_variant(M, kN, tp::kFoldG, kS, true, r))));
  std::vector<std::unique_ptr<l0::Kernel>> ks;
  const auto mk = [&](l0::Module& m, const char* e, uint32_t wg) -> l0::Kernel& {
    ks.push_back(std::make_unique<l0::Kernel>(m, e));
    ks.back()->group_size(wg);
    return *ks.back();
  };
  const auto linear = [&](l0::Module& m, uint32_t r, const Side& me) -> l0::Kernel& {
    l0::Kernel& k = mk(m, entry, 64);
    k.arg_ptr(0, w[r].ptr());
    k.arg_ptr(1, sc[r].ptr());
    k.arg_ptr(2, x[r].ptr());
    k.arg_ptr(3, me.part.ptr());
    return k;
  };
  const auto fold = [&](uint32_t r, const Side& me) -> l0::Kernel& {
    l0::Kernel& k = mk(*xm[r], "tp_res_fold", 256);
    k.arg_ptr(0, me.own.ptr());
    k.arg_ptr(1, me.land.ptr());
    k.arg_ptr(2, me.resid.ptr());
    k.arg_ptr(3, me.sumsq.ptr());
    k.arg_ptr(4, me.fctr.ptr());
    k.arg_ptr(5, me.state.ptr());
    k.arg<uint32_t>(6, kSpin);
    return k;
  };
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  for (uint32_t r = 0; r < 2; ++r) {   // unfused: the linear, tp_send into the peer's landing
    list.launch(linear(lm, r, *c[r]), kN / 64, kS);
    l0::Kernel& s = mk(*xm[r], "tp_send", 256);
    s.arg_ptr(0, c[r]->part.ptr());
    s.arg_ptr(1, c[1 - r]->land.ptr());
    s.arg_ptr(2, c[r]->sctr.ptr());
    s.arg_ptr(3, c[r]->own.ptr());
    list.launch(s, tp::kFoldG, M);
  }
  for (uint32_t r = 0; r < 2; ++r) list.launch(fold(r, *c[r]), tp::kFoldG, M);
  for (uint32_t r = 0; r < 2; ++r) {   // fused: the linear with the send as its tail
    l0::Kernel& k = linear(lfm, r, *f[r]);
    k.arg_ptr(4, f[1 - r]->land.ptr());
    k.arg_ptr(5, f[r]->sctr.ptr());
    k.arg_ptr(6, f[r]->own.ptr());
    list.launch(k, kN / 64, kS);
  }
  for (uint32_t r = 0; r < 2; ++r) list.launch(fold(r, *f[r]), tp::kFoldG, M);
  list.close();

  std::vector<uint32_t> r0w((r0.size() + 1) / 2);
  std::memcpy(r0w.data(), r0.data(), r0.size() * 2);
  for (uint32_t call = 1; call <= kCalls; ++call) {
    d.q.execute(list, &d.f);
    d.f.wait();
    const std::string at = "call " + std::to_string(call) + ": ";
    const uint32_t slot = call & 1u;
    std::vector<uint32_t> rf[2];
    for (uint32_t r = 0; r < 2; ++r) {
      const std::string side = at + (r ? "B (rank 1)" : "A (rank 0)");
      const std::vector<uint32_t> rc = d.read(c[r]->resid);
      rf[r] = d.read(f[r]->resid);
      report(side + ": residual bitwise the unfused path", rc == rf[r] && (call > 1 || rf[r] != r0w));
      report(side + ": sums of squares bitwise", d.read(c[r]->sumsq) == d.read(f[r]->sumsq));
      const std::vector<uint32_t> oc = d.read(c[r]->own), of = d.read(f[r]->own);
      report(side + ": folded row bitwise", std::equal(oc.begin(), oc.begin() + size_t(M) * kN, of.begin()));
      const std::vector<uint32_t> lc = d.read(c[r]->land), lf = d.read(f[r]->land);
      report(side + ": landing data words bitwise",
             std::equal(lc.begin() + L.data(slot), lc.begin() + L.data(slot) + size_t(M) * kN, lf.begin() + L.data(slot)));
      const std::vector<uint32_t> stc = d.read(c[r]->state), stf = d.read(f[r]->state);
      report(side + ": no wait failed", std::all_of(stc.begin(), stc.end(), [](uint32_t v) { return v == 0; }) &&
                                            std::all_of(stf.begin(), stf.end(), [](uint32_t v) { return v == 0; }));
      const std::vector<uint32_t> s = d.read(f[r]->sctr), fc = d.read(f[r]->fctr);
      const uint32_t J = L.chunks();
      report(side + ": every call counter at the call, every arrival counter at " +
                 std::to_string(kWgsPerChunk * call),
             std::all_of(s.begin(), s.begin() + J, [&](uint32_t v) { return v == call; }) &&
                 std::all_of(s.begin() + J, s.end(), [&](uint32_t v) { return v == kWgsPerChunk * call; }) &&
                 std::all_of(fc.begin(), fc.end(), [&](uint32_t v) { return v == call; }));
    }
    report(at + "A's residual bitwise B's", rf[0] == rf[1]);
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (l0::Context::gpu_count() < 1) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  Dev d;
  for (int i = 1; i < argc; ++i)
    for (uint32_t K : {3072u, 8704u}) run_shape(d, uint32_t(std::atoi(argv[i])), K);
  std::printf("tp_fold_send_test: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}

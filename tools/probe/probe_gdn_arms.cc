// probe_gdn_arms - Ornith's GDN prefill kernels against candidate builds of the same entries, on REAL inputs.
//
// Inputs: a B70_PF_GDN_DUMP directory (runtime/prefill/gdn.cc: gdn<call>_{xb,g,beta}.bin + .txt - the l2normed
// xb, g_cum and beta of one GDN layer's chunk). Per dump:
//   * production pf_gdn_intra makes w, u and A2; production pf_gdn_scan_dpas_split runs once from a zero state
//     to give a realistic starting state S0, then the timed scan runs from S0 (o and the final state);
//   * every arm runs on the same inputs; its outputs must equal production's word for word (intra: w, u, A2;
//     scan: o and the final state), or the arm is out;
//   * time: arm and production as interleaved pairs after a warm-up, the order alternating each round, median
//     of `--rounds`; global timestamps around the one launch (the state reset is outside them).
//
//   probe_gdn_arms --dump DIR [--calls 0,9,21] [--rounds 15]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/prefill/pf_kernels.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gdn_a2_pack.h"
#include "runtime/prefill/kernels.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;
constexpr uint32_t kDim = 128, kCt = 64;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
template <class T>
std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot read " + path);
  const size_t n = size_t(f.tellg());
  std::vector<T> v(n / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n));
  return v;
}

// The candidate binaries (tools/probe/CMakeLists.txt): {kind, binary, entry}.
struct Arm {
  const char* kind;   // "scan" or "intra"
  const char* bin;
  const char* entry;
};
const Arm kArms[] = {
    {"scan", "pf_gdn_scan_pf1_GK16V32", "pf_gdn_scan_dpas_split"},
    {"scan", "pf_gdn_scan_pf2_GK16V32", "pf_gdn_scan_dpas_split"},
    {"scan", "pf_gdn_scan_cvt_GK16V32", "pf_gdn_scan_dpas_split"},
    {"intra", "pf_gdn_intra_cvt_GK16V32", "pf_gdn_intra"},
    // DIAGNOSTIC (outputs incomplete, never bitwise): the intra kernel's stages priced by stopping early
    {"intra", "pf_gdn_intra_stop1_GK16V32", "pf_gdn_intra"},
    {"intra", "pf_gdn_intra_stop2_GK16V32", "pf_gdn_intra"},
    {"intra", "pf_gdn_intra_stop3_GK16V32", "pf_gdn_intra"},
    {"intra", "pf_gdn_intra_stop4_GK16V32", "pf_gdn_intra"},
};

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string dir;
    std::vector<int> calls = {0, 9, 21};
    int rounds = 15;
    std::string only;
    uint32_t tile = 1;   // --tile N: the dump's rows repeated N times (C = N x the dump's rows)
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--dump" && i + 1 < argc) dir = argv[++i];
      else if (a == "--rounds" && i + 1 < argc) rounds = std::max(1, std::atoi(argv[++i]));
      else if (a == "--only" && i + 1 < argc) only = argv[++i];
      else if (a == "--tile" && i + 1 < argc) tile = std::max(1, std::atoi(argv[++i]));
      else if (a == "--calls" && i + 1 < argc) {
        calls.clear();
        std::stringstream ss(argv[++i]);
        std::string tok;
        while (std::getline(ss, tok, ',')) calls.push_back(std::atoi(tok.c_str()));
      } else {
        std::fprintf(stderr, "usage: probe_gdn_arms --dump DIR [--calls 0,9,21] [--rounds 15] [--only BIN] [--tile N]\n");
        return 2;
      }
    }
    if (dir.empty()) throw std::runtime_error("--dump DIR is required");
    l0::Context ctx(0);
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    std::printf("# probe_gdn_arms: ZE_AFFINITY_MASK=%s, device %s, %d rounds\n", mask ? mask : "(unset)",
                ctx.name().c_str(), rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    l0::Mem d_stamp(ctx, l0::MemKind::Host, 64);
    uint64_t* stamp = d_stamp.as<uint64_t>();

    for (int call : calls) {
      const std::string base = dir + "/gdn" + std::to_string(call);
      uint32_t C = 0, hk = 0, H = 0, conv = 0;
      {
        std::ifstream f(base + ".txt");
        if (!(f >> C >> hk >> H >> conv)) throw std::runtime_error("cannot read " + base + ".txt");
      }
      std::vector<uint16_t> xb = read_bin<uint16_t>(base + "_xb.bin");
      std::vector<float> g = read_bin<float>(base + "_g.bin"), beta = read_bin<float>(base + "_beta.bin");
      if (xb.size() != size_t(C) * conv || g.size() != size_t(C) * H || beta.size() != g.size())
        throw std::runtime_error(base + ": sizes do not match the .txt");
      if (C % kCt != 0 && tile > 1) throw std::runtime_error("--tile needs whole 64-row sub-chunks");
      for (auto* v : {&g, &beta}) {
        const std::vector<float> one = *v;
        for (uint32_t r = 1; r < tile; ++r) v->insert(v->end(), one.begin(), one.end());
      }
      {
        const std::vector<uint16_t> one = xb;
        for (uint32_t r = 1; r < tile; ++r) xb.insert(xb.end(), one.begin(), one.end());
      }
      C *= tile;
      const uint32_t nch = (C + kCt - 1) / kCt;
      const size_t nwu = size_t(C) * H * kDim, na2 = size_t(nch) * H * runtime::prefill::kA2HlWords,
                   nS = size_t(H) * kDim * kDim;
      l0::Mem d_xb(ctx, l0::MemKind::Device, xb.size() * 2), d_g(ctx, l0::MemKind::Device, g.size() * 4),
          d_b(ctx, l0::MemKind::Device, g.size() * 4), d_w(ctx, l0::MemKind::Device, nwu * 2),
          d_u(ctx, l0::MemKind::Device, nwu * 2), d_A2(ctx, l0::MemKind::Device, na2 * 4),
          d_w2(ctx, l0::MemKind::Device, nwu * 2), d_u2(ctx, l0::MemKind::Device, nwu * 2),
          d_A22(ctx, l0::MemKind::Device, na2 * 4), d_o(ctx, l0::MemKind::Device, nwu * 4),
          d_S(ctx, l0::MemKind::Device, nS * 4), d_S0(ctx, l0::MemKind::Device, nS * 4);
      cx.copy(d_xb.ptr(), xb.data(), xb.size() * 2);
      cx.copy(d_g.ptr(), g.data(), g.size() * 4);
      cx.copy(d_b.ptr(), beta.data(), g.size() * 4);
      cx.wait();
      const std::string sfx = kernels::gdn_suffix(hk, H);
      l0::Kernel& kIn = kc(kernels::pf_gdn_intra_variant(hk, H), "pf_gdn_intra");
      l0::Kernel& kScan = kc(kernels::pf_gdn_scan_variant(hk, H), "pf_gdn_scan_dpas_split");
      auto intra = [&](l0::Kernel& k, void* w, void* u, void* a2) {
        cx.launch(k, H, nch, 1, {PtrArg(d_xb.ptr()), PtrArg(d_g.ptr()), PtrArg(d_b.ptr()), PtrArg(w), PtrArg(u),
                                 PtrArg(a2), arg_val(C)});
      };
      auto scan = [&](l0::Kernel& k) {
        cx.launch(k, H, 4, 1, {PtrArg(d_xb.ptr()), PtrArg(d_w.ptr()), PtrArg(d_u.ptr()), PtrArg(d_A2.ptr()),
                               PtrArg(d_g.ptr()), PtrArg(d_S.ptr()), PtrArg(d_o.ptr()), arg_val(C)});
      };
      // S0: one production scan from zero.
      {
        std::vector<float> zero(nS, 0.0f);
        intra(kIn, d_w.ptr(), d_u.ptr(), d_A2.ptr());
        cx.copy(d_S.ptr(), zero.data(), nS * 4);
        scan(kScan);
        cx.copy(d_S0.ptr(), d_S.ptr(), nS * 4);
        cx.wait();
      }
      auto read_scan = [&](std::vector<float>& o, std::vector<float>& S) {
        o.resize(nwu);
        S.resize(nS);
        cx.copy(o.data(), d_o.ptr(), nwu * 4);
        cx.copy(S.data(), d_S.ptr(), nS * 4);
        cx.wait();
      };
      std::vector<float> o_ref, S_ref;
      cx.copy(d_S.ptr(), d_S0.ptr(), nS * 4);
      scan(kScan);
      read_scan(o_ref, S_ref);
      std::vector<uint16_t> w_ref(nwu), u_ref(nwu);
      std::vector<uint32_t> a2_ref(na2);
      cx.copy(w_ref.data(), d_w.ptr(), nwu * 2);
      cx.copy(u_ref.data(), d_u.ptr(), nwu * 2);
      cx.copy(a2_ref.data(), d_A2.ptr(), na2 * 4);
      cx.wait();
      std::printf("\ndump %d: C %u, heads %u / %u, %u sub-chunks\n", call, C, hk, H, nch);

      for (const Arm& arm : kArms) {
        if (!only.empty() && only != arm.bin) continue;
        l0::Kernel& kA = kc(arm.bin, arm.entry);
        const bool is_scan = std::strcmp(arm.kind, "scan") == 0;
        bool same = true;
        if (is_scan) {
          std::vector<float> o, S;
          cx.copy(d_S.ptr(), d_S0.ptr(), nS * 4);
          scan(kA);
          read_scan(o, S);
          same = std::memcmp(o.data(), o_ref.data(), nwu * 4) == 0 && std::memcmp(S.data(), S_ref.data(), nS * 4) == 0;
        } else {
          intra(kA, d_w2.ptr(), d_u2.ptr(), d_A22.ptr());
          std::vector<uint16_t> w(nwu), u(nwu);
          std::vector<uint32_t> a2(na2);
          cx.copy(w.data(), d_w2.ptr(), nwu * 2);
          cx.copy(u.data(), d_u2.ptr(), nwu * 2);
          cx.copy(a2.data(), d_A22.ptr(), na2 * 4);
          cx.wait();
          same = std::memcmp(w.data(), w_ref.data(), nwu * 2) == 0 && std::memcmp(u.data(), u_ref.data(), nwu * 2) == 0 &&
                 std::memcmp(a2.data(), a2_ref.data(), na2 * 4) == 0;
        }
        auto timed = [&](bool cand) {
          if (is_scan) cx.copy(d_S.ptr(), d_S0.ptr(), nS * 4);
          cx.timestamp(stamp);
          if (is_scan) scan(cand ? kA : kScan);
          else if (cand) intra(kA, d_w2.ptr(), d_u2.ptr(), d_A22.ptr());
          else intra(kIn, d_w2.ptr(), d_u2.ptr(), d_A22.ptr());
          cx.timestamp(stamp + 1);
          cx.wait();
          return cx.timestamp_ms(stamp[0], stamp[1]);
        };
        for (int wu = 0; wu < 3; ++wu) { timed(false); timed(true); }
        std::vector<double> tc, ta;
        for (int r = 0; r < rounds; ++r) {
          if (r % 2) { ta.push_back(timed(true)); tc.push_back(timed(false)); }
          else { tc.push_back(timed(false)); ta.push_back(timed(true)); }
        }
        const double mc = median(tc), ma = median(ta);
        std::printf("ARM {\"dump\":%d,\"C\":%u,\"arm\":\"%s\",\"bitwise\":%s,\"prod_ms\":%.4f,\"arm_ms\":%.4f,\"ratio\":%.3f}\n",
                    call, C, arm.bin, same ? "true" : "false", mc, ma, ma / mc);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_gdn_arms FAILED: %s\n", e.what());
    return 2;
  }
}

// probe_gdn_intra - phase 2 Task 1 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): the
// GDN intra-chunk stages on REAL inputs, today's four kernels (control: pf_gdn_A, pf_gdn_solve,
// pf_gdn_wu, pf_gdn_A2_hilo) against the one-launch DPAS arm D-a (pf_gdn_intra.cl). Record:
// docs/probes/probe-prefill-phase2-2026-11-12.md.
//
// Inputs: a B70_PF_GDN_DUMP directory (runtime/prefill/gdn.cc: gdn<call>_{xb,g,beta}.bin + .txt, the
// l2normed xb, g_cum and beta of one GDN layer's chunk). Per dump and arm:
//   * error against an fp64 host walk of the same inputs (exact algebra, no bf16 roundings): A, T,
//     w, u, A2 (hi + lo) as rel L2 = |x - ref| / |ref| and max = max |x - ref| / max |ref|;
//   * today's scan (`pf_gdn_scan_dpas_split`) run on the arm's w / u / A2 from a zero state: o and the
//     final state against the fp64 scan, as rel L2 and cosine distance 1 - cos.
//   The gate (pre-registered in the record): every D-a metric <= 2x the control's.
// Then time: the control's four launches against D-a's one, interleaved pairs after a warm-up, median of
// `--rounds`, at C = the dump's rows and at 2C (the dump tiled twice). Global timestamps around the launches.
//
//   probe_gdn_intra --dump DIR [--calls 0,9,21] [--rounds 9] [--no-time]
#include <algorithm>
#include <cmath>
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
constexpr double kQScale = 0.08838834764831845;

float f32(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
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

struct Shape {
  uint32_t C, hk, hv, conv;
  uint32_t nch() const { return (C + kCt - 1) / kCt; }
  uint32_t q_off() const { return 0; }
  uint32_t k_off() const { return hk * kDim; }
  uint32_t v_off() const { return 2 * hk * kDim; }
};

// Everything one arm produces, read back.
struct Out {
  std::vector<float> A, T, A2, o, S;   // A / T / A2 [nch][hv][64][64], o [C][hv][128], S [hv][128][128]
  std::vector<uint16_t> w, u;          // [C][hv][128]
};

// The fp64 walk: exact algebra on the dumped (bf16 / fp32) inputs.
struct Ref {
  std::vector<double> A, T, A2, w, u, o, S;
};
Ref reference(const Shape& sh, const std::vector<uint16_t>& xb, const std::vector<float>& g,
              const std::vector<float>& beta) {
  const uint32_t H = sh.hv, vpk = sh.hv / sh.hk, nch = sh.nch();
  Ref r;
  r.A.assign(size_t(nch) * H * kCt * kCt, 0.0);
  r.T = r.A;
  r.A2 = r.A;
  r.w.assign(size_t(sh.C) * H * kDim, 0.0);
  r.u = r.w;
  r.o = r.w;
  r.S.assign(size_t(H) * kDim * kDim, 0.0);
  auto X = [&](uint32_t m, uint32_t off, uint32_t d) { return double(f32(xb[size_t(m) * sh.conv + off + d])); };
  std::vector<double> vn(size_t(kCt) * kDim);
  for (uint32_t h = 0; h < H; ++h) {
    const uint32_t kh = h / vpk;
    double* S = r.S.data() + size_t(h) * kDim * kDim;
    for (uint32_t t = 0; t < nch; ++t) {
      const uint32_t b = t * kCt, L = std::min(kCt, sh.C - b);
      const size_t tb = (size_t(t) * H + h) * kCt * kCt;
      double* A = r.A.data() + tb;
      double* T = r.T.data() + tb;
      double* A2 = r.A2.data() + tb;
      std::vector<double> gc(L), bt(L);
      for (uint32_t i = 0; i < L; ++i) {
        gc[i] = g[size_t(b + i) * H + h];
        bt[i] = beta[size_t(b + i) * H + h];
      }
      for (uint32_t i = 0; i < L; ++i)
        for (uint32_t j = 0; j <= i; ++j) {
          double kk = 0, qk = 0;
          for (uint32_t d = 0; d < kDim; ++d) {
            const double kj = X(b + j, sh.k_off() + kh * kDim, d);
            kk += X(b + i, sh.k_off() + kh * kDim, d) * kj;
            qk += X(b + i, sh.q_off() + kh * kDim, d) * kQScale * kj;
          }
          const double e = std::exp(gc[i] - gc[j]);
          if (i > j) A[i * kCt + j] = bt[i] * kk * e;
          A2[i * kCt + j] = qk * e;
        }
      for (uint32_t j = 0; j < L; ++j) {           // T = (I + A)^-1, column by column
        T[j * kCt + j] = 1.0;
        for (uint32_t i = j + 1; i < L; ++i) {
          double acc = 0;
          for (uint32_t l = j; l < i; ++l) acc += A[i * kCt + l] * T[l * kCt + j];
          T[i * kCt + j] = -acc;
        }
      }
      for (uint32_t i = 0; i < L; ++i)
        for (uint32_t x = 0; x < kDim; ++x) {
          double au = 0, aw = 0;
          for (uint32_t j = 0; j <= i; ++j) {
            au += T[i * kCt + j] * X(b + j, sh.v_off() + h * kDim, x) * bt[j];
            aw += T[i * kCt + j] * X(b + j, sh.k_off() + kh * kDim, x) * bt[j] * std::exp(gc[j]);
          }
          r.u[(size_t(b + i) * H + h) * kDim + x] = au;
          r.w[(size_t(b + i) * H + h) * kDim + x] = aw;
        }
      // the scan, from the chunk-start S
      for (uint32_t i = 0; i < L; ++i)
        for (uint32_t x = 0; x < kDim; ++x) {
          double acc = r.u[(size_t(b + i) * H + h) * kDim + x];
          for (uint32_t k = 0; k < kDim; ++k) acc -= r.w[(size_t(b + i) * H + h) * kDim + k] * S[k * kDim + x];
          vn[i * kDim + x] = acc;
        }
      for (uint32_t i = 0; i < L; ++i)
        for (uint32_t x = 0; x < kDim; ++x) {
          double qs = 0;
          for (uint32_t k = 0; k < kDim; ++k) qs += X(b + i, sh.q_off() + kh * kDim, k) * kQScale * S[k * kDim + x];
          double acc = qs * std::exp(gc[i]);
          for (uint32_t j = 0; j <= i; ++j) acc += A2[i * kCt + j] * vn[j * kDim + x];
          r.o[(size_t(b + i) * H + h) * kDim + x] = acc;
        }
      const double gl = gc[L - 1], dl = std::exp(gl);
      for (size_t p = 0; p < size_t(kDim) * kDim; ++p) S[p] *= dl;
      for (uint32_t i = 0; i < L; ++i) {
        const double sc = std::exp(gl - gc[i]);
        for (uint32_t k = 0; k < kDim; ++k) {
          const double kv = X(b + i, sh.k_off() + kh * kDim, k);
          for (uint32_t x = 0; x < kDim; ++x) S[k * kDim + x] += kv * vn[i * kDim + x] * sc;
        }
      }
    }
  }
  return r;
}

struct Err {
  double rel_l2 = 0, max_rel = 0, cos_dist = 0;
};
template <class F>
Err err(size_t n, const std::vector<double>& ref, F get) {
  double d2 = 0, r2 = 0, md = 0, mr = 0, dot = 0, x2 = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = get(i), y = ref[i];
    d2 += (x - y) * (x - y);
    r2 += y * y;
    x2 += x * x;
    dot += x * y;
    md = std::max(md, std::fabs(x - y));
    mr = std::max(mr, std::fabs(y));
  }
  Err e;
  e.rel_l2 = r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2);
  e.max_rel = mr > 0 ? md / mr : md;
  e.cos_dist = (x2 > 0 && r2 > 0) ? 1.0 - dot / std::sqrt(x2 * r2) : 0.0;
  return e;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string dir;
    std::vector<int> calls = {0, 9, 21};
    int rounds = 9;
    bool time = true;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--dump" && i + 1 < argc) dir = argv[++i];
      else if (a == "--rounds" && i + 1 < argc) rounds = std::max(1, std::atoi(argv[++i]));
      else if (a == "--no-time") time = false;
      else if (a == "--calls" && i + 1 < argc) {
        calls.clear();
        std::stringstream ss(argv[++i]);
        std::string tok;
        while (std::getline(ss, tok, ',')) calls.push_back(std::atoi(tok.c_str()));
      } else {
        std::fprintf(stderr, "usage: probe_gdn_intra --dump DIR [--calls 0,9,21] [--rounds 9] [--no-time]\n");
        return 2;
      }
    }
    if (dir.empty()) throw std::runtime_error("--dump DIR is required");
    l0::Context ctx(0);
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    std::printf("# probe_gdn_intra: control (pf_gdn_A, _solve, _wu, _A2_hilo) vs D-a (pf_gdn_intra)\n");
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n", mask ? mask : "(unset)", ctx.name().c_str(), rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    bool gate = true;

    for (int call : calls) {
      const std::string base = dir + "/gdn" + std::to_string(call);
      Shape sh{};
      {
        std::ifstream f(base + ".txt");
        if (!(f >> sh.C >> sh.hk >> sh.hv >> sh.conv)) throw std::runtime_error("cannot read " + base + ".txt");
      }
      const std::vector<uint16_t> xb = read_bin<uint16_t>(base + "_xb.bin");
      const std::vector<float> g = read_bin<float>(base + "_g.bin"), beta = read_bin<float>(base + "_beta.bin");
      if (xb.size() != size_t(sh.C) * sh.conv || g.size() != size_t(sh.C) * sh.hv || beta.size() != g.size())
        throw std::runtime_error(base + ": sizes do not match the .txt");
      const uint32_t H = sh.hv;
      const std::string wy = kernels::pf_gdn_wy_variant(sh.hk, sh.hv);
      const std::string scanv = kernels::pf_gdn_scan_variant(sh.hk, sh.hv);
      const std::string intra = kernels::pf_gdn_intra_variant(sh.hk, sh.hv);
      l0::Kernel& kA = kc(wy, "pf_gdn_A");
      l0::Kernel& kS = kc(wy, "pf_gdn_solve");
      l0::Kernel& kWu = kc(wy, "pf_gdn_wu");
      l0::Kernel& kA2 = kc(wy, "pf_gdn_A2_hilo");
      l0::Kernel& kScan = kc(scanv, "pf_gdn_scan_dpas_split");
      l0::Kernel& kIn = kc(intra, "pf_gdn_intra");
      l0::Kernel& kInD = kc("pf_gdn_intra_dbg" + kernels::gdn_suffix(sh.hk, sh.hv), "pf_gdn_intra");

      // 2C rows for the timing's second width: the dump tiled twice (the same values each half).
      const uint32_t C2 = 2 * sh.C, nch2 = (C2 + kCt - 1) / kCt;
      const size_t tri = size_t(nch2) * H * kCt * kCt;
      l0::Mem d_xb(ctx, l0::MemKind::Device, size_t(C2) * sh.conv * 2),
          d_g(ctx, l0::MemKind::Device, size_t(C2) * H * 4), d_b(ctx, l0::MemKind::Device, size_t(C2) * H * 4),
          d_A(ctx, l0::MemKind::Device, tri * 4), d_T(ctx, l0::MemKind::Device, tri * 4),
          d_A2(ctx, l0::MemKind::Device, size_t(nch2) * H * runtime::prefill::kA2HlWords * 4),
          d_w(ctx, l0::MemKind::Device, size_t(C2) * H * kDim * 2), d_u(ctx, l0::MemKind::Device, size_t(C2) * H * kDim * 2),
          d_o(ctx, l0::MemKind::Device, size_t(C2) * H * kDim * 4), d_S(ctx, l0::MemKind::Device, size_t(H) * kDim * kDim * 4);
      for (int half = 0; half < 2; ++half) {
        cx.copy(d_xb.as<uint8_t>() + size_t(half) * xb.size() * 2, xb.data(), xb.size() * 2);
        cx.copy(d_g.as<uint8_t>() + size_t(half) * g.size() * 4, g.data(), g.size() * 4);
        cx.copy(d_b.as<uint8_t>() + size_t(half) * g.size() * 4, beta.data(), g.size() * 4);
      }
      cx.wait();
      const uint32_t C = sh.C, nch = sh.nch();

      auto control = [&](uint32_t c, uint32_t n) {
        cx.launch(kA, H, n, 4, {PtrArg(d_xb.ptr()), PtrArg(d_g.ptr()), PtrArg(d_b.ptr()), PtrArg(d_A.ptr()), arg_val(c)});
        cx.launch(kS, H, n, 1, {PtrArg(d_A.ptr()), arg_val(c)});
        cx.launch(kWu, H, n, 2, {PtrArg(d_xb.ptr()), PtrArg(d_A.ptr()), PtrArg(d_g.ptr()), PtrArg(d_b.ptr()),
                                 PtrArg(d_w.ptr()), PtrArg(d_u.ptr()), arg_val(c)});
        cx.launch(kA2, H, n, 4, {PtrArg(d_xb.ptr()), PtrArg(d_g.ptr()), PtrArg(d_A2.ptr()), arg_val(c)});
      };
      auto da = [&](uint32_t c, uint32_t n, bool dbg) {
        if (dbg)
          cx.launch(kInD, H, n, 1, {PtrArg(d_xb.ptr()), PtrArg(d_g.ptr()), PtrArg(d_b.ptr()), PtrArg(d_w.ptr()),
                                    PtrArg(d_u.ptr()), PtrArg(d_A2.ptr()), arg_val(c), PtrArg(d_A.ptr()),
                                    PtrArg(d_T.ptr())});
        else
          cx.launch(kIn, H, n, 1, {PtrArg(d_xb.ptr()), PtrArg(d_g.ptr()), PtrArg(d_b.ptr()), PtrArg(d_w.ptr()),
                                   PtrArg(d_u.ptr()), PtrArg(d_A2.ptr()), arg_val(c)});
      };
      auto scan_and_read = [&](Out& o) {
        std::vector<float> zero(size_t(H) * kDim * kDim, 0.0f);
        cx.copy(d_S.ptr(), zero.data(), zero.size() * 4);
        cx.launch(kScan, H, 4, 1, {PtrArg(d_xb.ptr()), PtrArg(d_w.ptr()), PtrArg(d_u.ptr()), PtrArg(d_A2.ptr()),
                                   PtrArg(d_g.ptr()), PtrArg(d_S.ptr()), PtrArg(d_o.ptr()), arg_val(C)});
        o.w.resize(size_t(C) * H * kDim);
        o.u.resize(o.w.size());
        o.o.resize(o.w.size());
        o.S.resize(zero.size());
        std::vector<uint32_t> hl(size_t(nch) * H * runtime::prefill::kA2HlWords);
        cx.copy(o.w.data(), d_w.ptr(), o.w.size() * 2);
        cx.copy(o.u.data(), d_u.ptr(), o.u.size() * 2);
        cx.copy(o.o.data(), d_o.ptr(), o.o.size() * 4);
        cx.copy(o.S.data(), d_S.ptr(), o.S.size() * 4);
        cx.copy(hl.data(), d_A2.ptr(), hl.size() * 4);
        cx.wait();
        // unpack A2 = f32(hi) + f32(lo) into the fp32 tile layout
        o.A2.assign(size_t(nch) * H * kCt * kCt, 0.0f);
        for (size_t tl = 0; tl < size_t(nch) * H; ++tl) {
          const uint32_t* wds = hl.data() + tl * runtime::prefill::kA2HlWords;
          float* A2 = o.A2.data() + tl * kCt * kCt;
          for (uint32_t bnt = 0; bnt < 4; ++bnt)
            for (uint32_t bkb = 0; bkb <= bnt; ++bkb)
              for (uint32_t ln = 0; ln < 16; ++ln)
                for (uint32_t r = 0; r < 8; ++r) {
                  const uint32_t p = ((bnt * (bnt + 1)) / 2 + bkb) * 128 + ln * 8 + r;
                  const uint32_t hi = wds[p], lo = wds[runtime::prefill::kA2HlLo + p];
                  float* v = A2 + size_t(bnt * 16 + ln) * kCt + bkb * 16 + 2 * r;
                  v[0] = f32(uint16_t(hi)) + f32(uint16_t(lo));
                  v[1] = f32(uint16_t(hi >> 16)) + f32(uint16_t(lo >> 16));
                }
        }
      };

      // control: A, then T (in place), then w / u and A2
      Out oc, od;
      {
        cx.launch(kA, H, nch, 4, {PtrArg(d_xb.ptr()), PtrArg(d_g.ptr()), PtrArg(d_b.ptr()), PtrArg(d_A.ptr()), arg_val(C)});
        oc.A.resize(size_t(nch) * H * kCt * kCt);
        cx.copy(oc.A.data(), d_A.ptr(), oc.A.size() * 4);
        cx.wait();
        control(C, nch);
        oc.T.resize(oc.A.size());
        cx.copy(oc.T.data(), d_A.ptr(), oc.T.size() * 4);
        cx.wait();
        scan_and_read(oc);
      }
      {
        da(C, nch, true);
        od.A.resize(size_t(nch) * H * kCt * kCt);
        od.T.resize(od.A.size());
        cx.copy(od.A.data(), d_A.ptr(), od.A.size() * 4);
        cx.copy(od.T.data(), d_T.ptr(), od.T.size() * 4);
        cx.wait();
        scan_and_read(od);
        // the production binary writes the same w / u / A2 as the dbg one
        std::vector<uint16_t> w2(od.w.size());
        da(C, nch, false);
        cx.copy(w2.data(), d_w.ptr(), w2.size() * 2);
        cx.wait();
        const bool same = std::memcmp(w2.data(), od.w.data(), w2.size() * 2) == 0;
        std::printf("dump %d: D-a production binary w == dbg binary w: %s\n", call, same ? "yes" : "NO");
        gate = gate && same;
      }

      const Ref rf = reference(sh, xb, g, beta);
      // T on its live lower triangle (the control leaves rows >= L zero, D-a the identity there).
      auto mask_live = [&](std::vector<double>& ref, const std::vector<float>& x, bool lower_incl) {
        std::vector<double> v(ref.size(), 0.0);
        for (uint32_t t = 0; t < nch; ++t) {
          const uint32_t L = std::min(kCt, C - t * kCt);
          for (uint32_t h = 0; h < H; ++h)
            for (uint32_t i = 0; i < L; ++i)
              for (uint32_t j = 0; j < L && (lower_incl ? j <= i : j < i); ++j) {
                const size_t p = ((size_t(t) * H + h) * kCt + i) * kCt + j;
                v[p] = x[p];
              }
        }
        (void)ref;
        return v;
      };
      std::printf("\ndump %d: C %u, heads %u / %u\n", call, C, sh.hk, sh.hv);
      std::printf("| stage | control rel L2 | control max | D-a rel L2 | D-a max | D-a / control (L2, max) |\n"
                  "|---|---:|---:|---:|---:|---:|\n");
      auto row = [&](const char* name, Err c, Err d, bool cosine) {
        const double r1 = c.rel_l2 > 0 ? d.rel_l2 / c.rel_l2 : (d.rel_l2 > 0 ? 1e9 : 1.0);
        const double r2 = cosine ? (c.cos_dist > 0 ? d.cos_dist / c.cos_dist : (d.cos_dist > 0 ? 1e9 : 1.0))
                                 : (c.max_rel > 0 ? d.max_rel / c.max_rel : (d.max_rel > 0 ? 1e9 : 1.0));
        const bool ok = r1 <= 2.0 && r2 <= 2.0;
        gate = gate && ok;
        std::printf("| %s | %.3e | %.3e | %.3e | %.3e | %.2f, %.2f %s |\n", name, c.rel_l2, cosine ? c.cos_dist : c.max_rel,
                    d.rel_l2, cosine ? d.cos_dist : d.max_rel, r1, r2, ok ? "ok" : "OVER");
        std::printf("STAGE {\"dump\":%d,\"stage\":\"%s\",\"c_l2\":%.6e,\"c_2\":%.6e,\"d_l2\":%.6e,\"d_2\":%.6e,\"ok\":%s}\n",
                    call, name, c.rel_l2, cosine ? c.cos_dist : c.max_rel, d.rel_l2, cosine ? d.cos_dist : d.max_rel,
                    ok ? "true" : "false");
      };
      {
        std::vector<double> dummy;
        const auto cA = mask_live(dummy, oc.A, false), dA = mask_live(dummy, od.A, false);
        const auto cT = mask_live(dummy, oc.T, true), dT = mask_live(dummy, od.T, true);
        const size_t nt = rf.A.size();
        row("A", err(nt, rf.A, [&](size_t i) { return cA[i]; }), err(nt, rf.A, [&](size_t i) { return dA[i]; }), false);
        row("T", err(nt, rf.T, [&](size_t i) { return cT[i]; }), err(nt, rf.T, [&](size_t i) { return dT[i]; }), false);
        row("A2", err(nt, rf.A2, [&](size_t i) { return double(oc.A2[i]); }),
            err(nt, rf.A2, [&](size_t i) { return double(od.A2[i]); }), false);
        const size_t nw = rf.w.size();
        row("w", err(nw, rf.w, [&](size_t i) { return double(f32(oc.w[i])); }),
            err(nw, rf.w, [&](size_t i) { return double(f32(od.w[i])); }), false);
        row("u", err(nw, rf.u, [&](size_t i) { return double(f32(oc.u[i])); }),
            err(nw, rf.u, [&](size_t i) { return double(f32(od.u[i])); }), false);
        row("o (cos)", err(nw, rf.o, [&](size_t i) { return double(oc.o[i]); }),
            err(nw, rf.o, [&](size_t i) { return double(od.o[i]); }), true);
        row("state (cos)", err(rf.S.size(), rf.S, [&](size_t i) { return double(oc.S[i]); }),
            err(rf.S.size(), rf.S, [&](size_t i) { return double(od.S[i]); }), true);
      }

      if (!time) continue;
      l0::Mem d_stamp(ctx, l0::MemKind::Host, 64);
      uint64_t* stamp = d_stamp.as<uint64_t>();
      for (uint32_t c : {C, C2}) {
        const uint32_t n = (c + kCt - 1) / kCt;
        auto timed = [&](bool cand) {
          cx.timestamp(stamp);
          if (cand) da(c, n, false); else control(c, n);
          cx.timestamp(stamp + 1);
          cx.wait();
          return cx.timestamp_ms(stamp[0], stamp[1]);
        };
        for (int wu = 0; wu < 3; ++wu) { timed(false); timed(true); }
        std::vector<double> tc, td;
        for (int r = 0; r < rounds; ++r) {
          if (r % 2) { td.push_back(timed(true)); tc.push_back(timed(false)); }
          else { tc.push_back(timed(false)); td.push_back(timed(true)); }
        }
        const double mc = median(tc), md = median(td);
        std::printf("TIME {\"dump\":%d,\"C\":%u,\"control_ms\":%.4f,\"da_ms\":%.4f,\"ratio\":%.3f}\n", call, c, mc, md, md / mc);
      }
    }
    std::printf("\nGATE (every D-a stage within 2x the control's error, both binaries agree): %s\n", gate ? "PASS" : "FAIL");
    return gate ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_gdn_intra FAILED: %s\n", e.what());
    return 2;
  }
}

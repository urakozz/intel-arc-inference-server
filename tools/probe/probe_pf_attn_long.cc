// probe_pf_attn_long - the full-chunk prefill attention deep in the cache: today's binary against
// pf_flash_attn.cl's PAIR form, bitwise and timed on the production shapes.
//
//   probe_pf_attn_long <pos> <C> [--heads 24x4|16x2] [--kv8] [--arms <name>[,...]] [--rounds N] [--sub R]
//
// Inputs: Q bf16 [C][q-heads][256] N(0,1), K / V bf16 [pos + C][kv-heads][256] N(0,1) (one layer's cache).
// --kv8 (Qwen3.8's heads): the int8 KV cache's whole-chunk flash over the widened rows instead - K / V are int8 values
// in [-127, 127] as bf16 (kv8_widen's output), K / V scales fp16 in [1, 100] / 4096; the control is
// pf_flash_attn_kv8_raw, the arms kv8.cl's KV8_PAIR form (`pfal_kv8_H<hpw>`, and the production
// pf_flash_attn_kv8_raw_pair by its name).
// The control is the RPW 8 binary (pf_flash_attn[_Q16KV2], grid (ceil(C / 8), kv-heads, 1); Ornith ran a 32-row
// form of it before PAIR, bitwise the same and timed in the PAIR commit). Each arm writes O into its own buffer; every word of rows < C
// is compared with the control's (the count of differing words and the largest ulp distance are printed).
// Timing: L0 kernel timestamps, a 5-round warm-up, then N rounds interleaving the control and every arm; the
// headline is the median paired ratio arm / control. PROBE-ONLY: no runtime path binds an arm.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;

constexpr uint32_t kHD = 256;
constexpr double kPeakTflops = 183.45;

void copy(l0::Context& ctx, void* dst, const void* src, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(dst) + off, static_cast<const char*>(src) + off, std::min(kChunk, bytes - off));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::fprintf(stderr, "usage: probe_pf_attn_long <pos> <C> [--heads 24x4|16x2] [--kv8] [--arms a,b] [--rounds N]\n");
      return 2;
    }
    const uint32_t pos = uint32_t(std::stoul(argv[1])), C = uint32_t(std::stoul(argv[2]));
    bool ornith = false, kv8 = false;
    std::string arms_arg;
    int rounds = 7;
    uint32_t sub = 0;   // --sub R: each arm as launches of R rows (pos + r, Q and O offset by r); 0 = one launch
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (++i >= argc) throw std::runtime_error(a + " needs a value");
        return argv[i];
      };
      if (a == "--heads") ornith = val() == "16x2";
      else if (a == "--kv8") kv8 = true;
      else if (a == "--arms") arms_arg = val();
      else if (a == "--rounds") rounds = std::stoi(val());
      else if (a == "--sub") sub = uint32_t(std::stoul(val()));
      else throw std::runtime_error("unknown argument " + a);
    }
    if (kv8 && ornith) throw std::runtime_error("--kv8: Ornith has no int8-KV prompt path");
    const uint32_t qh = ornith ? 16 : 24, kvh = ornith ? 2 : 4, gqa = qh / kvh;
    const uint32_t depth = pos + C, rows = (C + 255) / 256 * 256;
    const std::string sfx = ornith ? "_Q16KV2" : "";
    if (arms_arg.empty())
      arms_arg = kv8      ? "pfal_kv8_H1,pfal_kv8_H2,pfal_kv8_H3,pfal_kv8_H6,pf_flash_attn_kv8_raw_pair"
                 : ornith ? "pfal_H1,pfal_H2,pfal_H4,pfal_H8"
                          : "pfal_H1,pfal_H2,pfal_H3,pfal_H6";

    l0::Context ctx(0);
    std::printf("# probe_pf_attn_long: %s%s, pos %u, C %u, depth %u; device %s\n", ornith ? "Ornith 16/2" : "Qwen3.8 24/4",
                kv8 ? " int8 KV" : "", pos, C, depth, ctx.name().c_str());
    std::mt19937 rng(777u + pos + C);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h(size_t(C) * qh * kHD);
    for (auto& x : h) x = common::f32_to_bf16(nd(rng));
    l0::Mem dq(ctx, l0::MemKind::Device, h.size() * 2);
    copy(ctx, dq.ptr(), h.data(), h.size() * 2);
    h.assign(size_t(depth) * kvh * kHD, 0);
    l0::Mem dk(ctx, l0::MemKind::Device, h.size() * 2), dv(ctx, l0::MemKind::Device, h.size() * 2);
    auto kv_value = [&]() { return kv8 ? common::f32_to_bf16(float(int(rng() % 255) - 127)) : common::f32_to_bf16(nd(rng)); };
    for (auto& x : h) x = kv_value();
    copy(ctx, dk.ptr(), h.data(), h.size() * 2);
    for (auto& x : h) x = kv_value();
    copy(ctx, dv.ptr(), h.data(), h.size() * 2);
    h.assign(kv8 ? size_t(depth) * kvh : 0, 0);
    l0::Mem dks(ctx, l0::MemKind::Device, std::max<size_t>(h.size() * 2, 64)),
        dvs(ctx, l0::MemKind::Device, std::max<size_t>(h.size() * 2, 64));
    if (kv8) {
      for (auto& x : h) x = common::kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
      copy(ctx, dks.ptr(), h.data(), h.size() * 2);
      for (auto& x : h) x = common::kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
      copy(ctx, dvs.ptr(), h.data(), h.size() * 2);
    }
    h.clear();
    h.shrink_to_fit();
    const size_t o_elems = size_t(qh) * rows * kHD;

    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    struct Path {
      std::string name;
      l0::Kernel* k;
      uint32_t gx, gz;
      std::unique_ptr<l0::Mem> out;
      std::vector<float> host;
      std::vector<double> ms, ratio;
    };
    std::vector<Path> paths;
    const char* fn = kv8 ? "pf_flash_attn_kv8" : "pf_flash_attn";
    const std::string ctl = kv8 ? kernels::pf_flash_attn_kv8_raw_variant(qh, kvh) : kernels::pf_flash_attn_variant(qh, kvh);
    paths.push_back({ctl, &kc(ctl, fn), (C + 7u) / 8u, 1, nullptr, {}, {}, {}});
    for (size_t st = 0; st <= arms_arg.size();) {
      const size_t e = std::min(arms_arg.find(',', st), arms_arg.size());
      const std::string nm = arms_arg.substr(st, e - st) + sfx;
      st = e + 1;
      if (!std::ifstream(kernels::path(nm)).good()) {
        std::printf("%s: NO BINARY\n", nm.c_str());
        continue;
      }
      const uint32_t hpw = nm.find("_H") == std::string::npos ? kernels::kPfFlashPairHeads
                                                              : uint32_t(std::stoul(nm.substr(nm.find("_H") + 2)));
      paths.push_back({nm, &kc(nm, fn), (C + 15u) / 16u, gqa / hpw, nullptr, {}, {}, {}});
    }
    for (Path& p : paths) {
      p.out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, o_elems * 4);
      l0::CmdList::immediate(ctx).fill(p.out->ptr(), 0u, o_elems * 4);
    }
    auto run = [&](Path& p) {
      if (kv8 && (sub == 0 || &p == &paths[0]))
        cx.launch(*p.k, p.gx, kvh, p.gz,
                  {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dks.ptr()), PtrArg(dv.ptr()), PtrArg(dvs.ptr()),
                   PtrArg(p.out->ptr()), arg_val(pos), arg_val(C), arg_val(rows)});
      else if (kv8)
        for (uint32_t r = 0; r < C; r += sub) {
          const uint32_t c = std::min(sub, C - r);
          cx.launch(*p.k, (c + 15u) / 16u, kvh, p.gz,
                    {PtrArg(static_cast<char*>(dq.ptr()) + size_t(r) * qh * kHD * 2), PtrArg(dk.ptr()), PtrArg(dks.ptr()),
                     PtrArg(dv.ptr()), PtrArg(dvs.ptr()), PtrArg(static_cast<char*>(p.out->ptr()) + size_t(r) * kHD * 4),
                     arg_val(pos + r), arg_val(c), arg_val(rows)});
        }
      else if (sub == 0 || &p == &paths[0])
        cx.launch(*p.k, p.gx, kvh, p.gz,
                  {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dv.ptr()), PtrArg(p.out->ptr()), arg_val(pos), arg_val(C),
                   arg_val(rows)});
      else
        for (uint32_t r = 0; r < C; r += sub) {
          const uint32_t c = std::min(sub, C - r);
          cx.launch(*p.k, (c + 15u) / 16u, kvh, p.gz,
                    {PtrArg(static_cast<char*>(dq.ptr()) + size_t(r) * qh * kHD * 2), PtrArg(dk.ptr()), PtrArg(dv.ptr()),
                     PtrArg(static_cast<char*>(p.out->ptr()) + size_t(r) * kHD * 4), arg_val(pos + r), arg_val(c),
                     arg_val(rows)});
        }
    };
    for (Path& p : paths) {
      run(p);
      cx.wait();
      p.host.resize(o_elems);
      copy(ctx, p.host.data(), p.out->ptr(), o_elems * 4);
    }
    for (size_t i = 1; i < paths.size(); ++i) {
      size_t differ = 0;
      int64_t worst = 0;
      for (uint32_t hh = 0; hh < qh; ++hh)
        for (size_t w = size_t(hh) * rows * kHD, e = w + size_t(C) * kHD; w < e; ++w) {
          int32_t a, b;
          std::memcpy(&a, &paths[0].host[w], 4);
          std::memcpy(&b, &paths[i].host[w], 4);
          if (a != b) {
            ++differ;
            worst = std::max<int64_t>(worst, std::llabs(int64_t(a) - int64_t(b)));
          }
        }
      std::printf("%s: words differing from %s: %zu of %zu, max ulp %lld -- %s\n", paths[i].name.c_str(), ctl.c_str(),
                  differ, size_t(qh) * C * kHD, (long long)worst, differ == 0 ? "BITWISE" : "NOT BITWISE");
    }
    for (Path& p : paths) p.host = {};
    std::fflush(stdout);
    if (rounds == 0) return 0;   // the byte check only

    cx.set_profiling(true);
    auto time_ms = [&](Path& p) {
      cx.take_launch_metrics();
      run(p);
      cx.wait();
      return cx.take_launch_metrics().gpu_ms;
    };
    for (int r = 0; r < 5; ++r)
      for (Path& p : paths) time_ms(p);
    for (int r = 0; r < rounds; ++r) {
      const double c = time_ms(paths[0]);
      paths[0].ms.push_back(c);
      for (size_t i = 1; i < paths.size(); ++i) {
        const double t = time_ms(paths[i]);
        paths[i].ms.push_back(t);
        paths[i].ratio.push_back(t / c);
      }
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v.empty() ? 0.0 : v[v.size() / 2];
    };
    double keys = 0;
    for (uint32_t m = 0; m < C; ++m) keys += double(pos) + m + 1;
    const double gflop = 4.0 * qh * kHD * keys / 1e9;
    std::printf("\n| path | median ms | arm / today (paired median, min..max) | TFLOP/s | %% of peak |\n|---|---:|---:|---:|---:|\n");
    for (Path& p : paths) {
      const double mt = med(p.ms);
      char rr[96] = "1";
      if (!p.ratio.empty())
        std::snprintf(rr, sizeof rr, "%.3f (%.3f..%.3f)", med(p.ratio), *std::min_element(p.ratio.begin(), p.ratio.end()),
                      *std::max_element(p.ratio.begin(), p.ratio.end()));
      std::printf("| %s | %.3f | %s | %.1f | %.1f |\n", p.name.c_str(), mt, rr, gflop / mt, 100.0 * gflop / mt / kPeakTflops);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "probe_pf_attn_long FAILED: %s\n", ex.what());
    return 2;
  }
}

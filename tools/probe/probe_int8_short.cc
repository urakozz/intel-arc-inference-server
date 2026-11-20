// probe_int8_short - where l0-int8's short-chunk MLP should take the bf16 walk. Qwen3.8's two h8
// linears (gate||up with the SiLU epilogue, down) of one dense layer at M rows, each arm the
// PRODUCTION entry points on random weights:
//   i8    linear_i8_silu + linear_i8 (runtime/prefill/int8.cc: quantiser, then per slab the rotated
//         int8 requant + pf_gemm_i8 - the requant is paid per chunk whatever M is)
//   bf16  linear_l0_silu + linear_l0 (runtime/prefill/linear_l0.cc: pf_gemm_q4 up to its row bounds,
//         then the slab walk with pf_gemm_small up to kPfSmallMaxRows, pf_gemm above)
// The summary multiplies by Qwen3.8's 64 layers: one chunk's MLP time on each path.
//
//   probe_int8_short [--rounds 7] [--rows 16,32,64,128,256,384,512,768,1024,2048]
//
// Weights are int4 words uniform, f16 scales U(0.005, 0.02), activations bf16 N(0, 1) (kernel
// TIMES do not depend on the values). Timing: each arm is the GPU span between two global
// timestamps around its launches on the in-order list (gaps included); a warm-up pass, then
// `rounds` rounds with the arms in a rotating order (interleaved). Medians.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill/linear_l0.h"

namespace {

constexpr uint32_t kMaxRows = 2048, kHidden = 5120, kInter = 17408, kLayers = 64;

uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
uint16_t f32_to_f16(float f) {   // positive normal values only (the scales)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const int e = int((u >> 23) & 0xFF) - 127 + 15;
  return uint16_t((e << 10) | ((u >> 13) & 0x3FF));
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// A layout-0 int4 weight [K][N] with its f16 scales, random.
loader::DeviceWeight make_weight(l0::Context& ctx, uint32_t K, uint32_t N, std::mt19937& rng) {
  std::vector<uint32_t> w(size_t(K / 8) * N);
  for (uint32_t& v : w) v = rng();
  std::vector<uint16_t> sc(size_t(K / 64) * N);
  std::uniform_real_distribution<float> u(0.005f, 0.02f);
  for (uint16_t& s : sc) s = f32_to_f16(u(rng));
  loader::DeviceWeight d{l0::Mem(ctx, l0::MemKind::Device, w.size() * 4), nullptr, model::GemvShape{K, N, 1, 0},
                         model::WeightKind::Int4};
  d.scales = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sc.size() * 2);
  l0::CmdList c = l0::CmdList::immediate(ctx);
  c.copy(d.mem.ptr(), w.data(), w.size() * 4);
  c.copy(d.scales->ptr(), sc.data(), sc.size() * 2);
  return d;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(tok);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rounds = 7;
    std::string rows_s = "16,32,64,128,256,384,512,768,1024,2048";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
      else if (a == "--rows" && i + 1 < argc) rows_s = argv[++i];
      else {
        std::fprintf(stderr, "usage: probe_int8_short [--rounds 7] [--rows 16,...,2048]\n");
        return 2;
      }
    }
    if (rounds < 1) throw std::runtime_error("--rounds must be >= 1");
    std::vector<uint32_t> rows;
    for (const std::string& r : split(rows_s)) {
      const uint32_t m = uint32_t(std::atoi(r.c_str()));
      if (m < 1 || m > kMaxRows) throw std::runtime_error("--rows: each in 1.." + std::to_string(kMaxRows));
      rows.push_back(m);
    }
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_int8_short: Qwen3.8's MLP (gate||up + down) on l0-int8's h8 walk vs the bf16 walk\n");
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n\n", mask ? mask : "(unset)", ctx.name().c_str(),
                rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    runtime::PrefillScratch s(ctx, 256, model::qwen38(), kMaxRows);
    runtime::prefill::Int8State q(ctx, kInter, kMaxRows);

    std::mt19937 rng(5);
    const loader::DeviceWeight gu = make_weight(ctx, kHidden, 2 * kInter, rng);
    const loader::DeviceWeight down = make_weight(ctx, kInter, kHidden, rng);
    q.scales(cx, kc, gu);
    q.scales(cx, kc, down);
    std::vector<uint16_t> xh(size_t(kMaxRows) * kInter);
    {
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (uint16_t& v : xh) v = f32_to_bf16(nd(rng));
    }
    l0::Mem x(ctx, l0::MemKind::Device, xh.size() * 2), xi(ctx, l0::MemKind::Device, xh.size() * 2);
    {
      l0::CmdList c = l0::CmdList::immediate(ctx);
      c.copy(x.ptr(), xh.data(), size_t(kMaxRows) * kHidden * 2);
    }
    l0::Mem ts(ctx, l0::MemKind::Host, 64);
    uint64_t* stamp = ts.as<uint64_t>();

    struct Arm {
      bool i8;
      uint32_t M;
    };
    std::vector<Arm> arms;
    for (uint32_t M : rows) {
      arms.push_back({true, M});
      arms.push_back({false, M});
    }
    auto run = [&](const Arm& a) {
      cx.timestamp(stamp);
      if (a.i8) {
        runtime::prefill::linear_i8_silu(cx, kc, s, q, gu, x.as<uint16_t>(), a.M, xi.as<uint16_t>(), kInter);
        runtime::prefill::linear_i8(cx, kc, s, q, down, xi.as<uint16_t>(), a.M);
      } else {
        runtime::prefill::linear_l0_silu(cx, kc, s, gu, x.as<uint16_t>(), a.M, xi.as<uint16_t>(), kInter);
        runtime::prefill::linear_l0(cx, kc, s, down, xi.as<uint16_t>(), a.M);
      }
      cx.timestamp(stamp + 1);
      cx.wait();
      return cx.timestamp_ms(stamp[0], stamp[1]);
    };
    for (const Arm& a : arms) run(a);
    std::vector<std::vector<double>> t(arms.size());
    for (int r = 0; r < rounds; ++r)
      for (size_t j = 0; j < arms.size(); ++j) {
        const size_t i = (j + size_t(r)) % arms.size();
        t[i].push_back(run(arms[i]));
      }

    std::printf("| rows | i8 ms / layer | bf16 ms / layer | i8 ms / chunk (x%u) | bf16 ms / chunk | bf16 / i8 |\n",
                kLayers);
    std::printf("|---:|---:|---:|---:|---:|---:|\n");
    for (size_t k = 0; k < rows.size(); ++k) {
      const double a = median(t[2 * k]), b = median(t[2 * k + 1]);
      std::printf("| %u | %.3f | %.3f | %.1f | %.1f | %.3f |\n", rows[k], a, b, a * kLayers, b * kLayers, b / a);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_int8_short FAILED: %s\n", e.what());
    return 2;
  }
}

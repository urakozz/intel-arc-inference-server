#pragma once
// Sleef_expf8_u10 (sleefsimdsp.c xexpf at AVX2 + FMA) in plain C++: torch's `Vectorized<float>::exp`, step for step,
// the fma as the AVX2 build has it (std::fma is exact, so the result does not depend on the host's libm or its FMA
// contraction). The same function as tests/kernels/k2_ref.h's `exp_torch` and the kernels' `exp_torch`
// (src/kernels/k2/k2_attn_eager.cl and the family copies), here for host code that needs a deterministic fp32 exp:
// spec 25b's load-time `A = -exp(A_log)`.
//
// What it is NOT: `torch.exp` on a contiguous fp32 CPU tensor of an x86 Linux wheel goes to MKL's vsExp, which has no
// host reproduction (tests/kernels/k2_ref.h measured the softmax's Sleef path, not torch.exp). docs/13-loader.md,
// "Nemotron 3.5 Bolt", records how far this function is from the reference's `torch.exp(A_log.float())` over every
// bf16 value of A_log's range.
#include <cmath>
#include <cstdint>
#include <cstring>

namespace common {

inline float exp_torch(float d) {
  const float kRLn2 = 1.442695040888963407359924681001892137426645954152985934135449406931f;
  const float kL2U = 0.693145751953125f, kL2L = 1.428606765330187045e-06f;
  if (d < -104.0f) return 0.0f;   // before the integer conversion: q would leave pow2i's range
  if (100.0f < d) return INFINITY;
  const int q = int(std::nearbyint(d * kRLn2));   // cvtps2dq: nearest-even
  const float qf = float(q);
  float s = std::fma(qf, -kL2U, d);
  s = std::fma(qf, -kL2L, s);
  float u = 0.000198527617612853646278381f;
  u = std::fma(u, s, 0.00139304355252534151077271f);
  u = std::fma(u, s, 0.00833336077630519866943359f);
  u = std::fma(u, s, 0.0416664853692054748535156f);
  u = std::fma(u, s, 0.166666671633720397949219f);
  u = std::fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + std::fma(ss, u, s);
  const auto pow2i = [](int e) {   // the exponent field only (e in -75 .. 75 here)
    const uint32_t b = uint32_t(e + 127) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  };
  u = u * pow2i(q >> 1);
  return u * pow2i(q - (q >> 1));
}

}  // namespace common

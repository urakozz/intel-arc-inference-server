// Spec 23f: writes tools/oracle/fixtures/rtn_int4_g64.json - loader/rtn.h's rtn_int4_g64 on a deterministic bf16
// matrix, so dflash_accept.py's restated int4g64 arm is held to the C++ quantiser bit for bit
// (test_dflash_accept.py::test_int4g64_equals_rtn_h). The matrix covers ties at half a step, a zero group (s16 0),
// a group whose scale is an f16 subnormal, the clamp at 0 / 15, and both signs.
//
//   c++ -std=c++17 -O1 -I src tools/oracle/fixtures/rtn_int4_g64_gen.cc -o /tmp/rtn_gen && /tmp/rtn_gen > \
//     tools/oracle/fixtures/rtn_int4_g64.json
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/bf16.h"
#include "loader/rtn.h"

namespace {
uint16_t to_bf16(float f) {   // round to nearest even (the checkpoint's own values are bf16)
  uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
}  // namespace

int main() {
  const uint32_t N = 6, K = 128;
  std::vector<uint16_t> w(size_t(N) * K);
  uint32_t x = 12345u;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t k = 0; k < K; ++k) {
      x = x * 1664525u + 1013904223u;
      float v = (float(int32_t(x >> 8) % 20001) - 10000.0f) / 2500.0f;   // [-4, 4]
      if (n == 1 && k < 64) v = 0.0f;                                     // a zero group
      if (n == 2 && k < 64) v *= 1e-6f;                                   // an f16-subnormal scale
      if (n == 3) v = float(int32_t(k % 15) - 7) * 0.5f;                  // exact half steps: ties
      if (n == 4 && k == 5) v = 37.0f;                                    // one large value: clamps elsewhere
      w[size_t(n) * K + k] = to_bf16(v);
    }
  std::vector<uint32_t> q(size_t(K / 8) * N);
  std::vector<uint16_t> s(size_t(K / 64) * N);
  loader::rtn_int4_g64(w.data(), K, N, q.data(), s.data());
  std::printf("{\"N\": %u, \"K\": %u,\n \"w_bf16\": [", N, K);
  for (size_t i = 0; i < w.size(); ++i) std::printf("%s%u", i ? "," : "", w[i]);
  std::printf("],\n \"qweight\": [");
  for (size_t i = 0; i < q.size(); ++i) std::printf("%s%u", i ? "," : "", q[i]);
  std::printf("],\n \"scales_f16\": [");
  for (size_t i = 0; i < s.size(); ++i) std::printf("%s%u", i ? "," : "", s[i]);
  std::printf("]}\n");
  return 0;
}

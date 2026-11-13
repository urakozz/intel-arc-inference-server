// kv8_sim_test - the f32 judges' int8 KV simulation (tools/oracle/int8_sim.py kv8_encode / kv8_dequant / kv8_fq,
// read by tools/oracle/qwen_forced.py --kv8) against src/common/kv8.h, bit for bit, on
// tools/oracle/kv8_sim_fixture.py's rows (tests/loader/kv8_sim_fixture.safetensors). No device: kv8_kernels_test
// holds kv8.h bitwise to the writer kernel (src/kernels/kv8.cl), so equal here is equal to the cache's bytes.
//
//   per row: the 256 int8 codes and the fp16 scale's bits equal kv8::encode's; deq equal kv8::dequant's bits; xhat
//   (the row back in the original basis) equal to fp32(unrotate(deq)) computed in fp64 - exact there (one row's deq
//   is a common scale times integers), so one rounding, and the bits must agree.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "check.h"
#include "common/kv8.h"
#include "loader/safetensors.h"

namespace kv8 = common::kv8;

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "tests/loader/kv8_sim_fixture.safetensors";
  const loader::MappedFile file(path);
  uint64_t hlen = 0;
  std::memcpy(&hlen, file.data(), 8);
  std::map<std::string, loader::TensorInfo> t;
  for (auto& [n, ti] : loader::SafetensorsSet::parse_header(file.data(), file.size())) t[n] = ti;
  auto raw = [&](const char* n, const char* dtype) {
    CHECK(t.count(n) == 1);
    CHECK_EQ(t.at(n).dtype, std::string(dtype));
    return file.data() + 8 + hlen + t.at(n).begin;
  };
  const uint32_t D = kv8::kHeadDim;
  const size_t R = t.at("x").shape.at(0);
  CHECK_EQ(t.at("x").shape.at(1), uint64_t(D));
  const uint16_t* x = reinterpret_cast<const uint16_t*>(raw("x", "BF16"));
  const int8_t* q = reinterpret_cast<const int8_t*>(raw("q", "I8"));
  const int32_t* s16 = reinterpret_cast<const int32_t*>(raw("s16", "I32"));
  const float* deq = reinterpret_cast<const float*>(raw("deq", "F32"));
  const float* xhat = reinterpret_cast<const float*>(raw("xhat", "F32"));
  size_t zero = 0, sub = 0, codes = 0;
  for (size_t r = 0; r < R; ++r) {
    float row[kv8::kHeadDim];
    for (uint32_t i = 0; i < D; ++i) row[i] = kv8::bits_f32(uint32_t(x[r * D + i]) << 16);
    int8_t want[kv8::kHeadDim];
    const uint16_t s = kv8::encode(row, want);
    CHECK_EQ(uint32_t(s16[r]), uint32_t(s));
    zero += s == 0;
    sub += s != 0 && s < 0x400;
    double v[kv8::kHeadDim];
    for (uint32_t i = 0; i < D; ++i) {
      CHECK_EQ(int(q[r * D + i]), int(want[i]));
      const float d = kv8::dequant(want[i], s);
      CHECK_EQ(kv8::f32_bits(deq[r * D + i]), kv8::f32_bits(d));
      v[i] = kv8::sign_neg(i) ? -double(d) : double(d);
      codes += want[i] != 0;
    }
    for (uint32_t h = 1; h < D; h <<= 1)
      for (uint32_t i = 0; i < D; ++i)
        if ((i & h) == 0) {
          const double a = v[i], c = v[i + h];
          v[i] = a + c;
          v[i + h] = a - c;
        }
    for (uint32_t i = 0; i < D; ++i) CHECK_EQ(kv8::f32_bits(xhat[r * D + i]), kv8::f32_bits(float(v[i] * 0.0625)));
  }
  CHECK(zero >= 1 && sub >= 1);   // the fixture reaches the zero scale and the fp16 subnormals
  std::printf("kv8_sim_test: %zu rows (%zu zero scales, %zu subnormal fp16 scales, %zu non-zero codes): int8_sim's "
              "codes, scales, deq and xhat bitwise kv8.h's\nkv8_sim_test: PASS\n", R, zero, sub, codes);
  return 0;
}

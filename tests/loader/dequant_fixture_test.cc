// The dequant contract across languages: the oracle's bf16 dequant of a
// random packed tensor must equal the C++ side bit-for-bit. If this fails,
// the loader and the golden tensors disagree about what the nibbles mean and
// NOTHING downstream can be trusted.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "loader/safetensors.h"

int main(int argc, char** argv) {
  std::string path = argc > 1 ? argv[1] : "tests/golden/dequant_fixture.safetensors";
  loader::MappedFile f(path);
  auto entries = loader::SafetensorsSet::parse_header(f.data(), f.size());
  uint64_t hlen = 0;
  std::memcpy(&hlen, f.data(), 8);
  const uint8_t* base = f.data() + 8 + hlen;

  const loader::TensorInfo *qw = nullptr, *sc = nullptr, *dq = nullptr;
  for (auto& [name, t] : entries) {
    if (name == "qweight") qw = &t;
    else if (name == "scales") sc = &t;
    else if (name == "dequant") dq = &t;
  }
  CHECK(qw && sc && dq);
  common::Int4Gptq w;
  w.K = uint32_t(qw->shape[0]) * 8;
  w.N = uint32_t(qw->shape[1]);
  const uint32_t* qp = reinterpret_cast<const uint32_t*>(base + qw->begin);
  const uint16_t* sp = reinterpret_cast<const uint16_t*>(base + sc->begin);
  w.qweight.assign(qp, qp + size_t(w.K / 8) * w.N);
  w.scales.assign(sp, sp + size_t(w.K / 64) * w.N);
  const uint16_t* golden = reinterpret_cast<const uint16_t*>(base + dq->begin);

  size_t checked = 0;
  for (uint32_t k = 0; k < w.K; ++k)
    for (uint32_t n = 0; n < w.N; ++n, ++checked) {
      uint16_t ours = common::f32_to_bf16(w.at(k, n));
      uint16_t theirs = golden[size_t(k) * w.N + n];
      if (ours != theirs) {
        uint32_t word = w.qweight[size_t(k / 8) * w.N + n];
        uint16_t scale = w.scales[size_t(k / 64) * w.N + n];
        std::fprintf(stderr,
                     "dequant mismatch at k=%u n=%u: ours=0x%04X oracle=0x%04X "
                     "(word=0x%08X nibble q=%u scale=0x%04X=%.9g)\n",
                     k, n, ours, theirs, word, (word >> (4 * (k % 8))) & 0xFu, scale,
                     double(common::f16_to_f32(scale)));
        return 1;
      }
    }
  CHECK_EQ(checked, size_t(w.K) * w.N);
  std::printf("dequant_fixture_test OK (%zu elements bit-exact)\n", checked);
  return 0;
}

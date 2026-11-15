#include "loader/kolibri1_repack.h"
#include "loader/rope_table.h"

namespace loader {

// kolibri_ref.rope_cos_sin, step for step in fp32 (the sliding layers' RoPE; the full layers
// have none):
//   inv  = 1.0 / (theta ** (arange(0, 128, 2).float() / 128))   fp32 pow, fp32 divide
//   freqs = inv @ position                                        one fp32 product
//   cos, sin of cat(freqs, freqs)  -> .to(bf16)
// Half a head of angles per position (dims i and i + 64 share one). tests/loader/
// kolibri1_rope_test.cc holds it to torch's values (tests/kernels/kolibri_fixture.h) bit for bit.
std::vector<float> kol_rope_table(const model::Kolibri1Desc& d, uint32_t max_len) {
  return bf16_rope_table(d.head_dim, float(d.rope_theta), max_len);   // 1e4 is exact in fp32
}

}  // namespace loader

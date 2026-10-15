#pragma once
#include <cstdint>
#include <vector>

// Spec 23c: Muse-Glimmer's RoPE table on the host - the sliding layers' cos / sin (the full layers are NoPE), the
// rotary module's values step for step in fp32 (transformers 5.19.0 MuseGlimmerTextRotaryEmbedding, default rope):
//   inv   = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))     fp32 pow, fp32 divide
//   ang   = position x inv                                                    one fp32 product
//   cos, sin of cat(ang, ang), then .to(bf16)
// Half a head of angles per position (dims i and i + 64 share one): fp32 [max_len][2][head_dim / 2] (cos then sin),
// each a bf16 value widened - what mg_attn_prep reads. tests/kernels/muse_ref_test.cc holds the rows to the module's
// (tests/kernels/muse_fixture.h) bit for bit at 0 .. 131071. Kolibri's loader::kol_rope_table's arithmetic at
// Muse's theta 5e5.
namespace runtime::muse {

std::vector<float> rope_table(uint32_t head_dim, float theta, uint32_t max_len);
inline size_t rope_table_bytes(uint32_t head_dim, uint32_t max_len) { return size_t(max_len) * head_dim * 4; }

}  // namespace runtime::muse

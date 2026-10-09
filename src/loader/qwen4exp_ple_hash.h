#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Spec 21b Task 4: the PLE n-gram hash of Qwen3.8-Flash-Next in EXACT integers - header-only arithmetic, so
// the planner (runtime/qwen4exp/qwen4exp_sizes.h: the host table's bytes) and the loader (loader/
// qwen4exp_ple.h: the file's I64 tensors held to it) share one copy, and 21c's q4_ple_gather has its host
// twin. Restated from transformers 5.19.0 modeling_qwen4_exp.py (M:1026-1166; 21a's facts sheet, "PLE ids"):
//
//   multipliers  m_i = 2 x (splitmix64((seed + 10007 x ple_index + 0x9E3779B97F4A7C15 x (i + 1)) mod 2^64)
//                mod half) + 1, half = max(1, ((2^63 - 1) / vocab) / 2): odd, and t x m < 2^63 for t < vocab
//   head sizes   head h's vocabulary is the (ple_index x heads + h + 1)-th prime after base - 1 (trial
//                division, M:1052-1066); offsets their running sum; the table padded to a multiple of 128
//   ids          t0 the token, t1 / t2 its predecessors with the EOS rule (q4_ple_history);
//                mixed2 = (t0 m0) ^ (t1 m1), mixed3 = mixed2 ^ (t2 m2); head h < 8 (bigram): mixed2 mod
//                size_h + offset_h, h >= 8 (trigram): mixed3 mod size_h + offset_h
//
// Nothing touches a float (Review Focus 3): every value is a non-negative int64 in uint64_t.
namespace loader {

inline uint64_t q4_splitmix64(uint64_t v) {
  v += 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}

// M:1040-1049 (_build_layer_multipliers). ngram must be 3 (the family's: bigram + trigram heads).
inline std::array<uint64_t, 3> q4_ple_multipliers(uint32_t vocab, uint32_t ngram, uint32_t ple_index, uint64_t seed) {
  if (ngram != 3) throw std::invalid_argument("q4_ple_multipliers: ngram_size " + std::to_string(ngram) + " (3 only)");
  const uint64_t max_long = (uint64_t(1) << 63) - 1;
  const uint64_t mmax = max_long / (vocab ? vocab : 1);
  const uint64_t half = mmax / 2 ? mmax / 2 : 1;
  const uint64_t base = seed + 10007ull * ple_index;
  std::array<uint64_t, 3> m{};
  for (uint32_t i = 0; i < 3; ++i) m[i] = 2 * (q4_splitmix64(base + 0x9E3779B97F4A7C15ull * (i + 1)) % half) + 1;
  return m;
}

inline bool q4_is_prime(uint64_t v) {
  if (v < 2) return false;
  if (v % 2 == 0) return v == 2;
  for (uint64_t d = 3; d * d <= v; d += 2)
    if (v % d == 0) return false;
  return true;
}

// M:1052-1105: the `heads` head sizes of PLE layer `ple_index` (each the next prime, walking from base - 1).
inline std::vector<uint64_t> q4_ple_primes(uint64_t base, uint32_t heads, uint32_t ple_index) {
  std::vector<uint64_t> out;
  uint64_t p = base - 1;
  const uint64_t skip = uint64_t(ple_index) * heads;
  for (uint64_t i = 0; i < skip + heads; ++i) {
    ++p;
    while (!q4_is_prime(p)) ++p;
    if (i >= skip) out.push_back(p);
  }
  return out;
}
inline std::vector<uint64_t> q4_ple_offsets(const std::vector<uint64_t>& sizes) {
  std::vector<uint64_t> o(sizes.size());
  uint64_t at = 0;
  for (size_t h = 0; h < sizes.size(); ++h) {
    o[h] = at;
    at += sizes[h];
  }
  return o;
}
inline uint64_t q4_ple_total_rows(const std::vector<uint64_t>& sizes) {
  uint64_t t = 0;
  for (uint64_t s : sizes) t += s;
  return t;
}

// The EOS rule (M:1107-1121): token `pos` of `seq` and its two predecessors, a predecessor that does not
// exist or lies at / before the last EOS strictly before `pos` reading as `eos`.
struct Q4PleHistory {
  uint32_t t0 = 0, t1 = 0, t2 = 0;
};
inline Q4PleHistory q4_ple_history(const uint32_t* seq, size_t pos, uint32_t eos) {
  int64_t last_eos = -1;
  for (size_t i = 0; i < pos; ++i)
    if (seq[i] == eos) last_eos = int64_t(i);
  Q4PleHistory h;
  h.t0 = seq[pos];
  h.t1 = int64_t(pos) - 1 >= 0 && int64_t(pos) - 1 > last_eos ? seq[pos - 1] : eos;
  h.t2 = int64_t(pos) - 2 >= 0 && int64_t(pos) - 2 > last_eos ? seq[pos - 2] : eos;
  return h;
}

// The 16 global row ids of one position (8 bigram heads, then 8 trigram heads).
inline std::array<uint64_t, 16> q4_ple_ids(uint32_t t0, uint32_t t1, uint32_t t2, const std::array<uint64_t, 3>& mult,
                                           const std::vector<uint64_t>& sizes, const std::vector<uint64_t>& offsets) {
  if (sizes.size() != 16 || offsets.size() != 16)
    throw std::invalid_argument("q4_ple_ids: 16 heads (8 bigram + 8 trigram) expected");
  const uint64_t mixed2 = (uint64_t(t0) * mult[0]) ^ (uint64_t(t1) * mult[1]);
  const uint64_t mixed3 = mixed2 ^ (uint64_t(t2) * mult[2]);
  std::array<uint64_t, 16> ids{};
  for (uint32_t h = 0; h < 16; ++h) ids[h] = (h < 8 ? mixed2 : mixed3) % sizes[h] + offsets[h];
  return ids;
}

// The int8 file's scale dtype (decision 7: f32 - spec 9's rule - or bf16).
enum class Q4PleScale { F32, Bf16 };
inline const char* q4_ple_scale_name(Q4PleScale s) { return s == Q4PleScale::Bf16 ? "bf16" : "f32"; }
inline size_t q4_ple_scale_bytes(Q4PleScale s) { return s == Q4PleScale::Bf16 ? 2 : 4; }
// The pinned host table: every head's rows x (dim int8 + one scale).
inline size_t q4_ple_table_bytes(const std::vector<uint64_t>& sizes, uint32_t dim, Q4PleScale s) {
  return size_t(q4_ple_total_rows(sizes)) * (dim + q4_ple_scale_bytes(s));
}

}  // namespace loader

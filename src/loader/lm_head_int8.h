#pragma once
// Spec 9 §3: the int8 `lm_head`, made at load from the checkpoint's bf16 tensor.
//
// **The quantisation** is plan 9a's, the one its accuracy verdict was measured on
// (docs/probe-lm-head-2026-09-28.md §1): symmetric, one fp32 scale per row,
// `s = max|W_row| / 127` (in fp32), `q = rne(W / s)` clamped to [-127, 127]. The
// hidden is not quantised (W8A16); the kernel accumulates `q x` in fp32 and
// multiplies by `s` once per row. A zero row gets `s = 0` and `q = 0`.
//
// **The layout** is gemv_i8w's (src/kernels/gemv_i8w.cl): `[N/16][K/16][16 k][16 n]`
// int8, so one `intel_sub_group_block_read_uc16` hands lane `n % 16` the 16 k of its
// column - 256 contiguous bytes per subgroup per step, the same 256 B `gemv_bf16`
// streams per step. Scales are a separate fp32 `[N]` allocation
// (`DeviceWeight::scales`), 993 280 B at the real shape.
#include <cstddef>
#include <cstdint>
#include <string>

namespace loader {

// Which `lm_head` the loader builds. `Checkpoint` is the tensor as shipped (bf16 on
// the gate checkpoint, int4 g64 on a `--quant_lm_head` one) - exactly the engine
// before spec 9. `Int8` quantises a bf16 head at load; it refuses a checkpoint whose
// head is already packed.
enum class LmHeadForm { Checkpoint, Int8 };

// The CLI spelling: "bf16" (the checkpoint's head; the name the flag has always
// meant on the gate checkpoint) or "int8". False on anything else.
bool parse_lm_head_form(const std::string& s, LmHeadForm& out);
const char* lm_head_form_name(LmHeadForm f);

// Byte offset of element (k, n) in the tiled int8 weight.
inline size_t int8_tiled_index(uint32_t K, uint32_t k, uint32_t n) {
  return (size_t(n / 16) * (K / 16) + k / 16) * 256 + size_t(k % 16) * 16 + n % 16;
}
// Weight bytes plus scale bytes: K N + 4 N.
inline size_t lm_head_int8_bytes(uint32_t K, uint32_t N) { return size_t(K) * N + size_t(N) * 4; }

// One bf16 row of K elements into q[K] (row order, not tiled); returns its scale.
float quantise_row_int8(const uint16_t* w, uint32_t K, int8_t* q);

// The whole [N][K] bf16 row-major tensor into the tiled layout above plus scales[N].
// `threads` = 0 uses std::thread::hardware_concurrency(). The result does not depend
// on the thread count (each row is one thread's, start to finish).
void quantise_int8_tiled(const uint16_t* w, uint32_t K, uint32_t N, int8_t* q_tiled,
                         float* scales, unsigned threads = 0);

// Spec 8 §11: the draft vocabulary's compact head. Rows `ids[0..n)` of a tiled [N][K]
// int8 head (the layout above) and their scales, gathered into the SAME tiled layout at
// N' = n: compact row j is row ids[j], byte for byte, and out_scales[j] = scales[ids[j]].
// So `gemv_i8w` at N = n over the result runs, for compact column j, the instruction
// sequence it runs for column ids[j] over the full head - the bitwise property
// draft_vocab_kernels_test checks on the card. n must be a multiple of 16 (whole tiles)
// and every id < N. `threads` as above; the result does not depend on it (each output
// tile is one thread's).
void gather_int8_tiled_rows(const int8_t* q_tiled, const float* scales, uint32_t K, uint32_t N,
                            const uint32_t* ids, uint32_t n, int8_t* out_tiled,
                            float* out_scales, unsigned threads = 0);

}  // namespace loader

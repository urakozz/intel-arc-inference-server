#pragma once
// Spec 9 §3: the int8 `lm_head`, made at load from the checkpoint's bf16 tensor.
//
// **The quantisation** is plan 9a's, the one its accuracy verdict was measured on
// symmetric, one fp32 scale per row,
// `s = max|W_row| / 127` (in fp32), `q = rne(W / s)` clamped to [-127, 127]. The
// hidden is not quantised (W8A16); the kernel accumulates `q x` in fp32 and
// multiplies by `s` once per row. A zero row gets `s = 0` and `q = 0`.
//
// **The layout** is gemv_i8w's (src/kernels/gemv_i8w.cl): 256-byte `[16 k][16 n]` int8
// tiles, so one `intel_sub_group_block_read_uc16` hands lane `n % 16` the 16 k of its
// column - 256 contiguous bytes per subgroup per step, the same 256 B `gemv_bf16`
// streams per step. Scales are a separate fp32 `[N]` allocation
// (`DeviceWeight::scales`), 993 280 B at the real shape. The tiles are ordered
// `[N/16][K/16]` (n-major: a 16-column tile's K walk contiguous) or, for an output head
// while kInt8HeadKMajor holds, `[K/16/B][N/16][B]` (k-major in blocks of B = kInt8HeadKBlock steps) -
// int8_tiled_index below.
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

// PERF-111: an int8 output head's tiles in k-major order, so at each 16-k step the launch's sub-groups read
// neighbouring 256 B rather than each walking its own K x 16 slab (int8 head 579 -> 591 GB/s on Qwen3.8, one card,
// M = 1). The same tiles, the same arithmetic in the same order: every output is bitwise the n-major order's.
// **The one switch for the whole layout:** the loader writes the heads by int8_tiled_index, and
// src/kernels/CMakeLists.txt reads the two lines below (keep their spelling) to build every reader of a tensor of a
// head width with GEMV_KMAJOR=1 - gemv_i8w, gemm_sm_i8w (the verify rows) and df_rerank. A head is an int8 tensor
// of one of these widths: the full vocabulary (248320, Kolibri's 128000), a tp card's half (124160) and Inkling 2's head and
// its folded Markov head (200064, plan 41f). No other int8 tiled tensor has one of these widths (the DFlash drafter's linears keep n-major: their k||v rows and df_w_slab_i8 read column ranges in place).
// Flash-Next's hyper-connection mixers (quantise_int8_g32_tiled, q4_hc.cl) keep their own n-major tiles.
// K-blocked (kInt8HeadKBlock, the kernels' GEMV_KMAJOR_B): kInt8HeadKBlock consecutive 16-k steps of a tile stay
// together and those blocks are k-major. Plain k-major (one step a block) lost 6-8 % on gemm_sm_i8w at 64 rows: that
// build is latency-bound (one weight read in flight a thread, ~410 GB/s), and a work-group's successive k-groups
// land N/16 tiles apart instead of continuing its own slab. Blocks of 8 and the 32 / 64-row builds' read ahead
// (gemm_sm I8_AHEAD) keep those rows level and leave the gains at M 1 (probe_gemv --tiles i8head).
inline constexpr bool kInt8HeadKMajor = true;
inline constexpr uint32_t kInt8HeadKBlock = 8;
inline constexpr uint32_t kInt8HeadWidths[] = {248320, 124160, 128000, 200064};
inline constexpr bool int8_k_major(uint32_t N) {
  if (!kInt8HeadKMajor) return false;
  for (uint32_t w : kInt8HeadWidths)
    if (w == N) return true;
  return false;
}

// Byte offset of element (k, n) in a tiled int8 [N][K] weight: k-major for a head (int8_k_major), else n-major.
inline size_t int8_tiled_index(uint32_t K, uint32_t N, uint32_t k, uint32_t n) {
  const size_t B = kInt8HeadKBlock;
  const size_t tile = int8_k_major(N) ? (size_t(k / 16) / B * (N / 16) + n / 16) * B + k / 16 % B
                                      : size_t(n / 16) * (K / 16) + k / 16;
  return tile * 256 + size_t(k % 16) * 16 + n % 16;
}
// Weight bytes plus scale bytes: K N + 4 N.
inline size_t lm_head_int8_bytes(uint32_t K, uint32_t N) { return size_t(K) * N + size_t(N) * 4; }

// One bf16 row of K elements into q[K] (row order, not tiled); returns its scale.
float quantise_row_int8(const uint16_t* w, uint32_t K, int8_t* q);

// The whole [N][K] bf16 row-major tensor into the tiled layout above (int8_tiled_index) plus scales[N].
// `threads` = 0 uses std::thread::hardware_concurrency(). The result does not depend
// on the thread count (each row is one thread's, start to finish).
void quantise_int8_tiled(const uint16_t* w, uint32_t K, uint32_t N, int8_t* q_tiled,
                         float* scales, unsigned threads = 0);

// PERF-66 (spec 21-QFN-v2 §7 lever 5): Flash-Next's hyper-connection mixers at load - the same tiled layout, one
// fp16 scale per 32 k of a row instead of one fp32 scale a row: for row n, group g (k in [32g, 32g + 32)),
// `s = fp16_rne(max|W| / 127)` (the quotient in fp32; subnormal halves kept), `q = rne(W / float(s))` clamped to
// [-127, 127]; a group whose s is 0 gets q = 0. scales_f16 is [K/32][N] (group-major: a sub-group's 16 columns
// read 32 contiguous bytes). K must be a multiple of 32, N of 16. `threads` as quantise_int8_tiled's; the result
// does not depend on it.
void quantise_int8_g32_tiled(const uint16_t* w, uint32_t K, uint32_t N, int8_t* q_tiled, uint16_t* scales_f16,
                             unsigned threads = 0);

}  // namespace loader

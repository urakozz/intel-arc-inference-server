#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Spec 26b Task 2: ModelOpt's NVFP4 (W4A16, group 16) decoded EXACTLY on the host - the LiLiCorr drafter's checkpoint
// form (spec 26 §3), never a kernel's: our loader turns it into one of our formats at load.
//
// An NVFP4 linear is three tensors: `weight` U8 [N][K/2] (two e2m1 codes a byte, element 2i in the LOW nibble and 2i + 1
// in the high one), `weight_scale` F8_E4M3 [N][K/16] (one scale s_b per 16 consecutive k) and `weight_scale_2` F32 [] (the
// tensor's global g). W = g x s_b x e. What is exact (tools/oracle/dflash_ref.py nvfp4_decode, the same rules):
//   - W' = s_b x e is exact in bf16 AND f16: e has at most 2 significant bits (0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6), s_b
//     at most 4 (e4m3fn; subnormals m x 2^-9; |s_b| <= 448), so W' has at most 6, |W'| in [2^-10, 2688];
//   - g x W' is NOT exact in bf16 (nor, in general, fp32): g stays apart and multiplies the linear's fp32 OUTPUT once
//     (y = g x (W' x) - an NVFP4 GEMM's alpha).
// The two e4m3fn NaN codes (S.1111.111: 0x7F, 0xFF) are refused naming the tensor and the element; e4m3fn has no inf.
//
// The drafter formats built from (W', g) (spec 26 §3, Decision 2):
//   exact          W' as bf16 + g (the loader's tiles + a per-column fp32 output scale)
//   int8 per row   from W' exactly: s'_r = amax_r(W') / 127, q = rne(W' / s'_r) - spec 9's quantiser, independent of g -
//                  and the stored row scale fp32(g x s'_r): the one g-dependent rounding (quantise_int8_tiled_g)
//   int4 g64 RTN   rtn.h's rule over g x W' in fp64 (rtn_int4_g64_g): amax exact, s16 = f16(2 amax / 15) rounded ONCE
//                  from fp64, q = clamp(rint(v / s16) + 8, 0, 15) in fp64 (dflash_accept.rtn_int4_g64_f64, 26a ruling 9)
namespace loader {

// e2m1 code (bit 3 the sign) -> its value; e4m3fn code -> its value (exact in fp32; NaN codes give NaN).
float e2m1_value(uint8_t code);
bool e4m3_is_nan(uint8_t code);
float e4m3_value(uint8_t code);

// W' bf16 row-major [N][K] from the U8 codes [N][K/2] and the e4m3 scales [N][K/16]. Throws std::runtime_error naming
// `name` and the scale's [row][block] on a NaN code, and when K is not a multiple of 16.
std::vector<uint16_t> decode_nvfp4(const uint8_t* weight, const uint8_t* scale, uint32_t K, uint32_t N,
                                   const std::string& name);

// fp64 -> f16 bits with ONE rounding (to nearest, ties to even; subnormals kept; overflow to inf) - numpy's
// npy_double_to_half, what dflash_accept.f64_to_f16_rne uses.
uint16_t f64_to_f16_rne(double x);

// rtn.h's GPTQ-native form (qweight [K/8][N], f16 scales [K/64][N]) of g x W' (W' bf16 row-major [N][K]) in fp64.
void rtn_int4_g64_g(const uint16_t* wp, double g, uint32_t K, uint32_t N, uint32_t* qweight, uint16_t* scales);

// spec 9's int8 tiles (loader::quantise_int8_tiled) of W' - the codes - and the stored scales fp32(g x s'_r).
void quantise_int8_tiled_g(const uint16_t* wp, float g, uint32_t K, uint32_t N, int8_t* q_tiled, float* scales,
                           unsigned threads = 0);

}  // namespace loader

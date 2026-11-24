#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#include "bf16.h"
#include "expert_fp.h"

__kernel void expert_fp_bytes(const __global uint* words, __global uint* out) {
  const uint lane = get_local_id(0);
  const uint k = get_group_id(0);
  const __global uint* tile = words + get_group_id(1) * TILE_U32;
  const float scale = fp_tile_scale(tile, lane, k);
  const float weight = fp_weight(FP_TILE_WORD(tile, lane, k), k % FP_VALUES_PER_WORD, scale);
  const size_t i = (get_group_id(1) * 64 + k) * 16 + lane;
  out[2 * i] = as_uint(weight);
  out[2 * i + 1] = rne_bf16(weight);
}

// 3f probe-only candidates: 1 native finite scales, 2 packed V plus scale broadcasts.
inline float probe_scale(__global const ushort* p, size_t index) {
  return vload_half(index, (__global const half*)p);
}

// K stays int8 in registers with its row's scale beside it (16 bytes + one float per lane,
// where 16 dequantised floats would be 16 GRF); the score dot dequantises at use, and since
// float(q8) * scale is exact the operand is the same value either way.
inline void load_k8(__global const char* restrict kv_k, __global const ushort* restrict k_scale,
                    uint p, uint j, bool ok, char* kq, float* ksc) {
  if (ok) {   // uniform within the sub-group
    const size_t r = (size_t)p * KV_HEADS + j;
    __global const uchar* row = (__global const uchar*)(kv_k + r * HD);
    *ksc = probe_scale(k_scale, r);
    const uchar8 a = intel_sub_group_block_read_uc8(row);
    const uchar8 b = intel_sub_group_block_read_uc8(row + 128);
    kq[0] = as_char(a.s0);  kq[1] = as_char(a.s1);  kq[2] = as_char(a.s2);  kq[3] = as_char(a.s3);
    kq[4] = as_char(a.s4);  kq[5] = as_char(a.s5);  kq[6] = as_char(a.s6);  kq[7] = as_char(a.s7);
    kq[8] = as_char(b.s0);  kq[9] = as_char(b.s1);  kq[10] = as_char(b.s2); kq[11] = as_char(b.s3);
    kq[12] = as_char(b.s4); kq[13] = as_char(b.s5); kq[14] = as_char(b.s6); kq[15] = as_char(b.s7);
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kq[t] = 0;
    *ksc = 0.0f;
  }
}

inline void load_v8(__global const char* restrict kv_v, __global const ushort* restrict v_scale,
                    uint p0, uint j, uint lid, uint last, float* vreg) {
#if KV8_PROBE_KIND == 2
  // A lane loads one packed pair per position. Broadcast the pair for dim lid,
  // then select its signed byte. Overlapping blocks stay within the head's
  // bounded surface; lanes past the last row read zero.
  ushort packed[WAVE_P];
  const uint lane = lid & 15u;
  intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)(kv_v + (size_t)j * HD),
      HD, last + 1, KV_HEADS * HD, (int2)((int)((lid >> 4) * 8u), (int)p0), packed);
  const uint p = p0 + lane;
  const float scale = p <= last ? probe_scale(v_scale, (size_t)p * KV_HEADS + j) : 0.0f;
#pragma unroll
  for (uint s = 0; s < WAVE_P; ++s) {
    const uint pair = intel_sub_group_shuffle((uint)packed[s], lane >> 1);
    const char value = as_char((uchar)(pair >> ((lane & 1u) * 8u)));
    const float sc = intel_sub_group_shuffle(scale, s);
    vreg[s] = p0 + s <= last ? (float)value * sc : 0.0f;
  }
#else
  for (uint s = 0; s < WAVE_P; ++s) {
    const uint p = p0 + s;
    if (p <= last) {
      const size_t r = (size_t)p * KV_HEADS + j;
      vreg[s] = (float)kv_v[r * HD + lid] * probe_scale(v_scale, r);
    } else {
      vreg[s] = 0.0f;   // never loaded: an unwritten slot cannot turn 0 x into a NaN
    }
  }
#endif
}

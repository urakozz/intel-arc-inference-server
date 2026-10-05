// tools/mac/opencl/intel_shim.h - the Intel OpenCL C extensions this repo's kernels use,
// for a compiler that does not ship them. Two modes, one file, so the syntax check and
// the Mac runs cannot disagree about a name:
//
//   default              DECLARATIONS ONLY, for `clang -x cl -cl-std=CL3.0 -target spir64
//                        -fsyntax-only -include intel_shim.h` (tools/mac_check.sh, section
//                        4). Apple's clang knows intel_reqd_sub_group_size and the
//                        cl_khr_subgroups builtins (sub_group_reduce_*, get_sub_group_*)
//                        for spir64, but none of the Intel functions ocloc has: the
//                        cl_intel_subgroups block reads and shuffles, their _short / _char
//                        forms, the 2D block I/O, the DPAS matrix-mad, the split barrier.
//                        They are declared below with the signatures of their Intel
//                        extension specs.
//
//   B70_CL_EMULATE       DEFINITIONS, for Apple's OpenCL 1.2 runtime on the Mac's GPU
//                        (tools/mac/clrun, section 5), which has no subgroups at all. A
//                        "subgroup" becomes B70_EMU_SG consecutive work-items of a 1-D
//                        work-group, and the block reads become the plain loads they are
//                        defined to be equivalent to. Only what can be emulated exactly
//                        without cross-lane communication is: the block reads/writes and
//                        the subgroup ids. Reductions, shuffles, DPAS and 2D block I/O are
//                        left undeclared there, so a kernel using them fails to BUILD on
//                        the Mac with the name in the error - never runs with a wrong
//                        stand-in.
//
// **The 2D block and matrix-mad lists are exactly the names this repo's kernels use, not
// the extension's full families.** ocloc on the box exposes only part of each family
// (pf_gemm.cl records ocloc rejecting intel_sub_group_2d_block_read_transpose_32b_16r16x1c,
// a name the spec's pattern suggests), so declaring the pattern would let a name pass here
// that fails on the box. A kernel that needs a new shape fails this check as
// "undeclared"; add it here once it is known to exist on bmg-g31 (an ocloc build on the
// box, or the driver's own header), and say so in the commit.
#ifndef B70_INTEL_SHIM_H
#define B70_INTEL_SHIM_H

#define B70_OVL __attribute__((overloadable))

#ifndef B70_CL_EMULATE
// ---------------------------------------------------------------------------------------
// Declarations (syntax check)
// ---------------------------------------------------------------------------------------

// cl_intel_subgroups (uint, no suffix) / cl_intel_subgroups_short (_us) /
// cl_intel_subgroups_char (_uc), each with cl_intel_subgroup_local_block_io's __local
// forms. Element i of lane l is p[i * SG + l], for SG = the subgroup size. Apple's clang
// declares none of them for spir64 (its opencl-c.h guards them behind a macro the SPIR
// target does not set), so all three families are declared here.
#define B70_BLOCK_IO(T, S)                                                                  \
  T B70_OVL intel_sub_group_block_read##S(const __global T* p);                            \
  T##2 B70_OVL intel_sub_group_block_read##S##2(const __global T* p);                      \
  T##4 B70_OVL intel_sub_group_block_read##S##4(const __global T* p);                      \
  T##8 B70_OVL intel_sub_group_block_read##S##8(const __global T* p);                      \
  T B70_OVL intel_sub_group_block_read##S(const __local T* p);                             \
  T##2 B70_OVL intel_sub_group_block_read##S##2(const __local T* p);                       \
  T##4 B70_OVL intel_sub_group_block_read##S##4(const __local T* p);                       \
  T##8 B70_OVL intel_sub_group_block_read##S##8(const __local T* p);                       \
  void B70_OVL intel_sub_group_block_write##S(__global T* p, T v);                         \
  void B70_OVL intel_sub_group_block_write##S##2(__global T* p, T##2 v);                   \
  void B70_OVL intel_sub_group_block_write##S##4(__global T* p, T##4 v);                   \
  void B70_OVL intel_sub_group_block_write##S##8(__global T* p, T##8 v);                   \
  void B70_OVL intel_sub_group_block_write##S(__local T* p, T v);                          \
  void B70_OVL intel_sub_group_block_write##S##2(__local T* p, T##2 v);                    \
  void B70_OVL intel_sub_group_block_write##S##4(__local T* p, T##4 v);                    \
  void B70_OVL intel_sub_group_block_write##S##8(__local T* p, T##8 v);
B70_BLOCK_IO(uint, )
B70_BLOCK_IO(ushort, _us)
B70_BLOCK_IO(uchar, _uc)
#undef B70_BLOCK_IO

// cl_intel_subgroups' shuffles, over every scalar and vector type the extensions (and
// cl_khr_fp16 / fp64) admit: data from lane `c`, from lane l + delta (down), l - delta
// (up), l ^ value (xor).
#define B70_SHUFFLE_T(T)                                                                    \
  T B70_OVL intel_sub_group_shuffle(T x, uint c);                                           \
  T B70_OVL intel_sub_group_shuffle_down(T cur, T next, uint delta);                        \
  T B70_OVL intel_sub_group_shuffle_up(T prev, T cur, uint delta);                          \
  T B70_OVL intel_sub_group_shuffle_xor(T x, uint value);
#define B70_SHUFFLE(T)                                                                      \
  B70_SHUFFLE_T(T) B70_SHUFFLE_T(T##2) B70_SHUFFLE_T(T##3) B70_SHUFFLE_T(T##4)              \
  B70_SHUFFLE_T(T##8) B70_SHUFFLE_T(T##16)
B70_SHUFFLE(char) B70_SHUFFLE(uchar) B70_SHUFFLE(short) B70_SHUFFLE(ushort)
B70_SHUFFLE(int) B70_SHUFFLE(uint) B70_SHUFFLE(long) B70_SHUFFLE(ulong)
B70_SHUFFLE(float) B70_SHUFFLE(double)
// half needs cl_khr_fp16 enabled to be a parameter type; enabled for these declarations
// only and disabled again, so a kernel that computes in half without its own pragma
// still fails here as it would under ocloc.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
B70_SHUFFLE(half)
#pragma OPENCL EXTENSION cl_khr_fp16 : disable
#undef B70_SHUFFLE
#undef B70_SHUFFLE_T
// The char family alone has a 16-wide form.
uchar16 B70_OVL intel_sub_group_block_read_uc16(const __global uchar* p);
uchar16 B70_OVL intel_sub_group_block_read_uc16(const __local uchar* p);
void B70_OVL intel_sub_group_block_write_uc16(__global uchar* p, uchar16 v);
void B70_OVL intel_sub_group_block_write_uc16(__local uchar* p, uchar16 v);

// cl_intel_subgroup_matrix_multiply_accumulate (DPAS), SIMD16, M = 8 rows: a = 8 rows of
// one lane's K slice, b = the lane's column, VNNI-packed into int8.
float8 B70_OVL intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc);
int8 B70_OVL intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc);

// cl_intel_subgroup_2d_block_io: (surface base, width in bytes, height in rows, pitch in
// bytes, (x in elements, y in rows), per-lane private destination/source).
#define B70_2D_READ(NAME, T)                                                                \
  void intel_sub_group_2d_block_read_##NAME(__global void* base, int width, int height,    \
                                            int pitch, int2 coord, __private T* dst);
#define B70_2D_WRITE(NAME, T)                                                               \
  void intel_sub_group_2d_block_write_##NAME(__global void* base, int width, int height,   \
                                             int pitch, int2 coord, __private T* src);
#define B70_2D_PREFETCH(NAME)                                                               \
  void intel_sub_group_2d_block_prefetch_##NAME(__global void* base, int width, int height, \
                                                int pitch, int2 coord);
B70_2D_READ(16b_8r16x2c, ushort)
B70_2D_READ(16b_16r16x1c, ushort)
B70_2D_READ(16b_32r16x2c, ushort)
B70_2D_READ(32b_8r16x1c, uint)
B70_2D_READ(32b_16r16x1c, uint)
B70_2D_READ(transform_16b_32r16x2c, uint)
B70_2D_READ(transpose_32b_16r8x1c, uint)
B70_2D_WRITE(16b_8r16x1c, ushort)
B70_2D_WRITE(32b_8r16x1c, uint)
B70_2D_PREFETCH(16b_8r16x2c)
B70_2D_PREFETCH(32b_4r16x1c)
#undef B70_2D_READ
#undef B70_2D_WRITE
#undef B70_2D_PREFETCH

// cl_intel_split_work_group_barrier.
void B70_OVL intel_work_group_barrier_arrive(cl_mem_fence_flags flags);
void B70_OVL intel_work_group_barrier_wait(cl_mem_fence_flags flags);
void B70_OVL intel_work_group_barrier_arrive(cl_mem_fence_flags flags, memory_scope scope);
void B70_OVL intel_work_group_barrier_wait(cl_mem_fence_flags flags, memory_scope scope);

#else  // B70_CL_EMULATE
// ---------------------------------------------------------------------------------------
// Emulation (Apple OpenCL 1.2: no subgroups). Valid for 1-D work-groups whose size is a
// multiple of B70_EMU_SG, which every kernel here with intel_reqd_sub_group_size has.
// ---------------------------------------------------------------------------------------
#ifndef B70_EMU_SG
#define B70_EMU_SG 16
#endif
// The attribute names a hardware SIMD width the emulation does not have; drop it (an
// empty __attribute__(()) is valid). reqd_work_group_size stays and is enforced.
#define intel_reqd_sub_group_size(n)

inline uint get_sub_group_local_id(void) { return (uint)get_local_id(0) % B70_EMU_SG; }
inline uint get_sub_group_id(void) { return (uint)get_local_id(0) / B70_EMU_SG; }
inline uint get_sub_group_size(void) { return B70_EMU_SG; }
inline uint get_max_sub_group_size(void) { return B70_EMU_SG; }
inline uint get_num_sub_groups(void) { return (uint)get_local_size(0) / B70_EMU_SG; }

// Block reads: element i of lane l is p[i * SG + l]. Each lane loads its own elements,
// which is what the hardware message returns; no lane needs another lane's data.
#define B70_EMU_RD(NAME, T, VT, N, AS)                                                      \
  inline VT B70_OVL NAME(const AS T* p) {                                                   \
    const uint l = get_sub_group_local_id();                                                \
    T v[N];                                                                                 \
    for (int i = 0; i < N; ++i) v[i] = p[i * B70_EMU_SG + l];                               \
    return vload##N(0, v);                                                                  \
  }
#define B70_EMU_RD1(NAME, T, AS)                                                            \
  inline T B70_OVL NAME(const AS T* p) { return p[get_sub_group_local_id()]; }
#define B70_EMU_WR(NAME, T, VT, N, AS)                                                      \
  inline void B70_OVL NAME(AS T* p, VT x) {                                                 \
    const uint l = get_sub_group_local_id();                                                \
    T v[N];                                                                                 \
    vstore##N(x, 0, v);                                                                     \
    for (int i = 0; i < N; ++i) p[i * B70_EMU_SG + l] = v[i];                               \
  }
#define B70_EMU_WR1(NAME, T, AS)                                                            \
  inline void B70_OVL NAME(AS T* p, T x) { p[get_sub_group_local_id()] = x; }
#define B70_EMU_FAMILY(S, T, AS)                                                            \
  B70_EMU_RD1(intel_sub_group_block_read##S, T, AS)                                         \
  B70_EMU_RD(intel_sub_group_block_read##S##2, T, T##2, 2, AS)                              \
  B70_EMU_RD(intel_sub_group_block_read##S##4, T, T##4, 4, AS)                              \
  B70_EMU_RD(intel_sub_group_block_read##S##8, T, T##8, 8, AS)                              \
  B70_EMU_WR1(intel_sub_group_block_write##S, T, AS)                                        \
  B70_EMU_WR(intel_sub_group_block_write##S##2, T, T##2, 2, AS)                             \
  B70_EMU_WR(intel_sub_group_block_write##S##4, T, T##4, 4, AS)                             \
  B70_EMU_WR(intel_sub_group_block_write##S##8, T, T##8, 8, AS)
B70_EMU_FAMILY(, uint, __global)
B70_EMU_FAMILY(, uint, __local)
B70_EMU_FAMILY(_us, ushort, __global)
B70_EMU_FAMILY(_us, ushort, __local)
B70_EMU_FAMILY(_uc, uchar, __global)
B70_EMU_FAMILY(_uc, uchar, __local)
B70_EMU_RD(intel_sub_group_block_read_uc16, uchar, uchar16, 16, __global)
B70_EMU_RD(intel_sub_group_block_read_uc16, uchar, uchar16, 16, __local)
B70_EMU_WR(intel_sub_group_block_write_uc16, uchar, uchar16, 16, __global)
B70_EMU_WR(intel_sub_group_block_write_uc16, uchar, uchar16, 16, __local)
#undef B70_EMU_FAMILY
#undef B70_EMU_RD
#undef B70_EMU_RD1
#undef B70_EMU_WR
#undef B70_EMU_WR1
#endif  // B70_CL_EMULATE

#undef B70_OVL
#endif  // B70_INTEL_SHIM_H

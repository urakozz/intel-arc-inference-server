# Spec 5a - production int8 prefill kernels and load-time scales, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the h8 kernels (per-token int8 activations after a Hadamard, per-channel int8 rotated weights, i8 x i8 GEMM) into the production kernel tree, bit-exact against CPU mirrors, together with the host-side state that owns their signs, scales and scratch.

**Architecture:** One OpenCL source, `src/kernels/prefill/pf_int8.cl`, is built as five variant families selected by `-D` (quantiser per K, requant per layout, GEMM plain or SiLU). The arithmetic is `tools/probe/probe_w8a8.cl`'s, unchanged. The probe measured it bit-exact and end-to-end accurate. A host unit `runtime/prefill/int8.{h,cc}` owns the rotation signs, the per-linear column scales (computed once by a GPU pass) and the int8 scratch, and exposes `linear_i8` and `linear_i8_silu` with `linear_l0`'s contracts. Nothing calls it yet; plan 5b wires it in.

**Tech Stack:** OpenCL C via ocloc (AOT, bmg-g31), Level Zero, C++20, CMake/ctest, run on the box via `tools/box.sh`.

**Spec:** `docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md` (stages T1, T2). The evidence is `docs/probe-w4a8-2026-09-23.md` sections 14 and 15.

## Global Constraints

- The checkpoint is unchanged: `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, int4 symmetric g64. Layout 0 (GPTQ-native) for every linear except GDN qkv‖z, which is layout 1 (16-column tiles of 136 u32: 128 of nibbles, then 16 f16 scales).
- R_K = D_K * blockdiag(H_1024) / 32 over each linear's K (5120, 6144, 17408). H is Sylvester (butterfly stages ascending over the natural index). D_K is **`std::mt19937(K)`, bit i = `rng() & 1`, 1 means -1**, generated once on the host.
- Activations: per token (row), symmetric, scale = amax / 127, `convert_char_sat_rte(v * (1 / scale))`, scale 0 means all zero.
- Weights: per output column, scale ws[n] = max_k |(W R_K)[k, n]| / 127 (1.0 if the max is 0), stored with inv[n] = 1 / ws[n], w8 = `convert_char_sat_rte((v * (1 / 32)) * inv[n])`.
- GEMM epilogue: `(float)acc * (ws[n] * xs[m])`. The SiLU epilogue uses **pf_gemm.cl's `PF_SILU_ROW` chain verbatim** (four RNE steps, plain `exp`).
- Every int8 kernel test is **bit-exact** against its CPU mirror (spec bar A1). No tolerances.
- No production caller changes in this plan; `step.cc` and `linear_l0.cc` are untouched.
- Commit messages follow the repo's style (`area: summary`, a body with the measured facts). Branch per task group, fast-forwarded to main when the operator says so.

## Review Focus

- **The two kernels' block layouts must agree on the rotation.** `pf_quant_had` holds element `16j + lane` and runs lane stages first; `pf_requant_rot` holds `64 lane + j` and runs register stages first. Both must equal the CPU canonical FWHT. Pinned by Task 4's cross test: quantise x, requant W, GEMM, compare against the unrotated fp32 product to the §15 accuracy.
- **Layout-1 addressing**, the one new load path. Pinned in Task 2 by building the layout-1 words with `common::repack_int4_layout1` and requiring bit-identity with the layout-0 run on the same weights.
- **Padded rows.** `M` is padded to 256 by the caller; rows [M, pad256(M)) of x hold stale data. The quantiser must turn them into finite values, never NaN, and the GEMM output for real rows must not depend on them. Pinned in Task 4 by filling the padding with NaN bf16 and checking real-row outputs are unchanged.
- **The column-max pass.** Its atomic_max-on-float-bits is valid only for non-negative values. Pinned in Task 3 against the CPU max on weights with negative-heavy columns.
- **A zero column.** A column of all-zero int4 codes gives max 0; the scale must be 1.0 and the outputs 0. Pinned in Task 3 by zeroing one column.

---

### Task 1: `pf_quant_had`, the rotating activation quantiser

**Files:**
- Create: `src/kernels/prefill/pf_int8.cl` (the quantiser section; later tasks append the other sections)
- Modify: `src/kernels/prefill/CMakeLists.txt` (append the pf_int8 block)
- Modify: `src/kernels/prefill/pf_kernels.h` (variant names)
- Create: `src/runtime/prefill/int8_signs.h` (the sign generator, header-only)
- Test: `tests/prefill/pf_int8_test.cc`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: kernel `pf_quant_had(const ushort* x, const float* sgn, char* xq, float* xs, uint ldx)`, WG 16*NBLK, grid = rows; variant `kernels::pf_quant_had_variant(K)` returning `"pf_quant_had_K" + K`; `runtime::prefill::int8_signs(uint32_t K) -> std::vector<float>` (±1) and `int8_sign_bits(uint32_t K) -> std::vector<uint32_t>` (bit k of word k/32 set means -1).

- [x] **Step 1: The sign generator**

`src/runtime/prefill/int8_signs.h`:

```cpp
#pragma once
#include <cstdint>
#include <random>
#include <vector>

// The rotation signs D_K of spec 5 (R_K = D_K blockdiag(H_1024) / 32): one
// mt19937 seeded with K, one bit per k, 1 meaning -1. Host-generated once;
// the kernels and every CPU mirror read the same vector.
namespace runtime::prefill {

inline std::vector<float> int8_signs(uint32_t K) {
  std::mt19937 rng(K);
  std::vector<float> d(K);
  for (float& v : d) v = (rng() & 1u) ? -1.0f : 1.0f;
  return d;
}

inline std::vector<uint32_t> int8_sign_bits(uint32_t K) {
  const std::vector<float> d = int8_signs(K);
  std::vector<uint32_t> b(K / 32, 0u);
  for (uint32_t k = 0; k < K; ++k)
    if (d[k] < 0.0f) b[k / 32] |= 1u << (k % 32);
  return b;
}

}  // namespace runtime::prefill
```

- [x] **Step 2: Write the failing test**

`tests/prefill/pf_int8_test.cc`, the quantiser case. The CPU mirror is the one `tools/probe/probe_w8a8.cc` validated: canonical FWHT per 1024-block after the signs, `* (1/32)`, amax, `iv = 1/scale`, `sat_rte(v * iv)`.

```cpp
// pf_int8 kernels vs CPU mirrors of their arithmetic, BIT-EXACT (spec 5 bar A1).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "common/repack.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "pf_harness.h"
#include "runtime/prefill/int8_signs.h"

namespace {
int8_t sat_rte(float v) {
  const float r = std::nearbyint(v);
  return int8_t(std::fmax(-128.0f, std::fmin(127.0f, r)));
}

// Canonical Sylvester FWHT per 1024-block, stages ascending, unnormalised.
void fwht_blocks(std::vector<float>& r) {
  for (size_t b = 0; b < r.size(); b += 1024)
    for (uint32_t h = 1; h < 1024; h <<= 1)
      for (uint32_t i = 0; i < 1024; i += 2 * h)
        for (uint32_t j = i; j < i + h; ++j) {
          const float a = r[b + j], c = r[b + j + h];
          r[b + j] = a + c;
          r[b + j + h] = a - c;
        }
}

void quant_case(pf_harness::Dev& d, uint32_t K, uint32_t rows) {
  const std::vector<uint16_t> x = pf_harness::random_bf16(size_t(rows) * K, 11 + K, -4.0f, 4.0f);
  const std::vector<float> sgn = runtime::prefill::int8_signs(K);
  l0::Mem dx = pf_harness::upload(d.ctx, d.imm, x);
  l0::Mem ds = pf_harness::upload(d.ctx, d.imm, sgn);
  l0::Mem dq(d.ctx, l0::MemKind::Device, size_t(rows) * K);
  l0::Mem dsc(d.ctx, l0::MemKind::Device, size_t(rows) * 4);
  l0::Module mod(d.ctx, kernels::path(kernels::pf_quant_had_variant(K)));
  l0::Kernel k = mod.kernel("pf_quant_had");
  k.group_size(16 * (K / 1024));
  k.arg_ptr(0, dx.ptr());
  k.arg_ptr(1, ds.ptr());
  k.arg_ptr(2, dq.ptr());
  k.arg_ptr(3, dsc.ptr());
  k.arg(4, K);   // ldx
  d.run(k, rows, 1);
  std::vector<int8_t> q(size_t(rows) * K);
  std::vector<float> sc(rows);
  pf_harness::download(d.imm, q, dq);
  pf_harness::download(d.imm, sc, dsc);
  size_t bad = 0;
  std::vector<float> r(K);
  for (uint32_t m = 0; m < rows; ++m) {
    for (uint32_t kk = 0; kk < K; ++kk) r[kk] = common::bf16_to_f32(x[size_t(m) * K + kk]) * sgn[kk];
    fwht_blocks(r);
    float amax = 0.0f;
    for (float& v : r) { v *= 1.0f / 32.0f; amax = std::fmax(amax, std::fabs(v)); }
    const float scale = amax / 127.0f;
    const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;
    bad += (scale != sc[m]);
    for (uint32_t kk = 0; kk < K; ++kk) bad += (sat_rte(r[kk] * iv) != q[size_t(m) * K + kk]);
  }
  std::printf("pf_quant_had K=%u rows=%u: %zu mismatches\n", K, rows, bad);
  CHECK(bad == 0);
}
}  // namespace

int main() {
  pf_harness::Dev d;
  for (uint32_t K : {5120u, 6144u, 17408u}) quant_case(d, K, 64);
  std::printf("pf_int8_test: PASS\n");
  return 0;
}
```

`tests/CMakeLists.txt`, inside the same `if(B70_PREFILL_ENABLED)` block that holds `pf_gemm_test`:

```cmake
  add_executable(pf_int8_test prefill/pf_int8_test.cc)
  target_include_directories(pf_int8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(pf_int8_test PRIVATE b70_l0)
  b70_target_kernel_dir(pf_int8_test)
  add_dependencies(pf_int8_test kernel_pf_quant_had_K5120 kernel_pf_quant_had_K6144
                   kernel_pf_quant_had_K17408)
  add_test(NAME pf_int8_test COMMAND pf_int8_test)
```

- [x] **Step 3: Run it to see it fail**

Run: `tools/box.sh test pf_int8_test`
Expected: FAIL at configure or build: `kernel_pf_quant_had_K5120` does not exist.

- [x] **Step 4: The kernel, the build rows and the variant name**

`src/kernels/prefill/pf_int8.cl` (the header and the quantiser section):

```c
// pf_int8 - spec 5's int8 prefill linears
// (docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md): the h8 path of
// docs/probe-w4a8-2026-09-23.md sections 14-15, promoted from
// tools/probe/probe_w8a8.cl with the arithmetic unchanged. One source, built
// per family by a define:
//   PF_QUANT_HAD  (NBLK)       pf_quant_had    bf16 x -> int8 (x R_K), per-token scale
//   PF_REQUANT_ROT (LAYOUT)    pf_requant_rot  int4 g64 -> per-channel int8 of (W R_K)
//                              pf_colmax_rot   max |W R_K| per column (load-time scales)
//   PF_GEMM_I8   (SILU_EPI)    pf_gemm_i8      i8 x i8, scales in the epilogue only
// R_K = D_K blockdiag(H_1024) / 32 over K. D_K comes from
// runtime/prefill/int8_signs.h. The quantiser holds block element 16 j + lane
// and the requant holds 64 lane + j; both run the canonical Sylvester FWHT on
// the natural index with ascending stages, so they rotate identically
// (pf_int8_test's cross case pins it).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#define SG 16
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

#ifdef PF_QUANT_HAD
#ifndef NBLK
#error "pf_quant_had: NBLK (= K / 1024) must be defined"
#endif
// One sub-group per 1024-block, NBLK sub-groups per token row. Lane l's value j
// is element 16 j + l of the block (sub-group block reads). FWHT stages
// h = 1..8 are lane shuffles and 16..512 register butterflies, ascending.
__attribute__((reqd_work_group_size(16 * NBLK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_quant_had(__global const ushort* restrict x, __global const float* restrict sgn,
                           __global char* restrict xq, __global float* restrict xs, uint ldx) {
  const uint K = NBLK * 1024u;
  const uint m = get_group_id(0);
  const uint b = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  __global const ushort* src = x + (size_t)m * ldx + b * 1024u;
  __local float red[NBLK];
  float v[64];
#pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    const ushort8 u = intel_sub_group_block_read_us8(src + r * 128u);
    const float8 sg = as_float8(intel_sub_group_block_read8(
        (__global const uint*)(sgn + b * 1024u + r * 128u)));
    v[8u * r + 0u] = bf16f(u.s0) * sg.s0; v[8u * r + 1u] = bf16f(u.s1) * sg.s1;
    v[8u * r + 2u] = bf16f(u.s2) * sg.s2; v[8u * r + 3u] = bf16f(u.s3) * sg.s3;
    v[8u * r + 4u] = bf16f(u.s4) * sg.s4; v[8u * r + 5u] = bf16f(u.s5) * sg.s5;
    v[8u * r + 6u] = bf16f(u.s6) * sg.s6; v[8u * r + 7u] = bf16f(u.s7) * sg.s7;
  }
#pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      const float o = sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
  }
#pragma unroll
  for (uint hj = 1u; hj < 64u; hj <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      if ((j & hj) == 0u) {
        const float a = v[j], c = v[j + hj];
        v[j] = a + c;
        v[j + hj] = a - c;
      }
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (uint j = 0; j < 64u; ++j) {
    v[j] *= (1.0f / 32.0f);
    amax = fmax(amax, fabs(v[j]));
  }
  amax = sub_group_reduce_max(amax);
  if (l == 0u) red[b] = amax;
  barrier(CLK_LOCAL_MEM_FENCE);
  amax = 0.0f;
  for (uint i = 0; i < (uint)NBLK; ++i) amax = fmax(amax, red[i]);
  const float scale = amax / 127.0f;
  const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;
  if (b == 0u && l == 0u) xs[m] = scale;
  __global uchar* dst = (__global uchar*)(xq + (size_t)m * K + b * 1024u);
#pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    uchar8 o;
    o.s0 = (uchar)convert_char_sat_rte(v[8u * r + 0u] * iv);
    o.s1 = (uchar)convert_char_sat_rte(v[8u * r + 1u] * iv);
    o.s2 = (uchar)convert_char_sat_rte(v[8u * r + 2u] * iv);
    o.s3 = (uchar)convert_char_sat_rte(v[8u * r + 3u] * iv);
    o.s4 = (uchar)convert_char_sat_rte(v[8u * r + 4u] * iv);
    o.s5 = (uchar)convert_char_sat_rte(v[8u * r + 5u] * iv);
    o.s6 = (uchar)convert_char_sat_rte(v[8u * r + 6u] * iv);
    o.s7 = (uchar)convert_char_sat_rte(v[8u * r + 7u] * iv);
    intel_sub_group_block_write_uc8(dst + r * 128u, o);
  }
}
#endif  // PF_QUANT_HAD
```

`fmax` ignores NaN operands, so padded rows holding NaN produce a finite scale. Task 4 pins this.

`src/kernels/prefill/CMakeLists.txt`, appended:

```cmake
# Spec 5 (plan 5a): the int8 prefill linears, one source, per-family defines.
# The quantiser and requant run at the default 128 GRF (probe_w8a8 measured 256
# halving their occupancy); the GEMM needs 256 for its 64 int32 accumulators.
set(PF_INT8_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_int8.cl)
foreach(K 5120 6144 17408)
  math(EXPR NBLK "${K} / 1024")
  add_ocloc_kernel(pf_quant_had_K${K} SOURCE ${PF_INT8_CL} DEFINES PF_QUANT_HAD=1 NBLK=${NBLK})
endforeach()
```

`src/kernels/prefill/pf_kernels.h`, before the closing `}  // namespace kernels`:

```cpp
// Spec 5: the int8 prefill linears (pf_int8.cl).
inline std::string pf_quant_had_variant(unsigned K) { return "pf_quant_had_K" + std::to_string(K); }
```

- [x] **Step 5: Run the test to see it pass**

Run: `tools/box.sh test pf_int8_test`
Expected: `pf_quant_had K=5120 rows=64: 0 mismatches` (likewise 6144 and 17408), then `pf_int8_test: PASS`.

- [x] **Step 6: Commit**

```bash
git add src/kernels/prefill/pf_int8.cl src/kernels/prefill/CMakeLists.txt src/kernels/prefill/pf_kernels.h \
        src/runtime/prefill/int8_signs.h tests/prefill/pf_int8_test.cc tests/CMakeLists.txt
git commit -m "kernels: pf_quant_had, the rotating int8 activation quantiser (spec 5 T1)"
```

---

### Task 2: `pf_requant_rot`, layout 0 and layout 1

**Files:**
- Modify: `src/kernels/prefill/pf_int8.cl` (append the requant section)
- Modify: `src/kernels/prefill/CMakeLists.txt`, `src/kernels/prefill/pf_kernels.h`
- Test: `tests/prefill/pf_int8_test.cc`, `tests/CMakeLists.txt` (dependencies)

**Interfaces:**
- Consumes: `int8_sign_bits(K)` (Task 1).
- Produces: kernel `pf_requant_rot(const uint* qw, const half* sc, const uint* sbits, const float* inv, uint* out, uint n0, uint Nfull, uint K, uint ldo)`, WG 256, grid (width/16, K/1024); `out` is VNNI-4 `[K/4][ldo]` (byte b of dword [k/4][n] = k 4d + b); variant `kernels::pf_requant_rot_variant(layout)` returning `"pf_requant_rot_L" + layout`. For layout 1, `sc` is unread (the scales are inline).

- [x] **Step 1: Write the failing test**

Append to `tests/prefill/pf_int8_test.cc` (inside the anonymous namespace), and call it from `main` for `(5120, 1024, L0)`, `(5120, 1024, L1)`, `(6144, 1024, L0)`, `(17408, 1024, L0)`. The same weights, packed both ways, must give identical output (the Review Focus layout-1 line):

```cpp
// W R_K per column on the CPU: dequant (q - 8) * s, signs, FWHT per 1024-block, * 1/32.
std::vector<float> rotated_column(const common::Int4Gptq& w, uint32_t n, const std::vector<float>& sgn) {
  std::vector<float> col(w.K);
  for (uint32_t k = 0; k < w.K; ++k) {
    const uint32_t word = w.qweight[size_t(k / 8) * w.N + n];
    const float s = common::f16_to_f32(w.scales[size_t(k / 64) * w.N + n]);
    col[k] = (float(int((word >> (4 * (k % 8))) & 0xFu) - 8) * s) * sgn[k];
  }
  fwht_blocks(col);
  for (float& v : col) v *= 1.0f / 32.0f;
  return col;
}

void requant_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t layout) {
  const common::Int4Gptq w = common::Int4Gptq::random(K, N, 23 + K + N);
  const std::vector<float> sgn = runtime::prefill::int8_signs(K);
  std::vector<float> inv(N);
  std::vector<std::vector<float>> cols(N);
  for (uint32_t n = 0; n < N; ++n) {
    cols[n] = rotated_column(w, n, sgn);
    float amax = 0.0f;
    for (float v : cols[n]) amax = std::fmax(amax, std::fabs(v));
    inv[n] = 1.0f / (amax > 0.0f ? amax / 127.0f : 1.0f);
  }
  const std::vector<uint32_t> words = layout == 0 ? w.qweight : w.tiled();
  l0::Mem dw = pf_harness::upload(d.ctx, d.imm, words);
  l0::Mem dsc = pf_harness::upload(d.ctx, d.imm, w.scales);
  l0::Mem dbits = pf_harness::upload(d.ctx, d.imm, runtime::prefill::int8_sign_bits(K));
  l0::Mem dinv = pf_harness::upload(d.ctx, d.imm, inv);
  l0::Mem dout(d.ctx, l0::MemKind::Device, size_t(K) * N);
  l0::Module mod(d.ctx, kernels::path(kernels::pf_requant_rot_variant(layout)));
  l0::Kernel k = mod.kernel("pf_requant_rot");
  k.group_size(256);
  k.arg_ptr(0, dw.ptr());
  k.arg_ptr(1, layout == 0 ? dsc.ptr() : nullptr);
  k.arg_ptr(2, dbits.ptr());
  k.arg_ptr(3, dinv.ptr());
  k.arg_ptr(4, dout.ptr());
  k.arg(5, 0u);   // n0
  k.arg(6, N);    // Nfull
  k.arg(7, K);
  k.arg(8, N);    // ldo
  d.run(k, N / 16, K / 1024);
  std::vector<uint32_t> got(size_t(K / 4) * N);
  pf_harness::download(d.imm, got, dout);
  size_t bad = 0;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t kk = 0; kk < K; ++kk) {
      const int8_t want = sat_rte(cols[n][kk] * inv[n]);
      const int8_t have = int8_t((got[size_t(kk / 4) * N + n] >> (8 * (kk % 4))) & 0xFFu);
      bad += (want != have);
    }
  std::printf("pf_requant_rot K=%u N=%u L%u: %zu mismatches\n", K, N, layout, bad);
  CHECK(bad == 0);
}
```

Add `kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1` to `pf_int8_test`'s `add_dependencies`.

- [x] **Step 2: Run it to see it fail**

Run: `tools/box.sh test pf_int8_test`
Expected: FAIL at build (`kernel_pf_requant_rot_L0` missing).

- [x] **Step 3: The kernel**

Append to `pf_int8.cl`. The load is `pw8_requant_rot2`'s for layout 0, with a layout-1 branch reading the tile the WG's 16 columns form. `pf_rot_load` is shared with Task 3's `pf_colmax_rot`:

```c
#ifdef PF_REQUANT_ROT
#ifndef LAYOUT
#error "pf_requant_rot: LAYOUT (0 GPTQ-native, 1 tiled) must be defined"
#endif
// One work-group = 16 columns x one 1024-k block; sub-group c owns column c,
// lane l owns k = 64 l .. 64 l + 63 (one scale group). Stages h = 1..32 are
// register butterflies, h = 64..512 lane shuffles, ascending. The int4 tile
// goes through SLM column-major at pitch 129 (conflict-free); signs arrive as a
// bitmask, applied as a sign-bit xor (bit-identical to multiplying by -1).
inline void pf_rot_load(__global const uint* restrict qw, __global const half* restrict sc,
                        __global const uint* restrict sbits, uint nbase, uint blk, uint Nfull,
                        uint K, __local uint* tin, __local float* tsc, float* v) {
  const uint lid = get_local_id(0);
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
#if LAYOUT == 0
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint t = lid + 256u * i;
    tin[(t % 16u) * 129u + t / 16u] = qw[(size_t)(blk * 128u + t / 16u) * Nfull + nbase + (t % 16u)];
  }
  tsc[(lid % 16u) * 17u + lid / 16u] =
      vload_half((size_t)(blk * 16u + lid / 16u) * Nfull + nbase + (lid % 16u), sc);
#else
  // The WG's 16 columns are exactly tile nt = nbase / 16. Tile (nt, group gi)
  // is 136 u32 at (nt G + gi) 136: word j of lane c at j 16 + c, then 16 f16
  // scales (common/repack.h, repack_int4_layout1). Row r of the block is
  // group blk 16 + r / 8, word r % 8.
  const uint G = K / 64u;
  __global const uint* tiles = qw + (size_t)(nbase / 16u) * G * 136u;
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint t = lid + 256u * i;
    const uint r = t / 16u, cc = t % 16u;
    tin[cc * 129u + r] = tiles[(size_t)(blk * 16u + r / 8u) * 136u + (r % 8u) * 16u + cc];
  }
  {
    const uint g = lid / 16u, cc = lid % 16u;
    const __global ushort* sp =
        (const __global ushort*)(tiles + (size_t)(blk * 16u + g) * 136u + 128u);
    tsc[cc * 17u + g] = (float)as_half(sp[cc]);
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

  const float s = tsc[c * 17u + l];
  const uint2 sb = vload2(0, sbits + blk * 32u + 2u * l);
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint word = tin[c * 129u + 8u * l + i];
#pragma unroll
    for (uint b = 0; b < 8u; ++b) {
      const uint j = 8u * i + b;
      const uint bit = ((j < 32u ? sb.x : sb.y) >> (j % 32u)) & 1u;
      v[j] = as_float(as_uint((float)((int)((word >> (4u * b)) & 0xFu) - 8) * s) ^ (bit << 31));
    }
  }
#pragma unroll
  for (uint h = 1u; h < 64u; h <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      if ((j & h) == 0u) {
        const float a = v[j], e = v[j + h];
        v[j] = a + e;
        v[j + h] = a - e;
      }
    }
  }
#pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      const float o = sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_requant_rot(__global const uint* restrict qw, __global const half* restrict sc,
                             __global const uint* restrict sbits, __global const float* restrict inv,
                             __global uint* restrict out, uint n0, uint Nfull, uint K, uint ldo) {
  const uint lid = get_local_id(0);
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  const uint blk = get_group_id(1);
  const uint nc = get_group_id(0) * 16u;
  const uint nbase = n0 + nc;
  __local uint tin[16 * 129];
  __local float tsc[16 * 17];
  __local uint tout[16 * 257];
  float v[64];
  pf_rot_load(qw, sc, sbits, nbase, blk, Nfull, K, tin, tsc, v);
  const float iv = inv[nbase + c];
#pragma unroll
  for (uint d = 0; d < 16u; ++d) {
    uint w = 0u;
#pragma unroll
    for (uint b = 0; b < 4u; ++b)
      w |= ((uint)(uchar)convert_char_sat_rte((v[4u * d + b] * (1.0f / 32.0f)) * iv)) << (8u * b);
    tout[c * 257u + 16u * l + d] = w;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
  for (uint i = 0; i < 16u; ++i) {
    const uint t = lid + 256u * i;
    out[(size_t)(blk * 256u + t / 16u) * ldo + nc + (t % 16u)] = tout[(t % 16u) * 257u + t / 16u];
  }
}
#endif  // PF_REQUANT_ROT
```

`CMakeLists.txt`, appended to the pf_int8 block:

```cmake
foreach(L 0 1)
  add_ocloc_kernel(pf_requant_rot_L${L} SOURCE ${PF_INT8_CL} DEFINES PF_REQUANT_ROT=1 LAYOUT=${L})
endforeach()
```

`pf_kernels.h`:

```cpp
inline std::string pf_requant_rot_variant(unsigned layout) { return "pf_requant_rot_L" + std::to_string(layout); }
```

- [x] **Step 4: Run the test to see it pass**

Run: `tools/box.sh test pf_int8_test`
Expected: four `pf_requant_rot ... 0 mismatches` lines, the L1 case included; PASS.

- [x] **Step 5: Commit**

```bash
git add src/kernels/prefill/pf_int8.cl src/kernels/prefill/CMakeLists.txt src/kernels/prefill/pf_kernels.h \
        tests/prefill/pf_int8_test.cc tests/CMakeLists.txt
git commit -m "kernels: pf_requant_rot, rotated per-channel int8 weights, layouts 0 and 1 (spec 5 T1)"
```

---

### Task 3: `pf_colmax_rot`, the load-time column scales

**Files:**
- Modify: `src/kernels/prefill/pf_int8.cl` (inside the `PF_REQUANT_ROT` section, after `pf_requant_rot`)
- Test: `tests/prefill/pf_int8_test.cc`

**Interfaces:**
- Consumes: `pf_rot_load` (Task 2).
- Produces: kernel `pf_colmax_rot(const uint* qw, const half* sc, const uint* sbits, uint* colmax, uint n0, uint Nfull, uint K)`, WG 256, grid (width/16, K/1024). It writes `atomic_max` of `as_uint(max |v / 32|)` into `colmax[n]`, which the caller zero-fills first. ws = max / 127 (1.0 when max is 0), inv = 1 / ws, both on the host.

- [x] **Step 1: Write the failing test**

Append, called from `main` for `(5120, 1024, 0)`, `(5120, 1024, 1)`, `(17408, 1024, 0)`. The weights get one all-zero column (n = 5) and one column forced negative-heavy (n = 7), the Review Focus lines:

```cpp
void colmax_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t layout) {
  common::Int4Gptq w = common::Int4Gptq::random(K, N, 31 + K + N);
  for (uint32_t r = 0; r < K / 8; ++r) {
    w.qweight[size_t(r) * N + 5] = 0x88888888u;   // q - 8 == 0 everywhere: an all-zero column
    w.qweight[size_t(r) * N + 7] = 0x00000000u;   // q - 8 == -8 everywhere: negative-heavy
  }
  const std::vector<float> sgn = runtime::prefill::int8_signs(K);
  const std::vector<uint32_t> words = layout == 0 ? w.qweight : w.tiled();
  l0::Mem dw = pf_harness::upload(d.ctx, d.imm, words);
  l0::Mem dsc = pf_harness::upload(d.ctx, d.imm, w.scales);
  l0::Mem dbits = pf_harness::upload(d.ctx, d.imm, runtime::prefill::int8_sign_bits(K));
  l0::Mem dmax = pf_harness::upload(d.ctx, d.imm, std::vector<uint32_t>(N, 0u));
  l0::Module mod(d.ctx, kernels::path(kernels::pf_requant_rot_variant(layout)));
  l0::Kernel k = mod.kernel("pf_colmax_rot");
  k.group_size(256);
  k.arg_ptr(0, dw.ptr());
  k.arg_ptr(1, layout == 0 ? dsc.ptr() : nullptr);
  k.arg_ptr(2, dbits.ptr());
  k.arg_ptr(3, dmax.ptr());
  k.arg(4, 0u);
  k.arg(5, N);
  k.arg(6, K);
  d.run(k, N / 16, K / 1024);
  std::vector<uint32_t> got(N);
  pf_harness::download(d.imm, got, dmax);
  size_t bad = 0;
  for (uint32_t n = 0; n < N; ++n) {
    const std::vector<float> col = rotated_column(w, n, sgn);
    float amax = 0.0f;
    for (float v : col) amax = std::fmax(amax, std::fabs(v));
    uint32_t want;
    std::memcpy(&want, &amax, 4);
    bad += (want != got[n]);
  }
  CHECK(got[5] == 0u);
  std::printf("pf_colmax_rot K=%u N=%u L%u: %zu mismatches (zero column max %u)\n", K, N, layout, bad,
              got[5]);
  CHECK(bad == 0);
}
```

- [x] **Step 2: Run it to see it fail**

Run: `tools/box.sh test pf_int8_test`
Expected: FAIL: `pf_colmax_rot` is not an entry point of `pf_requant_rot_L0`.

- [x] **Step 3: The kernel**

Inside `#ifdef PF_REQUANT_ROT`, after `pf_requant_rot`:

```c
// max |(W R_K)[k, n]| over one block, into colmax[n] as float bits (every value
// is >= 0, so unsigned order is float order). The caller zero-fills colmax and
// runs one launch over all blocks; ws and inv are derived on the host.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_colmax_rot(__global const uint* restrict qw, __global const half* restrict sc,
                            __global const uint* restrict sbits, __global uint* restrict colmax,
                            uint n0, uint Nfull, uint K) {
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  const uint blk = get_group_id(1);
  const uint nbase = n0 + get_group_id(0) * 16u;
  __local uint tin[16 * 129];
  __local float tsc[16 * 17];
  float v[64];
  pf_rot_load(qw, sc, sbits, nbase, blk, Nfull, K, tin, tsc, v);
  float amax = 0.0f;
#pragma unroll
  for (uint j = 0; j < 64u; ++j) amax = fmax(amax, fabs(v[j] * (1.0f / 32.0f)));
  amax = sub_group_reduce_max(amax);
  if (l == 0u) atomic_max((volatile __global uint*)(colmax + nbase + c), as_uint(amax));
}
```

- [x] **Step 4: Run the test to see it pass**

Run: `tools/box.sh test pf_int8_test`
Expected: three `pf_colmax_rot ... 0 mismatches (zero column max 0)` lines; PASS.

- [x] **Step 5: Commit**

```bash
git add src/kernels/prefill/pf_int8.cl tests/prefill/pf_int8_test.cc
git commit -m "kernels: pf_colmax_rot, load-time rotated column scales (spec 5 T2)"
```

---

### Task 4: `pf_gemm_i8`, plain and SiLU, and the cross test

**Files:**
- Modify: `src/kernels/prefill/pf_int8.cl` (append the GEMM section)
- Modify: `src/kernels/prefill/CMakeLists.txt`, `src/kernels/prefill/pf_kernels.h`
- Test: `tests/prefill/pf_int8_test.cc`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 1 to 3.
- Produces: kernel `pf_gemm_i8(const ushort* xq16, const float* xs, const uint* w8, const float* ws, float* C, uint M, uint K, uint N, uint ldxq, uint ldb, uint ldc)`, WG 512, grid (M/256, N/128). With `SILU_EPI=1` the same signature, where `C` carries bf16 x and `ldc` carries `ldx`; x column (n/32) 16 + n % 16 for gate columns, which is gate‖up's 16-column interleave. Variants: `kernels::pf_gemm_i8_variant(bool silu)` returning `"pf_gemm_i8"` or `"pf_gemm_i8_SILU"`.

- [x] **Step 1: Write the failing tests**

Append three cases:

1. `gemm_case(K, N, M)`: random int8 x and int8 w8 (VNNI-4), random positive xs and ws. CPU oracle: `float(int32 sum) * (ws[n] * xs[m])`, bit-exact on every output, for `(5120, 1024, 256)` and `(17408, 1024, 512)`.
2. `silu_case(K=5120, N=1024, M=256)`: the same GEMM with `SILU_EPI`. CPU mirror: `g = acc * (ws * xs)` for gate columns and `u` for up columns (16-column interleave), then the `PF_SILU_ROW` chain with `rne_bf16`, `silu_f32 = x / (1 + exp(-x))` (`std::exp` in float), bit-exact. The chain is `rne_bf16(bf16f(rne_bf16(silu(bf16f(rne_bf16(g))))) * bf16f(rne_bf16(u)))`.
3. `cross_case(K=5120, N=1024, M=256)`: the whole h8 path. bf16 x with rows 200..255 set to NaN bf16 (0x7FC0), then `pf_quant_had`, host colmax/ws/inv from `pf_colmax_rot`, `pf_requant_rot`, `pf_gemm_i8`. Requirements:
   - rows 0..199 relative L2 against `x W^T` (fp32 on the host from the dequantised int4) **≤ 3 %** (random data is harsher than real; §15 measured 0.57 % on real weights);
   - no NaN in rows 0..199;
   - rows 0..199 bit-identical to the same run with rows 200..255 zero instead of NaN. Padding cannot leak.

Write each with the helpers above (`sat_rte`, `fwht_blocks`, `rotated_column`) plus:

```cpp
uint16_t rne_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += ((u >> 16) & 1u) + 0x7FFFu;
  return uint16_t(u >> 16);
}
float bf16f(uint16_t h) { return common::bf16_to_f32(h); }
float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }
```

Add `kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU` to the test's `add_dependencies`.

- [x] **Step 2: Run them to see them fail**

Run: `tools/box.sh test pf_int8_test`
Expected: FAIL at build (`kernel_pf_gemm_i8` missing).

- [x] **Step 3: The kernel**

Append to `pf_int8.cl`. The mainloop is `pw8_gemm`'s (`tools/probe/probe_w8a8.cl`, `__kernel void pw8_gemm` through its `intel_work_group_barrier_wait`), copied without change except that the `sig` argument and its store are removed. Then the two epilogues:

```c
#ifdef PF_GEMM_I8
#ifndef SILU_EPI
#define SILU_EPI 0
#endif
#define CHUNK 64
#define WG_M 256
#define WG_N 128
#define SG_M 32
#define SG_N 32
#define NA 4
#define NB 2
#if SILU_EPI
// pf_gemm.cl's chain, verbatim (the numerics contract with pf_prep.cl).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
#endif

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gemm_i8(__global const ushort* restrict xq16, __global const float* restrict xs,
                         __global const uint* restrict w8, __global const float* restrict ws,
                         __global float* restrict C, uint M, uint K, uint N, uint ldxq, uint ldb,
                         uint ldc) {
  // ... pw8_gemm's body from `const uint m0 = ...` to the end of the k loop, unchanged ...

#if SILU_EPI
  // A sub-group's two atoms are nb and nb + 16 with nb a multiple of 32, so
  // atom 0 is a gate block and atom 1 its up block (gate||up's 16-column
  // interleave). The x column of gate column nb + lane is nb / 2 + lane.
  __global ushort* restrict X = (__global ushort*)C;
  const int x_w = (int)(N * 2u), x_h = (int)M, x_p = (int)(ldc * 2u);
  const float wg = as_float(intel_sub_group_block_read((__global const uint*)(ws + nb)));
  const float wu = as_float(intel_sub_group_block_read((__global const uint*)(ws + nb + 16u)));
#define PF_SILU_ROW_I8(r)                                                     \
  do {                                                                        \
    const float xsr = xs[mb + 8u * (uint)a + r];                              \
    const float g = (float)acc[a][0].s##r * (wg * xsr);                       \
    const float u = (float)acc[a][1].s##r * (wu * xsr);                       \
    const ushort g_b = rne_bf16(g), u_b = rne_bf16(u);                        \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));                        \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                                \
  } while (0)
#pragma unroll
  for (int a = 0; a < NA; ++a) {
    ushort xv[8];
    PF_SILU_ROW_I8(0); PF_SILU_ROW_I8(1); PF_SILU_ROW_I8(2); PF_SILU_ROW_I8(3);
    PF_SILU_ROW_I8(4); PF_SILU_ROW_I8(5); PF_SILU_ROW_I8(6); PF_SILU_ROW_I8(7);
    intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, x_w, x_h, x_p,
                                               (int2)((int)(nb >> 1), (int)(mb + 8u * (uint)a)), xv);
  }
#undef PF_SILU_ROW_I8
#else
  // pw8_gemm's epilogue, unchanged.
  // ... the `for (int b ...) { wsc ...; for (int a ...) { float8 o; ...; block_write } }` block ...
#endif
}
#endif  // PF_GEMM_I8
```

The two `// ...` lines stand for **verbatim copies** of the probe text named in them. The executor copies them from `tools/probe/probe_w8a8.cl` (the `pw8_gemm` kernel, lines 73 to 176 at commit `1f8eae6`) and removes only `sig` and its store. `mb`, `nb` and `acc` are defined in that copied body.

`CMakeLists.txt`:

```cmake
add_ocloc_kernel(pf_gemm_i8 SOURCE ${PF_INT8_CL} DEFINES PF_GEMM_I8=1
                 OPTIONS -cl-intel-256-GRF-per-thread)
add_ocloc_kernel(pf_gemm_i8_SILU SOURCE ${PF_INT8_CL} DEFINES PF_GEMM_I8=1 SILU_EPI=1
                 OPTIONS -cl-intel-256-GRF-per-thread)
```

`pf_kernels.h`:

```cpp
inline std::string pf_gemm_i8_variant(bool silu) { return silu ? "pf_gemm_i8_SILU" : "pf_gemm_i8"; }
```

- [x] **Step 4: Run the tests to see them pass**

Run: `tools/box.sh test pf_int8_test`
Expected: the gemm and silu cases `0 mismatches`, the cross case `rel L2 ... PASS, padding isolated`, and `pf_int8_test: PASS`. Also check the build log shows no `spilled` warning for `pf_gemm_i8` or `pf_gemm_i8_SILU`.

- [x] **Step 5: Commit**

```bash
git add src/kernels/prefill/pf_int8.cl src/kernels/prefill/CMakeLists.txt src/kernels/prefill/pf_kernels.h \
        tests/prefill/pf_int8_test.cc tests/CMakeLists.txt
git commit -m "kernels: pf_gemm_i8 with plain and SiLU epilogues, and the h8 cross test (spec 5 T1)"
```

---

### Task 5: `runtime::prefill::Int8State`, `linear_i8` and `linear_i8_silu`

**Files:**
- Create: `src/runtime/prefill/int8.h`, `src/runtime/prefill/int8.cc`
- Modify: `src/runtime/prefill/CMakeLists.txt` (add `int8.cc` to the prefill host library, beside `linear_l0.cc`)
- Test: `tests/prefill/linear_i8_test.cc`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: the Task 1 to 4 kernels, `int8_signs.h`, `loader::DeviceWeight`, `PrefillScratch::partials`, `Context::launch`, `KernelCache`, `pad256`.
- Produces (plan 5b relies on exactly these):

```cpp
namespace runtime::prefill {
// Everything the int8 path owns beyond PrefillScratch: per-K signs, per-linear
// column scales (built on first use of a weight, one GPU pass), int8 scratch.
class Int8State {
 public:
  explicit Int8State(l0::Context& ctx);
  // ws/inv for this weight, computed on first call (pf_colmax_rot + host
  // finish), cached by the weight's device address.
  const std::pair<l0::Mem, l0::Mem>& scales(Context& cx, KernelCache& kc,
                                            const loader::DeviceWeight& w);
  const l0::Mem& signs_f32(uint32_t K);    // [K] +-1
  const l0::Mem& sign_bits(uint32_t K);    // [K/32]
  l0::Mem& xq();                           // int8 [kC][17408]
  l0::Mem& xs();                           // fp32 [kC]
  l0::Mem& w8_slab();                      // u32 [17408/4][1024]
  size_t bytes() const;
};
// linear_l0's contract (fp32 partials [pad256(M)][N]) on the h8 path.
void linear_i8(Context& cx, KernelCache& kc, PrefillScratch& s, Int8State& q,
               const loader::DeviceWeight& w, const uint16_t* x, uint32_t M);
// linear_l0_silu's contract (bf16 out [pad256(M)][N/2] at pitch ldx, partials untouched).
void linear_i8_silu(Context& cx, KernelCache& kc, PrefillScratch& s, Int8State& q,
                    const loader::DeviceWeight& w, const uint16_t* x, uint32_t M,
                    uint16_t* out, uint32_t ldx);
size_t linear_i8_launches(const model::GemvShape& sh);   // 1 + 2 * N / 1024
}
```

- [ ] **Step 1: Write the failing test**

`tests/prefill/linear_i8_test.cc`: one random layout-0 weight (K 5120, N 2048, two slabs) and one random layout-1 weight (K 5120, N 2048) in `loader::DeviceWeight` form (upload `qweight` or `tiled()` into `mem`; layout 0 also gets `scales`), `M = 300` (not a multiple of 256). Run `linear_i8` and compare `partials` rows 0..299 against the host oracle of Task 4's cross case, at the same ≤ 3 % bar. For the layout-1 weight, also require **bit-identity with the layout-0 weight built from the same `Int4Gptq`**. Then run `linear_i8_silu` on a gate‖up-shaped weight (N 2048, 16-column interleave) and require bit-identity with a host mirror assembled from `linear_i8`'s fp32 output through the `PF_SILU_ROW` chain. Assert `linear_i8_launches({K 5120, N 2048}) == 5`.

- [ ] **Step 2: Run it to see it fail**

Run: `tools/box.sh test linear_i8_test`
Expected: FAIL at build (`runtime/prefill/int8.h` missing).

- [ ] **Step 3: Implement**

`int8.cc`:

- `scales()`: zero-fill a u32 [N] buffer, launch `pf_colmax_rot` over (N/16, K/1024) with `sign_bits(K)`, `cx.wait()`, download, and build `ws[n] = max > 0 ? max / 127 : 1`, `inv[n] = 1 / ws[n]`. Upload both and cache them in `std::map<const void*, std::pair<l0::Mem, l0::Mem>>` keyed by `w.mem.ptr()`. This runs once per weight, at the first int8 prefill: load-time in effect, and never inside a recorded chunk, because plan 5b calls `scales()` for every linear before the first chunk.
- `linear_i8()`: the same `require` checks as `linear_l0` (int4, N % 1024 == 0, K % 1024 == 0, K ≤ 17408, `pad256(M) <= kC`). Then:
  1. `pf_quant_had_variant(K)` over `pad256(M)` groups with (x, `signs_f32(K)`, `xq`, `xs`, `ldx = K`);
  2. per slab n0: `pf_requant_rot_variant(layout)` over (1024/16, K/1024) with (w.mem, layout 0 ? w.scales : nullptr, `sign_bits(K)`, inv, `w8_slab`, n0, N, K, 1024);
  3. `pf_gemm_i8_variant(false)` over (pad256(M)/256, 1024/128) with (`xq`, `xs`, `w8_slab`, `ws + n0`, `partials + n0`, pad256(M), K, 1024, K/2, 1024, N).
  
  Each launch is followed by `profile_wait` with new phases `kI8Quant`, `kI8Requant`, `kI8Gemm`, added to `profile.h`/`profile.cc` beside `kSlabDequant`. No host wait otherwise.
- `linear_i8_silu()`: the same, with `pf_gemm_i8_variant(true)`, `C = out + n0 / 2`, `ldc = ldx`, and the `require(ldx * 2 == N)` from `linear_l0_silu`.
- `bytes()`: the three scratch buffers plus every cached scale pair.

- [ ] **Step 4: Run the test to see it pass**

Run: `tools/box.sh test linear_i8_test`
Expected: PASS, printing the rel L2 per case and "layout 1 == layout 0 bitwise".

- [ ] **Step 5: Commit**

```bash
git add src/runtime/prefill/int8.h src/runtime/prefill/int8.cc src/runtime/prefill/CMakeLists.txt \
        src/runtime/prefill/profile.h src/runtime/prefill/profile.cc tests/prefill/linear_i8_test.cc \
        tests/CMakeLists.txt
git commit -m "runtime: Int8State, linear_i8 and linear_i8_silu, the h8 slab walk (spec 5 T1-T2)"
```

---

**Done when:** `tools/box.sh test "pf_int8_test|linear_i8_test"` passes, and the full suite (`tools/box.sh test`) is unchanged in its pass count. No production path calls `linear_i8` yet; that is plan 5b.

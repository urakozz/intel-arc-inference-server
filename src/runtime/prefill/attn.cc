#include "runtime/prefill/attn.h"

#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "runtime/prefill/gemm.h"

namespace runtime::prefill {
namespace {

using attn::kGroup;
using attn::kHeadDim;
using attn::kKvHeads;
using attn::kQHeads;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::attn_chunk: " + what);
}

// Row pitches, named so a GemmBatch field says which buffer it belongs to.
constexpr size_t kQRow = size_t(kQHeads) * kHeadDim;    // pf_q  [C][24][256] = 6144
constexpr size_t kKvRow = size_t(kKvHeads) * kHeadDim;  // cache [pos][4][256] = 1024

}  // namespace

void attn_prep_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, void* ctrl,
                     const float* qkv_partials, const float* fa_small, const float* rope,
                     uint16_t* kv_k, uint16_t* kv_v) {
  // Argument 5 is `attn_gate`, which the Q_BF16 build does not write (the gate
  // is read from `qkv_partials` by `pf_attn_gate`). Null rather than a spare
  // buffer, so a build that DID write it would fault at once instead of
  // silently filling scratch nobody reads.
  cx.launch(kc(kernels::pf_attn_prep_q16_variant(), "pf_attn_prep"), kQHeads + kKvHeads, C, 1,
            {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope),
             PtrArg(s.pf_q.ptr()), PtrArg(nullptr), PtrArg(kv_k), PtrArg(kv_v)});
}

void attn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                const uint16_t* q, const uint16_t* kv_k, const uint16_t* kv_v) {
  require(C > 0 && C <= PrefillScratch::kC, "C = " + std::to_string(C) + " is outside (0, kC]");
  const uint32_t depth = pos + C;
  require(depth <= s.max_len, "pos + C = " + std::to_string(depth) + " exceeds max_len " +
                                  std::to_string(s.max_len));
  require(s.max_len % 8 == 0, "max_len " + std::to_string(s.max_len) +
                                  " is not a multiple of 8; it is both GEMMs' row pitch");
  // sycl-tla's 2D block loads move 8 bf16 at a time, so `gemm_bf16_batched`
  // requires N (of QK^T) and K (of PV) to be multiples of 8. The rounded-up
  // width is what both GEMMs use; `pf_softmax_causal` writes an exact +0.0 in
  // columns [depth, npad) so the PV contraction over them contributes nothing,
  // and it never READS a score there, so whatever the QK^T GEMM computed out of
  // unwritten cache rows cannot reach a number. `max_len` is a multiple of 8 on
  // every path that gets here (it is the KV allocation's own extent), and the
  // rounding therefore never leaves the allocation.
  const uint32_t npad = (depth + 7u) & ~7u;
  require(npad <= s.max_len,
          "the 8-element GEMM padding of depth " + std::to_string(depth) + " exceeds max_len " +
              std::to_string(s.max_len));
  require(gemm_bf16_supports_transb(),
          "this build has no ColumnMajor-B chain, so QK^T cannot read the KV cache in place");

  const uint32_t ld = s.max_len;                       // pf_s / pf_p row pitch
  const size_t stride_l = size_t(C) * ld;              // one head slot of pf_s / pf_p
  const size_t stride_h = size_t(C) * kHeadDim;        // one q-head of pf_o
  require(s.pf_s.size() >= size_t(kGroup) * stride_l * sizeof(float), "pf_s is undersized");
  require(s.pf_p.size() >= size_t(kGroup) * stride_l * sizeof(uint16_t), "pf_p is undersized");
  require(s.pf_o.size() >= size_t(kQHeads) * stride_h * sizeof(float), "pf_o is undersized");

  float* S = s.pf_s.as<float>();
  uint16_t* P = s.pf_p.as<uint16_t>();
  float* O = s.pf_o.as<float>();

  for (uint32_t j = 0; j < kKvHeads; ++j) {
    // S[l][m][n] = sum_d q[m][6j+l][d] . k[n][j][d].  A is the six q-heads of
    // this group inside `pf_q`'s [C][24][256] row (pitch 6144, batch stride
    // 256); B is kv-head j's [depth][256] slab read TRANSPOSED in place at
    // pitch 1024, shared by all six (strideB = 0).
    GemmBatch qk{};
    qk.M = C;  qk.K = kHeadDim;  qk.N = npad;  qk.L = kGroup;
    qk.lda = kQRow;              // pf_q is [C][24][256]
    qk.ldb = kKvRow;             // the KV cache row is [4][256]
    qk.ldc = ld;
    qk.strideA = kHeadDim;       // the next q-head of the group
    qk.strideB = 0;              // GQA: one kv-head, six q-heads
    qk.strideC = stride_l;
    gemm_bf16_batched(cx, qk, q + size_t(j) * kGroup * kHeadDim,
                      kv_k + size_t(j) * kHeadDim, S, /*transB=*/true);
    cx.wait();   // SYCL -> L0: one compute queue, no device-side dependency (A24)

    cx.launch(kc(kernels::pf_attn_variant(), "pf_softmax_causal"), C, kGroup, 1,
              {PtrArg(S), PtrArg(P), arg_val(pos), arg_val(npad), arg_val(ld),
               arg_val(uint32_t(stride_l))});
    cx.wait();   // L0 -> SYCL

    // O[l][m][d] = sum_n P[l][m][n] . v[n][j][d].  B is kv-head j's v slab,
    // [depth][256] row-major at pitch 1024 -- no transpose, and again shared.
    GemmBatch pv{};
    pv.M = C;  pv.K = npad;  pv.N = kHeadDim;  pv.L = kGroup;
    pv.lda = ld;
    pv.ldb = kKvRow;
    pv.ldc = kHeadDim;
    pv.strideA = stride_l;
    pv.strideB = 0;
    pv.strideC = stride_h;
    gemm_bf16_batched(cx, pv, P, kv_v + size_t(j) * kHeadDim,
                      O + size_t(j) * kGroup * stride_h, /*transB=*/false);
  }
  cx.wait();     // SYCL -> L0, for the gate launch the caller makes next
}

void attn_gate_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C,
                     const float* qkv_partials, uint16_t* out) {
  const uint32_t stride_h = uint32_t(size_t(C) * kHeadDim);
  cx.launch(kc(kernels::pf_attn_variant(), "pf_attn_gate"), kQHeads, C, 1,
            {PtrArg(s.pf_o.ptr()), PtrArg(qkv_partials), PtrArg(out), arg_val(stride_h)});
}

}  // namespace runtime::prefill

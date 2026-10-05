#include "runtime/prefill/attn.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/profile.h"

namespace runtime::prefill {
namespace {

using attn::kHeadDim;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::attn_chunk: " + what);
}


std::atomic<int> g_mode{-1};   // -1: not yet read from the environment

}  // namespace

AttnMode attn_mode() {
  int m = g_mode.load(std::memory_order_relaxed);
  if (m < 0) {
    const char* v = std::getenv("B70_PREFILL_ATTN");
    m = int(v != nullptr && std::strcmp(v, "composed") == 0 ? AttnMode::Composed
                                                             : AttnMode::Flash);
    int expected = -1;
    if (!g_mode.compare_exchange_strong(expected, m)) m = expected;
  }
  return AttnMode(m);
}

void set_attn_mode_for_test(AttnMode m) { g_mode.store(int(m)); }

const char* attn_mode_name(AttnMode m) { return m == AttnMode::Flash ? "flash" : "composed"; }

void attn_prep_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, void* ctrl,
                     const float* qkv_partials, const float* fa_small, const float* rope,
                     uint16_t* kv_k, uint16_t* kv_v) {
  // Spec 15b: grid (q-heads + kv-heads, C) - 28 work-groups per row on Qwen3.8.
  const uint32_t q_heads = s.desc().fa_q_heads, kv_heads = s.desc().fa_kv_heads;
  // Argument 5 is `attn_gate`, which the Q_BF16 build does not write (the gate
  // is read from `qkv_partials` by `pf_attn_gate`). Null rather than a spare
  // buffer, so a build that DID write it would fault at once instead of
  // silently filling scratch nobody reads.
  cx.launch(kc(kernels::pf_attn_prep_q16_variant(q_heads, kv_heads), "pf_attn_prep"), q_heads + kv_heads, C, 1,
            {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope),
             PtrArg(s.pf_q.ptr()), PtrArg(nullptr), PtrArg(kv_k), PtrArg(kv_v)});
}

void attn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                const uint16_t* q, const uint16_t* kv_k, const uint16_t* kv_v,
                PrefillBackend backend) {
  require(C > 0 && C <= PrefillScratch::kC, "C = " + std::to_string(C) + " is outside (0, kC]");
  // Spec 15b: the head counts are the descriptor's (Qwen3.8: 24 q-heads, 4 kv-heads,
  // groups of 6). Row pitches, named so a GemmBatch field says which buffer it
  // belongs to: pf_q [C][q-heads][256] (6144) and the cache [pos][kv-heads][256] (1024).
  const uint32_t q_heads = s.desc().fa_q_heads, kv_heads = s.desc().fa_kv_heads;
  const uint32_t group = s.s_heads();
  const size_t q_row = size_t(q_heads) * kHeadDim, kv_row = size_t(kv_heads) * kHeadDim;
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
  const bool l0 = is_l0(backend);
  if (l0 && attn_mode() == AttnMode::Flash) {
    // Spec 6: one launch for all four kv groups, no score scratch. pf_o's layout is the
    // composed path's ([24][rows][256], stride rows * 256, rows = pad256(C)), so
    // pf_attn_gate is unchanged. The kernel reads the cache rows [0, pos + C) only and masks
    // past `depth` itself, so neither max_len's padding nor stale cache rows can reach it.
    // Spec 6c: RPW 8, grid (ceil(C / 8), 4, 1).
    // The WG size (96) comes from the kernel's reqd_work_group_size (Context::launch).
    const uint32_t rows = attn_rows(C, backend);
    require(s.pf_o.size() >= size_t(q_heads) * rows * kHeadDim * sizeof(float),
            "pf_o is undersized");
    cx.launch(kc(kernels::pf_flash_attn_variant(s.desc().fa_q_heads, s.desc().fa_kv_heads), "pf_flash_attn"), (C + 7u) / 8u, kv_heads,
              1,
              {PtrArg(q), PtrArg(kv_k), PtrArg(kv_v), PtrArg(s.pf_o.ptr()), arg_val(pos),
               arg_val(C), arg_val(rows)});
    profile_wait(cx, Phase::kAttnFlash);
    return;
  }
  if (!l0)
    require(gemm_bf16_supports_transb(),
            "this build has no ColumnMajor-B chain, so QK^T cannot read the KV cache in place");
  // Spec 2.1 §3.4 (L0): the depth pads to 256 -- pf_gemm's tile -- and every per-head slot is
  // strided by the 256-padded row count. sycl-tla keeps spec 2's 8-padding and C-row slots.
  const uint32_t rows = attn_rows(C, backend);
  const uint32_t npad = l0 ? pad256(depth) : (depth + 7u) & ~7u;
  require(npad <= s.max_len, "the padding of depth " + std::to_string(depth) + " exceeds max_len " +
                                 std::to_string(s.max_len));
  if (l0) require(s.max_len % 256 == 0, "the L0 backend needs max_len % 256 == 0 (spec 2.1 §3.1)");
  const uint32_t ld = s.max_len;                       // pf_s / pf_p row pitch
  const size_t stride_l = size_t(rows) * ld;           // one head slot of pf_s / pf_p
  const size_t stride_h = size_t(rows) * kHeadDim;     // one q-head of pf_o
  // Spec 6: the score scratch is lazy -- allocated here, on the first composed call.
  l0::Mem& pf_s = s.pf_s_buffer();
  l0::Mem& pf_p = s.pf_p_buffer();
  require(pf_s.size() >= size_t(group) * stride_l * sizeof(float), "pf_s is undersized");
  require(pf_p.size() >= size_t(group) * stride_l * sizeof(uint16_t), "pf_p is undersized");
  require(s.pf_o.size() >= size_t(q_heads) * stride_h * sizeof(float), "pf_o is undersized");
  float* S = pf_s.as<float>();
  uint16_t* P = pf_p.as<uint16_t>();
  float* O = s.pf_o.as<float>();

  // Spec S3 (`2026-09-22-prefill-parity-program-design.md` §3), L0 only: row `m`
  // is the query at absolute position `pos + m`, so the causal bound puts every
  // score it can use in columns [0, pos + m]. One GEMM of N = `npad` therefore
  // computes, on the second chunk, roughly half its columns for the mask to
  // discard. The rows are issued in blocks of `kPfGemmTile` instead, block b
  // covering [r0, r1) with `N_b = pad256(pos + r1)` -- the causal limit of its
  // LAST row, rounded up to the tile pf_gemm already works in.
  //
  // **QK^T only, and P·V deliberately whole -- that is a MEASUREMENT, not an
  // oversight.** The stage as designed blocked both. Profiled at 4096 ids in two
  // 2048 chunks (device 1, B70_PREFILL_PROFILE=1, one run each, diagnostic):
  // QK^T 59.0 -> 44.3 ms of L0 GPU time, but P·V 31.9 -> 66.3 ms, a net loss of
  // 19.7 ms on the two rows. The asymmetry is occupancy. P·V's N axis is ONE
  // 256-wide tile (`kHeadDim`), so its whole grid is (M/256, 1, 6) = 48
  // workgroups; blocking M to 256 leaves (1, 1, 6) = 6 workgroups per launch and
  // the card idles through eight serialized launches to save 29% of the MACs.
  // QK^T's N axis is `npad`/256 = 8..16 tiles, so a blocked launch still carries
  // 48-96 workgroups and keeps the saving. Blocking P·V is therefore recorded as
  // rejected on measurement; re-opening it needs a kernel that splits P·V's N or
  // its K, which is not this stage (runtime only).
  //
  // **Bitwise identity, and where it comes from.** Each QK^T output column is an
  // independent dot product, so dropping columns changes no surviving one, and
  // the dropped ones are never READ: `pf_softmax_causal`'s pass 1 (max) and pass
  // 2 (sum) both scan `j < nvalid = pos + m + 1` and nothing past it, and its
  // pass 3 stores an exact +0.0 in [nvalid, npad) without reading `s` there.
  // Every row of block b has `nvalid <= pos + r1 <= N_b`, so the columns the
  // block no longer computes are exactly the columns no pass touches -- they now
  // hold a stale `pf_s` value instead of a score computed from unwritten cache
  // rows, and neither is reachable. `p` is still written over all of [0, npad),
  // so P·V below contracts the identical operands it always did.
  const uint32_t blocks = attn_row_blocks(C, backend);
  for (uint32_t j = 0; j < kv_heads; ++j) {
    // S[l][m][n] = q_l[m] . k[n], the `group` q-heads of kv group j against its K cache rows,
    // read in place (transB). N is the padded causal depth; the softmax reads only [0, pos+m].
    const uint16_t* qj = q + size_t(j) * group * kHeadDim;
    const uint16_t* kj = kv_k + size_t(j) * kHeadDim;
    if (l0) {
      for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t r0 = b * kPfGemmTile, r1 = std::min(r0 + kPfGemmTile, C);
        GemmBatch qk{};
        qk.M = r1 - r0;  qk.K = kHeadDim;  qk.N = std::min(npad, pad256(pos + r1));
        qk.L = group;
        qk.lda = q_row;  qk.ldb = kv_row;  qk.ldc = ld;
        qk.strideA = kHeadDim;  qk.strideB = 0;  qk.strideC = stride_l;
        gemm_l0(cx, kc, qk, qj + size_t(r0) * q_row, kj, S + size_t(r0) * ld, /*transB=*/true);
      }
      // Diagnostic only; the timed path has no wait here. One wait for the whole
      // block loop, so the phase keeps its meaning and gains no perturbation:
      // `take_launch_metrics` drains every block's GPU time into this row.
      profile_wait(cx, Phase::kAttnQk);
    } else {
      GemmBatch qk{};
      qk.M = C;  qk.K = kHeadDim;  qk.N = npad;  qk.L = group;
      qk.lda = q_row;  qk.ldb = kv_row;  qk.ldc = ld;
      qk.strideA = kHeadDim;  qk.strideB = 0;  qk.strideC = stride_l;
      attn_qk_sycl(cx, qk, qj, kj, S);    // includes the SYCL -> L0 wait (A24)
    }

    // Unchanged, and deliberately so: the softmax sees the whole chunk at once,
    // reads only the causal prefix of each row, and zero-fills to `npad`.
    cx.launch(kc(kernels::pf_attn_variant(q_heads, kv_heads), "pf_softmax_causal"), C, group, 1,
              {PtrArg(S), PtrArg(P), arg_val(pos), arg_val(npad), arg_val(ld),
               arg_val(uint32_t(stride_l))});
    if (l0) profile_wait(cx, Phase::kAttnSm);

    // O[l][m][d] = sum_n P[l][m][n] . v[n][j][d]; K is the padded depth, whose extra P columns
    // the softmax wrote as exact +0.0 (spec §3.4's invariant on the KV rows they multiply).
    // ONE launch on both backends -- see the occupancy measurement above.
    GemmBatch pv{};
    pv.M = C;  pv.K = npad;  pv.N = kHeadDim;  pv.L = group;
    pv.lda = ld;  pv.ldb = kv_row;  pv.ldc = kHeadDim;
    pv.strideA = stride_l;  pv.strideB = 0;  pv.strideC = stride_h;
    const uint16_t* vj = kv_v + size_t(j) * kHeadDim;
    float* oj = O + size_t(j) * group * stride_h;
    if (l0) {
      gemm_l0(cx, kc, pv, P, vj, oj, /*transB=*/false);
      profile_wait(cx, Phase::kAttnPv);
    } else {
      timed_wait(cx, Phase::kAttnSm);     // L0 -> SYCL
      attn_pv_sycl(cx, pv, P, vj, oj);
      // Otherwise the next group's QK wait also consumes this group's PV.
      // The final group is drained by the mandatory boundary below.
      if (j + 1 < kv_heads) profile_wait(cx, Phase::kAttnPv);
    }
  }
  if (!l0) timed_wait(cx, Phase::kAttnPv); // SYCL -> L0, for the gate launch next
}

void attn_gate_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, uint32_t rows,
                     const float* qkv_partials, uint16_t* out) {
  const uint32_t stride_h = uint32_t(size_t(rows) * kHeadDim);   // pf_o's head slot (§3.4)
  const uint32_t q_heads = s.desc().fa_q_heads;                    // grid (q-heads, C)
  cx.launch(kc(kernels::pf_attn_variant(q_heads, s.desc().fa_kv_heads), "pf_attn_gate"), q_heads, C, 1,
            {PtrArg(s.pf_o.ptr()), PtrArg(qkv_partials), PtrArg(out), arg_val(stride_h)});
}

}  // namespace runtime::prefill

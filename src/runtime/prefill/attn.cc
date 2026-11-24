#include "runtime/prefill/attn.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/profile.h"

namespace runtime::prefill {
namespace {

using attn::kHeadDim;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::attn_chunk: " + what);
}


std::atomic<int> g_mode{int(AttnMode::Flash)};   // the flash attention unless a test sets Composed
std::atomic<int> g_split{int(FlashSplit::Auto)};  // 3l (2): the rule unless a test sets Off / On

constexpr uint32_t kTile = 64;   // pf_flash_attn's KT: spans are whole tiles

// 3l (2): the split-K rule, from flash_split_test --time ("3l (2)"; 16 FA layers, interleaved, card 1). Today's grid is ceil(C / 8) x kv-heads work-groups, each
// walking every key; the card keeps ~5 of these 96-lane 256-GRF work-groups per Xe core resident (32 cores).
// The key range [0, pos) is cut into n = max(64 / row blocks, 16) spans - ~256 work-groups for one or two
// row blocks, 16 spans for more - never a span under 256 keys (below it the merge and the per-span setup
// outweigh the walk), and no more spans than pf_o's spare rows hold (13 at C = 128). At 128k keys more spans
// (~2048 keys each) measured up to 1.17x faster still at C = 64 (84.5 vs 98.5 ms for 16 layers): a later
// refinement, not taken (the whole-chunk timings and the judged runs are of this rule). It is taken for
// C <= kSplitMaxRows at pos >= kSplitMinPos (at 512 keys a 64-row split was slower: 1.19 -> 1.42 ms for 16
// layers); larger chunks keep today's launch, bitwise as before (C = 256 deep in the cache would gain too -
// 0.88 x at 8192, 0.73 x at 131072 - but pf_o's spare rows hold only 6 of its spans: not built).
constexpr uint32_t kSplitMaxRows = 128;
constexpr uint32_t kSplitMinPos = 1024;
constexpr uint32_t kSplitMinSpan = 256;
// Ornith's 16 / 2 heads (PERF-12; flash_split_test --ornith --time, 10 FA layers, card 0, against the pair form with
// the gate fused that its walk ran before). Half Qwen3.8's kv-heads, so twice the spans for the same work-groups:
// n = max(128 / row blocks, 32), within pf_o's room and cut to whole waves of kOrnithWave work-groups (the card holds
// 128 of these 128-lane 256-GRF work-groups at once - 32 Xe cores x 4): a partial last wave costs a whole one, 64
// rows at 128k keys 33.8 ms at 24 spans vs 38.8 at the room's 27. 16 rows at 128k keys 130.7 -> 10.5 ms (the shared
// rule's 32 spans: 11.8), 64 rows 130.7 -> 33.8; a span under 512 keys loses from 64 rows up (64 rows at 8192 keys:
// 3.47 ms at 256, 2.62 at 512). Chunks to 256 rows split too (6 spans at 256 rows): 157.6 -> 121.6 ms at 128k, 315.4
// -> 242.2 at 262k, 9.58 -> 8.32 at 8192 keys - but not below 8192 keys (4096: 4.67 -> 4.73).
constexpr uint32_t kOrnithSplitMaxRows = 256;
constexpr uint32_t kOrnithWideMinPos = 8192;     // chunks over kSplitMaxRows
constexpr uint32_t kOrnithMinSpan = 512;         // chunks over 32 rows
constexpr uint32_t kOrnithWave = 128;
// PERF-12 follow-up (flash_split_test --time, 16 FA layers, card 0, bf16): Qwen3.8's 24 / 4 heads at C > 32 take
// pf_o's room cut to whole waves of kOrnithWave work-groups too - 64 rows at 128k keys 98.6 -> 82.8 ms (16 -> 24
// spans), 128 rows 180.0 -> 166.5 (13 -> 12) and at 64k 92.5 -> 87.4; shallower than kQwenWaveMinPos the old rule
// stays (8192 keys: 7.1 ms at 16 or 10 spans alike).
constexpr uint32_t kQwenWaveMinPos = 16384;
// int8 KV: up to this many rows the split reads the int8 rows in place (no kv8_widen of pos + C rows - the
// widening alone is ~1 ms a layer at 32k); above it, kv8_widen + the raw split (its tiles convert nothing).
constexpr uint32_t kSplitInPlaceMaxRows = 64;

// 3l (2): where a split chunk's spans go - pf_o past its `rows` rows: the unnormalised O
// [n][q-heads][c8][256], then m and l [n][q-heads][c8] each.
struct SplitArea {
  uint32_t n;
  float* part;
  float* ml;
};
SplitArea split_area(PrefillScratch& s, uint32_t pos, uint32_t C, uint32_t span) {
  const uint32_t q_heads = s.desc().fa_q_heads, c8 = (C + 7u) & ~7u;
  SplitArea a{flash_split_count(pos, span), nullptr, nullptr};
  a.part = s.pf_o.as<float>() + size_t(q_heads) * attn_rows(C, PrefillBackend::L0) * kHeadDim;
  a.ml = a.part + size_t(a.n) * q_heads * c8 * kHeadDim;
  require(size_t(a.ml - s.pf_o.as<float>()) + size_t(2) * a.n * q_heads * c8 <= s.pf_o.size() / sizeof(float),
          "the split's partials do not fit pf_o's spare rows");
  return a;
}
// The merge: the spans folded into pf_o's [q-heads][rows][256], grid (C, q-heads).
void launch_merge(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, uint32_t rows, const SplitArea& a) {
  const uint32_t q_heads = s.desc().fa_q_heads;
  cx.launch(kc(kernels::pf_flash_merge_variant(q_heads, s.desc().fa_kv_heads), "pf_flash_merge"), C, q_heads, 1,
            {PtrArg(a.part), PtrArg(a.ml), PtrArg(s.pf_o.ptr()), arg_val(C), arg_val(rows), arg_val(a.n)});
}

// PERF-114: kv8.cl KV8_I8S's q scales, fp32 [C][q-heads], in pf_q past its int8 rows at the chunk cap (pf_q is
// bf16 [kC][q-heads][256], so its second half is free once q is int8).
float* kv8_q_scales(PrefillScratch& s) {
  return reinterpret_cast<float*>(s.pf_q.as<int8_t>() + size_t(s.chunk_cap()) * s.desc().fa_q_heads * kHeadDim);
}

}  // namespace

bool kv8_int8_scores(const model::ModelDesc& d, PrefillBackend b, uint32_t pos, uint32_t C) {
  const bool built = (d.fa_q_heads == 24 && d.fa_kv_heads == 4) || (d.fa_q_heads == 12 && d.fa_kv_heads == 2);
  return b == PrefillBackend::L0Int8 && built && !flash_split_span(d, pos, C);
}

bool kv8_own_rows(const model::ModelDesc& d, PrefillBackend b, uint32_t pos, uint32_t C) {
  return b == PrefillBackend::L0Int8 && d.fa_q_heads == 12 && d.fa_kv_heads == 2 && flash_split_span(d, pos, C) != 0;
}

bool kv8_split_in_place(const model::ModelDesc& d, uint32_t pos, uint32_t C) {
  return C <= kSplitInPlaceMaxRows && flash_split_span(d, pos, C) != 0;
}

FlashSplit flash_split_mode() { return FlashSplit(g_split.load(std::memory_order_relaxed)); }

void set_flash_split_for_test(FlashSplit m) { g_split.store(int(m)); }

uint32_t flash_split_span(const model::ModelDesc& d, uint32_t pos, uint32_t C) {
  const FlashSplit mode = flash_split_mode();
  const bool built = (d.fa_q_heads == 24 && d.fa_kv_heads == 4) || (d.fa_q_heads == 12 && d.fa_kv_heads == 2) ||
                     (d.fa_q_heads == 16 && d.fa_kv_heads == 2);
  const bool ornith = d.fa_q_heads == 16 && d.fa_kv_heads == 2;
  if (mode == FlashSplit::Off || !built || pos < kTile || C > (ornith ? kOrnithSplitMaxRows : kSplitMaxRows)) return 0;
  if (mode == FlashSplit::Auto && (pos < kSplitMinPos || (C > kSplitMaxRows && pos < kOrnithWideMinPos))) return 0;
  // The spans' partials live in pf_o past its `rows` rows: n x c8 x (256 + 2) floats a head, within
  // the smallest chunk cap's (PrefillScratchDims::kCDefault) spare rows.
  const uint32_t c8 = (C + 7u) & ~7u, rows = attn_rows(C, PrefillBackend::L0);
  const uint32_t room = uint32_t(uint64_t(PrefillScratchDims::kCDefault - rows) * kHeadDim / (uint64_t(c8) * (kHeadDim + 2)));
  uint32_t n = ornith ? std::max(128u / (c8 / 8u), 32u) : std::max(64u / (c8 / 8u), 16u);
  // Qwen3.8 one card, C > 32 deep in the cache: Ornith's whole-wave cut over pf_o's room (kQwenWaveMinPos).
  const bool qwen_waves = d.fa_q_heads == 24 && d.fa_kv_heads == 4 && C > 32 && pos >= kQwenWaveMinPos;
  if (qwen_waves) n = room;
  if (ornith || qwen_waves) {   // whole waves: n x (row blocks x kv-heads) work-groups
    n = std::min(n, room);
    const uint32_t q = kOrnithWave / std::gcd(kOrnithWave, c8 / 8u * d.fa_kv_heads);
    if (n >= q) n -= n % q;
  }
  const uint32_t min_span = mode == FlashSplit::On ? kTile : ornith && C > 32 ? kOrnithMinSpan : kSplitMinSpan;
  uint32_t span = std::max(min_span, ((pos + n - 1) / n + kTile - 1) / kTile * kTile);
  while (flash_split_count(pos, span) > room) span += kTile;
  return flash_split_count(pos, span) > 1 || mode == FlashSplit::On ? span : 0;
}

AttnMode attn_mode() { return AttnMode(g_mode.load(std::memory_order_relaxed)); }

void set_attn_mode_for_test(AttnMode m) { g_mode.store(int(m)); }

const char* attn_mode_name(AttnMode m) { return m == AttnMode::Flash ? "flash" : "composed"; }

void attn_prep_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, void* ctrl,
                     const float* qkv_partials, const float* fa_small, const float* rope,
                     uint16_t* kv_k, uint16_t* kv_v, const void* pages, bool kv_only) {
  // Spec 15b: grid (q-heads + kv-heads, C) - 28 work-groups per row on Qwen3.8; kv_only (PERF-85) kv-heads.
  const uint32_t q_heads = s.desc().fa_q_heads, kv_heads = s.desc().fa_kv_heads;
  const uint32_t gx = kv_only ? kv_heads : q_heads + kv_heads;
  auto name = [&](std::string v) { return kv_only ? kernels::kv_only(v) : v; };
  // Argument 5 is `attn_gate`, which the Q_BF16 build does not write (the gate
  // is read from `qkv_partials` by `pf_attn_gate`). Null rather than a spare
  // buffer, so a build that DID write it would fault at once instead of
  // silently filling scratch nobody reads.
  if (pages) {   // plan 19v3b Task 0: a slot's paged KV
    cx.launch(kc(name(kernels::kv_paged(kernels::pf_attn_prep_q16_variant(q_heads, kv_heads))), "pf_attn_prep"),
              gx, C, 1,
              {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope), PtrArg(s.pf_q.ptr()),
               PtrArg(nullptr), PtrArg(kv_k), PtrArg(kv_v), PtrArg(pages)});
    return;
  }
  cx.launch(kc(name(kernels::pf_attn_prep_q16_variant(q_heads, kv_heads)), "pf_attn_prep"), gx, C, 1,
            {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope),
             PtrArg(s.pf_q.ptr()), PtrArg(nullptr), PtrArg(kv_k), PtrArg(kv_v)});
}

bool attn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                const uint16_t* q, const uint16_t* kv_k, const uint16_t* kv_v,
                PrefillBackend backend, const float* gate_src, uint16_t* gate_out, const void* pages) {
  require(C > 0 && C <= s.chunk_cap(), "C = " + std::to_string(C) + " is outside (0, " + std::to_string(s.chunk_cap()) + "], the scratch's chunk cap");
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
  // The depth pads to pf_gemm's 256 tile; the rounded-up width is what both GEMMs use.
  // `pf_softmax_causal` writes an exact +0.0 in columns [depth, npad) so the PV contraction
  // over them contributes nothing, and it never READS a score there, so whatever the QK^T
  // GEMM computed out of unwritten cache rows cannot reach a number.
  if (attn_mode() == AttnMode::Flash) {
    // Spec 6: one launch for all four kv groups, no score scratch. pf_o's layout is the
    // composed path's ([24][rows][256], stride rows * 256, rows = pad256(C)), so
    // pf_attn_gate is unchanged. The kernel reads the cache rows [0, pos + C) only and masks
    // past `depth` itself, so neither max_len's padding nor stale cache rows can reach it.
    // Spec 6c: RPW 8, grid (ceil(C / 8), 4, 1); phase 3 Task 7: RPW 32 for Ornith below.
    // The WG size comes from the kernel's reqd_work_group_size (Context::launch).
    const uint32_t rows = attn_rows(C, backend);
    require(s.pf_o.size() >= size_t(q_heads) * rows * kHeadDim * sizeof(float),
            "pf_o is undersized");
    if (const uint32_t span = flash_split_span(s.desc(), pos, C)) {   // 3l (2): a short chunk deep in the cache
      const SplitArea a = split_area(s, pos, C, span);
      if (pages)   // plan 19v3b Task 0: a slot's paged KV
        cx.launch(kc(kernels::kv_paged(kernels::pf_flash_attn_split_variant(q_heads, kv_heads)), "pf_flash_attn"),
                  (C + 7u) / 8u, kv_heads, a.n,
                  {PtrArg(q), PtrArg(kv_k), PtrArg(kv_v), PtrArg(a.part), arg_val(pos), arg_val(C), arg_val(rows),
                   PtrArg(a.ml), arg_val(span), PtrArg(pages)});
      else
        cx.launch(kc(kernels::pf_flash_attn_split_variant(q_heads, kv_heads), "pf_flash_attn"), (C + 7u) / 8u, kv_heads,
                  a.n, {PtrArg(q), PtrArg(kv_k), PtrArg(kv_v), PtrArg(a.part), arg_val(pos), arg_val(C), arg_val(rows),
                        PtrArg(a.ml), arg_val(span)});
      // Ornith's walk: the merge gates as it folds (bitwise pf_attn_gate on the merged pf_o).
      if (gate_out && q_heads == 16) {
        require(gate_src != nullptr, "the fused gate needs qkv_partials");
        cx.launch(kc(kernels::pf_flash_mergeg_variant(q_heads, kv_heads), "pf_flash_merge"), C, q_heads, 1,
                  {PtrArg(a.part), PtrArg(a.ml), PtrArg(s.pf_o.ptr()), arg_val(C), arg_val(rows), arg_val(a.n),
                   PtrArg(gate_src), PtrArg(gate_out)});
        profile_wait(cx, Phase::kAttnFlash);
        return true;
      }
      launch_merge(cx, kc, s, C, rows, a);
      profile_wait(cx, Phase::kAttnFlash);
      return false;
    }
    // The pair form (two sub-groups share 16 rows of a head, kPfFlashPairHeads heads a work-group) on Qwen3.8's 24 / 4
    // and Ornith's 16 / 2 heads - bitwise the RPW 8 binary, which a paged KV and the TP card's 12 / 2 keep.
    const bool pair = !pages && ((q_heads == 24 && kv_heads == 4) || (q_heads == 16 && kv_heads == 2));
    const size_t qkv_n = size_t(2 * q_heads + 2 * kv_heads) * kHeadDim;   // a gate source row (pf_flash_attn's QKV_N)
    if (pages) {   // plan 19v3b Task 0: a slot's paged KV
      require((q_heads == 24 && kv_heads == 4) || (q_heads == 16 && kv_heads == 2),
              "a paged KV (sequence slots) is Qwen3.8's or Ornith's: no other paged flash is built");
      // Plan 31 Task 12: Ornith's slots gate in the 8-row form's epilogue (bitwise the one-slot walk's pairg).
      const bool gated = gate_out && q_heads == 16;
      require(!gated || gate_src != nullptr, "the fused gate needs qkv_partials");
      for (uint32_t r = 0; r < C; r += kPairLaunchRows) {   // launched as the pair form is (attn_chunk_launches)
        const uint32_t c = std::min(kPairLaunchRows, C - r);
        if (gated)
          cx.launch(kc(kernels::kv_paged(kernels::pf_flash_attn_g_variant(q_heads, kv_heads)), "pf_flash_attn"),
                    (c + 7u) / 8u, kv_heads, 1,
                    {PtrArg(q + r * q_row), PtrArg(kv_k), PtrArg(kv_v), PtrArg(s.pf_o.as<float>() + size_t(r) * kHeadDim),
                     arg_val(pos + r), arg_val(c), arg_val(rows), PtrArg(gate_src + r * qkv_n),
                     PtrArg(gate_out + r * q_row), PtrArg(pages)});
        else
          cx.launch(kc(kernels::kv_paged(kernels::pf_flash_attn_variant(q_heads, kv_heads)), "pf_flash_attn"),
                    (c + 7u) / 8u, kv_heads, 1,
                    {PtrArg(q + r * q_row), PtrArg(kv_k), PtrArg(kv_v), PtrArg(s.pf_o.as<float>() + size_t(r) * kHeadDim),
                     arg_val(pos + r), arg_val(c), arg_val(rows), PtrArg(pages)});
      }
      profile_wait(cx, Phase::kAttnFlash);
      return gated;
    }
    if (!pair) {
      cx.launch(kc(kernels::pf_flash_attn_variant(q_heads, kv_heads), "pf_flash_attn"), (C + 7u) / 8u, kv_heads, 1,
                {PtrArg(q), PtrArg(kv_k), PtrArg(kv_v), PtrArg(s.pf_o.ptr()), arg_val(pos), arg_val(C),
                 arg_val(rows)});
      profile_wait(cx, Phase::kAttnFlash);
      return false;
    }
    const uint32_t gz = q_heads / kv_heads / kernels::kPfFlashPairHeads;
    // Phase 3: Ornith's walk passes the gate's source and target; its pair launch then gates in the epilogue (bitwise
    // pf_attn_gate).
    const bool gated = gate_out && q_heads == 16;
    require(!gated || gate_src != nullptr, "the fused gate needs qkv_partials");
    for (uint32_t r = 0; r < C; r += kPairLaunchRows) {
      const uint32_t c = std::min(kPairLaunchRows, C - r);
      const uint32_t gx = (c + kernels::kPfFlashPairRows - 1u) / kernels::kPfFlashPairRows;
      float* const o = s.pf_o.as<float>() + size_t(r) * kHeadDim;
      if (gated)
        cx.launch(kc(kernels::pf_flash_attn_pairg_variant(q_heads, kv_heads), "pf_flash_attn"), gx, kv_heads, gz,
                  {PtrArg(q + r * q_row), PtrArg(kv_k), PtrArg(kv_v), PtrArg(o), arg_val(pos + r), arg_val(c),
                   arg_val(rows), PtrArg(gate_src + r * qkv_n), PtrArg(gate_out + r * q_row)});
      else
        cx.launch(kc(kernels::pf_flash_attn_pair_variant(q_heads, kv_heads), "pf_flash_attn"), gx, kv_heads, gz,
                  {PtrArg(q + r * q_row), PtrArg(kv_k), PtrArg(kv_v), PtrArg(o), arg_val(pos + r), arg_val(c),
                   arg_val(rows)});
    }
    profile_wait(cx, Phase::kAttnFlash);
    return gated;
  }
  require(!pages, "a paged KV (sequence slots) prefills through the flash attention only");
  // Spec 2.1 §3.4: the depth pads to 256 -- pf_gemm's tile -- and every per-head slot is
  // strided by the 256-padded row count.
  const uint32_t rows = attn_rows(C, backend);
  const uint32_t npad = pad256(depth);
  require(npad <= s.max_len, "the padding of depth " + std::to_string(depth) + " exceeds max_len " +
                                 std::to_string(s.max_len));
  require(s.max_len % 256 == 0, "the L0 backend needs max_len % 256 == 0");
  const uint32_t ld = s.max_len;                       // pf_s / pf_p row pitch
  const size_t stride_l = size_t(rows) * ld;           // one head slot of pf_s / pf_p
  // pf_softmax_causal takes the head-slot stride as a uint32 argument (below): a chunk of
  // `rows` rows at this max_len must keep it under 2^32 elements, or the softmax would
  // index a wrapped stride. The composed path is diagnostic only; refuse, don't wrap.
  require(stride_l <= UINT32_MAX, "the pf_s / pf_p head stride rows x max_len = " + std::to_string(rows) +
                                      " x " + std::to_string(ld) + " = " + std::to_string(stride_l) +
                                      " exceeds the softmax's uint32 argument; use a smaller chunk or max_len");
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

  // Spec S3 (`2026-09-22-prefill-parity-program-design.md` §3): row `m`
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

    // Unchanged, and deliberately so: the softmax sees the whole chunk at once,
    // reads only the causal prefix of each row, and zero-fills to `npad`.
    cx.launch(kc(kernels::pf_attn_variant(q_heads, kv_heads), "pf_softmax_causal"), C, group, 1,
              {PtrArg(S), PtrArg(P), arg_val(pos), arg_val(npad), arg_val(ld),
               arg_val(uint32_t(stride_l))});
    profile_wait(cx, Phase::kAttnSm);

    // O[l][m][d] = sum_n P[l][m][n] . v[n][j][d]; K is the padded depth, whose extra P columns
    // the softmax wrote as exact +0.0 (spec §3.4's invariant on the KV rows they multiply).
    // ONE launch -- see the occupancy measurement above.
    GemmBatch pv{};
    pv.M = C;  pv.K = npad;  pv.N = kHeadDim;  pv.L = group;
    pv.lda = ld;  pv.ldb = kv_row;  pv.ldc = kHeadDim;
    pv.strideA = stride_l;  pv.strideB = 0;  pv.strideC = stride_h;
    const uint16_t* vj = kv_v + size_t(j) * kHeadDim;
    float* oj = O + size_t(j) * group * stride_h;
    gemm_l0(cx, kc, pv, P, vj, oj, /*transB=*/false);
    profile_wait(cx, Phase::kAttnPv);
  }
  return false;
}

void attn_gate_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, uint32_t rows,
                     const float* qkv_partials, uint16_t* out) {
  const uint32_t stride_h = uint32_t(size_t(rows) * kHeadDim);   // pf_o's head slot (§3.4)
  const uint32_t q_heads = s.desc().fa_q_heads;                    // grid (q-heads, C)
  cx.launch(kc(kernels::pf_attn_variant(q_heads, s.desc().fa_kv_heads), "pf_attn_gate"), q_heads, C, 1,
            {PtrArg(s.pf_o.ptr()), PtrArg(qkv_partials), PtrArg(out), arg_val(stride_h)});
}

// --- spec 12b: the int8 KV cache -------------------------------------------------------

void require_kv8_path(PrefillBackend) {
  if (attn_mode() != AttnMode::Flash)
    throw std::runtime_error(
        "runtime::prefill: --kv-cache int8 prefills through the flash attention only: the composed "
        "attention (a test's set_attn_mode_for_test) is the bf16-KV reference");
}

void attn_prep_chunk_kv8(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, void* ctrl,
                         const float* qkv_partials, const float* fa_small, const float* rope,
                         const KvLayer& kv, const void* pages, bool int8_q, bool own) {
  require(kv.int8(), "attn_prep_chunk_kv8 needs the int8 cache's scales");
  const uint32_t q_heads = s.desc().fa_q_heads, kv_heads = s.desc().fa_kv_heads;
  if (int8_q && pages) {   // PERF-114: the same launch with q as int8 and its scales (KV8_I8S)
    cx.launch(kc(kernels::kv_paged(kernels::pf_attn_prep_kv8_i8s_variant(q_heads, kv_heads)), "attn_prep_kv8"),
              q_heads + kv_heads, C, 1, {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope), PtrArg(s.pf_q.ptr()),
                  PtrArg(nullptr), PtrArg(kv.k), PtrArg(kv.v), PtrArg(kv.ks), PtrArg(kv.vs), PtrArg(kv8_q_scales(s)),
                  PtrArg(pages)});
    return;
  }
  if (int8_q) {
    cx.launch(kc(kernels::pf_attn_prep_kv8_i8s_variant(q_heads, kv_heads), "attn_prep_kv8"), q_heads + kv_heads, C, 1,
              {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope), PtrArg(s.pf_q.ptr()),
               PtrArg(nullptr), PtrArg(kv.k), PtrArg(kv.v), PtrArg(kv.ks), PtrArg(kv.vs), PtrArg(kv8_q_scales(s))});
    return;
  }
  // kv8.cl's attn_prep_kv8 at PF = 1: grid (q-heads + kv-heads, C), no gate (null, as for
  // pf_attn_prep_q16), bf16 rotated q into pf_q, the int8 rows and scales.
  if (own) {   // PERF-107: also the chunk's rows as bf16 into the widened scratch (kv8.cl KV8_OWN; a tp card, unpaged)
    auto* kr = s.kv8_raw_buffer().as<uint16_t>();
    auto* vr = kr + size_t(s.kv8_len) * kv_heads * kHeadDim;
    cx.launch(kc(kernels::pf_attn_prep_kv8_variant(q_heads, kv_heads, true), "attn_prep_kv8"), q_heads + kv_heads, C, 1,
              {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope), PtrArg(s.pf_q.ptr()),
               PtrArg(nullptr), PtrArg(kv.k), PtrArg(kv.v), PtrArg(kv.ks), PtrArg(kv.vs), PtrArg(kr), PtrArg(vr)});
    return;
  }
  if (pages) {   // plan 19v3b Task 0: a slot's paged KV
    cx.launch(kc(kernels::kv_paged(kernels::pf_attn_prep_kv8_variant(q_heads, kv_heads)), "attn_prep_kv8"),
              q_heads + kv_heads, C, 1,
              {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope), PtrArg(s.pf_q.ptr()),
               PtrArg(nullptr), PtrArg(kv.k), PtrArg(kv.v), PtrArg(kv.ks), PtrArg(kv.vs), PtrArg(pages)});
    return;
  }
  cx.launch(kc(kernels::pf_attn_prep_kv8_variant(q_heads, kv_heads), "attn_prep_kv8"),
            q_heads + kv_heads, C, 1,
            {PtrArg(ctrl), PtrArg(qkv_partials), PtrArg(fa_small), PtrArg(rope),
             PtrArg(s.pf_q.ptr()), PtrArg(nullptr), PtrArg(kv.k), PtrArg(kv.v), PtrArg(kv.ks),
             PtrArg(kv.vs)});
}

void attn_chunk_kv8(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                    const uint16_t* q, const KvLayer& kv, PrefillBackend backend, const void* pages) {
  require(C > 0 && C <= s.chunk_cap(), "C = " + std::to_string(C) + " is outside (0, " + std::to_string(s.chunk_cap()) + "], the scratch's chunk cap");
  require(kv.int8(), "attn_chunk_kv8 needs the int8 cache's scales");
  require_kv8_path(backend);
  const uint32_t q_heads = s.desc().fa_q_heads, kv_heads = s.desc().fa_kv_heads;
  require(pos + C <= s.max_len, "pos + C = " + std::to_string(pos + C) + " exceeds max_len " +
                                    std::to_string(s.max_len));
  // pf_flash_attn's launch shapes: grid (ceil(C / 8), kv-heads, 1), or the pair form's on Qwen3.8's heads; pf_o's
  // [q-heads][rows][256] at rows = pad256(C). Widen raw integers once for all query tiles;
  // flash still applies K's scale to scores and V's scale to probabilities, bit for bit.
  const uint32_t rows = attn_rows(C, backend);
  require(s.pf_o.size() >= size_t(q_heads) * rows * kHeadDim * sizeof(float), "pf_o is undersized");
  // 3j: pos is slot-relative (the layer's K / V are at the slot's kv_base), so the widened rows fit one sequence's
  // positions - kv8_len, the longest slot - and V starts there.
  require(pos + C <= s.kv8_len, "pos + C = " + std::to_string(pos + C) + " exceeds the widening scratch's " +
                                    std::to_string(s.kv8_len) + " positions");
  auto& raw = s.kv8_raw_buffer();
  auto* k = raw.as<uint16_t>();
  auto* v = k + size_t(s.kv8_len) * kv_heads * kHeadDim;
  const uint32_t count = (pos + C) * kv_heads * kHeadDim;
  const uint32_t span = flash_split_span(s.desc(), pos, C);
  // Plan 19v3b Task 0: a slot's paged KV binds every launch below in its paged twin (the merge and the widened rows
  // are the sequence's own, unpaged).
  const auto pg = [&](const std::string& v) { return pages ? kernels::kv_paged(v) : v; };
  if (kv8_int8_scores(s.desc(), backend, pos, C)) {   // PERF-114: kv8_widen keeping K int8, then the int8-score pair, as below
    const auto i8s = pg(kernels::pf_flash_attn_kv8_raw_pair_i8s_variant(q_heads, kv_heads));
    if (pages)
      cx.launch(kc(i8s, "kv8_widen"), (count + 4095u) / 4096u, 1, 1,
                {PtrArg(kv.k), PtrArg(kv.v), PtrArg(k), PtrArg(v), arg_val(count), PtrArg(pages)});
    else
      cx.launch(kc(i8s, "kv8_widen"), (count + 4095u) / 4096u, 1, 1,
                {PtrArg(kv.k), PtrArg(kv.v), PtrArg(k), PtrArg(v), arg_val(count)});
    const auto* q8 = reinterpret_cast<const int8_t*>(q);
    const float* qs = kv8_q_scales(s);
    for (uint32_t r = 0; r < C; r += kPairLaunchRows) {
      const uint32_t c = std::min(kPairLaunchRows, C - r), pr = pos + r;
      const uint32_t gx = (c + kernels::kPfFlashPairRows - 1u) / kernels::kPfFlashPairRows,
                     gz = q_heads / kv_heads / kernels::kPfFlashPairHeads;
      if (pages)
        cx.launch(kc(i8s, "pf_flash_attn_kv8"), gx, kv_heads, gz,
                  {PtrArg(q8 + size_t(r) * q_heads * kHeadDim), PtrArg(qs + size_t(r) * q_heads), PtrArg(k),
                   PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs), PtrArg(s.pf_o.as<float>() + size_t(r) * kHeadDim),
                   arg_val(pr), arg_val(c), arg_val(rows), PtrArg(pages)});
      else
        cx.launch(kc(i8s, "pf_flash_attn_kv8"), gx, kv_heads, gz,
                  {PtrArg(q8 + size_t(r) * q_heads * kHeadDim), PtrArg(qs + size_t(r) * q_heads), PtrArg(k),
                   PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs), PtrArg(s.pf_o.as<float>() + size_t(r) * kHeadDim),
                   arg_val(pr), arg_val(c), arg_val(rows)});
    }
    profile_wait(cx, Phase::kAttnFlash);
    return;
  }
  // PERF-107: the int8 pass's split chunk (kv8_own_rows) reads its own rows [pos, pos + C) as the bf16 its writer put
  // into the scratch (attn_prep_chunk_kv8's `own`), so only the rows before them are widened; every other chunk reads
  // every row from int8. A paged chunk (a pooled card) keeps the int8 reads: the own twins are unpaged.
  const bool own = !pages && kv8_own_rows(s.desc(), backend, pos, C);
  if (span && kv8_split_in_place(s.desc(), pos, C)) {   // 3l (2): a short chunk deep in the cache, read in place
    const SplitArea a = split_area(s, pos, C, span);
    l0::Kernel& kk = kc(pg(kernels::pf_flash_attn_kv8_split_variant(q_heads, kv_heads, own)), "pf_flash_attn_kv8");
    if (own)   // the tiles reaching pos take the own keys from the scratch
      cx.launch(kk, (C + 7u) / 8u, kv_heads, a.n,
                {PtrArg(q), PtrArg(kv.k), PtrArg(kv.ks), PtrArg(kv.v), PtrArg(kv.vs), PtrArg(a.part), arg_val(pos),
                 arg_val(C), arg_val(rows), PtrArg(a.ml), arg_val(span), arg_val(pos), PtrArg(k), PtrArg(v)});
    else if (pages)
      cx.launch(kk, (C + 7u) / 8u, kv_heads, a.n,
                {PtrArg(q), PtrArg(kv.k), PtrArg(kv.ks), PtrArg(kv.v), PtrArg(kv.vs), PtrArg(a.part), arg_val(pos),
                 arg_val(C), arg_val(rows), PtrArg(a.ml), arg_val(span), PtrArg(pages)});
    else
      cx.launch(kk, (C + 7u) / 8u, kv_heads, a.n,
                {PtrArg(q), PtrArg(kv.k), PtrArg(kv.ks), PtrArg(kv.v), PtrArg(kv.vs), PtrArg(a.part), arg_val(pos),
                 arg_val(C), arg_val(rows), PtrArg(a.ml), arg_val(span)});
    launch_merge(cx, kc, s, C, rows, a);
    profile_wait(cx, Phase::kAttnFlash);
    return;
  }
  const auto variant = pg(kernels::pf_flash_attn_kv8_raw_variant(q_heads, kv_heads));
  const uint32_t widened = own ? pos * kv_heads * kHeadDim : count;
  if (pages)
    cx.launch(kc(variant, "kv8_widen"), (widened + 4095u) / 4096u, 1, 1,
              {PtrArg(kv.k), PtrArg(kv.v), PtrArg(k), PtrArg(v), arg_val(widened), PtrArg(pages)});
  else
    cx.launch(kc(variant, "kv8_widen"), (widened + 4095u) / 4096u, 1, 1,
              {PtrArg(kv.k), PtrArg(kv.v), PtrArg(k), PtrArg(v), arg_val(widened)});
  if (span) {   // 3l (2): a short chunk deep in the cache, over the widened rows
    const SplitArea a = split_area(s, pos, C, span);
    l0::Kernel& kk = kc(pg(kernels::pf_flash_attn_kv8_raw_split_variant(q_heads, kv_heads, own)), "pf_flash_attn_kv8");
    if (own)
      cx.launch(kk, (C + 7u) / 8u, kv_heads, a.n,
                {PtrArg(q), PtrArg(k), PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs), PtrArg(a.part), arg_val(pos),
                 arg_val(C), arg_val(rows), PtrArg(a.ml), arg_val(span), arg_val(pos)});
    else if (pages)
      cx.launch(kk, (C + 7u) / 8u, kv_heads, a.n,
                {PtrArg(q), PtrArg(k), PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs), PtrArg(a.part), arg_val(pos),
                 arg_val(C), arg_val(rows), PtrArg(a.ml), arg_val(span), PtrArg(pages)});
    else
      cx.launch(kk, (C + 7u) / 8u, kv_heads, a.n,
                {PtrArg(q), PtrArg(k), PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs), PtrArg(a.part), arg_val(pos),
                 arg_val(C), arg_val(rows), PtrArg(a.ml), arg_val(span)});
    launch_merge(cx, kc, s, C, rows, a);
    profile_wait(cx, Phase::kAttnFlash);
    return;
  }
  // PERF-64: Ornith's 16 / 2 heads take the pair over a slot's paged KV too (its scales by page; bitwise the paged raw
  // binary); Qwen3.8's slots keep the 8-row raw form.
  const bool ornith = q_heads == 16 && kv_heads == 2;
  if (pages && !ornith)
    for (uint32_t r = 0; r < C; r += kPairLaunchRows) {   // launched as the pair form is (attn_chunk_launches)
      const uint32_t c = std::min(kPairLaunchRows, C - r);
      cx.launch(kc(variant, "pf_flash_attn_kv8"), (c + 7u) / 8u, kv_heads, 1,
                {PtrArg(q + size_t(r) * q_heads * kHeadDim), PtrArg(k), PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs),
                 PtrArg(s.pf_o.as<float>() + size_t(r) * kHeadDim), arg_val(pos + r), arg_val(c), arg_val(rows),
                 PtrArg(pages)});
    }
  else if ((q_heads == 24 && kv_heads == 4) || ornith)   // the pair form (bitwise the raw binary), as attn_chunk's
    for (uint32_t r = 0; r < C; r += kPairLaunchRows) {   // whole chunks and launched as it: kPairLaunchRows at a time
      const uint32_t c = std::min(kPairLaunchRows, C - r);
      l0::Kernel& kk = kc(pg(kernels::pf_flash_attn_kv8_raw_pair_variant(q_heads, kv_heads)), "pf_flash_attn_kv8");
      const uint32_t gx = (c + kernels::kPfFlashPairRows - 1u) / kernels::kPfFlashPairRows,
                     gz = q_heads / kv_heads / kernels::kPfFlashPairHeads;
      if (pages)
        cx.launch(kk, gx, kv_heads, gz,
                  {PtrArg(q + size_t(r) * q_heads * kHeadDim), PtrArg(k), PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs),
                   PtrArg(s.pf_o.as<float>() + size_t(r) * kHeadDim), arg_val(pos + r), arg_val(c), arg_val(rows),
                   PtrArg(pages)});
      else
        cx.launch(kk, gx, kv_heads, gz,
                  {PtrArg(q + size_t(r) * q_heads * kHeadDim), PtrArg(k), PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs),
                   PtrArg(s.pf_o.as<float>() + size_t(r) * kHeadDim), arg_val(pos + r), arg_val(c), arg_val(rows)});
    }
  else
    cx.launch(kc(variant, "pf_flash_attn_kv8"),
              (C + 7u) / 8u, kv_heads, 1,
              {PtrArg(q), PtrArg(k), PtrArg(kv.ks), PtrArg(v), PtrArg(kv.vs),
               PtrArg(s.pf_o.ptr()), arg_val(pos), arg_val(C), arg_val(rows)});
  profile_wait(cx, Phase::kAttnFlash);
}

void attn_gate_chunk_kv8(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C,
                         uint32_t rows, const float* qkv_partials, uint16_t* out) {
  const uint32_t stride_h = uint32_t(size_t(rows) * kHeadDim);
  const uint32_t q_heads = s.desc().fa_q_heads;
  cx.launch(kc(kernels::pf_attn_gate_kv8_variant(q_heads, s.desc().fa_kv_heads), "pf_attn_gate_kv8"),
            q_heads, C, 1,
            {PtrArg(s.pf_o.ptr()), PtrArg(qkv_partials), PtrArg(out), arg_val(stride_h)});
}

}  // namespace runtime::prefill

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/k2_horizon.h"
#include "runtime/buffer_sizes.h"   // spec 18e: KvCache, KvLayout, default_kv_cache
#include "runtime/memory_plan.h"

// Spec 18b: K2-Horizon's runtime allocations as device-free arithmetic, and the memory
// planner over them (`--max-len auto`, spec 6 §10) - runtime/buffer_sizes.h's and
// runtime/memory_plan.h's arrangement for K2: the K2Buffers constructor allocates exactly
// what these return, and the plan adds the same numbers up before anything is allocated.
namespace runtime::k2 {

// The decode list's rows in flight. K2 builds M = 1 only (no MTP head, spec 18 §2; batching
// is spec 13's), so the scratch is sized for one row and every K2 binary is `_M1`.
inline constexpr uint32_t kM = 1;

// Spec 18 §10: which decode attention every K2 capture binds - k2_attn.cl's flash form (fp32
// scores and probabilities through an online softmax, one rounding at the end) or
// k2_attn_eager.cl's eager form (the reference's own bf16 chain: scores and probabilities
// rounded to bf16, the softmax over the full row in torch's order). `B70_K2_ATTN=flash|eager`,
// read at each call (so a test can set it between engines, as B70_DECODE_ATTN is); unset or
// empty means the default; anything else throws. The choice lives here, beside the sizes,
// because eager needs a score row the buffers allocate (attn_scores_bytes) and two more
// launches a layer (decode_launches).
enum class K2Attn { Flash, Eager };
inline constexpr K2Attn kDefaultK2Attn = K2Attn::Flash;   // until the box A/B (spec 18 §10)
K2Attn k2_attn();
const char* k2_attn_name(K2Attn a);

// The persistent group: what survives a token boundary and Engine reset() zeroes.
//   control   runtime::Control, shared memory (the host writes cur_token, reads out_token)
//   kv_k/kv_v bf16 [layers][max_len][kv_heads][head_dim] each: 192 KiB per position for K and
//             V together on K2 (48 x 8 x 128 x 2 B x 2), 6.44 GB at 32k (spec 18 §1).
//             Spec 18e (`--kv-cache int8`, rotkv at head_dim 128): runtime::KvLayout at K2's
//             heads - int8 rows [layers][max_len][8][128], then every layer's fp16 scales
//             [layers][max_len][8] - 97.5 KiB per position (48 x 2 x (1024 + 16) B).
struct PersistentSizes {
  size_t control = 0, kv_k = 0, kv_v = 0;
  size_t kv() const { return kv_k + kv_v; }
  size_t total() const { return control + kv_k + kv_v; }
};
// Spec 18e: K2's cache in either form, every layer (runtime::KvLayout: at bf16 exactly the
// allocation above, a layer's rows at layer x max_len x kv_n x 2 B).
KvLayout kv_layout(const model::K2Desc& d, uint32_t max_len, KvCache kv);
// `kv`: the cache's form - default runtime::default_kv_cache() (B70_KV_CACHE, unset = bf16), what
// K2Buffers and K2Engine default to, so a gate test's B70_KV_CACHE=int8 twin plans what it builds.
PersistentSizes persistent_sizes(const model::K2Desc& d, uint32_t max_len,
                                 KvCache kv = default_kv_cache());
// One layer's K (or V) rows: max_len x kv_heads x head_dim bf16 (the bf16 form's; the int8
// form's layer is kv_layout(...).layer_rows()).
inline size_t kv_layer_bytes(const model::K2Desc& d, uint32_t max_len) {
  return size_t(max_len) * d.kv_n() * 2;
}

// --- spec 18d: KV-only snapshots (spec 7's prefix cache on K2) --------------------------
// K2 has no recurrent state (no GDN, no conv ring, no MTP head): what a session leaves behind
// between replays is its KV cache and `pos` (Control; cur_token is re-fed by the prefill that
// follows every restore, spec 7 §3.3 step 4). MoVA needs nothing of its own: the routed value
// mix IS the V row the cache holds (spec 18 §3; at int8 the mix rotated and quantised after
// the combine, spec 18 §13), and the routes / RoPE table are per-step scratch / constants. So
// a snapshot is K2Engine::kv_bytes(n) of KV and no state bytes (state_bytes() == 0).
//
// The host layout of positions [begin, end) is runtime::Engine::save_kv's without the MTP
// head: K's rows of every layer [layers][n][kv_heads][head_dim] (bf16 or int8), then at int8
// K's fp16 scales of every layer [layers][n][kv_heads]; then V the same. A run is one
// contiguous device range of one allocation (`tensor` 0 = kv_k, 1 = kv_v, `offset` bytes from
// its base), in host order; K2Engine::save_kv / load_kv walk this list, device-free here so
// tests/server/k2_serve_test.cc holds it to the layout on host buffers.
struct SnapshotRun {
  uint32_t tensor = 0;   // 0: kv_k, 1: kv_v
  size_t offset = 0, bytes = 0;
};
std::vector<SnapshotRun> kv_snapshot_runs(const KvLayout& lay, uint32_t begin, uint32_t end);
// Bytes of a snapshot of n positions: 2 x layers x n x KvLayout::pos_bytes() - 196,608 B a
// position at bf16, 99,840 B at int8 (48 layers x 8 heads x 128).
inline size_t kv_snapshot_bytes(const KvLayout& lay, uint32_t n_pos) {
  return 2 * size_t(lay.layers) * lay.pos_bytes() * n_pos;
}

// The decode scratch, sized for kM rows:
//   resid       bf16 [M][hidden]                        the residual stream
//   x           bf16 [M][max(hidden, dense_inter)]      norm outputs, SiLU x up
//   partials    fp32 [max S x N over the int4 linears][M]   every int4 GEMV's split-K output
//   norm_sumsq  fp32 [kNormG][M]                         prep_res_fold -> k2_norm_finish
//   attn_q, attn_gate  fp32 [M][q_n]
//   attn_part   fp32 [q_heads][TGT][M][130]              k2_attn_decode -> k2_attn_reduce
//   attn_out    bf16 [M][q_n]                            o_proj's input
//   router      fp32 [M][router_n]                       the MoE router GEMV's output
//   routes      u32  [layers][2][M][32]                  every layer's MoVA (0) and MoE (1)
//                                                        route rows - kept per layer so a step
//                                                        leaves the whole routing readable
//                                                        (the routing diagnostic, K2)
//   moe_h       bf16 [M][top_k + 1][moe_inter]           SiLU(gate) x up of every slot
//   logits      fp32 [M][vocab]
//   argmax_part fp32 [M][ceil(vocab / 1024)][2]
struct ScratchSizes {
  size_t resid = 0, x = 0, partials = 0, norm_sumsq = 0, attn_q = 0, attn_gate = 0,
         attn_part = 0, attn_out = 0, router = 0, routes = 0, moe_h = 0, logits = 0,
         argmax_part = 0;
  size_t total() const {
    return resid + x + partials + norm_sumsq + attn_q + attn_gate + attn_part + attn_out + router +
           routes + moe_h + logits + argmax_part;
  }
};
ScratchSizes scratch_sizes(const model::K2Desc& d);
// Eager's score row, fp32 [M][q_heads][max_len] (k2_attn_eager.cl attn_s: scores, then the
// probabilities in place); 0 under flash. 4 MiB at 32k on K2. Outside ScratchSizes because it
// scales with max_len; K2Buffers allocates it only under eager, plan() counts it in the
// decode state.
size_t attn_scores_bytes(const model::K2Desc& d, uint32_t max_len, K2Attn a = k2_attn());
// The largest S x N over the int4 linears: the partials row the capture binds every GEMV to.
size_t partials_floats(const model::K2Desc& d);

// The per-layer route rows inside `routes` (bytes).
inline constexpr uint32_t kRouteWords = 32;   // = kernels::k2::route::kWords (k2_capture.cc)
inline size_t mova_route_at(uint32_t layer) { return (size_t(layer) * 2 + 0) * kM * kRouteWords * 4; }
inline size_t moe_route_at(uint32_t layer) { return (size_t(layer) * 2 + 1) * kM * kRouteWords * 4; }

// The golden gate's per-layer tap: bf16 [layers][M][hidden].
inline size_t tap_bytes(const model::K2Desc& d) { return size_t(d.layers) * kM * d.hidden * 2; }

// The decode list's launch count (Review Focus 5; k2_capture.cc asserts its walk against it):
//   1 embed_gather
//   dense layer  12: fold + norm, q||k||gate||v GEMV, attn prep, decode + reduce, o_proj GEMV,
//                    fold + norm, gate||up GEMV, SiLU x up, down GEMV
//   sparse layer 15: fold + norm, q||k||gate||v_router GEMV, MoVA route, MoVA value experts,
//                    attn prep, decode + reduce, o_proj GEMV, fold + norm, router GEMV, MoE
//                    route, gate||up (8 + shared), down + combine + residual
//   5 at the boundary: the final fold + norm, lm_head, the two argmax stages
// K2: 1 + 3 x 12 + 45 x 15 + 5 = 717 - against spec 18 §2's ~1000 estimate and spec 4's
// 2067 (fixed slot launches). Under B70_K2_ATTN=eager every layer's decode + reduce is four
// launches (score, softmax, P·V, reduce): + 2 x 48 = 813.
size_t decode_launches(const model::K2Desc& d, K2Attn a = k2_attn());

// --- Spec 18c: the prefill chunk ----------------------------------------------------------
// One chunk of at most kPfC positions (= kernels::k2::kPfC, checked in k2_prefill.cc): every
// allocation K2PrefillScratch makes, as device-free arithmetic. None depends on max_len (the
// flash attention keeps no score scratch), so the prefill scratch is one constant per model.
inline constexpr uint32_t kPfC = 2048;
inline constexpr uint32_t kPfTm = 32;      // the grouped GEMM's tile rows (pf_moe_gemm TM)
inline constexpr uint32_t kPfSlab = 1024;  // a whole slab of an int4 linear
inline uint32_t pad256(uint32_t n) { return (n + 255) / 256 * 256; }

// The prefill linear's slab walk over N columns: slabs of kPfSlab and one tail of
// pad256(N - n0) (k2_pf_linear.cl zero-fills its padding). Its count, and the widest slab.
inline uint32_t pf_slabs(uint32_t N) { return (N + kPfSlab - 1) / kPfSlab; }
inline uint32_t pf_slab_width(uint32_t N, uint32_t n0) {
  return N - n0 >= kPfSlab ? kPfSlab : pad256(N - n0);
}
// The partials row pitch of a linear's output: pad256(N) (the tail slab's padded columns are
// written there, never into the next row). 10240 for the dense attention row, 9472 for MoVA's.
inline uint32_t pf_ld(uint32_t N) { return pad256(N); }

// The grouped GEMMs' padded tile count of a C-row chunk (k2_pf_moe.cl's header has the bound):
//   floor((C x top_k + experts x (TM - 1)) / TM) + (shared ? ceil(C / TM) : 0)
uint32_t pf_tiles(uint32_t experts, uint32_t top_k, bool shared, uint32_t C);
uint32_t pf_moe_tiles(const model::K2Desc& d, uint32_t C);    // 100 + shared, top-8
uint32_t pf_mova_tiles(const model::K2Desc& d, uint32_t C);   // 64, top-4, no shared expert

// The bf16 expert-weight batch (the grouped GEMMs' B operand, dequantised per chunk): one
// region holding every value block, every down block, or half the gate||up blocks (rounded
// up) - whichever is largest - so gate||up runs in two batches and down / value in one.
enum class PfGroup { Value, GateUp, Down };
size_t pf_block_bf16(const model::K2Desc& d, PfGroup g);
size_t pf_weight_batch_bytes(const model::K2Desc& d);
uint32_t pf_batch_blocks(const model::K2Desc& d, PfGroup g);
uint32_t pf_batches(const model::K2Desc& d, PfGroup g);

// K2PrefillScratch's allocations (kPfC rows each):
//   ids       u32  [kPfC]                       host memory: the chunk's ids
//   resid     bf16 [kPfC][hidden]               the residual stream
//   xn        bf16 [kPfC][hidden]               every grouped norm's output (the linears' A)
//   xi        bf16 [kPfC][dense_inter]          the dense SiLU x up (down's A)
//   partials  fp32 [kPfC][max pf_ld]            every slab GEMM's output (10240 at K2)
//   slab      bf16 [K][kPfSlab], the largest K  one dequantised slab
//   sumsq     fp32 [kNormG][kPfC]               pf_res_fold -> k2_norm_finish
//   attn_q, attn_gate  fp32 [kPfC][q_n]         k2_attn_prep's q and gate
//   attn_out  bf16 [kPfC][q_n]                  the gated attention, o_proj's A
//   logits    fp32 [kPfC][router_n]             the MoE router GEMV
//   routes    u32  [layers][2][kPfC][32]        every layer's MoVA (0) and MoE (1) route rows
//                                               of the LAST chunk (the routing diagnostic)
//   hdr       u32  [pf_hdr_words(max experts)]  the sort's header (one, reused)
//   tiles     u32  [max tmax(kPfC)][2]          the tile table
//   row_tok   u32  [max rows]                   the sorted rows' tokens
//   pair_row  u32  [kPfC][max top_k]            every (token, slot)'s sorted row
//   xg        bf16 [max rows][hidden]           the gathered A, then the MoE down's y
//   h         bf16 max([moe rows][moe_inter], [mova rows][kv_n])   gate||up's SiLU h, MoVA's v
//   w         bf16 pf_weight_batch_bytes        the expert weight batch
struct PrefillSizes {
  size_t ids = 0, resid = 0, xn = 0, xi = 0, partials = 0, slab = 0, sumsq = 0, attn_q = 0,
         attn_gate = 0, attn_out = 0, logits = 0, routes = 0, hdr = 0, tiles = 0, row_tok = 0,
         pair_row = 0, xg = 0, h = 0, w = 0;
  size_t total() const {
    return ids + resid + xn + xi + partials + slab + sumsq + attn_q + attn_gate + attn_out + logits +
           routes + hdr + tiles + row_tok + pair_row + xg + h + w;
  }
};
PrefillSizes prefill_sizes(const model::K2Desc& d);
uint32_t pf_ld_max(const model::K2Desc& d);
// A layer's route rows inside `routes` (bytes): which = 0 MoVA, 1 MoE.
inline size_t pf_route_at(uint32_t layer, uint32_t which) {
  return (size_t(layer) * 2 + which) * kPfC * kRouteWords * 4;
}

// The chunk's launch count (k2_prefill.cc asserts its walk against it), the same at every C:
//   1 embed
//   dense layer  2 (fold + norm) + 2 x slabs(q||k||gate||v) + attn prep + flash
//                + 2 x slabs(o_proj) + 2 + 2 x slabs(gate||up, SiLU fused) + 2 x slabs(down)
//   sparse layer 2 + 2 x slabs(q||k||gate||v_router) + MoVA [route, sort, gather,
//                2 x value batches, combine] + attn prep + flash + 2 x slabs(o_proj) + 2
//                + MoE [router GEMV, route, sort, gather, 2 x gate||up batches,
//                2 x down batches, combine]
// K2: 1 + 3 x 62 + 45 x 49 = 2392 (derived). The head adds kPrefillHeadLaunches once.
size_t prefill_chunk_launches(const model::K2Desc& d);
inline constexpr size_t kPrefillHeadLaunches = 5;   // fold + norm, lm_head, two argmax stages

// --- the planner ------------------------------------------------------------------------
// `model_bytes`: what load_k2 allocated except the RoPE table (K2LoadReport: bytes.total(),
// = loader::k2_weight_bytes). The components are runtime::MemoryComponents' (memory_line's
// format): model (+ the RoPE table at max_len), kv, decode state (control + scratch), and
// with `prefill` (spec 18c: b70-decode --prefill / --prefill-length) the prefill scratch - lazy on the
// engine (allocated by the first prefill), so a decode-only plan leaves it out. No int8
// prefill state (K2 prefills on the l0 backend only). `kv` (spec 18e): the cache's form - its
// KV term is planned in it, so int8 about doubles what auto can give the context (no other term
// moves: MoVA's staged row and the writer's scratch reuse attn_out, k2_capture.cc).
struct Plan : MemoryComponents {
  uint32_t max_len = 0;
  size_t rope = 0;
  KvCache kv_form = KvCache::Bf16;   // the cache's form (MemoryComponents::kv is its bytes)
};
Plan plan(const model::K2Desc& d, uint32_t max_len, size_t model_bytes, bool debug_tap = false,
          bool prefill = false, K2Attn a = k2_attn(), KvCache kv = default_kv_cache());
// The largest multiple of runtime::kMaxLenQuantum up to `cap` whose plan + reserve fits;
// 0 when not even min(kMinAutoMaxLen, cap) does. Throws std::invalid_argument below a quantum.
uint32_t max_len_that_fits(const model::K2Desc& d, size_t model_bytes, size_t device_bytes,
                           size_t reserve_bytes, uint32_t cap, bool prefill = false,
                           KvCache kv = default_kv_cache());
std::string describe(const Plan& p, size_t device_bytes, size_t reserve_bytes);

}  // namespace runtime::k2

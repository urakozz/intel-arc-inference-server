#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/lfm2moe_layout.h"
#include "model/lfm2moe.h"
#include "runtime/memory_plan.h"     // kMaxLenQuantum, kMinAutoMaxLen, kDefaultReserveGb
#include "runtime/pipeline_plan.h"   // pp_balance

// Spec 24b Task 5: LFM2.5-24B-A2B's memory plan as device-free arithmetic - `--max-len auto` (the
// largest multiple of 256 that fits AND is at most the trained context, 131072, Review Focus 4) and
// the two-card split by bytes (runtime::pp_balance, spec 16b's rule - not a second one). Host only
// (tests/runtime/lfm2moe_plan_test.cc). 24c adds the decode scratch and the launch counts, 24d the
// prefill scratch; both fields exist here already (0 until then) so this Plan's shape does not
// change under them.
namespace runtime::lfm2moe {

// --- spec 24c: the decode list's switches, scratch and launch counts ------------------------------------------------
inline constexpr uint32_t kM = 1;   // the decode list is M = 1 (no MTP head; 24d's chunk walk has its own scratch)

// Which decode attention every capture binds - lfm_attn.cl's flash (the default, held to fp64) or lfm_attn_eager.cl's
// eager form (the reference's bf16 chain): `B70_LFM_ATTN=flash|eager`, read at each call; unset or empty is the
// default; anything else throws (spec 18 §10.1's switch).
enum class LfmAttn { Flash, Eager };
LfmAttn parse_attn(const std::string& v);   // "" | "flash" | "eager"; else throws naming B70_LFM_ATTN
LfmAttn lfm_attn();                         // B70_LFM_ATTN
const char* lfm_attn_name(LfmAttn a);
// The fused arms (spec 24 §4.5, decision 6) - each bitwise the as-built list: `route` (the route recomputed in every
// gate||up work-group: -1 a MoE layer), `router` (the router rows inside ffn_norm's launch: -1 a MoE layer), `conv`
// (the conv as in_proj's epilogue on the conv-tile layout: -1 a conv layer). `B70_LFM_FUSE=none|all|<comma list of
// route,router,conv>`; unset or empty is none (decision 6's default until Task 8 records the box's A/B); anything else
// throws.
struct LfmFuse {
  bool route = false, router = false, conv = false;
  bool any() const { return route || router || conv; }
};
LfmFuse parse_fuse(const std::string& v);
LfmFuse lfm_fuse();                         // B70_LFM_FUSE
std::string lfm_fuse_name(LfmFuse f);       // "none" | "all" | "route,conv" ...

// One device's decode scratch, sized for kM rows (every device has the whole set: the lists are per device and scratch
// is per list), and never zeroed (no step reads scratch it has not written first):
//   h0, h1       bf16 [M][hidden]           the residual, ping-ponged between norm sites (Review Focus 3)
//   x            bf16 [M][hidden]           every norm's output
//   slices       fp32 [max S x N][M]        every int4 GEMV's split-K output (dense gate||up 23552 x S 1 widest)
//   conv_y       bf16 [M][hidden]           lfm_conv's y = out_proj's input
//   logits_r     fp32 [M][experts]          the router's logits
//   routes       u32  [layers][M][16]       every MoE layer's route row (kernels::lfm2moe::route; dense rows unused)
//   moe_h        bf16 [M][top_k][moe_inter] the 4 slots' SiLU(gate) x up
//   attn_q       fp32 [M][q_heads][64]
//   attn_part    fp32 [q_heads][TGT][M][66]
//   attn_out     bf16 [M][q_n]              o_proj's input
//   s_mlp        bf16 [M][dense_inter]      the dense layers' SiLU x up (down's input)
//   logits       fp32 [M][vocab]
//   argmax_part  fp32 [M][ceil(vocab / 1024)][2]
//   attn_scores  fp32 [M][q_heads][max_len] eager only (0 under flash) - the term that scales with max_len
struct ScratchSizes {
  size_t h0 = 0, h1 = 0, x = 0, slices = 0, conv_y = 0, logits_r = 0, routes = 0, moe_h = 0, attn_q = 0, attn_part = 0,
         attn_out = 0, s_mlp = 0, logits = 0, argmax_part = 0, attn_scores = 0;
  size_t total() const {
    return h0 + h1 + x + slices + conv_y + logits_r + routes + moe_h + attn_q + attn_part + attn_out + s_mlp + logits +
           argmax_part + attn_scores;
  }
};
ScratchSizes scratch_sizes(const model::Lfm2MoeDesc& d, uint32_t max_len, LfmAttn a);
inline constexpr uint32_t kRouteWords = 16;   // = kernels::lfm2moe::route::kWords
inline size_t route_at(uint32_t layer) { return size_t(layer) * kM * kRouteWords * 4; }
// The residual tap (debug): bf16 [layers][M][hidden] on every device (its layers' rows written).
inline size_t tap_bytes(const model::Lfm2MoeDesc& d) { return size_t(d.layers) * kM * d.hidden * 2; }
// One device's persistent group: control (shared), its attention layers' K and V [max_len][kv_heads][64] each, its
// conv layers' Bx rings [16][hidden]. An empty group still allocates one 64-byte line.
struct PersistentSizes {
  size_t control = 0, kv_k = 0, kv_v = 0, conv_ring = 0;
  size_t total() const { return control + kv_k + kv_v + conv_ring; }
};
PersistentSizes persistent_sizes(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev,
                                 uint32_t max_len);
// A layer's K (or V) rows in its device's group: [max_len][kv_heads][head_dim] bf16.
inline size_t kv_layer_bytes(const model::Lfm2MoeDesc& d, uint32_t max_len) {
  return size_t(max_len) * d.kv_n() * 2;
}
inline size_t ring_layer_bytes(const model::Lfm2MoeDesc& d) { return size_t(loader::lfm::kConvRing) * d.hidden * 2; }

// The decode list's launches on one device (the hand-off's copies and barriers excluded; peer's pp_send / pp_recv
// included), and their sum (spec 24 §4.5):
//   device 0     lfm_embed_gather
//   a layer      dense conv 8 (norm, in_proj, conv, out_proj, norm, gate||up, silu, down); conv + MoE 9 (norm,
//                in_proj, conv, out_proj, norm, router, route, gate||up, down); attention + MoE 11 (norm, q||k||v,
//                prep, attn, reduce, o, norm, router, route, gate||up, down); eager attention +2; the arms: route -1
//                and router -1 a MoE layer, conv -1 a conv layer
//   last device  the final norm (embedding_norm), the head GEMV, the two argmax stages
// 383 on one card (2 x 8 + 28 x 9 + 10 x 11 + 5), 403 eager, 277 with every arm; two cards +0 (copy) / +2 (peer).
size_t device_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, LfmAttn a, LfmFuse f,
                       PpHandoff h);
size_t decode_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, LfmAttn a, LfmFuse f, PpHandoff h);

// --- spec 24d: the prefill chunk ---------------------------------------------------------------------------------
// One chunk of at most kPfC positions (= kernels::lfm2moe::kPfC, checked in lfm2moe_prefill.cc): every allocation
// Lfm2MoePrefillScratch makes on a device, as device-free arithmetic. None depends on max_len (the flash attention keeps
// no score scratch), so the prefill scratch is one constant per device.
inline constexpr uint32_t kPfC = 2048;
inline constexpr uint32_t kPfTm = 32;                          // the grouped GEMM's tile rows (pf_moe_gemm TM)
inline constexpr uint32_t kPfSlab = 1024;                      // a whole slab of an int4 linear
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;     // one weight batch of bf16 expert blocks (Kolibri's)
// An int4 linear's slab walk: N / kPfSlab whole slabs (every LFM2 width is a multiple of 1024: pf_dequant_slab's
// rule); the partials row pitch is N.
inline uint32_t pf_slabs(uint32_t N) { return (N + kPfSlab - 1) / kPfSlab; }
// The grouped GEMMs' padded tile count of a C-row chunk (lfm_pf_moe.cl's header has the bound; no shared expert):
//   floor((C x top_k + experts x (TM - 1)) / TM)        318 at C = 2048
uint32_t pf_tiles(const model::Lfm2MoeDesc& d, uint32_t C);
// The weight batches: the 64 experts' bf16 [K][N] blocks in kPfBatchBytes regions - gate||up 12,582,912 B a block,
// 42 a batch, 2 batches; down 6,291,456 B, 85 a batch, 1 batch (derived; PROPOSED as Kolibri's 512 MiB).
size_t pf_block_gu_bytes(const model::Lfm2MoeDesc& d);
size_t pf_block_dn_bytes(const model::Lfm2MoeDesc& d);
uint32_t pf_batch_blocks_gu(const model::Lfm2MoeDesc& d);
uint32_t pf_batch_blocks_dn(const model::Lfm2MoeDesc& d);
uint32_t pf_batches_gu(const model::Lfm2MoeDesc& d);
uint32_t pf_batches_dn(const model::Lfm2MoeDesc& d);

// Lfm2MoePrefillScratch's allocations on one device (kPfC rows each; every device holds the whole set - the lists are
// per device):
//   ids        u32  [kPfC]                       host memory: the chunk's ids (read by device 0's gather)
//   h0, h1     bf16 [kPfC][hidden]               the residual, ping-ponged between fold sites (24c's rule)
//   x          bf16 [kPfC][hidden]               every norm's output
//   y          bf16 [kPfC][hidden]               the conv's y / the attention's output: out_proj's / o_proj's A
//   partials   fp32 [kPfC][widest N]             the slab GEMMs' output (in_proj 6144 widest; the dense gate||up's
//                                                SiLU epilogue writes s_mlp instead)
//   slab       bf16 [widest K][kPfSlab]          one slab (the dense down's K 11776 widest)
//   seed       bf16 [2][hidden]                  lfm_pf_conv_seed's: Bx of the two positions before the chunk
//   attn_q     fp32 [kPfC][q_n]                  lfm_attn_prep's q
//   logits     fp32 [kPfC][experts]              the router
//   routes     u32  [layers][kPfC][16]           every MoE layer's route rows of the LAST chunk (dense rows unused)
//   hdr        u32  [pf_hdr words]               the sort's header
//   tiles      u32  [tmax(kPfC)][2]              the tile table
//   row_tok    u32  [tmax x TM]                  the sorted rows' tokens
//   pair_row   u32  [kPfC][top_k]                every (token, slot)'s sorted row
//   xg         bf16 [tmax x TM][hidden]          the gathered A, then down's y
//   moe_h      bf16 [tmax x TM][moe_inter]       gate||up's SiLU h
//   s_mlp      bf16 [kPfC][dense_inter]          the dense layers' SiLU x up (down's A)
//   w          bf16 kPfBatchBytes                the expert weight batch
// An empty term is one 64-byte line.
struct PrefillSizes {
  size_t ids = 0, h0 = 0, h1 = 0, x = 0, y = 0, partials = 0, slab = 0, seed = 0, attn_q = 0, logits = 0, routes = 0,
         hdr = 0, tiles = 0, row_tok = 0, pair_row = 0, xg = 0, moe_h = 0, s_mlp = 0, w = 0;
  size_t total() const {
    return ids + h0 + h1 + x + y + partials + slab + seed + attn_q + logits + routes + hdr + tiles + row_tok +
           pair_row + xg + moe_h + s_mlp + w;
  }
};
PrefillSizes prefill_sizes(const model::Lfm2MoeDesc& d);
inline constexpr uint32_t kPfHdrWords = 80;   // = kernels::lfm2moe::pf_hdr::words()
inline size_t pf_route_at(uint32_t layer) { return size_t(layer) * kPfC * kRouteWords * 4; }
// The two-card prefill hand-off (decision 10): a second runtime::PipelineLink in `copy` mode over spec 16b's layout at
// chunk size - kPfC rows of h (8 MB) and, at a DENSE cut (24c's departure, spec 24 §14), the dense down's pending
// fp32 rows [kPfC][hidden] (16 MB) in the sums region; nothing else (the receiving norm recomputes Σ).
PpLandingLayout pf_landing_layout(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p);
size_t pf_link_bytes(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev);

// The chunk's launches on one device and their sum, the same at every C (the hand-off's copies, the event, the taps
// and the head excluded):
//   device 0         lfm_pf_embed_gather
//   a conv layer     20: operator norm, in_proj 6 slabs x (dequant + pf_gemm), lfm_pf_conv_seed, lfm_pf_conv,
//                    out_proj 2 x 2, ffn_norm
//   an attn layer    14: operator norm, q||k||v 3 x 2, lfm_attn_prep, lfm_pf_attn, o_proj 2 x 2, ffn_norm
//   a MoE FFN        + 11: lfm_router, lfm_route, sort, gather, gate||up 2 batches x (dequant + grouped GEMM), down
//                    1 x 2, the combine (+ the fold)
//   a dense FFN      + 50: gate||up 23 slabs x (dequant + pf_gemm_T0_SILU), down 2 x 2 (the next norm folds it)
// 1259 at 40 layers (1 + 2 x 70 + 28 x 31 + 10 x 25) on one card AND on two (the cut is 16b's: device 0 ends after
// layer s-1, whose h rows - and at a dense cut its pending down rows - cross; device 1 starts at layer s's norm).
size_t prefill_device_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev);
size_t prefill_chunk_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p);
// The tail after the last chunk, on the last device: the final norm over the last row (decode's M 1 binary), the
// head, the two argmax stages.
inline constexpr size_t kPrefillHeadLaunches = 4;

// One device's plan: weights (loader::lfm::device_weight_bytes), kv (its attention layers' K and V
// at max_len), conv_rings (its conv layers' Bx ring), rope (its own table, when it holds an
// attention layer), decode_scratch (24c: the Control block + scratch_sizes' total at the attention form - and,
// on two cards, the hand-off's buffers), prefill_scratch (24d, `prefill`: prefill_sizes + on two cards the prefill
// link), reserve (the device bytes the plan deliberately does not count - runtime::kDefaultReserveGb by default;
// max_len_auto's `reserve_gb` argument overrides it for that check).
struct Plan {
  size_t weights = 0, kv = 0, conv_rings = 0, rope = 0, decode_scratch = 0, prefill_scratch = 0, reserve = 0;
  uint32_t max_len = 0;
  size_t total() const { return weights + kv + conv_rings + rope + decode_scratch + prefill_scratch + reserve; }
};

// A device's attention / conv layer counts under placement `p` (what `plan` below scales kv and
// conv_rings by).
inline uint32_t device_attn_layers(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev) {
  uint32_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) n += d.is_attention(l) ? 1 : 0;
  return n;
}
inline uint32_t device_conv_layers(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev) {
  uint32_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) n += d.is_attention(l) ? 0 : 1;
  return n;
}

// device `dev`'s plan at `max_len` (reserve defaults to runtime::kDefaultReserveGb; max_len_auto
// overrides `.reserve` on its own copy to try a caller-given reserve_gb).
Plan plan(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, uint32_t max_len, bool int8_head,
          LfmAttn a = LfmAttn::Flash, bool prefill = false);

// Throws std::runtime_error naming every term of `p` (weights, kv, conv_rings, rope, decode_scratch,
// prefill_scratch, reserve, their sum) when p.total() exceeds device_bytes.
void require_fits(const Plan& p, size_t device_bytes);

// `--max-len auto`: the largest multiple of runtime::kMaxLenQuantum (256), at most
// d.trained_max_len (131072; Review Focus 4 - an explicit --max-len above it is the loader's own
// refusal, naming max_position_embeddings), whose plan at that length fits `device_bytes` on EVERY
// device of `p` (reserve_gb bytes held back on each). 0 when not even
// min(runtime::kMinAutoMaxLen, d.trained_max_len) does.
uint32_t max_len_auto(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, size_t device_bytes,
                      double reserve_gb, bool int8_head, LfmAttn a = LfmAttn::Flash, bool prefill = false);

// The two-card split by LAYER WEIGHTS alone (runtime::pp_balance; spec 24 §4.5's two-card note: the
// hand-off is the residual row, so there is no per-length KV asymmetry to balance against) - the
// split whose heavier device holds the fewest layer bytes, device 0's embedding and the last
// device's final norm / head counted as each side's fixed overhead.
uint32_t pp_split(const model::Lfm2MoeDesc& d, bool int8_head);

// "weights W GB, kv K GB, conv rings C B, rope R B, reserve Z GB, total T GB" (+ " of D GB" when
// `device_bytes` is given).
std::string describe(const Plan& p, size_t device_bytes = 0);

}  // namespace runtime::lfm2moe

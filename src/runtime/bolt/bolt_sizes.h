#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "loader/bolt_layout.h"
#include "model/bolt.h"
#include "runtime/memory_plan.h"     // kMaxLenQuantum, kMinAutoMaxLen, kDefaultReserveGb
#include "runtime/pipeline_plan.h"   // pp_balance

// Spec 25b Task 5: Nemotron 3.5 Bolt's memory plan as device-free arithmetic - the split by bytes (runtime::pp_balance, spec
// 16b's rule, not a second one) and `--max-len auto` (the largest multiple of 256 that fits EVERY device and is at most
// max_position_embeddings, 262144 - Decision 14). Host only (tests/runtime/bolt_plan_test.cc). 25c adds the decode scratch and
// the launch counts, 25d the prefill scratch: the BoltReserve below carries them as caller-given terms.
//
// Per device, the terms (all derived from model::BoltDesc and loader/bolt_layout.h - the plan IS the allocation, the loader
// asserts `weights` against the same formula):
//   weights         loader::bolt::device_weight_bytes (its blocks, the embedding on device 0, norm_f + the head on the last)
//   kv              its attention blocks' K and V at max_len (bf16; 8,192 B a position over all 8 blocks)
//   ssm             one SSM state (fp32) per Mamba block it holds
//   conv_rings      its Mamba blocks' raw-xBC ring (16 slots)
//   decode_scratch, prefill_scratch   per device (25c / 25d fill them)
//   reserve         runtime::kDefaultReserveGb (1.5 GB) unless given
namespace runtime::bolt {

struct BoltReserve {
  size_t decode_scratch = 0, prefill_scratch = 0;     // bytes, charged to every device
  bool bf16_state = false;                            // spec 25c, Decision 2's arm: the SSM state in bf16 (half the bytes)
};

struct DevicePlan {
  size_t weights = 0, kv = 0, ssm = 0, conv_rings = 0, decode_scratch = 0, prefill_scratch = 0, reserve = 0;
  size_t total() const { return weights + kv + ssm + conv_rings + decode_scratch + prefill_scratch + reserve; }
};

struct BoltPlan {
  model::BoltPlacement place;       // devices, split (the first block of device 1; layers on one device), layers
  uint32_t max_len = 0;
  std::vector<DevicePlan> device;   // one per device, in category bytes
};

// The plan on `device_free.size()` (1 or 2) devices. `max_len` 0 = auto (the largest multiple of 256, at most
// d.max_pos, that fits every device; throws when not even min(4096, max_pos) does); otherwise it must be a multiple of 256
// and at most d.max_pos. `split` (two devices only; default = pp_balance's, by block weights alone - the hand-off is the
// residual row, so there is no per-length KV asymmetry to balance against). Throws std::runtime_error naming the device, every
// term and its bytes when the plan does not fit - and, for the real model on one card, Decision 4 (spec 25 §4.2).
BoltPlan plan(const model::BoltDesc& d, uint32_t max_len, const std::vector<size_t>& device_free,
              const BoltReserve& extra = {}, std::optional<uint32_t> split = {},
              size_t reserve_bytes = size_t(kDefaultReserveGb * 1e9), bool int8_head = true);

// One device's terms at `max_len` under placement `p` (what plan() fits and the loader allocates).
DevicePlan device_plan(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len, bool int8_head,
                       const BoltReserve& extra, size_t reserve_bytes);

// Throws naming every term of `p` when its total exceeds device_bytes (the loader calls it before the first allocation).
void require_fits(const DevicePlan& p, uint32_t dev, uint32_t max_len, size_t device_bytes);

// The two-card split by BLOCK WEIGHTS alone (runtime::pp_balance): the split whose heavier device holds the fewest bytes,
// device 0's embedding and the last device's norm_f / head counted as each side's fixed overhead. 48 on the real model.
uint32_t pp_split(const model::BoltDesc& d, bool int8_head = true);

// "weights W GB, kv K GB, ssm S GB, conv rings C B, ... total T GB" (+ " of D GB" when `device_bytes` is given).
std::string describe(const DevicePlan& p, size_t device_bytes = 0);

// ==== spec 25c: the decode list's switches, scratch and launch counts (begin) ==========================================
inline constexpr uint32_t kM = 1;   // the decode list is M = 1

// Which decode attention every capture binds - Muse's flash mg_attn (the default, held to fp64) or its eager form (the
// reference's bf16 chain, ONE launch): `B70_BOLT_ATTN=flash|eager`; unset or empty is the default; anything else throws.
enum class BoltAttn { Flash, Eager };
BoltAttn parse_attn(const std::string& v);
BoltAttn bolt_attn();                       // B70_BOLT_ATTN
const char* bolt_attn_name(BoltAttn a);
// The fused arms (spec 25 §6.1, Decision 16) - each bitwise the as-built list: `router` (the E router rows inside the MoE
// block's norm launch: -1 a MoE block), `route` (the rank recomputed in every nb_moe_up work-group: -1 a MoE block), `conv`
// (the conv inside nb_ssm: -1 a Mamba block). `B70_BOLT_FUSE=none|all|<comma list of router,route,conv>`; unset or empty is
// none (Decision 16's default until Task 8 records the box's A/B); anything else throws.
struct BoltFuse {
  bool router = false, route = false, conv = false;
  bool any() const { return router || route || conv; }
};
BoltFuse parse_fuse(const std::string& v);
BoltFuse bolt_fuse();                       // B70_BOLT_FUSE
std::string bolt_fuse_name(BoltFuse f);     // "none" | "all" | "router,conv" ...
// The SSM state's precision (Decision 2): fp32 the default (the config's mamba_ssm_cache_dtype), bf16 the measured arm:
// `B70_BOLT_SSM=fp32|bf16`; anything else throws.
enum class BoltState { Fp32, Bf16 };
BoltState parse_state(const std::string& v);
BoltState bolt_state();                     // B70_BOLT_SSM
const char* bolt_state_name(BoltState s);

// The route row's words (= kernels::bolt::route::kWords, checked in bolt_capture.cc) and a block's row in `routes`.
inline constexpr uint32_t kRouteWords = 48;
inline size_t route_at(uint32_t layer) { return size_t(layer) * kM * kRouteWords * 4; }
// The residual tap (debug): bf16 [layers][M][hidden] on every device.
inline size_t tap_bytes(const model::BoltDesc& d) { return size_t(d.layers) * kM * d.hidden * 2; }

// The split-K slices block l leaves for the next norm to fold (no block folds its own output - spec 25 §6.1): {s1, s2} =
// Mamba out_proj's S, attention o_proj's S, a MoE block's fc2 and shared_down S (the latter's slices right after the
// former's in `slices`).
struct BoltPending {
  uint32_t s1 = 0, s2 = 0;
  uint32_t total() const { return s1 + s2; }
};
BoltPending pending_of(const model::BoltDesc& d, uint32_t layer);
// The hand-off at the cut (devices 2): h (kM x hidden bf16) and ALWAYS the pending slices of block split - 1 ((s1 + s2) x
// kM x hidden fp32 in the landing's sums region - spec 25 §4.1); device 1's first norm folds them.
PpLandingLayout landing_layout(const model::BoltDesc& d, const model::BoltPlacement& p);
BoltPending cut_pending(const model::BoltDesc& d, const model::BoltPlacement& p);

// One device's decode scratch at kM rows (every device has the whole set: the lists are per device), never zeroed (no step
// reads scratch it has not written first), but `ones` (filled once with bf16 1.0 - Decision 15's gate):
//   h0, h1, x        bf16 [M][4096]          the residual (ping-ponged between norm sites), every norm's output
//   slices           fp32 [widest][M]        every int4 GEMV's split-K output: the widest S x N of the linears, and a MoE
//                                            block's fc2 + shared_down pair ((s1 + s2) x 4096) side by side
//   xc, y, g         bf16 [M][10240], [M][8192], [M][8192]   nb_conv's, nb_ssm's, nb_gnorm's outputs
//   logits_r         fp32 [M][max E]         the router's logits (or the arm router's)
//   routes           u32  [layers][M][48]    every MoE block's route row
//   lat, sh          bf16 [M][1024], [M][5376]   nb_sh_act's
//   moe_a, moe_r     bf16 [M][9][max I], [M][1024]   the experts' relu2 activations, the combine
//   attn_q           fp32 [M][32][128];  attn_part fp32 [32][64][M][130];  attn_out, ones bf16 [M][4096]
//   attn_scores      fp32 [M][32][max_len]  eager only (0 under flash) - the term that scales with max_len
//   logits           fp32 [M][131072];  argmax_part fp32 [M][128][2]
// An empty term (a descriptor without that block kind) is one 64-byte line.
struct ScratchSizes {
  size_t h0 = 0, h1 = 0, x = 0, slices = 0, xc = 0, y = 0, g = 0, logits_r = 0, routes = 0, lat = 0, sh = 0, moe_a = 0,
         moe_r = 0, attn_q = 0, attn_part = 0, attn_out = 0, ones = 0, attn_scores = 0, logits = 0, argmax_part = 0;
  size_t total() const {
    return h0 + h1 + x + slices + xc + y + g + logits_r + routes + lat + sh + moe_a + moe_r + attn_q + attn_part + attn_out +
           ones + attn_scores + logits + argmax_part;
  }
};
ScratchSizes scratch_sizes(const model::BoltDesc& d, uint32_t max_len, BoltAttn a);
// One device's persistent group: control (shared), its attention blocks' K and V [max_len][2][128] each, its Mamba blocks'
// raw-xBC rings [16][10240] and SSM states [128][64][128] (fp32, or bf16 under the arm). An empty group is one 64-byte
// line (bolt_decode_scratch counts the lines plan()'s exact terms leave out).
struct PersistentSizes {
  size_t control = 0, kv_k = 0, kv_v = 0, conv_ring = 0, ssm = 0;
  size_t total() const { return control + kv_k + kv_v + conv_ring + ssm; }
};
PersistentSizes persistent_sizes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len,
                                 BoltState s);
inline size_t kv_layer_bytes(const model::BoltDesc& d, uint32_t max_len) { return size_t(max_len) * d.kv_n() * 2; }
inline size_t ring_layer_bytes(const model::BoltDesc& d) { return size_t(loader::bolt::kConvRing) * d.m_conv_dim * 2; }
inline size_t ssm_layer_bytes(const model::BoltDesc& d, BoltState s) {
  return size_t(d.m_heads) * d.m_head_dim * d.m_state * (s == BoltState::Bf16 ? 2 : 4);
}
// --- spec 25e: prefix-cache snapshots (spec 7's PrefixCache on Bolt, spec 25 §6.3) -------------------------------------------
// What a session leaves behind between replays, besides `pos`: the 8 attention blocks' growing KV (the cache's BLOCKS, as every
// model's) and, per Mamba block, the SSM state and the raw-xBC conv ring. The recurrent state exists for exactly ONE position - so a
// snapshot is taken only where a restore can use it (the block ends, spec 7's request end) and a hit restores the largest block end
// <= the common prefix (vLLM's `--mamba-cache-mode align`, the same idea).
//   state, per Mamba block in layer order: the SSM state [128][64][128] (fp32, or bf16 under B70_BOLT_SSM=bf16 - 167,772,160 B /
//   83,886,080 B over the 40 blocks), then the three raw-xBC columns BEFORE pos (positions pos - 1, pos - 2, pos - 3, in this
//   order, 10240 bf16 each: 2,457,600 B over the 40 blocks), written back at their ring slots (q % 16) on a restore - the decode
//   conv of pos reads exactly pos - 3 .. pos - 1, and a position below 0 is a zero row on the host (the conv reads 0 there; a
//   restore at pos 0..2 writes the zeros into the slots, which nothing else reads).
//   kv     [K | V][attention block, in layer order][end - begin][kv_heads x head_dim] bf16 (runtime::Engine's order): 8,192 B a
//          position at the real shapes.
// One host layout for --pp 1 and --pp 2 (spec 16b's rule: an entry moves between one card and two unchanged - every block is
// addressed through the placement, never by device order). A run is one contiguous range of one device allocation, in host order.
inline constexpr uint32_t kStateColumns = 3;   // conv_kernel - 1
enum class SnapTensor : uint32_t { KvK, KvV, ConvRing, Ssm };
struct SnapRun {
  uint32_t device = 0;
  SnapTensor tensor = SnapTensor::KvK;
  size_t offset = 0, bytes = 0;   // within the device allocation
  bool zero = false;              // a ring column of a position before 0 (state runs only)
};
// 167,772,160 + 2,457,600 = 170,229,760 B (fp32) / 86,343,680 B (bf16) at the real shapes.
inline size_t state_snapshot_bytes(const model::BoltDesc& d, BoltState s) {
  return size_t(d.mamba_layers()) * (ssm_layer_bytes(d, s) + size_t(kStateColumns) * d.m_conv_dim * 2);
}
// 8,192 B a position at the real shapes.
inline size_t kv_snapshot_bytes(const model::BoltDesc& d, uint32_t n_pos) {
  return size_t(n_pos) * 2 * d.attn_layers() * d.kv_n() * 2;
}
// Every Mamba block's SSM state and the ring columns of positions pos - 1 .. pos - 3. Throws when the ring cannot hold them.
std::vector<SnapRun> state_runs(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t pos, BoltState s);
// Positions [begin, end) of every attention block, K then V. Throws unless begin <= end <= max_len.
std::vector<SnapRun> kv_runs(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t max_len, uint32_t begin,
                             uint32_t end);

// What BoltReserve::decode_scratch must hold for device `dev`'s decode list: the control block, scratch_sizes' total and
// the 64-byte lines of any EMPTY persistent group (plan()'s kv / ssm / conv_rings terms are exact and 0 there) - so plan()
// IS the allocation (bolt_decode_test checks it on the card).
size_t decode_scratch_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len,
                            BoltAttn a, BoltState s);

// The decode list's launches on one device (the hand-off's copies and barriers excluded; peer's pp_send / pp_recv
// included), and their sum (spec 25 §6.1):
//   device 0     nb_embed_gather
//   a block      Mamba 6 (norm, in_proj, conv, ssm, gnorm, out_proj); attention 6 (norm, q||k||v, prep, decode, reduce,
//                o) - eager 5 (Muse's mg_attn_eager is ONE launch, Decision 15); MoE 9 (norm, router, route,
//                fc1||shared_up, sh_act, up, down, fc2, shared_down); the arms: conv -1 a Mamba block, router -1 and route
//                -1 a MoE block
//   last device  the final norm (norm_f, folding the last block), the head GEMV, the two argmax stages
// 653 on the real model (40 x 6 + 8 x 6 + 40 x 9 + 5), 645 eager, 533 with every arm; two cards +0 (copy) / +2 (peer).
size_t device_launches(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, BoltAttn a, BoltFuse f,
                       PpHandoff h);
size_t decode_launches(const model::BoltDesc& d, const model::BoltPlacement& p, BoltAttn a, BoltFuse f, PpHandoff h);
// ==== spec 25c (end) ===================================================================================================

// ==== spec 25d: the prefill chunk's scratch, launches and cuts (begin) =================================================
// The constants kernels::bolt spells (bolt_prefill.cc static_asserts them equal), device-free here so the host tests and
// the CLI's planner need no kernel header.
inline constexpr uint32_t kPfC = 2048, kPfSub = 64, kPfTm = 32, kPfSlab = 1024, kPfInLd = 18688;
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;   // one weight batch of bf16 expert blocks
inline constexpr uint32_t kPfHdrWords = 528, kPfMaxSub = 33, kPfTopK = 9;

// Where a prefill cuts a prompt (Review Focus 1, plan 25d's walk rule): at most `chunk` rows (a multiple of 64, at most
// kPfC), every chunk ending on the ABSOLUTE 64-grid unless the prompt ends first, so a chunk boundary is a sub-chunk
// boundary of nb_pf_mamba.cl's SSD and a prompt cut at any chunk width is bitwise one chunk; with a block hook (25e) never
// past the next multiple of `block` (2048: on the grid too). tests/kernels/bolt_pf_ref.h mirrors it (bolt_plan_test).
uint32_t prefill_chunk_rows(uint32_t pos, size_t remaining, uint32_t chunk, bool hooked, uint32_t block = 2048);
// Refuses a chunk width the walk does not take: 0 < chunk <= kPfC and chunk % 64 == 0 (throws naming it).
void check_prefill_chunk(uint32_t chunk);

// The grouped MoE's tile table bound (nb_pf_moe.cl): floor((9 C + E x 31) / 32).
inline uint32_t pf_tiles(uint32_t C, uint32_t E) { return (C * kPfTopK + E * (kPfTm - 1)) / kPfTm; }
// Experts a weight batch holds at I (kPfBatchBytes / one expert's bf16 up = down block: 1024 x I x 2 B) and a MoE block's
// batches (up and down alike).
inline uint32_t pf_batch_experts(uint32_t I) { return uint32_t(kPfBatchBytes / (size_t(1024) * I * 2)); }
inline uint32_t pf_batches(const model::BoltDesc& d, uint32_t layer) {
  const uint32_t per = pf_batch_experts(d.inter.at(layer));
  return (d.experts.at(layer) + per - 1) / per;
}
// A linear's 1024-column slabs (the last one a zero-padded tail of pad256 columns when N is not whole slabs) and its
// prefill rows' pitch pad256(N).
inline uint32_t pf_slabs(uint32_t N) { return (N + kPfSlab - 1) / kPfSlab; }
inline uint32_t pf_ld(uint32_t N) { return (N + 255) / 256 * 256; }
// Every MoE block's route rows of the last chunk: u32 [layers][kPfC][48] (read_prefill_routes).
inline size_t pf_route_at(uint32_t layer) { return size_t(layer) * kPfC * kRouteWords * 4; }

// One device's prefill scratch, sized for a chunk of kPfC rows (one constant a device: the largest term is the MoE's sorted
// rows x the largest I) - every allocation the walk binds, exactly these (runtime/bolt/bolt_prefill.h's BoltPrefillScratch):
//   ids            u32 [kPfC] (host memory: the chunk's ids)
//   h0, h1, x      bf16 [kPfC][4096]            the residual (ping-ponged across fold sites), every norm's output
//   pa             fp32 [kPfC][18688]           every linear's GEMM rows (the widest pitch: in_proj's pad256)
//   pb             fp32 [kPfC][4096]            the MoE block's second source (shared_down's rows)
//   slab           bf16 [8192][1024]            one dequantised slab (the largest K)
//   seed           bf16 [3][10240]              the conv's seed;  xc bf16 [kPfC][10240]
//   dt, cum        fp32 [kPfC][128]             cb  fp32 [33][8][64][64] (C B^T per sub-chunk and group)
//   y, g           bf16 [kPfC][8192]            the scan's y (or the attention's out [kPfC][4096]), the gated norm's out
//   attn_q         fp32 [kPfC][4096]            logits fp32 [kPfC][max E]
//   routes         u32 [layers][kPfC][48]       lat bf16 [kPfC][1024], sh bf16 [kPfC][5376], r bf16 [kPfC][1024]
//   hdr, tiles, row_tok, pair_row               u32 [528], [2 x T], [T x 32], [kPfC x 9], T = pf_tiles(kPfC, max E)
//                                               (1056 at E 496)
//   xg             bf16 [T x 32][1024]          the gathered latent, then the down GEMM's rows
//   moe_h          bf16 [T x 32][max I]         the up GEMM's relu2 rows
//   w              bf16 kPfBatchBytes           one weight batch
// An empty term (a descriptor without that block kind) is one 64-byte line.
struct PrefillSizes {
  size_t ids = 0, h0 = 0, h1 = 0, x = 0, pa = 0, pb = 0, slab = 0, seed = 0, xc = 0, dt = 0, cum = 0, cb = 0, y = 0, g = 0,
         attn_q = 0, logits = 0, routes = 0, lat = 0, sh = 0, r = 0, hdr = 0, tiles = 0, row_tok = 0, pair_row = 0, xg = 0,
         moe_h = 0, w = 0;
  size_t total() const {
    return ids + h0 + h1 + x + pa + pb + slab + seed + xc + dt + cum + cb + y + g + attn_q + logits + routes + lat + sh + r +
           hdr + tiles + row_tok + pair_row + xg + moe_h + w;
  }
};
PrefillSizes prefill_sizes(const model::BoltDesc& d);
// The prefill hand-off between two cards (a second PipelineLink in copy mode, at chunk size): h [kPfC][4096] bf16 and the
// pending GEMM rows of block split - 1 ([kPfC][4096] fp32 each: o_proj's / out_proj's, or fc2's and shared_down's).
PpLandingLayout pf_landing_layout(const model::BoltDesc& d, const model::BoltPlacement& p);
size_t pf_link_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev);
// What BoltReserve::prefill_scratch must hold for device `dev`: prefill_sizes' total + (two cards) its prefill link.
size_t prefill_scratch_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev);

// The prefill walk's launches on one device a chunk, the same at every C (runtime/bolt/bolt_prefill.cc asserts it):
//   device 0     nb_pf_embed_gather                                                                                 1
//   every block  the norm site (M 2048: _NF / _S1 / _S1_T1) + each int4 linear's 1024-column slabs x (dequant + pf_gemm_T0)
//   Mamba        + seed, conv, dt, C B^T, scan, gnorm                                       (in_proj 19 slabs, out_proj 4) 53
//   attention    + nb_attn_prep, mg_pf_flash_attn (flash or eager: one launch)              (q||k||v 5, o_proj 4)     21
//   MoE          + router, route, nb_sh_act, sort, gather, combine + 2 x (up batches + down batches)
//                                                          (fc1||shared_up 7, fc2 4, shared_down 4) 37 + 2 x 2 x batches
// The head (prefill_head, the last chunk only, the last device): 4 (decode's norm at M 1, the head, two argmax stages).
size_t prefill_device_launches(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev);
size_t prefill_chunk_launches(const model::BoltDesc& d, const model::BoltPlacement& p);
inline constexpr size_t kPrefillHeadLaunches = 4;
// ==== spec 25d (end) ===================================================================================================

}  // namespace runtime::bolt

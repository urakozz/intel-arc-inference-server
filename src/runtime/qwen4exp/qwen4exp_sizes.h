#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/qwen4exp_ple_hash.h"   // Q4PleScale (header-only)
#include "model/qwen4exp.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"     // PpBalance, pp_balance, kPpDevices

// Spec 21b Task 5: Qwen3.8-Flash-Next's device memory as device-free arithmetic - the weights (the loader's
// one formula, loader/qwen4exp_layout.h) and the persistent state, the planner over them (`--layers auto`,
// `--max-len auto`) and the split by bytes (spec 16b's runtime::pp_balance - not a second rule). Host only
// (tests/runtime/qwen4exp_plan_test.cc). 21c adds the decode scratch, the hand-off and the launch counts to
// this file, 21d the prefill scratch behind a `prefill` flag, 21e the MTP buffers (Kolibri's arrangement).
// Spec 21c (built): the decode scratch, the debug tap, the two-card hand-off and the launch counts below; a plan
// is the weights, the persistent state and the decode scratch (+ the link on two cards).
//
// **The full model does not fit two B70s** (spec 21 §3): its routed experts alone are 64.17 GB at int4 g64
// against ~62 GB usable, so `require_fits` refuses it by bytes, naming spec 22 (the expert-offload tier);
// the development mode `--layers N` (the first N layers, then the final mixer and the head) is planned per
// card by `layers_that_fit`.
namespace runtime::qwen4exp {

// One device's persistent state at max_len (zeroed by reset):
//   kv          bf16 [its QSA layers][max_len][2 kv heads][256], K and V    2048 B a position a layer
//   idx_keys    bf16 [its QSA layers][max_len / 4][128]                    the compressed indexer keys, 64 B a
//                                                                          position a layer
//   idx_tail    bf16 [its QSA layers][8][128]                              the raw keys of the open block, slot
//                                                                          p % 8 (21c: 8, not spec 21 §4.2's 4 -
//                                                                          plan 21c Review Focus 3)
//   gdn_state   fp32 [its GDN layers][48][128][128]                        3,145,728 B a layer
//   conv_ring   bf16 [its GDN layers][16][10240]                           Qwen3.8's ring (gdn_step reads it)
//   ple         u32 [16] id ring + bf16 [16][10240] conv ring              the device holding the PLE layer: slot
//                                                                          p % 16 (21c: the history is 2 ids and 9
//                                                                          rows; 16-slot rings are q4_ple.cl's
//                                                                          addressing, 21b planned [2] + [9])
// An empty group (a device with no QSA / no GDN layer, the PLE state off the PLE layer's device) still
// allocates one 64-byte line.
inline constexpr uint32_t kConvRing = 16;   // = PersistentDims::kConvRing (Qwen3.8's gdn_step ring)
inline constexpr uint32_t kIdxTail = 8;     // the open block's raw keys (kernels::qwen4exp::kTailSlots)
inline constexpr uint32_t kPleRing = 16;    // the PLE id and conv rings (kernels::qwen4exp::kPleRing)
struct PersistentSizes {
  size_t control = 0, kv = 0, idx_keys = 0, idx_tail = 0, gdn_state = 0, conv_ring = 0, ple = 0;
  size_t kv_total() const { return kv + idx_keys + idx_tail; }
  size_t state() const { return control + gdn_state + conv_ring + ple; }
  size_t total() const { return kv_total() + state(); }
};
PersistentSizes persistent_sizes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev,
                                 uint32_t max_len);
// The KV and compressed keys of one position over the descriptor's QSA layers: 12 x (2048 + 64) = 25,344 B.
size_t kv_bytes_per_pos(const model::Qwen4ExpDesc& d);
size_t gdn_state_bytes_per_layer(const model::Qwen4ExpDesc& d);   // 3,145,728
size_t conv_ring_bytes_per_layer(const model::Qwen4ExpDesc& d);   // 327,680
size_t ple_state_bytes(const model::Qwen4ExpDesc& d);             // 16 x 4 + 16 x 10240 x 2
// The PLE state's layout inside its allocation: the id ring at 0, the conv ring at ple_conv_off (64-aligned).
inline constexpr size_t kPleConvOff = 64;
// The host table spec 21 §4.3 pins (16 ranges of int8 rows + scales): 51.84 GB with bf16 scales (derived).
size_t host_ple_bytes(const model::Qwen4ExpDesc& d, loader::Q4PleScale s);

// --- spec 21c: the decode list ----------------------------------------------------------------------------
inline constexpr uint32_t kM = 1;   // decode builds M = 1 (21e's verify adds M = 2..4)
// Which decode attention every capture binds: q4_qsa_attn.cl's flash form or q4_qsa_attn_eager.cl's (the
// reference's bf16 chain). `B70_Q4_ATTN=flash|eager`, read at each call; unset or empty is the default;
// anything else throws (spec 18 §10.1's switch).
enum class Q4Attn { Flash, Eager };
inline constexpr Q4Attn kDefaultQ4Attn = Q4Attn::Flash;   // until the box A/B (spec 18 §10.1's rule)
Q4Attn q4_attn();
const char* q4_attn_name(Q4Attn a);
// The QSA selection row (kernels::qwen4exp: kListMax / kListRow / kCountWord), the route row, the slices.
inline constexpr uint32_t kListMax = 2052, kListRow = 2064, kCountWord = 2052;
inline constexpr uint32_t kRouteWords = 32, kAttnTgt = 32, kAttnPart = 258, kPleConsts = 35;
inline constexpr uint32_t kMaxS = 4;   // the widest split-K a mixer GEMV writes (out / o_proj S 4)
// One device's decode scratch, sized for kM rows and the whole descriptor (every device holds the whole set:
// the lists are per device and scratch is per list; the per-layer rows - routes, selections, diagnostics - are
// indexed by absolute layer / QSA index):
//   H, xn        bf16 [M][10240]        the 4-stream residual (materialised by the combines), the HC norm's output
//   x            bf16 [M][2560]         a block's input (up_mix's), the embedding row (embed_gather's)
//   partials     fp32 [max S x N][M]    the dense GEMVs' output (q||gate||k||v S2 x 13312 at most), the mixers'
//                                       split-K slices the next combine folds
//   down_f32     fp32 [M][336]          the HC down||inject GEMV (and the final mixer's 320)
//   inj          fp32 [M][4]            the pending inject weights (up_mix writes, the next combine reads)
//   ab, gdn_o    fp32 [M][128], fp32 [M][48][128]    a||b, gdn_step's output
//   idx_f32, idx_q   fp32 [M][640], fp32 [M][4][128] the indexer GEMV, the roped query heads
//   scores       fp32 [M][max_len / 4]  a row's block scores (the term that scales: 1 MiB at 262144)
//   list         u32 [qsa layers][M][kListRow]  every QSA layer's selection (positions, count) - the gate S rows
//   diag         fp32 [qsa layers][M][2]        the 512th / 513th scores
//   attn_q, attn_gate  fp32 [M][24][256]  attn_prep's
//   attn_part    fp32 [24][kAttnTgt][M][258]  flash's slices
//   attn_out     bf16 [M][6144]         the mixer's output into out_proj / o_proj (GDN: prep_gated_head's)
//   logits_r     fp32 [M][528]          the router GEMV (+ the shared gate)
//   routes       u32 [layers][M][32]    every layer's route row (the routing diagnostic)
//   moe_h, y     bf16 [M][11][640], bf16 [M][2560]
//   ple_e, ple_kv, ple_ids   bf16 [M][2560], fp32 [M][12800], u64 [M][16]   the PLE gather / projections / ids
//   ple_consts   u64 [35]               the hash constants (multipliers, sizes, offsets)
//   logits, argmax_part   fp32 [M][248320], fp32 [M][243][2]
struct ScratchSizes {
  size_t H = 0, xn = 0, x = 0, partials = 0, down_f32 = 0, inj = 0, ab = 0, gdn_o = 0, idx_f32 = 0, idx_q = 0,
         scores = 0, list = 0, diag = 0, attn_q = 0, attn_gate = 0, attn_part = 0, attn_out = 0, logits_r = 0,
         routes = 0, moe_h = 0, y = 0, ple_e = 0, ple_kv = 0, ple_ids = 0, ple_consts = 0, logits = 0,
         argmax_part = 0;
  size_t total() const {
    return H + xn + x + partials + down_f32 + inj + ab + gdn_o + idx_f32 + idx_q + scores + list + diag + attn_q +
           attn_gate + attn_part + attn_out + logits_r + routes + moe_h + y + ple_e + ple_kv + ple_ids + ple_consts +
           logits + argmax_part;
  }
};
ScratchSizes scratch_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len, Q4Attn a = q4_attn());
size_t partials_floats(const model::Qwen4ExpDesc& d);
inline size_t route_at(uint32_t layer) { return size_t(layer) * kM * kRouteWords * 4; }
inline size_t list_at(uint32_t qsa_index) { return size_t(qsa_index) * kM * kListRow * 4; }
inline size_t diag_at(uint32_t qsa_index) { return size_t(qsa_index) * kM * 2 * 4; }
// The residual tap (debug): bf16 [layers][M][10240] on every device (its layers' rows written).
inline size_t tap_bytes(const model::Qwen4ExpDesc& d) { return size_t(d.layers) * kM * d.hc_n() * 2; }
// The injected selection (spec 21 F3's debug input): host USM [qsa layers][M][kListRow] u32 per device.
inline size_t injected_bytes(const model::Qwen4ExpDesc& d) { return size_t(d.qsa_before(d.layers)) * kM * kListRow * 4; }

// The hand-off (spec 16b's buffers through the descriptor-free layout): the MATERIALISED 4-stream H crosses -
// 10240 bf16, 20 KB, as vLLM's PP carries it - and no norm sums (pp_landing_layout's second region empty; the
// stamp word still follows it, the flag its own page).
PpLandingLayout landing_layout(const model::Qwen4ExpDesc& d);
size_t link_bytes(const model::Qwen4ExpDesc& d, uint32_t dev);

// The decode list's launches on one device (the hand-off's copies and barriers excluded; peer's pp_send /
// pp_recv included), and their sum:
//   device 0     embed_gather
//   a GDN layer  15: combine_norm, hc down, up_mix, qkv||z GEMV, a||b, gdn_step, prep_gated_head _SIG, out_proj
//                GEMV, combine_norm, hc down, up_mix, router GEMV, q4_route, q4_moe_gate_up, q4_moe_down
//   a QSA layer  19 (eager 18): combine_norm, hc down, up_mix, q||gate||k||v GEMV, indexer GEMV, attn_prep,
//                q4_qsa_prep / _score / _select, q4_qsa_attn + _reduce (eager: q4_qsa_attn_eager), o_proj GEMV, the
//                MLP side's 8 as a GDN layer's
//   the PLE layer +4: combine_norm _Y_NN (materialise H), q4_ple_gather, key||value GEMV, q4_ple_block (3 when it
//                is device 1's first layer: the landed H is already materialised)
//   last device  the final mixer's combine_norm, its down GEMV, up_mix, lm_head, argmax x 2
// 779 at 48 layers on one card (1 + 36 x 15 + 12 x 19 + 4 + 6), 767 under eager; two cards +1 (device 0 ends with
// combine_norm _Y_NN: the materialised H crosses; device 1 starts at its first layer's norm-only _X), peer +2
// (pp_send, pp_recv). `injected` (spec 21 F3's run with the reference's selections fed in): every QSA layer
// without q4_qsa_score / _select, 2 fewer.
size_t device_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, Q4Attn a,
                       PpHandoff h, bool injected = false);
size_t decode_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, Q4Attn a, PpHandoff h,
                       bool injected = false);

// --- spec 21d: the prefill chunk ------------------------------------------------------------------------------
// One chunk of at most kPfC positions (= kernels::qwen4exp::kPfC, checked in qwen4exp_prefill.cc): every allocation
// Qwen4ExpPrefillScratch makes on a device, as device-free arithmetic.
inline constexpr uint32_t kPfC = 2048;
inline constexpr uint32_t kPfTm = 32;                          // the grouped GEMM's tile rows (pf_moe_gemm TM)
inline constexpr uint32_t kPfSlab = 1024;                      // a whole slab of a dense linear
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;     // one weight batch of bf16 expert blocks
inline constexpr uint32_t kPfDenseLast = 2050;                 // the last position with <= 2051 visible (dense flash)
// The dense linears' slab walk over N columns: slabs of kPfSlab and one tail of pad256(N - n0) (k2_pf_dequant_slab /
// kol_pf_bf16_slab zero-fill its padding); the output row pitch pad256(N).
inline uint32_t pf_pad256(uint32_t n) { return (n + 255) / 256 * 256; }
inline uint32_t pf_slabs(uint32_t N) { return (N + kPfSlab - 1) / kPfSlab; }
inline uint32_t pf_slab_width(uint32_t N, uint32_t n0) { return N - n0 >= kPfSlab ? kPfSlab : pf_pad256(N - n0); }
inline uint32_t pf_ld(uint32_t N) { return pf_pad256(N); }
// The rows of a chunk at `pos` with at most 2051 visible positions (p <= 2050): QSA is exactly causal attention there
// (spec 21 §2.3), so rows [0, dense) run spec 6's flash and rows [dense, C) the sparse flash over their own lists.
inline uint32_t pf_dense_rows(uint32_t pos, uint32_t C) {
  return pos > kPfDenseLast ? 0u : (C < kPfDenseLast + 1 - pos ? C : kPfDenseLast + 1 - pos);
}
// gdn_chunk_q4 (qwen4exp_prefill_gdn.h): its launches and the scratch bytes of one GDN chunk of C rows -
//   xb bf16 [C][10240], seed bf16 [3][10240], g / beta fp32 [C][48], A / A2 fp32 [C/64][48][64][64],
//   w / u bf16 [C][48][128], o fp32 [C][48][128]   (PrefillScratch's gdn_* fields at this family's - Qwen3.8's - shape)
inline constexpr size_t kGdnChunkLaunches = 10;
inline constexpr uint32_t kGdnChunk = 64;   // the WY sub-chunk (PrefillScratchDims::kGdnChunk)
struct GdnScratchSizes {
  size_t xb = 0, seed = 0, g = 0, beta = 0, A = 0, A2 = 0, w = 0, u = 0, o = 0;
  size_t total() const { return xb + seed + g + beta + A + A2 + w + u + o; }
};
GdnScratchSizes gdn_scratch_sizes(uint32_t C);
// The grouped GEMMs' padded tile count of a C-row chunk (q4_pf_moe.cl's header has the bound): 1200 at C = 2048.
uint32_t pf_tiles(const model::Qwen4ExpDesc& d, uint32_t C);
// The expert weight batches: blocks of [K][N] bf16 (513 a layer: 512 routed + the shared) in kPfBatchBytes regions -
// gate||up 6,553,600 B a block, 81 a batch, 7 batches; down 3,276,800 B, 163, 4 (derived).
size_t pf_block_gu_bytes(const model::Qwen4ExpDesc& d);
size_t pf_block_dn_bytes(const model::Qwen4ExpDesc& d);
uint32_t pf_batch_blocks_gu(const model::Qwen4ExpDesc& d);
uint32_t pf_batch_blocks_dn(const model::Qwen4ExpDesc& d);
uint32_t pf_batches_gu(const model::Qwen4ExpDesc& d);
uint32_t pf_batches_dn(const model::Qwen4ExpDesc& d);
// Qwen4ExpPrefillScratch's allocations on one device (kPfC rows each; every device holds the whole set - the walk is
// per device, the per-layer rows are indexed by absolute layer / QSA index):
//   ids        u32  [kPfC]                       host memory: the chunk's ids (the embedding, the PLE gather and ring)
//   H, xn      bf16 [kPfC][10240]                the 4-stream residual (materialised by the combines), the HC norm
//   x          bf16 [kPfC][2560]                 a block's input (up_mix's), the embedding rows, the PLE's e rows
//   down_f32   fp32 [kPfC][512]                  the HC down||inject GEMM (pitch pf_ld(336))
//   inj        fp32 [kPfC][4]                    the pending inject weights
//   partials   fp32 [kPfC][16384]                the slab GEMMs' output (qkv||z, q||gate||k||v, out / o, the PLE kv)
//   slab       bf16 [6144][1024]                 one slab of the widest K x 1024 (the HC down's 10240 x 512 fits)
//   idx_f32    fp32 [kPfC][768]                  the indexer GEMM (pitch pf_ld(640))
//   idx_q      fp32 [kPfC][4][128]               the roped indexer queries
//   scores     fp32 [kPfC][max_len / 4]          every row's block scores - the term that scales (268 MB at 131072)
//   lists      u32  [qsa layers][kPfC][2064]     every QSA layer's selection rows of the LAST chunk (16.9 MB a layer)
//   diag       fp32 [qsa layers][kPfC][2]        their 512th / 513th scores
//   q16        bf16 [kPfC][24][256]              pf_attn_prep_q16's q
//   o          fp32 [24][kPfC][256]              the flash / sparse attention output (pf_o's layout)
//   attn_out   bf16 [kPfC][6144]                 the gated mixer output (GDN's gated head, QSA's pf_attn_gate)
//   ab         fp32 [kPfC][128]                  a||b
//   gdn        gdn_chunk_q4's scratch (qwen4exp_prefill_gdn.h)
//   logits     fp32 [kPfC][528]                  the router GEMV
//   routes     u32  [layers][kPfC][32]           every layer's route rows of the LAST chunk
//   hdr, tiles, row_tok, pair_row                the sort's header, tile table [tmax][2], row tokens [tmax x 32],
//                                                pair rows [kPfC][10]
//   xg         bf16 [tmax x 32][2560]            the gathered A, then the down GEMM's y (also the PLE gate's gated /
//                                                gn rows: 2 x [kPfC][10240] fit)
//   h          bf16 [tmax x 32][640]             the gate||up GEMM's SiLU output
//   y          bf16 [kPfC][2560]                 the MoE block's output (the next combine folds it)
//   w          bf16 kPfBatchBytes                the expert weight batch
//   ple_ids    u64  [kPfC][16]                   the PLE row ids of the chunk's rows
struct PrefillSizes {
  size_t ids = 0, H = 0, xn = 0, x = 0, down_f32 = 0, inj = 0, partials = 0, slab = 0, idx_f32 = 0, idx_q = 0, scores = 0,
         lists = 0, diag = 0, q16 = 0, o = 0, attn_out = 0, ab = 0, gdn = 0, logits = 0, routes = 0, hdr = 0, tiles = 0,
         row_tok = 0, pair_row = 0, xg = 0, h = 0, y = 0, w = 0, ple_ids = 0;
  size_t total() const {
    return ids + H + xn + x + down_f32 + inj + partials + slab + idx_f32 + idx_q + scores + lists + diag + q16 + o +
           attn_out + ab + gdn + logits + routes + hdr + tiles + row_tok + pair_row + xg + h + y + w + ple_ids;
  }
};
PrefillSizes prefill_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len);
inline size_t pf_route_at(uint32_t layer) { return size_t(layer) * kPfC * kRouteWords * 4; }
inline size_t pf_list_at(uint32_t qsa_index) { return size_t(qsa_index) * kPfC * kListRow * 4; }
inline size_t pf_diag_at(uint32_t qsa_index) { return size_t(qsa_index) * kPfC * 2 * 4; }
// The injected selection on prefill (spec 21 F3's debug input): host USM [qsa layers][kPfC][kListRow] u32 per device,
// filled by the caller's injector before each chunk (Qwen4ExpEngine::set_prefill_injector).
inline size_t pf_injected_bytes(const model::Qwen4ExpDesc& d) { return size_t(d.qsa_before(d.layers)) * kPfC * kListRow * 4; }
// The two-card prefill hand-off: a second runtime::PipelineLink in `copy` mode over spec 16b's layout at chunk size -
// kPfC rows of the materialised H (42 MB) and no norm sums.
PpLandingLayout pf_landing_layout(const model::Qwen4ExpDesc& d);
size_t pf_link_bytes(const model::Qwen4ExpDesc& d, uint32_t dev);
// The prefill walk's launches on one device for the chunk at `pos` of C rows (the hand-off's copies and barriers
// excluded): they depend on the chunk's dense / sparse rows (pf_dense_rows):
//   device 0     pf_embed_gather                                                                              1
//   a gated residual  combine_norm, the HC down slab + pf_gemm, up_mix                                         4
//   a GDN layer  2 HC + qkv||z 16 slabs x 2, a||b, gdn_chunk_q4's 10, out_proj 3 slabs x 2 + the MoE's 27      84
//   a QSA layer  2 HC + q||gate||k||v 13 x 2, the indexer 1 x 2, pf_attn_prep_q16, q4_qsa_prep, q4_qsa_ring,
//                pf_attn_gate, o_proj 3 x 2 + the MoE's 27                                                    73
//                + q4_qsa_score / _select when the chunk has sparse rows (not injected)                       + 2
//                + pf_flash_attn when it has dense rows, + q4_pf_sparse_attn when it has sparse ones        + 1 / + 1
//   the MoE      router GEMV, q4_route, sort, gather, gate||up 7 batches x 2, down 4 x 2, combine            27
//   the PLE layer + combine_norm _Y_NN (not when it is device 1's first layer: H landed materialised), the gather,
//                key||value 13 slabs x 2, q4_pf_ple_gate / _conv / _ring                                    31 (30)
//   two cards    device 0 ends with combine_norm _Y_NN (the materialised H crosses)                          + 1
// The last chunk's head on the last device: kPrefillHeadLaunches (decode's final mixer, lm_head, argmax x 2).
size_t prefill_device_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, uint32_t pos,
                               uint32_t C, bool injected = false);
size_t prefill_chunk_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t pos, uint32_t C,
                              bool injected = false);
inline constexpr size_t kPrefillHeadLaunches = 6;
// A device's prefill scratch + its prefill link (two devices): what `prefill` plans into DevicePlan::prefill_scratch.
size_t prefill_state(const model::Qwen4ExpDesc& d, uint32_t devices, uint32_t dev, uint32_t max_len);

// --- the split by bytes (16b's rule) ---------------------------------------------------------------------
// layer_bytes[l] = loader::q4_layer_bytes(d, l).total() + its state at max_len (QSA: KV, keys, tail; GDN:
// state, ring; the PLE layer: + the PLE state).
std::vector<size_t> pp_layer_bytes(const model::Qwen4ExpDesc& d, uint32_t max_len);
// dev0_fixed = the embedding + the RoPE table + control; dev1_fixed = the final mixer + lm_head (+ the MTP
// head) + the RoPE table + control - then runtime::pp_balance.
PpBalance pp_split(const model::Qwen4ExpDesc& d, uint32_t max_len, bool int8_head, bool mtp, bool prefill = false);

// --- the planner ---------------------------------------------------------------------------------------
// One device's plan in memory_line()'s components: model = weights + RoPE; kv = KV + indexer keys + tails;
// decode_state = control + GDN state + conv ring + PLE state + (21c) the decode scratch (+ the tap with
// debug_tap) + the hand-off on two cards.
struct DevicePlan : MemoryComponents {
  uint32_t device = 0, first = 0, end = 0;   // its layers [first, end)
  size_t weights = 0, rope = 0, state = 0, scratch = 0, link = 0;
  bool mtp = false;                          // the MTP head's weights are in `weights` (the last device)
  bool whole = false;                        // the descriptor is the published 48 layers (not --layers N)
};
// `prefill` (spec 21d): each device's prefill scratch and prefill link (prefill_state) in prefill_scratch.
std::vector<DevicePlan> plan(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                             bool int8_head, bool mtp, bool debug_tap = false, bool prefill = false);
// Whether every device's plan + reserve fits its capacity.
bool fits(const std::vector<DevicePlan>& p, const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);
// `--layers auto`: the largest N in [ple_layer + 1, d.layers] whose truncated model (layers [0, N), the final
// mixer, lm_head, with mtp the head) fits `devices` cards of `device_bytes` at max_len - on two cards under
// the best split for that N (any split that fits); 0 when not even N = ple_layer + 1 does.
uint32_t layers_that_fit(const model::Qwen4ExpDesc& d, uint32_t devices, uint32_t max_len, bool int8_head, bool mtp,
                         size_t device_bytes, size_t reserve, bool prefill = false);
// The truncated descriptor and the placement layers_that_fit plans: N layers; two cards at the pp_split
// balance when it fits, else the first split that does.
model::Qwen4ExpDesc truncated(const model::Qwen4ExpDesc& d, uint32_t layers);
model::Q4Placement placement_for(const model::Qwen4ExpDesc& truncated, uint32_t devices, uint32_t max_len,
                                 bool int8_head, bool mtp, size_t device_bytes, size_t reserve, bool prefill = false);
// The largest multiple of kMaxLenQuantum, at most min(cap, 262144), whose plan + reserve fits every device
// (`--max-len auto`); 0 when not even min(kMinAutoMaxLen, cap) does.
uint32_t max_len_that_fits(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, bool int8_head, bool mtp,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap = 0,
                           bool prefill = false);
// Throws std::runtime_error naming the bytes, each device's capacity and spec 22 when the placement does not
// fit - the full model's refusal ("Qwen3.8-Flash-Next holds ~69 GB of weights at int4 g64 and two B70s
// ~62 GB: it runs whole only with spec 22's expert-offload tier; use --layers N").
void require_fits(const std::vector<DevicePlan>& p, const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);
// "plan at max_len N (one card, layers [0, L)):" or "pipeline plan at max_len N, split s (...):" + one
// memory_line()-style line per device.
std::string describe(const std::vector<DevicePlan>& p, const model::Q4Placement& pl, uint32_t max_len,
                     const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);

}  // namespace runtime::qwen4exp

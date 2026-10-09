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
// `rows`: kM (1) without the MTP head, kVerifyRows (4) with it - every M-strided term x rows (spec 21e).
ScratchSizes scratch_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len, Q4Attn a = q4_attn(), uint32_t rows = kM);
size_t partials_floats(const model::Qwen4ExpDesc& d, uint32_t rows = kM);
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
                               uint32_t C, bool injected = false, bool mtp = false);
size_t prefill_chunk_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t pos, uint32_t C,
                              bool injected = false, bool mtp = false);
inline constexpr size_t kPrefillHeadLaunches = 6;
// A device's prefill scratch + its prefill link (two devices): what `prefill` plans into DevicePlan::prefill_scratch.
size_t prefill_state(const model::Qwen4ExpDesc& d, uint32_t devices, uint32_t dev, uint32_t max_len);

// --- spec 21e: the MTP head, the verify and draft lists ------------------------------------------------------
// Spec 8's machinery (runtime/engine.h's draft / verify / commit contracts) on this family. With the head loaded:
//   - every device's decode scratch holds kVerifyRows rows (the verify lists run at M = k + 1 <= 4): the M-strided
//     terms of scratch_sizes x 4, the per-layer rows (routes, selections, diagnostics) strided by 4 rows;
//   - every device holds gdn_spec: (kGdnSlots - 1) more GDN states per GDN layer, LAYER-major ([its GDN layers][3]
//     [48][128][128] fp32) - gdn_step_slots_M<M>_G1 bakes ONE layer's state as SPEC_SLOT_STRIDE, so a layer's slots
//     are its state (slot 0) and gdn_spec's three after it, whatever the layer count (Qwen3.8's binaries bake a whole
//     48-layer slot; spec 8 §3.4's rule otherwise unchanged: verify row r writes slot (live + r) % 4, commit(j) makes
//     slot (live + j) % 4 live);
//   - the PLE layer's device holds the verify's PLE rows (gated, gn bf16 [4][10240]: q4_ple_block is M = 1 only, so
//     a verify at M > 1 runs q4_pf_ple's gate / conv / ring at C = 4 - 21d's cut of the same chain);
//   - the last device holds the head's buffers (MtpSizes): its Control (the head runs one position behind the main
//     model: draft step i at pos - 1 + i), its KV [K | V][max_len][512] bf16, compressed keys, 8-slot tail ring, its
//     selection row (draft step 0's, which later steps reuse - decision 5), the R rows hh [1 + 4][10240] bf16 (row 0
//     R_{pos-1}, rows 1 .. M the last verify's pre-mixer H), the fusion's rows (xe [4][2560], xh [4][10240] bf16; fe
//     [4][2560], fh [16][2560] fp32) and the draft logits rows [3][vocab] fp32.
// The plan carries all of it (plan(..., mtp)).
inline constexpr uint32_t kMaxDraft = 3;                  // drafts an iteration: verify at M <= 4
inline constexpr uint32_t kVerifyRows = kMaxDraft + 1;    // the decode scratch's rows with the head
inline constexpr uint32_t kGdnSlots = 4;                  // gdn_step SPEC_SLOTS' N_SLOTS (slot 0 = the layer's state)
// Decision 4 (`pre_fc_norm_hidden`): vLLM's single RMS over the 4 x 2560 values (the default, ruled) or one RMS per
// stream (llama.cpp's reading of the same [10240] weight). B70_Q4_MTP_NORM=single|per_stream, read at engine
// construction; unset or empty is single; anything else throws.
enum class MtpNorm { Single, PerStream };
MtpNorm mtp_norm();
const char* mtp_norm_name(MtpNorm n);
// Decision 5 (draft attention): the ruled form REUSES draft step 0's selection on steps 1.. (vLLM's opt-in
// index_share_for_mtp_iteration - 21a found it is not vLLM's default); `fresh` (vLLM's default) selects on every step.
// B70_Q4_MTP_SELECT=reuse|fresh, read at engine construction; unset or empty is reuse; anything else throws.
enum class MtpSelect { Reuse, Fresh };
MtpSelect mtp_select();
const char* mtp_select_name(MtpSelect s);

struct MtpSizes {
  size_t ctl = 0, kv = 0, idx_keys = 0, idx_tail = 0, list = 0, diag = 0, hh = 0, xe = 0, xh = 0, fe = 0, fh = 0,
         logits = 0, routes = 0;
  size_t persistent() const { return ctl + kv + idx_keys + idx_tail + hh; }   // zeroed by reset
  size_t total() const { return persistent() + list + diag + xe + xh + fe + fh + logits + routes; }
};
MtpSizes mtp_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len);
// gdn_spec on a device: (kGdnSlots - 1) x its GDN layers x 3,145,728 B (a 64-byte line when it has none).
size_t gdn_spec_bytes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev);
// The verify's PLE rows on the PLE layer's device: gated, gn bf16 [kVerifyRows][10240] each (0 elsewhere).
size_t ple_verify_bytes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev);
// What the head adds to a device's plan beyond its weights: gdn_spec, the PLE rows, (last device) MtpSizes, and the
// decode scratch's extra rows (scratch_sizes at kVerifyRows less at 1).
size_t mtp_device_state(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, uint32_t max_len);
// The scratch's per-layer row offsets at `rows` rows a layer (1 without the head, kVerifyRows with it).
inline size_t route_at_r(uint32_t layer, uint32_t rows) { return size_t(layer) * rows * kRouteWords * 4; }
inline size_t list_at_r(uint32_t qsa_index, uint32_t rows) { return size_t(qsa_index) * rows * kListRow * 4; }
inline size_t diag_at_r(uint32_t qsa_index, uint32_t rows) { return size_t(qsa_index) * rows * 2 * 4; }
// The decode link's landing with the head: kVerifyRows rows of H (a verify list hands off M rows).
PpLandingLayout landing_layout_rows(const model::Qwen4ExpDesc& d, uint32_t rows);
size_t link_bytes_rows(const model::Qwen4ExpDesc& d, uint32_t dev, uint32_t rows);

// The lists' launches (the copies excluded, as device_launches):
//   verify at M on a device: device_launches' list at M rows - the GDN step is gdn_step_slots, and on the PLE layer
//     at M > 1 q4_ple_block's one launch is q4_pf_ple's three (+ 2) - and, on the last device, the head's KV pass over
//     the M rows (positions pos - 1 .. pos + M - 2: the head on (R_q, t_{q+1})): q4_mtp_norm, fc_embedding, fc_hidden
//     (one M = 4 GEMV a row: a row's 4 streams), q4_mtp_fuse, the attn side's gated residual (3), q||gate||k||v,
//     the indexer, attn_prep (the head's K / V), q4_qsa_prep (its tail ring and compressed keys) - 10 + M
//   draft step (the last device, M = 1): q4_mtp_norm, fc_embedding, fc_hidden, q4_mtp_fuse, the head's QSA layer
//     (q4_qsa_score / _select on step 0 and, under `fresh`, on every step) and MoE, its own final mixer, lm_head,
//     argmax x 2: 27 + (select ? 2 : 0) flash, 26 + (select ? 2 : 0) eager
//   the prefill's head pass on the last device a chunk (Review Focus 4): combine _Y_NN (the chunk's rows of R), then
//     over the head's rows (pos - 1 .. pos + C - 2; C - 1 rows from pos 0): q4_mtp_norm _PF, fc_embedding and
//     fc_hidden as bf16 slabs + pf_gemm (3 slabs each), q4_mtp_fuse, the attn side's gated residual (4), the k||v slab
//     of q||gate||k||v (one slab + pf_gemm), the indexer (2), pf_attn_prep_q16, q4_qsa_prep _PF, q4_qsa_ring: 1 + 26
size_t verify_device_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, uint32_t M,
                              Q4Attn a, PpHandoff h);
size_t verify_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t M, Q4Attn a, PpHandoff h);
size_t draft_launches(Q4Attn a, bool select);
inline uint32_t mtp_prefill_rows(uint32_t pos, uint32_t C) { return pos == 0 ? C - 1 : C; }
size_t mtp_prefill_launches(uint32_t pos, uint32_t C);
// The prefill's R rows for the head pass on the last device: bf16 [kPfC + 1][10240] (row 0 R_{pos-1}).
inline size_t mtp_prefill_R_bytes(const model::Qwen4ExpDesc& d) { return size_t(kPfC + 1) * d.hc_n() * 2; }

// --- spec 21e: spec 7's prefix-cache snapshots --------------------------------------------------------------
// The STATE at position p is everything the next token reads that is not a per-position block (plan 21e Review
// Focus 1): every GDN layer's recurrent state and its conv ring's rows of p - 3 .. p - 1 (gdn_step's window), the
// PLE layer's id history p - 1, p - 2 (q4_ple_gather's) and its conv rows p - 9 .. p - 1 (q4_ple_block's dilated
// taps read p - 9, p - 6, p - 3), and per QSA layer the open block's raw keys (positions 4 floor(p / 4) .. p - 1,
// at most 3: the tail ring's slots p % 8). The BLOCKS are per position: the QSA layers' K / V rows and the
// compressed keys of the complete blocks [begin / 4, floor(end / 4)). With the MTP head (spec 21e Task 3) the state
// adds the head's open-block raw keys - the head runs one position behind (its KV at position q is the head on
// (R_q, t_{q+1}), so at main position p it holds positions [0, p - 1): its tail is 4 floor((p - 1) / 4) .. p - 2 -
// and R_{p-1}, the main model's pre-mixer 4-stream H of position p - 1 (the next draft step's and the next head
// pass's input: Qwen3.8's spec 8 `hh` row, Engine::state_bytes); the blocks add the head's K / V and keys, last.
//
// The HOST layouts (spec 16b's rule: --pp 2's is --pp 1's byte for byte - every layer addressed through the
// placement, in layer order, whichever device holds it; the head on the last device, last):
//   state  [GDN layers: fp32 state 3,145,728 B][GDN layers: conv rows p-3, p-2, p-1, 3 x 20,480 B]
//          [PLE ids p-1, p-2: u32 x 2 (EOS 248044 before 0)][PLE conv rows p-9 .. p-1: 9 x 20,480 B]
//          [QSA layers: raw keys 4 floor(p/4) .. p-1, padded to 3 rows of 256 B]
//          (mtp) [the head's raw keys 4 floor((p-1)/4) .. p-2, padded to 3 rows][R_{p-1}: 20,480 B]
//   kv     [QSA layers: K rows [begin, end) then V rows, 1024 B a row each][QSA layers: keys of blocks
//          [begin/4, floor(end/4)), 256 B each] (mtp) [the head's K, V, then its keys - last]
// 115,651,592 B of state at the real 48 layers (derived: 36 x 3,145,728 + 36 x 61,440 + 8 + 184,320 + 12 x 768;
// + 21,248 with the head) and 25,344 B a position of blocks (12 x (2048 + 64); + 2,112 with the head).
//
// A run is one contiguous range of one device allocation (`tensor`, `offset` into it), in host order:
//   Kv / IdxKeys / IdxTail / GdnState / ConvRing   the device's kv, idx_keys, idx_tail, gdn_state, conv_ring (offset
//                                                  from the allocation's start: the layer's slice + the row)
//   PleIds / PleRing   the PLE layer device's `ple` allocation (the id ring at 0, the conv ring at kPleConvOff)
//   MtpKv / MtpIdxKeys / MtpIdxTail / MtpHidden   the head's (the last device): [K | V][max_len][512] bf16,
//                                                  [max_len / 4][128], [8][128], R_{p-1} [10240] bf16
//   GdnState with the MTP head: the engine reads / writes the LIVE slot (spec 8's gdn_live: a commit may leave the
//   state in one of the verify's slots); load_state writes slot 0 and makes it live.
// `zero` runs are positions before 0: save_state writes the cold run's history to the host (zeros; the PLE ids as
// EOS 248044 - q4_ple_gather reads a missing predecessor as EOS), load_state copies the host's bytes back into
// those slots (the kernels never read them: every read of a position before 0 is masked by p >= q). `pad` runs are
// host-only filler (the tails' rows past the open block): zeros on save, nothing written on load.
enum class SnapTensor : uint32_t {
  Kv, IdxKeys, IdxTail, GdnState, ConvRing, PleIds, PleRing, MtpKv, MtpIdxKeys, MtpIdxTail, MtpHidden
};
const char* snap_tensor_name(SnapTensor t);
struct SnapRun {
  uint32_t device = 0;
  SnapTensor tensor = SnapTensor::Kv;
  size_t offset = 0, bytes = 0;   // within the device allocation
  bool zero = false;              // positions before 0: the cold history on the host
  bool pad = false;               // host-only filler: no device range
};
inline constexpr uint32_t kSnapTailRows = 3;   // the open block's raw keys (idx_compress - 1)
size_t state_snapshot_bytes(const model::Qwen4ExpDesc& d, bool mtp);
// `begin` a multiple of 4 (the prefix cache's begin is a multiple of kBlock 2048): KV rows of [begin, end) and the
// keys of the complete blocks [begin / 4, floor(end / 4)).
size_t kv_snapshot_bytes(const model::Qwen4ExpDesc& d, uint32_t begin, uint32_t end, bool mtp);
// Throws unless pos <= max_len (state) / 4 | begin <= end <= max_len (kv).
std::vector<SnapRun> state_runs(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t pos, bool mtp);
std::vector<SnapRun> kv_runs(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                             uint32_t begin, uint32_t end, bool mtp);
// The bytes of one device allocation a run addresses (the host test's simulated devices; the engine checks its
// own buffers against it): persistent_sizes' groups, the PLE state, and the head's (spec 21e Task 3: mtp_sizes).
size_t snap_tensor_bytes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, uint32_t max_len,
                         SnapTensor t);

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

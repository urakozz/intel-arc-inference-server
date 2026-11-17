#pragma once
#include <cstdint>
#include <vector>

#include "loader/qwen4exp_repack.h"
#include "model/qwen4exp.h"

// Plan 21-QFN-v2b Task 1: one tensor-parallel card's half of a Qwen3.8-Flash-Next layer, cut from the one-card repack
// (loader::Q4Checkpoint::repack_layer's device layouts) - host only, so every device byte of a card is a byte of the
// one-card block (tests/loader/qwen4exp_synth_host_test.cc joins the halves back, bitwise). Qwen3.8's TP rules
// (loader::tp_columns) at this family's widths; the card's shapes are model::q4_tp_card's:
//
//   column-parallel  qkv||z       the card's q, k (k-heads 8r..), v and z (v-heads 24r..) columns     2560 x 8192
//                    a||b         the card's v-heads' a and b columns (24 each), zero to 128 (bf16)    2560 x 128
//                    q||gate||k||v  its q-heads' q||gate (12 x 512), its kv head's k and v (256 each)  2560 x 6656
//   row-parallel     out_proj     K rows [3072r, + 3072) (its v-heads' outputs)                        3072 x 2560
//                    o_proj       K rows [3072r, + 3072) (its q-heads' outputs)                        3072 x 2560
//   the experts      gate||up     the 16-column interleave's first / second half: gate and up columns [320r, + 320) -
//                                 layout-1 n-tiles [40r, + 40), one contiguous half of each block    2560 x 640
//                    down         K groups [5r, + 5) of every n-tile (its 320 of the 640)               320 x 2560
//                    shared       the same two cuts (int4 layout-1 blocks, or Intel's bf16 tiles)
//   the GDN small    conv taps of its q, k, v channels (5120), -exp(A_log) / dt_bias of its v-heads; the gated norm
//   whole            both hyper-connections, the router, the indexer, the QSA small block, the PLE block
//
// int4 layout 0 cuts by column (words [K/8][N], scales [K/64][N]) or by K rows (a contiguous range of both); bf16
// tiles ([n_tile][k_octet][8][16]) are untiled, cut and retiled. No value is re-quantised.
namespace loader {

// The one-card column indices card `rank` takes of a column-parallel linear (GdnQkvz, GdnAb's real 96, QsaQkvg), in
// the card's column order.
std::vector<uint32_t> q4_tp_columns(const model::Qwen4ExpDesc& one, model::Q4LinearId id, uint32_t rank);
// The one-card conv channels card `rank` takes (q, k, v), in order.
std::vector<uint32_t> q4_tp_conv_channels(const model::Qwen4ExpDesc& one, uint32_t rank);
// Card `rank`'s half of layer `layer` (`full`: the one-card repack of that layer).
Q4HostLayer q4_tp_layer(const model::Qwen4ExpDesc& one, uint32_t layer, const Q4HostLayer& full, uint32_t rank);

// The cuts as host functions (q4_tp_layer's pieces; the int8 head and the tests use them).
// bf16 gemv tiles of {K, N}: the columns `cols` (padded with zero columns to `n_out`) / the K rows [k0, k0 + k).
std::vector<uint16_t> q4_tiles_cols(const std::vector<uint16_t>& tiles, uint32_t K, uint32_t N,
                                    const std::vector<uint32_t>& cols, uint32_t n_out);
std::vector<uint16_t> q4_tiles_krows(const std::vector<uint16_t>& tiles, uint32_t K, uint32_t N, uint32_t k0, uint32_t k);
// int4 layout-1 blocks of {K, N}: n-tiles [t0, t0 + nt) / K groups [g0, g0 + ng) of every n-tile.
std::vector<uint32_t> q4_l1_ntiles(const uint32_t* block, uint32_t K, uint32_t N, uint32_t t0, uint32_t nt);
std::vector<uint32_t> q4_l1_kgroups(const uint32_t* block, uint32_t K, uint32_t N, uint32_t g0, uint32_t ng);

}  // namespace loader

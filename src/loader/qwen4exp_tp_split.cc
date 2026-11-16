#include "loader/qwen4exp_tp_split.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "loader/qwen4exp_layout.h"

// qwen4exp_tp_split.cc - plan 21v2b Task 1: the header has the cuts in one table.
namespace loader {
namespace {

using model::Q4LinearId;
using model::Qwen4ExpDesc;
using model::WeightKind;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::logic_error("loader::q4_tp_layer: " + what);
}

// The card's share of [base, base + width): [base + rank x width / 2, + width / 2).
void range(std::vector<uint32_t>& out, uint32_t base, uint32_t width, uint32_t rank) {
  const uint32_t w = width / model::kQ4TpCards;
  for (uint32_t i = 0; i < w; ++i) out.push_back(base + rank * w + i);
}

size_t tile_at(uint32_t K8, uint32_t n, uint32_t k) { return ((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16; }

// int4 layout 0 by columns / by K rows.
Q4HostWeight l0_cols(const Q4HostWeight& w, const std::vector<uint32_t>& cols, const model::GemvShape& card) {
  const uint32_t K = w.shape.K, N = w.shape.N, n = uint32_t(cols.size());
  require(card.K == K && card.N == n, "a column cut's shape");
  Q4HostWeight o;
  o.shape = card;
  o.kind = WeightKind::Int4;
  o.words.resize(size_t(K / 8) * n);
  o.scales.resize(size_t(K / 64) * n);
  for (uint32_t r = 0; r < K / 8; ++r)
    for (uint32_t j = 0; j < n; ++j) o.words[size_t(r) * n + j] = w.words[size_t(r) * N + cols[j]];
  for (uint32_t r = 0; r < K / 64; ++r)
    for (uint32_t j = 0; j < n; ++j) o.scales[size_t(r) * n + j] = w.scales[size_t(r) * N + cols[j]];
  return o;
}
Q4HostWeight l0_krows(const Q4HostWeight& w, uint32_t k0, uint32_t k, const model::GemvShape& card) {
  const uint32_t N = w.shape.N;
  require(card.K == k && card.N == N && k0 % 64 == 0 && k % 64 == 0, "a K cut's shape");
  Q4HostWeight o;
  o.shape = card;
  o.kind = WeightKind::Int4;
  o.words.assign(w.words.begin() + size_t(k0 / 8) * N, w.words.begin() + size_t((k0 + k) / 8) * N);
  o.scales.assign(w.scales.begin() + size_t(k0 / 64) * N, w.scales.begin() + size_t((k0 + k) / 64) * N);
  return o;
}

// A dense linear's cut in its form: columns (`cols`, padded to the card's N) or K rows [k0, + card K).
Q4HostWeight cut(const Q4HostWeight& w, const model::GemvShape& card, const std::vector<uint32_t>* cols, uint32_t k0) {
  if (w.kind == WeightKind::Int4) return cols ? l0_cols(w, *cols, card) : l0_krows(w, k0, card.K, card);
  require(w.kind == WeightKind::Bf16, "a dense linear is int4 or bf16");
  Q4HostWeight o;
  o.shape = card;
  o.kind = WeightKind::Bf16;
  o.tiles = cols ? q4_tiles_cols(w.tiles, w.shape.K, w.shape.N, *cols, card.N)
                 : q4_tiles_krows(w.tiles, w.shape.K, w.shape.N, k0, card.K);
  return o;
}

// One expert's (or the shared expert's) gate||up and down halves, int4 layout-1 blocks.
void l1_halves(const uint32_t* gu, const uint32_t* dn, const Qwen4ExpDesc& one, uint32_t I, uint32_t rank,
               uint32_t* gu_out, uint32_t* dn_out) {
  const uint32_t H = one.hidden, nt = 2 * I / 16 / model::kQ4TpCards, ng = I / 64 / model::kQ4TpCards;
  const std::vector<uint32_t> a = q4_l1_ntiles(gu, H, 2 * I, rank * nt, nt);
  const std::vector<uint32_t> b = q4_l1_kgroups(dn, I, H, rank * ng, ng);
  std::memcpy(gu_out, a.data(), a.size() * 4);
  std::memcpy(dn_out, b.data(), b.size() * 4);
}

}  // namespace

std::vector<uint32_t> q4_tp_columns(const Qwen4ExpDesc& one, Q4LinearId id, uint32_t rank) {
  if (rank >= model::kQ4TpCards) throw std::invalid_argument("loader::q4_tp_columns: rank " + std::to_string(rank));
  std::vector<uint32_t> out;
  switch (id) {
    case Q4LinearId::GdnQkvz: {   // q (k-heads) | k (k-heads) | v (v-heads) | z (v-heads), x 128
      const uint32_t kq = one.gdn_k_heads * one.gdn_head, v = one.gdn_v_heads * one.gdn_head;
      range(out, 0, kq, rank);
      range(out, kq, kq, rank);
      range(out, 2 * kq, v, rank);
      range(out, 2 * kq + v, v, rank);
      break;
    }
    case Q4LinearId::GdnAb:   // the real columns: a (v-heads) | b (v-heads)
      range(out, 0, one.gdn_v_heads, rank);
      range(out, one.gdn_v_heads, one.gdn_v_heads, rank);
      break;
    case Q4LinearId::QsaQkvg:   // q||gate per q-head (512 each) | k | v (kv-heads x 256)
      range(out, 0, 2 * one.q_n(), rank);
      range(out, 2 * one.q_n(), one.kv_n(), rank);
      range(out, 2 * one.q_n() + one.kv_n(), one.kv_n(), rank);
      break;
    default:
      throw std::invalid_argument("loader::q4_tp_columns: not a column-parallel linear");
  }
  return out;
}

std::vector<uint32_t> q4_tp_conv_channels(const Qwen4ExpDesc& one, uint32_t rank) {
  std::vector<uint32_t> out;
  const uint32_t kq = one.gdn_k_heads * one.gdn_head, v = one.gdn_v_heads * one.gdn_head;
  range(out, 0, kq, rank);
  range(out, kq, kq, rank);
  range(out, 2 * kq, v, rank);
  return out;
}

std::vector<uint16_t> q4_tiles_cols(const std::vector<uint16_t>& tiles, uint32_t K, uint32_t N,
                                    const std::vector<uint32_t>& cols, uint32_t n_out) {
  require(tiles.size() == size_t(K) * N && n_out % 16 == 0 && cols.size() <= n_out && K % 8 == 0, "a tile cut's shape");
  std::vector<uint16_t> o(size_t(K) * n_out, 0);
  const uint32_t K8 = K / 8;
  for (uint32_t j = 0; j < cols.size(); ++j)
    for (uint32_t k = 0; k < K; ++k) o[tile_at(K8, j, k)] = tiles[tile_at(K8, cols[j], k)];
  return o;
}

std::vector<uint16_t> q4_tiles_krows(const std::vector<uint16_t>& tiles, uint32_t K, uint32_t N, uint32_t k0, uint32_t k) {
  require(tiles.size() == size_t(K) * N && k0 % 8 == 0 && k % 8 == 0 && k0 + k <= K, "a tile K cut's shape");
  std::vector<uint16_t> o(size_t(k) * N);
  const uint32_t K8 = K / 8, k8 = k / 8;
  for (uint32_t t = 0; t < N / 16; ++t)   // per n-tile its k-octets [k0 / 8, + k / 8): contiguous 128-element runs
    std::memcpy(&o[size_t(t) * k8 * 128], &tiles[(size_t(t) * K8 + k0 / 8) * 128], size_t(k8) * 128 * 2);
  return o;
}

std::vector<uint32_t> q4_l1_ntiles(const uint32_t* block, uint32_t K, uint32_t N, uint32_t t0, uint32_t nt) {
  const size_t tile_run = size_t(K / 64) * kQ4TileU32;   // one n-tile: its K / 64 tiles
  require(t0 + nt <= N / 16, "a layout-1 n-tile cut");
  return std::vector<uint32_t>(block + t0 * tile_run, block + (t0 + nt) * tile_run);
}

std::vector<uint32_t> q4_l1_kgroups(const uint32_t* block, uint32_t K, uint32_t N, uint32_t g0, uint32_t ng) {
  const uint32_t G = K / 64;
  require(g0 + ng <= G, "a layout-1 K-group cut");
  std::vector<uint32_t> o(size_t(N / 16) * ng * kQ4TileU32);
  for (uint32_t t = 0; t < N / 16; ++t)
    std::memcpy(&o[size_t(t) * ng * kQ4TileU32], block + (size_t(t) * G + g0) * kQ4TileU32, size_t(ng) * kQ4TileU32 * 4);
  return o;
}

Q4HostLayer q4_tp_layer(const Qwen4ExpDesc& one, uint32_t layer, const Q4HostLayer& full, uint32_t rank) {
  const Qwen4ExpDesc card = model::q4_tp_card(one, rank);
  const auto shape = [&](Q4LinearId id) { return card.linear(id).shape; };
  Q4HostLayer o;
  o.hc_attn = full.hc_attn;
  o.hc_mlp = full.hc_mlp;
  o.router = full.router;
  o.ple = full.ple;
  if (one.is_qsa(layer)) {
    const std::vector<uint32_t> c = q4_tp_columns(one, Q4LinearId::QsaQkvg, rank);
    o.qsa_qkvg = cut(full.qsa_qkvg, shape(Q4LinearId::QsaQkvg), &c, 0);
    o.qsa_idx = full.qsa_idx;
    o.qsa_o = cut(full.qsa_o, shape(Q4LinearId::QsaO), nullptr, rank * card.q_n());
    o.qsa_small = full.qsa_small;
  } else {
    const std::vector<uint32_t> c = q4_tp_columns(one, Q4LinearId::GdnQkvz, rank);
    o.gdn_qkvz = cut(full.gdn_qkvz, shape(Q4LinearId::GdnQkvz), &c, 0);
    const std::vector<uint32_t> ab = q4_tp_columns(one, Q4LinearId::GdnAb, rank);
    o.gdn_ab = cut(full.gdn_ab, shape(Q4LinearId::GdnAb), &ab, 0);
    o.gdn_out = cut(full.gdn_out, shape(Q4LinearId::GdnOut), nullptr, rank * card.gdn_z_n());
    // The small block: the card's conv channels' taps, its v-heads' decay and dt_bias, the gated norm whole.
    const SmallLayout s1 = q4_gdn_small(one), sc = q4_gdn_small(card);
    o.gdn_small.assign(sc.gdn_block_bytes, 0);
    const std::vector<uint32_t> ch = q4_tp_conv_channels(one, rank);
    const size_t tap_bytes = size_t(one.conv_taps) * 4;
    for (uint32_t i = 0; i < ch.size(); ++i)
      std::memcpy(o.gdn_small.data() + sc.gdn_off_conv + i * tap_bytes,
                  full.gdn_small.data() + s1.gdn_off_conv + ch[i] * tap_bytes, tap_bytes);
    const size_t hv = size_t(card.gdn_v_heads) * 4;
    std::memcpy(o.gdn_small.data() + sc.gdn_off_nega, full.gdn_small.data() + s1.gdn_off_nega + rank * hv, hv);
    std::memcpy(o.gdn_small.data() + sc.gdn_off_dtbias, full.gdn_small.data() + s1.gdn_off_dtbias + rank * hv, hv);
    std::memcpy(o.gdn_small.data() + sc.gdn_off_gated_norm, full.gdn_small.data() + s1.gdn_off_gated_norm,
                size_t(one.gdn_head) * 2);
  }
  // The routed experts: each block's halves at the card's block offsets.
  const size_t gu1 = q4_gate_up_block_bytes(one) / 4, dn1 = q4_down_block_bytes(one) / 4;
  const size_t guc = q4_gate_up_block_bytes(card) / 4, dnc = q4_down_block_bytes(card) / 4;
  require(full.gate_up.size() == gu1 * one.experts && full.down.size() == dn1 * one.experts,
          "the one-card layer holds every expert's blocks");
  o.gate_up.resize(guc * one.experts);
  o.down.resize(dnc * one.experts);
  for (uint32_t e = 0; e < one.experts; ++e)
    l1_halves(&full.gate_up[e * gu1], &full.down[e * dn1], one, one.moe_inter, rank, &o.gate_up[e * guc],
              &o.down[e * dnc]);
  // The shared expert: int4 the experts' cuts; Intel's bf16 tiles {2560, 1280} gate||up interleave16 (the card's
  // n-tiles a contiguous half, as layout 1's) then {640, 2560} by K.
  const size_t sg1 = q4_shared_gate_up_bytes(one), sgc = q4_shared_gate_up_bytes(card);
  o.shared.assign(sgc + q4_shared_down_bytes(card), 0);
  o.shared_gate_up = sgc;
  if (one.forms.shared == model::Q4Form::Int4) {
    l1_halves(reinterpret_cast<const uint32_t*>(full.shared.data()),
              reinterpret_cast<const uint32_t*>(full.shared.data() + sg1), one, one.shared_inter, rank,
              reinterpret_cast<uint32_t*>(o.shared.data()), reinterpret_cast<uint32_t*>(o.shared.data() + sgc));
  } else {
    const uint32_t H = one.hidden, I = one.shared_inter;
    std::memcpy(o.shared.data(), full.shared.data() + rank * sgc, sgc);   // n-tiles [40r, + 40): contiguous
    std::vector<uint16_t> dn(size_t(I) * H);
    std::memcpy(dn.data(), full.shared.data() + sg1, dn.size() * 2);
    const std::vector<uint16_t> half = q4_tiles_krows(dn, I, H, rank * (I / 2), I / 2);
    std::memcpy(o.shared.data() + sgc, half.data(), half.size() * 2);
  }
  const Q4LayerBytes want = q4_layer_bytes(card, layer), got = o.bytes();
  require(got.total() == want.total() && got.experts() == want.experts() && got.shared() == want.shared(),
          "layer " + std::to_string(layer) + "'s card bytes are not q4_layer_bytes(q4_tp_card)'s");
  return o;
}

}  // namespace loader

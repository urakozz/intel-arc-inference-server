#include "loader/moe.h"

#include <cstring>
#include <stdexcept>

#include "common/repack.h"

namespace loader {
namespace {

std::string kind_name(WKind k) { return k == WKind::Int4 ? "int4" : "bf16"; }

// One int4 linear of the expected shape, or a throw naming it.
LinearSrc int4_linear(const MoeSource& src, const std::string& part, uint32_t K, uint32_t N,
                      const std::string& what) {
  LinearSrc s = src.linear(part);
  if (s.kind != WKind::Int4)
    throw std::runtime_error(what + ": '" + part + "' is " + kind_name(s.kind) +
                             "; the MoE decode kernels read int4 g64 experts (spec 15 decision 1)");
  if (s.K != K || s.N != N)
    throw std::runtime_error(what + ": '" + part + "' is K=" + std::to_string(s.K) + " N=" +
                             std::to_string(s.N) + ", the descriptor says K=" + std::to_string(K) +
                             " N=" + std::to_string(N));
  return s;
}

// One bf16 [rows][K] linear, or a throw naming it.
LinearSrc bf16_linear(const MoeSource& src, const std::string& part, uint32_t rows, uint32_t K,
                      const std::string& what) {
  LinearSrc s = src.linear(part);
  if (s.kind != WKind::Bf16)
    throw std::runtime_error(what + ": '" + part + "' is " + kind_name(s.kind) +
                             "; the router and the shared expert's gate are read as bf16 (an "
                             "AutoRound export leaves them unquantised - loader/moe_layout.h)");
  if (s.N != rows || s.K != K)
    throw std::runtime_error(what + ": '" + part + "' is [" + std::to_string(s.N) + "][" +
                             std::to_string(s.K) + "], the descriptor says [" +
                             std::to_string(rows) + "][" + std::to_string(K) + "]");
  return s;
}

size_t int4_src_bytes(const LinearSrc& s) {
  return size_t(s.K / 8) * s.N * 4 + size_t(s.K / s.group) * s.N * 2;
}

// gate||up's column map, interleaved in 16-column blocks exactly as
// common::cols_interleave16 maps the dense gate||up: fused column 32 b + l is gate
// column 16 b + l, 32 b + 16 + l is up column 16 b + l. `g_off` / `u_off` select the
// columns inside a part (0 for separate gate / up tensors; 0 and I inside a
// per-expert fused gate_up tensor of 2 I columns).
std::vector<common::ColSource> gate_up_cols(const LinearSrc& g, uint32_t g_off, const LinearSrc& u,
                                            uint32_t u_off, uint32_t I) {
  std::vector<common::ColSource> cols;
  cols.reserve(size_t(2) * I);
  for (uint32_t base = 0; base < I; base += 16) {
    for (uint32_t l = 0; l < 16; ++l) cols.push_back({g.qweight, g.scales, g_off + base + l, g.N});
    for (uint32_t l = 0; l < 16; ++l) cols.push_back({u.qweight, u.scales, u_off + base + l, u.N});
  }
  return cols;
}

}  // namespace

void repack_moe_layer(const model::ModelDesc& d, const MoeSource& src, MoeHost& out,
                      const std::string& what) {
  if (!d.is_moe()) throw std::logic_error("loader::repack_moe_layer: " + d.name + " is not MoE");
  const model::MoeDesc& m = d.moe;
  const uint32_t H = d.hidden, I = m.expert_intermediate, E = m.experts;
  if (m.shared_intermediate != I)
    throw std::logic_error(what + ": the shared expert (" + std::to_string(m.shared_intermediate) +
                           ") is not a routed expert's width (" + std::to_string(I) +
                           ") - require_loadable should have refused this model");
  const MoeLayerBytes b = moe_layer_bytes(d);
  out.router.assign(b.router / 2, 0);
  out.gate_up.resize(b.gate_up() / 4);
  out.down.resize(b.down() / 4);
  out.int4_src_bytes = out.bf16_src_bytes = 0;

  // --- the router || shared gate rows, zero-padded to router_n, then tiled --------
  {
    const LinearSrc r = bf16_linear(src, kMoeRouter, E, H, what);
    const LinearSrc g = bf16_linear(src, kMoeSharedGate, 1, H, what);
    std::vector<uint16_t> rows(size_t(m.router_n()) * H, 0);
    std::memcpy(rows.data(), r.weight, size_t(E) * H * 2);
    std::memcpy(rows.data() + size_t(E) * H, g.weight, size_t(H) * 2);
    common::repack_bf16_tiled(rows.data(), H, m.router_n(), out.router.data());
    out.bf16_src_bytes += size_t(E + 1) * H * 2;
  }

  // --- the expert blocks: routed 0..E-1, the shared expert at E --------------------
  out.fused_gate_up = src.has(moe_expert_part(0, "gate_up_proj"));
  const model::FusedLinear& sgu = d.linear(model::LinearId::GateUp);   // the shared expert's
  const model::FusedLinear& sdn = d.linear(model::LinearId::Down);     // rows (spec 15b)
  const size_t gu_words = b.gate_up_block / 4, dn_words = b.down_block / 4;
  for (uint32_t blk = 0; blk <= E; ++blk) {
    const bool shared = blk == m.shared_block();
    // gate || up
    if (shared || !out.fused_gate_up) {
      if (!shared && src.has(moe_expert_part(blk, "gate_up_proj")))
        throw std::runtime_error(what + ": expert " + std::to_string(blk) +
                                 " ships a fused gate_up_proj while expert 0 ships gate_proj / "
                                 "up_proj - one per-expert form per checkpoint");
      const std::string gp = shared ? sgu.parts.at(0) : moe_expert_part(blk, "gate_proj");
      const std::string up = shared ? sgu.parts.at(1) : moe_expert_part(blk, "up_proj");
      const LinearSrc g = int4_linear(src, gp, H, I, what);
      const LinearSrc u = int4_linear(src, up, H, I, what);
      common::repack_int4_layout1_cols(H, 2 * I, gate_up_cols(g, 0, u, 0, I),
                                       out.gate_up.data() + blk * gu_words);
      out.int4_src_bytes += int4_src_bytes(g) + int4_src_bytes(u);
    } else {
      if (src.has(moe_expert_part(blk, "gate_proj")))
        throw std::runtime_error(what + ": expert " + std::to_string(blk) +
                                 " ships gate_proj while expert 0 ships a fused gate_up_proj - "
                                 "one per-expert form per checkpoint");
      const LinearSrc gu = int4_linear(src, moe_expert_part(blk, "gate_up_proj"), H, 2 * I, what);
      common::repack_int4_layout1_cols(H, 2 * I, gate_up_cols(gu, 0, gu, I, I),
                                       out.gate_up.data() + blk * gu_words);
      out.int4_src_bytes += int4_src_bytes(gu);
    }
    // down
    const std::string dp = shared ? sdn.parts.at(0) : moe_expert_part(blk, "down_proj");
    const LinearSrc dn = int4_linear(src, dp, I, H, what);
    common::repack_int4_layout1_cols(I, H, common::cols_concat({{dn.qweight, dn.scales, dn.N}}),
                                     out.down.data() + blk * dn_words);
    out.int4_src_bytes += int4_src_bytes(dn);
  }
}

}  // namespace loader

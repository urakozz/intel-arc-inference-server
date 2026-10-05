#include "loader/moe.h"

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/repack.h"
#include "loader/rtn.h"

namespace loader {
namespace {

std::string kind_name(WKind k) { return k == WKind::Int4 ? "int4" : "bf16"; }

// One int4 linear of the expected shape, or a throw naming it.
LinearSrc check_int4(LinearSrc s, const std::string& part, uint32_t K, uint32_t N,
                     const std::string& what) {
  if (s.kind != WKind::Int4)
    throw std::runtime_error(what + ": '" + part + "' is " + kind_name(s.kind) +
                             "; the MoE decode kernels read int4 g64 experts (spec 15 decision 1)");
  if (s.K != K || s.N != N)
    throw std::runtime_error(what + ": '" + part + "' is K=" + std::to_string(s.K) + " N=" +
                             std::to_string(s.N) + ", the descriptor says K=" + std::to_string(K) +
                             " N=" + std::to_string(N));
  return s;
}
LinearSrc int4_linear(const MoeSource& src, const std::string& part, uint32_t K, uint32_t N,
                      const std::string& what) {
  return check_int4(src.linear(part), part, K, N, what);
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

// Spec 15e: an expert linear quantised at load lives here while its block is repacked.
struct RtnLinear {
  std::vector<uint32_t> qweight;
  std::vector<uint16_t> scales;
};

// An expert linear of the expected shape as int4: as shipped, or - when `rtn` is given
// and the checkpoint ships it bf16 - quantised into `rtn` (loader/rtn.h).
LinearSrc expert_linear(const MoeSource& src, const std::string& part, uint32_t K, uint32_t N,
                        const std::string& what, RtnLinear* rtn, MoeHost& out) {
  if (rtn == nullptr) return int4_linear(src, part, K, N, what);
  LinearSrc s = src.linear(part);   // once: the loader's binding counts what it hands out
  if (s.kind != WKind::Bf16) return check_int4(std::move(s), part, K, N, what);
  if (s.K != K || s.N != N)
    throw std::runtime_error(what + ": '" + part + "' is K=" + std::to_string(s.K) + " N=" +
                             std::to_string(s.N) + ", the descriptor says K=" + std::to_string(K) +
                             " N=" + std::to_string(N));
  rtn->qweight.resize(size_t(K / 8) * N);
  rtn->scales.resize(size_t(K / 64) * N);
  rtn_int4_g64(s.weight, K, N, rtn->qweight.data(), rtn->scales.data());
  out.bf16_src_bytes += size_t(K) * N * 2;
  ++out.rtn_linears;
  LinearSrc q;
  q.kind = WKind::Int4;
  q.K = K;
  q.N = N;
  q.qweight = rtn->qweight.data();
  q.scales = rtn->scales.data();
  q.group = 64;
  q.name = s.name;
  return q;
}

size_t int4_src_bytes(const LinearSrc& s) {
  return size_t(s.K / 8) * s.N * 4 + size_t(s.K / s.group) * s.N * 2;
}
// The int4 checkpoint bytes of an expert linear: 0 when it was quantised here from bf16
// (expert_linear counted its bf16 bytes).
size_t int4_src_bytes(const LinearSrc& s, const RtnLinear* rtn) {
  return rtn != nullptr && s.qweight == rtn->qweight.data() ? 0 : int4_src_bytes(s);
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
                      const std::string& what, bool rtn_bf16_experts) {
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
  out.rtn_linears = 0;
  // Spec 15e: one buffer per linear of a block (gate, up or fused gate_up, down), reused.
  RtnLinear rg, ru, rd;
  RtnLinear* const pg = rtn_bf16_experts ? &rg : nullptr;
  RtnLinear* const pu = rtn_bf16_experts ? &ru : nullptr;
  RtnLinear* const pd = rtn_bf16_experts ? &rd : nullptr;

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
      const LinearSrc g = expert_linear(src, gp, H, I, what, pg, out);
      const LinearSrc u = expert_linear(src, up, H, I, what, pu, out);
      common::repack_int4_layout1_cols(H, 2 * I, gate_up_cols(g, 0, u, 0, I),
                                       out.gate_up.data() + blk * gu_words);
      out.int4_src_bytes += int4_src_bytes(g, pg) + int4_src_bytes(u, pu);
    } else {
      if (src.has(moe_expert_part(blk, "gate_proj")))
        throw std::runtime_error(what + ": expert " + std::to_string(blk) +
                                 " ships gate_proj while expert 0 ships a fused gate_up_proj - "
                                 "one per-expert form per checkpoint");
      const LinearSrc gu =
          expert_linear(src, moe_expert_part(blk, "gate_up_proj"), H, 2 * I, what, pg, out);
      common::repack_int4_layout1_cols(H, 2 * I, gate_up_cols(gu, 0, gu, I, I),
                                       out.gate_up.data() + blk * gu_words);
      out.int4_src_bytes += int4_src_bytes(gu, pg);
    }
    // down
    const std::string dp = shared ? sdn.parts.at(0) : moe_expert_part(blk, "down_proj");
    const LinearSrc dn = expert_linear(src, dp, I, H, what, pd, out);
    common::repack_int4_layout1_cols(I, H, common::cols_concat({{dn.qweight, dn.scales, dn.N}}),
                                     out.down.data() + blk * dn_words);
    out.int4_src_bytes += int4_src_bytes(dn, pd);
  }
}

}  // namespace loader

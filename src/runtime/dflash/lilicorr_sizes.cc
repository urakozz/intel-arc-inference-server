#include "runtime/dflash/lilicorr_sizes.h"

#include <algorithm>

#include "kernels/dflash_dims.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/dflash_sizes.h"

namespace runtime::dflash {

namespace kd = kernels::dflash;

namespace {
constexpr uint32_t kTopkChunk = 4096, kPb = 64;   // ll_topk's CH, ll_attn's PB (src/kernels/CMakeLists.txt Spec 26b)
}

uint32_t lili_rows(const model::DFlashDesc& d) { return d.block; }
uint32_t lili_ring_slots(const model::DFlashDesc& d) { return d.max_window(); }

LiliHeadScratch lili_head_scratch(const model::DFlashDesc& d) {
  LiliHeadScratch s;
  const size_t R = d.lil_rows(), Hl = d.lil_hidden;
  size_t at = 0;
  auto take = [&](size_t n) {
    const size_t o = at;
    at += n;
    return o;
  };
  s.tok = take(R * Hl);
  s.pass = take(size_t(d.lil_slots()) * Hl);
  s.feat = take(R * 5);
  s.fu = take(R * Hl);
  s.fm = take(R * Hl);
  s.xh = take(R * Hl);
  s.xn = take(R * Hl);
  s.qkv = take(R * 3 * Hl);
  s.o = take(R * Hl);
  s.u = take(R * d.lil_mlp);
  s.a = take(Hl);
  s.an = take(Hl);
  s.fh = take(R * Hl);
  s.ov = take(R * 2 * d.factor_dim);
  s.ao = take(d.factor_dim);
  s.lat = take(size_t(d.lil_slots()) * d.cand_topk * d.cand_topk);
  s.total = at;
  return s;
}

LiliSizes lili_sizes(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t vocab, uint32_t max_len) {
  const size_t M = lili_rows(d), S = d.lil_slots(), K = d.cand_topk, H = d.hidden, ring = lili_ring_slots(d);
  LiliSizes s;
  s.dctl = sizeof(LiliCtl);
  s.out = sizeof(LiliDraftOut);
  s.rope = (size_t(max_len) + M) * d.head_dim * 4;
  s.ring_k = size_t(d.layers) * ring * d.kv_n() * 2;
  s.ring_v = s.ring_k;
  s.anchor = H * 4;
  s.taps = M * d.fc_k(target_hidden) * 2;
  s.resid = s.h = M * H * 4;
  s.x = M * H * 2;
  s.xa = M * std::max({size_t(H), size_t(d.q_n()), size_t(d.inter)}) * 2;
  s.gout = M * std::max({size_t(d.qkv_n()), size_t(H), size_t(d.gate_up_n())}) * 4;
  s.co = M * d.kproj_n() * 4;
  s.q = M * d.q_n() * 4;
  s.kb = s.vb = M * d.kv_n() * 4;
  s.part = size_t(d.q_heads) * ((ring + kPb - 1) / kPb) * M * (d.head_dim + 2) * 4;
  s.kv = size_t(d.layers) * M * 2 * d.kv_n() * 4;
  s.logits = S * vocab * 4;
  s.tpart = S * ((vocab + kTopkChunk - 1) / kTopkChunk) * (2 * K + 2) * 4;
  s.cand = s.logit = s.lp = S * K * 4;
  s.head = lili_head_scratch(d).total * 4;
  return s;
}

size_t lili_drafter_bytes(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t vocab, loader::DraftFormat f,
                          uint32_t max_len) {
  return loader::lilicorr_weight_bytes(d, target_hidden, f, loader::lili_nvfp4_rows_published(d)).total() +
         lili_sizes(d, target_hidden, vocab, max_len).total();
}

runtime::bolt::BoltReserve lili_bolt_reserve(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t vocab,
                                             loader::DraftFormat f, uint32_t max_len, runtime::bolt::BoltReserve base) {
  base.drafter = lili_drafter_bytes(d, target_hidden, vocab, f, max_len);
  base.drafter_embed_replica = size_t(vocab) * target_hidden * 2;
  return base;
}

size_t lili_draft_launches(const model::DFlashDesc& d) {
  return 1 + size_t(d.layers) * 16 + 4 + 16 + size_t(7) * d.lil_layers + 1;
}
size_t lili_project_launches(const model::DFlashDesc& d) { return 4 + d.layers; }

size_t lili_ring_snapshot_bytes(const model::DFlashDesc& d) {
  return 2 * size_t(d.layers) * lili_ring_slots(d) * d.kv_n() * 2 + size_t(d.hidden) * 4;
}

uint64_t lili_drafter_key(const model::DFlashDesc& d, loader::DraftFormat f, const void* norms, size_t norm_bytes) {
  uint64_t h = drafter_key(d, f, norms, norm_bytes);
  auto mix = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) h = (h ^ ((v >> (8 * i)) & 0xFF)) * 0x100000001B3ull;
  };
  mix(0x4C494C49u);   // "LILI": the form - sinks, the candidates, the head
  mix(uint32_t(f) + 1);
  for (uint32_t v : {d.cand_topk, d.lil_hidden, d.lil_layers, d.lil_heads, d.lil_mlp, d.factor_dim}) mix(v);
  return h != 0 ? h : 1;
}

}  // namespace runtime::dflash

// probe_df_attn_seqs - plan 19v3b (Task 6's early check): the drafter's attention in the row-table form (SEQS: the
// 8 rows as blocks of several slots, each block over its own slot's ring at its own position) against today's 8 rows
// over one ring.
//
//   probe_df_attn_seqs [--rounds 9] [--reps 40] [--plain-ms 29.77]
//
// No checkpoint: random bf16 rings (5 layers x 2048 positions x 8 kv heads x 128 a slot, 4 slots, each its own
// allocation as DFlashSession's), random fp32 q / block K / V. Arms, each the drafter's 5 layers of (part, reduce):
//   ks4        the part before plan 19v3b (df_attn_..._dpas_k4, 128 GRF) + the _H1_B4 reduce: 8 rows K 7, one ring
//   ks4 g256   the same source at 256 GRF a thread (_dpas_k4_g256) - production since plan 19v3b
//   ks1 g256   the part at KS 1, 256 GRF (_dpas_g256), one ring - the SEQS form's one-block twin
//   SEQS b x r the row-table pair (_dpas_g256_SEQS + _H1_B4_SEQS): b blocks of r rows (K r - 1), b rings
// Timing: one list per arm of --reps passes, executed + fenced; rounds after a warm-up, arms rotated per round; the
// median ms a pass per arm and the median paired difference to ks4, also in plain steps (--plain-ms: the int8-head
// plain step at depth 4096, 6.46 ms / 0.217 from BENCHMARKS "the final form").
// Checks: ks4 g256 bitwise ks4; each SEQS block bitwise the ks1 g256 pair on that block alone (rows moved to 0.., its
// ring, its position, its K) - df_attn_dpas.cl's property R; and each SEQS block against the production pair alone
// (KS 4: another summation order) - the worst cosine and largest bf16 difference printed.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

namespace {

constexpr uint32_t kM = 8, kNH = 32, kNKV = 8, kHD = 128, kRing = 2048, kPB = 128, kTGT = kRing / kPB, kL = 5;
constexpr uint32_t kSeqs = 4, kPart = kHD + 2;
constexpr size_t kRingLayer = size_t(kRing) * kNKV * kHD * 2;   // bytes, K or V, one layer
constexpr uint32_t kNone = 0xFFFFFFFFu;
const std::string kBase = "df_attn_M8_Q32KV8_D128_R2048_W2048_P128";

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
uint16_t to_bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float bf(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
void upload(l0::Context& ctx, void* dev, const void* host, size_t bytes) {
  l0::CmdList c = l0::CmdList::immediate(ctx);
  c.copy(dev, host, bytes);
}
void download(l0::Context& ctx, void* host, const void* dev, size_t bytes) {
  l0::CmdList c = l0::CmdList::immediate(ctx);
  c.copy(host, dev, bytes);
}
const uint8_t* at(const void* p, size_t off) { return static_cast<const uint8_t*>(p) + off; }

struct Block {
  uint32_t row0, rows, pos, K;
};

}  // namespace

int main(int argc, char** argv) {
  uint32_t rounds = 9, reps = 40;
  double plain_ms = 6.46 / 0.217;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--rounds") rounds = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--reps") reps = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--plain-ms") plain_ms = std::stod(argv[i + 1]);
    else throw std::runtime_error("unknown flag " + a);
  }
  l0::Context ctx(0);
  l0::Queue queue(ctx);
  l0::Fence fence(queue);
  std::printf("# device: %s; %u layers, ring %u, %u rounds of %u passes, plain step %.2f ms\n", ctx.name().c_str(),
              kL, kRing, rounds, reps, plain_ms);

  // ---- data ------------------------------------------------------------------------------------------------------
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::vector<l0::Mem> ring_k, ring_v;   // per slot, 5 layers
  {
    std::vector<uint16_t> h(kRingLayer * kL / 2);
    for (uint32_t s = 0; s < kSeqs; ++s)
      for (int kv = 0; kv < 2; ++kv) {
        for (auto& x : h) x = to_bf(uni(rng));
        std::vector<l0::Mem>& dst = kv ? ring_v : ring_k;
        dst.emplace_back(ctx, l0::MemKind::Device, kRingLayer * kL);
        upload(ctx, dst.back().ptr(), h.data(), kRingLayer * kL);
      }
  }
  const size_t q_n = size_t(kM) * kNH * kHD, kb_n = size_t(kM) * kNKV * kHD;
  std::vector<float> q_h(q_n), kb_h(kb_n), vb_h(kb_n);
  for (auto& x : q_h) x = uni(rng) * 2.0f;
  for (auto& x : kb_h) x = uni(rng);
  for (auto& x : vb_h) x = uni(rng);
  l0::Mem q(ctx, l0::MemKind::Device, q_n * 4), kb(ctx, l0::MemKind::Device, kb_n * 4),
      vb(ctx, l0::MemKind::Device, kb_n * 4);
  l0::Mem part(ctx, l0::MemKind::Device, size_t(kNH) * kTGT * kM * kPart * 4);
  l0::Mem out(ctx, l0::MemKind::Device, size_t(kM) * kNH * kHD * 2);
  l0::Mem ctrl(ctx, l0::MemKind::Device, 256), dctl(ctx, l0::MemKind::Device, 64);
  const float scale = 1.0f / std::sqrt(float(kHD));

  // ---- kernels ---------------------------------------------------------------------------------------------------
  l0::Module m_k4(ctx, kernels::path(kBase + "_dpas_k4")), m_k4g(ctx, kernels::path(kBase + "_dpas_k4_g256"));
  l0::Module m_k1g(ctx, kernels::path(kBase + "_dpas_g256")), m_red(ctx, kernels::path(kBase + "_H1_B4"));
  l0::Module m_spart(ctx, kernels::path(kBase + "_dpas_g256_SEQS")), m_sred(ctx, kernels::path(kBase + "_H1_B4_SEQS"));
  // plan 41k: the 2D block read forms (BLK2D) at KS 1 / 2 / 4
  const std::string kBlk = "pdfa_M8_Q32KV8_D128_R2048_W2048_P128_dpas_k";
  l0::Module m_b1(ctx, kernels::path(kBlk + "1_g256_blk")), m_b2(ctx, kernels::path(kBlk + "2_g256_blk")),
      m_b4(ctx, kernels::path(kBlk + "4_g256_blk")), m_k2g(ctx, kernels::path(kBlk + "2_g256")),
      m_b2s(ctx, kernels::path(kBlk + "2_blk"));
  // One Kernel object per launch (a list bakes the arguments at append time; per-launch objects keep the lists apart).
  std::vector<l0::Kernel> ks;
  ks.reserve(8192);
  auto make = [&](l0::Module& mod, const char* name, uint32_t wg) -> l0::Kernel& {
    ks.push_back(mod.kernel(name));
    ks.back().group_size(wg);
    return ks.back();
  };
  const uint32_t G = kNH / kNKV;

  // a one-ring pair: part `mp` at key split `kss`, ctrl / dctl given (each block alone in its own control buffers)
  auto one_layer = [&](l0::CmdList& list, uint32_t l, const void* c, const void* dc, uint32_t slot, l0::Module& mp,
                       uint32_t kss) {
    l0::Kernel& p = make(mp, "df_attn_part", 16 * G * kss);
    p.arg_ptr(0, c);
    p.arg_ptr(1, dc);
    p.arg_ptr(2, q.ptr());
    p.arg_ptr(3, at(ring_k[slot].ptr(), l * kRingLayer));
    p.arg_ptr(4, at(ring_v[slot].ptr(), l * kRingLayer));
    p.arg(5, scale);
    p.arg_ptr(6, part.ptr());
    list.launch(p, kNKV, kTGT, 1);
    l0::Kernel& r = make(m_red, "df_attn_reduce", kHD);
    r.arg_ptr(0, c);
    r.arg_ptr(1, dc);
    r.arg_ptr(2, part.ptr());
    r.arg_ptr(3, q.ptr());
    r.arg_ptr(4, kb.ptr());
    r.arg_ptr(5, vb.ptr());
    r.arg(6, scale);
    r.arg_ptr(7, out.ptr());
    list.launch(r, kNH, kM);
  };
  auto seqs_layer = [&](l0::CmdList& list, uint32_t l, const void* tab, uint32_t blocks) {
    l0::Kernel& p = make(m_spart, "df_attn_part", 16 * G);
    p.arg_ptr(0, tab);
    p.arg_ptr(1, q.ptr());
    for (uint32_t s = 0; s < kSeqs; ++s) {
      const uint32_t slot = s < blocks ? s : 0;
      p.arg_ptr(2 + s, at(ring_k[slot].ptr(), l * kRingLayer));
      p.arg_ptr(6 + s, at(ring_v[slot].ptr(), l * kRingLayer));
    }
    p.arg(10, scale);
    p.arg_ptr(11, part.ptr());
    list.launch(p, kNKV, kTGT, blocks);
    l0::Kernel& r = make(m_sred, "df_attn_reduce", kHD);
    r.arg_ptr(0, tab);
    r.arg_ptr(1, part.ptr());
    r.arg_ptr(2, q.ptr());
    r.arg_ptr(3, kb.ptr());
    r.arg_ptr(4, vb.ptr());
    r.arg(5, scale);
    r.arg_ptr(6, out.ptr());
    list.launch(r, kNH, kM);
  };

  // the layouts: positions apart so the rings' offsets differ
  const uint32_t pos[kSeqs] = {4096, 5000, 7003, 9100};
  auto layout = [&](uint32_t blocks) {
    std::vector<Block> b;
    const uint32_t n = kM / blocks;
    for (uint32_t s = 0; s < blocks; ++s) b.push_back({s * n, n, pos[s], n - 1});
    return b;
  };
  auto table = [&](const std::vector<Block>& bl) {
    std::vector<uint32_t> t(kM + 8 * kSeqs, 0);
    for (uint32_t r = 0; r < kM; ++r) t[r] = kNone;
    for (uint32_t s = 0; s < bl.size(); ++s) {
      for (uint32_t r = bl[s].row0; r < bl[s].row0 + bl[s].rows; ++r) t[r] = s;
      t[kM + 8 * s] = bl[s].row0;
      t[kM + 8 * s + 1] = bl[s].rows;
      t[kM + 8 * s + 2] = bl[s].pos;
      t[kM + 8 * s + 3] = bl[s].K;
    }
    return t;
  };
  struct Layout {
    std::vector<Block> blocks;
    l0::Mem tab;
    std::vector<l0::Mem> c, dc;
  };
  std::vector<Layout> lay;
  for (uint32_t nb : {1u, 2u, 4u}) {
    Layout L{layout(nb), l0::Mem(ctx, l0::MemKind::Device, 256), {}, {}};
    const std::vector<uint32_t> t = table(L.blocks);
    upload(ctx, L.tab.ptr(), t.data(), t.size() * 4);
    for (const Block& b : L.blocks) {
      std::vector<uint32_t> c(64, 0), dc(16, 0);
      c[0] = b.pos;   // CTRL_POS
      dc[0] = b.K;    // DCTL_K
      L.c.emplace_back(ctx, l0::MemKind::Device, 256);
      L.dc.emplace_back(ctx, l0::MemKind::Device, 64);
      upload(ctx, L.c.back().ptr(), c.data(), 256);
      upload(ctx, L.dc.back().ptr(), dc.data(), 64);
    }
    lay.push_back(std::move(L));
  }

  // ---- checks (layer 0) ------------------------------------------------------------------------------------------
  auto run = [&](l0::CmdList& list) {
    queue.execute(list, &fence);
    fence.wait();
  };
  const size_t row_n = size_t(kNH) * kHD;
  auto out_of = [&](const std::function<void(l0::CmdList&)>& f) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    f(list);
    list.close();
    run(list);
    std::vector<uint16_t> o(kM * row_n);
    download(ctx, o.data(), out.ptr(), o.size() * 2);
    return o;
  };
  auto set_inputs = [&](const std::vector<float>& qs, const std::vector<float>& kbs, const std::vector<float>& vbs) {
    upload(ctx, q.ptr(), qs.data(), q_n * 4);
    upload(ctx, kb.ptr(), kbs.data(), kb_n * 4);
    upload(ctx, vb.ptr(), vbs.data(), kb_n * 4);
  };
  bool all_ok = true;
  set_inputs(q_h, kb_h, vb_h);
  {
    const auto a = out_of([&](l0::CmdList& li) { one_layer(li, 0, lay[0].c[0].ptr(), lay[0].dc[0].ptr(), 0, m_k4, 4); });
    const auto b = out_of([&](l0::CmdList& li) { one_layer(li, 0, lay[0].c[0].ptr(), lay[0].dc[0].ptr(), 0, m_k4g, 4); });
    std::printf("# check ks4 g256 against ks4, 8 rows one ring: %s\n", a == b ? "bitwise" : "DIFFERS");
    all_ok &= a == b;
  }
  // the 2D block read forms against the same KS's per-lane loads, at every layout's positions (wrapping atoms included)
  for (const Layout& L : lay)
    for (uint32_t s = 0; s < L.blocks.size(); ++s) {
      const void* c = L.c[s].ptr();
      const void* dc = L.dc[s].ptr();
      const auto k4 = out_of([&](l0::CmdList& li) { one_layer(li, 0, c, dc, s, m_k4g, 4); });
      const auto k1 = out_of([&](l0::CmdList& li) { one_layer(li, 0, c, dc, s, m_k1g, 1); });
      const auto b4 = out_of([&](l0::CmdList& li) { one_layer(li, 0, c, dc, s, m_b4, 4); });
      const auto b1 = out_of([&](l0::CmdList& li) { one_layer(li, 0, c, dc, s, m_b1, 1); });
      const auto k2 = out_of([&](l0::CmdList& li) { one_layer(li, 0, c, dc, s, m_k2g, 2); });
      const auto b2 = out_of([&](l0::CmdList& li) { one_layer(li, 0, c, dc, s, m_b2, 2); });
      const auto b2s = out_of([&](l0::CmdList& li) { one_layer(li, 0, c, dc, s, m_b2s, 2); });
      double worst = 1.0;
      for (uint32_t r = 0; r <= L.blocks[s].K && r < kM; ++r)
        for (uint32_t h = 0; h < kNH; ++h) {
          double xy = 0, xx = 0, yy = 0;
          for (uint32_t d = 0; d < kHD; ++d) {
            const size_t i = (size_t(r) * kNH + h) * kHD + d;
            const double x = bf(b2[i]), y = bf(k4[i]);
            xy += x * y, xx += x * x, yy += y * y;
          }
          if (xx > 0 && yy > 0) worst = std::min(worst, xy / std::sqrt(xx * yy));
        }
      std::printf("# check blk at pos %u K %u: KS 4 %s, KS 1 %s, KS 2 %s / %s (128 GRF); KS 2 against KS 4 worst "
                  "cosine %.8f\n", L.blocks[s].pos, L.blocks[s].K, b4 == k4 ? "bitwise" : "DIFFERS",
                  b1 == k1 ? "bitwise" : "DIFFERS", b2 == k2 ? "bitwise" : "DIFFERS", b2s == k2 ? "bitwise" : "DIFFERS",
                  worst);
      all_ok &= b4 == k4 && b1 == k1 && b2 == k2 && b2s == k2;
    }
  for (Layout& L : lay) {
    set_inputs(q_h, kb_h, vb_h);
    const auto got = out_of([&](l0::CmdList& li) { seqs_layer(li, 0, L.tab.ptr(), uint32_t(L.blocks.size())); });
    for (uint32_t s = 0; s < L.blocks.size(); ++s) {
      const Block& b = L.blocks[s];
      // the block's rows moved to rows 0.. of q / kb / vb; the rest of the rows as they were (past K: zeroed)
      std::vector<float> qs = q_h, kbs = kb_h, vbs = vb_h;
      const size_t kr = size_t(kNKV) * kHD;
      std::memcpy(qs.data(), q_h.data() + b.row0 * row_n, b.rows * row_n * 4);
      std::memcpy(kbs.data(), kb_h.data() + b.row0 * kr, b.rows * kr * 4);
      std::memcpy(vbs.data(), vb_h.data() + b.row0 * kr, b.rows * kr * 4);
      set_inputs(qs, kbs, vbs);
      const auto twin = out_of([&](l0::CmdList& li) { one_layer(li, 0, L.c[s].ptr(), L.dc[s].ptr(), s, m_k1g, 1); });
      const auto prod = out_of([&](l0::CmdList& li) { one_layer(li, 0, L.c[s].ptr(), L.dc[s].ptr(), s, m_k4g, 4); });
      const uint16_t* g = got.data() + size_t(b.row0) * row_n;
      const bool ok = std::memcmp(g, twin.data(), b.rows * row_n * 2) == 0;
      bool nonzero = false;
      double worst_cos = 1.0, max_abs = 0.0;
      for (uint32_t r = 0; r < b.rows; ++r)
        for (uint32_t h = 0; h < kNH; ++h) {
          double xy = 0, xx = 0, yy = 0;
          for (uint32_t d = 0; d < kHD; ++d) {
            const size_t i = size_t(r) * row_n + h * kHD + d;
            const double x = bf(g[i]), y = bf(prod[i]);
            nonzero |= g[i] != 0;
            xy += x * y, xx += x * x, yy += y * y;
            max_abs = std::max(max_abs, std::fabs(x - y));
          }
          if (xx > 0 && yy > 0) worst_cos = std::min(worst_cos, xy / std::sqrt(xx * yy));
        }
      std::printf("# check %zu blocks, block %u (rows %u..%u, pos %u, K %u): %s the KS 1 pair alone%s; against the "
                  "production pair alone worst cosine %.8f, max |diff| %.3g\n",
                  L.blocks.size(), s, b.row0, b.row0 + b.rows - 1, b.pos, b.K, ok ? "bitwise" : "DIFFERS",
                  nonzero ? "" : " (all zero!)", worst_cos, max_abs);
      all_ok &= ok && nonzero;
    }
  }
  set_inputs(q_h, kb_h, vb_h);

  // ---- arms ------------------------------------------------------------------------------------------------------
  struct Arm {
    std::string name;
    l0::CmdList list;
    std::vector<double> ms;
  };
  std::vector<Arm> arms;
  auto add_arm = [&](const std::string& name, const std::function<void(l0::CmdList&, uint32_t)>& layer) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (uint32_t r = 0; r < reps; ++r)
      for (uint32_t l = 0; l < kL; ++l) layer(list, l);
    list.close();
    arms.push_back({name, std::move(list), {}});
  };
  const void* c0 = lay[0].c[0].ptr();
  const void* d0 = lay[0].dc[0].ptr();
  add_arm("ks4, 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_k4, 4); });
  add_arm("ks4 g256, 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_k4g, 4); });
  add_arm("ks1 g256, 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_k1g, 1); });
  add_arm("ks4 g256 blk, 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_b4, 4); });
  add_arm("ks2 g256 blk, 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_b2, 2); });
  add_arm("ks2 g256, 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_k2g, 2); });
  add_arm("ks2 blk (128 GRF), 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_b2s, 2); });
  add_arm("ks1 g256 blk, 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { one_layer(li, l, c0, d0, 0, m_b1, 1); });
  add_arm("SEQS 1 x 8", [&](l0::CmdList& li, uint32_t l) { seqs_layer(li, l, lay[0].tab.ptr(), 1); });
  add_arm("SEQS 2 x 4", [&](l0::CmdList& li, uint32_t l) { seqs_layer(li, l, lay[1].tab.ptr(), 2); });
  add_arm("SEQS 4 x 2", [&](l0::CmdList& li, uint32_t l) { seqs_layer(li, l, lay[2].tab.ptr(), 4); });
  for (Arm& a : arms) run(a.list);   // warm-up
  for (uint32_t rd = 0; rd < rounds; ++rd)
    for (size_t i = 0; i < arms.size(); ++i) {
      Arm& a = arms[(i + rd) % arms.size()];
      Timer t;
      t.start();
      run(a.list);
      a.ms.push_back(t.ms() / reps);
    }
  std::printf("\n| arm (5 layers of part + reduce) | ms a pass (median) | - ks4 ms (median pair) | plain steps |\n"
              "|---|---:|---:|---:|\n");
  for (Arm& a : arms) {
    std::vector<double> d;
    for (uint32_t rd = 0; rd < rounds; ++rd) d.push_back(a.ms[rd] - arms[0].ms[rd]);
    const double dm = median(d);
    std::printf("| %s | %.4f | %+.4f | %+.4f |\n", a.name.c_str(), median(a.ms), dm, dm / plain_ms);
  }
  std::printf("\n# check: %s\n", all_ok ? "every check passed" : "FAILED");
  return all_ok ? 0 : 1;
}

// probe_df_attn_seqs - plan 19v3b (Task 6's early check): the drafter's attention in the row-table form (SEQS: the
// 8 rows as blocks of several slots, each block over its own slot's ring at its own position) against today's 8 rows
// over one ring.
//
//   probe_df_attn_seqs [--rounds 9] [--reps 40] [--plain-ms 29.77]
//
// No checkpoint: random bf16 rings (5 layers x 2048 positions x 8 kv heads x 128 a slot, 4 slots, each its own
// allocation as DFlashSession's), random fp32 q / block K / V. Arms, each the drafter's 5 layers of (part, reduce):
//   base   today's production pair (df_attn_..._dpas_k4 part + _H1_B4 reduce), 8 rows K 7, one ring
//   s1x8   the SEQS pair, one block of 8 rows (the form's own overhead)
//   s2x4   the SEQS pair, 2 blocks of 4 rows (K 3), two rings
//   s4x2   the SEQS pair, 4 blocks of 2 rows (K 1), four rings
//   b2 / b4  today's pair launched once a block (2 or 4 times a layer) - the form without a row table
// Timing: one list per arm of --reps passes, executed + fenced; rounds after a warm-up, arms rotated per round; the
// median ms a pass per arm and the median paired difference to base, also in plain steps (--plain-ms: the int8-head
// plain step at depth 4096, 6.46 ms / 0.217 from BENCHMARKS "the final form").
// Check first: each SEQS block's output rows are bitwise the production pair run on that block alone (rows moved to
// 0.., its ring, its position, its K) - df_attn_dpas.cl's property R.
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
constexpr uint32_t kKS = 4, kSeqs = 4, kPart = kHD + 2;
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
  l0::Module m_part(ctx, kernels::path(kBase + "_dpas_k4")), m_red(ctx, kernels::path(kBase + "_H1_B4"));
  l0::Module m_spart(ctx, kernels::path(kBase + "_dpas_k4_SEQS")), m_sred(ctx, kernels::path(kBase + "_H1_B4_SEQS"));
  // One Kernel object per launch: a list bakes the arguments at append time, but a Kernel's arguments are its state,
  // so per-launch objects keep the arms' lists independent of each other.
  std::vector<l0::Kernel> ks;
  ks.reserve(4096);
  auto make = [&](l0::Module& mod, const char* name, uint32_t wg) -> l0::Kernel& {
    ks.push_back(mod.kernel(name));
    ks.back().group_size(wg);
    return ks.back();
  };
  const uint32_t part_wg = 16 * (kNH / kNKV) * kKS;

  // the production pair over one ring: ctrl / dctl given (each block alone in its own control buffers)
  auto base_layer = [&](l0::CmdList& list, uint32_t l, const void* c, const void* dc, uint32_t slot) {
    l0::Kernel& p = make(m_part, "df_attn_part", part_wg);
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
    l0::Kernel& p = make(m_spart, "df_attn_part", part_wg);
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
    std::vector<uint32_t> t(kM + 4 * kSeqs, 0);
    for (uint32_t r = 0; r < kM; ++r) t[r] = kNone;
    for (uint32_t s = 0; s < bl.size(); ++s) {
      for (uint32_t r = bl[s].row0; r < bl[s].row0 + bl[s].rows; ++r) t[r] = s;
      t[kM + 4 * s] = bl[s].row0;
      t[kM + 4 * s + 1] = bl[s].rows;
      t[kM + 4 * s + 2] = bl[s].pos;
      t[kM + 4 * s + 3] = bl[s].K;
    }
    return t;
  };
  // per layout its table and, per block, the block-alone control buffers
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

  // ---- check: each block bitwise its block alone (layer 0) -------------------------------------------------------
  auto run = [&](l0::CmdList& list) {
    queue.execute(list, &fence);
    fence.wait();
  };
  const size_t row_bytes = size_t(kNH) * kHD * 2;
  bool all_ok = true;
  for (Layout& L : lay) {
    std::vector<uint16_t> got(size_t(kM) * kNH * kHD);
    upload(ctx, q.ptr(), q_h.data(), q_n * 4);
    upload(ctx, kb.ptr(), kb_h.data(), kb_n * 4);
    upload(ctx, vb.ptr(), vb_h.data(), kb_n * 4);
    {
      l0::CmdList list = l0::CmdList::regular(ctx);
      seqs_layer(list, 0, L.tab.ptr(), uint32_t(L.blocks.size()));
      list.close();
      run(list);
      download(ctx, got.data(), out.ptr(), got.size() * 2);
    }
    for (uint32_t s = 0; s < L.blocks.size(); ++s) {
      const Block& b = L.blocks[s];
      // the block's rows moved to rows 0.. of q / kb / vb; the rest of the rows as they were (past K: zeroed)
      std::vector<float> qs = q_h, kbs = kb_h, vbs = vb_h;
      const size_t qr = size_t(kNH) * kHD, kr = size_t(kNKV) * kHD;
      std::memcpy(qs.data(), q_h.data() + b.row0 * qr, b.rows * qr * 4);
      std::memcpy(kbs.data(), kb_h.data() + b.row0 * kr, b.rows * kr * 4);
      std::memcpy(vbs.data(), vb_h.data() + b.row0 * kr, b.rows * kr * 4);
      upload(ctx, q.ptr(), qs.data(), q_n * 4);
      upload(ctx, kb.ptr(), kbs.data(), kb_n * 4);
      upload(ctx, vb.ptr(), vbs.data(), kb_n * 4);
      l0::CmdList list = l0::CmdList::regular(ctx);
      base_layer(list, 0, L.c[s].ptr(), L.dc[s].ptr(), s);
      list.close();
      run(list);
      std::vector<uint16_t> want(size_t(kM) * kNH * kHD);
      download(ctx, want.data(), out.ptr(), want.size() * 2);
      const bool ok = std::memcmp(got.data() + size_t(b.row0) * kNH * kHD, want.data(), b.rows * row_bytes) == 0;
      bool nonzero = false;
      for (size_t i = 0; i < b.rows * size_t(kNH) * kHD; ++i) nonzero |= want[i] != 0;
      std::printf("# check %zu blocks, block %u (rows %u..%u, pos %u, K %u): %s%s\n", L.blocks.size(), s, b.row0,
                  b.row0 + b.rows - 1, b.pos, b.K, ok ? "bitwise" : "DIFFERS", nonzero ? "" : " (all zero!)");
      all_ok &= ok && nonzero;
    }
  }
  upload(ctx, q.ptr(), q_h.data(), q_n * 4);
  upload(ctx, kb.ptr(), kb_h.data(), kb_n * 4);
  upload(ctx, vb.ptr(), vb_h.data(), kb_n * 4);

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
  // base: 8 rows, K 7, one ring at 4096 (layout 0's block-alone control)
  add_arm("base 8 rows, one ring", [&](l0::CmdList& li, uint32_t l) { base_layer(li, l, lay[0].c[0].ptr(), lay[0].dc[0].ptr(), 0); });
  add_arm("SEQS 1 x 8", [&](l0::CmdList& li, uint32_t l) { seqs_layer(li, l, lay[0].tab.ptr(), 1); });
  add_arm("SEQS 2 x 4", [&](l0::CmdList& li, uint32_t l) { seqs_layer(li, l, lay[1].tab.ptr(), 2); });
  add_arm("SEQS 4 x 2", [&](l0::CmdList& li, uint32_t l) { seqs_layer(li, l, lay[2].tab.ptr(), 4); });
  add_arm("today's pair a block, 2 x 4", [&](l0::CmdList& li, uint32_t l) {
    for (uint32_t s = 0; s < 2; ++s) base_layer(li, l, lay[1].c[s].ptr(), lay[1].dc[s].ptr(), s);
  });
  add_arm("today's pair a block, 4 x 2", [&](l0::CmdList& li, uint32_t l) {
    for (uint32_t s = 0; s < 4; ++s) base_layer(li, l, lay[2].c[s].ptr(), lay[2].dc[s].ptr(), s);
  });
  for (Arm& a : arms) run(a.list);   // warm-up
  for (uint32_t rd = 0; rd < rounds; ++rd)
    for (size_t i = 0; i < arms.size(); ++i) {
      Arm& a = arms[(i + rd) % arms.size()];
      Timer t;
      t.start();
      run(a.list);
      a.ms.push_back(t.ms() / reps);
    }
  std::printf("\n| arm (5 layers of part + reduce) | ms a pass (median) | - base ms (median pair) | plain steps |\n"
              "|---|---:|---:|---:|\n");
  for (Arm& a : arms) {
    std::vector<double> d;
    for (uint32_t rd = 0; rd < rounds; ++rd) d.push_back(a.ms[rd] - arms[0].ms[rd]);
    const double dm = median(d);
    std::printf("| %s | %.4f | %+.4f | %+.4f |\n", a.name.c_str(), median(a.ms), dm, dm / plain_ms);
  }
  std::printf("\n# check: %s\n", all_ok ? "every block bitwise its block alone" : "FAILED");
  return all_ok ? 0 : 1;
}

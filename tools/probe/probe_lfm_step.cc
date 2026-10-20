// probe_lfm_step - spec 24c P0 (plan 24c Task 1): where an LFM2.5-24B-A2B decode step's time goes, BEFORE any
// LFM2 kernel exists. The step's launch list (tools/probe/lfm_step_shapes.h: 383 launches, 277 with every
// fused arm, a GEMV-only list, 383 noops) is bound to EXISTING binaries at LFM2's shapes and replayed on the
// card, the four lists interleaved:
//
//   int4 GEMVs   gemv.cl at K 2048 N 6144 / 2048 / 3072 / 23552 and K 11776 N 2048 - the engine's own
//                binaries (gemv_M1_K..._N..._S<s>_L<l>), at model::Lfm2MoeDesc's {S, layout} (24b's
//                PROVISIONAL picks) or, with --sweep-s, at the sweep's S picks
//   router       gemv_bf16_M1_K2048_N64_C16_S16 (fp32 logits)
//   route, MoE   moe.cl at EXPERTS 64 TOP_K 4 HIDDEN 2048 INTER 1536 ROUTER_N 80 (moe_M1_E64_T4_D2048_I1536):
//                its route, gate||up and down - FIVE slots (moe.cl always carries the shared expert, here a
//                65th random block), so gate||up / down read 25 % more than LFM2's four: the table prints the
//                raw time and x 4/5 as the PROPOSED 4-slot estimate
//   attention    Kolibri's full path at Q 32 KV 8 (kol_attn_prep_M1_N6144_S1_Q32KV8_F, kol_attn_M1_T32_Q32KV8_F):
//                head_dim 128, twice LFM2's KV bytes - raw and x 1/2 PROPOSED
//   norm         kol_norm_finish at K 2048 over 8 work-groups (kol_norm_M1_K2048_G20_W8: lfm_res_norm's grid),
//                ONE launch a site; stage A of the two-launch form (prep_res_fold_M1_K2048_SP0_G20) is timed
//                on its own row (--pair)
//   conv, silu, embed   noop (src/kernels/noop.cl)
//   head         gemv_i8w_M1_K2048_N128000, argmax.cl at VOCAB 128000 VOCAB_USED 125017 (the engine's
//                lfm_argmax_stage1_M1_V128000_U125017 / lfm_argmax_stage2_V128000)
//
// Weights are random (incompressible: probe_gemv's rule - the B70 compresses device memory losslessly), one
// allocation per layer and expert block as the engine has them; the KV is `depth` + 64 positions per
// attention layer. Each list is captured twice - plain (the fence wall) and profiled (a kernel-timestamp
// event per launch, docs/15-step-anatomy.md's instrument) - and replayed `--rounds` rounds after a
// 20-replay warm-up, the four lists rotated per round (interleaved pairs: the clock state moves, spec 24c
// Global Constraints). Before every replay the host rewrites Control::pos to `depth` (argmax stage 2
// advances it), so every replay is the same step.
//
//   probe_lfm_step [--depth D] [--rounds R] [--reps N] [--list a|fused|gemv|noop|all] [--sweep-s] [--pair]
//
// Output: one markdown table per list - per launch family the in-situ us (mean per replay), launches, Σ
// durations, the plain fence wall, the gap per launch ((wall - Σ) / launches); the whole step per list; the
// derived roofline (the list's bytes / 590 GB/s). --sweep-s: every int4 shape at S in {1, 2, 4, 8, 16}
// (S | K / 64) and layouts 0 / 1, in-situ us per launch over 512 MB of distinct weight copies; the pick per
// shape and layout is probe_gemv's rule (the smallest S within 3 % of the best).
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "lfm_step_shapes.h"
#include "model/lfm2moe.h"
#include "timer.h"

namespace {

using probe_lfm::Fam;
using probe_lfm::List;

constexpr double kMeasuredGBs = 590.0;   // probe_bw through this launch path (spec 24 §3)
constexpr uint32_t kExperts = 64, kSlots = 5, kInter = 1536, kHidden = 2048, kVocab = 128000;
constexpr uint32_t kAttnHd = 128, kAttnQ = 32, kAttnKv = 8, kAttnTgt = 32, kAttnPart = kAttnHd + 2;
constexpr uint32_t kNormWgs = 8, kNormG = 20, kArgmaxGroups = (kVocab + 1023) / 1024;
constexpr uint32_t kRouterN = 80;   // moe.cl's ROUTER_N: the 64 router rows + the shared gate's + padding
constexpr size_t kTile = 544;       // a layout-1 tile: 128 u32 of nibbles + 8 u32 of scales
inline size_t int4_bytes(uint32_t K, uint32_t N) { return size_t(N / 16) * (K / 64) * kTile; }
constexpr size_t kGuBlock = size_t(2 * kInter / 16) * (kHidden / 64) * kTile;   // 3,342,336
constexpr size_t kDnBlock = size_t(kHidden / 16) * (kInter / 64) * kTile;       // 1,671,168

// --- incompressible fill ------------------------------------------------------------------------------
struct Pool {
  std::vector<uint32_t> v;
  size_t off = 0;
  Pool() : v(size_t(64) << 18) {   // 64 MB of xorshift words
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (uint32_t& w : v) {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      w = uint32_t(s >> 11);
    }
  }
  // Fills `m` from the pool, starting at a rotating offset (so no two allocations share a line's content).
  void fill(l0::CmdList& imm, l0::Mem& m) {
    const size_t pool_bytes = v.size() * 4;
    size_t done = 0;
    while (done < m.size()) {
      const size_t at = (off * 4160) % (pool_bytes - 4096) & ~size_t(63);
      const size_t n = std::min(m.size() - done, pool_bytes - at);
      imm.copy(static_cast<uint8_t*>(m.ptr()) + done, reinterpret_cast<const uint8_t*>(v.data()) + at, n);
      done += n;
      ++off;
    }
  }
};

struct Opts {
  uint32_t depth = 4096, rounds = 9, reps = 20;
  std::string list = "all";
  bool sweep = false, pair = false;
};

[[noreturn]] void usage(const std::string& why) {
  std::fprintf(stderr, "probe_lfm_step: %s\n  probe_lfm_step [--depth D] [--rounds R] [--reps N] "
               "[--list a|fused|gemv|noop|all] [--sweep-s] [--pair]\n", why.c_str());
  std::exit(2);
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// One int4 shape's {S, layout} as the lists bind it.
struct Pick {
  uint32_t S = 1, L = 1;
};

class Rig {
 public:
  Rig(l0::Context& ctx, const model::Lfm2MoeDesc& d, uint32_t depth)
      : ctx_(ctx), d_(d), depth_(depth), max_len_(depth + 64), imm_(l0::CmdList::immediate(ctx)) {
    ctrl_ = mem(l0::MemKind::Shared, 256, false);
    std::memset(ctrl_->ptr(), 0, 256);
    for (uint32_t l = 0; l < d.layers; ++l) {
      Layer L;
      if (d.is_attention(l)) {
        L.qkv = mem(int4_bytes(kHidden, d.qkv_n()));
        L.o = mem(int4_bytes(d.q_n(), kHidden));
        L.kv_k = mem(size_t(max_len_) * kAttnKv * kAttnHd * 2);
        L.kv_v = mem(size_t(max_len_) * kAttnKv * kAttnHd * 2);
      } else {
        L.in = mem(int4_bytes(kHidden, 3 * kHidden));
        L.out = mem(int4_bytes(kHidden, kHidden));
      }
      if (d.is_dense(l)) {
        L.gu = mem(int4_bytes(kHidden, 2 * d.dense_inter));
        L.dn = mem(int4_bytes(d.dense_inter, kHidden));
      } else {
        L.router = mem(size_t(kHidden) * kExperts * 2);
        L.egu = mem((kExperts + 1) * kGuBlock);   // moe.cl's block EXPERTS is the shared slot
        L.edn = mem((kExperts + 1) * kDnBlock);
      }
      layers_.push_back(std::move(L));
    }
    head_ = mem(size_t(kHidden) * kVocab + size_t(kVocab) * 4);
    norm_w_ = mem(kHidden * 4);
    qkn_ = mem(2 * kAttnHd * 4);
    size_t part = 0;
    for (const probe_lfm::Int4Shape& s : probe_lfm::int4_shapes(d)) part = std::max(part, size_t(16) * s.N * 4);
    part = std::max(part, size_t(kAttnQ + 2 * kAttnKv) * kAttnHd * 4);   // kol_attn_prep reads N 6144 x S 1
    partials_ = mem(part);
    x_ = mem(size_t(d.dense_inter) * 2);
    h_ = mem(kHidden * 2);
    sumsq_ = mem(kNormG * 4);
    logits_r_ = mem(kRouterN * 4);
    route_ = mem(32 * 4);
    moe_h_ = mem(size_t(kSlots) * kInter * 2);
    attn_q_ = mem(size_t(kAttnQ) * kAttnHd * 4);
    attn_part_ = mem(size_t(kAttnQ) * kAttnTgt * kAttnPart * 4);
    attn_out_ = mem(size_t(kAttnQ) * kAttnHd * 2);
    logits_ = mem(size_t(kVocab) * 4);
    argmax_part_ = mem(size_t(kArgmaxGroups) * 2 * 4);
    noop_out_ = mem(4096);
    set_step();
  }

  void set_step() {
    uint32_t* c = static_cast<uint32_t*>(ctrl_->ptr());
    c[0] = depth_;   // pos
    c[1] = 1;        // n_active
    c[2] = 1000;     // cur_token[0]
  }
  size_t weight_bytes() const { return weight_bytes_; }
  l0::Context& ctx() { return ctx_; }
  l0::CmdList& imm() { return imm_; }
  Pool& pool() { return pool_; }

  // The list as one closed command list (and, profiled, one event per launch).
  struct Captured {
    l0::CmdList list;
    std::vector<std::unique_ptr<l0::Kernel>> kernels;
    std::vector<Fam> fam;
    std::unique_ptr<l0::EventPool> pool;
    std::vector<l0::Event> events;
  };
  std::unique_ptr<Captured> capture(const std::vector<probe_lfm::Launch>& launches, bool profiled,
                                    const std::map<Fam, Pick>& picks) {
    auto c = std::make_unique<Captured>(Captured{l0::CmdList::regular(ctx_), {}, {}, nullptr, {}});
    if (profiled) {
      c->pool = std::make_unique<l0::EventPool>(ctx_, uint32_t(launches.size()));
      c->events.reserve(launches.size());
      for (uint32_t i = 0; i < launches.size(); ++i) c->events.emplace_back(*c->pool, i);
    }
    for (size_t i = 0; i < launches.size(); ++i) {
      bind(*c, launches[i], picks, profiled ? &c->events[i] : nullptr);
      c->fam.push_back(launches[i].fam);
    }
    c->list.close();
    return c;
  }

  // Bytes a list streams per replay (derived): weights, the KV it reads at `depth`, the head.
  struct Bytes {
    double raw = 0, proposed = 0;   // proposed: moe.cl x 4/5 on the expert reads, hd128's KV x 1/2
  };
  Bytes bytes(const std::vector<probe_lfm::Launch>& launches) const {
    Bytes b;
    for (const probe_lfm::Launch& x : launches) {
      double r = 0, p = -1;
      switch (x.fam) {
        case Fam::InProj: r = int4_bytes(kHidden, 3 * kHidden); break;
        case Fam::OutProj: case Fam::OProj: r = int4_bytes(kHidden, kHidden); break;
        case Fam::Qkv: r = int4_bytes(kHidden, d_.qkv_n()); break;
        case Fam::DenseGateUp: r = int4_bytes(kHidden, 2 * d_.dense_inter); break;
        case Fam::DenseDown: r = int4_bytes(d_.dense_inter, kHidden); break;
        case Fam::Router: r = double(kHidden) * kExperts * 2; break;
        case Fam::GateUp: r = double(kSlots) * kGuBlock; p = 4.0 * kGuBlock; break;
        case Fam::Down: r = double(kSlots) * kDnBlock; p = 4.0 * kDnBlock; break;
        case Fam::Attn: r = double(depth_ + 1) * kAttnKv * kAttnHd * 2 * 2; p = r / 2; break;
        case Fam::Head: r = double(kHidden) * kVocab + double(kVocab) * 4; break;
        default: break;
      }
      b.raw += r;
      b.proposed += p < 0 ? r : p;
    }
    return b;
  }

  // --sweep-s: one shape at {S, L}, `n` launches over distinct weight copies (>= 512 MB) - in-situ us per launch
  // (median of the profiled replays' per-launch means) and the plain wall per launch.
  std::pair<double, double> time_gemv(uint32_t K, uint32_t N, uint32_t S, uint32_t L, uint32_t reps) {
    const size_t bytes = int4_bytes(K, N);
    const size_t copies = std::max<size_t>(8, (size_t(512) << 20) / bytes);
    std::vector<std::unique_ptr<l0::Mem>> w;
    for (size_t i = 0; i < copies; ++i) w.push_back(mem(bytes, false));
    for (auto& m : w) pool_.fill(imm_, *m);
    auto x = mem(size_t(K) * 2, false);
    pool_.fill(imm_, *x);
    auto out = mem(size_t(S) * N * 4, false);
    l0::Module& mod = module(kernels::gemv_variant(1, K, N, S, L));
    l0::CmdList plain = l0::CmdList::regular(ctx_), prof = l0::CmdList::regular(ctx_);
    l0::EventPool ep(ctx_, uint32_t(copies));
    std::vector<l0::Event> ev;
    ev.reserve(copies);
    for (uint32_t i = 0; i < copies; ++i) ev.emplace_back(ep, i);
    std::vector<std::unique_ptr<l0::Kernel>> ks;
    for (int pass = 0; pass < 2; ++pass)
      for (size_t i = 0; i < copies; ++i) {
        ks.push_back(std::make_unique<l0::Kernel>(mod.kernel("gemv")));
        l0::Kernel& k = *ks.back();
        k.group_size(64);
        k.arg_ptr(0, w[i]->ptr());
        k.arg_ptr(1, static_cast<uint8_t*>(w[i]->ptr()) + size_t(K / 8) * N * 4);   // layout 0's scales
        k.arg_ptr(2, x->ptr());
        k.arg_ptr(3, out->ptr());
        (pass ? prof : plain).launch(k, N / 64, S, 1, pass ? &ev[i] : nullptr);
      }
    plain.close();
    prof.close();
    l0::Queue q(ctx_);
    l0::Fence f(q);
    std::vector<double> wall, ins;
    for (uint32_t r = 0; r < reps + 5; ++r) {
      Timer t;
      t.start();
      q.execute(plain, &f);
      f.wait();
      const double ms = t.ms();
      for (auto& e : ev) e.reset();
      q.execute(prof, &f);
      f.wait();
      double sum = 0;
      for (auto& e : ev) sum += e.duration_us();
      if (r >= 5) {
        wall.push_back(ms * 1e3 / double(copies));
        ins.push_back(sum / double(copies));
      }
    }
    return {median(ins), median(wall)};
  }

 private:
  struct Layer {
    std::unique_ptr<l0::Mem> in, out, qkv, o, gu, dn, router, egu, edn, kv_k, kv_v;
  };

  std::unique_ptr<l0::Mem> mem(size_t bytes, bool count = true) { return mem(l0::MemKind::Device, bytes, count); }
  std::unique_ptr<l0::Mem> mem(l0::MemKind kind, size_t bytes, bool count = true) {
    auto m = std::make_unique<l0::Mem>(ctx_, kind, bytes);
    if (kind == l0::MemKind::Device) pool_.fill(imm_, *m);
    if (count) weight_bytes_ += bytes;
    return m;
  }
  l0::Module& module(const std::string& variant) {
    auto it = modules_.find(variant);
    if (it == modules_.end())
      it = modules_.emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant))).first;
    return *it->second;
  }
  l0::Kernel& kernel(Captured& c, const std::string& variant, const char* entry, uint32_t wg) {
    c.kernels.push_back(std::make_unique<l0::Kernel>(module(variant).kernel(entry)));
    c.kernels.back()->group_size(wg);
    return *c.kernels.back();
  }

  void gemv(Captured& c, const l0::Mem& w, uint32_t K, uint32_t N, const void* x, Pick p, l0::Event* ev) {
    l0::Kernel& k = kernel(c, kernels::gemv_variant(1, K, N, p.S, p.L), "gemv", 64);
    k.arg_ptr(0, w.ptr());
    k.arg_ptr(1, static_cast<const uint8_t*>(w.ptr()) + size_t(K / 8) * N * 4);   // layout 0's scales
    k.arg_ptr(2, x);
    k.arg_ptr(3, partials_->ptr());
    c.list.launch(k, N / 64, p.S, 1, ev);
  }

  void bind(Captured& c, const probe_lfm::Launch& x, const std::map<Fam, Pick>& picks, l0::Event* ev) {
    const Layer* L = x.layer >= 0 ? &layers_[size_t(x.layer)] : nullptr;
    const auto pick = [&](Fam f) { return picks.at(probe_lfm::shape_fam(f)); };
    switch (x.fam) {
      case Fam::Embed: case Fam::Conv: case Fam::Silu: case Fam::Noop: {
        l0::Kernel& k = kernel(c, "noop", "noop", 16);
        k.arg_ptr(0, noop_out_->ptr());
        c.list.launch(k, 1, 1, 1, ev);
        return;
      }
      case Fam::Norm: {   // kol_norm_finish(sumsq, resid, w, x)
        l0::Kernel& k = kernel(c, "kol_norm_M1_K2048_G20_W8", "kol_norm_finish", 256);
        k.arg_ptr(0, sumsq_->ptr());
        k.arg_ptr(1, h_->ptr());
        k.arg_ptr(2, norm_w_->ptr());
        k.arg_ptr(3, x_->ptr());
        c.list.launch(k, kNormWgs, 1, 1, ev);
        return;
      }
      case Fam::InProj: gemv(c, *L->in, kHidden, 3 * kHidden, x_->ptr(), pick(x.fam), ev); return;
      case Fam::OutProj: gemv(c, *L->out, kHidden, kHidden, x_->ptr(), pick(x.fam), ev); return;
      case Fam::Qkv: gemv(c, *L->qkv, kHidden, d_.qkv_n(), x_->ptr(), pick(x.fam), ev); return;
      case Fam::OProj: gemv(c, *L->o, d_.q_n(), kHidden, attn_out_->ptr(), pick(x.fam), ev); return;
      case Fam::DenseGateUp: gemv(c, *L->gu, kHidden, 2 * d_.dense_inter, x_->ptr(), pick(x.fam), ev); return;
      case Fam::DenseDown: gemv(c, *L->dn, d_.dense_inter, kHidden, x_->ptr(), pick(x.fam), ev); return;
      case Fam::AttnPrep: {   // kol_attn_prep(ctrl, partials, qkn, rope, attn_q, kv_k, kv_v), grid (Q + 2 KV, 1)
        l0::Kernel& k = kernel(c, "kol_attn_prep_M1_N6144_S1_Q32KV8_F", "kol_attn_prep", kAttnHd);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, partials_->ptr());
        k.arg_ptr(2, qkn_->ptr());
        k.arg_ptr(3, partials_->ptr());   // the RoPE table: not read in a full (NoPE) Kolibri layer
        k.arg_ptr(4, attn_q_->ptr());
        k.arg_ptr(5, L->kv_k->ptr());
        k.arg_ptr(6, L->kv_v->ptr());
        c.list.launch(k, kAttnQ + 2 * kAttnKv, 1, 1, ev);
        return;
      }
      case Fam::Attn: {   // kol_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part), grid (KV, TGT)
        l0::Kernel& k = kernel(c, "kol_attn_M1_T32_Q32KV8_F", "kol_attn_decode", kAttnHd);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, attn_q_->ptr());
        k.arg_ptr(2, L->kv_k->ptr());
        k.arg_ptr(3, L->kv_v->ptr());
        k.arg_ptr(4, attn_part_->ptr());
        c.list.launch(k, kAttnKv, kAttnTgt, 1, ev);
        return;
      }
      case Fam::AttnReduce: {   // kol_attn_reduce(ctrl, attn_part, attn_out), grid (Q, 1)
        l0::Kernel& k = kernel(c, "kol_attn_M1_T32_Q32KV8_F", "kol_attn_reduce", kAttnHd);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, attn_part_->ptr());
        k.arg_ptr(2, attn_out_->ptr());
        c.list.launch(k, kAttnQ, 1, 1, ev);
        return;
      }
      case Fam::Router: {   // gemv_bf16(w, x, out), tiling {16, 16}
        l0::Kernel& k = kernel(c, "gemv_bf16_M1_K2048_N64_C16_S16", "gemv_bf16", 16 * 16);
        k.arg_ptr(0, L->router->ptr());
        k.arg_ptr(1, x_->ptr());
        k.arg_ptr(2, logits_r_->ptr());
        c.list.launch(k, kExperts / 16, 1, 1, ev);
        return;
      }
      case Fam::Route: {   // moe_route(logits, route), grid (1, M), WG EXPERTS
        l0::Kernel& k = kernel(c, "moe_M1_E64_T4_D2048_I1536", "moe_route", kExperts);
        k.arg_ptr(0, logits_r_->ptr());
        k.arg_ptr(1, route_->ptr());
        c.list.launch(k, 1, 1, 1, ev);
        return;
      }
      case Fam::GateUp: {   // moe_gate_up(route, x, w_gu, h), grid (SLOTS x NT_UP / 4, M), WG 64 x UP_KS
        l0::Kernel& k = kernel(c, "moe_M1_E64_T4_D2048_I1536", "moe_gate_up", 64 * 4);
        k.arg_ptr(0, route_->ptr());
        k.arg_ptr(1, x_->ptr());
        k.arg_ptr(2, L->egu->ptr());
        k.arg_ptr(3, moe_h_->ptr());
        c.list.launch(k, kSlots * (2 * kInter / 16 / 4), 1, 1, ev);
        return;
      }
      case Fam::Down: {   // moe_down(route, h, w_dn, resid), grid (HIDDEN / 16, M), WG 16 x SLOTS x DN_KS
        l0::Kernel& k = kernel(c, "moe_M1_E64_T4_D2048_I1536", "moe_down", 16 * kSlots * 2);
        k.arg_ptr(0, route_->ptr());
        k.arg_ptr(1, moe_h_->ptr());
        k.arg_ptr(2, L->edn->ptr());
        k.arg_ptr(3, h_->ptr());
        c.list.launch(k, kHidden / 16, 1, 1, ev);
        return;
      }
      case Fam::Head: {   // gemv_i8w(w, scales, x, out), grid (N / 64)
        l0::Kernel& k = kernel(c, kernels::gemv_i8w_variant(1, kHidden, kVocab), "gemv_i8w", kernels::kGemvI8wCols);
        k.arg_ptr(0, head_->ptr());
        k.arg_ptr(1, static_cast<uint8_t*>(head_->ptr()) + size_t(kHidden) * kVocab);
        k.arg_ptr(2, x_->ptr());
        k.arg_ptr(3, logits_->ptr());
        c.list.launch(k, kVocab / kernels::kGemvI8wCols, 1, 1, ev);
        return;
      }
      case Fam::Argmax1: {   // argmax_stage1(logits, part), grid (groups, M)
        l0::Kernel& k = kernel(c, "lfm_argmax_stage1_M1_V128000_U125017", "argmax_stage1", 256);
        k.arg_ptr(0, logits_->ptr());
        k.arg_ptr(1, argmax_part_->ptr());
        c.list.launch(k, kArgmaxGroups, 1, 1, ev);
        return;
      }
      case Fam::Argmax2: {   // argmax_stage2(ctrl, part), grid (1)
        l0::Kernel& k = kernel(c, "lfm_argmax_stage2_V128000", "argmax_stage2", 256);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, argmax_part_->ptr());
        c.list.launch(k, 1, 1, 1, ev);
        return;
      }
      default:
        throw std::logic_error("probe_lfm_step: no binding for " + std::string(probe_lfm::fam_name(x.fam)));
    }
  }

 public:
  // --pair: stage A of the two-launch norm (prep_res_fold SP0 at K 2048, Ornith's binary), 81 sites a token.
  std::unique_ptr<Captured> capture_fold(uint32_t n, bool profiled) {
    auto c = std::make_unique<Captured>(Captured{l0::CmdList::regular(ctx_), {}, {}, nullptr, {}});
    if (profiled) {
      c->pool = std::make_unique<l0::EventPool>(ctx_, n);
      c->events.reserve(n);
      for (uint32_t i = 0; i < n; ++i) c->events.emplace_back(*c->pool, i);
    }
    for (uint32_t i = 0; i < n; ++i) {
      l0::Kernel& k = kernel(*c, "prep_res_fold_M1_K2048_SP0_G20", "prep_res_fold", 256);
      k.arg_ptr(0, partials_->ptr());
      k.arg_ptr(1, h_->ptr());
      k.arg_ptr(2, sumsq_->ptr());
      c->list.launch(k, kNormG, 1, 1, profiled ? &c->events[i] : nullptr);
      c->fam.push_back(Fam::Norm);
    }
    c->list.close();
    return c;
  }

 private:
  l0::Context& ctx_;
  const model::Lfm2MoeDesc& d_;
  uint32_t depth_, max_len_;
  l0::CmdList imm_;
  Pool pool_;
  size_t weight_bytes_ = 0;
  std::map<std::string, std::unique_ptr<l0::Module>> modules_;
  std::vector<Layer> layers_;
  std::unique_ptr<l0::Mem> ctrl_, head_, norm_w_, qkn_, partials_, x_, h_, sumsq_, logits_r_, route_, moe_h_, attn_q_,
      attn_part_, attn_out_, logits_, argmax_part_, noop_out_;
};

struct Result {
  std::vector<double> wall_us;                  // plain replays
  std::vector<double> prof_wall_us;             // profiled replays
  std::array<double, size_t(Fam::kCount)> fam_us{};   // Σ in-situ us per family, summed over the profiled replays
  std::array<size_t, size_t(Fam::kCount)> fam_n{};
  size_t prof_replays = 0;
  double sum_us = 0;                            // Σ durations, summed over the profiled replays
};

void replay(Rig& rig, l0::Queue& q, l0::Fence& f, Rig::Captured& plain, Rig::Captured& prof, Result& r, bool keep) {
  rig.set_step();
  Timer t;
  t.start();
  q.execute(plain.list, &f);
  f.wait();
  const double us = t.ms() * 1e3;
  rig.set_step();
  for (l0::Event& e : prof.events) e.reset();
  t.start();
  q.execute(prof.list, &f);
  f.wait();
  const double pus = t.ms() * 1e3;
  if (!keep) return;
  r.wall_us.push_back(us);
  r.prof_wall_us.push_back(pus);
  for (size_t i = 0; i < prof.events.size(); ++i) {
    const double d = prof.events[i].duration_us();
    r.fam_us[size_t(prof.fam[i])] += d;
    r.sum_us += d;
  }
  ++r.prof_replays;
}

void print_list(const char* name, const std::vector<probe_lfm::Launch>& launches, const Result& r, const Rig::Bytes& b,
                const Opts& o) {
  const double n = double(r.prof_replays ? r.prof_replays : 1);
  const double wall = median(r.wall_us), pwall = median(r.prof_wall_us), sum = r.sum_us / n;
  std::array<size_t, size_t(Fam::kCount)> cnt{};
  for (const probe_lfm::Launch& x : launches) ++cnt[size_t(x.fam)];
  std::printf("\n### list `%s`: %zu launches, depth %u (MEASURED: median of %zu plain replays, in-situ means of %zu "
              "profiled replays; %u rounds interleaved)\n\n", name, launches.size(), o.depth, r.wall_us.size(),
              r.prof_replays, o.rounds);
  std::printf("| family | launches | in-situ us / step | us / launch | share of Σ |\n|---|---:|---:|---:|---:|\n");
  for (size_t f = 0; f < size_t(Fam::kCount); ++f) {
    if (!cnt[f]) continue;
    const double us = r.fam_us[f] / n;
    std::printf("| %s | %zu | %.2f | %.3f | %.1f %% |\n", probe_lfm::fam_name(Fam(f)), cnt[f], us, us / double(cnt[f]),
                sum > 0 ? 100.0 * us / sum : 0.0);
  }
  std::printf("\n- Σ durations %.1f us; fence wall %.1f us plain (%.1f us profiled); gap per launch %.3f us plain "
              "((wall - Σ) / launches), %.3f us profiled\n", sum, wall, pwall,
              (wall - sum) / double(launches.size()), (pwall - sum) / double(launches.size()));
  std::printf("- step: %.3f ms plain = %.1f t/s (MEASURED)\n", wall / 1e3, wall > 0 ? 1e6 / wall : 0.0);
  std::printf("- roofline (DERIVED, %.0f GB/s): raw %.1f MB -> %.3f ms; PROPOSED 4-slot / hd64 corrections %.1f MB -> "
              "%.3f ms\n", kMeasuredGBs, b.raw / 1e6, b.raw / kMeasuredGBs / 1e6, b.proposed / 1e6,
              b.proposed / kMeasuredGBs / 1e6);
  const double moe = (r.fam_us[size_t(Fam::GateUp)] + r.fam_us[size_t(Fam::Down)]) / n;
  const double attn = r.fam_us[size_t(Fam::Attn)] / n;
  if (moe > 0 || attn > 0)
    std::printf("- PROPOSED step with the stand-ins corrected (MoE gate||up / down x 4/5: %.1f -> %.1f us; attention "
                "decode x 1/2: %.1f -> %.1f us): %.3f ms\n", moe, moe * 0.8, attn, attn * 0.5,
                (wall - moe * 0.2 - attn * 0.5) / 1e3);
}

}  // namespace

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) usage(std::string(what) + " needs a value");
      return argv[++i];
    };
    if (a == "--depth") o.depth = uint32_t(std::stoul(need("--depth")));
    else if (a == "--rounds") o.rounds = uint32_t(std::stoul(need("--rounds")));
    else if (a == "--reps") o.reps = uint32_t(std::stoul(need("--reps")));
    else if (a == "--list") o.list = need("--list");
    else if (a == "--sweep-s") o.sweep = true;
    else if (a == "--pair") o.pair = true;
    else usage("unknown argument " + a);
  }
  if (o.depth == 0 || o.depth + 64 > model::lfm2moe().trained_max_len) usage("--depth is 1 .. 131008");
  if (o.rounds == 0 || o.reps == 0) usage("--rounds and --reps are >= 1");
  std::vector<List> lists;
  if (o.list == "all") lists = {List::A, List::Fused, List::Gemv, List::Noop};
  else if (o.list == "a") lists = {List::A};
  else if (o.list == "fused") lists = {List::Fused};
  else if (o.list == "gemv") lists = {List::Gemv};
  else if (o.list == "noop") lists = {List::Noop};
  else usage("--list is a, fused, gemv, noop or all");

  const model::Lfm2MoeDesc& d = model::lfm2moe();
  l0::Context ctx(0);
  std::printf("# probe_lfm_step - LFM2.5-24B-A2B's decode step from existing kernels (spec 24c P0)\n\n"
              "device %s; depth %u, rounds %u, %u replays a list a round after a 20-replay warm-up\n",
              ctx.name().c_str(), o.depth, o.rounds, o.reps);

  // The lists' {S, layout} per int4 shape: 24b's descriptor (PROVISIONAL), or the sweep's S picks.
  std::map<Fam, Pick> picks;
  for (const model::LfmLinear& ln : d.linears(0)) {   // layer 0: dense conv (in_proj, out_proj, gate||up, down)
    if (ln.id == model::LfmLinearId::InProj) picks[Fam::InProj] = {ln.shape.S, ln.shape.layout};
    if (ln.id == model::LfmLinearId::OutProj) picks[Fam::OutProj] = {ln.shape.S, ln.shape.layout};
    if (ln.id == model::LfmLinearId::DenseGateUp) picks[Fam::DenseGateUp] = {ln.shape.S, ln.shape.layout};
    if (ln.id == model::LfmLinearId::DenseDown) picks[Fam::DenseDown] = {ln.shape.S, ln.shape.layout};
  }
  for (const model::LfmLinear& ln : d.linears(2))   // layer 2: attention
    if (ln.id == model::LfmLinearId::Qkv) picks[Fam::Qkv] = {ln.shape.S, ln.shape.layout};

  Rig rig(ctx, d, o.depth);
  std::printf("allocated %.2f GB (random, incompressible)\n", rig.weight_bytes() / 1e9);

  if (o.sweep) {
    std::printf("\n## --sweep-s: in-situ us per launch (MEASURED, median of %u profiled replays over >= 512 MB of "
                "distinct weight copies); pick = the smallest S within 3 %% of the shape's best (probe_gemv's rule)\n\n"
                "| shape | K | N | L | S | in-situ us | GB/s | wall us / launch |\n|---|---:|---:|---:|---:|---:|---:|---:|\n",
                o.reps);
    for (const probe_lfm::Int4Shape& s : probe_lfm::int4_shapes(d)) {
      std::array<std::pair<uint32_t, double>, 2> best_l{};
      for (uint32_t L = 0; L < 2; ++L) {
        std::vector<std::pair<uint32_t, double>> row;
        for (uint32_t S : {1u, 2u, 4u, 8u, 16u}) {
          if ((s.K / 64) % S != 0) continue;
          const std::pair<double, double> t = rig.time_gemv(s.K, s.N, S, L, o.reps);
          row.push_back({S, t.first});
          std::printf("| %s | %u | %u | %u | %u | %.3f | %.1f | %.3f |\n", probe_lfm::fam_name(s.fam), s.K, s.N, L, S,
                      t.first, int4_bytes(s.K, s.N) / t.first / 1e3, t.second);
        }
        double best = 1e30;
        for (auto& r : row) best = std::min(best, r.second);
        uint32_t pickS = row.front().first;
        for (auto& r : row)
          if (r.second <= best * 1.03) { pickS = r.first; break; }
        best_l[L] = {pickS, best};
      }
      const uint32_t L = picks.count(s.fam) ? picks[s.fam].L : 1;
      std::printf("| **pick** %s | | | L0: S%u (%.3f us) | L1: S%u (%.3f us) | lists use L%u (24b's) at S%u | | |\n",
                  probe_lfm::fam_name(s.fam), best_l[0].first, best_l[0].second, best_l[1].first, best_l[1].second, L,
                  best_l[L].first);
      picks[s.fam] = {best_l[L].first, L};
    }
  }
  std::printf("\nint4 GEMV cells the lists bind (%s): in_proj S%u L%u, out_proj / o_proj S%u L%u, q||k||v S%u L%u, "
              "dense gate||up S%u L%u, dense down S%u L%u\n", o.sweep ? "the sweep's S, 24b's layout" : "24b's descriptor, PROVISIONAL",
              picks[Fam::InProj].S, picks[Fam::InProj].L, picks[Fam::OutProj].S, picks[Fam::OutProj].L,
              picks[Fam::Qkv].S, picks[Fam::Qkv].L, picks[Fam::DenseGateUp].S, picks[Fam::DenseGateUp].L,
              picks[Fam::DenseDown].S, picks[Fam::DenseDown].L);

  struct Run {
    List which;
    std::vector<probe_lfm::Launch> launches;
    std::unique_ptr<Rig::Captured> plain, prof;
    Result r;
  };
  std::vector<Run> runs;
  for (List l : lists) {
    Run r;
    r.which = l;
    r.launches = probe_lfm::build_list(d, l);
    r.plain = rig.capture(r.launches, false, picks);
    r.prof = rig.capture(r.launches, true, picks);
    runs.push_back(std::move(r));
  }
  l0::Queue q(ctx);
  l0::Fence f(q);
  for (Run& r : runs)
    for (int i = 0; i < 20; ++i) replay(rig, q, f, *r.plain, *r.prof, r.r, false);
  for (uint32_t round = 0; round < o.rounds; ++round)
    for (size_t k = 0; k < runs.size(); ++k) {
      Run& r = runs[(k + round) % runs.size()];   // rotated per round
      for (uint32_t i = 0; i < o.reps; ++i) replay(rig, q, f, *r.plain, *r.prof, r.r, true);
    }
  for (Run& r : runs)
    print_list(probe_lfm::list_name(r.which), r.launches, r.r, rig.bytes(r.launches), o);

  if (runs.size() > 1) {
    std::printf("\n## the step per list (MEASURED, median plain fence wall)\n\n| list | launches | ms | t/s | vs a |\n"
                "|---|---:|---:|---:|---:|\n");
    const double a = median(runs[0].r.wall_us);
    for (Run& r : runs) {
      const double w = median(r.r.wall_us);
      std::printf("| %s | %zu | %.3f | %.1f | %+.2f %% |\n", probe_lfm::list_name(r.which), r.launches.size(), w / 1e3,
                  w > 0 ? 1e6 / w : 0.0, a > 0 ? 100.0 * (w - a) / a : 0.0);
    }
  }
  if (o.pair) {
    auto plain = rig.capture_fold(81, false), prof = rig.capture_fold(81, true);
    Result r;
    for (uint32_t i = 0; i < 20 + o.rounds * o.reps; ++i) replay(rig, q, f, *plain, *prof, r, i >= 20);
    std::printf("\n## --pair: prep_res_fold SP0 at K 2048 G 20 (stage A of the two-launch norm), 81 sites: %.3f us a "
                "launch in situ, %.1f us a token (MEASURED) - what the two-launch fallback would add to `a`\n",
                r.sum_us / double(r.prof_replays) / 81.0, r.sum_us / double(r.prof_replays));
  }
  std::printf("\nDecision rule (spec 24 decision 6, PROPOSED): every arm whose skeleton saving (`a` - `fused`, split by "
              "the removed families' in-situ us + gaps) is >= 1 %% of the step is built and A/B'd in plan 24c Task 8; "
              "the S picks become kernels::lfm2moe's table (PROVISIONAL until then).\n");
  return 0;
}

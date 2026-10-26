// probe_bolt_step - spec 25c P0 (plan 25c Task 1): where a Nemotron 3.5 Bolt decode step's time goes, BEFORE any
// nb_* kernel is bound. The step's launch list (tools/probe/bolt_step_shapes.h: 653 launches, 533 with every fused arm,
// a GEMV-only list, 653 noops, the SSM state's read-modify-write floor) is bound to EXISTING binaries at Bolt's shapes
// and replayed on ONE card, the lists interleaved:
//
//   int4 GEMVs   gemv.cl at K 4096 N 18560 (in_proj) / 4608 (q||k||v) / 4096 (o_proj) / 6400 (fc1||shared_up),
//                K 8192 N 4096 (out_proj), K 1024 N 4096 (fc2), K 5376 N 4096 (shared_down) - the engine's own binaries
//                (gemv_M1_K..._N..._S<s>_L<l>), at model::BoltDesc::linears' {S, layout} (25b's PROVISIONAL cells)
//                or, with --sweep-s, at the sweep's S picks; every dense linear its own random buffer
//   routers      gemv_bf16_M1_K4096_N<E>_C16_S16 at each MoE block's own E (240 .. 496; fp32 logits)
//   experts      moe.cl at EXPERTS 64 TOP_K 8 HIDDEN 1024 INTER 2048 (moe_M1_E64_T8_D1024_I2048): its gate||up and
//                down over a pool of 64 distinct expert blocks a MoE block + the shared slot (block 64), the ids from a
//                host-written route row (8 distinct random ids a block, + the shared slot: 9 distinct blocks a step,
//                none re-read inside a step). moe.cl is GATED (gate||up reads twice the up bytes) at I 2048 where Bolt's
//                experts are relu2 up-only at I 1536..2560: the table prints the raw time and the PROPOSED correction
//                (each block's Bolt bytes 9 x 2 x I x 544 over moe.cl's 9 x 3,342,336 - a byte scale, x 0.75 at I 2304)
//   SSM step     gdn_step_M1 at Qwen3.8's shape (48 x 128 x 128 fp32 = 3 MiB of state read and written a block,
//                against Bolt's 4 MiB): raw and x 4/3 PROPOSED; its own state buffer a block
//   attention    spec 23c's mg_attn at Q 32 KV 2 head_dim 128 full (NoPE) - Bolt's EXACT shape (Decision 15:
//                the engine binds these binaries): mg_attn_decode + mg_attn_reduce with a ones gate
//   norm         lfm_res_norm at K 4096 over 8 work-groups at split-K 1 (lfm_prep.cl - the one-launch norm site the
//                nb_res_norm family copies: every work-group folds the row and recomputes Σ), ONE launch a site;
//                --pair times the two-launch form (prep_res_fold_M1_K4096_SP0_G16 + kol_norm_M1_K4096_G16_W16)
//   head         gemv_i8w_M1_K4096_N131072, argmax.cl at VOCAB 131072 (the engine's nb_argmax_* binaries)
//   conv, gnorm, route, sh_act, attn_prep, embed   noop (src/kernels/noop.cl)
//   state        nb_state_rmw (src/kernels/bolt/nb_probe.cl): 4 MiB fp32 a Mamba block read and written in place;
//                state16: 2 MiB (bf16 - Decision 2's arm)
//
// Weights are random (incompressible: probe_gemv's rule - the B70 compresses device memory losslessly), one allocation
// per linear, expert pool and state as the engine has them; the KV is `depth` + 64 positions per attention block.
// Each list is captured twice - plain (the fence wall) and profiled (a kernel-timestamp event per launch, docs/15's
// instrument) - and replayed `--rounds` rounds after a 20-replay warm-up, the lists rotated per round (interleaved:
// the clock state moves, spec 25c Global Constraints). Before every replay the host rewrites Control::pos to `depth`
// (argmax stage 2 advances it), so every replay is the same step.
//
//   probe_bolt_step [--depth D] [--rounds R] [--reps N] [--list a|fused|gemv|noop|state|all] [--sweep-s] [--pair]
//
// Output: one markdown table per list - per launch family the in-situ us (mean per replay), launches, Σ durations,
// the plain fence wall, the gap per launch ((wall - Σ) / launches); the whole step per list; the derived roofline
// (the list's bytes / 590 GB/s), raw and with the PROPOSED stand-in corrections. --sweep-s: every int4 shape at S in
// {1, 2, 4, 8, 16} (S | K / 64) and layouts 0 / 1, in-situ us per launch over >= 512 MB of distinct weight copies;
// the pick per shape and layout is probe_gemv's rule (the smallest S within 3 % of the best).
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bolt_step_shapes.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "model/bolt.h"
#include "timer.h"

namespace {

using probe_bolt::Fam;
using probe_bolt::List;

constexpr double kMeasuredGBs = 590.0;   // probe_bw through this launch path (spec 25 §3)
constexpr uint32_t kHidden = 4096, kVocab = 131072;
// moe.cl's stand-in shape (moe_M1_E64_T8_D1024_I2048): 64 pool blocks + the shared one, 8 routed + the shared slot.
constexpr uint32_t kPool = 64, kMoeK = 8, kSlots = kMoeK + 1, kMoeHidden = 1024, kMoeInter = 2048;
constexpr uint32_t kMoeUpKs = 4, kMoeDnKs = 2, kMoeRw = 32;
constexpr size_t kTile = 544;   // a layout-1 tile: 128 u32 of nibbles + 8 u32 of scales
inline size_t int4_bytes(uint32_t K, uint32_t N) { return size_t(N / 16) * (K / 64) * kTile; }
constexpr size_t kGuBlock = size_t(2 * kMoeInter / 16) * (kMoeHidden / 64) * kTile;   // 2,228,224
constexpr size_t kDnBlock = size_t(kMoeHidden / 16) * (kMoeInter / 64) * kTile;       // 1,114,112
// spec 23c's mg_attn at Bolt's shape (mg_attn_M1_T64_Q32KV2_F).
constexpr uint32_t kQHeads = 32, kKvHeads = 2, kHd = 128, kAttnTgt = 64, kAttnPart = kHd + 2;
// gdn_step_M1 at Qwen3.8's shape: 48 v-heads x 128 x 128 fp32 (3 MiB), its buffers (gdn_step.cl's header).
constexpr uint32_t kGdnHeads = 48, kGdnDim = 128, kGdnConv = 10240, kGdnQkvz = 16384, kGdnSmallFloats = 41008 + 48;
constexpr size_t kGdnState = size_t(kGdnHeads) * kGdnDim * kGdnDim * 4;   // 3,145,728
constexpr size_t kBoltState = size_t(128) * 64 * 128 * 4;                  // 4,194,304 (spec 25 §1.2)
constexpr uint32_t kNormWgs = 8, kFoldG = 16, kArgmaxGroups = (kVocab + 1023) / 1024;

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
  std::fprintf(stderr, "probe_bolt_step: %s\n  probe_bolt_step [--depth D] [--rounds R] [--reps N] "
               "[--list a|fused|gemv|noop|state|all] [--sweep-s] [--pair]\n", why.c_str());
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
  Rig(l0::Context& ctx, const model::BoltDesc& d, uint32_t depth)
      : ctx_(ctx), d_(d), depth_(depth), max_len_(depth + 64), imm_(l0::CmdList::immediate(ctx)) {
    ctrl_ = mem(l0::MemKind::Shared, 256, false);
    std::memset(ctrl_->ptr(), 0, 256);
    std::mt19937 rng(25);
    for (uint32_t l = 0; l < d.layers; ++l) {
      Layer L;
      if (d.is_mamba(l)) {
        L.in = mem(int4_bytes(d.hidden, d.m_inproj));
        L.out = mem(int4_bytes(d.m_inner, d.hidden));
        L.gdn_state = mem(kGdnState);
        L.gdn_ring = mem(size_t(16) * kGdnConv * 2);
        L.state = mem(kBoltState);
      } else if (d.is_attention(l)) {
        L.qkv = mem(int4_bytes(d.hidden, d.qkv_n()));
        L.o = mem(int4_bytes(d.q_n(), d.hidden));
        L.kv_k = mem(size_t(max_len_) * kKvHeads * kHd * 2);
        L.kv_v = mem(size_t(max_len_) * kKvHeads * kHd * 2);
      } else {
        L.E = d.experts.at(l);
        L.I = d.inter.at(l);
        L.router = mem(size_t(d.hidden) * L.E * 2);
        L.fc1 = mem(int4_bytes(d.hidden, d.latent + d.shared_inter));
        L.fc2 = mem(int4_bytes(d.latent, d.hidden));
        L.shd = mem(int4_bytes(d.shared_inter, d.hidden));
        L.egu = mem((kPool + 1) * kGuBlock);   // moe.cl's block EXPERTS (64) is the shared slot
        L.edn = mem((kPool + 1) * kDnBlock);
        // The route row (moe.cl R_*): 8 distinct random pool ids, w 1/8 each (bf16 values), the shared gate 0.5.
        std::vector<uint32_t> row(kMoeRw, 0u), ids(kPool);
        for (uint32_t i = 0; i < kPool; ++i) ids[i] = i;
        std::shuffle(ids.begin(), ids.end(), rng);
        for (uint32_t k = 0; k < kMoeK; ++k) {
          row[k] = ids[k];
          const float w = 0.125f;
          std::memcpy(&row[8 + k], &w, 4);
        }
        const float sg = 0.5f;
        std::memcpy(&row[16], &sg, 4);
        L.route = mem(kMoeRw * 4, false);
        imm_.copy(L.route->ptr(), row.data(), kMoeRw * 4);
      }
      layers_.push_back(std::move(L));
    }
    head_ = mem(size_t(kHidden) * kVocab + size_t(kVocab) * 4);
    // Small scratch, not counted as weights. The activations are random words (any bf16 bit pattern: the GEMVs'
    // time does not depend on the values).
    norm_w_ = mem(kHidden * 4, false);
    partials_ = mem(size_t(16) * d.m_inproj * 4, false);   // the widest S x N the sweep can pick
    act_ = mem(size_t(d.m_inner) * 2, false);              // every GEMV input (K <= 8192)
    x_ = mem(kHidden * 2, false);
    h_a_ = mem(kHidden * 2, false);
    h_b_ = mem(kHidden * 2, false);
    logits_r_ = mem(512 * 4, false);
    moe_h_ = mem(size_t(kSlots) * kMoeInter * 2, false);
    moe_out_ = mem(kMoeHidden * 2, false);
    gdn_qkvz_ = mem(size_t(kGdnQkvz) * 4, false);
    gdn_ab_ = mem(128 * 4, false);
    gdn_small_ = mem(size_t(kGdnSmallFloats) * 4, false);
    gdn_o_ = mem(size_t(kGdnHeads) * kGdnDim * 4, false);
    attn_q_ = mem(size_t(kQHeads) * kHd * 4, false);
    attn_part_ = mem(size_t(kQHeads) * kAttnTgt * kAttnPart * 4, false);
    attn_out_ = mem(size_t(kQHeads) * kHd * 2, false);
    ones_ = mem(size_t(kQHeads) * kHd * 2, false);
    imm_.fill(ones_->ptr(), 0x3F803F80u, ones_->size());   // bf16 1.0: Decision 15's gate
    // Finite small inputs for the attention and the SSM stand-ins (the q row, the qkv||z partials, the small block),
    // so no NaN reaches an exp: 0x3C003C00 = bf16 / f16 pairs of 2^-7, read as fp32 = 7.8e-3.
    imm_.fill(attn_q_->ptr(), 0x3C003C00u, attn_q_->size());
    imm_.fill(gdn_qkvz_->ptr(), 0x3C003C00u, gdn_qkvz_->size());
    imm_.fill(gdn_ab_->ptr(), 0x3C003C00u, gdn_ab_->size());
    imm_.fill(gdn_small_->ptr(), 0x3C003C00u, gdn_small_->size());
    logits_ = mem(size_t(kVocab) * 4, false);
    argmax_part_ = mem(size_t(kArgmaxGroups) * 2 * 4, false);
    sumsq_ = mem(kFoldG * 4, false);
    noop_out_ = mem(4096, false);
    set_step();
  }

  void set_step() {
    uint32_t* c = static_cast<uint32_t*>(ctrl_->ptr());
    c[0] = depth_;   // pos
    c[1] = 1;        // n_active
    c[2] = 1000;     // cur_token[0]
  }
  size_t weight_bytes() const { return weight_bytes_; }

  // The list as one closed command list (and, profiled, one event per launch).
  struct Captured {
    l0::CmdList list;
    std::vector<std::unique_ptr<l0::Kernel>> kernels;
    std::vector<Fam> fam;
    std::vector<int> layer;
    std::unique_ptr<l0::EventPool> pool;
    std::vector<l0::Event> events;
  };
  std::unique_ptr<Captured> capture(const std::vector<probe_bolt::Launch>& launches, bool profiled,
                                    const std::map<Fam, Pick>& picks, bool state_bf16) {
    auto c = std::make_unique<Captured>(Captured{l0::CmdList::regular(ctx_), {}, {}, {}, nullptr, {}});
    if (profiled) {
      c->pool = std::make_unique<l0::EventPool>(ctx_, uint32_t(launches.size()));
      c->events.reserve(launches.size());
      for (uint32_t i = 0; i < launches.size(); ++i) c->events.emplace_back(*c->pool, i);
    }
    for (size_t i = 0; i < launches.size(); ++i) {
      bind(*c, launches[i], picks, state_bf16, profiled ? &c->events[i] : nullptr);
      c->fam.push_back(launches[i].fam);
      c->layer.push_back(launches[i].layer);
    }
    c->list.close();
    return c;
  }

  // Bytes a list streams per replay (derived): weights, the KV it reads at `depth`, the head, the states.
  struct Bytes {
    double raw = 0, proposed = 0;   // proposed: the experts at Bolt's bytes, the SSM state at 4 MiB
  };
  Bytes bytes(const std::vector<probe_bolt::Launch>& launches, bool state_bf16) const {
    Bytes b;
    for (const probe_bolt::Launch& x : launches) {
      double r = 0, p = -1;
      const Layer* L = x.layer >= 0 ? &layers_[size_t(x.layer)] : nullptr;
      switch (x.fam) {
        case Fam::InProj: r = int4_bytes(d_.hidden, d_.m_inproj); break;
        case Fam::OutProj: r = int4_bytes(d_.m_inner, d_.hidden); break;
        case Fam::Qkv: r = int4_bytes(d_.hidden, d_.qkv_n()); break;
        case Fam::OProj: r = int4_bytes(d_.q_n(), d_.hidden); break;
        case Fam::Fc1Up: r = int4_bytes(d_.hidden, d_.latent + d_.shared_inter); break;
        case Fam::Fc2: r = int4_bytes(d_.latent, d_.hidden); break;
        case Fam::ShDown: r = int4_bytes(d_.shared_inter, d_.hidden); break;
        case Fam::Router: r = double(d_.hidden) * L->E * 2; break;
        case Fam::MoeUp: r = double(kSlots) * kGuBlock; p = double(d_.top_k) * L->I * kTile; break;
        case Fam::MoeDown: r = double(kSlots) * kDnBlock; p = double(d_.top_k) * L->I * kTile; break;
        case Fam::Ssm: r = 2.0 * kGdnState; p = 2.0 * kBoltState; break;
        case Fam::Attn: r = double(depth_ + 1) * kKvHeads * kHd * 2 * 2; break;
        case Fam::Head: r = double(kHidden) * kVocab + double(kVocab) * 4; break;
        case Fam::State: r = 2.0 * kBoltState / (state_bf16 ? 2 : 1); break;
        default: break;
      }
      b.raw += r;
      b.proposed += p < 0 ? r : p;
    }
    return b;
  }
  // The PROPOSED byte scale of a stand-in launch (the experts: Bolt's 2 x I x 544 a slot over moe.cl's gated block;
  // the SSM: 4 MiB over 3 MiB), 1 elsewhere.
  double scale(Fam f, int layer) const {
    if (f == Fam::MoeUp) return double(d_.top_k) * layers_[size_t(layer)].I * kTile / (double(kSlots) * kGuBlock);
    if (f == Fam::MoeDown) return double(d_.top_k) * layers_[size_t(layer)].I * kTile / (double(kSlots) * kDnBlock);
    if (f == Fam::Ssm) return double(kBoltState) / double(kGdnState);
    return 1.0;
  }

  // --sweep-s: one shape at {S, L}, over distinct weight copies (>= 512 MB) - in-situ us per launch (median of the
  // profiled replays' per-launch means) and the plain wall per launch.
  std::pair<double, double> time_gemv(uint32_t K, uint32_t N, uint32_t S, uint32_t L, uint32_t reps) {
    const size_t bytes = int4_bytes(K, N);
    const size_t copies = std::max<size_t>(8, (size_t(512) << 20) / bytes);
    std::vector<std::unique_ptr<l0::Mem>> w;
    for (size_t i = 0; i < copies; ++i) w.push_back(mem(bytes, false));
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
        k.arg_ptr(2, act_->ptr());
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
    std::unique_ptr<l0::Mem> in, out, qkv, o, fc1, fc2, shd, router, egu, edn, route, kv_k, kv_v, gdn_state, gdn_ring,
        state;
    uint32_t E = 0, I = 0;
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

  void gemv(Captured& c, const l0::Mem& w, uint32_t K, uint32_t N, Pick p, l0::Event* ev) {
    l0::Kernel& k = kernel(c, kernels::gemv_variant(1, K, N, p.S, p.L), "gemv", 64);
    k.arg_ptr(0, w.ptr());
    k.arg_ptr(1, static_cast<const uint8_t*>(w.ptr()) + size_t(K / 8) * N * 4);   // layout 0's scales
    k.arg_ptr(2, act_->ptr());
    k.arg_ptr(3, partials_->ptr());
    c.list.launch(k, N / 64, p.S, 1, ev);
  }

  void bind(Captured& c, const probe_bolt::Launch& x, const std::map<Fam, Pick>& picks, bool state_bf16,
            l0::Event* ev) {
    const Layer* L = x.layer >= 0 ? &layers_[size_t(x.layer)] : nullptr;
    const auto pick = [&](Fam f) { return picks.at(f); };
    switch (x.fam) {
      case Fam::Embed: case Fam::Conv: case Fam::Gnorm: case Fam::Route: case Fam::ShAct: case Fam::AttnPrep:
      case Fam::Noop: {
        l0::Kernel& k = kernel(c, "noop", "noop", 16);
        k.arg_ptr(0, noop_out_->ptr());
        c.list.launch(k, 1, 1, 1, ev);
        return;
      }
      case Fam::Norm: {   // lfm_res_norm(h_in, slices, w, h_out, x) at S 1, grid (8, 1)
        l0::Kernel& k = kernel(c, "lfm_res_norm_M1_K4096_W8_S1", "lfm_res_norm", 256);
        k.arg_ptr(0, h_a_->ptr());
        k.arg_ptr(1, partials_->ptr());
        k.arg_ptr(2, norm_w_->ptr());
        k.arg_ptr(3, h_b_->ptr());
        k.arg_ptr(4, x_->ptr());
        c.list.launch(k, kNormWgs, 1, 1, ev);
        return;
      }
      case Fam::InProj: gemv(c, *L->in, d_.hidden, d_.m_inproj, pick(x.fam), ev); return;
      case Fam::OutProj: gemv(c, *L->out, d_.m_inner, d_.hidden, pick(x.fam), ev); return;
      case Fam::Qkv: gemv(c, *L->qkv, d_.hidden, d_.qkv_n(), pick(x.fam), ev); return;
      case Fam::OProj: gemv(c, *L->o, d_.q_n(), d_.hidden, pick(x.fam), ev); return;
      case Fam::Fc1Up: gemv(c, *L->fc1, d_.hidden, d_.latent + d_.shared_inter, pick(x.fam), ev); return;
      case Fam::Fc2: gemv(c, *L->fc2, d_.latent, d_.hidden, pick(x.fam), ev); return;
      case Fam::ShDown: gemv(c, *L->shd, d_.shared_inter, d_.hidden, pick(x.fam), ev); return;
      case Fam::Ssm: {   // gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o), grid (48, 4)
        l0::Kernel& k = kernel(c, "gdn_step_M1", "gdn_step", 256);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, gdn_qkvz_->ptr());
        k.arg_ptr(2, gdn_ab_->ptr());
        k.arg_ptr(3, gdn_small_->ptr());
        k.arg_ptr(4, L->gdn_ring->ptr());
        k.arg_ptr(5, L->gdn_state->ptr());
        k.arg_ptr(6, gdn_o_->ptr());
        c.list.launch(k, kGdnHeads, 4, 1, ev);
        return;
      }
      case Fam::Attn: {   // mg_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part), grid (KV, TGT)
        l0::Kernel& k = kernel(c, "mg_attn_M1_T64_Q32KV2_F", "mg_attn_decode", kHd);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, attn_q_->ptr());
        k.arg_ptr(2, L->kv_k->ptr());
        k.arg_ptr(3, L->kv_v->ptr());
        k.arg_ptr(4, attn_part_->ptr());
        c.list.launch(k, kKvHeads, kAttnTgt, 1, ev);
        return;
      }
      case Fam::AttnReduce: {   // mg_attn_reduce(ctrl, attn_part, gate, attn_out), grid (Q, M) - the ones gate
        l0::Kernel& k = kernel(c, "mg_attn_M1_T64_Q32KV2_F", "mg_attn_reduce", kHd);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, attn_part_->ptr());
        k.arg_ptr(2, ones_->ptr());
        k.arg_ptr(3, attn_out_->ptr());
        c.list.launch(k, kQHeads, 1, 1, ev);
        return;
      }
      case Fam::Router: {   // gemv_bf16(w, x, out), tiling {16, 16}, grid E / 16
        l0::Kernel& k = kernel(c, "gemv_bf16_M1_K4096_N" + std::to_string(L->E) + "_C16_S16", "gemv_bf16", 16 * 16);
        k.arg_ptr(0, L->router->ptr());
        k.arg_ptr(1, x_->ptr());
        k.arg_ptr(2, logits_r_->ptr());
        c.list.launch(k, L->E / 16, 1, 1, ev);
        return;
      }
      case Fam::MoeUp: {   // moe_gate_up(route, x, w_gu, h), grid (SLOTS x NT_UP / 4, M), WG 64 x UP_KS
        l0::Kernel& k = kernel(c, "moe_M1_E64_T8_D1024_I2048", "moe_gate_up", 64 * kMoeUpKs);
        k.arg_ptr(0, L->route->ptr());
        k.arg_ptr(1, act_->ptr());
        k.arg_ptr(2, L->egu->ptr());
        k.arg_ptr(3, moe_h_->ptr());
        c.list.launch(k, kSlots * (2 * kMoeInter / 16 / 4), 1, 1, ev);
        return;
      }
      case Fam::MoeDown: {   // moe_down(route, h, w_dn, resid), grid (HIDDEN / 16, M), WG 16 x SLOTS x DN_KS
        l0::Kernel& k = kernel(c, "moe_M1_E64_T8_D1024_I2048", "moe_down", 16 * kSlots * kMoeDnKs);
        k.arg_ptr(0, L->route->ptr());
        k.arg_ptr(1, moe_h_->ptr());
        k.arg_ptr(2, L->edn->ptr());
        k.arg_ptr(3, moe_out_->ptr());
        c.list.launch(k, kMoeHidden / 16, 1, 1, ev);
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
        l0::Kernel& k = kernel(c, "nb_argmax_stage1_M1_V131072", "argmax_stage1", 256);
        k.arg_ptr(0, logits_->ptr());
        k.arg_ptr(1, argmax_part_->ptr());
        c.list.launch(k, kArgmaxGroups, 1, 1, ev);
        return;
      }
      case Fam::Argmax2: {   // argmax_stage2(ctrl, part), grid (1)
        l0::Kernel& k = kernel(c, "nb_argmax_stage2_V131072", "argmax_stage2", 256);
        k.arg_ptr(0, ctrl_->ptr());
        k.arg_ptr(1, argmax_part_->ptr());
        c.list.launch(k, 1, 1, 1, ev);
        return;
      }
      case Fam::State: {   // nb_state_rmw(state, n4), grid (n4 / 256)
        const uint32_t n4 = uint32_t(kBoltState / (state_bf16 ? 2 : 1) / (state_bf16 ? 8 : 16));
        l0::Kernel& k = kernel(c, state_bf16 ? "nb_state_rmw_B16" : "nb_state_rmw_F32", "nb_state_rmw", 256);
        k.arg_ptr(0, L->state->ptr());
        k.arg<uint32_t>(1, n4);
        c.list.launch(k, (n4 + 255) / 256, 1, 1, ev);
        return;
      }
      default:
        throw std::logic_error("probe_bolt_step: no binding for " + std::string(probe_bolt::fam_name(x.fam)));
    }
  }

 public:
  // --pair: the two-launch norm (prep_res_fold SP0 at K 4096, G 16 + kol_norm_finish over 16 work-groups), 89 sites a
  // token (88 blocks' input norms + norm_f).
  std::unique_ptr<Captured> capture_pair(uint32_t n, bool profiled) {
    auto c = std::make_unique<Captured>(Captured{l0::CmdList::regular(ctx_), {}, {}, {}, nullptr, {}});
    if (profiled) {
      c->pool = std::make_unique<l0::EventPool>(ctx_, 2 * n);
      c->events.reserve(2 * n);
      for (uint32_t i = 0; i < 2 * n; ++i) c->events.emplace_back(*c->pool, i);
    }
    for (uint32_t i = 0; i < n; ++i) {
      {
        l0::Kernel& k = kernel(*c, "prep_res_fold_M1_K4096_SP0_G16", "prep_res_fold", 256);
        k.arg_ptr(0, partials_->ptr());
        k.arg_ptr(1, h_a_->ptr());
        k.arg_ptr(2, sumsq_->ptr());
        c->list.launch(k, kFoldG, 1, 1, profiled ? &c->events[2 * i] : nullptr);
      }
      {
        l0::Kernel& k = kernel(*c, "kol_norm_M1_K4096_G16_W16", "kol_norm_finish", 256);
        k.arg_ptr(0, sumsq_->ptr());
        k.arg_ptr(1, h_a_->ptr());
        k.arg_ptr(2, norm_w_->ptr());
        k.arg_ptr(3, x_->ptr());
        c->list.launch(k, kFoldG, 1, 1, profiled ? &c->events[2 * i + 1] : nullptr);
      }
      c->fam.push_back(Fam::Norm);
      c->fam.push_back(Fam::Norm);
      c->layer.push_back(-1);
      c->layer.push_back(-1);
    }
    c->list.close();
    return c;
  }

 private:
  l0::Context& ctx_;
  const model::BoltDesc& d_;
  uint32_t depth_, max_len_;
  l0::CmdList imm_;
  Pool pool_;
  size_t weight_bytes_ = 0;
  std::map<std::string, std::unique_ptr<l0::Module>> modules_;
  std::vector<Layer> layers_;
  std::unique_ptr<l0::Mem> ctrl_, head_, norm_w_, partials_, act_, x_, h_a_, h_b_, logits_r_, moe_h_, moe_out_,
      gdn_qkvz_, gdn_ab_, gdn_small_, gdn_o_, attn_q_, attn_part_, attn_out_, ones_, logits_, argmax_part_, sumsq_,
      noop_out_;
};

struct Result {
  std::vector<double> wall_us;                        // plain replays
  std::vector<double> prof_wall_us;                   // profiled replays
  std::array<double, size_t(Fam::kCount)> fam_us{};   // Σ in-situ us per family, summed over the profiled replays
  double scaled_us = 0;                               // Σ in-situ us x each launch's PROPOSED byte scale
  size_t prof_replays = 0;
  double sum_us = 0;                                  // Σ durations, summed over the profiled replays
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
    r.scaled_us += d * rig.scale(prof.fam[i], prof.layer[i]);
    r.sum_us += d;
  }
  ++r.prof_replays;
}

void print_list(const char* name, const std::vector<probe_bolt::Launch>& launches, const Result& r, const Rig::Bytes& b,
                const Opts& o) {
  const double n = double(r.prof_replays ? r.prof_replays : 1);
  const double wall = median(r.wall_us), pwall = median(r.prof_wall_us), sum = r.sum_us / n;
  std::array<size_t, size_t(Fam::kCount)> cnt{};
  for (const probe_bolt::Launch& x : launches) ++cnt[size_t(x.fam)];
  std::printf("\n### list `%s`: %zu launches, depth %u (MEASURED: median of %zu plain replays, in-situ means of %zu "
              "profiled replays; %u rounds interleaved)\n\n", name, launches.size(), o.depth, r.wall_us.size(),
              r.prof_replays, o.rounds);
  std::printf("| family | launches | in-situ us / step | us / launch | share of Σ |\n|---|---:|---:|---:|---:|\n");
  for (size_t f = 0; f < size_t(Fam::kCount); ++f) {
    if (!cnt[f]) continue;
    const double us = r.fam_us[f] / n;
    std::printf("| %s | %zu | %.2f | %.3f | %.1f %% |\n", probe_bolt::fam_name(Fam(f)), cnt[f], us, us / double(cnt[f]),
                sum > 0 ? 100.0 * us / sum : 0.0);
  }
  std::printf("\n- Σ durations %.1f us; fence wall %.1f us plain (%.1f us profiled); gap per launch %.3f us plain "
              "((wall - Σ) / launches), %.3f us profiled\n", sum, wall, pwall,
              (wall - sum) / double(launches.size()), (pwall - sum) / double(launches.size()));
  std::printf("- step: %.3f ms plain = %.1f t/s (MEASURED)\n", wall / 1e3, wall > 0 ? 1e6 / wall : 0.0);
  std::printf("- roofline (DERIVED, %.0f GB/s): raw %.1f MB -> %.3f ms; with the stand-ins at Bolt's bytes (PROPOSED) "
              "%.1f MB -> %.3f ms\n", kMeasuredGBs, b.raw / 1e6, b.raw / kMeasuredGBs / 1e6, b.proposed / 1e6,
              b.proposed / kMeasuredGBs / 1e6);
  const double corr = r.scaled_us / n - sum;   // the in-situ change the byte scales make
  if (corr != 0.0)
    std::printf("- PROPOSED step with the stand-ins scaled to Bolt's bytes (moe.cl's gated I 2048 blocks -> Bolt's "
                "relu2 up-only I_l blocks, gdn_step's 3 MiB state -> 4 MiB; per launch in situ x the byte ratio): "
                "Σ %.1f -> %.1f us, step %.3f -> %.3f ms\n", sum, sum + corr, wall / 1e3, (wall + corr) / 1e3);
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
  const model::BoltDesc& d = model::bolt();
  if (o.depth == 0 || o.depth + 64 > d.max_pos) usage("--depth is 1 .. " + std::to_string(d.max_pos - 64));
  if (o.rounds == 0 || o.reps == 0) usage("--rounds and --reps are >= 1");
  struct Want {
    List which;
    bool bf16;
    const char* name;
  };
  // Built by returning, not by assigning an initializer list: g++ 15 at -O2 reports a false -Wnonnull
  // in vector::assign's copy (memmove from a possibly-empty source).
  const std::vector<Want> lists = [&]() -> std::vector<Want> {
    if (o.list == "all")
      return {{List::A, false, "a"}, {List::Fused, false, "fused"}, {List::Gemv, false, "gemv"},
              {List::Noop, false, "noop"}, {List::State, false, "state"}, {List::State, true, "state16"}};
    if (o.list == "a") return {{List::A, false, "a"}};
    if (o.list == "fused") return {{List::Fused, false, "fused"}};
    if (o.list == "gemv") return {{List::Gemv, false, "gemv"}};
    if (o.list == "noop") return {{List::Noop, false, "noop"}};
    if (o.list == "state") return {{List::State, false, "state"}, {List::State, true, "state16"}};
    usage("--list is a, fused, gemv, noop, state or all");
    return {};
  }();

  l0::Context ctx(0);
  std::printf("# probe_bolt_step - Nemotron 3.5 Bolt's decode step from existing kernels (spec 25c P0)\n\n"
              "device %s; depth %u, rounds %u, %u replays a list a round after a 20-replay warm-up\n",
              ctx.name().c_str(), o.depth, o.rounds, o.reps);

  // The lists' {S, layout} per int4 shape: 25b's descriptor (PROVISIONAL), or the sweep's S picks.
  std::map<Fam, Pick> picks;
  for (const probe_bolt::Int4Shape& s : probe_bolt::int4_shapes(d))
    for (uint32_t l = 0; l < d.layers && !picks.count(s.fam); ++l)
      for (const model::BoltLinear& ln : d.linears(l))
        if (ln.id == probe_bolt::linear_of(s.fam)) picks[s.fam] = {ln.shape.S, ln.shape.layout};

  Rig rig(ctx, d, o.depth);
  std::printf("allocated %.2f GB of weights / pools / states (random, incompressible)\n", rig.weight_bytes() / 1e9);

  if (o.sweep) {
    std::printf("\n## --sweep-s: in-situ us per launch (MEASURED, median of %u profiled replays over >= 512 MB of "
                "distinct weight copies); pick = the smallest S within 3 %% of the shape's best (probe_gemv's rule)\n\n"
                "| shape | K | N | L | S | in-situ us | GB/s | wall us / launch |\n|---|---:|---:|---:|---:|---:|---:|---:|\n",
                o.reps);
    for (const probe_bolt::Int4Shape& s : probe_bolt::int4_shapes(d)) {
      std::array<std::pair<uint32_t, double>, 2> best_l{};
      for (uint32_t L = 0; L < 2; ++L) {
        std::vector<std::pair<uint32_t, double>> row;
        for (uint32_t S : {1u, 2u, 4u, 8u, 16u}) {
          if ((s.K / 64) % S != 0) continue;
          const std::pair<double, double> t = rig.time_gemv(s.K, s.N, S, L, o.reps);
          row.push_back({S, t.first});
          std::printf("| %s | %u | %u | %u | %u | %.3f | %.1f | %.3f |\n", probe_bolt::fam_name(s.fam), s.K, s.N, L, S,
                      t.first, int4_bytes(s.K, s.N) / t.first / 1e3, t.second);
        }
        double best = 1e30;
        for (auto& r : row) best = std::min(best, r.second);
        uint32_t pickS = row.front().first;
        for (auto& r : row)
          if (r.second <= best * 1.03) {
            pickS = r.first;
            break;
          }
        best_l[L] = {pickS, best};
      }
      const uint32_t L = picks.count(s.fam) ? picks[s.fam].L : 1;
      std::printf("| **pick** %s | | | L0: S%u (%.3f us) | L1: S%u (%.3f us) | lists use L%u (25b's) at S%u | | |\n",
                  probe_bolt::fam_name(s.fam), best_l[0].first, best_l[0].second, best_l[1].first, best_l[1].second, L,
                  best_l[L].first);
      picks[s.fam] = {best_l[L].first, L};
    }
  }
  std::printf("\nint4 GEMV cells the lists bind (%s):", o.sweep ? "the sweep's S, 25b's layout" : "25b's descriptor, PROVISIONAL");
  for (const probe_bolt::Int4Shape& s : probe_bolt::int4_shapes(d))
    std::printf(" %s S%u L%u;", probe_bolt::fam_name(s.fam), picks[s.fam].S, picks[s.fam].L);
  std::printf("\n");

  struct Run {
    Want want;
    std::vector<probe_bolt::Launch> launches;
    std::unique_ptr<Rig::Captured> plain, prof;
    Result r;
  };
  std::vector<Run> runs;
  for (const Want& w : lists) {
    Run r;
    r.want = w;
    r.launches = probe_bolt::build_list(d, w.which);
    r.plain = rig.capture(r.launches, false, picks, w.bf16);
    r.prof = rig.capture(r.launches, true, picks, w.bf16);
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
  for (Run& r : runs) print_list(r.want.name, r.launches, r.r, rig.bytes(r.launches, r.want.bf16), o);

  if (runs.size() > 1) {
    std::printf("\n## the step per list (MEASURED, median plain fence wall)\n\n| list | launches | ms | t/s | vs a |\n"
                "|---|---:|---:|---:|---:|\n");
    const double a = median(runs[0].r.wall_us);
    for (Run& r : runs) {
      const double w = median(r.r.wall_us);
      std::printf("| %s | %zu | %.3f | %.1f | %+.2f %% |\n", r.want.name, r.launches.size(), w / 1e3,
                  w > 0 ? 1e6 / w : 0.0, a > 0 ? 100.0 * (w - a) / a : 0.0);
    }
  }
  if (o.pair) {
    const uint32_t sites = d.layers + 1;
    auto plain = rig.capture_pair(sites, false), prof = rig.capture_pair(sites, true);
    Result r;
    for (uint32_t i = 0; i < 20 + o.rounds * o.reps; ++i) replay(rig, q, f, *plain, *prof, r, i >= 20);
    std::printf("\n## --pair: prep_res_fold SP0 at K 4096 G 16 + kol_norm_finish W16 (the two-launch norm), %u sites: "
                "%.3f us a site in situ, %.1f us a token (MEASURED) - against `a`'s one-launch norm family above\n",
                sites, r.sum_us / double(r.prof_replays) / sites, r.sum_us / double(r.prof_replays));
  }
  std::printf("\nDecision rule (spec 25 Decision 16, PROPOSED): every arm whose skeleton saving (`a` - `fused`, split by "
              "the removed families' in-situ us + gaps) is >= 1 %% of the step is built and A/B'd in plan 25c Task 8; "
              "the S picks become model::BoltDesc::linears' values (PROVISIONAL until then).\n");
  return 0;
}

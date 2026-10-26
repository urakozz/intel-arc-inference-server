// probe_draft_cost - spec 19a (plan 19a Task 4): what one DFlash2 block's weight reads cost
// on the card - the drafter's linears at M = 8 rows in int8 (W8A16, gemv_i8w), bf16
// (gemv_bf16) and our int4 W4A16 g64 (gemv.cl, layout 0), and the int8 target head over
// the block's K = 7 draft rows (full vocabulary and a 32k draft vocabulary).
//
//   probe_draft_cost [--calls 20] [--rounds 3] [--device 0] [--plain-ms <ms>]
//   probe_draft_cost --shapes muse [--mmax 16] [--depth 4096,32768] [--calls 10] [--rounds 3] [--device 0]
//
// No checkpoint: RANDOM WEIGHTS of the drafter's shapes (tools/probe/draft_cost_shapes.h,
// from z-lab/Qwen3.8-27B-DFlash2's config and safetensors header), one allocation per
// layer and linear, so no layer's weights sit in L2 for the next - the bytes a real pass
// streams. Random, not constant: the B70 compresses device memory, and a constant fill
// reads faster than DRAM (probe_verify_m's note). Timing only: no output is read.
//
// What is timed: GEMV launches only, in an in-order list per arm, executed + fenced `calls`
// times per round (a step's submit and wait included, as the engine pays them). Arms: per
// format, the whole block pass (per layer: attn conv proj, q||k||v, o_proj, mlp conv proj,
// gate||up, down, the commit's ctx k||v; then fc and the selector projection - 37 launches)
// and each linear alone over the same weights (its 5 layers, or once; a small linear
// repeated alone can stay L2-resident across calls, so read the pass row as the number);
// the two head arms. Every round runs every arm in a rotated order after a warm-up pass;
// the median over rounds of each arm's mean ms per call is printed. NOT in these numbers:
// the drafter's attention over its 2048-row ring, the norms, RoPE, the two convolutions,
// top-16 and the selector walk (none streams more than ~50 MB per block), and the int4
// arm's split-K folds (gemv.cl writes [S][M][N] partials; a consumer folds them).
// `--plain-ms` (the int8-head plain step, e.g. probe_mtp_steps' verify M=1 at the same
// depth) adds each row's share of a plain step.
//
// **`--shapes muse` (spec 23f Task 4)**: Muse-Glimmer-30B's verify cost at M = 1..mmax rows and one drafter block
// (draft_cost_shapes.h's muse namespace). The verify arm at M is the 52 layers' four int4 GEMVs over their own random
// weights (13.37 GB allocated), the int8 lm_head at M rows and mg_argmax.cl's two stages (the verify's argmax); for
// M = 9..16 also the TWO-PASS form (every GEMV and the head as an M = 8 launch plus an M - 8 one) so the better of the
// two is known. Each M is timed against M = 1 as INTERLEAVED PAIRS after a warm-up (1, M, M, 1, ... - sequential arms
// mis-state ratios by up to 35 % under clock drift), the median of `rounds` pair ratios printed beside the median ms.
// The attention term is ESTIMATED, not timed: at each --depth, 13 full layers' K + V (13,312 B a position) + 39
// windows of 2048 rows, read once per verify whatever M, at the bandwidth the M = 1 pass measured - added to give the
// verify's total and its ratio to M = 1. Then one drafter block at 16 rows per format (bf16 / int8 / int4 g64 layout
// 1, S 1): 5 layers' q||k||v, o, gate||up, down, the commit's ctx k||v, fc once, and the int8 head over the 15 draft
// rows (+ its top-16 is not a GEMV: not timed). NOT in the numbers: the norms, RoPE, the attention kernels' compute,
// the split-K folds and the taps' copies (small against the weight streams).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "draft_cost_shapes.h"
#include "kernels/kernels.h"
#include "kernels/muse_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

namespace {

enum class Fmt { Int8, Bf16, Int4 };
const char* fmt_name(Fmt f) { return f == Fmt::Int8 ? "int8" : f == Fmt::Bf16 ? "bf16" : "int4-g64"; }

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

struct Rng {
  uint32_t s;
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
};

// Device memory filled from the host with random words; `kind` keeps the values finite
// where a format reads them as floats (bf16 weights / activations, f16 / fp32 scales).
enum class Fill { Bytes, Bf16, F16, F32 };
uint32_t bf16_bits(uint32_t x) { return (x & 0x807fu) | ((0x77u + ((x >> 8) & 0xfu)) << 7); }
uint32_t f16_bits(uint32_t x) { return (x & 0x83ffu) | (0x08u << 10); }   // ~2^-7
std::unique_ptr<l0::Mem> random_mem(l0::Context& ctx, l0::CmdList& imm, size_t bytes, Fill kind,
                                    uint32_t seed, std::vector<uint32_t>& host) {
  auto m = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, bytes);
  host.resize((bytes + 3) / 4);
  Rng r{seed | 1u};
  for (uint32_t& w : host) {
    uint32_t v = r.next();
    if (kind == Fill::Bf16)   // two bf16, exponent in [-8, 7] (probe_verify_m's pattern)
      v = bf16_bits(v & 0xffffu) | (bf16_bits(v >> 16) << 16);
    else if (kind == Fill::F16)
      v = f16_bits(v & 0xffffu) | (f16_bits(v >> 16) << 16);
    else if (kind == Fill::F32)   // fp32 in [2^-8, 2^-7)
      v = (v & 0x007fffffu) | (0x77u << 23);
    w = v;
  }
  imm.copy(m->ptr(), host.data(), bytes);
  return m;
}

struct Launch {
  std::unique_ptr<l0::Kernel> k;
  uint32_t gx = 1, gy = 1;
};

// A site's weights in one format: fresh random device memory, one allocation per site.
struct Weights {
  const l0::Mem* w = nullptr;
  const l0::Mem* s = nullptr;   // int8: fp32 [N] row scales; int4 layout 0: f16 [K/64][N]; bf16 / layout 1: none
  double mb = 0;                // bytes one launch reads, MB
};

struct Arm {
  std::string name;
  double mb = 0;   // weight + scale MB read per call
  l0::CmdList list;
  std::vector<double> ms;
};

// The device, its lists and every allocation they point at - main()'s helpers, shared by both shapes.
struct Probe {
  explicit Probe(uint32_t device) : ctx(device), imm(l0::CmdList::immediate(ctx)), q(ctx), fence(q) {}
  l0::Context ctx;
  l0::CmdList imm;
  l0::Queue q;
  l0::Fence fence;
  std::vector<uint32_t> host;
  std::vector<std::unique_ptr<l0::Mem>> mems;
  std::vector<std::unique_ptr<l0::Kernel>> kernels_owned;
  std::vector<std::pair<std::string, std::unique_ptr<l0::Module>>> modules;
  std::vector<std::pair<std::pair<unsigned, unsigned>, const l0::Mem*>> acts;   // (rows, K) -> bf16 [rows][K]
  uint32_t seed = 0x19a4u;

  l0::Module& module(const std::string& v) {
    for (auto& [name, m] : modules)
      if (name == v) return *m;
    modules.emplace_back(v, std::make_unique<l0::Module>(ctx, kernels::path(v)));
    return *modules.back().second;
  }
  // Activations: one random bf16 [rows][K] per (rows, K), shared (a step's are L2-resident too).
  const l0::Mem* x_for(unsigned K, unsigned rows) {
    for (auto& [k, m] : acts)
      if (k.first == rows && k.second == K) return m;
    mems.push_back(random_mem(ctx, imm, size_t(rows) * K * 2, Fill::Bf16, seed += 7, host));
    acts.push_back({{rows, K}, mems.back().get()});
    return mems.back().get();
  }
  // Outputs one per launch (overwritten, never read) - or, with `shared_out` floats set (the Muse arms: 52 layers x
  // 24 arms of per-launch outputs would not fit), one scratch of that many floats every launch writes over (in-order
  // lists, timing only).
  size_t shared_out = 0;
  const l0::Mem* shared = nullptr;
  const l0::Mem* out_for(size_t floats) {
    if (shared_out != 0) {
      if (floats > shared_out) throw std::logic_error("probe_draft_cost: an output larger than the shared scratch");
      if (!shared) {
        mems.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, shared_out * 4));
        shared = mems.back().get();
      }
      return shared;
    }
    mems.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, floats * 4));
    return mems.back().get();
  }
  Weights weights(Fmt f, unsigned K, unsigned N, unsigned layout = 0) {
    Weights r;
    if (f == Fmt::Int8) {
      mems.push_back(random_mem(ctx, imm, size_t(K) * N, Fill::Bytes, seed += 7, host));
      r.w = mems.back().get();
      mems.push_back(random_mem(ctx, imm, size_t(N) * 4, Fill::F32, seed += 7, host));
      r.s = mems.back().get();
      r.mb = (double(K) * N + N * 4.0) / 1e6;
    } else if (f == Fmt::Bf16) {
      mems.push_back(random_mem(ctx, imm, size_t(K) * N * 2, Fill::Bf16, seed += 7, host));
      r.w = mems.back().get();
      r.mb = double(K) * N * 2 / 1e6;
    } else if (layout == 0) {   // int4 g64 symmetric, layout 0: w [K/8][N] u32, scales [K/64][N] f16
      mems.push_back(random_mem(ctx, imm, size_t(K) / 8 * N * 4, Fill::Bytes, seed += 7, host));
      r.w = mems.back().get();
      mems.push_back(random_mem(ctx, imm, size_t(K) / 64 * N * 2, Fill::F16, seed += 7, host));
      r.s = mems.back().get();
      r.mb = (double(K) / 2 * N + double(K) / 64 * N * 2) / 1e6;
    } else {   // layout 1: (N / 16) x (K / 64) tiles of 136 u32 (nibbles + f16 scales; random words: timing only)
      mems.push_back(random_mem(ctx, imm, size_t(N / 16) * (K / 64) * 136 * 4, Fill::F16, seed += 7, host));
      r.w = mems.back().get();
      r.s = r.w;   // gemv.cl's layout 1 reads its scales from the tiles; the argument is unused
      r.mb = double(N / 16) * (K / 64) * 136 * 4 / 1e6;
    }
    return r;
  }
  // One launch of a K x N linear at M rows over `w`, its activations from row `row0` of a [rows][K] buffer:
  // gemv_i8w (grid N / 64, WG 64), gemv_bf16 (kernels::gemv_bf16_tiling) or gemv.cl (grid (N / 64, S), WG 64).
  Launch make(Fmt f, unsigned M, unsigned K, unsigned N, unsigned S, const Weights& w, unsigned layout = 0,
              unsigned rows = 0, unsigned row0 = 0) {
    Launch l;
    const l0::Mem* xm = x_for(K, rows ? rows : M);
    const void* x = static_cast<const uint8_t*>(xm->ptr()) + size_t(row0) * K * 2;
    if (f == Fmt::Int8) {
      l.k = std::make_unique<l0::Kernel>(module(kernels::gemv_i8w_variant(M, K, N)), "gemv_i8w");
      l.k->group_size(kernels::kGemvI8wCols);
      l.k->arg_ptr(0, w.w->ptr());
      l.k->arg_ptr(1, w.s->ptr());
      l.k->arg_ptr(2, x);
      l.k->arg_ptr(3, out_for(size_t(M) * N)->ptr());
      l.gx = N / kernels::kGemvI8wCols;
    } else if (f == Fmt::Bf16) {
      const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(N);
      l.k = std::make_unique<l0::Kernel>(module(kernels::gemv_bf16_variant(M, K, N, t)), "gemv_bf16");
      l.k->group_size(t.cols * t.ksplit);
      l.k->arg_ptr(0, w.w->ptr());
      l.k->arg_ptr(1, x);
      l.k->arg_ptr(2, out_for(size_t(M) * N)->ptr());
      l.gx = N / t.cols;
    } else {
      l.k = std::make_unique<l0::Kernel>(module(kernels::gemv_variant(M, K, N, S, layout)), "gemv");
      l.k->group_size(64);
      l.k->arg_ptr(0, w.w->ptr());
      l.k->arg_ptr(1, w.s->ptr());
      l.k->arg_ptr(2, x);
      l.k->arg_ptr(3, out_for(size_t(S) * M * N)->ptr());
      l.gx = N / 64;
      l.gy = S;
    }
    return l;
  }
  void append(Arm& a, Launch l, double mb) {
    a.list.launch(*l.k, l.gx, l.gy);
    a.mb += mb;
    kernels_owned.push_back(std::move(l.k));
  }
  // mg_argmax.cl's two stages at M rows over a logits buffer (the verify's argmax).
  void append_argmax(Arm& a, unsigned M, const l0::Mem* ctrl) {
    namespace km = kernels::muse;
    // Its own buffers (not the shared GEMV scratch: stage 1 reads one while it writes the other).
    mems.push_back(random_mem(ctx, imm, size_t(M) * km::kVocab * 4, Fill::F32, seed += 7, host));
    const l0::Mem* logits = mems.back().get();
    mems.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(M) * km::argmax_groups() * 2 * 4));
    const l0::Mem* part = mems.back().get();
    mems.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, 16 * 4));
    const l0::Mem* vout = mems.back().get();
    auto k1 = std::make_unique<l0::Kernel>(module(km::vargmax1_variant(M)), "mg_vargmax_stage1");
    k1->group_size(km::kArgmaxWg);
    k1->arg_ptr(0, logits->ptr());
    k1->arg_ptr(1, part->ptr());
    a.list.launch(*k1, km::argmax_groups(), M);
    kernels_owned.push_back(std::move(k1));
    auto k2 = std::make_unique<l0::Kernel>(module(km::vargmax2_variant()), "mg_vargmax_stage2");
    k2->group_size(km::kArgmaxWg);
    k2->arg_ptr(0, ctrl->ptr());
    k2->arg_ptr(1, part->ptr());
    k2->arg_ptr(2, vout->ptr());
    a.list.launch(*k2, 1, 1);
    kernels_owned.push_back(std::move(k2));
  }
  double run(Arm& a, uint32_t n) {
    Timer t;
    t.start();
    for (uint32_t i = 0; i < n; ++i) {
      q.execute(a.list, &fence);
      fence.wait();
    }
    return t.ms() / n;
  }
  double allocated_gb() const {
    double b = 0;
    for (const auto& m : mems) b += double(m->size());
    return b / 1e9;
  }
};

std::vector<uint32_t> parse_list(const std::string& s) {
  std::vector<uint32_t> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) v.push_back(uint32_t(std::stoul(item)));
  return v;
}

int run_qwen38(uint32_t calls, uint32_t rounds, uint32_t device, double plain_ms) {
  using namespace draft_cost;
  Probe P(device);
  std::printf("device: %s\n", P.ctx.name().c_str());
  std::printf("probe_draft_cost: DFlash2 (z-lab/Qwen3.8-27B-DFlash2's shapes, %u layers) at M = %u"
              " rows, the int8 head at %u rows; RANDOM weights of the checkpoint's shapes (timing"
              " only)\n",
              kLayers, kRows, kHeadRows);
  std::vector<Arm> arms;
  const Fmt fmts[] = {Fmt::Int8, Fmt::Bf16, Fmt::Int4};
  for (Fmt f : fmts) {
    // The sites in the drafter's order: every per-layer linear of every layer, then the
    // once-per-block ones.
    std::vector<std::pair<const Linear*, Weights>> sites;
    for (unsigned layer = 0; layer < kLayers; ++layer)
      for (const Linear& L : kLinears)
        if (L.per_layer) sites.push_back({&L, P.weights(f, L.K, L.N)});
    for (const Linear& L : kLinears)
      if (!L.per_layer) sites.push_back({&L, P.weights(f, L.K, L.N)});
    // The block pass over every site, then each linear alone over the same sites.
    Arm pass{std::string(fmt_name(f)) + " block pass", 0, l0::CmdList::regular(P.ctx), {}};
    for (const auto& [L, w] : sites) P.append(pass, P.make(f, kRows, L->K, L->N, L->int4_s, w), w.mb);
    pass.list.close();
    arms.push_back(std::move(pass));
    for (const Linear& L : kLinears) {
      Arm one{std::string(fmt_name(f)) + " " + L.name + (L.per_layer ? " x5" : " x1"), 0,
              l0::CmdList::regular(P.ctx), {}};
      for (const auto& [l, w] : sites)
        if (l == &L) P.append(one, P.make(f, kRows, L.K, L.N, L.int4_s, w), w.mb);
      one.list.close();
      arms.push_back(std::move(one));
    }
  }
  const size_t first_head = arms.size();
  for (unsigned V : kHeadVocab) {
    Arm h{"int8 head x" + std::to_string(kHeadRows) + " rows, V " + std::to_string(V), 0,
          l0::CmdList::regular(P.ctx), {}};
    const Weights w = P.weights(Fmt::Int8, kHidden, V);
    P.append(h, P.make(Fmt::Int8, kHeadRows, kHidden, V, 1, w), w.mb);
    h.list.close();
    arms.push_back(std::move(h));
  }
  std::printf("allocated %.2f GB (random weights and buffers), %zu arms\n", P.allocated_gb(), arms.size());

  for (Arm& a : arms) P.run(a, 4);   // warm-up
  for (uint32_t r = 0; r < rounds; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      Arm& a = arms[(i + r) % arms.size()];
      a.ms.push_back(P.run(a, calls));
    }

  std::printf("probe_draft_cost: %u calls x %u rounds, arms rotated, median of the round means\n",
              calls, rounds);
  std::printf("| arm | ms | range | weight MB | GB/s |%s\n", plain_ms > 0 ? " x plain step |" : "");
  std::printf("|---|---:|---|---:|---:|%s\n", plain_ms > 0 ? "---:|" : "");
  for (Arm& a : arms) {
    std::vector<double> v = a.ms;
    std::sort(v.begin(), v.end());
    const double med = median(a.ms);
    std::printf("| %s | %.3f | %.3f-%.3f | %.1f | %.0f |", a.name.c_str(), med, v.front(),
                v.back(), a.mb, a.mb / med);
    if (plain_ms > 0) std::printf(" %.3f |", med / plain_ms);
    std::printf("\n");
  }
  // One block's GEMVs = the pass + the head's K rows, per format and head size.
  std::printf("one DFlash block, GEMVs only (the block pass + the int8 head's %u rows):\n",
              kHeadRows);
  for (size_t f = 0; f < 3; ++f) {
    const double pass = median(arms[f * (1 + kNumLinears)].ms);
    for (size_t h = 0; h < sizeof(kHeadVocab) / sizeof(kHeadVocab[0]); ++h) {
      const double head = median(arms[first_head + h].ms);
      std::printf("| block %s, head V %u | %.3f ms |", fmt_name(fmts[f]), kHeadVocab[h],
                  pass + head);
      if (plain_ms > 0) std::printf(" x plain step %.3f |", (pass + head) / plain_ms);
      std::printf("\n");
    }
  }
  return 0;
}

// ---- spec 23f: Muse-Glimmer ----------------------------------------------------------------------------------------

int run_muse(uint32_t calls, uint32_t rounds, uint32_t device, uint32_t mmax, const std::vector<uint32_t>& depths) {
  namespace dm = draft_cost::muse;
  if (mmax < 1 || mmax > dm::kMaxM) throw std::invalid_argument("--mmax 1..16");
  Probe P(device);
  P.shared_out = std::max<size_t>(size_t(8) * dm::kMaxM * 39936, size_t(dm::kMaxM) * dm::kVocab);   // gate||up S8
  std::printf("device: %s\n", P.ctx.name().c_str());
  std::printf("probe_draft_cost --shapes muse: the verify at M = 1..%u rows (52 layers x 4 int4 g64 GEMVs, the int8 "
              "head, mg_argmax) and one drafter block at 16 rows (bf16 / int8 / int4 g64); RANDOM weights of the "
              "checkpoints' shapes (timing only)\n",
              mmax);
  // The control block mg_vargmax_stage2 reads (n_active = M is set per arm before its timing).
  P.mems.push_back(std::make_unique<l0::Mem>(P.ctx, l0::MemKind::Shared, 128));
  l0::Mem* ctrl = P.mems.back().get();
  uint32_t* cw = ctrl->as<uint32_t>();
  std::fill(cw, cw + 32, 0u);
  // The target's weights: 52 layers x 4 sites, int4 layout 0 at their S, and the int8 head - shared by every M.
  std::vector<std::pair<const draft_cost::Linear*, Weights>> sites;
  for (unsigned l = 0; l < dm::kLayers; ++l)
    for (const draft_cost::Linear& L : dm::kTarget) sites.push_back({&L, P.weights(Fmt::Int4, L.K, L.N)});
  const Weights head = P.weights(Fmt::Int8, dm::kHidden, dm::kVocab);
  struct VerifyArm {
    uint32_t M;
    bool two_pass;
    Arm arm;
  };
  std::vector<VerifyArm> verify;
  auto build_verify = [&](uint32_t M, bool two_pass) {
    Arm a{"verify M " + std::to_string(M) + (two_pass ? " (two passes: 8 + " + std::to_string(M - 8) + ")" : ""), 0,
          l0::CmdList::regular(P.ctx), {}};
    for (const auto& [L, w] : sites) {
      if (!two_pass) {
        P.append(a, P.make(Fmt::Int4, M, L->K, L->N, L->int4_s, w), w.mb);
      } else {
        P.append(a, P.make(Fmt::Int4, 8, L->K, L->N, L->int4_s, w, 0, M, 0), w.mb);
        P.append(a, P.make(Fmt::Int4, M - 8, L->K, L->N, L->int4_s, w, 0, M, 8), w.mb);
      }
    }
    if (!two_pass) {
      P.append(a, P.make(Fmt::Int8, M, dm::kHidden, dm::kVocab, 1, head), head.mb);
    } else {
      P.append(a, P.make(Fmt::Int8, 8, dm::kHidden, dm::kVocab, 1, head, 0, M, 0), head.mb);
      P.append(a, P.make(Fmt::Int8, M - 8, dm::kHidden, dm::kVocab, 1, head, 0, M, 8), head.mb);
    }
    P.append_argmax(a, M, ctrl);
    a.list.close();
    verify.push_back({M, two_pass, std::move(a)});
  };
  for (uint32_t M = 1; M <= mmax; ++M) {
    build_verify(M, false);
    if (M > 8) build_verify(M, true);
  }
  // One drafter block per format.
  const Fmt fmts[] = {Fmt::Bf16, Fmt::Int8, Fmt::Int4};
  std::vector<Arm> blocks;
  const Weights dhead = P.weights(Fmt::Int8, dm::kHidden, dm::kVocab);
  for (Fmt f : fmts) {
    Arm a{std::string("drafter block ") + fmt_name(f) + " + int8 head x15", 0, l0::CmdList::regular(P.ctx), {}};
    for (unsigned l = 0; l < dm::kDraftLayers; ++l)
      for (const draft_cost::Linear& L : dm::kDrafter)
        if (L.per_layer) {
          const Weights w = P.weights(f, L.K, L.N, 1);
          P.append(a, P.make(f, dm::kDraftRows, L.K, L.N, 1, w, 1), w.mb);
        }
    for (const draft_cost::Linear& L : dm::kDrafter)
      if (!L.per_layer) {
        const Weights w = P.weights(f, L.K, L.N, 1);
        P.append(a, P.make(f, dm::kDraftRows, L.K, L.N, 1, w, 1), w.mb);
      }
    P.append(a, P.make(Fmt::Int8, dm::kDraftHeadRows, dm::kHidden, dm::kVocab, 1, dhead), dhead.mb);
    a.list.close();
    blocks.push_back(std::move(a));
  }
  std::printf("allocated %.2f GB (random weights and buffers), %zu verify arms, %zu drafter arms\n", P.allocated_gb(),
              verify.size(), blocks.size());

  // Interleaved pairs after a warm-up: per round, M = 1 then the arm, then the arm then M = 1 (the order alternates
  // with the round) - the ratio is the pair's.
  Arm& one = verify.front().arm;
  for (VerifyArm& v : verify) P.run(v.arm, 2);
  for (Arm& b : blocks) P.run(b, 2);
  std::vector<std::vector<double>> ratios(verify.size());
  std::vector<std::vector<double>> block_ratios(blocks.size());
  for (uint32_t r = 0; r < rounds; ++r) {
    for (size_t i = 0; i < verify.size(); ++i) {
      cw[1] = verify[i].M;   // Control::n_active (mg_vargmax_stage2's bound)
      double a, b;
      if (r % 2 == 0) {
        cw[1] = 1;
        a = P.run(one, calls);
        cw[1] = verify[i].M;
        b = P.run(verify[i].arm, calls);
      } else {
        cw[1] = verify[i].M;
        b = P.run(verify[i].arm, calls);
        cw[1] = 1;
        a = P.run(one, calls);
      }
      verify[i].arm.ms.push_back(b);
      ratios[i].push_back(b / a);
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
      cw[1] = 1;
      double a, b;
      if (r % 2 == 0) {
        a = P.run(one, calls);
        b = P.run(blocks[i], calls);
      } else {
        b = P.run(blocks[i], calls);
        a = P.run(one, calls);
      }
      blocks[i].ms.push_back(b);
      block_ratios[i].push_back(b / a);
    }
  }
  const double one_ms = median(verify.front().arm.ms);
  const double gbps = verify.front().arm.mb / one_ms;   // MB / ms = GB/s
  std::printf("probe_draft_cost --shapes muse: %u calls x %u rounds, interleaved pairs against M = 1, medians\n", calls,
              rounds);
  std::printf("the M = 1 pass: %.3f ms over %.0f MB = %.0f GB/s (the attention estimate's bandwidth)\n", one_ms,
              verify.front().arm.mb, gbps);
  std::printf("| arm | GEMVs + head + argmax ms | ratio to M = 1 (pairs) |");
  for (uint32_t d : depths) std::printf(" attn est. at %u ms | verify at %u ms | x M = 1 |", d, d);
  std::printf("\n|---|---:|---:|");
  for (size_t i = 0; i < depths.size(); ++i) std::printf("---:|---:|---:|");
  std::printf("\n");
  for (size_t i = 0; i < verify.size(); ++i) {
    const double ms = median(verify[i].arm.ms), ratio = median(ratios[i]);
    std::printf("| %s | %.3f | %.3f |", verify[i].arm.name.c_str(), ms, ratio);
    for (uint32_t d : depths) {
      const double attn = (dm::kFullKvBytesPerPos * d + dm::kWindowBytes) / 1e6 / gbps;   // MB / (MB/ms)
      std::printf(" %.3f | %.3f | %.3f |", attn, ms + attn, (ms + attn) / (one_ms + attn));
    }
    std::printf("\n");
  }
  std::printf("| arm | ms | x the M = 1 verify's GEMVs (pairs) | weight MB | GB/s |\n|---|---:|---:|---:|---:|\n");
  for (size_t i = 0; i < blocks.size(); ++i) {
    const double ms = median(blocks[i].ms);
    std::printf("| %s | %.3f | %.3f | %.0f | %.0f |\n", blocks[i].name.c_str(), ms, median(block_ratios[i]),
                blocks[i].mb, blocks[i].mb / ms);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t calls = 20, rounds = 3, device = 0, mmax = 16;
  double plain_ms = 0;
  std::string shapes = "qwen38";
  std::vector<uint32_t> depths = {4096, 32768};
  bool calls_given = false;
  if (argc % 2 == 0) {
    std::fprintf(stderr, "usage: probe_draft_cost [--shapes qwen38|muse] [--calls 20] [--rounds 3] [--device 0]"
                         " [--plain-ms <ms>] [--mmax 16] [--depth 4096,32768]\n");
    return 2;
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--calls") {
      calls = uint32_t(std::stoul(argv[i + 1]));
      calls_given = true;
    } else if (a == "--rounds") rounds = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--device") device = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--plain-ms") plain_ms = std::stod(argv[i + 1]);
    else if (a == "--shapes") shapes = argv[i + 1];
    else if (a == "--mmax") mmax = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--depth") depths = parse_list(argv[i + 1]);
    else {
      std::fprintf(stderr, "probe_draft_cost: unknown flag %s\n", a.c_str());
      return 2;
    }
  }
  if (calls == 0 || rounds == 0) return 2;
  try {
    if (shapes == "muse") return run_muse(calls_given ? calls : 10, rounds, device, mmax, depths);
    if (shapes != "qwen38") {
      std::fprintf(stderr, "probe_draft_cost: --shapes qwen38|muse, not %s\n", shapes.c_str());
      return 2;
    }
    return run_qwen38(calls, rounds, device, plain_ms);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_draft_cost: %s\n", e.what());
    return 1;
  }
}

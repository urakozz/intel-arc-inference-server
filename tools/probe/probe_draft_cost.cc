// probe_draft_cost - spec 19a (plan 19a Task 4): what one DFlash2 block's weight reads cost
// on the card - the drafter's linears at M = 8 rows in int8 (W8A16, gemv_i8w), bf16
// (gemv_bf16) and our int4 W4A16 g64 (gemv.cl, layout 0), and the int8 target head over
// the block's K = 7 draft rows (full vocabulary and a 32k draft vocabulary).
//
//   probe_draft_cost [--calls 20] [--rounds 3] [--device 0] [--plain-ms <ms>]
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
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "draft_cost_shapes.h"
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
  const l0::Mem* s = nullptr;   // int8: fp32 [N] row scales; int4: f16 [K/64][N]; bf16: none
  double mb = 0;                // bytes one launch reads, MB
};

struct Arm {
  std::string name;
  double mb = 0;   // weight + scale MB read per call
  l0::CmdList list;
  std::vector<double> ms;
};

}  // namespace

int main(int argc, char** argv) {
  uint32_t calls = 20, rounds = 3, device = 0;
  double plain_ms = 0;
  if (argc % 2 == 0) {
    std::fprintf(stderr, "usage: probe_draft_cost [--calls 20] [--rounds 3] [--device 0]"
                         " [--plain-ms <ms>]\n");
    return 2;
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--calls") calls = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--rounds") rounds = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--device") device = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--plain-ms") plain_ms = std::stod(argv[i + 1]);
    else {
      std::fprintf(stderr, "probe_draft_cost: unknown flag %s\n", a.c_str());
      return 2;
    }
  }
  if (calls == 0 || rounds == 0) return 2;
  using namespace draft_cost;
  l0::Context ctx(device);
  std::printf("device: %s\n", ctx.name().c_str());
  std::printf("probe_draft_cost: DFlash2 (z-lab/Qwen3.8-27B-DFlash2's shapes, %u layers) at M = %u"
              " rows, the int8 head at %u rows; RANDOM weights of the checkpoint's shapes (timing"
              " only)\n",
              kLayers, kRows, kHeadRows);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  std::vector<uint32_t> host;
  std::vector<std::unique_ptr<l0::Mem>> mems;   // everything the lists point at
  std::vector<std::unique_ptr<l0::Kernel>> kernels_owned;
  std::vector<std::pair<std::string, std::unique_ptr<l0::Module>>> modules;
  auto module = [&](const std::string& v) -> l0::Module& {
    for (auto& [name, m] : modules)
      if (name == v) return *m;
    modules.emplace_back(v, std::make_unique<l0::Module>(ctx, kernels::path(v)));
    return *modules.back().second;
  };
  uint32_t seed = 0x19a4u;
  // Activations: one random bf16 [8][K] per K, shared (a step's are L2-resident too);
  // outputs one per launch (overwritten, never read).
  std::vector<std::pair<unsigned, const l0::Mem*>> acts;
  auto x_for = [&](unsigned K) -> const l0::Mem* {
    for (auto& [k, m] : acts)
      if (k == K) return m;
    mems.push_back(random_mem(ctx, imm, size_t(kRows) * K * 2, Fill::Bf16, seed += 7, host));
    acts.push_back({K, mems.back().get()});
    return mems.back().get();
  };
  auto out_for = [&](size_t floats) -> const l0::Mem* {
    mems.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, floats * 4));
    return mems.back().get();
  };
  auto weights = [&](Fmt f, unsigned K, unsigned N) {
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
    } else {   // int4 g64 symmetric, layout 0: w [K/8][N] u32, scales [K/64][N] f16
      mems.push_back(random_mem(ctx, imm, size_t(K) / 8 * N * 4, Fill::Bytes, seed += 7, host));
      r.w = mems.back().get();
      mems.push_back(random_mem(ctx, imm, size_t(K) / 64 * N * 2, Fill::F16, seed += 7, host));
      r.s = mems.back().get();
      r.mb = (double(K) / 2 * N + double(K) / 64 * N * 2) / 1e6;
    }
    return r;
  };
  // One launch of a K x N linear at M rows over `w`: gemv_i8w (grid N / 64, WG 64),
  // gemv_bf16 (kernels::gemv_bf16_tiling) or gemv.cl (grid (N / 64, S), WG 64).
  auto make = [&](Fmt f, unsigned M, unsigned K, unsigned N, unsigned S, const Weights& w) {
    Launch l;
    const l0::Mem* x = x_for(K);
    if (f == Fmt::Int8) {
      l.k = std::make_unique<l0::Kernel>(module(kernels::gemv_i8w_variant(M, K, N)), "gemv_i8w");
      l.k->group_size(kernels::kGemvI8wCols);
      l.k->arg_ptr(0, w.w->ptr());
      l.k->arg_ptr(1, w.s->ptr());
      l.k->arg_ptr(2, x->ptr());
      l.k->arg_ptr(3, out_for(size_t(M) * N)->ptr());
      l.gx = N / kernels::kGemvI8wCols;
    } else if (f == Fmt::Bf16) {
      const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(N);
      l.k = std::make_unique<l0::Kernel>(module(kernels::gemv_bf16_variant(M, K, N, t)),
                                         "gemv_bf16");
      l.k->group_size(t.cols * t.ksplit);
      l.k->arg_ptr(0, w.w->ptr());
      l.k->arg_ptr(1, x->ptr());
      l.k->arg_ptr(2, out_for(size_t(M) * N)->ptr());
      l.gx = N / t.cols;
    } else {
      l.k = std::make_unique<l0::Kernel>(module(kernels::gemv_variant(M, K, N, S, 0)), "gemv");
      l.k->group_size(64);
      l.k->arg_ptr(0, w.w->ptr());
      l.k->arg_ptr(1, w.s->ptr());
      l.k->arg_ptr(2, x->ptr());
      l.k->arg_ptr(3, out_for(size_t(S) * M * N)->ptr());
      l.gx = N / 64;
      l.gy = S;
    }
    return l;
  };
  auto append = [&](Arm& a, Launch l, double mb) {
    a.list.launch(*l.k, l.gx, l.gy);
    a.mb += mb;
    kernels_owned.push_back(std::move(l.k));
  };

  std::vector<Arm> arms;
  const Fmt fmts[] = {Fmt::Int8, Fmt::Bf16, Fmt::Int4};
  for (Fmt f : fmts) {
    // The sites in the drafter's order: every per-layer linear of every layer, then the
    // once-per-block ones.
    std::vector<std::pair<const Linear*, Weights>> sites;
    for (unsigned layer = 0; layer < kLayers; ++layer)
      for (const Linear& L : kLinears)
        if (L.per_layer) sites.push_back({&L, weights(f, L.K, L.N)});
    for (const Linear& L : kLinears)
      if (!L.per_layer) sites.push_back({&L, weights(f, L.K, L.N)});
    // The block pass over every site, then each linear alone over the same sites.
    Arm pass{std::string(fmt_name(f)) + " block pass", 0, l0::CmdList::regular(ctx), {}};
    for (const auto& [L, w] : sites) append(pass, make(f, kRows, L->K, L->N, L->int4_s, w), w.mb);
    pass.list.close();
    arms.push_back(std::move(pass));
    for (const Linear& L : kLinears) {
      Arm one{std::string(fmt_name(f)) + " " + L.name + (L.per_layer ? " x5" : " x1"), 0,
              l0::CmdList::regular(ctx), {}};
      for (const auto& [l, w] : sites)
        if (l == &L) append(one, make(f, kRows, L.K, L.N, L.int4_s, w), w.mb);
      one.list.close();
      arms.push_back(std::move(one));
    }
  }
  const size_t first_head = arms.size();
  for (unsigned V : kHeadVocab) {
    Arm h{"int8 head x" + std::to_string(kHeadRows) + " rows, V " + std::to_string(V), 0,
          l0::CmdList::regular(ctx), {}};
    const Weights w = weights(Fmt::Int8, kHidden, V);
    append(h, make(Fmt::Int8, kHeadRows, kHidden, V, 1, w), w.mb);
    h.list.close();
    arms.push_back(std::move(h));
  }
  double dev_mb = 0;
  for (const auto& m : mems) dev_mb += m->size() / 1e6;
  std::printf("allocated %.2f GB (random weights and buffers), %zu arms\n", dev_mb / 1e3,
              arms.size());

  auto run = [&](Arm& a, uint32_t n) {
    Timer t;
    t.start();
    for (uint32_t i = 0; i < n; ++i) {
      q.execute(a.list, &fence);
      fence.wait();
    }
    return t.ms() / n;
  };
  for (Arm& a : arms) run(a, 4);   // warm-up
  for (uint32_t r = 0; r < rounds; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      Arm& a = arms[(i + r) % arms.size()];
      a.ms.push_back(run(a, calls));
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

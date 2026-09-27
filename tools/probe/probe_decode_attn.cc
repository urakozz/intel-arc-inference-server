// probe_decode_attn - spec 10 P0 and P1: decode attention at depth.
//
// Plan: docs/superpowers/plans/2026-09-28-spec10a-decode-attn-probe.md.
// Record: docs/probe-decode-attn-2026-09-28.md.
//
//   probe_decode_attn capture <snapshot> <ids> <outdir> [--steps 8] [--depths a,b,...]
//   probe_decode_attn arms <outdir> --M 1|4 [--depths a,b] [--arms t1,t2,...] [--tgt N]
//                          [--rounds 5] [--reps 10] [--time]
//
// capture (P0 + the real inputs): loads the checkpoint at max_len 131072, prefills the
// ids (tiled to the deepest depth), then for every depth rewinds Control::pos there and
//   * replays the PROFILED captured list --steps times (pos rewound before each), and
//     reports attn_prep / attn_decode / attn_reduce summed over the 16 FA layers, the
//     whole step's kernel sum, and the unprofiled list's fence wall (8 replays);
//   * then (after every depth is profiled) runs 4 plain steps from pos = depth and keeps
//     the last FA layer's attn_q, attn_gate and attn_out after each: the q of rows
//     m = 0..3 of an M = 4 verify at `depth`, and production's own M = 1 output for each.
// Finally it writes that layer's KV (FA index 15) for positions [0, deepest + 4).
//
// arms (P1): standalone, no checkpoint. Arm 0 is src/kernels/attn.cl's attn_decode +
// attn_reduce compiled at the probe's M (pda_v1_M<M>); every other arm is
// probe_decode_attn.cl at one -D set (pda_<tag>_M<M>, tag P<ppw>_L<load>_D<dot>_F<pf>_R<red>).
// Per arm and depth: output cosine per (q head, m) and max abs error against arm 0,
// bitwise repeatability (two runs), arm 0's bitwise agreement with production's dumped
// output, and (--time) interleaved rounds of --reps launches each, per-kernel L0
// timestamps, median per arm and the median paired ratio arm 0 / arm.
//
// PROBE-ONLY: no production file is edited.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "timer.h"

namespace {

constexpr uint32_t kMaxLen = 131072;
constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kRow = kQH * kHD;   // 6144
constexpr uint32_t kFa = 15;                                          // the dumped FA layer
constexpr uint32_t kCtrlTgt = 19;                                     // Control::pad[0]

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
std::vector<uint32_t> parse_list(const std::string& s) {
  std::vector<uint32_t> v;
  size_t st = 0;
  while (st < s.size()) {
    const size_t e = std::min(s.find(',', st), s.size());
    v.push_back(uint32_t(std::stoul(s.substr(st, e - st))));
    st = e + 1;
  }
  return v;
}
std::vector<std::string> parse_tags(const std::string& s) {
  std::vector<std::string> v;
  size_t st = 0;
  while (st < s.size()) {
    const size_t e = std::min(s.find(',', st), s.size());
    v.push_back(s.substr(st, e - st));
    st = e + 1;
  }
  return v;
}
void download(l0::Context& ctx, void* host, const void* dev, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(host) + off, static_cast<const char*>(dev) + off,
           std::min(kChunk, bytes - off));
  }
}
void upload(l0::Context& ctx, void* dev, const void* host, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(dev) + off, static_cast<const char*>(host) + off,
           std::min(kChunk, bytes - off));
  }
}
void write_file(const std::string& p, const void* d, size_t n) {
  std::ofstream f(p, std::ios::binary);
  f.write(static_cast<const char*>(d), std::streamsize(n));
  if (!f) throw std::runtime_error("cannot write " + p);
}
template <class T>
std::vector<T> read_file(const std::string& p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
  if (!f) throw std::runtime_error("cannot read " + p);
  return v;
}
float bf16f(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// ============================================================================
// capture: P0 and the inputs
// ============================================================================
int run_capture(int argc, char** argv) {
  if (argc < 5) throw std::runtime_error("capture <snapshot> <ids> <outdir> [--steps S] [--depths ..]");
  const std::string snap = argv[2], ids_path = argv[3], out = argv[4];
  uint32_t steps = 8;
  std::vector<uint32_t> depths = {4096, 32768, 65536, 130816};
  for (int i = 5; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--steps") steps = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--depths") depths = parse_list(argv[i + 1]);
    else throw std::runtime_error("unknown flag " + a);
  }
  const uint32_t deepest = *std::max_element(depths.begin(), depths.end());
  if (deepest + 4 > kMaxLen) throw std::runtime_error("depth + 4 exceeds 131072");

  std::vector<uint32_t> base;
  {
    std::ifstream f(ids_path);
    uint32_t v;
    while (f >> v) base.push_back(v);
    if (base.empty()) throw std::runtime_error("no ids in " + ids_path);
  }
  std::vector<uint32_t> ids(deepest);
  for (uint32_t i = 0; i < deepest; ++i) ids[i] = base[i % base.size()];
  std::printf("# capture: %zu ids in %s, tiled to %u; depths", base.size(), ids_path.c_str(),
              deepest);
  for (uint32_t d : depths) std::printf(" %u", d);
  std::printf("; %u profiled steps per depth\n", steps);

  l0::Context ctx(0);
  std::printf("# device: %s (%u EUs)\n", ctx.name().c_str(), ctx.eu_count());
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  eng.prepare_prefill();
  {
    Timer t;
    t.start();
    eng.prefill(ids);
    std::printf("# prefill %u ids: %.0f ms, pos %u\n", deepest, t.ms(), eng.pos());
  }
  runtime::DecodeBuffers& b = eng.buffers();
  runtime::Control* c = b.control.as<runtime::Control>();
  runtime::CapturedStep plain = runtime::build(ctx, eng.model(), b);
  runtime::ProfileEvents prof(ctx);
  runtime::CapturedStep instr = runtime::build(ctx, eng.model(), b, nullptr, &prof);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  auto run = [&](runtime::CapturedStep& s) {
    q.execute(s.list, &fence);
    fence.wait();
  };

  // ---- P0: the profiled list at each depth ----------------------------------
  std::printf("\n## P0: per-kernel time, max_len 131072, M = 1 (profiled list; Σ over the 16 "
              "FA layers, mean of %u steps; pos rewound before every step)\n\n", steps);
  std::printf("| depth | attn_prep us | attn_decode us | attn_reduce us | attn total ms | "
              "decode us/launch | decode GB/s (KV) | reduce share | step Σ ms | step wall "
              "ms (plain) | partial table MB/layer (derived) |\n|---|---|---|---|---|---|---|---|"
              "---|---|---|\n");
  for (uint32_t d : depths) {
    for (int w = 0; w < 2; ++w) {
      c->pos = d;
      c->n_active = 1;
      run(plain);
    }
    double prep = 0, dec = 0, red = 0, tot = 0;
    for (uint32_t s = 0; s < steps; ++s) {
      c->pos = d;
      c->n_active = 1;
      for (l0::Event& e : prof.events) e.reset();
      run(instr);
      for (size_t k = 0; k < instr.kernel_count; ++k) {
        const double us = prof.events[k].duration_us();
        const std::string& l = instr.labels[k];
        tot += us;
        if (l.find(" attn_prep") != std::string::npos) prep += us;
        else if (l.find(" attn_decode") != std::string::npos) dec += us;
        else if (l.find(" attn_reduce") != std::string::npos) red += us;
      }
    }
    prep /= steps; dec /= steps; red /= steps; tot /= steps;
    std::vector<double> walls;
    for (int s = 0; s < 8; ++s) {
      c->pos = d;
      c->n_active = 1;
      Timer t;
      t.start();
      run(plain);
      walls.push_back(t.ms());
    }
    const double kv_bytes = double(d + 1) * kKvH * kHD * 2 * 2;   // one layer, K + V
    const double part_mb = 24.0 * double((d + 64) / 64) * 258 * 4 / 1e6;
    std::printf("| %u | %.1f | %.1f | %.1f | %.3f | %.1f | %.0f | %.1f%% | %.3f | %.3f | %.2f |\n",
                d, prep, dec, red, (prep + dec + red) / 1e3, dec / 16, kv_bytes / (dec / 16) / 1e3,
                100.0 * red / (prep + dec + red), tot / 1e3, median(walls), part_mb);
    std::fflush(stdout);
  }

  // ---- the inputs: 4 plain steps from each depth -----------------------------
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  for (uint32_t d : depths) {
    std::vector<float> qv(4 * kRow), gv(4 * kRow);
    std::vector<uint16_t> ov(4 * kRow);
    c->pos = d;
    c->n_active = 1;
    for (uint32_t m = 0; m < 4; ++m) {
      run(plain);
      imm.copy(qv.data() + m * kRow, b.attn_q.ptr(), kRow * 4);
      imm.copy(gv.data() + m * kRow, b.attn_gate.ptr(), kRow * 4);
      imm.copy(ov.data() + m * kRow, b.attn_out.ptr(), kRow * 2);
    }
    if (c->pos != d + 4) throw std::runtime_error("pos did not advance by 4");
    const std::string sfx = "_" + std::to_string(d) + ".bin";
    write_file(out + "/q" + sfx, qv.data(), qv.size() * 4);
    write_file(out + "/g" + sfx, gv.data(), gv.size() * 4);
    write_file(out + "/o" + sfx, ov.data(), ov.size() * 2);
  }
  const size_t rows = deepest + 4, bytes = rows * kKvH * kHD * 2;
  const size_t layer = size_t(kMaxLen) * kKvH * kHD * 2;
  std::vector<uint16_t> kv(rows * kKvH * kHD);
  download(ctx, kv.data(), static_cast<char*>(b.kv_k.ptr()) + kFa * layer, bytes);
  write_file(out + "/kv_k.bin", kv.data(), bytes);
  download(ctx, kv.data(), static_cast<char*>(b.kv_v.ptr()) + kFa * layer, bytes);
  write_file(out + "/kv_v.bin", kv.data(), bytes);
  std::printf("\n# wrote q/g/o per depth and FA layer %u's KV rows [0, %zu) to %s\n", kFa, rows,
              out.c_str());
  return 0;
}

// ============================================================================
// arms: P1
// ============================================================================
struct ArmDef {
  std::string tag;   // "v1" or P<ppw>_L<load>_D<dot>_F<pf>_R<red>
  uint32_t ppw = 64, load = 0, dot = 0, pf = 0, red = 0;
  bool v1 = false;
};
ArmDef parse_arm(const std::string& t) {
  ArmDef a;
  a.tag = t;
  if (t == "v1") {
    a.v1 = true;
    return a;
  }
  if (std::sscanf(t.c_str(), "P%u_L%u_D%u_F%u_R%u", &a.ppw, &a.load, &a.dot, &a.pf, &a.red) != 5)
    throw std::runtime_error("bad arm tag " + t);
  return a;
}

struct Arm {
  ArmDef def;
  std::unique_ptr<l0::Module> mod;
  std::unique_ptr<l0::Kernel> dec, red;
  std::unique_ptr<l0::Mem> out;
  l0::CmdList once, timed;
  std::vector<l0::Event> ev;
  uint32_t reps = 0;
};

int run_arms(int argc, char** argv) {
  if (argc < 3) throw std::runtime_error("arms <outdir> --M m ...");
  const std::string dir = argv[2];
  uint32_t M = 1, tgt = 64, rounds = 5, reps = 10;
  std::vector<uint32_t> depths = {32768, 130816};
  std::vector<std::string> tags;
  bool timing = false;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error(a + " needs a value");
      return argv[i];
    };
    if (a == "--M") M = uint32_t(std::stoul(val()));
    else if (a == "--depths") depths = parse_list(val());
    else if (a == "--arms") tags = parse_tags(val());
    else if (a == "--tgt") tgt = uint32_t(std::stoul(val()));
    else if (a == "--rounds") rounds = uint32_t(std::stoul(val()));
    else if (a == "--reps") reps = uint32_t(std::stoul(val()));
    else if (a == "--time") timing = true;
    else throw std::runtime_error("unknown flag " + a);
  }
  const uint32_t deepest = *std::max_element(depths.begin(), depths.end());
  std::vector<ArmDef> defs = {parse_arm("v1")};
  for (const std::string& t : tags) defs.push_back(parse_arm(t));

  l0::Context ctx(0);
  const char* aff = std::getenv("ZE_AFFINITY_MASK");
  std::printf("# probe_decode_attn arms: M %u, tgt %u, rounds %u x %u reps, ZE_AFFINITY_MASK=%s, "
              "device %s\n", M, tgt, rounds, reps, aff ? aff : "(unset)", ctx.name().c_str());

  // KV: the dumped rows, the rest of the 131072 zero.
  const size_t layer = size_t(kMaxLen) * kKvH * kHD * 2;
  l0::Mem kk(ctx, l0::MemKind::Device, layer), vv(ctx, l0::MemKind::Device, layer);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.fill(kk.ptr(), 0u, layer);
  imm.fill(vv.ptr(), 0u, layer);
  {
    const size_t rows = deepest + M, bytes = rows * kKvH * kHD * 2;
    std::vector<uint16_t> h = read_file<uint16_t>(dir + "/kv_k.bin", rows * kKvH * kHD);
    upload(ctx, kk.ptr(), h.data(), bytes);
    h = read_file<uint16_t>(dir + "/kv_v.bin", rows * kKvH * kHD);
    upload(ctx, vv.ptr(), h.data(), bytes);
  }
  l0::Mem ctrl(ctx, l0::MemKind::Shared, 128);
  uint32_t* cw = static_cast<uint32_t*>(ctrl.ptr());
  std::memset(cw, 0, 128);
  l0::Mem aq(ctx, l0::MemKind::Device, size_t(M) * kRow * 4);
  l0::Mem ag(ctx, l0::MemKind::Device, size_t(M) * kRow * 4);
  const size_t part_bytes = size_t(24) * (kMaxLen / 64) * M * 258 * 4;
  l0::Mem part(ctx, l0::MemKind::Device, part_bytes);
  imm.fill(part.ptr(), 0u, part_bytes);
  l0::Mem counters(ctx, l0::MemKind::Device, 64);
  imm.fill(counters.ptr(), 0u, 64);
  l0::Mem dbg(ctx, l0::MemKind::Shared, 64);
  const size_t out_bytes = size_t(M) * kRow * 2;

  std::vector<Arm> arms;
  for (const ArmDef& d : defs) {
    const std::string bin = std::string(B70_KERNEL_DIR) + "/pda_" + d.tag + "_M" + std::to_string(M) + ".bin";
    if (!std::ifstream(bin).good()) {
      std::printf("# %s: NO BINARY (%s), skipped\n", d.tag.c_str(), bin.c_str());
      continue;
    }
    Arm a{d, std::make_unique<l0::Module>(ctx, bin), nullptr, nullptr,
          std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, out_bytes),
          l0::CmdList::regular(ctx), l0::CmdList::regular(ctx), {}, reps};
    a.dec = std::make_unique<l0::Kernel>(*a.mod, d.v1 ? "attn_decode" : "pda_decode");
    a.dec->group_size(256);
    a.dec->arg_ptr(0, ctrl.ptr());
    a.dec->arg_ptr(1, aq.ptr());
    a.dec->arg_ptr(2, kk.ptr());
    a.dec->arg_ptr(3, vv.ptr());
    a.dec->arg_ptr(4, part.ptr());
    if (!d.v1) {
      a.dec->arg_ptr(5, ag.ptr());
      a.dec->arg_ptr(6, a.out->ptr());
      a.dec->arg_ptr(7, counters.ptr());
    }
    const bool has_red = d.v1 || d.red == 0;
    if (has_red) {
      a.red = std::make_unique<l0::Kernel>(*a.mod, d.v1 ? "attn_reduce" : "pda_reduce");
      a.red->group_size(256);
      a.red->arg_ptr(0, ctrl.ptr());
      a.red->arg_ptr(1, part.ptr());
      a.red->arg_ptr(2, ag.ptr());
      a.red->arg_ptr(3, a.out->ptr());
      if (!d.v1) a.red->arg_ptr(4, dbg.ptr());
    }
    const uint32_t gy = d.v1 ? kMaxLen / 64 : kMaxLen / (d.ppw ? d.ppw : 64);
    a.once.launch(*a.dec, kKvH, gy);
    if (has_red) a.once.launch(*a.red, kQH, M);
    a.once.close();
    arms.push_back(std::move(a));
  }
  // Timed lists need events: one pool for all arms.
  const uint32_t per_arm = 2 * reps;
  l0::EventPool pool(ctx, uint32_t(arms.size()) * per_arm);
  {
    uint32_t idx = 0;
    for (Arm& a : arms) {
      const uint32_t gy = a.def.v1 ? kMaxLen / 64 : kMaxLen / (a.def.ppw ? a.def.ppw : 64);
      for (uint32_t r = 0; r < reps; ++r) {
        a.ev.emplace_back(pool, idx++);
        a.timed.launch(*a.dec, kKvH, gy, 1, &a.ev.back());
        if (a.red) {
          a.ev.emplace_back(pool, idx++);
          a.timed.launch(*a.red, kQH, M, 1, &a.ev.back());
        }
      }
      a.timed.close();
    }
  }
  l0::Queue q(ctx);
  l0::Fence fence(q);
  auto exec = [&](l0::CmdList& l) {
    q.execute(l, &fence);
    fence.wait();
  };
  bool all_ok = true;

  for (uint32_t d : depths) {
    std::vector<float> qv = read_file<float>(dir + "/q_" + std::to_string(d) + ".bin", 4 * kRow);
    std::vector<float> gv = read_file<float>(dir + "/g_" + std::to_string(d) + ".bin", 4 * kRow);
    std::vector<uint16_t> prod =
        read_file<uint16_t>(dir + "/o_" + std::to_string(d) + ".bin", 4 * kRow);
    upload(ctx, aq.ptr(), qv.data(), size_t(M) * kRow * 4);
    upload(ctx, ag.ptr(), gv.data(), size_t(M) * kRow * 4);
    cw[0] = d;          // pos
    cw[1] = M;          // n_active
    cw[kCtrlTgt] = tgt;

    std::printf("\n## depth %u, M %u (pos %u, rows %u..%u)\n\n", d, M, d, d, d + M - 1);
    std::printf("| arm | worst cos (h, m) | max abs | bitwise = arm 0 | repeatable | nb "
                "(partials/row) | partial MB written+read per layer |\n|---|---|---|---|---|---|---|\n");
    std::vector<uint16_t> ref(size_t(M) * kRow);
    std::map<std::string, bool> ok;
    for (Arm& a : arms) {
      std::vector<uint16_t> o1(size_t(M) * kRow), o2(size_t(M) * kRow);
      static_cast<uint32_t*>(dbg.ptr())[0] = 0;
      exec(a.once);
      download(ctx, o1.data(), a.out->ptr(), out_bytes);
      const uint32_t nb_dbg = static_cast<uint32_t*>(dbg.ptr())[0];
      imm.fill(a.out->ptr(), 0u, out_bytes);
      exec(a.once);
      download(ctx, o2.data(), a.out->ptr(), out_bytes);
      const bool rep = o1 == o2;
      if (a.def.v1) {
        ref = o1;
        const bool same_prod = std::memcmp(o1.data(), prod.data(), out_bytes) == 0;
        std::printf("| v1 (arm 0) | - | - | production dump: %s | %s | %u | %.2f |\n",
                    same_prod ? "identical" : "DIFFERS", rep ? "yes" : "NO", (d + 64) / 64,
                    2.0 * 24 * ((d + 64) / 64) * M * 258 * 4 / 1e6);
        ok[a.def.tag] = rep;
        all_ok = all_ok && rep;
        continue;
      }
      double worst = 2.0, maxabs = 0.0;
      uint32_t wh = 0, wm = 0;
      bool finite = true;
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t h = 0; h < kQH; ++h) {
          double xy = 0, xx = 0, yy = 0;
          for (uint32_t i = 0; i < kHD; ++i) {
            const size_t k = size_t(m) * kRow + h * kHD + i;
            const double x = bf16f(o1[k]), y = bf16f(ref[k]);
            if (!std::isfinite(x)) finite = false;
            xy += x * y; xx += x * x; yy += y * y;
            maxabs = std::max(maxabs, std::fabs(x - y));
          }
          const double cs = xy / std::sqrt(xx * yy);
          if (!(cs >= worst)) { worst = cs; wh = h; wm = m; }
        }
      const bool pass = finite && worst >= 0.99999 && rep;
      ok[a.def.tag] = pass;
      all_ok = all_ok && pass;
      uint32_t nb = nb_dbg;
      if (a.def.red) {   // fused: derive from the kernel's own ppw rule
        uint32_t p = a.def.ppw;
        if (p == 0) { p = (d + M + tgt - 1) / tgt; p = std::max(64u, (p + 63u) & ~63u); }
        nb = d / p + 1;
      }
      std::printf("| %s | %.7f (%u, %u)%s | %.3g | %s | %s | %u | %.2f |\n", a.def.tag.c_str(),
                  worst, wh, wm, finite ? "" : " NONFINITE", maxabs, o1 == ref ? "yes" : "no",
                  rep ? "yes" : "NO", nb, 2.0 * 24 * nb * M * 258 * 4 / 1e6);
    }
    if (!timing) continue;

    // ---- interleaved timing ------------------------------------------------------
    std::vector<std::vector<double>> dec_us(arms.size()), red_us(arms.size()), sum_us(arms.size()),
        ratio(arms.size());
    for (Arm& a : arms) exec(a.timed);   // warm-up round
    for (uint32_t r = 0; r < rounds; ++r) {
      std::vector<double> this_sum(arms.size());
      for (size_t k0 = 0; k0 < arms.size(); ++k0) {
        const size_t k = (k0 + r) % arms.size();   // rotate the order per round
        Arm& a = arms[k];
        for (l0::Event& e : a.ev) e.reset();
        exec(a.timed);
        double dsum = 0, rsum = 0;
        for (size_t e = 0; e < a.ev.size(); ++e) {
          const bool is_red = a.red && (e % 2 == 1);
          (is_red ? rsum : dsum) += a.ev[e].duration_us();
        }
        dec_us[k].push_back(dsum / reps);
        red_us[k].push_back(rsum / reps);
        sum_us[k].push_back((dsum + rsum) / reps);
        this_sum[k] = (dsum + rsum) / reps;
      }
      for (size_t k = 0; k < arms.size(); ++k) ratio[k].push_back(this_sum[0] / this_sum[k]);
    }
    const double kv_bytes = double(d + M) * kKvH * kHD * 2 * 2;
    std::printf("\n| arm | decode us | reduce us | sum us (range) | x arm 0 (paired median) | "
                "KV GB/s | derived attn ms/token (16 layers) |\n|---|---|---|---|---|---|---|\n");
    for (size_t k = 0; k < arms.size(); ++k) {
      auto s = sum_us[k];
      std::sort(s.begin(), s.end());
      std::printf("| %s%s | %.1f | %.1f | %.1f (%.1f-%.1f) | %.3f | %.0f | %.3f |\n",
                  arms[k].def.tag.c_str(), ok[arms[k].def.tag] ? "" : " (FAILED A1)",
                  median(dec_us[k]), median(red_us[k]), median(sum_us[k]), s.front(), s.back(),
                  median(ratio[k]), kv_bytes / median(sum_us[k]) / 1e3,
                  16 * median(sum_us[k]) / 1e3);
    }
    std::fflush(stdout);
  }
  std::printf("\n# A1 %s\n", all_ok ? "PASS (every arm)" : "FAIL (see rows)");
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: probe_decode_attn capture|arms ...\n");
      return 2;
    }
    const std::string mode = argv[1];
    if (mode == "capture") return run_capture(argc, argv);
    if (mode == "arms") return run_arms(argc, argv);
    throw std::runtime_error("unknown mode " + mode);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_decode_attn: %s\n", e.what());
    return 1;
  }
}

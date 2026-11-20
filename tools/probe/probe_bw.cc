// probe_bw: device-to-device read bandwidth with a plain L0 launch.
// Method (doc 01): 5 warm-ups, 20 timed iterations, buffers 2 GB and 8 GB.
// Timing wraps execute+fence; at >3 ms per kernel the ~20 us submit cost is <1%.
//
// The buffer is seeded with pseudo-random bytes, NOT a constant fill. The B70
// compresses device-local memory losslessly, so a buffer of identical 32-bit
// words costs almost no DRAM traffic and the probe reports ~1022 GB/s -- above
// the 608 GB/s theoretical peak. Random data is incompressible within a cache
// line, which is what model weights look like. See docs/01-hardware.md.
//
// `probe_bw --sweep` times the read forms of bw_sum.cl's BW_SWEEP / BW_LSC modules (PERF-111: what holds the
// read at ~590 GB/s against the 638 GB/s peak). Stages, each a table:
//   A  every form (load width, plain / block / 2D loads, loads in flight, SIMD16 / SIMD32), contiguous, both
//      address modes, 4096 sub-groups, work-group 256
//   L  the LSC cache-policy loads, the same grid (skipped when bw_sum_lsc did not build)
//   B  sub-group count 512..16384 x work-group 64..1024, for the base-equivalent form and stage A's best
//   C  the address pattern: contiguous against rows 4 / 16 / 64 KB and 64 KB + 256 B apart
//   P  base kernel against the best form as interleaved pairs (clock state hits both alike)
//   X  copy references: device -> device on the copy engine and on the compute queue
// Every point is the median of --iters runs after --warm runs, each run --reps launches in one list. Every form
// point's XOR is checked against the host's XOR of the bytes it covered; a mismatch prints "WRONG" and the point
// never counts as best. --file reads a real weight file region instead of random bytes (real bf16 weights may
// compress, which is the point of trying them). --point <kernel> <wg> <sub-groups> <pitch> <mode> times one.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <numeric>
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
// Fills `bytes` of device memory with incompressible data by staging one
// xorshift-filled host chunk and copying it repeatedly. Repetition at chunk
// granularity is harmless: compression works per cache line.
void seed_incompressible(l0::Context& ctx, l0::CmdList& imm, void* dst, size_t bytes) {
  const size_t chunk = 64ul << 20;
  l0::Mem host(ctx, l0::MemKind::Host, chunk);
  uint32_t* h = host.as<uint32_t>();
  uint32_t s = 0x9E3779B9u;
  for (size_t i = 0; i < chunk / 4; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    h[i] = s;
  }
  for (size_t off = 0; off < bytes; off += chunk)
    imm.copy(static_cast<char*>(dst) + off, h, std::min(chunk, bytes - off));
}

int run_default() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Module mod(ctx, kernels::path("bw_sum"));
  l0::Kernel k = mod.kernel("bw_sum");
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem out(ctx, l0::MemKind::Device, 1 << 20);

  const uint32_t wg = 256;
  const uint32_t groups = ctx.eu_count() * 8 * 16 / wg * 4;  // 4x oversubscribed threads
  std::printf("| buffer | GB/s (median of 20) | min | max |\n|---|---|---|---|\n");
  for (size_t gb : {2ul, 8ul}) {
    const size_t bytes = gb << 30;
    l0::Mem buf(ctx, l0::MemKind::Device, bytes);
    seed_incompressible(ctx, imm, buf.ptr(), bytes);
    const uint64_t n_vec = bytes / 16;
    k.arg_ptr(0, buf.ptr());
    k.arg(1, n_vec);
    k.arg_ptr(2, out.ptr());
    k.group_size(wg);
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, groups);
    list.close();
    std::vector<double> gbps;
    for (int it = 0; it < 25; ++it) {
      Timer t; t.start();
      q.execute(list, &fence);
      fence.wait();
      double ms = t.ms();
      if (it >= 5) gbps.push_back(double(bytes) / (ms * 1e6));
    }
    std::sort(gbps.begin(), gbps.end());
    std::printf("| %zu GB | %.0f | %.0f | %.0f |\n", gb, gbps[gbps.size() / 2], gbps.front(), gbps.back());
  }
  return 0;
}

// One read form of the sweep modules, named as bw_sum.cl names its kernels.
struct Form {
  char f;      // 'v' vector load, 'b' block read, 'd' 2D block read, 'l' LSC load at policy cc
  int cc;      // form l only
  uint32_t w;  // bytes a lane receives per load
  uint32_t u;  // loads in flight
  uint32_t simd;
  std::string name() const {
    char b[64];
    if (f == 'l') std::snprintf(b, sizeof b, "bw_l%d_w%u_u%u_s%u", cc, w, u, simd);
    else std::snprintf(b, sizeof b, "bw_%c_w%u_u%u_s%u", f, w, u, simd);
    return b;
  }
  uint32_t lr() const { return f == 'd' ? w / 4 : 1; }      // rows one load covers
  uint32_t lw() const { return f == 'd' ? 64 : simd * w; }  // bytes of each row it covers
};

// The forms bw_sum.cl instantiates; keep in step with its BW_V / BW_B / BW_D / BW_L lines.
std::vector<Form> sweep_forms() {
  std::vector<Form> v;
  for (char f : {'v', 'b'})
    for (uint32_t simd : {16u, 32u})
      for (uint32_t w : {16u, 32u, 64u, 128u})
        for (uint32_t u : {1u, 2u, 4u, 8u})
          if (w * u <= (simd == 16 ? 256u : 128u)) v.push_back({f, 0, w, u, simd});
  for (uint32_t w : {32u, 64u, 128u})
    for (uint32_t u : {1u, 2u, 4u, 8u})
      if (w * u <= 256) v.push_back({'d', 0, w, u, 16});
  return v;
}
std::vector<Form> lsc_forms() {
  std::vector<Form> v;
  for (int cc : {0, 1, 2, 3, 5})
    for (uint32_t u : {1u, 4u})
      for (uint32_t w : {16u, 32u}) v.push_back({'l', cc, w, u, 16});
  return v;
}

struct Point {
  Form form;
  uint32_t wg = 256;
  uint32_t sgs = 4096;  // target sub-group count
  uint64_t pitch = 0;   // 0: contiguous
  uint32_t mode = 0;
};

struct Result {
  bool ran = false;
  std::string why;  // why it did not run
  uint32_t sgs = 0;
  uint64_t covered = 0;
  double med = 0, lo = 0, hi = 0;
  int check = -1;  // 1 bytes match, 0 WRONG, -1 not checked
  std::vector<double> runs;
};

struct Opts {
  size_t gb = 2;
  std::string file;
  uint64_t offset = 0;
  int warm = 5, iters = 20, reps = 1;
  std::string stages = "ALBCPX";
};

class Sweep {
 public:
  explicit Sweep(const Opts& o)
      : o_(o), ctx_(0), q_(ctx_), fence_(q_), imm_(l0::CmdList::immediate(ctx_)),
        bytes_(o.gb << 30), buf_(ctx_, l0::MemKind::Device, bytes_), out_(ctx_, l0::MemKind::Device, 1 << 20),
        out_host_(ctx_, l0::MemKind::Host, 1 << 20) {
    seed();
    sweep_ = std::make_unique<l0::Module>(ctx_, kernels::path("bw_sum_sweep"));
    try {
      lsc_ = std::make_unique<l0::Module>(ctx_, kernels::path("bw_sum_lsc"));
    } catch (const std::exception& e) {
      std::printf("LSC forms unavailable (%s); stage L skipped\n", e.what());
    }
  }

  int run() {
    std::printf("buffer %zu GB (%s), %d warm-up + %d timed runs a point, %d launch%s a run\n\n", o_.gb,
                o_.file.empty() ? "random bytes" : o_.file.c_str(), o_.warm, o_.iters, o_.reps,
                o_.reps == 1 ? "" : "es");
    base_ = run_base();
    std::printf("base kernel bw_sum (wg 256, %u groups): %.1f GB/s (min %.1f, max %.1f)\n\n", base_groups(),
                base_.med, base_.lo, base_.hi);
    const Form vb{'v', 0, 16, 1, 16};  // the base kernel's own read: one uint4 a lane, grid stride
    Point best_pt{vb, 256, 4096, 0, 1};
    Result best = point(best_pt);
    auto consider = [&](const Point& p, const Result& r) {
      if (r.ran && r.check == 1 && r.med > best.med) { best = r; best_pt = p; }
    };
    if (has('A')) {
      header("A: read forms, contiguous");
      for (const Form& f : sweep_forms())
        for (uint32_t mode : {0u, 1u}) {
          Point p{f, 256, 4096, 0, mode};
          Result r = point(p);
          row(p, r);
          consider(p, r);
        }
    }
    if (has('L') && lsc_) {
      header("L: LSC cache-policy loads (cc 1 L1UC_L3UC, 2 L1UC_L3C, 3 L1C_L3UC, 5 L1S_L3UC; 0 default = control)");
      for (const Form& f : lsc_forms())
        for (uint32_t mode : {0u, 1u}) {
          Point p{f, 256, 4096, 0, mode};
          Result r = point(p);
          row(p, r);
          consider(p, r);
        }
    }
    if (has('B')) {
      header("B: sub-groups x work-group");
      std::vector<Point> seeds{Point{vb, 256, 4096, 0, 1}};
      if (best_pt.form.name() != vb.name() || best_pt.mode != 1) seeds.push_back(best_pt);
      for (const Point& s : seeds)
        for (uint32_t sgs : {512u, 1024u, 2048u, 4096u, 8192u, 16384u})
          for (uint32_t wg : {64u, 128u, 256u, 512u, 1024u}) {
            if (wg < s.form.simd) continue;
            Point p = s;
            p.sgs = sgs;
            p.wg = wg;
            Result r = point(p);
            row(p, r);
            consider(p, r);
          }
    }
    if (has('C')) {
      header("C: address pattern (pitch = bytes between the rows a sub-group walks)");
      std::vector<Form> forms{vb, {'b', 0, 32, 4, 16}, {'d', 0, 32, 1, 16}, {'d', 0, 128, 1, 16}};
      if (std::none_of(forms.begin(), forms.end(), [&](const Form& f) { return f.name() == best_pt.form.name(); }))
        forms.push_back(best_pt.form);
      for (const Form& f : forms)
        for (uint64_t pitch : {0ul, 4096ul, 16384ul, 65536ul, 65536ul + 256})
          for (uint32_t mode : {0u, 1u}) {
            Point p{f, best_pt.form.name() == f.name() ? best_pt.wg : 256u,
                    best_pt.form.name() == f.name() ? best_pt.sgs : 4096u, pitch, mode};
            Result r = point(p);
            row(p, r);
            if (pitch == 0) consider(p, r);
          }
    }
    std::printf("\nbest form: %s wg %u, %u sub-groups, mode %u: %.1f GB/s = %.1f %% of 638, %+.1f %% against the "
                "base kernel's %.1f\n",
                best_pt.form.name().c_str(), best_pt.wg, best.sgs, best_pt.mode, best.med, 100.0 * best.med / 638.0,
                100.0 * (best.med / base_.med - 1.0), base_.med);
    if (has('P')) pairs(best_pt);
    if (has('X')) copies();
    return 0;
  }

  int run_point(const Point& p) {
    Result r = point(p);
    header("point");
    row(p, r);
    return r.ran && r.check == 1 ? 0 : 1;
  }

 private:
  bool has(char c) const { return o_.stages.find(c) != std::string::npos; }

  // Random words, or a file region, into a host mirror (the XOR check reads it) and then the device.
  void seed() {
    mirror_.resize(bytes_ / 4);
    if (o_.file.empty()) {
      uint32_t s = 0x9E3779B9u;
      for (uint32_t& w : mirror_) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        w = s;
      }
    } else {
      std::ifstream f(o_.file, std::ios::binary);
      f.seekg(std::streamoff(o_.offset));
      f.read(reinterpret_cast<char*>(mirror_.data()), std::streamsize(bytes_));
      if (size_t(f.gcount()) != bytes_) {
        std::fprintf(stderr, "probe_bw: %s holds fewer than %zu bytes past offset %llu\n", o_.file.c_str(), bytes_,
                     (unsigned long long)o_.offset);
        std::exit(2);
      }
    }
    const size_t chunk = 64ul << 20;
    l0::Mem host(ctx_, l0::MemKind::Host, chunk);
    for (size_t off = 0; off < bytes_; off += chunk) {
      const size_t n = std::min(chunk, bytes_ - off);
      std::memcpy(host.ptr(), reinterpret_cast<const char*>(mirror_.data()) + off, n);
      imm_.copy(static_cast<char*>(buf_.ptr()) + off, host.ptr(), n);
    }
    block_xor_.assign(mirror_.size() / kBlockWords + 1, 0u);
    for (size_t i = 0; i < mirror_.size(); ++i) block_xor_[i / kBlockWords + 1] ^= mirror_[i];
    for (size_t b = 1; b < block_xor_.size(); ++b) block_xor_[b] ^= block_xor_[b - 1];  // prefix
  }
  uint32_t expected_xor(uint64_t covered) const {
    const size_t words = covered / 4, b = words / kBlockWords;
    uint32_t x = block_xor_[b];
    for (size_t i = b * kBlockWords; i < words; ++i) x ^= mirror_[i];
    return x;
  }

  // Median / min / max GB/s of `iters` timed submissions of `list` after `warm`.
  Result time(l0::CmdList& list, uint64_t bytes_per_run) {
    Result r;
    for (int it = 0; it < o_.warm + o_.iters; ++it) {
      Timer t; t.start();
      q_.execute(list, &fence_);
      fence_.wait();
      const double ms = t.ms();
      if (it >= o_.warm) r.runs.push_back(double(bytes_per_run) / (ms * 1e6));
    }
    std::vector<double> s = r.runs;
    std::sort(s.begin(), s.end());
    r.ran = true;
    r.med = s[s.size() / 2];
    r.lo = s.front();
    r.hi = s.back();
    return r;
  }

  uint32_t base_groups() const { return ctx_.eu_count() * 8 * 16 / 256 * 4; }
  std::unique_ptr<l0::CmdList> base_list() {
    if (!base_k_) {
      base_mod_ = std::make_unique<l0::Module>(ctx_, kernels::path("bw_sum"));
      base_k_ = std::make_unique<l0::Kernel>(base_mod_->kernel("bw_sum"));
      base_k_->arg_ptr(0, buf_.ptr());
      base_k_->arg(1, uint64_t(bytes_ / 16));
      base_k_->arg_ptr(2, out_.ptr());
      base_k_->group_size(256);
    }
    auto list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    for (int i = 0; i < o_.reps; ++i) list->launch(*base_k_, base_groups());
    list->close();
    return list;
  }
  Result run_base() {
    auto list = base_list();
    Result r = time(*list, bytes_ * o_.reps);
    r.covered = bytes_;
    return r;
  }

  // The kernel, grid and argument values of one point, or why it cannot run.
  struct Prepared {
    std::unique_ptr<l0::Kernel> k;
    std::unique_ptr<l0::CmdList> list;
    uint32_t sgs = 0;
    uint64_t covered = 0;
    std::string why;
  };
  Prepared prepare(const Point& p) {
    Prepared pr;
    const Form& f = p.form;
    l0::Module* m = f.f == 'l' ? lsc_.get() : sweep_.get();
    if (!m) { pr.why = "module not built"; return pr; }
    const uint64_t pitch = p.pitch ? p.pitch : f.lw();
    if (pitch % f.lw()) { pr.why = "pitch not a multiple of the load's row"; return pr; }
    if (p.wg % f.simd || p.wg > 1024) { pr.why = "work-group not a multiple of the SIMD width"; return pr; }
    const uint32_t cols = uint32_t(pitch / f.lw()), sgpw = p.wg / f.simd;
    const uint32_t step = sgpw / std::gcd(cols, sgpw);
    uint32_t S = std::max(1u, (p.sgs + cols / 2) / cols);
    S = std::max(step, (S + step / 2) / step * step);
    const uint64_t per_k = uint64_t(S) * f.u * f.lr() * pitch;
    const uint32_t k = uint32_t(bytes_ / per_k);
    pr.sgs = cols * S;
    if (k == 0) { pr.why = "buffer smaller than one pass"; return pr; }
    if (uint64_t(pr.sgs) * 4 > (1u << 20)) { pr.why = "too many sub-groups for the result buffer"; return pr; }
    pr.covered = per_k * k;
    pr.k = std::make_unique<l0::Kernel>(m->kernel(f.name().c_str()));
    pr.k->arg_ptr(0, buf_.ptr());
    pr.k->arg(1, pitch);
    pr.k->arg(2, cols);
    pr.k->arg(3, S);
    pr.k->arg(4, k);
    pr.k->arg(5, p.mode);
    pr.k->arg_ptr(6, out_.ptr());
    pr.k->group_size(p.wg);
    pr.list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    for (int i = 0; i < o_.reps; ++i) pr.list->launch(*pr.k, pr.sgs / sgpw);
    pr.list->close();
    return pr;
  }
  int check(const Prepared& pr) {
    imm_.copy(out_host_.ptr(), out_.ptr(), size_t(pr.sgs) * 4);
    const uint32_t* h = out_host_.as<uint32_t>();
    uint32_t x = 0;
    for (uint32_t i = 0; i < pr.sgs; ++i) x ^= h[i];
    return x == expected_xor(pr.covered) ? 1 : 0;
  }
  Result point(const Point& p) {
    Prepared pr = prepare(p);
    if (!pr.list) { Result r; r.why = pr.why; r.sgs = pr.sgs; return r; }
    Result r = time(*pr.list, pr.covered * o_.reps);
    r.sgs = pr.sgs;
    r.covered = pr.covered;
    r.check = check(pr);
    return r;
  }

  void header(const char* title) {
    std::printf("\n%s\n\n| kernel | form | W | U | SIMD | wg | sub-groups | pitch | mode | GB/s | min | max | "
                "bytes |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
                title);
  }
  void row(const Point& p, const Result& r) {
    const Form& f = p.form;
    char pitch[32];
    if (p.pitch) std::snprintf(pitch, sizeof pitch, "%llu", (unsigned long long)p.pitch);
    else std::snprintf(pitch, sizeof pitch, "contig");
    char form[16];
    if (f.f == 'l') std::snprintf(form, sizeof form, "l%d", f.cc);
    else std::snprintf(form, sizeof form, "%c", f.f);
    if (!r.ran) {
      std::printf("| %s | %s | %u | %u | %u | %u | %u | %s | %u | - | - | - | %s |\n", f.name().c_str(), form, f.w,
                  f.u, f.simd, p.wg, r.sgs, pitch, p.mode, r.why.c_str());
      return;
    }
    std::printf("| %s | %s | %u | %u | %u | %u | %u | %s | %u | %.1f | %.1f | %.1f | %s |\n", f.name().c_str(), form,
                f.w, f.u, f.simd, p.wg, r.sgs, pitch, p.mode, r.med, r.lo, r.hi, r.check == 1 ? "ok" : "WRONG");
    std::fflush(stdout);
  }

  // Base kernel and the best form alternated run by run, so clock and thermal state land on both alike.
  void pairs(const Point& best_pt) {
    Prepared pr = prepare(best_pt);
    if (!pr.list) return;
    auto base = base_list();
    std::vector<double> a, b;
    for (int it = 0; it < o_.warm + o_.iters; ++it) {
      for (int side = 0; side < 2; ++side) {
        Timer t; t.start();
        q_.execute(side ? *pr.list : *base, &fence_);
        fence_.wait();
        const double ms = t.ms();
        if (it >= o_.warm)
          (side ? b : a).push_back(double(side ? pr.covered : bytes_) * o_.reps / (ms * 1e6));
      }
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    const double ma = a[a.size() / 2], mb = b[b.size() / 2];
    std::printf("\nP: interleaved pairs, %d each: base %.1f GB/s, %s %.1f GB/s (%+.2f %%)\n", o_.iters, ma,
                best_pt.form.name().c_str(), mb, 100.0 * (mb / ma - 1.0));
  }

  // Device -> device copies of half the buffer into a second allocation: the copy engine (immediate list) and
  // the compute queue (regular list). Read GB/s counts the source bytes; traffic counts read + write.
  void copies() {
    const size_t n = bytes_ / 2;
    l0::Mem dst(ctx_, l0::MemKind::Device, n);
    std::vector<double> ce;
    l0::CmdList bcs = l0::CmdList::immediate_copy(ctx_);
    for (int it = 0; it < o_.warm + o_.iters; ++it) {
      Timer t; t.start();
      for (int i = 0; i < o_.reps; ++i) bcs.copy(dst.ptr(), buf_.ptr(), n);
      bcs.host_synchronize();
      const double ms = t.ms();
      if (it >= o_.warm) ce.push_back(double(n) * o_.reps / (ms * 1e6));
    }
    l0::CmdList list = l0::CmdList::regular(ctx_);
    for (int i = 0; i < o_.reps; ++i) list.copy(dst.ptr(), buf_.ptr(), n);
    list.close();
    Result cq = time(list, uint64_t(n) * o_.reps);
    std::sort(ce.begin(), ce.end());
    const double mce = ce[ce.size() / 2];
    std::printf("\nX: copy %zu MB device -> device\n\n| engine | read GB/s | read + write GB/s |\n|---|---|---|\n",
                n >> 20);
    std::printf("| copy engine | %.1f | %.1f |\n| compute queue | %.1f | %.1f |\n", mce, 2 * mce, cq.med, 2 * cq.med);
  }

  static constexpr size_t kBlockWords = 1u << 18;  // 1 MiB of words per prefix-XOR block
  Opts o_;
  l0::Context ctx_;
  l0::Queue q_;
  l0::Fence fence_;
  l0::CmdList imm_;
  size_t bytes_;
  l0::Mem buf_, out_, out_host_;
  std::vector<uint32_t> mirror_, block_xor_;
  std::unique_ptr<l0::Module> sweep_, lsc_, base_mod_;
  std::unique_ptr<l0::Kernel> base_k_;
  Result base_;
};

Form parse_form(const std::string& name) {
  Form f{};
  char c = 0;
  if (std::sscanf(name.c_str(), "bw_l%d_w%u_u%u_s%u", &f.cc, &f.w, &f.u, &f.simd) == 4) f.f = 'l';
  else if (std::sscanf(name.c_str(), "bw_%c_w%u_u%u_s%u", &c, &f.w, &f.u, &f.simd) == 4) f.f = c;
  else { std::fprintf(stderr, "probe_bw: %s is not a sweep kernel name\n", name.c_str()); std::exit(2); }
  return f;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) return run_default();
  Opts o;
  bool sweep = false, one = false;
  Point p;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "probe_bw: %s needs a value\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--sweep") sweep = true;
    else if (a == "--gb") o.gb = std::stoul(next());
    else if (a == "--file") o.file = next();
    else if (a == "--offset") o.offset = std::stoull(next());
    else if (a == "--warm") o.warm = std::stoi(next());
    else if (a == "--iters") o.iters = std::stoi(next());
    else if (a == "--reps") o.reps = std::stoi(next());
    else if (a == "--stages") o.stages = next();
    else if (a == "--point") {
      one = true;
      p.form = parse_form(next());
      p.wg = std::stoul(next());
      p.sgs = std::stoul(next());
      p.pitch = std::stoull(next());
      p.mode = std::stoul(next());
    } else {
      std::fprintf(stderr,
                   "usage: probe_bw                        2 and 8 GB through bw_sum\n"
                   "       probe_bw --sweep [--stages ALBCPX] [--gb N] [--file PATH [--offset BYTES]]\n"
                   "                        [--warm N] [--iters N] [--reps N]\n"
                   "       probe_bw --point <kernel> <wg> <sub-groups> <pitch, 0 contiguous> <mode 0|1> [options]\n");
      return 2;
    }
  }
  if (!sweep && !one) { std::fprintf(stderr, "probe_bw: pass --sweep or --point\n"); return 2; }
  Sweep s(o);
  return one ? s.run_point(p) : s.run();
}

// probe_verify_attn - spec 28c (plan 28c Task 3): the verify step's attention, today's v2 pair against
// attn_verify_dpas lever arms, on real q / KV at M rows: numerics (N1, N2a, N2b, repeatability) and
// interleaved paired timing.
//
// Plan: docs/superpowers/plans/2026-10-31-spec28c-verify-attn-build.md. Record: docs/probe-verify-attn-<date>.md.
//
//   probe_verify_attn arms <dump> [--m 2,3,4,5,6,7,8,16] [--depths 512,4096,32768,130816] [--kv bf16|int8]
//                         [--arms v2,pfa,<tags>] [--rounds 5] [--reps 10] [--time] [--device 0]
//                         [--trials 100] [--threads 8]
//
// <dump> is `probe_decode_attn capture --rows 8`'s directory: q_<d>.bin / g_<d>.bin (attn_q / attn_gate of R
// plain steps from depth d: the q of rows m = 0..R-1 of a verify at d) and FA layer 15's KV rows
// kv_k.bin / kv_v.bin [deepest + R][4][256] bf16. Under --kv int8 the cache and q are made on the host by
// src/common/kv8.h (tests/kernels/kv8_ref.h's Cache: K / V quantised as the writer stores them, q rotated) -
// bitwise what attn_prep_kv8 would have written. M = 16 reuses the R rows' q twice (positions d .. d + 15).
//
// Arms (per M, one list each):
//   v2      the production pair attn_v2_M<M>_T32 (decode + reduce); int8: attn_v2_M<M>_T32_KV8 at M <= 4 and the
//           probe-built control pvd_v2kv8_M<M> at M = 5..8 (kv8.cl's KV8_DEC under BATCH_ROWS - the decode
//           build stops at M = 4 - with a one-slot row table: row m at pos + m, kv base 0, which is the plain
//           decode). No v2 exists at M = 16: the DPAS arm is recorded alone there.
//   pfa     pf_flash_attn (pf_flash_attn_kv8) at C = M: bf16 q, exp2 - a reference, recorded only.
//   <tag>   attn_verify_dpas.cl at K<KT>_P<g|h>_S<KS>_D<DSPLIT>_Q<QSPLIT>_G<128|256> (pvd_<tag>_M<M>[_KV8])
//           followed by the v2 arm's own reduce binary (attn_reduce_v2[_kv8]); the binary's grf_count is
//           read from its zebin and an arm whose GRF mode is not its name's is refused.
// Per (arm, M, depth):
//   N1   the gated rows against tests/kernels/attn_verify_ref.h's fp64 reference: cosine per (q head, row),
//        worst and max abs, v2's on the same inputs beside it (bar 0.99999);
//   N2b  each row against v2's M = 1 row at d + m (cosine >= 0.99999, max abs);
//   rep  two runs bitwise.
// Per (tag, depth), over the --m list's M in 2..8: N2a (property R) - row 0 of depth d (q, gate) at a random slot
// s of a random M launch at pos d - s with the other rows random, --trials times: its gated row bitwise the
// same in every trial.
// --time: --rounds rounds of --reps launch pairs per arm, the arms interleaved in an order rotated per round after
// a warm-up round, per-kernel L0 timestamps; median (range) per arm and the median paired ratio v2 / arm with its
// range; the verdict line of plan 28c Task 4 (dpas when the paired median >= 1.03 and its low end > 1.00).
//
// PROBE-ONLY: no production file is edited.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/attn_verify_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {

namespace avr = attn_verify_ref;
constexpr uint32_t kMaxLen = 131072;
constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kRow = kQH * kHD;   // 6144
constexpr uint32_t kTgt = 32, kPart = 258, kMaxM = 16;
constexpr uint32_t kRowWords = 8;   // runtime::BatchRow (BROW_WORDS)

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  size_t st = 0;
  while (st < s.size()) {
    const size_t e = std::min(s.find(',', st), s.size());
    v.push_back(s.substr(st, e - st));
    st = e + 1;
  }
  return v;
}
std::vector<uint32_t> split_u(const std::string& s) {
  std::vector<uint32_t> v;
  for (const std::string& t : split(s)) v.push_back(uint32_t(std::stoul(t)));
  return v;
}
size_t file_size(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + p);
  return size_t(f.tellg());
}
template <class T>
std::vector<T> read_file(const std::string& p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
  if (!f) throw std::runtime_error("cannot read " + p);
  return v;
}
void upload(l0::Context& ctx, void* dev, const void* host, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(dev) + off, static_cast<const char*>(host) + off, std::min(kChunk, bytes - off));
  }
}
void download(l0::Context& ctx, void* host, const void* dev, size_t bytes) {
  l0::CmdList c = l0::CmdList::immediate(ctx);
  c.copy(host, dev, bytes);
}
bool exists(const std::string& p) { return bool(std::ifstream(p, std::ios::binary)); }

// The zebin's .ze_info carries `grf_count: <n>` per kernel (probe_small_m's reader). -1 when absent.
int grf_count(const std::string& bin, const std::string& kernel) {
  std::ifstream f(bin, std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t k = s.find("name: " + kernel);
  if (k == std::string::npos) k = 0;
  const size_t p = s.find("grf_count:", k);
  if (p == std::string::npos) return -1;
  return std::atoi(s.c_str() + p + 10);
}

// A lever tag K<KT>_P<g|h>_S<KS>_D<DSPLIT>_Q<QSPLIT>_G<128|256>.
struct Tag {
  std::string s;
  uint32_t kt = 0, ks = 0, dsplit = 0, q = 0, grf = 0;
  bool ph = false;
  uint32_t rb(uint32_t M) const { return ph ? 6 * ((M + 7) / 8) : (6 * M + 7) / 8; }
  uint32_t wg(uint32_t M) const { return 16 * rb(M) * (256 / dsplit) * ks; }
};
bool parse_tag(const std::string& t, Tag& a) {
  char p = 0;
  a.s = t;
  if (std::sscanf(t.c_str(), "K%u_P%c_S%u_D%u_Q%u_G%u", &a.kt, &p, &a.ks, &a.dsplit, &a.q, &a.grf) != 6) return false;
  if (p != 'g' && p != 'h') return false;
  a.ph = p == 'h';
  return true;
}

enum class Kind { V2, Pfa, Dpas };

struct Arm {
  std::string name;
  Kind kind = Kind::V2;
  Tag tag;
  uint32_t M = 0;
  std::unique_ptr<l0::Kernel> dec, red;
  uint32_t dgx = 0, dgy = 0, dgz = 1;
  std::unique_ptr<l0::Mem> out;   // gated bf16 [M][6144] (v2 / dpas with a reduce), pf_o fp32 [24][16][256] (pfa)
  std::unique_ptr<l0::CmdList> once, timed;
  std::vector<l0::Event> ev;
  bool has_red = false;
};

struct Ctx {
  l0::Context& ctx;
  bool kv8;
  std::map<std::string, std::unique_ptr<l0::Module>> mods;
  l0::Module* mod(const std::string& variant) {
    auto it = mods.find(variant);
    if (it != mods.end()) return it->second.get();
    const std::string p = kernels::path(variant);
    if (!exists(p)) return nullptr;
    auto m = std::make_unique<l0::Module>(ctx, p);
    l0::Module* r = m.get();
    mods[variant] = std::move(m);
    return r;
  }
};

// The host side of one depth: q rows (rotated under int8), gates, the fp64 reference per row (independent of M:
// row m's reference attends keys [0, d + m] whatever the launch), v2's M = 1 rows.
struct DepthData {
  uint32_t d = 0;
  std::vector<float> q, g;            // [16][6144]
  std::vector<uint16_t> q16;          // bf16 of q, for pfa
  std::vector<double> ref_o;          // [16][24][256] pre-gate fp64
  std::vector<uint16_t> ref_out;      // [16][6144] gated
  std::vector<uint16_t> m1;           // [16][6144] v2's M = 1 rows
  uint32_t nref = 0;                  // rows with a reference
};

double cos_rows(const uint16_t* a, const uint16_t* b, double* maxabs) {
  double xy = 0, xx = 0, yy = 0;
  for (uint32_t i = 0; i < kHD; ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    if (!std::isfinite(x)) return -2.0;
    xy += x * y;
    xx += x * x;
    yy += y * y;
    *maxabs = std::max(*maxabs, std::fabs(x - y));
  }
  if (xx == 0 || yy == 0) return xx == yy ? 1.0 : 0.0;
  return xy / std::sqrt(xx * yy);
}

uint16_t gate_bf16(float o, float g) {
  return common::f32_to_bf16(common::bf16_to_f32(common::f32_to_bf16(o)) * (1.0f / (1.0f + std::exp(-g))));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || std::string(argv[1]) != "arms") {
      std::fprintf(stderr, "usage: probe_verify_attn arms <dump> [--m ..] [--depths ..] [--kv bf16|int8] "
                           "[--arms v2,pfa,<tags>] [--rounds 5] [--reps 10] [--time] [--device 0] [--trials 100]\n");
      return 2;
    }
    const std::string dir = argv[2];
    std::vector<uint32_t> Ms = {2, 3, 4, 5, 6, 7, 8, 16}, depths = {512, 4096, 32768, 130816};
    std::vector<std::string> arm_names = {"v2", "pfa", "K32_Pg_S4_D256_Q1_G256"};
    uint32_t rounds = 5, reps = 10, device = 0, trials = 100, threads = 8;
    bool timing = false, kv8 = false;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (++i >= argc) throw std::runtime_error(a + " needs a value");
        return argv[i];
      };
      if (a == "--m") Ms = split_u(val());
      else if (a == "--depths") depths = split_u(val());
      else if (a == "--kv") {
        const std::string v = val();
        if (v != "bf16" && v != "int8") throw std::runtime_error("--kv is bf16 or int8");
        kv8 = v == "int8";
      } else if (a == "--arms") arm_names = split(val());
      else if (a == "--rounds") rounds = uint32_t(std::stoul(val()));
      else if (a == "--reps") reps = uint32_t(std::stoul(val()));
      else if (a == "--time") timing = true;
      else if (a == "--device") device = uint32_t(std::stoul(val()));
      else if (a == "--trials") trials = uint32_t(std::stoul(val()));
      else if (a == "--threads") threads = std::max(1u, uint32_t(std::stoul(val())));
      else throw std::runtime_error("unknown flag " + a);
    }
    for (uint32_t M : Ms)
      if (!((M >= 2 && M <= 8) || M == 16)) throw std::runtime_error("--m: 2..8 or 16");

    // ---- the dump ----------------------------------------------------------------------------
    const uint32_t deepest = *std::max_element(depths.begin(), depths.end());
    const size_t kv_rows = file_size(dir + "/kv_k.bin") / (size_t(kKvH) * kHD * 2);
    const uint32_t R = uint32_t(file_size(dir + "/q_" + std::to_string(deepest) + ".bin") / (size_t(kRow) * 4));
    if (R < 8) throw std::runtime_error("the dump holds " + std::to_string(R) + " rows; capture with --rows 8");
    if (kv_rows < size_t(deepest) + 8) throw std::runtime_error("the dump's KV stops before deepest + 8");
    l0::Context ctx(device);
    const char* aff = std::getenv("ZE_AFFINITY_MASK");
    std::printf("# probe_verify_attn arms: dump %s (%u rows, KV rows %zu), kv %s, M", dir.c_str(), R, kv_rows,
                kv8 ? "int8" : "bf16");
    for (uint32_t M : Ms) std::printf(" %u", M);
    std::printf(", depths");
    for (uint32_t d : depths) std::printf(" %u", d);
    std::printf(", rounds %u x %u reps%s, ZE_AFFINITY_MASK=%s, device %u %s\n", rounds, reps,
                timing ? " (timed)" : "", aff ? aff : "(unset)", device, ctx.name().c_str());

    std::vector<uint16_t> k16 = read_file<uint16_t>(dir + "/kv_k.bin", kv_rows * kKvH * kHD);
    std::vector<uint16_t> v16 = read_file<uint16_t>(dir + "/kv_v.bin", kv_rows * kKvH * kHD);
    std::unique_ptr<kv8_ref::Cache> c8;
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    const size_t layer16 = size_t(kMaxLen) * kKvH * kHD * 2;
    std::unique_ptr<l0::Mem> dk, dv, dks, dvs;
    if (kv8) {
      c8 = std::make_unique<kv8_ref::Cache>(kMaxLen);
      for (size_t row = 0; row < kv_rows * kKvH; ++row) c8->put(row, &k16[row * kHD], &v16[row * kHD]);
      dk = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->k.size());
      dv = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->v.size());
      dks = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->ks.size() * 2);
      dvs = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->vs.size() * 2);
      upload(ctx, dk->ptr(), c8->k.data(), c8->k.size());
      upload(ctx, dv->ptr(), c8->v.data(), c8->v.size());
      upload(ctx, dks->ptr(), c8->ks.data(), c8->ks.size() * 2);
      upload(ctx, dvs->ptr(), c8->vs.data(), c8->vs.size() * 2);
    } else {
      dk = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, layer16);
      dv = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, layer16);
      imm.fill(dk->ptr(), 0u, layer16);
      imm.fill(dv->ptr(), 0u, layer16);
      upload(ctx, dk->ptr(), k16.data(), k16.size() * 2);
      upload(ctx, dv->ptr(), v16.data(), v16.size() * 2);
    }
    const avr::Kv hkv = kv8 ? avr::Kv::int8(*c8) : avr::Kv::bf16(k16.data(), v16.data());

    l0::Mem ctrl(ctx, l0::MemKind::Shared, 256);
    uint32_t* cw = static_cast<uint32_t*>(ctrl.ptr());
    std::memset(cw, 0, 256);
    l0::Mem rows(ctx, l0::MemKind::Shared, size_t(kMaxM) * kRowWords * 4);
    uint32_t* rw = static_cast<uint32_t*>(rows.ptr());
    std::memset(rw, 0, size_t(kMaxM) * kRowWords * 4);
    auto set_pos = [&](uint32_t pos, uint32_t n) {
      cw[0] = pos;   // CTRL_POS
      cw[1] = n;     // CTRL_NACT
      for (uint32_t m = 0; m < kMaxM; ++m) {
        rw[m * kRowWords + 0] = pos + m;   // BROW_POS
        rw[m * kRowWords + 1] = 0;         // BROW_KVB
        rw[m * kRowWords + 4] = 0;         // BROW_SLOT
      }
    };
    l0::Mem aq(ctx, l0::MemKind::Device, size_t(kMaxM) * kRow * 4);
    l0::Mem ag(ctx, l0::MemKind::Device, size_t(kMaxM) * kRow * 4);
    l0::Mem aq16(ctx, l0::MemKind::Device, size_t(kMaxM) * kRow * 2);
    const size_t part_bytes = size_t(kQH) * kTgt * kMaxM * kPart * 4;
    l0::Mem part(ctx, l0::MemKind::Device, part_bytes);
    imm.fill(part.ptr(), 0u, part_bytes);
    constexpr uint32_t kPfRows = 16;
    const size_t pfo_bytes = size_t(kQH) * kPfRows * kHD * 4;

    // ---- the arms ------------------------------------------------------------------------------
    Ctx cx{ctx, kv8, {}};
    std::vector<Arm> arms;
    auto v2_variant = [&](uint32_t M) -> std::string {
      if (M > 8) return "";
      if (!kv8) return kernels::attn_v2_variant(M, kTgt);
      if (M <= 4) return kernels::attn_v2_kv8_variant(M, kTgt);
      return "pvd_v2kv8_M" + std::to_string(M);
    };
    auto bind_v2_dec = [&](l0::Kernel& k, bool rowsarg) {
      k.group_size(256);
      k.arg_ptr(0, ctrl.ptr());
      k.arg_ptr(1, aq.ptr());
      if (kv8) {
        k.arg_ptr(2, dk->ptr());
        k.arg_ptr(3, dks->ptr());
        k.arg_ptr(4, dv->ptr());
        k.arg_ptr(5, dvs->ptr());
        k.arg_ptr(6, part.ptr());
        if (rowsarg) k.arg_ptr(7, rows.ptr());
      } else {
        k.arg_ptr(2, dk->ptr());
        k.arg_ptr(3, dv->ptr());
        k.arg_ptr(4, part.ptr());
      }
    };
    auto make_red = [&](uint32_t M, l0::Mem& out) -> std::unique_ptr<l0::Kernel> {
      const std::string v = v2_variant(M);
      if (v.empty()) return nullptr;
      l0::Module* md = cx.mod(v);
      if (!md) return nullptr;
      auto k = std::make_unique<l0::Kernel>(*md, kv8 ? "attn_reduce_v2_kv8" : "attn_reduce_v2");
      k->group_size(256);
      k->arg_ptr(0, ctrl.ptr());
      k->arg_ptr(1, part.ptr());
      k->arg_ptr(2, ag.ptr());
      k->arg_ptr(3, out.ptr());
      if (kv8 && M > 4) k->arg_ptr(4, rows.ptr());
      return k;
    };
    for (uint32_t M : Ms) {
      for (const std::string& nm : arm_names) {
        Arm a;
        a.name = nm;
        a.M = M;
        if (nm == "v2") {
          a.kind = Kind::V2;
          const std::string v = v2_variant(M);
          l0::Module* md = v.empty() ? nullptr : cx.mod(v);
          if (!md) {
            std::printf("# v2 M=%u: no binary (%s), skipped\n", M, v.empty() ? "none exists at 16" : v.c_str());
            continue;
          }
          a.out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(M) * kRow * 2);
          a.dec = std::make_unique<l0::Kernel>(*md, kv8 ? "attn_decode_v2_kv8" : "attn_decode_v2");
          bind_v2_dec(*a.dec, kv8 && M > 4);
          a.dgx = kKvH;
          a.dgy = kTgt;
          a.red = make_red(M, *a.out);
          a.has_red = true;
        } else if (nm == "pfa") {
          a.kind = Kind::Pfa;
          const std::string v = kv8 ? "pf_flash_attn_kv8" : "pf_flash_attn";
          l0::Module* md = cx.mod(v);
          if (!md) {
            std::printf("# pfa: no binary %s, skipped\n", v.c_str());
            continue;
          }
          a.out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, pfo_bytes);
          a.dec = std::make_unique<l0::Kernel>(*md, v.c_str());
          a.dec->group_size(16 * 6);
          a.dgx = (M + 7) / 8;
          a.dgy = kKvH;
          uint32_t ai = 0;
          a.dec->arg_ptr(ai++, aq16.ptr());
          a.dec->arg_ptr(ai++, dk->ptr());
          if (kv8) a.dec->arg_ptr(ai++, dks->ptr());
          a.dec->arg_ptr(ai++, dv->ptr());
          if (kv8) a.dec->arg_ptr(ai++, dvs->ptr());
          a.dec->arg_ptr(ai++, a.out->ptr());
          a.dec->arg(ai++, uint32_t(0));   // pos, set per depth (the list is rebuilt per depth)
          a.dec->arg(ai++, M);
          a.dec->arg(ai++, kPfRows);
        } else {
          a.kind = Kind::Dpas;
          if (!parse_tag(nm, a.tag)) throw std::runtime_error("bad arm " + nm);
          const std::string v = "pvd_" + nm + "_M" + std::to_string(M) + (kv8 ? "_KV8" : "");
          l0::Module* md = cx.mod(v);
          if (!md) {
            std::printf("# %s M=%u: no binary %s, skipped\n", nm.c_str(), M, v.c_str());
            continue;
          }
          const std::string entry = kv8 ? "attn_verify_dpas_kv8" : "attn_verify_dpas";
          const int g = grf_count(kernels::path(v), entry);
          if (g != int(a.tag.grf)) {
            std::printf("# %s M=%u: REFUSED - the binary's grf_count is %d, its tag says %u\n", nm.c_str(), M, g,
                        a.tag.grf);
            continue;
          }
          a.out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(M) * kRow * 2);
          a.dec = std::make_unique<l0::Kernel>(*md, entry.c_str());
          a.dec->group_size(a.tag.wg(M));
          a.dec->arg_ptr(0, ctrl.ptr());
          a.dec->arg_ptr(1, aq.ptr());
          if (kv8) {
            a.dec->arg_ptr(2, dk->ptr());
            a.dec->arg_ptr(3, dks->ptr());
            a.dec->arg_ptr(4, dv->ptr());
            a.dec->arg_ptr(5, dvs->ptr());
            a.dec->arg_ptr(6, part.ptr());
          } else {
            a.dec->arg_ptr(2, dk->ptr());
            a.dec->arg_ptr(3, dv->ptr());
            a.dec->arg_ptr(4, part.ptr());
          }
          a.dgx = kKvH;
          a.dgy = kTgt;
          a.red = make_red(M, *a.out);
          a.has_red = a.red != nullptr;
        }
        arms.push_back(std::move(a));
      }
    }
    // v2 at M = 1 (N2b): one module, its own output.
    const std::string v1name = kv8 ? kernels::attn_v2_kv8_variant(1, kTgt) : kernels::attn_v2_variant(1, kTgt);
    l0::Module* m1mod = cx.mod(v1name);
    if (!m1mod) throw std::runtime_error("no " + v1name);
    l0::Mem m1out(ctx, l0::MemKind::Device, size_t(kRow) * 2);
    l0::Kernel m1dec(*m1mod, kv8 ? "attn_decode_v2_kv8" : "attn_decode_v2");
    bind_v2_dec(m1dec, false);
    l0::Kernel m1red(*m1mod, kv8 ? "attn_reduce_v2_kv8" : "attn_reduce_v2");
    m1red.group_size(256);
    m1red.arg_ptr(0, ctrl.ptr());
    m1red.arg_ptr(1, part.ptr());
    m1red.arg_ptr(2, ag.ptr());
    m1red.arg_ptr(3, m1out.ptr());
    l0::CmdList m1list = l0::CmdList::regular(ctx);
    m1list.launch(m1dec, kKvH, kTgt);
    m1list.launch(m1red, kQH, 1);
    m1list.close();

    l0::Queue q(ctx);
    l0::Fence fence(q);
    auto exec = [&](l0::CmdList& l) {
      q.execute(l, &fence);
      fence.wait();
    };
    // (Re)build an arm's lists - pfa bakes pos as an argument, so its lists are rebuilt per depth.
    std::unique_ptr<l0::EventPool> pool;
    auto build_lists = [&](uint32_t pos) {
      uint32_t per = 0;
      for (Arm& a : arms) per += reps * (a.red ? 2 : 1);
      for (Arm& a : arms) a.ev.clear();
      pool = std::make_unique<l0::EventPool>(ctx, std::max(1u, per));
      uint32_t idx = 0;
      for (Arm& a : arms) {
        if (a.kind == Kind::Pfa) a.dec->arg(kv8 ? 6u : 4u, pos);   // pf_flash_attn[_kv8]'s `pos`
        a.once = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx));
        a.once->launch(*a.dec, a.dgx, a.dgy, a.dgz);
        if (a.red) a.once->launch(*a.red, kQH, a.M);
        a.once->close();
        a.timed = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx));
        for (uint32_t r = 0; r < reps; ++r) {
          a.ev.emplace_back(*pool, idx++);
          a.timed->launch(*a.dec, a.dgx, a.dgy, a.dgz, &a.ev.back());
          if (a.red) {
            a.ev.emplace_back(*pool, idx++);
            a.timed->launch(*a.red, kQH, a.M, 1, &a.ev.back());
          }
        }
        a.timed->close();
      }
    };

    std::mt19937 rng(2028);
    bool all_ok = true;
    for (uint32_t d : depths) {
      // ---- host data for this depth ----------------------------------------------------------
      DepthData dd;
      dd.d = d;
      {
        const std::vector<float> qr = read_file<float>(dir + "/q_" + std::to_string(d) + ".bin", size_t(8) * kRow);
        const std::vector<float> gr = read_file<float>(dir + "/g_" + std::to_string(d) + ".bin", size_t(8) * kRow);
        dd.q.resize(size_t(kMaxM) * kRow);
        dd.g.resize(size_t(kMaxM) * kRow);
        for (uint32_t m = 0; m < kMaxM; ++m) {
          std::memcpy(&dd.q[size_t(m) * kRow], &qr[size_t(m % 8) * kRow], kRow * 4);
          std::memcpy(&dd.g[size_t(m) * kRow], &gr[size_t(m % 8) * kRow], kRow * 4);
        }
        if (kv8)
          for (uint32_t r = 0; r < kMaxM * kQH; ++r) {
            float y[kHD];
            common::kv8::rotate(&dd.q[size_t(r) * kHD], y);
            std::memcpy(&dd.q[size_t(r) * kHD], y, sizeof y);
          }
        dd.q16.resize(dd.q.size());
        for (size_t i = 0; i < dd.q.size(); ++i) dd.q16[i] = common::f32_to_bf16(dd.q[i]);
      }
      const uint32_t maxM = *std::max_element(Ms.begin(), Ms.end());
      dd.nref = std::min<uint32_t>(maxM, uint32_t(kv_rows) > d ? uint32_t(kv_rows) - d : 0);
      if (dd.nref < 8) throw std::runtime_error("the dump's KV does not reach depth + 8");
      dd.ref_o.assign(size_t(kMaxM) * kRow, 0.0);
      dd.ref_out.assign(size_t(kMaxM) * kRow, 0);
      {
        std::vector<std::thread> pool_t;
        const uint32_t jobs = dd.nref * kQH;
        for (uint32_t t = 0; t < threads; ++t)
          pool_t.emplace_back([&, t]() {
            for (uint32_t job = t; job < jobs; job += threads) {
              const uint32_t m = job / kQH, h = job % kQH;
              const size_t base = size_t(m) * kRow + size_t(h) * kHD;
              double rot[kHD];
              avr::attend_rot(&dd.q[base], hkv, h / 6, d + m + 1, rot);
              if (kv8) kv8_ref::unrotate64(rot, &dd.ref_o[base]);
              else std::memcpy(&dd.ref_o[base], rot, sizeof rot);
              for (uint32_t i = 0; i < kHD; ++i)
                dd.ref_out[base + i] = kv8_ref::gated(dd.ref_o[base + i], dd.g[base + i]);
            }
          });
        for (auto& th : pool_t) th.join();
      }
      upload(ctx, aq.ptr(), dd.q.data(), dd.q.size() * 4);
      upload(ctx, ag.ptr(), dd.g.data(), dd.g.size() * 4);
      upload(ctx, aq16.ptr(), dd.q16.data(), dd.q16.size() * 2);
      // v2 M = 1 rows at d + m: row m's q / gate copied to row 0.
      dd.m1.assign(size_t(kMaxM) * kRow, 0);
      for (uint32_t m = 0; m < dd.nref; ++m) {
        upload(ctx, aq.ptr(), &dd.q[size_t(m) * kRow], kRow * 4);
        upload(ctx, ag.ptr(), &dd.g[size_t(m) * kRow], kRow * 4);
        set_pos(d + m, 1);
        exec(m1list);
        download(ctx, &dd.m1[size_t(m) * kRow], m1out.ptr(), kRow * 2);
      }
      upload(ctx, aq.ptr(), dd.q.data(), dd.q.size() * 4);
      upload(ctx, ag.ptr(), dd.g.data(), dd.g.size() * 4);
      build_lists(d);

      // ---- per M: numerics, then timing ------------------------------------------------------
      for (uint32_t M : Ms) {
        if (M > dd.nref) {
          std::printf("\n## depth %u, M %u: skipped (the dump's KV stops at %zu rows)\n", d, M, kv_rows);
          continue;
        }
        set_pos(d, M);
        std::printf("\n## depth %u, M %u, %s KV (pos %u, rows' keys %u..%u)\n\n", d, M, kv8 ? "int8" : "bf16", d,
                    d + 1, d + M);
        std::printf("| arm | N1 worst cos (h, m) | N1 max abs | N2b vs v2 M=1 worst cos | N2b max abs | repeatable | "
                    "verdict |\n|---|---|---|---|---|---|---|\n");
        std::map<std::string, double> n1;
        double v2_n1 = -1;
        for (Arm& a : arms) {
          if (a.M != M) continue;
          std::vector<uint16_t> o1(size_t(M) * kRow), o2(size_t(M) * kRow);
          auto run_once = [&](std::vector<uint16_t>& o) {
            imm.fill(a.out->ptr(), 0u, a.kind == Kind::Pfa ? pfo_bytes : size_t(M) * kRow * 2);
            exec(*a.once);
            if (a.kind == Kind::Pfa) {
              std::vector<float> po(size_t(kQH) * kPfRows * kHD);
              download(ctx, po.data(), a.out->ptr(), pfo_bytes);
              for (uint32_t m = 0; m < M; ++m)
                for (uint32_t h = 0; h < kQH; ++h) {
                  float x[kHD], y[kHD];
                  for (uint32_t i = 0; i < kHD; ++i) x[i] = po[(size_t(h) * kPfRows + m) * kHD + i];
                  if (kv8) common::kv8::unrotate(x, y);
                  else std::memcpy(y, x, sizeof x);
                  for (uint32_t i = 0; i < kHD; ++i) {
                    const size_t k = size_t(m) * kRow + h * kHD + i;
                    o[k] = gate_bf16(y[i], dd.g[k]);
                  }
                }
            } else if (a.red) {
              download(ctx, o.data(), a.out->ptr(), size_t(M) * kRow * 2);
            } else {   // DPAS at M = 16: no reduce exists - attn_reduce_v2's merge on the host
              std::vector<float> p(size_t(kQH) * kTgt * M * kPart);
              download(ctx, p.data(), part.ptr(), p.size() * 4);
              for (uint32_t m = 0; m < M; ++m)
                for (uint32_t h = 0; h < kQH; ++h) {
                  const uint32_t nb = avr::nblocks(d, m);
                  float mx = -INFINITY, sm = 0, acc[kHD] = {};
                  for (uint32_t b = 0; b < nb; ++b) {
                    const float* pp = &p[((size_t(h) * kTgt + b) * M + m) * kPart];
                    const float nmx = std::fmax(mx, pp[0]);
                    const float ea = std::exp(mx - nmx), eb = std::exp(pp[0] - nmx);
                    sm = std::fma(sm, ea, pp[1] * eb);
                    for (uint32_t i = 0; i < kHD; ++i) acc[i] = std::fma(acc[i], ea, pp[2 + i] * eb);
                    mx = nmx;
                  }
                  float x[kHD], y[kHD];
                  for (uint32_t i = 0; i < kHD; ++i) x[i] = acc[i] / sm;
                  if (kv8) common::kv8::unrotate(x, y);
                  else std::memcpy(y, x, sizeof x);
                  for (uint32_t i = 0; i < kHD; ++i) {
                    const size_t k = size_t(m) * kRow + h * kHD + i;
                    o[k] = gate_bf16(y[i], dd.g[k]);
                  }
                }
            }
          };
          run_once(o1);
          run_once(o2);
          const bool rep = o1 == o2;
          double w1 = 2, ma1 = 0, w2 = 2, ma2 = 0;
          uint32_t wh = 0, wm = 0;
          for (uint32_t m = 0; m < M; ++m)
            for (uint32_t h = 0; h < kQH; ++h) {
              const size_t k = size_t(m) * kRow + h * kHD;
              const double c1 = cos_rows(&o1[k], &dd.ref_out[k], &ma1);
              if (c1 < w1) { w1 = c1; wh = h; wm = m; }
              w2 = std::min(w2, cos_rows(&o1[k], &dd.m1[k], &ma2));
            }
          n1[a.name] = w1;
          if (a.kind == Kind::V2) v2_n1 = w1;
          const bool pass = w1 >= 0.99999 && (a.kind == Kind::Pfa || w2 >= 0.99999) && rep;
          if (a.kind == Kind::Dpas) all_ok = all_ok && pass;
          std::printf("| %s | %.8f (%u, %u) | %.3g | %.8f | %.3g | %s | %s |\n", a.name.c_str(), w1, wh, wm, ma1, w2,
                      ma2, rep ? "yes" : "NO", a.kind == Kind::Pfa ? "recorded" : pass ? "N1/N2b pass" : "FAIL");
          std::printf("NUM kv=%s M=%u depth=%u arm=%s n1=%.9f n1_v2=%.9f n2b=%.9f rep=%d pass=%d\n",
                      kv8 ? "int8" : "bf16", M, d, a.name.c_str(), w1, v2_n1, w2, rep ? 1 : 0, pass ? 1 : 0);
        }
        if (!timing) continue;
        // ---- interleaved timing ----
        std::vector<Arm*> am;
        for (Arm& a : arms)
          if (a.M == M) am.push_back(&a);
        if (am.empty()) continue;
        size_t iv2 = am.size();
        for (size_t k = 0; k < am.size(); ++k)
          if (am[k]->kind == Kind::V2) iv2 = k;
        std::vector<std::vector<double>> dec_us(am.size()), red_us(am.size()), sum_us(am.size()), ratio(am.size());
        for (Arm* a : am) exec(*a->timed);   // warm-up
        for (uint32_t r = 0; r < rounds; ++r) {
          std::vector<double> this_sum(am.size());
          for (size_t k0 = 0; k0 < am.size(); ++k0) {
            const size_t k = (k0 + r) % am.size();
            Arm& a = *am[k];
            for (l0::Event& e : a.ev) e.reset();
            exec(*a.timed);
            double ds = 0, rs = 0;
            for (size_t e = 0; e < a.ev.size(); ++e) ((a.red && e % 2 == 1) ? rs : ds) += a.ev[e].duration_us();
            dec_us[k].push_back(ds / reps);
            red_us[k].push_back(rs / reps);
            sum_us[k].push_back((ds + rs) / reps);
            this_sum[k] = (ds + rs) / reps;
          }
          if (iv2 < am.size())
            for (size_t k = 0; k < am.size(); ++k) ratio[k].push_back(this_sum[iv2] / this_sum[k]);
        }
        std::printf("\n| arm | decode us | reduce us | sum us [range] | v2 / arm paired [range] | verdict |\n"
                    "|---|---|---|---|---|---|\n");
        for (size_t k = 0; k < am.size(); ++k) {
          std::vector<double> s = sum_us[k], rr = ratio[k];
          std::sort(s.begin(), s.end());
          std::sort(rr.begin(), rr.end());
          const double rm = median(ratio[k]);
          std::string verdict = "-";
          if (am[k]->kind == Kind::Dpas && !rr.empty())
            verdict = (rm >= 1.03 && rr.front() > 1.00 && n1[am[k]->name] >= 0.99999) ? "dpas" : "v2";
          if (am[k]->kind == Kind::Dpas && rr.empty()) verdict = "alone (no v2)";
          std::printf("| %s | %.1f | %.1f | %.1f [%.1f-%.1f] | %s | %s |\n", am[k]->name.c_str(), median(dec_us[k]),
                      median(red_us[k]), median(sum_us[k]), s.front(), s.back(),
                      rr.empty() ? "-" : (std::to_string(rm).substr(0, 5) + " [" + std::to_string(rr.front()).substr(0, 5) +
                                          "-" + std::to_string(rr.back()).substr(0, 5) + "]").c_str(),
                      verdict.c_str());
          std::printf("TIME kv=%s M=%u depth=%u arm=%s us=%.2f lo=%.2f hi=%.2f ratio=%.4f rlo=%.4f rhi=%.4f verdict=%s\n",
                      kv8 ? "int8" : "bf16", M, d, am[k]->name.c_str(), median(sum_us[k]), s.front(), s.back(),
                      rr.empty() ? 0.0 : rm, rr.empty() ? 0.0 : rr.front(), rr.empty() ? 0.0 : rr.back(), verdict.c_str());
        }
        std::fflush(stdout);
      }

      // ---- N2a: property R per DPAS tag over the M list's 2..8 ----
      if (trials) {
        std::normal_distribution<float> nd(0.f, 1.f);
        double rms = 0;
        for (uint32_t i = 0; i < kRow; ++i) rms += double(dd.q[i]) * dd.q[i];
        const float sig = float(std::sqrt(rms / kRow));
        std::map<std::string, std::vector<Arm*>> by_tag;
        for (Arm& a : arms)
          if (a.kind == Kind::Dpas && a.has_red && a.M <= 8) by_tag[a.name].push_back(&a);
        for (auto& [tag, list] : by_tag) {
          std::vector<uint16_t> first;
          uint32_t same = 0, done = 0;
          for (uint32_t t = 0; t < trials; ++t) {
            Arm& a = *list[rng() % list.size()];
            const uint32_t M = a.M, s = rng() % M;
            if (d < s) continue;
            std::vector<float> qx(size_t(M) * kRow), gx(size_t(M) * kRow);
            for (auto& x : qx) x = sig * nd(rng);
            for (auto& x : gx) x = nd(rng);
            std::memcpy(&qx[size_t(s) * kRow], dd.q.data(), kRow * 4);
            std::memcpy(&gx[size_t(s) * kRow], dd.g.data(), kRow * 4);
            upload(ctx, aq.ptr(), qx.data(), qx.size() * 4);
            upload(ctx, ag.ptr(), gx.data(), gx.size() * 4);
            set_pos(d - s, M);
            exec(*a.once);
            std::vector<uint16_t> row(kRow);
            download(ctx, row.data(), static_cast<char*>(a.out->ptr()) + size_t(s) * kRow * 2, kRow * 2);
            if (first.empty()) first = row;
            same += row == first;
            ++done;
          }
          const bool pass = same == done;
          all_ok = all_ok && pass;
          std::printf("N2A kv=%s depth=%u arm=%s trials=%u bitwise=%u %s\n", kv8 ? "int8" : "bf16", d, tag.c_str(), done,
                      same, pass ? "PASS" : "FAIL");
        }
        upload(ctx, aq.ptr(), dd.q.data(), dd.q.size() * 4);
        upload(ctx, ag.ptr(), dd.g.data(), dd.g.size() * 4);
      }
    }
    std::printf("\n# DPAS arms N1 / N2a / N2b: %s\n", all_ok ? "PASS" : "FAIL (see rows)");
    return all_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_verify_attn: %s\n", e.what());
    return 1;
  }
}

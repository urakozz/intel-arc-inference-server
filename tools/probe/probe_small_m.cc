// probe_small_m - spec 28 P0 (plan 28a): Qwen3.8's decode linears at M = 1..16 rows through
// today's kernels (gemv / gemv_i8w / gemv_bf16) and through stand-in small-M DPAS GEMMs
// (tools/probe/probe_small_m.cl), on one card.
//
//   probe_small_m check    [--shape all] [--m 1,2,4,8,16] [arm filters]
//   probe_small_m shapes   [--shape all] [--m 1,2,4,8,16] [--type bf16,i8i4] [--orient wB,wA]
//                          [--splitk grid|wg] [--wg 2x8|8x4] [--deq shift|magic] [--prefetch K]
//                          [--grf 128|256] [--s N] [--rounds 5] [--launches 40]
//   probe_small_m levers   [--shape all] [--orient wB,wA] [--rounds 5]
//   probe_small_m numerics <snapshot> [--shape all] [--trials 1000]
//   probe_small_m prod     [--shape all] [--m 2,4,8,16] [--rounds 5]   (plan 28b: production gemm_sm + levers;
//                          Qwen3.8's shapes, the token sum over them)
//   (every mode: [--device 0]; the card is whatever ZE_AFFINITY_MASK leaves at index 0)
//
// check:    every built variant of the selected shapes and M against the fp64 reference
//           (spec 28 N2; the int arm against the reference over its own quantised rows, the
//           quantiser bitwise against the host's), the controls too, and property R on random
//           rows (rows bitwise across M >= 2 and with the other rows randomised, a few trials).
//           No timing. Exit 1 on any failure.
// shapes:   per shape and M, three closed lists - the control at M = 1, the control at M and
//           the candidate at M (the int arm's list carries psm_quant before every GEMM) - each
//           `launches` launches cycling NB copies of the weights past the 24 MB L2, executed
//           in a rotating order per round after a warm-up (interleaved; docs/probes/probe-w4a8-
//           2026-09-23.md §14.4); medians over rounds of us/launch and of each round's ratios.
//           Then the GEMM sum of a token (each shape x its layers + the head) per arm at each M
//           against the control's at M = 1 - spec 28 §6 (a).
// levers:   at M = 8 (bf16): per shape and orientation, the base arm against each lever one at
//           a time (in-work-group split-K 2x8 / 8x4, magic dequant, prefetch 1 / 2 / 3 groups,
//           256 GRF), the S sweep, and the 2x8 work-group shape at M = 2 / 4 / 16 (8x4 at 16)
//           against the base at that M; interleaved pairs with the control at M = 1.
// numerics: an engine on the snapshot (int8 head, max_len 4096); real rows read back from its
//           decode scratch after a prefill (the final-norm hidden for the K = 5120 linears and
//           the head, layer 63's attention output for K = 6144, and layer 63's silu(gate) x up
//           computed on the host from those rows for down); layer 62's (qkv||z, out_proj) and
//           63's weights straight from the loaded model. N2 against the fp64 reference and
//           against gemv at the same M, real and random rows; property R over `trials` trials.
//
// Any timed arm whose binary carries a GRF mode other than its name says (the zebin's
// .ze_info `grf_count`) is refused before it runs (spec 28 §2.1's pitfall).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "gemm_sm_levers.h"
#include "gemm_sm_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/engine.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kMaxM = 16;

// --- shapes ---------------------------------------------------------------------------
// kind: 0 int4 (layout L), 2 int8 head, 3 bf16 a||b (control only)
struct Shape {
  const char* key;
  const char* name;
  uint32_t K, N, S, L, layers;
  uint32_t kind;
  uint32_t layer;   // numerics: the checkpoint layer the real weights come from
  model::LinearId id;
};
const Shape kShapes[] = {
    {"qkvz", "qkv‖z", 5120, 16384, 1, 1, 48, 0, 62, model::LinearId::QkvZ},
    {"out", "out_proj", 6144, 5120, 4, 0, 48, 0, 62, model::LinearId::OutProj},
    {"qkv", "q‖k‖v", 5120, 14336, 2, 0, 16, 0, 63, model::LinearId::Qkv},
    {"o", "o_proj", 6144, 5120, 4, 0, 16, 0, 63, model::LinearId::OProj},
    {"gateup", "gate‖up", 5120, 34816, 8, 0, 64, 0, 63, model::LinearId::GateUp},
    {"down", "down", 17408, 5120, 4, 0, 64, 0, 63, model::LinearId::Down},
    {"head", "lm_head int8", 5120, 248320, 1, 2, 1, 2, loader::kTopLevel, model::LinearId::LmHead},
    {"ab", "a‖b bf16", 5120, 128, 1, 0, 48, 3, 62, model::LinearId::AB},
};

// --- arms -----------------------------------------------------------------------------
struct Arm {
  char t = 'b';          // b: bf16 DPAS, i: int DPAS on int8 rows
  char o = 'B';          // B: weights on B, A: weights on A
  std::string tag;       // lever suffix ("" | _wg2x8 | _wg8x4 | _mg | _pf<G> | _g256)
  uint32_t S = 0;        // 0: the shape's production S
  uint32_t nsgn = 4, ksl = 1;
  bool wg() const { return ksl > 1; }
  bool g256() const { return tag.find("_g256") != std::string::npos; }
  std::string label() const {
    return std::string(t == 'b' ? "bf16" : "i8") + (o == 'B' ? " wB" : " wA") + tag +
           (S ? " S" + std::to_string(S) : "");
  }
};
uint32_t arm_S(const Arm& a, const Shape& s) { return a.wg() ? 1 : (a.S ? a.S : s.S); }
std::string arm_name(const Arm& a, const Shape& s, uint32_t M) {
  return "psm_" + std::string(1, a.t) + std::string(1, a.o) + a.tag + "_M" + std::to_string(M) +
         "_K" + std::to_string(s.K) + "_N" + std::to_string(s.N) + "_S" + std::to_string(arm_S(a, s)) +
         "_L" + std::to_string(s.L);
}
std::string ctl_name(const Shape& s, uint32_t M) {
  if (s.kind == 2)
    return M == 16 ? "psm_ctl_gemv_i8w_M16_K5120_N248320" : kernels::gemv_i8w_variant(M, s.K, s.N);
  if (s.kind == 3)
    return M == 16 ? "psm_ctl_gemv_bf16_M16_K5120_N128_C16_S16"
                   : kernels::gemv_bf16_variant(M, s.K, s.N, {16, 16});
  if (M == 16)
    return "psm_ctl_gemv_M16_K" + std::to_string(s.K) + "_N" + std::to_string(s.N) + "_S" +
           std::to_string(s.S) + "_L" + std::to_string(s.L);
  return kernels::gemv_variant(M, s.K, s.N, s.S, s.L);
}
Arm lever_arm(char o, const std::string& tag) {
  Arm a;
  a.o = o;
  a.tag = tag;
  if (tag == "_wg2x8") { a.nsgn = 2; a.ksl = 8; }
  if (tag == "_wg8x4") { a.nsgn = 8; a.ksl = 4; }
  return a;
}

bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return bool(f); }

// The zebin's .ze_info carries `grf_count: <n>` per kernel. -1 when it is not there.
int grf_count(const std::string& bin, const char* kernel = "psm") {
  std::ifstream f(bin, std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t k = s.find(std::string("name:") + " " + kernel);
  if (k == std::string::npos) k = 0;
  const size_t p = s.find("grf_count:", k);
  if (p == std::string::npos) return -1;
  return std::atoi(s.c_str() + p + 10);
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// --- device inputs for one shape --------------------------------------------------------
struct Inputs {
  const Shape* sh = nullptr;
  // host copies (copy 0)
  std::vector<uint32_t> words;     // int4 L0 qweight or L1 tiles
  std::vector<uint16_t> scales16;  // int4 L0
  std::vector<int8_t> q8;          // head
  std::vector<float> colscale;     // head
  std::vector<uint16_t> bf16w;     // a||b tiled
  size_t wbytes = 0;               // weight bytes read per launch (nibbles + scales)
  std::vector<std::unique_ptr<l0::Mem>> w, sc;   // NB copies
  std::unique_ptr<l0::Mem> x, xq, xs, out;
  gemm_sm::WeightView view() const {
    gemm_sm::WeightView v;
    v.K = sh->K; v.N = sh->N;
    if (sh->kind == 2) { v.kind = 2; v.q8 = q8.data(); v.colscale = colscale.data(); }
    else { v.kind = sh->L; v.words = words.data(); v.scales = scales16.data(); }
    return v;
  }
};

uint32_t xorshift(uint32_t& st) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; }

// Random, incompressible weights of the shape (the B70 compresses constant fills, doc 01).
void make_random_weights(Inputs& in, uint32_t seed) {
  const Shape& s = *in.sh;
  if (s.kind == 0) {
    common::Int4Gptq w = common::Int4Gptq::random(s.K, s.N, seed);
    in.scales16 = w.scales;
    in.words = s.L == 1 ? w.tiled() : w.qweight;
    in.wbytes = w.bytes();
  } else if (s.kind == 2) {
    in.q8.resize(size_t(s.K) * s.N);
    uint32_t st = seed * 2654435761u + 1;
    for (auto& v : in.q8) v = int8_t(int(xorshift(st) % 255) - 127);
    in.colscale.resize(s.N);
    for (uint32_t n = 0; n < s.N; ++n) in.colscale[n] = 1e-3f * (1.0f + float(xorshift(st) % 1000) / 1000.f);
    in.wbytes = in.q8.size() + in.colscale.size() * 4;
  } else {
    std::vector<uint16_t> rm(size_t(s.K) * s.N);
    uint32_t st = seed;
    for (auto& e : rm) e = uint16_t((xorshift(st) & 0x807f) | ((0x77 + ((st >> 8) & 0xf)) << 7));
    in.bf16w = common::Bf16Tiled::from_rowmajor(rm.data(), s.K, s.N).data;
    in.wbytes = in.bf16w.size() * 2;
  }
}

// Upload NB copies (NB so one cycle exceeds 72 MB = 3x L2; 1 when not timing).
void upload(l0::Context& ctx, Inputs& in, bool timing) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const Shape& s = *in.sh;
  const void* wsrc;
  size_t wb, sb = 0;
  const void* ssrc = nullptr;
  if (s.kind == 0) {
    wsrc = in.words.data(); wb = in.words.size() * 4;
    if (s.L == 0) { ssrc = in.scales16.data(); sb = in.scales16.size() * 2; }
  } else if (s.kind == 2) {
    wsrc = in.q8.data(); wb = in.q8.size(); ssrc = in.colscale.data(); sb = in.colscale.size() * 4;
  } else {
    wsrc = in.bf16w.data(); wb = in.bf16w.size() * 2;
  }
  const int NB = timing ? std::max(2, int((72u << 20) / wb) + 1) : 1;
  in.w.clear(); in.sc.clear();
  for (int i = 0; i < NB; ++i) {
    in.w.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, wb));
    imm.copy(in.w.back()->ptr(), wsrc, wb);
    in.sc.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, std::max<size_t>(sb, 64)));
    if (sb) imm.copy(in.sc.back()->ptr(), ssrc, sb);
  }
  in.x = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(kMaxM) * s.K * 2);
  in.xq = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(kMaxM) * s.K);
  in.xs = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, 4096);
  in.out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(16) * kMaxM * s.N * 4);
}

// --- kernels ------------------------------------------------------------------------------
// One bound launch: the module, the kernel, the grid.
struct Launch {
  std::unique_ptr<l0::Module> mod, qmod;
  std::unique_ptr<l0::Kernel> k, q;   // q: psm_quant (int arms)
  uint32_t gx = 1, gy = 1, M = 1, S = 1;
  bool is_int = false;
  std::string name;
};

std::unique_ptr<Launch> make_control(l0::Context& ctx, const Shape& s, uint32_t M) {
  auto L = std::make_unique<Launch>();
  L->name = ctl_name(s, M);
  L->M = M;
  L->mod = std::make_unique<l0::Module>(ctx, kernels::path(L->name));
  if (s.kind == 0) {
    L->k = std::make_unique<l0::Kernel>(*L->mod, "gemv");
    L->k->group_size(64);
    L->gx = s.N / 64; L->gy = s.S; L->S = s.S;
  } else if (s.kind == 2) {
    L->k = std::make_unique<l0::Kernel>(*L->mod, "gemv_i8w");
    L->k->group_size(64);
    L->gx = s.N / 64;
  } else {
    L->k = std::make_unique<l0::Kernel>(*L->mod, "gemv_bf16");
    L->k->group_size(16 * 16);
    L->gx = s.N / 16;
  }
  return L;
}
std::unique_ptr<Launch> make_arm(l0::Context& ctx, const Shape& s, const Arm& a, uint32_t M) {
  auto L = std::make_unique<Launch>();
  L->name = arm_name(a, s, M);
  L->M = M;
  L->mod = std::make_unique<l0::Module>(ctx, kernels::path(L->name));
  L->k = std::make_unique<l0::Kernel>(*L->mod, "psm");
  L->k->group_size(16 * a.nsgn * a.ksl);
  L->gx = s.N / (16 * a.nsgn);
  L->S = arm_S(a, s);
  L->gy = L->S;
  L->is_int = a.t == 'i';
  if (L->is_int) {
    L->q = std::make_unique<l0::Kernel>(*L->mod, "psm_quant");
    L->q->group_size(256);
  }
  return L;
}
void bind(Launch& L, const Inputs& in, int copy) {
  const Shape& s = *in.sh;
  if (L.name.rfind("psm_", 0) == 0 && L.name.rfind("psm_ctl", 0) != 0) {
    L.k->arg_ptr(0, in.w[copy]->ptr());
    L.k->arg_ptr(1, in.sc[copy]->ptr());
    L.k->arg_ptr(2, L.is_int ? in.xq->ptr() : in.x->ptr());
    L.k->arg_ptr(3, in.xs->ptr());
    L.k->arg_ptr(4, in.out->ptr());
    if (L.q) {
      L.q->arg_ptr(0, in.x->ptr());
      L.q->arg_ptr(1, in.xq->ptr());
      L.q->arg_ptr(2, in.xs->ptr());
    }
  } else if (s.kind == 3) {
    L.k->arg_ptr(0, in.w[copy]->ptr());
    L.k->arg_ptr(1, in.x->ptr());
    L.k->arg_ptr(2, in.out->ptr());
  } else {
    L.k->arg_ptr(0, in.w[copy]->ptr());
    L.k->arg_ptr(1, in.sc[copy]->ptr());
    L.k->arg_ptr(2, in.x->ptr());
    L.k->arg_ptr(3, in.out->ptr());
  }
}
void append(l0::CmdList& list, Launch& L) {
  if (L.q) list.launch(*L.q, L.M);
  list.launch(*L.k, L.gx, L.gy);
}

// Run once on copy 0 and read back the output summed over slices (S, M, N) -> [M][N], plus the
// raw partials when asked.
std::vector<float> run_once(l0::Context& ctx, l0::Queue& q, l0::Fence& f, Launch& L, const Inputs& in,
                            std::vector<float>* partials = nullptr) {
  const Shape& s = *in.sh;
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::CmdList list = l0::CmdList::regular(ctx);
  bind(L, in, 0);
  append(list, L);
  list.close();
  q.execute(list, &f);
  f.wait();
  const size_t mn = size_t(L.M) * s.N;
  std::vector<float> p(size_t(L.S) * mn);
  imm.copy(p.data(), in.out->ptr(), p.size() * 4);
  std::vector<float> o(mn, 0.f);
  for (uint32_t sl = 0; sl < L.S; ++sl)
    for (size_t i = 0; i < mn; ++i) o[i] += p[size_t(sl) * mn + i];
  if (partials) *partials = std::move(p);
  return o;
}

void set_rows(l0::Context& ctx, const Inputs& in, const std::vector<uint16_t>& x) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(in.x->ptr(), x.data(), x.size() * 2);
}

// The column ranges a reference is computed over: all of N up to 6144, else three 512-wide
// windows (start, middle, end).
std::vector<std::pair<uint32_t, uint32_t>> ref_cols(uint32_t N) {
  if (N <= 6144) return {{0, N}};
  return {{0, 512}, {N / 2 - 256, N / 2 + 256}, {N - 512, N}};
}
// Reference over the windows; threads over windows x row blocks are not needed at these sizes.
void reference(const gemm_sm::WeightView& v, const std::vector<float>& xf, uint32_t M,
               std::vector<double>& r, std::vector<double>& b) {
  r.clear(); b.clear();
  auto cols = ref_cols(v.N);
  std::vector<std::vector<double>> rr(cols.size()), bb(cols.size());
  std::vector<std::thread> th;
  for (size_t i = 0; i < cols.size(); ++i)
    th.emplace_back([&, i] { gemm_sm::ref(v, xf, M, rr[i], bb[i], cols[i].first, cols[i].second); });
  for (auto& t : th) t.join();
  r.assign(size_t(M) * v.N, 0.0);
  b.assign(size_t(M) * v.N, 0.0);
  for (size_t i = 0; i < cols.size(); ++i)
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = cols[i].first; n < cols[i].second; ++n) {
        r[size_t(m) * v.N + n] = rr[i][size_t(m) * v.N + n];
        b[size_t(m) * v.N + n] = bb[i][size_t(m) * v.N + n];
      }
}
gemm_sm::N2 n2_windows(const std::vector<float>& got, const std::vector<double>& r,
                       const std::vector<double>& b, uint32_t M, uint32_t N) {
  gemm_sm::N2 acc;
  std::vector<double> med;
  for (auto [n0, n1] : ref_cols(N)) {
    gemm_sm::N2 e = gemm_sm::n2(got, r, b, M, N, n0, n1);
    acc.max_ratio = std::max(acc.max_ratio, e.max_ratio);
    acc.max_abs = std::max(acc.max_abs, e.max_abs);
    acc.min_cos = std::min(acc.min_cos, e.min_cos);
    acc.over += e.over;
    acc.outputs += e.outputs;
    med.push_back(e.median_ratio);
  }
  acc.median_ratio = median(med);
  return acc;
}

// --- options --------------------------------------------------------------------------------
struct Opts {
  std::string mode, snapshot;
  std::vector<std::string> shapes;
  std::vector<uint32_t> ms = {1, 2, 4, 8, 16};
  std::vector<char> types = {'b', 'i'}, orients = {'B', 'A'};
  std::string splitk = "grid", wgshape = "2x8", deq = "shift";
  uint32_t prefetch = 0, grf = 128, s = 0, rounds = 5, launches = 40, device = 0, trials = 1000;
};
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
  return v;
}
std::vector<const Shape*> pick_shapes(const Opts& o, bool with_ab) {
  std::vector<const Shape*> v;
  for (const Shape& s : kShapes) {
    if (s.kind == 3 && !with_ab) continue;
    bool any = o.shapes.empty();
    for (auto& k : o.shapes) if (k == "all" || k == s.key) any = true;
    if (any) v.push_back(&s);
  }
  return v;
}
// The candidate arms of `shapes` / `check` from the filters.
std::vector<Arm> pick_arms(const Opts& o) {
  std::vector<Arm> v;
  for (char t : o.types)
    for (char r : o.orients) {
      Arm a;
      a.t = t; a.o = r;
      if (o.splitk == "wg") {
        a.tag += "_wg" + o.wgshape;
        a.nsgn = o.wgshape == "8x4" ? 8 : 2;
        a.ksl = o.wgshape == "8x4" ? 4 : 8;
      }
      if (o.deq == "magic") a.tag += "_mg";
      if (o.prefetch) a.tag += "_pf" + std::to_string((o.prefetch + 63) / 64);
      if (o.grf == 256) a.tag += "_g256";
      a.S = o.s;
      v.push_back(a);
    }
  return v;
}

std::vector<uint16_t> random_rows(uint32_t M, uint32_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.f, 1.f);
  std::vector<uint16_t> x(size_t(M) * K);
  for (auto& e : x) e = common::f32_to_bf16(d(rng));
  return x;
}

// ================================ check =====================================================
// Every variant name the CMake block builds for a shape (bf16 / int sweeps, levers, S sweep).
struct Variant { Arm a; uint32_t M; };
std::vector<Variant> all_variants(const Shape& s) {
  std::vector<Variant> v;
  for (char o : {'B', 'A'}) {
    for (uint32_t M : {1u, 2u, 4u, 8u, 16u}) {
      Arm b; b.o = o; v.push_back({b, M});
      Arm i; i.t = 'i'; i.o = o; v.push_back({i, M});
    }
    for (uint32_t M : {3u, 5u, 6u, 7u}) { Arm b; b.o = o; v.push_back({b, M}); }
    for (const char* t : {"_wg2x8", "_wg8x4", "_mg", "_pf1", "_pf2", "_pf3", "_g256"})
      v.push_back({lever_arm(o, t), 8});
    for (uint32_t M : {2u, 4u, 16u}) v.push_back({lever_arm(o, "_wg2x8"), M});
    v.push_back({lever_arm(o, "_wg8x4"), 16});
    if (s.kind == 0)
      for (uint32_t S : {1u, 2u, 4u, 8u, 16u}) {
        if (S == s.S) continue;
        Arm b; b.o = o; b.S = S; v.push_back({b, 8});
      }
  }
  return v;
}

int run_check(l0::Context& ctx, const Opts& o) {
  l0::Queue q(ctx);
  l0::Fence f(q);
  bool ok = true;
  size_t n_var = 0, n_fail = 0, n_missing = 0;
  for (const Shape* sp : pick_shapes(o, true)) {
    const Shape& s = *sp;
    Inputs in;
    in.sh = &s;
    make_random_weights(in, 1000 + s.K + s.N);
    upload(ctx, in, false);
    const gemm_sm::WeightView v = in.view();
    std::vector<uint16_t> x16 = random_rows(kMaxM, s.K, 77 + s.N);
    std::vector<float> xf16 = gemm_sm::bf16_rows(x16);
    std::printf("\n## %s - %u x %u (check)\n\n| variant | M | N2 max ratio | median | min cos | vs | result |\n"
                "|---|---|---|---|---|---|---|\n", s.name, s.K, s.N);
    // references per M (bf16 rows) and the int arm's (quantised rows), computed once per M
    std::map<uint32_t, std::vector<double>> R, B, RQ, BQ;
    std::map<uint32_t, std::vector<int8_t>> Q;
    std::map<uint32_t, std::vector<float>> QS;
    auto refs = [&](uint32_t M) {
      if (R.count(M)) return;
      std::vector<float> xf(xf16.begin(), xf16.begin() + size_t(M) * s.K);
      if (s.kind == 3) {
        // a||b: bf16 weights; the reference from the tiled bytes
        std::vector<double> r(size_t(M) * s.N, 0.0), b(size_t(M) * s.N, 0.0);
        const uint32_t K8 = s.K / 8;
        for (uint32_t n = 0; n < s.N; ++n)
          for (uint32_t m = 0; m < M; ++m) {
            double a = 0, bb = 0;
            for (uint32_t k = 0; k < s.K; ++k) {
              const double w = common::bf16_to_f32(in.bf16w[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16]);
              a += w * xf[size_t(m) * s.K + k];
              bb += std::fabs(w * xf[size_t(m) * s.K + k]);
            }
            r[size_t(m) * s.N + n] = a;
            b[size_t(m) * s.N + n] = bb;
          }
        R[M] = r; B[M] = b;
        return;
      }
      reference(v, xf, M, R[M], B[M]);
      gemm_sm::quant_rows_i8(xf, M, s.K, Q[M], QS[M]);
      reference(v, gemm_sm::dequant_rows_i8(Q[M], QS[M], M, s.K), M, RQ[M], BQ[M]);
    };
    // controls
    for (uint32_t M : o.ms) {
      std::unique_ptr<Launch> c;
      try { c = make_control(ctx, s, M); } catch (const std::exception& e) {
        std::printf("| %s | %u | - | - | - | - | **NO BINARY** |\n", ctl_name(s, M).c_str(), M);
        ++n_missing; ok = false; continue;
      }
      refs(M);
      set_rows(ctx, in, x16);
      std::vector<float> got = run_once(ctx, q, f, *c, in);
      gemm_sm::N2 e = n2_windows(got, R[M], B[M], M, s.N);
      std::printf("| %s (control) | %u | %.3g | %.3g | %.9f | fp64 | %s |\n", c->name.c_str(), M, e.max_ratio,
                  e.median_ratio, e.min_cos, e.pass() ? "pass" : "N2 over (recorded)");
    }
    if (s.kind == 3) continue;
    // every stand-in variant at the selected M
    std::map<std::string, std::map<uint32_t, std::vector<float>>> partial_rows;   // config -> M -> partials
    for (const Variant& vv : all_variants(s)) {
      if (std::find(o.ms.begin(), o.ms.end(), vv.M) == o.ms.end() &&
          !(vv.M == 3 || vv.M == 5 || vv.M == 6 || vv.M == 7))
        continue;
      const std::string name = arm_name(vv.a, s, vv.M);
      if (!file_exists(kernels::path(name))) {
        std::printf("| %s | %u | - | - | - | - | **NO BINARY** |\n", name.c_str(), vv.M);
        ++n_missing; ok = false; continue;
      }
      const int grf = grf_count(kernels::path(name));
      if (grf != (vv.a.g256() ? 256 : 128)) {
        std::printf("| %s | %u | - | - | - | - | **GRF %d, expected %d** |\n", name.c_str(), vv.M, grf,
                    vv.a.g256() ? 256 : 128);
        ++n_fail; ok = false; continue;
      }
      ++n_var;
      std::unique_ptr<Launch> L = make_arm(ctx, s, vv.a, vv.M);
      refs(vv.M);
      set_rows(ctx, in, x16);
      std::vector<float> parts;
      std::vector<float> got = run_once(ctx, q, f, *L, in, &parts);
      bool pass;
      gemm_sm::N2 e;
      const char* vs;
      std::string extra;
      if (vv.a.t == 'i') {
        // the quantiser bitwise against the host's, then N2 against the quantised rows' reference
        std::vector<int8_t> dq(size_t(vv.M) * s.K);
        std::vector<float> ds(vv.M);
        l0::CmdList imm = l0::CmdList::immediate(ctx);
        imm.copy(dq.data(), in.xq->ptr(), dq.size());
        imm.copy(ds.data(), in.xs->ptr(), ds.size() * 4);
        const bool qok = dq == Q[vv.M] && std::memcmp(ds.data(), QS[vv.M].data(), ds.size() * 4) == 0;
        e = n2_windows(got, RQ[vv.M], BQ[vv.M], vv.M, s.N);
        gemm_sm::N2 eb = n2_windows(got, R[vv.M], B[vv.M], vv.M, s.N);
        char buf[160];
        std::snprintf(buf, sizeof buf, "; vs the bf16 rows' fp64: max ratio %.3g, min cos %.7f; quantiser %s",
                      eb.max_ratio, eb.min_cos, qok ? "bitwise" : "**DIFFERS**");
        extra = buf;
        pass = e.pass() && qok;
        vs = "fp64 (q rows)";
      } else {
        e = n2_windows(got, R[vv.M], B[vv.M], vv.M, s.N);
        pass = e.pass();
        vs = "fp64";
      }
      if (!pass) { ok = false; ++n_fail; }
      std::printf("| %s | %u | %.3g | %.3g | %.9f | %s | %s%s |\n", name.c_str(), vv.M, e.max_ratio, e.median_ratio,
                  e.min_cos, vs, pass ? "pass" : "**FAIL**", extra.c_str());
      std::fflush(stdout);
      // property R material: the configuration without M, its partials per row
      Arm key = vv.a;
      partial_rows[std::string(1, key.t) + key.o + key.tag + "_S" + std::to_string(arm_S(key, s))][vv.M] = parts;
    }
    // Property R across M (same rows): row m's partials at every M >= 2 against the largest M built.
    std::printf("\nproperty R across M (same rows, bitwise):\n");
    for (auto& [cfg, byM] : partial_rows) {
      const uint32_t S = uint32_t(std::stoul(cfg.substr(cfg.rfind("_S") + 2)));
      uint32_t Mref = 0;
      for (auto& [M, p] : byM) Mref = std::max(Mref, M);
      if (Mref < 3) continue;
      size_t diff = 0, cmp = 0;
      for (auto& [M, p] : byM) {
        if (M < 2 || M == Mref) continue;
        for (uint32_t sl = 0; sl < S; ++sl)
          for (uint32_t m = 0; m < M; ++m) {
            ++cmp;
            if (std::memcmp(&p[(size_t(sl) * M + m) * s.N], &byM[Mref][(size_t(sl) * Mref + m) * s.N], s.N * 4))
              ++diff;
          }
      }
      std::printf("- %s: %zu row-slices compared against M = %u, %zu differ %s\n", cfg.c_str(), cmp, Mref, diff,
                  diff ? "**FAIL**" : cmp == 0 ? "(NOT TESTED: built at one M only)" : "");
      if (diff) { ok = false; ++n_fail; }
    }
    // Property R with the other rows random: base arms at M = 16 (and 8), 4 trials.
    for (char t : {'b', 'i'})
      for (char r : {'B', 'A'})
        for (uint32_t M : {8u, 16u}) {
          Arm a; a.t = t; a.o = r;
          if (!file_exists(kernels::path(arm_name(a, s, M)))) continue;
          std::unique_ptr<Launch> L = make_arm(ctx, s, a, M);
          std::vector<uint16_t> base(x16.begin(), x16.begin() + size_t(M) * s.K);
          set_rows(ctx, in, base);
          std::vector<float> p0;
          run_once(ctx, q, f, *L, in, &p0);
          size_t diff = 0;
          for (uint32_t trial = 0; trial < 4; ++trial) {
            const uint32_t keep = (trial * 5 + 3) % M;
            std::vector<uint16_t> xr = random_rows(M, s.K, 9000 + trial);
            std::copy(base.begin() + size_t(keep) * s.K, base.begin() + size_t(keep + 1) * s.K,
                      xr.begin() + size_t(keep) * s.K);
            set_rows(ctx, in, xr);
            std::vector<float> p;
            run_once(ctx, q, f, *L, in, &p);
            for (uint32_t sl = 0; sl < L->S; ++sl)
              if (std::memcmp(&p[(size_t(sl) * M + keep) * s.N], &p0[(size_t(sl) * M + keep) * s.N], s.N * 4)) ++diff;
          }
          std::printf("- %s other rows random, 4 trials: %zu row-slices differ %s\n", L->name.c_str(), diff,
                      diff ? "**FAIL**" : "");
          if (diff) { ok = false; ++n_fail; }
        }
  }
  std::printf("\ncheck: %zu stand-in variants run, %zu failures, %zu missing binaries -> %s\n", n_var, n_fail,
              n_missing, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ================================ timing =====================================================
struct Timed {
  std::unique_ptr<Launch> L;
  l0::CmdList list;
  uint32_t launches;
  std::vector<double> us;
};
std::unique_ptr<Timed> make_timed(l0::Context& ctx, std::unique_ptr<Launch> L, const Inputs& in, uint32_t launches) {
  auto T = std::make_unique<Timed>(Timed{std::move(L), l0::CmdList::regular(ctx), launches, {}});
  for (uint32_t i = 0; i < launches; ++i) {
    bind(*T->L, in, int(i % in.w.size()));
    append(T->list, *T->L);
  }
  T->list.close();
  return T;
}
double exec_us(l0::Queue& q, l0::Fence& f, Timed& T) {
  const auto t0 = Clock::now();
  q.execute(T.list, &f);
  f.wait();
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / T.launches;
}
// Rounds over a set of lists, the order rotating (and reversed on odd rounds), after a warm-up
// of 3 executions each. Fills each T->us with one value per round.
void interleave(l0::Queue& q, l0::Fence& f, std::vector<Timed*> ts, uint32_t rounds) {
  for (int w = 0; w < 3; ++w)
    for (Timed* t : ts) exec_us(q, f, *t);
  for (Timed* t : ts) t->us.clear();
  const size_t n = ts.size();
  for (uint32_t r = 0; r < rounds; ++r)
    for (size_t j = 0; j < n; ++j) {
      size_t i = (j + r) % n;
      if (r % 2) i = n - 1 - i;
      ts[i]->us.push_back(exec_us(q, f, *ts[i]));
    }
}
std::vector<double> ratios(const Timed& a, const Timed& b) {
  std::vector<double> v;
  for (size_t i = 0; i < a.us.size() && i < b.us.size(); ++i) v.push_back(a.us[i] / b.us[i]);
  return v;
}
std::string range_str(std::vector<double> v) {
  if (v.empty()) return "-";
  std::sort(v.begin(), v.end());
  char b[64];
  std::snprintf(b, sizeof b, "%.3f-%.3f", v.front(), v.back());
  return b;
}
bool grf_ok(const std::string& name, bool want256) {
  const int g = grf_count(kernels::path(name));
  if (g != (want256 ? 256 : 128)) {
    std::printf("**REFUSED %s: the binary's grf_count is %d, its name says %d**\n", name.c_str(), g, want256 ? 256 : 128);
    return false;
  }
  return true;
}
uint32_t launches_for(const Shape& s, uint32_t base) { return s.kind == 2 ? std::max(4u, base / 5) : base; }

void box_line() {
  std::printf("box load at start: ");
  std::fflush(stdout);
  if (std::system("uptime") != 0) std::puts("(uptime unavailable)");
}

int run_shapes(l0::Context& ctx, const Opts& o) {
  l0::Queue q(ctx);
  l0::Fence f(q);
  box_line();
  std::printf("timing: %u rounds, interleaved (rotating order, reversed on odd rounds) after 3 warm-up "
              "executions per list; us/launch over a list of N launches cycling NB weight copies\n", o.rounds);
  const std::vector<Arm> arms = pick_arms(o);
  // medians: ctl[shape][M], arm[label][shape][M] (us per launch)
  std::map<std::string, std::map<uint32_t, double>> ctl;
  std::map<std::string, std::map<std::string, std::map<uint32_t, double>>> arm_us;
  std::vector<const Shape*> done;
  for (const Shape* sp : pick_shapes(o, true)) {
    const Shape& s = *sp;
    Inputs in;
    in.sh = &s;
    make_random_weights(in, 42 + s.K + s.N);
    upload(ctx, in, true);
    set_rows(ctx, in, random_rows(kMaxM, s.K, 5 + s.K));
    const uint32_t nl = launches_for(s, o.launches);
    std::printf("\n## %s - %u x %u, %u layers, %.2f MB per launch, %zu weight copies, %u launches per list\n\n"
                "| arm | M | ctl M=1 us | ctl M us | arm us | arm GB/s | arm / ctl M (range) | arm / ctl M=1 (range) |\n"
                "|---|---|---|---|---|---|---|---|\n",
                s.name, s.K, s.N, s.layers, in.wbytes / 1e6, in.w.size(), nl);
    auto ctl1 = make_timed(ctx, make_control(ctx, s, 1), in, nl);
    for (uint32_t M : o.ms) {
      std::unique_ptr<Timed> ctlM;
      if (M != 1) ctlM = make_timed(ctx, make_control(ctx, s, M), in, nl);
      Timed* cM = M == 1 ? ctl1.get() : ctlM.get();
      std::vector<std::pair<std::string, std::unique_ptr<Timed>>> cands;
      if (s.kind != 3)
        for (const Arm& a : arms) {
          const std::string name = arm_name(a, s, M);
          if (!file_exists(kernels::path(name))) {
            std::printf("| %s | %u | NO BINARY %s |\n", a.label().c_str(), M, name.c_str());
            continue;
          }
          if (!grf_ok(name, a.g256())) continue;
          cands.emplace_back(a.label(), make_timed(ctx, make_arm(ctx, s, a, M), in, nl));
        }
      std::vector<Timed*> ts = {ctl1.get()};
      if (M != 1) ts.push_back(ctlM.get());
      for (auto& c : cands) ts.push_back(c.second.get());
      interleave(q, f, ts, o.rounds);
      const double c1 = median(ctl1->us), cm = median(cM->us);
      ctl[s.key][1] = c1;
      ctl[s.key][M] = cm;
      std::printf("| control %s | %u | %.1f | %.1f | - | %.0f | %.4f (%s) | - |\n", cM->L->name.c_str(), M, c1, cm,
                  in.wbytes / (cm * 1e3), median(ratios(*cM, *ctl1)), range_str(ratios(*cM, *ctl1)).c_str());
      for (auto& [label, c] : cands) {
        const double u = median(c->us);
        std::printf("| %s | %u | %.1f | %.1f | %.1f | %.0f | %.4f (%s) | %.4f (%s) |\n", label.c_str(), M, c1, cm, u,
                    in.wbytes / (u * 1e3), median(ratios(*c, *cM)), range_str(ratios(*c, *cM)).c_str(),
                    median(ratios(*c, *ctl1)), range_str(ratios(*c, *ctl1)).c_str());
        arm_us[label][s.key][M] = u;
      }
      std::fflush(stdout);
    }
    done.push_back(&s);
  }
  // Spec 28 §6 (a): a token's linears (layers x us; the head x 1) at M through an arm, against the
  // same shapes through today's kernels at M = 1. A shape the arm has no number for (a||b, or a
  // missing binary) enters the arm's sum through the control at M - it stays on today's kernel.
  std::printf("\n## the GEMM sum of a token over the %zu shapes timed (layers x median us) - derived\n\n"
              "| arm | M | sum us | x control M=1 sum | shapes on the arm |\n|---|---|---|---|---|\n", done.size());
  auto ctl_total = [&](uint32_t M) {
    double t = 0;
    for (const Shape* s : done) t += s->layers * ctl[s->key][M];
    return t;
  };
  const double base = ctl_total(1);
  for (uint32_t M : o.ms)
    std::printf("| control (today's kernels) | %u | %.1f | %.4f | - |\n", M, ctl_total(M), ctl_total(M) / base);
  // per arm, and the per-shape best of the two orientations of a type ("<type> best")
  std::map<std::string, std::map<std::string, std::map<uint32_t, double>>> best;
  for (auto& [label, byS] : arm_us)
    for (auto& [key, byM] : byS)
      for (auto& [M, u] : byM) {
        const size_t sp = label.find(' ');
        const std::string bl = label.substr(0, sp) + " best-orient" + label.substr(sp + 3);
        double& cur = best[bl][key][M];
        cur = cur == 0 ? u : std::min(cur, u);
      }
  for (auto* table : {&arm_us, &best})
    for (auto& [label, byS] : *table)
      for (uint32_t M : o.ms) {
        double t = 0;
        int on = 0;
        for (const Shape* s : done) {
          auto it = byS.find(s->key);
          if (it != byS.end() && it->second.count(M)) { t += s->layers * it->second.at(M); ++on; }
          else t += s->layers * ctl[s->key][M];
        }
        std::printf("| %s | %u | %.1f | %.4f | %d of %zu |\n", label.c_str(), M, t, t / base, on, done.size());
      }
  return 0;
}

int run_levers(l0::Context& ctx, const Opts& o) {
  l0::Queue q(ctx);
  l0::Fence f(q);
  box_line();
  std::printf("levers: bf16, one at a time against the base arm at the same M; %u rounds interleaved\n", o.rounds);
  for (const Shape* sp : pick_shapes(o, false)) {
    const Shape& s = *sp;
    Inputs in;
    in.sh = &s;
    make_random_weights(in, 42 + s.K + s.N);
    upload(ctx, in, true);
    set_rows(ctx, in, random_rows(kMaxM, s.K, 5 + s.K));
    const uint32_t nl = launches_for(s, o.launches);
    std::printf("\n## %s - %u x %u\n\n| orient | M | lever | base us | lever us | lever / base (range) | lever / ctl M=1 |\n"
                "|---|---|---|---|---|---|---|\n", s.name, s.K, s.N);
    auto ctl1 = make_timed(ctx, make_control(ctx, s, 1), in, nl);
    for (char o2 : o.orients) {
      struct Row { uint32_t M; Arm a; };
      std::vector<Row> rows;
      for (const char* t : {"_wg2x8", "_wg8x4", "_mg", "_pf1", "_pf2", "_pf3", "_g256"}) rows.push_back({8, lever_arm(o2, t)});
      if (s.kind == 0)
        for (uint32_t S : {1u, 2u, 4u, 8u, 16u})
          if (S != s.S) { Arm a; a.o = o2; a.S = S; rows.push_back({8, a}); }
      for (uint32_t M : {2u, 4u, 16u}) rows.push_back({M, lever_arm(o2, "_wg2x8")});
      rows.push_back({16, lever_arm(o2, "_wg8x4")});
      std::map<uint32_t, std::unique_ptr<Timed>> base;
      for (const Row& r : rows) {
        if (!base.count(r.M)) {
          Arm b; b.o = o2;
          base[r.M] = make_timed(ctx, make_arm(ctx, s, b, r.M), in, nl);
        }
        const std::string name = arm_name(r.a, s, r.M);
        if (!file_exists(kernels::path(name))) { std::printf("| %c | %u | %s | NO BINARY |\n", o2, r.M, name.c_str()); continue; }
        if (!grf_ok(name, r.a.g256())) continue;
        auto lv = make_timed(ctx, make_arm(ctx, s, r.a, r.M), in, nl);
        interleave(q, f, {ctl1.get(), base[r.M].get(), lv.get()}, o.rounds);
        std::printf("| w%c | %u | %s | %.1f | %.1f | %.4f (%s) | %.4f |\n", o2, r.M, r.a.label().c_str(),
                    median(base[r.M]->us), median(lv->us), median(ratios(*lv, *base[r.M])),
                    range_str(ratios(*lv, *base[r.M])).c_str(), median(ratios(*lv, *ctl1)));
        std::fflush(stdout);
      }
    }
  }
  return 0;
}

// ================================ prod (plan 28b) ============================================
// The production gemm_sm rows (kernels::kSmallMTuning) and their lever alternates (gemm_sm_levers.h) per Qwen3.8
// shape at each M >= 2 of --m, interleaved with today's gemv at M = 1 and at M: the lever decision of plan 28b at
// every M, and the GEMM sum of a token through the production rows against gemv's at M = 1 (spec 28 §6 (a)).
std::unique_ptr<Launch> make_prod(l0::Context& ctx, const kernels::SmallMTuning& t, uint32_t M) {
  auto L = std::make_unique<Launch>();
  L->name = kernels::gemm_sm_variant(M, t);
  L->M = M;
  L->mod = std::make_unique<l0::Module>(ctx, kernels::path(L->name));
  L->k = std::make_unique<l0::Kernel>(*L->mod, kernels::gemm_sm_entry(t));
  L->k->group_size(kernels::gemm_sm_wg_size(t));
  L->gx = t.N / kernels::gemm_sm_cols(t);
  L->gy = kernels::gemm_sm_grid_y(t);
  L->S = kernels::gemm_sm_out_s(t);
  return L;
}
int run_prod(l0::Context& ctx, const Opts& o) {
  l0::Queue q(ctx);
  l0::Fence f(q);
  box_line();
  std::vector<const Shape*> shapes = pick_shapes(o, false);
  std::printf("prod: the production gemm_sm rows and their lever alternates, %u rounds interleaved after 3 warm-up "
              "executions per list, with gemv at M = 1 and at M\n", o.rounds);
  std::map<std::string, std::map<uint32_t, double>> ctl, prod_us, best_us;
  std::vector<const Shape*> done;
  for (const Shape* sp : shapes) {
    const Shape& s = *sp;
    const unsigned lay = s.kind == 2 ? 2 : s.L;
    const kernels::SmallMTuning* t = kernels::small_m_tuning(s.K, s.N, s.kind == 2 ? 1 : s.S, lay);
    if (!t) { std::printf("%s: no kSmallMTuning row\n", s.name); continue; }
    std::vector<kernels::SmallMTuning> alts;
    for (const kernels::SmallMTuning& a : gemm_sm::lever_rows())
      if (a.K == s.K && a.N == s.N && a.layout == lay && kernels::gemm_sm_variant(8, a) != kernels::gemm_sm_variant(8, *t))
        alts.push_back(a);
    Inputs in;
    in.sh = &s;
    make_random_weights(in, 42 + s.K + s.N);
    upload(ctx, in, true);
    set_rows(ctx, in, random_rows(kMaxM, s.K, 5 + s.K));
    const uint32_t nl = launches_for(s, o.launches);
    std::printf("\n## %s - %u x %u, %u layers\n\n| kernel | M | us | / gemv M=1 (range) | / gemv M | / production (range) |\n"
                "|---|---|---|---|---|---|\n", s.name, s.K, s.N, s.layers);
    auto ctl1 = make_timed(ctx, make_control(ctx, s, 1), in, nl);
    for (uint32_t M : o.ms) {
      if (M < 2) continue;
      std::unique_ptr<Timed> ctlM;
      if (file_exists(kernels::path(ctl_name(s, M)))) ctlM = make_timed(ctx, make_control(ctx, s, M), in, nl);
      if (!grf_ok(kernels::gemm_sm_variant(M, *t), t->grf256)) continue;
      auto pr = make_timed(ctx, make_prod(ctx, *t, M), in, nl);
      std::vector<std::unique_ptr<Timed>> al;
      for (const auto& a : alts) {
        const std::string n = kernels::gemm_sm_variant(M, a);
        if (!file_exists(kernels::path(n))) { std::printf("| %s | %u | NO BINARY |\n", n.c_str(), M); continue; }
        if (!grf_ok(n, a.grf256)) continue;
        al.push_back(make_timed(ctx, make_prod(ctx, a, M), in, nl));
      }
      std::vector<Timed*> ts = {ctl1.get(), pr.get()};
      if (ctlM) ts.push_back(ctlM.get());
      for (auto& a : al) ts.push_back(a.get());
      interleave(q, f, ts, o.rounds);
      const double c1 = median(ctl1->us), pu = median(pr->us);
      ctl[s.key][1] = c1;
      prod_us[s.key][M] = pu;
      double best = pu;
      auto vs_m = [&](const Timed& x) { return ctlM ? median(ratios(x, *ctlM)) : 0.0; };
      if (ctlM)
        std::printf("| %s (gemv) | %u | %.1f | %.4f (%s) | 1 | - |\n", ctlM->L->name.c_str(), M, median(ctlM->us),
                    median(ratios(*ctlM, *ctl1)), range_str(ratios(*ctlM, *ctl1)).c_str());
      std::printf("| **%s** (production) | %u | %.1f | %.4f (%s) | %.4f | 1 |\n", pr->L->name.c_str(), M, pu,
                  median(ratios(*pr, *ctl1)), range_str(ratios(*pr, *ctl1)).c_str(), vs_m(*pr));
      for (auto& a : al) {
        const double u = median(a->us);
        best = std::min(best, u);
        std::printf("| %s | %u | %.1f | %.4f (%s) | %.4f | %.4f (%s) |\n", a->L->name.c_str(), M, u,
                    median(ratios(*a, *ctl1)), range_str(ratios(*a, *ctl1)).c_str(), vs_m(*a),
                    median(ratios(*a, *pr)), range_str(ratios(*a, *pr)).c_str());
      }
      best_us[s.key][M] = best;
      std::fflush(stdout);
    }
    done.push_back(&s);
  }
  std::printf("\n## the GEMM sum of a token (layers x median us, the head once) - derived\n\n"
              "| M | gemv M=1 sum us | production sum us | x gemv M=1 | best-of-alternates x gemv M=1 |\n|---|---|---|---|---|\n");
  double base = 0;
  for (const Shape* s : done) base += s->layers * ctl[s->key][1];
  for (uint32_t M : o.ms) {
    if (M < 2) continue;
    double p = 0, b = 0;
    for (const Shape* s : done) {
      p += s->layers * prod_us[s->key][M];
      b += s->layers * best_us[s->key][M];
    }
    std::printf("| %u | %.1f | %.1f | %.4f | %.4f |\n", M, base, p, p / base, b / base);
  }
  return 0;
}

// ================================ numerics ==================================================
std::vector<uint32_t> read_ids(const std::string& p) {
  std::vector<uint32_t> v;
  FILE* f = std::fopen(p.c_str(), "r");
  if (!f) return v;
  unsigned x;
  while (std::fscanf(f, "%u", &x) == 1) v.push_back(x);
  std::fclose(f);
  return v;
}

int run_numerics(l0::Context& ctx, const Opts& o) {
  l0::Queue q(ctx);
  l0::Fence f(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const uint32_t max_len = 4096;
  std::printf("loading %s (int8 head, max_len %u)\n", o.snapshot.c_str(), max_len);
  runtime::Engine eng(ctx, loader::load(ctx, o.snapshot, max_len, false, loader::LmHeadForm::Int8), max_len);
  // Real rows: two prompts of tests/golden/prompts/long32k.ids (cwd = the source tree), 16 greedy
  // tokens each; after every token the final-norm hidden (x[0, 5120)) and layer 63's attention
  // output (attn_out[0, 6144)).
  std::vector<uint32_t> file = read_ids("tests/golden/prompts/long32k.ids");
  if (file.size() < 3000) throw std::runtime_error("tests/golden/prompts/long32k.ids not found (cwd = source tree)");
  std::vector<uint16_t> rows5120, rows6144;
  runtime::DecodeBuffers& b = eng.buffers();
  for (uint32_t p = 0; p < 2; ++p) {
    eng.reset();
    std::vector<uint32_t> ids(file.begin() + p * 1500, file.begin() + p * 1500 + 1024);
    eng.prefill(ids);
    for (uint32_t t = 0; t < 16; ++t) {
      eng.generate(1);
      std::vector<uint16_t> xr(5120), ar(6144);
      imm.copy(xr.data(), b.x.ptr(), 5120 * 2);
      imm.copy(ar.data(), b.attn_out.ptr(), 6144 * 2);
      rows5120.insert(rows5120.end(), xr.begin(), xr.end());
      rows6144.insert(rows6144.end(), ar.begin(), ar.end());
    }
  }
  const uint32_t nreal = uint32_t(rows5120.size() / 5120);
  std::printf("real rows: %u (final-norm hidden, K 5120) and %u (layer 63 attention output, K 6144)\n", nreal,
              uint32_t(rows6144.size() / 6144));
  // Real weights to the host (the device copies are bound straight from the model).
  auto host_of = [&](const l0::Mem& m) {
    std::vector<uint8_t> h(m.size());
    imm.copy(h.data(), m.ptr(), m.size());
    return h;
  };
  std::map<std::string, Inputs> ins;
  auto get_inputs = [&](const Shape& s) -> Inputs& {
    Inputs& in = ins[s.key];
    if (in.sh) return in;
    in.sh = &s;
    const loader::DeviceWeight& dw = eng.model().linears.at({s.layer, s.id});
    std::vector<uint8_t> h = host_of(dw.mem);
    if (s.kind == 2) {
      in.q8.assign(reinterpret_cast<int8_t*>(h.data()), reinterpret_cast<int8_t*>(h.data()) + size_t(s.K) * s.N);
      std::vector<uint8_t> sc = host_of(*dw.scales);
      in.colscale.assign(reinterpret_cast<float*>(sc.data()), reinterpret_cast<float*>(sc.data()) + s.N);
    } else {
      if (dw.shape.layout != s.L || dw.shape.S != s.S) throw std::runtime_error(std::string(s.key) + ": loaded {S, layout} differ from the probe's table");
      in.words.assign(reinterpret_cast<uint32_t*>(h.data()), reinterpret_cast<uint32_t*>(h.data()) + h.size() / 4);
      if (s.L == 0) {
        std::vector<uint8_t> sc = host_of(*dw.scales);
        in.scales16.assign(reinterpret_cast<uint16_t*>(sc.data()), reinterpret_cast<uint16_t*>(sc.data()) + sc.size() / 2);
      }
    }
    upload(ctx, in, false);
    return in;
  };
  // down's real rows: layer 63's gate||up on the K 5120 rows, then prep_silu_mul's rounding
  // (interleaved 16-column blocks: g at (k/16)*32 + k%16, u at +16).
  std::vector<uint16_t> rows17408;
  {
    const Shape& gu = kShapes[4];
    Inputs& in = get_inputs(gu);
    std::vector<double> r, bd;
    std::vector<float> xf = gemm_sm::bf16_rows(rows5120);
    std::vector<std::thread> th;
    const uint32_t nt = 8;
    std::vector<std::vector<double>> rr(nt), bb(nt);
    for (uint32_t i = 0; i < nt; ++i)
      th.emplace_back([&, i] { gemm_sm::ref(in.view(), xf, nreal, rr[i], bb[i], gu.N / nt * i, gu.N / nt * (i + 1)); });
    for (auto& t : th) t.join();
    rows17408.resize(size_t(nreal) * 17408);
    for (uint32_t m = 0; m < nreal; ++m)
      for (uint32_t k = 0; k < 17408; ++k) {
        const uint32_t gf = (k / 16) * 32 + k % 16, uf = gf + 16;
        const size_t gi = size_t(m) * gu.N + gf, ui = size_t(m) * gu.N + uf;
        const double g = rr[gf / (gu.N / nt)][gi], u = rr[uf / (gu.N / nt)][ui];
        const float gb = common::bf16_to_f32(common::f32_to_bf16(float(g)));
        const float ub = common::bf16_to_f32(common::f32_to_bf16(float(u)));
        const float sb = common::bf16_to_f32(common::f32_to_bf16(gb / (1.0f + std::exp(-gb))));
        rows17408[size_t(m) * 17408 + k] = common::f32_to_bf16(sb * ub);
      }
  }
  auto real_rows = [&](const Shape& s, uint32_t M, uint32_t off) {
    const std::vector<uint16_t>& src = s.K == 5120 ? rows5120 : s.K == 6144 ? rows6144 : rows17408;
    std::vector<uint16_t> x(size_t(M) * s.K);
    for (uint32_t m = 0; m < M; ++m)
      std::copy(src.begin() + size_t((m + off) % nreal) * s.K, src.begin() + size_t((m + off) % nreal + 1) * s.K,
                x.begin() + size_t(m) * s.K);
    return x;
  };
  bool all_ok = true;
  for (const Shape* sp : pick_shapes(o, false)) {
    const Shape& s = *sp;
    Inputs& in = get_inputs(s);
    const gemm_sm::WeightView v = in.view();
    std::printf("\n## %s - %u x %u, layer %s\n\n| arm | M | rows | N2 vs fp64: max ratio | median | min cos | "
                "N2 vs gemv: max ratio | result |\n|---|---|---|---|---|---|---|---|\n",
                s.name, s.K, s.N, s.layer == loader::kTopLevel ? "head" : std::to_string(s.layer).c_str());
    for (uint32_t M : o.ms)
      for (int real = 1; real >= 0; --real) {
        std::vector<uint16_t> x = real ? real_rows(s, M, 0) : random_rows(M, s.K, 31 + M);
        std::vector<float> xf = gemm_sm::bf16_rows(x);
        std::vector<double> R, B;
        reference(v, xf, M, R, B);
        set_rows(ctx, in, x);
        std::unique_ptr<Launch> c = make_control(ctx, s, M);
        std::vector<float> gv = run_once(ctx, q, f, *c, in);
        gemm_sm::N2 ec = n2_windows(gv, R, B, M, s.N);
        std::printf("| gemv (control) | %u | %s | %.3g | %.3g | %.9f | - | %s |\n", M, real ? "real" : "random",
                    ec.max_ratio, ec.median_ratio, ec.min_cos, ec.pass() ? "pass" : "over");
        std::vector<double> gvd(gv.begin(), gv.end());
        std::vector<int8_t> qr;
        std::vector<float> qs;
        gemm_sm::quant_rows_i8(xf, M, s.K, qr, qs);
        std::vector<double> RQ, BQ;
        reference(v, gemm_sm::dequant_rows_i8(qr, qs, M, s.K), M, RQ, BQ);
        for (char t : {'b', 'i'})
          for (char r : {'B', 'A'}) {
            Arm a; a.t = t; a.o = r;
            std::unique_ptr<Launch> L = make_arm(ctx, s, a, M);
            std::vector<float> got = run_once(ctx, q, f, *L, in);
            gemm_sm::N2 e = n2_windows(got, R, B, M, s.N);
            gemm_sm::N2 eg = n2_windows(got, gvd, B, M, s.N);
            bool pass;
            std::string res;
            if (t == 'b') {
              pass = e.pass() && eg.pass();
              res = pass ? "N2 pass" : "**N2 FAIL**";
            } else {
              gemm_sm::N2 eq = n2_windows(got, RQ, BQ, M, s.N);
              pass = eq.pass();
              char buf[96];
              std::snprintf(buf, sizeof buf, "kernel vs q-rows fp64 %s (%.3g); N2 vs bf16 rows: %s", pass ? "pass" : "**FAIL**",
                            eq.max_ratio, e.pass() ? "pass" : "over (type rule)");
              res = buf;
            }
            if (!pass) all_ok = false;
            std::printf("| %s | %u | %s | %.3g | %.3g | %.9f | %.3g | %s |\n", a.label().c_str(), M, real ? "real" : "random",
                        e.max_ratio, e.median_ratio, e.min_cos, eg.max_ratio, res.c_str());
            std::fflush(stdout);
          }
      }
    // Property R: `trials` trials over the four base arms - a random row kept, the others random
    // (real or random rows alternating), at M = 16, 8, 4, 2 in turn; the kept row's partials bitwise
    // equal to its run at M = 16 (and across M: the row placed at the same index).
    for (char t : {'b', 'i'})
      for (char r : {'B', 'A'}) {
        Arm a; a.t = t; a.o = r;
        std::map<uint32_t, std::unique_ptr<Launch>> Ls;
        for (uint32_t M : {2u, 4u, 8u, 16u}) Ls[M] = make_arm(ctx, s, a, M);
        std::mt19937 rng(1234 + s.N);
        size_t diff = 0, cmp = 0;
        for (uint32_t trial = 0; trial < o.trials; ++trial) {
          const uint32_t M = trial % 4 == 0 ? 16 : trial % 4 == 1 ? 8 : trial % 4 == 2 ? 4 : 2;
          const uint32_t keep = rng() % M;
          // the reference: the kept row alone at index `keep` of an M = 16 run with zero rows elsewhere
          std::vector<uint16_t> row = (trial % 2) ? real_rows(s, 1, rng() % nreal) : random_rows(1, s.K, rng());
          std::vector<uint16_t> x16(size_t(16) * s.K, 0);
          std::copy(row.begin(), row.end(), x16.begin() + size_t(keep) * s.K);
          set_rows(ctx, in, x16);
          std::vector<float> p16;
          run_once(ctx, q, f, *Ls[16], in, &p16);
          std::vector<uint16_t> xm = random_rows(M, s.K, rng());
          std::copy(row.begin(), row.end(), xm.begin() + size_t(keep) * s.K);
          set_rows(ctx, in, xm);
          std::vector<float> pm;
          run_once(ctx, q, f, *Ls[M], in, &pm);
          for (uint32_t sl = 0; sl < Ls[M]->S; ++sl) {
            ++cmp;
            if (std::memcmp(&pm[(size_t(sl) * M + keep) * s.N], &p16[(size_t(sl) * 16 + keep) * s.N], s.N * 4)) ++diff;
          }
        }
        std::printf("- property R, %s: %u trials (M 16/8/4/2, the kept row against itself at M = 16 among zero rows),"
                    " %zu row-slices compared, %zu differ%s\n", a.label().c_str(), o.trials, cmp, diff,
                    diff ? " **FAIL**" : "");
        if (diff) all_ok = false;
      }
  }
  std::printf("\nnumerics: %s\n", all_ok ? "PASS (bf16 arms: N2 and R; int arms: kernel vs its own rows and R)" : "**FAIL**");
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: probe_small_m check|shapes|levers|numerics <snapshot> [options] (see the header)\n");
    return 2;
  }
  Opts o;
  o.mode = argv[1];
  int i = 2;
  if (o.mode == "numerics") {
    if (argc < 3) { std::fprintf(stderr, "numerics needs <snapshot>\n"); return 2; }
    o.snapshot = argv[2];
    i = 3;
    o.ms = {1, 2, 4, 8, 16};
  }
  for (; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--shape") o.shapes = split(v);
    else if (a == "--m") { o.ms.clear(); for (auto& t : split(v)) o.ms.push_back(uint32_t(std::stoul(t))); }
    else if (a == "--type") { o.types.clear(); for (auto& t : split(v)) o.types.push_back(t == "bf16" ? 'b' : 'i'); }
    else if (a == "--orient") { o.orients.clear(); for (auto& t : split(v)) o.orients.push_back(t == "wB" ? 'B' : 'A'); }
    else if (a == "--splitk") o.splitk = v;
    else if (a == "--wg") o.wgshape = v;
    else if (a == "--deq") o.deq = v;
    else if (a == "--prefetch") o.prefetch = uint32_t(std::stoul(v));
    else if (a == "--grf") o.grf = uint32_t(std::stoul(v));
    else if (a == "--s") o.s = uint32_t(std::stoul(v));
    else if (a == "--rounds") o.rounds = uint32_t(std::stoul(v));
    else if (a == "--launches") o.launches = uint32_t(std::stoul(v));
    else if (a == "--device") o.device = uint32_t(std::stoul(v));
    else if (a == "--trials") o.trials = uint32_t(std::stoul(v));
    else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 2; }
  }
  for (uint32_t M : o.ms)
    if (M < 1 || M > 16) { std::fprintf(stderr, "M %u outside [1, 16]\n", M); return 2; }
  l0::Context ctx(o.device);
  std::printf("device: %s; probe_small_m %s\n", ctx.name().c_str(), o.mode.c_str());
  if (o.mode == "check") return run_check(ctx, o);
  if (o.mode == "shapes") return run_shapes(ctx, o);
  if (o.mode == "levers") return run_levers(ctx, o);
  if (o.mode == "numerics") return run_numerics(ctx, o);
  if (o.mode == "prod") return run_prod(ctx, o);
  std::fprintf(stderr, "unknown mode %s\n", o.mode.c_str());
  return 2;
}

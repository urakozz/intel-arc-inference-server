// probe_vattn_ab - spec 28c's open error (docs/07-open-questions.md, "Recently answered"): attn_verify_dpas against attn_v2 on a REAL
// DFlash run, the operator's plan of 2026-11-04 step 2 (a direct A/B on the failing input).
//
//   probe_vattn_ab <snapshot> <ids file> <n> [--max-len 16384] [--bins a,b,...] [--thresh T] [--all]
//                  [--lm-head bf16|int8] [--follow v2|dpas] [--watch A,B] [--at G,G,...]
//
// b70-decode --ids <file> --n <n> --prefill --spec dflash's loop (generate_dflash, K = 7), with every verify run
// TWICE on the same state: forced to DPAS (CaptureOptions::force_attn, the lists recaptured) and then forced to v2 -
// v2 last, so the KV rows, GDN slots and taps the commit reads are v2's and the run is the plain stream (--follow dpas:
// v2 first and DPAS last - the commit reads DPAS's, the production stream at M = 8, e.g. dflash_gpu_test's K = 7
// run with --lm-head int8, the test's head). Per
// iteration a line (Control and the live GDN slot restored between the two verifies: the list moves pos, row 0 steps the slot in place): the rows where the two verify argmaxes differ inside the accepted prefix (where the DPAS run's
// stream would leave the plain one) and each row's max |logit difference|.
//
// At the first three iterations with a flip or a kept row whose logits differ by more than --thresh (4.0) (every
// iteration with --all) the verify is analysed layer by layer: the capture's
// attn_dump (CaptureOptions, test / probe only) holds each FA layer's attn_q / attn_gate / attn_out of both runs, and
// the engine's KV cache holds v2's keys, so each FA layer's attention is REPLAYED on v2's inputs - the production
// attn_verify_dpas binary (and every --bins variant: tools/probe/CMakeLists.txt's pvab_* builds of the same source,
// e.g. 128 GRF, -cl-opt-disable, the EAGER scalar walk), v2's pair, and the fp64 reference
// (tests/kernels/attn_verify_ref.h). Printed per layer: each arm's worst (row, q head) cosine and max |diff| against
// fp64 over the gated bf16 rows; then, at the worst (layer, row, head) of the production binary, its partials against
// v2's and fp64's block by block (mx, sm, acc cosine) - the first differing key block. The replays read the inputs of
// the run that went LAST (its q / gate dump and the K / V rows it left in the cache).
//
// Spec 28 (dflash_gpu_test t5_test-box's verify-vs-plain rows): --watch A,B prints, at every kept row, both arms'
// logit gap A - B (in bf16 ulps of the row's top logit) and argmaxes; --at G,... analyses the iterations whose kept
// rows decide the generated indices G (in place of the first three by --thresh). Index 0 is the prefill's id, as in
// the tests' streams (row r of an iteration whose input x is generated id g decides id g + r + 1).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cli/dflash_args.h"
#include "common/bf16.h"
#include "kernels/attn_verify_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/gemv_kind.h"

namespace {

namespace avr = attn_verify_ref;
constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kRow = kQH * kHD, kTgt = 32, kPart = 258, kM = 8;
constexpr uint32_t kV = model::Qwen35::kVocab;

std::vector<uint32_t> read_ids(const std::string& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot read " + p);
  std::vector<uint32_t> v;
  for (uint32_t x; f >> x;) v.push_back(x);
  return v;
}

double cos_bf(const uint16_t* a, const uint16_t* b, double* maxabs) {
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

// One FA layer's attention replayed: a decode kernel (DPAS variant or v2) + v2's reduce over the engine's KV.
struct Replay {
  l0::Context& ctx;
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem ctrl{ctx, l0::MemKind::Shared, sizeof(runtime::Control)};
  l0::Mem aq{ctx, l0::MemKind::Device, size_t(kM) * kRow * 4};
  l0::Mem ag{ctx, l0::MemKind::Device, size_t(kM) * kRow * 4};
  l0::Mem part{ctx, l0::MemKind::Device, size_t(kQH) * kTgt * kM * kPart * 4};
  l0::Mem out{ctx, l0::MemKind::Device, size_t(kM) * kRow * 2};
  std::vector<std::unique_ptr<l0::Module>> keep;
  std::vector<std::pair<std::string, l0::Module*>> mods;
  explicit Replay(l0::Context& c) : ctx(c) {}
  l0::Module& mod(const std::string& v) {
    for (auto& p : mods)
      if (p.first == v) return *p.second;
    keep.push_back(std::make_unique<l0::Module>(ctx, kernels::path(v)));
    mods.push_back({v, keep.back().get()});
    return *keep.back();
  }
  // `dec_variant` empty: v2's pair. Returns the gated bf16 rows [M][6144]; `parts` the partials [24][TGT][M][258].
  std::vector<uint16_t> run(const std::string& dec_variant, void* k, void* v, uint32_t pos, uint32_t M,
                            std::vector<float>* parts = nullptr) {
    runtime::Control* c = ctrl.as<runtime::Control>();
    c->pos = pos;
    c->n_active = M;
    const std::string v2 = kernels::attn_v2_variant(M, kTgt);
    l0::Module& rm = mod(v2);
    const bool dpas = !dec_variant.empty();
    l0::Module& dm = dpas ? mod(dec_variant) : rm;
    l0::Kernel dec(dm, dpas ? "attn_verify_dpas" : "attn_decode_v2");
    if (dpas) {
      ze_kernel_properties_t p{};
      p.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
      zeKernelGetProperties(dec.handle(), &p);
      dec.group_size(p.requiredGroupSizeX);
    } else {
      dec.group_size(256);
    }
    dec.arg_ptr(0, ctrl.ptr());
    dec.arg_ptr(1, aq.ptr());
    dec.arg_ptr(2, k);
    dec.arg_ptr(3, v);
    dec.arg_ptr(4, part.ptr());
    l0::Kernel red(rm, "attn_reduce_v2");
    red.group_size(256);
    red.arg_ptr(0, ctrl.ptr());
    red.arg_ptr(1, part.ptr());
    red.arg_ptr(2, ag.ptr());
    red.arg_ptr(3, out.ptr());
    imm.fill(part.ptr(), 0xFFFFFFFFu, part.size());   // NaN: a partial the kernel does not write must not be read
    imm.fill(out.ptr(), 0u, out.size());
    l0::CmdList l = l0::CmdList::regular(ctx);
    l.launch(dec, kKvH, kTgt);
    l.launch(red, kQH, M);
    l.close();
    q.execute(l, &fence);
    fence.wait();
    std::vector<uint16_t> o(size_t(M) * kRow);
    imm.copy(o.data(), out.ptr(), o.size() * 2);
    if (parts) {
      parts->resize(size_t(kQH) * kTgt * M * kPart);
      imm.copy(parts->data(), part.ptr(), parts->size() * 4);
    }
    return o;
  }
};

struct Dump {   // one run's attn_dump, host side
  std::vector<uint8_t> b;
  const float* q(uint32_t f) const { return reinterpret_cast<const float*>(&b[size_t(f) * runtime::kAttnDumpLayerBytes]); }
  const float* g(uint32_t f) const { return q(f) + kM * kRow; }
  const uint16_t* o(uint32_t f) const {
    return reinterpret_cast<const uint16_t*>(&b[size_t(f) * runtime::kAttnDumpLayerBytes + size_t(kM) * kRow * 8]);
  }
};

// `in`: the dump of the run that went last (its q / gate and the K / V it wrote are the replays' inputs); `other`: the
// other run's; `in_v2`: the last run was v2's.
void analyse(runtime::Engine& eng, Replay& rp, const Dump& in, const Dump& other, bool in_v2, uint32_t pos, uint32_t M,
             uint32_t n_fa, const std::vector<std::string>& bins) {
  const Dump& dv = in;
  const Dump& dd = other;
  runtime::DecodeBuffers& b = eng.buffers();
  const uint32_t keys = pos + M;
  std::printf("  analysis at pos %u, M %u (%u keys at the last row)\n", pos, M, keys);
  struct W { double c = 2, ab = 0; uint32_t m = 0, h = 0; };
  double worst_c = 2;
  uint32_t wf = 0, wm = 0, wh = 0;
  for (uint32_t f = 0; f < n_fa; ++f) {
    const runtime::KvLayer kv = b.kv_lay.layer(b.kv_k.ptr(), b.kv_v.ptr(), f);
    // host K / V rows [0, keys) for the reference
    std::vector<uint16_t> hk(size_t(keys) * kKvH * kHD), hv(hk.size());
    rp.imm.copy(hk.data(), kv.k, hk.size() * 2);
    rp.imm.copy(hv.data(), kv.v, hv.size() * 2);
    rp.imm.copy(rp.aq.ptr(), dv.q(f), size_t(M) * kRow * 4);
    rp.imm.copy(rp.ag.ptr(), dv.g(f), size_t(M) * kRow * 4);
    const avr::Result ref = avr::run(pos, M, dv.q(f), dv.g(f), avr::Kv::bf16(hk.data(), hv.data()));
    // the run's own outputs: the last run's dump against its kernel's replay (fidelity), the other run's dump against
    // the replays (drift)
    const std::vector<uint16_t> o2 = rp.run("", kv.k, kv.v, pos, M);
    const std::vector<uint16_t> olast = in_v2 ? o2 : rp.run(bins[0], kv.k, kv.v, pos, M);
    const bool same = std::memcmp(olast.data(), dv.o(f), olast.size() * 2) == 0;
    const bool qsame = std::memcmp(dd.q(f), dv.q(f), size_t(M) * kRow * 4) == 0;
    std::ostringstream line;
    char buf[256];
    auto grade = [&](const std::vector<uint16_t>& o, const char* name, bool track) {
      W w;
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t h = 0; h < kQH; ++h) {
          double ab = 0;
          const size_t k = size_t(m) * kRow + h * kHD;
          const double c = cos_bf(&o[k], &ref.out[k], &ab);
          w.ab = std::max(w.ab, ab);
          if (c < w.c) w.c = c, w.m = m, w.h = h;
        }
      std::snprintf(buf, sizeof buf, "  %s %.8f (r%u h%u, max|d| %.3g)", name, w.c, w.m, w.h, w.ab);
      line << buf;
      if (track && w.c < worst_c) worst_c = w.c, wf = f, wm = w.m, wh = w.h;
    };
    grade(o2, "v2", false);
    for (size_t i = 0; i < bins.size(); ++i) {
      const std::vector<uint16_t> o = rp.run(bins[i], kv.k, kv.v, pos, M);
      grade(o, bins[i].c_str(), i == 0);
    }
    // the DPAS run's own dump vs fp64 on v2's inputs only means something where its q equals v2's
    if (qsame) grade(std::vector<uint16_t>(dd.o(f), dd.o(f) + size_t(M) * kRow), in_v2 ? "dpas-run" : "v2-run", false);
    std::printf("  FA %2u: replay==%s run %s, q dpas-run==v2-run %s |%s\n", f, in_v2 ? "v2" : "dpas",
                same ? "yes" : "NO", qsame ? "yes" : "no", line.str().c_str());
  }
  // partials at the production binary's worst (layer, row, head)
  std::printf("  worst of %s: FA %u row %u q head %u (cos %.8f); its blocks:\n", bins[0].c_str(), wf, wm, wh, worst_c);
  const runtime::KvLayer kv = b.kv_lay.layer(b.kv_k.ptr(), b.kv_v.ptr(), wf);
  std::vector<uint16_t> hk(size_t(keys) * kKvH * kHD), hv(hk.size());
  rp.imm.copy(hk.data(), kv.k, hk.size() * 2);
  rp.imm.copy(hv.data(), kv.v, hv.size() * 2);
  rp.imm.copy(rp.aq.ptr(), dv.q(wf), size_t(M) * kRow * 4);
  rp.imm.copy(rp.ag.ptr(), dv.g(wf), size_t(M) * kRow * 4);
  std::vector<float> pd, p2;
  rp.run(bins[0], kv.k, kv.v, pos, M, &pd);
  rp.run("", kv.k, kv.v, pos, M, &p2);
  const float* qrow = dv.q(wf) + (size_t(wm) * kQH + wh) * kHD;
  const std::vector<avr::Block> rb = avr::row_blocks(qrow, avr::Kv::bf16(hk.data(), hv.data()), wh / 6, pos, wm);
  for (uint32_t blk = 0; blk < rb.size(); ++blk) {
    const size_t at = ((size_t(wh) * kTgt + blk) * M + wm) * kPart;
    const float* a = &pd[at];
    const float* c = &p2[at];
    const double ca = avr::cos_q(a + 2, rb[blk].acc), cc = avr::cos_q(c + 2, rb[blk].acc);
    std::printf("    blk %2u: mx dpas %.6f v2 %.6f fp64 %.6f | sm dpas %.6f v2 %.6f fp64 %.6f | acc cos dpas %.8f v2 %.8f%s\n",
                blk, a[0], c[0], rb[blk].mx, a[1], c[1], rb[blk].sm, ca, cc,
                (ca < 0.9999 || std::fabs(a[1] - rb[blk].sm) > 1e-3 * rb[blk].sm) ? "  <--" : "");
  }
  // the row's scores near its bound: which keys the reference sees above the row max
  const uint32_t bound = pos + wm;
  std::printf("    row %u bound %u; fp64 scores of keys bound-3 .. bound+%u:", wm, bound, M - 1 - wm);
  for (uint32_t p = bound >= 3 ? bound - 3 : 0; p < keys; ++p)
    std::printf(" %u:%.3f", p, avr::score(qrow, avr::Kv::bf16(hk.data(), hv.data()), wh / 6, p));
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: probe_vattn_ab <snapshot> <ids> <n> [--max-len L] [--bins a,b] [--all] [--lm-head bf16|int8] "
                 "[--follow v2|dpas] [--watch A,B] [--at G,...]\n");
    return 2;
  }
  const std::string snap = argv[1];
  const std::vector<uint32_t> ids = read_ids(argv[2]);
  const uint32_t n = uint32_t(std::stoul(argv[3]));
  uint32_t max_len = 16384;
  bool all = false;
  double thresh = 4.0;   // analyse an iteration whose kept rows differ by more than this (logits), at most 3
  std::vector<std::string> bins = {kernels::attn_verify_dpas_variant(kM, kTgt, false)};
  loader::LmHeadForm head = loader::LmHeadForm::Checkpoint;
  bool follow_dpas = false, watch = false;
  uint32_t watch_a = 0, watch_b = 0;
  std::vector<uint32_t> at;   // generated indices whose iteration is analysed
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--max-len" && i + 1 < argc) {
      max_len = uint32_t(std::stoul(argv[++i]));
    } else if (a == "--bins" && i + 1 < argc) {
      std::stringstream s(argv[++i]);
      for (std::string t; std::getline(s, t, ',');)
        if (!t.empty()) bins.push_back(t);
    } else if (a == "--thresh" && i + 1 < argc) {
      thresh = std::stod(argv[++i]);
    } else if (a == "--all") {
      all = true;
    } else if (a == "--lm-head" && i + 1 < argc) {
      if (!loader::parse_lm_head_form(argv[++i], head)) throw std::runtime_error("--lm-head bf16|int8");
    } else if (a == "--follow" && i + 1 < argc) {
      const std::string f = argv[++i];
      if (f != "v2" && f != "dpas") throw std::runtime_error("--follow v2|dpas");
      follow_dpas = f == "dpas";
    } else if (a == "--watch" && i + 1 < argc) {
      if (std::sscanf(argv[++i], "%u,%u", &watch_a, &watch_b) != 2) throw std::runtime_error("--watch A,B");
      watch = true;
    } else if (a == "--at" && i + 1 < argc) {
      std::stringstream s(argv[++i]);
      for (std::string t; std::getline(s, t, ',');)
        if (!t.empty()) at.push_back(uint32_t(std::stoul(t)));
    } else {
      throw std::runtime_error("unknown argument " + a);
    }
  }
  l0::Context ctx(0u);
  std::printf("probe_vattn_ab on %s; arms:", ctx.name().c_str());
  for (const auto& s : bins) std::printf(" %s", s.c_str());
  std::printf("\n");
  loader::LoadedModel model = loader::load(ctx, snap, max_len, /*mtp=*/false, head);
  std::printf("lm_head %s, the stream follows %s\n", loader::lm_head_form_name(head), follow_dpas ? "DPAS" : "v2");
  cli::dflash::Args da;
  da.on = true;
  cli::dflash::Loaded dl = cli::dflash::load(ctx, da, model);
  const uint32_t n_fa = model.desc->fa_layers;
  auto drafter = std::make_unique<runtime::dflash::DFlashDrafter>(ctx, std::move(dl.w), dl.head_rows, max_len, da.max_k);
  runtime::Engine eng(ctx, std::move(model), max_len, /*debug_resid=*/false, runtime::KvCache::Bf16, drafter.get());
  drafter->attach(eng, /*draft_vocab=*/false);
  eng.prepare_prefill();
  eng.prefill(ids, 0);
  std::printf("prefill: %zu ids, pos %u, %u FA layers\n", ids.size(), eng.pos(), n_fa);

  l0::Mem dump_d(ctx, l0::MemKind::Device, size_t(n_fa) * runtime::kAttnDumpLayerBytes);
  l0::Mem dump_v(ctx, l0::MemKind::Device, size_t(n_fa) * runtime::kAttnDumpLayerBytes);
  runtime::CaptureOptions od = runtime::attn_forced(runtime::VerifyAttnForce::Dpas);
  od.attn_dump = dump_d.ptr();
  runtime::CaptureOptions ov = runtime::attn_forced(runtime::VerifyAttnForce::V2);
  ov.attn_dump = dump_v.ptr();
  Replay rp(ctx);
  const size_t gdn_bytes = eng.buffers().gdn_state.size();
  l0::Mem gdn_keep(ctx, l0::MemKind::Device, gdn_bytes);
  std::vector<float> ld(size_t(kM) * kV), lv(size_t(kM) * kV);
  std::vector<uint32_t> stream;
  uint32_t iter = 0, flips = 0, analysed = 0;
  runtime::TapDrafter& d = *eng.tap_drafter();
  while (stream.size() < n) {
    const uint32_t k = std::min(da.max_k, eng.max_verify_k());
    const uint32_t x = eng.buffers().control.as<runtime::Control>()->cur_token[0];
    const uint32_t pos = eng.pos();
    if (k == 0) {
      eng.verify(0);
      stream.push_back(x);
      eng.commit(0, eng.verify_ids()[0]);
      ++iter;
      continue;
    }
    const runtime::dflash::DFlashDraftOut& o = d.draft(eng, k, 0.0f, 0);
    const std::vector<uint32_t> drafts(o.ids, o.ids + k);
    // verify row 0 steps the LIVE GDN slot in place (gdn_step.cl SPEC_SLOTS), so the second verify needs it back
    const uint32_t live = eng.buffers().control.as<runtime::Control>()->gdn_live;
    void* slot = live == 0 ? eng.buffers().gdn_state.ptr()
                           : static_cast<uint8_t*>(eng.drafter_slots()->ptr()) + size_t(live - 1) * gdn_bytes;
    rp.imm.copy(gdn_keep.ptr(), slot, gdn_bytes);
    runtime::Control* ctl = eng.buffers().control.as<runtime::Control>();
    const runtime::Control ctl_keep = *ctl;   // the list moves pos on the device
    std::vector<uint32_t> idd, idv;
    auto run_arm = [&](bool dpas) {   // one verify on the kept state; the second restores Control and the live slot
      eng.recapture_verify(dpas ? od : ov);
      eng.verify(k);
      (dpas ? idd : idv).assign(eng.verify_ids(), eng.verify_ids() + k + 1);
      rp.imm.copy((dpas ? ld : lv).data(), eng.verify_logits_device(), size_t(k + 1) * kV * 4);
    };
    run_arm(!follow_dpas);
    rp.imm.copy(slot, gdn_keep.ptr(), gdn_bytes);
    *ctl = ctl_keep;
    run_arm(follow_dpas);
    const std::vector<uint32_t>& idf = follow_dpas ? idd : idv;   // the run the stream follows
    uint32_t jv = 0;
    while (jv < k && drafts[jv] == idv[jv]) ++jv;
    uint32_t jd = 0;
    while (jd < k && drafts[jd] == idd[jd]) ++jd;
    const uint32_t j = follow_dpas ? jd : jv;
    // the other run leaves the followed stream at the first row r <= j whose argmax differs
    int first = -1;
    for (uint32_t r = 0; r <= j; ++r)
      if (idd[r] != idv[r]) {
        first = int(r);
        break;
      }
    std::printf("iter %3u pos %4u gen %3zu k %u accept v2 %u dpas %u |", iter, pos, stream.size(), k, jv, jd);
    double worst_kept = 0;   // the largest |logit difference| over the rows the stream keeps (0 .. j)
    for (uint32_t r = 0; r <= k; ++r) {
      double mx = 0;
      for (uint32_t t = 0; t < kV; ++t)
        mx = std::max(mx, double(std::fabs(ld[size_t(r) * kV + t] - lv[size_t(r) * kV + t])));
      if (r <= j) worst_kept = std::max(worst_kept, mx);
      std::printf(" %.3f%s", mx, idd[r] != idv[r] ? "*" : "");
    }
    std::printf("\n");
    if (first >= 0) {
      ++flips;
      const uint32_t r = uint32_t(first);
      const float* a = &lv[size_t(r) * kV];
      const float* b2 = &ld[size_t(r) * kV];
      std::printf("  FLIP at row %u (generated index %zu): v2 id %u (v2 logit %.4f, dpas logit %.4f), dpas id %u (v2 logit "
                  "%.4f, dpas logit %.4f); drafts:",
                  r, stream.size() + r, idv[r], a[idv[r]], b2[idv[r]], idd[r], a[idd[r]], b2[idd[r]]);
      for (uint32_t t : drafts) std::printf(" %u", t);
      std::printf("; input ids: %u", x);
      for (uint32_t t = 0; t < k; ++t) std::printf(" %u", drafts[t]);
      std::printf("\n");
    }
    if (watch)
      for (uint32_t r = 0; r <= j; ++r) {
        auto gap = [&](const std::vector<float>& L, uint32_t& top) {
          const float* row = &L[size_t(r) * kV];
          top = 0;
          for (uint32_t t = 1; t < model::Qwen35::kVocabUsed; ++t)
            if (row[t] > row[top]) top = t;
          int e = 0;
          std::frexp(std::fabs(row[top]), &e);
          return double(row[watch_a] - row[watch_b]) / std::ldexp(1.0, e - 8);   // golden::bf16_ulp_at
        };
        uint32_t tv = 0, td = 0;
        const double gv = gap(lv, tv), gd = gap(ld, td);
        std::printf("  watch: row %u decides generated id %zu: %u - %u: v2 %+.2f ulp (top %u), dpas %+.2f ulp (top %u)\n",
                    r, stream.size() + r + 1, watch_a, watch_b, gv, tv, gd, td);
      }
    bool at_hit = false;
    for (uint32_t g : at) at_hit |= g > stream.size() && g <= stream.size() + j + 1;   // row g - stream.size() - 1
    if (((first >= 0 || worst_kept > thresh) && analysed < 3 && at.empty()) || all || at_hit) {
      Dump hd, hv;
      hd.b.resize(dump_d.size());
      hv.b.resize(dump_v.size());
      rp.imm.copy(hd.b.data(), dump_d.ptr(), hd.b.size());
      rp.imm.copy(hv.b.data(), dump_v.ptr(), hv.b.size());
      if (follow_dpas)
        analyse(eng, rp, hd, hv, false, pos, k + 1, n_fa, bins);
      else
        analyse(eng, rp, hv, hd, true, pos, k + 1, n_fa, bins);
      ++analysed;
    }
    stream.push_back(x);
    for (uint32_t i = 0; i < j; ++i) stream.push_back(drafts[i]);
    eng.commit(j, idf[j]);
    ++iter;
  }
  std::printf("probe_vattn_ab: %u iterations, %u with a flip inside the accepted prefix\n", iter, flips);
  std::printf("stream:");
  for (size_t i = 0; i < std::min<size_t>(stream.size(), n); ++i) std::printf(" %u", stream[i]);
  std::printf("\n");
  return 0;
}

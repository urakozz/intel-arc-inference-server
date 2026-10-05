// Spec 8 §11 on the card: the draft vocabulary's kernels, at the real head shape
// (K 5120 x N 248320, vocab_used 248077), at every compiled |V'| (32k / 64k / 128k), for
// both head forms - int8 (`--lm-head int8`, gemv_i8w) and bf16 (`--lm-head bf16`,
// gemv_bf16 at lm_head's {64, 1} tiling).
//
//   1. **The compact GEMV is the full one, bitwise, row for row.** V' is chosen by
//      loader::select_draft_vocab (Qwen3.8's 33 added tokens, a random ranked list, the
//      lowest ids) and its rows gathered by loader::gather_int8_tiled_rows /
//      gather_bf16_tiled_rows - the loader's own path. The GEMV at N = |V'| over the
//      compact head must give, at compact column j, exactly the bits the same GEMV at
//      N = 248320 gives at column ids[j]: the same source, the same per-column instruction
//      sequence, only the grid differs. This is what makes a V' draft's logits the full
//      head's logits restricted to V'.
//   2. **The mapped argmax is the full output's argmax restricted to V'.** The draft list's
//      tail as capture.cc records it (compact GEMV, dv_argmax_stage1, dv_argmax_stage2) on
//      one closed list: out_token[0] and cur_token[0] equal the host argmax of the FULL
//      GEMV output over V''s ids (ties to the lowest id), pos advances by 1 per replay,
//      and a second replay gives the same id. The scatter leaves the full-vocabulary row
//      equal to the full output at V''s ids and -inf everywhere else, bitwise.
//   3. **Planted cases** (once, on the int8 run's binaries - the stages do not depend on
//      the head form) on synthetic compact logits: an exact tie (the lower index = the
//      lower id wins), a row of -inf with two finite values, NaNs (never win); and
//      n_active != 1 writes nothing.
//
// It also prints the tail's time against the full GEMV's (informational; the in-situ
// number is mtp_gpu_test --bench's).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "kernels/kernels.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/draft_vocab.h"
#include "loader/lm_head_int8.h"
#include "runtime/control.h"

namespace {
constexpr uint32_t K = 5120, N = 248320, kUsed = 248077;
constexpr uint32_t kNegInf = 0xFF800000u;

// The host argmax over `ids` of `full` (ids ascending: the first strict maximum is the
// lowest id among equal maxima; NaN never compares greater).
uint32_t argmax_over(const std::vector<float>& full, const std::vector<uint32_t>& ids) {
  float bv = -INFINITY;
  uint32_t best = 0x7FFFFFFFu;
  for (uint32_t id : ids)
    if (full[id] > bv || (full[id] == bv && id < best)) {
      bv = full[id];
      best = id;
    }
  return best;
}

struct Env {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem xbuf{ctx, l0::MemKind::Device, size_t(K) * 2};
  l0::Mem row{ctx, l0::MemKind::Device, size_t(N) * 4};   // the draft's full-vocabulary row
  l0::Mem part{ctx, l0::MemKind::Device, 256 * 2 * 4};
  l0::Mem ctrl_mem{ctx, l0::MemKind::Shared, sizeof(runtime::Control)};
  runtime::Control* ctrl = ctrl_mem.as<runtime::Control>();
  std::vector<uint32_t> added, eos, ranked;
  Env() {
    const std::vector<uint16_t> x = random_bf16(K, 4242);
    imm.copy(xbuf.ptr(), x.data(), size_t(K) * 2);
    for (uint32_t id = 248044; id <= 248076; ++id) added.push_back(id);
    eos = {248046, 248044};
    ranked.resize(60000);
    std::mt19937 g(17);
    for (uint32_t& id : ranked) id = g() % N;
  }
};

// One head form's weights on the device and how to bind its GEMV. `elem` bytes per weight;
// int8 also has fp32 row scales.
struct Head {
  bool int8;
  l0::Mem w, s;
  Head(l0::Context& ctx, bool i8, uint32_t n)
      : int8(i8),
        w(ctx, l0::MemKind::Device, size_t(n) * K * (i8 ? 1 : 2)),
        s(ctx, l0::MemKind::Device, i8 ? size_t(n) * 4 : 4) {}
  std::string variant(uint32_t n) const {
    return int8 ? kernels::gemv_i8w_variant(1, K, n)
                : kernels::gemv_bf16_variant(1, K, n, kernels::gemv_bf16_tiling(n));
  }
  // gemv_i8w(w, scales, x, out) / gemv_bf16(w, x, out); both WG 64, grid n / 64.
  void bind(l0::Kernel& k, const l0::Mem& x, const l0::Mem& out) const {
    int a = 0;
    k.arg_ptr(a++, w.ptr());
    if (int8) k.arg_ptr(a++, s.ptr());
    k.arg_ptr(a++, x.ptr());
    k.arg_ptr(a, out.ptr());
  }
};

// The planted argmax cases on the stage binary of size nv (see the header).
int planted(Env& e, l0::Kernel& k1, l0::Kernel& k2, const l0::Mem& ids_dev,
            const std::vector<uint32_t>& ids, uint32_t nv) {
  int failures = 0;
  std::mt19937 g(nv);
  for (int c = 0; c < 3; ++c) {
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> lc(nv);
    for (float& v : lc) v = nd(g);
    if (c == 0) {   // an exact tie for the maximum
      const uint32_t a = g() % nv, b = g() % nv;
      lc[a] = lc[b] = 100.f;
    } else if (c == 1) {   // -inf but two
      std::fill(lc.begin(), lc.end(), -INFINITY);
      lc[g() % nv] = -5.f;
      lc[nv - 1] = -5.f;
    } else {   // NaNs never win
      for (int i = 0; i < 64; ++i) lc[g() % nv] = NAN;
    }
    l0::Mem syn(e.ctx, l0::MemKind::Device, size_t(nv) * 4);
    e.imm.copy(syn.ptr(), lc.data(), size_t(nv) * 4);
    k1.arg_ptr(0, syn.ptr());
    k1.arg_ptr(1, ids_dev.ptr());
    k1.arg_ptr(2, e.part.ptr());
    k1.arg_ptr(3, e.row.ptr());
    k2.arg_ptr(0, e.ctrl_mem.ptr());
    k2.arg_ptr(1, e.part.ptr());
    k2.arg_ptr(2, ids_dev.ptr());
    l0::CmdList st = l0::CmdList::regular(e.ctx);
    st.launch(k1, nv / kernels::kDraftVocabChunk);
    st.launch(k2, 1);
    st.close();
    std::memset(e.ctrl, 0, sizeof *e.ctrl);
    e.ctrl->pos = 7;
    e.ctrl->n_active = 1;
    e.q.execute(st, &e.fence);
    e.fence.wait();
    std::vector<float> f(N, -INFINITY);
    for (uint32_t j = 0; j < nv; ++j) f[ids[j]] = lc[j];
    const uint32_t w = argmax_over(f, ids);
    const bool ok = e.ctrl->out_token[0] == w && e.ctrl->cur_token[0] == w && e.ctrl->pos == 8;
    std::printf("  planted case %d: id %u want %u %s\n", c, e.ctrl->out_token[0], w, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
    // n_active 2 (not the draft list's 1): stage 2 writes nothing.
    std::memset(e.ctrl, 0, sizeof *e.ctrl);
    e.ctrl->pos = 9;
    e.ctrl->n_active = 2;
    e.ctrl->out_token[0] = 88;
    e.ctrl->cur_token[0] = 77;
    e.q.execute(st, &e.fence);
    e.fence.wait();
    if (!(e.ctrl->pos == 9 && e.ctrl->out_token[0] == 88 && e.ctrl->cur_token[0] == 77)) {
      std::printf("  n_active 2 wrote the control block: FAIL\n");
      ++failures;
    }
  }
  return failures;
}

// Cases 1 and 2 (and 3 when `plant`) for one head form.
int run_form(Env& e, bool int8, bool plant) {
  // A random head straight in the tiled layout (the bitwise property does not care where
  // the bytes came from): int8 bytes and fp32 row scales, or bf16 words of N(0, 0.02).
  std::vector<int8_t> h8;
  std::vector<float> scales;
  std::vector<uint16_t> h16;
  std::mt19937 g(int8 ? 811 : 812);
  if (int8) {
    h8.resize(size_t(N) * K);
    for (size_t i = 0; i < h8.size(); i += 4) {
      const uint32_t r = g();
      for (int b = 0; b < 4; ++b) {
        const int8_t v = int8_t(r >> (8 * b));
        h8[i + b] = v == -128 ? 0 : v;
      }
    }
    scales.resize(N);
    std::uniform_real_distribution<float> sd(1e-4f, 2e-3f);
    for (float& v : scales) v = sd(g);
  } else {
    h16.resize(size_t(N) * K);
    std::normal_distribution<float> wd(0.f, 0.02f);
    for (uint16_t& v : h16) v = common::f32_to_bf16(wd(g));
  }
  Head full(e.ctx, int8, N);
  if (int8) {
    e.imm.copy(full.w.ptr(), h8.data(), h8.size());
    e.imm.copy(full.s.ptr(), scales.data(), size_t(N) * 4);
  } else {
    e.imm.copy(full.w.ptr(), h16.data(), h16.size() * 2);
  }
  const char* entry = int8 ? "gemv_i8w" : "gemv_bf16";
  const char* form = int8 ? "int8" : "bf16";

  // The full GEMV, once.
  std::vector<float> fout(N);
  l0::Mem ofull(e.ctx, l0::MemKind::Device, size_t(N) * 4);
  l0::Module mfull(e.ctx, kernels::path(full.variant(N)));
  l0::Kernel kfull = mfull.kernel(entry);
  kfull.group_size(kernels::kGemvI8wCols);   // 64 for both: gemv_bf16's {64, 1} at lm_head
  full.bind(kfull, e.xbuf, ofull);
  double us_full = 0;
  {
    l0::CmdList list = l0::CmdList::regular(e.ctx);
    list.launch(kfull, N / 64);
    list.close();
    e.q.execute(list, &e.fence);
    e.fence.wait();
    e.imm.copy(fout.data(), ofull.ptr(), size_t(N) * 4);
    us_full = time_list(e.q, e.fence, list, 1);
  }

  int failures = 0;
  for (uint32_t nv : loader::kDraftVocabSizes) {
    CHECK(int8 || (kernels::gemv_bf16_tiling(nv).cols == 64 && kernels::gemv_bf16_tiling(nv).ksplit == 1));
    const std::vector<uint32_t> ids = loader::select_draft_vocab(e.added, e.eos, e.ranked, kUsed, nv);
    Head comp(e.ctx, int8, nv);
    if (int8) {
      std::vector<int8_t> cw(size_t(nv) * K);
      std::vector<float> cs(nv);
      loader::gather_int8_tiled_rows(h8.data(), scales.data(), K, N, ids.data(), nv, cw.data(), cs.data());
      e.imm.copy(comp.w.ptr(), cw.data(), cw.size());
      e.imm.copy(comp.s.ptr(), cs.data(), size_t(nv) * 4);
    } else {
      std::vector<uint16_t> cw(size_t(nv) * K);
      loader::gather_bf16_tiled_rows(h16.data(), K, N, ids.data(), nv, cw.data());
      e.imm.copy(comp.w.ptr(), cw.data(), cw.size() * 2);
    }
    l0::Mem ids_dev(e.ctx, l0::MemKind::Device, size_t(nv) * 4);
    e.imm.copy(ids_dev.ptr(), ids.data(), size_t(nv) * 4);
    l0::Mem logits_c(e.ctx, l0::MemKind::Device, size_t(nv) * 4);

    l0::Module mg(e.ctx, kernels::path(comp.variant(nv)));
    l0::Module ma(e.ctx, kernels::path(kernels::dv_argmax_variant(nv)));
    l0::Kernel kg = mg.kernel(entry);
    kg.group_size(64);
    l0::Kernel k1 = ma.kernel("dv_argmax_stage1");
    k1.group_size(256);
    l0::Kernel k2 = ma.kernel("dv_argmax_stage2");
    k2.group_size(256);

    // 1 + 2: the draft list's tail, as captured.
    comp.bind(kg, e.xbuf, logits_c);
    k1.arg_ptr(0, logits_c.ptr());
    k1.arg_ptr(1, ids_dev.ptr());
    k1.arg_ptr(2, e.part.ptr());
    k1.arg_ptr(3, e.row.ptr());
    k2.arg_ptr(0, e.ctrl_mem.ptr());
    k2.arg_ptr(1, e.part.ptr());
    k2.arg_ptr(2, ids_dev.ptr());
    l0::CmdList tail = l0::CmdList::regular(e.ctx);
    tail.launch(kg, nv / 64);
    tail.launch(k1, nv / kernels::kDraftVocabChunk);
    tail.launch(k2, 1);
    tail.close();
    e.imm.fill(e.row.ptr(), kNegInf, e.row.size());
    std::memset(e.ctrl, 0, sizeof *e.ctrl);
    e.ctrl->pos = 100;
    e.ctrl->n_active = 1;
    e.ctrl->cur_token[0] = 12345;
    e.q.execute(tail, &e.fence);
    e.fence.wait();

    std::vector<float> comp_out(nv), row_out(N);
    e.imm.copy(comp_out.data(), logits_c.ptr(), size_t(nv) * 4);
    e.imm.copy(row_out.data(), e.row.ptr(), size_t(N) * 4);
    uint32_t differ = 0;
    for (uint32_t j = 0; j < nv; ++j)
      if (std::memcmp(&comp_out[j], &fout[ids[j]], 4) != 0) ++differ;
    const uint32_t want = argmax_over(fout, ids);
    const bool tok_ok =
        e.ctrl->out_token[0] == want && e.ctrl->cur_token[0] == want && e.ctrl->pos == 101;
    std::vector<uint8_t> in(N, 0);
    for (uint32_t id : ids) in[id] = 1;
    uint32_t row_bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
      const float w = in[i] ? fout[i] : -INFINITY;
      if (std::memcmp(&row_out[i], &w, 4) != 0) ++row_bad;
    }
    // The full-vocabulary argmax for reference: where V' agrees with it, acceptance holds.
    std::vector<uint32_t> all_used(kUsed);
    for (uint32_t i = 0; i < kUsed; ++i) all_used[i] = i;
    const uint32_t full_best = argmax_over(fout, all_used);
    e.q.execute(tail, &e.fence);   // a replay: the same id, pos + 1 again
    e.fence.wait();
    const bool replay_ok = e.ctrl->out_token[0] == want && e.ctrl->pos == 102;
    const double us_c = time_list(e.q, e.fence, tail, 1);
    std::printf("%s |V'| %6u: compact gemv %u/%u columns differ from the full head's (bar 0); "
                "argmax id %u (want %u over V', full-vocab %u) pos %u %s; replay %s; scatter row "
                "%u entries off; tail %.1f us vs full gemv %.1f us\n",
                form, nv, differ, nv, e.ctrl->out_token[0], want, full_best, e.ctrl->pos,
                tok_ok ? "ok" : "FAIL", replay_ok ? "ok" : "FAIL", row_bad, us_c, us_full);
    if (differ || !tok_ok || !replay_ok || row_bad) ++failures;
    if (plant) failures += planted(e, k1, k2, ids_dev, ids, nv);
  }
  return failures;
}
}  // namespace

int main() {
  Env e;
  int failures = run_form(e, /*int8=*/true, /*plant=*/true);
  failures += run_form(e, /*int8=*/false, /*plant=*/false);
  std::puts(failures ? "draft_vocab_kernels_test FAIL" : "draft_vocab_kernels_test OK");
  return failures ? 1 : 0;
}

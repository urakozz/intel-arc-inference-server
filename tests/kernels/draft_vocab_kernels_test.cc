// Spec 8 §11 on the card: the draft vocabulary's kernels, at the real head shape
// (K 5120 x N 248320 int8, vocab_used 248077), at every compiled |V'| (32k / 64k / 128k).
//
//   1. **The compact GEMV is the full one, bitwise, row for row.** V' is chosen by
//      loader::select_draft_vocab (Qwen3.8's 33 added tokens, a random ranked list, the
//      lowest ids) and its rows gathered by loader::gather_int8_tiled_rows - the loader's
//      own path. gemv_i8w at N = |V'| over the compact head must give, at compact column
//      j, exactly the bits gemv_i8w at N = 248320 gives at column ids[j]: the same source,
//      the same per-column instruction sequence, only the grid differs. This is what makes
//      a V' draft's logits the full head's logits restricted to V'.
//   2. **The mapped argmax is the full output's argmax restricted to V'.** The draft list's
//      tail as capture.cc records it (compact GEMV, dv_argmax_stage1, dv_argmax_stage2) on
//      one closed list: out_token[0] and cur_token[0] equal the host argmax of the FULL
//      GEMV output over V''s ids (ties to the lowest id), pos advances by 1 per replay,
//      and a second replay gives the same id. The scatter leaves the full-vocabulary row
//      equal to the full output at V''s ids and -inf everywhere else, bitwise.
//   3. **Planted cases** on synthetic compact logits: an exact tie (the lower index =
//      the lower id wins), a row of -inf with two finite values, NaNs (never win); and
//      n_active != 1 writes nothing.
//
// It also prints the compact GEMV's time against the full one's (informational; the
// in-situ number is mtp_gpu_test --bench's).
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

struct Dv {
  l0::Mem w, s, ids, logits_c;
  Dv(l0::Context& ctx, uint32_t nv)
      : w(ctx, l0::MemKind::Device, size_t(nv) * K),
        s(ctx, l0::MemKind::Device, size_t(nv) * 4),
        ids(ctx, l0::MemKind::Device, size_t(nv) * 4),
        logits_c(ctx, l0::MemKind::Device, size_t(nv) * 4) {}
};
}  // namespace

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // A random int8 head straight in the tiled layout (the bitwise property does not care
  // where the bytes came from) and random row scales.
  std::vector<int8_t> head(size_t(N) * K);
  std::vector<float> scales(N);
  {
    std::mt19937 g(811);
    for (size_t i = 0; i < head.size(); i += 4) {
      const uint32_t r = g();
      for (int b = 0; b < 4; ++b) {
        const int8_t v = int8_t(r >> (8 * b));
        head[i + b] = v == -128 ? 0 : v;
      }
    }
    std::uniform_real_distribution<float> sd(1e-4f, 2e-3f);
    for (float& v : scales) v = sd(g);
  }
  l0::Mem wfull(ctx, l0::MemKind::Device, head.size()), sfull(ctx, l0::MemKind::Device, size_t(N) * 4);
  imm.copy(wfull.ptr(), head.data(), head.size());
  imm.copy(sfull.ptr(), scales.data(), size_t(N) * 4);
  const std::vector<uint16_t> x = random_bf16(K, 4242);
  l0::Mem xbuf(ctx, l0::MemKind::Device, size_t(K) * 2);
  imm.copy(xbuf.ptr(), x.data(), size_t(K) * 2);

  // The full GEMV, once.
  std::vector<float> full(N);
  l0::Mem ofull(ctx, l0::MemKind::Device, size_t(N) * 4);
  l0::Module mfull(ctx, kernels::path(kernels::gemv_i8w_variant(1, K, N)));
  l0::Kernel kfull = mfull.kernel("gemv_i8w");
  kfull.group_size(kernels::kGemvI8wCols);
  kfull.arg_ptr(0, wfull.ptr());
  kfull.arg_ptr(1, sfull.ptr());
  kfull.arg_ptr(2, xbuf.ptr());
  kfull.arg_ptr(3, ofull.ptr());
  double us_full = 0;
  {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(kfull, N / kernels::kGemvI8wCols);
    list.close();
    q.execute(list, &fence);
    fence.wait();
    imm.copy(full.data(), ofull.ptr(), size_t(N) * 4);
    us_full = time_list(q, fence, list, 1);
  }

  std::vector<uint32_t> added;
  for (uint32_t id = 248044; id <= 248076; ++id) added.push_back(id);
  const std::vector<uint32_t> eos = {248046, 248044};
  std::vector<uint32_t> ranked(60000);
  {
    std::mt19937 g(17);
    for (uint32_t& id : ranked) id = g() % N;
  }

  l0::Mem row(ctx, l0::MemKind::Device, size_t(N) * 4);   // the draft's full-vocabulary row
  l0::Mem part(ctx, l0::MemKind::Device, 256 * 2 * 4);
  l0::Mem ctrl_mem(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  auto* ctrl = ctrl_mem.as<runtime::Control>();
  int failures = 0;

  for (uint32_t nv : loader::kDraftVocabSizes) {
    const std::vector<uint32_t> ids = loader::select_draft_vocab(added, eos, ranked, kUsed, nv);
    std::vector<int8_t> comp(size_t(nv) * K);
    std::vector<float> cs(nv);
    loader::gather_int8_tiled_rows(head.data(), scales.data(), K, N, ids.data(), nv, comp.data(),
                                   cs.data());
    Dv dv(ctx, nv);
    imm.copy(dv.w.ptr(), comp.data(), comp.size());
    imm.copy(dv.s.ptr(), cs.data(), size_t(nv) * 4);
    imm.copy(dv.ids.ptr(), ids.data(), size_t(nv) * 4);

    l0::Module mg(ctx, kernels::path(kernels::gemv_i8w_variant(1, K, nv)));
    l0::Module ma(ctx, kernels::path(kernels::dv_argmax_variant(nv)));
    l0::Kernel kg = mg.kernel("gemv_i8w");
    kg.group_size(kernels::kGemvI8wCols);
    l0::Kernel k1 = ma.kernel("dv_argmax_stage1");
    k1.group_size(256);
    l0::Kernel k2 = ma.kernel("dv_argmax_stage2");
    k2.group_size(256);
    auto bind_stages = [&](const l0::Mem& logits_c) {
      k1.arg_ptr(0, logits_c.ptr());
      k1.arg_ptr(1, dv.ids.ptr());
      k1.arg_ptr(2, part.ptr());
      k1.arg_ptr(3, row.ptr());
      k2.arg_ptr(0, ctrl_mem.ptr());
      k2.arg_ptr(1, part.ptr());
      k2.arg_ptr(2, dv.ids.ptr());
    };

    // 1 + 2: the draft list's tail, as captured.
    kg.arg_ptr(0, dv.w.ptr());
    kg.arg_ptr(1, dv.s.ptr());
    kg.arg_ptr(2, xbuf.ptr());
    kg.arg_ptr(3, dv.logits_c.ptr());
    bind_stages(dv.logits_c);
    l0::CmdList tail = l0::CmdList::regular(ctx);
    tail.launch(kg, nv / kernels::kGemvI8wCols);
    tail.launch(k1, nv / kernels::kDraftVocabChunk);
    tail.launch(k2, 1);
    tail.close();
    imm.fill(row.ptr(), kNegInf, row.size());
    std::memset(ctrl, 0, sizeof *ctrl);
    ctrl->pos = 100;
    ctrl->n_active = 1;
    ctrl->cur_token[0] = 12345;
    q.execute(tail, &fence);
    fence.wait();

    std::vector<float> comp_out(nv), row_out(N);
    imm.copy(comp_out.data(), dv.logits_c.ptr(), size_t(nv) * 4);
    imm.copy(row_out.data(), row.ptr(), size_t(N) * 4);
    uint32_t differ = 0;
    for (uint32_t j = 0; j < nv; ++j)
      if (std::memcmp(&comp_out[j], &full[ids[j]], 4) != 0) ++differ;
    const uint32_t want = argmax_over(full, ids);
    const bool tok_ok = ctrl->out_token[0] == want && ctrl->cur_token[0] == want && ctrl->pos == 101;
    std::vector<uint8_t> in(N, 0);
    for (uint32_t id : ids) in[id] = 1;
    uint32_t row_bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
      const float w = in[i] ? full[i] : -INFINITY;
      if (std::memcmp(&row_out[i], &w, 4) != 0) ++row_bad;
    }
    // The full-vocabulary argmax for reference: where V' agrees with it, acceptance holds.
    std::vector<uint32_t> all_used(kUsed);
    for (uint32_t i = 0; i < kUsed; ++i) all_used[i] = i;
    const uint32_t full_best = argmax_over(full, all_used);
    q.execute(tail, &fence);   // a replay: the same id, pos + 1 again
    fence.wait();
    const bool replay_ok = ctrl->out_token[0] == want && ctrl->pos == 102;
    const double us_c = time_list(q, fence, tail, 1);
    std::printf("|V'| %6u: compact gemv %u/%u columns differ from the full head's (bar 0); argmax id "
                "%u (want %u over V', full-vocab %u) pos %u %s; replay %s; scatter row %u entries "
                "off; tail %.1f us vs full gemv %.1f us\n",
                nv, differ, nv, ctrl->out_token[0], want, full_best, ctrl->pos,
                tok_ok ? "ok" : "FAIL", replay_ok ? "ok" : "FAIL", row_bad, us_c, us_full);
    if (differ || !tok_ok || !replay_ok || row_bad) ++failures;

    // 3. Planted cases on synthetic compact logits (stages only).
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
      l0::Mem syn(ctx, l0::MemKind::Device, size_t(nv) * 4);
      imm.copy(syn.ptr(), lc.data(), size_t(nv) * 4);
      bind_stages(syn);
      l0::CmdList st = l0::CmdList::regular(ctx);
      st.launch(k1, nv / kernels::kDraftVocabChunk);
      st.launch(k2, 1);
      st.close();
      std::memset(ctrl, 0, sizeof *ctrl);
      ctrl->pos = 7;
      ctrl->n_active = 1;
      q.execute(st, &fence);
      fence.wait();
      std::vector<float> f(N, -INFINITY);
      for (uint32_t j = 0; j < nv; ++j) f[ids[j]] = lc[j];
      const uint32_t w = argmax_over(f, ids);
      const bool ok = ctrl->out_token[0] == w && ctrl->cur_token[0] == w && ctrl->pos == 8;
      std::printf("  planted case %d: id %u want %u %s\n", c, ctrl->out_token[0], w, ok ? "ok" : "FAIL");
      if (!ok) ++failures;
      // n_active 2 (not the draft list's 1): stage 2 writes nothing.
      std::memset(ctrl, 0, sizeof *ctrl);
      ctrl->pos = 9;
      ctrl->n_active = 2;
      ctrl->out_token[0] = 88;
      ctrl->cur_token[0] = 77;
      q.execute(st, &fence);
      fence.wait();
      if (!(ctrl->pos == 9 && ctrl->out_token[0] == 88 && ctrl->cur_token[0] == 77)) {
        std::printf("  n_active 2 wrote the control block: FAIL\n");
        ++failures;
      }
    }
  }
  std::puts(failures ? "draft_vocab_kernels_test FAIL" : "draft_vocab_kernels_test OK");
  return failures ? 1 : 0;
}

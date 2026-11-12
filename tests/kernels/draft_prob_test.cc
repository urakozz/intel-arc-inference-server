// Open-work 13: draft_prob.cl - the draft's softmax max and normaliser against a double-precision host reference, on
// the whole row (Qwen3.8's 248320 / 248077) and a TP card's half with a masked tail (124160 / 123917): a peaked row,
// a flat one, a row whose largest logits are masked ids, a draft vocabulary's row (-inf outside a few ids), and the
// two halves merged as runtime::TpEngine::draft merges them. Each probability within 1e-5 of the reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"

namespace {

static_assert(offsetof(runtime::Control, draft_stat) == 4 * kernels::ctrl_index::kDraftStat,
              "CTRL_DSTAT disagrees with runtime::Control");

constexpr uint32_t kWG = 256, kChunk = 1024;

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
};

// (max, normaliser) of the row's first `used` entries, in double.
std::pair<double, double> ref(const std::vector<float>& v, uint32_t used) {
  double m = -std::numeric_limits<double>::infinity();
  for (uint32_t k = 0; k < used; ++k) m = std::max(m, double(v[k]));
  double s = 0;
  for (uint32_t k = 0; k < used; ++k) s += std::exp(double(v[k]) - m);
  return {m, s};
}

// Runs the variant over `row`; returns (max, normaliser) from the Control.
std::pair<float, float> run(Dev& d, uint32_t rows, uint32_t used, const std::vector<float>& row) {
  l0::Mem logits(d.ctx, l0::MemKind::Device, size_t(rows) * 4);
  l0::Mem part(d.ctx, l0::MemKind::Device, size_t(256) * 2 * 4);
  l0::Mem ctrl_mem(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  auto* ctrl = ctrl_mem.as<runtime::Control>();
  *ctrl = runtime::Control{};
  l0::Module mod(d.ctx, kernels::path(kernels::draft_prob_variant(rows, used)));
  l0::Kernel k1 = mod.kernel("draft_prob_stage1");
  k1.group_size(kWG);
  k1.arg_ptr(0, logits.ptr());
  k1.arg_ptr(1, part.ptr());
  l0::Kernel k2 = mod.kernel("draft_prob_stage2");
  k2.group_size(kWG);
  k2.arg_ptr(0, ctrl_mem.ptr());
  k2.arg_ptr(1, part.ptr());
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(k1, (rows + kChunk - 1) / kChunk, 1);
  list.launch(k2, 1, 1);
  list.close();
  d.imm.copy(logits.ptr(), row.data(), size_t(rows) * 4);
  d.q.execute(list, &d.fence);
  d.fence.wait();
  float m, s;
  std::memcpy(&m, &ctrl->draft_stat[0], 4);
  std::memcpy(&s, &ctrl->draft_stat[1], 4);
  return {m, s};
}

std::vector<float> noise(uint32_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> v(n);
  for (float& e : v) e = u(rng);
  return v;
}

void check_row(Dev& d, const char* name, uint32_t rows, uint32_t used, const std::vector<float>& v) {
  const auto [m, s] = run(d, rows, used, v);
  const auto [rm, rs] = ref(v, used);
  std::printf("draft_prob %-34s N%u U%u: max %.6f (ref %.6f), p %.7f (ref %.7f)\n", name, rows, used, m, rm, 1.0 / s,
              1.0 / rs);
  CHECK_NEAR(m, rm, 0.0);
  CHECK_NEAR(1.0 / s, 1.0 / rs, 1e-5);
}

}  // namespace

int main() {
  Dev d;
  using Shape = std::pair<uint32_t, uint32_t>;
  for (const auto& [rows, used] : {Shape{248320, 248077}, Shape{124160, 123917}, Shape{124160, 124160}}) {
    {
      auto v = noise(rows, 1, -8.f, 2.f);
      v[777] = 24.f;   // a sure draft: p near 1
      check_row(d, "peaked", rows, used, v);
    }
    {
      auto v = noise(rows, 2, -0.5f, 0.5f);   // unsure: p near 1 / used
      check_row(d, "flat", rows, used, v);
    }
    {
      auto v = noise(rows, 3, -4.f, 4.f);
      v[used - 1] = 9.f;
      for (uint32_t k = used; k < rows; ++k) v[k] = 50.f;   // masked ids, never counted
      check_row(d, "largest logits masked", rows, used, v);
    }
    {
      std::vector<float> v(rows, -std::numeric_limits<float>::infinity());   // a draft vocabulary's row
      v[5] = 3.f;
      v[6] = 2.f;
      v[70000] = 1.f;
      check_row(d, "-inf outside a few ids", rows, used, v);
    }
  }
  {
    // The TP merge: two halves of one row, each card's (max, normaliser), merged on the host.
    auto whole = noise(248320, 4, -6.f, 3.f);
    whole[200000] = 12.f;
    whole[3] = 11.f;
    const std::vector<float> a(whole.begin(), whole.begin() + 124160), b(whole.begin() + 124160, whole.end());
    const auto [ma, sa] = run(d, 124160, 124160, a);
    const auto [mb, sb] = run(d, 124160, 123917, b);
    const float m = std::max(ma, mb);
    const double p = 1.0 / (sa * std::exp(ma - m) + sb * std::exp(mb - m));
    const auto [rm, rs] = ref(whole, 248077);
    std::printf("draft_prob TP halves merged: p %.7f (ref %.7f), max %.6f (ref %.6f)\n", p, 1.0 / rs, m, rm);
    CHECK_NEAR(p, 1.0 / rs, 1e-5);
  }
  std::puts("draft_prob_test OK");
  return 0;
}

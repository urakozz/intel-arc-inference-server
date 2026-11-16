// Plan 21v2b Task 2: Qwen3.8-Flash-Next on `--parallel tp` against `--parallel pp` (box only; label
// checkpoint;qwen4exp;tp). Both engines are loaded side by side at --layers N, every expert resident (pp's halves of
// the layers and tp's halves of every layer fit two cards at N = 18):
//   qwen4exp_tp_test <checkpoint> <ids> [layers 18] [steps 10000] [bench_pairs 4]
//   (a) the first 32 ids forced one a replay through both: each row's argmax equal to pp's, or different only inside
//       the near-tie window (pp's logit of tp's id within 2 bf16 ulps of pp's top); after every row both cards' H and
//       route rows bitwise equal
//   (b) `steps` greedy steps from position 0 with no crossing failure (submit throws on a timeout or stale word)
//   (c) decode at the ids' length (4096 for q4exp_4k.ids): both engines fed the prompt, then after a warm-up
//       `bench_pairs` interleaved pairs of 64 generated tokens each; the median t/s of each and tp / pp, and one card's
//       launch count against the tp list's
// Exit 77 (SKIP) with one GPU or without the checkpoint.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp/qwen4exp_tp_engine.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;

uint32_t top(const std::vector<float>& l, uint32_t used) {
  uint32_t b = 0;
  for (uint32_t i = 1; i < used; ++i)
    if (l[i] > l[b]) b = i;
  return b;
}
float bf16_ulp(float v) {
  int e = 0;
  std::frexp(std::fabs(v), &e);
  return std::ldexp(1.0f, e - 8);
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: qwen4exp_tp_test <checkpoint> <ids> [layers] [steps] [bench_pairs]\n");
    return 2;
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  const uint32_t layers = argc > 3 ? uint32_t(std::atoi(argv[3])) : 18;
  const uint32_t steps = argc > 4 ? uint32_t(std::atoi(argv[4])) : 10000;
  const uint32_t pairs = argc > 5 ? uint32_t(std::atoi(argv[5])) : 4;
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::open(rig, 1);
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: one GPU\n");
    return 77;
  }
  qwen4exp_rig::open(rig, 2);
  const uint32_t max_len = 16384;
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, layers);
  const std::vector<uint32_t> ids = qwen4exp_rig::read_ids(argv[2], d.vocab);
  CHECK(ids.size() >= 32 && ids.size() + 64 * (pairs + 1) < max_len);
  // pp: the planner's split for N layers
  rq::Qwen4ExpEngine pp(rig.ctx,
                        loader::load_qwen4exp(rig.ctx, snap, max_len,
                                              model::Q4Placement::two(d, rq::pp_split(d, max_len, true, false).split),
                                              loader::LmHeadForm::Int8, layers),
                        max_len);
  rq::Qwen4ExpTpEngine tp(*rig.c0, *rig.c1,
                          loader::load_qwen4exp(rig.ctx, snap, max_len, model::Q4Placement::tp2(d, model::Q4TpEnds::Split),
                                                loader::LmHeadForm::Int8, layers),
                          max_len);
  std::printf("tp: %u layers, launches pp %zu (both devices), tp %zu a card (tp_device_launches %zu)\n", layers,
              pp.launches(), tp.launches(0), rq::tp_device_launches(d));
  CHECK_EQ(tp.launches(0), rq::tp_device_launches(d));
  CHECK_EQ(tp.launches(1), rq::tp_device_launches(d));

  // (a) the forced rows
  uint32_t same = 0, ties = 0, outside = 0;
  double worst_cos = 1.0;
  for (uint32_t i = 0; i < 32; ++i) {
    pp.ingest({ids[i]});
    tp.ingest({ids[i]});
    const std::vector<float> lp = pp.read_logits(), lt = tp.read_logits();
    const uint32_t a = pp.pending(), b = tp.pending();
    CHECK_EQ(a, top(lp, d.vocab_used));
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (uint32_t v = 0; v < d.vocab_used; ++v) {
      dot += double(lp[v]) * lt[v];
      na += double(lp[v]) * lp[v];
      nb += double(lt[v]) * lt[v];
      mx = std::max(mx, double(std::fabs(lp[v] - lt[v])));
    }
    const double cs = dot / std::sqrt(na * nb);
    worst_cos = std::min(worst_cos, cs);
    const float gap = lp[a] - lp[b];
    std::printf("tp: row %2u: pp %6u tp %6u, logits cos %.6f, max |diff| %.4f%s\n", i, a, b, cs, mx,
                a == b ? "" : (", pp's gap to tp's id " + std::to_string(gap) + " (2 ulps " +
                               std::to_string(2 * bf16_ulp(lp[a])) + ")").c_str());
    if (a == b)
      ++same;
    else if (gap <= 2 * bf16_ulp(lp[a]))
      ++ties;
    else
      ++outside;
    CHECK(tp.read_H(0) == tp.read_H(1));
    CHECK(tp.read_routes(0) == tp.read_routes(1));
    // keep both on pp's id so the rows stay comparable
    pp.set_token(a);
    tp.set_token(a);
  }
  std::printf("tp: (a) 32 forced rows: %u argmax equal to pp's, %u inside the near-tie window, %u outside; logits "
              "cos >= %.6f; both cards' H and routes bitwise every row\n", same, ties, outside, worst_cos);

  // (b) the long run
  tp.reset();
  tp.set_token(ids[0]);
  tp.generate(steps);
  for (uint32_t r = 0; r < 2; ++r) CHECK_EQ(tp.state_words(r)[runtime::tp::kStatus], 0u);
  std::printf("tp: (b) %u greedy steps from position 0: 0 timeout / stale words (%.2f t/s, the context growing)\n",
              steps, tp.last_tok_per_s());

  // (c) decode at the prompt's length, interleaved pairs after a warm-up
  pp.reset();
  tp.reset();
  pp.ingest(ids);
  tp.ingest(ids);
  pp.generate(64);
  tp.generate(64);
  std::vector<double> tpp, ttp;
  for (uint32_t p = 0; p < pairs; ++p) {
    pp.generate(64);
    tpp.push_back(pp.last_tok_per_s());
    tp.generate(64);
    ttp.push_back(tp.last_tok_per_s());
    std::printf("tp: (c) pair %u at pos ~%u: pp %.2f t/s, tp %.2f t/s (%.3fx)\n", p, tp.pos(), tpp.back(), ttp.back(),
                ttp.back() / tpp.back());
  }
  const double mp = median(tpp), mt = median(ttp);
  std::printf("tp: (c) decode at %zu ids, %u layers, every expert resident: pp %.2f t/s (%.3f ms), tp %.2f t/s "
              "(%.3f ms): %.3fx (the stop rule: 1.15x)\n", ids.size(), layers, mp, 1000 / mp, mt, 1000 / mt, mt / mp);
  CHECK_EQ(outside, 0u);   // (a)'s gate, after (b) and (c) have run
  std::printf("qwen4exp_tp_test OK\n");
  return 0;
}

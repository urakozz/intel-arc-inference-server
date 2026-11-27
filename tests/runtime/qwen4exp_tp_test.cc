// Plan 21-QFN-v2b Task 2: Qwen3.8-Flash-Next on `--parallel tp` against `--parallel pp` (GPU only; label
// checkpoint;qwen4exp;tp), at --layers N with every expert resident, or the whole model with each engine's tier. One engine a process (each pins the 51.8 GB PLE
// table; two do not fit beside a CPU job), so a comparison is two runs and a ratio is interleaved runs:
//   qwen4exp_tp_test <checkpoint> <ids> pp|tp <rows file> [layers 18] [steps 10000] [bench 4] [profile replays]
//   (pp-kv8 / tp-kv8, plan 21-QFN-v3a: the same with the int8 KV cache on both sides)
//   (a) the first 32 ids forced one a replay: pp writes each row's logits to <rows file>; tp reads them and holds each
//       row's argmax equal to pp's, or different only inside the near-tie window (pp's logit of tp's id within 2 bf16
//       ulps of pp's top), printing each row's logits cosine and max difference and counting the rows whose logits are pp's
//       bitwise (the crossings keep one card's summation order); after every row both cards' H and route
//       rows bitwise equal (tp)
//   (b) tp: `steps` greedy steps from position 0 with no crossing failure (submit throws on a timeout or stale word)
//   (c) decode at the ids' length (4096 for q4exp_4k.ids): the prompt fed, a warm-up of 64, then `bench` runs of 64
//       generated tokens; each run's t/s and the median
//   (d) tp, with [profile replays]: card 0's list's device time a launch, summed by kernel
//   with force:<ids> in that place instead: only the f32 judge's rows (below), no (a) - (d); forcepf:<ids> the same with
//   the 4500 ids prefilled
// Exit 77 (SKIP) with one GPU or without the checkpoint.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
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
  if (argc < 5) {
    std::fprintf(stderr, "usage: qwen4exp_tp_test <checkpoint> <ids> pp|tp <rows file> [layers] [steps] [bench]\n");
    return 2;
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  const uint32_t layers = argc > 5 ? uint32_t(std::atoi(argv[5])) : 18;
  const uint32_t steps = argc > 6 ? uint32_t(std::atoi(argv[6])) : 10000;
  const uint32_t pairs = argc > 7 ? uint32_t(std::atoi(argv[7])) : 4;
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::open(rig, 1);
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: one GPU\n");
    return 77;
  }
  qwen4exp_rig::open(rig, 2);
  const uint32_t max_len = 16384;
  std::string mode = argv[3];
  const std::string rows_path = argv[4];
  const bool kv8 = mode.size() > 4 && mode.compare(mode.size() - 4, 4, "-kv8") == 0;   // plan 21-QFN-v3a: the int8 KV cache
  if (kv8) mode.resize(mode.size() - 4);
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, layers, kv8);
  const std::vector<uint32_t> ids = qwen4exp_rig::read_ids(argv[2], d.vocab);
  CHECK(ids.size() >= 32 && ids.size() + 64 * (pairs + 1) < max_len);
  CHECK(mode == "pp" || mode == "tp");
  const size_t V = d.vocab;
  std::unique_ptr<rq::Qwen4ExpEngine> pp;
  std::unique_ptr<rq::Qwen4ExpTpEngine> tp;
  // The whole model: each engine's expert tier as the CLI plans it (the routing profile's most-routed experts, the
  // budget with the prefill scratch planned, 32.53 GB cards less the 1.5 GB reserve); --layers N: every expert resident.
  const bool whole = layers == loader::qwen4exp_checkpoint_desc(snap).layers;
  const std::array<size_t, runtime::kPpDevices> caps = {rig.c0->memory_bytes(), rig.c1->memory_bytes()};
  const size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const std::vector<uint32_t> prof = whole ? loader::load_q4_profile(snap, d) : std::vector<uint32_t>{};
  const std::vector<uint32_t>* pc = prof.empty() ? nullptr : &prof;
  if (mode == "pp") {
    const model::Q4Placement pl = model::Q4Placement::two(d, rq::pp_split(d, max_len, true, false, true).split);
    std::optional<rq::ExpertTier> t;
    if (whole) t = rq::fill_tier(d, rq::expert_budget(d, pl, max_len, true, false, true, caps, reserve), pc);
    pp = std::make_unique<rq::Qwen4ExpEngine>(
        rig.ctx, loader::load_qwen4exp(rig.ctx, snap, max_len, pl, loader::LmHeadForm::Int8, layers, false, t ? &*t : nullptr,
                                       nullptr, rq::kPfC, kv8),
        max_len);
  } else {
    const model::Q4Placement pl = model::Q4Placement::tp2(d, model::Q4TpEnds::Split);
    std::optional<rq::ExpertTier> t0, t1;
    if (whole) {
      const rq::ExpertBudget b = rq::tp_expert_budget(d, pl, max_len, true, false, true, caps, reserve);
      t0 = rq::tp_fill_tier(d, b, 0, pc);
      t1 = rq::tp_fill_tier(d, b, 1, pc);
      std::printf("tp: the whole model: f %.4f / %.4f a card (%s)\n", rq::tp_card_f(b, 0), rq::tp_card_f(b, 1),
                  t0->source.c_str());
    }
    tp = std::make_unique<rq::Qwen4ExpTpEngine>(
        *rig.c0, *rig.c1,
        loader::load_qwen4exp(rig.ctx, snap, max_len, pl, loader::LmHeadForm::Int8, layers, false, t0 ? &*t0 : nullptr,
                              t1 ? &*t1 : nullptr, rq::kPfC, kv8),
        max_len);
  }
  const auto ingest = [&](uint32_t id) { pp ? pp->ingest({id}) : tp->ingest({id}); };
  const auto pending = [&] { return pp ? pp->pending() : tp->pending(); };
  const auto set_token = [&](uint32_t id) { pp ? pp->set_token(id) : tp->set_token(id); };
  const auto logits = [&] { return pp ? pp->read_logits() : tp->read_logits(); };
  if (tp) {
    std::printf("tp: %u layers, %zu launches a card (tp_device_launches %zu)\n", layers, tp->launches(0),
                rq::tp_device_launches(d));
    CHECK_EQ(tp->launches(0), rq::tp_device_launches(d));
    CHECK_EQ(tp->launches(1), rq::tp_device_launches(d));
  } else {
    std::printf("tp: pp at %u layers, %zu launches (both devices)\n", layers, pp->launches());
  }

  // force:<ids> (the f32 judge's rows, plan 21-QFN-v3a): <ids> cycled to 4500 positions fed one a replay (the f32
  // reference's prompt, qwen4exp_oracle.sh intel-f32), then the reference's 32 forced ids; each of the 33 decision rows'
  // logits -> <rows file> (pp) / <rows file>.tp (tp), judged by serve_judge.py rows against the reference's `logits`.
  // forcepf:<ids>: the 4500 ids in one prefill (the int8 cache's prompt-pass writer and reader), then the same rows.
  const std::string force_arg = argc > 8 ? argv[8] : "";
  const bool force_pf = force_arg.rfind("forcepf:", 0) == 0;
  if (force_pf || force_arg.rfind("force:", 0) == 0) {
    const std::vector<uint32_t> forced = qwen4exp_rig::read_ids(force_arg.substr(force_arg.find(':') + 1), d.vocab);
    CHECK_EQ(forced.size(), size_t(32));
    std::vector<float> rows;
    std::vector<uint32_t> cycled(4500);
    for (uint32_t p = 0; p < 4500; ++p) cycled[p] = ids[p % ids.size()];
    if (force_pf) pp ? pp->prefill(cycled) : tp->prefill(cycled);
    else for (const uint32_t id : cycled) ingest(id);
    for (uint32_t j = 0; j <= forced.size(); ++j) {
      const std::vector<float> lt = logits();
      rows.insert(rows.end(), lt.begin(), lt.end());
      if (j < forced.size()) ingest(forced[j]);
    }
    const std::string out = pp ? rows_path : rows_path + ".tp";
    FILE* f = std::fopen(out.c_str(), "wb");
    CHECK(f && std::fwrite(rows.data(), 4, rows.size(), f) == rows.size());
    std::fclose(f);
    std::printf("tp: %s's 33 decision rows (4500 cycled ids%s + 32 forced) -> %s\n", mode.c_str(),
                force_pf ? " prefilled" : "", out.c_str());
    return 0;
  }
  // (a) the forced rows; every row continues on pp's id (tp reads it from the rows file's argmax)
  std::vector<float> ref;
  if (tp) {
    ref.resize(32 * V);
    FILE* f = std::fopen(rows_path.c_str(), "rb");
    CHECK(f && std::fread(ref.data(), 4, ref.size(), f) == ref.size());
    std::fclose(f);
  }
  std::vector<float> mine;
  uint32_t same = 0, ties = 0, outside = 0, bitwise = 0;
  double worst_cos = 1.0;
  for (uint32_t i = 0; i < 32; ++i) {
    ingest(ids[i]);
    const std::vector<float> lt = logits();
    if (pp) {
      CHECK_EQ(pending(), top(lt, d.vocab_used));
      mine.insert(mine.end(), lt.begin(), lt.end());
      continue;
    }
    const std::vector<float> lp(ref.begin() + i * V, ref.begin() + (i + 1) * V);
    const uint32_t a = top(lp, d.vocab_used), b = pending();
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (uint32_t v = 0; v < d.vocab_used; ++v) {
      dot += double(lp[v]) * lt[v];
      na += double(lp[v]) * lp[v];
      nb += double(lt[v]) * lt[v];
      mx = std::max(mx, double(std::fabs(lp[v] - lt[v])));
    }
    const double cs = dot / std::sqrt(na * nb);
    bitwise += std::memcmp(lp.data(), lt.data(), V * 4) == 0;
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
    CHECK(tp->read_H(0) == tp->read_H(1));
    CHECK(tp->read_routes(0) == tp->read_routes(1));
    set_token(a);
  }
  if (pp) {
    FILE* f = std::fopen(rows_path.c_str(), "wb");
    CHECK(f && std::fwrite(mine.data(), 4, mine.size(), f) == mine.size());
    std::fclose(f);
    std::printf("tp: (a) pp's 32 forced rows -> %s\n", rows_path.c_str());
  } else {
    std::printf("tp: (a) 32 forced rows: %u argmax equal to pp's, %u inside the near-tie window, %u outside; %u rows' "
                "logits bitwise pp's, cos >= %.6f; both cards' H and routes bitwise every row\n", same, ties, outside,
                bitwise, worst_cos);
    // (b) the long run
    tp->reset();
    tp->set_token(ids[0]);
    tp->generate(steps);
    for (uint32_t r = 0; r < 2; ++r) CHECK_EQ(tp->state_words(r)[runtime::tp::kStatus], 0u);
    std::printf("tp: (b) %u greedy steps from position 0: 0 timeout / stale words (%.2f t/s, the context growing)\n",
                steps, tp->last_tok_per_s());
  }

  // (c) decode at the prompt's length
  pp ? pp->reset() : tp->reset();
  for (uint32_t id : ids) ingest(id);
  const auto gen = [&] {
    if (pp) {
      pp->generate(64);
      return pp->last_tok_per_s();
    }
    tp->generate(64);
    return tp->last_tok_per_s();
  };
  gen();
  std::vector<double> ts;
  for (uint32_t p = 0; p < pairs; ++p) {
    ts.push_back(gen());
    std::printf("tp: (c) %s run %u at pos ~%u: %.2f t/s\n", mode.c_str(), p, pp ? pp->pos() : tp->pos(), ts.back());
  }
  const double m = median(ts);
  std::printf("tp: (c) %s decode at %zu ids, %u layers, every expert resident: %.2f t/s (%.3f ms a token)\n",
              mode.c_str(), ids.size(), layers, m, 1000 / m);
  // (d) optional: card 0's list profiled over `prof` replays (a timestamp event a launch), summed by kernel entry
  if (tp && argc > 8) {
    const uint32_t R = uint32_t(std::atoi(argv[8]));
    const std::vector<double> us = tp->profile_decode(0, R);
    const std::vector<std::string>& lab = tp->step(0).labels;
    std::vector<std::pair<std::string, std::pair<double, uint32_t>>> by;
    double total = 0;
    for (size_t i = 0; i < us.size(); ++i) {
      const std::string& l = lab[i];
      const size_t a = l.find(' '), b = l.find(' ', a + 1);
      const std::string key = l.substr(a + 1, b - a - 1) + " " + l.substr(b + 1);
      auto it = std::find_if(by.begin(), by.end(), [&](const auto& e) { return e.first == key; });
      if (it == by.end()) by.push_back({key, {0.0, 0u}}), it = by.end() - 1;
      it->second.first += us[i];
      ++it->second.second;
      total += us[i];
    }
    std::sort(by.begin(), by.end(), [](const auto& x, const auto& y) { return x.second.first > y.second.first; });
    std::printf("tp: (d) card 0's list, %zu launches, sum of device times %.1f us (%u replays):\n", us.size(), total, R);
    for (const auto& e : by)
      std::printf("tp: (d)   %8.1f us  %4u x  %s\n", e.second.first, e.second.second, e.first.c_str());
  }
  CHECK_EQ(outside, 0u);   // (a)'s gate, after (b) and (c) have run
  std::printf("qwen4exp_tp_test OK\n");
  return 0;
}

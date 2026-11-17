// Plan 21v2b Task 4: Qwen3.8-Flash-Next's prefill on `--parallel tp` against `--parallel pp` (box only; label
// checkpoint;qwen4exp;tp). One engine a process (each pins the 51.8 GB PLE table), so a comparison is two runs:
//   qwen4exp_tp_prefill_test <checkpoint> <ids> pp|tp|ppdump|tpdump <rows file> [layers 18] [n 4500] [bench 0]
// The prompt is <ids> cycled to n positions (4500: three chunks - all dense, straddling the dense range, a 404-row
// tail); --layers N with every expert resident, the whole model with each engine's tier as the CLI plans it.
//   pp  one prefill; writes the last logits, the first id, the last chunk's route rows and the control (its prefill's
//       last logits' cosine against its decode fill) to <rows file>
//   tp  (1) the walk: each chunk's launches are 2 x prefill_device_launches at the card's shapes, the head's 2 x
//           kPrefillHeadLaunches; both cards' Controls agree after the merge
//       (2) determinism: two prefills (reset between) leave the same last logits, first id and both cards' H, bitwise
//       (3) against pp's prefill: the last logits' cosine, the first id equal or inside pp's two-bf16-ulp window
//           (gated); the last chunk's routes - (row, layer) pairs with pp's expert set, the rest split by pp's p10 - p11
//           gap (near-ties within kTieTol) - printed
//       (4) against tp's own decode fill (ingest, one id a replay): the last logits' cosine within 2x the distance of
//           pp's own prefill from its decode fill (the control pp writes; qwen4exp_prefill_test's rule for a truncated
//           model), the first id; 32 greedy ids from each - printed
//   ppdump  pp's prefill, then its last logits, first id, the last chunk's route rows and its last QSA layer's
//       block scores to <rows file> (no decode fill)
//   tpdump  tp's (1) and (2), then its last logits, first id, H (card 0's) and the last chunk's route rows to <rows
//       file> - two builds' files `cmp` equal when a change keeps tp's prefill bitwise
//   bench > 0: only the timing - a 2048-id warm-up prefill, then `bench` prefills of each length in n (a comma list,
//   one load), each one's t/s; under B70_PREFILL_PROFILE=1 each one's phases (runtime/prefill/profile.h)
// Exit 77 (SKIP) with one GPU or without the checkpoint.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/qwen4exp_kernels.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp/qwen4exp_tp_engine.h"
#include "runtime/prefill/profile.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;
namespace kq = kernels::qwen4exp;
using Clock = std::chrono::steady_clock;
constexpr float kTieTol = 1e-3f;   // qwen4exp_prefill_test's near-tie window on the router's p10 - p11

float bf16_ulp(float v) {
  int e = 0;
  std::frexp(std::fabs(v), &e);
  return std::ldexp(1.0f, e - 8);
}
double cosine(const std::vector<float>& a, const std::vector<float>& b, uint32_t used) {
  double dot = 0, na = 0, nb = 0;
  for (uint32_t v = 0; v < used; ++v) {
    dot += double(a[v]) * b[v];
    na += double(a[v]) * a[v];
    nb += double(b[v]) * b[v];
  }
  return dot / std::sqrt(na * nb);
}
float bits(uint32_t w) {
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: qwen4exp_tp_prefill_test <checkpoint> <ids> pp|tp|ppdump|tpdump <rows file> [layers] [n] [bench]\n");
    return 2;
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  const uint32_t layers = argc > 5 ? uint32_t(std::atoi(argv[5])) : 18;
  // n, or (bench) a comma list of lengths timed in one load - the prompt is the longest
  std::vector<uint32_t> lens;
  for (const char* p = argc > 6 ? argv[6] : "4500"; *p;) {
    lens.push_back(uint32_t(std::strtoul(p, const_cast<char**>(&p), 10)));
    if (*p == ',') ++p;
  }
  const uint32_t n = *std::max_element(lens.begin(), lens.end());
  const uint32_t bench = argc > 7 ? uint32_t(std::atoi(argv[7])) : 0;
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::open(rig, 1);
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: one GPU\n");
    return 77;
  }
  qwen4exp_rig::open(rig, 2);
  const uint32_t max_len = std::max<uint32_t>(16384, (n + 64 + 1023) / 1024 * 1024);
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, layers);
  const std::vector<uint32_t> src = qwen4exp_rig::read_ids(argv[2], d.vocab);
  CHECK(!src.empty() && n >= 1);
  std::vector<uint32_t> prompt(n);
  for (uint32_t i = 0; i < n; ++i) prompt[i] = src[i % src.size()];
  const bool dump = std::string(argv[3]) == "tpdump" || std::string(argv[3]) == "ppdump";
  const std::string mode = dump ? std::string(argv[3]).substr(0, 2) : argv[3], rows_path = argv[4];
  CHECK(mode == "pp" || mode == "tp");
  const uint32_t V = d.vocab;
  const bool whole = layers == loader::qwen4exp_checkpoint_desc(snap).layers;
  const std::array<size_t, runtime::kPpDevices> caps = {rig.c0->memory_bytes(), rig.c1->memory_bytes()};
  const size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const std::vector<uint32_t> prof = whole ? loader::load_q4_profile(snap, d) : std::vector<uint32_t>{};
  const std::vector<uint32_t>* pc = prof.empty() ? nullptr : &prof;
  std::unique_ptr<rq::Qwen4ExpEngine> pp;
  std::unique_ptr<rq::Qwen4ExpTpEngine> tp;
  if (mode == "pp") {
    const model::Q4Placement pl = model::Q4Placement::two(d, rq::pp_split(d, max_len, true, false, true).split);
    std::optional<rq::ExpertTier> t;
    if (whole) t = rq::fill_tier(d, rq::expert_budget(d, pl, max_len, true, false, true, caps, reserve), pc);
    pp = std::make_unique<rq::Qwen4ExpEngine>(
        rig.ctx, loader::load_qwen4exp(rig.ctx, snap, max_len, pl, loader::LmHeadForm::Int8, layers, false, t ? &*t : nullptr),
        max_len);
    pp->prepare_prefill();
  } else {
    const model::Q4Placement pl = model::Q4Placement::tp2(d, model::Q4TpEnds::Split);
    std::optional<rq::ExpertTier> t0, t1;
    if (whole) {
      const rq::ExpertBudget b = rq::tp_expert_budget(d, pl, max_len, true, false, true, caps, reserve);
      t0 = rq::tp_fill_tier(d, b, 0, pc);
      t1 = rq::tp_fill_tier(d, b, 1, pc);
      std::printf("tp-prefill: the whole model: f %.4f / %.4f a card\n", rq::tp_card_f(b, 0), rq::tp_card_f(b, 1));
    }
    tp = std::make_unique<rq::Qwen4ExpTpEngine>(
        *rig.c0, *rig.c1,
        loader::load_qwen4exp(rig.ctx, snap, max_len, pl, loader::LmHeadForm::Int8, layers, false, t0 ? &*t0 : nullptr,
                              t1 ? &*t1 : nullptr),
        max_len);
    tp->prepare_prefill();
  }
  const auto reset = [&] { pp ? pp->reset() : tp->reset(); };
  const auto prefill = [&](const std::vector<uint32_t>& ids) { pp ? pp->prefill(ids) : tp->prefill(ids); };
  const auto logits = [&] { return pp ? pp->read_logits() : tp->read_logits(); };
  const auto pending = [&] { return pp ? pp->pending() : tp->pending(); };
  std::printf("tp-prefill: %s, %u layers, max_len %u, prompt %u ids\n", mode.c_str(), layers, max_len, n);

  if (bench > 0) {
    reset();
    prefill(std::vector<uint32_t>(prompt.begin(), prompt.begin() + std::min<uint32_t>(n, rq::kPfC)));   // warm-up
    for (uint32_t len : lens)
      for (uint32_t i = 0; i < bench; ++i) {
        reset();
        if (runtime::prefill::profile_enabled()) runtime::prefill::profile_reset();
        const Clock::time_point t0 = Clock::now();
        prefill(std::vector<uint32_t>(prompt.begin(), prompt.begin() + len));
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (runtime::prefill::profile_enabled())   // B70_PREFILL_PROFILE=1: the phases, both cards' lists summed
          runtime::prefill::profile_report((mode + " " + std::to_string(len)).c_str(), ms);
        std::printf("tp-prefill: %s bench %u: %u ids in %.1f ms = %.1f t/s\n", mode.c_str(), i, len, ms,
                    len * 1000.0 / ms);
      }
    std::printf("qwen4exp_tp_prefill_test OK\n");
    return 0;
  }

  const uint32_t lastC = n - ((n - 1) / rq::kPfC) * rq::kPfC;
  if (pp) {
    reset();
    prefill(prompt);
    const std::vector<float> lg = logits();
    const std::vector<uint32_t> rt = pp->read_prefill_routes();
    const uint32_t first = pending();
    if (dump) {
      std::vector<float> sc = pp->read_prefill_last_qsa_scores();
      const uint32_t stride = max_len / 4, p0 = n - lastC;
      for (uint32_t m = 0; m < rq::kPfC; ++m)   // only the rows' complete blocks are scores (the rest is stale)
        std::fill(sc.begin() + size_t(m) * stride + (m < lastC ? (p0 + m + 1) / 4 : 0),
                  sc.begin() + size_t(m + 1) * stride, 0.0f);
      FILE* f = std::fopen(rows_path.c_str(), "wb");
      CHECK(f && std::fwrite(lg.data(), 4, lg.size(), f) == lg.size() && std::fwrite(&first, 4, 1, f) == 1 &&
            std::fwrite(rt.data(), 4, rt.size(), f) == rt.size() && std::fwrite(sc.data(), 4, sc.size(), f) == sc.size());
      std::fclose(f);
      std::printf("tp-prefill: pp's last logits, first id, routes and last QSA layer's scores -> %s\n", rows_path.c_str());
      std::printf("qwen4exp_tp_prefill_test OK\n");
      return 0;
    }
    reset();   // the control: pp's own prefill against its decode fill
    pp->ingest(prompt);
    const float ctl = float(cosine(lg, logits(), d.vocab_used));
    FILE* f = std::fopen(rows_path.c_str(), "wb");
    CHECK(f && std::fwrite(lg.data(), 4, lg.size(), f) == lg.size() && std::fwrite(&first, 4, 1, f) == 1 &&
          std::fwrite(rt.data(), 4, rt.size(), f) == rt.size() && std::fwrite(&ctl, 4, 1, f) == 1);
    std::fclose(f);
    std::printf("tp-prefill: pp's prefill: first id %u, last logits cosine against pp's decode fill %.7f (the control); "
                "the last logits and the last chunk's routes -> %s\n", first, double(ctl), rows_path.c_str());
    std::printf("qwen4exp_tp_prefill_test OK\n");
    return 0;
  }

  // (1) the walk
  const size_t l0 = tp->prefill_launches();
  reset();
  prefill(prompt);
  size_t want = 2 * rq::kPrefillHeadLaunches;
  const model::Qwen4ExpDesc c0 = model::q4_tp_card(d, 0);
  for (uint32_t p = 0; p < n; p += rq::kPfC)
    want += 2 * (rq::prefill_device_launches(c0, model::Q4Placement::one(c0), 0, p, std::min(rq::kPfC, n - p)) +
                 rq::tp_split_launches(c0, p));
  CHECK_EQ(tp->prefill_launches() - l0, want);
  CHECK_EQ(tp->pos(), n);
  const std::vector<float> a = logits();
  const uint32_t a_first = pending();
  const std::vector<uint16_t> h0 = tp->read_H(0);
  CHECK(h0 == tp->read_H(1));
  std::printf("tp-prefill: (1) walk: %zu launches for %u ids on both cards; pos %u; both cards' H bitwise\n", want, n,
              tp->pos());
  // (2) determinism
  reset();
  prefill(prompt);
  const bool same = logits() == a && pending() == a_first && tp->read_H(0) == h0 && tp->read_H(1) == h0;
  std::printf("tp-prefill: (2) two prefills: last logits, first id, H %s\n", same ? "bitwise" : "DIFFER");
  CHECK(same);
  const std::vector<uint32_t> tr = tp->read_prefill_routes();
  if (dump) {
    FILE* f = std::fopen(rows_path.c_str(), "wb");
    CHECK(f && std::fwrite(a.data(), 4, a.size(), f) == a.size() && std::fwrite(&a_first, 4, 1, f) == 1 &&
          std::fwrite(h0.data(), 2, h0.size(), f) == h0.size() && std::fwrite(tr.data(), 4, tr.size(), f) == tr.size());
    std::fclose(f);
    std::printf("tp-prefill: tp's last logits, first id, H and the last chunk's routes -> %s\n", rows_path.c_str());
    std::printf("qwen4exp_tp_prefill_test OK\n");
    return 0;
  }
  // (3) against pp's prefill
  std::vector<float> pl(V);
  uint32_t p_first = 0;
  float ctl = 0;
  std::vector<uint32_t> prt(size_t(d.layers) * rq::kPfC * rq::kRouteWords);
  {
    FILE* f = std::fopen(rows_path.c_str(), "rb");
    CHECK(f && std::fread(pl.data(), 4, V, f) == V && std::fread(&p_first, 4, 1, f) == 1 &&
          std::fread(prt.data(), 4, prt.size(), f) == prt.size() && std::fread(&ctl, 4, 1, f) == 1);
    std::fclose(f);
  }
  const double cs = cosine(pl, a, d.vocab_used);
  const float gap = pl[p_first] - pl[a_first];
  const bool tok_ok = a_first == p_first || gap <= 2 * bf16_ulp(pl[p_first]);
  uint64_t rows = 0, exact = 0, near = 0, beyond = 0;
  for (uint32_t l = 0; l < d.layers; ++l)
    for (uint32_t r = 0; r < lastC; ++r) {
      const uint32_t* x = prt.data() + rq::pf_route_at(l) / 4 + size_t(r) * rq::kRouteWords;
      const uint32_t* y = tr.data() + rq::pf_route_at(l) / 4 + size_t(r) * rq::kRouteWords;
      std::vector<uint32_t> g(x, x + kq::kTopK), w(y, y + kq::kTopK);
      std::sort(g.begin(), g.end());
      std::sort(w.begin(), w.end());
      ++rows;
      if (g == w) {
        ++exact;
        continue;
      }
      const float pg = bits(x[kq::route::kP10]) - bits(x[kq::route::kP11]);
      (pg <= kTieTol ? near : beyond) += 1;
    }
  std::printf("tp-prefill: (3) against pp's prefill: last logits cosine %.7f, first id tp %u / pp %u%s; the last chunk's "
              "routes: %llu (row, layer) - %llu pp's set, %llu near-ties, %llu beyond\n", cs, a_first, p_first,
              a_first == p_first ? "" : (" (pp's gap " + std::to_string(gap) + ")").c_str(), (unsigned long long)rows,
              (unsigned long long)exact, (unsigned long long)near, (unsigned long long)beyond);
  // (4) against tp's decode fill
  std::vector<uint32_t> g_pf;
  {
    tp->generate(32, [&](uint32_t id) { g_pf.push_back(id); });
  }
  reset();
  tp->ingest(prompt);
  const std::vector<float> dl = logits();
  const uint32_t d_first = pending();
  std::vector<uint32_t> g_dec;
  tp->generate(32, [&](uint32_t id) { g_dec.push_back(id); });
  uint32_t agree = 0;
  while (agree < 32 && g_pf[agree] == g_dec[agree]) ++agree;
  const double cd = cosine(dl, a, d.vocab_used);
  std::printf("tp-prefill: (4) against tp's decode fill: last logits cosine %.7f (pp's control %.7f; bar: within 2x its "
              "distance), first id prefill %u / decode %u; 32 greedy ids agree for the first %u\n", cd, double(ctl),
              a_first, d_first, agree);
  CHECK(tok_ok);
  CHECK(1 - cd <= 2 * (1 - double(ctl)) + 1e-9);
  std::printf("qwen4exp_tp_prefill_test OK\n");
  return 0;
}

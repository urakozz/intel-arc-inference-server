// probe_slots_speed - spec 13c (plan 13c Task 5): the decode speed of runtime::Engine's sequence slots, through the
// engine API (host writes, submit, fence - what the server pays per step, without HTTP and the scheduler).
//
//   probe_slots_speed <snapshot> [--kv bf16|int8] [--slots L0,L1,..] [--depth 4096] [--steps 48] [--rounds 3]
//
// One engine (Qwen3.8, its MTP head, the int8 lm_head - the served form) with the given slots (default 4 x 32768),
// every slot prefilled with `depth` ids of tests/golden/prompts/long32k.ids (each from its own offset). Then, per
// round, every arm in a rotated order, `steps` steps each, greedy - real drafts, real acceptance:
//
//   1 plain        slot 0 alone, its own lists: verify(0) + commit (today's --spec off step with the head loaded)
//   1 mtp K=k      slot 0 alone, its own lists: draft(k) + verify(k) + the longest agreeing prefix (today's --spec mtp)
//   n plain        n slots, one verify_rows step of n rows (k = 0 each) + n commits
//   n K=a,b,..     n slots at those K: the drafts of every slot with K > 0 - ONE draft_rows step per draft index
//                  (build_draft_rows: the head and lm_head read once for all of them; a lone drafting slot its own
//                  draft lists) - then one verify_rows of sum (K + 1) rows, per slot the agreeing prefix and its commit;
//                  "own drafts": every slot's own M = 1 draft lists instead (the comparison)
//
// Printed per arm (median over rounds): ms a step, ids a step (all slots), aggregate t/s (ids / s, the card's
// throughput), t/s per request, and against the lone plain step. "draft:" is the draft lists' share of an MTP step.
// The rule the server's SlotsEngine takes from it (mtp_active_max: above how many rows a step drafts nothing) is the
// largest n whose best MTP arm beats n plain.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/engine.h"

namespace {
using Clock = std::chrono::steady_clock;
using runtime::Engine;

std::vector<uint32_t> read_ids(const std::string& p) {
  std::vector<uint32_t> v;
  FILE* f = std::fopen(p.c_str(), "r");
  if (!f) return v;
  unsigned x;
  while (std::fscanf(f, "%u", &x) == 1) v.push_back(x);
  std::fclose(f);
  return v;
}
std::vector<uint32_t> parse_list(const std::string& s) {
  std::vector<uint32_t> v;
  size_t i = 0;
  while (i < s.size()) {
    const size_t j = s.find(',', i);
    v.push_back(uint32_t(std::stoul(s.substr(i, j == std::string::npos ? std::string::npos : j - i))));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return v;
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct Arm {
  std::string name;
  uint32_t n = 1, k = 0;     // n slots; k: one slot's K (n = 1)
  std::vector<uint32_t> ks;  // n > 1: every slot's K
  bool own = false;          // n > 1: every slot's own draft lists (no draft_rows)
  std::vector<double> ms, ids, draft_ms;
  uint32_t rows() const {
    uint32_t r = 0;
    for (uint32_t k : ks) r += k + 1;
    return n == 1 ? k + 1 : r;
  }
  bool plain() const {
    for (uint32_t k : ks) if (k) return false;
    return k == 0;
  }
};
Arm arm(std::string name, uint32_t n, uint32_t k) {
  Arm a;
  a.name = std::move(name);
  a.n = n;
  a.k = k;
  return a;
}
Arm rows_arm(const std::vector<uint32_t>& ks, bool own) {
  Arm a;
  a.n = uint32_t(ks.size());
  a.ks = ks;
  a.own = own;
  bool plain = true;
  for (uint32_t k : ks) plain = plain && k == 0;
  a.name = std::to_string(a.n) + (plain ? " plain" : " K=");
  if (!plain)
    for (size_t i = 0; i < ks.size(); ++i) a.name += (i ? "," : "") + std::to_string(ks[i]);
  if (own) a.name += " own drafts";
  return a;
}

// One step of an arm; returns ids emitted, adds the draft time to *draft_ms.
uint32_t step(Engine& e, const Arm& a, double* draft_ms) {
  if (a.n == 1) {
    e.select(0);
    uint32_t j = 0;
    if (a.k > 0) {
      const Clock::time_point t0 = Clock::now();
      e.draft(a.k);
      *draft_ms += ms_since(t0);
    }
    e.verify(a.k);
    while (j < a.k && e.draft_ids()[j] == e.verify_ids()[j]) ++j;
    e.commit(j, e.verify_ids()[j]);
    return j + 1;
  }
  std::vector<Engine::Run> runs, drafting;
  std::vector<std::vector<uint32_t>> d(a.n);
  for (uint32_t q = 0; q < a.n; ++q) {
    runs.push_back({q, a.ks[q]});
    if (a.ks[q] > 0) drafting.push_back({q, a.ks[q]});
  }
  const Clock::time_point t0 = Clock::now();
  if (drafting.size() >= 2 && !a.own) {
    e.draft_rows(drafting);
  } else {
    for (const Engine::Run& r : drafting) {
      e.select(r.slot);
      e.draft(r.k);
    }
  }
  *draft_ms += ms_since(t0);
  for (const Engine::Run& r : drafting) {
    e.select(r.slot);
    d[r.slot] = e.draft_ids();
  }
  e.verify_rows(runs);
  uint32_t ids = 0, r0 = 0;
  for (uint32_t q = 0; q < a.n; ++q) {
    const uint32_t* out = e.rows_ids() + r0;
    const uint32_t k = a.ks[q];
    uint32_t j = 0;
    while (j < k && d[q][j] == out[j]) ++j;
    e.select(q);
    e.commit(j, out[j]);
    ids += j + 1;
    r0 += k + 1;
  }
  return ids;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <snapshot> [--kv bf16|int8] [--slots L0,..] [--depth N] [--steps N] [--rounds N]\n",
                 argv[0]);
    return 2;
  }
  std::string kv_s = "bf16";
  std::vector<uint32_t> slots = {32768, 32768, 32768, 32768};
  uint32_t depth = 4096, steps = 48, rounds = 3;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--kv") kv_s = next();
    else if (a == "--slots") slots = parse_list(next());
    else if (a == "--depth") depth = uint32_t(std::stoul(next()));
    else if (a == "--steps") steps = uint32_t(std::stoul(next()));
    else if (a == "--rounds") rounds = uint32_t(std::stoul(next()));
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  const runtime::KvCache kv = runtime::parse_kv_cache(kv_s.c_str());
  const std::vector<uint32_t> file = read_ids("tests/golden/prompts/long32k.ids");
  if (file.empty()) {
    std::fprintf(stderr, "tests/golden/prompts/long32k.ids not found (run from the source tree)\n");
    return 2;
  }
  uint32_t total = 0;
  for (uint32_t l : slots) total += l;
  const uint32_t S = uint32_t(slots.size());
  l0::Context ctx(0);
  Engine e(ctx, loader::load(ctx, argv[1], total, /*mtp=*/true, loader::LmHeadForm::Int8), total, false, kv, nullptr,
           slots);
  e.prepare_prefill();
  std::printf("probe_slots_speed: %u slots (%s), %s KV, depth %u, %u steps an arm, median of %u rounds\n%s\n", S,
              runtime::slot_layout_name(slots).c_str(), runtime::kv_cache_name(kv), depth, steps, rounds,
              e.memory_line().c_str());
  for (uint32_t q = 0; q < S; ++q) {
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < depth; ++i) ids.push_back(file[(size_t(q) * 7919 + i) % file.size()]);
    e.select(q);
    const Clock::time_point t0 = Clock::now();
    e.prefill(ids);
    std::printf("prefill: slot %u, %u ids in %.2f s\n", q, depth, ms_since(t0) / 1000);
  }
  std::vector<Arm> arms;
  arms.push_back(arm("1 plain", 1, 0));
  for (uint32_t k = 1; k <= 3; ++k) arms.push_back(arm("1 mtp K=" + std::to_string(k), 1, k));
  // n slots: plain; K the same everywhere (rows draft and, for the record, own drafts); and the 8-row budget's two
  // allocations at 3 / 4 slots - the oldest request first (3, then what is left) or one at a time round the rows.
  for (uint32_t n = 2; n <= S; ++n) {
    arms.push_back(rows_arm(std::vector<uint32_t>(n, 0), false));
    for (uint32_t k = 1; k <= 3; ++k)
      if (n * (k + 1) <= Engine::kMaxRows) {
        arms.push_back(rows_arm(std::vector<uint32_t>(n, k), false));
        if (k == 1) arms.push_back(rows_arm(std::vector<uint32_t>(n, k), true));
      }
  }
  if (S >= 3) {
    arms.push_back(rows_arm({3, 2, 0}, false));
    arms.push_back(rows_arm({2, 2, 1}, false));
    arms.push_back(rows_arm({3, 0, 0}, false));
  }
  if (S >= 4) {
    arms.push_back(rows_arm({3, 1, 0, 0}, false));
    arms.push_back(rows_arm({3, 0, 0, 0}, false));
  }
  for (Arm& a : arms) {   // warm-up
    double d = 0;
    for (int i = 0; i < 4; ++i) step(e, a, &d);
  }
  for (uint32_t r = 0; r < rounds; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      Arm& a = arms[(i + r) % arms.size()];
      double d = 0;
      uint32_t ids = 0;
      const Clock::time_point t0 = Clock::now();
      for (uint32_t s = 0; s < steps; ++s) ids += step(e, a, &d);
      const double ms = ms_since(t0);
      a.ms.push_back(ms / steps);
      a.ids.push_back(double(ids) / steps);
      a.draft_ms.push_back(d / steps);
    }
  const double base = median(arms[0].ms);
  std::printf("| arm | rows a step | ms a step | ids a step | aggregate t/s | t/s per request | x the lone plain step | "
              "draft ms a step |\n");
  for (const Arm& a : arms) {
    const double ms = median(a.ms), ids = median(a.ids);
    std::printf("| %s | %u | %.2f | %.3f | %.1f | %.1f | %.3f | %.2f |\n", a.name.c_str(), a.rows(), ms, ids,
                ids * 1000 / ms, ids * 1000 / ms / a.n, ms / base, median(a.draft_ms));
  }
  // The rule: per n, the best arm, and whether drafting beats plain rows there.
  std::printf("rule: ");
  for (uint32_t n = 1; n <= S; ++n) {
    double plain = 0, best = 0;
    std::string best_name;
    for (const Arm& a : arms) {
      if (a.n != n) continue;
      const double tps = median(a.ids) * 1000 / median(a.ms);
      if (a.plain()) plain = tps;
      if (tps > best) {
        best = tps;
        best_name = a.name;
      }
    }
    std::printf("%u requests: best %s (%.1f t/s, plain %.1f)%s", n, best_name.c_str(), best, plain, n < S ? "; " : "\n");
  }
  return 0;
}

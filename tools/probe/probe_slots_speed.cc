// probe_slots_speed - spec 13c (plan 13c Task 5): the decode speed of runtime::Engine's sequence slots, through the
// engine API (host writes, submit, fence - what the server pays per step, without HTTP and the scheduler).
//
//   probe_slots_speed <snapshot> [--kv bf16|int8] [--seqs 4] [--cap 32768] [--depth 4096] [--steps 48] [--rounds 3]
//                     [--depths D0,D1,..] [--arms "NAME|NAME|.."]
//
// --seqs / --cap: the engine's sequences and each one's cap; they share a KV page pool of seqs x cap positions.
// --depths: each slot's prefill length (default every slot `depth`) - e.g. slot 0 deep beside short slots.
// --arms: only the arms of these names (as printed, e.g. "1 plain|4 K=1,1,1,1"), in the order below.
// --profile-rows N (plan 31 Task 13): instead of the arms, the row list at 2 / 4 / 6 / 8 rows (slots 0 and 1, M / 2 - 1
//   drafts each) built with per-launch events and replayed N times: ms a step per kernel (its variant with the M taken
//   out, summed over the layers) at each M - where rows past 4 cost. Events add a flush each: shares, not times.
//
// --wide N (plan 40a Task 3): instead of the arms, T(W) - one row step across the slots at W = 8 / 16 / 32 rows (every
//   slot K = W / S - 1 drafts: ids of the prompt file, not the head's), against slot 0's plain step (its own verify(0)),
//   interleaved, N steps an arm a round: each step's verify_rows (the staging, the replay, the wait) timed; then every
//   run committed whole ("all": its k + 1 rows pending - the next step replays them, the served case) or at its pending
//   id ("0": one row). Medians over the rounds, and x the plain step. The MTP head's KV fill rides in every row list.
// --moe-ab (PERF-120, with --wide on a MoE model): the 16- and 32-row arms twice - the engine's lists (the union read)
//   and rebuilt with the per-row MoE block (Engine::probe_rows_moe_union) - each round both halves, A B then B A.
//
// One engine (Qwen3.8, its MTP head, the int8 lm_head - the served form) with the given sequences (default 4 up to
// 32768, a pool of 131072 positions),
// every slot prefilled with `depth` ids of tests/golden/prompts/long32k.ids (each from its own offset). Then, per
// round, every arm in a rotated order, `steps` steps each, greedy - real drafts, real acceptance:
//
//   1 plain        slot 0 alone, its own lists: verify(0) + commit (today's --spec off step with the head loaded)
//   1 plain s<q>   the same on slot q >= 1 (its own depth, --depths)
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/buffer_sizes.h"
#include "runtime/buffers.h"
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
  uint32_t slot = 0;         // n = 1: the slot
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
    e.select(a.slot);
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
    std::fprintf(stderr,
                 "usage: %s <snapshot> [--kv bf16|int8] [--seqs N] [--cap N] [--depth N] [--steps N] [--rounds N]\n",
                 argv[0]);
    return 2;
  }
  std::string kv_s = "bf16";
  uint32_t S = 4, cap = 32768;
  uint32_t depth = 4096, steps = 48, rounds = 3;
  std::vector<uint32_t> depths;
  std::vector<std::string> only;
  uint32_t profile_rows = 0, wide = 0;
  bool moe_ab = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--kv") kv_s = next();
    else if (a == "--seqs") S = uint32_t(std::stoul(next()));
    else if (a == "--cap") cap = uint32_t(std::stoul(next()));
    else if (a == "--depth") depth = uint32_t(std::stoul(next()));
    else if (a == "--steps") steps = uint32_t(std::stoul(next()));
    else if (a == "--rounds") rounds = uint32_t(std::stoul(next()));
    else if (a == "--depths") depths = parse_list(next());
    else if (a == "--profile-rows") profile_rows = uint32_t(std::stoul(next()));
    else if (a == "--wide") wide = uint32_t(std::stoul(next()));
    else if (a == "--moe-ab") moe_ab = true;
    else if (a == "--arms") {
      const std::string l = next();
      for (size_t b = 0; b <= l.size();) {
        const size_t e2 = std::min(l.find('|', b), l.size());
        only.push_back(l.substr(b, e2 - b));
        b = e2 + 1;
      }
    }
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (S < 2 || cap % runtime::kKvPage != 0) {
    std::fprintf(stderr, "--seqs needs two sequences or more and --cap whole KV pages of %u\n", runtime::kKvPage);
    return 2;
  }
  const runtime::KvCache kv = runtime::parse_kv_cache(kv_s.c_str());
  const std::vector<uint32_t> file = read_ids("tests/golden/prompts/long32k.ids");
  if (file.empty()) {
    std::fprintf(stderr, "tests/golden/prompts/long32k.ids not found (run from the source tree)\n");
    return 2;
  }
  const uint32_t total = S * cap;   // the pool: every sequence at its cap at once
  l0::Context ctx(0);
  Engine e(ctx, loader::load(ctx, argv[1], total, /*mtp=*/true, loader::LmHeadForm::Int8), total, false, kv, nullptr,
           std::vector<uint32_t>(S, cap), total / runtime::kKvPage);
  e.prepare_prefill();
  std::printf("probe_slots_speed: %u sequences up to %u sharing %u KV pages, %s KV, depth %u, %u steps an arm, median of "
              "%u rounds\n%s\n",
              S, cap, e.kv_pool_pages(), runtime::kv_cache_name(kv), depth, steps, rounds, e.memory_line().c_str());
  if (depths.empty()) depths.assign(S, depth);
  if (depths.size() != S) {
    std::fprintf(stderr, "--depths names %zu slots, the engine has %u\n", depths.size(), S);
    return 2;
  }
  for (uint32_t q = 0; q < S; ++q) {
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < depths[q]; ++i) ids.push_back(file[(size_t(q) * 7919 + i) % file.size()]);
    e.select(q);
    const Clock::time_point t0 = Clock::now();
    e.prefill(ids);
    std::printf("prefill: slot %u, %u ids in %.2f s\n", q, depths[q], ms_since(t0) / 1000);
  }
  if (wide) {
    struct W {
      uint32_t rows = 0;   // 0: slot 0's plain step
      bool all = false;
      std::vector<double> ms;
      bool via16 = false;    // PERF-120: the step on the 16-row list (Engine::probe_rows_wide_from)
      bool per_row = false;  // PERF-120 --moe-ab: the lists with the per-row MoE block
    };
    std::vector<W> ws{{0, false, {}}};
    moe_ab = moe_ab && e.model().desc->is_moe() && e.max_rows() > Engine::kMaxRows;
    for (const uint32_t rows : {4u, 8u, 16u, 32u})
      if (rows % S == 0 && rows / S >= 1 && rows / S <= Engine::kMaxRows && rows <= e.max_rows()) {
        ws.push_back({rows, true, {}});   // the served case: every run committed whole
        if (rows <= Engine::kMaxRows && e.max_rows() > Engine::kMaxRows) ws.push_back({rows, true, {}, true});
        if (rows > Engine::kMaxRows && moe_ab) ws.push_back({rows, true, {}, false, true});
      }
    size_t fid = 0;
    auto one = [&](W& w) {
      if (w.rows == 0) {
        e.select(0);
        const Clock::time_point t0 = Clock::now();
        e.verify(0);
        const double ms = ms_since(t0);
        e.commit(0, e.verify_ids()[0]);
        return ms;
      }
      const uint32_t k = w.rows / S - 1;
      std::vector<Engine::Run> runs;
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        for (uint32_t j = 1; j <= k; ++j) e.control()->cur_token[j] = file[fid++ % file.size()];
        runs.push_back({q, k});
      }
      e.probe_rows_wide_from(w.via16 ? 2 : 0);
      const Clock::time_point t0 = Clock::now();
      e.verify_rows(runs);
      const double ms = ms_since(t0);
      e.probe_rows_wide_from(0);
      const std::vector<uint32_t> out(e.rows_ids(), e.rows_ids() + w.rows);
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        const uint32_t j = w.all ? k : 0;
        e.commit(j, out[q * (k + 1) + j]);
      }
      return ms;
    };
    // --moe-ab: a round's halves on the union lists and on the per-row ones (A B, then B A); a wide arm in its own half
    const auto half = [&](bool per_row) {
      if (moe_ab) e.probe_rows_moe_union(!per_row);
      return [per_row](const W& w) { return w.rows <= Engine::kMaxRows || w.per_row == per_row; };
    };
    for (const bool pr : {false, true}) {
      const auto in = half(pr);
      for (W& w : ws)
        if (in(w))
          for (int i = 0; i < 4; ++i) one(w);
      if (!moe_ab) break;
    }
    for (uint32_t r = 0; r < rounds; ++r)
      for (uint32_t hf = 0; hf < (moe_ab ? 2u : 1u); ++hf) {
        const auto in = half((r + hf) % 2 == 1);
        for (uint32_t i = 0; i < wide; ++i)
          for (size_t a = 0; a < ws.size(); ++a) {   // interleaved: every arm once a step, the order rotated
            W& w = ws[(a + i + r) % ws.size()];
            if (in(w)) w.ms.push_back(one(w));
          }
      }
    if (moe_ab) e.probe_rows_moe_union(true);
    const double base = median(ws[0].ms);
    std::printf("\nT(W): one row step of %u slots x K = W / %u - 1, every arm once a step (rotated), %u steps x %u rounds; "
                "slot 0 ends at %u\n| rows | committed | list | ms (median) | x the plain step |\n|---|---|---|---|---|\n",
                S, S, wide, rounds, (e.select(0), e.pos()));
    for (const W& w : ws)
      std::printf("| %s | %s | %s | %.2f | %.3f |\n", w.rows ? std::to_string(w.rows).c_str() : "1 (plain)",
                  w.rows ? (w.all ? "all" : "0") : "-",
                  w.via16 ? "the 16-row list" : w.per_row ? "per-row MoE" : "engine",
                  median(w.ms), median(w.ms) / base);
    // PERF-120 (Ornith): the distinct routed experts a step of W rows reads, mean a layer, from the route rows of one
    // step at each width (moe.cl reads every row's TOP_K experts on its own: W x TOP_K expert reads a layer)
    if (const model::ModelDesc& d = *e.model().desc; d.is_moe()) {
      const runtime::MoeScratchLayout ml = runtime::moe_scratch_layout(d);
      l0::CmdList imm = l0::CmdList::immediate(ctx);
      std::printf("\ndistinct routed experts a layer (of %u, top %u) in one row step:\n| rows | distinct | reads |\n"
                  "|---|---|---|\n", d.moe.experts, d.moe.top_k);
      for (W& w : ws) {
        if (!w.rows || w.via16 || w.per_row) continue;
        one(w);
        std::vector<uint32_t> route(size_t(w.rows) * kernels::moe_route::kWords);
        double sum = 0;
        for (uint32_t l = 0; l < d.layers; ++l) {
          imm.copy(route.data(), e.buffers().moe->as<uint8_t>() + ml.route_at(l), route.size() * 4);
          std::vector<bool> seen(d.moe.experts, false);
          uint32_t n = 0;
          for (uint32_t r = 0; r < w.rows; ++r)
            for (uint32_t k = 0; k < d.moe.top_k; ++k) {
              const uint32_t x = route[size_t(r) * kernels::moe_route::kWords + kernels::moe_route::kIds + k];
              if (x < d.moe.experts && !seen[x]) seen[x] = true, ++n;
            }
          sum += n;
        }
        std::printf("| %u | %.1f | %u |\n", w.rows, sum / d.layers, w.rows * d.moe.top_k);
      }
    }
    // PERF-120: the key blocks' arithmetic - one step of 8 and of 32 rows from the same state, each count against
    // the engine's lists
    {
      std::vector<l0::Mem> st;
      std::vector<uint32_t> pos(S), pend(S);
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        e.flush_projection();
        pos[q] = e.pos();
        pend[q] = e.control()->cur_token[0];
        st.emplace_back(ctx, l0::MemKind::Host, e.state_bytes());
        e.save_state(st.back().ptr());
      }
      l0::CmdList imm = l0::CmdList::immediate(ctx);
      for (const uint32_t rows : {8u}) {
        if (e.max_rows() <= Engine::kMaxRows || rows % S) continue;
        const uint32_t k = rows / S - 1;
        std::vector<std::vector<float>> logits;
        std::vector<std::vector<uint32_t>> ids;
        const std::vector<uint32_t> ts = {0, 1};   // 1: on the 16-row list
        for (const uint32_t t : ts) {
          std::vector<Engine::Run> runs;
          for (uint32_t q = 0; q < S; ++q) {
            e.select(q);
            e.load_state(st[q].ptr(), pos[q]);
            e.control()->cur_token[0] = pend[q];
            for (uint32_t j = 1; j <= k; ++j) e.control()->cur_token[j] = file[(997 * q + j) % file.size()];
            runs.push_back({q, k});
          }
          e.probe_rows_wide_from(t ? 2 : 0);
          e.verify_rows(runs);
          e.probe_rows_wide_from(0);
          ids.emplace_back(e.rows_ids(), e.rows_ids() + rows);
          logits.emplace_back(size_t(rows) * model::Qwen35::kVocab);
          imm.copy(logits.back().data(), e.rows_logits_device(), logits.back().size() * 4);
        }
        for (size_t t = 1; t < ts.size(); ++t) {
          uint32_t same = 0;
          double worst = 0;
          for (size_t i = 0; i < rows; ++i) same += ids[t][i] == ids[0][i];
          for (size_t i = 0; i < logits[0].size(); ++i)
            worst = std::max(worst, double(std::fabs(logits[t][i] - logits[0][i])));
          std::printf("%u rows on the 16-row list against their own, from the same state: %u of %u ids equal, "
                      "max |logit diff| %.3e\n", rows, same, rows, worst);
        }
      }
    }
    if (!profile_rows) return 0;
    // With --profile-rows N: each width's list built with per-launch events over the table the last verify_rows of that
    // width staged (the session is not committed after), N replays - ms a step by kernel family (the variant with its
    // M taken out), summed over the layers. Events add a flush each: shares, not times.
    l0::Queue pq(ctx);
    l0::Fence pf(pq);
    const std::regex m_part("_M[0-9]+|_W[0-9]+");
    const std::string head = "L" + std::to_string(e.model().desc->layers) + " ";
    std::map<std::string, std::map<uint32_t, double>> fam;
    std::map<uint32_t, double> tot;
    std::vector<uint32_t> Ws;
    for (const W& w : ws)
      if (w.rows && !w.via16 && !w.per_row) Ws.push_back(w.rows);
    for (const uint32_t rows : Ws) {
      const uint32_t k = rows / S - 1;
      std::vector<Engine::Run> runs;
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        for (uint32_t j = 1; j <= k; ++j) e.control()->cur_token[j] = file[fid++ % file.size()];
        runs.push_back({q, k});
      }
      e.verify_rows(runs);
      runtime::ProfileEvents prof(ctx);
      runtime::BatchRowsBind rb = e.rows_bind();
      if (rows > Engine::kMaxRows) {   // the engine's wide lists' binding (Engine::build_rows_wide)
        rb.attn_blocks = Engine::kRowsAttnBlocks;
        rb.moe_union = e.model().desc->is_moe();
      }
      runtime::CapturedStep l = runtime::build_rows(ctx, e.model(), e.buffers(), rb, rows, &prof);
      std::vector<double> acc(l.kernel_count, 0.0);
      for (uint32_t i = 0; i < profile_rows + 4; ++i) {
        for (l0::Event& ev : prof.events) ev.reset();
        pq.execute(l.list, &pf);
        pf.wait();
        if (i >= 4)
          for (size_t j = 0; j < l.kernel_count; ++j) acc[j] += prof.events[j].duration_us();
      }
      for (size_t j = 0; j < l.kernel_count; ++j) {
        const std::string& lb = l.labels[j];
        const size_t a = lb.find(' ');
        const std::string key = std::regex_replace(lb.substr(a + 1), m_part, "") + (lb.rfind(head, 0) == 0 ? " (head)" : "");
        fam[key][rows] += acc[j] / profile_rows / 1e3;
        tot[rows] += acc[j] / profile_rows / 1e3;
      }
    }
    std::printf("\nprofiled row lists (%u slots x K = W / %u - 1; ms a step, events add a flush each):\n| kernel |", S, S);
    for (const uint32_t rows : Ws) std::printf(" W=%u |", rows);
    std::printf("\n|---|");
    for (size_t i = 0; i < Ws.size(); ++i) std::printf("---|");
    std::printf("\n");
    for (auto& [name, v] : fam) {
      std::printf("| %s |", name.c_str());
      for (const uint32_t rows : Ws) std::printf(" %.3f |", v[rows]);
      std::printf("\n");
    }
    std::printf("| sum |");
    for (const uint32_t rows : Ws) std::printf(" %.3f |", tot[rows]);
    std::printf("\n");
    return 0;
  }
  if (profile_rows) {
    l0::Queue q(ctx);
    l0::Fence f(q);
    const std::vector<uint32_t> Ms = {2, 4, 6, 8};
    std::map<std::string, std::map<uint32_t, double>> fam;
    std::map<uint32_t, double> tot;
    const std::regex m_part("_M[0-9]+");
    const std::string head = "L" + std::to_string(e.model().desc->layers) + " ";   // the MTP head's launches
    for (const uint32_t M : Ms) {
      const uint32_t k = M / 2 - 1;
      for (uint32_t s = 0; s < 2 && k > 0; ++s) {
        e.select(s);
        e.draft(k);
      }
      e.verify_rows({{0, k}, {1, k}});   // stages the row table and the controls; nothing is committed
      runtime::ProfileEvents prof(ctx);
      runtime::CapturedStep l = runtime::build_rows(ctx, e.model(), e.buffers(), e.rows_bind(), M, &prof);
      std::vector<double> acc(l.kernel_count, 0.0);
      for (uint32_t i = 0; i < profile_rows + 4; ++i) {
        for (l0::Event& ev : prof.events) ev.reset();
        q.execute(l.list, &f);
        f.wait();
        if (i >= 4)
          for (size_t j = 0; j < l.kernel_count; ++j) acc[j] += prof.events[j].duration_us();
      }
      for (size_t j = 0; j < l.kernel_count; ++j) {
        const std::string& lb = l.labels[j];
        const size_t a = lb.find(' ');
        const std::string key = std::regex_replace(lb.substr(a + 1), m_part, "") + (lb.rfind(head, 0) == 0 ? " (head)" : "");
        fam[key][M] += acc[j] / profile_rows / 1e3;
        tot[M] += acc[j] / profile_rows / 1e3;
      }
    }
    std::printf("\nprofiled row lists (slots 0 and 1, M / 2 - 1 drafts each; ms a step, events add a flush each):\n"
                "| kernel | M=2 | M=4 | M=6 | M=8 | M=8 - M=4 |\n|---|---|---|---|---|---|\n");
    for (auto& [name, v] : fam)
      std::printf("| %s | %.3f | %.3f | %.3f | %.3f | %+.3f |\n", name.c_str(), v[2], v[4], v[6], v[8], v[8] - v[4]);
    std::printf("| sum | %.3f | %.3f | %.3f | %.3f | %+.3f |\n", tot[2], tot[4], tot[6], tot[8], tot[8] - tot[4]);
    return 0;
  }
  std::vector<Arm> arms;
  arms.push_back(arm("1 plain", 1, 0));
  for (uint32_t k = 1; k <= 3; ++k) arms.push_back(arm("1 mtp K=" + std::to_string(k), 1, k));
  for (uint32_t q = 1; q < S; ++q) {
    arms.push_back(arm("1 plain s" + std::to_string(q), 1, 0));
    arms.back().slot = q;
  }
  // n slots: plain; K the same everywhere (rows draft and, for the record, own drafts) within the engine's rows; and the
  // 8-row budget's two
  // allocations at 3 / 4 slots - the oldest request first (3, then what is left) or one at a time round the rows.
  for (uint32_t n = 2; n <= S; ++n) {
    arms.push_back(rows_arm(std::vector<uint32_t>(n, 0), false));
    for (uint32_t k = 1; k <= 3; ++k)
      if (n * (k + 1) <= e.max_rows()) {   // plan 40a Task 5: 16 rows (K 3 x 4) where the wide lists exist
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
  // Plan 31 Task 13: the scheduler's deal (scheduler.cc: every row its pending id, the drafts one at a time round the
  // rows, each up to K 3) at a row budget of 6 and 7 - the arms a smaller budget runs, beside the 8-row ones above.
  for (const uint32_t budget : {6u, 7u})
    for (uint32_t n = 2; n <= S; ++n) {
      std::vector<uint32_t> ks(n, 0);
      for (uint32_t left = budget - n, i = 0; left > 0 && i < 3 * n; ++i)
        if (ks[i % n] < 3) {
          ++ks[i % n];
          --left;
        }
      Arm a = rows_arm(ks, false);
      bool have = false;
      for (const Arm& x : arms) have = have || x.name == a.name;
      if (!have) arms.push_back(std::move(a));
    }
  if (!only.empty()) {
    std::vector<Arm> kept;
    for (Arm& a : arms)
      if (std::find(only.begin(), only.end(), a.name) != only.end()) kept.push_back(std::move(a));
    if (kept.size() != only.size()) {
      std::fprintf(stderr, "--arms: %zu of the %zu names are arms of this layout\n", kept.size(), only.size());
      return 2;
    }
    arms = std::move(kept);
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
      if (a.n != n || a.slot != 0) continue;
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

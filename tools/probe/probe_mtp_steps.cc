// probe_mtp_steps - spec 8 plan 8b: the production MTP lists' wall time per call.
//
//   probe_mtp_steps <snapshot> [depth = 4096] [calls = 32] [rounds = 3] [lm_head = bf16]
//                   [draft_vocab = off] [max_len = 16384] [max_m = 4]
//
// `lm_head` (spec 9 H2): bf16 (the checkpoint's) or int8; the draft list reads it too.
// `draft_vocab` (spec 8 §11): off, 32k, 64k or 128k (either head form) - the draft list reads
// the compact head instead; V' = tokenizer.json's added tokens (the EOS ids among them on
// Qwen3.8 and Agnes) and the lowest ids, no ranked list (the step time depends on |V'|
// only). The draft arms then price `--mtp-cost`'s draft row for that size (spec 8 §10).
//
// `max_len` (spec 8 §12, box queue row 6): the engine's length - 16384 as before by default;
// the cost table at depth runs e.g. `... 32768 32 3 int8 off 65536` and `... 120000 32 3 int8
// off 131072`. It must leave room for the drift below (depth + 1024 <= max_len).
//
// One engine with the head, `depth` ids of tests/golden/prompts/long32k.ids prefilled
// (the file repeated past its end; cwd = the source tree). Arms, timed as the engine API runs them (host writes, submit,
// fence wait - what 8c's loop pays): verify(k) + commit(0) for k = 0..3, i.e. the verify
// list at M = k + 1, and draft(k) for k = 1..3. Each round runs every arm `calls` times,
// arms in a rotated order, after a warm-up pass; the median over rounds of each arm's
// mean ms/call is printed. commit(0) advances pos by one per call, so a round moves the
// depth by 4 x calls positions (0.8% of 4096 at the defaults) - the same drift for every
// arm, as the order rotates.
//
// `max_m` (spec 19a, plan 19a Task 4: DFlash's verify cost): 4 is everything above,
// unchanged. 5..8 adds verify arms at M = 5..max_m - Qwen3.8 with the int8 head only, a
// build with B70_VERIFY_M8 (the default). The engine's own verify lists stay M = 1..4 over
// its 4 GDN slots; the M = 5..8 lists are runtime::build_verify over a SECOND, probe-owned
// MtpBuffers with 8 slots (+7 gdn_state copies, 1.06 GB on Qwen3.8), driven exactly as
// Engine::verify(k) + commit(0) drive theirs: the head's control (that buffer's hctl) at
// pos - 1 with the M ids, submit, fence, then the commit's 10 KB hidden copy and pos + 1.
// gdn_live stays 0 throughout (commit(0) never moves it), so every list reads and writes
// slot 0 as its row 0 - the same state as the engine's. Draft rows are ids of the file
// past `depth` (the cost does not depend on them). In this mode pos is REWOUND to `depth`
// before every arm's calls, so every arm runs at depth .. depth + calls (the default mode's
// drift would move 4k by two thirds over the larger arm set), and a second table times
// INTERLEAVED PAIRS - verify M = 1 then verify M, the pair's order alternating by round,
// after the warm-up - and prints the median over rounds of each pair's ratio (sequential
// arms mis-state ratios by up to 35% when the clock moves between them).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/draft_vocab.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
std::vector<uint32_t> read_ids(const std::string& p) {
  std::vector<uint32_t> v;
  FILE* f = std::fopen(p.c_str(), "r");
  if (!f) return v;
  unsigned x;
  while (std::fscanf(f, "%u", &x) == 1) v.push_back(x);
  std::fclose(f);
  return v;
}
double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <snapshot> [depth] [calls] [rounds] [bf16|int8] [off|32k|64k|128k]"
                 " [max_len] [max_m 4..8]\n",
                 argv[0]);
    return 2;
  }
  const uint32_t depth = argc > 2 ? std::stoul(argv[2]) : 4096;
  const uint32_t calls = argc > 3 ? std::stoul(argv[3]) : 32;
  const uint32_t rounds = argc > 4 ? std::stoul(argv[4]) : 3;
  const uint32_t max_len = argc > 7 ? std::stoul(argv[7]) : 16384;
  const uint32_t max_m = argc > 8 ? std::stoul(argv[8]) : 4;
  if (max_m < 4 || max_m > runtime::MtpDims::kMaxSlots) {
    std::fprintf(stderr, "max_m %u is outside [4, %u]\n", max_m, runtime::MtpDims::kMaxSlots);
    return 2;
  }
  const bool wide = max_m > 4;   // spec 19a: the M = 5..8 arms
  std::vector<uint32_t> ids = read_ids("tests/golden/prompts/long32k.ids");
  if (ids.empty()) return 2;
  const std::vector<uint32_t> file = ids;
  while (ids.size() < depth) ids.insert(ids.end(), file.begin(), file.end());
  ids.resize(depth);
  if (size_t(depth) + 1024 > max_len) {
    std::fprintf(stderr, "depth %u leaves no room for the arms at max_len %u\n", depth, max_len);
    return 2;
  }
  l0::Context ctx(0);
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  if (argc > 5 && !loader::parse_lm_head_form(argv[5], lm_head)) return 2;
  if (wide && lm_head != loader::LmHeadForm::Int8) {
    std::fprintf(stderr, "max_m %u: the verify lists at M = 5..8 are built for the int8 head only"
                         " (B70_VERIFY_M8, src/kernels/CMakeLists.txt) - pass int8\n", max_m);
    return 2;
  }
  loader::DraftVocabSpec dv;
  if (argc > 6) {
    if (!loader::parse_draft_vocab(argv[6], dv.size)) return 2;
    if (dv.size != 0)
      dv.added = loader::added_token_ids_file(loader::resolve_snapshot(argv[1]) + "tokenizer.json");
  }
  std::printf("lm_head: %s, draft vocab %s\n", loader::lm_head_form_name(lm_head),
              loader::draft_vocab_name(dv.size).c_str());
  if (max_len != 16384) std::printf("max_len: %u\n", max_len);
  runtime::Engine e(ctx, loader::load(ctx, argv[1], max_len, /*mtp=*/true, lm_head, dv), max_len);
  e.prefill(ids);
  using Clock = std::chrono::steady_clock;

  // Spec 19a: the M = 5..max_m verify lists over a probe-owned 8-slot MtpBuffers.
  std::unique_ptr<runtime::MtpBuffers> mtp8;
  std::vector<runtime::CapturedStep> wide_lists;   // [M - 5]
  std::unique_ptr<l0::Queue> wq;
  std::unique_ptr<l0::Fence> wfence;
  std::unique_ptr<l0::CmdList> wimm;
  runtime::Control* ctl = e.buffers().control.as<runtime::Control>();
  if (wide) {
    mtp8 = std::make_unique<runtime::MtpBuffers>(ctx, max_len, *e.model().desc, e.draft_vocab(),
                                                 e.kv_cache(), runtime::MtpDims::kMaxSlots);
    for (uint32_t M = 5; M <= max_m; ++M)
      wide_lists.push_back(runtime::build_verify(ctx, e.model(), e.buffers(), *mtp8, M));
    wq = std::make_unique<l0::Queue>(ctx);
    wfence = std::make_unique<l0::Fence>(*wq);
    wimm = std::make_unique<l0::CmdList>(l0::CmdList::immediate(ctx));
    std::printf("verify M = 5..%u: probe-owned MtpBuffers, %u GDN slots, %.2f GB; %zu launches"
                " per list (engine's M = 1..4: %zu)\n",
                max_m, mtp8->slots, mtp8->bytes() / 1e9, wide_lists.front().kernel_count,
                e.verify_step(1).kernel_count);
    if (ctl->gdn_live != 0) {
      std::fprintf(stderr, "gdn_live is %u after the prefill, not 0\n", ctl->gdn_live);
      return 1;
    }
  }
  const size_t hid = size_t(e.model().desc->hidden) * 2;
  // One M >= 5 verify + commit(0): Engine::verify(k) + commit(0, verify_ids()[0]) on the
  // probe's lists (engine.cc), the head's control in mtp8's hctl.
  auto wide_call = [&](uint32_t M) {
    runtime::Control* h = mtp8->hctl.as<runtime::Control>();
    const uint32_t pos = ctl->pos;
    for (uint32_t r = 1; r < M; ++r) ctl->cur_token[r] = file[(depth + r) % file.size()];
    ctl->n_active = M;
    h->pos = pos - 1;
    h->n_active = M;
    for (uint32_t r = 0; r < M; ++r) h->cur_token[r] = ctl->cur_token[r];
    wq->execute(wide_lists[M - 5].list, wfence.get());
    wfence->wait();
    wimm->copy(mtp8->hh.ptr(), mtp8->hh.as<uint8_t>() + hid, hid);
    ctl->pos = pos + 1;
    ctl->n_active = 1;
    ctl->cur_token[0] = ctl->out_token[0];
  };

  struct Arm {
    std::string name;
    uint32_t k;
    bool draft;
    std::vector<double> ms;
  };
  std::vector<Arm> arms;
  for (uint32_t k = 0; k + 1 <= max_m; ++k)
    arms.push_back({"verify M=" + std::to_string(k + 1), k, false, {}});
  for (uint32_t k = 1; k <= 3; ++k) arms.push_back({"draft k=" + std::to_string(k), k, true, {}});
  auto run = [&](Arm& a, uint32_t n) {
    if (wide) ctl->pos = depth;   // spec 19a mode: every arm at the same depth
    const auto t0 = Clock::now();
    for (uint32_t i = 0; i < n; ++i) {
      if (a.draft) {
        e.draft(a.k);
      } else if (a.k + 1 > 4) {
        wide_call(a.k + 1);
      } else {
        e.verify(a.k);
        e.commit(0, e.verify_ids()[0]);
      }
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / n;
  };
  for (Arm& a : arms) run(a, 8);   // warm-up
  for (uint32_t r = 0; r < rounds; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      Arm& a = arms[(i + r) % arms.size()];
      a.ms.push_back(run(a, calls));
    }
  if (wide)
    std::printf("probe_mtp_steps: depth %u (rewound before every arm; up to +%u), %u calls x %u"
                " rounds, verify M = 1..%u\n",
                depth, calls, calls, rounds, max_m);
  else
    std::printf("probe_mtp_steps: depth %u..%u, %u calls x %u rounds\n", depth, e.pos(), calls,
                rounds);
  const double base = [&] {
    std::vector<double> v = arms[0].ms;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }();
  for (Arm& a : arms) {
    std::vector<double> v = a.ms;
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    std::printf("| %-12s | %8.3f ms | x M=1 verify %.3f | range %.3f-%.3f |\n", a.name.c_str(), med,
                med / base, v.front(), v.back());
  }

  if (wide) {
    // Interleaved pairs (spec 19a): verify M = 1 and verify M back to back, the order
    // alternating by round, `calls` each; the ratio of the pair; median over rounds.
    std::printf("interleaved pairs (verify M=1 and verify M back to back, order alternating by"
                " round, %u calls each, median of %u):\n", calls, rounds);
    Arm& one = arms[0];
    for (uint32_t M = 2; M <= max_m; ++M) {
      Arm& cand = arms[M - 1];
      std::vector<double> ratio, ms1, msm;
      for (uint32_t r = 0; r < rounds; ++r) {
        double t1, tm;
        if ((r + M) % 2 == 0) {
          t1 = run(one, calls);
          tm = run(cand, calls);
        } else {
          tm = run(cand, calls);
          t1 = run(one, calls);
        }
        ratio.push_back(tm / t1);
        ms1.push_back(t1);
        msm.push_back(tm);
      }
      std::vector<double> rs = ratio;
      std::sort(rs.begin(), rs.end());
      std::printf("| pair M=%u     | %8.3f ms | x M=1 verify %.3f | range %.3f-%.3f | M=1 %.3f ms |\n",
                  M, median_of(msm), median_of(ratio), rs.front(), rs.back(), median_of(ms1));
    }
  }
  return 0;
}

// Spec 16d (plan 16d Review Focus 2 and 5, spec 16 P2: spec 7's C1 / C2 with --pp 2): the
// prefix cache's request path - server::PrefixSession over b70-serve's adapters, no HTTP - on
// two cards is BITWISE the same path on one card. Where prefix_gpu_test judges cached against
// cold under the tie-aware rule (the chunks differ), here both sides take the SAME path - the
// same restores, the same tail chunks, the same decode steps - so PP's P1 makes every byte
// equal: the plan (kind, restart point), the final turn's ids and every step's logits row, and
// the session at the end (save_state / save_kv over [0, pos)).
//
//   one card    EngineAdapter over runtime::Engine on device 0 (b70-serve's --pp 1 adapter)
//   two cards   cli::pp::PipelineEngineAdapterT<runtime::PipelineEngine> (b70-serve --pp 2),
//               copy at the auto split (argv [4]: copy | peer)
//
// Sequences (prefix_gpu_test's, L = long32k.ids, C = code.ids repeated): a continue, b prompt
// end, c mid-block, d block boundary, e side request (every KV block from the host), f the same
// prompt. Each turn decodes 32 ids; with MTP (argv [3] K > 0, the head loaded on both) every
// turn decodes through step_many(greedy, K) (the server's loop), and the comparison is the ids
// and the session (the speculative bursts do not leave a per-step logits row).
//
// argv: [1] snapshot, [2] prompts dir, [3] K (0 = no head), [4] copy | peer, [5] split_last
// (1 = PrefixSession's opt-in split, b70-serve --prefix-split-last), [6] `int8` for the int8
// head (b70-serve's default). B70_KV_CACHE selects the KV form. Exits 77 (SKIP) with fewer
// than two GPUs or without peer access.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "cli/pipeline_serve_adapter.h"
#include "cli/serve_adapters.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"
#include "server/prefix_cache.h"

namespace {
using model::Qwen35;
using Ids = std::vector<uint32_t>;
using Bytes = std::vector<uint8_t>;
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 32;
bool g_split = false;
uint32_t g_k = 0;

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
Ids repeat_to(const Ids& src, size_t n) {
  Ids out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}

// What one side of the comparison needs of its engine besides the adapter.
struct Side {
  server::EngineIface& ad;
  std::function<void()> reset;
  std::function<std::vector<float>()> row;   // the last step's logits row (device 1's on two cards)
  std::function<Bytes()> session;            // save_state + save_kv(0, pos)
};

struct Result {
  std::vector<server::PrefixSession::Report> plans;
  Ids ids;
  std::vector<float> rows;
  Bytes session;
};

// One request: begin, n ids (step, or step_many's bursts kept to n as the server keeps them),
// the request-end snapshot. Every step's row is recorded when `rows` is given.
Ids turn(server::PrefixSession& s, Side& side, const Ids& prompt, uint32_t n, Result& r,
         std::vector<float>* rows) {
  r.plans.push_back(s.begin(prompt, g_split));
  Ids out;
  server::Sampling greedy;
  while (out.size() < n) {
    const Ids burst = g_k ? side.ad.step_many(greedy, g_k) : Ids{side.ad.step(greedy)};
    if (rows && !g_k) {
      const std::vector<float> row = side.row();
      rows->insert(rows->end(), row.begin(), row.end());
    }
    for (uint32_t id : burst) {
      if (out.size() == n) break;
      out.push_back(id);
      s.fed(id);
    }
    if (out.size() == n && g_k) side.ad.truncate_to(uint32_t(prompt.size() + n));
  }
  s.end();
  return out;
}

Result run_seq(Side& side, char n, const Ids& L, const Ids& C, server::HostAlloc& alloc, size_t bytes) {
  side.reset();
  Result r;
  server::PrefixSession s(side.ad, bytes, &alloc);
  Ids final_prompt;
  if (n == 'a' || n == 'b') {
    const Ids t1 = slice(L, 0, 3000);
    const Ids g1 = turn(s, side, t1, 64, r, nullptr);
    final_prompt = n == 'a' ? cat(cat(t1, g1), slice(L, 3000, 3700)) : cat(t1, slice(C, 0, 700));
  } else if (n == 'c' || n == 'd') {
    (void)turn(s, side, slice(L, 0, 6000), kGen, r, nullptr);
    final_prompt = cat(slice(L, 0, n == 'c' ? 5000 : 4096), slice(C, 0, 700));
  } else if (n == 'e') {
    const Ids t1 = slice(L, 0, 5000);
    const Ids g1 = turn(s, side, t1, 64, r, nullptr);
    (void)turn(s, side, slice(L, 20000, 20300), 16, r, nullptr);
    final_prompt = cat(cat(t1, g1), slice(L, 5000, 5700));
  } else {
    const Ids t1 = slice(L, 0, 3500);
    (void)turn(s, side, t1, kGen, r, nullptr);
    (void)turn(s, side, slice(L, 20000, 20300), 8, r, nullptr);
    final_prompt = t1;
  }
  r.ids = turn(s, side, final_prompt, kGen, r, &r.rows);
  r.session = side.session();
  return r;
}

template <class E>
Bytes session_of(E& e) {
  Bytes b(e.state_bytes());
  e.save_state(b.data());
  Bytes kv(e.kv_bytes(e.pos()));
  e.save_kv(0, e.pos(), kv.data());
  b.insert(b.end(), kv.begin(), kv.end());
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  g_k = argc > 3 ? uint32_t(std::atoi(argv[3])) : 0u;
  runtime::PpHandoff handoff = runtime::PpHandoff::Copy;
  if (argc > 4) CHECK(runtime::parse_pp_handoff(argv[4], handoff));
  g_split = argc > 5 && std::atoi(argv[5]) != 0;
  const loader::LmHeadForm head = argc > 6 && std::string(argv[6]) == "int8" ? loader::LmHeadForm::Int8
                                                                             : loader::LmHeadForm::Checkpoint;
  const runtime::KvCache kv = runtime::default_kv_cache();
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --pp 2 needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1)) {
    std::printf("SKIP: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  const Ids L = golden::read_ids(pdir + "/long32k.ids");
  const Ids C = repeat_to(golden::read_ids(pdir + "/code.ids"), 700);
  CHECK(L.size() >= 20300);
  const size_t cache_bytes = size_t(6) << 30;
  PinnedAlloc alloc(d0, cache_bytes);   // one context: both devices' copies reach it
  const char seqs[] = {'a', 'b', 'c', 'd', 'e', 'f'};
  std::printf("pp_prefix_gpu_test: mtp K %u, %s, split_last %d, %s head, kv %s\n", g_k,
              runtime::pp_handoff_name(handoff), int(g_split), loader::lm_head_form_name(head),
              runtime::kv_cache_name(kv));

  // --- one card, device 0 ----------------------------------------------------------------------
  std::vector<Result> ref;
  {
    runtime::Engine e(d0, loader::load(d0, snap, kMaxLen, g_k > 0, head), kMaxLen, false, kv);
    e.prepare_prefill();
    EngineAdapter ad(e, Qwen35::kVocabUsed, g_k);
    l0::CmdList imm = l0::CmdList::immediate(d0);
    Side side{ad, [&] { ad.reset(); },
              [&] {
                std::vector<float> r(Qwen35::kVocab);
                imm.copy(r.data(), e.buffers().logits.ptr(), r.size() * 4);
                return r;
              },
              [&] { return session_of(e); }};
    for (char n : seqs) ref.push_back(run_seq(side, n, L, C, alloc, cache_bytes));
  }

  // --- two cards ---------------------------------------------------------------------------------
  loader::LoadedModel full = loader::load(d0, snap, kMaxLen, g_k > 0, head);
  const runtime::PpWeights w = runtime::pp_weights(full);
  runtime::PpExtras x;
  x.mtp = g_k > 0;
  x.hook = true;
  const uint32_t split = runtime::pp_auto_split(*full.desc, w, kMaxLen, kv, runtime::PrefillPath{}, x);
  runtime::PipelineOptions opt;
  opt.handoff = handoff;
  runtime::PipelineEngine e(d0, d1, runtime::place_stages(d0, d1, std::move(full), split), kMaxLen, opt, kv);
  e.prepare_prefill();
  cli::pp::PipelineEngineAdapterT<runtime::PipelineEngine> ad(e, Qwen35::kVocabUsed, Qwen35::kVocab, g_k);
  Side side{ad, [&] { ad.reset(); }, [&] { return e.read_logits(); }, [&] { return session_of(e); }};
  bool ok = true;
  for (size_t i = 0; i < sizeof seqs; ++i) {
    const Result got = run_seq(side, seqs[i], L, C, alloc, cache_bytes);
    const Result& want = ref[i];
    bool same = got.plans.size() == want.plans.size();
    for (size_t t = 0; same && t < got.plans.size(); ++t)
      same = got.plans[t].kind == want.plans[t].kind && got.plans[t].restart == want.plans[t].restart;
    const bool ids = got.ids == want.ids;
    const bool rows = got.rows.size() == want.rows.size() &&
                      std::memcmp(got.rows.data(), want.rows.data(), got.rows.size() * 4) == 0;
    const bool sess = got.session == want.session;
    std::printf("  %c: final turn %s at %u; plans %s, %zu ids %s, %zu rows %s, session (%zu B) %s\n",
                seqs[i], server::plan_kind_name(got.plans.back().kind), got.plans.back().restart,
                same ? "equal" : "DIFFER", got.ids.size(), ids ? "bitwise" : "DIFFER",
                got.rows.size() / Qwen35::kVocab, rows ? "bitwise" : "DIFFER", got.session.size(),
                sess ? "bitwise" : "DIFFER");
    ok = ok && same && ids && rows && sess;
  }
  if (!ok) {
    std::fprintf(stderr, "pp_prefix_gpu_test FAILED (split %u)\n", split);
    return 1;
  }
  std::printf("pp_prefix_gpu_test OK: sequences a-f through b70-serve's adapters, two cards (split %u) "
              "bitwise one card\n", split);
  return 0;
}

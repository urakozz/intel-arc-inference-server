// Spec 7 C2 (plan 7c Task 3): cached against cold on the card. The request path
// (server::PrefixSession over EngineAdapter over the real runtime::Engine, no HTTP) runs a
// multi-turn sequence; its last turn's first 32 greedy tokens and their logits rows are
// judged against a cold run of the same final prompt (reset() + one full prefill), under
// the golden gate's tie-aware rule with the COLD run as the reference: a row whose
// reference top-1 is unique must match exactly, except for the l0-int8 near-tie allowance
// (the reference's runner-up with a top-2 margin <= 1 bf16 ulp, `prefill_gate_test`'s
// argv[7] rule, 1 on l0-int8). The cold run is teacher-forced to the cached run's tokens,
// so every row is conditioned on the same history. Not bitwise: the tail's chunks begin at
// the restart point, not at 0.
//
// Sequences (L = long32k.ids, C = the golden code prompt's ids repeated):
//   a  turn 2 = turn 1's prompt L[0:3000] + its 64 generated ids + L[3000:3700]: continue
//   b  turn 2 = L[0:3000] + C[0:700] (turn 1's generated ids dropped): restore at 3000
//   c  turn 1 L[0:6000]; turn 2 = L[0:5000] + C[0:700]: mid-block, restore at 4096
//   d  turn 1 L[0:6000]; turn 2 = L[0:4096] + C[0:700]: on the block boundary, 4096
//   e  turn 1 L[0:5000] + 64 generated, a 300-id side request L[20000:20300], turn 2 =
//      turn 1 + generated + L[5000:5700]: restore at 5064 with every KV block from the host
//   f  turn 1 L[0:3500], a side request, turn 2 = turn 1's prompt again: restore at 3500
//      from the prompt-end snapshot with its first id, no prefill -- the 32 tokens must be
//      turn 1's own, bitwise (spec 7 §8 amendment).
//   g  the control for b, no cache: b's final prompt prefilled as two calls split at 3000,
//      against one call -- what the chunk boundary alone costs.
// argv: [1] snapshot, [2] prompts dir, [3] backend l0 | l0-int8 (default: the build's),
//       [4] near-tie flips allowed per sequence (default 1 on l0-int8, else 0).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "cli/serve_adapters.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
#include "runtime/prefill/backend.h"
#include "server/prefix_cache.h"

namespace {
using model::Qwen35;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 32;

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

// prefill_gate_test's argv[7] rule, verbatim.
bool bf16_near_tie(const float* row, uint32_t used, uint32_t top1, uint32_t* second) {
  uint32_t s2 = top1 == 0 ? 1u : 0u;
  for (uint32_t i = 0; i < used; ++i)
    if (i != top1 && row[i] > row[s2]) s2 = i;
  *second = s2;
  int e = 0;
  std::frexp(std::fabs(row[top1]), &e);
  const double ulp = std::ldexp(1.0, e - 8);
  return double(row[top1]) - double(row[s2]) <= ulp;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b, uint32_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (uint32_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb);
}

struct Rig {
  runtime::Engine& eng;
  EngineAdapter& ad;
  PinnedAlloc& alloc;
  l0::CmdList imm;
  std::vector<float> row(bool from_prefill) {
    std::vector<float> r(Qwen35::kVocab);
    const l0::Mem& src = from_prefill ? eng.prefill_scratch()->logits : eng.buffers().logits;
    imm.copy(r.data(), src.ptr(), size_t(Qwen35::kVocab) * 4);
    return r;
  }
};

// One request through the cache: begin, n greedy steps, the request-end snapshot.
Ids turn(server::PrefixSession& s, EngineAdapter& ad, const Ids& prompt, uint32_t n,
         server::PrefixSession::Report* rep = nullptr) {
  const auto r = s.begin(prompt);
  if (rep) *rep = r;
  std::printf("    turn of %zu ids: %s at %u, kv %.1f MB, restore %.1f ms, prefill %.1f ms,"
              " store %.1f ms\n", prompt.size(), server::plan_kind_name(r.kind), r.restart,
              r.kv_bytes / 1e6, r.restore_ms, r.prefill_ms, r.store_ms);
  Ids out;
  server::Sampling greedy;
  for (uint32_t i = 0; i < n; ++i) {
    out.push_back(ad.step(greedy));
    s.fed(out.back());
  }
  s.end();
  return out;
}

struct Seq {
  std::string name;
  server::PrefixCache::Plan::Kind kind;
  uint32_t restart;
};

bool run_seq(Rig& rig, const Seq& want, const Ids& L, const Ids& C, uint32_t allowed) {
  std::printf("\n== sequence %s ==\n", want.name.c_str());
  rig.eng.reset();
  Ids final_prompt, cand_tok, turn1_gen;
  std::vector<std::vector<float>> cand_rows;
  server::PrefixSession::Report rep;
  if (want.name[0] == 'g') {
    // The control for b: no cache, the same final prompt prefilled as two calls split at
    // 3000 -- the chunking difference alone, judged the same way.
    final_prompt = cat(slice(L, 0, 3000), slice(C, 0, 700));
    rig.eng.prefill(slice(final_prompt, 0, 3000));
    rig.eng.prefill(slice(final_prompt, 3000, final_prompt.size()));
    rep.kind = want.kind;
    rep.restart = want.restart;
    server::Sampling greedy;
    for (uint32_t p = 0; p < kGen; ++p) {
      cand_rows.push_back(rig.row(p == 0));
      cand_tok.push_back(rig.ad.step(greedy));
    }
  } else {
    server::PrefixSession s(rig.ad, rig.alloc.mem.size(), &rig.alloc);
    const char n = want.name[0];
    if (n == 'a' || n == 'b') {
      const Ids t1 = slice(L, 0, 3000);
      const Ids g1 = turn(s, rig.ad, t1, 64);
      final_prompt = n == 'a' ? cat(cat(t1, g1), slice(L, 3000, 3700)) : cat(t1, slice(C, 0, 700));
    } else if (n == 'c' || n == 'd') {
      (void)turn(s, rig.ad, slice(L, 0, 6000), 32);
      final_prompt = cat(slice(L, 0, n == 'c' ? 5000 : 4096), slice(C, 0, 700));
    } else if (n == 'e') {
      const Ids t1 = slice(L, 0, 5000);
      const Ids g1 = turn(s, rig.ad, t1, 64);
      (void)turn(s, rig.ad, slice(L, 20000, 20300), 16);
      final_prompt = cat(cat(t1, g1), slice(L, 5000, 5700));
    } else {
      const Ids t1 = slice(L, 0, 3500);
      turn1_gen = turn(s, rig.ad, t1, kGen);
      (void)turn(s, rig.ad, slice(L, 20000, 20300), 8);
      final_prompt = t1;
    }
    rep = s.begin(final_prompt);
    std::printf("    final turn of %zu ids: %s at %u, kv %.1f MB, restore %.1f ms, prefill %.1f"
                " ms, store %.1f ms\n", final_prompt.size(), server::plan_kind_name(rep.kind),
                rep.restart, rep.kv_bytes / 1e6, rep.restore_ms, rep.prefill_ms, rep.store_ms);
    const bool prefilled = rep.restart < final_prompt.size();
    server::Sampling greedy;
    for (uint32_t p = 0; p < kGen; ++p) {
      if (prefilled) cand_rows.push_back(rig.row(p == 0));
      cand_tok.push_back(rig.ad.step(greedy));
      s.fed(cand_tok.back());
    }
  }   // the session (and its block hook) ends here; the cold run has no hook
  bool ok = rep.kind == want.kind && rep.restart == want.restart;
  if (!ok)
    std::printf("  ** plan %s at %u, expected %s at %u **\n", server::plan_kind_name(rep.kind),
                rep.restart, server::plan_kind_name(want.kind), want.restart);

  if (want.name[0] == 'f') {
    uint32_t same = 0;
    for (uint32_t p = 0; p < kGen; ++p) same += cand_tok[p] == turn1_gen[p];
    std::printf("  %s: %u/%u tokens bitwise equal to turn 1's own%s\n", want.name.c_str(), same,
                kGen, same == kGen ? "" : "  ** FAIL **");
    return ok && same == kGen;
  }
  if (cand_rows.empty()) return false;

  // The cold reference, teacher-forced to the cached run's tokens.
  rig.eng.reset();
  rig.eng.prefill(final_prompt);
  uint32_t det_exact = 0, near = 0, bad = 0, first_diff = kGen, worst = 0;
  double min_cos = 1.0;
  for (uint32_t p = 0; p < kGen; ++p) {
    const std::vector<float> ref = rig.row(p == 0);
    const double c = cosine(cand_rows[p], ref, Qwen35::kVocabUsed);
    if (c < min_cos) {
      min_cos = c;
      worst = p;
    }
    const golden::GoldenDecision gd =
        golden::golden_decision(ref.data(), Qwen35::kVocab, Qwen35::kVocabUsed);
    const uint32_t id = cand_tok[p];
    if (id != gd.set[0] && first_diff == kGen) first_diff = p;
    if (gd.determined()) {
      uint32_t second = 0;
      if (id == gd.set[0]) {
        ++det_exact;
      } else if (near < allowed && bf16_near_tie(ref.data(), Qwen35::kVocabUsed, gd.set[0],
                                                   &second) && id == second) {
        ++near;
      } else {
        ++bad;
      }
    } else if (!gd.contains(id)) {
      ++bad;
    }
    if (id != gd.set[0] || p == 0)
      std::printf("    row %2u: cached %6u cold %6u %s cos %.9f%s\n", p, id, gd.set[0],
                  gd.determined() ? "det" : "TIE", c, id == gd.set[0] ? "" : "  <== differs");
    rig.eng.ingest({id});
  }
  std::printf("  %s: %s at %u of %zu | %u/%u exact, %u near-tie (allowed %u), %u bad, first"
              " differing row %s, logits min cos %.9f (row %u)%s\n", want.name.c_str(),
              server::plan_kind_name(rep.kind), rep.restart, final_prompt.size(), det_exact, kGen,
              near, allowed, bad, first_diff == kGen ? "none" : std::to_string(first_diff).c_str(),
              min_cos, worst, bad == 0 && ok ? "" : "  ** FAIL **");
  return ok && bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  runtime::PrefillBackend backend = runtime::prefill::default_prefill_backend();
  if (argc > 3) CHECK(runtime::parse_prefill_backend(argv[3], backend));
  const uint32_t allowed = argc > 4 ? uint32_t(std::atoi(argv[4]))
                                    : (backend == runtime::PrefillBackend::L0Int8 ? 1u : 0u);
  const Ids L = golden::read_ids(pdir + "/long32k.ids");
  const Ids C = repeat_to(golden::read_ids(pdir + "/code.ids"), 700);
  CHECK(L.size() >= 20300);

  l0::Context ctx(0);
  runtime::Engine eng(ctx, loader::load(ctx, snap, kMaxLen), kMaxLen);
  eng.set_prefill_backend(backend);
  eng.prepare_prefill();
  std::printf("prefix_gpu_test: backend %s, near-tie allowance %u\n",
              runtime::prefill_backend_name(eng.prefill_backend()), allowed);
  EngineAdapter ad(eng, Qwen35::kVocabUsed);
  PinnedAlloc alloc(ctx, size_t(6) << 30);
  Rig rig{eng, ad, alloc, l0::CmdList::immediate(ctx)};

  using K = server::PrefixCache::Plan;
  const std::vector<Seq> seqs = {{"a continue", K::Continue, 3064},
                                 {"b prompt end", K::Restore, 3000},
                                 {"c mid-block", K::Restore, 4096},
                                 {"d block boundary", K::Restore, 4096},
                                 {"e side request", K::Restore, 5064},
                                 {"f same prompt", K::Restore, 3500},
                                 {"g control for b: no cache, prefill split at 3000", K::Cold, 0}};
  bool all = true;
  for (const Seq& s : seqs) all = run_seq(rig, s, L, C, allowed) && all;
  if (!all) {
    std::fprintf(stderr, "prefix_gpu_test FAILED\n");
    return 1;
  }
  std::printf("\nprefix_gpu_test OK: sequences a-f, cached against cold under the tie-aware"
              " rule, same-prompt bitwise\n");
  return 0;
}

// Spec 18d Review Focus 3 on the card: K2-Horizon's KV-only snapshots (spec 7 C1 / C2). Box only
// (labels `checkpoint;k2`), SKIP (77) without the int4 checkpoint. Written blind on the Mac.
//
//   0. the snapshot calls on the device: a prompt of 4100 ids prefilled; save_kv(0, 4100) to
//      pinned host memory, reset() (the cache zeroed), load_kv, save again - the two host copies
//      byte-identical (every layer's K / V rows, at int8 their scales). Then C1, bitwise: restore
//      [0, 4096) and pos 4096 (load_state - K2 has no state bytes), prefill [4096, 4100), 32 greedy
//      ids and their logits rows against a cold prefill of the 4100 ids - the chunks are the same
//      ([0, 2048), [2048, 4096), [4096, 4100)) and K2's prefill is keyed to absolute positions
//      (spec 18 §11), so the two are bitwise.
//   1. C2: prefix_gpu_test's sequences a-g through server::PrefixSession over
//      cli::k2::K2EngineAdapterT<K2Engine> (b70-serve's adapter), the cached turn's 32 ids and
//      logits rows judged against a cold run under the tie-aware rule with the COLD run as the
//      reference (teacher-forced), and the rows that are bitwise counted. Restores at a block
//      end (c, d, f) start the tail's chunks where the cold run's start: those rows are expected
//      bitwise; b, e restore at a prompt / request end (a split the argument also predicts
//      bitwise - prefill_split_k2_test is its gate; a difference there is read, not failed).
//
// The prompts are tests/golden/prompts' (Qwen3.8's ids, all < 250,624, legal K2 ids - the gate is
// about the mechanics, not the text). argv: [1] snapshot, [2] prompts dir, [3] near-tie flips
// allowed per sequence (default 0), [4] split: 1 = --prefix-split-last's plans. The KV form is
// B70_KV_CACHE's (the `_kv8` twin sets int8).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "cli/k2_serve_adapter.h"
#include "cli/serve_adapters.h"   // PinnedAlloc
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/k2_loader.h"
#include "loader/snapshot.h"
#include "runtime/k2/k2_engine.h"
#include "server/prefix_cache.h"

namespace {
using Ids = std::vector<uint32_t>;
using Adapter = cli::k2::K2EngineAdapterT<runtime::k2::K2Engine>;
constexpr uint32_t kMaxLen = 16384, kGen = 32;
bool g_split = false;

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

// prefix_gpu_test's near-tie rule (prefill_gate_test's argv[7]), verbatim.
bool bf16_near_tie(const float* row, uint32_t used, uint32_t top1, uint32_t* second) {
  uint32_t s2 = top1 == 0 ? 1u : 0u;
  for (uint32_t i = 0; i < used; ++i)
    if (i != top1 && row[i] > row[s2]) s2 = i;
  *second = s2;
  int e = 0;
  std::frexp(std::fabs(row[top1]), &e);
  return double(row[top1]) - double(row[s2]) <= std::ldexp(1.0, e - 8);
}

Ids turn(server::PrefixSession& s, Adapter& ad, const Ids& prompt, uint32_t n,
         server::PrefixSession::Report* rep = nullptr) {
  const auto r = s.begin(prompt, g_split);
  if (rep) *rep = r;
  std::printf("    turn of %zu ids: %s at %u, kv %.1f MB, restore %.1f ms, prefill %.1f ms, store %.1f ms\n",
              prompt.size(), server::plan_kind_name(r.kind), r.restart, r.kv_bytes / 1e6, r.restore_ms,
              r.prefill_ms, r.store_ms);
  Ids out;
  server::Sampling greedy;
  for (uint32_t i = 0; i < n; ++i) {
    out.push_back(ad.step(greedy));
    s.fed(out.back());
  }
  s.end();
  return out;
}

// ---- 0. the device calls and C1 --------------------------------------------------------------
bool part0(l0::Context& ctx, runtime::k2::K2Engine& eng, const Ids& L) {
  const Ids p = slice(L, 0, 4100);
  const size_t bytes = eng.kv_bytes(4100);
  l0::Mem a(ctx, l0::MemKind::Host, bytes), b(ctx, l0::MemKind::Host, bytes);
  eng.reset();
  eng.prefill(p);
  eng.save_kv(0, 4100, a.ptr());
  eng.reset();
  eng.load_kv(0, 4100, a.ptr());
  eng.save_kv(0, 4100, b.ptr());
  const bool same = std::memcmp(a.ptr(), b.ptr(), bytes) == 0;
  std::printf("0. save_kv / reset / load_kv / save_kv of 4100 positions (%.1f MB, %s KV): %s\n", bytes / 1e6,
              runtime::kv_cache_name(eng.kv_cache()), same ? "byte-identical" : "DIFFER  ** FAIL **");

  // C1: cold, then restored at the block end 4096.
  eng.reset();
  eng.prefill(p);
  std::vector<std::vector<float>> cold_rows;
  Ids cold;
  for (uint32_t i = 0; i < kGen; ++i) {
    cold_rows.push_back(eng.read_logits());
    cold.push_back(eng.generate(1)[0]);
  }
  // A snapshot's host layout is per range (layer-major), so [0, 4096) is saved as its own range
  // - what the store's block entries hold - not cut out of a's 4100.
  eng.reset();
  eng.prefill(slice(p, 0, 4096));
  l0::Mem c(ctx, l0::MemKind::Host, eng.kv_bytes(4096));
  eng.save_kv(0, 4096, c.ptr());
  eng.reset();
  eng.load_kv(0, 4096, c.ptr());
  eng.load_state(nullptr, 4096);
  eng.prefill(slice(p, 4096, 4100));
  uint32_t ids_same = 0, rows_same = 0;
  for (uint32_t i = 0; i < kGen; ++i) {
    rows_same += eng.read_logits() == cold_rows[i];
    ids_same += eng.generate(1)[0] == cold[i];
  }
  const bool c1 = ids_same == kGen && rows_same == kGen;
  std::printf("   C1: restored at 4096 + prefill [4096, 4100): %u/%u ids, %u/%u logits rows bitwise the cold "
              "run's%s\n", ids_same, kGen, rows_same, kGen, c1 ? "" : "  ** FAIL **");
  return same && c1;
}

// ---- 1. C2 -------------------------------------------------------------------------------------
struct Seq {
  std::string name;
  server::PrefixCache::Plan::Kind kind;
  uint32_t restart;
};

bool run_seq(runtime::k2::K2Engine& eng, Adapter& ad, PinnedAlloc& alloc, const Seq& want, const Ids& L,
             const Ids& C, uint32_t allowed) {
  std::printf("\n== sequence %s ==\n", want.name.c_str());
  eng.reset();
  Ids final_prompt, cand_tok, turn1_gen;
  std::vector<std::vector<float>> cand_rows;
  server::PrefixSession::Report rep;
  server::Sampling greedy;
  if (want.name[0] == 'g') {
    final_prompt = cat(slice(L, 0, 3000), slice(C, 0, 700));
    if (g_split) {
      eng.prefill(slice(final_prompt, 0, 2999));
      eng.prefill(slice(final_prompt, 2999, final_prompt.size() - 1));
      eng.ingest({final_prompt.back()});
    } else {
      eng.prefill(slice(final_prompt, 0, 3000));
      eng.prefill(slice(final_prompt, 3000, final_prompt.size()));
    }
    rep.kind = want.kind;
    rep.restart = want.restart;
    for (uint32_t p = 0; p < kGen; ++p) {
      cand_rows.push_back(eng.read_logits());
      cand_tok.push_back(ad.step(greedy));
    }
  } else {
    server::PrefixSession s(ad, alloc.mem.size(), &alloc);
    const char n = want.name[0];
    if (n == 'a' || n == 'b') {
      const Ids t1 = slice(L, 0, 3000);
      const Ids g1 = turn(s, ad, t1, 64);
      final_prompt = n == 'a' ? cat(cat(t1, g1), slice(L, 3000, 3700)) : cat(t1, slice(C, 0, 700));
    } else if (n == 'c' || n == 'd') {
      (void)turn(s, ad, slice(L, 0, 6000), 32);
      final_prompt = cat(slice(L, 0, n == 'c' ? 5000 : 4096), slice(C, 0, 700));
    } else if (n == 'e') {
      const Ids t1 = slice(L, 0, 5000);
      const Ids g1 = turn(s, ad, t1, 64);
      (void)turn(s, ad, slice(L, 20000, 20300), 16);
      final_prompt = cat(cat(t1, g1), slice(L, 5000, 5700));
    } else {
      const Ids t1 = slice(L, 0, 3500);
      turn1_gen = turn(s, ad, t1, kGen);
      (void)turn(s, ad, slice(L, 20000, 20300), 8);
      final_prompt = t1;
    }
    rep = s.begin(final_prompt, g_split);
    std::printf("    final turn of %zu ids: %s at %u, kv %.1f MB, restore %.1f ms, prefill %.1f ms, store %.1f ms\n",
                final_prompt.size(), server::plan_kind_name(rep.kind), rep.restart, rep.kv_bytes / 1e6,
                rep.restore_ms, rep.prefill_ms, rep.store_ms);
    for (uint32_t p = 0; p < kGen; ++p) {
      cand_rows.push_back(eng.read_logits());
      cand_tok.push_back(ad.step(greedy));
      s.fed(cand_tok.back());
    }
  }
  bool ok = rep.kind == want.kind && rep.restart == want.restart;
  if (!ok)
    std::printf("  ** plan %s at %u, expected %s at %u **\n", server::plan_kind_name(rep.kind), rep.restart,
                server::plan_kind_name(want.kind), want.restart);
  if (want.name[0] == 'f') {
    uint32_t same = 0;
    for (uint32_t p = 0; p < kGen; ++p) same += cand_tok[p] == turn1_gen[p];
    std::printf("  %s: %u/%u ids bitwise turn 1's own%s\n", want.name.c_str(), same, kGen,
                same == kGen ? "" : "  ** FAIL **");
    return ok && same == kGen;
  }

  // The cold reference, teacher-forced to the cached run's ids.
  eng.reset();
  eng.prefill(final_prompt);
  const uint32_t V = eng.vocab();
  uint32_t det_exact = 0, near = 0, bad = 0, bitwise = 0, first_diff = kGen;
  for (uint32_t p = 0; p < kGen; ++p) {
    const std::vector<float> ref = eng.read_logits();
    bitwise += ref == cand_rows[p];
    const golden::GoldenDecision gd = golden::golden_decision(ref.data(), V, V);
    const uint32_t id = cand_tok[p];
    if (id != gd.set[0] && first_diff == kGen) first_diff = p;
    if (gd.determined()) {
      uint32_t second = 0;
      if (id == gd.set[0])
        ++det_exact;
      else if (near < allowed && bf16_near_tie(ref.data(), V, gd.set[0], &second) && id == second)
        ++near;
      else
        ++bad;
    } else if (!gd.contains(id)) {
      ++bad;
    }
    eng.ingest({id});
  }
  std::printf("  %s: %s at %u of %zu | %u/%u exact, %u near-tie (allowed %u), %u bad, first differing row %s, "
              "%u/%u logits rows bitwise the cold run's%s\n",
              want.name.c_str(), server::plan_kind_name(rep.kind), rep.restart, final_prompt.size(), det_exact,
              kGen, near, allowed, bad, first_diff == kGen ? "none" : std::to_string(first_diff).c_str(), bitwise,
              kGen, bad == 0 && ok ? "" : "  ** FAIL **");
  return ok && bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  const uint32_t allowed = argc > 3 ? uint32_t(std::atoi(argv[3])) : 0u;
  g_split = argc > 4 && std::atoi(argv[4]) != 0;
  try {
    (void)loader::resolve_snapshot(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no K2-Horizon checkpoint at '%s' (%s)\n", snap.c_str(), e.what());
    return 77;
  }
  const Ids L = golden::read_ids(pdir + "/long32k.ids");
  const Ids C = repeat_to(golden::read_ids(pdir + "/code.ids"), 700);
  CHECK(L.size() >= 20300);

  l0::Context ctx(0);
  runtime::k2::K2Engine eng(ctx, loader::load_k2(ctx, snap, kMaxLen, loader::LmHeadForm::Int8), kMaxLen);
  eng.prepare_prefill();
  std::printf("k2_prefix_gpu_test: %s KV, int8 head, attention %s, near-tie allowance %u, split_last %d, "
              "a snapshot position %zu B\n", runtime::kv_cache_name(eng.kv_cache()),
              runtime::k2::k2_attn_name(runtime::k2::k2_attn()), allowed, int(g_split), eng.kv_bytes(1));
  bool all = part0(ctx, eng, L);

  Adapter ad(eng, eng.vocab());
  CHECK_EQ(ad.state_bytes(), size_t(0));
  // 8 GiB: e's turn 1 stores 2 blocks + snapshots (~0.4 GB a block at bf16) besides the rest.
  PinnedAlloc alloc(ctx, size_t(8) << 30);
  using K = server::PrefixCache::Plan;
  const std::vector<Seq> seqs = {{"a continue", K::Continue, 3064},
                                 {"b prompt end", K::Restore, g_split ? 2999u : 3000u},
                                 {"c mid-block", K::Restore, 4096},
                                 {"d block boundary", K::Restore, 4096},
                                 {"e side request", K::Restore, 5064},
                                 {"f same prompt", K::Restore, g_split ? 3499u : 2048u},
                                 {"g control for b: its split, no cache", K::Cold, 0}};
  for (const Seq& s : seqs) all = run_seq(eng, ad, alloc, s, L, C, allowed) && all;
  if (!all) {
    std::fprintf(stderr, "k2_prefix_gpu_test FAILED\n");
    return 1;
  }
  std::printf("\nk2_prefix_gpu_test OK: the snapshot calls byte-exact, C1 bitwise at a block end, sequences "
              "a-g cached against cold under the tie-aware rule, same-prompt bitwise\n");
  return 0;
}

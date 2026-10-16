// Spec 19d Task 1 on the card (spec 19 decision 5; 19c's deferred minor 8): the DFlash drafter's ring across a
// prefix-cache restore, through the request path b70-serve runs - server::PrefixSession over EngineAdapter
// (--spec dflash, the int8 drafter beside the gate checkpoint's int8 head, bf16 KV), the server's greedy loop at a
// fixed K = 7 (step_many, the last burst truncated). Turns: A = long32k[0:3000], B = the golden code prompt
// repeated, a side request between turns that overwrites the engine's state and ring.
//
//   K    the drafter names the store: d.key() != 0 and the session's chain root is server::store_root(0, d.key());
//        the snapshot is the plain engine's state plus the ring (state_bytes, 41,943,040 B more).
//   R1   restore-and-continue: turn 1 = A (96 ids), the side request, turn 2 = A + turn 1's ids + B[0:700] - restored
//        from turn 1's request-end snapshot (pos 3096) - against the uninterrupted session (turn 1 then turn 2 by
//        Continue, no side request): after turn 2's tail the drafter's ring is BITWISE the uninterrupted one's, and
//        the next 128 ids AND every iteration's accepted count are identical (the same state, ring and KV give the
//        same drafts). Control (printed, not gated): the same restore with the ring zeroed after load_state (19c's
//        behaviour, a cold ring) - its ids still identical (lossless), its acceptance lower.
//   R2   a restore at a block end followed by a tail at least the window long: turn 1 = long32k[0:5000] (block-end
//        snapshots at 2048 and 4096), the side request, turn 2 = long32k[0:4096] + B[0:2500] restored at 4096 -
//        its ring BITWISE a cold run's of the same prompt (the same chunks from 4096 on; the tail's window covers
//        every slot), the next 64 ids and acceptances identical. The mid-call caveat, recorded: with a 300-id tail
//        instead, the count of window rows that differ from a cold run's (positions turn 1's call never projected -
//        spec 19 §12).
//
// usage: dflash_prefix_gpu_test <target snapshot> <drafter>   (cwd = the source tree). SKIP 77 without either.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "cli/serve_adapters.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/engine.h"
#include "server/prefix_cache.h"

namespace {
using Ids = std::vector<uint32_t>;
using golden::read_ids;
constexpr uint32_t kMaxLen = 16384, kK = 7;

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

// 19c's restore, for the control arm: the ring zeroed right after the snapshot's state is loaded (a cold drafter).
struct ColdRingAdapter : EngineAdapter {
  using EngineAdapter::EngineAdapter;
  void load_state(const void* host, uint32_t p) override {
    EngineAdapter::load_state(host, p);
    eng.tap_drafter()->reset(imm);
  }
};

struct Gen {
  Ids ids;
  std::vector<uint32_t> accepted;   // per iteration
  server::PrefixSession::Report rep;
};

// One request: begin, then the server's speculative loop at K = 7 for `n` ids (fed, the last burst truncated), end.
// `after_begin` runs between the prefill and the first decode step (the ring is read there).
Gen turn(server::PrefixSession& s, EngineAdapter& ad, const Ids& prompt, uint32_t n,
         const std::function<void()>& after_begin = {}) {
  Gen g;
  g.rep = s.begin(prompt);
  std::printf("    turn of %zu ids: %s at %u, restore %.1f ms, prefill %.1f ms, store %.1f ms\n", prompt.size(),
              server::plan_kind_name(g.rep.kind), g.rep.restart, g.rep.restore_ms, g.rep.prefill_ms,
              g.rep.store_ms);
  if (after_begin) after_begin();
  server::Sampling greedy;
  while (g.ids.size() < n) {
    const Ids burst = ad.step_many(greedy, kK);
    g.accepted.push_back(uint32_t(burst.size() - 1));
    const size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    for (size_t i = 0; i < keep; ++i) {
      g.ids.push_back(burst[i]);
      s.fed(burst[i]);
    }
    if (keep < burst.size()) ad.truncate_to(ad.pos() - uint32_t(burst.size() - keep));
  }
  s.end();
  return g;
}

// The drafter's whole ring (ring_k then ring_v), as save_ring lays it out.
std::vector<uint8_t> ring(l0::CmdList& imm, runtime::dflash::DFlashDrafter& d) {
  std::vector<uint8_t> out(d.ring_bytes());
  d.save_ring(imm, out.data());
  return out;
}
// Window rows (positions [end - W, end)) whose bytes differ between two rings.
uint32_t rows_differ(const runtime::dflash::DFlashDrafter& d, const std::vector<uint8_t>& a,
                     const std::vector<uint8_t>& b, uint32_t end) {
  const model::DFlashDesc& dd = d.desc();
  const size_t row = size_t(dd.kv_n()) * 2, slots = dd.window[0], half = a.size() / 2;
  uint32_t n = 0;
  for (uint32_t p = end - d.window(); p < end; ++p) {
    bool same = true;
    for (uint32_t l = 0; l < dd.layers && same; ++l)
      for (size_t kv : {size_t(0), half}) {
        const size_t at = kv + (size_t(l) * slots + p % slots) * row;
        same &= std::equal(a.begin() + at, a.begin() + at + row, b.begin() + at);
      }
    n += !same;
  }
  return n;
}

double acceptance(const Gen& g) {
  uint64_t a = 0;
  for (uint32_t x : g.accepted) a += x;
  return g.accepted.empty() ? 0.0 : double(a) / double(g.accepted.size() * kK);
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap, dsnap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
    dsnap = loader::resolve_snapshot(argc > 2 ? argv[2] : "z-lab/Qwen3.8-27B-DFlash2");
  } catch (const std::exception& ex) {
    std::printf("SKIP: %s\n", ex.what());
    return 77;
  }
  const Ids L = read_ids("tests/golden/prompts/long32k.ids");
  const Ids C = repeat_to(read_ids("tests/golden/prompts/code.ids"), 4000);
  CHECK(L.size() >= 20500);
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, loader::LmHeadForm::Int8);
  loader::DFlashWeights w = loader::load_drafter(ctx, dsnap, model, loader::DraftFormat::Int8);
  const uint32_t head_rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  runtime::dflash::DFlashDrafter d(ctx, std::move(w), head_rows, kMaxLen, kK);
  runtime::Engine e(ctx, std::move(model), kMaxLen, /*debug_resid=*/false, runtime::KvCache::Bf16, &d);
  d.attach(e, false);
  e.prepare_prefill();
  EngineAdapter ad(e, model::Qwen35::kVocabUsed, kK, /*lookup=*/false, /*dflash=*/true);
  ColdRingAdapter cold_ad(e, model::Qwen35::kVocabUsed, kK, false, true);
  PinnedAlloc alloc(ctx, size_t(6) << 30);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  bool ok = true;

  // --- K ------------------------------------------------------------------------------------------------------
  {
    const size_t gdn = e.buffers().gdn_state.size(), conv = e.buffers().conv_ring.size();
    bool k_ok = d.key() != 0 && d.ring_bytes() == 41943040 && e.state_bytes() == gdn + conv + d.ring_bytes();
    server::PrefixSession s(ad, alloc.mem.size(), &alloc);
    k_ok &= s.cache()->kv_form() == server::store_root(0, d.key()) && s.cache()->kv_form() != 0;
    std::printf("K: drafter key %016llx, state %zu B (+ the ring %zu B), the store's root = store_root(bf16, key): "
                "%s\n",
                (unsigned long long)d.key(), e.state_bytes(), d.ring_bytes(), k_ok ? "ok" : "WRONG");
    ok &= k_ok;
  }

  // --- R1: restore-and-continue ----------------------------------------------------------------------------------
  {
    const Ids A = slice(L, 0, 3000), B = slice(C, 0, 700), side = slice(L, 20000, 20500);
    Gen ref1, ref2, got1, got2, cold2;
    std::vector<uint8_t> ring_ref, ring_got;
    std::printf("R1 uninterrupted:\n");
    {
      server::PrefixSession s(ad, alloc.mem.size(), &alloc);
      ref1 = turn(s, ad, A, 96);
      ref2 = turn(s, ad, cat(cat(A, ref1.ids), B), 128, [&] { ring_ref = ring(imm, d); });
      ok &= ref2.rep.kind == server::PrefixCache::Plan::Continue;
    }
    std::printf("R1 restored:\n");
    {
      server::PrefixSession s(ad, alloc.mem.size(), &alloc);
      got1 = turn(s, ad, A, 96);
      (void)turn(s, ad, side, 16);
      got2 = turn(s, ad, cat(cat(A, got1.ids), B), 128, [&] { ring_got = ring(imm, d); });
      ok &= got2.rep.kind == server::PrefixCache::Plan::Restore && got2.rep.restart == 3096;
    }
    std::printf("R1 cold ring (19c's restore):\n");
    {
      server::PrefixSession s(cold_ad, alloc.mem.size(), &alloc);
      (void)turn(s, cold_ad, A, 96);
      (void)turn(s, cold_ad, side, 16);
      cold2 = turn(s, cold_ad, cat(cat(A, got1.ids), B), 128);
    }
    const bool same1 = got1.ids == ref1.ids, ring_same = ring_got == ring_ref;
    const bool ids_same = got2.ids == ref2.ids, acc_same = got2.accepted == ref2.accepted;
    std::printf("R1: turn 1 identical %s; the ring after the restored tail %s the uninterrupted one's (%u window rows "
                "differ); 128 ids %s, per-iteration acceptance %s (%.3f vs %.3f); the cold-ring control: ids %s, "
                "acceptance %.3f\n",
                same1 ? "yes" : "NO", ring_same ? "equals (bitwise)" : "DIFFERS from",
                rows_differ(d, ring_got, ring_ref, 3796), ids_same ? "identical" : "DIFFER",
                acc_same ? "identical" : "DIFFERS", acceptance(got2), acceptance(ref2),
                cold2.ids == ref2.ids ? "identical" : "DIFFER", acceptance(cold2));
    ok &= same1 && ring_same && ids_same && acc_same && cold2.ids == ref2.ids;
  }

  // --- R2: a restore at a block end ----------------------------------------------------------------------------
  {
    const Ids side = slice(L, 20000, 20500);
    const auto arm = [&](uint32_t tail, std::vector<uint8_t>* cold_ring, std::vector<uint8_t>* got_ring, Gen* cold,
                         Gen* got) {
      const Ids t2 = cat(slice(L, 0, 4096), slice(C, 0, tail));
      {   // the cold run of the same prompt (its own store: Cold, reset, the hooked chunks from 0)
        server::PrefixSession s(ad, alloc.mem.size(), &alloc);
        *cold = turn(s, ad, t2, 64, [&] { *cold_ring = ring(imm, d); });
        ok &= cold->rep.kind == server::PrefixCache::Plan::Cold;
      }
      server::PrefixSession s(ad, alloc.mem.size(), &alloc);
      (void)turn(s, ad, slice(L, 0, 5000), 8);
      (void)turn(s, ad, side, 8);
      *got = turn(s, ad, t2, 64, [&] { *got_ring = ring(imm, d); });
      ok &= got->rep.kind == server::PrefixCache::Plan::Restore && got->rep.restart == 4096;
      return uint32_t(t2.size());
    };
    std::vector<uint8_t> cr, gr;
    Gen cg, gg;
    std::printf("R2 (tail 2500):\n");
    const uint32_t end = arm(2500, &cr, &gr, &cg, &gg);
    const bool r2 = cr == gr && cg.ids == gg.ids && cg.accepted == gg.accepted;
    std::printf("R2: restored at 4096 + a 2500-id tail: the ring %s a cold run's (%u window rows differ); 64 ids and "
                "acceptance %s\n",
                cr == gr ? "equals (bitwise)" : "DIFFERS from", rows_differ(d, gr, cr, end),
                cg.ids == gg.ids && cg.accepted == gg.accepted ? "identical" : "DIFFER");
    ok &= r2;
    std::printf("R2 caveat (tail 300):\n");
    const uint32_t end2 = arm(300, &cr, &gr, &cg, &gg);
    std::printf("R2 caveat: restored at 4096 + a 300-id tail: %u of %u window rows differ from a cold run's (positions "
                "turn 1's call did not project; not gated); ids %s, acceptance %.3f vs the cold run's %.3f\n",
                rows_differ(d, gr, cr, end2), d.window(), cg.ids == gg.ids ? "identical" : "DIFFER", acceptance(gg),
                acceptance(cg));
    ok &= cg.ids == gg.ids;
  }

  std::printf("dflash_prefix_gpu_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

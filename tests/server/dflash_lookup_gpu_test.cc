// Spec 19f on the card, through EngineAdapter - what b70-serve --spec dflash-lookup's loop calls, no HTTP: the engine
// with the DFlash drafter (int8, beside the gate checkpoint's int8 head, bf16 KV), server::LookupInDFlash over a
// server::PromptLookup exactly as Server::generate drives them, AdaptiveK over MtpCost::dflash for the DFlash
// iterations.
//
//   D2   greedy, 256 ids, against the plain step() loop of the same engine on the three golden prompts, the 36 A4
//        tool-call prompts and the edit / copy set's 8k / 32k prompts (tests/golden/edit-copy: far copies, where lookup
//        enters). The bar is dflash_gpu_test's: IDENTITY (the verify rows equal M = 1's on today's binding, D3; when
//        spec 28b rebinds the M >= 2 lists to gemm_sm, D2 becomes spec 19 §13's - zero non-tie divergences - for
//        dflash_gpu_test and this test alike). Per prompt: lookup iterations / ids in lookup, DFlash iterations, ids an
//        iteration; per edit / copy prompt also plain --spec dflash's ids an iteration and the host-timed ms an id of
//        plain / dflash / dflash-lookup (busy-box timing, informational: r70.speed is the measurement).
//   RF4  the drafter stays warm: after a dflash-lookup run that used lookup (reemit-8k, 256 ids) the drafter's ring
//        (save_ring) is BITWISE the ring the plain step() loop leaves at the same position - every committed row
//        projected, from the lookup verifies too (on today's binding; after 28b the same comparison by cosine >=
//        0.99999 is printed beside it). A truncate_to inside a lookup burst, then continued: the plain run's ids.
//   S    a seeded sampled dflash-lookup run (T 0.7, top-k 20, top-p 0.95) twice from fresh adapters: ids bitwise (the
//        lookup drafts' point-mass acceptance and the DFlash walk draw from the request's RNG); its lookup share
//        printed.
//
// usage: dflash_lookup_gpu_test <target snapshot> <drafter>   (cwd = the source tree). SKIP 77 without either.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/serve_adapters.h"
#include "golden_common.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/engine.h"
#include "server/adaptive_k.h"
#include "server/lookup_in_dflash.h"
#include "server/prompt_lookup.h"

namespace {
using golden::read_ids;
constexpr uint32_t kMaxLen = 40960, kGen = 256, kK = 7;

struct Prompt {
  std::string name;
  std::vector<uint32_t> ids;
  bool far = false;   // an edit / copy prompt
};
std::vector<Prompt> prompts() {
  std::vector<Prompt> out;
  for (const char* n : {"code", "prose", "cjk"}) out.push_back({n, read_ids(std::string("tests/golden/prompts/") + n + ".ids")});
  std::ifstream f("tests/golden/toolcall/manifest.json");
  CHECK(f.good());
  nlohmann::json manifest;
  f >> manifest;
  for (const auto& e : manifest) {
    const std::string n = e.at("name").get<std::string>();
    out.push_back({n, read_ids("tests/golden/toolcall/" + n + ".ids")});
  }
  for (const char* d : {"8k", "32k"})
    for (const char* k : {"reemit", "rename", "quote"}) {
      const std::string n = std::string(k) + "-" + d;
      out.push_back({n, read_ids("tests/golden/edit-copy/" + n + ".ids"), true});
    }
  return out;
}

struct Gen {
  std::vector<uint32_t> ids;
  uint64_t iters = 0, lookup_iters = 0, lookup_ids = 0;
  double ms = 0;
};

server::AdaptiveKOptions ak_options() {
  server::AdaptiveKOptions o;
  o.max_k = kK;
  o.cost = server::MtpCost::dflash(true, true);
  return o;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// The plain step() loop (lookup = false, dflash = false) or plain --spec dflash (AdaptiveK, step_many).
Gen generate_plain(EngineAdapter& a, const std::vector<uint32_t>& prompt, uint32_t n, const server::Sampling& s,
                   bool dflash) {
  a.reset();
  a.prefill(prompt);
  Gen g;
  server::AdaptiveK ak(ak_options());
  const auto t0 = std::chrono::steady_clock::now();
  while (g.ids.size() < n) {
    if (!dflash) {
      g.ids.push_back(a.step(s));
      continue;
    }
    const uint32_t k = ak.next();
    const std::vector<uint32_t> burst = a.step_many(s, k);
    ak.observe(k, std::min<uint32_t>(k, uint32_t(burst.size() - 1)));
    ++g.iters;
    const size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    g.ids.insert(g.ids.end(), burst.begin(), burst.begin() + keep);
    if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
  }
  a.flush_commit();
  g.ms = ms_since(t0);
  return g;
}

// Server::generate's dflash-lookup loop minus the text. `cut_at` > 0: the first lookup burst of >= 3 ids is truncated
// to 2 (RF4's truncate inside a lookup burst).
Gen generate_lookup(EngineAdapter& a, const std::vector<uint32_t>& prompt, uint32_t n, const server::Sampling& s,
                    uint32_t window, bool cut = false, bool* did_cut = nullptr) {
  a.reset();
  a.prefill(prompt);
  Gen g;
  server::AdaptiveK ak(ak_options());
  server::PromptLookup m;
  m.append(prompt);
  server::LookupInDFlashOptions lo;
  lo.window = window;
  lo.max_k = kK;
  server::LookupInDFlash lid(lo);
  const auto t0 = std::chrono::steady_clock::now();
  while (g.ids.size() < n) {
    const uint32_t k = ak.next();
    uint32_t proposed = 0;
    std::vector<uint32_t> burst = a.step_drafts_or_many(s, k, [&](uint32_t x) {
      m.append(x);
      const std::vector<uint32_t> d = lid.propose(m);
      proposed = uint32_t(d.size());
      return d;
    });
    ++g.iters;
    const uint32_t kept = uint32_t(burst.size() - 1);
    if (proposed > 0) {
      lid.observe(proposed, kept);
      ++g.lookup_iters;
    } else {
      ak.observe(k, std::min(k, kept));
    }
    size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    if (cut && proposed > 0 && did_cut && !*did_cut && keep >= 3) {
      keep = 2;
      *did_cut = true;
    }
    if (proposed > 0) g.lookup_ids += keep;
    g.ids.insert(g.ids.end(), burst.begin(), burst.begin() + keep);
    // the matcher holds what the session holds: burst[0] went in through the proposer, the dropped ids never do
    for (size_t i = 1; i < keep; ++i) m.append(burst[i]);
    if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
  }
  a.flush_commit();
  g.ms = ms_since(t0);
  return g;
}

size_t first_diff(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    if (a[i] != b[i]) return i;
  return a.size() == b.size() ? size_t(-1) : std::min(a.size(), b.size());
}

float bf16(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<uint8_t> ring_of(EngineAdapter& a, runtime::dflash::DFlashDrafter& d) {
  std::vector<uint8_t> r(d.ring_bytes());
  d.save_ring(a.imm, r.data());
  return r;
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
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, loader::LmHeadForm::Int8);
  loader::DFlashWeights w = loader::load_drafter(ctx, dsnap, model, loader::DraftFormat::Int8);
  const uint32_t head_rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  runtime::dflash::DFlashDrafter d(ctx, std::move(w), head_rows, kMaxLen, kK);
  runtime::Engine e(ctx, std::move(model), kMaxLen, /*debug_resid=*/false, runtime::KvCache::Bf16, &d);
  d.attach(e, false);
  EngineAdapter lk(e, model::Qwen35::kVocabUsed, kK, /*lookup=*/false, /*dflash=*/true, /*dflash_lookup=*/true);
  EngineAdapter df(e, model::Qwen35::kVocabUsed, kK, false, true);
  const uint32_t window = lk.dflash_window();
  CHECK_EQ(window, d.window());
  std::printf("drafter window %u, K <= %u, max_len %u\n", window, kK, kMaxLen);
  bool ok = true;
  server::Sampling greedy;

  // --- D2 -------------------------------------------------------------------------------------------------
  uint64_t fails = 0, far_lookup_iters = 0, tot_iters = 0, tot_lookup_ids = 0, tot_ids = 0;
  for (const Prompt& p : prompts()) {
    const Gen plain = generate_plain(df, p.ids, kGen, greedy, false);
    const Gen g = generate_lookup(lk, p.ids, kGen, greedy, window);
    const size_t at = first_diff(g.ids, plain.ids);
    std::printf("D2 %-28s %s, %llu iterations (%llu lookup, %llu ids in lookup), %.2f ids an iteration",
                p.name.c_str(), at == size_t(-1) ? "identical" : ("DIFFERS at " + std::to_string(at)).c_str(),
                (unsigned long long)g.iters, (unsigned long long)g.lookup_iters, (unsigned long long)g.lookup_ids,
                double(kGen) / double(std::max<uint64_t>(1, g.iters)));
    if (p.far) {
      const Gen pd = generate_plain(df, p.ids, kGen, greedy, true);
      std::printf("; dflash %.2f ids an iteration%s; ms an id plain %.2f dflash %.2f dflash-lookup %.2f",
                  double(kGen) / double(std::max<uint64_t>(1, pd.iters)),
                  pd.ids == plain.ids ? "" : " (dflash DIFFERS)", plain.ms / kGen, pd.ms / kGen, g.ms / kGen);
      far_lookup_iters += g.lookup_iters;
    }
    std::printf("\n");
    fails += at != size_t(-1);
    tot_iters += g.iters;
    tot_lookup_ids += g.lookup_ids;
    tot_ids += kGen;
  }
  std::printf("D2: %llu divergences; %.2f ids an iteration, %.1f %% of the ids from lookup; lookup iterations on the "
              "edit / copy prompts %llu (busy-box timing above, informational)\n",
              (unsigned long long)fails, double(tot_ids) / double(tot_iters), 100.0 * tot_lookup_ids / tot_ids,
              (unsigned long long)far_lookup_iters);
  ok &= fails == 0;
  ok &= far_lookup_iters > 0;   // the far copies entered lookup (else D2 proved nothing new)

  // --- RF4: the warm ring -------------------------------------------------------------------------------------
  {
    const std::vector<uint32_t> prompt = read_ids("tests/golden/edit-copy/reemit-8k.ids");
    const Gen plain = generate_plain(df, prompt, kGen, greedy, false);
    const std::vector<uint8_t> ring_plain = ring_of(df, d);
    const Gen g = generate_lookup(lk, prompt, kGen, greedy, window);
    const std::vector<uint8_t> ring_lk = ring_of(lk, d);
    const bool bitwise = ring_lk == ring_plain;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i + 1 < ring_lk.size(); i += 2) {
      uint16_t x, y;
      std::memcpy(&x, &ring_lk[i], 2);
      std::memcpy(&y, &ring_plain[i], 2);
      const double fx = bf16(x), fy = bf16(y);
      dot += fx * fy;
      na += fx * fx;
      nb += fy * fy;
    }
    const double cos = dot / std::sqrt(std::max(na * nb, 1e-300));
    std::printf("RF4: after %llu lookup iterations (%llu ids) the ring (%zu B) is %s the plain loop's (cosine %.9f); "
                "ids %s\n",
                (unsigned long long)g.lookup_iters, (unsigned long long)g.lookup_ids, ring_lk.size(),
                bitwise ? "BITWISE" : "NOT bitwise", cos, g.ids == plain.ids ? "identical" : "DIFFER");
    ok &= g.lookup_iters > 0 && bitwise && g.ids == plain.ids;
    bool did_cut = false;
    const Gen c = generate_lookup(lk, prompt, kGen, greedy, window, /*cut=*/true, &did_cut);
    const bool rf4c = did_cut && c.ids == plain.ids;
    std::printf("RF4: a lookup burst truncated to 2 ids, then continued: %s\n",
                !did_cut ? "no lookup burst of 3 to truncate" : rf4c ? "equals the plain run" : "DIFFERS");
    ok &= rf4c;
  }

  // --- S: seeded sampling, reproducible ----------------------------------------------------------------------
  {
    server::Sampling s;
    s.greedy = false;
    s.temperature = 0.7f;
    s.top_k = 20;
    s.top_p = 0.95f;
    s.seed = 42;
    s.has_seed = true;
    const std::vector<uint32_t> prompt = read_ids("tests/golden/edit-copy/reemit-8k.ids");
    Gen g[2];
    for (int r = 0; r < 2; ++r) {
      EngineAdapter fresh(e, model::Qwen35::kVocabUsed, kK, false, true, true);   // its RNG seeds from the request
      g[r] = generate_lookup(fresh, prompt, 128, s, window);
    }
    const bool same = g[0].ids == g[1].ids;
    std::printf("S: seeded sampled dflash-lookup run twice: %s; %llu lookup iterations, %llu of 128 ids in lookup\n",
                same ? "bitwise" : "DIFFERS", (unsigned long long)g[0].lookup_iters,
                (unsigned long long)g[0].lookup_ids);
    ok &= same;
  }

  std::printf("dflash_lookup_gpu_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

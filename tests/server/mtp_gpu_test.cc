// Spec 8 M3 / M4 / M5 on the card (plan 8c Task 3), and the D1/D2 bench (Task 4): the
// real engine through EngineAdapter - exactly what b70-serve's loop calls - with no HTTP.
//
// Gate mode:   mtp_gpu_test <snapshot> <backend>            (cwd = the source tree)
//   M3  greedy, 256 ids, K = 1, 2, 3 against the plain step() loop of the same engine,
//       on the three golden prompts and the 36 A4 tool-call prompts. Plan 8b's M2 showed
//       the M = k + 1 rows bitwise equal to M = 1, so the bar is IDENTITY: any divergence
//       is a failure (reported with its index), not a near-tie.
//   RF1 on the card: a run truncated inside an accepted burst (truncate_to) and then
//       continued equals the plain run.
//   M5  the same greedy K = 3 run twice: ids and the final logits row bitwise.
//   M4  a seeded sampled K = 3 run twice, from fresh adapters: ids bitwise.
//
// Bench mode:  mtp_gpu_test <snapshot> <backend> --bench K [--sampled] [--depth D]
//   One arm of D1/D2: K = 0 loads NO head (today's server), K > 0 loads it. Prints one
//   line per prompt: generated ids, generation ms (prefill excluded), tokens/s, acceptance.
//   Golden prompts are placed at depth D (default 4096) behind a prefix of long32k.ids;
//   the four P0 tool-call scenarios run at their own depth. A driver interleaves arms.
//
// MTP lists are compiled at max_len 16384 only (spec 8 §8 A9): every run here is 16384.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/serve_adapters.h"
#include "golden_common.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/engine.h"

namespace {
using model::Qwen35;
using golden::read_ids;

constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 256;

struct Prompt {
  std::string name;
  std::vector<uint32_t> ids;
};

std::vector<Prompt> golden_prompts(const std::string& src) {
  std::vector<Prompt> out;
  for (const char* n : {"code", "prose", "cjk"})
    out.push_back({n, read_ids(src + "/tests/golden/prompts/" + n + ".ids")});
  return out;
}

std::vector<Prompt> toolcall_prompts(const std::string& src, bool p0_only) {
  std::ifstream f(src + "/tests/golden/toolcall/manifest.json");
  CHECK(f.good());
  nlohmann::json manifest;
  f >> manifest;
  std::vector<Prompt> out;
  for (const auto& e : manifest) {
    const std::string n = e.at("name").get<std::string>();
    if (p0_only && n != "t3_rename-openai" && n != "t4_comment-loader" && n != "t5_test-step" &&
        n != "t4_comment-box")
      continue;
    out.push_back({n, read_ids(src + "/tests/golden/toolcall/" + n + ".ids")});
  }
  return out;
}

struct Gen {
  std::vector<uint32_t> ids;
  double ms = 0;
  uint64_t iters = 0, drafted = 0, accepted = 0;
};

// What server.cc's loop does, minus the text: prefill, then step() or step_many() until
// n ids, truncating the last burst to exactly n.
Gen generate(EngineAdapter& a, const std::vector<uint32_t>& prompt, uint32_t n,
             const server::Sampling& s) {
  a.reset();
  a.prefill(prompt);
  Gen g;
  const auto t0 = std::chrono::steady_clock::now();
  while (g.ids.size() < n) {
    if (a.mtp_k() == 0) {
      g.ids.push_back(a.step(s));
      continue;
    }
    const std::vector<uint32_t> burst = a.step_many(s, a.mtp_k());
    const size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    g.ids.insert(g.ids.end(), burst.begin(), burst.begin() + keep);
    if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
  }
  a.flush_commit();
  g.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  g.iters = a.mtp_iters;
  g.drafted = a.mtp_drafted;
  g.accepted = a.mtp_accepted;
  return g;
}

size_t first_diff(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return i;
  return a.size() == b.size() ? SIZE_MAX : n;
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

struct Loaded {
  l0::Context ctx;
  runtime::Engine eng;
  Loaded(const std::string& snap, runtime::PrefillBackend b, bool mtp)
      : ctx(l0::Context::kFromEnv),
        eng(ctx, loader::load(ctx, snap, kMaxLen, mtp), kMaxLen) {
    eng.set_prefill_backend(b);
    eng.prepare_prefill();
  }
};

int gate(const std::string& snap, runtime::PrefillBackend backend) {
  Loaded L(snap, backend, true);
  const std::string src = ".";
  std::vector<Prompt> prompts = golden_prompts(src);
  for (Prompt& p : toolcall_prompts(src, false)) prompts.push_back(std::move(p));
  const server::Sampling greedy;   // greedy = true
  EngineAdapter plain(L.eng, Qwen35::kVocabUsed, 0);
  EngineAdapter spec[3] = {EngineAdapter(L.eng, Qwen35::kVocabUsed, 1),
                           EngineAdapter(L.eng, Qwen35::kVocabUsed, 2),
                           EngineAdapter(L.eng, Qwen35::kVocabUsed, 3)};
  int divergences = 0;
  uint64_t tot_d[3] = {0, 0, 0}, tot_a[3] = {0, 0, 0}, tot_i[3] = {0, 0, 0};
  std::printf("M3 (%s): greedy %u ids, --mtp K vs --mtp 0; bar: identical\n",
              runtime::prefill_backend_name(backend), kGen);
  std::printf("%-24s %6s | %-22s | %-22s | %-22s\n", "prompt", "ids", "K=1 same acc tok/it",
              "K=2 same acc tok/it", "K=3 same acc tok/it");
  for (const Prompt& p : prompts) {
    const Gen ref = generate(plain, p.ids, kGen, greedy);
    std::printf("%-24s %6zu", p.name.c_str(), p.ids.size());
    for (int k = 0; k < 3; ++k) {
      const Gen g = generate(spec[k], p.ids, kGen, greedy);
      const size_t d = first_diff(ref.ids, g.ids);
      if (d != SIZE_MAX) ++divergences;
      tot_d[k] += g.drafted;
      tot_a[k] += g.accepted;
      tot_i[k] += g.iters;
      char same[32];
      if (d == SIZE_MAX) std::snprintf(same, sizeof(same), "yes");
      else std::snprintf(same, sizeof(same), "NO@%zu", d);
      std::printf(" | %-6s %5.3f %5.2f    ", same, g.drafted ? double(g.accepted) / g.drafted : 0.0,
                  g.iters ? double(kGen) / g.iters : 0.0);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  for (int k = 0; k < 3; ++k)
    std::printf("M3 pooled K=%d: acceptance %.4f (%llu/%llu), %.3f ids per iteration\n", k + 1,
                double(tot_a[k]) / tot_d[k], (unsigned long long)tot_a[k],
                (unsigned long long)tot_d[k], double(kGen * prompts.size()) / tot_i[k]);
  std::printf("M3: %d divergences over %zu prompts x 3 K\n", divergences, prompts.size());

  // RF1 on the card: keep only 1 id of the first burst with >= 3, then continue.
  int rf1_fail = 0;
  for (size_t pi = 0; pi < 3; ++pi) {
    const Prompt& p = prompts[pi];
    const Gen ref = generate(plain, p.ids, 96, greedy);
    EngineAdapter& a = spec[2];
    a.reset();
    a.prefill(p.ids);
    std::vector<uint32_t> ids;
    bool cut = false;
    while (ids.size() < 96) {
      std::vector<uint32_t> burst = a.step_many(greedy, a.mtp_k());
      if (!cut && burst.size() >= 3) {
        a.truncate_to(a.pos() - uint32_t(burst.size() - 1));
        burst.resize(1);
        cut = true;
      }
      for (uint32_t id : burst)
        if (ids.size() < 96) ids.push_back(id);
    }
    a.flush_commit();
    const bool ok = cut && first_diff(std::vector<uint32_t>(ref.ids.begin(), ref.ids.end()),
                                      ids) == SIZE_MAX;
    std::printf("RF1 %s: truncated inside a burst %s, 96 ids %s\n", p.name.c_str(),
                cut ? "yes" : "NO", ok ? "equal to plain" : "DIFFER");
    if (!ok) ++rf1_fail;
  }

  // M5: the same greedy K = 3 run twice, ids and the final logits row.
  int m5_fail = 0;
  for (size_t pi = 0; pi < 3; ++pi) {
    uint64_t h[2];
    std::vector<uint32_t> ids[2];
    for (int r = 0; r < 2; ++r) {
      ids[r] = generate(spec[2], prompts[pi].ids, kGen, greedy).ids;
      std::vector<float> row(Qwen35::kVocab);
      l0::CmdList imm = l0::CmdList::immediate(L.ctx);
      imm.copy(row.data(), L.eng.buffers().logits.ptr(), row.size() * sizeof(float));
      h[r] = fnv(row.data(), row.size() * sizeof(float), fnv(ids[r].data(), ids[r].size() * 4));
    }
    const bool ok = h[0] == h[1] && ids[0] == ids[1];
    std::printf("M5 %s: run 1 %016llx run 2 %016llx %s\n", prompts[pi].name.c_str(),
                (unsigned long long)h[0], (unsigned long long)h[1], ok ? "bitwise" : "DIFFER");
    if (!ok) ++m5_fail;
  }

  // M4 end to end: seeded sampled K = 3, twice from fresh adapters (fresh RNGs).
  int m4_fail = 0;
  server::Sampling sampled;
  sampled.greedy = false;
  sampled.temperature = 1.0f;
  sampled.top_p = 0.95f;
  sampled.top_k = 20;
  sampled.has_seed = true;
  sampled.seed = 8;
  for (size_t pi = 0; pi < 3; ++pi) {
    std::vector<uint32_t> ids[2];
    double acc = 0;
    for (int r = 0; r < 2; ++r) {
      EngineAdapter a(L.eng, Qwen35::kVocabUsed, 3);
      const Gen g = generate(a, prompts[pi].ids, kGen, sampled);
      ids[r] = g.ids;
      acc = g.drafted ? double(g.accepted) / g.drafted : 0;
    }
    const bool ok = ids[0] == ids[1];
    std::printf("M4 seeded sampled %s: acceptance %.3f, two runs %s\n", prompts[pi].name.c_str(),
                acc, ok ? "bitwise" : "DIFFER");
    if (!ok) ++m4_fail;
  }
  const bool pass = divergences == 0 && rf1_fail == 0 && m5_fail == 0 && m4_fail == 0;
  std::printf("mtp_gpu_test (%s): %s\n", runtime::prefill_backend_name(backend),
              pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int bench(const std::string& snap, runtime::PrefillBackend backend, uint32_t k, bool sampled,
          uint32_t depth) {
  Loaded L(snap, backend, k > 0);
  const std::string src = ".";
  const std::vector<uint32_t> filler = read_ids(src + "/tests/golden/prompts/long32k.ids");
  std::vector<Prompt> prompts;
  for (Prompt p : golden_prompts(src)) {
    CHECK(depth > p.ids.size() && depth - p.ids.size() <= filler.size());
    std::vector<uint32_t> ids(filler.begin(), filler.begin() + (depth - p.ids.size()));
    ids.insert(ids.end(), p.ids.begin(), p.ids.end());
    prompts.push_back({"golden/" + p.name + "@" + std::to_string(depth), ids});
  }
  for (Prompt& p : toolcall_prompts(src, true)) prompts.push_back({"toolcall/" + p.name, p.ids});
  server::Sampling s;
  if (sampled) {
    s.greedy = false;
    s.temperature = 1.0f;
    s.top_p = 0.95f;
    s.top_k = 20;
    s.has_seed = true;
    s.seed = 1234;
  }
  EngineAdapter a(L.eng, Qwen35::kVocabUsed, k);
  (void)generate(a, prompts[0].ids, 32, s);   // warm-up
  for (const Prompt& p : prompts) {
    const Gen g = generate(a, p.ids, kGen, s);
    std::printf("BENCH k=%u mode=%s prompt=%s depth=%zu ids=%zu ms=%.1f tps=%.3f acc=%.4f "
                "per_iter=%.3f\n",
                k, sampled ? "sampled" : "greedy", p.name.c_str(), p.ids.size(), g.ids.size(),
                g.ms, 1000.0 * g.ids.size() / g.ms,
                g.drafted ? double(g.accepted) / g.drafted : 0.0,
                g.iters ? double(g.ids.size()) / g.iters : 1.0);
    std::fflush(stdout);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: mtp_gpu_test <snapshot> <backend> [--bench K [--sampled] [--depth D]]\n");
    return 2;
  }
  runtime::PrefillBackend backend{};
  CHECK(runtime::parse_prefill_backend(argv[2], backend));
  int bench_k = -1;
  bool sampled = false;
  uint32_t depth = 4096;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--bench" && i + 1 < argc) bench_k = std::atoi(argv[++i]);
    else if (a == "--sampled") sampled = true;
    else if (a == "--depth" && i + 1 < argc) depth = uint32_t(std::atoi(argv[++i]));
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  try {
    if (bench_k >= 0) return bench(argv[1], backend, uint32_t(bench_k), sampled, depth);
    return gate(argv[1], backend);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mtp_gpu_test: %s\n", e.what());
    return 1;
  }
}

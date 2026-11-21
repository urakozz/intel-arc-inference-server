// PERF-113: how far an int8 prompt pass moves the model's next-token distribution from the bf16 pass, at long depth.
//   probe_int8_kl <snapshot> <backend> <G> <ids file>[:<len>]... [--chunk-cap 2048|4096] [--max-len N]
// Per prompt (its first <len> ids, all of them without one), on ONE engine: prefill on l0 and decode G greedy ids,
// keeping the logits row before each (G + 1 rows: the prefill's last row, then G decode steps); then prefill on
// <backend> (l0-int8) and force the same G ids, keeping its rows. Decode is the same code on both sides, so every
// difference is what the prompt pass wrote (KV, GDN state, conv ring). PERF-114: l0-int8 runs on the int8 context (KV
// cache), so the reference (l0 on the bf16 cache, the default) runs on its own engine first (two do not fit one card)
// and the difference is all of what the int8 pass changes, decode reading the int8 cache included. Per prompt and over
// all of them:
// top-1 agreement (argmax over the used vocabulary), mean and worst KL(l0 || backend) in nats (softmax in fp64), and
// the first row whose top-1 differs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/engine.h"
#include "runtime/prefill_backend.h"

namespace {
using Ids = std::vector<uint32_t>;
using model::Qwen35;

Ids read_ids(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("probe_int8_kl: cannot read " + path);
  Ids v;
  uint32_t x = 0;
  while (in >> x) v.push_back(x);
  return v;
}

uint32_t top1(const std::vector<float>& r) {
  return uint32_t(std::max_element(r.begin(), r.begin() + Qwen35::kVocabUsed) - r.begin());
}

// KL(p || q) over the used vocabulary, p = softmax(a), q = softmax(b), fp64.
double kl(const std::vector<float>& a, const std::vector<float>& b) {
  const uint32_t n = Qwen35::kVocabUsed;
  const double ma = *std::max_element(a.begin(), a.begin() + n), mb = *std::max_element(b.begin(), b.begin() + n);
  double za = 0, zb = 0;
  for (uint32_t i = 0; i < n; ++i) {
    za += std::exp(double(a[i]) - ma);
    zb += std::exp(double(b[i]) - mb);
  }
  const double la = std::log(za), lb = std::log(zb);
  double s = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const double lpa = double(a[i]) - ma - la, lpb = double(b[i]) - mb - lb;
    s += std::exp(lpa) * (lpa - lpb);
  }
  return s;
}

// The G + 1 logits rows of one prompt on the engine's current backend; `force` empty: greedy, filled with the ids.
std::vector<std::vector<float>> rows(runtime::Engine& eng, l0::CmdList& imm, const Ids& prompt, uint32_t G,
                                     Ids& force) {
  std::vector<std::vector<float>> out(G + 1, std::vector<float>(Qwen35::kVocab));
  const bool greedy = force.empty();
  eng.reset();
  eng.prefill(prompt);
  imm.copy(out[0].data(), eng.prefill_scratch()->logits.ptr(), size_t(Qwen35::kVocab) * 4);
  for (uint32_t g = 0; g < G; ++g) {
    if (greedy) force.push_back(top1(out[g]));
    eng.ingest({force[g]});
    imm.copy(out[g + 1].data(), eng.buffers().logits.ptr(), size_t(Qwen35::kVocab) * 4);
  }
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: probe_int8_kl <snapshot> <backend> <G> <ids file>[:<len>]... "
                         "[--chunk-cap 2048|4096] [--max-len N]\n");
    return 2;
  }
  runtime::PrefillBackend backend{};
  if (!runtime::parse_prefill_backend(argv[2], backend)) return 2;
  const uint32_t G = uint32_t(std::stoul(argv[3]));
  uint32_t cap = 4096, max_len = 66560;
  std::vector<std::pair<std::string, Ids>> prompts;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--chunk-cap" && i + 1 < argc) { cap = uint32_t(std::stoul(argv[++i])); continue; }
    if (a == "--max-len" && i + 1 < argc) { max_len = uint32_t(std::stoul(argv[++i])); continue; }
    const size_t c = a.rfind(':');
    const std::string path = c == std::string::npos ? a : a.substr(0, c);
    Ids ids = read_ids(path);
    if (c != std::string::npos) ids.resize(std::min(ids.size(), size_t(std::stoul(a.substr(c + 1)))));
    if (ids.size() + G > max_len) throw std::runtime_error("probe_int8_kl: " + a + " + G exceeds --max-len");
    prompts.emplace_back(a, std::move(ids));
  }
  l0::Context ctx(0);
  const bool int8 = backend == runtime::PrefillBackend::L0Int8;
  const std::string snap = loader::resolve_snapshot(argv[1]);
  const auto engine = [&](runtime::KvCache kv) {
    auto e = std::make_unique<runtime::Engine>(ctx, loader::load(ctx, snap, max_len), max_len, false, kv);
    e->set_prefill_chunk_cap(cap);
    return e;
  };
  std::unique_ptr<runtime::Engine> eng = engine(int8 ? runtime::KvCache::Bf16 : runtime::default_kv_cache());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::printf("probe_int8_kl: l0 against %s%s, chunk cap %u, %u decode rows after each prompt\n",
              runtime::prefill_backend_name(backend), int8 ? " (int8 KV; l0 on the bf16 KV)" : "", cap, G);
  // Every prompt's l0 rows first, then the backend's against them.
  std::vector<std::vector<std::vector<float>>> refs;
  std::vector<Ids> forces(prompts.size());
  eng->set_prefill_backend(runtime::PrefillBackend::L0);
  for (size_t i = 0; i < prompts.size(); ++i) refs.push_back(rows(*eng, imm, prompts[i].second, G, forces[i]));
  if (int8) {
    eng.reset();   // its device memory free before the second load
    eng = engine(runtime::KvCache::Int8);
  }
  eng->set_prefill_backend(backend);
  uint32_t all = 0, all_agree = 0;
  double all_kl = 0, all_worst = 0;
  for (size_t i = 0; i < prompts.size(); ++i) {
    const auto& [name, ids] = prompts[i];
    const auto& ref = refs[i];
    const auto got = rows(*eng, imm, ids, G, forces[i]);
    uint32_t agree = 0;
    int first = -1;
    double sum = 0, worst = 0;
    for (uint32_t r = 0; r <= G; ++r) {
      const bool same = top1(ref[r]) == top1(got[r]);
      agree += same;
      if (!same && first < 0) first = int(r);
      const double k = kl(ref[r], got[r]);
      sum += k;
      worst = std::max(worst, k);
    }
    std::printf("KL %s ids %zu rows %u top1 %u/%u kl_mean %.3e kl_max %.3e first_diff %d\n", name.c_str(), ids.size(),
                G + 1, agree, G + 1, sum / (G + 1), worst, first);
    all += G + 1;
    all_agree += agree;
    all_kl += sum;
    all_worst = std::max(all_worst, worst);
  }
  std::printf("KL total rows %u top1 %u/%u (%.2f %%) kl_mean %.3e kl_max %.3e\n", all, all_agree, all,
              100.0 * all_agree / all, all_kl / all, all_worst);
  return 0;
}

// Spec 26c Task 4 (D2 on the cards): `--spec dflash` greedy equals `--spec off` token for token - SergiioB's Bolt on two cards
// with the real LiLiCorr drafter (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) on device 1, through the server's own
// path (cli::bolt::BoltEngineAdapterT::step_many over cli/bolt_spec.h's drafter). Box only (labels checkpoint;bolt;lilicorr).
//   bolt_dflash_test <bolt checkpoint> <lilicorr snapshot> <prompt.ids>... [--format exact|int8|int4] [--tokens N]
//
//   1. The reference: 25c's decode list (no verify list yet - D0's engine), every prompt prefilled and N greedy ids generated.
//   2. enable_spec (the union MoE) and the drafter; per prompt: the same prefill, then step_many at every fixed K 1..7 and with
//      AdaptiveK over MtpCost::bolt_lilicorr (the server's policy) - every id equals the reference's; the acceptance (accepted
//      drafts a verify, MEASURED) printed per K - D5's mean accepted length on these prompts.
//   3. Every draft reads a valid anchor (a projected row: the prefill's last, then every commit's newest).
// Exit 77 (SKIP) without two GPUs, the Bolt export or the drafter.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "check.h"
#include "cli/bolt_serve.h"
#include "cli/bolt_spec.h"
#include "server/adaptive_k.h"

namespace {
namespace rb = runtime::bolt;
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: bolt_dflash_test <bolt checkpoint> <lilicorr snapshot> <prompt.ids>... [--format F] "
                         "[--tokens N]\n");
    return 2;
  }
  const std::string snap = bolt_rig::bolt_snapshot(argv[1]);
  if (snap.empty() || l0::Context::gpu_count() < 2) {
    std::printf("SKIP: needs SergiioB's Bolt export (%s) and two GPUs\n", argv[1]);
    return 77;
  }
  cli::bolt::SpecArgs sa;
  sa.spec = "dflash";
  sa.draft_model = argv[2];
  sa.have_model = true;
  uint32_t tokens = 128;
  std::vector<std::string> prompts;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--format" && i + 1 < argc) sa.format_arg = argv[++i];
    else if (a == "--tokens" && i + 1 < argc) tokens = uint32_t(std::atoi(argv[++i]));
    else prompts.push_back(a);
  }
  try {
    cli::bolt::check_spec(sa, 2, runtime::PpHandoff::Copy, true);
    (void)loader::LiliCheckpoint(sa.draft_model);
  } catch (const std::exception& e) {
    std::printf("SKIP: the drafter: %s\n", e.what());
    return 77;
  }
  bolt_rig::Options o;
  o.devices = 2;
  o.max_len = 16384;
  bolt_rig::Rig rig;
  bolt_rig::build(rig, snap, o);
  rb::BoltEngine& e = *rig.eng;
  const model::BoltDesc& d = e.model().desc;

  // 1. the reference: 25c's decode list
  std::vector<std::vector<uint32_t>> ids, want;
  for (const std::string& p : prompts) {
    ids.push_back(bolt_rig::read_ids(p, d.vocab));
    CHECK(!ids.back().empty());
    e.reset();
    e.prefill(ids.back());
    want.push_back(e.generate(tokens));
  }

  // 2. the verify lists and the drafter
  e.enable_spec(rb::VerifyMoe::Union);
  cli::bolt::BoltDrafter dr = cli::bolt::attach_drafter(e, sa, snap);
  std::printf("drafter: %s, %s (%zu B), key %016llx\n", sa.draft_model.c_str(), loader::draft_format_name(sa.format),
              dr.drafter->bytes(), (unsigned long long)e.drafter_key());
  cli::bolt::BoltEngineAdapterT<rb::BoltEngine> a(e, d.vocab, 7, false, cli::bolt::draft_hooks(e, *dr.drafter), sa.q);
  server::Sampling greedy;
  int bad = 0;
  for (uint32_t K = 1; K <= 8; ++K) {   // K 8: AdaptiveK
    uint64_t acc = 0, iters = 0;
    for (size_t p = 0; p < prompts.size(); ++p) {
      a.reset();
      a.prefill(ids[p]);
      server::AdaptiveKOptions ko;
      ko.max_k = 7;
      ko.cost = server::MtpCost::bolt_lilicorr(true, cli::bolt::format_index(sa.format), true);
      server::AdaptiveK ak(ko);
      std::vector<uint32_t> got;
      while (got.size() < tokens) {
        const uint32_t k = K <= 7 ? K : ak.next();
        const uint64_t a0 = a.spec_accepted();
        const std::vector<uint32_t> run = a.step_many(greedy, k);
        if (K > 7) ak.observe(k == 0 ? 0 : k, uint32_t(a.spec_accepted() - a0));
        got.insert(got.end(), run.begin(), run.end());
        if (run.size() > 1 || k > 0) CHECK(dr.drafter->readback().anchor_valid == 1);   // the anchor is a projected row
      }
      a.flush_commit();
      got.resize(tokens);
      if (got != want[p]) {
        std::fprintf(stderr, "%s, %s: --spec dflash greedy differs from --spec off\n", prompts[p].c_str(),
                     K <= 7 ? ("K " + std::to_string(K)).c_str() : "AdaptiveK");
        ++bad;
      }
      acc += a.spec_accepted();
      iters += a.spec_iterations();
      a.report();
    }
    std::printf("%s: greedy %s --spec off on %zu prompts; %.3f tokens a verify (MEASURED, %llu iterations)\n",
                K <= 7 ? ("K " + std::to_string(K)).c_str() : "AdaptiveK", bad ? "DIFFERS from" : "=", prompts.size(),
                iters ? 1.0 + double(acc) / double(iters) : 0.0, (unsigned long long)iters);
  }
  CHECK_EQ(bad, 0);
  std::printf("bolt_dflash_test OK (D2: greedy lossless, %s drafter)\n", loader::draft_format_name(sa.format));
  return 0;
}

// Plan 21-QFN-v2b Task 5 Step 0: Qwen3.8-Flash-Next's MTP head under `--parallel tp` (box only; label
// checkpoint;qwen4exp;tp). One engine a process (each pins the 51.8 GB PLE table), so the comparison is two runs:
//   qwen4exp_tp_mtp_test <checkpoint> <ids> plain|mtp <ids file> [layers 18] [n 64]
// The prompts are <ids> cycled to 120 and 2100 positions, filled by decode (ingest, one id a step).
//   plain  tp without the head: each prompt's n greedy ids (the plain decode list) -> <ids file>
//   mtp    tp with the head whole on card 1:
//          (1) plain steps (verify(0) + commit(0)): the n ids equal the plain run's, bitwise (every verify row is the
//              M = 1 step's arithmetic)
//          (2) spec 8's greedy loop at K = 1..3 (draft k, verify, accept the drafts equal to the verify's argmax,
//              commit): the n ids equal the plain run's; acceptance printed
//          (3) a snapshot at the prompt's end restored into a reset engine, then the loop at K 3: the same ids
//          (4) the prompt PREFILLED (the head's pass on card 1), then the loop at K 3: the plain run's prefilled ids
//   plain also writes each prompt's n greedy ids after a prefill (after the ingest ones)
// Exit 77 (SKIP) with one GPU or without the checkpoint.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp/qwen4exp_tp_engine.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;
using Ids = std::vector<uint32_t>;

Ids speculative(rq::Qwen4ExpTpEngine& e, uint32_t K, uint32_t n, uint64_t& drafted, uint64_t& accepted) {
  Ids out;
  while (out.size() < n) {
    const uint32_t k = std::min(K, e.max_verify_k());
    if (k == 0) {
      out.push_back(e.generate(1)[0]);
      continue;
    }
    const uint32_t x = e.pending();
    e.draft(k);
    e.verify(k);
    const uint32_t* v = e.verify_ids();
    uint32_t j = 0;
    while (j < k && e.draft_ids()[j] == v[j]) ++j;
    out.push_back(x);
    for (uint32_t i = 0; i < j; ++i) out.push_back(e.draft_ids()[i]);
    drafted += k;
    accepted += j;
    e.commit(j, v[j]);
  }
  out.resize(n);
  return out;
}

size_t first_diff(const Ids& a, const Ids& b) {
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
  return i;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: qwen4exp_tp_mtp_test <checkpoint> <ids> plain|mtp <ids file> [layers] [n]\n");
    return 2;
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  const uint32_t layers = argc > 5 ? uint32_t(std::atoi(argv[5])) : 18;
  const uint32_t n = argc > 6 ? uint32_t(std::atoi(argv[6])) : 64;
  const std::string mode = argv[3], path = argv[4];
  CHECK(mode == "plain" || mode == "mtp");
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::open(rig, 1);
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: one GPU\n");
    return 77;
  }
  qwen4exp_rig::open(rig, 2);
  const uint32_t max_len = 16384;
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, layers);
  const Ids src = qwen4exp_rig::read_ids(argv[2], d.vocab);
  CHECK(!src.empty());
  const bool head = mode == "mtp";
  rq::Qwen4ExpTpEngine e(*rig.c0, *rig.c1,
                         loader::load_qwen4exp(rig.ctx, snap, max_len, model::Q4Placement::tp2(d, model::Q4TpEnds::Split),
                                               loader::LmHeadForm::Int8, layers, head),
                         max_len);
  CHECK_EQ(e.mtp(), head);
  const std::vector<size_t> lens = {120, 2100};
  const auto prompt = [&](size_t len) {
    Ids p(len);
    for (size_t i = 0; i < len; ++i) p[i] = src[i % src.size()];
    return p;
  };
  if (!head) {
    FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    for (size_t len : lens) {
      e.reset();
      e.ingest(prompt(len));
      const Ids g = e.generate(n);
      CHECK(std::fwrite(g.data(), 4, g.size(), f) == g.size());
      std::printf("tp-mtp: plain, prompt %zu: %u greedy ids, %.1f t/s\n", len, n, e.last_tok_per_s());
    }
    for (size_t len : lens) {
      e.reset();
      e.prefill(prompt(len));
      const Ids g = e.generate(n);
      CHECK(std::fwrite(g.data(), 4, g.size(), f) == g.size());
    }
    std::fclose(f);
    std::printf("qwen4exp_tp_mtp_test OK\n");
    return 0;
  }
  std::vector<Ids> plain(lens.size(), Ids(n)), plain_pf(lens.size(), Ids(n));
  {
    FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f != nullptr);
    for (Ids& g : plain) CHECK(std::fread(g.data(), 4, n, f) == n);
    for (Ids& g : plain_pf) CHECK(std::fread(g.data(), 4, n, f) == n);
    std::fclose(f);
  }
  int bad = 0;
  for (size_t i = 0; i < lens.size(); ++i) {
    const Ids p = prompt(lens[i]);
    e.reset();
    e.ingest(p);
    const Ids g = e.generate(n);
    const size_t f1 = first_diff(g, plain[i]);
    std::printf("tp-mtp: (1) prompt %zu, plain steps with the head: %s (first difference at %zu of %u), %.1f t/s\n",
                lens[i], f1 == n ? "the plain run's ids" : "DIFFER", f1, n, e.last_tok_per_s());
    bad += f1 != n;
    for (uint32_t K = 1; K <= rq::kMaxDraft; ++K) {
      e.reset();
      e.ingest(p);
      uint64_t drafted = 0, accepted = 0;
      const Ids s = speculative(e, K, n, drafted, accepted);
      const size_t f2 = first_diff(s, plain[i]);
      std::printf("tp-mtp: (2) prompt %zu, K %u: %s (first difference at %zu), acceptance %.3f (%llu of %llu)\n",
                  lens[i], K, f2 == n ? "the plain run's ids" : "DIFFER", f2, drafted ? double(accepted) / drafted : 0.0,
                  (unsigned long long)accepted, (unsigned long long)drafted);
      bad += f2 != n;
    }
    // (3) the snapshot round trip at the prompt's end
    e.reset();
    e.ingest(p);
    std::vector<uint8_t> st(e.state_bytes()), kv(e.kv_bytes(uint32_t(p.size())));
    const uint32_t at = e.pos(), pend = e.pending();
    e.save_state(st.data());
    e.save_kv(0, at, kv.data());
    e.reset();
    e.load_kv(0, at, kv.data());
    e.load_state(st.data(), at);
    e.set_token(pend);
    uint64_t drafted = 0, accepted = 0;
    const Ids s = speculative(e, rq::kMaxDraft, n, drafted, accepted);
    const size_t f3 = first_diff(s, plain[i]);
    std::printf("tp-mtp: (3) prompt %zu, restored snapshot, K 3: %s (first difference at %zu), acceptance %.3f\n",
                lens[i], f3 == n ? "the plain run's ids" : "DIFFER", f3, drafted ? double(accepted) / drafted : 0.0);
    bad += f3 != n;
    // (4) prefilled
    e.reset();
    e.prefill(p);
    drafted = accepted = 0;
    const Ids s4 = speculative(e, rq::kMaxDraft, n, drafted, accepted);
    const size_t f4 = first_diff(s4, plain_pf[i]);
    std::printf("tp-mtp: (4) prompt %zu, prefilled, K 3: %s (first difference at %zu), acceptance %.3f\n", lens[i],
                f4 == n ? "the plain run's prefilled ids" : "DIFFER", f4, drafted ? double(accepted) / drafted : 0.0);
    bad += f4 != n;
  }
  CHECK_EQ(bad, 0);
  std::printf("qwen4exp_tp_mtp_test OK\n");
  return 0;
}

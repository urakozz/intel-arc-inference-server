// Spec 26c Task 1 (D3 on the card): Bolt's verify rows at M = 2..8 are bitwise the M = 1 steps. Box only (labels
// checkpoint;bolt;lilicorr).
//   bolt_verify_test <checkpoint> [plain] [eager] [b16]
// <checkpoint>: 25b's synthetic (n12-*: 12 blocks, one card) or SergiioB's export (two cards, the copy hand-off). `plain`: the
// verify's routed experts in the plain M-row form (B70_BOLT_VERIFY_MOE=plain) instead of the union (Decision 10).
//
// The reference is 25c's DECODE list (no verify list exists yet: D0's engine): from the state S after a prompt prefilled, eight
// fixed ids t_0 .. t_7 stepped one replay each (set_token, generate(1)), recording after each step m its logits row, the KV row
// of every attention block at pos + m, the raw-xBC column (pos + m) % 16 of every Mamba block, the SSM state of three Mamba
// blocks (the first, the last on device 0, the last) and every tap block's output (debug_tap). Then enable_spec and a
// recording drafter (bolt_spec_rig::Recorder on the last card), and for every M = 2..8, from S restored (save / load_state +
// _kv), verify(t_1 .. t_{M-1}) with t_0 pending:
//   - read_verify_logits row m == the reference's logits after step m, bitwise
//   - the KV rows pos .. pos + M - 1 and the ring columns == the reference's, bitwise
//   - the SSM slot (live + m) % 8 of the three blocks == the reference's state after step m, bitwise
//   - the taps the verify wrote for row m (device 0's three crossed before the hand-off) == the reference's tap blocks
//   - the list's launches == runtime::bolt::verify_launches(M) (one captured list per M, exactly M real rows)
// and the M = 1 verify list (the slot SSM) the same against step 0. Exit 77 (SKIP) when the checkpoint is absent or does not fit.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "bolt_spec_rig.h"
#include "check.h"
#include "runtime/bolt/bolt_sizes.h"

namespace {
namespace rb = runtime::bolt;
constexpr uint32_t kMaxLen = 4096, kSteps = 8;

struct Step {
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> kv, cols;   // per attention block: K || V rows; per Mamba block: the column
  std::vector<std::vector<float>> states;        // the three Mamba blocks'
  std::vector<uint16_t> taps;                    // [taps][hidden]
};

std::vector<uint32_t> watched(const model::BoltDesc& d, const model::BoltPlacement& p) {
  uint32_t last0 = 0;
  for (uint32_t l = 0; l < p.end(0); ++l)
    if (d.is_mamba(l)) last0 = d.mamba_index(l);
  return {0, last0, d.mamba_layers() - 1};
}

Step record(rb::BoltEngine& e, uint32_t at, const std::vector<uint32_t>& taps_of, const std::vector<uint16_t>* debug_h) {
  const model::BoltDesc& d = e.model().desc;
  Step s;
  for (uint32_t a = 0; a < d.attn_layers(); ++a) {
    std::vector<uint16_t> k = e.read_kv(a, at, 1, false), v = e.read_kv(a, at, 1, true);
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(k);
  }
  for (uint32_t m = 0; m < d.mamba_layers(); ++m) {
    const std::vector<uint16_t> ring = e.read_conv_ring(m);
    const size_t slot = at % 16;
    s.cols.emplace_back(ring.begin() + long(slot * d.m_conv_dim), ring.begin() + long((slot + 1) * d.m_conv_dim));
  }
  if (debug_h)
    for (uint32_t t : taps_of) s.taps.insert(s.taps.end(), debug_h->begin() + long(size_t(t) * d.hidden),
                                             debug_h->begin() + long(size_t(t + 1) * d.hidden));
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bolt_verify_test <checkpoint> [plain] [eager] [b16]\n");
    return 2;
  }
  const std::string snap = bolt_rig::bolt_snapshot(argv[1]);
  if (snap.empty()) {
    std::printf("SKIP: %s is not a Bolt checkpoint\n", argv[1]);
    return 77;
  }
  bolt_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  rb::VerifyMoe moe = rb::VerifyMoe::Union;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "plain") moe = rb::VerifyMoe::Plain;
    if (a == "eager") o.attn = rb::BoltAttn::Eager;
    if (a == "b16") o.state = rb::BoltState::Bf16;
  }
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(snap, 0);
  o.devices = pre.layers == 88 ? 2 : 1;
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: the real model needs two GPUs\n");
    return 77;
  }
  if (!bolt_rig::fits(pre, o, true)) {
    std::printf("SKIP: %s does not fit\n", snap.c_str());
    return 77;
  }
  bolt_rig::Rig rig;
  bolt_rig::build(rig, snap, o);
  rb::BoltEngine& e = *rig.eng;
  const model::BoltDesc& d = e.model().desc;
  const std::vector<uint32_t> tl = bolt_spec_rig::tap_layers(d), mw = watched(d, e.model().place);
  std::vector<uint32_t> prompt;
  for (uint32_t i = 0; i < 300; ++i) prompt.push_back((i * 7919u + 13u) % d.vocab);
  const std::vector<uint32_t> t = {17, 1013, 4242, 99, 23456, 777, 1, 65000};

  // The reference: 25c's decode list, one replay a step.
  e.prefill(prompt);
  const bolt_spec_rig::Session S = bolt_spec_rig::save(e);
  const uint32_t pos = S.pos;
  std::vector<Step> ref;
  for (uint32_t m = 0; m < kSteps; ++m) {
    e.set_token(t[m]);
    e.generate(1);
    const std::vector<uint16_t> h = e.read_debug_h();
    Step s = record(e, pos + m, tl, &h);
    s.logits = e.read_logits();
    for (uint32_t b : mw) s.states.push_back(e.read_ssm_state(b));
    ref.push_back(std::move(s));
  }

  // The verify lists, a recording drafter on the last card.
  e.enable_spec(moe);
  bolt_spec_rig::Recorder rec(e.context(e.devices() - 1), tl, d.hidden);
  e.attach_drafter(rec);
  int bad = 0;
  for (uint32_t M = 1; M <= kSteps; ++M) {
    bolt_spec_rig::load(e, S, t[0]);
    e.verify(t.data() + 1, M - 1);
    std::vector<float> lg(size_t(M) * d.vocab);
    e.read_verify_logits(lg.data(), M);
    const std::vector<uint16_t> taps = rec.read_taps(M);
    CHECK_EQ(e.verify_launches(M), rb::verify_launches(d, e.model().place, M, o.attn, moe));
    const uint32_t live = e.ssm_live();
    for (uint32_t m = 0; m < M; ++m) {
      const Step got = record(e, pos + m, tl, nullptr);
      const std::string w = "M " + std::to_string(M) + " row " + std::to_string(m);
      if (std::memcmp(lg.data() + size_t(m) * d.vocab, ref[m].logits.data(), size_t(d.vocab) * 4) != 0) {
        std::fprintf(stderr, "%s: the logits differ from the decode step's\n", w.c_str());
        ++bad;
      }
      if (got.kv != ref[m].kv || got.cols != ref[m].cols) {
        std::fprintf(stderr, "%s: KV rows or ring columns differ\n", w.c_str());
        ++bad;
      }
      for (size_t b = 0; b < mw.size(); ++b)
        if (e.read_ssm_slot(mw[b], (live + m) % 8) != ref[m].states[b]) {
          std::fprintf(stderr, "%s: Mamba block %u's slot %u differs from the state after the decode step\n", w.c_str(), mw[b],
                       (live + m) % 8);
          ++bad;
        }
      const size_t row = tl.size() * size_t(d.hidden);
      if (std::vector<uint16_t>(taps.begin() + long(m * row), taps.begin() + long((m + 1) * row)) != ref[m].taps) {
        std::fprintf(stderr, "%s: the taps differ from the decode step's blocks %u..%u\n", w.c_str(), tl.front(), tl.back());
        ++bad;
      }
    }
    std::printf("M %u (%s MoE): %zu launches; logits, KV rows, ring columns, SSM slots and taps %s the decode steps'\n", M,
                rb::verify_moe_name(moe), e.verify_launches(M), bad ? "DIFFER from" : "= (bitwise)");
  }
  CHECK_EQ(bad, 0);
  std::printf("bolt_verify_test OK (%s, %u device(s), %s attention, %s state, %s MoE)\n", d.name.c_str(), e.devices(),
              rb::bolt_attn_name(o.attn), rb::bolt_state_name(o.state), rb::verify_moe_name(moe));
  return 0;
}

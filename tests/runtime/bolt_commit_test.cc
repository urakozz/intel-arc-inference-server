// Spec 26c Task 2 (Review Focus 1 on the card): after a verify of M rows and a commit of j accepted drafts, the live SSM slot on
// both cards holds exactly the state after j + 1 plain steps, and decoding continues as the plain run. Box only (labels
// checkpoint;bolt;lilicorr).
//   bolt_commit_test <checkpoint> [plain] [b16]
//
// The reference is 25c's decode list (before enable_spec): from the state S after a prompt, the eight ids t_0 .. t_7 one replay
// each, then from each prefix t_0 .. t_j (j = 0 / 3 / 6 / 7) the id t_{j+1} (or 5 for j = 7) and 64 greedy ids - recording the
// state of three Mamba blocks after step j and the 64 ids. Then enable_spec (no drafter: --spec lookup's lists) and, from S
// restored, verify(t_1 .. t_7) at M 8 and commit(j, the next id): every one of the three blocks' LIVE slot (read_ssm_state)
// equals the reference state after step j on its card, and generate(64) (verify(0) + commit through the M = 1 slot list)
// gives the reference's 64 ids. vLLM 3dc6def360's cases: M = 1 (a verify of nothing and commit(0) = one decode step), and a
// ragged sequence of verifies M = 8, 1, 5, 2 with commits j = 6, 0, 2, 1 in turn - the final live states and 16 greedy ids
// after it equal the decode run of the accepted rows. (The mutant committing j + 1 is the Mac's nb_slots_run.)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "bolt_spec_rig.h"
#include "check.h"
#include "runtime/bolt/bolt_sizes.h"

namespace {
namespace rb = runtime::bolt;
constexpr uint32_t kMaxLen = 4096;

std::vector<uint32_t> watched(const model::BoltDesc& d, const model::BoltPlacement& p) {
  uint32_t last0 = 0;
  for (uint32_t l = 0; l < p.end(0); ++l)
    if (d.is_mamba(l)) last0 = d.mamba_index(l);
  return {0, last0, d.mamba_layers() - 1};
}
std::vector<std::vector<float>> states(rb::BoltEngine& e, const std::vector<uint32_t>& mw) {
  std::vector<std::vector<float>> s;
  for (uint32_t b : mw) s.push_back(e.read_ssm_state(b));
  return s;
}
// The decode list from S: every id of `feed` but the last stepped (one replay each), the last left pending - the states after
// the last stepped id, then generate(greedy): [the pending id, the greedy ids after it] (Engine::generate's contract), what a
// commit(j, next) followed by generate gives.
std::vector<uint32_t> plain(rb::BoltEngine& e, const bolt_spec_rig::Session& S, const std::vector<uint32_t>& feed,
                            uint32_t greedy, const std::vector<uint32_t>& mw, std::vector<std::vector<float>>* st) {
  bolt_spec_rig::load(e, S, feed.front());
  for (size_t i = 0; i + 1 < feed.size(); ++i) {
    e.generate(1);
    e.set_token(feed[i + 1]);
  }
  if (st) *st = states(e, mw);
  return e.generate(greedy);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bolt_commit_test <checkpoint> [plain] [b16]\n");
    return 2;
  }
  const std::string snap = bolt_rig::bolt_snapshot(argv[1]);
  if (snap.empty()) {
    std::printf("SKIP: %s is not a Bolt checkpoint\n", argv[1]);
    return 77;
  }
  bolt_rig::Options o;
  o.max_len = kMaxLen;
  rb::VerifyMoe moe = rb::VerifyMoe::Union;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "plain") moe = rb::VerifyMoe::Plain;
    if (a == "b16") o.state = rb::BoltState::Bf16;
  }
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(snap, 0);
  o.devices = pre.layers == 88 ? 2 : 1;
  if ((o.devices == 2 && l0::Context::gpu_count() < 2) || !bolt_rig::fits(pre, o, true)) {
    std::printf("SKIP: the placement cannot run here\n");
    return 77;
  }
  bolt_rig::Rig rig;
  bolt_rig::build(rig, snap, o);
  rb::BoltEngine& e = *rig.eng;
  const model::BoltDesc& d = e.model().desc;
  const std::vector<uint32_t> mw = watched(d, e.model().place);
  std::vector<uint32_t> prompt;
  for (uint32_t i = 0; i < 200; ++i) prompt.push_back((i * 104729u + 7u) % d.vocab);
  const std::vector<uint32_t> t = {17, 1013, 4242, 99, 23456, 777, 1, 65000};
  e.prefill(prompt);
  const bolt_spec_rig::Session S = bolt_spec_rig::save(e);

  // --- the references (25c's decode list) ---
  const std::vector<uint32_t> js = {0, 3, 6, 7};
  std::vector<std::vector<std::vector<float>>> ref_states(js.size());
  std::vector<std::vector<uint32_t>> ref_ids(js.size());
  for (size_t i = 0; i < js.size(); ++i) {
    const uint32_t j = js[i], next = j + 1 < t.size() ? t[j + 1] : 5;
    std::vector<uint32_t> feed(t.begin(), t.begin() + j + 1);
    feed.push_back(next);
    ref_ids[i] = plain(e, S, feed, 64, mw, &ref_states[i]);
  }
  // the ragged sequence's accepted rows: verify inputs [x, drafts], commits j - the rows 0..j of each and the next id
  const std::vector<uint32_t> Ms = {8, 1, 5, 2}, Jr = {6, 0, 2, 1};
  std::vector<std::vector<uint32_t>> ins = {{17, 1013, 4242, 99, 23456, 777, 1, 65000}, {300}, {301, 302, 303, 304, 305},
                                            {401, 402}};
  std::vector<uint32_t> accepted;
  for (size_t it = 0; it < Ms.size(); ++it)
    accepted.insert(accepted.end(), ins[it].begin(), ins[it].begin() + Jr[it] + 1);
  std::vector<std::vector<float>> ragged_states;
  std::vector<uint32_t> after = accepted;
  after.push_back(500);
  const std::vector<uint32_t> ragged_ids = plain(e, S, after, 16, mw, &ragged_states);
  // M = 1: one decode step from S
  std::vector<std::vector<float>> one_state;
  const std::vector<uint32_t> one_ids = plain(e, S, {t[0], t[1]}, 16, mw, &one_state);

  // --- the verify lists ---
  e.enable_spec(moe);
  int bad = 0;
  for (size_t i = 0; i < js.size(); ++i) {
    const uint32_t j = js[i], next = j + 1 < t.size() ? t[j + 1] : 5;
    bolt_spec_rig::load(e, S, t[0]);
    e.verify(t.data() + 1, 7);
    e.commit(j, next);
    CHECK_EQ(e.pos(), S.pos + j + 1);
    if (states(e, mw) != ref_states[i]) {
      std::fprintf(stderr, "commit j=%u of 7: a live slot is not the state after step %u\n", j, j);
      ++bad;
    }
    if (e.generate(64) != ref_ids[i]) {
      std::fprintf(stderr, "commit j=%u of 7: the 64 ids after it are not the plain run's\n", j);
      ++bad;
    }
    std::printf("M 8, commit j = %u (live slot %u): the live states = the decode run's after step %u on %u card(s); 64 ids "
                "after it %s\n", j, e.ssm_live(), j, e.devices(), bad ? "DIFFER" : "= the plain run's");
  }
  {   // 3dc6def360: M = 1
    bolt_spec_rig::load(e, S, t[0]);
    e.verify(nullptr, 0);
    e.commit(0, t[1]);
    if (states(e, mw) != one_state || e.generate(16) != one_ids) {
      std::fprintf(stderr, "M = 1: the slot list's step is not the decode step\n");
      ++bad;
    }
  }
  {   // 3dc6def360: ragged M 8, 1, 5, 2
    bolt_spec_rig::load(e, S, ins[0][0]);
    for (size_t it = 0; it < Ms.size(); ++it) {
      e.set_token(ins[it][0]);
      e.verify(ins[it].data() + 1, Ms[it] - 1);
      const uint32_t next = it + 1 < Ms.size() ? ins[it + 1][0] : 500;
      e.commit(Jr[it], next);
    }
    if (states(e, mw) != ragged_states || e.generate(16) != ragged_ids) {
      std::fprintf(stderr, "ragged M 8 / 1 / 5 / 2: the live states or the ids after it are not the decode run's\n");
      ++bad;
    }
    std::printf("ragged M 8 / 1 / 5 / 2, commits 6 / 0 / 2 / 1: the live states and 16 ids %s the decode run's\n",
                bad ? "DIFFER from" : "=");
  }
  CHECK_EQ(bad, 0);
  std::printf("bolt_commit_test OK (%s, %u device(s), %s MoE, %s state)\n", d.name.c_str(), e.devices(),
              rb::verify_moe_name(moe), rb::bolt_state_name(o.state));
  return 0;
}

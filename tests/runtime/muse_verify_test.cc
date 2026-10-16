// Spec 23f Task 8 on the card (D3, and the commit / prefill-tail halves of Review Focus 3): MuseEngine's verify lists
// at M = 1..16 rows are bitwise the M = 1 decode steps. Box only (labels checkpoint;muse;dflash).
//
//   muse_verify_test <checkpoint> <ids>
// <checkpoint>: the 4-layer synthetic one (muse_oracle.sh synth-ckpt; the drafter's taps at layers 0 / 2) or the real
// one (taps 1 / 13 / 25 / 37 / 49). The prompt is <ids> cycled to P = 64 and P = 2100 positions (2100: every query of
// the block reads across the 2048 window's edge and the 4096-slot ring's rows written since the prompt).
//
//   1. The reference (A): reset, ingest the prompt, 16 plain steps (the M = 1 decode list) - per step the raw logits
//      row, the id, and the K / V rows of positions P .. P + 15 of the first full layer and the first sliding layer.
//   2. D3: for every M = 1..16, reset + ingest, verify([x_P, t_{P+1} .. t_{P+M-1}]) - row r's logits bitwise A's step
//      r, verify_ids()[r] = A's id, the K / V rows of P .. P + M - 1 bitwise; pos untouched by the verify; commit(M - 1,
//      ids[M - 1]) leaves pos P + M and the pending id A's.
//   3. With a drafter (a TapSink fake that records what the engine hands it): every ingest / plain step runs verify(0)
//      + commit(0) - logits bitwise A's (mg_pf_embed / mg_vargmax for mg_embed / argmax), and each commit projects
//      exactly its committed position; a verify at M = 16 writes taps rows bitwise the M = 1 steps' taps of the same
//      positions, and commit(5, .) projects positions P .. P + 5 only (a rejected row's taps never reach the ring).
//   4. The prefill tail: prefill(P = 2100) with the drafter - the projection covers [P - 2048, P) exactly once, 16
//      rows a replay, and the staged taps match the decode path's taps of the same positions to cosine >= 0.999
//      (prefill's linears and attention are decode's to a cosine bar, plan 23d Review Focus 5); with a block hook
//      (P = 4196) every position before a mid-prompt block end is projected before its hook runs (its snapshot's ring).
// Exit 77 (SKIP) when the checkpoint or the ids are absent.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/memory.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/muse_rig.h"
#include "runtime/tap_drafter.h"

namespace {
namespace rm = runtime::muse;
constexpr uint32_t kRows = 16;

// A drafter as the engine sees it: a real tap buffer, and a projection list that does nothing on the device (one
// 4-byte copy) - the host records each call's positions and the tap rows the engine left for it.
struct FakeSink : runtime::TapSink {
  FakeSink(l0::Context& ctx, std::vector<uint32_t> layers, uint32_t hidden)
      : layers_(std::move(layers)), hidden_(hidden),
        taps_(ctx, l0::MemKind::Shared, size_t(kRows) * layers_.size() * hidden * 2),
        dummy_(ctx, l0::MemKind::Device, 8), list_(l0::CmdList::regular(ctx)) {
    list_.copy(static_cast<uint8_t*>(dummy_.ptr()) + 4, dummy_.ptr(), 4);
    list_.close();
  }
  uint32_t max_k() const override { return 15; }
  const std::vector<uint32_t>& tap_layers() const override { return layers_; }
  l0::Mem& taps() override { return taps_; }
  uint32_t window() const override { return 2048; }
  l0::CmdList& projection(uint32_t pos, uint32_t n) override {
    calls.emplace_back(pos, n);
    const size_t row = layers_.size() * size_t(hidden_) * 2;
    for (uint32_t r = 0; r < n; ++r) {
      std::vector<uint16_t> v(row / 2);
      std::memcpy(v.data(), static_cast<const uint8_t*>(taps_.ptr()) + r * row, row);
      rows[pos + r] = std::move(v);
    }
    return list_;
  }
  void reset(l0::CmdList&) override {
    calls.clear();
    rows.clear();
  }
  size_t bytes() const override { return taps_.size(); }
  size_t ring_bytes() const override { return 0; }
  void save_ring(l0::CmdList&, void*) const override {}
  void load_ring(l0::CmdList&, const void*) override {}
  uint64_t key() const override { return 0x23F; }

  std::vector<std::pair<uint32_t, uint32_t>> calls;
  std::map<uint32_t, std::vector<uint16_t>> rows;   // position -> its taps [tap][hidden]

 private:
  std::vector<uint32_t> layers_;
  uint32_t hidden_;
  l0::Mem taps_, dummy_;
  l0::CmdList list_;
};

struct Ref {
  std::vector<std::vector<float>> logits;   // [16] rows: after positions P .. P + 15
  std::vector<uint32_t> ids;                // [17]: x_P = t_P, t_{P+1} .. t_{P+16}
  std::vector<std::vector<uint16_t>> kv;    // [layer pick][k | v] rows of P .. P + 15
};

std::vector<uint32_t> layers_of(const model::MuseGlimmerDesc& d) {
  uint32_t full = d.layers, sl = d.layers;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.is_full(l) && full == d.layers) full = l;
    if (!d.is_full(l) && sl == d.layers) sl = l;
  }
  std::vector<uint32_t> v;
  if (full < d.layers) v.push_back(full);
  if (sl < d.layers) v.push_back(sl);
  return v;
}

std::vector<uint16_t> kv_rows(rm::MuseEngine& e, uint32_t layer, uint32_t first, uint32_t n) {
  std::vector<uint16_t> k = e.read_kv(layer, first, n, false), v = e.read_kv(layer, first, n, true);
  k.insert(k.end(), v.begin(), v.end());
  return k;
}

Ref reference(rm::MuseEngine& e, const std::vector<uint32_t>& prompt) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  e.reset();
  e.ingest(prompt);
  Ref r;
  const uint32_t P = e.pos();
  r.ids.push_back(e.pending());
  for (uint32_t s = 0; s < kRows; ++s) {
    e.generate(1);
    r.logits.push_back(e.read_logits());
    r.ids.push_back(e.pending());
  }
  for (uint32_t l : layers_of(d)) r.kv.push_back(kv_rows(e, l, P, kRows));
  return r;
}

double cosine(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  return aa == 0 && bb == 0 ? 1.0 : ab / std::sqrt(aa * bb);
}

int d3(rm::MuseEngine& e, const std::vector<uint32_t>& prompt, const Ref& A) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  const uint32_t V = d.vocab;
  int bad = 0;
  std::vector<float> host(size_t(kRows) * V);
  for (uint32_t M = 1; M <= kRows; ++M) {
    e.reset();
    e.ingest(prompt);
    const uint32_t P = e.pos();
    bad += e.pending() != A.ids[0];
    e.verify(&A.ids[1], M - 1);
    bad += e.pos() != P;
    e.read_verify_logits(host.data(), M);
    int rows_bad = 0, ids_bad = 0, kv_bad = 0;
    for (uint32_t r = 0; r < M; ++r) {
      rows_bad += std::memcmp(&host[size_t(r) * V], A.logits[r].data(), size_t(V) * 4) != 0;
      ids_bad += e.verify_ids()[r] != A.ids[r + 1];
    }
    const std::vector<uint32_t> ls = layers_of(d);
    for (size_t i = 0; i < ls.size(); ++i) {
      const std::vector<uint16_t> got = kv_rows(e, ls[i], P, M);
      const size_t half = A.kv[i].size() / 2, mrow = got.size() / 2;
      kv_bad += std::memcmp(got.data(), A.kv[i].data(), mrow * 2) != 0 ||
                std::memcmp(got.data() + mrow, A.kv[i].data() + half, mrow * 2) != 0;
    }
    e.commit(M - 1, e.verify_ids()[M - 1]);
    const bool commit_ok = e.pos() == P + M && e.pending() == A.ids[M];
    std::printf("  M %2u at P %u: logits rows %s, ids %s, KV rows %s, commit %s\n", M, P, rows_bad ? "DIFFER" : "bitwise",
                ids_bad ? "DIFFER" : "=", kv_bad ? "DIFFER" : "bitwise", commit_ok ? "ok" : "WRONG");
    bad += rows_bad + ids_bad + kv_bad + !commit_ok;
  }
  return bad;
}

int with_drafter(rm::MuseEngine& e, FakeSink& f, const std::vector<uint32_t>& prompt, const Ref& A) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  int bad = 0;
  // ingest + 16 plain steps through verify(0) + commit(0)
  e.reset();
  e.ingest(prompt);
  const uint32_t P = e.pos();
  bad += f.calls.size() != prompt.size();
  for (size_t i = 0; i < f.calls.size(); ++i) bad += f.calls[i] != std::make_pair(uint32_t(i), 1u);
  bad += e.pending() != A.ids[0];
  std::map<uint32_t, std::vector<uint16_t>> step_taps;
  int logits_bad = 0;
  for (uint32_t s = 0; s < kRows; ++s) {
    e.generate(1);
    logits_bad += e.read_logits() != A.logits[s];
    bad += e.pending() != A.ids[s + 1];
    step_taps[P + s] = f.rows.at(P + s);
  }
  std::printf("  drafter attached: ingest projects one position a commit (%zu calls), plain steps' logits %s\n",
              prompt.size(), logits_bad ? "DIFFER" : "bitwise the decode list's");
  bad += logits_bad;
  // a verify at M = 16: taps rows bitwise the M = 1 steps'; commit(5) projects P .. P + 5 only
  e.reset();
  e.ingest(prompt);
  f.calls.clear();
  f.rows.clear();
  e.verify(&A.ids[1], kRows - 1);
  e.commit(5, e.verify_ids()[5]);
  bad += f.calls.size() != 1 || f.calls[0] != std::make_pair(P, 6u);
  int taps_bad = 0;
  for (uint32_t r = 0; r < 6; ++r) taps_bad += f.rows.at(P + r) != step_taps.at(P + r);
  bad += f.rows.size() != 6 || taps_bad;
  std::printf("  verify M 16 + commit(5): projection (%u, %u), taps rows 0..5 %s the M = 1 steps'\n",
              f.calls.empty() ? 0 : f.calls[0].first, f.calls.empty() ? 0 : f.calls[0].second,
              taps_bad ? "DIFFER from" : "bitwise");
  (void)d;
  return bad;
}

int prefill_tail(rm::MuseEngine& e, FakeSink& f, const std::vector<uint32_t>& prompt) {
  int bad = 0;
  e.reset();
  e.ingest(prompt);   // the decode path's taps of every position
  const std::map<uint32_t, std::vector<uint16_t>> dec = f.rows;
  e.reset();
  e.prefill(prompt);
  const uint32_t P = uint32_t(prompt.size()), lo = P > 2048 ? P - 2048 : 0;
  uint32_t next = lo;
  for (const auto& [pos, n] : f.calls) {
    bad += pos != next || n == 0 || n > kRows;
    next = pos + n;
  }
  bad += next != P;
  double worst = 1;
  for (uint32_t p = lo; p < P; ++p) worst = std::fmin(worst, cosine(f.rows.at(p), dec.at(p)));
  std::printf("  prefill tail at P %u: %zu projections cover [%u, %u) %s; staged taps vs decode's: worst cosine %.6f\n", P,
              f.calls.size(), lo, next, bad ? "WRONGLY" : "exactly", worst);
  bad += worst < 0.999;
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snap = muse_rig::muse_snapshot(argc > 1 ? argv[1] : "");
  if (snap.empty()) {
    std::printf("SKIP: no Muse-Glimmer checkpoint at '%s'\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  const model::MuseGlimmerDesc d0 = loader::muse_checkpoint_desc(snap);
  const std::vector<uint32_t> ids = muse_rig::read_ids(argc > 2 ? argv[2] : "", d0.vocab);
  if (ids.empty()) {
    std::printf("SKIP: no ids at '%s'\n", argc > 2 ? argv[2] : "");
    return 77;
  }
  muse_rig::Rig rig;
  rm::MuseOptions o;
  o.max_len = 8192;
  o.prefill = true;
  o.spec.rows = kRows;   // the verify lists; no drafter term (the fake holds nothing on the device worth planning)
  muse_rig::build(rig, snap, 1, o);
  rm::MuseEngine& e = *rig.eng;
  const model::MuseGlimmerDesc& d = e.weights().desc;
  int bad = 0;
  // A is the plain decode list's (recorded before the drafter is attached: ingest / generate then run verify(0)).
  std::vector<std::vector<uint32_t>> prompts;
  std::vector<Ref> refs;
  for (uint32_t P : {64u, 2100u}) {
    std::vector<uint32_t> prompt(P);
    for (uint32_t i = 0; i < P; ++i) prompt[i] = ids[i % ids.size()];
    std::printf("D3 at P %u (%u layers):\n", P, d.layers);
    refs.push_back(reference(e, prompt));
    bad += d3(e, prompt, refs.back());
    prompts.push_back(std::move(prompt));
  }
  const std::vector<uint32_t> taps = d.layers >= 50 ? std::vector<uint32_t>{1, 13, 25, 37, 49}
                                                    : std::vector<uint32_t>{0, d.layers - 2};
  FakeSink f(e.context(0), taps, d.hidden);
  e.attach_drafter(f);
  for (size_t i = 0; i < prompts.size(); ++i) {
    std::printf("with the drafter at P %zu:\n", prompts[i].size());
    bad += with_drafter(e, f, prompts[i], refs[i]);
  }
  bad += prefill_tail(e, f, prompts.back());
  {   // the final review's fix: with a block hook, a mid-prompt block end's snapshot carries a full ring - every row of a
      // chunk ending at a block end is projected before the hook runs
    std::vector<uint32_t> prompt(4196);
    for (uint32_t i = 0; i < prompt.size(); ++i) prompt[i] = ids[i % ids.size()];
    std::vector<std::pair<uint32_t, size_t>> at_hook;   // (block end, positions projected so far)
    e.set_block_hook([&](uint32_t end, bool) { at_hook.emplace_back(end, f.rows.size()); });
    e.reset();
    e.prefill(prompt);
    e.set_block_hook({});
    bool ok = f.rows.size() == prompt.size() && f.rows.begin()->first == 0 && f.rows.rbegin()->first == prompt.size() - 1;
    for (const auto& [end, n] : at_hook) ok &= end == prompt.size() || n == end;   // [0, E) projected at block end E
    std::printf("  hooked prefill of %zu: %zu positions projected, at the block ends %s\n", prompt.size(), f.rows.size(),
                ok ? "every position before E" : "MISSING positions");
    bad += !ok;
  }
  if (bad) {
    std::printf("muse_verify_test: FAIL (%d)\n", bad);
    return 1;
  }
  std::printf("muse_verify_test: PASS\n");
  return 0;
}

// Spec 26c Task 5 (Decision 7 on the cards): Bolt's prefill stages the drafter's taps and projects each chunk's rows inside the
// drafter's window (runtime::bolt::lili_prefill_r0; 26b's 8-row projection list replayed - the GEMM path is not built). Box only
// (labels checkpoint;bolt;lilicorr).
//   bolt_dflash_prefill_test <bolt checkpoint> <lilicorr snapshot> [--format exact|int8|int4]
//
// For prompts of 4096, 5000 and 300 ids (the ring and the anchor read through LiliDrafter::save_ring):
//   1. chunks of 2048, 1024 and 512 (every split a multiple of 64): the ring and the anchor bitwise equal (the prefill is
//      bitwise one chunk at any such split - 25d - and the projection list is row-local)
//   2. with the block hook (chunks end at 2048 / 4096): the ring the hook sees at 4096 (its snapshot's) is bitwise the ring
//      after a prefill of the first 4096 ids alone - a block end's ring is complete (Decision 7)
//   3. against the decode path (ingest: verify(0) + commit, one projected row a step, decode's taps): every ring row and the
//      anchor at cosine >= 0.999 (prefill's taps are decode's to a cosine bar, spec 25 §6.2 - the bar ESTIMATED; the worst
//      MEASURED cosine printed)
// Exit 77 (SKIP) without two GPUs, the Bolt export or the drafter.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "check.h"
#include "cli/bolt_spec.h"
#include "common/bf16.h"

namespace {
namespace rb = runtime::bolt;

std::vector<uint8_t> ring_of(rb::BoltEngine& e, runtime::dflash::LiliDrafter& d) {
  std::vector<uint8_t> h(d.ring_bytes());
  l0::CmdList imm = l0::CmdList::immediate(e.context(e.devices() - 1));
  d.save_ring(imm, h.data());
  return h;
}

// The worst cosine over the ring's rows (bf16 [layers][K | V][1024][kv x 128]) and the anchor (fp32 [hidden]).
double worst_cosine(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t hidden) {
  const size_t anchor = hidden * 4, ring = a.size() - anchor, row = 256;   // a ring row: 2 kv heads x 128 bf16
  double worst = 1.0;
  auto cos = [&](const double* x, const double* y, size_t n) {
    double xy = 0, xx = 0, yy = 0;
    for (size_t i = 0; i < n; ++i) {
      xy += x[i] * y[i];
      xx += x[i] * x[i];
      yy += y[i] * y[i];
    }
    return xx == 0 && yy == 0 ? 1.0 : xy / std::sqrt(xx * yy + 1e-300);
  };
  std::vector<double> x(row), y(row);
  for (size_t off = 0; off + row * 2 <= ring; off += row * 2) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(a.data() + off);
    const uint16_t* q = reinterpret_cast<const uint16_t*>(b.data() + off);
    for (size_t i = 0; i < row; ++i) {
      x[i] = common::bf16_to_f32(p[i]);
      y[i] = common::bf16_to_f32(q[i]);
    }
    worst = std::min(worst, cos(x.data(), y.data(), row));
  }
  std::vector<double> ax(hidden), ay(hidden);
  const float* fa = reinterpret_cast<const float*>(a.data() + ring);
  const float* fb = reinterpret_cast<const float*>(b.data() + ring);
  for (size_t i = 0; i < hidden; ++i) {
    ax[i] = fa[i];
    ay[i] = fb[i];
  }
  return std::min(worst, cos(ax.data(), ay.data(), hidden));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: bolt_dflash_prefill_test <bolt checkpoint> <lilicorr snapshot> [--format F]\n");
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
  for (int i = 3; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--format") sa.format_arg = argv[++i];
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
  e.enable_spec(rb::VerifyMoe::Union);
  cli::bolt::BoltDrafter dr = cli::bolt::attach_drafter(e, sa, snap);
  int bad = 0;
  for (uint32_t L : {4096u, 5000u, 300u}) {
    std::vector<uint32_t> P;
    for (uint32_t i = 0; i < L; ++i) P.push_back((i * 2654435761u + 11u) % d.vocab);
    std::vector<std::vector<uint8_t>> rings;
    for (uint32_t chunk : {2048u, 1024u, 512u}) {
      e.reset();
      e.set_block_hook({});
      e.prefill(P, chunk);
      rings.push_back(ring_of(e, *dr.drafter));
    }
    if (rings[1] != rings[0] || rings[2] != rings[0]) {
      std::fprintf(stderr, "L %u: the ring / anchor differ between chunks of 2048, 1024 and 512\n", L);
      ++bad;
    }
    if (L > 4096) {   // a block end's ring is complete
      std::vector<uint8_t> at4096;
      e.reset();
      e.set_block_hook([&](uint32_t end, bool block_end) {
        if (block_end && end == 4096) at4096 = ring_of(e, *dr.drafter);
      });
      e.prefill(P);
      e.set_block_hook({});
      e.reset();
      e.prefill(std::vector<uint32_t>(P.begin(), P.begin() + 4096));
      if (at4096.empty() || at4096 != ring_of(e, *dr.drafter)) {
        std::fprintf(stderr, "L %u: the ring at the block end 4096 is not a 4096-id prefill's\n", L);
        ++bad;
      }
    }
    e.reset();   // the decode path: one projected row a step
    e.ingest(P);
    const double c = worst_cosine(rings[0], ring_of(e, *dr.drafter), d.hidden);
    if (c < 0.999) {
      std::fprintf(stderr, "L %u: the prefill's ring vs the decode path's: worst cosine %.6f < 0.999\n", L, c);
      ++bad;
    }
    std::printf("L %u: the ring and the anchor bitwise across chunks 2048 / 1024 / 512%s; vs the decode path worst cosine "
                "%.6f (MEASURED)\n", L, L > 4096 ? ", complete at the block end 4096" : "", c);
  }
  CHECK_EQ(bad, 0);
  std::printf("bolt_dflash_prefill_test OK (%s drafter)\n", loader::draft_format_name(sa.format));
  return 0;
}

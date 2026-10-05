// argmax_run - src/kernels/argmax.cl (both stages) on the Mac's OpenCL GPU against a host
// reference (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   argmax_run [M VOCAB VOCAB_USED]     defaults 3 5000 4990
//
// The cases are the ones argmax.cl argues about in its header: an exact tie at the top
// of row 0 (the LOWER index must win), a larger logit in the masked tail of row 1 (index
// >= VOCAB_USED, must never win), plain random rows after that; then the control-block
// bookkeeping (out_token[m], cur_token = the last one, pos += M). Unlike gemv_i8w this
// is compare-and-copy, so the bar IS exact equality.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"

namespace {
// argmax.cl's control-block offsets, as src/kernels/CMakeLists.txt's CTRL_DEFINES.
constexpr int kPos = 0, kNact = 1, kCur = 2, kOut = 10, kCtrlWords = 64;
}  // namespace

int main(int argc, char** argv) {
  const int M = argc > 3 ? std::atoi(argv[1]) : 3;
  const int V = argc > 3 ? std::atoi(argv[2]) : 5000;
  const int VU = argc > 3 ? std::atoi(argv[3]) : 4990;
  if (M < 1 || M > 8 || V < 1024 || VU < 2 || VU > V) {
    std::fprintf(stderr, "argmax_run: need 1 <= M <= 8, VOCAB >= 1024, 2 <= VOCAB_USED <= VOCAB\n");
    return 2;
  }
  const int groups = (V + 1023) / 1024;
  if (groups > 256) {
    std::fprintf(stderr, "argmax_run: VOCAB > 256 * 1024 does not fit stage 2's one work-group\n");
    return 2;
  }
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ld(-10.f, 10.f);
  std::vector<float> logits(static_cast<size_t>(M) * V);
  for (auto& v : logits) v = ld(rng);
  // Row 0: an exact tie at the top, the two indices in different stage-1 groups (at the
  // defaults), so the tie is settled by stage 2's tree: the lower index must win.
  logits[VU / 3 * 2] = 50.f;
  logits[VU / 3] = 50.f;
  // Row 1: the largest value of the row in the masked tail, and the real maximum below.
  if (M > 1 && VU < V) {
    logits[static_cast<size_t>(V) + V - 1] = 1000.f;
    logits[static_cast<size_t>(V) + 17] = 60.f;
  }

  std::vector<unsigned> want(M);
  for (int m = 0; m < M; ++m) {
    float bv = -INFINITY;
    unsigned bi = 0x7FFFFFFFu;
    for (int k = 0; k < VU; ++k) {
      const float v = logits[static_cast<size_t>(m) * V + k];
      if (v > bv || (v == bv && static_cast<unsigned>(k) < bi)) { bv = v; bi = static_cast<unsigned>(k); }
    }
    want[m] = bi;
  }

  try {
    clrun::Device dev;
    const std::vector<std::string> defs = {
        "CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
        "VOCAB=" + std::to_string(V), "VOCAB_USED=" + std::to_string(VU), "M=" + std::to_string(M)};
    clrun::Program prog(dev, "src/kernels/argmax.cl", defs);
    std::vector<unsigned> ctrl(kCtrlWords, 0);
    ctrl[kPos] = 7;
    ctrl[kNact] = static_cast<unsigned>(M);
    clrun::Buffer bl(dev, logits), bp(dev, static_cast<size_t>(M) * groups * 2 * sizeof(float)),
        bc(dev, ctrl);
    prog.run("argmax_stage1", {static_cast<size_t>(groups) * 256, static_cast<size_t>(M)}, {256, 1},
             bl, bp);
    prog.run("argmax_stage2", {256}, {256}, bc, bp);
    const std::vector<unsigned> got = bc.read<unsigned>();

    int bad = 0;
    for (int m = 0; m < M; ++m) {
      const unsigned g = got[kOut + m];
      std::printf("  row %d: device %u, host %u%s\n", m, g, want[m], g == want[m] ? "" : "  <-- MISMATCH");
      bad += g != want[m];
    }
    if (got[kCur] != want[M - 1]) {
      std::printf("  cur_token %u, want %u  <-- MISMATCH\n", got[kCur], want[M - 1]);
      ++bad;
    }
    if (got[kPos] != 7u + static_cast<unsigned>(M)) {
      std::printf("  pos %u, want %u  <-- MISMATCH\n", got[kPos], 7u + M);
      ++bad;
    }
    std::printf("argmax M=%d VOCAB=%d VOCAB_USED=%d on %s: %s\n", M, V, VU, dev.name().c_str(),
                bad ? "DISAGREES with the host reference (indicative: read the kernel before the box)"
                    : "agrees with the host reference (indicative only; the B70 is the real test)");
    return bad ? 1 : 0;
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "argmax_run: %s\n", e.what());
    return 1;
  }
}

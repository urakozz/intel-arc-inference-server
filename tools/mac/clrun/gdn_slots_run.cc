// gdn_slots_run - spec 19c: the GDN state at the last ACCEPTED token after a verify at M = K + 1 <= 8, on the Mac's
// OpenCL GPU (INDICATIVE ONLY: clrun.h says what this can and cannot show). src/kernels/gdn_step.cl, one GDN layer at
// Qwen3.8's shapes (48 v-heads x 128 x 128), random inputs.
//
// The engine never rewinds a GDN state: its verify list (runtime::build_verify_taps, gdn_step_slots_M<M>_N8) writes
// the state after EVERY row m into slot (live + m) % 8, and committing j accepted drafts is the host setting
// live = (live + j) % 8 (runtime::Engine::commit). The rows past j leave stale slots and stale conv-ring positions
// that nothing reads before they are rewritten. This run holds that to the plain kernel (no SPEC_SLOTS, M = 1)
// stepping the same rows one at a time:
//
//   1. M = 8 from live slot 5 at pos 14 (the conv ring wraps at 16 inside the rows): every slot (5 + m) % 8 and every
//      output row m equal the plain kernel's state / output after row m - bitwise (the same arithmetic, the window
//      from registers instead of the ring).
//   2. Partial acceptance, j = 0, 3, 6 and 7 of 7 drafts: from the M = 8 launch's slots and ring, live = (5 + j) % 8
//      and pos 14 + j + 1, an M = 3 slot launch over three NEW rows (a different continuation - the target's sample
//      after the rejection and two more): its outputs and slots equal the plain kernel run through rows 0..j and then
//      the new rows - bitwise. So the state after the verify, committed at j, is exactly the state at the last
//      accepted token, and the rejected rows' slots and ring positions never reach a later row.
//   3. The same at M = 5 (K = 4), j = 2, from live slot 7 (the slot index wraps).
//
//   gdn_slots_run          (no arguments)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"

namespace {

constexpr int kHeads = 48, kDim = 128, kQkvzN = 16384, kConvRows = 10240, kRing = 16, kAb = 128;
constexpr size_t kState = size_t(kHeads) * kDim * kDim;   // one layer's state, floats
constexpr int kSmall = 41056;                              // conv [10240][4], negA [48] at 40960, dt_bias [48] at 41008
constexpr int kCtrlWords = 64, kPos = 0, kNact = 1, kLive = 19;

std::vector<std::string> defines(int M, bool slots) {
  std::vector<std::string> d = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
                                "NEGA_OFF=40960", "DTBIAS_OFF=41008", "M=" + std::to_string(M)};
  if (slots) {
    d.push_back("SPEC_SLOTS=1");
    d.push_back("CTRL_LIVE=19");
    d.push_back("SPEC_SLOT_STRIDE=" + std::to_string(kState));   // one layer's slot (the engine's: 48 layers)
    d.push_back("N_SLOTS=8");
  }
  return d;
}

struct Rows {   // the per-row inputs: the qkv||z partials and the a||b row
  std::vector<std::vector<float>> qkvz, ab;
};
Rows make_rows(std::mt19937& g, int n) {
  std::normal_distribution<float> nd(0.0f, 0.6f);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  Rows r;
  for (int i = 0; i < n; ++i) {
    std::vector<float> q(kQkvzN), a(kAb, 0.0f);
    for (float& x : q) x = nd(g);
    for (int h = 0; h < 96; ++h) a[h] = ud(g);
    r.qkvz.push_back(q);
    r.ab.push_back(a);
  }
  return r;
}
template <class T>
std::vector<T> concat(const std::vector<std::vector<T>>& v, int from, int n) {
  std::vector<T> out;
  for (int i = from; i < from + n; ++i) out.insert(out.end(), v[i].begin(), v[i].end());
  return out;
}

struct Plain {   // the plain kernel (M = 1, no slots) stepping rows one at a time
  clrun::Device& dev;
  clrun::Program& p1;
  clrun::Buffer& small;
  // Runs `rows` from (state, ring) at pos: returns the state after each row and each row's output.
  void run(std::vector<float> state, std::vector<uint16_t> ring, uint32_t pos, const Rows& r,
           std::vector<std::vector<float>>& states, std::vector<std::vector<float>>& outs) {
    clrun::Buffer bs(dev, state), br(dev, ring), bo(dev, size_t(kHeads) * kDim * 4);
    for (size_t i = 0; i < r.qkvz.size(); ++i) {
      std::vector<uint32_t> ctrl(kCtrlWords, 0);
      ctrl[kPos] = pos + uint32_t(i);
      ctrl[kNact] = 1;
      clrun::Buffer bc(dev, ctrl), bq(dev, r.qkvz[i]), ba(dev, r.ab[i]);
      p1.run("gdn_step", {size_t(kHeads) * 256, 4}, {256, 1}, bc, bq, ba, small, br, bs, bo);
      states.push_back(bs.read<float>());
      outs.push_back(bo.read<float>());
    }
  }
};

int compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    std::printf("  %s: size %zu vs %zu  <-- MISMATCH\n", what, a.size(), b.size());
    return 1;
  }
  size_t diff = 0;
  double ma = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], 4) != 0) {
      ++diff;
      ma = std::fmax(ma, std::fabs(double(a[i]) - b[i]));
    }
  if (diff) std::printf("  %s: %zu of %zu differ (max abs %.3e)  <-- MISMATCH\n", what, diff, a.size(), ma);
  return diff ? 1 : 0;
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    clrun::Program p1(dev, "src/kernels/gdn_step.cl", defines(1, false));
    std::mt19937 g(19);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> small(kSmall, 0.0f);
    for (int i = 0; i < kConvRows * 4; ++i) small[i] = nd(g) * 6.0f;
    for (int h = 0; h < kHeads; ++h) {
      small[40960 + h] = -std::exp(std::uniform_real_distribution<float>(-2.0f, 1.0f)(g));   // -exp(A_log)
      small[41008 + h] = std::uniform_real_distribution<float>(-1.0f, 1.0f)(g);
    }
    std::vector<float> state0(kState);
    for (float& x : state0) x = nd(g);
    std::vector<uint16_t> ring0(size_t(kRing) * kConvRows);
    for (uint16_t& x : ring0) {
      const float f = nd(g) * 10.0f;
      uint32_t u;
      std::memcpy(&u, &f, 4);
      x = uint16_t(u >> 16);
    }
    clrun::Buffer bsmall(dev, small);
    Plain plain{dev, p1, bsmall};
    int bad = 0, cases = 0;

    // One verify at M = k + 1 from live slot L at pos P, then for each j a 3-row continuation from the commit.
    const auto verify_case = [&](int k, uint32_t L, uint32_t P, const std::vector<int>& js) {
      const int M = k + 1;
      clrun::Program pm(dev, "src/kernels/gdn_step.cl", defines(M, true));
      clrun::Program p3(dev, "src/kernels/gdn_step.cl", defines(3, true));
      const Rows rows = make_rows(g, M);
      std::vector<std::vector<float>> ref_s, ref_o;
      plain.run(state0, ring0, P, rows, ref_s, ref_o);
      // the slots: the initial state in slot L
      std::vector<float> s0 = L == 0 ? state0 : std::vector<float>(kState, 0.0f);
      std::vector<float> spec(size_t(7) * kState, 0.0f);
      if (L != 0) std::copy(state0.begin(), state0.end(), spec.begin() + size_t(L - 1) * kState);
      std::vector<uint32_t> ctrl(kCtrlWords, 0);
      ctrl[kPos] = P;
      ctrl[kNact] = uint32_t(M);
      ctrl[kLive] = L;
      clrun::Buffer bc(dev, ctrl), bq(dev, concat(rows.qkvz, 0, M)), ba(dev, concat(rows.ab, 0, M)), br(dev, ring0),
          bs(dev, s0), bsp(dev, spec), bo(dev, size_t(M) * kHeads * kDim * 4);
      pm.run("gdn_step", {size_t(kHeads) * 256, 4}, {256, 1}, bc, bq, ba, bsmall, br, bs, bo, bsp);
      const std::vector<float> got_s0 = bs.read<float>(), got_spec = bsp.read<float>(), got_o = bo.read<float>();
      const std::vector<uint16_t> ring_after = br.read<uint16_t>();
      const auto slot = [&](uint32_t s) {
        return s == 0 ? got_s0
                      : std::vector<float>(got_spec.begin() + size_t(s - 1) * kState,
                                           got_spec.begin() + size_t(s) * kState);
      };
      int b = 0;
      for (int m = 0; m < M; ++m) {
        const std::string w = "M=" + std::to_string(M) + " row " + std::to_string(m);
        b += compare((w + " slot").c_str(), slot((L + m) % 8), ref_s[m]);
        b += compare((w + " output").c_str(),
                     std::vector<float>(got_o.begin() + size_t(m) * kHeads * kDim,
                                        got_o.begin() + size_t(m + 1) * kHeads * kDim),
                     ref_o[m]);
      }
      std::printf("verify M=%d from live %u at pos %u: every row's slot and output %s the plain kernel's\n", M, L, P,
                  b ? "DIFFER from" : "equal");
      bad += b;
      ++cases;
      // the commits
      for (int j : js) {
        const Rows next = make_rows(g, 3);
        // the reference: plain through rows 0..j, then the new rows
        Rows path;
        for (int m = 0; m <= j; ++m) {
          path.qkvz.push_back(rows.qkvz[m]);
          path.ab.push_back(rows.ab[m]);
        }
        for (int m = 0; m < 3; ++m) {
          path.qkvz.push_back(next.qkvz[m]);
          path.ab.push_back(next.ab[m]);
        }
        std::vector<std::vector<float>> want_s, want_o;
        plain.run(state0, ring0, P, path, want_s, want_o);
        // the engine: live = (L + j) % 8 over the verify's slots and ring, pos P + j + 1, M = 3
        const uint32_t live = (L + uint32_t(j)) % 8;
        std::vector<uint32_t> c2(kCtrlWords, 0);
        c2[kPos] = P + uint32_t(j) + 1;
        c2[kNact] = 3;
        c2[kLive] = live;
        clrun::Buffer bc2(dev, c2), bq2(dev, concat(next.qkvz, 0, 3)), ba2(dev, concat(next.ab, 0, 3)),
            br2(dev, ring_after), bs2(dev, got_s0), bsp2(dev, got_spec), bo2(dev, size_t(3) * kHeads * kDim * 4);
        p3.run("gdn_step", {size_t(kHeads) * 256, 4}, {256, 1}, bc2, bq2, ba2, bsmall, br2, bs2, bo2, bsp2);
        const std::vector<float> s0b = bs2.read<float>(), spb = bsp2.read<float>(), ob = bo2.read<float>();
        const auto slot2 = [&](uint32_t s) {
          return s == 0 ? s0b
                        : std::vector<float>(spb.begin() + size_t(s - 1) * kState, spb.begin() + size_t(s) * kState);
        };
        int bj = compare("committed state", slot(live), want_s[j]);
        for (int m = 0; m < 3; ++m) {
          const std::string w = "j=" + std::to_string(j) + " next row " + std::to_string(m);
          bj += compare((w + " slot").c_str(), slot2((live + uint32_t(m)) % 8), want_s[j + 1 + m]);
          bj += compare((w + " output").c_str(),
                        std::vector<float>(ob.begin() + size_t(m) * kHeads * kDim,
                                           ob.begin() + size_t(m + 1) * kHeads * kDim),
                        want_o[j + 1 + m]);
        }
        std::printf("  commit j=%d of %d (live %u -> %u, pos %u): the live state is the plain kernel's after row %d, "
                    "and 3 new rows from it %s\n",
                    j, k, L, live, P + j + 1, j, bj ? "DIFFER" : "equal the plain run's");
        bad += bj;
        ++cases;
      }
    };
    verify_case(7, 5, 14, {0, 3, 6, 7});
    verify_case(4, 7, 29, {2});
    std::printf("gdn_step slots (N_SLOTS 8) on %s: %d cases, %s\n", dev.name().c_str(), cases,
                bad ? "DISAGREE with the plain kernel (indicative: read the kernel before the box)"
                    : "agree bitwise with the plain kernel (indicative only; the B70 is the real test)");
    return bad ? 1 : 0;
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "gdn_slots_run: %s\n", e.what());
    return 1;
  }
}

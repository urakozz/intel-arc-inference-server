// nb_slots_run - spec 26c Task 2, Review Focus 1: Bolt's Mamba state at the last ACCEPTED token after a verify at M = K + 1
// <= 8, on the Mac's OpenCL GPU (INDICATIVE ONLY: clrun.h says what this can and cannot show). src/kernels/bolt/nb_mamba.cl,
// one Mamba block at Bolt's shapes (128 heads x 64 x state 128, conv 4 over the 10240 xBC channels), random inputs - 19c's
// gdn_slots_run carried to Bolt.
//
// The engine never rewinds an SSM state: its verify lists (runtime::bolt build_verify) run nb_conv at M rows (the raw-xBC
// ring's slots pos .. pos + M - 1) and nb_ssm's slot build (`_SL8`: SPEC_SLOTS 8, the state after row m into slot (live + m)
// % 8, slot 0 the block's own state, slots 1..7 the block's 7 extra copies side by side - SPEC_SLOT_STRIDE one state), and
// committing j accepted drafts is the host setting Control::gdn_live = (live + j) % 8 on both cards (BoltEngine::commit).
// The rows past j leave stale slots and stale ring columns that nothing reads before they are rewritten. This run holds
// that to the plain kernels (nb_conv / nb_ssm at M = 1, in place, no slots) stepping the same rows one at a time:
//
//   1. M = 8 from live slot 5 at pos 14 (the ring wraps at 16 inside the rows): every slot (5 + m) % 8 and every y row equal
//      the plain kernels' state / output after row m - bitwise. Every accepted count j = 0 .. 7: from the verify's slots and
//      ring, live (5 + j) % 8 and pos 14 + j + 1, three NEW rows through the M = 3 slot build equal the plain run through
//      rows 0..j and then the new rows - and the j + 1 mutant (live one slot further) is caught (its y differs).
//   2. M = 5 (K = 4) from live 7 (the slot index wraps), j = 2 of 4; the bf16-state arm (SSM_B16) at M = 4, j = 1.
//   3. vLLM 3dc6def360's cases: M = 1 (the drafter's plain token through the M = 1 slot build) equals the plain step; a
//      ragged sequence of verifies M = 8, 1, 5, 2 with commits j = 6, 0, 2, 1 in turn, each from the previous commit's
//      live slot and ring - every verify row's y and slot, and the final live state, equal one plain run of the accepted
//      rows (the rejected rows of each verify are never stepped by the plain run).
//
//   nb_slots_run           (no arguments)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"

namespace {

constexpr int kH = 128, kP = 64, kN = 128, kXbc = 10240, kInner = 8192, kInN = 18560, kRing = 16;
constexpr size_t kState = size_t(kH) * kP * kN;   // one block's state, elements (SPEC_SLOT_STRIDE)
constexpr int kCtrlWords = 32, kPos = 0, kNact = 1, kLive = 19;
constexpr uint32_t kSlots = 8;

const std::vector<std::string> kCtrl = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18"};
std::vector<std::string> conv_defs(int M) {
  std::vector<std::string> d = kCtrl;
  d.push_back("M=" + std::to_string(M));
  d.push_back("CONV_S=1");
  return d;
}
std::vector<std::string> ssm_defs(int M, bool slots, bool b16) {
  std::vector<std::string> d = kCtrl;
  d.push_back("M=" + std::to_string(M));
  d.push_back("SSM_S=1");
  d.push_back(std::string("SSM_B16=") + (b16 ? "1" : "0"));
  d.push_back("SSM_CONV=0");
  if (slots) {
    d.push_back("SPEC_SLOTS=8");
    d.push_back("CTRL_LIVE=19");
    d.push_back("SPEC_SLOT_STRIDE=" + std::to_string(kState));
  }
  return d;
}

using Row = std::vector<float>;   // one row's in_proj output [18560] (S 1)

uint16_t bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

struct Mixer {
  clrun::Device& dev;
  clrun::Buffer &cw, &cb, &A, &D, &dtb;
  bool b16;
  size_t esz() const { return b16 ? 2 : 4; }
};

// The state as raw bytes (fp32 or bf16 elements) so both arms compare bitwise.
using Bytes = std::vector<uint8_t>;

// The plain kernels (M = 1, in place) stepping `rows` from (state, ring) at pos: the state after each row, each row's y.
void plain_run(Mixer& mx, Bytes state, std::vector<uint16_t> ring, uint32_t pos, const std::vector<Row>& rows,
               std::vector<Bytes>& states, std::vector<std::vector<uint16_t>>& ys) {
  clrun::Program pc(mx.dev, "src/kernels/bolt/nb_mamba.cl", conv_defs(1));
  clrun::Program ps(mx.dev, "src/kernels/bolt/nb_mamba.cl", ssm_defs(1, false, mx.b16));
  clrun::Buffer bs(mx.dev, state), br(mx.dev, ring), xc(mx.dev, size_t(kXbc) * 2), y(mx.dev, size_t(kInner) * 2);
  for (size_t i = 0; i < rows.size(); ++i) {
    std::vector<uint32_t> ctrl(kCtrlWords, 0);
    ctrl[kPos] = pos + uint32_t(i);
    ctrl[kNact] = 1;
    clrun::Buffer bc(mx.dev, ctrl), sl(mx.dev, rows[i]);
    pc.run("nb_conv", {size_t(kXbc)}, {256}, bc, sl, mx.cw, mx.cb, br, xc);
    ps.run("nb_ssm", {size_t(512) * 256}, {256}, bc, sl, xc, mx.A, mx.D, mx.dtb, bs, y);
    states.push_back(bs.read<uint8_t>());
    ys.push_back(y.read<uint16_t>());
  }
}

// The slot state: slot 0 (the block's own) and slots 1..7 side by side.
struct Slots {
  Bytes s0, spec;
  Bytes slot(uint32_t s, size_t esz) const {
    if (s == 0) return s0;
    return Bytes(spec.begin() + long((s - 1) * kState * esz), spec.begin() + long(s * kState * esz));
  }
};

// One verify of M rows through nb_conv (M) and the slot build of nb_ssm (M), from live `live` at pos.
void verify(Mixer& mx, int M, uint32_t live, uint32_t pos, const std::vector<Row>& rows, Slots& st,
            std::vector<uint16_t>& ring, std::vector<std::vector<uint16_t>>& ys) {
  clrun::Program pc(mx.dev, "src/kernels/bolt/nb_mamba.cl", conv_defs(M));
  clrun::Program ps(mx.dev, "src/kernels/bolt/nb_mamba.cl", ssm_defs(M, true, mx.b16));
  std::vector<float> slices;
  for (const Row& r : rows) slices.insert(slices.end(), r.begin(), r.end());
  std::vector<uint32_t> ctrl(kCtrlWords, 0);
  ctrl[kPos] = pos;
  ctrl[kNact] = uint32_t(M);
  ctrl[kLive] = live;
  clrun::Buffer bc(mx.dev, ctrl), sl(mx.dev, slices), br(mx.dev, ring), xc(mx.dev, size_t(M) * kXbc * 2),
      y(mx.dev, size_t(M) * kInner * 2), b0(mx.dev, st.s0), bsp(mx.dev, st.spec);
  pc.run("nb_conv", {size_t(kXbc)}, {256}, bc, sl, mx.cw, mx.cb, br, xc);
  ps.run("nb_ssm", {size_t(512) * 256}, {256}, bc, sl, xc, mx.A, mx.D, mx.dtb, b0, y, bsp);
  st.s0 = b0.read<uint8_t>();
  st.spec = bsp.read<uint8_t>();
  ring = br.read<uint16_t>();
  const std::vector<uint16_t> yall = y.read<uint16_t>();
  ys.clear();
  for (int m = 0; m < M; ++m) ys.emplace_back(yall.begin() + long(m) * kInner, yall.begin() + long(m + 1) * kInner);
}

int compare(const std::string& what, const Bytes& a, const Bytes& b) {
  if (a != b) {
    size_t diff = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) diff += a[i] != b[i];
    std::printf("  %s: %zu of %zu bytes differ  <-- MISMATCH\n", what.c_str(), diff, a.size());
    return 1;
  }
  return 0;
}
int compare(const std::string& what, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  if (a != b) {
    std::printf("  %s: differs  <-- MISMATCH\n", what.c_str());
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::mt19937 g(26);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<float> cw(size_t(kXbc) * 4), cb(kXbc), A(kH), D(kH), dtb(kH);
    for (float& x : cw) x = nd(g) * 0.6f;
    for (float& x : cb) x = nd(g) * 0.2f;
    for (int h = 0; h < kH; ++h) {
      A[h] = -std::exp(std::uniform_real_distribution<float>(-2.0f, 1.0f)(g));   // -exp(A_log)
      D[h] = ud(g);
      dtb[h] = ud(g) - 2.0f;
    }
    clrun::Buffer bcw(dev, cw), bcb(dev, cb), bA(dev, A), bD(dev, D), bdt(dev, dtb);
    const auto make_rows = [&](int n) {
      std::vector<Row> v;
      for (int i = 0; i < n; ++i) {
        Row r(kInN);
        for (float& x : r) x = nd(g);
        v.push_back(r);
      }
      return v;
    };
    const auto make_state = [&](bool b16) {
      Bytes s(kState * (b16 ? 2 : 4));
      for (size_t i = 0; i < kState; ++i) {
        const float f = nd(g) * 0.1f;
        if (b16) {
          const uint16_t h = bf16(f);
          std::memcpy(&s[i * 2], &h, 2);
        } else {
          std::memcpy(&s[i * 4], &f, 4);
        }
      }
      return s;
    };
    std::vector<uint16_t> ring0(size_t(kRing) * kXbc);
    for (uint16_t& x : ring0) x = bf16(nd(g) * 2.0f);
    int bad = 0, cases = 0;

    // A verify at M from live L at pos P, then for each j a 3-row continuation from the commit, and the j + 1 mutant.
    const auto verify_case = [&](bool b16, int M, uint32_t L, uint32_t P, const std::vector<int>& js) {
      Mixer mx{dev, bcw, bcb, bA, bD, bdt, b16};
      const size_t es = mx.esz();
      const Bytes state0 = make_state(b16);
      const std::vector<Row> rows = make_rows(M);
      std::vector<Bytes> ref_s;
      std::vector<std::vector<uint16_t>> ref_y;
      plain_run(mx, state0, ring0, P, rows, ref_s, ref_y);
      Slots st{L == 0 ? state0 : Bytes(kState * es, 0), Bytes(7 * kState * es, 0)};
      if (L != 0) std::copy(state0.begin(), state0.end(), st.spec.begin() + long((L - 1) * kState * es));
      std::vector<uint16_t> ring = ring0;
      std::vector<std::vector<uint16_t>> ys;
      verify(mx, M, L, P, rows, st, ring, ys);
      int b = 0;
      for (int m = 0; m < M; ++m) {
        const std::string w = std::string(b16 ? "bf16 " : "") + "M=" + std::to_string(M) + " row " + std::to_string(m);
        b += compare(w + " slot", st.slot((L + uint32_t(m)) % kSlots, es), ref_s[size_t(m)]);
        b += compare(w + " y", ys[size_t(m)], ref_y[size_t(m)]);
      }
      std::printf("%sverify M=%d from live %u at pos %u: every row's slot and y %s the plain kernels'\n", b16 ? "bf16 " : "",
                  M, L, P, b ? "DIFFER from" : "equal");
      bad += b;
      ++cases;
      for (int j : js) {
        const std::vector<Row> next = make_rows(3);
        std::vector<Row> path(rows.begin(), rows.begin() + j + 1);
        path.insert(path.end(), next.begin(), next.end());
        std::vector<Bytes> want_s;
        std::vector<std::vector<uint16_t>> want_y;
        plain_run(mx, state0, ring0, P, path, want_s, want_y);
        int bj = 0;
        for (int mut = 0; mut < 2; ++mut) {   // 0: the commit; 1: the j + 1 mutant (when there is a row j + 1)
          if (mut && j + 1 >= M) continue;
          const uint32_t live = (L + uint32_t(j + mut)) % kSlots;
          Slots s2 = st;
          std::vector<uint16_t> r2 = ring;
          std::vector<std::vector<uint16_t>> y2;
          verify(mx, 3, live, P + uint32_t(j) + 1, next, s2, r2, y2);
          int d = mut ? 0 : compare("committed state", st.slot(live, es), want_s[size_t(j)]);
          for (int m = 0; m < 3; ++m) {
            const std::string w = "j=" + std::to_string(j) + " next row " + std::to_string(m);
            if (mut) {
              d += y2[size_t(m)] != want_y[size_t(j + 1 + m)] ? 1 : 0;
              continue;
            }
            d += compare(w + " slot", s2.slot((live + uint32_t(m)) % kSlots, es), want_s[size_t(j + 1 + m)]);
            d += compare(w + " y", y2[size_t(m)], want_y[size_t(j + 1 + m)]);
          }
          if (mut && d == 0) {
            std::printf("  j=%d: the j + 1 mutant (live %u) was NOT caught  <-- MISMATCH\n", j, live);
            ++bj;
          }
          if (!mut) bj += d;
        }
        std::printf("  commit j=%d of %d (live %u -> %u, pos %u): the live state is the plain kernels' after row %d, 3 new "
                    "rows from it %s; the j + 1 mutant %s\n",
                    j, M - 1, L, (L + uint32_t(j)) % kSlots, P + j + 1, j, bj ? "DIFFER" : "equal the plain run's",
                    j + 1 < M ? "caught" : "(no row j + 1)");
        bad += bj;
        ++cases;
      }
    };
    verify_case(false, 8, 5, 14, {0, 1, 2, 3, 4, 5, 6, 7});
    verify_case(false, 5, 7, 29, {2});
    verify_case(true, 4, 3, 6, {1});
    verify_case(false, 1, 4, 40, {0});   // 3dc6def360: a query of length 1 through the slot build

    {   // 3dc6def360's ragged case: verifies M = 8, 1, 5, 2 with commits j = 6, 0, 2, 1, each from the last commit
      Mixer mx{dev, bcw, bcb, bA, bD, bdt, false};
      const Bytes state0 = make_state(false);
      Slots st{state0, Bytes(7 * kState * 4, 0)};
      std::vector<uint16_t> ring = ring0;
      uint32_t live = 0, pos = 9;
      std::vector<Row> accepted;   // the plain run's rows: every verify's rows 0..j
      int b = 0;
      const int Ms[] = {8, 1, 5, 2}, Js[] = {6, 0, 2, 1};
      for (int it = 0; it < 4; ++it) {
        const std::vector<Row> rows = make_rows(Ms[it]);
        std::vector<std::vector<uint16_t>> ys;
        verify(mx, Ms[it], live, pos, rows, st, ring, ys);
        // the reference: one plain run of every row accepted so far, then this verify's rows
        std::vector<Row> path = accepted;
        path.insert(path.end(), rows.begin(), rows.end());
        std::vector<Bytes> want_s;
        std::vector<std::vector<uint16_t>> want_y;
        plain_run(mx, state0, ring0, 9, path, want_s, want_y);
        for (int m = 0; m < Ms[it]; ++m) {
          const size_t at = accepted.size() + size_t(m);
          b += compare("ragged verify " + std::to_string(it) + " row " + std::to_string(m) + " y", ys[size_t(m)], want_y[at]);
          b += compare("ragged verify " + std::to_string(it) + " row " + std::to_string(m) + " slot",
                       st.slot((live + uint32_t(m)) % kSlots, 4), want_s[at]);
        }
        accepted.insert(accepted.end(), rows.begin(), rows.begin() + Js[it] + 1);
        live = (live + uint32_t(Js[it])) % kSlots;
        pos += uint32_t(Js[it]) + 1;
      }
      std::vector<Bytes> fin_s;
      std::vector<std::vector<uint16_t>> fin_y;
      plain_run(mx, state0, ring0, 9, accepted, fin_s, fin_y);
      b += compare("ragged: the final live state", st.slot(live, 4), fin_s.back());
      std::printf("ragged M 8 / 1 / 5 / 2, commits 6 / 0 / 2 / 1 (pos 9 -> %u, live %u): every row and the final live "
                  "state %s one plain run of the %zu accepted rows\n",
                  pos, live, b ? "DIFFER from" : "equal", accepted.size());
      bad += b;
      ++cases;
    }
    std::printf("nb_ssm slots (SPEC_SLOTS 8) + nb_conv at M rows on %s: %d cases, %s\n", dev.name().c_str(), cases,
                bad ? "DISAGREE with the plain kernels (indicative: read the kernels before the box)"
                    : "agree bitwise with the plain kernels (indicative only; the B70 is the real test)");
    return bad ? 1 : 0;
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "nb_slots_run: %s\n", e.what());
    return 1;
  }
}

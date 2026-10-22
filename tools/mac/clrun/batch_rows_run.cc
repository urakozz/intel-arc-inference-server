// batch_rows_run - spec 13a: the BATCH_ROWS builds of gdn_step.cl, attn.cl's attn_prep and kv8.cl's attn_prep_kv8 -
// rows of DIFFERENT sequences in one launch - against the production kernels stepping each sequence alone, on the
// Mac's OpenCL GPU (INDICATIVE ONLY: clrun.h says what this can and cannot show). Qwen3.8's shapes, one layer, random
// inputs. The attention pair (attn_v2.cl, kv8.cl's KV8_DEC) uses shuffles and 2D block reads the Mac cannot run: its
// row build is held on the box (probe_batch_step's check, queue row 66).
//
//   1. gdn_step: four slots, each with its own GDN state and conv ring (slot 0 the `state` / `conv_ring` arguments,
//      slots 1..3 the pools at one state / one ring a slot), positions 14 (the ring wraps inside a run), 2 (the
//      window reaches below position 0), 300 and 47. Three tables - four 1-row runs; two 4-row runs (an MTP-shaped
//      step); runs of 3, 1 and 2 rows in slot order 0, 2, 1 - each against the plain M = 1 kernel stepping the same
//      rows slot by slot from the slot's state and ring: every row's output, every row's dst state and every slot's
//      ring after the step bitwise.
//   2. attn_prep (bf16) and attn_prep_kv8: five rows of three slots in a 3 x 512-position cache (runs of 2, 1, 2 at
//      positions 100, 5 and 400), against the M = 1 kernel per row over that slot's 512 positions alone: attn_q,
//      attn_gate and the K / V rows (and at int8 their scales) bitwise, and every other cache row untouched.
//
//   batch_rows_run          (no arguments)
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
constexpr size_t kRingN = size_t(kRing) * kConvRows;      // one layer's ring, ushorts
constexpr int kSmall = 41056;
constexpr int kCtrlWords = 64, kPos = 0, kNact = 1;
constexpr int kRowWords = 8;   // runtime::BatchRow
const std::vector<std::string> kCtrl = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18"};
const std::vector<std::string> kRows = {"BATCH_ROWS=1", "BROW_WORDS=8", "BROW_POS=0", "BROW_KVB=1",
                                        "BROW_SRC=2",   "BROW_DST=3",   "BROW_SLOT=4"};

std::vector<std::string> join(std::vector<std::string> a, const std::vector<std::string>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

int compare(const std::string& what, const void* a, const void* b, size_t bytes) {
  if (std::memcmp(a, b, bytes) == 0) return 0;
  size_t diff = 0;
  for (size_t i = 0; i < bytes; ++i) diff += static_cast<const uint8_t*>(a)[i] != static_cast<const uint8_t*>(b)[i];
  std::printf("  %s: %zu of %zu bytes differ  <-- MISMATCH\n", what.c_str(), diff, bytes);
  return 1;
}

struct Row {   // runtime::BatchRow
  uint32_t pos, kv_base, src, dst, slot;
};
std::vector<uint32_t> table(const std::vector<Row>& rows) {
  std::vector<uint32_t> t(size_t(8) * kRowWords, 0);
  for (size_t m = 0; m < rows.size(); ++m) {
    t[m * kRowWords + 0] = rows[m].pos;
    t[m * kRowWords + 1] = rows[m].kv_base;
    t[m * kRowWords + 2] = rows[m].src;
    t[m * kRowWords + 3] = rows[m].dst;
    t[m * kRowWords + 4] = rows[m].slot;
  }
  return t;
}

// --- 1. gdn_step ------------------------------------------------------------------------------------------------
int gdn_case(clrun::Device& dev, clrun::Program& plain, clrun::Buffer& small, std::mt19937& g, const char* name,
             const std::vector<std::vector<float>>& states, const std::vector<std::vector<uint16_t>>& rings,
             const std::vector<uint32_t>& slot_pos, const std::vector<std::pair<uint32_t, uint32_t>>& runs) {
  std::normal_distribution<float> nd(0.0f, 0.6f);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  const uint32_t S = uint32_t(states.size());
  std::vector<Row> rows;
  for (const auto& r : runs)   // (slot, rows): src = the slot's index, dst = S + the row's index
    for (uint32_t j = 0; j < r.second; ++j)
      rows.push_back({slot_pos[r.first] + j, 0, r.first, S + uint32_t(rows.size()), r.first});
  const uint32_t M = uint32_t(rows.size());
  std::vector<std::vector<float>> qkvz(M, std::vector<float>(kQkvzN)), ab(M, std::vector<float>(kAb, 0.0f));
  for (auto& q : qkvz)
    for (float& x : q) x = nd(g);
  for (auto& a : ab)
    for (int h = 0; h < 96; ++h) a[h] = ud(g);
  std::vector<float> q_all, a_all;
  for (uint32_t m = 0; m < M; ++m) {
    q_all.insert(q_all.end(), qkvz[m].begin(), qkvz[m].end());
    a_all.insert(a_all.end(), ab[m].begin(), ab[m].end());
  }
  // The batched launch: state index 0 = slot 0's state, the pool = slots 1..S-1 then one dst a row.
  std::vector<float> pool(size_t(S - 1 + M) * kState, 0.0f);
  for (uint32_t s = 1; s < S; ++s) std::copy(states[s].begin(), states[s].end(), pool.begin() + size_t(s - 1) * kState);
  std::vector<uint16_t> rpool(size_t(S - 1) * kRingN);
  for (uint32_t s = 1; s < S; ++s) std::copy(rings[s].begin(), rings[s].end(), rpool.begin() + size_t(s - 1) * kRingN);
  clrun::Program pm(dev, "src/kernels/gdn_step.cl",
                    join(join(kCtrl, kRows), {"NEGA_OFF=40960", "DTBIAS_OFF=41008", "M=" + std::to_string(M),
                                              "SPEC_SLOT_STRIDE=" + std::to_string(kState),
                                              "RING_SLOT_STRIDE=" + std::to_string(kRingN)}));
  std::vector<uint32_t> ctrl(kCtrlWords, 0);
  ctrl[kPos] = 0xDEADu;   // nothing in the row build reads Control::pos
  ctrl[kNact] = M;
  clrun::Buffer bc(dev, ctrl), bq(dev, q_all), ba(dev, a_all), br0(dev, rings[0]), bs0(dev, states[0]),
      bo(dev, size_t(M) * kHeads * kDim * 4), bpool(dev, pool), brpool(dev, rpool), bt(dev, table(rows));
  pm.run("gdn_step", {size_t(kHeads) * 256, 4}, {256, 1}, bc, bq, ba, small, br0, bs0, bo, bpool, brpool, bt);
  const std::vector<float> got_o = bo.read<float>(), got_pool = bpool.read<float>(), got_s0 = bs0.read<float>();
  const std::vector<uint16_t> got_r0 = br0.read<uint16_t>(), got_rp = brpool.read<uint16_t>();
  int bad = 0;
  if (compare("state index 0 (slot 0's src, never a dst)", got_s0.data(), states[0].data(), kState * 4)) ++bad;
  // The reference: the plain M = 1 kernel over each run, from the slot's own state and ring.
  uint32_t m0 = 0;
  for (const auto& r : runs) {
    const uint32_t s = r.first;
    clrun::Buffer rs(dev, states[s]), rr(dev, rings[s]), ro(dev, size_t(kHeads) * kDim * 4);
    for (uint32_t j = 0; j < r.second; ++j) {
      const uint32_t m = m0 + j;
      std::vector<uint32_t> c(kCtrlWords, 0);
      c[kPos] = slot_pos[s] + j;
      c[kNact] = 1;
      clrun::Buffer rc(dev, c), rq(dev, qkvz[m]), ra(dev, ab[m]);
      plain.run("gdn_step", {size_t(kHeads) * 256, 4}, {256, 1}, rc, rq, ra, small, rr, rs, ro);
      const std::vector<float> want_s = rs.read<float>(), want_o = ro.read<float>();
      const std::string w = std::string(name) + " row " + std::to_string(m) + " (slot " + std::to_string(s) + ")";
      bad += compare(w + " output", got_o.data() + size_t(m) * kHeads * kDim, want_o.data(), want_o.size() * 4);
      bad += compare(w + " dst state", got_pool.data() + size_t(S - 1 + m) * kState, want_s.data(), kState * 4);
    }
    const std::vector<uint16_t> want_r = rr.read<uint16_t>();
    const uint16_t* got_r = s == 0 ? got_r0.data() : got_rp.data() + size_t(s - 1) * kRingN;
    bad += compare(std::string(name) + " slot " + std::to_string(s) + "'s ring", got_r, want_r.data(), kRingN * 2);
    m0 += r.second;
  }
  for (uint32_t s = 1; s < S; ++s)   // the srcs in the pool are never written
    bad += compare(std::string(name) + " slot " + std::to_string(s) + "'s src state",
                   got_pool.data() + size_t(s - 1) * kState, states[s].data(), kState * 4);
  std::printf("gdn_step rows, %s (M = %u): every row's output and dst state, every slot's ring %s\n", name, M,
              bad ? "DIFFER from the plain kernel's" : "equal the plain kernel's run of that slot alone");
  return bad;
}

// --- 2. attn_prep / attn_prep_kv8 -------------------------------------------------------------------------------
constexpr int kQh = 24, kKvh = 4, kHd = 256, kQkvN = 14336, kLen = 512, kSlots = 3;
int prep_case(clrun::Device& dev, std::mt19937& g, bool kv8) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const std::vector<Row> rows = {{100, 1 * kLen, 0, 0, 1}, {101, 1 * kLen, 0, 0, 1}, {5, 0, 0, 0, 0},
                                 {400, 2 * kLen, 0, 0, 2}, {401, 2 * kLen, 0, 0, 2}};
  const uint32_t M = uint32_t(rows.size());
  // q||gate (12288) || k || v, two split-K slices [2][M][14336]
  std::vector<std::vector<float>> qkv(M, std::vector<float>(2 * kQkvN));
  for (auto& r : qkv)
    for (float& x : r) x = nd(g) * 0.5f;
  std::vector<float> part(size_t(2) * M * kQkvN);
  for (uint32_t s = 0; s < 2; ++s)
    for (uint32_t m = 0; m < M; ++m)
      std::copy(qkv[m].begin() + size_t(s) * kQkvN, qkv[m].begin() + size_t(s + 1) * kQkvN,
                part.begin() + (size_t(s) * M + m) * kQkvN);
  std::vector<float> small(512);
  for (float& x : small) x = 1.0f + nd(g) * 0.1f;
  std::vector<float> rope(size_t(kLen) * 64);
  for (int p = 0; p < kLen; ++p)
    for (int i = 0; i < 32; ++i) {
      const double a = p * std::pow(10000000.0, -double(i) / 32.0);
      rope[size_t(p) * 64 + i] = float(std::cos(a));
      rope[size_t(p) * 64 + 32 + i] = float(std::sin(a));
    }
  const size_t row_b = size_t(kKvh) * kHd * (kv8 ? 1 : 2);   // one position's K (or V) bytes
  const size_t total = size_t(kSlots) * kLen;
  std::vector<uint8_t> kfill(total * row_b);
  for (size_t i = 0; i < kfill.size(); ++i) kfill[i] = uint8_t(i * 131 + 7);   // the untouched rows' sentinel
  std::vector<uint16_t> sfill(total * kKvh, 0x3C00);
  std::vector<std::string> defs =
      kv8 ? join(kCtrl, {"QNORM_OFF=0", "KNORM_OFF=256", "KV8_PREP=1", "QKV_S=2"})
          : join(kCtrl, {"QNORM_OFF=0", "KNORM_OFF=256", "ATTN_BLOCK=64"});
  const char* src = kv8 ? "src/kernels/kv8.cl" : "src/kernels/attn.cl";
  const char* entry = kv8 ? "attn_prep_kv8" : "attn_prep";
  clrun::Program pm(dev, src, join(join(defs, kRows), {"M=" + std::to_string(M)}));
  clrun::Program p1(dev, src, join(defs, {"M=1"}));
  std::vector<uint32_t> ctrl(kCtrlWords, 0);
  ctrl[kPos] = 0xDEADu;
  ctrl[kNact] = M;
  clrun::Buffer bc(dev, ctrl), bp(dev, part), bsm(dev, small), br(dev, rope), bq(dev, size_t(M) * kQh * kHd * 4),
      bg(dev, size_t(M) * kQh * kHd * 4), bk(dev, kfill), bv(dev, kfill), bks(dev, sfill), bvs(dev, sfill),
      bt(dev, table(rows));
  if (kv8)
    pm.run(entry, {size_t(kQh + kKvh) * 256, M}, {256, 1}, bc, bp, bsm, br, bq, bg, bk, bv, bks, bvs, bt);
  else
    pm.run(entry, {size_t(kQh + kKvh) * 256, M}, {256, 1}, bc, bp, bsm, br, bq, bg, bk, bv, bt);
  const std::vector<float> got_q = bq.read<float>(), got_g = bg.read<float>();
  std::vector<uint8_t> got_k = bk.read<uint8_t>(), got_v = bv.read<uint8_t>();
  std::vector<uint16_t> got_ks = bks.read<uint16_t>(), got_vs = bvs.read<uint16_t>();
  int bad = 0;
  std::vector<bool> written(total, false);
  for (uint32_t m = 0; m < M; ++m) {
    const Row& r = rows[m];
    std::vector<uint32_t> c(kCtrlWords, 0);
    c[kPos] = r.pos;
    c[kNact] = 1;
    std::vector<uint8_t> k1(size_t(kLen) * row_b, 0), v1(size_t(kLen) * row_b, 0);
    std::vector<uint16_t> ks1(size_t(kLen) * kKvh, 0), vs1(size_t(kLen) * kKvh, 0);
    clrun::Buffer rc(dev, c), rp(dev, qkv[m]), rq(dev, size_t(kQh) * kHd * 4), rg(dev, size_t(kQh) * kHd * 4),
        rk(dev, k1), rv(dev, v1), rks(dev, ks1), rvs(dev, vs1);
    if (kv8)
      p1.run(entry, {size_t(kQh + kKvh) * 256, 1}, {256, 1}, rc, rp, bsm, br, rq, rg, rk, rv, rks, rvs);
    else
      p1.run(entry, {size_t(kQh + kKvh) * 256, 1}, {256, 1}, rc, rp, bsm, br, rq, rg, rk, rv);
    const std::vector<float> wq = rq.read<float>(), wg = rg.read<float>();
    const std::vector<uint8_t> wk = rk.read<uint8_t>(), wv = rv.read<uint8_t>();
    const std::vector<uint16_t> wks = rks.read<uint16_t>(), wvs = rvs.read<uint16_t>();
    const std::string w = std::string(kv8 ? "attn_prep_kv8" : "attn_prep") + " row " + std::to_string(m);
    bad += compare(w + " attn_q", got_q.data() + size_t(m) * kQh * kHd, wq.data(), wq.size() * 4);
    bad += compare(w + " attn_gate", got_g.data() + size_t(m) * kQh * kHd, wg.data(), wg.size() * 4);
    const size_t at = size_t(r.kv_base) + r.pos;
    written[at] = true;
    bad += compare(w + " K row", got_k.data() + at * row_b, wk.data() + size_t(r.pos) * row_b, row_b);
    bad += compare(w + " V row", got_v.data() + at * row_b, wv.data() + size_t(r.pos) * row_b, row_b);
    if (kv8) {
      bad += compare(w + " K scales", got_ks.data() + at * kKvh, wks.data() + size_t(r.pos) * kKvh, kKvh * 2);
      bad += compare(w + " V scales", got_vs.data() + at * kKvh, wvs.data() + size_t(r.pos) * kKvh, kKvh * 2);
    }
  }
  size_t moved = 0;
  for (size_t p = 0; p < total; ++p)
    if (!written[p]) {
      moved += std::memcmp(got_k.data() + p * row_b, kfill.data() + p * row_b, row_b) != 0;
      moved += std::memcmp(got_v.data() + p * row_b, kfill.data() + p * row_b, row_b) != 0;
      if (kv8) moved += got_ks[p * kKvh] != 0x3C00 || got_vs[p * kKvh] != 0x3C00;
    }
  if (moved) std::printf("  %zu rows outside the five written ones moved  <-- MISMATCH\n", moved);
  bad += moved ? 1 : 0;
  std::printf("%s rows (M = %u, three slots of %d positions): q, gate and every K / V row%s %s\n",
              kv8 ? "attn_prep_kv8" : "attn_prep", M, kLen, kv8 ? " and scale" : "",
              bad ? "DIFFER from the M = 1 kernel's" : "equal the M = 1 kernel's on that slot alone; no other row moved");
  return bad;
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::mt19937 g(13);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> small(kSmall, 0.0f);
    for (int i = 0; i < kConvRows * 4; ++i) small[i] = nd(g) * 6.0f;
    for (int h = 0; h < kHeads; ++h) {
      small[40960 + h] = -std::exp(std::uniform_real_distribution<float>(-2.0f, 1.0f)(g));
      small[41008 + h] = std::uniform_real_distribution<float>(-1.0f, 1.0f)(g);
    }
    std::vector<std::vector<float>> states(4, std::vector<float>(kState));
    std::vector<std::vector<uint16_t>> rings(4, std::vector<uint16_t>(kRingN));
    for (auto& s : states)
      for (float& x : s) x = nd(g);
    for (auto& r : rings)
      for (uint16_t& x : r) x = clrun::f32_to_bf16(nd(g) * 10.0f);
    clrun::Buffer bsmall(dev, small);
    clrun::Program plain(dev, "src/kernels/gdn_step.cl",
                         join(kCtrl, {"NEGA_OFF=40960", "DTBIAS_OFF=41008", "M=1"}));
    const std::vector<uint32_t> pos = {14, 2, 300, 47};
    int bad = 0;
    bad += gdn_case(dev, plain, bsmall, g, "4 x 1", states, rings, pos, {{0, 1}, {1, 1}, {2, 1}, {3, 1}});
    bad += gdn_case(dev, plain, bsmall, g, "2 x 4", states, rings, pos, {{1, 4}, {3, 4}});
    bad += gdn_case(dev, plain, bsmall, g, "3 + 1 + 2", states, rings, pos, {{0, 3}, {2, 1}, {1, 2}});
    bad += prep_case(dev, g, false);
    bad += prep_case(dev, g, true);
    std::printf("batch rows on %s: %s\n", dev.name().c_str(),
                bad ? "DISAGREE with the production kernels (indicative: read the kernel before the box)"
                    : "agree bitwise with the production kernels per sequence (indicative only; the B70 is the real test)");
    return bad ? 1 : 0;
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "batch_rows_run: %s\n", e.what());
    return 1;
  }
}

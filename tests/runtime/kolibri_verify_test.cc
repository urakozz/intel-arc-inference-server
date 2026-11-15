// Kolibri DSpark on the card (NOT YET RUN - waits for the operator's first card run): KolibriEngine's verify lists against its
// plain decode list, on a synthetic checkpoint or the real one, one card or two (pp2[:split]). No drafter: the drafts are the
// plain run's own ids (every one accepted) or, every other iteration, with the last one wrong (a partial commit).
//
//   1. The reference: prefill the prompt, then N plain greedy ids and each step's logits row.
//   2. The same prompt through enable_spec's lists: verify at K = 1..7 in turn (M = 2..8, the attention's second half at
//      M > 4), row r's argmax == the plain run's id at that position and its logits row against the plain row - cosine >=
//      0.99999 (spec 28 N4's bar; expected bitwise: every kernel computes row m as at M = 1 - the count is printed), then
//      commit(j, t) by greedy acceptance; the ids, the partial commits included, equal the plain run's token for token
//      (greedy lossless, D2), and the plain steps through the M = 1 verify list (generate with the lists on) too.
//   3. The verify list's launches: verify_device_launches' at every M captured.
// SKIP 77 without the checkpoint, or two cards for pp2.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "kolibri_rig.h"
#include "runtime/kolibri/kolibri_sizes.h"
#include "server/spec_accept.h"

namespace {

namespace rk = runtime::kolibri;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 96;

double cosine(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snapdir = kolibri_rig::kolibri_snapshot(argc > 1 ? argv[1] : "");
  if (snapdir.empty()) {
    std::printf("SKIP: no Kolibri-1 checkpoint at '%s'\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  kolibri_rig::Options o;
  o.max_len = kMaxLen;
  for (int i = 3; i < argc; ++i) {
    const std::string x = argv[i];
    if (x == "pp2") {
      o.devices = 2;
    } else if (x.rfind("pp2:", 0) == 0) {
      o.devices = 2;
      o.split = uint32_t(std::strtoul(x.c_str() + 4, nullptr, 10));
    } else if (x == "int8") {
      o.int8_head = true;
    } else {
      std::fprintf(stderr, "kolibri_verify_test: unknown flag %s (pp2[:split], int8)\n", x.c_str());
      return 2;
    }
  }
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(snapdir);
  if (o.devices == 1 && !rk::fits_one_card(pre, o.int8_head, size_t(32530000000ull), size_t(1.5e9), true)) {
    std::printf("SKIP: %s (%u layers) does not fit one card\n", snapdir.c_str(), pre.layers);
    return 77;
  }
  Ids prompt = kolibri_rig::read_ids(argc > 2 ? argv[2] : "", pre.vocab);
  if (prompt.empty()) prompt = {27498, 2625, 19919, 32, 49, 56, 55, 49, 346, 7524, 744, 38897, 573};
  kolibri_rig::Rig rig;
  const uint32_t V = pre.vocab;

  // --- 1. the reference ------------------------------------------------------------------------------------------------
  Ids ref;
  std::vector<float> ref_logits;   // row i: the head row whose argmax is ref[i + 1]
  {
    kolibri_rig::build(rig, snapdir, o);
    rk::KolibriEngine& e = *rig.eng;
    e.reset();
    e.prefill(prompt);
    ref.push_back(e.pending());
    for (uint32_t i = 0; i < kN; ++i) {
      e.generate(1);
      ref.push_back(e.pending());
      const std::vector<float> row = e.read_logits();
      ref_logits.insert(ref_logits.end(), row.begin(), row.end());
    }
    std::printf("1. the reference: %u plain ids after a %zu-id prompt (%u card(s))\n", kN, prompt.size(), o.devices);
  }

  // --- 2. the verify lists -------------------------------------------------------------------------------------------
  {
    kolibri_rig::build(rig, snapdir, o);
    rk::KolibriEngine& e = *rig.eng;
    e.enable_spec();
    e.reset();
    e.prefill(prompt);
    CHECK_EQ(e.pending(), ref[0]);
    const uint32_t base = e.pos();
    Ids out = {e.pending()};
    std::vector<float> rows(size_t(8) * V);
    double worst = 1.0;
    size_t bitwise = 0, compared = 0;
    uint32_t it = 0, partial = 0;
    while (out.size() <= kN) {
      const uint32_t i = e.pos() - base;   // ref[i] is pending
      const uint32_t k = std::min<uint32_t>({1 + it % 7, e.max_verify_k(), kN - i - 1});
      if (k == 0) {   // a plain step through the M = 1 verify list
        e.generate(1);
        out.push_back(e.pending());
        continue;
      }
      Ids drafts(ref.begin() + i + 1, ref.begin() + i + 1 + k);
      const bool wrong = (it % 2) == 1;
      if (wrong) drafts[k - 1] = (drafts[k - 1] + 1) % V;
      e.verify(drafts.data(), k);
      e.read_verify_logits(rows.data(), k + 1);
      const uint32_t good_rows = wrong ? k : k + 1;   // row r's input is right for r <= k - 1 (+ row k when all are)
      for (uint32_t r = 0; r < good_rows; ++r) {
        CHECK_EQ(e.verify_ids()[r], ref[i + r + 1]);
        const float* a = rows.data() + size_t(r) * V;
        const float* b = ref_logits.data() + size_t(i + r) * V;
        worst = std::min(worst, cosine(a, b, V));
        bitwise += std::equal(a, a + V, b) ? 1 : 0;
        ++compared;
      }
      const server::AcceptResult acc = server::accept_greedy(e.verify_ids(), drafts.data(), k);
      CHECK_EQ(acc.accepted, wrong ? k - 1 : k);
      partial += acc.accepted < k ? 1 : 0;
      e.commit(acc.accepted, acc.next_token);
      out.insert(out.end(), drafts.begin(), drafts.begin() + acc.accepted);
      out.push_back(acc.next_token);
      ++it;
    }
    out.resize(kN + 1);
    CHECK(out == ref);
    CHECK(worst >= 0.99999);
    std::printf("2. %u verifies (%u partial commits): ids == the plain run's; %zu rows compared, %zu bitwise, worst cosine "
                "%.7f\n", it, partial, compared, bitwise, worst);
    // --- 3. launches ---
    for (uint32_t M = 2; M <= 8; ++M) {
      size_t want = 0;
      for (uint32_t dev = 0; dev < e.devices(); ++dev)
        want += rk::verify_device_launches(e.model().desc, e.model().placement, dev, M);
      CHECK_EQ(e.verify_launches(M), want);
    }
    std::printf("3. the verify lists' launches at M 2..8 are verify_device_launches'\n");
  }
  std::printf("kolibri_verify_test: OK\n");
  return 0;
}

// prefill_split_test - a prompt prefilled in ONE Engine::prefill call against the same
// prompt prefilled in TWO calls split at an offset, and against M = 1 decode (ingest).
// Written for the fix-prefill-continuation investigation (2026-09-28); the finding it
// encodes is in docs/probe-prefill-continuation-2026-09-28.md.
//
// Per comparison, on one loaded model: the last-row logits cosine (the prefill's head),
// every FA layer's K and V cache row [0, N) (row cosine, min of K and V -- the cache row p
// of FA layer f is a function of the hidden state entering layer f at position p, so it
// localises a divergence to a layer and a row), and every GDN layer's recurrent state.
//
// **What a split changes, and therefore what the bars are.** Every kernel on the prefill
// walk is row-local or keyed to ABSOLUTE positions (the linears, flash attention's key
// tiles, RoPE, the KV writes, the conv ring) except one: the chunked gated delta rule cuts
// its 64-position chunks relative to the CALL's first id. So
//   * a split at a multiple of 64 must be BITWISE the one-call run (so must a second
//     one-call run);
//   * any other split re-rounds the GDN scan and nothing else. That is a perturbation of
//     the size of GDN's own rounding -- layer 0's state, whose inputs are identical, moves
//     by ~3e-6 in cosine -- and a real continuation bug (a wrong conv seed, a lost state, a
//     wrong RoPE/KV offset) moves every row after the split from the first FA layer on.
//     Gated: last-row logits, FA layers 0..3's every row, the median and 1st percentile
//     over all (layer, row) pairs, GDN layer 0's state.
//   * NOT gated: individual deep-layer rows. The model amplifies a rounding-sized
//     perturbation at a few positions (987, 1050, 1212, 1565, 2091 of long32k) to a K/V
//     cosine as low as 0.08 by FA layer 10; the CPU oracle (tools/oracle/dump.py) shows
//     the ONE-call run itself is 0.938 from it at row 987, and the oracle's own chunked run
//     (last_logits.py, chunk 1000) moves those rows as far. They are printed as the worst.
// The same bars hold against M = 1 decode (a different chain: recurrent GDN, fp32 q).
//
// usage: prefill_split_test <snapshot> <ids> [backend l0|l0-int8] [N] [splits,..] [decode 0|1]
//        prefill_split_test <snapshot> <ids> <backend> --oracle <dump.safetensors> <rows,..>
//          diagnostic: row r's logits from a one-call prefill of ids[0, r], from two-call
//          prefills split at 7 / 1000 / 2047, and from M = 1 decode, against the CPU
//          oracle's (tools/oracle/dump.py, every prompt position's logits).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/backend.h"

namespace {
using Ids = std::vector<uint32_t>;
using model::Qwen35;
constexpr uint32_t kMaxLen = 16384;   // a compiled decode max_len (src/kernels/CMakeLists.txt)
constexpr size_t kRow = 4 * 256;      // one position of one FA layer's K (or V), bf16
constexpr uint32_t kFa = 16, kGdn = 48;
constexpr uint32_t kShallow = 4;      // FA layers 0..3 = model layers 3, 7, 11, 15

struct Snap {
  std::vector<float> logits;          // kVocab
  std::vector<uint16_t> kv;           // [K l0..15][V l0..15], each [N][1024]
  std::vector<uint8_t> state;         // gdn_state (fp32) then conv_ring (bf16)
};

Snap snap(runtime::Engine& e, l0::CmdList& imm, uint32_t n) {
  Snap s;
  s.logits.resize(Qwen35::kVocab);
  imm.copy(s.logits.data(), e.prefill_scratch()->logits.ptr(), s.logits.size() * 4);
  s.kv.resize(e.kv_bytes(n) / 2);
  e.save_kv(0, n, s.kv.data());
  s.state.resize(e.state_bytes());
  e.save_state(s.state.data());
  return s;
}

double cos_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    ab += x * y;  aa += x * x;  bb += y * y;
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}
double cos_f32(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];  aa += double(a[i]) * a[i];  bb += double(b[i]) * b[i];
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}

struct Result {
  double logits = 1;
  double shallow = 1;                 // worst row of FA layers 0..kShallow-1
  uint32_t shallow_row = 0, shallow_layer = 0;
  double median = 1, p01 = 1;         // over every (FA layer, row) pair
  double deep = 1;                    // the worst pair anywhere -- recorded, not gated
  uint32_t deep_row = 0, deep_layer = 0, below9 = 0;
  double gdn0 = 1, state_worst = 1;   // GDN layer 0's state (identical inputs); the worst
  uint32_t state_worst_layer = 0;
};

Result compare(const Snap& a, const Snap& b, uint32_t n, bool verbose) {
  Result r;
  r.logits = cos_f32(a.logits.data(), b.logits.data(), Qwen35::kVocabUsed);
  const size_t plane = size_t(n) * kRow;
  std::vector<double> all;
  all.reserve(size_t(kFa) * n);
  for (uint32_t f = 0; f < kFa; ++f) {
    double lw = 1;
    uint32_t lrow = 0;
    for (uint32_t p = 0; p < n; ++p) {
      const size_t ok = size_t(f) * plane + size_t(p) * kRow;
      const size_t ov = size_t(kFa + f) * plane + size_t(p) * kRow;
      const double c = std::min(cos_bf16(&a.kv[ok], &b.kv[ok], kRow),
                                cos_bf16(&a.kv[ov], &b.kv[ov], kRow));
      all.push_back(c);
      if (c < lw) { lw = c; lrow = p; }
      r.below9 += c < 0.9;
    }
    if (f < kShallow && lw < r.shallow) { r.shallow = lw; r.shallow_row = lrow; r.shallow_layer = f; }
    if (lw < r.deep) { r.deep = lw; r.deep_row = lrow; r.deep_layer = f; }
    if (verbose) std::printf("      fa %2u: worst row %4u cos %.6f\n", f, lrow, lw);
  }
  std::sort(all.begin(), all.end());
  r.median = all[all.size() / 2];
  r.p01 = all[all.size() / 100];
  const size_t gdn_layer = size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim;
  const float* sa = reinterpret_cast<const float*>(a.state.data());
  const float* sb = reinterpret_cast<const float*>(b.state.data());
  for (uint32_t g = 0; g < kGdn; ++g) {
    const double c = cos_f32(sa + g * gdn_layer, sb + g * gdn_layer, gdn_layer);
    if (g == 0) r.gdn0 = c;
    if (c < r.state_worst) { r.state_worst = c; r.state_worst_layer = g; }
  }
  return r;
}

void print(const char* what, const Result& r) {
  std::printf("  %-14s logits %.6f | fa0-%u worst %.6f (fa %u row %u) | all rows median %.7f"
              " p01 %.6f, %u < 0.9, worst %.4f (fa %u row %u) | gdn L0 state %.9f, worst %.6f"
              " (L%u)\n", what, r.logits, kShallow - 1, r.shallow, r.shallow_layer, r.shallow_row,
              r.median, r.p01, r.below9, r.deep, r.deep_layer, r.deep_row, r.gdn0, r.state_worst,
              r.state_worst_layer);
}

// The gated bars (header). Calibrated on long32k[0, 2600), both backends, 2026-09-28:
// see docs/probe-prefill-continuation-2026-09-28.md for the measured values they sit under.
struct Bars {
  double logits, shallow, median, p01, gdn0;
};
bool check(const Result& r, const Bars& b) {
  const bool pass = r.logits >= b.logits && r.shallow >= b.shallow && r.median >= b.median &&
                    r.p01 >= b.p01 && r.gdn0 >= b.gdn0;
  if (!pass)
    std::printf("    ** FAIL ** bars: logits >= %.4f, fa0-%u every row >= %.4f, median >= %.5f,"
                " p01 >= %.3f, gdn L0 state >= %.6f\n", b.logits, kShallow - 1, b.shallow,
                b.median, b.p01, b.gdn0);
  return pass;
}

std::vector<uint32_t> parse_list(const std::string& s) {
  std::vector<uint32_t> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    v.push_back(uint32_t(std::stoul(s.substr(i, j - i))));
    i = j + 1;
  }
  return v;
}

int oracle_rows(const std::string& snapdir, const Ids& all, runtime::PrefillBackend backend,
                const std::string& path, const std::vector<uint32_t>& rows) {
  golden::Golden g(path);
  const size_t T = g.dim("logits", 2, 0), V = g.dim("logits", 2, 1);
  CHECK_EQ(V, size_t(Qwen35::kVocab));
  const float* orc = g.f32("logits", T * V);
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, snapdir, kMaxLen), kMaxLen);
  e.set_prefill_backend(backend);
  e.prepare_prefill();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::printf("oracle rows: backend %s, attention %s, oracle %s (%zu rows)\n",
              runtime::prefill_backend_name(backend),
              runtime::prefill::attn_mode_name(runtime::prefill::attn_mode()), path.c_str(), T);
  auto top1 = [](const float* v) {
    return uint32_t(std::max_element(v, v + Qwen35::kVocabUsed) - v);
  };
  std::vector<float> lg(Qwen35::kVocab);
  for (uint32_t r : rows) {
    if (r >= T) continue;
    const Ids p(all.begin(), all.begin() + r + 1);
    const float* o = orc + size_t(r) * V;
    auto report = [&](const char* what) {
      std::printf("  row %4u %-14s cos vs oracle %.6f  top1 %6u (oracle %6u)\n", r, what,
                  cos_f32(lg.data(), o, Qwen35::kVocabUsed), top1(lg.data()), top1(o));
    };
    e.reset();
    e.prefill(p);
    imm.copy(lg.data(), e.prefill_scratch()->logits.ptr(), lg.size() * 4);
    report("one call");
    for (uint32_t s : {7u, 1000u, 2047u}) {
      if (s >= r) continue;
      e.reset();
      e.prefill(Ids(p.begin(), p.begin() + s));
      e.prefill(Ids(p.begin() + s, p.end()));
      imm.copy(lg.data(), e.prefill_scratch()->logits.ptr(), lg.size() * 4);
      char w[32];
      std::snprintf(w, sizeof w, "split %u", s);
      report(w);
    }
    e.reset();
    e.ingest(p);
    imm.copy(lg.data(), e.buffers().logits.ptr(), lg.size() * 4);
    report("decode M=1");
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 6 && std::string(argv[4]) == "--oracle") {
    runtime::PrefillBackend b = runtime::prefill::default_prefill_backend();
    CHECK(runtime::parse_prefill_backend(argv[3], b));
    return oracle_rows(argv[1], golden::read_ids(argv[2]), b, argv[5], parse_list(argv[6]));
  }
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <snapshot> <ids> [backend] [N] [splits] [decode]\n", argv[0]);
    return 2;
  }
  const std::string snapdir = argv[1];
  runtime::PrefillBackend backend = runtime::prefill::default_prefill_backend();
  if (argc > 3) CHECK(runtime::parse_prefill_backend(argv[3], backend));
  const uint32_t N = argc > 4 ? uint32_t(std::atoi(argv[4])) : 2600;
  const std::vector<uint32_t> splits =
      parse_list(argc > 5 ? argv[5] : "1,7,16,63,64,1000,2047,2048,2049");
  const bool with_decode = argc > 6 ? std::atoi(argv[6]) != 0 : true;
  const bool verbose = std::getenv("B70_SPLIT_VERBOSE") != nullptr;
  const Ids all = golden::read_ids(argv[2]);
  CHECK(all.size() >= N && N + 1 <= kMaxLen);
  const Ids ids(all.begin(), all.begin() + N);

  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, snapdir, kMaxLen), kMaxLen);
  e.set_prefill_backend(backend);
  e.prepare_prefill();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const bool int8 = backend == runtime::PrefillBackend::L0Int8;
  std::printf("prefill_split_test: backend %s, attention %s, N %u\n",
              runtime::prefill_backend_name(backend),
              runtime::prefill::attn_mode_name(runtime::prefill::attn_mode()), N);

  // Measured worst over splits 1..2049 (and, second, decode), long32k[0, 2600):
  //   l0       logits 0.999882 / 0.999911, fa0-3 0.999764 / 0.999810, median 0.99994 /
  //            0.99993, p01 0.99579 / 0.99552, GDN L0 state 0.9999974 / 0.9999923
  //   l0-int8  logits 0.999513 / 0.999443, fa0-3 0.998538 / 0.997932, median 0.99961 /
  //            0.99953, p01 0.97736 / 0.97998, GDN L0 state 0.9999972 / 0.9999763
  // Each bar sits a few times the measured distance below it; a lost conv seed at the
  // split (the fault injected when this test was written) fails the GDN L0 and fa0-3 bars
  // by orders of magnitude (docs/probe-prefill-continuation-2026-09-28.md).
  const Bars split_bars = int8 ? Bars{0.998, 0.996, 0.999, 0.95, 0.99999}
                               : Bars{0.9995, 0.999, 0.9998, 0.99, 0.99999};
  const Bars decode_bars = int8 ? Bars{0.998, 0.995, 0.999, 0.95, 0.9999}
                                : Bars{0.9995, 0.999, 0.9998, 0.99, 0.99995};

  e.reset();
  e.prefill(ids);
  const Snap ref = snap(e, imm, N);
  bool ok = true;
  // The conv ring's slots other than the three live ones ((N-3..N-1) % 16) hold whatever
  // the last chunk that wrote them left, which nothing reads and which legitimately
  // differs with the chunking; the GDN state and the live slots must be equal.
  const size_t gdn_bytes = size_t(kGdn) * Qwen35::kGdnVHeads * Qwen35::kGdnHeadDim *
                           Qwen35::kGdnHeadDim * 4;
  constexpr size_t kRingSlot = 10240 * 2, kRingDepth = 16;
  auto bitwise = [&](const Snap& s) {
    if (s.logits != ref.logits || s.kv != ref.kv) return false;
    if (std::memcmp(s.state.data(), ref.state.data(), gdn_bytes) != 0) return false;
    for (size_t l = 0; l < kGdn; ++l)
      for (uint32_t p = N - 3; p < N; ++p) {
        const size_t off = gdn_bytes + (l * kRingDepth + p % kRingDepth) * kRingSlot;
        if (std::memcmp(s.state.data() + off, ref.state.data() + off, kRingSlot) != 0)
          return false;
      }
    return true;
  };
  {
    e.reset();
    e.prefill(ids);
    const bool same = bitwise(snap(e, imm, N));
    std::printf("  one call twice: %s\n", same ? "bitwise equal" : "DIFFERS  ** FAIL **");
    ok = ok && same;
  }
  for (uint32_t s : splits) {
    if (s == 0 || s >= N) continue;
    e.reset();
    e.prefill(Ids(ids.begin(), ids.begin() + s));
    e.prefill(Ids(ids.begin() + s, ids.end()));
    CHECK_EQ(e.pos(), N);
    const Snap two = snap(e, imm, N);
    char what[32];
    std::snprintf(what, sizeof what, "split %u:", s);
    if (s % 64 == 0) {
      // The GDN chunks fall where the one-call run's do: nothing may differ.
      const bool same = bitwise(two);
      std::printf("  %-14s %s\n", what, same ? "bitwise equal (a multiple of 64)"
                                             : "DIFFERS from one call  ** FAIL **");
      if (!same) print(what, compare(ref, two, N, verbose));
      ok = ok && same;
      continue;
    }
    const Result r = compare(ref, two, N, verbose);
    print(what, r);
    ok = check(r, split_bars) && ok;
  }
  if (with_decode) {
    // M = 1 decode over the first N - 1 ids, then the prefill of the last one: KV rows
    // [0, N - 1) and the GDN state are decode's, the logits row is a one-id prefill's head.
    e.reset();
    e.ingest(Ids(ids.begin(), ids.end() - 1));
    e.prefill(Ids(ids.end() - 1, ids.end()));
    const Result r = compare(ref, snap(e, imm, N), N, verbose);
    print("decode M=1:", r);
    ok = check(r, decode_bars) && ok;
  }
  std::printf("prefill_split_test %s\n", ok ? "OK" : "FAILED");
  return ok ? 0 : 1;
}

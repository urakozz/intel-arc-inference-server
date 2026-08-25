// golden_gate_test - the trust chain closed (spec §11, plan 3 Task 8).
//
// Everything before this test compares a kernel against a host reference that
// *this project wrote*. This one compares the whole engine against a reference
// it did not write: `transformers` 5.15 on CPU, the pure-torch
// gated-delta-rule, run over the same checkpoint dequantised by the same rule
// the C++ loader is bit-compared against (tools/oracle/README.md). If the two
// agree on 3 x 32 greedy tokens, element-exact, then 774 kernels, the loader,
// the repack, the capture and the decode loop are all right together.
//
// The gate, per prompt:
//   * ingest the committed prompt ids ONE AT A TIME, reading the per-layer
//     tap after each (Engine::ingest has no per-token callback; one id per
//     call is byte-identical to one call with all of them, because each call
//     is just a replay per id);
//   * generate 32 ids as 32 x generate(1), reading `buffers().logits` before
//     each step - that fp32 row is exactly the row whose argmax the step is
//     about to consume, so a mismatch can be reported with both top-5s;
//   * **GATE**: the 32 ids == golden `tokens`, element-exact.
//
// Diagnostics - printed for every prompt whether or not the gate passes, and
// evaluated only after all three prompts have printed, so a failure still
// leaves the complete picture in the log:
//   * per layer, min-over-t cosine (fp64) of the tap against the oracle;
//   * the tail the tap cannot see, `b.resid` vs golden `resid.L63`;
//   * the GDN recurrent state after the prompt, per GDN layer;
//   * per generated step, the cosine of the two logit rows.
//
// **What tap[i] is.** NOT layer i's output. `runtime/capture.h` is the single
// authority: the tap copies after layer i's last kernel, and the residual
// stream is advanced only by `prep_res_norm`, so tap[i] holds the hidden state
// with layer i's *mixer* folded in and layer i's MLP still un-folded in
// `partials` (layer i+1's leading prep folds it). The oracle's `resid.L{i}` is
// the layer *output* - post-MLP - so the comparator is built, not read:
//
//     expected_tap[i][t] = bf16( resid.L{i-1}[t] + mixer.L{i}[t] )   i >= 1
//     expected_tap[0][t] = bf16( embed[ids[t]]   + mixer.L0[t]  )
//
// summed in fp32 from the two golden bf16 tensors and rounded RNE, because
// that is the arithmetic the engine's `prep_res_norm` does. Layer 0 has no
// `resid.L-1`: the oracle dumps no embedding tensor, so the rows are gathered
// host-side out of `LoadedModel::embed`, which is the same bf16 table the
// device gathers from. Layer 63's post-MLP residual has NO tap at all (there
// is no tap 64) - `b.resid` after the fence covers it instead, and the logits
// and the tokens cover it again.
//
// Label `checkpoint golden`: needs the real 19 GB checkpoint, a B70, and the
// ~1 GB of oracle output that lives only on the box. Absent goldens are a SKIP
// (exit 77), not a failure - the Mac never has them.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "loader/safetensors.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/engine.h"

namespace {
using model::Qwen35;

constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 32;
constexpr uint32_t kHid = Qwen35::kHidden;  // 5120
constexpr size_t kGdnElems =
    size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim;  // 48*128*128
// The cosine bar for every tensor comparison in this test - and it marks rows
// `**LOW**`, it does not fail the run. Spec §11 is explicit that token equality
// is the gate and tensors are diagnostics, "because 64 layers of bf16 residual
// drift make a hard tensor bound either useless or flaky", and the `code`
// prompt is the standing demonstration: 36 of its 3904 tap comparisons sit
// below this bar on 3 of 61 positions while all 32 token ids are exact
// (docs/14-golden-gate.md).
//
// `gdn_state` was gating until the Task 8 review ruled it back to a diagnostic
// (2026-08-25). Measured margin at the time of that ruling: worst 0.999537065
// (`code`, L60), i.e. 5.4e-4 of headroom on a bound the spec never asked to be
// hard. Anything that legitimately shifts numerics - or simply a longer prompt
// - could put it under, and failing the golden gate on a recurrent-state
// cosine would say "the engine is wrong" when the tokens say otherwise.
constexpr double kBar = 0.999;
const char* const kPrompts[] = {"prose", "code", "cjk"};

// --- the golden file ------------------------------------------------------
// One safetensors file, mmapped, header parsed by the production parser
// (`SafetensorsSet::parse_header`) - SafetensorsSet itself wants a sharded
// checkpoint with an index.json, which a single oracle dump is not.
class Golden {
 public:
  explicit Golden(const std::string& path) : file_(path) {
    uint64_t hlen = 0;
    CHECK(file_.size() >= 8);
    std::memcpy(&hlen, file_.data(), 8);
    CHECK(hlen <= file_.size() - 8);
    data_start_ = size_t(8 + hlen);
    for (auto& e : loader::SafetensorsSet::parse_header(file_.data(), file_.size()))
      tensors_.emplace(e.first, e.second);
  }
  const loader::TensorInfo& info(const std::string& n) const {
    auto it = tensors_.find(n);
    if (it == tensors_.end()) {
      std::fprintf(stderr, "golden file has no tensor '%s'\n", n.c_str());
      std::exit(1);
    }
    return it->second;
  }
  // shape[i] with the rank checked first: a 1-D tensor indexed at [1] is
  // undefined behaviour, and a golden file is untrusted input like any other.
  uint64_t dim(const std::string& n, size_t rank, size_t i) const {
    const loader::TensorInfo& t = info(n);
    if (t.shape.size() != rank) {
      std::fprintf(stderr, "golden '%s' is %zu-D, expected %zu-D\n", n.c_str(), t.shape.size(),
                   rank);
      std::exit(1);
    }
    return t.shape[i];
  }
  // dtype, element count AND byte extent, the last mirroring what
  // loader/safetensors.cc does for the checkpoint: `data_offsets` is a claim in
  // a header, and a truncated or mislabelled dump must exit(1) with a message
  // rather than SIGBUS somewhere inside a cosine loop minutes later.
  const void* raw(const std::string& n, const char* dtype, size_t want_elems) const {
    const loader::TensorInfo& t = info(n);
    if (t.dtype != dtype) {
      std::fprintf(stderr, "golden '%s' is %s, expected %s\n", n.c_str(), t.dtype.c_str(), dtype);
      std::exit(1);
    }
    size_t elems = 1;
    for (uint64_t d : t.shape) elems *= size_t(d);
    if (elems != want_elems) {
      std::fprintf(stderr, "golden '%s' has %zu elements, expected %zu\n", n.c_str(), elems,
                   want_elems);
      std::exit(1);
    }
    const size_t esz = std::strcmp(dtype, "BF16") == 0 ? 2 : 4;   // F32 / I32 are 4
    const size_t want_bytes = want_elems * esz;
    if (t.begin > t.end || t.end - t.begin != want_bytes) {
      std::fprintf(stderr, "golden '%s' spans %llu bytes ([%llu, %llu)), expected %zu\n", n.c_str(),
                   (unsigned long long)(t.end - t.begin), (unsigned long long)t.begin,
                   (unsigned long long)t.end, want_bytes);
      std::exit(1);
    }
    const size_t avail = file_.size() - data_start_;
    if (t.end > avail) {
      std::fprintf(stderr,
                   "golden '%s' ends at %llu, past the file's %zu-byte data section - truncated?\n",
                   n.c_str(), (unsigned long long)t.end, avail);
      std::exit(1);
    }
    return file_.data() + data_start_ + t.begin;
  }
  const uint16_t* bf16(const std::string& n, size_t e) const {
    return static_cast<const uint16_t*>(raw(n, "BF16", e));
  }
  const float* f32(const std::string& n, size_t e) const {
    return static_cast<const float*>(raw(n, "F32", e));
  }
  const int32_t* i32(const std::string& n, size_t e) const {
    return static_cast<const int32_t*>(raw(n, "I32", e));
  }

 private:
  loader::MappedFile file_;
  size_t data_start_ = 0;
  std::map<std::string, loader::TensorInfo> tensors_;
};

// --- metrics, fp64 throughout ---------------------------------------------
struct Metric {
  double cos = 1.0;   // dot / (|a| |b|)
  double rel = 0.0;   // |a - b| / |b|   (b = the oracle)
  double nb = 0.0;    // |b|, the oracle's norm - a cosine is only as meaningful
  double err = 0.0;   // |a - b|          as the magnitude it is divided by
};
Metric compare(const double* a, const double* b, size_t n) {
  double ab = 0, aa = 0, bb = 0, dd = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = a[i] - b[i];
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
    dd += d * d;
  }
  Metric m;
  m.cos = (aa > 0.0 && bb > 0.0) ? ab / (std::sqrt(aa) * std::sqrt(bb)) : (aa == bb ? 1.0 : 0.0);
  m.nb = std::sqrt(bb);
  m.err = std::sqrt(dd);
  m.rel = bb > 0.0 ? m.err / m.nb : m.err;
  return m;
}
// bf16 words widened to fp64 before anything is accumulated.
Metric compare_bf16(const uint16_t* a, const uint16_t* b, size_t n, std::vector<double>& sa,
                    std::vector<double>& sb) {
  sa.resize(n);
  sb.resize(n);
  for (size_t i = 0; i < n; ++i) {
    sa[i] = common::bf16_to_f32(a[i]);
    sb[i] = common::bf16_to_f32(b[i]);
  }
  return compare(sa.data(), sb.data(), n);
}
Metric compare_f32(const float* a, const float* b, size_t n, std::vector<double>& sa,
                   std::vector<double>& sb) {
  sa.resize(n);
  sb.resize(n);
  for (size_t i = 0; i < n; ++i) {
    sa[i] = a[i];
    sb[i] = b[i];
  }
  return compare(sa.data(), sb.data(), n);
}

// argmax with the engine's rule: indices >= VOCAB_USED cannot win, and an
// exact tie goes to the LOWER index (src/kernels/argmax.cl). torch's argmax
// breaks ties the same way, so the two are compared like with like.
uint32_t argmax_masked(const float* row, uint32_t n, uint32_t used) {
  uint32_t best = 0;
  float bv = -INFINITY;
  for (uint32_t i = 0; i < n && i < used; ++i)
    if (row[i] > bv) { bv = row[i]; best = i; }
  return best;
}
uint32_t argmax_full(const float* row, uint32_t n) { return argmax_masked(row, n, n); }

void print_top5(const char* tag, const float* row, uint32_t n, uint32_t used) {
  std::vector<std::pair<float, uint32_t>> v;
  v.reserve(n);
  for (uint32_t i = 0; i < n && i < used; ++i) v.emplace_back(row[i], i);
  std::partial_sort(v.begin(), v.begin() + 5, v.end(),
                    [](const std::pair<float, uint32_t>& x, const std::pair<float, uint32_t>& y) {
                      return x.first != y.first ? x.first > y.first : x.second < y.second;
                    });
  std::printf("      %-8s", tag);
  for (int i = 0; i < 5; ++i) std::printf("  %u:%.5f", v[size_t(i)].second, double(v[size_t(i)].first));
  std::printf("\n");
}

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open prompt ids: %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<uint32_t> ids;
  for (long long v; f >> v;) {
    CHECK(v >= 0 && v < (long long)Qwen35::kVocabUsed);
    ids.push_back(uint32_t(v));
  }
  return ids;
}
bool exists(const std::string& p) { return std::ifstream(p).good(); }

// One prompt's verdict, printed again in the summary at the end.
struct Verdict {
  std::string name;
  uint32_t n_prompt = 0, exact = 0;
  int first_bad = -1;
  double tap_min_cos = 1.0, tail_min_cos = 1.0, gdn_min_cos = 1.0, logit_min_cos = 1.0;
  uint32_t tap_min_layer = 0, tap_min_t = 0, gdn_min_layer = 0;
};
}  // namespace

int main(int argc, char** argv) {
  const std::string gdir = argc > 1 ? argv[1] : "oracle-out";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  const std::string snap = argc > 3 ? argv[3] : "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ";

  for (const char* p : kPrompts) {
    const std::string g = gdir + "/" + p + ".golden.safetensors";
    if (!exists(g)) {
      std::printf(
          "SKIP: %s is absent. The oracle dumps live only on the box "
          "(~1 GB, .gitignore'd) - see tools/oracle/README.md, 'Running it'.\n",
          g.c_str());
      return 77;
    }
  }

  l0::Context ctx(0);
  // One load, one Engine, three prompts: reset() zeroes exactly the persistent
  // group between them, which is what starting a fresh session means. Loading
  // 19 GB three times would cost ~6 minutes and prove nothing extra.
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  CHECK(model.embed.size() >= size_t(Qwen35::kVocab) * kHid * 2);
  runtime::Engine eng(ctx, std::move(model), kMaxLen, /*debug_resid=*/true);
  CHECK(eng.debug_resid());
  CHECK_EQ(eng.step().kernel_count, size_t(774));   // 645 + lever L1's 129
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  const size_t gdn_stride = kGdnElems * sizeof(float);
  CHECK_EQ(eng.buffers().gdn_state.size(), gdn_stride * 48);

  std::vector<Verdict> verdicts;
  std::vector<double> sa, sb;  // fp64 scratch, reused

  for (const char* pname : kPrompts) {
    Verdict v;
    v.name = pname;
    const std::vector<uint32_t> ids = read_ids(pdir + "/" + pname + ".ids");
    Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size());
    v.n_prompt = T;
    CHECK(T > 0 && T <= 64);
    // The one standing consistency check between a committed .ids file and the
    // golden dump it is the prompt of: same number of positions. (The ids were
    // additionally verified byte-equal to each file's `prompt_ids` metadata by
    // hand when they were committed; `parse_header` drops `__metadata__`, so
    // the test cannot re-check that, and plumbing it through is not worth it.)
    CHECK_EQ(g.dim("resid.L0", 2, 0), uint64_t(T));
    CHECK_EQ(g.dim("resid.L0", 2, 1), uint64_t(kHid));
    const uint32_t V = uint32_t(g.dim("logits", 2, 1));
    const uint32_t Lrows = uint32_t(g.dim("logits", 2, 0));
    CHECK_EQ(Lrows, T + kGen);
    const uint32_t Vcmp = std::min(V, Qwen35::kVocab);
    std::printf(
        "\n================ %s: %u prompt ids, %u generated, golden vocab %u "
        "(engine %u padded / %u usable) ================\n",
        pname, T, kGen, V, Qwen35::kVocab, Qwen35::kVocabUsed);

    // ---- 1. ingest, one id per replay, tapping after each -------------------
    eng.reset();
    CHECK_EQ(eng.pos(), uint32_t(0));
    std::vector<std::vector<uint16_t>> tap(T);
    std::vector<std::vector<uint16_t>> tail(T);   // b.resid after the fence
    std::vector<std::vector<uint16_t>> emb(T);    // the embedding row the device gathered
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      tap[t] = eng.read_debug_resid();
      CHECK_EQ(tap[t].size(), size_t(Qwen35::kLayers) * kHid);
      tail[t].resize(kHid);
      imm.copy(tail[t].data(), eng.buffers().resid.ptr(), size_t(kHid) * 2);
      emb[t].resize(kHid);
      imm.copy(emb[t].data(),
               static_cast<const uint8_t*>(eng.model().embed.ptr()) + size_t(ids[t]) * kHid * 2,
               size_t(kHid) * 2);
    }
    CHECK_EQ(eng.pos(), T);

    // ---- 2. the per-layer tap vs the built comparator -----------------------
    std::printf("  layer diagnostics (min over the %u prompt positions; expected tap =\n"
                "  bf16(resid.L{i-1}[t] + mixer.L{i}[t]), embed[id] standing in for resid.L-1).\n"
                "  |oracle| and |err| are at the argmin t: a cosine is only as meaningful as\n"
                "  the magnitude it divides by, so both are printed next to it.\n"
                "    layer kind    min cos      at t   median cos   max relL2   |oracle|      |err|\n", T);
    std::vector<uint16_t> expect(kHid);
    std::vector<double> cos_lt(size_t(Qwen35::kLayers) * T), nb_lt(size_t(Qwen35::kLayers) * T);
    uint32_t n_low_pairs = 0;
    std::vector<uint32_t> low_t;   // the distinct positions that go below the bar
    std::vector<double> med_l(Qwen35::kLayers, 1.0);   // per-layer upper-median cosine
    for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
      const std::string ls = std::to_string(l);
      const uint16_t* mix = g.bf16("mixer.L" + ls, size_t(T) * kHid);
      const uint16_t* prev =
          l == 0 ? nullptr : g.bf16("resid.L" + std::to_string(l - 1), size_t(T) * kHid);
      double lmin = 2.0, lrel = 0.0, lnb = 0.0, lerr = 0.0;
      uint32_t at = 0;
      for (uint32_t t = 0; t < T; ++t) {
        const uint16_t* base = prev ? prev + size_t(t) * kHid : emb[t].data();
        for (uint32_t k = 0; k < kHid; ++k)
          expect[k] = common::f32_to_bf16(common::bf16_to_f32(base[k]) +
                                          common::bf16_to_f32(mix[size_t(t) * kHid + k]));
        const Metric m = compare_bf16(tap[t].data() + size_t(l) * kHid, expect.data(), kHid, sa, sb);
        cos_lt[size_t(l) * T + t] = m.cos;
        nb_lt[size_t(l) * T + t] = m.nb;
        if (m.cos < kBar) {
          ++n_low_pairs;
          if (std::find(low_t.begin(), low_t.end(), t) == low_t.end()) low_t.push_back(t);
        }
        if (m.cos < lmin) { lmin = m.cos; at = t; lnb = m.nb; lerr = m.err; }
        lrel = std::max(lrel, m.rel);
      }
      if (lmin < v.tap_min_cos) { v.tap_min_cos = lmin; v.tap_min_layer = l; v.tap_min_t = at; }
      // Upper median: element T/2 of the sorted row, which for even T is the
      // upper of the two central values rather than their mean. Named that way
      // in the header so nobody averages two of these and calls it a median.
      std::vector<double> row(cos_lt.begin() + size_t(l) * T, cos_lt.begin() + size_t(l + 1) * T);
      std::nth_element(row.begin(), row.begin() + T / 2, row.end());
      med_l[l] = row[T / 2];
      std::printf("      %2u  %-4s  %.9f   %4u   %.9f   %.3e   %9.3f  %9.3f%s\n", l,
                  Qwen35::is_fa(l) ? "FA" : "GDN", lmin, at, med_l[l], lrel, lnb, lerr,
                  lmin < kBar ? "   **LOW**" : "");
    }
    // Printed rather than derived by hand afterwards: this is the number the
    // docs quote as "the typical comparison", and the split says whether the
    // softmax path (FA) diverges more than the GEMV path alone (GDN).
    {
      double all = 0, gdn = 0, fa = 0;
      uint32_t ng = 0, nf = 0;
      for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
        all += med_l[l];
        if (Qwen35::is_fa(l)) { fa += med_l[l]; ++nf; } else { gdn += med_l[l]; ++ng; }
      }
      std::printf("  mean of the 64 per-layer upper-medians: %.9f   (GDN %u: %.9f, FA %u: %.9f)\n",
                  all / Qwen35::kLayers, ng, gdn / ng, nf, fa / nf);
    }
    std::printf("  census: %u of the %u (layer, t) tap comparisons are below %.3f, on %zu of the"
                " %u positions:", n_low_pairs, uint32_t(Qwen35::kLayers) * T, kBar, low_t.size(), T);
    std::sort(low_t.begin(), low_t.end());
    for (uint32_t t : low_t) std::printf(" %u", t);
    std::printf("%s\n", low_t.empty() ? " (none)" : "");

    // Attribution at the worst position. The tap error alone cannot say whether
    // a layer produced a wrong output or merely carried an earlier one, so the
    // layer's own contribution is compared too: in the engine it is
    // tap[i] - tap[i-1] (mixer_i + mlp_{i-1}, the two things folded between the
    // two copies), and the oracle dumps both halves of that as mixer.L{i} and
    // mlp.L{i-1}. Layer 0's contribution is tap[0] - embed = mixer.L0.
    {
      const uint32_t tw = v.tap_min_t;
      std::printf("  attribution at the worst position t=%u (contribution = mixer.L{i} +"
                  " mlp.L{i-1}):\n"
                  "    layer kind   tap cos     |tap err|   contrib cos  |contrib|  |contrib err|\n",
                  tw);
      std::vector<double> ce(kHid), co(kHid);
      for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
        const std::string ls = std::to_string(l);
        const uint16_t* mix = g.bf16("mixer.L" + ls, size_t(T) * kHid) + size_t(tw) * kHid;
        const uint16_t* prev =
            l == 0 ? nullptr : g.bf16("resid.L" + std::to_string(l - 1), size_t(T) * kHid) +
                                   size_t(tw) * kHid;
        const uint16_t* base = prev ? prev : emb[tw].data();
        for (uint32_t k = 0; k < kHid; ++k)
          expect[k] = common::f32_to_bf16(common::bf16_to_f32(base[k]) + common::bf16_to_f32(mix[k]));
        const Metric mt =
            compare_bf16(tap[tw].data() + size_t(l) * kHid, expect.data(), kHid, sa, sb);
        const uint16_t* pmlp =
            l == 0 ? nullptr : g.bf16("mlp.L" + std::to_string(l - 1), size_t(T) * kHid) +
                                   size_t(tw) * kHid;
        const uint16_t* eprev = l == 0 ? emb[tw].data() : tap[tw].data() + size_t(l - 1) * kHid;
        for (uint32_t k = 0; k < kHid; ++k) {
          ce[k] = common::bf16_to_f32(tap[tw][size_t(l) * kHid + k]) - common::bf16_to_f32(eprev[k]);
          co[k] = common::bf16_to_f32(mix[k]) + (pmlp ? common::bf16_to_f32(pmlp[k]) : 0.0f);
        }
        const Metric mc = compare(ce.data(), co.data(), kHid);
        std::printf("      %2u  %-4s  %.9f  %9.3f   %.9f  %9.3f  %9.3f\n", l,
                    Qwen35::is_fa(l) ? "FA" : "GDN", mt.cos, mt.err, mc.cos, mc.nb, mc.err);
      }
    }
    // The worst layer, position by position. A cosine that dips at two or three
    // positions and recovers is a different animal from one that decays with
    // depth, and this line is what tells the two apart.
    std::printf("  worst tap layer L%u, every position (t: cos / |oracle|):\n",
                v.tap_min_layer);
    for (uint32_t t = 0; t < T; ++t)
      std::printf("      %2u: %.6f %8.2f%s", t, cos_lt[size_t(v.tap_min_layer) * T + t],
                  nb_lt[size_t(v.tap_min_layer) * T + t], (t % 4 == 3 || t + 1 == T) ? "\n" : "");
    // The tail the tap cannot reach: layer 63 post-MLP, which `b.resid` holds
    // after every fence (capture.h). There is no tap 64 and none is invented.
    {
      const uint16_t* r63 = g.bf16("resid.L63", size_t(T) * kHid);
      double lmin = 2.0, lrel = 0.0, lnb = 0.0, lerr = 0.0;
      uint32_t at = 0;
      for (uint32_t t = 0; t < T; ++t) {
        const Metric m = compare_bf16(tail[t].data(), r63 + size_t(t) * kHid, kHid, sa, sb);
        if (m.cos < lmin) { lmin = m.cos; at = t; lnb = m.nb; lerr = m.err; }
        lrel = std::max(lrel, m.rel);
      }
      v.tail_min_cos = lmin;
      std::printf("      63+ MLP (b.resid vs resid.L63)  min cos %.9f at t=%u, max relL2 %.3e,"
                  " |oracle| %.3f, |err| %.3f%s\n",
                  lmin, at, lrel, lnb, lerr, lmin < kBar ? "   **LOW**" : "");
    }

    // ---- 3. the GDN recurrent state after the prompt ------------------------
    std::printf("  gdn_state after the prompt (%zu fp32 per layer, 48 GDN layers):\n", kGdnElems);
    {
      std::vector<float> st(kGdnElems);
      uint32_t gi = 0, n_low = 0;
      double gmin = 2.0, gmax = -2.0, relmax = 0.0;
      for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
        if (Qwen35::is_fa(l)) continue;
        imm.copy(st.data(),
                 static_cast<const uint8_t*>(eng.buffers().gdn_state.ptr()) + size_t(gi) * gdn_stride,
                 gdn_stride);
        const Metric m = compare_f32(st.data(), g.f32("gdn_state.L" + std::to_string(l), kGdnElems),
                                     kGdnElems, sa, sb);
        if (m.cos < gmin) { gmin = m.cos; v.gdn_min_layer = l; }
        gmax = std::max(gmax, m.cos);
        relmax = std::max(relmax, m.rel);
        if (m.cos < kBar) {
          ++n_low;
          std::printf("      L%-2u cos %.9f relL2 %.3e   **LOW**\n", l, m.cos, m.rel);
        }
        ++gi;
      }
      CHECK_EQ(gi, uint32_t(48));
      v.gdn_min_cos = gmin;
      std::printf("      min cos %.9f (L%u), max cos %.9f, max relL2 %.3e, %u/48 below %.3f\n",
                  gmin, v.gdn_min_layer, gmax, relmax, n_low, kBar);
    }

    // ---- 4. THE GATE: 32 greedy ids, element-exact --------------------------
    CHECK_EQ(g.dim("tokens", 1, 0), uint64_t(kGen));
    const int32_t* gtok = g.i32("tokens", kGen);
    const float* glog = g.f32("logits", size_t(Lrows) * V);
    std::vector<float> dec(Qwen35::kVocab);
    auto read_logits = [&] {
      imm.copy(dec.data(), eng.buffers().logits.ptr(), size_t(Qwen35::kVocab) * sizeof(float));
    };
    read_logits();  // the row whose argmax is generated token 0: golden logits[T-1]

    std::printf("  greedy %u:  step  engine  golden   logit-cos   engine-argmax golden-argmax "
                "(masked / full)\n", kGen);
    std::vector<uint32_t> etok;
    etok.reserve(kGen);
    for (uint32_t p = 0; p < kGen; ++p) {
      const float* grow = glog + size_t(p == 0 ? T - 1 : T + p - 1) * V;
      const Metric lm = compare_f32(dec.data(), grow, Vcmp, sa, sb);
      v.logit_min_cos = std::min(v.logit_min_cos, lm.cos);
      // Both compared argmaxes see the SAME width, so a future dump narrower
      // than the engine's padded row cannot silently desymmetrize the pair.
      // The device-consistency check below is separate and must keep the full
      // engine width, because the device argmaxes the whole padded row.
      const uint32_t ea = argmax_masked(dec.data(), Vcmp, Qwen35::kVocabUsed);
      const uint32_t gam = argmax_masked(grow, Vcmp, Qwen35::kVocabUsed);
      const uint32_t ea_dev = argmax_masked(dec.data(), Qwen35::kVocab, Qwen35::kVocabUsed);
      const uint32_t gaf = argmax_full(grow, V);   // the pad-tail check wants all of V

      const uint32_t id = eng.generate(1)[0];
      etok.push_back(id);
      // The device's two-stage argmax and a host argmax over the same fp32 row
      // must agree; if they ever do not, the bug is in argmax, not upstream.
      CHECK_EQ(id, ea_dev);
      const bool ok = id == uint32_t(gtok[p]);
      if (ok) ++v.exact;
      if (!ok && v.first_bad < 0) v.first_bad = int(p);
      std::printf("            %4u  %6u  %6u   %.9f   %6u %6u %6u %s%s\n", p, id,
                  uint32_t(gtok[p]), lm.cos, ea, gam, gaf, ok ? "" : "  <== MISMATCH",
                  gam != gaf ? "  [golden argmax is a padding id]" : "");
      if (!ok && int(p) == v.first_bad) {
        std::printf("      first mismatch at generated position %u: engine %u, golden %u"
                    "  (decision row = golden logits[%u], cos %.9f, relL2 %.3e)\n",
                    p, id, uint32_t(gtok[p]), p == 0 ? T - 1 : T + p - 1, lm.cos, lm.rel);
        print_top5("engine", dec.data(), Qwen35::kVocab, Qwen35::kVocabUsed);
        print_top5("golden", grow, V, Qwen35::kVocabUsed);
      }
      read_logits();
    }
    std::printf("  %s: %u/%u tokens exact%s\n", pname, v.exact, kGen,
                v.first_bad < 0 ? "" : ("  (first mismatch at " + std::to_string(v.first_bad) + ")").c_str());
    std::printf("  engine:");
    for (uint32_t id : etok) std::printf(" %u", id);
    std::printf("\n  golden:");
    for (uint32_t p = 0; p < kGen; ++p) std::printf(" %u", uint32_t(gtok[p]));
    std::printf("\n");
    verdicts.push_back(v);
  }

  // ---- 5. the verdict, once every prompt has printed its diagnostics -------
  std::printf("\n================ golden gate ================\n");
  std::printf("  the gate is the exact/%u column alone; every cosine below is a diagnostic\n"
              "  prompt   ids   exact/%u   tap min cos (layer,t)   L63 tail   gdn min cos   "
              "logit min cos\n", kGen, kGen);
  for (const Verdict& v : verdicts)
    std::printf("  %-7s %4u   %2u/%u      %.9f (%2u,%2u)   %.9f  %.9f (L%u)  %.9f\n", v.name.c_str(),
                v.n_prompt, v.exact, kGen, v.tap_min_cos, v.tap_min_layer, v.tap_min_t,
                v.tail_min_cos, v.gdn_min_cos, v.gdn_min_layer, v.logit_min_cos);

  bool bad = false;
  for (const Verdict& v : verdicts) {
    if (v.exact != kGen) {
      std::fprintf(stderr, "GATE FAILED: %s reproduced %u/%u tokens (first mismatch %d)\n",
                   v.name.c_str(), v.exact, kGen, v.first_bad);
      bad = true;
    }
    // gdn_state is NOT gated - see kBar's comment. It prints **LOW** per layer
    // like the tap does, and the token ids are what decide this test.
  }
  if (bad) return 1;
  std::printf("golden_gate_test OK: 3 prompts x %u greedy tokens, element-exact against the "
              "CPU oracle\n", kGen);
  return 0;
}

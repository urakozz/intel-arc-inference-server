#pragma once
// The golden gate's reusable half, extracted VERBATIM from
// `tests/golden/golden_gate_test.cc` (plan 6b Task 12 Step 1) so that spec 2's
// `prefill_gate_test` grades a prefilled engine on exactly the machinery the
// decode gate is graded on -- the same tie-aware decision rule, the same
// masking, the same metrics, the same golden-file validation.
//
// **This file is a MOVE, not a rewrite.** Every comment came across word for
// word, and two of them are load-bearing records rather than explanation:
// `kBar`'s (the 2026-08-25 ruling that put `gdn_state` back to a diagnostic)
// and `golden_decision`'s (the controller's 2026-08-26 tie-aware gate
// semantics). Paraphrasing either would delete the ruling. The acceptance for
// the move is re-running BOTH golden gates and diffing the output against the
// run recorded before it: a pure move that changes a number is not a pure move.
//
// `golden_gate_test.cc` keeps its `kPrompts`, its `Verdict` and its `main`.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "loader/safetensors.h"
#include "model/qwen35.h"

namespace golden {
using model::Qwen35;

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

// --- the controller's gate-semantics ruling, 2026-08-26 --------------------
//
// **A golden decision row whose top-1 is not unique is UNDETERMINED.** The
// oracle's logits are bf16 values widened to fp32 (`dump.py` records
// `out.logits[0].to(torch.float32)` and `out.logits` from a bf16 model IS
// bf16), so a decision row carries ~8 mantissa bits and two candidates can land
// on the same word. When they do, `torch.argmax` returns the lower index - and
// that is a property of torch's tie-break, not an output of the model. Asserting
// it would make this gate grade the engine on a coin toss (docs/14, "The
// RTN-checkpoint gate").
//
// This is NOT a tolerance and it is NOT a relaxation of the bar. On a row where
// the maximum IS unique the gate is exactly as strict as it always was: one id,
// element-exact. The ruling only declines to assert an answer the reference does
// not contain. It applies to BOTH checkpoints - the published one's `cjk` golden
// set has had two such rows since it was made.
struct GoldenDecision {
  std::vector<uint32_t> set;   // ascending; set[0] is what torch's argmax returned
  float value = 0.0f;
  bool determined() const { return set.size() == 1; }
  bool contains(uint32_t id) const {
    return std::find(set.begin(), set.end(), id) != set.end();
  }
};
// Same masking rule as argmax_masked, so the set and the argmax cannot disagree
// about which columns are eligible. fp32 `==` here is bit-equality for every
// value a finite logit can take, which is what "the reference does not
// distinguish them" means.
//
// **The set is empty only if the whole row is NaN.** NaN compares false against
// everything, so neither loop below would ever fire - and every caller indexes
// `set[0]`. `dump.py` does NOT rule that out: its finiteness abort covers
// `resid.L{n-1}` only, never a `logits` row (see its "Hard aborts" list). An
// earlier version of this comment said that it did, and that was wrong. The
// CHECK below turns a corrupt future golden into a named failure at a named
// line instead of undefined behaviour. It cannot fire on any golden file that
// exists today - every logit row in all nine is finite.
GoldenDecision golden_decision(const float* row, uint32_t n, uint32_t used) {
  GoldenDecision d;
  const uint32_t lim = n < used ? n : used;
  float bv = -INFINITY;
  for (uint32_t i = 0; i < lim; ++i)
    if (row[i] > bv) bv = row[i];
  d.value = bv;
  for (uint32_t i = 0; i < lim; ++i)
    if (row[i] == bv) d.set.push_back(i);
  CHECK(!d.set.empty());   // an all-NaN golden logits row - see above
  return d;
}

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

}  // namespace golden

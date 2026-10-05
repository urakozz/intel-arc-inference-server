// Spec 9 L1/L2 on the card (plan 9b Task 3 Step 1): the engine's int8-head logits against
// its bf16-head logits on the SAME hidden state, at M = 1 decode.
//
//   probe_lm_head_l1 <snapshot> <prompts dir> <golden dir> [<toolcall set> <toolcall out>]
//
// The engine runs with the checkpoint's bf16 head, teacher-forced: each golden prompt
// followed by its 32 golden tokens (the golden file's `tokens`) and, with the last two
// arguments, each A4 tool-call scenario (tests/golden/toolcall/<name>.ids) followed by
// the bf16 oracle's own output (<toolcall out>/<name>.bf16.ids) - plan 9a's sources, all
// 36 scenarios here. The context minus its last id is prefilled; every later id is one
// decode replay, and each replay is a decision row (plan 9a's rows: the last context id
// through the last continuation id). After each replay gemv_i8w runs on that step's own
// post-final-norm hidden `x` with the int8 head quantised on the host exactly as the
// loader does (loader::quantise_int8_tiled over the checkpoint's lm_head.weight), and
// its logits are compared with the bf16 head's from the same step.
//
// Reported, over ids < vocab_used (spec 9 §4 as amended in §8):
//   cosine min/mean; argmax mismatches and how many are bf16 near-ties (the int8 pick
//   within one bf16 ulp of the bf16 top-1); top-20 set equality, raw and under the
//   amendment's rule (a difference counts only if the bf16 rank-20/21 gap is >= 0.05);
//   KL(p_bf16 || p_int8) unfiltered at T 1.0 and 0.6 (mean, p99, max; bars 1e-4 / 1e-3);
//   and the dropped mass under generation_config's filter (top-k 20, top-p 0.95).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/shape_suffix.h"   // kRefHidden: the lm_head binaries' K (Qwen3.8)
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
#include "golden_common.h"

namespace {
using model::Qwen35;
constexpr uint32_t kHid = kernels::kRefHidden, kV = Qwen35::kVocab, kUsed = Qwen35::kVocabUsed;
constexpr uint32_t kGen = 32;

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream in(path);
  std::vector<uint32_t> ids;
  uint32_t id = 0;
  while (in >> id) ids.push_back(id);
  if (ids.empty()) throw std::runtime_error("no ids in " + path);
  return ids;
}

// The golden file's `tokens` (I32 [32]), through the golden gate's own reader.
std::vector<uint32_t> golden_tokens(const std::string& path) {
  golden::Golden g(path);
  const int32_t* t = g.i32("tokens", kGen);
  return std::vector<uint32_t>(t, t + kGen);
}

std::vector<double> softmax(const float* l, double T) {
  double mx = -1e300;
  for (uint32_t i = 0; i < kUsed; ++i) mx = std::max(mx, double(l[i]) / T);
  std::vector<double> p(kUsed);
  double z = 0;
  for (uint32_t i = 0; i < kUsed; ++i) z += (p[i] = std::exp(double(l[i]) / T - mx));
  for (double& v : p) v /= z;
  return p;
}
// Ids the engine's sample() keeps: top-k by logit, softmax at T, the top-p prefix whose
// mass first reaches p (src/cli/serve_adapters.h).
std::vector<uint32_t> kept(const float* l, double T, uint32_t k, double top_p) {
  std::vector<uint32_t> idx(kUsed);
  std::iota(idx.begin(), idx.end(), 0u);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](uint32_t a, uint32_t b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
  idx.resize(k);
  double mx = double(l[idx[0]]) / T, z = 0;
  std::vector<double> w(k);
  for (uint32_t i = 0; i < k; ++i) z += (w[i] = std::exp(double(l[idx[i]]) / T - mx));
  double c = 0;
  std::vector<uint32_t> out;
  for (uint32_t i = 0; i < k; ++i) {
    out.push_back(idx[i]);
    c += w[i] / z;
    if (c >= top_p) break;
  }
  return out;
}
double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))];
}
// Per-set accumulators for the metrics above.
struct Acc {
  size_t rows = 0, am_mis = 0, am_tie = 0, top_eq = 0, top_eq_amended = 0;
  double cos_min = 1, cos_sum = 0, max_gap_mis = 0, drop1 = 0, drop06 = 0;
  std::vector<double> kl1, kl06;
  void add(const float* a, const float* b) {
    ++rows;
    double ab = 0, aa = 0, bb = 0;
    for (uint32_t i = 0; i < kUsed; ++i)
      ab += double(a[i]) * b[i], aa += double(a[i]) * a[i], bb += double(b[i]) * b[i];
    const double c = ab / std::sqrt(aa * bb);
    cos_min = std::min(cos_min, c);
    cos_sum += c;
    const uint32_t ta = uint32_t(std::max_element(a, a + kUsed) - a);
    const uint32_t tb = uint32_t(std::max_element(b, b + kUsed) - b);
    if (ta != tb) {
      ++am_mis;
      int e = 0;
      std::frexp(std::fabs(a[ta]), &e);
      if (double(a[ta]) - double(a[tb]) <= std::ldexp(1.0, e - 8)) ++am_tie;
    }
    std::vector<uint32_t> ia(kUsed), ib(kUsed);
    std::iota(ia.begin(), ia.end(), 0u);
    std::iota(ib.begin(), ib.end(), 0u);
    auto by = [](const float* l) {
      return [l](uint32_t x, uint32_t y) { return l[x] > l[y] || (l[x] == l[y] && x < y); };
    };
    std::partial_sort(ia.begin(), ia.begin() + 21, ia.end(), by(a));
    std::partial_sort(ib.begin(), ib.begin() + 20, ib.end(), by(b));
    std::vector<uint32_t> sa(ia.begin(), ia.begin() + 20), sb(ib.begin(), ib.begin() + 20);
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    const double gap = double(a[ia[19]]) - double(a[ia[20]]);
    if (sa == sb) {
      ++top_eq;
      ++top_eq_amended;
    } else {
      max_gap_mis = std::max(max_gap_mis, gap);
      if (gap < 0.05) ++top_eq_amended;
    }
    for (double T : {1.0, 0.6}) {
      const std::vector<double> p = softmax(a, T), qd = softmax(b, T);
      double kl = 0;
      for (uint32_t i = 0; i < kUsed; ++i)
        if (p[i] > 0) kl += p[i] * (std::log(p[i]) - std::log(std::max(qd[i], 1e-300)));
      (T == 1.0 ? kl1 : kl06).push_back(std::max(kl, 0.0));
      const std::vector<uint32_t> ka = kept(a, T, 20, 0.95), kb = kept(b, T, 20, 0.95);
      double za = 0, dm = 0;
      for (uint32_t id : ka) za += p[id];
      for (uint32_t id : ka)
        if (std::find(kb.begin(), kb.end(), id) == kb.end()) dm += p[id] / za;
      double& d = T == 1.0 ? drop1 : drop06;
      d = std::max(d, dm);
    }
  }
  void merge(const Acc& o) {
    rows += o.rows, am_mis += o.am_mis, am_tie += o.am_tie, top_eq += o.top_eq;
    top_eq_amended += o.top_eq_amended;
    cos_min = std::min(cos_min, o.cos_min), cos_sum += o.cos_sum;
    max_gap_mis = std::max(max_gap_mis, o.max_gap_mis);
    drop1 = std::max(drop1, o.drop1), drop06 = std::max(drop06, o.drop06);
    kl1.insert(kl1.end(), o.kl1.begin(), o.kl1.end());
    kl06.insert(kl06.end(), o.kl06.begin(), o.kl06.end());
  }
  static double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
  }
  bool pass() const {
    return cos_min >= 0.9999 && am_mis == am_tie && double(top_eq_amended) >= 0.99 * double(rows) &&
           mean(kl1) <= 1e-4 && pct(kl1, 0.99) <= 1e-3 && mean(kl06) <= 1e-4 &&
           pct(kl06, 0.99) <= 1e-3;
  }
  void print(const char* name) const {
    std::printf("\n[%s] %zu rows\n", name, rows);
    std::printf("  cosine min %.7f mean %.7f  (bar >= 0.9999)\n", cos_min, cos_sum / double(rows));
    std::printf("  argmax mismatches %zu (bf16 near-ties %zu)\n", am_mis, am_tie);
    std::printf("  top-20 equal %zu/%zu = %.1f%%; amended (gap >= 0.05 counts) %zu/%zu = %.1f%%"
                " (bar 99%%); max bf16 rank-20/21 gap on a differing row %.4f\n",
                top_eq, rows, 100.0 * double(top_eq) / double(rows), top_eq_amended, rows,
                100.0 * double(top_eq_amended) / double(rows), max_gap_mis);
    std::printf("  KL unfiltered T 1.0: mean %.2e p99 %.2e max %.2e  (bars 1e-4 / 1e-3)\n",
                mean(kl1), pct(kl1, 0.99), pct(kl1, 1.0));
    std::printf("  KL unfiltered T 0.6: mean %.2e p99 %.2e max %.2e  (bars 1e-4 / 1e-3)\n",
                mean(kl06), pct(kl06, 0.99), pct(kl06, 1.0));
    std::printf("  dropped mass (k 20, p 0.95), max: T 1.0 %.4f, T 0.6 %.4f  (recorded, not gated)\n",
                drop1, drop06);
    std::printf("  amended L1/L2: %s\n", pass() ? "pass" : "FAIL");
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: probe_lm_head_l1 <snapshot> <prompts dir> <golden dir> "
                 "[<toolcall set dir> <toolcall out dir>]\n");
    return 2;
  }
  const std::string snap = loader::resolve_snapshot(argv[1]);
  l0::Context ctx(0);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // The int8 head, quantised on the host exactly as the loader does, resident beside the
  // engine's bf16 one (1.27 GB).
  loader::SafetensorsSet set(snap);
  const loader::LinearSrc src = loader::LinearSrc::classify(set, "lm_head");
  if (src.kind != loader::WKind::Bf16) throw std::runtime_error("the checkpoint's lm_head is not bf16");
  l0::Mem wq(ctx, l0::MemKind::Device, size_t(kV) * kHid), ws(ctx, l0::MemKind::Device, size_t(kV) * 4);
  {
    std::vector<int8_t> q(size_t(kV) * kHid);
    std::vector<float> s(kV);
    loader::quantise_int8_tiled(src.weight, kHid, kV, q.data(), s.data());
    imm.copy(wq.ptr(), q.data(), q.size());
    imm.copy(ws.ptr(), s.data(), s.size() * 4);
  }
  runtime::Engine eng(ctx, loader::load(ctx, snap, 16384), 16384);
  l0::Mem out(ctx, l0::MemKind::Device, size_t(kV) * 4);
  l0::Module mod(ctx, kernels::path(kernels::gemv_i8w_variant(1, kHid, kV)));
  l0::Kernel k = mod.kernel("gemv_i8w");
  k.group_size(kernels::kGemvI8wCols);
  k.arg_ptr(0, wq.ptr());
  k.arg_ptr(1, ws.ptr());
  k.arg_ptr(2, eng.buffers().x.ptr());   // the hidden the bf16 head just read
  k.arg_ptr(3, out.ptr());
  l0::Queue queue(ctx);
  l0::Fence fence(queue);
  l0::CmdList list = l0::CmdList::regular(ctx);
  list.launch(k, kV / kernels::kGemvI8wCols);
  list.close();
  std::vector<float> la(kV), lb(kV);
  // One decision row: the bf16 head's logits from the step just replayed, and gemv_i8w on
  // that step's own hidden.
  auto row = [&](Acc& acc) {
    queue.execute(list, &fence);
    fence.wait();
    imm.copy(la.data(), eng.buffers().logits.ptr(), size_t(kV) * 4);
    imm.copy(lb.data(), out.ptr(), size_t(kV) * 4);
    acc.add(la.data(), lb.data());
  };
  // ctx ids then cont ids, rows from the last ctx id through the last cont id.
  auto source = [&](const std::vector<uint32_t>& c, const std::vector<uint32_t>& cont, Acc& acc) {
    eng.reset();
    if (c.size() > 1) eng.prefill(std::vector<uint32_t>(c.begin(), c.end() - 1));
    eng.ingest({c.back()});
    row(acc);
    for (uint32_t id : cont) {
      eng.ingest({id});
      row(acc);
    }
  };

  Acc golden, tool;
  for (const char* p : {"prose", "code", "cjk"}) {
    const std::vector<uint32_t> c = read_ids(std::string(argv[2]) + "/" + p + ".ids");
    source(c, golden_tokens(std::string(argv[3]) + "/" + p + ".golden.safetensors"), golden);
    std::printf("golden %s: %zu ctx ids, rows so far %zu\n", p, c.size(), golden.rows);
  }
  golden.print("golden");
  if (argc > 5) {
    std::ifstream man(std::string(argv[4]) + "/manifest.json");
    std::stringstream ss;
    ss << man.rdbuf();
    const std::string m = ss.str();
    // names from the manifest's "name": "..." fields (plain scan; the file is ours).
    for (size_t at = m.find("\"name\""); at != std::string::npos; at = m.find("\"name\"", at + 1)) {
      const size_t q0 = m.find('"', m.find(':', at) + 1), q1 = m.find('"', q0 + 1);
      const std::string name = m.substr(q0 + 1, q1 - q0 - 1);
      const std::vector<uint32_t> c = read_ids(std::string(argv[4]) + "/" + name + ".ids");
      const std::vector<uint32_t> cont = read_ids(std::string(argv[5]) + "/" + name + ".bf16.ids");
      source(c, cont, tool);
      std::printf("toolcall %s: %zu ctx + %zu cont ids, rows so far %zu\n", name.c_str(), c.size(),
                  cont.size(), tool.rows);
    }
    tool.print("toolcall");
  }
  Acc all = golden;
  all.merge(tool);
  all.print("all");
  std::printf("L1/L2 (amended, all rows): %s\n", all.pass() ? "PASS" : "FAIL");
  return all.pass() ? 0 : 1;
}

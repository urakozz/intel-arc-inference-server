// Spec 21e Task 3 Step 3 (spec 21 F5; plan 21e Review Focus 2, 3, 4): Qwen3.8-Flash-Next's MTP head and spec 8's draft /
// verify / commit on the card - box only (labels checkpoint;qwen4exp). The synthetic checkpoints (21b's make_synth.py
// --mtp: 4 real-width layers and the head, its bf16 experts RTN int4 g64 at load) carry it; Intel's --layers 18 when
// present.
//
//   qwen4exp_mtp_test <checkpoint> <fixture dir> <ids> [m1] [m2] [m3] [cost] [pp2[:split]] [layers:N] [int8]
//
// (no mode: all four). <fixture dir> holds tools/oracle/qwen4exp_mtp_fixture.py's mtp_{single,per_stream}.safetensors
// for this checkpoint (box: r34.fixture); without them M1 SKIPs and the rest runs.
//   M1  the head against 21a's port (tools/oracle/qwen4exp_mtp.py MtpHead, vLLM's form, the engine-format experts:
//       RTN int4 g64 dequantised), both pre_fc_norm_hidden forms (B70_Q4_MTP_NORM): per fixture sample i the engine
//       at pos 1 (one id prefilled), R_0 replaced by the fixture's R_i (write_mtp_R), the pending id t_i, draft(3) -
//       steps at positions 0, 1, 2, steps 1 and 2 on step 0's list (decision 5) - against chain(R_i, [t_i], 3):
//       every step's logits cosine >= 0.9999 (PROPOSED), its id tie-aware equal (golden_decision; the comparison
//       stops at the first legitimate divergence), the last step's pre-mixer H cosine >= 0.9999 (PROPOSED);
//   M2  (F5, Review Focus 2) verify rows are decode rows: from S0 (a prompt prefilled - a short one, and 2100 ids so
//       the selection is active) the reference is 4 plain steps (verify(0) + commit(0)); for k = 1..3 the restored S0
//       verifies the true greedy drafts at M = k + 1 and row r is BITWISE step r: logits, route rows, every QSA
//       layer's selection, K / V written at pos + r, the GDN state in slot (live + r) % 4; then commit(k) and 8 more
//       plain ids equal the reference's;
//   M3  (F5) greedy lossless: 64 ids by draft / verify / commit at K = 1, 2, 3 (spec 8's greedy acceptance) equal
//       generate(64) with the head - and with the head off (a second engine): the short and the 2100-id prompts;
//   cost the verify's cost: ms a draft step and a verify at K = 0..3 (median of 16 iterations, interleaved), and per
//       layer the union of experts over the verify rows at K = 3 (read_verify_routes: spec 22 P0.6's MTP term) -
//       recorded, no bar.
// Exit 77 (SKIP) when the checkpoint is absent, has no MTP head, or two GPUs are needed.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "qwen4exp_rig.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {

namespace rq = runtime::qwen4exp;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192;
constexpr double kCos = 0.9999;   // PROPOSED (plan 21e Task 3 Step 3)

double cosine(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb + 1e-300);
}
float bf(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
double cosine_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  std::vector<float> fa(n), fb(n);
  for (size_t i = 0; i < n; ++i) {
    fa[i] = bf(a[i]);
    fb[i] = bf(b[i]);
  }
  return cosine(fa.data(), fb.data(), n);
}

Ids cycle(const Ids& base, size_t n) {
  Ids v(n);
  for (size_t i = 0; i < n; ++i) v[i] = base[i % base.size()];
  return v;
}

void set_env(const char* k, const char* v) {
#ifdef _WIN32
  _putenv_s(k, v);
#else
  setenv(k, v, 1);
#endif
}

// --- M1 -----------------------------------------------------------------------------------------------------------
int run_m1(qwen4exp_rig::Rig& rig, const std::string& snap, qwen4exp_rig::Options o, const std::string& fixdir) {
  int bad = 0, ran = 0;
  for (const char* norm : {"single", "per_stream"}) {
    const std::string path = fixdir + "/mtp_" + norm + ".safetensors";
    if (!golden::exists(path)) {
      std::printf("M1 %s: SKIP (no %s: tools/oracle/qwen4exp_mtp_fixture.py, r34.fixture)\n", norm, path.c_str());
      continue;
    }
    set_env("B70_Q4_MTP_NORM", norm);
    o.mtp = true;
    qwen4exp_rig::build(rig, snap, o);
    rq::Qwen4ExpEngine& e = *rig.eng;
    const golden::Golden g(path);
    const uint32_t V = e.vocab(), HN = e.model().desc.hc_n(), K = 3;
    const uint32_t S = uint32_t(g.dim("R", 2, 0));
    const uint16_t* R = g.bf16("R", size_t(S) * HN);
    const int32_t* tok = g.i32("tokens", S);
    const float* lg = g.f32("logits", size_t(S) * K * V);
    const int32_t* ids = g.i32("draft", size_t(S) * K);
    const uint16_t* M3 = g.bf16("M", size_t(S) * HN);
    std::vector<float> q(V);
    double worst = 1.0, worst_h = 1.0;
    uint32_t steps = 0, undetermined = 0, diverged = 0;
    for (uint32_t i = 0; i < S; ++i) {
      e.reset();
      e.prefill({uint32_t(tok[i])});   // pos 1: the head's step 0 at position 0 on an empty head cache
      e.write_mtp_R(std::vector<uint16_t>(R + size_t(i) * HN, R + size_t(i + 1) * HN));
      e.set_token(uint32_t(tok[i]));
      e.draft(K);
      bool same = true;
      for (uint32_t s = 0; s < K && same; ++s) {
        e.read_draft_logits_into(q.data(), s);
        const float* ref = lg + (size_t(i) * K + s) * V;
        const double c = cosine(q.data(), ref, V);
        worst = std::min(worst, c);
        const golden::GoldenDecision dec = golden::golden_decision(ref, V, e.model().desc.vocab_used);
        const uint32_t got = e.draft_ids()[s];
        ++steps;
        if (!dec.determined()) ++undetermined;
        if (c < kCos || !dec.contains(got)) {
          std::fprintf(stderr, "M1 %s sample %u step %u: cosine %.6f, engine id %u, reference %d (set of %zu)\n", norm, i,
                       s, c, got, ids[size_t(i) * K + s], dec.set.size());
          ++bad;
        }
        if (got != uint32_t(ids[size_t(i) * K + s])) {   // a tie the reference broke its own way: the chains part
          same = false;
          ++diverged;
        }
      }
      if (same) {
        const std::vector<uint16_t> H = e.read_draft_H();
        const double ch = cosine_bf16(H.data(), M3 + size_t(i) * HN, HN);
        worst_h = std::min(worst_h, ch);
        if (ch < kCos) {
          std::fprintf(stderr, "M1 %s sample %u: the last step's pre-mixer H cosine %.6f\n", norm, i, ch);
          ++bad;
        }
      }
    }
    ++ran;
    std::printf("M1 %s (%s selection): %u samples x %u steps (%u compared, %u undetermined, %u chains parted on a tie): "
                "logits cosine min %.6f, pre-mixer H cosine min %.6f%s\n", norm, rq::mtp_select_name(e.mtp_select()), S,
                K, steps, undetermined, diverged, worst, worst_h, bad ? "" : " - PASS");
  }
  set_env("B70_Q4_MTP_NORM", "single");
  if (ran == 0) std::printf("M1: SKIP (no fixture)\n");
  return bad;
}

// --- M2 -----------------------------------------------------------------------------------------------------------
struct StepRef {
  std::vector<float> logits;
  std::vector<uint32_t> routes;                     // [layers][32]
  std::vector<std::vector<uint32_t>> sel;           // per QSA layer
  std::vector<std::vector<uint16_t>> k, v;          // per QSA layer: the row written at pos + r
  std::vector<std::vector<float>> gdn;              // per GDN layer: the live state after the step
  uint32_t id = 0;
};

StepRef record(rq::Qwen4ExpEngine& e, uint32_t r, uint32_t M, uint32_t n) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  StepRef s;
  std::vector<float> rows(size_t(M) * d.vocab);
  e.read_logits_into(rows.data(), M);
  s.logits.assign(rows.begin() + size_t(r) * d.vocab, rows.begin() + size_t(r + 1) * d.vocab);
  const std::vector<uint32_t> rt = e.read_verify_routes();
  for (uint32_t l = 0; l < d.layers; ++l)
    s.routes.insert(s.routes.end(), rt.begin() + (size_t(l) * M + r) * rq::kRouteWords,
                    rt.begin() + (size_t(l) * M + r + 1) * rq::kRouteWords);
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.is_qsa(l)) {
      s.sel.push_back(e.read_verify_selection(l, r));
      s.k.push_back(e.read_kv(l, n + r, 1, false));
      s.v.push_back(e.read_kv(l, n + r, 1, true));
    } else {
      s.gdn.push_back(e.read_gdn_slot(l, (e.gdn_live() + r) % rq::kGdnSlots));
    }
  }
  s.id = e.verify_ids()[r];
  return s;
}

std::string first_diff(const StepRef& a, const StepRef& b) {
  if (a.logits != b.logits) return "logits";
  if (a.routes != b.routes) return "route rows";
  if (a.sel != b.sel) return "selections";
  if (a.k != b.k || a.v != b.v) return "K / V";
  if (a.gdn != b.gdn) return "GDN state";
  if (a.id != b.id) return "argmax";
  return "";
}

int run_m2(rq::Qwen4ExpEngine& e, const Ids& base) {
  int bad = 0;
  for (const size_t len : {size_t(120), size_t(2100)}) {
    const Ids prompt = cycle(base, len);
    e.reset();
    e.prefill(prompt);
    const uint32_t n = e.pos();
    std::vector<uint8_t> st(e.state_bytes()), kv(e.kv_bytes(n));
    e.save_state(st.data());
    e.save_kv(0, n, kv.data());
    Ids g{e.pending()};
    // the reference: plain steps (verify(0) + commit(0)) from S0
    std::vector<StepRef> ref;
    for (uint32_t r = 0; r < rq::kVerifyRows; ++r) {
      e.verify(0);
      ref.push_back(record(e, 0, 1, n + r));
      g.push_back(ref.back().id);
      e.commit(0, ref.back().id);
    }
    // the greedy sequence from S0: g[0..3] stepped above, then generate's ids (its first is g[4], the pending one)
    Ids full(g.begin(), g.begin() + rq::kVerifyRows);
    const Ids cont = e.generate(8);
    full.insert(full.end(), cont.begin(), cont.end());
    for (uint32_t k = 1; k <= rq::kMaxDraft; ++k) {
      e.reset();
      e.load_kv(0, n, kv.data());
      e.load_state(st.data(), n);
      e.set_token(g[0]);
      for (uint32_t i = 0; i < k; ++i) e.set_draft_input(i, g[1 + i]);
      e.verify(k);
      for (uint32_t r = 0; r <= k; ++r) {
        // row r's GDN slot is (live + r) % 4 with live 0 after load_state; record() reads that slot
        const std::string diff = first_diff(record(e, r, k + 1, n), ref[r]);
        if (!diff.empty()) {
          std::fprintf(stderr, "M2 prompt %zu, k %u: row %u differs from the plain step %u at %s\n", len, k, r, r,
                       diff.c_str());
          ++bad;
        }
      }
      e.commit(k, g[k + 1]);
      // after commit(k) pos is n + k + 1 and the pending id g[k + 1]: generate's 8 ids are full[k + 1 .. k + 9)
      const Ids after = e.generate(8);
      const Ids want(full.begin() + k + 1, full.begin() + k + 9);
      if (after != want) {
        std::fprintf(stderr, "M2 prompt %zu, k %u: the ids after commit(%u) are not the plain run's\n", len, k, k);
        ++bad;
      }
    }
    std::printf("M2 prompt of %zu ids (pos %u): verify rows at M = 2, 3, 4 %s M = 1's steps (logits, routes, "
                "selections, K / V, GDN slots, ids)\n", len, n, bad ? "DIFFER from" : "bitwise");
  }
  return bad;
}

// --- M3 -----------------------------------------------------------------------------------------------------------
// spec 8's greedy loop: draft k, verify, accept the drafts while they equal the verify's argmax, commit.
Ids speculative(rq::Qwen4ExpEngine& e, uint32_t K, uint32_t n, uint64_t* drafted = nullptr, uint64_t* accepted = nullptr) {
  Ids out;
  while (out.size() < n) {
    const uint32_t k = std::min(K, e.max_verify_k());
    if (k == 0) {
      out.push_back(e.generate(1)[0]);
      continue;
    }
    const uint32_t x = e.pending();
    e.draft(k);
    e.verify(k);
    const uint32_t* v = e.verify_ids();
    uint32_t j = 0;
    while (j < k && e.draft_ids()[j] == v[j]) ++j;
    out.push_back(x);
    for (uint32_t i = 0; i < j; ++i) out.push_back(e.draft_ids()[i]);
    if (drafted) *drafted += k;
    if (accepted) *accepted += j;
    e.commit(j, v[j]);
  }
  out.resize(n);
  return out;
}

int run_m3(qwen4exp_rig::Rig& rig, const std::string& snap, qwen4exp_rig::Options o, const Ids& base) {
  int bad = 0;
  std::vector<Ids> plain;
  o.mtp = false;
  qwen4exp_rig::build(rig, snap, o);
  for (const size_t len : {size_t(120), size_t(2100)}) {
    rig.eng->reset();
    rig.eng->prefill(cycle(base, len));
    plain.push_back(rig.eng->generate(64));
  }
  o.mtp = true;
  qwen4exp_rig::build(rig, snap, o);
  rq::Qwen4ExpEngine& e = *rig.eng;
  size_t pi = 0;
  for (const size_t len : {size_t(120), size_t(2100)}) {
    const Ids prompt = cycle(base, len);
    e.reset();
    e.prefill(prompt);
    const Ids with_head = e.generate(64);
    if (with_head != plain[pi]) {
      std::fprintf(stderr, "M3 prompt %zu: generate() with the head is not the plain engine's\n", len);
      ++bad;
    }
    for (uint32_t K = 1; K <= rq::kMaxDraft; ++K) {
      e.reset();
      e.prefill(prompt);
      uint64_t dr = 0, ac = 0;
      const Ids got = speculative(e, K, 64, &dr, &ac);
      const bool ok = got == plain[pi];
      bad += ok ? 0 : 1;
      std::printf("M3 prompt %zu, K %u: 64 ids %s the plain run (drafts accepted %llu / %llu - a truncated or "
                  "synthetic model's acceptance means nothing, spec 22 measures it)\n", len, K,
                  ok ? "bitwise" : "DIFFER from", (unsigned long long)ac, (unsigned long long)dr);
    }
    ++pi;
  }
  return bad;
}

// --- the verify's cost --------------------------------------------------------------------------------------------
void run_cost(rq::Qwen4ExpEngine& e, const Ids& base) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  e.reset();
  e.prefill(cycle(base, 2100));
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); };
  std::vector<std::vector<double>> v(rq::kMaxDraft + 1), dr(rq::kMaxDraft + 1);
  for (int it = 0; it < 16; ++it)
    for (uint32_t k = 0; k <= rq::kMaxDraft; ++k) {   // interleaved
      if (k > 0) {
        const auto t0 = Clock::now();
        e.draft(k);
        dr[k].push_back(ms(t0) / k);
      }
      const auto t1 = Clock::now();
      e.verify(k);
      v[k].push_back(ms(t1));
      e.commit(0, e.verify_ids()[0]);
    }
  const auto median = [](std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x.empty() ? 0.0 : x[x.size() / 2];
  };
  std::printf("cost (%u layers, %u device(s), %s attention): verify M = 1 %.3f ms", d.layers, e.devices(),
              rq::q4_attn_name(e.attention()), median(v[0]));
  for (uint32_t k = 1; k <= rq::kMaxDraft; ++k)
    std::printf(", M = %u %.3f ms (x%.2f)", k + 1, median(v[k]), median(v[k]) / median(v[0]));
  std::printf("; a draft step %.3f ms (step 0 selects)\n", median(dr[1]));
  std::printf("cost: launches - verify M = 1..4: %zu %zu %zu %zu, draft steps %zu %zu %zu, plain %zu\n",
              e.verify_list_launches(1), e.verify_list_launches(2), e.verify_list_launches(3), e.verify_list_launches(4),
              e.draft_list_launches(0), e.draft_list_launches(1), e.draft_list_launches(2), e.launches());
  // the per-layer union of experts over the 4 verify rows (spec 22 P0.6's MTP term)
  e.set_draft_input(0, e.pending());
  e.set_draft_input(1, e.pending());
  e.set_draft_input(2, e.pending());
  e.verify(rq::kMaxDraft);
  const std::vector<uint32_t> rt = e.read_verify_routes();
  e.commit(0, e.verify_ids()[0]);
  const uint32_t M = rq::kVerifyRows;
  std::printf("cost: experts a layer over the %u verify rows (union of 10 each; the shared expert apart):", M);
  double sum = 0;
  for (uint32_t l = 0; l < d.layers; ++l) {
    std::set<uint32_t> u;
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t s = 0; s < d.top_k; ++s) u.insert(rt[(size_t(l) * M + r) * rq::kRouteWords + s]);
    sum += double(u.size());
    std::printf(" L%u %zu", l, u.size());
  }
  std::printf("; mean %.2f (10 = every row the same experts, 40 = all distinct)\n", sum / d.layers);
}

}  // namespace

int main(int argc, char** argv) {
  std::string why;
  const std::string snapdir = qwen4exp_rig::qwen4exp_snapshot(argc > 1 ? argv[1] : "", &why);
  if (snapdir.empty()) {
    std::printf("SKIP: no Qwen3.8-Flash-Next checkpoint at '%s' (%s)\n", argc > 1 ? argv[1] : "", why.c_str());
    return 77;
  }
  const std::string fixdir = argc > 2 ? argv[2] : "";
  qwen4exp_rig::Options o;
  o.max_len = kMaxLen;
  bool m1 = false, m2 = false, m3 = false, cost = false;
  for (int i = 4; i < argc; ++i) {
    const std::string x = argv[i];
    if (x == "m1") m1 = true;
    else if (x == "m2") m2 = true;
    else if (x == "m3") m3 = true;
    else if (x == "cost") cost = true;
    else if (x == "pp2") o.devices = 2;
    else if (x.rfind("pp2:", 0) == 0) {
      o.devices = 2;
      o.split = uint32_t(std::strtoul(x.c_str() + 4, nullptr, 10));
    } else if (x.rfind("layers:", 0) == 0) {
      o.layers = uint32_t(std::strtoul(x.c_str() + 7, nullptr, 10));
    } else if (x == "int8") {
      o.int8_head = true;
    } else {
      std::fprintf(stderr, "qwen4exp_mtp_test: unknown flag %s (m1 m2 m3 cost pp2[:split] layers:N int8)\n", x.c_str());
      return 2;
    }
  }
  if (!m1 && !m2 && !m3 && !cost) m1 = m2 = m3 = cost = true;
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::Qwen4ExpDesc pre = loader::qwen4exp_checkpoint_desc(snapdir, o.layers);
  const Ids base = qwen4exp_rig::read_ids(argc > 3 ? argv[3] : "", pre.vocab);
  CHECK(!base.empty());
  qwen4exp_rig::Rig rig;
  int bad = 0;
  try {
    if (m1) bad += run_m1(rig, snapdir, o, fixdir);
    if (m2 || cost) {
      o.mtp = true;
      qwen4exp_rig::build(rig, snapdir, o);
      std::printf("engine: %u layers, %u device(s), the head (%s norm, %s selection); verify launches M = 1..4: %zu %zu "
                  "%zu %zu; draft steps %zu / %zu\n", rig.eng->model().desc.layers, rig.eng->devices(),
                  rq::mtp_norm_name(rig.eng->mtp_norm()), rq::mtp_select_name(rig.eng->mtp_select()),
                  rig.eng->verify_list_launches(1), rig.eng->verify_list_launches(2), rig.eng->verify_list_launches(3),
                  rig.eng->verify_list_launches(4), rig.eng->draft_list_launches(0), rig.eng->draft_list_launches(1));
      if (m2) bad += run_m2(*rig.eng, base);
      if (cost) run_cost(*rig.eng, base);
    }
    if (m3) bad += run_m3(rig, snapdir, o, base);
  } catch (const std::exception& ex) {
    const std::string w = ex.what();
    if (w.find("mtp") != std::string::npos && w.find("load") != std::string::npos) {
      std::printf("SKIP: %s\n", w.c_str());
      return 77;
    }
    throw;
  }
  std::printf("qwen4exp_mtp_test %s\n", bad == 0 ? "OK" : "FAILED");
  return bad == 0 ? 0 : 1;
}

// Spec 15e's MTP gates on Ornith 1.5 35B-A3B, on the card (needs the Ornith int4
// checkpoint; exit 77 - SKIP - without it, as 15c / 15d's tests do):
//
//   L   the lists: the head is the MoE head (its bf16 experts int4 g64 RTN at load, 771
//       linears from the published head), its device bytes are loader::mtp_head_bytes; the
//       verify lists are verify_launches() = 536 at M = 1..4, the drafts draft_launches() =
//       24 (the head's FFN is the 4-launch MoE block);
//   M2  (spec 15e Review Focus 3; spec 8 A6 on Ornith) from a prefilled state S0 and the
//       true greedy drafts g[1..k], verify(k) for k = 1..3. Since spec 28b the verify rows'
//       dense projections run gemm_sm (DPAS) and are no longer bitwise the M = 1 rows, so the
//       bars are spec 28 N4 + N3 (as on Qwen3.8): every row r against the r-th plain step
//       from S0 (verify(0) + commit) - logits cosine >= 0.99999, GDN state slot cosine >=
//       0.99999, its id equal or a golden near-tie of the plain row (tests/server/greedy_tie.h);
//       the 40 layers' route rows against the plain step are RECORDED (identical rows counted:
//       a router input that moved in the last bits may pick another expert at a near-tie); and
//       N3 - every row of verify(k) BITWISE equal (logits, GDN slot, route rows) to the same
//       row of verify(3) (M = 4): a row does not depend on M >= 2. (Before 28b: bitwise
//       against the plain step - the MoE block at M rows routes each row as M = 1 does, which
//       still holds, now under N3.)
//   M3  greedy MTP is lossless (`--mtp K` == `--mtp 0`): K = 1, 2, 3 iterations (draft,
//       verify, accept the agreeing prefix, commit) emit the greedy ids of an engine loaded
//       WITHOUT the head (the plain decode list), 128 of them, on prose / code / cjk, after
//       an l0-int8 prefill (which fills the head's KV through step_mtp_kv) - identical, or from
//       the first divergence the f32 judge rejects no more of its ids than of the plain run's
//       (spec 28 N6, the operator's ruling 2026-11-03; tests/server/spec_judge.h), the golden
//       tie rule without a judge; M2's logits rows likewise by the f32 judge where one exists;
//   H   the head's own MoE layer ran: after a draft, the MoE scratch's head slot (layer 40)
//       holds a route row of 8 distinct expert ids < 256 whose weights sum to ~1.
//
// The acceptance rate and the verify cost at K = 1..3 (Review Focus 2) are printed, not
// gated - the box records them.
//
// usage: ornith_mtp_test <snapshot> <prompts dir> [lm_head form: bf16|int8]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "../server/greedy_tie.h"
#include "../server/spec_judge.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "loader/moe_layout.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/buffer_sizes.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
using model::Qwen35;
constexpr uint32_t kMaxLen = 16384;
constexpr size_t V = Qwen35::kVocab, VU = Qwen35::kVocabUsed;

struct Ctx {
  l0::Context& ctx;
  runtime::Engine& e;
  runtime::Control* ctl;
  l0::CmdList imm;
  size_t gdn_bytes;
};

struct Snap {   // state + KV [0, pos) + the pending id (mtp_verify_test's)
  l0::Mem st, kv;
  uint32_t pos, pending;
  explicit Snap(Ctx& c)
      : st(c.ctx, l0::MemKind::Host, c.e.state_bytes()),
        kv(c.ctx, l0::MemKind::Host, c.e.kv_bytes(c.e.pos())), pos(c.e.pos()),
        pending(c.ctl->cur_token[0]) {
    c.e.save_state(st.ptr());
    c.e.save_kv(0, pos, kv.ptr());
  }
  void restore(Ctx& c) const {
    c.e.load_state(st.ptr(), pos);
    c.e.load_kv(0, pos, kv.ptr());
    c.ctl->cur_token[0] = pending;
    c.ctl->n_active = 1;
  }
};

std::vector<uint8_t> read(Ctx& c, const void* dev, size_t bytes) {
  std::vector<uint8_t> v(bytes);
  c.imm.copy(v.data(), dev, bytes);
  return v;
}
std::vector<uint8_t> gdn_slot(Ctx& c, uint32_t s) {
  const void* src = s == 0 ? c.e.buffers().gdn_state.ptr()
                           : c.e.mtp_buffers()->gdn_spec.as<uint8_t>() + size_t(s - 1) * c.gdn_bytes;
  return read(c, src, c.gdn_bytes);
}
// Every main layer's route row `m` (moe_scratch_layout: [layer slot][kM][32] u32).
std::vector<uint8_t> routes(Ctx& c, uint32_t m) {
  const model::ModelDesc& d = *c.e.model().desc;
  const runtime::MoeScratchLayout ml = runtime::moe_scratch_layout(d);
  const size_t row = size_t(runtime::kMoeRouteWords) * 4;
  std::vector<uint8_t> out;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::vector<uint8_t> r =
        read(c, c.e.buffers().moe->as<uint8_t>() + ml.route_at(l) + size_t(m) * row, row);
    out.insert(out.end(), r.begin(), r.end());
  }
  return out;
}

struct Ref {
  std::vector<uint32_t> g;                  // g[0] pending at S0, g[i+1] = step i's argmax
  std::vector<std::vector<uint8_t>> logits, state, route;   // step r <= kMaxDraft
};
Ref reference(Ctx& c, const Snap& s0, uint32_t T) {
  s0.restore(c);
  Ref r;
  r.g.push_back(c.ctl->cur_token[0]);
  for (uint32_t i = 0; i < T; ++i) {
    c.e.verify(0);
    const bool keep = i <= runtime::Engine::kMaxDraft;
    if (keep) {
      r.logits.push_back(read(c, c.e.verify_logits_device(), V * 4));
      r.route.push_back(routes(c, 0));
    }
    c.e.commit(0, c.e.verify_ids()[0]);
    if (keep) r.state.push_back(gdn_slot(c, c.ctl->gdn_live));
    r.g.push_back(c.ctl->cur_token[0]);
  }
  return r;
}

double cosine(const uint8_t* a8, const uint8_t* b8, size_t n) {
  const float* a = reinterpret_cast<const float*>(a8);
  const float* b = reinterpret_cast<const float*>(b8);
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb);
}
// Route rows: how many of the 40 layers' rows are identical to the reference's.
size_t same_routes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  const size_t row = size_t(runtime::kMoeRouteWords) * 4;
  size_t n = 0;
  for (size_t off = 0; off + row <= a.size(); off += row) n += std::memcmp(a.data() + off, b.data() + off, row) == 0;
  return n;
}

bool m2(Ctx& c, const Snap& s0, const Ref& ref, const char* what, const std::vector<uint32_t>& prompt,
        const std::string& head_form) {
  bool ok = true;
  const uint32_t kK = runtime::Engine::kMaxDraft;
  // spec 28 N4 by the f32 judge (the operator's ruling 2026-11-03): the context prompt + g[0 .. kK + 1]
  const spec_judge::RowJudge rj = spec_judge::open_rows(
      golden::ForcedJudge{}, std::string("ornith_mtp_m2_") + head_form + "_" + what, prompt,
      std::vector<uint32_t>(ref.g.begin(), ref.g.begin() + kK + 2), uint32_t(V));
  // N3's reference: verify(kK) - M = 4 - with the true drafts, every row's logits, slot and routes
  std::vector<std::vector<uint8_t>> lg4(kK + 1), st4(kK + 1), rt4(kK + 1);
  for (int pass = 0; pass < 2; ++pass)
    for (uint32_t k = pass == 0 ? kK : 1; pass == 0 ? k == kK : k <= kK; ++k) {
      s0.restore(c);
      for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
      c.e.verify(k);
      const std::vector<uint8_t> lg = read(c, c.e.verify_logits_device(), size_t(k + 1) * V * 4);
      for (uint32_t r = 0; r <= k; ++r) {
        const std::vector<uint8_t> row(lg.begin() + size_t(r) * V * 4, lg.begin() + size_t(r + 1) * V * 4);
        const std::vector<uint8_t> st = gdn_slot(c, (c.ctl->gdn_live + r) % runtime::MtpBuffers::kSlots);
        const std::vector<uint8_t> rt = routes(c, r);
        if (pass == 0) {
          lg4[r] = row;
          st4[r] = st;
          rt4[r] = rt;
          continue;
        }
        // N4 against the plain step
        const double cl = cosine(row.data(), ref.logits[r].data(), VU);
        const double cs = cosine(st.data(), ref.state[r].data(), st.size() / 4);
        const size_t rs = same_routes(rt, ref.route[r]);
        const uint32_t id = c.e.verify_ids()[r];
        const greedy_tie::RefStep rs_plain =
            greedy_tie::ref_step(reinterpret_cast<const float*>(ref.logits[r].data()), uint32_t(VU));
        const bool id_ok = id == ref.g[r + 1] ||
                           (rs_plain.top1 == ref.g[r + 1] && rs_plain.second == id && rs_plain.margin_ulps <= 1.0);
        // N3 against the same row at M = 4
        const bool n3 = row == lg4[r] && st == st4[r] && rt == rt4[r];
        const spec_judge::RowVerdict rv =
            spec_judge::row(reinterpret_cast<const float*>(row.data()),
                            reinterpret_cast<const float*>(ref.logits[r].data()), rj, r + 1, cl, uint32_t(VU));
        std::printf("  M2 %s k=%u row %u: logits cos %.9f, GDN slot cos %.9f, routes %zu / 40 identical to the plain "
                    "step (recorded), id %s; N3: %s the M = 4 row%s\n",
                    what, k, r, cl, cs, rs, id == ref.g[r + 1] ? "ok" : id_ok ? "near-tie" : "DIFFERS",
                    n3 ? "bitwise" : "DIFFERS from", spec_judge::row_line(rv).c_str());
        ok &= rv.pass && cs >= 0.99999 && id_ok && n3;
      }
      if (pass == 1) c.e.commit(k, c.e.verify_ids()[k]);
    }
  return ok;
}

// One greedy speculative iteration at depth k: the ids it emits (mtp_verify_test's).
std::vector<uint32_t> iterate(Ctx& c, uint32_t k, size_t* drafted, size_t* accepted) {
  std::vector<uint32_t> out{c.ctl->cur_token[0]};
  const uint32_t kk = std::min(k, c.e.max_verify_k());
  if (kk == 0) {
    c.e.verify(0);
    c.e.commit(0, c.e.verify_ids()[0]);
    return out;
  }
  c.e.draft(kk);
  c.e.verify(kk);
  uint32_t j = 0;
  while (j < kk && c.e.draft_ids()[j] == c.e.verify_ids()[j]) ++j;
  for (uint32_t i = 0; i < j; ++i) out.push_back(c.e.draft_ids()[i]);
  c.e.commit(j, c.e.verify_ids()[j]);
  *drafted += kk;
  *accepted += j;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;   // b70-serve's default (spec 9)
  if (argc > 3) CHECK(loader::parse_lm_head_form(argv[3], lm_head));
  try {
    (void)loader::resolve_snapshot(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no Ornith int4 checkpoint at '%s' (%s). None is published; spec 15a "
                "decides its source - see the box validation queue.\n",
                snap.c_str(), e.what());
    return 77;
  }
  l0::Context ctx(0);
  constexpr uint32_t kGen = 128;
  const char* const prompts[] = {"prose", "code", "cjk"};
  // M3's reference first, from an engine without the head (two engines do not fit at once).
  std::vector<std::vector<uint32_t>> plain;
  std::vector<std::vector<greedy_tie::RefStep>> plain_steps;   // the row that decided each plain id (N6)
  {
    runtime::Engine p0(ctx, loader::load(ctx, snap, kMaxLen, /*mtp=*/false, lm_head), kMaxLen);
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    std::vector<float> row(V);
    for (const char* p : prompts) {
      p0.reset();
      p0.prefill(golden::read_ids(pdir + "/" + p + ".ids"));
      std::vector<uint32_t> ids;
      std::vector<greedy_tie::RefStep> st{{UINT32_MAX, UINT32_MAX, 1e30}};   // the prefill's id: never a tie
      for (uint32_t i = 0; i < kGen; ++i) {
        ids.push_back(p0.generate(1)[0]);
        imm.copy(row.data(), p0.buffers().logits.ptr(), row.size() * 4);
        st.push_back(greedy_tie::ref_step(row.data(), uint32_t(VU)));
      }
      plain.push_back(ids);
      plain_steps.push_back(st);
    }
  }
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/true, lm_head);
  const model::ModelDesc& d = *model.desc;
  CHECK(d.mtp_head_moe());

  // ---- L: the head and the lists ----------------------------------------------------
  CHECK(model.mtp && model.mtp->moe && !model.mtp->gate_up);
  CHECK_EQ(model.report.mtp_bytes, loader::mtp_head_bytes(d));
  std::printf("head: %zu tensors, %.3f GB checkpoint, %.3f GB device, %zu expert linears RTN at "
              "load in %.1f s\n",
              model.report.mtp_tensors, model.report.mtp_checkpoint_bytes / 1e9,
              model.report.mtp_bytes / 1e9, model.report.mtp_rtn_linears,
              model.report.mtp_rtn_seconds);
  if (model.report.mtp_rtn_linears == 3 * size_t(d.moe.blocks())) {   // the published head
    CHECK_EQ(model.report.mtp_tensors, d.mtp_checkpoint_tensors());
    CHECK_EQ(model.report.mtp_checkpoint_bytes, d.mtp_checkpoint_bytes());
  }
  CHECK_EQ(model.report.unconsumed, size_t(0));
  runtime::Engine e(ctx, std::move(model), kMaxLen);
  CHECK(e.mtp());
  CHECK_EQ(runtime::verify_launches(d), size_t(536));
  CHECK_EQ(runtime::draft_launches(d), size_t(24));
  for (uint32_t M = 1; M <= runtime::MtpBuffers::kSlots; ++M)
    CHECK_EQ(e.verify_step(M).kernel_count, runtime::verify_launches(d));
  for (uint32_t i = 0; i < runtime::Engine::kMaxDraft; ++i)
    CHECK_EQ(e.draft_step(i).kernel_count, runtime::draft_launches(d));
  std::printf("lists: verify M=1..4 %zu launches, draft %zu\n", e.verify_step(4).kernel_count,
              e.draft_step(0).kernel_count);

  Ctx c{ctx, e, e.buffers().control.as<runtime::Control>(), l0::CmdList::immediate(ctx),
        size_t(d.gdn_layers) * d.gdn_v_heads * 128 * 128 * 4};
  bool ok = true;

  // ---- M2 ---------------------------------------------------------------------------
  for (const char* p : prompts) {
    e.reset();
    e.prefill(golden::read_ids(pdir + "/" + p + ".ids"));
    const Snap s0(c);
    const Ref ref = reference(c, s0, runtime::Engine::kMaxDraft + 2);
    ok &= m2(c, s0, ref, p, golden::read_ids(pdir + "/" + p + ".ids"), loader::lm_head_form_name(lm_head));
  }
  std::printf("M2: %s\n", ok ? "PASS" : "FAIL");

  // ---- M3 and H -----------------------------------------------------------------------
  bool m3 = true, head_ok = true;
  for (size_t pi = 0; pi < 3; ++pi) {
    const char* p = prompts[pi];
    const std::vector<uint32_t> ids = golden::read_ids(pdir + "/" + p + ".ids");
    // The head-loaded engine's own plain steps (verify(0) + commit): the timing baseline, and
    // a second equality - they must be the head-less engine's ids too.
    e.reset();
    e.prefill(ids);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<uint32_t> own = e.generate(kGen);
    const double plain_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  M3 %s K=0 (head loaded): %s the head-less engine's ids\n", p,
                own == plain[pi] ? "==" : "!=");
    m3 &= own == plain[pi];
    for (uint32_t K = 1; K <= runtime::Engine::kMaxDraft; ++K) {
      e.reset();
      e.prefill(ids);
      std::vector<uint32_t> got;
      size_t drafted = 0, accepted = 0, iters = 0;
      const auto t1 = std::chrono::steady_clock::now();
      while (got.size() < kGen) {
        const std::vector<uint32_t> o = iterate(c, K, &drafted, &accepted);
        got.insert(got.end(), o.begin(), o.end());
        ++iters;
      }
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
      got.resize(kGen);
      const spec_judge::StreamVerdict sv = spec_judge::streams(
          golden::ForcedJudge{}, std::string("ornith_mtp_m3_") + loader::lm_head_form_name(lm_head) + "_" + p, ids,
          plain[pi], plain_steps[pi], got, uint32_t(V), uint32_t(VU));
      std::printf("  M3 %s K=%u: plain over %u ids: %s; %zu/%zu drafts accepted (%.3f), %zu "
                  "iterations, %.2f ms/id vs plain %.2f\n",
                  p, K, kGen, sv.line().c_str(), accepted, drafted, drafted ? double(accepted) / double(drafted) : 0.0,
                  iters, ms / kGen, plain_ms / kGen);
      m3 &= sv.pass;
      // H: the head's MoE layer wrote its route row (layer slot `layers`, row 0).
      const runtime::MoeScratchLayout ml = runtime::moe_scratch_layout(d);
      std::vector<uint32_t> row(runtime::kMoeRouteWords);
      c.imm.copy(row.data(), e.buffers().moe->as<uint8_t>() + ml.route_at(d.layers), row.size() * 4);
      float wsum = 0;
      for (uint32_t k = 0; k < d.moe.top_k; ++k) {
        float w;
        std::memcpy(&w, &row[8 + k], 4);
        wsum += w;
        head_ok &= row[k] < d.moe.experts;
        for (uint32_t j = 0; j < k; ++j) head_ok &= row[j] != row[k];
      }
      head_ok &= std::fabs(wsum - 1.0f) < 0.02f;
    }
  }
  std::printf("M3: %s\nH (the head's route row): %s\n", m3 ? "PASS" : "FAIL", head_ok ? "ok" : "BAD");
  ok &= m3 && head_ok;
  if (ok) std::puts("ornith_mtp_test OK");
  return spec_judge::exit_code(ok, "ornith_mtp_test");
}

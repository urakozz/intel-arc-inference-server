// Spec 15e's MTP gates on Ornith 1.5 35B-A3B, on the card (needs the Ornith int4
// checkpoint; exit 77 - SKIP - without it, as 15c / 15d's tests do):
//
//   L   the lists: the head is the MoE head (its bf16 experts int4 g64 RTN at load, 771
//       linears from the published head), its device bytes are loader::mtp_head_bytes; the
//       verify lists are verify_launches() = 536 at M = 1..4, the drafts draft_launches() =
//       24 (the head's FFN is the 4-launch MoE block);
//   M2  (spec 15e Review Focus 3; spec 8 A6 on Ornith) from a prefilled state S0 and the
//       true greedy drafts g[1..k], verify(k) for k = 1..3: every row r BITWISE equal to the
//       r-th plain step from S0 (verify(0) + commit) - its logits row, its GDN state slot,
//       and every one of the 40 layers' route rows (expert ids, weights, shared gate) - the
//       MoE block at M rows routes each row as M = 1 does;
//   M3  greedy MTP is lossless (`--mtp K` == `--mtp 0`): K = 1, 2, 3 iterations (draft,
//       verify, accept the agreeing prefix, commit) emit exactly the greedy ids of an engine
//       loaded WITHOUT the head (the plain decode list), 128 of them, on prose / code /
//       cjk, after an l0-int8 prefill (which fills the head's KV through step_mtp_kv);
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
constexpr size_t V = Qwen35::kVocab;

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

bool m2(Ctx& c, const Snap& s0, const Ref& ref, const char* what) {
  bool ok = true;
  for (uint32_t k = 1; k <= runtime::Engine::kMaxDraft; ++k) {
    s0.restore(c);
    for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
    c.e.verify(k);
    const std::vector<uint8_t> lg = read(c, c.e.verify_logits_device(), size_t(k + 1) * V * 4);
    for (uint32_t r = 0; r <= k; ++r) {
      const bool lbits = std::memcmp(lg.data() + size_t(r) * V * 4, ref.logits[r].data(), V * 4) == 0;
      const bool sbits =
          gdn_slot(c, (c.ctl->gdn_live + r) % runtime::MtpBuffers::kSlots) == ref.state[r];
      const bool rbits = routes(c, r) == ref.route[r];
      const bool id_ok = c.e.verify_ids()[r] == ref.g[r + 1];
      std::printf("  M2 %s k=%u row %u: logits %s, GDN slot %s, 40 layers' routes %s, id %s\n", what,
                  k, r, lbits ? "bitwise" : "DIFFER", sbits ? "bitwise" : "DIFFERS",
                  rbits ? "bitwise" : "DIFFER", id_ok ? "ok" : "DIFFERS");
      ok &= lbits && sbits && rbits && id_ok;
    }
    c.e.commit(k, c.e.verify_ids()[k]);
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
  {
    runtime::Engine p0(ctx, loader::load(ctx, snap, kMaxLen, /*mtp=*/false, lm_head), kMaxLen);
    for (const char* p : prompts) {
      p0.reset();
      p0.prefill(golden::read_ids(pdir + "/" + p + ".ids"));
      plain.push_back(p0.generate(kGen));
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
    const Ref ref = reference(c, s0, runtime::Engine::kMaxDraft + 1);
    ok &= m2(c, s0, ref, p);
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
      const bool same = got == plain[pi];
      std::printf("  M3 %s K=%u: %s plain over %u ids; %zu/%zu drafts accepted (%.3f), %zu "
                  "iterations, %.2f ms/id vs plain %.2f\n",
                  p, K, same ? "==" : "!=", kGen, accepted, drafted,
                  drafted ? double(accepted) / double(drafted) : 0.0, iters, ms / kGen,
                  plain_ms / kGen);
      m3 &= same;
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
  if (!ok) return 1;
  std::puts("ornith_mtp_test OK");
  return 0;
}

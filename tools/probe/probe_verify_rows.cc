// probe_verify_rows - spec 28 (dflash_gpu_test t5_test-box rows 137 / 143: the DFlash verify row picks another id
// than the plain row of the same context): WHERE does an M = 8 verify row depart from the M = 1 plain row - the
// history the verify lists wrote (KV rows, GDN states of ~220 earlier positions), or the row's own arithmetic at
// M = 8 (gemm_sm linears, the verify attention, the multi-row GDN step)?
//
//   probe_verify_rows <snapshot> <prompt ids> <stream ids> <G,G,...> [--lm-head bf16|int8] [--watch A,B]
//                     [--out STEM] [--max-len 16384]
//
// <stream ids>: the generated stream (index 0 = the prefill's id, as the tests' streams), e.g. a DFlash run's. For
// each G - the generated id that row decides - two states at pos = prompt + G - 1 with x = stream[G - 1] pending:
//   plain   reset, prefill(prompt), ingest(stream[0, G - 1)) one id at a time: the plain (M = 1) history
//   dflash  the production DFlash loop (b70-decode --spec dflash's, K = 7, greedy; the lists at the rule) from the
//           prompt until an iteration starts at that pos - its stream checked against <stream ids> - the history the
//           verify lists wrote (the probe stops with a message when no iteration starts there)
// and on each state the row deciding G is computed by every arm, on the SAME state (Control and the live GDN slot
// restored between arms; the arm `rule` runs first and again last, and the two must be bitwise):
//   m1          verify(0): the M = 1 list (gemv linears, decode attention) - the plain row
//   rule        verify(7) at the production rule (M = 8: gemm_sm, attn_verify_dpas from its threshold)
//   v2          verify(7), attention forced to v2, linears gemm_sm
//   v2_gemv     verify(7), attention v2, every linear forced to gemv
//   dpas_gemv   verify(7), attention DPAS, every linear gemv
// The drafts of a verify(7) arm are stream[G .. G + 6] (row 0 reads none of them: causal). Printed per arm: row 0's
// argmax, the gap A - B in bf16 ulps of the row's top (--watch; golden::bf16_ulp_at), max |logit - m1's|, and the
// cosine of each drafter tap (the residual after layers 5 / 19 / 33 / 47 / 61, TapDrafter::taps() row 0) against m1's.
// --out STEM writes STEM.<G>.<state>.<arm>.logits.f32 ([V]) and .taps.bf16 ([5][5120]) for a comparison against
// the f32 references (tools/oracle/row_taps_ref.py: `out` [64][H] at the same row).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cli/dflash_args.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/gemv_kind.h"

namespace {

constexpr uint32_t kV = model::Qwen35::kVocab, kK = 7;

std::vector<uint32_t> read_ids(const std::string& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot read " + p);
  std::vector<uint32_t> v;
  for (uint32_t x; f >> x;) v.push_back(x);
  return v;
}

void write(const std::string& path, const void* p, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(p, 1, n, f) != n) throw std::runtime_error("cannot write " + path);
  std::fclose(f);
}

double ulp_at(float x) {   // golden::bf16_ulp_at
  int e = 0;
  std::frexp(std::fabs(x), &e);
  return std::ldexp(1.0, e - 8);
}

struct Arm {
  const char* name;
  bool m1;
  runtime::CaptureOptions opt;
};

struct Row {   // one arm's row 0
  std::vector<float> logits;
  std::vector<uint16_t> taps;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: probe_verify_rows <snapshot> <prompt ids> <stream ids> <G,...> [--lm-head bf16|int8] "
                 "[--watch A,B] [--out STEM] [--max-len L]\n");
    return 2;
  }
  const std::vector<uint32_t> prompt = read_ids(argv[2]);
  const std::vector<uint32_t> stream = read_ids(argv[3]);
  std::vector<uint32_t> gens;
  {
    std::stringstream s(argv[4]);
    for (std::string t; std::getline(s, t, ',');)
      if (!t.empty()) gens.push_back(uint32_t(std::stoul(t)));
  }
  std::sort(gens.begin(), gens.end());
  loader::LmHeadForm head = loader::LmHeadForm::Checkpoint;
  uint32_t max_len = 16384, wa = 0, wb = 0;
  bool watch = false;
  std::string out;
  for (int i = 5; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--lm-head") {
      if (!loader::parse_lm_head_form(argv[i + 1], head)) return 2;
    } else if (a == "--watch") {
      if (std::sscanf(argv[i + 1], "%u,%u", &wa, &wb) != 2) return 2;
      watch = true;
    } else if (a == "--out") {
      out = argv[i + 1];
    } else if (a == "--max-len") {
      max_len = uint32_t(std::stoul(argv[i + 1]));
    } else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  for (uint32_t g : gens)
    if (g == 0 || g + kK > stream.size()) throw std::runtime_error("G " + std::to_string(g) + " is outside the stream");

  l0::Context ctx(0u);
  loader::LoadedModel model = loader::load(ctx, argv[1], max_len, /*mtp=*/false, head);
  cli::dflash::Args da;
  da.on = true;
  cli::dflash::Loaded dl = cli::dflash::load(ctx, da, model);
  auto drafter = std::make_unique<runtime::dflash::DFlashDrafter>(ctx, std::move(dl.w), dl.head_rows, max_len, da.max_k);
  runtime::Engine eng(ctx, std::move(model), max_len, /*debug_resid=*/false, runtime::KvCache::Bf16, drafter.get());
  drafter->attach(eng, /*draft_vocab=*/false);
  eng.prepare_prefill();
  runtime::TapDrafter& d = *eng.tap_drafter();
  const uint32_t P = uint32_t(prompt.size()), H = 5120;
  const size_t n_taps = d.tap_layers().size(), tap_row = n_taps * H;
  std::printf("probe_verify_rows on %s: lm_head %s, prompt %u ids, stream %zu ids, taps after layers", ctx.name().c_str(),
              loader::lm_head_form_name(head), P, stream.size());
  for (uint32_t l : d.tap_layers()) std::printf(" %u", l);
  std::printf("\n");

  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const size_t gdn_bytes = eng.buffers().gdn_state.size();
  l0::Mem gdn_keep(ctx, l0::MemKind::Device, gdn_bytes);
  using runtime::CaptureOptions;
  using runtime::LinearForce;
  using runtime::VerifyAttnForce;
  auto opts = [](VerifyAttnForce a, LinearForce l) {
    CaptureOptions o;
    o.force_attn = a;
    o.force_linear = l;
    return o;
  };
  const std::vector<Arm> arms = {
      {"m1", true, {}},
      {"rule", false, {}},
      {"v2", false, opts(VerifyAttnForce::V2, LinearForce::Auto)},
      {"v2_gemv", false, opts(VerifyAttnForce::V2, LinearForce::Gemv)},
      {"dpas_gemv", false, opts(VerifyAttnForce::Dpas, LinearForce::Gemv)},
      {"rule", false, {}},
  };

  // Every arm on the state at pos with x pending, restored between arms; returns each arm's row 0.
  auto run_arms = [&](uint32_t g, const char* state) {
    runtime::Control* ctl = eng.buffers().control.as<runtime::Control>();
    const runtime::Control keep = *ctl;
    const uint32_t live = ctl->gdn_live;
    void* slot = live == 0 ? eng.buffers().gdn_state.ptr()
                           : static_cast<uint8_t*>(eng.drafter_slots()->ptr()) + size_t(live - 1) * gdn_bytes;
    imm.copy(gdn_keep.ptr(), slot, gdn_bytes);
    std::vector<Row> rows;
    for (const Arm& a : arms) {
      *ctl = keep;
      imm.copy(slot, gdn_keep.ptr(), gdn_bytes);
      ctl->cur_token[0] = stream[g - 1];
      const uint32_t k = a.m1 ? 0 : kK;
      for (uint32_t r = 1; r <= k; ++r) ctl->cur_token[r] = stream[g - 1 + r];
      eng.recapture_verify(a.opt);
      eng.verify(k);
      Row row;
      row.logits.resize(kV);
      row.taps.resize(tap_row);
      imm.copy(row.logits.data(), eng.verify_logits_device(), size_t(kV) * 4);
      imm.copy(row.taps.data(), d.taps().ptr(), tap_row * 2);
      rows.push_back(std::move(row));
    }
    // back to the production lists and the kept state (the caller commits or resets next)
    *ctl = keep;
    imm.copy(slot, gdn_keep.ptr(), gdn_bytes);
    eng.recapture_verify({});
    const Row& m1 = rows[0];
    for (size_t i = 0; i < arms.size(); ++i) {
      const Row& r = rows[i];
      uint32_t top = 0;
      for (uint32_t t = 1; t < model::Qwen35::kVocabUsed; ++t)
        if (r.logits[t] > r.logits[top]) top = t;
      double mx = 0;
      for (uint32_t t = 0; t < kV; ++t) mx = std::max(mx, double(std::fabs(r.logits[t] - m1.logits[t])));
      std::printf("  G %u %-6s %-9s top %6u", g, state, arms[i].name, top);
      if (watch)
        std::printf("  %u - %u %+7.2f ulp", wa, wb, double(r.logits[wa] - r.logits[wb]) / ulp_at(r.logits[top]));
      std::printf("  max|d logit| vs m1 %.3f  tap cos vs m1:", mx);
      for (size_t t = 0; t < n_taps; ++t) {
        double xy = 0, xx = 0, yy = 0;
        for (uint32_t h = 0; h < H; ++h) {
          const double x = common::bf16_to_f32(r.taps[t * H + h]), y = common::bf16_to_f32(m1.taps[t * H + h]);
          xy += x * y, xx += x * x, yy += y * y;
        }
        std::printf(" %.7f", xy / std::sqrt(xx * yy));
      }
      std::printf("\n");
      if (!out.empty() && i + 1 < arms.size()) {
        const std::string stem = out + "." + std::to_string(g) + "." + state + "." + arms[i].name;
        write(stem + ".logits.f32", r.logits.data(), r.logits.size() * 4);
        write(stem + ".taps.bf16", r.taps.data(), r.taps.size() * 2);
      }
    }
    const bool same = rows[1].logits == rows.back().logits && rows[1].taps == rows.back().taps;
    std::printf("  G %u %s: the rule arm first and last %s\n", g, state, same ? "bitwise (the restore holds)" : "DIFFER");
    std::fflush(stdout);
    if (!same) throw std::runtime_error("the state was not restored between arms");
  };

  // --- the plain history
  for (uint32_t g : gens) {
    eng.reset();
    eng.prefill(prompt, 0);
    eng.ingest(std::vector<uint32_t>(stream.begin(), stream.begin() + (g - 1)));
    if (eng.pos() != P + g - 1) throw std::runtime_error("plain: pos " + std::to_string(eng.pos()));
    run_arms(g, "plain");
  }

  // --- the DFlash history: the production loop until an iteration starts at pos P + G - 1
  eng.reset();
  eng.prefill(prompt, 0);
  std::vector<uint32_t> got;
  size_t next = 0;
  while (next < gens.size() && got.size() + 1 < stream.size()) {
    const uint32_t x = eng.buffers().control.as<runtime::Control>()->cur_token[0];
    if (got.size() + 1 == gens[next]) {
      run_arms(gens[next], "dflash");
      ++next;
      continue;
    }
    if (got.size() + 1 > gens[next]) {
      std::printf("  G %u: no DFlash iteration starts at pos %u (the run passed it)\n", gens[next], P + gens[next] - 1);
      ++next;
      continue;
    }
    const uint32_t k = std::min(da.max_k, eng.max_verify_k());
    const runtime::dflash::DFlashDraftOut& o = d.draft(eng, k, 0.0f, 0);
    const std::vector<uint32_t> drafts(o.ids, o.ids + k);
    eng.verify(k);
    uint32_t j = 0;
    while (j < k && drafts[j] == eng.verify_ids()[j]) ++j;
    got.push_back(x);
    for (uint32_t i = 0; i < j; ++i) got.push_back(drafts[i]);
    eng.commit(j, eng.verify_ids()[j]);
    for (size_t i = 0; i < got.size() && i < stream.size(); ++i)
      if (got[i] != stream[i])
        throw std::runtime_error("the DFlash run left <stream ids> at generated id " + std::to_string(i));
  }
  std::printf("probe_verify_rows: done\n");
  return 0;
}

// Spec 8 §3.2 (plan 8b Task 3, Review Focus 2): prefill fills the MTP head's KV.
//
// The head's K/V at position p is a function of (h_p, x[p+1], p) only. The GATE isolates
// what this task wrote - the chunked head fill - from the main model: the prefill's own
// post-final-norm rows h_p are read back chunk by chunk (Engine::mtp_prefill_hidden), and
// the reference fills every row p with the M = 1 decode head on exactly those inputs
// (draft(1) at pos p + 1 with hh row 0 = h_p and pending x[p+1]; the draft list writes
// head-KV row p). Every row 0..n-2 must match at cosine >= 0.9999, K and V separately.
// Chunks of 2048 (a block edge) and of 1000 (edges that are not), 4096 ids.
//
// RECORDED, not gated: the same rows against a run that never prefilled (4096 plain M = 1
// steps, main model at M = 1 too). That comparison includes the main model's own
// prefill-vs-decode difference, which is not this plan's: on 2026-09-28 it measured median
// 0.99971 with 15 rows below 0.9 - rows where the main hidden itself differs (h cosine
// 0.27 at row 2091). Plan 8b's text asked for the per-row 0.9999 bar against that run;
// the spec 8 amendment records why the gate moved to the isolating reference above.
//
// Review Focus 2: row n-1 has no next id during prefill. After the prefill, one MTP
// iteration: draft(3) fills row n-1 at M = 1 (the reference fill) and verify(3)'s head
// rows rewrite it at M = 4 from the same inputs; the two must agree at >= 0.9999.
//
// usage: mtp_prefill_test <snapshot>   (cwd = the source tree)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
constexpr size_t kRow = 4 * 256;   // one position of the head's K (or V), bf16
const size_t H = model::qwen38().hidden;   // spec 15b: the descriptor's (Agnes's too)

double cos_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}

struct HeadKv {
  std::vector<uint16_t> k, v;
};
HeadKv read_head(l0::CmdList& imm, runtime::Engine& e, uint32_t n_pos) {
  runtime::MtpBuffers& mb = *e.mtp_buffers();
  HeadKv r{std::vector<uint16_t>(n_pos * kRow), std::vector<uint16_t>(n_pos * kRow)};
  imm.copy(r.k.data(), mb.kv_k.ptr(), r.k.size() * 2);
  imm.copy(r.v.data(), mb.kv_v.ptr(), r.v.size() * 2);
  return r;
}

struct Cmp {
  double worst = 1, median = 1;
  uint32_t worst_row = 0, below = 0, below9 = 0;
};
Cmp compare(const HeadKv& a, const HeadKv& b, uint32_t rows, double bar) {
  std::vector<std::pair<double, uint32_t>> w;
  Cmp c;
  for (uint32_t p = 0; p < rows; ++p) {
    const double x = std::min(cos_bf16(&a.k[p * kRow], &b.k[p * kRow], kRow),
                              cos_bf16(&a.v[p * kRow], &b.v[p * kRow], kRow));
    w.push_back({x, p});
    c.below += x < bar;
    c.below9 += x < 0.9;
  }
  std::sort(w.begin(), w.end());
  c.worst = w.front().first;
  c.worst_row = w.front().second;
  c.median = w[w.size() / 2].first;
  return c;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  std::vector<uint32_t> ids = golden::read_ids("tests/golden/prompts/long32k.ids");
  const uint32_t n = 4096;
  CHECK(ids.size() >= n);
  ids.resize(n);
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, snap, 16384, /*mtp=*/true), 16384);
  runtime::Control* ctl = e.buffers().control.as<runtime::Control>();
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // Recorded: the never-prefilled M = 1 run.
  e.reset();
  e.ingest(ids);
  CHECK_EQ(e.pos(), n);
  const HeadKv plain = read_head(imm, e, n);

  bool ok = true;
  for (uint32_t chunk : {2048u, 1000u}) {
    // The candidate: chunked prefill, one Engine::prefill call per chunk (prefill is
    // incremental; the head's row 0 of a chunk is the previous call's last hidden, the
    // same path as one call's internal chunks), reading every chunk's hidden rows back.
    e.reset();
    std::vector<uint16_t> h(size_t(n) * H);
    for (uint32_t off = 0; off < n; off += chunk) {
      const uint32_t C = std::min(chunk, n - off);
      e.prefill(std::vector<uint32_t>(ids.begin() + off, ids.begin() + off + C));
      imm.copy(&h[size_t(off) * H], e.mtp_prefill_hidden()->as<uint16_t>() + H,
               size_t(C) * H * 2);
    }
    CHECK_EQ(e.pos(), n);
    const uint32_t pending = ctl->cur_token[0];
    const HeadKv got = read_head(imm, e, n);

    // Review Focus 2: row n-1 by the first draft (M = 1), then by verify(3) (M = 4).
    e.draft(3);
    const HeadKv d1 = read_head(imm, e, n);
    e.verify(3);
    const HeadKv v4 = read_head(imm, e, n);
    const double rk = cos_bf16(&d1.k[(n - 1) * kRow], &v4.k[(n - 1) * kRow], kRow);
    const double rv = cos_bf16(&d1.v[(n - 1) * kRow], &v4.v[(n - 1) * kRow], kRow);
    uint32_t j = 0;
    while (j < 3 && e.draft_ids()[j] == e.verify_ids()[j]) ++j;
    e.commit(j, e.verify_ids()[j]);

    // The isolating reference: the M = 1 decode head on the prefill's own inputs.
    for (uint32_t p = 0; p + 1 < n; ++p) {
      ctl->pos = p + 1;
      ctl->cur_token[0] = ids[p + 1];
      imm.copy(e.mtp_buffers()->hh.ptr(), &h[size_t(p) * H], H * 2);
      e.draft(1);
    }
    const HeadKv ref = read_head(imm, e, n);
    const Cmp g = compare(ref, got, n - 1, 0.9999);
    const Cmp r = compare(plain, got, n - 1, 0.9999);
    std::printf("chunk %4u: head KV rows 0..%u vs the M = 1 head on the same inputs: worst %.7f"
                " (row %u), median %.7f, below 0.9999: %u  -> %s\n",
                chunk, n - 2, g.worst, g.worst_row, g.median, g.below,
                g.below == 0 ? "PASS" : "FAIL");
    std::printf("            (recorded) vs the never-prefilled M = 1 run: worst %.4f (row %u),"
                " median %.6f, below 0.9999 %u, below 0.9 %u\n",
                r.worst, r.worst_row, r.median, r.below, r.below9);
    std::printf("            RF2: row %u, first draft (M = 1) vs verify(3) (M = 4): K %.7f V %.7f"
                "  -> %s   (pending %u, j = %u)\n",
                n - 1, rk, rv, rk >= 0.9999 && rv >= 0.9999 ? "PASS" : "FAIL", pending, j);
    ok &= g.below == 0 && rk >= 0.9999 && rv >= 0.9999;
  }
  CHECK(ok);
  std::puts("mtp_prefill_test OK");
  return 0;
}

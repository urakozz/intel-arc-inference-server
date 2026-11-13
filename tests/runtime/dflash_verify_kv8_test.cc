// dflash_verify_kv8_test - the DFlash verify at M = 2..8 on the int8 KV cache against the plain M = 1 path
// (2026-11-13: dflash_gpu_kv8_test t1_define-rotation K = 7 picked id 248068 at generated id 200 where the plain
// int8 row and the int8-simulating f32 judge pick 248046 by 34.6 bf16 ulps; docs/probes/probe-dflash-kv8-verify-m8-2026-11-13.md).
//
// The engine (int8 head, int8 drafter, --kv-cache int8) runs the plain greedy loop (verify(0) + commit) and the
// DFlash greedy loop (draft(K) -> verify(K) -> the longest agreeing prefix -> commit) from the same prefill, with the
// verify lists recaptured per arm (Engine::recapture_verify, test only):
//   H  (bitwise) attention v2 and every linear gemv: each verify row's arithmetic is the M = 1 list's, so for every
//      fixed K = 1..7 the stream equals the plain one and the KV cache rows the verify lists wrote at the generated
//      positions - the int8 codes and the fp16 scales of every FA layer - equal the plain run's bit for bit. This is
//      the M = 2..8 int8 writer (attn_prep_M<M>_KV8), v2's int8 pair at M rows, the multi-row GDN step and the
//      commit; a difference prints the K and the first differing byte of the save_kv() image.
//   A  (recorded) the production rule, v2 with gemm_sm, DPAS with gemv, DPAS with gemm_sm at K = 3..7: the first
//      difference against the plain stream and the plain row's margin there - the two approximations spec 28 / 28c
//      rule on (judged by the f32 judge in dflash_gpu_kv8_test, not here).
// Prompts: tests/golden/toolcall/t1_define-rotation.ids (960 ids, 256 generated) and tests/golden/prompts/long32k.ids
// cut to 32000 (64 generated).
//
// usage: dflash_verify_kv8_test <target snapshot> <drafter>   (cwd = the source tree). SKIP 77 without either.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "../server/greedy_tie.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/engine.h"
#include "runtime/gemv_kind.h"

namespace {
using model::Qwen35;
constexpr uint32_t kMaxLen = 40960, kK = 7;

struct Run {
  std::vector<uint32_t> ids;                 // ids[0] = the prefill's id
  std::vector<greedy_tie::RefStep> steps;    // plain only: steps[i] decided ids[i + 1]
  std::vector<uint8_t> kv;                   // KV rows [P, P + n)
};

Run plain(runtime::Engine& e, const std::vector<uint32_t>& prompt, uint32_t n) {
  e.reset();
  e.prefill(prompt);
  runtime::Control* ctl = e.buffers().control.as<runtime::Control>();
  l0::CmdList imm = l0::CmdList::immediate(e.context());
  std::vector<float> row(Qwen35::kVocab);
  Run r;
  for (uint32_t i = 0; i <= n; ++i) {
    r.ids.push_back(ctl->cur_token[0]);
    e.verify(0);
    imm.copy(row.data(), e.verify_logits_device(), row.size() * 4);
    r.steps.push_back(greedy_tie::ref_step(row.data(), uint32_t(Qwen35::kVocabUsed)));
    e.commit(0, e.verify_ids()[0]);
  }
  const uint32_t P = uint32_t(prompt.size());
  r.kv.resize(e.kv_bytes(n));
  e.save_kv(P, P + n, r.kv.data());
  return r;
}

Run dflash(runtime::Engine& e, runtime::dflash::DFlashDrafter& d, const std::vector<uint32_t>& prompt, uint32_t n,
           uint32_t K) {
  e.reset();
  e.prefill(prompt);
  runtime::Control* ctl = e.buffers().control.as<runtime::Control>();
  Run r;
  while (r.ids.size() <= n) {
    const uint32_t x = ctl->cur_token[0];
    const uint32_t k = std::min(K, e.max_verify_k());
    const runtime::dflash::DFlashDraftOut& o = d.draft(e, k, 0.0f, 0);
    const std::vector<uint32_t> drafts(o.ids, o.ids + k);
    e.verify(k);
    uint32_t j = 0;
    while (j < k && drafts[j] == e.verify_ids()[j]) ++j;
    r.ids.push_back(x);
    r.ids.insert(r.ids.end(), drafts.begin(), drafts.begin() + j);
    e.commit(j, e.verify_ids()[j]);
  }
  const uint32_t P = uint32_t(prompt.size());
  r.kv.resize(e.kv_bytes(n));
  e.save_kv(P, P + n, r.kv.data());
  return r;
}

size_t first_diff(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return i;
  return n;
}

// The first differing byte of two save_kv() images of n positions, named (KvLayout: per layer K int8 [n][4][256]
// ... then the scales; save_kv's order is the engine's, so only the byte offset and the row it falls in are printed).
std::string kv_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  size_t i = 0, bad = 0;
  while (i < a.size() && a[i] == b[i]) ++i;
  if (i == a.size()) return "bitwise";
  for (size_t t = i; t < a.size(); ++t) bad += a[t] != b[t];
  return "first differing byte " + std::to_string(i) + " of " + std::to_string(a.size()) + " (" + std::to_string(bad) +
         " bytes differ)";
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap, dsnap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
    dsnap = loader::resolve_snapshot(argc > 2 ? argv[2] : "z-lab/Qwen3.8-27B-DFlash2");
  } catch (const std::exception& ex) {
    std::printf("SKIP: %s\n", ex.what());
    return 77;
  }
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, loader::LmHeadForm::Int8);
  loader::DFlashWeights w = loader::load_drafter(ctx, dsnap, model, loader::DraftFormat::Int8);
  const uint32_t head_rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  runtime::dflash::DFlashDrafter d(ctx, std::move(w), head_rows, kMaxLen, kK);
  runtime::Engine e(ctx, std::move(model), kMaxLen, /*debug_resid=*/false, runtime::KvCache::Int8, &d);
  d.attach(e, /*draft_vocab=*/false);
  std::printf("dflash_verify_kv8_test on %s: KV cache %s\n", ctx.name().c_str(), runtime::kv_cache_name(e.kv_cache()));

  auto opts = [](runtime::VerifyAttnForce a, runtime::LinearForce l) {
    runtime::CaptureOptions o;
    o.force_attn = a;
    o.force_linear = l;
    return o;
  };
  struct Prompt {
    std::string name;
    std::vector<uint32_t> ids;
    uint32_t n;
  };
  std::vector<uint32_t> deep = golden::read_ids("tests/golden/prompts/long32k.ids");
  CHECK(deep.size() >= 32000);
  deep.resize(32000);
  const std::vector<Prompt> prompts = {
      {"t1_define-rotation", golden::read_ids("tests/golden/toolcall/t1_define-rotation.ids"), 256},
      {"long32k[:32000]", deep, 64},
  };

  bool ok = true;
  for (const Prompt& p : prompts) {
    e.recapture_verify({});
    const Run ref = plain(e, p.ids, p.n);
    std::printf("%s: %zu prompt ids, %u generated\n", p.name.c_str(), p.ids.size(), p.n);
    // H: v2 + gemv, every K
    e.recapture_verify(opts(runtime::VerifyAttnForce::V2, runtime::LinearForce::Gemv));
    for (uint32_t K = 1; K <= kK; ++K) {
      const Run r = dflash(e, d, p.ids, p.n, K);
      const size_t fd = first_diff(r.ids, ref.ids, p.n);
      const std::string kd = kv_diff(r.kv, ref.kv);
      const bool pass = fd == p.n && kd == "bitwise";
      ok &= pass;
      std::printf("  H %s v2+gemv K=%u: ids %s, KV rows [P, P + %u) %s - %s\n", p.name.c_str(), K,
                  fd == p.n ? "identical" : ("first difference at " + std::to_string(fd)).c_str(), p.n, kd.c_str(),
                  pass ? "PASS" : "FAIL");
    }
    // A: the approximating arms at K = 3..7 (recorded)
    const struct {
      const char* name;
      runtime::CaptureOptions o;
    } arms[] = {{"rule", {}},
                {"v2+gemm_sm", opts(runtime::VerifyAttnForce::V2, runtime::LinearForce::Auto)},
                {"dpas+gemv", opts(runtime::VerifyAttnForce::Dpas, runtime::LinearForce::Gemv)},
                {"dpas+gemm_sm", opts(runtime::VerifyAttnForce::Dpas, runtime::LinearForce::Auto)}};
    for (const auto& a : arms) {
      e.recapture_verify(a.o);
      for (uint32_t K = 3; K <= kK; ++K) {
        const Run r = dflash(e, d, p.ids, p.n, K);
        const size_t fd = first_diff(r.ids, ref.ids, p.n);
        if (fd == p.n) {
          std::printf("  A %s %s K=%u: identical\n", p.name.c_str(), a.name, K);
        } else {
          uint32_t rank = 0;
          double gap = 0;
          greedy_tie::rank_of(ref.steps[fd - 1], r.ids[fd], &rank, &gap);
          std::printf("  A %s %s K=%u: first difference at %zu (%u vs plain %u: rank %u, %.2f bf16 ulp below the "
                      "plain top)\n",
                      p.name.c_str(), a.name, K, fd, r.ids[fd], ref.ids[fd], rank, gap);
        }
      }
    }
  }
  e.recapture_verify({});
  std::printf("dflash_verify_kv8_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

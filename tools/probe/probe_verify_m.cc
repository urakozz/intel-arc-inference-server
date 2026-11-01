// probe_verify_m: spec 8 P0 (plan 8a Task 3) - the main model's verify step at
// M = 1..4 rows, the draft step's pieces, the GDN slot copy and the host round trip.
//
//   probe_verify_m <snapshot> [--depth 4096] [--replays 64] [--rounds 3] [--device 0]
//
// Needs a build with -DB70_DECODE_EXTRA_M=ON (the M = 2..4 decode-list variants
// and the bf16 head-shape GEMVs). Everything is timed on the REAL captured list
// (runtime::build at M), not kernels in isolation, after one prefill of --depth
// ids: every layer, lm_head and argmax at M rows.
//
// 1. Verify step: for each round, the four M arms in a rotated order (round r
//    starts at M = 1 + r % 4); each arm rewinds Control::pos to the prefill's
//    end, sets n_active = M, warms up 8 replays, then times --replays replays
//    (execute + fence each, what Engine::generate does). Reported: median over
//    rounds of ms/step, its range, ms/row, and the median paired ratio to the
//    M = 1 arm of the same round.
// 2. Draft pieces: a profiled M = 1 list (per-launch events) gives lm_head,
//    the two argmax stages, embed_gather and one full-attention layer's
//    non-GEMV launches at this depth; the head's five bf16 GEMV shapes
//    (fc 10240->5120, qkv 5120->14336, o 6144->5120, gate||up 5120->34816,
//    down 17408->5120) are timed in isolation (gemv_bf16, default tiling),
//    interleaved, median of 5 rounds of 50.
// 3. The GDN slot copy: 151 MB (48 x 48 x 128 x 128 fp32) device -> device on
//    an immediate list, median of 5 after a warm-up.
// 4. Host round trip: execute + fence of a one-noop list; a 4-byte and a 5 MB
//    device -> host readback on an immediate list; medians of 200 / 200 / 20.
//
// Spec 28b: the 1v rows also time each M >= 2 list forced onto today's gemv (runtime::CaptureOptions), interleaved,
// and the profile groups gemm_sm / gemm_sm_i8w with the GEMVs they replace.
// `--verify8 1` (spec 28a, plan 28a Task 3) replaces all of the above with the VERIFY lists at
// M = 1..8: the model loaded with its MTP head and the int8 lm_head, M = 1..4 over the engine's
// own 4-slot MtpBuffers and M = 5..8 over a probe-owned 8-slot one (probe_mtp_steps' B70_VERIFY_M8
// arrangement), every list driven as Engine::verify(k) + commit(0) drives its own, with pos
// rewound to the prefill's end before every call:
//   1v. the unprofiled verify step at M = 1..8, interleaved pairs against M = 1 (the pair's
//       order alternating by round), median of --rounds over --replays calls each;
//   1p. the profiled lists (runtime::build_verify with ProfileEvents): per-family ms at M = 1..8
//       and the grouping spec 28 §6 needs - int4 GEMV, the int8 head, attention, gdn_step,
//       everything else - with the non-GEMV share applied to 1v's unprofiled step (the events'
//       flush per launch inflates the profiled absolutes; shares, not times, carry over).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "timer.h"

namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

// A few hundred ids of real text (the bench prompt, repeated): decode reads every
// weight per token regardless of the ids, the attention reads `depth` positions.
constexpr uint32_t kPrompt[] = {
    760,   72103, 506, 37119, 557,   11012, 3213, 310,   6512, 279, 61789, 272, 1072,  2272,
    279,   197616, 2271, 13,   469,   68042, 29123, 7247, 383,  279, 1387,  12615, 1345, 279,
    49813, 78911, 1141, 20459, 13,    3113,  7840,  279,  2981, 1000, 381,  16850, 1495, 13};
constexpr size_t kPromptLen = sizeof(kPrompt) / sizeof(kPrompt[0]);

// ---- spec 28a: the verify lists at M = 1..8 (--verify8) ---------------------------------
int run_verify8(l0::Context& ctx, const std::string& path, uint32_t depth, uint32_t replays,
                uint32_t rounds) {
  uint32_t max_len = 16384;
  while (max_len < depth + 1024) max_len *= 2;
  runtime::Engine e(ctx, loader::load(ctx, path, max_len, /*mtp=*/true, loader::LmHeadForm::Int8), max_len);
  std::vector<uint32_t> ids(depth);
  for (uint32_t i = 0; i < depth; ++i) ids[i] = kPrompt[i % kPromptLen];
  {
    Timer t; t.start();
    e.prefill(ids);
    std::printf("verify8: int8 head, max_len %u, prefill %u ids: %.0f ms, pos %u\n", max_len, depth, t.ms(), e.pos());
  }
  runtime::Control* ctl = e.buffers().control.as<runtime::Control>();
  if (ctl->gdn_live != 0) {
    std::fprintf(stderr, "gdn_live %u after the prefill, not 0\n", ctl->gdn_live);
    return 1;
  }
  runtime::MtpBuffers* mtp4 = e.mtp_buffers();
  auto mtp8 = std::make_unique<runtime::MtpBuffers>(ctx, max_len, *e.model().desc, e.draft_vocab(), e.kv_cache(),
                                                    runtime::MtpDims::kMaxSlots);
  auto buf_for = [&](uint32_t M) -> runtime::MtpBuffers& { return M <= 4 ? *mtp4 : *mtp8; };
  std::map<uint32_t, runtime::CapturedStep> lists, lists_g;
  for (uint32_t M = 1; M <= 8; ++M)
    lists.emplace(M, runtime::build_verify(ctx, e.model(), e.buffers(), buf_for(M), M));
  // Spec 28b: the same lists forced onto today's gemv (runtime::CaptureOptions) - the pre-28b step, in the same
  // session, interleaved with the production (gemm_sm) one.
  const runtime::CaptureOptions force_gemv{runtime::LinearForce::Gemv};
  for (uint32_t M = 2; M <= 8; ++M)
    lists_g.emplace(M, runtime::build_verify(ctx, e.model(), e.buffers(), buf_for(M), M, nullptr, force_gemv));
  std::printf("verify lists M = 1..8: %zu launches each (M = 5..8 over a probe-owned %u-slot MtpBuffers, %.2f GB)\n",
              lists.at(1).kernel_count, mtp8->slots, mtp8->bytes() / 1e9);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const size_t hid = size_t(e.model().desc->hidden) * 2;
  const uint32_t pos0 = ctl->pos;
  // One verify(M - 1) + commit(0), pos rewound first: Engine::verify + commit on our list.
  auto call = [&](l0::CmdList& list, uint32_t M) {
    runtime::MtpBuffers& mb = buf_for(M);
    runtime::Control* h = mb.hctl.as<runtime::Control>();
    ctl->pos = pos0;
    for (uint32_t r = 1; r < M; ++r) ctl->cur_token[r] = kPrompt[r % kPromptLen];
    ctl->n_active = M;
    h->pos = pos0 - 1;
    h->n_active = M;
    for (uint32_t r = 0; r < M; ++r) h->cur_token[r] = ctl->cur_token[r];
    q.execute(list, &fence);
    fence.wait();
    imm.copy(mb.hh.ptr(), mb.hh.as<uint8_t>() + hid, hid);
    ctl->n_active = 1;
    ctl->cur_token[0] = ctl->out_token[0];
  };
  auto arm_on = [&](std::map<uint32_t, runtime::CapturedStep>& ls, uint32_t M, uint32_t n) {
    double s = 0;
    for (uint32_t i = 0; i < n; ++i) {
      Timer t; t.start();
      call(ls.at(M).list, M);
      s += t.ms();
    }
    return s / n;
  };
  auto arm = [&](uint32_t M, uint32_t n) { return arm_on(lists, M, n); };
  for (uint32_t M = 1; M <= 8; ++M) arm(M, 8);   // warm-up
  for (uint32_t M = 2; M <= 8; ++M) arm_on(lists_g, M, 8);
  std::map<uint32_t, std::vector<double>> ms, ratio, ms1, msg, ratiog;
  for (uint32_t r = 0; r < rounds; ++r)
    for (uint32_t M = 1; M <= 8; ++M) {
      // M = 1, M (production) and M forced onto gemv, the order rotating by round
      double t1 = 0, tm = 0, tg = 0;
      for (uint32_t j = 0; j < 3; ++j) {
        const uint32_t w = (j + r + M) % 3;
        if (w == 0) t1 = arm(1, replays);
        else if (w == 1) tm = arm(M, replays);
        else if (M >= 2) tg = arm_on(lists_g, M, replays);
      }
      ms[M].push_back(tm);
      ms1[M].push_back(t1);
      ratio[M].push_back(tm / t1);
      if (M >= 2) {
        msg[M].push_back(tg);
        ratiog[M].push_back(tg / t1);
      }
    }
  std::printf("\n1v. the verify step (unprofiled), interleaved with M = 1 (and, spec 28b, with the same list forced onto "
              "gemv), %u calls each, median of %u, depth %u:\n| M | ms/step | range | x M=1 (paired median) | ratio "
              "range | M=1 ms in the pairs | gemv-forced ms | gemv-forced x M=1 |\n|---|---|---|---|---|---|---|---|\n",
              replays, rounds, depth);
  std::map<uint32_t, double> step;
  for (uint32_t M = 1; M <= 8; ++M) {
    auto v = ms[M], rr = ratio[M];
    std::sort(v.begin(), v.end());
    std::sort(rr.begin(), rr.end());
    step[M] = median(ms[M]);
    std::printf("| %u | %.3f | %.3f-%.3f | %.4f | %.4f-%.4f | %.3f | %.3f | %.4f |\n", M, step[M], v.front(), v.back(),
                median(ratio[M]), rr.front(), rr.back(), median(ms1[M]), M >= 2 ? median(msg[M]) : step[M],
                M >= 2 ? median(ratiog[M]) : 1.0);
  }
  // 1p. the profiled lists, per family
  std::map<std::string, std::map<uint32_t, double>> fam;
  std::map<uint32_t, double> tot;
  for (uint32_t M = 1; M <= 8; ++M) {
    runtime::ProfileEvents prof(ctx);
    runtime::CapturedStep instr = runtime::build_verify(ctx, e.model(), e.buffers(), buf_for(M), M, &prof);
    std::vector<double> acc(instr.kernel_count, 0.0);
    const uint32_t n = 16;
    for (uint32_t i = 0; i < n + 4; ++i) {
      for (l0::Event& ev : prof.events) ev.reset();
      call(instr.list, M);
      if (i >= 4)
        for (size_t k = 0; k < instr.kernel_count; ++k) acc[k] += prof.events[k].duration_us();
    }
    for (size_t k = 0; k < instr.kernel_count; ++k) {
      const std::string& l = instr.labels[k];
      const size_t a = l.find(' '), z = l.find(' ', a + 1);
      std::string f = l.substr(a + 1, z - a - 1);
      // the MTP head's own linears (after the main model, label "--" or its layer) stay apart
      if (f.rfind("gemv", 0) == 0 && l.find("N248320") == std::string::npos && l.rfind("--", 0) == 0) f += " (head)";
      fam[f][M] += acc[k] / n / 1e3;
      tot[M] += acc[k] / n / 1e3;
    }
  }
  auto row = [](const std::string& name, std::map<uint32_t, double>& v) {
    std::printf("| %s |", name.c_str());
    for (uint32_t M = 1; M <= 8; ++M) std::printf(" %.3f |", v[M]);
    std::printf(" %.2f |\n", v[1] > 0 ? v[8] / v[1] : 0.0);
  };
  std::printf("\n1p. per kernel family, profiled verify lists (ms/step; events add a flush each - shares, not times):\n"
              "| family | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 | M=8 / M=1 |\n|---|---|---|---|---|---|---|---|---|---|\n");
  std::map<std::string, std::map<uint32_t, double>> grp;
  for (auto& [f, v] : fam) {
    row(f, v);
    const std::string g = f == "gemv" || f == "gemm_sm"   ? "int4 GEMV (gemv)"
                          : f == "gemv_i8w" || f == "gemm_sm_i8w" ? "int8 head (gemv_i8w)"
                          : f.rfind("attn", 0) == 0       ? "attention (attn*)"
                          : f.rfind("gdn_step", 0) == 0   ? "gdn_step"
                                                          : "everything else";
    for (uint32_t M = 1; M <= 8; ++M) grp[g][M] += v[M];
  }
  row("sum", tot);
  std::printf("\ngrouped (spec 28 §6), profiled ms/step:\n"
              "| group | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 | M=8 / M=1 |\n|---|---|---|---|---|---|---|---|---|---|\n");
  for (auto& [g, v] : grp) row(g, v);
  row("sum", tot);
  std::printf("\nthe non-GEMV share (all but the int4 GEMV and the int8 head) applied to 1v's unprofiled step:\n"
              "| M | profiled sum | non-GEMV profiled | share | unprofiled step ms | non-GEMV ms (derived) | GEMV ms (derived) |\n"
              "|---|---|---|---|---|---|---|\n");
  for (uint32_t M = 1; M <= 8; ++M) {
    const double gemv = grp["int4 GEMV (gemv)"][M] + grp["int8 head (gemv_i8w)"][M];
    const double share = tot[M] > 0 ? (tot[M] - gemv) / tot[M] : 0;
    std::printf("| %u | %.3f | %.3f | %.4f | %.3f | %.3f | %.3f |\n", M, tot[M], tot[M] - gemv, share, step[M],
                share * step[M], (1 - share) * step[M]);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: probe_verify_m <snapshot> [--depth 4096] [--replays 64] "
                         "[--rounds 3] [--device 0]\n");
    return 2;
  }
  const std::string path = argv[1];
  uint32_t depth = 4096, replays = 64, rounds = 3, device = 0, max_len = 16384, verify8 = 0;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const uint32_t v = uint32_t(std::stoul(argv[i + 1]));
    if (a == "--depth") depth = v;
    else if (a == "--replays") replays = v;
    else if (a == "--rounds") rounds = v;
    else if (a == "--device") device = v;
    else if (a == "--verify8") verify8 = v;
    else throw std::runtime_error("unknown flag " + a);
  }

  l0::Context ctx(device);
  std::printf("device: %s\n", ctx.name().c_str());
  if (verify8) return run_verify8(ctx, path, depth, replays, rounds);
  loader::LoadedModel model = loader::load(ctx, path, max_len);
  runtime::Engine eng(ctx, std::move(model), max_len);
  std::vector<uint32_t> ids(depth);
  for (uint32_t i = 0; i < depth; ++i) ids[i] = kPrompt[i % kPromptLen];
  eng.prepare_prefill();
  {
    Timer t; t.start();
    eng.prefill(ids);
    std::printf("prefill %u ids: %.0f ms, pos %u\n", depth, t.ms(), eng.pos());
  }

  runtime::DecodeBuffers& b = eng.buffers();
  runtime::Control* c = b.control.as<runtime::Control>();
  const uint32_t pos0 = c->pos;
  const uint32_t tok0 = c->cur_token[0];
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // ---- 1. the verify step at M = 1..4 -------------------------------------
  std::map<uint32_t, runtime::CapturedStep> lists;
  for (uint32_t M = 1; M <= 4; ++M) {
    lists.emplace(M, runtime::build(ctx, eng.model(), b, nullptr, nullptr, M));
    std::printf("captured M=%u: %zu kernels, %zu modules\n", M, lists.at(M).kernel_count,
                lists.at(M).modules.size());
  }
  auto arm = [&](uint32_t M, uint32_t n) {
    c->pos = pos0;
    c->n_active = M;
    for (uint32_t m = 0; m < M; ++m) c->cur_token[m] = m == 0 ? tok0 : kPrompt[m];
    std::vector<double> step;
    for (uint32_t i = 0; i < 8 + n; ++i) {
      if (c->pos + M > max_len) c->pos = pos0;
      Timer t; t.start();
      q.execute(lists.at(M).list, &fence);
      fence.wait();
      if (i >= 8) step.push_back(t.ms());
      c->n_active = M;   // argmax_stage2 leaves cur_token/pos; n_active is ours
    }
    double sum = 0;
    for (double s : step) sum += s;
    return sum / n;
  };
  std::map<uint32_t, std::vector<double>> ms, ratio;
  arm(1, 32);  // warm-up burst
  for (uint32_t r = 0; r < rounds; ++r) {
    std::map<uint32_t, double> this_round;
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t M = 1 + (r + j) % 4;
      this_round[M] = arm(M, replays);
    }
    for (auto& [M, v] : this_round) {
      ms[M].push_back(v);
      ratio[M].push_back(v / this_round[1]);
    }
  }
  std::printf("\n| M | ms/step (median of %u) | range | ms/row | x M=1 step (paired median) |\n"
              "|---|---|---|---|---|\n", rounds);
  for (uint32_t M = 1; M <= 4; ++M) {
    auto v = ms[M];
    std::sort(v.begin(), v.end());
    std::printf("| %u | %.3f | %.3f-%.3f | %.3f | %.4f |\n", M, median(ms[M]), v.front(),
                v.back(), median(ms[M]) / M, median(ratio[M]));
  }

  // ---- 1b. where the M rows cost: per-kernel-family time, profiled lists ----
  {
    std::map<std::string, std::map<uint32_t, double>> fam;
    std::map<uint32_t, double> tot;
    for (uint32_t M = 1; M <= 4; ++M) {
      runtime::ProfileEvents prof(ctx);
      runtime::CapturedStep instr = runtime::build(ctx, eng.model(), b, nullptr, &prof, M);
      std::vector<double> acc(instr.kernel_count, 0.0);
      const uint32_t n = 16;
      c->pos = pos0; c->n_active = M;
      for (uint32_t i = 0; i < n + 4; ++i) {
        for (l0::Event& e : prof.events) e.reset();
        q.execute(instr.list, &fence);
        fence.wait();
        c->n_active = M;
        if (i >= 4)
          for (size_t k = 0; k < instr.kernel_count; ++k) acc[k] += prof.events[k].duration_us();
      }
      for (size_t k = 0; k < instr.kernel_count; ++k) {
        const std::string& l = instr.labels[k];
        const size_t a = l.find(' '), z = l.find(' ', a + 1);
        std::string f = l.substr(a + 1, z - a - 1);
        if (f == "gemv_bf16") f += l.find("N248320") != std::string::npos ? " lm_head" : " a||b";
        fam[f][M] += acc[k] / n / 1e3;
        tot[M] += acc[k] / n / 1e3;
      }
    }
    std::printf("\nper kernel family, profiled lists (ms/step, events add a flush each):\n"
                "| family | M=1 | M=2 | M=3 | M=4 | M=4 / M=1 |\n|---|---|---|---|---|---|\n");
    for (auto& [f, v] : fam)
      std::printf("| %s | %.3f | %.3f | %.3f | %.3f | %.2f |\n", f.c_str(), v[1], v[2], v[3], v[4],
                  v[4] / v[1]);
    std::printf("| sum | %.3f | %.3f | %.3f | %.3f | %.2f |\n", tot[1], tot[2], tot[3], tot[4],
                tot[4] / tot[1]);
  }

  // ---- 2. draft pieces -------------------------------------------------------
  {
    runtime::ProfileEvents prof(ctx);
    runtime::CapturedStep instr = runtime::build(ctx, eng.model(), b, nullptr, &prof, 1);
    std::vector<double> acc(instr.kernel_count, 0.0);
    const uint32_t n = 32;
    c->pos = pos0; c->n_active = 1; c->cur_token[0] = tok0;
    for (uint32_t i = 0; i < n + 4; ++i) {
      for (l0::Event& e : prof.events) e.reset();
      q.execute(instr.list, &fence);
      fence.wait();
      if (i >= 4)
        for (size_t k = 0; k < instr.kernel_count; ++k) acc[k] += prof.events[k].duration_us();
    }
    double lm = 0, amax = 0, emb = 0, fa_small = 0, total = 0;
    for (size_t k = 0; k < instr.kernel_count; ++k) {
      const std::string& l = instr.labels[k];
      const double us = acc[k] / n;
      total += us;
      if (l.rfind("--", 0) == 0) {
        if (l.find("N248320") != std::string::npos) lm += us;
        else if (l.find("argmax") != std::string::npos) amax += us;
        else if (l.find("embed_gather") != std::string::npos) emb += us;
      }
      if (l.rfind("L3 ", 0) == 0 && l.find(" gemv") == std::string::npos) fa_small += us;
    }
    std::printf("\nprofiled M=1 list at depth %u (mean of %u, events add a flush each - shares, "
                "not the bench step): sum %.1f us\n", depth, n, total);
    std::printf("| piece | us |\n|---|---|\n| lm_head (bf16 GEMV) | %.1f |\n| argmax (2 stages) | "
                "%.1f |\n| embed_gather | %.1f |\n| FA layer 3 non-GEMV launches (norms, attn trio,"
                " silu) | %.1f |\n", lm, amax, emb, fa_small);
  }
  {
    struct Shape { const char* what; uint32_t K, N; };
    const Shape shapes[] = {{"fc", 10240, 5120},        {"q||k||v", 5120, 14336},
                            {"o_proj", 6144, 5120},     {"gate||up", 5120, 34816},
                            {"down", 17408, 5120}};
    struct Arm { Shape s; std::unique_ptr<l0::Module> mod; std::unique_ptr<l0::Kernel> k;
                 std::unique_ptr<l0::Mem> w, x, out; l0::CmdList list; };
    std::vector<Arm> arms;
    for (const Shape& s : shapes) {
      const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(s.N);
      Arm a{s, std::make_unique<l0::Module>(ctx, kernels::path(kernels::gemv_bf16_variant(1, s.K, s.N, t))),
            nullptr, std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(s.K) * s.N * 2),
            std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(s.K) * 2),
            std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(s.N) * 4),
            l0::CmdList::regular(ctx)};
      // Random finite bf16 (exponent kept in [-8, 7]): a constant fill compresses
      // on this device and read 800-1300 GB/s in a first run, above DRAM bandwidth.
      {
        std::vector<uint16_t> h(size_t(s.K) * s.N);
        uint32_t st = 0x9e3779b9u ^ s.K ^ (s.N << 7);
        for (uint16_t& e : h) {
          st ^= st << 13; st ^= st >> 17; st ^= st << 5;
          e = uint16_t((st & 0x807f) | ((0x77 + ((st >> 8) & 0xf)) << 7));
        }
        imm.copy(a.w->ptr(), h.data(), h.size() * 2);
        imm.copy(a.x->ptr(), h.data(), size_t(s.K) * 2);
      }
      a.k = std::make_unique<l0::Kernel>(*a.mod, "gemv_bf16");
      a.k->group_size(t.cols * t.ksplit);
      a.k->arg_ptr(0, a.w->ptr());
      a.k->arg_ptr(1, a.x->ptr());
      a.k->arg_ptr(2, a.out->ptr());
      for (int i = 0; i < 50; ++i) a.list.launch(*a.k, s.N / t.cols);
      a.list.close();
      arms.push_back(std::move(a));
    }
    std::vector<std::vector<double>> us(arms.size());
    for (int r = 0; r < 6; ++r)
      for (size_t i = 0; i < arms.size(); ++i) {
        Timer t; t.start();
        q.execute(arms[i].list, &fence);
        fence.wait();
        if (r > 0) us[i].push_back(t.ms() * 1e3 / 50);
      }
    double sum = 0;
    std::printf("\n| head bf16 GEMV (M=1, isolated) | K | N | MB | us | GB/s |\n|---|---|---|---|---|---|\n");
    for (size_t i = 0; i < arms.size(); ++i) {
      const double u = median(us[i]), mb = double(arms[i].s.K) * arms[i].s.N * 2 / 1e6;
      sum += u;
      std::printf("| %s | %u | %u | %.1f | %.1f | %.0f |\n", arms[i].s.what, arms[i].s.K,
                  arms[i].s.N, mb, u, mb / u * 1e3);
    }
    std::printf("| sum | | | | %.1f | |\n", sum);
  }

  // ---- 3. the GDN slot copy ------------------------------------------------
  {
    const size_t bytes = size_t(48) * 48 * 128 * 128 * 4;
    l0::Mem dst(ctx, l0::MemKind::Device, bytes);
    imm.copy(dst.ptr(), b.gdn_state.ptr(), bytes);   // warm-up
    std::vector<double> t5;
    for (int i = 0; i < 5; ++i) {
      Timer t; t.start();
      imm.copy(dst.ptr(), b.gdn_state.ptr(), bytes);
      t5.push_back(t.ms());
    }
    std::printf("\nGDN slot copy %.1f MB device->device (immediate list): median %.3f ms (%.0f GB/s "
                "read+write %.0f GB/s)\n", bytes / 1e6, median(t5), bytes / median(t5) / 1e6,
                2 * bytes / median(t5) / 1e6);
  }

  // ---- 4. host round trip ----------------------------------------------------
  {
    l0::Module mod(ctx, kernels::path("noop"));
    l0::Kernel k = mod.kernel("noop");
    l0::Mem out(ctx, l0::MemKind::Device, 4096);
    k.arg_ptr(0, out.ptr());
    k.group_size(16);
    l0::CmdList one = l0::CmdList::regular(ctx);
    one.launch(k, 1);
    one.close();
    std::vector<double> sub, rb4, rb5;
    for (int i = 0; i < 220; ++i) {
      Timer t; t.start();
      q.execute(one, &fence);
      fence.wait();
      if (i >= 20) sub.push_back(t.ms() * 1e3);
    }
    uint32_t v4 = 0;
    for (int i = 0; i < 220; ++i) {
      Timer t; t.start();
      imm.copy(&v4, b.logits.ptr(), 4);
      if (i >= 20) rb4.push_back(t.ms() * 1e3);
    }
    const size_t five = 5u << 20;
    std::vector<uint8_t> host(five);
    for (int i = 0; i < 25; ++i) {
      Timer t; t.start();
      imm.copy(host.data(), b.logits.ptr(), std::min(five, b.logits.size()));
      if (i >= 5) rb5.push_back(t.ms() * 1e3);
    }
    l0::Mem pinned(ctx, l0::MemKind::Host, five);
    std::vector<double> rb5p;
    for (int i = 0; i < 25; ++i) {
      Timer t; t.start();
      imm.copy(pinned.ptr(), b.logits.ptr(), std::min(five, b.logits.size()));
      if (i >= 5) rb5p.push_back(t.ms() * 1e3);
    }
    std::printf("\n| host round trip | us (median) |\n|---|---|\n| execute + fence, one-noop list | "
                "%.1f |\n| 4 B device->host readback (immediate) | %.1f |\n| %.1f MB logits "
                "device->host, pageable (immediate) | %.1f |\n| same into L0 host (pinned) memory "
                "| %.1f |\n", median(sub), median(rb4), std::min(five, b.logits.size()) / 1e6,
                median(rb5), median(rb5p));
  }
  return 0;
}

// Spec 8 M1 (plan 8b Task 2): the MTP head's draft logits against the CPU reference
// head (tools/oracle/mtp_ref.py --dump), on the golden prompts.
//
// For each prompt of n ids and the engine's own greedy continuation cont[] (plan 8a's
// dumps), the reference holds, for rows i < R, the head's depth-1 logits on
// (h[n-1+i], ids[n+i] = cont[i]) at position n-1+i, attending over depth-1 keys of every
// earlier row - with h the main model's post-final-norm hidden (docs/probe-mtp §1).
// The engine reaches the same state by prefilling the prompt and then teacher-forcing
// cont[] one plain MTP step at a time (verify(0) + commit(0, cont[i+1])); draft(1) at
// each i is the head on exactly that pair. Bar (spec §4 M1): per-row cosine >= 0.999
// and the reference's argmax among the engine's top 2.
//
// Also here: an engine loaded WITHOUT the head allocates no MtpBuffers and throws on
// every MTP call (the "MTP off is today's engine" half of Task 1's contract).
//
// usage: mtp_head_test <snapshot> <prompts dir> <ref dir> <cont dir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
using model::Qwen35;

double cosine(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb);
}
uint32_t argmax(const float* v, size_t n) {
  return uint32_t(std::max_element(v, v + n) - v);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <snapshot> <prompts dir> <ref dir> <cont dir>\n", argv[0]);
    return 2;
  }
  const std::string snap = argv[1], prompts = argv[2], ref_dir = argv[3], cont_dir = argv[4];
  const char* names[] = {"code", "prose", "cjk"};
  for (const char* n : names)
    if (!std::ifstream(ref_dir + "/" + n + ".mtp.safetensors").good()) {
      std::fprintf(stderr, "SKIP: %s/%s.mtp.safetensors is missing (tools/oracle/mtp_ref.py --dump)\n",
                   ref_dir.c_str(), n);
      return 77;
    }
  l0::Context ctx(0);

  {   // MTP off: nothing allocated, every MTP call refused.
    runtime::Engine off(ctx, loader::load(ctx, snap, 16384), 16384);
    CHECK(!off.mtp());
    CHECK(off.mtp_buffers() == nullptr);
    bool threw = false;
    try {
      off.draft(1);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }

  runtime::Engine e(ctx, loader::load(ctx, snap, 16384, /*mtp=*/true), 16384);
  CHECK(e.mtp());
  std::printf("draft list %zu launches, verify M=1..4 %zu/%zu/%zu/%zu launches\n",
              e.draft_step(0).kernel_count, e.verify_step(1).kernel_count,
              e.verify_step(2).kernel_count, e.verify_step(3).kernel_count,
              e.verify_step(4).kernel_count);
  runtime::Control* ctl = e.buffers().control.as<runtime::Control>();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const size_t V = Qwen35::kVocab, VU = Qwen35::kVocabUsed;
  std::vector<float> got(V);

  bool ok = true;
  double worst = 1.0, sum = 0;
  size_t rows_total = 0, top1 = 0;
  for (const char* n : names) {
    golden::Golden ref(ref_dir + "/" + n + ".mtp.safetensors");
    const uint32_t R = uint32_t(ref.dim("logits", 2, 0));
    const float* rl = ref.f32("logits", size_t(R) * V);
    const int32_t* rpos = ref.i32("pos", R);
    const int32_t* rnext = ref.i32("next", R);
    const std::vector<uint32_t> prompt = golden::read_ids(prompts + "/" + n + ".ids");
    const std::vector<uint32_t> cont = golden::read_ids(cont_dir + "/" + n + ".cont256.ids");
    CHECK(cont.size() > R);
    e.reset();
    e.prefill(prompt);
    std::printf("%s: n=%zu, engine argmax after prefill %u, cont[0] %u\n", n, prompt.size(),
                ctl->cur_token[0], cont[0]);
    double pw = 1.0;
    for (uint32_t i = 0; i < R; ++i) {
      CHECK_EQ(uint32_t(rpos[i]), uint32_t(prompt.size()) - 1 + i);
      CHECK_EQ(uint32_t(rnext[i]), cont[i]);
      ctl->cur_token[0] = cont[i];   // teacher forced: the pending id
      e.draft(1);
      imm.copy(got.data(), e.mtp_logits_device(), V * 4);
      const float* r = rl + size_t(i) * V;
      const double c = cosine(got.data(), r, VU);
      const uint32_t ra = argmax(r, VU), ea = argmax(got.data(), VU);
      // the engine's top 2
      uint32_t e2 = ea == 0 ? 1 : 0;
      for (uint32_t v = 0; v < VU; ++v)
        if (v != ea && got[v] > got[e2]) e2 = v;
      const bool in2 = ra == ea || ra == e2;
      if (c < 0.999 || !in2) {
        ok = false;
        std::printf("  row %u pos %d: cos %.6f ref argmax %u engine top2 %u,%u  FAIL\n", i,
                    rpos[i], c, ra, ea, e2);
      }
      top1 += ra == ea;
      pw = std::min(pw, c);
      sum += c;
      ++rows_total;
      e.verify(0);
      e.commit(0, cont[i + 1]);
    }
    worst = std::min(worst, pw);
    std::printf("%s: %u rows, worst cosine %.6f\n", n, R, pw);
  }
  std::printf("M1: %zu rows, worst cosine %.6f, mean %.6f, top-1 agreement %zu/%zu -> %s\n",
              rows_total, worst, sum / rows_total, top1, rows_total, ok ? "PASS" : "FAIL");
  CHECK(ok);
  std::puts("mtp_head_test OK");
  return 0;
}

// Open-work 35: replay one served context on a MoE model - the prompt through Engine::prefill on the given backend
// (chunk cap 4096, the int8 head, max_len 16384, as b70-serve ran it), the generated ids before the row through
// ingest (decode steps, teacher-forced), then the row's logits: its top two ids and the gap in bf16 ulp of the top.
// B70_PF_L0_FORM (router, ab, intra, experts) switches l0-int8's groups back to their l0 form.
//
//   probe_moe_int8_replay <snapshot> <context.ids> <prompt_len> <l0|l0-int8> [id ...]
//
// context.ids: the prompt then the generated ids before the row (serve_judge's <key>.prompt.ids). The extra ids
// print their logits too.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill_backend.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: probe_moe_int8_replay <snapshot> <context.ids> <prompt_len> <l0|l0-int8> [id ...]\n");
    return 2;
  }
  std::ifstream in(argv[2]);
  std::vector<uint32_t> ctx_ids;
  for (uint32_t id; in >> id;) ctx_ids.push_back(id);
  const size_t n = std::stoul(argv[3]);
  runtime::PrefillBackend b;
  if (n == 0 || n > ctx_ids.size() || !runtime::parse_prefill_backend(argv[4], b)) return 2;
  const char* form = std::getenv("B70_PF_L0_FORM");
  constexpr uint32_t kMaxLen = 16384;
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, argv[1], kMaxLen, false, loader::LmHeadForm::Int8), kMaxLen);
  e.set_prefill_backend(b);
  e.set_prefill_chunk_cap(4096);
  e.prepare_prefill();
  e.prefill(std::vector<uint32_t>(ctx_ids.begin(), ctx_ids.begin() + n));
  const auto* ctl = e.buffers().control.as<runtime::Control>();
  size_t diverged = 0;
  for (size_t i = n; i < ctx_ids.size(); ++i) {
    if (!diverged && ctl->cur_token[0] != ctx_ids[i]) diverged = i;
    e.ingest({ctx_ids[i]});
  }
  std::vector<float> v(model::Qwen35::kVocab);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(v.data(), (n == ctx_ids.size() ? e.prefill_scratch()->logits : e.buffers().logits).ptr(),
           v.size() * sizeof(float));
  uint32_t t1 = 0, t2 = 1;
  for (uint32_t i = 0; i < e.model().desc->vocab_used; ++i) {
    if (v[i] > v[t1]) { t2 = t1; t1 = i; }
    else if (i != t1 && v[i] > v[t2]) t2 = i;
  }
  const float ulp = std::ldexp(1.0f, std::ilogb(std::fabs(v[t1])) - 7);   // bf16: 8 significant bits
  std::printf("ARM %s form=%s: top %u (%.4f) second %u (%.4f) gap %.2f bf16 ulp; greedy %s the context%s",
              argv[4], form ? form : "-", t1, v[t1], t2, v[t2], (v[t1] - v[t2]) / ulp,
              diverged ? "left" : "kept", diverged ? (" at " + std::to_string(diverged - n)).c_str() : "");
  for (int a = 5; a < argc; ++a) {
    const uint32_t id = std::stoul(argv[a]);
    std::printf("; id %u %.4f (%.2f ulp below top)", id, v[id], (v[t1] - v[id]) / ulp);
  }
  std::printf("\n");
  return 0;
}

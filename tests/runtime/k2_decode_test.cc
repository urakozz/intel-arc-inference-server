// Spec 18b Task 3: K2-Horizon's decode gates on the card that are not the golden gate (box
// only, labels `checkpoint;k2`):
//
//   1. the list: K2Engine's captured step is runtime::k2::decode_launches() = 717 launches
//      (embed + 3 x 12 + 45 x 15 + 5), labelled, every layer's launches in the walk's order
//      (Review Focus 5 - recorded against spec 18 §2's ~1000 estimate);
//   2. the plan: runtime::k2::plan at this max_len equals what the engine holds, component by
//      component (the `--max-len auto` planner is the allocations' own arithmetic);
//   3. K3, replay determinism, bitwise: one prompt ingested and 16 tokens generated twice,
//      reset between - the ids, the final logits row, every layer's MoVA and MoE route rows
//      and the whole KV cache identical. No sum anywhere is atomic or completion-ordered.
//
// argv: <snapshot> <prompt ids> [int8]. Exit 77 (SKIP) when the checkpoint is not here.
// The prompt only has to be legal ids (< 250624): tests/golden/prompts/prose.ids (Qwen3.8's
// ids) is, and 18a's K2 ids are too.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "loader/snapshot.h"
#include "runtime/k2/k2_engine.h"
#include "runtime/k2/k2_sizes.h"

namespace {

constexpr uint32_t kMaxLen = 16384, kGen = 16;

std::vector<uint32_t> read_ids(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  std::vector<uint32_t> ids;
  for (long long v; f >> v;) {
    CHECK(v >= 0 && v < 250624);
    ids.push_back(uint32_t(v));
  }
  CHECK(!ids.empty());
  return ids;
}

struct Snap {
  std::vector<uint32_t> tokens, routes;
  std::vector<float> logits;
  std::vector<uint8_t> kv;
  bool operator==(const Snap& o) const {
    return tokens == o.tokens && routes == o.routes && logits == o.logits && kv == o.kv;
  }
};

Snap run_once(runtime::k2::K2Engine& eng, l0::CmdList& imm, const std::vector<uint32_t>& ids) {
  eng.reset();
  eng.ingest(ids);
  Snap s;
  s.tokens = eng.generate(kGen);
  s.logits = eng.read_logits();
  s.routes = eng.read_routes();
  // The KV positions this session wrote, every layer (K then V).
  const runtime::k2::K2Buffers& b = eng.buffers();
  const size_t row = size_t(b.desc.kv_n()) * 2, used = size_t(eng.pos()) * row;
  s.kv.resize(size_t(b.desc.layers) * used * 2);
  for (uint32_t l = 0; l < b.desc.layers; ++l) {
    imm.copy(s.kv.data() + size_t(l) * used * 2, b.kv_k_layer(l), used);
    imm.copy(s.kv.data() + size_t(l) * used * 2 + used, b.kv_v_layer(l), used);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string prompt = argc > 2 ? argv[2] : "tests/golden/prompts/prose.ids";
  const bool int8 = argc > 3 && std::string(argv[3]) == "int8";
  try {
    (void)loader::resolve_snapshot(arg);
  } catch (const std::exception& e) {
    std::printf("SKIP: no K2-Horizon checkpoint at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  l0::Context ctx(0);
  loader::K2LoadedModel model = loader::load_k2(
      ctx, arg, kMaxLen, int8 ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint);
  const model::K2Desc& d = *model.desc;
  const size_t weights = model.report.bytes.total();
  runtime::k2::K2Engine eng(ctx, std::move(model), kMaxLen);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // ---- 1. the list ------------------------------------------------------------------------
  CHECK_EQ(runtime::k2::decode_launches(d), size_t(717));
  CHECK_EQ(eng.step().kernel_count, size_t(717));
  CHECK_EQ(eng.step().labels.size(), size_t(717));
  CHECK(eng.step().labels.front().find("embed_gather") != std::string::npos);
  CHECK(eng.step().labels.back().find("argmax_stage2") != std::string::npos);
  size_t moe_down = 0, mova = 0;
  for (const std::string& l : eng.step().labels) {
    moe_down += l.find(" k2_moe_down ") != std::string::npos;
    mova += l.find(" k2_mova_value ") != std::string::npos;
  }
  CHECK_EQ(moe_down, size_t(45));
  CHECK_EQ(mova, size_t(45));
  std::printf("list: %zu launches per token (spec 18 §2 estimated ~1000; spec 4's slot design 2067), "
              "%zu modules\n", eng.step().kernel_count, eng.step().modules.size());

  // ---- 2. the plan against the allocation ---------------------------------------------------
  {
    const runtime::k2::Plan p = runtime::k2::plan(d, kMaxLen, weights);
    const runtime::MemoryComponents u = eng.memory_use();
    std::printf("%s\n%s\n", eng.memory_line().c_str(),
                runtime::k2::describe(p, ctx.memory_bytes(), 0).c_str());
    CHECK_EQ(u.model, p.model);
    CHECK_EQ(u.kv, p.kv);
    CHECK_EQ(u.decode_state, p.decode_state);
    CHECK_EQ(u.prefill_scratch, size_t(0));
  }

  // ---- 3. K3: replay determinism ------------------------------------------------------------
  const std::vector<uint32_t> ids = read_ids(prompt);
  const Snap a = run_once(eng, imm, ids), b = run_once(eng, imm, ids);
  if (!(a == b)) {
    std::fprintf(stderr, "replay determinism FAILED: tokens %s, logits %s, routes %s, KV %s\n",
                 a.tokens == b.tokens ? "same" : "DIFFER", a.logits == b.logits ? "same" : "DIFFER",
                 a.routes == b.routes ? "same" : "DIFFER", a.kv == b.kv ? "same" : "DIFFER");
    return 1;
  }
  std::printf("K3: two runs of %zu ids + %u tokens bitwise identical (ids, logits, every layer's "
              "MoVA / MoE route rows, the KV cache of %u positions)\n", ids.size(), kGen, eng.pos());
  std::printf("tokens:");
  for (uint32_t t : a.tokens) std::printf(" %u", t);
  std::printf("\nk2_decode_test OK\n");
  return 0;
}

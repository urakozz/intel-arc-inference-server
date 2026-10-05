// Spec 6 §10 (max_len auto) on the card: the memory plan equals what the engine
// actually allocates, component by component and buffer by buffer, at two max_lens -
// an explicit one (the loader's default 16384) and `auto` (the largest the plan fits
// beside the weights with the default reserve, reached the way the CLIs reach it:
// load at 4096, plan, loader::set_max_len). Device + checkpoint.
//
//   memory_plan_box_test <snapshot-or-repo> [mtp] [max_len|auto ...]
//
// `mtp` loads the MTP head (spec 8) and plans its buffers. The lengths default to
// "16384 auto". The default l0-int8 backend and the attention mode of
// B70_PREFILL_ATTN (flash unless set) are planned and run. One short prefill runs
// first, so the lazy buffers (the MTP prefill hidden rows, the slab on an MTP engine)
// exist when the engine is measured. At `auto` the engine fills the card up to the
// reserve: if the reserve is too small for what the plan does not count (driver,
// kernel modules, command lists), this test is where the allocation fails.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "check.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/engine.h"
#include "runtime/memory_plan.h"
#include "runtime/prefill/attn.h"

namespace {

void check_engine(l0::Context& ctx, const std::string& arg, bool mtp, const std::string& len_arg) {
  const std::string snap = loader::resolve_snapshot(arg);
  const uint32_t trained = loader::trained_context(snap);
  const bool is_auto = len_arg == "auto";
  const size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const runtime::PrefillPath path{
      true, runtime::PrefillBackend::L0Int8,
      runtime::prefill::attn_mode() == runtime::prefill::AttnMode::Composed};

  // The serving default head (spec 9: int8), as b70-serve loads it.
  loader::LoadedModel m = loader::load(ctx, snap, is_auto ? runtime::kMinAutoMaxLen
                                                          : uint32_t(std::stoul(len_arg)),
                                       mtp, loader::LmHeadForm::Int8);
  const model::ModelDesc& d = *m.desc;
  const size_t weights = m.report.total() - m.report.rope_bytes;
  uint32_t L = m.max_len;
  if (is_auto) {
    L = runtime::max_len_that_fits(d, mtp, weights, ctx.memory_bytes(), reserve, trained, path);
    CHECK(L != 0);
    loader::set_max_len(ctx, m, L);
  }
  const runtime::MemoryPlan p = runtime::plan(d, L, mtp, weights, path);
  std::printf("%s\n", runtime::describe(p, ctx.memory_bytes(), reserve).c_str());
  // The table the loader holds is the planner's figure.
  CHECK_EQ(m.report.rope_bytes, p.rope);
  CHECK_EQ(m.report.total(), p.model);

  runtime::Engine eng(ctx, std::move(m), L);
  eng.prepare_prefill();
  std::printf("before the first prefill: %s\n", eng.memory_line().c_str());
  // One short prefill: the lazy buffers a session builds on its first chunk.
  std::vector<uint32_t> ids(300);
  for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = 1000 + i;
  eng.prefill(ids);
  std::printf("after a prefill:          %s\n", eng.memory_line().c_str());

  // The five components memory_line() prints.
  const runtime::MemoryComponents u = eng.memory_use();
  CHECK_EQ(u.model, p.model);
  CHECK_EQ(u.kv, p.kv);
  CHECK_EQ(u.decode_state, p.decode_state);
  CHECK_EQ(u.prefill_scratch, p.prefill_scratch);
  CHECK_EQ(u.int8, p.int8);
  CHECK_EQ(u.total(), p.total());

  // Buffer by buffer: every allocation is its size function's value.
  const runtime::PersistentSizes ps = runtime::PersistentDims::sizes(L, d);
  runtime::DecodeBuffers& b = eng.buffers();
  CHECK_EQ(b.control.size(), ps.control);
  CHECK_EQ(b.gdn_state.size(), ps.gdn_state);
  CHECK_EQ(b.conv_ring.size(), ps.conv_ring);
  CHECK_EQ(b.kv_k.size(), ps.kv_k);
  CHECK_EQ(b.kv_v.size(), ps.kv_v);
  const runtime::DecodeScratchSizes ds = runtime::DecodeScratchDims::sizes(L, d);
  CHECK_EQ(b.resid.size(), ds.resid);
  CHECK_EQ(b.x.size(), ds.x);
  CHECK_EQ(b.partials.size(), ds.partials);
  CHECK_EQ(b.ab_out.size(), ds.ab_out);
  CHECK_EQ(b.norm_sumsq.size(), ds.norm_sumsq);
  CHECK_EQ(b.gdn_o.size(), ds.gdn_o);
  CHECK_EQ(b.attn_q.size(), ds.attn_q);
  CHECK_EQ(b.attn_gate.size(), ds.attn_gate);
  CHECK_EQ(b.attn_part.size(), ds.attn_part);
  CHECK_EQ(b.attn_out.size(), ds.attn_out);
  CHECK_EQ(b.logits.size(), ds.logits);
  CHECK_EQ(b.argmax_part.size(), ds.argmax_part);
  const runtime::PrefillScratch* pf = eng.prefill_scratch();
  CHECK(pf != nullptr);
  const runtime::PrefillScratchSizes s = runtime::PrefillScratchDims::sizes(L, d);
  CHECK_EQ(pf->bytes(), s.eager());
  CHECK_EQ(pf->lazy_bytes(), p.prefill_lazy);
  CHECK_EQ(pf->pf_s_bytes(), path.composed_attn ? s.pf_s : size_t{0});
  if (mtp) {
    const runtime::MtpBuffers* mb = eng.mtp_buffers();
    CHECK(mb != nullptr);
    const runtime::MtpSizes ms = runtime::MtpDims::sizes(L, d);
    CHECK_EQ(mb->hctl.size(), ms.hctl);
    CHECK_EQ(mb->gdn_spec.size(), ms.gdn_spec);
    CHECK_EQ(mb->kv_k.size(), ms.kv_k);
    CHECK_EQ(mb->kv_v.size(), ms.kv_v);
    CHECK_EQ(mb->hh.size(), ms.hh);
    CHECK_EQ(mb->dh.size(), ms.dh);
    CHECK_EQ(mb->logits.size(), ms.logits);
    CHECK(eng.mtp_prefill_hidden() != nullptr);
    CHECK_EQ(eng.mtp_prefill_hidden()->size(), p.mtp_hidden);
  }
  // The engine decodes at this length: a few tokens after the prefill.
  const std::vector<uint32_t> out = eng.generate(4);
  CHECK_EQ(out.size(), size_t{4});
  std::printf("%s %s at max_len %u: plan == allocation, %zu B (%.3f GB)\n", d.name.c_str(),
              mtp ? "+ MTP" : "", L, p.total(), p.total() / 1e9);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  int i = 2;
  const bool mtp = argc > i && std::string(argv[i]) == "mtp";
  if (mtp) ++i;
  std::vector<std::string> lens;
  for (; i < argc; ++i) lens.push_back(argv[i]);
  if (lens.empty()) lens = {"16384", "auto"};
  l0::Context ctx(0);
  std::printf("device: %s, %zu B (%.3f GB)\n", ctx.name().c_str(), ctx.memory_bytes(),
              ctx.memory_bytes() / 1e9);
  for (const std::string& len : lens) check_engine(ctx, arg, mtp, len);
  std::puts("memory_plan_box_test OK");
  return 0;
}

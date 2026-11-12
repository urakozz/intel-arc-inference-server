// Spec 19 v2: profiling observes the draft and projection lists without changing their results.
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/dflash/dflash_drafter.h"

// Same real weights and ring, with only the attention grid changed. Interleave timestamp measurements.
void check_attn_heads(runtime::Engine& e, runtime::dflash::DFlashDrafter& d, uint32_t K) {
  std::array<std::unique_ptr<runtime::ProfileEvents>, 3> events;
  std::array<std::unique_ptr<runtime::CapturedStep>, 3> lists;
  for (unsigned h = 0; h < 3; ++h) {
    events[h] = std::make_unique<runtime::ProfileEvents>(e.context());
    runtime::CaptureOptions opt;
    opt.dflash_attn_heads = h;
    opt.profile = events[h].get();
    lists[h] = std::make_unique<runtime::CapturedStep>(runtime::dflash::build_dflash_draft(
        e.context(), d.weights(), runtime::dflash::dflash_target(e.model()), d.buffers(), e.buffers().control, opt));
    CHECK_EQ(lists[h]->kernel_count, 87u);
  }
  const auto base = d.draft(e, K, 0.0f, 0);
  l0::CmdList imm = l0::CmdList::immediate(e.context());
  std::vector<uint8_t> hidden(d.buffers().h.size()), observed(hidden.size());
  imm.copy(hidden.data(), d.buffers().h.ptr(), hidden.size());
  std::array<std::vector<double>, 3> timings;
  for (unsigned r = 0; r < 6; ++r)
    for (unsigned a = 0; a < 3; ++a) {
      const unsigned h = (a + r) % 3;
      for (auto& event : events[h]->events) event.reset();
      e.replay_aux(lists[h]->list);
      CHECK(std::memcmp(&base, d.buffers().readback(), sizeof(base)) == 0);
      imm.copy(observed.data(), d.buffers().h.ptr(), observed.size());
      CHECK(hidden == observed);
      double ms = 0;
      for (size_t i = 0; i < lists[h]->labels.size(); ++i)
        if (lists[h]->labels[i].find(" df_attn_part ") != std::string::npos)
          ms += events[h]->events[i].duration_us() / 1000.0;
      CHECK(ms > 0);
      if (r) timings[h].push_back(ms);
    }
  for (unsigned h = 0; h < 3; ++h) {
    std::sort(timings[h].begin(), timings[h].end());
    std::printf("DFLASH_ATTN_HEADS pos=%u K=%u heads=%u ms=%.6f readback_hidden=bitwise\n",
                e.pos(), K, h ? h : 4, timings[h][2]);
  }
}

int main(int argc, char** argv) {
  std::string target, drafter;
  try {
    target = loader::resolve_snapshot(argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
    drafter = loader::resolve_snapshot(argc > 2 ? argv[2] : "z-lab/Qwen3.8-27B-DFlash2");
  } catch (const std::exception& ex) {
    std::printf("SKIP: %s\n", ex.what());
    return 77;
  }
  constexpr uint32_t max_len = 8192;
  l0::Context ctx(0);
  auto model = loader::load(ctx, target, max_len, false, loader::LmHeadForm::Int8);
  auto weights = loader::load_drafter(ctx, drafter, model, loader::DraftFormat::Int8);
  const auto rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  runtime::dflash::DFlashDrafter d(ctx, std::move(weights), rows, max_len, 7);
  runtime::Engine e(ctx, std::move(model), max_len, false, runtime::KvCache::Bf16, &d);
  d.attach(e, false);
  e.prefill({760, 72103, 506, 37119, 557, 11012, 3213, 310, 6512, 279, 61789, 272, 1072, 2272, 279, 197616});
  const runtime::dflash::DFlashDraftOut plain = d.draft(e, 7, 0.0f, 0);
  runtime::ProfileEvents draft_events(ctx), project_events(ctx);
  runtime::CaptureOptions opt;
  opt.profile = &draft_events;
  auto draft = runtime::dflash::build_dflash_draft(ctx, d.weights(), runtime::dflash::dflash_target(e.model()),
                                                 d.buffers(), e.buffers().control, opt);
  opt.profile = &project_events;
  auto project = runtime::dflash::build_dflash_project(ctx, d.weights(), d.buffers(), opt);
  CHECK(draft.labels == d.draft_step()->labels);
  CHECK(project.labels == d.project_step().labels);
  CHECK_EQ(draft_events.events.size(), draft.kernel_count);
  CHECK_EQ(project_events.events.size(), project.kernel_count);
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    for (auto& event : draft_events.events) event.reset();
    e.replay_aux(draft.list);
    CHECK(std::memcmp(&plain, d.buffers().readback(), sizeof(plain)) == 0);
    double kernel_us = 0;
    for (const auto& event : draft_events.events) {
      CHECK(event.duration_us() > 0);
      kernel_us += event.duration_us();
    }
    CHECK(draft_events.events.back().elapsed_us(draft_events.events.front()) >= kernel_us);
    CHECK(draft_events.events.front().elapsed_us(draft_events.events.front()) == draft_events.events.front().duration_us());
  }
  check_attn_heads(e, d, 7);
  e.verify(0);
  e.commit(0, e.verify_ids()[0]);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<uint8_t> ring(d.ring_bytes()), profiled(ring.size());
  d.save_ring(imm, ring.data());
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    for (auto& event : project_events.events) event.reset();
    e.replay_aux(project.list);
    d.save_ring(imm, profiled.data());
    CHECK(ring == profiled);
    for (const auto& event : project_events.events) CHECK(event.duration_us() > 0);
  }
  l0::Mem state(ctx, l0::MemKind::Host, e.state_bytes());
  e.save_state(state.ptr());
  const uint32_t pos = e.pos(), anchor = e.control()->cur_token[0];
  d.draft(e, 7, 0.0f, 0);
  e.verify(7);
  const std::vector<uint32_t> ids(e.verify_ids(), e.verify_ids() + 8);
  std::vector<uint8_t> taps(d.taps().size()), observed(taps.size());
  imm.copy(taps.data(), d.taps().ptr(), taps.size());
  runtime::ProfileEvents verify_events(ctx);
  opt.profile = &verify_events;
  opt.profile_m = 8;
  e.recapture_verify(opt);
  CHECK_EQ(verify_events.events.size(), e.verify_step_at(8, pos + 1).kernel_count);
  e.load_state(state.ptr(), pos);
  e.control()->cur_token[0] = anchor;
  d.draft(e, 7, 0.0f, 0);
  for (auto& event : verify_events.events) event.reset();
  e.verify(7);
  CHECK(ids == std::vector<uint32_t>(e.verify_ids(), e.verify_ids() + 8));
  imm.copy(observed.data(), d.taps().ptr(), observed.size());
  CHECK(taps == observed);
  for (const auto& event : verify_events.events) CHECK(event.duration_us() > 0);
  // M = 3 switches to a second attention list at 4096 keys. Profile the list actually replayed.
  e.reset();
  e.prefill(std::vector<uint32_t>(4096, 760));
  e.save_state(state.ptr());
  const uint32_t deep_pos = e.pos(), deep_anchor = e.control()->cur_token[0];
  check_attn_heads(e, d, 2);
  d.draft(e, 2, 0.0f, 0);
  e.verify(2);
  const std::vector<uint32_t> deep_ids(e.verify_ids(), e.verify_ids() + 3);
  runtime::ProfileEvents deep_events(ctx);
  opt.profile = &deep_events;
  opt.profile_m = 3;
  opt.profile_keys = deep_pos + 1;
  e.recapture_verify(opt);
  CHECK_EQ(deep_events.events.size(), e.verify_step_at(3, deep_pos + 1).kernel_count);
  e.load_state(state.ptr(), deep_pos);
  e.control()->cur_token[0] = deep_anchor;
  d.draft(e, 2, 0.0f, 0);
  for (auto& event : deep_events.events) event.reset();
  e.verify(2);
  CHECK(deep_ids == std::vector<uint32_t>(e.verify_ids(), e.verify_ids() + 3));
  for (const auto& event : deep_events.events) CHECK(event.duration_us() > 0);
  e.recapture_verify({});   // release the profiled list before its event pool
  std::printf("dflash_profile_test OK: %zu draft and %zu projection events; readback/ring and shallow/deep verify bitwise\n",
              draft_events.events.size(), project_events.events.size());
}

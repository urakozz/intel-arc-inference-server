// Spec 16b P1 (plan 16b Review Focus 1-3, 5): decode across two B70s is BITWISE the
// single-card decode on device 0. Every comparison is memcmp - a difference is a bug, not
// noise: PP changes where layers run, not what they compute (the same kernels in the same
// order on the same data, plus the hand-off's copies).
//
//   reference  one card (runtime::Engine on device 0): the prose prompt ingested, then 64
//              greedy ids; its logits row, every layer's GDN state and conv ring
//              (save_state), every FA layer's KV over [0, pos) (save_kv) and the Control
//              block. And a spec 7 restore: a 4395-id prefill, the snapshot at 4395, one id
//              ingested after it, 64 greedy ids - the same five things.
//   each configuration  --pipeline 2 with copy and peer, the auto split, and two uneven
//              cuts (s = 5: one FA layer on device 0; s = 62: one FA layer on device 1):
//              the same ingest + 64 ids, everything above bitwise, both Controls equal after
//              every phase (pos in step); then the restore - the SAME snapshot bytes loaded
//              into the pipeline (the host layouts are the single card's), one id, 64 ids,
//              bitwise; then (first configuration) a reset and the whole run again, bitwise
//              (replay determinism with --pipeline 2).
//
// argv: [1] the checkpoint, [2] `int8` for the int8 lm_head. B70_KV_CACHE selects the KV form
// (pp_decode_kv8_test). Exits 77 (SKIP) with fewer than two GPUs or without peer access.
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"

namespace {
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 64;
constexpr uint32_t kRestoreAt = 4395;   // 4096 + 299: not a block end, pos % 16 = 11

struct Result {
  std::vector<uint32_t> ids;
  std::vector<float> logits;
  std::vector<uint8_t> state, kv;
  runtime::Control ctl{};
};

std::vector<uint8_t> state_of(size_t n, const std::function<void(void*)>& save) {
  std::vector<uint8_t> v(n);
  save(v.data());
  return v;
}

std::vector<uint32_t> repeat_to(const std::vector<uint32_t>& src, size_t n) {
  std::vector<uint32_t> out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}

// The reference's end-of-run record on one card.
Result record(l0::Context& ctx, runtime::Engine& e, std::vector<uint32_t> ids) {
  Result r;
  r.ids = std::move(ids);
  r.logits.resize(model::Qwen35::kVocab);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(r.logits.data(), e.buffers().logits.ptr(), r.logits.size() * sizeof(float));
  r.state = state_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
  r.kv = state_of(e.kv_bytes(e.pos()), [&](void* h) { e.save_kv(0, e.pos(), h); });
  std::memcpy(&r.ctl, e.buffers().control.ptr(), sizeof(runtime::Control));
  return r;
}

Result record(runtime::PipelineEngine& e, std::vector<uint32_t> ids) {
  Result r;
  r.ids = std::move(ids);
  r.logits = e.read_logits();
  r.state = state_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
  r.kv = state_of(e.kv_bytes(e.pos()), [&](void* h) { e.save_kv(0, e.pos(), h); });
  r.ctl = e.control(1);
  return r;
}

// Review Focus 3: the two Control blocks are one at every step boundary.
void check_in_step(const runtime::PipelineEngine& e, const char* where) {
  const runtime::Control& a = e.control(0);
  const runtime::Control& b = e.control(1);
  if (std::memcmp(&a, &b, sizeof(runtime::Control)) != 0) {
    std::fprintf(stderr, "%s: the two Control blocks differ (pos %u / %u, cur %u / %u)\n", where,
                 a.pos, b.pos, a.cur_token[0], b.cur_token[0]);
    CHECK(false);
  }
}

size_t first_diff(const void* a, const void* b, size_t n) {
  const auto* x = static_cast<const uint8_t*>(a);
  const auto* y = static_cast<const uint8_t*>(b);
  for (size_t i = 0; i < n; ++i)
    if (x[i] != y[i]) return i;
  return n;
}

void check_same(const Result& ref, const Result& got, const std::string& what) {
  bool ok = true;
  const auto report = [&](const char* field, size_t at, size_t n) {
    if (at == n) return;
    std::fprintf(stderr, "%s: %s differs at byte %zu of %zu\n", what.c_str(), field, at, n);
    ok = false;
  };
  if (ref.ids != got.ids) {
    size_t i = 0;
    while (i < ref.ids.size() && i < got.ids.size() && ref.ids[i] == got.ids[i]) ++i;
    std::fprintf(stderr, "%s: ids differ at %zu\n", what.c_str(), i);
    ok = false;
  }
  report("logits", first_diff(ref.logits.data(), got.logits.data(), ref.logits.size() * 4),
         ref.logits.size() * 4);
  CHECK_EQ(ref.state.size(), got.state.size());
  report("GDN state + conv ring", first_diff(ref.state.data(), got.state.data(), ref.state.size()),
         ref.state.size());
  CHECK_EQ(ref.kv.size(), got.kv.size());
  report("KV", first_diff(ref.kv.data(), got.kv.data(), ref.kv.size()), ref.kv.size());
  CHECK_EQ(ref.ctl.pos, got.ctl.pos);
  CHECK_EQ(ref.ctl.cur_token[0], got.ctl.cur_token[0]);
  CHECK(ok);
  std::printf("%s: %zu ids, logits, %zu B of state, %zu B of KV bitwise; pos %u\n", what.c_str(),
              got.ids.size(), got.state.size(), got.kv.size(), got.ctl.pos);
  std::fflush(stdout);
}

struct Config {
  runtime::PpHandoff handoff;
  uint32_t split;   // 0 = auto
  bool replay;
};
}  // namespace

int main(int argc, char** argv) {
  const std::string checkpoint =
      argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const loader::LmHeadForm head = argc > 2 && std::string(argv[2]) == "int8"
                                      ? loader::LmHeadForm::Int8
                                      : loader::LmHeadForm::Checkpoint;
  const runtime::KvCache kv = runtime::default_kv_cache();
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --pipeline 2 needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1)) {
    std::printf("SKIP: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  const std::vector<uint32_t> prompt = golden::read_ids("tests/golden/prompts/prose.ids");
  const std::vector<uint32_t> longp =
      repeat_to(golden::read_ids("tests/golden/prompts/long32k.ids"), kRestoreAt + 1);

  // --- the single-card reference on device 0 ---------------------------------------------
  Result ref, ref_restore;
  std::vector<uint8_t> snap_state, snap_kv;
  {
    runtime::Engine e(d0, loader::load(d0, checkpoint, kMaxLen, false, head), kMaxLen, false, kv);
    e.ingest(prompt);
    std::vector<uint32_t> ids = e.generate(kGen);
    ref = record(d0, e, std::move(ids));
    // Spec 7: prefill to 4395, snapshot, one id by decode, 64 ids.
    e.reset();
    e.prepare_prefill();
    e.prefill(std::vector<uint32_t>(longp.begin(), longp.begin() + kRestoreAt));
    CHECK_EQ(e.pos(), kRestoreAt);
    snap_state = state_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
    snap_kv = state_of(e.kv_bytes(kRestoreAt), [&](void* h) { e.save_kv(0, kRestoreAt, h); });
    e.ingest({longp[kRestoreAt]});
    std::vector<uint32_t> rids = e.generate(kGen);
    ref_restore = record(d0, e, std::move(rids));
    std::printf("reference (one card, device 0): %u ids, restore at %u\n", kGen, kRestoreAt);
  }

  const model::ModelDesc* desc = nullptr;
  for (const Config c : {Config{runtime::PpHandoff::Copy, 0, true},
                         Config{runtime::PpHandoff::Peer, 0, true},
                         Config{runtime::PpHandoff::Copy, 5, false},
                         Config{runtime::PpHandoff::Peer, 62, false}}) {
    loader::LoadedModel full = loader::load(d0, checkpoint, kMaxLen, false, head);
    desc = full.desc;
    const runtime::PpWeights w = runtime::pp_weights(full);
    const uint32_t split = c.split ? c.split : runtime::pp_auto_split(*desc, w, kMaxLen, kv);
    runtime::PipelineOptions opt;
    opt.handoff = c.handoff;
    runtime::PipelineEngine e(d0, d1, runtime::place_stages(d0, d1, std::move(full), split), kMaxLen,
                              opt, kv);
    const std::string what = std::string("--pipeline 2, ") + runtime::pp_handoff_name(c.handoff) +
                             ", split " + std::to_string(split) + (c.split ? "" : " (auto)");
    // Review Focus 5: the split is the descriptor's, the launches the single card's.
    CHECK_EQ(e.split(), split);
    CHECK_EQ(e.stage(0).gdn + e.stage(1).gdn, desc->gdn_layers);
    CHECK_EQ(e.stage(0).fa + e.stage(1).fa, desc->fa_layers);
    const size_t peer = c.handoff == runtime::PpHandoff::Peer ? 2 : 0;
    CHECK_EQ(e.step(0).kernel_count + e.step(1).kernel_count, runtime::decode_launches(*desc) + peer);
    for (uint32_t i = 0; i < runtime::kPpDevices; ++i)
      std::printf("%s: %s\n", what.c_str(), e.memory_line(i).c_str());

    for (int round = 0; round < (c.replay ? 2 : 1); ++round) {
      e.reset();
      check_in_step(e, "after reset");
      e.ingest(prompt);
      check_in_step(e, "after ingest");
      CHECK_EQ(e.pos(), uint32_t(prompt.size()));
      std::vector<uint32_t> ids = e.generate(kGen);
      check_in_step(e, "after generate");
      check_same(ref, record(e, std::move(ids)), what + (round ? " (replayed after reset)" : ""));
    }

    // Spec 7 under PP: the one-card snapshot, restored into the pipeline.
    e.reset();
    e.load_state(snap_state.data(), kRestoreAt);
    e.load_kv(0, kRestoreAt, snap_kv.data());
    CHECK_EQ(e.control(0).pos, kRestoreAt);
    check_in_step(e, "after load_state");
    // The restore wrote exactly the snapshot: read it back through the pipeline's layouts.
    CHECK(state_of(e.state_bytes(), [&](void* h) { e.save_state(h); }) == snap_state);
    CHECK(state_of(e.kv_bytes(kRestoreAt), [&](void* h) { e.save_kv(0, kRestoreAt, h); }) == snap_kv);
    e.ingest({longp[kRestoreAt]});
    std::vector<uint32_t> rids = e.generate(kGen);
    check_in_step(e, "after the restored run");
    check_same(ref_restore, record(e, std::move(rids)), what + ", restored at 4395");
  }
  std::printf("pp_decode_test: OK (%s, %s KV)\n", desc ? desc->name.c_str() : "?",
              runtime::kv_cache_name(kv));
  return 0;
}

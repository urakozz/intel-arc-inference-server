// Spec 16c P1 / P2 (plan 16c Review Focus 1-4): prefill across two B70s is BITWISE the
// single-card prefill on device 0. PP changes where layers run, not what they compute: the
// same chunks (runtime::prefill_chunk_rows), the same kernels in the same order on the same
// data, plus the hand-off's copies. Every comparison is a byte comparison - a difference is
// a bug, and the test names the first differing region.
//
//   reference  one card (runtime::Engine on device 0, the backend in argv[2]): for each case
//              below, the prefill's logits row, every layer's GDN state and conv ring
//              (save_state), every FA layer's KV over [0, pos) (save_kv), the Control block,
//              and 16 greedy ids decoded after it (the token's way back through the mirror).
//   cases      `3 chunks`  4103 ids at chunk 2048 (two whole chunks and a 7-row tail);
//              `16 chunks` 32768 ids (the pipeline full for 14 chunks);
//              `1 chunk`   2 ids;
//              `split`     prefill_split under the pipeline: 4103 ids in two calls split at
//                          2500, chunk 1000 (a continuation off the 64-row GDN grid);
//              `hooks`     5000 ids with spec 7's block hook: at 2048 and 4096 (mid-prompt,
//                          read from the shadows while the next chunk runs) and at 5000, the
//                          snapshot (save_state + save_kv) each hook takes;
//              `hook throws` the same with a hook that throws at 2048: pos 2048 and the
//                          state the 2048 snapshot holds, as on one card;
//              `one -> two` a one-card prefill of 3000 ids, its snapshot restored into the
//                          pipeline, the remaining 1103 prefilled there = one card's 4103 in
//                          two calls; `two -> one`: the pipeline's own 3000-id snapshot is the
//                          one card's byte for byte, so restoring it on one card IS that run.
//   configurations  --pp 2 with copy and peer at the auto split, and the extreme cuts
//              (s = 5: one FA layer on device 0; s = 62: one FA layer on device 1).
//
// Also asserted: each device's prefill launches equal step_stage_launches' arithmetic (plus
// the hand-off's kernels and the head), and the two Control blocks are one after every
// prefill. argv: [1] the checkpoint, [2] the backend (l0 | l0-int8, default l0-int8), [3]
// `int8` for the int8 lm_head. B70_KV_CACHE selects the KV form (pp_prefill_kv8_test).
// Exits 77 (SKIP) with fewer than two GPUs or without peer access.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"
#include "runtime/prefill/backend.h"

namespace {
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 40960;   // the 32768-id case and its decode
constexpr uint32_t kGen = 16;

// A host image too large to keep (32768 positions of KV are 2 GB): FNV-1a per 16 MiB block,
// so a difference is located to a block without holding both images.
struct Digest {
  size_t bytes = 0;
  std::vector<uint64_t> blocks;
};
constexpr size_t kBlockBytes = size_t(16) << 20;
Digest digest(const std::vector<uint8_t>& v) {
  Digest d;
  d.bytes = v.size();
  for (size_t o = 0; o < v.size(); o += kBlockBytes) {
    uint64_t h = 1469598103934665603ull;
    const size_t e = std::min(v.size(), o + kBlockBytes);
    for (size_t i = o; i < e; ++i) h = (h ^ v[i]) * 1099511628211ull;
    d.blocks.push_back(h);
  }
  return d;
}
// "" when equal, else where they first differ.
std::string differ(const Digest& a, const Digest& b) {
  if (a.bytes != b.bytes) return "sizes " + std::to_string(a.bytes) + " / " + std::to_string(b.bytes);
  for (size_t i = 0; i < a.blocks.size(); ++i)
    if (a.blocks[i] != b.blocks[i])
      return "bytes [" + std::to_string(i * kBlockBytes) + ", " +
             std::to_string(std::min(a.bytes, (i + 1) * kBlockBytes)) + ")";
  return "";
}

struct Record {
  std::vector<float> logits;   // the prefill's head row
  Digest state, kv;
  runtime::Control ctl{};
  Ids gen;                     // kGen greedy ids after the prefill
};

template <class E>
Digest state_digest(const E& e) {
  std::vector<uint8_t> v(e.state_bytes());
  e.save_state(v.data());
  return digest(v);
}
template <class E>
Digest kv_digest(const E& e, uint32_t n) {
  std::vector<uint8_t> v(e.kv_bytes(n));
  e.save_kv(0, n, v.data());
  return digest(v);
}

Record record(runtime::Engine& e, l0::CmdList& imm) {
  Record r;
  r.logits.resize(model::Qwen35::kVocab);
  imm.copy(r.logits.data(), e.prefill_scratch()->logits.ptr(), r.logits.size() * 4);
  r.state = state_digest(e);
  r.kv = kv_digest(e, e.pos());
  std::memcpy(&r.ctl, e.buffers().control.ptr(), sizeof(runtime::Control));
  r.gen = e.generate(kGen);
  return r;
}
Record record(runtime::PipelineEngine& e) {
  Record r;
  r.logits = e.read_prefill_logits();
  r.state = state_digest(e);
  r.kv = kv_digest(e, e.pos());
  r.ctl = e.control(1);
  CHECK(std::memcmp(&e.control(0), &e.control(1), sizeof(runtime::Control)) == 0);
  r.gen = e.generate(kGen);
  return r;
}

void check_same(const Record& ref, const Record& got, const std::string& what) {
  bool ok = true;
  const auto bad = [&](const char* field, const std::string& where) {
    if (where.empty()) return;
    std::fprintf(stderr, "%s: %s differs (%s)\n", what.c_str(), field, where.c_str());
    ok = false;
  };
  if (ref.logits != got.logits) {
    size_t i = 0;
    while (i < ref.logits.size() && std::memcmp(&ref.logits[i], &got.logits[i], 4) == 0) ++i;
    bad("the prefill's logits row", "first at vocab row " + std::to_string(i));
  }
  bad("GDN state + conv ring", differ(ref.state, got.state));
  bad("KV", differ(ref.kv, got.kv));
  if (ref.ctl.pos != got.ctl.pos || ref.ctl.cur_token[0] != got.ctl.cur_token[0])
    bad("Control", "pos " + std::to_string(got.ctl.pos) + " / " + std::to_string(ref.ctl.pos) +
                       ", first id " + std::to_string(got.ctl.cur_token[0]) + " / " +
                       std::to_string(ref.ctl.cur_token[0]));
  if (ref.gen != got.gen) bad("the decoded ids", "after the prefill");
  CHECK(ok);
  std::printf("%s: logits, %zu B of state, %zu B of KV bitwise, pos %u, %zu ids decoded after\n",
              what.c_str(), got.state.bytes, got.kv.bytes, got.ctl.pos, got.gen.size());
  std::fflush(stdout);
}

// One block hook's snapshot: (end, is_block_end, state, KV [0, end)).
struct Snap {
  uint32_t end;
  bool block;
  Digest state, kv;
  bool operator==(const Snap& o) const {
    return end == o.end && block == o.block && differ(state, o.state).empty() && differ(kv, o.kv).empty();
  }
};

Ids repeat_to(const Ids& src, size_t n) {
  Ids out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}
Ids head(const Ids& v, size_t n) { return Ids(v.begin(), v.begin() + long(n)); }
Ids tail(const Ids& v, size_t from) { return Ids(v.begin() + long(from), v.end()); }

struct Case {
  const char* name;
  size_t n;
  uint32_t chunk;
  size_t split;   // 0: one call
};
const Case kCases[] = {{"3 chunks", 4103, 2048, 0}, {"16 chunks", 32768, 2048, 0},
                       {"1 chunk", 2, 2048, 0},     {"split", 4103, 1000, 2500}};
constexpr uint32_t kHookN = 5000, kRestoreAt = 3000, kRestoreN = 4103;

struct Config {
  runtime::PpHandoff handoff;
  uint32_t split;   // 0 = auto
  bool all;         // every case (else the short ones)
};
}  // namespace

int main(int argc, char** argv) {
  const std::string checkpoint = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  runtime::PrefillBackend backend = runtime::PrefillBackend::L0Int8;
  if (argc > 2) CHECK(runtime::parse_prefill_backend(argv[2], backend) && runtime::is_l0(backend));
  const loader::LmHeadForm lm = argc > 3 && std::string(argv[3]) == "int8" ? loader::LmHeadForm::Int8
                                                                           : loader::LmHeadForm::Checkpoint;
  const runtime::KvCache kv = runtime::default_kv_cache();
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --pp 2 needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1)) {
    std::printf("SKIP: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  const Ids prompt = repeat_to(golden::read_ids("tests/golden/prompts/long32k.ids"), 32768);

  // --- the single-card references on device 0 --------------------------------------------
  std::vector<Record> ref;
  std::vector<Snap> ref_hooks;
  Digest ref_throw_state, ref_throw_kv;
  Record ref_restore;
  std::vector<uint8_t> snap_state, snap_kv;   // the one-card snapshot at kRestoreAt
  {
    runtime::Engine e(d0, loader::load(d0, checkpoint, kMaxLen, false, lm), kMaxLen, false, kv);
    e.set_prefill_backend(backend);
    e.prepare_prefill();
    l0::CmdList imm = l0::CmdList::immediate(d0);
    for (const Case& c : kCases) {
      e.reset();
      const Ids p = head(prompt, c.n);
      if (c.split) {
        e.prefill(head(p, c.split), c.chunk);
        e.prefill(tail(p, c.split), c.chunk);
      } else {
        e.prefill(p, c.chunk);
      }
      ref.push_back(record(e, imm));
    }
    e.reset();
    e.set_block_hook([&](uint32_t end, bool block) {
      ref_hooks.push_back({end, block, state_digest(e), kv_digest(e, end)});
    });
    e.prefill(head(prompt, kHookN));
    e.set_block_hook({});
    CHECK_EQ(ref_hooks.size(), size_t{3});
    // The throwing hook: one card stops with pos at the block end.
    e.reset();
    e.set_block_hook([](uint32_t end, bool) {
      if (end == 2048) throw std::runtime_error("hook");
    });
    bool threw = false;
    try {
      e.prefill(head(prompt, kHookN));
    } catch (const std::runtime_error&) {
      threw = true;
    }
    e.set_block_hook({});
    CHECK(threw && e.pos() == 2048);
    ref_throw_state = state_digest(e);
    ref_throw_kv = kv_digest(e, 2048);
    // The restore pair: 3000 ids, the snapshot, the rest.
    e.reset();
    e.prefill(head(prompt, kRestoreAt));
    snap_state.resize(e.state_bytes());
    e.save_state(snap_state.data());
    snap_kv.resize(e.kv_bytes(kRestoreAt));
    e.save_kv(0, kRestoreAt, snap_kv.data());
    e.prefill(Ids(prompt.begin() + kRestoreAt, prompt.begin() + kRestoreN));
    ref_restore = record(e, imm);
    std::printf("references (one card, device 0, %s): %zu cases, 3 hooks, a throwing hook, the "
                "restore pair\n", runtime::prefill_backend_name(backend), ref.size());
  }

  const model::ModelDesc* desc = nullptr;
  for (const Config cf : {Config{runtime::PpHandoff::Copy, 0, true}, Config{runtime::PpHandoff::Peer, 0, true},
                          Config{runtime::PpHandoff::Copy, 5, false}, Config{runtime::PpHandoff::Peer, 62, false}}) {
    loader::LoadedModel full = loader::load(d0, checkpoint, kMaxLen, false, lm);
    desc = full.desc;
    const runtime::PpWeights w = runtime::pp_weights(full);
    const uint32_t split = cf.split ? cf.split : runtime::pp_auto_split(*desc, w, kMaxLen, kv);
    runtime::PipelineOptions opt;
    opt.handoff = cf.handoff;
    runtime::PipelineEngine e(d0, d1, runtime::place_stages(d0, d1, std::move(full), split), kMaxLen, opt, kv);
    e.set_prefill_backend(backend);
    e.prepare_prefill();
    const std::string what = std::string("--pp 2 ") + runtime::prefill_backend_name(backend) + ", " +
                             runtime::pp_handoff_name(cf.handoff) + ", split " + std::to_string(split) +
                             (cf.split ? "" : " (auto)");
    for (uint32_t i = 0; i < runtime::kPpDevices; ++i)
      std::printf("%s: %s\n", what.c_str(), e.memory_line(i).c_str());

    for (size_t k = 0; k < sizeof kCases / sizeof kCases[0]; ++k) {
      const Case& c = kCases[k];
      if (!cf.all && c.n > 5000) continue;
      e.reset();
      const Ids p = head(prompt, c.n);
      if (c.split) {
        e.prefill(head(p, c.split), c.chunk);
        e.prefill(tail(p, c.split), c.chunk);
      } else {
        e.prefill(p, c.chunk);
      }
      const runtime::PipelineEngine::PrefillStats& st = e.last_prefill();
      for (uint32_t i = 0; i < runtime::kPpDevices; ++i) CHECK_EQ(st.launches[i], st.expected_launches[i]);
      std::printf("%s, %s: %u chunks, wall %.1f ms, device 0 busy %.1f ms, device 1 busy %.1f ms\n",
                  what.c_str(), c.name, st.chunks, st.wall_ms, st.busy_ms[0], st.busy_ms[1]);
      check_same(ref[k], record(e), what + ", " + c.name);
    }

    // Spec 7: the hooks, their snapshots - mid-prompt ones from the shadows - bitwise.
    std::vector<Snap> hooks;
    e.reset();
    e.set_block_hook([&](uint32_t end, bool block) {
      CHECK_EQ(e.pos(), end);
      hooks.push_back({end, block, state_digest(e), kv_digest(e, end)});
    });
    e.prefill(head(prompt, kHookN));
    e.set_block_hook({});
    CHECK(hooks.size() == ref_hooks.size());
    for (size_t i = 0; i < hooks.size(); ++i) {
      if (!(hooks[i] == ref_hooks[i]))
        std::fprintf(stderr, "%s: hook %zu at %u: state %s, kv %s\n", what.c_str(), i, hooks[i].end,
                     differ(ref_hooks[i].state, hooks[i].state).c_str(),
                     differ(ref_hooks[i].kv, hooks[i].kv).c_str());
      CHECK(hooks[i] == ref_hooks[i]);
    }
    std::printf("%s, hooks: at 2048, 4096 (from the shadows) and 5000, each snapshot bitwise\n", what.c_str());
    // A throwing hook: pos 2048 and the 2048 state, as one card leaves them.
    e.reset();
    e.set_block_hook([](uint32_t end, bool) {
      if (end == 2048) throw std::runtime_error("hook");
    });
    bool threw = false;
    try {
      e.prefill(head(prompt, kHookN));
    } catch (const std::runtime_error& x) {
      threw = std::string(x.what()) == "hook";
    }
    e.set_block_hook({});
    CHECK(threw);
    CHECK_EQ(e.pos(), 2048u);
    CHECK(differ(state_digest(e), ref_throw_state).empty());
    CHECK(differ(kv_digest(e, 2048), ref_throw_kv).empty());
    std::printf("%s, a hook that throws at 2048: pos 2048, the state and KV one card leaves\n", what.c_str());

    // One card -> two: the one-card snapshot at 3000 into the pipeline, the rest there.
    e.reset();
    e.load_state(snap_state.data(), kRestoreAt);
    e.load_kv(0, kRestoreAt, snap_kv.data());
    e.prefill(Ids(prompt.begin() + kRestoreAt, prompt.begin() + kRestoreN));
    check_same(ref_restore, record(e), what + ", one card's 3000 -> two cards' 1103");
    // Two -> one (first configuration only: one more load of the whole model on device 0
    // needs the pipeline gone).
    if (cf.handoff == runtime::PpHandoff::Copy && cf.split == 0) {
      e.reset();
      e.prefill(head(prompt, kRestoreAt));
      std::vector<uint8_t> s2(e.state_bytes()), k2(e.kv_bytes(kRestoreAt));
      e.save_state(s2.data());
      e.save_kv(0, kRestoreAt, k2.data());
      CHECK(s2 == snap_state);
      CHECK(k2 == snap_kv);
      std::printf("%s: the 3000-id snapshot equals one card's byte for byte (so two -> one restores"
                  " the one-card state)\n", what.c_str());
    }
  }
  std::printf("pp_prefill_test: OK (%s, %s, %s KV)\n", desc ? desc->name.c_str() : "?",
              runtime::prefill_backend_name(backend), runtime::kv_cache_name(kv));
  return 0;
}

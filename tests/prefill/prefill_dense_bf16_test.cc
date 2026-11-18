// prefill_dense_bf16_test - open-work 38: a MoE model's prompt pass with its mixer linears held in bf16
// (Engine::set_prefill_dense_bf16, prefill::make_dense_slabs) is BITWISE the pass that unpacks them every chunk.
//
// One engine, the first 6000 ids of long32k at three chunk widths (the session's cap, 1024, 400 - all past
// pf_gemm_q4's 384 rows, so every mixer linear walks its slabs): first every case unpacking, then the slabs made
// and every case again from a reset. The persistent state (GDN states, conv rings, KV) and cur_token must be byte
// for byte the same; the second pass must launch fewer kernels (no dequant: the path was taken); and the prefill
// scratch must have grown by exactly runtime::dense_slab_bytes (what the memory plan counts).
//
// argv: [1] snapshot (the Ornith int4 checkpoint), [2] prompt dir; `--chunk-cap 4096` (tests/chunk_cap_arg.h).
// Exit 77 (SKIP) when the checkpoint is not on this machine.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "chunk_cap_arg.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "runtime/buffer_sizes.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {

constexpr uint32_t kMaxLen = 8192, kIds = 6000;
const uint32_t kChunks[] = {0 /* = the cap */, 1024, 400};

void same_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const std::string& what) {
  CHECK_EQ(a.size(), b.size());
  size_t diff = 0, first = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) {
      if (diff == 0) first = i;
      ++diff;
    }
  if (diff != 0) std::fprintf(stderr, "%s: %zu of %zu bytes differ, first at byte %zu\n", what.c_str(), diff, a.size(), first);
  CHECK_EQ(diff, size_t(0));
}

struct Run {
  std::vector<uint8_t> gdn_state, conv_ring, kv_k, kv_v;
  uint32_t token = 0;
  size_t launches = 0;
};

Run run(runtime::Engine& eng, l0::CmdList& imm, const std::vector<uint32_t>& ids, uint32_t chunk) {
  Run r;
  eng.reset();
  const size_t before = eng.prefill_launches();
  eng.prefill(ids, chunk);
  r.launches = eng.prefill_launches() - before;
  auto grab = [&](std::vector<uint8_t>& dst, const l0::Mem& m) {
    dst.resize(m.size());
    imm.copy(dst.data(), m.ptr(), m.size());
  };
  grab(r.gdn_state, eng.buffers().gdn_state);
  grab(r.conv_ring, eng.buffers().conv_ring);
  grab(r.kv_k, eng.buffers().kv_k);
  grab(r.kv_v, eng.buffers().kv_v);
  r.token = eng.buffers().control.as<runtime::Control>()->cur_token[0];
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  b70test::take_chunk_cap_arg(argc, argv);
  const std::string snap = argc > 1 ? argv[1] : "urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  try {
    (void)loader::resolve_snapshot(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no Ornith int4 checkpoint at '%s' (%s)\n", snap.c_str(), e.what());
    return 77;
  }
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  const size_t dense = runtime::dense_slab_bytes(*model.desc);
  CHECK(dense != 0);   // a MoE model
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<uint32_t> ids = golden::read_ids(pdir + "/long32k.ids");
  CHECK(ids.size() >= kIds);
  ids.resize(kIds);

  eng.prepare_prefill();
  std::vector<Run> unpacked;
  for (uint32_t chunk : kChunks) unpacked.push_back(run(eng, imm, ids, chunk));

  const size_t scratch0 = eng.memory_use().prefill_scratch;   // after the walk's own lazy buffers
  eng.set_prefill_dense_bf16(true);
  eng.prepare_prefill();
  CHECK_EQ(eng.memory_use().prefill_scratch - scratch0, dense);
  for (size_t i = 0; i < std::size(kChunks); ++i) {
    const uint32_t C = kChunks[i] ? kChunks[i] : eng.prefill_chunk_cap();
    const Run b = run(eng, imm, ids, kChunks[i]);
    const Run& a = unpacked[i];
    const std::string what = "C=" + std::to_string(C);
    same_bytes(a.gdn_state, b.gdn_state, what + " gdn_state");
    same_bytes(a.conv_ring, b.conv_ring, what + " conv_ring");
    same_bytes(a.kv_k, b.kv_k, what + " kv_k");
    same_bytes(a.kv_v, b.kv_v, what + " kv_v");
    CHECK_EQ(a.token, b.token);
    CHECK(b.launches < a.launches);
    std::printf("  C=%-5u %u ids: state and KV bitwise identical, cur_token %u; launches %zu -> %zu\n", C, kIds, a.token,
                a.launches, b.launches);
  }
  std::printf("prefill_dense_bf16_test OK: %zu chunk widths bitwise identical with the mixer linears in bf16 "
              "(%.3f GB, the plan's dense_slab_bytes)\n",
              std::size(kChunks), dense / 1e9);
  return 0;
}

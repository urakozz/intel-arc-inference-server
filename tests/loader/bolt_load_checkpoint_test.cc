// Spec 25b Task 5: load_bolt on the card (box only; label `checkpoint;bolt`), one device or two.
//   bolt_load_checkpoint_test <snapshot> [int8|bf16] [layers] [split] [max_len]
// Loads the checkpoint (one of Task 1's synthetic ones in either spelling, or our real g64 export, the served one,
// optionally truncated to its first `layers` blocks - development mode); `split` > 0: two devices, blocks [0, split) |
// [split, L). Prints the report, and asserts: zero unconsumed tensors, each part's weight bytes equal to
// loader::bolt::device_weight_bytes AND to runtime::bolt::plan's `weights` (the plan IS the allocation), and sampled
// blocks (the first and last of the model, a Mamba block, an attention block, a MoE block's first and last expert) read back
// from EACH device equal to the host repack, plus sampled elements of the embedding and the head. Exit 77 (SKIP) when the
// snapshot is not on this machine, or `split` asks for two devices on a one-GPU host.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/bolt_layout.h"
#include "loader/bolt_loader.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/bolt/bolt_sizes.h"

namespace {
std::vector<uint8_t> readback(l0::CmdList& imm, const l0::Mem& m, size_t off, size_t bytes) {
  std::vector<uint8_t> v(bytes);
  imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, bytes);
  return v;
}
bool same(const std::vector<uint8_t>& got, const void* want) { return std::memcmp(got.data(), want, got.size()) == 0; }
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ";
  const bool bf16_head = argc > 2 && std::string(argv[2]) == "bf16";
  const uint32_t layers = argc > 3 ? uint32_t(std::stoul(argv[3])) : 0;
  const uint32_t split = argc > 4 ? uint32_t(std::stoul(argv[4])) : 0;
  const uint32_t max_len = argc > 5 ? uint32_t(std::stoul(argv[5])) : 8192;
  std::string snap;
  try {
    snap = loader::resolve_snapshot(arg);
    if (!loader::is_bolt_checkpoint(snap)) throw std::runtime_error("not a nemotron_h_omni / nemotron_h_puzzle checkpoint");
  } catch (const std::exception& e) {
    std::printf("SKIP: no Nemotron 3.5 Bolt checkpoint at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  if (split != 0 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: split %u asks for two devices; this host has %u\n", split, l0::Context::gpu_count());
    return 77;
  }
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(snap, layers);
  l0::Context ctx0(0);
  std::unique_ptr<l0::Context> ctx1;
  std::vector<l0::Context*> devs = {&ctx0};
  if (split != 0) {
    ctx1 = std::make_unique<l0::Context>(ctx0, 1);
    devs.push_back(ctx1.get());
  }
  const model::BoltPlacement pl = split ? model::BoltPlacement::two(pre, split) : model::BoltPlacement::one(pre);
  loader::BoltLoadOptions o;
  o.int8_head = !bf16_head;
  o.max_len = max_len;
  o.layers = layers;
  loader::BoltLoaded m = loader::load_bolt(devs, snap, o, pl);
  const model::BoltDesc& d = m.desc;
  CHECK_EQ(d.layers, pre.layers);
  CHECK_EQ(m.unconsumed, size_t(0));
  CHECK_EQ(m.parts.size(), size_t(pl.devices));

  // the plan equals the allocation: runtime::bolt::plan at the same placement, int8 head and max_len
  std::vector<size_t> free_bytes;
  for (l0::Context* c : devs) free_bytes.push_back(c->memory_bytes());
  const runtime::bolt::BoltPlan plan = runtime::bolt::plan(
      d, o.max_len, free_bytes, {}, split ? std::optional<uint32_t>(split) : std::nullopt,
      size_t(runtime::kDefaultReserveGb * 1e9), o.int8_head);
  CHECK_EQ(plan.place.split, pl.split);
  for (uint32_t dev = 0; dev < pl.devices; ++dev) {
    const loader::BoltDevicePart& P = m.parts[dev];
    CHECK_EQ(P.layers.size(), size_t(pl.count(dev)));
    CHECK_EQ(P.bytes, loader::bolt::device_weight_bytes(d, pl, dev, o.int8_head));
    CHECK_EQ(P.bytes, plan.device[dev].weights);
    std::printf("  plan device %u: %s\n", dev, runtime::bolt::describe(plan.device[dev], free_bytes[dev]).c_str());
    CHECK_EQ(bool(P.embed_tiles), dev == 0);
    CHECK_EQ(bool(P.final_norm), dev + 1 == pl.devices);
    CHECK_EQ(bool(P.head), dev + 1 == pl.devices);
    if (P.head) CHECK(P.head->kind == (o.int8_head ? model::WeightKind::Int8 : model::WeightKind::Bf16));
  }

  // every device allocation base is a multiple of 64 B (the LSC 2D-load rule; spec 25b Global Constraints): asserted on the
  // card, not assumed from the layout arithmetic
  {
    size_t n_alloc = 0;
    auto aligned = [&](const l0::Mem* mem) {
      if (!mem) return;
      CHECK_EQ(reinterpret_cast<uintptr_t>(mem->ptr()) % loader::bolt::kAlign, uintptr_t(0));
      ++n_alloc;
    };
    auto dw = [&](const std::unique_ptr<loader::DeviceWeight>& w) {
      if (!w) return;
      aligned(&w->mem);
      aligned(w->scales.get());
    };
    for (const loader::BoltDevicePart& P : m.parts) {
      aligned(P.embed_tiles.get());
      aligned(P.final_norm.get());
      dw(P.head);
      for (const loader::BoltLayer& L : P.layers) {
        for (const auto* w : {&L.in_proj, &L.out_proj, &L.qkv, &L.o_proj, &L.fc1_shared_up, &L.fc2, &L.shared_down, &L.router}) dw(*w);
        aligned(L.experts.get());
        aligned(L.small.get());
      }
    }
    std::printf("  %zu device allocations, every base a multiple of %zu B\n", n_alloc, loader::bolt::kAlign);
  }

  loader::SafetensorsSet set(snap);
  const std::string trunk = m.trunk;
  std::mt19937 rng(11);
  // Sampled blocks on EACH device read back equal to the host repack (a fresh repack from the same shards).
  std::vector<uint32_t> sample = {0, d.layers - 1};
  for (uint32_t l = 0; l < d.layers; ++l)
    if (d.is_attention(l) && sample.size() < 4) sample.push_back(l);
  for (uint32_t l = 0; l < d.layers; ++l)
    if (d.is_moe(l)) {
      sample.push_back(l);
      break;
    }
  const model::BoltDesc full = loader::bolt_checkpoint_desc(snap, 0);   // the checkpoint's own block count
  loader::BoltCheckpoint ck(full, set, trunk);
  loader::BoltHostLayer host;
  size_t checked = 0;
  for (uint32_t l : sample) {
    const uint32_t dev = pl.device_of(l);
    const loader::BoltDevicePart& P = m.parts[dev];
    l0::CmdList imm = l0::CmdList::immediate(*devs[dev]);
    const loader::BoltLayer& L = P.layer(l);
    ck.repack_layer(l, host);
    auto linear = [&](const std::unique_ptr<loader::DeviceWeight>& w, const loader::BoltHostLinear& h) {
      CHECK(w);
      CHECK(same(readback(imm, w->mem, 0, h.words.size() * 4), h.words.data()));
      if (!h.scales.empty()) {   // layout 0: the separate f16 scales
        CHECK(w->scales);
        CHECK(same(readback(imm, *w->scales, 0, h.scales.size() * 2), h.scales.data()));
      }
      ++checked;
    };
    CHECK(same(readback(imm, *L.small, 0, host.small.size() * 4), host.small.data()));
    if (d.is_mamba(l)) {
      linear(L.in_proj, host.in_proj);
      linear(L.out_proj, host.out_proj);
    } else if (d.is_attention(l)) {
      linear(L.qkv, host.qkv);
      linear(L.o_proj, host.o_proj);
    } else {
      linear(L.fc1_shared_up, host.fc1_shared_up);
      linear(L.fc2, host.fc2);
      linear(L.shared_down, host.shared_down);
      CHECK(L.router);
      CHECK(same(readback(imm, L.router->mem, 0, host.router.size() * 2), host.router.data()));
      const uint32_t E = d.experts[l], I = d.inter[l];
      const size_t block = loader::bolt::expert_block_bytes(d.latent, I);
      for (uint32_t e : {0u, E / 2, E - 1}) {
        const size_t off = loader::bolt::expert_up_offset(d.latent, I, e);
        CHECK(same(readback(imm, *L.experts, off, block), reinterpret_cast<const uint8_t*>(host.experts.data()) + off));
        ++checked;
      }
    }
  }
  // sampled elements of the embedding (device 0) and the head (the last device), through gemv_bf16's / the int8 tiles' indices
  {
    l0::CmdList imm0 = l0::CmdList::immediate(*devs[0]);
    const uint16_t* emb = ck.embed();
    for (int s = 0; s < 64; ++s) {
      const uint32_t k = rng() % d.hidden, n = rng() % d.vocab;
      const std::vector<uint8_t> got =
          readback(imm0, *m.parts[0].embed_tiles, loader::bolt::embed_row_index(d.hidden, k, n) * 2, 2);
      CHECK(same(got, &emb[size_t(n) * d.hidden + k]));
    }
    const loader::BoltDevicePart& last = m.parts.back();
    l0::CmdList imml = l0::CmdList::immediate(*devs.back());
    const uint16_t* head = ck.lm_head();
    for (int s = 0; s < 64; ++s) {
      const uint32_t k = rng() % d.hidden, n = rng() % d.vocab;
      if (o.int8_head) {
        // the int8 tile at (k, n) times its row scale is within half a step of the bf16 weight
        const std::vector<uint8_t> q = readback(imml, last.head->mem, loader::int8_tiled_index(d.hidden, k, n), 1);
        const std::vector<uint8_t> sc = readback(imml, *last.head->scales, size_t(n) * 4, 4);
        float scale;
        std::memcpy(&scale, sc.data(), 4);
        const float w = common::bf16_to_f32(head[size_t(n) * d.hidden + k]);
        CHECK(std::fabs(float(int8_t(q[0])) * scale - w) <= 0.5f * scale + 1e-12f);
      } else {
        const std::vector<uint8_t> got =
            readback(imml, last.head->mem, loader::bolt::embed_row_index(d.hidden, k, n) * 2, 2);
        CHECK(same(got, &head[size_t(n) * d.hidden + k]));
      }
    }
  }
  std::printf("bolt_load_checkpoint_test OK: %s, %u blocks, split %u, group mask %u, %zu total B, %zu sampled blocks read back, "
              "%.1f s\n", snap.c_str(), d.layers, split, m.group_mask, m.total_bytes(), checked, m.seconds);
  return 0;
}

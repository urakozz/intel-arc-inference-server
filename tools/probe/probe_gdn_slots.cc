// Spec 19 v3d Task 3 Step 1: where the GDN step's excess at 8 rows goes - the row chain or the per-row snapshots.
// probe_gdn_slots [--calls 16] [--rounds 5] [--ref-dir DIR] [--variant NAME]...
// One list of 48 gdn_step launches (Qwen3.8's GDN layers, each at its own slice of a 48-layer state, ring and
// slots, from live slot 0) per arm, interleaved by round after a warm-up:
//   slots_M1   gdn_step_slots_M1_N8: the DFlash 1-row list's step (one state read, one write)
//   plain_M8   probe_gdn_plain_M8: the shipped source without SPEC_SLOTS at M = 8 - the same chain over 8 rows,
//              the state written once at the end (the compute's share)
//   slots_M8   gdn_step_slots_M8_N8: the verify list's step (8 snapshot writes)
// chain = plain_M8 - slots_M1, snapshots = slots_M8 - plain_M8. Inputs are random: the step's cost does not depend
// on the values. --ref-dir adds ref_M8: DIR's gdn_step_slots_M8_N8.bin (another build's kernel directory), the A/B
// of a gdn_step change in the same rounds. --variant adds an M = 8 slots arm from another binary of this build,
// compared with slots_M8 (Step 2's store forms - block writes, LSC cache controls - measured this way 2026-11-13:
// none saved more than 0.08 ms, the uncached ones cost 8-15 ms; removed).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"

namespace {
constexpr uint32_t kLayers = 48, kHeads = 48, kDim = 128, kChunks = 4, kQkvzN = 16384, kAbStride = 128;
constexpr uint32_t kRing = 16, kConvRows = 10240, kSlots = 8, kM = 8;
constexpr size_t kSmallFloats = 41056;   // conv [10240][4], negA [48] at 40960, dt_bias [48] at 41008
constexpr size_t kState = size_t(kHeads) * kDim * kDim;   // one layer
constexpr size_t kSlot = kLayers * kState;                // SPEC_SLOT_STRIDE

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  try {
    unsigned calls = 16, rounds = 5;
    std::string ref_dir;
    std::vector<std::string> variants;
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string a = argv[i];
      if (a == "--calls") calls = std::stoul(argv[i + 1]);
      else if (a == "--rounds") rounds = std::stoul(argv[i + 1]);
      else if (a == "--ref-dir") ref_dir = argv[i + 1];
      else if (a == "--variant") variants.push_back(argv[i + 1]);
      else throw std::runtime_error("unknown option " + a);
    }
    l0::Context ctx(0);
    l0::Queue q(ctx);
    l0::Fence fence(q);
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.5f);
    auto upload = [&](l0::Mem& m, size_t floats, float scale) {
      std::vector<float> v(floats);
      for (auto& x : v) x = nd(rng) * scale;
      imm.copy(m.ptr(), v.data(), floats * 4);
    };
    l0::Mem ctrl(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    l0::Mem qkvz(ctx, l0::MemKind::Device, size_t(kM) * kQkvzN * 4), ab(ctx, l0::MemKind::Device, size_t(kM) * kAbStride * 4);
    l0::Mem small(ctx, l0::MemKind::Device, kSmallFloats * 4);
    l0::Mem ring(ctx, l0::MemKind::Device, size_t(kLayers) * kRing * kConvRows * 2);
    l0::Mem state(ctx, l0::MemKind::Device, kSlot * 4), spec(ctx, l0::MemKind::Device, (kSlots - 1) * kSlot * 4);
    l0::Mem out(ctx, l0::MemKind::Device, size_t(kM) * kHeads * kDim * 4);
    upload(qkvz, size_t(kM) * kQkvzN, 1.f);
    upload(ab, size_t(kM) * kAbStride, 1.f);
    std::vector<float> sm(kSmallFloats);
    for (auto& x : sm) x = nd(rng) * 0.3f;
    for (uint32_t h = 0; h < kHeads; ++h) { sm[40960 + h] = -0.5f; sm[41008 + h] = 0.1f; }
    imm.copy(small.ptr(), sm.data(), sm.size() * 4);
    imm.fill(ring.ptr(), 0, ring.size());
    upload(state, kSlot, 0.5f);
    imm.fill(spec.ptr(), 0, spec.size());
    runtime::Control* c = ctrl.as<runtime::Control>();
    *c = runtime::Control{};
    c->pos = 4096;
    c->gdn_live = 0;

    struct Arm {
      std::string name, variant;
      uint32_t rows;
      bool slots;
    };
    std::vector<Arm> arms = {{"slots_M1", kernels::gdn_step_slots_variant(1, 48, 16, 48, 8), 1, true},
                                   {"plain_M8", "probe_gdn_plain_M8", 8, false},
                                   {"slots_M8", kernels::gdn_step_slots_variant(8, 48, 16, 48, 8), 8, true}};
    if (!ref_dir.empty()) arms.push_back({"ref_M8", ref_dir + "/gdn_step_slots_M8_N8.bin", 8, true});
    for (const auto& v : variants) arms.push_back({v, v, 8, true});
    std::vector<std::unique_ptr<l0::Module>> mods;
    std::vector<std::unique_ptr<l0::Kernel>> ks;
    std::vector<l0::CmdList> lists;
    for (const Arm& a : arms) {
      mods.push_back(std::make_unique<l0::Module>(ctx, a.name == "ref_M8" ? a.variant : kernels::path(a.variant)));
      l0::CmdList list = l0::CmdList::regular(ctx);
      for (uint32_t g = 0; g < kLayers; ++g) {
        ks.push_back(std::make_unique<l0::Kernel>(*mods.back(), "gdn_step"));
        l0::Kernel& k = *ks.back();
        k.group_size(256);
        k.arg_ptr(0, ctrl.ptr());
        k.arg_ptr(1, qkvz.ptr());
        k.arg_ptr(2, ab.ptr());
        k.arg_ptr(3, small.ptr());
        k.arg_ptr(4, ring.as<uint16_t>() + size_t(g) * kRing * kConvRows);
        k.arg_ptr(5, state.as<float>() + size_t(g) * kState);
        k.arg_ptr(6, out.ptr());
        if (a.slots) k.arg_ptr(7, spec.as<float>() + size_t(g) * kState);
        list.launch(k, kHeads, kChunks);
      }
      list.close();
      lists.push_back(std::move(list));
    }
    auto run = [&](size_t a) {
      c->n_active = arms[a].rows;
      const auto t0 = std::chrono::steady_clock::now();
      q.execute(lists[a], &fence);
      fence.wait();
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    std::vector<std::vector<double>> ms(arms.size());
    for (unsigned r = 0; r <= rounds; ++r)
      for (size_t i = 0; i < arms.size(); ++i) {
        const size_t a = r % 2 ? arms.size() - 1 - i : i;   // the order flips every round
        double sum = 0;
        for (unsigned n = 0; n < calls; ++n) sum += run(a);
        if (r) ms[a].push_back(sum / calls);
      }
    for (size_t a = 0; a < arms.size(); ++a)
      std::printf("GDN_SLOTS arm=%s ms=%.4f (48 launches, wall per list)\n", arms[a].name.c_str(), median(ms[a]));
    std::printf("GDN_SLOTS_SPLIT excess8=%.4f chain=%.4f snapshots=%.4f\n", median(ms[2]) - median(ms[0]),
                median(ms[1]) - median(ms[0]), median(ms[2]) - median(ms[1]));
    for (size_t a = 3; a < arms.size(); ++a)
      std::printf("GDN_SLOTS_AB slots_M8 - %s = %.4f ms\n", arms[a].name.c_str(), median(ms[2]) - median(ms[a]));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_gdn_slots: %s\n", e.what());
    return 2;
  }
}

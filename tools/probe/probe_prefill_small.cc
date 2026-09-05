// T6 -- direct M=2048 device-timestamp measurements of the small kernels
// between the prefill path's linears.
//
// Two batteries in ONE run, so the comparison between them is free of any
// session-to-session drift:
//
//   1. **decode-S**: the production `src/kernels/prep.cl` source compiled at
//      M = 2048, folding decode's split-K (`prep_silu_mul` at SILU_S = 8,
//      `prep_res_fold` at SP4). These are probe-only binaries; no runtime
//      variant helper names them. This is the battery whose 360.888-473.221 ms
//      per chunk T6 first composed -- and it was the wrong battery, because the
//      prefill path runs S = 1 (plan 6b ruling R1) and those two rows therefore
//      carry 8x and 4x the traffic prefill will generate.
//   2. **prefill S=1**: the PRODUCTION prefill binaries from
//      src/kernels/prefill/, at runtime `M` -- the same objects
//      tests/prefill/pf_*_test.cc grade bit-for-bit against decode at M = 1.
//      No M appears in their names or on their command lines; `m_count` is a
//      kernel argument.
//
// Every row prints the bytes it moves and the bandwidth that implies, because
// the roofline comparison is the point: these kernels carry ~0% of prefill's
// FLOPs, so a row far under the device's measured 590 GB/s (docs/01) is a row
// with something left in it, and a row near it is finished.
//
// Timing is the Level Zero kernel timestamp median of eight replays after
// dropping the first three; inputs are finite, incompressible random data
// (B70's lossless compression would otherwise time a different workload).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {

constexpr uint32_t kM = 2048;          // PrefillScratch::kC (ruling A13)
constexpr uint32_t kReplays = 8;
constexpr uint32_t kDropped = 3;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kIntermediate = 17408;
constexpr uint32_t kGateUp = 34816;
constexpr uint32_t kGatedOut = 6144;
constexpr uint32_t kQkvN = 14336;      // attn_prep's q||gate||k||v row
constexpr uint32_t kQHeads = 24, kKvHeads = 4, kHeadDim = 256;
constexpr uint32_t kAbN = 128;         // a||b, zero-padded from 96
constexpr uint32_t kFoldGroups = 20;
constexpr uint32_t kVocab = 248320;
constexpr uint32_t kCtrlWords = 32;
constexpr double kDeviceBw = 590e9;    // docs/01, measured in situ

uint32_t xorshift(uint32_t& value) {
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

// Incompressible, finite values avoid B70's lossless-compression timing trap.
void fill_f32(l0::CmdList& upload, void* dst, size_t bytes, uint32_t seed) {
  constexpr size_t kChunkWords = (16u << 20) / sizeof(uint32_t);
  std::vector<uint32_t> host(kChunkWords);
  for (size_t offset = 0; offset < bytes;) {
    for (uint32_t& word : host) word = (xorshift(seed) & 0x807fffffu) | 0x3e000000u;
    const size_t n = std::min(bytes - offset, host.size() * sizeof(uint32_t));
    upload.copy(static_cast<std::byte*>(dst) + offset, host.data(), n);
    offset += n;
  }
}

void fill_bf16(l0::CmdList& upload, void* dst, size_t bytes, uint32_t seed) {
  constexpr size_t kChunkWords = (16u << 20) / sizeof(uint16_t);
  std::vector<uint16_t> host(kChunkWords);
  for (size_t offset = 0; offset < bytes;) {
    for (uint16_t& word : host) word = uint16_t((xorshift(seed) & 0x807fu) | 0x3e00u);
    const size_t n = std::min(bytes - offset, host.size() * sizeof(uint16_t));
    upload.copy(static_cast<std::byte*>(dst) + offset, host.data(), n);
    offset += n;
  }
}

enum class Kind { ResFold, NormFinish, Silu, GatedHead, Embed, AttnPrep, AbProj };

struct Case {
  const char* label;      // what the row is called in the table
  std::string binary;
  const char* entry;
  Kind kind;
  bool runtime_m;         // true: the last argument is `m_count`
  uint32_t wg;
  uint32_t gx, gy;
  uint32_t calls_per_chunk;
  double bytes;           // unique bytes moved per launch (derived; see the doc)
};

struct Buffers {
  l0::Mem &partials, &resid, &sumsq, &norm_w, &x, &gdn_o, &gated_w;
  l0::Mem &ids, &embed, &ctrl, &fa_small, &rope, &attn_q, &attn_gate, &kv_k, &kv_v;
  l0::Mem &ab_w, &ab_out;
};

double time_case(l0::Context& ctx, l0::Queue& queue, l0::Fence& fence, l0::Event& event,
                 const Case& c, Buffers& b) {
  l0::Module module(ctx, kernels::path(c.binary));
  l0::Kernel kernel = module.kernel(c.entry);
  kernel.group_size(c.wg);
  uint32_t next = 0;
  auto ptr = [&](l0::Mem& m) { kernel.arg_ptr(next++, m.ptr()); };
  switch (c.kind) {
    case Kind::ResFold:    ptr(b.partials); ptr(b.resid); ptr(b.sumsq); break;
    case Kind::NormFinish: ptr(b.sumsq); ptr(b.resid); ptr(b.norm_w); ptr(b.x); break;
    case Kind::Silu:       ptr(b.partials); ptr(b.x); break;
    case Kind::GatedHead:  ptr(b.partials); ptr(b.gdn_o); ptr(b.gated_w); ptr(b.x); break;
    case Kind::Embed:      ptr(b.ids); ptr(b.embed); ptr(b.resid); break;
    case Kind::AttnPrep:
      ptr(b.ctrl); ptr(b.partials); ptr(b.fa_small); ptr(b.rope);
      ptr(b.attn_q); ptr(b.attn_gate); ptr(b.kv_k); ptr(b.kv_v);
      break;
    // `resid` IS the a||b activation rectangle: bf16 [M][5120], the stride
    // ruling R2 says the producer writes and this consumer reads.
    case Kind::AbProj:     ptr(b.ab_w); ptr(b.resid); ptr(b.ab_out); break;
  }
  if (c.runtime_m) kernel.arg<uint32_t>(next, kM);

  l0::CmdList list = l0::CmdList::regular(ctx);
  list.launch(kernel, c.gx, c.gy, 1, &event);
  list.close();
  std::array<double, kReplays> samples{};
  for (double& sample : samples) {
    event.reset();
    queue.execute(list, &fence);
    fence.wait();
    sample = event.duration_us();
  }
  std::sort(samples.begin() + kDropped, samples.end());
  return samples[kDropped + (kReplays - kDropped) / 2];
}

void run_battery(const char* title, const char* note, const Case* cases, size_t n,
                 l0::Context& ctx, l0::Queue& queue, l0::Fence& fence, l0::Event& event,
                 Buffers& b) {
  std::printf("\n## %s\n%s\n\n", title, note);
  std::printf("| kernel | grid work-groups | MB/launch | us/launch | GB/s | calls/chunk | ms/chunk |\n");
  std::printf("|---|---:|---:|---:|---:|---:|---:|\n");
  double total = 0;
  for (size_t i = 0; i < n; ++i) {
    const Case& c = cases[i];
    const double us = time_case(ctx, queue, fence, event, c, b);
    const double ms = us * c.calls_per_chunk / 1000.0;
    total += ms;
    std::printf("| `%s` | %u x %u = %u | %.2f | %.3f | %.1f | %u | %.3f |\n", c.label, c.gx, c.gy,
                c.gx * c.gy, c.bytes / 1e6, us, c.bytes / (us * 1e-6) / 1e9, c.calls_per_chunk, ms);
  }
  std::printf("| **total** | | | | | | **%.3f** |\n", total);
}

}  // namespace

int main() {
  try {
    l0::Context ctx(0);
    l0::Queue queue(ctx);
    l0::Fence fence(queue);
    l0::CmdList upload = l0::CmdList::immediate(ctx);

    // The largest reader is gate||up's EIGHT fp32 split-K planes -- decode's
    // rectangle, kept so the decode-S battery is measured on the real thing.
    // Every S = 1 kernel reads the first plane of the same buffer, which is
    // exactly the [C][N] fp32 rectangle L2's GEMM will emit (ruling R1).
    l0::Mem partials(ctx, l0::MemKind::Device, size_t(8) * kM * kGateUp * sizeof(float));
    l0::Mem resid(ctx, l0::MemKind::Device, size_t(kM) * kHidden * sizeof(uint16_t));
    l0::Mem sumsq(ctx, l0::MemKind::Device, size_t(kFoldGroups) * kM * sizeof(float));
    l0::Mem norm_w(ctx, l0::MemKind::Device, size_t(kHidden) * sizeof(float));
    l0::Mem x(ctx, l0::MemKind::Device, size_t(kM) * kIntermediate * sizeof(uint16_t));
    l0::Mem gdn_o(ctx, l0::MemKind::Device, size_t(kM) * kGatedOut * sizeof(float));
    l0::Mem gated_w(ctx, l0::MemKind::Device, size_t(kGatedOut) * sizeof(uint16_t));
    // The embedding table is the REAL [248320][5120] bf16 one (2.54 GB): the
    // gather's cost is a function of how far apart the rows it pulls are, so a
    // stand-in table would measure a different kernel.
    l0::Mem embed(ctx, l0::MemKind::Device, size_t(kVocab) * kHidden * sizeof(uint16_t));
    l0::Mem ids(ctx, l0::MemKind::Device, size_t(kM) * sizeof(uint32_t));
    l0::Mem ctrl(ctx, l0::MemKind::Shared, kCtrlWords * sizeof(uint32_t));
    l0::Mem fa_small(ctx, l0::MemKind::Device, size_t(512) * sizeof(float));
    l0::Mem rope(ctx, l0::MemKind::Device, size_t(kM) * 64 * sizeof(float));
    l0::Mem attn_q(ctx, l0::MemKind::Device, size_t(kM) * kQHeads * kHeadDim * sizeof(float));
    l0::Mem attn_gate(ctx, l0::MemKind::Device, size_t(kM) * kQHeads * kHeadDim * sizeof(float));
    l0::Mem kv_k(ctx, l0::MemKind::Device, size_t(kM) * kKvHeads * kHeadDim * sizeof(uint16_t));
    l0::Mem kv_v(ctx, l0::MemKind::Device, size_t(kM) * kKvHeads * kHeadDim * sizeof(uint16_t));
    l0::Mem ab_w(ctx, l0::MemKind::Device, size_t(kAbN) * kHidden * sizeof(uint16_t));
    l0::Mem ab_out(ctx, l0::MemKind::Device, size_t(kM) * kAbN * sizeof(float));

    fill_f32(upload, partials.ptr(), partials.size(), 0x9e3779b9u);
    fill_bf16(upload, resid.ptr(), resid.size(), 0x85ebca6bu);
    fill_f32(upload, norm_w.ptr(), norm_w.size(), 0xc2b2ae35u);
    fill_f32(upload, gdn_o.ptr(), gdn_o.size(), 0x27d4eb2fu);
    fill_bf16(upload, gated_w.ptr(), gated_w.size(), 0x165667b1u);
    fill_bf16(upload, embed.ptr(), embed.size(), 0x9e3779b1u);
    fill_f32(upload, fa_small.ptr(), fa_small.size(), 0x2545f491u);
    fill_f32(upload, rope.ptr(), rope.size(), 0x9e3779bbu);
    fill_bf16(upload, ab_w.ptr(), ab_w.size(), 0x94d049bbu);
    {
      // A chunk's ids are arbitrary tokens: rows scattered over the table, not
      // a run. That is the gather pattern prefill actually issues.
      std::vector<uint32_t> host(kM);
      uint32_t seed = 0x1b873593u;
      for (uint32_t& id : host) id = xorshift(seed) % kVocab;
      upload.copy(ids.ptr(), host.data(), host.size() * sizeof(uint32_t));
      uint32_t* c = ctrl.as<uint32_t>();
      for (uint32_t i = 0; i < kCtrlWords; ++i) c[i] = 0;
      c[kernels::ctrl_index::kPos] = 0;
      c[kernels::ctrl_index::kNActive] = kM;
    }

    Buffers b{partials, resid, sumsq, norm_w, x, gdn_o, gated_w,
              ids, embed, ctrl, fa_small, rope, attn_q, attn_gate, kv_k, kv_v, ab_w, ab_out};

    // Bytes per launch: unique bytes moved, derived from the model's shapes.
    // A value re-read inside one launch is counted once, and `norm_w`, `sumsq`,
    // `gated_w`, `fa_small` and the RoPE table are cache-resident after the
    // first work-group touches them.
    const double b_res_fold_s8 = 4.0 * kM * kHidden * 4 + 2.0 * kM * kHidden * 2 + kFoldGroups * kM * 4.0;
    const double b_res_fold_s1 = 1.0 * kM * kHidden * 4 + 2.0 * kM * kHidden * 2 + kFoldGroups * kM * 4.0;
    const double b_norm_finish = 2.0 * kM * kHidden * 2 + kFoldGroups * kM * 4.0 + kHidden * 4.0;
    const double b_silu_s8 = 8.0 * kM * kGateUp * 4 + double(kM) * kIntermediate * 2;
    const double b_silu_s1 = 1.0 * kM * kGateUp * 4 + double(kM) * kIntermediate * 2;
    const double b_gated = 2.0 * kM * kGatedOut * 4 + double(kM) * kGatedOut * 2;
    const double b_embed = 2.0 * kM * kHidden * 2;
    const double b_attn_prep = double(kM) * kQkvN * 4 + 2.0 * kM * kQHeads * kHeadDim * 4 +
                               2.0 * kM * kKvHeads * kHeadDim * 2 + double(kM) * 64 * 4;
    const double b_ab = double(kM) * kHidden * 2 + double(kAbN) * kHidden * 2 +
                        double(kM) * kAbN * 4;

    // calls/chunk: 64 layers (48 GDN + 16 full attention). 129 norms = 2 per
    // layer + the final one; 64 MLPs; 48 GDN gated heads and a||b projections;
    // 16 attention preps; one embedding gather.
    const Case decode_cases[] = {
        {"prep_res_fold (SP4)", "pfs_res_fold_sp4_M2048", "prep_res_fold", Kind::ResFold, false,
         256, kFoldGroups, kM, 129, b_res_fold_s8},
        {"prep_norm_finish", "pfs_norm_finish_M2048", "prep_norm_finish", Kind::NormFinish, false,
         256, kFoldGroups, kM, 129, b_norm_finish},
        {"prep_silu_mul (S8)", "pfs_silu_mul_M2048", "prep_silu_mul", Kind::Silu, false,
         256, 5, kM, 64, b_silu_s8},
        {"prep_gated_head", "pfs_gated_head_M2048", "prep_gated_head", Kind::GatedHead, false,
         128, 48, kM, 48, b_gated},
    };
    const Case prefill_cases[] = {
        {"pf_silu_mul", kernels::pf_silu_mul_variant(), "pf_silu_mul", Kind::Silu, true,
         256, 5, kM, 64, b_silu_s1},
        {"pf_res_fold (SP1)", kernels::pf_res_fold_variant(kHidden, 1, kFoldGroups),
         "pf_res_fold", Kind::ResFold, true, 256, kFoldGroups, kM, 129, b_res_fold_s1},
        {"pf_norm_finish", kernels::pf_norm_finish_variant(kHidden, kFoldGroups, kFoldGroups),
         "pf_norm_finish", Kind::NormFinish, true, 256, kFoldGroups, kM, 129, b_norm_finish},
        {"pf_gated_head", kernels::pf_gated_head_variant(), "pf_gated_head", Kind::GatedHead, true,
         128, 48, kM, 48, b_gated},
        {"pf_attn_prep", kernels::pf_attn_prep_variant(), "pf_attn_prep", Kind::AttnPrep, false,
         256, kQHeads + kKvHeads, kM, 16, b_attn_prep},
        {"pf_ab_proj", kernels::pf_ab_proj_variant(), "pf_ab_proj", Kind::AbProj, true,
         256, kAbN / 16, (kM + 7) / 8, 48, b_ab},
        {"pf_embed_gather", kernels::pf_embed_gather_variant(), "pf_embed_gather", Kind::Embed,
         true, 256, 1, kM, 1, b_embed},
    };

    l0::EventPool pool(ctx, 1);
    l0::Event event(pool, 0);
    std::printf("# T6 small kernels at M=%u: device timestamps, %u replays, first %u dropped; "
                "inputs are finite incompressible random data.\n", kM, kReplays, kDropped);
    std::printf("# GB/s is the row's own bytes over its own time; the device measures "
                "%.0f GB/s (docs/01-hardware.md).\n", kDeviceBw / 1e9);
    run_battery("decode-S battery (the wrong one, kept as the control)",
                "prep.cl's production source at M=2048 and DECODE's split-K: prep_silu_mul folds "
                "SILU_S=8, prep_res_fold SP4. Probe-only binaries; no runtime helper names them.",
                decode_cases, sizeof(decode_cases) / sizeof(Case), ctx, queue, fence, event, b);
    run_battery("prefill S=1 battery (the production binaries)",
                "src/kernels/prefill/'s runtime-M binaries -- the objects "
                "tests/prefill/pf_*_test.cc grade bit-for-bit against decode at M=1. No M in any "
                "name or on any command line; m_count is a kernel argument.",
                prefill_cases, sizeof(prefill_cases) / sizeof(Case), ctx, queue, fence, event, b);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "probe_prefill_small FAILED: %s\n", error.what());
    return 1;
  }
}

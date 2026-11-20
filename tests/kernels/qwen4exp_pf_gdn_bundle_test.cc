// PERF-20 / PERF-21: Qwen3.8-Flash-Next's GDN walk on Qwen3.8's one-card bitwise forms (runtime/prefill/gdn.cc
// dense_bundle) against the plain forms it replaced, on the card with synthetic data (no checkpoint):
//   new  gdn_chunk_q4 - pf_gdn_front _PL (a lane's partials read first), pf_gdn_scan _OB (o as the bf16 word the head
//        rounds it to, q.S beside w.S, two barriers a step), pf_gated_head _SIG_OB (one sub-group a (position, head)
//        on that bf16 o)
//   old  main's walk before them, launch for launch: pf_gdn_front, pf_gdn_scan, pf_gated_head _SIG (fp32 o)
// at one card's 16 / 48 heads and a tensor-parallel card's 8 / 24 (_GK8V24). Each walk runs 2048 positions in one
// call, then a 300-row continuation (a non-zero state, the ring's seed, a partial sub-chunk): the state, the conv ring
// and every row of y must be BITWISE old == new, and the new walk in 64-row calls BITWISE the new walk in one.
// PERF-8: the new walk on the qkv||z partials as bf16 words (qkvz_bf16: the front / head _PB, here the host's rne of
// the fp32 partials - pf_gemm_T0_B16's words, qwen4exp_prefill_test gdn-pb holds the GEMM to them) BITWISE the old walk
// on the fp32 partials too. The builds' spill sizes are printed (spill 0 required of the new front and head; the _OB
// scan spills 512 B at both head counts - at 16 / 48 it is Qwen3.8's one-card binary, which spills so on main - and is
// faster). `--time` adds each replaced kernel's time old against new at C 2048 (interleaved pairs after a warm-up, the
// median of 15) and the whole walk's (fp32 and _PB).
// Needs a B70; exit 77 (SKIP) without a GPU.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill/kernels.h"
#include "runtime/qwen4exp/qwen4exp_prefill_gdn.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {

namespace rq = runtime::qwen4exp;
using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;
constexpr uint32_t kHd = 128, kP = 2048, kTail = 300, kRows = kP + kTail, kConvWg = 256;

// main's gdn_chunk_q4 before PERF-20 / PERF-21, statement for statement (the a||b rows at pitch 128).
void old_walk(runtime::prefill::Context& cx, runtime::prefill::KernelCache& kc, const rq::GdnChunkBuffers& b,
              uint32_t pos, uint32_t C) {
  const uint32_t K = b.k_heads, V = b.v_heads, conv_groups = (2 * K + V) * kHd / kConvWg;
  const uint32_t nch = (C + rq::kGdnChunk - 1) / rq::kGdnChunk;
  const void* gated_w = static_cast<const uint8_t*>(b.small) + b.gated_norm_off;
  const std::string conv = kernels::pf_gdn_conv_variant(K, V);
  cx.launch(kc(conv, "pf_gdn_seed"), conv_groups, 1, 1, {PtrArg(b.ring), PtrArg(b.seed), arg_val(pos)});
  cx.launch(kc(kernels::pf_gdn_front_variant(K, V), "pf_gdn_front"), conv_groups, (C + 31) / 32, 1,
            {PtrArg(b.qkvz), PtrArg(b.seed), PtrArg(b.small), PtrArg(b.xb), PtrArg(b.ring), arg_val(pos), arg_val(C)});
  cx.launch(kc(conv, "pf_gdn_gate"), V, nch, 1, {PtrArg(b.ab), PtrArg(b.small), PtrArg(b.g), PtrArg(b.beta), arg_val(C)});
  cx.launch(kc(kernels::pf_gdn_intra_variant(K, V), "pf_gdn_intra"), V, nch, 1,
            {PtrArg(b.xb), PtrArg(b.g), PtrArg(b.beta), PtrArg(b.w), PtrArg(b.u), PtrArg(b.A2), arg_val(C)});
  cx.launch(kc(kernels::pf_gdn_scan_variant(K, V), runtime::prefill::gdn_scan_entry_name()), V, 4, 1,
            {PtrArg(b.xb), PtrArg(b.w), PtrArg(b.u), PtrArg(b.A2), PtrArg(b.g), PtrArg(b.state), PtrArg(b.o), arg_val(C)});
  cx.launch(kc(kernels::pf_gated_head_variant(K, V) + "_SIG", "pf_gated_head"), V, C, 1,
            {PtrArg(b.qkvz), PtrArg(b.o), PtrArg(gated_w), PtrArg(b.y), arg_val(C)});
}

uint32_t spill(l0::Kernel& k) {
  ze_kernel_properties_t pr{};
  pr.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  zeKernelGetProperties(k.handle(), &pr);
  return pr.spillMemSize;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class T>
size_t differ(const std::vector<T>& a, const std::vector<T>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

// One shape's walks: the state, the ring and y (all kRows rows) a walk leaves.
struct Run {
  std::vector<float> state;
  std::vector<uint16_t> ring, y;
};

bool shape(l0::Context& ctx, runtime::prefill::Context& cx, runtime::prefill::KernelCache& kc, uint32_t K, uint32_t V,
           bool timing) {
  const uint32_t conv = (2 * K + V) * kHd, qkvz_n = conv + V * kHd, mix = V * kHd;
  const size_t nega = size_t(conv) * 4, gated_off = (nega + 2 * V) * 4, small_bytes = gated_off + kHd * 2;
  const size_t state_bytes = size_t(V) * kHd * kHd * 4, ring_bytes = size_t(rq::kConvRing) * conv * 2;
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  // gdn_chunk_test's long-memory fixture (negA in [-4, -1], dt_bias -4) at these heads
  std::mt19937 rng(0x21d + V);
  std::normal_distribution<float> q(0.0f, 0.5f), a(0.0f, 1.0f), cw(0.0f, 0.5f);
  std::uniform_real_distribution<float> negA(-4.0f, -1.0f), gw(0.5f, 1.5f);
  std::vector<float> qkvz(size_t(kRows) * qkvz_n), ab(size_t(kRows) * 128, 0.0f), small(small_bytes / 4, 0.0f);
  for (float& v : qkvz) v = q(rng);
  for (uint32_t m = 0; m < kRows; ++m)
    for (uint32_t h = 0; h < 2 * V; ++h) ab[size_t(m) * 128 + h] = a(rng);
  for (size_t i = 0; i < nega; ++i) small[i] = cw(rng);
  for (uint32_t h = 0; h < V; ++h) {
    small[nega + h] = negA(rng);
    small[nega + V + h] = -4.0f;
  }
  uint16_t* gwp = reinterpret_cast<uint16_t*>(small.data()) + gated_off / 2;
  for (uint32_t i = 0; i < kHd; ++i) gwp[i] = common::f32_to_bf16(gw(rng));
  const auto up = [&](const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.copy(m.ptr(), p, bytes);
    return m;
  };
  std::vector<uint16_t> qkvz16(qkvz.size());
  for (size_t i = 0; i < qkvz.size(); ++i) qkvz16[i] = common::f32_to_bf16(qkvz[i]);
  const l0::Mem dq16 = up(qkvz16.data(), qkvz16.size() * 2);
  const l0::Mem dq = up(qkvz.data(), qkvz.size() * 4), da = up(ab.data(), ab.size() * 4), ds = up(small.data(), small_bytes);
  const rq::GdnScratchSizes gs = rq::gdn_scratch_sizes(rq::kPfC);
  // per walk (old / new): its state, ring, y and scratch, so the timed kernels read their own walk's o
  struct Bufs {
    l0::Mem st, rg, y, scratch;
    rq::GdnChunkBuffers b;
  };
  const auto make = [&]() {
    Bufs w{l0::Mem(ctx, l0::MemKind::Device, state_bytes), l0::Mem(ctx, l0::MemKind::Device, ring_bytes),
           l0::Mem(ctx, l0::MemKind::Device, size_t(kRows) * mix * 2), l0::Mem(ctx, l0::MemKind::Device, gs.total()), {}};
    rq::GdnChunkBuffers& b = w.b;
    b.k_heads = K;
    b.v_heads = V;
    b.state = w.st.as<float>();
    b.ring = w.rg.as<uint16_t>();
    b.small = ds.ptr();
    b.gated_norm_off = gated_off;
    uint8_t* base = static_cast<uint8_t*>(w.scratch.ptr());
    size_t off = 0;
    const std::pair<void**, size_t> parts[] = {{&b.xb, gs.xb}, {&b.seed, gs.seed}, {&b.g, gs.g}, {&b.beta, gs.beta},
                                               {&b.A, gs.A}, {&b.A2, gs.A2}, {&b.w, gs.w}, {&b.u, gs.u}, {&b.o, gs.o}};
    for (auto [p, n] : parts) {
      *p = base + off;
      off += n;
    }
    return w;
  };
  Bufs wo = make(), wn = make();
  using Walk = std::function<void(const rq::GdnChunkBuffers&, uint32_t, uint32_t)>;
  const Walk old_fn = [&](const rq::GdnChunkBuffers& b, uint32_t pos, uint32_t C) { old_walk(cx, kc, b, pos, C); };
  const Walk new_fn = [&](const rq::GdnChunkBuffers& b, uint32_t pos, uint32_t C) {
    cx.reset_launches();
    rq::gdn_chunk_q4(cx, kc, b, pos, C);
    CHECK_EQ(cx.launches(), rq::gdn_chunk_q4_launches());
  };
  // positions [0, kP) in calls of `width`, then the kTail rows in one; `pb`: the partials as bf16 words
  const auto walk = [&](Bufs& w, const Walk& fn, uint32_t width, bool pb = false) {
    imm.fill(w.st.ptr(), 0u, state_bytes);
    imm.fill(w.rg.ptr(), 0u, ring_bytes);
    imm.fill(w.y.ptr(), 0u, w.y.size());
    std::vector<std::pair<uint32_t, uint32_t>> calls;
    for (uint32_t p = 0; p < kP; p += width) calls.push_back({p, width});
    calls.push_back({kP, kTail});
    for (auto [p, C] : calls) {
      rq::GdnChunkBuffers b = w.b;
      b.qkvz = pb ? static_cast<const void*>(dq16.as<uint16_t>() + size_t(p) * qkvz_n)
                  : static_cast<const void*>(dq.as<float>() + size_t(p) * qkvz_n);
      b.qkvz_bf16 = pb;
      b.ab = da.as<float>() + size_t(p) * 128;
      b.y = w.y.as<uint16_t>() + size_t(p) * mix;
      fn(b, p, C);
      cx.wait();
    }
    Run r{std::vector<float>(state_bytes / 4), std::vector<uint16_t>(ring_bytes / 2), std::vector<uint16_t>(size_t(kRows) * mix)};
    imm.copy(r.state.data(), w.st.ptr(), state_bytes);
    imm.copy(r.ring.data(), w.rg.ptr(), ring_bytes);
    imm.copy(r.y.data(), w.y.ptr(), r.y.size() * 2);
    return r;
  };
  const Run old_one = walk(wo, old_fn, kP), new_64 = walk(wn, new_fn, rq::kGdnChunk), new_one = walk(wn, new_fn, kP);
  const size_t ds_ = differ(old_one.state, new_one.state), dr = differ(old_one.ring, new_one.ring),
               dy = differ(old_one.y, new_one.y);
  const bool split = new_one.state == new_64.state && new_one.ring == new_64.ring && new_one.y == new_64.y;
  const Run new_pb = walk(wn, new_fn, kP, true);
  const bool pb_bit = old_one.state == new_pb.state && old_one.ring == new_pb.ring && old_one.y == new_pb.y;
  std::printf("GDN %u / %u heads, %u + %u rows: old against new - state %zu, ring %zu, y %zu words differ (%s); new in "
              "64-row calls against one call: %s\n",
              K, V, kP, kTail, ds_, dr, dy, ds_ + dr + dy == 0 ? "bitwise" : "DIFFER", split ? "bitwise" : "DIFFER");
  std::printf("  the new walk on bf16 partials (_PB) against the old on fp32: %s\n", pb_bit ? "bitwise" : "DIFFER");
  // the builds' spills
  const char* const scan_entry = runtime::prefill::gdn_scan_entry_name();
  struct Pair {
    const char* what;
    l0::Kernel &old_k, &new_k;
  };
  const Pair pairs[] = {
      {"pf_gdn_front", kc(kernels::pf_gdn_front_variant(K, V), "pf_gdn_front"),
       kc(kernels::pf_gdn_front_variant(K, V) + "_PL", "pf_gdn_front")},
      {"pf_gdn_scan", kc(kernels::pf_gdn_scan_variant(K, V), scan_entry),
       kc(kernels::pf_gdn_scan_variant(K, V) + "_OB", scan_entry)},
      {"pf_gated_head _SIG", kc(kernels::pf_gated_head_variant(K, V) + "_SIG", "pf_gated_head"),
       kc(kernels::pf_gated_head_variant(K, V) + "_SIG_OB", "pf_gated_head_sg")},
  };
  bool spill_free = true;
  for (const Pair& p : pairs) {
    std::printf("  spill %-20s old %u B, new %u B\n", p.what, spill(p.old_k), spill(p.new_k));
    spill_free = spill_free && (&p == &pairs[1] || spill(p.new_k) == 0);   // the _OB scan: printed only
  }
  if (timing) {
    // each walk at pos 0, C kP once (its scratch holds this call's xb / g / w / u / A2 / o), the state it started
    // from kept to reset the scan before each timed run
    l0::Mem s0(ctx, l0::MemKind::Device, state_bytes), st_stamp(ctx, l0::MemKind::Host, 64);
    imm.fill(s0.ptr(), 0u, state_bytes);
    for (Bufs* w : {&wo, &wn}) {
      w->b.qkvz = dq.as<float>();
      w->b.ab = da.as<float>();
      w->b.y = w->y.as<uint16_t>();
      imm.fill(w->st.ptr(), 0u, state_bytes);
    }
    old_fn(wo.b, 0, kP);
    new_fn(wn.b, 0, kP);
    cx.wait();
    uint64_t* stamp = st_stamp.as<uint64_t>();
    const auto time_pair = [&](const char* what, const std::function<void()>& o, const std::function<void()>& n) {
      const auto timed = [&](bool is_new) {
        cx.copy(wo.st.ptr(), s0.ptr(), state_bytes);
        cx.copy(wn.st.ptr(), s0.ptr(), state_bytes);
        cx.timestamp(stamp);
        if (is_new) n(); else o();
        cx.timestamp(stamp + 1);
        cx.wait();
        return cx.timestamp_ms(stamp[0], stamp[1]);
      };
      for (int r = 0; r < 3; ++r) { timed(false); timed(true); }
      std::vector<double> to, tn;
      for (int r = 0; r < 15; ++r) {
        if (r % 2) { tn.push_back(timed(true)); to.push_back(timed(false)); }
        else { to.push_back(timed(false)); tn.push_back(timed(true)); }
      }
      std::printf("  time %-20s C %u: old %.4f ms, new %.4f ms (x%.3f)\n", what, kP, median(to), median(tn),
                  median(tn) / median(to));
    };
    const uint32_t conv_groups = conv / kConvWg;
    const auto launch_front = [&](l0::Kernel& k, const rq::GdnChunkBuffers& b) {
      cx.launch(k, conv_groups, (kP + 31) / 32, 1,
                {PtrArg(b.qkvz), PtrArg(b.seed), PtrArg(b.small), PtrArg(b.xb), PtrArg(b.ring), arg_val(0u), arg_val(kP)});
    };
    const auto launch_scan = [&](l0::Kernel& k, const rq::GdnChunkBuffers& b) {
      cx.launch(k, V, 4, 1,
                {PtrArg(b.xb), PtrArg(b.w), PtrArg(b.u), PtrArg(b.A2), PtrArg(b.g), PtrArg(b.state), PtrArg(b.o), arg_val(kP)});
    };
    const void* gated_w = static_cast<const uint8_t*>(ds.ptr()) + gated_off;
    time_pair("pf_gdn_front", [&] { launch_front(pairs[0].old_k, wo.b); }, [&] { launch_front(pairs[0].new_k, wn.b); });
    time_pair("pf_gdn_scan", [&] { launch_scan(pairs[1].old_k, wo.b); }, [&] { launch_scan(pairs[1].new_k, wn.b); });
    time_pair("pf_gated_head _SIG",
              [&] {
                cx.launch(pairs[2].old_k, V, kP, 1,
                          {PtrArg(wo.b.qkvz), PtrArg(wo.b.o), PtrArg(gated_w), PtrArg(wo.b.y), arg_val(kP)});
              },
              [&] {
                cx.launch(pairs[2].new_k, V / 8, kP, 1,
                          {PtrArg(wn.b.qkvz), PtrArg(wn.b.o), PtrArg(gated_w), PtrArg(wn.b.y), arg_val(kP)});
              });
    time_pair("the walk (6 launches)", [&] { old_fn(wo.b, 0, kP); }, [&] { new_fn(wn.b, 0, kP); });
    rq::GdnChunkBuffers bpb = wn.b;
    bpb.qkvz = dq16.ptr();
    bpb.qkvz_bf16 = true;
    time_pair("the walk, new _PB", [&] { old_fn(wo.b, 0, kP); }, [&] { new_fn(bpb, 0, kP); });
  }
  return ds_ + dr + dy == 0 && split && pb_bit && spill_free;
}

}  // namespace

int main(int argc, char** argv) {
  const bool timing = argc > 1 && std::string(argv[1]) == "--time";
  if (l0::Context::gpu_count() < 1) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  runtime::prefill::KernelCache kc(ctx);
  std::printf("qwen4exp_pf_gdn_bundle_test on %s (scan entry %s)\n", ctx.name().c_str(),
              runtime::prefill::gdn_scan_entry_name());
  bool ok = true;
  const std::pair<uint32_t, uint32_t> heads[] = {{16, 48}, {8, 24}};   // one card / pp, a tp card
  for (auto [K, V] : heads) ok = shape(ctx, cx, kc, K, V, timing) && ok;
  if (!ok) {
    std::printf("qwen4exp_pf_gdn_bundle_test FAIL\n");
    return 1;
  }
  std::printf("qwen4exp_pf_gdn_bundle_test OK\n");
  return 0;
}

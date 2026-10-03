#include "runtime/prefill/int8.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/prefill/pf_kernels.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/int8_signs.h"
#include "runtime/prefill/profile.h"

namespace runtime::prefill {
namespace {
constexpr uint32_t kNs = kernels::kPfSlabWidth;
constexpr uint32_t kRotBlock = 1024;   // the Hadamard block; every K is a multiple
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::linear_i8: " + what);
}

// What pf_colmax_rot / pf_requant_rot need of the weight: int4, either layout,
// whole 1024-k rotation blocks up to the largest K the scratch holds.
void check_weight(const loader::DeviceWeight& w, uint32_t max_k) {
  const model::GemvShape& sh = w.shape;
  require(w.kind == model::WeightKind::Int4, "the h8 path's linears are all int4; this one is not");
  require(sh.K % kRotBlock == 0, "K = " + std::to_string(sh.K) + " is not a whole number of 1024-k rotation blocks");
  require(sh.K <= max_k, "K = " + std::to_string(sh.K) + " exceeds the int8 scratch's " +
                             std::to_string(max_k));
  require(sh.layout == 0 || sh.layout == 1, "layout " + std::to_string(sh.layout) + " is neither 0 nor 1");
  require(sh.layout != 0 || w.scales != nullptr, "layout 0 weight has no independent scales allocation");
}

// Everything linear_l0's walks require, on the h8 path.
void check_walk(const loader::DeviceWeight& w, uint32_t M, uint32_t max_k) {
  check_weight(w, max_k);
  const model::GemvShape& sh = w.shape;
  require(sh.N % kNs == 0, "N = " + std::to_string(sh.N) + " is not a whole number of 1024-column slabs");
  require(M > 0 && pad256(M) <= PrefillScratch::kC,
          "M = " + std::to_string(M) + " padded to 256 exceeds the kC-row scratch");
}

// The shared walk: quantise x once, then per slab requant into w8_slab and one
// GEMM whose output pointer and pitch `c_at(n0)` / `ldc` choose the epilogue.
// All on cx's in-order list; no host wait except the instrument's.
template <class COut>
void walk(Context& cx, KernelCache& kc, Int8State& q, const loader::DeviceWeight& w,
          const uint16_t* x, uint32_t M, bool silu, COut c_at, uint32_t ldc) {
  const model::GemvShape& sh = w.shape;
  const std::pair<l0::Mem, l0::Mem>& sc = q.scales(cx, kc, w);   // cached after the first call
  const uint32_t Mp = pad256(M);
  const uint32_t ldxq = sh.K / 2;
  const l0::Mem& sgn = q.signs_f32(sh.K);
  const l0::Mem& bits = q.sign_bits(sh.K);
  l0::Kernel& quant = kc(kernels::pf_quant_had_variant(sh.K), "pf_quant_had");
  l0::Kernel& requant = kc(kernels::pf_requant_rot_variant(sh.layout), "pf_requant_rot");
  l0::Kernel& gemm = kc(kernels::pf_gemm_i8_variant(silu), "pf_gemm_i8");
  const void* scales = sh.layout == 0 ? w.scales->ptr() : nullptr;
  const float* ws = sc.first.as<float>();
  l0::Mem& w8 = q.w8_slab();

  // x [pad256(M)][K] at pitch K -> xq [pad256(M)][K] int8, xs [pad256(M)].
  cx.launch(quant, Mp, 1, 1,
            {PtrArg(x), PtrArg(sgn.ptr()), PtrArg(q.xq().ptr()), PtrArg(q.xs().ptr()),
             arg_val(sh.K)});
  profile_wait(cx, Phase::kI8Quant);
  for (uint32_t n0 = 0; n0 < sh.N; n0 += kNs) {
    cx.launch(requant, kNs / 16, sh.K / kRotBlock, 1,
              {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(bits.ptr()), PtrArg(sc.second.ptr()),
               PtrArg(w8.ptr()), arg_val(n0), arg_val(sh.N), arg_val(sh.K), arg_val(kNs)});
    profile_wait(cx, Phase::kI8Requant);
    cx.launch(gemm, Mp / 256, kNs / 128, 1,
              {PtrArg(q.xq().ptr()), PtrArg(q.xs().ptr()), PtrArg(w8.ptr()), PtrArg(ws + n0),
               PtrArg(c_at(n0)), arg_val(Mp), arg_val(sh.K), arg_val(kNs), arg_val(ldxq),
               arg_val(kNs), arg_val(ldc)});
    profile_wait(cx, Phase::kI8Gemm);
  }
}
}  // namespace

Int8State::Int8State(l0::Context& ctx, uint32_t max_k)
    : ctx_(ctx),
      max_k_(max_k),
      imm_(l0::CmdList::immediate(ctx)),
      xq_(ctx, l0::MemKind::Device, size_t{PrefillScratch::kC} * max_k),
      xs_(ctx, l0::MemKind::Device, size_t{PrefillScratch::kC} * 4),
      w8_(ctx, l0::MemKind::Device, size_t{max_k / 4} * kNs * 4) {}

const l0::Mem& Int8State::signs_f32(uint32_t K) {
  auto it = signs_.find(K);
  if (it == signs_.end()) {
    const std::vector<float> d = int8_signs(K);
    it = signs_.emplace(K, l0::Mem(ctx_, l0::MemKind::Device, d.size() * 4)).first;
    imm_.copy(it->second.ptr(), d.data(), d.size() * 4);
  }
  return it->second;
}

const l0::Mem& Int8State::sign_bits(uint32_t K) {
  auto it = bits_.find(K);
  if (it == bits_.end()) {
    const std::vector<uint32_t> b = int8_sign_bits(K);
    it = bits_.emplace(K, l0::Mem(ctx_, l0::MemKind::Device, b.size() * 4)).first;
    imm_.copy(it->second.ptr(), b.data(), b.size() * 4);
  }
  return it->second;
}

const std::pair<l0::Mem, l0::Mem>& Int8State::scales(Context& cx, KernelCache& kc,
                                                     const loader::DeviceWeight& w) {
  auto it = scales_.find(w.mem.ptr());
  if (it != scales_.end()) return it->second;
  check_weight(w, max_k_);
  const model::GemvShape& sh = w.shape;
  require(sh.N % 16 == 0, "N = " + std::to_string(sh.N) + " is not a whole number of 16-column tiles");
  // colmax[n] = float bits of max_k |(W R_K)[k, n]|, by atomic_max over blocks
  // from zero, then ws = max / 127 (1 for an all-zero column), inv = 1 / ws.
  const std::vector<uint32_t> zero(sh.N, 0u);
  l0::Mem colmax(ctx_, l0::MemKind::Device, size_t{sh.N} * 4);
  imm_.copy(colmax.ptr(), zero.data(), zero.size() * 4);
  const l0::Mem& bits = sign_bits(sh.K);
  const void* scales = sh.layout == 0 ? w.scales->ptr() : nullptr;
  cx.launch(kc(kernels::pf_requant_rot_variant(sh.layout), "pf_colmax_rot"), sh.N / 16,
            sh.K / kRotBlock, 1,
            {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(bits.ptr()), PtrArg(colmax.ptr()),
             arg_val(0u), arg_val(sh.N), arg_val(sh.K)});
  cx.wait();
  std::vector<uint32_t> mx(sh.N);
  imm_.copy(mx.data(), colmax.ptr(), mx.size() * 4);
  std::vector<float> ws(sh.N), inv(sh.N);
  for (uint32_t n = 0; n < sh.N; ++n) {
    float m;
    std::memcpy(&m, &mx[n], 4);
    ws[n] = m > 0.0f ? m / 127.0f : 1.0f;
    inv[n] = 1.0f / ws[n];
  }
  l0::Mem dws(ctx_, l0::MemKind::Device, ws.size() * 4);
  l0::Mem dinv(ctx_, l0::MemKind::Device, inv.size() * 4);
  imm_.copy(dws.ptr(), ws.data(), ws.size() * 4);
  imm_.copy(dinv.ptr(), inv.data(), inv.size() * 4);
  return scales_.emplace(w.mem.ptr(), std::pair<l0::Mem, l0::Mem>(std::move(dws), std::move(dinv)))
      .first->second;
}

l0::Mem& Int8State::xq() { return xq_; }
l0::Mem& Int8State::xs() { return xs_; }
l0::Mem& Int8State::w8_slab() { return w8_; }

size_t Int8State::bytes() const {
  size_t b = xq_.size() + xs_.size() + w8_.size();
  for (const auto& [ptr, p] : scales_) b += p.first.size() + p.second.size();
  return b;
}

size_t linear_i8_launches(const model::GemvShape& sh) { return 1 + 2 * (size_t(sh.N) / kNs); }

void linear_i8(Context& cx, KernelCache& kc, PrefillScratch& s, Int8State& q,
               const loader::DeviceWeight& w, const uint16_t* x, uint32_t M) {
  check_walk(w, M, q.max_k());
  const model::GemvShape& sh = w.shape;
  require(s.partials.size() >= size_t(pad256(M)) * sh.N * 4,
          "`partials` is smaller than the [pad256(M)][N] fp32 this linear writes");
  float* out = s.partials.as<float>();
  walk(cx, kc, q, w, x, M, /*silu=*/false, [out](uint32_t n0) { return out + n0; }, sh.N);
}

void linear_i8_silu(Context& cx, KernelCache& kc, PrefillScratch& s, Int8State& q,
                    const loader::DeviceWeight& w, const uint16_t* x, uint32_t M,
                    uint16_t* out, uint32_t ldx) {
  (void)s;   // partials untouched: the epilogue writes bf16 x straight into `out`
  check_walk(w, M, q.max_k());
  const model::GemvShape& sh = w.shape;
  require(size_t(ldx) * 2 == size_t(sh.N),
          "ldx = " + std::to_string(ldx) + " is not N/2 for N = " + std::to_string(sh.N));
  // Every slab base is a multiple of 1024 and hence of 32, which keeps the
  // kernel's "atom 0 is gate" true of the global column too.
  static_assert(kNs % 32 == 0, "the slab width must keep the gate||up 32-column phase");
  walk(cx, kc, q, w, x, M, /*silu=*/true, [out](uint32_t n0) { return out + n0 / 2; }, ldx);
}

}  // namespace runtime::prefill

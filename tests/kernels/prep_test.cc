// prep.cl vs the CPU reference in prep_ref.h.
//
// The bar here is **bit-exactness**, not a tolerance. Every op in these kernels
// is a rounded scalar chain (bf16 in, fp32 inside one op, bf16 out) plus one
// fp32 variance tree whose order prep_ref.h reproduces exactly (lane i sums
// k = i, i+WG, … with `fma`, then the pairwise SLM tree 256->128->…->1). So:
//
//   * `prep_res_norm` - `x_out` AND the in-place-updated `resid` must be
//     bit-identical to the reference, at S_PREV = 0 and 16.
//   * `prep_silu_mul` / `prep_gated_head` - everything except the final `silu`
//     factor is likewise exact, but OpenCL allows 3 ulp on fp32 `exp` where the
//     host's `expf` is ~0.5, so the *final* value is compared at **2 bf16 ulp**.
//     To keep the exact bar on the rest of the chain, each of those kernels also
//     runs a case whose silu argument is 30.0f: `exp(-30) ≈ 9.4e-14` is far
//     below `2^-24`, so `1 + exp(-30)` is exactly 1.0f in fp32 on *any*
//     conforming implementation and `silu(30) = 30.0f` on both sides. Those
//     cases are asserted bit-exact, which pins the norm/mul chain (`n_b`, `t_b`,
//     `s_b`) that is otherwise not directly observable.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "prep_ref.h"

namespace {

std::vector<float> random_f32(size_t n, uint32_t seed, float mean, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(mean, sigma);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}

std::vector<uint16_t> random_bf16(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = common::f32_to_bf16(d(rng));
  return v;
}

// bf16 words ordered as integers: the distance between two keys is the number
// of representable bf16 values between them (one "ulp" of the format).
int32_t bf16_key(uint16_t v) {
  return (v & 0x8000u) ? -int32_t(v & 0x7FFFu) : int32_t(v);
}

struct Cmp {
  uint32_t max_ulp = 0;
  size_t exact = 0, n = 0, worst = 0;
};

Cmp compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  Cmp c;
  c.n = ref.size();
  CHECK_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] == ref[i]) { ++c.exact; continue; }
    int32_t d = bf16_key(got[i]) - bf16_key(ref[i]);
    uint32_t u = uint32_t(d < 0 ? -d : d);
    if (u > c.max_ulp) { c.max_ulp = u; c.worst = i; }
  }
  return c;
}

void require(const Cmp& c, uint32_t tol, const char* what) {
  if (c.max_ulp > tol) {
    std::fprintf(stderr, "%s: max %u bf16 ulp > tol %u at index %zu (%zu/%zu exact)\n", what,
                 c.max_ulp, tol, c.worst, c.exact, c.n);
  }
  CHECK(c.max_ulp <= tol);
}

template <class T>
l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const std::vector<T>& v) {
  l0::Mem m(ctx, l0::MemKind::Device, v.size() * sizeof(T));
  imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
  return m;
}

template <class T>
void download(l0::CmdList& imm, std::vector<T>& v, const l0::Mem& m) {
  imm.copy(v.data(), m.ptr(), v.size() * sizeof(T));
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // One launch of one kernel, recorded in a regular list and executed once.
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }
};

// ---------------------------------------------------------------------------

void case_res_norm(Dev& d, uint32_t M, uint32_t K, uint32_t S_PREV) {
  const size_t np = S_PREV ? size_t(S_PREV) * M * K : 1;
  std::vector<float> partials = random_f32(np, 100 + S_PREV, 0.f, 1.f);
  std::vector<uint16_t> resid = random_bf16(size_t(M) * K, 200 + S_PREV, -1.f, 1.f);
  std::vector<float> norm_w(K);
  {
    std::mt19937 rng(300 + S_PREV);
    std::uniform_real_distribution<float> d05(-0.05f, 0.05f);
    for (auto& w : norm_w) w = 1.0f + d05(rng);   // device holds (1 + w), fp32
  }

  std::vector<uint16_t> resid_ref = resid, x_ref(size_t(M) * K, 0);
  prep_ref::res_norm(partials.data(), resid_ref.data(), norm_w.data(), x_ref.data(), M, K, S_PREV);

  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem rbuf = upload(d.ctx, d.imm, resid);
  l0::Mem wbuf = upload(d.ctx, d.imm, norm_w);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * K * 2);
  l0::Module mod(d.ctx, kernels::path(kernels::prep_res_norm_variant(M, K, S_PREV)));
  l0::Kernel k = mod.kernel("prep_res_norm");
  k.group_size(256);
  k.arg_ptr(0, pbuf.ptr());
  k.arg_ptr(1, rbuf.ptr());
  k.arg_ptr(2, wbuf.ptr());
  k.arg_ptr(3, xbuf.ptr());
  d.run(k, 1, M);

  std::vector<uint16_t> x_got(size_t(M) * K), resid_got(size_t(M) * K);
  download(d.imm, x_got, xbuf);
  download(d.imm, resid_got, rbuf);

  Cmp cx = compare(x_got, x_ref), cr = compare(resid_got, resid_ref);
  std::printf("prep_res_norm M=%u K=%u SP=%u: x_out %zu/%zu exact (max %u ulp), "
              "resid %zu/%zu exact (max %u ulp)\n",
              M, K, S_PREV, cx.exact, cx.n, cx.max_ulp, cr.exact, cr.n, cr.max_ulp);
  require(cx, 0, "prep_res_norm x_out");
  require(cr, 0, "prep_res_norm resid");
}

// `exact_silu`: force every gate column to sum to 30.0f, making silu(gate)
// provably 30.0f on host and device alike, so the whole chain is bit-exact.
void case_silu_mul(Dev& d, uint32_t M, bool exact_silu) {
  const uint32_t S = prep_ref::kSiluS, FN = prep_ref::kSiluFusedN, N = prep_ref::kSiluN;
  std::vector<float> partials = random_f32(size_t(S) * M * FN, exact_silu ? 401 : 400, 0.f, 1.f);
  if (exact_silu)
    for (size_t s = 0; s < S; ++s)
      for (size_t m = 0; m < M; ++m)
        for (size_t j = 0; j < FN; ++j)
          if ((j / 16) % 2 == 0) partials[(s * M + m) * FN + j] = 7.5f;   // 4 slices -> 30.0

  std::vector<uint16_t> x_ref(size_t(M) * N, 0);
  prep_ref::silu_mul(partials.data(), x_ref.data(), M);

  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * N * 2);
  l0::Module mod(d.ctx, kernels::path(kernels::prep_silu_mul_variant(M)));
  l0::Kernel k = mod.kernel("prep_silu_mul");
  k.group_size(256);
  k.arg_ptr(0, pbuf.ptr());
  k.arg_ptr(1, xbuf.ptr());
  d.run(k, (N + 4095) / 4096, M);

  std::vector<uint16_t> x_got(size_t(M) * N);
  download(d.imm, x_got, xbuf);
  Cmp c = compare(x_got, x_ref);
  const uint32_t tol = exact_silu ? 0 : 2;
  std::printf("prep_silu_mul M=%u %s: %zu/%zu exact (max %u ulp, tol %u)\n", M,
              exact_silu ? "silu(30)=30 exact-case" : "random", c.exact, c.n, c.max_ulp, tol);
  require(c, tol, "prep_silu_mul x_out");
}

// `exact_silu`: z = 30.0f everywhere - see the note above; this is what makes
// the gated norm's own chain (o_b -> var tree -> n_b -> t_b) observable exactly.
void case_gated_head(Dev& d, uint32_t M, bool exact_silu) {
  const uint32_t S = prep_ref::kGatedS, QN = prep_ref::kQkvzN, H = prep_ref::kGatedHeads;
  const uint32_t D = prep_ref::kHeadDim, ON = prep_ref::kGatedOutN;
  std::vector<float> qkvz = random_f32(size_t(S) * M * QN, exact_silu ? 501 : 500, 0.f, 1.f);
  if (exact_silu)
    for (size_t i = 0; i < qkvz.size(); ++i) qkvz[i] = 30.0f;
  std::vector<float> gdn_o = random_f32(size_t(M) * H * D, 600, 0.f, 1.f);
  std::vector<uint16_t> gated_w = random_bf16(D, 700, -0.5f, 0.5f);   // plain w, bf16

  std::vector<uint16_t> x_ref(size_t(M) * ON, 0);
  prep_ref::gated_head(qkvz.data(), gdn_o.data(), gated_w.data(), x_ref.data(), M);

  l0::Mem qbuf = upload(d.ctx, d.imm, qkvz);
  l0::Mem obuf = upload(d.ctx, d.imm, gdn_o);
  l0::Mem wbuf = upload(d.ctx, d.imm, gated_w);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * ON * 2);
  l0::Module mod(d.ctx, kernels::path(kernels::prep_gated_head_variant(M)));
  l0::Kernel k = mod.kernel("prep_gated_head");
  k.group_size(128);
  k.arg_ptr(0, qbuf.ptr());
  k.arg_ptr(1, obuf.ptr());
  k.arg_ptr(2, wbuf.ptr());
  k.arg_ptr(3, xbuf.ptr());
  d.run(k, H, M);

  std::vector<uint16_t> x_got(size_t(M) * ON);
  download(d.imm, x_got, xbuf);
  Cmp c = compare(x_got, x_ref);
  const uint32_t tol = exact_silu ? 0 : 2;
  std::printf("prep_gated_head M=%u %s: %zu/%zu exact (max %u ulp, tol %u)\n", M,
              exact_silu ? "silu(30)=30 exact-case" : "random", c.exact, c.n, c.max_ulp, tol);
  require(c, tol, "prep_gated_head x_out");
}

}  // namespace

int main() {
  Dev d;
  case_res_norm(d, 1, 5120, 0);
  case_res_norm(d, 1, 5120, 16);
  case_silu_mul(d, 1, false);
  case_silu_mul(d, 1, true);
  case_gated_head(d, 1, false);
  case_gated_head(d, 1, true);
  std::puts("prep_test OK");
  return 0;
}

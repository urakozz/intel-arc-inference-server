// PERF-102 (plan 41d Task 1): Inkling's int8 cache kernels against src/common/kv8.h at head_dim 128.
//   * the writer: the fused prep's INK_KV8 build stores encode_n<128> of exactly the bf16 rows the default build
//     stores, and q rotated by rotate_q_n<128> of the default build's q - bitwise;
//   * the readers: kol_attn's KV8 decode (M 1 and 8) and kol_pf_attn's KV8 prompt flash over a cache encoded by
//     kv8.h, against a double-precision twin on the same codes (relative bias, per-token sink, un-rotation, gate);
//     the flash also bitwise across a split of its rows.
// inkling2_kv8_test [BASELINE_KERNEL_DIR]: the readers also run from another build's binaries (an earlier reader form)
// on the same inputs, and their outputs must be the same bits (PERF-42's carry: same values, same sum order).
#include <bit>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/inkling2_kernels.h"
#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/control.h"

namespace {
namespace ink = kernels::inkling2;
namespace kv8 = common::kv8;
using D = ink::D;
constexpr uint32_t HD = D::head_dim, KV = D::kv_heads, Q = D::q_heads, KW = D::kv_n, REL = D::relative_dim,
                   EXT = D::relative_extent;

float wave(size_t i, size_t mod) { return float(int(i % mod) - int(mod / 2)); }

template <class... A>
void args(l0::Kernel& k, A... a) {
  uint32_t i = 0;
  ([&] {
    if constexpr (std::is_pointer_v<A> || std::is_null_pointer_v<A>) k.arg_ptr(i++, a);
    else k.arg(i++, a);
  }(), ...);
}

// A cache of `rows` positions: raw rows encoded by kv8.h after the bf16 rounding the writer applies.
struct Cache {
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
  explicit Cache(uint32_t rows) : k(size_t(rows) * KW), v(k.size()), ks(size_t(rows) * KV), vs(ks.size()) {
    for (uint32_t p = 0; p < rows; ++p)
      for (uint32_t j = 0; j < KV; ++j) {
        float rk[HD], rv[HD];
        for (uint32_t i = 0; i < HD; ++i) {
          const size_t at = (size_t(p) * KV + j) * HD + i;
          rk[i] = common::bf16_to_f32(common::f32_to_bf16(wave(at * 229 + p, 37) / 16.f));
          rv[i] = common::bf16_to_f32(common::f32_to_bf16(wave(at * 359 + 7 * p, 43) / 24.f));
        }
        ks[p * KV + j] = kv8::encode_n<HD>(rk, &k[(size_t(p) * KV + j) * HD]);
        vs[p * KV + j] = kv8::encode_n<HD>(rv, &v[(size_t(p) * KV + j) * HD]);
      }
  }
  double kd(uint32_t p, uint32_t j, uint32_t i) const {
    return kv8::dequant(k[(size_t(p) * KV + j) * HD + i], ks[p * KV + j]);
  }
  double vd(uint32_t p, uint32_t j, uint32_t i) const {
    return kv8::dequant(v[(size_t(p) * KV + j) * HD + i], vs[p * KV + j]);
  }
};

// The twin: row at position P, q head h (rotated q), keys 0..P: softmax with the sink in the denominator, the
// relative bias under EXT, the output un-rotated (FWHT(s . o) / 8) and gated.
std::vector<double> twin(const Cache& c, const float* qr, const float* rel, const std::vector<float>& table,
                         float sink, const float* gate, uint32_t P, uint32_t h) {
  const uint32_t j = h / (Q / KV);
  std::vector<double> w(P + 1);
  double mx = sink;
  for (uint32_t t = 0; t <= P; ++t) {
    double dot = 0, bias = 0;
    for (uint32_t i = 0; i < HD; ++i) dot += double(qr[i]) * c.kd(t, j, i);
    if (P - t < EXT)
      for (uint32_t d = 0; d < REL; ++d) bias += double(rel[d]) * table[d * EXT + P - t];
    w[t] = dot / std::sqrt(double(HD)) + bias;
    mx = std::max(mx, w[t]);
  }
  double denom = std::exp(sink - mx);
  for (auto& x : w) denom += (x = std::exp(x - mx));
  std::vector<double> o(HD, 0.);
  for (uint32_t i = 0; i < HD; ++i) {
    for (uint32_t t = 0; t <= P; ++t) o[i] += w[t] * c.vd(t, j, i);
    o[i] = kv8::sign_neg(i) ? -o[i] / denom : o[i] / denom;
  }
  for (uint32_t s = 1; s < HD; s <<= 1)
    for (uint32_t i = 0; i < HD; ++i)
      if (!(i & s)) {
        const double a = o[i], b = o[i + s];
        o[i] = a + b;
        o[i + s] = a - b;
      }
  for (uint32_t i = 0; i < HD; ++i) o[i] = o[i] / 8 / (1 + std::exp(-double(gate[i])));
  return o;
}

// The device's bf16 head row against the twin: cosine and the largest error against the row's largest value.
void compare(const uint16_t* got, const std::vector<double>& want, const char* what) {
  double cross = 0, ng = 0, nw = 0, err = 0, top = 0;
  for (uint32_t i = 0; i < HD; ++i) {
    const double g = common::bf16_to_f32(got[i]);
    cross += g * want[i];
    ng += g * g;
    nw += want[i] * want[i];
    err = std::max(err, std::abs(g - want[i]));
    top = std::max(top, std::abs(want[i]));
  }
  const double cos = cross / std::sqrt(ng * nw);
  if (cos < .99999 || err > top / 128) std::printf("%s: cosine %.8f, max error %.3g of %.3g\n", what, cos, err, top);
  CHECK(cos >= .99999 && err <= top / 128);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string baseline = argc > 1 ? argv[1] : "";
  l0::Context ctx(0);
  auto list = l0::CmdList::immediate(ctx);
  const auto upload = [&](const auto& data) {
    l0::Mem m(ctx, l0::MemKind::Device, data.size() * sizeof(data[0]));
    list.copy(m.ptr(), data.data(), m.size());
    return m;
  };
  const auto download = [&](auto& data, const l0::Mem& m) { list.copy(data.data(), m.ptr(), m.size()); };

  // ---- the writer: the decode prep, default and INK_KV8, on the same inputs (a full visit, 3 rows at 4095) ------
  {
    constexpr uint32_t M = 3, pos = 4095, rows = pos + M;
    std::vector<float> q(M * D::q_n), rel(M * Q * REL), qs(D::q_n), rs(Q * REL), k(M * KW), v(M * KW),
        kh(3 * KW), vh(3 * KW), kt(4 * KW), vt(4 * KW), kd(M * 64), vd(M * 64);
    const auto fill = [](std::vector<float>& a, size_t seed, size_t mod, float scale) {
      for (size_t i = 0; i < a.size(); ++i) a[i] = wave(i * seed + 3, mod) * scale;
    };
    fill(q, 131, 29, 1 / 8.f); fill(rel, 17, 13, 1 / 4.f); fill(qs, 7, 11, 1 / 16.f); fill(rs, 5, 9, 1 / 8.f);
    fill(k, 193, 31, 1 / 4.f); fill(v, 211, 37, 1 / 4.f); fill(kh, 23, 17, 1 / 8.f); fill(vh, 29, 19, 1 / 8.f);
    fill(kt, 37, 7, 1 / 16.f); fill(vt, 41, 9, 1 / 16.f); fill(kd, 43, 5, 1 / 32.f); fill(vd, 47, 7, 1 / 32.f);
    for (auto& x : qs) x += 1.f;
    for (auto& x : rs) x += 1.f;
    std::vector<runtime::Control> control(1);
    control[0].pos = pos;
    control[0].n_active = M;
    auto ctrl = upload(control);
    auto qsd = upload(qs), rsd = upload(rs), kdv = upload(k), vdv = upload(v), khd = upload(kh), vhd = upload(vh),
         ktd = upload(kt), vtd = upload(vt), kpd = upload(kd), vpd = upload(vd);
    const float eps = 1e-6f, clip = 22.6274f, r_clip = 8.f, alpha = .1f;
    const uint32_t floor = 8192, ring = 0;

    auto q0 = upload(q), r0 = upload(rel);
    std::vector<uint16_t> c16(size_t(rows) * KW);
    auto ck16 = upload(c16), cv16 = upload(c16);
    l0::Module plain(ctx, kernels::path(std::string(ink::kPrep) + "_FUSED"));
    auto prep = plain.kernel("ink_qkv_prep");
    prep.group_size(HD);
    args(prep, q0.ptr(), r0.ptr(), qsd.ptr(), rsd.ptr(), kdv.ptr(), vdv.ptr(), khd.ptr(), vhd.ptr(), ktd.ptr(),
         vtd.ptr(), kpd.ptr(), vpd.ptr(), ck16.ptr(), cv16.ptr(), ctrl.ptr(), ring, eps, clip, r_clip, floor, alpha);
    list.launch(prep, Q + 2 * KV, M);

    auto q8 = upload(q), r8 = upload(rel);
    std::vector<int8_t> c8(size_t(rows) * KW);
    std::vector<uint16_t> s8(size_t(rows) * KV);
    auto ck8 = upload(c8), cv8 = upload(c8), ks8 = upload(s8), vs8 = upload(s8);
    l0::Module int8(ctx, kernels::path(std::string(ink::kPrep) + "_FUSED_KV8"));
    auto prep8 = int8.kernel("ink_qkv_prep");
    prep8.group_size(HD);
    args(prep8, q8.ptr(), r8.ptr(), qsd.ptr(), rsd.ptr(), kdv.ptr(), vdv.ptr(), khd.ptr(), vhd.ptr(), ktd.ptr(),
         vtd.ptr(), kpd.ptr(), vpd.ptr(), ck8.ptr(), cv8.ptr(), ks8.ptr(), vs8.ptr(), ctrl.ptr(), ring, eps, clip,
         r_clip, floor, alpha);
    list.launch(prep8, Q + 2 * KV, M);

    std::vector<float> qa(q.size()), qb(q.size()), ra(rel.size()), rb(rel.size());
    download(qa, q0); download(qb, q8); download(ra, r0); download(rb, r8);
    CHECK(ra == rb);
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t h = 0; h < Q; ++h) {
        float want[HD];
        kv8::rotate_q_n<HD>(&qa[(r * Q + h) * HD], want);
        for (uint32_t i = 0; i < HD; ++i)
          CHECK_EQ(std::bit_cast<uint32_t>(qb[(r * Q + h) * HD + i]), std::bit_cast<uint32_t>(want[i]));
      }
    for (const bool key : {true, false}) {
      download(c16, key ? ck16 : cv16);
      download(c8, key ? ck8 : cv8);
      download(s8, key ? ks8 : vs8);
      for (uint32_t p = pos; p < rows; ++p)
        for (uint32_t j = 0; j < KV; ++j) {
          float x[HD];
          int8_t code[HD];
          for (uint32_t i = 0; i < HD; ++i) x[i] = common::bf16_to_f32(c16[(size_t(p) * KV + j) * HD + i]);
          CHECK_EQ(s8[p * KV + j], kv8::encode_n<HD>(x, code));
          for (uint32_t i = 0; i < HD; ++i) CHECK_EQ(c8[(size_t(p) * KV + j) * HD + i], code[i]);
        }
    }
    std::puts("Inkling int8 cache writer: codes, scales and the rotated q bitwise kv8.h's (head_dim 128)");
  }

  // ---- the readers over a kv8.h cache ---------------------------------------------------------------------------
  constexpr uint32_t pos = 1530, C = 21, depth = pos + C;
  const Cache cache(depth);
  auto kc = upload(cache.k), vc = upload(cache.v), ksd = upload(cache.ks), vsd = upload(cache.vs);
  std::vector<float> q(C * D::q_n), gate(q.size()), rel(C * Q * REL), table(REL * EXT), sinks(C * 64);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = wave(i * 7, 15) / 6.f;
    gate[i] = wave(i, 23) / 8.f;
  }
  for (size_t i = 0; i < rel.size(); ++i) rel[i] = wave(i, 17) / 16.f;
  for (size_t i = 0; i < table.size(); ++i) table[i] = wave(i, 19) / 16.f;
  for (size_t i = 0; i < sinks.size(); ++i) sinks[i] = wave(i, 23) / 4.f + 2.f;
  auto qd = upload(q), gd = upload(gate), rd = upload(rel), td = upload(table), sd = upload(sinks);

  const auto bin = [&](const std::string& name, bool base) {
    return base ? baseline + "/" + name + ".bin" : kernels::path(name);
  };
  for (const uint32_t M : {1u, 8u}) {
    std::vector<runtime::Control> control(1);
    control[0].pos = pos;
    control[0].n_active = M;
    auto ctrl = upload(control);
    std::vector<float> parts(size_t(Q) * ink::kPartitions * M * (HD + 2));
    std::vector<uint16_t> out(M * D::q_n);
    auto pd = upload(parts), od = upload(out);
    const auto run_decode = [&](bool base) {
      l0::Module attn(ctx, bin(ink::attn(M, false) + "_KV8", base));
      auto decode = attn.kernel("kol_attn_decode");
      decode.group_size(HD);
      args(decode, ctrl.ptr(), qd.ptr(), kc.ptr(), vc.ptr(), ksd.ptr(), vsd.ptr(), pd.ptr(), rd.ptr(), td.ptr());
      list.launch(decode, KV, ink::kPartitions);
      auto reduce = attn.kernel("kol_attn_reduce");
      reduce.group_size(HD);
      args(reduce, ctrl.ptr(), pd.ptr(), od.ptr(), sd.ptr(), gd.ptr());
      list.launch(reduce, Q, M);
      std::vector<uint16_t> got(out.size());
      download(got, od);
      return got;
    };
    out = run_decode(false);
    if (!baseline.empty()) CHECK(run_decode(true) == out);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t h : {0u, 2u, 7u, 23u}) {
        const size_t at = (size_t(m) * Q + h) * HD;
        compare(&out[at], twin(cache, &q[at], &rel[(m * Q + h) * REL], table, sinks[m * 64 + h], &gate[at], pos + m, h),
                "decode");
      }
  }
  std::puts("Inkling int8 cache decode reader (M 1 and 8) matches its double twin");

  {
    std::vector<uint16_t> all(C * D::q_n), split(all.size());
    auto od = upload(all);
    l0::Module module(ctx, kernels::path(ink::prefill_attn(false) + "_KV8"));
    std::unique_ptr<l0::Module> base_module;
    if (!baseline.empty()) base_module = std::make_unique<l0::Module>(ctx, bin(ink::prefill_attn(false) + "_KV8", true));
    auto kernel = module.kernel("kol_pf_flash_attn");
    kernel.group_size(16);
    const auto launch = [&](l0::Kernel& k, uint32_t first, uint32_t n) {
      args(k, static_cast<float*>(qd.ptr()) + first * D::q_n, kc.ptr(), vc.ptr(), ksd.ptr(), vsd.ptr(),
           static_cast<uint16_t*>(od.ptr()) + first * D::q_n, pos + first, n,
           static_cast<float*>(rd.ptr()) + first * Q * REL, td.ptr(), static_cast<float*>(sd.ptr()) + first * 64,
           static_cast<float*>(gd.ptr()) + first * D::q_n);
      list.launch(k, (n + 7) / 8, KV, Q / KV);
    };
    const auto run = [&](uint32_t first, uint32_t n) { launch(kernel, first, n); };
    run(0, C);
    download(all, od);
    run(0, 11);
    run(11, C - 11);
    download(split, od);
    CHECK(all == split);
    // The bound PAIR form (16 rows of a head on two sub-groups, a kv head's q heads a work-group): bitwise the 8-row
    // form, whole and split.
    {
      l0::Module pair_module(ctx, kernels::path(ink::prefill_attn(false) + "_KV8_PAIR"));
      auto pair = pair_module.kernel("kol_pf_flash_attn");
      pair.group_size(ink::kPrefillAttnGroup);
      const auto run_pair = [&](uint32_t first, uint32_t n) {
        args(pair, static_cast<float*>(qd.ptr()) + first * D::q_n, kc.ptr(), vc.ptr(), ksd.ptr(), vsd.ptr(),
             static_cast<uint16_t*>(od.ptr()) + first * D::q_n, pos + first, n,
             static_cast<float*>(rd.ptr()) + first * Q * REL, td.ptr(), static_cast<float*>(sd.ptr()) + first * 64,
             static_cast<float*>(gd.ptr()) + first * D::q_n);
        list.launch(pair, (n + 15) / 16, KV, 1);
      };
      std::vector<uint16_t> got(all.size());
      run_pair(0, C);
      download(got, od);
      CHECK(got == all);
      run_pair(0, 11);
      run_pair(11, C - 11);
      download(got, od);
      CHECK(got == all);
    }
    std::puts("Inkling int8 cache prompt flash: the PAIR form is bitwise the 8-row form");
    if (base_module) {
      auto base_kernel = base_module->kernel("kol_pf_flash_attn");
      base_kernel.group_size(16);
      launch(base_kernel, 0, C);
      std::vector<uint16_t> base(all.size());
      download(base, od);
      CHECK(base == all);
      std::puts("Inkling int8 cache readers: decode (M 1 and 8) and prompt flash bitwise the baseline build's");
    }
    std::vector<float> qb(q.size());
    for (size_t i = 0; i < q.size(); ++i) qb[i] = common::bf16_to_f32(common::f32_to_bf16(q[i]));   // the flash's q
    for (uint32_t row : {0u, 4u, 5u, 10u, 11u, C - 1})
      for (uint32_t h : {0u, 2u, 7u, 23u}) {
        const size_t at = (size_t(row) * Q + h) * HD;
        compare(&all[at], twin(cache, &qb[at], &rel[(row * Q + h) * REL], table, sinks[row * 64 + h], &gate[at],
                               pos + row, h), "prompt flash");
      }
  }
  std::puts("Inkling int8 cache prompt flash matches its double twin and is bitwise across a split");
}

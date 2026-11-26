#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/inkling2_kernels.h"
#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"

int main() {
  namespace ink = kernels::inkling2;
  using D = ink::D;
  constexpr uint32_t pos = 4091, rows = 21, depth = pos + rows;
  using common::bf16_to_f32;
  using common::f32_to_bf16;
  l0::Context ctx(0);
  auto list = l0::CmdList::immediate(ctx);
  const auto upload = [&](const auto& data) {
    l0::Mem m(ctx, l0::MemKind::Device, data.size() * sizeof(data[0]));
    list.copy(m.ptr(), data.data(), m.size());
    return m;
  };
  std::vector<float> q(rows * D::q_n), gate(q.size()), relative(rows * D::q_heads * D::relative_dim),
      table(D::relative_dim * D::relative_extent), sink(rows * 64);
  const auto wave = [](size_t i, size_t mod) { return float(int(i % mod) - int(mod / 2)); };
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = wave(i, 15) / 16.f;
    gate[i] = wave(i, 23) / 8.f;
  }
  for (size_t i = 0; i < relative.size(); ++i) relative[i] = wave(i, 17) / 16.f;
  for (size_t i = 0; i < table.size(); ++i) table[i] = wave(i, 19) / 16.f;
  for (size_t i = 0; i < sink.size(); ++i) sink[i] = wave(i, 23) / 4.f + 4.f;
  auto qd = upload(q), gd = upload(gate), rd = upload(relative), td = upload(table), sd = upload(sink);
  for (bool local : {true, false}) {
    const uint32_t cache_rows = local ? ink::kWindowRing : depth;
    std::vector<uint16_t> keys(cache_rows * D::kv_n), values(keys.size());
    const auto kv_row = [&](uint32_t p) { return (local ? p % cache_rows : p) * D::kv_n; };
    // Populate first the live previous context, then the new rows over the ring wrap.
    for (uint32_t p = local ? pos - D::window : 0; p < depth; ++p)
      for (uint32_t i = 0; i < D::kv_n; ++i) {
        const size_t ix = kv_row(p) + i;
        keys[ix] = f32_to_bf16(wave(size_t(p) * 229 + i, 37) / 32.f);
        values[ix] = f32_to_bf16(wave(size_t(p) * 359 + i, 43) / 32.f);
      }
    auto kd = upload(keys), vd = upload(values);
    std::vector<uint16_t> all(q.size()), split(q.size());
    auto out = upload(all);
    l0::Module module(ctx, kernels::path(ink::prefill_attn(local)));
    auto k = module.kernel("kol_pf_flash_attn");
    k.group_size(16);
    const auto run = [&](uint32_t first, uint32_t n) {
      k.arg_ptr(0, static_cast<float*>(qd.ptr()) + first * D::q_n);
      k.arg_ptr(1, kd.ptr());
      k.arg_ptr(2, vd.ptr());
      k.arg_ptr(3, static_cast<uint16_t*>(out.ptr()) + first * D::q_n);
      k.arg(4, pos + first);
      k.arg(5, n);
      k.arg_ptr(6, static_cast<float*>(rd.ptr()) + first * D::q_heads * D::relative_dim);
      k.arg_ptr(7, td.ptr());
      k.arg_ptr(8, static_cast<float*>(sd.ptr()) + first * 64);
      k.arg_ptr(9, static_cast<float*>(gd.ptr()) + first * D::q_n);
      list.launch(k, (n + 7) / 8, D::kv_heads, D::q_heads / D::kv_heads);
    };
    run(0, rows);
    list.copy(all.data(), out.ptr(), out.size());
    run(0, 11);
    run(11, rows - 11);
    list.copy(split.data(), out.ptr(), out.size());
    CHECK(all == split);
    // The bound PAIR form (16 rows of a head on two sub-groups, a kv head's q heads a work-group): bitwise the 8-row
    // form, whole and split.
    {
      l0::Module pair_module(ctx, kernels::path(ink::prefill_attn(local) + "_PAIR"));
      auto pk = pair_module.kernel("kol_pf_flash_attn");
      pk.group_size(ink::kPrefillAttnGroup);
      const auto run_pair = [&](uint32_t first, uint32_t n) {
        pk.arg_ptr(0, static_cast<float*>(qd.ptr()) + first * D::q_n);
        pk.arg_ptr(1, kd.ptr());
        pk.arg_ptr(2, vd.ptr());
        pk.arg_ptr(3, static_cast<uint16_t*>(out.ptr()) + first * D::q_n);
        pk.arg(4, pos + first);
        pk.arg(5, n);
        pk.arg_ptr(6, static_cast<float*>(rd.ptr()) + first * D::q_heads * D::relative_dim);
        pk.arg_ptr(7, td.ptr());
        pk.arg_ptr(8, static_cast<float*>(sd.ptr()) + first * 64);
        pk.arg_ptr(9, static_cast<float*>(gd.ptr()) + first * D::q_n);
        list.launch(pk, (n + 15) / 16, D::kv_heads, 1);
      };
      run_pair(0, rows);
      list.copy(split.data(), out.ptr(), out.size());
      CHECK(all == split);
      run_pair(0, 11);
      run_pair(11, rows - 11);
      list.copy(split.data(), out.ptr(), out.size());
      CHECK(all == split);
    }
    // PERF-12 (full layers): the split-K form, three key spans then the merge, within the FP32 twin's tolerance of the
    // whole walk on every (row, head).
    if (!local) {
      constexpr uint32_t spans = 3, span = pos / spans / 64 * 64;
      std::vector<float> partials(size_t(spans) * rows * D::q_heads * (D::head_dim + 2));
      auto parts = upload(partials);
      l0::Module split_module(ctx, kernels::path(ink::prefill_attn(false) + "_PAIR_SPLIT"));
      auto sk = split_module.kernel("kol_pf_flash_attn");
      sk.group_size(ink::kPrefillAttnGroup);
      sk.arg_ptr(0, qd.ptr());
      sk.arg_ptr(1, kd.ptr());
      sk.arg_ptr(2, vd.ptr());
      sk.arg_ptr(3, out.ptr());
      sk.arg(4, pos);
      sk.arg(5, rows);
      sk.arg_ptr(6, rd.ptr());
      sk.arg_ptr(7, td.ptr());
      sk.arg_ptr(8, sd.ptr());
      sk.arg_ptr(9, gd.ptr());
      sk.arg(10, span);
      sk.arg_ptr(11, parts.ptr());
      list.launch(sk, (rows + 15) / 16, D::kv_heads, spans);
      auto merge = split_module.kernel("kol_pf_flash_merge");
      merge.group_size(D::head_dim);
      merge.arg_ptr(0, parts.ptr());
      merge.arg_ptr(1, out.ptr());
      merge.arg(2, rows);
      merge.arg(3, spans);
      merge.arg_ptr(4, sd.ptr());
      merge.arg_ptr(5, gd.ptr());
      list.launch(merge, D::q_heads, rows);
      list.copy(split.data(), out.ptr(), out.size());
      for (size_t at = 0; at < all.size(); at += D::head_dim) {
        double cross = 0., na = 0., nb = 0.;
        for (uint32_t i = 0; i < D::head_dim; ++i) {
          const double a = bf16_to_f32(all[at + i]), b = bf16_to_f32(split[at + i]);
          cross += a * b;
          na += a * a;
          nb += b * b;
        }
        CHECK(cross / std::sqrt(na * nb) >= .99999);
      }
    }
    for (uint32_t row : {0u, 4u, 5u, 10u, 11u, rows - 1}) {
      const uint32_t p = pos + row, lo = local ? p - D::window + 1 : 0;
      for (uint32_t head : {0u, 2u, 7u, 23u}) {
        const uint32_t kh = head / (D::q_heads / D::kv_heads);
        const size_t d0 = (row * D::q_heads + head) * D::head_dim;
        const size_t r0 = (row * D::q_heads + head) * D::relative_dim;
        const float s = sink[row * 64 + head];
        std::vector<float> prob;
        prob.reserve(p + 1 - lo);
        float mx = s;
        for (uint32_t key = lo; key <= p; ++key) {
          const auto offset = kv_row(key) + kh * D::head_dim;
          float dot = 0.f, bias = 0.f;
          for (uint32_t i = 0; i < D::head_dim; ++i) dot += q[d0 + i] * bf16_to_f32(keys[offset + i]);
          if (p - key < D::relative_extent)
            for (uint32_t i = 0; i < D::relative_dim; ++i)
              bias = std::fma(relative[r0 + i], table[i * D::relative_extent + p - key], bias);
          prob.push_back(dot / std::sqrt(float(D::head_dim)) + bias);
          mx = std::max(mx, prob.back());
        }
        float denom = std::exp(s - mx);
        for (auto& w : prob) {
          w = std::exp(w - mx);
          denom += w;
        }
        double cross = 0., norm_actual = 0., norm_want = 0.;
        for (uint32_t i = 0; i < D::head_dim; ++i) {
          float acc = 0.f;
          for (uint32_t key = lo; key <= p; ++key)
            acc += prob[key - lo] * bf16_to_f32(values[kv_row(key) + kh * D::head_dim + i]);
          const float want = acc / denom / (1.f + std::exp(-gate[d0 + i]));
          const float actual = bf16_to_f32(all[d0 + i]);
          cross += actual * want;
          norm_actual += actual * actual;
          norm_want += want * want;
        }
        CHECK(cross / std::sqrt(norm_actual * norm_want) >= .99999);
      }
    }
  }
  std::puts("Inkling prompt attention matches the FP32 twin, is bitwise across a split at the window wrap, and the PAIR "
            "form is bitwise the 8-row form; the split-K form agrees with the whole walk");
}

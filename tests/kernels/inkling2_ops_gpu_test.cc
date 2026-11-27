#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <initializer_list>
#include <span>
#include <type_traits>
#include <vector>

#include "check.h"
#include "common/expert_fp.h"
#include "kernels/inkling2_kernels.h"
#include "kernels/inkling2_ref.h"
#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/control.h"

int main() {
  namespace ink = kernels::inkling2;
  constexpr uint32_t H = ink::D::hidden;
  l0::Context ctx(0);
  auto list = l0::CmdList::immediate(ctx);
  const auto upload = [&](const auto& a) {
    l0::Mem m(ctx, l0::MemKind::Device, a.size() * sizeof(a[0]));
    list.copy(m.ptr(), a.data(), m.size());
    return m;
  };
  l0::Module glue(ctx, kernels::path(std::string(ink::kGlue)));
  const auto bits = [](float x) { return std::bit_cast<uint32_t>(x); };
  std::vector<float> x(H, 2.f), q(H, 0.f), mix(H);
  auto input = upload(x), query = upload(q), out = upload(mix);
  auto depth = glue.kernel("ink_depth_mix");
  depth.group_size(128);
  depth.arg_ptr(0, input.ptr());
  depth.arg_ptr(1, query.ptr());
  depth.arg_ptr(2, out.ptr());
  depth.arg(3, uint32_t(1));
  depth.arg(4, uint32_t(1));
  depth.arg(5, 1e-6f);
  depth.arg_ptr(6, nullptr);
  list.launch(depth, 1, 1);
  list.copy(mix.data(), out.ptr(), out.size());
  const auto want = inkling2_ref::depth_mix(x, q, 1e-6f);
  for (size_t i = 0; i < mix.size(); ++i) CHECK_EQ(bits(mix[i]), bits(want[i]));

  std::vector<float> bank(2 * H), gamma(H);
  for (size_t i = 0; i < H; ++i) {
    bank[i] = float(int(i % 9) - 4);
    bank[H + i] = float(i % 13);
    gamma[i] = .5f + float(i % 8) / 8.f;
  }
  std::vector<uint16_t> normed(H);
  auto bank_dev = upload(std::span(bank).first(H)), last_dev = upload(std::span(bank).subspan(H));
  auto gamma_dev = upload(gamma), normed_dev = upload(normed);
  auto mix_norm = glue.kernel("ink_depth_mix_norm");
  mix_norm.group_size(128);
  mix_norm.arg_ptr(0, bank_dev.ptr());
  mix_norm.arg_ptr(1, query.ptr());
  mix_norm.arg_ptr(2, normed_dev.ptr());
  mix_norm.arg(3, 2u);
  mix_norm.arg(4, 1u);
  mix_norm.arg(5, 1e-6f);
  mix_norm.arg_ptr(6, gamma_dev.ptr());
  mix_norm.arg_ptr(7, nullptr);
  mix_norm.arg_ptr(8, nullptr);
  mix_norm.arg_ptr(9, last_dev.ptr());
  list.launch(mix_norm, 1, 1);
  list.copy(normed.data(), normed_dev.ptr(), normed_dev.size());
  const auto mixed = inkling2_ref::depth_mix(bank, q, 1e-6f);
  const float inv = 1.f / std::sqrt(inkling2_ref::sqnorm(mixed) / H + 1e-6f);
  for (size_t i = 0; i < H; ++i)
    CHECK_EQ(normed[i], common::f32_to_bf16((mixed[i] * inv) * gamma[i]));

  std::vector<float> gain(H), forget(H), message(H);
  for (size_t i = 0; i < H; ++i) {
    gain[i] = i % 2 ? 100.f : -100.f;
    forget[i] = i % 3 ? 0.f : 100.f;
    message[i] = float(i % 11);
  }
  auto gains = upload(gain), forgets = upload(forget), messages = upload(message);
  auto controller = glue.kernel("ink_controller_blend");
  controller.group_size(128);
  controller.arg_ptr(0, input.ptr());
  controller.arg_ptr(1, messages.ptr());
  controller.arg_ptr(2, forgets.ptr());
  controller.arg_ptr(3, gains.ptr());
  controller.arg_ptr(4, out.ptr());
  list.launch(controller, H / 128, 1);
  list.copy(mix.data(), out.ptr(), out.size());
  for (size_t i = 0; i < H; ++i)
    CHECK_EQ(bits(mix[i]), bits(inkling2_ref::controller(x[i], message[i], forget[i], gain[i])));

  std::vector<float> hist(3 * H), taps(4 * H);
  for (size_t i = 0; i < hist.size(); ++i) hist[i] = float(i % 11);
  for (size_t i = 0; i < taps.size(); ++i) taps[i] = (int(i % 7) - 3) * .125f;
  auto history = upload(hist), weights = upload(taps);
  auto post = glue.kernel("ink_post_conv");
  post.group_size(128);
  post.arg_ptr(0, input.ptr());
  post.arg_ptr(1, history.ptr());
  post.arg_ptr(2, weights.ptr());
  post.arg_ptr(3, nullptr);
  post.arg_ptr(4, out.ptr());
  post.arg_ptr(5, normed_dev.ptr());
  list.launch(post, H / 128, 1);
  list.copy(mix.data(), out.ptr(), out.size());
  list.copy(normed.data(), normed_dev.ptr(), normed_dev.size());
  for (size_t i = 0; i < mix.size(); ++i) {
    const float expected = inkling2_ref::convolve(x, hist, H, 0, i, taps);
    CHECK_EQ(bits(mix[i]), bits(expected));
    CHECK_EQ(normed[i], common::f32_to_bf16(expected));
  }

  constexpr uint32_t K = ink::D::kv_n, D = ink::D::head_dim;
  std::vector<float> kv(K), kv_hist(3 * K), kv_taps(4 * K);
  std::copy_n(x.begin(), K, kv.begin());
  std::copy_n(hist.begin(), 3 * K, kv_hist.begin());
  std::copy_n(taps.begin(), 4 * K, kv_taps.begin());
  auto kin = upload(kv), kh = upload(kv_hist), kw = upload(kv_taps);
  std::vector<uint16_t> cache(K);
  auto ko = upload(cache);
  std::vector<runtime::Control> control(1);
  control[0].n_active = 1;
  auto ctrl = upload(control);
  l0::Module prep(ctx, kernels::path(std::string(ink::kPrep)));
  auto prepare_kv = prep.kernel("ink_kv_prep");
  prepare_kv.group_size(D);
  prepare_kv.arg_ptr(0, kin.ptr());
  prepare_kv.arg_ptr(1, kh.ptr());
  prepare_kv.arg_ptr(2, kw.ptr());
  prepare_kv.arg_ptr(3, nullptr);
  prepare_kv.arg_ptr(4, ko.ptr());
  prepare_kv.arg_ptr(5, ctrl.ptr());
  prepare_kv.arg(6, 0u);
  prepare_kv.arg(8, 1e-6f);
  prepare_kv.arg(9, 2.f);
  for (const uint32_t key : {0u, 1u}) {
    prepare_kv.arg(7, key);
    list.launch(prepare_kv, ink::D::kv_heads, 1);
    list.copy(cache.data(), ko.ptr(), ko.size());
    for (uint32_t h = 0; h < ink::D::kv_heads; ++h) {
      std::array<float, D> expected;
      for (uint32_t i = 0; i < D; ++i)
        expected[i] = inkling2_ref::convolve(kv, kv_hist, K, 0, h * D + i, kv_taps);
      inkling2_ref::head_scale(expected, 2.f, 1e-6f, key == 0);
      for (uint32_t i = 0; i < D; ++i)
        CHECK_EQ(cache[h * D + i], common::f32_to_bf16(expected[i]));
    }
  }

  constexpr uint32_t visits = 2, verify_rows = 3, accepted = 2;
  constexpr std::array<uint32_t, 4> widths{K, K, H, H};
  std::vector<float> pending(visits * verify_rows * 2 * (K + H));
  for (size_t i = 0; i < pending.size(); ++i) pending[i] = float(i);
  auto pending_dev = upload(pending);
  std::vector<std::vector<float>> history_host;
  std::vector<l0::Mem> history_dev;
  for (auto width : widths) {
    history_host.emplace_back(visits * 3 * width, -1.f);
    history_dev.push_back(upload(history_host.back()));
  }
  auto commit = glue.kernel("ink_commit_conv");
  commit.group_size(128);
  commit.arg_ptr(0, pending_dev.ptr());
  for (uint32_t stream = 0; stream < widths.size(); ++stream) commit.arg_ptr(stream + 1, history_dev[stream].ptr());
  commit.arg(5, verify_rows);
  commit.arg(6, accepted);
  commit.arg_ptr(7, nullptr);
  list.launch(commit, 2 * (K + H) / 128, visits);
  size_t offset = 0;
  for (uint32_t stream = 0; stream < widths.size(); ++stream) {
    const uint32_t width = widths[stream];
    std::vector<float> observed(history_host[stream].size());
    list.copy(observed.data(), history_dev[stream].ptr(), history_dev[stream].size());
    for (uint32_t v = 0; v < visits; ++v) {
      const size_t src = (v * 2 * (K + H) + offset) * verify_rows;
      const size_t dst = v * 3 * width;
      inkling2_ref::commit_conv(std::span(pending).subspan(src, verify_rows * width),
                               std::span(history_host[stream]).subspan(dst, 3 * width), width, accepted);
    }
    CHECK(observed == history_host[stream]);
    offset += width;
  }

  constexpr uint32_t Q = ink::D::q_heads, E = ink::D::relative_extent, R = ink::D::relative_dim;
  std::vector<float> attn_q(Q * D), rel_q(Q * R), rel_table(R * E), sinks(64), gate(Q * D);
  for (uint32_t h = 0; h < Q; ++h) rel_q[h * R] = 1.f;
  rel_table[0] = .5f;
  std::vector<uint16_t> keys(16 * K), values(16 * K, common::f32_to_bf16(2.f)), attention(Q * D);
  std::vector<float> parts(Q * ink::kPartitions * (D + 2));
  auto aq = upload(attn_q), rq = upload(rel_q), rt = upload(rel_table), si = upload(sinks), ga = upload(gate);
  auto ca_k = upload(keys), ca_v = upload(values), ap = upload(parts), ao = upload(attention);
  l0::Module attn(ctx, kernels::path(ink::attn(1, false)));
  auto decode = attn.kernel("kol_attn_decode");
  decode.group_size(D);
  decode.arg_ptr(0, ctrl.ptr());
  decode.arg_ptr(1, aq.ptr());
  decode.arg_ptr(2, ca_k.ptr());
  decode.arg_ptr(3, ca_v.ptr());
  decode.arg_ptr(4, ap.ptr());
  decode.arg_ptr(5, rq.ptr());
  decode.arg_ptr(6, rt.ptr());
  list.launch(decode, ink::D::kv_heads, ink::kPartitions);
  auto reduce = attn.kernel("kol_attn_reduce");
  reduce.group_size(D);
  reduce.arg_ptr(0, ctrl.ptr());
  reduce.arg_ptr(1, ap.ptr());
  reduce.arg_ptr(2, ao.ptr());
  reduce.arg_ptr(3, si.ptr());
  reduce.arg_ptr(4, ga.ptr());
  list.launch(reduce, Q, 1);
  list.copy(attention.data(), ao.ptr(), ao.size());
  for (uint32_t h = 0; h < Q; ++h) {
    const float bias = inkling2_ref::relative_bias(std::span(rel_q).subspan(h * R, R), rel_table, E, 0);
    const auto want_attn = inkling2_ref::gated_value(2.f, 1.f + std::exp(-bias), 0.f);
    for (uint32_t i = 0; i < D; ++i) CHECK_EQ(attention[h * D + i], want_attn);
  }

  std::vector<float> logits(128), bias(128);
  for (uint32_t i = 0; i < 8; ++i) bias[120 + i] = .25f;
  auto logits_dev = upload(logits), bias_dev = upload(bias);
  std::vector<uint32_t> route(32);
  auto route_dev = upload(route);
  l0::Module router(ctx, kernels::path(ink::route()));
  auto route_k = router.kernel("kol_route");
  route_k.group_size(128);
  route_k.arg_ptr(0, logits_dev.ptr());
  route_k.arg_ptr(1, bias_dev.ptr());
  route_k.arg_ptr(2, route_dev.ptr());
  list.launch(route_k, 1, 1);
  list.copy(route.data(), route_dev.ptr(), route_dev.size());
  for (uint32_t i = 0; i < 8; ++i) {
    CHECK_EQ(route[i], 120 + i);
    CHECK_EQ(route[8 + i], bits(1.f));
  }
  CHECK_EQ(route[24], bits(.5f));
  // Overlapping rows share four experts. Compare both output stages with
  // the per-row kernels, including distinct expert ids and both pack formats.
  constexpr uint32_t M = 2, top = ink::D::top_k, inter = ink::D::inter, experts = 12;
  std::vector<uint32_t> routes(M * 32);
  for (uint32_t r = 0; r < M; ++r)
    for (uint32_t slot = 0; slot < top; ++slot) {
      routes[r * 32 + slot] = 4 * r + slot;
      routes[r * 32 + 8 + slot] = bits(float(slot + 1) / 32.f);
    }
  std::vector<uint16_t> expert_x(M * H), expert_h(M * top * inter);
  for (size_t i = 0; i < expert_x.size(); ++i) expert_x[i] = common::f32_to_bf16(float(int(i % 11) - 5) / 32.f);
  std::vector<float> residual(M * H), expert_history(3 * H), expert_taps(4 * H);
  for (size_t i = 0; i < residual.size(); ++i) residual[i] = float(i % 17) / 32.f;
  for (size_t i = 0; i < expert_history.size(); ++i) expert_history[i] = float(i % 13) / 8.f;
  for (size_t i = 0; i < expert_taps.size(); ++i) expert_taps[i] = float(int(i % 5) - 2) / 32.f;
  auto er = upload(routes), ex = upload(expert_x), eh = upload(expert_h), uh = upload(expert_h);
  auto y = upload(residual), raw = upload(residual), rs = upload(residual);
  auto hh = upload(expert_history), et = upload(expert_taps);
  l0::Mem plan(ctx, l0::MemKind::Device, M * top * 2 * sizeof(uint32_t));
  std::vector<uint16_t> output_taps(M * 4 * H);
  auto tap_dev = upload(output_taps);
  for (const auto format : {common::ExpertFpFormat::Nvfp4, common::ExpertFpFormat::Mxfp8}) {
    const auto packed = [&](uint32_t k, uint32_t n) {
      const bool nv = format == common::ExpertFpFormat::Nvfp4;
      const auto words = common::expert_fp_tile_words(format);
      const uint32_t tiles = n / 16 * (k / 64), stride = tiles * words + nv;
      std::vector<uint32_t> w(experts * stride);
      for (uint32_t e = 0; e < experts; ++e) {
        for (uint32_t t = 0; t < tiles; ++t) {
          auto* tile = w.data() + e * stride + t * words;
          const uint32_t values = 32 * common::expert_fp_code_bits(format);
          for (uint32_t i = 0; i < values; ++i)
            tile[i] = nv ? ((t + i + e) % 2 ? 0x12395921u : 0x94122219u)
                         : ((t + i + e) % 2 ? 0x18249c20u : 0x22981919u);
          std::fill(tile + values, tile + words, nv ? 0x30303030u : 0x7c7c7c7cu);
        }
        if (nv) w[(e + 1) * stride - 1] = bits(.5f + float(e) / 32.f);
      }
      return upload(w);
    };
    auto up = packed(H, inter * 2), down = packed(inter, H);
    l0::Module moe(ctx, kernels::path(ink::moe(M, format)));
    auto run = [&](const char* name, uint32_t wg, uint32_t gx, uint32_t gy, std::initializer_list<void*> args) {
      auto kernel = moe.kernel(name);
      kernel.group_size(wg);
      uint32_t j = 0;
      for (auto ptr : args) kernel.arg_ptr(j++, ptr);
      list.launch(kernel, gx, gy);
    };
    run("kol_moe_gate_up", 256, top * inter / 32, M, {er.ptr(), ex.ptr(), up.ptr(), nullptr, eh.ptr()});
    run("kol_moe_down", 128, H / 16, M, {er.ptr(), eh.ptr(), down.ptr(), nullptr, y.ptr()});
    run("kol_moe_union_plan", 128, 1, 1, {er.ptr(), plan.ptr()});
    run("kol_moe_union_gate_up", 256, inter / 32, top * M, {plan.ptr(), ex.ptr(), up.ptr(), uh.ptr()});
    l0::Mem contrib(ctx, l0::MemKind::Device, M * top * H * sizeof(float));
    run("kol_moe_union_down", 64, H / 64, top * M, {plan.ptr(), uh.ptr(), down.ptr(), contrib.ptr()});
    list.copy(rs.ptr(), residual.data(), rs.size());
    run("kol_moe_union_finish", 128, H / 128, 1, {er.ptr(), contrib.ptr(), hh.ptr(), et.ptr(), raw.ptr(), rs.ptr(), tap_dev.ptr()});

    std::vector<uint16_t> union_h(expert_h.size());
    list.copy(expert_h.data(), eh.ptr(), eh.size());
    list.copy(union_h.data(), uh.ptr(), uh.size());
    CHECK(expert_h == union_h);
    std::vector<float> expected(residual.size()), union_y(residual.size());
    list.copy(expected.data(), y.ptr(), y.size());
    list.copy(union_y.data(), raw.ptr(), raw.size());
    for (size_t i = 0; i < expected.size(); ++i) CHECK_EQ(bits(expected[i]), bits(union_y[i]));
    list.copy(union_y.data(), rs.ptr(), rs.size());
    list.copy(output_taps.data(), tap_dev.ptr(), tap_dev.size());
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t i = 0; i < H; ++i) {
        const size_t at = r * H + i;
        const float conv = inkling2_ref::convolve(expected, expert_history, H, r, i, expert_taps);
        CHECK_EQ(bits(conv + residual[at]), bits(union_y[at]));
        CHECK_EQ(output_taps[r * 4 * H + i], common::f32_to_bf16(union_y[at]));
      }
  }

  {
    // Decode's split mixes and norms (INK_GLUE_SPLIT) against the one-work-group kernels: 2 rows, 3 streams, 4 states.
    constexpr uint32_t rows = 2, streams = 3, count = 4, split = 8;
    uint32_t seed = 12345;
    const auto next = [&](float scale) {
      seed = seed * 1664525u + 1013904223u;
      return (float(seed >> 8) / float(1u << 24) - .5f) * scale;
    };
    const auto values = [&](size_t n, float scale) {
      std::vector<float> v(n);
      for (auto& x : v) x = next(scale);
      return v;
    };
    auto states_dev = upload(values((count - 1) * rows * H, 4.f)), last_dev = upload(values(rows * H, 4.f));
    auto queries_dev = upload(values(streams * H, 1.f)), messages_dev = upload(values(streams * rows * H, 3.f));
    auto fg_dev = upload(values(2 * streams * rows * H, 4.f));
    std::vector<float> g(3 * H);
    for (auto& x : g) x = 1.f + next(1.f);
    auto g0 = upload(std::span(g).first(H)), g1 = upload(std::span(g).subspan(H, H)), g2 = upload(std::span(g).subspan(2 * H));
    l0::Module ctrl_glue(ctx, kernels::path(std::string(ink::kControllerGlue)));
    l0::Module split_glue(ctx, kernels::path(std::string(ink::kSplitGlue)));
    const auto run = [&](l0::Module& module, const char* entry, uint32_t group, uint32_t gx, uint32_t gy, uint32_t gz,
                         auto... args) {
      auto k = module.kernel(entry);
      k.group_size(group);
      uint32_t i = 0;
      const auto set = [&](auto a) {
        if constexpr (std::is_pointer_v<decltype(a)> || std::is_null_pointer_v<decltype(a)>) k.arg_ptr(i++, a);
        else k.arg(i++, a);
      };
      (set(args), ...);
      list.launch(k, gx, gy, gz);
    };
    const auto read = [&](const l0::Mem& m, auto zero) {
      std::vector<decltype(zero)> v(m.size() / sizeof(zero));
      list.copy(v.data(), m.ptr(), m.size());
      return v;
    };
    const auto same = [&](const auto& a, const auto& b) {
      CHECK_EQ(a.size(), b.size());
      for (size_t i = 0; i < a.size(); ++i) CHECK_EQ(a[i], b[i]);
    };
    l0::Mem parts(ctx, l0::MemKind::Device, streams * rows * 10 * 128 * sizeof(float));
    l0::Mem norm_parts(ctx, l0::MemKind::Device, streams * rows * 2 * 128 * sizeof(float));
    const float eps = 1e-6f;
    void* st = states_dev.ptr(), *la = last_dev.ptr(), *qu = queries_dev.ptr(), *me = messages_dev.ptr();
    void* pa = parts.ptr(), *np = norm_parts.ptr();
    // ink_depth_mix_norm, three streams
    l0::Mem want(ctx, l0::MemKind::Device, streams * rows * H * 2), got(ctx, l0::MemKind::Device, streams * rows * H * 2);
    l0::Mem mixed(ctx, l0::MemKind::Device, streams * rows * H * 4), mixed2(ctx, l0::MemKind::Device, streams * rows * H * 4);
    run(glue, "ink_depth_mix_norm", 128, rows, streams, 1, st, qu, want.ptr(), count, rows, eps, g0.ptr(), g1.ptr(),
        g2.ptr(), la);
    run(split_glue, "ink_split_sums", 16, split, rows, streams, st, la, qu, pa, count, rows, 0u);
    run(split_glue, "ink_split_mix", 128, H / 128, rows, streams, st, la, pa, mixed.ptr(), count, rows);
    run(split_glue, "ink_split_squares", 16, split, rows, streams, mixed.ptr(), np, rows);
    run(split_glue, "ink_split_norm", 16, split, rows, streams, mixed.ptr(), np, got.ptr(), rows, eps, g0.ptr(),
        g1.ptr(), g2.ptr());
    same(read(want, uint16_t{}), read(got, uint16_t{}));
    // ink_post_mix_norm, one stream, no attention residual
    run(ctrl_glue, "ink_post_mix_norm", 128, rows, 1, 1, st, la, qu, want.ptr(), g0.ptr(), count, rows, eps, nullptr,
        nullptr, nullptr, nullptr);
    run(split_glue, "ink_split_sums", 16, split, rows, 1, st, la, qu, pa, count, rows, 0u);
    run(split_glue, "ink_split_mix", 128, H / 128, rows, 1, st, la, pa, mixed.ptr(), count, rows);
    run(split_glue, "ink_split_squares", 16, split, rows, 1, mixed.ptr(), np, rows);
    run(split_glue, "ink_split_norm", 16, split, rows, 1, mixed.ptr(), np, got.ptr(), rows, eps, g0.ptr(), nullptr,
        nullptr);
    same(read(want, uint16_t{}), read(got, uint16_t{}));
    // ink_mix_cat (a shared and a per-stream query), then ink_controller_norm on its mixed rows
    for (const uint32_t shared : {1u, 0u}) {
      l0::Mem cat(ctx, l0::MemKind::Device, streams * rows * 2 * H * 2), cat2(ctx, l0::MemKind::Device, cat.size());
      run(ctrl_glue, "ink_mix_cat", 128, rows, streams, 1, st, la, qu, me, mixed.ptr(), cat.ptr(), count, rows, shared,
          eps, nullptr, nullptr, nullptr, nullptr);
      run(split_glue, "ink_split_sums", 16, split, rows, streams, st, la, qu, pa, count, rows, shared);
      run(split_glue, "ink_split_mix_cat", 128, H / 128, rows, streams, st, la, pa, mixed2.ptr(), count, rows);
      run(split_glue, "ink_split_cat_squares", 16, split, rows, streams, mixed2.ptr(), me, np, rows);
      run(split_glue, "ink_split_cat", 16, split, rows, streams, mixed2.ptr(), me, np, cat2.ptr(), rows, eps);
      same(read(mixed, uint32_t{}), read(mixed2, uint32_t{}));
      same(read(cat, uint16_t{}), read(cat2, uint16_t{}));
      run(ctrl_glue, "ink_controller_norm", 128, rows, streams, 1, mixed.ptr(), me, fg_dev.ptr(), want.ptr(), g0.ptr(),
          g1.ptr(), g2.ptr(), rows, eps);
      run(split_glue, "ink_split_blend", 128, H / 128, rows, streams, mixed2.ptr(), me, fg_dev.ptr(), rows, streams);
      run(split_glue, "ink_split_squares", 16, split, rows, streams, mixed2.ptr(), np, rows);
      run(split_glue, "ink_split_norm", 16, split, rows, streams, mixed2.ptr(), np, got.ptr(), rows, eps, g0.ptr(),
          g1.ptr(), g2.ptr());
      same(read(want, uint16_t{}), read(got, uint16_t{}));
    }
    // The router: the fused one-work-group kernel against the per-tile logits and the unfused selection.
    std::vector<uint16_t> rw(128 * H), rx(rows * H);
    for (auto& v : rw) v = common::f32_to_bf16(next(.2f));
    for (auto& v : rx) v = common::f32_to_bf16(next(2.f));
    auto rw_dev = upload(rw), rx_dev = upload(rx), rb_dev = upload(values(128, .1f));
    l0::Mem fused(ctx, l0::MemKind::Device, rows * 32 * 4), split_route(ctx, l0::MemKind::Device, rows * 32 * 4);
    l0::Mem logits(ctx, l0::MemKind::Device, rows * 128 * 4);
    list.fill(fused.ptr(), 0, fused.size());
    list.fill(split_route.ptr(), 0, split_route.size());
    l0::Module fused_route(ctx, kernels::path(ink::route() + "_M1_FUSED")), plain_route(ctx, kernels::path(ink::route()));
    run(fused_route, "kol_route", 128 * ink::kRouteSplit, 1, rows, 1, rw_dev.ptr(), rx_dev.ptr(), rb_dev.ptr(), fused.ptr());
    run(fused_route, "kol_route_logits", 16 * ink::kRouteSplit, 8, rows, 1, rw_dev.ptr(), rx_dev.ptr(), logits.ptr());
    run(plain_route, "kol_route", 128, 1, rows, 1, logits.ptr(), rb_dev.ptr(), split_route.ptr());
    same(read(fused, uint32_t{}), read(split_route, uint32_t{}));
    // The controllers' GEMVs on the DPAS: each row of the 8-row build is the one-row build's bits (down: the cat rows
    // into the rank, bf16 out; up: the rank into the forget and gain rows, fp32 out).
    {
      constexpr uint32_t R = 8, rank = ink::D::controller_rank;
      std::vector<uint16_t> dw(rank * 2 * H), uw(2 * H * rank), cx(R * 2 * H), lx(R * rank);
      for (auto* v : {&dw, &uw, &cx, &lx})
        for (auto& x : *v) x = common::f32_to_bf16(next(.5f));
      auto dw_dev = upload(dw), uw_dev = upload(uw), cx_dev = upload(cx), lx_dev = upload(lx);
      const auto* gain = static_cast<const uint16_t*>(uw_dev.ptr()) + size_t(H) * rank;
      l0::Mem down_r(ctx, l0::MemKind::Device, R * rank * 2), down_1(ctx, l0::MemKind::Device, R * rank * 2);
      l0::Mem up_r(ctx, l0::MemKind::Device, 2 * R * H * 4), up_1(ctx, l0::MemKind::Device, 2 * R * H * 4);
      l0::Module dn8(ctx, kernels::path(ink::controller(R, true))), dn1(ctx, kernels::path(ink::controller(1, true)));
      l0::Module up8(ctx, kernels::path(ink::controller(R, false))), up1(ctx, kernels::path(ink::controller(1, false)));
      const uint32_t dg = ink::kControllerDown.cols * ink::kControllerDown.ksplit, dx = rank / ink::kControllerDown.cols;
      const void* none = nullptr;
      run(dn8, "gemv_bf16", dg, dx, 1, 1, dw_dev.ptr(), none, none, none, none, none, cx_dev.ptr(), down_r.ptr());
      run(up8, "gemv_bf16", 64, H / 64, 1, 2, uw_dev.ptr(), none, none, gain, none, none, lx_dev.ptr(), up_r.ptr());
      std::vector<float> up_rows(2 * R * H);
      for (uint32_t r = 0; r < R; ++r) {
        run(dn1, "gemv_bf16", dg, dx, 1, 1, dw_dev.ptr(), none, none, none, none, none,
            static_cast<const uint16_t*>(cx_dev.ptr()) + size_t(r) * 2 * H, static_cast<uint16_t*>(down_1.ptr()) + r * rank);
        run(up1, "gemv_bf16", 64, H / 64, 1, 2, uw_dev.ptr(), none, none, gain, none, none,
            static_cast<const uint16_t*>(lx_dev.ptr()) + size_t(r) * rank, up_1.ptr());
        const auto one = read(up_1, float{});
        for (uint32_t g = 0; g < 2; ++g)
          std::copy_n(one.begin() + size_t(g) * H, H, up_rows.begin() + (size_t(g) * R + r) * H);
      }
      same(read(down_r, uint16_t{}), read(down_1, uint16_t{}));
      const auto up8_bits = read(up_r, float{});
      for (size_t i = 0; i < up_rows.size(); ++i) CHECK_EQ(bits(up8_bits[i]), bits(up_rows[i]));
    }
  }

  std::puts("Inkling GPU depth mix, controller blend, short convolution, KV, relative attention and routing match host bits. Expert unions match per-row bits. Split mixes and norms match the one-work-group kernels' bits. The controllers' verify rows match the one-row GEMVs' bits.");
}

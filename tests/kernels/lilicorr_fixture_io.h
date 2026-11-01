#pragma once
// Spec 26b: tests/loader/lilicorr_dflash_fixture.safetensors (tools/oracle/dflash_fixture.py --lilicorr) read into the
// LiLiCorr twin's structures (tests/kernels/lilicorr_ref.h) - shared by lilicorr_ref_test (the twin vs dflash_ref.py's
// LiLiCorr form) and tools/mac/clrun/lilicorr_run.cc (the kernels at the fixture's tiny shapes on the Mac's GPU). The
// fixture carries the tiny checkpoint's own tensors (`ckpt.*`); they are written back out as a checkpoint directory and
// read by the loader's host half (loader::LiliCheckpoint) - so the twin runs on exactly the device forms the loader builds.
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common/json.h"
#include "common/repack.h"
#include "kernels/dflash_fixture_io.h"
#include "kernels/lilicorr_ref.h"
#include "loader/dflash_fixture.h"
#include "loader/lilicorr_repack.h"
#include "loader/nvfp4.h"
#include "loader/rtn.h"

namespace llfix {

using dffix::Fixture;

// The backbone's form: Reference (fp32 W', g on the output), or a device format.
enum class Form { Reference, Exact, Int8, Int4 };
// The target head's: Reference (fp32), Bf16 (gemv_bf16's tiles), Int8 (spec 9).
enum class HeadForm { Reference, Bf16, Int8 };

// gemv_bf16's tiling rule (kernels::gemv_bf16_tiling, which needs B70_KERNEL_DIR): N / 64 >= 32 work-groups -> KSPLIT 1.
inline uint32_t bf16_ksplit(uint32_t N) { return N / 64 >= 32 ? 1u : 16u; }

// The fixture's `ckpt.*` tensors + config_lilicorr as a checkpoint directory (one per process and tag).
inline std::string write_checkpoint(const Fixture& fx, const std::string& tag) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / ("lilicorr_fixture_" + tag + "_" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  dflash_fixture::Tensors ts;
  for (const auto& [name, ti] : fx.t) {
    if (name.rfind("ckpt.", 0) != 0) continue;
    dflash_fixture::Tensor t{ti.dtype, ti.shape, std::vector<uint8_t>(ti.end - ti.begin)};
    std::memcpy(t.bytes.data(), fx.raw(name), t.bytes.size());
    ts[name.substr(5)] = std::move(t);
  }
  dflash_fixture::write_safetensors(dir / "model.safetensors", ts);
  std::ofstream(dir / "config.json") << fx.meta_str("config_lilicorr");
  return dir.string();
}

// One backbone linear (its parts' bases, concatenated by rows) in `f`, through the loader's own pieces where it has them.
inline llref::GLin glin(const loader::LiliCheckpoint& ck, const std::vector<std::string>& bases, Form f,
                        const loader::DFlashHostLinear* dev) {
  llref::GLin g;
  std::vector<loader::LiliDecoded> parts;
  for (const std::string& b : bases) parts.push_back(ck.decoded(b));
  const uint32_t K = parts[0].K;
  uint32_t N = 0;
  bool any = false;
  for (const loader::LiliDecoded& p : parts) {
    N += p.N;
    any |= p.nvfp4;
  }
  g.lin.K = K;
  g.lin.N = N;
  if (f == Form::Reference) {
    for (const loader::LiliDecoded& p : parts) {
      for (uint16_t v : p.wp) g.lin.f32.push_back(dfref::bf(v));
      if (any) g.gs.insert(g.gs.end(), p.N, p.g);
    }
  } else if (f == Form::Exact) {
    g.lin.bf16 = dev->bf16;
    g.lin.ksplit = bf16_ksplit(N);
    g.gs = dev->gs;
  } else if (f == Form::Int8) {
    g.lin.i8 = dev->i8;
    g.lin.scales = dev->scales;
  } else {   // GPTQ-native (dfref::gemv_int4 reads it): each part by its rule, columns concatenated
    g.lin.q4.assign(size_t(K / 8) * N, 0);
    g.lin.s4.assign(size_t(K / 64) * N, 0);
    uint32_t n0 = 0;
    for (const loader::LiliDecoded& p : parts) {
      std::vector<uint32_t> qw(size_t(K / 8) * p.N);
      std::vector<uint16_t> sc(size_t(K / 64) * p.N);
      if (p.nvfp4)
        loader::rtn_int4_g64_g(p.wp.data(), double(p.g), K, p.N, qw.data(), sc.data());
      else
        loader::rtn_int4_g64(p.wp.data(), K, p.N, qw.data(), sc.data());
      for (uint32_t r = 0; r < K / 8; ++r)
        for (uint32_t n = 0; n < p.N; ++n) g.lin.q4[size_t(r) * N + n0 + n] = qw[size_t(r) * p.N + n];
      for (uint32_t r = 0; r < K / 64; ++r)
        for (uint32_t n = 0; n < p.N; ++n) g.lin.s4[size_t(r) * N + n0 + n] = sc[size_t(r) * p.N + n];
      n0 += p.N;
    }
  }
  return g;
}

inline loader::DraftFormat device_format(Form f) {
  return f == Form::Int8 ? loader::DraftFormat::Int8 : f == Form::Int4 ? loader::DraftFormat::Int4G64
                                                                        : loader::DraftFormat::Exact;
}

inline llref::HeadLin head_lin(const loader::LiliHostHead& h) {
  llref::HeadLin o;
  o.K = h.K;
  o.N = h.N;
  o.wt.resize(h.wt.size());
  for (size_t i = 0; i < h.wt.size(); ++i) o.wt[i] = dfref::bf(h.wt[i]);
  o.gs = h.gs;
  o.bias = h.bias;
  return o;
}

// The LiLiCorr twin's drafter weights from `ck` in `f` (the target - embedding, head, mask row - left to the caller: the
// fixture's, or a card test's synthetic one; V / used = the config's vocab), the ring geometry (ring = window; `pb` offsets
// per context block). `threads`: the loader's quantiser threads (the result does not depend on it).
inline llref::Weights drafter_weights(const loader::LiliCheckpoint& ck, Form f, uint32_t pb, unsigned threads = 1) {
  const model::DFlashDesc& d = ck.desc();
  llref::Weights w;
  llref::Dims& D = w.d;
  D.version = 2;
  D.layers = d.layers;
  D.H = d.hidden;
  D.NH = d.q_heads;
  D.NKV = d.kv_heads;
  D.HD = d.head_dim;
  D.I = d.inter;
  D.V = d.vocab;
  D.used = d.vocab;
  D.M = d.block;
  D.ring = d.max_window();
  D.win = d.max_window();
  D.pb = pb;
  D.taps = uint32_t(d.taps.size());
  D.th = ck.target_hidden();
  D.group = d.conv_group;
  D.ctaps = d.conv_taps;
  D.rank = 0;
  D.topk = d.cand_topk;
  D.eps = d.eps;
  D.in_scale = d.in_scale;
  D.scale = float(1.0 / std::sqrt(double(d.head_dim)));
  D.theta = d.theta;
  D.S = d.lil_slots();
  D.k = d.cand_topk;
  D.Hl = d.lil_hidden;
  D.lheads = d.lil_heads;
  D.lmlp = d.lil_mlp;
  D.F = d.factor_dim;
  D.llayers = d.lil_layers;
  D.veps = d.vector_eps;
  D.lscale = d.lil_logit_scale;
  D.token_proj = loader::lili_has_token_proj(d);
  const loader::DraftFormat df = device_format(f);
  auto dev = [&](const loader::DFlashHostLinear& h) { return std::make_unique<loader::DFlashHostLinear>(h); };
  std::unique_ptr<loader::DFlashHostLinear> fcd = f == Form::Reference || f == Form::Int4 ? nullptr : dev(ck.fc(df, threads));
  w.fc = glin(ck, {"fc"}, f, fcd.get());
  w.hidden_norm = ck.hidden_norm();
  w.final_norm = ck.final_norm();
  const std::vector<float> qk = ck.qk_norms(), cb = ck.conv_bases(), sk = ck.sinks();
  const size_t cbl = size_t(2) * 2 * d.conv_taps * d.hidden;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    llref::Layer y;
    auto one = [&](loader::DFlashLinear which, const std::vector<std::string>& bases) {
      std::unique_ptr<loader::DFlashHostLinear> h =
          f == Form::Reference || f == Form::Int4 ? nullptr : dev(ck.layer_linear(l, which, df, threads));
      return glin(ck, bases, f, h.get());
    };
    using DL = loader::DFlashLinear;
    y.qkv = one(DL::Qkv, {L + "self_attn.q_proj", L + "self_attn.k_proj", L + "self_attn.v_proj"});
    y.o = one(DL::O, {L + "self_attn.o_proj"});
    y.gate_up = one(DL::GateUp, {L + "mlp.gate_proj", L + "mlp.up_proj"});
    y.down = one(DL::Down, {L + "mlp.down_proj"});
    y.akp = one(DL::AttnKproj, {L + "attention_conv.kernel_projection"});
    y.mkp = one(DL::MlpKproj, {L + "mlp_conv.kernel_projection"});
    const std::vector<float> n = ck.layer_norms(l);
    y.in_norm.assign(n.begin(), n.begin() + d.hidden);
    y.post_norm.assign(n.begin() + d.hidden, n.end());
    y.q_norm.assign(qk.begin() + long(l * 2 * d.head_dim), qk.begin() + long((l * 2 + 1) * d.head_dim));
    y.k_norm.assign(qk.begin() + long((l * 2 + 1) * d.head_dim), qk.begin() + long((l * 2 + 2) * d.head_dim));
    y.conv_base.assign(cb.begin() + long(l * cbl), cb.begin() + long((l + 1) * cbl));
    y.sink.assign(sk.begin() + long(l * d.q_heads), sk.begin() + long((l + 1) * d.q_heads));
    w.layers.push_back(std::move(y));
  }
  // the LiLiCorr head (always exact)
  using LH = loader::LiliHead;
  llref::Head& h = w.lil;
  if (D.token_proj) h.tok = head_lin(ck.head_linear(LH::TokenProj));
  h.pass = head_lin(ck.head_linear(LH::PassHidden));
  h.fup = head_lin(ck.head_linear(LH::FeatUp));
  h.fdown = head_lin(ck.head_linear(LH::FeatDown));
  h.ctx = head_lin(ck.head_linear(LH::ContextProj));
  h.factor = head_lin(ck.head_linear(LH::Factor));
  h.outin = head_lin(ck.head_linear(LH::OutIn));
  h.aout = head_lin(ck.head_linear(LH::AnchorOut));
  for (uint32_t l = 0; l < d.lil_layers; ++l) {
    h.inproj.push_back(head_lin(ck.head_linear(LH::InProj, l)));
    h.outproj.push_back(head_lin(ck.head_linear(LH::OutProj, l)));
    h.up.push_back(head_lin(ck.head_linear(LH::MlpUp, l)));
    h.down.push_back(head_lin(ck.head_linear(LH::MlpDown, l)));
  }
  const std::vector<float> sm = ck.head_small();
  const loader::LiliHeadSmall o = loader::lili_head_small(d);
  auto take = [&](size_t at, size_t n) { return std::vector<float>(sm.begin() + long(at), sm.begin() + long(at + n)); };
  h.fn_w = take(o.fn_w, 5);
  h.fn_b = take(o.fn_b, 5);
  h.slot = take(o.slot, size_t(D.S) * D.Hl);
  h.rank = take(o.rank, size_t(D.k) * D.Hl);
  h.rel = take(o.rel, size_t(D.lheads) * (2 * D.M - 1));
  h.same = take(o.same, D.lheads);
  for (uint32_t l = 0; l < d.lil_layers; ++l) {
    const std::vector<float> a = take(o.attn_norm(l, D.Hl), D.Hl), m = take(o.mlp_norm(l, D.Hl), D.Hl);
    h.attn_norm.insert(h.attn_norm.end(), a.begin(), a.end());
    h.mlp_norm.insert(h.mlp_norm.end(), m.begin(), m.end());
  }
  h.out_norm = take(o.out_norm, D.Hl);
  h.anchor_norm = take(o.anchor_norm, D.Hl);
  return w;
}

// The target's half: its bf16 [V][H] embedding (row-major) and head (`head` row-major [V][H], in `hf`); the mask row.
inline void set_target(llref::Weights& w, uint32_t mask_id, std::vector<uint16_t> embed, const std::vector<uint16_t>& head,
                       HeadForm hf, uint32_t used) {
  llref::Dims& D = w.d;
  D.V = uint32_t(head.size() / D.H);
  D.used = used;
  const dffix::Form hff = hf == HeadForm::Reference ? dffix::Form::Reference
                          : hf == HeadForm::Bf16    ? dffix::Form::Bf16
                                                    : dffix::Form::Int8;
  w.head = dffix::make_lin(head, D.H, D.V, hff, bf16_ksplit(D.V));
  w.embed = std::move(embed);
  w.mask_row.resize(D.H);
  for (uint32_t i = 0; i < D.H; ++i) w.mask_row[i] = dfref::bf(w.embed[size_t(mask_id) * D.H + i]);
}

// The fixture's whole twin: the drafter from `ck` in `f`, the fixture's target (embedding, head in `hf`).
inline llref::Weights weights(const Fixture& fx, const loader::LiliCheckpoint& ck, Form f, HeadForm hf, uint32_t pb) {
  llref::Weights w = drafter_weights(ck, f, pb);
  set_target(w, ck.desc().mask_id, fx.bf16("target.embed"), fx.bf16("target.head"), hf,
             uint32_t(std::stoul(fx.meta_str("vocab_used"))));
  return w;
}

// The ring after projecting positions 0 .. p - 1 (8 rows a replay, the projection list's limit) and the anchor (position
// p - 1's row; zeros and invalid at p 0).
struct RingAt {
  dfref::Ring ring;
  std::vector<float> anchor;
  bool valid = false;
};
inline RingAt ring_at(const Fixture& fx, const llref::Weights& w, dfref::Numerics nm, const std::vector<float>& rope,
                      uint32_t p) {
  RingAt r;
  r.ring = dfref::make_ring(w.d);
  r.anchor.assign(w.d.H, 0.0f);
  const std::vector<uint16_t> taps = fx.bf16("taps");
  for (uint32_t pos = 0; pos < p; pos += 8) {
    const uint32_t n = std::min(8u, p - pos);
    r.anchor = llref::project(w, nm, &taps[size_t(pos) * w.d.fck()], n, pos, rope, r.ring);
    r.valid = true;
  }
  return r;
}

}  // namespace llfix

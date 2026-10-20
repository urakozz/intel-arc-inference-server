// lfm_step_shapes.h - spec 24c P0 (plan 24c Task 1): LFM2.5-24B-A2B's decode step as a launch list,
// device-free, so tests/probe/probe_lfm_step_shapes_test.cc holds the counts on the Mac and
// tools/probe/probe_lfm_step.cc binds every slot to an EXISTING kernel at LFM2's shapes (no LFM2
// kernel exists when P0 runs - spec 24 §4.5: "P0 comes first and measures it").
//
// The list is spec 24 §4.5's, slot for slot (model::Lfm2MoeDesc decides each layer's kind):
//
//   dense conv layer (0-1), 8   norm · in_proj · conv · out_proj · norm · gate||up · silu · down
//   conv + MoE layer, 9         norm · in_proj · conv · out_proj · norm · router · route · gate||up · down
//   attention + MoE layer, 11   norm · q||k||v · attn_prep · attn · attn_reduce · o_proj · norm · router ·
//                               route · gate||up · down
//   head, 5                     embed (first) · norm (embedding_norm) · head GEMV · argmax 1 · argmax 2
//
// 383 a token at 40 layers (2 x 8 + 28 x 9 + 10 x 11 + 5); `Fused` drops one conv slot per conv layer
// (arm conv: the conv as the in_proj GEMV's epilogue), one route and one router slot per MoE layer (arms
// route and router): 277. `Gemv` keeps only the GEMV-family slots (int4 GEMVs, the bf16 router, the
// head, the MoE gate||up / down) - the list's bytes floor. `Noop` is 383 noops (probe_replay's floor at
// this count).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "model/lfm2moe.h"

namespace probe_lfm {

enum class Fam {
  Embed, Norm, InProj, Conv, OutProj, Qkv, AttnPrep, Attn, AttnReduce, OProj, Router, Route, GateUp, Down,
  DenseGateUp, Silu, DenseDown, Head, Argmax1, Argmax2, Noop, kCount
};
enum class List { A, Fused, Gemv, Noop };

struct Launch {
  Fam fam;
  int layer;   // -1: the head's launches (no layer)
};

inline const char* fam_name(Fam f) {
  switch (f) {
    case Fam::Embed: return "embed";
    case Fam::Norm: return "norm";
    case Fam::InProj: return "in_proj";
    case Fam::Conv: return "conv";
    case Fam::OutProj: return "out_proj";
    case Fam::Qkv: return "q||k||v";
    case Fam::AttnPrep: return "attn_prep";
    case Fam::Attn: return "attn";
    case Fam::AttnReduce: return "attn_reduce";
    case Fam::OProj: return "o_proj";
    case Fam::Router: return "router";
    case Fam::Route: return "route";
    case Fam::GateUp: return "moe gate||up";
    case Fam::Down: return "moe down";
    case Fam::DenseGateUp: return "dense gate||up";
    case Fam::Silu: return "silu_mul";
    case Fam::DenseDown: return "dense down";
    case Fam::Head: return "head";
    case Fam::Argmax1: return "argmax 1";
    case Fam::Argmax2: return "argmax 2";
    case Fam::Noop: return "noop";
    default: return "?";
  }
}
inline const char* list_name(List l) {
  switch (l) {
    case List::A: return "a";
    case List::Fused: return "fused";
    case List::Gemv: return "gemv";
    default: return "noop";
  }
}

// The GEMV-family slots: what the `gemv` list keeps (the bytes a token must stream).
inline bool is_gemv(Fam f) {
  switch (f) {
    case Fam::InProj: case Fam::OutProj: case Fam::Qkv: case Fam::OProj: case Fam::Router: case Fam::GateUp:
    case Fam::Down: case Fam::DenseGateUp: case Fam::DenseDown: case Fam::Head:
      return true;
    default:
      return false;
  }
}

// The int4 GEMVs a layer runs, as {K, N} (spec 24 §3; model::Lfm2MoeDesc::linears' shapes) - the slots the
// skeleton binds to gemv.cl and `--sweep-s` times at every S and layout.
struct Int4Shape {
  Fam fam;
  uint32_t K, N;
};
inline std::vector<Int4Shape> int4_shapes(const model::Lfm2MoeDesc& d) {
  return {{Fam::InProj, d.hidden, 3 * d.hidden},
          {Fam::OutProj, d.hidden, d.hidden},
          {Fam::Qkv, d.hidden, d.qkv_n()},
          {Fam::DenseGateUp, d.hidden, 2 * d.dense_inter},
          {Fam::DenseDown, d.dense_inter, d.hidden}};
}
// o_proj shares out_proj's shape (q_n == hidden): one sweep row serves both.
inline Fam shape_fam(Fam f) { return f == Fam::OProj ? Fam::OutProj : f; }

inline std::vector<Launch> build_list(const model::Lfm2MoeDesc& d, List which) {
  std::vector<Launch> a;
  auto put = [&](Fam f, int l) { a.push_back({f, l}); };
  put(Fam::Embed, -1);
  for (uint32_t l = 0; l < d.layers; ++l) {
    const int L = int(l);
    put(Fam::Norm, L);
    if (d.is_attention(l)) {
      put(Fam::Qkv, L);
      put(Fam::AttnPrep, L);
      put(Fam::Attn, L);
      put(Fam::AttnReduce, L);
      put(Fam::OProj, L);
    } else {
      put(Fam::InProj, L);
      put(Fam::Conv, L);
      put(Fam::OutProj, L);
    }
    put(Fam::Norm, L);
    if (d.is_dense(l)) {
      put(Fam::DenseGateUp, L);
      put(Fam::Silu, L);
      put(Fam::DenseDown, L);
    } else {
      put(Fam::Router, L);
      put(Fam::Route, L);
      put(Fam::GateUp, L);
      put(Fam::Down, L);
    }
  }
  put(Fam::Norm, -1);
  put(Fam::Head, -1);
  put(Fam::Argmax1, -1);
  put(Fam::Argmax2, -1);
  if (which == List::A) return a;
  std::vector<Launch> out;
  for (const Launch& x : a) {
    if (which == List::Noop) {
      out.push_back({Fam::Noop, x.layer});
    } else if (which == List::Gemv) {
      if (is_gemv(x.fam)) out.push_back(x);
    } else {   // Fused: every arm of spec 24 §4.5 applied (route, router, conv)
      if (x.fam == Fam::Conv || x.fam == Fam::Route || x.fam == Fam::Router) continue;
      out.push_back(x);
    }
  }
  return out;
}

// Spec 24 §4.5's derived counts: 2 x 8 + 28 x 9 + 10 x 11 + 5 at the real 40 layers.
inline size_t derived_a(const model::Lfm2MoeDesc& d) {
  size_t n = 5;
  for (uint32_t l = 0; l < d.layers; ++l) n += d.is_dense(l) ? 8 : (d.is_attention(l) ? 11 : 9);
  return n;
}
inline size_t derived_fused(const model::Lfm2MoeDesc& d) {
  return derived_a(d) - d.conv_layers() - 2 * size_t(d.moe_layers());
}

}  // namespace probe_lfm

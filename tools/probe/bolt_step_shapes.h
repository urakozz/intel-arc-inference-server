// bolt_step_shapes.h - spec 25c P0 (plan 25c Task 1): Nemotron 3.5 Bolt's decode step as a launch list, device-free,
// so tests/kernels/probe_bolt_step_shapes_test.cc holds the counts on the Mac and tools/probe/probe_bolt_step.cc binds
// every slot to an EXISTING kernel at Bolt's shapes (no nb_* kernel is bound when P0 runs - spec 25 §3: "25c's P0
// measures every term before the default list is chosen").
//
// The list is spec 25 §6.1's, slot for slot (model::BoltDesc decides each block's kind):
//
//   Mamba (40), 6       norm · in_proj · conv · ssm · gnorm · out_proj
//   attention (8), 6    norm · q||k||v · attn_prep · attn · attn_reduce · o_proj
//   MoE (40), 9         norm · router · route · fc1||shared_up · sh_act · moe_up · moe_down · fc2 · shared_down
//   head, 5             embed (first) · norm (norm_f) · head GEMV · argmax 1 · argmax 2
//
// 653 a token at the real 88 blocks (40 x 6 + 8 x 6 + 40 x 9 + 5); `Fused` drops one conv slot a Mamba block (arm conv:
// the conv inside nb_ssm), one router and one route slot a MoE block (arms router and route): 533. `Gemv` keeps the
// GEMV-family slots (the int4 GEMVs, the bf16 routers, the routed experts' up / down, the head) - the list's bytes
// floor. `Noop` is 653 noops (probe_replay's floor at this count). `State` is one read-modify-write of a Mamba block's
// SSM state a Mamba block (40): the state traffic's floor on its own (4 MiB fp32 a block; 2 MiB with --state-bf16).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/bolt.h"

namespace probe_bolt {

enum class Fam {
  Embed, Norm, InProj, Conv, Ssm, Gnorm, OutProj, Qkv, AttnPrep, Attn, AttnReduce, OProj, Router, Route, Fc1Up, ShAct,
  MoeUp, MoeDown, Fc2, ShDown, Head, Argmax1, Argmax2, Noop, State, kCount
};
enum class List { A, Fused, Gemv, Noop, State };

struct Launch {
  Fam fam;
  int layer;   // -1: the head's launches (no block)
};

inline const char* fam_name(Fam f) {
  switch (f) {
    case Fam::Embed: return "embed";
    case Fam::Norm: return "norm";
    case Fam::InProj: return "in_proj";
    case Fam::Conv: return "conv";
    case Fam::Ssm: return "ssm";
    case Fam::Gnorm: return "gnorm";
    case Fam::OutProj: return "out_proj";
    case Fam::Qkv: return "q||k||v";
    case Fam::AttnPrep: return "attn_prep";
    case Fam::Attn: return "attn";
    case Fam::AttnReduce: return "attn_reduce";
    case Fam::OProj: return "o_proj";
    case Fam::Router: return "router";
    case Fam::Route: return "route";
    case Fam::Fc1Up: return "fc1||shared_up";
    case Fam::ShAct: return "sh_act";
    case Fam::MoeUp: return "moe up";
    case Fam::MoeDown: return "moe down";
    case Fam::Fc2: return "fc2";
    case Fam::ShDown: return "shared_down";
    case Fam::Head: return "head";
    case Fam::Argmax1: return "argmax 1";
    case Fam::Argmax2: return "argmax 2";
    case Fam::Noop: return "noop";
    case Fam::State: return "state rmw";
    default: return "?";
  }
}
inline const char* list_name(List l) {
  switch (l) {
    case List::A: return "a";
    case List::Fused: return "fused";
    case List::Gemv: return "gemv";
    case List::Noop: return "noop";
    default: return "state";
  }
}

// The GEMV-family slots: what the `gemv` list keeps (the bytes a token must stream).
inline bool is_gemv(Fam f) {
  switch (f) {
    case Fam::InProj: case Fam::OutProj: case Fam::Qkv: case Fam::OProj: case Fam::Router: case Fam::Fc1Up:
    case Fam::MoeUp: case Fam::MoeDown: case Fam::Fc2: case Fam::ShDown: case Fam::Head:
      return true;
    default:
      return false;
  }
}

// The int4 GEMVs a token runs, as {K, N} (spec 25 §6.1; model::BoltDesc::linears' shapes) - the slots the skeleton
// binds to gemv.cl and `--sweep-s` times at every S and layout. o_proj (K 4096, N 4096) is its own shape.
struct Int4Shape {
  Fam fam;
  uint32_t K, N;
};
inline std::vector<Int4Shape> int4_shapes(const model::BoltDesc& d) {
  return {{Fam::InProj, d.hidden, d.m_inproj},
          {Fam::OutProj, d.m_inner, d.hidden},
          {Fam::Qkv, d.hidden, d.qkv_n()},
          {Fam::OProj, d.q_n(), d.hidden},
          {Fam::Fc1Up, d.hidden, d.latent + d.shared_inter},
          {Fam::Fc2, d.latent, d.hidden},
          {Fam::ShDown, d.shared_inter, d.hidden}};
}
// The descriptor's linear id of a GEMV slot (its {S, layout} cell).
inline model::BoltLinearId linear_of(Fam f) {
  switch (f) {
    case Fam::InProj: return model::BoltLinearId::InProj;
    case Fam::OutProj: return model::BoltLinearId::OutProj;
    case Fam::Qkv: return model::BoltLinearId::Qkv;
    case Fam::OProj: return model::BoltLinearId::OProj;
    case Fam::Fc1Up: return model::BoltLinearId::Fc1SharedUp;
    case Fam::Fc2: return model::BoltLinearId::Fc2;
    default: return model::BoltLinearId::SharedDown;
  }
}

inline std::vector<Launch> build_list(const model::BoltDesc& d, List which) {
  std::vector<Launch> a;
  auto put = [&](Fam f, int l) { a.push_back({f, l}); };
  if (which == List::State) {
    for (uint32_t l = 0; l < d.layers; ++l)
      if (d.is_mamba(l)) put(Fam::State, int(l));
    return a;
  }
  put(Fam::Embed, -1);
  for (uint32_t l = 0; l < d.layers; ++l) {
    const int L = int(l);
    put(Fam::Norm, L);
    if (d.is_mamba(l)) {
      put(Fam::InProj, L);
      put(Fam::Conv, L);
      put(Fam::Ssm, L);
      put(Fam::Gnorm, L);
      put(Fam::OutProj, L);
    } else if (d.is_attention(l)) {
      put(Fam::Qkv, L);
      put(Fam::AttnPrep, L);
      put(Fam::Attn, L);
      put(Fam::AttnReduce, L);
      put(Fam::OProj, L);
    } else {
      put(Fam::Router, L);
      put(Fam::Route, L);
      put(Fam::Fc1Up, L);
      put(Fam::ShAct, L);
      put(Fam::MoeUp, L);
      put(Fam::MoeDown, L);
      put(Fam::Fc2, L);
      put(Fam::ShDown, L);
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
    } else {   // Fused: every arm of spec 25 §6.1 applied (router, route, conv)
      if (x.fam == Fam::Conv || x.fam == Fam::Route || x.fam == Fam::Router) continue;
      out.push_back(x);
    }
  }
  return out;
}

// Spec 25 §6.1's derived counts: 40 x 6 + 8 x 6 + 40 x 9 + 5 at the real 88 blocks.
inline size_t derived_a(const model::BoltDesc& d) {
  size_t n = 5;
  for (uint32_t l = 0; l < d.layers; ++l) n += d.is_moe(l) ? 9 : 6;
  return n;
}
inline size_t derived_fused(const model::BoltDesc& d) {
  return derived_a(d) - d.mamba_layers() - 2 * size_t(d.moe_layers());
}

}  // namespace probe_bolt

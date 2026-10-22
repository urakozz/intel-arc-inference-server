#include "runtime/bolt/bolt_capture.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "kernels/bolt_kernels.h"
#include "kernels/kernels.h"
#include "loader/bolt_layout.h"
#include "runtime/control.h"

// bolt_capture.cc - one device's Nemotron 3.5 Bolt decode list (bolt_capture.h has the walk in one table). Every binding
// site names the kernel it honours; argument orders are each __kernel's, in src/kernels/bolt/*.cl and the reused sources.
namespace runtime::bolt {
namespace {

namespace kb = kernels::bolt;

static_assert(kRouteWords == kb::route::kWords, "the Bolt route row's width moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4 &&
                  offsetof(Control, out_token) == kernels::ctrl_index::kOutToken * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(kM == 1, "the Bolt decode binaries are M = 1");

uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::bolt::build: " + what);
}
bool aligned64(const void* p) { return reinterpret_cast<uintptr_t>(p) % 64 == 0; }

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::BoltDevicePart& part, const model::BoltDesc& d, BoltBuffers& b, BoltFuse fuse,
       bool int8_head, const StageLink* link, l0::Mem* tap, ProfileEvents* prof)
      : ctx_(ctx), part_(part), d_(d), b_(b), fuse_(fuse), int8_(int8_head), link_(link), tap_(tap), prof_(prof),
        step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    check();
    if (prof_) prof_->events.clear();
    const uint32_t dev = part_.device;
    if (dev == 0) embed();
    if (link_ && dev == 1) handoff_in();
    for (uint32_t l = part_.first; l < part_.end; ++l) {
      layer_ = int(l);
      block(l);
      pending_tap_ = int(l);   // its output is folded by the next norm site
    }
    layer_ = -1;
    if (link_ && dev == 0) handoff_out();
    if (dev + 1 == b_.placement.devices) head();
    const PpHandoff hm = link_ ? link_->mode : kDefaultPpHandoff;
    const size_t want = device_launches(d_, b_.placement, dev, b_.attn, fuse_, hm);
    require(step_.kernel_count == want, "device " + std::to_string(dev) + "'s decode list has " +
                                            std::to_string(step_.kernel_count) + " launches, not the " + std::to_string(want) +
                                            " runtime::bolt::device_launches gives (" + bolt_attn_name(b_.attn) +
                                            " attention, fuse " + bolt_fuse_name(fuse_) + ")");
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    require(!prof_ || prof_->events.size() == step_.kernel_count, "a launch got no profiling event");
    step_.list.close();
    return std::move(step_);
  }

 private:
  void check() {
    // The kernels' shapes are Bolt's (kernels::bolt); a descriptor may differ only in its block count.
    require(d_.hidden == kb::kHidden && d_.m_heads == kb::kMHeads && d_.m_head_dim == kb::kMHeadDim &&
                d_.m_groups == kb::kMGroups && d_.m_state == kb::kMState && d_.m_conv == kb::kMConv &&
                d_.m_inproj == kb::kInProjN && d_.m_conv_dim == kb::kXbc && d_.m_inner == kb::kInner &&
                d_.q_heads == kb::kQHeads && d_.kv_heads == kb::kKvHeads && d_.head_dim == kb::kHd &&
                d_.latent == kb::kLatent && d_.shared_inter == kb::kShared && d_.top_k == kb::kTopK &&
                d_.vocab == kb::kVocab && d_.max_experts() <= kb::kMaxExperts && loader::bolt::kConvRing == kb::kConvRing,
            "the descriptor is not Nemotron 3.5 Bolt's shape (" + d_.name + "); the kernels bake it");
    require(b_.desc.layers == d_.layers, "the buffers were sized for another descriptor");
    require(part_.device == b_.device, "the part is device " + std::to_string(part_.device) + ", the buffers device " +
                                           std::to_string(b_.device));
    require(part_.first == b_.placement.first(b_.device) && part_.end == b_.placement.end(b_.device) &&
                part_.layers.size() == part_.end - part_.first,
            "the loaded part's blocks are not the placement's");
    require(b_.placement.devices == 1 || link_ != nullptr, "two devices need the hand-off's binding");
    require(b_.placement.devices == 2 || link_ == nullptr, "a hand-off on one device");
    if (part_.device == 0) require(bool(part_.embed_tiles), "device 0 holds no embedding tiles");
    const bool last = part_.device + 1 == b_.placement.devices;
    if (last) {
      require(part_.final_norm && part_.head, "the last device holds no norm_f / head");
      require((part_.head->kind == model::WeightKind::Int8) == int8_, "the loaded head is not the asked-for form");
    }
    const PersistentSizes ps = persistent_sizes(d_, b_.placement, b_.device, b_.max_len, b_.state);
    require(b_.kv_k.size() == ps.kv_k && b_.conv_ring.size() == ps.conv_ring && b_.ssm.size() == ps.ssm,
            "the persistent group is not persistent_sizes'");
    const ScratchSizes ss = scratch_sizes(d_, b_.max_len, b_.attn);
    require(b_.slices.size() == ss.slices && b_.routes.size() == ss.routes && b_.attn_part.size() == ss.attn_part &&
                b_.h0.size() == ss.h0 && b_.h1.size() == ss.h1 && b_.moe_a.size() == ss.moe_a,
            "the decode scratch is not runtime::bolt::scratch_sizes'");
    require(bool(b_.attn_scores) == (b_.attn == BoltAttn::Eager && d_.attn_layers() > 0),
            "eager's score row exists exactly under eager");
    // Plan 25c's alignment rule: KV rows (256 B), ring rows (20,480 B), SSM state slices, expert blocks - 64 B bases.
    require(kv_layer_bytes(d_, b_.max_len) % 64 == 0 && ring_layer_bytes(d_) % 64 == 0 &&
                ssm_layer_bytes(d_, b_.state) % 64 == 0 && (size_t(d_.m_conv_dim) * 2) % 64 == 0 &&
                (size_t(d_.kv_n()) * 2) % 64 == 0,
            "a per-block slice is not a 64-byte multiple");
    for (const l0::Mem* m : {&b_.kv_k, &b_.kv_v, &b_.conv_ring, &b_.ssm, &b_.slices})
      require(aligned64(m->ptr()), "a persistent / scratch base is not 64-byte aligned");
    for (uint32_t l = part_.first; l < part_.end; ++l)
      if (d_.is_moe(l)) {
        const loader::BoltLayer& L = part_.layer(l);
        require(L.experts && aligned64(L.experts->ptr()) &&
                    loader::bolt::expert_block_bytes(d_.latent, d_.inter.at(l)) % 64 == 0,
                "block " + std::to_string(l) + "'s expert blocks are not 64-byte aligned");
      }
    if (tap_) require(tap_->size() >= tap_bytes(d_), "the tap is smaller than [layers][M][hidden] bf16");
    kb::Arms arms;
    arms.router = fuse_.router;
    arms.route = fuse_.route;
    arms.conv = fuse_.conv;
    const std::vector<std::string> need =
        kb::decode_variants(d_, int8_, b_.attn == BoltAttn::Eager, link_ && link_->mode == PpHandoff::Peer,
                            b_.state == BoltState::Bf16, arms);
    for (const std::string& v : need)   // a missing binary is named before a command is appended
      require(std::ifstream(kernels::path(v)).good(), v + " is not compiled (" + kernels::path(v) + "): build with B70_BOLT=ON");
  }

  l0::Mem& h(int i) { return i ? b_.h1 : b_.h0; }
  void tap_row(uint32_t l, int buf) {
    step_.list.copy(at(*tap_, size_t(l) * kM * d_.hidden * 2), h(buf).ptr(), size_t(kM) * d_.hidden * 2);
  }

  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg) {
    auto it = step_.modules.find(variant);
    if (it == step_.modules.end())
      it = step_.modules.emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant))).first;
    step_.kernels.push_back(std::make_unique<l0::Kernel>(*it->second, entry));
    l0::Kernel& k = *step_.kernels.back();
    k.group_size(wg);
    pending_entry_ = entry;
    pending_variant_ = variant;
    return k;
  }
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    require(!pending_entry_.empty(), "launch() without a preceding kernel()");
    l0::Event* signal = nullptr;
    if (prof_) {
      require(step_.kernel_count < ProfileEvents::kProfileCapacity,
              "the decode walk has more than " + std::to_string(ProfileEvents::kProfileCapacity) + " launches");
      prof_->events.emplace_back(prof_->pool, static_cast<uint32_t>(step_.kernel_count));
      signal = &prof_->events.back();
    }
    step_.list.launch(k, gx, gy, 1, signal);
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " + pending_entry_ + " " +
                           pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }

  const float* small(uint32_t l, size_t off) const { return reinterpret_cast<const float*>(at(*part_.layer(l).small, off)); }

  // nb_embed_gather(ctrl, tiles, h) - the gathered embedding becomes the residual h[cur]; nothing pending.
  void embed() {
    l0::Kernel& k = kernel(kb::embed_variant(kM), "nb_embed_gather", kb::kEmbedWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, part_.embed_tiles->ptr());
    k.arg_ptr(2, h(cur_).ptr());
    launch(k, 1, kM);
    pending_ = {};
  }

  // nb_res_norm(h_in, slices, slices2, w, h_out, x[, router tiles, logits_r, n_exp]), grid (kNormWgs [+ E / 16], M): folds
  // the pending slices (h[cur] -> h[1 - cur], then cur flips) or only normalises (_NF). The folded h is the previous
  // block's output: its tap row.
  void norm(const float* w, const loader::DeviceWeight* router = nullptr, uint32_t n_exp = 0) {
    const BoltPending p = pending_;
    const int in = cur_, out = p.s1 ? 1 - cur_ : cur_;
    l0::Kernel& k = kernel(kb::res_norm_variant(kM, p.s1, p.s2, router != nullptr), "nb_res_norm", kb::kNormWg);
    k.arg_ptr(0, h(in).ptr());
    k.arg_ptr(1, b_.slices.ptr());
    k.arg_ptr(2, at(b_.slices, size_t(p.s1) * kM * d_.hidden * 4));   // the second source right after the first's
    k.arg_ptr(3, w);
    k.arg_ptr(4, h(out).ptr());
    k.arg_ptr(5, b_.x.ptr());
    if (router) {
      require(router->shape.K == d_.hidden && router->shape.N == n_exp && router->kind == model::WeightKind::Bf16,
              "the router is not bf16 [experts][hidden] tiles");
      k.arg_ptr(6, router->mem.ptr());
      k.arg_ptr(7, b_.logits_r.ptr());
      k.arg<uint32_t>(8, n_exp);
    }
    launch(k, kb::kNormWgs + (router ? n_exp / 16 : 0), kM);
    // Review Focus 4: a fold writes the buffer it did not read; the next reader reads the one written.
    require(!p.s1 || &h(out) != &h(in), "a fold site would rewrite the buffer its work-groups read");
    cur_ = out;
    if (tap_ && pending_tap_ >= 0) {
      tap_row(uint32_t(pending_tap_), cur_);
      pending_tap_ = -1;
    }
    pending_ = {};
  }

  // An int4 GEMV into the split-K slices at byte offset `off`: gemv(w, scales, x, out), grid (N / 64, S), at the weight's
  // own layout (the descriptor's {S, layout}: layout 0 reads its separate scales, layout 1 the tiles' inline ones).
  void gemv(const loader::DeviceWeight& w, const void* x, size_t off = 0) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Int4, "a Bolt linear is int4 (spec 25 §5)");
    require(s.layout == 1 || bool(w.scales), "a layout-0 linear has no scales allocation");
    require(b_.slices.size() >= off + size_t(s.S) * kM * s.N * 4, "the slices are smaller than [S][M][N]");
    l0::Kernel& k = kernel(kernels::gemv_variant(kM, s.K, s.N, s.S, s.layout), "gemv", 64);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, s.layout == 0 ? w.scales->ptr() : w.mem.ptr());   // layout 1: not read
    k.arg_ptr(2, x);
    k.arg_ptr(3, at(b_.slices, off));
    launch(k, s.N / 64, s.S);
  }

  void block(uint32_t l) {
    const loader::BoltLayer& L = part_.layer(l);
    const loader::bolt::SmallBlock sb = loader::bolt::small_block(d_, l);
    require(L.small && L.small->size() == sb.bytes, "block " + std::to_string(l) + "'s small block is not small_block's");
    if (d_.is_mamba(l))
      mamba(l, L, sb);
    else if (d_.is_attention(l))
      attention(l, L, sb);
    else
      moe(l, L, sb);
  }

  void mamba(uint32_t l, const loader::BoltLayer& L, const loader::bolt::SmallBlock& sb) {
    require(L.in_proj && L.out_proj && L.in_proj->shape.N == d_.m_inproj && L.out_proj->shape.K == d_.m_inner,
            "block " + std::to_string(l) + "'s Mamba linears");
    norm(small(l, sb.norm));
    gemv(*L.in_proj, b_.x.ptr());
    const uint32_t S = L.in_proj->shape.S;
    if (!fuse_.conv) {   // nb_conv(ctrl, slices, cw, cb, ring, xc), grid (10240 / 256)
      l0::Kernel& k = kernel(kb::conv_variant(kM, S), "nb_conv", kb::kConvWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.slices.ptr());
      k.arg_ptr(2, small(l, sb.conv_w));
      k.arg_ptr(3, small(l, sb.conv_b));
      k.arg_ptr(4, b_.ring_layer(l));
      k.arg_ptr(5, b_.xc.ptr());
      launch(k, d_.m_conv_dim / kb::kConvWg, 1);
    }
    {   // nb_ssm(ctrl, slices, xc | cw, cb, ring, A, D, dt_bias, state, y), grid (128 x 4)
      l0::Kernel& k = kernel(kb::ssm_variant(kM, S, fuse_.conv, b_.state == BoltState::Bf16), "nb_ssm", kb::kSsmWg);
      uint32_t a = 0;
      k.arg_ptr(a++, b_.control.ptr());
      k.arg_ptr(a++, b_.slices.ptr());
      if (fuse_.conv) {
        k.arg_ptr(a++, small(l, sb.conv_w));
        k.arg_ptr(a++, small(l, sb.conv_b));
        k.arg_ptr(a++, b_.ring_layer(l));
      } else {
        k.arg_ptr(a++, b_.xc.ptr());
      }
      k.arg_ptr(a++, small(l, sb.A));
      k.arg_ptr(a++, small(l, sb.D));
      k.arg_ptr(a++, small(l, sb.dt_bias));
      k.arg_ptr(a++, b_.ssm_layer(l));
      k.arg_ptr(a++, b_.y.ptr());
      launch(k, kb::ssm_groups(), 1);
    }
    {   // nb_gnorm(slices, y, w, out), grid (8, M)
      l0::Kernel& k = kernel(kb::gnorm_variant(kM, S), "nb_gnorm", kb::kGnormWg);
      k.arg_ptr(0, b_.slices.ptr());
      k.arg_ptr(1, b_.y.ptr());
      k.arg_ptr(2, small(l, sb.gnorm));
      k.arg_ptr(3, b_.g.ptr());
      launch(k, d_.m_inner / kb::kGnormGroup, kM);
    }
    gemv(*L.out_proj, b_.g.ptr());
    pending_ = {L.out_proj->shape.S, 0};
  }

  void attention(uint32_t l, const loader::BoltLayer& L, const loader::bolt::SmallBlock& sb) {
    require(L.qkv && L.o_proj && L.qkv->shape.N == d_.qkv_n() && L.o_proj->shape.N == d_.hidden,
            "block " + std::to_string(l) + "'s attention linears");
    norm(small(l, sb.norm));
    gemv(*L.qkv, b_.x.ptr());
    void* kv_k = b_.k_layer(l);
    void* kv_v = b_.v_layer(l);
    {   // nb_attn_prep(ctrl, slices, attn_q, kv_k, kv_v), grid (ceil(4608 / 256), M)
      l0::Kernel& k = kernel(kb::attn_prep_variant(kM, L.qkv->shape.S), "nb_attn_prep", kb::kPrepWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.slices.ptr());
      k.arg_ptr(2, b_.attn_q.ptr());
      k.arg_ptr(3, kv_k);
      k.arg_ptr(4, kv_v);
      launch(k, (d_.qkv_n() + kb::kPrepWg - 1) / kb::kPrepWg, kM);
    }
    const std::string v = kb::attn_variant(kM, b_.attn == BoltAttn::Eager);
    if (b_.attn == BoltAttn::Eager) {   // mg_attn_eager(ctrl, attn_q, kv_k, kv_v, gate, attn_s, attn_out, stride)
      l0::Kernel& k = kernel(v, "mg_attn_eager", kb::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kv_k);
      k.arg_ptr(3, kv_v);
      k.arg_ptr(4, b_.ones.ptr());
      k.arg_ptr(5, b_.attn_scores->ptr());
      k.arg_ptr(6, b_.attn_out.ptr());
      k.arg<uint32_t>(7, b_.max_len);
      launch(k, d_.q_heads, kM);
    } else {
      {   // mg_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part), grid (kv heads, TGT)
        l0::Kernel& k = kernel(v, "mg_attn_decode", kb::kAttnWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.attn_q.ptr());
        k.arg_ptr(2, kv_k);
        k.arg_ptr(3, kv_v);
        k.arg_ptr(4, b_.attn_part.ptr());
        launch(k, d_.kv_heads, kb::kAttnTgt);
      }
      {   // mg_attn_reduce(ctrl, attn_part, gate, attn_out), grid (q heads, M) - the ones gate: rne(rne(o) x 1) = rne(o)
        l0::Kernel& k = kernel(v, "mg_attn_reduce", kb::kAttnWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.attn_part.ptr());
        k.arg_ptr(2, b_.ones.ptr());
        k.arg_ptr(3, b_.attn_out.ptr());
        launch(k, d_.q_heads, kM);
      }
    }
    gemv(*L.o_proj, b_.attn_out.ptr());
    pending_ = {L.o_proj->shape.S, 0};
  }

  void moe(uint32_t l, const loader::BoltLayer& L, const loader::bolt::SmallBlock& sb) {
    const uint32_t E = d_.experts.at(l), I = d_.inter.at(l);
    require(L.router && L.experts && L.fc1_shared_up && L.fc2 && L.shared_down &&
                L.fc1_shared_up->shape.N == d_.latent + d_.shared_inter && L.fc2->shape.N == d_.hidden &&
                L.shared_down->shape.N == d_.hidden && L.experts->size() == loader::bolt::layer_expert_bytes(d_.latent, E, I),
            "block " + std::to_string(l) + "'s MoE block");
    require(E % 16 == 0 && E <= kb::kMaxExperts && E > kb::kTopK, "block " + std::to_string(l) + "'s E " + std::to_string(E));
    norm(small(l, sb.norm), fuse_.router ? L.router.get() : nullptr, E);
    void* rt = at(b_.routes, route_at(l));
    const float* bias = small(l, sb.e_bias);
    if (!fuse_.router) {   // nb_router(x, w, logits, n_exp), grid (E / 16, M)
      l0::Kernel& k = kernel(kb::router_variant(kM), "nb_router", kb::kRouterWg);
      k.arg_ptr(0, b_.x.ptr());
      k.arg_ptr(1, L.router->mem.ptr());
      k.arg_ptr(2, b_.logits_r.ptr());
      k.arg<uint32_t>(3, E);
      launch(k, E / 16, kM);
    }
    if (!fuse_.route) {   // nb_route(logits, bias, route, n_exp), grid (1, M)
      l0::Kernel& k = kernel(kb::route_variant(kM), "nb_route", kb::kRouteWg);
      k.arg_ptr(0, b_.logits_r.ptr());
      k.arg_ptr(1, bias);
      k.arg_ptr(2, rt);
      k.arg<uint32_t>(3, E);
      launch(k, 1, kM);
    }
    gemv(*L.fc1_shared_up, b_.x.ptr());
    {   // nb_sh_act(slices, lat, sh), grid (ceil(6400 / 256), M)
      l0::Kernel& k = kernel(kb::sh_act_variant(kM, L.fc1_shared_up->shape.S), "nb_sh_act", kb::kActWg);
      k.arg_ptr(0, b_.slices.ptr());
      k.arg_ptr(1, b_.lat.ptr());
      k.arg_ptr(2, b_.sh.ptr());
      launch(k, (kb::kFc1N + kb::kActWg - 1) / kb::kActWg, kM);
    }
    {   // nb_moe_up([logits, bias,] route, lat, w, a, n_exp), grid (9 x I / 64, M)
      l0::Kernel& k = kernel(kb::moe_up_variant(kM, I, fuse_.route), "nb_moe_up", kb::moe_up_wg());
      uint32_t a = 0;
      if (fuse_.route) {
        k.arg_ptr(a++, b_.logits_r.ptr());
        k.arg_ptr(a++, bias);
      }
      k.arg_ptr(a++, rt);
      k.arg_ptr(a++, b_.lat.ptr());
      k.arg_ptr(a++, L.experts->ptr());
      k.arg_ptr(a++, b_.moe_a.ptr());
      k.arg<uint32_t>(a++, E);
      launch(k, kb::moe_up_groups(I), kM);
    }
    {   // nb_moe_down(route, a, w, r, n_exp), grid (1024 / 16, M)
      l0::Kernel& k = kernel(kb::moe_down_variant(kM, I), "nb_moe_down", kb::moe_down_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.moe_a.ptr());
      k.arg_ptr(2, L.experts->ptr());
      k.arg_ptr(3, b_.moe_r.ptr());
      k.arg<uint32_t>(4, E);
      launch(k, kb::moe_down_groups(), kM);
    }
    gemv(*L.fc2, b_.moe_r.ptr());
    gemv(*L.shared_down, b_.sh.ptr(), size_t(L.fc2->shape.S) * kM * d_.hidden * 4);   // right after fc2's slices
    pending_ = {L.fc2->shape.S, L.shared_down->shape.S};
  }

  void head() {
    norm(static_cast<const float*>(part_.final_norm->ptr()));   // norm_f, folding the last block
    const loader::DeviceWeight& lm = *part_.head;
    require(lm.shape.N == d_.vocab && lm.shape.K == d_.hidden, "the head is not [vocab][hidden]");
    if (int8_) {   // gemv_i8w(w, scales, x, out), grid (N / 64)
      require(bool(lm.scales), "the int8 head has no scales");
      l0::Kernel& k = kernel(kb::head_variant(true), "gemv_i8w", kernels::kGemvI8wCols);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, lm.scales->ptr());
      k.arg_ptr(2, b_.x.ptr());
      k.arg_ptr(3, b_.logits.ptr());
      launch(k, d_.vocab / kernels::kGemvI8wCols);
    } else {   // gemv_bf16(w, x, out) over lm_head's own tiles (untied)
      const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(d_.vocab);
      l0::Kernel& k = kernel(kb::head_variant(false), "gemv_bf16", t.cols * t.ksplit);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, b_.logits.ptr());
      launch(k, d_.vocab / t.cols);
    }
    {
      l0::Kernel& k = kernel(kb::argmax1_variant(kM), "argmax_stage1", kb::kArgmaxWg);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, kb::argmax_groups(), kM);
    }
    {
      l0::Kernel& k = kernel(kb::argmax2_variant(), "argmax_stage2", kb::kArgmaxWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
    require(kb::argmax_groups() <= kb::kArgmaxWg, "argmax_stage2 folds at most 256 groups");
  }

  // --- the hand-off (spec 16b's; lfm2moe_capture.cc's at every cut: h and the pending slices) ----------------------------
  size_t h_bytes() const { return size_t(kM) * d_.hidden * 2; }
  size_t cut_slices_bytes() const { return size_t(cut_pending(d_, b_.placement).total()) * kM * d_.hidden * 4; }
  void check_link() const {
    require(link_->land_resid && link_->resid_cap >= h_bytes() && link_->sumsq_cap == cut_slices_bytes() &&
                (cut_slices_bytes() == 0 || link_->land_sumsq),
            "the landing buffer does not hold " + std::to_string(h_bytes()) + " B of the row and exactly " +
                std::to_string(cut_slices_bytes()) + " B of pending slices");
    if (link_->mode == PpHandoff::Copy) require(link_->event != nullptr, "the copy hand-off needs its cross-device event");
  }
  void handoff_out() {
    check_link();
    const BoltPending cp = cut_pending(d_, b_.placement);
    require(pending_.s1 == cp.s1 && pending_.s2 == cp.s2, "device 0's list ends with other slices pending than the cut's");
    if (link_->mode == PpHandoff::Copy) {
      step_.list.copy(link_->land_resid, h(cur_).ptr(), h_bytes());
      if (cut_slices_bytes()) step_.list.copy(link_->land_sumsq, b_.slices.ptr(), cut_slices_bytes());
      step_.list.barrier_signal(link_->event);
      return;
    }
    require(link_->flag && link_->send_seq, "the peer hand-off needs its flag and counter");
    // pp_send(resid, sumsq, land_resid, land_sumsq, flag, seq, resid_words, sumsq_words), grid (1)
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_send", kernels::kPpHandoffWg);
    k.arg_ptr(0, h(cur_).ptr());
    k.arg_ptr(1, b_.slices.ptr());
    k.arg_ptr(2, link_->land_resid);
    k.arg_ptr(3, link_->land_sumsq);
    k.arg_ptr(4, link_->flag);
    k.arg_ptr(5, link_->send_seq);
    k.arg<uint32_t>(6, static_cast<uint32_t>(h_bytes() / 4));
    k.arg<uint32_t>(7, static_cast<uint32_t>(cut_slices_bytes() / 4));
    launch(k, 1);
  }
  void handoff_in() {
    check_link();
    cur_ = 0;
    if (link_->mode == PpHandoff::Copy) {
      step_.list.wait_event(link_->event);
      step_.list.copy(h(0).ptr(), link_->land_resid, h_bytes());
      if (cut_slices_bytes()) step_.list.copy(b_.slices.ptr(), link_->land_sumsq, cut_slices_bytes());
    } else {
      require(link_->flag && link_->recv_state && link_->spin_limit > 0,
              "the peer hand-off needs its flag, its state words and a spin bound");
      // pp_recv(land_resid, land_sumsq, flag, state, resid, sumsq, resid_words, sumsq_words, spin_limit)
      l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_recv", kernels::kPpHandoffWg);
      k.arg_ptr(0, link_->land_resid);
      k.arg_ptr(1, link_->land_sumsq);
      k.arg_ptr(2, link_->flag);
      k.arg_ptr(3, link_->recv_state);
      k.arg_ptr(4, h(0).ptr());
      k.arg_ptr(5, b_.slices.ptr());
      k.arg<uint32_t>(6, static_cast<uint32_t>(h_bytes() / 4));
      k.arg<uint32_t>(7, static_cast<uint32_t>(cut_slices_bytes() / 4));
      k.arg<uint32_t>(8, link_->spin_limit);
      launch(k, 1);
    }
    pending_ = cut_pending(d_, b_.placement);   // the first norm folds them
    pending_tap_ = int(b_.placement.split) - 1;  // device 0 could not tap its last block: its output is folded here
  }

  l0::Context& ctx_;
  const loader::BoltDevicePart& part_;
  const model::BoltDesc& d_;
  BoltBuffers& b_;
  BoltFuse fuse_;
  bool int8_;
  const StageLink* link_;
  l0::Mem* tap_;
  ProfileEvents* prof_;
  CapturedStep step_;
  int layer_ = -1;
  int cur_ = 0;               // the residual's current buffer: h0 or h1
  BoltPending pending_;       // split-K slices waiting in b_.slices for the next norm to fold
  int pending_tap_ = -1;      // the block whose output the next fold produces
  std::string pending_entry_, pending_variant_;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::BoltDevicePart& part, const model::BoltDesc& d, BoltBuffers& b,
                   BoltFuse fuse, bool int8_head, const StageLink* link, l0::Mem* tap, ProfileEvents* prof) {
  Walk w(ctx, part, d, b, fuse, int8_head, link, tap, prof);
  return w.run();
}

}  // namespace runtime::bolt

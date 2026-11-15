#include "runtime/dflash/dspark_drafter.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/dflash_kernels.h"
#include "kernels/kernels.h"
#include "loader/dflash_layout.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_sizes.h"

// dspark_drafter.cc - the DSpark drafter's two captured lists and DSparkDrafter (dspark_drafter.h has the walks in order).
// Every binding site names the kernel it honours; argument orders are each __kernel's in src/kernels/dflash/
// {df_prep,df_attn,df_topk,df_select}.cl and the reused gemv_bf16.cl / gemv_i8w.cl.
namespace runtime::dflash {
namespace {

namespace kd = kernels::dflash;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::dflash DSpark: " + what);
}
const uint8_t* at(const void* p, size_t off) { return static_cast<const uint8_t*>(p) + off; }

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b)
      : ctx_(ctx), w_(w), d_(w.desc), b_(b), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep draft(const l0::Mem& embed, const loader::DeviceWeight& hw, const l0::Mem& control) {
    check();
    require(hw.kind == model::WeightKind::Int8 || hw.kind == model::WeightKind::Bf16,
            "the target's head is int4: no draft binaries for it - load the head int8 or bf16");
    require(hw.shape.K == d_.hidden && hw.shape.N == d_.vocab && b_.head_rows == d_.vocab,
            "the target's head is not [" + std::to_string(d_.vocab) + "][" + std::to_string(d_.hidden) + "]");
    require(embed.size() >= size_t(d_.vocab) * d_.hidden * 2, "the embedding replica is not [vocab][hidden] bf16");
    require(control.size() >= sizeof(Control), "the control block is smaller than runtime::Control");
    for (const std::string& v : kd::dspark_draft_variants(hw.kind == model::WeightKind::Int8))
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_KOLIBRI and B70_DFLASH on");
    const void* ctrl = control.ptr();
    const uint32_t M = kd::kM, R = kd::kR, H = d_.hidden, V = d_.vocab;
    const kd::Shape& s = kd::kDSpark;
    layer_ = -1;
    {   // df_embed(ctrl, embed, mask_row, in_scale, resid), grid (1, M)
      l0::Kernel& k = kernel(kd::prep_variant(s), "df_embed", kd::kWg);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, embed.ptr());
      k.arg_ptr(2, w_.mask_row->ptr());
      k.arg(3, d_.in_scale);
      k.arg_ptr(4, b_.resid.ptr());
      launch(k, 1, M);
    }
    const size_t ring_layer = size_t(d_.window.at(0)) * d_.kv_n() * 2;
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      const loader::DFlashLayerWeights& L = w_.layers[l];
      const float* norms = L.norms.as<float>();
      norm(b_.resid.ptr(), norms);
      gemv(L.qkv, b_.x.ptr(), b_.gout.ptr());
      {   // df_qkv_prep(ctrl, qkv, qn_kn, rope, q, kb, vb), grid (NH + 2 NKV, M), WG HD
        l0::Kernel& k = kernel(kd::prep_variant(s), "df_qkv_prep", d_.head_dim);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.gout.ptr());
        k.arg_ptr(2, at(w_.qk_norm->ptr(), loader::dflash_qk_norm_off(d_, l, 0) * 4));
        k.arg_ptr(3, b_.rope.ptr());
        k.arg_ptr(4, b_.q.ptr());
        k.arg_ptr(5, b_.kb.ptr());
        k.arg_ptr(6, b_.vb.ptr());
        launch(k, d_.q_heads + 2 * d_.kv_heads, M);
      }
      {   // df_attn_part(ctrl, dctl, q, ring_k, ring_v, scale, part), grid (NKV, ring / PB), WG 128
        l0::Kernel& k = kernel(kd::dspark_attn_variant(), "df_attn_part", kd::kAttnWg);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.dctl.ptr());
        k.arg_ptr(2, b_.q.ptr());
        k.arg_ptr(3, at(b_.ring_k.ptr(), l * ring_layer));
        k.arg_ptr(4, at(b_.ring_v.ptr(), l * ring_layer));
        k.arg(5, scale_);
        k.arg_ptr(6, b_.part.ptr());
        launch(k, d_.kv_heads, d_.window.at(l) / kd::kAttnPb);
      }
      {   // df_attn_reduce(ctrl, dctl, part, q, kb, vb, scale, out), grid (NH, M), WG HD - the causal build
        l0::Kernel& k = kernel(kd::dspark_attn_variant(), "df_attn_reduce", d_.head_dim);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.dctl.ptr());
        k.arg_ptr(2, b_.part.ptr());
        k.arg_ptr(3, b_.q.ptr());
        k.arg_ptr(4, b_.kb.ptr());
        k.arg_ptr(5, b_.vb.ptr());
        k.arg(6, scale_);
        k.arg_ptr(7, b_.xa.ptr());
        launch(k, d_.q_heads, M);
      }
      gemv(L.o, b_.xa.ptr(), b_.gout.ptr());
      add(b_.gout.ptr());
      norm(b_.resid.ptr(), norms + H);
      gemv(L.gate_up, b_.x.ptr(), b_.gout.ptr());
      {   // df_silu_mul(gu, a), grid (ceil(I / 256), M)
        l0::Kernel& k = kernel(kd::prep_variant(s), "df_silu_mul", kd::kWg);
        k.arg_ptr(0, b_.gout.ptr());
        k.arg_ptr(1, b_.xa.ptr());
        launch(k, (d_.inter + kd::kWg - 1) / kd::kWg, M);
      }
      gemv(L.down, b_.xa.ptr(), b_.gout.ptr());
      add(b_.gout.ptr());
    }
    layer_ = -1;
    norm(b_.resid.ptr(), w_.final_norm->as<float>());
    {   // the target's head over rows 0..6 (anchor-first: row r predicts p + r + 1), gemv_i8w / gemv_bf16 at M 7
      if (hw.kind == model::WeightKind::Int8) {
        l0::Kernel& k = kernel(kernels::gemv_i8w_variant(R, H, V), "gemv_i8w", kernels::kGemvI8wCols);
        k.arg_ptr(0, hw.mem.ptr());
        k.arg_ptr(1, hw.scales->ptr());
        k.arg_ptr(2, b_.x.ptr());
        k.arg_ptr(3, b_.logits.ptr());
        launch(k, V / kernels::kGemvI8wCols);
      } else {
        const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(V);
        l0::Kernel& k = kernel(kernels::gemv_bf16_variant(R, H, V, tl), "gemv_bf16", tl.cols * tl.ksplit);
        k.arg_ptr(0, hw.mem.ptr());
        k.arg_ptr(1, b_.x.ptr());
        k.arg_ptr(2, b_.logits.ptr());
        launch(k, V / tl.cols);
      }
    }
    {   // df_topk_stage1(logits, part), grid (chunks, R); df_topk_stage2(part, ids, out_mult, softcap, cand, unary), grid (R)
      l0::Kernel& k1 = kernel(kd::dspark_topk_variant(), "df_topk_stage1", kd::kWg);
      k1.arg_ptr(0, b_.logits.ptr());
      k1.arg_ptr(1, b_.tpart.ptr());
      launch(k1, (V + kd::kTopkChunk - 1) / kd::kTopkChunk, R);
      l0::Kernel& k2 = kernel(kd::dspark_topk_variant(), "df_topk_stage2", kd::kWg);
      k2.arg_ptr(0, b_.tpart.ptr());
      k2.arg_ptr(1, b_.cand.ptr());   // the full head's build never reads it
      k2.arg(2, d_.out_mult);
      k2.arg(3, d_.softcap);
      k2.arg_ptr(4, b_.cand.ptr());
      k2.arg_ptr(5, b_.unary.ptr());
      launch(k2, R);
    }
    {   // df_select(ctrl, dctl, cand, unary, hid, pred, succ, out), grid (1), WG 256: the Markov walk (V2, hid = ones)
      l0::Kernel& k = kernel(kd::select_variant(true, s), "df_select", kd::kWg);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.dctl.ptr());
      k.arg_ptr(2, b_.cand.ptr());
      k.arg_ptr(3, b_.unary.ptr());
      k.arg_ptr(4, b_.hid->ptr());
      k.arg_ptr(5, w_.pred->ptr());
      k.arg_ptr(6, w_.succ->ptr());
      k.arg_ptr(7, b_.out.ptr());
      launch(k, 1);
    }
    return finish(draft_launches(d_), "draft");
  }

  CapturedStep project() {
    check();
    const uint32_t M = kd::kM;
    const uint32_t kvn = 2 * d_.kv_n();
    for (const std::string& v :
         {kd::prep_variant(kd::kDSpark), kernels::gemv_bf16_variant(M, w_.fc->shape.K, d_.hidden, kernels::gemv_bf16_tiling(d_.hidden)),
          kernels::gemv_bf16_variant(M, d_.hidden, kvn, kernels::gemv_bf16_tiling(kvn))})
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_KOLIBRI and B70_DFLASH on");
    layer_ = -1;
    gemv(*w_.fc, b_.taps.ptr(), b_.gout.ptr());
    norm(b_.gout.ptr(), w_.hidden_norm->as<float>());   // hidden_norm
    // per layer the k||v rows [q_n, qkv_n) of q||k||v (gemv_bf16's n-tile-major tiles: the weight from row q_n on)
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      const loader::DeviceWeight& qkv = w_.layers[l].qkv;
      const size_t K = d_.hidden, rows = d_.q_n(), n = 2 * d_.kv_n();
      const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(uint32_t(n));
      l0::Kernel& k = kernel(kernels::gemv_bf16_variant(M, uint32_t(K), uint32_t(n), tl), "gemv_bf16", tl.cols * tl.ksplit);
      k.arg_ptr(0, at(qkv.mem.ptr(), rows * K * 2));
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, at(b_.kv.ptr(), size_t(l) * M * n * 4));
      launch(k, uint32_t(n) / tl.cols);
    }
    layer_ = -1;
    {   // df_ctx_store(dctl, kv, qk_norm, rope, ring_k, ring_v), grid (L x 2 NKV, M), WG HD
      l0::Kernel& k = kernel(kd::prep_variant(kd::kDSpark), "df_ctx_store", d_.head_dim);
      k.arg_ptr(0, b_.dctl.ptr());
      k.arg_ptr(1, b_.kv.ptr());
      k.arg_ptr(2, w_.qk_norm->ptr());
      k.arg_ptr(3, b_.rope.ptr());
      k.arg_ptr(4, b_.ring_k.ptr());
      k.arg_ptr(5, b_.ring_v.ptr());
      launch(k, d_.layers * 2 * d_.kv_heads, M);
    }
    return finish(project_launches(d_), "projection");
  }

 private:
  // The kernels bake the published drafter's shapes (kernels::dflash::kDSpark): model::check_dspark, again here.
  void check() {
    model::check_dspark(d_);
    require(w_.format == loader::DraftFormat::Bf16, "the DSpark drafter runs bf16 (an int8 / int4 drafter is a later step)");
    require(w_.target_hidden == kd::kDSpark.H && w_.pred && w_.succ && w_.mask_row,
            "the weights are not a DSpark drafter's beside Kolibri-1 (hidden 2560, the Markov head, the mask row)");
    require(b_.desc.layers == d_.layers && b_.target_hidden == w_.target_hidden && b_.hid &&
                b_.ring_k.size() == size_t(d_.layers) * d_.window.at(0) * d_.kv_n() * 2,
            "the buffers were sized for another drafter");
    require(b_.max_len >= 1, "the buffers' max_len is 0");
  }

  // df_norm(src, w, h, x), grid (1, M)
  void norm(const void* src, const float* w) {
    l0::Kernel& k = kernel(kd::prep_variant(kd::kDSpark), "df_norm", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, w);
    k.arg_ptr(2, b_.h.ptr());
    k.arg_ptr(3, b_.x.ptr());
    launch(k, 1, kd::kM);
  }
  // df_add(src, resid), grid (ceil(H / 256), M): DFlash v1's residual add (no convolution)
  void add(const void* src) {
    l0::Kernel& k = kernel(kd::prep_variant(kd::kDSpark), "df_add", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, b_.resid.ptr());
    launch(k, (d_.hidden + kd::kWg - 1) / kd::kWg, kd::kM);
  }
  // A backbone linear at M 8: gemv_bf16(w, x, out), grid (N / cols)
  void gemv(const loader::DeviceWeight& w, const void* x, const void* out) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Bf16, "a DSpark linear is not bf16");
    const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(s.N);
    l0::Kernel& k = kernel(kernels::gemv_bf16_variant(kd::kM, s.K, s.N, tl), "gemv_bf16", tl.cols * tl.ksplit);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, x);
    k.arg_ptr(2, out);
    launch(k, s.N / tl.cols);
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
    step_.list.launch(k, gx, gy, 1, nullptr);
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " + pending_entry_ + " " +
                           pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }
  CapturedStep finish(size_t want, const char* what) {
    require(step_.kernel_count == want, std::string("the ") + what + " list has " + std::to_string(step_.kernel_count) +
                                            " launches, not the " + std::to_string(want) +
                                            " runtime::dflash's formula gives");
    require(step_.kernels.size() == step_.kernel_count && step_.labels.size() == step_.kernel_count,
            "a kernel went unlaunched or unlabelled");
    step_.list.close();
    return std::move(step_);
  }

  l0::Context& ctx_;
  const loader::DFlashWeights& w_;
  const model::DFlashDesc& d_;
  DFlashBuffers& b_;
  CapturedStep step_;
  const float scale_ = float(1.0 / std::sqrt(double(d_.head_dim)));
  int layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

CapturedStep build_dspark_draft(l0::Context& ctx, const loader::DFlashWeights& w, const l0::Mem& embed,
                                const loader::DeviceWeight& head, DFlashBuffers& b, const l0::Mem& control) {
  return Walk(ctx, w, b).draft(embed, head, control);
}
CapturedStep build_dspark_project(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b) {
  return Walk(ctx, w, b).project();
}

DSparkDrafter::DSparkDrafter(l0::Context& ctx, loader::DFlashWeights w, uint32_t max_len, uint32_t max_k)
    : w_(std::move(w)),
      b_(ctx, w_.desc, w_.target_hidden, w_.desc.vocab, max_len, loader::DraftFormat::Bf16),
      max_k_(max_k),
      project_(build_dspark_project(ctx, w_, b_)) {
  require(w_.desc.dspark(), "the weights are not a DSpark drafter's (model::DFlashForm::DSpark)");
  require(max_k >= 1 && max_k <= w_.desc.max_k(), "max_k " + std::to_string(max_k) + " is outside 1..7 (the block of 8)");
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  reset(imm);
  std::vector<uint8_t> norms(w_.hidden_norm->size() + w_.final_norm->size());
  imm.copy(norms.data(), w_.hidden_norm->ptr(), w_.hidden_norm->size());
  imm.copy(norms.data() + w_.hidden_norm->size(), w_.final_norm->ptr(), w_.final_norm->size());
  key_ = drafter_key(w_.desc, w_.format, norms.data(), norms.size());
}

void DSparkDrafter::attach(l0::Context& ctx, const l0::Mem& embed, const loader::DeviceWeight& head, const l0::Mem& control) {
  require(draft_ == nullptr, "attach() twice");
  draft_ = std::make_unique<CapturedStep>(build_dspark_draft(ctx, w_, embed, head, b_, control));
}

const DFlashDraftOut& DSparkDrafter::draft(const std::function<void(l0::CmdList&)>& replay, uint32_t k, float temperature,
                                           uint64_t seed) {
  require(draft_ != nullptr, "draft before attach");
  require(k >= 1 && k <= max_k_, "k " + std::to_string(k) + " is outside 1.." + std::to_string(max_k_));
  require(temperature >= 0.0f, "the temperature must be >= 0");
  DFlashCtl& c = *b_.ctl();
  c.k = w_.desc.max_k();   // 7: the whole causal block walked - a draft of k is the walk's first k
  c.temperature = temperature;
  c.seed_lo = uint32_t(seed);
  c.seed_hi = uint32_t(seed >> 32);
  replay(draft_->list);
  const DFlashDraftOut& out = *b_.readback();
  require(out.k == c.k, "the draft list walked " + std::to_string(out.k) + " rows, not " + std::to_string(c.k));
  return out;
}

l0::CmdList& DSparkDrafter::projection(uint32_t pos, uint32_t n) {
  require(n >= 1 && n <= kernels::dflash::kM, "a projection takes 1..8 rows, not " + std::to_string(n));
  require(size_t(pos) + n <= size_t(b_.max_len) + kernels::dflash::kM,
          "positions " + std::to_string(pos) + " + " + std::to_string(n) + " are past the RoPE table");
  DFlashCtl& c = *b_.ctl();
  c.proj_pos = pos;
  c.proj_n = n;
  return project_.list;
}

void DSparkDrafter::reset(l0::CmdList& imm) {
  b_.zero(imm);
  imm.fill(b_.hid->ptr(), 0x3F800000u, b_.hid->size());   // the walk's hidden rows: 1.0f (the Markov bias unscaled)
}

void DSparkDrafter::save_ring(l0::CmdList& imm, void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  imm.copy(h, b_.ring_k.ptr(), b_.ring_k.size());
  imm.copy(h + b_.ring_k.size(), b_.ring_v.ptr(), b_.ring_v.size());
}

void DSparkDrafter::load_ring(l0::CmdList& imm, const void* host) {
  const auto* h = static_cast<const uint8_t*>(host);
  imm.copy(b_.ring_k.ptr(), h, b_.ring_k.size());
  imm.copy(b_.ring_v.ptr(), h + b_.ring_k.size(), b_.ring_v.size());
}

}  // namespace runtime::dflash

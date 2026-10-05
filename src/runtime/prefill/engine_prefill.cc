// `Engine::prefill` and its two accessors -- the ONLY members of
// `runtime::Engine` that are not compiled into `b70_runtime`.
//
// **Why they live here.** `b70_runtime` is linked by every decode binary and
// every decode test, and cmake/prefill.cmake's whole arrangement is that none
// of those acquires a dependency on libb70_prefill.so. `Engine::prefill` calls
// into `runtime::prefill::Context` (an icpx-built .so symbol) and into
// `gemm_bf16` (sycl-tla). Defining it in this archive keeps the split exact: a
// target that never calls `prefill()` links what it always linked, and a target
// that does calls `b70_link_prefill()` and gets the .so and the two rpaths.
// The member declarations stay in `runtime/engine.h`; only the definitions move.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill/step.h"

namespace runtime {

struct PrefillEngine {
  prefill::Context cx;
  prefill::KernelCache kc;
  struct Chunk {
    uint32_t pos, rows;
    PrefillBackend backend;   // a recording holds one backend's walk, never another's
    prefill::AttnMode attn;   // ... and one attention path's (spec 6)
    std::unique_ptr<prefill::Context::Recording> recording;
  };
  // Spec 5's int8 state (signs, per-linear column scales, int8 scratch), created on the
  // first l0-int8 prefill. Declared before `chunks`: recordings that name its buffers are
  // destroyed first.
  std::unique_ptr<prefill::Int8State> int8;
  // Declared after kc/cx: recordings are destroyed before kernels/context.
  std::vector<Chunk> chunks;
  explicit PrefillEngine(l0::Context& c) : cx(c), kc(c) {}
};

namespace {
void destroy_prefill(PrefillEngine* p) { delete p; }
}  // namespace

size_t Engine::prefill_launches() const { return pfx_ ? pfx_->cx.launches() : 0; }

PrefillBackend Engine::prefill_backend() const {
  return pf_backend_ ? *pf_backend_ : prefill::default_prefill_backend();
}

bool Engine::prefill_sycl_side_created() const { return pfx_ && pfx_->cx.has_sycl(); }

std::string Engine::memory_line() const {
  const double gb = 1e9;
  const size_t model = model_.report.total();
  const size_t kv = persist_.kv_k.size() + persist_.kv_v.size();
  // Spec 8: the MTP head's buffers (its KV, the GDN slots, hh/dh, draft logits) count as
  // decode state; the head's weights are in `model` (LoadReport::mtp_bytes).
  const size_t decode = persist_.bytes() - kv + decode_scratch_.bytes() +
                        (mtp_ ? mtp_->bytes() : 0) + (mtp_pf_hid_ ? mtp_pf_hid_->size() : 0);
  const size_t pf = pf_ ? pf_->bytes() + pf_->lazy_bytes() : 0;
  const size_t i8 = pfx_ && pfx_->int8 ? pfx_->int8->bytes() : 0;
  const size_t total = model + kv + decode + pf + i8;
  uint32_t n = 0;
  zeDeviceGetMemoryProperties(ctx_.device(), &n, nullptr);
  std::vector<ze_device_memory_properties_t> props(n);
  for (auto& p : props) p.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  if (n) zeDeviceGetMemoryProperties(ctx_.device(), &n, props.data());
  size_t device = 0;
  for (const auto& p : props) device += p.totalSize;
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "memory: model %.3f GB, kv %.3f GB, decode state %.3f GB, prefill scratch %.3f GB,"
                " int8 %.3f GB, total %.3f GB of %.3f GB",
                model / gb, kv / gb, decode / gb, pf / gb, i8 / gb, total / gb, device / gb);
  return buf;
}

void Engine::prepare_prefill() {
  // Ruling R7: both allocations are lazy, so a decode-only Engine's device
  // residency is byte-identical to what it was before the buffer split.
  if (!pf_) pf_.reset(new PrefillScratch(ctx_, buffers_.max_len, *model_.desc));
  if (!pfx_)
    pfx_ = std::unique_ptr<PrefillEngine, void (*)(PrefillEngine*)>(new PrefillEngine(ctx_),
                                                                    &destroy_prefill);
  if (prefill_backend() != PrefillBackend::L0Int8) return;
  if (!pfx_->int8)   // the largest int4 K is down's: the MLP intermediate (spec 14)
    pfx_->int8 = std::make_unique<prefill::Int8State>(ctx_, model_.desc->intermediate);
  // Every int4 linear's rotated column scales (spec 5 T2), outside any chunk and
  // any recording: cached by weight, so after the first call this loop only
  // looks them up, and no recorded list ever holds the host finish of
  // pf_colmax_rot.
  for (const auto& [key, w] : model_.linears)
    if (key.second != model::LinearId::LmHead && w.kind == model::WeightKind::Int4)
      pfx_->int8->scales(pfx_->cx, pfx_->kc, w);
}

void Engine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  if (ids.empty()) throw std::runtime_error("runtime::Engine::prefill: no ids");
  if (chunk == 0) chunk = PrefillScratch::kC;
  if (chunk > PrefillScratch::kC)
    throw std::runtime_error("runtime::Engine::prefill: chunk " + std::to_string(chunk) +
                             " exceeds PrefillScratch::kC " +
                             std::to_string(PrefillScratch::kC));
  for (uint32_t id : ids)
    if (id >= model::Qwen35::kVocab)
      throw std::runtime_error(
          "runtime::Engine::prefill: id " + std::to_string(id) +
          " is outside the vocabulary (" + std::to_string(model::Qwen35::kVocab) +
          " rows) -- pf_embed_gather has no debug_flag channel, so the host is the only bound");
  const uint32_t base = control_->pos;
  if (size_t(base) + ids.size() > size_t(buffers_.max_len))
    throw std::runtime_error(
        "runtime::Engine::prefill: pos " + std::to_string(base) + " + " +
        std::to_string(ids.size()) + " ids exceeds max_len " +
        std::to_string(buffers_.max_len) +
        " -- the KV cache and the RoPE table stop there. Start a new session with reset(), or"
        " load the model and the engine with a larger max_len.");

  prepare_prefill();

  const PrefillBackend backend = prefill_backend();
  if (backend == PrefillBackend::SyclTla && !prefill::sycl_available())
    throw std::runtime_error("runtime::Engine::prefill: the sycl-tla backend needs the SYCL"
                             " component, and this build has none; set_prefill_backend(L0)");
  const char* replay_env = std::getenv("B70_PREFILL_REPLAY");
  const bool replay = pf_replay_.value_or(replay_env && std::strcmp(replay_env, "1") == 0);
  if (replay && !is_l0(backend))
    throw std::runtime_error("runtime::Engine::prefill: replay requires the L0 or l0-int8 backend");

  prefill::Int8State* q = backend == PrefillBackend::L0Int8 ? pfx_->int8.get() : nullptr;
  const prefill::AttnMode attn = prefill::attn_mode();

  // With a block hook (spec 7 §3.2) every chunk ends at a block end or at the prompt
  // end, so each completed block is a storable one; without one, uniform chunks.
  const bool hooked = static_cast<bool>(block_hook_);
  // Spec 8: with the MTP head, every chunk also fills the head's KV (step_mtp_kv), which
  // needs the chunk's post-final-norm hidden rows and h_{pos-1} (MtpBuffers::hh row 0).
  // The prefill GDN path reads slot 0 directly, so a live slot left by a commit is
  // copied there first.
  uint16_t* hid = nullptr;
  if (mtp_) {
    if (!is_l0(backend))
      throw std::runtime_error("runtime::Engine::prefill: the MTP head's prefill runs on the L0"
                               " backends only (pf_gemm)");
    mtp_normalise_live();
    verify_k_ = kNoVerify;
    if (!mtp_pf_hid_)
      mtp_pf_hid_ = std::make_unique<l0::Mem>(ctx_, l0::MemKind::Device,
                                              mtp_prefill_hidden_bytes(*model_.desc));
    hid = mtp_pf_hid_->as<uint16_t>();
  }
  const size_t hid_row = size_t(model_.desc->hidden) * 2;
  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    C = uint32_t(std::min<size_t>(chunk, ids.size() - off));
    if (hooked) C = std::min(C, kBlock - (base + uint32_t(off)) % kBlock);
    pfx_->cx.wait();                       // before touching `ids` or `Control`
    std::memcpy(pf_->ids.ptr(), ids.data() + off, size_t(C) * 4);
    control_->pos = base + uint32_t(off);
    control_->n_active = C;
    if (mtp_) {
      const uint32_t pos = base + uint32_t(off);
      hctl_->pos = pos == 0 ? 0 : pos - 1;
      hctl_->n_active = prefill::mtp_kv_rows(pos, C);
      imm_.copy(hid, mtp_->hh.ptr(), hid_row);   // h_{pos-1} into row 0
    }
    auto encode = [&] {
      prefill::step_chunk(pfx_->cx, pfx_->kc, *pf_, model_, buffers_.max_len, control_,
                        base + uint32_t(off), C, persist_.gdn_state, persist_.conv_ring,
                        persist_.kv_k, persist_.kv_v, backend, q);
      if (mtp_)
        prefill::step_mtp_kv(pfx_->cx, pfx_->kc, *pf_, model_, hctl_, base + uint32_t(off), C,
                             hid, mtp_->kv_k.as<uint16_t>(), mtp_->kv_v.as<uint16_t>());
    };
    if (replay) {
      const uint32_t pos = base + uint32_t(off);
      auto it = std::find_if(pfx_->chunks.begin(), pfx_->chunks.end(), [&](const auto& entry) {
        return entry.pos == pos && entry.rows == C && entry.backend == backend &&
               entry.attn == attn;
      });
      if (it == pfx_->chunks.end()) {
        // FIFO bound limits retained command storage for long/incremental sessions.
        if (pfx_->chunks.size() == 8) pfx_->chunks.erase(pfx_->chunks.begin());
        pfx_->chunks.push_back({pos, C, backend, attn, pfx_->cx.capture(encode)});
        it = pfx_->chunks.end() - 1;
      }
      pfx_->cx.replay(*it->recording);
    } else {
      encode();
    }
    pfx_->cx.wait();                       // the chunk's state has landed
    if (mtp_)   // h_{end-1}: the next chunk's row 0, a snapshot's hidden, the first draft's h
      imm_.copy(mtp_->hh.ptr(), reinterpret_cast<uint8_t*>(hid) + size_t(C) * hid_row, hid_row);
    const uint32_t end = base + uint32_t(off) + C;
    if (hooked && off + C < ids.size() && end % kBlock == 0) {
      // Mid-prompt block end: the state is exactly at `end`; say so in Control before
      // the hook runs, so a hook that throws leaves pos matching the chunks written.
      control_->pos = end;
      control_->n_active = 0;
      block_hook_(end, true);
    }
  }

  // The tail: pos = base + L - 1 and n_active = 1 make `argmax_stage2` leave
  // pos = base + L and cur_token[0] = the first generated id. `last` is the last
  // row of the last chunk (C rows).
  const uint32_t last = C - 1;
  control_->pos = base + uint32_t(ids.size()) - 1;
  control_->n_active = 1;
  prefill::step_head(pfx_->cx, pfx_->kc, *pf_, model_, control_, last,
                     mtp_ ? reinterpret_cast<uint8_t*>(hid) + size_t(C) * hid_row : nullptr);
  pfx_->cx.wait();
  if (hooked) {
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
}

}  // namespace runtime

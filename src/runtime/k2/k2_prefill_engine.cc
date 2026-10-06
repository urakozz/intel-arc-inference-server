// K2Engine::prefill and its accessors - the only K2Engine members not compiled into
// b70_k2_runtime (k2_engine.h says why: Engine::prefill's arrangement, engine_prefill.cc).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/k2_kernels.h"
#include "kernels/kernels.h"
#include "runtime/control.h"
#include "runtime/k2/k2_engine.h"
#include "runtime/k2/k2_prefill.h"
#include "runtime/k2/k2_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace runtime::k2 {

struct K2PrefillState {
  prefill::Context cx;
  prefill::KernelCache kc;
  K2PrefillScratch s;
  struct Chunk {
    uint32_t pos, rows;
    bool eager;   // a recording holds one attention variant's walk
    std::unique_ptr<prefill::Context::Recording> recording;
  };
  // Declared after cx / kc / s: recordings are destroyed first.
  std::vector<Chunk> chunks;
  K2PrefillState(l0::Context& c, const model::K2Desc& d) : cx(c), kc(c), s(c, d) {}
};

namespace {
void destroy_prefill(K2PrefillState* p) { delete p; }
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::k2::K2Engine::prefill: " + what);
}
}  // namespace

void K2Engine::prepare_prefill() {
  if (pf_) return;
  const model::K2Desc& d = *model_.desc;
  // A missing binary is named here, before anything is allocated or appended (the decode
  // capture's rule); the head's binaries are decode's, which the capture already checked.
  const bool kv8 = buffers_.kv_cache() == KvCache::Int8;   // spec 18e: the int8 cache's binaries
  for (const std::string& v : kernels::k2::prefill_variants(d, kv8))
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_K2=ON (spec 18c)" +
                (kv8 ? " and B70_KV8=ON (spec 18e)" : ""));
  pf_ = std::unique_ptr<K2PrefillState, void (*)(K2PrefillState*)>(new K2PrefillState(ctx_, d),
                                                                    &destroy_prefill);
  pf_bytes_ = pf_->s.bytes();
  require(pf_bytes_ == prefill_sizes(d).total(), "the prefill scratch is not prefill_sizes()'s");
}

size_t K2Engine::prefill_launches() const { return pf_ ? pf_->cx.launches() : 0; }

std::vector<uint32_t> K2Engine::read_prefill_routes() {
  require(bool(pf_), "read_prefill_routes before the first prefill");
  pf_->cx.wait();
  std::vector<uint32_t> v(pf_->s.routes.size() / 4);
  imm_.copy(v.data(), pf_->s.routes.ptr(), pf_->s.routes.size());
  return v;
}

void K2Engine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::K2Desc& d = *model_.desc;
  require(!ids.empty(), "no ids");
  if (chunk == 0) chunk = kPfC;
  require(chunk <= kPfC, "chunk " + std::to_string(chunk) + " exceeds kPfC " + std::to_string(kPfC));
  for (uint32_t id : ids)
    require(id < d.vocab, "id " + std::to_string(id) + " is outside the vocabulary (" +
                              std::to_string(d.vocab) + " rows) - pf_embed_gather has no "
                              "debug_flag channel, so the host is the only bound");
  const uint32_t base = control_->pos;
  require(size_t(base) + ids.size() <= buffers_.max_len,
          "pos " + std::to_string(base) + " + " + std::to_string(ids.size()) + " ids exceeds max_len " +
              std::to_string(buffers_.max_len) + " - the KV cache and the RoPE table stop there");

  prepare_prefill();
  const bool eager = prefill_attn_eager();
  const char* env = std::getenv("B70_PREFILL_REPLAY");
  const bool replay = pf_replay_ >= 0 ? pf_replay_ == 1 : (env && std::strcmp(env, "1") == 0);

  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    C = uint32_t(std::min<size_t>(chunk, ids.size() - off));
    const uint32_t pos = base + uint32_t(off);
    pf_->cx.wait();   // before touching the ids buffer or Control
    std::memcpy(pf_->s.ids.ptr(), ids.data() + off, size_t(C) * 4);
    control_->pos = pos;
    control_->n_active = C;
    auto encode = [&] { prefill_chunk(pf_->cx, pf_->kc, pf_->s, model_, buffers_, pos, C, eager); };
    if (replay) {
      auto it = std::find_if(pf_->chunks.begin(), pf_->chunks.end(), [&](const K2PrefillState::Chunk& c) {
        return c.pos == pos && c.rows == C && c.eager == eager;
      });
      if (it == pf_->chunks.end()) {
        // A FIFO bound keeps the retained command storage of long sessions finite.
        if (pf_->chunks.size() == 8) pf_->chunks.erase(pf_->chunks.begin());
        pf_->chunks.push_back({pos, C, eager, pf_->cx.capture(encode)});
        it = pf_->chunks.end() - 1;
      }
      pf_->cx.replay(*it->recording);
    } else {
      encode();
    }
    pf_->cx.wait();   // the chunk's KV rows have landed
  }

  // The tail: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave pos = base + L
  // and the first generated id in cur_token[0]; `C - 1` is the last chunk's last row.
  control_->pos = base + uint32_t(ids.size()) - 1;
  control_->n_active = 1;
  prefill_head(pf_->cx, pf_->kc, pf_->s, model_, buffers_, C - 1);
  pf_->cx.wait();
}

}  // namespace runtime::k2

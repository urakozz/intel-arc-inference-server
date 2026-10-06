#include "runtime/pipeline_place.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "l0/cmdlist.h"
#include "l0/memory.h"

namespace runtime {
namespace {

size_t weight_bytes(const loader::DeviceWeight& w) {
  return w.mem.size() + (w.scales ? w.scales->size() : 0);
}
size_t moe_layer_device_bytes(const loader::MoeLayer& l) {
  return weight_bytes(l.router) + l.gate_up.size() + l.down.size();
}

// Moves device allocations from device 0 to device 1: allocate on device 1, copy through
// one USM host staging buffer in bounded chunks (no peer access needed, no full-size host
// copy of a 2.5 GB lm_head), return the new allocation. The source is freed by its owner.
class Mover {
 public:
  Mover(l0::Context& d0, l0::Context& d1)
      : d1_(d1),
        imm0_(l0::CmdList::immediate(d0)),
        imm1_(l0::CmdList::immediate(d1)),
        stage_(d0, l0::MemKind::Host, kChunk) {}

  l0::Mem move(const l0::Mem& src) {
    l0::Mem dst(d1_, l0::MemKind::Device, src.size());
    for (size_t off = 0; off < src.size(); off += kChunk) {
      const size_t n = std::min(kChunk, src.size() - off);
      imm0_.copy(stage_.ptr(), static_cast<const uint8_t*>(src.ptr()) + off, n);
      imm1_.copy(static_cast<uint8_t*>(dst.ptr()) + off, stage_.ptr(), n);
    }
    return dst;
  }
  loader::DeviceWeight move(const loader::DeviceWeight& w) {
    return loader::DeviceWeight{move(w.mem),
                                w.scales ? std::make_unique<l0::Mem>(move(*w.scales)) : nullptr,
                                w.shape, w.kind};
  }

 private:
  static constexpr size_t kChunk = size_t{64} << 20;   // 64 MiB of pinned host memory
  l0::Context& d1_;
  l0::CmdList imm0_, imm1_;
  l0::Mem stage_;
};

}  // namespace

PpWeights pp_weights(const loader::LoadedModel& m) {
  if (!m.desc) throw std::invalid_argument("runtime::pp_weights: the model has no descriptor");
  const model::ModelDesc& d = *m.desc;
  if (m.layer_small.size() != d.layers || (d.is_moe() && m.moe.size() != d.layers))
    throw std::invalid_argument("runtime::pp_weights: the model does not hold all " +
                                std::to_string(d.layers) + " layers (a stage model?)");
  PpWeights w;
  w.layer.assign(d.layers, 0);
  for (const auto& [key, dw] : m.linears) {
    if (key.first == loader::kTopLevel)
      w.head += weight_bytes(dw);
    else
      w.layer.at(key.first) += weight_bytes(dw);
  }
  for (uint32_t l = 0; l < d.layers; ++l) {
    w.layer[l] += m.layer_small[l].norms.size() + m.layer_small[l].gdn.size();
    if (!m.moe.empty()) w.layer[l] += moe_layer_device_bytes(m.moe[l]);
  }
  for (const auto& [layer, dw] : m.ab_prefill) w.layer.at(layer) += weight_bytes(dw);
  w.embed = m.embed.size();
  w.head += m.final_norm.size();
  return w;
}

size_t model_device_bytes(const loader::LoadedModel& m) {
  size_t t = m.embed.size() + m.final_norm.size() + m.rope.size();
  for (const auto& kv : m.linears) t += weight_bytes(kv.second);
  for (const loader::SmallTensors& s : m.layer_small) t += s.norms.size() + s.gdn.size();
  for (const loader::MoeLayer& l : m.moe) t += moe_layer_device_bytes(l);
  for (const auto& kv : m.ab_prefill) t += weight_bytes(kv.second);
  if (m.draft_vocab) t += m.draft_vocab->bytes();
  return t;
}

std::vector<loader::LoadedModel> place_stages(l0::Context& d0, l0::Context& d1,
                                              loader::LoadedModel full, uint32_t split) {
  if (!full.desc) throw std::invalid_argument("runtime::place_stages: the model has no descriptor");
  const model::ModelDesc& d = *full.desc;
  if (full.mtp)
    throw std::runtime_error("pipeline parallel with the MTP head is spec 16d (the head and its "
                             "embedding on device 1); load without --mtp");
  if (full.draft_vocab)
    throw std::runtime_error("pipeline parallel with a draft vocabulary is spec 16d");
  if (d1.handle() != d0.handle())
    throw std::invalid_argument("runtime::place_stages: device 1 is not a view of device 0's "
                                "Level Zero context (spec 16 decision 1: one context)");
  require_split(d, split);
  if (full.layer_small.size() != d.layers || (d.is_moe() && full.moe.size() != d.layers))
    throw std::invalid_argument("runtime::place_stages: the model does not hold all " +
                                std::to_string(d.layers) + " layers");

  Mover mv(d0, d1);
  // Device 1: the final norm and the RoPE table now, a placeholder for the embedding (no
  // stage-1 list binds it). Braced initialisers run left to right.
  loader::LoadedModel s1{{},
                         {},
                         l0::Mem(d1, l0::MemKind::Device, 64),
                         mv.move(full.final_norm),
                         mv.move(full.rope),
                         {},
                         full.max_len,
                         full.trained_max_len,
                         nullptr,
                         full.desc,
                         nullptr,
                         {},
                         {}};
  // Layers [split, layers) and the top-level linears (lm_head): moved, then freed on
  // device 0 as each one goes, so device 0's peak is the whole model and no more.
  for (auto it = full.linears.begin(); it != full.linears.end();) {
    if (it->first.first >= split) {   // kTopLevel (65535) is past every layer
      s1.linears.emplace(it->first, mv.move(it->second));
      it = full.linears.erase(it);
    } else {
      ++it;
    }
  }
  s1.layer_small.reserve(d.layers - split);
  for (uint32_t l = split; l < d.layers; ++l)
    s1.layer_small.push_back(loader::SmallTensors{mv.move(full.layer_small[l].norms),
                                                  mv.move(full.layer_small[l].gdn)});
  while (full.layer_small.size() > split) full.layer_small.pop_back();
  if (!full.moe.empty()) {
    s1.moe.reserve(d.layers - split);
    for (uint32_t l = split; l < d.layers; ++l) {
      const loader::MoeLayer& src = full.moe[l];
      s1.moe.push_back(loader::MoeLayer{mv.move(src.router), mv.move(src.gate_up), mv.move(src.down)});
    }
    while (full.moe.size() > split) full.moe.pop_back();
  }
  // Spec 15 §13: an int4 a||b's bf16 prefill copies (keyed by layer) go with their layers.
  // Decode never reads them (16b refuses prefill); 16c's per-device prefill will.
  for (auto it = full.ab_prefill.begin(); it != full.ab_prefill.end();) {
    if (it->first >= split) {
      s1.ab_prefill.emplace(it->first, mv.move(it->second));
      it = full.ab_prefill.erase(it);
    } else {
      ++it;
    }
  }
  // What loader::set_max_len reads to re-table the RoPE later (its other fields describe the
  // load, which stage 0's report keeps).
  s1.report.rope_bytes = full.report.rope_bytes;
  s1.report.small_bytes = full.report.rope_bytes + s1.final_norm.size();
  // Device 0 no longer needs the final norm: a placeholder in its place frees it.
  {
    l0::Mem tiny(d0, l0::MemKind::Device, 64);
    full.final_norm.swap(tiny);
  }

  std::vector<loader::LoadedModel> out;
  out.reserve(kPpDevices);
  out.push_back(std::move(full));
  out.push_back(std::move(s1));
  return out;
}

}  // namespace runtime

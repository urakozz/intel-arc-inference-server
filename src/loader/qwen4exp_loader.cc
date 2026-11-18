#include "loader/qwen4exp_loader.h"

#include <sys/stat.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string_view>
#include <print>
#include <stdexcept>

#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/qwen4exp_repack.h"
#include "loader/qwen4exp_tp_split.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/memory_plan.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace loader {
namespace {
std::unique_ptr<DeviceWeight> upload_w(l0::Context& ctx, l0::CmdList& imm, const Q4HostWeight& w, size_t& bytes) {
  bytes += w.bytes();
  if (w.kind == model::WeightKind::Int4)
    return std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, w.words.data(), w.words.size() * 4),
                                                       upload_u(ctx, imm, w.scales.data(), w.scales.size() * 2), w.shape,
                                                       model::WeightKind::Int4});
  return std::make_unique<DeviceWeight>(
      DeviceWeight{upload(ctx, imm, w.tiles.data(), w.tiles.size() * 2), nullptr, w.shape, model::WeightKind::Bf16});
}

// Plan 21-QFN-v2a Task 8: a layer's routed experts by the tier's row `res` ([experts], 1 = VRAM): the resident experts'
// blocks into the layer's VRAM arena (id order), the rest into its pinned host mirror (gate||up blocks, then down
// blocks, id order); `row` gets the 1024 block addresses (gate||up of expert e at [e], down at [512 + e]).
void upload_experts(l0::Context& ctx, l0::CmdList& imm, const Q4HostLayer& h, uint32_t experts, const uint8_t* res,
                    Q4Layer& L, size_t& bytes, size_t& host_bytes, uint64_t* row) {
  const size_t gu = h.gate_up.size() * 4 / experts, dn = h.down.size() * 4 / experts;
  const uint8_t* gsrc = reinterpret_cast<const uint8_t*>(h.gate_up.data());
  const uint8_t* dsrc = reinterpret_cast<const uint8_t*>(h.down.data());
  uint32_t nv = 0;
  for (uint32_t e = 0; e < experts; ++e) nv += res[e] != 0;
  const uint32_t nh = experts - nv;
  if (nv != 0) {
    std::vector<uint8_t> g(size_t(nv) * gu), dd(size_t(nv) * dn);
    for (uint32_t e = 0, s = 0; e < experts; ++e)
      if (res[e]) {
        std::memcpy(g.data() + size_t(s) * gu, gsrc + size_t(e) * gu, gu);
        std::memcpy(dd.data() + size_t(s) * dn, dsrc + size_t(e) * dn, dn);
        ++s;
      }
    L.gate_up = upload_u(ctx, imm, g.data(), g.size());
    L.down = upload_u(ctx, imm, dd.data(), dd.size());
    bytes += g.size() + dd.size();
  }
  if (nh != 0) {
    // One pinned range a layer, at most experts x 2.6 MB (1.34 GB): under the 4 GiB a range both cards may map.
    L.mirror = std::make_unique<l0::Mem>(ctx, l0::MemKind::HostLocked, size_t(nh) * (gu + dn), size_t(2) << 20);
    uint8_t* m = L.mirror->as<uint8_t>();
    for (uint32_t e = 0, s = 0; e < experts; ++e)
      if (!res[e]) {
        std::memcpy(m + size_t(s) * gu, gsrc + size_t(e) * gu, gu);
        std::memcpy(m + size_t(nh) * gu + size_t(s) * dn, dsrc + size_t(e) * dn, dn);
        ++s;
      }
    host_bytes += L.mirror->size();
  }
  if (!row) return;
  for (uint32_t e = 0, s = 0, t = 0; e < experts; ++e) {
    if (res[e]) {
      row[e] = uint64_t(uintptr_t(L.gate_up->as<uint8_t>() + size_t(s) * gu));
      row[experts + e] = uint64_t(uintptr_t(L.down->as<uint8_t>() + size_t(s) * dn));
      ++s;
    } else {
      row[e] = uint64_t(uintptr_t(L.mirror->as<uint8_t>() + size_t(t) * gu));
      row[experts + e] = uint64_t(uintptr_t(L.mirror->as<uint8_t>() + size_t(nh) * gu + size_t(t) * dn));
      ++t;
    }
  }
}

// One host layer onto its device (the fields of the other layer kind stay null). `res` / `row`: the expert tier
// (upload_experts); null `res`: every expert in one VRAM range, as before the tier.
Q4Layer upload_layer(l0::Context& ctx, l0::CmdList& imm, const Q4HostLayer& h, size_t& bytes,
                     const uint8_t* res = nullptr, uint32_t experts = 0, size_t* host_bytes = nullptr,
                     uint64_t* row = nullptr) {
  Q4Layer L;
  const auto blob = [&](const std::vector<uint8_t>& v) {
    bytes += v.size();
    return v.empty() ? nullptr : upload_u(ctx, imm, v.data(), v.size());
  };
  L.hc_attn = blob(h.hc_attn);
  L.hc_mlp = blob(h.hc_mlp);
  if (h.gdn_qkvz.bytes()) {
    L.gdn_qkvz = upload_w(ctx, imm, h.gdn_qkvz, bytes);
    L.gdn_ab = upload_w(ctx, imm, h.gdn_ab, bytes);
    L.gdn_out = upload_w(ctx, imm, h.gdn_out, bytes);
    L.gdn_small = blob(h.gdn_small);
  }
  if (h.qsa_qkvg.bytes()) {
    L.qsa_qkvg = upload_w(ctx, imm, h.qsa_qkvg, bytes);
    L.qsa_idx = upload_w(ctx, imm, h.qsa_idx, bytes);
    L.qsa_o = upload_w(ctx, imm, h.qsa_o, bytes);
    L.qsa_small = blob(h.qsa_small);
  }
  L.router = upload_w(ctx, imm, h.router, bytes);
  if (res) {
    upload_experts(ctx, imm, h, experts, res, L, bytes, *host_bytes, row);
  } else {
    L.gate_up = upload_u(ctx, imm, h.gate_up.data(), h.gate_up.size() * 4);
    L.down = upload_u(ctx, imm, h.down.data(), h.down.size() * 4);
    bytes += (h.gate_up.size() + h.down.size()) * 4;
  }
  L.shared = blob(h.shared);
  L.shared_down_offset = h.shared_gate_up;
  L.ple = blob(h.ple);
  return L;
}

// The checkpoint's PLE I64 tensors against the formula for `d` (Review Focus 3): refused naming the tensor.
void check_ple_constants(const model::Qwen4ExpDesc& d, const Q4PleConstants& c) {
  const std::string p = model::Qwen4ExpDesc::layer_prefix(d.ple_layer) + "ple.ple_embedding.";
  if (c.multipliers != q4_ple_multipliers(d.vocab, d.ngram, 0, d.ple_seed))
    throw std::runtime_error("load_qwen4exp: '" + p + "layer_multipliers' differs from the formula (M:1040-1049, seed " +
                             std::to_string(d.ple_seed) + ")");
  const std::vector<uint64_t> sizes = q4_ple_primes(d.ple_base, d.ple_heads, 0);
  if (c.sizes != sizes)
    throw std::runtime_error("load_qwen4exp: '" + p + "ngram_heads_vocab_sizes' differs from the primes after " +
                             std::to_string(d.ple_base) + " (ngram_vocab_size_base)");
  if (c.offsets != q4_ple_offsets(sizes))
    throw std::runtime_error("load_qwen4exp: '" + p + "ngram_heads_offsets' is not the head sizes' running sum");
}

struct Opened {
  std::string snap;
  common::json::Value cfg;
};
Opened open(const std::string& snapshot_or_repo) {
  Opened o;
  o.snap = resolve_snapshot(snapshot_or_repo);
  o.cfg = read_json(o.snap + "config.json");
  const std::string mt = model_type_of(o.cfg);
  if (!model::is_qwen4exp_model_type(mt))
    throw std::runtime_error("load_qwen4exp: " + o.snap + " is model_type '" + mt + "', not qwen4_exp");
  return o;
}

model::Qwen4ExpDesc limited(const model::Qwen4ExpDesc& full, uint32_t layers_limit) {
  if (layers_limit == 0) return full;
  if (layers_limit > full.layers)
    throw std::runtime_error("--layers " + std::to_string(layers_limit) + ": the checkpoint has " +
                             std::to_string(full.layers) + " layers");
  return runtime::qwen4exp::truncated(full, layers_limit);
}

// Plan 21-QFN-v2b: load_qwen4exp under the tensor-parallel placement (qwen4exp_loader.h). Each layer is repacked once and
// cut into both cards' halves; everything else as the one-card load, per card.
Q4LoadedModel load_tp(const std::vector<l0::Context*>& devices, const Opened& o, Q4Checkpoint& ck,
                      const SafetensorsSet& set, const model::Qwen4ExpDesc& full, const model::Qwen4ExpDesc& d,
                      const QuantScan& scan, uint32_t max_len, const model::Q4Placement& placement, LmHeadForm lm_form,
                      const std::array<size_t, runtime::kPpDevices>& caps, size_t reserve, const char* forced,
                      const runtime::qwen4exp::ExpertTier* tier, const runtime::qwen4exp::ExpertTier* tier1, bool mtp,
                      std::chrono::steady_clock::time_point t0) {
  namespace rq = runtime::qwen4exp;
  const bool int8 = lm_form == LmHeadForm::Int8;
  const model::Qwen4ExpDesc card[2] = {model::q4_tp_card(d, 0), model::q4_tp_card(d, 1)};
  rq::ExpertTier T[2];
  for (uint32_t r = 0; r < 2; ++r) {
    const rq::ExpertTier* given = r == 0 ? tier : tier1;
    T[r] = given ? *given
           : forced && *forced
               ? rq::tp_fill_tier(d, rq::tp_expert_budget(d, placement, max_len, int8, mtp, true, caps, reserve), r,
                                  nullptr, forced)
               : rq::all_resident(d);
    if (T[r].layers != d.layers || T[r].experts != d.experts)
      throw std::invalid_argument("load_qwen4exp: card " + std::to_string(r) + "'s expert tier is not " +
                                  std::to_string(d.layers) + " layers x " + std::to_string(d.experts) + " experts");
  }
  const char* ex = std::getenv("B70_Q4_EXPERTS");
  const std::string form = ex && *ex ? ex : "table";
  if (form != "table" && form != "direct")
    throw std::runtime_error("B70_Q4_EXPERTS=" + form + ": table (the default) or direct");
  if (form == "direct" && !(T[0].all() && T[1].all()))
    throw std::runtime_error("B70_Q4_EXPERTS=direct reads every expert from the layer's VRAM range, but experts are in "
                             "host memory: unset it");
  {
    const std::vector<rq::DevicePlan> pl = rq::tp_plan(d, placement, max_len, int8, mtp);
    std::vector<rq::DevicePlan> fit;
    for (uint32_t r = 0; r < 2; ++r) {
      std::vector<rq::DevicePlan> one{pl[r]};
      rq::apply_tier(one, card[r], T[r]);
      one[0].device = r;
      fit.push_back(one[0]);
    }
    rq::require_fits(fit, caps, reserve);
  }
  const std::string ple_dir = q4_ple_dir(o.snap);
  check_ple_constants(d, ck.ple_constants());

  Q4LoadedModel m;
  m.desc = d;
  m.placement = placement;
  m.max_len = max_len;
  m.int8_head = int8;
  m.mtp = mtp;
  m.checkpoint_layers = full.layers;
  m.subnormal_scales = scan.subnormal_scales;
  if (d.layers < full.layers) m.skipped += ck.skip_layers_from(d.layers);
  const size_t visual = ck.skip_prefix("model.visual."), shards = ck.skip_ple_shards();
  const size_t mtp_skipped = mtp ? 0 : ck.skip_prefix("mtp.");
  check_q4_host_fit(Q4PleHost(ple_dir).bytes(), T[0].host_bytes(card[0]) + T[1].host_bytes(card[1]), 0,
                    q4_mem_available());
  m.ple_device = 0;
  m.ple = load_q4_ple(*devices[0], d, ple_dir);
  {
    std::vector<uint64_t> ptrs(m.ple.ptrs->size() / 8);
    l0::CmdList imm0 = l0::CmdList::immediate(*devices[0]);
    imm0.copy(ptrs.data(), m.ple.ptrs->ptr(), ptrs.size() * 8);
    l0::CmdList imm1 = l0::CmdList::immediate(*devices[1]);
    m.ple_ptrs1 = upload_u(*devices[1], imm1, ptrs.data(), ptrs.size() * 8);
  }
  m.tier = T[0];
  m.tier1 = T[1];
  m.expert_table = form == "table";

  const std::vector<float> rope = q4_rope_table(d, max_len);
  std::vector<l0::CmdList> imm;
  std::vector<std::vector<uint64_t>> rows(2);
  m.parts.resize(2);
  for (uint32_t r = 0; r < 2; ++r) {
    imm.push_back(l0::CmdList::immediate(*devices[r]));
    Q4DevicePart& P = m.parts[r];
    P.device = r;
    P.first = 0;
    P.end = d.layers;
    if (placement.holds_embed(r)) {
      P.embed = upload_u(*devices[r], imm[r], ck.embed(), q4_embed_bytes(d));
      P.bytes += q4_embed_bytes(d);
    }
    P.rope = upload_u(*devices[r], imm[r], rope.data(), rope.size() * 4);
    P.rope_bytes = rope.size() * 4;
    if (m.expert_table) rows[r].assign(size_t(d.layers) * 2 * d.experts, 0);
  }
  Q4HostLayer host;
  for (uint32_t l = 0; l < d.layers; ++l) {
    ck.repack_layer(l, host);
    for (uint32_t r = 0; r < 2; ++r) {
      const Q4HostLayer half = q4_tp_layer(d, l, host, r);
      Q4DevicePart& P = m.parts[r];
      uint64_t* row = rows[r].empty() ? nullptr : rows[r].data() + size_t(l) * 2 * d.experts;
      if (T[r].n_vram(l) == d.experts && !row)
        P.layers.push_back(upload_layer(*devices[r], imm[r], half, P.bytes));
      else
        P.layers.push_back(upload_layer(*devices[r], imm[r], half, P.bytes, T[r].vram.data() + size_t(l) * d.experts,
                                        d.experts, &m.mirror_bytes, row));
      if (!row) continue;
      const auto s0 = std::chrono::steady_clock::now();
      const uint32_t base = uint32_t(row - rows[r].data());
      for (uint32_t i = 0; i < 2 * d.experts; ++i) {   // the load check's samples, as one card's
        const bool up = i < d.experts;
        const size_t B = (up ? half.gate_up.size() : half.down.size()) * 4 / d.experts;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(up ? half.gate_up.data() : half.down.data()) +
                             size_t(up ? i : i - d.experts) * B;
        const auto take = [&](size_t off) {
          uint64_t w = 0;
          std::memcpy(&w, src + off, 8);
          P.check_idx.push_back(base + i);
          P.check_off.push_back(uint32_t(off));
          P.check_words.push_back(w);
        };
        for (size_t off = 0; off + 8 < B; off += size_t(1) << 20) take(off);
        take(B - 8);
      }
      m.expert_check_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
    }
  }
  const std::vector<uint8_t> fm = ck.final_mixer();
  const uint16_t* lm = ck.lm_head();
  for (uint32_t r = 0; r < 2; ++r) {
    l0::Context& ctx = *devices[r];
    Q4DevicePart& P = m.parts[r];
    if (!rows[r].empty()) {
      P.expert_table = upload_u(ctx, imm[r], rows[r].data(), rows[r].size() * 8);
      std::vector<uint64_t> back(rows[r].size());
      imm[r].copy(back.data(), P.expert_table->ptr(), back.size() * 8);
      if (back != rows[r])
        throw std::runtime_error("load_qwen4exp: card " + std::to_string(r) + "'s expert table reads back different "
                                 "addresses than were written");
      for (uint32_t l = 0; l < d.layers; ++l)
        P.layers[l].expert_row = P.expert_table->as<uint8_t>() + size_t(l) * 2 * d.experts * 8;
    }
    P.final_mixer = upload_u(ctx, imm[r], fm.data(), fm.size());
    P.bytes += fm.size();
    P.lm_head = upload_lm_head(ctx, imm[r], lm + size_t(r) * card[r].vocab * d.hidden, d.hidden, card[r].vocab, int8,
                               m.lm_head_quant_seconds, P.bytes);
    const bool head = mtp && placement.holds_mtp(r);   // the MTP head whole, one card's layer (Task 5 Step 0)
    if (head) {
      ck.repack_mtp(host);
      P.mtp = std::make_unique<Q4Layer>(upload_layer(ctx, imm[r], host, P.bytes));
      const std::vector<uint8_t> fc = ck.mtp_fc(), mx = ck.mtp_mixer();
      P.mtp_fc = upload_u(ctx, imm[r], fc.data(), fc.size());
      P.mtp_mixer = upload_u(ctx, imm[r], mx.data(), mx.size());
      P.bytes += fc.size() + mx.size();
    }
    size_t host_experts = 0;
    for (uint32_t l = 0; l < d.layers; ++l)
      host_experts += size_t(d.experts - T[r].n_vram(l)) * (q4_gate_up_block_bytes(card[r]) + q4_down_block_bytes(card[r]));
    const size_t want = q4_device_weight_bytes(card[r], model::Q4Placement::one(card[r]), 0, int8, false) -
                        q4_embed_bytes(card[r]) + (placement.holds_embed(r) ? q4_embed_bytes(d) : 0) - host_experts +
                        (head ? q4_mtp_bytes(d) : 0);
    if (P.bytes != want)
      throw std::logic_error("load_qwen4exp: card " + std::to_string(r) + " allocated " + std::to_string(P.bytes) +
                             " B of weights, the card plan says " + std::to_string(want));
  }
  std::string names;
  m.unconsumed = ck.unconsumed(&names);
  if (m.unconsumed != 0)
    throw std::runtime_error("load_qwen4exp: " + std::to_string(m.unconsumed) + " checkpoint tensors were not loaded: " +
                             names);
  for (uint32_t l = 0; l < d.layers; ++l) m.read_per_token += q4_layer_read_per_token(card[0], l);
  m.read_per_token += q4_final_mixer_bytes(d) + q4_lm_head_bytes(card[0], int8);
  m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double gb = 1e9;
  std::print("load_qwen4exp: {}\n  model       {} ({}): {:d} of {:d} layers, tensor parallel over two cards (the "
             "embedding {}), dense {}, routed experts int4 g{:d}; {:d} tensors, 0 unconsumed\n",
             o.snap.c_str(), d.name.c_str(), d.architecture.c_str(), d.layers, full.layers,
             placement.ends == model::Q4TpEnds::Both ? "on both cards" : "on card 0",
             model::q4_form_name(d.forms.dense), d.forms.expert_group, set.tensors().size());
  std::print("  skipped     by design: {:d} visual tensors, {:d} bf16 PLE shards{}\n", visual, shards,
             mtp ? "" : (", " + std::to_string(mtp_skipped) + " mtp.* (the head not loaded)").c_str());
  for (const Q4DevicePart& P : m.parts) {
    uint32_t nv = 0;
    for (uint32_t l = 0; l < d.layers; ++l) nv += (P.device == 0 ? T[0] : T[1]).n_vram(l);
    std::print("  card {:d}      its half of every layer: weights {:13d} B {:7.3f} GB{} + the {} head's half{}, {:d} of "
               "{:d} experts' halves in VRAM (f {:.3f})\n", P.device, P.bytes, P.bytes / gb, P.embed ? " + embed" : "",
               int8 ? "int8" : "bf16", P.mtp ? " + the MTP head whole (RTN int4 g64 experts)" : "", nv,
               d.layers * d.experts, double(nv) / (d.layers * d.experts));
  }
  std::print("  host        the PLE table {:.3f} GB (pointer tables on both cards), the experts' halves {:.3f} GB pinned; "
             "read/token a card {:.3f} GB (derived); load {:.1f} s\n", m.ple.bytes / gb, m.mirror_bytes / gb,
             m.read_per_token / gb, m.seconds);
  return m;
}

}  // namespace

size_t Q4LoadedModel::total_bytes() const {
  size_t t = 0;
  for (const Q4DevicePart& p : parts) t += p.bytes + p.rope_bytes;
  return t;
}

bool is_qwen4exp_checkpoint(const std::string& snapshot_dir) {
  return model::is_qwen4exp_model_type(model_type_of(read_json(snapshot_dir + "config.json")));
}

std::vector<uint32_t> load_q4_profile(const std::string& snapshot_or_repo, const model::Qwen4ExpDesc& full,
                                      std::string* path_out) {
  std::string s = resolve_snapshot(snapshot_or_repo);
  while (!s.empty() && s.back() == '/') s.pop_back();
  const std::string path = s + "-expert-profile.safetensors";
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) return {};
  if (path_out) *path_out = path;
  const MappedFile f(path);
  uint64_t hlen = 0;
  if (f.size() >= 8) std::memcpy(&hlen, f.data(), 8);
  if (f.size() < 8 || hlen > f.size() - 8 ||
      std::string_view(reinterpret_cast<const char*>(f.data()) + 8, hlen).find("\"spec22-profile-1\"") ==
          std::string_view::npos)
    throw std::runtime_error(path + " is not a routing profile written by tools/oracle/offload_curve.py profile - "
                             "move it away, or remake it with that command");
  for (const auto& [name, t] : SafetensorsSet::parse_header(f.data(), f.size())) {
    if (name != "counts") continue;
    const uint64_t n = uint64_t(full.layers) * full.experts;
    if (t.dtype != "U32" || t.shape.size() != 2 || t.shape[0] != full.layers || t.shape[1] != full.experts ||
        t.end - t.begin != n * 4 || 8 + hlen + t.end > f.size())
      throw std::runtime_error(path + ": 'counts' is not u32 [" + std::to_string(full.layers) + "][" +
                               std::to_string(full.experts) + "] - a profile of another model");
    std::vector<uint32_t> c(n);
    std::memcpy(c.data(), f.data() + 8 + hlen + t.begin, n * 4);
    return c;
  }
  throw std::runtime_error(path + " holds no 'counts' tensor (offload_curve.py profile writes it)");
}

model::Qwen4ExpDesc qwen4exp_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit) {
  const Opened o = open(snapshot_or_repo);
  const SafetensorsSet set(o.snap);   // the headers only: the forms are read from the names
  const model::Qwen4ExpDesc pre = model::qwen4exp_desc(o.cfg, model::Q4Forms{});
  return limited(model::qwen4exp_desc(o.cfg, q4_forms(set, pre)), layers_limit);
}

Q4LoadedModel load_qwen4exp(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                            uint32_t max_len, const model::Q4Placement& placement, LmHeadForm lm_form,
                            uint32_t layers_limit, bool mtp, const runtime::qwen4exp::ExpertTier* tier,
                            const runtime::qwen4exp::ExpertTier* tier1) {
  const auto t0 = std::chrono::steady_clock::now();
  const Opened o = open(snapshot_or_repo);
  const QuantConfig qc = check_qwen4exp_checkpoint_config(o.cfg);
  SafetensorsSet set(o.snap);
  // The checkpoint's own descriptor (every layer), then its forms from the names.
  const model::Qwen4ExpDesc pre = model::qwen4exp_desc(o.cfg, model::Q4Forms{});
  const model::Qwen4ExpDesc full = model::qwen4exp_desc(o.cfg, q4_forms(set, pre));
  if (max_len == 0) throw std::runtime_error(full.name + ": max_len 0 is not a model length");
  if (max_len > full.trained_max_len)
    throw std::runtime_error(full.name + ": max_len " + std::to_string(max_len) + " exceeds the trained context " +
                             std::to_string(full.trained_max_len));
  if (max_len % runtime::kMaxLenQuantum != 0)
    throw std::runtime_error(full.name + ": max_len " + std::to_string(max_len) + " is not a multiple of " +
                             std::to_string(runtime::kMaxLenQuantum) + " (the indexer's blocks of 4 and the prefill grid)");
  const QuantScan scan = assert_quant_invariants(set);
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error("load_qwen4exp: config.json declares no desc_act but the checkpoint ships " +
                             std::to_string(scan.g_idx_tensors) + " g_idx tensors");
  check_quant_scan(qc, scan);
  if (scan.ct_linears != 0)
    throw std::runtime_error("load_qwen4exp: compressed-tensors linears in a qwen4_exp checkpoint - this engine serves "
                             "AutoRound auto_round:auto_gptq exports only (one format)");
  if (qc.group_size != full.forms.expert_group)
    throw std::runtime_error("load_qwen4exp: quantization_config.group_size is " + std::to_string(qc.group_size) +
                             " but the routed experts' scales are g" + std::to_string(full.forms.expert_group) +
                             " - the label and the bytes disagree");
  Q4Checkpoint ck(full, set);
  ck.check_names(mtp);   // by name, both ways, before a byte is repacked

  const model::Qwen4ExpDesc d = limited(full, layers_limit);
  model::validate(placement, d);
  if (devices.size() != placement.devices)
    throw std::invalid_argument("load_qwen4exp: " + std::to_string(devices.size()) + " device(s) for a " +
                                std::to_string(placement.devices) + "-device placement");
  const bool int8 = lm_form == LmHeadForm::Int8;
  // The expert tier (plan 21-QFN-v2a Task 8): where each routed expert lives, and which form the layers bind. Without
  // one, every expert is resident - or B70_Q4_RESIDENT's test placement (any caller: the golden tests and tools run
  // under it unchanged).
  std::array<size_t, runtime::kPpDevices> caps{};
  for (uint32_t i = 0; i < placement.devices; ++i) caps[i] = devices[i]->memory_bytes();
  const size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const char* forced = std::getenv("B70_Q4_RESIDENT");
  if (placement.tp) {
    return load_tp(devices, o, ck, set, full, d, scan, max_len, placement, lm_form, caps, reserve, forced, tier, tier1, mtp,
                   t0);
  }
  const runtime::qwen4exp::ExpertTier T =
      tier ? *tier
      : forced && *forced
          ? runtime::qwen4exp::fill_tier(
                d, runtime::qwen4exp::expert_budget(d, placement, max_len, lm_form == LmHeadForm::Int8, mtp, true, caps, reserve),
                nullptr, forced)
          : runtime::qwen4exp::all_resident(d);
  if (T.layers != d.layers || T.experts != d.experts)
    throw std::invalid_argument("load_qwen4exp: the expert tier is not " + std::to_string(d.layers) + " layers x " +
                                std::to_string(d.experts) + " experts");
  const char* ex = std::getenv("B70_Q4_EXPERTS");
  const std::string form = ex && *ex ? ex : "table";
  if (form != "table" && form != "direct")
    throw std::runtime_error("B70_Q4_EXPERTS=" + form + ": table (the default) or direct");
  if (form == "direct" && !T.all())
    throw std::runtime_error("B70_Q4_EXPERTS=direct reads every expert from the layer's VRAM range, but " +
                             std::to_string(T.host_bytes(d)) + " B of experts are in host memory: unset it");
  // The plan before the first allocation: a model that does not fit is refused here, naming the bytes.
  {
    std::vector<runtime::qwen4exp::DevicePlan> pl = runtime::qwen4exp::plan(d, placement, max_len, int8, mtp);
    runtime::qwen4exp::apply_tier(pl, d, T);
    runtime::qwen4exp::require_fits(pl, caps, reserve);
  }
  const std::string ple_dir = q4_ple_dir(o.snap);
  check_ple_constants(d, ck.ple_constants());

  Q4LoadedModel m;
  m.desc = d;
  m.placement = placement;
  m.max_len = max_len;
  m.int8_head = int8;
  m.mtp = mtp;
  m.checkpoint_layers = full.layers;
  m.subnormal_scales = scan.subnormal_scales;
  if (d.layers < full.layers) m.skipped += ck.skip_layers_from(d.layers);
  const size_t visual = ck.skip_prefix("model.visual."), shards = ck.skip_ple_shards();
  const size_t mtp_skipped = mtp ? 0 : ck.skip_prefix("mtp.");
  // Everything this load pins - the PLE table and the experts' mirrors - against host memory before the first pin.
  // The prefix cache is 0 here: settle counted an explicit --prefix-cache-gb, and auto sizes itself after the load.
  check_q4_host_fit(Q4PleHost(ple_dir).bytes(), T.host_bytes(d), 0, q4_mem_available());
  // The PLE table, once, through the context of the device holding the PLE layer.
  m.ple_device = placement.device_of(d.ple_layer);
  m.ple = load_q4_ple(*devices[m.ple_device], d, ple_dir);
  m.tier = T;
  m.expert_table = form == "table";

  const std::vector<float> rope = q4_rope_table(d, max_len);
  Q4HostLayer host;
  m.parts.resize(placement.devices);
  for (uint32_t dev = 0; dev < placement.devices; ++dev) {
    l0::Context& ctx = *devices[dev];
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    Q4DevicePart& P = m.parts[dev];
    P.device = dev;
    P.first = placement.first(dev);
    P.end = placement.end(dev);
    if (dev == 0) {
      P.embed = upload_u(ctx, imm, ck.embed(), q4_embed_bytes(d));
      P.bytes += q4_embed_bytes(d);
    }
    if (q4_device_has_qsa(d, placement, dev, mtp)) {
      P.rope = upload_u(ctx, imm, rope.data(), rope.size() * 4);
      P.rope_bytes = rope.size() * 4;
      if (P.rope_bytes != d.rope_table_bytes(max_len))
        throw std::logic_error("load_qwen4exp: the RoPE table is not Qwen4ExpDesc::rope_table_bytes");
    }
    std::vector<uint64_t> rows(m.expert_table ? size_t(P.end - P.first) * 2 * d.experts : 0);
    for (uint32_t l = P.first; l < P.end; ++l) {
      ck.repack_layer(l, host);
      uint64_t* row = rows.empty() ? nullptr : rows.data() + size_t(l - P.first) * 2 * d.experts;
      if (T.n_vram(l) == d.experts && !row)
        P.layers.push_back(upload_layer(ctx, imm, host, P.bytes));
      else
        P.layers.push_back(upload_layer(ctx, imm, host, P.bytes, T.vram.data() + size_t(l) * d.experts, d.experts,
                                        &m.mirror_bytes, row));
      if (!row) continue;
      // The load check's samples of this layer's 1024 blocks, from the checkpoint's bytes (Q4DevicePart::check_*).
      const auto s0 = std::chrono::steady_clock::now();
      const uint32_t base = uint32_t(row - rows.data());
      for (uint32_t i = 0; i < 2 * d.experts; ++i) {
        const bool up = i < d.experts;
        const size_t B = (up ? host.gate_up.size() : host.down.size()) * 4 / d.experts;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(up ? host.gate_up.data() : host.down.data()) +
                             size_t(up ? i : i - d.experts) * B;
        const auto take = [&](size_t o) {
          uint64_t w = 0;
          std::memcpy(&w, src + o, 8);
          P.check_idx.push_back(base + i);
          P.check_off.push_back(uint32_t(o));
          P.check_words.push_back(w);
        };
        for (size_t o = 0; o + 8 < B; o += size_t(1) << 20) take(o);
        take(B - 8);
      }
      m.expert_check_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
    }
    if (!rows.empty()) {
      // The table, then read back: the rows the kernels will read are the rows the map says.
      P.expert_table = upload_u(ctx, imm, rows.data(), rows.size() * 8);
      std::vector<uint64_t> back(rows.size());
      imm.copy(back.data(), P.expert_table->ptr(), back.size() * 8);   // an immediate list: complete on return
      if (back != rows)
        throw std::runtime_error("load_qwen4exp: device " + std::to_string(dev) + "'s expert table reads back different "
                                 "addresses than were written");
      for (uint32_t l = P.first; l < P.end; ++l)
        P.layers[l - P.first].expert_row = P.expert_table->as<uint8_t>() + size_t(l - P.first) * 2 * d.experts * 8;
    }
    if (dev + 1 == placement.devices) {
      const std::vector<uint8_t> fm = ck.final_mixer();
      P.final_mixer = upload_u(ctx, imm, fm.data(), fm.size());
      P.bytes += fm.size();
      // lm_head: spec 9's int8 rows + fp32 scales for gemv_i8w, or bf16 tiles for gemv_bf16.
      P.lm_head = upload_lm_head(ctx, imm, ck.lm_head(), d.hidden, d.vocab, int8, m.lm_head_quant_seconds, P.bytes);
      if (mtp) {
        ck.repack_mtp(host);
        P.mtp = std::make_unique<Q4Layer>(upload_layer(ctx, imm, host, P.bytes));
        const std::vector<uint8_t> fc = ck.mtp_fc(), mm = ck.mtp_mixer();
        P.mtp_fc = upload_u(ctx, imm, fc.data(), fc.size());
        P.mtp_mixer = upload_u(ctx, imm, mm.data(), mm.size());
        P.bytes += fc.size() + mm.size();
      }
    }
    size_t host_experts = 0;
    for (uint32_t l = P.first; l < P.end; ++l)
      host_experts += size_t(d.experts - T.n_vram(l)) * (q4_gate_up_block_bytes(d) + q4_down_block_bytes(d));
    const size_t want = q4_device_weight_bytes(d, placement, dev, int8, mtp) - host_experts;
    if (P.bytes != want)
      throw std::logic_error("load_qwen4exp: device " + std::to_string(dev) + " allocated " + std::to_string(P.bytes) +
                             " B of weights, loader::q4_device_weight_bytes less the experts in host memory says " +
                             std::to_string(want));
  }

  std::string names;
  m.unconsumed = ck.unconsumed(&names);
  if (m.unconsumed != 0)
    throw std::runtime_error("load_qwen4exp: " + std::to_string(m.unconsumed) + " checkpoint tensors were not loaded: " +
                             names);
  for (uint32_t l = 0; l < d.layers; ++l) m.read_per_token += q4_layer_read_per_token(d, l);
  m.read_per_token += q4_final_mixer_bytes(d) + q4_lm_head_bytes(d, int8);
  m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double gb = 1e9;
  std::print("load_qwen4exp: {}\n  model       {} ({}): {:d} of {:d} layers ({:d} QSA), dense {}, shared {}, routed experts "
             "int4 g{:d}{}, {:d} subnormal f16 scales; {:d} tensors, every name accounted for, 0 unconsumed\n",
             o.snap.c_str(), d.name.c_str(), d.architecture.c_str(), d.layers, full.layers, d.qsa_before(d.layers),
             model::q4_form_name(d.forms.dense), model::q4_form_name(d.forms.shared), d.forms.expert_group,
             d.forms.expert_group == 128 ? " (expanded exactly to g64)" : "", scan.subnormal_scales,
             set.tensors().size());
  std::print("  skipped     by design: {:d} visual tensors, {:d} bf16 PLE shards (the int8 file replaces them){}\n", visual,
             shards, mtp ? "" : (", " + std::to_string(mtp_skipped) + " mtp.* (the head not loaded)").c_str());
  if (m.skipped)
    std::print("  note        development mode (--layers {:d}): layers {:d}..{:d} not loaded ({:d} tensors consumed by "
               "design); the final mixer reads layer {:d}'s streams\n",
               d.layers, d.layers, full.layers - 1, m.skipped, d.layers - 1);
  for (const Q4DevicePart& P : m.parts)
    std::print("  device {:d}    layers [{:d}, {:d}): weights {:13d} B {:7.3f} GB{}{}{}, RoPE {:d} B (max_len {:d})\n", P.device, P.first,
               P.end, P.bytes, P.bytes / gb, P.embed ? " + embed" : "",
               P.lm_head ? (int8 ? " + int8 head" : " + bf16 head") : "", P.mtp ? " + MTP head (RTN int4 g64 experts)" : "",
               P.rope_bytes, max_len);
  std::print("  host PLE    {:d} + {:d} host-USM ranges via device {:d}: {:13d} B {:7.3f} GB ({} scales), pinned in {:.1f} s; {:d} "
             "page tags read back, {:d} rows compared; MemAvailable {:.1f} GB before, {:.1f} GB after\n",
             d.ple_heads, d.ple_heads, m.ple_device, m.ple.bytes, m.ple.bytes / gb, q4_ple_scale_name(m.ple.scale),
             m.ple.seconds, m.ple.page_words.size(), m.ple.rows_checked, m.ple.mem_before / gb, m.ple.mem_after / gb);
  {
    uint32_t nv = 0;
    size_t samples = 0;
    for (uint32_t l = 0; l < d.layers; ++l) nv += T.n_vram(l);
    for (const Q4DevicePart& P : m.parts) samples += P.check_words.size();
    std::print("  experts     {:d} of {:d} routed experts in VRAM (f {:.3f}; {}), {:d} B {:.3f} GB in pinned host memory "
               "(one range a layer); the kernels read them through {}\n",
               nv, d.layers * d.experts, double(nv) / (d.layers * d.experts), T.source.c_str(), m.mirror_bytes,
               m.mirror_bytes / gb,
               m.expert_table ? std::format("each layer's table of block addresses (read back equal; {:d} block words "
                                            "sampled for the device's check)", samples).c_str()
                              : "the layer's range (direct)");
  }
  std::print("  read/token  {:13d} B {:7.3f} GB   (top-{:d} + shared experts, HC, mixers, routers, head; derived)\n"
             "  load        {:.1f} s\n",
             m.read_per_token, m.read_per_token / gb, d.top_k, m.seconds);
  return m;
}

}  // namespace loader

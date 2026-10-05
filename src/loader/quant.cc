#include "loader/quant.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace loader {

void check_align(const void* p, size_t a, const std::string& name) {
  if (reinterpret_cast<uintptr_t>(p) % a != 0)
    throw std::runtime_error("tensor '" + name + "': mmapped data is not " + std::to_string(a) +
                             "-byte aligned - cannot reinterpret_cast");
}

namespace {

// 4 hex digits, so a bad f16 is reported as the bit pattern it actually is.
std::string hex16(uint16_t v) {
  static const char* d = "0123456789ABCDEF";
  return {d[(v >> 12) & 0xF], d[(v >> 8) & 0xF], d[(v >> 4) & 0xF], d[v & 0xF]};
}

// The two-dimensional tensors this file indexes by [0]/[1]. safetensors lets a
// header declare any rank, so the rank is checked before the index rather than
// after the crash.
void check_rank2(const TensorInfo& t, const std::string& name) {
  if (t.shape.size() != 2)
    throw std::runtime_error(name + ": rank " + std::to_string(t.shape.size()) +
                             ", expected 2 (a [K/8][N] or [N][K] matrix)");
}

}  // namespace

QuantConfig QuantConfig::parse(const common::json::Value& config_json) {
  const common::json::Value* qcv = config_json.find("quantization_config");
  if (!qcv) throw std::runtime_error("config.json has no quantization_config");
  QuantConfig q;
  q.bits = uint32_t(qcv->at("bits").num());
  q.group_size = uint32_t(qcv->at("group_size").num());
  q.sym = qcv->at("sym").boolean();
  // `desc_act` is GPTQ's key. auto-round's `auto_round:auto_gptq` writer does
  // not emit it - it never permutes - so absence means false. Recorded as
  // inferred, and `loader::load` proves it from the shipped `g_idx` count
  // rather than taking the silence on trust.
  if (const common::json::Value* da = qcv->find("desc_act"); da && !da->is_null()) {
    q.desc_act = da->boolean();
    q.desc_act_declared = true;
  } else {
    q.desc_act = false;
    q.desc_act_declared = false;
  }
  if (const common::json::Value* qm = qcv->find("quant_method")) q.quant_method = qm->str();
  if (const common::json::Value* pf = qcv->find("packing_format")) q.packing_format = pf->str();
  // The label decides nothing about a tensor (that is `classify`), but an
  // unrecognised one means an unrecognised *packing*, and a wrong nibble order
  // is a silently wrong model. Both observed spellings are named; anything else
  // stops here with the value it found.
  if (!q.quant_method.empty() && q.quant_method != "gptq" && q.quant_method != "auto-round")
    throw std::runtime_error("quantization_config.quant_method '" + q.quant_method +
                             "' is neither 'gptq' nor 'auto-round' - this loader implements the "
                             "GPTQ v1 packing those two share (docs/02, docs/13)");
  if (!q.packing_format.empty() && q.packing_format != "auto_round:auto_gptq")
    throw std::runtime_error("quantization_config.packing_format '" + q.packing_format +
                             "' is not 'auto_round:auto_gptq' - the qweight[K/8][N] nibble order "
                             "this loader repacks is that format's");
  if (const common::json::Value* dyn = qcv->find("dynamic")) {
    for (const auto& [rule, unused] : dyn->obj()) {
      (void)unused;
      if (rule.rfind("-:", 0) != 0)
        throw std::runtime_error(
            "quantization_config.dynamic has a non-exclusion rule '" + rule +
            "' - this is the broken-MTP-config pattern (BENCHMARKS.md); fix the checkpoint");
      ++q.dynamic_rule_count;
    }
  }
  // auto-round's `extra_config` replaces `dynamic`: a per-module object, not a
  // regex list, so it can be read exactly instead of approximately. `bits: 16`
  // is an exclusion; `bits: 4` is a module the quantiser claims it packed, and
  // for those the group size and symmetry MUST be one the loader implements -
  // g64, or g128 expanded to g64 at load - since an asymmetric module or any
  // other group would dequantise wrong with no other warning. The
  // `+:` hazard `dynamic` guards against has no analogue here: a claim that a
  // module is packed is checked against the shipped tensors by
  // `LinearSrc::classify` at every load site.
  if (const common::json::Value* ec = qcv->find("extra_config")) {
    for (const auto& [module, rule] : ec->obj()) {
      if (!rule.is_object())
        throw std::runtime_error("quantization_config.extra_config['" + module +
                                 "'] is not an object - this loader reads auto-round's per-module "
                                 "form {bits, group_size, sym, ...}");
      const common::json::Value* b = rule.find("bits");
      const uint32_t mb = b ? uint32_t(b->num()) : q.bits;
      if (mb == 16) {
        ++q.extra_excluded;
        continue;
      }
      if (mb != 4)
        throw std::runtime_error("quantization_config.extra_config['" + module + "'] has bits=" +
                                 std::to_string(mb) + " - this loader implements 4 (packed) and "
                                 "16 (left in fp) only");
      const common::json::Value* gs = rule.find("group_size");
      const common::json::Value* sy = rule.find("sym");
      const uint32_t mg = gs ? uint32_t(gs->num()) : q.group_size;
      const bool ms = sy ? sy->boolean() : q.sym;
      if ((mg != 64 && mg != 128) || !ms)
        throw std::runtime_error("quantization_config.extra_config['" + module +
                                 "'] is int4 g" + std::to_string(mg) + " sym=" +
                                 (ms ? "true" : "false") +
                                 ", but the loader implements g64 and g128 symmetric only");
      ++q.extra_quantised;
    }
  }
  if (q.bits != 4 || (q.group_size != 64 && q.group_size != 128) || !q.sym || q.desc_act)
    throw std::runtime_error("unsupported quantization: need int4 g64 or g128 sym desc_act=false, got bits=" +
                             std::to_string(q.bits) + " g=" + std::to_string(q.group_size) +
                             " sym=" + (q.sym ? "true" : "false") +
                             " desc_act=" + (q.desc_act ? "true" : "false"));
  return q;
}

LinearSrc LinearSrc::classify(const SafetensorsSet& set, const std::string& prefix) {
  const auto& ts = set.tensors();
  auto qw = ts.find(prefix + ".qweight");
  if (qw != ts.end()) {
    auto sc = ts.find(prefix + ".scales");
    if (sc == ts.end()) throw std::runtime_error(prefix + ": qweight without scales");
    if (qw->second.dtype != "I32") throw std::runtime_error(prefix + ".qweight dtype " + qw->second.dtype);
    if (sc->second.dtype != "F16") throw std::runtime_error(prefix + ".scales dtype " + sc->second.dtype);
    check_rank2(qw->second, prefix + ".qweight");
    check_rank2(sc->second, prefix + ".scales");
    LinearSrc s;
    s.kind = WKind::Int4;
    s.K = uint32_t(qw->second.shape[0]) * 8;
    s.N = uint32_t(qw->second.shape[1]);
    // The scales' row count names the group: K/64 rows is g64, K/128 rows is
    // g128 (the two never coincide for K > 0). Anything else is refused.
    const uint64_t rows = sc->second.shape[0];
    if (s.K % 64 != 0 || sc->second.shape[1] != s.N ||
        (rows != s.K / 64 && !(s.K % 128 == 0 && rows == s.K / 128)))
      throw std::runtime_error(prefix + ".scales shape [" + std::to_string(rows) + "][" +
                               std::to_string(sc->second.shape[1]) + "] matches neither g64 [" +
                               std::to_string(s.K / 64) + "] nor g128 [" +
                               std::to_string(s.K / 128) + "] rows of N=" + std::to_string(s.N));
    check_align(set.data(qw->second), alignof(uint32_t), prefix + ".qweight");
    check_align(set.data(sc->second), alignof(uint16_t), prefix + ".scales");
    s.qweight = reinterpret_cast<const uint32_t*>(set.data(qw->second));
    s.scales = reinterpret_cast<const uint16_t*>(set.data(sc->second));
    if (rows != s.K / 64) {
      // g128 onto the g64 kernels: g64 rows 2j and 2j+1 are g128 row j. Every
      // weight keeps its own scale, so the dequant is exact; see quant.h.
      auto e = std::make_shared<std::vector<uint16_t>>(size_t(s.K / 64) * s.N);
      for (uint32_t j = 0; j < s.K / 128; ++j) {
        const uint16_t* src = s.scales + size_t(j) * s.N;
        std::copy_n(src, s.N, e->data() + size_t(2 * j) * s.N);
        std::copy_n(src, s.N, e->data() + size_t(2 * j + 1) * s.N);
      }
      s.group = 128;
      s.scales = e->data();
      s.expanded_scales = std::move(e);
    }
    s.name = prefix;
    return s;
  }
  auto w = ts.find(prefix + ".weight");
  if (w != ts.end()) {
    if (w->second.dtype != "BF16") throw std::runtime_error(prefix + ".weight dtype " + w->second.dtype);
    check_rank2(w->second, prefix + ".weight");
    LinearSrc s;
    s.kind = WKind::Bf16;
    s.N = uint32_t(w->second.shape[0]);
    s.K = uint32_t(w->second.shape[1]);
    check_align(set.data(w->second), alignof(uint16_t), prefix + ".weight");
    s.weight = reinterpret_cast<const uint16_t*>(set.data(w->second));
    s.name = prefix;
    return s;
  }
  throw std::runtime_error("no qweight or weight for linear '" + prefix + "'");
}

QuantScan assert_quant_invariants(const SafetensorsSet& set) {
  QuantScan scan;
  for (const auto& [name, t] : set.tensors()) {
    if (name.size() > 7 && name.compare(name.size() - 7, 7, ".qzeros") == 0) {
      check_align(set.data(t), alignof(uint32_t), name);
      const uint32_t* p = reinterpret_cast<const uint32_t*>(set.data(t));
      size_t n = set.bytes(t) / 4;
      for (size_t i = 0; i < n; ++i)
        if (p[i] != 0x77777777u)
          throw std::runtime_error(name + "[" + std::to_string(i) + "] = " +
                                   std::to_string(p[i]) + ", expected 0x77777777 (sym zero-point 8)");
    } else if (name.size() > 6 && name.compare(name.size() - 6, 6, ".g_idx") == 0) {
      ++scan.g_idx_tensors;
      check_align(set.data(t), alignof(int32_t), name);
      const int32_t* p = reinterpret_cast<const int32_t*>(set.data(t));
      size_t n = set.bytes(t) / 4;
      // The group comes from the module's own .scales rows (K = n entries
      // here), the same rule `classify` dequantises by, so a g_idx that agrees
      // with a different group than its scales is caught.
      size_t group = 64;
      const auto sc = set.tensors().find(name.substr(0, name.size() - 6) + ".scales");
      if (sc != set.tensors().end() && !sc->second.shape.empty() && sc->second.shape[0] != 0 &&
          n % sc->second.shape[0] == 0)
        group = n / sc->second.shape[0];
      for (size_t i = 0; i < n; ++i)
        if (p[i] != int32_t(i / group))
          throw std::runtime_error(name + "[" + std::to_string(i) + "] = " +
                                   std::to_string(p[i]) + ", expected identity k/" +
                                   std::to_string(group));
    } else if (name.size() > 7 && name.compare(name.size() - 7, 7, ".scales") == 0) {
      // The dequant is scale * (q - 8) with no guard, so a NaN/Inf scale
      // poisons a whole group of 64 weights and there is nowhere downstream
      // that would notice. 0.76 GB of f16, ~0.3 s - cheaper than finding it in
      // a logit. Subnormals are counted, not rejected: they occur in this
      // checkpoint (0x00A8 in layers.10.linear_attn.in_proj_qkv.scales, and
      // more) and they are perfectly meaningful f16 values.
      check_align(set.data(t), alignof(uint16_t), name);
      const uint16_t* p = reinterpret_cast<const uint16_t*>(set.data(t));
      size_t n = set.bytes(t) / 2;
      for (size_t i = 0; i < n; ++i) {
        const uint32_t exp = (p[i] >> 10) & 0x1Fu, man = p[i] & 0x3FFu;
        if (exp == 0x1Fu)
          throw std::runtime_error(name + "[" + std::to_string(i) + "] = 0x" +
                                   hex16(p[i]) + ", an f16 " + (man ? "NaN" : "Inf") +
                                   " - the dequant has no guard for it");
        if (exp == 0 && man != 0) ++scan.subnormal_scales;
      }
    }
  }
  return scan;
}

}  // namespace loader

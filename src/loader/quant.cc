#include "loader/quant.h"
#include <stdexcept>

namespace loader {

QuantConfig QuantConfig::parse(const common::json::Value& config_json) {
  const common::json::Value* qcv = config_json.find("quantization_config");
  if (!qcv) throw std::runtime_error("config.json has no quantization_config");
  QuantConfig q;
  q.bits = uint32_t(qcv->at("bits").num());
  q.group_size = uint32_t(qcv->at("group_size").num());
  q.sym = qcv->at("sym").boolean();
  q.desc_act = qcv->at("desc_act").boolean();
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
  if (q.bits != 4 || q.group_size != 64 || !q.sym || q.desc_act)
    throw std::runtime_error("unsupported quantization: need int4 g64 sym desc_act=false, got bits=" +
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
    LinearSrc s;
    s.kind = WKind::Int4;
    s.K = uint32_t(qw->second.shape[0]) * 8;
    s.N = uint32_t(qw->second.shape[1]);
    if (sc->second.shape[0] != s.K / 64 || sc->second.shape[1] != s.N)
      throw std::runtime_error(prefix + ".scales shape mismatch");
    s.qweight = reinterpret_cast<const uint32_t*>(set.data(qw->second));
    s.scales = reinterpret_cast<const uint16_t*>(set.data(sc->second));
    s.name = prefix;
    return s;
  }
  auto w = ts.find(prefix + ".weight");
  if (w != ts.end()) {
    if (w->second.dtype != "BF16") throw std::runtime_error(prefix + ".weight dtype " + w->second.dtype);
    LinearSrc s;
    s.kind = WKind::Bf16;
    s.N = uint32_t(w->second.shape[0]);
    s.K = uint32_t(w->second.shape[1]);
    s.weight = reinterpret_cast<const uint16_t*>(set.data(w->second));
    s.name = prefix;
    return s;
  }
  throw std::runtime_error("no qweight or weight for linear '" + prefix + "'");
}

void assert_quant_invariants(const SafetensorsSet& set) {
  for (const auto& [name, t] : set.tensors()) {
    if (name.size() > 7 && name.compare(name.size() - 7, 7, ".qzeros") == 0) {
      const uint32_t* p = reinterpret_cast<const uint32_t*>(set.data(t));
      size_t n = set.bytes(t) / 4;
      for (size_t i = 0; i < n; ++i)
        if (p[i] != 0x77777777u)
          throw std::runtime_error(name + "[" + std::to_string(i) + "] = " +
                                   std::to_string(p[i]) + ", expected 0x77777777 (sym zero-point 8)");
    } else if (name.size() > 6 && name.compare(name.size() - 6, 6, ".g_idx") == 0) {
      const int32_t* p = reinterpret_cast<const int32_t*>(set.data(t));
      size_t n = set.bytes(t) / 4;
      for (size_t i = 0; i < n; ++i)
        if (p[i] != int32_t(i / 64))
          throw std::runtime_error(name + "[" + std::to_string(i) + "] = " +
                                   std::to_string(p[i]) + ", expected identity k/64");
    }
  }
}

}  // namespace loader

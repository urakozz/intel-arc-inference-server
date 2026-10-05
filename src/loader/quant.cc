#include "loader/quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/bf16.h"
#include "common/kv8.h"

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

bool ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() > n && s.compare(s.size() - n, n, suf) == 0;
}

// A JSON value as the message names it: strings quoted, the rest as written.
std::string show(const common::json::Value& v) {
  if (v.is_null()) return "null";
  if (v.is_bool()) return v.boolean() ? "true" : "false";
  if (v.is_string()) return "\"" + v.str() + "\"";
  if (v.is_number()) {
    const double d = v.num();
    return d == double(int64_t(d)) ? std::to_string(int64_t(d)) : std::to_string(d);
  }
  return v.is_array() ? "[...]" : "{...}";
}

[[noreturn]] void ct_refuse(const std::string& field, const std::string& value,
                            const std::string& why) {
  throw std::runtime_error("quantization_config." + field + " = " + value + ": " + why +
                           " - this loader reads compressed-tensors pack-quantized int4 "
                           "symmetric g64 / g128 only (spec 20 §9, docs/13)");
}

// An object, `{}` or null: "nothing configured". Anything with content is.
bool ct_empty(const common::json::Value* v) {
  return v == nullptr || v->is_null() || (v->is_object() && v->obj().empty());
}

// llm-compressor's compressed-tensors config (QuantConfig::parse's comment). Every
// field that changes the arithmetic is checked against the one scheme the GPTQ-layout
// kernels compute after the repack; nothing is defaulted silently.
QuantConfig parse_compressed_tensors(const common::json::Value& qc) {
  QuantConfig q;
  q.quant_method = "compressed-tensors";
  q.compressed_tensors = true;
  const common::json::Value* fmt = qc.find("format");
  if (!fmt || !fmt->is_string())
    ct_refuse("format", fmt ? show(*fmt) : "absent",
              "the packing is undeclared (pack-quantized is the one implemented)");
  if (fmt->str() != "pack-quantized")
    ct_refuse("format", show(*fmt), "not pack-quantized - another tensor layout");
  q.packing_format = fmt->str();
  if (const common::json::Value* st = qc.find("quantization_status");
      st && !st->is_null() && (!st->is_string() || st->str() != "compressed"))
    ct_refuse("quantization_status", show(*st),
              "the weights are not stored compressed (weight_packed)");
  if (const common::json::Value* sp = qc.find("sparsity_config"); !ct_empty(sp))
    ct_refuse("sparsity_config", "{...}", "a sparse-compressed checkpoint");
  if (const common::json::Value* tr = qc.find("transform_config"); !ct_empty(tr))
    ct_refuse("transform_config", "{...}",
              "transforms (rotations) put the weights in another basis the kernels do not undo");
  if (const common::json::Value* kv = qc.find("kv_cache_scheme"); kv && !kv->is_null())
    q.ct_kv_cache_scheme = true;
  if (const common::json::Value* ig = qc.find("ignore"); ig && ig->is_array())
    q.ct_ignore = ig->arr().size();
  const common::json::Value* groups = qc.find("config_groups");
  if (!groups || !groups->is_object() || groups->obj().empty())
    ct_refuse("config_groups", groups ? show(*groups) : "absent", "no quantisation scheme");
  q.bits = 4;
  q.sym = true;
  q.desc_act = false;
  q.desc_act_declared = true;   // compressed-tensors' own vocabulary: `actorder`
  for (const auto& [gname, g] : groups->obj()) {
    const std::string at = "config_groups." + gname;
    if (!g.is_object()) ct_refuse(at, show(g), "not an object");
    if (const common::json::Value* f = g.find("format");
        f && !f->is_null() && (!f->is_string() || f->str() != "pack-quantized"))
      ct_refuse(at + ".format", show(*f), "not pack-quantized - another tensor layout");
    for (const char* act : {"input_activations", "output_activations"})
      if (const common::json::Value* a = g.find(act); a && !a->is_null())
        ct_refuse(at + "." + act, "{...}",
                  "quantised activations (W4A8 / W8A8) - this engine runs W4A16");
    const common::json::Value* w = g.find("weights");
    if (!w || !w->is_object())
      ct_refuse(at + ".weights", w ? show(*w) : "absent", "the group quantises no weights");
    const std::string wf = at + ".weights.";
    auto need = [&](const char* key) -> const common::json::Value& {
      const common::json::Value* v = w->find(key);
      if (!v || v->is_null()) ct_refuse(wf + key, "absent", "required");
      return *v;
    };
    const common::json::Value& nb = need("num_bits");
    if (!nb.is_number() || nb.num() != 4) ct_refuse(wf + "num_bits", show(nb), "not 4");
    const common::json::Value& ty = need("type");
    if (!ty.is_string() || ty.str() != "int") ct_refuse(wf + "type", show(ty), "not int");
    const common::json::Value& sy = need("symmetric");
    if (!sy.is_bool() || !sy.boolean())
      ct_refuse(wf + "symmetric", show(sy),
                "asymmetric (zero points): every int4 kernel would need a zero-point variant, "
                "and no exact conversion to symmetric int4 exists");
    const common::json::Value& sg = need("strategy");
    if (!sg.is_string() || sg.str() != "group")
      ct_refuse(wf + "strategy", show(sg), "not group");
    const common::json::Value& gs = need("group_size");
    if (!gs.is_number() || (gs.num() != 64 && gs.num() != 128))
      ct_refuse(wf + "group_size", show(gs), "not 64 or 128");
    if (const common::json::Value* dy = w->find("dynamic");
        dy && !dy->is_null() && !(dy->is_bool() && !dy->boolean()))
      ct_refuse(wf + "dynamic", show(*dy), "dynamic weight quantisation");
    if (const common::json::Value* bs = w->find("block_structure"); bs && !bs->is_null())
      ct_refuse(wf + "block_structure", show(*bs), "block quantisation");
    // actorder: "weight" (alias "static") quantises in activation order but assigns the
    // groups in the stored column order - nothing to undo at run time, no g_idx written.
    // "group" (alias "dynamic"; `true` in older configs) stores a per-column g_idx.
    std::string ao;
    bool ao_group = false;
    if (const common::json::Value* a = w->find("actorder"); a && !a->is_null()) {
      if (a->is_bool()) {
        ao = a->boolean() ? "true" : "";
        ao_group = a->boolean();
      } else if (a->is_string() && (a->str() == "weight" || a->str() == "static")) {
        ao = a->str();
      } else if (a->is_string() && (a->str() == "group" || a->str() == "dynamic")) {
        ao = a->str();
        ao_group = true;
      } else {
        ct_refuse(wf + "actorder", show(*a), "an unknown activation ordering");
      }
    }
    if (q.ct_config_groups == 0) {
      q.group_size = uint32_t(gs.num());
      q.ct_actorder = ao;
    }
    q.ct_actorder_group = q.ct_actorder_group || ao_group;
    ++q.ct_config_groups;
  }
  return q;
}

// bf16 -> f16 when the value survives exactly, else a throw naming where it came from.
// bf16 keeps 8 significant bits, f16 11, so a normal-range value always fits; what can
// fail is the range: above 65504, or below 2^-17 where f16's subnormals run out of bits.
uint16_t bf16_to_f16_exact(uint16_t b, const std::string& name, size_t i) {
  const float f = common::bf16_to_f32(b);
  if (!std::isfinite(f))
    throw std::runtime_error(name + "[" + std::to_string(i) + "] = 0x" + hex16(b) + ", a bf16 " +
                             (std::isnan(f) ? "NaN" : "Inf") + " - the dequant has no guard for it");
  const uint16_t h = common::kv8::f32_to_f16_rne(f);
  if (common::f16_to_f32(h) != f || std::signbit(common::f16_to_f32(h)) != std::signbit(f))
    throw std::runtime_error(name + "[" + std::to_string(i) + "] = 0x" + hex16(b) + " (bf16 " +
                             std::to_string(f) + ") is not exactly representable in f16, the "
                             "scale type the kernels read - refusing to round a scale");
  return h;
}

// One f16 word's NaN / Inf check (the dequant has no guard) and subnormal count.
void check_f16_scale(uint16_t v, const std::string& name, size_t i, size_t& subnormal) {
  const uint32_t exp = (v >> 10) & 0x1Fu, man = v & 0x3FFu;
  if (exp == 0x1Fu)
    throw std::runtime_error(name + "[" + std::to_string(i) + "] = 0x" + hex16(v) + ", an f16 " +
                             (man ? "NaN" : "Inf") + " - the dequant has no guard for it");
  if (exp == 0 && man != 0) ++subnormal;
}

// g_idx must be k / group for every k: anything else is an activation-order permutation.
void check_identity(const int32_t* p, size_t n, size_t group, const std::string& name) {
  for (size_t i = 0; i < n; ++i)
    if (p[i] != int32_t(i / group))
      throw std::runtime_error(name + "[" + std::to_string(i) + "] = " + std::to_string(p[i]) +
                               ", expected identity k/" + std::to_string(group));
}

// The g64 expansion of g128 scales [K/128][N]: g64 rows 2j and 2j+1 are g128 row j.
std::shared_ptr<std::vector<uint16_t>> expand_g128(const uint16_t* src, uint32_t K, uint32_t N) {
  auto e = std::make_shared<std::vector<uint16_t>>(size_t(K / 64) * N);
  for (uint32_t j = 0; j < K / 128; ++j) {
    const uint16_t* row = src + size_t(j) * N;
    std::copy_n(row, N, e->data() + size_t(2 * j) * N);
    std::copy_n(row, N, e->data() + size_t(2 * j + 1) * N);
  }
  return e;
}

// compressed-tensors pack-quantized, symmetric, group (QuantConfig::parse): the exact
// repack onto the GPTQ layout. See LinearSrc's comment for the two transposes.
LinearSrc classify_ct(const SafetensorsSet& set, const std::string& prefix,
                      const TensorInfo& wp) {
  const auto& ts = set.tensors();
  const std::string pn = prefix + ".weight_packed";
  if (ts.count(prefix + ".qweight"))
    throw std::runtime_error(prefix + ": ships both .qweight (GPTQ) and .weight_packed "
                             "(compressed-tensors) - one packing per linear");
  if (ts.count(prefix + ".weight_zero_point"))
    throw std::runtime_error(prefix + ".weight_zero_point: an asymmetric compressed-tensors "
                             "linear - refused (spec 20 §9: no zero-point kernels, no exact "
                             "conversion to symmetric int4)");
  auto sc = ts.find(prefix + ".weight_scale");
  auto sh = ts.find(prefix + ".weight_shape");
  if (sc == ts.end() || sh == ts.end())
    throw std::runtime_error(prefix + ": weight_packed without " +
                             (sc == ts.end() ? "weight_scale" : "weight_shape"));
  if (wp.dtype != "I32") throw std::runtime_error(pn + " dtype " + wp.dtype + ", expected I32");
  check_rank2(wp, pn);
  check_rank2(sc->second, prefix + ".weight_scale");
  const std::string& sdt = sc->second.dtype;
  if (sdt != "F16" && sdt != "BF16")
    throw std::runtime_error(prefix + ".weight_scale dtype " + sdt + ", expected F16 or BF16");
  // weight_shape: the unpacked [N][K], I64 (or I32) [2]. Read through memcpy - a 16-byte
  // tensor's offset is the writer's to choose.
  const TensorInfo& si = sh->second;
  if ((si.dtype != "I64" && si.dtype != "I32") || si.shape.size() != 1 || si.shape[0] != 2)
    throw std::runtime_error(prefix + ".weight_shape: " + si.dtype + " of rank " +
                             std::to_string(si.shape.size()) + ", expected I64 [2]");
  uint64_t dims[2];
  if (si.dtype == "I64") {
    int64_t v[2];
    std::memcpy(v, set.data(si), 16);
    dims[0] = uint64_t(v[0]);
    dims[1] = uint64_t(v[1]);
  } else {
    int32_t v[2];
    std::memcpy(v, set.data(si), 8);
    dims[0] = uint64_t(v[0]);
    dims[1] = uint64_t(v[1]);
  }
  const uint64_t N = wp.shape[0], K = wp.shape[1] * 8;
  if (dims[0] != N || dims[1] != K)
    throw std::runtime_error(prefix + ".weight_shape [" + std::to_string(dims[0]) + ", " +
                             std::to_string(dims[1]) + "] is not weight_packed's [" +
                             std::to_string(N) + "][" + std::to_string(K) + " / 8] - a padded "
                             "or mismatched pack");
  const uint64_t G = sc->second.shape[1];
  if (K % 64 != 0 || sc->second.shape[0] != N ||
      (G != K / 64 && !(K % 128 == 0 && G == K / 128)))
    throw std::runtime_error(prefix + ".weight_scale shape [" + std::to_string(sc->second.shape[0]) +
                             "][" + std::to_string(G) + "] matches neither g64 [" +
                             std::to_string(N) + "][" + std::to_string(K / 64) + "] nor g128 [" +
                             std::to_string(N) + "][" + std::to_string(K / 128) + "]");
  const uint32_t group = uint32_t(K / G);
  check_align(set.data(wp), alignof(uint32_t), pn);
  check_align(set.data(sc->second), alignof(uint16_t), prefix + ".weight_scale");
  LinearSrc s;
  s.kind = WKind::Int4;
  s.packing = Int4Packing::CompressedTensors;
  s.K = uint32_t(K);
  s.N = uint32_t(N);
  s.group = group;
  s.name = prefix;
  if (auto gi = ts.find(prefix + ".weight_g_idx"); gi != ts.end()) {
    if (gi->second.dtype != "I32" || gi->second.shape.size() != 1 || gi->second.shape[0] != K)
      throw std::runtime_error(prefix + ".weight_g_idx: " + gi->second.dtype + ", expected I32 [" +
                               std::to_string(K) + "]");
    check_align(set.data(gi->second), alignof(int32_t), prefix + ".weight_g_idx");
    check_identity(reinterpret_cast<const int32_t*>(set.data(gi->second)), K, group,
                   prefix + ".weight_g_idx");
    s.ct_g_idx = true;
  }
  // qweight[r][n] = weight_packed[n][r]: the int32 words verbatim, 64 x 64 blocks so
  // both sides stay in cache.
  const uint32_t* src = reinterpret_cast<const uint32_t*>(set.data(wp));
  const size_t R = K / 8;
  auto qw = std::make_shared<std::vector<uint32_t>>(R * N);
  constexpr size_t B = 64;
  for (size_t n0 = 0; n0 < N; n0 += B)
    for (size_t r0 = 0; r0 < R; r0 += B)
      for (size_t n = n0; n < std::min<size_t>(n0 + B, N); ++n)
        for (size_t r = r0; r < std::min<size_t>(r0 + B, R); ++r)
          (*qw)[r * N + n] = src[n * R + r];
  // scales[g][n] = weight_scale[n][g], as f16.
  const uint16_t* ssrc = reinterpret_cast<const uint16_t*>(set.data(sc->second));
  const bool bf16 = sdt == "BF16";
  std::vector<uint16_t> st(size_t(G) * N);
  size_t subnormal = 0;
  for (size_t n = 0; n < N; ++n)
    for (size_t g = 0; g < G; ++g) {
      const size_t i = n * G + g;
      const uint16_t h = bf16 ? bf16_to_f16_exact(ssrc[i], prefix + ".weight_scale", i) : ssrc[i];
      check_f16_scale(h, prefix + ".weight_scale", i, subnormal);
      st[g * N + n] = h;
    }
  std::shared_ptr<std::vector<uint16_t>> scales =
      group == 128 ? expand_g128(st.data(), s.K, s.N)
                   : std::make_shared<std::vector<uint16_t>>(std::move(st));
  s.qweight = qw->data();
  s.owned_qweight = std::move(qw);
  s.scales = scales->data();
  s.expanded_scales = std::move(scales);
  return s;
}

}  // namespace

QuantConfig QuantConfig::parse(const common::json::Value& config_json) {
  const common::json::Value* qcv = config_json.find("quantization_config");
  if (!qcv) throw std::runtime_error("config.json has no quantization_config");
  if (const common::json::Value* qm = qcv->find("quant_method");
      qm && qm->is_string() && qm->str() == "compressed-tensors")
    return parse_compressed_tensors(*qcv);
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
                             "' is neither 'gptq' nor 'auto-round' nor 'compressed-tensors' - "
                             "this loader implements the GPTQ v1 packing the first two share "
                             "and an exact repack of the third's symmetric pack-quantized form "
                             "(docs/02, docs/13)");
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
      auto e = expand_g128(s.scales, s.K, s.N);
      s.group = 128;
      s.scales = e->data();
      s.expanded_scales = std::move(e);
    }
    s.name = prefix;
    return s;
  }
  if (auto wp = ts.find(prefix + ".weight_packed"); wp != ts.end())
    return classify_ct(set, prefix, wp->second);
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
  throw std::runtime_error("no qweight, weight_packed or weight for linear '" + prefix + "'");
}

WKind LinearSrc::kind_of(const SafetensorsSet& set, const std::string& prefix) {
  const auto& ts = set.tensors();
  if (ts.count(prefix + ".qweight") || ts.count(prefix + ".weight_packed")) return WKind::Int4;
  if (ts.count(prefix + ".weight")) return WKind::Bf16;
  throw std::runtime_error("no qweight, weight_packed or weight for linear '" + prefix + "'");
}

std::vector<std::string> LinearSrc::suffixes() const {
  if (kind == WKind::Bf16) return {".weight"};
  if (packing == Int4Packing::Gptq) return {".qweight", ".scales"};
  std::vector<std::string> v{".weight_packed", ".weight_scale", ".weight_shape"};
  if (ct_g_idx) v.push_back(".weight_g_idx");
  return v;
}

const std::vector<std::string>& LinearSrc::marker_suffixes() {
  static const std::vector<std::string> m{".qweight", ".weight_packed", ".weight"};
  return m;
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
    } else if (ends_with(name, ".weight_zero_point")) {
      throw std::runtime_error(name + ": an asymmetric compressed-tensors linear - refused "
                               "(spec 20 §9: no zero-point kernels, no exact conversion to "
                               "symmetric int4)");
    } else if (ends_with(name, ".weight_packed")) {
      ++scan.ct_linears;
    } else if (ends_with(name, ".qweight")) {
      ++scan.gptq_linears;
    } else if (ends_with(name, ".weight_g_idx")) {
      ++scan.ct_g_idx_tensors;
      // The group from the module's weight_scale columns (K / G), as classify reads it.
      const std::string base = name.substr(0, name.size() - 13);
      const auto sc = set.tensors().find(base + ".weight_scale");
      const size_t n = set.bytes(t) / 4;
      if (t.dtype != "I32" || sc == set.tensors().end() || sc->second.shape.size() != 2 ||
          sc->second.shape[1] == 0 || n % sc->second.shape[1] != 0)
        throw std::runtime_error(name + ": " + t.dtype + " beside no rank-2 " + base +
                                 ".weight_scale that divides its " + std::to_string(n) +
                                 " entries - cannot tell its group");
      check_align(set.data(t), alignof(int32_t), name);
      check_identity(reinterpret_cast<const int32_t*>(set.data(t)), n, n / sc->second.shape[1],
                     name);
    } else if (ends_with(name, ".weight_scale")) {
      // compressed-tensors scales [N][K/g]: the group from the sibling weight_packed
      // (K = its columns x 8), each value finite and, bf16, exactly an f16.
      const std::string base = name.substr(0, name.size() - 13);
      const auto wp = set.tensors().find(base + ".weight_packed");
      if (wp == set.tensors().end() || wp->second.shape.size() != 2 || t.shape.size() != 2 ||
          t.shape[1] == 0)
        throw std::runtime_error(name + ": no rank-2 " + base + ".weight_packed beside it - "
                                 "not a pack-quantized linear (an fp8 or unpacked checkpoint?)");
      const uint64_t K = wp->second.shape[1] * 8;
      if (K % t.shape[1] != 0 || (K / t.shape[1] != 64 && K / t.shape[1] != 128))
        throw std::runtime_error(name + ": [" + std::to_string(t.shape[0]) + "][" +
                                 std::to_string(t.shape[1]) + "] for K=" + std::to_string(K) +
                                 " is neither g64 nor g128");
      scan.ct_group_mask |= K / t.shape[1] == 64 ? 1u : 2u;
      if (t.dtype != "F16" && t.dtype != "BF16")
        throw std::runtime_error(name + " dtype " + t.dtype + ", expected F16 or BF16");
      check_align(set.data(t), alignof(uint16_t), name);
      const uint16_t* p = reinterpret_cast<const uint16_t*>(set.data(t));
      const size_t n = set.bytes(t) / 2;
      const bool bf16 = t.dtype == "BF16";
      scan.ct_bf16_scales = scan.ct_bf16_scales || bf16;
      for (size_t i = 0; i < n; ++i)
        check_f16_scale(bf16 ? bf16_to_f16_exact(p[i], name, i) : p[i], name, i,
                        scan.subnormal_scales);
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

void check_quant_scan(const QuantConfig& qc, const QuantScan& scan) {
  if (qc.compressed_tensors && scan.gptq_linears != 0)
    throw std::runtime_error("config.json says compressed-tensors but the checkpoint ships " +
                             std::to_string(scan.gptq_linears) + " GPTQ .qweight tensors - the "
                             "label and the bytes disagree");
  if (!qc.compressed_tensors && scan.ct_linears != 0)
    throw std::runtime_error("config.json says quant_method '" + qc.quant_method +
                             "' but the checkpoint ships " + std::to_string(scan.ct_linears) +
                             " compressed-tensors .weight_packed tensors - the label and the "
                             "bytes disagree");
  if (qc.ct_actorder_group && scan.ct_g_idx_tensors != scan.ct_linears)
    throw std::runtime_error(
        "quantization_config actorder \"" + qc.ct_actorder + "\" stores a per-column group "
        "map, but only " + std::to_string(scan.ct_g_idx_tensors) + " of " +
        std::to_string(scan.ct_linears) + " linears ship a weight_g_idx - an activation-order "
        "permutation the checkpoint did not write down is refused (spec 20 §9)");
}

std::string ct_conversion_note(const QuantScan& scan) {
  if (scan.ct_linears == 0) return {};
  const bool g64 = scan.ct_group_mask & 1u, g128 = scan.ct_group_mask & 2u;
  const std::string groups = g64 && g128 ? "g64 and g128" : g128 ? "g128" : "g64";
  const std::string expand = g64 && g128 ? "g64 kept, g128 -> g64 scales expanded"
                             : g128      ? "g128 -> g64 scales expanded"
                                         : "g64, no expansion";
  return "converting compressed-tensors pack-quantized (" + groups + ", " +
         std::to_string(scan.ct_linears) + " linears): weight_packed [N][K/8] -> our qweight "
         "[K/8][N], weight_scale [N][K/g]" + (scan.ct_bf16_scales ? " bf16" : "") +
         " -> scales [K/g][N]" + (scan.ct_bf16_scales ? " f16 (exact)" : "") + " (" + expand +
         "). The model is supported, but expect reduced quality compared to our recommended "
         "format: AutoRound GPTQ W4A16 g64 symmetric.";
}

}  // namespace loader

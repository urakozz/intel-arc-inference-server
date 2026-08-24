#include "loader/loader.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/safetensors.h"

namespace loader {
namespace {

using model::Qwen35;

// ---------------------------------------------------------------------------
// The per-layer "small tensor" blocks.
//
// Each layer gets at most two device allocations that are not GEMV weights: a
// norms block (both layernorms, always) and a kind-specific block. Offsets are
// compile-time constants so the kernels can be compiled against them; the
// static_asserts below are what fails if this file is edited without the
// kernels being told.
//
// norms  [0]      input_layernorm  (1 + w) bf16 [5120]
//        [10240]  post_attention_layernorm (1 + w) bf16 [5120]
//
// GDN block (docs/03-models.md "Layer math"):
//        [0]      conv1d      fp32 [10240][4]   (source is bf16 [10240][1][4])
//        [163840] -exp(A_log) fp32 [48]
//        [164032] dt_bias     fp32 [48]
//        [164224] linear_attn.norm.weight  plain w bf16 [128]  (RMSNormGated:
//                 no +1 - it is the one norm in the model without it)
//
// FA block:
//        [0]      q_norm (1 + w) bf16 [256]
//        [512]    k_norm (1 + w) bf16 [256]
// ---------------------------------------------------------------------------
constexpr size_t kConvRows = 10240, kConvTaps = 4;  // qkv width x conv kernel dim
constexpr size_t kNormsOffInput = 0;
constexpr size_t kNormsOffPost = kNormsOffInput + size_t(Qwen35::kHidden) * 2;
constexpr size_t kNormsBlockBytes = kNormsOffPost + size_t(Qwen35::kHidden) * 2;
static_assert(kNormsBlockBytes == 20480, "norms block layout changed");

constexpr size_t kGdnOffConv = 0;
constexpr size_t kGdnOffNegA = kGdnOffConv + kConvRows * kConvTaps * 4;
constexpr size_t kGdnOffDtBias = kGdnOffNegA + size_t(Qwen35::kGdnVHeads) * 4;
constexpr size_t kGdnOffGatedNorm = kGdnOffDtBias + size_t(Qwen35::kGdnVHeads) * 4;
constexpr size_t kGdnBlockBytes = kGdnOffGatedNorm + size_t(Qwen35::kGdnHeadDim) * 2;
static_assert(kGdnOffNegA % 4 == 0 && kGdnOffDtBias % 4 == 0, "fp32 fields must be 4-aligned");
static_assert(kGdnOffGatedNorm % 2 == 0, "bf16 field must be 2-aligned");
static_assert(kGdnBlockBytes == 164480, "GDN small block layout changed");

constexpr size_t kFaOffQNorm = 0;
constexpr size_t kFaOffKNorm = kFaOffQNorm + size_t(Qwen35::kFaHeadDim) * 2;
constexpr size_t kFaBlockBytes = kFaOffKNorm + size_t(Qwen35::kFaHeadDim) * 2;
static_assert(kFaBlockBytes == 1024, "FA small block layout changed");

// `W` from docs/03-models.md's byte accounting (measured 2026-08-24):
// 12.163 qweight + 0.760 scales + 0.052 bf16 smalls + 2.543 lm_head.
constexpr double kDocW = 15.519e9;

// ---------------------------------------------------------------------------
// Name mapping (requirement 1). The checkpoint carries three top-level
// namespaces; the model description knows only the language model's, with the
// prefix stripped, plus `lm_head` which is genuinely top level.
// ---------------------------------------------------------------------------
constexpr const char* kLmPrefix = "model.language_model.";

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// Rebuilds the checkpoint name of a stripped (model-description) name.
std::string ckpt_name(const std::string& stripped) {
  return starts_with(stripped, "lm_head") ? stripped : kLmPrefix + stripped;
}

struct NameView {
  std::map<std::string, std::string> names;  // stripped -> checkpoint name
  std::set<std::string> consumed;            // stripped names actually loaded
  size_t mtp_skipped = 0, visual_skipped = 0, qzeros = 0, g_idx = 0;
};

NameView build_view(const SafetensorsSet& set) {
  NameView v;
  for (const auto& [name, info] : set.tensors()) {
    (void)info;
    if (starts_with(name, "model.visual.")) {          // no vision tower in v1
      ++v.visual_skipped;
    } else if (starts_with(name, "mtp.")) {            // no speculation in v1
      ++v.mtp_skipped;
    } else if (starts_with(name, kLmPrefix)) {
      std::string stripped = name.substr(std::strlen(kLmPrefix));
      if (stripped.size() > 7 && stripped.compare(stripped.size() - 7, 7, ".qzeros") == 0) ++v.qzeros;
      if (stripped.size() > 6 && stripped.compare(stripped.size() - 6, 6, ".g_idx") == 0) ++v.g_idx;
      v.names.emplace(std::move(stripped), name);
    } else {
      v.names.emplace(name, name);                     // lm_head.weight
    }
  }
  return v;
}

// A mmapped tensor is reinterpret_cast to uint32_t/uint16_t; the safetensors
// data section is 8-aligned on every file seen so far, but a loud failure beats
// undefined behaviour if a future checkpoint pads differently.
void check_align(const void* p, size_t a, const std::string& name) {
  if (reinterpret_cast<uintptr_t>(p) % a != 0)
    throw std::runtime_error("tensor '" + name + "': mmapped data is not " + std::to_string(a) +
                             "-byte aligned - cannot reinterpret_cast");
}

const TensorInfo& take(NameView& v, const SafetensorsSet& set, const std::string& stripped) {
  auto it = v.names.find(stripped);
  if (it == v.names.end())
    throw std::runtime_error("checkpoint has no tensor '" + stripped + "' (looked for '" +
                             ckpt_name(stripped) + "')");
  v.consumed.insert(stripped);
  return set.tensors().at(it->second);
}

// A bf16 tensor of an exact element count - the doc-03 shape table, enforced by
// name for everything that is not a GEMV weight.
const uint16_t* bf16_of(NameView& v, const SafetensorsSet& set, const std::string& stripped,
                        size_t elems) {
  const TensorInfo& t = take(v, set, stripped);
  if (t.dtype != "BF16")
    throw std::runtime_error("tensor '" + stripped + "': dtype " + t.dtype + ", expected BF16");
  if (set.bytes(t) != elems * 2)
    throw std::runtime_error("tensor '" + stripped + "': " + std::to_string(set.bytes(t) / 2) +
                             " elements, expected " + std::to_string(elems));
  const uint8_t* p = set.data(t);
  check_align(p, alignof(uint16_t), stripped);
  return reinterpret_cast<const uint16_t*>(p);
}

// Gemma-style RMSNorm: the kernel wants a plain multiply, so the `+1` is baked
// in here. One fp32 add, one round-to-nearest-even cast back - the same single
// rounding discipline as every other conversion in this project.
void bake_one_plus(const uint16_t* w, uint16_t* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = common::f32_to_bf16(1.0f + common::bf16_to_f32(w[i]));
}

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem mem(ctx, l0::MemKind::Device, bytes);
  imm.copy(mem.ptr(), src, bytes);
  return mem;
}

const char* id_name(model::LinearId id) {
  switch (id) {
    case model::LinearId::QkvZ: return "QkvZ";
    case model::LinearId::AB: return "AB";
    case model::LinearId::OutProj: return "OutProj";
    case model::LinearId::GateUp: return "GateUp";
    case model::LinearId::Down: return "Down";
    case model::LinearId::Qkv: return "Qkv";
    case model::LinearId::OProj: return "OProj";
    case model::LinearId::LmHead: return "LmHead";
  }
  return "?";
}

// Host staging, allocated once and reused. The largest int4 linear is gate‖up
// (2176 n-tiles x 80 k-groups x 136 u32 = 94.7 MB); the largest bf16 tile
// buffer is lm_head (2.54 GB); the row-major bf16 concat buffer only ever holds
// a‖b padded to [128][5120].
struct Staging {
  std::vector<uint32_t> i4;
  std::vector<uint16_t> bf_src;
  std::vector<uint16_t> bf_tiled;
};

// One fused linear: classify every part, check the preconditions the repack
// helpers document but cannot check (Task-3 review minor, closed here at the
// first real consumer - these run in release too, load is one-shot), repack
// into staging and upload with a single copy.
DeviceWeight load_linear(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                         NameView& view, const std::string& layer_prefix,
                         const model::FusedLinear& fl, Staging& st, LoadReport& rep) {
  const model::GemvShape& sh = fl.shape;
  const std::string id = std::string(id_name(fl.id)) + " (" + layer_prefix + fl.parts[0] + ")";
  if (sh.K % 64 != 0 || sh.N % 16 != 0)
    throw std::runtime_error(id + ": shape K=" + std::to_string(sh.K) + " N=" +
                             std::to_string(sh.N) + " violates K%64==0, N%16==0");

  std::vector<LinearSrc> srcs;
  uint32_t n_sum = 0;
  for (const std::string& part : fl.parts) {
    LinearSrc s = LinearSrc::classify(set, ckpt_name(layer_prefix + part));
    const bool int4 = s.kind == WKind::Int4;
    if (int4 != (fl.kind == model::WeightKind::Int4))
      throw std::runtime_error(id + ": part '" + part + "' is " + (int4 ? "int4" : "bf16") +
                               " but the model description says " +
                               (fl.kind == model::WeightKind::Int4 ? "int4" : "bf16") +
                               " - the checkpoint's dynamic exclusions moved");
    if (int4) {
      check_align(s.qweight, alignof(uint32_t), s.name + ".qweight");
      check_align(s.scales, alignof(uint16_t), s.name + ".scales");
      view.consumed.insert(layer_prefix + part + ".qweight");
      view.consumed.insert(layer_prefix + part + ".scales");
    } else {
      check_align(s.weight, alignof(uint16_t), s.name + ".weight");
      view.consumed.insert(layer_prefix + part + ".weight");
    }
    if (s.K != sh.K)
      throw std::runtime_error(id + ": part '" + part + "' has K=" + std::to_string(s.K) +
                               ", the shape table says " + std::to_string(sh.K));
    n_sum += s.N;
    srcs.push_back(s);
  }
  // `pad_n` is the model description's *unpadded* N - the columns the checkpoint
  // actually ships; 0 means "no padding, the parts fill shape.N exactly".
  const uint32_t n_want = fl.pad_n != 0 ? fl.pad_n : sh.N;
  if (n_sum != n_want)
    throw std::runtime_error(id + ": parts sum to N=" + std::to_string(n_sum) + ", the model " +
                             "description says " + std::to_string(n_want) + " (zero-padded to " +
                             std::to_string(sh.N) + ")");
  if (n_sum > sh.N)
    throw std::runtime_error(id + ": unpadded N=" + std::to_string(n_sum) + " exceeds shape N=" +
                             std::to_string(sh.N));

  if (fl.kind == model::WeightKind::Int4) {
    if (n_sum != sh.N)
      throw std::runtime_error(id + ": int4 N-padding is not implemented (" +
                               std::to_string(n_sum) + " -> " + std::to_string(sh.N) + ")");
    std::vector<common::Part> parts;
    for (const LinearSrc& s : srcs) parts.push_back({s.qweight, s.scales, s.N});
    std::vector<common::ColSource> cols;
    if (fl.fuse == model::Fuse::Interleave16) {
      if (parts.size() != 2 || parts[0].N != parts[1].N || parts[0].N % 16 != 0)
        throw std::runtime_error(id + ": interleave16 needs two parts of equal, 16-divisible N");
      cols = common::cols_interleave16(parts[0], parts[1]);
    } else {
      cols = common::cols_concat(parts);
    }
    if (cols.size() != sh.N)
      throw std::runtime_error(id + ": column map has " + std::to_string(cols.size()) +
                               " entries, need " + std::to_string(sh.N));
    const size_t words = size_t(sh.N / 16) * (sh.K / 64) * 136;
    if (words > st.i4.size())
      throw std::runtime_error(id + ": " + std::to_string(words * 4) +
                               " bytes exceeds the int4 staging buffer");
    common::repack_int4_layout1_cols(sh.K, sh.N, cols, st.i4.data());
    // The 136-u32 tile is 128 u32 of nibbles + 8 u32 of scales, so the two
    // buckets below add up to exactly the bytes uploaded.
    rep.int4_bytes += size_t(sh.K) * sh.N / 2;
    rep.scale_bytes += size_t(sh.K) * sh.N / 32;
    return {upload(ctx, imm, st.i4.data(), words * 4), sh, fl.kind};
  }

  const size_t elems = size_t(sh.N) * sh.K;
  if (elems > st.bf_tiled.size())
    throw std::runtime_error(id + ": " + std::to_string(elems * 2) +
                             " bytes exceeds the bf16 staging buffer");
  if (fl.fuse == model::Fuse::Single && n_sum == sh.N) {
    // lm_head: no concat, no padding - tile straight out of the mmap.
    common::repack_bf16_tiled(srcs[0].weight, sh.K, sh.N, st.bf_tiled.data());
  } else {
    // a‖b: two [48][5120] parts stacked to [96][5120], zero rows to [128][5120].
    if (elems > st.bf_src.size())
      throw std::runtime_error(id + ": row-major concat exceeds the bf16 source buffer");
    std::memset(st.bf_src.data(), 0, elems * 2);
    size_t row = 0;
    for (const LinearSrc& s : srcs) {
      std::memcpy(st.bf_src.data() + row * sh.K, s.weight, size_t(s.N) * sh.K * 2);
      row += s.N;
    }
    common::repack_bf16_tiled(st.bf_src.data(), sh.K, sh.N, st.bf_tiled.data());
  }
  if (fl.id == model::LinearId::LmHead)
    rep.lm_head_bytes += size_t(n_sum) * sh.K * 2;
  else
    rep.bf16_linear_bytes += size_t(n_sum) * sh.K * 2;
  rep.pad_bytes += size_t(sh.N - n_sum) * sh.K * 2;
  return {upload(ctx, imm, st.bf_tiled.data(), elems * 2), sh, fl.kind};
}

// Everything in a layer that is not a GEMV weight, packed into the two blocks
// documented at the top of this file. `widen` accumulates the bytes this
// conversion adds over the checkpoint's own bf16 - the W cross-check itemises
// them rather than hiding them in a tolerance.
SmallTensors load_small(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                        NameView& view, const model::LayerDesc& ld, LoadReport& rep,
                        size_t& widen) {
  const std::string lp = Qwen35::layer_prefix(ld.index);
  std::vector<uint16_t> norms(kNormsBlockBytes / 2);
  bake_one_plus(bf16_of(view, set, lp + "input_layernorm.weight", Qwen35::kHidden),
                norms.data() + kNormsOffInput / 2, Qwen35::kHidden);
  bake_one_plus(bf16_of(view, set, lp + "post_attention_layernorm.weight", Qwen35::kHidden),
                norms.data() + kNormsOffPost / 2, Qwen35::kHidden);
  rep.small_bytes += kNormsBlockBytes;
  l0::Mem norms_mem = upload(ctx, imm, norms.data(), kNormsBlockBytes);

  if (ld.kind == model::LayerKind::FA) {
    std::vector<uint16_t> b(kFaBlockBytes / 2);
    bake_one_plus(bf16_of(view, set, lp + "self_attn.q_norm.weight", Qwen35::kFaHeadDim),
                  b.data() + kFaOffQNorm / 2, Qwen35::kFaHeadDim);
    bake_one_plus(bf16_of(view, set, lp + "self_attn.k_norm.weight", Qwen35::kFaHeadDim),
                  b.data() + kFaOffKNorm / 2, Qwen35::kFaHeadDim);
    rep.small_bytes += kFaBlockBytes;
    return {std::move(norms_mem), upload(ctx, imm, b.data(), kFaBlockBytes)};
  }

  std::vector<uint8_t> b(kGdnBlockBytes, 0);
  // conv1d.weight bf16 [10240][1][4] -> fp32 [10240][4]: the 4-tap depthwise
  // state is accumulated in fp32, so the taps are widened once here.
  {
    const uint16_t* w = bf16_of(view, set, lp + "linear_attn.conv1d.weight", kConvRows * kConvTaps);
    std::vector<float> f(kConvRows * kConvTaps);
    for (size_t i = 0; i < f.size(); ++i) f[i] = common::bf16_to_f32(w[i]);
    std::memcpy(b.data() + kGdnOffConv, f.data(), f.size() * 4);
    widen += f.size() * 4 - f.size() * 2;
  }
  // A_log -> -exp(A_log): the decay is only ever used as exp(g) with
  // g = -exp(A_log) * softplus(...), so the exp is hoisted to load time.
  {
    const uint16_t* w = bf16_of(view, set, lp + "linear_attn.A_log", Qwen35::kGdnVHeads);
    std::vector<float> f(Qwen35::kGdnVHeads);
    for (size_t i = 0; i < f.size(); ++i) f[i] = -std::exp(common::bf16_to_f32(w[i]));
    std::memcpy(b.data() + kGdnOffNegA, f.data(), f.size() * 4);
    widen += f.size() * 4 - f.size() * 2;
  }
  {
    const uint16_t* w = bf16_of(view, set, lp + "linear_attn.dt_bias", Qwen35::kGdnVHeads);
    std::vector<float> f(Qwen35::kGdnVHeads);
    for (size_t i = 0; i < f.size(); ++i) f[i] = common::bf16_to_f32(w[i]);
    std::memcpy(b.data() + kGdnOffDtBias, f.data(), f.size() * 4);
    widen += f.size() * 4 - f.size() * 2;
  }
  // RMSNormGated is plain w - the one norm in the model without the +1.
  std::memcpy(b.data() + kGdnOffGatedNorm,
              bf16_of(view, set, lp + "linear_attn.norm.weight", Qwen35::kGdnHeadDim),
              size_t(Qwen35::kGdnHeadDim) * 2);
  rep.small_bytes += kGdnBlockBytes;
  return {std::move(norms_mem), upload(ctx, imm, b.data(), kGdnBlockBytes)};
}

// cos/sin[p][0..1][i] for the 64 rotary dims (partial_rotary_factor 0.25 of
// head_dim 256), i < 32. Text-mode interleaved mRoPE copies one position id
// onto every frequency stream, so this is plain RoPE (docs/03 "Layer math").
std::vector<float> rope_table(uint32_t max_len) {
  const uint32_t half = Qwen35::kRotaryDim / 2;
  std::vector<float> t(size_t(max_len) * 2 * half);
  std::vector<double> inv(half);
  for (uint32_t i = 0; i < half; ++i)
    inv[i] = std::pow(Qwen35::kRopeTheta, -2.0 * double(i) / double(Qwen35::kRotaryDim));
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < half; ++i) {
      const double a = double(p) * inv[i];
      t[(size_t(p) * 2 + 0) * half + i] = float(std::cos(a));
      t[(size_t(p) * 2 + 1) * half + i] = float(std::sin(a));
    }
  return t;
}

}  // namespace

size_t LoadReport::total() const {
  return int4_bytes + scale_bytes + bf16_linear_bytes + embed_bytes + lm_head_bytes + small_bytes +
         pad_bytes;
}

LoadedModel load(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);

  std::ifstream cf(snap + "config.json");
  if (!cf) throw std::runtime_error("cannot read " + snap + "config.json");
  std::stringstream cs;
  cs << cf.rdbuf();
  const QuantConfig qc = QuantConfig::parse(common::json::parse(cs.str()));

  SafetensorsSet set(snap);
  assert_quant_invariants(set);   // requirement 6: before a single byte is repacked
  NameView view = build_view(set);

  // embed_tokens is uploaded row-major and verbatim: it is gathered one row per
  // token, so no tiling helps and the mmap is already the canonical layout.
  // Copied, not bound by reference: gcc 13's -Wdangling-reference cannot see
  // that the referent lives in the set's map, not in the temporary name.
  const TensorInfo emb = take(view, set, "embed_tokens.weight");
  if (emb.dtype != "BF16" || emb.shape.size() != 2 || emb.shape[0] != Qwen35::kVocab ||
      emb.shape[1] != Qwen35::kHidden)
    throw std::runtime_error("embed_tokens.weight: expected BF16 [" +
                             std::to_string(Qwen35::kVocab) + "][" +
                             std::to_string(Qwen35::kHidden) + "], got " + emb.dtype);
  // The model's final pre-lm_head RMSNorm - the one norm that belongs to no
  // layer. Same Gemma-style (1 + w) bake as all the others.
  std::vector<uint16_t> fnorm(Qwen35::kHidden);
  bake_one_plus(bf16_of(view, set, "norm.weight", Qwen35::kHidden), fnorm.data(),
                Qwen35::kHidden);
  const std::vector<float> rope = rope_table(max_len);

  LoadedModel m{{},
                {},
                l0::Mem(ctx, l0::MemKind::Device, set.bytes(emb)),
                l0::Mem(ctx, l0::MemKind::Device, fnorm.size() * 2),
                l0::Mem(ctx, l0::MemKind::Device, rope.size() * 4),
                {}};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(m.embed.ptr(), set.data(emb), set.bytes(emb));
  m.report.embed_bytes += set.bytes(emb);
  imm.copy(m.final_norm.ptr(), fnorm.data(), fnorm.size() * 2);
  m.report.small_bytes += fnorm.size() * 2;
  const size_t rope_bytes = rope.size() * 4;
  imm.copy(m.rope.ptr(), rope.data(), rope_bytes);
  m.report.small_bytes += rope_bytes;

  Staging st;
  st.i4.resize(size_t(Qwen35::shape(model::LinearId::GateUp).N / 16) *
               (Qwen35::shape(model::LinearId::GateUp).K / 64) * 136);
  st.bf_src.resize(size_t(Qwen35::shape(model::LinearId::AB).N) *
                   Qwen35::shape(model::LinearId::AB).K);
  st.bf_tiled.resize(size_t(Qwen35::shape(model::LinearId::LmHead).N) *
                     Qwen35::shape(model::LinearId::LmHead).K);

  size_t widen = 0;
  const std::vector<model::LayerDesc> layers = Qwen35::layers();
  m.layer_small.reserve(layers.size());
  for (const model::LayerDesc& ld : layers) {
    const std::string lp = Qwen35::layer_prefix(ld.index);
    for (const model::FusedLinear& fl : ld.linears)
      m.linears.emplace(std::make_pair(ld.index, fl.id),
                        load_linear(ctx, imm, set, view, lp, fl, st, m.report));
    m.layer_small.push_back(load_small(ctx, imm, set, view, ld, m.report, widen));
  }
  {
    model::FusedLinear head{model::LinearId::LmHead, Qwen35::shape(model::LinearId::LmHead),
                            model::WeightKind::Bf16, model::Fuse::Single, {"lm_head"}, 0};
    m.linears.emplace(std::make_pair(uint32_t(65535), model::LinearId::LmHead),
                      load_linear(ctx, imm, set, view, "", head, st, m.report));
  }

  std::string first_unconsumed;
  for (const auto& [stripped, full] : view.names) {
    (void)full;
    if (view.consumed.count(stripped)) continue;
    if (stripped.size() > 7 && stripped.compare(stripped.size() - 7, 7, ".qzeros") == 0) continue;
    if (stripped.size() > 6 && stripped.compare(stripped.size() - 6, 6, ".g_idx") == 0) continue;
    if (m.report.unconsumed++ == 0) first_unconsumed = stripped;
  }

  const LoadReport& r = m.report;
  const double gb = 1e9;
  // The RoPE table is resident but NOT streamed per token: the decode step reads
  // one position's 2 x 32 floats (~256 B), not the 4.19 MB table. It therefore
  // sits outside the read-per-token figure on both sides of the cross-check,
  // and outside `small` in the printout - but inside total(), because the
  // seven buckets must account for every byte allocated.
  const size_t small_resident = r.small_bytes - rope_bytes;
  const size_t per_token = r.int4_bytes + r.scale_bytes + r.bf16_linear_bytes + r.pad_bytes +
                           r.lm_head_bytes + small_resident;
  const double expected = kDocW + double(r.pad_bytes) + double(widen);
  const double delta = (double(per_token) - expected) / expected;
  m.report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf(
      "loader: %s\n"
      "  quant     int4 g%u sym desc_act=false, %zu dynamic exclusion rules\n"
      "  tensors   %zu language-model + lm_head; skipped %zu visual, %zu mtp;\n"
      "            dropped %zu qzeros + %zu g_idx (invariants asserted), %zu unconsumed%s%s\n"
      "  linears   %zu fused weights, %zu layers\n"
      "  int4        %13zu B  %7.3f GB\n"
      "  scales      %13zu B  %7.3f GB\n"
      "  bf16_linear %13zu B  %7.3f GB   (a‖b real rows; %.3f GB with padding)\n"
      "  lm_head     %13zu B  %7.3f GB\n"
      "  small       %13zu B  %7.3f GB   (per-layer blocks + final norm)\n"
      "  pad         %13zu B  %7.3f GB\n"
      "  read/token  %13zu B  %7.3f GB\n"
      "  embed       %13zu B  %7.3f GB   (resident, gathered - not per-token)\n"
      "  rope        %13zu B  %7.3f GB   (resident, ~256 B per token - not per-token)\n"
      "  total       %13zu B  %7.3f GB\n"
      "  W check     %.3f GB vs %.3f GB expected = %.3f doc-03 + %.3f pad + %.6f widen"
      "  ->  %+.3f%%\n"
      "  load        %.1f s\n",
      snap.c_str(), qc.group_size, qc.dynamic_rule_count, view.names.size() - 1,
      view.visual_skipped, view.mtp_skipped, view.qzeros, view.g_idx, r.unconsumed,
      r.unconsumed ? " incl. " : "", first_unconsumed.c_str(), m.linears.size(),
      m.layer_small.size(), r.int4_bytes, r.int4_bytes / gb, r.scale_bytes, r.scale_bytes / gb,
      r.bf16_linear_bytes, r.bf16_linear_bytes / gb, (r.bf16_linear_bytes + r.pad_bytes) / gb,
      r.lm_head_bytes, r.lm_head_bytes / gb, small_resident, small_resident / gb,
      r.pad_bytes, r.pad_bytes / gb, per_token, per_token / gb, r.embed_bytes, r.embed_bytes / gb,
      rope_bytes, rope_bytes / gb, r.total(), r.total() / gb, per_token / gb, expected / gb,
      kDocW / gb, r.pad_bytes / gb, widen / gb, delta * 100.0, m.report.seconds);

  if (std::fabs(delta) > 0.02)
    throw std::runtime_error("resident read-per-token bytes " + std::to_string(per_token) +
                             " differ from docs/03-models.md's W plus itemised padding/widening " +
                             std::to_string(uint64_t(expected)) + " by " +
                             std::to_string(delta * 100.0) + "% (> 2%) - the table above is the breakdown");
  return m;
}

}  // namespace loader

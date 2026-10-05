#include "loader/loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/fold.h"
#include "loader/moe.h"
#include "loader/safetensors.h"
#include "loader/small_layout.h"
#include "loader/trained_context.h"

namespace loader {
namespace {

using model::Qwen35;

// The per-layer small-tensor block offsets live in loader/small_layout.h - the
// one place they are defined, shared with model::LayerDesc::small_tensors (the
// table this file walks) and with plan 3's kernels. Nothing here re-derives an
// offset or names a tensor: `load_small` executes the table, and both throw if
// the table and the header disagree.

// `W` - the read-per-token bytes with a bf16 lm_head - is the model
// descriptor's `doc_w` (spec 14): Qwen3.8's is docs/03-models.md's byte
// accounting (measured 2026-08-24): 12.163 qweight + 0.760 scales + 0.052 bf16
// smalls + 2.543 lm_head = 15.519 GB. Agnes's 18.344 GB is the same four
// categories summed from its safetensors headers (14.816 + 0.926 + 0.059 +
// 2.543; the fold moves bytes, it adds none).

// doc-03's `W` was measured over a checkpoint with a **bf16** `lm_head`. A
// checkpoint that packs it (spec 1.6 §5.1) reads 0.675 GB there instead of
// 2.543, and that difference goes on the EXPECTED side of the cross-check as
// an itemised term - never into a widened tolerance, which is the same rule
// the padding and the fp32 widening follow.
// Spec 15b: per model, K = hidden (Qwen3.8 in the comments).
size_t lm_head_bf16_bytes(const model::ModelDesc& d) {               // 2 542 796 800
  return size_t(d.hidden) * Qwen35::kVocab * 2;
}
size_t lm_head_int4_bytes(const model::ModelDesc& d) {
  return size_t(d.hidden) * Qwen35::kVocab / 2 +                     //   635 699 200
         size_t(d.hidden) * Qwen35::kVocab / 32;                     //    39 731 200
}

// Bytes this loader adds over the checkpoint's own bf16 by widening a tensor to
// fp32 at load. Itemised rather than hidden in a tolerance: the W cross-check
// adds exactly this to its expected side and the report prints the split.
struct Widen {
  size_t norm = 0;   // the RMSNorm family, stored fp32 (1 + w)
  size_t gdn = 0;    // conv1d, -exp(A_log), dt_bias - the fp32 recurrence
  size_t total() const { return norm + gdn; }
};

// ---------------------------------------------------------------------------
// Name mapping (requirement 1). The checkpoint carries three top-level
// namespaces; the model description knows only the language model's, with the
// prefix stripped, plus `lm_head` which is genuinely top level.
// ---------------------------------------------------------------------------
constexpr const char* kLmPrefix = "model.language_model.";

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// Rebuilds the checkpoint name of a stripped (model-description) name: the
// prefix back on, and the model's name map run backwards (spec 14 §3.2: an
// engine `linear_attn.` is Agnes's `delta_attn.`; the identity on Qwen3.8).
std::string ckpt_name(const model::ModelDesc& d, const std::string& stripped) {
  return starts_with(stripped, "lm_head") ? stripped : kLmPrefix + d.to_checkpoint(stripped);
}

bool has_suffix(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() > n && s.compare(s.size() - n, n, suf) == 0;
}

// `self_attn.k_scale` / `self_attn.v_scale`: an fp8 KV cache's calibration (NameView).
bool is_kv_scale(const std::string& s) {
  return has_suffix(s, "self_attn.k_scale") || has_suffix(s, "self_attn.v_scale");
}

struct NameView {
  std::map<std::string, std::string> names;  // stripped -> checkpoint name
  std::set<std::string> consumed;            // stripped names actually loaded
  size_t mtp_skipped = 0, visual_skipped = 0, qzeros = 0, g_idx = 0;
  // compressed-tensors' fp8 KV-cache calibration (`self_attn.k_scale` / `v_scale`,
  // QuantConfig::ct_kv_cache_scheme): dropped by name - this engine's KV cache is bf16
  // or its own int8 - and counted, like qzeros.
  size_t kv_scales = 0;
  size_t top_level = 0;                      // names outside model.language_model.
};

// Every kept name is mapped to its ENGINE spelling here, once (spec 14 §3.2:
// Agnes's `delta_attn.` / `global_attn.` -> `linear_attn.` / `self_attn.`, the
// MTP head's included); everything downstream binds engine names only.
NameView build_view(const SafetensorsSet& set, bool keep_mtp, const model::ModelDesc& d) {
  NameView v;
  for (const auto& [name, info] : set.tensors()) {
    (void)info;
    if (starts_with(name, "model.visual.")) {          // no vision tower in v1
      ++v.visual_skipped;
    } else if (starts_with(name, "mtp.") && keep_mtp) {   // spec 8: kept on request
      v.names.emplace(d.to_engine(name), name);             // top-level, name-mapped
    } else if (starts_with(name, "mtp.")) {            // no speculation unless asked
      // The RTN checkpoint's `mtp.*` are 29 tensors in their own shard
      // (`model_extra_tensors.safetensors`, pointed at by the index like any
      // other) and 8 of them are int4 rather than the published checkpoint's
      // all-bf16 15. Skipping is by NAME, before the shard matters: the file is
      // mmapped and its header parsed because the index names it, and then
      // nothing in it is ever read. It is counted here, so it lands in the
      // report's `mtp` figure and can never land in `unconsumed`.
      ++v.mtp_skipped;
    } else if (starts_with(name, kLmPrefix)) {
      std::string stripped = d.to_engine(name.substr(std::strlen(kLmPrefix)));
      if (has_suffix(stripped, ".qzeros")) ++v.qzeros;
      if (has_suffix(stripped, ".g_idx")) ++v.g_idx;
      if (is_kv_scale(stripped)) ++v.kv_scales;
      v.names.emplace(std::move(stripped), name);
    } else {
      // lm_head - `.weight` on the published checkpoint, `.qweight`/`.qzeros`/
      // `.scales` on one that packed it. The drop counters cover this branch
      // too, so an int4 head's qzeros are reported as dropped rather than
      // silently missing from the tally.
      if (has_suffix(name, ".qzeros")) ++v.qzeros;
      if (has_suffix(name, ".g_idx")) ++v.g_idx;
      ++v.top_level;
      v.names.emplace(name, name);
    }
  }
  return v;
}

const TensorInfo& take(NameView& v, const SafetensorsSet& set, const std::string& stripped) {
  auto it = v.names.find(stripped);
  if (it == v.names.end())
    throw std::runtime_error("checkpoint has no tensor '" + stripped + "' (engine name; the " +
                             "checkpoint spelling is the model's name map run backwards)");
  v.consumed.insert(stripped);
  return set.tensors().at(it->second);
}

// ---------------------------------------------------------------------------
// Small tensors: the model description's table is walked, never second-guessed.
// ---------------------------------------------------------------------------

// The device element size implied by a bake kind. small_layout.h's block
// offsets are derived from these, and load_small's bounds check pins the two
// together - a table entry that would overrun its block throws by name.
size_t bake_elem_bytes(model::SmallBake b) {
  return b == model::SmallBake::PlainBf16 ? 2 : 4;
}
size_t src_elem_bytes(const model::SmallTensor& t) {
  return std::strcmp(t.dtype, "F32") == 0 ? 4 : 2;
}

// The source bytes of one table entry, with the dtype and the exact element
// count the model description declares enforced by name.
const uint8_t* small_src(NameView& v, const SafetensorsSet& set, const std::string& full,
                         const model::SmallTensor& t) {
  const TensorInfo& info = take(v, set, full);
  if (info.dtype != t.dtype)
    throw std::runtime_error("tensor '" + full + "': dtype " + info.dtype + ", expected " +
                             t.dtype);
  const size_t esz = src_elem_bytes(t);
  if (set.bytes(info) != size_t(t.elems) * esz)
    throw std::runtime_error("tensor '" + full + "': " + std::to_string(set.bytes(info) / esz) +
                             " elements, expected " + std::to_string(t.elems));
  const uint8_t* p = set.data(info);
  check_align(p, esz, full);
  return p;
}

// The bakes themselves (docs/13-loader.md "What the loader bakes in").
void bake_small(const model::SmallTensor& t, const uint8_t* src, uint8_t* dst, Widen& widen) {
  const size_t n = t.elems;
  const uint16_t* w = reinterpret_cast<const uint16_t*>(src);
  std::vector<float> f;
  switch (t.bake) {
    case model::SmallBake::PlainBf16:
      // RMSNormGated: plain w, and bf16 - the reference's own parameter dtype
      // is bf16 and it multiplies in the bf16 domain, so widening would move
      // away from the oracle rather than towards it.
      std::memcpy(dst, src, n * 2);
      return;
    case model::SmallBake::OnePlusWFp32:
      // Gemma-style RMSNorm: x*rsqrt(mean(x^2)+eps)*(1 + w). The reference does
      // the whole product in fp32, so the `+1` is added in fp32 and STORED
      // fp32 - no cast back to bf16, which removes a rounding the oracle never
      // had (up to 0.39% on a multiplier near 1). Ruling 2026-08-25.
      f.resize(n);
      for (size_t i = 0; i < n; ++i) f[i] = 1.0f + common::bf16_to_f32(w[i]);
      widen.norm += n * 2;
      break;
    case model::SmallBake::NegExpFp32:
      // The GDN decay is only ever used as exp(g) with
      // g = -exp(A_log) * softplus(a + dt_bias): the exp is a per-head
      // constant, hoisted out of 48 layers x every token.
      f.resize(n);
      for (size_t i = 0; i < n; ++i) f[i] = -std::exp(common::bf16_to_f32(w[i]));
      widen.gdn += n * 2;
      break;
    case model::SmallBake::RawFp32Widen:
      // The recurrence and the 4-tap depthwise conv accumulate in fp32:
      // widening once at load saves a convert per tap per token.
      f.resize(n);
      for (size_t i = 0; i < n; ++i) f[i] = common::bf16_to_f32(w[i]);
      widen.gdn += n * 2;
      break;
    case model::SmallBake::RawFp32:
      std::memcpy(dst, src, n * 4);   // already fp32 in the checkpoint
      return;
  }
  std::memcpy(dst, f.data(), n * 4);
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
    case model::LinearId::kCount: break;   // the table size, not a linear
  }
  return "?";
}

// Host staging, allocated once and reused, and sized from the linears this
// load will actually repack - **which is not a constant across checkpoints**.
// With a bf16 `lm_head` the largest int4 linear is `gate‖up` (2176 n-tiles x
// 80 k-groups x 136 u32 = 94.7 MB) and the bf16 tile buffer must hold
// `lm_head`'s 2.54 GB. With an int4 `lm_head` the two swap places: the int4
// buffer has to hold 15520 x 80 x 136 u32 = **675.4 MB** - 7.1x `gate‖up`, the
// largest int4 tensor this repack has ever seen - and the bf16 tile buffer
// only ever holds `a‖b` at [128][5120], 1.3 MB. Sizing both for the maximum
// would cost 2.5 GB of host RSS for nothing.
//
// The staging bound is not the only place that has to hold: `load_linear`
// throws by name if a linear does not fit what it was given, which is what
// turns a mis-sized buffer into a message instead of a heap overrun.
struct Staging {
  std::vector<int8_t> i8;           // spec 9: the int8 lm_head, tiled (lm_head_int8.h)
  std::vector<float> i8_scales;     // its fp32 row scales
  std::vector<uint32_t> i4;
  std::vector<uint16_t> i4_scales;
  std::vector<uint16_t> bf_src;
  std::vector<uint16_t> bf_tiled;
};

// The int4 weight staging words one linear needs. Layout 0 keeps the GPTQ
// [K/8][N] words; layout 1 builds n-tiles x k-groups x 136 words, including
// its inline scales.
size_t int4_words(const model::GemvShape& s) {
  return s.layout == 0 ? size_t(s.K / 8) * s.N
                       : size_t(s.N / 16) * (s.K / 64) * 136;
}
size_t int4_scale_elems(const model::GemvShape& s) {
  return s.layout == 0 ? size_t(s.K / 64) * s.N : 0;
}

// Spec 14 §2: join each part with its fold part (Fold::N: columns, part i with
// fold part i; Fold::K: rows, the one part with the one fold part), on packed
// int4 (loader/fold.h). The joined tensors live in `keep`; the returned sources
// point into it. Every fold part must be int4 - the parallel FFN is quantised
// exactly like the MLP it joins (the checkpoint's modules_in_block_to_quantize).
std::vector<LinearSrc> apply_fold(const SafetensorsSet& set, NameView& view,
                                  const model::ModelDesc& desc, const std::string& layer_prefix,
                                  const model::FusedLinear& fl, std::vector<LinearSrc> srcs,
                                  std::vector<PackedInt4>& keep, const std::string& id) {
  const bool by_n = fl.fold == model::Fold::N;
  if (fl.kind != model::WeightKind::Int4)
    throw std::runtime_error(id + ": a fold is implemented for int4 linears only");
  if (by_n ? fl.fold_parts.size() != srcs.size()
           : (fl.fold_parts.size() != 1 || srcs.size() != 1))
    throw std::runtime_error(id + ": " + std::to_string(fl.fold_parts.size()) + " fold parts for " +
                             std::to_string(srcs.size()) + " parts (Fold::N pairs them, Fold::K "
                             "joins one with one)");
  keep.reserve(srcs.size());
  for (size_t i = 0; i < srcs.size(); ++i) {
    const std::string& fp = fl.fold_parts[i];
    const LinearSrc f = LinearSrc::classify(set, ckpt_name(desc, layer_prefix + fp));
    if (f.kind != WKind::Int4 || srcs[i].kind != WKind::Int4)
      throw std::runtime_error(id + ": fold part '" + fp + "' or its partner is not int4");
    for (const std::string& suffix : f.suffixes()) view.consumed.insert(layer_prefix + fp + suffix);
    const PackedInt4View a{srcs[i].K, srcs[i].N, srcs[i].qweight, srcs[i].scales};
    const PackedInt4View b{f.K, f.N, f.qweight, f.scales};
    keep.push_back(by_n ? fold_n(a, b) : fold_k(a, b));
    srcs[i].K = keep.back().K;
    srcs[i].N = keep.back().N;
    srcs[i].qweight = keep.back().qweight.data();
    srcs[i].scales = keep.back().scales.data();
  }
  return srcs;
}

// One fused linear: classify every part, check the preconditions the repack
// helpers document but cannot check (Task-3 review minor, closed here at the
// first real consumer - these run in release too, load is one-shot), repack
// into staging and upload with a single copy.
DeviceWeight load_linear(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                         NameView& view, const model::ModelDesc& desc,
                         const std::string& layer_prefix, const model::FusedLinear& fl,
                         Staging& st, LoadReport& rep) {
  const model::GemvShape& sh = fl.shape;
  const std::string id = std::string(id_name(fl.id)) + " (" + layer_prefix + fl.parts[0] + ")";
  if (sh.K % 64 != 0 || sh.N % 16 != 0)
    throw std::runtime_error(id + ": shape K=" + std::to_string(sh.K) + " N=" +
                             std::to_string(sh.N) + " violates K%64==0, N%16==0");
  // `layout` is meaningful for int4 only - bf16 rows carry 0 as a documented
  // filler. Both supported int4 layouts have explicit paths below; rejecting
  // any other value here prevents it from silently taking layout 1's bytes.
  if (fl.kind == model::WeightKind::Int4 && sh.layout > 1)
    throw std::runtime_error(id + ": unsupported int4 layout " + std::to_string(sh.layout));

  std::vector<LinearSrc> srcs;
  for (const std::string& part : fl.parts) {
    // classify() checks the alignment of every pointer it casts (M6).
    LinearSrc s = LinearSrc::classify(set, ckpt_name(desc, layer_prefix + part));
    const bool int4 = s.kind == WKind::Int4;
    if (int4 != (fl.kind == model::WeightKind::Int4))
      throw std::runtime_error(id + ": part '" + part + "' is " + (int4 ? "int4" : "bf16") +
                               " but the model description says " +
                               (fl.kind == model::WeightKind::Int4 ? "int4" : "bf16") +
                               " - the checkpoint's dynamic exclusions moved");
    for (const std::string& suffix : s.suffixes()) view.consumed.insert(layer_prefix + part + suffix);
    srcs.push_back(s);
  }
  // Spec 14 §2: Agnes's parallel FFN joined onto the parts on packed int4 BEFORE
  // the column map, so gate'||up' interleaves exactly as gate||up does. `folded`
  // owns the joined tensors for the rest of this call; srcs[i] then points at them.
  std::vector<PackedInt4> folded;
  if (fl.fold != model::Fold::None) srcs = apply_fold(set, view, desc, layer_prefix, fl, srcs, folded, id);
  uint32_t n_sum = 0;
  for (size_t i = 0; i < srcs.size(); ++i) {
    if (srcs[i].K != sh.K)
      throw std::runtime_error(id + ": part '" + fl.parts[i] + "' has K=" + std::to_string(srcs[i].K) +
                               ", the shape table says " + std::to_string(sh.K));
    n_sum += srcs[i].N;
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
    const size_t words = int4_words(sh);
    if (words > st.i4.size())
      throw std::runtime_error(id + ": " + std::to_string(words * 4) +
                               " bytes exceeds the int4 staging buffer");
    const size_t scale_elems = int4_scale_elems(sh);
    if (scale_elems > st.i4_scales.size())
      throw std::runtime_error(id + ": " + std::to_string(scale_elems * 2) +
                               " bytes exceeds the layout-0 scale staging buffer");
    if (sh.layout == 0)
      common::repack_int4_layout0_cols(sh.K, sh.N, cols, st.i4.data(), st.i4_scales.data());
    else
      common::repack_int4_layout1_cols(sh.K, sh.N, cols, st.i4.data());
    // The 136-u32 tile is 128 u32 of nibbles + 8 u32 of scales, so the two
    // buckets below add up to exactly the bytes uploaded. `lm_head` keeps its
    // OWN bucket in both kinds - it is the one row whose format the checkpoint
    // chooses, so the report has to show it separately for the two to be
    // comparable at all, and the W cross-check adjusts exactly this line.
    if (fl.id == model::LinearId::LmHead) {
      rep.lm_head_bytes += size_t(sh.K) * sh.N / 2 + size_t(sh.K) * sh.N / 32;
    } else {
      rep.int4_bytes += size_t(sh.K) * sh.N / 2;
      rep.scale_bytes += size_t(sh.K) * sh.N / 32;
    }
    l0::Mem weight = upload(ctx, imm, st.i4.data(), words * 4);
    std::unique_ptr<l0::Mem> scales;
    if (sh.layout == 0)
      scales = std::make_unique<l0::Mem>(
          upload(ctx, imm, st.i4_scales.data(), scale_elems * sizeof(uint16_t)));
    return {std::move(weight), std::move(scales), sh, fl.kind};
  }

  if (fl.kind == model::WeightKind::Int8) {
    // Spec 9 §3: only `lm_head`, one bf16 part, quantised on the host into the
    // tiled int8 layout gemv_i8w reads plus an fp32 scale per row.
    if (fl.id != model::LinearId::LmHead || srcs.size() != 1 || n_sum != sh.N)
      throw std::runtime_error(id + ": int8 is implemented for the unpadded lm_head only");
    if (size_t(sh.N) * sh.K > st.i8.size() || sh.N > st.i8_scales.size())
      throw std::runtime_error(id + ": exceeds the int8 staging buffer");
    const auto q0 = std::chrono::steady_clock::now();
    quantise_int8_tiled(srcs[0].weight, sh.K, sh.N, st.i8.data(), st.i8_scales.data());
    rep.lm_head_quant_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
    rep.lm_head_bytes += lm_head_int8_bytes(sh.K, sh.N);
    l0::Mem weight = upload(ctx, imm, st.i8.data(), size_t(sh.N) * sh.K);
    auto scales = std::make_unique<l0::Mem>(
        upload(ctx, imm, st.i8_scales.data(), size_t(sh.N) * sizeof(float)));
    return {std::move(weight), std::move(scales), sh, fl.kind};
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
  return {upload(ctx, imm, st.bf_tiled.data(), elems * 2), nullptr, sh, fl.kind};
}

// Everything in a layer that is not a GEMV weight, packed into the two blocks
// of loader/small_layout.h. This function knows no tensor names and no offsets:
// it walks model::LayerDesc::small_tensors, which carries both (fix I3). The
// two throws below are what fires if the table and the header disagree.
SmallTensors load_small(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                        NameView& view, const model::ModelDesc& desc,
                        const model::LayerDesc& ld, LoadReport& rep, Widen& widen) {
  const std::string lp = Qwen35::layer_prefix(ld.index);
  const SmallLayout sl = desc.small_layout();
  const size_t kind_bytes = ld.kind == model::LayerKind::FA ? kFaBlockBytes : sl.gdn_block_bytes;
  std::vector<uint8_t> norms(sl.norms_block_bytes, 0), kind(kind_bytes, 0);
  size_t filled_norms = 0, filled_kind = 0;

  for (const model::SmallTensor& t : ld.small_tensors) {
    const bool in_norms = t.block == model::SmallBlock::Norms;
    std::vector<uint8_t>& dst = in_norms ? norms : kind;
    const size_t bytes = size_t(t.elems) * bake_elem_bytes(t.bake);
    if (size_t(t.offset) + bytes > dst.size())
      throw std::runtime_error(
          "small tensor '" + lp + t.name + "': " + std::to_string(bytes) + " B at offset " +
          std::to_string(t.offset) + " overruns its " + std::to_string(dst.size()) +
          "-byte block - model::LayerDesc::small_tensors and loader/small_layout.h disagree");
    bake_small(t, small_src(view, set, lp + t.name, t), dst.data() + t.offset, widen);
    (in_norms ? filled_norms : filled_kind) += bytes;
  }
  if (filled_norms != norms.size() || filled_kind != kind.size())
    throw std::runtime_error(
        "layer " + std::to_string(ld.index) + ": the small-tensor table fills " +
        std::to_string(filled_norms) + "/" + std::to_string(norms.size()) + " norms B and " +
        std::to_string(filled_kind) + "/" + std::to_string(kind.size()) +
        " kind-block B - every byte of a block must have an owner");

  rep.small_bytes += norms.size() + kind.size();
  return {upload(ctx, imm, norms.data(), norms.size()),
          upload(ctx, imm, kind.data(), kind.size())};
}

// Spec 15c: one MoE layer's host copy (loader/moe.h) onto the card as three allocations
// (loader/moe_layout.h) - the main layers' and (spec 15e) the MTP head's.
MoeLayer upload_moe_layer(l0::Context& ctx, l0::CmdList& imm, const model::ModelDesc& desc,
                          const MoeHost& host) {
  const MoeLayerBytes b = moe_layer_bytes(desc);
  if (host.router.size() * 2 != b.router || host.gate_up.size() * 4 != b.gate_up() ||
      host.down.size() * 4 != b.down())
    throw std::logic_error("loader: the MoE host copy is not the moe_layer_bytes() layout");
  return MoeLayer{{upload(ctx, imm, host.router.data(), b.router), nullptr,
                   model::GemvShape{desc.hidden, desc.moe.router_n(), 1, 0},
                   model::WeightKind::Bf16},
                  upload(ctx, imm, host.gate_up.data(), b.gate_up()),
                  upload(ctx, imm, host.down.data(), b.down())};
}

// Spec 8 §3.1: the MTP head. The published checkpoint ships exactly 15 bf16
// `mtp.*` tensors (docs/03-models.md); anything else (the RTN checkpoint's 29,
// 8 of them int4) is refused by name rather than half-loaded. The linears are
// laid out for gemv_bf16 (common::repack_bf16_tiled) with the main model's
// fusions: q||k||v concatenated like the FA layers' Qkv, gate/up interleaved in
// 16-column blocks like GateUp (prep_silu_mul's gflat/uflat).
//
// Spec 15e: a MoE head (ModelDesc::mtp_head_moe - Ornith's 785 bf16 tensors) has the
// same fc, attention and norms, and as its FFN one MoE layer of the main model's shape:
// `mtp.layers.0.mlp.{gate, shared_expert_gate, experts.E.*, shared_expert.*}`, repacked
// into the main layers' device form by loader::repack_moe_layer with its bf16 experts
// quantised at load (loader/rtn.h; an int4-shipped expert is repacked as shipped). Every
// name it reads is marked consumed, so an extra head tensor shows up in `unconsumed`.
std::unique_ptr<MtpHead> load_mtp(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                                  NameView& view, const model::ModelDesc& desc, LoadReport& rep,
                                  Widen& widen) {
  const bool moe_head = desc.mtp_head_moe();
  if (desc.mtp_intermediate == 0 && !moe_head)
    throw std::runtime_error(desc.name + ": the descriptor names no MTP head FFN");
  size_t n_mtp = 0;
  for (const auto& [name, info] : set.tensors()) {
    (void)info;
    n_mtp += starts_with(name, "mtp.");
  }
  if (!moe_head && n_mtp != kMtpTensors)
    throw std::runtime_error("the MTP head must be the published checkpoint's " +
                             std::to_string(kMtpTensors) + " bf16 mtp.* tensors; this one has " +
                             std::to_string(n_mtp));
  // One bf16 [rows][K] tensor, checked by name, dtype and shape.
  auto tensor = [&](const std::string& name, uint32_t rows, uint32_t K) {
    const TensorInfo info = take(view, set, name);
    const bool ok = info.dtype == "BF16" &&
                    (K == 0 ? info.shape.size() == 1 && info.shape[0] == rows
                            : info.shape.size() == 2 && info.shape[0] == rows && info.shape[1] == K);
    if (!ok)
      throw std::runtime_error("MTP tensor '" + name + "': expected BF16 [" + std::to_string(rows) +
                               (K ? "][" + std::to_string(K) : std::string()) + "], got " +
                               info.dtype);
    rep.mtp_checkpoint_bytes += set.bytes(info);
    ++rep.mtp_tensors;
    const uint8_t* p = set.data(info);
    check_align(p, 2, name);
    return reinterpret_cast<const uint16_t*>(p);
  };
  std::vector<uint16_t> rows_buf, tiled;
  // parts stacked row-major ([N][K]), or interleaved in 16-row blocks, then tiled.
  auto linear = [&](const std::vector<std::string>& parts, const std::vector<uint32_t>& ns,
                    uint32_t K, bool interleave16) {
    uint32_t N = 0;
    for (uint32_t n : ns) N += n;
    const size_t elems = size_t(N) * K;
    rows_buf.assign(elems, 0);
    tiled.resize(elems);
    std::vector<const uint16_t*> src;
    for (size_t i = 0; i < parts.size(); ++i) src.push_back(tensor(parts[i], ns[i], K));
    if (interleave16) {
      for (uint32_t r = 0; r < N; ++r) {
        const uint32_t blk = r / 32, in = r % 32;
        const uint16_t* s = src[in < 16 ? 0 : 1] + (size_t(blk) * 16 + in % 16) * K;
        std::memcpy(rows_buf.data() + size_t(r) * K, s, size_t(K) * 2);
      }
    } else {
      size_t row = 0;
      for (size_t i = 0; i < src.size(); ++i) {
        std::memcpy(rows_buf.data() + row * K, src[i], size_t(ns[i]) * K * 2);
        row += ns[i];
      }
    }
    common::repack_bf16_tiled(rows_buf.data(), K, N, tiled.data());
    rep.mtp_bytes += elems * 2;
    return DeviceWeight{upload(ctx, imm, tiled.data(), elems * 2), nullptr,
                        model::GemvShape{K, N, 1, 0}, model::WeightKind::Bf16};
  };
  const std::string L = "mtp.layers.0.";
  // The head's shapes are the descriptor's (spec 15b): its own MLP width
  // `mtp_intermediate` is 17408 on Qwen3.8 AND on Agnes (no parallel FFN in the
  // head; spec 14 §1), whatever the main model's folded intermediate is; the
  // attention follows the main model's heads. Qwen3.8's numbers in the comments.
  // Names are engine names (the view mapped Agnes's `global_attn.` already).
  const uint32_t H = desc.hidden, I = desc.mtp_intermediate, KV = desc.fa_kv_n();
  DeviceWeight fc = linear({"mtp.fc.weight"}, {H}, 2 * H, false);          // 5120 x 10240
  DeviceWeight qkv = linear({L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight",
                             L + "self_attn.v_proj.weight"},
                            {desc.fa_q_proj_n(), KV, KV}, H, false);        // 12288, 1024, 1024
  DeviceWeight o =
      linear({L + "self_attn.o_proj.weight"}, {H}, desc.fa_value_dim(), false);   // K 6144
  std::unique_ptr<DeviceWeight> gate_up, down;
  std::unique_ptr<MoeLayer> moe;
  if (!moe_head) {
    gate_up = std::make_unique<DeviceWeight>(
        linear({L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"}, {I, I}, H, true));  // 17408
    down = std::make_unique<DeviceWeight>(linear({L + "mlp.down_proj.weight"}, {H}, I, false));
  } else {
    // Spec 15e: the MoE layer, by the main layers' MoeSource binding with the head's
    // prefix (`mtp.` names are not under model.language_model., and are already engine
    // names in the view), counting what it reads into the head's report fields.
    MoeSource src;
    src.has = [&](const std::string& part) {
      for (const std::string& m : LinearSrc::marker_suffixes())
        if (view.names.count(L + part + m) != 0) return true;
      return false;
    };
    src.linear = [&](const std::string& part) {
      const std::string eng = L + part;
      LinearSrc s = LinearSrc::classify(set, desc.to_checkpoint(eng));
      for (const std::string& suffix : s.suffixes()) {
        const TensorInfo info = take(view, set, eng + suffix);
        rep.mtp_checkpoint_bytes += set.bytes(info);
        ++rep.mtp_tensors;
      }
      return s;
    };
    const auto r0 = std::chrono::steady_clock::now();
    MoeHost host;
    repack_moe_layer(desc, src, host, "the MTP head", /*rtn_bf16_experts=*/true);
    rep.mtp_rtn_linears = host.rtn_linears;
    rep.mtp_rtn_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();
    moe = std::make_unique<MoeLayer>(upload_moe_layer(ctx, imm, desc, host));
    rep.mtp_bytes += moe_layer_bytes(desc).total();
  }
  auto h = std::make_unique<MtpHead>(MtpHead{
      std::move(fc), std::move(qkv), std::move(o), std::move(gate_up), std::move(down),
      l0::Mem(ctx, l0::MemKind::Device, mtp_norms_bytes(desc)),
      l0::Mem(ctx, l0::MemKind::Device, kFaBlockBytes), std::move(moe)});
  // The RMSNorms, baked like every other (1 + w) fp32 norm.
  std::vector<uint8_t> norms(mtp_norms_bytes(desc)), fa(kFaBlockBytes);
  const std::pair<const char*, MtpNorm> nrm[] = {
      {"mtp.pre_fc_norm_embedding.weight", kMtpNormPreE},
      {"mtp.pre_fc_norm_hidden.weight", kMtpNormPreH},
      {"mtp.layers.0.input_layernorm.weight", kMtpNormInput},
      {"mtp.layers.0.post_attention_layernorm.weight", kMtpNormPost},
      {"mtp.norm.weight", kMtpNormFinal}};
  for (const auto& [name, row] : nrm) {
    const size_t off = mtp_norm_off(desc, row);
    const model::SmallTensor d{name, H, "BF16", model::SmallBlock::Norms,
                               uint32_t(off), model::SmallBake::OnePlusWFp32};
    bake_small(d, reinterpret_cast<const uint8_t*>(tensor(name, H, 0)), norms.data() + off,
               widen);
  }
  for (const auto& [name, off] :
       {std::pair<std::string, size_t>{L + "self_attn.q_norm.weight", kFaOffQNorm},
        std::pair<std::string, size_t>{L + "self_attn.k_norm.weight", kFaOffKNorm}}) {
    const model::SmallTensor d{name, model::Qwen35::kFaHeadDim, "BF16", model::SmallBlock::Kind,
                               uint32_t(off), model::SmallBake::OnePlusWFp32};
    bake_small(d, reinterpret_cast<const uint8_t*>(tensor(name, model::Qwen35::kFaHeadDim, 0)),
               fa.data() + off, widen);
  }
  imm.copy(h->norms.ptr(), norms.data(), norms.size());
  imm.copy(h->fa.ptr(), fa.data(), fa.size());
  rep.mtp_bytes += norms.size() + fa.size();
  // What was read: a dense head is exactly the 15 bf16 tensors; a MoE head is held to the
  // all-bf16 figures when every expert linear arrived bf16 (the published checkpoint), and
  // otherwise to "every head tensor consumed" (load()'s unconsumed check).
  const bool all_bf16 = !moe_head || rep.mtp_rtn_linears == 3 * size_t(desc.moe.blocks());
  if (all_bf16 && (rep.mtp_tensors != desc.mtp_checkpoint_tensors() ||
                   rep.mtp_checkpoint_bytes != desc.mtp_checkpoint_bytes()))
    throw std::runtime_error("the MTP head consumed " + std::to_string(rep.mtp_tensors) +
                             " tensors / " + std::to_string(rep.mtp_checkpoint_bytes) +
                             " B, expected " + std::to_string(desc.mtp_checkpoint_tensors()) +
                             " / " + std::to_string(desc.mtp_checkpoint_bytes()));
  if (rep.mtp_bytes != mtp_head_bytes(desc))
    throw std::logic_error("loader: the MTP head allocated " + std::to_string(rep.mtp_bytes) +
                           " B, not the " + std::to_string(mtp_head_bytes(desc)) +
                           " B loader::mtp_head_bytes() plans");
  return h;
}

// Spec 15c: one MoE layer - the router || shared gate, every routed expert and the
// shared expert - repacked on the host (loader/moe.h, reusing `host` across layers)
// and uploaded as three allocations (loader/moe_layout.h). Every name it reads is
// marked consumed, so a checkpoint that ships an expert tensor this does not read
// shows up in `unconsumed` by name.
MoeLayer load_moe_layer(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                        NameView& view, const model::ModelDesc& desc, uint32_t layer,
                        MoeHost& host, LoadReport& rep) {
  const std::string lp = Qwen35::layer_prefix(layer);
  MoeSource src;
  src.has = [&](const std::string& part) {
    for (const std::string& m : LinearSrc::marker_suffixes())
      if (view.names.count(lp + part + m) != 0) return true;
    return false;
  };
  src.linear = [&](const std::string& part) {
    LinearSrc s = LinearSrc::classify(set, ckpt_name(desc, lp + part));
    for (const std::string& suffix : s.suffixes()) view.consumed.insert(lp + part + suffix);
    return s;
  };
  repack_moe_layer(desc, src, host, "layer " + std::to_string(layer));
  MoeLayer ml = upload_moe_layer(ctx, imm, desc, host);
  rep.moe_bytes += moe_layer_bytes(desc).total();
  return ml;
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

// load() and set_max_len()'s one bound (spec 6 §10). `trained` 0 = config.json
// declares no max_position_embeddings: nothing to hold max_len to.
void check_max_len(const model::ModelDesc& desc, uint32_t max_len, uint32_t trained) {
  if (max_len == 0) throw std::runtime_error(desc.name + ": max_len 0 is not a model length");
  if (trained != 0 && max_len > trained)
    throw std::runtime_error(
        desc.name + ": max_len " + std::to_string(max_len) + " exceeds the trained context " +
        std::to_string(trained) + " (config.json max_position_embeddings) - positions past it "
        "are RoPE angles the model never saw");
}

}  // namespace

size_t LoadReport::total() const {
  return int4_bytes + scale_bytes + bf16_linear_bytes + embed_bytes + lm_head_bytes + small_bytes +
         pad_bytes + mtp_bytes + moe_bytes;
}

LoadedModel load(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len,
                 bool mtp, LmHeadForm lm_form, const DraftVocabSpec& draft_vocab) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);

  std::ifstream cf(snap + "config.json");
  if (!cf) throw std::runtime_error("cannot read " + snap + "config.json");
  std::stringstream cs;
  cs << cf.rdbuf();
  const common::json::Value config = common::json::parse(cs.str());
  // Spec 14 §3.1: the model is picked from the checkpoint - no flag. An unknown
  // architecture throws here, naming config.json's value.
  const common::json::Value* archs = config.find("architectures");
  if (!archs || !archs->is_array() || archs->arr().empty() || !archs->arr()[0].is_string())
    throw std::runtime_error(snap + "config.json has no architectures[0] string");
  const model::ModelDesc& desc = model::desc_for_architecture(archs->arr()[0].str());
  // Spec 15b: a described model the engine cannot run yet (Ornith: "MoE not
  // implemented (spec 15c)") stops here, before a byte is read - and before the
  // quantisation config is parsed, so a bf16-only checkpoint (Ornith ships no
  // quantization_config) gets this message rather than the quant parser's.
  model::require_loadable(desc);
  const QuantConfig qc = QuantConfig::parse(config);
  // Spec 6 §10: the trained context bounds max_len; whether it fits on the card is the
  // memory plan's question (runtime/memory_plan.h), asked by the CLI once the weights'
  // bytes are known. Spec 14 §3.3's fixed 65536 ceiling for Agnes was that question
  // answered once, by hand, for one head form.
  const uint32_t trained = trained_context(config);
  check_max_len(desc, max_len, trained);

  SafetensorsSet set(snap);
  // Requirement 6: before a single byte is repacked. Returns what it counted
  // but did not reject (subnormal f16 scales, which this checkpoint has).
  const QuantScan scan = assert_quant_invariants(set);
  // The content proof behind an auto-round config's silence on `desc_act`
  // (quant.h). A permutation needs a `g_idx` to carry it; if none is shipped
  // there is nothing for `desc_act: true` to have meant, and the layout-1
  // repack's arithmetic group index is safe. If one IS shipped under a config
  // that never declared the key, this stops - the scan already proved every
  // entry is the identity, but "the config did not say" plus "there is a
  // permutation vector" is exactly the combination nobody should assume about.
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error(
        "config.json declares no desc_act (auto-round's spelling of false) but the checkpoint "
        "ships " + std::to_string(scan.g_idx_tensors) +
        " g_idx tensors - refusing to infer that no activation-order permutation exists. Add "
        "\"desc_act\": false to quantization_config if that is what the quantiser meant.");
  // The label against the bytes (GPTQ vs compressed-tensors) and compressed-tensors'
  // `actorder` "group" against its shipped g_idx (quant.h).
  check_quant_scan(qc, scan);
  // Operator requirement (2026-10-05): one note per load, not per tensor, when the
  // checkpoint is compressed-tensors - what is converted, and the recommended format.
  const std::string quant_note = ct_conversion_note(scan);
  if (!quant_note.empty()) std::fprintf(stderr, "loader: %s\n", quant_note.c_str());
  NameView view = build_view(set, mtp, desc);
  Widen widen;

  // **`lm_head`'s kind is the checkpoint's to choose, and it is chosen by
  // content.** `classify` looks for `lm_head.qweight` and falls back to
  // `lm_head.weight`, exactly as it does for every per-layer linear; the
  // config's `extra_config`/`dynamic` claims are never consulted (docs/02).
  // Everything downstream - the staging sizes, the byte buckets, the W
  // cross-check, the shape and layout the capture binds - follows from this
  // one line, so a checkpoint that packs the head and one that does not are
  // the same code path with a different row.
  const bool ckpt_int4 = LinearSrc::kind_of(set, ckpt_name(desc, "lm_head")) == WKind::Int4;
  // Spec 9: the int8 form is made from the bf16 tensor, so it needs one.
  if (lm_form == LmHeadForm::Int8 && ckpt_int4)
    throw std::runtime_error(
        "--lm-head int8 quantises a bf16 lm_head at load; this checkpoint ships it int4");
  const model::FusedLinear& lm_row = desc.lm_head(
      ckpt_int4 ? model::WeightKind::Int4
                : (lm_form == LmHeadForm::Int8 ? model::WeightKind::Int8 : model::WeightKind::Bf16));
  const bool lm_int4 = lm_row.kind == model::WeightKind::Int4;
  const bool lm_int8 = lm_row.kind == model::WeightKind::Int8;
  // Spec 8 §11: a draft vocabulary is a subset of the MTP draft's head (int8 or bf16), at a
  // size the compact GEMV and argmax are compiled for - all knowable before a byte is read.
  if (draft_vocab.size != 0) {
    if (!mtp)
      throw std::runtime_error("--draft-vocab drafts with the MTP head: it needs --mtp");
    if (lm_int4)
      throw std::runtime_error(
          "--draft-vocab gathers rows of an int8 or bf16 lm_head (spec 8 §11); this "
          "checkpoint ships it int4");
    if (std::find(std::begin(kDraftVocabSizes), std::end(kDraftVocabSizes), draft_vocab.size) ==
        std::end(kDraftVocabSizes))
      throw std::runtime_error("--draft-vocab " + std::to_string(draft_vocab.size) +
                               ": the compiled sizes are 32768, 65536 and 131072");
  }

  // embed_tokens is uploaded row-major and verbatim: it is gathered one row per
  // token, so no tiling helps and the mmap is already the canonical layout.
  // Copied, not bound by reference: gcc 13's -Wdangling-reference cannot see
  // that the referent lives in the set's map, not in the temporary name.
  const TensorInfo emb = take(view, set, "embed_tokens.weight");
  if (emb.dtype != "BF16" || emb.shape.size() != 2 || emb.shape[0] != Qwen35::kVocab ||
      emb.shape[1] != desc.hidden)
    throw std::runtime_error("embed_tokens.weight: expected BF16 [" +
                             std::to_string(Qwen35::kVocab) + "][" +
                             std::to_string(desc.hidden) + "], got " + emb.dtype);
  // The model's final pre-lm_head RMSNorm - the one norm that belongs to no
  // layer, so it is not in any LayerDesc's table; described here in the same
  // terms and baked by the same code, into its own allocation at offset 0.
  const model::SmallTensor fnorm_desc{"norm.weight",        desc.hidden,
                                      "BF16",               model::SmallBlock::Norms,
                                      0,                    model::SmallBake::OnePlusWFp32};
  std::vector<uint8_t> fnorm(desc.small_layout().final_norm_bytes);
  bake_small(fnorm_desc, small_src(view, set, fnorm_desc.name, fnorm_desc), fnorm.data(), widen);
  const std::vector<float> rope = rope_table(max_len);

  LoadedModel m{{},
                {},
                l0::Mem(ctx, l0::MemKind::Device, set.bytes(emb)),
                l0::Mem(ctx, l0::MemKind::Device, fnorm.size()),
                l0::Mem(ctx, l0::MemKind::Device, rope.size() * 4),
                {},
                max_len,
                trained,
                nullptr,
                &desc,
                nullptr,
                {}};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  m.report.quant_note = quant_note;
  imm.copy(m.embed.ptr(), set.data(emb), set.bytes(emb));
  m.report.embed_bytes += set.bytes(emb);
  imm.copy(m.final_norm.ptr(), fnorm.data(), fnorm.size());
  m.report.small_bytes += fnorm.size();
  const size_t rope_bytes = rope.size() * 4;
  imm.copy(m.rope.ptr(), rope.data(), rope_bytes);
  m.report.small_bytes += rope_bytes;
  m.report.rope_bytes = rope_bytes;

  const std::vector<model::LayerDesc> layers = desc.layer_descs();
  Staging st;
  const model::GemvShape& ab = desc.shape(model::LinearId::AB);
  size_t max_i4_words = lm_int4 ? int4_words(lm_row.shape) : 0;
  size_t max_i4_scales = lm_int4 ? int4_scale_elems(lm_row.shape) : 0;
  for (const model::LayerDesc& ld : layers)
    for (const model::FusedLinear& fl : ld.linears)
      if (fl.kind == model::WeightKind::Int4) {
        max_i4_words = std::max(max_i4_words, int4_words(fl.shape));
        max_i4_scales = std::max(max_i4_scales, int4_scale_elems(fl.shape));
      }
  st.i4.resize(max_i4_words);
  st.i4_scales.resize(max_i4_scales);
  st.bf_src.resize(size_t(ab.N) * ab.K);
  st.bf_tiled.resize(std::max(size_t(ab.N) * ab.K,
                              lm_row.kind == model::WeightKind::Bf16
                                  ? size_t(lm_row.shape.N) * lm_row.shape.K
                                  : size_t(0)));
  if (lm_int8) {
    st.i8.resize(size_t(lm_row.shape.N) * lm_row.shape.K);
    st.i8_scales.resize(lm_row.shape.N);
  }

  m.layer_small.reserve(layers.size());
  // Spec 15c: a MoE model's FFN is the MoE block - its GateUp / Down table rows (the
  // shared expert) go into the expert blocks' last slot, not into `linears`.
  MoeHost moe_host;
  double moe_s = 0;
  for (const model::LayerDesc& ld : layers) {
    const std::string lp = Qwen35::layer_prefix(ld.index);
    for (const model::FusedLinear& fl : ld.linears) {
      if (desc.is_moe() &&
          (fl.id == model::LinearId::GateUp || fl.id == model::LinearId::Down))
        continue;
      m.linears.emplace(std::make_pair(ld.index, fl.id),
                        load_linear(ctx, imm, set, view, desc, lp, fl, st, m.report));
    }
    m.layer_small.push_back(load_small(ctx, imm, set, view, desc, ld, m.report, widen));
    if (desc.is_moe()) {
      const auto r0 = std::chrono::steady_clock::now();
      m.moe.push_back(load_moe_layer(ctx, imm, set, view, desc, ld.index, moe_host, m.report));
      moe_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();
    }
  }
  m.report.moe_repack_seconds = moe_s;
  if (m.report.moe_bytes != moe_bytes(desc))
    throw std::logic_error("loader: the MoE layers allocated " + std::to_string(m.report.moe_bytes) +
                           " B, not the " + std::to_string(moe_bytes(desc)) +
                           " B loader::moe_bytes() plans");
  m.linears.emplace(std::make_pair(kTopLevel, model::LinearId::LmHead),
                    load_linear(ctx, imm, set, view, desc, "", lm_row, st, m.report));
  // Spec 8 §11: V' and its compact head, gathered from the host copy of the head
  // `load_linear` just made and uploaded - st.i8 / st.i8_scales for int8 (the
  // quantisation), st.bf_tiled for bf16 (the tiling) - so no byte comes back from the
  // device and the compact rows are byte-for-byte the uploaded head's. The selection masks
  // at the model's vocab_used, the bound the main argmax masks at (argmax.cl VOCAB_USED).
  if (draft_vocab.size != 0) {
    const auto d0 = std::chrono::steady_clock::now();
    const model::GemvShape& lm = lm_row.shape;
    // Every size from loader::draft_vocab_bytes - the formula runtime::plan counts with
    // (spec 6 §10: the planner must see these, as they are outside LoadReport::total()).
    const DraftVocabBytes nb = draft_vocab_bytes(draft_vocab.size, lm.K, lm_int8);
    auto dv = std::make_unique<DraftVocab>(DraftVocab{
        {l0::Mem(ctx, l0::MemKind::Device, nb.head), nullptr,
         model::GemvShape{lm.K, draft_vocab.size, 1, 0},
         lm_int8 ? model::WeightKind::Int8 : model::WeightKind::Bf16},
        l0::Mem(ctx, l0::MemKind::Device, nb.ids),
        select_draft_vocab(draft_vocab.added, draft_vocab.eos, draft_vocab.ranked, desc.vocab_used,
                           draft_vocab.size, &m.report.draft_vocab_counts)});
    if (lm_int8) {
      std::vector<int8_t> rows(size_t(draft_vocab.size) * lm.K);
      std::vector<float> scales(draft_vocab.size);
      gather_int8_tiled_rows(st.i8.data(), st.i8_scales.data(), lm.K, lm.N, dv->host_ids.data(),
                             draft_vocab.size, rows.data(), scales.data());
      imm.copy(dv->head.mem.ptr(), rows.data(), nb.head);
      dv->head.scales = std::make_unique<l0::Mem>(upload(ctx, imm, scales.data(), nb.scales));
    } else {
      if (st.bf_tiled.size() < size_t(lm.N) * lm.K)
        throw std::runtime_error("--draft-vocab: the bf16 staging buffer does not hold lm_head");
      std::vector<uint16_t> rows(size_t(draft_vocab.size) * lm.K);
      gather_bf16_tiled_rows(st.bf_tiled.data(), lm.K, lm.N, dv->host_ids.data(),
                             draft_vocab.size, rows.data());
      imm.copy(dv->head.mem.ptr(), rows.data(), nb.head);
    }
    imm.copy(dv->ids.ptr(), dv->host_ids.data(), nb.ids);
    m.report.draft_vocab_bytes = dv->bytes();
    if (m.report.draft_vocab_bytes != nb.total())
      throw std::logic_error("loader: the draft vocabulary allocated " +
                             std::to_string(m.report.draft_vocab_bytes) + " B, not the " +
                             std::to_string(nb.total()) + " B draft_vocab_bytes() plans");
    m.report.draft_vocab_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
    m.draft_vocab = std::move(dv);
  }
  // The MTP head's widening goes into its own Widen: the W cross-check is over
  // the main model's read-per-token bytes, which the head is not part of.
  Widen mtp_widen;
  if (mtp) m.mtp = load_mtp(ctx, imm, set, view, desc, m.report, mtp_widen);

  // Up to five names, so a checkpoint that grew a family of tensors says which
  // family rather than making the reader re-run with a debugger.
  std::string unconsumed;
  for (const auto& [stripped, full] : view.names) {
    (void)full;
    if (view.consumed.count(stripped)) continue;
    if (has_suffix(stripped, ".qzeros")) continue;
    if (has_suffix(stripped, ".g_idx")) continue;
    if (is_kv_scale(stripped)) continue;
    if (m.report.unconsumed++ < 5) unconsumed += (unconsumed.empty() ? "" : ", ") + stripped;
  }
  if (m.report.unconsumed > 5) unconsumed += ", …";

  const LoadReport& r = m.report;
  const double gb = 1e9;
  const std::string fold_note =
      desc.has_parallel_ffn()
          ? " (" + std::to_string(desc.intermediate - desc.parallel_ffn) + " + parallel FFN " +
                std::to_string(desc.parallel_ffn) + ", folded at load)"
          : std::string();
  // Spec 9 §3: the form, and for int8 the host quantisation time.
  char lm_buf[160];
  if (lm_int8)
    std::snprintf(lm_buf, sizeof lm_buf,
                  "int8 per row + fp32 scales, quantised at load from bf16 in %.0f ms",
                  r.lm_head_quant_seconds * 1e3);
  else
    std::snprintf(lm_buf, sizeof lm_buf, "%s, by checkpoint content",
                  lm_int4 ? "int4 g64" : "bf16");
  const std::string lm_desc = lm_buf;
  char mtp_buf[240];
  std::snprintf(mtp_buf, sizeof mtp_buf, "MTP head: %zu tensors, %.3f GB in the checkpoint",
                r.mtp_tensors, r.mtp_checkpoint_bytes / 1e9);
  std::string mtp_line = mtp_buf;
  if (r.mtp_rtn_linears != 0) {   // spec 15e: a MoE head's bf16 experts, int4 g64 RTN at load
    std::snprintf(mtp_buf, sizeof mtp_buf, "; %zu expert linears int4 g64 RTN at load in %.1f s",
                  r.mtp_rtn_linears, r.mtp_rtn_seconds);
    mtp_line += mtp_buf;
  }
  // The two vocabularies print as one line, naming whichever one this
  // checkpoint spoke - a report that always said "dynamic exclusion rules"
  // would be silently wrong about an auto-round config.
  const std::string rules =
      qc.compressed_tensors
          ? "compressed-tensors pack-quantized, actorder " +
                (qc.ct_actorder.empty() ? std::string("none") : qc.ct_actorder) + ", " +
                std::to_string(qc.ct_ignore) + " ignore entries, " +
                std::to_string(scan.ct_linears) + " linears repacked to GPTQ layout" +
                (view.kv_scales != 0 ? ", " + std::to_string(view.kv_scales) +
                                             " fp8 KV scales dropped"
                                       : std::string())
      : qc.dynamic_rule_count != 0
          ? std::to_string(qc.dynamic_rule_count) + " dynamic exclusion rules"
          : std::to_string(qc.extra_excluded) + " extra_config exclusions + " +
                std::to_string(qc.extra_quantised) + " explicit int4";
  // The RoPE table is resident but NOT streamed per token: the decode step reads
  // one position's 2 x 32 floats (~256 B), not the 4.19 MB table. It therefore
  // sits outside the read-per-token figure on both sides of the cross-check,
  // and outside `small` in the printout - but inside total(), because the
  // seven buckets must account for every byte allocated.
  const size_t small_resident = r.small_bytes - rope_bytes;
  // Spec 15c: a MoE layer is read in part - the router, top_k experts and the shared
  // expert (MoeLayerBytes::per_token, derived); the other experts are resident only.
  const size_t moe_per_token =
      desc.is_moe() ? size_t(desc.layers) * moe_layer_bytes(desc).per_token(desc.moe.top_k) : 0;
  const size_t per_token = r.int4_bytes + r.scale_bytes + r.bf16_linear_bytes + r.pad_bytes +
                           r.lm_head_bytes + small_resident + moe_per_token;
  m.report.read_per_token = per_token;
  // doc-03's W measured a bf16 lm_head. A packed one is an itemised term on the
  // expected side, exactly like the padding and the fp32 widening - the check
  // stays at 2%, and the two sides move together or the load fails.
  const double lm_adjust =
      lm_int4   ? double(lm_head_int4_bytes(desc)) - double(lm_head_bf16_bytes(desc))
      : lm_int8 ? double(lm_head_int8_bytes(lm_row.shape.K, lm_row.shape.N)) -
                      double(lm_head_bf16_bytes(desc))
                : 0.0;
  const double expected = desc.doc_w + double(r.pad_bytes) + double(widen.total()) + lm_adjust;
  const bool have_w = desc.doc_w != 0;   // Ornith: none yet (spec 15c, below)
  const double delta = have_w ? (double(per_token) - expected) / expected : 0.0;
  m.report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf(
      "loader: %s\n"
      "  model     %s (%s): %u layers = %u GDN + %u FA, MLP intermediate %u%s\n"
      "  quant     int4 g%u sym desc_act=false (%s), %s\n"
      "  tensors   %zu language-model + %zu top-level; skipped %zu visual, %zu mtp;\n"
      "            dropped %zu qzeros + %zu g_idx (invariants asserted), %zu unconsumed%s%s\n"
      "  scales    %zu subnormal f16 (exact on device and in f16_to_f32; not an error)\n"
      "  linears   %zu fused weights, %zu layers\n"
      "  int4        %13zu B  %7.3f GB\n"
      "  scales      %13zu B  %7.3f GB\n"
      "  bf16_linear %13zu B  %7.3f GB   (a‖b real rows; %.3f GB with padding)\n"
      "  lm_head     %13zu B  %7.3f GB   (%s)\n"
      "  small       %13zu B  %7.3f GB   (per-layer blocks + final norm)\n"
      "  pad         %13zu B  %7.3f GB\n"
      "  read/token  %13zu B  %7.3f GB\n"
      "  embed       %13zu B  %7.3f GB   (resident, gathered - not per-token)\n"
      "  rope        %13zu B  %7.3f GB   (resident, ~256 B per token - not per-token)\n"
      "  mtp         %13zu B  %7.3f GB   (%s)\n"
      "  total       %13zu B  %7.3f GB\n"
      "  W check     %.3f GB vs %.3f GB expected = %.3f doc-03 + %.3f pad + %.6f widen"
      " %+.3f lm_head\n"
      "              widen = %zu B RMSNorm fp32 (1+w) + %zu B GDN fp32  ->  %+.3f%%%s\n"
      "  load        %.1f s\n",
      snap.c_str(), desc.name.c_str(), desc.architecture.c_str(), desc.layers, desc.gdn_layers,
      desc.fa_layers, desc.intermediate, fold_note.c_str(), qc.group_size,
      qc.desc_act_declared ? "declared" : "inferred, 0 g_idx",
      rules.c_str(), view.names.size() - view.top_level, view.top_level,
      view.visual_skipped, view.mtp_skipped, view.qzeros, view.g_idx, r.unconsumed,
      r.unconsumed ? " incl. " : "", unconsumed.c_str(), scan.subnormal_scales, m.linears.size(),
      m.layer_small.size(), r.int4_bytes, r.int4_bytes / gb, r.scale_bytes, r.scale_bytes / gb,
      r.bf16_linear_bytes, r.bf16_linear_bytes / gb, (r.bf16_linear_bytes + r.pad_bytes) / gb,
      r.lm_head_bytes, r.lm_head_bytes / gb, lm_desc.c_str(),
      small_resident, small_resident / gb,
      r.pad_bytes, r.pad_bytes / gb, per_token, per_token / gb, r.embed_bytes, r.embed_bytes / gb,
      rope_bytes, rope_bytes / gb, r.mtp_bytes, r.mtp_bytes / gb,
      mtp ? mtp_line.c_str() : "not loaded; --mtp loads it", r.total(), r.total() / gb, per_token / gb, expected / gb,
      desc.doc_w / gb, r.pad_bytes / gb, widen.total() / gb, lm_adjust / gb, widen.norm, widen.gdn,
      delta * 100.0, have_w ? "" : "   (void: no measured W for this model)", m.report.seconds);
  if (m.draft_vocab) {
    const DraftVocabCounts& c = r.draft_vocab_counts;
    std::printf("  draft vocab %s: %u ids (%u added/EOS + %u ranked + %u lowest), %zu B %.3f GB"
                " - %s + u32 ids, gathered from lm_head in %.0f ms (spec 8 §11; not in total)\n",
                draft_vocab_name(m.draft_vocab->size()).c_str(), m.draft_vocab->size(), c.forced,
                c.ranked, c.lowest, r.draft_vocab_bytes, r.draft_vocab_bytes / gb,
                lm_int8 ? "int8 rows + fp32 scales" : "bf16 rows",
                r.draft_vocab_seconds * 1e3);
  }

  if (desc.is_moe()) {
    const MoeLayerBytes mb = moe_layer_bytes(desc);
    std::printf("  moe       %13zu B  %7.3f GB   (%u layers x [router||gate %zu + %u x (gate||up %zu"
                " + down %zu)], per-expert %s; %.3f GB read per token: top-%u + shared; repacked"
                " in %.1f s; all of it in total, the per-token share in read/token)\n",
                r.moe_bytes, r.moe_bytes / gb, desc.layers, mb.router, mb.blocks,
                mb.gate_up_block, mb.down_block,
                moe_host.fused_gate_up ? "gate_up_proj" : "gate_proj / up_proj",
                moe_per_token / gb, desc.moe.top_k, r.moe_repack_seconds);
  }
  // doc_w 0: no measured W exists for this model yet (Ornith, spec 15a) - nothing to
  // cross-check against, and a derived W would only restate the bytes above.
  if (!have_w) {
    std::printf("  W check   skipped: %s has no measured W yet (model_desc.cc doc_w; the box "
                "measures it, spec 15c)\n",
                desc.name.c_str());
    return m;
  }
  if (std::fabs(delta) > 0.02)
    throw std::runtime_error("resident read-per-token bytes " + std::to_string(per_token) +
                             " differ from docs/03-models.md's W plus itemised padding/widening " +
                             std::to_string(uint64_t(expected)) + " by " +
                             std::to_string(delta * 100.0) + "% (> 2%) - the table above is the breakdown");
  return m;
}

void set_max_len(l0::Context& ctx, LoadedModel& m, uint32_t max_len) {
  check_max_len(*m.desc, max_len, m.trained_max_len);
  const std::vector<float> rope = rope_table(max_len);
  const size_t bytes = rope.size() * 4;
  // The planner's figure for the same table (spec 6 §10) - one formula, checked here.
  if (bytes != Qwen35::rope_table_bytes(max_len))
    throw std::logic_error("loader::set_max_len: the RoPE table is " + std::to_string(bytes) +
                           " B, model::Qwen35::rope_table_bytes says " +
                           std::to_string(Qwen35::rope_table_bytes(max_len)));
  // The new table is uploaded before the old one is freed: a failed allocation leaves
  // the model as it was.
  l0::Mem fresh(ctx, l0::MemKind::Device, bytes);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(fresh.ptr(), rope.data(), bytes);
  m.rope.swap(fresh);   // `fresh` now holds the old table and frees it on return
  m.report.small_bytes = m.report.small_bytes - m.report.rope_bytes + bytes;
  m.report.rope_bytes = bytes;
  m.max_len = max_len;
}

}  // namespace loader

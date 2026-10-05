// Spec 15 §13: the GDN a||b projection in the form the checkpoint ships it (loader/ab.h),
// host-only.
//
// The published Ornith int4 checkpoint (urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ)
// quantised linear_attn.in_proj_a / in_proj_b - int4 g64 sym, qweight [256][32], f16 scales
// [32][32] per layer - where Qwen3.8's and Agnes's exports keep them bf16. The test WRITES a
// one-layer safetensors fixture at Ornith's real a/b shape (K 2048, 32 + 32 columns) in both
// forms, reads it back through loader::SafetensorsSet and LinearSrc::classify exactly as the
// loader does, and checks:
//
//   1. the row choice: an int4 in_proj_a selects ModelDesc::ab(Int4) (K 2048, N 128, S 1,
//      layout 1, pad_n 64); a bf16 one selects the table's AB row ITSELF (the same object as
//      desc.linear(AB): the bf16 a||b path is untouched), on Ornith and on Qwen3.8;
//   2. the int4 repack, layout 1 (the row's) and layout 0: every nibble and every f16 scale
//      of the device tiles is its source column's - a at [0, 32), b at [32, 64) - and every
//      padded column [64, 128) is the int4 zero (nibble 8, scale +0), so the device
//      dequant scale x (q - 8) equals the source's EXACTLY on every weight (fp32), and is
//      exactly 0 on the padding. The scales include negative, subnormal and zero values (the
//      real checkpoint's a/b scales are signed, and it has subnormal ones);
//   3. the prefill copy (pf_ab_proj's bf16 weight): untiled, element (n, k) is
//      rne_bf16((q - 8) x scale) of the source - pf_dequant_slab.cl's arithmetic - and 0 on
//      the padded rows; i.e. the bf16 rounding of the very weights decode multiplies by;
//   4. the refusals: bf16 parts handed to the int4 map, too many columns for the row, an
//      interleaved row that would need padding;
//   5. spec 20 §9: the same a / b as compressed-tensors pack-quantized take the same
//      LinearSrc::classify conversion and give the same device bytes and prefill copy.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "loader/ab.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/model_desc.h"

namespace {

namespace fs = std::filesystem;
using model::LinearId;
using model::WeightKind;

// --- a minimal safetensors writer (one file + its index), as ornith_repack_test's ---------
struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};

void write_checkpoint(const fs::path& dir, const std::map<std::string, Tensor>& ts) {
  fs::create_directories(dir);
  std::string hdr = "{";
  uint64_t off = 0;
  bool first = true;
  for (const auto& [name, t] : ts) {
    hdr += (first ? "" : ",");
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) +
           "]}";
    off += t.bytes.size();
    off = (off + 7) / 8 * 8;
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';
  std::ofstream f(dir / "model-00001-of-00001.safetensors", std::ios::binary);
  const uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  uint64_t pos = 0;
  for (const auto& [name, t] : ts) {
    (void)name;
    f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    pos += t.bytes.size();
    const uint64_t pad = (pos + 7) / 8 * 8 - pos;
    const char zeros[8] = {};
    f.write(zeros, std::streamsize(pad));
    pos += pad;
  }
  std::ofstream ix(dir / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  first = true;
  for (const auto& [name, t] : ts) {
    (void)t;
    ix << (first ? "" : ",") << "\"" << name << "\":\"model-00001-of-00001.safetensors\"";
    first = false;
  }
  ix << "}}";
}

template <class T>
std::vector<uint8_t> as_bytes(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// A GPTQ int4 g64 linear of K x N, as the AutoRound export writes it: qweight [K/8][N],
// signed f16 scales [K/64][N], qzeros 0x77777777, g_idx k / 64.
struct Packed {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;
  std::vector<uint16_t> scales;
  uint32_t nibble(uint32_t k, uint32_t n) const {
    return (qweight[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 15u;
  }
  uint16_t scale(uint32_t k, uint32_t n) const { return scales[size_t(k / 64) * N + n]; }
  float weight(uint32_t k, uint32_t n) const {
    return float(int(nibble(k, n)) - 8) * common::f16_to_f32(scale(k, n));
  }
};
Packed random_packed(uint32_t K, uint32_t N, std::mt19937& rng) {
  Packed p;
  p.K = K;
  p.N = N;
  p.qweight.resize(size_t(K / 8) * N);
  for (uint32_t& w : p.qweight) w = rng();
  p.scales.resize(size_t(K / 64) * N);
  std::uniform_real_distribution<float> sd(-0.02f, 0.02f);   // signed, as the real a/b scales
  for (uint16_t& s : p.scales) s = common::f32_to_f16(sd(rng));
  p.scales[3] = 0x0001;    // the smallest subnormal f16
  p.scales[7] = 0x83FF;    // the largest negative subnormal
  p.scales[11] = 0x0000;   // a zero scale
  return p;
}
void add_packed(std::map<std::string, Tensor>& ts, const std::string& prefix, const Packed& p) {
  ts[prefix + ".qweight"] = {"I32", {p.K / 8, p.N}, as_bytes(p.qweight)};
  ts[prefix + ".scales"] = {"F16", {p.K / 64, p.N}, as_bytes(p.scales)};
  std::vector<uint32_t> qz(size_t(p.K / 64) * (p.N / 8), 0x77777777u);
  ts[prefix + ".qzeros"] = {"I32", {p.K / 64, p.N / 8}, as_bytes(qz)};
  std::vector<int32_t> gi(p.K);
  for (uint32_t k = 0; k < p.K; ++k) gi[k] = int32_t(k / 64);
  ts[prefix + ".g_idx"] = {"I32", {p.K}, as_bytes(gi)};
}

fs::path temp_root() {
  const fs::path r = fs::temp_directory_path() / ("b70_ab_int4_test_" + std::to_string(::getpid()));
  fs::create_directories(r);
  return r;
}

const std::string kA = "model.language_model.layers.0.linear_attn.in_proj_a";
const std::string kB = "model.language_model.layers.0.linear_attn.in_proj_b";

// Decode layout 1 back: the word of (k-octet r, column n) and the scale of (group g, column n).
uint32_t l1_word(const std::vector<uint32_t>& t, uint32_t K, uint32_t r, uint32_t n) {
  const uint32_t G = K / 64, g = r / 8, j = r % 8;
  return t[(size_t(n / 16) * G + g) * 136 + j * 16 + n % 16];
}
uint16_t l1_scale(const std::vector<uint32_t>& t, uint32_t K, uint32_t g, uint32_t n) {
  const uint32_t G = K / 64, l = n % 16;
  const uint32_t w = t[(size_t(n / 16) * G + g) * 136 + 128 + l / 2];
  return uint16_t(l % 2 == 0 ? w & 0xFFFFu : w >> 16);
}

// 2. Every device nibble / scale against its source column, and the padding the int4 zero.
// `word(r, n)` / `scale(g, n)` read the repacked form back.
template <class Word, class Scale>
void check_columns(const model::GemvShape& sh, const Packed& a, const Packed& b, Word word,
                   Scale scale, const char* what) {
  size_t real = 0, pad = 0;
  for (uint32_t n = 0; n < sh.N; ++n) {
    const Packed* src = n < a.N ? &a : (n < a.N + b.N ? &b : nullptr);
    const uint32_t c = n < a.N ? n : n - a.N;
    for (uint32_t r = 0; r < sh.K / 8; ++r) {
      const uint32_t w = word(r, n);
      if (src) {
        CHECK_EQ(w, src->qweight[size_t(r) * src->N + c]);
      } else {
        CHECK_EQ(w, loader::kInt4ZeroWord);
      }
    }
    for (uint32_t g = 0; g < sh.K / 64; ++g) {
      const uint16_t s = scale(g, n);
      CHECK_EQ(s, src ? src->scales[size_t(g) * src->N + c] : uint16_t(0));
    }
    // The device dequant (gemv.cl: float(q - 8) * float(scale), fp32) equals the source's
    // exactly, and is exactly 0 on the padding.
    for (uint32_t k = 0; k < sh.K; ++k) {
      const int q = int((word(k / 8, n) >> (4 * (k % 8))) & 15u);
      const float dev = float(q - 8) * common::f16_to_f32(scale(k / 64, n));
      if (src) {
        CHECK(dev == src->weight(k, c));
        ++real;
      } else {
        CHECK(dev == 0.0f);
        ++pad;
      }
    }
  }
  std::printf("  %s: %zu real weights exact, %zu padded weights exactly 0\n", what, real, pad);
}

void check_int4(const fs::path& root) {
  std::mt19937 rng(20261005);
  const model::ModelDesc& o = model::ornith();
  const uint32_t H = o.hidden, V = o.gdn_v_heads;   // 2048, 32
  const Packed a = random_packed(H, V, rng), b = random_packed(H, V, rng);
  std::map<std::string, Tensor> ts;
  add_packed(ts, kA, a);
  add_packed(ts, kB, b);
  const fs::path dir = root / "int4";
  write_checkpoint(dir, ts);
  const loader::SafetensorsSet set(dir.string() + "/");
  // The loader's own scans pass on it (qzeros, g_idx identity, finite scales).
  const loader::QuantScan scan = loader::assert_quant_invariants(set);
  CHECK_EQ(scan.g_idx_tensors, size_t(2));
  CHECK(scan.subnormal_scales >= 4);

  // 1. the row choice
  const loader::LinearSrc sa = loader::LinearSrc::classify(set, kA);
  const loader::LinearSrc sb = loader::LinearSrc::classify(set, kB);
  CHECK(sa.kind == loader::WKind::Int4 && sb.kind == loader::WKind::Int4);
  CHECK_EQ(sa.K, H);
  CHECK_EQ(sa.N, V);
  const model::FusedLinear& row = loader::ab_row_for(o, sa.kind);
  CHECK(&row == &o.ab(WeightKind::Int4));
  CHECK(row.kind == WeightKind::Int4);
  CHECK_EQ(row.shape.K, H);
  CHECK_EQ(row.shape.N, uint32_t(128));
  CHECK_EQ(row.shape.S, uint32_t(1));
  CHECK_EQ(row.shape.layout, uint32_t(1));
  CHECK_EQ(row.pad_n, 2 * V);

  // 2. the repack, layout 1 (the row's) and layout 0, through the loader's column map.
  loader::Int4Pad pad;
  const std::vector<common::ColSource> cols = loader::int4_row_cols(row, {sa, sb}, pad, "AB");
  CHECK_EQ(cols.size(), size_t(128));
  CHECK_EQ(pad.N, uint32_t(64));
  const model::GemvShape& sh = row.shape;
  std::vector<uint32_t> l1(size_t(sh.N / 16) * (sh.K / 64) * 136);
  common::repack_int4_layout1_cols(sh.K, sh.N, cols, l1.data());
  check_columns(
      sh, a, b, [&](uint32_t r, uint32_t n) { return l1_word(l1, sh.K, r, n); },
      [&](uint32_t g, uint32_t n) { return l1_scale(l1, sh.K, g, n); }, "layout 1");
  std::vector<uint32_t> l0(size_t(sh.K / 8) * sh.N);
  std::vector<uint16_t> l0s(size_t(sh.K / 64) * sh.N);
  common::repack_int4_layout0_cols(sh.K, sh.N, cols, l0.data(), l0s.data());
  check_columns(
      sh, a, b, [&](uint32_t r, uint32_t n) { return l0[size_t(r) * sh.N + n]; },
      [&](uint32_t g, uint32_t n) { return l0s[size_t(g) * sh.N + n]; }, "layout 0");

  // 3. the prefill copy, untiled (common::repack_bf16_tiled's index).
  std::vector<uint16_t> tiled(size_t(sh.N) * sh.K);
  loader::dequant_int4_bf16_tiled(sh, cols, tiled.data());
  const uint32_t K8 = sh.K / 8;
  size_t nonzero = 0;
  for (uint32_t n = 0; n < sh.N; ++n)
    for (uint32_t k = 0; k < sh.K; ++k) {
      const uint16_t got = tiled[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16];
      if (n < 2 * V) {
        const Packed& p = n < V ? a : b;
        const uint32_t c = n < V ? n : n - V;
        CHECK_EQ(got, common::f32_to_bf16(p.weight(k, c)));   // rne_bf16 of the exact weight
        nonzero += got != 0 && got != 0x8000;
      } else {
        CHECK_EQ(got, uint16_t(0));
      }
    }
  CHECK(nonzero > size_t(sh.K) * 2 * V / 2);   // the fixture is not degenerate
  std::printf("  prefill copy: 64 x %u rne_bf16 weights equal, 64 zero rows\n", sh.K);

  // 5. Spec 20 §9: the same a / b written as compressed-tensors pack-quantized (weight_packed
  // [N][K/8], weight_scale [N][K/64], weight_shape) go through the same classify - the one
  // conversion point - and repack to the same device bytes and the same prefill copy.
  std::map<std::string, Tensor> ct;
  for (const auto& [prefix, p] : {std::pair<std::string, const Packed*>{kA, &a}, {kB, &b}}) {
    std::vector<uint32_t> wp(size_t(p->N) * (p->K / 8));
    for (uint32_t r = 0; r < p->K / 8; ++r)
      for (uint32_t n = 0; n < p->N; ++n) wp[size_t(n) * (p->K / 8) + r] = p->qweight[size_t(r) * p->N + n];
    std::vector<uint16_t> ws(size_t(p->N) * (p->K / 64));
    for (uint32_t g = 0; g < p->K / 64; ++g)
      for (uint32_t n = 0; n < p->N; ++n) ws[size_t(n) * (p->K / 64) + g] = p->scales[size_t(g) * p->N + n];
    ct[prefix + ".weight_packed"] = {"I32", {p->N, p->K / 8}, as_bytes(wp)};
    ct[prefix + ".weight_scale"] = {"F16", {p->N, p->K / 64}, as_bytes(ws)};
    ct[prefix + ".weight_shape"] = {"I64", {2}, as_bytes(std::vector<int64_t>{p->N, p->K})};
  }
  const fs::path ct_dir = root / "ct";
  write_checkpoint(ct_dir, ct);
  const loader::SafetensorsSet ct_set(ct_dir.string() + "/");
  const loader::LinearSrc ca = loader::LinearSrc::classify(ct_set, kA);
  const loader::LinearSrc cb = loader::LinearSrc::classify(ct_set, kB);
  CHECK(loader::LinearSrc::kind_of(ct_set, kA) == loader::WKind::Int4);
  CHECK(ca.packing == loader::Int4Packing::CompressedTensors);
  CHECK(&loader::ab_row_for(o, ca.kind) == &row);
  loader::Int4Pad ct_pad;
  const std::vector<common::ColSource> ct_cols = loader::int4_row_cols(row, {ca, cb}, ct_pad, "AB");
  std::vector<uint32_t> ct_l1(l1.size());
  common::repack_int4_layout1_cols(sh.K, sh.N, ct_cols, ct_l1.data());
  CHECK(ct_l1 == l1);
  std::vector<uint16_t> ct_tiled(tiled.size());
  loader::dequant_int4_bf16_tiled(sh, ct_cols, ct_tiled.data());
  CHECK(ct_tiled == tiled);
  std::printf("  compressed-tensors a / b: the same layout-1 bytes and prefill copy\n");
}

void check_bf16(const fs::path& root) {
  std::mt19937 rng(7);
  std::map<std::string, Tensor> ts;
  for (const std::string& n : {kA, kB}) {
    std::vector<uint16_t> w(size_t(32) * 2048);
    for (uint16_t& v : w) v = common::f32_to_bf16(std::normal_distribution<float>(0.f, 0.02f)(rng));
    ts[n + ".weight"] = {"BF16", {32, 2048}, as_bytes(w)};
  }
  const fs::path dir = root / "bf16";
  write_checkpoint(dir, ts);
  const loader::SafetensorsSet set(dir.string() + "/");
  const loader::LinearSrc sa = loader::LinearSrc::classify(set, kA);
  CHECK(sa.kind == loader::WKind::Bf16);
  // 1. the bf16 row is the table's own AB row - the object load_linear always took.
  for (const model::ModelDesc* d : {&model::ornith(), &model::qwen38(), &model::agnes()}) {
    const model::FusedLinear& row = loader::ab_row_for(*d, sa.kind);
    CHECK(&row == &d->linear(LinearId::AB));
    CHECK(row.kind == WeightKind::Bf16);
    CHECK_EQ(row.shape.N, uint32_t(128));
    CHECK_EQ(row.shape.S, uint32_t(1));
    CHECK_EQ(row.shape.layout, uint32_t(0));
    CHECK_EQ(row.pad_n, d->gdn_ab_n());
  }
  CHECK_EQ(model::qwen38().linear(LinearId::AB).shape.K, uint32_t(5120));
  CHECK_EQ(model::qwen38().linear(LinearId::AB).pad_n, uint32_t(96));
  // 4. bf16 parts are not the int4 map's.
  bool threw = false;
  try {
    loader::Int4Pad pad;
    (void)loader::int4_row_cols(model::ornith().ab(WeightKind::Int4),
                                {sa, loader::LinearSrc::classify(set, kB)}, pad, "AB");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("  bf16 a||b: the table row itself on Ornith, Qwen3.8 and Agnes\n");
}

void check_refusals() {
  std::mt19937 rng(3);
  const Packed p = random_packed(128, 96, rng);
  loader::LinearSrc s;
  s.kind = loader::WKind::Int4;
  s.K = 128;
  s.N = 96;
  s.qweight = p.qweight.data();
  s.scales = p.scales.data();
  s.name = "wide";
  model::FusedLinear row{LinearId::AB, {128, 128, 1, 1}, WeightKind::Int4, model::Fuse::Concat,
                         {"a", "b"}, 128, model::Fold::None, {}};
  auto throws = [](auto&& f) {
    try {
      f();
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  loader::Int4Pad pad;
  CHECK(throws([&] { (void)loader::int4_row_cols(row, {s, s}, pad, "AB"); }));   // 192 > 128
  row.fuse = model::Fuse::Interleave16;
  row.shape.N = 256;
  CHECK(throws([&] { (void)loader::int4_row_cols(row, {s, s}, pad, "AB"); }));   // would pad
  row.fuse = model::Fuse::Concat;
  row.shape.N = 192;
  CHECK_EQ(loader::int4_row_cols(row, {s, s}, pad, "AB").size(), size_t(192));   // exact fit
  std::printf("  refusals: too many columns, a padded interleave; an exact fit takes no pad\n");
}

}  // namespace

int main() {
  const fs::path root = temp_root();
  check_int4(root);
  check_bf16(root);
  check_refusals();
  for (const char* sub : {"int4", "ct", "bf16"}) {   // file by file, as ornith_repack_test
    fs::remove(root / sub / "model-00001-of-00001.safetensors");
    fs::remove(root / sub / "model.safetensors.index.json");
    fs::remove(root / sub);
  }
  fs::remove(root);
  std::puts("ab_int4_test OK");
  return 0;
}

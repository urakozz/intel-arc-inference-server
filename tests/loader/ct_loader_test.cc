// Spec 20 §9: the compressed-tensors symmetric `pack-quantized` loader (loader/quant.h),
// host only. docs/13-loader.md "compressed-tensors symmetric checkpoints".
//
// The test WRITES its fixtures - small checkpoints with one linear quantised from known
// float weights - reads them back through loader::SafetensorsSet and
// LinearSrc::classify exactly as the loaders do, and checks:
//
//   1. exactness, g64 and g128, f16 and bf16 scales: every weight the converted
//      (GPTQ-layout) linear dequantises to - by common::Int4Gptq::at, the kernels' host
//      reference, and by decoding the layout-1 tiles the MoE / K2 / layout-1 GEMV read -
//      equals the format definition's dequant (u - 8) x s of the compressed-tensors
//      tensors, which equals the quantisation q x s of the source floats; and the
//      converted qweight / scales are BYTE-equal to a GPTQ checkpoint of the same q, s;
//   2. the startup note (operator requirement 2026-10-05): present for a compressed-
//      tensors checkpoint with the group found and the linear count, absent for a GPTQ /
//      AutoRound g64 one;
//   3. refusals: asymmetric (config and a shipped weight_zero_point), group 32, 8-bit,
//      a format other than pack-quantized, an actorder permutation (a non-identity
//      weight_g_idx; "group" with no g_idx), a bf16 scale f16 cannot hold, the label
//      disagreeing with the tensors;
//   4. real configs (argv[1], tests/loader/ct/): RedHatAI/Qwen3.8-27B-INT4 and
//      RedHatAI/Qwen3-32B-quantized.w4a16 accepted; Padakovec/Kolibri-1-W4A16-GPTQ and
//      RedHatAI/Qwen3-8B-quantized.w4a16 (asymmetric) and
//      halt95/Qwen3.8-Flash-Next-W4A16-Merlin (an int8 channel group) refused by field.
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "common/json.h"
#include "common/kv8.h"
#include "common/repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"

namespace {

namespace fs = std::filesystem;

struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};
using Tensors = std::map<std::string, Tensor>;

const char* kShard = "model-00001-of-00001.safetensors";

void write_checkpoint(const fs::path& dir, const Tensors& ts) {
  fs::create_directories(dir);
  std::string hdr = "{";
  uint64_t off = 0;
  bool first = true;
  for (const auto& [name, t] : ts) {
    hdr += (first ? "" : ",");
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + t.bytes.size()) + "]}";
    off = (off + t.bytes.size() + 7) / 8 * 8;
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';
  std::ofstream f(dir / kShard, std::ios::binary);
  const uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  for (const auto& [name, t] : ts) {
    (void)name;
    f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    const size_t pad = (t.bytes.size() + 7) / 8 * 8 - t.bytes.size();
    const char zeros[8] = {};
    f.write(zeros, std::streamsize(pad));
  }
  std::ofstream ix(dir / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  first = true;
  for (const auto& [name, t] : ts) {
    (void)t;
    ix << (first ? "" : ",") << "\"" << name << "\":\"" << kShard << "\"";
    first = false;
  }
  ix << "}}";
}

void remove_checkpoint(const fs::path& dir) {
  fs::remove(dir / kShard);
  fs::remove(dir / "model.safetensors.index.json");
  fs::remove(dir);
}

template <class T>
std::vector<uint8_t> as_bytes(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// One linear quantised from known floats: the source of truth both forms are written from.
struct Quantised {
  uint32_t K = 0, N = 0, group = 64;
  bool bf16 = false;               // the scale dtype compressed-tensors ships
  std::vector<int8_t> q;           // [N][K], in [-8, 7]
  std::vector<uint16_t> s;         // [N][K/group], f16 or bf16 bits
  float scale(uint32_t n, uint32_t k) const {
    const uint16_t b = s[size_t(n) * (K / group) + k / group];
    return bf16 ? common::bf16_to_f32(b) : common::f16_to_f32(b);
  }
  float value(uint32_t n, uint32_t k) const { return float(q[size_t(n) * K + k]) * scale(n, k); }
};

// Symmetric round-to-nearest of random floats w [N][K]: s = amax / 7.5, rounded to the
// scale dtype, round to nearest even, f16 subnormals kept (bf16 values at or above 2^-17
// are exactly f16, as real scales are); q = clamp(rint(w / s), -8, 7). One group is
// pinned to a scale that is subnormal in f16 (~2.6e-5: exact in both dtypes).
Quantised quantise(uint32_t K, uint32_t N, uint32_t group, bool bf16, uint32_t seed) {
  Quantised r;
  r.K = K;
  r.N = N;
  r.group = group;
  r.bf16 = bf16;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> wd(-1.f, 1.f), ad(0.05f, 2.f);
  r.q.resize(size_t(N) * K);
  r.s.resize(size_t(N) * (K / group));
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t g = 0; g < K / group; ++g) {
      const float amp = (n == 3 && g == 0) ? 2e-4f : ad(rng);
      std::vector<float> w(group);
      float amax = 0;
      for (float& x : w) {
        x = amp * wd(rng);
        amax = std::fmax(amax, std::fabs(x));
      }
      const uint16_t sb =
          bf16 ? common::f32_to_bf16(amax / 7.5f) : common::kv8::f32_to_f16_rne(amax / 7.5f);
      r.s[size_t(n) * (K / group) + g] = sb;
      const float sf = bf16 ? common::bf16_to_f32(sb) : common::f16_to_f32(sb);
      for (uint32_t j = 0; j < group; ++j) {
        float qv = std::rint(w[j] / sf);
        qv = std::fmin(std::fmax(qv, -8.f), 7.f);
        r.q[size_t(n) * K + g * group + j] = int8_t(qv);
      }
    }
  return r;
}

// compressed-tensors pack-quantized: weight_packed int32 [N][K/8], 8 consecutive k per
// word, k = 8r in the low nibble, each as q + 8; weight_scale [N][K/g]; weight_shape [N, K].
void add_ct(Tensors& ts, const std::string& prefix, const Quantised& w) {
  std::vector<uint32_t> packed(size_t(w.N) * (w.K / 8), 0);
  for (uint32_t n = 0; n < w.N; ++n)
    for (uint32_t k = 0; k < w.K; ++k)
      packed[size_t(n) * (w.K / 8) + k / 8] |=
          uint32_t(w.q[size_t(n) * w.K + k] + 8) << (4 * (k % 8));
  ts[prefix + ".weight_packed"] = {"I32", {w.N, w.K / 8}, as_bytes(packed)};
  ts[prefix + ".weight_scale"] = {w.bf16 ? "BF16" : "F16", {w.N, w.K / w.group}, as_bytes(w.s)};
  ts[prefix + ".weight_shape"] = {"I64", {2}, as_bytes(std::vector<int64_t>{w.N, w.K})};
}

// The same q, s as a GPTQ v1 linear: qweight [K/8][N], f16 scales [K/g][N], qzeros 0x77777777.
void add_gptq(Tensors& ts, const std::string& prefix, const Quantised& w) {
  std::vector<uint32_t> qw(size_t(w.K / 8) * w.N, 0);
  for (uint32_t n = 0; n < w.N; ++n)
    for (uint32_t k = 0; k < w.K; ++k)
      qw[size_t(k / 8) * w.N + n] |= uint32_t(w.q[size_t(n) * w.K + k] + 8) << (4 * (k % 8));
  std::vector<uint16_t> sc(size_t(w.K / w.group) * w.N);
  for (uint32_t n = 0; n < w.N; ++n)
    for (uint32_t g = 0; g < w.K / w.group; ++g) {
      const uint16_t b = w.s[size_t(n) * (w.K / w.group) + g];
      sc[size_t(g) * w.N + n] = w.bf16 ? common::kv8::f32_to_f16_rne(common::bf16_to_f32(b)) : b;
    }
  ts[prefix + ".qweight"] = {"I32", {w.K / 8, w.N}, as_bytes(qw)};
  ts[prefix + ".scales"] = {"F16", {w.K / w.group, w.N}, as_bytes(sc)};
  std::vector<uint32_t> qz(size_t(w.K / w.group) * (w.N / 8), 0x77777777u);
  ts[prefix + ".qzeros"] = {"I32", {w.K / w.group, w.N / 8}, as_bytes(qz)};
}

// The format definition's dequant, read straight off the compressed-tensors tensors.
float ct_dequant(const Tensors& ts, const std::string& prefix, uint32_t K, uint32_t group,
                 bool bf16, uint32_t n, uint32_t k) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(ts.at(prefix + ".weight_packed").bytes.data());
  const uint16_t* s = reinterpret_cast<const uint16_t*>(ts.at(prefix + ".weight_scale").bytes.data());
  const int u = int((p[size_t(n) * (K / 8) + k / 8] >> (4 * (k % 8))) & 15u);
  const uint16_t sb = s[size_t(n) * (K / group) + k / group];
  return float(u - 8) * (bf16 ? common::bf16_to_f32(sb) : common::f16_to_f32(sb));
}

template <class F>
bool throws_with(F&& f, const std::string& needle) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) return true;
    std::fprintf(stderr, "threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  std::fprintf(stderr, "did not throw (wanted '%s')\n", needle.c_str());
  return false;
}

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

// A config.json around one compressed-tensors weights scheme, with `edit` applied as a
// raw-text substitution of the default weights object.
common::json::Value ct_config(const std::string& weights, const std::string& extra = "") {
  return common::json::parse(
      R"({"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized",)"
      R"("quantization_status":"compressed","ignore":["lm_head"],)" + extra +
      R"("config_groups":{"group_0":{"targets":["Linear"],"input_activations":null,)"
      R"("output_activations":null,"weights":)" + weights + "}}}}");
}
const std::string kSym128 =
    R"({"num_bits":4,"type":"int","symmetric":true,"group_size":128,"strategy":"group",)"
    R"("actorder":"static","dynamic":false,"block_structure":null})";

std::string with(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  CHECK(at != std::string::npos);
  return s.replace(at, from.size(), to);
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 1);
  const std::string cfg_dir = argv[1];
  const fs::path root = fs::temp_directory_path() / ("ct_loader_test_" + std::to_string(getpid()));
  const uint32_t K = 256, N = 48;

  // ---- 1. exactness, g64 / g128 x f16 / bf16 ------------------------------------------
  for (uint32_t group : {64u, 128u})
    for (bool bf16 : {false, true}) {
      const Quantised w = quantise(K, N, group, bf16, 7 + group + (bf16 ? 1 : 0));
      Tensors ct, gq;
      add_ct(ct, "lin", w);
      add_gptq(gq, "lin", w);
      const fs::path dc = root / "ct", dg = root / "gptq";
      write_checkpoint(dc, ct);
      write_checkpoint(dg, gq);
      {
        loader::SafetensorsSet sc(dc.string() + "/"), sg(dg.string() + "/");
        const loader::QuantScan scan = loader::assert_quant_invariants(sc);
        CHECK_EQ(scan.ct_linears, size_t(1));
        CHECK_EQ(scan.gptq_linears, size_t(0));
        CHECK_EQ(scan.ct_group_mask, group == 64 ? 1u : 2u);
        CHECK_EQ(scan.ct_bf16_scales, bf16);
        CHECK(scan.subnormal_scales >= 1);   // the pinned group, counted not refused
        CHECK(loader::LinearSrc::kind_of(sc, "lin") == loader::WKind::Int4);
        const loader::LinearSrc c = loader::LinearSrc::classify(sc, "lin");
        const loader::LinearSrc g = loader::LinearSrc::classify(sg, "lin");
        CHECK(c.kind == loader::WKind::Int4);
        CHECK(c.packing == loader::Int4Packing::CompressedTensors);
        CHECK(g.packing == loader::Int4Packing::Gptq);
        CHECK_EQ(c.K, K);
        CHECK_EQ(c.N, N);
        CHECK_EQ(c.group, group);
        CHECK(c.owned_qweight && c.qweight == c.owned_qweight->data());
        CHECK(c.expanded_scales && c.scales == c.expanded_scales->data());
        CHECK_EQ(c.suffixes().size(), size_t(3));
        CHECK(c.suffixes()[0] == ".weight_packed" && c.suffixes()[2] == ".weight_shape");
        // byte-equal to the GPTQ checkpoint of the same q, s
        CHECK(std::memcmp(c.qweight, g.qweight, size_t(K / 8) * N * 4) == 0);
        CHECK(std::memcmp(c.scales, g.scales, size_t(K / 64) * N * 2) == 0);
        // every weight: the kernels' host dequant == the format's dequant == q x s
        common::Int4Gptq h;
        h.K = K;
        h.N = N;
        h.qweight.assign(c.qweight, c.qweight + size_t(K / 8) * N);
        h.scales.assign(c.scales, c.scales + size_t(K / 64) * N);
        const std::vector<uint32_t> tiles = h.tiled();   // layout 1 (MoE, K2 experts)
        for (uint32_t n = 0; n < N; ++n)
          for (uint32_t k = 0; k < K; ++k) {
            const float want = w.value(n, k);
            CHECK_EQ(ct_dequant(ct, "lin", K, group, bf16, n, k), want);
            CHECK_EQ(h.at(k, n), want);
            const uint32_t* tile =
                tiles.data() + (size_t(n / 16) * (K / 64) + k / 64) * common::Int4Gptq::kTileU32;
            const uint32_t word = tile[((k % 64) / 8) * 16 + n % 16];
            const uint32_t sw = tile[128 + (n % 16) / 2];
            const uint16_t s16 = uint16_t(n % 2 == 0 ? sw & 0xFFFFu : sw >> 16);
            const int u = int((word >> (4 * (k % 8))) & 15u);
            CHECK_EQ(float(u - 8) * common::f16_to_f32(s16), want);
          }
        // a copy keeps both owned buffers alive
        loader::LinearSrc copy = c;
        CHECK_EQ(copy.qweight, c.qweight);
        // ---- 2. the note: present for compressed-tensors, absent for GPTQ ------------
        const std::string note = loader::ct_conversion_note(scan);
        CHECK(note.find("converting compressed-tensors pack-quantized") != std::string::npos);
        CHECK(note.find("1 linears") != std::string::npos);
        CHECK(note.find("weight_packed [N][K/8] -> our qweight [K/8][N]") != std::string::npos);
        CHECK(note.find("weight_scale [N][K/g]") != std::string::npos);
        CHECK(note.find(group == 128 ? "g128 -> g64 scales expanded" : "g64, no expansion") !=
              std::string::npos);
        CHECK(note.find("expect reduced quality compared to our recommended format: AutoRound "
                        "GPTQ W4A16 g64 symmetric") != std::string::npos);
        CHECK_EQ(note.find("bf16") != std::string::npos, bf16);
        const loader::QuantScan gscan = loader::assert_quant_invariants(sg);
        CHECK_EQ(gscan.gptq_linears, size_t(1));
        CHECK(loader::ct_conversion_note(gscan).empty());
      }
      remove_checkpoint(dc);
      remove_checkpoint(dg);
      std::printf("  g%u %s scales: exact (%u x %u weights), byte-equal to GPTQ, note ok\n", group,
                  bf16 ? "bf16" : "f16", N, K);
    }
  {
    // A checkpoint mixing g64 and g128 linears says both.
    loader::QuantScan s;
    s.ct_linears = 3;
    s.ct_group_mask = 3;
    CHECK(loader::ct_conversion_note(s).find("g64 and g128") != std::string::npos);
  }

  // A native AutoRound g64 checkpoint (the recommended format) prints no note.
  {
    Tensors ts;
    add_gptq(ts, "m.lin", quantise(K, N, 64, false, 99));
    const fs::path d = root / "autoround";
    write_checkpoint(d, ts);
    {
      loader::SafetensorsSet set(d.string() + "/");
      const loader::QuantScan scan = loader::assert_quant_invariants(set);
      CHECK(loader::ct_conversion_note(scan).empty());
      const loader::QuantConfig qc = loader::QuantConfig::parse(common::json::parse(
          R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,)"
          R"("quant_method":"auto-round","packing_format":"auto_round:auto_gptq"}})"));
      loader::check_quant_scan(qc, scan);   // label and bytes agree
      // ...and a compressed-tensors label over these GPTQ bytes is refused.
      CHECK(throws_with([&] { loader::check_quant_scan(loader::QuantConfig::parse(ct_config(kSym128)), scan); },
                        "the label and the bytes disagree"));
    }
    remove_checkpoint(d);
  }

  // ---- 3. tensor refusals ----------------------------------------------------------------
  auto refused = [&](const std::function<void(Tensors&)>& edit, const std::string& needle,
                     const char* what, bool by_scan) {
    Tensors ts;
    add_ct(ts, "lin", quantise(K, N, 128, true, 5));
    edit(ts);
    const fs::path d = root / "bad";
    write_checkpoint(d, ts);
    bool ok;
    {
      loader::SafetensorsSet set(d.string() + "/");
      ok = throws_with([&] { loader::LinearSrc::classify(set, "lin"); }, needle);
      if (by_scan) ok = ok && throws_with([&] { loader::assert_quant_invariants(set); }, needle);
    }
    remove_checkpoint(d);
    CHECK(ok);
    std::printf("  refused: %s\n", what);
  };
  refused([](Tensors& t) {
            t["lin.weight_zero_point"] = {"I32", {N / 8, K / 128}, std::vector<uint8_t>(N / 8 * K / 128 * 4, 0)};
          },
          "weight_zero_point", "a shipped weight_zero_point (asymmetric)", true);
  refused([](Tensors& t) {
            std::vector<int32_t> gi(K);
            for (uint32_t k = 0; k < K; ++k) gi[k] = int32_t(k / 128);
            std::swap(gi[3], gi[200]);   // an activation-order permutation
            t["lin.weight_g_idx"] = {"I32", {K}, as_bytes(gi)};
          },
          "lin.weight_g_idx[3]", "a non-identity weight_g_idx (actorder permutation)", true);
  refused([](Tensors& t) {
            uint16_t* s = reinterpret_cast<uint16_t*>(t["lin.weight_scale"].bytes.data());
            s[5] = 0x0DA2u;   // bf16 ~1e-30: below f16's smallest subnormal
          },
          "lin.weight_scale[5]", "a bf16 scale f16 cannot hold exactly", true);
  refused([](Tensors& t) {
            uint16_t* s = reinterpret_cast<uint16_t*>(t["lin.weight_scale"].bytes.data());
            s[7] = 0x7FC0u;   // bf16 NaN
          },
          "NaN", "a NaN scale", true);
  refused([](Tensors& t) {
            t["lin.weight_scale"].shape = {N, K / 32};   // g32: same bytes, other shape
            t["lin.weight_scale"].bytes.resize(size_t(N) * (K / 32) * 2, 0x3C);
          },
          "neither g64", "group 32 scales", true);
  refused([](Tensors& t) { t["lin.weight_shape"].bytes = as_bytes(std::vector<int64_t>{N, K - 8}); },
          "weight_shape", "a weight_shape that is not the pack's", false);
  refused([](Tensors& t) { t.erase("lin.weight_shape"); }, "without weight_shape",
          "a missing weight_shape", false);
  {
    // An identity weight_g_idx is accepted and consumed.
    Tensors ts;
    add_ct(ts, "lin", quantise(K, N, 128, false, 6));
    std::vector<int32_t> gi(K);
    for (uint32_t k = 0; k < K; ++k) gi[k] = int32_t(k / 128);
    ts["lin.weight_g_idx"] = {"I32", {K}, as_bytes(gi)};
    const fs::path d = root / "gidx";
    write_checkpoint(d, ts);
    {
      loader::SafetensorsSet set(d.string() + "/");
      const loader::QuantScan scan = loader::assert_quant_invariants(set);
      CHECK_EQ(scan.ct_g_idx_tensors, size_t(1));
      const loader::LinearSrc s = loader::LinearSrc::classify(set, "lin");
      CHECK(s.ct_g_idx);
      CHECK_EQ(s.suffixes().back(), std::string(".weight_g_idx"));
      // actorder "group" is satisfied by it...
      const loader::QuantConfig qg =
          loader::QuantConfig::parse(ct_config(with(kSym128, "\"static\"", "\"group\"")));
      CHECK(qg.ct_actorder_group);
      loader::check_quant_scan(qg, scan);
      // ...and refused without it.
      loader::QuantScan none = scan;
      none.ct_g_idx_tensors = 0;
      CHECK(throws_with([&] { loader::check_quant_scan(qg, none); }, "did not write down"));
    }
    remove_checkpoint(d);
  }
  CHECK(loader::LinearSrc::marker_suffixes().size() == 3);

  // ---- 3b. config refusals, each naming the field --------------------------------------
  auto cfg_refused = [](const common::json::Value& cfg, const std::string& needle, const char* what) {
    CHECK(throws_with([&] { loader::QuantConfig::parse(cfg); }, needle));
    std::printf("  config refused: %s\n", what);
  };
  {
    const loader::QuantConfig q = loader::QuantConfig::parse(ct_config(kSym128));
    CHECK(q.compressed_tensors && q.bits == 4 && q.sym && !q.desc_act && q.desc_act_declared);
    CHECK_EQ(q.group_size, 128u);
    CHECK_EQ(q.ct_actorder, std::string("static"));
    CHECK(!q.ct_actorder_group);
    CHECK_EQ(q.ct_ignore, size_t(1));
    CHECK_EQ(loader::QuantConfig::parse(ct_config(with(kSym128, "128", "64"))).group_size, 64u);
    CHECK(!loader::QuantConfig::parse(ct_config(with(kSym128, "\"static\"", "null"))).ct_actorder_group);
  }
  cfg_refused(ct_config(with(kSym128, "\"symmetric\":true", "\"symmetric\":false")),
              "weights.symmetric = false", "asymmetric");
  cfg_refused(ct_config(with(kSym128, "128", "32")), "weights.group_size = 32", "group 32");
  cfg_refused(ct_config(with(kSym128, "\"num_bits\":4", "\"num_bits\":8")), "weights.num_bits = 8",
              "8-bit");
  cfg_refused(ct_config(with(kSym128, "\"group\",", "\"channel\",")), "weights.strategy = \"channel\"",
              "strategy channel");
  cfg_refused(ct_config(with(kSym128, "\"int\"", "\"float\"")), "weights.type = \"float\"",
              "float weights");
  cfg_refused(ct_config(with(kSym128, "\"static\"", "\"rotated\"")), "weights.actorder",
              "an unknown actorder");
  cfg_refused(common::json::parse(with(
                  R"({"quantization_config":{"quant_method":"compressed-tensors","format":"FMT",)"
                  R"("config_groups":{"g":{"weights":{}}}}})",
                  "FMT", "int-quantized")),
              "format = \"int-quantized\"", "format int-quantized");
  cfg_refused(ct_config(kSym128, R"("transform_config":{"config_groups":{"u":{}}},)"),
              "transform_config", "a rotation transform");
  cfg_refused(ct_config(kSym128, R"("quantization_status":"frozen",)"), "quantization_status",
              "weights not compressed");
  cfg_refused(common::json::parse(
                  R"({"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized",)"
                  R"("config_groups":{"group_0":{"input_activations":{"num_bits":8},"weights":)" +
                  kSym128 + "}}}}"),
              "group_0.input_activations", "W4A8 activations");

  // ---- 4. real configs -------------------------------------------------------------------
  {
    const loader::QuantConfig q =
        loader::QuantConfig::parse(read_json(cfg_dir + "/redhatai-qwen3.8-27b-int4.config.json"));
    CHECK(q.compressed_tensors);
    CHECK_EQ(q.group_size, 128u);
    CHECK_EQ(q.ct_actorder, std::string("static"));
    CHECK(!q.ct_actorder_group);
    CHECK(q.ct_kv_cache_scheme);   // its fp8 KV scales are dropped by name
    CHECK_EQ(q.ct_ignore, size_t(304));
    std::printf("  RedHatAI/Qwen3.8-27B-INT4: accepted (sym int4 g128, actorder static, "
                "304 ignore entries, kv_cache_scheme)\n");
  }
  {
    const loader::QuantConfig q =
        loader::QuantConfig::parse(read_json(cfg_dir + "/redhatai-qwen3-32b-w4a16.config.json"));
    CHECK(q.compressed_tensors && q.group_size == 128 && q.ct_actorder == "weight");
    std::printf("  RedHatAI/Qwen3-32B-quantized.w4a16: accepted (sym int4 g128, actorder weight)\n");
  }
  cfg_refused(read_json(cfg_dir + "/padakovec-kolibri-1-w4a16-gptq.config.json"),
              "config_groups.group_0.weights.symmetric = false",
              "Padakovec/Kolibri-1-W4A16-GPTQ (asymmetric g128)");
  cfg_refused(read_json(cfg_dir + "/redhatai-qwen3-8b-w4a16.config.json"),
              "config_groups.group_0.weights.symmetric = false",
              "RedHatAI/Qwen3-8B-quantized.w4a16 (asymmetric g128)");
  cfg_refused(read_json(cfg_dir + "/halt95-qwen3.8-flash-next-w4a16-merlin.config.json"),
              "config_groups.group_1.weights.num_bits = 8",
              "halt95/Qwen3.8-Flash-Next-W4A16-Merlin (an int8 channel group)");

  fs::remove(root);
  std::puts("ct_loader_test OK");
  return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"

// Writes one safetensors file: 8-byte LE header length, header JSON, data.
// The header is space-padded so the data section starts 8-aligned, exactly as
// the real writer does - loader::check_align refuses to cast anything less.
static void write_st(const std::string& path, const std::string& header,
                     const std::vector<uint8_t>& data) {
  std::string h = header;
  while ((8 + h.size()) % 8 != 0) h.push_back(' ');
  std::ofstream f(path, std::ios::binary);
  uint64_t n = h.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(h.data(), std::streamsize(h.size()));
  f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

// One .qzeros (8 words) + one .g_idx (128 entries, group 64) laid end to end.
// `bad_gidx` >= 0 breaks that one g_idx entry (a permuted checkpoint, i.e.
// desc_act smuggled in behind a `false` label).
static std::vector<uint8_t> qz_gidx_bytes(uint32_t qzeros_word5, int bad_gidx = -1) {
  std::vector<uint32_t> w(8, 0x77777777u);
  w[5] = qzeros_word5;
  for (uint32_t k = 0; k < 128; ++k) w.push_back(k / 64);
  if (bad_gidx >= 0) w[8 + size_t(bad_gidx)] = 3;   // identity would be 0 or 1
  std::vector<uint8_t> bytes(w.size() * 4);
  std::memcpy(bytes.data(), w.data(), bytes.size());
  return bytes;
}

int main() {
  using common::json::parse;
  // QuantConfig: accepts the checkpoint's spelling, rejects "+:" rules.
  auto ok = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
      "desc_act":false,"quant_method":"gptq","dynamic":{"-:.*mtp.*":{}}}})");
  loader::QuantConfig qc = loader::QuantConfig::parse(ok);
  CHECK_EQ(qc.group_size, uint32_t(64));
  CHECK_EQ(qc.dynamic_rule_count, size_t(1));
  bool threw = false;
  try {
    auto bad = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "desc_act":false,"dynamic":{"+:.*mtp.*":{}}}})");
    loader::QuantConfig::parse(bad);
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try {
    auto g128 = parse(R"({"quantization_config":{"bits":4,"group_size":128,"sym":true,"desc_act":false}})");
    loader::QuantConfig::parse(g128);
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);

  // repack_int4_layout1 == Int4Gptq::tiled() (after the refactor this is
  // one implementation; the check pins the wrapper wiring).
  common::Int4Gptq w = common::Int4Gptq::random(128, 64, 7);
  std::vector<uint32_t> a = w.tiled();
  std::vector<uint32_t> b(a.size());
  common::repack_int4_layout1(w.qweight.data(), w.scales.data(), w.K, w.N, b.data());
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK_EQ(a[i], b[i]);

  // cols_concat: two parts side by side == one repack of manually concatenated weights.
  common::Int4Gptq p1 = common::Int4Gptq::random(128, 32, 8);
  common::Int4Gptq p2 = common::Int4Gptq::random(128, 32, 9);
  common::Int4Gptq cat; cat.K = 128; cat.N = 64;
  cat.qweight.resize(size_t(cat.K / 8) * cat.N);
  cat.scales.resize(size_t(cat.K / 64) * cat.N);
  for (uint32_t r = 0; r < cat.K / 8; ++r)
    for (uint32_t n = 0; n < 64; ++n)
      cat.qweight[size_t(r) * 64 + n] =
          n < 32 ? p1.qweight[size_t(r) * 32 + n] : p2.qweight[size_t(r) * 32 + (n - 32)];
  for (uint32_t g = 0; g < cat.K / 64; ++g)
    for (uint32_t n = 0; n < 64; ++n)
      cat.scales[size_t(g) * 64 + n] =
          n < 32 ? p1.scales[size_t(g) * 32 + n] : p2.scales[size_t(g) * 32 + (n - 32)];
  std::vector<uint32_t> want = cat.tiled();
  std::vector<uint32_t> got(want.size());
  std::vector<common::ColSource> cols = common::cols_concat(
      {{p1.qweight.data(), p1.scales.data(), 32}, {p2.qweight.data(), p2.scales.data(), 32}});
  common::repack_int4_layout1_cols(128, 64, cols, got.data());
  for (size_t i = 0; i < want.size(); ++i) CHECK_EQ(want[i], got[i]);

  // cols_interleave16: block n_tile 0 is a[0:16], tile 1 is b[0:16], tile 2 a[16:32]...
  std::vector<common::ColSource> il = common::cols_interleave16(
      {p1.qweight.data(), p1.scales.data(), 32}, {p2.qweight.data(), p2.scales.data(), 32});
  CHECK_EQ(il.size(), size_t(64));
  CHECK_EQ(il[0].qweight, p1.qweight.data());  CHECK_EQ(il[0].n, uint32_t(0));
  CHECK_EQ(il[16].qweight, p2.qweight.data()); CHECK_EQ(il[16].n, uint32_t(0));
  CHECK_EQ(il[32].qweight, p1.qweight.data()); CHECK_EQ(il[32].n, uint32_t(16));
  CHECK_EQ(il[63].qweight, p2.qweight.data()); CHECK_EQ(il[63].n, uint32_t(31));

  // assert_quant_invariants over a synthetic two-tensor set: the good pair
  // passes; a single corrupted qzeros word and a single non-identity g_idx
  // entry each throw, naming the tensor and the element index.
  const std::string dir = "/tmp/b70_quant_test";
  CHECK(system(("mkdir -p " + dir).c_str()) == 0);
  const std::string hdr =
      R"({"blk.qzeros":{"dtype":"I32","shape":[2,4],"data_offsets":[0,32]},)"
      R"("blk.g_idx":{"dtype":"I32","shape":[128],"data_offsets":[32,544]}})";
  const std::string ix = R"({"weight_map":{"blk.qzeros":"FILE","blk.g_idx":"FILE"}})";
  write_st(dir + "/good.safetensors", hdr, qz_gidx_bytes(0x77777777u));
  write_st(dir + "/bad.safetensors", hdr, qz_gidx_bytes(0x77770777u));
  write_st(dir + "/badg.safetensors", hdr, qz_gidx_bytes(0x77777777u, 70));
  {
    std::string good_ix = ix;
    good_ix.replace(good_ix.find("FILE"), 4, "good.safetensors");
    good_ix.replace(good_ix.find("FILE"), 4, "good.safetensors");
    std::ofstream(dir + "/model.safetensors.index.json") << good_ix;
  }
  loader::assert_quant_invariants(loader::SafetensorsSet(dir + "/"));
  {
    std::string bad_ix = ix;
    bad_ix.replace(bad_ix.find("FILE"), 4, "bad.safetensors");
    bad_ix.replace(bad_ix.find("FILE"), 4, "bad.safetensors");
    std::ofstream(dir + "/model.safetensors.index.json") << bad_ix;
  }
  threw = false;
  try {
    loader::assert_quant_invariants(loader::SafetensorsSet(dir + "/"));
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    threw = msg.find("blk.qzeros") != std::string::npos && msg.find("[5]") != std::string::npos;
  }
  CHECK(threw);

  // The g_idx branch: element 70 is 3 where the identity says 1. Until now the
  // only exercised failure was qzeros, so this half of the invariant was
  // asserted by inspection only.
  {
    std::string badg_ix = ix;
    badg_ix.replace(badg_ix.find("FILE"), 4, "badg.safetensors");
    badg_ix.replace(badg_ix.find("FILE"), 4, "badg.safetensors");
    std::ofstream(dir + "/model.safetensors.index.json") << badg_ix;
  }
  threw = false;
  try {
    loader::assert_quant_invariants(loader::SafetensorsSet(dir + "/"));
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    threw = msg.find("blk.g_idx") != std::string::npos && msg.find("[70]") != std::string::npos &&
            msg.find("identity") != std::string::npos;
  }
  CHECK(threw);

  // .scales: a NaN/Inf f16 is a hard error (the dequant has no guard for it);
  // a subnormal is counted and allowed, because the real checkpoint has them
  // and the device reads the f16 word natively.
  {
    const std::string shdr =
        R"({"blk.scales":{"dtype":"F16","shape":[1,4],"data_offsets":[0,8]}})";
    const std::string six = R"({"weight_map":{"blk.scales":"FILE"}})";
    auto scale_bytes = [](uint16_t bad) {
      const uint16_t v[4] = {0x3C00u /*1.0*/, 0x00A8u /*subnormal*/, 0x3800u /*0.5*/, bad};
      std::vector<uint8_t> b(8);
      std::memcpy(b.data(), v, 8);
      return b;
    };
    write_st(dir + "/sc_ok.safetensors", shdr, scale_bytes(0x0000u));
    write_st(dir + "/sc_inf.safetensors", shdr, scale_bytes(0x7C00u));
    std::string ok_ix = six;
    ok_ix.replace(ok_ix.find("FILE"), 4, "sc_ok.safetensors");
    std::ofstream(dir + "/model.safetensors.index.json") << ok_ix;
    loader::QuantScan sc = loader::assert_quant_invariants(loader::SafetensorsSet(dir + "/"));
    CHECK_EQ(sc.subnormal_scales, size_t(1));
    // ...and f16_to_f32 decodes that subnormal exactly rather than flushing it.
    CHECK_EQ(common::f16_to_f32(0x00A8u), float(0xA8) * 5.9604644775390625e-08f);
    CHECK(common::f16_to_f32(0x00A8u) > 0.0f);

    std::string inf_ix = six;
    inf_ix.replace(inf_ix.find("FILE"), 4, "sc_inf.safetensors");
    std::ofstream(dir + "/model.safetensors.index.json") << inf_ix;
    threw = false;
    try {
      loader::assert_quant_invariants(loader::SafetensorsSet(dir + "/"));
    } catch (const std::runtime_error& e) {
      const std::string msg = e.what();
      threw = msg.find("blk.scales") != std::string::npos && msg.find("[3]") != std::string::npos &&
              msg.find("Inf") != std::string::npos;
    }
    CHECK(threw);
  }

  std::puts("quant_test OK");
  return 0;
}

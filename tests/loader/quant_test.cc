#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "common/json.h"
#include "common/kv8.h"
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

int main(int argc, char** argv) {
  using common::json::parse;
  // QuantConfig: accepts the checkpoint's spelling, rejects "+:" rules.
  auto ok = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
      "desc_act":false,"quant_method":"gptq","dynamic":{"-:.*mtp.*":{}}}})");
  loader::QuantConfig qc = loader::QuantConfig::parse(ok);
  CHECK_EQ(qc.group_size, uint32_t(64));
  CHECK_EQ(qc.dynamic_rule_count, size_t(1));
  CHECK(qc.desc_act_declared);
  CHECK_EQ(qc.quant_method, std::string("gptq"));
  CHECK_EQ(qc.extra_excluded, size_t(0));

  // The SECOND spelling: auto-round's `auto_round:auto_gptq`, as written by
  // tools/quantize_qwen38_rtn.sh (2026-08-26). No `desc_act` key, no `dynamic`
  // key, an `extra_config` object instead - and an `lm_head` entry that says
  // the head itself was packed. Every field below is copied from the real
  // config.json of models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64.
  auto ar = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
      "data_type":"int","autoround_version":"0.14.2","quant_method":"auto-round",
      "packing_format":"auto_round:auto_gptq",
      "extra_config":{
        "lm_head":{"act_bits":16,"act_data_type":"float","act_dynamic":true,"act_group_size":64,
                   "act_sym":true,"bits":4,"data_type":"int","group_size":64,
                   "rotation_config":null,"super_bits":null,"super_group_size":null,"sym":true},
        "model.language_model.layers.0.linear_attn.in_proj_a":{"bits":16,"data_type":"fp"},
        "model.language_model.layers.0.linear_attn.in_proj_b":{"bits":16,"data_type":"fp"},
        "mtp.fc":{"bits":16,"data_type":"fp"}}}})");
  loader::QuantConfig aq = loader::QuantConfig::parse(ar);
  CHECK_EQ(aq.group_size, uint32_t(64));
  CHECK(aq.sym);
  CHECK(!aq.desc_act);           // inferred, not read
  CHECK(!aq.desc_act_declared);  // and the loader knows it was inferred
  CHECK_EQ(aq.quant_method, std::string("auto-round"));
  CHECK_EQ(aq.packing_format, std::string("auto_round:auto_gptq"));
  CHECK_EQ(aq.dynamic_rule_count, size_t(0));
  CHECK_EQ(aq.extra_excluded, size_t(3));
  CHECK_EQ(aq.extra_quantised, size_t(1));

  bool threw = false;
  // An extra_config module quantised at a group size the loader does not
  // implement is a hard stop naming the module - the failure mode `dynamic`'s
  // regexes could only be approximated at. g128 is implemented (expanded to
  // g64 at load); g256 is not.
  {
    auto g128mod = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "quant_method":"auto-round","extra_config":{"lm_head":{"bits":4,"group_size":128,"sym":true}}}})");
    CHECK_EQ(loader::QuantConfig::parse(g128mod).extra_quantised, size_t(1));
  }
  try {
    auto g256mod = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "quant_method":"auto-round","extra_config":{"lm_head":{"bits":4,"group_size":256,"sym":true}}}})");
    loader::QuantConfig::parse(g256mod);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("lm_head") != std::string::npos;
  }
  CHECK(threw);
  threw = false;
  try {
    auto asym = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "quant_method":"auto-round","extra_config":{"lm_head":{"bits":4,"group_size":128,"sym":false}}}})");
    loader::QuantConfig::parse(asym);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("lm_head") != std::string::npos;
  }
  CHECK(threw);
  threw = false;
  try {
    auto pf = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "quant_method":"auto-round","packing_format":"auto_round:awq"}})");
    loader::QuantConfig::parse(pf);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("packing_format") != std::string::npos;
  }
  CHECK(threw);
  threw = false;
  // An unrecognised quant_method is an unrecognised PACKING, and a wrong nibble
  // order is a silently wrong model - so it stops here naming the value.
  try {
    auto qm = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "quant_method":"awq"}})");
    loader::QuantConfig::parse(qm);
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    threw = msg.find("quant_method") != std::string::npos && msg.find("awq") != std::string::npos;
  }
  CHECK(threw);
  threw = false;
  // desc_act=true is still refused whichever vocabulary declares it.
  try {
    auto da = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "desc_act":true,"quant_method":"auto-round"}})");
    loader::QuantConfig::parse(da);
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try {
    auto bad = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "desc_act":false,"dynamic":{"+:.*mtp.*":{}}}})");
    loader::QuantConfig::parse(bad);
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  {
    // A whole-checkpoint g128 sym config (the common GPTQ export) is accepted.
    auto g128 = parse(R"({"quantization_config":{"bits":4,"group_size":128,"sym":true,"desc_act":false}})");
    CHECK_EQ(loader::QuantConfig::parse(g128).group_size, uint32_t(128));
  }
  for (const char* g : {"32", "256"}) {
    threw = false;
    try {
      auto bad_g = parse(std::string(R"({"quantization_config":{"bits":4,"group_size":)") + g +
                         R"(,"sym":true,"desc_act":false}})");
      loader::QuantConfig::parse(bad_g);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
  }
  threw = false;

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
  {
    // The scan also COUNTS the g_idx tensors it walked. That count is what
    // `loader::load` uses to prove an auto-round checkpoint's undeclared
    // `desc_act` really is false: no g_idx anywhere means no mechanism for an
    // activation-order permutation to exist (docs/13).
    loader::QuantScan gs = loader::assert_quant_invariants(loader::SafetensorsSet(dir + "/"));
    CHECK_EQ(gs.g_idx_tensors, size_t(1));
    // ...and this is the pair the loader refuses: a config that never declared
    // `desc_act` (so the loader would infer false) over a checkpoint that DOES
    // ship a permutation vector. The identity scan above has already passed on
    // this very tensor - "every entry is the identity" is not the same claim as
    // "the quantiser said it does not permute", and the loader declines to
    // infer the second from silence. `loader::load` owns the throw; what is
    // asserted here is the two inputs it decides on, so the rule stays readable
    // without a device and a 19 GB checkpoint.
    auto ar_nodesc = parse(R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,
        "quant_method":"auto-round","packing_format":"auto_round:auto_gptq"}})");
    const loader::QuantConfig nq = loader::QuantConfig::parse(ar_nodesc);
    CHECK(!nq.desc_act_declared);          // the loader will infer
    CHECK_EQ(gs.g_idx_tensors, size_t(1)); // ...but there IS a g_idx here
    // => loader::load throws. The condition it evaluates, spelled out:
    CHECK(!nq.desc_act_declared && gs.g_idx_tensors != 0);
  }
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
    CHECK_EQ(sc.g_idx_tensors, size_t(0));   // the auto-round shape: none shipped
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

  // g128 onto the g64 kernels. A K=256, N=16 linear packed at g128 (2 scale
  // rows) is classified as g64 with 4 rows, rows 2j and 2j+1 equal to g128 row
  // j, and every weight dequantises to the g128 value exactly. Its g_idx is
  // the identity k/128 (checked against the module's own scales rows); a k/64
  // g_idx beside g128 scales is the inconsistency the scan refuses.
  {
    const uint32_t K = 256, N = 16;
    common::Int4Gptq w = common::Int4Gptq::random(K, N, 11);   // nibbles; its g64 scales unused
    std::vector<uint16_t> s128(size_t(K / 128) * N);
    for (size_t i = 0; i < s128.size(); ++i) s128[i] = uint16_t(0x3000u + 37u * i);  // distinct
    std::vector<int32_t> gidx(K);
    for (uint32_t k = 0; k < K; ++k) gidx[k] = int32_t(k / 128);
    std::vector<uint8_t> bytes;
    auto put = [&bytes](const void* p, size_t n) {
      const uint8_t* b = static_cast<const uint8_t*>(p);
      bytes.insert(bytes.end(), b, b + n);
    };
    put(w.qweight.data(), w.qweight.size() * 4);   // [0, 2048)
    put(s128.data(), s128.size() * 2);             // [2048, 2112)
    put(gidx.data(), gidx.size() * 4);             // [2112, 3136)
    const std::string ghdr =
        R"({"lin.qweight":{"dtype":"I32","shape":[32,16],"data_offsets":[0,2048]},)"
        R"("lin.scales":{"dtype":"F16","shape":[2,16],"data_offsets":[2048,2112]},)"
        R"("lin.g_idx":{"dtype":"I32","shape":[256],"data_offsets":[2112,3136]}})";
    const std::string gix =
        R"({"weight_map":{"lin.qweight":"g128.safetensors","lin.scales":"g128.safetensors",)"
        R"("lin.g_idx":"g128.safetensors"}})";
    write_st(dir + "/g128.safetensors", ghdr, bytes);
    std::ofstream(dir + "/model.safetensors.index.json") << gix;
    {
      loader::SafetensorsSet set(dir + "/");
      CHECK_EQ(loader::assert_quant_invariants(set).g_idx_tensors, size_t(1));
      loader::LinearSrc ls = loader::LinearSrc::classify(set, "lin");
      CHECK(ls.kind == loader::WKind::Int4);
      CHECK_EQ(ls.K, K);
      CHECK_EQ(ls.N, N);
      CHECK_EQ(ls.group, uint32_t(128));
      CHECK(ls.expanded_scales != nullptr);
      CHECK_EQ(ls.expanded_scales->size(), size_t(K / 64) * N);
      CHECK_EQ(ls.scales, ls.expanded_scales->data());
      common::Int4Gptq got = w;   // same nibbles, the expanded g64 scales
      got.scales.assign(ls.scales, ls.scales + size_t(K / 64) * N);
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n) {
          const uint32_t word = w.qweight[size_t(k / 8) * N + n];
          const int q = int((word >> (4 * (k % 8))) & 0xFu);
          const float want = float(q - 8) * common::f16_to_f32(s128[size_t(k / 128) * N + n]);
          CHECK_EQ(got.at(k, n), want);
        }
      // A copy shares the expansion, so the pointer outlives the original.
      loader::LinearSrc copy = ls;
      ls = loader::LinearSrc{};
      CHECK_EQ(copy.scales, copy.expanded_scales->data());
      CHECK_EQ(copy.scales[N], s128[0]);       // g64 row 1 == g128 row 0
      CHECK_EQ(copy.scales[2 * N], s128[N]);   // g64 row 2 == g128 row 1
    }
    for (uint32_t k = 0; k < K; ++k) gidx[k] = int32_t(k / 64);
    std::memcpy(bytes.data() + 2112, gidx.data(), gidx.size() * 4);
    write_st(dir + "/g128.safetensors", ghdr, bytes);
    threw = false;
    try {
      loader::assert_quant_invariants(loader::SafetensorsSet(dir + "/"));
    } catch (const std::runtime_error& e) {
      const std::string msg = e.what();
      threw = msg.find("lin.g_idx") != std::string::npos && msg.find("[64]") != std::string::npos &&
              msg.find("k/128") != std::string::npos;
    }
    CHECK(threw);
    // Scales whose rows are neither K/64 nor K/128 are refused by classify.
    const std::string bhdr =
        R"({"lin.qweight":{"dtype":"I32","shape":[32,16],"data_offsets":[0,2048]},)"
        R"("lin.scales":{"dtype":"F16","shape":[1,16],"data_offsets":[2048,2080]}})";
    write_st(dir + "/g128.safetensors", bhdr, std::vector<uint8_t>(bytes.begin(), bytes.begin() + 2080));
    std::ofstream(dir + "/model.safetensors.index.json")
        << R"({"weight_map":{"lin.qweight":"g128.safetensors","lin.scales":"g128.safetensors"}})";
    threw = false;
    try {
      loader::LinearSrc::classify(loader::SafetensorsSet(dir + "/"), "lin");
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()).find("matches neither") != std::string::npos;
    }
    CHECK(threw);
  }

  // ===== Spec 25b Task 3: the exact relaxations Nemotron 3.5 Bolt's checkpoint needs (spec 25 §5.2) =====
  // (a) SergiioB's quantization_config (the published file's, typed here with a subset of its 44 '-:' rules): GPTQ
  // packing, g128, and ONE '+:' rule that RESTATES the top level. Parsed; the restatement is counted.
  const std::string sb_head =
      R"({"quantization_config":{"autoround_version":"0.17.0","bits":4,"damp_percent":0.01,"data_type":"int",
         "desc_act":false,"group_size":128,"lm_head":false,"low_gpu_mem_usage":true,"provider":"auto-round",
         "quant_method":"gptq","sym":true,"true_sequential":false,"dynamic":{)";
  const std::string sb_excl =
      R"("-:.*embeddings.*":{},"-:.*lm_head.*":{},"-:.*vision_model.*":{},"-:.*vision_projector.*":{},
         "-:.*language_model\\.model\\.layers\\.10\\.mixer\\.gate.*":{},
         "-:.*language_model\\.model\\.layers\\.12\\.mixer\\.gate.*":{})";
  const std::string sb_rule = R"("+:.*mixer\\.out_proj$":{"bits":4,"group_size":128,"sym":true},)";
  {
    const loader::QuantConfig sb = loader::QuantConfig::parse(parse(sb_head + sb_rule + sb_excl + "}}}"));
    CHECK_EQ(sb.group_size, uint32_t(128));
    CHECK(sb.sym && !sb.desc_act && sb.desc_act_declared);
    CHECK_EQ(sb.quant_method, std::string("gptq"));
    CHECK_EQ(sb.dynamic_rule_count, size_t(6));      // the '-:' exclusions, as before
    CHECK_EQ(sb.dynamic_restating, size_t(1));       // the '+:' restatement, counted apart
    // a config with no '+:' rule counts none
    CHECK_EQ(loader::QuantConfig::parse(parse(sb_head + sb_excl + "}}}")).dynamic_restating, size_t(0));
    // a restating rule under a g64 top level is the same thing
    const loader::QuantConfig g64 = loader::QuantConfig::parse(parse(
        R"({"quantization_config":{"bits":4,"group_size":64,"sym":true,"desc_act":false,"quant_method":"gptq",
           "dynamic":{"+:.*out_proj$":{"bits":4,"group_size":64,"sym":true}}}})"));
    CHECK_EQ(g64.dynamic_restating, size_t(1));
    // two restating rules count two
    CHECK_EQ(loader::QuantConfig::parse(parse(
        R"({"quantization_config":{"bits":4,"group_size":128,"sym":true,"desc_act":false,"dynamic":{
           "+:.*a$":{"bits":4,"group_size":128,"sym":true},"+:.*b$":{"sym":true,"bits":4,"group_size":128}}}})")
        ).dynamic_restating, size_t(2));
    if (argc > 1) {   // the published file itself: 44 '-:' rules and the one '+:'
      std::ifstream f(argv[1]);
      CHECK(f.good());
      std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      const loader::QuantConfig pub = loader::QuantConfig::parse(parse(text));
      CHECK_EQ(pub.group_size, uint32_t(128));
      CHECK_EQ(pub.dynamic_rule_count, size_t(44));
      CHECK_EQ(pub.dynamic_restating, size_t(1));
    }
  }
  // (b) the broken-MTP guard is intact: a '+:' rule that changes bits / group / sym, carries any other key, omits one of
  // the three, or is empty is still refused, naming the rule; a '-:' rule behaves as before.
  for (const char* body : {
           R"({"bits":8,"group_size":128,"sym":true})",                     // bits changed
           R"({"bits":4,"group_size":64,"sym":true})",                      // group changed under a g128 top level
           R"({"bits":4,"group_size":128,"sym":false})",                    // sym changed
           R"({"bits":4,"group_size":128,"sym":true,"data_type":"fp"})",    // another key
           R"({"bits":4,"group_size":128,"sym":true,"act_bits":8})",        // another key
           R"({"bits":4,"group_size":128})",                                // a key omitted
           R"({"bits":16,"data_type":"fp"})",                               // an fp exclusion spelt as a '+:' rule
           R"({})",                                                         // empty (the pre-25b refusal)
           R"(7)"}) {                                                       // not an object
    threw = false;
    try {
      loader::QuantConfig::parse(parse(sb_head + R"("+:.*mixer\\.out_proj$":)" + body + "," + sb_excl + "}}}"));
    } catch (const std::runtime_error& e) {
      const std::string msg = e.what();
      threw = msg.find("+:.*mixer") != std::string::npos && msg.find("quantization_config.dynamic") != std::string::npos;
      if (!threw) std::fprintf(stderr, "refusal of %s named something else: %s\n", body, e.what());
    }
    if (!threw) std::fprintf(stderr, "'+:' rule %s was not refused\n", body);
    CHECK(threw);
  }
  threw = false;
  try {   // a restating rule does not excuse a different TOP level
    loader::QuantConfig::parse(parse(R"({"quantization_config":{"bits":4,"group_size":128,"sym":true,"desc_act":false,
        "dynamic":{"+:.*x$":{"bits":4,"group_size":64,"sym":true}}}})"));
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);

  // (c) + (d): a GPTQ linear K=256, N=16 at g128 with BF16 `.scales` (a future export's; signed, as AutoRound writes them)
  // converts exactly through bf16_to_f16_exact, the sign bit survives the exact g128 -> g64 expansion, and assert_quant_
  // invariants / check_quant_scan accept it. A value f16 cannot hold exactly is refused naming the tensor and the element.
  {
    const uint32_t K = 256, N = 16;
    common::Int4Gptq w = common::Int4Gptq::random(K, N, 21);
    auto bytes_of = [&](const std::vector<uint16_t>& scales) {
      std::vector<uint8_t> bytes;
      auto put = [&bytes](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), b, b + n);
      };
      put(w.qweight.data(), w.qweight.size() * 4);   // [0, 2048)
      put(scales.data(), scales.size() * 2);         // [2048, 2112)
      const std::vector<uint32_t> qz(2 * N / 8, 0x77777777u);
      put(qz.data(), qz.size() * 4);                 // [2112, 2128)
      return bytes;
    };
    auto lin_dir = [&](const char* sdtype, const std::vector<uint16_t>& scales) {
      const std::string hdr = std::string(R"({"lin.qweight":{"dtype":"I32","shape":[32,16],"data_offsets":[0,2048]},)") +
                              R"("lin.scales":{"dtype":")" + sdtype + R"(","shape":[2,16],"data_offsets":[2048,2112]},)" +
                              R"("lin.qzeros":{"dtype":"I32","shape":[2,2],"data_offsets":[2112,2128]}})";
      write_st(dir + "/lin.safetensors", hdr, bytes_of(scales));
      std::ofstream(dir + "/model.safetensors.index.json")
          << R"({"weight_map":{"lin.qweight":"lin.safetensors","lin.scales":"lin.safetensors","lin.qzeros":"lin.safetensors"}})";
    };
    // 32 scales: signed, normal and f16-subnormal (2^-20 is a bf16 and an exact f16 subnormal), zero never.
    std::vector<float> vals(2 * N);
    for (uint32_t i = 0; i < 2 * N; ++i) {
      const float mag = (i % 5 == 4) ? std::ldexp(1.0f, -20) : 0.0125f * float(1 + i % 7);
      vals[i] = (i % 2 == 0) ? -mag : mag;
    }
    std::vector<uint16_t> bf(vals.size()), h16(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
      bf[i] = common::f32_to_bf16(vals[i]);
      h16[i] = common::kv8::f32_to_f16_rne(common::bf16_to_f32(bf[i]));   // (common::f32_to_f16 flushes subnormals)
      CHECK_EQ(common::f16_to_f32(h16[i]), common::bf16_to_f32(bf[i]));   // the test's own values are exact f16
    }
    lin_dir("BF16", bf);
    {
      loader::SafetensorsSet set(dir + "/");
      const loader::QuantScan scan = loader::assert_quant_invariants(set);
      CHECK_EQ(scan.gptq_linears, size_t(1));
      const loader::QuantConfig q = loader::QuantConfig::parse(parse(
          R"({"quantization_config":{"bits":4,"group_size":128,"sym":true,"desc_act":false,"quant_method":"gptq"}})"));
      loader::check_quant_scan(q, scan);
      const loader::LinearSrc ls = loader::LinearSrc::classify(set, "lin");
      CHECK(ls.kind == loader::WKind::Int4);
      CHECK_EQ(ls.group, uint32_t(128));
      CHECK(ls.expanded_scales != nullptr);
      CHECK_EQ(ls.scales, ls.expanded_scales->data());
      CHECK_EQ(ls.expanded_scales->size(), size_t(K / 64) * N);
      for (uint32_t j = 0; j < 2; ++j)
        for (uint32_t n = 0; n < N; ++n) {
          CHECK_EQ(ls.scales[size_t(2 * j) * N + n], h16[size_t(j) * N + n]);       // both g64 rows of g128 row j
          CHECK_EQ(ls.scales[size_t(2 * j + 1) * N + n], h16[size_t(j) * N + n]);   // bit for bit, SIGN INCLUDED
          CHECK_EQ(bool(ls.scales[size_t(2 * j) * N + n] & 0x8000u), vals[size_t(j) * N + n] < 0);
        }
      // the dequant is (q - 8) x scale with the signed scale
      common::Int4Gptq got = w;
      got.scales.assign(ls.scales, ls.scales + size_t(K / 64) * N);
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n) {
          const int q4 = int((w.qweight[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 0xFu);
          CHECK_EQ(got.at(k, n), float(q4 - 8) * common::bf16_to_f32(bf[size_t(k / 128) * N + n]));
        }
      // the layout-1 and layout-0 repack carry every scale word (sign bit too) verbatim
      std::vector<uint32_t> t1(size_t(N / 16) * (K / 64) * 136);
      common::repack_int4_layout1(ls.qweight, ls.scales, K, N, t1.data());
      for (uint32_t g = 0; g < K / 64; ++g)
        for (uint32_t l = 0; l < 16; l += 2) {
          const uint32_t word = t1[size_t(g) * 136 + 128 + l / 2];
          CHECK_EQ(uint16_t(word & 0xFFFFu), ls.scales[size_t(g) * N + l]);
          CHECK_EQ(uint16_t(word >> 16), ls.scales[size_t(g) * N + l + 1]);
        }
      std::vector<uint32_t> q0(size_t(K / 8) * N);
      std::vector<uint16_t> s0(size_t(K / 64) * N);
      common::repack_int4_layout0_cols(K, N, common::cols_concat({{ls.qweight, ls.scales, N}}), q0.data(), s0.data());
      CHECK(std::memcmp(s0.data(), ls.scales, s0.size() * 2) == 0);
    }
    // an f16 export of the same values: the same expanded words (the BF16 path equals the F16 path for exact values)
    lin_dir("F16", h16);
    {
      loader::SafetensorsSet set(dir + "/");
      const loader::QuantScan scan = loader::assert_quant_invariants(set);
      CHECK(scan.subnormal_scales > 0);   // the 2^-20 ones, counted not rejected
      const loader::LinearSrc ls = loader::LinearSrc::classify(set, "lin");
      for (uint32_t j = 0; j < 2; ++j)
        for (uint32_t n = 0; n < N; ++n) {
          CHECK_EQ(ls.scales[size_t(2 * j) * N + n], h16[size_t(j) * N + n]);
          CHECK_EQ(ls.scales[size_t(2 * j + 1) * N + n], h16[size_t(j) * N + n]);
        }
    }
    // a BF16 g64 linear (4 scale rows) is converted too
    {
      std::vector<uint16_t> bf64(size_t(K / 64) * N);
      for (size_t i = 0; i < bf64.size(); ++i) bf64[i] = bf[i % bf.size()];
      std::vector<uint8_t> bytes;
      auto put = [&bytes](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), b, b + n);
      };
      put(w.qweight.data(), w.qweight.size() * 4);
      put(bf64.data(), bf64.size() * 2);   // [2048, 2176)
      write_st(dir + "/lin.safetensors",
               R"({"lin.qweight":{"dtype":"I32","shape":[32,16],"data_offsets":[0,2048]},)"
               R"("lin.scales":{"dtype":"BF16","shape":[4,16],"data_offsets":[2048,2176]}})", bytes);
      std::ofstream(dir + "/model.safetensors.index.json")
          << R"({"weight_map":{"lin.qweight":"lin.safetensors","lin.scales":"lin.safetensors"}})";
      loader::SafetensorsSet set(dir + "/");
      const loader::LinearSrc ls = loader::LinearSrc::classify(set, "lin");
      CHECK_EQ(ls.group, uint32_t(64));
      for (size_t i = 0; i < bf64.size(); ++i)
        CHECK_EQ(ls.scales[i], common::kv8::f32_to_f16_rne(common::bf16_to_f32(bf64[i])));
    }
    // a bf16 scale f16 cannot hold exactly is refused naming the tensor and the element - in classify AND in the scan
    for (const auto& [bad_f, tag] : {std::pair<float, const char*>{std::ldexp(1.0f, -26), "below f16's subnormals"},
                                      {70000.0f, "above 65504"}, {-std::ldexp(1.0f, -26), "negative, below"}}) {
      std::vector<uint16_t> evil = bf;
      evil[5] = common::f32_to_bf16(bad_f);
      lin_dir("BF16", evil);
      loader::SafetensorsSet set(dir + "/");
      threw = false;
      try {
        loader::LinearSrc::classify(set, "lin");
      } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        threw = msg.find("lin.scales[5]") != std::string::npos && msg.find("not exactly representable") != std::string::npos;
        if (!threw) std::fprintf(stderr, "classify (%s): %s\n", tag, e.what());
      }
      CHECK(threw);
      threw = false;
      try {
        loader::assert_quant_invariants(set);
      } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("lin.scales[5]") != std::string::npos;
        if (!threw) std::fprintf(stderr, "scan (%s): %s\n", tag, e.what());
      }
      CHECK(threw);
    }
    // a BF16 NaN scale is refused too (the dequant has no guard)
    {
      std::vector<uint16_t> evil = bf;
      evil[3] = 0x7FC0;
      lin_dir("BF16", evil);
      loader::SafetensorsSet set(dir + "/");
      threw = false;
      try {
        loader::LinearSrc::classify(set, "lin");
      } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("lin.scales[3]") != std::string::npos; }
      CHECK(threw);
    }
    // any OTHER scale dtype is still refused (F32)
    {
      write_st(dir + "/lin.safetensors",
               R"({"lin.qweight":{"dtype":"I32","shape":[32,16],"data_offsets":[0,2048]},)"
               R"("lin.scales":{"dtype":"F32","shape":[2,16],"data_offsets":[2048,2176]}})",
               std::vector<uint8_t>(2176, 0));
      std::ofstream(dir + "/model.safetensors.index.json")
          << R"({"weight_map":{"lin.qweight":"lin.safetensors","lin.scales":"lin.safetensors"}})";
      threw = false;
      try {
        loader::LinearSrc::classify(loader::SafetensorsSet(dir + "/"), "lin");
      } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("lin.scales dtype F32") != std::string::npos; }
      CHECK(threw);
    }
  }

  std::puts("quant_test OK");
  return 0;
}

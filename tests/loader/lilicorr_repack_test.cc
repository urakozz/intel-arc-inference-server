// Spec 26b Task 2 (host): the LiLiCorr drafter's loader host half (loader::LiliCheckpoint) - NVFP4 decoded exactly into the
// three drafter formats, the head always exact, the names both ways and the bytes. No device.
//
//   1. A tiny synthetic checkpoint in the published names and dtypes (tests/loader/lilicorr_fixture.h): the names both
//      ways (a missing tensor, an unknown one, a scale without its U8 weight, a U8 weight without its scales, NVFP4
//      without a quantization_config, a shipped mask embedding - each refused by name); 25 -> here 16 NVFP4 linears.
//   2. The formats: Exact - W' in gemv_bf16's tiles + g per column (gate||up: two g's); a BF16 linear's tiles as shipped,
//      no output scale. Int8 - spec 9's codes of W' INDEPENDENT of g (the same checkpoint at another g: the same codes,
//      the scales x the ratio) and the scales fp32(g x s'_r). Int4G64 - rtn from g x W' in fp64 (rtn_int4_g64_g), a BF16
//      linear by rtn.h. The head in every arm: W'^T, g, its bias.
//   3. The bytes: every piece's host bytes sum to loader::lilicorr_weight_bytes in each format; the published config's
//      derived bytes (spec 26 §4 / §12): exact 2.152 GB + the head 0.074, int8 1.077 + 0.074, int4 g64 0.572 + 0.074.
//   4. The real checkpoint when the HF cache holds it (argv[2]: a repo id or a directory; SKIP that part otherwise):
//      179 tensors, 25 NVFP4 linears, held to Bolt's shape; one backbone linear and one head linear decoded (W' exact,
//      the int8 codes and the head's g).
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/lilicorr_fixture.h"
#include "loader/lilicorr_layout.h"
#include "loader/lilicorr_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/nvfp4.h"
#include "loader/rtn.h"
#include "model/bolt.h"

namespace fs = std::filesystem;
namespace lx = lilicorr_fixture;

namespace {

bool throws_with(const std::function<void()>& f, const std::string& needle) {
  try {
    f();
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) {
      std::printf("  refused: %s\n", e.what());
      return true;
    }
    std::fprintf(stderr, "  threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  std::fprintf(stderr, "  did not throw (wanted '%s')\n", needle.c_str());
  return false;
}

fs::path tmpdir(const std::string& tag) {
  fs::path p = fs::temp_directory_path() / ("lilicorr_repack_test_" + tag + "_" + std::to_string(::getpid()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

// W' and g of one fixture linear, decoded independently of the loader (or the BF16 rows as written).
struct Ref {
  std::vector<uint16_t> wp;
  float g = 1.0f;
  bool nvfp4 = false;
};
Ref ref_of(const lx::Tensors& ts, const std::string& base, uint32_t K, uint32_t N) {
  Ref r;
  const lx::Tensor& w = ts.at(base + ".weight");
  if (w.dtype == "U8") {
    r.nvfp4 = true;
    r.wp = loader::decode_nvfp4(w.bytes.data(), ts.at(base + ".weight_scale").bytes.data(), K, N, base);
    std::memcpy(&r.g, ts.at(base + ".weight_scale_2").bytes.data(), 4);
  } else {
    r.wp.resize(w.bytes.size() / 2);
    std::memcpy(r.wp.data(), w.bytes.data(), w.bytes.size());
  }
  return r;
}

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: lilicorr_repack_test <lilicorr config.json> [<real LiLiCorr repo id or dir>]\n");
    return 2;
  }
  const lx::Shape S;
  const fs::path dir = tmpdir("tiny");
  const lx::Tensors ts = lx::make(S, 7);
  lx::write(dir, S, ts);

  // --- 1. names both ways ------------------------------------------------------------------------------------------
  const loader::LiliCheckpoint ck(dir.string());
  const model::DFlashDesc& d = ck.desc();
  {
    CHECK(d.lilicorr() && d.hidden == 64 && d.lil_hidden == 32 && d.lil_mlp == 64 && d.cand_topk == 8);
    CHECK(ck.fc_k() == 6 * 64 && ck.target_hidden() == 64 && ck.tensors() == ts.size());
    CHECK(ck.nvfp4_linears() == lx::nvfp4_bases(S).size() && ck.nvfp4_linears() == 16);
    ck.check_target(model::DFlashTargetShape{"tiny", 12, 64, 1024});
    CHECK(throws_with([&] { ck.check_target(model::DFlashTargetShape{"tiny", 12, 64, 512}); }, "vocab_size 1024"));
    auto variant = [&](const std::string& tag, const std::function<void(lx::Tensors&)>& edit, lx::Shape s = lx::Shape{}) {
      lx::Tensors t = ts;
      edit(t);
      const fs::path v = tmpdir(tag);
      lx::write(v, s, t);
      return v.string();
    };
    std::string v = variant("missing", [](lx::Tensors& t) { t.erase("layers.1.self_attn.attention_sink_bias"); });
    CHECK(throws_with([&] { (void)loader::LiliCheckpoint(v); }, "missing tensor layers.1.self_attn.attention_sink_bias"));
    v = variant("unknown", [&](lx::Tensors& t) { t["lilicorr.extra"] = ts.at("lilicorr.same_slot_bias"); });
    CHECK(throws_with([&] { (void)loader::LiliCheckpoint(v); }, "unknown tensor lilicorr.extra"));
    v = variant("orphan", [&](lx::Tensors& t) { t["layers.0.self_attn.q_proj.weight_scale_2"] = ts.at("fc.weight_scale_2"); });
    CHECK(throws_with([&] { (void)loader::LiliCheckpoint(v); }, "outside a complete NVFP4 linear"));
    v = variant("noscale", [](lx::Tensors& t) { t.erase("layers.0.mlp.down_proj.weight_scale"); });
    CHECK(throws_with([&] { (void)loader::LiliCheckpoint(v); }, "missing tensor layers.0.mlp.down_proj.weight_scale"));
    v = variant("dtype", [&](lx::Tensors& t) { t["norm.weight"].dtype = "F16"; });
    CHECK(throws_with([&] { (void)loader::LiliCheckpoint(v); }, "norm.weight is F16, not BF16"));
    lx::Shape noq;
    noq.quant_config = false;
    v = variant("noqc", [](lx::Tensors&) {}, noq);
    CHECK(throws_with([&] { (void)loader::LiliCheckpoint(v); }, "without a quantization_config"));
    v = variant("mask", [](lx::Tensors&) {});
    std::ofstream(fs::path(v) / "mask_embedding.safetensors") << "x";
    CHECK(throws_with([&] { (void)loader::LiliCheckpoint(v); }, "mask_embedding.safetensors"));
    v = variant("nan", [](lx::Tensors& t) { t["layers.1.mlp.up_proj.weight_scale"].bytes[3 * 4 + 2] = 0xFF; });
    const loader::LiliCheckpoint bad(v);   // opens (headers only); the decode names the element
    CHECK(throws_with([&] { (void)bad.decoded("layers.1.mlp.up_proj"); }, "layers.1.mlp.up_proj: weight_scale[3][2]"));
    std::printf("1. the tiny checkpoint: %zu tensors, %zu NVFP4 linears; missing / unknown / orphan scales / dtype / no "
                "quantization_config / a mask file / a NaN scale refused by name\n",
                ck.tensors(), ck.nvfp4_linears());
  }

  // --- 2. the formats ----------------------------------------------------------------------------------------------
  {
    using F = loader::DraftFormat;
    const uint32_t H = 64, I = 128;
    const Ref gate = ref_of(ts, "layers.1.mlp.gate_proj", H, I), up = ref_of(ts, "layers.1.mlp.up_proj", H, I);
    CHECK(gate.nvfp4 && up.nvfp4 && gate.g != up.g);
    // Exact
    {
      const loader::DFlashHostLinear h = ck.layer_linear(1, loader::DFlashLinear::GateUp, F::Exact);
      std::vector<uint16_t> t(size_t(H) * 2 * I), rows(gate.wp);
      rows.insert(rows.end(), up.wp.begin(), up.wp.end());
      common::repack_bf16_tiled(rows.data(), H, 2 * I, t.data());
      CHECK(h.kind == model::WeightKind::Bf16 && h.K == H && h.N == 2 * I && h.bf16 == t && h.gs.size() == 2 * I);
      for (uint32_t n = 0; n < 2 * I; ++n) CHECK(h.gs[n] == (n < I ? gate.g : up.g));
      const loader::DFlashHostLinear q = ck.layer_linear(0, loader::DFlashLinear::Qkv, F::Exact);
      CHECK(q.gs.empty() && q.N == 96);   // BF16 parts: gemv_bf16 as shipped
      const Ref fcr = ref_of(ts, "fc", 6 * 64, 64);
      const loader::DFlashHostLinear fc = ck.fc(F::Exact);
      CHECK(fc.gs == std::vector<float>(64, fcr.g));
    }
    // Int8: the codes of W', independent of g
    {
      const lx::Tensors t2 = lx::make(S, 7, 3.25f);   // the same codes and scales, every g x 3.25
      const fs::path d2 = tmpdir("g2");
      lx::write(d2, S, t2);
      const loader::LiliCheckpoint ck2(d2.string());
      const loader::DFlashHostLinear a = ck.layer_linear(1, loader::DFlashLinear::GateUp, F::Int8, 1);
      const loader::DFlashHostLinear b = ck2.layer_linear(1, loader::DFlashLinear::GateUp, F::Int8, 3);
      CHECK(a.kind == model::WeightKind::Int8 && a.i8 == b.i8 && a.gs.empty());
      std::vector<int8_t> q(size_t(H) * I);
      std::vector<float> s(I);
      loader::quantise_int8_tiled(gate.wp.data(), H, I, q.data(), s.data(), 1);
      CHECK(std::vector<int8_t>(a.i8.begin(), a.i8.begin() + long(q.size())) == q);
      for (uint32_t n = 0; n < I; ++n) CHECK(a.scales[n] == gate.g * s[n]);
      const Ref gate2 = ref_of(t2, "layers.1.mlp.gate_proj", H, I);
      for (uint32_t n = 0; n < I; ++n) CHECK(b.scales[n] == gate2.g * s[n]);
      fs::remove_all(d2);
    }
    // Int4 g64: from g x W' in fp64; a BF16 linear by rtn.h
    {
      const loader::DFlashHostLinear h = ck.layer_linear(1, loader::DFlashLinear::GateUp, F::Int4G64);
      std::vector<uint32_t> qw(size_t(H / 8) * I), tiles(size_t(I / 16) * (H / 64) * 136);
      std::vector<uint16_t> sc(size_t(H / 64) * I);
      loader::rtn_int4_g64_g(gate.wp.data(), double(gate.g), H, I, qw.data(), sc.data());
      common::repack_int4_layout1(qw.data(), sc.data(), H, I, tiles.data());
      CHECK(h.kind == model::WeightKind::Int4 && std::vector<uint32_t>(h.int4.begin(), h.int4.begin() + long(tiles.size())) == tiles);
      const Ref o = ref_of(ts, "layers.0.self_attn.o_proj", 64, 64);
      CHECK(!o.nvfp4);
      const loader::DFlashHostLinear ho = ck.layer_linear(0, loader::DFlashLinear::O, F::Int4G64);
      std::vector<uint32_t> qo(size_t(64 / 8) * 64), to(size_t(64 / 16) * 1 * 136);
      std::vector<uint16_t> so(64);
      loader::rtn_int4_g64(o.wp.data(), 64, 64, qo.data(), so.data());
      common::repack_int4_layout1(qo.data(), so.data(), 64, 64, to.data());
      CHECK(ho.int4 == to);
    }
    // Bf16 is not a LiLiCorr format
    CHECK(throws_with([&] { (void)ck.fc(F::Bf16); }, "exact is"));
    // the head: always exact - W'^T, g, the bias
    {
      const loader::LiliHostHead t = ck.head_linear(loader::LiliHead::TokenProj);
      const Ref r = ref_of(ts, "lilicorr.token_proj", 64, 32);
      CHECK(r.nvfp4 && t.K == 64 && t.N == 32 && t.gs == std::vector<float>(32, r.g));
      for (uint32_t n = 0; n < 32; ++n)
        for (uint32_t k = 0; k < 64; ++k) CHECK(t.wt[size_t(k) * 32 + n] == r.wp[size_t(n) * 64 + k]);
      const uint16_t* bias = reinterpret_cast<const uint16_t*>(ts.at("lilicorr.token_proj.bias").bytes.data());
      for (uint32_t n = 0; n < 32; ++n) CHECK(t.bias[n] == common::bf16_to_f32(bias[n]));
      const loader::LiliHostHead oi = ck.head_linear(loader::LiliHead::OutIn);
      CHECK(oi.K == 32 && oi.N == 64 && oi.gs == std::vector<float>(64, 1.0f));
      const loader::LiliHostHead ip = ck.head_linear(loader::LiliHead::InProj, 1);
      CHECK(ip.K == 32 && ip.N == 96);
      CHECK(throws_with([&] { (void)ck.head_linear(loader::LiliHead::MlpUp, 2); }, "no layer 2"));
    }
    std::printf("2. exact = W' tiles + g per column (gate||up: two g's), BF16 linears as shipped; int8 codes independent "
                "of g, scales fp32(g x s'); int4 g64 from g x W' in fp64 (BF16: rtn.h); the head W'^T + g + bias\n");
  }

  // --- 3. the bytes ------------------------------------------------------------------------------------------------
  {
    using F = loader::DraftFormat;
    for (F f : {F::Exact, F::Int8, F::Int4G64}) {
      size_t lin = ck.fc(f).bytes();
      for (uint32_t l = 0; l < d.layers; ++l)
        for (loader::DFlashLinear w : {loader::DFlashLinear::Qkv, loader::DFlashLinear::O, loader::DFlashLinear::GateUp,
                                       loader::DFlashLinear::Down, loader::DFlashLinear::AttnKproj,
                                       loader::DFlashLinear::MlpKproj})
          lin += ck.layer_linear(l, w, f).bytes();
      const size_t small = (ck.hidden_norm().size() + ck.final_norm().size() + ck.qk_norms().size() +
                            ck.conv_bases().size() + ck.sinks().size() + d.hidden) * 4 +
                           size_t(d.layers) * ck.layer_norms(0).size() * 4;
      size_t head = ck.head_small().size() * 4;
      for (loader::LiliHead w : {loader::LiliHead::TokenProj, loader::LiliHead::PassHidden, loader::LiliHead::FeatUp,
                                 loader::LiliHead::FeatDown, loader::LiliHead::ContextProj, loader::LiliHead::Factor,
                                 loader::LiliHead::OutIn, loader::LiliHead::AnchorOut})
        head += ck.head_linear(w).bytes();
      for (uint32_t l = 0; l < d.lil_layers; ++l)
        for (loader::LiliHead w : {loader::LiliHead::InProj, loader::LiliHead::OutProj, loader::LiliHead::MlpUp,
                                   loader::LiliHead::MlpDown})
          head += ck.head_linear(w, l).bytes();
      const loader::LiliBytes b = loader::lilicorr_weight_bytes(d, 64, f, ck.backbone_nvfp4_rows());
      CHECK_EQ(b.linears, lin);
      CHECK_EQ(b.small, small);
      CHECK_EQ(b.head, head);
    }
    CHECK_EQ(ck.backbone_nvfp4_rows(), loader::lili_nvfp4_rows_published(d));
    const model::DFlashDesc real = model::dflash_desc_from_config(common::json::parse(slurp(argv[1])));
    const uint64_t rows = loader::lili_nvfp4_rows_published(real);
    const loader::LiliBytes ex = loader::lilicorr_weight_bytes(real, 4096, F::Exact, rows),
                            i8 = loader::lilicorr_weight_bytes(real, 4096, F::Int8, rows),
                            i4 = loader::lilicorr_weight_bytes(real, 4096, F::Int4G64, rows);
    std::printf("3. the published drafter (derived): backbone exact %.4f GB (+ %llu B of output scales), int8 %.4f GB, "
                "int4 g64 %.4f GB; the head %.4f GB in every arm; small %zu B\n",
                double(ex.linears) / 1e9, (unsigned long long)(rows * 4), double(i8.linears) / 1e9,
                double(i4.linears) / 1e9, double(ex.head) / 1e9, ex.small);
    CHECK(std::fabs(double(ex.linears) / 1e9 - 2.152) < 0.0015 && std::fabs(double(i8.linears) / 1e9 - 1.077) < 0.0015);
    CHECK(std::fabs(double(i4.linears) / 1e9 - 0.572) < 0.0015 && std::fabs(double(ex.head) / 1e9 - 0.074) < 0.0015);
    CHECK(ex.head == i8.head && ex.head == i4.head && ex.small == i4.small);
  }

  // --- 4. the real checkpoint (when cached) --------------------------------------------------------------------------
  if (argc >= 3) {
    std::unique_ptr<loader::LiliCheckpoint> real;
    try {
      real = std::make_unique<loader::LiliCheckpoint>(argv[2]);
    } catch (const std::exception& e) {
      std::printf("4. SKIP: the real LiLiCorr checkpoint is not here (%s)\n", e.what());
    }
    if (real) {
      const model::BoltDesc& bolt = model::bolt();
      real->check_target(model::DFlashTargetShape{bolt.name, bolt.layers, bolt.hidden, bolt.vocab});
      CHECK(real->tensors() == 179 && real->nvfp4_linears() == 25);
      model::check_lilicorr(real->desc());
      const loader::LiliDecoded dn = real->decoded("layers.0.mlp.down_proj");
      CHECK(dn.nvfp4 && dn.K == 12288 && dn.N == 4096 && std::isfinite(dn.g) && dn.g > 0.0f);
      const loader::DFlashHostLinear q = real->layer_linear(0, loader::DFlashLinear::Down, loader::DraftFormat::Int8);
      std::vector<int8_t> codes(dn.wp.size());
      std::vector<float> s(dn.N);
      loader::quantise_int8_tiled(dn.wp.data(), dn.K, dn.N, codes.data(), s.data());
      CHECK(q.i8 == codes);
      const loader::LiliHostHead hp = real->head_linear(loader::LiliHead::PassHidden);
      CHECK(hp.K == 4096 && hp.N == 1024 && hp.gs[0] != 1.0f);
      CHECK(real->backbone_nvfp4_rows() == loader::lili_nvfp4_rows_published(real->desc()));
      std::printf("4. the real checkpoint: 179 tensors, 25 NVFP4 linears, held to %s; layers.0.mlp.down_proj decoded "
                  "(g %.6g), its int8 codes = spec 9's of W'; pass_hidden_proj's g %.6g\n",
                  bolt.name.c_str(), double(dn.g), double(hp.gs[0]));
    }
  }
  fs::remove_all(dir);
  std::printf("lilicorr_repack_test OK\n");
  return 0;
}

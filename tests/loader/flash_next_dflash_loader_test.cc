// Plan 21-QFN-v3e Task 1 (host): the Flash-Next DFlash 2 drafter's loader arm (loader::flash_next, the generic
// DFlash loader at kernels::dflash::kFlashNext's shape set), no device.
//
//   flash_next_dflash_loader_test [published config.json] [real drafter snapshot]
//
//   1. The published config (tests/model/flash_next_dflash/config.json): held to kFlashNext and to Flash-Next
//      (flash_next::check), and every linear's {K, N} of the shape set (flash_next::linear_shapes) is the pair the
//      kernels' binaries bake: fc 12800 x 2560, q||k||v 2560 x 6144, o 4096 x 2560, gate||up 2560 x 15360, down
//      7680 x 2560, the conv kernel projections 2560 x 640, the selector 2560 x 256, the context k||v 2560 x 2048;
//      the load forms' byte totals are printed (the planner's numbers).
//   2. A tiny synthetic checkpoint (tests/loader/dflash_fixture.h): opens through the arm as Int8 (the bf16
//      release's form); every linear's device form equals quantise_int8_tiled over the concatenated bf16 rows with
//      one fp32 scale a row, every piece's {K, N} where linear_shapes says, the codebooks the shard's own bf16 rows,
//      the norms the fp32 widenings; a wrong name and a wrong shape are refused naming them.
//   3. A packed synthetic export (GPTQ qweight / qzeros / scales + quantization_config): opens as Int4G64 with
//      require_packed_g64, every linear bitwise the bf16 checkpoint's at-load int4 form; a quantised folder that
//      leaves a linear in bf16 is refused under require_packed_g64, a wrong qzeros word too.
//   4. With the real snapshot (on the box): the header check - opens, passes flash_next::check against Flash-Next,
//      and its expected names are exactly the shard's tensors. No payload is read.
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/dflash_fixture.h"
#include "loader/dflash_layout.h"
#include "loader/dflash_repack.h"
#include "loader/flash_next_dflash.h"
#include "loader/lm_head_int8.h"
#include "loader/rtn.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include <algorithm>
#include "model/qwen4exp.h"

namespace fs = std::filesystem;
namespace fx = dflash_fixture;

namespace {

bool throws_with(const std::function<void()>& f, const std::string& needle) {
  try {
    f();
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) return true;
    std::fprintf(stderr, "  threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  std::fprintf(stderr, "  did not throw (wanted '%s')\n", needle.c_str());
  return false;
}

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

model::DFlashDesc parse(const std::string& text) { return model::dflash_desc_from_config(common::json::parse(text)); }

fs::path tmpdir(const std::string& tag) {
  fs::path p = fs::temp_directory_path() / ("fn_dflash_loader_" + tag + "_" + std::to_string(::getpid()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

std::vector<uint16_t> rows_of(const fx::Tensors& ts, const std::vector<std::string>& names) {
  std::vector<uint16_t> out;
  for (const std::string& n : names) {
    const fx::Tensor& t = ts.at(n);
    const uint16_t* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
    out.insert(out.end(), p, p + t.bytes.size() / 2);
  }
  return out;
}

// The int8 device form a linear of these rows must have (spec 9's quantiser, one fp32 scale a row).
void check_int8_linear(const loader::DFlashHostLinear& h, const std::vector<uint16_t>& rows, uint32_t K,
                       uint32_t N) {
  CHECK(h.kind == model::WeightKind::Int8);
  CHECK_EQ(h.K, K);
  CHECK_EQ(h.N, N);
  CHECK_EQ(rows.size(), size_t(K) * N);
  std::vector<int8_t> q(size_t(K) * N);
  std::vector<float> s(N);
  loader::quantise_int8_tiled(rows.data(), K, N, q.data(), s.data(), 1);
  CHECK(h.i8 == q);
  CHECK(h.scales == s);
  CHECK(h.bf16.empty() && h.int4.empty());
  CHECK_EQ(h.bytes(), loader::dflash_linear_bytes(K, N, loader::DraftFormat::Int8));
}

// The int4 g64 device form (layout 1) a linear of these rows must have.
void check_int4_linear(const loader::DFlashHostLinear& h, const std::vector<uint16_t>& rows, uint32_t K,
                       uint32_t N) {
  CHECK(h.kind == model::WeightKind::Int4);
  CHECK_EQ(h.K, K);
  CHECK_EQ(h.N, N);
  std::vector<uint32_t> q(size_t(K / 8) * N), want(size_t(N / 16) * (K / 64) * 136);
  std::vector<uint16_t> sc(size_t(K / 64) * N);
  loader::rtn_int4_g64(rows.data(), K, N, q.data(), sc.data());
  common::repack_int4_layout1(q.data(), sc.data(), K, N, want.data());
  CHECK(h.int4 == want);
  CHECK_EQ(h.bytes(), loader::dflash_linear_bytes(K, N, loader::DraftFormat::Int4G64));
}

// Builds every linear of the tiny checkpoint at `f` and holds each piece to the shape set and the layout function.
void check_all_linears(const loader::DFlashCheckpoint& ck, const fx::Tensors& ts, loader::DraftFormat f) {
  const model::DFlashDesc& d = ck.desc();
  const uint32_t TH = ck.target_hidden();
  const auto shapes = loader::flash_next::linear_shapes(d, TH);
  size_t i = 0;
  auto next = [&]() { return shapes.at(i++); };
  auto one = [&](const loader::DFlashHostLinear& h, const std::vector<std::string>& parts) {
    const auto [K, N] = next();
    const std::vector<uint16_t> rows = rows_of(ts, parts);
    if (f == loader::DraftFormat::Int8) check_int8_linear(h, rows, K, N);
    else check_int4_linear(h, rows, K, N);
  };
  one(ck.fc(f), {"fc.weight"});
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    using W = loader::DFlashLinear;
    one(ck.layer_linear(l, W::Qkv, f),
        {L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight", L + "self_attn.v_proj.weight"});
    one(ck.layer_linear(l, W::O, f), {L + "self_attn.o_proj.weight"});
    one(ck.layer_linear(l, W::GateUp, f), {L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"});
    one(ck.layer_linear(l, W::Down, f), {L + "mlp.down_proj.weight"});
    one(ck.layer_linear(l, W::AttnKproj, f), {L + "attention_conv.kernel_projection.weight"});
    one(ck.layer_linear(l, W::MlpKproj, f), {L + "mlp_conv.kernel_projection.weight"});
  }
  one(ck.selector_projection(f), {"candidate_selector.hidden_projection.weight"});
  next();   // the context k||v: q||k||v's rows in place, no piece of its own
  CHECK_EQ(i, shapes.size());
}

// The packed synthetic export of `ts` (every linear as GPTQ qweight / qzeros / scales, rtn's words).
fx::Tensors packed_of(const fx::Tensors& ts) {
  fx::Tensors pk;
  for (const auto& [n, t] : ts) {
    const bool linear = t.shape.size() == 2 && n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0;
    if (!linear) {
      pk[n] = t;
      continue;
    }
    const uint32_t N = uint32_t(t.shape[0]), K = uint32_t(t.shape[1]);
    std::vector<uint32_t> q(size_t(K / 8) * N);
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    loader::rtn_int4_g64(reinterpret_cast<const uint16_t*>(t.bytes.data()), K, N, q.data(), sc.data());
    const std::string b = n.substr(0, n.size() - 7);
    auto bytes = [](const auto& v) {
      return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(v.data()),
                                  reinterpret_cast<const uint8_t*>(v.data()) + v.size() * sizeof(v[0]));
    };
    pk[b + ".qweight"] = {"I32", {K / 8, N}, bytes(q)};
    pk[b + ".scales"] = {"F16", {K / 64, N}, bytes(sc)};
    pk[b + ".qzeros"] = {"I32", {K / 64, N / 8}, bytes(std::vector<uint32_t>(size_t(K / 64) * (N / 8), 0x77777777u))};
  }
  return pk;
}

const char* kQuantConfig = "\"quantization_config\":{\"bits\":4,\"group_size\":64,\"sym\":true,"
                           "\"quant_method\":\"auto-round\",\"packing_format\":\"auto_round:auto_gptq\"}";

void write_packed(const fs::path& dir, const fx::Shape& s, const fx::Tensors& t, const std::string& q) {
  fx::write(dir, s, t);
  std::string c = fx::config_json(s);
  c.insert(c.size() - 1, "," + q);
  std::ofstream(dir / "config.json") << c;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cfg = argc > 1 ? argv[1] : "tests/model/flash_next_dflash/config.json";
  const std::string snapshot = argc > 2 ? argv[2] : "";
  const model::Qwen4ExpDesc& fn = model::qwen4exp();
  const model::DFlashTargetShape target{fn.name, fn.layers, fn.hidden, fn.vocab};

  // --- 1. the published config and the shape set ---------------------------------------------------------
  const model::DFlashDesc d = parse(slurp(cfg));
  loader::flash_next::check(d, target);   // passes: kFlashNext's shapes, Flash-Next's target
  {
    const auto shapes = loader::flash_next::linear_shapes(d, fn.hidden);
    const std::vector<std::pair<uint32_t, uint32_t>> want = {
        {12800, 2560},   // fc: 5 taps x 2560
        {2560, 6144}, {4096, 2560}, {2560, 15360}, {7680, 2560}, {2560, 640}, {2560, 640},   // layer 0
    };
    for (size_t k = 0; k < want.size(); ++k) CHECK(shapes.at(k) == want[k]);
    for (uint32_t l = 1; l < d.layers; ++l)
      for (uint32_t j = 1; j < 7; ++j) CHECK(shapes.at(1 + l * 6 + j - 1) == want[j]);
    CHECK(shapes.at(1 + d.layers * 6) == (std::pair<uint32_t, uint32_t>{2560, 256}));   // the selector's
    CHECK(shapes.at(2 + d.layers * 6) == (std::pair<uint32_t, uint32_t>{2560, 2048}));  // the context k||v
    CHECK_EQ(shapes.size(), 2 + d.layers * 6 + 1);
    for (loader::DraftFormat f : {loader::DraftFormat::Int8, loader::DraftFormat::Int4G64}) {
      const loader::DFlashBytes b = loader::dflash_weight_bytes(d, fn.hidden, f);
      std::printf("shape set at %s: linears %zu B, small %zu B, codebooks %zu B, total %.3f GB\n",
                  loader::draft_format_name(f), b.linears, b.small, b.codebooks, b.total() / 1e9);
    }
    std::puts("1. the published config held to kFlashNext and Flash-Next; linear_shapes == the kernels' pairs");
  }

  // --- 2. the tiny synthetic checkpoint, the bf16 release's form (Int8) ------------------------------------
  fx::Shape s;
  s.inter = 128;   // down's K must hold whole int4 groups of 64
  const fx::Tensors ts = fx::make(s, 7);
  const fs::path dir = tmpdir("bf16");
  fx::write(dir, s, ts);
  {
    loader::DraftFormat f;
    const loader::DFlashCheckpoint ck = loader::flash_next::open(dir.string(), f);
    CHECK(f == loader::DraftFormat::Int8 && !ck.packed());
    CHECK_EQ(ck.tensors(), ts.size());
    check_all_linears(ck, ts, loader::DraftFormat::Int8);
    // the codebooks stay at source precision, the norms widen to fp32
    for (bool succ : {false, true}) {
      const fx::Tensor& t = ts.at(succ ? "candidate_selector.successor_codebook"
                                       : "candidate_selector.predecessor_codebook");
      CHECK(std::memcmp(ck.codebook(succ), t.bytes.data(), t.bytes.size()) == 0);
    }
    const auto widen = [&](const std::string& n) {
      const fx::Tensor& t = ts.at(n);
      const uint16_t* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
      std::vector<float> v(t.bytes.size() / 2);
      for (size_t k = 0; k < v.size(); ++k) v[k] = common::bf16_to_f32(p[k]);
      return v;
    };
    CHECK(ck.hidden_norm() == widen("hidden_norm.weight"));
    CHECK(ck.final_norm() == widen("norm.weight"));
    // a wrong name and a wrong shape, refused naming them
    auto refused = [&](fx::Tensors t2, const std::string& needle) {
      const fs::path d2 = tmpdir("bad");
      fx::write(d2, s, t2);
      const bool ok = throws_with([&] { loader::DraftFormat ff; loader::flash_next::open(d2.string(), ff); },
                                  needle);
      fs::remove_all(d2);
      return ok;
    };
    {
      fx::Tensors t2 = ts;
      t2.erase("layers.1.mlp.down_proj.weight");
      CHECK(refused(t2, "layers.1.mlp.down_proj.weight"));
    }
    {
      fx::Tensors t2 = ts;
      t2["layers.0.self_attn.q_norm.weight"].shape = {8, 2};
      CHECK(refused(t2, "layers.0.self_attn.q_norm.weight"));
    }
    {
      fx::Tensors t2 = ts;
      t2["lm_head.weight"] = ts.at("norm.weight");
      CHECK(refused(t2, "lm_head.weight"));
    }
    // a drafter that is not Flash-Next's is refused by the hold
    CHECK(throws_with([&] { loader::flash_next::check(ck.desc(), target); }, "Flash-Next DFlash drafter"));
    std::puts("2. tiny checkpoint: Int8 at load, every tensor where the shape set says, int8 rows and scales, "
              "codebooks bf16, wrong name / shape refused");
  }

  // --- 3. the packed synthetic export (Int4G64) --------------------------------------------------------------
  {
    const fx::Tensors pk = packed_of(ts);
    const fs::path pdir = tmpdir("packed"), bdir = tmpdir("plain");
    write_packed(pdir, s, pk, kQuantConfig);
    fx::write(bdir, s, ts);
    loader::DraftFormat f;
    const loader::DFlashCheckpoint ck = loader::flash_next::open(pdir.string(), f, /*require_packed_g64=*/true);
    CHECK(f == loader::DraftFormat::Int4G64 && ck.packed());
    const loader::DFlashCheckpoint bf(bdir.string());
    check_all_linears(ck, ts, loader::DraftFormat::Int4G64);
    // bitwise the bf16 checkpoint's at-load int4 form
    const loader::DraftFormat i4 = loader::DraftFormat::Int4G64;
    CHECK(ck.fc(i4).int4 == bf.fc(i4).int4);
    CHECK(ck.selector_projection(i4).int4 == bf.selector_projection(i4).int4);
    for (uint32_t l = 0; l < s.layers; ++l)
      for (auto w : {loader::DFlashLinear::Qkv, loader::DFlashLinear::O, loader::DFlashLinear::GateUp,
                     loader::DFlashLinear::Down, loader::DFlashLinear::AttnKproj, loader::DFlashLinear::MlpKproj})
        CHECK(ck.layer_linear(l, w, i4).int4 == bf.layer_linear(l, w, i4).int4);
    // a quantised folder that leaves a linear in bf16 is refused under require_packed_g64
    {
      fx::Tensors t2 = pk;
      t2.erase("fc.qweight");
      t2.erase("fc.scales");
      t2.erase("fc.qzeros");
      t2["fc.weight"] = ts.at("fc.weight");
      const fs::path d2 = tmpdir("packed_bf16");
      write_packed(d2, s, t2, kQuantConfig);
      CHECK(throws_with([&] { loader::DraftFormat ff; loader::flash_next::open(d2.string(), ff, true); },
                        "must ship as a packed GPTQ linear"));
      fs::remove_all(d2);
    }
    {
      fx::Tensors t2 = pk;
      t2["layers.1.mlp.down_proj.qzeros"].bytes[0] = 0x78;
      const fs::path d2 = tmpdir("packed_qz");
      write_packed(d2, s, t2, kQuantConfig);
      CHECK(throws_with([&] { loader::DraftFormat ff; loader::flash_next::open(d2.string(), ff, true); },
                        "qzeros"));
      fs::remove_all(d2);
    }
    fs::remove_all(pdir);
    fs::remove_all(bdir);
    std::puts("3. packed export: Int4G64, bitwise the at-load int4 form, incomplete packing / wrong qzeros refused");
  }
  fs::remove_all(dir);

  // --- 4. the real release's header check (on the box) -------------------------------------------------------
  if (!snapshot.empty()) {
    loader::DraftFormat f;
    const loader::DFlashCheckpoint ck = loader::flash_next::open(snapshot, f);
    CHECK(f == loader::DraftFormat::Int8 && !ck.packed());
    loader::flash_next::check(ck, target);
    const std::vector<std::string> want = loader::dflash_expected_names(ck.desc());
    const loader::SafetensorsSet shards(loader::resolve_snapshot(snapshot), true);
    std::vector<std::string> got;
    for (const auto& [name, t] : shards.tensors()) got.push_back(loader::dflash_canon(name));
    std::sort(got.begin(), got.end());
    CHECK(got == want);
    const loader::DFlashBytes b = loader::dflash_weight_bytes(ck.desc(), ck.target_hidden(), f);
    std::printf("4. release %s: %zu tensors, names both ways, Int8 total %.3f GB (linears %.3f, codebooks %.3f)\n",
                snapshot.c_str(), shards.tensors().size(), b.total() / 1e9, b.linears / 1e9, b.codebooks / 1e9);
  }
  std::printf("flash_next_dflash_loader_test: PASS\n");
  return 0;
}

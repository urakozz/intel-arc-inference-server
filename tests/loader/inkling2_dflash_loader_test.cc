// Plan 41k Task 2 (host): the DFlash 2 drafter loaded in DSpark's place (loader::Inkling2Checkpoint with its folder).
//   inkling2_dflash_loader_test <Inkling snapshot> <DFlash 2 snapshot>
// The drafter's descriptor replaces DSpark's; its draft arena holds every tensor in its device form, checked bitwise
// against the checkpoint read here independently: every linear int8 with one fp32 scale a row (quantise_int8_tiled),
// q || k || v and gate || up rows concatenated first, q_norm || k_norm, the other norms and the conv bases as fp32, the codebooks as stored, and the mask
// row embed_norm(embed[200056]) rounded to bf16. Refused in plain words: a folder that is not the drafter (its config
// changed), an incomplete quantization config. No device.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <unistd.h>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/lm_head_int8.h"
#include "loader/inkling2_loader.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

namespace fs = std::filesystem;

namespace {

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

bool refused(const std::string& target, const std::string& dir, const std::string& needle) {
  try {
    const loader::Inkling2Checkpoint c(target, dir);
  } catch (const std::exception& e) {
    std::printf("  refused: %s\n", e.what());
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  CHECK_EQ(argc, 3);
  const std::string target = argv[1], dir = loader::resolve_snapshot(argv[2]);
  const loader::Inkling2Checkpoint c(target, dir);
  const auto& d = c.desc().draft;
  CHECK(c.layout().draft_format == loader::DraftFormat::Int8);
  CHECK(c.has_drafter() && c.dflash() && d.v2() && d.softcap_after && d.layers == 5 && d.hidden == 2304);
  using L = loader::Inkling2WeightLayout;
  const auto& lay = c.layout();
  std::printf("DFlash 2 drafter arena: %zu bytes\n", lay.bytes[L::kDraft]);

  // the device form, captured from stream()
  std::vector<std::byte> arena(lay.bytes[L::kDraft]);
  c.stream([&](uint32_t a, size_t off, std::span<const std::byte> bytes) {
    if (a == L::kDraft) std::memcpy(arena.data() + off, bytes.data(), bytes.size());
  }, true);

  // the checkpoint read independently
  loader::SafetensorsSet st(dir, true);
  const auto raw = [&](const std::string& n) {
    const auto& t = st.tensors().at(n);
    const auto* p = reinterpret_cast<const uint16_t*>(st.data(t));
    return std::vector<uint16_t>(p, p + (t.end - t.begin) / 2);
  };
  const auto at = [&](const std::string& n, size_t bytes) {
    const auto& w = lay.weights.at(n);
    CHECK_EQ(w.bytes, bytes);
    return arena.data() + w.offset;
  };
  size_t checked = 0;
  const auto tiled = [&](const std::string& name, const std::vector<std::string>& parts) {
    std::vector<uint16_t> rows;
    for (const auto& p : parts) {
      const auto v = raw(p);
      rows.insert(rows.end(), v.begin(), v.end());
    }
    const auto& w = lay.weights.at(name);
    std::vector<int8_t> want(rows.size());
    std::vector<float> scales(w.N);
    loader::quantise_int8_tiled(rows.data(), w.K, w.N, want.data(), scales.data());
    CHECK(std::memcmp(at(name, want.size()), want.data(), want.size()) == 0);
    CHECK(std::memcmp(arena.data() + w.scales, scales.data(), scales.size() * 4) == 0);
    ++checked;
  };
  const auto floats = [&](const std::string& name, const std::vector<std::string>& parts) {
    std::vector<float> want;
    for (const auto& p : parts)
      for (const auto v : raw(p)) want.push_back(common::bf16_to_f32(v));
    CHECK(std::memcmp(at(name, want.size() * 4), want.data(), want.size() * 4) == 0);
    ++checked;
  };
  tiled("fc.weight", {"fc.weight"});
  floats("hidden_norm.weight", {"hidden_norm.weight"});
  floats("norm.weight", {"norm.weight"});
  tiled("candidate_selector.hidden_projection.weight", {"candidate_selector.hidden_projection.weight"});
  for (const char* n : {"candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"}) {
    const auto v = raw(n);
    CHECK(std::memcmp(at(n, v.size() * 2), v.data(), v.size() * 2) == 0);
    ++checked;
  }
  for (uint32_t l = 0; l < d.layers; ++l) {
    const auto p = "layers." + std::to_string(l) + ".", s = p + "self_attn.";
    tiled(s + "qkv_proj.weight", {s + "q_proj.weight", s + "k_proj.weight", s + "v_proj.weight"});
    tiled(s + "o_proj.weight", {s + "o_proj.weight"});
    tiled(p + "mlp.gate_up_proj.weight", {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
    tiled(p + "mlp.down_proj.weight", {p + "mlp.down_proj.weight"});
    floats(s + "qk_norm", {s + "q_norm.weight", s + "k_norm.weight"});
    for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight"}) floats(p + n, {p + n});
    for (const char* cv : {"attention_conv.", "mlp_conv."}) {
      floats(p + cv + "base_kernel", {p + cv + "base_kernel"});
      tiled(p + cv + "kernel_projection.weight", {p + cv + "kernel_projection.weight"});
    }
  }
  {   // the mask row
    loader::SafetensorsSet ts(loader::resolve_snapshot(target));
    const auto& e = ts.tensors().at("model.llm.embed.weight");
    const auto& n = ts.tensors().at("model.llm.embed_norm.weight");
    const auto* row = reinterpret_cast<const uint16_t*>(ts.data(e)) + size_t(d.mask_id) * d.hidden;
    const auto* w = reinterpret_cast<const uint16_t*>(ts.data(n));
    float ss = 0.f;
    for (uint32_t i = 0; i < d.hidden; ++i) ss += common::bf16_to_f32(row[i]) * common::bf16_to_f32(row[i]);
    const float inv = 1.f / std::sqrt(ss / d.hidden + c.desc().eps);
    std::vector<uint16_t> want(d.hidden);
    for (uint32_t i = 0; i < d.hidden; ++i)
      want[i] = common::f32_to_bf16((common::bf16_to_f32(row[i]) * inv) * common::bf16_to_f32(w[i]));
    CHECK(std::memcmp(at("df2.mask_embed", d.hidden * 2), want.data(), want.size() * 2) == 0);
    ++checked;
  }
  std::printf("%zu weights bitwise the checkpoint's device form\n", checked);

  // refusals: a changed config, an incomplete quantization config (each a copy of the folder's config beside its weights' link)
  const fs::path tmp = fs::temp_directory_path() / ("inkling2_dflash_loader_test_" + std::to_string(::getpid()));
  fs::create_directories(tmp);
  fs::create_symlink(fs::path(dir) / "model.safetensors", tmp / "model.safetensors");
  const std::string cfg = slurp(dir + "config.json");
  const auto write = [&](std::string text) { std::ofstream(tmp / "config.json") << text; };
  std::string t = cfg;
  t.replace(t.find("\"after_correction\""), 18, "\"before_correction\"");
  write(t);
  CHECK(refused(target, tmp.string(), "after the correction"));
  t = cfg;
  t.insert(t.find("\"attention_bias\""), "\"quantization_config\": {\"quant_method\": \"auto-round\"}, ");
  write(t);
  CHECK(refused(target, tmp.string(), "bits"));
  fs::remove_all(tmp);
  std::printf("inkling2_dflash_loader_test OK\n");
  return 0;
}

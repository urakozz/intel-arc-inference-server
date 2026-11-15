// Kolibri DSpark (host): the DSpark drafter through the DFlash loader's host half (loader::DFlashCheckpoint), no device.
//
//   1. A tiny DSpark checkpoint (tests/loader/dflash_fixture.h's v1 tensors + the Markov head + the confidence head, a
//      speculators config.json): the DSpark form; the names both ways (a missing markov_w1, an unknown tensor, a confidence
//      head without enable_confidence_head - each refused by name); the codebooks are the Markov head's own rows
//      (pred = markov_w1, succ = markov_w2); every linear's bf16 device form is fixture 19b's; the bytes formula counts the
//      two Markov tables.
//   2. Held to a target by its parts (DFlashTargetShape): fc's K = taps x the target's hidden, the taps inside the target.
//   3. The real drafter's bytes at bf16 (the planner's number, printed): 5 layers at 2560 / 6144 + fc 12800 x 2560 + the two
//      128000 x 256 Markov tables.
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "loader/dflash_fixture.h"
#include "loader/dflash_layout.h"
#include "loader/dflash_repack.h"
#include "model/dflash.h"

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

// The speculators form of the fixture's v1 shape: the layer config nested, the taps as aux ids (+ 1), the DSpark keys.
std::string dspark_config(const fx::Shape& s, uint32_t rank, bool confidence) {
  std::string aux, types;
  for (size_t i = 0; i < s.taps.size(); ++i) aux += (i ? "," : "") + std::to_string(s.taps[i] + 1);
  for (size_t i = 0; i < s.layer_types.size(); ++i) types += (i ? "," : "") + ("\"" + s.layer_types[i] + "\"");
  return "{\"architectures\":[\"DSparkDraftModel\"],\"aux_hidden_state_layer_ids\":[" + aux + "],\"block_size\":" +
         std::to_string(s.block) + ",\"confidence_head_with_markov\":true,\"draft_vocab_size\":" + std::to_string(s.vocab) +
         ",\"enable_confidence_head\":" + (confidence ? "true" : "false") + ",\"markov_head_type\":\"vanilla\"," +
         "\"markov_rank\":" + std::to_string(rank) + ",\"mask_token_id\":" + std::to_string(s.mask_id) +
         ",\"sample_from_anchor\":true,\"sliding_window_non_causal\":false,\"speculators_model_type\":\"dspark\"," +
         "\"tie_word_embeddings\":false,\"transformer_layer_config\":{\"attention_bias\":false,\"head_dim\":" +
         std::to_string(s.head_dim) + ",\"hidden_act\":\"silu\",\"hidden_size\":" + std::to_string(s.hidden) +
         ",\"intermediate_size\":" + std::to_string(s.inter) + ",\"layer_types\":[" + types +
         "],\"model_type\":\"qwen3\",\"num_attention_heads\":" + std::to_string(s.q_heads) + ",\"num_hidden_layers\":" +
         std::to_string(s.layers) + ",\"num_key_value_heads\":" + std::to_string(s.kv_heads) +
         ",\"rms_norm_eps\":1e-06,\"rope_parameters\":{\"rope_theta\":10000.0,\"rope_type\":\"default\"},\"sliding_window\":" +
         std::to_string(s.window) + ",\"vocab_size\":" + std::to_string(s.vocab) + "}}";
}

}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("dspark_loader_test_" + std::to_string(getpid()));
  fs::remove_all(root);
  fx::Shape s;
  s.version = 1;
  const uint32_t rank = 16;
  fx::Tensors ts = fx::make(s, 7);
  fx::Rng r(11);
  ts["markov_head.markov_w1.weight"] = fx::bf16_tensor(r, {s.vocab, rank}, 0.5f);
  ts["markov_head.markov_w2.weight"] = fx::bf16_tensor(r, {s.vocab, rank}, 0.5f);
  ts["confidence_head.proj.weight"] = fx::bf16_tensor(r, {1, uint64_t(s.hidden) + rank}, 0.1f);
  ts["confidence_head.proj.bias"] = fx::bf16_tensor(r, {1}, 0.1f);
  auto write = [&](const fs::path& dir, const fx::Tensors& t, bool confidence) {
    fs::create_directories(dir);
    fx::write_safetensors(dir / "model.safetensors", t);
    std::ofstream(dir / "config.json") << dspark_config(s, rank, confidence);
  };

  // --- 1. the checkpoint ------------------------------------------------------------------------------------------------
  const fs::path good = root / "good";
  write(good, ts, true);
  const loader::DFlashCheckpoint ck(good.string());
  const model::DFlashDesc& d = ck.desc();
  CHECK(d.dspark() && !d.v2() && d.sel_rank == rank && d.conf_in == s.hidden + rank);
  CHECK((d.taps == s.taps) && d.block == s.block && d.max_k() == s.block - 1);
  CHECK(d.causal == std::vector<bool>(s.layers, true));
  CHECK(std::memcmp(ck.codebook(false), ts["markov_head.markov_w1.weight"].bytes.data(), size_t(s.vocab) * rank * 2) == 0);
  CHECK(std::memcmp(ck.codebook(true), ts["markov_head.markov_w2.weight"].bytes.data(), size_t(s.vocab) * rank * 2) == 0);
  const loader::DFlashHostLinear qkv = ck.layer_linear(0, loader::DFlashLinear::Qkv, loader::DraftFormat::Bf16);
  CHECK(qkv.kind == model::WeightKind::Bf16 && qkv.K == s.hidden && qkv.N == d.qkv_n());
  const loader::DFlashBytes b = loader::dflash_weight_bytes(d, s.target_hidden, loader::DraftFormat::Bf16);
  CHECK_EQ(b.codebooks, size_t(2) * s.vocab * rank * 2);
  {
    fx::Tensors t = ts;
    t.erase("markov_head.markov_w1.weight");
    write(root / "no_w1", t, true);
    CHECK(throws_with([&] { loader::DFlashCheckpoint c((root / "no_w1").string()); }, "missing tensor markov_head.markov_w1"));
    t = ts;
    t["markov_head.extra"] = fx::bf16_tensor(r, {4}, 0.1f);
    write(root / "unknown", t, true);
    CHECK(throws_with([&] { loader::DFlashCheckpoint c((root / "unknown").string()); }, "unknown tensor markov_head.extra"));
    write(root / "no_conf", ts, false);
    CHECK(throws_with([&] { loader::DFlashCheckpoint c((root / "no_conf").string()); }, "unknown tensor confidence_head"));
    t = ts;
    t.erase("confidence_head.proj.weight");
    t.erase("confidence_head.proj.bias");
    write(root / "conf_off", t, false);
    const loader::DFlashCheckpoint c((root / "conf_off").string());
    CHECK(c.desc().conf_in == 0);
  }
  std::printf("1. the tiny DSpark checkpoint: names both ways, the Markov tables as the codebooks, %zu B in bf16\n", b.total());

  // --- 2. held to a target by its parts ---------------------------------------------------------------------------------
  ck.check_target(model::DFlashTargetShape{"tiny", s.target_layers, s.target_hidden, s.vocab});
  CHECK(throws_with([&] { ck.check_target(model::DFlashTargetShape{"tiny", 3, s.target_hidden, s.vocab}); },
                    "target_layer_ids"));
  CHECK(throws_with([&] { ck.check_target(model::DFlashTargetShape{"tiny", s.target_layers, 32, s.vocab}); },
                    "hidden_size"));
  std::printf("2. held to a target by its parts\n");

  // --- 3. the real drafter's bytes --------------------------------------------------------------------------------------
  {
    model::DFlashDesc real;
    real.form = model::DFlashForm::DSpark;
    real.version = 1;
    real.layers = 5;
    real.hidden = 2560;
    real.q_heads = 20;
    real.kv_heads = 4;
    real.head_dim = 128;
    real.inter = 6144;
    real.vocab = 128000;
    real.sel_rank = 256;
    real.taps = {3, 13, 23, 33, 43};
    const loader::DFlashBytes rb = loader::dflash_weight_bytes(real, 2560, loader::DraftFormat::Bf16);
    const size_t per_layer = (size_t(2560) * 3584 + 2560ull * 2560 + 2560ull * 12288 + 6144ull * 2560) * 2;
    CHECK_EQ(rb.linears, size_t(12800) * 2560 * 2 + 5 * per_layer);
    CHECK_EQ(rb.codebooks, size_t(2) * 128000 * 256 * 2);
    std::printf("3. the published drafter in bf16: %zu B of linears, %zu B of Markov tables, %zu B in all (DERIVED)\n",
                rb.linears, rb.codebooks, rb.total());
  }
  fs::remove_all(root);
  std::printf("dspark_loader_test: OK\n");
  return 0;
}

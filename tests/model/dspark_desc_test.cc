// Kolibri DSpark (host): the DSpark drafter's descriptor - model::DFlashDesc's DSpark form read from the published
// speculators config (tests/model/dspark/config.json, intuit/Kolibri-1-dspark @ a59d5884), held to its kernels
// (model::check_dspark) and to Kolibri-1 (model::check_dflash_target against model::kolibri1()'s shape). No device.
//
//   1. The published config -> the form: the DFlash v1 backbone (5 layers, hidden 2560, 20 / 4 x 128, FF 6144, every layer
//      sliding at 2048 and causal, RoPE 1e4, eps 1e-6), block 8 (K <= 7), mask 127998, taps 3 / 13 / 23 / 33 / 43 (the aux
//      ids 4 / 14 / 24 / 34 / 44 minus one), the Markov rank 256 as sel_rank, the confidence head 2560 + 256.
//   2. Held to Kolibri-1 (hidden 2560, 128000 rows, 50 layers): another hidden, another vocabulary, a tap past the target.
//   3. Refused by name: sample_from_anchor false / absent, markov_head_type gated, markov_rank 0, a quantization_config,
//      sliding_window_non_causal true, an aux id 0, no transformer_layer_config, a draft vocabulary; by check_dspark: 16 q
//      heads, window 1024, block 16.
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "model/dflash.h"
#include "model/kolibri1.h"

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

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
std::string with(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  CHECK(at != std::string::npos);
  return s.replace(at, from.size(), to);
}
model::DFlashDesc parse(const std::string& text) { return model::dflash_desc_from_config(common::json::parse(text)); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: dspark_desc_test <dspark config.json>\n");
    return 2;
  }
  const std::string real = slurp(argv[1]);
  const model::Kolibri1Desc& kol = model::kolibri1();
  const model::DFlashTargetShape tshape{kol.name, kol.layers, kol.hidden, kol.vocab};

  // --- 1. the published config ------------------------------------------------------------------------------------------
  const model::DFlashDesc d = parse(real);
  {
    CHECK(d.dspark() && !d.v2());
    CHECK(d.architecture == "DSparkDraftModel" && model::is_dflash_architecture(d.architecture));
    CHECK(d.layers == 5 && d.hidden == 2560 && d.q_heads == 20 && d.kv_heads == 4 && d.head_dim == 128);
    CHECK(d.inter == 6144 && d.vocab == 128000 && d.block == 8 && d.max_k() == 7 && d.mask_id == 127998);
    CHECK((d.taps == std::vector<uint32_t>{3, 13, 23, 33, 43}));
    CHECK(d.window == std::vector<uint32_t>(5, 2048) && d.causal == std::vector<bool>(5, true));
    CHECK(d.context_reach(0) == 2047);
    CHECK(d.sel_rank == 256 && d.sel_top_k == 16 && d.conf_in == 2560 + 256);
    CHECK(d.conv_taps == 0 && d.conv_group == 0);
    CHECK(d.eps == 1e-6f && d.theta == 10000.0 && d.neox && d.out_mult == 1.0f && d.softcap == 0.0f);
    CHECK(d.fc_k(kol.hidden) == 12800 && d.qkv_n() == 2560 + 1024 && d.gate_up_n() == 12288);
    model::check_dspark(d);
    model::check_dflash_target(d, tshape);
    std::printf("1. the published DSpark config: 5 causal sliding layers of 2048, block 8 anchor-first, taps 3/13/23/33/43, "
                "Markov rank 256, confidence head %u wide; check_dspark and Kolibri-1 pass\n", d.conf_in);
  }

  // --- 2. held to Kolibri-1 ---------------------------------------------------------------------------------------------
  {
    CHECK(throws_with([&] { model::check_dflash_target(d, {kol.name, 50, 4096, 128000}); }, "hidden_size"));
    CHECK(throws_with([&] { model::check_dflash_target(d, {kol.name, 50, 2560, 131072}); }, "vocab_size"));
    CHECK(throws_with([&] { model::check_dflash_target(d, {kol.name, 40, 2560, 128000}); }, "target_layer_ids"));
    std::printf("2. held to Kolibri-1: hidden, vocabulary and tap depth refused by name\n");
  }

  // --- 3. refusals ------------------------------------------------------------------------------------------------------
  {
    CHECK(throws_with([&] { parse(with(real, "\"sample_from_anchor\": true", "\"sample_from_anchor\": false")); },
                      "sample_from_anchor"));
    CHECK(throws_with([&] { parse(with(real, "\"sample_from_anchor\": true,", "")); }, "sample_from_anchor"));
    CHECK(throws_with([&] { parse(with(real, "\"vanilla\"", "\"gated\"")); }, "markov_head_type"));
    CHECK(throws_with([&] { parse(with(real, "\"markov_rank\": 256", "\"markov_rank\": 0")); }, "markov_rank"));
    CHECK(throws_with([&] { parse(with(real, "\"dtype\"", "\"quantization_config\": {\"quant_method\": \"gptq\"}, \"dtype\"")); },
                      "quantization_config"));
    CHECK(throws_with([&] { parse(with(real, "\"sliding_window_non_causal\": false", "\"sliding_window_non_causal\": true")); },
                      "sliding_window_non_causal"));
    CHECK(throws_with([&] { parse(with(real, "    4,\n    14", "    0,\n    14")); }, "aux_hidden_state_layer_ids"));
    CHECK(throws_with([&] { parse(with(real, "\"transformer_layer_config\"", "\"layer_config\"")); },
                      "transformer_layer_config"));
    CHECK(throws_with([&] { parse(with(real, "\"draft_vocab_size\": 128000", "\"draft_vocab_size\": 32000")); },
                      "draft_vocab_size"));
    CHECK(throws_with([&] { model::check_dspark(parse(with(real, "\"num_attention_heads\": 20", "\"num_attention_heads\": 16"))); },
                      "heads"));
    CHECK(throws_with([&] { model::check_dspark(parse(with(real, "\"sliding_window\": 2048", "\"sliding_window\": 1024"))); },
                      "sliding_window"));
    CHECK(throws_with([&] { model::check_dspark(parse(with(real, "\"block_size\": 8", "\"block_size\": 16"))); }, "block_size"));
    std::printf("3. refusals by name\n");
  }
  std::printf("dspark_desc_test: OK\n");
  return 0;
}
